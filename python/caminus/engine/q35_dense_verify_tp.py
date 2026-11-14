"""The dense qwen35 line's T-token verify passes on TWO cards (speculative decoding with tensor parallelism), on the
two-card verify modules (kernels/k3_q35_{swift,tcap}_tp_ver.hip: the two-card module k3_q35_*_tp's ops with the verify
T-token forms and OP_XCHG; a T-token partial of T * D floats exchanges as-is: OP_XCHG's Seg.N is the payload's floats,
so the engine's exchange slots at spec_T * D cover it). engine/q35_dense_verify.py's one-card class, per card, with the
engine's tape structure (engine/q35_dense_tp.py: the layer's attention / Gated DeltaNet output and feed-forward output as
partials, two OP_XCHG per block, card 0's partial first on both cards, so the replicated residual stays the same bits).

The engine is a Dense35TP with spec_T >= T (its weights, caches, states and bounds; each Gated DeltaNet block keeps
spec_T tokens' saved inputs, the guard takes commits up to spec_T). The logits are each card's vocabulary shard for the
T rows; the greedy next token after each row is the first maximum over [card 0's rows | card 1's rows] (card 0's on a tie).
Each token's sums are the decode's statements, so a pass of T tokens gives T decode steps' bits: the gate is the passes'
logits bitwise the two-card decode steps' at their positions (tests/engine/test_q35_verify_tp.py)."""
import sys

import numpy as np
import torch

from . import k3
from .q35_dense import attn_part_elems
from .q35_dense_verify import MAX_T
from ..device import capacity


class Dense35VerifyTP:
    @staticmethod
    def _spec(eng, T, nfeat=0):
        """every buffer of one card's verify tape: name -> (elements, dtype); per card [card 0, card 1]"""
        D, hd, nh, nkv, S = eng.D, eng.hd, eng.nh, eng.nkv, eng.S
        nhc, nkvc = nh // 2, nkv // 2
        QHc = nhc * hd
        f32, f16, i32 = torch.float32, torch.float16, torch.int32
        out = []
        for c in (0, 1):
            gl = [d[c] for d in eng.L if d[c]['kind'] == 'gdn']
            CHl = max([d['CH'] for d in gl], default=4); NVl = max([d['NV'] for d in gl], default=4)
            KDl = max([d['NK'] * S for d in gl], default=4)
            fc = eng.f0 if c == 0 else eng.ffn - eng.f0
            nlog = eng.vh if c == 0 else eng.vocab - eng.vh
            n = dict(x=D, h=D, blk=D, blkp=D, ffp=D, ffpp=D, g=fc, u=fc, al=NVl, be=NVl, gates=2 * NVl, qkv=CHl, conv=CHl,
                     qk=2 * KDl, z=NVl * S, o=NVl * S, qf=2 * QHc, kr=nkvc * hd, vr=nkvc * hd, qo=QHc, ko=nkvc * hd,
                     att=QHc, logits=nlog)
            sp = {k: (T * v, f32) for k, v in n.items()}
            sp.update({k: (T * v, f16) for k, v in dict(h16=D, a16=fc, fin16=NVl * S, gat16=QHc).items()})
            sp['stage'] = (32, i32)
            sp['part'] = (attn_part_elems(T, nhc, eng.max_ctx + eng.spec_T - 1), f32)   # (the split attention pair's records)
            if nfeat and c == eng.feature_card:
                sp.update(feat=(nfeat * T * D, f32), fh=(T * D, f32), fh16=(T * D, f16))
            out.append(sp)
        return out

    @staticmethod
    def need_mib(T, eng=None, max_ctx=None, nfeat=0):
        """MiB of device memory one verify tape adds per card (with eng: from its dimensions and shares; without: an upper
        bound for the 27B shapes at T <= 4, to pass as the engine's vram_extra_mib before it exists, max_ctx the engine's:
        the split attention records grow with it, 12 query heads per card at most)"""
        if eng is None:
            if max_ctx is None: raise ValueError('Dense35VerifyTP.need_mib: max_ctx (the records of the split attention pair) without eng')
            return (10.0 * T + 4.0 + 4 * attn_part_elems(T, 12, int(max_ctx) + MAX_T - 1) / 2 ** 20
                    + 2 * T * 5120 * 4 / 2 ** 20    # (+ blkp / ffpp: the partials the exchanges read)
                    + ((4 * nfeat + 6) * T * 5120 / 2 ** 20 if nfeat else 0.0))
        sz = {torch.float32: 4, torch.float16: 2, torch.int32: 4}
        return [sum(n * sz[dt] for n, dt in sp.values()) / 2 ** 20 + 2.0 for sp in Dense35VerifyTP._spec(eng, T, nfeat)]

    def __init__(self, eng, T, module=None, per_wgp=None, vram_check=True, feature_layers=()):
        """eng: a Dense35TP with spec_T >= T (its weights, caches, states, bounds and exchange slots). T: tokens per pass,
        1 .. 4. module: None (the engine's line: k3.Q35_TPV_OF[eng.module]) or that module. per_wgp: workgroups per WGP of
        each card's tape grid ([card 0, card 1]; None: the engine's per card)"""
        if isinstance(T, bool) or not isinstance(T, int) or not 1 <= T <= MAX_T: raise ValueError(f'Dense35VerifyTP: T {T!r}: 1 .. {MAX_T}')
        if not hasattr(eng, 'spec_T'): raise ValueError(f'Dense35VerifyTP: the engine {type(eng).__name__} has no spec_T (a Dense35TP)')
        if module is None: module = k3.Q35_TPV_OF[eng.module]
        if module != k3.Q35_TPV_OF.get(eng.module):
            raise ValueError(f'Dense35VerifyTP: module {module!r} is not the verify module of the engine\'s module '
                             f'{eng.module} ({k3.Q35_TPV_OF})')
        self.q8 = bool(getattr(eng, 'q8', False))
        if (module in k3.Q35_KV_Q8_MODULES) != self.q8:
            raise ValueError('Dense35VerifyTP: target and verify KV formats must match')
        if eng.spec_T < T: raise ValueError(f'Dense35VerifyTP: the engine keeps {eng.spec_T} tokens\' saved inputs (spec_T), the tape runs {T}')
        if eng.nl_build != eng.nl: raise ValueError('Dense35VerifyTP: an engine built with debug_layers')
        fl = [int(l) for l in feature_layers]
        if any(not 1 <= l < eng.nl for l in fl) or len(set(fl)) != len(fl):
            raise ValueError(f'Dense35VerifyTP: feature_layers {fl} (distinct, 1 .. {eng.nl - 1})')
        self.eng, self.T, self.module, self.feature_layers = eng, T, module, fl
        self.features = None
        self.mib = self.need_mib(T, eng, nfeat=len(fl))
        if vram_check: capacity.check(eng.devs, self.mib, f'Dense35VerifyTP T {T}: its tape buffers ({self.mib[0]:.1f} / {self.mib[1]:.1f} MiB)')
        pw = per_wgp if per_wgp is not None else eng.per_wgp_c
        self.per_wgp_c = [int(pw[0]), int(pw[1])]
        self.bails = []
        self.E, self.buf, self.dstage, self.logits2, self.seg_names = [], [], [], [], [{}, {}]
        # the per-pass inputs, one shared pinned block (one copy to each card's stage): [0 .. 15] the pass parameters
        # ([1] = the previous pass's tokens to commit), [16 .. 16 + T) the tokens, [16 + T] the first position,
        # [17 + T] the exchanges' epoch base (pass * n_ex: every exchange flag only grows)
        self.hstage = torch.zeros(32, dtype=torch.int32, pin_memory=True)
        for c in (0, 1):
            with torch.cuda.device(eng.devs[c]):
                self._build(c)
        self.d_max = [(torch.zeros(T, device=f'cuda:{eng.devs[c]}'), torch.zeros(T, dtype=torch.int64, device=f'cuda:{eng.devs[c]}')) for c in (0, 1)]
        self.h_max = [(torch.zeros(T, pin_memory=True), torch.zeros(T, dtype=torch.int64, pin_memory=True)) for c in (0, 1)]
        self.h_st = [torch.zeros(k3.STATUS_WORDS, dtype=torch.int32, pin_memory=True) for c in (0, 1)]
        self.ev_done = []
        for c in (0, 1):
            with torch.cuda.device(eng.devs[c]): self.ev_done.append(torch.cuda.Event())
        self.h_greedy = torch.zeros(T, dtype=torch.int64, pin_memory=True)

    def _build(self, c):
        eng, T = self.eng, self.T
        D, hd, S = eng.D, eng.hd, eng.S
        nhc, nkvc = eng.nh // 2, eng.nkv // 2
        with torch.cuda.device(eng.devs[c]):
            s = {k: torch.zeros(n, dtype=dt, device='cuda') for k, (n, dt) in self._spec(eng, T, len(self.feature_layers))[c].items()}
            if 'feat' in s: self.features = s['feat'].view(len(self.feature_layers), T, D)
            self.buf.append(s)
            ds = s['stage']
            self.dstage.append(ds)
            toks, pos, epoch = ds[16:16 + T], ds[16 + T:17 + T], ds[17 + T:18 + T]
            E = k3.Engine(module=self.module, grid=k3.resident_grid(self.module, self.per_wgp_c[c]), T=T, limit_us=eng.limit_us)
            E.params = ds[0:16]
            E.guard_T = eng.spec_T   # (the device bound on the commit, as the decode tape's: the host bounds it by the last pass)
            self.E.append(E)
            nlog = eng.vh if c == 0 else eng.vocab - eng.vh
            self.logits2.append(s['logits'].view(T, nlog))
            self._tape(c, E, s, toks, pos, epoch)

    def _tape(self, c, E, s, toks, pos, epoch):
        eng, T = self.eng, self.T
        q8 = self.q8
        D, hd, S = eng.D, eng.hd, eng.S
        nhc, nkvc = eng.nh // 2, eng.nkv // 2
        names = self.seg_names[c]
        fslot = {l: k for k, l in enumerate(self.feature_layers)} if c == eng.feature_card else {}

        def named(si, name):
            names[si] = name; return si

        def mv(W, x16, out, dep, x32=None):
            """W's matvec on the T tokens' fp16 activations x16 [T, K] -> out [T, rows] (an F32 matrix: on x32)"""
            if W['fmt'] == 'F32': return named(E.f32_mv(W['w'], x32, out, rows_per_task=4, dep=dep), W['name'])
            if W['fmt'] in (k3.OWN_Q35T_1RT if eng.tcap else k3.OWN_1RT_FMT):
                kw = dict(tokens=T, x_stride=W['K'], out_stride=W['rows']) if T > 1 else {}
                return named(E.own_matvec_1rt(W['w'], W['fmt'], W['rows'], W['K'], x16, out, rows_per_task=eng.rpt, dep=dep, **kw),
                             W['name'])
            return named(E.own_matvec16(W['w'], W['fmt'], W['rows'], W['K'], x16, out, rows_per_task=eng.rpt16, dep=dep, tokens=T),
                         W['name'])

        def xchg(partial, out, x, dep):
            return named(E.xchg(partial, out, eng.slots[1 - c].slot(x), eng.slots[c].slot(x), epoch, x, c,
                                local_only=eng.local_only, dep=dep), f'xchg {x}')

        prev = E.own_emb(eng.emb_dev[c], eng.emb.type, eng.vocab, D, toks, s['x'], tokens=T)
        for il, dd in enumerate(eng.L):
            d = dd[c]
            n1 = E.own_add_rms16(s['x'], d['attn_norm'], s['h'], s['h16'], eng.eps, d=s['ffp'] if il else None,
                                 slots=1 if il else 0, dep=prev, tokens=T)
            fc = None
            if il in fslot:
                k = fslot[il]
                fc = E.own_add_rms16(s['feat'][k * T * D:(k + 1) * T * D], d['attn_norm'], s['fh'], s['fh16'], eng.eps,
                                     d=s['x'], slots=1, dep=n1, tokens=T)
            if d['kind'] == 'gdn':
                NV, NK, CH = d['NV'], d['NK'], d['CH']; KD = NK * S
                a_ = mv(d['al'], s['h16'], s['al'][:T * NV], n1, x32=s['h']); b_ = mv(d['be'], s['h16'], s['be'][:T * NV], n1, x32=s['h'])
                gt = E.also(E.gdn_gates(s['al'][:T * NV], s['be'][:T * NV], d['dt'], d['a'], s['gates'][:2 * T * NV], dep=a_), b_)
                q_ = mv(d['qkv'], s['h16'], s['qkv'][:T * CH], n1)
                c_ = E.gdn_conv(s['qkv'][:T * CH], d['conv_w'], d['conv_st'], s['conv'], d['convx'], dep=q_)
                l_ = E.gdn_l2(s['conv'], s['qk'], 2 * NK, S, eng.eps, x_stride=CH, dep=c_)
                z_ = mv(d['z'], s['h16'], s['z'], n1)
                st = E.also(E.gdn_step(s['qk'], s['conv'][2 * KD:], s['gates'][:2 * T * NV], d['gdn'], s['o'], NV, S, NK, CH, dep=l_), gt)
                sv = E.gdn_save(s['qk'], s['conv'][2 * KD:], s['gates'][:2 * T * NV], d['gdn'], NV, S, NK, CH, dep=st)
                nm = E.also(E.gdn_norm16(s['o'][:T * NV * S], d['nw'], s['z'][:T * NV * S], s['fin16'][:T * NV * S], NV, S, eng.eps, dep=st), z_)
                o_ = E.also(mv(d['so'], s['fin16'], s['blkp'], nm), sv)
            else:
                k_ = mv(d['k'], s['h16'], s['kr'], n1); v_ = mv(d['v'], s['h16'], s['vr'], n1)
                kd = E.qk_nr(s['kr'], hd, nkvc, hd, d['kn'], pos, s['ko'], eng.nrot, eng.theta_scale, eng.eps, dep=k_)
                if q8:
                    kd = E.hadamard(s['ko'], 256, dep=kd)
                    v_ = E.hadamard(s['vr'], 64, dep=v_)
                ap = E.also(E.kv_append(s['ko'], s['vr'], d['cache'], pos, q8=q8, dep=kd), v_)
                q_ = mv(d['q'], s['h16'], s['qf'], n1)
                qd = E.qk_nr(s['qf'], 2 * hd, nhc, hd, d['qn'], pos, s['qo'], eng.nrot, eng.theta_scale, eng.eps, dep=q_)
                if q8: qd = E.hadamard(s['qo'], 256, dep=qd)
                akw = dict(max_cells=eng.max_ctx + eng.spec_T - 1, part=s['part']) if E.split_attn() else dict(batched=True)
                a_ = E.also(E.attn_dec(s['qo'], d['cache'], pos, s['att'], nhc, nkvc, hd, 1.0 / np.sqrt(hd), q8=q8, dep=ap, **akw), qd)
                if q8: a_ = E.hadamard(s['att'], 64, dep=a_)
                g_ = E.gate_mul16(s['att'], s['qf'], 2 * hd, hd, s['gat16'], hd, dep=a_)
                o_ = mv(d['o'], s['gat16'], s['blkp'], g_)
            x1 = xchg(s['blkp'], s['blk'], 2 * il, o_)
            n2 = E.own_add_rms16(s['x'], d['post_norm'], s['h'], s['h16'], eng.eps, d=s['blk'], slots=1, dep=x1, tokens=T)
            if fc is not None: E.also(n2, fc)
            g_ = mv(d['fg'], s['h16'], s['g'], n2); u_ = mv(d['fu'], s['h16'], s['u'], n2)
            sw = E.also(E.own_swiglu16(s['g'], s['u'], s['a16'], dep=u_), g_)
            f_ = mv(d['fd'], s['a16'], s['ffpp'], sw)
            prev = xchg(s['ffpp'], s['ffp'], 2 * il + 1, f_)
        nf = E.own_add_rms16(s['x'], eng.out_norm[c], s['h'], s['h16'], eng.eps, d=s['ffp'], slots=1, dep=prev, tokens=T)
        mv(eng.head[c], s['h16'], s['logits'], nf)
        E.build()

    def logits_full(self):
        """the full logits [T, vocab] on card 0 (a fresh tensor: card 1's rows copied over)"""
        eng = self.eng
        return torch.cat([self.logits2[0], self.logits2[1].to(f'cuda:{eng.devs[0]}')], dim=1)

    def logits(self, t):
        """row t's logits (fp32 [vocab], on card 0; valid until the next pass)"""
        return self.logits_full()[t]

    def run(self, tokens, pos0, commit, argmax=True):
        """T tokens at positions pos0 .. pos0 + T - 1 in one launch on both cards (waited for), after committing the previous
        pass's first `commit` tokens (0 .. the engine's spec_T; 0 after a reset or a prefill) into the recurrent states.
        Returns the greedy next token after each row (each card's max of its rows; the first maximum over
        [card 0's rows | card 1's rows], card 0's on a tie; argmax False: None). commit is at most the tokens of the last
        pass on the engine's states (eng.commit_max(): a decode step 1, a verify pass its T; 0 after a reset or a prefill),
        since the saved records past them are an older pass's. Refused before anything is written: a poisoned or broken
        engine, a span outside the KV cache, a commit out of range, a token outside the vocabulary. A wait that gave up
        runs the pass again (commit 0: the first launch committed already); the device bound guard poisons the engine until
        its reset()."""
        eng, T = self.eng, self.T
        if eng.broken: raise getattr(eng, '_broken_error', k3.NonFiniteLogits)(f'Dense35VerifyTP pass: refused: {eng.broken}')
        if eng.poisoned: raise k3.ContextBoundsError(f'Dense35VerifyTP pass: refused until reset(): {eng.poisoned}')
        if len(tokens) != T: raise ValueError(f'Dense35VerifyTP pass: {len(tokens)} tokens for a T = {T} tape')
        k3.check_span(pos0, T, eng.max_ctx, commit=commit, ntok=eng.commit_max(), what='Dense35VerifyTP pass')
        toks = [int(t) for t in tokens]
        if any(not 0 <= t < eng.vocab for t in toks): raise ValueError(f'Dense35VerifyTP pass: a token of {toks} outside the vocabulary ({eng.vocab})')
        for t in range(T): self.hstage[16 + t] = toks[t]
        self.hstage[16 + T] = int(pos0)
        eng.fresh = False; eng.last_T = T   # (the next pass on these states commits at most T)
        for attempt in range(eng.max_retries + 1):
            bad = self._pass(pos0, commit if attempt == 0 else 0, argmax)
            if not bad: break
            self.bails.append((int(pos0), attempt, bad)); eng.bails.append((int(pos0), attempt, bad))
            print(f'k3 qwen35 TP verify: a wait gave up at position {pos0} (attempt {attempt + 1}), running the pass again: '
                  + '; '.join(bad), file=sys.stderr, flush=True)
        else:
            raise RuntimeError(f'k3 qwen35 TP verify: a wait gave up at position {pos0} in {eng.max_retries + 1} attempts: ' + '; '.join(bad))
        vmax = [self.h_max[c][0] for c in (0, 1)]
        if argmax and not all(bool(torch.isfinite(v).all()) for v in vmax):
            eng.mark_broken(f'non-finite logits in a two-card verify pass at position {pos0} (row maxima {[v.tolist() for v in vmax]})')
        if not argmax and not bool(torch.isfinite(torch.cat([self.logits2[0], self.logits2[1].to(self.logits2[0].device)], dim=1).amax(dim=1)).all()):
            eng.mark_broken(f'non-finite logits in a two-card verify pass at position {pos0}')
        return self.h_greedy.tolist() if argmax else None

    def _pass(self, pos0, commit, argmax):
        eng, T = self.eng, self.T
        self.hstage[1] = int(commit)
        base = eng.token * eng.n_ex; eng.token += 1          # a fresh epoch per launch (a retry too): exchange flags only grow
        if base + eng.n_ex >= 2 ** 31: raise RuntimeError('Dense35VerifyTP: the exchange epoch would pass int32')
        self.hstage[17 + T] = base
        for c in (0, 1):
            with torch.cuda.device(eng.devs[c]):
                self.dstage[c].copy_(self.hstage, non_blocking=True)   # (the pinned block is rewritten only after the waits)
                if self.features is not None and c == eng.feature_card: self.features.zero_()
                self.E[c].run(check=False)
                if argmax:
                    torch.max(self.logits2[c], dim=1, out=self.d_max[c])
                    self.h_max[c][0].copy_(self.d_max[c][0], non_blocking=True)
                    self.h_max[c][1].copy_(self.d_max[c][1], non_blocking=True)
                self.h_st[c].copy_(self.E[c].sync[2 * k3.MAXSEG:2 * k3.MAXSEG + k3.STATUS_WORDS], non_blocking=True)
                self.ev_done[c].record()
        for c in (0, 1): self.ev_done[c].synchronize()
        if self.q8:
            for c in (0, 1):
                try:
                    self.E[c].check_kv_error()
                except k3.InvalidKVEncoding as exc:
                    eng.mark_broken(f'card {c}: {exc}', error_cls=k3.InvalidKVEncoding)
        oob = []
        for c in (0, 1):
            b = k3.bounds_status(self.h_st[c].tolist())
            if b: oob.append(f'card {c} (device {eng.devs[c]}): {b}')
        if oob:
            for c in (0, 1):
                with torch.cuda.device(eng.devs[c]): torch.cuda.synchronize(); self.E[c].sync.zero_()
            eng.poisoned = f'the device bound guard caught an out-of-range value at verify position {pos0}: ' + '; '.join(oob)
            raise k3.ContextBoundsError(f'k3 qwen35 TP verify: {eng.poisoned}')
        bad = []
        for c in (0, 1):
            st = self.h_st[c].tolist()
            if st[1]:
                with torch.cuda.device(eng.devs[c]):
                    torch.cuda.synchronize()
                    names = {v: k for k, v in vars(k3).items() if k.startswith('OP_')}
                    bad.append(f'card {c} (device {eng.devs[c]}): segment {st[2]} = {names.get(self.E[c].segs[st[2]].op)} '
                               f'{self.seg_names[c].get(st[2], "")} (seen {st[3]}); {self.E[c].stall_report()}')
                    self.E[c].sync.zero_()
        if not bad and argmax:
            (v0, x0), (v1, x1) = [(v.numpy(), x.numpy()) for v, x in self.h_max]
            # the first maximum over [card 0's rows | card 1's rows]: card 0's on a tie; a NaN counts as the maximum (argmax)
            g = self.h_greedy.numpy()
            for t in range(T): g[t] = int(x0[t]) if (v0[t] >= v1[t] or v0[t] != v0[t]) else eng.vh + int(x1[t])
        return bad
