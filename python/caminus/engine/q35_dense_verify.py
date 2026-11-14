"""The dense qwen35 line (Swift 1.5): T-token verify passes for speculative decoding, on the verify module
(kernels/k3_q35_swift_ver.hip: the Swift decode module's ops with their T-token forms).

A pass runs T tokens at positions pos0 .. pos0 + T - 1 in one launch, on the decode engine's weights, KV caches and
Gated DeltaNet states. Token t's logits, its KV rows and the recurrent states it leaves are bit for bit what T decode
steps of Dense35Own give for the same tokens: every sum is the decode tile's statements on the same operands, in the
same order (the one-round-trip tiles' T-token forms, kernels/own_quant_mt.h; the rows tiles per token; the glue ops one
task per token and head).

The recurrent state is committed lazily, as in decode: a pass stores the state from before its own tokens and keeps
their inputs; the next pass (verify or decode) first replays the `commit` tokens of the previous pass that were
accepted. So the caller passes, with every pass, how many tokens of the previous pass to keep:

    eng = Dense35Own(path, max_ctx=C, spec_T=4, vram_extra_mib=Dense35Verify.need_mib(4, max_ctx=C))
    ver = Dense35Verify(eng, 4)
    top = ver.run(tokens4, pos0, commit)     # the greedy token after each of the 4 (argmax on the card)
    # accept a of them (1 .. 4): the next pass, at pos0 + a, commits a
    eng.pos_host = pos0 + a; eng.step_token(toks, pos0 + a, ...) or ver.run(..., pos0 + a, a)

A wrong token at row r leaves rows 0 .. r - 1 right (causal); its KV rows from r on are overwritten by the next pass at
those positions, and the states never take a token that was not committed. The decode engine's spec_T must be at least
T (its states keep spec_T tokens' inputs); the engine's position is not moved here: the caller accepts a prefix.
The feed-forward runs as three segments (gate, up, SwiGLU: the fused gate | up tile has no T-token form); the gates
check that they give the fused segment's bits."""
import sys

import numpy as np
import torch

from . import k3
from .q35_dense import attn_part_elems
from ..device import capacity

MAX_T = 4                        # own_quant_mt.h's OWN_MT
MODULE = 'k3_q35_swift_ver'
MODULE_TCAP = 'k3_q35_tcap_ver'   # (ThinkingCap's: its formats' T-token tiles, kernels/own_quant_tcap_mt.h)


class Dense35Verify:
    @staticmethod
    def _spec(eng, T, nfeat=0):
        """every buffer of one verify tape: name -> (elements, dtype); nfeat: the exported feature layers (feat [nfeat, T, D]
        and the copy segments' scratch norm outputs)"""
        D, hd, nh, nkv, S, NK, NV, ffn = eng.D, eng.hd, eng.nh, eng.nkv, eng.S, eng.NK, eng.NV, eng.ffn
        CH, KD, VD, QH = 2 * S * NK + S * NV, NK * S, NV * S, nh * hd
        f32, f16, i32 = torch.float32, torch.float16, torch.int32
        n = dict(x=D, h=D, blk=D, ffp=D, g=ffn, u=ffn, al=NV, be=NV, gates=2 * NV, qkv=CH, conv=CH, qk=2 * KD, z=VD, o=VD,
                 qf=2 * QH, kr=nkv * hd, vr=nkv * hd, qo=QH, ko=nkv * hd, att=QH, logits=eng.vocab)
        sp = {k: (T * v, f32) for k, v in n.items()}
        sp.update({k: (T * v, f16) for k, v in dict(h16=D, a16=ffn, fin16=VD, gat16=QH).items()})
        sp['stage'] = (32, i32)
        sp['part'] = (attn_part_elems(T, nh, eng.max_ctx + eng.spec_T - 1), f32)   # (the split attention pair's records)
        if nfeat:
            sp['feat'] = (nfeat * T * D, f32); sp['fh'] = (T * D, f32); sp['fh16'] = (T * D, f16)
        return sp

    @staticmethod
    def need_mib(T, eng=None, max_ctx=None, nfeat=0):
        """MiB of device memory one verify tape adds (with eng: from its dimensions; without: an upper bound for the 27B
        shapes at T <= 4, to pass as the decode engine's vram_extra_mib before it exists, max_ctx the engine's: the split
        attention records grow with it, 24 query heads at most)"""
        if eng is None:
            if max_ctx is None: raise ValueError('Dense35Verify.need_mib: max_ctx (the records of the split attention pair) without eng')
            return (10.0 * T + 4.0 + 4 * attn_part_elems(T, 24, int(max_ctx) + MAX_T - 1) / 2 ** 20
                    + (nfeat + 2) * T * 5120 * 4 / 2 ** 20 * (nfeat > 0))
        sz = {torch.float32: 4, torch.float16: 2, torch.int32: 4}
        return sum(n * sz[dt] for n, dt in Dense35Verify._spec(eng, T, nfeat).values()) / 2 ** 20 + 2.0   # (+ the tape and its sync words)

    def __init__(self, eng, T, module=None, per_wgp=None, vram_check=True, feature_layers=()):
        """eng: a Dense35Own (Swift's or ThinkingCap's module) with spec_T >= T (its weights, caches, states and bounds); T:
        tokens per pass, 1 .. 4; module: None (the engine's line: k3_q35_swift_ver, or k3_q35_tcap_ver on ThinkingCap's
        module) or one of those modules' _trace builds; per_wgp: workgroups per WGP of this tape's grid
        (None: the decode engine's; the bits are the same at every grid); feature_layers: layers whose entering residual
        each pass copies to self.features [len, T, D] fp32 (a DFlash drafter's target layers: the residual after the
        previous layer's feed-forward is added, before the layer's norm; llama.cpp's layer input). The copy is the
        module's own add-rms op on a zeroed buffer (0 + x, exact), so no device code changes and the logits stay bitwise"""
        if isinstance(T, bool) or not isinstance(T, int) or not 1 <= T <= MAX_T: raise ValueError(f'Dense35Verify: T {T!r}: 1 .. {MAX_T}')
        q8 = bool(getattr(eng, 'q8', False))
        if module is None:
            module = MODULE_TCAP if eng.tcap else MODULE
            if q8:
                if eng.module in k3.Q35R_MODULES:
                    raise ValueError('Dense35Verify: GSQ Q8 verify requires the GSQ verify implementation')
                module = k3.Q35_KV_Q8_OF[module]
        if (module in k3.Q35_KV_Q8_MODULES) != q8:
            raise ValueError('Dense35Verify: target and verify KV formats must match')
        line = k3.Q35TV_MODULES if eng.tcap else tuple(m for m in k3.Q35SV_MODULES if m not in k3.Q35M_MODULES + k3.Q35DF_MODULES)
        if module not in line:
            raise ValueError(f'Dense35Verify: module {module!r} is not a verify module of the engine\'s line {line} (the engine runs {eng.module})')
        if eng.spec_T < T: raise ValueError(f'Dense35Verify: the engine keeps {eng.spec_T} tokens\' saved inputs (spec_T), the tape runs {T}')
        if eng.nl_build != eng.nl: raise ValueError('Dense35Verify: an engine built with debug_layers')
        fl = [int(l) for l in feature_layers]
        if any(not 1 <= l < eng.nl for l in fl) or len(set(fl)) != len(fl): raise ValueError(f'Dense35Verify: feature_layers {fl} (distinct, 1 .. {eng.nl - 1})')
        self.eng, self.T, self.module, self.feature_layers = eng, T, module, fl
        self.mib = self.need_mib(T, eng, nfeat=len(fl))
        if vram_check: capacity.check([eng.dev], [self.mib], f'Dense35Verify T {T}: its tape buffers ({self.mib:.1f} MiB)')
        self.per_wgp = int(per_wgp) if per_wgp else eng.per_wgp
        self.bails = []
        with torch.cuda.device(eng.dev): self._build()

    def _build(self):
        eng, T = self.eng, self.T
        q8 = bool(getattr(eng, 'q8', False))
        D, hd, nh, nkv, S, NK, NV, ffn = eng.D, eng.hd, eng.nh, eng.nkv, eng.S, eng.NK, eng.NV, eng.ffn
        CH, KD = 2 * S * NK + S * NV, NK * S
        s = self.buf = {k: torch.zeros(n, dtype=dt, device='cuda') for k, (n, dt) in self._spec(eng, T, len(self.feature_layers)).items()}
        fslot = {l: k for k, l in enumerate(self.feature_layers)}
        self.features = s['feat'].view(len(self.feature_layers), T, D) if self.feature_layers else None
        # the per-pass inputs, one block of int32: [0 .. 15] the pass parameters ([1] = the previous pass's tokens to commit),
        # [16 .. 16 + T) the tokens, [16 + T] the first position
        self.hstage = torch.zeros(32, dtype=torch.int32, pin_memory=True)
        self.dstage = s['stage']; self.toks, self.pos = self.dstage[16:16 + T], self.dstage[16 + T:17 + T]
        E = self.E = k3.Engine(module=self.module, grid=k3.resident_grid(self.module, self.per_wgp), T=T, limit_us=eng.limit_us)
        E.params = self.dstage[0:16]
        E.guard_T = eng.spec_T   # (the device bound on the commit, as the decode tape's: the host bounds it by the last pass)
        self.seg_names = {}
        # the attention segments' arguments: the split pair's records over the cache's rows (the verify modules are split
        # modules, k3.Q35_SPLIT_MODULES; T rows, one buffer for every block), else the one-head op's batched loads
        akw = (dict(max_cells=eng.max_ctx + eng.spec_T - 1, part=s['part']) if E.split_attn() else dict(batched=True))

        def named(si, name):
            self.seg_names[si] = name; return si

        def mv(W, x16, out, dep, x32=None):
            """W's matvec on the T tokens' fp16 activations x16 [T, K] -> out [T, rows] (an F32 matrix: on x32)"""
            if W['fmt'] == 'F32': return named(E.f32_mv(W['w'], x32, out, rows_per_task=4, dep=dep), W['name'])
            if W['fmt'] in eng.one_rt:
                kw = dict(tokens=T, x_stride=W['K'], out_stride=W['rows']) if T > 1 else {}
                return named(E.own_matvec_1rt(W['w'], W['fmt'], W['rows'], W['K'], x16, out, rows_per_task=eng.rpt, dep=dep, **kw),
                             W['name'])
            return named(E.own_matvec16(W['w'], W['fmt'], W['rows'], W['K'], x16, out, rows_per_task=eng.rpt16, dep=dep, tokens=T),
                         W['name'])

        prev = E.own_emb(eng.emb_dev, eng.emb.type, eng.vocab, D, self.toks, s['x'], tokens=T)
        for il, d in enumerate(eng.L):
            n1 = E.own_add_rms16(s['x'], d['attn_norm'], s['h'], s['h16'], eng.eps, d=s['ffp'] if il else None,
                                 slots=1 if il else 0, dep=prev, tokens=T)
            fc = None
            if il in fslot:                      # (feat[k] += x on a buffer zeroed before the pass; its norm goes to scratch)
                k = fslot[il]
                fc = E.own_add_rms16(s['feat'][k * T * D:(k + 1) * T * D], d['attn_norm'], s['fh'], s['fh16'], eng.eps, d=s['x'],
                                     slots=1, dep=n1, tokens=T)
            if d['kind'] == 'gdn':
                a_ = mv(d['al'], s['h16'], s['al'], n1, x32=s['h']); b_ = mv(d['be'], s['h16'], s['be'], n1, x32=s['h'])
                gt = E.also(E.gdn_gates(s['al'], s['be'], d['dt'], d['a'], s['gates'], dep=a_), b_)
                q_ = mv(d['qkv'], s['h16'], s['qkv'], n1)
                c_ = E.gdn_conv(s['qkv'], d['conv_w'], d['conv_st'], s['conv'], d['convx'], dep=q_)
                l_ = E.gdn_l2(s['conv'], s['qk'], 2 * NK, S, eng.eps, x_stride=CH, dep=c_)
                z_ = mv(d['z'], s['h16'], s['z'], n1)
                st = E.also(E.gdn_step(s['qk'], s['conv'][2 * KD:], s['gates'], d['gdn'], s['o'], NV, S, NK, CH, dep=l_), gt)
                sv = E.gdn_save(s['qk'], s['conv'][2 * KD:], s['gates'], d['gdn'], NV, S, NK, CH, dep=st)
                nm = E.also(E.gdn_norm16(s['o'], d['nw'], s['z'], s['fin16'], NV, S, eng.eps, dep=st), z_)
                o_ = E.also(mv(d['so'], s['fin16'], s['blk'], nm), sv)
            else:
                k_ = mv(d['k'], s['h16'], s['kr'], n1); v_ = mv(d['v'], s['h16'], s['vr'], n1)
                kd = E.qk_nr(s['kr'], hd, nkv, hd, d['kn'], self.pos, s['ko'], eng.nrot, eng.theta_scale, eng.eps, dep=k_)
                if q8:
                    kd = E.hadamard(s['ko'], 256, dep=kd)
                    v_ = E.hadamard(s['vr'], 64, dep=v_)
                ap = E.also(E.kv_append(s['ko'], s['vr'], d['cache'], self.pos, q8=q8, dep=kd), v_)
                q_ = mv(d['q'], s['h16'], s['qf'], n1)
                qd = E.qk_nr(s['qf'], 2 * hd, nh, hd, d['qn'], self.pos, s['qo'], eng.nrot, eng.theta_scale, eng.eps, dep=q_)
                if q8: qd = E.hadamard(s['qo'], 256, dep=qd)
                a_ = E.also(E.attn_dec(s['qo'], d['cache'], self.pos, s['att'], nh, nkv, hd, 1.0 / np.sqrt(hd), dep=ap,
                                       q8=q8, **akw), qd)
                if q8: a_ = E.hadamard(s['att'], 64, dep=a_)
                g_ = E.gate_mul16(s['att'], s['qf'], 2 * hd, hd, s['gat16'], hd, dep=a_)
                o_ = mv(d['o'], s['gat16'], s['blk'], g_)
            n2 = E.own_add_rms16(s['x'], d['post_norm'], s['h'], s['h16'], eng.eps, d=s['blk'], slots=1, dep=o_, tokens=T)
            if fc is not None: E.also(n2, fc)    # (the copy runs beside the layer's matvecs; x's next writer, this add, waits for it)
            g_ = mv(d['fg'], s['h16'], s['g'], n2); u_ = mv(d['fu'], s['h16'], s['u'], n2)
            sw = E.also(E.own_swiglu16(s['g'], s['u'], s['a16'], dep=u_), g_)
            prev = mv(d['fd'], s['a16'], s['ffp'], sw)
        nf = E.own_add_rms16(s['x'], eng.out_norm, s['h'], s['h16'], eng.eps, d=s['ffp'], slots=1, dep=prev, tokens=T)
        mv(eng.head, s['h16'], s['logits'], nf)
        E.build()
        self.logits2 = s['logits'].view(T, eng.vocab)
        self.d_max = (torch.zeros(T, device='cuda'), torch.zeros(T, dtype=torch.int64, device='cuda'))
        self.h_max = torch.zeros(T, dtype=torch.int64, pin_memory=True)
        self.h_vmax = torch.zeros(T, pin_memory=True)   # (the rows' max values: a non-finite one breaks the engine)
        self.h_st = torch.zeros(k3.STATUS_WORDS, dtype=torch.int32, pin_memory=True)
        self.ev_done = torch.cuda.Event()

    def logits(self, t):
        """row t's logits (fp32 [vocab], on the device; valid until the next pass)"""
        return self.logits2[t]

    def run(self, tokens, pos0, commit, argmax=True):
        """T tokens at positions pos0 .. pos0 + T - 1 in one launch (waited for), after committing the previous pass's
        first `commit` tokens (0 .. the engine's spec_T; 0 after a reset or a prefill) into the recurrent states. Returns
        the greedy next token after each row (argmax on the card, the first index on ties; argmax False: None). commit
        is at most the tokens of the last pass on the engine's states (eng.commit_max(): a decode step 1, a verify pass
        its T; 0 after a reset or a prefill), since the saved records past them are an older pass's. Refused
        before anything is written: a poisoned engine, a span outside the KV cache, a commit out of range, a token
        outside the vocabulary. A wait that gave up runs the pass again (commit 0: the first launch committed already);
        the device bound guard poisons the engine until its reset()."""
        eng, T, E = self.eng, self.T, self.E
        if getattr(eng, 'broken', None): raise getattr(eng, '_broken_error', k3.NonFiniteLogits)(f'Dense35Verify pass: refused: {eng.broken}')
        if eng.poisoned: raise k3.ContextBoundsError(f'Dense35Verify pass: refused until reset(): {eng.poisoned}')
        if len(tokens) != T: raise ValueError(f'Dense35Verify pass: {len(tokens)} tokens for a T = {T} tape')
        k3.check_span(pos0, T, eng.max_ctx, commit=commit, ntok=eng.commit_max(), what='Dense35Verify pass')
        toks = [int(t) for t in tokens]
        if any(not 0 <= t < eng.vocab for t in toks): raise ValueError(f'Dense35Verify pass: a token of {toks} outside the vocabulary ({eng.vocab})')
        for t in range(T): self.hstage[16 + t] = toks[t]
        self.hstage[16 + T] = int(pos0)
        eng.fresh = False; eng.last_T = T   # (the next pass on these states commits at most T)
        with torch.cuda.device(eng.dev):
            for attempt in range(eng.max_retries + 1):
                bad = self._pass(pos0, commit if attempt == 0 else 0, argmax)
                if not bad: break
                self.bails.append((int(pos0), attempt, bad)); eng.bails.append((int(pos0), attempt, bad))
                print(f'k3 qwen35 verify: a wait gave up at position {pos0} (attempt {attempt + 1}), running the pass again: '
                      + '; '.join(bad), file=sys.stderr, flush=True)
            else:
                raise RuntimeError(f'k3 qwen35 verify: a wait gave up at position {pos0} in {eng.max_retries + 1} attempts: ' + '; '.join(bad))
        eng.token += 1
        if argmax and not bool(torch.isfinite(self.h_vmax).all()):
            eng.mark_broken(f'non-finite logits in a verify pass at position {pos0} (row maxima {self.h_vmax.tolist()})')
        if not argmax and not bool(torch.isfinite(self.logits2.amax(dim=1)).all()):   # (sampled passes: the rows' max on the card)
            eng.mark_broken(f'non-finite logits in a verify pass at position {pos0}')
        return self.h_max.tolist() if argmax else None

    def _pass(self, pos0, commit, argmax):
        eng, E = self.eng, self.E
        self.hstage[1] = int(commit)
        self.dstage.copy_(self.hstage, non_blocking=True)   # (the pinned block is rewritten only after the wait below)
        if self.features is not None: self.features.zero_()   # (the copy segments add into it)
        E.run(check=False)
        if argmax:
            torch.max(self.logits2, dim=1, out=self.d_max)
            self.h_max.copy_(self.d_max[1], non_blocking=True)
            self.h_vmax.copy_(self.d_max[0], non_blocking=True)
        self.h_st.copy_(E.sync[2 * k3.MAXSEG:2 * k3.MAXSEG + k3.STATUS_WORDS], non_blocking=True)
        self.ev_done.record()
        self.ev_done.synchronize()
        if getattr(eng, 'q8', False):
            try:
                E.check_kv_error()
            except k3.InvalidKVEncoding as exc:
                eng.mark_broken(str(exc), error_cls=k3.InvalidKVEncoding)
        st = self.h_st.tolist()
        b = k3.bounds_status(st)
        if b:
            torch.cuda.synchronize(); E.sync.zero_()
            eng.poisoned = f'the device bound guard caught an out-of-range value at verify position {pos0}: {b}'
            raise k3.ContextBoundsError(f'k3 qwen35 verify: {eng.poisoned}')
        if st[1]:
            torch.cuda.synchronize()
            names = {v: k for k, v in vars(k3).items() if k.startswith('OP_')}
            rep = f'segment {st[2]} = {names.get(E.segs[st[2]].op)} {self.seg_names.get(st[2], "")} (seen {st[3]}); {E.stall_report()}'
            E.sync.zero_()
            return [rep]
        return []
