"""Qwen3-30B-A3B (qwen3moe): T-token verify passes for speculative decoding, on the verify module
(kernels/k3_q3m_ver.hip: the decode module's ops in their T-token forms).

A pass runs T tokens at positions pos0 .. pos0 + T - 1 in one launch: token t's logits, its KV cache row and (optionally)
the residual entering the tap layers are bit for bit what T decode steps of Q3MoEOne give for the same tokens, because
every sum is the decode tile's statements on the same operands. A pass with a wrong token at row r leaves rows 0 .. r - 1
right (causal); the rows from r on are overwritten by the next pass at those positions.

    eng = Q3MoEOne(path, max_ctx=..., vram_extra_mib=Q3MoEVerify.need_mib(4))
    ver = Q3MoEVerify(eng, 4)
    ver.run(tokens4, pos0)        # then ver.logits(t), ver.feats(t)

What the tape shares between the tokens of a pass and what it cannot (measured: docs/dev/qwen3-30b-a3b.md):
  dense matvecs   one segment per matrix for all tokens: a row pair's weight words are loaded once and every token's sums
                  run on them (own_quant_mt.h);
  experts         8 slots per token in one grouped segment; slots that hold the same expert are computed by the first
                  of them (OWN_UNION), each on its own input;
  attention       a row-only segment per token writes the pass's cache rows first, then the T attentions run side by
                  side (each reads the context once: T times one token's attention).
The engine's position is not moved: the caller accepts a prefix and sets eng.pos_host."""
import numpy as np
import torch

from . import k3
from ..device import capacity

MAX_T = 4                        # own_quant_mt.h's OWN_MT, and 8 expert slots per token in one 32-lane compare
DFLASH_TAPS = (1, 12, 23, 34, 45)   # the residual entering these layers: the DFlash speculator's target features


def _join(E, segs):
    """a segment that is done when every one of segs is (a tree of no-ops, two dependencies each)"""
    segs = list(segs)
    while len(segs) > 1:
        nxt = [E.also(E.nop(1, 0, dep=segs[i]), segs[i + 1]) for i in range(0, len(segs) - 1, 2)]
        if len(segs) % 2: nxt.append(segs[-1])
        segs = nxt
    return segs[0]


class Q3MoEVerify:
    @staticmethod
    def _spec(eng, T, taps):
        """every buffer of one verify tape: name -> (elements, dtype)"""
        D, hd, nh, nkv, KU, FF = eng.D, eng.hd, eng.nh, eng.nkv, eng.k_used, eng.ffn
        f32, f16, i32 = torch.float32, torch.float16, torch.int32
        return dict(x=(T * D, f32), h=(T * D, f32), h16=(T * D, f16), qkv=(T * (nh * hd + 2 * nkv * hd), f32),
                    att16=(T * nh * hd, f16), o=(T * D, f32), logit=(T * eng.n_exp, f32), ids=(len(eng.L) * T * KU, i32),
                    wts=(T * KU, f32), a16=(T * KU * FF, f16), d=(T * KU * D, f32), logits=(T * eng.vocab, f32),
                    feat=(max(1, len(taps)) * T * D, f32), fh=(T * D, f32), fh16=(T * D, f16), neg=(T, f32), stage=(T + 1, i32),
                    ats=(T * k3.Engine.atts_scratch_floats(nkv, eng.splits), f32))

    @staticmethod
    def need_mib(T, eng=None, taps=DFLASH_TAPS):
        """MiB of device memory one verify tape adds (with eng: from its dimensions; without: an upper bound for T <= 4
        to pass as Q3MoEOne's vram_extra_mib before the engine exists)"""
        if eng is None: return 16.0 * T
        sz = {torch.float32: 4, torch.float16: 2, torch.int32: 4}
        return sum(n * sz[dt] for n, dt in Q3MoEVerify._spec(eng, T, taps).values()) / 2 ** 20 + 2.0   # (+ the tape and its sync words)

    def __init__(self, eng, T, module='k3_q3m_ver', per_wgp=None, taps=DFLASH_TAPS, vram_check=True):
        """eng: a Q3MoEOne (its weights, cache and position bounds); T: tokens per pass, 1 .. 4; module: k3_q3m_ver or
        k3_q3m_ver_trace; taps: layers whose entering residual each pass keeps per token (feats)"""
        if not isinstance(T, int) or not 1 <= T <= MAX_T: raise ValueError(f'Q3MoEVerify: T {T!r}: 1 .. {MAX_T}')
        if module not in k3.Q3M_VER_MODULES: raise ValueError(f'Q3MoEVerify: module {module!r} is not a verify module {k3.Q3M_VER_MODULES}')
        if eng.k_used != 8: raise ValueError(f'Q3MoEVerify: {eng.k_used} experts per token (the grouped segments take 8 slots per token)')
        taps = tuple(int(t) for t in taps)
        if any(not 0 <= t < len(eng.L) for t in taps) or len(set(taps)) != len(taps): raise ValueError(f'Q3MoEVerify: taps {taps}')
        self.eng, self.T, self.taps, self.module = eng, T, taps, module
        self.mib = self.need_mib(T, eng, taps)
        if vram_check: capacity.check([eng.dev], [self.mib], f'Q3MoEVerify T {T}: its tape buffers ({self.mib:.1f} MiB)')
        self._build(per_wgp)

    def _build(self, per_wgp):
        eng, T, taps = self.eng, self.T, self.taps
        D, hd, nh, nkv, KU, FF = eng.D, eng.hd, eng.nh, eng.nkv, eng.k_used, eng.ffn
        M = nkv * hd; sc = 1.0 / np.sqrt(hd); QW = nh * hd + 2 * M; NL = len(eng.L); uni = T > 1
        with torch.cuda.device(eng.dev):
            spec = self._spec(eng, T, taps)
            S = self.S = {k: torch.zeros(n, dtype=dt, device='cuda') for k, (n, dt) in spec.items()}
            S['neg'].fill_(-1.0)
            self.hst = torch.zeros(T + 1, dtype=torch.int32, pin_memory=True)
            dst = S['stage']; pw = dst[T:T + 1]                     # the token ids, then the pass's first position
            na = k3.Engine.atts_scratch_floats(nkv, eng.splits)
            E = self.E = k3.Engine(module=self.module, grid=k3.resident_grid(self.module, per_wgp or eng.per_wgp), T=1,
                                   limit_us=eng.limit_us)
            mv = lambda w, x16, out, dep, rpt='qkv', **kw: E.own_matvec_1rt(w[3], w[0], w[1], w[2], x16, out, dep=dep,
                                                                           rows_per_task=eng.RPT[rpt], **kw)
            mt = lambda xs, ys: dict(tokens=T, x_stride=xs, out_stride=ys) if T > 1 else {}
            prev = E.own_emb(eng.emb_dev, 'Q4_K', int(eng.emb.shape[1]), D, dst[:T], S['x'], tokens=T)
            for il, d in enumerate(eng.L):
                ids = S['ids'][il * T * KU:(il + 1) * T * KU]
                at = lambda t, dep, **kw: E.own_atts(S['qkv'][t * QW:(t + 1) * QW], S['ats'][t * na:(t + 1) * na], d['qk_par'],
                                                     d['cache'], pw, S['att16'][t * nh * hd:(t + 1) * nh * hd], nh, nkv, hd, sc,
                                                     splits=eng.splits, lmin=eng.lmin, dep=dep, tok=t, **kw)
                n1 = E.own_add_rms16(S['x'], d['attn_norm'], S['h'], S['h16'], eng.eps, d=S['d'] if il else None,
                                     slots=KU if il else 0, w=S['wts'] if il else None, dep=prev, tokens=T)
                pend = None
                if il in taps:      # feat = x entering the layer: feat += -feat (off the layer's critical path), then += x
                    i = taps.index(il); f = S['feat'][i * T * D:(i + 1) * T * D]
                    za = E.own_add_rms16(f, d['attn_norm'], S['fh'], S['fh16'], eng.eps, d=f, slots=1, w=S['neg'], dep=prev, tokens=T)
                    pend = E.also(E.own_add_rms16(f, d['attn_norm'], S['fh'], S['fh16'], eng.eps, d=S['x'], slots=1, dep=n1, tokens=T), za)
                v_ = mv(d['attn_v'], S['h16'], S['qkv'][nh * hd + M:], n1, **mt(D, QW))
                k_ = mv(d['attn_k'], S['h16'], S['qkv'][nh * hd:], n1, **mt(D, QW))
                jn = E.also(E.nop(1, 0, dep=k_), v_)
                q_ = mv(d['attn_q'], S['h16'], S['qkv'], n1, **mt(D, QW))
                if T == 1: c2 = E.also(at(0, q_), jn)
                else:               # every row of the pass first, then the T attentions side by side
                    rows = _join(E, [at(t, jn, row_only=True) for t in range(T)])
                    c2 = _join(E, [E.also(at(t, q_, no_store=True), rows) for t in range(T)])
                o_ = mv(d['attn_output'], S['att16'], S['o'], c2, rpt='o', **mt(nh * hd, D))
                n2 = E.own_add_rms16(S['x'], d['ffn_norm'], S['h'], S['h16'], eng.eps, d=S['o'], slots=1, dep=o_, tokens=T)
                if pend is not None: n2 = E.also(n2, pend)
                rl = E.f32_mv(d['router'], S['h'], S['logit'], dep=n2, tokens=T)
                ro = E.router(S['logit'], KU, ids, S['wts'], dep=rl, tokens=T)
                sw = mv(d['ffn_gate_exps'], S['h16'], S['a16'], ro, rpt='gu', ids=ids, total_rows=eng.n_exp * FF,
                        up=d['ffn_up_exps'][3], x_per_token=True, union=uni)
                prev = mv(d['ffn_down_exps'], S['a16'], S['d'], sw, rpt='down', ids=ids, total_rows=eng.n_exp * D,
                          x_per_slot=True, union=uni)
            nf = E.own_add_rms16(S['x'], eng.out_norm, S['h'], S['h16'], eng.eps, d=S['d'], slots=KU, w=S['wts'], dep=prev, tokens=T)
            mv(eng.head, S['h16'], S['logits'], nf, rpt='head', **mt(D, eng.vocab))
            E.build()

    def run(self, tokens, pos0, trace=None):
        """T tokens at positions pos0 .. pos0 + T - 1 in one launch (waited for). Refused before anything is written:
        a poisoned engine, a span outside the KV cache, a token outside the vocabulary. A wait that gave up is run again
        (the engine's max_retries); the device bound guard poisons the engine until its reset()."""
        eng, T, E = self.eng, self.T, self.E
        if eng.poisoned: raise k3.ContextBoundsError(f'Q3MoEVerify pass: refused until reset(): {eng.poisoned}')
        if len(tokens) != T: raise ValueError(f'Q3MoEVerify pass: {len(tokens)} tokens for a T = {T} tape')
        k3.check_span(pos0, T, eng.max_ctx, what='Q3MoEVerify pass')
        toks = [int(t) for t in tokens]
        if any(not 0 <= t < eng.vocab for t in toks): raise ValueError(f'Q3MoEVerify pass: a token of {toks} outside the vocabulary ({eng.vocab})')
        for t in range(T): self.hst[t] = toks[t]
        self.hst[T] = int(pos0)
        na = k3.Engine.atts_scratch_floats(eng.nkv, eng.splits)
        with torch.cuda.device(eng.dev):
            for attempt in range(eng.max_retries + 1):
                self.S['stage'].copy_(self.hst, non_blocking=True)   # (the pinned block is rewritten only after the wait in run)
                try:
                    E.run() if trace is None else E.run(trace=trace)
                    torch.cuda.synchronize()
                    return
                except k3.ContextBoundsError as e:       # (the host checks above should have caught it: never retried)
                    eng.poisoned = f'the device bound guard refused a verify pass at position {pos0}: {e}'
                    raise
                except RuntimeError as e:                # a wait gave up: E.run cleared the status words
                    if 'a wait gave up' not in str(e): raise
                    torch.cuda.synchronize()
                    for t in range(T): self.S['ats'][t * na:t * na + k3.Engine.ATS_CNT].zero_()   # (OP_OWN_ATTS's merge counters)
                    eng.bails.append((int(pos0), attempt, [str(e)])); bad = str(e)
            raise RuntimeError(f'k3 qwen3moe verify: a wait gave up at position {pos0} in {eng.max_retries + 1} attempts: {bad}')

    def logits(self, t):
        """row t's logits (fp32 [vocab], on the device; valid until the next pass)"""
        V = self.eng.vocab
        return self.S['logits'][t * V:(t + 1) * V]

    def feats(self, t):
        """row t's tap features: the residual entering each tap layer, concatenated (fp32 [len(taps) * D])"""
        T, D = self.T, self.eng.D
        return torch.cat([self.S['feat'][(i * T + t) * D:(i * T + t + 1) * D] for i in range(len(self.taps))])

    def distinct(self):
        """distinct experts per layer in the last pass (the mean over the layers), of T * 8 slots"""
        ids = self.S['ids'].view(len(self.eng.L), -1).cpu().numpy()
        return float(np.mean([len(set(r.tolist())) for r in ids]))
