"""The batched prefill of the dense qwen35 line on TWO cards (caminus.engine.q35_dense_tp.Dense35TP), on Caminus's own code:
caminus.prefill.q35own's chunk, each card running its share (its heads, its feed-forward rows, its states and KV rows: the
decode engine's own per-card weights and states, Dense35TP.L[il][c], nothing copied) with the same kernels, and the two
partial sums per block (the attention / Gated DeltaNet output, the feed-forward output) summed between the cards by
caminus.prefill.xchg.Exchange: each card's partial rows pushed to the other (hipMemcpyPeerAsync on copy streams), then
p0 + p1 in place on both cards, card 0's partial first (the decode's OP_XCHG rule: the replicated residual stays the same
bits on both cards). Ordering is HIP events only.

The exchange runs between the two halves of a block, serialized (no micro-chunk overlap yet): a chunk of n tokens moves
2 nl x n x D fp32 each way. The last layer stops after its recurrent state or its cache rows on each card, as the one-card
prefill. The hand-off is the shared driver's (caminus.prefill.driver.PrefillDriver), as Q35OwnPrefill."""
from types import SimpleNamespace

import numpy as np
import torch

from ..device import capacity
from ..engine import k3
from .attn import Attn
from .driver import PrefillDriver, PrefillRefused
from .gdn import Gdn
from .glue import Glue
from .q35own import Ops, layout, buffer_need_mib, MODULE, MODULE_TCAP, FORMATS, FORMATS_TCAP, CROSSOVER
from .xchg import Exchange, xchg_bytes   # (not .tp: it imports the vendored MMQ path and gguf-py)


def local_shapes(m, c):
    """card c's shapes of a Dense35TP (the ones layout() reads): its heads, its feed-forward rows"""
    _, kc, VH = m.gsplit[c]
    return SimpleNamespace(D=m.D, nh=m.nh // 2, nkv=m.nkv // 2, hd=m.hd, ffn=m.f0 if c == 0 else m.ffn - m.f0, S=m.S,
                           NK=kc, NV=len(VH))


def tp_shapes(path, ffn_share=0.5):
    """[card 0's, card 1's] local shapes from a dense qwen35 GGUF's header alone (before any load)"""
    from .q35own import shapes_of
    from ..engine.q35_dense_tp import ffn_cut
    s = shapes_of(path); f0 = ffn_cut(s.ffn, ffn_share)
    return [SimpleNamespace(D=s.D, nh=s.nh // 2, nkv=s.nkv // 2, hd=s.hd, ffn=f0 if c == 0 else s.ffn - f0, S=s.S, NK=s.NK // 2,
                            NV=s.NV // 2) for c in (0, 1)]


def buffer_need_tp_mib(shapes, T):
    """per card: Q35OwnPrefillTP's buffers at chunk T (its layout at its local shapes + the exchange's receive rows)"""
    return [buffer_need_mib(s, T) + xchg_bytes(T, s.D) / capacity.MIB for s in shapes]


class Q35OwnPrefillTP(PrefillDriver):
    """the batched prefill of a Dense35TP engine m, chunks of up to `chunk` tokens; buffers preallocated on both cards
    (checked against each card's guard line first)"""
    MM_KEYS = dict(gdn=('qkv', 'z', 'so'), attn=('q', 'k', 'v', 'o'))

    def __init__(self, m, chunk=1024, crossover=None, check=True):
        self.m, self.chunk = m, int(chunk)
        if not 16 <= self.chunk <= 4096: raise ValueError(f'Q35OwnPrefillTP: chunk {chunk} (16 .. 4096)')
        if m.nl_build != m.nl: raise ValueError('Q35OwnPrefillTP: a debug_layers engine is refused')
        if m.local_only: raise ValueError('Q35OwnPrefillTP: a local_only engine is refused (its numbers are not the model\'s)')
        self.devs = [torch.device(f'cuda:{d}') for d in m.devs]
        self.crossover = CROSSOVER if crossover is None else int(crossover)
        self.tcap = bool(m.tcap)
        fm = FORMATS_TCAP if self.tcap else FORMATS
        bad = []
        for il, dd in enumerate(m.L):
            L = dd[0]
            bad += [f"blk.{il} {k}: {L[k]['fmt']}" for k in self.MM_KEYS[L['kind']] + ('fg', 'fu', 'fd') if L[k]['fmt'] not in fm]
            if L['kind'] == 'gdn': bad += [f"blk.{il} {k}: {L[k]['fmt']}" for k in ('al', 'be') if L[k]['fmt'] not in fm + ('F32',)]
        if m.emb.type not in (('Q4_K', 'IQ2_S') if self.tcap else ('Q4_K',)): bad.append(f'token_embd: {m.emb.type}')
        if bad: raise ValueError(f'Q35OwnPrefillTP: {len(bad)} matrices without a prefill kernel (first: {"; ".join(bad[:4])})')
        if m.D != 5120: raise ValueError(f'Q35OwnPrefillTP: n_embd {m.D} (the F32 alpha / beta kernel is pf_f32_mv_k5120)')
        attn = {6: 'prefill_attn_q35d', 8: 'prefill_attn_q35'}.get(m.nh // m.nkv)
        if attn is None: raise ValueError(f'Q35OwnPrefillTP: {m.nh // m.nkv} query heads per kv head')
        T = self.chunk
        self.sh = [local_shapes(m, c) for c in (0, 1)]
        self.need_mib = buffer_need_tp_mib(self.sh, T)
        if check: capacity.check(list(m.devs), self.need_mib[:], f'Q35OwnPrefillTP buffers (chunk {T}, both cards)')
        self.O, self.Gl, self.Gd, self.A, self.B = [], [], [], [], []
        for c in (0, 1):
            dev = self.devs[c]; f32, f16, i32 = layout(self.sh[c], T)
            with torch.cuda.device(dev):
                self.O.append(Ops(dev, module=MODULE_TCAP if self.tcap else MODULE))
                self.Gl.append(Glue(dev, module='prefill_glue_q35d')); self.Gd.append(Gdn(dev, module='prefill_gdn_q35'))
                self.A.append(Attn(dev, name=attn))
                b = {}
                for n, k in f32.items(): b[n] = torch.zeros(k, dtype=torch.float32, device=dev)
                for n, k in f16.items(): b[n] = torch.zeros(k, dtype=torch.float16, device=dev)
                for n, k in i32.items(): b[n] = torch.zeros(k, dtype=torch.int32, device=dev)
                self.B.append(b)
        self.streams = [torch.cuda.current_stream(d) for d in self.devs]
        self.X = Exchange(m.devs, T, m.D, self.streams, check=False)          # (its rows are counted in need_mib above)
        self.host_toks = torch.zeros(T, dtype=torch.int32, pin_memory=True)
        self.host_ev = [None, None]
        self.n_vocab = int(m.vocab)
        self.emb_rb = m.D // 256 * (82 if m.emb.type == 'IQ2_S' else 144)
        self.last = None

    def check_drafter(self, draft):
        if draft is not None: raise PrefillRefused('Q35OwnPrefillTP: no drafter yet')

    def numerics(self):
        """the engine key's prefill part: own WMMA GEMMs (fp16 x, exact weights, fp32 sums), the two cards' partials summed in
        fp32, card 0's first"""
        return dict(wmma_out='f32', router_logits='none', moe='none', q6k='own-wmma', dense='own-wmma', tp='xchg-fp32-c0-first')

    def _ab(self, c, W, h, h16, out, n):
        s = self.sh[c]
        if W['fmt'] == 'F32': self.Gl[c].f32_mv(W['w'], s.NV, s.D, h, out, n)
        else: self.O[c].mm(W, h16, s.D, out, s.NV, n=n)

    def _first(self, c, toks, a, b):
        """card c: the chunk's tokens staged and the embedding rows"""
        m, O, B = self.m, self.O[c], self.B[c]; n = b - a
        with torch.cuda.device(self.devs[c]):
            B['toks'][:n].copy_(self.host_toks[:n], non_blocking=True)
            ev = torch.cuda.Event(); ev.record(); self.host_ev[c] = ev
            (O.emb_iq2s if m.emb.type == 'IQ2_S' else O.emb)(m.emb_dev[c], self.emb_rb, self.n_vocab, B['toks'], B['x'][:n * m.D], n, m.D)

    def _mix(self, c, il, n, p):
        """card c, block il, the first half: the norm, then its Gated DeltaNet heads or attention heads into blk (its partial).
        Returns False on the last block (its states / cache rows written; nothing more)"""
        m, O, Gl, Gd, A, B, s = self.m, self.O[c], self.Gl[c], self.Gd[c], self.A[c], self.B[c], self.sh[c]
        L = m.L[il][c]
        D, nh, nkv, hd, S, NK, NV, eps = s.D, s.nh, s.nkv, s.hd, s.S, s.NK, s.NV, m.eps
        CH, KD, VD, QH, M = 2 * S * NK + S * NV, NK * S, NV * S, nh * hd, nkv * hd
        x, h, blk, ffp, h16, x16, W = (B['x'][:n * D], B['h'][:n * D], B['blk'][:n * D], B['ffp'][:n * D], B['h16'][:n * D],
                                       B['x16'], B['work'])
        last = il == m.nl - 1
        with torch.cuda.device(self.devs[c]):
            O.add_rms16(x, L['attn_norm'], eps, h16, n, D, d=ffp if il else None, h=h)
            if L['kind'] == 'gdn':
                al, be, gt = B['al'][:n * NV], B['be'][:n * NV], B['gates'][:n * 2 * NV]
                o0 = 0
                qkv = W[o0:o0 + n * CH]; o0 += n * CH
                conv = W[o0:o0 + n * CH]; o0 += n * CH
                qk = W[o0:o0 + n * 2 * KD]; o0 += n * 2 * KD
                z = W[o0:o0 + n * VD]; o0 += n * VD
                att = W[o0:o0 + n * VD]
                self._ab(c, L['al'], h, h16, al, n); self._ab(c, L['be'], h, h16, be, n)
                Gd.gates(al, be, L['dt'], L['a'], gt, n, NV)
                O.mm(L['qkv'], h16, D, qkv, CH, n=n)
                Gl.gdn_conv(qkv, L['conv_w'], L['conv_st'], conv, CH, n)
                Gd.l2(conv, qk, n, 2 * NK, eps, ldx=CH)
                Gd.step(qk, conv[2 * KD:], gt, L['gdn'], att, n, NV, NK, CH)
                if last: return False
                O.mm(L['z'], h16, D, z, VD, n=n)
                O.gdn_norm16(att, L['nw'], z, x16, n, NV, S, eps)
                O.mm(L['so'], x16, VD, blk, D, n=n)
            else:
                o0 = 0
                qf = W[o0:o0 + n * 2 * QH]; o0 += n * 2 * QH
                kr = W[o0:o0 + n * M]; o0 += n * M
                vr = W[o0:o0 + n * M]; o0 += n * M
                ko = W[o0:o0 + n * M]; o0 += n * M
                qo = W[o0:o0 + n * QH]; o0 += n * QH
                att = W[o0:o0 + n * QH]
                O.mm(L['k'], h16, D, kr, M, n=n); O.mm(L['v'], h16, D, vr, M, n=n)
                A.qk_nr(kr, hd, nkv, hd, L['kn'], m.nrot, m.theta_scale, eps, p, ko, n)
                O.kv_append16(ko, M, vr, M, L['cache'], M, p, n, m.max_ctx, B['err'])
                if last: return False
                O.mm(L['q'], h16, D, qf, 2 * QH, n=n)
                A.qk_nr(qf, 2 * hd, nh, hd, L['qn'], m.nrot, m.theta_scale, eps, p, qo, n)
                for r0, r1 in O.attn_rows(p, n):
                    O.attn16(qo, L['cache'], p, n, nh, nkv, hd, 1.0 / float(np.sqrt(hd)), att, m.max_ctx, r0, r1)
                O.gate_mul16(att, qf, x16, n * QH, hd, 2 * hd, hd)
                O.mm(L['o'], x16, QH, blk, D, n=n)
        return True

    def _ffn(self, c, il, n):
        """card c, block il, the second half: x += blk (the summed block), the norm, its feed-forward rows into ffp (its
        partial)"""
        m, O, B, s = self.m, self.O[c], self.B[c], self.sh[c]
        L = m.L[il][c]; D, ffn = s.D, s.ffn
        x, h, blk, ffp, h16, x16, W = (B['x'][:n * D], B['h'][:n * D], B['blk'][:n * D], B['ffp'][:n * D], B['h16'][:n * D],
                                       B['x16'], B['work'])
        with torch.cuda.device(self.devs[c]):
            O.add_rms16(x, L['post_norm'], m.eps, h16, n, D, d=blk, h=h)
            g, u = W[:n * ffn], W[n * ffn:2 * n * ffn]
            O.mm(L['fg'], h16, D, g, ffn, n=n); O.mm(L['fu'], h16, D, u, ffn, n=n)
            O.swiglu(g, u, x16, n * ffn)
            O.mm(L['fd'], x16, ffn, ffp, D, n=n)

    def _xchg(self, key, n):
        """the two cards' partial rows [0, n) of buffer key summed in place on both (card 0's first)"""
        t = self.X.post([self.B[0][key], self.B[1][key]], 0, n)
        self.X.sum(t)

    def _chunk(self, toks, a, b):
        n, p = b - a, a
        for c in (0, 1):
            if self.host_ev[c] is not None: self.host_ev[c].synchronize()   # (the stage is rewritten after its last copies)
        self.host_toks[:n].numpy()[:] = np.asarray(toks[a:b], dtype=np.int32)
        for c in (0, 1): self._first(c, toks, a, b)
        for il in range(self.m.nl):
            more = [self._mix(c, il, n, p) for c in (0, 1)]
            if not all(more): break
            self._xchg('blk', n)
            for c in (0, 1): self._ffn(c, il, n)
            self._xchg('ffp', n)

    def check_flags(self, what='prefill'):
        for c in (0, 1):
            e = int(self.B[c]['err'].item())
            if e:
                self.B[c]['err'].zero_()
                raise k3.ContextBoundsError(f'{what}: the KV append refused rows on card {c} (error word {e:#x})')

    def _run_chunk(self, toks, k, j, a, b, segs, bounds):
        self._chunk(toks, a, b)

    def _run_settle(self):
        self._sync_devices(); self.X.settle(); self.check_flags('prefill')

    def _sync_devices(self):
        for d in self.devs: torch.cuda.synchronize(d)
