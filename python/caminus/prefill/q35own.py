"""The batched prefill of the dense qwen35 line on Caminus's own code (caminus.engine.q35_dense.Dense35Own: Swift 1.5 Q4_K_M
first): the layer loop of a chunk over the decode engine's own weights and states (Dense35Own.L: references, nothing
copied). One card; no ggml-cuda code: the GEMMs are kernels/q35_prefill.hip's (WMMA, the weights exact, fp16 activations,
fp32 sums), every other op runs its decode op's statements over the chunk's rows.

Per layer of a chunk of n tokens at positions p .. p + n - 1, the decode tape's ops in its order:
  h, h16 = add_rms16(x (+ the previous layer's feed-forward rows), attn_norm)          q35p_add_rms16 (OP_OWN_ADD_RMS16)
  GDN       alpha / beta (F32: pf_f32_mv_k5120, OP_F32_MV's order; a quantized pair: the own GEMM), gates (pf_gdn_gates),
            attn_qkv (GEMM), the conv + its state (pf_gdn_conv / pf_conv_state), l2 (pf_gdn_l2), attn_gate (GEMM), the
            recurrence (pf_gdn_step: the decode's state in place, read once, written once), the output norm with the SiLU
            gate as fp16 (q35p_gdn_norm16), ssm_out (GEMM)
  attention attn_q / k / v (GEMMs); k norm + RoPE (pf_qk_nr), the f16 KV append (q35p_kv_append16), q norm + RoPE,
            the attention over positions 0 .. pos (q35p_attn16: own_op_attn_dec's batched form), the output gate as fp16
            (q35p_gate_mul16), attn_output (GEMM)
  h, h16 = add_rms16(x += block, post_attention_norm)
  FFN       gate / up (GEMMs), fp16 silu(g) u (q3p_swiglu), down (GEMM) -> added at the next layer's norm
The last layer stops after its recurrent state or its cache rows: its output feeds only the prompt's logits, which a
prefill never returns (the last prompt token goes through a K3 pass: the first logits come from decode numerics).
Every op but the GEMMs is bitwise its K3 decode op on the same inputs (gate G1, tests/prefill/test_q35_prefill_ops.py);
the GEMMs sum in WMMA order, so the prefill is G-model gated (prefill-then-decode against the float64 reference, with
llama.cpp's own prefill-then-decode as the bound). Chunk geometry changes no bit (a row's ops see only its own row and the
rows before it, in the same order at every chunking).

The hand-off is the shared driver's (caminus.prefill.driver.PrefillDriver): run(toks, i0, commit, p1) prefills
[i0 - commit, p1) (committed state: the GDN states, the conv states and the KV rows, written in place), sets m.fresh and
returns (i0 = p1, commit = 0): the next K3 pass commits nothing. No drafter yet (draft must be None)."""
import ctypes
import math

import numpy as np
import torch

from ..device import capacity
from ..engine import k3
from . import q3m_ops
from .attn import Attn
from .driver import PrefillDriver, PrefillRefused
from .gdn import Gdn
from .glue import Glue

I, F = ctypes.c_int, ctypes.c_float
NULL = ctypes.c_void_p(0)
MODULE = 'q35_prefill'
MODULE_TCAP = 'q35_prefill_tcap'   # (ThinkingCap's formats: kernels/q35_prefill_tcap.hip, Swift's object + the IQ GEMMs)
FORMATS = ('Q4_K', 'Q6_K', 'Q8_0', 'Q5_K')
FORMATS_TCAP = FORMATS + ('IQ4_XS', 'IQ3_XXS', 'IQ3_S', 'IQ2_S')
MM_KERNEL = {'Q8_0': 'q35p_mm_q8_0', 'Q5_K': 'q35p_mm_q5k'}
# the codebook formats (the tcap object only): the decode engine's copies in place (IQ4_XS raw blocks; the grid formats
# in the record layout), one more argument: the buffer's rows (Rtot)
IQ_KERNEL = {'IQ4_XS': 'q3p_mm_iq4xs', 'IQ3_XXS': 'q3p_mm_iq3xxs', 'IQ3_S': 'q3p_mm_iq3s', 'IQ2_S': 'q35t_mm_iq2s'}
# token-positions per attention launch (rows x the deepest position they read: one workgroup per (row, query head) walks
# every position below its own). Unmeasured; the G1 test times the largest launch it runs
ATTN_TP = 1 << 20
# T*: below it a prompt part runs as K3 one-token passes (bitwise the decode). Not measured yet: Qwen3.6's 12 until the
# speed test sets it
CROSSOVER = 12


class Ops(q3m_ops.Ops):
    """kernels/q35_prefill.hip's launchers: q3m_prefill's (included as they are) + the dense qwen35 line's own"""

    def __init__(self, dev, module=MODULE):
        super().__init__(dev, module)
        self.module = module

    def mm(self, W, x16, ldx, y, ldy, n=None, work=None):
        """y[t] = W x16[t] for t < n (dense only here). W: Dense35Own's dict (fmt, rows, K, w)"""
        if isinstance(W, dict): W = (W['fmt'], W['rows'], W['K'], W['w'])
        fmt, R, K, wb = W[:4]
        if fmt in IQ_KERNEL:
            if self.module != MODULE_TCAP: raise ValueError(f'q35 mm: {fmt} needs the {MODULE_TCAP} object (this one: {self.module})')
            if K % 256 or work is not None or n is None: raise ValueError(f'q35 mm: {fmt}: K {K} (a multiple of 256), dense rows (n) only')
            n = int(n); grid = (-(-n // q3m_ops.MM_M), -(-R // q3m_ops.MM_N))
            getattr(self.mod, IQ_KERNEL[fmt])(grid, (256,), wb, I(R), I(R), I(K), x16, I(ldx), NULL, y, I(ldy), NULL, NULL, NULL, I(n))
            return
        if fmt not in MM_KERNEL: return super().mm((fmt, R, K, wb), x16, ldx, y, ldy, n=n, work=work)
        if K % 256: raise ValueError(f'q35 mm: K {K} is not a multiple of 256')
        if work is not None or n is None: raise ValueError(f'q35 mm: {fmt} takes dense rows (n) only')
        n = int(n)
        grid = (-(-n // q3m_ops.MM_M), -(-R // q3m_ops.MM_N))
        getattr(self.mod, MM_KERNEL[fmt])(grid, (256,), wb, I(R), I(K), x16, I(ldx), NULL, y, I(ldy), NULL, NULL, NULL, I(n))

    def emb_iq2s(self, W, rb, vocab, toks, x, n, N):
        """x [n][N] = the IQ2_S embedding rows toks[i] (OP_OWN_EMB's IQ2_S statements: bitwise the host decoder)"""
        if self.module != MODULE_TCAP: raise ValueError(f'q35 emb_iq2s: the {MODULE_TCAP} object only')
        self.mod.q35t_emb_iq2s((int(n),), (256,), W, I(rb), I(vocab), toks, x, I(N))

    def add_rms16(self, x, nw, eps, h16, n, N, d=None, h=None):
        """per row: x += d (d None: no add); h = rms_norm(x) * nw (fp32, always written: the kernel's fp16 rounding
        depends on it, see kernels/q35_prefill.hip); h16 = the decode op's fp16 copy"""
        if N % 256 or not 256 <= N <= 65536: raise ValueError(f'q35 add_rms16: N {N} (a multiple of 256)')
        if h is None: raise ValueError('q35 add_rms16: h (the fp32 output) is required')
        self.mod.q35p_add_rms16((int(n),), (256,), x, NULL if d is None else d, nw, F(eps), h, h16, I(N))

    def gdn_norm16(self, o, w, z, y16, n, H, K, eps):
        self.mod.q35p_gdn_norm16((int(n) * int(H),), (256,), o, w, z, y16, I(K), F(eps))

    def kv_append16(self, k, ldk, v, ldv, cache, M, pos0, n, rows, err):
        """cache rows pos0 .. pos0 + n - 1 = (k, v) as fp16; rows: the positions the cache holds (k3.check_span first)"""
        alloc = cache.numel() // (2 * int(M))
        if not 0 < int(rows) <= alloc: raise ValueError(f'q35 kv_append16: rows {rows} of a cache of {alloc}')
        k3.check_span(pos0, n, int(rows), what='q35 kv_append16')
        self.mod.q35p_kv_append16((int(n),), (256,), k, I(ldk), v, I(ldv), cache, I(M), I(pos0), I(rows), err)

    def attn16(self, q, cache, pos0, n, heads, nkv, D, scale, out, rows, r0=0, r1=None):
        """rows r0 .. r1 - 1 of a chunk at positions pos0 .. pos0 + n - 1 (one launch): out[r, h] = the attention of q[r, h]
        over cache positions 0 .. pos0 + r. q / out: the chunk's [n, heads, D] fp32 (launch views from row r0)"""
        r1 = int(n) if r1 is None else int(r1)
        if D % 128 or D > 512 or heads % nkv or not 0 <= r0 < r1 <= n: raise ValueError(f'q35 attn16: D {D} heads {heads} nkv {nkv} rows [{r0}, {r1}) of {n}')
        alloc = cache.numel() // (2 * nkv * D)
        if not 0 < int(rows) <= alloc: raise ValueError(f'q35 attn16: rows {rows} of a cache of {alloc}')
        k3.check_span(pos0, n, int(rows), what='q35 attn16')
        off = r0 * heads * D
        if q.numel() < r1 * heads * D or out.numel() < r1 * heads * D: raise ValueError('q35 attn16: q / out smaller than the rows')
        es = q.element_size()
        self.mod.q35p_attn16(((r1 - r0) * heads,), (256,), ctypes.c_void_p(q.data_ptr() + off * es), cache, I(pos0), I(r0),
                             I(heads), I(nkv), I(D), F(scale), ctypes.c_void_p(out.data_ptr() + off * es))

    def attn_rows(self, pos0, n):
        """[(r0, r1)]: the chunk's rows in launches of at most ATTN_TP token-positions"""
        out, r = [], 0
        while r < n:
            r1 = r + max(1, ATTN_TP // (pos0 + n))
            out.append((r, min(n, r1))); r = min(n, r1)
        return out

    def gate_mul16(self, a, gate, y16, N, D, stride, off):
        self.mod.q35p_gate_mul16((min(-(-int(N) // 256), 8192),), (256,), a, gate, y16, ctypes.c_longlong(int(N)), I(D), I(stride), I(off))


def layout(m, T):
    """(f32 counts, f16 counts, i32 counts) of Q35OwnPrefill's buffers for engine m (Dense35Own, or shapes_of) at chunk T.
    `work` is shared by the layer kinds and the feed-forward (each is dead before the next one writes)"""
    D, nh, nkv, hd, ffn, S, NK, NV = m.D, m.nh, m.nkv, m.hd, m.ffn, m.S, m.NK, m.NV
    CH, KD, VD, QH = 2 * S * NK + S * NV, NK * S, NV * S, nh * hd
    gdn = 2 * CH + 2 * KD + 2 * VD                       # qkv, conv, qk, z, o
    att = 2 * QH + 3 * nkv * hd + 2 * QH                 # qf, kr, vr, ko, qo, att
    f32 = dict(x=T * D, h=T * D, blk=T * D, ffp=T * D, al=T * NV, be=T * NV, gates=T * 2 * NV,
               work=T * max(gdn, att, 2 * ffn))
    f16 = dict(h16=T * D, x16=T * max(VD, QH, ffn))     # (the output GEMMs' fp16 input: fin16 / gat16 / the swiglu)
    i32 = dict(toks=T, err=1)
    return f32, f16, i32


def buffer_need_mib(m, T):
    f32, f16, i32 = layout(m, T)
    return (4 * sum(f32.values()) + 2 * sum(f16.values()) + 4 * sum(i32.values())) / capacity.MIB


def shapes_of(path):
    """the shapes layout() reads, from a dense qwen35 GGUF's header alone (before any load)"""
    from types import SimpleNamespace
    from .. import ggufio
    from ..engine.q35_dense import _shapes
    F = ggufio.GGUFFile(path)
    try: return SimpleNamespace(**_shapes(F))
    finally: F.close()


class Q35OwnPrefill(PrefillDriver):
    """the batched prefill of a Dense35Own engine m, chunks of up to `chunk` tokens; buffers preallocated (checked against
    the card's guard line first: capacity.check)"""

    def __init__(self, m, chunk=1024, crossover=None, check=True):
        self.m, self.chunk = m, int(chunk)
        if not 16 <= self.chunk <= 4096: raise ValueError(f'Q35OwnPrefill: chunk {chunk} (16 .. 4096)')
        if m.nl_build != m.nl: raise ValueError('Q35OwnPrefill: a debug_layers engine is refused')
        self.dev = dev = torch.device(f'cuda:{m.dev}')
        self.crossover = CROSSOVER if crossover is None else int(crossover)
        self.tcap = bool(getattr(m, 'tcap', False))
        fm = FORMATS_TCAP if self.tcap else FORMATS
        bad = [f"blk.{il} {k}: {L[k]['fmt']}" for il, L in enumerate(m.L) for k in self.MM_KEYS[L['kind']] + ('fg', 'fu', 'fd')
               if L[k]['fmt'] not in fm]
        bad += [f"blk.{il} {k}: {L[k]['fmt']}" for il, L in enumerate(m.L) if L['kind'] == 'gdn' for k in ('al', 'be')
                if L[k]['fmt'] not in fm + ('F32',)]
        if m.emb.type not in (('Q4_K', 'IQ2_S') if self.tcap else ('Q4_K',)): bad.append(f'token_embd: {m.emb.type}')
        if bad: raise ValueError(f'Q35OwnPrefill: {len(bad)} matrices without a prefill kernel (first: {"; ".join(bad[:4])})')
        if m.D != 5120: raise ValueError(f'Q35OwnPrefill: n_embd {m.D} (the F32 alpha / beta kernel is pf_f32_mv_k5120)')
        attn = {6: 'prefill_attn_q35d', 8: 'prefill_attn_q35'}.get(m.nh // m.nkv)
        if attn is None: raise ValueError(f'Q35OwnPrefill: {m.nh // m.nkv} query heads per kv head')
        T = self.chunk
        self.need_mib = buffer_need_mib(m, T)
        if check: capacity.check([dev.index], [self.need_mib], f'Q35OwnPrefill buffers (chunk {T})')
        f32, f16, i32 = layout(m, T)
        with torch.cuda.device(dev):
            self.O = Ops(dev, module=MODULE_TCAP if self.tcap else MODULE)
            self.Gl = Glue(dev, module='prefill_glue_q35d')
            self.Gd = Gdn(dev, module='prefill_gdn_q35')
            self.A = Attn(dev, name=attn)
            for n, c in f32.items(): setattr(self, n, torch.zeros(c, dtype=torch.float32, device=dev))
            for n, c in f16.items(): setattr(self, n, torch.zeros(c, dtype=torch.float16, device=dev))
            for n, c in i32.items(): setattr(self, n, torch.zeros(c, dtype=torch.int32, device=dev))
        self.host_toks = torch.zeros(T, dtype=torch.int32, pin_memory=True)
        self.host_ev = None
        self.n_vocab = int(m.vocab)
        self.emb_rb = m.D // 256 * (82 if m.emb.type == 'IQ2_S' else 144)
        self.last = None

    MM_KEYS = dict(gdn=('qkv', 'z', 'so'), attn=('q', 'k', 'v', 'o'))

    def check_drafter(self, draft):
        if draft is not None: raise PrefillRefused('Q35OwnPrefill: no drafter yet')

    def numerics(self):
        """the engine key's prefill part: own WMMA GEMMs (fp16 x, exact weights, fp32 sums)"""
        return dict(wmma_out='f32', router_logits='none', moe='none', q6k='own-wmma', dense='own-wmma')

    def _ab(self, W, h, h16, out, n):
        """an ssm alpha / beta row block: F32 (OP_F32_MV's order on the fp32 h) or a quantized matrix (the own GEMM on h16)"""
        m = self.m
        if W['fmt'] == 'F32': self.Gl.f32_mv(W['w'], m.NV, m.D, h, out, n)
        else: self.O.mm(W, h16, m.D, out, m.NV, n=n)

    def _chunk(self, toks, a, b):
        m, O, Gl, Gd, A = self.m, self.O, self.Gl, self.Gd, self.A
        n, p = b - a, a
        D, nh, nkv, hd, ffn, S, NK, NV, eps = m.D, m.nh, m.nkv, m.hd, m.ffn, m.S, m.NK, m.NV, m.eps
        CH, KD, VD, QH, M = 2 * S * NK + S * NV, NK * S, NV * S, nh * hd, nkv * hd
        if self.host_ev is not None: self.host_ev.synchronize()   # (the stage is rewritten after its last copy finished)
        self.host_toks[:n].numpy()[:] = np.asarray(toks[a:b], dtype=np.int32)
        with torch.cuda.device(self.dev):
            self.toks[:n].copy_(self.host_toks[:n], non_blocking=True)
            ev = torch.cuda.Event(); ev.record(); self.host_ev = ev
        x, h, blk, ffp, h16, x16 = self.x[:n * D], self.h[:n * D], self.blk[:n * D], self.ffp[:n * D], self.h16[:n * D], self.x16
        W = self.work
        (O.emb_iq2s if m.emb.type == 'IQ2_S' else O.emb)(m.emb_dev, self.emb_rb, self.n_vocab, self.toks, x, n, D)
        scale = 1.0 / float(np.sqrt(hd))
        last = m.nl - 1
        for il, L in enumerate(m.L):
            O.add_rms16(x, L['attn_norm'], eps, h16, n, D, d=ffp if il else None, h=h)
            if L['kind'] == 'gdn':
                al, be, gt = self.al[:n * NV], self.be[:n * NV], self.gates[:n * 2 * NV]
                o0 = 0
                qkv = W[o0:o0 + n * CH]; o0 += n * CH
                conv = W[o0:o0 + n * CH]; o0 += n * CH
                qk = W[o0:o0 + n * 2 * KD]; o0 += n * 2 * KD
                z = W[o0:o0 + n * VD]; o0 += n * VD
                att = W[o0:o0 + n * VD]
                self._ab(L['al'], h, h16, al, n); self._ab(L['be'], h, h16, be, n)
                Gd.gates(al, be, L['dt'], L['a'], gt, n, NV)
                O.mm(L['qkv'], h16, D, qkv, CH, n=n)
                Gl.gdn_conv(qkv, L['conv_w'], L['conv_st'], conv, CH, n)
                Gd.l2(conv, qk, n, 2 * NK, eps, ldx=CH)
                Gd.step(qk, conv[2 * KD:], gt, L['gdn'], att, n, NV, NK, CH)
                if il == last: break                       # (the last layer: only its recurrent states are needed)
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
                O.kv_append16(ko, M, vr, M, L['cache'], M, p, n, m.max_ctx, self.err)
                if il == last: break                       # (the last layer: only its cache rows are needed)
                O.mm(L['q'], h16, D, qf, 2 * QH, n=n)
                A.qk_nr(qf, 2 * hd, nh, hd, L['qn'], m.nrot, m.theta_scale, eps, p, qo, n)
                for r0, r1 in O.attn_rows(p, n):
                    O.attn16(qo, L['cache'], p, n, nh, nkv, hd, scale, att, m.max_ctx, r0, r1)
                O.gate_mul16(att, qf, x16, n * QH, hd, 2 * hd, hd)
                O.mm(L['o'], x16, QH, blk, D, n=n)
            O.add_rms16(x, L['post_norm'], eps, h16, n, D, d=blk, h=h)
            g, u = W[:n * ffn], W[n * ffn:2 * n * ffn]
            O.mm(L['fg'], h16, D, g, ffn, n=n); O.mm(L['fu'], h16, D, u, ffn, n=n)
            O.swiglu(g, u, x16, n * ffn)
            O.mm(L['fd'], x16, ffn, ffp, D, n=n)

    def check_flags(self, what='prefill'):
        e = int(self.err.item())
        if e:
            self.err.zero_()
            raise k3.ContextBoundsError(f'{what}: the KV append refused rows (error word {e:#x})')

    def _run_chunk(self, toks, k, j, a, b, segs, bounds):
        self._chunk(toks, a, b)

    def _run_settle(self):
        torch.cuda.synchronize(self.dev); self.check_flags('prefill')

    def _sync_devices(self):
        torch.cuda.synchronize(self.dev)
