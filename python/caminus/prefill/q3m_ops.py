"""Launch wrappers of kernels/q3m_prefill.hip: the batched prefill's own kernels for qwen3moe (Qwen3-30B-A3B), no ggml-cuda
code. A matrix is the decode engine's tuple (caminus.engine.q3moe_one.Q3MoEOne._w): (format, rows per matrix, K, device
bytes in the decode's layout: Q4_K raw blocks, Q6_K planes); an expert tensor is all its experts' rows in one buffer.

  Ops(dev).mm(W, x16, ldx, y, ldy, n=...)                  dense: y[t] = W x16[t] for t < n (fp32 out, row stride ldy)
  Ops(dev).mm(W, x16, ldx, y, ldy, work=(items, nitems, bound, xs, yd))   experts: the grouped pairs (group())
  emb / add_rms16 / router / topk / group / swiglu / qk_rope / attn: one kernel each (kernels/q3m_prefill.hip)
The study object q3m_prefill_iq (kernels/q3m_prefill_iq.hip, docs/dev/iq-quant-study.md) adds mm for the codebook formats
(IQ4_XS raw blocks; IQ3_XXS / IQ3_S / IQ2_XXS in engine.own_layout.iq_records' layout); on the shipped object the host
refuses them."""
import ctypes

import torch

from ..device import hip
from ..engine import k3
from ..engine.own_layout import IQ_REC, IQ_PAD

I, F = ctypes.c_int, ctypes.c_float
NULL = ctypes.c_void_p(0)
MM_M, MM_N = 64, 128        # a GEMM workgroup: tokens x rows
AT_T = 16                   # q3p_attn: tokens per workgroup
BLOCK_BYTES = {'Q4_K': 144, 'Q6_K': 210}
IQ_MODULE = 'q3m_prefill_iq'                                      # the study object: the shipped kernels + the four below
IQ_KERNEL = {'IQ4_XS': 'q3p_mm_iq4xs', 'IQ3_XXS': 'q3p_mm_iq3xxs', 'IQ3_S': 'q3p_mm_iq3s', 'IQ2_XXS': 'q3p_mm_iq2xxs'}


def formats(module):
    """the weight formats a prefill object's mm takes"""
    return tuple(BLOCK_BYTES) + (tuple(IQ_KERNEL) if module == IQ_MODULE else ())


def items_bound(npairs, n_exp):
    """the most work items q3p_group can write for npairs pairs over n_exp experts"""
    return -(-npairs // MM_M) + n_exp


class Ops:
    def __init__(self, dev, module='q3m_prefill'):
        self.dev = torch.device(dev) if not isinstance(dev, torch.device) else dev
        self.module = module
        with torch.cuda.device(self.dev):
            self.mod = hip.Module(module)

    # -- GEMMs -------------------------------------------------------------------------------------------------------------
    def mm(self, W, x16, ldx, y, ldy, n=None, work=None):
        """y[dst] = W x16[src] in fp32. Dense (n tokens: src = dst = t) or experts (work = (items, nitems, bound, xs, yd) from
        group(): bound = the grid's work-item count, the device's *nitems decides). x16: fp16, row stride ldx; y: fp32, row
        stride ldy (a view's data_ptr: a column offset into a wider row)"""
        fmt, R, K, wb = W[:4]   # (a row-scale study tensor carries a fifth entry, its scales: its format is refused below)
        if K % 256: raise ValueError(f'q3m mm: K {K} is not a multiple of 256')
        if (work is None) == (n is None): raise ValueError('q3m mm: n (dense) or work (experts), one of them')
        if work is None:
            items, nitems, bound, xs, yd = NULL, NULL, -(-int(n) // MM_M), NULL, NULL
            ntok = int(n)
        else:
            items, nitems, bound, xs, yd = work
            ntok = 0
        grid = (int(bound), -(-R // MM_N))
        if fmt == 'Q4_K':
            self.mod.q3p_mm_q4k(grid, (256,), wb, I(R), I(K), x16, I(ldx), xs, y, I(ldy), yd, items, nitems, I(ntok))
        elif fmt == 'Q6_K':
            rtot = wb.numel() // (K // 256 * BLOCK_BYTES['Q6_K'])
            self.mod.q3p_mm_q6k(grid, (256,), wb, I(R), I(rtot), I(K), x16, I(ldx), xs, y, I(ldy), yd, items, nitems, I(ntok))
        elif fmt in IQ_KERNEL and self.module == IQ_MODULE:
            if fmt in IQ_REC:                                # (the record layout: its planes are placed by the buffer's rows)
                rtot, rem = divmod(wb.numel() - IQ_PAD, K // 256 * IQ_REC[fmt][0])
                if rem or rtot < R: raise ValueError(f'q3m mm: {fmt} buffer of {wb.numel()} B is not iq_records of rows x {K}')
            else: rtot = 0
            getattr(self.mod, IQ_KERNEL[fmt])(grid, (256,), wb, I(R), I(rtot), I(K), x16, I(ldx), xs, y, I(ldy), yd, items, nitems, I(ntok))
        else:
            raise ValueError(f'q3m mm: {fmt} on {self.module} ({", ".join(formats(self.module))})')

    # -- the glue ----------------------------------------------------------------------------------------------------------
    def emb(self, W, rb, vocab, toks, x, n, N):
        self.mod.q3p_emb((n,), (256,), W, I(rb), I(vocab), toks, x, I(N))

    def add_rms16(self, x, nw, eps, h16, n, N, d=None, w=None, S=0, h=None):
        if N % 256 or N > 2048: raise ValueError(f'q3m add_rms16: N {N} (a multiple of 256, <= 2048)')
        self.mod.q3p_add_rms16((n,), (256,), x, NULL if d is None else d, NULL if w is None else w, I(S), nw, F(eps),
                               NULL if h is None else h, h16, I(N))

    def router(self, h, Wr, E, K, logits, n):
        if K % 128: raise ValueError(f'q3m router: K {K} (a multiple of 128)')
        self.mod.q3p_router((n,), (256,), h, Wr, I(E), I(K), logits)

    def topk(self, logits, E, k, floor_, ids, wts, err, n):
        if E > 128 or k > 32: raise ValueError(f'q3m topk: E {E} (<= 128), k {k} (<= 32)')
        self.mod.q3p_topk((n,), (32,), logits, I(E), I(k), F(floor_), ids, wts, err)

    def group(self, ids, npairs, k, E, xs, yd, items, nitems, err):
        if E > 1024: raise ValueError(f'q3m group: E {E} (<= 1024)')
        self.mod.q3p_group((1,), (1024,), ids, I(npairs), I(k), I(E), xs, yd, items, nitems, err)

    def swiglu(self, g, u, y16, n):
        self.mod.q3p_swiglu((min(-(-n // 256), 4096),), (256,), g, u, y16, I(n))

    def _span(self, what, cache, pos0, n, nkv, rows):
        """the host check before a position-indexed launch (neither kernel clamps): positions pos0 .. pos0 + n - 1 must be
        among the cache's first `rows` positions (2 * nkv * 128 fp16 per row; None: every row of the allocation; an
        engine's cache has padding rows past its max_ctx, so its prefill passes max_ctx), else k3.ContextBoundsError
        and nothing is launched"""
        if n < 1 or nkv < 1: raise ValueError(f'q3m {what}: n {n}, nkv {nkv}')
        alloc = cache.numel() // (2 * nkv * 128)
        if rows is not None and not 0 <= int(rows) <= alloc: raise ValueError(f'q3m {what}: rows {rows} of a cache of {alloc}')
        k3.check_span(pos0, n, alloc if rows is None else int(rows), what=f'q3m {what}')

    def qk_rope(self, qkv, ldq, nh, nkv, par, pos0, q16, cache, n, rows=None):
        self._span('qk_rope', cache, pos0, n, nkv, rows)
        if ldq < (nh + 2 * nkv) * 128 or qkv.numel() < (n - 1) * ldq + (nh + 2 * nkv) * 128 or q16.numel() < n * nh * 128:
            raise ValueError(f'q3m qk_rope: {n} tokens need qkv rows of {(nh + 2 * nkv) * 128} (stride {ldq}) and {n * nh * 128} q16')
        self.mod.q3p_qk_rope((n, nh + nkv), (128,), qkv, I(ldq), I(nh), I(nkv), par, I(pos0), q16, cache)

    def attn_blocks(self, pos0, n):
        """the 16-position blocks q3p_attn runs for a chunk at positions [pos0, pos0 + n): (first block, count)"""
        b0 = pos0 // AT_T
        return b0, (pos0 + n - 1) // AT_T - b0 + 1

    def attn(self, q16, cache, pos0, n, nkv, scale, out16, i0=0, i1=None, rows=None):
        """blocks [i0, i1) of the chunk's attn_blocks (one launch; the caller splits long chunks by blocks)"""
        self._span('attn', cache, pos0, n, nkv, rows)
        b0, nb = self.attn_blocks(pos0, n)
        i1 = nb if i1 is None else i1
        if not 0 <= i0 < i1 <= nb: raise ValueError(f'q3m attn: blocks [{i0}, {i1}) of {nb}')
        if min(q16.numel(), out16.numel()) < n * nkv * 8 * 128:
            raise ValueError(f'q3m attn: {n} tokens need {n * nkv * 8 * 128} values of q16 and out16')
        import math
        self.mod.q3p_attn((i1 - i0, nkv), (256,), q16, cache, I(pos0), I(n), I(b0 + i0), I(nkv),
                          F(scale * math.log2(math.e)), out16)
