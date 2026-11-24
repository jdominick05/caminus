"""Launch wrappers of kernels/pf_q3m.hip: the batched prefill's own kernels for qwen3moe (Qwen3-30B-A3B), no ggml-cuda
code. A matrix is the decode engine's tuple (caminus.engine.q3moe_one.Q3MoEOne._w): (format, rows per matrix, K, device
bytes in the decode's layout: Q4_K raw blocks, Q6_K planes); an expert tensor is all its experts' rows in one buffer.

  Ops(dev).mm(W, x16, ldx, y, ldy, n=...)                  dense: y[t] = W x16[t] for t < n (fp32 out, row stride ldy)
  Ops(dev).mm(W, x16, ldx, y, ldy, work=(items, nitems, bound, xs, yd))   experts: the grouped pairs (group())
  Ops(dev).mm_gu(Wg, Wu, x16, ldx, y16, ldy, work)          experts: fp16 silu(gate) up in one launch
  (on pf_q3m both run kernels/pf_q3m_mm.hip's wide tiles, each row bitwise the narrow tiles': wide = False
  runs q3p_mm_q4k / q3p_mm_q6k and gate, up, swiglu)
  emb / add_rms16 / router / topk / group / swiglu / qk_rope / attn: one kernel each (kernels/pf_q3m.hip)
The study object pf_q3m_iq (kernels/pf_q3m_iq.hip, docs/dev/iq-quant-study.md) adds mm for the codebook formats
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
MMW_N = 256                 # (rows per workgroup of the wide tiles, kernels/pf_q3m_mm.hip MMW_N)
WIDE_MODULE = 'pf_q3m_mm'   # the 30B's wide kernels (q3p_mm_q4k_w / q3p_mm_q4k_gu / q3p_mm_q6k_w; kernels/pf_q3m_mm.hip)
AT_T = 16                   # q3mp_attn: tokens per workgroup
BLOCK_BYTES = {'Q4_K': 144, 'Q6_K': 210}
KVQ8_MODULE = 'pf_q3m_kvq8'                                  # the Q8 KV cache's qk_rope / attn (kernels/objects.tsv pf_q3m_kvq8)
# the long-context mode's qk_rope (YaRN: Q3P_YARN, the frequency table and attention factor after eps in par): F16 and Q8
# cache; every other kernel, the Q8 attention included, stays on its own object
YARN_MODULE = {False: 'pf_q3m_yarn', True: 'pf_q3m_kvq8_yarn'}
YARN_PAR = 2 * 128 + 2 + k3.YARN_TABLE_FLOATS                       # floats of a YaRN par block (q_norm | k_norm | theta_scale, eps | table)
IQ_MODULE = 'pf_q3m_iq'                                      # the study object: the shipped kernels + the four below
IQ_KERNEL = {'IQ4_XS': 'q3p_mm_iq4xs', 'IQ3_XXS': 'q3p_mm_iq3xxs', 'IQ3_S': 'q3p_mm_iq3s', 'IQ2_XXS': 'q3p_mm_iq2xxs'}


def formats(module):
    """the weight formats a prefill object's mm takes"""
    return tuple(BLOCK_BYTES) + (tuple(IQ_KERNEL) if module == IQ_MODULE else ())


def items_bound(npairs, n_exp):
    """the most work items q3p_group can write for npairs pairs over n_exp experts"""
    return -(-npairs // MM_M) + n_exp


class Ops:
    def __init__(self, dev, module='pf_q3m', q8=False, yarn=False, wide=None):
        """q8: the KV cache is Q8_0 rows (uint8, kernels/kv_codec_ops.h's format): qk_rope / attn run pf_q3m_kvq8's
        Q8 kernels (every other kernel stays on `module`). yarn: the long-context mode: qk_rope runs YARN_MODULE's build
        (the YaRN table in par: YARN_PAR floats), every other kernel as without it. wide: the Q4_K / Q6_K GEMMs on
        WIDE_MODULE's wide tiles and the experts' gate | up fused (None: with module pf_q3m; False: the narrow tiles
        and gate, up, swiglu, as the two-card prefill runs)"""
        self.dev = torch.device(dev) if not isinstance(dev, torch.device) else dev
        self.module, self.q8, self.yarn = module, bool(q8), bool(yarn)
        if self.yarn and module != 'pf_q3m': raise ValueError(f'q3m Ops: the long-context mode runs with pf_q3m only (not {module})')
        # the wide tiles (two row tiles per wave, bitwise the narrow tiles' rows) and the fused gate | up (+ swiglu); the
        # attributes switch them per launch (benchmarks' A / B) once the object is loaded. A dense GEMM goes wide from
        # wide_min_wg workgroups (half the card's compute units): on the XT the grouped experts and every dense 30B shape
        # are faster wide at 64+ workgroups (a 2048-token chunk; attn_output at 512 tokens, 64), the k / v projections at
        # 512 tokens (16 workgroups) 1.6-1.8x slower (docs/dev/yardsticks.md)
        self.wide = self.gu = (module == 'pf_q3m') if wide is None else bool(wide)
        self.wide_min_wg = torch.cuda.get_device_properties(self.dev).multi_processor_count // 2
        with torch.cuda.device(self.dev):
            self.mod = hip.Module(module)
            self.wmod = hip.Module(WIDE_MODULE) if self.wide else None
            self.kvmod = hip.Module(KVQ8_MODULE) if self.q8 else None
            self.ropemod = hip.Module(YARN_MODULE[self.q8]) if self.yarn else (self.kvmod if self.q8 else self.mod)

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
        if self.wide and fmt in BLOCK_BYTES and (work is not None or int(bound) * -(-R // MMW_N) >= self.wide_min_wg):
            gw = (int(bound), -(-R // MMW_N))
            if fmt == 'Q4_K':
                self.wmod.q3p_mm_q4k_w(gw, (256,), wb, I(R), I(K), x16, I(ldx), xs, y, I(ldy), yd, items, nitems, I(ntok))
            else:
                rtot = wb.numel() // (K // 256 * BLOCK_BYTES['Q6_K'])
                self.wmod.q3p_mm_q6k_w(gw, (256,), wb, I(R), I(rtot), I(K), x16, I(ldx), xs, y, I(ldy), yd, items, nitems, I(ntok))
            return
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

    def mm_gu(self, Wg, Wu, x16, ldx, y16, ldy, work):
        """y16[dst] = fp16(silu(Wg x16[src]) (Wu x16[src])) for the grouped pairs (work from group()): the gate GEMM, the up
        GEMM and swiglu in one launch (kernels/pf_q3m_mm.hip q3p_mm_q4k_gu: the same bits). Wg, Wu: Q4_K expert tensors of
        the same shape"""
        if not self.gu: raise ValueError(f'q3m mm_gu: not on {self.module} (gu off)')
        (fg, R, K, wg), (fu, Ru, Ku, wu) = Wg[:4], Wu[:4]
        if fg != 'Q4_K' or fu != 'Q4_K' or (R, K) != (Ru, Ku) or wg.numel() != wu.numel():
            raise ValueError(f'q3m mm_gu: gate {fg} [{R} x {K}], up {fu} [{Ru} x {Ku}] (two Q4_K tensors of one shape)')
        if K % 256: raise ValueError(f'q3m mm_gu: K {K} is not a multiple of 256')
        if y16.dtype != torch.float16: raise ValueError(f'q3m mm_gu: y16 is {y16.dtype} (fp16)')
        items, nitems, bound, xs, yd = work
        self.wmod.q3p_mm_q4k_gu((int(bound), -(-R // MM_N)), (256,), wg, wu, I(R), I(K), x16, I(ldx), xs, y16, I(ldy), yd, items, nitems)

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
        q8 = getattr(self, 'q8', False)   # (a subclass that sets no q8: the F16 cache)
        want = torch.uint8 if q8 else torch.float16
        if cache.dtype != want: raise ValueError(f'q3m {what}: a {"Q8" if q8 else "F16"} prefill needs a {want} cache, not {cache.dtype}')
        alloc = k3.kv_rows(cache, nkv * 128, q8)   # (the allocation's rows: Q8 rows are 2 nkv * 136 bytes)
        if rows is not None and not 0 <= int(rows) <= alloc: raise ValueError(f'q3m {what}: rows {rows} of a cache of {alloc}')
        k3.check_span(pos0, n, alloc if rows is None else int(rows), what=f'q3m {what}')

    def qk_rope(self, qkv, ldq, nh, nkv, par, pos0, q16, cache, n, rows=None, err=None):
        """err (the Q8 cache only, required there): an int32 word a refused block ORs 2 into (the caller reads it after
        the chunk: caminus.errors.InvalidKVEncoding)"""
        self._span('qk_rope', cache, pos0, n, nkv, rows)
        if ldq < (nh + 2 * nkv) * 128 or qkv.numel() < (n - 1) * ldq + (nh + 2 * nkv) * 128 or q16.numel() < n * nh * 128:
            raise ValueError(f'q3m qk_rope: {n} tokens need qkv rows of {(nh + 2 * nkv) * 128} (stride {ldq}) and {n * nh * 128} q16')
        yarn = getattr(self, 'yarn', False)
        if par.dtype != torch.float32 or par.numel() < (YARN_PAR if yarn else 2 * 128 + 2):   # (the YaRN build reads the table past eps)
            raise ValueError(f'q3m qk_rope: par needs {YARN_PAR if yarn else 2 * 128 + 2} fp32 values'
                             f'{" (q_norm | k_norm | theta_scale, eps | the YaRN table)" if yarn else ""}, not {par.numel()} {par.dtype}')
        rope = getattr(self, 'ropemod', None) or (self.kvmod if getattr(self, 'q8', False) else self.mod)
        if getattr(self, 'q8', False):
            if err is None or err.dtype != torch.int32 or err.numel() < 1: raise ValueError('q3m qk_rope: a Q8 cache needs its int32 error word')
            if not cache.is_contiguous() or cache.data_ptr() % 4: raise ValueError('q3m qk_rope: the Q8 cache must be contiguous and 4-byte aligned')
            rope.q3p_qk_rope_q8((n, nh + nkv), (128,), qkv, I(ldq), I(nh), I(nkv), par, I(pos0), q16, cache, err)
            return
        if err is not None: raise ValueError('q3m qk_rope: err is the Q8 cache\'s word')
        rope.q3p_qk_rope((n, nh + nkv), (128,), qkv, I(ldq), I(nh), I(nkv), par, I(pos0), q16, cache)

    def attn_blocks(self, pos0, n):
        """the 16-position blocks q3mp_attn runs for a chunk at positions [pos0, pos0 + n): (first block, count)"""
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
        if getattr(self, 'q8', False):
            if not cache.is_contiguous() or cache.data_ptr() % 4: raise ValueError('q3m attn: the Q8 cache must be contiguous and 4-byte aligned')
            self.kvmod.q3mp_attn_q8((i1 - i0, nkv), (256,), q16, cache, I(pos0), I(n), I(b0 + i0), I(8 * nkv), I(nkv),
                                    F(scale * math.log2(math.e)), out16)
            return
        self.mod.q3mp_attn((i1 - i0, nkv), (256,), q16, cache, I(pos0), I(n), I(b0 + i0), I(8 * nkv), I(nkv),
                           F(scale * math.log2(math.e)), out16)
