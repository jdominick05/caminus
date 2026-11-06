"""F.linear(x, w) for bf16 x [..., K], w [N, K] on our WMMA GEMM (caminus.gemm): forward x . w^T and dgrad dy . w
(through a cached w^T for frozen weights) are bitwise equal to rocBLAS's bf16 results on the S1 step's shapes except
K >= 7680 dgrads, where rocBLAS switches to another summation order; wgrad stays on torch (same call F.linear's
backward makes). Shapes the kernel does not tile fall back to F.linear."""
import torch
import torch.nn.functional as F
from .gemm import gemm_tn, supported

STATS = {'ours': 0, 'fallback': 0}


_TABLE = None


def _table():
    """build/gemm_table.json (tools/bench_gemm.py): per measured shape, our best tile where it beats rocBLAS, else rocBLAS"""
    global _TABLE
    if _TABLE is None:
        import json, os
        from .kernels import kernel_dir
        p = os.path.join(kernel_dir()[0], 'gemm_table.json')   # (a checkout's build/)
        _TABLE = {(r['M'], r['N'], r['K']): r['use'] for r in json.load(open(p))['rows']} if os.path.exists(p) else {}
        _TABLE.update({k: v for k, v in _nearest_index(_TABLE).items()})
    return _TABLE


def _nearest_index(table):
    """(None, N, K) -> sorted [(M, use)] so shapes with an unmeasured row count (slices) take the nearest measured M"""
    idx = {}
    for (M, N, K), use in table.items(): idx.setdefault((None, N, K), []).append((M, use))
    return {k: sorted(v) for k, v in idx.items()}


def tile_for(M, N, K):
    """our tile for this shape, or None for rocBLAS: the measured table's choice (tools/bench_gemm.py), for an unmeasured
    row count the nearest measured one with the same N and K; shapes the table lacks stay on rocBLAS"""
    tab = _table()
    use = tab.get((M, N, K))
    if use is None and (None, N, K) in tab:   # nearest measured row count for this (N, K), by ratio
        import math
        use = min(tab[(None, N, K)], key=lambda mu: abs(math.log(mu[0] / M)))[1]
    if use is None or use == 'rocblas': return None   # unmeasured shapes stay on rocBLAS
    t = tuple(int(v) for v in use.split('x'))
    return t if supported(M, N, K, t) else None


def _mm(a, b):   # a [M, K] . b [N, K]^T on our kernel, or None if it does not tile or rocBLAS is faster
    M, K = a.shape; N = b.shape[0]
    t = tile_for(M, N, K)
    if t is None or a.stride(1) != 1 or b.stride(1) != 1 or a.stride(0) % 8 or b.stride(0) % 8: return None
    return gemm_tn(a, b, tile=t)


class Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, wt):
        K = x.shape[-1]; N = w.shape[0]
        x2 = x.reshape(-1, K)
        y = _mm(x2, w)
        if y is None: STATS['fallback'] += 1; y = F.linear(x2, w)
        else: STATS['ours'] += 1
        ctx.save_for_backward(x2 if w.requires_grad else None, w, wt)
        ctx.xshape = x.shape
        return y.view(*x.shape[:-1], N)

    @staticmethod
    def backward(ctx, gy):
        x2, w, wt = ctx.saved_tensors
        g2 = gy.reshape(-1, w.shape[0])
        if g2.stride(1) != 1 or g2.stride(0) % 8: g2 = g2.contiguous()
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = _mm(g2, wt if wt is not None else w.t().contiguous())
            if dx is None: STATS['fallback'] += 1; dx = g2 @ w
            else: STATS['ours'] += 1
            dx = dx.view(ctx.xshape)
        if ctx.needs_input_grad[1]: dw = g2.t() @ x2
        return dx, dw, None


def linear(x, w, wt=None):
    """drop-in for F.linear(x, w) (bf16, no bias); wt: a cached contiguous w^T for frozen w"""
    assert x.dtype == w.dtype == torch.bfloat16
    return Linear.apply(x, w, wt)
