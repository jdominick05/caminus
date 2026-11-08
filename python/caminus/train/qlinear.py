"""The integer W4A4 linear: the forward on the iu4 GEMM from the activation's codes (caminus.train.actq.actq_codes) and the
weight's codes, either per-output-channel (D1: s_x * s_w * (Q_x . Q_w), exact in any order) or g128 (exact int32 dot
per 128-group, groups summed in fp32 in a fixed order, then s_x: deterministic across our kernels); the backward is the
fake-quant proxy's STE: d xq = dy . W_deq and d W_deq = dy^T . xq with the bf16 dequantized operands (W_deq carries the
weight quantizer's own STE outside this function). Shapes the iu4 kernel does not tile fall back to F.linear(xq, W_deq)
(proxy numerics), counted in STATS."""
import torch
import torch.nn.functional as F
from ..device.gemm import gemm_tn_iu4, gemm_tn_iu4g, gemm_tn_seg, pack_int4

STATS = {'iu4': 0, 'fallback': 0}


def iu4_tile(M, N, K):
    """64-wide tiles: 145-167 TOPS on the S1 shapes vs 102-124 for 128 x 128 (tests/device/test_gemm_iu4.py --speed, XTX)"""
    if K % 128: return None
    for t in ((64, 128), (128, 64), (128, 128)):
        if M % t[0] == 0 and N % t[1] == 0: return t
    return None


def weight_codes(w_rot, scale):
    """per-channel codes of a rotated weight [N, K] for scales [N, 1]: (packed codes, fp32 scales [N], bf16 dequant)"""
    with torch.no_grad():
        q = torch.clamp(torch.round(w_rot.float() / scale), -8, 7)
        return pack_int4(q.to(torch.int8)), scale.reshape(-1).float().contiguous(), (q * scale).to(torch.bfloat16)


def weight_codes_g128(w_rot, b=128, bits=4):
    """qat.fq_weight's g128 quantizer (no LSQ) as codes: (packed codes, fp32 group scales [N, K / b], bf16 dequant equal to
    fq_weight's output)"""
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        o, n = w_rot.shape; wg = w_rot.float().reshape(o, n // b, b)
        s = (wg.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-8)
        q = torch.clamp(torch.round(wg / s), -qmax - 1, qmax)
        return (pack_int4(q.reshape(o, n).to(torch.int8)), s.reshape(o, n // b).contiguous(),
                (q * s).reshape(o, n).to(torch.bfloat16))


def iu4g_tile(M, N, K):
    if K % 128: return None
    for t in ((64, 128), (128, 64)):
        if M % t[0] == 0 and N % t[1] == 0: return t
    return None


class CodesLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xq, wdeq, cx, sx, cw, sw):
        K = xq.shape[-1]; N = wdeq.shape[0]
        xq2 = xq.reshape(-1, K)
        grouped = sw.dim() == 2   # per-(column, 128-group) scales: the exact per-group form
        t = (iu4g_tile if grouped else iu4_tile)(xq2.shape[0], N, K)
        if t is None:
            STATS['fallback'] += 1; y = F.linear(xq2, wdeq)
        elif grouped:
            STATS['iu4'] += 1; y = gemm_tn_iu4g(cx, cw, sw, rs=sx, tile=t)
        else:
            STATS['iu4'] += 1; y = gemm_tn_iu4(cx, cw, rs=sx, cs=sw, tile=t)
        ctx.save_for_backward(xq2 if wdeq.requires_grad else None, wdeq)
        ctx.xshape = xq.shape
        return y.view(*xq.shape[:-1], N)

    @staticmethod
    def backward(ctx, gy):
        xq2, wdeq = ctx.saved_tensors
        g2 = gy.reshape(-1, wdeq.shape[0])
        dxq = (g2 @ wdeq).view(ctx.xshape) if ctx.needs_input_grad[0] else None
        dw = g2.t() @ xq2 if ctx.needs_input_grad[1] else None
        return dxq, dw, None, None, None, None


def codes_linear(xq, wdeq, cx, sx, cw, sw):
    """y = sx * sw * (codes_x . codes_w) forward (= F.linear(xq, wdeq) up to the operands' bf16 rounding), STE backward"""
    return CodesLinear.apply(xq, wdeq, cx, sx, cw, sw)


class CodesLinearSplit(torch.autograd.Function):
    """CodesLinear whose output is returned as its column pieces (sizes) and whose dgrad multiplies the pieces' gradients
    straight from their own memory (caminus.device.gemm.gemm_tn_seg against the cached w^T): no cat of the gradients. wt is
    w_deq^T [K, N] contiguous (a frozen weight's cache) or None."""
    @staticmethod
    def forward(ctx, xq, wdeq, cx, sx, cw, sw, wt, sizes):
        y = CodesLinear.forward(ctx, xq, wdeq, cx, sx, cw, sw)
        ctx.wt, ctx.sizes = wt, sizes
        return tuple(y.split(sizes, -1))

    @staticmethod
    def backward(ctx, *gs):
        xq2, wdeq = ctx.saved_tensors
        segs = []
        for g, n in zip(gs, ctx.sizes):
            if g is None: g = torch.zeros(*ctx.xshape[:-1], n, device=wdeq.device, dtype=torch.bfloat16)
            g = g.reshape(-1, n)
            if g.stride(1) != 1 or g.stride(0) % 8 or g.data_ptr() % 16: g = g.contiguous()
            segs.append(g)
        dxq = dw = None
        if ctx.needs_input_grad[0]:
            wt = ctx.wt if ctx.wt is not None else wdeq.t().contiguous()
            dxq = gemm_tn_seg(segs, wt).view(ctx.xshape)
        if ctx.needs_input_grad[1]:
            dw = torch.cat([g.t() @ xq2 for g in segs])
        return dxq, dw, None, None, None, None, None, None


def codes_linear_split(xq, wdeq, cx, sx, cw, sw, wt, sizes):
    return CodesLinearSplit.apply(xq, wdeq, cx, sx, cw, sw, wt, tuple(sizes))
