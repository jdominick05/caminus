"""Fused activation fake-quant: actq(x) == qat.fq_act(qat.rotate(x, 128), bits, clip) under bf16 autocast.
Forward and backward each run as one kernel (kernels/actq.hip); y is always bf16, dx has x's dtype."""
import ctypes
import torch
from ..device.hip import Module

_mod = {}


def _m():
    dev = torch.cuda.current_device()
    if dev not in _mod: _mod[dev] = Module('actq')
    return _mod[dev]


def _sfx(x):
    if x.dtype == torch.bfloat16: return 'bf16'
    if x.dtype == torch.float32: return 'f32'
    raise TypeError(f'actq: unsupported dtype {x.dtype}')


class ActQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits, clip):
        K = x.shape[-1]
        assert K % 128 == 0 and K <= 8192, K
        x2 = x.reshape(-1, K).contiguous(); M = x2.shape[0]
        y = torch.empty(M, K, device=x.device, dtype=torch.bfloat16)
        if M:
            getattr(_m(), f'actq_fwd_{_sfx(x2)}')((M,), (256,), x2, y, ctypes.c_void_p(None), ctypes.c_int(K),
                                                  ctypes.c_float(clip), ctypes.c_int(2 ** (bits - 1) - 1))
        ctx.save_for_backward(x2); ctx.bits, ctx.clip, ctx.shape = bits, clip, x.shape
        return y.reshape(*x.shape[:-1], K)

    @staticmethod
    def backward(ctx, gy):
        (x2,) = ctx.saved_tensors; M, K = x2.shape
        g2 = gy.reshape(-1, K).to(torch.bfloat16).contiguous()
        dx = torch.empty_like(x2)
        if M:
            getattr(_m(), f'actq_bwd_{_sfx(x2)}')((M,), (256,), x2, g2, dx, ctypes.c_int(K), ctypes.c_float(ctx.clip),
                                                  ctypes.c_int(2 ** (ctx.bits - 1) - 1))
        return dx.reshape(ctx.shape), None, None


def actq(x, bits=4, clip=1.0):
    return ActQ.apply(x, bits, clip)


def scales(x, bits=4, clip=1.0):
    """per-row scales s, for tests"""
    K = x.shape[-1]; x2 = x.reshape(-1, K).contiguous(); M = x2.shape[0]
    y = torch.empty(M, K, device=x.device, dtype=torch.bfloat16); s = torch.empty(M, device=x.device)
    getattr(_m(), f'actq_fwd_{_sfx(x2)}')((M,), (256,), x2, y, s, ctypes.c_int(K), ctypes.c_float(clip),
                                          ctypes.c_int(2 ** (bits - 1) - 1))
    return s


def codes(x, bits=4, clip=1.0, dequant=False):
    """the forward's integer codes, packed two per byte along K (caminus.device.gemm.pack_int4 layout) [M, K / 2] uint8, and the
    row scales [M] fp32 (y = codes * s); with dequant=True also the bf16 values actq() returns"""
    assert bits <= 4
    K = x.shape[-1]; x2 = x.reshape(-1, K).contiguous(); M = x2.shape[0]
    c = torch.empty(M, K // 2, device=x.device, dtype=torch.uint8); s = torch.empty(M, device=x.device)
    y = torch.empty(M, K, device=x.device, dtype=torch.bfloat16) if dequant else None
    getattr(_m(), f'actq_fwd_codes_{_sfx(x2)}')((M,), (256,), x2, y if y is not None else ctypes.c_void_p(None), s, c,
                                                ctypes.c_int(K), ctypes.c_float(clip), ctypes.c_int(2 ** (bits - 1) - 1))
    return (c, s, y) if dequant else (c, s)


class ActQCodes(torch.autograd.Function):
    """actq() that also returns the packed int4 codes and row scales from the same pass (non-differentiable outputs);
    backward as ActQ"""
    @staticmethod
    def forward(ctx, x, bits, clip):
        K = x.shape[-1]
        assert K % 128 == 0 and K <= 8192 and bits <= 4, K
        x2 = x.reshape(-1, K).contiguous(); M = x2.shape[0]
        y = torch.empty(M, K, device=x.device, dtype=torch.bfloat16)
        c = torch.empty(M, K // 2, device=x.device, dtype=torch.uint8); s = torch.empty(M, device=x.device)
        if M:
            getattr(_m(), f'actq_fwd_codes_{_sfx(x2)}')((M,), (256,), x2, y, s, c, ctypes.c_int(K), ctypes.c_float(clip),
                                                        ctypes.c_int(2 ** (bits - 1) - 1))
        ctx.save_for_backward(x2); ctx.bits, ctx.clip, ctx.shape = bits, clip, x.shape
        ctx.mark_non_differentiable(c, s)
        return y.reshape(*x.shape[:-1], K), c, s

    @staticmethod
    def backward(ctx, gy, gc, gs):
        return ActQ.backward(ctx, gy)


def actq_codes(x, bits=4, clip=1.0):
    """(bf16 fake-quantized x with autograd, packed codes [M, K / 2], row scales [M])"""
    return ActQCodes.apply(x, bits, clip)


class SiluMulActQ(torch.autograd.Function):
    """the MLP's act = bf16(bf16(silu(g)) * u) for gu = [g | u] fused with the down projection's activation quant: returns
    (bf16 fake-quantized act, packed int4 codes, row scales) without materializing act; backward recomputes act and
    returns d gu = [dg | du] (torch's MulBackward0 and silu_backward roundings). gu may be a row-strided view."""
    @staticmethod
    def forward(ctx, gu, bits, clip):
        M, F2 = gu.shape; F = F2 // 2
        assert gu.dtype == torch.bfloat16 and gu.stride(1) == 1 and F % 128 == 0 and F <= 8192 and bits <= 4
        y = torch.empty(M, F, device=gu.device, dtype=torch.bfloat16)
        c = torch.empty(M, F // 2, device=gu.device, dtype=torch.uint8); s = torch.empty(M, device=gu.device)
        if M:
            _m().silu_mul_actq_fwd((M,), (256,), gu, ctypes.c_int(gu.stride(0)), ctypes.c_int(F), y, s, c,
                                   ctypes.c_float(clip), ctypes.c_int(2 ** (bits - 1) - 1))
        ctx.save_for_backward(gu); ctx.bits, ctx.clip = bits, clip
        ctx.mark_non_differentiable(c, s)
        return y, c, s

    @staticmethod
    def backward(ctx, gy, gc, gs):
        (gu,) = ctx.saved_tensors; M, F2 = gu.shape; F = F2 // 2
        g2 = gy.to(torch.bfloat16).contiguous()
        dgu = torch.empty(M, F2, device=gu.device, dtype=torch.bfloat16)
        if M:
            _m().silu_mul_actq_bwd((M,), (256,), gu, ctypes.c_int(gu.stride(0)), ctypes.c_int(F), g2, dgu,
                                   ctypes.c_float(ctx.clip), ctypes.c_int(2 ** (ctx.bits - 1) - 1))
        return dgu, None, None


def silu_mul_actq(gu, bits=4, clip=1.0):
    """(xq, codes, scales) = actq_codes(silu(g) * u) for gu [M, 2F] bf16, fused (see SiluMulActQ)"""
    return SiluMulActQ.apply(gu, bits, clip)


class BlockEntry(torch.autograd.Function):
    """h = RMSNorm(x) (never materialized) -> h's activation quant (bf16 values, packed codes, row scales) and nab gate
    projections bf16(bf16(h) . wab^T) in one row pass; backward: dx from dxq and the projections' grads (frozen norm and
    projection weights). wab: bf16 [nab, d] (a_proj rows then b_proj rows), or None."""
    @staticmethod
    def forward(ctx, x, nw, wab, eps, bits, clip):
        d = x.shape[-1]; x2 = x.reshape(-1, d).contiguous(); M = x2.shape[0]
        assert x2.dtype == torch.float32 and d % 128 == 0 and d <= 1024 and bits <= 4
        nab = 0 if wab is None else wab.shape[0]
        assert nab <= 16 and nab % 2 == 0
        y = torch.empty(M, d, device=x.device, dtype=torch.bfloat16)
        c = torch.empty(M, d // 2, device=x.device, dtype=torch.uint8); s = torch.empty(M, device=x.device)
        ab = torch.empty(M, nab, device=x.device, dtype=torch.bfloat16); rstd = torch.empty(M, device=x.device)
        nul = ctypes.c_void_p(None)
        if M:
            _m().block_entry_fwd((M,), (256,), x2, nw, wab if nab else nul, ctypes.c_int(nab), ctypes.c_float(eps), y, s, c,
                                 ab if nab else nul, rstd, ctypes.c_int(d), ctypes.c_float(clip), ctypes.c_int(2 ** (bits - 1) - 1))
        ctx.save_for_backward(x2, nw, rstd, wab if nab else None)
        ctx.bits, ctx.clip, ctx.nab, ctx.shape = bits, clip, nab, x.shape
        ctx.mark_non_differentiable(c, s)
        return y.view(*x.shape), c, s, ab.view(*x.shape[:-1], nab)

    @staticmethod
    def backward(ctx, dy, dc, ds, dab):
        x2, nw, rstd, wab = ctx.saved_tensors; M, d = x2.shape; nab = ctx.nab
        dy2 = dy.reshape(M, d).to(torch.bfloat16).contiguous()
        dab2 = (dab.reshape(M, nab).to(torch.bfloat16).contiguous() if dab is not None
                else torch.zeros(M, nab, device=x2.device, dtype=torch.bfloat16)) if nab else None
        dx = torch.empty_like(x2); nul = ctypes.c_void_p(None)
        if M:
            _m().block_entry_bwd((M,), (256,), x2, nw, rstd, wab if nab else nul, ctypes.c_int(nab), dy2,
                                 dab2 if nab else nul, dx, ctypes.c_int(d), ctypes.c_float(ctx.clip),
                                 ctypes.c_int(2 ** (ctx.bits - 1) - 1))
        return dx.view(ctx.shape), None, None, None, None, None


def block_entry(x, nw, wab, eps, bits=4, clip=1.0):
    """(xq, codes, scales, ab) for h = RMSNorm(x; nw, eps), fused (see BlockEntry)"""
    return BlockEntry.apply(x, nw, wab, eps, bits, clip)
