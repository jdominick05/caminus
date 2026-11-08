"""Fused glue for the S1 block (kernels/fuse.hip)."""
import ctypes
import torch
from ..device.hip import Module

_mod = {}


def _m():
    dev = torch.cuda.current_device()
    if dev not in _mod: _mod[dev] = Module('fuse')
    return _mod[dev]


class ResidualSlices(torch.autograd.Function):
    """x + m + mlp + scatter(ys -> rows): x fp32 [..., d]; m, mlp bf16 [..., d]; ys bf16 [n, d], the routed rows' slice
    outputs in section-sorted order, pos int32 [rows] = each row's index into ys or -1 (route()); torch's add order"""
    @staticmethod
    def forward(ctx, x, m, mlp, ys, pos):
        d = x.shape[-1]
        assert x.dtype == torch.float32 and m.dtype == mlp.dtype == torch.bfloat16 and d % 4 == 0
        x, m, mlp = x.contiguous(), m.contiguous(), mlp.contiguous()
        has = ys is not None and ys.numel() > 0
        if has: ys = ys.contiguous(); assert ys.dtype == torch.bfloat16
        out = torch.empty_like(x); n4 = x.numel() // 4
        nul = ctypes.c_void_p(None)
        _m().residual_slices_fwd(((n4 + 255) // 256,), (256,), x, m, mlp, ys if has else nul, pos if has else nul, out,
                                 ctypes.c_int(d), ctypes.c_long(n4))
        ctx.save_for_backward(pos if has else None); ctx.nys = ys.shape[0] if has else 0; ctx.has = has
        return out

    @staticmethod
    def backward(ctx, dout):
        (pos,) = ctx.saved_tensors
        dout = dout.contiguous(); d = dout.shape[-1]; n4 = dout.numel() // 4
        dm = torch.empty(dout.shape, device=dout.device, dtype=torch.bfloat16); dmlp = torch.empty_like(dm)
        dys = torch.empty(ctx.nys, d, device=dout.device, dtype=torch.bfloat16) if ctx.has else None
        nul = ctypes.c_void_p(None)
        _m().residual_slices_bwd(((n4 + 255) // 256,), (256,), dout, pos if ctx.has else nul, dm, dmlp,
                                 dys if ctx.has else nul, ctypes.c_int(d), ctypes.c_long(n4))
        return dout, dm, dmlp, dys, None


def residual_slices(x, m, mlp, ys, pos):
    return ResidualSlices.apply(x, m, mlp, ys, pos)
