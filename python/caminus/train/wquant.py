"""Per-output-channel weight scale with the MSE-optimal clip ratio, one fused pass per row (kernels/wquant.hip)."""
import ctypes
import torch
from ..device.hip import Module

_mod, _rat = {}, {}
RATIOS = [0.5 + 0.02 * i for i in range(26)]   # training/tools/s1_patches.py channel_scale's candidates


def _m():
    dev = torch.cuda.current_device()
    if dev not in _mod: _mod[dev] = Module('wquant')
    return _mod[dev]


def channel_scale(w, bits=4, ratios=None):
    """w [N, K] (bf16 or fp32) -> scales [N, 1] fp32"""
    ratios = ratios or RATIOS
    assert len(ratios) <= 32
    key = (w.device, tuple(ratios))
    if key not in _rat: _rat[key] = torch.tensor(ratios, device=w.device, dtype=torch.float32)
    w2 = w.detach().contiguous(); N, K = w2.shape
    s = torch.empty(N, device=w.device)
    sfx = 'bf16' if w2.dtype == torch.bfloat16 else 'f32'
    if w2.dtype not in (torch.bfloat16, torch.float32): w2 = w2.float(); sfx = 'f32'
    getattr(_m(), f'channel_scale_{sfx}')((N,), (256,), w2, _rat[key], s, ctypes.c_int(K), ctypes.c_int(len(ratios)),
                                          ctypes.c_int(2 ** (bits - 1) - 1))
    return s[:, None]
