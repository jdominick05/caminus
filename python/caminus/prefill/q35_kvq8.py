"""Checked internal Q8 prefill launches; not a model cache-format switch.

The caller supplies transformed Q/K/V and every buffer. Cache rows contain K
then V, with 32 values per 34-byte Q8 block. No rotation or allocation occurs
here. Append error bits are sticky: a refused block may leave other blocks
written, so the caller must check the error word before consuming any result.
Device numerical parity is a separate GPU gate.
"""
import ctypes
import math
from numbers import Integral, Real

import torch

from ..device.bufcheck import LaunchCheck
from ..device.hip import Module
from ..engine.k3 import check_span

I, F = ctypes.c_int, ctypes.c_float
INT_MAX = 2 ** 31 - 1
_check = LaunchCheck('q35_kvq8', align=4)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= INT_MAX:
        raise ValueError(f'q35_kvq8: {name} must be an integer in [{minimum}, {INT_MAX}]')
    return int(value)


def _cache(cache, M, rows):
    row_bytes = 2 * (M // 32) * 34
    if not isinstance(cache, torch.Tensor) or cache.dtype != torch.uint8 or cache.numel() % row_bytes:
        raise ValueError(f'q35_kvq8: cache must be uint8 whole rows of {row_bytes} bytes')
    physical = _integer(cache.numel() // row_bytes, 'physical rows', 1)
    rows = _integer(rows, 'rows', 1)
    if rows > physical:
        raise ValueError('q35_kvq8: rows exceed the cache view physical rows')
    return rows, row_bytes


class Ops:
    """Load the shared object once, before capture, on an explicitly indexed GPU.

    Registered host-mapped buffers are admitted by LaunchCheck, including err.
    This class queues work only; error inspection must wait for device completion
    (including mapped host words) before treating an append as successful.
    """
    def __init__(self, dev):
        if not isinstance(dev, (str, torch.device)):
            raise ValueError('q35_kvq8: device must be an explicitly indexed CUDA device')
        self.dev = torch.device(dev)
        if self.dev.type != 'cuda' or self.dev.index is None:
            raise ValueError('q35_kvq8: device must be an explicitly indexed CUDA device')
        with torch.cuda.device(self.dev):
            self.mod = Module('q35_prefill_kvq8')

    def append(self, k, ldk, v, ldv, cache, M, pos0, n, rows, err):
        """Append n fp32 rows with element strides; cache may include a spec tail."""
        M = _integer(M, 'M', 32)
        if M % 32:
            raise ValueError('q35_kvq8: M must be divisible by 32')
        ldk, ldv = _integer(ldk, 'ldk', M), _integer(ldv, 'ldv', M)
        pos0, n = _integer(pos0, 'pos0'), _integer(n, 'n', 1)
        rows, rb = _cache(cache, M, rows)
        check_span(pos0, n, rows, what='Q8 prefill append')
        _check(self.dev, 'q35p_kv_append_q8',
               dict(k=(k, torch.float32, (n - 1) * ldk + M),
                    v=(v, torch.float32, (n - 1) * ldv + M),
                    cache=(cache, torch.uint8, rows * rb, 2), err=(err, torch.int32, 1)), ('cache', 'err'))
        with torch.cuda.device(self.dev):
            self.mod.q35p_kv_append_q8((n,), (256,), k, I(ldk), v, I(ldv), cache, I(M), I(pos0), I(rows), err)

    def attn_wm(self, q, cache, pos0, n, heads, nkv, D, scale, out, rows, b0=None, b1=None, force_fp32=False):
        """Attend in absolute eight-row blocks; Q/out views cover the whole chunk."""
        pos0, n = _integer(pos0, 'pos0'), _integer(n, 'n', 1)
        heads, nkv, D = _integer(heads, 'heads', 1), _integer(nkv, 'nkv', 1), _integer(D, 'D', 1)
        if D != 256 or heads % nkv or not 1 <= heads // nkv <= 8:
            raise ValueError('q35_kvq8: attention requires D=256 and 1..8 query heads per KV head')
        M = _integer(nkv * D, 'KV width', 32)
        rows, rb = _cache(cache, M, rows)
        check_span(pos0, n, rows, what='Q8 prefill attention')
        B0, B1 = pos0 // 8, (pos0 + n + 7) // 8
        b0 = B0 if b0 is None else _integer(b0, 'b0')
        b1 = B1 if b1 is None else _integer(b1, 'b1', 1)
        if not B0 <= b0 < b1 <= B1 or b1 * 8 > INT_MAX or ((pos0 + n + 31) // 32) * 32 > INT_MAX:
            raise ValueError('q35_kvq8: invalid or overflowing attention tile range')
        if isinstance(scale, bool) or not isinstance(scale, Real):
            raise ValueError('q35_kvq8: scale must be finite')
        try:
            scale_log2 = F(float(scale) * 1.4426950408889634)
        except (ValueError, OverflowError) as e:
            raise ValueError('q35_kvq8: scale must be finite fp32') from e
        if not math.isfinite(scale_log2.value):
            raise ValueError('q35_kvq8: scale_log2 must be finite fp32')
        if type(force_fp32) is not bool:
            raise ValueError('q35_kvq8: force_fp32 must be bool')
        count = n * heads * D
        _check(self.dev, 'q35p_attn_wm_q8',
               dict(q=(q, torch.float32, count, 16), cache=(cache, torch.uint8, rows * rb, 2),
                    out=(out, torch.float32, count)), ('out',))
        with torch.cuda.device(self.dev):
            self.mod.q35p_attn_wm_q8((b1 - b0, nkv), (256,), q, cache, I(pos0), I(n), I(b0), I(heads), I(nkv),
                                    scale_log2, out, I(force_fp32))
