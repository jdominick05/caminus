"""Internal staged own-Q8 KV append; not a model cache-format switch.

The caller supplies every buffer and any required transforms. Rows contain K
blocks then V blocks, with 32 values per 34-byte block; no rotation is performed.
The device encoder's finite-input domain follows caminus.ref_kv. Device byte
parity, including subnormal/FTZ behavior, remains unverified.

append() queues work only after host layout/span/alias checks. The caller zeros
err before a sequence and calls check(err) before consuming its output. Error
bits are sticky: 1 span, 2 numerical domain, 4 layout. A failed numerical block
is untouched, but other blocks may already be written: ANY error invalidates the
whole result. This launcher allocates no buffers and provides no decoder.
"""
import ctypes
from numbers import Integral

import torch

from .bufcheck import LaunchCheck
from .hip import Module

INT_MAX = 2 ** 31 - 1
ERR_SPAN, ERR_NUMERICAL, ERR_LAYOUT = 1, 2, 4
_check = LaunchCheck('kv_codec', align=4)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= INT_MAX:
        raise ValueError(f'kv_codec: {name} must be an integer in [{minimum}, {INT_MAX}]')
    return int(value)


class Q8KVCodec:
    """The standalone kv_codec module on an explicitly indexed CUDA device.

    An injected module supports CPU launch recorders. Otherwise the HIP module
    is loaded lazily, on the first fully validated append, under this device's
    context and current stream. Constructing the launcher never initializes HIP.
    """
    def __init__(self, device, module=None):
        if not isinstance(device, (str, torch.device)):
            raise ValueError('kv_codec: device must be an explicitly indexed CUDA device')
        self.dev = torch.device(device)
        if self.dev.type != 'cuda' or self.dev.index is None:
            raise ValueError('kv_codec: device must be an explicitly indexed CUDA device')
        self.module = module

    def append(self, k, ldk, v, ldv, M, cache, pos0, T, err, rows=None):
        """Append T fp32 rows of M values, with element strides ldk / ldv.

        All tensors must be contiguous on this device, or registered host-mapped
        buffers. Their own views must cover every touched element; K and V may
        alias each other, while cache and err may not overlap another operand.
        cache is uint8 with an exact whole number of physical rows. rows limits
        the exposed prefix, defaulting to those physical rows. Padding belongs
        to the caller. err is an int32 buffer whose first word is used, not reset.
        """
        M = _integer(M, 'M', 32)
        if M % 32:
            raise ValueError('kv_codec: M must be divisible by 32')
        T = _integer(T, 'T', 1)
        ldk = _integer(ldk, 'ldk', M)
        ldv = _integer(ldv, 'ldv', M)
        pos0 = _integer(pos0, 'pos0')
        row_bytes = 2 * (M // 32) * 34
        if not isinstance(cache, torch.Tensor) or cache.dtype != torch.uint8 or cache.numel() % row_bytes:
            raise ValueError(f'kv_codec: cache must be uint8 whole rows of {row_bytes} bytes')
        physical_rows = _integer(cache.numel() // row_bytes, 'physical rows', 1)
        rows = physical_rows if rows is None else _integer(rows, 'rows', 1)
        if rows > physical_rows:
            raise ValueError(f'kv_codec: rows {rows} exceed the cache view of {physical_rows} physical rows')
        from ..engine.k3 import check_span
        check_span(pos0, T, rows, what='own Q8 KV append')
        nk, nv = (T - 1) * ldk + M, (T - 1) * ldv + M
        _check(self.dev, 'own_kv_q8_append',
               dict(k=(k, torch.float32, nk), v=(v, torch.float32, nv),
                    cache=(cache, torch.uint8, rows * row_bytes, 2), err=(err, torch.int32, 1)),
               ('cache', 'err'))
        with torch.cuda.device(self.dev):
            if self.module is None:
                self.module = Module('kv_codec')
            self.module.own_kv_q8_append((T,), (256,), k, ctypes.c_int(ldk), v, ctypes.c_int(ldv), ctypes.c_int(M),
                                       cache, ctypes.c_int(pos0), ctypes.c_int(rows), err)

    def check(self, err):
        """Wait for this device, then read the first error word and refuse every nonzero result.

        Does not clear err or repair partially written rows. After a refusal the
        caller must discard/reinitialize the affected cache before using it.
        Synchronization also covers mapped host words and appends on another
        stream of this device. Call outside graph capture.
        """
        _check(self.dev, 'own_kv_q8_check', dict(err=(err, torch.int32, 1)), ())
        torch.cuda.synchronize(self.dev)
        e = int(err.view(-1)[0].item())
        if e:
            reasons = [name for bit, name in ((ERR_SPAN, 'span'), (ERR_NUMERICAL, 'numerical domain'), (ERR_LAYOUT, 'layout'))
                       if e & bit]
            if e & ~(ERR_SPAN | ERR_NUMERICAL | ERR_LAYOUT):
                reasons.append('unknown error bits')
            raise RuntimeError(f'kv_codec: whole append result is unusable (err {e:#x}: {", ".join(reasons)})')
