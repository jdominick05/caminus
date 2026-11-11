"""The two-card exchange's host side without gguf-py (the own-code two-card engine, engine/q35_dense_tp.py, imports
nothing of it): the HIP runtime handle and the exchange's receive slots. engine/flashnext_tp.py re-exports both names."""
import ctypes
import os

import numpy as np
import torch


def _hip():
    return ctypes.CDLL(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libamdhip64.so'))


class Slots:
    """receive slots for the exchanges, in uncached VRAM of one card (a peer's PCIe writes are only seen there):
    n slots of 16 header words (word 0 = the epoch flag) + N floats"""
    def __init__(self, dev, n, N):
        hip = _hip(); self.stride = (16 + N) * 4; size = n * self.stride; p = ctypes.c_void_p()
        hip.hipSetDevice(dev)
        assert hip.hipExtMallocWithFlags(ctypes.byref(p), ctypes.c_size_t(size), ctypes.c_uint(3)) == 0   # Uncached
        assert hip.hipMemset(p, 0, ctypes.c_size_t(size)) == 0 and hip.hipDeviceSynchronize() == 0
        self.base, self.size, self.dev = p.value, size, dev

    def slot(self, i):
        return self.base + i * self.stride

    def flags(self):
        """the epoch words of every slot (host copy, for debugging)"""
        hip = _hip(); buf = (ctypes.c_uint32 * (self.size // 4))()
        hip.hipSetDevice(self.dev); hip.hipMemcpy(buf, ctypes.c_void_p(self.base), ctypes.c_size_t(self.size), 2)
        return np.frombuffer(buf, dtype=np.uint32)[::self.stride // 4]
