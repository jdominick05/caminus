"""Exactly sized GPU allocations for the attention tests that check a kernel never reads past a cache's last byte:
exact_alloc(nbytes) is its own hipMalloc of exactly nbytes, so a read past the last row's last byte faults instead of
landing in a neighbouring allocation (tests/prefill/test_q35_prefill_attn.py, tests/engine/test_dec_q35m_attn.py)."""
import ctypes

import torch
from caminus.device import hip as _hip


class _DLDevice(ctypes.Structure):
    _fields_ = [('device_type', ctypes.c_int32), ('device_id', ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [('code', ctypes.c_uint8), ('bits', ctypes.c_uint8), ('lanes', ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [('data', ctypes.c_void_p), ('device', _DLDevice), ('ndim', ctypes.c_int32), ('dtype', _DLDataType),
                ('shape', ctypes.POINTER(ctypes.c_int64)), ('strides', ctypes.POINTER(ctypes.c_int64)),
                ('byte_offset', ctypes.c_uint64)]


class _DLManaged(ctypes.Structure):
    pass


_DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManaged))
_DLManaged._fields_ = [('dl_tensor', _DLTensor), ('manager_ctx', ctypes.c_void_p), ('deleter', _DELETER)]
_EXACT = {}     # pointer -> (the DLPack structs, kept alive until torch's deleter runs)


@_DELETER
def _exact_free(m):
    p = m.contents.dl_tensor.data
    _EXACT.pop(p, None)
    _hip.check(_hip.hip().hipFree(ctypes.c_void_p(p)))


def exact_alloc(nbytes):
    """a uint8 device tensor that is its own hipMalloc of exactly nbytes (freed with the tensor), and the (base, size)
    hipMemGetAddressRange reports for its first byte"""
    h = _hip.hip(); p = ctypes.c_void_p(); base = ctypes.c_void_p(); size = ctypes.c_size_t()
    h.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    h.hipFree.argtypes = [ctypes.c_void_p]
    h.hipMemGetAddressRange.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p]
    _hip.check(h.hipMalloc(ctypes.byref(p), ctypes.c_size_t(int(nbytes))))
    _hip.check(h.hipMemGetAddressRange(ctypes.byref(base), ctypes.byref(size), p))
    shape = (ctypes.c_int64 * 1)(int(nbytes))
    m = _DLManaged()
    m.dl_tensor = _DLTensor(p.value, _DLDevice(10, torch.cuda.current_device()), 1, _DLDataType(1, 8, 1),  # kDLROCM, uint8
                            shape, None, 0)
    m.manager_ctx = None; m.deleter = _exact_free
    _EXACT[p.value] = (m, shape)
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.py_object
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    t = torch.from_dlpack(PyCapsule_New(ctypes.addressof(m), b'dltensor', None))
    return t, (base.value, size.value)
