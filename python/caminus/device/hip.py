"""Launch Caminus's code objects from a PyTorch process.

The HIP module API is called through the libamdhip64 that torch already loaded, so the process holds one HIP runtime
and our launches go on torch's streams. Kernels are `extern "C"` and take plain pointers and scalars. The code objects come
from caminus.device.kernels.kernel_dir() (CAMINUS_KERNEL_DIR, a checkout's build/, else the installed package's hsaco/gfx1100).
"""
import ctypes, os
import torch
from . import kernels as _kernels

_hip = None
# caminus.prefill.graph's recorder while a prefill chunk is captured into a HIP graph or its position-dependent nodes are
# patched (None: every launch goes straight to HIP). See Kernel.__call__
RECORD = None
BUILD, BUILD_FROM = _kernels.kernel_dir()   # (directory, where it came from)


def hip():
    global _hip
    if _hip is None:
        if not getattr(torch.version, 'hip', None):
            from .torchcheck import require
            require('caminus.device.hip')                  # (a CUDA / CPU torch: the ROCm build's install command)
        torch.cuda.init()
        h = ctypes.CDLL(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libamdhip64.so'), mode=ctypes.RTLD_GLOBAL)
        h.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        h.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        h.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
        h.hipGetErrorString.restype = ctypes.c_char_p
        _hip = h
    return _hip


def check(err):
    if err != 0:
        raise RuntimeError(f'HIP error {err}: {hip().hipGetErrorString(err).decode()}')


class Kernel:
    def __init__(self, fn, name):
        self.fn, self.name = fn, name

    def __call__(self, grid, block, *args, shmem=0, stream=None):
        """args: torch tensors (passed as device pointers) or ctypes scalars (c_int, c_float, ...)"""
        vals = [ctypes.c_void_p(a.data_ptr()) if isinstance(a, torch.Tensor) else a for a in args]
        params = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in vals])
        grid, block = (tuple(grid) + (1, 1))[:3], (tuple(block) + (1, 1))[:3]
        s = torch.cuda.current_stream().cuda_stream if stream is None else stream
        r = RECORD
        if r is not None and r.intercept(self, grid, block, vals, shmem, s): return   # (patching a captured graph: no launch)
        check(hip().hipModuleLaunchKernel(self.fn, *grid, *block, shmem, ctypes.c_void_p(s), params, None))
        if r is not None: r.launched(self, grid, block, vals, shmem, s)


class Module:
    """one code object, loaded on the current device"""

    def __init__(self, name):
        p = os.path.join(BUILD, f'{name}.hsaco')
        if not os.path.isfile(p):
            raise FileNotFoundError(_kernels.missing_message(name, BUILD, BUILD_FROM))
        data = open(p, 'rb').read()
        self._image = ctypes.create_string_buffer(data, len(data))   # kept alive with the module
        self.handle = ctypes.c_void_p()
        check(hip().hipModuleLoadData(ctypes.byref(self.handle), self._image))
        self._fns = {}

    def __getattr__(self, name):
        if name.startswith('_') or name == 'handle':
            raise AttributeError(name)
        if name not in self._fns:
            f = ctypes.c_void_p()
            check(hip().hipModuleGetFunction(ctypes.byref(f), self.handle, name.encode()))
            self._fns[name] = Kernel(f, name)
        return self._fns[name]
