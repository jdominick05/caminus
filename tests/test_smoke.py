"""Toolchain smoke test: code object loads in torch's HIP runtime, launches on torch's stream in order, one runtime.
  python tests/test_smoke.py"""
import ctypes, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import torch
from caminus.hip import Module

m = Module('smoke')
n = 1 << 20
x = torch.randn(n, device='cuda')
y = torch.randn(n, device='cuda')
ref = 3.0 * x + y                                   # queued on the current stream
m.saxpy(((n + 255) // 256,), (256,), y, x, ctypes.c_float(3.0), ctypes.c_int(n))
d = (y - ref).abs().max().item()                     # torch reads y after our kernel, no explicit sync
print(f'saxpy max abs error {d:.3e}')
assert d < 1e-5

# stream ordering on a side stream: torch writes, we update, torch reads, all queued without host syncs
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    y2 = torch.zeros(n, device='cuda'); y2.add_(1.0)
    m.saxpy(((n + 255) // 256,), (256,), y2, x, ctypes.c_float(2.0), ctypes.c_int(n))
    out = y2 * 1.0
s.synchronize()
d2 = (out - (2.0 * x + 1.0)).abs().max().item()
print(f'side-stream ordering max abs error {d2:.3e}')
assert d2 < 1e-5

libs = {l.split()[-1] for l in open('/proc/self/maps') if 'libamdhip64' in l}
print('HIP runtimes mapped:', libs)
assert len(libs) == 1

# host overhead per launch (tiny kernel; the GPU is not the limit)
torch.cuda.synchronize(); k = 2000; t = time.perf_counter()
for _ in range(k):
    m.saxpy((1,), (64,), y, x, ctypes.c_float(1.0), ctypes.c_int(64))
host_us = (time.perf_counter() - t) / k * 1e6; torch.cuda.synchronize()
print(f'host cost per launch through ctypes: {host_us:.1f} us')
print('OK')
