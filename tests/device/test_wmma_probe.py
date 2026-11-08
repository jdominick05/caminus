"""Probe gfx1100's iu4 WMMA operand encoding (kernels/wmma_probe.hip) against an int64 reference: nibble order,
signedness at -8 and +7, whether lanes 16-31 must duplicate A/B, and the C layout.
  python tests/device/test_wmma_probe.py"""
import ctypes, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import torch
from caminus.device.hip import Module

m = Module('wmma_probe')
torch.manual_seed(0)
cases = {'random': (torch.randint(-8, 8, (16, 16)), torch.randint(-8, 8, (16, 16))),
         'extremes': (torch.full((16, 16), -8), torch.full((16, 16), 7)),
         'one-hot k': (torch.eye(16, dtype=torch.long) * 3, torch.arange(16).repeat(16, 1) % 16 - 8)}
for mode, desc in [(1, 'low nibble first, duplicated'), (0, 'high nibble first, duplicated'), (3, 'low nibble first, upper half-wave zero')]:
    res = []
    for name, (a, b) in cases.items():
        ref = a.to(torch.int64) @ b.to(torch.int64).T
        d = torch.empty(16, 16, dtype=torch.int32, device='cuda')
        m.iu4_probe((1,), (32,), a.to(torch.int8).cuda(), b.to(torch.int8).cuda(), d, ctypes.c_int(mode))
        torch.cuda.synchronize()
        res.append(f"{name}: {'exact' if torch.equal(d.cpu().to(torch.int64), ref) else 'MISMATCH'}")
    print(f'mode {mode} ({desc}): ' + ', '.join(res))
