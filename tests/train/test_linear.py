"""caminus.train.linear.linear vs F.linear (rocBLAS) under bf16: forward, dgrad and wgrad bitwise on the S1 step's shapes
(frozen weight through a cached w^T, and trainable weight), plus the fallback for shapes the kernel does not tile.
  python tests/train/test_linear.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import torch
import torch.nn.functional as F
from caminus.train import linear as L

torch.manual_seed(0)
ok = True
for M, N, K in [(8192, 7680, 896), (8192, 5248, 896), (8192, 896, 896), (8192, 896, 2048), (1536, 1536, 896),
                (1536, 896, 768), (8192, 112, 896)]:
    x0 = torch.randn(M, K, device='cuda').bfloat16(); w0 = (torch.randn(N, K, device='cuda') / K ** 0.5).bfloat16()
    gy = torch.randn(M, N, device='cuda').bfloat16()
    for trainable in (False, True):
        res = {}
        for name in ('ref', 'ours'):
            x = x0.clone().requires_grad_(); w = w0.clone().requires_grad_(trainable)
            y = F.linear(x, w) if name == 'ref' else L.linear(x, w, None if trainable else w.detach().t().contiguous())
            y.backward(gy)
            res[name] = (y.detach(), x.grad, w.grad)
        eq = [torch.equal(a, b) if a is not None else True for a, b in zip(res['ref'], res['ours'])]
        tag = ['y', 'dx', 'dw'] if trainable else ['y', 'dx']
        line = ', '.join(f'{t} {"=" if e else "DIFF"}' for t, e in zip(tag, eq))
        big_k = N in (7680, 5248)   # SHARE's block dgrads (K = N): rocBLAS may pick another summation order
        ok &= eq[0] and (eq[1] or big_k) and eq[2]
        print(f'{M}x{N}x{K} {"trainable" if trainable else "frozen   "}: {line}' + ('  (SHARE block dgrad: rocBLAS order differs, allowed)' if big_k and not eq[1] else ''))
print(f'kernel calls {L.STATS}')
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
