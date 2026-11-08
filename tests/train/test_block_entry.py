"""caminus.train.actq.block_entry vs the reference block entry under bf16 autocast: h = F.rms_norm(x) (fp32), actq_codes(h),
a = F.linear(h, Wa), b = F.linear(h, Wb) (autocast bf16), on real block inputs of the phase-E model (blocks 0 and 3)
and random ones; forward agreement (bitwise share) and backward dx relative error, plus speed.
  python tests/train/test_block_entry.py"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import torch
import torch.nn.functional as F
from caminus.train.actq import actq_codes, block_entry

torch.manual_seed(0)
ok = True
d, M = 896, 8192
for case in ('random', 'heavy-tailed'):
    x = torch.randn(M, d, device='cuda') * (1 if case == 'random' else torch.randn(M, 1, device='cuda').exp() * 3)
    nw = (torch.rand(d, device='cuda') + 0.5); Wa = torch.randn(7, d, device='cuda') / 30; Wb = torch.randn(7, d, device='cuda') / 30
    dy = torch.randn(M, d, device='cuda').bfloat16(); da = torch.randn(M, 7, device='cuda').bfloat16(); db = torch.randn(M, 7, device='cuda').bfloat16()
    xr = x.clone().requires_grad_()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        h = F.rms_norm(xr, (d,), nw, 1e-6)
        y0, c0, s0 = actq_codes(h, 4, 1.0)
        a0, b0 = F.linear(h, Wa), F.linear(h, Wb)
    torch.autograd.backward([y0, a0, b0], [dy, da, db]); dx0 = xr.grad
    xn = x.clone().requires_grad_()
    wab = torch.cat([Wa, Wb]).bfloat16().contiguous()
    y1, c1, s1, ab = block_entry(xn, nw, wab, 1e-6, 4, 1.0)
    a1, b1 = ab[:, :7], ab[:, 7:]
    torch.autograd.backward([y1, a1, b1], [dy, da, db]); dx1 = xn.grad
    eq = lambda u, v: f'{100 * (u == v).float().mean().item():.2f}% equal'
    rel = ((dx1 - dx0).norm() / dx0.norm()).item()
    print(f'{case}: xq {eq(y1, y0)}, codes {eq(c1, c0)}, scales {eq(s1, s0)}, a {eq(a1, a0)}, b {eq(b1, b0)} | dx rel {rel:.2e}')
    ok &= (y1 == y0).float().mean().item() > 0.999 and rel < 1e-2
def bench(fn, n=20):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3
def ref():
    xr = x.clone().requires_grad_()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        h = F.rms_norm(xr, (d,), nw, 1e-6); y0 = actq_codes(h, 4, 1.0)[0]; a0, b0 = F.linear(h, Wa), F.linear(h, Wb)
    torch.autograd.backward([y0, a0, b0], [dy, da, db])
def new():
    xn = x.clone().requires_grad_(); y1, _, _, ab = block_entry(xn, nw, wab, 1e-6, 4, 1.0)
    torch.autograd.backward([y1, ab[:, :7], ab[:, 7:]], [dy, da, db])
print(f'fwd+bwd {bench(ref):.2f} -> {bench(new):.2f} ms (M {M})')
print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
