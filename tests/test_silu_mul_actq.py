"""caminus.actq.silu_mul_actq vs torch's F.silu(g) * u (bf16 under autocast) followed by actq_codes: forward (xq, codes,
scales) and backward (d gu) bitwise, on contiguous gu and on a row-strided view of a wider GEMM output (SHARE's layout),
at the S1 widths (MLP 2048, slices 768 / 512 / 384 / 256), plus speed.
  python tests/test_silu_mul_actq.py"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import torch
import torch.nn.functional as F
from caminus.actq import actq_codes, silu_mul_actq

torch.manual_seed(0)
ok = True


def bench(fn, n=20):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3


for M, Fw, view in [(8192, 2048, False), (8192, 2048, True), (1536, 768, False), (1024, 512, False), (768, 384, False), (512, 256, False)]:
    base = (torch.randn(M, 7680 if view else 2 * Fw, device='cuda') * 2).bfloat16()
    gu0 = base[:, 3584:3584 + 2 * Fw] if view else base
    base[0, :8] = torch.tensor([30., -30., 0., -0.0, 1e-3, -88., 88., 3.])   # silu extremes in the first row
    gy = torch.randn(M, Fw, device='cuda').bfloat16()
    # reference: torch ops under autocast
    gu = gu0.detach().clone().requires_grad_() if not view else gu0.detach().requires_grad_()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        g, u = gu.chunk(2, -1); act = F.silu(g) * u
        y_ref, c_ref, s_ref = actq_codes(act, 4, 1.0)
    y_ref.backward(gy); d_ref = gu.grad.clone()
    gu2 = gu0.detach().requires_grad_()
    y, c, s = silu_mul_actq(gu2, 4, 1.0)
    y.backward(gy); d_new = gu2.grad
    res = [torch.equal(y, y_ref), torch.equal(c, c_ref), torch.equal(s, s_ref), torch.equal(d_new, d_ref)]
    ok &= all(res)
    nd = (d_new != d_ref).sum().item()
    def ref_fb():
        g_ = gu0.detach().requires_grad_()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            a, b = g_.chunk(2, -1); yy = actq_codes(F.silu(a) * b, 4, 1.0)[0]
        yy.backward(gy)
    def new_fb():
        g_ = gu0.detach().requires_grad_(); silu_mul_actq(g_, 4, 1.0)[0].backward(gy)
    print(f'M {M} F {Fw}{" (strided view)" if view else ""}: xq {res[0]}, codes {res[1]}, scales {res[2]}, d gu {res[3]}'
          f'{f" ({nd} differ)" if nd else ""} | fwd+bwd {bench(ref_fb):.2f} -> {bench(new_fb):.2f} ms')
print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
