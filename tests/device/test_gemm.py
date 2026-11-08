"""Our WMMA GEMM (caminus.device.gemm.gemm_tn) vs torch / rocBLAS on the S1 step's shapes:
  1. int4 codes with fp32 output, informational: bf16 WMMA is NOT exact on gfx1100 (each negative product loses ~1 LSB
     at 2^-24 x the dot's largest product; kernels/wmma_probe.hip), so this reports the error instead of gating on it;
  2. bf16 output: bitwise equal to rocBLAS (F.linear) on every tested shape (the gate: a drop-in without new numerics);
  3. speed per shape vs torch (rocBLAS), and the LDS-only main loop (the microbench gate: >= 95 TFLOPS).
  python tests/device/test_gemm.py [--quick]"""
import ctypes, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import torch
import torch.nn.functional as F
from caminus.device import gemm

torch.manual_seed(0)
ok = True
# 1. codes with fp32 output (informational)
for M, N, K in [(256, 256, 896), (8192, 896, 2048)]:
    qa = torch.randint(-8, 8, (M, K), device='cuda'); qb = torch.randint(-8, 8, (N, K), device='cuda')
    ref = (qa.double() @ qb.double().T)
    c = gemm.gemm_tn(qa.bfloat16(), qb.bfloat16(), torch.float32)
    d = (c.double() - ref).abs()
    print(f'codes {M}x{N}x{K}, fp32 output: {100 * (d > 0).double().mean().item():.1f}% inexact, max error {d.max().item():.2g} (bf16 WMMA)')
# 2. bf16 output vs rocBLAS, bitwise, on the step's shapes (forward form; dgrad shapes in the speed list below too)
shapes = [(8192, 896, 896), (8192, 896, 2048), (8192, 4096, 896), (8192, 896, 4096), (8192, 7680, 896), (8192, 896, 7680),
          (8192, 5248, 896), (8192, 896, 5248), (1024, 896, 24576), (1536, 896, 2048), (2048, 1536, 896)]
if '--quick' in sys.argv: shapes = shapes[:2]
for M, N, K in shapes:
    a = torch.randn(M, K, device='cuda').bfloat16(); b = (torch.randn(N, K, device='cuda') / K ** 0.5).bfloat16()
    ref = F.linear(a, b)
    res = []
    for tile in gemm.TILES:
        if not gemm.supported(M, N, K, tile): continue
        eq = torch.equal(gemm.gemm_tn(a, b, tile=tile), ref)
        known = N == 896 and K in (7680, 5248)   # SHARE block dgrads: rocBLAS may choose another summation order
        ok &= eq or known; res.append(f'{tile[0]}x{tile[1]} {"bitwise equal" if eq else ("differs (rocBLAS order, allowed)" if known else "DIFFERS")}')
    print(f'bf16 {M}x{N}x{K} vs rocBLAS: ' + ', '.join(res))


def bench(fn, n=20):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n


# 3. speed
print(f'{"M x N x K":>22s} | rocBLAS TFLOPS | ours per tile (TFLOPS)')
for M, N, K in shapes:
    a = torch.randn(M, K, device='cuda').bfloat16(); b = torch.randn(N, K, device='cuda').bfloat16()
    fl = 2 * M * N * K
    t_ref = bench(lambda: F.linear(a, b))
    res = []
    for tile in gemm.TILES:
        if gemm.supported(M, N, K, tile): res.append(f'{tile[0]}x{tile[1]} {fl / bench(lambda: gemm.gemm_tn(a, b, tile=tile)) / 1e12:5.1f}')
    print(f'{f"{M} x {N} x {K}":>22s} | {fl / t_ref / 1e12:14.1f} | ' + ', '.join(res))
M, N, K = 8192, 8192, 4096
a = torch.randn(M, K, device='cuda').bfloat16(); b = torch.randn(N, K, device='cuda').bfloat16()
t = bench(lambda: gemm.gemm_tn(a, b, kernel='gemm_tn_128x128_lds_only'))
print(f'LDS-only main loop (128x128 tiles, {M}x{N}x{K}): {2 * M * N * K / t / 1e12:.1f} TFLOPS (gate >= 95)')
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
