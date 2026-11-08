"""The iu4 W4A4 GEMM (caminus.device.gemm.gemm_tn_iu4): exact against an int64 reference (fp32 output, with and without the
per-row x per-channel scale epilogue), bf16 output = the fp32 result rounded, and speed in TOPS vs the bf16 kernel and
rocBLAS on the same shapes (run on an idle card for the speed part).
  python tests/device/test_gemm_iu4.py [--speed]"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import torch
import torch.nn.functional as F
from caminus.device import gemm

torch.manual_seed(0)
ok = True
for M, N, K in [(128, 128, 128), (256, 384, 896), (8192, 896, 2048), (1024, 7680, 896)]:
    qa = torch.randint(-8, 8, (M, K), device='cuda'); qb = torch.randint(-8, 8, (N, K), device='cuda')
    qa[0] = 7; qb[0] = -8; qa[1] = -8; qb[1] = -8
    ref = (qa.double() @ qb.double().T).long()        # exact in double; |acc| <= 64 K < 2^24, so exact in fp32 too
    pa, pb = gemm.pack_int4(qa), gemm.pack_int4(qb)
    rs = torch.rand(M, device='cuda') + 0.5; cs = torch.rand(N, device='cuda') + 0.5
    for tile in gemm.TILES_IU4:
        if M % tile[0] or N % tile[1]: continue
        c = gemm.gemm_tn_iu4(pa, pb, out_dtype=torch.float32, tile=tile)
        cs_ = gemm.gemm_tn_iu4(pa, pb, rs, cs, torch.float32, tile=tile)
        cb = gemm.gemm_tn_iu4(pa, pb, rs, cs, torch.bfloat16, tile=tile)
        want = (ref.float() * cs[None, :]) * rs[:, None]
        e1, e2, e3 = torch.equal(c, ref.float()), torch.equal(cs_, want), torch.equal(cb, want.bfloat16())
        ok &= e1 and e2 and e3
        print(f'{M}x{N}x{K} tile {tile}: codes exact {e1}, scaled exact {e2}, bf16 = rounded fp32 {e3}')

# per-group scales: the reference runs the FMA chain in float64 then rounds each step to fp32 (exact: |P_g| <= 8192)
for M, N, K in [(128, 128, 128), (256, 384, 896), (1024, 7680, 896), (8192, 896, 2048)]:
    qa = torch.randint(-8, 8, (M, K), device='cuda'); qb = torch.randint(-8, 8, (N, K), device='cuda')
    pa, pb = gemm.pack_int4(qa), gemm.pack_int4(qb)
    G = K // 128; gs = (torch.rand(N, G, device='cuda') + 0.1) / 64; rs = torch.rand(M, device='cuda') + 0.5
    acc = torch.zeros(M, N, device='cuda', dtype=torch.float32)
    for g in range(G):
        P = qa[:, 128 * g:128 * (g + 1)].double() @ qb[:, 128 * g:128 * (g + 1)].double().T
        acc = (P * gs[:, g].double()[None, :] + acc.double()).float()
    want = acc * rs[:, None]
    for tile in gemm.TILES_IU4G:
        if M % tile[0] or N % tile[1]: continue
        c32 = gemm.gemm_tn_iu4g(pa, pb, gs, rs, torch.float32, tile=tile)
        cb = gemm.gemm_tn_iu4g(pa, pb, gs, rs, torch.bfloat16, tile=tile)
        e1 = torch.equal(c32, want); e2 = torch.equal(cb, want.bfloat16()); ok &= e1 and e2
        print(f'grouped {M}x{N}x{K} tile {tile}: fp32 = FMA-chain reference {e1} ({(c32 != want).sum().item()} differ), bf16 = rounded {e2}')

if '--speed' in sys.argv:
    def bench(fn, n=20):
        for _ in range(3): fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / n
    print(f'{"M x N x K":>20s} | iu4 per-channel per tile (TOPS) | iu4 per-group | bf16 ours best | rocBLAS bf16 (TFLOPS)')
    for M, N, K in [(8192, 896, 896), (8192, 896, 2048), (8192, 7680, 896), (8192, 4096, 896), (8192, 5248, 896), (8192, 8192, 4096)]:
        qa = gemm.pack_int4(torch.randint(-8, 8, (M, K), device='cuda')); qb = gemm.pack_int4(torch.randint(-8, 8, (N, K), device='cuda'))
        a = torch.randn(M, K, device='cuda').bfloat16(); b = torch.randn(N, K, device='cuda').bfloat16()
        fl = 2 * M * N * K
        iu4 = {t: fl / bench(lambda: gemm.gemm_tn_iu4(qa, qb, tile=t)) / 1e12 for t in gemm.TILES_IU4 if M % t[0] == 0 and N % t[1] == 0}
        bf = max(fl / bench(lambda: gemm.gemm_tn(a, b, tile=t)) / 1e12 for t in gemm.TILES if gemm.supported(M, N, K, t))
        rb = fl / bench(lambda: F.linear(a, b)) / 1e12
        gs = torch.rand(N, K // 128, device='cuda')
        iu4g = {t: fl / bench(lambda: gemm.gemm_tn_iu4g(qa, qb, gs, tile=t)) / 1e12 for t in gemm.TILES_IU4G if M % t[0] == 0 and N % t[1] == 0}
        print(f'{f"{M}x{N}x{K}":>20s} | ' + ', '.join(f'{t[0]}x{t[1]} {v:5.1f}' for t, v in iu4.items()) + ' | grouped ' +
              ', '.join(f'{t[0]}x{t[1]} {v:5.1f}' for t, v in iu4g.items()) + f' | {bf:5.1f} | {rb:5.1f}')
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
