"""K3 engine skeleton (kernels/k3.hip): (1) matvec segments bitwise equal to gemm_tn_iu4g on every S1 shape; (2) a
400-segment dependency chain and a dependent matvec chain at 1x / 3x / 5x oversubscribed grids: bitwise, status never
set; (3) speed of a synthetic S1-shaped token tape (12 blocks x [block projection -> o+down stage] + the head: 25
dependent segments) in us per launch, vs the cost model.
  HIP_VISIBLE_DEVICES=0 tools/guard.sh LOG 600 python -I tests/test_k3_engine.py   (the guard runs gpu-high: decode bursts at auto clocks)"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import torch
from caminus import gemm, k3

torch.manual_seed(0)
ok = True
wgps = torch.cuda.get_device_properties(0).multi_processor_count


def mats(N, K):
    q = torch.randint(-8, 8, (N, K), device='cuda'); gs = ((torch.rand(N, K // 128, device='cuda') + 0.05) / 50).contiguous()
    return q, k3.repack_mv(q), gs


def ref_row(q, gs, a, rs, dtype):
    xa = a.to(torch.int64)[None].expand(64, -1).contiguous()
    r = torch.full((64,), rs, device='cuda')
    return gemm.gemm_tn_iu4g(gemm.pack_int4(xa), gemm.pack_int4(q), gs, r, dtype, tile=(64, 128))[0]


# (1) correctness through the engine
eng = k3.Engine(); checks = []
for N, K in [(7680, 896), (5248, 896), (896, 896), (896, 2048), (1536, 896), (896, 768), (24576, 896)]:
    q, w, gs = mats(N, K); a = torch.randint(-8, 8, (K,), device='cuda').to(torch.int8); rs = 0.7
    # the tuned task sizes (40 / 16 / 128 rows) and sizes that leave a partial last task (5248 / 40, 896 / 12, 7680 / 57)
    for dt, rpt in ((torch.bfloat16, 32), (torch.float32, 8), (torch.bfloat16, 40 if K <= 1024 else 16),
                    (torch.float32, 57 if K <= 1024 else 12), (torch.bfloat16, 64 if K <= 1024 else 16)):
        y = torch.empty(N, device='cuda', dtype=dt); eng.matvec(w, gs, a, rs, y, rows_per_task=rpt)
        checks.append((N, K, dt, y, q, gs, a, rs))
eng.build().run()
for N, K, dt, y, q, gs, a, rs in checks:
    e = torch.equal(y, ref_row(q, gs, a, rs, dt)); ok &= e
    if not e or dt == torch.float32: print(f'matvec {N}x{K} {str(dt)[6:]}: {"bitwise" if e else "DIFF"}')
print(f'matvec segments: {len(checks)} (7 shapes x 5 task sizes incl. partial last tasks and the tuned 40 / 16 / 64), all bitwise: {ok}')

# (2) dependency chains under oversubscription
for mult in (1, 3, 5):
    grid = wgps * 4 * mult
    e1 = k3.Engine(grid=grid)
    prev = -1
    for i in range(400): prev = e1.nop(ntask=48, work=50, dep=prev)
    e1.build()
    try:
        for _ in range(5): e1.run()
        chain_ok = True
    except RuntimeError as ex: chain_ok = False; print(ex)
    # dependent matvecs: each segment's input is an independent vector, but it waits on the previous segment
    e2 = k3.Engine(grid=grid); outs = []; prev = -1
    for i in range(24):
        N, K = (7680, 896) if i % 2 == 0 else (896, 2048)
        q, w, gs = mats(N, K); a = torch.randint(-8, 8, (K,), device='cuda').to(torch.int8)
        y = torch.empty(N, device='cuda', dtype=torch.bfloat16); prev = e2.matvec(w, gs, a, 0.5, y, dep=prev); outs.append((y, q, gs, a))
    e2.build()
    try:
        e2.run(); mv_ok = all(torch.equal(y, ref_row(q, gs, a, 0.5, torch.bfloat16)) for y, q, gs, a in outs)
    except RuntimeError as ex: mv_ok = False; print(ex)
    ok &= chain_ok and mv_ok
    print(f'grid {grid} ({mult}x of 4 WGs per WGP): 400-segment chain {"ok" if chain_ok else "FAIL"}, dependent matvecs {"bitwise" if mv_ok else "FAIL"}')

# (3) speed: a synthetic S1 token tape
def token_tape(grid, rpt_a=40, rpt_s2=16, rpt_head=64):
    e = k3.Engine(grid=grid); prev = -1; byts = 0; e.check = []
    for blk in range(12):
        N = 5248 if (blk + 1) % 4 == 0 else 7680
        for (n, k, rpt) in ((N, 896, rpt_a), (896, 2944, rpt_s2)):      # stage 1, then o_proj + down as one stage
            q, w, gs = mats(n, k); a = torch.randint(-8, 8, (k,), device='cuda').to(torch.int8)
            y = torch.empty(n, device='cuda', dtype=torch.bfloat16)
            prev = e.matvec(w, gs, a, 0.5, y, rows_per_task=rpt, dep=prev); e.check.append((y, q, gs, a))
            byts += n * k // 2 + gs.numel() * 4
    q, w, gs = mats(24576, 896); a = torch.randint(-8, 8, (896,), device='cuda').to(torch.int8)
    y = torch.empty(24576, device='cuda', dtype=torch.bfloat16)
    e.matvec(w, gs, a, 0.5, y, rows_per_task=rpt_head, dep=prev); e.check.append((y, q, gs, a))
    byts += 24576 * 448 + gs.numel() * 4
    return e.build(), byts


for grid in (wgps * 2, wgps * 4, wgps * 8):
    e, byts = token_tape(grid)
    e.run()
    torch.cuda.synchronize(); n = 50; t = time.perf_counter()
    for _ in range(n): e.run(check=False)
    torch.cuda.synchronize(); us = (time.perf_counter() - t) / n * 1e6
    e.run()   # status check
    tape_ok = all(torch.equal(y, ref_row(q, gs, a, 0.5, torch.bfloat16)) for y, q, gs, a in e.check); ok &= tape_ok
    if not tape_ok: print('token tape outputs: DIFF')
    print(f'S1-shaped token tape, grid {grid}: {us:.1f} us/token ({1e6 / us:.0f} tok/s), {byts / 2**20:.1f} MiB -> {byts / us / 1e6:.2f} TB/s')
print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
