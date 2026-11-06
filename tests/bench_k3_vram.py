# I0: K3 matvec streaming from VRAM (REAP-308-shaped layers, weights far beyond the Infinity Cache) on one card.
#   HIP_VISIBLE_DEVICES=0|1 tools/guard.sh LOG 600 python -I tests/bench_k3_vram.py [layers=24]
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import torch
from caminus import k3
torch.manual_seed(0)
dev = torch.cuda.get_device_properties(0); print(dev.name, f'{dev.total_memory / 1e9:.1f} GB')
def mats(N, K):
    q = torch.randint(-8, 8, (N, K), device='cuda', dtype=torch.int8)
    gs = ((torch.rand(N, K // 128, device='cuda') + 0.05) / 50).contiguous()
    return k3.repack_mv(q.to(torch.int64)), gs
# one REAP-308-like "layer": attn-ish 7168x2560 + 10 experts (gate 640x2560, up 640x2560, down 2560x640)
LAYER = [(7168, 2560, 16)] + [(640, 2560, 16), (640, 2560, 16), (2560, 640, 32)] * 10
NL = int(sys.argv[1]) if len(sys.argv) > 1 else 24          # distinct layers of weights (all different memory)
W = []
for l in range(NL):
    for n, k, r in LAYER: W.append((mats(n, k), n, k, r))
byts = sum(n * k // 2 + n * (k // 128) * 4 for _, n, k, r in W)
print(f'{len(W)} matvecs, {byts / 2**30:.2f} GiB of weights+scales')
def tape(dep):
    e = k3.Engine(grid=4 * dev.multi_processor_count); prev = -1
    for (w, gs), n, k, r in W:
        a = torch.randint(-8, 8, (k,), device='cuda').to(torch.int8)
        prev = e.matvec(w, gs, a, 0.5, torch.empty(n, device='cuda', dtype=torch.bfloat16), rows_per_task=r, dep=prev if dep else -1)
    return e.build()
for dep in (False, True):
    e = tape(dep); e.run(); torch.cuda.synchronize()
    t = time.perf_counter(); n = 10
    for _ in range(n): e.run(check=False)
    torch.cuda.synchronize(); us = (time.perf_counter() - t) / n * 1e6; e.run()
    print(f'{"chained (dependent)" if dep else "independent"}: {us:.0f} us per pass = {byts / us / 1e3:.0f} GB/s')
