"""K3 engine cost model, part 1 (kernels/sync_bench.hip): microseconds per stage handoff in a persistent kernel on this
card, for claim/done (the engine's sync; safe under partial residency) vs an arrival barrier (needs a co-resident grid)
vs one launch per stage; by blocks per WGP and threads per block, with empty stages and with a little work per task.
Also checks the err flag (a wait that gave up) stays 0.
  HIP_VISIBLE_DEVICES=0 python benchmarks/bench_sync.py"""
import ctypes, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import torch
from caminus.device.hip import Module

m = Module('sync_bench')
props = torch.cuda.get_device_properties(0)
wgps = props.multi_processor_count   # HIP reports WGPs here on gfx11 (WGP mode): 48 on the XTX = 96 CUs
SYNC_BYTES = 4 * (512 * 2 + 4)
sy = torch.zeros(SYNC_BYTES // 4, dtype=torch.int32, device='cuda')
sink = torch.zeros(1, device='cuda')
STAGES, SPIN = 400, 2_000_000
print(f'{props.name}: {wgps} WGPs = {2 * wgps} CUs; {STAGES} stages per launch')


def timed(fn, reps=20):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1e3   # us per launch


err_any = False
print(f'{"blocks/WGP":>10s} {"threads":>7s} {"work":>5s} | claim/done us/stage | arrival us/stage | one launch per stage us   (grid = blocks/WGP x {wgps})')
for threads in (128, 256):
    for bpw in (0.5, 1, 2, 4):
        grid = int(wgps * bpw)
        for work in (0, 200):
            args = lambda: (sy, ctypes.c_int(STAGES), ctypes.c_uint(grid), ctypes.c_int(work), ctypes.c_int(SPIN), sink)
            cd = timed(lambda: m.claim_done((grid,), (threads,), *args())) / STAGES
            ar = timed(lambda: m.arrival((grid,), (threads,), *args())) / STAGES if bpw <= 2 else float('nan')
            em = timed(lambda: [m.empty_stage((grid,), (threads,), sink, ctypes.c_int(work)) for _ in range(50)]) / 50
            torch.cuda.synchronize()
            e = int(sy[-1].item()); err_any |= bool(e)
            print(f'{bpw:10.1f} {threads:7d} {work:5d} | {cd:18.2f} | {ar:16.2f} | {em:10.2f}' + ('   ERR' if e else ''))
            sy.zero_()
print('err flag: ' + ('SET (a wait timed out)' if err_any else 'never set'))
