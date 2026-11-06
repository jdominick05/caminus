"""K3 engine cost model, part 3: the hand-off cost UNDER LOAD. Each stage streams a fixed amount of cache-resident
bytes (like a stage of block matvecs: 384 tasks per stage at every grid size), chained by claim/done waits vs run on
without waits; the difference per stage is what synchronization costs while memory is busy.
  HIP_VISIBLE_DEVICES=0 python tests/bench_sync_load.py"""
import ctypes, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import torch
from caminus.hip import Module

m = Module('sync_bench')
wgps = torch.cuda.get_device_properties(0).multi_processor_count
sy = torch.zeros((4 * (512 * 2 + 4)) // 4, dtype=torch.int32, device='cuda'); out = torch.zeros(1, dtype=torch.int32, device='cuda')
STAGES, SPIN = 240, 4_000_000
TASK_SET = [int(t) for t in os.environ.get('TASKS', '384').split(',')]
print(f'{torch.cuda.get_device_name()}: {wgps} WGPs; {STAGES} stages')


def timed(fn, reps=10):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1e3   # us


print(f'{"tasks":>5s} {"KB/stage":>8s} {"grid":>5s} | no-wait us/stage (TB/s) | chained us/stage | hop cost under load us')
for TASKS in TASK_SET:
  for stage_kb in (256, 1024, 3456):     # 3456 KB ~ one GDN block projection's int4 weights
      chunk16 = stage_kb * 1024 // 16 // TASKS
      nreg = max(1, (64 * 2**20) // (TASKS * chunk16 * 16))            # keep the touched set ~64 MiB (cache-resident)
      buf = torch.randint(0, 2**31, (nreg * TASKS * chunk16 * 4,), dtype=torch.int32, device='cuda')
      for grid in (wgps * 2, wgps * 4):
          args = lambda w: (sy, buf, ctypes.c_int(STAGES), ctypes.c_uint(TASKS), ctypes.c_uint(chunk16), ctypes.c_int(nreg),
                            ctypes.c_int(w), ctypes.c_int(SPIN), out)
          free = timed(lambda: m.stream_stages((grid,), (256,), *args(0))) / STAGES
          chain = timed(lambda: m.stream_stages((grid,), (256,), *args(1))) / STAGES
          torch.cuda.synchronize(); err = int(sy[-1].item()); sy.zero_()
          bytes_ = TASKS * chunk16 * 16
          print(f'{TASKS:5d} {stage_kb:8d} {grid:5d} | {free:10.2f} ({bytes_ / free / 1e6:4.2f}) | {chain:16.2f} | {chain - free:6.2f}' + ('  ERR' if err else ''))
      del buf
