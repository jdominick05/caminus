"""Q3MoEOne decode speed (engine/q3moe_one.py): GPU time per token (events around step(): the staged token + position
copy, one launch, the sync) over --tokens after --warm, feeding back its own argmax; median, p10 / p90, min / max.
  tools/guard.sh LOG 900 python -I benchmarks/q3moe_speed.py GGUF [--tokens 128] [--warm 8]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from caminus.engine.q3moe_one import Q3MoEOne
ap = argparse.ArgumentParser(); ap.add_argument('gguf'); ap.add_argument('--tokens', type=int, default=128)
ap.add_argument('--warm', type=int, default=8); A = ap.parse_args()
print(torch.cuda.get_device_name(0), flush=True)
eng = Q3MoEOne(A.gguf, max_ctx=A.warm + A.tokens + 8)
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]; tm, wall = [], []; tok = 785
for i in range(A.warm + A.tokens):
    t1 = time.perf_counter(); ev[0].record(); lg = eng.step(tok); ev[1].record(); torch.cuda.synchronize()
    tok = int(torch.argmax(lg))
    if i >= A.warm: tm.append(ev[0].elapsed_time(ev[1])); wall.append(1e3 * (time.perf_counter() - t1))
ms = float(np.median(tm))
print(f'decode: {ms:.3f} ms/token = {1e3 / ms:.1f} tok/s (GPU events; p10 {np.percentile(tm, 10):.3f}, p90 {np.percentile(tm, 90):.3f}, '
      f'min {min(tm):.3f}, max {max(tm):.3f}); host wall incl. argmax read {np.median(wall):.3f} ms; positions {A.warm}..{A.warm + A.tokens}')
print('RESULT DONE')
