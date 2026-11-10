"""Qwen3-30B-A3B decode speed by context depth (Q3MoEOne, engine/q3moe_one.py): ms per token at chosen positions, every
arm at the same positions in turn (interleaved), medians over --tokens steps and --rounds rounds. The position is set
directly (no prompt is processed first): decode time does not depend on the cache's values, only on how many rows the
attention reads, so a 40K-position measurement takes seconds instead of 40,960 real passes. Prints one line per depth
and arm, then RESULT DONE.
  tools/guard.sh LOG 1800 python -I benchmarks/q3moe_depth.py GGUF [--depths 8,1024,4096,16384,40900] [--max-ctx N]
      [--arms new=k3_gguf_q3m_lean old=k3_gguf_q3m_own:attn_split=0 ...]
An arm is name=module[@workgroups per WGP][:option=int ...] (engine options: attn_split, splits, lmin, gh, ...)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from caminus.engine import k3
from caminus.engine.q3moe_one import Q3MoEOne
ap = argparse.ArgumentParser()
ap.add_argument('gguf'); ap.add_argument('--depths', default='8,1024,2048,4096,8192')
ap.add_argument('--max-ctx', type=int, default=None, help='default: the deepest position + the tokens timed')
ap.add_argument('--tokens', type=int, default=24); ap.add_argument('--rounds', type=int, default=3)
ap.add_argument('--arms', nargs='+', default=['default=k3_gguf_q3m_lean'])
A = ap.parse_args()
depths = [int(x) for x in A.depths.split(',')]
ARMS = []
for a in A.arms:
    name, spec = a.split('=', 1); spec, *kv = spec.split(':'); mod, _, g = spec.partition('@')
    if mod not in k3.MODULE_CAPS:   # (a measurement build of the full module's ops under another name)
        k3.MODULE_CAPS[mod] = k3.MODULE_CAPS['k3_gguf_q3m_own']; k3.Q3M_MODULES = k3.Q3M_MODULES + (mod,)
    o = dict(module=mod)
    if g: o['per_wgp'] = int(g)
    for x in kv: k_, v_ = x.split('='); o[k_] = int(v_)
    ARMS.append((name, o))
print(torch.cuda.get_device_name(0), flush=True)
max_ctx = A.max_ctx or max(depths) + A.tokens + 1
eng = Q3MoEOne(A.gguf, max_ctx=max_ctx, **ARMS[0][1])
print(f'max_ctx {max_ctx}: {eng.vram_mib:.0f} MiB', flush=True)
keys = ('E', 'S', 'dstage', 'hstage', 'tok', 'pos')
forms = {}
for name, opt in ARMS:
    for k, v in opt.items(): setattr(eng, k, v)
    with torch.cuda.device(eng.dev): eng._build_tape()
    forms[name] = tuple(getattr(eng, k) for k in keys)
def use(n):
    for k, v in zip(keys, forms[n]): setattr(eng, k, v)
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
names = list(forms)
res = {}
for P in depths:
    med = {n: [] for n in names}
    for rd in range(A.rounds):
        tm = {n: [] for n in names}; tok = 785
        for i in range(A.tokens):
            for n in names[i % len(names):] + names[:i % len(names)]:
                use(n); eng.pos_host = P + i
                ev[0].record(); lg = eng.step(tok); ev[1].record(); torch.cuda.synchronize()
                tm[n].append(ev[0].elapsed_time(ev[1]))
            tok = int(torch.argmax(lg))
        for n in names: med[n].append(float(np.median(tm[n][4:])))
    for n in names:
        m = float(np.median(med[n])); res[(P, n)] = m
        print(f'  depth {P:6d}  {n:10s} {m:7.3f} ms/token = {1e3 / m:6.1f} tok/s (rounds {min(med[n]):.3f} .. {max(med[n]):.3f})', flush=True)
for n in names:
    x = np.array(depths, dtype=np.float64); y = np.array([res[(P, n)] for P in depths])
    if len(x) > 1:
        sl, ic = np.polyfit(x[1:] if len(x) > 2 else x, y[1:] if len(x) > 2 else y, 1)
        print(f'  {n:10s} fit (depths past the first): {ic:.3f} ms + {sl * 1e3:.4f} us per position per token')
print('RESULT DONE')
