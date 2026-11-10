"""Qwen3-30B-A3B batched prefill speed (caminus.prefill.q3m.Q3MPrefill): positions [0, L) of random tokens per length,
medians over --rounds, prompt tok/s = L / seconds (prefill only: the last token's K3 pass is the decode's). With --ops,
one more run per length with every launch timed by HIP events (the time per op kind: GEMM by matrix, attention, glue).
  tools/guard.sh LOG 1800 python -I benchmarks/q3moe_prefill.py GGUF [--lengths 512,4096] [--chunk 1024] [--max-ctx N] [--ops]"""
import argparse, collections, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from caminus.engine.q3moe_one import Q3MoEOne
from caminus.prefill.q3m import Q3MPrefill
ap = argparse.ArgumentParser()
ap.add_argument('gguf'); ap.add_argument('--lengths', default='512,4096'); ap.add_argument('--chunk', type=int, default=1024)
ap.add_argument('--max-ctx', type=int, default=None); ap.add_argument('--rounds', type=int, default=3); ap.add_argument('--ops', action='store_true')
A = ap.parse_args()
Ls = [int(x) for x in A.lengths.split(',')]
print(torch.cuda.get_device_name(0), flush=True)
eng = Q3MoEOne(A.gguf, max_ctx=A.max_ctx or max(Ls) + 8)
pf = Q3MPrefill(eng, chunk=A.chunk)
print(f'max_ctx {eng.max_ctx}, chunk {A.chunk}: engine {eng.vram_mib:.0f} MiB + prefill buffers {pf.need_mib:.0f} MiB', flush=True)
g = np.random.default_rng(0)
for L in Ls:
    toks = g.integers(0, 151000, L).tolist()
    ts = []
    for r in range(A.rounds + 1):
        eng.reset(); torch.cuda.synchronize(); t0 = time.perf_counter()
        pf.run(toks, 0, 0, L); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    s = float(np.median(ts[1:]))
    print(f'  L {L:6d}: {s * 1e3:9.1f} ms = {L / s:7.0f} tok/s prefill (rounds {min(ts[1:]) * 1e3:.1f} .. {max(ts[1:]) * 1e3:.1f})', flush=True)
    if A.ops:
        acc = collections.defaultdict(float); evs = []
        O = pf.ops
        for name in ('mm', 'emb', 'add_rms16', 'router', 'topk', 'group', 'swiglu', 'qk_rope', 'attn'):
            f = getattr(O, name)
            def wrap(*a, _f=f, _n=name, **kw):
                key = _n
                if _n == 'mm':
                    W = a[0]; key = f'mm {W[0]} {W[1]}x{W[2]}' + (' experts' if 'work' in kw else '')
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); _f(*a, **kw); e1.record(); evs.append((key, e0, e1))
            setattr(O, name, wrap)
        eng.reset(); pf.run(toks, 0, 0, L); torch.cuda.synchronize()
        for k, e0, e1 in evs: acc[k] += e0.elapsed_time(e1)
        for name in ('mm', 'emb', 'add_rms16', 'router', 'topk', 'group', 'swiglu', 'qk_rope', 'attn'):
            delattr(O, name)
        tot = sum(acc.values())
        for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
            print(f'      {k:34s} {v:9.2f} ms  {100 * v / tot:5.1f} %', flush=True)
print('RESULT DONE')
