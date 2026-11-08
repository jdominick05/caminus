"""The split attention pair (OP_ATTN_PART + OP_ATTN_COMB, kernels/k3_attn_split.h) alone at a given context length: a tape
of just the pair on a zeroed q8_0 KV cache of P cells (the timing does not depend on the values), the shapes of the one-card
Qwen3.6 engine (16 query heads over 2 kv heads, head dim 256) or --nkv 1 (Qwen3-Next's per-card share). Prints the median
launch time over REPS runs (host events around E.run: one persistent launch, so it includes the launch) and, with the trace
build (--trace: k3_gguf_q35h_trace), each segment's span (first claim to last publish, the 100 MHz steady counter) and
its task count. Run it on the XT alone first when the kernel changed.
  tools/guard.sh LOG SECS python -I benchmarks/attn_bench.py [--ctx 2048,8192,32768] [--nkv 2] [--reps 50] [--trace] [--module M]
      [--ss 64|128: M is a scratch build with that ATTN_SS, registered for the run]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np, torch
from caminus.engine import k3

a = sys.argv[1:]
def opt(n, d, f=str):
    if n in a: i = a.index(n); v = f(a[i + 1]); del a[i:i + 2]; return v
    return d
CTX = [int(v) for v in opt('--ctx', '2048,8192,32768').split(',')]; NKV = opt('--nkv', 2, int); REPS = opt('--reps', 50, int)
TRACE = '--trace' in a
MOD = opt('--module', 'k3_gguf_q35h_trace' if TRACE else 'k3_gguf_q35h')
SS_SCRATCH = opt('--ss', None, int)
if SS_SCRATCH is not None:   # a scratch build (build/<MOD>.hsaco compiled with -DATTN_SS=...): registered for this run only
    k3.MODULE_CAPS[MOD] = k3.MODULE_CAPS['k3_gguf_q35h']; k3.ATTN_G_OF[MOD] = 8; k3.ATTN_SS_OF[MOD] = SS_SCRATCH
    k3.ATTN_SPLIT_MODULES = k3.ATTN_SPLIT_MODULES + (MOD,)
D = 256; G = k3.ATTN_G_OF[MOD]; H = G * NKV
torch.cuda.set_per_process_memory_fraction(0.9, 0)
for P in CTX:
    cache = torch.zeros(P * k3.kv_pos_bytes(NKV * D, True), dtype=torch.uint8, device='cuda')
    q = torch.randn(H * D, device='cuda'); out = torch.zeros(H * D, device='cuda'); pos = torch.full((1,), P - 1, dtype=torch.int32, device='cuda')
    SS = k3.ATTN_SS_OF.get(MOD, k3.ATTN_SS); ns = (P + SS - 1) // SS; part = torch.zeros(H * ns * k3.ATTN_RS, device='cuda')
    E = k3.Engine(module=MOD, grid=k3.resident_grid(MOD), T=1, limit_us=200_000)
    E.attn_dec(q, cache, pos, out, H, NKV, D, 1.0 / 16, q8=True, part=part, max_cells=P)
    E.build(); nseg = len(E.segs); grid = E.grid
    for _ in range(5): E.run(check=True)
    ts = []
    for _ in range(REPS):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); E.run(check=False); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3)
    line = f'ctx {P:6d} (splits {ns:5d} x {NKV} kv heads, {H} heads): launch median {np.median(ts):8.1f} us (p10 {np.percentile(ts, 10):.1f}, p90 {np.percentile(ts, 90):.1f})'
    if TRACE:
        tr = torch.zeros(grid * nseg * 8, dtype=torch.int64, device='cuda'); spans = []
        for _ in range(10):
            tr.zero_(); E.run(trace=tr); torch.cuda.synchronize(); T = tr.view(grid, nseg, 8).cpu().numpy().astype(np.float64)
            row = []
            for s in range(nseg):
                st, en = T[:, s, 0], T[:, s, 7]; w = en > 0
                row.append((en[w].max() - st[w].min()) / 100.0 if w.any() else float('nan'))
            spans.append(row)
        sp = np.median(np.array(spans), axis=0)
        line += ' | spans: ' + ', '.join(f'{nm} {v:.1f} us' for nm, v in zip(('PART', 'COMB'), sp))
    print(line, flush=True)
    del cache, part, E; torch.cuda.empty_cache()
