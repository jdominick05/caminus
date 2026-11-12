"""What a T-token verify pass costs on the dense qwen35 line (Swift 1.5; engine/q35_dense_verify.py) against one decode
step at the same depth: after an own prefill of D positions of real text, ms per pass for T = 1 .. 4 and per decode step,
interleaved by round, medians of --rounds (each pass synchronized; every pass at the same position and commit 0, so the
state it reads is the prefill's). Prints the ratio to a decode step: the tokens a pass must yield to pay for itself.
  tools/guard.sh LOG 1200 python -I benchmarks/q35_verify.py GGUF [--depths 512,4096] [--rounds 9]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, glob, os, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'python'))
import numpy as np
import torch
from caminus import ggufio
from caminus.bpe import BPE
from caminus.engine.q35_dense import Dense35Own
from caminus.engine.q35_dense_verify import Dense35Verify, MAX_T
from caminus.prefill.q35own import Q35OwnPrefill, buffer_need_mib, shapes_of

ap = argparse.ArgumentParser(); ap.add_argument('gguf'); ap.add_argument('--depths', default='512,4096')
ap.add_argument('--rounds', type=int, default=9); ap.add_argument('--per-wgp', default='', help='verify grids to add, e.g. 6,7')
A = ap.parse_args()
depths = [int(x) for x in A.depths.split(',')]
tk = BPE(ggufio.GGUFFile(A.gguf).meta)
text = ''.join(open(f).read() for f in sorted(glob.glob(os.path.join(ROOT, 'docs', '**', '*.md'), recursive=True)))
ids = tk.encode(text)
if len(ids) < max(depths) + MAX_T + 1: raise SystemExit(f'the docs are {len(ids)} tokens: lower --depths')
CH = 512
grids = [int(w) for w in A.per_wgp.split(',') if w]
eng = Dense35Own(A.gguf, max_ctx=max(depths) + MAX_T + 8, spec_T=MAX_T,
                 vram_extra_mib=(MAX_T + len(grids)) * Dense35Verify.need_mib(MAX_T) + buffer_need_mib(shapes_of(A.gguf), CH) + 16)
print(torch.cuda.get_device_name(0), eng.module, f'{eng.per_wgp} workgroups per WGP', flush=True)
pf = Q35OwnPrefill(eng, chunk=CH)
arms = {f'T={T}': Dense35Verify(eng, T) for T in range(1, MAX_T + 1)}
arms.update({f'T={MAX_T} wgp {w}': Dense35Verify(eng, MAX_T, per_wgp=w) for w in grids})
sync = lambda: torch.cuda.synchronize(eng.dev)
for D in depths:
    eng.reset(); pf.run(ids[:D], 0, 0, D); sync()
    ms = {'decode': []} | {k: [] for k in arms}
    for r in range(A.rounds + 1):
        t0 = time.perf_counter(); eng.verify(ids, D, 0); sync(); dt = time.perf_counter() - t0
        if r: ms['decode'].append(dt)
        for k, v in arms.items():
            t0 = time.perf_counter(); v.run(ids[D:D + v.T], D, 0); sync(); dt = time.perf_counter() - t0
            if r: ms[k].append(dt)
    d = 1e3 * float(np.median(ms['decode']))
    print(f'depth {D}: decode step {d:.2f} ms', flush=True)
    for k in arms:
        m = 1e3 * float(np.median(ms[k]))
        print(f'depth {D}: verify {k:<12} {m:6.2f} ms per pass ({m / d:.2f}x a decode step)', flush=True)
check_bails = len(eng.bails)
print(f'bails {check_bails}')
print('RESULT DONE')
