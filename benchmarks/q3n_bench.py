"""Qwen3-Coder-Next decode speed in the K3 engine (caminus.engine.qwen3next_tp.Qwen3NextTP, both cards), as benchmarks/fn_bench.py
measures Flash-Next: greedy generation, each token from the previous pass (step_token: the argmax on the devices, only
(value, index) read back), per token the host work (token embedding row) and both tapes. The prompt is the code smoke
text (llama.cpp's own token ids from tools/reference/q3n_refs.sh), teacher-forced one token per pass; then N greedy tokens,
timed per token, REPS times from the same prompt state (reset + refill), median over the tokens after the first 8.
At --depth D the prompt is followed by D - len(prompt) more teacher-forced positions (the code text repeated) before
generating (one-token passes: M1 has no batched prefill). --t4 also times the T = 4 verify pass (k3_gguf_q3n_mt).
usage: tools/guard.sh LOG SECS python -I benchmarks/q3n_bench.py REFS [--n 128] [--reps 3] [--depth D] [--t4] [--model GGUF]
Compare: llama-bench tg128 on the same two cards ($ARTIFACTS/q3n/BASELINE.md): 81.8 tok/s at depth 0, 72.6 at 32K."""
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
from caminus import devpaths  # noqa: E402  (catalog models and ARTIFACTS from the environment: CLAUDE.md, "Models and data")
import numpy as np
import torch

a = sys.argv[1:]
def opt(name, default, f=str):
    if name in a: i = a.index(name); v = f(a[i + 1]); del a[i:i + 2]; return v
    return default
N = opt('--n', 128, int); REPS = opt('--reps', 3, int); DEPTH = opt('--depth', 0, int)
PATH = opt('--model', devpaths.catalog('qwen3-coder-next', 'Q3N'))
T4 = '--t4' in a
if T4: a.remove('--t4')
if len(a) != 1: raise SystemExit(__doc__)
REFS = a[0]
def engine_class(path):
    """the engine for the GGUF's architecture (the registry, caminus.arch): qwen3next -> Qwen3NextTP (both cards), qwen35moe /
    qwen35 -> Qwen35One (one card); the Coder's FlashNextTP takes other arguments: refused here"""
    from caminus import arch
    a = arch.of(path)
    if a.kind not in ('tp', 'one-card'): raise SystemExit(f'{path}: arch {arch.meta(path)["arch"]!r} has no engine here')
    return a.engine_class()


import caminus.engine.flashnext   # noqa: E402,F401  (gguf-py on the path)

for d in range(torch.cuda.device_count()): torch.cuda.set_per_process_memory_fraction(0.97 if d == 0 else 0.95, d)   # (one visible card: device 0)
code = [int(v) for v in open(f'{REFS}/e2e_code_decode/tokens.txt').read().split()]
prompt = list(code)
while len(prompt) < DEPTH: prompt += code
prompt = prompt[:max(DEPTH, len(code))]
max_ctx = (len(prompt) + N + 64 + 255) // 256 * 256
t0 = time.time(); m = engine_class(PATH)(PATH, max_ctx=max_ctx); print(f'built in {time.time() - t0:.0f} s; max_ctx {max_ctx}, prompt {len(prompt)} positions, modules {[e.module for e in m.engines]}', flush=True)
meds = []
for rep in range(REPS):
    m.reset(); toks = list(prompt)
    t0 = time.perf_counter()
    for i in range(len(prompt) - 1): m.step_token(toks, i)
    fill = time.perf_counter() - t0
    ts = []
    for i in range(len(prompt) - 1, len(prompt) - 1 + N):
        [torch.cuda.synchronize(d) for d in range(torch.cuda.device_count())]; t0 = time.perf_counter()
        nxt = m.step_token(toks, i); ts.append(time.perf_counter() - t0); toks.append(int(nxt))
    w = np.array(ts[8:]) * 1e3; meds.append(np.median(w))
    print(f'rep {rep}: fill {len(prompt) - 1} positions {fill:.1f} s ({(len(prompt) - 1) / fill:.0f} tok/s); decode {N} tokens median {np.median(w):.3f} ms = {1e3 / np.median(w):.1f} tok/s '
          f'(p10 {np.percentile(w, 10):.3f}, p90 {np.percentile(w, 90):.3f} ms); replicas {m.check_replicas()}', flush=True)
print(f'DECODE depth {len(prompt)}: median of {REPS} rep medians {np.median(meds):.3f} ms/token = {1e3 / np.median(meds):.1f} tok/s (reps {", ".join(f"{v:.3f}" for v in meds)})')
print('greedy continuation (token ids, the last rep):', toks[len(prompt):len(prompt) + 24])
if T4:
    del m; torch.cuda.empty_cache()
    m = engine_class(PATH)(PATH, max_ctx=max_ctx, ntok=4); m.reset(); ts = []
    n = (len(prompt) - 1) // 4 * 4
    for i0 in range(0, n, 4):
        [torch.cuda.synchronize(d) for d in range(torch.cuda.device_count())]; t0 = time.perf_counter()
        m.verify_tokens(prompt, i0, 0 if i0 == 0 else 4); ts.append(time.perf_counter() - t0)
    w = np.array(ts[8:]) * 1e3
    print(f'T=4 verify pass: median {np.median(w):.3f} ms ({4e3 / np.median(w):.1f} positions/s) over {len(w)} passes; replicas {m.check_replicas()}')
