"""Flash-Next decode speed in the K3 engine, end to end: greedy generation (each token from the previous logits'
argmax, D-277 start: the prompt's tokens first), per token the host work (token embedding row, PLE hash and table
rows), both tapes and the residual copy between the cards. Prints tok/s and the per-token split.
  python benchmarks/fn_bench.py [n_tokens=128] [model.gguf] [--tp]  (--tp: the two-card tensor-parallel engine)"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np, torch
from caminus.engine.flashnext import FlashNextK3

TP = '--tp' in sys.argv
if TP: sys.argv.remove('--tp')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
PATH = sys.argv[2] if len(sys.argv) > 2 else devpaths.reap()
for d in (0, 1): torch.cuda.set_per_process_memory_fraction(0.95 if d == 1 else 0.97, d)   # XTX: the fixed 0.5 GB reserve
if TP:
    from caminus.engine.flashnext_tp import FlashNextTP
kw = {k: f(os.environ[e]) for k, e, f in (('gdn_keys', 'GK', int), ('moe_blocks', 'MB', int), ('vocab_share', 'VS', float)) if e in os.environ}
m = FlashNextTP(PATH, module=os.environ.get('MODULE') or None, **kw) if TP else FlashNextK3(PATH)
if kw: print(kw)
prompt = [248044, 760, 22414, 5285, 315, 13]     # EOT first (D-277), then a short public prompt's tokens
toks = list(prompt)
for i in range(len(prompt) - 1): m.step(toks, i)
t_all, t_host = [], []
for i in range(len(prompt) - 1, len(prompt) - 1 + N):
    torch.cuda.synchronize(0); torch.cuda.synchronize(1); t0 = time.perf_counter()
    lo = m.step(toks, i)
    nxt = int(torch.argmax(lo).item()); t1 = time.perf_counter()
    toks.append(nxt); t_all.append(t1 - t0)
w = np.array(t_all[8:]) * 1e3
print(f'{N} tokens: median {np.median(w):.2f} ms/token = {1e3 / np.median(w):.1f} tok/s (mean {1e3 / w.mean():.1f} tok/s)')
for di, e in enumerate([] if TP else m.engines):
    with torch.cuda.device(m.devs[di]):
        ts = []
        for _ in range(20):
            torch.cuda.synchronize(); t0 = time.perf_counter(); e.run(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    print(f'device {m.devs[di]} tape alone: {np.median(ts) * 1e3:.2f} ms')
