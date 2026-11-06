"""Soak the Flash-Next engine: greedy decoding for N tokens (the cache and states reset every `chunk` tokens), counting
wait bails (RuntimeError from Engine.run) per device and segment op, and the slowest steps. With --tp a bailed pass is
run again inside FlashNextTP.verify: those recovered bails (m.bails) are listed by position and op; a RuntimeError there
means every retry gave up (unrecovered).
  python tools/fn_soak.py [N=3000] [chunk=400] [--tp] [--model model.gguf]"""
import os, sys, time, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np, torch
from caminus.flashnext import FlashNextK3
TP = '--tp' in sys.argv
if TP: sys.argv.remove('--tp')
from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
PATH = devpaths.reap()
if '--model' in sys.argv: i = sys.argv.index('--model'); PATH = sys.argv[i + 1]; del sys.argv[i:i + 2]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
CH = int(sys.argv[2]) if len(sys.argv) > 2 else 400
for d in (0, 1): torch.cuda.set_per_process_memory_fraction(0.95 if d == 1 else 0.97, d)   # XTX: the fixed 0.5 GB reserve
if TP:
    from caminus.flashnext_tp import FlashNextTP
m = (FlashNextTP if TP else FlashNextK3)(PATH)
rep_bad = 0
bails, ts = collections.Counter(), []
toks = [248044, 760]; done = 0; t_start = time.time()
while done < N:
    i = len(toks) - 1
    t0 = time.perf_counter()
    try:
        nxt = int(torch.argmax(m.step(toks, i)).item())
        if TP and done % 50 == 0 and m.check_replicas() != (True, True): rep_bad += 1
    except RuntimeError as ex:
        msg = str(ex); dev = msg.split('on device ')[1].split()[0] if 'on device' in msg else ('tp' if TP else '?')
        op = msg.split('= ')[1].split()[0] if '= ' in msg else '?'
        bails[(dev, op)] += 1; print('BAIL', msg[:160]); nxt = 13
    ts.append(time.perf_counter() - t0); toks.append(nxt); done += 1
    if len(toks) > CH: m.reset(); toks = [248044, 760 + done % 1000]
w = np.array(ts) * 1e3
if TP:
    print(f'replica mismatches (every 50th token): {rep_bad}')
    rec = collections.Counter((b.split(' = ')[1].split()[0] if ' = ' in b else '?') for _, _, bad in m.bails for b in bad)
    print(f'recovered bails (a pass run again inside verify): {len(m.bails)} at positions {sorted({p for p, _, _ in m.bails})}, '
          f'retry attempts max {max((a + 1 for _, a, _ in m.bails), default=0)}, ops {dict(rec)}')
print(f'{done} tokens ({done * 2} launches) in {time.time() - t_start:.0f} s: bails raised{" (unrecovered)" if TP else ""} {sum(bails.values())} {dict(bails)}; '
      f'step median {np.median(w):.2f} ms, p99 {np.percentile(w, 99):.2f} ms, max {w.max():.2f} ms')
