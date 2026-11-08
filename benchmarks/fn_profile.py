"""Where a Flash-Next (REAP-308) token's time goes in the K3 engine: runs FlashNextK3 on build/k3_gguf_trace.hsaco (the
engine with K3_TRACE timestamps), warms the caches over the first positions of a gg_golden token file, then traces
several steps. Per segment: its completion C[s] = the latest last-publish of any workgroup that ran one of its tasks.
The critical path is walked back from the last segment through whichever dependency (dep / dep2) completed later; a
segment on it costs C[s] - C[that dependency]. Prints per device the token time, the critical path summed by label
(weight tensor name for matvecs, else op name; median over the traced steps) and its segment count.
  python benchmarks/fn_profile.py GOLDEN_DIR [model.gguf] [warm=32] [runs=5]"""
import os, sys, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np, torch
from caminus.engine.flashnext import FlashNextK3
from caminus.engine import k3

GD = sys.argv[1]
from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
PATH = sys.argv[2] if len(sys.argv) > 2 else devpaths.reap()
WARM = int(sys.argv[3]) if len(sys.argv) > 3 else 32
RUNS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
tokens = [int(v) for v in open(f'{GD}/tokens.txt').read().split()]
for d in (0, 1): torch.cuda.set_per_process_memory_fraction(0.92 if d == 1 else 0.90, d)
m = FlashNextK3(PATH, module='k3_gguf_trace')
names = {v: k[3:] for k, v in vars(k3).items() if k.startswith('OP_')}
for i in range(WARM): m.step(tokens, i)
per = [collections.defaultdict(list) for _ in m.engines]; tok = [[] for _ in m.engines]; seg_cost = [[] for _ in m.engines]; npath = [[] for _ in m.engines]
for r in range(RUNS):
    trs = []
    for di, e in enumerate(m.engines):
        with torch.cuda.device(m.devs[di]): trs.append(torch.zeros(e.grid * len(e.segs) * 8, dtype=torch.int64, device='cuda'))
    m.step(tokens, WARM + r, traces=trs)
    for di, e in enumerate(m.engines):
        S = len(e.segs); T = trs[di].view(e.grid, S, 8).cpu().numpy().astype(np.float64)
        t0 = T[:, 0, 0][T[:, 0, 0] > 0].min()
        ran = T[:, :, 4] < np.array([sg.ntask for sg in e.segs])[None, :]
        C = np.where(ran, T[:, :, 7], 0).max(0)
        C = (C - t0) / 100.0                               # 100 MHz -> us
        deps = [[d for d in (sg.dep, sg.dep2p1 - 1) if d >= 0] for sg in e.segs]
        cost = np.zeros(S); s = int(np.argmax(C)); path = 0
        while True:                                        # walk the critical path back
            pd = max(deps[s], key=lambda d: C[d]) if deps[s] else -1
            cost[s] = C[s] - (C[pd] if pd >= 0 else 0.0); path += 1
            if pd < 0: break
            s = pd
        tok[di].append(C.max()); seg_cost[di].append(cost); npath[di].append(path)
        acc = collections.defaultdict(float)
        for s in range(S):
            if cost[s] == 0: continue
            lab = m.seg_names[di].get(s)
            lab = lab.split('.', 2)[-1].replace('.weight', '') if lab else names.get(e.segs[s].op, str(e.segs[s].op))
            if lab == 'F32_MV': lab += f' N={e.segs[s].N}'
            acc[lab] += cost[s]
        for k, v in acc.items(): per[di][k].append(v)
for di, e in enumerate(m.engines):
    print(f'\ndevice {m.devs[di]}: {len(e.segs)} segments, token {np.median(tok[di]):.0f} us (layers {m.layer_seg[di][0][0]}..{m.layer_seg[di][-1][0]})')
    rows = sorted(((np.median(v), k) for k, v in per[di].items()), reverse=True)
    tot = sum(r[0] for r in rows)
    for v, k in rows[:20]: print(f'  {k:28s} {v:8.0f} us  {100 * v / tot:5.1f}%')
    sc = np.median(np.array(seg_cost[di]), 0); on = sc[sc > 0]
    print(f'  critical path: {int(np.median(npath[di]))} of {len(e.segs)} segments, median {np.median(on):.1f} us each')

if os.environ.get('PATH_LAYER'):          # print one layer's critical path (last traced step, device 0)
    L = int(os.environ['PATH_LAYER']); e = m.engines[0]; ls = dict(m.layer_seg[0]); a, b = ls[L], ls.get(L + 1, len(e.segs))
    sc = seg_cost[0][-1]
    for s_ in range(a, b):
        if sc[s_] > 0:
            lab = m.seg_names[0].get(s_) or names.get(e.segs[s_].op)
            print(f'  seg {s_:4d} {lab:40s} ntask {e.segs[s_].ntask:4d} dep {e.segs[s_].dep:4d}/{e.segs[s_].dep2p1 - 1:4d}  {sc[s_]:6.1f} us')
