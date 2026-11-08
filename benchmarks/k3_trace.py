"""Where a K3 token's time goes: runs the synthetic S1 token tape (12 blocks x [block projection -> o+down stage] + the
head, 25 dependent matvec segments) on build/k3_trace.hsaco (the engine with K3_TRACE timestamps) and prints, per
segment, the median over 20 launches of: the hop (end of the dependency wait - the dependency's last publish), the first
task's compute split into activation-code load / dots + P_g to LDS / FMA chain + store, the publish, how many workgroups
claimed only after the dependency was done (their weights were not prefetched), and the segment's span.
  [RPT=40,16,64] HIP_VISIBLE_DEVICES=0 tools/guard.sh LOG 600 python -I benchmarks/k3_trace.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np, torch
from caminus.engine import k3
torch.manual_seed(0)
def mats(N, K):
    q = torch.randint(-8, 8, (N, K), device='cuda'); gs = ((torch.rand(N, K // 128, device='cuda') + 0.05) / 50).contiguous()
    return k3.repack_mv(q), gs
RA, RS2, RH = (int(v) for v in os.environ.get('RPT', '40,16,64').split(','))   # rows per task: stage 1, stage 2, head
SH = []
for blk in range(12):
    N = 5248 if (blk + 1) % 4 == 0 else 7680
    SH += [(N, 896, RA), (896, 2944, RS2)]
SH.append((24576, 896, RH))
e = k3.Engine(grid=192, module="k3_trace"); prev = -1
for n, k, rpt in SH:
    w, gs = mats(n, k); a = torch.randint(-8, 8, (k,), device='cuda').to(torch.int8)
    prev = e.matvec(w, gs, a, 0.5, torch.empty(n, device='cuda', dtype=torch.bfloat16), rows_per_task=rpt, dep=prev)
e.build(); S = len(SH)
tr = torch.zeros(192 * S * 8, dtype=torch.int64, device='cuda')
for _ in range(20): e.run(check=False)
runs = []
for _ in range(20):
    tr.zero_(); e.run(trace=tr); torch.cuda.synchronize(); runs.append(tr.view(192, S, 8).cpu().numpy().astype(np.float64))
# per run: times in us relative to the earliest event; stats per segment, then median over runs
def stats(T):
    t0 = T[:, :, 0].min(); T = T.copy(); T[:, :, [0, 1, 2, 3, 5, 6]] = (T[:, :, [0, 1, 2, 3, 5, 6]] - t0) / 100.0   # 100 MHz -> us
    out = []
    for s in range(S):
        ntask = (SH[s][0] + SH[s][2] - 1) // SH[s][2]
        w = T[:, s, 4] < ntask                              # WGs that ran a task
        start, waited, comp, pub, act, pbar = T[w, s, 0], T[w, s, 1], T[w, s, 2], T[w, s, 3], T[w, s, 5], T[w, s, 6]
        dep_done = out[-1]['done'] if s else 0.0
        out.append(dict(seg=s, n=int(w.sum()), first_claim=start.min(), last_claim=start.max(), wait_end_med=np.median(waited),
                        hop=np.median(waited) - dep_done, comp_med=np.median(comp - waited), comp_max=(comp - waited).max(),
                        pub_med=np.median(pub - comp), act=np.median(act - waited), dots=np.median(pbar - act), chain=np.median(comp - pbar), done=pub.max(), late_start=(start > dep_done).sum()))
    return out
allr = [stats(T) for T in runs]
keys = ['n', 'hop', 'comp_med', 'comp_max', 'pub_med', 'late_start', 'act', 'dots', 'chain']
print('seg  shape          tasks/WGs  hop(us) compute med/max = act load + dots/LDS + chain/store   publish  late WGs  seg span')
for s in range(S):
    m = {k: np.median([r[s][k] for r in allr]) for k in keys}
    span = np.median([r[s]['done'] - (r[s - 1]['done'] if s else 0) for r in allr])
    print(f'{s:3d}  {SH[s][0]:5d}x{SH[s][1]:<5d}  {int(m["n"]):4d}       {m["hop"]:5.2f}   {m["comp_med"]:5.2f} / {m["comp_max"]:5.2f} = {m["act"]:5.2f} + {m["dots"]:5.2f} + {m["chain"]:5.2f}           {m["pub_med"]:5.2f}    {int(m["late_start"]):4d}    {span:5.2f}')
tot = np.median([r[-1]['done'] for r in allr]); print(f'total (first claim to last publish): {tot:.1f} us')

if os.environ.get('SLOW'):   # the slowest workgroups of a few segments in one launch: claim->wait, then the compute parts
    T = runs[len(runs) // 2].copy(); t0 = T[:, :, 0].min(); T[:, :, [0, 1, 2, 3, 5, 6]] = (T[:, :, [0, 1, 2, 3, 5, 6]] - t0) / 100.0
    for s in (2, 3, 4, 24):
        ntask = (SH[s][0] + SH[s][2] - 1) // SH[s][2]
        w = np.where(T[:, s, 4] < ntask)[0]
        comp = T[w, s, 2] - T[w, s, 1]; order = w[np.argsort(-comp)]
        print(f'seg {s}: slowest WGs (wg, wgp, task): claim->wait end, act, dots, chain | fastest for contrast')
        for g in list(order[:5]) + list(order[-2:]):
            r = T[g, s]
            print(f'   wg {g:3d} wgp {g % 48:2d} task {int(r[4]):3d}: {r[1] - r[0]:5.2f} | {r[5] - r[1]:5.2f} {r[6] - r[5]:5.2f} {r[2] - r[6]:5.2f}')
