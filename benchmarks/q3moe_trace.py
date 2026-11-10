"""Where a Qwen3-30B-A3B decode token's time goes in Q3MoEOne (engine/q3moe_one.py), from K3_TRACE timestamps (module
k3_gguf_q3m_own_trace): per segment, the hop (its dependencies' last publish -> its first workgroup past the wait) and the
span (first workgroup past the wait -> its last publish); the critical path (walked back through each segment's later
dependency) summed per op kind; medians over --tokens traced tokens after --warm untraced ones. Also the host's view:
step() wall time against the launch's GPU time (events) on the plain module.
  tools/guard.sh LOG 900 python -I benchmarks/q3moe_trace.py GGUF [--warm 8] [--tokens 10]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, collections, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from caminus.engine import k3
from caminus.engine.q3moe_one import Q3MoEOne
ap = argparse.ArgumentParser(); ap.add_argument('gguf'); ap.add_argument('--warm', type=int, default=8)
ap.add_argument('--tokens', type=int, default=10)
ap.add_argument('--detail', type=int, default=None, help='print this layer\'s segments (last traced token)')
ap.add_argument('--modules', nargs='+', default=['k3_gguf_q3m_own_trace'], help='one engine per run: the card holds one copy')
A = ap.parse_args()
OPN = {v: k[3:] for k, v in vars(k3).items() if k.startswith('OP_') and isinstance(v, int)}


def seg_name(sg):
    n = OPN.get(sg.op, str(sg.op))
    if n == 'OWN_MV':
        role = {(4096, 2048): 'q', (512, 2048): 'k|v', (2048, 4096): 'o', (151936, 2048): 'head'}.get((sg.N, sg.K))
        if role is None: role = 'down' if sg.K == 768 else 'gate|up'
        return 'mv ' + role
    return n.lower()


print(torch.cuda.get_device_name(0), flush=True)
for module in A.modules:
    eng = Q3MoEOne(A.gguf, max_ctx=256, module=module); E = eng.E
    S = len(E.segs); G = E.grid; names = [seg_name(sg) for sg in E.segs]
    tok = 785
    for _ in range(A.warm): tok = int(torch.argmax(eng.step(tok)))
    if module.endswith('_trace'):
        tr = torch.zeros(G * S * 8, dtype=torch.int64, device='cuda'); per_seg = []
        totals = []
        for _ in range(A.tokens):
            tr.zero_(); tok = int(torch.argmax(eng.step(tok, trace=tr)))
            T = tr.view(G, S, 8).cpu().numpy().astype(np.float64)
            t0 = T[:, :, 0][T[:, :, 0] > 0].min()
            start, end = np.zeros(S), np.zeros(S)
            for s in range(S):
                ran = T[:, s, 4] < E.segs[s].ntask
                start[s] = (T[ran, s, 1].min() - t0) / 100.0; end[s] = (T[ran, s, 7].max() - t0) / 100.0
            dd = np.zeros(S)
            for s in range(S):
                deps = [d for d in (E.segs[s].dep, E.segs[s].dep2p1 - 1) if d >= 0]
                dd[s] = max((end[d] for d in deps), default=0.0)
            per_seg.append((start, end, dd)); totals.append(end.max())
            if A.detail is not None and len(totals) == A.tokens:      # the last traced token, one layer's segments
                per = (S - 3) // len(eng.L); n0 = 1 + per * A.detail      # (the embedding, then per segments per layer)
                print(f'\nlayer {A.detail}, last traced token (us from the token start): seg name ntask nWG | claim min/med | '
                      f'wait end min/med | first task end min | last publish | deps end')
                for s_ in range(max(0, n0), min(S, n0 + per + 2)):
                    ran = T[:, s_, 4] < E.segs[s_].ntask; c = (T[ran, s_, 0] - t0) / 100; w = (T[ran, s_, 1] - t0) / 100
                    f = (T[ran, s_, 2] - t0) / 100
                    p7 = (T[ran, s_, 7] - t0) / 100      # each workgroup's last publish in this segment
                    op1 = (T[ran, s_, 2] - T[ran, s_, 1]) / 100; pb1 = (T[ran, s_, 3] - T[ran, s_, 2]) / 100   # first task: op, publish
                    q = lambda a: f'{np.median(a):5.1f} {np.percentile(a, 90):5.1f} {a.max():5.1f}'
                    print(f'  {s_:4d} {names[s_]:13s} {E.segs[s_].ntask:5d} {int(ran.sum()):4d} | {c.min():8.1f} {np.median(c):8.1f} | '
                          f'{w.min():8.1f} {np.median(w):8.1f} | {f.min():8.1f} | {end[s_]:8.1f} | {dd[s_]:8.1f} | WG done '
                          f'p50 {np.median(p7):8.1f} p90 {np.percentile(p7, 90):8.1f} | 1st task op p50/p90/max {q(op1)} | publish {q(pb1)}')
        st = np.median(np.stack([p[0] for p in per_seg]), 0); en = np.median(np.stack([p[1] for p in per_seg]), 0)
        de = np.median(np.stack([p[2] for p in per_seg]), 0)
        # critical path from the last-finishing segment back through the later-finishing dependency
        path, s = [], int(np.argmax(en))
        while s >= 0:
            path.append(s)
            deps = [d for d in (E.segs[s].dep, E.segs[s].dep2p1 - 1) if d >= 0]
            s = max(deps, key=lambda d: en[d]) if deps else -1
        crit = collections.defaultdict(lambda: [0, 0.0, 0.0])
        for s in path:
            c = crit[names[s]]; c[0] += 1; c[1] += max(0.0, st[s] - de[s]); c[2] += en[s] - st[s]
        tot = float(np.median(totals))
        print(f'\ntraced token: {tot / 1e3:.3f} ms first claim -> last publish (median of {A.tokens}); {S} segments, '
              f'critical path {len(path)} segments: hops {sum(c[1] for c in crit.values()) / 1e3:.3f} ms, '
              f'spans {sum(c[2] for c in crit.values()) / 1e3:.3f} ms')
        print(f'  {"op on the critical path":24s} {"n":>4s} {"hops ms":>8s} {"spans ms":>9s} {"hop us":>7s} {"span us":>8s}')
        for k, (n, h, sp) in sorted(crit.items(), key=lambda kv: -(kv[1][1] + kv[1][2])):
            print(f'  {k:24s} {n:4d} {h / 1e3:8.3f} {sp / 1e3:9.3f} {h / n:7.2f} {sp / n:8.2f}')
    else:
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]; gpu, wall = [], []
        for _ in range(A.tokens * 3):
            t1 = time.perf_counter(); ev[0].record(); lg = eng.step(tok); ev[1].record(); torch.cuda.synchronize()
            wall.append(time.perf_counter() - t1); gpu.append(ev[0].elapsed_time(ev[1])); tok = int(torch.argmax(lg))
        print(f'plain module: step() wall median {1e3 * np.median(wall):.3f} ms, events around it {np.median(gpu):.3f} ms')
    del eng, E; torch.cuda.empty_cache()
print('RESULT DONE')
