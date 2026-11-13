"""Where a dense qwen35 verify pass's time goes (engine/q35_dense_verify.py on a _trace module: K3_TRACE timestamps), at
T = 1 and T = 4 on the same state: per segment kind (op and format, a matvec with its shape) the summed span (its first
workgroup past the wait -> its last publish), medians over --passes traced passes after --warm untraced ones, and what
T = 4 adds over T = 1 per kind (the T-token tiles' extra cost per token is arithmetic, not bytes). Every pass at position
--depth with commit 0 (the state the steps before it left: no replay).
  tools/guard.sh LOG 1200 python -I benchmarks/q35_verify_trace.py GGUF [--depth 512] [--passes 7]
Swift's module is k3_q35_swift_ver_trace, ThinkingCap's k3_q35_tcap_ver_trace (chosen from the file).
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, collections, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from caminus.device import capacity
from caminus.engine import k3
from caminus.engine.q35_dense import Dense35Own
from caminus.engine.q35_dense_verify import Dense35Verify

ap = argparse.ArgumentParser(); ap.add_argument('gguf'); ap.add_argument('--depth', type=int, default=512)
ap.add_argument('--passes', type=int, default=7); ap.add_argument('--warm', type=int, default=3)
A = ap.parse_args()
OPN = {v: k[3:] for k, v in vars(k3).items() if k.startswith('OP_') and isinstance(v, int)}
mc = A.depth + 16
eng = Dense35Own(A.gguf, max_ctx=mc, spec_T=4, vram_extra_mib=2 * Dense35Verify.need_mib(4, max_ctx=mc) + 64)
mod = 'k3_q35_tcap_ver_trace' if eng.tcap else 'k3_q35_swift_ver_trace'
print(torch.cuda.get_device_name(0), eng.module, '->', mod, flush=True)
toks = [(7 * i + 3) % 5000 + 100 for i in range(mc)]
for p in range(A.depth): eng.verify(toks, p, 0 if p == 0 else 1)   # (decode steps up to the depth: the state every pass reads)


def kind(sg):
    n = OPN.get(sg.op, str(sg.op))
    if n == 'OWN_MV': return f'mv {k3.OWN_FMT_NAME.get(sg.work, sg.work)} {sg.N}x{sg.K}'
    if n == 'OWN_MV16': return f'mv16 {k3.GG_FMT_NAME.get(sg.work, sg.work)} {sg.N}x{sg.K}'
    return n.lower()


res = {}
for T in (1, 4):
    v = Dense35Verify(eng, T, module=mod); E = v.E; S, G = len(E.segs), E.grid
    tr_mib = G * S * 64 / 2 ** 20 + 1
    capacity.check([eng.dev], [tr_mib], f'q35_verify_trace: the trace buffer ({tr_mib:.0f} MiB)')
    tr = torch.zeros(G * S * 8, dtype=torch.int64, device='cuda')
    names = [kind(sg) for sg in E.segs]
    seq = toks[A.depth:A.depth + T]
    for _ in range(A.warm): v.run(seq, A.depth, 0, argmax=False)
    acc, tot = collections.defaultdict(list), []
    for _ in range(A.passes):
        v.run(seq, A.depth, 0, argmax=False)                     # (the guard and the state as a real pass)
        tr.zero_(); v.dstage.copy_(v.hstage); E.run(check=True, trace=tr); torch.cuda.synchronize()
        X = tr.view(G, S, 8).cpu().numpy().astype(np.float64)
        t0 = X[:, :, 0][X[:, :, 0] > 0].min()
        per = collections.defaultdict(float)
        last = 0.0
        for s in range(S):
            ran = X[:, s, 4] < E.segs[s].ntask
            if not ran.any(): continue
            st, en = (X[ran, s, 1].min() - t0) / 100.0, (X[ran, s, 7].max() - t0) / 100.0
            per[names[s]] += en - st; last = max(last, en)
        for k_, x in per.items(): acc[k_].append(x)
        tot.append(last)
    res[T] = ({k_: float(np.median(x)) for k_, x in acc.items()}, float(np.median(tot)))
    print(f'T={T}: traced pass {res[T][1] / 1e3:.3f} ms ({S} segments; spans overlap, so they sum past the pass)', flush=True)
    del v, E, tr; torch.cuda.empty_cache()
k1, k4 = res[1][0], res[4][0]
print(f'\n{"segment kind":34s} {"T=1 us":>9s} {"T=4 us":>9s} {"+us":>9s} {"x":>6s}')
for k_ in sorted(k4, key=lambda q: -(k4[q] - k1.get(q, 0.0))):
    a, b = k1.get(k_, 0.0), k4[k_]
    print(f'{k_:34s} {a:9.0f} {b:9.0f} {b - a:9.0f} {b / a if a else 0:6.2f}')
print('RESULT DONE')
