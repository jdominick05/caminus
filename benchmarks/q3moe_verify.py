"""What a T-token verify pass of Qwen3-30B-A3B costs (Q3MoEVerify, engine/q3moe_verify.py) against plain decode steps
(Q3MoEOne) on the same real consecutive tokens (this repository's docs as text) over a prefilled context: ms per pass
for each T, interleaved with the decode steps over --rounds rounds (the first round is a warm-up), the distinct experts
per layer, and with --trace the pass's critical path summed per op kind (K3_TRACE timestamps, module k3_q3m_ver_trace).
With --accept (the measured fraction of passes in which draft 1, 2, 3 is accepted; a draft counts only when every draft
before it was) it also prints the ms per accepted token for each T: a pass of T tokens checks T - 1 drafts and always
yields one token of its own.
Every pass is first checked bit for bit against the decode steps. Prints one line per depth, then RESULT PASS | FAIL.
  tools/guard.sh LOG 1800 python -I benchmarks/q3moe_verify.py GGUF [--depths 512,4096] [--T 1,2,3,4] [--trace 4]
      [--accept 0.693,0.457,0.253]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, collections, glob, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'python'))
import numpy as np
import torch
from caminus import ggufio
from caminus.bpe import BPE
from caminus.device import capacity
from caminus.engine import k3
from caminus.engine.q3moe_one import Q3MoEOne
from caminus.engine.q3moe_verify import Q3MoEVerify, MAX_T
from caminus.prefill.q3m import Q3MPrefill
ap = argparse.ArgumentParser()
ap.add_argument('gguf'); ap.add_argument('--depths', default='512,4096')
ap.add_argument('--T', default='1,2,3,4'); ap.add_argument('--tokens', type=int, default=48); ap.add_argument('--rounds', type=int, default=6)
ap.add_argument('--trace', type=int, default=0, help='T of the traced pass (0: none)')
ap.add_argument('--accept', default='', help='fraction of passes in which draft 1, 2, 3 is accepted, e.g. 0.693,0.457,0.253')
A = ap.parse_args()
depths = [int(x) for x in A.depths.split(',')]; Ts = [int(x) for x in A.T.split(',')]; G = A.tokens
acc = [float(x) for x in A.accept.split(',')] if A.accept else []
if any(not 1 <= T <= MAX_T or G % T for T in Ts): raise SystemExit(f'--T: 1 .. {MAX_T}, each dividing --tokens {G}')
tk = BPE(ggufio.GGUFFile(A.gguf).meta)
text = ''.join(open(f).read() for f in sorted(glob.glob(os.path.join(ROOT, 'docs', '*.md')) + glob.glob(os.path.join(ROOT, 'docs', '*', '*.md')))
               + [os.path.join(ROOT, 'AGENTS.md'), os.path.join(ROOT, 'README.md')])
ids_all = tk.encode(text)
n_ctx = int(ggufio.GGUFFile(A.gguf).meta['qwen3moe.context_length'])
if max(depths) + G > min(n_ctx, len(ids_all)): raise SystemExit(f'--depths: the deepest + {G} tokens must fit {min(n_ctx, len(ids_all))} (the context, the docs\' tokens)')
print(torch.cuda.get_device_name(0), flush=True)
# beside the engine: a verify tape per T (and the traced one), the prefill (which projects its own buffers), and this
# script's reference logits and trace buffer
tapes = sum(Q3MoEVerify.need_mib(T) for T in Ts) + (Q3MoEVerify.need_mib(A.trace) if A.trace else 0)
eng = Q3MoEOne(A.gguf, max_ctx=min(n_ctx, max(depths) + G + 64), vram_extra_mib=tapes)
vers = {T: Q3MoEVerify(eng, T) for T in Ts}
vt = Q3MoEVerify(eng, A.trace, module='k3_q3m_ver_trace') if A.trace else None
own_mib = 8 + (vt.E.grid * len(vt.E.segs) * 8 * 8 / 2 ** 20 if vt else 0)
capacity.check([eng.dev], [own_mib], f'q3moe_verify: its trace buffer ({own_mib:.0f} MiB)')
pf = Q3MPrefill(eng, chunk=512)
print(f'engine {eng.vram_mib:.0f} MiB at max_ctx {eng.max_ctx}; {len(ids_all)} tokens of text; tapes: ' +
      ', '.join(f'T {T}: {len(v.E.segs)} segments' for T, v in vers.items()), flush=True)
ok = True
bits = lambda t: t.detach().cpu().numpy().view(np.uint32).copy()
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
OPN = {v: k[3:] for k, v in vars(k3).items() if k.startswith('OP_') and isinstance(v, int)}
def seg_name(sg):
    n = OPN.get(sg.op, str(sg.op))
    if n == 'OWN_MV':
        if (sg.pad1 & 255) & k3.OWN_GROUPED: return 'mv down' if sg.K == eng.ffn else 'mv gate|up'
        return 'mv ' + {eng.nh * eng.hd: 'q', eng.nkv * eng.hd: 'k|v', eng.D: 'o', eng.vocab: 'head'}.get(sg.N, str(sg.N))
    if n == 'OWN_ATTS': return 'attention row' if (sg.pad1 >> 8) & 1 else 'attention'
    return {'OWN_ADD_RMS16': 'add + norm', 'F32_MV': 'router matvec', 'NOP': 'join', 'OWN_EMB': 'embedding'}.get(n, n.lower())
# expected tokens per pass of T: its own token + the drafts accepted before the first refusal (T - 1 drafts at most)
exp_tok = lambda T: 1.0 + sum(acc[:T - 1])

for N in depths:
    toks = ids_all[N:N + G]
    eng.reset(); pf.run(ids_all[:N], 0, 0, N); eng.pos_host = N
    ref = [bits(eng.step(t)) for t in toks]
    dis = {}
    for T, ver in vers.items():
        bad = 0; d = []
        for b in range(G // T):
            ver.run(toks[b * T:(b + 1) * T], N + b * T); d.append(ver.distinct())
            bad += sum(int((bits(ver.logits(t)) != ref[b * T + t]).sum()) for t in range(T))
        dis[T] = float(np.mean(d)); ok &= bad == 0
        if bad: print(f'FAIL depth {N}, T {T}: {bad} logit words differ from the decode steps\'', flush=True)
    dec, res = [], {T: [] for T in Ts}
    for rd in range(A.rounds):
        eng.pos_host = N; ev[0].record()
        for t in toks: eng.step(t)
        ev[1].record(); torch.cuda.synchronize(); dec.append(ev[0].elapsed_time(ev[1]) / G)
        for T, ver in vers.items():
            nb = G // T; ev[0].record()
            for b in range(nb): ver.run(toks[b * T:(b + 1) * T], N + b * T)
            ev[1].record(); torch.cuda.synchronize(); res[T].append(ev[0].elapsed_time(ev[1]) / nb)
    d = float(np.median(dec[1:])); med = {T: float(np.median(v[1:])) for T, v in res.items()}
    print(f'depth {N:6d}: decode {d:.3f} ms per token | ' + ' | '.join(
        f'T {T}: {med[T]:.3f} ms per pass, {dis[T]:.1f} of {8 * T} experts distinct' +
        (f', {med[T] / exp_tok(T):.3f} ms per accepted token ({exp_tok(T):.3f} per pass)' if acc and T > 1 else '') for T in Ts), flush=True)
    if vt:
        E = vt.E; S = len(E.segs); Gd = E.grid; T = vt.T
        names = [seg_name(sg) for sg in E.segs]
        tr = torch.zeros(Gd * S * 8, dtype=torch.int64, device='cuda'); per_seg, totals = [], []
        for it in range(2 + G // T):
            b = it % (G // T); tr.zero_(); vt.run(toks[b * T:(b + 1) * T], N + b * T, trace=tr)
            if it < 2: continue
            Tm = tr.view(Gd, S, 8).cpu().numpy().astype(np.float64); t0 = Tm[:, :, 0][Tm[:, :, 0] > 0].min()
            st_, en_ = np.zeros(S), np.zeros(S)
            for s in range(S):
                ran = Tm[:, s, 4] < E.segs[s].ntask
                st_[s] = (Tm[ran, s, 1].min() - t0) / 100.0; en_[s] = (Tm[ran, s, 7].max() - t0) / 100.0
            dd = np.array([max((en_[x] for x in (E.segs[s].dep, E.segs[s].dep2p1 - 1) if x >= 0), default=0.0) for s in range(S)])
            per_seg.append((st_, en_, dd)); totals.append(en_.max())
        st_, en_, de = (np.median(np.stack([p[i] for p in per_seg]), 0) for i in range(3))
        path, s = [], int(np.argmax(en_))
        while s >= 0:                                       # back through each segment's later dependency
            path.append(s)
            deps = [x for x in (E.segs[s].dep, E.segs[s].dep2p1 - 1) if x >= 0]
            s = max(deps, key=lambda x: en_[x]) if deps else -1
        crit = collections.defaultdict(lambda: [0, 0.0, 0.0])
        for s in path:
            c = crit[names[s]]; c[0] += 1; c[1] += max(0.0, st_[s] - de[s]); c[2] += en_[s] - st_[s]
        print(f'  trace T {T} at depth {N}: {np.median(totals) / 1e3:.3f} ms per pass; {S} segments, {len(path)} on the critical path')
        for k, (n, h, sp) in sorted(crit.items(), key=lambda kv: -(kv[1][1] + kv[1][2])):
            print(f'    {k:14s} n {n:4d}  waits {h / 1e3:6.3f} ms  spans {sp / 1e3:7.3f} ms  ({sp / n:7.2f} us each)', flush=True)
        del tr
if eng.bails: print('bails:', eng.bails); ok = False
print('RESULT', 'PASS' if ok else 'FAIL')
