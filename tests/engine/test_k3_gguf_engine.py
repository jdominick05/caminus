"""GGUF ops in the persistent K3 engine (build/k3_gguf.hsaco): on real Flash-Next REAP-308-UI layer-0 tensors, the
engine's q8_1 bytes and matvec outputs (IQ3_XXS gate/up and IQ4_NL down of 10 experts, Q6_K attn_qkv, Q8_0 hc_attn_down)
are bitwise equal to the standalone kernels, independent and chained; then the speed of a chained expert tape that
reads distinct experts (~22 MB per pseudo-layer, beyond the Infinity Cache).
  HIP_VISIBLE_DEVICES=0 tools/guard.sh LOG 600 python -I tests/engine/test_k3_gguf_engine.py [model.gguf]"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
import torch
from caminus.engine import gguf as G, k3

from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
PATH = sys.argv[1] if len(sys.argv) > 1 else devpaths.reap()
r = G.GGUFReader(PATH, 'r'); T = {t.name: t for t in r.tensors}
torch.manual_seed(0); ok = True


def up(name, slab=0):
    t = T[name]; d = np.asarray(t.data); rows = d[slab] if d.ndim == 3 else d
    return torch.from_numpy(np.ascontiguousarray(rows)).cuda(), rows.shape[0], int(t.shape[0]), rows.shape[1], t.tensor_type.name


experts = list(range(10))
W = {('gate', e): up('blk.0.ffn_gate_exps.weight', e) for e in experts}
W.update({('up', e): up('blk.0.ffn_up_exps.weight', e) for e in experts})
W.update({('down', e): up('blk.0.ffn_down_exps.weight', e) for e in experts})
W[('qkv', 0)] = up('blk.0.attn_qkv.weight'); W[('hc', 0)] = up('blk.0.hc_attn_down.weight')
X = {k: torch.randn(n, device='cuda') * 0.5 for k, n in (('d', 2560), ('f', 640), ('h', 10240))}
need = {'gate': 'd', 'up': 'd', 'qkv': 'd', 'down': 'f', 'hc': 'h'}

for chained in (False, True):
    e = k3.Engine(module='k3_gguf'); yq, outs = {}, {}; prev = -1
    for k, x in X.items():
        yq[k] = torch.empty((x.numel() + 31) // 32 * 36, dtype=torch.uint8, device='cuda')
        prev = e.gg_q8_1(x, yq[k], dep=prev if chained else -1)
    for key, (w, nrows, ncols, rb, fmt) in W.items():
        outs[key] = torch.empty(nrows, device='cuda')
        prev = e.gg_matvec(w, fmt, nrows, ncols, rb, yq[need[key[0]]], outs[key], rows_per_task=16, dep=prev if chained else -1)
    e.build().run()
    qok = all(torch.equal(yq[k], G.quantize_q8_1(X[k])) for k in X)
    bad = [key for key, (w, nrows, ncols, rb, fmt) in W.items()
           if not torch.equal(outs[key], G.matvec(w, G.QT[fmt], nrows, ncols, G.quantize_q8_1(X[need[key[0]]])))]
    ok &= qok and not bad
    print(f'{"chained" if chained else "independent"}: q8_1 bytes {"equal" if qok else "DIFFER"}; {len(W) - len(bad)}/{len(W)} matvecs bitwise equal{" " + str(bad) if bad else ""}')

# speed: 30 pseudo-layers x [q8_1 -> 20 gate/up -> q8_1 -> 10 down], distinct experts per pseudo-layer
L = 30; e = k3.Engine(module='k3_gguf'); prev = -1; byts = 0
ws = {}
for l in range(L):
    for kind in ('gate', 'up', 'down'):
        for j in range(10):
            ws[(kind, l, j)] = up(f'blk.0.ffn_{kind}_exps.weight', (l * 10 + j) % 308)
xd = torch.randn(2560, device='cuda'); xf = torch.randn(640, device='cuda')
for l in range(L):
    yd = torch.empty(80 * 36, dtype=torch.uint8, device='cuda'); yf = torch.empty(20 * 36, dtype=torch.uint8, device='cuda')
    q = e.gg_q8_1(xd, yd, dep=prev); last = []
    for kind in ('gate', 'up'):
        for j in range(10):
            w, nrows, ncols, rb, fmt = ws[(kind, l, j)]; byts += w.numel()
            last.append(e.gg_matvec(w, fmt, nrows, ncols, rb, yd, torch.empty(nrows, device='cuda'), rows_per_task=16, dep=q))
    q2 = e.gg_q8_1(xf, yf, dep=last[-1])
    for j in range(10):
        w, nrows, ncols, rb, fmt = ws[('down', l, j)]; byts += w.numel()
        prev = e.gg_matvec(w, fmt, nrows, ncols, rb, yf, torch.empty(nrows, device='cuda'), rows_per_task=32, dep=q2)
e.build(); e.run(); torch.cuda.synchronize()
t0 = time.perf_counter(); n = 10
for _ in range(n): e.run(check=False)
torch.cuda.synchronize(); us = (time.perf_counter() - t0) / n * 1e6; e.run()
print(f'{torch.cuda.get_device_name(0)}: {L} chained pseudo-layers of 30 expert matvecs, {byts / 1e6:.0f} MB: {us:.0f} us = {us / L:.1f} us per layer, {byts / us / 1e3:.0f} GB/s')
print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
