"""Q3MoEOne tape forms A/B on one engine's weights and KV caches (engine/q3moe_one.py): the same token sequence through
each form at the same positions (each form its own caches would differ: so one form runs a sequence, the caches are
reset by position, then the other), logits compared bitwise per position (G-self), then decode speed interleaved token
by token at different positions (median GPU time).
  tools/guard.sh LOG 900 python -I benchmarks/q3moe_ab.py GGUF [--tokens 64]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from caminus.engine.q3moe_one import Q3MoEOne
ap = argparse.ArgumentParser(); ap.add_argument('gguf'); ap.add_argument('--tokens', type=int, default=64); A = ap.parse_args()
print(torch.cuda.get_device_name(0), flush=True)
ARMS = (('3-stage/16', False, 16), ('fused/16', True, 16), ('fused/auto', True, 'auto'))
eng = Q3MoEOne(A.gguf, max_ctx=2 * A.tokens * len(ARMS) + 64, fused=ARMS[0][1], rpt=ARMS[0][2])
forms = {ARMS[0][0]: (eng.E, eng.S, eng.dstage, eng.hstage, eng.tok, eng.pos)}
for name, fused, rpt in ARMS[1:]:
    eng.fused, eng.rpt = fused, rpt
    with torch.cuda.device(eng.dev): eng._build_tape()
    forms[name] = (eng.E, eng.S, eng.dstage, eng.hstage, eng.tok, eng.pos)
print('rows per task:', eng.RPT, flush=True)
def use(k): eng.E, eng.S, eng.dstage, eng.hstage, eng.tok, eng.pos = forms[k]
# (1) bitwise: the same 24 tokens from position 0 through each form (the KV caches are overwritten position by position);
#     each form twice (is a form bitwise itself?), the residual / norm / expert ids / logits kept per position
seq = [785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 9856, 374, 19846, 13, 576, 6722, 315]
KEYS = ('x', 'h', 'ids', 'wts', 'logits')
def run(k):
    use(k); eng.reset(); r = []
    for t in seq:
        eng.step(t); r.append({b: eng.S[b].clone().cpu().numpy() for b in KEYS})
    return r
out = {k: run(k) for k in forms}; again = {k: run(k) for k in forms}
bit = lambda a, b: np.array_equal(a.view(np.uint32), b.view(np.uint32))
for k in forms:
    ok = all(bit(a[b], c[b]) for a, c in zip(out[k], again[k]) for b in KEYS)
    print(('PASS' if ok else 'FAIL') + f' {k} bitwise itself on a second run', flush=True)
BASE = ARMS[0][0]
for name in list(forms)[1:]:
    eq = all(bit(a['logits'], b['logits']) for a, b in zip(out[BASE], out[name]))
    d = [float(np.abs(a['logits'].astype(np.float64) - b['logits'].astype(np.float64)).max()) for a, b in zip(out[BASE], out[name])]
    t1 = sum(int(a['logits'].argmax()) == int(b['logits'].argmax()) for a, b in zip(out[BASE], out[name]))
    print(f"  {name} vs {BASE}: {'bitwise' if eq else 'not bitwise'}; max |d logits| pos 0 {d[0]:.3e}, all {max(d):.3e}; top-1 equal {t1} / {len(seq)}", flush=True)
# (2) speed, interleaved
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]; tm = {k: [] for k in forms}; tok = 785
eng.reset()
for i in range(A.tokens):
    for k in forms:
        use(k); ev[0].record(); lg = eng.step(tok); ev[1].record(); torch.cuda.synchronize()
        tm[k].append(ev[0].elapsed_time(ev[1])); tok = int(torch.argmax(lg))
for k, t in tm.items():
    t = t[8:]; print(f'  {k:11s} {np.median(t):.3f} ms/token = {1e3 / np.median(t):.1f} tok/s (p10 {np.percentile(t, 10):.3f}, p90 {np.percentile(t, 90):.3f})')
print('RESULT DONE')
