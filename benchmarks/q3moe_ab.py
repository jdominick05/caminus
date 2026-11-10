"""Q3MoEOne tape forms A/B on one engine's weights and KV caches (engine/q3moe_one.py): the same token sequence through
each form at the same positions (each form its own caches would differ: so one form runs a sequence, the caches are
reset by position, then the other), logits compared bitwise per position (G-self), then decode speed interleaved token
by token at different positions (median GPU time; --rounds rounds).
  tools/guard.sh LOG 900 python -I benchmarks/q3moe_ab.py GGUF [--tokens 64] [--rounds 3]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from caminus.engine.q3moe_one import Q3MoEOne
ap = argparse.ArgumentParser(); ap.add_argument('gguf'); ap.add_argument('--tokens', type=int, default=64)
ap.add_argument('--rounds', type=int, default=3, help='speed rounds (each: every form, interleaved token by token)')
ap.add_argument('--positions', type=int, default=24, help='positions of the bitwise comparison (the 24-token text repeated)')
ap.add_argument('--grids', type=int, nargs='+', default=None, help='arms = the default tape at these workgroups per WGP '
                '(the first is the base) instead of the tape forms'); A = ap.parse_args()
print(torch.cuda.get_device_name(0), flush=True)
ARMS = (('attn 5 seg', dict(attn_one=False)), ('attn 1 seg', dict(attn_one=True)))   # (name, Q3MoEOne tape options); the first is the base
if A.grids: ARMS = tuple((f'grid {g}', dict(per_wgp=g)) for g in A.grids)
# beside the engine's own projection: one further tape per arm after the first, and a position's buffers cloned for the
# comparison (x, h, ids, wts, logits: under 1 MiB, copied to the host at once)
eng = Q3MoEOne(A.gguf, max_ctx=max(2 * A.tokens * len(ARMS), A.positions) + 64,
               vram_extra_mib=(len(ARMS) - 1) * Q3MoEOne.TAPE_MIB + 2, **ARMS[0][1])
forms = {ARMS[0][0]: (eng.E, eng.S, eng.dstage, eng.hstage, eng.tok, eng.pos)}
for name, opt in ARMS[1:]:
    for k, v in opt.items(): setattr(eng, k, v)
    with torch.cuda.device(eng.dev): eng._build_tape()
    forms[name] = (eng.E, eng.S, eng.dstage, eng.hstage, eng.tok, eng.pos)
print('rows per task:', eng.RPT, flush=True)
def use(k): eng.E, eng.S, eng.dstage, eng.hstage, eng.tok, eng.pos = forms[k]
# (1) bitwise: the same 24 tokens from position 0 through each form (the KV caches are overwritten position by position);
#     each form twice (is a form bitwise itself?), the residual / norm / expert ids / logits kept per position
seq = [785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 9856, 374, 19846, 13, 576, 6722, 315]
seq = (seq * (A.positions // len(seq) + 1))[:A.positions]
KEYS = ('x', 'h', 'ids', 'wts', 'logits')
def run(k):
    use(k); eng.reset(); r = []
    for i, t in enumerate(seq):   # (every buffer at the first 24 positions, then the logits only)
        eng.step(t); r.append({b: eng.S[b].clone().cpu().numpy() for b in (KEYS if i < 24 else ('logits',))})
    return r
out = {k: run(k) for k in forms}; again = {k: run(k) for k in forms}
bit = lambda a, b: np.array_equal(a.view(np.uint32), b.view(np.uint32))
for k in forms:
    ok = all(bit(a[b], c[b]) for a, c in zip(out[k], again[k]) for b in a)
    print(('PASS' if ok else 'FAIL') + f' {k} bitwise itself on a second run', flush=True)
BASE = ARMS[0][0]
for name in list(forms)[1:]:
    eq = all(bit(a['logits'], b['logits']) for a, b in zip(out[BASE], out[name]))
    d = [float(np.abs(a['logits'].astype(np.float64) - b['logits'].astype(np.float64)).max()) for a, b in zip(out[BASE], out[name])]
    t1 = sum(int(a['logits'].argmax()) == int(b['logits'].argmax()) for a, b in zip(out[BASE], out[name]))
    print(f"  {name} vs {BASE}: {'bitwise' if eq else 'not bitwise'}; max |d logits| pos 0 {d[0]:.3e}, all {max(d):.3e}; top-1 equal {t1} / {len(seq)}", flush=True)
# (2) speed, interleaved token by token; A.rounds rounds, each from position 0 (the median of the rounds' medians)
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]; med = {k: [] for k in forms}
for rd in range(A.rounds):
    tm = {k: [] for k in forms}; tok = 785; eng.reset()
    for i in range(A.tokens):
        for k in forms:
            use(k); ev[0].record(); lg = eng.step(tok); ev[1].record(); torch.cuda.synchronize()
            tm[k].append(ev[0].elapsed_time(ev[1])); tok = int(torch.argmax(lg))
    for k, t in tm.items(): med[k].append(float(np.median(t[8:])))
    print(f'  round {rd}: ' + ', '.join(f'{k} {med[k][-1]:.3f}' for k in forms) + ' ms/token', flush=True)
for k in forms:
    m = float(np.median(med[k]))
    print(f'  {k:11s} {m:.3f} ms/token = {1e3 / m:.1f} tok/s (median of {A.rounds} rounds; min {min(med[k]):.3f}, max {max(med[k]):.3f})')
print('RESULT DONE')
