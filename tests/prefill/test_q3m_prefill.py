"""The qwen3moe batched prefill (caminus.prefill.q3m.Q3MPrefill) end to end against the float64 reference
(tools/reference/q3moe_ref.py: N teacher-forced positions of a public text):

  G-model    positions [0, P) prefilled, then P .. N - 1 one K3 pass each: KLD(ref || ours) and top-1 at those positions,
             against the decode-only engine at the same positions and against llama.cpp's own prefill-then-decode (its
             logits rows from build/gg_golden --lc 'tfx NAME TOKENS P N-P P 1 OUT', one card, f16 KV): the gate is mean
             KLD <= llama.cpp's and top-1 >= llama.cpp's (without --llama only the numbers are printed)
  geometry   the prefill in chunks of 16 and in one chunk, and with the prompt cache's stops: the same cache rows, bit for
             bit (every GEMM row and attention query is computed from its own inputs only), so the same logits after
  bounds     a span past max_ctx is refused before any write; the engine's poisoned flag refuses a prefill
  tools/guard.sh LOG 900 python -I tests/prefill/test_q3m_prefill.py GGUF --ref REF.npz [--llama LL.bin] [--from P]
New kernel code: a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np, torch
from caminus.device import capacity
from caminus.engine import k3
from caminus.engine.q3moe_one import Q3MoEOne
from caminus.prefill.driver import PrefillRefused
from caminus.prefill.q3m import Q3MPrefill

ap = argparse.ArgumentParser(); ap.add_argument('gguf'); ap.add_argument('--ref', required=True)
ap.add_argument('--llama', default=None); ap.add_argument('--from', dest='P', type=int, default=64)
ap.add_argument('--module', default=None)
A = ap.parse_args()
z = np.load(A.ref); seq, ref = [int(v) for v in z['seq']], z['logits']; N, P = len(seq), A.P
print(torch.cuda.get_device_name(0), flush=True)
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


def lsm(v):
    v = v.astype(np.float64); v = v - v.max(); return v - np.log(np.exp(v).sum())


def score(rows, cand):
    kl = [float(np.sum(np.exp(lsm(r)) * (lsm(r) - lsm(c)))) for r, c in zip(rows, cand)]
    return float(np.mean(kl)), float(np.percentile(kl, 90)), float(np.max(kl)), int(sum(np.argmax(r) == np.argmax(c) for r, c in zip(rows, cand)))


eng = Q3MoEOne(A.gguf, max_ctx=N + 64, module=A.module)
pf = Q3MPrefill(eng, chunk=128)
# beside the engine and the prefill (which project their own memory): three snapshots of the prefilled cache rows of
# every layer (the default chunking, chunks of 16, with stops), kept for the bitwise comparisons
snap_mib = 3 * eng.nl * P * 2 * eng.nkv * eng.hd * 2 / 2 ** 20 + 2
capacity.check([eng.dev], [snap_mib], f'test_q3m_prefill: its cache snapshots ({P} rows x {eng.nl} layers x 3: {snap_mib:.0f} MiB)')
cache_bytes = lambda: [L['cache'][:P * 2 * eng.nkv * eng.hd].clone() for L in eng.L]   # (the prefilled rows: later ones hold the last run's decode)


def decode_only():
    eng.reset(); return np.stack([eng.step(t).float().cpu().numpy() for t in seq])


def prefill_then_decode(chunk=None, stops=()):
    eng.reset()
    st = []
    pf.run(seq, 0, 0, P, chunk=chunk, stops=stops, at_stop=(lambda b, s: st.append(b)) if stops else None)
    rows = cache_bytes()
    eng.pos_host = P
    out = np.stack([eng.step(t).float().cpu().numpy() for t in seq[P:]])
    return out, rows, st


dec = decode_only()
ours, rows1, _ = prefill_then_decode()
so, sd = score(ref[P:], ours), score(ref[P:], dec[P:])
fmt = lambda s: f'mean KLD {s[0]:.3e} (p90 {s[1]:.3e}, max {s[2]:.3e}), top-1 {s[3]}/{N - P}'
print(f'  prefill [0, {P}) then decode: {fmt(so)}', flush=True)
print(f'  decode only, the same positions: {fmt(sd)}', flush=True)
if A.llama:
    ll = np.fromfile(A.llama, dtype=np.float32).reshape(N - P, -1)
    sl = score(ref[P:], ll)
    print(f'  llama.cpp prefill then decode: {fmt(sl)}', flush=True)
    check(so[0] <= sl[0], f'G-model: mean KLD vs float64 {so[0]:.3e} <= llama.cpp\'s prefill-then-decode {sl[0]:.3e}')
    check(so[3] >= sl[3], f'G-model: top-1 {so[3]} >= llama.cpp\'s {sl[3]}')
check(so[0] <= 2.0 * sd[0] + 1e-4, f'prefill-then-decode KLD within 2x the decode-only engine\'s at the same positions ({so[0]:.3e} vs {sd[0]:.3e})')
# geometry: chunks of 16, one chunk with the cache's stops
o16, rows16, _ = prefill_then_decode(chunk=16)
ost, rowsst, st = prefill_then_decode(stops=(16, 40, 41))
same = lambda ra, rb: all(torch.equal(x.view(torch.int16), y.view(torch.int16)) for x, y in zip(ra, rb))
check(same(rows1, rows16), 'chunks of 16: the same cache rows, bit for bit')
check(np.array_equal(o16, ours), 'chunks of 16: the same logits after, bit for bit')
check(same(rows1, rowsst) and np.array_equal(ost, ours) and st == [16, 40, 41], f'stops at 16, 40, 41 (at_stop {st}): the same rows and logits')
# bounds
eng.reset()
try:
    pf.run(seq * 2, 0, 0, eng.max_ctx + 1); check(False, 'a span past max_ctx was not refused')
except PrefillRefused as e:
    check(True, f'a span past max_ctx is refused before any write ({str(e)[:70]})')
eng.poisoned = 'test'
try:
    pf.run(seq, 0, 0, P); check(False, 'a poisoned engine was not refused')
except PrefillRefused:
    check(True, 'a poisoned engine refuses the prefill')
eng.poisoned = None
print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
