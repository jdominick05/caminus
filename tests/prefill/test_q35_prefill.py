"""The qwen35 lines' own batched prefills end to end (the file's architecture picks them: qwen35, the dense line,
caminus.prefill.q35own.Q35OwnPrefill on Dense35Own: Swift 1.5 Q4_K_M; qwen35moe, caminus.prefill.q35moe.Q35MoEPrefill on
Moe35Own: Qwen3.6-35B-A3B, F16 or (--kv q8) Q8_0 KV) against the float64 reference (tools/reference/q35_ref.py: N teacher-forced positions of a public text):

  G-model    positions [0, P) prefilled, then P .. N - 1 one K3 pass each: KLD(ref || ours) and top-1 at those positions,
             against the decode-only engine at the same positions and against llama.cpp's own prefill-then-decode (its
             logits rows from build/gg_golden --lc 'tfx NAME TOKENS P N-P P 1 OUT', one card, f16 KV): the gate is mean
             KLD <= llama.cpp's and top-1 >= llama.cpp's (without --llama only the numbers are printed; --kv q8 compares
             with the Q8 decode-only engine only: llama.cpp's Q8 cache is not Caminus's rotated one)
  geometry   the prefill in chunks of 16 and in one chunk, and with the prompt cache's stops: the same cache rows and
             recurrent states, bit for bit, so the same logits after
  bounds     a span past max_ctx is refused before any write; the engine's poisoned flag refuses a prefill
  speed      (--speed L) prompt tok/s of a prefill of L positions from empty at chunk 2048 and 1024 (medians of 3; reported,
             not gated), and the slowest attention launch of the deepest chunk (every launch well under 200 ms)
  tools/guard.sh LOG 1800 python -I tests/prefill/test_q35_prefill.py GGUF --ref REF.npz [--llama LL.bin] [--from P] [--speed L]
                 [--kv f16|q8]
New kernel code: a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np, torch
from caminus.device import capacity
from caminus import ggufio
from caminus.engine import k3
from caminus.prefill.driver import PrefillRefused

ap = argparse.ArgumentParser(); ap.add_argument('gguf'); ap.add_argument('--ref', required=True)
ap.add_argument('--llama', default=None); ap.add_argument('--from', dest='P', type=int, default=64)
ap.add_argument('--speed', type=int, default=0); ap.add_argument('--kv', default='f16', choices=('f16', 'q8'))
A = ap.parse_args()
Q8 = A.kv == 'q8'
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


max_ctx = max(N + 64, A.speed + 64)
CHUNK = 2048 if A.speed else 128
_F = ggufio.GGUFFile(A.gguf); ARCH = _F.arch; _F.close()
if ARCH == 'qwen35moe':
    from caminus.engine.q35_moe import Moe35Own as Engine
    from caminus.prefill.q35moe import Q35MoEPrefill as Prefill, buffer_need_mib, shapes_of
elif ARCH == 'qwen35':
    from caminus.engine.q35_dense import Dense35Own as Engine
    from caminus.prefill.q35own import Q35OwnPrefill as Prefill, buffer_need_mib, shapes_of
else: sys.exit(f'test_q35_prefill: architecture {ARCH!r} (qwen35, qwen35moe)')
if Q8 and ARCH != 'qwen35moe': sys.exit('test_q35_prefill: --kv q8 is the qwen35moe line\'s')
eng = Engine(A.gguf, max_ctx=max_ctx, vram_extra_mib=buffer_need_mib(shapes_of(A.gguf), CHUNK), **({'q8': True} if Q8 else {}))
pf = Prefill(eng, chunk=CHUNK)
NVSS = eng.NV * eng.S * eng.S
attn_L = [L for L in eng.L if L['kind'] == 'attn']; gdn_L = [L for L in eng.L if L['kind'] == 'gdn']
conv_n = gdn_L[0]['conv_st'].numel()
POS_B = k3.kv_pos_bytes(eng.nkv * eng.hd, Q8)          # (a cache position's bytes)
snap_mib = 3 * (len(attn_L) * P * POS_B + len(gdn_L) * (NVSS + conv_n) * 4) / 2 ** 20 + 2
capacity.check([eng.dev], [snap_mib], f'test_q35_prefill: its state snapshots ({snap_mib:.0f} MiB)')


def state_bytes():
    """the prefilled cache rows of every attention layer and the recurrent states of every GDN layer (later cache rows hold
    the last run's decode)"""
    return ([L['cache'].view(torch.uint8)[:P * POS_B].clone() for L in attn_L] +
            [L['conv_st'].clone() for L in gdn_L] + [L['gdn'][:NVSS].clone() for L in gdn_L])


def decode_only():
    eng.reset(); return np.stack([eng.step(t).float().cpu().numpy() for t in seq])


def prefill_then_decode(chunk=None, stops=()):
    eng.reset()
    st = []
    pf.run(seq, 0, 0, P, chunk=chunk, stops=stops, at_stop=(lambda b, s: st.append(b)) if stops else None)
    rows = state_bytes()
    eng.pos_host = P
    out = np.stack([eng.step(t).float().cpu().numpy() for t in seq[P:]])
    return out, rows, st


dec = decode_only()
ours, rows1, _ = prefill_then_decode()
so, sd = score(ref[P:], ours), score(ref[P:], dec[P:])
fmt = lambda s: f'mean KLD {s[0]:.3e} (p90 {s[1]:.3e}, max {s[2]:.3e}), top-1 {s[3]}/{N - P}'
print(f'  prefill [0, {P}) then decode: {fmt(so)}', flush=True)
print(f'  decode only, the same positions: {fmt(sd)}', flush=True)
if A.llama and not Q8:
    ll = np.fromfile(A.llama, dtype=np.float32).reshape(N - P, -1)
    sl = score(ref[P:], ll)
    print(f'  llama.cpp prefill then decode: {fmt(sl)}', flush=True)
    check(so[0] <= sl[0], f'G-model: mean KLD vs float64 {so[0]:.3e} <= llama.cpp\'s prefill-then-decode {sl[0]:.3e}')
    check(so[3] >= sl[3], f'G-model: top-1 {so[3]} >= llama.cpp\'s {sl[3]}')
check(so[0] <= 2.0 * sd[0] + 1e-4, f'prefill-then-decode KLD within 2x the decode-only engine\'s at the same positions + 1e-4 ({so[0]:.3e} vs {sd[0]:.3e})')
o16, rows16, _ = prefill_then_decode(chunk=16)
ost, rowsst, st = prefill_then_decode(stops=(16, 40, 41))
same = lambda ra, rb: all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(ra, rb))   # (bit for bit)
check(same(rows1, rows16), 'chunks of 16: the same cache rows and recurrent states, bit for bit')
check(np.array_equal(o16, ours), 'chunks of 16: the same logits after, bit for bit')
check(same(rows1, rowsst) and np.array_equal(ost, ours) and st == [16, 40, 41], f'stops at 16, 40, 41 (at_stop {st}): the same states and logits')
eng.reset()
try:
    pf.run(seq * (max_ctx // N + 2), 0, 0, eng.max_ctx + 1); check(False, 'a span past max_ctx was not refused')
except PrefillRefused as e:
    check(True, f'a span past max_ctx is refused before any write ({str(e)[:70]})')
eng.poisoned = 'test'
try:
    pf.run(seq, 0, 0, P); check(False, 'a poisoned engine was not refused')
except PrefillRefused:
    check(True, 'a poisoned engine refuses the prefill')
eng.poisoned = None
if A.speed:
    L_ = A.speed; toks = (seq * (L_ // N + 1))[:L_]
    # first the slowest attention launch: the deepest rows of the last chunk (one launch walks every position below its
    # rows; the cache's contents do not change its time). The full prefills only if it is well under the 200 ms bound
    O, L0 = pf.O, attn_L[0]
    n = min(pf.chunk, L_); p = L_ - n; r = O.attn_blocks(p, n)[-1]
    capacity.check([eng.dev], [2 * n * eng.nh * eng.hd * 4 / 2 ** 20 + 2], 'test_q35_prefill: the attention timing buffers')
    q = torch.randn(n * eng.nh * eng.hd, device=f'cuda:{eng.dev}'); out = torch.zeros_like(q)
    ts = []
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        O.attn_wm(q, L0['cache'], p, n, eng.nh, eng.nkv, eng.hd, 1.0 / np.sqrt(eng.hd), out, eng.max_ctx, *r)
        torch.cuda.synchronize(); ts.append(1e3 * (time.perf_counter() - t0))
    dt = sorted(ts)[1]; del q, out
    check(dt < 100.0, f'the slowest attention launch (blocks {r[0]} .. {r[1] - 1} of a chunk at {p}, up to position {L_ - 1}): {dt:.1f} ms (< 100)')
    if dt < 100.0:
        for chunk in (2048, 1024):
            if chunk > pf.chunk: continue
            ts = []
            for _ in range(3):
                eng.reset(); torch.cuda.synchronize(); t0 = time.perf_counter()
                pf.run(toks, 0, 0, L_, chunk=chunk); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
            med = sorted(ts)[1]
            print(f'NOTE speed (prefill): {L_} positions from empty at chunk {chunk}: {1e3 * med:.0f} ms, {L_ / med:.0f} tok/s '
                  f'(runs {", ".join(f"{1e3 * t:.0f}" for t in ts)} ms)', flush=True)
print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
