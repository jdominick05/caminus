"""The qwen3moe two-card batched prefill (caminus.prefill.q3m_tp.Q3MPrefillTP on Q3MoETP) on the cards.

--probe (ONE card: HIP_VISIBLE_DEVICES=<the non-display card>; run it first): the new device op alone, small inputs
  - q3p_slot_sum (kernels/pf_q3m_tp.hip) against a CPU emulation of its fp32 fma chain (exact: each fma emulated in
    float64 with round-to-odd, then rounded to fp32 once), bit for bit, at S 1 / 2 / 8, N 256 / 2048, 1 / 7 / 64 tokens
  - the same rows against the one-card prefill's add (q3p_add_rms16 of the slots onto a zeroed residual: the same fma
    chain, then 0 + it): equal value for value (a -0 / +0 difference excepted)
  - canary rows past the output and the inputs unchanged after every launch
Two cards (--tiny first: tests/engine/tiny_q3m.py's file with two-card shapes; then the GGUF), both visible, card 0 first:
  G-model    positions [0, P) prefilled on both cards, then P .. N - 1 one two-card K3 pass each: KLD(ref || ours) and top-1
             against the two-card decode-only engine at the same positions, and (--llama) against llama.cpp: its rows of
             the same positions (a decode-only file of N rows such as the 128-token reference's: its rows [P, N), a
             stricter bound than llama.cpp's own prefill-then-decode; a file of N - P rows: its prefill-then-decode)
             The gate: mean KLD <= llama.cpp's and top-1 >= llama.cpp's; prefill-then-decode within 2x decode-only + 1e-4.
             --tiny: the float64 host forward (caminus.ref_q3moe) is the reference; top-1 at least decode-only's, KLD < 1e-3
  replicas   after the prefill (its residual and routing ids) and after every decode pass that follows (the engine's)
  geometry   a GEOM-token prompt (several pieces a chunk): the default (overlap, 2+ pieces), one piece, serial, chunks of 16
             and the cache's stops give the same KV rows on both cards, bit for bit
  bounds     a span past max_ctx is refused before any write (the caches unchanged); the poisoned flag refuses a prefill
  speed      (--speed L) prompt tok/s from empty at chunk 2048, overlap and serial (medians of 3; prefill, two cards)
  --kv q8    the engine and prefill with Q8_0 KV rows (Q3MoETP(q8=True), Q3MPrefillTP's Q8 path: pf_q3m_kvq8's row and
             attention kernels at each card's local shapes): every check above on the byte caches (the caches start as
             0xFF bytes, a NaN half scale, and "written" means every block decodes finite), plus the rows: each card's
             last-layer rows of the prefill [0, P) byte for byte caminus.ref_kv.encode_q8_kv(hadamard128(K), V) of the halves
             the F16 row kernel (q3p_qk_rope) stores for that card's q | k | v (the chunk's own last-layer buffer), the
             one-card Q8 rows' encoding of the same values; both cards' KV error words 0
  HIP_VISIBLE_DEVICES=<XT> tools/guard.sh LOG 300 python -I tests/prefill/test_q3m_prefill_tp.py --probe
  CAMINUS_UPLOAD_LOG=ULOG tools/guard.sh LOG 900 python -I tests/prefill/test_q3m_prefill_tp.py --tiny --scratch DIR
  CAMINUS_UPLOAD_LOG=ULOG tools/guard.sh LOG 1800 python -I tests/prefill/test_q3m_prefill_tp.py GGUF --ref REF.npz [--llama LL.bin]
      [--from P] [--geom 600] [--speed L] [--kv q8]
Prints PASS / FAIL lines and RESULT PASS | FAIL."""
import argparse, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python')); sys.path.insert(0, os.path.join(HERE, '..', 'engine'))
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('gguf', nargs='?'); ap.add_argument('--probe', action='store_true'); ap.add_argument('--tiny', action='store_true')
ap.add_argument('--scratch', default=None); ap.add_argument('--ref', default=None); ap.add_argument('--llama', default=None)
ap.add_argument('--from', dest='P', type=int, default=64); ap.add_argument('--geom', type=int, default=600)
ap.add_argument('--speed', type=int, default=0)
ap.add_argument('--kv', default='f16', choices=('f16', 'q8'), help='the KV cache format (q8: Q3MoETP(q8=True), its Q8 prefill)')
A = ap.parse_args()
if sum([bool(A.probe), bool(A.tiny), bool(A.gguf)]) != 1: raise SystemExit('test_q3m_prefill_tp: --probe, --tiny or a GGUF (one of them)')
if not os.environ.get('CAMINUS_GUARD'): raise SystemExit('test_q3m_prefill_tp: run it under tools/guard.sh')
from caminus.device import capacity
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


def lsm(v):
    v = v.astype(np.float64); v = v - v.max(); return v - np.log(np.exp(v).sum())


def score(rows, cand):
    kl = [float(np.sum(np.exp(lsm(r)) * (lsm(r) - lsm(c)))) for r, c in zip(rows, cand)]
    return float(np.mean(kl)), float(np.percentile(kl, 90)), float(np.max(kl)), int(sum(np.argmax(r) == np.argmax(c) for r, c in zip(rows, cand)))


def fma32(a, b, c):
    """fp32 fma, exact: a b is exact in float64; a + b's error by TwoSum; round-to-odd in float64 (53 >= 2 x 24 + 2 bits), then
    one rounding to fp32"""
    a, b, c = (np.asarray(v, np.float32).astype(np.float64) for v in (a, b, c))
    p = a * b; s = p + c; bb = s - p; e = (p - (s - bb)) + (c - bb)
    even = (s.view(np.int64) & 1) == 0
    adj = (e != 0) & even
    s = np.where(adj, np.nextafter(s, np.where(e > 0, np.inf, -np.inf)), s)
    return s.astype(np.float32)


if A.probe:
    if torch.cuda.device_count() != 1:
        raise SystemExit(f'--probe: {torch.cuda.device_count()} cards visible: run it on one card alone (HIP_VISIBLE_DEVICES=<the non-display card>)')
    from caminus.prefill.q3m_tp import SlotSum
    from caminus.prefill import q3m_ops
    print(torch.cuda.get_device_name(0), flush=True)
    NMAX, SMAX, NT, CAN = 2048, 8, 64, 4096
    need = (4 * (NT * SMAX * NMAX + NT * SMAX + 2 * (NT * NMAX + CAN) + 2 * NT * NMAX) + 2 * NT * NMAX) / 2 ** 20 + 8
    capacity.check([0], [need], f'test_q3m_prefill_tp --probe buffers ({need:.0f} MiB)')
    dev = torch.device('cuda:0')
    ssum, O = SlotSum(dev), q3m_ops.Ops(dev, 'pf_q3m')
    g = np.random.default_rng(11)
    SENT = np.float32(-1234.5)
    y = torch.zeros(NT * NMAX + CAN, device=dev); xz = torch.zeros(NT * NMAX + CAN, device=dev)
    h16 = torch.zeros(NT * NMAX, dtype=torch.float16, device=dev)
    nw = torch.ones(NMAX, device=dev)
    for S in (1, 2, 8):
        for N in (256, 2048):
            for n in (1, 7, 64):
                d = (g.standard_normal((n, S, N)) * g.choice([1e-3, 1.0, 50.0], size=(n, S, 1))).astype(np.float32)
                d[0, 0, :4] = [0.0, -0.0, 3.0e38, -1.0e-30]   # (no denormal: the device's fp32 denormal mode is not under test)
                w = g.random((n, S)).astype(np.float32); w /= w.sum(1, keepdims=True)
                dd, wd = torch.from_numpy(d.reshape(-1)).to(dev), torch.from_numpy(w.reshape(-1)).to(dev)
                y.fill_(float(SENT)); xz.zero_(); xz[n * N:].fill_(float(SENT))
                ssum(dd, wd, S, y, n, N)
                O.add_rms16(xz, nw[:N], 1e-6, h16, n, N, d=dd, w=wd, S=S)
                torch.cuda.synchronize()
                acc = np.zeros((n, N), np.float32)
                for s in range(S): acc = fma32(w[:, s:s + 1], d[:, s, :], acc)
                yo = y.cpu().numpy(); xo = xz.cpu().numpy()
                got = yo[:n * N].reshape(n, N)
                same = np.array_equal(got.view(np.uint32), acc.view(np.uint32))
                eq_add = bool(np.all((got == xo[:n * N].reshape(n, N)) | (np.isnan(got) & np.isnan(xo[:n * N].reshape(n, N)))))
                can = bool(np.all(yo[n * N:] == SENT)) and bool(np.all(xo[n * N:] == SENT))
                inp = np.array_equal(dd.cpu().numpy(), d.reshape(-1)) and np.array_equal(wd.cpu().numpy(), w.reshape(-1))
                check(same and eq_add and can and inp, f'q3p_slot_sum S {S}, N {N}, {n} tokens: the fp32 fma chain\'s bits (CPU, exact); '
                      f'= q3p_add_rms16 onto a zeroed residual; canaries and inputs unchanged'
                      + ('' if same else f' ({int((got.view(np.uint32) != acc.view(np.uint32)).sum())} differ)'))
    print('RESULT', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)

# ---- two cards ------------------------------------------------------------------------------------------------------------
if torch.cuda.device_count() < 2: raise SystemExit(f'test_q3m_prefill_tp: {torch.cuda.device_count()} card(s) visible: two needed')
from caminus.engine.q3moe_tp import Q3MoETP
from caminus.prefill.driver import PrefillRefused
from caminus.prefill.q3m_tp import Q3MPrefillTP, need_tp_mib, tp_shapes, GRID
print(torch.cuda.get_device_name(0), '+', torch.cuda.get_device_name(1), flush=True)
if A.tiny:
    import tiny_q3m
    if not A.scratch: raise SystemExit('--tiny writes its file under --scratch DIR')
    os.makedirs(A.scratch, exist_ok=True)
    path = os.path.join(A.scratch, 'tiny-q3m-tp.gguf'); info = tiny_q3m.write(path, tp=True)
    g = np.random.default_rng(5); seq = g.integers(0, info['V'], 96).tolist()
    from caminus.ref_q3moe import Q3MoERef
    rm = Q3MoERef(path); ref = np.stack([rm.step(t) for t in seq])
else:
    if not A.ref: raise SystemExit('a GGUF needs --ref REF.npz')
    path = A.gguf
    z = np.load(A.ref); seq, ref = [int(v) for v in z['seq']], z['logits']
N, P = len(seq), A.P
if not 16 <= P < N: raise SystemExit(f'--from {P}: 16 .. {N - 1}')
GEOM = A.geom
max_ctx = max(N, GEOM, A.speed) + 64
CHUNK = 2048
sh = tp_shapes(path)
Q8 = A.kv == 'q8'
eng = Q3MoETP(path, max_ctx=max_ctx, vram_extra_mib=need_tp_mib(sh, CHUNK), q8=Q8)
pf = Q3MPrefillTP(eng, chunk=CHUNK)
print(f'module {eng.module}, KV {A.kv}, expert rows {eng.exp_rows}, kv heads {eng.kvc}; prefill chunk {CHUNK}, micro {pf.micro}, mode {pf.mode}; '
      f'prefill {pf.need_mib[0]:.0f} / {pf.need_mib[1]:.0f} MiB', flush=True)
from caminus.engine import k3
rb = [k3.kv_pos_bytes(k * eng.hd, Q8) for k in eng.kvc]                     # (bytes per position on each card)
row = [b // eng.L[0][c]['cache'].element_size() for c, b in enumerate(rb)]  # (cache elements per position: halves or bytes)
# beside the engine and the prefill: the geometry's snapshots of the prefilled cache rows (5 held at once) and the bounds check's;
# q8: the F16 row oracle's buffers (P rows of halves, the q16 rows)
snap_mib = [6 * eng.nl * GEOM * rb[c] / 2 ** 20 + 2 + ((P * 2 * eng.kvc[c] * eng.hd + P * eng.kvc[c] * 8 * eng.hd) * 2 / 2 ** 20 if Q8 else 0)
            for c in (0, 1)]
capacity.check(list(eng.devs), snap_mib, f'test_q3m_prefill_tp: its cache snapshots ({snap_mib[0]:.0f} / {snap_mib[1]:.0f} MiB)')
rep_bad = []


def kv(n):
    """both cards' cache rows [0, n) of every layer (copies)"""
    return [eng.L[il][c]['cache'][:n * row[c]].clone() for c in (0, 1) for il in range(eng.nl)]


def same(a, b):
    return all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(a, b))


def run_rows(toks):
    rows = []
    for t in toks:
        rows.append(eng.step(t).float().cpu().numpy())
        if not all(eng.check_replicas()): rep_bad.append(eng.pos_host - 1)
    return np.stack(rows)


def poison_kv():
    """NaN in every cache row of both cards: a prefill that skips or misplaces a row leaves NaN for the decode to read (q8:
    every byte 0xFF, so every block's half scale is a NaN)"""
    for il in range(len(eng.L)):
        for c in (0, 1): eng.L[il][c]['cache'].fill_(255 if Q8 else float('nan'))
    torch.cuda.synchronize(eng.devs[0]); torch.cuda.synchronize(eng.devs[1])


def written(rows_list):
    """every row of the snapshot holds finite values: the halves (F16) or each block's decoded values (Q8)"""
    if not Q8: return all(bool(torch.isfinite(x.float()).all()) for x in rows_list)
    from caminus import ref_kv
    for k, x in enumerate(rows_list):
        c = k // eng.nl   # (kv(): card 0's layers, then card 1's)
        dk, dv = ref_kv.decode_q8_kv(x.cpu().numpy().reshape(-1, rb[c]), ref_kv.Q8KVLayout(eng.kvc[c], eng.hd))
        if not (np.isfinite(dk).all() and np.isfinite(dv).all()): return False
    return True


def q8_rows_check():
    """after the default prefill of [0, P) (one chunk): each card's last-layer rows against encode_q8_kv(hadamard128(K), V)
    of the F16 row kernel's halves for that card's last-layer q | k | v (pf.B[c]['qkv']: the last layer stops after its rows)"""
    from caminus import ref_kv
    from caminus.prefill import q3m_ops
    good = True
    for c in (0, 1):
        s_ = pf.sh[c]; Mk = s_.nkv * s_.hd; ldq = s_.nh * s_.hd + 2 * Mk; lay = ref_kv.Q8KVLayout(s_.nkv, s_.hd)
        dev = pf.devs[c]
        with torch.cuda.device(dev):
            oops = q3m_ops.Ops(dev, 'pf_q3m')
            oc = torch.zeros(P * 2 * Mk, dtype=torch.float16, device=dev); oq = torch.zeros(P * s_.nh * s_.hd, dtype=torch.float16, device=dev)
            oops.qk_rope(pf.B[c]['qkv'], ldq, s_.nh, s_.nkv, eng.L[-1][c]['qk_par'], 0, oq, oc, P); torch.cuda.synchronize(dev)
        hh = oc.float().cpu().numpy().reshape(P, 2, s_.nkv, s_.hd)
        want = ref_kv.encode_q8_kv(ref_kv.hadamard128(np.ascontiguousarray(hh[:, 0])), np.ascontiguousarray(hh[:, 1]), lay)
        good &= np.array_equal(eng.L[-1][c]['cache'][:P * rb[c]].cpu().numpy().reshape(P, -1), want)
        good &= int(pf.B[c]['kverr'].item()) == 0
        del oc, oq
    check(good, f'Q8 rows: each card\'s last-layer {P} prefilled rows byte for byte encode_q8_kv(hadamard128(K), V) of the F16 row '
                f'kernel\'s halves for that card\'s q | k | v (the one-card Q8 rows\' encoding); both KV error words 0')


eng.reset(); dec = run_rows(seq)
eng.reset(); poison_kv(); pf.run(seq, 0, 0, P); torch.cuda.synchronize(eng.devs[0]); torch.cuda.synchronize(eng.devs[1])
prep = pf.check_replicas()
if Q8:
    check(P <= CHUNK and pf.q8 and pf.numerics().get('kv') == 'q8_0' and all(o.q8 for o in pf.O), 'the prefill runs the Q8 row and attention kernels')
    q8_rows_check()
eng.pos_host = P
ours = run_rows(seq[P:])
check(all(prep), f'after the prefill: the residual rows and the routing ids the same bits on both cards ({prep})')
check(not rep_bad and not eng.bails, 'the replicas (residual, every layer\'s router ids) equal after every decode pass, after a prefill too; 0 bails'
      + (f' (differ at {rep_bad[:8]})' if rep_bad else '') + (f' ({len(eng.bails)} bails)' if eng.bails else ''))
so, sd = score(ref[P:], ours), score(ref[P:], dec[P:])
fmt = lambda s: f'mean KLD {s[0]:.3e} (p90 {s[1]:.3e}, max {s[2]:.3e}), top-1 {s[3]}/{N - P}'
print(f'  two cards: prefill [0, {P}) then decode: {fmt(so)}', flush=True)
print(f'  two cards: decode only, the same positions: {fmt(sd)}', flush=True)
check(so[0] <= 2.0 * sd[0] + 1e-4, f'prefill-then-decode KLD within 2x the decode-only engine\'s + 1e-4 ({so[0]:.3e} vs {sd[0]:.3e})')
if A.tiny and Q8:   # (both runs on the Q8 cache: the top-1 comparison stands; 1e-3 is the F16 bar, reported only here)
    check(so[3] >= sd[3], f'tiny, Q8 KV: against the float64 forward top-1 at least the decode-only engine\'s ({so[3]} vs {sd[3]}); '
          f'mean KLD {so[0]:.3e} (reported only: the F16 bar is < 1e-3)')
elif A.tiny:
    check(so[3] >= sd[3] and so[0] < 1e-3, f'tiny: against the float64 forward top-1 at least the decode-only engine\'s ({so[3]} vs {sd[3]}) '
          f'and mean KLD < 1e-3 ({so[0]:.3e})')
if A.llama:
    ll = np.fromfile(A.llama, dtype=np.float32)
    rows_ll = ll.size // ref.shape[1]
    ll = ll.reshape(rows_ll, -1)
    if rows_ll == N: ll, what = ll[P:], 'one-token decode at the same positions (stricter than its prefill-then-decode)'
    elif rows_ll == N - P: what = 'prefill-then-decode'
    else: raise SystemExit(f'--llama: {rows_ll} rows (N {N} or N - P {N - P})')
    sl = score(ref[P:], ll)
    print(f'  llama.cpp {what}: {fmt(sl)}', flush=True)
    check(so[0] <= sl[0], f'G-model: mean KLD vs the reference {so[0]:.3e} <= llama.cpp\'s {sl[0]:.3e}')
    check(so[3] >= sl[3], f'G-model: top-1 {so[3]} >= llama.cpp\'s {sl[3]}')

# geometry: a prompt of GEOM tokens, several pieces a chunk
gtoks = (seq * (GEOM // N + 1))[:GEOM]


def geom_rows(chunk=None, stops=(), micro=None, mode=None, min_piece=None):
    m0 = (pf.micro, pf.mode, pf.min_piece)
    pf.micro = m0[0] if micro is None else micro; pf.mode = m0[1] if mode is None else mode
    pf.min_piece = m0[2] if min_piece is None else min_piece
    eng.reset(); poison_kv(); st = []
    try:
        pf.run(gtoks, 0, 0, GEOM, chunk=chunk, stops=stops, at_stop=(lambda b, s: st.append(b)) if stops else None)
        torch.cuda.synchronize(eng.devs[0]); torch.cuda.synchronize(eng.devs[1])
        return kv(GEOM), st
    finally:
        pf.micro, pf.mode, pf.min_piece = m0


from caminus.prefill.q3m_tp import micro_bounds
pieces = micro_bounds(GEOM, 4, min_piece=GRID)
r_ov, _ = geom_rows(micro=4, min_piece=GRID)
check(len(pieces) >= 2 and all(pf.check_replicas()), f'geometry: {GEOM} positions in one chunk of {len(pieces)} pieces {pieces} (overlap); replicas equal')
r_one, _ = geom_rows(micro=1)
check(written(r_ov),
      f'the {GEOM} positions\' KV rows are all written: finite on both cards in every layer (the caches start as NaN' + (' half scales)' if Q8 else ')'))
check(same(r_ov, r_one), 'one piece: the same KV rows on both cards, bit for bit')
r_ser, _ = geom_rows(micro=4, min_piece=GRID, mode='serial')
check(same(r_ov, r_ser), 'serial exchange: the same KV rows on both cards, bit for bit')
r_16, _ = geom_rows(chunk=16)
check(same(r_ov, r_16), 'chunks of 16: the same KV rows on both cards, bit for bit')
r_st, st = geom_rows(micro=4, min_piece=GRID, stops=(16, 40, 41, 300))
check(same(r_ov, r_st) and st == [16, 40, 41, 300], f'stops at 16, 40, 41, 300 (at_stop {st}): the same KV rows on both cards, bit for bit')
del r_one, r_ser, r_16, r_st

# bounds
eng.reset()
before = kv(GEOM)
try:
    pf.run(gtoks * (max_ctx // GEOM + 2), 0, 0, eng.max_ctx + 1); check(False, 'a span past max_ctx was not refused')
except PrefillRefused as e:
    check(same(before, kv(GEOM)), f'a span past max_ctx is refused before any write: the caches unchanged ({str(e)[:70]})')
del before
eng.poisoned = 'test'
try:
    pf.run(seq, 0, 0, P); check(False, 'a poisoned engine was not refused')
except PrefillRefused:
    check(True, 'a poisoned engine refuses the prefill')
eng.poisoned = None

if A.speed:
    L_ = A.speed; toks = (seq * (L_ // N + 1))[:L_]
    for mode in ('overlap', 'serial'):
        ts = []
        pf.mode = mode
        for _ in range(4):
            eng.reset(); pf._sync_devices(); t0 = time.perf_counter()
            pf.run(toks, 0, 0, L_); pf._sync_devices(); ts.append(time.perf_counter() - t0)
        med = float(np.median(ts[1:]))
        print(f'NOTE speed (prefill, two cards, {mode}, micro {pf.micro}): {L_} positions from empty at chunk {CHUNK}: {1e3 * med:.0f} ms, '
              f'{L_ / med:.0f} tok/s (runs {", ".join(f"{1e3 * t:.0f}" for t in ts[1:])} ms; first {1e3 * ts[0]:.0f})', flush=True)
    pf.mode = 'overlap'
print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
