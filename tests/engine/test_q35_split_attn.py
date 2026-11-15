"""The dense qwen35 own modules' decode attention (the split pair of kernels/k3_attn_split.h, built into k3_q35_swift /
k3_q35_swift_ver / k3_q35_tcap by kernels/k3_q35_ops.h under K3_Q35_ATTN_SPLIT) on one attention layer's shapes (24 query
heads over 4 kv heads, head dim 256, f16 cache), a tape of just the pair on random q and cache rows:
  - bitwise the GGUF module's pair on the same inputs (k3_gguf_q35d for T = 1, k3_gguf_q35d_mt for T = 2, 3, 4: the same
    source, its PART unpacked), at every list length tried: the verify module k3_q35_swift_ver runs the packed PART (one
    task per (split, kv head) for every pass row: kernels/k3_attn_split.h op_attn_part_packed) and must give its bits;
    lengths 34 / 35 / 4098 put a split under only some of a pass's rows
  - a T-row pass's row r bitwise the one-row pass at position pos + r (the split layout depends only on the list length)
  - the packed PART refuses T above k3.ATTN_PACK_T and a QSA selection (identity lists only)
  - against float64 attention: max |difference| <= 1e-5 x max |V|
  - the q8_0 cache refused on the F16 own modules (their device code reads f16 only)
--kv q8 runs the same pair on the separate Q8 objects (k3_q35_{swift,tcap,gsq}_kvq8, the verify objects
k3_q35_{swift,tcap}_ver_kvq8 with the unpacked PART) over a q8_0 cache that caminus.ref_kv encodes on the host:
  - bitwise the GGUF module's pair on the same cache bytes (k3_gguf_q35d with T = 1, k3_gguf_q35d_mt with T = 2, 3, 4)
  - a T-row pass's row r bitwise the one-row pass at position pos + r
  - against float64 attention over the same decoded rows (caminus.ref_kv.decode_q8_kv): max |difference| <= 1e-5 x
    max |V|. The bound is the F16 arm's: an int8 code times a half scale is exact in fp32, so the device reads the
    reference's values exactly; what is left is the same fp32 score, softmax and sum path. Not yet measured for Q8.
  - the Q8 objects' own Hadamard and append ops (kernels/k3_q35_kvq8_ops.h) on known K / V, K256 and V64 as the tapes
    run them, T 1 on the decode objects and T 2 / 3 / 4 on the verify objects: the transformed rows bitwise a host fp32
    Sylvester transform in the op's order, the cache bytes bitwise caminus.ref_kv.encode_q8_kv of those rows (only rows
    p0 .. p0 + T - 1 written); the output H64 once and twice bitwise the host transform (twice is the input within
    1e-6 x max |x| in fp32); a NaN in K raises InvalidKVEncoding from the sticky error word
  - the Q8 objects refuse an f16 cache, a QSA selection and T 5 (the attention arms above run no rotation)
  tools/guard.sh LOG 600 python -I tests/engine/test_q35_split_attn.py [--lens 1,2,31,32,33,34,35,1000,4096,4098,16389] [--kv q8]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>). Prints PASS / FAIL and RESULT."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np, torch
from caminus.device import capacity
from caminus.engine import k3

ap = argparse.ArgumentParser(); ap.add_argument('--lens', default='1,2,31,32,33,34,35,1000,4096,4098,16389')
ap.add_argument('--kv', choices=('f16', 'q8'), default='f16', help='q8: the separate Q8 objects on a q8_0 cache')
A = ap.parse_args()
Q8 = A.kv == 'q8'
H, NKV, D, G = 24, 4, 256, 6
LENS = [int(v) for v in A.lens.split(',')]
ROWS = max(LENS) + 4
SCALE = 1.0 / 16.0
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


need = (ROWS * k3.kv_pos_bytes(NKV * D, Q8) + 4 * H * ((ROWS + 31) // 32) * k3.ATTN_RS * 4 * 2 + 64 * H * D * 4) / 2 ** 20 + 64
capacity.check([0], [need], f'test_q35_split_attn ({need:.0f} MiB)')
print(torch.cuda.get_device_name(0), f'| {H} q heads / {NKV} kv heads x {D}, cache rows {ROWS}, {A.kv} cache', flush=True)
g = torch.Generator(device='cuda').manual_seed(7)
if Q8:   # (the rows encoded on the host by the CPU reference; the reference reads them back decoded)
    from caminus import ref_kv
    lay = ref_kv.Q8KVLayout(NKV, D)
    rng = np.random.default_rng(7)
    raw = np.empty((ROWS, lay.row_bytes), dtype=np.uint8)
    for r0 in range(0, ROWS, 2048):
        r1 = min(ROWS, r0 + 2048)
        kx = rng.normal(0.0, 0.5, (r1 - r0, NKV, D)).astype(np.float32)
        vx = rng.normal(0.0, 0.5, (r1 - r0, NKV, D)).astype(np.float32)
        raw[r0:r1] = ref_kv.encode_q8_kv(kx, vx, lay)
    kd, vd = ref_kv.decode_q8_kv(raw, lay)
    kv = np.stack((kd, vd), axis=1).astype(np.float64)   # [rows][K | V][kv head][D], the decoded q8_0 values
    del kd, vd
    cache = torch.from_numpy(raw.reshape(-1)).to('cuda')
else:
    cache = torch.empty(ROWS * 2 * NKV * D, dtype=torch.float16, device='cuda').normal_(0.0, 0.5, generator=g)   # (in place: no fp32 temporaries)
    kv = cache.view(ROWS, 2, NKV, D).cpu().double().numpy()   # (to the host first: no float64 copy on the card)
vmax = float(np.abs(kv[:, 1]).max())


def run(module, T, pos, q):
    """the pair on `module` with T rows from position pos: out fp32 [T][H][D] (a fresh engine and records)"""
    E = k3.Engine(module=module, grid=k3.resident_grid(module), T=T, limit_us=500_000)
    out = torch.zeros(T * H * D, device='cuda'); p = torch.full((1,), pos, dtype=torch.int32, device='cuda')
    E.attn_dec(q, cache, p, out, H, NKV, D, SCALE, q8=Q8, max_cells=ROWS)
    E.build(); E.run(check=True); torch.cuda.synchronize()
    return out.view(T, H, D).clone()


def ref(qrow, pos):
    """float64 attention of one row at position pos: [H][D]"""
    qq = qrow.double().cpu().numpy().reshape(H, D); o = np.zeros((H, D))
    for h in range(H):
        gk = h // G
        s = kv[:pos + 1, 0, gk] @ qq[h] * SCALE; s = np.exp(s - s.max()); o[h] = (s / s.sum()) @ kv[:pos + 1, 1, gk]
    return o


def sylvester(x, n):
    """the host fp32 Walsh-Hadamard transform of each chunk of n values, in the device op's order (kernels/k3_q35_kvq8_ops.h
    q35_op_hadamard: for h = 1, 2, 4 ..: a + b and a - b at i, i + h; then x 1/sqrt(n), exact for n = 64 / 256)"""
    y = np.array(x, dtype=np.float32).reshape(-1, n)
    h = 1
    while h < n:
        z = y.reshape(-1, n // (2 * h), 2, h)
        a, b = z[:, :, 0, :].copy(), z[:, :, 1, :].copy()
        z[:, :, 0, :] = a + b; z[:, :, 1, :] = a - b
        h *= 2
    return (y * np.float32(1.0 / np.sqrt(n))).reshape(np.shape(x))


def refuses(fn, want):
    """True when fn raises a ValueError whose text holds `want` (not another ValueError on the way)"""
    try:
        fn(); return False
    except ValueError as ex:
        if want not in str(ex): print(f'  (a different ValueError: {ex})', flush=True)
        return want in str(ex)


if Q8:
    ONE, VER = ('k3_q35_swift_kvq8', 'k3_q35_tcap_kvq8', 'k3_q35_gsq_kvq8'), ('k3_q35_swift_ver_kvq8', 'k3_q35_tcap_ver_kvq8')
    bit, rows_ok, err = True, True, 0.0
    for n in LENS:
        pos = n - 1
        q1 = torch.randn(H * D, generator=g, device='cuda') * 0.1
        a = run(ONE[0], 1, pos, q1)
        bit &= torch.equal(a, run('k3_gguf_q35d', 1, pos, q1)) and all(torch.equal(a, run(m, 1, pos, q1)) for m in ONE[1:])
        err = max(err, float(np.abs(a[0].double().cpu().numpy() - ref(q1, pos)).max()))
        for T in (2, 3, 4):
            if n < T: continue
            qT = torch.randn(T * H * D, generator=g, device='cuda') * 0.1
            p0 = n - T
            bT = run('k3_gguf_q35d_mt', T, p0, qT)
            for m in VER:
                aT = run(m, T, p0, qT)
                bit &= torch.equal(aT, bT)
                for r in range(T):
                    rows_ok &= torch.equal(aT[r], run(ONE[0], 1, p0 + r, qT[r * H * D:(r + 1) * H * D].contiguous())[0])
        print(f'  list length {n}: max |own Q8 - float64| so far {err:.2e}', flush=True)
    check(bit, f'the Q8 objects\' pair (T 1: {", ".join(ONE)}; T 2 / 3 / 4: {", ".join(VER)}, unpacked PART) is bitwise '
               f'k3_gguf_q35d\'s / k3_gguf_q35d_mt\'s on the same q8_0 bytes at list lengths {LENS}')
    check(rows_ok, 'a Q8 verify pass\'s rows (T = 2 / 3 / 4) are bitwise the one-row Q8 passes at their positions')
    check(err <= 1e-5 * vmax, f'against float64 attention over the same decoded q8_0 rows: max |difference| {err:.2e} '
                              f'<= 1e-5 x max |V| ({vmax:.2f})')

    # the Q8 objects' own Hadamard and append ops (kernels/k3_q35_kvq8_ops.h) on known K / V, as the tapes run them:
    # K256 and V64 in place, then the append of the T rows at p0
    def append_run(module, T, p0, kx, vx, rows=8):
        E = k3.Engine(module=module, grid=k3.resident_grid(module), T=T, limit_us=500_000)
        kd = torch.from_numpy(kx.reshape(-1).copy()).to('cuda'); vd = torch.from_numpy(vx.reshape(-1).copy()).to('cuda')
        c = torch.zeros(rows * k3.kv_pos_bytes(NKV * D, True), dtype=torch.uint8, device='cuda')
        p = torch.full((1,), p0, dtype=torch.int32, device='cuda')
        hk = E.hadamard(kd, 256); hv = E.hadamard(vd, 64)
        E.also(E.kv_append(kd, vd, c, p, q8=True, dep=hk), hv)
        E.build(); E.run(check=True); torch.cuda.synchronize()
        return c.cpu().numpy().reshape(rows, -1), kd.cpu().numpy(), vd.cpu().numpy()

    def h_run(module, x, n, times):
        """`times` in-place transforms of x (fp32) in chunks of n, one segment each, in order"""
        E = k3.Engine(module=module, grid=k3.resident_grid(module), T=1, limit_us=500_000)
        xd = torch.from_numpy(x.copy()).to('cuda'); s = -1
        for _ in range(times): s = E.hadamard(xd, n, dep=s)
        E.build(); E.run(check=True); torch.cuda.synchronize()
        return xd.cpu().numpy()

    rng = np.random.default_rng(11)
    app_ok, hk_ok = True, True
    for m, T, p0 in [(o, 1, 5) for o in ONE] + [(v, T, 3) for v in VER for T in (2, 3, 4)]:
        kx = rng.normal(0.0, 0.5, (T, NKV, D)).astype(np.float32); vx = rng.normal(0.0, 0.5, (T, NKV, D)).astype(np.float32)
        kh, vh = sylvester(kx, 256), sylvester(vx, 64)
        want = np.zeros((8, lay.row_bytes), dtype=np.uint8); want[p0:p0 + T] = ref_kv.encode_q8_kv(kh, vh, lay)
        got, kd_, vd_ = append_run(m, T, p0, kx, vx)
        hk_ok &= np.array_equal(kd_.view(np.uint32), kh.reshape(-1).view(np.uint32)) and np.array_equal(vd_.view(np.uint32), vh.reshape(-1).view(np.uint32))
        app_ok &= np.array_equal(got, want)
    check(hk_ok, 'the Q8 objects\' Hadamard op (K 256, V 64) is bitwise the host fp32 Sylvester transform (same butterflies, x 1/sqrt(n))')
    check(app_ok, 'the Q8 append writes ref_kv.encode_q8_kv\'s bytes of the transformed rows at p0 .. p0 + T - 1 and nothing else '
                  f'(T 1 on {", ".join(ONE)}; T 2 / 3 / 4 on {", ".join(VER)})')
    x = rng.normal(0.0, 1.0, 4 * H * D).astype(np.float32)   # (four pass rows of attention output)
    h1 = sylvester(x, 64); h2 = sylvester(h1, 64); rt = float(np.abs(h2 - x).max() / np.abs(x).max())
    check(all(np.array_equal(h_run(m, x, 64, 1).view(np.uint32), h1.view(np.uint32))
              and np.array_equal(h_run(m, x, 64, 2).view(np.uint32), h2.view(np.uint32)) for m in ONE + VER),
          f'the output H64 (once, twice) is bitwise the host transform on every Q8 object; twice is the input within '
          f'{rt:.1e} x max |x| (host fp32, the bound below)')
    check(rt <= 1e-6, f'the H64 round trip in fp32: max |H(H(x)) - x| {rt:.1e} x max |x| <= 1e-6')
    kx = rng.normal(0.0, 0.5, (1, NKV, D)).astype(np.float32); kx[0, 2, 37] = np.nan
    try: append_run(ONE[0], 1, 0, kx, np.zeros_like(kx)); poisoned = False
    except k3.InvalidKVEncoding: poisoned = True
    check(poisoned, f'{ONE[0]}: a NaN in K sets the sticky Q8 error word and the launch raises InvalidKVEncoding')
    z1 = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt, device='cuda')
    f16c = z1(64 * 2 * NKV * D, torch.float16); pz = z1(1, torch.int32)
    for m in ONE + VER:
        E = k3.Engine(module=m, grid=k3.resident_grid(m), T=1, limit_us=500_000)
        check(refuses(lambda: E.attn_dec(z1(H * D), f16c, pz, z1(H * D), H, NKV, D, SCALE, q8=False, max_cells=64), 'Q8 KV cache only'),
              f'{m} refuses an f16 cache')
    E = k3.Engine(module=VER[0], grid=k3.resident_grid(VER[0]), T=4, limit_us=500_000)
    sel = z1(4 * k3.QSA_SELW, torch.int32)
    check(refuses(lambda: E._attn_split(z1(4 * H * D), cache, pz, sel, z1(4 * H * D), H, NKV, D, SCALE, True, -1, min(64, ROWS), None),
                  'identity list only'), f'{VER[0]} refuses a QSA selection')
    check(refuses(lambda: k3.Engine(module=VER[0], grid=k3.resident_grid(VER[0]), T=5, limit_us=500_000), 'verify T = 1 .. 4'),
          f'{VER[0]} refuses T 5')
    print('RESULT', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)

bit, rows_ok, err = True, True, 0.0
for n in LENS:
    pos = n - 1
    q1 = torch.randn(H * D, generator=g, device='cuda') * 0.1
    a = run('k3_q35_swift', 1, pos, q1); b = run('k3_gguf_q35d', 1, pos, q1); c = run('k3_q35_tcap', 1, pos, q1)
    bit &= torch.equal(a, b) and torch.equal(a, c)
    err = max(err, float(np.abs(a[0].double().cpu().numpy() - ref(q1, pos)).max()))
    for T in (2, 3, 4):
        if n < T: continue
        qT = torch.randn(T * H * D, generator=g, device='cuda') * 0.1
        p0 = n - T
        aT = run('k3_q35_swift_ver', T, p0, qT); bT = run('k3_gguf_q35d_mt', T, p0, qT)
        bit &= torch.equal(aT, bT)
        for r in range(T):
            rows_ok &= torch.equal(aT[r], run('k3_q35_swift', 1, p0 + r, qT[r * H * D:(r + 1) * H * D].contiguous())[0])
    print(f'  list length {n}: max |own - float64| so far {err:.2e}', flush=True)
check(bit, f'the own modules\' pair (packed PART at T 2 / 3 / 4) is bitwise k3_gguf_q35d\'s (T 1) and k3_gguf_q35d_mt\'s (unpacked) at list lengths {LENS}')
check(rows_ok, 'a T = 2 / 3 / 4 pass\'s rows are bitwise the one-row passes at their positions')
check(err <= 1e-5 * vmax, f'against float64 attention: max |difference| {err:.2e} <= 1e-5 x max |V| ({vmax:.2f})')
E = k3.Engine(module='k3_q35_swift', grid=k3.resident_grid('k3_q35_swift'), T=1, limit_us=500_000)
try:
    q8c = torch.zeros(64 * k3.kv_pos_bytes(NKV * D, True), dtype=torch.uint8, device='cuda')
    E.attn_dec(torch.zeros(H * D, device='cuda'), q8c, torch.zeros(1, dtype=torch.int32, device='cuda'), torch.zeros(H * D, device='cuda'),
               H, NKV, D, SCALE, q8=True, max_cells=64)
    E.build(); refused = False
except ValueError:
    refused = True
check(refused, 'the q8_0 cache is refused on the own modules (f16 only)')
for what, kw in (('T 5', dict(T=5)), ('a QSA selection', dict(T=4, sel=True)), ('a q8_0 cache', dict(T=4, q8=True))):
    # (a q8_0 cache: the F16 object's own format check refuses it before the packed PART's check is reached)
    want = 'F16 KV cache only' if kw.get('q8') else 'the packed PART'
    try:
        Ex = k3.Engine(module='k3_q35_swift_ver', grid=k3.resident_grid('k3_q35_swift_ver'), T=kw['T'], limit_us=500_000)
        T_ = kw['T']; sel = torch.zeros(T_ * k3.QSA_SELW, dtype=torch.int32, device='cuda') if kw.get('sel') else None
        q8 = bool(kw.get('q8'))
        Ex._attn_split(torch.zeros(T_ * H * D, device='cuda'), q8c if q8 else cache, torch.zeros(1, dtype=torch.int32, device='cuda'), sel,
                       torch.zeros(T_ * H * D, device='cuda'), H, NKV, D, SCALE, q8, -1, 64, None)
        refused = False
    except ValueError as ex:
        refused = want in str(ex)   # (the expected refusal, not another ValueError on the way)
        if not refused: print(f'  ({what}: a different ValueError: {ex})', flush=True)
    check(refused, f'k3_q35_swift_ver refuses {what} ({want!r})')
print('RESULT', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
