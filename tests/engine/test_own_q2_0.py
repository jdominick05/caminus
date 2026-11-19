"""The own Q2_0 tile (kernels/own_quant_q2_0.h through kernels/own_mv_q2_0.hip) on real Q2_0 expert tensors, one card:
  exact    every row of each case's matrix through own_mv_q2_0_k3 on two fp16 activations (heavy-tailed, and smooth):
           rel = max |out - ref| / sum |w| |x| <= 1e-6 per case, ref the float64 dot of caminus.own_dequant's rows (bitwise
           gguf-py's) with the fp16 values the tile reads: the strict exact-dot check (tests/engine/test_k3_gguf_gsq.py
           G1(c) as written); non-finite outputs fail it. The GPU run checks SPOT_ROWS rows of each case (the first, the
           last and a seeded sample) itself, so its RESULT covers the decode; it saves every output (--save FILE, only when
           its own checks pass) and --check FILE does every row on the CPU, after the GPU window
  grouped  own_moe_gu_q2_0 / own_moe_down_q2_0 on 8 slots (one expert twice, ids -1 and E + 3 clamped to 0 and E - 1)
           against own_mv_q2_0_k3 on each slot's expert rows: the gate / up / down sums bit for bit, every row; the fused
           gate | up output = fp16(silu(gate) * up) of those sums to fp16 rounding
  tails    a launch whose rows end inside a workgroup (16 rows): the plain rows on a 13-row slice, the grouped kernels on
           one slot of a 14-row matrix view (their rows go in pairs: R even), into NaN-filled 16-row outputs: the rows
           bit for bit the full launch's, the rows past them still NaN
Cases: per Q2_0 expert role of the file its first and last layer at full K (Qwen3.6: gate / up K 2048, down K 512) and
the down's block-column cuts (K 512: 256 | 256, 384 | 128, the two-card forms); with --coder GGUF also the Coder's Q2_0
down (K 640) and its two-card cut 384 | 256. Every K takes the tile's lane form for it (own_quant_q2_0.h: half-wave rows
at K <= 512).
  HIP_VISIBLE_DEVICES=<one card> tools/guard.sh LOG 600 python -I tests/engine/test_own_q2_0.py GGUF [--coder CODER] --save OUT.pt
  python -I tests/engine/test_own_q2_0.py --check OUT.pt        (CPU: the exact check on the saved outputs; reads the GGUFs)
New kernel code: run it on a non-display card alone first. Prints PASS / FAIL lines and RESULT PASS | FAIL."""
import argparse, ctypes, os, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'tools', 'gates')); sys.path.insert(0, os.path.join(ROOT, 'python'))
import numpy as np
from caminus import ggufio, own_dequant
from gacc import block_cols   # noqa: E402  (tools/gates/gacc.py)

ap = argparse.ArgumentParser()
ap.add_argument('gguf', nargs='?')
ap.add_argument('--coder', default=None, help="the GSQ-RCO Coder's shard holding Q2_0 experts (its down, K 640)")
ap.add_argument('--save', default=None, help='GPU run: write the exact check\'s outputs here (refuses an existing file)')
ap.add_argument('--check', default=None, help='CPU: the float64 exact check on a --save file')
A = ap.parse_args()
ok_all = True
QB = 18          # bytes per 64-value block
SPOT_ROWS = 256  # rows per case the GPU run checks against float64 itself


def check(c, what):
    global ok_all
    print(('PASS ' if c else 'FAIL ') + what, flush=True); ok_all &= bool(c)


def cases(path, coder):
    """(the file, [(tensor, (b0, b1) block columns or None)]): per Q2_0 expert role its first and last layer whole, and
    the first layer's down at the two-card cuts"""
    F = ggufio.GGUFFile(path); out = []
    by = {}
    for n, t in F.T.items():
        if t.type == 'Q2_0' and n.endswith('_exps.weight'): by.setdefault(n.split('.')[2], []).append(int(n.split('.')[1]))
    for role, ls in sorted(by.items()):
        for il in sorted({min(ls), max(ls)}):
            n = f'blk.{il}.{role}.weight'; nb = F.T[n].shape[0] // 64
            out.append((n, None))
            if role == 'ffn_down_exps' and il == min(ls):   # (the Coder: 384 | 256 of 640; Qwen3.6: 256 | 256, 384 | 128 of 512)
                out += [(n, c) for c in ([(0, 6), (6, nb)] if coder else [(0, nb // 2), (nb // 2, nb), (0, 6), (6, nb)])]
    return F, out


def matrix(F, name, cut):
    """(raw rows [E * R, rb] uint8, E, R, K) of a case"""
    t = F.T[name]; K, R, E = t.shape; raw = np.array(t.data()).reshape(E * R, -1)   # (a copy: writable, off the mapping)
    if cut is None: return raw, E, R, K
    return block_cols(raw, QB, *cut), E, R, (cut[1] - cut[0]) * 64


def xs(K, seed):
    """the two activations of a case (fp16): heavy-tailed (Student t3, 8 spikes x 20) and smooth (a sine plus noise)"""
    rng = np.random.default_rng(seed)
    v = rng.standard_t(3, K); v[rng.choice(K, 8, replace=False)] *= 20; v /= np.sqrt((v ** 2).mean())
    s = np.sin(np.arange(K) * 0.37) + 0.1 * rng.standard_normal(K)
    return np.stack([v, s]).astype(np.float16)


def label(name, cut, K):
    return f'{name} K {K}' + (f' (block columns {cut[0]}:{cut[1]})' if cut else '')


def strict_rel(raw, K, x, y, rows=None):
    """the exact-dot check's max |y - ref| / sum |w| |x| over rows (default every row) and both vectors: raw [n, rb] the
    case's rows, x [2, K] the fp16 activations, y [2, n] the outputs; inf when an output checked is not finite"""
    rows = np.arange(raw.shape[0]) if rows is None else np.asarray(rows)
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)[:, rows]
    if not np.isfinite(y).all(): return float('inf')
    rel = 0.0
    for a in range(0, len(rows), 8192):
        rr = rows[a:a + 8192]
        W = own_dequant.dequant(np.ascontiguousarray(raw[rr]).reshape(-1), 'Q2_0', len(rr) * K).astype(np.float64).reshape(len(rr), K)
        S_ = np.maximum(np.abs(W) @ np.abs(x).T, 1e-30)                         # [rows, 2]
        rel = max(rel, float((np.abs(y[:, a:a + 8192].T - W @ x.T) / S_).max()))
    return rel


# ---- CPU: the float64 side of the exact check ---------------------------------------------------------------------------
if A.check:
    import torch
    S = torch.load(A.check, weights_only=False)
    Fs = {}
    worst = 0.0
    check(len(S['cases']) > 0, f"{A.check}: {len(S['cases'])} cases saved (a save with none checks nothing)")
    for c in S['cases']:
        F = Fs.setdefault(c['path'], ggufio.GGUFFile(c['path']))
        raw, E, R, K = matrix(F, c['name'], c['cut'])
        t0 = time.time()
        ok_shape = c['y'].shape == (2, raw.shape[0]) and c['x'].shape == (2, K)
        rel = strict_rel(raw, K, c['x'], c['y']) if ok_shape else float('inf')   # (inf: an output not finite, or a wrong shape)
        worst = max(worst, rel)
        check(rel <= 1e-6, f"exact {label(c['name'], c['cut'], K)}: {raw.shape[0]} rows x 2 vectors, max |out - ref| / sum |w||x| "
                           f"{rel:.2e} (<= 1e-6; {time.time() - t0:.0f} s)")
    print(f"worst {worst:.2e} over {len(S['cases'])} cases (saved by the GPU run on {S['device']})")
    print('RESULT', 'PASS' if ok_all else 'FAIL'); sys.exit(0 if ok_all else 1)

# ---- GPU ------------------------------------------------------------------------------------------------------------------
if not A.gguf: sys.exit('test_own_q2_0: name a GGUF with Q2_0 experts (or --check FILE)')
if A.save and os.path.exists(A.save): sys.exit(f'--save: {A.save} exists: refusing to overwrite')
import torch
from caminus.device import capacity
from caminus.device.hip import Module
dev = torch.device('cuda:0')
OWN = Module('own_mv_q2_0')
C = []
for path in [A.gguf] + ([A.coder] if A.coder else []):
    F, cs = cases(path, path == A.coder)
    if not cs: check(False, f'{os.path.basename(path)}: no Q2_0 expert tensors'); continue
    C += [(path, F, n, cut) for n, cut in cs]
saved = dict(device=torch.cuda.get_device_name(0), cases=[])


def plain(w, rows, K, x16, out):
    OWN.own_mv_q2_0_k3(((rows + 15) // 16,), (256,), w, ctypes.c_int(rows), ctypes.c_int(K), x16, out)


def grouped_args(ids, ns, R, E, K):
    """the grouped kernels' grid and scalars; refuses what their row pairs and clamp assume (R even, E > 0)"""
    assert R % 2 == 0 and R > 0 and E > 0 and ns > 0 and K % 64 == 0 and ids.numel() == ns, (R, E, ns, K)
    return ((ns * R + 15) // 16,), (256,), ctypes.c_int(ns), ctypes.c_int(R), ctypes.c_int(E), ctypes.c_int(K)


def moe_down(W, ids, ns, R, E, K, x16, y):
    g, b, *sc = grouped_args(ids, ns, R, E, K); OWN.own_moe_down_q2_0(g, b, W, ids, sc[0], sc[1], sc[2], sc[3], x16, y)


def moe_gu(Wg, Wu, ids, ns, R, E, K, x16, y16, gs, us):
    assert (gs is None) == (us is None), 'gs / us: both or neither'
    g, b, *sc = grouped_args(ids, ns, R, E, K)
    nul = ctypes.c_void_p(0)
    OWN.own_moe_gu_q2_0(g, b, Wg, Wu, ids, sc[0], sc[1], sc[2], sc[3], x16, y16, nul if gs is None else gs, nul if us is None else us)


def nan16(dt=torch.float32):
    return torch.full((16,), float('nan'), dtype=dt, device=dev)


def tail_ok(out, want, n):
    """rows [0, n) bit for bit want, rows [n, 16) still NaN"""
    return torch.equal(out[:n], want) and bool(torch.isnan(out[n:]).all())


for i, (path, F, name, cut) in enumerate(C):
    raw, E, R, K = matrix(F, name, cut)
    capacity.check([0], [3 * raw.nbytes / 2 ** 20 + 64], f'test_own_q2_0 {name}')
    W = torch.from_numpy(np.ascontiguousarray(raw)).to(dev).reshape(-1)
    X = xs(K, i); X0 = torch.from_numpy(X[0]).to(dev)
    y = torch.empty(2, E * R, device=dev)
    for v in range(2): plain(W, E * R, K, torch.from_numpy(X[v]).to(dev), y[v])
    torch.cuda.synchronize()
    yc = y.cpu().numpy()
    fin = bool(np.isfinite(yc).all()) and float(np.abs(yc).max()) > 0
    srows = np.unique(np.concatenate([[0, E * R - 1], np.random.default_rng(7 + i).choice(E * R, SPOT_ROWS - 2, replace=False)]))
    rel = strict_rel(raw, K, X, yc, srows)
    check(fin and rel <= 1e-6, f'{label(name, cut, K)} [{E} x {R}]: every output finite; spot exact check on {len(srows)} rows x 2 vectors '
          f'(first, last, seeded): max |out - ref| / sum |w||x| {rel:.2e} (<= 1e-6; every row: --check)')
    saved['cases'].append(dict(path=path, name=name, cut=cut, x=X, y=yc))
    rb = raw.shape[1]
    # tails: 13 rows of the plain kernel (rows R + 3 ..), and one slot of the [E * R // 14, 14] view for the grouped ones
    o = nan16(); plain(W[(R + 3) * rb:(R + 16) * rb], 13, K, X0, o); torch.cuda.synchronize()
    tails = [('plain 13 rows', tail_ok(o, y[0, R + 3:R + 16], 13))]
    E14, ids14 = E * R // 14, torch.tensor([1], dtype=torch.int32, device=dev)
    # grouped: 8 slots, one expert twice, ids -1 and E + 3 (clamped to 0 and E - 1)
    rng = np.random.default_rng(100 + i)
    ids_np = np.concatenate([rng.choice(E, 5, replace=False), [-1, E + 3]]).astype(np.int32)
    ids_np = np.insert(ids_np, 3, ids_np[1]).astype(np.int32); eff = np.clip(ids_np, 0, E - 1)
    ids = torch.from_numpy(ids_np).to(dev); ns = len(ids_np)
    xsl = torch.from_numpy(np.stack([xs(K, 1000 * i + s)[0] for s in range(ns)])).to(dev)   # [ns, K] per-slot inputs
    role = name.split('.')[2]
    un = name.replace('ffn_gate_exps', 'ffn_up_exps')
    if role == 'ffn_down_exps':
        want = torch.empty(ns, R, device=dev)
        for s in range(ns): plain(W[int(eff[s]) * R * rb:(int(eff[s]) + 1) * R * rb], R, K, xsl[s], want[s])
        got = torch.full((ns, R), float('nan'), device=dev)
        moe_down(W, ids, ns, R, E, K, xsl, got)
        o = nan16(); moe_down(W, ids14, 1, 14, E14, K, X0, o); torch.cuda.synchronize()
        tails.append(('grouped down, one slot of 14 rows', tail_ok(o, y[0, 14:28], 14)))
        check(torch.equal(got, want), f'{label(name, cut, K)}: grouped down ({ns} slots, ids {ids_np.tolist()}) '
              'bit for bit the plain rows of each slot\'s expert on its own input')
    elif role == 'ffn_gate_exps' and (un not in F.T or F.T[un].type != 'Q2_0'):
        check(False, f'{name}: no Q2_0 {un} beside it')
    elif role == 'ffn_gate_exps':
        rawu = np.array(F.T[un].data()).reshape(E * R, -1)
        Wu = torch.from_numpy(np.ascontiguousarray(rawu)).to(dev).reshape(-1)
        wu = torch.empty(ns, R, device=dev); wg = torch.empty(ns, R, device=dev)
        for s in range(ns):
            e0 = int(eff[s])
            plain(W[e0 * R * rb:(e0 + 1) * R * rb], R, K, xsl[0], wg[s]); plain(Wu[e0 * R * rb:(e0 + 1) * R * rb], R, K, xsl[0], wu[s])
        y16 = torch.full((ns, R), float('nan'), dtype=torch.float16, device=dev)
        gs = torch.full((ns, R), float('nan'), device=dev); us = torch.full((ns, R), float('nan'), device=dev)
        moe_gu(W, Wu, ids, ns, R, E, K, xsl[0], y16, gs, us)
        u14 = torch.empty(28, device=dev); plain(Wu, 28, K, X0, u14)
        o16, og, ou = nan16(torch.float16), nan16(), nan16(); moe_gu(W, Wu, ids14, 1, 14, E14, K, X0, o16, og, ou)
        n16 = nan16(torch.float16); moe_gu(W, Wu, ids14, 1, 14, E14, K, X0, n16, None, None)   # (no fp32 sums: both pointers null)
        torch.cuda.synchronize()
        tails.append(('grouped gate | up, one slot of 14 rows (y16, gate and up sums; y16 again without the sums)',
                      tail_ok(og, y[0, 14:28], 14) and tail_ok(ou, u14[14:28], 14) and bool(torch.isnan(o16[14:]).all())
                      and torch.equal(o16[:14], n16[:14]) and bool(torch.isnan(n16[14:]).all())))
        g, u = wg.double().cpu().numpy(), wu.double().cpu().numpy()
        expr = g / (1.0 + np.exp(-g)) * u; got16 = y16.double().cpu().numpy()
        tol = 2e-3 * np.abs(expr) + 1e-3 * np.sqrt((expr ** 2).mean())
        check(torch.equal(gs, wg) and torch.equal(us, wu) and bool((np.abs(got16 - expr) <= tol).all()),
              f'{label(name, cut, K)}: grouped gate | up ({ns} slots, ids {ids_np.tolist()}): the gate and up sums '
              f'bit for bit the plain rows, y16 = fp16(silu(g) * u) to rounding (max |diff| {np.abs(got16 - expr).max():.2e})')
        del Wu
    check(all(ok for _, ok in tails), f'{label(name, cut, K)}: tails ' + '; '.join(f'{n} {"ok" if ok else "WRONG"}' for n, ok in tails) +
          ' (rows bit for bit the full launch\'s, the rest of the 16 still NaN)')
    del W; torch.cuda.empty_cache()
if A.save and not ok_all:
    print(f'--save: a check failed: {A.save} not written')
elif A.save:
    if os.path.exists(A.save): sys.exit(f'--save: {A.save} appeared during the run: refusing to overwrite')
    torch.save(saved, A.save); print(f'saved {A.save}: {len(saved["cases"])} cases')
print(f'PENDING the exact check on every row (this RESULT covers {SPOT_ROWS} rows per case): ' +
      (f'run --check {A.save} on the CPU' if A.save else 'rerun with --save FILE, then --check FILE on the CPU'))
print('RESULT', 'PASS' if ok_all else 'FAIL'); sys.exit(0 if ok_all else 1)
