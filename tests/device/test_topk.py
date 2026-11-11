"""caminus.device.topk on a GPU: the device sampler's candidates on Caminus's own kernel (kernels/own_topk.hip), against an
exact host oracle and against torch.topk / index_select, then its time beside torch's.

  tools/guard.sh LOG 600 env HIP_VISIBLE_DEVICES=<a non-display card> python -I tests/device/test_topk.py [--no-timing]

The oracle is the contract itself, in numpy: per row the k largest values by the kernel's key (NaN first, -0.0 just below
+0.0), value descending and id ascending, ties at the cut keeping the lowest ids; values compared as bits. Rows: random,
heavy ties, ties exactly at the cut, -inf rows with a few finite values, NaNs, signed zeros, a row narrower than k; logits
shards as column slices of a wider tensor (the engines' layout); 1 to 9 rows. Canaries around every output must hold,
reruns must be bitwise equal, and inputs the kernel cannot take must be refused before a launch.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import torch  # noqa: E402

from caminus.device import capacity, topk  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--no-timing', action='store_true')
A = ap.parse_args()
FAILS = []


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok:
        FAILS.append(what)


def keys(x):
    """the kernel's order key of float32 values (numpy uint32)"""
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    k = np.where(u & 0x80000000, (~u) & 0xffffffff, u | 0x80000000)
    nan = ((u & 0x7f800000) == 0x7f800000) & ((u & 0x007fffff) != 0)
    return np.where(nan, 0xffffffff, k).astype(np.uint64)


def oracle(row, k):
    """(values float32, ids) of the k largest by the contract"""
    kk = min(k, len(row))
    o = np.lexsort((np.arange(len(row)), -keys(row).astype(np.float64)))[:kk]
    return row[o], o


dev = torch.device('cuda', 0)
name = torch.cuda.get_device_name(dev)
print(f'NOTE device: {name} (HIP_VISIBLE_DEVICES={os.environ.get("HIP_VISIBLE_DEVICES")})', flush=True)
VH, ROWS, WIDTH = 75968, 9, 176                      # (a Qwen vocabulary's half; the session's rows and kmax at T = 9)
capacity.check([dev.index], [(ROWS * (2 * VH + 3) * 4 * 3 + ROWS * WIDTH * 16 * 3) / capacity.MIB], 'test_topk inputs and buffers')
rng = np.random.default_rng(7)


def make(pattern, R, n):
    x = rng.standard_normal((R, n)).astype(np.float32) * 4
    if pattern == 'ties':
        x = np.round(x, 1)                           # (a few hundred distinct values per row: ties everywhere)
    elif pattern == 'cut':
        x = rng.standard_normal((R, n)).astype(np.float32) - 20
        for r in range(R):                           # (k - 3 large distinct values, then 12 equal ones straddling the cut)
            ids = rng.choice(n, 150, replace=False)
            x[r, ids[:133]] = 10 + np.arange(133, dtype=np.float32) * 0.01
            x[r, ids[133:145]] = 5.0
    elif pattern == 'neginf':
        x[:] = -np.inf
        for r in range(R):
            x[r, rng.choice(n, 5, replace=False)] = rng.standard_normal(5)
    elif pattern == 'nan':
        for r in range(R):
            x[r, rng.choice(n, 3, replace=False)] = np.nan
    elif pattern == 'zeros':
        x[:] = 0.0
        x[:, ::3] = -0.0
        x[:, rng.choice(n, 20, replace=False)] = 1.0
    return x


def put(x, wide):
    """x as a column slice of a wider float32 tensor (the engines' second shard: logits[:, vh:]); returns (base, view)"""
    R, n = x.shape
    if not wide:
        t = torch.from_numpy(x).to(dev)
        return t, t
    base = torch.full((R, 2 * n + 3), 123.0, device=dev)
    base[:, n + 3:] = torch.from_numpy(x).to(dev)
    return base, base[:, n + 3:]


cases = [('random', 1, VH, 136, True), ('random', 9, VH, 136, True), ('random', 5, VH, 1, False),
         ('random', 3, VH, 176, True), ('ties', 9, VH, 136, True), ('cut', 4, VH, 136, False),
         ('cut', 9, VH, 140, True), ('neginf', 3, VH, 136, False), ('nan', 4, VH, 136, True),
         ('zeros', 3, VH, 64, False), ('random', 2, 8, 136, False), ('random', 3, 1000, 176, True),
         ('random', 1, 75967, 100, True)]
for pattern, R, n, k, wide in cases:
    cand = topk.Candidates(dev, ROWS, WIDTH)
    for t, fill in ((cand.val, -7.0), (cand.idx, -7), (cand.pen, -7.0)):
        t.fill_(fill)
    x = make(pattern, R, n)
    base, L = put(x, wide)
    pid = np.unique(np.concatenate(([0, n - 1], rng.choice(n, min(40, n), replace=False))))
    kk = cand.run(L, k, pid)
    torch.cuda.synchronize()
    v, ix, pv = cand.val.cpu().numpy(), cand.idx.cpu().numpy(), cand.pen.cpu().numpy()
    exact = True
    for r in range(R):
        ov, oi = oracle(x[r], k)
        exact &= np.array_equal(ix[r, :kk], oi) and np.array_equal(v[r, :kk].view(np.uint32), ov.view(np.uint32))
    pen_ok = all(np.array_equal(pv[r, :len(pid)].view(np.uint32), x[r, pid].view(np.uint32)) for r in range(R))
    canary = (np.all(v[:R, kk:] == -7) and np.all(ix[:R, kk:] == -7) and np.all(pv[:R, len(pid):] == -7)
              and np.all(v[R:] == -7) and np.all(ix[R:] == -7) and np.all(pv[R:] == -7))
    side = (not wide) or bool(torch.all(base[:, :n + 3] == 123.0))
    tv, ti = torch.topk(L, kk, dim=1)
    tvn = tv.cpu().numpy()
    nan_rows = np.isnan(x[:R]).any(1)
    same_t = all(np.array_equal(np.sort(v[r, :kk]), np.sort(tvn[r])) for r in range(R) if not nan_rows[r]
                 ) if pattern != 'zeros' else True   # (torch's ±0 order differs; NaN never compares equal)
    again = cand.run(L, k, pid); torch.cuda.synchronize()
    rerun = again == kk and np.array_equal(cand.val.cpu().numpy(), v) and np.array_equal(cand.idx.cpu().numpy(), ix)
    check(exact and pen_ok and canary and side and same_t and rerun,
          f'{pattern:7s} R={R} n={n} k={k}{" (a column slice)" if wide else ""}: oracle {exact}, penalty values {pen_ok}, '
          f'canaries {canary and side}, torch.topk values {same_t}, rerun bitwise {rerun}')

# ---- refusals before any launch
cand = topk.Candidates(dev, ROWS, WIDTH)
x16 = torch.zeros(2, 100, device=dev, dtype=torch.float16)
xs = torch.zeros(2, 200, device=dev)[:, ::2]
xc = torch.zeros(2, 100)
for what, f in (('k 0', lambda: cand.run(torch.zeros(2, 100, device=dev), 0)),
                ('k past the buffers', lambda: cand.run(torch.zeros(2, 100, device=dev), WIDTH + 1)),
                ('rows past the buffers', lambda: cand.run(torch.zeros(ROWS + 1, 100, device=dev), 8)),
                ('float16 logits', lambda: cand.run(x16, 8)),
                ('strided columns', lambda: cand.run(xs, 8)),
                ('host logits', lambda: cand.run(xc, 8)),
                ('a penalty id outside the shard', lambda: cand.run(torch.zeros(2, 100, device=dev), 8, [5, 100])),
                ('a width past the kernel', lambda: topk.Candidates(dev, 2, topk.KCAP + 1))):
    try:
        f(); check(False, f'refused: {what}')
    except ValueError:
        check(True, f'refused: {what}')
check(cand.launches == 0, 'no launch for a refused input')

# ---- time beside torch.topk + index_select (interleaved, median of 5 rounds of 200)
if not A.no_timing:
    for R in (1, 9):
        x = torch.from_numpy(make('random', R, VH)).to(dev)
        pid = np.unique(rng.choice(VH, 40, replace=False)); pidt = torch.as_tensor(pid, device=dev)
        cand = topk.Candidates(dev, ROWS, WIDTH)

        def own():
            cand.run(x, 136, pid)

        def ref():
            torch.topk(x, 136, dim=1); x.index_select(1, pidt)
        med = {}
        for nm, f in (('own', own), ('torch', ref)):
            f(); torch.cuda.synchronize()
        rounds = {'own': [], 'torch': []}
        for _ in range(5):
            for nm, f in (('own', own), ('torch', ref)):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(200):
                    f()
                e1.record(); torch.cuda.synchronize()
                rounds[nm].append(e0.elapsed_time(e1) * 1e3 / 200)
        for nm in rounds:
            med[nm] = float(np.median(rounds[nm]))
        print(f'TIME R={R} n={VH} k=136 + 40 penalty ids: own {med["own"]:.1f} us, torch.topk + index_select '
              f'{med["torch"]:.1f} us per call (median of 5 x 200, GPU time between events; host launch cost included)', flush=True)
print('RESULT', 'FAIL' if FAILS else 'PASS')
sys.exit(1 if FAILS else 0)
