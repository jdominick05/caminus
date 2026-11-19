"""The own Q2_0 tile (kernels/own_quant_q2_0.h through kernels/own_mv_q2_0.hip) on real Q2_0 expert tensors, one card:
  closure  (CPU, always) own_mv_q2_0.hip includes no ggml / llama.cpp header (tools/own_closure.audit)
  exact    every row of each case's matrix through own_mv_q2_0_k3 on two fp16 activations (heavy-tailed, and smooth):
           rel = max |out - ref| / sum |w| |x| <= 1e-6 per case, ref the float64 dot of caminus.own_dequant's rows (bitwise
           gguf-py's) with the fp16 values the tile reads: the strict exact-dot check (tests/engine/test_k3_gguf_gsq.py
           G1(c) as written). The GPU run saves the outputs (--save FILE); --check FILE does the float64 side on the CPU,
           after the GPU window
  grouped  own_moe_gu_q2_0 / own_moe_down_q2_0 on 8 slots (one expert twice, ids -1 and E + 3 clamped to 0 and E - 1)
           against own_mv_q2_0_k3 on each slot's expert rows: the gate / up / down sums bit for bit, every row; the fused
           gate | up output = fp16(silu(gate) * up) of those sums to fp16 rounding
Cases: per Q2_0 expert role of the file its first and last layer at full K (Qwen3.6: gate / up K 2048, down K 512) and
the down's block-column cuts (K 512: 256 | 256, 384 | 128, the two-card forms); with --coder GGUF also the Coder's Q2_0
down (K 640) and its two-card cut 384 | 256. Every K takes the tile's lane form for it (own_quant_q2_0.h: half-wave rows
at K <= 512).
  HIP_VISIBLE_DEVICES=<one card> tools/guard.sh LOG 600 python -I tests/engine/test_own_q2_0.py GGUF [--coder CODER] --save OUT.pt
  python -I tests/engine/test_own_q2_0.py --check OUT.pt        (CPU: the exact check on the saved outputs; reads the GGUFs)
New kernel code: run it on a non-display card alone first. Prints PASS / FAIL lines and RESULT PASS | FAIL."""
import argparse, ctypes, os, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'tools')); sys.path.insert(0, os.path.join(ROOT, 'python'))
import numpy as np
from caminus import ggufio, own_dequant
from own_closure import audit   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('gguf', nargs='?')
ap.add_argument('--coder', default=None, help="the GSQ-RCO Coder's shard holding Q2_0 experts (its down, K 640)")
ap.add_argument('--save', default=None, help='GPU run: write the exact check\'s outputs here (refuses an existing file)')
ap.add_argument('--check', default=None, help='CPU: the float64 exact check on a --save file')
A = ap.parse_args()
ok_all = True
QB = 18   # bytes per 64-value block


def check(c, what):
    global ok_all
    print(('PASS ' if c else 'FAIL ') + what, flush=True); ok_all &= bool(c)


cl, missing, bad = audit('own_mv_q2_0.hip')
check(not bad and not missing, f'own_mv_q2_0.hip: includes {sorted(cl - {"own_mv_q2_0.hip"})} (no ggml header; every one in kernels/)')


def cuts(raw, nb, b0, b1):
    """block columns [b0, b1) of every row of raw [rows, nb * 18] (a card's K cut)"""
    return np.ascontiguousarray(raw.reshape(raw.shape[0], nb, QB)[:, b0:b1]).reshape(raw.shape[0], -1)


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
    return cuts(raw, K // 64, *cut), E, R, (cut[1] - cut[0]) * 64


def xs(K, seed):
    """the two activations of a case (fp16): heavy-tailed (Student t3, 8 spikes x 20) and smooth (a sine plus noise)"""
    rng = np.random.default_rng(seed)
    v = rng.standard_t(3, K); v[rng.choice(K, 8, replace=False)] *= 20; v /= np.sqrt((v ** 2).mean())
    s = np.sin(np.arange(K) * 0.37) + 0.1 * rng.standard_normal(K)
    return np.stack([v, s]).astype(np.float16)


def label(name, cut, K):
    return f'{name} K {K}' + (f' (block columns {cut[0]}:{cut[1]})' if cut else '')


# ---- CPU: the float64 side of the exact check ---------------------------------------------------------------------------
if A.check:
    import torch
    S = torch.load(A.check, weights_only=False)
    Fs = {}
    worst = 0.0
    for c in S['cases']:
        F = Fs.setdefault(c['path'], ggufio.GGUFFile(c['path']))
        raw, E, R, K = matrix(F, c['name'], c['cut'])
        x = c['x'].astype(np.float64); y = c['y'].astype(np.float64)          # [2, K], [2, E * R]
        rel = 0.0; t0 = time.time()
        for r0 in range(0, raw.shape[0], 8192):
            r1 = min(raw.shape[0], r0 + 8192)
            W = own_dequant.dequant(raw[r0:r1].reshape(-1), 'Q2_0', (r1 - r0) * K).astype(np.float64).reshape(r1 - r0, K)
            S_ = np.maximum(np.abs(W) @ np.abs(x).T, 1e-30)                      # [rows, 2]
            rel = max(rel, float((np.abs(y[:, r0:r1].T - W @ x.T) / S_).max()))
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


for i, (path, F, name, cut) in enumerate(C):
    raw, E, R, K = matrix(F, name, cut)
    capacity.check([0], [3 * raw.nbytes / 2 ** 20 + 64], f'test_own_q2_0 {name}')
    W = torch.from_numpy(np.ascontiguousarray(raw)).to(dev).reshape(-1)
    X = xs(K, i)
    y = torch.empty(2, E * R, device=dev)
    for v in range(2): plain(W, E * R, K, torch.from_numpy(X[v]).to(dev), y[v])
    torch.cuda.synchronize()
    fin = bool(torch.isfinite(y).all()) and float(y.abs().max()) > 0
    saved['cases'].append(dict(path=path, name=name, cut=cut, x=X, y=y.cpu().numpy()))
    # grouped: 8 slots, one expert twice, ids -1 and E + 3 (clamped to 0 and E - 1)
    rng = np.random.default_rng(100 + i)
    ids_np = np.concatenate([rng.choice(E, 5, replace=False), [-1, E + 3]]).astype(np.int32)
    ids_np = np.insert(ids_np, 3, ids_np[1]).astype(np.int32); eff = np.clip(ids_np, 0, E - 1)
    ids = torch.from_numpy(ids_np).to(dev); ns = len(ids_np)
    rb = raw.shape[1]
    xsl = torch.from_numpy(np.stack([xs(K, 1000 * i + s)[0] for s in range(ns)])).to(dev)   # [ns, K] per-slot inputs
    role = name.split('.')[2]
    if role == 'ffn_down_exps':
        want = torch.empty(ns, R, device=dev)
        for s in range(ns): plain(W[int(eff[s]) * R * rb:(int(eff[s]) + 1) * R * rb], R, K, xsl[s], want[s])
        got = torch.full((ns, R), float('nan'), device=dev)
        OWN.own_moe_down_q2_0(((ns * R + 15) // 16,), (256,), W, ids, ctypes.c_int(ns), ctypes.c_int(R), ctypes.c_int(E),
                              ctypes.c_int(K), xsl, got)
        torch.cuda.synchronize()
        check(fin and torch.equal(got, want), f'{label(name, cut, K)} [{E} x {R}]: finite; grouped down ({ns} slots, ids {ids_np.tolist()}) '
              'bit for bit the plain rows of each slot\'s expert on its own input')
    elif role == 'ffn_gate_exps':
        un = name.replace('ffn_gate_exps', 'ffn_up_exps')
        if un not in F.T or F.T[un].type != 'Q2_0': check(False, f'{name}: no Q2_0 {un} beside it'); continue
        rawu = np.array(F.T[un].data()).reshape(E * R, -1)
        Wu = torch.from_numpy(np.ascontiguousarray(rawu)).to(dev).reshape(-1)
        wu = torch.empty(ns, R, device=dev); wg = torch.empty(ns, R, device=dev)
        for s in range(ns):
            e0 = int(eff[s])
            plain(W[e0 * R * rb:(e0 + 1) * R * rb], R, K, xsl[0], wg[s]); plain(Wu[e0 * R * rb:(e0 + 1) * R * rb], R, K, xsl[0], wu[s])
        y16 = torch.full((ns, R), float('nan'), dtype=torch.float16, device=dev)
        gs = torch.full((ns, R), float('nan'), device=dev); us = torch.full((ns, R), float('nan'), device=dev)
        OWN.own_moe_gu_q2_0(((ns * R + 15) // 16,), (256,), W, Wu, ids, ctypes.c_int(ns), ctypes.c_int(R), ctypes.c_int(E),
                            ctypes.c_int(K), xsl[0], y16, gs, us)
        torch.cuda.synchronize()
        g, u = wg.double().cpu().numpy(), wu.double().cpu().numpy()
        expr = g / (1.0 + np.exp(-g)) * u; got16 = y16.double().cpu().numpy()
        tol = 2e-3 * np.abs(expr) + 1e-3 * np.sqrt((expr ** 2).mean())
        check(fin and torch.equal(gs, wg) and torch.equal(us, wu) and bool((np.abs(got16 - expr) <= tol).all()),
              f'{label(name, cut, K)} [{E} x {R}]: finite; grouped gate | up ({ns} slots, ids {ids_np.tolist()}): the gate and up sums '
              f'bit for bit the plain rows, y16 = fp16(silu(g) * u) to rounding (max |diff| {np.abs(got16 - expr).max():.2e})')
        del Wu
    else:
        check(fin, f'{label(name, cut, K)}: finite (the up is checked with its gate)')
    del W; torch.cuda.empty_cache()
if A.save:
    if os.path.exists(A.save): sys.exit(f'--save: {A.save} appeared during the run: refusing to overwrite')
    torch.save(saved, A.save); print(f'saved {A.save}: {len(saved["cases"])} cases (run --check on it for the exact check)')
print('RESULT', 'PASS' if ok_all else 'FAIL'); sys.exit(0 if ok_all else 1)
