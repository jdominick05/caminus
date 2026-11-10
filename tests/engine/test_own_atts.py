"""OP_OWN_ATTS (the decode attention split over positions, kernels/k3_own_atts.h) against OP_OWN_ATTN (one task per query
head: the oracle) and a float64 host reference, on random inputs at chosen positions (cache rows below the position
pre-filled, the same bytes in every form's cache). Needs no model.

Checks, per position:
  - the cache row written at the position: bitwise OP_OWN_ATTN's (the same norm / RoPE / f16 statements);
  - the fp16 outputs against float64: mean and max |y - ref| / rms(ref) no worse than OP_OWN_ATTN's by more than a
    rounding margin (the split changes only the summation order);
  - determinism: a second run, and engines at other grids (K3_PER_WGP 3 / 7 workgroups per WGP), give the same bits.
Positions cover the split layout's edges (npos at multiples of 32 and of lmin, one split / several, the last split
holding a single position, the chunk edges 128 / 129) and depth (default up to 8,192; --deep adds 16K / 40K, which need
the cache to fit: f16 rows 2 KiB per position, three copies).
  tools/guard.sh LOG 600 python -I tests/engine/test_own_atts.py [--module k3_q3m_lean] [--deep]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np, torch
from caminus.engine import k3
ap = argparse.ArgumentParser()
ap.add_argument('--module', default='k3_q3m_lean'); ap.add_argument('--oracle-module', default='k3_gguf_q3m_own')
ap.add_argument('--splits', type=int, default=64); ap.add_argument('--lmin', type=int, default=128)
ap.add_argument('--deep', action='store_true'); ap.add_argument('--grids', default='3,7')
ap.add_argument('--every', type=int, default=0, help='also every position 0 .. N - 1')
ap.add_argument('--gh', default='8', help='heads per task: the first is the reference form, the others must give its bits (1 / 2 / 4: ATS_GH_RUNTIME builds only)')
A = ap.parse_args()
nh, nkv, hd = 32, 4, 128
M = nkv * hd
POS = [0, 1, 2, 30, 31, 32, 33, 63, 64, 127, 128, 129, 130, 255, 256, 257, 383, 384, 511, 512, 513, 1000, 1023, 1024, 1025,
       2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193, 8250]
if A.deep: POS += [16383, 16384, 16385, 24577, 32767, 40959]
if A.every: POS = sorted(set(POS) | set(range(A.every)))
CTX = max(POS) + 1
g = torch.Generator().manual_seed(11)
dev = 'cuda'
qn = (torch.rand(hd, generator=g) * 2 + 0.5); kn = (torch.rand(hd, generator=g) * 2 + 0.5)
ts = float(np.float32(1e6) ** np.float32(-2.0 / hd)); eps = 1e-6; scale = 1.0 / np.sqrt(hd)
par = torch.cat([qn, kn, torch.tensor([ts, eps], dtype=torch.float32)]).to(dev).contiguous()
# the cache below every position: random K / V rows (f16), the same bytes in each form's cache
base = (torch.randn((CTX + k3.Engine.ATS_PAD) * 2 * M, generator=g) * 1.5).to(torch.float16)   # (+ the split op's padding rows)
B = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt, device=dev)


def oracle():
    d = dict(q=B(nh * hd), kv=B(2 * M), out=B(nh * hd, torch.float16), cache=base.to(dev), pos=B(1, torch.int32))
    E = k3.Engine(module=A.oracle_module, grid=k3.resident_grid(A.oracle_module), T=1, limit_us=200_000)
    E.own_attn(d['q'], d['kv'], par, d['cache'], d['pos'], d['out'], nh, nkv, hd, scale); E.build()
    return E, d


def split(per_wgp=None, gh=None, S=None):
    gh = gh or GH[0]; S = S or A.splits
    d = dict(qkv=B(nh * hd + 2 * M), scr=B(k3.Engine.atts_scratch_floats(nkv, S, gh)), out=B(nh * hd, torch.float16),
             cache=base.to(dev), pos=B(1, torch.int32))
    E = k3.Engine(module=A.module, grid=k3.resident_grid(A.module, per_wgp), T=1, limit_us=200_000)
    E.own_atts(d['qkv'], d['scr'], par, d['cache'], d['pos'], d['out'], nh, nkv, hd, scale, splits=S, lmin=A.lmin, heads_per_task=gh)
    E.build()
    return E, d


def ref64(q, k, v, cache_np, p):
    """float64: q / k norm + NEOX RoPE (n_rot 128), the position's k / v rounded to f16 as the cache holds them, attention
    over 0 .. p with 8 query heads per kv head"""
    q = q.reshape(nh, hd).astype(np.float64); k = k.reshape(nkv, hd).astype(np.float64); v = v.reshape(nkv, hd)
    def nr(x, w):
        x = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w.astype(np.float64)
        j = np.arange(hd // 2); th = p * (np.float64(1e6) ** (-2.0 * j / hd))
        c, s = np.cos(th), np.sin(th); x0, x1 = x[..., :hd // 2], x[..., hd // 2:]
        return np.concatenate([x0 * c - x1 * s, x1 * c + x0 * s], -1)
    qr = nr(q, qn.numpy()); kr = nr(k, kn.numpy()).astype(np.float16).astype(np.float64)
    rows = cache_np[:p * 2 * M].reshape(p, 2, nkv, hd).astype(np.float64)
    K = np.concatenate([rows[:, 0], kr[None]], 0); V = np.concatenate([rows[:, 1], v.astype(np.float16).astype(np.float64)[None]], 0)
    out = np.zeros((nh, hd))
    for h in range(nh):
        gg = h // (nh // nkv); sc = K[:, gg] @ qr[h] * scale; e = np.exp(sc - sc.max())
        out[h] = (e[:, None] * V[:, gg]).sum(0) / e.sum()
    return out.reshape(-1)


u16 = lambda t: t.cpu().numpy().view(np.uint16)
GH = [int(x) for x in A.gh.split(',')]
EO, O = oracle(); ES, S = split()
# the forms that must give the reference form's bits: other grids, other head groups (every position), and fewer splits
# (16, 4: while npos <= S * lmin both layouts are L = lmin)
others = ([(f'grid {w}', 1 << 30) + split(int(w)) for w in A.grids.split(',') if w] +
          [(f'heads per task {gh}', 1 << 30) + split(gh=gh) for gh in GH[1:]] +
          [(f'splits {Sx}', Sx * A.lmin) + split(S=Sx) for Sx in (16, 4) if Sx < A.splits])
base_np = base.numpy().copy()
ok = True
def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)
worst = []
row_bad = det_bad = grid_bad = 0
for p in POS:
    q = torch.randn(nh * hd, generator=g) * 3; k = torch.randn(M, generator=g) * 3; v = torch.randn(M, generator=g)
    O['q'].copy_(q); O['kv'][:M].copy_(k); O['kv'][M:].copy_(v); O['pos'].fill_(p)
    S['qkv'][:nh * hd].copy_(q); S['qkv'][nh * hd:nh * hd + M].copy_(k); S['qkv'][nh * hd + M:].copy_(v); S['pos'].fill_(p)
    EO.run(); torch.cuda.synchronize(); ES.run(); torch.cuda.synchronize()
    ys = u16(S['out']).copy()
    rr = slice(p * 2 * M, (p + 1) * 2 * M)
    row_bad += int((u16(O['cache'][rr]) != u16(S['cache'][rr])).sum())
    ES.run(); torch.cuda.synchronize(); det_bad += int((u16(S['out']) != ys).sum())
    for name, lim, E2, S2 in others:
        S2['qkv'].copy_(S['qkv']); S2['pos'].fill_(p); E2.run(); torch.cuda.synchronize()
        if p + 1 <= lim:
            nb = int((u16(S2['out']) != ys).sum()); grid_bad += nb
            if nb: print(f'    {name}: {nb} values differ at position {p}', flush=True)
    ref = ref64(q.numpy(), k.numpy(), v.numpy(), base_np, p)
    base_np[rr] = S['cache'][rr].cpu().numpy()   # (later positions read the row this one wrote)
    rms = np.sqrt((ref ** 2).mean())
    eo = np.abs(O['out'].float().cpu().numpy() - ref) / rms; es = np.abs(S['out'].float().cpu().numpy() - ref) / rms
    worst.append((p, eo.mean(), es.mean(), eo.max(), es.max()))
    print(f'  position {p:6d}: oracle mean {eo.mean():.3e} max {eo.max():.3e} | split mean {es.mean():.3e} max {es.max():.3e}', flush=True)
check(row_bad == 0, f'cache rows bitwise OP_OWN_ATTN\'s ({row_bad} values differ)')
check(det_bad == 0, f'a second run gives the same bits ({det_bad} values differ)')
check(grid_bad == 0, f'grids {A.grids} per WGP, heads per task {GH[1:]} and fewer splits give the same bits ({grid_bad} values differ)')
w = np.array([x[1:] for x in worst])
check(np.all(w[:, 1] <= w[:, 0] * 1.10 + 2e-5), f'mean error vs float64 within 10 % of the oracle\'s at every position '
      f'(split {w[:, 1].mean():.3e} vs oracle {w[:, 0].mean():.3e} on average)')
check(np.all(w[:, 3] <= w[:, 2] * 1.5 + 1e-3), f'max error vs float64 within the oracle\'s rounding (split {w[:, 3].max():.3e}, oracle {w[:, 2].max():.3e})')
print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
