"""The own DFlash 2 drafter module (engine/q35_dflash.py on kernels/k3_q35_dflash.hip) against the float64 reference
(caminus.ref_dflash.RefDFlash, anchored on llama.cpp's own drafter: tools/reference/dflash35_anchor.py), on the same inputs:
  inject: synthetic target features [5, n, D] (the drafter's ops do not care where they come from) -> each drafter layer's
          ring rows K / V vs the reference's cache rows (worst relative error);
  draft : the block [last, MASK x 3] -> the logits rows 1 .. 3 (worst relative error) and the 3 drafts (equal, or reported
          with the reference's margin at the first difference);
  long  : --long N positions injected (default 4200: past the 2,048-position window and the 4,096-row ring's wrap), then a
          draft there against the reference (the window clamp per query and the ring's wrap);
  grids : the same inject + draft at 3, 6 and 7 workgroups per WGP: ring rows, logits and drafts bitwise the default grid's.
The target engine is cut after one block (Dense35Own(debug_layers=1)): the drafter reads only its head and embedding, so the
module's first run fits the non-display card alone. Prints PASS / FAIL lines and RESULT PASS | FAIL.
  tools/guard.sh LOG 1800 python -I tests/engine/test_q35_dflash.py SWIFT_GGUF DFLASH_GGUF [--n 20] [--long 4200] [--rel 0.03]"""
import argparse, gc, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'python'))
import numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument('gguf'); ap.add_argument('drafter'); ap.add_argument('--n', type=int, default=20)
ap.add_argument('--long', type=int, default=4200); ap.add_argument('--rel', type=float, default=0.03)
A = ap.parse_args()
from caminus import ggufio
from caminus.engine.q35_dense import Dense35Own
from caminus.engine.q35_dflash import Dflash35, RING
from caminus.ref_dflash import RefDFlash

print(torch.cuda.get_device_name(0), flush=True)
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


rel = lambda a, b: float(np.abs(np.asarray(a, np.float64) - b).max() / max(np.abs(b).max(), 1e-30))
vocab = int(ggufio.GGUFFile(A.gguf).T['output.weight'].shape[1])
dmib = Dflash35.need_mib(A.drafter, vocab)
eng = Dense35Own(A.gguf, max_ctx=A.long + 8, debug_layers=1, vram_extra_mib=dmib)
rng = np.random.default_rng(1)
GRIDS = (3, 6, 7)


def run(dr, feats, n0, last):
    """inject feats at 0 .., draft at n0: (ring rows [nl, n, 2, M] fp16 bits, logits rows bits, drafts)"""
    for d in dr.L: d['ring'].zero_()
    dr.inject(0, torch.from_numpy(feats)); torch.cuda.synchronize()   # (host rows: inject copies T rows a launch into its own projected buffer, no other VRAM)
    got = dr.draft(last, n0)
    M = dr.nkv * dr.hd; n = feats.shape[1]
    ring = np.stack([d['ring'].view(RING, 2, M)[:min(n, RING)].cpu().numpy().view(np.uint16) for d in dr.L])
    return ring, dr.buf['logits'].view(4, dr.vocab).cpu().numpy().copy(), got


def compare(dr, ref, feats, n0, last, what):
    n = feats.shape[1]
    ring, lg, got = run(dr, feats, n0, last)
    for c0 in range(0, n, 512): ref.inject(c0, feats[:, c0:c0 + 512])   # (chunks: the float64 rows stay small)
    M = dr.nkv * dr.hd; wk = wv = 0.0
    for il, d in enumerate(dr.L):   # (the last min(n, RING) positions: each in row p & (RING - 1))
        ps = np.arange(max(0, n - RING + 4), n)   # (the draft's block at n0 .. n0 + 3 rewrote the rows of n0 - RING ..: outside its window)
        rr = d['ring'].view(RING, 2, M)[torch.from_numpy(ps & (RING - 1))].float().cpu().numpy()
        wk = max(wk, rel(rr[:, 0].reshape(len(ps), dr.nkv, dr.hd), ref.K[il, ps])); wv = max(wv, rel(rr[:, 1].reshape(len(ps), dr.nkv, dr.hd), ref.V[il, ps]))
    check(wk < A.rel and wv < A.rel, f'{what}: inject of {n} rows x {dr.nl} layers, worst relative error K {wk:.2e}, V {wv:.2e} (bound {A.rel})')
    want, rlg, _ = ref.draft(last, n0, 3)
    wl = rel(lg[1:], rlg[1:])
    check(wl < A.rel, f'{what}: draft at {n0}: logits rows 1 .. 3, worst relative error {wl:.2e} (bound {A.rel})')
    if got == want: check(True, f'{what}: the 3 drafts {got} == the reference\'s')
    else:
        i = next(j for j in range(3) if got[j] != want[j]); top = np.sort(rlg[i + 1])[-2:]
        check(False, f'{what}: drafts {got} vs the reference\'s {want} (first difference at {i}: the reference\'s top-2 logit gap {top[1] - top[0]:.3e})')
    return ring, lg, got


dr = Dflash35(A.drafter, eng)
print(f'drafter: inject {len(dr.Ei.segs)} segments, draft {len(dr.Ed.segs)} segments, {dr.mib:.0f} MiB', flush=True)
nl = len(dr.layers_tgt)
short = (rng.standard_normal((nl, A.n, dr.D)) * 0.6).astype(np.float32)
base = compare(dr, RefDFlash(A.drafter, A.gguf, max_pos=A.n + 8), short, A.n, 220, 'short')
if A.long:
    longf = (rng.standard_normal((nl, A.long, dr.D)) * 0.6).astype(np.float32)
    compare(dr, RefDFlash(A.drafter, A.gguf, max_pos=A.long + 8), longf, A.long, 220, 'long (window + ring wrap)')
del dr; gc.collect(); torch.cuda.empty_cache()
for w in GRIDS:
    dw = Dflash35(A.drafter, eng, per_wgp=w)
    r = run(dw, short, A.n, 220)
    check(np.array_equal(r[0], base[0]) and np.array_equal(r[1].view(np.uint32), base[1].view(np.uint32)) and r[2] == base[2],
          f'{w} workgroups per WGP: ring rows, logits and drafts bitwise the default grid\'s')
    del dw; gc.collect(); torch.cuda.empty_cache()
print('RESULT ' + ('PASS' if ok else 'FAIL'), flush=True)
sys.exit(0 if ok else 1)
