"""caminus.own_dequant (our host decoders) bitwise against gguf-py's dequantize (the oracle) on real tensors: up to
--rows rows each of every Q4_K / Q6_K / F32 tensor kind in the file (layer 0, a middle and the last layer, the experts'
first, a middle and the last expert, the head and the embedding). Prints PASS / FAIL and RESULT PASS | FAIL.
  python -I tests/io/test_own_dequant.py GGUF [--rows 64]"""
import argparse, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
from caminus import ggufio, own_dequant
from caminus.config import ensure_gguf
ensure_gguf()
from gguf import quants, GGMLQuantizationType   # (the oracle only)
ap = argparse.ArgumentParser(); ap.add_argument('gguf'); ap.add_argument('--rows', type=int, default=64); A = ap.parse_args()
F = ggufio.GGUFFile(A.gguf)
nl = int(F.hp('block_count') or 1)
pick = [t for t in F.tensors if t.type in ('Q4_K', 'Q6_K', 'F32') and
        (not re.match(r'blk\.\d+\.', t.name) or int(t.name.split('.')[1]) in (0, nl // 2, nl - 1))]
bad = 0; nvals = 0
for t in pick:
    R, rb = t.rows(); K = t.shape[0]
    for r0 in sorted({0, R // 2, max(0, R - A.rows)}):
        r1 = min(R, r0 + A.rows)
        ours = own_dequant.rows(t, r0, r1)
        raw = np.asarray(t.data()[r0 * rb:r1 * rb]).reshape(r1 - r0, rb)
        ref = (raw.view(np.float32).reshape(r1 - r0, K) if t.type == 'F32' else
               quants.dequantize(raw, GGMLQuantizationType[t.type]).astype(np.float32).reshape(r1 - r0, K))
        if not np.array_equal(ours.view(np.uint32), ref.view(np.uint32)):
            bad += 1; print(f'  differs: {t.name} rows {r0}..{r1}: max |d| {np.abs(ours - ref).max():.3e}')
        nvals += ours.size
print(('PASS' if bad == 0 else f'FAIL {bad}') + f' own_dequant bitwise gguf-py on {len(pick)} tensors ({nvals / 1e6:.1f} M values)')
print('RESULT', 'PASS' if bad == 0 else 'FAIL')
