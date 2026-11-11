"""caminus.own_dequant (our host decoders) bitwise against gguf-py's dequantize (the oracle) on real tensors: up to
--rows rows each (the first, the middle and the last rows) of every tensor kind in the file, per format: for each (format,
name within a block) the first, the middle and the last block that has it (so a format only two tensors use is covered,
and a multi-token-prediction block after the trunk too), plus the head and the embedding. A format own_dequant has no
decoder for FAILs. Prints PASS / FAIL and RESULT PASS | FAIL.
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
HAVE = set(own_dequant.DECODERS) | {'F32', 'F16'}
groups = {}
for t in F.tensors:
    m = re.match(r'blk\.\d+\.(.*)', t.name)
    groups.setdefault((t.type, m.group(1) if m else t.name), []).append(t)
pick = [g[i] for g in groups.values() for i in sorted({0, len(g) // 2, len(g) - 1})]
missing = sorted({t.type for t in pick} - HAVE)
if missing: print(f'FAIL no own decoder for {missing}')
pick = [t for t in pick if t.type in HAVE]
bad = len(missing); nvals = 0; per = {}
for t in pick:
    R, rb = t.rows(); K = t.shape[0]
    for r0 in sorted({0, R // 2, max(0, R - A.rows)}):
        r1 = min(R, r0 + A.rows)
        ours = own_dequant.rows(t, r0, r1)
        raw = np.asarray(t.data()[r0 * rb:r1 * rb]).reshape(r1 - r0, rb)
        ref = (raw.view(np.float32).reshape(r1 - r0, K) if t.type == 'F32' else
               raw.view(np.float16).astype(np.float32).reshape(r1 - r0, K) if t.type == 'F16' else
               quants.dequantize(raw, GGMLQuantizationType[t.type]).astype(np.float32).reshape(r1 - r0, K))
        if not np.array_equal(ours.view(np.uint32), ref.view(np.uint32)):
            bad += 1; print(f'  differs: {t.name} rows {r0}..{r1}: max |d| {np.abs(ours - ref).max():.3e}')
        nvals += ours.size; per[t.type] = per.get(t.type, 0) + ours.size
print(('PASS' if bad == 0 else f'FAIL {bad}') + f' own_dequant bitwise gguf-py on {len(pick)} tensors ({nvals / 1e6:.1f} M values: ' +
      ', '.join(f'{k} {v / 1e6:.1f} M' for k, v in sorted(per.items())) + ')')
print('RESULT', 'PASS' if bad == 0 else 'FAIL')
