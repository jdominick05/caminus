"""caminus.own_dequant (our host decoders) bitwise against gguf-py's dequantize (the oracle): first on --random random
blocks of every format it decodes (every bit pattern of the indices, signs and scales occurs; d is a small fp16), with
the codebooks of caminus.iq_tables against gguf-py's tables; then, given a file, on real tensors: up to --rows rows each
of every tensor kind in a decoded format (layer 0, a middle and the last layer, the experts' first, a middle and the last
expert, the head and the embedding). Prints PASS / FAIL and RESULT PASS | FAIL.
  python -I tests/io/test_own_dequant.py [GGUF] [--rows 64] [--random 4096]"""
import argparse, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
from caminus import ggufio, iq_tables, own_dequant
from caminus.config import ensure_gguf
ensure_gguf()
from gguf import quants, GGMLQuantizationType   # (the oracle only)
ap = argparse.ArgumentParser(); ap.add_argument('gguf', nargs='?'); ap.add_argument('--rows', type=int, default=64)
ap.add_argument('--random', type=int, default=4096, help='random blocks per format'); A = ap.parse_args()
bad = 0; nvals = 0
rng = np.random.default_rng(20261110)
for name, (qk, bs, _) in sorted(own_dequant.DECODERS.items()):
    raw = rng.integers(0, 256, (A.random, bs), dtype=np.uint8)
    for o in {'Q4_K': (0, 2), 'Q6_K': (208,)}.get(name, (0,)):          # (the fp16 block scales: finite and small)
        raw[:, o:o + 2] = (rng.standard_normal(A.random) * 0.01).astype(np.float16).view(np.uint8).reshape(-1, 2)
    ours = own_dequant.dequant(raw.reshape(-1), name, A.random * qk)
    ref = quants.dequantize(raw, GGMLQuantizationType[name]).astype(np.float32).reshape(-1)
    ok = np.array_equal(ours.view(np.uint32), ref.view(np.uint32)); bad += not ok
    print(('PASS' if ok else 'FAIL') + f' {name}: {A.random} random blocks bitwise gguf-py')
for cls in (quants.IQ2_XXS, quants.IQ3_XXS, quants.IQ3_S): cls.init_grid()
tabs = [('KVALUES_IQ4', iq_tables.KVALUES_IQ4, np.array(quants.IQ4_NL.kvalues)),
        ('GRID_IQ2_XXS', iq_tables.GRID_IQ2_XXS, quants.IQ2_XXS.grid.reshape(256, 8)),
        ('GRID_IQ3_XXS', iq_tables.GRID_IQ3_XXS, quants.IQ3_XXS.grid.reshape(256, 4)),
        ('GRID_IQ3_S', iq_tables.GRID_IQ3_S, quants.IQ3_S.grid.reshape(512, 4)),
        ('the sign bytes', own_dequant._signs7(np.arange(128)), np.frombuffer(quants.IQ2_XXS.ksigns, np.uint8))]
for name, ours, ref in tabs:
    ok = ours.shape == ref.shape and np.array_equal(ours.astype(np.float64), ref.astype(np.float64)); bad += not ok
    print(('PASS' if ok else 'FAIL') + f' {name} equals the table of gguf-py')
if A.gguf is None:
    print('RESULT', 'PASS' if bad == 0 else 'FAIL'); sys.exit(1 if bad else 0)
F = ggufio.GGUFFile(A.gguf)
nl = int(F.hp('block_count') or 1)
pick = [t for t in F.tensors if (t.type in own_dequant.DECODERS or t.type == 'F32') and
        (not re.match(r'blk\.\d+\.', t.name) or int(t.name.split('.')[1]) in (0, nl // 2, nl - 1))]
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
print('RESULT', 'PASS' if bad == 0 else 'FAIL'); sys.exit(1 if bad else 0)
