"""caminus.own_dequant (our host decoders) bitwise against gguf-py's dequantize (the oracle): first on --random random
blocks of every format it decodes (every bit pattern of the indices, signs and scales occurs; d is a small fp16), with
the codebooks of caminus.iq_tables against gguf-py's tables; then, given a file, on real tensors: up to --rows rows
each (the first, the middle and the last rows) of every tensor kind in the file, per format: for each (format, name
within a block) the first, the middle and the last block that has it (so a format only two tensors use is covered,
and a multi-token-prediction block after the trunk too), plus the head and the embedding. A format own_dequant has
no decoder for FAILs. Prints PASS / FAIL and RESULT PASS | FAIL.
  python -I tests/io/test_own_dequant.py [GGUF] [--rows 64] [--random 4096]"""
import argparse, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
from caminus import ggufio, iq_tables, own_dequant
from caminus.engine import own_layout
from caminus.config import ensure_gguf
ensure_gguf()
from gguf import quants, GGMLQuantizationType   # (the oracle only)
ap = argparse.ArgumentParser(); ap.add_argument('gguf', nargs='?'); ap.add_argument('--rows', type=int, default=64)
ap.add_argument('--random', type=int, default=4096, help='random blocks per format'); A = ap.parse_args()
bad = 0; nvals = 0
rng = np.random.default_rng(20261110)
for name, (qk, bs, _) in sorted(own_dequant.DECODERS.items()):
    raw = rng.integers(0, 256, (A.random, bs), dtype=np.uint8)
    for o in {'Q4_K': (0, 2), 'Q5_K': (0, 2), 'Q6_K': (208,), 'Q4_1': (0, 2), 'Q2_K': (80, 82), 'IQ1_M': ()}.get(name, (0,)):   # (the fp16 block scales: finite and small)
        raw[:, o:o + 2] = (rng.standard_normal(A.random) * 0.01).astype(np.float16).view(np.uint8).reshape(-1, 2)
    if name == 'IQ1_M':   # (its fp16 d is split over the top nibbles of the four scale words: a finite one, nibble by nibble)
        d = (rng.standard_normal(A.random) * 0.01).astype(np.float16).view(np.uint16)
        w = raw[:, 48:56].copy().view(np.uint16).reshape(-1, 4) & np.uint16(0x0FFF)
        for k in range(4): w[:, k] |= ((d >> np.uint16(4 * k)) & np.uint16(0xF)) << np.uint16(12)
        raw[:, 48:56] = w.view(np.uint8).reshape(-1, 8)
    ours = own_dequant.dequant(raw.reshape(-1), name, A.random * qk)
    ref = quants.dequantize(raw, GGMLQuantizationType[name]).astype(np.float32).reshape(-1)
    ok = np.array_equal(ours.view(np.uint32), ref.view(np.uint32)); bad += not ok
    print(('PASS' if ok else 'FAIL') + f' {name}: {A.random} random blocks bitwise gguf-py')
try:   # the fp32 reference's torch decoders (tools/reference/q3moe_ref32.py) are the host decoders' bits; skipped without torch
    import importlib.util, torch
    sp = importlib.util.spec_from_file_location('q3moe_ref32', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tools',
                                                                            'reference', 'q3moe_ref32.py'))
    r32 = importlib.util.module_from_spec(sp); sp.loader.exec_module(r32)
    for name, (qk, bs, f) in sorted(own_dequant.DECODERS.items()):
        if name not in r32.DECODERS: continue   # (the reference decodes the formats of the files it reads: not Q5_K)
        raw = rng.integers(0, 256, 300 * bs, dtype=np.uint8)
        with np.errstate(invalid='ignore'): a = f(raw)
        b = r32.DECODERS[name](torch.from_numpy(raw)).numpy(); fin = ~np.isnan(a)   # (random scales can be NaN: the same places; the bits elsewhere)
        ok = np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[fin].view(np.uint32), b[fin].view(np.uint32)) and fin.mean() > 0.9; bad += not ok
        print(('PASS' if ok else 'FAIL') + f' {name}: the fp32 reference\'s torch decoder is bitwise the host decoder (300 random blocks; NaN in the same places)')
except ImportError:
    print('SKIP the fp32 reference\'s torch decoders (no torch here)')
for name, (bs, rs, hs) in sorted(own_layout.IQ_REC.items()):   # the kernels' record layout: the same bytes, and back
    raw = rng.integers(0, 256, 6 * 3 * bs, dtype=np.uint8)
    rec = own_layout.iq_records(raw, 6, 768, name)
    ok = (rec.size == raw.size + own_layout.IQ_PAD and np.array_equal(own_layout.iq_from_records(rec, 6, 768, name), raw)
          and np.array_equal(np.sort(rec[:raw.size]), np.sort(raw))); bad += not ok
    print(('PASS' if ok else 'FAIL') + f' {name}: the record layout holds the blocks\' bytes and inverts')
for rows, K in ((6, 5120), (3, 96)):   # Q4_0 planes (kernels/own_quant_q4_0.h): codes | d, as the tile decodes them, and back
    raw = rng.integers(0, 256, rows * (K // 32) * 18, dtype=np.uint8)
    pl = own_layout.q4_0_planes(raw, rows, K)
    u = pl[:rows * K // 32 * 16].reshape(rows, K // 32, 16); dd = pl[rows * K // 32 * 16:].view(np.float16).reshape(rows, K // 32)
    q = np.concatenate([u & 15, u >> 4], -1).astype(np.float32) - 8    # unit: value k in byte k's low nibble, 16 + k high
    want = own_dequant.DECODERS['Q4_0'][2](raw).reshape(rows, K)
    got = (dd.astype(np.float32)[:, :, None] * q).reshape(rows, K)
    ok = (pl.size == raw.size and np.array_equal(own_layout.q4_0_from_planes(pl, rows, K), raw)
          and np.array_equal(np.isnan(got), np.isnan(want)) and np.array_equal(np.nan_to_num(got), np.nan_to_num(want)))
    try: own_layout.q4_0_planes(raw[:-1], rows, K); ok = False
    except ValueError: pass
    bad += not ok
    print(('PASS' if ok else 'FAIL') + f' Q4_0 [{rows} x {K}]: the planes decode to the blocks\' values, invert, and refuse a short tensor')
for name, (bits, bv, bs) in sorted(own_layout.RS_CONT.items()):   # the row-scale planes: the codes as the kernel reads them
    for K in (2048, 768):
        rows = 10; raw = rng.integers(0, 256, (rows, K // bv, bs), dtype=np.uint8)
        d = (rng.standard_normal(rows) * 0.01).astype(np.float16); raw[:, :, :2] = d.view(np.uint8).reshape(rows, 1, 2)
        codes, sc = own_layout.rs_planes(raw.reshape(-1), rows, K, name)
        ok = (codes.size == rows * K * bits // 8 and np.array_equal(sc, d.astype(np.float32))
              and np.array_equal(own_layout.rs_from_planes(codes, sc, rows, K, name), raw.reshape(-1)))
        if bits == 4:   # a unit of 32 values: 16 bytes, value b in byte b's low nibble, b + 16 in its high one; d (q - 8)
            u = codes.reshape(rows, K // 32, 16); q = np.concatenate([u & 15, u >> 4], -1).astype(np.float32) - 8
        else:           # a dword of 16 values: (dword >> 2 i) & 0x03030303 = values 4 i .. 4 i + 3 as four bytes; d (q - 1)
            dw = codes.reshape(rows, K // 16, 4).copy().view('<u4')[..., 0]
            q = np.stack([((dw >> (2 * i)) & 0x03030303).astype('<u4').view(np.uint8).reshape(rows, K // 16, 4) for i in range(4)], -2).astype(np.float32) - 1
        ok = ok and np.array_equal((sc[:, None] * q.reshape(rows, K)).astype(np.float32), own_dequant.DECODERS[name][2](raw.reshape(-1)).reshape(rows, K))
        two = raw.copy(); two[3, 5, 0] ^= 1
        try: own_layout.rs_planes(two.reshape(-1), rows, K, name); ok = False
        except ValueError: pass
        bad += not ok
        print(('PASS' if ok else 'FAIL') + f' {name} K {K}: the row-scale planes decode to the container\'s values, invert, and refuse a row with two scales')
for cls in (quants.IQ2_XXS, quants.IQ2_XS, quants.IQ2_S, quants.IQ3_XXS, quants.IQ3_S, quants.IQ1_S): cls.init_grid()
tabs = [('KVALUES_IQ4', iq_tables.KVALUES_IQ4, np.array(quants.IQ4_NL.kvalues)),
        ('GRID_IQ2_XXS', iq_tables.GRID_IQ2_XXS, quants.IQ2_XXS.grid.reshape(256, 8)),
        ('GRID_IQ2_XS', iq_tables.GRID_IQ2_XS, quants.IQ2_XS.grid.reshape(512, 8)),
        ('GRID_IQ2_S', iq_tables.GRID_IQ2_S, quants.IQ2_S.grid.reshape(1024, 8)),
        ('GRID_IQ1_S', iq_tables.GRID_IQ1_S, quants.IQ1_S.grid.reshape(2048, 8)),
        ('IQ1_DELTA', np.array([iq_tables.IQ1_DELTA]), np.array([quants.IQ1_S.delta])),
        ('GRID_IQ3_XXS', iq_tables.GRID_IQ3_XXS, quants.IQ3_XXS.grid.reshape(256, 4)),
        ('GRID_IQ3_S', iq_tables.GRID_IQ3_S, quants.IQ3_S.grid.reshape(512, 4)),
        ('the sign bytes', own_dequant._signs7(np.arange(128)), np.frombuffer(quants.IQ2_XXS.ksigns, np.uint8))]
for name, ours, ref in tabs:
    ok = ours.shape == ref.shape and np.array_equal(ours.astype(np.float64), ref.astype(np.float64)); bad += not ok
    print(('PASS' if ok else 'FAIL') + f' {name} equals the table of gguf-py')
hdr = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kernels', 'iq_grids.h')
ok = open(hdr).read() == iq_tables.c_header(); bad += not ok
print(('PASS' if ok else 'FAIL') + ' kernels/iq_grids.h is the tables\' own header (iq_tables.c_header)')
ok = open(os.path.join(os.path.dirname(hdr), 'iq2s_grid.h')).read() == iq_tables.c_header_iq2s(); bad += not ok
print(('PASS' if ok else 'FAIL') + ' kernels/iq2s_grid.h is the tables\' own packed IQ2_S header (iq_tables.c_header_iq2s)')
if A.gguf is None:
    print('RESULT', 'PASS' if bad == 0 else 'FAIL'); sys.exit(1 if bad else 0)
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
bad += len(missing); nvals = 0; per = {}
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
print('RESULT', 'PASS' if bad == 0 else 'FAIL'); sys.exit(1 if bad else 0)
