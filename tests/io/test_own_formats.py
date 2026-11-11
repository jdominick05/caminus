"""CPU checks of Caminus's own format code that need no model file (the bitwise checks against the oracles on real files are
tests/io/test_own_dequant.py and tests/io/test_bpe.py):
  - caminus.own_dequant's decoders (Q4_K, Q5_K, Q6_K, Q8_0, Q4_0) on random blocks, bitwise against each format's definition
    written one value at a time (a value's index -> its bits in the block -> the float32 statement)
  - caminus.bpe.pretokenize: the qwen2 and the qwen35 pre-tokenizers on texts with combining marks, against the pieces
    their patterns give (qwen35 keeps a mark on its word; qwen2 cuts before it), and equal on text without marks
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  python -I tests/io/test_own_formats.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
from caminus import own_dequant
from caminus.bpe import pretokenize

ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


f32 = np.float32
rng = np.random.default_rng(5)


def fp16(b, o):
    return f32(np.frombuffer(bytes(b[o:o + 2]), np.float16)[0])


def k_scale_min(s, j):
    """sub-block j's 6-bit scale and min from the 12 bytes Q4_K and Q5_K share"""
    if j < 4: return s[j] & 63, s[j + 4] & 63
    return (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4)


def def_q4_k(b, i):
    d, dmin, s, qs = fp16(b, 0), fp16(b, 2), b[4:16], b[16:144]
    j, n, l = i // 32, i // 64, i % 32
    sc, m = k_scale_min(s, j)
    q = (qs[n * 32 + l] >> 4) if j % 2 else (qs[n * 32 + l] & 0xF)
    return (d * f32(sc)) * f32(q) - dmin * f32(m)


def def_q5_k(b, i):
    d, dmin, s, qh, qs = fp16(b, 0), fp16(b, 2), b[4:16], b[16:48], b[48:176]
    j, n, l = i // 32, i // 64, i % 32
    sc, m = k_scale_min(s, j)
    q = ((qs[n * 32 + l] >> 4) if j % 2 else (qs[n * 32 + l] & 0xF)) + 16 * ((qh[l] >> j) & 1)
    return (d * f32(sc)) * f32(q) - dmin * f32(m)


def def_q6_k(b, i):
    ql, qh, sc, d = b[0:128], b[128:192], np.frombuffer(bytes(b[192:208]), np.int8), fp16(b, 208)
    n, r = i // 128, i % 128
    qd, l = r // 32, r % 32
    lo = ql[n * 64 + l + 32 * (qd % 2)]
    q = ((lo >> 4) if qd >= 2 else (lo & 0xF)) | (((qh[n * 32 + l] >> (2 * qd)) & 3) << 4)
    return (d * f32(sc[n * 8 + 2 * qd + l // 16])) * f32(q - 32)


def def_q8_0(b, i):
    return fp16(b, 0) * f32(np.frombuffer(bytes(b[2:34]), np.int8)[i])


def def_q4_0(b, i):
    qs = b[2:18]
    return fp16(b, 0) * f32(((qs[i - 16] >> 4) if i >= 16 else (qs[i] & 0xF)) - 8)


for name, (qk, bs, fn), one, dpos in (('Q4_K', own_dequant.DECODERS['Q4_K'], def_q4_k, (0, 2)), ('Q5_K', own_dequant.DECODERS['Q5_K'], def_q5_k, (0, 2)),
                                      ('Q6_K', own_dequant.DECODERS['Q6_K'], def_q6_k, (208,)), ('Q8_0', own_dequant.DECODERS['Q8_0'], def_q8_0, (0,)),
                                      ('Q4_0', own_dequant.DECODERS['Q4_0'], def_q4_0, (0,))):
    nb = 24
    raw = rng.integers(0, 256, (nb, bs), dtype=np.uint8)
    for o in dpos:                                         # finite fp16 scales (a random bit pattern could be inf / nan)
        raw[:, o:o + 2] = rng.normal(0, 0.05, nb).astype(np.float16).view(np.uint8).reshape(nb, 2)
    raw[0] = 0; raw[1, [k for k in range(bs) if all(k not in (o, o + 1) for o in dpos)]] = 255   # the all-zero and all-one codes
    got = fn(raw.reshape(-1)); assert got.dtype == np.float32 and got.size == nb * qk
    with np.errstate(over='ignore'):
        blocks = [raw[k].tolist() for k in range(nb)]              # (Python ints: no uint8 wrap-around in q - 32)
        want = np.array([one(blocks[k], i) for k in range(nb) for i in range(qk)], np.float32)
    check(np.array_equal(got.view(np.uint32), want.view(np.uint32)) and np.array_equal(
        own_dequant.dequant(raw.reshape(-1), name, nb * qk).view(np.uint32), want.view(np.uint32)),
          f'{name}: {nb} random blocks ({nb * qk} values) bitwise the per-value definition')
try: own_dequant.dequant(np.zeros(64, np.uint8), 'IQ9_Z', 32); refused = False
except ValueError: refused = True
check(refused, 'a format without a decoder is refused by name')

M = '́'    # a combining acute accent (category Mn)
cases = [   # text, the qwen2 pieces, the qwen35 pieces
    ('a' + M + 'b', ['a', M + 'b'], ['a' + M + 'b']),
    (' ' + M + 'x', [' ' + M, 'x'], [' ' + M + 'x']),
    ('5' + M, ['5', M], ['5', M]),
    ('!' + M + '?', ['!' + M + '?'], ['!' + M, '?']),
    ('q' + M + ' z' + M + M, ['q', M, ' z', M + M], ['q' + M, ' z' + M + M]),
    (M, [M], [M]),
    ("x's" + M, ['x', "'s", M], ['x', "'s", M]),
    ('क्ष', ['क', '्ष'], ['क्ष']),          # Devanagari ka + virama + ssa
    ('-' + M + '\n', ['-' + M + '\n'], ['-' + M, '\n']),
]
bad = [(t, pretokenize(t), pretokenize(t, True)) for t, a, b in cases if pretokenize(t) != a or pretokenize(t, True) != b]
check(not bad, f'pretokenize: {len(cases)} texts with combining marks give the qwen2 and the qwen35 pieces ({bad[:2]})')
plain = ["Hello world! It's 3.14, don't   stop\r\n\n  x\ty", ' leading', 'trailing  ', '', 'a1b2 c3', "'ll 'RE x'd", '日本語 text、ok']
check(all(pretokenize(t) == pretokenize(t, True) and ''.join(pretokenize(t)) == t for t in plain),
      f'pretokenize: without marks the two agree, and the pieces rejoin to the text ({len(plain)} texts)')
check(all(''.join(pretokenize(t, mk)) == t for t, _, _ in cases for mk in (False, True)), 'pretokenize: the pieces rejoin to the text with marks too')
print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
