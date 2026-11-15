"""Compile the actual own KV header as CPU C++ and compare with the numpy oracle.

python -I tests/io/test_own_kv_codec.py --scratch DIR

No HIP runtime, GPU, model or generated gate reference. This establishes host
source parity only; the separate guarded GPU gate is required before device use.
"""
import argparse
import ctypes
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from caminus.ref_kv import Q8KVLayout, encode_q8_kv


def valid_blocks():
    """Fixed difficult values plus seeded finite blocks, shared by the GPU gate."""
    def bits(word):
        return np.array([word], np.uint32).view(np.float32)[0]

    rows = []
    def add(values):
        row = np.zeros(32, np.float32)
        row[:len(values)] = values
        rows.append(row)

    add([0, -0.0])
    add([127, .5, -.5, 1.5, -1.5, 2.5, -2.5,
         np.nextafter(np.float32(.5), np.float32(0)),
         -np.nextafter(np.float32(.5), np.float32(0)),
         np.nextafter(np.float32(.5), np.float32(1))])
    add([bits(0x42fe1382), bits(0x427e0a3d), -bits(0x427e0a3d)])
    for scale in (0x3f801000, 0x3f803000):
        add([bits(scale) * np.float32(127)])
    add([3, bits(0x3c418306), -bits(0x3c418306)])
    add([np.float32(1.27e-6), -np.float32(1.27e-6)])
    add([np.float32(127 * 2**-24), -np.float32(127 * 2**-24)])
    # Nonzero fp32-subnormal d, with a finite reciprocal: do not silently flush
    # this accepted reference-domain block to zero before generating the codes.
    for exponent in (-121, -120):
        add([np.float32(2.0**exponent), -np.float32(2.0**exponent)])
    add([np.float32(127 * 65504), -np.float32(127 * 65504)])
    # A finite scale just below the half-overflow rounding midpoint is legal.
    add([np.nextafter(np.float32(65520), np.float32(0)) * np.float32(127)])
    rng = np.random.default_rng(8103)
    random = rng.uniform(-1, 1, size=(4096, 32)).astype(np.float32)
    random *= np.exp2(rng.integers(-90, 23, size=(4096, 1))).astype(np.float32)
    return np.ascontiguousarray(np.concatenate((np.stack(rows), random)))


def invalid_blocks():
    rows = []
    for value in (np.nan, np.inf, -np.inf, np.finfo(np.float32).max,
                  np.nextafter(np.float32(0), np.float32(1)),
                  np.float32(2**-128), np.float32(65520 * 127)):
        # Last lane catches implementations that ignore a nonfinite input in amax.
        row = np.zeros(32, np.float32)
        row[-1] = value
        rows.append(row)
    return np.stack(rows)


def oracle(blocks):
    assert len(blocks) % 2 == 0
    return encode_q8_kv(blocks[::2, None, :], blocks[1::2, None, :],
                        Q8KVLayout(1, 32)).reshape(-1, 34)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scratch', required=True)
    a = ap.parse_args()
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    if not compiler or not shutil.which(compiler[0]):
        print('SKIP own KV host codec: C++ compiler unavailable')
        return
    scratch = Path(a.scratch).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    src, lib = scratch / 'codec.cpp', scratch / 'codec.so'
    src.write_text('''#include "own_kv_codec.h"
extern "C" void encode(const float *x, unsigned char *y, unsigned char *ok, int n) {
    for (int i = 0; i < n; i++)
        ok[i] = own_kv_q8_encode(x + (size_t)i * 32, (OwnKvQ8Block *)(y + (size_t)i * 34));
}
''')
    subprocess.run(compiler + ['-std=c++17', '-O3', '-fno-fast-math', '-ffp-contract=off',
                              '-shared', '-fPIC', '-I', str(ROOT / 'kernels'),
                              str(src), '-o', str(lib)], check=True, timeout=60)
    fn = ctypes.CDLL(str(lib)).encode
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    fn.restype = None
    good, bad = valid_blocks(), invalid_blocks()
    blocks = np.concatenate((good, bad))
    raw = np.full((len(blocks), 34), 0xa5, np.uint8)
    accepted = np.zeros(len(blocks), np.uint8)
    fn(blocks.ctypes.data, raw.ctypes.data, accepted.ctypes.data, len(blocks))
    assert accepted[:len(good)].all(), 'valid host blocks refused'
    want = oracle(good)
    assert np.array_equal(raw[:len(good)], want), 'compiled host bytes differ from independent numpy source oracle'
    print(f'PASS compiled own header: {len(good)} blocks byte-identical to numpy (ties, scale rounding, zeros, subnormals, random)')
    assert not accepted[len(good):].any(), 'out-of-domain block accepted'
    assert (raw[len(good):] == 0xa5).all(), 'refused block changed destination bytes'
    for b in bad:
        try:
            oracle(np.stack((b, b)))
        except ValueError:
            continue
        raise AssertionError('invalid fixture was accepted by independent numpy reference')
    print(f'PASS {len(bad)} nonfinite/scale-domain refusals leave block bytes untouched')
    print('RESULT PASS (CPU source parity; GPU parity unverified)')


if __name__ == '__main__':
    main()
