"""Guarded GPU byte gate for the staged own Q8 KV writer (no model support claim).

Run non-display-first under tools/guard.sh in an authorized GPU window:
  tools/guard.sh LOG 180 python -I tests/device/test_kv_codec_gpu.py --json OUT

Only valid input/address spans are launched. Unsafe negative span/layout cases
belong to test_kv_codec.py's CPU launch recorder. Numerical refusal is host-tested
in test_own_kv_codec.py; this gate checks the full valid reference domain, including
half subnormals, against independent reference bytes. A mismatch is a gate failure.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
spec = importlib.util.spec_from_file_location('codec_vectors', ROOT / 'tests/io/test_own_kv_codec.py')
vectors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vectors)


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--json', required=True, help='write measurements even when byte parity fails')
    a = ap.parse_args()
    import torch
    from caminus.device import capacity
    from caminus.device.kv_codec import Q8KVCodec
    from caminus.ref_kv import Q8KVLayout, encode_q8_kv

    dev = torch.device('cuda:0')
    torch.cuda.set_device(dev)
    # Pessimistic sum of every small allocation below, including staging/canaries.
    capacity.check([dev.index], [64.0], 'own Q8 KV byte gate (all staging and canaries)')
    codec = Q8KVCodec(dev)
    blocks = vectors.valid_blocks()
    evidence = dict(kind='own Q8 KV byte gate', cases=[], failures=[])
    completed = False
    try:
        for heads, dim in ((1, 32), (1, 128), (1, 256), (3, 128), (3, 256), (4, 256), (32, 256)):
            layout = Q8KVLayout(heads, dim)
            M, nb = layout.values, layout.values // 32
            T = (len(blocks) + 2 * nb - 1) // (2 * nb)
            source = np.resize(blocks, (T, 2, heads, dim)).astype(np.float32)
            k, v = source[:, 0], source[:, 1]
            want = encode_q8_kv(k, v, layout)
            # Different padded row strides exercise extent/indexing independently.
            ldk, ldv, pos0 = M + 7, M + 13, 2
            kin = torch.full((T, ldk), 999.0, dtype=torch.float32, device=dev)
            vin = torch.full((T, ldv), -999.0, dtype=torch.float32, device=dev)
            kin[:, :M].copy_(torch.from_numpy(k.copy().reshape(T, M)))
            vin[:, :M].copy_(torch.from_numpy(v.copy().reshape(T, M)))
            rows = T + pos0  # the final input row writes the last legal cache row
            # Nontrivial 2-byte-aligned cache pointer; guards outside the view too.
            backing = torch.full((rows * layout.row_bytes + 4,), 0xa5, dtype=torch.uint8, device=dev)
            cache = backing[2:-2]
            err = torch.zeros(1, dtype=torch.int32, device=dev)
            codec.append(kin, ldk, vin, ldv, M, cache, pos0, T, err)
            codec.check(err)
            got = cache.cpu().numpy().reshape(rows, layout.row_bytes)
            padding = np.concatenate((got[:pos0].reshape(-1), got[pos0 + T:].reshape(-1),
                                      backing[:2].cpu().numpy(), backing[-2:].cpu().numpy()))
            actual = got[pos0:pos0 + T]
            equal, guards = np.array_equal(actual, want), bool((padding == 0xa5).all())
            record = dict(heads=heads, dim=dim, T=T, ldk=ldk, ldv=ldv, pos0=pos0,
                          row_bytes=layout.row_bytes, input_sha256=sha(source),
                          expected_sha256=sha(want), actual_sha256=sha(actual),
                          byte_equal=equal, guards_unchanged=guards)
            if not equal:
                first = np.argwhere(actual != want)[0]
                record['first_mismatch'] = dict(row=int(first[0]), byte=int(first[1]),
                                                expected=int(want[tuple(first)]), actual=int(actual[tuple(first)]))
            evidence['cases'].append(record)
            if not equal or not guards:
                evidence['failures'].append(f'{heads} heads x {dim}: bytes {equal}, guards {guards}')
            print(f'{"PASS" if equal and guards else "FAIL"} {heads} heads x {dim}, {T} rows: bytes {equal}, guards {guards}', flush=True)

            # The identical valid input as a prefix and tail, with a fresh cache.
            chunked = torch.full_like(cache, 0xa5)
            cuts = sorted(set((0, 1, T // 2, T)))
            for lo, hi in zip(cuts, cuts[1:]):
                if hi > lo:
                    codec.append(kin[lo:], ldk, vin[lo:], ldv, M, chunked, pos0 + lo, hi - lo, err)
            codec.check(err)
            identical = torch.equal(cache, chunked)
            record['chunk_bytes_equal'] = identical
            if not identical:
                evidence['failures'].append(f'{heads} heads x {dim}: chunk bytes differ')
            print(f'{"PASS" if identical else "FAIL"} chunk cuts {cuts}: all cache bytes identical', flush=True)
        completed = True
    except Exception as exc:
        evidence['failures'].append(f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if not completed and not evidence['failures']:
            evidence['failures'].append('gate interrupted before completion')
        out = Path(a.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(evidence, indent=2) + '\n')
        print('RESULT ' + ('FAIL' if evidence['failures'] else 'PASS'), flush=True)
    if evidence['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
