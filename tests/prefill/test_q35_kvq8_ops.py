"""Guarded valid-input gate for the staged own Q8 dense prefill object.

Run non-display-first in an authorized GPU window, after build/objcheck:
  tools/guard.sh LOG 900 python -I tests/prefill/test_q35_kvq8_ops.py --json OUT.json

HIP_VISIBLE_DEVICES must name one physical GPU. --allow-display permits a later
authorized display-card repeat; the first run must use a non-display card.
--cpu-fixtures checks reference fixtures and row-choice geometry only, with no
torch import, HIP initialization or device work. --help is also CPU-only.

Append bytes use caminus.ref_kv's independent codec. Attention is compared to
float64 over those same decoded bytes, with full fp32 queries. The acceptance
limits are the existing test_q35_prefill_attn.py limits: max <2e-2 and mean <2e-3
of reference RMS. No tolerance or reference is generated from device output.
Only valid spans, buffers and numerical codec inputs are launched. Refusals and
unsafe address cases belong to CPU launcher tests.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from caminus.ref_kv import Q8KVLayout, decode_q8_kv, encode_q8_kv

D = 256
MAX_REL, MEAN_REL = 2e-2, 2e-3   # existing dense prefill attention gate; unchanged


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def fixtures():
    """Independent finite CPU inputs, including valid wire values wider than half."""
    cases = [('zero', 1, 0, 17), ('boundary32', 4, 31, 18), ('odd', 1, 601, 19),
             ('deep', 1, 8191, 9), ('peaked', 1, 5000, 17),
             ('wide-k-tail', 1, 31, 17), ('wide-v-tail', 1, 31, 17),
             ('wide-q', 1, 31, 17), ('wide-v-all', 1, 3, 9), ('literal-signed', 1, 31, 17)]
    for seed, (name, nkv, pos0, n) in enumerate(cases, 8120):
        rng = np.random.default_rng(seed)
        heads, rows = 6 * nkv, pos0 + n + 3
        k = (rng.standard_normal((rows, nkv, D)) * .5).astype(np.float32)
        v = (rng.standard_normal((rows, nkv, D)) * .5).astype(np.float32)
        q = (rng.standard_normal((n, heads, D)) * .15).astype(np.float32)
        if name == 'peaked':
            u = rng.choice(np.array([-1, 1], np.float32), D)
            q[:] = u * .25 + (rng.standard_normal(q.shape) * .02).astype(np.float32)
            k[7, 0] = 2 * u
            k[pos0 + 8, 0] = 2 * u
        if name == 'wide-k-tail':
            k[pos0 + 7, 0] = np.float32(131072)
        if name == 'wide-v-tail':
            v[pos0 + 7, 0] = np.float32(131072)
        if name == 'wide-q':
            q[7, 0] = np.float32(70000)
        if name == 'wide-v-all':
            v[:] = np.float32(131072)
        layout = Q8KVLayout(nkv, D)
        raw = encode_q8_kv(k, v, layout)
        if name == 'literal-signed':
            # Literal finite decode domain includes negative scales and -128
            # codes, wider than encode's positive-scale [-127,127] domain.
            blocks = raw.reshape(rows, 2, nkv, D // 32, 34)
            block = blocks[pos0 + 7, 1, 0, 0]
            block[:2] = np.array([-1024], dtype='<f2').view(np.uint8)
            block[2:] = np.int8(-128).view(np.uint8)
        dk, dv = decode_q8_kv(raw, layout)
        yield dict(name=name, nkv=nkv, heads=heads, pos0=pos0, n=n, rows=rows,
                   k=k, v=v, q=q, raw=raw, decoded_k=dk, decoded_v=dv, layout=layout)


def unsafe_rows(q, k, v, pos0):
    """Expected FP32 fallback per query row/KV head, independent of chunk extent."""
    n, heads, _ = q.shape
    nkv, group = k.shape[1], heads // k.shape[1]
    with np.errstate(over='ignore', invalid='ignore'):
        kv_bad = (~np.isfinite(k.astype(np.float16)) | ~np.isfinite(v.astype(np.float16))).any(axis=2)
        q_bad = ~np.isfinite(q.astype(np.float16)).reshape(n, nkv, group, D).all(axis=(2, 3))
    return q_bad | np.maximum.accumulate(kv_bad, axis=0)[pos0:pos0 + n]


def ref_rows(case, rows):
    """Full-value float64 oracle over literal decoded Q8 rows, before half staging."""
    k, v = case['decoded_k'].astype(np.float64), case['decoded_v'].astype(np.float64)
    q, group = case['q'].astype(np.float64), case['heads'] // case['nkv']
    out = np.empty((len(rows), case['heads'], D), np.float64)
    for i, r in enumerate(rows):
        end = case['pos0'] + r + 1
        for h in range(case['heads']):
            scores = k[:end, h // group] @ q[r, h] / np.sqrt(D)
            weights = np.exp(scores - scores.max())
            out[i, h] = (weights @ v[:end, h // group]) / weights.sum()
    return out


def relative(a, b):
    rms = float(np.sqrt(np.mean(b * b)))
    delta = np.abs(a - b)
    if rms == 0:
        return (0., 0.) if not delta.any() else (float('inf'), float('inf'))
    return float(delta.max() / rms), float(delta.mean() / rms)


def attention_verdict(actual, reference):
    """The GPU verdict checks every row and refuses an unwritten/nonfinite row."""
    if actual.shape != reference.shape:
        return False, float('inf'), float('inf')
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        return False, float('inf'), float('inf')
    mx, mn = relative(actual, reference)
    return mx < MAX_REL and mn < MEAN_REL, mx, mn


def check(evidence, name, passed, **details):
    passed = bool(passed)
    # A failed device run can produce NaN/inf error measurements. Keep the JSON
    # valid and write the failure report instead of failing its serialization.
    details = {k: (str(v) if isinstance(v, float) and not np.isfinite(v) else v) for k, v in details.items()}
    evidence['checks'].append(dict(name=name, passed=passed, **details))
    if not passed:
        evidence['failures'].append(name)
    print(('PASS ' if passed else 'FAIL ') + name, flush=True)


def cpu_fixtures(evidence):
    for case in fixtures():
        name, n = case['name'], case['n']
        mask = unsafe_rows(case['q'], case['decoded_k'], case['decoded_v'], case['pos0'])
        check(evidence, name + ': finite full-value fixture',
              np.isfinite(case['q']).all() and np.isfinite(case['decoded_k']).all()
              and np.isfinite(case['decoded_v']).all(), wire_sha256=sha(case['raw']))
        for cut in sorted({1, 5, n // 2, n - 1}):
            joined = np.concatenate((unsafe_rows(case['q'][:cut], case['decoded_k'], case['decoded_v'], case['pos0']),
                                     unsafe_rows(case['q'][cut:], case['decoded_k'], case['decoded_v'], case['pos0'] + cut)))
            check(evidence, f'{name}: fallback selection independent of cut {cut}', np.array_equal(mask, joined))
        if name.startswith('wide-') or name == 'literal-signed':
            check(evidence, name + ': valid half-overflow domain exercised', mask.any())
        else:
            check(evidence, name + ': normal WMMA domain exercised', not mask.any())
        if name in ('wide-k-tail', 'wide-v-tail', 'literal-signed'):
            check(evidence, name + ': earlier causal rows remain on WMMA', not mask[:7].any() and mask[7:].all())
        if name == 'wide-q':
            check(evidence, name + ': query overflow affects only its row', mask.sum() == 1 and mask[7, 0])
        if name == 'wide-v-all':
            ref = ref_rows(case, [0, n - 1])
            check(evidence, name + ': float64 result preserves values above half range',
                  np.isfinite(ref).all() and (np.abs(ref) > np.finfo(np.float16).max).all())
    check(evidence, 'CPU fixtures imported no torch', 'torch' not in sys.modules)
    # Row1 was not among the former [0,n//2,n-1] sample. Exercise the exact
    # verdict used by the device gate, including omission and nonfinite cases.
    reference = np.full((9, 6, D), 1.25, np.float64)
    check(evidence, 'complete output verdict accepts the exact all-row reference',
          attention_verdict(reference.copy(), reference)[0])
    for label, value in (('unwritten NaN sentinel', np.nan), ('nonfinite infinity', np.inf),
                         ('wrong finite sentinel', 193.25)):
        damaged = reference.copy()
        damaged[1] = value
        check(evidence, f'complete output verdict rejects {label} in formerly unsampled row1',
              not attention_verdict(damaged, reference)[0])


def selected_card(allow_display):
    from caminus.cmd.guarded import under_guard
    from caminus.device import host
    if not under_guard():
        raise RuntimeError('GPU gate requires an outer tools/guard.sh in an authorized GPU window')
    visible = os.environ.get('HIP_VISIBLE_DEVICES', '')
    if not re.fullmatch(r'[0-9]+', visible):
        raise RuntimeError('GPU gate requires HIP_VISIBLE_DEVICES naming one physical GPU index')
    if any(os.environ.get(k) for k in ('ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')):
        raise RuntimeError('GPU gate requires unambiguous HIP_VISIBLE_DEVICES without other visibility overrides')
    matches = [g for g in host.gpus() if g['hip'] == int(visible) and g['status'] in ('supported', 'untested')]
    if len(matches) != 1:
        raise RuntimeError('selected GPU could not be verified from sysfs/KFD before HIP initialization')
    card = matches[0]
    if card['display'] is not False and not allow_display:
        raise RuntimeError('new code requires a verified non-display card first; later repeats use --allow-display explicitly')
    return {k: card[k] for k in ('card', 'pci', 'hip', 'gfx', 'display')}


def provenance():
    from caminus.device import kernels
    directory, where = kernels.kernel_dir()
    out = dict(kernel_source=where, objects={})
    for name in ('q35_prefill_kvq8', 'prefill_attn_q35d'):
        path = Path(directory) / (name + '.hsaco')
        out['objects'][name] = dict(sha256=kernels.sha256(path), bytes=path.stat().st_size)
    info = Path(directory) / 'BUILD-INFO'
    if info.exists():
        values = dict(line.rstrip('\n').split('\t', 1) for line in info.read_text().splitlines() if '\t' in line)
        out['build_info'] = {k: values.get(k) for k in ('source_commit', 'source_dirty', 'kernel_tree_sha256', 'arch',
                                                       'toolchain', 'toolchain_version', 'compiler')}
    try:
        out['gate_commit'] = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        out['gate_commit'] = None
    out['gate_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return out


def gpu_gate(evidence, allow_display):
    # No torch import, cuda query, module load or allocation precedes this check.
    evidence['card'] = selected_card(allow_display)
    evidence['provenance'] = provenance()
    import torch
    from caminus.device import capacity
    from caminus.prefill.attn import Attn
    from caminus.prefill.q35_kvq8 import Ops
    dev = torch.device('cuda:0')
    capacity.check([0], [128.0], 'Q8 prefill op gate (all device fixtures, outputs and canaries)')
    torch.cuda.set_device(dev)
    torch.cuda.set_per_process_memory_fraction(.85, dev)
    evidence['gpu_executed'] = True
    evidence['runtime'] = dict(torch=torch.__version__, hip=getattr(torch.version, 'hip', None), name=torch.cuda.get_device_name(dev))
    ops, attn = Ops(dev), Attn(dev, name='prefill_attn_q35d')

    def put(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(dev)

    def guarded_output(n, heads):
        backing = torch.full((n * heads * D + 8,), 193.25, dtype=torch.float32, device=dev)
        output = backing[4:-4]
        output.fill_(float('nan'))   # an omitted write cannot look like a finite valid result
        return backing, output

    def output_guards(backing):
        return bool(torch.all(backing[:4] == 193.25) and torch.all(backing[-4:] == 193.25))

    def attend(q, cache, pos0, n, heads, nkv, out, rows, force=False, split=False):
        b0, b1 = pos0 // 8, (pos0 + n + 7) // 8
        blocks = [(b, b + 1) for b in range(b0, b1)] if split else [(b0, b1)]
        for lo, hi in blocks:
            ops.attn_wm(q, cache, pos0, n, heads, nkv, D, 1 / np.sqrt(D), out, rows, lo, hi, force_fp32=force)

    # Existing standalone transform, also used by the new high-level prefill.
    rng = np.random.default_rng(912)
    for width in (64, 256):
        x = rng.standard_normal((7, width)).astype(np.float32)
        expected = x.copy()
        for stride in (1 << j for j in range(width.bit_length() - 1)):
            view = expected.reshape(7, -1, 2, stride)
            a, b = view[:, :, 0].copy(), view[:, :, 1].copy()
            view[:, :, 0], view[:, :, 1] = a + b, a - b
        expected *= np.float32(1 / np.sqrt(width))
        actual = put(x)
        attn.hadamard(actual, width, len(x))
        torch.cuda.synchronize(dev)
        check(evidence, f'Hadamard{width}: independent fp32 butterflies', np.array_equal(actual.cpu().numpy().view(np.uint32), expected.view(np.uint32)))

    # The actual shared writer, including padded source strides, an offset cache
    # view aligned to only two bytes, last-legal-row writes and external canaries.
    spec = importlib.util.spec_from_file_location('codec_vectors', ROOT / 'tests/io/test_own_kv_codec.py')
    vectors = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vectors)
    difficult = vectors.valid_blocks()
    for nkv in (1, 4):
        layout = Q8KVLayout(nkv, D)
        m, nb = layout.values, 2 * layout.values // 32
        n = (len(difficult) + nb - 1) // nb
        source = np.resize(difficult, (n, 2, nkv, D)).astype(np.float32)
        expected = encode_q8_kv(source[:, 0], source[:, 1], layout)
        for pos0 in (0, 31, 32):
            rows, ldk, ldv = pos0 + n, m + 7, m + 13
            k = torch.full((n, ldk), 999., dtype=torch.float32, device=dev)
            v = torch.full((n, ldv), -999., dtype=torch.float32, device=dev)
            k[:, :m].copy_(put(source[:, 0].reshape(n, m)))
            v[:, :m].copy_(put(source[:, 1].reshape(n, m)))
            backing = torch.full((rows * layout.row_bytes + 4,), 0xa5, dtype=torch.uint8, device=dev)
            cache, err = backing[2:-2], torch.zeros(1, dtype=torch.int32, device=dev)
            ops.append(k, ldk, v, ldv, cache, m, pos0, n, rows, err)
            torch.cuda.synchronize(dev)
            got = cache.cpu().numpy().reshape(rows, layout.row_bytes)
            guards = bool((got[:pos0] == 0xa5).all() and (backing[:2].cpu().numpy() == 0xa5).all()
                          and (backing[-2:].cpu().numpy() == 0xa5).all())
            equal = np.array_equal(got[pos0:], expected)
            check(evidence, f'append {nkv} KV heads at {pos0}: exact bytes and guards',
                  int(err.item()) == 0 and equal and guards, expected_sha256=sha(expected), actual_sha256=sha(got[pos0:]))
            backing2 = torch.full_like(backing, 0xa5)
            chunked = backing2[2:-2]
            for lo, hi in zip((0, 1, n // 2), (1, n // 2, n)):
                if hi > lo:
                    ops.append(k[lo:], ldk, v[lo:], ldv, chunked, m, pos0 + lo, hi - lo, rows, err)
            torch.cuda.synchronize(dev)
            check(evidence, f'append {nkv} KV heads at {pos0}: chunk bytes and guards identical',
                  int(err.item()) == 0 and torch.equal(backing, backing2))

    for case in fixtures():
        name, n, heads, nkv, pos0, rows = (case[k] for k in ('name', 'n', 'heads', 'nkv', 'pos0', 'rows'))
        raw = case['raw']
        backing = torch.full((raw.size + 4,), 0xa5, dtype=torch.uint8, device=dev)
        cache = backing[2:-2]
        cache.copy_(put(raw.reshape(-1)))
        q = put(case['q'].reshape(-1))
        auto_back, auto = guarded_output(n, heads)
        full_back, full = guarded_output(n, heads)
        attend(q, cache, pos0, n, heads, nkv, auto, rows)
        attend(q, cache, pos0, n, heads, nkv, full, rows, force=True)
        torch.cuda.synchronize(dev)
        checked_rows = list(range(n))   # include every deep/peaked row, not a sampled subset
        reference = ref_rows(case, checked_rows)
        for label, output in (('auto', auto), ('forced-fp32', full)):
            actual = output.view(n, heads, D).cpu().numpy().astype(np.float64)
            check(evidence, f'{name}: {label} all output rows are finite', np.isfinite(actual).all())
            passed, mx, mn = attention_verdict(actual, reference)
            check(evidence, f'{name}: {label} vs full-value float64',
                  passed, max_relative=mx, mean_relative=mn, rows=checked_rows, wire_sha256=sha(raw))
        mask = unsafe_rows(case['q'], case['decoded_k'], case['decoded_v'], pos0)
        if mask.any():
            indices = np.repeat(mask, heads // nkv, axis=1)
            a = auto.view(n, heads, D).cpu().numpy().view(np.uint32)
            b = full.view(n, heads, D).cpu().numpy().view(np.uint32)
            check(evidence, name + ': unsafe causal rows use exact forced-FP32 result', np.array_equal(a[indices], b[indices]))
        if name == 'wide-v-all':
            check(evidence, name + ': finite output above half range preserved',
                  bool(torch.isfinite(auto).all() and (auto.abs() > np.finfo(np.float16).max).all()))

        split_back, split = guarded_output(n, heads)
        attend(q, cache, pos0, n, heads, nkv, split, rows, split=True)
        torch.cuda.synchronize(dev)
        check(evidence, name + ': split launches write finite values to every output row', torch.isfinite(split).all())
        check(evidence, name + ': one absolute block per launch is bitwise identical', torch.equal(auto.view(torch.int32), split.view(torch.int32)))
        all_output_guards = output_guards(auto_back) and output_guards(full_back) and output_guards(split_back)
        for cut in sorted({1, 7, n // 2, n - 1}):
            cut_back, output = guarded_output(n, heads)
            offset = cut * heads * D
            attend(q[:offset], cache, pos0, cut, heads, nkv, output[:offset], rows)
            attend(q[offset:], cache, pos0 + cut, n - cut, heads, nkv, output[offset:], rows)
            torch.cuda.synchronize(dev)
            check(evidence, f'{name}: chunk cut {cut} writes finite values to every output row', torch.isfinite(output).all())
            check(evidence, f'{name}: chunk cut {cut} is bitwise identical', torch.equal(auto.view(torch.int32), output.view(torch.int32)))
            all_output_guards &= output_guards(cut_back)
        unchanged = np.array_equal(cache.cpu().numpy(), raw.reshape(-1))
        check(evidence, name + ': attention leaves cache and all external guards unchanged',
              unchanged and bool((backing[:2].cpu().numpy() == 0xa5).all())
              and bool((backing[-2:].cpu().numpy() == 0xa5).all()) and all_output_guards)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--json', help='write checks, object provenance and failure status')
    ap.add_argument('--cpu-fixtures', action='store_true', help='CPU reference/selection checks only; no torch import')
    ap.add_argument('--allow-display', action='store_true', help='explicit later display-card repeat after non-display validation')
    args = ap.parse_args()
    if not args.cpu_fixtures and not args.json:
        ap.error('--json is required for the GPU gate')
    evidence = dict(kind='Q8 dense prefill CPU fixtures' if args.cpu_fixtures else 'Q8 dense prefill op gate',
                    checks=[], failures=[], gpu_executed=False, completed=False, device_verified=False,
                    acceptance=dict(max_relative=MAX_REL, mean_relative=MEAN_REL,
                                    source='tests/prefill/test_q35_prefill_attn.py'))
    try:
        if args.cpu_fixtures:
            cpu_fixtures(evidence)
        else:
            gpu_gate(evidence, args.allow_display)
        evidence['completed'] = True
        evidence['device_verified'] = evidence['gpu_executed'] and not evidence['failures']
    except BaseException as exc:
        evidence['failures'].append(f'{type(exc).__name__}: {exc}')
        print(f'FAIL {type(exc).__name__}: {exc}', flush=True)
    finally:
        if args.json:
            path = Path(args.json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n')
        suffix = ' (CPU fixtures only; device validation pending)' if args.cpu_fixtures else ''
        print('RESULT ' + ('FAIL' if evidence['failures'] or not evidence['completed'] else 'PASS') + suffix, flush=True)
    return 1 if evidence['failures'] or not evidence['completed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
