"""CPU test of the per-model sampling defaults caminus serve gives a request (caminus.models.SAMPLING,
caminus.serve.sampler.model_defaults): the catalog's recommendation first, then the GGUF's general.sampling.* where the
catalog gives nothing, then SamplingParams' built-in defaults; a request's own fields still win. No GPU, no model file: the
catalog files are sparse files of the catalog's name and size (caminus.models.model_of finds a model by both, never
opening the file).

  python -I tests/test_model_sampling.py --scratch DIR      RESULT PASS

The served surface (/props, /v1/models, a request's effective params in /slots) is tests/serve/test_model_sampling.py."""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'python'))

from caminus import models                                    # noqa: E402
from caminus.serve.sampler import SamplingParams, model_defaults   # noqa: E402

FAILS = []


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok:
        FAILS.append(what)


def f32(x):
    return float(np.float32(x))


def catalog_file(scratch, key):
    """a sparse file with the catalog item's name and size (what model_of identifies)"""
    it = models.BY_KEY[key]
    p = os.path.join(scratch, 'catalog', key, os.path.basename(it.path))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'wb') as f:
        f.truncate(it.size)
    return p


def six(p):
    return (p.temperature, p.top_p, p.top_k, p.min_p, p.presence_penalty, p.repeat_penalty)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', required=True)
    S = ap.parse_args().scratch
    os.makedirs(S, exist_ok=True)

    # every served model carries a recommendation, every field set or None (the card states nothing)
    missing = [m.name for m in models.ALL if m.name not in models.PAUSED and m.name not in models.SAMPLING]
    check(not missing, f'catalog: every served model has recommended sampling ({missing or "all"})')
    check(all(isinstance(r.max_output, int) and r.max_output > 0 for r in models.SAMPLING.values()),
          'catalog: every recommendation has a max_output')
    check(set(models.SAMPLING) <= {m.name for m in models.ALL}, 'catalog: SAMPLING names only catalog models')

    # Qwen3-30B-A3B: its GGUF has no general.sampling.* keys; the card's thinking-mode set and presence 1.5
    q3 = catalog_file(S, 'qwen3-30b-a3b')
    p, mo = model_defaults(q3, {})
    check(six(p) == (0.6, 0.95, 20, 0.0, 1.5, 1.0) and mo == 32768,
          f'qwen3-30b-a3b without GGUF keys: 0.6 / 0.95 / 20 / 0 / presence 1.5 / repeat 1.0, max_output 32768 ({six(p)}, {mo})')
    check(p.penalties_on, 'qwen3-30b-a3b: the presence penalty is on by default')
    # the catalog wins over the GGUF's keys; a field the catalog leaves None (its repeat penalty) takes the GGUF's
    p, _ = model_defaults(q3, {'general.sampling.temp': 0.9, 'general.sampling.top_k': 40,
                               'general.sampling.penalty_repeat': 1.1})
    check((p.temperature, p.top_k) == (0.6, 20), f'qwen3-30b-a3b: the catalog wins over the GGUF ({p.temperature}, {p.top_k})')
    check(abs(p.repeat_penalty - 1.1) < 1e-12, f'qwen3-30b-a3b: a field the catalog leaves None takes the GGUF\'s ({p.repeat_penalty})')

    # the Qwen3.8 line: 1.0 / 0.95 / 20 / 0, penalties off, 393,216 output; the same as their GGUF headers gave before
    # (float32 0.95 = the header's 0.949999988: the sampler compares in float32)
    hdr = {'general.sampling.temp': 1.0, 'general.sampling.top_k': 20, 'general.sampling.top_p': 0.949999988079071}
    before = SamplingParams.from_gguf_meta(hdr)
    for key in ('swift-1.5', 'thinkingcap', 'qwen3.8-27b', 'coder-1'):
        path = catalog_file(S, key)
        p, mo = model_defaults(path, hdr)
        p0, _ = model_defaults(path, {})
        check(six(p) == six(p0) == (1.0, 0.95, 20, 0.0, 0.0, 1.0) and mo == 393216,
              f'{key}: 1.0 / 0.95 / 20 / 0, presence 0, repeat 1, max_output 393216 with or without the header ({six(p)}, {mo})')
        check([f32(v) for v in six(p)] == [f32(v) for v in six(before)] and not p.penalties_on,
              f'{key}: as float32, the values its GGUF header gave before (no behaviour change)')

    # a file outside the catalog: the GGUF's keys, then the built-in defaults; no max_output
    other = os.path.join(S, 'other', 'Qwen3-30B-A3B-Q4_K_M.gguf')
    os.makedirs(os.path.dirname(other), exist_ok=True)
    with open(other, 'wb') as f:
        f.truncate(4096)                                   # (the catalog's name at another size: not the catalog's file)
    p, mo = model_defaults(other, {'general.sampling.temp': 0.7})
    check(models.sampling_of(other) is None and p.temperature == 0.7 and p.presence_penalty == 0.0 and mo is None,
          'outside the catalog: the GGUF\'s temperature, the built-in penalties, no max_output')
    p, mo = model_defaults(other, {})
    check(six(p) == six(SamplingParams()) and mo is None, 'outside the catalog without keys: SamplingParams\' defaults')
    check(model_defaults(None, {})[1] is None, 'no path: the GGUF / built-in layers only')

    # a request's own fields win over every default (temperature 0 included: greedy), the rest stay the model's
    base, _ = model_defaults(q3, {})
    r = base.with_request({'temperature': 0.2})
    check((r.temperature, r.presence_penalty, r.top_p) == (0.2, 1.5, 0.95), f'request: temperature 0.2 overrides, presence stays 1.5 ({six(r)})')
    r = base.with_request({'temperature': 0, 'presence_penalty': 0.0, 'top_k': 5, 'min_p': None})
    check(six(r) == (0.0, 0.95, 5, 0.0, 0.0, 1.0) and r.greedy and not r.penalties_on,
          f'request: temperature 0 / presence 0 / top_k 5 override, None keeps the default ({six(r)})')

    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
