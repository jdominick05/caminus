"""CPU test of the per-model sampling defaults caminus serve gives a request (caminus.models.SAMPLING,
caminus.serve.sampler.model_defaults): the catalog's recommendation first, then the GGUF's general.sampling.* where the
catalog gives nothing, then SamplingParams' built-in defaults; a request's own fields still win. No GPU, no model file: the
catalog files are sparse files of the catalog's name and size (caminus.models.model_of finds a model by both, never
opening the file). Also the chat template a catalog model is served with (caminus.models.CHAT_TEMPLATES: Qwen3-30B-A3B keeps
the official GGUF's template) and the 30B file's own local directory.

  python -I tests/test_model_sampling.py --scratch DIR      RESULT PASS

The served surface (/props, /v1/models, a request's effective params in /slots) is tests/serve/test_model_sampling.py."""
import argparse
import hashlib
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'python'))

from caminus import models                                    # noqa: E402
from caminus.serve.sampler import Sampler, SamplingParams, greedy_defaults, model_defaults   # noqa: E402

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


def _gs(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


def tokenizer_gguf(path, template, size=None, ids=None):
    """a GGUF v3 header with no tensors and only the tokenizer keys the serve vocabulary reads, its chat template `template`:
    Qwen3's 151,646 ids with the 256 byte-level tokens at 0..255 (no merges: text encodes byte by byte), unused fillers, and
    '<|endoftext|>' / '<|im_start|>' / '<|im_end|>' at 151643..151645 as in the real vocabulary. ids: the
    tokenizer.ggml.*_token_id keys to write ({'bos': 151643, ...}). size: grown to that many bytes, sparse (the catalog's
    size, so model_of takes it for the catalog file)"""
    from caminus.bpe import byte_alphabet
    b2c = byte_alphabet()
    toks = [b2c[i] for i in range(256)] + [f'<filler{i}>' for i in range(256, 151643)] + ['<|endoftext|>', '<|im_start|>', '<|im_end|>']
    types = [1] * 256 + [5] * (151643 - 256) + [3, 3, 3]
    s8 = lambda v: struct.pack('<I', 8) + _gs(v)
    kv = [('general.architecture', s8('qwen3moe')), ('tokenizer.ggml.model', s8('gpt2')), ('tokenizer.ggml.pre', s8('qwen2')),
          ('tokenizer.ggml.tokens', struct.pack('<IIQ', 9, 8, len(toks)) + b''.join(_gs(t) for t in toks)),
          ('tokenizer.ggml.token_type', struct.pack('<IIQ', 9, 5, len(types)) + struct.pack(f'<{len(types)}i', *types)),
          ('tokenizer.ggml.merges', struct.pack('<IIQ', 9, 8, 0)),
          ('tokenizer.ggml.add_bos_token', struct.pack('<IB', 7, 0)),
          ('tokenizer.chat_template', s8(template))]
    kv += [(f'tokenizer.ggml.{k}_token_id', struct.pack('<II', 4, v)) for k, v in (ids or {}).items()]
    out = b'GGUF' + struct.pack('<IQQ', 3, 0, len(kv)) + b''.join(_gs(k) + v for k, v in kv)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(out)
        if size: f.truncate(size)
    return path


# the special-id keys of the two real headers (2026-11-18): Qwen's official Q4_K_M and unsloth's (no bos key, another pad)
OFFICIAL_IDS = dict(bos=151643, eos=151645, padding=151643)
UNSLOTH_IDS = dict(eos=151645, padding=151654)


def template_checks(S):
    """what of the official file's tokenizer header a catalog model is served with (models.CHAT_TEMPLATES, SPECIAL_IDS,
    applied by serve.vocab.load_gguf_vocab), the official file kept by path (models.PREVIOUS_FILES) and the 30B's own local
    directory"""
    from caminus.serve.template import ChatTemplate, PromptBuilder
    from caminus.serve.tokenizer import QwenTokenizer
    from caminus.serve.vocab import load_gguf_vocab
    it = models.BY_KEY['qwen3-30b-a3b']
    shipped = models.chat_template_of(catalog_file(S, 'qwen3-30b-a3b'))
    head, sep, body = (shipped or '').partition('-#}\n')
    # the shipped file is the official GGUF's template (Qwen/Qwen3-30B-A3B-GGUF e4d4baf) below its comment header: the text's
    # sha256 as the header records it, and the header renders to nothing
    check(bool(sep) and head.startswith('{#-') and hashlib.sha256(body.encode()).hexdigest() ==
          '57f1fd00f0013a2be96aa79b857391f27e23df5b5f847072b524c897e24d0361',
          'qwen3-30b-a3b: the shipped chat template is the official GGUF\'s text (sha256 57f1fd00...) below a comment header')
    msgs = [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'q1'},
            {'role': 'assistant', 'content': 'a1', 'reasoning_content': 'r1'}, {'role': 'user', 'content': 'q2'}]
    for kw in ({}, {'enable_thinking': False}):
        check(ChatTemplate(shipped).render(msgs, **kw) == ChatTemplate(body).render(msgs, **kw),
              f'qwen3-30b-a3b: the comment header renders to nothing {kw or ""}')
    base = os.path.basename(it.path)
    # the fixtures stay small (a sparse file of the catalog's 18.6 GB would be memory-mapped whole by the reader): model_of
    # takes them for the catalog model through a PREVIOUS_FILES entry of their own size for the length of the check
    # the official file's header outside the catalog (another name): its own template and ids, the reference for the prompts
    ref = tokenizer_gguf(os.path.join(S, 'tpl', 'official', 'official.gguf'), body, ids=OFFICIAL_IDS)
    # unsloth's header (its own template, no bos key, pad 151654) as the catalog's file: served with the official ones
    uns = tokenizer_gguf(os.path.join(S, 'tpl', 'catalog', base), 'OWN {{ messages }}', ids=UNSLOTH_IDS)
    # the official header under the catalog's file name (the official file kept by path): the catalog model's, all of it
    prev = tokenizer_gguf(os.path.join(S, 'tpl', 'previous', base), body, ids=OFFICIAL_IDS)
    saved = models.PREVIOUS_FILES
    models.PREVIOUS_FILES = dict(saved, **{'qwen3-30b-a3b': saved['qwen3-30b-a3b'] + tuple(
        (base, os.path.getsize(f)) for f in (uns, prev))})
    try:
        vr, vu = load_gguf_vocab(ref), load_gguf_vocab(uns)
        check(models.model_of(ref) is None and (vr.bos_id, vr.eos_id, vr.pad_id) == (151643, 151645, 151643) and vr.chat_template == body,
              'the official header outside the catalog: its own template and ids (bos 151643, eos 151645, pad 151643)')
        check(vu.chat_template == shipped and (vu.bos_id, vu.eos_id, vu.pad_id) == (151643, 151645, 151643) and len(vu.tokens) == 151646,
              'qwen3-30b-a3b catalog file with unsloth\'s header: served with the shipped template and the official ids '
              f'({vu.bos_id}, {vu.eos_id}, {vu.pad_id}), the rest of the vocabulary read')
        for label, p in (('the catalog file (unsloth\'s header)', uns), ('the official file kept by path', prev)):
            ok = True
            for kw in (None, {'enable_thinking': False}):
                a = PromptBuilder.from_tokenizer(QwenTokenizer.from_gguf(ref, engine='own')).build(msgs, client_kwargs=kw).ids
                b = PromptBuilder.from_tokenizer(QwenTokenizer.from_gguf(p, engine='own')).build(msgs, client_kwargs=kw).ids
                ok = ok and a == b and a[:2] == [151643, 151644]
            check(ok, f'qwen3-30b-a3b, {label}: the served prompt ids equal the official header\'s, BOS 151643 first, thinking on and off')
    finally:
        models.PREVIOUS_FILES = saved
    # the real PREVIOUS_FILES entry: the official file's name and size (a sparse file, never opened: model_of reads its size)
    kept = os.path.join(S, 'tpl', 'kept', base)
    os.makedirs(os.path.dirname(kept), exist_ok=True)
    with open(kept, 'wb') as f:
        f.truncate(saved['qwen3-30b-a3b'][0][1])
    check(models.model_of(kept) is models.BY_NAME['qwen3-30b-a3b'] and models.sampling_of(kept) == models.SAMPLING['qwen3-30b-a3b']
          and models.chat_template_of(kept) == shipped and models.special_ids_of(kept) == models.SPECIAL_IDS['qwen3-30b-a3b'],
          'the official file kept by path (PREVIOUS_FILES, its name and size): still the catalog model, its sampling, template and ids')
    p = tokenizer_gguf(os.path.join(S, 'tpl', 'other', 'x', base), 'OWN {{ messages }}', ids=UNSLOTH_IDS)
    v = load_gguf_vocab(p)
    check(v.chat_template == 'OWN {{ messages }}' and v.bos_id == -1 and v.pad_id == 151654,
          'a file outside the catalog: its own chat template and ids')
    check(set(models.CHAT_TEMPLATES) | set(models.SPECIAL_IDS) | set(models.PREVIOUS_FILES) <= {m.name for m in models.ALL},
          'CHAT_TEMPLATES, SPECIAL_IDS and PREVIOUS_FILES name only catalog models')
    check(all(n != os.path.basename(BY.path) or z != BY.size for name, fs in models.PREVIOUS_FILES.items()
              for n, z in fs for BY in [models.BY_KEY[models.BY_NAME[name].items[0]]]),
          'PREVIOUS_FILES: never the current file')
    # the unsloth file's own directory: beside the official file's place (same repository name and file name), and no two
    # catalog items share a local path
    rels = [models.local_rel(i) for i in models.ITEMS]
    check(models.local_rel(it) == 'Qwen3-30B-A3B-GGUF-unsloth/Qwen3-30B-A3B-Q4_K_M.gguf' and len(set(rels)) == len(rels),
          f'qwen3-30b-a3b: its own local directory ({models.local_rel(it)}), every catalog path distinct')


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
    p, pn, mo = model_defaults(q3, {})
    check(six(p) == (0.6, 0.95, 20, 0.0, 1.5, 1.0) and mo == 32768,
          f'qwen3-30b-a3b without GGUF keys: 0.6 / 0.95 / 20 / 0 / presence 1.5 / repeat 1.0, max_output 32768 ({six(p)}, {mo})')
    check(p.penalties_on, 'qwen3-30b-a3b: the presence penalty is on by default')
    # its card's non-thinking set (the -nothink id, enable_thinking false)
    check(pn is not None and six(pn) == (0.7, 0.8, 20, 0.0, 1.5, 1.0),
          f'qwen3-30b-a3b without thinking: 0.7 / 0.8 / 20 / 0 / presence 1.5 ({pn and six(pn)})')
    # the catalog wins over the GGUF's keys; a field the catalog leaves None (its repeat penalty) takes the GGUF's
    p, pn, _ = model_defaults(q3, {'general.sampling.temp': 0.9, 'general.sampling.top_k': 40,
                                   'general.sampling.penalty_repeat': 1.1})
    check((p.temperature, p.top_k, pn.temperature, pn.top_k) == (0.6, 20, 0.7, 20),
          f'qwen3-30b-a3b: the catalog wins over the GGUF, both sets ({p.temperature}, {p.top_k}, {pn.temperature}, {pn.top_k})')
    check(abs(p.repeat_penalty - 1.1) < 1e-12 and abs(pn.repeat_penalty - 1.1) < 1e-12,
          f'qwen3-30b-a3b: a field the catalog leaves None takes the GGUF\'s ({p.repeat_penalty}, {pn.repeat_penalty})')

    # the Qwen3.8 line: 1.0 / 0.95 / 20 / 0, penalties off, 393,216 output; the same as their GGUF headers gave before
    # (float32 0.95 = the header's 0.949999988: the sampler compares in float32)
    hdr = {'general.sampling.temp': 1.0, 'general.sampling.top_k': 20, 'general.sampling.top_p': 0.949999988079071}
    before = SamplingParams.from_gguf_meta(hdr)
    for key in ('swift-1.5', 'thinkingcap', 'qwen3.8-27b', 'coder-1'):
        path = catalog_file(S, key)
        p, pn, mo = model_defaults(path, hdr)
        p0, _, _ = model_defaults(path, {})
        check(six(p) == six(p0) == (1.0, 0.95, 20, 0.0, 0.0, 1.0) and mo == 393216,
              f'{key}: 1.0 / 0.95 / 20 / 0, presence 0, repeat 1, max_output 393216 with or without the header ({six(p)}, {mo})')
        check(pn is None, f'{key}: one set (its card has no separate non-thinking set)')
        check([f32(v) for v in six(p)] == [f32(v) for v in six(before)] and not p.penalties_on,
              f'{key}: as float32, the values its GGUF header gave before (no behaviour change)')

    # a file outside the catalog: the GGUF's keys, then the built-in defaults; no max_output
    other = os.path.join(S, 'other', 'Qwen3-30B-A3B-Q4_K_M.gguf')
    os.makedirs(os.path.dirname(other), exist_ok=True)
    with open(other, 'wb') as f:
        f.truncate(4096)                                   # (the catalog's name at another size: not the catalog's file)
    p, pn, mo = model_defaults(other, {'general.sampling.temp': 0.7})
    check(models.sampling_of(other) is None and p.temperature == 0.7 and p.presence_penalty == 0.0 and mo is None
          and pn is None, 'outside the catalog: the GGUF\'s temperature, the built-in penalties, one set, no max_output')
    p, pn, mo = model_defaults(other, {})
    check(six(p) == six(SamplingParams()) and mo is None, 'outside the catalog without keys: SamplingParams\' defaults')
    check(model_defaults(None, {})[1:] == (None, None), 'no path: the GGUF / built-in layers only')

    # a request's own fields win over every default (temperature 0 included: greedy), the rest stay the model's
    base, _, _ = model_defaults(q3, {})
    r = base.with_request({'temperature': 0.2})
    check((r.temperature, r.presence_penalty, r.top_p) == (0.2, 1.5, 0.95), f'request: temperature 0.2 overrides, presence stays 1.5 ({six(r)})')
    r = base.with_request({'temperature': 0, 'presence_penalty': 0.0, 'top_k': 5, 'min_p': None})
    check(six(r) == (0.0, 0.95, 5, 0.0, 0.0, 1.0) and r.greedy and not r.penalties_on,
          f'request: temperature 0 / presence 0 / top_k 5 override, None keeps the default ({six(r)})')

    # a greedy request takes no default penalty (the server's rule after with_request: sampler.greedy_defaults); a sampled
    # one keeps 1.5; a penalty the greedy request sends itself is kept
    req = {'temperature': 0}
    g = greedy_defaults(base.with_request(req), req)
    check((g.temperature, g.presence_penalty, g.repeat_penalty, g.frequency_penalty) == (0.0, 0.0, 1.0, 0.0) and not g.penalties_on,
          f'greedy: qwen3-30b-a3b at temperature 0 takes presence 0 ({six(g)})')
    old = SamplingParams().with_request(req)          # (what the 30B's greedy requests ran at before the catalog: no GGUF keys)
    check(g == old, f'greedy: qwen3-30b-a3b\'s greedy params are exactly the ones before the catalog ({g} vs {old})')
    req = {'temperature': 0.7}
    check(base.with_request(req).presence_penalty == 1.5, 'sampled: qwen3-30b-a3b at temperature 0.7 keeps presence 1.5')
    req = {'temperature': 0, 'presence_penalty': 0.5, 'repeat_penalty': True}
    g = greedy_defaults(base.with_request(req), req)
    check((g.presence_penalty, g.repeat_penalty) == (0.5, 1.0), f'greedy: an explicit presence_penalty is kept ({six(g)})')
    req = {'temperature': 0, 'repeat_penalty': 1.1}
    g = greedy_defaults(base.with_request({'temperature': 0, 'repeat_penalty': 1.1}), req)
    check((g.presence_penalty, g.repeat_penalty) == (0.0, 1.1), f'greedy: an explicit repeat_penalty is kept, presence off ({six(g)})')

    # the same greedy token under the old params (before the catalog: SamplingParams' defaults, the 30B's GGUF has no keys)
    # and the new ones (the catalog's thinking or non-thinking set, the greedy rule), on random logits rows with random
    # histories: the server's greedy draw is the argmax of the penalized row, which top_p (0.8 vs 0.95) never changes.
    # Rows with ties at the maximum included (values rounded): the first maximum wins under both
    _, nothink, _ = model_defaults(q3, {})
    rng = np.random.default_rng(20261117)
    V = 4096
    rows = [rng.standard_normal(V).astype(np.float32) * 4 for _ in range(48)]
    rows += [np.round(rng.standard_normal(V) * 2).astype(np.float32) for _ in range(16)]   # (ties at the max)
    for label, model_base in (('thinking', base), ('-nothink (top_p 0.8 vs 0.95)', nothink)):
        for req in ({'temperature': 0}, {'temperature': 0, 'presence_penalty': 0.5}, {'temperature': 0, 'repeat_penalty': 1.3}):
            old = Sampler(SamplingParams().with_request(req))
            new = Sampler(greedy_defaults(model_base.with_request(req), req))
            same = 0
            for row in rows:
                hist = [int(x) for x in rng.integers(0, V, 80)] + [int(np.argmax(row))] * 3   # (the max in the penalty window)
                same += old.sample(row, hist) == new.sample(row, hist)
            check(same == len(rows) and new.params.greedy,
                  f'greedy token {label}, request {req}: old and new params pick the same token on {same} / {len(rows)} rows')

    template_checks(S)
    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
