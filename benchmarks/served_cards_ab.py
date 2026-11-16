"""A served A/B of Qwen3-30B-A3B on one card against two cards: `caminus serve` (tools/caminus_serve.py, through
tools/serve_smoke1.Server1) once per arm and round, the same greedy /v1/completions requests in each, by prompt depth.
The source of the docs/dev/benchmarks.md row for the model's two-card serving.

  tools/guard.sh LOG 1800 python -I benchmarks/served_cards_ab.py --model GGUF --scratch DIR \\
      [--depths 8,4096,16384,32768,40768] [--n 128] [--rounds 3] [--max-ctx 40960]
  python -I benchmarks/served_cards_ab.py --model GGUF --scratch DIR --prompts-only   # (CPU: build and print the prompts)

Arms: one card (HIP_VISIBLE_DEVICES=0, --cards 1) and two cards (HIP_VISIBLE_DEVICES=0,1, --cards 2), both --draft off
(this model has no drafter), own code (--block-imports gguf,tokenizers: the server refuses those imports). Round r starts
one server per arm, one card first, each a fresh process running alone (the cards' other work stops first: a GPU window).

Prompts: token ids built here from public text only, this checkout's own source (AGENTS.md, then python/caminus/**/*.py in
sorted order, the vendored code left out, each file under a `==> path <==` line, as an agent reading the code would see
it), encoded with the model's own tokenizer the way the server builds it (caminus.serve.tokenizer.QwenTokenizer with the
architecture's BPE engine, plain text: a literal special-token string in the source stays ordinary text). Each depth is
exactly that many prompt tokens, the vocabulary's BOS first (the server would put it there), then a header line whose
first word starts with a different letter per depth, then the source: no depth is a prefix of another, the prompts share
only the BOS, and the builder refuses anything else. Every depth + --n must fit --max-ctx (the file's 40,960 by default).
Requests go in ascending depth, so each is longer than the line before it and runs in the main slot (a shorter unrelated
prompt can be routed to the side slot), each after a 3-token warm-up request (its own first word, 4 new tokens: the first
passes' costs) that is shorter than the first depth.

Not from the prompt cache: each arm server gets a fresh scratch directory per round (its own --cache-dir and --flight-dir),
deleted after that server stops (the disk tier writes GBs); within a server the depths share no prefix. Every measured
request must report cached tokens 0 and usage.prompt_tokens == its depth (RESULT FAIL otherwise).

Measured per depth, from the server's own timings: decode tok/s (predicted_per_second over the generated tokens),
prefill tok/s raw (prompt_per_second: prompt tokens / prompt_ms) and net (prompt tokens / (prompt_ms - the request's
restore + park ms from its flight line): before overwriting 4,096 or more positions of the live line the server parks
that line to the disk tier inside the prompt phase, so from the second batched depth on the raw number includes a disk
write of the previous depth). Reported: per depth the median and range over --rounds of each arm, the ratio two / one of
the medians, the generated token counts, the slot each request ran in, and how many replies' text equals the first one-card
round's (the server returns text, not ids; the two-card engine is not bitwise by design, so this is information, not a
gate). RESULT DONE (a benchmark: no speed gate) unless a server fails (not ready, a request not HTTP 200, a non-zero
exit) or a cache hit is seen: RESULT FAIL. All numbers also go to DIR/served_cards_ab.json."""
import argparse
import hashlib
import json
import os
import shutil
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import serve_smoke as S           # noqa: E402  (request helpers, safe_dir, flight_lines, LIVE / stop_servers)
from serve_smoke1 import Server1  # noqa: E402

if not S.INSTALLED: sys.path.insert(0, os.path.join(ROOT, 'python'))   # (the tokenizer: the checkout's, as the server's)

MAX_CTX = 40960                  # Qwen3-30B-A3B's context_length: the largest max_ctx served without the long-context mode
DEPTHS = '8,4096,16384,32768,40768'
# one header word per depth: each starts with a different letter, so the prompts' first tokens after the BOS differ
WORDS = ('Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo', 'Foxtrot', 'Golf', 'Hotel', 'India', 'Juliett', 'Kilo', 'Lima')
WARM_WORD, WARM_N = 'Zulu', 4    # the warm-up: BOS + 2 tokens of its own word, 4 new tokens
ARMS = (('one', '0', 1), ('two', '0,1', 2))   # (name, HIP_VISIBLE_DEVICES, --cards)


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n


def parse_depths(text):
    """--depths: positive integers, strictly ascending (the requests run in this order: each longer than the line before),
    at most one per header word"""
    try:
        ds = [int(x) for x in str(text).split(',') if x.strip()]
    except ValueError:
        raise ValueError(f'--depths {text!r}: comma-separated integers') from None
    if not ds: raise ValueError('--depths: none given')
    if any(d < 1 for d in ds): raise ValueError(f'--depths {ds}: every depth is at least 1 token')
    if any(b <= a for a, b in zip(ds, ds[1:])): raise ValueError(f'--depths {ds}: strictly ascending, no repeats')
    if len(ds) > len(WORDS): raise ValueError(f'--depths: at most {len(WORDS)} depths')
    return ds


def check_fit(depths, n, max_ctx):
    """refuse a depth whose prompt + n generated tokens does not fit max_ctx (the server would cut the reply short:
    caminus.serve.server's _max_new keeps prompt + max_tokens <= max_ctx on a one-token-per-pass engine)"""
    if n < 1: raise ValueError(f'--n {n}: at least 1 token')
    bad = [d for d in depths if d + n > max_ctx]
    if bad: raise ValueError(f'depth(s) {bad} + --n {n} > --max-ctx {max_ctx}: the prompt and its reply must fit')


def source_files(root=ROOT):
    """the public text, in a fixed order: AGENTS.md, then every .py under python/caminus sorted by path (no vendored code)"""
    out = ['AGENTS.md']
    base = os.path.join(root, 'python', 'caminus')
    found = []
    for d, subs, files in os.walk(base):
        subs[:] = sorted(s for s in subs if s not in ('_vendor', '__pycache__'))
        found += [os.path.relpath(os.path.join(d, f), root).replace(os.sep, '/') for f in files if f.endswith('.py')]
    return out + sorted(found)


def source_chunks(root=ROOT):
    for rel in source_files(root):
        with open(os.path.join(root, rel), encoding='utf-8') as f:
            yield f'==> {rel} <==\n{f.read()}\n'


def build_prompts(depths, encode, bos, root=ROOT):
    """{depth: token ids}: exactly depth ids each, bos first (when the vocabulary has one), a header line naming the depth
    with its own first word, then the source read in order. encode(text) -> ids is plain-text BPE (no special tokens).
    Refuses (ValueError) a source too short or two prompts sharing more than the BOS."""
    need = max(depths)
    body = []
    for chunk in source_chunks(root):
        body += encode(chunk)
        if len(body) >= need: break
    if len(body) < need: raise ValueError(f'the source text gives {len(body)} tokens, fewer than the deepest prompt ({need})')
    head = [] if bos is None else [int(bos)]
    prompts = {}
    for word, d in zip(WORDS, depths):
        ids = head + encode(f'{word} ({d} tokens): the Caminus source tree, read file by file.\n') + body
        prompts[d] = [int(x) for x in ids[:d]]
    check_distinct(prompts, bos)
    return prompts


def check_distinct(prompts, bos):
    """no prompt is a prefix of another, and any two share at most the BOS (so no depth can be served from another's
    prompt-cache checkpoint)"""
    allowed = 0 if bos is None else 1
    ks = sorted(prompts)
    for i, a in enumerate(ks):
        if len(prompts[a]) != a: raise ValueError(f'depth {a}: {len(prompts[a])} tokens')
        for b in ks[i + 1:]:
            m = lcp(prompts[a], prompts[b])
            if m > allowed: raise ValueError(f'depths {a} and {b} share their first {m} tokens (at most {allowed})')


def warmup_ids(encode, bos, first_depth):
    """the warm-up prompt: BOS + 2 tokens of WARM_WORD, or None when its line (prompt + WARM_N - 1 positions) would not be
    shorter than the first depth (the first measured request would then be routed to the side slot)"""
    ids = ([] if bos is None else [int(bos)]) + [int(x) for x in encode(f'{WARM_WORD} warm-up')[:2]]
    return ids if len(ids) + WARM_N - 1 <= first_depth else None


def tokenizer(model):
    """the model's tokenizer as caminus serve builds it (serve/cli.py: QwenTokenizer.from_gguf with the architecture's engine)"""
    from caminus import arch
    from caminus.serve.tokenizer import QwenTokenizer
    return QwenTokenizer.from_gguf(model, engine=arch.of(model).tokenizer)


def ids_sha(ids):
    return hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------------- the GPU part
class Failed(Exception):
    pass


class Arm(argparse.Namespace):
    pass


def complete(port, ids, n):
    r = S.request(port, 'POST', '/v1/completions', dict(model='m', prompt=list(ids), max_tokens=n, temperature=0, stream=False))
    j = r.get('json')
    if r.get('status') != 200 or not isinstance(j, dict) or not j.get('choices'):
        raise Failed(f'/v1/completions: HTTP {r.get("status")} {str(j or r.get("text"))[:200]}')
    return j


def run_server(A, rnd, name, card, cards, prompts, warm):
    """one arm's server for one round: the warm-up, then each depth ascending; returns its rows. The scratch directory
    (cache + flight) is deleted after the server stops"""
    scratch = os.path.join(A.scratch, f'r{rnd}-{name}')
    if os.path.exists(scratch): raise Failed(f'{scratch} exists: each server starts on a fresh directory')
    os.makedirs(scratch, mode=0o700)
    a = Arm(model=A.model, max_ctx=A.max_ctx, scratch=scratch, alias=None, cards=cards, draft='off',
            block_imports='gguf,tokenizers', card=card)
    rows, rc = [], None
    try:
        s = Server1(f'serve[{name} r{rnd}]', a)
        try:
            if not s.wait_ready(): raise Failed(f'the {name}-card server did not come up (round {rnd})')
            if warm is not None: complete(s.port, warm, WARM_N)
            for d, ids in prompts.items():
                j = complete(s.port, ids, A.n)
                tm = j.get('timings') or {}; us = j.get('usage') or {}
                st = S.get(s.port, '/caminus/status')[1]
                last = ((((st or {}).get('cache') or {}).get('route') or {}).get('last') or {}) if isinstance(st, dict) else {}
                rows.append(dict(depth=d, text=j['choices'][0].get('text') or '', prompt_tokens=us.get('prompt_tokens'),
                                 cache_n=tm.get('cache_n'), prompt_n=tm.get('prompt_n'), prompt_ms=tm.get('prompt_ms'),
                                 prompt_tps=tm.get('prompt_per_second'), predicted_n=tm.get('predicted_n'),
                                 decode_tps=tm.get('predicted_per_second'), route=last.get('route'), reason=last.get('reason')))
                print(f'round {rnd} {name} {d:6d}: decode {tm.get("predicted_per_second")} tok/s ({tm.get("predicted_n")} tokens), '
                      f'prefill {tm.get("prompt_per_second")} tok/s ({tm.get("prompt_n")} new, {tm.get("cache_n")} cached), '
                      f'slot {last.get("route")}', flush=True)
        finally:
            rc = s.stop()
        # the flight lines (written as each request ends): the restore + park ms inside each request's prompt phase
        fl = [x for x in S.flight_lines(os.path.join(scratch, 'flight')) if x.get('route') == 'completions']
        fl = fl[-len(rows):] if rows else []
        for row, x in zip(rows, fl):
            row['restore_ms'] = (x.get('cache') or {}).get('restore_ms')
        if len(fl) != len(rows): print(f'NOTE round {rnd} {name}: {len(fl)} flight lines for {len(rows)} requests', flush=True)
        if rc != 0: raise Failed(f'the {name}-card server exited {rc} (round {rnd})')
        return rows
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def net_tps(row):
    ms, rs, n = row.get('prompt_ms'), row.get('restore_ms'), row.get('prompt_n')
    if not ms or not n or rs is None or ms - rs <= 0: return None
    return n / ((ms - rs) / 1e3)


def stats(v):
    v = [x for x in v if x]
    return (statistics.median(v), min(v), max(v)) if v else (None, None, None)


def fmt(m):
    med, lo, hi = m
    return '-' if med is None else f'{med:7.1f} [{lo:.1f}-{hi:.1f}]'


def report(A, prompts, res):
    names = [a[0] for a in ARMS]
    print(f'\nserved Qwen3-30B-A3B, one card (0) vs two cards (0,1), --draft off, {A.n} greedy tokens, median [range] of '
          f'{A.rounds} round(s):')
    for key, title in (('decode_tps', 'decode tok/s'), ('prompt_tps', 'prefill tok/s, raw (server timings)'),
                       ('net', 'prefill tok/s, net of the restore / park ms')):
        print(f'\n{title}\n{"depth":>7s} {"one card":>24s} {"two cards":>24s} {"two/one":>8s}')
        for d in prompts:
            vals = {nm: stats([net_tps(r) if key == 'net' else r.get(key) for r in res[nm][d]]) for nm in names}
            a, b = vals['one'][0], vals['two'][0]
            print(f'{d:7d} {fmt(vals["one"]):>24s} {fmt(vals["two"]):>24s} {(b / a if a and b else 0):8.2f}')
    print('\ngenerated tokens per round (one | two), slot, restore + park ms, reply text equal to the first one-card round:')
    for d in prompts:
        ref = res['one'][d][0]['text'] if res['one'][d] else None
        eq = {nm: f'{sum(r["text"] == ref for r in res[nm][d])} / {len(res[nm][d])}' for nm in names}
        print(f'{d:7d} n {[r.get("predicted_n") for r in res["one"][d]]} | {[r.get("predicted_n") for r in res["two"][d]]}; '
              f'slot {sorted({str(r.get("route")) for nm in names for r in res[nm][d]})}; restore ms '
              f'{[r.get("restore_ms") for r in res["one"][d]]} | {[r.get("restore_ms") for r in res["two"][d]]}; '
              f'equal replies one {eq["one"]}, two {eq["two"]} (information: the two-card engine is not bitwise by design)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--model', required=True); ap.add_argument('--scratch', required=True)
    ap.add_argument('--depths', default=DEPTHS); ap.add_argument('--n', type=int, default=128)
    ap.add_argument('--rounds', type=int, default=3); ap.add_argument('--max-ctx', type=int, default=MAX_CTX)
    ap.add_argument('--prompts-only', action='store_true', help='build the prompts, print their counts and digests, exit (CPU)')
    A = ap.parse_args(argv)
    try:
        depths = parse_depths(A.depths)
        check_fit(depths, A.n, A.max_ctx)
        if A.rounds < 1: raise ValueError(f'--rounds {A.rounds}: at least 1')
    except ValueError as e:
        print(f'served_cards_ab: {e}', file=sys.stderr); return 2
    A.scratch = S.safe_dir(A.scratch); os.makedirs(A.scratch, mode=0o700, exist_ok=True)
    t0 = time.perf_counter()
    tok = tokenizer(A.model)
    prompts = build_prompts(depths, tok.encode_plain, tok.bos_id)
    warm = warmup_ids(tok.encode_plain, tok.bos_id, depths[0])
    print(f'prompts built in {time.perf_counter() - t0:.1f} s (BOS {tok.bos_id}; warm-up {warm}):', flush=True)
    for d, ids in prompts.items():
        print(f'  depth {d:6d}: {len(ids)} tokens, sha {ids_sha(ids)}, starts {ids[:4]}', flush=True)
    if A.prompts_only: return 0
    res = {nm: {d: [] for d in prompts} for nm, _, _ in ARMS}
    fails = []
    try:
        for rnd in range(A.rounds):
            for nm, card, cards in ARMS:
                for row in run_server(A, rnd, nm, card, cards, prompts, warm):
                    res[nm][row['depth']].append(row)
                    if row['cache_n'] != 0 or row['prompt_tokens'] != row['depth'] or row['prompt_n'] != row['depth']:
                        fails.append(f'round {rnd} {nm} depth {row["depth"]}: cached {row["cache_n"]}, prompt tokens '
                                     f'{row["prompt_tokens"]}, new {row["prompt_n"]} (want 0 cached, {row["depth"]} new)')
    except Exception as e:                       # noqa: BLE001 - a failed server or request ends the run as RESULT FAIL
        fails.append(str(e) if isinstance(e, Failed) else f'{type(e).__name__}: {e}')
    finally:
        S.stop_servers()
    with open(os.path.join(A.scratch, 'served_cards_ab.json'), 'w') as f:   # (first: a report error cannot lose the run)
        json.dump(dict(model=os.path.basename(A.model), depths=depths, n=A.n, rounds=A.rounds, max_ctx=A.max_ctx,
                       prompt_sha={d: ids_sha(ids) for d, ids in prompts.items()}, results=res, fails=fails), f)
    report(A, prompts, res)
    for f in fails: print(f'FAIL {f}', flush=True)
    print('RESULT ' + ('FAIL' if fails else 'DONE'), flush=True)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
