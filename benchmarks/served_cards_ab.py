"""A served A/B of Qwen3-30B-A3B on one card against two cards: `caminus serve` (tools/caminus_serve.py, through
tools/serve_smoke1.Server1) once per arm and round, the same greedy /v1/completions requests in each, by prompt depth.
The source of the docs/dev/benchmarks.md row for the model's two-card serving.

  tools/guard.sh LOG 1800 python -I benchmarks/served_cards_ab.py --model GGUF --scratch DIR \\
      [--depths 8,4096,16384,32768,40768] [--n 128] [--rounds 3] [--max-ctx 40960]
  tools/guard.sh LOG 900 python -I benchmarks/served_cards_ab.py --model GGUF --scratch DIR --probe --rounds 1 ...
  python -I benchmarks/served_cards_ab.py --model GGUF --scratch DIR --prompts-only   # (CPU: build and print the prompts)

Arms: one card (HIP_VISIBLE_DEVICES=0, --cards 1) and two cards (HIP_VISIBLE_DEVICES=0,1, --cards 2), both --draft off
(this model has no drafter), own code (--block-imports gguf,tokenizers: the server refuses those imports), and both
--no-disk-cache: no disk tier, so no park or spill writes a file inside a timed prompt phase (with one, the server parks
the live line before a request overwrites 4,096 or more of its positions, and a full VRAM tier spills a checkpoint;
both happen inside prompt_ms). Each server's own log must say `disk off`, and its card line (`caminus serve: cards ...`)
must name one card for the one-card arm and two for the other, the one-card arm's card being the two-card arm's card 0
(placement puts the bigger card first), else RESULT FAIL. Round r starts one server per arm, one card first, each a
fresh process running alone (the cards' other work stops first: a GPU window).

Prompts: token ids built here from public text only, this checkout's own source (AGENTS.md, then python/caminus/**/*.py in
sorted order, the vendored code left out, each file under a `==> path <==` line, as an agent reading the code would see
it; a file git does not track there is refused, so the prompts are the commit's text), encoded with the model's own
tokenizer the way the server builds it (caminus.serve.tokenizer.QwenTokenizer with the architecture's BPE engine, plain
text: a literal special-token string in the source stays ordinary text). Each depth is exactly that many prompt tokens,
the vocabulary's BOS first (the server would put it there), then a header line whose first word starts with a different
letter per depth, then the source: no depth is a prefix of another, the prompts share only the BOS, and the builder
refuses anything else. Every depth + --n must fit --max-ctx (the file's 40,960 by default), and each depth must be at
least the previous depth + --n: requests go in ascending depth, so each is at least as long as the line before it and
runs in the main slot (a shorter unrelated prompt is routed to the side slot), each after a 3-token warm-up request (its
own first word, 4 new tokens: the first passes' costs) that is shorter than the first depth. The slot each request ran in
is read from /caminus/status; one not in the main slot is a FAIL.

Not from the prompt cache: each arm server gets a fresh scratch directory per round (its own --cache-dir, unused without
the disk tier, and --flight-dir), deleted after that server stops; within a server the depths share no prefix. A run
refuses to start while an `r<N>-<arm>` directory from an earlier run is in DIR (a run the guard killed leaves its
current one: `rm -rf DIR/r[0-9]*-one DIR/r[0-9]*-two` after checking it is that).

Measured per depth, from the server's own timings: decode tok/s (predicted_per_second over the generated tokens) and
prefill tok/s (prompt_per_second: prompt tokens / prompt_ms; without the disk tier no file I/O is in it, only the VRAM
checkpoint copies every served prompt pays). Every measured request must report cached tokens 0, usage.prompt_tokens ==
prompt_n == its depth, exactly --n generated tokens (an early end of text would compare different lengths), the main
slot and positive rates, else RESULT FAIL. Reported: per depth the median, range and count over the rounds of each arm,
the ratio two / one of the medians (a cell of fewer than 3 is marked: not a claim), the generated token counts, the slot,
and how many replies' text equals the first one-card round's (the server returns text, not ids; the two-card engine is not
bitwise by design, so this is information, not a gate). --rounds is at least 3 (the median of >= 3 runs a speed claim
needs); --probe allows fewer and marks the report as a probe. RESULT DONE (a benchmark: no speed gate) unless a server or
a check above fails (not ready, a request not HTTP 200, a non-zero exit, SIGTERM): RESULT FAIL. All numbers, each
server's card and disk lines included, also go to DIR/served_cards_ab.json."""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import serve_smoke as S           # noqa: E402  (request helpers, safe_dir, LIVE / stop_servers)
from serve_smoke1 import Server1  # noqa: E402

if not S.INSTALLED: sys.path.insert(0, os.path.join(ROOT, 'python'))   # (the tokenizer: the checkout's, as the server's)

MAX_CTX = 40960                  # Qwen3-30B-A3B's context_length: the largest max_ctx served without the long-context mode
DEPTHS = '8,4096,16384,32768,40768'
MIN_ROUNDS = 3                   # AGENTS.md: a speed claim is the median of at least 3 runs
# one header word per depth: each starts with a different letter, so the prompts' first tokens after the BOS differ
WORDS = ('Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo', 'Foxtrot', 'Golf', 'Hotel', 'India', 'Juliett', 'Kilo', 'Lima')
WARM_WORD, WARM_N = 'Zulu', 4    # the warm-up: BOS + 2 tokens of its own word, 4 new tokens
ARMS = (('one', '0', 1), ('two', '0,1', 2))   # (name, HIP_VISIBLE_DEVICES, --cards)
SERVE_EXTRA = ('--no-disk-cache',)            # both arms: no disk tier (no park / spill file I/O inside a timed prompt)
STALE = re.compile(r'^r\d+-(%s)$' % '|'.join(a[0] for a in ARMS))
PLACE = re.compile(r'caminus serve: cards (.*) \(HIP_VISIBLE_DEVICES=([0-9,]+)\)')


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
    caminus.serve.server's _max_new keeps prompt + max_tokens <= max_ctx on a one-token-per-pass engine), and a depth
    shorter than the previous depth's line (its prompt + n): the prompt cache keeps a request in the main slot only when
    the live line is not longer than it (cache/prompt.py route: main_n <= n), else an unrelated prompt goes to the side slot"""
    if n < 1: raise ValueError(f'--n {n}: at least 1 token')
    bad = [d for d in depths if d + n > max_ctx]
    if bad: raise ValueError(f'depth(s) {bad} + --n {n} > --max-ctx {max_ctx}: the prompt and its reply must fit')
    close = [(a, b) for a, b in zip(depths, depths[1:]) if b < a + n]
    if close: raise ValueError(f'depths {close}: each depth at least the previous + --n {n} (else the side slot serves it)')


def check_rounds(rounds, probe):
    if rounds < 1: raise ValueError(f'--rounds {rounds}: at least 1')
    if rounds < MIN_ROUNDS and not probe:
        raise ValueError(f'--rounds {rounds}: at least {MIN_ROUNDS} for a measurement (the median of >= 3 runs); --probe for fewer')


def tracked(root):
    """the paths git tracks under root's AGENTS.md and python/caminus, or None when root is no git checkout (or no git)"""
    try:
        r = subprocess.run(['git', '-C', root, 'ls-files', '-z', '--', 'AGENTS.md', 'python/caminus'],
                           capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0: return None
    return {x.decode('utf-8', 'surrogateescape') for x in r.stdout.split(b'\0') if x}


def source_files(root=ROOT):
    """the public text, in a fixed order: AGENTS.md, then every .py under python/caminus sorted by path (no vendored code);
    refuses (ValueError) a file git does not track, or a root that is no git checkout"""
    out = ['AGENTS.md']
    base = os.path.join(root, 'python', 'caminus')
    found = []
    for d, subs, files in os.walk(base):
        subs[:] = sorted(s for s in subs if s not in ('_vendor', '__pycache__'))
        found += [os.path.relpath(os.path.join(d, f), root).replace(os.sep, '/') for f in files if f.endswith('.py')]
    out += sorted(found)
    tr = tracked(root)
    if tr is None: raise ValueError(f'{root}: not a git checkout (the prompts are built from tracked files only)')
    extra = [f for f in out if f not in tr]
    if extra: raise ValueError(f'{len(extra)} file(s) git does not track would enter the prompts: {extra[:5]}')
    return out


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


def stale_dirs(scratch):
    """per-server directories an earlier run left in scratch (a run the guard killed skips its cleanup)"""
    try: names = os.listdir(scratch)
    except OSError: return []
    return sorted(os.path.join(scratch, x) for x in names if STALE.match(x) and os.path.isdir(os.path.join(scratch, x)))


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


def placement(lines):
    """the server's card line (serve/cli.py: `caminus serve: cards A + B, split ... (HIP_VISIBLE_DEVICES=i,j)`) ->
    dict(line, cards=[each card as placement describes it], hip=[its HIP indices]), or None when there is none"""
    for ln in lines:
        m = PLACE.search(ln)
        if m:
            return dict(line=ln.strip(), cards=m.group(1).split(', split ')[0].split(' + '), hip=m.group(2).split(','))
    return None


def disk_line(lines):
    """the server's engine line (`... engine built ..., disk on: PATH | off, ...`): 'off', 'on' or None when not seen"""
    for ln in lines:
        if 'engine built' in ln:
            return 'off' if ', disk off,' in ln else ('on' if ', disk on' in ln else None)
    return None


def check_cards(one, two):
    """the one-card arm's card is the two-card arm's card 0, and the arms name one and two cards"""
    if one is None or two is None: return [f'no card line from the {"one" if one is None else "two"}-card server']
    out = []
    if len(one['cards']) != 1: out.append(f'one-card server on {one["cards"]}')
    if len(two['cards']) != 2: out.append(f'two-card server on {two["cards"]}')
    if not out and one['cards'][0] != two['cards'][0]:
        out.append(f'the one-card server ran on {one["cards"][0]}, the two-card server\'s card 0 is {two["cards"][0]}')
    return out


def positive(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0


def row_fails(row, n):
    """what is wrong with one measured request (empty when it is a valid measurement): a cache hit, a token count that is
    not the depth, a reply not exactly n tokens, a request outside the main slot, a missing or zero rate"""
    d, out = row['depth'], []
    if row.get('cache_n') != 0 or row.get('prompt_tokens') != d or row.get('prompt_n') != d:
        out.append(f'cached {row.get("cache_n")}, prompt tokens {row.get("prompt_tokens")}, new {row.get("prompt_n")} '
                   f'(want 0 cached, {d} new)')
    if row.get('predicted_n') != n: out.append(f'{row.get("predicted_n")} generated tokens (want {n})')
    if row.get('route') != 'main': out.append(f'slot {row.get("route")} ({row.get("reason")}), want main')
    for k in ('decode_tps', 'prompt_tps'):
        if not positive(row.get(k)): out.append(f'{k} {row.get(k)!r}')
    return out


def run_server(A, rnd, name, card, cards, prompts, warm):
    """one arm's server for one round: the warm-up, then each depth ascending; returns (rows, server info). The scratch
    directory (cache + flight) is deleted after the server stops"""
    scratch = os.path.join(A.scratch, f'r{rnd}-{name}')
    if os.path.exists(scratch): raise Failed(f'{scratch} exists: each server starts on a fresh directory')
    os.makedirs(scratch, mode=0o700)
    a = Arm(model=A.model, max_ctx=A.max_ctx, scratch=scratch, alias=None, cards=cards, draft='off',
            block_imports='gguf,tokenizers', card=card, extra=SERVE_EXTRA)
    rows, rc, s = [], None, None
    try:
        s = Server1(f'serve[{name} r{rnd}]', a)
        try:
            if not s.wait_ready(): raise Failed(f'the {name}-card server did not come up (round {rnd})')
            if disk_line(s.lines) != 'off': raise Failed(f'the {name}-card server\'s disk tier is {disk_line(s.lines)}, want off')
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
        if rc != 0: raise Failed(f'the {name}-card server exited {rc} (round {rnd})')
        return rows, dict(round=rnd, arm=name, rc=rc, placement=placement(s.lines), disk=disk_line(s.lines))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def run(A, prompts, warm):
    """every round, each arm in ARMS order: (res {arm: {depth: [rows]}}, servers [info], fails [text]). A server failure
    or a card mismatch ends the run; a bad measured request is a FAIL that lets the run go on"""
    res = {nm: {d: [] for d in prompts} for nm, _, _ in ARMS}
    servers, fails = [], []
    try:
        for rnd in range(A.rounds):
            pl = {}
            for nm, card, cards in ARMS:
                rows, info = run_server(A, rnd, nm, card, cards, prompts, warm)
                servers.append(info); pl[nm] = info['placement']
                print(f'round {rnd} {nm}: {(info["placement"] or {}).get("line")}', flush=True)
                for row in rows:
                    res[nm][row['depth']].append(row)
                    fails += [f'round {rnd} {nm} depth {row["depth"]}: {x}' for x in row_fails(row, A.n)]
            bad = check_cards(pl.get('one'), pl.get('two'))
            if bad: raise Failed(f'round {rnd}: ' + '; '.join(bad))
    except Exception as e:                       # noqa: BLE001 - a failed server or request ends the run as RESULT FAIL
        fails.append(str(e) if isinstance(e, Failed) else f'{type(e).__name__}: {e}')
    finally:
        S.stop_servers()
    return res, servers, fails


def stats(v):
    """(median, min, max, count) of the valid values (row_fails reports the others)"""
    v = [x for x in v if positive(x)]
    return (statistics.median(v), min(v), max(v), len(v)) if v else (None, None, None, 0)


def fmt(m):
    med, lo, hi, k = m
    return '-' if med is None else f'{med:7.1f} [{lo:.1f}-{hi:.1f}] n={k}'


def report(A, prompts, res):
    names = [a[0] for a in ARMS]
    print(f'\nserved Qwen3-30B-A3B, one card (0) vs two cards (0,1), --draft off, no disk tier, {A.n} greedy tokens, '
          f'median [range] n=valid rounds of {A.rounds}' + (' (PROBE: not a claim)' if getattr(A, 'probe', False) else '') + ':')
    for key, title in (('decode_tps', 'decode tok/s'), ('prompt_tps', 'prefill tok/s (server timings)')):
        print(f'\n{title}\n{"depth":>7s} {"one card":>30s} {"two cards":>30s} {"two/one":>8s}')
        for d in prompts:
            vals = {nm: stats([r.get(key) for r in res[nm][d]]) for nm in names}
            a, b = vals['one'][0], vals['two'][0]
            ratio = f'{b / a:8.2f}' if a and b else f'{"-":>8s}'
            low = min(vals['one'][3], vals['two'][3]) < MIN_ROUNDS
            print(f'{d:7d} {fmt(vals["one"]):>30s} {fmt(vals["two"]):>30s} {ratio}' + ('  (n<3: not a claim)' if low else ''))
    print('\ngenerated tokens per round (one | two), slot, reply text equal to the first one-card round:')
    for d in prompts:
        ref = res['one'][d][0]['text'] if res['one'][d] else None
        eq = {nm: f'{sum(r["text"] == ref for r in res[nm][d])} / {len(res[nm][d])}' for nm in names}
        print(f'{d:7d} n {[r.get("predicted_n") for r in res["one"][d]]} | {[r.get("predicted_n") for r in res["two"][d]]}; '
              f'slot {sorted({str(r.get("route")) for nm in names for r in res[nm][d]})}; '
              f'equal replies one {eq["one"]}, two {eq["two"]} (information: the two-card engine is not bitwise by design)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--model', required=True); ap.add_argument('--scratch', required=True)
    ap.add_argument('--depths', default=DEPTHS); ap.add_argument('--n', type=int, default=128)
    ap.add_argument('--rounds', type=int, default=MIN_ROUNDS); ap.add_argument('--max-ctx', type=int, default=MAX_CTX)
    ap.add_argument('--probe', action='store_true', help=f'allow --rounds below {MIN_ROUNDS}: a try-out, not a claim')
    ap.add_argument('--prompts-only', action='store_true', help='build the prompts, print their counts and digests, exit (CPU)')
    A = ap.parse_args(argv)
    try:
        depths = parse_depths(A.depths)
        check_fit(depths, A.n, A.max_ctx)
        check_rounds(A.rounds, A.probe)
    except ValueError as e:
        print(f'served_cards_ab: {e}', file=sys.stderr); return 2
    A.scratch = S.safe_dir(A.scratch); os.makedirs(A.scratch, mode=0o700, exist_ok=True)
    stale = [] if A.prompts_only else stale_dirs(A.scratch)
    if stale:
        print(f'served_cards_ab: an earlier run left {stale}: check and remove them (rm -rf on those paths), then rerun',
              file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    tok = tokenizer(A.model)
    try:
        prompts = build_prompts(depths, tok.encode_plain, tok.bos_id)
    except ValueError as e:
        print(f'served_cards_ab: {e}', file=sys.stderr); return 2
    warm = warmup_ids(tok.encode_plain, tok.bos_id, depths[0])
    print(f'prompts built in {time.perf_counter() - t0:.1f} s (BOS {tok.bos_id}; warm-up {warm}):', flush=True)
    for d, ids in prompts.items():
        print(f'  depth {d:6d}: {len(ids)} tokens, sha {ids_sha(ids)}, starts {ids[:4]}', flush=True)
    if A.prompts_only: return 0

    def term(signum, frame):                     # the guard's TERM: unwind through the cleanups (servers, scratch), once
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise Failed('SIGTERM (the guard stopped the run)')
    prev = signal.signal(signal.SIGTERM, term)
    try:
        res, servers, fails = run(A, prompts, warm)
    finally:
        signal.signal(signal.SIGTERM, prev)
    with open(os.path.join(A.scratch, 'served_cards_ab.json'), 'w') as f:   # (first: a report error cannot lose the run)
        json.dump(dict(model=os.path.basename(A.model), depths=depths, n=A.n, rounds=A.rounds, probe=A.probe,
                       max_ctx=A.max_ctx, serve_extra=list(SERVE_EXTRA), prompt_sha={d: ids_sha(ids) for d, ids in prompts.items()},
                       servers=servers, results=res, fails=fails), f)
    report(A, prompts, res)
    for f in fails: print(f'FAIL {f}', flush=True)
    print('RESULT ' + ('FAIL' if fails else 'DONE'), flush=True)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
