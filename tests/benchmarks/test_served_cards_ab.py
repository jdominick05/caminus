"""The served one-card / two-card A/B's CPU parts (no GPU, no server, no model needed).

  python -I tests/benchmarks/test_served_cards_ab.py --scratch DIR

Checks benchmarks/served_cards_ab.py: --depths parsing (ascending, distinct, positive), the fit check (every depth + --n
within --max-ctx, each depth at least the previous + --n, refused otherwise, before anything is loaded), --rounds (at least
3 unless --probe), and the prompts: exactly depth tokens each, the BOS first, no prompt a prefix of another and any two
sharing only the BOS, the same ids on every build, the source read in its fixed order without vendored code and only from
files git tracks (needs a git checkout and git), a too-short source refused, and the warm-up shorter than the first
depth. Then the run-time checks with a stand-in server, client and status (no process starts): the serve flags
(--no-disk-cache on both arms), each request's row checks (a cache hit, a wrong token count, a short reply, the side slot,
a zero rate are FAILs), the card line and disk line readers, a card mismatch between the arms, the report (counts per
cell, n<3 marked) and the stale-directory check. A stand-in byte tokenizer covers every case; when $CODER names a GGUF
(CI's vocab-only file is enough) the model line's own BPE (caminus.serve.tokenizer.QwenTokenizer, engine 'own') builds a
set of prompts too. Prints PASS / FAIL lines and RESULT PASS | FAIL."""
import argparse
import contextlib
import io
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'benchmarks'))
import served_cards_ab as B  # noqa: E402

FAILS = []


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)


def refused(fn, *a):
    try:
        fn(*a)
    except ValueError:
        return True
    return False


def byte_encode(text):
    return list(text.encode('utf-8'))   # (a stand-in tokenizer: one id per byte, ids 0..255)


BOS = 1000


def prompt_checks(prompts, depths, bos, label):
    check(list(prompts) == depths and all(len(prompts[d]) == d for d in depths),
          f'{label}: each depth exactly that many tokens ({[len(prompts[d]) for d in depths]})')
    check(all(p[0] == bos for p in prompts.values()) if bos is not None else True, f'{label}: the BOS first')
    pairs = [(a, b) for i, a in enumerate(depths) for b in depths[i + 1:]]
    shared = {(a, b): B.lcp(prompts[a], prompts[b]) for a, b in pairs}
    want = 0 if bos is None else 1
    check(all(v == want for v in shared.values()), f'{label}: any two prompts share exactly {want} leading token(s) {shared}')
    check(not any(prompts[b][:a] == prompts[a] for a, b in pairs), f'{label}: no prompt is a prefix of a longer one')


ONE = 'RX 7900 XTX card1 (0000:03:00.0)'
TWO = 'RX 7900 XT card2 (0000:07:00.0)'


def card_line(cards, hip):
    return f'caminus serve: cards {cards} (HIP_VISIBLE_DEVICES={hip})'


ENGINE = ('caminus serve: engine built in 9.0 s: max_ctx 40960 (usable 40960), qsa False, drafter none, prefill chunk 2048 '
          '(T* 64), cache 6 VRAM slots (100 / 100 MiB), disk off, side slot 4096 positions (routing on)')


class FakeServer:
    """Server1's surface for run_server: lines, port, wait_ready, stop (no process)"""
    made = []
    swap = False   # True: the two-card server puts the other card first (a card mismatch)

    def __init__(self, tag, a):
        self.a, self.port = a, 1
        two = f'{TWO} + {ONE}' if FakeServer.swap else f'{ONE} + {TWO}'
        self.lines = [card_line(ONE, '0') if a.cards == 1 else card_line(two + ', split ffn 2/3', '0,1'), ENGINE]
        FakeServer.made.append(self)

    def wait_ready(self): return True

    def stop(self): return 0


def run_checks(scratch, prompts):
    """the run-time part with stand-ins for the server, the client and the status read"""
    # the readers
    pl = B.placement(['x', card_line(f'{ONE} + {TWO}, split ffn 2/3', '0,1') + ', drafter on card 1'])
    check(pl is not None and pl['cards'] == [ONE, TWO] and pl['hip'] == ['0', '1'] and B.placement(['nothing']) is None,
          f'the card line: cards and HIP indices ({pl})')
    check(B.disk_line([ENGINE]) == 'off' and B.disk_line([ENGINE.replace('disk off', 'disk on: /x')]) == 'on'
          and B.disk_line(['x']) is None, 'the disk line: off, on, not seen')
    one, two = B.placement([card_line(ONE, '0')]), B.placement([card_line(f'{ONE} + {TWO}', '0,1')])
    swapped = B.placement([card_line(f'{TWO} + {ONE}', '1,0')])
    check(B.check_cards(one, two) == [] and B.check_cards(one, swapped) and B.check_cards(two, two)
          and B.check_cards(None, two), 'cards: the one-card arm on the two-card arm\'s card 0, else a FAIL')

    # one request's row checks
    good = dict(depth=4096, cache_n=0, prompt_tokens=4096, prompt_n=4096, predicted_n=128, route='main', reason='main_shorter',
                decode_tps=50.0, prompt_tps=1000.0)
    check(B.row_fails(good, 128) == [], 'row: a valid measurement passes')
    for over, what in ((dict(cache_n=4095, prompt_n=1), 'a cache hit'), (dict(prompt_tokens=4097), 'prompt tokens not the depth'),
                       (dict(predicted_n=40), 'a reply shorter than --n'), (dict(route='side', reason='unrelated'), 'the side slot'),
                       (dict(route=None), 'no status read'), (dict(prompt_tps=0.0), 'a zero prefill rate'),
                       (dict(decode_tps=None), 'no decode rate')):
        check(len(B.row_fails(dict(good, **over), 128)) == 1, f'row: {what} is a FAIL')

    # a whole run on stand-ins
    def fake_complete(port, ids, n):
        return dict(choices=[{'text': 'reply'}], usage={'prompt_tokens': len(ids)},
                    timings=dict(cache_n=0, prompt_n=len(ids), prompt_ms=1.0, prompt_per_second=1000.0 + len(ids) / 1000,
                                 predicted_n=n if len(ids) != 4096 or FakeServer.short is False else n - 1,
                                 predicted_per_second=50.0))
    status = (200, {'cache': {'route': {'last': {'route': 'main', 'reason': 'main_shorter'}}}})
    saved = (B.Server1, B.complete, B.S.get)
    B.Server1, B.complete, B.S.get = FakeServer, fake_complete, (lambda port, path: status)
    try:
        depths = list(prompts)
        A = B.Arm(model='m.gguf', max_ctx=B.MAX_CTX, scratch=tempfile.mkdtemp(prefix='run-', dir=scratch), n=128, rounds=3, probe=False)
        FakeServer.made, FakeServer.swap, FakeServer.short = [], False, False
        res, servers, fails = B.run(A, prompts, [BOS, 1, 2])
        check(fails == [] and len(FakeServer.made) == 6 and all(len(res[nm][d]) == 3 for nm in ('one', 'two') for d in depths),
              f'run: 3 rounds x 2 arms, every depth measured, no FAIL ({fails})')
        check(all(s.a.extra == ('--no-disk-cache',) and s.a.draft == 'off' for s in FakeServer.made)
              and [s.a.cards for s in FakeServer.made] == [1, 2] * 3, 'run: both arms --no-disk-cache, --draft off, one card first')
        check([x['disk'] for x in servers] == ['off'] * 6 and servers[1]['placement']['cards'] == [ONE, TWO],
              'run: each server\'s disk and card lines recorded')
        check(B.stale_dirs(A.scratch) == [], 'run: each server\'s scratch directory removed')
        out = io.StringIO()
        with contextlib.redirect_stdout(out): B.report(A, prompts, res)
        txt = out.getvalue()
        check('n=3' in txt and 'not a claim' not in txt and '1.00' in txt, 'report: n=3 per cell, ratio 1.00, no not-a-claim mark')

        A1 = B.Arm(**dict(vars(A), rounds=1, probe=True))
        res1 = {nm: {d: v[:1] for d, v in r.items()} for nm, r in res.items()}
        out = io.StringIO()
        with contextlib.redirect_stdout(out): B.report(A1, prompts, res1)
        check('n<3: not a claim' in out.getvalue() and 'PROBE' in out.getvalue(), 'report: a one-round probe is marked not a claim')

        FakeServer.short = True
        _, _, fails = B.run(A, prompts, [BOS, 1, 2])
        check(len(fails) == 6 and all('generated tokens' in f for f in fails), f'run: a short reply at 4,096 FAILs each time ({len(fails)})')
        FakeServer.short, FakeServer.swap = False, True
        _, _, fails = B.run(A, prompts, [BOS, 1, 2])
        check(len(fails) == 1 and 'card 0' in fails[0], f'run: a two-card server whose card 0 is not the one-card arm\'s ends it ({fails})')
        FakeServer.swap = False
        os.makedirs(os.path.join(A.scratch, 'r0-two'))
        check(B.stale_dirs(A.scratch) == [os.path.join(A.scratch, 'r0-two')], 'a directory an earlier run left is found')
        _, _, fails = B.run(A, prompts, [BOS, 1, 2])
        check(len(fails) == 1 and 'exists' in fails[0], 'run: a server directory already there is refused')
        os.rmdir(os.path.join(A.scratch, 'r0-two'))
    finally:
        B.Server1, B.complete, B.S.get = saved


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--scratch', required=True)
    a = ap.parse_args()
    os.makedirs(a.scratch, exist_ok=True)
    default = B.parse_depths(B.DEPTHS)

    # --depths
    check(default == [8, 4096, 16384, 32768, 40768], f'the default depths {default}')
    check(all(refused(B.parse_depths, t) for t in ('8,4096,4096', '4096,8', '0,8', '-1', 'eight', '', ','.join(map(str, range(1, 20))))),
          '--depths refused: a repeat, descending, zero, negative, not a number, none, more depths than header words')

    # the fit check: depth + n <= max_ctx
    check(not refused(B.check_fit, default, 128, B.MAX_CTX) and not refused(B.check_fit, default, 192, B.MAX_CTX),
          'fit: the default depths with --n 128 and 192 (40,768 + 192 = 40,960) fit max_ctx 40,960')
    check(refused(B.check_fit, default, 193, B.MAX_CTX) and refused(B.check_fit, [8, 4096], 4096, 8191)
          and refused(B.check_fit, [8], 0, B.MAX_CTX), 'fit refused: 40,768 + 193 > 40,960; 4,096 + 4,096 > 8,191; --n 0')
    check(refused(B.check_fit, [4096, 4160], 128, B.MAX_CTX) and not refused(B.check_fit, [4096, 4224], 128, B.MAX_CTX),
          'fit: 4,160 after 4,096 with --n 128 refused (the side slot would serve it), 4,224 accepted')
    check(refused(B.check_rounds, 2, False) and refused(B.check_rounds, 0, True) and not refused(B.check_rounds, 1, True)
          and not refused(B.check_rounds, 3, False), '--rounds: 3 or more, 1 or 2 only with --probe, never 0')
    for argv, what in ((['--n', '193'], '--n 193 at depth 40,768'), (['--depths', '8,8'], '--depths 8,8'),
                       (['--max-ctx', '4096', '--depths', '8,4096'], 'depth 4,096 at --max-ctx 4096'),
                       (['--rounds', '0'], '--rounds 0'), (['--rounds', '2'], '--rounds 2 without --probe'),
                       (['--depths', '4096,4160'], 'depth 4,160 after 4,096 at --n 128')):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):   # (refused before the model is opened: the path does not exist)
            try:
                rc = B.main(['--model', os.path.join(a.scratch, 'absent.gguf'), '--scratch', a.scratch] + argv)
            except Exception as e:              # noqa: BLE001 - not refused: it went on to open the absent model
                rc = f'{type(e).__name__}'
        check(rc == 2 and 'served_cards_ab:' in err.getvalue(), f'main refuses {what} with exit 2 before loading anything ({rc})')

    # the source text
    files = B.source_files()
    check(files[0] == 'AGENTS.md' and files[1:] == sorted(files[1:]) and all(f.startswith('python/caminus/') for f in files[1:])
          and not any('/_vendor/' in f for f in files) and all(os.path.isfile(os.path.join(ROOT, f)) for f in files),
          f'source: AGENTS.md, then {len(files) - 1} sorted python/caminus files, no vendored code')

    # prompts from the stand-in tokenizer, on this checkout's source
    p1 = B.build_prompts(default, byte_encode, BOS)
    prompt_checks(p1, default, BOS, 'byte tokenizer')
    check(p1 == B.build_prompts(default, byte_encode, BOS), 'the same ids on a second build')
    check(B.ids_sha(p1[4096]) != B.ids_sha(p1[16384]), 'distinct digests per depth')
    pn = B.build_prompts([8, 300, 5000], byte_encode, None)
    prompt_checks(pn, [8, 300, 5000], None, 'no BOS in the vocabulary')
    lead = byte_encode('Charlie (16384 tokens): the Caminus source tree, read file by file.\n==> AGENTS.md <==\n')
    check(p1[16384][1:1 + len(lead)] == lead, 'the header names the depth with its own word, the source follows')
    check(refused(B.check_distinct, {4: [BOS, 5, 6, 7], 8: [BOS, 5, 9, 9, 9, 9, 9, 9]}, BOS)
          and refused(B.check_distinct, {4: [BOS, 5, 6, 7], 8: [BOS, 5, 6, 7, 1, 2, 3, 4]}, BOS)
          and refused(B.check_distinct, {4: [BOS, 5, 6]}, BOS), 'check_distinct refuses a shared prefix, a prefix, a wrong length')

    # a source too short for the deepest prompt (a stand-in tree, its own git checkout: AGENTS.md + one module)
    root = tempfile.mkdtemp(prefix='tree-', dir=a.scratch)   # (fresh on every run: it becomes a git checkout below)
    os.makedirs(os.path.join(root, 'python', 'caminus', '_vendor'), exist_ok=True)
    for rel, text in (('AGENTS.md', 'agents\n'), ('python/caminus/a.py', 'x = 1\n'), ('python/caminus/_vendor/v.py', 'v\n')):
        with open(os.path.join(root, rel), 'w') as f: f.write(text)
    check(refused(B.source_files, root), 'a tree that is no git checkout is refused')
    git = ['git', '-C', root, '-c', 'init.defaultBranch=main']
    subprocess.run(git + ['init', '-q'], check=True)
    subprocess.run(git + ['add', 'AGENTS.md', 'python'], check=True)
    check(B.source_files(root) == ['AGENTS.md', 'python/caminus/a.py'], 'the vendored directory is skipped')
    check(refused(B.build_prompts, [8, 4096], byte_encode, BOS, root), 'a source shorter than the deepest prompt is refused')
    with open(os.path.join(root, 'python', 'caminus', 'b.py'), 'w') as f: f.write('y = 2\n')
    check(refused(B.source_files, root), 'an untracked .py under python/caminus is refused')

    # the warm-up: shorter than the first depth (else none)
    w = B.warmup_ids(byte_encode, BOS, 8)
    check(w == [BOS] + byte_encode('Zu') and B.warmup_ids(byte_encode, BOS, 5) is None and B.lcp(w, p1[8]) == 1,
          f'warm-up {w}: BOS + 2 tokens of its own word, none when its line is not shorter than the first depth')

    run_checks(a.scratch, p1)

    # the model line's own BPE, when a GGUF with the vocabulary is here
    coder = os.environ.get('CODER', '')
    if coder and os.path.isfile(coder):
        from caminus.serve.tokenizer import QwenTokenizer
        tok = QwenTokenizer.from_gguf(coder, engine='own')
        ds = [8, 512, 2048]
        pr = B.build_prompts(ds, tok.encode_plain, tok.bos_id)
        prompt_checks(pr, ds, tok.bos_id, 'the own BPE ($CODER\'s vocabulary)')
    else:
        print('SKIP the own-BPE build: $CODER is not set to a GGUF', flush=True)

    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'), flush=True)
    return 0 if not FAILS else 1


if __name__ == '__main__':
    sys.exit(main())
