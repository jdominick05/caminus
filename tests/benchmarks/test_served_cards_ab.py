"""The served one-card / two-card A/B's prompt builder and argument checks (CPU: no GPU, no server, no model needed).

  python -I tests/benchmarks/test_served_cards_ab.py --scratch DIR

Checks benchmarks/served_cards_ab.py: --depths parsing (ascending, distinct, positive), the fit check (every depth + --n
within --max-ctx, refused otherwise, before anything is loaded), and the prompts: exactly depth tokens each, the BOS first,
no prompt a prefix of another and any two sharing only the BOS, the same ids on every build, the source read in its fixed
order without vendored code, a too-short source refused, and the warm-up shorter than the first depth. A stand-in byte
tokenizer covers every case; when $CODER names a GGUF (CI's vocab-only file is enough) the model line's own BPE
(caminus.serve.tokenizer.QwenTokenizer, engine 'own') builds a set of prompts too. Prints PASS / FAIL lines and
RESULT PASS | FAIL."""
import argparse
import contextlib
import io
import os
import sys

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
    for argv, what in ((['--n', '193'], '--n 193 at depth 40,768'), (['--depths', '8,8'], '--depths 8,8'),
                       (['--max-ctx', '4096', '--depths', '8,4096'], 'depth 4,096 at --max-ctx 4096'),
                       (['--rounds', '0'], '--rounds 0')):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):   # (refused before the model is opened: the path does not exist)
            rc = B.main(['--model', os.path.join(a.scratch, 'absent.gguf'), '--scratch', a.scratch] + argv)
        check(rc == 2 and 'served_cards_ab:' in err.getvalue(), f'main refuses {what} with exit 2 before loading anything')

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

    # a source too short for the deepest prompt (a stand-in tree: AGENTS.md + one module)
    root = os.path.join(a.scratch, 'tree')
    os.makedirs(os.path.join(root, 'python', 'caminus', '_vendor'), exist_ok=True)
    for rel, text in (('AGENTS.md', 'agents\n'), ('python/caminus/a.py', 'x = 1\n'), ('python/caminus/_vendor/v.py', 'v\n')):
        with open(os.path.join(root, rel), 'w') as f: f.write(text)
    check(B.source_files(root) == ['AGENTS.md', 'python/caminus/a.py'], 'the vendored directory is skipped')
    check(refused(B.build_prompts, [8, 4096], byte_encode, BOS, root), 'a source shorter than the deepest prompt is refused')

    # the warm-up: shorter than the first depth (else none)
    w = B.warmup_ids(byte_encode, BOS, 8)
    check(w == [BOS] + byte_encode('Zu') and B.warmup_ids(byte_encode, BOS, 5) is None and B.lcp(w, p1[8]) == 1,
          f'warm-up {w}: BOS + 2 tokens of its own word, none when its line is not shorter than the first depth')

    # the net prefill rate
    check(B.net_tps(dict(prompt_ms=2000.0, restore_ms=1000.0, prompt_n=4096)) == 4096.0
          and B.net_tps(dict(prompt_ms=2000.0, restore_ms=None, prompt_n=4096)) is None, 'net prefill: tokens / (prompt - restore ms)')

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
