"""The privacy and secret scan of a tree (CI runs it on every push: .github/workflows/ci.yml). CPU only, no network.

  python -I tools/privacy_scan.py [DIR]          every file under DIR (default: the repository root), .git skipped; exit 1 on a hit
  python -I tools/privacy_scan.py --digest TERM  the digest of a private term, to add it to PRIVATE below

What fails: secret patterns (API tokens of Anthropic / OpenAI / Hugging Face / GitHub / PyPI / AWS / Slack / DigitalOcean /
Google, private-key blocks, `token = "..."` assignments), machine paths (a home directory, removable-media mounts, macOS /
Windows user directories), agent job directories, session links, quoted maintainer instructions, and private names. The private
names (the development machine's user and host names, private repositories and files) are matched by salted sha256 digests of
lowercased words, word pairs and path segment pairs, so this file names none of them; the patterns are written so that they
do not match themselves. A hit prints the file, the line and the category, never more than the matching text."""
import hashlib
import os
import re
import sys

SALT = 'caminus-privacy:'
PRIVATE = {
    "27815a1e49194271f890377a9cbfc67ea3ee9471d26db9ec74b03a75fd922134",
    "60b66e5b3c2418f28bf226991319529c313b31ffd6571a7d39845514fa13511c",
    "cc73a5cb0f346973cd138750478b6440c0e566c9e0c6472da34dfae6ae781c3f",
    "31d515bd323ea7be5373244043173dbce7e9eeca5e56930dca67d9cffae0b12f",
    "b3ddac62c7deaf0d98666a69c7dc25a9ebe3074c6c821013fdec8375baa630ea",
    "c7e276554b59ccdf4ba7ee470e48f303e23be7d8c45f7796b68781dcc078ae5b",
    "6a53e5da06ecdc9cbc09c133a6d3d6680ed12f1124728430465ab48dd2711ffe",
    "81060ee21f9b8a7133e34378edfd02e1f23e595e922f2fee384d0c766b8cb604",
    "05c6276a8453ce7035577de019082ee4b4b256a6ed7844792b1e190693f8ec7c",
    "d80ddb4afa9deeecf38ab0b62d97f4eed8045bc54c6d9bfba33e6958e62c5a4c",
    "e496e57c86017f03585f4985335dd2fc3afe7f16fe5158a5a3314ebd5195872e",
    "fbd42452e3c1dcef87d83a384fb2202bb935dbb09e85455f6088eb728fdb57e4",
    "f05d5ad9a5bfa6931dd0d3dc7d80b197f716b74ce29b6b87bf4d499e93c20b8b",
    "971e048553939ece912b1362c1167b9cf8b23849a4c26b706b341bde8283474f",
    "9ec5fccc9f3b15caecf66f60cf3bf47242f4a5232e54e5b1beec0014eb5ac59f",
    "3981c46f1065a5b199d9a0f316df23ae8de34085c2cf030b904d325006e0e8e8",
    "b2e8ea81fe054e4a13d963f98f6913834a7ce9fcbd25546d6460da5ece2943f6",
    "e752eefd815cdc8e21f6b7aab0ed695c395727c0b7c9ed31c54fb41dcc816b0b",
    "a00b8b29225f04f66cfac85d5c8b3607ed36c4802f550dff7f4ae222319a54c8",
    "5266853cc33191d5e65eecfcdcd9f89c93e10d266907511ccfd17d68604ca937",
}
PATTERNS = {
    'secret': [r'sk-ant-[A-Za-z0-9_-]{8,}', r'\bsk-[A-Za-z0-9]{32,}', r'\bhf_[A-Za-z0-9]{30,}', r'\bgh[pousr]_[A-Za-z0-9]{30,}',
               r'github_pat_[A-Za-z0-9_]{20,}', r'\bpypi-[A-Za-z0-9_-]{40,}', r'\bAKIA[0-9A-Z]{16}\b', r'\bxox[abposr]-[A-Za-z0-9-]{10,}',
               r'\bdop_v1_[a-f0-9]{20,}', r'\bAIza[0-9A-Za-z_-]{35}\b', r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
               r'(?i)\b(api[_-]?key|secret|password|token)\s*[:=]\s*["\'][A-Za-z0-9_\-]{16,}["\']'],
    'machine path': [r'/home/[a-z_][a-z0-9_-]*/', r'/run/medi[a]/', r'/media/[a-z_][a-z0-9_-]*/', r'\b[0-9A-F]{16}\b(?=/)',
                     r'/Users/[A-Za-z]', r'[A-Z]:\\\\Users'],
    'agent job directory': [r'\.claude/jo[b]s', r'\bjobs/[0-9a-f]{8}\b'],
    'session link': [r'Claude-Sessio[n]', r'claude\.ai/code/sessio[n]', r'claude\.ai/cha[t]/'],
    'quoted instruction': [r'\(user,? 1[01]-\d\d: "', r'user decision 1[01]-\d\d, "'],
}
WORD = re.compile(r'[a-z0-9~][a-z0-9._~-]*')
PATH = re.compile(r'[a-z0-9._~-]+(?:/[a-z0-9._~-]+)+')


def digest(term):
    return hashlib.sha256((SALT + term.lower()).encode()).hexdigest()


def private_hits(line):
    """the private terms in a line, by digest: each word (and the word without a trailing extension or punctuation), each pair
    of adjacent words, each pair of adjacent path segments"""
    low = line.lower()
    words = WORD.findall(low)
    cands = set()
    for w in words:
        w2 = w.rstrip('.,;:')
        cands.update((w, w2, w2.rsplit('.', 1)[0]))
    cands.update(f'{a.rstrip(".,;:")} {b.rstrip(".,;:")}' for a, b in zip(words, words[1:]))
    for p in PATH.findall(low):
        seg = p.split('/')
        cands.update(f'{a}/{b}' for a, b in zip(seg, seg[1:]))
    return sorted(c for c in cands if digest(c) in PRIVATE)


def walk(d):
    for base, dirs, files in os.walk(d):
        dirs[:] = sorted(x for x in dirs if not (base == d and x == '.git'))
        for f in sorted(files):
            yield os.path.relpath(os.path.join(base, f), d)


def scan(d):
    pats = {k: [re.compile(p) for p in v] for k, v in PATTERNS.items()}
    hits = []
    nfiles = 0
    for rel in walk(d):
        for k, ps in pats.items():
            if any(p.search(rel) for p in ps):
                hits.append((rel, 0, k, rel))
        for t in private_hits(rel):
            hits.append((rel, 0, 'private name', '(a private term in the path)'))
        p = os.path.join(d, rel)
        if os.path.islink(p):
            continue
        with open(p, 'rb') as f:
            data = f.read()
        nfiles += 1
        if b'\0' in data[:8192]:
            continue                                  # binary (code objects, images)
        for i, line in enumerate(data.decode('utf-8', errors='replace').splitlines(), 1):
            for k, ps in pats.items():
                for x in ps:
                    m = x.search(line)
                    if m:
                        hits.append((rel, i, k, m.group(0)))
            if private_hits(line):
                hits.append((rel, i, 'private name', '(a private term: see the digest list)'))
    for rel, i, k, what in hits:
        print(f'FAIL {k}: {rel}:{i}: {what!r}')
    print(f'privacy_scan {d}: {nfiles} files, {len(hits)} hits')
    print('RESULT', 'FAIL' if hits else 'CLEAN')
    return 1 if hits else 0


def main(argv):
    if argv[:1] == ['--digest'] and len(argv) == 2:
        print(digest(argv[1]))
        return 0
    if argv[:1] in (['-h'], ['--help']) or len(argv) > 1:
        print(__doc__)
        return 0 if argv[:1] in (['-h'], ['--help']) else 2
    return scan(os.path.abspath(argv[0]) if argv else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
