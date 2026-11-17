"""The anti-decay checks of ENTROPY.md that read the tree: what it names that is not there, and what it holds that nothing
uses. CPU, standard library only, no network; the CPU suite `entropy` (tests/tools/test_entropy.py) runs it on this tree.

  python -I tools/entropy.py [DIR]           the checks on DIR (default: this repository); exit 1 on a FAIL
  python -I tools/entropy.py --prune [DIR]   first drop the baseline entries that stopped applying (it never adds one)

  refs      a path a doc names is a file of the tree: any path under a top-level directory, and in a code span or a
            fenced block any source file's name (`engine/k3.py`, `k3_run.hip`, `./build.sh`). A name matches a tracked path
            or the end of one, `*` and `{a,b}` as the shell reads them. DOCS_SKIP lists the docs that name other trees.
  orphans   a test script (tests/**/test_*.py), a tool or a benchmark is named in some other file: a suite, a gate script,
            a doc with its command, an import. A directory with a run_all.py is discovered whole.
  copies    no two files of 512 bytes or more are the same bytes (test data, third-party text and __init__.py aside), and
            the pairs the baseline declares under "copies" stay equal.

The baseline, tools/entropy.json, lists by name what was already wrong when a check arrived. A check fails on anything not
listed and on a listed entry that no longer applies, so the file is exactly what is wrong today and only shrinks; adding
an entry is an edit by hand that the pull request explains (tools/pr_ready.py, `size`, asks for it)."""
import argparse
import collections
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = 'tools/entropy.json'
THIRD_PARTY = ('kernels/ggml_mmq/', 'python/caminus/_vendor/', 'packaging/build-image/licenses/', 'LICENSES/', 'LICENSE')
WALK_SKIP = {'.git', 'build', 'dist', '__pycache__', '.pybuild', 'node_modules'}

# refs: the docs that are not read as naming this tree (a path, or a directory with its slash) -> why
DOCS_SKIP = {
    'CHANGELOG.md': 'past releases, as they were',
    'CLA.md': 'legal text',
    'THIRD_PARTY.md': 'names the upstream trees',
    '.github/': 'templates',
    'python/caminus/share/harness/': 'prompts shipped to another account: their paths are the reader\'s',
    'packaging/harness/': 'the same prompts, for a checkout\'s wrappers',
    'docs/dev/engine-design.md': 'the measured history: it names the files of earlier trees',
    'docs/specs/sweep-262k.md': 'one sweep\'s report',
}
CODE = 'py|hip|h|go|sh|cpp|c'
COMP = r'(?:[\w.*-]|\{[\w.,*/-]*\})+'
PATHTOK = re.compile(r'(?<![\w./$~<{%-])((?:' + COMP + r'/)*' + COMP + r'\.(?:' + CODE +
                     r'|md|json|toml|yml|yaml|svg|txt|sha256|in|cfg|service))(?![\w/-])')
SPAN = re.compile(r'`([^`\n]+)`')
NOT_PATHS = {'llama.cpp'}                                  # (a project's name)


def files_of(root):
    """the tree's files, '/'-separated and sorted: what git tracks plus the new files it does not ignore; without a .git
    (or a git that can list it), every file outside the build outputs"""
    if os.path.lexists(os.path.join(root, '.git')):
        r = subprocess.run(['git', '-C', root, 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], capture_output=True)
        if r.returncode == 0:
            out = r.stdout.decode('utf-8', 'surrogateescape').split('\0')
            return sorted({f for f in out if f and os.path.lexists(os.path.join(root, f))})
        print(f'entropy: git cannot list {root}, walking it instead: {r.stderr.decode(errors="replace").strip()[:200]}', file=sys.stderr)
    out = []
    for base, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in WALK_SKIP and not d.endswith('.egg-info')]
        out += [os.path.relpath(os.path.join(base, n), root).replace(os.sep, '/') for n in names if not n.endswith('.pyc')]
    return sorted(out)


class Tree:
    def __init__(self, root):
        self.root = root
        self.files = files_of(root)
        self.set = set(self.files)
        self.top = {f.split('/')[0] for f in self.files if '/' in f}
        self.ends = {'/'.join(p[i:]) for p in (f.split('/') for f in self.files) for i in range(len(p))}

    def data(self, f):
        """a file's bytes; b'' for a link or one that is gone"""
        p = os.path.join(self.root, f)
        if os.path.islink(p) or not os.path.isfile(p):
            return b''
        with open(p, 'rb') as fh:
            return fh.read()

    def text(self, f):
        d = self.data(f)
        return '' if b'\0' in d[:8192] else d.decode('utf-8', errors='replace')


def third_party(f):
    return f.startswith(THIRD_PARTY)


def braces(tok):
    """a{b,c}d -> [abd, acd]"""
    m = re.search(r'\{([^{}]*)\}', tok)
    if not m:
        return [tok]
    return [x for alt in m.group(1).split(',') for x in braces(tok[:m.start()] + alt + tok[m.end():])]


def resolves(tree, tok):
    for t in braces(tok):
        if '*' in t:
            if not any(fnmatch.fnmatchcase(f, t) or fnmatch.fnmatchcase(f, '*/' + t) for f in tree.files):
                return False
        elif t not in tree.ends:
            return False
    return True


def named_paths(tree, text):
    """(line, name) of every path a doc's text names: see refs in the module text"""
    out, fenced = [], False
    for n, line in enumerate(text.split('\n'), 1):
        if line.lstrip().startswith('```'):
            fenced = not fenced
            continue
        spans = [line] if fenced else SPAN.findall(line)
        for m in PATHTOK.finditer(line):
            tok = re.sub(r'^\*\*(?=\w)', '', m.group(1))                  # (**bold**)
            while tok.startswith('./'):
                tok = tok[2:]
            first, _, rest = tok.partition('/')
            if tok in NOT_PATHS:
                continue
            if rest and first in tree.top:
                out.append((n, tok))                                      # a path of this tree, wherever it stands
            elif not any(tok in s for s in spans) or not re.fullmatch(CODE, tok.rsplit('.', 1)[1]):
                continue                                                  # elsewhere: a source file's name, as code
            elif rest and re.fullmatch(r'[A-Z][A-Z0-9_]*', first):
                continue                                                  # DIR/x.py: a placeholder
            elif not rest and '.' in tok.rsplit('.', 1)[0]:
                continue                                                  # a.b.c: a module, not a file
            else:
                out.append((n, tok))
    return out


class Run:
    def __init__(self, tree, base, prune):
        self.tree, self.base, self.prune = tree, base, prune
        self.fails = self.pruned = 0

    def fail(self, check, what):
        self.fails += 1
        print(f'FAIL {check}: {what}')

    def known(self, check, found, listed, show, what=''):
        """found against the baseline's list: a new one fails (show(x) says where it is and what to do); a listed one that
        no longer applies fails too, or is dropped with --prune. Returns the entries to keep and the failures."""
        new, stale = sorted(set(found) - set(listed)), sorted(set(listed) - set(found))
        for x in new:
            self.fail(check, show(x))
        if self.prune:
            self.pruned += len(stale)
        else:
            for x in stale:
                self.fail(check, f'{BASELINE} lists {what}{x}, which no longer applies: python -I tools/entropy.py --prune')
        return sorted(set(listed) & set(found)), len(new) + (0 if self.prune else len(stale))

    def refs(self):
        t, listed = self.tree, self.base.get('refs', {})
        docs = [f for f in t.files if f.endswith('.md') and not third_party(f)
                and not any(f == k or k.endswith('/') and f.startswith(k) for k in DOCS_SKIP)]
        keep, bad, nnames = {}, 0, 0
        for d in docs + sorted(set(listed) - set(docs)):                  # (a listed doc that is gone or skipped: all stale)
            names = named_paths(t, t.text(d)) if d in docs else []
            nnames += len(names)
            dead = {}
            for n, tok in names:
                if not resolves(t, tok):
                    dead.setdefault(tok, n)
            k, b = self.known('refs', dead, listed.get(d, []), lambda x: f'{d}:{dead[x]}: names {x}, which is not in the tree: '
                              f'correct the name, or delete the sentence', f'{d} naming ')
            bad += b
            if k:
                keep[d] = k
        self.base['refs'] = keep
        if not bad:
            print(f'PASS refs: {nnames} paths named in {len(docs)} docs, none newly missing'
                  f' ({sum(map(len, keep.values()))} known dead, listed in {BASELINE})')

    def orphans(self):
        t = self.tree
        subj = [f for f in t.files if not third_party(f) and os.path.dirname(f) + '/run_all.py' not in t.set and (
            f.startswith('tests/') and re.fullmatch(r'test_.*\.py', os.path.basename(f))
            or f.startswith(('tools/', 'benchmarks/')) and f.endswith(('.py', '.sh', '.cpp', '.hip')))]
        want = {os.path.basename(f) for f in subj} | {os.path.basename(f).rsplit('.', 1)[0] for f in subj}
        namers = collections.defaultdict(set)                             # a wanted name -> up to two files that say it
        for f in t.files:
            if f == BASELINE or third_party(f):
                continue
            for w in set(re.findall(r'[A-Za-z0-9_][\w.-]*', t.text(f))):
                w = w.rstrip('.-')
                for k in {w, *w.split('.')} & want:
                    if len(namers[k]) < 2:
                        namers[k].add(f)

        def named(f):
            b = os.path.basename(f)
            stem = b.rsplit('.', 1)[0]
            return bool(namers[b] - {f}) or len(stem) > 5 and bool(namers[stem] - {f})

        keep, bad = self.known('orphans', [f for f in subj if not named(f)], self.base.get('orphans', []),
                               lambda x: f'{x}: no other file names it: delete it, or name it where it is run (a suite, a gate '
                                         f'script, the doc that gives its command)')
        self.base['orphans'] = keep
        if not bad:
            print(f'PASS orphans: {len(subj)} tests, tools and benchmarks, no new one that nothing names'
                  f' ({len(keep)} known orphans, listed in {BASELINE})')

    def copies(self):
        t, pairs, bad = self.tree, [tuple(p) for p in self.base.get('copies', [])], 0
        for a, b in pairs:
            if a not in t.set or b not in t.set:
                bad += 1
                self.fail('copies', f'{BASELINE} declares {a} = {b}, and one of them is gone: drop the pair')
            elif t.data(a) != t.data(b):
                bad += 1
                self.fail('copies', f'{a} and {b} are declared copies and differ: make them equal, or delete one')
        declared = {frozenset(p) for p in pairs}
        by = collections.defaultdict(list)
        for f in t.files:
            if third_party(f) or f.startswith('tests/') or 'testdata' in f.split('/') or f.endswith('__init__.py'):
                continue
            d = t.data(f)
            if len(d) >= 512:
                by[hashlib.sha256(d).digest()].append(f)
        for same in by.values():
            for i, a in enumerate(same):
                for b in same[i + 1:]:
                    if frozenset((a, b)) not in declared:
                        bad += 1
                        self.fail('copies', f'{a} and {b} are the same bytes: delete one or make it a link (a pair that must '
                                            f'stay two files is declared under "copies" in {BASELINE})')
        if not bad:
            print(f'PASS copies: no two files are the same bytes ({len(pairs)} declared pairs, equal)')


def check(root, prune=False):
    tree = Tree(root)
    path = os.path.join(root, BASELINE)
    base = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            base = json.load(f)
    run = Run(tree, base, prune)
    for c in (run.refs, run.orphans, run.copies):
        c()
    if run.pruned:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(base, indent=1, sort_keys=True) + '\n')
        print(f'entropy: {run.pruned} entries pruned from {BASELINE}')
    print(f'entropy {root}: {len(tree.files)} files, {run.fails} failures')
    print('RESULT ' + ('PASS' if not run.fails else f'FAIL ({run.fails})'))
    return 1 if run.fails else 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prune', action='store_true', help='drop the baseline entries that no longer apply, then check')
    ap.add_argument('dir', nargs='?', default=ROOT)
    a = ap.parse_args(argv)
    return check(os.path.abspath(a.dir), a.prune)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
