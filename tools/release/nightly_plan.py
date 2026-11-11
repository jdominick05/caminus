"""The nightly build's decisions, with no build in them (docs/dev/releasing.md, "Nightly builds"): tools/release/nightly.sh
asks which version to build and whether a commit already has its nightly, and it and .github/workflows/nightly.yml check a
nightly tag before anything is published from it. CPU only, no network; git reads the repository given by --repo (default:
this checkout; a bare repository works).

  nightly_plan.py version --commit C --date YYYYMMDD   the version of commit C's nightly: X.Y.Z.devYYYYMMDD, where X.Y.Z is
                                                       C's __version__ when no tag vX.Y.Z exists yet, else its next patch
                                                       (a dev version sorts before its release and after the last one)
  nightly_plan.py built --base C                       the nightly tag whose commit's only parent is C: prints it, or
                                                       prints nothing and exits 3 when C has no nightly
  nightly_plan.py check-tag vX.Y.Z.devYYYYMMDD (--base C | --branch REF)
                                                       exit 1 with every failure printed unless the tag is a nightly of
                                                       reviewed code: its commit has one parent, the parent is C (--base)
                                                       or an ancestor of REF (--branch), the commit changes only
                                                       python/caminus/__init__.py (its __version__ line, to the tag's
                                                       version) and packaging/gfx1100-kernels.sha256 (hash lines only),
                                                       and the version is a dev version of the parent's own or next patch

A nightly commit sits on no branch: it is the branch head plus the version and the code-object hashes of that night's
build, reachable only from its tag, so the files built at the tag verify the way a release's do
(tools/release/verify_release.py)."""
import argparse
import datetime
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INIT = 'python/caminus/__init__.py'
HASHES = 'packaging/gfx1100-kernels.sha256'
REL_RE = re.compile(r'v(\d+)\.(\d+)\.(\d+)')
DEV_RE = re.compile(r'v((\d+)\.(\d+)\.(\d+))\.dev(\d{8})')
VERSION_LINE = re.compile(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]\s*$")
HASH_LINE = re.compile(r'[0-9a-f]{64}  [A-Za-z0-9_]+\.hsaco')


class PlanError(Exception):
    pass


def git(repo, *a, check=True):
    r = subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True)
    if check and r.returncode:
        raise PlanError(f'git {" ".join(a)}: {r.stderr.strip() or r.stdout.strip()}')
    return r


def tags(repo):
    return git(repo, 'tag', '-l', 'v*').stdout.split()


def file_at(repo, commit, path):
    return git(repo, 'show', f'{commit}:{path}').stdout


def version_of(text):
    """the __version__ a python/caminus/__init__.py text holds, or None"""
    for line in text.splitlines():
        m = VERSION_LINE.match(line)
        if m:
            return m.group(1)
    return None


def next_patch(v):
    a, b, c = (int(x) for x in v.split('.'))
    return f'{a}.{b}.{c + 1}'


def valid_date(d):
    try:
        return bool(re.fullmatch(r'\d{8}', d)) and bool(datetime.date(int(d[:4]), int(d[4:6]), int(d[6:])))
    except ValueError:
        return False


def dev_version(tree_version, all_tags, date):
    """X.Y.Z.devDATE for a tree whose __version__ is tree_version, given the repository's tags"""
    if not re.fullmatch(r'\d+\.\d+\.\d+', tree_version or ''):
        raise PlanError(f'the tree\'s __version__ is {tree_version!r}, not X.Y.Z')
    if not valid_date(date):
        raise PlanError(f'{date!r} is not a date YYYYMMDD')
    base = next_patch(tree_version) if f'v{tree_version}' in all_tags else tree_version
    key = tuple(int(x) for x in base.split('.'))
    newer = sorted(t for t in all_tags if REL_RE.fullmatch(t) and tuple(int(x) for x in REL_RE.fullmatch(t).groups()) >= key)
    if newer:
        raise PlanError(f'release tag(s) {newer} are not older than {base}: the tree\'s __version__ ({tree_version}) is behind them')
    return f'{base}.dev{date}'


def parents(repo, commit):
    return git(repo, 'rev-list', '--parents', '-n', '1', f'{commit}^{{commit}}').stdout.split()[1:]


def built(repo, base):
    """the nightly tag whose commit's only parent is base, or None"""
    base = git(repo, 'rev-parse', '--verify', '--quiet', f'{base}^{{commit}}').stdout.strip()
    for t in sorted(tags(repo), reverse=True):
        if DEV_RE.fullmatch(t) and parents(repo, f'refs/tags/{t}') == [base]:
            return t
    return None


def check_tag(repo, tag, base=None, branch=None):
    """the reasons the tag is not a nightly of reviewed code ([] = it is one)"""
    m = DEV_RE.fullmatch(tag)
    if not m:
        return [f'{tag!r} is not a nightly tag (vX.Y.Z.devYYYYMMDD)']
    ver, date = tag[1:], m.group(5)
    fails = []
    if not valid_date(date):
        fails.append(f'{date} is not a date')
    ref = f'refs/tags/{tag}'
    if git(repo, 'rev-parse', '--verify', '--quiet', f'{ref}^{{commit}}', check=False).returncode:
        return fails + [f'no tag {tag} here']
    ps = parents(repo, ref)
    if len(ps) != 1:
        return fails + [f'the tag\'s commit has {len(ps)} parents, not one']
    par = ps[0]
    if base is not None:
        want = git(repo, 'rev-parse', '--verify', '--quiet', f'{base}^{{commit}}', check=False).stdout.strip()
        if par != want:
            fails.append(f'the tag\'s parent is {par[:12]}, not {base[:12]}')
    if branch is not None and git(repo, 'merge-base', '--is-ancestor', par, branch, check=False).returncode:
        fails.append(f'the tag\'s parent {par[:12]} is not on {branch}')
    changed = sorted(git(repo, 'diff', '--name-only', '--no-renames', par, ref).stdout.split('\n')[:-1])
    extra = [p for p in changed if p not in (INIT, HASHES)]
    if extra:
        fails.append(f'the tag\'s commit changes more than the version and the hashes: {extra}')
    if INIT not in changed:
        fails.append(f'the tag\'s commit does not change {INIT}')
    old, new = file_at(repo, par, INIT).splitlines(), file_at(repo, ref, INIT).splitlines()
    diff = [(a, b) for a, b in zip(old, new) if a != b]
    if len(old) != len(new) or len(diff) != 1 or not VERSION_LINE.match(diff[0][0]) or version_of(diff[0][1]) != ver:
        fails.append(f'{INIT} differs from the parent\'s in more than its __version__ line set to {ver}')
    pv = version_of('\n'.join(old))
    if not re.fullmatch(r'\d+\.\d+\.\d+', pv or '') or m.group(1) not in (pv, next_patch(pv)):
        fails.append(f'{ver} is not a dev version of the parent\'s {pv} or of its next patch')
    bad = [line for line in file_at(repo, ref, HASHES).splitlines() if not HASH_LINE.fullmatch(line)]
    if bad:
        fails.append(f'{HASHES} holds a line that is not "<sha256>  <name>.hsaco": {bad[0][:80]!r}')
    return fails


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', default=ROOT)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('version'); p.add_argument('--commit', default='HEAD'); p.add_argument('--date', required=True)
    p = sub.add_parser('built'); p.add_argument('--base', required=True)
    p = sub.add_parser('check-tag'); p.add_argument('tag')
    g = p.add_mutually_exclusive_group(required=True); g.add_argument('--base'); g.add_argument('--branch')
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'version':
            print(dev_version(version_of(file_at(a.repo, a.commit, INIT)), tags(a.repo), a.date))
            return 0
        if a.cmd == 'built':
            t = built(a.repo, a.base)
            if t:
                print(t)
            return 0 if t else 3
        fails = check_tag(a.repo, a.tag, base=a.base, branch=a.branch)
        for f in fails:
            print(f'FAIL {f}')
        print(f'nightly_plan check-tag {a.tag}:', 'FAIL' if fails else 'PASS')
        return 1 if fails else 0
    except PlanError as e:
        print(f'nightly_plan: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
