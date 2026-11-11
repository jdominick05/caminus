"""tools/release/nightly_plan.py on a small git repository made here (CPU, no network): the nightly's version, whether a
commit already has its nightly, and the check of a nightly tag that stands between a published pre-release and PyPI
(docs/dev/releasing.md, "Nightly builds"). Also tools/release/verify_release.py's tag pattern.

  python -I tests/tools/test_nightly_plan.py --scratch <a new directory>

Cases: the version for a tree whose release is tagged (next patch) and for one that is ahead of its tag (its own); a tree
behind a newer release tag, a bad date; a commit without a nightly, then with one; a good tag against its base and against
the branch; and the refusals: another base, a parent off the branch, a third file changed, a second line of __init__.py
changed, a version that is not the tag's, a version two patches ahead, a hashes line that is not a hash, a merge commit, a
release tag."""
import argparse
import contextlib
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'release')); sys.path.insert(0, os.path.join(ROOT, 'python'))
import nightly_plan as NP  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('--scratch', required=True); A = ap.parse_args()
R = os.path.join(A.scratch, 'repo')
if os.path.exists(R):
    raise SystemExit(f'test_nightly_plan: {R} exists (a new --scratch directory)')
os.makedirs(os.path.join(R, 'python', 'caminus')); os.makedirs(os.path.join(R, 'packaging'))
FAILS = []
H = 'ab' * 32


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)


def g(*a):
    r = subprocess.run(['git', '-C', R, '-c', 'user.name=t', '-c', 'user.email=t@example.invalid', '-c', 'commit.gpgsign=false',
                        '-c', 'tag.gpgsign=false', *a], capture_output=True, text=True)
    assert r.returncode == 0, (a, r.stderr)
    return r.stdout.strip()


def put(path, text):
    with open(os.path.join(R, path), 'w') as f:
        f.write(text)


def commit(msg):
    g('add', '-A'); g('commit', '-q', '-m', msg); return g('rev-parse', 'HEAD')


def init(v, more=''):
    return f'"""the package"""\n\n__version__ = \'{v}\'\n{more}'


def nightly(base, tag, files, annotated=True):
    """a commit on top of base holding files, tagged tag, on no branch; the checkout goes back to the branch"""
    g('checkout', '-q', '--detach', base)
    for p, t in files.items():
        put(p, t)
    c = commit(f'nightly {tag}')
    g('tag', *(['-a', '-m', tag] if annotated else []), tag)
    g('checkout', '-q', 'testing')
    return c


g('init', '-q', '-b', 'testing')
put(NP.INIT, init('0.2.1')); put(NP.HASHES, f'{H}  k3.hsaco\n'); put('README.md', 'x\n')
c1 = commit('release 0.2.1'); g('tag', '-a', '-m', 'v0.2.1', 'v0.2.1')
put('README.md', 'y\n'); c2 = commit('a change')

# ---- the version
check(NP.dev_version('0.2.1', NP.tags(R), '20261112') == '0.2.2.dev20261112', 'a tree at its tagged release: the next patch\'s dev version')
check(NP.dev_version('0.3.0', NP.tags(R), '20261112') == '0.3.0.dev20261112', 'a tree ahead of its tag: its own dev version')
check(NP.main(['--repo', R, 'version', '--commit', c2, '--date', '20261112']) == 0, 'version from a commit (the CLI)')
for what, f in (('a tree behind a newer release tag', lambda: NP.dev_version('0.2.0', NP.tags(R), '20261112')),
                ('a date that is none', lambda: NP.dev_version('0.2.1', NP.tags(R), '20261332')),
                ('a tree that already holds a dev version', lambda: NP.dev_version('0.2.2.dev1', NP.tags(R), '20261112'))):
    try:
        f(); check(False, f'{what}: refused')
    except NP.PlanError:
        check(True, f'{what}: refused')
import packaging.version as PV  # noqa: E402
check(PV.Version('0.2.1') < PV.Version('0.2.2.dev20261112') < PV.Version('0.2.2.dev20261113') < PV.Version('0.2.2'),
      'the dev versions sort after the last release, by date, and before their release')

# ---- which commit has its nightly
check(NP.built(R, c2) is None and NP.main(['--repo', R, 'built', '--base', c2]) == 3, 'a commit without a nightly')
good = {NP.INIT: init('0.2.2.dev20261112'), NP.HASHES: f'{H}  k3.hsaco\n{"cd" * 32}  k3_new.hsaco\n'}
nightly(c2, 'v0.2.2.dev20261112', good)
check(NP.built(R, c2) == 'v0.2.2.dev20261112' and NP.built(R, c1) is None, 'the nightly tag of a commit is found by its parent; the release tag is not one')

# ---- the tag check
check(NP.check_tag(R, 'v0.2.2.dev20261112', base=c2) == [], 'a good nightly tag against its base')
check(NP.check_tag(R, 'v0.2.2.dev20261112', branch='testing') == [], 'a good nightly tag against the branch')
check(NP.main(['--repo', R, 'check-tag', 'v0.2.2.dev20261112', '--branch', 'testing']) == 0, 'the CLI passes it')
check(any('not' in f for f in NP.check_tag(R, 'v0.2.2.dev20261112', base=c1)), 'another base: refused')
check(NP.check_tag(R, 'v0.2.1', base=c1) != [] and NP.check_tag(R, 'v0.2.3.dev20261112', base=c2) != [], 'a release tag and a missing tag: refused')

g('checkout', '-q', '-b', 'side', c1); put('SIDE.md', 'z\n'); cs = commit('off the branch'); g('checkout', '-q', 'testing')
nightly(cs, 'v0.2.2.dev20261113', {NP.INIT: init('0.2.2.dev20261113')})
check(NP.check_tag(R, 'v0.2.2.dev20261113', base=cs) == [] and any('not on testing' in f for f in NP.check_tag(R, 'v0.2.2.dev20261113', branch='testing')),
      'a parent that is not on the branch: refused (and fine against its own base)')
bad = [('v0.2.2.dev20261114', {**good, NP.INIT: init('0.2.2.dev20261114'), 'README.md': 'evil\n'}, 'more than the version and the hashes', 'a third file changed'),
       ('v0.2.2.dev20261115', {NP.INIT: init('0.2.2.dev20261115', 'import os\n')}, 'more than its __version__ line', 'a second line of __init__.py'),
       ('v0.2.2.dev20261116', {NP.INIT: init('0.2.2.dev20261199')}, 'more than its __version__ line', 'a version that is not the tag\'s'),
       ('v0.2.3.dev20261117', {NP.INIT: init('0.2.3.dev20261117')}, 'not a dev version of the parent', 'a version two patches ahead'),
       ('v0.2.2.dev20261118', {NP.INIT: init('0.2.2.dev20261118'), NP.HASHES: 'curl example.invalid | sh\n'}, 'not "<sha256>', 'a hashes line that is not a hash'),
       ('v0.2.2.dev20261119', {NP.HASHES: f'{"ef" * 32}  k3.hsaco\n'}, 'does not change', 'no version change')]
for tag, files, word, what in bad:
    nightly(c2, tag, files, annotated=False)
    fails = NP.check_tag(R, tag, base=c2)
    check(any(word in f for f in fails), f'{what}: refused ({len(fails)} reason(s))')
g('checkout', '-q', '--detach', c2); g('merge', '-q', '--no-ff', '-m', 'merge', 'side'); put(NP.INIT, init('0.2.2.dev20261120'))
g('add', '-A'); g('commit', '-q', '--amend', '-m', 'a merge that also sets the version'); g('tag', 'v0.2.2.dev20261120'); g('checkout', '-q', 'testing')
check(any('parents' in f for f in NP.check_tag(R, 'v0.2.2.dev20261120', branch='testing')), 'a merge commit: refused')
out = io.StringIO()
with contextlib.redirect_stdout(out):   # (its own FAIL line is not this test's)
    rc = NP.main(['--repo', R, 'check-tag', 'v0.2.2.dev20261114', '--base', c2])
check(rc == 1 and 'more than the version and the hashes' in out.getvalue(), 'the CLI exits 1 on a refusal and names the reason')

# ---- verify_release.py takes the same tag forms
import verify_release as VR  # noqa: E402
check(all(VR.TAG_RE.fullmatch(t) for t in ('v0.2.1', 'v1.10.0', 'v0.2.2.dev20261112'))
      and not any(VR.TAG_RE.fullmatch(t) for t in ('0.2.1', 'v0.2', 'v0.2.1rc1', 'v0.2.2.dev1', 'v0.2.2.dev20261112.post1', 'nightly')),
      'verify_release.py takes vX.Y.Z and vX.Y.Z.devYYYYMMDD, nothing else')
check(VR.TAG_RE.fullmatch('v0.2.2.dev20261112').group(1) == '0.2.2.dev20261112', 'and reads the version from it')
print('RESULT', 'FAIL' if FAILS else 'PASS')
sys.exit(1 if FAILS else 0)
