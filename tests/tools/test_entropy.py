"""tools/entropy.py on small trees made here (CPU, no git), then on this tree: each check's PASS and FAIL, the baseline
and --prune; last, that the repository itself passes (its FAIL lines are printed as the tool wrote them).

  python -I tests/tools/test_entropy.py --scratch <a new directory>

Cases: a clean tree (names that match the end of a path, a pattern, a brace list, a fenced command; a module, a
placeholder, a project's name and a skipped doc are not read as paths); a doc naming a missing path under a top-level
directory, as a source name in a code span and in a fenced block; the baseline accepting them by name; a listed entry
that no longer applies, and a listed doc that is gone, failing until --prune drops them; --prune adding nothing; a test,
a tool and a benchmark nothing names (naming itself does not count, a short stem needs the whole file name, a
run_all.py directory is discovered); two files with the same bytes, a declared pair, a declared pair that differs, and
what is not a copy (test data, a link, a small file)."""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
ENTROPY = os.path.join(ROOT, 'tools', 'entropy.py')

ap = argparse.ArgumentParser(); ap.add_argument('--scratch', required=True); A = ap.parse_args()
S = os.path.join(A.scratch, 'entropy')
if os.path.exists(S):
    raise SystemExit(f'test_entropy: {S} exists (a new --scratch directory)')
os.makedirs(S)
FAILS, N = [], [0]


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)


def tree(files, base=None):
    """a new tree under the scratch directory from {path: text}, with a baseline when one is given"""
    N[0] += 1
    d = os.path.join(S, f'tree-{N[0]}')
    for p, text in dict(files, **({'tools/entropy.json': json.dumps(base)} if base is not None else {})).items():
        os.makedirs(os.path.dirname(os.path.join(d, p)), exist_ok=True)
        with open(os.path.join(d, p), 'w') as f: f.write(text)
    return d


def run(d, *args):
    r = subprocess.run([sys.executable, '-I', ENTROPY, *args, d], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def fails(out, key): return [l for l in out.splitlines() if l.startswith(f'FAIL {key}:')]
def listed(d): return json.load(open(os.path.join(d, 'tools', 'entropy.json')))


GOOD = {
    'README.md': 'See `engine/k3.py`, python/pkg/engine/k3.py, tests/*.py, `k3_{a,b}.hip` and docs/a.md.\n'
                 '`pkg.device.hip` is a module, `DIR/out.py` a placeholder, llama.cpp and `llama.cpp` a project, `notes/x.md` yours.\n'
                 '```\n./build.sh && python -I tools/run.py\n```\n',
    'CHANGELOG.md': '0.1 removed tools/long_gone.py.\n',
    'build.sh': '', 'docs/a.md': 'plain\n', 'python/pkg/engine/k3.py': '', 'kernels/k3_a.hip': '', 'kernels/k3_b.hip': '',
    'tools/run.py': '# runs tests/test_a.py\n', 'tests/test_a.py': '', 'tests/serve/run_all.py': '', 'tests/serve/test_found.py': '',
}

# ---- a clean tree
rc, out = run(tree(GOOD))
check(rc == 0 and 'RESULT PASS' in out and out.count('\nPASS ') + out.startswith('PASS ') == 3, 'a clean tree passes its three checks')

# ---- refs
dead = dict(GOOD, **{'README.md': GOOD['README.md'] + 'Also `engine/gone.py` and docs/none.md.\n```\n./missing.sh\n```\n'})
rc, out = run(tree(dead))
got = fails(out, 'refs')
check(rc == 1 and len(got) == 3 and 'README.md:6: names engine/gone.py' in out and 'names docs/none.md' in out
      and 'README.md:8: names missing.sh' in out, 'a missing path fails by doc and line: under a top-level directory, in a code span, in a fenced block')
known = {'refs': {'README.md': ['docs/none.md', 'engine/gone.py', 'missing.sh']}}
rc, out = run(tree(dead, known))
check(rc == 0 and '3 known dead' in out, 'the baseline accepts the dead names it lists')
stale = {'refs': {'README.md': ['engine/gone.py'], 'gone.md': ['x.py']}, 'orphans': ['tools/old.py']}
d = tree(GOOD, stale)
rc, out = run(d)
check(rc == 1 and len(fails(out, 'refs')) == 2 and len(fails(out, 'orphans')) == 1 and out.count('--prune') == 3,
      'a listed entry that no longer applies fails, a listed doc that is gone too')
rc, out = run(d, '--prune')
check(rc == 0 and '3 entries pruned' in out and listed(d) == {'orphans': [], 'refs': {}}, '--prune drops them and the tree passes')
d = tree(dead, {'refs': {'README.md': ['engine/gone.py', 'engine/fixed.py']}})
rc, out = run(d, '--prune')
check(rc == 1 and len(fails(out, 'refs')) == 2 and listed(d)['refs'] == {'README.md': ['engine/gone.py']},
      '--prune adds nothing: the unlisted dead names still fail')

# ---- orphans
lone = dict(GOOD, **{'tools/lonely.py': '# python -I tools/lonely.py\n', 'tests/test_b.py': '', 'benchmarks/probe.hip': '',
                     'tools/tui.py': '', 'docs/a.md': 'the tui and the probe\n'})
rc, out = run(tree(lone))
check(rc == 1 and sorted(l.split()[2].rstrip(':') for l in fails(out, 'orphans')) ==
      ['benchmarks/probe.hip', 'tests/test_b.py', 'tools/lonely.py', 'tools/tui.py'] and 'delete it' in out,
      'a test, a tool and a benchmark nothing names fail (naming itself does not count; a short stem is not a name)')
named = dict(lone, **{'docs/a.md': 'tools/lonely.py, tui.py, test_b and probe.hip\n'})
rc, out = run(tree(named))
check(rc == 0, 'each passes once another file names it')
rc, out = run(tree(lone, {'orphans': ['benchmarks/probe.hip', 'tests/test_b.py', 'tools/lonely.py', 'tools/tui.py']}))
check(rc == 0 and '4 known orphans' in out, 'the baseline accepts the orphans it lists')

# ---- copies
body = 'the same rules\n' * 40
two = dict(GOOD, **{'packaging/rules.md': body, 'python/pkg/share/rules.md': body, 'tests/data/a.txt': body, 'panel/testdata/b.txt': body,
                    'docs/small-1.md': 'x\n', 'docs/small-2.md': 'x\n'})
d = tree(two)
os.symlink('../python/pkg/share/rules.md', os.path.join(d, 'tools', 'rules.md'))
rc, out = run(d)
check(rc == 1 and len(fails(out, 'copies')) == 1 and 'packaging/rules.md and python/pkg/share/rules.md are the same bytes' in out
      and 'delete one or make it a link' in out, 'two files with the same bytes fail; test data, a link and small files are not copies')
pair = {'copies': [['packaging/rules.md', 'python/pkg/share/rules.md']]}
rc, out = run(tree(two, pair))
check(rc == 0 and '1 declared pairs' in out, 'a declared pair passes while it is equal')
rc, out = run(tree(dict(two, **{'packaging/rules.md': body + 'and one more\n'}), pair))
check(rc == 1 and 'are declared copies and differ' in out, 'a declared pair that differs fails')

# ---- this tree
rc, out = run(ROOT)
print('\n'.join(l for l in out.splitlines() if l.startswith('FAIL ')), end='\n' if rc else '', flush=True)
check(rc == 0, 'this tree: ' + next((l for l in out.splitlines() if l.startswith('entropy ')), out.strip()[-200:]))

print('RESULT PASS' if not FAILS else f'RESULT FAIL ({len(FAILS)})')
sys.exit(0 if not FAILS else 1)
