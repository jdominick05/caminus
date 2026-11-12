"""tools/pr_ready.py against a stand-in `gh` written here (CPU, no network): each check's PASS and FAIL, and the bounded wait.

  python -I tests/tools/test_pr_ready.py --scratch <a new directory>

The stand-in answers `gh pr view`, `gh api graphql` and `gh repo view` from a scenario file: a list of pull request
views (one per call, the last repeated) and the review threads.
Cases: a clean pull request; the reviewer's receipt and its state comment as the verdict; a head pushed after the
review; a merge of `testing` after the review (refused, accepted with --allow-base-merges, and a later non-merge commit
still refused); open findings at the head; no review at all; an unresolved thread; a failed check, and a cancelled
duplicate beside its success; a kernel change whose body names no gate or no card; a decision id, a machine path and a
session link in the body or a commit message; the wrong base, a draft, a conflict; --wait while CI runs (passes once
it finishes), a bound shorter than one poll, no wait on a failure, a wait for a review that was asked for; no `gh`."""
import argparse
import json
import os
import stat
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
PR_READY = os.path.join(ROOT, 'tools', 'pr_ready.py')

ap = argparse.ArgumentParser(); ap.add_argument('--scratch', required=True); A = ap.parse_args()
S = os.path.join(A.scratch, 'pr_ready')
if os.path.exists(S):
    raise SystemExit(f'test_pr_ready: {S} exists (a new --scratch directory)')
os.makedirs(os.path.join(S, 'bin')); os.makedirs(os.path.join(S, 'empty'))
FAILS = []

GH = '''#!%s
import json, os, sys
path = os.environ['PR_READY_SCENARIO']; sc = json.load(open(path)); a = sys.argv[1:]
if a[:2] == ['pr', 'view']:
    n = int(open(path + '.n').read()) if os.path.exists(path + '.n') else 0
    open(path + '.n', 'w').write(str(n + 1))
    print(json.dumps(sc['views'][min(n, len(sc['views']) - 1)]))
elif a[:2] == ['api', 'graphql']:
    print(json.dumps({'data': {'repository': {'pullRequest': {'reviewThreads': {'nodes': sc['threads']}}}}}))
elif a[:2] == ['repo', 'view']:
    print(json.dumps({'nameWithOwner': 'o/r'}))
else:
    sys.exit(2)
''' % sys.executable
with open(os.path.join(S, 'bin', 'gh'), 'w') as f: f.write(GH)
os.chmod(os.path.join(S, 'bin', 'gh'), os.stat(os.path.join(S, 'bin', 'gh')).st_mode | stat.S_IXUSR)

BOT = 'chatgpt-codex-connector'
C1, C2, C3 = 'a1' * 20, 'b2' * 20, 'c3' * 20
N = [0]


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)


def commit(oid, head, body=''): return {'oid': oid, 'messageHeadline': head, 'messageBody': body}
def receipt(oid, clean=True, at=1):
    return {'author': {'login': BOT}, 'submittedAt': f'2026-01-01T00:00:{at:02d}Z',
            'body': f'**Reviewed commit:** `{oid[:10]}`\n' + ("Didn't find any major issues." if clean else 'Found one issue.')}
def state(oid, open_=False, at=2):
    st = {'code_review': {'commit_sha': oid, 'status': 'completed', 'has_unresolved_findings': open_}}
    return {'author': {'login': BOT}, 'createdAt': f'2026-01-01T00:00:{at:02d}Z', 'body': f'<!-- codex-review-state:v1 {json.dumps(st)} -->'}
def said(login, body, at=3): return {'author': {'login': login}, 'createdAt': f'2026-01-01T00:00:{at:02d}Z', 'body': body}
def ci(name='CPU suites', conclusion='SUCCESS', status='COMPLETED'): return {'name': name, 'status': status, 'conclusion': conclusion}
def thread(resolved): return {'isResolved': resolved, 'isOutdated': False, 'path': 'docs/x.md', 'comments': {'nodes': [{'author': {'login': BOT}, 'body': 'a finding'}]}}


def view(**kw):
    v = dict(baseRefName='testing', headRefOid=C1, state='OPEN', isDraft=False, mergeable='MERGEABLE', title='docs : a line', body='Docs only.',
             files=[{'path': 'docs/x.md'}], commits=[commit(C1, 'docs : a line')], comments=[], reviews=[receipt(C1)], statusCheckRollup=[ci()])
    v.update(kw)
    return v


def run(views, *args, threads=(), path=None, repo=True):
    N[0] += 1
    sc = os.path.join(S, f'scenario-{N[0]}.json')
    with open(sc, 'w') as f: json.dump({'views': views if isinstance(views, list) else [views], 'threads': list(threads)}, f)
    env = dict(os.environ, PR_READY_SCENARIO=sc, PATH=path if path is not None else os.path.join(S, 'bin') + os.pathsep + os.environ.get('PATH', ''))
    t = time.monotonic()
    r = subprocess.run([sys.executable, '-I', PR_READY, '7', *(['-R', 'o/r'] if repo else []), *args], capture_output=True, text=True, env=env)
    calls = int(open(sc + '.n').read()) if os.path.exists(sc + '.n') else 0
    return r.returncode, r.stdout + r.stderr, calls, time.monotonic() - t


def line(out, key): return next((l for l in out.splitlines() if l[5:].startswith(key + ':')), '')


# ---- the passing forms
rc, out, _, _ = run(view())
check(rc == 0 and out.strip().endswith('RESULT PASS') and out.count('PASS ') == 6, f'a clean pull request passes its six checks ({out.strip().splitlines()[-1]})')
rc, out, _, _ = run(view(), repo=False)
check(rc == 0, 'without -R the repository comes from `gh repo view`')
rc, out, _, _ = run(view(reviews=[], comments=[state(C1)]))
check(rc == 0 and line(out, 'review').startswith('PASS'), 'the reviewer\'s state comment at the head is a verdict too')
rc, out, _, _ = run(view(reviews=[receipt(C1, clean=False)], comments=[state(C1, open_=False)]), threads=[thread(True)])
check(rc == 0, 'a finding at the head that the state comment reports resolved, its thread resolved: ready')

# ---- review
two = dict(headRefOid=C2, commits=[commit(C1, 'docs : a line'), commit(C2, 'docs : another line')])
rc, out, _, _ = run(view(**two))
check(rc == 1 and f'last reviewed commit {C1[:10]}, head is {C2[:10]}' in line(out, 'review'), 'a head pushed after the review fails, naming both commits')
merged = dict(headRefOid=C2, commits=[commit(C1, 'docs : a line'), commit(C2, "Merge remote-tracking branch 'origin/testing' into docs/x")])
rc, out, _, _ = run(view(**merged))
check(rc == 1 and line(out, 'review').startswith('FAIL'), 'a merge of testing after the review is a new head: fails')
rc, out, _, _ = run(view(**merged), '--allow-base-merges')
check(rc == 0 and 'merges of testing after it accepted' in line(out, 'review'), '--allow-base-merges accepts it, and says so')
rc, out, _, _ = run(view(headRefOid=C3, commits=merged['commits'] + [commit(C3, 'docs : one more')]), '--allow-base-merges')
check(rc == 1 and line(out, 'review').startswith('FAIL'), '--allow-base-merges still refuses a later commit that is not such a merge')
rc, out, _, _ = run(view(reviews=[receipt(C1, clean=False)]))
check(rc == 1 and 'findings open' in line(out, 'review'), 'findings open at the head fail')
rc, out, _, _ = run(view(reviews=[]))
check(rc == 1 and 'has not reviewed' in line(out, 'review'), 'no review at all fails')
rc, out, _, _ = run(view(reviews=[{'author': {'login': 'someone'}, 'submittedAt': '2026-01-01T00:00:01Z', 'body': receipt(C1)['body']}]))
check(rc == 1, 'another account\'s receipt is not the reviewer\'s')
rc, out, _, _ = run(view(reviews=[{'author': {'login': 'someone'}, 'submittedAt': '2026-01-01T00:00:01Z', 'body': receipt(C1)['body']}]), '--reviewer', 'someone')
check(rc == 0, '--reviewer names another review account')

# ---- threads, ci
rc, out, _, _ = run(view(), threads=[thread(True), thread(False)])
check(rc == 1 and '2 review threads, 1 unresolved' in line(out, 'threads') and 'docs/x.md' in out, 'an unresolved thread fails and is listed with its file')
rc, out, _, _ = run(view(statusCheckRollup=[ci(), ci('Vendored code', 'FAILURE')]))
check(rc == 1 and "failed: ['Vendored code']" in line(out, 'ci'), 'a failed check fails, by name')
rc, out, _, _ = run(view(statusCheckRollup=[ci('CLA', 'CANCELLED'), ci('CLA', 'SUCCESS'), ci()]))
check(rc == 0, 'a cancelled run beside a success of the same check is superseded')
rc, out, _, _ = run(view(statusCheckRollup=[ci('CLA', 'CANCELLED'), ci()]))
check(rc == 1 and line(out, 'ci').startswith('FAIL'), 'a check that was only cancelled fails')
rc, out, _, _ = run(view(statusCheckRollup=[]))
check(rc == 1 and 'none reported' in line(out, 'ci'), 'no checks at all is not a pass')

# ---- gates named for kernel / tape changes
kern = dict(files=[{'path': 'kernels/k3_x.hip'}, {'path': 'docs/x.md'}])
rc, out, _, _ = run(view(**kern))
check(rc == 1 and 'no GPU gate named' in line(out, 'gates') and 'no card named' in line(out, 'gates'), 'a kernel change whose body names no gate and no card fails')
rc, out, _, _ = run(view(body='Gates: bitwise against the previous reference.', **kern))
check(rc == 1 and 'no card named' in line(out, 'gates') and 'no GPU gate' not in line(out, 'gates'), 'gates without a card still fail')
rc, out, _, _ = run(view(body='Gates: bitwise against the previous reference, XT then XTX.', files=[{'path': 'python/caminus/engine/k3.py'}]))
check(rc == 0 and line(out, 'gates').startswith('PASS'), 'a tape change that names its gates and cards passes')

# ---- private text (the strings are put together here so this file holds none)
did, home, link = 'D-' + '281', '/ho' + 'me/someone/notes', 'https://claude' + '.ai/code/session_x'
rc, out, _, _ = run(view(body=f'See {did}.'))
check(rc == 1 and 'decision id in body' in line(out, 'text'), 'a decision id in the body fails')
rc, out, _, _ = run(view(commits=[commit(C1, 'docs : a line', f'Measured in {home}.')]))
check(rc == 1 and f'machine path in {C1[:10]}' in line(out, 'text'), 'a machine path in a commit message fails, naming the commit')
rc, out, _, _ = run(view(title=f'docs : {link}'))
check(rc == 1 and 'session link in title' in line(out, 'text'), 'a session link in the title fails')

# ---- base
rc, out, _, _ = run(view(baseRefName='main'))
check(rc == 1 and 'base is main' in line(out, 'base'), 'a pull request into another branch fails')
rc, out, _, _ = run(view(isDraft=True, mergeable='CONFLICTING'))
check(rc == 1 and 'a draft' in line(out, 'base') and 'merge conflict' in line(out, 'base'), 'a draft with a conflict fails on both')
rc, out, _, _ = run(view(state='MERGED'))
check(rc == 0, 'a merged pull request can still be checked')

# ---- the bounded wait
busy = view(statusCheckRollup=[ci(status='IN_PROGRESS', conclusion='')])
rc, out, calls, dt = run([busy, busy, view()], '--wait', '20', '--poll', '1')
check(rc == 0 and calls == 3 and out.count('waiting on [\'ci\']') == 2 and dt < 15, f'--wait polls while CI runs and passes when it finishes ({calls} looks, {dt:.1f} s)')
rc, out, calls, dt = run(busy, '--wait', '3', '--poll', '1')
check(rc == 1 and 2 <= calls <= 4 and 'running: [\'CPU suites\']' in line(out, 'ci') and dt < 10, f'--wait stops at its bound and fails ({calls} looks, {dt:.1f} s)')
rc, out, calls, _ = run(busy, '--wait', '1', '--poll', '30')
check(rc == 1 and calls == 1, 'a bound shorter than one poll looks once')
rc, out, calls, _ = run(busy, '--wait', '20', '--poll', '1', threads=[thread(False)])
check(rc == 1 and calls == 1 and 'waiting' not in out, '--wait does not wait when something has already failed')
rc, out, calls, _ = run(busy)
check(rc == 1 and calls == 1, 'without --wait a running check fails at once')
asked = view(comments=[said('author', '@codex review')], **two)
rc, out, calls, _ = run([asked, view(reviews=[receipt(C1), receipt(C2, at=5)], **two)], '--wait', '20', '--poll', '1')
check(rc == 0 and calls == 2 and 'waiting on [\'review\']' in out, 'after a review was asked for, --wait waits for the reviewer')
rc, out, calls, _ = run(view(**two), '--wait', '20', '--poll', '1')
check(rc == 1 and calls == 1, 'a head nobody asked a review for is a failure, not a wait')

# ---- no gh
rc, out, _, _ = run(view(), path=os.path.join(S, 'empty'))
check(rc != 0 and 'gh CLI is not on PATH' in out and 'Traceback' not in out, 'without gh on PATH: one line saying so')

print('RESULT', 'FAIL' if FAILS else 'PASS')
sys.exit(1 if FAILS else 0)
