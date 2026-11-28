"""CPU test of tools/emberlab (the Ember lab): scripted brains on the local host, no model, no GPU, no SSH.

  python -I tests/tools/test_emberlab.py --scratch DIR

Checks:
  suite      the basic suite's scripted solver passes every check through the real resident and its gates (a harness
             change that breaks a solvable task shows here); the claiming script never passes a check, and each of its
             finished goals is counted as a false completion
  resume     a second `run` into the same folder runs nothing again
  compare    two arms over the same runs give "no difference"; the across-the-board rule on made-up cells
  session    a driven lab session: the request the model would see waits for us, our tool call reaches the resident,
             the timings a driven:caminus brain sends are the resident's engine speed, an interrupt closes the waiting call (the driven brain records the hang-up),
             close stops the daemon and the job workers it leaves behind (KILL after the grace, named), never a
             process a stale state file names
  tui        (with tmux) keys in, an exact screen and its SVG out; --read-only refuses a goal before anything runs
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LAB = [sys.executable, '-I', str(ROOT / 'tools/emberlab/lab.py')]
TUI = [sys.executable, '-I', str(ROOT / 'tools/emberlab/tui.py')]
ap = argparse.ArgumentParser(); ap.add_argument('--scratch', required=True)
ap.add_argument('--prepare-only', action='store_true',
                help='prepare --scratch and exit (the checks below test this clearing rule with it)')
a = ap.parse_args()
S = Path(a.scratch).resolve()
MARK = S / '.emberlab-test-scratch'


def _force(p):
    """Remove one entry, even a read-only directory a killed run left behind (the 0o555 checks).

    Bottom-up by hand rather than shutil.rmtree: rmtree's fd-based walk hands its error hook a bare
    name, which cannot be chmod'ed and retried from there, and a 0o555 parent refuses its children's
    unlinks. Setting the directory writable first, then removing what it holds, always terminates.
    """
    import stat
    if p.is_symlink() or not p.is_dir():
        try:
            p.unlink()
        except PermissionError:
            p.chmod(stat.S_IRWXU)
            p.unlink()
        return
    p.chmod(stat.S_IRWXU)
    for c in p.iterdir():
        _force(c)
    p.rmdir()


def prepare_scratch():
    """A directory this test may clear: empty, or one it marked itself on an earlier run.

    The checks below drive real runs into S; a previous run leaves results, plan.md digests and a
    scratch home behind, and the next run would read them as its own (a stale digest, a KeyError on a
    record it never wrote). So a marked scratch is cleared first. A directory holding anyone else's
    files is REFUSED and left exactly as it is: --scratch can be a typo, and that typo must not delete.
    """
    if S.exists() and not S.is_dir():
        sys.exit(f'refusing --scratch {S}: it exists and is not a directory')
    if S.is_dir() and not MARK.exists() and any(S.iterdir()):
        sys.exit(f'refusing --scratch {S}: it is not empty and carries no {MARK.name} marker.\n'
                 f'That marker means "a previous run of this test made this directory and it may be '
                 f'cleared". Pass a fresh or empty directory, or delete {S} yourself.')
    S.mkdir(parents=True, exist_ok=True)
    for p in list(S.iterdir()):
        if p.name != MARK.name:
            _force(p)
    if MARK.exists():
        MARK.chmod(0o644)
    MARK.write_text('this directory is a scratch of tests/tools/test_emberlab.py; a later run clears it\n')
    print(f'scratch {S} ready (cleared; marked {MARK.name})', flush=True)


prepare_scratch()
if a.prepare_only:
    sys.exit(0)
FAILS = []
sys.path.insert(0, str(ROOT / 'tools/emberlab'))
import report  # noqa: E402


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok:
        FAILS.append(what)


def lab(*a, timeout=600):
    return subprocess.run(LAB + list(a), capture_output=True, text=True, timeout=timeout)


# ---- suite
out = S / 'run'
p = lab('run', '--suite', 'basic', '--brain', 'scripted:solve', '--brain', 'scripted:claim', '--out', str(out))
rows = report.load(out)
solve = [r for r in rows if r['brain'] == 'scripted:solve']
claim = [r for r in rows if r['brain'] == 'scripted:claim']
check(p.returncode == 0 and len(rows) == 10, f'suite: 10 runs recorded, exit 0 ({p.returncode}; {p.stderr[-300:]})')
check(len(solve) == 5 and all(r['check'] and r['finished'] and not r['false_completion'] for r in solve),
      'suite: the scripted solver passes all five checks through the gates: '
      + ', '.join(f"{r['task']}={r.get('state')}/{r.get('check')}" for r in solve))
check(len(claim) == 5 and not any(r['check'] for r in claim) and all(r['false_completion'] == r['finished'] for r in claim)
      and any(r['false_completion'] for r in claim),   # (vacuous if no claim goal ever finished)
      'suite: the claiming script passes no check; every goal it finished is a false completion')
check(all((out / 'runs' / r['id'] / f).exists() for r in rows for f in ('events.jsonl', 'check.out', 'run.json')),
      'suite: each run keeps its events, check output and record')
check(all(r.get('commit') and 'dirty' in r for r in rows), 'suite: each record names the runtime commit and dirty flag')

# ---- resume
p = lab('run', '--suite', 'basic', '--brain', 'scripted:solve', '--out', str(out))
check('5 runs (0 to do)' in p.stdout and len((out / 'results.jsonl').read_text().splitlines()) == 10,
      'resume: a second run into the same folder runs nothing again')
res = out / 'results.jsonl'
lines = res.read_text().splitlines()
first = json.loads(lines[0]); first.update(state='error', error='ssh dropped')
res.write_text('\n'.join([json.dumps(first)] + lines[1:]) + '\n')
p = lab('run', '--suite', 'basic', '--brain', 'scripted:' + first['brain'].split(':')[-1], '--out', str(out))
check('(1 to do)' in p.stdout and {r['id']: r for r in report.load(out)}[first['id']].get('state') != 'error',
      f'resume: a run that ended in error runs again ({p.stdout[-200:]})')

# ---- compare
arms = S / 'arms.json'
arms.write_text(json.dumps({'base': {}, 'b': {'config': {'max_tokens': 2048}}}))
out2 = S / 'ab'
p = lab('run', '--suite', 'basic', '--brain', 'scripted:solve', '--arms', str(arms), '--only', 'edit,read', '--out', str(out2))
c = report.compare(report.load(out2), 'base', 'b')
check(p.returncode == 0 and c['verdict'].startswith('no difference'), f'compare: same runs on two arms: {c["verdict"]}')
mk = lambda brain, arm, n, ok, false=0: [dict(id=f'{brain}{arm}{i}', brain=brain, arm=arm, repeat=i, task='t', check=i < ok,
                                             finished=i < ok + false, false_completion=ok <= i < ok + false) for i in range(n)]
v = lambda rows: report.compare(rows, 'A', 'B')['verdict']
check(v(mk('x', 'A', 3, 1) + mk('x', 'B', 3, 2) + mk('y', 'A', 3, 2) + mk('y', 'B', 3, 2)).startswith('better across the board'),
      'compare: one brain up, none down = better across the board')
check(v(mk('x', 'A', 3, 1) + mk('x', 'B', 3, 3) + mk('y', 'A', 3, 2) + mk('y', 'B', 3, 1)).startswith('worse for y'),
      'compare: a brain that drops makes the arm worse, whatever the others gain')
check(v(mk('x', 'A', 3, 2) + mk('x', 'B', 3, 2, false=1)).startswith('worse for x'),
      'compare: more false completions at the same pass rate is worse')
errored = mk('x', 'B', 3, 2)
errored[2].update(state='error')                # an infrastructure failure, not a failed check of the harness
c2 = report.compare(mk('x', 'A', 3, 2) + errored, 'A', 'B')
check(c2['brains'][0]['arm']['runs'] == 2 and c2['errored'] == ['xB2'] and 'left out' in report.compare_md(c2),
      f'compare: a run that ended in error is left out and named, not a failed check ({c2["verdict"]})')

# ---- session (driven: we are the model)
so = S / 'sessions'
p = lab('open', 'd', '--brain', 'driven:caminus', '--host', 'local', '--fixture', 'basic', '--out', str(so), '--ttl', '300')
check(p.returncode == 0, f'session: open ({p.stderr[-300:]})')
bad = lab('say', 'd', 'a steer for no goal', '--goal', 'zzzz', '--out', str(so))
check(bad.returncode != 0 and 'matches 0 goals' in (bad.stderr + bad.stdout) and 'sent (' not in bad.stdout,
      'session: say --goal with an unknown short id is refused, not reported as sent')
lab('say', 'd', 'List the files in your work folder.', '--out', str(so))
t = lab('turn', 'd', '--out', str(so), '--wait', '20')
check(t.returncode == 0 and 'terminal' in t.stdout and 'List the files' in t.stdout,
      'session: the waiting request shows the tools offered and our goal')
r = lab('reply', 'd', '--out', str(so), '--call', 'read', '--args', json.dumps({'path': 'work/inventory.py'}), '--settle', '3', '--width', '4000',
        '--timings', json.dumps({'prompt_n': 1488, 'prompt_per_second': 1240.2, 'predicted_n': 500, 'predicted_per_second': 62.4}))
check('read' in r.stdout and 'total_value' in r.stdout, 'session: our tool call ran in the resident and its result came back')
s = json.loads((so / 'sessions/d/session.json').read_text())
tp = Path(s['home']) / 'throughput.json'
tp = json.loads(tp.read_text()) if tp.exists() else {}
check((tp.get('source'), tp.get('tps'), tp.get('prompt_tps')) == ('engine', 62.4, 1240.2),
      f'session: a driven:caminus brain plays a caminus serve, and the resident takes its timings as the engine speed ({tp})')
# an Esc while the next request waits: the resident closes the call, and the driven brain records the hang-up
t = lab('turn', 'd', '--out', str(so), '--wait', '20')
BOOT = "import sys;sys.path.insert(0,sys.argv[1]);from caminus.__main__ import main;sys.argv=sys.argv[1:];sys.exit(main())"
agent = [sys.executable, '-I', '-c', BOOT, str(ROOT / 'python'), 'ember', 'agent', '--local']
st = subprocess.run(agent + ['status', 'lab', '--home', s['home'], '--json'], capture_output=True, text=True, timeout=60)
gid = json.loads(st.stdout)['goals'][-1]['id'] if st.returncode == 0 else ''
it = subprocess.run(agent + ['interrupt', 'lab', '--home', s['home'], gid, '--json'], capture_output=True, text=True, timeout=60)
gone = []
for _ in range(50):
    gone = sorted(Path(s['driven']).glob('*.gone.json'))
    if gone:
        break
    time.sleep(0.2)
t2 = lab('turn', 'd', '--out', str(so), '--wait', '1')
check(t.returncode == 0 and it.returncode == 0 and '"queued": true' in it.stdout and gone and 'hung up' in t2.stdout,
      f'session: an interrupt closes the waiting call and the driven brain records the hang-up ({it.stdout.strip()} {t2.stdout.strip()[:200]})')
# close: box and terminal workers outlive their daemon by design, so close ends them too. Fake jobs in the session's
# home: a worker that ignores TERM (close must KILL it), one that obeys, their commands (each its own group, as the
# resident starts them), and a stale state.json naming a process that is not the session's (a recycled pid: left alone).
procs = []


def spawn(code, *args, env=None):
    p = subprocess.Popen([sys.executable, '-I', '-c', code, *args], start_new_session=True, env=env)
    procs.append(p)
    return p.pid


def gone(pid):
    p = next((p for p in procs if p.pid == pid), None)
    if p is not None:
        return p.poll() is not None
    try:
        return open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()[0] in 'ZX'
    except OSError:
        return True


IGNORE_TERM = 'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(120)'
SLEEP = 'import time;time.sleep(120)'
home = Path(s['home'])
stubborn, polite, stale = (home / 'jobs' / (c * 32) for c in 'abc')
command_env = {**os.environ, 'WORKSPACE': str(home)}
jobs = {stubborn: (spawn(IGNORE_TERM, str(stubborn)), spawn(SLEEP, env=command_env)),
        polite: (spawn(SLEEP, str(polite)), spawn(SLEEP, env=command_env)),
        stale: (spawn(SLEEP), None)}
for job, (worker, cmd) in jobs.items():
    job.mkdir(parents=True)
    (job / 'state.json').write_text(json.dumps(dict(phase='running', worker=worker, pid=cmd)))
# and one real terminal job, started the way the resident starts it (its own worker, command and paths)
sys.path.insert(0, str(ROOT / 'python'))
from caminus.ember.resident.terminal import Terminal  # noqa: E402
real = Terminal(home).call('d' * 32, dict(command='sleep 120', yield_ms=500))
daemon = json.loads((home / 'heartbeat.json').read_text())['pid']
time.sleep(0.5)
t0 = time.monotonic()
c = lab('close', 'd', '--out', str(so), timeout=120)
took = time.monotonic() - t0
time.sleep(0.5)
after = c.stdout.split('had to SIGKILL')[-1]
check(real.get('running') and c.returncode == 0
      and all(gone(p) for p in [daemon, *jobs[stubborn], *jobs[polite], real['worker'], real['pid']]),
      f'session: close ends the daemon, the job workers and their commands ({c.stdout.strip()} {c.stderr[-300:]})')
check('had to SIGKILL' in c.stdout and f'worker {"a" * 8} ({jobs[stubborn][0]})' in after
      and f'worker {"b" * 8}' not in after and took < 60,
      f'session: close says what it had to SIGKILL after the grace, and only that ({took:.0f} s)')
check(not gone(jobs[stale][0]), "session: close leaves alone a process a stale state.json names that is not the session's")
check((so / 'sessions/d/session.closed.json').exists(), 'session: close files the session as closed')
try:
    os.killpg(s['daemon_pid'], 0)
    alive = True
except ProcessLookupError:
    alive = False
check(not alive, 'session: close stops the daemon')
for p in procs:                 # the test's own fake processes: the decoy, and anything a failed close left
    if p.poll() is None:
        p.kill()
    p.wait()
for pid in (real.get('worker'), real.get('pid')):
    try:
        os.killpg(pid, 9) if pid and not gone(pid) else None
    except OSError:
        pass

# ---- propose
ph = S / 'simhome'
ph.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'tools/emberlab'))
import proposal  # noqa: E402
import runner  # noqa: E402


def sim_record(home, **fields):
    """A `proposed` record in a scratch home: the reflection's output before anyone has measured it."""
    sim, store, rt = proposal._open(home)
    try:
        return sim.record(rt, 'g-propose', dict(dict(problem_group='propose test', change='turn sim off in the arm',
                                                     predicted='no difference on one task', status='proposed'),
                                                **fields),
                          cfg=dict(model='m', provider='p'))
    finally:
        store.close()


r1 = sim_record(ph)
po = S / 'propose'
p = lab('propose', '--sim-home', str(ph), '--record', r1['id'], '--brain', 'scripted:solve',
        '--config', '{"sim": false}', '--only', 'read', '--out', str(po), '--dry-run')
check(p.returncode == 0 and f"proposal {r1['id']}" in p.stdout and 'runs: 6' in p.stdout and not po.exists()
      and '- host: local' in p.stdout and '- timeout: 1500s per goal' in p.stdout,
      f'propose: --dry-run prints the plan (2 arms x 3 repeats), the host and the timeout, and creates nothing '
      f'({p.returncode}; {p.stdout[-300:]})')
p = lab('propose', '--sim-home', str(ph), '--record', r1['id'], '--brain', 'anthropic:8091#claude-haiku-5-5',
        '--config', '{"sim": false}', '--out', str(po))
check(p.returncode != 0 and '--approve-paid' in (p.stderr + p.stdout) and not po.exists(),
      f'propose: a brain billed through the relay does not run without --approve-paid ({p.returncode}; '
      f'{(p.stderr + p.stdout)[-300:]})')
p = lab('propose', '--sim-home', str(ph), '--record', r1['id'], '--brain', 'anthropic:8091#claude-haiku-5-5',
        '--host', 'local', '--approve-paid', '--config', '{"sim": false}', '--out', str(po))
check(p.returncode != 0 and '--unsafe-local' in (p.stderr + p.stdout) and not po.exists(),
      f'propose: a paid real brain on the local host needs --unsafe-local too, like lab.py run ({p.returncode}; '
      f'{(p.stderr + p.stdout)[-300:]})')
# the flag that unlocks the local host is one name only: `--unsafe-local`. The alias this branch once carried
# (`--unsafe-free`) is gone, so `run --help` must not list it and `propose` must not accept it (#385 P1-1)
for cmd in ('run', 'propose'):
    h = lab(cmd, '--help')
    check('--unsafe-free' not in h.stdout and '--unsafe-free' not in h.stderr,
          f'{cmd}: --unsafe-free is not a flag (the local-host permission has one name, --unsafe-local)')
p = lab('propose', '--sim-home', str(ph), '--record', r1['id'], '--brain', 'scripted:solve', '--out', str(po),
        '--unsafe-free')
check(p.returncode != 0 and 'unrecognized arguments' in (p.stderr + p.stdout) and not po.exists(),
      f'propose: --unsafe-free is refused as an unknown flag, not silently honoured ({p.returncode}; '
      f'{(p.stderr + p.stdout)[-200:]})')
p = lab('propose', '--sim-home', str(ph), '--record', r1['id'], '--brain', 'scripted:solve', '--out', str(po))
check(p.returncode != 0 and 'measure nothing' in (p.stderr + p.stdout),
      'propose: two identical arms are refused (the proposal names no change)')
# only a `proposed` record is the input: a `recorded` one's change is already in the base tree, so its A/B would
# compare that change with itself (and with the override), not judge the prediction (#385 review P2)
r0 = sim_record(ph, status='recorded')
p = lab('propose', '--sim-home', str(ph), '--record', r0['id'], '--brain', 'scripted:solve',
        '--config', '{"sim": false}', '--only', 'read', '--out', str(S / 'propose-recorded'))
check(p.returncode != 0 and 'not proposed' in (p.stderr + p.stdout)
      and not (S / 'propose-recorded' / 'results.jsonl').exists()
      and proposal.load_record(ph, r0['id'])['status'] == 'recorded',
      f'propose: a recorded (not proposed) record is refused before anything runs ({p.returncode}; '
      f'{(p.stderr + p.stdout)[-300:]})')
p = lab('propose', '--sim-home', str(ph), '--record', r1['id'], '--brain', 'scripted:solve',
        '--config', '{"sim": false}', '--only', 'read', '--out', str(po))
closed = proposal.load_record(ph, r1['id'])
arm_home = [json.loads((po / 'runs' / r['id'] / 'home' / 'config.json').read_text())
            for r in report.load(po) if r['arm'] == 'arm']
base_home = [json.loads((po / 'runs' / r['id'] / 'home' / 'config.json').read_text())
             for r in report.load(po) if r['arm'] == 'base']
check(p.returncode == 0 and closed['status'] == 'observed' and 'no difference' in closed['observed']
      and 'scripted:solve 3/3 -> 3/3' in closed['observed'],
      f'propose: the runs close the record with the per-brain numbers ({p.returncode}; '
      f'{closed["observed"][:200]}; {p.stderr[-200:]})')
# `--only` narrows the task set, and the observation must say which tasks ran: an `eval_context` that says only
# "suite basic" is a verdict over an unknown part of the suite (#385 review P2)
check('tasks 1/5 (read)' in (closed.get('eval_context') or ''),
      f'propose: the observation names the tasks that ran ({closed.get("eval_context", "")[-300:]})')
# one shape for the arms: `arms.json` and the run's `plan.json` are both `Arm.describe()` (#385 review P2, Growth).
# plan.json drops the env VALUES and keeps their names and a digest: `--env` carries relay keys and home paths, and
# plan.json is a plain file beside the rows, so the values belong only in arms.json, which `load_arms` runs from
# (#385 review P2).
spec = json.loads((po / 'arms.json').read_text())
plan_j = json.loads((po / 'plan.json').read_text())
_stripped = {x['name']: {k: v for k, v in x.items() if k not in ('env', 'env_keys', 'env_sha')} for x in plan_j['arms']}
check(_stripped == {n: {k: v for k, v in a.items() if k != 'env'} for n, a in spec['arms'].items()}
      and all('env' not in x and 'env_keys' in x and 'env_sha' in x for x in plan_j['arms'])
      and spec['plan']['suite'] == 'basic'
      and spec['plan']['brains'] == ['scripted:solve'] and spec['plan']['only'] == ['read']
      and 'approvals' in spec['plan'] and plan_j.get('approvals') is None,
      f'propose: arms.json is plan.json\'s arms and the plan that reads them ({sorted(spec.get("arms", {}))}, '
      f'{spec.get("plan")})')
# the env values stay out of plan.json, and the digest still names them exactly: a reader with arms.json can check
# that plan.json describes the same env without plan.json ever holding the value (#385 review P2)
se = S / 'sim-env'
sim, store, rt = proposal._open(se)
renv = sim.record(rt, 'g-propose', dict(change='c', predicted='p', status='proposed'))
store.close()
# (its own variable: the plan.md check below reads `p.stdout`, and reassigning `p` here compared another command)
pe = S / 'propose-env'
penv = lab('propose', '--sim-home', str(se), '--record', renv['id'], '--brain', 'scripted:solve',
           '--config', '{"sim": false}', '--env', '{"SIM_TOKEN": "s3cret-value"}', '--only', 'read',
           '--out', str(pe), '--dry-run')
plain = (pe / 'plan.json').read_text() if (pe / 'plan.json').exists() else ''
check(penv.returncode == 0 and 's3cret-value' not in penv.stdout,
      f'propose: --env values never reach the printed plan ({penv.returncode}; {penv.stdout[-200:]})')
check(not (pe / 'plan.json').exists() and plain == '',
      'propose: --dry-run writes no plan.json at all')
# the printed plan is the owner's record of what they approved; arms.json and plan.json hold the machine identity,
# so a reader of the folder should not have to reconstruct the human-facing text (#385 review P2). The file is
# written after the claim, and its text is the text that was printed, digest and all.
check((po / 'plan.md').exists() and (po / 'plan.md').read_text() in p.stdout
      and f"proposal {r1['id']}" in (po / 'plan.md').read_text()
      and spec['plan']['suite_sha'][:12] in (po / 'plan.md').read_text(),
      f'propose: the printed plan is written into the folder, digest included ({p.stdout[:200]!r}; '
      f'{(po / "plan.md").read_text()[:200] if (po / "plan.md").exists() else "no plan.md"!r})')
# `propose` starts its runs from `lab.py run`'s own parser defaults, not a hand-built field list: a flag added to the
# run parser is then present with its own default, so a per-flag copy cannot go stale (#385 P2).
# A field `cmd_run` reads that the defaults lack is the same drift, so read cmd_run's own source and compare.
import inspect  # noqa: E402
import lab as lab_mod  # noqa: E402
_defaults = vars(lab_mod.run_defaults())
_reads = set(re.findall(r'(?<![\w.])a\.([a-z_]+)', inspect.getsource(lab_mod.cmd_run)))
check(not (_reads - set(_defaults)) and {'suite', 'brain', 'arms', 'out', 'repeats', 'host', 'unsafe_local',
                                         'jobs', 'parallel', 'timeout', 'rerun', 'only'} <= set(_defaults),
      f'propose: run_defaults answers every field cmd_run reads, so a propose-started run cannot miss a new flag '
      f'({sorted(_reads - set(_defaults))}; {sorted(_defaults)})')
check('--unsafe-local' in lab('run', '--help').stdout and 'required' not in lab('run', '--help').stderr,
      'propose: `run` still parses its own flags from the shared list')
check(closed['observed_by']['model'] == 'emberlab' and closed['brain']['model'] == 'm',
      'propose: the observation names emberlab, the predictor keeps the brain that made the prediction')
check(arm_home and base_home and all(c.get('sim') is False for c in arm_home) and all('sim' not in c for c in base_home),
      'propose: the arm run home has sim=false and the base run home does not (the proposal is the only difference)')
p = lab('propose', '--sim-home', str(ph), '--record', r1['id'], '--brain', 'scripted:solve',
        '--config', '{"sim": false}', '--only', 'read', '--out', str(po))
check(p.returncode != 0 and 'already observed' in (p.stderr + p.stdout),
      'propose: a closed record is not measured again')
p = lab('propose', '--sim-home', str(ph), '--record', 'nostore', '--brain', 'scripted:solve',
        '--config', '{"sim": false}', '--out', str(po))
check(p.returncode != 0 and 'no SIM record' in (p.stderr + p.stdout),
      'propose: a record the log does not hold is refused before anything runs')
# `--goal` names the goal the observation lives in, and an observation always lands in the record's own goal
# (`sim.observe` passes `prior.get('goal') or gid`): an id that is not that goal was accepted and then silently
# ignored, so the help promised a move the log cannot make (#385 review P2). A different goal is refused before a
# run; the record's own id, or a prefix of it as `say --goal` resolves one, is accepted.
r_goal = sim_record(ph, status='proposed')
wrong_goal = S / 'propose-wrong-goal'
p = lab('propose', '--sim-home', str(ph), '--record', r_goal['id'], '--brain', 'scripted:solve',
        '--config', '{"sim": false}', '--only', 'read', '--goal', 'g-other', '--out', str(wrong_goal))
check(p.returncode != 0 and 'lives in goal' in (p.stderr + p.stdout) and 'not g-other' in (p.stderr + p.stdout)
      and 'g-other' not in proposal.load_record(ph, r_goal['id'])['goal']
      and not (wrong_goal / 'results.jsonl').exists()
      and proposal.load_record(ph, r_goal['id'])['status'] == 'proposed',
      f"propose: a --goal other than the record's own is refused before anything runs ({p.returncode}; "
      f'{(p.stderr + p.stdout)[-300:]})')
own_goal = S / 'propose-own-goal'
p = lab('propose', '--sim-home', str(ph), '--record', r_goal['id'], '--brain', 'scripted:solve',
        '--config', '{"sim": false}', '--only', 'read', '--goal', 'g-prop', '--out', str(own_goal), '--dry-run')
check(p.returncode == 0 and f"proposal {r_goal['id']}" in p.stdout and not own_goal.exists(),
      f"propose: the record's own goal (a short id, as `say --goal` resolves one) is accepted ({p.returncode}; "
      f'{(p.stderr + p.stdout)[-300:]})')
# a second experiment in the same --out would reuse the first one's rows by id (task.brain.arm.rN)
shared = S / 'propose-shared'
r2 = sim_record(ph, status='proposed')
p = lab('propose', '--sim-home', str(ph), '--record', r2['id'], '--brain', 'scripted:solve',
        '--config', '{"sim": false}', '--only', 'read', '--out', str(shared))
check(p.returncode == 0 and proposal.load_record(ph, r2['id'])['status'] == 'observed',
      f'propose: the first experiment in a folder closes its record ({p.returncode}; {p.stderr[-200:]})')
r3 = sim_record(ph, status='proposed')
p = lab('propose', '--sim-home', str(ph), '--record', r3['id'], '--brain', 'scripted:solve',
        '--config', '{"sim": true}', '--only', 'read', '--out', str(shared))
check(p.returncode != 0 and 'fresh --out' in (p.stderr + p.stdout)
      and proposal.load_record(ph, r3['id'])['status'] != 'observed',
      f'propose: a shared --out with different arms is refused and the record stays open ({p.returncode}; '
      f'{(p.stderr + p.stdout)[-300:]})')
r4 = sim_record(ph, status='proposed')
p = lab('propose', '--sim-home', str(ph), '--record', r4['id'], '--brain', 'scripted:solve',
        '--config', '{"sim": false}', '--only', 'read', '--out', str(shared))
check(p.returncode == 0 and '0 to do' in p.stdout
      and proposal.load_record(ph, r4['id'])['status'] == 'observed',
      f'propose: the same arms resume the folder and close the record ({p.returncode}; '
      f'{(p.stdout + p.stderr)[-300:]})')
# the same arms with another --brain (or suite, --only, repeats) is a different experiment: without this the record
# would be closed with rows this command never chose (#385 review P1)
r7 = sim_record(ph, status='proposed')
p = lab('propose', '--sim-home', str(ph), '--record', r7['id'], '--brain', 'scripted:claim',
        '--config', '{"sim": false}', '--only', 'read', '--out', str(shared))
check(p.returncode != 0 and 'fresh --out' in (p.stderr + p.stdout) and 'brains' in (p.stderr + p.stdout)
      and proposal.load_record(ph, r7['id'])['status'] != 'observed',
      f'propose: a shared --out with another brain is refused and the record stays open ({p.returncode}; '
      f'{(p.stderr + p.stdout)[-300:]})')
# a verdict of `incomplete` (a brain with no run of one arm) says nothing about the proposal, and observe is
# permanent: the record stays open (#385 review P2)
inc = S / 'propose-incomplete'
r8 = sim_record(ph, status='proposed')
ns = argparse.Namespace(sim_home=str(ph), record=r8['id'], suite='basic', brain=['scripted:solve'], repeats=1,
                        only='read', host=None, jobs=1, parallel=None, out=str(inc),
                        config='{"sim": false}', env=None, base_config=None, base_env=None,
                        arm_checkout=None, base_checkout=None, dry_run=False, approve_paid=False,
                        unsafe_local=False, prediction_error=None, goal=None)


def half_run(runs):
    """`lab.py run` with only the base arm done: the arm rows are missing, so the verdict is incomplete."""
    (inc / 'results.jsonl').write_text(json.dumps(dict(id='read.scripted-solve.base.r1', brain='scripted:solve',
                                                       arm='base', task='read', repeat=1, check=True)) + '\n')
    return 0


try:
    proposal.cmd_propose(ns, half_run)
    err = ''
except SystemExit as e:
    err = str(e)
check('incomplete' in err and proposal.load_record(ph, r8['id'])['status'] == 'proposed',
      f'propose: an incomplete verdict leaves the record open ({err[:200]})')
# `incomplete` only asks that every brain has both arms. A rerun into a folder whose first pass errored on half its
# runs can hold ONE valid row where the plan asked for two, and report.compare's "fewer than 3 repeats" is advisory:
# the record would close (permanent) on a 1-sample A/B (#385 review P2)
part = S / 'propose-partial'
r13 = sim_record(ph, status='proposed')
nsp = argparse.Namespace(sim_home=str(ph), record=r13['id'], suite='basic', brain=['scripted:solve'], repeats=2,
                         only='read', host=None, jobs=1, parallel=None, out=str(part),
                         config='{"sim": false}', env=None, base_config=None, base_env=None,
                         arm_checkout=None, base_checkout=None, dry_run=False, approve_paid=False,
                         unsafe_local=False, prediction_error=None, goal=None)


def part_run(runs):
    """`lab.py run` after a first pass errored out: both arms have rows, but arm r2 has none."""
    rows = [dict(id='read.scripted-solve.base.r1', brain='scripted:solve', arm='base', task='read', repeat=1, check=True),
            dict(id='read.scripted-solve.base.r2', brain='scripted:solve', arm='base', task='read', repeat=2, check=True),
            dict(id='read.scripted-solve.arm.r1', brain='scripted:solve', arm='arm', task='read', repeat=1, check=True)]
    (part / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return 0


try:
    proposal.cmd_propose(nsp, part_run)
    err13 = 'closed instead of refusing'
except SystemExit as e:
    err13 = str(e)
check('planned runs have no valid row' in err13 and proposal.load_record(ph, r13['id'])['status'] == 'proposed',
      f'propose: a cell short of the plan\'s repeats leaves the record open ({err13[:200]})')
# the digest arms.json claims is the digest the rows must have used, and `lab.py run` re-reads the suite when it runs:
# a fixture edited while the A/B ran leaves the folder naming bytes nothing measured, and the record would close on a
# mix of both. The re-read after the runs keeps the record open (#385 review P1, the same family as suite_sha).
mid = S / 'suite-mid'
shutil.copytree(runner.load_suite('basic')['dir'], mid)
rmid = sim_record(ph, status='proposed')
nsm = argparse.Namespace(sim_home=str(ph), record=rmid['id'], suite=str(mid), brain=['scripted:solve'], repeats=1,
                         only='read', host=None, jobs=1, parallel=None, out=str(S / 'propose-mid'),
                         config='{"sim": false}', env=None, base_config=None, base_env=None,
                         arm_checkout=None, base_checkout=None, dry_run=False, approve_paid=False,
                         unsafe_local=False, prediction_error=None, goal=None)


def mid_run(runs):
    """`lab.py run` while the suite is edited under it: the rows are complete, but for other bytes."""
    rows = [dict(id='read.scripted-solve.base.r1', brain='scripted:solve', arm='base', task='read', repeat=1, check=True),
            dict(id='read.scripted-solve.arm.r1', brain='scripted:solve', arm='arm', task='read', repeat=1, check=True)]
    (S / 'propose-mid').mkdir(parents=True, exist_ok=True)
    ((S / 'propose-mid') / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (mid / 'fixture' / 'inventory.py').write_text((mid / 'fixture' / 'inventory.py').read_text() + '\n# mid-run\n')
    return 0


try:
    proposal.cmd_propose(nsm, mid_run)
    errm = 'closed instead of refusing'
except SystemExit as e:
    errm = str(e)
check('changed while the arms ran' in errm and proposal.load_record(ph, rmid['id'])['status'] == 'proposed',
      f'propose: a suite edited while the arms ran leaves the record open ({errm[:200]})')
# an ARM checkout that moves while the runs execute (a commit, a rebase, an edit of the dirty tree) measured other
# bytes than arms.json names: the same re-read for the arms keeps the record open (#385 review P1). The in-memory
# Arm cannot answer this - it read commit/tree once, before the runs - so the check reads the checkout again and
# compares it with the folder's own record.
co_base, co_arm = S / 'drift-base', S / 'drift-arm'
for d in (co_base, co_arm):
    (d / 'python/caminus/ember/resident').mkdir(parents=True, exist_ok=True)
    (d / 'python/caminus/ember/resident/cli.py').write_text('print(1)\n')
    (d / 'python/other.py').write_text('X = 1\n')
rdrift = sim_record(ph, status='proposed')
nsd = argparse.Namespace(sim_home=str(ph), record=rdrift['id'], suite='basic', brain=['scripted:solve'], repeats=1,
                         only='read', host=None, jobs=1, parallel=None, out=str(S / 'propose-drift'),
                         config=None, env='{"A": "1"}', base_config=None, base_env=None,
                         arm_checkout=str(co_arm), base_checkout=str(co_base), dry_run=False, approve_paid=False,
                         unsafe_local=False, prediction_error=None, goal=None)


def drift_run(runs):
    """`lab.py run` while the arm checkout is edited under it: the rows are complete, but for other bytes."""
    rows = [dict(id='read.scripted-solve.base.r1', brain='scripted:solve', arm='base', task='read', repeat=1, check=True),
            dict(id='read.scripted-solve.arm.r1', brain='scripted:solve', arm='arm', task='read', repeat=1, check=True)]
    (S / 'propose-drift').mkdir(parents=True, exist_ok=True)
    ((S / 'propose-drift') / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (co_arm / 'python/other.py').write_text('X = 2\n')
    return 0


try:
    proposal.cmd_propose(nsd, drift_run)
    err_d = 'closed instead of refusing'
except SystemExit as e:
    err_d = str(e)
check('the arms changed while they ran' in err_d and 'tree_sha' in err_d
      and proposal.load_record(ph, rdrift['id'])['status'] == 'proposed',
      f'propose: an arm checkout edited while the runs ran leaves the record open ({err_d[:200]})')
# and the same function says nothing about an arm that did not move, so the guard cannot turn every run into a refusal
fresh_arms = proposal.arms_from_proposal(dict(env={'A': '1'}), str(co_base), str(co_arm))
check(proposal.arms_drift(fresh_arms, proposal.arms_spec(fresh_arms)) == '',
      'propose: an arm that did not move during its run is not drift, so the guard refuses only what moved')
# the runs read their config/env from the folder's arms.json (`run --arms`), not from the Arm object this command
# holds: editing that file while the runs execute measures other bytes than the folder's identity, and the record
# would close on a mix of both. So the re-read compares config/env too, prints digests (an `--env` may carry a relay
# key), and a file that is gone or no longer names an arm refuses rather than passing (#385 review P1).
import threading  # noqa: E402
fdir = S / 'arms-drift-file'
fdir.mkdir(parents=True, exist_ok=True)
farms = proposal.arms_from_proposal(dict(env={'A': '1'}), str(co_base), str(co_arm))
fspec = proposal.arms_spec(farms)
afile = fdir / 'arms.json'
afile.write_text(json.dumps(fspec))
check(proposal.arms_drift(farms, fspec, arms_file=afile) == '',
      'propose: an arms.json that still says what the folder claimed is not drift')
ed = json.loads(json.dumps(fspec))
ed['arms']['arm']['env'] = {'A': '2'}
afile.write_text(json.dumps(ed))
msg = proposal.arms_drift(farms, fspec, arms_file=afile)
check('env' in msg and 'claimed' in msg and 'A' not in msg,
      f'propose: an arms.json env edited during the runs is drift, and the message digests the values ({msg[:160]})')
ed = json.loads(json.dumps(fspec))
del ed['arms']['arm']
afile.write_text(json.dumps(ed))
msg = proposal.arms_drift(farms, fspec, arms_file=afile)
check('no longer names it' in msg,
      f'propose: an arms.json that dropped an arm towards the end of the runs refuses ({msg[:160]})')
afile.unlink()
msg = proposal.arms_drift(farms, fspec, arms_file=afile)
check('cannot be read back' in msg,
      f'propose: an arms.json gone during the runs refuses instead of passing the checkout check ({msg[:160]})')
# on a filesystem without hard links the create fallback makes the name and fills it after, so a loser of that
# race can read an empty file: the read is retried, so a whole spec that lands within the window compares as
# usual instead of a refusal saying 'is not JSON' (#385 review P2)
race = S / 'claim-race'
race.mkdir(parents=True, exist_ok=True)
rfile = race / 'arms.json'
rfile.write_text('')
rspec = proposal.arms_spec(farms, dict(suite='basic', suite_sha='x', brains=[], only=[], repeats=1, approvals=[]))


def whole_later():
    time.sleep(0.12)
    rfile.write_text(json.dumps(rspec))


th = threading.Thread(target=whole_later)
th.start()
res = proposal.claim_folder(rfile, rspec)
th.join()
check(res == '', f'propose: a loser of the create race reads the whole arms.json, not "is not JSON" ({res[:160]})')
junk = race / 'junk.json'
junk.write_text('not json at all')
res = proposal.claim_folder(junk, rspec)
check('is not JSON and was not written by this command' in res,
      f'propose: a file that never becomes JSON is still refused, named as not this command\'s ({res[:160]})')
# a checkout that MOVED (a commit, a rebase) is a different arm even at the same path: `lab.py run` skips every id
# already in results.jsonl, so without this a new record would be closed with the previous rows (#385 review P1)
moved = json.loads((shared / 'arms.json').read_text())
moved['arms']['arm']['commit'] = 'deadbeef0000'
(shared / 'arms.json').write_text(json.dumps(moved))
r5 = sim_record(ph, status='proposed')
p = lab('propose', '--sim-home', str(ph), '--record', r5['id'], '--brain', 'scripted:solve',
        '--config', '{"sim": false}', '--only', 'read', '--out', str(shared))
check(p.returncode != 0 and 'fresh --out' in (p.stderr + p.stdout) and 'commit' in (p.stderr + p.stdout)
      and proposal.load_record(ph, r5['id'])['status'] != 'observed',
      f'propose: a folder whose checkout moved is refused and the record stays open ({p.returncode}; '
      f'{(p.stderr + p.stdout)[-300:]})')
# a record closed while the arms ran is not observed a second time: the resident's own `_record` refuses it inside
# the same transaction as the append (verified: with the unfixed code the refusal surfaced as a ValueError
# traceback), so the command reports it as a refusal and keeps the runs (#385 review P2)
r9 = sim_record(ph, status='proposed')
sim, store, rt = proposal._open(ph)
try:
    sim.observe(rt, 'g-propose', r9['id'], 'closed by the resident while the arms ran')
finally:
    store.close()
try:
    proposal.write_observation(ph, r9['id'], 'the lab measured it')
    err9 = ''
except (SystemExit, ValueError) as e:
    err9 = str(e)
newest = proposal.load_record(ph, r9['id'])
check('not closed' in err9 and 'already observed' in err9 and newest['status'] == 'observed'
      and newest['observed'] == 'closed by the resident while the arms ran',
      f'propose: a record closed during the runs is not observed twice, and says so ({err9[:200]}; '
      f'{newest.get("observed")!r})')
# a read must create nothing: `Store(home)` made the home, its schema and its journal mode on every load_record, so
# `propose --dry-run` could touch the daemon's live database, and a mistyped --sim-home was created empty and then
# reported "no SIM record" (#385 review P2). One cleanly closed store has no -wal/-shm (a live writer holds both),
# so the read is the file alone and even a read-only folder works.
quiet = S / 'sim-readonly'
sim, store, rt = proposal._open(quiet)
rq = sim.record(rt, 'g-propose', dict(change='c', predicted='p', status='proposed'))
store.close()
before = {p.name: p.stat().st_mtime_ns for p in quiet.iterdir()}
got = proposal.load_record(quiet, rq['id'])
after = {p.name: p.stat().st_mtime_ns for p in quiet.iterdir()}
check(got and got['id'] == rq['id'] and before == after and sorted(after) == ['agent.sqlite3'],
      f'propose: a read leaves the home byte-identical (no -wal/-shm, no migration; {sorted(before)} -> '
      f'{sorted(after)})')
quiet.chmod(0o555)
try:
    ro = proposal.load_record(quiet, rq['id'])
finally:
    quiet.chmod(0o755)
check(ro and ro['id'] == rq['id'],
      'propose: a read works on a read-only home (immutable read: no lock file, no journal, no schema write)')
empty = S / 'sim-no-store'
empty.mkdir(parents=True, exist_ok=True)
try:
    proposal.load_record(empty, 'x')
    err_empty = ''
except SystemExit as e:
    err_empty = str(e)
check('no session store' in err_empty and not list(empty.iterdir()),
      f'propose: a --sim-home with no agent.sqlite3 is refused and not created ({err_empty[:120]!r})')
# the observation is appended through `sim.tool`, the one path a resident's own /sim takes, so a home whose log is
# off refuses it instead of accepting what its own tool would reject (#385 review P2)
off = S / 'sim-off'
sim, store, rt = proposal._open(off)
roff = sim.record(rt, 'g-propose', dict(change='c', predicted='p', status='proposed'))
store.close()
(off / 'config.json').write_text(json.dumps({'sim': False}))
try:
    proposal.write_observation(off, roff['id'], 'the lab measured it')
    err_off = ''
except SystemExit as e:
    err_off = str(e)
check('SIM log is off' in err_off and proposal.load_record(off, roff['id'])['status'] == 'proposed',
      f'propose: a home with sim=false refuses the observation, as its own /sim would ({err_off[:160]!r})')
# a store that cannot be read (not a database, a lock held past the busy timeout) is a refusal like every other in this
# module, not a traceback: `sim._get` catches OperationalError on its first query and falls back to `_load`, which
# re-runs it unguarded, so a locked store escaped from `load_record` as a sqlite3 traceback (#385 review P2)
broken = S / 'sim-broken'
broken.mkdir(parents=True, exist_ok=True)
(broken / 'agent.sqlite3').write_text('this is not a database\n')
try:
    proposal.load_record(broken, 'x')
    err_broken = 'returned instead of refusing'
except SystemExit as e:
    err_broken = str(e)
except Exception as e:                      # noqa: BLE001 - the traceback this check exists to prevent
    err_broken = f'{type(e).__name__}: {e}'
check('could not be read' in err_broken and str(broken) in err_broken and 'nothing was changed' in err_broken,
      f'propose: an unreadable SIM store is refused with the path, not a traceback ({err_broken[:160]!r})')
p = lab('propose', '--sim-home', str(broken), '--record', 'x', '--brain', 'scripted:solve',
        '--config', '{"sim": false}', '--only', 'read', '--out', str(S / 'propose-broken'), '--dry-run')
check(p.returncode != 0 and 'Traceback' not in p.stderr and 'could not be read' in (p.stderr + p.stdout),
      f'propose: the command line exits on an unreadable store with one message, no traceback ({p.returncode}; '
      f'{(p.stderr + p.stdout)[-200:]})')
# `--dry-run` promises "print the plan and create nothing", so it asks no endpoint either: `resolve_all` issues a
# network GET /models, and an unreachable endpoint made the dry run fail instead of printing the plan (#385 review P2).
# A real run still resolves before any identity is built (the check above on resolve_all pins that).
r_off = sim_record(ph)
unreach = S / 'propose-unreachable'
p = lab('propose', '--sim-home', str(ph), '--record', r_off['id'], '--brain', 'openai:http://127.0.0.1:1/v1',
        '--config', '{"sim": false}', '--only', 'read', '--out', str(unreach), '--dry-run', timeout=120)
check(p.returncode == 0 and f"proposal {r_off['id']}" in p.stdout and '?@http://127.0.0.1:1/v1' in p.stdout
      and 'asked for its model when the run starts' in p.stdout and not unreach.exists(),
      f'propose: --dry-run prints the plan without asking an unreachable endpoint ({p.returncode}; '
      f'{(p.stderr + p.stdout)[-300:]})')
p = lab('propose', '--sim-home', str(ph), '--record', r_off['id'], '--brain', 'openai:http://127.0.0.1:1/v1',
        '--config', '{"sim": false}', '--only', 'read', '--out', str(unreach), timeout=120)
check(p.returncode != 0 and 'lists no model' in (p.stderr + p.stdout) and not unreach.exists(),
      f'propose: a real run still asks the endpoint and refuses when it cannot name a model ({p.returncode}; '
      f'{(p.stderr + p.stdout)[-300:]})')
# the folder's identity is created once, exclusively: a concurrent command that finds arms.json compares instead of
# rewriting it, so the last writer cannot win over the first (#385 review P2)
cf = S / 'claim'
cf.mkdir(parents=True, exist_ok=True)
spec_a = dict(arms={'base': dict(checkout='a', commit='1', dirty=False, config={}, env={}),
                    'arm': dict(checkout='b', commit='2', dirty=False, config={}, env={})},
              plan=dict(suite='basic', brains=['scripted:solve'], only=[], repeats=1))
spec_b = json.loads(json.dumps(spec_a))
spec_b['arms']['arm']['commit'] = '3'
why = proposal.claim_folder(cf / 'arms.json', spec_a)
(cf / 'arms.json').write_text(json.dumps(spec_a, indent=2))     # as another writer would have left it
first = (cf / 'arms.json').read_text()
why2 = proposal.claim_folder(cf / 'arms.json', spec_b)
why3 = proposal.claim_folder(cf / 'arms.json', spec_a)
check(why == '' and why2 and 'commit' in why2 and why3 == '' and (cf / 'arms.json').read_text() == first,
      f'propose: the first command to create arms.json owns it, the second compares instead of rewriting '
      f'({why!r}, {why2!r}, {why3!r})')
# the file appears whole: it is linked into place after it is written, so a loser of the race never reads a partial
# one, and a file that is not JSON was not left by this command (the message used to say 'is not JSON', naming the
# wrong reason: #385 review P2)
(cf / 'trunc.json').write_text('{"arms": {"base": ')
why4 = proposal.claim_folder(cf / 'trunc.json', spec_a)
check(why4 and 'not written by this command' in why4 and not list(cf.glob('.trunc.json.*.tmp')),
      f'propose: a non-JSON arms.json names what it is, and the temporary file is gone ({why4!r})')


def _raise(exc):
    def boom(*a, **k):
        raise exc
    return boom


# a filesystem that cannot hard-link (overlay/CIFS: EPERM/ENOSYS) falls back to an exclusive create instead of dying
# with a traceback after the plan printed; a create that fails for another reason is a refusal naming both errors
# (#385 review P2)
nolink = cf / 'nolink.json'
real_link, real_open = os.link, os.open
os.link = _raise(OSError(13, 'Permission denied'))
try:
    why5 = proposal.claim_folder(nolink, spec_a)
    why6 = proposal.claim_folder(nolink, spec_b)
finally:
    os.link = real_link
check(why5 == '' and why6 and 'commit' in why6 and json.loads(nolink.read_text())['plan']['suite'] == 'basic'
      and not list(cf.glob('.nolink.json.*.tmp')),
      f'propose: an unsupported hard link falls back to an exclusive create, then compares ({why5!r}, {why6!r})')
os.link = _raise(OSError(13, 'Permission denied'))
os.open = _raise(OSError(30, 'Read-only file system'))
try:
    why7 = proposal.claim_folder(cf / 'nope.json', spec_a)
finally:
    os.link, os.open = real_link, real_open
check(why7 and 'cannot be created exclusively' in why7 and not (cf / 'nope.json').exists(),
      f'propose: a folder that holds neither a hard link nor an exclusive create is refused, not a traceback '
      f'({why7!r})')
# two identical `propose` commands into one --out both ran every id: the second claim compared equal, and each run's
# host.fresh rmtreed the folder the other command's same-id run was using, so the owner was billed twice for one
# experiment. The folder's identity file now carries an exclusive lock from the claim until the observation (#385
# review P2); the lock is exclusive and does not wait, since waiting would put a second run behind the first and bill
# it anyway.
lk = cf / 'locked.json'
why8 = proposal.claim_folder(lk, spec_a)
held, busy8 = proposal.claim_lock(lk)
check(why8 == '' and held != -1 and busy8 == '',
      f'propose: a claimed folder is locked by its own command ({why8!r}, {busy8!r})')
second, busy9 = proposal.claim_lock(lk)
check(second == -1 and 'locked by another' in busy9 and 'fresh --out' in busy9,
      f'propose: a second command over a locked folder is refused, not queued ({busy9!r})')
os.close(held)
third, busy10 = proposal.claim_lock(lk)
check(third != -1 and busy10 == '', f'propose: the folder is free again once the first command closes it ({busy10!r})')
os.close(third)
# the lock is held for the whole command, not only around the claim: a stub run that asks for it, as a second
# `propose` would while the first one's arms run, is refused and nothing is observed
r14 = sim_record(ph, status='proposed')
locked = S / 'propose-locked'
ns4 = argparse.Namespace(sim_home=str(ph), record=r14['id'], suite='basic', brain=['scripted:solve'], repeats=1,
                         only='read', host=None, jobs=1, parallel=None, out=str(locked),
                         config='{"sim": false}', env=None, base_config=None, base_env=None,
                         arm_checkout=None, base_checkout=None, dry_run=False, approve_paid=False,
                         unsafe_local=False, prediction_error=None, goal=None)
while_running = []


def locked_run(runs):
    """A second `propose` over the same --out, started while this one's arms run."""
    fd, why = proposal.claim_lock(locked / 'arms.json')
    while_running.append(why)
    if fd != -1:
        os.close(fd)
    return 1


try:
    proposal.cmd_propose(ns4, locked_run)
    err14 = 'returned instead of refusing'
except SystemExit as e:
    err14 = str(e)
check('a run ended in error' in err14 and while_running and 'locked by another' in while_running[0]
      and proposal.load_record(ph, r14['id'])['status'] == 'proposed',
      f'propose: the folder stays locked while its arms run, so a second command cannot join ({while_running!r})')
# an `openai:URL` brain with no #MODEL is asked for the model it lists before any identity is built: its label used to
# be `?@URL`, so a model swapped between an errored run and its rerun passed arms_mismatch and the folder never named
# the model that answered (#385 review P2). The same resolve runs for `run` and `propose` (brains.resolve_all).
import brains as brains_mod
import http.server as http_server
import threading as threading_mod


class _Models(http_server.BaseHTTPRequestHandler):
    ids = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        raw = json.dumps({'data': [{'id': i} for i in self.ids]}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


srv = http_server.ThreadingHTTPServer(('127.0.0.1', 0), _Models)
threading_mod.Thread(target=srv.serve_forever, daemon=True).start()
url = f'http://127.0.0.1:{srv.server_port}/v1'
try:
    _Models.ids = ['m-one']
    b1 = brains_mod.parse('openai:' + url)
    before = b1.label()
    brains_mod.resolve_all([b1])
    _Models.ids = ['m-two']
    b2 = brains_mod.parse('openai:' + url)
    brains_mod.resolve_all([b2])
    spec_one = proposal.arms_spec([runner.Arm('base', ROOT), runner.Arm('arm', ROOT)],
                                  dict(suite='basic', brains=sorted([b1.label()]), only=[], repeats=1,
                                       approvals=None))
    spec_two = proposal.arms_spec([runner.Arm('base', ROOT), runner.Arm('arm', ROOT)],
                                  dict(suite='basic', brains=sorted([b2.label()]), only=[], repeats=1,
                                       approvals=None))
    _Models.ids = []
    b3 = brains_mod.parse('openai:' + url)
    try:
        brains_mod.resolve_all([b3])
        err15 = ''
    except SystemExit as e:
        err15 = str(e)
finally:
    srv.shutdown()
    srv.server_close()
check(before == f'?@{url}' and b1.label() == f'm-one@{url}' and b2.label() == f'm-two@{url}'
      and b2.slug() != b1.slug() and 'brains' in proposal.arms_mismatch(spec_one, spec_two),
      f'propose: an openai brain names the model its endpoint lists, and a swap is another identity '
      f'({before!r}, {b1.label()!r}, {b2.label()!r})')
check('lists no model' in err15 and b3.label() == f'?@{url}',
      f'propose: an endpoint that lists no model is refused, not recorded as ?@URL ({err15[:160]!r})')
# what a run measured is the content, not the commit: an arm checkout is usually dirty (that is what a harness A/B
# runs), so editing the dirty tree between two runs into one --out left commit and dirty=True identical and the
# second run skipped every id as done, closing the record on the previous numbers (#385 review P1). The fingerprint
# covers <checkout>/python, the runtime an arm runs, and ignores __pycache__/.pyc, which a run writes itself.
fake = S / 'fingerprint-co'
(fake / 'python/caminus/ember/resident').mkdir(parents=True, exist_ok=True)
(fake / 'python/caminus/ember/resident/cli.py').write_text('print(1)\n')
(fake / 'python/other.py').write_text('X = 1\n')
fa = runner.Arm('base', fake)
sha1 = fa.tree_sha
check(sha1 == runner._tree_sha(fake / 'python') and len(sha1) == 64,
      f'propose: an arm fingerprints the bytes under its python/ tree ({sha1[:12]})')
(fake / 'python/__pycache__').mkdir(exist_ok=True)
(fake / 'python/__pycache__/other.cpython-311.pyc').write_bytes(b'\x00\x01')
(fake / 'python/other.pyc').write_bytes(b'\x00\x02')
check(runner.Arm('base', fake).tree_sha == sha1,
      'propose: a __pycache__ or .pyc a run writes itself does not change the fingerprint')
(fake / 'python/other.py').write_text('X = 2\n')
sha2 = runner.Arm('base', fake).tree_sha
plug = dict(suite='basic', suite_sha='s', brains=[], only=[], repeats=1, approvals=None)
moved2 = proposal.arms_spec([fa], plug)
same = proposal.arms_spec([runner.Arm('base', fake)], plug)
check(sha2 != sha1 and 'tree_sha' in proposal.arms_mismatch(moved2, same)
      and proposal.arms_mismatch(same, same) == '',
      f'propose: an edit to a dirty tree is another arm ({proposal.arms_mismatch(moved2, same)!r})')
# a suite's name is not its content: editing a fixture or a task's check in place kept the name and the task keys, so
# a folder filled before the edit looked like this experiment's and the record closed on a verdict the edited suite
# never measured (#385 review P1). The digest covers suite.json and every file under fixture/.
suite_copy = S / 'suite-copy'
shutil.copytree(runner.load_suite('basic')['dir'], suite_copy)
digest0 = runner.suite_sha(runner.load_suite(suite_copy))
fix = suite_copy / 'fixture' / 'inventory.py'
fix.write_text(fix.read_text() + '\n# edited\n')
digest1 = runner.suite_sha(runner.load_suite(suite_copy))
sj = json.loads((suite_copy / 'suite.json').read_text())
sj['tasks'][0]['check'] += ' and True'
(suite_copy / 'suite.json').write_text(json.dumps(sj))
digest2 = runner.suite_sha(runner.load_suite(suite_copy))
before = proposal.arms_spec([fa], dict(plug, suite_sha=digest0))
after = proposal.arms_spec([fa], dict(plug, suite_sha=digest1))
check(digest0 == runner.suite_sha(runner.load_suite('basic')) and digest1 != digest0 and digest2 != digest1
      and 'suite_sha' in proposal.arms_mismatch(before, after),
      f'propose: an edited fixture or task is another suite, and a copy of the installed one has its digest '
      f'({proposal.arms_mismatch(before, after)!r})')
# an --out written by another command has no arms.json to compare: `lab.py run` skips every id already in
# results.jsonl, so `propose` into a folder a previous experiment left answered 0 to do and closed THIS record on
# that experiment's rows (#385 review P1). It is refused, and the files that make it foreign are named.
foreign = S / 'propose-foreign-out'
foreign.mkdir(parents=True, exist_ok=True)
(foreign / 'results.jsonl').write_text(json.dumps(dict(id='read.scripted-solve.base.r1')) + '\n')
(foreign / 'plan.json').write_text('{}')
r14 = sim_record(ph, status='proposed')
p = lab('propose', '--sim-home', str(ph), '--record', r14['id'], '--brain', 'scripted:solve',
        '--config', '{"sim": false}', '--only', 'read', '--out', str(foreign))
check(p.returncode != 0 and 'arms.json' in (p.stderr + p.stdout) and 'fresh --out' in (p.stderr + p.stdout)
      and proposal.load_record(ph, r14['id'])['status'] == 'proposed'
      and json.loads((foreign / 'results.jsonl').read_text())['id'] == 'read.scripted-solve.base.r1',
      f'propose: an --out holding another experiment without arms.json is refused ({p.returncode}; '
      f'{(p.stderr + p.stdout)[-300:]})')
empty = S / 'propose-empty-out'
empty.mkdir(parents=True, exist_ok=True)
check(proposal.provenance_refusal(empty, empty / 'arms.json') == ''
      and proposal.provenance_refusal(foreign, foreign / 'arms.json') != '',
      "propose: only a folder holding someone else's rows is refused, an empty one is not")
(empty / 'arms.json').write_text('{}')
check(proposal.provenance_refusal(empty, empty / 'arms.json') == '',
      'propose: a folder carrying arms.json is left to claim_folder to compare')
# the approval flags granted are part of the plan record and of the folder's identity; a folder written
# before approvals were recorded has no such key and still resumes (#385 review P2)
check(proposal.approvals_granted(argparse.Namespace(unsafe_local=True, approve_paid=True))
      == ['--approve-paid', '--unsafe-local'] and proposal.approvals_granted(argparse.Namespace()) is None,
      'propose: the approval flags are recorded as given, and none is None so an older folder still resumes')
# approval_missing reads every optional flag with getattr: a namespace built by hand (a test, `lab.py run`)
# without approve_paid asks for the approval, it does not raise AttributeError (#385 rev-muse P2)
paid_brain = argparse.Namespace(label=lambda: 'relay:x', paid=True, real=False)
check(proposal.approval_missing(argparse.Namespace(host='agent'), [paid_brain])
      == ["--approve-paid (billed through the owner's relay: relay:x)"]
      and proposal.approval_missing(argparse.Namespace(host='agent', approve_paid=True), [paid_brain]) == [],
      'propose: a hand-built namespace without approve_paid asks for it rather than raising')
sp = proposal.arms_spec([argparse.Namespace(name='base', describe=lambda: {'checkout': 'a', 'commit': '1'}),
                         argparse.Namespace(name='arm', describe=lambda: {'checkout': 'b', 'commit': '2'})],
                        dict(suite='basic', brains=['scripted:solve'], only=[], repeats=1,
                             approvals=['--unsafe-local']))
old = dict(sp, plan={k: v for k, v in sp['plan'].items() if k != 'approvals'})
check(sp['plan']['approvals'] == ['--unsafe-local']
      and 'approvals' in proposal.arms_mismatch(sp, dict(sp, plan=dict(sp['plan'], approvals=['--approve-paid'])))
      and proposal.arms_mismatch(sp, dict(sp, plan=dict(sp['plan'], approvals=None)))
      and proposal.arms_mismatch(old, dict(sp, plan=dict(sp['plan'], approvals=None))) == '',
      f'propose: another approval is another experiment, and a folder from before approvals resumes a run that '
      f'grants none ({sp["plan"]}; {proposal.arms_mismatch(old, sp)!r})')
# `observed_text` runs after every arm has been paid for: a cell without median steps is rendered, not a KeyError
# that leaves the record open, and a number the rows do not carry is not invented (#385 review P2)
txt = proposal.observed_text(dict(verdict='no difference', errored=[], brains=[
    dict(brain='scripted:solve', base=dict(passed=1, runs=1, false_completions=0, median_steps=7),
         arm=dict(passed=1, runs=1, false_completions=0, median_steps=None))]))
check('median steps: scripted:solve 7 -> not measured' in txt,
      f'propose: a cell without median steps is recorded as not measured, not a KeyError ({txt!r})')
# one rule for both commands: a real model on the local host needs --unsafe-local, and there is one name for it
import brains as brains_mod
import hosts as hosts_mod
real = [brains_mod.parse('anthropic:8091#claude-haiku-5-5')]
scripted = [brains_mod.parse('scripted:solve')]
check(hosts_mod.local_refusal('local', real) and not hosts_mod.local_refusal(None, real)  # no --host: agent
      and not hosts_mod.local_refusal('local', real, unsafe_local=True)
      and not hosts_mod.local_refusal('agent', real) and not hosts_mod.local_refusal('local', scripted),
      'propose: local_refusal is one rule (real + local host + no --unsafe-local), read by run and propose')
r11 = sim_record(ph, status='proposed')
free = S / 'propose-free'
p = lab('propose', '--sim-home', str(ph), '--record', r11['id'], '--brain', 'anthropic:8091#claude-haiku-5-5',
        '--host', 'local', '--approve-paid', '--unsafe-local', '--config', '{"sim": false}', '--only', 'read',
        '--out', str(free), '--dry-run')
check(p.returncode == 0 and '- host: local' in p.stdout
      and '- approvals: --approve-paid, --unsafe-local' in p.stdout
      and '- timeout: 1500s per goal' in p.stdout and not free.exists(),
      f'propose: --unsafe-local unlocks the local host as it does for run, and the plan names the host, the approval '
      f'flags granted as given, and the timeout ({p.returncode}; {(p.stderr or p.stdout)[-300:]})')
# the documented exit code: 1 on a `worse` verdict, and the record is still observed (docs/dev/ember.md)
worse = S / 'propose-worse'
r10 = sim_record(ph, status='proposed')
ns2 = argparse.Namespace(sim_home=str(ph), record=r10['id'], suite='basic', brain=['scripted:solve'], repeats=1,
                         only='read', host=None, jobs=1, parallel=None, out=str(worse),
                         config='{"sim": false}', env=None, base_config=None, base_env=None,
                         arm_checkout=None, base_checkout=None, dry_run=False, approve_paid=False,
                         unsafe_local=False, prediction_error=None, goal=None)


def worse_run(runs):
    """`lab.py run` with an arm that passes less than base: report.compare's verdict is `worse`."""
    rows = [dict(id='read.scripted-solve.base.r1', brain='scripted:solve', arm='base', task='read', repeat=1,
                 check=True),
            dict(id='read.scripted-solve.arm.r1', brain='scripted:solve', arm='arm', task='read', repeat=1,
                 check=False)]
    (worse / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return 0


try:
    code = proposal.cmd_propose(ns2, worse_run)
except SystemExit as e:
    code = str(e)
check(code == 1 and proposal.load_record(ph, r10['id'])['status'] == 'observed',
      f'propose: a `worse` verdict exits 1 and still closes the record ({code!r})')
# a record the resident closes between the plan and the runs refuses before the first run: the status guard at the
# top reads one snapshot, and without the re-read a closed record was caught only after every arm had run and been
# billed (#385 review P1)
r12 = sim_record(ph, status='proposed')
raced = S / 'propose-raced'
ns3 = argparse.Namespace(sim_home=str(ph), record=r12['id'], suite='basic', brain=['scripted:solve'], repeats=1,
                         only='read', host=None, jobs=1, parallel=None, out=str(raced),
                         config='{"sim": false}', env=None, base_config=None, base_env=None,
                         arm_checkout=None, base_checkout=None, dry_run=False, approve_paid=False,
                         unsafe_local=False, prediction_error=None, goal=None)
claimed = []
real_claim = proposal.claim_folder


def claim_then_close(arms_file, spec):
    """The resident closes the record in the window between the plan and the runs, as a live one may."""
    why = real_claim(arms_file, spec)
    claimed.append(why)
    sim, store, rt = proposal._open(ph)
    try:
        sim.observe(rt, 'g-propose', r12['id'], 'closed by the resident while the plan printed')
    finally:
        store.close()
    return why


def must_not_run(runs):
    raise AssertionError('the arms ran although the record was closed before them')


proposal.claim_folder = claim_then_close
try:
    proposal.cmd_propose(ns3, must_not_run)
    err12 = 'returned instead of refusing'
except SystemExit as e:
    err12 = str(e)
finally:
    proposal.claim_folder = real_claim
check('no longer proposed' in err12 and 'nothing was run' in err12 and claimed == ['']
      and proposal.load_record(ph, r12['id'])['status'] == 'observed'
      and not (raced / 'results.jsonl').exists(),
      f'propose: a record closed before the runs refuses with nothing spent ({err12[:200]})')
# the plan says which flag makes the arms differ, so a second observer need not infer it from arms.json (#385
# review P2). The dry-run line covers the flag in the rendered plan; the helper covers the shapes a flag can take.
r13 = sim_record(ph, status='proposed')
p = lab('propose', '--sim-home', str(ph), '--record', r13['id'], '--brain', 'scripted:solve',
        '--config', '{"sim": false}', '--env', '{"SIM_X": "1"}', '--only', 'read', '--out', str(S / 'propose-diff'),
        '--dry-run')
check(p.returncode == 0 and '- difference: --config, --env' in p.stdout,
      f'propose: the plan names the config/env flags that carry the proposal ({p.returncode}; {p.stdout[-300:]})')
mech = proposal.arm_mechanism
check(mech({'config': {'sim': False}}, [argparse.Namespace(checkout=Path('/a')),
                                        argparse.Namespace(checkout=Path('/b'))]).startswith('--arm-checkout (base runs'),
      'propose: two checkouts are named as --arm-checkout, not inferred from the arms line')
check(mech({'base_config': {'sim': True}}, [argparse.Namespace(checkout=Path('/a')),
                                            argparse.Namespace(checkout=Path('/a'))]) == '--base-config',
      'propose: a base-side override is named as --base-config')
check(mech({}, [argparse.Namespace(checkout=Path('/a')), argparse.Namespace(checkout=Path('/a'))])
      == '(none: the two arms are identical)',
      'propose: a difference nothing produced says so rather than naming no flag')

# ---- scratch hygiene
# rev-muse P2 on #385: the checks drive real runs into --scratch and a previous run leaves results,
# plan.md digests and a scratch home behind; a re-run into the same dir used to fail spuriously
# (KeyError 'observed_by', a stale digest) instead of starting clean. What is pinned here: a marked
# scratch is cleared and reused, and a directory holding someone else's files is refused untouched.
foreign = S.parent / (S.name + '-foreign')
if foreign.exists():
    _force(foreign)
foreign.mkdir(parents=True)
(foreign / 'keep.txt').write_text('not mine to delete\n')
p = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), '--scratch', str(foreign),
                    '--prepare-only'], capture_output=True, text=True, timeout=120)
check(p.returncode != 0 and MARK.name in p.stdout + p.stderr and (foreign / 'keep.txt').read_text() == 'not mine to delete\n',
      f'scratch: a non-empty directory without the marker is refused and left alone ({p.returncode}; {(p.stdout + p.stderr)[:160]})')
reused = S.parent / (S.name + '-reuse')
if reused.exists():
    _force(reused)
# a real first run: the dir starts empty, prepare marks it, then the checks leave their state behind
p0 = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), '--scratch', str(reused),
                     '--prepare-only'], capture_output=True, text=True, timeout=120)
stale = reused / 'run'
stale.mkdir()
(stale / 'results.jsonl').write_text('{"stale": true}\n')
p1 = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), '--scratch', str(reused),
                     '--prepare-only'], capture_output=True, text=True, timeout=120)
cleared1 = not stale.exists()
stale.mkdir(parents=True)
(stale / 'results.jsonl').write_text('{"stale": true, "second run": true}\n')
p2 = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), '--scratch', str(reused),
                     '--prepare-only'], capture_output=True, text=True, timeout=120)
check(p0.returncode == 0 and p1.returncode == 0 and cleared1 and p2.returncode == 0
      and not stale.exists() and (reused / MARK.name).exists(),
      f'scratch: a marked scratch is cleared and reused, twice ({p0.returncode}/{p1.returncode}/{p2.returncode}; '
      f'{(p1.stdout + p2.stdout)[:160]})')
_force(foreign)
_force(reused)

# ---- tui
if shutil.which('tmux'):
    to = S / 'tui'
    p = subprocess.run(TUI + ['--out', str(to), '--size', '60x8', '--cmd', "printf 'ready\\n'; read x; echo got:$x; sleep 20",
                              'wait:ready', 'type:hi', 'key:Enter', 'wait:got:hi', 'shot:s'], capture_output=True, text=True, timeout=60)
    check(p.returncode == 0 and 'got:hi' in (to / 's.txt').read_text() and (to / 's.svg').read_text().startswith('<svg'),
          'tui: keys in, the exact screen and its SVG out')
else:
    print('SKIP tui: tmux is not installed')
p = subprocess.run(TUI + ['--out', str(S / 'ro'), '--cmd', 'true', '--read-only', 'key:BTab', 'type:fix the bug', 'key:Enter'],
                   capture_output=True, text=True, timeout=30)
check(p.returncode == 2 and 'type:fix the bug' in p.stdout and 'key:BTab' not in p.stdout.split('refused')[1],
      'tui: --read-only refuses a goal (and the Enter after it) before anything runs')

print('RESULT', 'FAIL' if FAILS else 'PASS')
sys.exit(1 if FAILS else 0)
