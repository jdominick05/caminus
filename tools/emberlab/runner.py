# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 jdominick05
"""Suites, arms, the run matrix and one run.

A suite is a folder: suite.json (tasks: key, kind, prompt, check, optional scripts for the scripted brain) and fixture/.
Each run copies fixture/ into a fresh work folder (DIR in the prompt and the scripts), keeps a pristine copy of the
suite's `pristine` paths in DIR/.orig for checks that compare, runs ONE goal on a fresh resident home with `once`, reads
the goal's events, then runs the task's check in DIR with sh. The check decides; the resident's own verdict is recorded
beside it, so a goal that finished with a failing check counts as a false completion.

An arm is a version of the harness: a checkout (its resident runtime) plus config.json keys and environment variables.
A/B over arms alternates their order every repeat, so neither arm always runs first.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUITES = REPO / 'benchmarks' / 'ember' / 'suites'
DONE_STATES = {'completed', 'paused', 'error', 'cancelled', 'waiting_user', 'waiting_root', 'waiting_plan'}
MERGE = ("import json,sys;p=sys.argv[1];c=json.load(open(p));c.update(json.load(sys.stdin));"
         "open(p+'.tmp','w').write(json.dumps(c,indent=1));__import__('os').replace(p+'.tmp',p)")


def load_suite(name_or_path):
    d = Path(name_or_path)
    if not (d / 'suite.json').exists():
        d = SUITES / str(name_or_path)
    s = json.loads((d / 'suite.json').read_text())
    s['dir'] = d
    keys = [t['key'] for t in s['tasks']]
    if len(set(keys)) != len(keys):
        raise ValueError(f'suite {s["name"]}: task keys repeat')
    return s


def _tree_sha(root):
    """A content digest of a checkout's runtime tree: the files' paths and bytes, not the commit.

    `commit` and `dirty` do not identify what a run measured: an arm checkout is usually dirty (that is what a
    harness A/B runs), so editing the dirty tree between two runs into one `--out` leaves `commit` and `dirty=True`
    identical and `lab.py run` skips every run id as already done, closing the record on the previous numbers
    (Codex P1, #385). A digest of the bytes changes when the bytes change, whether git can see it or not.

    Over `<root>/python` only: that is the resident runtime an arm runs (`hosts.py` bundles the same directory), and
    including the rest of a checkout would make the identity change for reasons a run cannot observe. `__pycache__`
    and `.pyc` are skipped because a run writes them itself: a digest that changes when the tree is merely imported
    would refuse the second run of the same code.
    """
    h = hashlib.sha256()
    files = sorted(p for p in Path(root).rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    for p in files:
        rel = p.relative_to(root).as_posix()
        data = p.read_bytes()
        h.update(rel.encode() + b'\0' + str(len(data)).encode() + b'\0' + data)
    return h.hexdigest()


def suite_sha(suite):
    """A content digest of a suite's definition and fixtures: suite.json plus every file under fixture/.

    A suite's name is not its content. Editing a task's check, a prompt or a fixture file in place keeps the name and
    the task keys, so a folder filled before the edit looks like this experiment's and `lab.py run` skips its ids;
    the record then closes on a verdict the edited suite never measured (Codex P1, #385). The digest covers the
    definition and the fixture bytes, which together are what a run gives the goal and what the check reads.

    `dirs`/`pristine` entries are paths inside the fixture tree, so the fixture hashing covers them; an empty or
    missing fixture directory contributes nothing, as before.
    """
    d = Path(suite['dir'])
    h = hashlib.sha256(b'suite.json\0' + (d / 'suite.json').read_bytes())
    fixture = d / 'fixture'
    if fixture.is_dir():
        for p in sorted(x for x in fixture.rglob('*')
                        if x.is_file() and '__pycache__' not in x.parts and x.suffix != '.pyc'):
            rel = p.relative_to(fixture).as_posix()
            data = p.read_bytes()
            h.update(b'\0fixture/' + rel.encode() + b'\0' + str(len(data)).encode() + b'\0' + data)
    return h.hexdigest()


def checkout_state(checkout):
    """What identifies the code an arm runs right now: its commit, whether `python` is dirty, and the tree digest.

    A function, not just `Arm.__init__`'s opening move, because the same three values are read again after the runs:
    a checkout that moved (a commit, a rebase, an edit of the dirty tree) while the A/B ran measured other bytes than
    the folder's arms.json names, and a record closed on that mix is permanent (#385 review P1).
    """
    checkout = Path(checkout).expanduser().resolve()
    git = lambda *a: subprocess.run(['git', '-C', str(checkout), *a], capture_output=True, text=True).stdout.strip()
    return dict(commit=git('rev-parse', '--short=12', 'HEAD') or 'unknown',
                # a dirty tree is what a harness A/B usually runs: say so in every record
                dirty=bool(git('status', '--porcelain', '--untracked-files=no', '--', 'python')),
                # the content the commit and the dirty flag cannot name: a dirty tree edited between two runs keeps
                # both identical, and `lab.py run` then skips every id and reads the previous rows (Codex P1, #385)
                tree_sha=_tree_sha(checkout / 'python'))


class Arm:
    def __init__(self, name, checkout, config=None, env=None):
        if not name.replace('-', '').replace('_', '').isalnum():
            raise ValueError(f'arm name {name!r}: letters, digits, - and _ only')
        self.name, self.checkout = name, Path(checkout).expanduser().resolve()
        self.config, self.env = dict(config or {}), dict(env or {})
        if not (self.checkout / 'python' / 'caminus' / 'ember' / 'resident' / 'cli.py').exists():
            raise ValueError(f'arm {name}: {self.checkout} is not a Caminus checkout')
        self.commit, self.dirty, self.tree_sha = (checkout_state(self.checkout)[k]
                                                  for k in ('commit', 'dirty', 'tree_sha'))
        self.runtime_sha = None           # the agent host's uploaded zipapp name, known only after prepare()

    def describe(self):
        """One arm in the shape both plan.json and an `--arms` file read: everything that makes it this arm.

        `checkout` and the `config`/`env` dicts are here because `load_arms` must be able to run from a file this
        writes, and the commit and dirty flag are here because a run id is task.brain.arm.rN and carries no record
        id, so a folder whose checkout moved would reuse the previous rows by id.
        """
        return dict(name=self.name, checkout=str(self.checkout), commit=self.commit, dirty=self.dirty,
                    tree_sha=self.tree_sha, config=dict(self.config), env=dict(self.env),
                    runtime_sha=self.runtime_sha)

    def env_sha(self):
        """A digest of the env the arm runs with, order-independent: the same env has the same digest anywhere."""
        return hashlib.sha256(json.dumps(sorted(self.env.items())).encode()).hexdigest()

    def plan_arm(self):
        """`describe()` as `plan.json` records it: the env values become their names and a digest.

        `--env` is how an arm gets a relay key or a home path, and `plan.json` was writing those values into a plain
        file beside the rows. `arms.json` keeps them because `load_arms` runs the arm from that file; `plan.json`
        needs only what makes the plan reproducible, so it names the keys and `env_sha`, and a digest can be checked
        against `arms.json` without the value ever being written twice (#385 review P2).
        """
        d = self.describe()
        d.pop('env')
        d.update(env_keys=sorted(self.env), env_sha=self.env_sha())
        return d


def load_arms(specs, arms_file):
    """--arm NAME[=CHECKOUT] (repeatable) and/or --arms FILE ({"NAME": {"checkout", "config", "env"}})"""
    arms = []
    if arms_file:
        spec = json.loads(Path(arms_file).read_text())
        # `proposal.py` writes {"arms": ..., "plan": ...}: the plan keeps a folder to one experiment, and the arms
        # are `describe()` dicts, so this file and plan.json speak one shape. A hand-written --arms file is the map.
        for name, v in (spec['arms'] if isinstance(spec.get('arms'), dict) else spec).items():
            arms.append(Arm(name, v.get('checkout') or REPO, v.get('config'), v.get('env')))
    for s in specs or []:
        name, _, path = s.partition('=')
        arms.append(Arm(name, path or REPO))
    if not arms:
        arms = [Arm('base', REPO)]
    if len({a.name for a in arms}) != len(arms):
        raise ValueError('arm names repeat')
    return arms


def plan(suite, brains, arms, repeats, only=None):
    tasks = [t for t in suite['tasks'] if not only or t['key'] in only]
    if only and len(tasks) != len(only):
        raise ValueError(f'unknown task in --only: {sorted(set(only) - {t["key"] for t in tasks})}')
    runs = []
    for r in range(1, repeats + 1):
        order = arms if r % 2 else arms[::-1]
        for t in tasks:
            for b in brains:
                for a in order:
                    runs.append(dict(id=f'{t["key"]}.{b.slug()}.{a.name}.r{r}', task=t, brain=b, arm=a, repeat=r))
    return runs


def metrics(events):
    calls = [e for e in events if e.get('kind') == 'tool_result']
    errors = {}
    for e in calls:
        data = e.get('data') or {}
        res = data.get('result')
        if isinstance(res, dict) and res.get('error'):
            errors.setdefault(data.get('name') or '?', [0, str(res['error'])[:200]])[0] += 1
    names = {}
    for e in calls:
        n = (e.get('data') or {}).get('name') or '?'
        names[n] = names.get(n, 0) + 1
    return dict(tool_calls=len(calls), tool_errors=sum(v[0] for v in errors.values()), errors=errors, tools=names)


def _json_tail(raw):
    for line in reversed(raw.decode(errors='replace').strip().splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def execute(run, host, suite, timeout, out):
    """one run, start to record; raises nothing for a run that fails (the record says how)"""
    t, b, a = run['task'], run['brain'], run['arm']
    rdir = host.fresh(run['id'])
    home, work, bridge = f'{rdir}/home', f'{rdir}/home/work', f'{rdir}/bridge'
    local_rec = Path(out) / 'runs' / run['id']
    local_rec.mkdir(parents=True, exist_ok=True)
    rec = dict(id=run['id'], suite=suite['name'], task=t['key'], kind=t.get('kind'), brain=b.label(), brain_kind=b.kind,
               arm=a.name, commit=a.commit, dirty=a.dirty, tree_sha=a.tree_sha, runtime_sha=a.runtime_sha,
               repeat=run['repeat'], host=host.name, ts=time.time())
    rt = host.runtime(a)
    env_prefix = ['env', *[f'{k}={v}' for k, v in a.env.items()]] if a.env else []

    def agent(op, *args, secs=120, stdin=None):
        return host.run([*env_prefix, *rt, op, 'lab', '--home', home, '--json', *args], secs, stdin=stdin)

    try:
        host.put_tree(suite['dir'] / 'fixture', work)
        for d in suite.get('dirs', []):
            host.write(f'{work}/{d}/.keep', b'')
        for p in suite.get('pristine', []):
            src = suite['dir'] / 'fixture' / p
            if src.is_dir():
                host.put_tree(src, f'{work}/.orig/{p}')
            else:
                host.write(f'{work}/.orig/{p}', src.read_bytes())
        for sub in ('req', 'res'):
            host.write(f'{bridge}/{sub}/.keep', b'')
        if hasattr(b, 'load'):
            b.load(run['id'], t, work)
        r = agent('init', *b.init_args(run['id']), '--bridge', bridge, '--bus', f'{home}/mail.sqlite3')
        if r.returncode:
            raise RuntimeError('init failed: ' + (r.stderr or r.stdout).decode()[-400:])
        cfg = {**b.config(), **a.config}
        if cfg:
            r = host.run(['python3', '-I', '-c', MERGE, f'{home}/config.json'], 60, stdin=json.dumps(cfg).encode())
            if r.returncode:
                raise RuntimeError('config merge failed: ' + r.stderr.decode()[-300:])
        prompt = t['prompt'].replace('DIR', work)
        t0 = time.time()
        r = agent('once', '--goal', prompt, '--max-seconds', str(timeout), secs=timeout + 120)
        rec['seconds'] = round(time.time() - t0, 1)
        (local_rec / 'once.out').write_bytes(r.stdout + b'\n--- stderr\n' + r.stderr)
        goal = _json_tail(r.stdout) or {}
        rec['goal_id'] = goal.get('id')
        rec['state'] = goal.get('state') or ('timeout' if r.returncode == 124 else 'error')
        rec['steps'] = goal.get('steps')
        if goal.get('error'):
            rec['error'] = str(goal['error'])[:300]
        events, after = [], 0
        while True:
            page = _json_tail(agent('events', '--after', str(after), secs=180).stdout)
            if not isinstance(page, list) or not page:
                break
            events += [e for e in page if isinstance(e, dict)]
            after = max(int(e.get('seq') or after) for e in page)
            if len(page) < 200:
                break
        if rec.get('goal_id'):
            events = [e for e in events if e.get('goal') in (rec['goal_id'], None)]
        (local_rec / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
        rec.update(metrics(events))
        st = _json_tail(agent('status').stdout) or {}
        rec['spend'] = (st.get('spend_goals') or {}).get(rec.get('goal_id'))
        chk = host.run(['sh', '-c', t['check']], 120, cwd=work)
        rec['check'] = chk.returncode == 0
        (local_rec / 'check.out').write_bytes(chk.stdout + b'\n--- stderr\n' + chk.stderr)
    except Exception as e:                       # (one run's failure is a record, never the end of the matrix)
        rec.setdefault('state', 'error')
        rec['error'] = f'{type(e).__name__}: {e}'[:400]
        rec.setdefault('check', False)
    rec['finished'] = rec.get('state') == 'completed'
    rec['false_completion'] = rec['finished'] and not rec.get('check')
    (local_rec / 'run.json').write_text(json.dumps(rec, indent=1))
    return rec
