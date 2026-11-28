# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 jdominick05
"""Measure a proposal: a `proposed` SIM record becomes two arms, and the runs close it.

Issue #341, step 3. The post-goal reflection writes what tool, rule or prompt change would have made a goal faster or
more reliable, as a SIM record with status `proposed`. This turns one into an experiment and appends what the lab
showed, so the record's prediction is judged by runs instead of by argument:

  base  the tree as it runs now
  arm   the same tree with the proposal applied: `--config` / `--env` overrides, or another `--arm-checkout`

Both arms run through `lab.py run` itself, so there is one run path, one results file and one verdict
(`report.compare`: an arm is better only if it is better across the board). The record then gets the outcome with
`sim.observe`, carrying the per-brain numbers the verdict rests on; `observed_by` names `emberlab`, the predictor's
own brain stays named in `brain`.

What runs is the owner's decision, not this command's default. The plan (suite, arms, brains, number of runs, which
brains bill) is printed first, and nothing starts until `--approve-paid` for a brain billed through the owner's relay
(a `anthropic:` or `openai+relay:` brain; the tool A/B brain is Haiku there), and `--unsafe-local` for a real model
on the local host. `--dry-run` stops after the plan.

  python -I tools/emberlab/lab.py propose --sim-home HOME --record ID --suite basic \\
      --brain anthropic:8091#claude-haiku-5-5 --repeats 3 --config '{"sim": false}'

The record says *what* to change and the command line says *how*, because a proposal in words is not a switch. A
proposal about code names `--arm-checkout DIR`: `base` runs this checkout and `arm` runs DIR, so DIR should be this
tree plus the change, and the diff between the two arms is then only that change.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import sqlite3
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brains as brains_mod     # noqa: E402
import hosts as hosts_mod       # noqa: E402
import report as report_mod     # noqa: E402
import runner                   # noqa: E402

RESIDENT = Path(runner.REPO) / 'python'


def _rt(store, observer=None):
    """A runtime-shaped object for the SIM module: it needs a store, a home and the session's config, not a model.

    The config is the home's own `config.json`, so the SIM switch a resident honours is the switch this command
    honours: a home with `sim: false` refuses an observation instead of accepting one its own `/sim` would not
    (#385 review P2). `observer` names who appends: **emberlab**, the lab's own name in `observed_by`, since the
    resident's brain is not the one reading the runs.
    """
    cfg = {}
    try:
        cfg = json.loads((store.home / 'config.json').read_text()) or {}
    except FileNotFoundError:
        pass
    except ValueError as e:
        raise SystemExit(f'{store.home}/config.json is not JSON ({e}): the SIM switch cannot be read')
    if observer:
        cfg = dict(cfg, model=observer, provider='')
    return type('RT', (), {'store': store, 'home': store.home, 'config': cfg})()


class _Read:
    """A session's `agent.sqlite3` opened read-only: no mkdir, no schema, no journal-mode change, no lock file.

    `Store(home)` creates the home, its tables and its journal mode before the first query, and a command that
    only reads a record must not do any of that to a live resident's database: `--dry-run` promises to create
    nothing, and even a real run reads the record before it knows it will write (#385 review P2).

    A WAL database with no connection open has no `-wal`/`-shm`: a live writer holds both for as long as it is
    attached, so their absence says nobody is attached and the file alone is the whole state. Then `immutable=1`
    reads it without a lock or a sidecar. With a writer attached, `mode=ro` joins the existing sidecars instead;
    that reader never writes either, so the only cost is that it may wait for the writer's lock.
    """
    def __init__(self, path):
        self.home = path.parent
        attached = path.with_name(path.name + '-wal').exists() or path.with_name(path.name + '-shm').exists()
        uri = f'{path.as_uri()}?{"mode=ro" if attached else "immutable=1"}'
        self.db = sqlite3.connect(uri, uri=True, timeout=10, isolation_level=None)
        self.db.row_factory = sqlite3.Row

    def close(self):
        self.db.close()


def _open(home, write=True, observer=None):
    """The session's store, and the resident's SIM module from this lab's own tree (one log format, one writer).

    `write=False` opens the database read-only, and a home with no `agent.sqlite3` is refused rather than created:
    a read is a read (#385 review P2).
    """
    if str(RESIDENT) not in sys.path:
        sys.path.insert(0, str(RESIDENT))
    from caminus.ember.resident import sim
    from caminus.ember.resident.store import Store
    path = Path(home).expanduser().resolve() / 'agent.sqlite3'
    if write:
        store = Store(path.parent)
    elif path.is_file():
        store = _Read(path)
    else:
        raise SystemExit(f'no session store at {path}: `propose` reads a resident\'s SIM log and does not create a '
                         'home (check --sim-home)')
    try:
        rt = _rt(store, observer)
    except BaseException:
        store.close()
        raise
    return sim, store, rt


def load_record(home, rid):
    """One record's newest version from a session's home, or None; the store is opened read-only.

    A store that cannot be read (locked by a live writer past the busy timeout, not a database, a file the user
    cannot open) is refused as this module refuses everything else: one `SystemExit` naming the path and the reason,
    not a traceback from a command that has already said it only reads (#385 review P2). `sim._get` catches
    `OperationalError` on its first query and falls back to `_load`, which re-runs the same query unguarded, so the
    failure can surface from either query and both are wrapped here, around the one call that reads.
    """
    try:
        sim, store, rt = _open(home, write=False)
    except sqlite3.Error as e:
        raise SystemExit(f'the SIM store in {Path(home).expanduser().resolve()} could not be read ({e}): nothing was '
                         f'changed')
    try:
        try:
            return sim._get(rt, rid)
        except sqlite3.Error as e:
            raise SystemExit(f'the SIM store in {Path(home).expanduser().resolve()} could not be read ({e}): nothing '
                             f'was changed (a live writer holds its lock past the busy timeout, or the file is not a '
                             f'database)')
    finally:
        store.close()


def write_observation(home, rid, observed, eval_context='', prediction_error='', goal=None):
    """Append the outcome to the record the arms measured: one new version, status 'observed'.

    It goes through `sim.tool`, the one path a resident's `/sim` takes, so the lab cannot append where the resident
    would refuse: the `sim: false` switch that stops an A/B arm from reading the shared lessons stops this write
    too, rather than a home with the log off accepting an observation its own tool would reject (#385 review P2).

    The "a closed record stays closed" rule is the resident's own, and it already holds across the race: `observe`
    reads the newest version and appends in one `_atomic` transaction (`BEGIN IMMEDIATE`), and `_record` refuses a
    version of an already-observed record. So a resident that closes the record while the arms run is not raced
    over; what the command owes is to report that refusal as one, not as a traceback (#385 review P2).
    """
    sim, store, rt = _open(home, observer='emberlab')
    try:
        try:
            rec = sim.tool(rt, goal, dict(action='observe', id=rid, observed=observed,
                                          eval_context=eval_context, prediction_error=prediction_error))
            return rec['record']
        except ValueError as e:
            # the record was closed (or removed) between the plan and the runs: say which, and leave it as it is
            raise SystemExit(f'the runs are kept but the record is not closed: {e}')
    finally:
        store.close()


def arms_from_proposal(prop, base_checkout=None, arm_checkout=None):
    """Two arms from one record: `base` runs now, `arm` has the proposal applied."""
    base_path = base_checkout or runner.REPO
    if arm_checkout and Path(arm_checkout).expanduser().resolve() != Path(base_path).expanduser().resolve():
        return [runner.Arm('base', base_path, prop.get('base_config'), prop.get('base_env')),
                runner.Arm('arm', arm_checkout, prop.get('config'), prop.get('env'))]
    checkout = arm_checkout or base_path
    base = runner.Arm('base', checkout, prop.get('base_config'), prop.get('base_env'))
    arm = runner.Arm('arm', checkout, prop.get('config'), prop.get('env'))
    if base.config == arm.config and base.env == arm.env:
        raise ValueError('the proposal changes neither config nor env, and no --arm-checkout names the change: two '
                         'identical arms measure nothing')
    return [base, arm]


ARM_KEYS = ('checkout', 'commit', 'dirty', 'tree_sha', 'config', 'env')
PLAN_KEYS = ('suite', 'suite_sha', 'brains', 'only', 'repeats', 'approvals')
APPROVAL_FLAGS = ('--approve-paid', '--unsafe-local')


def approvals_granted(a):
    """The approval flags this command line actually carries, as given, or None when it carries none.

    The flag that unlocked a run is part of what a folder records: a log that names none of them hid the permission a
    run held (the review asked for plan.json/the record to name it). None, not [], when nothing was granted: a folder
    written before approvals were recorded has no such key, and a run that grants nothing must still resume it.

    A flag that is not on the command line is read with getattr, so a namespace built by hand (`lab.py run`, a test)
    does not have to add every name.
    """
    got = [f for f in APPROVAL_FLAGS if getattr(a, f[2:].replace('-', '_'), False)]
    return got or None


def arms_spec(arms, plan=None):
    """What a folder records as its identity: each arm as `Arm.describe` writes it, and the plan that reads the rows.

    The commit and the dirty flag belong to the identity, not just the path: a run id is task.brain.arm.rN and
    carries no record id, and `lab.py run` skips every id already in results.jsonl, so a checkout that moved (a
    commit, a rebase) would close a new record with the previous rows (#385 review: reproduced by committing
    between two runs into one folder).

    The suite, brains, `--only`, repeats and the approval flags granted belong to it for the same reason: one folder
    holds one experiment. A first run that errored partway leaves its rows, and a rerun with another `--brain` (or
    another suite or task set) would pass the arms check and close its record with numbers this command never chose
    (#385 review P1). The approvals are the same kind of fact: a folder filled under `--unsafe-local` was allowed to
    do what a later command without it was not, and resuming must not silently drop that (#385 review P2).
    """
    return dict(arms={x.name: x.describe() for x in arms},
                plan=None if plan is None else {k: plan.get(k) for k in PLAN_KEYS})


def _arms_of(spec):
    """The arms map of a folder spec: `{"arms": ..., "plan": ...}` from `arms_spec`, or a bare map (an older one)."""
    if isinstance(spec, dict) and isinstance(spec.get('arms'), dict):
        return spec['arms']
    return spec if isinstance(spec, dict) else {}


def arms_mismatch(prior, spec):
    """Why the runs already in a folder are not this experiment's, or '' when they are."""
    pa, sa = _arms_of(prior), _arms_of(spec)
    for name in sorted(set(pa) | set(sa)):
        if not isinstance(pa.get(name), dict) or name not in sa:
            return f'arm {name}: the folder holds {pa.get(name)!r}'
        for k in ARM_KEYS:
            if pa[name].get(k) != sa[name].get(k):
                return f'arm {name}: {k} was {pa[name].get(k)!r}, this run is {sa[name].get(k)!r}'
    sp = spec.get('plan') if isinstance(spec, dict) else None
    if sp is not None:
        pp = prior.get('plan') if isinstance(prior, dict) else None
        if not isinstance(pp, dict):
            return 'the folder does not record the plan it runs (an older arms.json)'
        for k in PLAN_KEYS:
            if pp.get(k) != sp.get(k):
                return f'plan: {k} was {pp.get(k)!r}, this run is {sp.get(k)!r}'
    return ''


ARM_CONTENT = ('config', 'env')
RUN_OWNS = ('arms', 'out')       # the run fields `cmd_propose` computes; see cmd_propose


def _digest(value):
    """A short, order-independent digest of an arm's config or env: what a refusal may print about them.

    `--env` carries a relay key or a home path (see `Arm.plan_arm`), so a message names the digest and never the
    values: the owner still sees which of the two moved, and the folder's log does not repeat the secret.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def arms_drift(arms, spec, arms_file=None):
    """What an arm says now that differs from the identity the folder recorded, or ''.

    `arms.json` (and `plan.json` beside it) names the commit, the tree digest and the config/env each arm held when
    the command claimed the folder. The runs then take minutes to hours, and an arm checkout is usually a dirty
    tree: committing, rebasing or editing it while the A/B runs measures other bytes than the folder names, and the
    record would close on a mix of both - permanent, since an observed record is append-only. Read the same values
    again and compare them with the folder's own record, not with the in-memory `Arm`: the folder is what a reader
    trusts (#385 review P1). The suite gets the same re-read; this is the arms' half.

    `config` and `env` belong to the identity too (`ARM_KEYS`, which `arms_mismatch` enforces at claim time), and
    the runs read them from `arms_file` - `lab.py run --arms arms.json` - not from the in-memory arms, so that file
    is re-read here and must still say what the folder claimed. A file that is gone or unreadable is a refusal, not
    a pass: the identity cannot be confirmed from an absent record. Values are compared whole, and a refusal prints
    digests, never the env's values (#385 review P1).
    """
    recorded = _arms_of(spec)
    live = recorded
    if arms_file is not None:
        try:
            live = _arms_of(json.loads(Path(arms_file).read_text()))
        except (OSError, ValueError) as e:
            return f'the folder record {arms_file} cannot be read back ({type(e).__name__}: {e})'
    for arm in arms:
        was = recorded.get(arm.name) if isinstance(recorded.get(arm.name), dict) else {}
        now = runner.checkout_state(arm.checkout)
        for k in ('commit', 'tree_sha'):
            if now[k] != was.get(k):
                return (f'arm {arm.name}: {k} was {str(was.get(k))[:12]!r} when the folder was claimed and is '
                        f'{str(now[k])[:12]!r} now ({arm.checkout})')
        # what the runs are actually told: the file the command claims, when there is one; the in-memory arm
        # otherwise (a caller that hands `arms_drift` no file is asking about the checkout alone)
        said = live.get(arm.name) if isinstance(live.get(arm.name), dict) else None
        if arms_file is not None and said is None:
            return f'arm {arm.name}: the folder record {arms_file} no longer names it'
        for k in ARM_CONTENT:
            now_v = said.get(k) if said is not None else getattr(arm, k)
            if (now_v or {}) != (was.get(k) or {}):
                where = f'{arms_file}' if said is not None else 'the arms this command holds'
                return (f'arm {arm.name}: {k} was {_digest(was.get(k) or {})} when the folder was claimed and is '
                        f'{_digest(now_v or {})} now ({where})')
    return ''


def provenance_refusal(out, arms_file):
    """Why `out` already holds results this command did not write, or '' when it is fresh or its own.

    `lab.py run` skips every run id already in results.jsonl, and a run id is task.brain.arm.rN: it names no
    experiment, no suite digest and no provenance. So an `--out` left by an earlier run (a `lab.py run`, or a
    `propose` that lost its arms.json) answers `0 to do` here, and `cmd_propose` then reads that earlier experiment's
    rows, prints its verdict and closes *this* record with it — permanently, since an observed record is append-only
    (Codex P1, #385).

    `arms.json` is the folder's provenance: only a folder that carries one can be compared, and `claim_folder`
    refuses a different one. A folder without it but with results, a plan or per-run records is not this
    experiment's, and adopting it is the bug: refuse, and name the files so the owner can look before choosing.
    An empty folder, or one holding only this command's own `arms.json`, is not refused (the old behaviour).
    """
    if arms_file.exists():
        return ''                     # claim_folder compares it with this spec
    found = [p.name for p in (out / 'results.jsonl', out / 'plan.json') if p.exists()]
    if not found and (out / 'runs').is_dir() and any((out / 'runs').iterdir()):
        found.append('runs/')
    if not found:
        return ''
    return (f'{out} already holds an experiment without arms.json ({", ".join(found)}), so its run ids would be '
            'skipped and its rows read as this experiment\'s: use a fresh --out (or remove those files if they are '
            'not wanted)')


def claim_folder(arms_file, spec):
    """Make `arms_file` this experiment's, or say why the folder already holds another.

    The create is exclusive (`O_CREAT|O_EXCL`): two `propose` commands starting at once into one `--out` would both
    read the same absent file and both write it, last writer wins, and the shared `results.jsonl` would then hold
    both experiments' rows by id — the reuse bug the plan check closes for the sequential case (#385 review P2).
    An existing file is only read, never rewritten: the folder's identity is written once.

    The create is `os.link` of a fully written temporary file in the same folder: `O_CREAT|O_EXCL` made the name
    first and filled it after, so a second `propose` starting in that window read an empty or half-written file and
    was told 'is not JSON' - a message about the wrong thing (#385 review). A hard link either fails because the
    name exists or makes the whole file appear at once, and it fails the same way on every filesystem this runs on.

    A filesystem that cannot hard-link (overlay, CIFS, some network mounts: EPERM/ENOSYS/EOPNOTSUPP) falls back to
    an exclusive create of the name and the whole spec written inside it: refusing there would make `propose`
    unusable in those folders, and the only cost is that a loser of a concurrent race can see the file before it is
    whole. A create that fails for any other reason is a refusal naming both errors, not a traceback (#385 review P2).
    """
    tmp = arms_file.with_name('.%s.%d.tmp' % (arms_file.name, os.getpid()))
    tmp.write_text(json.dumps(spec, indent=1))
    try:
        os.link(str(tmp), str(arms_file))
        return ''
    except FileExistsError:
        pass
    except OSError as e:
        try:
            fd = os.open(str(arms_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            pass
        except OSError as e2:
            return (f'{arms_file} cannot be created exclusively (hard link: {e}; create: {e2}): use a fresh --out '
                    'on a local filesystem')
        else:
            with os.fdopen(fd, 'w') as f:
                f.write(json.dumps(spec, indent=1))
                f.flush()
                os.fsync(f.fileno())
            return ''
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    try:
        prior = json.loads(arms_file.read_text())
    except ValueError:
        # this command writes the file whole or not at all, so a file that is not JSON was made by something else.
        # One race is ours, though: on a filesystem without hard links the fallback creates the name first and
        # writes the spec into it, and a loser of that race can read the name while it is still empty. Give the
        # winner a moment before saying the file is not ours (#385 review P2).
        for _ in range(8):
            time.sleep(0.05)
            try:
                prior = json.loads(arms_file.read_text())
                break
            except (OSError, ValueError):
                continue
        else:
            return (f'{arms_file} is not JSON and was not written by this command: use a fresh --out (or fix or '
                    'remove the file)')
    return arms_mismatch(prior, spec)


def claim_lock(arms_file):
    """The folder's exclusive lock, held from the claim until the observation is written; or the refusal, and -1.

    `claim_folder` makes a *second identical* command compare equal, and each `lab.py run` reads its `done` set
    once, so both commands ran every id: each run's `host.fresh` rmtrees the folder the other command's same-id
    run is using, and the owner is billed twice for one experiment (#385 review P2). An advisory `flock` on
    `arms.json` turns the second command into a refusal instead: the file exists once the folder is claimed and
    is never rewritten, so the lock stays on the folder's identity for as long as the first command holds it.

    `LOCK_NB` is the point: waiting would put a second run behind the first and still bill it. A filesystem
    without advisory locks fails the `flock` call, which is a refusal too, naming it.
    """
    try:
        fd = os.open(str(arms_file), os.O_RDONLY)
    except OSError as e:
        return -1, f'{arms_file} cannot be opened to lock it ({e}): use a fresh --out'
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        os.close(fd)
        return -1, (f'{arms_file} is locked by another `propose` ({e.strerror or e}): one command owns a folder '
                    'until its record closes, so let it finish (or use a fresh --out)')
    return fd, ''


def approval_missing(a, brains):
    """What the owner still has to allow before this run starts: the plan alone never starts one.

    The local-host rule is `hosts.local_refusal`, the same function `lab.py run` calls: a second copy of the rule here
    answered the same command line differently (#385 review), so both commands call it and neither can drift. A flag
    that is not on the command line is read with getattr, so a caller building the namespace by hand (a test,
    `lab.py run`) does not have to add it.
    """
    paid = [b.label() for b in brains if getattr(b, 'paid', False)]
    missing = []
    if paid and not getattr(a, 'approve_paid', False):
        missing.append("--approve-paid (billed through the owner's relay: " + ', '.join(paid) + ')')
    why = hosts_mod.local_refusal(a.host, brains, unsafe_local=getattr(a, 'unsafe_local', False))
    if why:
        real = [b.label() for b in brains if getattr(b, 'real', False)]
        missing.append('--unsafe-local (a real model runs as you on the local host: ' + ', '.join(real) + ')')
    return missing


def arm_mechanism(prop, arms):
    """What makes the two arms differ: `--arm-checkout`, or the config/env override that carries the proposal.

    The plan prints the arms' paths and configs, but not which flag produced the difference, so a second observer
    auditing the plan against `arms.json` had to infer it (#385 review P2).
    """
    if len(arms) == 2 and Path(arms[0].checkout) != Path(arms[1].checkout):
        return f'--arm-checkout (base runs {arms[0].checkout}, arm runs {arms[1].checkout})'
    named = [f'--{k.replace("_", "-")}' for k in ('config', 'env', 'base_config', 'base_env')
             if isinstance(prop.get(k), dict)]
    return ', '.join(named) or '(none: the two arms are identical)'


def suite_line(suite, tasks, repeats, suite_digest=None):
    """The plan's `- suite:` line: name, tasks, repeats and (when known) the content digest.

    The digest is printed because the name is not the content: a suite edited in place keeps its name and task keys,
    so a reader of a plan from an earlier version cannot tell it apart without the digest (#385 review P1).
    """
    line = f"- suite: {suite['name']} ({tasks} tasks), {repeats} repeat(s)"
    return line + (f", content {suite_digest[:12]}" if suite_digest else '')


def plan_md(prop, suite, brains, arms, repeats, out, tasks, host=None, timeout=None, unlocked='', approvals=None,
            suite_digest=None):
    """The plan the owner reads before anything runs, and the one thing that names everything that will run.

    The host decides where a real model runs (as the user, or on the agent account) and the timeout is how long one
    goal may take, so a plan without them did not name everything (#385 review). `host` is the resolved kind
    (`hosts.kind`), not the flag: `--host local` and a real brain with no flag resolve to the same place. The
    approval flags are printed as given, so a plan names the permission a run holds rather than just that it holds
    one (#385 review P2).
    """
    paid = [b.label() for b in brains if getattr(b, 'paid', False)]
    real = [b.label() for b in brains if getattr(b, 'real', False)]
    unresolved = [b.label() for b in brains if b.label().startswith('?@')]
    runs = len(arms) * len(brains) * tasks * max(1, repeats)
    host = host or hosts_mod.kind(None, brains)
    timeout = suite.get('timeout', 1500) if timeout is None else timeout
    lines = [f"proposal {prop['id']} · {prop.get('problem_group') or 'unclassified'}", '',
             f"- change: {(prop.get('change') or '').strip()[:400]}",
             f"- predicted: {(prop.get('predicted') or '').strip()[:400]}",
             suite_line(suite, tasks, repeats, suite_digest),
             f"- host: {host}" + (' (a real model runs as you here: '
                                    + (unlocked or '--unsafe-local') + ')'
                                    if host == 'local' and real else ''),
             f"- timeout: {timeout}s per goal",
             '- approvals: ' + (', '.join(approvals) if approvals else
                                'none (nothing billed, no real model runs as you)'),
             f"- brains: {', '.join(b.label() for b in brains)}"
             + (' (?@ is asked for its model when the run starts: a dry run asks no endpoint)' if unresolved else ''),
             f"- arms: {', '.join(a.name + ' @ ' + str(a.checkout) + ' ' + a.commit for a in arms)}",
             f"- difference: {arm_mechanism(prop, arms)}",
             f"- runs: {runs} in {out}",
             '- verdict: an arm is better only if it is better across the board (report.compare)']
    if paid:
        lines.append(f"- **billed through the owner's relay: {', '.join(paid)}** (the relay's daily cap is the guard)")
    return '\n'.join(lines)


def _steps(v):
    """One cell's median steps for the record, or 'not measured' when the row did not carry them.

    The record is written after every run has been paid for: a malformed row must not raise a KeyError there and
    leave the record open, and a number the rows do not carry must not be invented (#385 review P2).
    """
    v = v.get('median_steps') if isinstance(v, dict) else None
    return f'{v:g}' if isinstance(v, (int, float)) else 'not measured'


def observed_text(verdict):
    """The record's `observed` field: the verdict and the per-brain numbers it rests on."""
    cells = [o for o in verdict['brains'] if 'missing' not in o]
    parts = [verdict['verdict']]
    if cells:
        parts.append('passed base -> arm: ' + ', '.join(
            f"{o['brain']} {o['base']['passed']}/{o['base']['runs']} -> {o['arm']['passed']}/{o['arm']['runs']}"
            for o in cells))
        parts.append('false completions: ' + ', '.join(
            f"{o['brain']} {o['base']['false_completions']} -> {o['arm']['false_completions']}" for o in cells))
        parts.append('median steps: ' + ', '.join(
            f"{o['brain']} {_steps(o['base'])} -> {_steps(o['arm'])}" for o in cells))
    if verdict.get('errored'):
        parts.append(f"{len(verdict['errored'])} run(s) ended in error and are left out: "
                     + ', '.join(verdict['errored'][:5]))
    return '; '.join(parts)


def _json_arg(text, what):
    if not text:
        return None
    try:
        value = json.loads(text)
    except ValueError as e:
        raise ValueError(f'--{what} is not JSON: {e}')
    if not isinstance(value, dict):
        raise ValueError(f'--{what} must be a JSON object')
    return value


def cmd_propose(a, run, run_defaults=None):
    """`lab.py propose ...`: the plan first, the runs only with the owner's flags, and the record closed by numbers.

    `run` is `lab.cmd_run` and `run_defaults` is `lab.run_defaults`: the run's own parser supplies the namespace the
    runs start from, so a flag added to `lab.py run` reaches a propose-started run with its own default. A direct
    caller may pass only a stub `run`, which reads no field of the namespace.
    """
    home = Path(a.sim_home).expanduser().resolve()
    rec = load_record(home, a.record)
    if rec is None:
        raise SystemExit(f'no SIM record {a.record} in {home}')
    if rec.get('status') == 'observed':
        raise SystemExit(f'SIM record {a.record} is already observed: a closed record is not measured again')
    if rec.get('status') != 'proposed':
        # `recorded` is a prediction whose change is already in the tree: measuring it compares the tree with itself.
        raise SystemExit(f"SIM record {a.record} is {rec.get('status') or 'unstated'}, not proposed: only a "
                         "`proposed` record (a change nobody has made yet) is measured; a `recorded` one's change is "
                         'already in the tree, so an A/B of it measures nothing')
    # an observation lives in the record's own goal (`sim.observe` attaches it there whatever goal its observer is
    # in), so a --goal naming a different goal is refused before the runs rather than accepted and then ignored
    # (#385 review P2). A short id is a prefix, as `say --goal` resolves one; a record with no goal takes the one
    # given, since that is the id `sim.observe` falls back to.
    rec_goal = rec.get('goal') or ''
    if a.goal and rec_goal and not (rec_goal == a.goal or rec_goal.startswith(a.goal)):
        raise SystemExit(f'SIM record {a.record} lives in goal {rec_goal[:8]}, not {a.goal[:8]}: an observation is '
                         'appended to the record\'s own goal, so --goal cannot move it (drop --goal to record the '
                         'observation where the prediction lives)')
    try:
        prop = dict(rec, config=_json_arg(a.config, 'config'), env=_json_arg(a.env, 'env'),
                    base_config=_json_arg(a.base_config, 'base-config'), base_env=_json_arg(a.base_env, 'base-env'))
    except ValueError as e:
        raise SystemExit(str(e))
    suite = runner.load_suite(a.suite)
    # before the plan prints and before arms_spec: an `openai:URL` brain with no `#MODEL` labels itself `?@URL`
    # until its endpoint is asked, and that label is part of the folder's identity (#385 review P2). A dry run asks
    # nothing: `resolve` is a network `GET /models`, so an unreachable endpoint would fail the command whose whole
    # promise is "print the plan and create nothing" (#385 review P2). The plan then names the label as it stands and
    # says the endpoint is asked when the run starts; a real run resolves here, before any identity is built.
    brains = [brains_mod.parse(s, a.parallel) for s in a.brain]
    if not a.dry_run:
        brains = brains_mod.resolve_all(brains)
    try:
        arms = arms_from_proposal(prop, a.base_checkout, a.arm_checkout)
    except ValueError as e:
        raise SystemExit(str(e))
    out = Path(a.out).expanduser().resolve() if a.out else home / 'emberlab' / 'sim-ab' / a.record
    only = [k for k in (a.only or '').split(',') if k]
    tasks = [t for t in suite['tasks'] if not only or t['key'] in only]
    if only and len(tasks) != len(only):
        raise SystemExit(f'unknown task in --only: {sorted(set(only) - {t["key"] for t in tasks})}')
    approvals = approvals_granted(a)
    # one read of the suite's bytes, used for the printed plan and for the folder's identity: hashing twice could
    # name a digest the printed plan does not show if the suite changed in between (it is the identity, not a label)
    suite_digest = runner.suite_sha(suite)
    plan_text = plan_md(prop, suite, brains, arms, a.repeats, out, len(tasks),
                        host=hosts_mod.kind(a.host, brains), timeout=suite.get('timeout', 1500),
                        unlocked=' or '.join(f for f in approvals or () if f != '--approve-paid'),
                        approvals=approvals, suite_digest=suite_digest)
    print(plan_text, flush=True)
    missing = approval_missing(a, brains)
    if missing:
        raise SystemExit('not started: ' + '; '.join(missing))
    if a.dry_run:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    arms_file = out / 'arms.json'
    foreign = provenance_refusal(out, arms_file)
    if foreign:
        raise SystemExit(foreign)
    # the folder holds one experiment: the arms AND the plan that reads their rows (suite, brains, tasks, repeats),
    # and the first of two concurrent commands to create arms.json owns it (a run id is task.brain.arm.rN, so a
    # shared folder would reuse the other experiment's rows by id)
    spec = arms_spec(arms, dict(suite=suite['name'], suite_sha=suite_digest,
                                brains=sorted(b.label() for b in brains), only=sorted(only),
                                repeats=a.repeats, approvals=approvals))
    why = claim_folder(arms_file, spec)
    if why:
        raise SystemExit(f'{out} holds a different experiment ({why}): use a fresh --out (a shared folder '
                         "would reuse the other experiment's rows by id)")

    # a second command over the same folder compares equal and would run every id again: each run's host.fresh
    # rmtrees the folder the first command's same-id run is using, and both commands are billed for the same runs
    # (#385 review P2). Lock the folder's identity file from the claim until the observation is written.
    lock, busy = claim_lock(arms_file)
    if busy:
        raise SystemExit(busy)
    try:
        # the status guard above reads one snapshot. A resident that closes the record while the plan prints would
        # otherwise be caught only by `write_observation`, after every arm has run and been billed (#385 review P1):
        # re-read it here, before the first run, so a closed record refuses with nothing spent.
        again = load_record(home, a.record)
        if again is None or again.get('status') != 'proposed':
            what = 'gone from the log' if again is None else repr(again.get('status') or 'unstated')
            raise SystemExit(f'SIM record {a.record} is no longer proposed ({what}): nothing was run (it changed '
                             'while the plan printed)')
        # the printed plan is the owner's own record of what they approved; arms.json and plan.json carry the machine
        # identity, and a reader of the folder should not have to reconstruct the human-facing text (#385 review P2).
        # Written after the claim, so two concurrent commands over one folder cannot race for it.
        (out / 'plan.md').write_text(plan_text)
        print(f'lab.py run --suite {a.suite} --arms {arms_file} --out {out}', flush=True)
        # the run namespace starts from `lab.py run`'s own parser defaults (lab.run_defaults) instead of a hand-built
        # field list: a flag added to `run` is then present here at its default, and a flag that had to be copied by
        # hand cannot be missed per flag (#385 review P2). The import is late because lab imports this module.
        if run_defaults is None:
            import lab as lab_mod
            run_defaults = lab_mod.run_defaults
        runs = run_defaults()
        # every field both namespaces carry is copied, not listed: `run`'s defaults supply every field `cmd_run`
        # reads, propose's own parser supplies the ones its record pins, and a flag added to both reaches the runs
        # with no second place to edit. The two the runs must not take from a command line are RUN_OWNS: the arms
        # come from the folder's own file, and `out` is the folder this command resolved (a bare --out may be None
        # here and is checked against the plan's). A run flag propose does not carry keeps `run`'s default, which
        # the coverage test in tests/tools/test_emberlab.py pins (#385 review P2).
        for k, v in list(vars(runs).items()):
            if k not in RUN_OWNS and hasattr(a, k):
                setattr(runs, k, getattr(a, k))
        runs.arms, runs.out = str(arms_file), str(out)
        if run(runs):
            raise SystemExit('a run ended in error: the record stays open (rerun the same command into the same folder)')
        # the digest the folder claimed is the digest the runs must have used, and `lab.py run` re-reads the suite: an edit
        # made between the plan and the rows (a fixture tweaked while the A/B runs) leaves arms.json naming bytes nothing
        # ran, and the record would close on a mix of both. Re-read the bytes, and keep the record open if they moved.
        if runner.suite_sha(suite) != suite_digest:
            raise SystemExit(f'the suite {a.suite} changed while the arms ran: the rows measured other bytes than '
                             f'{out}/arms.json names, so the record stays open (use a fresh --out for the edited suite)')
        # the same re-read for the arms: a checkout committed, rebased or edited during the runs measured other bytes
        # than arms.json/plan.json name, and the record would close on a mix of both (#385 review P1). The runs read
        # their config/env from arms_file itself, so that file is re-read too and must still say what was claimed
        # (#385 review P1, this round).
        moved = arms_drift(arms, spec, arms_file=arms_file)
        if moved:
            raise SystemExit(f'the arms changed while they ran ({moved}): the rows measured other bytes than '
                             f'{out}/arms.json names, so the record stays open (use a fresh --out for the moved checkout)')
        verdict = report_mod.compare(report_mod.load(out), 'base', 'arm')
        print(report_mod.compare_md(verdict), flush=True)
        measured = (f"tools/emberlab propose: suite {suite['name']}, brains "
                    f"{', '.join(b.label() for b in brains)}, tasks {len(tasks)}/{len(suite['tasks'])}"
                    + (f" ({', '.join(t['key'] for t in tasks)})" if only else '')
                    + f", repeats {a.repeats}, {out}")
        if verdict['verdict'].startswith('incomplete'):
            # a brain with no run of one arm cannot judge the proposal, and `observe` is permanent: leave it open
            raise SystemExit(f"the runs do not judge the proposal ({verdict['verdict']}): the record stays open "
                             f'(rerun the same command into the same folder; {measured})')
        # `incomplete` only asks that every brain has both arms. A folder whose first pass errored on half its runs and
        # was then rerun can hold one valid row where the plan asked for three, and report.compare's "fewer than 3
        # repeats" is advisory: the record would close (permanent, append-only) on a 1-sample A/B (#385 review P2). The
        # plan is deterministic, so require every planned run to have a row that did not end in error - the same test
        # `lab.py run` uses to decide a run is done - rather than re-deriving a per-cell count rule.
        planned = [r['id'] for r in runner.plan(suite, brains, arms, max(1, a.repeats), only or None)]
        have = {r['id'] for r in report_mod.load(out) if r.get('state') != 'error'}
        lacking = [i for i in planned if i not in have]
        if lacking:
            raise SystemExit(f'the runs do not judge the proposal ({len(lacking)} of {len(planned)} planned runs have no '
                             f'valid row, e.g. {", ".join(lacking[:3])}): the record stays open (rerun the same command '
                             f'into the same folder; {measured})')
        # sim.observe overwrites eval_context, so the proposing turn's own context is carried forward, then the run's
        prior_ctx = (rec.get('eval_context') or '').strip()
        write_observation(home, a.record, observed_text(verdict),
                          eval_context=f'{prior_ctx} | measured: {measured}' if prior_ctx else measured,
                          prediction_error=a.prediction_error or '', goal=a.goal)
        print(f"SIM record {a.record}: observed", flush=True)
        return 0 if verdict['verdict'].startswith(('better', 'no difference')) else 1
    finally:
        os.close(lock)
