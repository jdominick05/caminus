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
(a `anthropic:` or `openai+relay:` brain; the tool A/B brain is Haiku there), and `--unsafe-local` or `--unsafe-free`
for a real model on the local host. `--dry-run` stops after the plan.

  python -I tools/emberlab/lab.py propose --sim-home HOME --record ID --suite basic \\
      --brain anthropic:8091#claude-haiku-5-5 --repeats 3 --config '{"sim": false}'

The record says *what* to change and the command line says *how*, because a proposal in words is not a switch. A
proposal about code names `--arm-checkout DIR`: `base` runs this checkout and `arm` runs DIR, so DIR should be this
tree plus the change, and the diff between the two arms is then only that change.
"""
from pathlib import Path
import json
import os
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brains as brains_mod     # noqa: E402
import hosts as hosts_mod       # noqa: E402
import report as report_mod     # noqa: E402
import runner                   # noqa: E402

RESIDENT = Path(runner.REPO) / 'python'


def _rt(store):
    """A runtime-shaped object for the SIM module: it needs a store, a home and a config, not a live model."""
    return type('RT', (), {'store': store, 'home': store.home, 'config': {}})()


def _open(home):
    """The session's store, and the resident's SIM module from this lab's own tree (one log format, one writer)."""
    if str(RESIDENT) not in sys.path:
        sys.path.insert(0, str(RESIDENT))
    from caminus.ember.resident import sim
    from caminus.ember.resident.store import Store
    store = Store(Path(home).expanduser().resolve())
    return sim, store, _rt(store)


def load_record(home, rid):
    """One record's newest version from a session's home, or None."""
    sim, store, rt = _open(home)
    try:
        return sim._get(rt, rid)
    finally:
        store.close()


def write_observation(home, rid, observed, eval_context='', prediction_error='', goal=None):
    """Append the outcome to the record the arms measured: one new version, status 'observed'.

    The "a closed record stays closed" rule is the resident's own, and it already holds across the race: `observe`
    reads the newest version and appends in one `_atomic` transaction (`BEGIN IMMEDIATE`), and `_record` refuses a
    version of an already-observed record. So a resident that closes the record while the arms run is not raced
    over; what the command owes is to report that refusal as one, not as a traceback (#385 review P2).
    """
    sim, store, rt = _open(home)
    try:
        try:
            return sim.observe(rt, goal, rid, observed, eval_context, prediction_error,
                               cfg=dict(model='emberlab', provider='', effort=''))
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


ARM_KEYS = ('checkout', 'commit', 'dirty', 'config', 'env')
PLAN_KEYS = ('suite', 'brains', 'only', 'repeats')


def arms_spec(arms, plan=None):
    """What a folder records as its identity: each arm as `Arm.describe` writes it, and the plan that reads the rows.

    The commit and the dirty flag belong to the identity, not just the path: a run id is task.brain.arm.rN and
    carries no record id, and `lab.py run` skips every id already in results.jsonl, so a checkout that moved (a
    commit, a rebase) would close a new record with the previous rows (#385 review: reproduced by committing
    between two runs into one folder).

    The suite, brains, `--only` and repeats belong to it for the same reason: one folder holds one experiment. A
    first run that errored partway leaves its rows, and a rerun with another `--brain` (or another suite or task
    set) would pass the arms check and close its record with numbers this command never chose (#385 review P1).
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
    """
    tmp = arms_file.with_name('.%s.%d.tmp' % (arms_file.name, os.getpid()))
    tmp.write_text(json.dumps(spec, indent=1))
    try:
        os.link(str(tmp), str(arms_file))
        return ''
    except FileExistsError:
        pass
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    try:
        prior = json.loads(arms_file.read_text())
    except ValueError:
        # this command writes the file whole or not at all, so a file that is not JSON was made by something else
        return (f'{arms_file} is not JSON and was not written by this command: use a fresh --out (or fix or remove '
                'the file)')
    return arms_mismatch(prior, spec)


def approval_missing(a, brains):
    """What the owner still has to allow before this run starts: the plan alone never starts one.

    The local-host rule is `hosts.local_refusal`, the same function `lab.py run` calls, and its own `--unsafe-free`
    is the same permission there: this command used to accept either flag and then set `unsafe_local` on the run
    namespace, so `propose --unsafe-free` started a paid real brain as the user where `run --unsafe-free` was
    refused (#385 review). A flag that is not on the command line is read with getattr, so a caller building the
    namespace by hand (a test, `lab.py run`) does not have to add it.
    """
    paid = [b.label() for b in brains if getattr(b, 'paid', False)]
    missing = []
    if paid and not a.approve_paid:
        missing.append("--approve-paid (billed through the owner's relay: " + ', '.join(paid) + ')')
    why = hosts_mod.local_refusal(a.host, brains, unsafe_local=getattr(a, 'unsafe_local', False),
                                  unsafe_free=getattr(a, 'unsafe_free', False))
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


def plan_md(prop, suite, brains, arms, repeats, out, tasks, host=None, timeout=None, unlocked=''):
    """The plan the owner reads before anything runs, and the one thing that names everything that will run.

    The host decides where a real model runs (as the user, or on the agent account) and the timeout is how long one
    goal may take, so a plan without them did not name everything (#385 review). `host` is the resolved kind
    (`hosts.kind`), not the flag: `--host local` and a real brain with no flag resolve to the same place.
    """
    paid = [b.label() for b in brains if getattr(b, 'paid', False)]
    real = [b.label() for b in brains if getattr(b, 'real', False)]
    runs = len(arms) * len(brains) * tasks * max(1, repeats)
    host = host or hosts_mod.kind(None, brains)
    timeout = suite.get('timeout', 1500) if timeout is None else timeout
    lines = [f"proposal {prop['id']} · {prop.get('problem_group') or 'unclassified'}", '',
             f"- change: {(prop.get('change') or '').strip()[:400]}",
             f"- predicted: {(prop.get('predicted') or '').strip()[:400]}",
             f"- suite: {suite['name']} ({tasks} tasks), {repeats} repeat(s)",
             f"- host: {host}" + (' (a real model runs as you here: '
                                    + (unlocked or '--unsafe-local or --unsafe-free') + ')'
                                    if host == 'local' and real else ''),
             f"- timeout: {timeout}s per goal",
             f"- brains: {', '.join(b.label() for b in brains)}",
             f"- arms: {', '.join(a.name + ' @ ' + str(a.checkout) + ' ' + a.commit for a in arms)}",
             f"- difference: {arm_mechanism(prop, arms)}",
             f"- runs: {runs} in {out}",
             '- verdict: an arm is better only if it is better across the board (report.compare)']
    if paid:
        lines.append(f"- **billed through the owner's relay: {', '.join(paid)}** (the relay's daily cap is the guard)")
    return '\n'.join(lines)


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
            f"{o['brain']} {o['base']['median_steps']:g} -> {o['arm']['median_steps']:g}" for o in cells))
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


def cmd_propose(a, run):
    """`lab.py propose ...`: the plan first, the runs only with the owner's flags, and the record closed by numbers."""
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
    try:
        prop = dict(rec, config=_json_arg(a.config, 'config'), env=_json_arg(a.env, 'env'),
                    base_config=_json_arg(a.base_config, 'base-config'), base_env=_json_arg(a.base_env, 'base-env'))
    except ValueError as e:
        raise SystemExit(str(e))
    suite = runner.load_suite(a.suite)
    brains = [brains_mod.parse(s, a.parallel) for s in a.brain]
    try:
        arms = arms_from_proposal(prop, a.base_checkout, a.arm_checkout)
    except ValueError as e:
        raise SystemExit(str(e))
    out = Path(a.out).expanduser().resolve() if a.out else home / 'emberlab' / 'sim-ab' / a.record
    only = [k for k in (a.only or '').split(',') if k]
    tasks = [t for t in suite['tasks'] if not only or t['key'] in only]
    if only and len(tasks) != len(only):
        raise SystemExit(f'unknown task in --only: {sorted(set(only) - {t["key"] for t in tasks})}')
    unlocked = ('--unsafe-free' if getattr(a, 'unsafe_free', False) else '') or (
        '--unsafe-local' if getattr(a, 'unsafe_local', False) else '')
    print(plan_md(prop, suite, brains, arms, a.repeats, out, len(tasks),
                  host=hosts_mod.kind(a.host, brains), timeout=suite.get('timeout', 1500), unlocked=unlocked),
          flush=True)
    missing = approval_missing(a, brains)
    if missing:
        raise SystemExit('not started: ' + '; '.join(missing))
    if a.dry_run:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    arms_file = out / 'arms.json'
    # the folder holds one experiment: the arms AND the plan that reads their rows (suite, brains, tasks, repeats),
    # and the first of two concurrent commands to create arms.json owns it (a run id is task.brain.arm.rN, so a
    # shared folder would reuse the other experiment's rows by id)
    spec = arms_spec(arms, dict(suite=suite['name'], brains=sorted(b.label() for b in brains), only=sorted(only),
                                repeats=a.repeats))
    why = claim_folder(arms_file, spec)
    if why:
        raise SystemExit(f'{out} holds a different experiment ({why}): use a fresh --out (a shared folder '
                         "would reuse the other experiment's rows by id)")
    # the status guard above reads one snapshot. A resident that closes the record while the plan prints would
    # otherwise be caught only by `write_observation`, after every arm has run and been billed (#385 review P1):
    # re-read it here, before the first run, so a closed record refuses with nothing spent.
    again = load_record(home, a.record)
    if again is None or again.get('status') != 'proposed':
        what = 'gone from the log' if again is None else repr(again.get('status') or 'unstated')
        raise SystemExit(f'SIM record {a.record} is no longer proposed ({what}): nothing was run (it changed '
                         'while the plan printed)')
    print(f'lab.py run --suite {a.suite} --arms {arms_file} --out {out}', flush=True)
    # both flags are carried as given: cmd_run asks hosts.local_refusal about the resolved host, and that rule
    # reads either name. Folding --unsafe-free into unsafe_local here would hide which flag allowed the run.
    runs = type('Run', (), dict(suite=a.suite, brain=list(a.brain), arm=None, arms=str(arms_file), out=str(out),
                                repeats=a.repeats, only=a.only, host=a.host,
                                unsafe_local=a.unsafe_local, unsafe_free=a.unsafe_free,
                                jobs=a.jobs, parallel=a.parallel, timeout=None, rerun=False))()
    if run(runs):
        raise SystemExit('a run ended in error: the record stays open (rerun the same command into the same folder)')
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
    # sim.observe overwrites eval_context, so the proposing turn's own context is carried forward, then the run's
    prior_ctx = (rec.get('eval_context') or '').strip()
    write_observation(home, a.record, observed_text(verdict),
                      eval_context=f'{prior_ctx} | measured: {measured}' if prior_ctx else measured,
                      prediction_error=a.prediction_error or '', goal=a.goal)
    print(f"SIM record {a.record}: observed", flush=True)
    return 0 if verdict['verdict'].startswith(('better', 'no difference')) else 1
