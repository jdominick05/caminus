#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 jdominick05
"""The Ember lab: develop and test Ember (the agent harness) against real agents and a scorer that does not trust them.

Batch (measure):
  lab.py run --suite basic --brain scripted:solve --out DIR [--arm NAME[=CHECKOUT] ...] [--arms FILE] [--repeats N]
             [--host local|agent] [--only KEY,...] [--jobs N] [--timeout S]
  lab.py report DIR [--md FILE]                one table per brain x arm, and per task
  lab.py compare DIR --base A --arm B          the A/B verdict: better only if better across the board
  lab.py propose --sim-home HOME --record ID --brain BRAIN [--config JSON | --arm-checkout DIR] ...
                                               a `proposed` SIM record becomes two arms; the runs close it (Haiku
                                               through the owner's relay is the tool A/B brain; billed runs need
                                               --approve-paid; see proposal.py)

Sessions (design with the agent):
  lab.py open NAME --brain BRAIN --out DIR [--host agent] [--fixture SUITE]   a lab agent, its daemon up
  lab.py say NAME TEXT [--goal ID] [--wait S]  a goal (or a steer to goal ID); prints what it did
  lab.py turn NAME [--system]                  driven brain: the request waiting for YOU, the model, to answer
  lab.py reply NAME --call TOOL --args JSON | --text TEXT [--timings JSON]
  lab.py events NAME [--all] / status NAME
  lab.py close NAME                            stop its daemon, job workers (boxes, Chrome) and driven brain; KILLs what ignores TERM

Brains (brains.py): scripted:NAME, driven (you answer each model call; :caminus plays a caminus serve, whose replies carry
`reply --timings`), openai:URL[#MODEL], openai+relay:URL#MODEL,
anthropic:PORT#MODEL (through the owner's relay; the lab never handles a key). Hosts (hosts.py): local (scratch folders as
you; no real model unless --unsafe-local) and agent (the agent account, ~/lab/; the default for real brains).

Results: DIR/results.jsonl (one record per run, appended; a rerun of `run` skips the ids already there, except runs that
ended in error), DIR/runs/ID/
(once.out, events.jsonl, check.out, run.json). Records name the arm's commit and whether its tree was dirty. Nothing
here belongs in the repository: results hold model output and may hold your paths.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import brains as brains_mod    # noqa: E402
import hosts as hosts_mod      # noqa: E402
import proposal as proposal_mod  # noqa: E402
import report as report_mod    # noqa: E402
import runner                  # noqa: E402
import sessions                # noqa: E402


def cmd_run(a):
    suite = runner.load_suite(a.suite)
    arms = runner.load_arms(a.arm, a.arms)
    bs = [brains_mod.parse(s, a.parallel) for s in a.brain]
    # an `openai:URL` brain with no `#MODEL` labels itself `?@URL` until its endpoint is asked, and that label is
    # in every run id and in plan.json: ask first, so both name the model that answers (#385 review P2)
    brains_mod.resolve_all(bs)
    host_kind = hosts_mod.kind(a.host, bs)
    # one rule for both commands (hosts.local_refusal), so `run` and `propose` cannot answer the same command line
    # two ways (#385 review)
    why = hosts_mod.local_refusal(a.host, bs, unsafe_local=a.unsafe_local)
    if why:
        raise SystemExit(why)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    host = hosts_mod.make(host_kind, out, runner.REPO)
    only = [k for k in (a.only or '').split(',') if k] or None
    runs = runner.plan(suite, bs, arms, a.repeats, only)
    # a run that ended in error (an SSH drop, an upload failure, a brain outage) runs again; a finished one is skipped
    done = {r['id'] for r in report_mod.load(out) if r.get('state') != 'error'} if not a.rerun else set()
    todo = [r for r in runs if r['id'] not in done]
    paid = [b.label() for b in bs if b.paid]
    print(f'{len(runs)} runs ({len(todo)} to do) on the {host_kind} host: suite {suite["name"]}, '
          f'brains {", ".join(b.label() for b in bs)}, arms {", ".join(a_.name for a_ in arms)}'
          + (f'; billed through the relay: {", ".join(paid)}' if paid else ''), flush=True)
    for b in bs:
        b.start()
    host.prepare(arms)
    # the suite digest is part of the plan a reader must be able to reproduce: an in-place edit of a fixture or a
    # task keeps `suite`'s name and every task key, so the name alone does not say which bytes ran (#385 review P1)
    (out / 'plan.json').write_text(json.dumps(dict(suite=suite['name'], suite_sha=runner.suite_sha(suite),
                                                   host=host_kind, repeats=a.repeats,
                                                   brains=[b.label() for b in bs], arms=[x.plan_arm() for x in arms],
                                                   # the flag that unlocked the host, as given: the record says which
                                                   # permission a run held, not just that it held one (#385 review P2)
                                                   approvals=proposal_mod.approvals_granted(a),
                                                   runs=[r['id'] for r in runs]), indent=1))
    lock = threading.Lock()
    gates = {b.spec: threading.Semaphore(b.parallel) for b in bs}
    timeout = a.timeout or suite.get('timeout', 1500)

    def one(r):
        with gates[r['brain'].spec]:
            rec = runner.execute(r, host, suite, timeout, out)
        with lock:
            with (out / 'results.jsonl').open('a') as f:
                f.write(json.dumps(rec) + '\n')
            fc = ' FALSE COMPLETION' if rec['false_completion'] else ''
            print(f"  {rec['id']:44s} {rec.get('state', '?'):12s} check={'pass' if rec.get('check') else 'FAIL'}{fc} "
                  f"steps={rec.get('steps')} {rec.get('seconds')}s calls={rec.get('tool_calls')} errors={rec.get('tool_errors')}"
                  + (f" ({rec['error'][:100]})" if rec.get('error') and rec.get('state') == 'error' else ''), flush=True)
        return rec

    try:
        with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            list(ex.map(one, todo))
    finally:
        for b in bs:
            b.stop()
    rows = [r for r in report_mod.load(out) if r['id'] in {x['id'] for x in runs}]
    print(report_mod.table(rows))
    return 0 if rows and all(r.get('state') != 'error' for r in rows) else 1


def cmd_report(a):
    rows = report_mod.load(a.dir)
    if not rows:
        raise SystemExit(f'no results in {a.dir}')
    md = report_mod.table(rows)
    print(md)
    if a.md:
        Path(a.md).write_text(md + '\n')
    return 0


def cmd_compare(a):
    c = report_mod.compare(report_mod.load(a.dir), a.base, a.arm)
    print(json.dumps(c, indent=1) if a.json else report_mod.compare_md(c))
    return 0 if c['verdict'].startswith(('better', 'no difference')) else 1


def add_run_args(p, required=True):
    """`run`'s flags, in one place: `main` and `run_defaults` (which `propose` starts runs with) read the same list.

    A flag added here reaches every caller; before, `propose` built the run namespace by hand and a flag added to
    `run` silently did not reach a propose-started run (#385 review P2).
    """
    p.add_argument('--suite', required=required, help='a suite name under benchmarks/ember/suites, or a folder')
    p.add_argument('--brain', action='append', required=required)
    p.add_argument('--arm', action='append', help='NAME[=CHECKOUT] (default: base = this checkout)')
    p.add_argument('--arms', help='JSON {NAME: {checkout, config, env}}, or the arms.json `propose` writes')
    p.add_argument('--out', required=required)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--only', help='comma-separated task keys')
    p.add_argument('--host', choices=('local', 'agent'))
    p.add_argument('--unsafe-local', action='store_true', help='let a real model run as you (the local host)')
    p.add_argument('--jobs', type=int, default=4, help='runs at once (each brain still caps its own: --parallel)')
    p.add_argument('--parallel', type=int, help="override every brain's parallel cap")
    p.add_argument('--timeout', type=int, help='seconds per goal (default: the suite\'s)')
    p.add_argument('--rerun', action='store_true', help='run every id again, even those already in results.jsonl')
    return p


def run_defaults():
    """Every field `cmd_run` reads, at its parser default; `propose` sets the ones its record pins.

    The defaults come from `add_run_args` itself (not a copy of them), so a flag added to `run` is present with its
    own default and `propose` cannot drift from `run`'s parser.
    """
    ap = argparse.ArgumentParser(add_help=False)
    return add_run_args(ap, required=False).parse_args([])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = add_run_args(sub.add_parser('run'))
    p = sub.add_parser('report')
    p.add_argument('dir')
    p.add_argument('--md')
    p = sub.add_parser('compare')
    p.add_argument('dir')
    p.add_argument('--base', required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--json', action='store_true')

    p = sub.add_parser('propose')
    p.add_argument('--sim-home', required=True,
                   help="the resident's home whose SIM log holds the record (it may be live: each write is one "
                        'BEGIN IMMEDIATE transaction, WAL, busy_timeout, the path `caminus ember agent sim` uses)')
    p.add_argument('--record', required=True, help='the id of a `proposed` SIM record')
    p.add_argument('--suite', default='basic')
    p.add_argument('--brain', action='append', required=True)
    p.add_argument('--repeats', type=int, default=3,
                   help='runs per cell (default 3: report.compare calls fewer noise-prone)')
    p.add_argument('--only', help='comma-separated task keys')
    p.add_argument('--host', choices=('local', 'agent'))
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--parallel', type=int)
    p.add_argument('--out', help='default: SIM-HOME/emberlab/sim-ab/RECORD')
    p.add_argument('--config', help="the proposal as config.json keys for arm, JSON")
    p.add_argument('--env', help="the proposal as environment variables for arm, JSON")
    p.add_argument('--base-config', help='config.json keys for base (rarely: base runs as it runs now)')
    p.add_argument('--base-env')
    p.add_argument('--arm-checkout', help='the checkout with the proposed code (default: this checkout)')
    p.add_argument('--base-checkout')
    p.add_argument('--dry-run', action='store_true',
                   help='print the plan and create nothing (also asks no brain endpoint for its model: `?@URL` is '
                        'named as it stands and resolved when the run starts)')
    p.add_argument('--approve-paid', action='store_true',
                   help="run a brain billed through the owner's relay (the plan names it first)")
    p.add_argument('--unsafe-local', action='store_true', help='let a real model run as you (the local host)')
    p.add_argument('--prediction-error', help='where the prediction was wrong, filled in after the run; a review '
                                             'says whether the miss is understood, so leave it empty unless it is - '
                                             "sim.observe treats a non-empty value as the record's final answer on "
                                             'why the prediction missed and the log is append-only, so a typo is '
                                             'permanent')
    p.add_argument('--goal', help="the goal id the observation lives in: the record's own (a different one is "
                                  'refused, since an observation is appended to the record\'s goal)')

    p = sub.add_parser('open')
    p.add_argument('name')
    p.add_argument('--brain', required=True, help='driven, driven:judge, driven:caminus, openai:URL[#MODEL], anthropic:PORT#MODEL, ...')
    p.add_argument('--out', required=True)
    p.add_argument('--host', choices=('local', 'agent'), default='agent')
    p.add_argument('--arm', default='base')
    p.add_argument('--checkout', help='the runtime to run (default: this checkout)')
    p.add_argument('--fixture', help="a suite whose fixture/ goes into the work folder")
    p.add_argument('--ttl', type=int, default=4 * 3600, help='seconds the daemon (and a driven brain) live')
    p.add_argument('--replace', action='store_true')
    p.add_argument('--unsafe-local', action='store_true')
    for name in ('say', 'turn', 'reply', 'events', 'status', 'close'):
        p = sub.add_parser(name)
        p.add_argument('name')
        p.add_argument('--out', required=True)
        p.add_argument('--width', type=int, default=500, help='characters shown per item')
        if name == 'say':
            p.add_argument('text', help="the goal or message ('-': stdin)")
            p.add_argument('--goal', help='steer this running goal instead of starting one')
            p.add_argument('--wait', type=int, help='seconds to wait for the goal to settle (default 900; driven 0)')
        if name == 'turn':
            p.add_argument('--system', action='store_true', help='print the system prompt too')
            p.add_argument('--last', type=int, default=6, help='messages shown from the end')
            p.add_argument('--wait', type=float, default=10)
        if name == 'reply':
            p.add_argument('--call')
            p.add_argument('--args')
            p.add_argument('--text')
            p.add_argument('--timings', help="a driven:caminus brain sends these with the answer (caminus serve's "
                                             'timings JSON: predicted_per_second, prompt_per_second, ...)')
            p.add_argument('--settle', type=float, default=1.5, help='seconds to wait before printing what followed')
        if name == 'events':
            p.add_argument('--all', action='store_true')
    p = sub.add_parser('serve-driven', help=argparse.SUPPRESS)
    p.add_argument('ctl')
    p.add_argument('--ttl', type=int, default=14400)
    p.add_argument('--judge', action='store_true')
    p.add_argument('--caminus', action='store_true')
    a = ap.parse_args(argv)
    if a.cmd == 'serve-driven':
        brains_mod.serve_driven(a.ctl, a.judge, a.ttl, a.caminus)
        return 0
    return dict(run=cmd_run, report=cmd_report, compare=cmd_compare,
                propose=lambda a: proposal_mod.cmd_propose(a, cmd_run, run_defaults),
                open=sessions.open_session, say=sessions.say,
                turn=sessions.turn, reply=sessions.reply, events=sessions.events, status=sessions.status,
                close=sessions.close)[a.cmd](a)


if __name__ == '__main__':
    raise SystemExit(main())
