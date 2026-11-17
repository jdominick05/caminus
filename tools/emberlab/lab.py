#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 jdominick05
"""The Ember lab: develop and test Ember (the agent harness) against real agents and a scorer that does not trust them.

Batch (measure):
  lab.py run --suite basic --brain scripted:solve --out DIR [--arm NAME[=CHECKOUT] ...] [--arms FILE] [--repeats N]
             [--host local|agent] [--only KEY,...] [--jobs N] [--timeout S]
  lab.py report DIR [--md FILE]                one table per brain x arm, and per task
  lab.py compare DIR --base A --arm B          the A/B verdict: better only if better across the board

Sessions (design with the agent):
  lab.py open NAME --brain BRAIN --out DIR [--host agent] [--fixture SUITE]   a lab agent, its daemon up
  lab.py say NAME TEXT [--goal ID] [--wait S]  a goal (or a steer to goal ID); prints what it did
  lab.py turn NAME [--system]                  driven brain: the request waiting for YOU, the model, to answer
  lab.py reply NAME --call TOOL --args JSON | --text TEXT [--timings JSON]
  lab.py events NAME [--all] / status NAME / close NAME

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
import report as report_mod    # noqa: E402
import runner                  # noqa: E402
import sessions                # noqa: E402


def cmd_run(a):
    suite = runner.load_suite(a.suite)
    arms = runner.load_arms(a.arm, a.arms)
    bs = [brains_mod.parse(s, a.parallel) for s in a.brain]
    host_kind = a.host or ('agent' if any(b.real for b in bs) else 'local')
    if host_kind == 'local' and any(b.real for b in bs) and not a.unsafe_local:
        raise SystemExit('a real model on the local host gets a shell as you: use --host agent (or --unsafe-local)')
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
    (out / 'plan.json').write_text(json.dumps(dict(suite=suite['name'], host=host_kind, repeats=a.repeats,
                                                   brains=[b.label() for b in bs], arms=[x.describe() for x in arms],
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('run')
    p.add_argument('--suite', required=True, help='a suite name under benchmarks/ember/suites, or a folder')
    p.add_argument('--brain', action='append', required=True)
    p.add_argument('--arm', action='append', help='NAME[=CHECKOUT] (default: base = this checkout)')
    p.add_argument('--arms', help='JSON {NAME: {checkout, config, env}}')
    p.add_argument('--out', required=True)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--only', help='comma-separated task keys')
    p.add_argument('--host', choices=('local', 'agent'))
    p.add_argument('--unsafe-local', action='store_true', help='let a real model run as you (the local host)')
    p.add_argument('--jobs', type=int, default=4, help='runs at once (each brain still caps its own: --parallel)')
    p.add_argument('--parallel', type=int, help="override every brain's parallel cap")
    p.add_argument('--timeout', type=int, help='seconds per goal (default: the suite\'s)')
    p.add_argument('--rerun', action='store_true', help='run every id again, even those already in results.jsonl')
    p = sub.add_parser('report')
    p.add_argument('dir')
    p.add_argument('--md')
    p = sub.add_parser('compare')
    p.add_argument('dir')
    p.add_argument('--base', required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--json', action='store_true')

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
    return dict(run=cmd_run, report=cmd_report, compare=cmd_compare, open=sessions.open_session, say=sessions.say,
                turn=sessions.turn, reply=sessions.reply, events=sessions.events, status=sessions.status,
                close=sessions.close)[a.cmd](a)


if __name__ == '__main__':
    raise SystemExit(main())
