# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 jdominick05
"""Read a lab folder's results.jsonl: a table per brain and arm, and an A/B verdict between two arms.

The verdict follows one rule: an arm is better only if it is better across the board. No brain's pass rate drops, no
brain's false completions rise, and at least one brain improves on one of them. With fewer than three repeats per cell
the verdict says the result is noise-prone; the goal sets are small.
"""
import json
from pathlib import Path
import statistics


def load(out):
    p = Path(out) / 'results.jsonl'
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    latest = {}
    for r in rows:                      # a rerun of one id replaces it
        latest[r['id']] = r
    return list(latest.values())


def cell(rows):
    n = len(rows)
    med = lambda k: statistics.median([r[k] for r in rows if isinstance(r.get(k), (int, float))] or [0])
    spend = [r['spend'] for r in rows if isinstance(r.get('spend'), (int, float))]
    calls = sum(r.get('tool_calls') or 0 for r in rows)
    return dict(runs=n, passed=sum(bool(r.get('check')) for r in rows), finished=sum(bool(r.get('finished')) for r in rows),
                false_completions=sum(bool(r.get('false_completion')) for r in rows),
                errors=sum(r.get('tool_errors') or 0 for r in rows), calls=calls,
                harness_errors=sum(1 for r in rows if r.get('state') == 'error'),
                median_steps=med('steps'), median_seconds=med('seconds'), spend=round(sum(spend), 4) if spend else None,
                repeats=len({r.get('repeat') for r in rows}))


def group(rows, *keys):
    g = {}
    for r in rows:
        g.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    return g


def table(rows):
    lines = ['| brain | arm | passed | false completions | finished | tool errors / calls | median steps | median s | spend $ |',
             '|---|---|---|---|---|---|---|---|---|']
    for (brain, arm), rs in sorted(group(rows, 'brain', 'arm').items(), key=lambda kv: tuple(map(str, kv[0]))):
        c = cell(rs)
        lines.append(f"| {brain} | {arm} | {c['passed']} / {c['runs']} | {c['false_completions']} | {c['finished']} | "
                     f"{c['errors']} / {c['calls']} | {c['median_steps']:g} | {c['median_seconds']:g} | "
                     f"{'-' if c['spend'] is None else c['spend']} |")
    lines += ['', '| task | ' + ' | '.join(sorted({f"{r['brain']} {r['arm']}" for r in rows})) + ' |']
    cols = sorted({(r['brain'], r['arm']) for r in rows}, key=lambda x: (str(x[0]), str(x[1])))
    lines.append('|---|' + '---|' * len(cols))
    by = group(rows, 'task', 'brain', 'arm')
    for task in sorted({r['task'] for r in rows}):
        cells = []
        for b, a in cols:
            rs = by.get((task, b, a), [])
            c = cell(rs) if rs else None
            cells.append('-' if not c else f"{c['passed']}/{c['runs']}" + (f" ({c['false_completions']} false)"
                                                                            if c['false_completions'] else ''))
        lines.append(f'| {task} | ' + ' | '.join(cells) + ' |')
    errs = [r for r in rows if r.get('state') == 'error']
    if errs:
        lines += ['', f'{len(errs)} run(s) ended in error (harness or brain): ' +
                  '; '.join(f"{r['id']}: {str(r.get('error'))[:120]}" for r in errs[:5])]
    return '\n'.join(lines)


def compare(rows, base, arm):
    """per brain: pass rate and false completions, arm against base; the across-the-board verdict"""
    out, better, worse = [], [], []
    thin = False
    for brain in sorted({r['brain'] for r in rows}):
        b = [r for r in rows if r['brain'] == brain and r['arm'] == base]
        x = [r for r in rows if r['brain'] == brain and r['arm'] == arm]
        if not b or not x:
            out.append(dict(brain=brain, missing=base if not b else arm))
            continue
        cb, cx = cell(b), cell(x)
        thin = thin or min(cb['repeats'], cx['repeats']) < 3
        pb, px = cb['passed'] / cb['runs'], cx['passed'] / cx['runs']
        fb, fx = cb['false_completions'] / cb['runs'], cx['false_completions'] / cx['runs']
        row = dict(brain=brain, base=cb, arm=cx, pass_delta=round(px - pb, 3), false_delta=round(fx - fb, 3))
        if px < pb or fx > fb:
            worse.append(brain)
        elif px > pb or fx < fb:
            better.append(brain)
        out.append(row)
    missing = [o for o in out if 'missing' in o]
    verdict = ('incomplete: a brain lacks one of the arms' if missing else
               'worse for ' + ', '.join(worse) if worse else
               'better across the board' if better else 'no difference')
    if thin and not missing:
        verdict += ' (fewer than 3 repeats per cell: noise-prone)'
    return dict(base=base, arm=arm, verdict=verdict, brains=out)


def compare_md(c):
    lines = [f"A/B {c['arm']} against {c['base']}: **{c['verdict']}**", '',
             '| brain | passed base -> arm | false completions base -> arm | tool errors base -> arm | median steps base -> arm |',
             '|---|---|---|---|---|']
    for o in c['brains']:
        if 'missing' in o:
            lines.append(f"| {o['brain']} | no runs for {o['missing']} | | | |")
            continue
        b, x = o['base'], o['arm']
        lines.append(f"| {o['brain']} | {b['passed']}/{b['runs']} -> {x['passed']}/{x['runs']} | "
                     f"{b['false_completions']} -> {x['false_completions']} | {b['errors']} -> {x['errors']} | "
                     f"{b['median_steps']:g} -> {x['median_steps']:g} |")
    return '\n'.join(lines)
