"""The release benchmark runner: reruns the release rows of docs/dev/benchmarks.md on the release commit and writes the
release's table.

  python -I tools/release/bench_runner.py check [--set NAME=PATH ...]
  python -I tools/release/bench_runner.py plan [--window-s 3000]
  python -I tools/release/bench_runner.py run --window N --out DIR [--set ...] [--keep-down]
  python -I tools/release/bench_runner.py table --out DIR [--md FILE]

tools/release/bench_rows.json holds every row of the doc (an inventory of docs/dev/benchmarks.md), each with a tier:
  release      rerun on the release commit (the shipped models' speed, served, accuracy and drafting rows)
  diagnostic   kept with its dated number, marked not rerun (verify traces, matvec censuses, before/after studies, the
               262K soak, the bitwise kernel suite)
  yardstick    llama.cpp's numbers, kept with their date (its build is unchanged), marked not rerun
  derived      cites other rows (a cross-reference or a range): updated by hand from the rows it cites
  static / not-shipped   facts and the models a release does not ship: listed, never run
(0.3.0: the owner's choice, D-341 in the decision log.) A release row names a run; rows that one run measures share it.

check  CPU only: every run's command expands (the model paths from the catalog, the drafters, $ARTIFACTS; others with
       --set NAME=PATH), its tool exists, every --flag it passes appears in the tool's --help, and the files it names
       exist. A release run that fails check is not run until it is fixed in bench_rows.json.
plan   packs the runs into GPU windows per card set (nd: the card with no display, disp: the display card, both), each at
       most --window-s of estimated time (a longer run gets a window of its own), and prints them.
run    runs one window of the plan through the caminus-gpu-window skill's script (preflight, the server stopped and
       restarted, every step under tools/guard.sh); a run with DIR/<run>.done is skipped, so a window resumes.
table  writes the table: for each row, the doc's line, the measured lines matched in its run's log (or why there are
       none), and for the rows not rerun, their tier and the doc's date. Numbers go into docs/dev/benchmarks.md by hand
       from this table, with the release commit named."""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, 'tools', 'release', 'bench_rows.json')
WINDOW = os.path.expanduser('~/.claude/skills/caminus-gpu-window/scripts/gpu-window.sh')
CARD_ORDER = ('nd', 'disp', 'both')


def load():
    with open(MANIFEST) as f:
        return json.load(f)


def defaults():
    """the placeholders the catalog and the config can fill (models, drafters, artifacts, python)"""
    sys.path.insert(0, os.path.join(ROOT, 'python'))
    out = {'PY': sys.executable, 'HOME': os.path.expanduser('~'),
           'ARTIFACTS': os.environ.get('ARTIFACTS') or os.path.join(
               os.environ.get('XDG_DATA_HOME') or os.path.expanduser('~/.local/share'), 'caminus', 'artifacts')}
    try:
        from caminus import models
        from caminus.cmd import manage
        for key, name in (('CODER', 'coder'), ('SWIFT', 'swift-1.5'), ('TCAP', 'thinkingcap'), ('Q38', 'qwen3.8-27b'),
                          ('Q3M', 'qwen3-30b-a3b')):
            e = manage.model_entry(models.BY_NAME[name])
            if e.get('path'): out[key] = e['path']
            for item, p in (e.get('paths') or {}).items():
                if p and item.startswith('dflash2'): out['DFLASH27'] = p
                elif p and item.startswith('dflash') and name == 'coder': out['DFLASH'] = p
    except Exception as x:                       # (no catalog here: every model placeholder needs --set)
        print(f'bench_runner: the catalog did not load ({type(x).__name__}: {x}); pass the model paths with --set', file=sys.stderr)
    return out


def values(sets, scratch=None):
    v = defaults()
    for s in sets or ():
        k, _, p = s.partition('=')
        v[k] = os.path.expanduser(p)
    if scratch: v['SCRATCH'] = scratch
    return v


def expand(cmd, v):
    """the command with $NAME placeholders filled; the names it could not fill"""
    missing = []

    def sub(m):
        k = m.group(1)
        if k in v: return v[k]
        missing.append(k)
        return m.group(0)
    return re.sub(r'\$([A-Z][A-Z0-9_]*)', sub, cmd), sorted(set(missing))


def tool_of(cmd):
    """the script a command runs (the first *.py or tools/benchmarks path), or None"""
    for t in shlex.split(cmd.replace(';', ' ; ')):
        if re.search(r'\.(py|sh)$', t) and ('/' in t) and not t.startswith('$SCRATCH'):
            return t
    return None


def check(a):
    m, v = load(), values(a.set, scratch='/nonexistent-scratch')
    bad = 0
    helps = {}
    blocked = 0
    for rid, r in sorted(m['runs'].items()):
        if r.get('blocked'):
            blocked += 1
            print(f'BLOCKED {rid} [{r["cards"]}, est {r["est"]} s]: {r["blocked"]}')
            continue
        cmd, missing = expand(r['command'], v)
        probs = []
        if missing and set(missing) - {'SCRATCH'}: probs.append('unfilled: ' + ', '.join(sorted(set(missing) - {'SCRATCH'})) + ' (--set NAME=PATH)')
        tool = tool_of(cmd)
        if tool and not os.path.isabs(tool) and not os.path.exists(os.path.join(ROOT, tool)):
            probs.append(f'tool missing: {tool}')
        elif tool and tool.endswith('.py') and not tool.startswith('/'):
            if tool not in helps:
                try:
                    p = subprocess.run([sys.executable, '-I', tool, '--help'], cwd=ROOT, capture_output=True, text=True, timeout=120,
                                       env=dict(os.environ, HIP_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES=''))
                    helps[tool] = (p.stdout + p.stderr) if 'usage' in (p.stdout + p.stderr).lower() else None
                except subprocess.TimeoutExpired:
                    helps[tool] = None
            h = helps[tool]
            if h is None:
                probs.append(f'{tool}: no --help (flags not checked)')
            else:
                for flag in sorted(set(re.findall(r'(?<![\w-])(--[a-z][a-z0-9-]*)', cmd))):
                    if not re.search(re.escape(flag) + r'(?![a-z0-9-])', h): probs.append(f'{tool}: {flag} not in its --help')
        for tok in shlex.split(cmd.replace(';', ' ; ')):
            if tok.startswith('/') and not tok.startswith('/nonexistent-scratch') and not os.path.exists(tok) and '*' not in tok:
                probs.append(f'missing file: {tok}')
        hard = [p for p in probs if 'flags not checked' not in p]
        bad += bool(hard)
        print(('FAIL ' if hard else ('NOTE ' if probs else 'PASS ')) + f'{rid} [{r["cards"]}, est {r["est"]} s]' + (': ' + '; '.join(probs) if probs else ''))
    print(f'RESULT {"FAIL" if bad else "PASS"}: {len(m["runs"])} runs, {bad} to fix in {os.path.relpath(MANIFEST, ROOT)}, {blocked} blocked')
    return 1 if bad else 0


def windows(m, window_s):
    out = []
    for card in CARD_ORDER:
        cur, t = [], 0
        for rid, r in sorted(((k, x) for k, x in m['runs'].items() if x['cards'] == card and not x.get('blocked')),
                             key=lambda kv: (-kv[1]['est'], kv[0])):
            if cur and t + r['est'] > window_s:
                out.append((card, cur)); cur, t = [], 0
            cur.append(rid); t += r['est']
        if cur: out.append((card, cur))
    return out


def plan(a):
    m = load()
    ws = windows(m, a.window_s)
    tot = 0
    for i, (card, rids) in enumerate(ws):
        est = sum(m['runs'][r]['est'] for r in rids); tot += est
        print(f'window {i}: {card}, {len(rids)} run(s), estimate {est / 60:.0f} min: {", ".join(rids)}')
    print(f'{len(ws)} windows, estimate {tot / 3600:.1f} h of GPU time (the estimates are rough; bash-timing history refines them)')
    for rid, r in sorted(m['runs'].items()):
        if r.get('blocked'): print(f'blocked, not planned: {rid} ({r["est"]} s): {r["blocked"]}')
    return 0


def run(a):
    m = load()
    ws = windows(m, a.window_s)
    if not 0 <= a.window < len(ws): raise SystemExit(f'bench_runner: --window {a.window}: the plan has windows 0 .. {len(ws) - 1}')
    card, rids = ws[a.window]
    os.makedirs(a.out, exist_ok=True)
    steps, todo = [], []
    for rid in rids:
        if os.path.exists(os.path.join(a.out, rid + '.done')):
            print(f'skip {rid}: done'); continue
        r = m['runs'][rid]
        cmd, missing = expand(r['command'], values(a.set, scratch=os.path.join(a.out, 'scratch', rid)))
        if missing: raise SystemExit(f'bench_runner: {rid}: unfilled {", ".join(missing)} (run check)')
        os.makedirs(os.path.join(a.out, 'scratch', rid), exist_ok=True)
        steps.append(f'{card}:{int(r["timeout"])}:{cmd}'); todo.append(rid)
    if not steps:
        print('window done already'); return 0
    wout = os.path.join(a.out, f'window-{a.window:02d}')
    argv = [WINDOW, '--no-new-code', '--repo', ROOT, '--out', wout] + (['--keep-down'] if a.keep_down else []) + steps
    print('running:', ' '.join(shlex.quote(x) for x in argv[:6]), f'... {len(steps)} step(s)', flush=True)
    rc = subprocess.call(argv)
    for i, rid in enumerate(todo, 1):
        log = os.path.join(wout, f'{i:02d}-{card}.log')
        if os.path.exists(log):
            os.replace(log, os.path.join(a.out, rid + '.log'))
            if os.path.exists(log + '.out'): os.replace(log + '.out', os.path.join(a.out, rid + '.log.out'))
            open(os.path.join(a.out, rid + '.done'), 'w').write(f'window {a.window} rc {rc}\n')
    print(f'window {a.window} ended rc {rc}; logs in {a.out}')
    return rc


def table(a):
    m = load()
    doc = open(os.path.join(ROOT, m['doc'])).read().splitlines()
    lines = ['| Row | Tier | The doc (before) | Measured on the release commit |', '|---|---|---|---|']
    for r in m['rows']:
        if r['tier'] in ('not-shipped', 'static'): continue
        before = doc[r['doc_line'] - 1].strip() if 0 < r['doc_line'] <= len(doc) else ''
        before = re.sub(r'\s+', ' ', before)[:160].replace('|', '/')
        if r['tier'] == 'derived':
            got = 'derived: update by hand from the rows it cites'
        elif r['tier'] != 'release' or not r['run']:
            got = f'not rerun for this release ({r["tier"]})'
        elif m['runs'][r['run']].get('blocked'):
            got = 'not rerun: ' + m['runs'][r['run']]['blocked']
        else:
            log = os.path.join(a.out, r['run'] + '.log')
            if not os.path.exists(log):
                got = 'NOT RUN'
            else:
                text = open(log, errors='replace').read() + (open(log + '.out', errors='replace').read() if os.path.exists(log + '.out') else '')
                hits = []
                for pat in re.findall(r'\^[^;]+', r['parse'] or ''):
                    try:
                        hits += [l.strip() for l in text.splitlines() if re.search(pat.strip().split(' ; ')[0], l)][:3]
                    except re.error:
                        pass
                res = re.findall(r'RESULT (PASS|FAIL|DONE)', text)
                got = ('; '.join(dict.fromkeys(hits))[:300] or 'no line matched: read the log') + (f' (RESULT {res[-1]})' if res else '')
        lines.append(f'| {r["id"]} | {r["tier"]} | {before} | {got.replace("|", "/")} |')
    out = '\n'.join(lines) + '\n'
    if a.md:
        open(a.md, 'w').write(out); print(f'wrote {a.md}')
    else:
        sys.stdout.write(out)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check'); c.add_argument('--set', action='append', default=[])
    p = sub.add_parser('plan'); p.add_argument('--window-s', type=int, default=3000)
    r = sub.add_parser('run'); r.add_argument('--window', type=int, required=True); r.add_argument('--out', required=True)
    r.add_argument('--window-s', type=int, default=3000); r.add_argument('--set', action='append', default=[])
    r.add_argument('--keep-down', action='store_true')
    t = sub.add_parser('table'); t.add_argument('--out', required=True); t.add_argument('--md')
    a = ap.parse_args(argv)
    return dict(check=check, plan=plan, run=run, table=table)[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
