"""CPU test of tools/release/bench_runner.py (the release benchmark runner): no GPU, no model files.

  python -I tests/tools/test_bench_runner.py --scratch DIR

Checks:
  manifest   tools/release/bench_rows.json: every release row names a run, every run has a command, cards (nd / disp /
             both) and a timeout; no machine path in it (placeholders only); every row's doc_line is a non-blank line
             of the doc
  expand     $NAME placeholders fill from the values given; an unknown name is reported, never left silently
  plan       windows per card set in the order nd, disp, both; each within the budget unless one run alone is longer;
             blocked runs are never planned; every unblocked run is in exactly one window
  table      a release row reads its run's log (matched lines and the RESULT); a missing log is NOT RUN; a blocked run and
             the rows not rerun say so"""
import argparse
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ap = argparse.ArgumentParser(); ap.add_argument('--scratch', required=True)
A = ap.parse_args()
os.makedirs(A.scratch, exist_ok=True)
FAILS = []


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)


spec = importlib.util.spec_from_file_location('bench_runner', os.path.join(ROOT, 'tools', 'release', 'bench_runner.py'))
br = importlib.util.module_from_spec(spec); spec.loader.exec_module(br)

# ---- the manifest
m = br.load()
rel = [r for r in m['rows'] if r['tier'] == 'release']
check(rel and all(r['run'] in m['runs'] for r in rel), f'manifest: each of {len(rel)} release rows names a run')
check(all(r['command'] and r['cards'] in br.CARD_ORDER and int(r['timeout']) > 0 for r in m['runs'].values()),
      f'manifest: each of {len(m["runs"])} runs has a command, cards nd / disp / both and a timeout')
text = json.dumps(m)
check(os.path.expanduser('~') not in text and '/home/' not in text, 'manifest: no machine path (placeholders only)')
check({r['tier'] for r in m['rows']} <= {'release', 'derived', 'diagnostic', 'yardstick', 'static', 'not-shipped'}, 'manifest: known tiers only')
doc = open(os.path.join(ROOT, m['doc'])).read().splitlines()
off = [r['id'] for r in m['rows'] if not (0 < r['doc_line'] <= len(doc) and doc[r['doc_line'] - 1].strip())]
check(not off, f'manifest: every row\'s doc_line is a non-blank line of {m["doc"]} ({len(doc)} lines){": " + ", ".join(off[:8]) if off else ""}')

# ---- expand
cmd, missing = br.expand('$PY -I x.py --model $SWIFT --ref $REFS27/a.npz --out $SCRATCH/o', {'PY': 'py', 'SWIFT': '/m/s.gguf', 'SCRATCH': '/s'})
check(cmd == 'py -I x.py --model /m/s.gguf --ref $REFS27/a.npz --out /s/o' and missing == ['REFS27'],
      'expand: known names filled, an unknown one reported and left visible')

# ---- plan
fake = {'runs': {
    'a': dict(command='x', cards='both', timeout=900, est=2000), 'b': dict(command='x', cards='both', timeout=900, est=1500),
    'c': dict(command='x', cards='nd', timeout=900, est=600), 'd': dict(command='x', cards='disp', timeout=900, est=4000),
    'e': dict(command='x', cards='nd', timeout=900, est=100, blocked='refs gone'), 'f': dict(command='x', cards='both', timeout=900, est=900)}}
ws = br.windows(fake, 3000)
cards = [c for c, _ in ws]
flat = [r for _, rs in ws for r in rs]
check(cards == sorted(cards, key=br.CARD_ORDER.index), f'plan: windows in the order nd, disp, both ({cards})')
check(sorted(flat) == ['a', 'b', 'c', 'd', 'f'] and 'e' not in flat, 'plan: every unblocked run once, the blocked one never')
check(all(sum(fake['runs'][r]['est'] for r in rs) <= 3000 or len(rs) == 1 for _, rs in ws),
      'plan: each window within the budget unless one run alone is longer (d: 4000 s alone)')
check(len(br.windows(m, 3000)) > 0 and not any(m['runs'][r].get('blocked') for _, rs in br.windows(m, 3000) for r in rs),
      'plan: the real manifest plans, with no blocked run in it')

# ---- table
out = os.path.join(A.scratch, 'runs'); os.makedirs(out, exist_ok=True)
br_load = br.load
mini = {'doc': m['doc'], 'runs': {'r1': dict(command='x', cards='nd', timeout=60, est=1),
                                  'r2': dict(command='x', cards='nd', timeout=60, est=1),
                                  'r3': dict(command='x', cards='nd', timeout=60, est=1, blocked='refs gone')},
        'rows': [dict(id='one', doc_line=1, tier='release', run='r1', parse='^decode ([\\d.]+) tok/s'),
                 dict(id='two', doc_line=1, tier='release', run='r2', parse='^x'),
                 dict(id='three', doc_line=1, tier='release', run='r3', parse='^x'),
                 dict(id='four', doc_line=1, tier='yardstick', run=None, parse='')]}
open(os.path.join(out, 'r1.log'), 'w').write('noise\ndecode 98.2 tok/s\nRESULT PASS\n')
br.load = lambda: mini
md = os.path.join(A.scratch, 'table.md')
br.table(argparse.Namespace(out=out, md=md))
br.load = br_load
t = open(md).read()
row = {l.split('|')[1].strip(): l for l in t.splitlines() if l.startswith('| ') and not l.startswith('| Row')}
check('decode 98.2 tok/s' in row['one'] and 'RESULT PASS' in row['one'], 'table: a release row shows its matched line and RESULT')
check('NOT RUN' in row['two'], 'table: a release row whose run left no log is NOT RUN')
check('not rerun: refs gone' in row['three'], 'table: a blocked run says why it was not rerun')
check('not rerun for this release (yardstick)' in row['four'], 'table: a yardstick row is marked not rerun')

# ---- a step's verdict and the parse fragments
for name, body, want in (('ok', 'decode 9 ms\nRESULT PASS\n', True), ('done', 'RESULT DONE\n', True), ('bad', 'x\nRESULT FAIL\n', False),
                         ('plain', '96 tokens: median 9.9 ms/token\n', True), ('trace', 'Traceback (most recent call last):\n', False),
                         ('empty', '', False)):
    f = os.path.join(A.scratch, name + '.log'); open(f, 'w').write(body)
    check(br.step_ok(f) is want, f'step_ok: {name} -> {want} (.done only for a step that finished well)')
pats = br.patterns("^plain (\\w+) rep2: .* tok/s ; also 'greedy: .* decode' after a header")
check(any(re.search(p, 'plain prose rep2: 128 tokens = 104.6 tok/s') for p in pats) and any(re.search(p, 'greedy: 64 tokens decode 41 tok/s') for p in pats),
      f'patterns: ^-anchored and quoted fragments both match ({pats})')

print('RESULT', 'FAIL' if FAILS else 'PASS')
sys.exit(1 if FAILS else 0)
