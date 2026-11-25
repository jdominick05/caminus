"""The docs check (CPU, no imports beyond the standard library): what the user and contributor docs promise exists.

  python -I tests/docs/test_docs.py

- every relative link in README.md, CONTRIBUTING.md, GLOSSARY.md, docs/*.md and docs/dev/*.md resolves to a file in the tree,
  and a link's #anchor to a heading of that file (GitHub's slug rule);
- every `caminus COMMAND` the user docs (README.md, docs/*.md) show is a command `caminus` has
  (caminus.__main__.COMMANDS, read from the file, not imported);
- the README's setup table ("Your cards | What runs | Start with") holds exactly the rows of caminus.models.SETUPS, the
  one source caminus wizard and Ember's default read (python/caminus/models.py, parsed, not imported);
- every cell of the README's scoreboard ("Model | Prefill GEMMs ...") is "not measured" or a link to its row's section in
  docs/dev/yardsticks.md;
- (the paths a doc names, AGENTS.md's command lines among them, are checked by tools/entropy.py, `refs`);
- docs/panel.svg, the README's picture of the control panel, shows the version python/caminus/__init__.py holds (a
  release sets the version and redraws it: python -I tools/release/panel_shot.py; a nightly's .dev version is skipped).
Prints PASS / FAIL per check and RESULT PASS / RESULT FAIL (n)."""
import ast, glob, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
fails = []


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what)
    if not ok: fails.append(what)


def slug(h):
    """GitHub's heading anchor: lower case, punctuation but hyphens and spaces dropped, spaces to hyphens"""
    h = re.sub(r'[`*_]', '', h.strip().lower())
    h = re.sub(r'[^\w\- ]', '', h)
    return h.replace(' ', '-')


def anchors(path):
    out = set()
    in_code = False
    for line in open(path, encoding='utf-8'):
        if line.startswith('```'): in_code = not in_code
        if not in_code and re.match(r'#{1,6} ', line): out.add(slug(line.lstrip('#')))
    return out


def strip_code(text):
    return re.sub(r'```.*?```', '', text, flags=re.S)


docs = [p for p in ['README.md', 'CONTRIBUTING.md', 'GLOSSARY.md'] if os.path.exists(os.path.join(ROOT, p))]
docs += sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, 'docs', '*.md')) + glob.glob(os.path.join(ROOT, 'docs', 'dev', '*.md')))
nlinks, bad = 0, []
for d in docs:
    text = strip_code(open(os.path.join(ROOT, d), encoding='utf-8').read())
    for target in re.findall(r'\]\(([^)\s]+)\)', text) + re.findall(r'<img src="([^"]+)"', text):
        if re.match(r'[a-z]+:', target): continue          # https:, mailto:
        path, _, frag = target.partition('#')
        full = os.path.normpath(os.path.join(ROOT, os.path.dirname(d), path)) if path else os.path.join(ROOT, d)
        nlinks += 1
        if not os.path.exists(full): bad.append(f'{d}: {target} (no such file)'); continue
        if frag and full.endswith('.md') and frag not in anchors(full): bad.append(f'{d}: {target} (no heading #{frag})')
check(not bad, f'links: {nlinks} relative links in {len(docs)} docs resolve' + (f'; broken: {bad}' if bad else ''))

src = open(os.path.join(ROOT, 'python', 'caminus', '__main__.py'), encoding='utf-8').read()
commands = set(ast.literal_eval(re.search(r'^COMMANDS = (\(.*?\))', src, re.S | re.M).group(1)))
user = ['README.md'] + [d for d in docs if d.startswith('docs/') and not d.startswith('docs/dev/')]
seen, unknown = set(), []
for d in user:
    for cmd in re.findall(r'(?<![\w/.-])caminus ([a-z][a-z-]*)', open(os.path.join(ROOT, d), encoding='utf-8').read()):
        seen.add(cmd)
        if cmd not in commands: unknown.append(f'{d}: caminus {cmd}')
check(not unknown, f'commands: the user docs show {len(seen)} caminus commands, all real' + (f'; unknown: {sorted(set(unknown))}' if unknown else ''))

# the README's setup table is caminus.models.SETUPS row by row (a plain literal there: read, not imported)
msrc = open(os.path.join(ROOT, 'python', 'caminus', 'models.py'), encoding='utf-8').read()
node = next(n for n in ast.parse(msrc).body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'SETUPS')
want = [f"| {s['row']} | {s['what']} | {s['start_text']} |" for s in ast.literal_eval(node.value)]
readme = open(os.path.join(ROOT, 'README.md'), encoding='utf-8').read().splitlines()
head = next((i for i, l in enumerate(readme) if l.startswith('| Your cards | What runs | Start with |')), None)
got = readme[head + 2:head + 2 + len(want) + 1] if head is not None else []
check(got[:len(want)] == want and (len(got) == len(want) or not got[-1].startswith('|')),
      'README setup table: the rows of caminus.models.SETUPS' + ('' if got[:len(want)] == want else
      ': write these rows under its header:\n' + '\n'.join(want)))

# the README's scoreboard: each cell is "not measured" or a link to its measurement in docs/dev/yardsticks.md (the
# anchors are checked with the other links above), so the README holds verdicts only and the numbers one file
head = next((i for i, l in enumerate(readme) if l.startswith('| Model | Prefill GEMMs')), None)
rows = []
if head is not None:
    for l in readme[head + 2:]:
        if not l.startswith('|'): break
        rows.append([c.strip() for c in l.strip('|').split('|')[1:]])
loose = [c for r in rows for c in r if c != 'not measured' and not re.fullmatch(r'\[[^\]]+\]\(docs/dev/yardsticks\.md#[\w-]+\)', c)]
check(bool(rows) and not loose, f'README scoreboard: {len(rows)} rows, every cell a yardsticks.md link or "not measured"'
      + ('' if rows else ': no table under "| Model | Prefill GEMMs"') + (f'; other cells: {loose}' if loose else ''))

ver = re.search(r"^__version__ = '([^']+)'", open(os.path.join(ROOT, 'python', 'caminus', '__init__.py'), encoding='utf-8').read(), re.M).group(1)
shot = os.path.join(ROOT, 'docs', 'panel.svg')
# the version is the header row's (y 13) version-shaped word: the old header put it after "control panel", the new one
# ("CAMINUS · panel") draws it at the right edge
drawn = re.search(r'<text x="[\d.]+" y="13"[^>]*>(\d+\.\d+\.\d+[^<]*)</text>', open(shot, encoding='utf-8').read()) if os.path.exists(shot) else None
drawn = drawn.group(1) if drawn else None
if 'dev' in ver:
    check(True, f'panel picture: version {ver} is a nightly\'s, not checked (the picture shows {drawn})')
else:
    check(drawn == ver, f'panel picture: docs/panel.svg shows caminus {drawn}, the tree is {ver}'
          + ('' if drawn == ver else ': redraw it with python -I tools/release/panel_shot.py'))

print(f'RESULT PASS' if not fails else f'RESULT FAIL ({len(fails)})')
sys.exit(0 if not fails else 1)
