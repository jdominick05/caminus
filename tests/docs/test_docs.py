"""The docs check (CPU, no imports beyond the standard library): what the user and contributor docs promise exists.

  python -I tests/docs/test_docs.py

- every relative link in README.md, CONTRIBUTING.md, GLOSSARY.md, docs/*.md and docs/dev/*.md resolves to a file in the tree,
  and a link's #anchor to a heading of that file (GitHub's slug rule);
- every `caminus COMMAND` the user docs (README.md, docs/*.md) show is a command `caminus` has
  (caminus.__main__.COMMANDS, read from the file, not imported);
- every script AGENTS.md's command lines name (tools/..., tests/..., benchmarks/..., build.sh) exists (the public AGENTS.md:
  packaging/public/AGENTS.md in the development tree, AGENTS.md in the public one).
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

ag = os.path.join(ROOT, 'packaging', 'public', 'AGENTS.md')
ag = ag if os.path.exists(ag) else os.path.join(ROOT, 'AGENTS.md')
block = '\n'.join(re.findall(r'```\n(.*?)```', open(ag, encoding='utf-8').read(), re.S))
paths = sorted(set(re.findall(r'(?<![\w/$.-])((?:tools|tests|benchmarks|panel)/[\w./-]+\.(?:py|sh)|build\.sh)', block)))
missing = [p for p in paths if not os.path.exists(os.path.join(ROOT, p.lstrip('./')))]
check(paths and not missing, f'AGENTS.md: the {len(paths)} scripts its commands name exist' + (f'; missing: {missing}' if missing else ''))

ver = re.search(r"^__version__ = '([^']+)'", open(os.path.join(ROOT, 'python', 'caminus', '__init__.py'), encoding='utf-8').read(), re.M).group(1)
shot = os.path.join(ROOT, 'docs', 'panel.svg')
drawn = re.search(r'>control</text><text[^>]*>panel</text><text[^>]*>([^<]+)</text>', open(shot, encoding='utf-8').read()) if os.path.exists(shot) else None
drawn = drawn.group(1) if drawn else None
if 'dev' in ver:
    check(True, f'panel picture: version {ver} is a nightly\'s, not checked (the picture shows {drawn})')
else:
    check(drawn == ver, f'panel picture: docs/panel.svg shows caminus {drawn}, the tree is {ver}'
          + ('' if drawn == ver else ': redraw it with python -I tools/release/panel_shot.py'))

print(f'RESULT PASS' if not fails else f'RESULT FAIL ({len(fails)})')
sys.exit(0 if not fails else 1)
