# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 jdominick05
"""Redraw docs/panel.svg, the README's picture of `caminus panel`, from this tree.

  python -I tools/release/panel_shot.py [--out DIR]       # needs Go; writes DIR/panel.svg (default: docs/)

Every release runs it after setting the version (docs/dev/releasing.md, "Making a release"); tests/docs/test_docs.py
fails while the picture names another version than python/caminus/__init__.py.

The picture is the panel's real render() (panel/screenshots_test.go, TestPanelShot) in a GNOME terminal window. What the
tree knows comes from here: the version, the model catalog (caminus.models.CATALOG: names, titles, cards, download
sizes, speeds, notes) and the Qwen Code pin. The host it shows is made up in the test (two cards, one server), and
nothing about this machine goes in: no paths, no files on disk, no live numbers."""
import argparse, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'python'))

import caminus                                   # noqa: E402
from caminus import models                       # noqa: E402
from caminus.cmd import qwen_code                # noqa: E402


def tree_state():
    rows = [dict(name=m.name, title=m.title, arch=m.arch, cards=m.cards,
                 size_gb=round(sum(models.BY_KEY[k].size for k in m.items) / 1e9, 1), speed=m.speed, notes=m.notes)
            for m in models.CATALOG]
    return dict(version=caminus.__version__, qwen_code_pin=qwen_code.PIN, models=rows)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='panel_shot.py', description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs'), help='directory for panel.svg (default: docs/)')
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    state = tree_state()
    env = dict(os.environ, CAMINUS_SHOTS=out, CAMINUS_SHOT_STATE=json.dumps(state))
    r = subprocess.run(['go', 'test', '-run', '^TestPanelShot$', '-count=1', '-v', '.'], cwd=os.path.join(ROOT, 'panel'),
                       env=env, capture_output=True, text=True)
    sys.stdout.write(''.join(l + '\n' for l in r.stdout.splitlines() if 'panel.svg' in l or 'FAIL' in l or 'SKIP' in l))
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-4000:] + r.stderr[-4000:])
        return r.returncode
    print(f'{os.path.join(out, "panel.svg")}: caminus {state["version"]}, {len(state["models"])} models')
    return 0


if __name__ == '__main__':
    sys.exit(main())
