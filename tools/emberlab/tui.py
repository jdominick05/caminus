#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 jdominick05
"""Drive a terminal program in a hidden tmux window and look at it: keys in, exact screens out.

  tui.py --out DIR [--size 160x48] (--cmd 'PROGRAM ARGS' | --session NAME --lab-out LABDIR [--panel PATH]) STEP...

Steps, in order:
  key:NAME        a tmux key (Enter, Escape, BTab = shift+tab, C-p, Up, Tab, x ...)
  type:TEXT       literal text
  wait:REGEX[@S]  until the screen matches (default 15 s); fails the run on timeout
  expect:REGEX    the screen matches now      absent:REGEX   the screen does not match now
  sleep:S         pause                       resize:WxH     resize the window (the program gets SIGWINCH)
                                              shot:NAME      DIR/NAME.ans (with colour), .txt and .svg (.png with --png)

--session opens the Ember chat on a lab session's home (lab.py open), through a CLI wrapper like tools/ember_demo.py's;
--cmd runs anything, e.g. `caminus ember ui` on your real account. Against a real account pass --read-only: then only
look-around keys pass (navigation, Escape, shift+tab, ^P, and Enter right after typing one of a few read-only slash
commands); a step that would send a goal, start, stop or destroy anything is refused before the run begins.

tmux runs on its own socket inside DIR at a fixed size and is killed at the end, whatever happened. No tmux: SKIP.
Exit 0 when every step passed, 1 when a step failed, 2 for usage, 3 for SKIP.
"""
import argparse
import html
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import unicodedata

READ_ONLY_KEYS = {'BTab', 'C-p', 'Escape', 'Up', 'Down', 'Left', 'Right', 'Tab', 'PageUp', 'PageDown', 'Home', 'End', 'q'}
READ_ONLY_SLASH = {'/help', '/theme', '/context', '/sessions', '/approvals', '/diff', '/memory', '/goals'}


class Tmux:
    def __init__(self, sock, w, h):
        self.sock, self.w, self.h = str(sock), w, h

    def tm(self, *a, check=True):
        return subprocess.run(['tmux', '-S', self.sock, *a], capture_output=True, text=True, check=check)

    def start(self, cmd, env=None):
        # the program sees a true-colour xterm (tmux would say screen, and colour libraries then drop to 256 colours)
        envs = ' '.join(f'{k}={shlex.quote(v)}' for k, v in (env or {}).items())
        self.tm('-f', '/dev/null', 'new-session', '-d', '-s', 'lab', '-x', str(self.w), '-y', str(self.h),
                f'env {envs} sh -c {shlex.quote(cmd)}' if envs else cmd)
        self.tm('set-option', '-t', 'lab', 'remain-on-exit', 'on', check=False)

    def keys(self, *k):
        self.tm('send-keys', '-t', 'lab', *k)

    def text(self, s):
        self.tm('send-keys', '-t', 'lab', '-l', s)

    def screen(self, ansi=False):
        return self.tm('capture-pane', '-p', '-t', 'lab', *(['-e'] if ansi else [])).stdout

    def kill(self):
        self.tm('kill-server', check=False)


# ---- ANSI -> SVG (the cells as the program drew them; enough SGR for a TUI: 16 / 256 / true colour, bold, reverse)

BASE16 = ['#15181c', '#ec6f60', '#8fd081', '#ecc95e', '#7fb2e5', '#c49bd8', '#5fc4b0', '#ece8e1',
          '#39424b', '#f08a7e', '#a6dc9a', '#f2d886', '#9cc4ec', '#d3b3e3', '#86d3c3', '#ffffff']
FG, BG = '#ece8e1', '#15181c'


def c256(n):
    if n < 16:
        return BASE16[n]
    if n < 232:
        n -= 16
        f = lambda v: 0 if v == 0 else 55 + v * 40
        return '#%02x%02x%02x' % (f(n // 36), f(n // 6 % 6), f(n % 6))
    v = 8 + (n - 232) * 10
    return '#%02x%02x%02x' % (v, v, v)


def cells(ansi):
    """rows of (char, fg, bg, bold) per cell"""
    rows = []
    st = dict(fg=None, bg=None, bold=False, rev=False)
    for line in ansi.split('\n'):
        row, i = [], 0
        while i < len(line):
            m = re.match(r'\x1b\[([0-9;:]*)m', line[i:])
            if m:
                ps = [int(p) if p.isdigit() else 0 for p in re.split('[;:]', m.group(1) or '0')]
                j = 0
                while j < len(ps):
                    p = ps[j]
                    if p == 0:
                        st = dict(fg=None, bg=None, bold=False, rev=False)
                    elif p == 1:
                        st['bold'] = True
                    elif p == 22:
                        st['bold'] = False
                    elif p == 7:
                        st['rev'] = True
                    elif p == 27:
                        st['rev'] = False
                    elif 30 <= p <= 37 or 90 <= p <= 97:
                        st['fg'] = BASE16[p - 30 if p < 90 else p - 82]
                    elif 40 <= p <= 47 or 100 <= p <= 107:
                        st['bg'] = BASE16[p - 40 if p < 100 else p - 92]
                    elif p in (38, 48) and j + 1 < len(ps):
                        key = 'fg' if p == 38 else 'bg'
                        if ps[j + 1] == 5 and j + 2 < len(ps):
                            st[key] = c256(ps[j + 2])
                            j += 2
                        elif ps[j + 1] == 2 and j + 4 < len(ps):
                            st[key] = '#%02x%02x%02x' % tuple(ps[j + 2:j + 5])
                            j += 4
                    elif p == 39:
                        st['fg'] = None
                    elif p == 49:
                        st['bg'] = None
                    j += 1
                i += m.end()
                continue
            m = re.match(r'\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07', line[i:])
            if m:
                i += m.end()
                continue
            ch = line[i]
            fg, bg = st['fg'] or FG, st['bg'] or BG
            if st['rev']:
                fg, bg = bg, fg
            row.append((ch, fg, bg, st['bold']))
            if unicodedata.east_asian_width(ch) in 'WF':
                row.append(('', fg, bg, st['bold']))
            i += 1
        rows.append(row)
    return rows


def svg(ansi, cols, rows_n, cw=8.4, ch=17):
    rows = cells(ansi)[:rows_n]
    W, H = cols * cw, rows_n * ch
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" viewBox="0 0 {W:.0f} {H:.0f}">',
           f'<rect width="100%" height="100%" fill="{BG}"/>',
           '<g font-family="DejaVu Sans Mono, Noto Sans Mono, monospace" font-size="14">']
    for y, row in enumerate(rows):
        x = 0
        while x < len(row):                      # backgrounds, merged per run
            bg = row[x][2]
            x2 = x
            while x2 < len(row) and row[x2][2] == bg:
                x2 += 1
            if bg != BG:
                out.append(f'<rect x="{x * cw:.1f}" y="{y * ch}" width="{(x2 - x) * cw:.1f}" height="{ch}" fill="{bg}"/>')
            x = x2
        x = 0
        while x < len(row):                      # text, merged per run of the same style
            fg, bold = row[x][1], row[x][3]
            x2 = x
            while x2 < len(row) and (row[x2][1], row[x2][3]) == (fg, bold):
                x2 += 1
            t = ''.join(c[0] for c in row[x:x2])
            if t.strip():
                out.append(f'<text x="{x * cw:.1f}" y="{y * ch + ch - 4}" fill="{fg}"{" font-weight=\"bold\"" if bold else ""}'
                           f' xml:space="preserve">{html.escape(t)}</text>')
            x = x2
    out.append('</g></svg>')
    return '\n'.join(out)


def png(svg_path, w, h):
    chrome = shutil.which('google-chrome') or shutil.which('chromium')
    if not chrome:
        return None
    out = svg_path.with_suffix('.png')
    subprocess.run([chrome, '--headless=new', '--disable-gpu', '--hide-scrollbars', f'--window-size={int(w)},{int(h)}',
                    f'--screenshot={out}', svg_path.as_uri()], capture_output=True, timeout=60)
    return out if out.exists() else None


# ---- the panel on a lab session

WRAPPER = '''#!/bin/sh
# the panel's CLI calls, pointed at one lab session's resident home (written by tools/emberlab/tui.py)
if [ "$1 $2" = "env agent" ] || [ "$1 $2" = "ember agent" ]; then
  shift 2; op="$1"; shift 2
  exec {runtime} "$op" lab --home {home} "$@"
fi
if [ "$1 $2" = "ember run" ] || [ "$1 $2" = "env run" ]; then
  shift 2; [ "$1" = "-s" ] && shift
  {run}
fi
echo "emberlab: '$*' is not part of a lab session" >&2; exit 2
'''


def session_cmd(a, out):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sessions
    s = sessions._load(a.lab_out, a.session)
    h = sessions._host(s)
    import runner
    rt = h.runtime(runner.Arm(s['arm'], s['checkout']))
    if s['host'] == 'agent':
        rt = h.ssh + ['python3', '-I', s['program']]
    wrapper = out / 'lab-cli.sh'
    # `ember run -s SCRIPT` (path completion, attachments, ! shell mode): the script in the session's home, as its user
    if s['host'] == 'agent':
        run = ' '.join(shlex.quote(x) for x in h.ssh) + ' "cd ' + shlex.quote(s['home']) + ' && $*"; exit $?'
    else:
        run = 'cd ' + shlex.quote(s['home']) + ' && exec sh -c "$*"'
    wrapper.write_text(WRAPPER.format(runtime=' '.join(shlex.quote(x) for x in rt), home=shlex.quote(s['home']), run=run))
    wrapper.chmod(0o700)
    # a local session's own bridge, so its approvals and questions show; an agent-host session's stay on that host
    bridge = Path(s['home']).parent / 'bridge' if s['host'] == 'local' else out / 'bridge'
    for sub in ('req', 'res'):
        (bridge / sub).mkdir(parents=True, exist_ok=True)
    panel = a.panel
    if not panel:
        panel = str(out / 'caminus-panel')
        repo = Path(__file__).resolve().parents[2]
        b = subprocess.run(['go', 'build', '-o', panel, '.'], cwd=repo / 'panel', capture_output=True, text=True)
        if b.returncode:
            raise SystemExit('building the panel failed (needs Go, or pass --panel):\n' + b.stderr[-1500:])
    return ' '.join(shlex.quote(x) for x in [panel, 'env', '-demo', '-agent', 'lab', '-bridge', str(bridge),
                                               '-audit', str(out / 'audit.jsonl'), '-cli', json.dumps([str(wrapper)])])


def check_read_only(steps):
    bad, prev_slash = [], False
    for st in steps:
        kind, _, arg = st.partition(':')
        if kind == 'key':
            ok = arg in READ_ONLY_KEYS or (arg == 'Enter' and prev_slash)
            if not ok:
                bad.append(st)
            prev_slash = False
        elif kind == 'type':
            if arg.strip() not in READ_ONLY_SLASH:
                bad.append(st)
            prev_slash = arg.strip() in READ_ONLY_SLASH
        else:
            prev_slash = prev_slash and kind in ('sleep', 'wait', 'expect', 'absent', 'shot')
    return bad


def run(a, steps):
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not shutil.which('tmux'):
        print('SKIP: tmux is not installed')
        return 3
    w, h = map(int, a.size.lower().split('x'))
    cmd = a.cmd or session_cmd(a, out)
    t = Tmux(out / 'tmux.sock', w, h)
    log, fails = [], []
    env = {'TERM': 'xterm-256color', 'COLORTERM': 'truecolor'}
    if a.session:
        # a lab panel never reads or writes your own panel settings (theme, glyphs, last session): its own folders
        for k in ('XDG_CONFIG_HOME', 'XDG_STATE_HOME', 'XDG_CACHE_HOME'):
            env[k] = str(out / 'xdg' / k.split('_')[1].lower())
            Path(env[k]).mkdir(parents=True, exist_ok=True)
    t.start(cmd, env=env)
    try:
        for st in steps:
            kind, _, arg = st.partition(':')
            ok = True
            if kind == 'key':
                t.keys(arg)
                time.sleep(a.step_delay)
            elif kind == 'type':
                t.text(arg)
                time.sleep(a.step_delay)
            elif kind == 'sleep':
                time.sleep(float(arg))
            elif kind == 'resize':
                w, h = map(int, arg.lower().split('x'))
                t.tm('resize-window', '-t', 'lab', '-x', str(w), '-y', str(h))
                t.w, t.h = w, h
                time.sleep(a.step_delay)
            elif kind == 'wait':
                rx, _, secs = arg.rpartition('@') if re.search(r'@[\d.]+$', arg) else (arg, '', '15')
                end = time.time() + float(secs)
                while not re.search(rx, t.screen()) and time.time() < end:
                    time.sleep(0.2)
                ok = bool(re.search(rx, t.screen()))
            elif kind == 'expect':
                ok = bool(re.search(arg, t.screen()))
            elif kind == 'absent':
                ok = not re.search(arg, t.screen())
            elif kind == 'shot':
                ansi = t.screen(ansi=True)
                (out / f'{arg}.ans').write_text(ansi)
                (out / f'{arg}.txt').write_text(t.screen())
                sp = out / f'{arg}.svg'
                sp.write_text(svg(ansi, t.w, t.h))
                if a.png:
                    png(sp, t.w * 8.4, t.h * 17)
            else:
                raise SystemExit(f'unknown step {st!r}')
            log.append(('PASS ' if ok else 'FAIL ') + st)
            print(log[-1], flush=True)
            if not ok:
                fails.append(st)
                (out / 'failed.txt').write_text(t.screen())
                if not a.keep_going:
                    break
    finally:
        t.kill()
    print('RESULT', 'FAIL' if fails else 'PASS')
    return 1 if fails else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', default='160x48')
    ap.add_argument('--cmd')
    ap.add_argument('--session')
    ap.add_argument('--lab-out')
    ap.add_argument('--panel')
    ap.add_argument('--read-only', action='store_true', help='a real account: look-around keys only')
    ap.add_argument('--png', action='store_true', help='also a PNG of each shot (headless Chrome, when installed)')
    ap.add_argument('--keep-going', action='store_true')
    ap.add_argument('--step-delay', type=float, default=0.3)
    ap.add_argument('steps', nargs='*')
    a = ap.parse_args(argv)
    if bool(a.cmd) == bool(a.session) or (a.session and not a.lab_out):
        ap.error('one of --cmd or --session (with --lab-out)')
    if a.read_only and (bad := check_read_only(a.steps)):
        print('refused (--read-only): ' + ', '.join(bad))
        return 2
    return run(a, a.steps)


if __name__ == '__main__':
    raise SystemExit(main())
