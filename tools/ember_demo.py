#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 jdominick05
"""An interactive demo of `caminus ember`: the real chat TUI on the real resident runtime, with a scripted model.

    python3 -I tools/ember_demo.py [--scratch DIR] [--panel PATH] [--fast] [--keep]

Nothing here needs the agent account, SSH, a GPU, a model server or an API key. It makes a resident home in the
scratch folder (default: $XDG_CACHE_HOME/caminus/env-demo), runs the resident daemon there (`ember agent --local`), and
answers its model calls from a local scripted server that plays a goal through: it plans with a todo list, writes its
own done check, looks around in a terminal, writes and edits a file (diff cards), asks for a root command (the
approval card), asks you a question, verifies and finishes past the gate and the judge. Every reply streams, with
thinking, so the live tail shows.

What to try (the panel's /help lists the rest):
- Enter sends the goal waiting in the composer (or type your own); the empty chat shows Ember, the mascot
- while it works, type a message (it is delivered at its next step) or /steer TEXT (it says what it changed)
- the approval card: Esc, then a a approves (in the demo nothing runs as root: there is no harness sudo here, so the
  approval is recorded and the command is not executed), d denies, /deny TEXT says what to do instead
- the question: Esc, then a answers it (a number picks a choice); unanswered, it times out after two minutes and the
  agent takes its recommended option
- Ctrl+G goals, Ctrl+A approvals, /context, /audit, /memory, /recap, /glyphs, Ctrl+E opens every card
- /halt then /halt clear: the emergency stop

The scripted model is a stand-in, not a model: it ignores what you ask beyond quoting it. --fast streams quicker.
--keep leaves the scratch home (its goals, journal and events) for a look afterwards.
"""
import argparse
import http.server
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
BOOT = "import sys;sys.path.insert(0,sys.argv[1]);from caminus.__main__ import main;sys.argv=sys.argv[1:];sys.exit(main())"
AGENT = 'demo'   # not 'resident': the panel's per-agent files (last session, recap) stay the real one's
PACE = [0.035]

# ---------------------------------------------------------------------------------------------------------- the story


def goal_text(msgs):
    for m in msgs:
        if m.get('role') == 'user':
            c = m.get('content')
            if isinstance(c, list):
                c = ' '.join(p.get('text', '') for p in c if isinstance(p, dict))
            return ' '.join(str(c).split())[:160]
    return 'the goal'


def last_human(msgs):
    """The newest user message after the goal's first one (a message, a steer, an answer), or None."""
    users = [m for m in msgs if m.get('role') == 'user']
    if len(users) < 2:
        return None
    last = msgs[-1]
    if last.get('role') != 'user':
        return None
    c = last.get('content')
    return ' '.join(str(c).split())[:200] if c else None


def steps_done(msgs):
    done = set()
    for m in msgs:
        for c in m.get('tool_calls') or []:
            cid = c.get('id', '')
            if cid.startswith('demo-') and cid.split('-')[1].isdigit():
                done.add(int(cid.split('-')[1]))
    return done


def last_tool_result(msgs):
    for m in reversed(msgs):
        if m.get('role') == 'tool':
            return str(m.get('content', ''))
        if m.get('role') == 'assistant':
            return ''
    return ''


def last_call(msgs):
    for m in reversed(msgs):
        for c in m.get('tool_calls') or []:
            return (c.get('function') or {}).get('name')
    return None


def script(goal):
    """The goal's steps: (thinking, text, tool name, args). Each runs once, in order."""
    return [
        (f'The goal is: "{goal}". A gate that a shell can check: the report exists and has its result.',
         'I will start with my done check: report.md exists and has a Result section.',
         'define_done', dict(check='report.md exists and has a "## Result" section', gate='grep -q "^## Result" report.md')),
        (f'The goal is: "{goal}". First a done check the gate can test, then a short todo list.',
         'My plan.',
         'todo', dict(items=[dict(text='look around the home', state='in_progress'),
                             dict(text='write the report', state='pending'),
                             dict(text='verify and finish', state='pending')])),
        ('Look before writing: what is in the home, what machine this is.',
         '',
         'terminal', dict(command='ls -la && uname -sr && date', yield_ms=1500)),
        ('Write the report. The gate will still fail: no Result section yet.',
         'Writing a first draft.',
         'write', dict(path='report.md', content=f'# Report\n\nGoal: {goal}\n\n## Notes\n\n- looked around the home\n- '
                                                  'nothing here needs root\n')),
        ('Now the section the gate looks for.',
         'Adding the result.',
         'edit', dict(path='report.md', old='## Notes\n', new='## Result\n\nDone in the demo: the steps above ran for '
                                                                 'real in this scratch home.\n\n## Notes\n')),
        ('A root request, to show the approval card. Approvals do not block: I carry on while it waits.',
         'One thing needs root; I will ask and keep going.',
         'ask_root', dict(command='apt-get install -y ripgrep', reason='a faster search for the next goal (demo: '
                                                                         'nothing runs as root here)')),
        ('And a question, with my recommendation, so an unanswered question has a default.',
         '',
         'ask_user', dict(question='Should the report also list the files in the home?',
                          choices=['yes, list them', 'no, keep it short'], recommended='no, keep it short')),
        ('The gate passes now; record the evidence for the judge.',
         'Checking my work.',
         'verify', dict(evidence='grep finds "## Result" in report.md (the gate); report.md was written and edited above')),
        ('Everything on the list is done.',
         '',
         'todo', dict(items=[dict(text='look around the home', state='done'),
                             dict(text='write the report', state='done'),
                             dict(text='verify and finish', state='done')])),
        ('Finish. The judge reads the evidence, not my word.',
         '',
         'finish', dict(summary=f'Wrote report.md for "{goal}", with a Result section the gate checks.')),
    ]


def next_turn(body):
    msgs = body.get('messages') or []
    goal = goal_text(msgs)
    steps = script(goal)
    done = steps_done(msgs)
    human = last_human(msgs)
    result = last_tool_result(msgs)
    prefix = ''
    if human and not human.startswith('['):
        prefix = f'Noted: "{human[:120]}". I will take that into account. '
    elif human:
        prefix = 'Got it. '
    # After the script's own finish, react the way an agent should to what came back.
    last = last_call(msgs)
    stamp = int(time.time() * 1000)
    if last in ('finish', 'wait', 'verify') and 9 in done:
        if last == 'verify':
            return ('Checked again; finish.', prefix + 'Finishing.', 'finish',
                    dict(summary=f'Wrote report.md for "{goal}", with a Result section the gate checks.'), f'demo-x-finish-{stamp}')
        if 'still waiting for your human' in result:
            return ('A request is still open; finishing must wait for it.',
                    prefix + 'Waiting for your answer before I finish.',
                    'wait', dict(until='message', timeout=150), f'demo-x-wait-{stamp}')
        return ('A steer or a new message came in since I last checked: run the check again before finishing.',
                prefix + 'Re-checking against your latest message.',
                'verify', dict(evidence='grep still finds "## Result" in report.md (the gate); re-checked after the '
                               'latest message'), f'demo-x-verify-{stamp}')
    nxt = next((i for i in range(len(steps)) if i not in done), None)
    if nxt is None:
        return ('Done with the script; finish.', prefix + 'Finishing.', 'finish',
                dict(summary=f'Wrote report.md for "{goal}".'), f'demo-x-finish-{stamp}')
    think, text, name, args = steps[nxt]
    return think, prefix + text, name, args, f'demo-{nxt}-{name}'


# ---------------------------------------------------------------------------------------------------------- the server


class Model(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, value, status=200):
        raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.send_json(dict(data=[dict(id='demo', object='model')], n_ctx=131072,
                            default_generation_settings=dict(n_ctx=131072)))

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
        if self.path.endswith('/count_tokens'):
            return self.send_json(dict(input_tokens=len(json.dumps(body)) // 4))
        msgs = body.get('messages') or [{}]
        first = str(msgs[0].get('content', ''))
        usage = dict(prompt_tokens=len(json.dumps(msgs)) // 4, completion_tokens=40)
        if first.startswith('JUDGE'):
            verdict = json.dumps(dict(verdict='met', reason='report.md exists, the gate passed and verify names it'))
            return self.reply_text(body, verdict, usage)
        if not body.get('tools'):
            # /btw, a compaction summary or any other tool-less call
            return self.reply_text(body, 'In the demo this answer is scripted: the agent would answer from the goal\'s '
                                   'context here, with no tools and nothing kept.', usage)
        think, text, name, args, cid = next_turn(body)
        call = dict(index=0, id=cid, type='function', function=dict(name=name, arguments=json.dumps(args)))
        if not body.get('stream'):
            msg = dict(role='assistant', content=text, reasoning_content=think, tool_calls=[
                dict(id=cid, type='function', function=call['function'])])
            return self.send_json(dict(choices=[dict(message=msg, finish_reason='tool_calls')], usage=usage))
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        try:
            for kind, words in (('reasoning_content', think), ('content', text)):
                for w in words.split(' ') if words else []:
                    self.chunk(dict(choices=[dict(index=0, delta={kind: w + ' '})]))
                    time.sleep(PACE[0])
            self.chunk(dict(choices=[dict(index=0, delta=dict(tool_calls=[call]))]))
            self.chunk(dict(choices=[dict(index=0, delta={}, finish_reason='tool_calls')], usage=usage))
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass                         # Esc interrupted the turn: the runtime hung up

    def reply_text(self, body, text, usage):
        if not body.get('stream'):
            return self.send_json(dict(choices=[dict(message=dict(role='assistant', content=text),
                                                     finish_reason='stop')], usage=usage))
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        for w in text.split(' '):
            self.chunk(dict(choices=[dict(index=0, delta=dict(content=w + ' '))]))
            time.sleep(PACE[0] / 2)
        self.chunk(dict(choices=[dict(index=0, delta={}, finish_reason='stop')], usage=usage))
        self.wfile.write(b'data: [DONE]\n\n')
        self.wfile.flush()

    def chunk(self, value):
        self.wfile.write(('data: ' + json.dumps(value) + '\n\n').encode())
        self.wfile.flush()


# ---------------------------------------------------------------------------------------------------------- the rest

WRAPPER = '''#!/bin/sh
# caminus ember demo: the panel's CLI calls, pointed at the demo's resident home (written by tools/ember_demo.py)
PY={py}; ROOT={root}; HOME_DIR={home}
if [ "$1 $2" = "env agent" ]; then
  shift 2; op="$1"; name="$2"; shift 2
  exec "$PY" -I -c '{boot}' "$ROOT/python" env agent --local "$op" "$name" --home "$HOME_DIR" "$@"
fi
if [ "$1 $2" = "env run" ]; then
  shift 2; [ "$1" = "-s" ] && shift
  cd "$HOME_DIR" && exec sh -c "$*"
fi
echo "caminus ember demo: '$*' is not part of the demo" >&2; exit 2
'''


def q(s):
    return "'" + str(s).replace("'", "'\\''") + "'"


def agent(home, op, *args, stdin=None):
    return subprocess.run([sys.executable, '-I', '-c', BOOT, str(ROOT / 'python'), 'env', 'agent', '--local', op, AGENT,
                           '--home', str(home), *args], input=stdin, capture_output=True, text=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    cache = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache')
    ap.add_argument('--scratch', type=Path, default=cache / 'caminus' / 'env-demo')
    ap.add_argument('--panel', help='a built caminus-panel (default: build panel/ into the scratch folder)')
    ap.add_argument('--fast', action='store_true', help='stream the replies quicker')
    ap.add_argument('--keep', action='store_true', help='keep the scratch home afterwards')
    ap.add_argument('--no-tui', action='store_true', help='set everything up, print the panel command and wait (tests)')
    a = ap.parse_args(argv)
    if a.fast:
        PACE[0] = 0.008
    scratch = a.scratch.expanduser().resolve()
    marker = scratch / '.ember-demo'
    if scratch.exists() and any(scratch.iterdir()) and not marker.exists():
        # only a folder this demo made is ever deleted: --scratch ~/projects would have removed the whole tree
        print(f'ember_demo: {scratch} is not empty and not a demo folder (no .ember-demo); pass an empty or new '
              'folder', file=sys.stderr)
        return 2
    if scratch.exists():
        shutil.rmtree(scratch)
    home, bridge = scratch / 'home', scratch / 'bridge'
    home.mkdir(parents=True)
    marker.write_text('made by tools/ember_demo.py; it may delete this folder\n')
    bridge.mkdir()
    for sub in ('req', 'res'):
        (bridge / sub).mkdir()

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f'http://127.0.0.1:{server.server_port}/v1'

    r = agent(home, 'init', '--model', 'demo', '--endpoint', endpoint, '--bridge', str(bridge),
              '--bus', str(scratch / 'mail.sqlite3'), '--json')
    if r.returncode:
        sys.exit('demo: init failed: ' + r.stderr.strip())
    conf = json.loads((home / 'config.json').read_text())
    conf.update(question_timeout=120, context=131072)     # an unanswered question carries on after two minutes
    (home / 'config.json').write_text(json.dumps(conf, indent=1))
    (home / 'notes.txt').write_text('A file the agent finds when it looks around.\n')

    wrapper = scratch / 'demo-cli'
    wrapper.write_text(WRAPPER.format(py=q(sys.executable), root=q(ROOT), home=q(home), boot=BOOT))
    wrapper.chmod(0o755)

    panel = a.panel
    if not panel:
        panel = str(scratch / 'caminus-panel')
        b = subprocess.run(['go', 'build', '-o', panel, '.'], cwd=ROOT / 'panel', capture_output=True, text=True)
        if b.returncode:
            sys.exit('demo: building the panel failed (needs Go, or pass --panel):\n' + b.stderr[-2000:])

    log = (scratch / 'daemon.log').open('w')
    daemon = subprocess.Popen([sys.executable, '-I', '-c', BOOT, str(ROOT / 'python'), 'env', 'agent', '--local', 'run',
                               AGENT, '--home', str(home)], stdout=log, stderr=log, start_new_session=True)

    cmd = [panel, 'env', '-demo', '-agent', AGENT, '-bridge', str(bridge), '-audit', str(scratch / 'audit.jsonl'),
           '-cli', json.dumps([str(wrapper)]),
           # it opens on the empty chat (Ember and the welcome), with a first goal in the composer: Enter sends it
           '-draft', 'Write a short report about this machine and what is in my home.']
    try:
        if a.no_tui:
            print(' '.join(q(c) for c in cmd), flush=True)
            while daemon.poll() is None:
                time.sleep(1)
        else:
            subprocess.run(cmd)
    except KeyboardInterrupt:
        pass
    finally:
        if daemon.poll() is None:
            os.killpg(daemon.pid, signal.SIGTERM)
            try:
                daemon.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(daemon.pid, signal.SIGKILL)
        server.shutdown()
        if a.keep:
            print(f'demo: the scratch home is kept at {scratch}')
        else:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == '__main__':
    main()
