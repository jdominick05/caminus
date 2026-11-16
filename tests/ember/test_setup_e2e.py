"""caminus ember from a clean account to the first reply, against a stub brain: the real setup, as a new user meets it.

  CAMINUS_E2E_ROOT=1 python -I tests/ember/test_setup_e2e.py --scratch DIR

It creates a Unix account, a group and /srv/caminus-e2e-bridge with sudo, so it runs only where both hold: passwordless
sudo and CAMINUS_E2E_ROOT=1 (CI sets it; a workstation never runs it by accident). Otherwise it prints SKIP and passes.

The steps: `caminus ember setup` (the owner's key; the root script written; says to run it with sudo) -> the root script
with sudo (PASS lines) -> the script again (changes nothing) -> `caminus ember setup` again in the bridge group (the agent
side; every check passes) -> `caminus ember agent install` / `init` against the stub brain -> `agent once` with a goal ->
the goal completes with the stub's summary. Everything it made is removed at the end.
"""
import argparse
import collections
import http.server
import json
import os
import secrets
import shlex
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAILS = []


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok:
        FAILS.append(what)


def call(name, args):
    return dict(role='assistant', content='', tool_calls=[dict(id='c' + secrets.token_hex(4), type='function',
                                                             function=dict(name=name, arguments=json.dumps(args)))])


class Stub(http.server.BaseHTTPRequestHandler):
    """An OpenAI-compatible brain that plays a script; the judge (a system prompt starting JUDGE) always says met."""
    replies = collections.deque()
    seen = []

    def log_message(self, *a):
        pass

    def send(self, value, status=200):
        raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.send(dict(n_ctx=65536, data=[dict(id='fixture')]))

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path.endswith('/count_tokens'):          # the resident's context counter, not a turn
            return self.send(dict(input_tokens=len(json.dumps(body)) // 4))
        msgs = body.get('messages') or [{}]
        usage = dict(prompt_tokens=30, completion_tokens=10)
        if str(msgs[0].get('content', '')).startswith('JUDGE'):
            verdict = json.dumps(dict(verdict='met', reason='the stub judge accepts'))
            return self.send(dict(choices=[dict(message=dict(role='assistant', content=verdict), finish_reason='stop')],
                                  usage=usage))
        self.seen.append(body)
        if not self.replies:
            return self.send(dict(error='stub exhausted'), 500)
        self.send(dict(choices=[dict(message=self.replies.popleft(), finish_reason='stop')], usage=usage))


def sh(cmd, env, **kw):
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=kw.pop('timeout', 300), **kw)
    return r.returncode, (r.stdout or '') + (r.stderr or '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', required=True)
    a = ap.parse_args()
    if os.environ.get('CAMINUS_E2E_ROOT') != '1' or not sys.platform.startswith('linux') or \
            subprocess.run(['sudo', '-n', 'true'], capture_output=True).returncode != 0:
        print('SKIP: needs passwordless sudo and CAMINUS_E2E_ROOT=1 (it creates a Unix account; CI sets both)')
        print('RESULT PASS')
        return 0
    os.makedirs(a.scratch, exist_ok=True)
    agent = 'cae2e' + secrets.token_hex(3)
    bridge = '/srv/caminus-e2e-bridge'
    env = dict(os.environ, CAMINUS_ENV_USER=agent, CAMINUS_ENV_BRIDGE=bridge,
               CAMINUS_ENV_KEY=os.path.join(a.scratch, 'key'), CAMINUS_STATE_DIR=os.path.join(a.scratch, 'state'),
               CAMINUS_ENV_AUDIT=os.path.join(a.scratch, 'state', 'env-audit.jsonl'),
               XDG_DATA_HOME=os.path.join(a.scratch, 'data'), PYTHONPATH=os.path.join(ROOT, 'python'))
    caminus = [sys.executable, '-I', os.path.join(ROOT, 'tools', 'caminus.py')]
    in_group = lambda argv: ['sg', 'caminus-env', '-c', ' '.join(shlex.quote(x) for x in argv)]
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        code, out = sh(caminus + ['env', 'setup'], env)
        script = os.path.join(env['XDG_DATA_HOME'], 'caminus', 'env-setup.sh')
        check(code == 1 and 'sudo sh ' + script in out and os.path.exists(script),
              'setup on a clean machine: writes the root script and says to run it with sudo')
        check(os.path.exists(env['CAMINUS_ENV_KEY']), "setup made the owner's key")

        code, out = sh(['sudo', '-n', 'sh', script], env, timeout=600)
        check(code == 0 and 'FAIL' not in out and 'setup: done' in out, 'the root script: every check passes')
        if code:
            print(out[-3000:])
        code, again = sh(['sudo', '-n', 'sh', script], env, timeout=300)
        changes = [ln for ln in again.splitlines() if ln.startswith('setup: ') and not ln.startswith('setup: done')]
        check(code == 0 and not changes, 'the root script again: changes nothing (%s)' % (changes or 'no change lines'))

        code, out = sh(in_group(caminus + ['env', 'setup']), env)
        check(code == 0 and 'FAIL' not in out, 'setup again in the bridge group: the agent side, every check passes')
        if code:
            print(out[-3000:])

        code, out = sh(in_group(caminus + ['env', 'agent', 'install']), env)
        check(code == 0, 'agent install: the runtime reaches the agent account')
        endpoint = 'http://127.0.0.1:%d/v1' % server.server_port
        code, out = sh(in_group(caminus + ['env', 'agent', 'init', 'resident', '--model', 'fixture',
                                           '--endpoint', endpoint, '--json']), env)
        check(code == 0, 'agent init against the stub brain')
        Stub.replies.extend([call('define_done', dict(check='the file hello.txt exists')),
                             call('write', dict(path='hello.txt', content='hello')),
                             call('verify', dict(evidence='hello.txt written')),
                             call('finish', dict(summary='Wrote hello.txt.'))])
        code, out = sh(in_group(caminus + ['env', 'agent', 'once', 'resident', '--goal', 'Write hello.txt.', '--json']),
                       env, timeout=300)
        try:
            goal = json.loads(out[out.find('{'):])
        except ValueError:
            goal = {}
        check(goal.get('state') == 'completed' and 'hello' in json.dumps(goal),
              'the first goal completes through the agent account (%s)' % goal.get('state'))
        check(len(Stub.seen) >= 4, 'the stub brain was asked %d times' % len(Stub.seen))
        if goal.get('state') != 'completed':
            print(out[-3000:])
    finally:
        server.shutdown()
        subprocess.run(['sudo', '-n', 'loginctl', 'disable-linger', agent], capture_output=True)
        subprocess.run(['sudo', '-n', 'pkill', '-u', agent], capture_output=True)
        subprocess.run(['sudo', '-n', 'userdel', '-r', agent], capture_output=True)
        subprocess.run(['sudo', '-n', 'rm', '-rf', bridge], capture_output=True)
    print('RESULT ' + ('PASS' if not FAILS else 'FAIL (%d failed)' % len(FAILS)))
    return 0 if not FAILS else 1


if __name__ == '__main__':
    sys.exit(main())
