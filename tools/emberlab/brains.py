# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 jdominick05
"""Brains for the Ember lab: what answers the resident's model calls in a run.

  scripted:NAME            an in-process stand-in that plays a task's script NAME (suite.json "scripts"): no model, no
                           GPU, no key; the lab's own tests and harness rules run on it
  openai:URL[#MODEL]       an OpenAI-compatible server, e.g. a local `caminus serve` (default model: the first it lists)
  openai+relay:URL#MODEL   the same through the owner's key relay (billed: the relay adds the key and holds the cap)
  anthropic:PORT#MODEL     Anthropic's Messages API through the owner's relay on 127.0.0.1:PORT (`caminus ember relay`)

The lab never reads, holds or passes an API key: a paid brain is always reached through the relay, whose daily cap is
the spend guard. A brain's `parallel` is how many runs may use it at once: a local server answers one stream at a time.
"""
from collections import deque
import http.server
import json
import threading
import urllib.request


class Brain:
    kind = ''
    parallel = 1
    paid = False
    real = True            # a real model drives the resident's tools (the local host refuses it without --unsafe-local)

    def __init__(self, spec):
        self.spec = spec

    def label(self):
        return self.spec

    def slug(self):
        return ''.join(c if c.isalnum() or c in '-.' else '-' for c in self.label())[:60].strip('-')

    def init_args(self, key):
        """the resident's `init` arguments for one run (key: the run id)"""
        raise NotImplementedError

    def config(self):
        """keys merged into the run's config.json after init"""
        return {}

    def start(self):
        pass

    def stop(self):
        pass


def _split_model(rest):
    url, _, model = rest.partition('#')
    return url.rstrip('/'), model or None


class OpenAI(Brain):
    kind = 'openai'

    def __init__(self, spec, relay=False):
        super().__init__(spec)
        self.url, self.model = _split_model(spec.split(':', 1)[1])
        self.relay = relay
        self.paid = relay
        if relay and not self.model:
            raise ValueError(f'{spec}: a relayed brain names its model (URL#MODEL)')

    def start(self):
        if not self.model:
            with urllib.request.urlopen(self.url + '/models', timeout=10) as r:
                ids = [m.get('id') for m in json.load(r).get('data', []) if m.get('id')]
            if not ids:
                raise ValueError(f'{self.url} lists no model')
            self.model = ids[0]

    def label(self):
        return f'{self.model or "?"}@{self.url}'

    def init_args(self, key):
        return ['--model', self.model, '--endpoint', self.url]

    def config(self):
        return dict(relay=True) if self.relay else {}


class Anthropic(Brain):
    kind = 'anthropic'
    paid = True
    parallel = 2

    def __init__(self, spec):
        super().__init__(spec)
        port, self.model = _split_model(spec.split(':', 1)[1])
        if not port.isdigit() or not self.model:
            raise ValueError(f'{spec}: anthropic:PORT#MODEL (the owner\'s relay port)')
        self.url = f'http://127.0.0.1:{port}'

    def label(self):
        return f'{self.model}@relay'

    def init_args(self, key):
        # the Messages API reports no window: the resident needs one (200K: what every current Claude model takes)
        return ['--model', self.model, '--endpoint', self.url, '--api', 'anthropic', '--context', '200000']

    def config(self):
        return dict(relay=True)


class _Handler(http.server.BaseHTTPRequestHandler):
    """The stand-in model: per run, a queue of replies; the finish judge (a request whose first message starts with
    JUDGE) answers met. An exhausted queue answers HTTP 500, so a script that is too short ends the goal in error
    instead of passing by accident."""
    queues = {}
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def respond(self, value, status=200):
        raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.rstrip('/').endswith('/models'):
            self.respond(dict(data=[dict(id='scripted')]))
        else:
            self.respond(dict(n_ctx=65536))

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
        if self.path.endswith('/count_tokens'):
            return self.respond(dict(input_tokens=len(json.dumps(body)) // 4))
        usage = dict(prompt_tokens=len(json.dumps(body)) // 4, completion_tokens=20)
        msgs = body.get('messages') or [{}]
        if str(msgs[0].get('content', '')).startswith('JUDGE'):
            verdict = json.dumps(dict(verdict='met', reason='the scripted judge accepts'))
            return self.respond(dict(choices=[dict(message=dict(role='assistant', content=verdict),
                                                   finish_reason='stop')], usage=usage))
        key = self.path.strip('/').split('/')[0]
        with self.lock:
            q = self.queues.get(key)
            step = q.popleft() if q else None
        if step is None:
            return self.respond(dict(error='the script is exhausted'), 500)
        name, args = step
        import uuid
        msg = dict(role='assistant', content='', tool_calls=[dict(id='call-' + uuid.uuid4().hex[:12], type='function',
                   function=dict(name=name, arguments=json.dumps(args)))])
        self.respond(dict(choices=[dict(message=msg, finish_reason='tool_calls')], usage=usage))


class Scripted(Brain):
    kind = 'scripted'
    parallel = 8
    real = False

    def __init__(self, spec):
        super().__init__(spec)
        self.script = spec.split(':', 1)[1]
        self.server = None

    def start(self):
        if self.server is None:
            self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
            threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

    def load(self, key, task, workdir):
        """queue the task's script for run `key`, with DIR replaced by the run's work folder"""
        steps = (task.get('scripts') or {}).get(self.script)
        if steps is None:
            raise ValueError(f'task {task["key"]} has no script {self.script!r}')
        fill = lambda v: v.replace('DIR', workdir) if isinstance(v, str) else v
        with _Handler.lock:
            _Handler.queues[key] = deque((name, {k: fill(v) for k, v in args.items()}) for name, args in steps)

    def init_args(self, key):
        return ['--model', 'scripted', '--endpoint', f'http://127.0.0.1:{self.server.server_port}/{key}/v1']


class _DrivenHandler(_Handler):
    """You are the model: each request is written to CTL/NNNNNN.req.json and answered by CTL/NNNNNN.rep.json, which
    `lab.py reply` writes (a tool call or text). The judge answers met unless the brain was opened with :judge.
    A caller that hangs up while its request waits (an Esc, a /cancel: the resident closes the call so a real server
    stops generating) is noticed as caminus serve notices it, and recorded in CTL/NNNNNN.gone.json (when, after how
    long); that request is no longer waiting."""
    ctl = None
    judge = False
    wait = 3600

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
        if self.path.endswith('/count_tokens'):
            return self.respond(dict(input_tokens=len(json.dumps(body)) // 4))
        usage = dict(prompt_tokens=len(json.dumps(body)) // 4, completion_tokens=20)
        msgs = body.get('messages') or [{}]
        if str(msgs[0].get('content', '')).startswith('JUDGE') and not self.judge:
            verdict = json.dumps(dict(verdict='met', reason='the driven judge accepts'))
            return self.respond(dict(choices=[dict(message=dict(role='assistant', content=verdict), finish_reason='stop')],
                                     usage=usage))
        import os, time
        from pathlib import Path
        ctl = Path(self.ctl)
        with self.lock:
            n = 1 + max([int(p.name[:6]) for p in ctl.glob('*.req.json')] or [0])
            req = ctl / f'{n:06d}.req.json'
            tmp = req.with_suffix('.tmp')
            tmp.write_text(json.dumps(body))
            os.replace(tmp, req)
        rep = ctl / f'{n:06d}.rep.json'
        t0 = time.time()
        end = t0 + self.wait
        while time.time() < end and not rep.exists():
            if self.hung_up():
                gone = ctl / f'{n:06d}.gone.json'
                tmp = gone.with_suffix('.tmp')
                tmp.write_text(json.dumps(dict(ts=time.time(), waited_s=round(time.time() - t0, 2))))
                os.replace(tmp, gone)
                self.close_connection = True
                return
            time.sleep(0.2)
        if not rep.exists():
            return self.respond(dict(error='no reply from the driver in time'), 504)
        r = json.loads(rep.read_text())
        if r.get('call'):
            msg = dict(role='assistant', content=r.get('text') or '', tool_calls=[dict(id=f'call-driven-{n}', type='function',
                       function=dict(name=r['call'], arguments=json.dumps(r.get('args') or {})))])
            reason = 'tool_calls'
        else:
            msg, reason = dict(role='assistant', content=r.get('text') or ''), 'stop'
        self.respond(dict(choices=[dict(message=msg, finish_reason=reason)], usage=usage))


    def hung_up(self):
        """the caller closed its end (readable with nothing to read), the check caminus serve makes while it works"""
        import select, socket
        try:
            r, _, _ = select.select([self.connection], [], [], 0)
            return bool(r) and self.connection.recv(1, socket.MSG_PEEK) == b''
        except OSError:
            return True


def serve_driven(ctl, judge=False, ttl=14400):
    """the driven brain's server process (`lab.py open --brain driven`); prints its port, then serves until ttl"""
    import time
    _DrivenHandler.ctl, _DrivenHandler.judge = str(ctl), judge
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _DrivenHandler)
    print(srv.server_port, flush=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(ttl)


def parse(spec, parallel=None):
    kind = spec.split(':', 1)[0]
    if ':' not in spec:
        raise ValueError(f'brain {spec!r}: scripted:NAME, openai:URL[#MODEL], openai+relay:URL#MODEL or anthropic:PORT#MODEL')
    b = (Scripted(spec) if kind == 'scripted' else OpenAI(spec) if kind == 'openai' else
         OpenAI(spec, relay=True) if kind == 'openai+relay' else Anthropic(spec) if kind == 'anthropic' else None)
    if b is None:
        raise ValueError(f'unknown brain kind {kind!r}')
    if parallel:
        b.parallel = parallel
    return b
