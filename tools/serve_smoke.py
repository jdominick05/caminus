#!/usr/bin/env python3
"""End-to-end smoke of `caminus serve` as an end user meets it (readiness): the REAL server (tools/caminus_serve.py at its
DEFAULT configuration: max_ctx 262,144, DFlash, the batched prefill, the prompt cache with its disk tier, side-slot routing, the
flight recorder, the hardware reader) started as a child process, driven by a stdlib HTTP client over both APIs; then the SSE
chat route + tool parser timed against the direct engine (in this process). Everything under tools/guard.sh, both cards, one
engine at a time; scratch directories only (the disk tier and the flight recorder never use ~/.cache/caminus here).

  flock <job>/tmp/gpu.lock tools/guard.sh LOG 3600 python -I tools/serve_smoke.py \\
        --scratch <job scratch>/<step>/smoke [--model GGUF] [--stages server timing] [--reps 5] [--json OUT] [--keep-flight]
        [--llm-top BIN]

Stages (each check prints PASS / FAIL; RESULT PASS at the end when every check passed):
  server  `python -I tools/caminus_serve.py --port 0 --cache-dir S/kv --flight-dir S/flight` (nothing else: the defaults), 1 Hz
          sysfs VRAM over the whole stage (the 262K fit: XTX + the 512 MiB reserve <= 23,800, XT <= 19,850, at ready, after the
          100K request and at the peak). Server 1:
            /health 503 while loading, 200 when ready; /v1/models, /props, /caminus/status, /slots, /metrics;
            A: an OpenAI tool conversation (k3-coder: thinking; streaming with stream_options.include_usage, reasoning_content
               sent back, 4 synthetic tools): a multi-turn tool loop, each call's arguments valid JSON for its schema (a turn
               with no natural call is retried with tool_choice required, reported); a short side request between turns
               (k3-coder-nothink, max_tokens 24) routed to the side slot; every later turn reuses the conversation;
            B: an Anthropic tool conversation (k3-coder-nothink; system as blocks with cache_control, a ~12K-token synthetic system
               text like a harness's, anthropic-version / -beta headers, tool_use / tool_result ids) with a thinking side call
               (k3-coder, thinking enabled) between turns; /v1/messages/count_tokens == the request's input tokens;
            abort mid-generation (a streaming turn closed after 10 text deltas): the server finishes at the next pass, the turn
               resent in full reuses its prompt;
            the flight recorder from outside (one JSONL line per generation request, 0600 / 0700, no request text) and a bundle
               on demand; no request text in /metrics, /caminus/status or /slots;
            SIGTERM: the shutdown park of the live conversation (B), exit 0.
          Server 2 (same directories):
            B's last turn resent: restored from the disk tier, the same reply (greedy); B's next turn continues it (live);
            cancel mid-prefill (a cold ~24K-token prompt, the client gone after 4 s): stopped at a chunk boundary, the resend
               continues the processed prefix (live);
            a ~100K-token request (synthetic records with one needle; /slots progress polled from outside): HTTP 200, the
               needle's answer reported;
            SIGTERM: the shutdown park (the 100K line), exit 0.
  timing  the SSE chat route + tool parser against the direct engine (the lead's readiness item; the overhead gate was measured
          on /v1/completions only): the serving engine at max_ctx 262,144 in this process (tests/test_serve_k3.Rig: K3Server on
          the real K3Session, the stats hook, the hardware reader, the flight recorder and card timing on), the requests from a
          CLIENT CHILD PROCESS (like a harness: its SSE parsing never holds this process's GIL); per depth (a short prompt, ~32K)
          and rep: POST /v1/chat/completions stream + tools + tool_choice required, POST /v1/messages stream + tools +
          tool_choice any (k3-coder-nothink, greedy: every generated token is tool-call text through the parser), /v1/completions
          on the same token ids (the earlier gate's route, reported), then the direct engine from the same prompt-end state
          (prepare + first logits + DFlashK3Drafter.resume: tests/test_serve_k3 stage_overhead's arm); interleaved medians of
          --reps; decode ms/token = the server's predicted_ms / predicted_n (the overhead gate's definition) vs the direct arm,
          against the 3 % bar; tokens equal; the client's end-to-end wall time reported.
Inputs are synthetic text generated here (a marker string checks that no request text reaches the metrics or the flight
recorder); nothing private, no probe contexts. At the end the disk tier is purged with tools/kv_cache.py purge and (unless
--keep-flight, for a tools/replay.py run of the bundle afterwards) the flight directory with tools/flight.py purge."""
import argparse
import atexit
import json
import os
import random
import re
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'gates'))
import common as C   # noqa: E402  (tools/gates/common.py: CLAUDE.md's paths; CPU only)

FAILS, NUMBERS = [], {}
MARK = 'quokka-lantern-7731'            # in every synthetic prompt: must never appear in the metrics, the status, /slots or flight files
UA = 'caminus-serve-smoke/1'
GUARD_XTX, GUARD_XT, RESERVE = 23800, 19850, 512
# CAMINUS_SMOKE_INSTALLED=1: the server stage runs the INSTALLED package (`python -I -m caminus serve`, the purges through
# `caminus kv-cache` / `caminus flight`) with this interpreter (a venv's); the timing stage imports the checkout's tests and
# is refused then
INSTALLED = os.environ.get('CAMINUS_SMOKE_INSTALLED') == '1'


def tool(name, *args):
    """a runtime tool's command: tools/<name>.py from this checkout, or the installed package's `caminus <command>`"""
    if INSTALLED:
        return [sys.executable, '-I', '-m', 'caminus', {'kv_cache': 'kv-cache'}.get(name, name), *args]
    return [sys.executable, '-I', os.path.join(ROOT, 'tools', name + '.py'), *args]
EOS = (248046, 248044)


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)
    return ok


def host_note():
    comp = open('/proc/sys/vm/compaction_proactiveness').read().strip()
    mem = next(int(l.split()[1]) for l in open('/proc/meminfo') if l.startswith('MemAvailable:'))
    return f'compaction_proactiveness {comp}, MemAvailable {mem / 1024:.0f} MiB'


def vram_used():
    return [int(open(f'/sys/class/drm/{c}/device/mem_info_vram_used').read()) / 2**20 for c in ('card1', 'card2')]


def kfd_pids():
    try: return {int(p) for p in os.listdir('/sys/class/kfd/kfd/proc') if p.isdigit()}
    except OSError: return set()


def safe_dir(d):
    d = os.path.realpath(os.path.expanduser(d))
    for bad in (os.path.expanduser('~/.cache/caminus'), C.ARTIFACTS, '/tmp'):
        bad = os.path.realpath(bad)
        if d == bad or d.startswith(bad + os.sep): raise SystemExit(f'--scratch {d}: refused (under {bad}): use a job scratch directory')
    return d


# ------------------------------------------------------------------------------------------------ a stdlib HTTP client
def request(port, method, path, body=None, headers=None, close_after_s=None, close_when=None, timeout=1800):
    """one HTTP/1.1 request on a raw socket (Connection: close). Returns dict(status, headers, json | text, events, t_send,
    t_end, t_first, closed). SSE bodies are split into events as they arrive, each with its arrival time (perf_counter);
    close_after_s: close the socket that long after sending, before any reply (a client that leaves during the prompt);
    close_when(events): close as soon as it returns True (a client that leaves mid-generation)"""
    data = b'' if body is None else json.dumps(body).encode()
    hdr = {'Host': f'127.0.0.1:{port}', 'User-Agent': UA, 'Accept': '*/*', 'Connection': 'close'}
    if body is not None: hdr.update({'Content-Type': 'application/json', 'Content-Length': str(len(data))})
    hdr.update(headers or {})
    s = socket.create_connection(('127.0.0.1', port), timeout=timeout)
    s.sendall((f'{method} {path} HTTP/1.1\r\n' + ''.join(f'{k}: {v}\r\n' for k, v in hdr.items()) + '\r\n').encode() + data)
    out = dict(status=None, headers={}, json=None, text=None, events=[], t_send=time.perf_counter(), t_end=None, t_first=None, closed=False)
    if close_after_s is not None:
        time.sleep(close_after_s); s.close(); out.update(closed=True, t_end=time.perf_counter()); return out
    buf = b''
    while b'\r\n\r\n' not in buf:
        x = s.recv(65536)
        if not x: break
        buf += x
    head, _, rest = buf.partition(b'\r\n\r\n')
    lines = head.decode('latin-1').split('\r\n')
    out['status'] = int(lines[0].split()[1]) if lines and len(lines[0].split()) > 1 else None
    for l in lines[1:]:
        k, _, v = l.partition(':'); out['headers'][k.strip().lower()] = v.strip()
    if out['headers'].get('content-type', '').startswith('text/event-stream'):
        pend = rest
        while True:
            while b'\n\n' in pend:
                raw, _, pend = pend.partition(b'\n\n')
                ev = {'t': time.perf_counter(), 'event': None, 'data': ''}
                for l in raw.decode('utf-8').split('\n'):
                    if l.startswith('event:'): ev['event'] = l[6:].strip()
                    elif l.startswith('data:'): ev['data'] += l[5:].lstrip()
                if out['t_first'] is None: out['t_first'] = ev['t']
                out['events'].append(ev)
                if close_when is not None and close_when(out['events']):
                    s.close(); out.update(closed=True, t_end=time.perf_counter()); return out
            x = s.recv(65536)
            if not x: break
            pend += x
        out['t_end'] = time.perf_counter()
    else:
        n = int(out['headers'].get('content-length', -1))
        while n < 0 or len(rest) < n:
            x = s.recv(65536)
            if not x: break
            rest += x
        out['t_end'] = time.perf_counter()
        txt = rest.decode('utf-8', 'replace')
        try: out['json'] = json.loads(txt)
        except ValueError: out['text'] = txt
    s.close()
    return out


def get(port, path):
    r = request(port, 'GET', path, timeout=60)
    return r['status'], (r['json'] if r['json'] is not None else r['text'])


def openai_collect(r):
    """an OpenAI chat stream -> the reply: content, reasoning, tool calls, finish, usage, timings, protocol notes"""
    m = dict(content='', reasoning='', calls={}, finish=None, usage=None, timings=None, done=False, role=False, error=None, n_delta=0)
    for ev in r['events']:
        if ev['data'] == '[DONE]': m['done'] = True; continue
        try: o = json.loads(ev['data'])
        except ValueError: m['error'] = f'bad event {ev["data"][:80]!r}'; continue
        if 'error' in o: m['error'] = o['error']; continue
        for ch in o.get('choices') or []:
            d = ch.get('delta') or {}
            if d.get('role') == 'assistant': m['role'] = True
            if d.get('content'): m['content'] += d['content']; m['n_delta'] += 1
            if d.get('reasoning_content'): m['reasoning'] += d['reasoning_content']; m['n_delta'] += 1
            for tc in d.get('tool_calls') or []:
                c = m['calls'].setdefault(tc.get('index', 0), {'id': None, 'name': '', 'arguments': ''})
                if tc.get('id'): c['id'] = tc['id']
                f = tc.get('function') or {}
                c['name'] += f.get('name') or ''
                c['arguments'] += f.get('arguments') or ''
            if ch.get('finish_reason'): m['finish'] = ch['finish_reason']
        if o.get('usage'): m['usage'] = o['usage']
        if o.get('timings'): m['timings'] = o['timings']
    m['calls'] = [m['calls'][k] for k in sorted(m['calls'])]
    return m


def anthropic_collect(r):
    """an Anthropic message stream -> the reply blocks (thinking / text / tool_use), stop reason, usage, timings, the event order"""
    m = dict(blocks={}, stop=None, usage=None, timings=None, start_usage=None, order=[], error=None, n_delta=0)
    for ev in r['events']:
        try: o = json.loads(ev['data'])
        except ValueError: m['error'] = f'bad event {ev["data"][:80]!r}'; continue
        t = o.get('type'); m['order'].append(t)
        if t == 'message_start': m['start_usage'] = (o.get('message') or {}).get('usage')
        elif t == 'content_block_start':
            cb = dict(o.get('content_block') or {}); cb.setdefault('text', ''); cb.setdefault('thinking', ''); cb['json'] = ''
            m['blocks'][o.get('index')] = cb
        elif t == 'content_block_delta':
            b = m['blocks'].setdefault(o.get('index'), {'type': '?', 'text': '', 'thinking': '', 'json': ''})
            d = o.get('delta') or {}
            if d.get('type') == 'text_delta': b['text'] += d.get('text', ''); m['n_delta'] += 1
            elif d.get('type') == 'thinking_delta': b['thinking'] += d.get('thinking', ''); m['n_delta'] += 1
            elif d.get('type') == 'input_json_delta': b['json'] += d.get('partial_json', '')
        elif t == 'message_delta':
            m['stop'] = (o.get('delta') or {}).get('stop_reason'); m['usage'] = o.get('usage'); m['timings'] = o.get('timings')
        elif t == 'error': m['error'] = o.get('error')
    m['blocks'] = [m['blocks'][k] for k in sorted(m['blocks'])]
    for b in m['blocks']:
        if b.get('type') == 'tool_use':
            try: b['input'] = json.loads(b['json']) if b['json'] else (b.get('input') or {})
            except ValueError: b['input'] = None
    return m


# ------------------------------------------------------------------------------------------------ synthetic inputs
NOUNS = ('cache', 'buffer', 'request', 'module', 'parser', 'thread', 'socket', 'config file', 'retry loop', 'timeout', 'queue',
         'handler', 'token', 'schema', 'record', 'stream', 'commit', 'branch', 'unit test', 'fixture', 'logger', 'worker',
         'signal', 'shard', 'index', 'layer', 'driver', 'client', 'migration', 'lock file')
VERBS = ('validate', 'flush', 'parse', 'retry', 'serialize', 'close', 'open', 'merge', 'split', 'check', 'measure', 'refuse',
         'log', 'rename', 'sort', 'review', 'document', 'profile')
ADJS = ('small', 'explicit', 'bounded', 'typed', 'idempotent', 'local', 'sorted', 'stable', 'short', 'separate', 'readable')


def guidelines(seed, n_chars, title):
    """a deterministic synthetic 'project guidelines' text (a harness's system prompt stand-in) of about n_chars characters"""
    rng = random.Random(seed)
    out = [f'# {title}\n\nInternal reference tag: {MARK}. Follow every rule below when you work in this repository.\n']
    i = 0
    while sum(len(x) for x in out) < n_chars:
        i += 1
        n1, n2, n3 = rng.sample(NOUNS, 3)
        v1, v2 = rng.sample(VERBS, 2)
        out.append(f'{i}. When you {v1} a {n1}, {v2} the {n2} first and keep the {n3} {rng.choice(ADJS)}. '
                   f'Prefer a {rng.choice(ADJS)} {rng.choice(NOUNS)} over a clever one; explain any exception in the commit message.\n')
    return ''.join(out)


def code_file(seed, n_chars):
    """a deterministic synthetic Python source file of about n_chars characters"""
    rng = random.Random(seed)
    out = [f'"""Synthetic module ({MARK}) used by the Caminus smoke test."""\nimport os\n\n']
    i = 0
    while sum(len(x) for x in out) < n_chars:
        i += 1
        a, b = rng.sample(('path', 'size', 'count', 'limit', 'offset', 'name', 'value', 'items', 'flags'), 2)
        out.append(f'def {rng.choice(VERBS)}_{i}({a}, {b}=None):\n'
                   f'    """{rng.choice(VERBS).capitalize()} the {rng.choice(NOUNS)} for {a}."""\n'
                   f'    if {b} is None:\n        {b} = {rng.randint(1, 999)}\n'
                   f'    return [{a}] * min(int({b}), {rng.randint(2, 64)})\n\n\n')
    return ''.join(out)


CITIES = ('Lisbon', 'Osaka', 'Quito', 'Tallinn', 'Accra', 'Perth', 'Bergen', 'Cusco', 'Hanoi', 'Tucson', 'Graz', 'Leeds')


def records(seed, n, needle_at, needle):
    rng = random.Random(seed)
    lines = [f'Synthetic inventory export ({MARK}). One record per line.\n']
    for i in range(n):
        extra = f' passphrase={needle}' if i == needle_at else ''
        lines.append(f'record {i:06d}: city={rng.choice(CITIES)} code={rng.randint(10000, 99999)} '
                     f'qty={rng.randint(1, 500)} status={rng.choice(("open", "held", "shipped", "void"))}{extra}\n')
    return ''.join(lines)


FILES = {
    'src/config.py': 'TIMEOUT = 45\nRETRIES = 3\nBACKOFF_S = 1.5\nLOG_LEVEL = "info"\n',
    'README.md': '# demo\n\nA tiny service that fetches records over HTTP, retries on failure and writes them to disk.\n'
                 'Run `python -m demo` to start it. Tests live in tests/.\n',
    'src/net.py': 'import time\nfrom .config import TIMEOUT, RETRIES, BACKOFF_S\n\n\ndef fetch(url, get):\n'
                  '    for i in range(RETRIES):\n        try:\n            return get(url, timeout=TIMEOUT)\n'
                  '        except OSError:\n            time.sleep(BACKOFF_S * (i + 1))\n    raise RuntimeError("gave up")\n',
}
LISTING = 'README.md\nsrc/\nsrc/config.py\nsrc/net.py\ntests/\ntests/test_net.py\n'


def run_tool(name, args):
    """the synthetic tools' results (deterministic)"""
    args = args if isinstance(args, dict) else {}
    if name == 'read_file':
        p = str(args.get('path', '')).lstrip('./')
        return FILES.get(p, f'error: {p}: no such file')
    if name == 'list_dir':
        return LISTING
    if name == 'run_shell':
        return 'exit status 0\n3 passed in 0.12s\n' if 'test' in str(args.get('command', '')) else 'exit status 0\n'
    if name == 'write_file':
        return f'wrote {len(str(args.get("content", "")))} bytes to {args.get("path")}'
    return f'error: unknown tool {name}'


TOOL_DEFS = [
    ('read_file', 'Read a text file from the workspace and return its contents.', {'path': {'type': 'string', 'description': 'path relative to the repository root'}}, ['path']),
    ('list_dir', 'List the files below a directory of the workspace.', {'path': {'type': 'string', 'description': 'directory, "." for the root'}}, ['path']),
    ('run_shell', 'Run a shell command in the workspace and return its output.', {'command': {'type': 'string'}, 'timeout_s': {'type': 'integer'}}, ['command']),
    ('write_file', 'Create or overwrite a text file in the workspace.', {'path': {'type': 'string'}, 'content': {'type': 'string'}}, ['path', 'content']),
]
OPENAI_TOOLS = [{'type': 'function', 'function': {'name': n, 'description': d, 'parameters': {'type': 'object', 'properties': p, 'required': r}}}
                for n, d, p, r in TOOL_DEFS]
ANTHROPIC_TOOLS = [{'name': n, 'description': d, 'input_schema': {'type': 'object', 'properties': p, 'required': r}} for n, d, p, r in TOOL_DEFS]
REQUIRED = {n: r for n, _, _, r in TOOL_DEFS}
ANTHROPIC_HEADERS = {'anthropic-version': '2023-06-01', 'anthropic-beta': 'interleaved-thinking-2025-05-14', 'x-api-key': 'caminus-local'}
OPENAI_HEADERS = {'Authorization': 'Bearer caminus-local'}


def reply_of(m):
    """an Anthropic reply's content as one comparable string: every block's type, text / thinking, tool name and input"""
    return json.dumps([[b.get('type'), b.get('text', ''), b.get('thinking', ''), b.get('name'), b.get('input')]
                       for b in (m or {}).get('blocks', [])], sort_keys=True)


def args_ok(name, args):
    """a tool call's arguments: a JSON object of a declared tool with its required keys"""
    return name in REQUIRED and isinstance(args, dict) and all(k in args for k in REQUIRED[name])


# ------------------------------------------------------------------------------------------------ the server child
LIVE = []      # every server child this process started: never left running (stop_servers in main's finally, atexit backstop)


def stop_servers():
    """SIGTERM every server child still running (its clean shutdown: drain, park, exit), SIGKILL past 240 s: a failure anywhere in
    the server stage must not leave an engine holding both cards outside the guard's watch once this process exits"""
    for srv in LIVE:
        if srv.p.poll() is None:
            print(f'{srv.tag}: still running at the end of the stage: stopping it', flush=True)
            try:
                srv.stop()
            except Exception:                    # noqa: BLE001 - last resort
                srv.p.kill()


atexit.register(stop_servers)


class Server:
    def __init__(self, tag, scratch, model):
        self.tag, self.lines, self.port, self.pid, self.loading503 = tag, [], None, None, False
        cmd = ([sys.executable, '-I', '-m', 'caminus', 'serve'] if INSTALLED else [sys.executable, '-I', os.path.join(ROOT, 'tools', 'caminus_serve.py')]) + [
               '--port', '0', '--model', model, '--cache-dir', os.path.join(scratch, 'kv'), '--flight-dir', os.path.join(scratch, 'flight')]
        print(f'{tag}: starting {" ".join(cmd[2:])} ({host_note()}; VRAM {[round(x) for x in vram_used()]} MiB)', flush=True)
        self.t0 = time.time()
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  env=dict(os.environ, PYTHONUNBUFFERED='1'))
        LIVE.append(self)
        self.ready = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            line = line.rstrip('\n')
            self.lines.append(line)
            print(f'  {self.tag}| {line}', flush=True)
            if self.port is None and 'http://' in line:
                m = re.search(r'http://[^:]+:(\d+)', line)
                if m: self.port = int(m.group(1))
                m = re.search(r'pid=(\d+)', line)
                if m: self.pid = int(m.group(1))
            if 'ready on' in line:
                self.ready.set()
        self.ready.set()

    def wait_ready(self, timeout=1200):
        t0 = time.time()
        while not self.ready.is_set() and time.time() - t0 < timeout:
            if self.port is not None and not self.loading503:
                try:
                    st, _ = get(self.port, '/health'); self.loading503 = st == 503
                except OSError:
                    pass
            time.sleep(0.5)
        self.load_s = time.time() - self.t0
        return self.ready.is_set() and self.p.poll() is None

    def stop(self, timeout=240):
        """SIGTERM (a clean shutdown: drain, the shutdown park, exit); SIGKILL only past the timeout"""
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
        try:
            self.p.wait(timeout)
        except subprocess.TimeoutExpired:
            print(f'{self.tag}: no exit {timeout} s after SIGTERM: SIGKILL', flush=True)
            self.p.kill(); self.p.wait(60)
        time.sleep(0.5)
        return self.p.returncode


class VramSampler:
    def __init__(self):
        self.peak, self.samples, self.stop_ev = [0.0, 0.0], [], threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stop_ev.is_set():
            u = vram_used(); self.samples.append((time.time(), u))
            self.peak = [max(a, b) for a, b in zip(self.peak, u)]
            self.stop_ev.wait(1.0)


def wait_teardown(idle, timeout=60):
    """after a server exits: no other KFD process and both cards back near their idle use (within 300 MiB)"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        u = vram_used()
        if not (kfd_pids() - {os.getpid()}) and u[0] <= idle[0] + 300 and u[1] <= idle[1] + 300:
            return True, u
        time.sleep(1)
    return False, vram_used()


def vram_ok(u, what):
    return check(u[0] + RESERVE <= GUARD_XTX and u[1] <= GUARD_XT,
                 f'262K fit ({what}): XTX {u[0]:.0f} MiB (+ the {RESERVE} reserve = {u[0] + RESERVE:.0f} <= {GUARD_XTX}), '
                 f'XT {u[1]:.0f} <= {GUARD_XT}')


def wait_idle(port, timeout=600):
    """until the server's one slot is idle (a client that left: the job ends at its next pass boundary)"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        st, sl = get(port, '/slots')
        if st == 200 and isinstance(sl, list) and sl and not sl[0].get('is_processing'):
            st2, s = get(port, '/caminus/status')
            if st2 == 200 and not s['server']['requests_processing'] and not s['server']['queue_depth']:
                return True
        time.sleep(0.2)
    return False


def flight_lines(fdir):
    out = []
    for f in sorted(os.listdir(fdir)) if os.path.isdir(fdir) else []:
        if f.startswith('requests-') and f.endswith('.jsonl'):
            out += [json.loads(x) for x in open(os.path.join(fdir, f)) if x.strip()]
    return out


def mode(p):
    return stat.S_IMODE(os.stat(p).st_mode)


def walk_modes(root):
    """every file under root is 0600 and every directory 0700 (the flight recorder's rule)"""
    bad = []
    for d, dirs, files in os.walk(root):
        if mode(d) != 0o700: bad.append((d, oct(mode(d))))
        for f in files:
            p = os.path.join(d, f)
            if not os.path.islink(p) and mode(p) != 0o600: bad.append((p, oct(mode(p))))
    return bad


def text_free(fdir):
    """no file of the flight directory (JSONL lines, bundle.json) holds the request text marker"""
    hits = []
    for d, _, files in os.walk(fdir):
        for f in files:
            if f.endswith(('.jsonl', '.json')):
                p = os.path.join(d, f)
                if MARK in open(p, encoding='utf-8', errors='replace').read(): hits.append(p)
    return hits


# ------------------------------------------------------------------------------------------------ conversations
class Smoke:
    def __init__(self, scratch, model):
        self.scratch, self.model = scratch, model
        self.fdir = os.path.join(scratch, 'flight')
        self.n_gen = 0                       # generation requests that reached a server (flight lines expected)
        self.srv = None
        self.b_last_text, self.b_msgs, self.b_abort_body, self.b_reply, self.b_reply_n = '', None, None, None, None

    @property
    def port(self): return self.srv.port

    def chat(self, body, close_when=None, close_after_s=None):
        b = dict(body); b.setdefault('stream', True)
        if b['stream']: b.setdefault('stream_options', {'include_usage': True})
        self.n_gen += 1
        r = request(self.port, 'POST', '/v1/chat/completions', b, OPENAI_HEADERS, close_when=close_when, close_after_s=close_after_s)
        if close_after_s is not None: return r, None
        return r, (openai_collect(r) if b['stream'] and r['events'] else None)

    def messages(self, body, close_when=None):
        b = dict(body); b.setdefault('stream', True)
        self.n_gen += 1
        r = request(self.port, 'POST', '/v1/messages', b, ANTHROPIC_HEADERS, close_when=close_when)
        return r, (anthropic_collect(r) if b['stream'] and r['events'] else None)

    def status(self):
        st, s = get(self.port, '/caminus/status'); return s if st == 200 else {}

    def route_last(self):
        return ((self.status().get('cache') or {}).get('route') or {}).get('last') or {}

    # ---- A: OpenAI, k3-coder (thinking), a tool loop with a side call between turns
    def conv_a(self):
        sys_a = guidelines(11, 9000, 'Project guidelines for the demo service')
        msgs = [{'role': 'system', 'content': 'You are a coding agent working in the repository /work/demo. Use the tools to '
                                              'inspect files before you answer.\n\n' + sys_a},
                {'role': 'user', 'content': 'Read src/config.py with the read_file tool and tell me the value of TIMEOUT.'}]
        res, prev, trips, forced = [], None, 0, 0
        for turn in range(1, 5):
            body = dict(model='k3-coder', messages=msgs, tools=OPENAI_TOOLS, temperature=0.0, max_tokens=4096)
            r, m = self.chat(body)
            u = (m or {}).get('usage') or {}
            ok = r['status'] == 200 and m is not None and m['role'] and m['done'] and m['finish'] and not m['error']
            check(ok, f'A turn {turn} (OpenAI SSE, k3-coder): HTTP {r["status"]}, role chunk {m and m["role"]}, finish {m and m["finish"]}, '
                      f'[DONE] {m and m["done"]}, usage {u.get("prompt_tokens")} prompt / {(u.get("prompt_tokens_details") or {}).get("cached_tokens")} '
                      f'cached / {u.get("completion_tokens")} completion, reasoning {len(m["reasoning"]) if m else 0} chars, '
                      f'{len(m["calls"]) if m else 0} tool call(s)')
            if not ok: return res
            t = m['timings'] or {}
            res.append(dict(turn=turn, prompt=u.get('prompt_tokens'), cached=(u.get('prompt_tokens_details') or {}).get('cached_tokens'),
                            completion=u.get('completion_tokens'), restore=t.get('restore'), finish=m['finish'],
                            calls=[c['name'] for c in m['calls']], ttft_ms=t.get('ttft_ms'), decode_ms_per_token=t.get('predicted_per_token_ms'),
                            prefill_tok_s=t.get('prompt_per_second'), drafted=t.get('drafted'), accepted=t.get('accepted')))
            if prev is not None:
                check(res[-1]['cached'] >= prev - 4 and t.get('restore') in ('live', 'checkpoint'),
                      f'A turn {turn}: the conversation reused across the turn (and the side call): cached {res[-1]["cached"]} >= the previous '
                      f'prompt {prev} - 4, from {t.get("restore")}')
            prev = u.get('prompt_tokens')
            if not m['calls'] and turn == 1:
                # no natural tool call on the first turn: once more with tool_choice required (the server's forced call), reported
                forced += 1
                print(f'A turn 1: no natural tool call (finish {m["finish"]}); retrying with tool_choice required', flush=True)
                r, m = self.chat(dict(body, tool_choice='required'))
                if r['status'] != 200 or m is None or not m['calls']:
                    check(False, f'A turn 1 with tool_choice required: HTTP {r["status"]}, calls {m and m["calls"]}'); return res
                prev = (m['usage'] or {}).get('prompt_tokens')
            if not m['calls']:
                break
            calls = []
            for c in m['calls']:
                try: a = json.loads(c['arguments'] or '{}')
                except ValueError: a = None
                check(args_ok(c['name'], a) and bool(c['id']), f'A turn {turn}: tool call {c["name"]}({c["arguments"][:120]}) id {c["id"]}: '
                                                                f'arguments a JSON object with the schema\'s required keys')
                calls.append((c, a))
            msgs = msgs + [{'role': 'assistant', 'content': m['content'] or None, 'reasoning_content': m['reasoning'] or None,
                            'tool_calls': [{'id': c['id'], 'type': 'function', 'function': {'name': c['name'], 'arguments': c['arguments']}}
                                           for c, _ in calls]}]
            msgs += [{'role': 'tool', 'tool_call_id': c['id'], 'content': run_tool(c['name'], a)} for c, a in calls]
            trips += 1
            if turn == 1:
                self.side_openai(msgs[1]['content'])
        NUMBERS['A'] = dict(turns=res, tool_round_trips=trips, forced=forced)
        check(trips >= 1, f'A: the OpenAI tool loop ran {trips} tool round trip(s) ({forced} turn(s) forced with tool_choice required)')
        self.a_msgs = msgs
        return res

    def side_openai(self, user_text):
        body = dict(model='k3-coder-nothink', stream=False, temperature=0.0, max_tokens=24,
                    messages=[{'role': 'user', 'content': 'Write a 3-5 word title for this request. Reply with the title only.\n\n' + user_text}])
        self.n_gen += 1
        r = request(self.port, 'POST', '/v1/chat/completions', body, OPENAI_HEADERS)
        last = self.route_last()
        ok = r['status'] == 200 and r['json'] and r['json']['choices'][0]['message'].get('content') is not None
        check(ok and last.get('route') == 'side',
              f'side call (OpenAI, k3-coder-nothink, max_tokens 24, non-streaming): HTTP {r["status"]} '
              f'{ok and repr(r["json"]["choices"][0]["message"]["content"][:60])}; routed {last.get("route")} ({last.get("reason")})')

    # ---- B: Anthropic, k3-coder-nothink, harness-sized system blocks, a tool loop with a thinking side call between turns
    def b_body(self, msgs, max_tokens=8192):
        return dict(model='k3-coder-nothink', max_tokens=max_tokens, temperature=0.0, tools=ANTHROPIC_TOOLS, messages=msgs,
                    system=[{'type': 'text', 'text': 'You are a command-line coding assistant. Be concise; use the tools to read the '
                                                     'repository before answering.'},
                            {'type': 'text', 'text': self.sys_b, 'cache_control': {'type': 'ephemeral'}}])

    def conv_b(self):
        self.sys_b = guidelines(22, 48000, 'Repository handbook')
        msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': 'List the repository root with list_dir, then open README.md with '
                                                                       'read_file and summarize the project in two sentences.'}]}]
        st, ct = None, None
        res, prev, trips, forced = [], None, 0, 0
        for turn in range(1, 6):
            body = self.b_body(msgs)
            if turn == 1:
                st = request(self.port, 'POST', '/v1/messages/count_tokens', {k: v for k, v in body.items() if k != 'max_tokens'},
                             ANTHROPIC_HEADERS)
                ct = (st['json'] or {}).get('input_tokens')
            r, m = self.messages(body)
            u = (m or {}).get('usage') or {}
            order = (m or {}).get('order') or []
            ok = (r['status'] == 200 and m is not None and not m['error'] and order[:1] == ['message_start'] and order[-1:] == ['message_stop']
                  and 'message_delta' in order)
            check(ok, f'B turn {turn} (Anthropic SSE, k3-coder-nothink): HTTP {r["status"]}, events {order[:1]} .. {order[-2:]}, stop '
                      f'{m and m["stop"]}, usage input {u.get("input_tokens")} + cache_read {u.get("cache_read_input_tokens")}, output '
                      f'{u.get("output_tokens")}, blocks {[b.get("type") for b in (m or {}).get("blocks", [])]}')
            if not ok:
                self.b_msgs = msgs if turn > 1 else None; return res
            t = m['timings'] or {}
            prompt = (u.get('input_tokens') or 0) + (u.get('cache_read_input_tokens') or 0)
            if turn == 1:
                check(ct == prompt, f'B: /v1/messages/count_tokens {ct} == the request\'s input tokens {prompt} (HTTP {st["status"]})')
            uses = [b for b in m['blocks'] if b.get('type') == 'tool_use']
            res.append(dict(turn=turn, prompt=prompt, cached=u.get('cache_read_input_tokens'), completion=u.get('output_tokens'),
                            restore=t.get('restore'), stop=m['stop'], calls=[b.get('name') for b in uses], ttft_ms=t.get('ttft_ms'),
                            decode_ms_per_token=t.get('predicted_per_token_ms'), prefill_tok_s=t.get('prompt_per_second')))
            if prev is not None:
                check(res[-1]['cached'] >= prev - 4 and t.get('restore') in ('live', 'checkpoint'),
                      f'B turn {turn}: the conversation reused across the turn (and the side call): cached {res[-1]["cached"]} >= the previous '
                      f'prompt {prev} - 4, from {t.get("restore")}')
            prev = prompt
            if not uses and turn == 1:
                forced += 1
                print(f'B turn 1: no natural tool_use (stop {m["stop"]}); retrying with tool_choice any', flush=True)
                r, m = self.messages(dict(body, tool_choice={'type': 'any'}))
                uses = [b for b in (m or {}).get('blocks', []) if b.get('type') == 'tool_use']
                if r['status'] != 200 or not uses:
                    check(False, f'B turn 1 with tool_choice any: HTTP {r["status"]}, blocks {m and m["blocks"]}'); return res
            if not uses:
                self.b_last_text = ''.join(b.get('text', '') for b in m['blocks'] if b.get('type') == 'text')
                break
            for b in uses:
                check(args_ok(b.get('name'), b.get('input')) and bool(b.get('id')),
                      f'B turn {turn}: tool_use {b.get("name")}({json.dumps(b.get("input"))[:120]}) id {b.get("id")}: input a JSON object with '
                      f'the schema\'s required keys')
            content = [{'type': 'text', 'text': b['text']} for b in m['blocks'] if b.get('type') == 'text' and b.get('text')]
            content += [{'type': 'tool_use', 'id': b['id'], 'name': b['name'], 'input': b['input']} for b in uses]
            msgs = msgs + [{'role': 'assistant', 'content': content},
                           {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': b['id'], 'content': run_tool(b['name'], b['input'])}
                                                        for b in uses]}]
            trips += 1
            if turn == 1:
                self.side_anthropic()
        NUMBERS['B'] = dict(turns=res, tool_round_trips=trips, forced=forced)
        check(trips >= 1, f'B: the Anthropic tool loop ran {trips} tool round trip(s) ({forced} turn(s) forced with tool_choice any)')
        self.b_msgs = msgs
        return res

    def side_anthropic(self):
        body = dict(model='k3-coder', max_tokens=256, temperature=0.0, thinking={'type': 'enabled', 'budget_tokens': 128},
                    messages=[{'role': 'user', 'content': 'Does this message start a new topic? Answer with JSON {"isNewTopic": true|false}.\n\n'
                                                          'Message: how do I list files?'}])
        r, m = self.messages(body)
        last = self.route_last()
        kinds = [b.get('type') for b in (m or {}).get('blocks', [])]
        check(r['status'] == 200 and m is not None and not m['error'] and last.get('route') == 'side',
              f'side call (Anthropic, k3-coder with thinking enabled, max_tokens 256, streaming): HTTP {r["status"]}, blocks {kinds}, '
              f'stop {m and m["stop"]}; routed {last.get("route")} ({last.get("reason")})')

    # ---- abort mid-generation, then the turn resent
    def abort_b(self):
        if self.b_msgs is None:
            check(False, 'abort mid-generation: not run (conversation B did not complete)'); return None
        # a long text reply: the file is in the message and tool_choice none (a tool call is held until it closes, so a reply that
        # calls a tool streams no deltas to leave in the middle of)
        ask = ('Here is src/net.py:\n\n```python\n' + FILES['src/net.py'] + '```\n\nWithout calling any tool, explain in detail (at '
               'least 300 words) how this retry logic behaves when every attempt fails, and what you would change.')
        msgs = self.b_msgs + [{'role': 'assistant', 'content': [{'type': 'text', 'text': self.b_last_text or 'Done.'}]},
                              {'role': 'user', 'content': [{'type': 'text', 'text': ask}]}]
        body = dict(self.b_body(msgs, max_tokens=1024), tool_choice={'type': 'none'})

        def ten_deltas(evs):
            return sum(1 for e in evs if e['event'] == 'content_block_delta') >= 10
        r, m = self.messages(body, close_when=ten_deltas)
        idle = wait_idle(self.port)
        n_ev = len(r['events'])
        r2, m2 = self.messages(body)
        u = (m2 or {}).get('usage') or {}
        t = (m2 or {}).get('timings') or {}
        prompt = (u.get('input_tokens') or 0) + (u.get('cache_read_input_tokens') or 0)
        check(r['closed'] and idle and r2['status'] == 200 and m2 and m2['stop'] and u.get('cache_read_input_tokens', 0) >= prompt - 4
              and t.get('restore') in ('live', 'checkpoint'),
              f'abort mid-generation: the client left after {n_ev} events (10 text deltas); the server idle again {idle}; the turn resent: '
              f'HTTP {r2["status"]}, stop {m2 and m2["stop"]}, cached {u.get("cache_read_input_tokens")} of {prompt} from {t.get("restore")} '
              f'(the prompt kept)')
        self.b_abort_body = body
        self.b_reply = reply_of(m2)
        self.b_reply_n = u.get('output_tokens')
        txt = ''.join(b.get('text', '') for b in (m2 or {}).get('blocks', []) if b.get('type') == 'text')
        print(f'abort: the resent turn\'s reply: {u.get("output_tokens")} tokens, stop {m2 and m2["stop"]}, {len(txt)} chars of text', flush=True)
        NUMBERS['abort'] = dict(events_before_close=n_ev, resend=dict(prompt=prompt, cached=u.get('cache_read_input_tokens'),
                                                                      restore=t.get('restore'), output=u.get('output_tokens')))
        return m2

    # ---- what the server shows from outside
    def outside(self, phase):
        st_m, met = get(self.port, '/metrics')
        st_s, sts = get(self.port, '/caminus/status')
        st_l, slots = get(self.port, '/slots')
        st_p, props = get(self.port, '/props')
        clean = all(MARK not in (x if isinstance(x, str) else json.dumps(x)) for x in (met, sts, slots, props))
        rt = ((sts or {}).get('cache') or {}).get('route') or {}
        side_n = sum((rt.get('side') or {}).values())
        prom = {}
        for line in (met or '').splitlines():
            if line and not line.startswith('#'):
                k, _, v = line.rpartition(' ')
                try: prom[k] = float(v)
                except ValueError: pass
        need = ('llamacpp:prompt_tokens_total', 'llamacpp:tokens_predicted_total', 'caminus:clock_check_ok')
        check(st_m == 200 and st_s == 200 and st_l == 200 and st_p == 200 and all(k in prom for k in need) and prom['llamacpp:tokens_predicted_total'] > 0
              and (sts or {}).get('version') == 1 and len((sts or {}).get('cards') or []) == 2 and clean,
              f'{phase}: /metrics ({len(prom)} series; prompt tokens {prom.get("llamacpp:prompt_tokens_total")}, predicted '
              f'{prom.get("llamacpp:tokens_predicted_total")}, clock_check_ok {prom.get("caminus:clock_check_ok")}), /caminus/status v'
              f'{(sts or {}).get("version")} ({len((sts or {}).get("cards") or [])} cards; routes main {rt.get("main")} side {rt.get("side")}), '
              f'/slots, /props: HTTP {st_m} / {st_s} / {st_l} / {st_p}; no request text in any of them: {clean}')
        NUMBERS.setdefault('outside', {})[phase] = dict(routes=rt, cache_sources=((sts or {}).get('cache') or {}).get('sources'),
                                                        cache_events=((sts or {}).get('cache') or {}).get('events'),
                                                        clock_check=((sts or {}).get('clock_check') or {}).get('summary'),
                                                        reliability=(sts or {}).get('reliability'), side_calls=side_n)
        return sts

    def flight_check(self, phase):
        t0 = time.time()
        while time.time() - t0 < 30 and len(flight_lines(self.fdir)) < self.n_gen:
            time.sleep(0.2)
        lines = flight_lines(self.fdir)
        keys = ('request_id', 'harness', 'profile', 'sampling', 'cache', 'ttft_split_ms', 'decode_ms_per_token', 'dflash', 'tool_calls',
                'loop', 'verdict')
        full = [x for x in lines if x.get('status') == 200 and not x.get('client_gone')]
        bad_modes = walk_modes(self.fdir)
        hits = text_free(self.fdir)
        check(len(lines) == self.n_gen and all(all(k in x for k in keys) for x in full) and not bad_modes and not hits,
              f'{phase}: the flight recorder (from outside, {self.fdir}): {len(lines)} JSONL line(s) for {self.n_gen} generation requests, '
              f'every completed one with {", ".join(keys)}; slots {[(x.get("cache") or {}).get("slot") for x in lines]}; modes '
              f'{"0600 / 0700" if not bad_modes else bad_modes[:3]}; request text in no flight file: {not hits}')
        return lines

    def bundle(self):
        r = request(self.port, 'POST', '/caminus/flight/bundle', {}, OPENAI_HEADERS)
        b = r['json'] or {}
        ref = b.get('cache_ref') or {}
        d = b.get('path')
        bj = os.path.join(d, 'bundle.json') if d else None
        ok = bool(r['status'] == 200 and bj and os.path.exists(bj))
        meta = json.load(open(bj)) if ok else {}
        ids = (meta.get('request') or {}).get('prompt')
        modes = walk_modes(d) if ok else ['?']
        clean = ok and MARK not in open(bj, encoding='utf-8').read()
        check(ok and not modes and clean and isinstance(ids, list) and len(ids) > 0 and ref.get('state'),
              f'flight bundle on demand (POST /caminus/flight/bundle): HTTP {r["status"]}, {d}, reason {b.get("reason")}, start '
              f'{ref.get("source")} (state {ref.get("state")}), {len(ids or [])} prompt token ids, modes '
              f'{"0600 / 0700" if not modes else modes[:3]}, no request text in bundle.json: {clean}')
        NUMBERS['bundle'] = dict(dir=d, request_id=b.get('request_id'), cache_ref=ref)
        return d


def stage_server(a):
    sc = Smoke(a.scratch, a.model)
    os.makedirs(os.path.join(a.scratch, 'kv'), mode=0o700, exist_ok=True)
    idle = vram_used()
    print(f'server stage: idle VRAM {[round(x) for x in idle]} MiB; KFD processes {sorted(kfd_pids())}; {host_note()}', flush=True)
    vs = VramSampler()
    # ------------------------------------------------------------------ server 1
    sc.srv = s1 = Server('srv1', a.scratch, a.model)
    ok = s1.wait_ready()
    built = next((l for l in s1.lines if 'engine built in' in l), '')
    check(ok and s1.loading503, f'srv1: /health 503 while loading, ready after {s1.load_s:.0f} s (engine line: {built.split("caminus serve: ")[-1][:240]})')
    if not ok:
        s1.stop(); vs.stop_ev.set(); return
    u_ready = vram_used(); vram_ok(u_ready, 'srv1 ready')
    st, models = get(s1.port, '/v1/models')
    ids = [x.get('id') for x in (models or {}).get('data', [])]
    st2, props = get(s1.port, '/props')
    n_ctx = (props or {}).get('n_ctx') or ((props or {}).get('default_generation_settings') or {}).get('n_ctx')
    check(st == 200 and {'k3-coder', 'k3-coder-nothink'} <= set(ids) and st2 == 200 and props.get('engine') == 'caminus' and (n_ctx or 0) >= 262000,
          f'srv1: /v1/models {ids}; /props engine {props.get("engine")}, n_ctx {n_ctx} (the default 262,144 configuration), alias '
          f'{props.get("model_alias")}')
    NUMBERS['srv1'] = dict(load_s=round(s1.load_s, 1), engine=built, vram_ready=u_ready, n_ctx=n_ctx, host=host_note())
    sc.conv_a()
    sc.conv_b()
    sc.abort_b()
    sc.outside('srv1')
    sc.flight_check('srv1')
    bdir = sc.bundle()
    if a.llm_top:
        llm_top_check(a, s1.port)
    rc = s1.stop()
    park = [l for l in s1.lines if 'shutdown: park' in l]
    check(rc == 0 and park and "'unchanged'" not in park[0] and 'None' not in park[0],
          f'srv1: SIGTERM -> a clean shutdown with the live conversation parked: {park[:1]}; exit {rc}')
    NUMBERS['srv1']['shutdown'] = [l for l in s1.lines if 'shutdown' in l or 'stopped in' in l]
    td, u = wait_teardown(idle)
    check(td, f'srv1 gone: no KFD process, VRAM back to {[round(x) for x in u]} MiB (idle {[round(x) for x in idle]})')
    # ------------------------------------------------------------------ server 2: the same directories
    sc.srv = s2 = Server('srv2', a.scratch, a.model)
    ok = s2.wait_ready()
    check(ok, f'srv2 (the same --cache-dir / --flight-dir): ready after {s2.load_s:.0f} s')
    if not ok:
        s2.stop(); vs.stop_ev.set(); return
    vram_ok(vram_used(), 'srv2 ready')
    if sc.b_abort_body is None:
        check(False, 'after the restart: the disk-tier resume not run (conversation B did not complete)')
    else:
        restart_check(sc)
    cancel_check(sc)
    long_check(sc, vs)
    sc.outside('srv2')
    sc.flight_check('srv2')
    rc = s2.stop()
    park = [l for l in s2.lines if 'shutdown: park' in l]
    check(rc == 0 and park and 'None' not in park[0], f'srv2: SIGTERM -> a clean shutdown: {park[:1]}; exit {rc}')
    NUMBERS['srv2'] = dict(load_s=round(s2.load_s, 1), shutdown=[l for l in s2.lines if 'shutdown' in l or 'stopped in' in l])
    td, u = wait_teardown(idle)
    check(td, f'srv2 gone: no KFD process, VRAM back to {[round(x) for x in u]} MiB')
    vs.stop_ev.set()
    vram_ok(vs.peak, 'the peak over both servers, 1 Hz sysfs')
    NUMBERS['vram'] = dict(idle=idle, ready=u_ready, peak=vs.peak)
    NUMBERS['bundle_dir'] = bdir
    print(f'server stage done: VRAM peak {[round(x) for x in vs.peak]} MiB; {host_note()}', flush=True)


def restart_check(sc):
    r, m = sc.messages(sc.b_abort_body)
    u = (m or {}).get('usage') or {}; t = (m or {}).get('timings') or {}
    prompt = (u.get('input_tokens') or 0) + (u.get('cache_read_input_tokens') or 0)
    reply = reply_of(m)
    check(r['status'] == 200 and t.get('restore') == 'disk' and u.get('cache_read_input_tokens', 0) >= prompt - 4,
          f'after the restart: B\'s last turn resent: HTTP {r["status"]}, restored from {t.get("restore")} (the disk tier), cached '
          f'{u.get("cache_read_input_tokens")} of {prompt}, restore {t.get("ttft_ms")} ms TTFT')
    check(reply == sc.b_reply and u.get('output_tokens') == sc.b_reply_n,
          f'after the restart: the same reply as before the restart (greedy): {len(reply)} chars / {u.get("output_tokens")} tokens vs '
          f'{len(sc.b_reply)} / {sc.b_reply_n}; equal {reply == sc.b_reply}')
    NUMBERS['restart'] = dict(prompt=prompt, cached=u.get('cache_read_input_tokens'), restore=t.get('restore'), ttft_ms=t.get('ttft_ms'),
                              same_reply=reply == sc.b_reply)
    text = ''.join(b.get('text', '') for b in (m or {}).get('blocks', []) if b.get('type') == 'text') or 'Done.'
    msgs = sc.b_abort_body['messages'] + [{'role': 'assistant', 'content': [{'type': 'text', 'text': text}]},
                                         {'role': 'user', 'content': [{'type': 'text', 'text': 'Thanks. In one sentence: what is the main risk?'}]}]
    r, m = sc.messages(dict(sc.b_body(msgs, max_tokens=512), tool_choice={'type': 'none'}))
    u = (m or {}).get('usage') or {}; t = (m or {}).get('timings') or {}
    check(r['status'] == 200 and t.get('restore') in ('live', 'checkpoint') and u.get('cache_read_input_tokens', 0) >= prompt - 4,
          f'after the restart: B\'s next turn continues the conversation in VRAM: HTTP {r["status"]}, from {t.get("restore")}, cached '
          f'{u.get("cache_read_input_tokens")} >= {prompt} - 4')


def cancel_check(sc):
    text = code_file(33, 75000)          # ~24K tokens
    body = dict(model='k3-coder-nothink', temperature=0.0,
                messages=[{'role': 'user', 'content': 'Here is a module:\n\n```python\n' + text + '```\n\nWhich function has the largest '
                                                      'default value? Answer with its name.'}])
    r, _ = sc.chat(body, close_after_s=4.0)
    idle = wait_idle(sc.port)
    lines = flight_lines(sc.fdir)
    t0 = time.time()
    while time.time() - t0 < 20 and len(lines) < sc.n_gen:
        time.sleep(0.2); lines = flight_lines(sc.fdir)
    gone = lines[-1] if lines else {}
    r2, m2 = sc.chat(dict(body, max_tokens=32))
    u = (m2 or {}).get('usage') or {}; t = (m2 or {}).get('timings') or {}
    cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens')
    src = ((sc.status().get('cache') or {}).get('sources') or {})
    check(r['closed'] and idle and gone.get('client_gone') and r2['status'] == 200 and t.get('restore') == 'live'
          and 0 < (cached or 0) < u.get('prompt_tokens', 0),
          f'cancel mid-prefill: a cold {u.get("prompt_tokens")}-token prompt, the client gone after 4 s (flight line client_gone '
          f'{gone.get("client_gone")}, finish {gone.get("finish")}); resent: HTTP {r2["status"]}, from {t.get("restore")}, cached {cached} '
          f'(the prefix processed before the cut); cache sources so far {src}')
    NUMBERS['cancel'] = dict(prompt=u.get('prompt_tokens'), cached=cached, restore=t.get('restore'), prefill_tok_s=t.get('prompt_per_second'),
                             sources=src)


def long_check(sc, vs):
    needle = 'amber-falcon-2291'
    probe = records(44, 2000, -1, needle)
    st = request(sc.port, 'POST', '/v1/messages/count_tokens', {'model': 'k3-coder-nothink', 'messages': [{'role': 'user', 'content': probe}]},
                 ANTHROPIC_HEADERS)
    per = (st['json'] or {}).get('input_tokens', 50000) / 2000.0
    n = int(101500 / per) + 1
    needle_at = int(n * 0.37)                       # inside the text (Qwen tokenizes digits one by one: ~30 tokens a record)
    text = records(44, n, needle_at, needle)
    q = f'\n\nWhich passphrase is stated in record {needle_at:06d}? Answer with the passphrase only.'
    body = dict(model='k3-coder-nothink', temperature=0.0, max_tokens=48, messages=[{'role': 'user', 'content': text + q}])
    prog, done = [], threading.Event()

    def poll():
        while not done.is_set():
            s_, sl = get(sc.port, '/slots')
            if s_ == 200 and sl and sl[0].get('is_processing'):
                prog.append((round(time.time(), 1), sl[0].get('n_prompt_tokens_processed')))
            done.wait(2.0)
    th = threading.Thread(target=poll, daemon=True); th.start()
    r, m = sc.chat(body)
    done.set(); th.join(5)
    u = (m or {}).get('usage') or {}; t = (m or {}).get('timings') or {}
    after = vram_used()
    answer = (m or {}).get('content', '')
    seen = sorted({p for _, p in prog if p})
    check(r['status'] == 200 and m and m['finish'] and u.get('prompt_tokens', 0) >= 100000 and len(seen) >= 3,
          f'100K request: HTTP {r["status"]}, {u.get("prompt_tokens")} prompt tokens (cold), finish {m and m["finish"]}, prefill '
          f'{t.get("prompt_per_second")} tok/s, TTFT {t.get("ttft_ms")} ms; /slots progress seen from outside {seen[:3]} .. {seen[-2:]} '
          f'({len(seen)} distinct); the answer {answer[:60]!r} (needle {needle}: {"found" if needle in answer else "NOT found"}, reported)')
    vram_ok(after, 'after the 100K request')
    NUMBERS['long'] = dict(records=n, needle_at=needle_at, prompt=u.get('prompt_tokens'), prefill_tok_s=t.get('prompt_per_second'), ttft_ms=t.get('ttft_ms'),
                           decode_ms_per_token=t.get('predicted_per_token_ms'), needle_found=needle in answer, answer=answer[:80],
                           progress=seen[:4] + seen[-4:], vram_after=after)


def llm_top_check(a, port):
    """llm-top -once against the live server (its Caminus backend), with a config that points it at this port and at an empty
    sessions directory (it never reads ~/.qwen here)"""
    d = os.path.join(a.scratch, 'llm-top'); os.makedirs(os.path.join(d, 'sessions'), exist_ok=True)
    cfg = os.path.join(d, 'llm-top.json')
    json.dump({'server_url': f'http://127.0.0.1:{port}', 'backend': 'auto', 'sessions_dir': os.path.join(d, 'sessions'),
               'hold_file': os.path.join(d, 'hold'), 'server_config': os.path.join(d, 'none.json')}, open(cfg, 'w'))
    try:
        p = subprocess.run([a.llm_top, '-config', cfg, '-once', '140', '60'], capture_output=True, text=True, timeout=60)
        out = p.stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        out = f'{type(e).__name__}: {e}'
    check('CAMINUS' in out, f'llm-top -once ({a.llm_top}) against the live server: the CAMINUS panel present {"CAMINUS" in out} '
                            f'({len(out.splitlines())} lines)')


# ------------------------------------------------------------------------------------------------ timing (in-process)
def client_main():
    """--client: a request loop in a separate process (like a harness): one JSON request per stdin line -> one JSON result line"""
    for line in sys.stdin:
        q = json.loads(line)
        r = request(q['port'], 'POST', q['path'], q['body'], q.get('headers'))
        out = dict(status=r['status'], wall_ms=(r['t_end'] - r['t_send']) * 1e3,
                   first_ms=None if r['t_first'] is None else (r['t_first'] - r['t_send']) * 1e3)
        if q['path'] == '/v1/chat/completions':
            m = openai_collect(r)
            out.update(timings=m['timings'], usage=m['usage'], finish=m['finish'], calls=[c['name'] for c in m['calls']], error=m['error'])
        elif q['path'] == '/v1/messages':
            m = anthropic_collect(r)
            out.update(timings=m['timings'], usage=m['usage'], finish=m['stop'], error=m['error'],
                       calls=[b.get('name') for b in m['blocks'] if b.get('type') == 'tool_use'])
        else:
            j = r['json'] or {}
            out.update(timings=j.get('timings'), usage=j.get('usage'), finish=(j.get('choices') or [{}])[0].get('finish_reason'), calls=[])
        sys.stdout.write(json.dumps(out) + '\n'); sys.stdout.flush()


def stage_timing(a):
    if INSTALLED:
        check(False, 'timing: refused with CAMINUS_SMOKE_INSTALLED=1 (it imports the checkout\'s tests): run --stages server')
        return
    import numpy as np
    import torch
    sys.path.insert(0, os.path.join(ROOT, 'python')); sys.path.insert(0, os.path.join(ROOT, 'tests', 'e2e')); sys.path.insert(0, os.path.join(ROOT, 'tests', 'cache'))
    import test_serve_k3 as TS
    from caminus.decode import longctx
    for d in (0, 1): torch.cuda.set_per_process_memory_fraction(0.97 if d == 0 else 0.95, d)
    assert 'XTX' in torch.cuda.get_device_name(0) and 'XTX' not in torch.cuda.get_device_name(1), 'cuda:0 must be the XTX, cuda:1 the XT'
    pcd = os.path.join(a.scratch, 'timing', 'pc'); os.makedirs(pcd, mode=0o700, exist_ok=True)
    print(f'timing stage: {host_note()}; VRAM {[round(x) for x in vram_used()]} MiB', flush=True)
    R = TS.Rig(a.model, 262144, pcd)
    cl = subprocess.Popen([sys.executable, '-I', os.path.abspath(__file__), '--client'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def ask(path, body, headers):
        cl.stdin.write(json.dumps(dict(port=R.srv.port, path=path, body=body, headers=headers)) + '\n'); cl.stdin.flush()
        return json.loads(cl.stdout.readline())

    def flight_of(n_before):
        R.wait_idle(); time.sleep(0.3)
        lines = flight_lines(R.flight.root)
        return lines[n_before:] if len(lines) > n_before else []
    try:
        vram_ok(vram_used(), 'the timing rig at max_ctx 262,144')
        sys_t = 'You are a coding agent. Create files with the write_file tool; write each file with one call.'
        task = ('Use the write_file tool once per file to create these six files: src/a.py, src/b.py, src/c.py, src/d.py, src/e.py and '
                'src/f.py. Each file is a small Python module with a docstring and two short functions.')
        big = code_file(55, 97000)          # ~31K tokens (this text runs ~3.1 characters per token)
        cases = [('short', [{'role': 'system', 'content': sys_t}, {'role': 'user', 'content': task}])]
        if not a.small:
            cases.append(('32K', [{'role': 'system', 'content': sys_t},
                                  {'role': 'user', 'content': 'Read src/big.py, then follow my next instruction.'},
                                  {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_big', 'type': 'function', 'function': {
                                      'name': 'read_file', 'arguments': json.dumps({'path': 'src/big.py'})}}]},
                                  {'role': 'tool', 'tool_call_id': 'call_big', 'content': big},
                                  {'role': 'user', 'content': task}]))
        n = 512
        res = {}
        for name, msgs in cases:
            oa = dict(model='k3-coder-nothink', messages=msgs, tools=OPENAI_TOOLS, tool_choice='required', max_tokens=n, temperature=0.0,
                      stream=True, stream_options={'include_usage': True})
            amsgs = [{'role': 'user', 'content': msgs[1]['content']}] if len(msgs) == 2 else [
                {'role': 'user', 'content': msgs[1]['content']},
                {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'call_big', 'name': 'read_file', 'input': {'path': 'src/big.py'}}]},
                {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_big', 'content': big}, {'type': 'text', 'text': task}]}]
            an = dict(model='k3-coder-nothink', system=sys_t, messages=amsgs, tools=ANTHROPIC_TOOLS, tool_choice={'type': 'any'},
                      max_tokens=n, temperature=0.0, stream=True)
            R.pc.reset()
            w = ask('/v1/chat/completions', oa, OPENAI_HEADERS)          # cold: the prompt-end checkpoint exists afterwards
            ids_o, g0 = list(R.jobs[-1].prompt_ids), list(R.jobs[-1].result.gen_ids)
            w2 = ask('/v1/messages', an, ANTHROPIC_HEADERS)
            ids_a, g0a = list(R.jobs[-1].prompt_ids), list(R.jobs[-1].result.gen_ids)
            print(f'timing {name}: {len(ids_o)} prompt tokens (OpenAI) / {len(ids_a)} (Anthropic), same ids {ids_o == ids_a}; {len(g0)} / '
                  f'{len(g0a)} generated; calls {w.get("calls")} / {w2.get("calls")}; finish {w.get("finish")} / {w2.get("finish")}', flush=True)
            arms = {k: [] for k in ('openai_sse', 'anthropic_sse', 'completions', 'direct')}
            wall = {k: [] for k in arms}
            tools_seen = []
            ok_tok = True
            for rep in range(a.reps):
                for arm, path, body, hdr, ids, g in (('openai_sse', '/v1/chat/completions', oa, OPENAI_HEADERS, ids_o, g0),
                                                     ('anthropic_sse', '/v1/messages', an, ANTHROPIC_HEADERS, ids_a, g0a),
                                                     ('completions', '/v1/completions', dict(prompt=ids_o, max_tokens=n, temperature=0.0),
                                                      OPENAI_HEADERS, ids_o, g0)):
                    nf = len(flight_lines(R.flight.root))
                    o = ask(path, body, hdr)
                    t = o.get('timings') or {}
                    if o['status'] != 200 or not t.get('predicted_n'):
                        check(False, f'timing {name} {arm} rep {rep}: HTTP {o["status"]} {o.get("error")}'); continue
                    arms[arm].append(t['predicted_ms'] / t['predicted_n']); wall[arm].append(o['wall_ms'])
                    got = list(R.jobs[-1].result.gen_ids)
                    if got != g: ok_tok = False; check(False, f'timing {name} {arm} rep {rep}: tokens differ from the first run')
                    if arm != 'completions':
                        fl = flight_of(nf)
                        tc = (fl[-1].get('tool_calls') if fl else None) or {}
                        tools_seen.append((arm, tc.get('ok'), (tc.get('failures') or {}).get('cut')))
                # the direct engine from the same prompt-end state (stage_overhead's arm; card timing off for it)
                R.wait_idle(); ct = R.sess.card_timing
                if ct: R.m.enable_card_timing(False)
                te = time.perf_counter()
                o_ = R.pc.prepare(ids_o)
                lg, st2 = longctx.first_logits(R.m, o_['st'], ids_o, draft=R.dr); R.pc.extend(ids_o, st2)
                t0 = time.perf_counter()
                y0 = int(np.argmax(lg))
                out, st3 = R.dr.resume(R.m, st2, list(ids_o) + [y0], len(g0) - 1)
                t1 = time.perf_counter()
                R.pc.commit(out, st3)
                if ct: R.m.enable_card_timing(True)
                arms['direct'].append((t1 - t0) * 1e3 / len(g0)); wall['direct'].append((t1 - te) * 1e3)
                if out[len(ids_o):] != g0: ok_tok = False; check(False, f'timing {name}: direct tokens differ from the server\'s (rep {rep})')
            med = {k: float(np.median(v)) if v else None for k, v in arms.items()}
            medw = {k: float(np.median(v)) if v else None for k, v in wall.items()}
            ov = {k: (med[k] / med['direct'] - 1) * 100 for k in ('openai_sse', 'anthropic_sse', 'completions') if med[k] and med['direct']}
            res[name] = dict(prompt_tokens=len(ids_o), generated=len(g0), ms_per_token=arms, median=med, overhead_pct=ov, wall_ms=wall,
                             median_wall_ms=medw, tool_calls=tools_seen, same_ids=ids_o == ids_a, tokens_equal=ok_tok)
            parsed = all((x[1] or 0) + (x[2] or 0) >= 1 for x in tools_seen)
            for arm in ('openai_sse', 'anthropic_sse'):
                check(arm in ov and ov[arm] <= 3.0 and ok_tok and parsed,
                      f'timing {name} ({len(ids_o)} prompt tokens, {len(g0)} generated, greedy DFlash, stats / hw / flight / card timing on): '
                      f'{arm} {med[arm]:.3f} vs direct {med["direct"]:.3f} ms/token (medians of {a.reps}, interleaved; '
                      f'{[round(x, 3) for x in arms[arm]]} vs {[round(x, 3) for x in arms["direct"]]}): {ov.get(arm, float("nan")):+.2f} % '
                      f'(bar 3 %); tokens equal {ok_tok}; tool-call text through the parser (ok / cut per request) {[(x[1], x[2]) for x in tools_seen if x[0] == arm][:3]}')
            print(f'timing {name}: /v1/completions (the earlier gate\'s route, reported) {med["completions"]:.3f} ms/token '
                  f'({ov.get("completions", float("nan")):+.2f} %); client end-to-end wall medians (ms): '
                  f'{ {k: round(v, 1) for k, v in medw.items() if v} } ; {host_note()}', flush=True)
        NUMBERS['timing'] = res
    finally:
        cl.stdin.close(); cl.wait(30)
        R.close()
        subprocess.run(tool('flight', 'purge', '--dir', R.flight.root))


def purge(a):
    for cmd in (tool('kv_cache', 'purge', '--dir', os.path.join(a.scratch, 'kv')),
                None if a.keep_flight else tool('flight', 'purge', '--dir',
                                            os.path.join(a.scratch, 'flight'))):
        if cmd is None or not os.path.isdir(cmd[-1]): continue
        p = subprocess.run(cmd, capture_output=True, text=True)
        print(f'purge: {" ".join(cmd[2:])}: exit {p.returncode}: {(p.stdout + p.stderr).strip()[-300:]}', flush=True)


def main():
    if sys.argv[1:] == ['--client']:
        return client_main()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scratch', required=True, help='a scratch directory (refused under ~/.cache/caminus, $ARTIFACTS, /tmp)')
    ap.add_argument('--model', default=C.CODER)
    ap.add_argument('--stages', nargs='+', default=['server', 'timing'], choices=('server', 'timing'))
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--small', action='store_true', help='timing: the short case only')
    ap.add_argument('--json', default=None)
    ap.add_argument('--keep-flight', action='store_true', help='keep the server stage\'s flight directory (its bundle, for tools/replay.py)')
    ap.add_argument('--llm-top', default=None, help='an llm-top binary with the Caminus backend: llm-top -once against the live server')
    a = ap.parse_args()
    a.scratch = safe_dir(a.scratch); os.makedirs(a.scratch, mode=0o700, exist_ok=True)
    print(f'serve_smoke: {host_note()}; stages {a.stages}; scratch {a.scratch}', flush=True)
    t0 = time.time()
    try:
        for name, fn in (('server', stage_server), ('timing', stage_timing)):
            if name not in a.stages: continue
            t1 = time.time()
            try:
                fn(a)
            except Exception as e:                # noqa: BLE001 - reported as a FAIL; the servers are stopped below
                traceback.print_exc()
                check(False, f'{name} stage: {type(e).__name__}: {e}')
            finally:
                stop_servers()
            print(f'== {name} stage {time.time() - t1:.0f} s', flush=True)
    finally:
        stop_servers()
        purge(a)
        if a.json: json.dump(dict(fails=FAILS, numbers=NUMBERS), open(a.json, 'w'), indent=1, default=str)
    print(f'serve_smoke: {time.time() - t0:.0f} s; {host_note()}')
    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'))
    return 0 if not FAILS else 1


if __name__ == '__main__':
    sys.exit(main())
