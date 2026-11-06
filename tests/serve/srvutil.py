"""Helpers for the HTTP server tests: an in-process K3Server on 127.0.0.1:0 with a MockSession, raw HTTP / SSE
clients, scripted replies, and the conversions from template-form messages to OpenAI / Anthropic wire messages."""
import http.client
import json
import os
import socket
import tempfile
import time

import common
from caminus.serve.engine import MockSession
from caminus.serve.mock_engine import ScriptedEngine
from caminus.serve.server import K3Server

IM_END = '<|im_end|>'


def make_server(engine=None, max_ctx=32768, pass_sleep=0.0, fail_at=None, session_kw=None, **kw):
    """session_kw: MockSession's metric options (stats, synth, disk_slots, checkpoints). The test process keeps its
    name (proc_name=None): tests/serve/test_metrics.py checks the launcher's in a child process."""
    tok = common.tokenizer()
    eng = engine if engine is not None else ScriptedEngine(ntok=4)
    sess = MockSession(eng, max_ctx=max_ctx, eos_ids=(248046, 248044), pass_sleep=pass_sleep, fail_at=fail_at,
                       **(session_kw or {}))
    log_dir = tempfile.mkdtemp(prefix='k3-serve-test-')
    kw.setdefault('proc_name', None)
    srv = K3Server(sess, tok, port=0, log_dir=log_dir, builder=common.builder(), **kw).start()
    srv.eng = eng
    return srv


def log_rows(srv):
    p = srv.log.path
    if not p or not os.path.exists(p):
        return []
    with open(p, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def wait_rows(srv, n, timeout=10.0):
    t = time.time() + timeout
    while time.time() < t:
        rows = log_rows(srv)
        if len(rows) >= n:
            return rows
        time.sleep(0.02)
    return log_rows(srv)


def request(srv, method, path, body=None, headers=None, raw=None, timeout=120):
    """-> (status, headers dict, bytes)"""
    c = http.client.HTTPConnection('127.0.0.1', srv.port, timeout=timeout)
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    h = {'Content-Type': 'application/json'}
    h.update(headers or {})
    c.request(method, path, data, h)
    r = c.getresponse()
    out = r.read()
    c.close()
    return r.status, {k.lower(): v for k, v in r.getheaders()}, out


def post_json(srv, path, body, **kw):
    st, h, data = request(srv, 'POST', path, body, **kw)
    return st, h, json.loads(data) if data else None


def parse_sse(data):
    """bytes -> [(event or None, data str)] and the list of non-event lines (comments)"""
    events, comments = [], []
    for block in data.decode('utf-8').split('\n\n'):
        if not block.strip():
            continue
        ev, dl = None, []
        for line in block.split('\n'):
            if line.startswith(':'):
                comments.append(line)
            elif line.startswith('event:'):
                ev = line[6:].strip()
            elif line.startswith('data:'):
                dl.append(line[5:].lstrip(' '))
        events.append((ev, '\n'.join(dl)))
    return events, comments


def sse_post(srv, path, body):
    st, h, data = request(srv, 'POST', path, body)
    return st, h, parse_sse(data)[0] if st == 200 else data


def raw_socket_request(srv, path, body):
    s = socket.create_connection(('127.0.0.1', srv.port))
    data = json.dumps(body).encode()
    s.sendall(f'POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n'
              f'Content-Length: {len(data)}\r\n\r\n'.encode() + data)
    return s


# ------------------------------------------------------------------------------------------------ scripts
def reply_ids(msgs, reply, tools, prompt):
    """the canonical ids of `reply` after `prompt` (as tests/serve/test_append_only.py builds them)"""
    b, tok = common.builder(), common.tokenizer()
    pm = b.render_marked(msgs, tools, prompt.kwargs, True)
    fm = b.render_marked(msgs + [reply], tools, prompt.kwargs, False)
    assert fm.startswith(pm)
    tail = fm[len(pm):]
    tail = tail[:tail.index(IM_END) + len(IM_END)]
    return tok.encode_prompt(tail)


def script(eng, msgs, reply, tools, model='k3-coder', client=None):
    """register the scripted reply; returns (prompt ids, reply ids)"""
    p = common.builder().build(msgs, tools, model=model, client_kwargs=client)
    r = reply_ids(msgs, reply, tools, p)
    eng.register(p.ids, r)
    return p.ids, r


# ------------------------------------------------------------------------------------------------ wire forms
def openai_wire(m):
    """a template-form user / system / tool message -> OpenAI wire"""
    return {'role': m['role'], 'content': m['content']}


def anthropic_user_blocks(msgs, tool_ids):
    """consecutive template-form tool / user messages -> one Anthropic user message's content blocks"""
    blocks, k = [], 0
    for m in msgs:
        if m['role'] == 'tool':
            blocks.append({'type': 'tool_result', 'tool_use_id': tool_ids[k] if k < len(tool_ids) else f'toolu_x{k}',
                           'content': m['content']})
            k += 1
        elif m['role'] == 'user':
            blocks.append({'type': 'text', 'text': m['content']})
        else:
            raise AssertionError(m['role'])
    return blocks


def same_reply(tc, got, want, where=''):
    """got: {'reasoning_content', 'content', 'tool_calls': [(name, args dict)]}"""
    tc.assertEqual((got['reasoning_content'] or '').strip(), (want.get('reasoning_content') or '').strip(), where)
    tc.assertEqual((got['content'] or '').strip(), (want.get('content') or '').strip(), where)
    calls = [(c['function']['name'], c['function']['arguments']) for c in want.get('tool_calls') or ()]
    tc.assertEqual(got['tool_calls'], calls, where)
