"""The flight recorder (caminus.serve.flight; M7 ops item 2), CPU only, on scratch directories and MockSession:
  * the directory 0700 and every file 0600 whatever the umask (an existing directory's mode fixed), one owner (the lock);
  * the size cap (the oldest bundles first, then the oldest JSONL days, today's kept), retention, purge, tools/flight.py
    (list beside a server, purge refused while a server holds the directory);
  * the server: one JSONL line per generation request (errors and client-gone included) with the plan's fields and no request
    text; replay bundles on demand (POST /caminus/flight/bundle, the most recent request or one of the last 8) and
    automatically on a reasoning loop, a tool-call parse failure and retry exhaustion (before the session reset).
The GPU half (the start state saved, tools/replay.py bitwise) is tests/e2e/test_serve_k3.py flight + tools/replay.py.
Run:  HIP_VISIBLE_DEVICES= python -I tests/serve/run_all.py"""
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest

import common  # noqa: F401  (sys.path)
import srvutil as U
import synth
from caminus.serve import flight as F
from caminus.serve.engine import MockSession
from caminus.serve.mock_engine import ScriptedEngine
from caminus.serve.server import K3Server

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYS = {'role': 'system', 'content': 'You are a coding agent.'}


def mode(p):
    return stat.S_IMODE(os.stat(p).st_mode)


def scratch():
    return tempfile.mkdtemp(prefix='k3-flight-')


class TestRecorder(unittest.TestCase):
    def test_perms_and_lock(self):
        root = os.path.join(scratch(), 'flight')
        old = os.umask(0)
        try:
            fr = F.FlightRecorder(root)
            fr.record({'a': 1})
            b = fr.new_bundle('abc', 'api')
            fr.finish_bundle(b, {'x': 1}, first_logits=[1.0, 2.0])
        finally:
            os.umask(old)
        self.assertEqual(mode(root), 0o700)
        self.assertEqual(mode(fr.bdir), 0o700)
        self.assertEqual(mode(b), 0o700)
        for p in [fr.day_path(), os.path.join(b, 'bundle.json'), os.path.join(b, 'first_logits.npy'), os.path.join(root, '.lock')]:
            self.assertEqual(mode(p), 0o600, p)
        with self.assertRaises(F.FlightError):
            F.FlightRecorder(root)                      # one owner
        self.assertTrue(F.held(root))
        fr.close()
        self.assertFalse(F.held(root))
        os.chmod(root, 0o755)                           # an existing directory gets 0700
        F.FlightRecorder(root).close()
        self.assertEqual(mode(root), 0o700)
        meta = json.load(open(os.path.join(b, 'bundle.json')))
        self.assertEqual((meta['format'], meta['first_logits']['n']), (1, 2))

    def test_cap_retention_purge(self):
        t = [1_000_000.0]
        root = os.path.join(scratch(), 'flight')
        fr = F.FlightRecorder(root, cap_gb=10e3 / 1e9, retention_days=7, clock=lambda: t[0])
        names = []
        for k in range(4):
            b = fr.new_bundle(f'r{k}', 'api')
            F.write_private(os.path.join(b, 'blob'), b'x' * 3000)
            os.utime(b, (t[0] + k, t[0] + k))
            names.append(os.path.basename(b))
        old_day = os.path.join(root, 'requests-19700101.jsonl')
        F.write_private(old_day, b'y' * 500)
        os.utime(old_day, (t[0] - 10, t[0] - 10))
        fr.enforce()                                    # 12.5 KB > 10 KB: the oldest bundle goes first
        left = [os.path.basename(p) for k, p, nb, mt in fr.entries() if k == 'bundle']
        self.assertEqual(left, names[1:])
        self.assertTrue(os.path.exists(old_day))        # the bundles go first
        fr.enforce(incoming=4000)                       # room for 4 KB more
        self.assertLessEqual(fr.total_bytes() + 4000, 10e3)
        t[0] += 8 * 86400                               # retention: everything older than 7 days
        fr.record({'today': 1})
        fr.enforce()
        self.assertEqual([k for k, p, nb, mt in fr.entries()], ['jsonl'])     # today's line only
        b = fr.new_bundle('r9', 'loop')
        out = fr.purge(all=True)
        self.assertIn(os.path.basename(b), out)
        self.assertEqual(fr.entries(), [])
        fr.close()

    def test_cli(self):
        root = os.path.join(scratch(), 'flight')
        fr = F.FlightRecorder(root)
        fr.new_bundle('r1', 'api')
        py = sys.executable
        env = dict(os.environ, HIP_VISIBLE_DEVICES='')
        r = subprocess.run([py, '-I', os.path.join(ROOT, 'tools', 'flight.py'), 'list', '--dir', root], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('-r1-api', r.stdout)
        r = subprocess.run([py, '-I', os.path.join(ROOT, 'tools', 'flight.py'), 'purge', '--dir', root], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 2)               # held by "a server" (this recorder)
        fr.close()
        r = subprocess.run([py, '-I', os.path.join(ROOT, 'tools', 'flight.py'), 'purge', '--dir', root], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(os.listdir(os.path.join(root, 'bundles')), [])


class _StateSession(MockSession):
    """MockSession with K3Session's bundle hooks (bundle_state / request_marks / export_bundle), so the worker's record and
    bundle paths run as on the real engine"""
    def prefill(self, ids, cancel=None, marks=None, max_new=None):
        r = super().prefill(ids, cancel, marks, max_new)
        self._prompt, self._first = list(ids), (None if r is None else list(r[:8]))
        return r

    def request_marks(self):
        return dict(bails=0, recoveries=0)

    def bundle_state(self, prompt=None):
        if getattr(self, '_prompt', None) is None or (prompt is not None and list(prompt) != self._prompt):
            return dict(pe=None, source=None, cached=None, first=None, generated=[])
        return dict(pe=(len(self._prompt) - 1, 1), source=self.last_prefill.source, cached=self.last_prefill.cached_n,
                    first=self._first, generated=list(self.tokens)[len(self._prompt):])

    def export_bundle(self, path, rec, flight):
        return dict(engine=dict(key_sha='mock'), cache_ref=dict(source=rec.get('source'), prompt_end=list(rec['pe']) if rec.get('pe') else None,
                                                               state=None, note='mock state'))


class TestServer(unittest.TestCase):
    session_cls = MockSession

    def setUp(self):
        self.root = os.path.join(scratch(), 'flight')
        self.fr = F.FlightRecorder(self.root)
        self.eng = ScriptedEngine(ntok=4)
        self.sess = self.session_cls(self.eng, max_ctx=32768, eos_ids=(248046, 248044), fail_at=set())
        self.srv = K3Server(self.sess, common.tokenizer(), port=0, log_dir=None, builder=common.builder(), proc_name=None,
                            flight=self.fr).start()

    def tearDown(self):
        self.srv.stop()
        self.fr.close()

    def chat(self, msgs, reply, tools=None, ua=None, **body):
        p, r = U.script(self.eng, msgs, reply, tools)
        b = {'model': 'k3-coder', 'messages': msgs, 'temperature': 0}
        if tools:
            b['tools'] = tools
        b.update(body)
        return U.post_json(self.srv, '/v1/chat/completions', b, headers={'User-Agent': ua} if ua else None)

    def lines(self, n, timeout=10):
        t = time.time() + timeout
        while time.time() < t:
            p = self.fr.day_path()
            if os.path.exists(p):
                rows = [json.loads(x) for x in open(p) if x.strip()]
                if len(rows) >= n:
                    return rows
            time.sleep(0.02)
        return []

    def bundles(self):
        out = []
        for k, p, nb, mt in self.fr.entries():
            if k == 'bundle' and os.path.exists(os.path.join(p, 'bundle.json')):
                out.append(json.load(open(os.path.join(p, 'bundle.json'))))
        return out

    def test_line_and_on_demand_bundle(self):
        msgs = [SYS, {'role': 'user', 'content': 'Flight: say a secret word.'}]
        reply = {'role': 'assistant', 'reasoning_content': 'Think briefly.', 'content': 'Pumpernickel.'}
        st, h, j = self.chat(msgs, reply, ua='QwenCode/0.0.14 (linux; x64)', seed=77)
        p_ids, r_ids = U.script(self.eng, msgs, reply, None)
        self.assertEqual(st, 200)
        rows = self.lines(1)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        for k in ('request_id', 'harness', 'profile', 'sampling', 'prompt_tokens', 'cached_tokens', 'completion_tokens', 'cache',
                  'ttft_split_ms', 'decode_ms_per_token', 'dflash', 'engine', 'finish', 'tool_calls', 'loop', 'verdict', 'status'):
            self.assertIn(k, r)
        self.assertEqual((r['harness'], r['profile'], r['sampling']['seed'], r['status'], r['loop']['flagged']), ('qwen-code', 'k3-coder', 77, 200, False))
        self.assertEqual(r['cache']['source'], 'cold')
        text = open(self.fr.day_path()).read()
        self.assertNotIn('secret word', text)           # no request text
        self.assertNotIn('Pumpernickel', text)
        self.assertEqual(mode(self.fr.day_path()), 0o600)
        # on demand: the most recent request
        st, h, b = U.post_json(self.srv, '/caminus/flight/bundle', {})
        self.assertEqual(st, 200, b)
        self.assertEqual(b['request_id'], r['request_id'])
        meta = json.load(open(os.path.join(b['path'], 'bundle.json')))
        self.assertEqual(meta['reason'], 'api')
        self.assertEqual(meta['request']['generated'], [int(t) for t in r_ids])          # the token log: the reply as generated
        self.assertEqual(meta['request']['prompt'], [int(t) for t in p_ids])
        self.assertEqual(meta['request']['params']['seed'], 77)
        self.assertGreater(len(meta['request']['prompt']), 10)
        if self.session_cls is MockSession:
            self.assertIn('MockSession', meta['cache_ref']['note'])
        else:
            self.assertEqual((meta['cache_ref']['note'], meta['cache_ref']['prompt_end'][0]), ('mock state', len(p_ids) - 1))
            self.assertEqual(meta['first_logits']['n'], 8)
        st, h, e = U.post_json(self.srv, '/caminus/flight/bundle', {'request_id': 'nope'})
        self.assertEqual(st, 404)
        st, h, ls = U.request(self.srv, 'GET', '/caminus/flight')[0], None, json.loads(U.request(self.srv, 'GET', '/caminus/flight')[2])
        self.assertEqual(len([x for x in ls['entries'] if x['kind'] == 'bundle']), 1)
        st, h, pg = U.post_json(self.srv, '/caminus/flight/purge', {})
        self.assertEqual(st, 200)
        self.assertEqual([k for k, *_ in self.fr.entries()], [])

    def test_auto_bundles(self):
        # a reasoning loop (thinking on): the engine thread's verdict -> an automatic 'loop' bundle
        line = 'Maybe the import path is wrong, check it again.'
        self.chat([SYS, {'role': 'user', 'content': 'Flight: why does it fail?'}],
                  {'role': 'assistant', 'reasoning_content': '\n'.join([line] * 4), 'content': 'Fixed the path.'})
        # a tool call of a tool the request did not declare -> 'tool_parse'
        tools = synth.TOOLS
        msgs = [SYS, {'role': 'user', 'content': 'Flight: read b.py.'}]
        reply = {'role': 'assistant', 'reasoning_content': 'Read it.', 'content': '',
                 'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {'name': 'launch_rockets', 'arguments': {'n': 3}}}]}
        p, r = U.script(self.eng, msgs, reply, tools + [{'type': 'function', 'function': {'name': 'launch_rockets', 'description': 'x',
                                                         'parameters': {'type': 'object', 'properties': {'n': {'type': 'integer'}}}}}])
        self.eng.register(U.common.builder().build(msgs, tools, model='k3-coder').ids, r)
        U.post_json(self.srv, '/v1/chat/completions', {'model': 'k3-coder', 'messages': msgs, 'temperature': 0, 'tools': tools})
        # retry exhaustion: the next pass fails -> 'retry_exhausted' before the reset
        self.sess.fail_at = {self.sess.total_passes + 2}
        self.chat([SYS, {'role': 'user', 'content': 'Flight: fail please.'}], {'role': 'assistant', 'reasoning_content': 'x', 'content': 'y' * 40})
        rows = self.lines(3)
        t = time.time() + 10
        while len(self.bundles()) < 3 and time.time() < t:
            time.sleep(0.05)
        reasons = sorted(b['reason'] for b in self.bundles())
        self.assertEqual(reasons, ['loop', 'retry_exhausted', 'tool_parse'], [ (r['verdict'], r['status']) for r in rows])
        self.assertEqual(rows[0]['verdict'], 'loop')
        self.assertTrue(rows[0]['loop']['flagged'])
        self.assertEqual(rows[1]['verdict'], 'tool_parse')
        self.assertEqual(rows[1]['tool_calls']['failures']['undeclared'], 1)
        self.assertEqual(rows[2]['status'], 503)
        for b in self.bundles():
            self.assertEqual(b['format'], 1)


class TestServerStateSession(TestServer):
    """the same with a session that has K3Session's bundle hooks (the record merges its state; the GPU run's TypeError)"""
    session_cls = _StateSession


class TestRoundTrip(unittest.TestCase):
    """a bundle's sampling parameters survive bundle.json (tools/replay.py rebuilds SamplingParams(**params)): every field,
    the defaults and each one set, through json.dumps / loads"""

    def test_sampling_params(self):
        import dataclasses
        from caminus.serve.sampler import SamplingParams
        base = SamplingParams().resolve()
        cases = [base]
        for f in dataclasses.fields(SamplingParams):
            v = getattr(base, f.name)
            if isinstance(v, bool):
                nv = not v
            elif isinstance(v, int):
                nv = v + 3
            elif isinstance(v, float):
                nv = v + 0.25
            else:
                continue
            cases.append(dataclasses.replace(base, **{f.name: nv}))
        for p in cases:
            d = json.loads(json.dumps(dataclasses.asdict(p), default=str))
            self.assertEqual(SamplingParams(**d), p, d)


if __name__ == '__main__':
    unittest.main()
