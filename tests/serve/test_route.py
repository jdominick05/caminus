"""M7 finish (impl:m7-finish), CPU only, on MockSession: the request's max_new reaches a session that asks for it
(wants_max_new: K3Session's side-slot routing needs to know a side call's generation fits the side slot), the routing decision
in /metrics (caminus:cache_route_total, caminus:cache_side_active) and /caminus/status (cache.route), the flight line's
cache.slot; caminus serve's --log-dir off by default and --no-side-route; a client that resets its keep-alive connection draws
no traceback (the server's handle_error), any other handler error still does.
Run:  HIP_VISIBLE_DEVICES= python -I tests/serve/run_all.py"""
import contextlib
import io
import json
import socket
import struct
import time
import unittest

import common  # noqa: F401  (sys.path)
import srvutil as U
from caminus.serve import cli
from caminus.serve.engine import MockSession


class _RouteSession(MockSession):
    """MockSession that asks for max_new (as K3Session does) and reports a routing decision per request: 'side' for a prompt
    shorter than side_below tokens, else 'main' (a stand-in for PromptCache.route; the K3 session's own rule is tested in
    tests/test_prompt_cache.py cpu_route and on the GPU in tests/test_serve_k3.py side)"""
    wants_max_new = True
    side_below = 0

    def plan(self, ids, max_new=None):
        self.plans = getattr(self, 'plans', []) + [(len(ids), max_new)]
        return super().plan(ids)

    def prefill(self, ids, cancel=None, marks=None, max_new=None):
        self.prefills = getattr(self, 'prefills', []) + [(len(ids), max_new)]
        side = len(ids) < self.side_below
        dec = dict(route='side' if side else 'main', reason='unrelated' if side else 'match', shared=3, root=None,
                   need=len(ids) + int(max_new) + self.ntok, main_n=100, side_rows=8192)
        if self.stats is not None:
            self.stats.route(dec)
            self.stats.cache_update(None, {'side_active': side, 'side_rows': 8192})
        out = super().prefill(ids, cancel, marks=marks)
        self.last_prefill.slot = 'side' if side else 'main'
        return out


def _server(sess, flight=None):
    from caminus.serve.server import K3Server
    return K3Server(sess, U.common.tokenizer(), port=0, log_dir=None, builder=U.common.builder(), proc_name=None,
                    flight=flight).start()


class TestMaxNew(unittest.TestCase):
    def test_max_new_reaches_the_session(self):
        eng = U.ScriptedEngine(ntok=4)
        sess = _RouteSession(eng, max_ctx=4096, eos_ids=(248046, 248044))
        srv = _server(sess)
        try:
            ids = [248044] + list(range(1000, 1040))
            st, _, j = U.post_json(srv, '/v1/completions', {'prompt': ids, 'max_tokens': 7, 'temperature': 0})
            self.assertEqual(st, 200)
            msgs = [{'role': 'user', 'content': 'Say hi.'}]
            st2, _, _ = U.post_json(srv, '/v1/chat/completions', {'messages': msgs, 'temperature': 0})   # no max_tokens
            self.assertEqual(st2, 200)
        finally:
            srv.stop()
        # the worker's plan and run_generation's prefill both get the job's max_new: 7, then the context room (no max_tokens)
        self.assertEqual(sess.plans[0], (41, 7))
        self.assertEqual(sess.prefills[0], (41, 7))
        n2 = sess.prefills[1][0]
        self.assertEqual(sess.plans[1][1], sess.prefills[1][1])
        self.assertEqual(sess.prefills[1][1], sess.max_ctx - n2)

    def test_plain_session_unchanged(self):
        """a session without wants_max_new is called as before (no max_new keyword)"""
        eng = U.ScriptedEngine(ntok=4)
        sess = MockSession(eng, max_ctx=4096, eos_ids=(248046, 248044))
        calls = []
        orig = sess.prefill

        def prefill(ids, cancel=None, marks=None, **kw):
            calls.append(kw)
            return orig(ids, cancel, marks=marks)
        sess.prefill = prefill
        srv = _server(sess)
        try:
            st, _, _ = U.post_json(srv, '/v1/completions', {'prompt': [248044, 5, 6, 7], 'max_tokens': 3, 'temperature': 0})
        finally:
            srv.stop()
        self.assertEqual(st, 200)
        self.assertEqual(calls, [{}])


class TestRouteMetrics(unittest.TestCase):
    def test_counters_status_and_flight_slot(self):
        import tempfile
        from caminus.serve.flight import FlightRecorder
        eng = U.ScriptedEngine(ntok=4)
        sess = _RouteSession(eng, max_ctx=4096, eos_ids=(248046, 248044))
        sess.side_below = 20
        fl = FlightRecorder(tempfile.mkdtemp(prefix='k3-route-flight-'))
        srv = _server(sess, flight=fl)
        try:
            long_ = [248044] + list(range(2000, 2040))
            short = [248044] + list(range(3000, 3010))
            for ids in (long_, short, long_, short, short):
                st, _, _ = U.post_json(srv, '/v1/completions', {'prompt': ids, 'max_tokens': 4, 'temperature': 0})
                self.assertEqual(st, 200)
            text = U.request(srv, 'GET', '/metrics')[2].decode()
            s = json.loads(U.request(srv, 'GET', '/caminus/status')[2])
            time.sleep(0.2)
            with open(fl.day_path(), encoding='utf-8') as f:
                rows = [json.loads(x) for x in f if x.strip()]
        finally:
            srv.stop()
            fl.close()
        r = s['cache']['route']
        self.assertEqual(r['side']['unrelated'], 3)
        self.assertEqual(r['main']['match'], 2)
        self.assertEqual(sum(r['main'].values()), 2)
        self.assertEqual(r['last']['route'], 'side')
        self.assertEqual(r['last']['need'], 11 + 4 + 4)
        self.assertEqual(r['side_active'], True)
        self.assertEqual(r['side_rows'], 8192)
        self.assertIn('caminus:cache_route_total{route="side",reason="unrelated"} 3', text)
        self.assertIn('caminus:cache_route_total{route="main",reason="match"} 2', text)
        self.assertIn('caminus:cache_route_total{route="main",reason="no_fit"} 0', text)
        self.assertIn('caminus:cache_side_active 1', text)
        self.assertEqual([x['cache']['slot'] for x in rows], ['main', 'side', 'main', 'side', 'side'])

    def test_unknown_reason(self):
        from caminus.serve.stats import EngineStats
        st = EngineStats()
        st.route(dict(route='side', reason='nonsense'))
        st.route(dict(route='bogus', reason='match'))
        st.route(None)
        self.assertEqual(st.cache_route['side']['unrelated'], 1)
        self.assertEqual(st.cache_route['main']['match'], 1)
        off = EngineStats(enabled=False)
        off.route(dict(route='side', reason='unrelated'))
        self.assertEqual(off.cache_route['side']['unrelated'], 0)


class TestCli(unittest.TestCase):
    def test_log_dir_off_and_side_route_flag(self):
        o = cli.parse([])
        self.assertEqual(o['log_dir'], '')
        self.assertFalse(o['no_side_route'])
        o = cli.parse(['--no-side-route', '--log-dir', '/x/y'])
        self.assertTrue(o['no_side_route'])
        self.assertEqual(o['log_dir'], '/x/y')


class TestQuietReset(unittest.TestCase):
    def test_reset_connection_draws_no_traceback(self):
        """a keep-alive client that resets its connection (RST) while the handler waits for its next request: before the fix
        socketserver printed 'Exception occurred during processing of request' with a ConnectionResetError traceback (seen at
        caminus serve's shutdown); a real handler error still prints"""
        eng = U.ScriptedEngine(ntok=4)
        srv = _server(MockSession(eng, max_ctx=4096, eos_ids=(248046, 248044)))
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                for _ in range(3):
                    s = socket.create_connection(('127.0.0.1', srv.port))
                    s.sendall(b'GET /health HTTP/1.1\r\nHost: x\r\n\r\n')
                    buf = b''
                    while b'\r\n\r\n' not in buf:
                        buf += s.recv(4096)
                    time.sleep(0.05)
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))   # close() sends an RST
                    s.close()
                time.sleep(0.3)
                quiet = err.getvalue()
                try:
                    raise ValueError('a real handler error')
                except ValueError:
                    srv.httpd.handle_error(None, ('127.0.0.1', 0))
        finally:
            srv.stop()
        self.assertNotIn('ConnectionResetError', quiet)
        self.assertNotIn('Exception occurred', quiet)
        self.assertIn('ValueError: a real handler error', err.getvalue())


if __name__ == '__main__':
    unittest.main()
