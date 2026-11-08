"""Server operations (impl:m7-ops), CPU only, on MockSession: the clean shutdown of `caminus serve` (cli.shutdown: stop
accepting, cancel the running job at its next pass boundary, the queued ones before they start, wait for the engine thread,
park the live conversation within a bound, release the cache), the worker's cancel_all, the systemd unit file's safety lines.
Run:  HIP_VISIBLE_DEVICES= python -I tests/serve/run_all.py"""
import os
import threading
import time
import unittest

import common  # noqa: F401  (sys.path)
import srvutil as U
from caminus.serve import cli
from caminus.serve.engine import MockSession

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _ParkSession(MockSession):
    """MockSession with K3Session's shutdown_park hook (records its calls; optionally slow)"""
    park_sleep = 0.0

    def shutdown_park(self):
        self.parks = getattr(self, 'parks', 0) + 1
        self.park_thread = threading.current_thread().name
        time.sleep(self.park_sleep)
        return 'parked-entry'


class _PC:
    def __init__(self):
        self.closed, self.last_park, self.park_error = 0, {'bytes': 1 << 20}, None

    def close(self):
        self.closed += 1


def _server(sess):
    tok = U.common.tokenizer()
    from caminus.serve.server import K3Server
    return K3Server(sess, tok, port=0, log_dir=None, builder=U.common.builder(), proc_name=None).start()


class TestShutdown(unittest.TestCase):
    def _run(self, park_sleep, park_s):
        sess = _ParkSession(U.ScriptedEngine(ntok=4), max_ctx=32768, pass_sleep=0.02)
        sess.park_sleep = park_sleep
        srv = _server(sess)
        out = {}

        def client():
            out['r'] = U.post_json(srv, '/v1/completions', {'prompt': 'count to a thousand', 'max_tokens': 3000, 'temperature': 0})
        t = threading.Thread(target=client); t.start()
        t0 = time.time()
        while not srv.worker.busy and time.time() - t0 < 10:
            time.sleep(0.01)
        time.sleep(0.2)                                  # (a few passes in)
        queued = srv.submit(srv.completion_spec({'prompt': 'second', 'max_tokens': 8}))
        pc = _PC()
        logs = []
        ts = time.time()
        res = cli.shutdown(srv, sess, pc, drain_s=10.0, park_s=park_s, log=lambda *a, **k: logs.append(a[0]))
        dt = time.time() - ts
        t.join(10)
        return sess, srv, pc, res, dt, out, queued, logs

    def test_clean_shutdown(self):
        sess, srv, pc, res, dt, out, queued, logs = self._run(0.0, 5.0)
        self.assertFalse(srv.worker.thread.is_alive())                         # the engine thread exited
        self.assertEqual(srv.worker.history[-1][0] if srv.worker.history else None, 'completions')
        r = out.get('r')
        self.assertIsNotNone(r)
        self.assertEqual(r[0], 200)
        self.assertEqual(queued.result.finish, 'cancel')                       # queued: never reached the engine
        self.assertEqual(queued.result.gen_ids, [])
        self.assertEqual((sess.parks, res, pc.closed), (1, 'parked-entry', 1))  # parked once, from the caller's thread, then closed
        self.assertNotEqual(sess.park_thread, 'k3-engine')
        self.assertLess(dt, 5.0, logs)
        self.assertTrue(any('park' in x for x in logs), logs)

    def test_park_bound(self):
        sess, srv, pc, res, dt, out, queued, logs = self._run(3.0, 0.5)
        self.assertIsNone(res)
        self.assertLess(dt, 2.5)                                               # the bound, not the park's 3 s
        self.assertTrue(any('did not finish within' in x for x in logs), logs)


class TestUnitFile(unittest.TestCase):
    """packaging/systemd/caminus.service: a FILE in the repo (never installed by a test); its safety lines"""

    def test_unit(self):
        p = os.path.join(ROOT, 'packaging', 'systemd', 'caminus.service')
        with open(p) as f:
            u = f.read()
        lines = [x.strip() for x in u.splitlines() if x.strip() and not x.strip().startswith('#')]
        kv = {}
        for x in lines:
            if '=' in x and not x.startswith('['):
                k, v = x.split('=', 1)
                kv.setdefault(k, []).append(v)
        self.assertIn('fn-server.service', ' '.join(kv['Conflicts']))
        self.assertEqual(kv['Restart'], ['no'])                                 # never restarted after a fault
        self.assertIn('!%h/.local/state/k3/GPU_FAULT_STOP', kv['ConditionPathExists'])
        es = kv['ExecStart'][0]
        self.assertIn('tools/guard.sh', es)                                     # under the guard, as every GPU process
        self.assertIn('--port 8092', es)
        self.assertIn(' -I ', es)
        self.assertEqual(kv['KillSignal'], ['SIGTERM'])
        self.assertGreaterEqual(int(kv['TimeoutStopSec'][0]), int(cli.DEFAULTS['drain_s'] + cli.DEFAULTS['park_s']))
        self.assertEqual(int(cli.DEFAULTS['idle_park_s']), 28800)               # 8 h (user, 11-05)
        # final round 2: a stop signals the server alone (ExecStop: caminus-stop TERMs it by PID and waits for the guard), so
        # the guard keeps watching and prints its rc line and gpu-high resets the clocks; KillMode=mixed (systemd's own SIGTERM
        # only to a main process still running after ExecStop). Its wait covers the drain + park and stays under TimeoutStopSec
        self.assertEqual(kv['KillMode'], ['mixed'])
        # ($MAINPID last: systemd drops the word when the main process has already exited, so the wait must come first)
        stop = kv['ExecStop'][0].split()
        self.assertEqual((stop[0], stop[2]), ('%h/src/caminus/packaging/systemd/caminus-stop', '$MAINPID'))
        script = os.path.join(ROOT, 'packaging', 'systemd', 'caminus-stop')
        self.assertTrue(os.access(script, os.X_OK))
        wait = int(stop[1])
        self.assertGreaterEqual(wait, int(cli.DEFAULTS['drain_s'] + cli.DEFAULTS['park_s']))
        self.assertLess(wait, int(kv['TimeoutStopSec'][0]))


class TestKvCacheCli(unittest.TestCase):
    """tools/kv_cache.py list / purge on a directory a server holds (its lock): refused with a one-line message and exit 2, no
    traceback (final round 3; before: a CacheError traceback); a free directory lists"""

    def test_refusal(self):
        import subprocess
        import sys
        import tempfile
        from caminus.cache import prompt as pc
        tool = os.path.join(ROOT, 'tools', 'kv_cache.py')
        env = dict(os.environ, HIP_VISIBLE_DEVICES='')
        with tempfile.TemporaryDirectory() as d:
            held = pc.DiskStore(d, staging=pc.Staging(piece=pc.PIECE, nbuf=1, pin=False))   # (as a running server holds it)
            try:
                for cmd in ('list', 'purge'):
                    r = subprocess.run([sys.executable, '-I', tool, cmd, '--dir', d], capture_output=True, text=True, timeout=120, env=env)
                    self.assertEqual(r.returncode, 2, r.stderr)
                    self.assertNotIn('Traceback', r.stderr)
                    self.assertIn(f'kv_cache: {cmd} refused', r.stderr)
                    self.assertEqual(len(r.stderr.strip().splitlines()), 1, r.stderr)
            finally:
                held.close()
            r = subprocess.run([sys.executable, '-I', tool, 'list', '--dir', d], capture_output=True, text=True, timeout=120, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('0 entries', r.stdout)


class _FakePC:
    """the prompt cache's park bookkeeping as K3Session.idle / shutdown_park see it (pc.parked = ((tok_sha(log), i0, commit),
    name) once a park landed; park() of an unchanged slot returns its name without writing)"""
    def __init__(self, log, fail=False):
        self.log, self.st, self.parked, self.side, self.store = list(log), dict(i0=len(log), commit=1), None, None, object()
        self.auto_park_min, self.writes, self.fail, self.park_error = 4096, 0, fail, None

    def fence(self):
        pass

    def live_tokens(self):
        return self.log[:self.st['i0']]

    def park(self, background=False):
        from caminus.cache.prompt import tok_sha
        sig = (tok_sha(self.log), len(self.log), 1)
        if self.parked is not None and self.parked[0] == sig:
            return None if background else self.parked[1]
        if self.fail:
            self.park_error = 'park refused: cap'
            return None
        self.writes += 1
        self.parked = (sig, f'park-{self.writes}')
        return None if background else self.parked[1]


class TestIdlePark(unittest.TestCase):
    """K3Session.idle / shutdown_park's rules on a stand-in cache (the GPU stage test_serve_k3 bgpark runs the real one): an
    idle park only when the line gained >= auto_park_min positions since its last park, a shutdown park when it changed at all"""

    def _sess(self, pc):
        from caminus.serve.k3_session import K3Session
        s = K3Session.__new__(K3Session)
        s.pc, s._park_mark, s._park_pending = pc, None, None   # (the session reads the log through pc)
        s.counters = {'parks': 0, 'idle_skipped': 0}
        return s

    def test_rules(self):
        pc = _FakePC(range(3000))
        s = self._sess(pc)
        s.idle()                                                 # 3,000 < 4,096: not worth a write
        self.assertEqual((pc.writes, s.counters['idle_skipped']), (0, 1))
        self.assertEqual(s.shutdown_park(), 'park-1')            # a shutdown parks any change
        self.assertEqual(s.shutdown_park(), 'unchanged')
        pc.log = list(range(3000 + 4095)); pc.st = dict(i0=len(pc.log), commit=1)
        s.idle()                                                 # gained 4,095 since the shutdown park: skipped
        self.assertEqual(pc.writes, 1)
        pc.log = list(range(3000 + 4096)); pc.st = dict(i0=len(pc.log), commit=1)
        s.idle()                                                 # gained 4,096: a background park
        self.assertEqual((pc.writes, s.counters['parks']), (2, 2))
        s.idle()                                                 # landed (pc.parked): nothing gained since
        self.assertEqual(pc.writes, 2)
        pc.log = [7] + list(range(5000)); pc.st = dict(i0=len(pc.log), commit=1)  # another line of 5,001 positions: its whole log counts
        s.idle()
        self.assertEqual(pc.writes, 3)

    def test_failed_park_is_retried(self):
        pc = _FakePC(range(5000), fail=True)
        s = self._sess(pc)
        s.idle()
        self.assertEqual((pc.writes, s._park_mark), (0, None))  # refused: no mark, so the next idle tries again
        pc.fail = False
        s.idle()
        self.assertEqual(pc.writes, 1)


if __name__ == '__main__':
    unittest.main()
