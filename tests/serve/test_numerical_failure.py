"""Terminal numerical failures through real CPU workers; no vocab, server, sockets or GPU.

python -I tests/serve/test_numerical_failure.py

Startup/shutdown use the actual CLI functions extracted by AST, with module
imports and engine/cache construction replaced by CPU fakes.
"""
import ast
import collections
import contextlib
import io
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from caminus.errors import InvalidKVEncoding, NonFiniteLogits, NumericalFailure
from caminus.cache.livestate import is_bail
from caminus.serve.engine import EngineError, MockSession
from caminus.serve.generate import run_generation
from caminus.serve.k3_session import K3Session
from caminus.serve.mock_engine import MockEngine
from caminus.serve import sampler
from caminus.serve.server import K3Handler
from caminus.serve.stats import EngineStats
from caminus.serve.worker import BundleJob, EngineBroken, EngineWorker, Job


class BytesTokenizer:
    def decode_bytes(self, ids, skip_special=False):
        return bytes(int(i) % 128 for i in ids)


class RefusingEngine(MockEngine):
    def __init__(self, error):
        self.error = error
        self.entered, self.release = threading.Event(), threading.Event()
        self.verify_calls = 0
        super().__init__(ntok=4)

    def verify(self, *args, **kwargs):
        self.verify_calls += 1
        self.entered.set()
        if not self.release.wait(5):
            raise AssertionError('test did not release the mock pass')
        raise self.error


def job():
    return Job([1, 2, 3, 4], sampler.SamplingParams(temperature=0.0), 8)


def terminal(j):
    events = []
    while True:
        event = j.events.get(timeout=5)
        events.append(event)
        if event[0] in ('error', 'done'):
            return events


def cli_functions():
    path = ROOT / 'python/caminus/serve/cli.py'
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('build_engine', 'shutdown')
            or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'EXIT_BROKEN' for t in n.targets)]
    scope = dict(__package__='caminus.serve', __name__='numerical_cli_fixture', time=time, threading=threading)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), scope)
    return scope


class TerminalFailures(unittest.TestCase):
    def health(self, worker):
        # Call the real health handler without constructing BaseHTTPRequestHandler
        # or an HTTPServer: no socket or listener exists.
        handler = K3Handler.__new__(K3Handler)
        handler.server_obj = NS(worker=worker, loading=False, count=Mock())
        handler.path = '/health'
        handler._read_body, handler._send = Mock(), Mock()
        handler.do_GET()
        return handler._send.call_args.args

    def test_hierarchy_preserves_nonfinite_runtimeerror_compatibility(self):
        for cls in (InvalidKVEncoding, NonFiniteLogits):
            self.assertTrue(issubclass(cls, NumericalFailure))
            self.assertTrue(issubclass(cls, RuntimeError))
        self.assertFalse(issubclass(InvalidKVEncoding, NonFiniteLogits))

    def test_terminal_mock_pass_breaks_worker_once_without_recovery_or_park(self):
        errors = [cls('synthetic terminal state: a wait gave up during Q8 encoding')
                  for cls in (InvalidKVEncoding, NonFiniteLogits, NumericalFailure)] + [InvalidKVEncoding()]
        for error in errors:
            with self.subTest(error=repr(error)), contextlib.redirect_stderr(io.StringIO()):
                eng = RefusingEngine(error)
                reason = str(error) or type(error).__name__
                sess = MockSession(eng, max_ctx=4096)
                sess.plan, sess.prefill = Mock(wraps=sess.plan), Mock(wraps=sess.prefill)
                sess.reset = Mock(side_effect=AssertionError('terminal error must not reset'))
                sess.abort = Mock(side_effect=AssertionError('terminal error must not abort-recover'))
                idle_called = threading.Event()
                sess.idle = Mock(side_effect=idle_called.set)
                sess.shutdown_park = Mock(side_effect=AssertionError('terminal error must not park'))
                worker = EngineWorker(sess, BytesTokenizer(), idle_s=0.005, flight=NS())
                worker._record, worker._bundle = Mock(), Mock()
                worker.recent['prior-request'] = dict(request_id='prior-request')
                hits, notified = [], threading.Event()

                def broken(reason):
                    hits.append(reason)
                    notified.set()

                worker.on_broken = broken
                jobs = [job(), job(), job()]
                bundle = BundleJob('prior-request')
                try:
                    worker.submit(jobs[0])
                    self.assertTrue(eng.entered.wait(5))
                    worker.submit(jobs[1])
                    worker.submit(bundle)
                    worker.submit(jobs[2])
                    eng.release.set()
                    events = [terminal(j) for j in jobs]
                    self.assertEqual(terminal(bundle)[-1][0:2], ('error', 'engine'))
                    self.assertTrue(notified.wait(5))
                    self.assertEqual(len(hits), 1)
                    self.assertEqual(hits[0], reason)
                    self.assertEqual(worker.broken, reason)
                    self.assertTrue(all(es[-1][0:2] == ('error', 'engine') for es in events))
                    self.assertTrue(all('broken' in es[-1][2] for es in events))
                    self.assertFalse(any(e[0] in ('text', 'done') for es in events for e in es))
                    self.assertEqual([e[0] for e in events[1]], ['error'])
                    self.assertEqual([e[0] for e in events[2]], ['error'])
                    with self.assertRaises(EngineBroken):
                        worker.submit(job())
                    self.assertEqual(eng.verify_calls, 1)
                    self.assertEqual(sess.plan.call_count, 1)
                    self.assertEqual(sess.prefill.call_count, 1)
                    self.assertEqual(worker.stats['engine_errors'], 1)
                    self.assertEqual((worker.stats['internal_errors'], worker.stats['aborts'], worker.stats['resets']), (0, 0, 0))
                    status, body = self.health(worker)
                    self.assertEqual(status, 503)
                    self.assertIn(reason, body['error']['message'])
                    self.assertFalse(idle_called.wait(0.025))
                    sess.reset.assert_not_called()
                    sess.abort.assert_not_called()
                    sess.shutdown_park.assert_not_called()
                    worker._record.assert_not_called()
                    worker._bundle.assert_not_called()
                finally:
                    eng.release.set()
                    self.assertTrue(worker.stop(timeout=5), 'worker thread must exit')

    def test_normal_bundle_job_still_dispatches_before_any_failure(self):
        sess = MockSession(MockEngine(ntok=4), max_ctx=4096)
        worker = EngineWorker(sess, BytesTokenizer(), idle_s=None, flight=NS())
        record = dict(request_id='prior-request')
        worker.recent['prior-request'] = record
        worker._bundle = Mock(return_value=('synthetic-bundle', dict(cache_ref={'source': 'synthetic'})))
        try:
            bundle = worker.submit(BundleJob('prior-request'))
            events = terminal(bundle)
            self.assertEqual(events[-1][0], 'done')
            self.assertEqual(bundle.result['path'], 'synthetic-bundle')
            worker._bundle.assert_called_once_with(record, 'api')
            self.assertIsNone(worker.broken)
        finally:
            self.assertTrue(worker.stop(timeout=5))

    def test_original_nonfinite_prompt_and_sampler_still_raise_nonfinite_logits(self):
        class NaNEngine(MockEngine):
            def logits(self, prefix):
                return np.full(self.V, np.nan, dtype=np.float32)

        for temperature in (0.0, 0.7):
            sess = MockSession(NaNEngine(ntok=4), max_ctx=4096)
            with self.subTest(temperature=temperature), self.assertRaises(NonFiniteLogits):
                run_generation(sess, [1, 2, 3, 4], sampler.Sampler(sampler.SamplingParams(temperature=temperature)), 8, lambda ids: None)
        weights = np.array([np.nan, np.nan], dtype=np.float32)
        with self.assertRaises(NonFiniteLogits):
            sampler.draw(sampler.Candidates(np.array([3, 5]), weights, weights, weights), 0.5)

    def test_numerical_diagnostics_never_trigger_timeout_recovery(self):
        for cls in (NumericalFailure, InvalidKVEncoding, NonFiniteLogits):
            self.assertFalse(is_bail(cls('a wait gave up: context for a terminal failure')))
        self.assertTrue(is_bail(RuntimeError('a wait gave up: a retryable engine timeout')))
        self.assertFalse(is_bail(ValueError('a wait gave up')))

    def k3_prompt_session(self, error, phase='prepare'):
        sess = K3Session.__new__(K3Session)
        sess.pc = NS(prompt=Mock(side_effect=error), phase=phase, live=NS(known=True), live_tokens=lambda: [])
        sess.stats, sess.pf = None, NS(last=None)
        sess.side_route, sess.max_recoveries, sess._hooks = False, 2, NS()
        sess._refused = Mock(side_effect=AssertionError('terminal prompt must not be reported as a recoverable refusal'))
        return sess

    def test_actual_k3_prompt_preserves_terminal_type_and_invalidates_live_state(self):
        for cls in (InvalidKVEncoding, NonFiniteLogits):
            for phase in ('prepare', 'first', None):
                with self.subTest(error=cls.__name__, phase=phase):
                    error = cls('a wait gave up: invalid numerical state')
                    sess = self.k3_prompt_session(error, phase)
                    with self.assertRaises(cls) as caught:
                        sess.prefill([1, 2, 3])
                    self.assertIs(caught.exception, error)
                    self.assertFalse(sess.pc.live.known)
                    sess._refused.assert_not_called()
                    self.assertEqual(sess.pc.prompt.call_count, 1)

    def test_actual_k3_prompt_failure_reaches_worker_as_terminal(self):
        for cls in (InvalidKVEncoding, NonFiniteLogits):
            with self.subTest(error=cls.__name__), contextlib.redirect_stderr(io.StringIO()):
                error = cls('a wait gave up: refused partial KV cache')
                sess = self.k3_prompt_session(error)
                sess.plan, sess.request_marks = Mock(return_value=0), Mock(return_value={})
                sess.eos_ids, sess.ntok = (), 1
                sess.reset, sess.abort, sess.idle = Mock(), Mock(), Mock()
                worker = EngineWorker(sess, BytesTokenizer(), idle_s=None)
                hits, notified = [], threading.Event()
                worker.on_broken = lambda why: (hits.append(why), notified.set())
                try:
                    j = worker.submit(job())
                    self.assertEqual(terminal(j)[-1][0:2], ('error', 'engine'))
                    self.assertTrue(notified.wait(5))
                    self.assertEqual(hits, [str(error)])
                    self.assertEqual(worker.broken, str(error))
                    self.assertFalse(sess.pc.live.known)
                    sess.reset.assert_not_called()
                    sess.abort.assert_not_called()
                    sess._refused.assert_not_called()
                finally:
                    self.assertTrue(worker.stop(timeout=5))

    def test_other_prompt_runtime_failure_keeps_recoverable_engine_error(self):
        sess = self.k3_prompt_session(RuntimeError('ordinary prefill failure'))
        with self.assertRaisesRegex(EngineError, 'prefill: ordinary prefill failure'):
            sess.prefill([1, 2, 3])

    def abort_session(self, error, stage='target'):
        sess = K3Session.__new__(K3Session)
        sess.counters = collections.Counter()
        sess.pc = NS(launched=Mock(return_value=stage != 'draft'), live=NS(known=True),
                     settle=Mock(return_value='kept'))
        sess.m = NS(finish_pass=Mock(side_effect=error if stage == 'target' else None))
        sess.df = NS(token=2, step_wait=Mock(side_effect=error if stage == 'draft' else None))
        sess._step_ep, sess._inflight = 1, 'step'
        sess.pc.st = dict(i0=3, commit=1)
        sess._sync, sess._pass, sess._dev = Mock(), None, None
        sess._forget = Mock()
        return sess

    def test_actual_abort_preserves_terminal_drain_and_skips_settle(self):
        for cls in (InvalidKVEncoding, NonFiniteLogits):
            for stage in ('target', 'draft'):
                with self.subTest(error=cls.__name__, stage=stage):
                    error = cls('invalid cache discovered during drain')
                    sess = self.abort_session(error, stage)
                    with self.assertRaises(cls) as caught:
                        sess.abort()
                    self.assertIs(caught.exception, error)
                    self.assertFalse(sess.pc.live.known)
                    sess.pc.settle.assert_not_called()
                    sess._sync.assert_not_called()
                    sess._forget.assert_not_called()

    def test_ordinary_drain_error_still_settles_as_not_drained(self):
        sess = self.abort_session(RuntimeError('ordinary drain failure'))
        sess.abort()
        sess.pc.settle.assert_called_once_with(returned=False, drained=False)
        sess._sync.assert_called_once()
        self.assertTrue(sess.pc.live.known)

    def test_actual_abort_sync_and_settle_terminal_failures_preserve_type(self):
        for stage in ('sync', 'settle'):
            with self.subTest(stage=stage):
                error = InvalidKVEncoding('terminal ' + stage)
                sess = self.abort_session(None)
                (sess._sync if stage == 'sync' else sess.pc.settle).side_effect = error
                with self.assertRaises(InvalidKVEncoding) as caught:
                    sess.abort()
                self.assertIs(caught.exception, error)
                self.assertFalse(sess.pc.live.known)
                sess._forget.assert_not_called()
                if stage == 'sync':
                    sess.pc.settle.assert_not_called()

    def test_actual_abort_terminal_reaches_worker_without_reset_or_more_work(self):
        with contextlib.redirect_stderr(io.StringIO()):
            error = InvalidKVEncoding('partial append discovered by abort')
            sess = self.abort_session(error)
            entered, release, notified = threading.Event(), threading.Event(), threading.Event()

            def drain(*args):
                entered.set()
                if not release.wait(5):
                    raise AssertionError('test did not release the drain')
                raise error

            sess.m.finish_pass.side_effect = drain
            sess.plan, sess.prefill = Mock(return_value=0), Mock(side_effect=ValueError('host error after async launch'))
            sess.eos_ids, sess.ntok = (), 1
            sess.request_marks, sess.reset, sess.idle = Mock(return_value={}), Mock(), Mock()
            worker = EngineWorker(sess, BytesTokenizer(), idle_s=None)
            hits = []
            worker.on_broken = lambda why: (hits.append(why), notified.set())
            try:
                first = worker.submit(job())
                self.assertTrue(entered.wait(5))
                queued, bundle = worker.submit(job()), worker.submit(BundleJob())
                release.set()
                for j in (first, queued, bundle):
                    self.assertEqual(terminal(j)[-1][0:2], ('error', 'engine'))
                self.assertTrue(notified.wait(5))
                self.assertEqual(hits, [str(error)])
                self.assertEqual(self.health(worker)[0], 503)
                self.assertFalse(sess.pc.live.known)
                sess.pc.settle.assert_not_called()
                sess.reset.assert_not_called()
                self.assertEqual(sess.prefill.call_count, 1)
                self.assertEqual(worker.stats['resets'], 0)
                with self.assertRaises(EngineBroken):
                    worker.submit(job())
            finally:
                release.set()
                self.assertTrue(worker.stop(timeout=5))

    def test_abort_and_reset_fallback_terminal_failures_break_worker(self):
        for stage in ('abort', 'reset', 'engine_reset'):
            with self.subTest(stage=stage), contextlib.redirect_stderr(io.StringIO()):
                sess = MockSession(MockEngine(ntok=4), max_ctx=4096)
                sess.prefill = Mock(side_effect=EngineError('ordinary engine error') if stage == 'engine_reset'
                                    else ValueError('host failure'))
                sess.abort = Mock(side_effect=InvalidKVEncoding('terminal abort') if stage == 'abort'
                                  else RuntimeError('ordinary abort failure'))
                sess.reset = Mock(side_effect=InvalidKVEncoding('terminal reset'))
                stats = EngineStats(enabled=True, window=4)
                worker = EngineWorker(sess, BytesTokenizer(), idle_s=None, stats=stats)
                notified = threading.Event()
                worker.on_broken = lambda why: notified.set()
                try:
                    self.assertEqual(terminal(worker.submit(job()))[-1][0:2], ('error', 'engine'))
                    self.assertTrue(notified.wait(5))
                    self.assertEqual(self.health(worker)[0], 503)
                    self.assertEqual((stats.requests, stats.finish['error']), (1, 1))
                    self.assertEqual((stats.engine_errors, stats.internal_errors), (1, 0))
                    self.assertEqual((stats.phase, stats.processing), ('idle', 0))
                    if stage == 'abort':
                        sess.reset.assert_not_called()
                    else:
                        sess.reset.assert_called_once()
                finally:
                    self.assertTrue(worker.stop(timeout=5))

    def test_actual_idle_terminal_failure_is_not_swallowed(self):
        sess = K3Session.__new__(K3Session)
        error = InvalidKVEncoding('terminal idle fence')
        sess.pc = NS(store=object(), fence=Mock(side_effect=error), live=NS(known=True))
        sess.pc.live_tokens = lambda: [1]
        with self.assertRaises(InvalidKVEncoding) as caught:
            sess.idle()
        self.assertIs(caught.exception, error)
        self.assertFalse(sess.pc.live.known)

    def test_ordinary_worker_abort_fallback_keeps_internal_error(self):
        for abort_error in (None, RuntimeError('ordinary abort failure')):
            with self.subTest(abort_error=repr(abort_error)), contextlib.redirect_stderr(io.StringIO()):
                sess = MockSession(MockEngine(ntok=4), max_ctx=4096)
                sess.prefill = Mock(side_effect=ValueError('host failure'))
                sess.abort, sess.reset = Mock(side_effect=abort_error), Mock()
                stats = EngineStats(enabled=True, window=4)
                worker = EngineWorker(sess, BytesTokenizer(), idle_s=None, stats=stats)
                try:
                    self.assertEqual(terminal(worker.submit(job()))[-1][0:2], ('error', 'internal'))
                    self.assertIsNone(worker.broken)
                    sess.abort.assert_called_once()
                    self.assertEqual(sess.reset.call_count, int(abort_error is not None))
                    self.assertEqual(worker.stats['engine_errors'], 0)
                    self.assertEqual((stats.requests, stats.finish['error']), (1, 1))
                    self.assertEqual((stats.engine_errors, stats.internal_errors), (0, 1))
                    self.assertEqual((stats.phase, stats.processing), ('idle', 0))
                finally:
                    self.assertTrue(worker.stop(timeout=5))

    def test_automatic_bundle_terminal_routes_do_not_reset_or_answer_twice(self):
        for stage in ('after_done', 'before_reset'):
            with self.subTest(stage=stage), contextlib.redirect_stderr(io.StringIO()):
                sess = MockSession(MockEngine(ntok=4), max_ctx=4096)
                if stage == 'before_reset':
                    sess.prefill = Mock(side_effect=EngineError('ordinary engine failure'))
                sess.reset, sess.abort = Mock(), Mock()
                worker = EngineWorker(sess, BytesTokenizer(), idle_s=None, flight=NS())
                worker._record, worker._verdict = Mock(return_value={}), Mock(return_value='synthetic')
                worker._auto_ok = Mock(return_value=True)
                worker._bundle = Mock(side_effect=InvalidKVEncoding('terminal automatic bundle'))
                notified = threading.Event()
                worker.on_broken = lambda why: notified.set()
                try:
                    j = worker.submit(job())
                    events = terminal(j)
                    self.assertTrue(notified.wait(5))
                    self.assertEqual(events[-1][0], 'done' if stage == 'after_done' else 'error')
                    self.assertTrue(j.events.empty(), 'no second terminal event after an answer')
                    self.assertEqual(worker.stats['engine_errors'], 1)
                    self.assertEqual(worker.stats['resets'], 0)
                    self.assertEqual(self.health(worker)[0], 503)
                    sess.reset.assert_not_called()
                    sess.abort.assert_not_called()
                finally:
                    self.assertTrue(worker.stop(timeout=5))

    def test_idle_and_bundle_terminal_routes_break_worker(self):
        for stage in ('idle', 'bundle'):
            with self.subTest(stage=stage), contextlib.redirect_stderr(io.StringIO()):
                sess = MockSession(MockEngine(ntok=4), max_ctx=4096)
                error = InvalidKVEncoding('terminal ' + stage)
                sess.idle = Mock(side_effect=error)
                sess.reset, sess.abort = Mock(), Mock()
                worker = EngineWorker(sess, BytesTokenizer(), idle_s=0.005 if stage == 'idle' else None, flight=NS())
                worker._bundle = Mock(side_effect=error)
                worker.recent['prior'] = dict(request_id='prior')
                hits, notified = [], threading.Event()
                worker.on_broken = lambda why: (hits.append(why), notified.set())
                try:
                    if stage == 'idle':
                        j = job()
                        j.cancel.set()
                        self.assertEqual(terminal(worker.submit(j))[-1][0], 'done')
                    else:
                        self.assertEqual(terminal(worker.submit(BundleJob('prior')))[-1][0:2], ('error', 'engine'))
                    self.assertTrue(notified.wait(5))
                    self.assertEqual(hits, [str(error)])
                    self.assertEqual(self.health(worker)[0], 503)
                    sess.reset.assert_not_called()
                    sess.abort.assert_not_called()
                    with self.assertRaises(EngineBroken):
                        worker.submit(BundleJob('prior'))
                finally:
                    self.assertTrue(worker.stop(timeout=5))


class StartupShutdown(unittest.TestCase):
    def setUp(self):
        self.cli = cli_functions()
        self.m = NS(devs=(0,), selfcheck=Mock(), max_ctx=128, qsa=False)
        self.pf = NS(chunk=16, crossover=4)
        self.build = Mock(return_value=(self.m, self.pf, None, None))
        self.cli['_arch'] = NS(ARCHS={'synthetic': NS(build=self.build)})
        self.pc = NS(n_slots=0, tier_mib=(1.0,), store=None, side_rows=16)
        self.cache = ModuleType('caminus.cache.prompt')
        self.cache.PromptCache, self.cache.DEFAULT_CAP_GB = Mock(return_value=self.pc), 1.0
        cache_package = ModuleType('caminus.cache')
        cache_package.prompt = self.cache
        self.k3 = ModuleType('caminus.serve.k3_session')
        self.k3.K3Session = Mock(return_value=NS(max_ctx=128, side_route=False))
        self.torch = ModuleType('torch')
        self.torch.cuda = NS(synchronize=Mock())
        self.patch = patch.dict(sys.modules, {'torch': self.torch, 'caminus.cache': cache_package,
                                             'caminus.cache.prompt': self.cache, 'caminus.serve.k3_session': self.k3})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.opt = dict(arch='synthetic', model='synthetic.gguf', no_disk_cache=True, cache_dir='unused', cache_cap_gb=None,
                        idle_park_s=0, no_side_route=False)

    def test_constructor_and_selfcheck_numerical_failures_exit75_before_cache(self):
        self.assertEqual(self.cli['EXIT_BROKEN'], 75)
        for cls in (InvalidKVEncoding, NonFiniteLogits):
            for stage in ('constructor', 'selfcheck'):
                with self.subTest(error=cls.__name__, stage=stage):
                    error = cls('synthetic terminal startup failure')
                    self.build.side_effect = error if stage == 'constructor' else None
                    self.m.selfcheck.side_effect = error if stage == 'selfcheck' else None
                    with self.assertRaises(SystemExit) as caught:
                        self.cli['build_engine'](self.opt, None, log=Mock())
                    self.assertEqual(caught.exception.code, 75)
                    self.cache.PromptCache.assert_not_called()
                    self.k3.K3Session.assert_not_called()

    def test_other_startup_exceptions_are_not_reclassified(self):
        for cls in (ValueError, RuntimeError):
            for stage in ('constructor', 'selfcheck'):
                with self.subTest(error=cls.__name__, stage=stage):
                    error = cls('ordinary startup input refusal')
                    self.build.side_effect = error if stage == 'constructor' else None
                    self.m.selfcheck.side_effect = error if stage == 'selfcheck' else None
                    with self.assertRaises(cls) as caught:
                        self.cli['build_engine'](self.opt, None, log=Mock())
                    self.assertIs(caught.exception, error)
                    self.cache.PromptCache.assert_not_called()

    def test_valid_startup_still_constructs_cache_and_session(self):
        sess, parts = self.cli['build_engine'](self.opt, None, log=Mock())
        self.assertIs(sess, self.k3.K3Session.return_value)
        self.assertIs(parts['m'], self.m)
        self.cache.PromptCache.assert_called_once()
        self.m.selfcheck.assert_called_once()
        self.torch.cuda.synchronize.assert_called_once_with(0)

    def test_broken_shutdown_does_not_park_but_closes_cache(self):
        for cls in (InvalidKVEncoding, NonFiniteLogits):
            with self.subTest(error=cls.__name__):
                sess = NS(shutdown_park=Mock(side_effect=AssertionError('broken state must not be parked')))
                srv = NS(drain=Mock(return_value=True), worker=NS(broken=str(cls('terminal numerical failure'))),
                         stop=Mock(), flight=None)
                pc, log = NS(close=Mock()), Mock()
                self.assertIsNone(self.cli['shutdown'](srv, sess, pc, log=log))
                sess.shutdown_park.assert_not_called()
                srv.stop.assert_called_once()
                pc.close.assert_called_once()
                self.assertTrue(any('no park' in c.args[0] for c in log.call_args_list))


if __name__ == '__main__':
    result = unittest.main(exit=False).result
    print('RESULT', 'PASS' if result.wasSuccessful() else 'FAIL')
    sys.exit(not result.wasSuccessful())
