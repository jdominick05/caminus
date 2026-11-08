"""The engine session contract (caminus.serve.engine: EngineSession's declared members, contract_problems), CPU only:
  * MockSession and every session subclass the serve tests use conform (contract_problems == []); a session missing a
    call's parameter, or with no way to sample, is reported;
  * verify() returns nothing the server reads; the base class's host sampling goes through pass_rows();
  * the optional members' defaults (idle / shutdown_park / request_marks / bundle_state / export_bundle) on a session
    without that state;
  * the monitoring surface on MockSession: every key path of /props, /slots, /caminus/status and every /metrics series
    in surface_keys.json (recorded before the session contract was declared, 11-07) is still served (a new one is
    reported, not refused: the pin catches a field dropped by a change to the session or the stats, never an addition).
K3Session's own conformance (it needs torch) is checked in tests/cache/test_prompt_cache.py cpu.
Run:  HIP_VISIBLE_DEVICES= python -I tests/serve/run_all.py"""
import json
import os
import re
import tempfile
import unittest

import common  # noqa: F401  (sys.path)
import srvutil as U
from caminus.serve.engine import EngineSession, MockSession, contract_problems
from caminus.serve.hw import HwReader, write_fixture
from caminus.serve.mock_engine import MockDrafter, ScriptedEngine, SynthProfile

HERE = os.path.dirname(os.path.abspath(__file__))


def _subclasses():
    """the MockSession subclasses the serve tests define (their modules import nothing heavy)"""
    import test_flight, test_ops, test_route, test_stats
    return [test_flight._StateSession, test_ops._ParkSession, test_route._RouteSession, test_stats._PlainSession]


class TestContract(unittest.TestCase):
    def test_mock_and_test_sessions_conform(self):
        for cls in [MockSession] + _subclasses():
            s = cls(ScriptedEngine(ntok=4), max_ctx=4096, eos_ids=(248046, 248044))
            self.assertEqual(contract_problems(s), [], cls.__name__)
            self.assertEqual(contract_problems(cls), [], cls.__name__ + ' (the class)')

    def test_problems_reported(self):
        class OldPrefill(MockSession):
            def prefill(self, ids, cancel=None, marks=None):       # the signature before max_new reached every session
                return super().prefill(ids, cancel, marks)

        class NoSampling(EngineSession):
            ntok, max_ctx, n_vocab, eos_ids = 4, 4096, 1000, (1,)
            tokens = ()
        self.assertIn("prefill: no parameter 'max_new'", contract_problems(OldPrefill(ScriptedEngine(ntok=4))))
        p = contract_problems(NoSampling())
        self.assertTrue(any('neither sample_pass' in x for x in p), p)
        self.assertEqual(contract_problems(object()).count('not an EngineSession'), 1)

    def test_verify_and_pass_rows(self):
        eng = ScriptedEngine(ntok=4)
        s = MockSession(eng, max_ctx=4096)
        s.prefill([248044, 5, 6, 7])
        self.assertIsNone(s.verify([8, 9]))                     # the server never reads verify's logits
        s.accept(2)
        rows = s.pass_rows([10])                                # host sampling's pass: logits [1, V]
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), s.n_vocab)
        s.accept(1)
        self.assertEqual(list(s.tokens), [248044, 5, 6, 7, 8, 9, 10][:len(s.tokens)])

    def test_defaults_of_a_stateless_session(self):
        s = MockSession(ScriptedEngine(ntok=4), max_ctx=4096)
        self.assertIsNone(s.idle_s)
        self.assertIsNone(s.idle())
        self.assertIsNone(s.shutdown_park())
        self.assertEqual((s.request_marks(), s.bundle_state([1, 2])), ({}, {}))
        ref = s.export_bundle('/nonexistent', dict(source='cold'), None)['cache_ref']
        self.assertEqual((ref['source'], ref['state']), ('cold', None))
        self.assertTrue((s.drafts_real, s.wants_marks, s.drafter) == (True, False, None))


def _paths(o, p=''):
    out = set()
    if isinstance(o, dict):
        for k, v in o.items():
            out.add(f'{p}/{k}:{type(v).__name__}')
            if not (isinstance(v, dict) and all(re.fullmatch(r'\d+|[a-z_]+=.*', str(x)) for x in v) and len(v) > 0 and p.count('/') > 6):
                out |= _paths(v, f'{p}/{k}')
    elif isinstance(o, list) and o:
        out |= _paths(o[0], p + '[]')
    return out


class TestSurfacePin(unittest.TestCase):
    """the server's monitoring surface on MockSession after one chat request (the rig of tests/serve/test_metrics.py's
    TestSurface: a drafter, a fixture sysfs tree, the synthetic engine numbers, an 8-slot disk tier)"""

    def test_surface_keeps_every_pinned_field(self):
        with open(os.path.join(HERE, 'surface_keys.json')) as f:
            pin = json.load(f)
        eng = ScriptedEngine(ntok=4)
        srv = U.make_server(eng, drafter=MockDrafter(eng, 0.85), hw=HwReader(root=write_fixture(tempfile.mkdtemp(prefix='k3-hw-'))),
                            hw_interval=0, session_kw={'synth': SynthProfile(sleep=False), 'disk_slots': 8})
        try:
            msgs = [{'role': 'system', 'content': 'You are a coding agent.'}, {'role': 'user', 'content': 'hello'}]
            U.script(eng, msgs, {'role': 'assistant', 'content': 'hi there'}, None)
            st, _, j = U.post_json(srv, '/v1/chat/completions', {'model': 'k3-coder', 'messages': msgs, 'temperature': 0})
            self.assertEqual(st, 200, j)
            got = {}
            for path in ('/props', '/caminus/status', '/slots'):
                st, _, data = U.request(srv, 'GET', path)
                got[path] = _paths(json.loads(data))
            st, _, data = U.request(srv, 'GET', '/metrics')
            got['/metrics'] = {re.split(r'[{ ]', ln)[0] for ln in data.decode().split('\n') if ln and not ln.startswith('#')}
        finally:
            srv.stop()
        for path, keys in pin.items():
            missing = sorted(set(keys) - got[path])
            self.assertEqual(missing, [], f'{path}: pinned fields no longer served')
            new = sorted(got[path] - set(keys))
            if new:
                print(f'  (surface pin: {path} serves {len(new)} field(s) not in surface_keys.json: {new[:5]})')


if __name__ == '__main__':
    unittest.main()
