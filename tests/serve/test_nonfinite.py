"""Non-finite logits fail loudly (caminus.errors.NonFiniteLogits): the prompt row's check in run_generation, the sampler's
empty distribution, and the server that stops serving (503 on /health and on new requests) instead of streaming the
argmax of NaN. A served load once produced NaN from block ~52 on for every prompt; the greedy reply was token 0 x64."""
import unittest

import numpy as np

import common  # noqa: F401  (sys.path)
import srvutil as U
from caminus.errors import NonFiniteLogits
from caminus.serve import sampler as S
from caminus.serve.engine import MockSession
from caminus.serve.generate import run_generation
from caminus.serve.mock_engine import ScriptedEngine


class NaNEngine(ScriptedEngine):
    """every logits row all NaN (a model whose weights went bad at load)"""
    def logits(self, prefix):
        return np.full(self.V, np.nan, dtype=np.float32)


class TestNonFinite(unittest.TestCase):
    def test_prompt_row_greedy(self):
        sess = MockSession(NaNEngine(ntok=4), max_ctx=4096)
        prm = S.SamplingParams(temperature=0.0)
        with self.assertRaises(NonFiniteLogits):
            run_generation(sess, [1, 2, 3, 4], S.Sampler(prm), 16, lambda ids: None)

    def test_prompt_row_sampled(self):
        sess = MockSession(NaNEngine(ntok=4), max_ctx=4096)
        prm = S.SamplingParams(temperature=0.7, seed=1)
        with self.assertRaises(NonFiniteLogits):
            run_generation(sess, [1, 2, 3, 4], S.Sampler(prm), 16, lambda ids: None)

    def test_draw_empty(self):
        w = np.array([np.nan, np.nan], dtype=np.float32)
        c = S.Candidates(np.array([3, 5]), w, w, w)
        with self.assertRaises(NonFiniteLogits):
            S.draw(c, 0.5)

    def test_server_stops_serving(self):
        srv = U.make_server(engine=NaNEngine(ntok=4))
        hits = []
        srv.worker.on_broken = hits.append            # (caminus serve: stop the loop, exit 75)
        try:
            st, _, body = U.post_json(srv, '/v1/completions', {'prompt': 'hello', 'max_tokens': 8, 'temperature': 0})
            self.assertNotEqual(st, 200)
            self.assertIn('broken', str(body))
            self.assertEqual(len(hits), 1)                   # the broken callback ran once, with the reason
            self.assertIn('non-finite', hits[0])
            st2, _, _ = U.request(srv, 'GET', '/health')
            self.assertEqual(st2, 503)                       # unhealthy with the reason, until a restart
            st3, _, body3 = U.post_json(srv, '/v1/completions', {'prompt': 'again', 'max_tokens': 8, 'temperature': 0})
            self.assertEqual(st3, 503)                       # refused before reaching the engine
            self.assertIn('broken', str(body3))
        finally:
            srv.stop()


if __name__ == '__main__':
    unittest.main()
