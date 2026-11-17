"""The served surface of the per-model sampling defaults (caminus.models.SAMPLING through caminus.serve.sampler), CPU only,
on the mock engine: /props reports the effective defaults (default_generation_settings.params) and the max_output hint,
/v1/models carries max_output in its meta, a request without sampling fields runs at the model's defaults and a request's
own fields win (the request's effective params in /slots). The resolution itself: tests/test_model_sampling.py.
Run:  HIP_VISIBLE_DEVICES= python -I tests/serve/run_all.py"""
import json
import unittest

import common  # noqa: F401  (sys.path)
import srvutil as U
from caminus import models
from caminus.serve.sampler import SamplingParams

SYS = {'role': 'system', 'content': 'You are a coding agent.'}


def get_json(srv, path):
    st, h, data = U.request(srv, 'GET', path)
    return st, json.loads(data)


class TestQ3MoEDefaults(unittest.TestCase):
    """a server given Qwen3-30B-A3B's catalog defaults (its GGUF has no general.sampling.* keys: from_recommended over {})"""

    @classmethod
    def setUpClass(cls):
        rec = models.SAMPLING['qwen3-30b-a3b']
        cls.srv = U.make_server(sampling=SamplingParams.from_recommended(rec, {}), max_output=rec.max_output)

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def chat(self, user, **body):
        msgs = [SYS, {'role': 'user', 'content': user}]
        U.script(self.srv.eng, msgs, {'role': 'assistant', 'content': 'done'}, None)
        b = {'model': 'k3-coder', 'messages': msgs, 'max_tokens': 64}
        b.update(body)
        st, h, j = U.post_json(self.srv, '/v1/chat/completions', b)
        self.assertEqual(st, 200, j)
        st, sl = get_json(self.srv, '/slots')
        return sl[0]['params']

    def test_props_reports_effective_defaults(self):
        st, p = get_json(self.srv, '/props')
        self.assertEqual(st, 200)
        g = p['default_generation_settings']
        q = g['params']
        self.assertEqual((q['temperature'], q['top_p'], q['top_k'], q['min_p'], q['presence_penalty'], q['repeat_penalty']),
                         (0.6, 0.95, 20, 0.0, 1.5, 1.0))
        self.assertEqual(g['max_output'], 32768)
        self.assertEqual(q['n_predict'], -1, 'the hint does not set n_predict (llm-top reads it)')

    def test_models_meta_max_output(self):
        st, j = get_json(self.srv, '/v1/models')
        self.assertEqual(st, 200)
        self.assertTrue(j['data'])
        self.assertTrue(all(m['meta']['max_output'] == 32768 for m in j['data']))

    def test_request_without_fields_runs_at_defaults(self):
        q = self.chat('Defaults: say done.')
        self.assertEqual((q['temperature'], q['top_p'], q['top_k'], q['presence_penalty']), (0.6, 0.95, 20, 1.5))

    def test_request_fields_win(self):
        q = self.chat('Override: say done.', temperature=0.2)
        self.assertEqual((q['temperature'], q['presence_penalty']), (0.2, 1.5))
        q = self.chat('Override 2: say done.', temperature=0, presence_penalty=0.0, top_k=5)
        self.assertEqual((q['temperature'], q['presence_penalty'], q['top_k']), (0.0, 0.0, 5))


class TestQwen38Defaults(unittest.TestCase):
    """a server given Swift 1.5's catalog defaults over its GGUF header's keys: what the header gave before"""

    def test_props(self):
        rec = models.SAMPLING['swift-1.5']
        hdr = {'general.sampling.temp': 1.0, 'general.sampling.top_k': 20, 'general.sampling.top_p': 0.949999988079071}
        srv = U.make_server(sampling=SamplingParams.from_recommended(rec, hdr), max_output=rec.max_output)
        try:
            st, p = get_json(srv, '/props')
            g = p['default_generation_settings']
            q = g['params']
            self.assertEqual((q['temperature'], q['top_p'], q['top_k'], q['min_p'], q['presence_penalty'], q['repeat_penalty']),
                             (1.0, 0.95, 20, 0.0, 0.0, 1.0))
            self.assertEqual(g['max_output'], 393216)
        finally:
            srv.stop()


class TestNoRecommendation(unittest.TestCase):
    """a server built without catalog defaults (a file outside the catalog; the other serve tests): max_output null"""

    def test_props_null_max_output(self):
        srv = U.make_server()
        try:
            st, p = get_json(srv, '/props')
            self.assertIsNone(p['default_generation_settings']['max_output'])
            st, j = get_json(srv, '/v1/models')
            self.assertTrue(all(m['meta']['max_output'] is None for m in j['data']))
        finally:
            srv.stop()


if __name__ == '__main__':
    unittest.main()
