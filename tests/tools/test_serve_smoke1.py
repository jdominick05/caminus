"""The one-card smoke's strict tool acceptance check (CPU, synthetic SSE, no sockets or model).

  python -I tests/tools/test_serve_smoke1.py

Only request() is replaced: the smoke's chat() and existing OpenAI SSE collector still assemble every reply.
"""
import contextlib
import copy
import io
import json
import os
import sys
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import serve_smoke1 as smoke  # noqa: E402


def call(id='weather-a', city='Lisbon', **kw):
    return dict(id=id, name='get_weather', arguments=json.dumps({'city': city}), **kw)


def stream(calls=(), content='', reasoning='', finish=None, *, status=200, role=True, done=True, error=None):
    events = []

    def emit(delta=None, finish=None):
        events.append({'data': json.dumps({'choices': [{'delta': delta or {}, 'finish_reason': finish}]})})

    if role:
        emit({'role': 'assistant'})
    if content:
        emit({'content': content})
    if reasoning:
        emit({'reasoning_content': reasoning})
    for i, c in enumerate(calls):
        # Both names and arguments arrive in fragments, through the real collector.
        name = c.get('name', '')
        emit({'tool_calls': [{'index': i, 'id': c.get('id'), 'type': 'function', 'function': {'name': name[:4]}}]})
        emit({'tool_calls': [{'index': i, 'function': {'name': name[4:]}}]})
        for j in range(0, len(c.get('arguments', '')), 3):
            emit({'tool_calls': [{'index': i, 'function': {'arguments': c['arguments'][j:j + 3]}}]})
    emit(finish=finish or ('tool_calls' if calls else 'stop'))
    if error is not None:
        events.append({'data': json.dumps({'error': error})})
    if done:
        events.append({'data': '[DONE]'})
    return dict(status=status, events=events, closed=False)


def answer(body):
    observations = [json.loads(m['content']) for m in body['messages'] if m['role'] == 'tool']
    return stream(content=' '.join(f'{a["city"]}: {a["temp_c"]} C, {a["sky"]} ({a["observation_code"]}).'
                                  for a in observations))


class ToolRoundTrip(unittest.TestCase):
    def run_round_trip(self, *replies):
        requests, checks, pending = [], [], iter(replies)

        def request(port, method, path, body):
            self.assertEqual((port, method, path), (1234, 'POST', '/v1/chat/completions'))
            requests.append(copy.deepcopy(body))
            reply = next(pending)  # Any unexpected extra request fails the test.
            return reply(body) if callable(reply) else reply

        log = io.StringIO()
        with patch.object(smoke.S, 'request', request), contextlib.redirect_stdout(log):
            stats = smoke.weather_round_trip(1234, 'chosen-model', 'synthetic-marker',
                                             check_fn=lambda ok, text: checks.append((ok, text)))
        return stats, requests, checks, log.getvalue()

    def assert_failed(self, run, count):
        stats, requests, checks, _ = run
        self.assertFalse(stats['answered'])
        self.assertEqual(len(requests), count)
        self.assertTrue(any(not ok for ok, _ in checks), checks)

    def test_natural_calls_replay_all_ids_and_results(self):
        calls = [call(id='exact-id-1'), call(id='exact-id-2', city='Porto')]
        stats, requests, checks, log = self.run_round_trip(
            stream(calls, content='I will check.', reasoning='Use current observations.'), answer)
        self.assertEqual(stats, dict(forced=0, calls=2, answered=True))
        self.assertTrue(all(ok for ok, _ in checks), checks)
        self.assertEqual(log, '')
        first, result = requests
        self.assertNotIn('tool_choice', first)
        self.assertEqual(result['tool_choice'], 'none')
        self.assertEqual(result['messages'][:2], first['messages'])
        assistant = result['messages'][2]
        self.assertEqual(assistant['content'], 'I will check.')
        self.assertEqual(assistant['reasoning_content'], 'Use current observations.')
        self.assertEqual(assistant['tool_calls'], [
            dict(id=c['id'], type='function', function=dict(name=c['name'], arguments=c['arguments'])) for c in calls])
        tools = result['messages'][3:]
        self.assertEqual([m['tool_call_id'] for m in tools], ['exact-id-1', 'exact-id-2'])
        observations = [json.loads(m['content']) for m in tools]
        self.assertEqual([a['city'] for a in observations], ['Lisbon', 'Porto'])
        codes = [a['observation_code'] for a in observations]
        self.assertEqual(len(set(codes)), 2)
        for code in codes:
            self.assertNotIn(code, json.dumps(first))
            self.assertNotIn(code, json.dumps(assistant))
        for request in requests:
            self.assertEqual(request['model'], 'chosen-model')
            self.assertTrue(request['stream'])
            self.assertEqual(request['tools'], smoke.TOOLS)
            self.assertEqual((request['temperature'], request['top_p'], request['max_tokens']), (0.7, 0.9, 512))
        self.assertNotIn('chat_template_kwargs', first)

    def test_forced_retry_is_reported_and_keeps_model_and_prompt(self):
        stats, requests, checks, log = self.run_round_trip(
            stream(content='It may be sunny.'), stream([call()]), answer)
        self.assertEqual(stats, dict(forced=1, calls=1, answered=True))
        self.assertTrue(all(ok for ok, _ in checks), checks)
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[1], dict(requests[0], tool_choice='required'))
        self.assertEqual({r['model'] for r in requests}, {'chosen-model'})
        self.assertEqual(requests[-1]['tool_choice'], 'none')
        self.assertNotIn('It may be sunny.', json.dumps(requests[-1]))
        self.assertEqual(log.count('retrying the same model with tool_choice required'), 1)

    def test_no_calls_cannot_pass_or_retry_more_than_once(self):
        for text in ('', 'No tool is needed.'):
            with self.subTest(text=text):
                run = self.run_round_trip(stream(content=text), stream(content=text))
                self.assert_failed(run, 2)
                self.assertEqual(run[0], dict(forced=1, calls=0, answered=False))

    def test_malformed_calls_fail_before_any_result_or_retry(self):
        malformed = [
            dict(call(), name='undeclared'), dict(call(), id=None), dict(call(), id=''), dict(call(), id=' \t'),
            dict(call(), id=12), dict(call(), arguments=''), dict(call(), arguments='{'),
            dict(call(), arguments='[]'), dict(call(), arguments='null'), dict(call(), arguments='42'),
            dict(call(), arguments='{"city":"Lisbon", "other":NaN}'),
            dict(call(), arguments='{"city":"Lisbon", "other":Infinity}'),
            dict(call(), arguments='[' * 1100 + ']' * 1100),
        ] + [dict(call(), arguments=json.dumps(a)) for a in ({}, {'city': None}, {'city': 4}, {'city': []},
                                                           {'city': {}}, {'city': ''}, {'city': ' \n\t'})]
        for c in malformed:
            with self.subTest(call=c):
                self.assert_failed(self.run_round_trip(stream([c])), 1)
        # Even the first valid call must not be replayed when another call is invalid or repeats its ID.
        for bad in (dict(call(id='b'), arguments='[]'), call()):
            with self.subTest(batch=bad):
                self.assert_failed(self.run_round_trip(stream([call(), bad])), 1)

    def test_call_shape_refusals_do_not_crash(self):
        for calls in ([None], ['not a call'], [dict(call(), arguments=None)], [dict(call(), arguments={})]):
            with self.subTest(calls=calls):
                self.assertIsNone(smoke._weather_args(calls))

    def test_malformed_required_reply_is_not_replayed(self):
        self.assert_failed(self.run_round_trip(stream(content='No call.'), stream([dict(call(), arguments='[]')])), 2)

    def test_call_turn_requires_completed_protocol_and_tool_finish(self):
        bad_replies = [stream([call()], status=500), stream([call()], role=False), stream([call()], done=False),
                       stream([call()], error={}), stream([call()], error={'message': 'generation failed'}),
                       stream([call()], finish='length'), stream([call()], finish='stop')]
        for reply in bad_replies:
            with self.subTest(reply=reply):
                self.assert_failed(self.run_round_trip(reply), 1)
        # Invalid natural streams must fail, rather than obtaining a pass through forcing.
        for reply in (stream(status=400), stream(role=False), stream(done=False), stream(error='bad stream')):
            with self.subTest(reply=reply):
                self.assert_failed(self.run_round_trip(reply), 1)
        self.assert_failed(self.run_round_trip(stream(), stream([call()], done=False)), 2)

    def test_result_requires_all_observations_and_usable_weather_text(self):
        calls = [call(), call(id='b', city='Porto')]
        bad_content = ['', ' \n', 'It is 19 C and clear.', '19 C: weather-observation-9999',
                       '19 C: weather-observation-00010 weather-observation-00020',
                       'weather-observation-0001 weather-observation-0002',
                       '20 C: weather-observation-0001 weather-observation-0002',
                       '19 C: weather-observation-0001']
        for content in bad_content:
            with self.subTest(content=content):
                self.assert_failed(self.run_round_trip(stream(calls), stream(content=content)), 2)

    def test_result_rejects_further_calls_errors_and_incomplete_streams(self):
        text = '19 C and clear: weather-observation-0001'
        bad_replies = [stream(content=text, status=500), stream(content=text, done=False), stream(content=text, role=False),
                       stream(content=text, error={}), stream(content=text, error={'message': 'bad result'}),
                       stream(content=text, finish='length'), stream([call(id='again')], content=text)]
        for reply in bad_replies:
            with self.subTest(reply=reply):
                self.assert_failed(self.run_round_trip(stream([call()]), reply), 2)


if __name__ == '__main__':
    unittest.main()
