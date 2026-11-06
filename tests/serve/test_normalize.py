"""Request normalization: OpenAI and Anthropic bodies -> template input. Synthetic requests only."""
import copy
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir
import common  # noqa: E402
import synth  # noqa: E402
from common import hf_render  # noqa: E402

from caminus.serve.normalize import (RequestError, anthropic_to_template, openai_chat_to_template,  # noqa: E402
                                    parse_tool_arguments, pin_billing_header)

HDR = 'x-anthropic-billing-header: cc_version=2.1.101.0d4; cc_entrypoint=cli; cch=12ab3;'


def oa_req(**kw):
    r = {'model': 'k3-coder', 'messages': [
        {'role': 'system', 'content': 'You are a coding agent.'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'List the files.'}]},
        {'role': 'assistant', 'content': None, 'reasoning_content': 'Use ls.',
         'tool_calls': [{'id': 'call_1', 'type': 'function',
                         'function': {'name': 'run_shell', 'arguments': '{"command": "ls -la", "timeout_s": 30}'}}]},
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'a.py\nb.py\n'},
    ], 'tools': synth.TOOLS}
    r.update(kw)
    return r


def an_req(**kw):
    r = {'model': 'k3-coder', 'max_tokens': 1000, 'system': [
        {'type': 'text', 'text': HDR}, {'type': 'text', 'text': 'You are a coding agent.', 'cache_control': {
            'type': 'ephemeral'}}],
        'thinking': {'type': 'enabled', 'budget_tokens': 4000},
        'tools': [{'name': f['function']['name'], 'description': f['function']['description'],
                   'input_schema': f['function']['parameters']} for f in synth.TOOLS],
        'messages': [
            {'role': 'user', 'content': 'List the files.'},
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'Use ls.', 'signature': ''},
                {'type': 'tool_use', 'id': 'toolu_1', 'name': 'run_shell', 'input': {'command': 'ls -la',
                                                                                    'timeout_s': 30}},
                {'type': 'tool_use', 'id': 'toolu_2', 'name': 'read_file', 'input': {'path': 'a.py'}}]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'toolu_1', 'content': 'a.py\nb.py\n'},
                {'type': 'tool_result', 'tool_use_id': 'toolu_2', 'content': [{'type': 'text', 'text': 'print(1)'}]},
                {'type': 'text', 'text': '<system-reminder>be brief</system-reminder>'}]},
        ]}
    r.update(kw)
    return r


class TestNormalize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = common.builder()

    def test_parse_tool_arguments(self):
        self.assertEqual(parse_tool_arguments('{"a": 1}'), {'a': 1})
        self.assertEqual(parse_tool_arguments(''), {})
        self.assertEqual(parse_tool_arguments('  '), {})
        self.assertEqual(parse_tool_arguments(None), {})
        self.assertEqual(parse_tool_arguments({'x': [1]}), {'x': [1]})
        for bad in ('{"a": ', '[1, 2]', '"s"', 3):
            with self.assertRaises(RequestError):
                parse_tool_arguments(bad)

    def test_openai_string_args_render_like_hf_objects(self):
        n = openai_chat_to_template(oa_req())
        call = n['messages'][2]['tool_calls'][0]
        self.assertEqual(call['function']['arguments'], {'command': 'ls -la', 'timeout_s': 30})
        self.assertEqual(n['messages'][1]['content'], 'List the files.')
        p = self.b.build(n['messages'], n['tools'], model=n['model'], client_kwargs=n['client_kwargs'])
        self.assertEqual(p.text, hf_render(n['messages'], n['tools'], True, **p.kwargs))
        self.assertIn('<parameter=command>\nls -la\n</parameter>\n<parameter=timeout_s>\n30\n</parameter>', p.text)
        # the raw request (string args) is refused by the template itself
        raw = copy.deepcopy(oa_req()['messages'])
        raw[2]['content'] = ''
        with self.assertRaises(Exception):
            hf_render(raw, synth.TOOLS)

    def test_openai_kwargs_and_roles(self):
        n = openai_chat_to_template(oa_req(reasoning_effort='low', chat_template_kwargs={
            'enable_thinking': False, 'preserve_thinking': True, 'other': 1}))
        self.assertEqual(n['client_kwargs'], {'reasoning_effort': 'low', 'enable_thinking': False,
                                              'preserve_thinking': True})
        n = openai_chat_to_template({'model': 'm', 'messages': [
            {'role': 'developer', 'content': 'dev'}, {'role': 'user', 'content': 'u'},
            {'role': 'system', 'content': 'late'}, {'role': 'assistant', 'content': '<think>\nR\n</think>\n\nC'},
            {'role': 'assistant', 'content': 'x', 'reasoning': 'r2'}]})
        self.assertEqual([m['role'] for m in n['messages']], ['system', 'user', 'user', 'assistant', 'assistant'])
        self.assertEqual(n['messages'][3]['reasoning_content'], '\nR\n')
        self.assertEqual(n['messages'][3]['content'], 'C')
        self.assertEqual(n['messages'][4]['reasoning_content'], 'r2')
        self.assertIsNone(n['tools'])

    def test_openai_errors(self):
        for bad in ({}, {'messages': []}, {'messages': [{'role': 'wizard', 'content': 'x'}]},
                    {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'x'}}]}]},
                    {'messages': [{'role': 'assistant', 'tool_calls': [{'function': {'name': 'f',
                                                                                     'arguments': '{bad'}}]}]},
                    {'messages': [{'role': 'user', 'content': 'x'}], 'tools': [{'type': 'function'}]}):
            with self.subTest(bad=json.dumps(bad)[:60]):
                with self.assertRaises(RequestError):
                    openai_chat_to_template(bad)

    def test_billing_header_pin(self):
        a = pin_billing_header(HDR + 'You are Claude Code.')
        b = pin_billing_header(HDR.replace('12ab3', '9be0c').replace('0d4', '7f3') + 'You are Claude Code.')
        self.assertEqual(a, b)
        self.assertEqual(a, 'x-anthropic-billing-header: cc_version=2.1.101.fff; cc_entrypoint=cli; cch=fffff;'
                            'You are Claude Code.')
        self.assertEqual(pin_billing_header('no header cch=12345;'), 'no header cch=12345;')
        self.assertEqual(pin_billing_header('x-anthropic-billing-header: cc_version=2.1.101; cch=1;'),
                         'x-anthropic-billing-header: cc_version=2.1.101; cch=f;')

    def test_anthropic(self):
        n = anthropic_to_template(an_req())
        m = n['messages']
        self.assertEqual([x['role'] for x in m], ['system', 'user', 'assistant', 'tool', 'tool', 'user'])
        self.assertTrue(m[0]['content'].startswith('x-anthropic-billing-header: cc_version=2.1.101.fff;'))
        self.assertTrue(m[0]['content'].endswith(';\nYou are a coding agent.'))
        self.assertEqual(m[2]['reasoning_content'], 'Use ls.')
        self.assertEqual([c['function']['name'] for c in m[2]['tool_calls']], ['run_shell', 'read_file'])
        self.assertEqual(m[4]['content'], 'print(1)')
        self.assertEqual(n['tools'][0]['function']['parameters'], synth.TOOLS[0]['function']['parameters'])
        self.assertEqual(n['client_kwargs'], {})
        p = self.b.build(m, n['tools'], model=n['model'], client_kwargs=n['client_kwargs'])
        self.assertEqual(p.text, hf_render(m, n['tools'], True, **p.kwargs))
        # a different cch / session stamp gives the same prompt ids
        r2 = an_req()
        r2['system'][0]['text'] = HDR.replace('12ab3', '00c1d').replace('0d4', 'b2a')
        n2 = anthropic_to_template(r2)
        self.assertEqual(self.b.build(n2['messages'], n2['tools']).ids, p.ids)

    def test_anthropic_thinking_policy(self):
        r = an_req()
        del r['thinking']
        self.assertEqual(anthropic_to_template(r)['client_kwargs'], {'enable_thinking': False})
        r['thinking'] = {'type': 'disabled'}
        self.assertEqual(anthropic_to_template(r)['client_kwargs'], {'enable_thinking': False})
        r['thinking'] = {'type': 'adaptive'}
        r['output_config'] = {'effort': 'high'}
        self.assertEqual(anthropic_to_template(r)['client_kwargs'], {'reasoning_effort': 'high'})

    def test_anthropic_errors(self):
        for bad in ({'messages': [{'role': 'system', 'content': 'x'}]},
                    {'messages': [{'role': 'user', 'content': [{'type': 'image', 'source': {}}]}]},
                    {'messages': [{'role': 'user', 'content': 'x'}], 'tools': [{'type': 'web_search_20250305',
                                                                                'name': 'web_search'}]},
                    {'system': [{'type': 'image'}], 'messages': [{'role': 'user', 'content': 'x'}]}):
            with self.subTest(bad=json.dumps(bad)[:60]):
                with self.assertRaises(RequestError):
                    anthropic_to_template(bad)

    def test_openai_and_anthropic_same_prompt(self):
        """the same conversation through either API renders the same prompt (system joined, tools wrapped)"""
        an = anthropic_to_template(an_req())
        oa = openai_chat_to_template({'model': 'k3-coder', 'tools': synth.TOOLS, 'messages': [
            {'role': 'system', 'content': an['messages'][0]['content']},
            {'role': 'user', 'content': 'List the files.'},
            {'role': 'assistant', 'content': '', 'reasoning_content': 'Use ls.', 'tool_calls': [
                {'id': 'a', 'type': 'function', 'function': {'name': 'run_shell',
                                                             'arguments': '{"command":"ls -la","timeout_s":30}'}},
                {'id': 'b', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path":"a.py"}'}}]},
            {'role': 'tool', 'content': 'a.py\nb.py\n'}, {'role': 'tool', 'content': 'print(1)'},
            {'role': 'user', 'content': '<system-reminder>be brief</system-reminder>'}]})
        pa = self.b.build(an['messages'], an['tools'], 'k3-coder', an['client_kwargs'])
        po = self.b.build(oa['messages'], oa['tools'], 'k3-coder', {'enable_thinking': True})
        self.assertEqual(pa.ids, po.ids)


if __name__ == '__main__':
    unittest.main()
