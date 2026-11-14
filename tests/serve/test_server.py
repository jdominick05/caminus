"""S4 gates (T0, CPU only, MockSession): the HTTP server through the anthropic SDK (streaming and not, tools, thinking)
and raw-HTTP OpenAI requests (SSE conformance incl. tool_calls deltas), scripted multi-turn agent loops that show
append-only prompts and count prefix reuse exactly (turn n+1 reuses prompt n + reply n - 1 tokens), stop strings,
max_tokens cuts, tool_choice forcing, /v1/completions, malformed requests, client disconnects, engine faults, the FIFO
queue and Anthropic pings. Everything is synthesized (tests/serve/synth.py); no session data."""
import json
import os
import socket
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import srvutil as U  # noqa: E402
import synth  # noqa: E402
from caminus.serve import deltas as D  # noqa: E402
from caminus.serve.generate import run_generation  # noqa: E402
from caminus.serve.engine import MockSession  # noqa: E402
from caminus.serve.mock_engine import MockDrafter, MockEngine, ScriptedEngine  # noqa: E402
from caminus.serve.sampler import Sampler, SamplingParams  # noqa: E402
from caminus.serve.server import POLL_S  # noqa: E402

import anthropic  # noqa: E402

ANTH_TOOLS = [{'name': t['function']['name'], 'description': t['function']['description'],
               'input_schema': t['function']['parameters']} for t in synth.TOOLS]


def unique_conversation(seed, **kw):
    """synth.agent_conversation with the seed written into the first user message, so two conversations never share
    a prompt (the scripted engine refuses two replies for one prompt)"""
    tools, steps = synth.agent_conversation(seed, **kw)
    first = next(m for m in steps[0][0] if m['role'] == 'user')     # shared by reference with every later step
    first['content'] = f'[task {seed}] ' + first['content']
    return tools, steps


def sdk(srv):
    return anthropic.Anthropic(base_url=f'http://127.0.0.1:{srv.port}', api_key='test-key', max_retries=0,
                               timeout=120)


def openai_turn(srv, body, stream):
    """-> (assistant message as a client keeps it, usage, finish_reason, timings)"""
    if not stream:
        st, h, j = U.post_json(srv, '/v1/chat/completions', body)
        assert st == 200, (st, j)
        ch = j['choices'][0]
        return ch['message'], j['usage'], ch['finish_reason'], j['timings']
    st, h, evs = U.sse_post(srv, '/v1/chat/completions', {**body, 'stream': True})
    assert st == 200, (st, evs)
    chunks = [json.loads(d) for _, d in evs if d != '[DONE]']
    acc = D.apply_openai_deltas([c['choices'][0]['delta'] for c in chunks if c['choices']])
    fin = [c['choices'][0]['finish_reason'] for c in chunks if c['choices'] and c['choices'][0]['finish_reason']]
    usage = [c['usage'] for c in chunks if c.get('usage')][-1]
    timings = [c['timings'] for c in chunks if c.get('timings')][-1]
    msg = {'role': 'assistant', 'content': acc['content'] or None, 'reasoning_content': acc['reasoning_content']}
    if acc['tool_calls']:
        msg['tool_calls'] = [{'id': c['id'], 'type': 'function',
                              'function': {'name': c['name'], 'arguments': c['arguments']}} for c in acc['tool_calls']]
    return msg, usage, fin[0], timings


def openai_got(msg):
    return {'reasoning_content': msg.get('reasoning_content'), 'content': msg.get('content'),
            'tool_calls': [(c['function']['name'], json.loads(c['function']['arguments']))
                           for c in msg.get('tool_calls') or ()]}


def anthropic_got(blocks):
    return {'reasoning_content': ''.join(b['thinking'] for b in blocks if b['type'] == 'thinking'),
            'content': ''.join(b['text'] for b in blocks if b['type'] == 'text'),
            'tool_calls': [(b['name'], b['input']) for b in blocks if b['type'] == 'tool_use']}


# ==================================================================================================== agent loops
class TestAgentLoops(unittest.TestCase):
    """multi-turn agent loops through the MockEngine: the prompt of turn n+1 extends prompt n + the reply exactly,
    so the session reuses len(prompt n) + len(reply n) - 1 tokens (all but the unfed EOS)"""

    @classmethod
    def setUpClass(cls):
        eng = ScriptedEngine(ntok=4)
        cls.srv = U.make_server(eng, drafter=MockDrafter(eng, quality=0.85))

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def _expect_cached(self, k, scripts, usage_cached, where):
        if k:
            p, r = scripts[k - 1]
            self.assertEqual(usage_cached, len(p) + len(r) - 1, where)

    def test_openai_loop(self):
        srv = self.srv
        turns = total_prompt = total_cached = 0
        for ci, (seed, temp) in enumerate(((11, 0.0), (12, 1.0), (13, 0.6))):
            tools, steps = unique_conversation(seed, rounds=2, user_turns=2)
            scripts = [U.script(srv.eng, m, r, tools) for m, r in steps]
            wire, cursor, last_ids = [], 0, []
            for k, (msgs, reply) in enumerate(steps):
                tk = 0
                for m in msgs[cursor:]:
                    w = U.openai_wire(m)
                    if m['role'] == 'tool':
                        w['tool_call_id'] = last_ids[tk] if tk < len(last_ids) else 'call_x'
                        tk += 1
                    wire.append(w)
                body = {'model': 'k3-coder', 'messages': wire, 'tools': tools, 'temperature': temp, 'seed': 100 + k,
                        'stream_options': {'include_usage': True}}
                msg, usage, fin, tim = openai_turn(srv, body, stream=(k + ci) % 2 == 1)
                where = f'conv {seed} turn {k}'
                U.same_reply(self, openai_got(msg), reply, where)
                self.assertEqual(fin, 'tool_calls' if reply.get('tool_calls') else 'stop', where)
                self.assertEqual(usage['prompt_tokens'], len(scripts[k][0]), where)
                self.assertEqual(usage['completion_tokens'], len(scripts[k][1]), where)
                self._expect_cached(k, scripts, usage['prompt_tokens_details']['cached_tokens'], where)
                self.assertEqual(tim['cache_n'], usage['prompt_tokens_details']['cached_tokens'])
                total_prompt += usage['prompt_tokens']
                total_cached += usage['prompt_tokens_details']['cached_tokens'] if k else 0
                wire.append(msg)
                last_ids = [c['id'] for c in msg.get('tool_calls') or ()]
                cursor = len(msgs) + 1
                turns += 1
        print(f'\n[loop/openai] {turns} turns (stream and not, T 0/1.0/0.6, drafting): replies exact, '
              f'cached = prompt n + reply n - 1 on every later turn; {total_cached}/{total_prompt} prompt tokens reused')

    def test_anthropic_sdk_loop(self):
        srv = self.srv
        cl = sdk(srv)
        turns = total_prompt = total_cached = 0
        modes = {'create': 0, 'create_stream': 0, 'stream_helper': 0}
        for ci, seed in enumerate((21, 22)):
            tools, steps = unique_conversation(seed, rounds=2, user_turns=2)
            scripts = [U.script(srv.eng, m, r, tools) for m, r in steps]
            system = steps[0][0][0]['content'] if steps[0][0][0]['role'] == 'system' else None
            wire, cursor, last_ids = [], (1 if system is not None else 0), []
            for k, (msgs, reply) in enumerate(steps):
                new = msgs[cursor:]
                if len(new) == 1 and new[0]['role'] == 'user':
                    wire.append({'role': 'user', 'content': new[0]['content']})
                else:
                    wire.append({'role': 'user', 'content': U.anthropic_user_blocks(new, last_ids)})
                kw = dict(model='k3-coder', max_tokens=4096, messages=wire, tools=ANTH_TOOLS,
                          thinking={'type': 'enabled', 'budget_tokens': 2048}, extra_body={'temperature': 1.0})
                if system is not None:
                    kw['system'] = system
                mode = ('create', 'create_stream', 'stream_helper')[(k + ci) % 3]
                modes[mode] += 1
                where = f'conv {seed} turn {k} ({mode})'
                if mode == 'create':
                    m = cl.messages.create(**kw)
                    blocks = [b.model_dump(exclude_none=True) for b in m.content]
                    usage, stop_reason = m.usage, m.stop_reason
                elif mode == 'stream_helper':
                    with cl.messages.stream(**kw) as s:
                        m = s.get_final_message()
                    blocks = [b.model_dump(exclude_none=True) for b in m.content]
                    usage, stop_reason = m.usage, m.stop_reason
                else:
                    blocks, usage, stop_reason = self._raw_anthropic_stream(cl, kw, where)
                U.same_reply(self, anthropic_got(blocks), reply, where)
                self.assertEqual(stop_reason, 'tool_use' if reply.get('tool_calls') else 'end_turn', where)
                prompt_n = usage.input_tokens + usage.cache_read_input_tokens
                self.assertEqual(prompt_n, len(scripts[k][0]), where)
                self.assertEqual(usage.output_tokens, len(scripts[k][1]), where)
                self._expect_cached(k, scripts, usage.cache_read_input_tokens, where)
                total_prompt += prompt_n
                total_cached += usage.cache_read_input_tokens if k else 0
                wire.append({'role': 'assistant', 'content': blocks})
                last_ids = [b['id'] for b in blocks if b['type'] == 'tool_use']
                cursor = len(msgs) + 1
                turns += 1
        print(f'\n[loop/anthropic SDK] {turns} turns {modes}: replies exact, thinking + tool_use round trip, '
              f'cached exact; {total_cached}/{total_prompt} prompt tokens reused')

    def _raw_anthropic_stream(self, cl, kw, where):
        """create(stream=True): check the event order and rebuild the content blocks from the raw events"""
        types, blocks, open_idx = [], [], None
        usage = stop_reason = None
        partial = {}
        for ev in cl.messages.create(stream=True, **kw):
            types.append(ev.type)
            if ev.type == 'message_start':
                self.assertEqual(ev.message.content, [])
                usage = ev.message.usage
            elif ev.type == 'content_block_start':
                self.assertIsNone(open_idx, where)
                self.assertEqual(ev.index, len(blocks), where)
                open_idx = ev.index
                blocks.append(ev.content_block.model_dump(exclude_none=True))
            elif ev.type == 'content_block_delta':
                self.assertEqual(ev.index, open_idx, where)
                b, d = blocks[ev.index], ev.delta
                if d.type == 'thinking_delta':
                    b['thinking'] += d.thinking
                elif d.type == 'text_delta':
                    b['text'] += d.text
                elif d.type == 'input_json_delta':
                    partial[ev.index] = partial.get(ev.index, '') + d.partial_json
                elif d.type == 'signature_delta':
                    self.assertEqual(b['type'], 'thinking')
            elif ev.type == 'content_block_stop':
                self.assertEqual(ev.index, open_idx, where)
                if ev.index in partial:
                    blocks[ev.index]['input'] = json.loads(partial[ev.index])
                open_idx = None
            elif ev.type == 'message_delta':
                stop_reason = ev.delta.stop_reason
                out_usage = ev.usage
        self.assertEqual(types[0], 'message_start', where)
        self.assertEqual(types[-2:], ['message_delta', 'message_stop'], where)
        self.assertIsNone(open_idx)
        usage.output_tokens = out_usage.output_tokens
        return blocks, usage, stop_reason


# ==================================================================================================== OpenAI SSE
class TestOpenAIStreamConformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = U.make_server(ScriptedEngine(ntok=1))
        cls.inc = U.make_server(ScriptedEngine(ntok=1), incremental_tool_args=True)
        cls.cases = []
        for seed in range(40):
            tools, steps = unique_conversation(300 + seed, rounds=3, user_turns=1)
            for msgs, reply in steps:
                if len(reply.get('tool_calls') or ()) >= 2:
                    cls.cases.append((tools, msgs, reply))
            if len(cls.cases) >= 4:
                break
        for srv in (cls.srv, cls.inc):
            for tools, msgs, reply in cls.cases:
                U.script(srv.eng, msgs, reply, tools)

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()
        cls.inc.stop()

    def _check(self, srv, tools, msgs, reply, include_usage):
        body = {'model': 'k3-coder', 'messages': msgs_wire(msgs), 'tools': tools, 'temperature': 0, 'stream': True}
        if include_usage:
            body['stream_options'] = {'include_usage': True}
        st, h, data = U.request(srv, 'POST', '/v1/chat/completions', body)
        self.assertEqual(st, 200)
        self.assertTrue(h['content-type'].startswith('text/event-stream'))
        evs, comments = U.parse_sse(data)
        self.assertTrue(all(e is None for e, _ in evs), 'OpenAI SSE has no event: lines')
        self.assertEqual(evs[-1][1], '[DONE]')
        self.assertEqual(sum(1 for _, d in evs if d == '[DONE]'), 1)
        chunks = [json.loads(d) for _, d in evs[:-1]]
        ids = {c['id'] for c in chunks}
        self.assertEqual(len(ids), 1)
        self.assertTrue(next(iter(ids)).startswith('chatcmpl-'))
        self.assertEqual({c['object'] for c in chunks}, {'chat.completion.chunk'})
        self.assertEqual(len({c['created'] for c in chunks}), 1)
        self.assertEqual({c['model'] for c in chunks}, {'k3-coder'})
        choice_chunks = [c for c in chunks if c['choices']]
        for c in choice_chunks:
            self.assertEqual(len(c['choices']), 1)
            self.assertEqual(c['choices'][0]['index'], 0)
        self.assertEqual(choice_chunks[0]['choices'][0]['delta'], {'role': 'assistant', 'content': ''})
        fins = [(i, c['choices'][0]['finish_reason']) for i, c in enumerate(choice_chunks)
                if c['choices'][0]['finish_reason'] is not None]
        self.assertEqual(len(fins), 1)
        self.assertEqual(fins[0], (len(choice_chunks) - 1, 'tool_calls'))
        self.assertEqual(choice_chunks[-1]['choices'][0]['delta'], {})
        if include_usage:
            self.assertEqual(chunks[-1]['choices'], [])
            u = chunks[-1]['usage']
            self.assertEqual(u['total_tokens'], u['prompt_tokens'] + u['completion_tokens'])
            self.assertEqual(sum(1 for c in chunks if c.get('usage')), 1)
        else:
            self.assertIn('usage', choice_chunks[-1])
        # deltas: reasoning first, then content / tool calls; tool call deltas well-formed
        seen, pieces, phase = {}, {}, 0
        for c in choice_chunks[1:-1]:
            d = c['choices'][0]['delta']
            self.assertEqual(len(d), 1, d)
            k = next(iter(d))
            if k == 'reasoning_content':
                self.assertEqual(phase, 0, 'reasoning after content / calls')
            else:
                phase = 1
            if k == 'tool_calls':
                self.assertEqual(len(d['tool_calls']), 1)
                tc = d['tool_calls'][0]
                i = tc['index']
                if i not in seen:
                    self.assertEqual(i, len(seen), 'tool call indexes appear in order')
                    self.assertTrue(tc['id'].startswith('call_'))
                    self.assertEqual(tc['type'], 'function')
                    self.assertTrue(tc['function']['name'])
                    self.assertEqual(tc['function']['arguments'], '')
                    seen[i] = tc['function']['name']
                    pieces[i] = []
                else:
                    self.assertEqual(set(tc), {'index', 'function'})
                    self.assertEqual(set(tc['function']), {'arguments'})
                    pieces[i].append(tc['function']['arguments'])
        want = [(c['function']['name'], c['function']['arguments']) for c in reply['tool_calls']]
        got = [(seen[i], json.loads(''.join(pieces[i]))) for i in sorted(seen)]
        self.assertEqual(got, want)
        # the non-streaming response carries the same calls, arguments byte-equal to the concatenated deltas
        body.pop('stream')
        body.pop('stream_options', None)
        st, h, j = U.post_json(srv, '/v1/chat/completions', body)
        calls = j['choices'][0]['message']['tool_calls']
        self.assertEqual([(c['function']['name'], c['function']['arguments']) for c in calls],
                         [(seen[i], ''.join(pieces[i])) for i in sorted(seen)])
        self.assertEqual(j['choices'][0]['finish_reason'], 'tool_calls')
        return max(len(p) for p in pieces.values())

    def test_conformance_whole_calls(self):
        n = 0
        for tools, msgs, reply in self.cases:
            for iu in (False, True):
                self.assertEqual(self._check(self.srv, tools, msgs, reply, iu), 1, 'whole-call mode: one piece')
                n += 1
        print(f'\n[sse/openai] {n} streams with 2-3 parallel calls: framing, ids, role, one finish_reason, '
              f'usage chunk, [DONE], tool_calls deltas == non-stream arguments')

    def test_conformance_incremental(self):
        most = 0
        for tools, msgs, reply in self.cases:
            most = max(most, self._check(self.inc, tools, msgs, reply, True))
        self.assertGreater(most, 1, 'incremental mode streams argument pieces')
        print(f'\n[sse/openai incremental] up to {most} argument pieces per call; same conformance')


def msgs_wire(msgs):
    out = []
    for m in msgs:
        if m['role'] == 'assistant':
            w = {'role': 'assistant', 'content': m.get('content') or None,
                 'reasoning_content': m.get('reasoning_content')}
            if m.get('tool_calls'):
                w['tool_calls'] = [{'id': c['id'], 'type': 'function',
                                    'function': {'name': c['function']['name'],
                                                 'arguments': json.dumps(c['function']['arguments'],
                                                                         ensure_ascii=False)}}
                                   for c in m['tool_calls']]
            out.append(w)
        elif m['role'] == 'tool':
            out.append({'role': 'tool', 'tool_call_id': 'call_x', 'content': m['content']})
        else:
            out.append({'role': m['role'], 'content': m['content']})
    return out


# ==================================================================================================== stops, cuts, forcing
SYS = {'role': 'system', 'content': 'You are a coding agent.'}


class TestStopsAndForcing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        eng = ScriptedEngine(ntok=4)
        cls.srv = U.make_server(eng, drafter=MockDrafter(eng, quality=1.0))
        cls.tok = common.tokenizer()

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def test_stop_strings_and_checkpoint_reuse(self):
        srv = self.srv
        msgs = [SYS, {'role': 'user', 'content': 'Is it fixed? Answer with a status line.'}]
        reply = {'role': 'assistant', 'reasoning_content': 'Check the test output first.',
                 'content': 'Status: the test passes now. END_STATUS trailing text that must not appear.'}
        p, r = U.script(srv.eng, msgs, reply, None)
        # OpenAI, stream and not
        for stream in (False, True):
            body = {'model': 'k3-coder', 'messages': msgs, 'stop': ['END_STATUS', 'never'], 'temperature': 0}
            msg, usage, fin, tim = openai_turn(srv, body, stream)
            self.assertEqual(fin, 'stop')
            self.assertEqual(msg['content'], 'Status: the test passes now. ')
            self.assertEqual(msg['reasoning_content'].strip(), reply['reasoning_content'])
            self.assertLess(usage['completion_tokens'], len(r))
        # Anthropic: stop_reason stop_sequence + the matched string
        m = sdk(srv).messages.create(model='k3-coder', max_tokens=300, system=SYS['content'],
                                     messages=[{'role': 'user', 'content': msgs[1]['content']}],
                                     thinking={'type': 'enabled', 'budget_tokens': 1024},
                                     stop_sequences=['passes now', 'END_STATUS'])
        self.assertEqual((m.stop_reason, m.stop_sequence), ('stop_sequence', 'passes now'))
        self.assertEqual(m.content[-1].text, 'Status: the test ')
        # next turn after a stop: the client sends the cut reply; the live state is not a prefix any more, the
        # prompt-end checkpoint is
        nxt = msgs + [{'role': 'assistant', 'content': 'Status: the test ',
                       'reasoning_content': reply['reasoning_content']},
                      {'role': 'user', 'content': 'Go on.'}]
        st, h, j = U.post_json(srv, '/v1/chat/completions', {'messages': nxt, 'max_tokens': 2, 'temperature': 0})
        self.assertEqual(st, 200)
        self.assertEqual(j['usage']['prompt_tokens_details']['cached_tokens'], len(p))
        self.assertEqual(j['timings']['restore'], 'checkpoint')
        # a stop inside the reasoning ends the generation there too (matched on the raw text)
        st, h, j = U.post_json(srv, '/v1/chat/completions', {'messages': msgs, 'stop': 'output', 'temperature': 0})
        self.assertEqual(j['choices'][0]['message']['reasoning_content'].strip(), 'Check the test')
        rows = U.wait_rows(srv, 1)
        print(f'\n[stop] OpenAI finish stop / Anthropic stop_sequence; after a stop the next turn restores the '
              f'prompt-end checkpoint ({len(p)} tokens); stop_in_draft rows: '
              f'{sum(1 for x in rows if x.get("stop_in_draft"))}')

    def test_max_tokens_cut_call(self):
        srv, tok = self.srv, self.tok
        tools = synth.TOOLS
        msgs = [SYS, {'role': 'user', 'content': 'Patch main.py.'}]
        reply = {'role': 'assistant', 'reasoning_content': 'Edit it.', 'content': '',
                 'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {
                     'name': 'edit_file', 'arguments': {'path': 'src/main.py', 'old_str': 'a = 1\n' * 10,
                                                        'new_str': 'b = 2\n' * 20}}}]}
        p, r = U.script(srv.eng, msgs, reply, tools)
        n = next(k for k in range(len(r)) if tok.decode(r[:k]).count('<parameter=new_str>\n') and
                 tok.decode(r[:k]).count('b = 2') >= 3)
        body = {'messages': msgs, 'tools': tools, 'max_tokens': n, 'temperature': 0}
        msg, usage, fin, _ = openai_turn(srv, body, False)
        self.assertEqual(fin, 'length')
        self.assertNotIn('tool_calls', msg)
        self.assertEqual(usage['completion_tokens'], n)
        st, h, evs = U.sse_post(srv, '/v1/chat/completions', {**body, 'stream': True})
        chunks = [json.loads(d) for _, d in evs if d != '[DONE]']
        self.assertFalse(any('tool_calls' in c['choices'][0]['delta'] for c in chunks if c['choices']),
                         'a cut call is never streamed (whole-call mode)')
        self.assertEqual([c['choices'][0]['finish_reason'] for c in chunks if c['choices']][-1], 'length')
        m = sdk(srv).messages.create(model='k3-coder', max_tokens=n, system=SYS['content'], tools=ANTH_TOOLS,
                                     messages=[{'role': 'user', 'content': 'Patch main.py.'}],
                                     thinking={'type': 'enabled', 'budget_tokens': 1024})
        self.assertEqual(m.stop_reason, 'max_tokens')
        self.assertFalse([b for b in m.content if b.type == 'tool_use'])
        with sdk(srv).messages.stream(model='k3-coder', max_tokens=n, system=SYS['content'], tools=ANTH_TOOLS,
                                      messages=[{'role': 'user', 'content': 'Patch main.py.'}],
                                      thinking={'type': 'enabled', 'budget_tokens': 1024}) as s:
            fm = s.get_final_message()
        self.assertEqual(fm.stop_reason, 'max_tokens')
        self.assertFalse([b for b in fm.content if b.type == 'tool_use'])

    def test_cut_call_sweep(self):
        """fix round: a call cut by max_tokens anywhere before its </function> (inside a value, between parameters,
        right after the name) is never reported, on OpenAI non-stream / whole-call stream, Anthropic create and the
        SDK stream; once </function> is out the call is reported (finish still length)"""
        srv, tok = self.srv, self.tok
        tools = synth.TOOLS
        msgs = [SYS, {'role': 'user', 'content': 'Patch util.py (cut sweep).'}]
        reply = {'role': 'assistant', 'reasoning_content': 'Edit.', 'content': '',
                 'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {
                     'name': 'edit_file', 'arguments': {'path': 'util.py', 'old_str': 'a = 1\n',
                                                        'new_str': 'b = 2\n'}}}]}
        p, r = U.script(srv.eng, msgs, reply, tools)
        text_at = [tok.decode(r[:k]) for k in range(len(r) + 1)]
        i_open = next(k for k in range(len(r) + 1) if '<tool_call>' in text_at[k])
        i_close = next(k for k in range(len(r) + 1) if '</function>' in text_at[k])
        body = {'messages': msgs, 'tools': tools, 'temperature': 0}
        anth = dict(model='k3-coder', system=SYS['content'], tools=ANTH_TOOLS, temperature=0,
                    messages=[{'role': 'user', 'content': msgs[1]['content']}],
                    thinking={'type': 'enabled', 'budget_tokens': 1024})
        n = 0
        for mt in range(i_open, i_close + 2):
            want_call = mt >= i_close
            for stream in (False, True):
                msg, usage, fin, _ = openai_turn(srv, {**body, 'max_tokens': mt}, stream)
                self.assertEqual(fin, 'length', (mt, stream))
                self.assertEqual(bool(msg.get('tool_calls')), want_call, (mt, stream, text_at[mt][-60:]))
            st, h, j = U.post_json(srv, '/v1/messages', {**anth, 'max_tokens': mt})
            self.assertEqual((st, j['stop_reason']), (200, 'max_tokens'))
            self.assertEqual(bool([b for b in j['content'] if b['type'] == 'tool_use']), want_call, mt)
            if mt % 4 == 0 or want_call:
                kw = {k: v for k, v in anth.items() if k != 'temperature'}
                with sdk(srv).messages.stream(**kw, max_tokens=mt, extra_body={'temperature': 0}) as st_:
                    fm = st_.get_final_message()
                self.assertEqual(bool([b for b in fm.content if b.type == 'tool_use']), want_call, mt)
            n += 1
        print(f'\n[cut sweep] max_tokens at each of {n} tokens from <tool_call> to </function>: no call reported '
              'before </function> (OpenAI non-stream / stream, Anthropic create / SDK stream), reported after')

    def _forced(self, model, tool_choice_openai, tool_choice_anth, forced_text, cont_text, want_args, thinking=True):
        srv, tok = self.srv, self.tok
        tools = synth.TOOLS
        msgs = [SYS, {'role': 'user', 'content': f'Look at the readme ({model}, {forced_text!r}).'}]
        client = None if thinking else {'enable_thinking': False}
        reply = {'role': 'assistant', 'reasoning_content': 'I know this already.' if thinking else '',
                 'content': 'The readme says hello.'}
        p, r = U.script(srv.eng, msgs, reply, tools, model=model, client=client)
        forced = tok.encode(forced_text)
        if thinking:
            k = r.index(tok.special['</think>'])
            base = p + r[:k + 1] + forced
        else:
            base = p + forced
        srv.eng.register(base, tok.encode(cont_text))
        body = {'model': model, 'messages': msgs, 'tools': tools, 'tool_choice': tool_choice_openai, 'temperature': 0}
        for stream in (False, True):
            msg, usage, fin, _ = openai_turn(srv, body, stream)
            self.assertEqual(fin, 'tool_calls')
            self.assertEqual(openai_got(msg)['tool_calls'], [('read_file', want_args)])
            if not thinking:
                self.assertEqual(usage['prompt_tokens'], len(p) + len(forced))
        rows = U.wait_rows(srv, 1)
        self.assertEqual(rows[-1]['forced'], len(forced) if thinking else 0)
        kw = dict(model=model, max_tokens=500, system=SYS['content'], tools=ANTH_TOOLS, tool_choice=tool_choice_anth,
                  messages=[{'role': 'user', 'content': msgs[1]['content']}])
        if thinking:
            kw['thinking'] = {'type': 'enabled', 'budget_tokens': 1024}
        m = sdk(srv).messages.create(**kw)
        self.assertEqual(m.stop_reason, 'tool_use')
        self.assertEqual([(b.name, b.input) for b in m.content if b.type == 'tool_use'], [('read_file', want_args)])
        # append-only after a forced call: the next turn reuses everything but the unfed EOS
        nxt = msgs_wire(msgs) + [msg, {'role': 'tool', 'tool_call_id': msg['tool_calls'][0]['id'], 'content': 'hi'}]
        st, h, j = U.post_json(srv, '/v1/chat/completions', {'model': model, 'messages': nxt, 'tools': tools,
                                                             'max_tokens': 1, 'temperature': 0})
        gen = (r[:r.index(tok.special['</think>']) + 1] + forced if thinking else forced) + tok.encode(cont_text)
        self.assertEqual(j['usage']['prompt_tokens_details']['cached_tokens'], len(p) + len(gen) - 1)

    def test_forced_call_respects_max_tokens(self):
        """fix round: the forced '<tool_call>...' ids count against max_tokens (they used to overrun it), on
        OpenAI stream / non-stream and Anthropic, and against the context room"""
        srv, tok = self.srv, self.tok
        tools = synth.TOOLS
        msgs = [SYS, {'role': 'user', 'content': 'Look at the readme (max_tokens sweep).'}]
        reply = {'role': 'assistant', 'reasoning_content': 'Short.', 'content': 'Hello.'}
        p, r = U.script(srv.eng, msgs, reply, tools)
        forced = tok.encode('\n\n<tool_call>\n<function=read_file>\n')
        k = r.index(tok.special['</think>'])
        srv.eng.register(p + r[:k + 1] + forced,
                         tok.encode('<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call><|im_end|>'))
        tc_o = {'type': 'function', 'function': {'name': 'read_file'}}
        seen = []
        for mt in range(k + 1, k + len(forced) + 3):
            body = {'messages': msgs, 'tools': tools, 'tool_choice': tc_o, 'temperature': 0, 'max_tokens': mt}
            for stream in (False, True):
                msg, usage, fin, _ = openai_turn(srv, body, stream)
                self.assertLessEqual(usage['completion_tokens'], mt, (mt, stream))
                self.assertEqual((fin, usage['completion_tokens']), ('length', mt))
                self.assertNotIn('tool_calls', msg, (mt, stream))
            st, h, j = U.post_json(srv, '/v1/messages', {
                'model': 'k3-coder', 'max_tokens': mt, 'system': SYS['content'], 'tools': ANTH_TOOLS,
                'tool_choice': {'type': 'tool', 'name': 'read_file'}, 'thinking': {'type': 'enabled',
                                                                                   'budget_tokens': 1024},
                'messages': [{'role': 'user', 'content': msgs[1]['content']}], 'temperature': 0})
            self.assertEqual((st, j['stop_reason'], j['usage']['output_tokens']), (200, 'max_tokens', mt))
            self.assertFalse([b for b in j['content'] if b['type'] == 'tool_use'], mt)
            seen.append(mt)
        # the context room below the forced length: max_tokens absent, max_ctx just above the prompt
        eng2 = ScriptedEngine(ntok=4)
        p2, r2 = U.script(eng2, msgs, reply, tools)
        for room in (k + 2, k + 4):
            srv2 = U.make_server(eng2, max_ctx=len(p2) + room)
            try:
                msg, usage, fin, _ = openai_turn(srv2, {'messages': msgs, 'tools': tools, 'tool_choice': tc_o,
                                                        'temperature': 0}, False)
                self.assertEqual((fin, usage['completion_tokens']), ('length', room))
                self.assertLessEqual(usage['prompt_tokens'] + usage['completion_tokens'], srv2.max_ctx)
                self.assertEqual(srv2.worker.stats['internal_errors'], 0)
            finally:
                srv2.stop()
        print(f'\n[forcing] max_tokens {seen[0]}..{seen[-1]} around a {len(forced)}-token forced call: '
              'completion_tokens == max_tokens, finish length, on OpenAI stream / non-stream and Anthropic; '
              'context room clamp holds')

    def test_tool_choice_named_thinking(self):
        self._forced('k3-coder', {'type': 'function', 'function': {'name': 'read_file'}},
                     {'type': 'tool', 'name': 'read_file'}, '\n\n<tool_call>\n<function=read_file>\n',
                     '<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call><|im_end|>',
                     {'path': 'README.md'})

    def test_tool_choice_required_thinking(self):
        self._forced('k3-coder', 'required', {'type': 'any'}, '\n\n<tool_call>\n',
                     '<function=read_file>\n<parameter=path>\ndocs/a.md\n</parameter>\n<parameter=limit>\n5\n'
                     '</parameter>\n</function>\n</tool_call><|im_end|>', {'path': 'docs/a.md', 'limit': 5})

    def test_tool_choice_named_nothink(self):
        self._forced('k3-coder-nothink', {'type': 'function', 'function': {'name': 'read_file'}},
                     {'type': 'tool', 'name': 'read_file'}, '<tool_call>\n<function=read_file>\n',
                     '<parameter=path>\nx.py\n</parameter>\n</function>\n</tool_call><|im_end|>', {'path': 'x.py'},
                     thinking=False)

    def test_tool_choice_none(self):
        srv = self.srv
        tools = synth.TOOLS
        msgs = [SYS, {'role': 'user', 'content': 'Read the readme (none).'}]
        reply = {'role': 'assistant', 'reasoning_content': 'Read it.', 'content': '',
                 'tool_calls': [{'id': 'c', 'type': 'function',
                                 'function': {'name': 'read_file', 'arguments': {'path': 'README.md'}}}]}
        U.script(srv.eng, msgs, reply, tools)
        msg, usage, fin, _ = openai_turn(srv, {'messages': msgs, 'tools': tools, 'tool_choice': 'none',
                                               'temperature': 0}, False)
        self.assertEqual(fin, 'stop')
        self.assertNotIn('tool_calls', msg)
        self.assertIn('<tool_call>\n<function=read_file>', msg['content'])


# ==================================================================================================== completions, misc routes
class TestCompletionsAndRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = U.make_server(ScriptedEngine(ntok=4))
        cls.tok = common.tokenizer()

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def test_completions(self):
        srv, tok = self.srv, self.tok
        prompt = 'def add(a, b):\n    """Add two numbers."""\n'
        ids = [tok.bos_id] + tok.encode(prompt)
        srv.eng.register(ids, tok.encode('    return a + b\n') + [tok.eos_id])
        st, h, j = U.post_json(srv, '/v1/completions', {'model': 'k3-coder', 'prompt': prompt, 'max_tokens': 50,
                                                        'temperature': 0})
        self.assertEqual((st, j['object'], j['choices'][0]['text'], j['choices'][0]['finish_reason']),
                         (200, 'text_completion', '    return a + b\n', 'stop'))
        self.assertEqual(j['usage']['prompt_tokens'], len(ids))
        for arr in (ids, ids[1:]):          # a token array; 248044 is put first when missing
            st, h, j = U.post_json(srv, '/v1/completions', {'prompt': arr, 'max_tokens': 50, 'temperature': 0})
            self.assertEqual(j['choices'][0]['text'], '    return a + b\n')
            self.assertEqual(j['usage']['prompt_tokens'], len(ids))
        st, h, evs = U.sse_post(srv, '/v1/completions', {'prompt': prompt, 'max_tokens': 50, 'temperature': 0,
                                                         'stream': True})
        self.assertEqual(evs[-1][1], '[DONE]')
        ch = [json.loads(d) for _, d in evs[:-1]]
        self.assertEqual(''.join(c['choices'][0]['text'] for c in ch), '    return a + b\n')
        self.assertEqual(ch[-1]['choices'][0]['finish_reason'], 'stop')
        st, h, j = U.post_json(srv, '/v1/completions', {'prompt': prompt, 'max_tokens': 3, 'temperature': 0})
        self.assertEqual(j['choices'][0]['finish_reason'], 'length')

    def test_utf8_stream(self):
        srv = self.srv
        msgs = [{'role': 'user', 'content': 'Say it in emoji.'}]
        text = 'Fertig 🎉🎉 — 完成。✅ 🧑‍💻 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 한국어'
        U.script(srv.eng, msgs, {'role': 'assistant', 'reasoning_content': 'Emoji: 🎉.', 'content': text}, None)
        st, h, evs = U.sse_post(srv, '/v1/chat/completions', {'messages': msgs, 'stream': True, 'temperature': 0})
        deltas = [json.loads(d)['choices'][0]['delta'] for _, d in evs[:-1] if json.loads(d)['choices']]
        pieces = [d['content'] for d in deltas if d.get('content')]
        self.assertEqual(''.join(pieces), text)
        self.assertFalse(any('�' in p for p in pieces))
        m = sdk(srv).messages.create(model='k3-coder', max_tokens=200, messages=msgs,
                                     thinking={'type': 'enabled', 'budget_tokens': 1024})
        self.assertEqual(m.content[-1].text, text)

    def test_routes(self):
        srv = self.srv
        st, h, data = U.request(srv, 'GET', '/health')
        self.assertEqual((st, json.loads(data)), (200, {'status': 'ok'}))
        st, h, data = U.request(srv, 'GET', '/v1/models')
        ids = [m['id'] for m in json.loads(data)['data']]
        self.assertIn('k3-coder', ids)
        self.assertIn('k3-coder-nothink', ids)
        st, h, data = U.request(srv, 'GET', '/props')
        self.assertEqual(json.loads(data)['n_ctx'], srv.max_ctx)
        st, h, data = U.request(srv, 'GET', '/metrics')
        self.assertIn(b'caminus:worker_total{counter="jobs"}', data)
        self.assertIn(b'llamacpp:prompt_tokens_total', data)
        tools, steps = synth.agent_conversation(5, rounds=1, user_turns=1)
        msgs = steps[0][0]
        want = len(common.builder().build(msgs, tools).ids)
        r = sdk(srv).messages.count_tokens(model='k3-coder', system=msgs[0]['content'], tools=ANTH_TOOLS,
                                           messages=[{'role': 'user', 'content': msgs[1]['content']}],
                                           thinking={'type': 'enabled', 'budget_tokens': 1024})
        self.assertEqual(r.input_tokens, want)


# ==================================================================================================== malformed
class TestMalformed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = U.make_server(ScriptedEngine(ntok=4), max_ctx=4096)

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def test_cases(self):
        srv = self.srv
        ok_chat = {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 2}
        ok_anth = {'model': 'k3-coder', 'max_tokens': 2, 'messages': [{'role': 'user', 'content': 'hi'}]}
        long = 'word ' * 5000
        cases = [
            ('/v1/chat/completions', b'{not json', 400),
            ('/v1/chat/completions', b'[1, 2]', 400),
            ('/v1/chat/completions', b'\xff\xfe', 400),
            ('/v1/chat/completions', {}, 400),
            ('/v1/chat/completions', {'messages': []}, 400),
            ('/v1/chat/completions', {'messages': [{'role': 'wizard', 'content': 'x'}]}, 400),
            ('/v1/chat/completions', {'messages': [{'role': 'user', 'content': [{'type': 'image_url',
                                                                                 'image_url': {'url': 'x'}}]}]}, 400),
            ('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 5}]}, 400),
            ('/v1/chat/completions', {'messages': [{'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'a', 'type': 'function', 'function': {'name': 'f', 'arguments': '{bad'}}]}]}, 400),
            ('/v1/chat/completions', {**ok_chat, 'n': 2}, 400),
            ('/v1/chat/completions', {**ok_chat, 'response_format': {'type': 'json_schema', 'json_schema': {}}}, 400),
            ('/v1/chat/completions', {**ok_chat, 'logprobs': True}, 400),
            ('/v1/chat/completions', {**ok_chat, 'stop': 5}, 400),
            ('/v1/chat/completions', {**ok_chat, 'max_tokens': 0}, 400),
            ('/v1/chat/completions', {**ok_chat, 'max_tokens': 'ten'}, 400),
            ('/v1/chat/completions', {**ok_chat, 'temperature': 'hot'}, 400),
            ('/v1/chat/completions', {**ok_chat, 'temperature': -1}, 400),
            ('/v1/chat/completions', {**ok_chat, 'stream': 'yes'}, 400),
            ('/v1/chat/completions', {**ok_chat, 'tool_choice': 'required'}, 400),
            ('/v1/chat/completions', {**ok_chat, 'tools': synth.TOOLS,
                                      'tool_choice': {'type': 'function', 'function': {'name': 'nope'}}}, 400),
            ('/v1/chat/completions', {**ok_chat, 'tools': 'all'}, 400),
            ('/v1/chat/completions', {**ok_chat, 'chat_template_kwargs': 'x'}, 400),
            ('/v1/chat/completions', {'messages': [{'role': 'user', 'content': long}]}, 400),
            ('/v1/messages', {'messages': [{'role': 'user', 'content': 'hi'}]}, 400),           # no max_tokens
            ('/v1/messages', {**ok_anth, 'messages': [{'role': 'system', 'content': 'x'}]}, 400),
            ('/v1/messages', {**ok_anth, 'messages': [{'role': 'user', 'content': [{'type': 'image'}]}]}, 400),
            ('/v1/messages', {**ok_anth, 'system': 5}, 400),
            ('/v1/messages', {**ok_anth, 'tools': [{'name': 'web', 'type': 'web_search_20250305'}]}, 400),
            ('/v1/messages', {**ok_anth, 'tool_choice': {'type': 'sometimes'}}, 400),
            ('/v1/messages', {**ok_anth, 'messages': [{'role': 'user', 'content': long}]}, 400),
            ('/v1/messages/count_tokens', {'messages': 'x'}, 400),
            ('/v1/completions', {'prompt': 5}, 400),
            ('/v1/completions', {'prompt': [1, 2, 10 ** 9]}, 400),
            ('/v1/completions', {'prompt': ['a', 'b']}, 400),
            ('/v1/completions', {'prompt': 'x', 'echo': True}, 400),
            ('/v1/nothing', {}, 404),
            ('/v1/messages/batches', {}, 404),
            # fix round: non-finite / overflowing numbers, lone surrogates, deep nesting, tool_choice shapes
            ('/v1/chat/completions', json.dumps(ok_chat)[:-1].encode() + b', "temperature": NaN}', 400),
            ('/v1/chat/completions', json.dumps(ok_chat)[:-1].encode() + b', "frequency_penalty": -Infinity}', 400),
            ('/v1/chat/completions', json.dumps(ok_chat)[:-1].encode() + b', "temperature": 1e999}', 400),
            ('/v1/chat/completions', json.dumps(ok_chat)[:-1].encode() + b', "max_tokens": 1e999}', 400),
            ('/v1/chat/completions', {**ok_chat, 'max_tokens': 10 ** 400}, 400),
            ('/v1/chat/completions', {**ok_chat, 'temperature': 1e308}, 400),
            ('/v1/chat/completions', {**ok_chat, 'presence_penalty': 1e308}, 400),
            ('/v1/chat/completions', {**ok_chat, 'repeat_penalty': 0}, 400),
            ('/v1/chat/completions', {**ok_chat, 'top_k': 10 ** 400}, 400),
            ('/v1/chat/completions', {**ok_chat, 'seed': 2 ** 70}, 400),
            ('/v1/chat/completions', {**ok_chat, 'penalty_last_n': -5}, 400),
            ('/v1/chat/completions', {**ok_chat, 'top_p': 0}, 400),
            ('/v1/chat/completions', {**ok_chat, 'temperature': True}, 400),
            ('/v1/chat/completions', b'{"messages": [{"role": "user", "content": "a\\ud800b"}], "max_tokens": 2}', 400),
            ('/v1/chat/completions', b'{"model": "\\udfff", "messages": [{"role": "user", "content": "hi"}]}', 400),
            ('/v1/chat/completions', b'[' * 200000, 400),
            ('/v1/chat/completions', {**ok_chat, 'tools': synth.TOOLS,
                                      'tool_choice': {'type': 'function', 'function': {}}}, 400),
            ('/v1/messages', json.dumps(ok_anth)[:-1].encode() + b', "temperature": NaN}', 400),
            ('/v1/messages', b'{"model": "k3-coder", "max_tokens": 2, "messages": [{"role": "user", '
                             b'"content": "\\ud83d"}]}', 400),
            ('/v1/messages', {**ok_anth, 'tools': ANTH_TOOLS, 'tool_choice': {'type': 'tool'}}, 400),
            ('/v1/messages', {**ok_anth, 'tools': ANTH_TOOLS, 'tool_choice': {'type': 'tool', 'name': 5}}, 400),
            ('/v1/messages', {**ok_anth, 'tools': ANTH_TOOLS,
                              'tool_choice': {'type': 'any', 'disable_parallel_tool_use': 'yes'}}, 400),
            ('/v1/completions', b'{"prompt": "\\ud800", "max_tokens": 2}', 400),
            ('/v1/completions', b'{"prompt": "x", "max_tokens": 2, "temperature": Infinity}', 400),
            ('/v1/completions', b'{"prompt": "x", "max_tokens": 2, "seed": NaN}', 400),
        ]
        for path, body, want in cases:
            raw = body if isinstance(body, bytes) else None
            st, h, data = U.request(srv, 'POST', path, None if raw is not None else body, raw=raw)
            self.assertEqual(st, want, (path, body if not isinstance(body, dict) else str(body)[:120], data[:300]))
            j = json.loads(data)
            if path.startswith('/v1/messages'):
                self.assertEqual(j['type'], 'error')
                self.assertTrue(j['error']['type'] and j['error']['message'])
            else:
                self.assertTrue(j['error']['type'] and j['error']['message'])
        # the context error shapes harnesses look for
        st, h, data = U.request(srv, 'POST', '/v1/messages', {**ok_anth, 'messages': [{'role': 'user',
                                                                                         'content': long}]})
        self.assertIn('prompt is too long', json.loads(data)['error']['message'])
        st, h, data = U.request(srv, 'POST', '/v1/chat/completions', {'messages': [{'role': 'user', 'content': long}]})
        self.assertEqual(json.loads(data)['error']['code'], 'context_length_exceeded')
        # wrong methods, missing / oversized bodies
        self.assertEqual(U.request(srv, 'GET', '/v1/chat/completions')[0], 405)
        self.assertEqual(U.request(srv, 'POST', '/v1/models', {})[0], 405)
        self.assertEqual(self._raw(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n\r\n')[:12], b'HTTP/1.1 411')
        self.assertEqual(self._raw(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 999999999999'
                                   b'\r\n\r\n')[:12], b'HTTP/1.1 413')
        # chunked request bodies are accepted
        body = json.dumps(ok_chat).encode()
        resp = self._raw(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n'
                         b'Connection: close\r\n\r\n' + f'{len(body):x}\r\n'.encode() + body + b'\r\n0\r\n\r\n')
        self.assertEqual(resp[:12], b'HTTP/1.1 200')
        # keep-alive across errors: one connection, bad then good requests
        import http.client
        c = http.client.HTTPConnection('127.0.0.1', srv.port, timeout=30)
        for body, want in ((b'{bad', 400), (json.dumps(ok_chat).encode(), 200), (b'{}', 400),
                           (json.dumps(ok_chat).encode(), 200)):
            c.request('POST', '/v1/chat/completions', body, {'Content-Type': 'application/json'})
            r = c.getresponse()
            r.read()
            self.assertEqual(r.status, want)
        c.close()
        # still healthy, and the SDK sees typed errors
        with self.assertRaises(anthropic.BadRequestError):
            sdk(srv).messages.create(model='k3-coder', max_tokens=5, messages=[{'role': 'user', 'content': long}])
        st, h, j = U.post_json(srv, '/v1/chat/completions', ok_chat)
        self.assertEqual(st, 200)
        # a valid surrogate pair is fine (an emoji)
        st, h, data = U.request(srv, 'POST', '/v1/chat/completions',
                                raw=b'{"messages": [{"role": "user", "content": "\\ud83d\\ude00"}], "max_tokens": 2}')
        self.assertEqual(st, 200)
        self.assertEqual(srv.worker.stats['internal_errors'], 0)
        self.assertEqual(srv.worker.stats['resets'], 0)
        print(f'\n[malformed] {len(cases)} bad bodies + 405/411/413 + chunked + keep-alive across errors: '
              'statuses and error shapes as specified; server healthy after; 0 internal errors, 0 resets')

    def _raw(self, data):
        s = socket.create_connection(('127.0.0.1', self.srv.port), timeout=10)
        s.sendall(data)
        out = b''
        try:
            while True:
                b = s.recv(65536)
                if not b:
                    break
                out += b
                if b'\r\n\r\n' in out and b'Content-Length' in out:
                    hdr, _, rest = out.partition(b'\r\n\r\n')
                    cl = int([ln for ln in hdr.split(b'\r\n') if ln.lower().startswith(b'content-length')][0]
                             .split(b':')[1])
                    if len(rest) >= cl:
                        break
        except socket.timeout:
            pass
        s.close()
        return out


# ==================================================================================================== disconnects
class TestDisconnect(unittest.TestCase):
    """client gone mid-stream, during a non-streaming request and while queued: the job is cancelled at the next
    pass boundary, logged as client_gone, and the engine state stays consistent (the next request reuses it)"""

    @classmethod
    def setUpClass(cls):
        tok = common.tokenizer()
        eng = MockEngine(ntok=2, vocab=tok.n_vocab, eos_id=None, seed=5)
        cls.srv = U.make_server(eng, pass_sleep=0.01)
        cls.tok = tok

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def _gone_row(self, n_before, timeout=5.0):
        t = time.time() + timeout
        while time.time() < t:
            rows = U.log_rows(self.srv)
            new = [r for r in rows[n_before:] if r.get('client_gone')]
            if new:
                return new[-1]
            time.sleep(0.02)
        self.fail('no client_gone row')

    def _read_events(self, s, n):
        buf = b''
        while buf.count(b'data:') < n:
            b = s.recv(4096)
            if not b:
                break
            buf += b
        return buf

    def test_mid_stream(self):
        srv, tok = self.srv, self.tok
        cases = [('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'go'}], 'stream': True,
                                           'max_tokens': 20000}),
                 ('/v1/messages', {'model': 'k3-coder', 'max_tokens': 20000, 'stream': True,
                                   'messages': [{'role': 'user', 'content': 'go'}]}),
                 ('/v1/completions', {'prompt': [tok.bos_id, 1000, 1001, 1002], 'stream': True, 'max_tokens': 20000})]
        for path, body in cases:
            n0 = len(U.log_rows(srv))
            s = U.raw_socket_request(srv, path, body)
            self._read_events(s, 5)
            t_close = time.time()
            s.close()
            row = self._gone_row(n0)
            dt = time.time() - t_close
            self.assertEqual(row['finish'], 'cancel')
            self.assertLess(row['completion_tokens'], 2000)
            self.assertLess(dt, 3.0)
            print(f'\n[disconnect] {path} stream: cancelled after {row["completion_tokens"]} tokens, '
                  f'row {dt * 1e3:.0f} ms after close')
        # the engine is consistent: a continuation of the last prompt reuses its prompt-end checkpoint
        st, h, j = U.post_json(srv, '/v1/completions', {'prompt': [tok.bos_id, 1000, 1001, 1002, 7, 8],
                                                        'max_tokens': 2})
        self.assertEqual((st, j['usage']['prompt_tokens_details']['cached_tokens']), (200, 4))

    def test_sdk_stream_close(self):
        srv = self.srv
        n0 = len(U.log_rows(srv))
        with sdk(srv).messages.stream(model='k3-coder', max_tokens=20000,
                                      messages=[{'role': 'user', 'content': 'go on'}]) as s:
            for i, ev in enumerate(s):
                if i >= 6:
                    break
        row = self._gone_row(n0)
        self.assertEqual(row['finish'], 'cancel')

    def test_non_stream_and_queued(self):
        srv, tok = self.srv, self.tok
        n0 = len(U.log_rows(srv))
        s = U.raw_socket_request(srv, '/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'x'}],
                                                               'max_tokens': 20000})
        time.sleep(0.4)
        s.close()
        row = self._gone_row(n0)
        self.assertEqual(row['finish'], 'cancel')
        self.assertGreater(row['completion_tokens'], 0)
        # queued: A runs, B waits and leaves; B never reaches the engine
        n0 = len(U.log_rows(srv))
        res = {}
        ta = threading.Thread(target=lambda: res.setdefault('a', U.post_json(
            srv, '/v1/completions', {'prompt': [tok.bos_id, 5, 6], 'max_tokens': 80})))
        ta.start()
        time.sleep(0.1)
        jobs0 = srv.worker.stats['jobs']
        s = U.raw_socket_request(srv, '/v1/completions', {'prompt': [tok.bos_id, 9, 9], 'max_tokens': 80})
        time.sleep(0.3)
        s.close()
        ta.join()
        row = self._gone_row(n0)
        self.assertEqual(res['a'][0], 200)
        self.assertEqual(row.get('completion_tokens'), 0)
        time.sleep(0.2)
        self.assertEqual(srv.worker.stats['jobs'], jobs0, 'the cancelled queued job did not run')


# ==================================================================================================== queue, faults, pings
class TestQueueFaultsPings(unittest.TestCase):
    def test_fifo_one_sequence(self):
        tok = common.tokenizer()
        srv = U.make_server(MockEngine(ntok=1, vocab=tok.n_vocab, eos_id=None, seed=2), pass_sleep=0.005)
        try:
            out = {}
            ths = []
            for i in range(4):
                t = threading.Thread(target=lambda i=i: out.setdefault(i, U.post_json(
                    srv, '/v1/completions', {'prompt': [tok.bos_id, 100 + i], 'max_tokens': 25})))
                t.start()
                ths.append(t)
                time.sleep(0.05)
            for t in ths:
                t.join()
            self.assertTrue(all(out[i][0] == 200 for i in range(4)))
            hist = srv.worker.history
            for (_, sa, a0, a1), (_, sb, b0, b1) in zip(hist, hist[1:]):
                self.assertLessEqual(a1, b0, 'two jobs overlapped on the engine')
                self.assertLess(sa, sb, 'not FIFO')
            q = [out[i][2]['timings']['queue_ms'] for i in range(4)]
            self.assertGreater(max(q[1:]), 20.0)
            print(f'\n[queue] 4 concurrent requests ran one at a time, queue_ms {[round(x) for x in q]}')
            # a full queue answers 503 / 529 at once
            srv.worker.max_queue = 1
            ths = [threading.Thread(target=lambda: U.post_json(srv, '/v1/completions', {
                'prompt': [tok.bos_id, 1], 'max_tokens': 60})) for _ in range(2)]
            for t in ths:
                t.start()
                time.sleep(0.05)
            st, h, j = U.post_json(srv, '/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'x'}]})
            self.assertEqual((st, j['error']['type']), (503, 'server_error'))
            st, h, j = U.post_json(srv, '/v1/messages', {'model': 'k3-coder', 'max_tokens': 3,
                                                         'messages': [{'role': 'user', 'content': 'x'}]})
            self.assertEqual((st, j['error']['type']), (529, 'overloaded_error'))
            for t in ths:
                t.join()
        finally:
            srv.stop()

    def test_engine_faults(self):
        tok = common.tokenizer()
        fail = set()
        srv = U.make_server(MockEngine(ntok=2, vocab=tok.n_vocab, eos_id=None, seed=4),
                            fail_at=lambda n: n in fail)
        try:
            sess = srv.sess
            chat = {'messages': [{'role': 'user', 'content': 'hello there'}], 'max_tokens': 40}
            anth = {'model': 'k3-coder', 'max_tokens': 40, 'messages': [{'role': 'user', 'content': 'hello there'}]}
            # during prefill, non-stream and stream: a real status before any header
            for path, body, want, et in (('/v1/chat/completions', chat, 503, 'server_error'),
                                         ('/v1/chat/completions', {**chat, 'stream': True}, 503, 'server_error'),
                                         ('/v1/messages', anth, 529, 'overloaded_error'),
                                         ('/v1/messages', {**anth, 'stream': True}, 529, 'overloaded_error')):
                fail.add(sess.total_passes + 1)
                st, h, data = U.request(srv, 'POST', path, body)
                j = json.loads(data)
                self.assertEqual(st, want, path)
                self.assertEqual(j['error']['type'], et)
                st, h, j = U.post_json(srv, '/v1/chat/completions', chat)        # recovered, cold
                self.assertEqual(st, 200)
                self.assertEqual(j['usage']['prompt_tokens_details']['cached_tokens'], 0)
            # mid-stream: an SSE error event (OpenAI: an error object; Anthropic: event: error)
            for path, body in (('/v1/chat/completions', {**chat, 'stream': True}),
                               ('/v1/messages', {**anth, 'stream': True})):
                fail.add(sess.total_passes + 15)
                st, h, data = U.request(srv, 'POST', path, body)
                self.assertEqual(st, 200)
                evs, _ = U.parse_sse(data)
                if path == '/v1/messages':
                    self.assertEqual(evs[-1][0], 'error')
                    self.assertEqual(json.loads(evs[-1][1])['error']['type'], 'overloaded_error')
                else:
                    self.assertEqual(json.loads(evs[-1][1])['error']['type'], 'server_error')
            fail.add(sess.total_passes + 15)
            with self.assertRaises(anthropic.APIStatusError):
                with sdk(srv).messages.stream(**anth) as s:
                    s.get_final_message()
            fail.add(sess.total_passes + 1)
            with self.assertRaises(anthropic.APIStatusError) as cm:
                sdk(srv).messages.create(**anth)
            self.assertEqual(cm.exception.status_code, 529)
            st, h, j = U.post_json(srv, '/v1/chat/completions', chat)
            self.assertEqual(st, 200)
            self.assertEqual(srv.worker.stats['engine_errors'], 8)
            rows = U.log_rows(srv)
            self.assertEqual(sum(1 for r in rows if r['status'] in (503, 529)), 8)
            print('\n[faults] engine errors: 503 / 529 before headers, SSE error events mid-stream, SDK raises; '
                  'the session resets and the next request is served')
        finally:
            srv.stop()

    def test_pings(self):
        eng = ScriptedEngine(ntok=4)
        srv = U.make_server(eng, pass_sleep=0.004, ping_interval=0.25)
        try:
            tools, steps = synth.agent_conversation(77, rounds=1, user_turns=1)
            msgs, reply = steps[0]
            p, r = U.script(eng, msgs, reply, tools)
            kw = dict(model='k3-coder', max_tokens=4096, system=msgs[0]['content'], tools=ANTH_TOOLS,
                      messages=[{'role': 'user', 'content': msgs[1]['content']}],
                      thinking={'type': 'enabled', 'budget_tokens': 1024})
            st, h, data = U.request(srv, 'POST', '/v1/messages', {**kw, 'stream': True})
            evs, _ = U.parse_sse(data)
            names = [e for e, _ in evs]
            self.assertEqual(names[0], 'message_start')
            first = names.index('content_block_start')
            self.assertGreater(names[:first].count('ping'), 0, names[:10])
            with sdk(srv).messages.stream(**{**kw, 'messages': kw['messages']}) as s:
                m = s.get_final_message()
            U.same_reply(self, anthropic_got([b.model_dump(exclude_none=True) for b in m.content]), reply)
            print(f'\n[ping] {names.count("ping")} pings during a {len(p)}-token prefill; the SDK ignores them')
            # queued behind a long job: message_start (provisional usage) comes before the first ping, and the
            # message_delta carries the real usage (the second turn reuses the first: cache_read > 0), which the SDK
            # applies to its final message
            tok = common.tokenizer()
            msgs2, reply2 = steps[1]
            p2, r2 = U.script(eng, msgs2, reply2, tools)
            blocks = [b.model_dump(exclude_none=True) for b in m.content]
            ids = [b['id'] for b in blocks if b['type'] == 'tool_use']
            kw2 = {**kw, 'messages': kw['messages'] + [{'role': 'assistant', 'content': blocks},
                                                       {'role': 'user',
                                                        'content': U.anthropic_user_blocks(msgs2[3:], ids)}]}
            want_cached = len(p) + len(r) - 1
            for via_sdk in (False, True):
                res = {}
                entered, release, expired = threading.Event(), threading.Event(), threading.Event()
                pings = [0]
                run = srv.worker._run
                handler = srv.httpd.RequestHandlerClass
                send_sse = handler._sse

                def held_run(job):
                    entered.set()
                    if not release.wait(10):
                        expired.set()
                    return run(job)

                def observed_sse(h, data, event=None):
                    out = send_sse(h, data, event)
                    if event == 'ping':
                        pings[0] += 1
                        if pings[0] >= 2:
                            release.set()
                    return out

                # A started client thread is not necessarily admitted, and a
                # finite job can finish while the next client is setting up.
                # Hold the actual worker until two real ping frames are written;
                # the raw client below still verifies those bytes and ordering.
                blocker = threading.Thread(target=lambda: res.setdefault('a', U.post_json(
                    srv, '/v1/completions', {'prompt': [tok.bos_id, 3, 4, 5], 'max_tokens': 200}, timeout=10)))
                with patch.object(srv.worker, '_run', held_run), patch.object(handler, '_sse', observed_sse):
                    blocker.start()
                    try:
                        self.assertTrue(entered.wait(10), 'blocker never reached the worker')
                        if not via_sdk:
                            st, h, data = U.request(srv, 'POST', '/v1/messages', {**kw2, 'stream': True})
                            evs, _ = U.parse_sse(data)
                            names = [e for e, _ in evs]
                            self.assertEqual(names[0], 'message_start', names[:8])
                            self.assertGreater(names[:names.index('content_block_start')].count('ping'), 1, names[:8])
                            ms = json.loads(evs[0][1])['message']['usage']
                            md = json.loads([d for e, d in evs if e == 'message_delta'][-1])['usage']
                            self.assertEqual((ms['input_tokens'], ms['cache_read_input_tokens']), (len(p2), 0))
                            self.assertEqual((md['input_tokens'], md['cache_read_input_tokens']),
                                             (len(p2) - want_cached, want_cached))
                        else:
                            with sdk(srv).messages.stream(**kw2) as s:
                                m2 = s.get_final_message()
                            U.same_reply(self, anthropic_got([b.model_dump(exclude_none=True) for b in m2.content]), reply2)
                            self.assertEqual((m2.usage.input_tokens, m2.usage.cache_read_input_tokens),
                                             (len(p2) - want_cached, want_cached))
                        self.assertFalse(expired.is_set(), 'no two ping frames while the worker was held')
                        self.assertGreaterEqual(pings[0], 2)
                    finally:
                        release.set()
                        blocker.join(timeout=10)
                self.assertFalse(blocker.is_alive(), 'blocking client did not finish')
                self.assertEqual(res['a'][0], 200)
            print(f'[ping] queued: message_start first with provisional usage (input {ms["input_tokens"]}, '
                  f'cache_read 0), pings, then message_delta with the real usage (input {md["input_tokens"]}, '
                  f'cache_read {md["cache_read_input_tokens"]}); the SDK final message has the real numbers')
        finally:
            srv.stop()


# ==================================================================================================== robustness (fix round)
class TestRobustness(unittest.TestCase):
    """fix round of S4: a bad request can never cost the cache.
    * every sampling field x hostile values (NaN, +-Infinity, overflowing literals, huge / tiny finite values, wrong
      types): 200 or 400, never 500; no internal error, no session reset; a turn that extends an earlier prompt
      still reuses it afterwards;
    * a host-side exception on the engine thread (injected into the sampler mid-decode, after a verify and before its
      accept): 500 'internal error', the session is aborted (the pass rolled back), not reset; the strict mock checks
      the state, and the next extending turn reuses the checkpoints;
    * cancel latency: flag -> the worker's done event, and socket close -> flag (the POLL_S poll)."""

    def test_sampling_fuzz_keeps_cache(self):
        tok = common.tokenizer()
        eng = MockEngine(ntok=4, vocab=tok.n_vocab, eos_id=None, seed=3)
        srv = U.make_server(eng)
        srv.worker.sess.n_checkpoints = 100000     # the ~400 valid requests below must not LRU-evict P's checkpoint
        try:
            P = [tok.bos_id] + list(range(2000, 2300))
            st, h, j = U.post_json(srv, '/v1/completions', {'prompt': P, 'max_tokens': 3, 'temperature': 0})
            self.assertEqual(st, 200)
            fields = ['temperature', 'top_p', 'min_p', 'top_k', 'repeat_penalty', 'frequency_penalty',
                      'presence_penalty', 'repeat_last_n', 'penalty_last_n', 'min_keep', 'seed']
            values = ['NaN', 'Infinity', '-Infinity', '1e999', '-1e999', '1e308', '-1e308', '1e30', '-1e30',
                      str(10 ** 400), '-' + str(10 ** 400), '-2', '-1', '0', '1e-30', '0.5', '2.5', '40', '"x"',
                      'true', '[]', '{}']
            nonfinite = {'NaN', 'Infinity', '-Infinity', '1e999', '-1e999'}
            counts = {200: 0, 400: 0}
            for f in fields:
                for v in values:
                    for route, raw in (('/v1/completions', '{"prompt": [%d, 1, 2], "max_tokens": 3, "%s": %s}'
                                        % (tok.bos_id, f, v)),
                                       ('/v1/chat/completions', '{"messages": [{"role": "user", "content": "q"}], '
                                        '"max_tokens": 3, "%s": %s}' % (f, v))):
                        st, h, data = U.request(srv, 'POST', route, raw=raw.encode())
                        self.assertIn(st, (200, 400), (route, f, v, data[:200]))
                        if v in nonfinite:
                            self.assertEqual(st, 400, (f, v))
                        counts[st] += 1
            for v in ('NaN', '1e999', '1e308'):
                raw = ('{"model": "k3-coder", "max_tokens": 3, "messages": [{"role": "user", "content": "q"}], '
                       '"temperature": %s}' % v)
                self.assertEqual(U.request(srv, 'POST', '/v1/messages', raw=raw.encode())[0], 400, v)
            st = srv.worker.stats
            self.assertEqual((st['internal_errors'], st['engine_errors'], st['resets']), (0, 0, 0))
            st, h, j = U.post_json(srv, '/v1/completions', {'prompt': P + [7, 8], 'max_tokens': 3, 'temperature': 0})
            self.assertEqual((st, j['usage']['prompt_tokens_details']['cached_tokens']), (200, len(P)))
            print(f'\n[robust] {len(fields)} sampling fields x {len(values)} values x 2 routes: '
                  f'{counts[200]} x 200, {counts[400]} x 400, 0 x 500; 0 internal errors, 0 resets; '
                  f'the next extending turn reused {len(P)} of {len(P) + 2}')
        finally:
            srv.stop()

    def test_internal_error_keeps_cache(self):
        from caminus.serve import worker as W
        tok = common.tokenizer()
        eng = MockEngine(ntok=4, vocab=tok.n_vocab, eos_id=None, seed=4)
        srv = U.make_server(eng)
        orig = W.Sampler

        class Boom(orig):
            calls = 0

            def sample_and_match(self, *a, **kw):
                if self.params.seed == 4242:
                    Boom.calls += 1
                    if Boom.calls == 3:
                        raise RuntimeError('injected host-side bug')
                return super().sample_and_match(*a, **kw)

        W.Sampler = Boom
        try:
            P = [tok.bos_id] + list(range(3000, 3300))
            st, h, j = U.post_json(srv, '/v1/completions', {'prompt': P, 'max_tokens': 3, 'temperature': 0})
            self.assertEqual(st, 200)
            # the failing request extends P and dies after its third verify, before the accept
            for path, body, want in (
                    ('/v1/completions', {'prompt': P + [9], 'max_tokens': 40, 'seed': 4242}, 500),
                    ('/v1/messages', {'model': 'k3-coder', 'max_tokens': 40, 'seed': 4242,
                                      'messages': [{'role': 'user', 'content': 'boom'}]}, 500)):
                Boom.calls = 0
                st, h, data = U.request(srv, 'POST', path, body)
                self.assertEqual(st, want, data[:200])
                msg = json.loads(data)['error']['message']
                self.assertTrue(msg.startswith('internal error: RuntimeError'), msg)
            stats = srv.worker.stats
            self.assertEqual((stats['internal_errors'], stats['aborts'], stats['resets']), (2, 2, 0))
            # the cache survived: P's and P+[9]'s prompt-end checkpoints are reused (the strict mock checks the state)
            st, h, j = U.post_json(srv, '/v1/completions', {'prompt': P + [9, 10], 'max_tokens': 3,
                                                            'temperature': 0})
            self.assertEqual((st, j['usage']['prompt_tokens_details']['cached_tokens']), (200, len(P) + 1))
            st, h, j = U.post_json(srv, '/v1/completions', {'prompt': P + [11], 'max_tokens': 3, 'temperature': 0})
            self.assertEqual((st, j['usage']['prompt_tokens_details']['cached_tokens']), (200, len(P)))
            self.assertEqual(srv.worker.stats['resets'], 0)
            print(f'\n[robust] injected host-side exceptions mid-decode: 500 internal error, session aborted '
                  f'(not reset); next extending turns reused {len(P) + 1} and {len(P)}')
        finally:
            W.Sampler = orig
            srv.stop()

    def test_cancel_latency(self):
        from caminus.serve.worker import Job
        tok = common.tokenizer()
        eng = MockEngine(ntok=2, vocab=tok.n_vocab, eos_id=None, seed=5)
        srv = U.make_server(eng, pass_sleep=0.01)
        try:
            flag_stop = []
            for i in range(5):
                job = srv.worker.submit(Job([tok.bos_id, 40 + i, 41], SamplingParams(seed=i).resolve(), 20000))
                n_text = 0
                while n_text < 5:
                    ev = job.events.get(timeout=10)
                    n_text += ev[0] == 'text'
                t0 = time.perf_counter()
                job.cancel.set()
                while True:
                    ev = job.events.get(timeout=10)
                    if ev[0] == 'done':
                        break
                flag_stop.append((time.perf_counter() - t0) * 1e3)
                self.assertEqual(ev[1].finish, 'cancel')
            self.assertLess(max(flag_stop), 30.0, flag_stop)
            close_row = []
            for i in range(5):
                n0 = len(U.log_rows(srv))
                s = U.raw_socket_request(srv, '/v1/completions', {'prompt': [tok.bos_id, 60 + i, 61],
                                                                  'max_tokens': 20000})
                time.sleep(0.3)
                t0 = time.time()
                s.close()
                t_end = t0 + 5
                while time.time() < t_end and len(U.log_rows(srv)) == n0:
                    time.sleep(0.002)
                close_row.append((time.time() - t0) * 1e3)
                self.assertEqual(U.log_rows(srv)[-1].get('finish'), 'cancel')
            print(f'\n[cancel] flag -> stop {min(flag_stop):.0f}-{max(flag_stop):.0f} ms (10 ms passes; spec <= 30); '
                  f'socket close -> log row {min(close_row):.0f}-{max(close_row):.0f} ms (poll {POLL_S * 1e3:.0f} ms '
                  'while non-streaming)')
        finally:
            srv.stop()


# ==================================================================================================== host overhead (proxy)
class TestHostOverheadProxy(unittest.TestCase):
    """NOT the <= 3% gate (that needs the GPU engine, T1). A CPU proxy: per-token wall time of a streamed request
    through the server against the same generation loop called directly, both on the scripted mock (no sleep)."""

    def test_proxy(self):
        tok = common.tokenizer()
        eng = ScriptedEngine(ntok=1)
        srv = U.make_server(eng)
        try:
            msgs = [{'role': 'user', 'content': 'Write it out.'}]
            reply = {'role': 'assistant', 'reasoning_content': 'Plan. ' * 40, 'content': synth.CODE[0] * 6}
            p, r = U.script(eng, msgs, reply, None)
            sess = MockSession(eng, max_ctx=32768)
            best_direct = best_srv = 1e9
            for _ in range(3):
                sess.reset()
                res = run_generation(sess, p, Sampler(SamplingParams(temperature=0.0)), 4096, lambda ids: None)
                best_direct = min(best_direct, res.decode_ms / len(res.gen_ids))
                srv.sess.reset()
                st, h, evs = U.sse_post(srv, '/v1/chat/completions', {'messages': msgs, 'stream': True,
                                                                      'temperature': 0})
                tim = [json.loads(d)['timings'] for _, d in evs[:-1] if 'timings' in json.loads(d)][0]
                best_srv = min(best_srv, tim['predicted_ms'] / tim['predicted_n'])
            print(f'\n[overhead proxy, CPU mock, not the T1 gate] {len(r)} tokens: direct loop '
                  f'{best_direct:.3f} ms/token, through the server (detok + stop + SSE thread) {best_srv:.3f} ms/token')
        finally:
            srv.stop()


if __name__ == '__main__':
    unittest.main()
