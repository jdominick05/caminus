"""T0 gate for the output parser (spec-server.md section 8, step S2), CPU only.

  Parity      toolparse.OutputParser == llama.cpp's parser (tools/qwen_parse_ref.cpp, which runs common/chat* from
              ignite-llama's CPU build) on N_CASES synthesized well-formed outputs (parsegen.py), each under 7
              chunkings: one shot, per character, 3 random, the S1 tokenizer's token pieces, and random cuts. Exact:
              reasoning_content, content, tool names and argument strings byte for byte. At every cut boundary the
              text streamed so far equals llama.cpp's partial parse (reasoning, content, tool names).
  Streaming   deltas concatenate to the final fields (arguments are append-only); the OpenAI accumulation and the
              anthropic SDK's own accumulator (accumulate_event) rebuild the message.
  Truncation  every case cut at random points (max_tokens): no exception, every call's arguments are a valid JSON
              object, deltas still concatenate to the result.
  Malformed   pinned expected values where the parser is lenient (llama.cpp drops these calls or text silently);
              llama.cpp's own result is printed next to ours for the record.
Run:  python -I tests/serve/run_all.py   (or this file alone)"""
import collections
import json
import os
import random
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from parsegen import TOOLS, make_cases  # noqa: E402
from caminus.serve import deltas  # noqa: E402
from caminus.serve.toolparse import (ALL_TYPES, T_ARR, T_BOOL, T_INT, T_NULL, T_NUM, T_OBJ, T_STR,  # noqa: E402
                                    OutputParser, ToolSet, coerce_value, json_kind, parse_text, schema_types)

N_CASES = int(os.environ.get('K3_PARSE_CASES', '600'))
SEED = int(os.environ.get('K3_PARSE_SEED', '1'))


def _parser(c):
    return OutputParser(c['tools'], thinking=c['enable_thinking'], parse_tools=c['tool_choice'] != 'none',
                        parallel=c['parallel'])


def _run(c, chunks):
    p = _parser(c)
    events, snaps, pos = [], {}, 0
    for ch in chunks:
        events += p.feed(ch)
        pos += len(ch)
        snaps[pos] = (p.msg.reasoning_content, p.msg.content, [tc.name for tc in p.msg.tool_calls])
    events += p.finish()
    return p.result(), events, snaps


def _concat(events, n_calls):
    r = ''.join(e[1] for e in events if e[0] == 'reasoning')
    c = ''.join(e[1] for e in events if e[0] == 'content')
    args = collections.defaultdict(str)
    names = {}
    for e in events:
        if e[0] == 'tool_args':
            args[e[1]] += e[2]
        elif e[0] == 'tool_start':
            assert e[1] not in names, 'tool_start twice'
            names[e[1]] = e[2]
    return r, c, [(names.get(i), args[i]) for i in range(n_calls)]


def _token_pieces(text):
    tok = common.tokenizer()
    ids = tok.encode(text, parse_special=True)
    if tok.decode(ids) != text:
        return None
    dec = tok.stream_decoder()
    pieces = [dec.feed(i) for i in ids]
    pieces.append(dec.flush())
    return [p for p in pieces if p]


def _anthropic_rebuild(evs):
    """feed message_start + our block events + message_delta/stop through the anthropic SDK's accumulator"""
    from anthropic.lib.streaming._messages import accumulate_event
    start = {'type': 'message_start', 'message': {
        'id': 'msg_test', 'type': 'message', 'role': 'assistant', 'model': 'k3-coder', 'content': [],
        'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 1, 'output_tokens': 0}}}
    snap, bufs = None, {}
    for e in [start] + evs + [{'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None},
                               'usage': {'output_tokens': 1}}, {'type': 'message_stop'}]:
        snap = accumulate_event(event=e, current_snapshot=snap, json_bufs=bufs)
    return snap


class SchemaTypes(unittest.TestCase):
    def test_shapes(self):
        S = lambda *t: frozenset(t)  # noqa: E731
        root = {'$defs': {'O': {'type': 'object', 'properties': {}}, 'L': {'$ref': '#/$defs/L'}}}
        cases = [
            ({'type': 'string'}, S(T_STR)), ({'type': 'integer'}, S(T_INT)), ({'type': 'number'}, S(T_NUM, T_INT)),
            ({'type': ['string', 'null']}, S(T_STR, T_NULL)), ({'enum': ['a', 'b']}, S(T_STR)),
            ({'enum': [1, 'x', None]}, S(T_INT, T_STR, T_NULL)), ({'const': 2.5}, S(T_NUM)), ({}, ALL_TYPES),
            ({'anyOf': [{'type': 'boolean'}, {'type': 'array'}]}, S(T_BOOL, T_ARR)),
            ({'oneOf': [{'type': 'string'}], 'anyOf': [{'type': 'integer'}]}, S(T_STR)),
            ({'$ref': '#/$defs/O'}, S(T_OBJ)), ({'$ref': '#/$defs/L'}, S()), ({'$ref': '#/nope'}, ALL_TYPES),
            ({'properties': {}}, S(T_OBJ)), ({'items': {}}, S(T_ARR)), ({'format': 'uuid4'}, S(T_STR)),
            ({'format': 'uri'}, ALL_TYPES), ({'allOf': [{'type': ['string', 'integer']}, {'type': 'integer'}]}, S(T_INT)),
            ({'type': 'string', 'allOf': [{'enum': [1]}]}, S(T_INT)), ({'type': ['integer', 'string'], 'enum': ['a']}, S(T_STR)),
        ]
        for schema, want in cases:
            self.assertEqual(schema_types(schema, dict(root, **schema)), want, schema)

    def test_json_kind_and_coerce(self):
        for text, kind in [('1', 'number'), ('-0', 'number'), ('01', None), ('1.', None), ('1e5', 'number'),
                           (' 1', None), ('1 ', None), ('"a\nb"', 'string'), ('[1, 2]', 'array'), ('[1,]', None),
                           ('{"a" : 1}', 'object'), ('{"a":1,}', None), ('NaN', None), ('true', 'bool'),
                           ('nul', None), ('"\\x"', None), ('"\\u00e9"', 'string'), ('[1-2]', None)]:
            self.assertEqual(json_kind(text), kind, text)
        S = lambda *t: frozenset(t)  # noqa: E731
        self.assertEqual(coerce_value('[1,2]', S(T_STR, T_NUM)), ('"[1,2]"', False))
        self.assertEqual(coerce_value('3', S(T_STR, T_INT)), ('3', False))
        self.assertEqual(coerce_value('"q"', S(T_STR, T_NULL)), ('"\\"q\\""', False))
        self.assertEqual(coerce_value('"q"', S(T_INT)), ('"q"', False))
        self.assertEqual(coerce_value(' 30 ', S(T_INT)), ('30', True))
        self.assertEqual(coerce_value("{'a': True}", S(T_OBJ)), ('{"a": true}', True))
        self.assertEqual(coerce_value('abc', S(T_INT)), ('"abc"', True))


class Parity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if common.qwen_parse_ref_binary() is None:
            raise unittest.SkipTest('tools/qwen_parse_ref could not be built')
        cls.cases = make_cases(N_CASES, seed=SEED)
        rng = random.Random(SEED + 1000)
        for c in cls.cases:
            t = c['text']
            c['cuts'] = sorted(set(rng.randint(0, len(t)) for _ in range(min(len(t), 16))))
            c['partial_at'] = [len(t[:x].encode('utf-8')) for x in c['cuts']]
        t0 = time.time()
        cls.refs = common.qwen_parse_ref([{k: v for k, v in c.items() if k != 'cuts'} for c in cls.cases])
        cls.ref_s = time.time() - t0

    def test_parity_all_chunkings(self):
        rng = random.Random(SEED + 2000)
        st = collections.Counter()
        cover = collections.Counter()
        fails = []
        for ci, (c, r) in enumerate(zip(self.cases, self.refs)):
            self.assertIsNone(r.get('setup_error'), r)
            self.assertIsNone(r['error'], (r['error'], c['text']))
            ref = r['final']
            t = c['text']
            cover['thinking' if c['enable_thinking'] else 'no_thinking'] += 1
            cover['tool_choice_' + c['tool_choice'] if c['tools'] else 'no_tools'] += 1
            cover['parallel_false'] += not c['parallel']
            cover[f'calls_{len(ref["tool_calls"])}'] += 1
            cover['with_reasoning'] += bool(ref['reasoning_content'])
            cover['with_content'] += bool(ref['content'])
            chunkings = {'one': [t], 'char': list(t)}
            for k in range(3):
                ch, i = [], 0
                while i < len(t):
                    n = rng.randint(1, 12)
                    ch.append(t[i:i + n])
                    i += n
                chunkings[f'rand{k}'] = ch
            pieces = _token_pieces(t)
            if pieces is not None:
                chunkings['tokens'] = pieces
            else:
                st['token_roundtrip_skipped'] += 1
            ch, prev = [], 0
            for x in c['cuts'] + [len(t)]:
                ch.append(t[prev:x])
                prev = x
            chunkings['cuts'] = ch
            for name, chunks in chunkings.items():
                msg, evs, snaps = _run(c, chunks)
                st['runs'] += 1
                mine = msg.as_dict()
                if mine != ref:
                    fails.append((ci, name, 'final', mine, ref))
                    continue
                st['final_equal'] += 1
                rr, cc, calls = _concat(evs, len(msg.tool_calls))
                if (rr, cc, calls) != (msg.reasoning_content, msg.content,
                                       [(x.name, x.arguments) for x in msg.tool_calls]):
                    fails.append((ci, name, 'deltas', (rr, cc, calls), mine))
                if name == 'cuts':
                    for x, part in zip(c['cuts'], r['partials']):
                        self.assertNotIn('error', part)
                        got = snaps.get(x)
                        if got is None:
                            continue
                        st['partials'] += 1
                        want = (part['reasoning_content'], part['content'], [tc['name'] for tc in part['tool_calls']])
                        if got != want:
                            fails.append((ci, name, f'partial@{x}', got, want))
                        else:
                            st['partials_equal'] += 1
            ts = ToolSet(c['tools'])
            for tc in ref['tool_calls']:
                st['calls'] += 1
                args = json.loads(tc['arguments'], strict=False)
                st['args_valid_json'] += isinstance(args, dict)
                for k, v in args.items():
                    types = ts.types(tc['name'], k)
                    kind = 'string' if types == frozenset((T_STR,)) else 'mixed' if T_STR in types else 'typed'
                    cover[f'param_{kind}'] += 1
                    if kind != 'string':
                        cover[f'param_{kind}_as_{type(v).__name__}'] += 1
        print(f'\n[parser parity] cases={len(self.cases)} ref={self.ref_s:.1f}s {dict(st)}\n[parser coverage] '
              f'{dict(cover)}', file=sys.stderr)
        for f in fails[:5]:
            print('FAIL', f, '\n  text:', repr(self.cases[f[0]]['text'][:400]), file=sys.stderr)
        self.assertEqual(fails, [])
        self.assertGreaterEqual(len(self.cases), 500)
        self.assertEqual(st['final_equal'], st['runs'])

    def test_adapters(self):
        """OpenAI deltas and the anthropic SDK's accumulator rebuild the parsed message (well-formed cases)"""
        rng = random.Random(SEED + 3000)
        for c in self.cases[:200]:
            _check_adapters(self, c, c['text'], rng)


def _check_adapters(tc, c, text, rng):
    ch, i = [], 0
    while i < len(text):
        k = rng.randint(1, 8)
        ch.append(text[i:i + k])
        i += k
    p = _parser(c)
    oa, an = deltas.OpenAIStream(), deltas.AnthropicStream()
    od, ae = [], []
    for piece in ch:
        ev = p.feed(piece)
        od += oa.deltas(ev)
        ae += an.events(ev)
    ev = p.finish()
    od += oa.deltas(ev)
    ae += an.events(ev) + an.close()
    msg = p.result()
    acc = deltas.apply_openai_deltas(od)
    tc.assertEqual(acc['reasoning_content'], msg.reasoning_content)
    tc.assertEqual(acc['content'], msg.content)
    tc.assertEqual([(x['id'], x['name'], x['arguments']) for x in acc['tool_calls']],
                   [(oa.ids[i], x.name, x.arguments) for i, x in enumerate(msg.tool_calls)])
    om = deltas.openai_message(msg, ids=oa.ids)
    tc.assertEqual(om.get('reasoning_content', ''), msg.reasoning_content)
    tc.assertEqual([x['id'] for x in om.get('tool_calls', [])], [oa.ids[i] for i, _ in msg.usable_calls()])
    # anthropic: indices in order, every block closed, the SDK's rebuild equals the non-streaming blocks
    starts = [e for e in ae if e['type'] == 'content_block_start']
    stops = [e for e in ae if e['type'] == 'content_block_stop']
    tc.assertEqual([e['index'] for e in starts], list(range(len(starts))))
    tc.assertEqual([e['index'] for e in stops], list(range(len(starts))))
    snap = _anthropic_rebuild(ae)
    got = [b.to_dict() for b in snap.content]
    for b in got:
        b.pop('citations', None)
    tc.assertEqual(got, deltas.anthropic_content(msg, ids=an.ids, keep_cut=True), text)


class Truncation(unittest.TestCase):
    def test_cut_anywhere(self):
        """max_tokens can stop the stream anywhere: the result stays well-formed"""
        rng = random.Random(SEED + 4000)
        n = 0
        for c in make_cases(N_CASES, seed=SEED + 1):
            t = c['text']
            for cut in sorted(set(rng.randint(0, len(t)) for _ in range(6))):
                text = t[:cut]
                ch, i = [], 0
                while i < len(text):
                    k = rng.randint(1, 10)
                    ch.append(text[i:i + k])
                    i += k
                msg, evs, _ = _run(c, ch)
                rr, cc, calls = _concat(evs, len(msg.tool_calls))
                self.assertEqual((rr, cc), (msg.reasoning_content, msg.content))
                self.assertEqual(calls, [(x.name, x.arguments) for x in msg.tool_calls])
                for x in msg.tool_calls:
                    self.assertIsInstance(json.loads(x.arguments, strict=False), dict, (text, x))
                one = parse_text(text, c['tools'], c['enable_thinking'], c['tool_choice'] != 'none', c['parallel'])
                self.assertEqual(one.as_dict(), msg.as_dict())
                if n % 5 == 0:
                    _check_adapters(self, c, text, rng)
                n += 1
        print(f'\n[parser truncation] {n} cut streams', file=sys.stderr)


def _status(call):
    return 'complete' if call.complete else 'cut' if call.cut else 'open'


TC = '<tool_call>\n<function={n}>\n{p}</function>\n</tool_call>'


def _p(name, value):
    return f'<parameter={name}>\n{value}\n</parameter>\n'


# (name, text, kwargs, expected {reasoning_content, content, tool_calls: [(name, arguments, complete)]})
MALFORMED = [
    ('undeclared tool kept', 'r</think>' + TC.format(n='nope', p=_p('a', '1')), {},
     ('r', '', [('nope', '{"a":1}', 'complete')])),
    ('alias: __ flattened', 'r</think>' + TC.format(n='mcp_web_fetch_page', p=_p('url', 'http://x')), {},
     ('r', '', [('mcp__web__fetch_page', '{"url":"http://x"}', 'complete')])),
    ('alias: case', 'r</think>' + TC.format(n='Read_File', p=_p('path', 'a')), {},
     ('r', '', [('read_file', '{"path":"a"}', 'complete')])),
    ('optional before required', 'r</think>' + TC.format(n='read_file', p=_p('limit', '5') + _p('path', 'a')), {},
     ('r', '', [('read_file', '{"limit":5,"path":"a"}', 'complete')])),
    ('typed value with spaces', 'r</think>' + TC.format(n='read_file', p=_p('path', 'a') + _p('limit', ' 30 ')), {},
     ('r', '', [('read_file', '{"path":"a","limit":30}', 'complete')])),
    ('typed value not JSON', 'r</think>' + TC.format(n='read_file', p=_p('path', 'a') + _p('limit', 'all')), {},
     ('r', '', [('read_file', '{"path":"a","limit":"all"}', 'complete')])),
    ('python literals', 'r</think>' + TC.format(n='run_shell', p=_p('command', 'ls') + _p('background', 'True')
                                                + _p('env', "{'A': 'b'}")), {},
     ('r', '', [('run_shell', '{"command":"ls","background":true,"env":{"A": "b"}}', 'complete')])),
    ('undeclared parameter', 'r</think>' + TC.format(n='read_file', p=_p('path', 'a') + _p('mode', '[1]')
                                                     + _p('why', 'because')), {},
     ('r', '', [('read_file', '{"path":"a","mode":[1],"why":"because"}', 'complete')])),
    ('no newlines around tags', 'r</think><tool_call><function=read_file><parameter=path>a.txt</parameter>'
                                '<parameter=limit>3</parameter></function></tool_call>', {},
     ('r', '', [('read_file', '{"path":"a.txt","limit":3}', 'complete')])),
    ('text after the calls', 'r</think>' + TC.format(n='read_file', p=_p('path', 'a')) + '\nDone, reading.', {},
     ('r', 'Done, reading.', [('read_file', '{"path":"a"}', 'complete')])),
    ('text between calls', 'r</think>' + TC.format(n='read_file', p=_p('path', 'a')) + '\nand\n'
     + TC.format(n='read_file', p=_p('path', 'b')), {},
     ('r', 'and\n', [('read_file', '{"path":"a"}', 'complete'), ('read_file', '{"path":"b"}', 'complete')])),
    ('content ends in "<"', 'r</think>\n\nx < y and y <', {}, ('r', 'x < y and y <', [])),
    ('reasoning cut in a tag', 'thinking about </thi', {}, ('thinking about </thi', '', [])),
    ('cut in a string value', 'r</think>' + '<tool_call>\n<function=run_shell>\n<parameter=command>\nrm -rf bu',
     {}, ('r', '', [('run_shell', '{"command":"rm -rf bu"}', 'cut')])),
    ('cut after a string value', 'r</think>' + '<tool_call>\n<function=run_shell>\n<parameter=command>\nls\n</param',
     {}, ('r', '', [('run_shell', '{"command":"ls"}', 'cut')])),
    ('cut in a typed value', 'r</think>' + '<tool_call>\n<function=run_shell>\n' + _p('command', 'ls')
     + '<parameter=env>\n{"A": "', {}, ('r', '', [('run_shell', '{"command":"ls"}', 'cut')])),
    ('cut in the name', 'r</think>ok <tool_call>\n<function=run_sh', {}, ('r', 'ok ', [])),
    ('cut right after <tool_call>', 'r</think>ok <tool_call>', {}, ('r', 'ok ', [])),
    ('Hermes JSON call', 'r</think><tool_call>\n{"name": "read_file", "arguments": {"path": "a"}}\n</tool_call>', {},
     ('r', '', [('read_file', '{"path": "a"}', 'complete')])),
    ('<tool_call> then prose', 'r</think>Use <tool_call> tags to call tools.', {},
     ('r', 'Use <tool_call> tags to call tools.', [])),
    ('missing </function>', 'r</think><tool_call>\n<function=read_file>\n' + _p('path', 'a') + '</tool_call>', {},
     ('r', '', [('read_file', '{"path":"a"}', 'complete')])),
    ('missing </tool_call> between calls', 'r</think><tool_call>\n<function=read_file>\n' + _p('path', 'a')
     + '</function>\n' + TC.format(n='read_file', p=_p('path', 'b')), {},
     ('r', '', [('read_file', '{"path":"a"}', 'complete'), ('read_file', '{"path":"b"}', 'complete')])),
    ('parallel off: first call only', 'r</think>' + TC.format(n='read_file', p=_p('path', 'a'))
     + TC.format(n='read_file', p=_p('path', 'b')), {'parallel': False},
     ('r', '', [('read_file', '{"path":"a"}', 'complete')])),
    ('</parameter> at the end', 'r</think><tool_call>\n<function=read_file>\n<parameter=path>\na\n</parameter>', {},
     ('r', '', [('read_file', '{"path":"a"}', 'open')])),
    ('no tools: tags are content', 'r</think>' + TC.format(n='read_file', p=_p('path', 'a')), {'tools': []},
     ('r', TC.format(n='read_file', p=_p('path', 'a')), [])),
    ('tool_choice none', 'r</think>see ' + TC.format(n='read_file', p=_p('path', 'a')), {'parse_tools': False},
     ('r', 'see ' + TC.format(n='read_file', p=_p('path', 'a')), [])),
    ('thinking off', '</think>\n\nhi', {'thinking': False}, ('', '</think>\n\nhi', [])),
]


class Malformed(unittest.TestCase):
    def test_pinned(self):
        rows = []
        refs = None
        if common.qwen_parse_ref_binary() is not None:
            refs = common.qwen_parse_ref([{
                'tools': kw.get('tools', TOOLS), 'enable_thinking': kw.get('thinking', True),
                'tool_choice': 'none' if kw.get('parse_tools') is False else 'auto',
                'parallel': kw.get('parallel', True), 'text': text} for _, text, kw, _ in MALFORMED])
        rng = random.Random(5)
        for k, (name, text, kw, (want_r, want_c, want_calls)) in enumerate(MALFORMED):
            c = {'tools': kw.get('tools', TOOLS), 'enable_thinking': kw.get('thinking', True),
                 'tool_choice': 'none' if kw.get('parse_tools') is False else 'auto', 'parallel': kw.get('parallel', True)}
            rand, i = [], 0
            while i < len(text):
                n = rng.randint(1, 9)
                rand.append(text[i:i + n])
                i += n
            for chunks in ([text], list(text), [text[i:i + 3] for i in range(0, len(text), 3)], rand):
                msg, evs, _ = _run(c, chunks)
                got = (msg.reasoning_content, msg.content, [(x.name, x.arguments, _status(x)) for x in msg.tool_calls])
                self.assertEqual(got, (want_r, want_c, want_calls), name)
                self.assertEqual(_concat(evs, len(msg.tool_calls))[:2], (msg.reasoning_content, msg.content))
            _check_adapters(self, c, text, rng)
            if refs is not None:
                ref = refs[k]['final']
                same = ref is not None and (ref['reasoning_content'], ref['content'],
                                            [(x['name'], x['arguments']) for x in ref['tool_calls']]) == \
                    (want_r, want_c, [(x[0], x[1]) for x in want_calls])
                rows.append(f'  {"same" if same else "DIFF"}  {name}: llama.cpp -> '
                            f'{json.dumps(ref, ensure_ascii=False) if ref else refs[k]["error"]}')
        if rows:
            print('\n[parser malformed] ours pinned; llama.cpp for the record:\n' + '\n'.join(rows), file=sys.stderr)


class StopReasons(unittest.TestCase):
    def test_cut_call_is_not_a_call(self):
        head = 'r</think><tool_call>\n<function=run_shell>\n<parameter=command>\nrm -rf build'
        for text, stop, oa, an, n in [
                (head, 'eos', 'stop', 'end_turn', 0),                               # EOS inside the value
                (head, 'stop', 'stop', 'stop_sequence', 0),                         # a stop string inside it
                (head, 'length', 'length', 'max_tokens', 0),                        # max_tokens inside it
                (head + '\n</parameter>\n', 'eos', 'tool_calls', 'tool_use', 1),   # EOS after the value
                (head + '\n</parameter>\n</function>\n</tool_call>', 'eos', 'tool_calls', 'tool_use', 1)]:
            msg = parse_text(text, TOOLS)
            self.assertEqual(len(msg.tool_calls), 1)
            self.assertEqual(deltas.openai_finish_reason(msg, stop), oa, (text, stop))
            self.assertEqual(deltas.anthropic_stop_reason(msg, stop), an, (text, stop))
            self.assertEqual(len(deltas.openai_message(msg).get('tool_calls', [])), n)
            self.assertEqual(sum(b['type'] == 'tool_use' for b in deltas.anthropic_content(msg)), n)
            if n:
                self.assertEqual(deltas.anthropic_content(msg)[-1]['input'], {'command': 'rm -rf build'})


class Basics(unittest.TestCase):
    def test_no_partial_tags_in_deltas(self):
        p = OutputParser(TOOLS)
        out = []
        for ch in 'abc</th' + 'ink>\n\nhello <tool_' + 'call>\n<function=read_file>\n<parameter=path>\nx\n</parameter>\n':
            out += p.feed(ch)
        texts = [e[1] for e in out if e[0] in ('reasoning', 'content')]
        self.assertEqual(texts, ['a', 'b', 'c', 'h', 'e', 'l', 'l', 'o', ' '])
        self.assertEqual([e[0] for e in out if e[0].startswith('tool')][:1], ['tool_start'])
        out += p.feed('</function>\n</tool_call>')
        self.assertEqual(out[-1], ('tool_end', 0))
        self.assertEqual(p.held, 0)

    def test_toolset_forms(self):
        ts = ToolSet([{'name': 'a.b', 'input_schema': {'type': 'object', 'properties': {'x': {'type': 'integer'}}}},
                      {'type': 'function', 'function': {'name': 'c', 'parameters': {'type': 'object'}}}])
        self.assertEqual(ts.types('a.b', 'x'), frozenset((T_INT,)))
        for alias in ('a__b', 'a=b', 'a_b', 'b', ' "a.b" '):
            self.assertEqual(ts.resolve(alias), 'a.b', alias)
        self.assertEqual(ts.resolve('zzz'), 'zzz')
        self.assertEqual(ts.types('c', 'y'), ALL_TYPES)

    def test_feed_cost(self):
        """per-token parse cost (host share of a ~10 ms token)"""
        cases = make_cases(200, seed=9)
        pieces = [(c, _token_pieces(c['text'])) for c in cases]
        n, t0 = 0, time.perf_counter()
        for c, ps in pieces:
            if ps is None:
                continue
            p = _parser(c)
            for x in ps:
                p.feed(x)
            p.finish()
            n += len(ps)
        us = (time.perf_counter() - t0) / max(n, 1) * 1e6
        print(f'\n[parser cost] {us:.1f} us per token piece over {n} pieces', file=sys.stderr)
        self.assertLess(us, 100.0)


if __name__ == '__main__':
    unittest.main()
