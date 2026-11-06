"""Synthesized model outputs for the output-parser gate (spec-server.md T0, step S2). Everything is made up.

make_cases(n, seed) returns well-formed cases: outputs llama.cpp's grammar admits (llama-server samples tool calls
under a lazy grammar), the zone where toolparse.OutputParser must equal llama.cpp's parser exactly. Each case is
{'tools', 'enable_thinking', 'tool_choice', 'parallel', 'text'}; 'text' is the model's output after the
generation prompt.

What the generator keeps out of the zone (the parser is lenient there by design; see Malformed in
test_toolparse.py): text ending in a prefix of '</think>' or '<tool_call>' at the end of the stream, and string
values with '</parameter>' followed (after whitespace) by '<parameter=', '</function>' or '</tool_call>'."""
import json
import random
import re

from synth import CODE, PROSE, SHELL

TOOLS = [
    {'type': 'function', 'function': {'name': 'run_shell', 'description': 'Run a command.', 'parameters': {
        'type': 'object', 'properties': {
            'command': {'type': 'string'}, 'timeout_s': {'type': 'number'}, 'cwd': {'type': 'string'},
            'env': {'type': 'object', 'additionalProperties': {'type': 'string'}}, 'background': {'type': 'boolean'}},
        'required': ['command']}}},
    {'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a file.', 'parameters': {
        'type': 'object', 'properties': {
            'path': {'type': 'string'}, 'offset': {'type': 'integer'}, 'limit': {'type': 'integer'}},
        'required': ['path']}}},
    {'type': 'function', 'function': {'name': 'edit_file', 'description': 'Edit a file.', 'parameters': {
        'type': 'object', 'properties': {
            'path': {'type': 'string'}, 'old_str': {'type': 'string'}, 'new_str': {'type': 'string'},
            'replace_all': {'type': 'boolean'}},
        'required': ['path', 'old_str', 'new_str']}}},
    {'type': 'function', 'function': {'name': 'search', 'description': 'Search.', 'parameters': {
        'type': 'object', '$defs': {'Opts': {'type': 'object', 'properties': {'case': {'type': 'boolean'}}}},
        'properties': {
            'pattern': {'type': 'string'},
            'glob': {'type': ['string', 'null']},
            'max_results': {'type': 'integer', 'minimum': 1},
            'kinds': {'type': 'array', 'items': {'type': 'string'}},
            'mode': {'enum': ['fast', 'full']},
            'level': {'enum': [1, 2, 'max']},
            'extra': {},
            'opts': {'$ref': '#/$defs/Opts'},
            'limit': {'oneOf': [{'type': 'string'}, {'type': 'integer'}]},
            'score': {'anyOf': [{'type': 'number'}, {'type': 'null'}]},
            'tag': {'const': 'v1'},
            'when': {'format': 'date-time'},
            'meta': {'properties': {'a': {'type': 'integer'}}},
            'list2': {'items': {'type': 'integer'}},
            'nullable_obj': {'type': ['object', 'null']},
        },
        'required': ['pattern']}}},
    {'type': 'function', 'function': {'name': 'mcp__web__fetch_page', 'description': 'Fetch.', 'parameters': {
        'type': 'object', 'properties': {'url': {'type': 'string', 'format': 'uri'}, 'max_bytes': {'type': 'integer'}},
        'required': ['url']}}},
    {'type': 'function', 'function': {'name': 'list_todos', 'description': 'No arguments.', 'parameters': {
        'type': 'object', 'properties': {}}}},
]

LOOKALIKES = ['<think>', '</thinking>', '<tool_', '<tool_c', '</th', '</ think>', '<tool call>', '< tool_call>',
              '<', '<<', '</', '<function=x>', '<parameter=y>', '</tool_call>', '</function>', '</parameter>',
              '<|im_end|>', '<br/>', 'a < b > c']
UNICODE = ['héllo wörld', '世界の検索', '🎉🚀', 'Ωmega ≤ ∞', 'naïve café', ' nbsp', 'العربية', '🧑‍💻']
STRING_VALUES = (CODE + SHELL + [p for p in PROSE] + UNICODE + [
    '', ' ', '\n', 'ls -la', '    indented\n        twice\n', 'trailing newline\n', '\nleading newline',
    'two\n\ntrailing\n\n', 'quote " and backslash \\ and both \\"', 'tab\there', 'ctrl \x01\x07\x1f end',
    'del \x7f char', 'x\n</param', 'a </parameter> b', '</parameter>', '<parameter=evil>', '<tool_call>\n<function=x>',
    '</function>', '</tool_call> then text', 'json-ish {"a": 1}', '[1, 2]', 'true', 'null', '42', '-3.5e2',
    'line\r\nwindows\r\n', '  line sep  ', 'end with lt <', 'end with tag </think>'])


def _bad_value(v):
    if '\n</parameter>\n' in v:
        return True
    for m in re.finditer(re.escape('</parameter>'), v):
        r = v[m.end():].lstrip(' \t\n\v\f\r')
        if r.startswith(('<parameter=', '</function>', '</tool_call>')):
            return True
        if m.end() == len(v) or not r:
            return True                 # value ends in '</parameter>' + ws: the closing tag would follow it
    return False


def _rand_text(rng, n_min=0, n_max=80, extra=()):
    alphabet = 'abcdefghij klmnop\n\t<>/="\'\\{}[]:,.!?-_' + 'é世🎉'
    pool = list(LOOKALIKES) + list(extra)
    out = []
    while len(''.join(out)) < rng.randint(n_min, n_max):
        r = rng.random()
        if r < 0.15:
            out.append(rng.choice(pool))
        elif r < 0.3:
            out.append(rng.choice(PROSE + UNICODE))
        else:
            out.append(''.join(rng.choice(alphabet) for _ in range(rng.randint(1, 12))))
    return ''.join(out)


def _rand_string_value(rng):
    while True:
        v = rng.choice(STRING_VALUES) if rng.random() < 0.6 else _rand_text(rng, 0, 120)
        if not _bad_value(v):
            return v


def _rand_json(rng, depth=0):
    r = rng.random()
    if depth > 2 or r < 0.5:
        return rng.choice([0, 1, -7, 42, 3.25, -0.5, 1e21, 2.5e-7, True, False, None, 'str', 'with "quote"',
                           'nl\nin', 'é世🎉', '</parameter>', '\\path\\x', ''])
    if r < 0.75:
        return [_rand_json(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    return {rng.choice(['a', 'b', 'key', 'ü', 'x y', '<t>']) + str(i): _rand_json(rng, depth + 1)
            for i in range(rng.randint(0, 3))}


def _dump(rng, v):
    style = rng.randint(0, 3)
    if style == 0:
        return json.dumps(v, ensure_ascii=False)
    if style == 1:
        return json.dumps(v, ensure_ascii=False, separators=(',', ':'))
    if style == 2:
        return json.dumps(v, ensure_ascii=True)
    return json.dumps(v, ensure_ascii=False, indent=2)


def _number_text(rng, integer):
    if integer:
        return rng.choice(['0', '1', '-1', '30', '120', '9007199254740993', '-0'])
    return rng.choice(['0', '1.5', '-2', '3.0', '1e5', '1E+5', '2.5e-3', '-0.0', '100', '6.02e23', '12'])


def _value_for(rng, schema_types):
    """(text, ok): a value the grammar admits for these types"""
    from caminus.serve.toolparse import T_ARR, T_BOOL, T_INT, T_NULL, T_NUM, T_OBJ, T_STR
    if schema_types == frozenset((T_STR,)):
        return _rand_string_value(rng)
    choices = []
    if T_OBJ in schema_types:
        choices.append(lambda: _dump(rng, {k: _rand_json(rng, 1) for k in rng.sample(['a', 'b', 'c', 'é'], rng.randint(0, 3))}))
    if T_ARR in schema_types:
        choices.append(lambda: _dump(rng, [_rand_json(rng, 1) for _ in range(rng.randint(0, 4))]))
    if T_NUM in schema_types:
        choices.append(lambda: _number_text(rng, False))
    if T_INT in schema_types:
        choices.append(lambda: _number_text(rng, True))
    if T_BOOL in schema_types:
        choices.append(lambda: rng.choice(['true', 'false']))
    if T_NULL in schema_types:
        choices.append(lambda: 'null')
    if T_STR in schema_types:
        choices.append(lambda: _rand_string_value(rng))
        choices.append(lambda: rng.choice(['[1,2]', '{"a":1}', 'true', 'null', '7', '"quoted"', '1.5x']))
    else:
        # no string type: any JSON value is admitted by the parser (the grammar narrows it; llama.cpp's parser
        # does not), so strings and other kinds appear too
        choices.append(lambda: _dump(rng, _rand_json(rng)))
    return rng.choice(choices)()


def _tool_call(rng, tool):
    from caminus.serve.toolparse import ToolSet
    fn = tool['function']
    ts = ToolSet([tool])
    props = fn['parameters'].get('properties', {})
    req = [p for p in fn['parameters'].get('required', []) if p in props]
    opt = [p for p in props if p not in req]
    order = rng.sample(req, len(req)) + rng.sample(opt, rng.randint(0, min(len(opt), 8)))
    parts = ['<tool_call>\n<function=' + fn['name'] + '>\n']
    for p in order:
        parts.append('<parameter=' + p + '>\n' + _value_for(rng, ts.types(fn['name'], p)) + '\n</parameter>\n')
    parts.append('</function>\n</tool_call>')
    return ''.join(parts)


def _safe_tail(s, delims):
    """s with any trailing proper prefix of a delimiter removed (such tails are out of the parity zone)"""
    changed = True
    while changed and s:
        changed = False
        for d in delims:
            for k in range(min(len(d) - 1, len(s)), 0, -1):
                if s.endswith(d[:k]):
                    s = s[:-k] + 'z'
                    changed = True
                    break
    return s


def _no(s, *words):
    for w in words:
        s = s.replace(w, w[:2] + '_' + w[2:])
    return s


def make_case(rng):
    r = rng.random()
    if r < 0.12:
        tools, tool_choice = [], 'auto'
    elif r < 0.2:
        tools, tool_choice = rng.sample(TOOLS, rng.randint(1, 3)), 'none'
    else:
        tools, tool_choice = rng.sample(TOOLS, rng.randint(1, len(TOOLS))), rng.choice(['auto', 'auto', 'required'])
    thinking = rng.random() < 0.75
    parallel = rng.random() < 0.8
    parse_tools = bool(tools) and tool_choice != 'none'
    parts = []
    ended_in_reasoning = False
    if thinking:
        lead = rng.choice(['', '', '\n', '  ', '\n\n'])
        reasoning = _no(_rand_text(rng, 0, 200), '</think>', '<tool_call>')
        if rng.random() < 0.1:
            reasoning = rng.choice(['', ' ', '\n\n'])
        end = rng.random()
        if end < 0.08:
            parts.append(lead + _safe_tail(reasoning, ('</think>', '<tool_call>')))   # stream ends while thinking
            ended_in_reasoning = True
        elif end < 0.18 and parse_tools:
            parts.append(lead + reasoning)                                           # a call ends the reasoning
        else:
            parts.append(lead + reasoning + rng.choice(['\n</think>\n\n', '</think>', '\n</think>', '</think>\n']))
    if not ended_in_reasoning:
        calls = []
        if parse_tools and rng.random() < 0.8:
            n = rng.randint(1, 3) if parallel else 1
            calls = [_tool_call(rng, rng.choice(tools)) for _ in range(n)]
        content = ''
        if rng.random() < 0.6:
            content = _rand_text(rng, 1, 150, extra=['<tool_call>\n<function=read_file>\n'] if not parse_tools else [])
            if parse_tools:
                content = _no(content, '<tool_call>')
            if parts and parts[-1].endswith('<tool_call>'):
                pass
            if not thinking and not parts:
                content = rng.choice(['', '\n', ' ']) + content
        if parts and not parts[-1].endswith(('</think>', '\n', ' ')) and thinking and not calls:
            pass
        if thinking and parts and not parts[-1].rstrip(' \n').endswith('</think>'):
            content = ''               # the reasoning was ended by '<tool_call>': no content before the call
            if not calls:
                calls = [_tool_call(rng, rng.choice(tools))]
        if calls:
            parts.append(content + rng.choice(['', '\n', '\n\n']))
            seps = ['', '\n', '\n\n', ' ', '\n \n']
            parts.append(calls[0] + ''.join(rng.choice(seps) + c for c in calls[1:]) + rng.choice(['', '\n', ' \n']))
        else:
            parts.append(_safe_tail(content, ('<tool_call>',)) if parse_tools else content)
    text = ''.join(parts)
    return {'tools': tools, 'enable_thinking': thinking, 'tool_choice': tool_choice, 'parallel': parallel,
            'text': text}


def make_cases(n, seed=0):
    rng = random.Random(seed)
    return [make_case(rng) for _ in range(n)]
