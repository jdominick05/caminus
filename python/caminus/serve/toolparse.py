# The tool-name alias rule (ToolSet.resolve: a declared `ns.member` tool also answers to `ns__member`, `ns=member`,
# `ns_member` and a unique bare `member`; a declared `a__b` tool to `a_b`; an alias only when unambiguous, an exact
# name always first) follows the scheme of Strata (serve/frontend.py, OutputParser.__init__). The code here is
# written from scratch and line-reviewed; Strata is distributed under the following license:
#
# MIT License
#
# Copyright (c) 2026 Niko1221 and the Strata contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Streaming output parser for the K3 server (spec-server.md section 3, step S2).

Splits the model's text into reasoning, content and Qwen XML tool calls

    <tool_call>\\n<function=NAME>\\n<parameter=P>\\nVALUE\\n</parameter>\\n...</function>\\n</tool_call>

incrementally: feed() takes any piece of text (a token's text, a character, a random chunk) and returns events;
finish() ends the stream. The events never contain part of a tag: text that may still become a tag is held back
until it is decided. Events:

    ('reasoning', text)        reasoning_content delta
    ('content', text)          content delta
    ('tool_start', i, name)    tool call i opened, its name is complete
    ('tool_args', i, text)     a piece of call i's arguments JSON; the pieces concatenate to the final string
    ('tool_end', i)            call i is complete (or closed at the end of the stream)

REFERENCE. The rules for well-formed output are llama.cpp's (ignite-llama common/parsers/qwen3-coder.cpp, the PEG
parser it builds, and common_chat_peg_mapper in common/chat-peg-parser.cpp), gated by tests/serve/test_toolparse.py
against tools/reference/qwen_parse_ref.cpp (llama.cpp's own parser) on >= 500 synthesized outputs under random chunkings:
  * the generation prompt ends '<think>\\n', so a thinking turn starts in the reasoning; with enable_thinking=false it
    starts in the content;
  * reasoning: leading whitespace dropped, trailing kept; it ends at '</think>' (consumed) or at '<tool_call>' (not
    consumed, with or without tools); whitespace-only reasoning is empty;
  * content: leading whitespace dropped, trailing kept; with tools (and tool_choice != none) it ends at
    '<tool_call>'; without, '<tool_call>' is plain content;
  * arguments: '{' + '"name":' VALUE joined by ',' + '}', keys and string values JSON-escaped (ensure_ascii off);
    a parameter value is everything up to '\\n</parameter>\\n'; its JSON schema types decide its form
    (value_types, common/json-schema.cpp; ported in schema_types):
      - string only: the raw text as a JSON string;
      - no string: the raw text if it is one JSON value (any kind), as written;
      - string and others: the raw text if it is a JSON value of one of the other kinds (object, array, number,
        boolean, null), else a JSON string;
  * calls follow each other with only whitespace between; with parallel_tool_calls=false only the first counts.

LENIENCY (K3 has no grammar sampler, so the model can write what llama-server's lazy grammar would forbid; where
llama.cpp drops the call or the rest of the output silently, this parser keeps it; each case is pinned by a test in
tests/serve/test_toolparse.py, class Malformed):
  * an undeclared tool name is kept as written, after alias resolution (ToolSet.resolve);
  * parameters in any order; an undeclared parameter is typed as 'any';
  * a typed value that is not JSON: surrounding whitespace dropped, Python literals (True, 'x') converted, else it
    becomes a JSON string, so the arguments are always valid JSON;
  * whitespace instead of the exact newlines around the tags; '</parameter>' without its newlines when
    '<parameter=', '</function>' or '</tool_call>' follows; a missing '</function>' or '</tool_call>';
  * '<tool_call>' followed by a JSON object {"name": ..., "arguments": ...} (Hermes form) is a call;
  * '<tool_call>' followed by anything else is content, as is text after the last call;
  * at finish(): text held back as a possible tag prefix is emitted (llama.cpp drops it); a call cut off by the end
    of the stream (max_tokens, a stop string, EOS) is closed: an open string value gets its quote, a typed value that
    was not complete is dropped, the braces are closed, and the call is marked complete=False, plus cut=True when the
    end fell inside a value (its arguments are then not safe to run; deltas.py leaves cut calls out of the
    non-streaming bodies and of the stop reason). A call whose name was not complete is dropped. The deltas already
    streamed are never taken back: what a streaming response does with a cut call is the server's decision (S4).
"""
import json
import re

WS = ' \t\n\v\f\r'           # C isspace(): llama.cpp's space() and trims
THINK_END = '</think>'
TOOL_OPEN = '<tool_call>'
TOOL_CLOSE = '</tool_call>'
FUNC_OPEN = '<function='
FUNC_CLOSE = '</function>'
PARAM_OPEN = '<parameter='
PARAM_CLOSE = '</parameter>'
MAX_NAME = 256                # a function or parameter name longer than this is not a tag

T_NULL, T_BOOL, T_INT, T_NUM, T_STR, T_ARR, T_OBJ = 'null', 'boolean', 'integer', 'number', 'string', 'array', 'object'
ALL_TYPES = frozenset((T_NULL, T_BOOL, T_INT, T_NUM, T_STR, T_ARR, T_OBJ))
_KIND_OK = {T_OBJ: 'object', T_ARR: 'array', T_NUM: 'number', T_INT: 'number', T_BOOL: 'bool', T_NULL: 'null'}


# ---------------------------------------------------------------------------------------------------- JSON schema
def _json_type(v):
    if v is None:
        return T_NULL
    if isinstance(v, bool):
        return T_BOOL
    if isinstance(v, int):
        return T_INT
    if isinstance(v, float):
        return T_NUM
    if isinstance(v, str):
        return T_STR
    if isinstance(v, list):
        return T_ARR
    return T_OBJ


def _resolve_ref(root, ref):
    if not isinstance(ref, str) or not ref.startswith('#/'):
        return None
    node = root
    for sel in ref[2:].split('/'):
        if isinstance(node, dict) and sel in node:
            node = node[sel]
        elif isinstance(node, list) and sel.isdigit() and int(sel) < len(node):
            node = node[int(sel)]
        else:
            return None
    return node


def schema_types(schema, root=None, _visiting=None):
    """the JSON value types a schema admits: llama.cpp's common_chat_schema::value_types() over the node its
    schema builder makes (common/json-schema.cpp), same precedence: $ref, oneOf/anyOf, type list, const, enum,
    then type (structural keywords when there is none). A schema llama.cpp would reject counts as 'any'."""
    root = schema if root is None else root
    visiting = set() if _visiting is None else _visiting
    if not isinstance(schema, dict):
        return ALL_TYPES
    if '$ref' in schema:
        target = _resolve_ref(root, schema['$ref'])
        if target is None:
            return ALL_TYPES
        if id(target) in visiting:
            return frozenset()           # a cycle contributes no type, as in llama.cpp
        visiting.add(id(target))
        try:
            return schema_types(target, root, visiting)
        finally:
            visiting.discard(id(target))
    for key in ('oneOf', 'anyOf'):
        if key in schema:
            alts = schema[key]
            if not isinstance(alts, list) or not alts:
                return ALL_TYPES
            out = frozenset()
            for a in alts:
                out |= schema_types(a, root, visiting)
            return out
    t = schema.get('type')
    if isinstance(t, list):
        if not t:
            return ALL_TYPES
        out = frozenset()
        for one in t:
            alt = dict(schema)
            alt['type'] = one
            out |= schema_types(alt, root, visiting)
        return out
    if 'const' in schema:
        return frozenset((_json_type(schema['const']),))
    if 'enum' in schema:
        vals = schema['enum']
        if not isinstance(vals, list) or not vals:
            return ALL_TYPES
        return frozenset(_json_type(v) for v in vals)

    def all_of():
        alts = schema['allOf']
        if not isinstance(alts, list) or not alts:
            return ALL_TYPES
        out = ALL_TYPES
        for a in alts:
            out &= schema_types(a, root, visiting)
        return out

    has_props = 'properties' in schema or ('additionalProperties' in schema and schema['additionalProperties'] is not True)
    if t is None:
        if has_props:
            return frozenset((T_OBJ,))
        if 'allOf' in schema:
            return all_of()
        if 'items' in schema or 'prefixItems' in schema:
            return frozenset((T_ARR,))
        fmt = schema.get('format')
        if ('pattern' in schema or 'minLength' in schema or 'maxLength' in schema or
                fmt in ('date', 'time', 'date-time', 'uuid') or
                (isinstance(fmt, str) and len(fmt) == 5 and fmt[:4] == 'uuid' and '1' <= fmt[4] <= '5')):
            return frozenset((T_STR,))
        return ALL_TYPES
    if t == 'object':
        return all_of() if (not has_props and 'allOf' in schema) else frozenset((T_OBJ,))
    if t == 'string':
        return all_of() if 'allOf' in schema else frozenset((T_STR,))
    if t == 'array':
        return frozenset((T_ARR,))
    if t == 'integer':
        return frozenset((T_INT,))
    if t == 'number':
        return frozenset((T_NUM, T_INT))
    if t == 'boolean':
        return frozenset((T_BOOL,))
    if t == 'null':
        return frozenset((T_NULL,))
    return ALL_TYPES


def _tool_fields(tool):
    """(name, parameters schema) of an OpenAI tool ({"type": "function", "function": {...}}), a bare function
    ({"name", "parameters"}) or an Anthropic tool ({"name", "input_schema"})"""
    if not isinstance(tool, dict):
        return None, None
    fn = tool.get('function') if isinstance(tool.get('function'), dict) else tool
    name = fn.get('name')
    params = fn.get('parameters', fn.get('input_schema'))
    return (name if isinstance(name, str) else None), (params if isinstance(params, dict) else {})


class ToolSet:
    """declared tools: name -> {parameter: frozenset of JSON types}, plus the alias table"""

    def __init__(self, tools=None):
        self.params = {}
        for tool in tools or ():
            name, schema = _tool_fields(tool)
            if not name:
                continue
            props = schema.get('properties') if isinstance(schema, dict) else None
            types = {}
            if isinstance(props, dict):
                for p, ps in props.items():
                    types[p] = schema_types(ps, schema)
            self.params[name] = types
        aliases = {}
        for name in self.params:
            if '.' in name:
                ns, member = name.rsplit('.', 1)
                for a in (f'{ns}__{member}', f'{ns}={member}', f'{ns}_{member}', member):
                    aliases.setdefault(a, set()).add(name)
            if '__' in name:
                aliases.setdefault(name.replace('__', '_'), set()).add(name)
        self.aliases = {a: next(iter(ns)) for a, ns in aliases.items() if len(ns) == 1 and a not in self.params}
        folded = {}
        for name in list(self.params) + list(self.aliases):
            folded.setdefault(name.lower(), set()).add(self.aliases.get(name, name))
        self.folded = {a: next(iter(ns)) for a, ns in folded.items() if len(ns) == 1}

    def __bool__(self):
        return bool(self.params)

    def resolve(self, raw):
        """the declared tool a written name means: exact, else an unambiguous alias, else an unambiguous
        case-insensitive match of a name or alias, else the name as written"""
        name = raw.strip(WS).strip('"\'').strip(WS)
        if name in self.params:
            return name
        return self.aliases.get(name) or self.folded.get(name.lower(), name)

    def types(self, tool, param):
        """JSON types of a parameter; 'any' for an undeclared tool or parameter"""
        return self.params.get(tool, {}).get(param, ALL_TYPES)


# ---------------------------------------------------------------------------------------------------- JSON values
class _Fail(Exception):
    pass


_NUM = re.compile(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?')
_HEX = frozenset('0123456789abcdefABCDEF')


def _ws(s, i):
    while i < len(s) and s[i] in WS:
        i += 1
    return i


def _jstr(s, i):
    i += 1
    while True:
        if i >= len(s):
            raise _Fail
        c = s[i]
        if c == '"':
            return i + 1
        if c == '\\':
            if i + 1 >= len(s):
                raise _Fail
            e = s[i + 1]
            if e in '"\\/bfnrt':
                i += 2
            elif e == 'u':
                if i + 6 > len(s) or any(h not in _HEX for h in s[i + 2:i + 6]):
                    raise _Fail
                i += 6
            else:
                raise _Fail
        else:
            i += 1


def _jval(s, i, depth=0):
    if i >= len(s) or depth > 512:
        raise _Fail
    c = s[i]
    if c == '{':
        i = _ws(s, i + 1)
        if i < len(s) and s[i] == '}':
            return 'object', i + 1
        while True:
            if i >= len(s) or s[i] != '"':
                raise _Fail
            i = _ws(s, _jstr(s, i))
            if i >= len(s) or s[i] != ':':
                raise _Fail
            _, i = _jval(s, _ws(s, i + 1), depth + 1)
            i = _ws(s, i)
            if i < len(s) and s[i] == ',':
                i = _ws(s, i + 1)
                continue
            if i < len(s) and s[i] == '}':
                return 'object', i + 1
            raise _Fail
    if c == '[':
        i = _ws(s, i + 1)
        if i < len(s) and s[i] == ']':
            return 'array', i + 1
        while True:
            _, i = _jval(s, i, depth + 1)
            i = _ws(s, i)
            if i < len(s) and s[i] == ',':
                i = _ws(s, i + 1)
                continue
            if i < len(s) and s[i] == ']':
                return 'array', i + 1
            raise _Fail
    if c == '"':
        return 'string', _jstr(s, i)
    if c == '-' or '0' <= c <= '9':
        m = _NUM.match(s, i)
        if not m or (m.end() < len(s) and s[m.end()] in '0123456789.eE+-'):
            raise _Fail          # llama.cpp's number ends with a negative lookahead on [0-9.eE+-]
        return 'number', m.end()
    for lit, kind in (('true', 'bool'), ('false', 'bool'), ('null', 'null')):
        if s.startswith(lit, i):
            return kind, i + len(lit)
    raise _Fail


def json_kind(text):
    """'object' | 'array' | 'string' | 'number' | 'bool' | 'null' when text is exactly one JSON value under
    llama.cpp's PEG JSON grammar (no surrounding whitespace; raw control characters allowed inside strings), else
    None"""
    try:
        kind, end = _jval(text, 0)
    except _Fail:
        return None
    return kind if end == len(text) else None


def _pythonish_to_json(text):
    """single-quoted strings -> double-quoted, True/False/None outside strings -> true/false/null"""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == '\\' else 1
            out.append(text[i:j + 1])
            i = j + 1
        elif c == "'":
            j, buf = i + 1, []
            while j < n and text[j] != "'":
                if text[j] == '\\' and j + 1 < n:
                    buf.append("'" if text[j + 1] == "'" else text[j:j + 2])
                    j += 2
                    continue
                buf.append('\\"' if text[j] == '"' else text[j])
                j += 1
            out.append('"' + ''.join(buf) + '"')
            i = j + 1
        elif c.isalpha() or c == '_':
            j = i
            while j < n and (text[j].isalnum() or text[j] == '_'):
                j += 1
            word = text[i:j]
            out.append({'True': 'true', 'False': 'false', 'None': 'null'}.get(word, word))
            i = j
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def _jesc(s):
    """a JSON string's inside, as nlohmann's dump() writes it (ensure_ascii off)"""
    return json.dumps(s, ensure_ascii=False)[1:-1]


def coerce_value(raw, types):
    """(JSON text for a parameter's raw value, lenient) given the parameter's schema types. lenient is True when
    llama.cpp's parser would have rejected the value."""
    if types == frozenset((T_STR,)):
        return '"' + _jesc(raw) + '"', False
    if T_STR in types:
        allowed = {_KIND_OK[t] for t in types if t in _KIND_OK}
        if json_kind(raw) in allowed:
            return raw, False
        return '"' + _jesc(raw) + '"', False
    if json_kind(raw) is not None:
        return raw, False
    for cand in (raw.strip(WS), _pythonish_to_json(raw.strip(WS))):
        if json_kind(cand) is not None:
            return cand, True
    return '"' + _jesc(raw) + '"', True


# ---------------------------------------------------------------------------------------------------- results
class ToolCall:
    """complete: the call was closed by its tags. cut: the end of the stream fell inside a value (or a Hermes JSON
    body), so the arguments hold a truncated value (string) or lack one (typed): not safe to run. A call with
    complete=False and cut=False only lost its closing tags; every value in it is whole."""
    __slots__ = ('name', 'arguments', 'complete', 'cut', 'raw_name')

    def __init__(self, name, raw_name=None):
        self.name = name
        self.raw_name = raw_name if raw_name is not None else name
        self.arguments = ''
        self.complete = False
        self.cut = False

    def as_dict(self):
        return {'name': self.name, 'arguments': self.arguments}

    def __repr__(self):
        return f'ToolCall({self.name!r}, {self.arguments!r}, complete={self.complete}, cut={self.cut})'


class ParsedMessage:
    """reasoning_content, content, tool_calls; order: the content and the calls in the order the model wrote them,
    as ['text', str] and ['call', index] items (text after a call is possible only through the leniency rules)"""

    def __init__(self):
        self.reasoning_content = ''
        self.content = ''
        self.tool_calls = []
        self.order = []

    def usable_calls(self):
        """(index, call) of the calls that are safe to report as calls (not cut inside a value)"""
        return [(i, c) for i, c in enumerate(self.tool_calls) if not c.cut]

    def as_dict(self):
        return {'reasoning_content': self.reasoning_content, 'content': self.content,
                'tool_calls': [c.as_dict() for c in self.tool_calls]}


def _hold_point(buf, delims):
    """(pos, delim): the first position where one of delims starts (delim) or where a proper prefix of one runs to
    the end of buf (None); (len(buf), None) when neither. Everything before pos is decided text."""
    best, which = len(buf), None
    for d in delims:
        p = buf.find(d)
        if p != -1 and p < best:
            best, which = p, d
    for d in delims:
        for k in range(min(len(d) - 1, len(buf)), 0, -1):
            if buf.endswith(d[:k]):
                q = len(buf) - k
                if q < best:
                    best, which = q, None
                break
    return best, which


def _is_prefix_of_any(s, words):
    return any(w.startswith(s) for w in words)


# ---------------------------------------------------------------------------------------------------- parser
# states
R_LEAD, R, C_LEAD, C, T_OPEN, T_NAME, T_BODY, P_NAME, P_VALUE, T_END, T_AFTER, T_JSON, DONE, R_OPEN = range(14)
THINK_OPEN = '<think>'

_BODY_WORDS = (PARAM_OPEN, FUNC_CLOSE, TOOL_CLOSE, TOOL_OPEN)
_VALUE_FOLLOW = (PARAM_OPEN, FUNC_CLOSE, TOOL_CLOSE)


class OutputParser:
    """incremental reasoning / content / tool-call splitter; see the module docstring.

    tools: the request's tools (OpenAI, bare or Anthropic form); thinking: the request thinks (enable_thinking true);
    think_open: the generation prompt itself opened the think block (it ends '<think>\\n': Qwen3.6's template). False (the
    Qwen3 template: thinking on leaves the prompt at 'assistant\\n' and the model writes '<think>' first): one leading
    '<think>' of the output is consumed, and output that does not start with it is content, not reasoning (llama.cpp's
    reasoning parse when thinking is not forced open: an answer without tags must not end up with empty content); parse_tools:
    False for tool_choice none (tool calls stay content); parallel: parallel_tool_calls (False: only the first call
    counts, the rest of the output is dropped)."""

    def __init__(self, tools=None, thinking=True, parse_tools=True, parallel=True, think_open=True):
        self.toolset = tools if isinstance(tools, ToolSet) else ToolSet(tools)
        self.parse_tools = bool(parse_tools) and bool(self.toolset)
        self.parallel = parallel
        self.state = (R_LEAD if think_open else R_OPEN) if thinking else C_LEAD
        self.buf = ''
        self.msg = ParsedMessage()
        self.notes = []             # leniency applied (for logs / metrics)
        self.finished = False
        self._raw = ''              # text consumed since '<tool_call>' while the call's name is not known yet
        self._call = None           # the open ToolCall
        self._nargs = 0
        self._param = None          # (name, types) of the open parameter
        self._out = []

    # ------------------------------------------------------------------------------------------- emit helpers
    def _emit_text(self, kind, text):
        if not text:
            return
        if kind == 'reasoning':
            self.msg.reasoning_content += text
        else:
            self.msg.content += text
            if self.msg.order and self.msg.order[-1][0] == 'text':
                self.msg.order[-1][1] += text
            else:
                self.msg.order.append(['text', text])
        if self._out and self._out[-1][0] == kind:
            self._out[-1] = (kind, self._out[-1][1] + text)
        else:
            self._out.append((kind, text))

    def _emit_args(self, text):
        if not text:
            return
        self._call.arguments += text
        i = len(self.msg.tool_calls) - 1
        if self._out and self._out[-1][0] == 'tool_args' and self._out[-1][1] == i:
            self._out[-1] = ('tool_args', i, self._out[-1][2] + text)
        else:
            self._out.append(('tool_args', i, text))

    def _start_call(self, raw_name):
        name = self.toolset.resolve(raw_name)
        if name != raw_name:
            self.notes.append(f'tool name {raw_name!r} -> {name!r}')
        if name not in self.toolset.params:
            self.notes.append(f'undeclared tool {name!r}')
        self._call = ToolCall(name, raw_name)
        self.msg.tool_calls.append(self._call)
        self.msg.order.append(['call', len(self.msg.tool_calls) - 1])
        self._out.append(('tool_start', len(self.msg.tool_calls) - 1, name))
        self._nargs = 0
        self._raw = ''
        self._emit_args('{')

    def _end_call(self, complete=True):
        self._emit_args('}')
        self._call.complete = complete
        self._out.append(('tool_end', len(self.msg.tool_calls) - 1))
        self._call = None
        self._param = None

    def _key(self, name):
        sep = ',' if self._nargs else ''
        self._nargs += 1
        return sep + json.dumps(name.strip(WS), ensure_ascii=False) + ':'

    def _call_to_content(self):
        """a '<tool_call>' that did not open a call: its text (held in _raw + buf) becomes content"""
        self.notes.append('<tool_call> without a call: kept as content')
        self._emit_text('content', TOOL_OPEN + self._raw)
        self._raw = ''
        self.state = C

    # ------------------------------------------------------------------------------------------- public API
    def feed(self, text):
        """add text; returns the events it decided"""
        if self.finished:
            raise RuntimeError('feed() after finish()')
        self.buf += text
        self._run(final=False)
        out, self._out = self._out, []
        return out

    def finish(self):
        """end of the stream (EOS, a stop string, max_tokens or a cancel): flush and close; returns the last
        events"""
        if not self.finished:
            self.finished = True
            self._run(final=True)
            self._close(final=True)
        out, self._out = self._out, []
        return out

    def result(self):
        return self.msg

    @property
    def held(self):
        """characters fed but not decided yet"""
        return len(self.buf)

    # ------------------------------------------------------------------------------------------- the machine
    def _run(self, final):
        while True:
            before = (self.state, len(self.buf))
            self._step(final)
            if (self.state, len(self.buf)) == before:
                return

    def _step(self, final):
        st, buf = self.state, self.buf
        if st == R_OPEN:            # (the model opens its own think block: one leading '<think>' consumed, held while partial;
            i = _ws(buf, 0); rest = buf[i:]   #  no '<think>': the output is content)
            if rest.startswith(THINK_OPEN):
                self.buf = rest[len(THINK_OPEN):]; self.state = R_LEAD
            elif rest and (final or not THINK_OPEN.startswith(rest)):
                self.buf = rest; self.state = C_LEAD
            else:
                self.buf = rest
            return
        if st in (R_LEAD, C_LEAD):
            i = _ws(buf, 0)
            self.buf = buf[i:]
            if self.buf:
                self.state = R if st == R_LEAD else C
            return

        if st == R:
            pos, d = _hold_point(buf, (THINK_END, TOOL_OPEN))
            if final and d is None:
                pos = len(buf)
            self._emit_text('reasoning', buf[:pos])
            self.buf = buf[pos:]
            if d == THINK_END:
                self.buf = self.buf[len(THINK_END):]
                self.state = C_LEAD
            elif d == TOOL_OPEN:
                self.state = C_LEAD          # not consumed: the content state sees it
            return

        if st == C:
            if not self.parse_tools:
                self._emit_text('content', buf)
                self.buf = ''
                return
            pos, d = _hold_point(buf, (TOOL_OPEN,))
            if final and d is None:
                pos = len(buf)
            self._emit_text('content', buf[:pos])
            self.buf = buf[pos:]
            if d == TOOL_OPEN:
                self.buf = self.buf[len(TOOL_OPEN):]
                self._raw = ''
                self.state = T_OPEN
            return

        if st == T_OPEN:
            i = _ws(buf, 0)
            rest = buf[i:]
            if rest.startswith(FUNC_OPEN):
                if buf[:i] != '\n':
                    self.notes.append('whitespace after <tool_call> is not one newline')
                self._raw += buf[:i + len(FUNC_OPEN)]
                self.buf = rest[len(FUNC_OPEN):]
                self.state = T_NAME
            elif rest.startswith('{'):
                self._raw += buf[:i]
                self.buf = rest
                self.state = T_JSON
            elif rest and not FUNC_OPEN.startswith(rest):
                self._call_to_content()       # final with a bare or partial opener: dropped in _close
            return

        if st == T_NAME:
            m = re.search(r'[>\n<]', buf)
            if m is None:
                if len(buf) > MAX_NAME:
                    self._call_to_content()
                return                        # final: dropped in _close (name not complete)
            if m.group() == '>' and m.end() == len(buf) and not final:
                return                        # as llama.cpp: the name counts once the newline after '>' is seen
            raw_name = buf[:m.start()]
            if m.group() != '>' or not raw_name.strip(WS):
                self._call_to_content()
                return
            self.buf = buf[m.end():]
            self._start_call(raw_name)
            self.state = T_BODY
            return

        if st == T_BODY:
            i = _ws(buf, 0)
            rest = buf[i:]
            if rest.startswith(PARAM_OPEN):
                self.buf = rest[len(PARAM_OPEN):]
                self.state = P_NAME
            elif rest.startswith(FUNC_CLOSE):
                self.buf = rest[len(FUNC_CLOSE):]
                self._end_call()
                self.state = T_END
            elif rest.startswith(TOOL_CLOSE):
                self.notes.append('missing </function>')
                self.buf = rest[len(TOOL_CLOSE):]
                self._end_call()
                self.state = T_AFTER
            elif rest.startswith(TOOL_OPEN):
                self.notes.append('missing </function></tool_call>')
                self.buf = rest
                self._end_call()
                self.state = T_AFTER
            elif rest and not _is_prefix_of_any(rest, _BODY_WORDS):
                j = rest.find('<', 1)
                self.notes.append('text inside <function> dropped')
                self.buf = rest[j:] if j != -1 else ''
            else:
                self.buf = rest
            return

        if st == P_NAME:
            m = re.search(r'[>\n<]', buf)
            if m is None:
                if len(buf) > MAX_NAME:
                    self.notes.append('parameter name too long: dropped')
                    self.buf = ''
                    self.state = T_BODY
                return
            if m.group() == '<':
                self.notes.append('unterminated <parameter=: dropped')
                self.buf = buf[m.start():]
                self.state = T_BODY
                return
            if m.group() == '>' and m.end() == len(buf) and not final:
                return                        # wait for the newline that should follow '>'
            name = buf[:m.start()].strip(WS)
            rest = buf[m.end():]
            if m.group() == '>' and rest.startswith('\n'):
                rest = rest[1:]
            elif m.group() == '>':
                self.notes.append('no newline after <parameter=...>')
            else:
                self.notes.append('<parameter= closed by a newline')
            if self._call.name in self.toolset.params and name not in self.toolset.params[self._call.name]:
                self.notes.append(f'undeclared parameter {name!r}')
            types = self.toolset.types(self._call.name, name)
            self._param = (name, types)
            self.buf = rest
            if types == frozenset((T_STR,)):
                self._emit_args(self._key(name) + '"')
            self.state = P_VALUE
            return

        if st == P_VALUE:
            end, resume, decided = self._value_end(buf, final)
            name, types = self._param
            stream = types == frozenset((T_STR,))
            if end is None:
                if stream:
                    self._emit_args(_jesc(buf[:decided]))
                    self.buf = buf[decided:]
                return
            value = buf[:end]
            if stream:
                self._emit_args(_jesc(value) + '"')
            else:
                text, lenient = coerce_value(value, types)
                if lenient:
                    self.notes.append(f'parameter {name!r}: value coerced')
                self._emit_args(self._key(name) + text)
            self._param = None
            self.buf = buf[resume:]
            self.state = T_BODY
            return

        if st == T_END:
            i = _ws(buf, 0)
            rest = buf[i:]
            if rest.startswith(TOOL_CLOSE):
                self.buf = rest[len(TOOL_CLOSE):]
                self.state = T_AFTER
            elif rest and not TOOL_CLOSE.startswith(rest):
                self.notes.append('missing </tool_call>')
                self.buf = rest
                self.state = T_AFTER
            else:
                self.buf = rest
            return

        if st == T_AFTER:
            if not self.parallel:
                if buf.strip(WS):
                    self.notes.append('parallel_tool_calls=false: output after the first call dropped')
                self.buf = ''
                self.state = DONE
                return
            i = _ws(buf, 0)
            rest = buf[i:]
            if rest.startswith(TOOL_OPEN):
                self.buf = rest[len(TOOL_OPEN):]
                self._raw = ''
                self.state = T_OPEN
            elif rest and (final or not TOOL_OPEN.startswith(rest)):
                self.notes.append('text after the tool calls: kept as content')
                self.buf = rest
                self.state = C
            else:
                self.buf = rest
            return

        if st == T_JSON:
            p = buf.find(TOOL_CLOSE)
            if p == -1:
                return                        # final: tried in _close
            self._json_call(buf[:p], buf[:p + len(TOOL_CLOSE)])
            self.buf = buf[p + len(TOOL_CLOSE):]
            return

        if st == DONE:
            self.buf = ''

    def _value_end(self, buf, final):
        """(end, resume, decided) for a parameter value in buf: end = where the value ends (None while not known),
        resume = where parsing continues, decided = how much of buf is value text for sure (for streaming)"""
        i = 0
        while True:
            q = buf.find(PARAM_CLOSE, i)
            if q == -1:
                break
            nl = q > 0 and buf[q - 1] == '\n'
            after = buf[q + len(PARAM_CLOSE):]
            if nl and after.startswith('\n'):
                return q - 1, q + len(PARAM_CLOSE), q - 1          # '\n</parameter>\n': llama.cpp's terminator
            r = after.lstrip(WS)
            if any(r.startswith(w) for w in _VALUE_FOLLOW):
                self.notes.append('</parameter> without its newlines')
                return (q - 1 if nl else q), q + len(PARAM_CLOSE), (q - 1 if nl else q)
            if not final and (not r or _is_prefix_of_any(r, _VALUE_FOLLOW) or (nl and not after)):
                return None, None, (q - 1 if nl else q)            # undecided until more text arrives
            if final and not r:
                return (q - 1 if nl else q), q + len(PARAM_CLOSE), 0  # '</parameter>' at the very end
            i = q + 1
        # nothing decided from i on: hold back from where a terminator could still start (at the end of the
        # stream, the held tail is dropped by _close)
        pos, _ = _hold_point(buf[i:], ('\n' + PARAM_CLOSE, PARAM_CLOSE))
        return None, None, i + pos

    def _json_call(self, body, raw):
        """<tool_call>{"name": ..., "arguments": ...}</tool_call> (Hermes form)"""
        try:
            obj = json.loads(body.strip(WS))
        except ValueError:
            obj = None
        name = obj.get('name') if isinstance(obj, dict) else None
        if not isinstance(name, str) or not name.strip():
            self.notes.append('<tool_call> JSON without a name: kept as content')
            self._emit_text('content', TOOL_OPEN + self._raw + raw)
            self._raw = ''
            self.state = C
            return
        args = obj.get('arguments', obj.get('parameters', {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {'arguments': args}
        if not isinstance(args, dict):
            args = {'arguments': args}
        self.notes.append('Hermes JSON tool call')
        self._start_call(name)
        self._emit_args(json.dumps(args, ensure_ascii=False)[1:])  # '{' already emitted
        self._call.complete = True
        self._out.append(('tool_end', len(self.msg.tool_calls) - 1))
        self._call = None
        self.state = T_AFTER

    def _close(self, final):
        st = self.state
        if st == T_JSON:
            body = self.buf
            self.buf = ''
            try:
                obj = json.loads(body.strip(WS))
            except ValueError:
                obj = None
            if isinstance(obj, dict) and isinstance(obj.get('name'), str):
                self._json_call(body, body)
                self.msg.tool_calls[-1].complete = False   # the JSON parsed whole: only '</tool_call>' is missing
            return
        if st == P_VALUE:
            name, types = self._param
            if types == frozenset((T_STR,)):
                self._emit_args('"')          # the decided text is out; the held tail may be a cut-off terminator
            else:
                self.notes.append(f'parameter {name!r} cut off: dropped')
            self._call.cut = True
            self.buf = ''
            self.state = T_BODY
        if self._call is not None:
            self.notes.append('tool call cut off: closed')
            self._end_call(complete=False)
        self.buf = ''


def parse_text(text, tools=None, thinking=True, parse_tools=True, parallel=True):
    """one-shot parse (the non-streaming path)"""
    p = OutputParser(tools, thinking=thinking, parse_tools=parse_tools, parallel=parallel)
    p.feed(text)
    p.finish()
    return p.result()
