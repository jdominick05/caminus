# Parts of this file (pin_billing_header; the OpenAI/Anthropic conversion shape) are rewritten, line-reviewed, from Strata (serve/frontend.py),
# which is distributed under the following license:
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

"""Request normalization: OpenAI Chat Completions and Anthropic Messages requests -> the chat template's input
(messages, tools, client template kwargs). spec-server.md section 3, step S1.

Written for K3 from the formats themselves and from llama.cpp's server (tools/server/server-chat.cpp: the
Anthropic -> OpenAI conversion and normalize_anthropic_billing_header). pin_billing_header (cch and the cc_version
4th part, with its length bounds) is rewritten and line-reviewed from Strata's pin_billing_stamp (MIT, see the header).

Output message form (what the template reads):
  {'role': 'system'|'user'|'assistant'|'tool', 'content': str,
   'reasoning_content': str (assistant, optional),
   'tool_calls': [{'id': str, 'type': 'function', 'function': {'name': str, 'arguments': dict}}] (assistant)}
Tools are OpenAI-shaped: {'type': 'function', 'function': {'name', 'description', 'parameters'}}.

Choices (documented, tested in tests/serve/test_normalize.py):
  * string tool arguments are parsed into objects (the template raises on a non-empty string); '' / None -> {};
    a string that is not a JSON object is a 400;
  * a leading '<think>...</think>' block inside an assistant's content (clients that send reasoning inline) moves
    to reasoning_content when the message has none;
  * system/developer messages after the first non-system message become user messages (the template raises);
  * images and other non-text parts are a 400 (K3 has no vision path);
  * Anthropic: system blocks are joined with '\\n'; tool_result blocks become tool messages, in block order, before
    the same message's text; a request without "thinking" (or with type disabled) renders with thinking off;
  * Claude Code's billing header (first system text) has its per-request cch stamp and the per-session 4th part
    of cc_version pinned to 'f's, so the system prompt is the same on every request (llama.cpp #21793)."""
import json
import re


class RequestError(ValueError):
    """a malformed request: HTTP 400"""


# ------------------------------------------------------------------------------------------------ helpers
def parse_tool_arguments(args, name='?'):
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        if not args.strip():
            return {}
        try:
            v = json.loads(args)
        except json.JSONDecodeError as e:
            raise RequestError(f'tool call "{name}": arguments are not valid JSON ({e.msg})') from None
        if not isinstance(v, dict):
            raise RequestError(f'tool call "{name}": arguments must be a JSON object')
        return v
    raise RequestError(f'tool call "{name}": arguments must be an object or a JSON string')


_TEXT_TYPES = ('text', 'input_text', 'output_text')


def text_of(content, where):
    """str, None, or a list of text parts -> str (parts are concatenated, as the template's render_content does)"""
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, dict) and p.get('type', 'text') in _TEXT_TYPES and isinstance(p.get('text'), str):
                out.append(p['text'])
            else:
                kind = p.get('type') if isinstance(p, dict) else type(p).__name__
                raise RequestError(f'{where}: unsupported content part {kind!r} (text only)')
        return ''.join(out)
    raise RequestError(f'{where}: content must be a string or a list of parts')


_INLINE_THINK = re.compile(r'\A\s*<think>(.*?)</think>\s*', re.S)


def split_inline_reasoning(content):
    """'<think>R</think>C' -> (R, C); otherwise (None, content)"""
    m = _INLINE_THINK.match(content)
    if not m:
        return None, content
    return m.group(1), content[m.end():]


BILLING_HEADER = 'x-anthropic-billing-header:'


def pin_billing_header(text, window=200):
    """Claude Code's 'x-anthropic-billing-header: cc_version=2.1.101.0d4; cc_entrypoint=cli; cch=12ab3;' prefix:
    the cch value (per request) and cc_version's 4th and later dot parts (per session) become 'f's. Only a header at
    the very start of the text is touched, and only within its first `window` characters; lengths are kept."""
    if not isinstance(text, str) or not text.startswith(BILLING_HEADER):
        return text
    s = list(text)
    i = text.find('cch=', len(BILLING_HEADER))
    if 0 <= i < window:
        a = i + 4
        b = text.find(';', a)
        if a < b <= a + 16:
            s[a:b] = 'f' * (b - a)
    i = text.find('cc_version=', len(BILLING_HEADER))
    if 0 <= i < window:
        a = i + len('cc_version=')
        b = text.find(';', a)
        if a < b <= a + 64:
            parts = text[a:b].split('.')
            if len(parts) > 3:
                t = a + len('.'.join(parts[:3])) + 1
                s[t:b] = 'f' * (b - t)
    return ''.join(s)


def _late_system_to_user(messages):
    seen_other = False
    for m in messages:
        if m['role'] in ('system', 'developer'):
            if seen_other:
                m['role'] = 'user'
            else:
                m['role'] = 'system'
        else:
            seen_other = True
    return messages


def _openai_tools(tools):
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise RequestError('tools must be a list')
    out = []
    for i, t in enumerate(tools):
        if not isinstance(t, dict):
            raise RequestError(f'tools[{i}] must be an object')
        if t.get('type', 'function') != 'function':
            raise RequestError(f'tools[{i}]: unsupported tool type {t.get("type")!r}')
        fn = t.get('function') if 'function' in t else t
        if not isinstance(fn, dict) or not isinstance(fn.get('name'), str) or not fn['name']:
            raise RequestError(f'tools[{i}]: missing function name')
        out.append(t if 'function' in t and t.get('type') == 'function' else {'type': 'function', 'function': fn})
    return out or None


def _assistant(content, reasoning, calls):
    m = {'role': 'assistant', 'content': content}
    if reasoning is None and isinstance(content, str):
        r, c = split_inline_reasoning(content)
        if r is not None:
            reasoning, m['content'] = r, c
    if reasoning is not None:
        m['reasoning_content'] = reasoning
    if calls:
        m['tool_calls'] = calls
    return m


# ------------------------------------------------------------------------------------------------ OpenAI
def openai_chat_to_template(req):
    """OpenAI /v1/chat/completions body -> {'model', 'messages', 'tools', 'client_kwargs'}"""
    if not isinstance(req, dict):
        raise RequestError('body must be a JSON object')
    msgs = req.get('messages')
    if not isinstance(msgs, list) or not msgs:
        raise RequestError("'messages' must be a non-empty list")
    out = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            raise RequestError(f'messages[{i}] must be an object')
        role = m.get('role')
        where = f'messages[{i}].content'
        if role in ('system', 'developer', 'user', 'tool'):
            out.append({'role': role, 'content': text_of(m.get('content'), where)})
        elif role == 'assistant':
            reasoning = m.get('reasoning_content')
            if reasoning is None and isinstance(m.get('reasoning'), str):
                reasoning = m['reasoning']
            calls = []
            for j, c in enumerate(m.get('tool_calls') or []):
                if not isinstance(c, dict):
                    raise RequestError(f'messages[{i}].tool_calls[{j}] must be an object')
                fn = c.get('function', c)
                name = fn.get('name') if isinstance(fn, dict) else None
                if not isinstance(name, str) or not name:
                    raise RequestError(f'messages[{i}].tool_calls[{j}]: missing function name')
                calls.append({'id': c.get('id') or '', 'type': 'function',
                              'function': {'name': name, 'arguments': parse_tool_arguments(fn.get('arguments'), name)}})
            out.append(_assistant(text_of(m.get('content'), where), reasoning, calls))
        else:
            raise RequestError(f'messages[{i}]: unsupported role {role!r}')
    client = {}
    reasoning = req.get('reasoning')
    if req.get('reasoning_effort') is not None:
        client['reasoning_effort'] = req['reasoning_effort']
    elif isinstance(reasoning, dict) and reasoning.get('effort') is not None:
        client['reasoning_effort'] = reasoning['effort']
    ctk = req.get('chat_template_kwargs')
    if ctk is not None and not isinstance(ctk, dict):
        raise RequestError('chat_template_kwargs must be an object')
    for k, v in (ctk or {}).items():
        if k in ('enable_thinking', 'reasoning_effort', 'preserve_thinking'):
            client[k] = v
    return {'model': req.get('model'), 'messages': _late_system_to_user(out),
            'tools': _openai_tools(req.get('tools')), 'client_kwargs': client}


# ------------------------------------------------------------------------------------------------ Anthropic
def _anthropic_tools(tools):
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise RequestError('tools must be a list')
    out = []
    for i, t in enumerate(tools):
        if not isinstance(t, dict) or not isinstance(t.get('name'), str):
            raise RequestError(f'tools[{i}]: missing name')
        if t.get('type') not in (None, 'custom'):
            raise RequestError(f'tools[{i}]: server tool {t.get("type")!r} is not supported')
        fn = {'name': t['name']}
        if t.get('description') is not None:
            fn['description'] = t['description']
        fn['parameters'] = t.get('input_schema', {'type': 'object', 'properties': {}})
        out.append({'type': 'function', 'function': fn})
    return out or None


def anthropic_to_template(req):
    """Anthropic /v1/messages body -> {'model', 'messages', 'tools', 'client_kwargs'}"""
    if not isinstance(req, dict):
        raise RequestError('body must be a JSON object')
    out = []
    system = req.get('system')
    if system is not None:
        if isinstance(system, str):
            text = pin_billing_header(system)
        elif isinstance(system, list):
            parts = []
            for j, b in enumerate(system):
                if not isinstance(b, dict) or b.get('type') != 'text' or not isinstance(b.get('text'), str):
                    raise RequestError(f'system[{j}]: text blocks only')
                parts.append(pin_billing_header(b['text']))
            text = '\n'.join(parts)
        else:
            raise RequestError('system must be a string or a list of text blocks')
        if text:
            out.append({'role': 'system', 'content': text})
    msgs = req.get('messages')
    if not isinstance(msgs, list) or not msgs:
        raise RequestError("'messages' must be a non-empty list")
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get('role') not in ('user', 'assistant'):
            raise RequestError(f'messages[{i}]: role must be user or assistant')
        role, content = m['role'], m.get('content')
        if isinstance(content, str):
            out.append({'role': role, 'content': content} if role == 'user' else _assistant(content, None, []))
            continue
        if not isinstance(content, list):
            raise RequestError(f'messages[{i}].content must be a string or a list of blocks')
        texts, reasoning, calls = [], None, []
        for j, b in enumerate(content):
            kind = b.get('type') if isinstance(b, dict) else None
            where = f'messages[{i}].content[{j}]'
            if kind == 'text':
                texts.append(text_of(b.get('text'), where))
            elif kind == 'thinking' and role == 'assistant':
                reasoning = (reasoning or '') + (b.get('thinking') or '')
            elif kind == 'redacted_thinking' and role == 'assistant':
                continue
            elif kind == 'tool_use' and role == 'assistant':
                name = b.get('name')
                if not isinstance(name, str) or not name:
                    raise RequestError(f'{where}: tool_use without a name')
                calls.append({'id': b.get('id') or '', 'type': 'function',
                              'function': {'name': name, 'arguments': parse_tool_arguments(b.get('input'), name)}})
            elif kind == 'tool_result' and role == 'user':
                out.append({'role': 'tool', 'content': text_of(b.get('content'), where + '.content')})
            else:
                raise RequestError(f'{where}: unsupported block {kind!r} in a {role} message')
        if role == 'user':
            if texts:
                out.append({'role': 'user', 'content': ''.join(texts)})
        elif texts or reasoning is not None or calls:
            out.append(_assistant(''.join(texts), reasoning, calls))
    client = {}
    thinking = req.get('thinking')
    if not isinstance(thinking, dict) or thinking.get('type') == 'disabled':
        client['enable_thinking'] = False
    oc = req.get('output_config')
    if isinstance(oc, dict) and oc.get('effort') is not None:
        client['reasoning_effort'] = oc['effort']
    return {'model': req.get('model'), 'messages': _late_system_to_user(out),
            'tools': _anthropic_tools(req.get('tools')), 'client_kwargs': client}
