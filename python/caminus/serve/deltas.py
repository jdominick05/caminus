"""OpenAI and Anthropic shapes of the output parser's events (spec-server.md section 2-3, step S2).

toolparse.OutputParser emits ('reasoning', t), ('content', t), ('tool_start', i, name), ('tool_args', i, piece)
and ('tool_end', i). This module turns them into
  * OpenAI chat.completion.chunk deltas (choices[0].delta): reasoning_content, content, and tool_calls entries
    {index, id, type, function: {name, arguments}} whose argument pieces concatenate to the call's JSON;
  * Anthropic content-block events: content_block_start / content_block_delta / content_block_stop for thinking
    (thinking_delta, closed with an empty signature_delta as llama-server does, server-task.cpp:910-919), text
    (text_delta) and tool_use (input_json_delta). One block is open at a time; it is closed before the next starts,
    as the Anthropic API does;
and builds the non-streaming message bodies from a ParsedMessage. Calls the stream cut inside a value
(ToolCall.cut) are left out of the non-streaming bodies and do not make the stop reason tool_calls / tool_use; in a
streaming response their deltas are already out (what to do then is step S4's). Envelope events (message_start, message_delta
with usage, message_stop, pings) and the HTTP framing are the server's (step S4)."""
import json
import secrets
import string

_ALNUM = string.ascii_letters + string.digits


def new_id(prefix, n=24):
    return prefix + ''.join(secrets.choice(_ALNUM) for _ in range(n))


# ---------------------------------------------------------------------------------------------------- OpenAI
class OpenAIStream:
    """events -> list of delta dicts for chat.completion.chunk choices[0].delta"""

    def __init__(self, id_fn=None):
        self.id_fn = id_fn or (lambda: new_id('call_'))
        self.ids = []

    def deltas(self, events):
        out = []
        for ev in events:
            kind = ev[0]
            if kind == 'reasoning':
                out.append({'reasoning_content': ev[1]})
            elif kind == 'content':
                out.append({'content': ev[1]})
            elif kind == 'tool_start':
                _, i, name = ev
                while len(self.ids) <= i:
                    self.ids.append(self.id_fn())
                out.append({'tool_calls': [{'index': i, 'id': self.ids[i], 'type': 'function',
                                            'function': {'name': name, 'arguments': ''}}]})
            elif kind == 'tool_args':
                out.append({'tool_calls': [{'index': ev[1], 'function': {'arguments': ev[2]}}]})
        return out


def openai_finish_reason(msg, stop):
    """stop: 'eos' | 'stop' (a stop string) | 'length' (max_tokens)"""
    if stop == 'length':
        return 'length'
    return 'tool_calls' if msg.usable_calls() else 'stop'


def openai_message(msg, ids=None, id_fn=None):
    """the non-streaming choices[0].message; ids: the call ids already used in a stream (kept), else new ones.
    Cut calls are left out."""
    id_fn = id_fn or (lambda: new_id('call_'))
    calls = msg.usable_calls()
    out = {'role': 'assistant', 'content': msg.content if msg.content or not calls else None}
    if msg.reasoning_content:
        out['reasoning_content'] = msg.reasoning_content
    if calls:
        ids = list(ids or ())
        while len(ids) < len(msg.tool_calls):
            ids.append(id_fn())
        out['tool_calls'] = [{'id': ids[i], 'type': 'function',
                              'function': {'name': c.name, 'arguments': c.arguments}} for i, c in calls]
    return out


def apply_openai_deltas(deltas):
    """a client's view: accumulate deltas into {reasoning_content, content, tool_calls: [{id, name, arguments}]}
    (what the OpenAI SDK's stream helpers do; used by the tests)"""
    acc = {'reasoning_content': '', 'content': '', 'tool_calls': []}
    for d in deltas:
        acc['reasoning_content'] += d.get('reasoning_content') or ''
        acc['content'] += d.get('content') or ''
        for tc in d.get('tool_calls') or ():
            i = tc['index']
            while len(acc['tool_calls']) <= i:
                acc['tool_calls'].append({'id': None, 'name': '', 'arguments': ''})
            slot = acc['tool_calls'][i]
            if tc.get('id'):
                slot['id'] = tc['id']
            fn = tc.get('function') or {}
            slot['name'] += fn.get('name') or ''
            slot['arguments'] += fn.get('arguments') or ''
    return acc


# ---------------------------------------------------------------------------------------------------- Anthropic
class AnthropicStream:
    """events -> list of Anthropic content-block event dicts (the SSE data objects; their 'type' is the SSE event
    name). Call close() at the end of the stream to close the open block."""

    def __init__(self, id_fn=None):
        self.id_fn = id_fn or (lambda: new_id('toolu_'))
        self.index = -1
        self.open = None            # 'thinking' | 'text' | ('tool', i)
        self.ids = []
        self.tool_blocks = 0

    def _close(self, out):
        if self.open is None:
            return
        if self.open == 'thinking':
            out.append({'type': 'content_block_delta', 'index': self.index,
                        'delta': {'type': 'signature_delta', 'signature': ''}})
        out.append({'type': 'content_block_stop', 'index': self.index})
        self.open = None

    def _start(self, out, kind, block):
        self._close(out)
        self.index += 1
        self.open = kind
        out.append({'type': 'content_block_start', 'index': self.index, 'content_block': block})

    def events(self, events):
        out = []
        for ev in events:
            kind = ev[0]
            if kind == 'reasoning':
                if self.open != 'thinking':
                    self._start(out, 'thinking', {'type': 'thinking', 'thinking': '', 'signature': ''})
                out.append({'type': 'content_block_delta', 'index': self.index,
                            'delta': {'type': 'thinking_delta', 'thinking': ev[1]}})
            elif kind == 'content':
                if self.open != 'text':
                    self._start(out, 'text', {'type': 'text', 'text': ''})
                out.append({'type': 'content_block_delta', 'index': self.index,
                            'delta': {'type': 'text_delta', 'text': ev[1]}})
            elif kind == 'tool_start':
                _, i, name = ev
                while len(self.ids) <= i:
                    self.ids.append(self.id_fn())
                self._start(out, ('tool', i), {'type': 'tool_use', 'id': self.ids[i], 'name': name, 'input': {}})
                self.tool_blocks += 1
            elif kind == 'tool_args':
                if self.open == ('tool', ev[1]):
                    out.append({'type': 'content_block_delta', 'index': self.index,
                                'delta': {'type': 'input_json_delta', 'partial_json': ev[2]}})
            elif kind == 'tool_end':
                if self.open == ('tool', ev[1]):
                    self._close(out)
        return out

    def close(self):
        out = []
        self._close(out)
        return out


def anthropic_stop_reason(msg, stop):
    """stop: 'eos' | 'stop' (a stop string) | 'length' (max_tokens)"""
    if stop == 'length':
        return 'max_tokens'
    if msg.usable_calls():
        return 'tool_use'
    return 'stop_sequence' if stop == 'stop' else 'end_turn'


def tool_input(arguments):
    try:
        v = json.loads(arguments, strict=False)
    except ValueError:
        return {}
    return v if isinstance(v, dict) else {}


def anthropic_content(msg, ids=None, id_fn=None, keep_cut=False):
    """the message's content blocks in the order the model wrote them (the order the stream uses); cut calls are
    left out unless keep_cut (to compare with a stream that sent them)"""
    id_fn = id_fn or (lambda: new_id('toolu_'))
    blocks = []
    if msg.reasoning_content:
        blocks.append({'type': 'thinking', 'thinking': msg.reasoning_content, 'signature': ''})
    ids = list(ids or ())
    while len(ids) < len(msg.tool_calls):
        ids.append(id_fn())
    for kind, v in msg.order:
        if kind == 'text':
            blocks.append({'type': 'text', 'text': v})
        elif keep_cut or not msg.tool_calls[v].cut:
            c = msg.tool_calls[v]
            blocks.append({'type': 'tool_use', 'id': ids[v], 'name': c.name, 'input': tool_input(c.arguments)})
    return blocks
