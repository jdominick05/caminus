"""Model transports: OpenAI-compatible chat completions (the local server) and Anthropic's Messages API.
Standard library only; no engine, GPU or agent-CLI dependency."""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from . import effort


# a speed is measured only over a window this long with this many tokens: a reply that arrived in one burst (Haiku
# with thinking) divided its tokens by microseconds and the status line read 31,746,037 tok/s
SPEED_MIN_S = 0.25
SPEED_MIN_TOKENS = 16
SHORT_CALL = 4096          # a call with less output room than this runs without thinking (see _anthropic)
# models that refuse thinking off (current Opus and Sonnet; the advisor runs on them): short calls keep thinking there
THINKING_REQUIRED = frozenset(('claude-opus-5-5', 'claude-opus-5', 'claude-sonnet-5-5', 'claude-sonnet-5',
                               'claude-fable-5-2', 'claude-fable-5-1', 'claude-fable-5'))


def split_content(content):
    """An OpenAI-compatible message content as (text, thinking). Most servers send a string; Mistral sends typed blocks
    for its reasoning models: [{type: thinking, thinking: [{type: text, text}] or str}, {type: text, text}]."""
    if content is None or isinstance(content, str):
        return content or '', ''
    text, thinking = [], []
    for b in content if isinstance(content, list) else [content]:
        if isinstance(b, str):
            text.append(b); continue
        if not isinstance(b, dict):
            continue
        if b.get('type') == 'thinking':
            inner = b.get('thinking')
            for x in inner if isinstance(inner, list) else [inner]:
                thinking.append(x.get('text', '') if isinstance(x, dict) else (x or ''))
        elif b.get('type') in ('text', None) and isinstance(b.get('text'), str):
            text.append(b['text'])
    return ''.join(text), ''.join(thinking)


class ModelError(Exception):
    pass


class ModelContextError(ModelError):
    pass


class ModelTruncated(ModelError):
    def __init__(self, partial, reason='model output budget exhausted; increase --max-tokens and resume this goal',
                 retryable=False):
        super().__init__(reason)
        self.partial = partial
        # A budget truncation needs a human decision; a stream cut is the server going away mid-reply and
        # the same turn succeeds once it is back, so the runtime retries it instead of pausing the goal.
        self.retryable = retryable


class ModelRefused(ModelError):
    """the provider declined the request (Messages API stop_reason 'refusal'): retrying the same turn cannot help"""
    status = 400


ANTHROPIC_VERSION = '2026-10-21'

# USD per million tokens (input, output, cache read, 5-minute cache write), below and above a prompt-length line.
# From the Claude API reference as cached 2026-11-03: check live pricing before budgeting on these numbers.
PRICES = {'claude-haiku-5-5': ((0.10, 0.50, 0.01, 0.125), (0.50, 2.50, 0.05, 0.625), 100_000),
          'claude-sonnet-5-5': ((2.00, 10.00, 0.10, 2.50),) * 2 + (10**9,),
          'claude-opus-5-5': ((4.00, 20.00, 0.20, 5.00),) * 2 + (10**9,),
          'claude-opus-4-8': ((5.00, 25.00, 0.50, 6.25),) * 2 + (10**9,)}
# Models that accept Anthropic's server-side refusal fallback in its "default" form (Haiku 5.5 does not: its refusals
# have no server-side fallback). The same list as the Qwen Code advisor's ANTHROPIC_FALLBACK_MODELS.
FALLBACK_MODELS = ('claude-opus-5-5', 'claude-opus-5', 'claude-sonnet-5-5', 'claude-fable-5-1')
FALLBACK_BETA = 'server-side-fallback-2026-07-01'


# Prices a provider published (the model catalog: DeepInfra's per-model pricing), in the PRICES row form. Filled from
# a config's `prices` when the runtime starts, so a catalog model is metered like a built-in one.
CATALOG_PRICES = {}


def remember_prices(model, prices):
    """prices: {input, output, cache_read} in USD per million tokens, as the catalog stores them."""
    if model and prices and prices.get('input') is not None and prices.get('output') is not None:
        inp, out = float(prices['input']), float(prices['output'])
        read = float(prices['cache_read']) if prices.get('cache_read') is not None else inp
        CATALOG_PRICES[model] = ((inp, out, read, inp),) * 2 + (10**9,)


def price(model, usage, served=None):
    """the USD cost of one call's usage on a priced model, or None when the model's price is not known. When a fallback
    served the call (served differs from model), it is priced at the serving model's rates, or at the highest known
    rates when that model is not in the table: an estimate that errs high."""
    row = PRICES.get(model) or CATALOG_PRICES.get(model)
    if row is None:
        return None
    if served and served != model:
        row = PRICES.get(served) or max(PRICES.values(), key=lambda r: r[0][1])
    low, high, line = row
    usage = usage or {}
    # Anthropic reports cache reads as cache_read_input_tokens; OpenAI-style providers (DeepInfra) inside
    # prompt_tokens_details.cached_tokens, already counted in prompt_tokens. Missing the second priced every cached
    # token at the full input rate (4-5x on a cached DeepSeek prompt).
    read = usage.get('cache_read_input_tokens') or (usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
    wrote = usage.get('cache_creation_input_tokens') or 0
    out = usage.get('completion_tokens', usage.get('output_tokens')) or 0
    # last_usage reports prompt_tokens as everything sent; the uncached part is what remains after the cache.
    total = usage.get('prompt_tokens')
    fresh = (total - read - wrote) if isinstance(total, int) else (usage.get('input_tokens') or 0)
    rate = high if fresh + read + wrote > line else low
    return (max(0, fresh) * rate[0] + out * rate[1] + read * rate[2] + wrote * rate[3]) / 1e6
LOOPBACK = ('127.0.0.1', 'localhost', '::1')


def http_error(e):
    """a ModelError carrying the status, the provider's reason (the body's first 2 KB), any retry-after and the
    provider's request id (for a support ticket)"""
    try:
        detail = e.read(2048).decode('utf-8', 'replace').strip()
    except (OSError, ValueError):
        detail = ''
    finally:
        e.close()
    err = ModelError(str(e) + (': ' + detail if detail else ''))
    err.status = e.code      # a refusal must not be retried like an outage
    err.request_id = (e.headers or {}).get('request-id')
    try:
        err.retry_after = float((e.headers or {}).get('retry-after') or 0) or None
    except (TypeError, ValueError):
        err.retry_after = None
    return err


class Cancelled(Exception):
    """the goal was cancelled while the model was generating; nothing of this reply is stored"""


class Model:
    def __init__(self, config, opener=urllib.request.urlopen):
        self.config, self.opener = config, opener
        self.base = config['endpoint'].rstrip('/')
        parsed = urllib.parse.urlsplit(self.base)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('endpoint must be an HTTP(S) URL without embedded credentials')
        self.root = self.base[:-3] if self.base.endswith('/v1') else self.base
        self.api = config.get('api', 'openai')
        if self.api not in ('openai', 'anthropic'):
            raise ValueError('api must be openai or anthropic')
        # A remote provider bills each call: the runtime bounds its retries (a local server is waited for).
        # A paid provider reached through the owner's key relay listens on loopback but bills like a remote one.
        self.direct = parsed.hostname not in LOOPBACK
        self.remote = self.direct or bool(config.get('relay'))
        self.context = config.get('context')
        if self.api == 'anthropic' and self.context is None:
            raise ValueError('the anthropic api needs an explicit context window (--context)')
        self.counter = True
        self.last_usage = {}
        self.last_speed = None
        self.last_call = {}      # the newest Messages API call: requested and serving model, fallback, stop, request id

    def headers(self, stream=False, beta=None):
        name = self.config.get('api_key_env', 'CAMINUS_AGENT_API_KEY')
        key = os.environ.get(name)
        headers = {'Content-Type': 'application/json'}
        if stream:
            headers['Accept'] = 'text/event-stream'
        if self.api == 'anthropic':
            # No key is refused before anything is sent, unless the endpoint is a local relay that adds it.
            if key:
                headers['x-api-key'] = key
            elif self.direct:
                err = ModelError('the API key variable %s is not set in the agent environment' % name)
                err.status = 401
                raise err
            headers['anthropic-version'] = ANTHROPIC_VERSION
            if beta:
                headers['anthropic-beta'] = beta
        else:
            headers['Authorization'] = 'Bearer ' + (key or 'caminus-local')
        return headers

    def request(self, url, body=None):
        req = urllib.request.Request(url, None if body is None else json.dumps(body).encode(), self.headers())
        try:
            with self.opener(req, timeout=float(self.config.get('timeout', 900))) as r:
                raw = r.read(16 * 1024 * 1024 + 1)
                if len(raw) > 16 * 1024 * 1024:
                    raise ModelError('model response exceeds16MiB')
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            raise http_error(e) from e
        except (OSError, ValueError) as e:
            raise ModelError(str(e)) from e

    def probe(self, timeout=3.0):
        """whether the server answers at all (a local server's /props, a short timeout); a remote API is assumed up"""
        if self.remote or self.api == 'anthropic':
            return True
        try:
            with self.opener(urllib.request.Request(self.root + '/props', None, self.headers()), timeout=timeout) as r:
                r.read(65536)
            return True
        except (OSError, ValueError, urllib.error.HTTPError):
            return False

    def context_size(self):
        if self.context is None:
            p = self.request(self.root + '/props')
            self.context = p.get('n_ctx')
        if isinstance(self.context, bool) or not isinstance(self.context, int) or self.context < 1024:
            raise ModelError('backend must report a positive context window, or set --context explicitly')
        return self.context

    def reply_limit(self):
        """A reply's token limit (effort.reply_limit): config max_tokens; else a local brain the card's maximum output up
        to 64K, a paid one 4096 (its catalog maximum, up to 64K, is set when it is picked)."""
        return effort.reply_limit(self.config, self.remote)

    def complete(self, messages, tools=None, max_tokens=None, model=None, on_delta=None):
        """One reply. With on_delta(kind, value) the reply streams: kind is 'reasoning', 'content' or 'tool'
        (a tool call's name starts), value is the text or the name. on_delta returning False cancels: the
        stream is closed and Cancelled is raised, so a cancelled turn stores no reply and runs no tool."""
        remaining = self.context_size() - self.tokens(messages, tools or []) - 256
        if remaining < 1:
            raise ModelContextError('request exceeds the model context; compact or shorten the goal')
        if self.api == 'anthropic':
            return self._anthropic(messages, tools, min(max_tokens or self.reply_limit(), remaining), model, on_delta)
        limit = min(max_tokens or self.reply_limit(), remaining)
        # the effort in force (effort.py): a local model takes it as the server's model id for the level, a provider
        # with reasoning_effort as that field; a short side call (the judge's) runs with thinking off
        choice = effort.resolve(self.config, short=limit < SHORT_CALL)
        body = dict(model=model or choice['model'] or self.config['model'], messages=messages,
                    stream=on_delta is not None, max_tokens=limit)
        # sampling: none for a local brain (its server applies the model's own defaults: a fixed 0.6 overrode the
        # Qwen3.8 cards' 1.0), a paid model's card values when known, the config's own when set
        body.update(effort.sampling(self.config, self.remote))
        if choice['reasoning_effort']:
            body['reasoning_effort'] = choice['reasoning_effort']   # (DeepInfra, Mistral; 'none' turns it off)
        if choice['thinking_off']:
            body['chat_template_kwargs'] = dict(enable_thinking=False)   # a local server with no -nothink id
        if self.config.get('provider_kind') == 'mistral':
            # Mistral routes requests to its prompt cache by this key (a hint, not a promise): one stable key per agent,
            # since its turns share the system prompt and history as their prefix. Without it the cache hit 0 %.
            import hashlib
            body['prompt_cache_key'] = 'ember-' + hashlib.sha256(
                ('%s|%s' % (self.config.get('name') or '', body['model'])).encode()).hexdigest()[:24]
        if body['max_tokens'] < SHORT_CALL:
            # A short side call (the finish judge's 600 tokens): a reasoning model spent all of it thinking and the
            # verdict came back cut off (GLM 5.3 on Mistral: a finish let through unchecked). Where the provider has a
            # switch it is turned off; either way the call gets room for thinking (billed per token written, so the
            # room costs nothing unused).
            body['max_tokens'] = min(SHORT_CALL, remaining)   # (thinking is off for it: effort.resolve(short=True))
        if tools:
            body['tools'] = tools
        if on_delta is not None:
            # usage in the final chunk gives the generated token count for a client-measured decode speed
            body['stream_options'] = dict(include_usage=True)
            return self._stream(body, on_delta)
        result = self.request(self.base + '/chat/completions', body)
        self.last_usage = result.get('usage') or {}
        self.last_speed = None     # no decode timing without streamed pieces: nothing is reported rather than a wrong number
        try:
            m = result['choices'][0]['message']
            if result['choices'][0].get('finish_reason') == 'length':
                raise ModelTruncated(m)
            return self._message(m)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise ModelError('invalid model response: ' + str(e)) from e

    def _message(self, m):
        """the reply in the shape the store keeps: validated tool calls, no unknown keys"""
        text, thinking = split_content(m.get('content'))
        out = dict(role='assistant', content=text)
        if m.get('reasoning_content') or thinking:
            out['reasoning_content'] = (m.get('reasoning_content') or '') + thinking
        calls = m.get('tool_calls') or []
        if calls:
            ids = [c['id'] for c in calls]
            if len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
                raise ValueError('tool call ids must be nonempty and unique')
            for c in calls:
                if not isinstance(c['function']['name'], str) or not isinstance(c['function']['arguments'], str):
                    raise ValueError('invalid tool call')
            out['tool_calls'] = [dict(id=c['id'], type='function', function=c['function']) for c in calls]
        if not out['content'] and not calls:
            raise ValueError('empty assistant response')
        return out

    def _stream(self, body, on_delta):
        """read the SSE reply, hand each piece to on_delta, assemble the message complete() would return.
        A backend that answers with one whole JSON body instead of a stream is accepted and reported once."""
        req = urllib.request.Request(self.base + '/chat/completions', json.dumps(body).encode(),
                                     self.headers(stream=True))
        content, reasoning, calls, finish, usage = [], [], {}, None, {}
        self.last_speed = None
        t_first = t_last = None    # decode window: first choice-chunk to the last chunk read, prefill excluded
        done_seen = False
        try:
            with self.opener(req, timeout=float(self.config.get('timeout', 900))) as r:
                for raw in r:
                    line = raw.decode('utf-8', 'replace').strip()
                    if not line or line.startswith(':'):
                        continue
                    if not line.startswith('data:'):
                        if not line.startswith('{'):
                            continue
                        whole = json.loads(line)      # not SSE: the whole reply at once
                        m = whole['choices'][0]['message']
                        text, thinking = split_content(m.get('content'))
                        for kind, value in (('reasoning', (m.get('reasoning_content') or '') + thinking), ('content', text)):
                            if value and on_delta(kind, value) is False:
                                raise Cancelled()
                        for call in m.get('tool_calls') or []:
                            if on_delta('tool', call['function']['name']) is False:
                                raise Cancelled()
                        self.last_usage = whole.get('usage') or {}
                        if whole['choices'][0].get('finish_reason') == 'length':
                            raise ModelTruncated(m)
                        return self._message(m)
                    data = line[5:].strip()
                    if data == '[DONE]':
                        done_seen = True
                        break
                    chunk = json.loads(data)
                    now = time.monotonic()
                    if chunk.get('choices') and t_first is None:
                        t_first = now
                    t_last = now
                    if chunk.get('usage'):
                        usage = chunk['usage']
                    for choice in chunk.get('choices') or []:
                        if choice.get('finish_reason'):
                            finish = choice['finish_reason']
                        delta = choice.get('delta') or {}
                        if delta.get('reasoning_content'):
                            reasoning.append(delta['reasoning_content'])
                            if on_delta('reasoning', delta['reasoning_content']) is False:
                                raise Cancelled()
                        if delta.get('content'):
                            text, thinking = split_content(delta['content'])   # (Mistral: typed blocks)
                            if thinking:
                                reasoning.append(thinking)
                                if on_delta('reasoning', thinking) is False:
                                    raise Cancelled()
                            if text:
                                content.append(text)
                                if on_delta('content', text) is False:
                                    raise Cancelled()
                        for call in delta.get('tool_calls') or []:
                            slot = calls.setdefault(call.get('index', 0), dict(id='', name='', arguments=''))
                            if call.get('id'):
                                slot['id'] = call['id']
                            fn = call.get('function') or {}
                            if fn.get('name'):
                                first = not slot['name']
                                slot['name'] += fn['name']
                                if first and on_delta('tool', slot['name']) is False:
                                    raise Cancelled()
                            if fn.get('arguments'):
                                slot['arguments'] += fn['arguments']
        except Cancelled:
            raise
        except urllib.error.HTTPError as e:
            raise http_error(e) from e
        except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
            raise ModelError(str(e)) from e
        self.last_usage = usage
        tokens = usage.get('completion_tokens') if isinstance(usage, dict) else None
        if isinstance(tokens, int) and tokens >= SPEED_MIN_TOKENS and t_first and t_last - t_first >= SPEED_MIN_S:
            self.last_speed = dict(tokens=tokens, seconds=round(t_last - t_first, 3),
                                   tps=round(tokens / (t_last - t_first), 1))
        try:
            out = dict(role='assistant', content=''.join(content))
            if reasoning:
                out['reasoning_content'] = ''.join(reasoning)
            if finish is None and not done_seen:
                # The connection closed before the turn ended: a proxy or backend cut the stream. The text
                # so far is shown with the error, and nothing of this reply is stored or executed.
                raise ModelTruncated(out, 'the reply ended before it was complete; nothing was executed',
                                     retryable=True)
            if finish == 'length':
                raise ModelTruncated(out)
            ordered = [calls[i] for i in sorted(calls)]
            if ordered:
                return self._message(dict(out, tool_calls=[
                    dict(id=c['id'], type='function', function=dict(name=c['name'], arguments=c['arguments']))
                    for c in ordered]))
            return self._message(out)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise ModelError('invalid model response: ' + str(e)) from e

    def tokens(self, messages, tools):
        if self.counter and self.api == 'anthropic':
            try:
                body = self._to_anthropic(messages, tools)
                body.pop('max_tokens', None)
                value = self.request(self.base + '/messages/count_tokens', body)['input_tokens']
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError('invalid token count')
                return value
            except (ModelError, KeyError, ValueError, TypeError) as e:
                if getattr(e, 'status', 0) in (401, 402, 403):
                    raise              # a missing or refused key is not a counting problem
                self.counter = False
        if self.counter:
            try:
                system, converted = [], []
                for m in messages:
                    if m['role'] == 'system':
                        system.append(m['content'])
                        continue
                    if m['role'] == 'tool':
                        converted.append(dict(role='user', content=[dict(type='tool_result', tool_use_id=m['tool_call_id'], content=m['content'])]))
                        continue
                    blocks = [dict(type='text', text=m['content'])] if m.get('content') else []
                    for call in m.get('tool_calls') or []:
                        f = call['function']
                        try:
                            args = json.loads(f['arguments'])
                        except ValueError:
                            args = dict(raw_arguments=f['arguments'])
                        blocks.append(dict(type='tool_use', id=call['id'], name=f['name'], input=args))
                    converted.append(dict(role=m['role'], content=blocks))
                ts = [dict(name=t['function']['name'], description=t['function']['description'], input_schema=t['function']['parameters']) for t in tools]
                result = self.request(self.base + '/messages/count_tokens', dict(model=self.config['model'], system='\n'.join(system), messages=converted, tools=ts))
                value = result['input_tokens']
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError('invalid token count')
                return value
            except (ModelError, KeyError, ValueError):
                # Other compatible providers need not implement Caminus's counter.
                # UTF8 bytes plus framing is conservative for byte-based tokenizers.
                self.counter = False
        return len(json.dumps(dict(messages=messages, tools=tools), ensure_ascii=False).encode()) + 256 * len(messages)

    # ---- Anthropic Messages API ---------------------------------------------------------------------------

    def _to_anthropic(self, messages, tools, model=None):
        """the resident's OpenAI-shaped request as a Messages API body.

        Prompt caching: tools and the leading system prompt are the stable prefix (breakpoint 1 on the system
        block); the newest stored message carries breakpoint 2, so each turn reads the whole history from the
        cache and writes only what it added. The trailing system message (memory, loop state) is rebuilt every
        turn and therefore sits after both breakpoints, as a mid-conversation system message."""
        system, rest = [], list(messages)
        while rest and rest[0]['role'] == 'system':
            system.append(rest.pop(0)['content'])
        tail = rest.pop()['content'] if rest and rest[-1]['role'] == 'system' else None
        out = []

        def add(role, blocks):
            if not blocks:
                return
            if out and out[-1]['role'] == role:      # steers and tool results arrive as separate messages
                out[-1]['content'].extend(blocks)
            else:
                out.append(dict(role=role, content=list(blocks)))

        for m in rest:
            role = m['role']
            if role == 'system':                     # a mid-history system message: kept as an operator note
                add('user', [dict(type='text', text='[system] ' + m['content'])])
            elif role == 'tool':
                add('user', [dict(type='tool_result', tool_use_id=m['tool_call_id'], content=m['content'] or '')])
            elif role == 'assistant':
                blocks = [dict(type='text', text=m['content'])] if m.get('content') else []
                for call in m.get('tool_calls') or []:
                    f = call['function']
                    try:
                        args = json.loads(f['arguments']) if f['arguments'].strip() else {}
                    except ValueError:
                        args = dict(raw_arguments=f['arguments'])
                    if not isinstance(args, dict):
                        args = dict(value=args)
                    blocks.append(dict(type='tool_use', id=call['id'], name=f['name'], input=args))
                add('assistant', blocks)
            else:
                add('user', [dict(type='text', text=m['content'] or ' ')])
        if out and out[0]['role'] != 'user':
            out.insert(0, dict(role='user', content=[dict(type='text', text='(conversation resumed)')]))
        if out:
            if self.config.get('cache', True):
                out[-1]['content'][-1]['cache_control'] = dict(type='ephemeral')
        if tail:
            if out and out[-1]['role'] == 'user':
                out.append(dict(role='system', content=tail))
            else:
                # a system message must follow a user turn, and a final assistant turn would be a prefill
                add('user', [dict(type='text', text='[system] ' + tail)])
        elif out and out[-1]['role'] == 'assistant':
            add('user', [dict(type='text', text='Continue.')])
        body = dict(model=model or self.config['model'], messages=out)
        if system:
            body['system'] = [dict(type='text', text='\n\n'.join(system))]
            if self.config.get('cache', True):     # (a one-shot call, like the advisor's, would pay the write premium for nothing)
                body['system'][0]['cache_control'] = dict(type='ephemeral')
        if tools:
            body['tools'] = [dict(name=t['function']['name'], description=t['function'].get('description', ''),
                                  input_schema=t['function']['parameters']) for t in tools]
        # Thinking 'disabled' is accepted up to effort high, so the effort is set rather than defaulted. Adaptive (the
        # default) leaves it to the model; the reply's thinking blocks are skipped and never replayed: the API
        # accepts a history with all of them removed (only removing some from the middle is an edit), so the model
        # thinks afresh every turn and the history stays valid.
        # Adaptive by default (the switch-off is simply not sent: current models then think as they see fit, older ones
        # do not think at all, so no model refuses it). On a CTF benchmark the same agent solved 10 of 14 tasks with
        # thinking and 2 without; Qwen Code sends adaptive thinking too. thinking 'disabled' in the config turns it off.
        if self.config.get('thinking', 'adaptive') == 'disabled':
            body['thinking'] = dict(type='disabled')
        choice = effort.resolve(self.config)
        if choice['thinking_off']:
            body['thinking'] = dict(type='disabled')           # (effort none: sent with effort high, effort.py)
        body['output_config'] = dict(effort=choice['anthropic_effort'] or 'medium')
        if self.config.get('fallbacks') and body['model'] in FALLBACK_MODELS:
            # A safety classifier can decline a legitimate request (cyber work on one's own machine trips it); the
            # server then re-runs it on a fallback model in the same call. Off for models that do not accept it.
            body['fallbacks'] = self.config['fallbacks']
        return body

    def _anthropic(self, messages, tools, max_tokens, model, on_delta):
        body = self._to_anthropic(messages, tools, model)
        body['max_tokens'] = max_tokens
        body['stream'] = True
        # A short side call (the finish judge's 600 tokens, a fresh look, a theme) answers without thinking: thinking
        # shares the output budget, and with it on the judge's verdict came back empty in 20 of 54 calls on a CTF
        # benchmark run (each one a finish let through unchecked). Main turns keep their thinking.
        if (max_tokens or 0) < SHORT_CALL and 'thinking' not in body and body['model'] not in THINKING_REQUIRED:
            body['thinking'] = dict(type='disabled')
        if body.get('thinking') == dict(type='disabled') and body['output_config']['effort'] not in ('low', 'medium', 'high'):
            body['output_config']['effort'] = 'high'   # thinking off is accepted up to high: the nearest level that runs
        return self._anthropic_stream(body, on_delta or (lambda kind, value: None))

    def _anthropic_stream(self, body, on_delta):
        """read a Messages API event stream into the reply shape complete() returns"""
        req = urllib.request.Request(self.base + '/messages', json.dumps(body).encode(),
                                     self.headers(stream=True, beta=FALLBACK_BETA if 'fallbacks' in body else None))
        text, blocks, stop, usage, details = [], {}, None, {}, None
        served, declined, fallback_ran, request_id = body.get('model'), [], False, None
        self.last_call = dict(model=body.get('model'))
        self.last_speed = None
        t_first = t_last = None
        done = False
        try:
            with self.opener(req, timeout=float(self.config.get('timeout', 900))) as r:
                request_id = (getattr(r, 'headers', None) or {}).get('request-id')
                for raw in r:
                    line = raw.decode('utf-8', 'replace').strip()
                    if not line.startswith('data:'):
                        continue          # 'event:' names repeat the data's type; blank lines end an event
                    ev = json.loads(line[5:].strip())
                    kind = ev.get('type')
                    now = time.monotonic()
                    if kind == 'message_start':
                        usage.update((ev.get('message') or {}).get('usage') or {})
                        served = (ev.get('message') or {}).get('model') or served
                    elif kind == 'content_block_start':
                        # the decode window opens with the first block, not its first delta: a thinking block whose
                        # text is not shown streams only its signature at the end, yet its tokens are in output_tokens
                        # (Haiku 5.5 with adaptive thinking read 1281 tok/s: all its thinking over the answer's time)
                        if t_first is None:
                            t_first = now
                        block = ev['content_block']
                        if block.get('type') == 'fallback':
                            # a switch point: the model named in 'from' declined and the one in 'to' continued
                            fallback_ran = True
                            declined.append(((block.get('from') or {}).get('model')))
                            served = (block.get('to') or {}).get('model') or served
                        blocks[ev['index']] = slot = dict(type=block['type'], id=block.get('id', ''),
                                                          name=block.get('name', ''), json='')
                        if slot['type'] == 'tool_use' and on_delta('tool', slot['name']) is False:
                            raise Cancelled()
                    elif kind == 'content_block_delta':
                        if t_first is None:
                            t_first = now
                        t_last = now
                        delta = ev['delta']
                        if delta.get('type') == 'text_delta':
                            text.append(delta['text'])
                            if on_delta('content', delta['text']) is False:
                                raise Cancelled()
                        elif delta.get('type') == 'input_json_delta':
                            blocks[ev['index']]['json'] += delta.get('partial_json', '')
                    elif kind == 'message_delta':
                        stop = (ev.get('delta') or {}).get('stop_reason') or stop
                        details = (ev.get('delta') or {}).get('stop_details') or details
                        usage.update(ev.get('usage') or {})
                        if any((i or {}).get('type') == 'fallback_message' for i in (usage.get('iterations') or [])):
                            fallback_ran = True
                    elif kind == 'message_stop':
                        done = True
                        break
                    elif kind == 'error':
                        err = ev.get('error') or {}
                        e = ModelError('%s: %s' % (err.get('type', 'error'), err.get('message', '')))
                        e.request_id = request_id
                        raise e
        except (Cancelled, ModelError):
            raise
        except urllib.error.HTTPError as e:
            raise http_error(e) from e
        except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
            raise ModelError(str(e)) from e
        self.last_call = dict(model=body.get('model'), served=served, fallback_ran=fallback_ran,
                              declined_by=[m for m in declined if m], stop_reason=stop,
                              stop_category=(details or {}).get('category'), request_id=request_id)
        read = usage.get('cache_read_input_tokens') or 0
        wrote = usage.get('cache_creation_input_tokens') or 0
        out_tokens = usage.get('output_tokens') or 0
        self.last_usage = dict(prompt_tokens=(usage.get('input_tokens') or 0) + read + wrote,
                               completion_tokens=out_tokens, cache_read_input_tokens=read,
                               cache_creation_input_tokens=wrote)
        if out_tokens >= SPEED_MIN_TOKENS and t_first and t_last - t_first >= SPEED_MIN_S:
            self.last_speed = dict(tokens=out_tokens, seconds=round(t_last - t_first, 3),
                                   tps=round(out_tokens / (t_last - t_first), 1))
        out = dict(role='assistant', content=''.join(text))
        if not done:
            raise ModelTruncated(out, 'the reply ended before it was complete; nothing was executed', retryable=True)
        if stop == 'refusal':
            category = (details or {}).get('category')
            e = ModelRefused('the provider declined this request' + (' (%s)' % category if category else '')
                             + ('' if not fallback_ran else ' after a server-side fallback')
                             + '; rephrase the goal or use another model')
            e.category, e.request_id, e.call = category, request_id, self.last_call
            raise e
        if stop == 'max_tokens':
            raise ModelTruncated(out)
        calls = []
        for i in sorted(blocks):
            b = blocks[i]
            if b['type'] == 'tool_use':
                calls.append(dict(id=b['id'], type='function',
                                  function=dict(name=b['name'], arguments=b['json'] or '{}')))
        try:
            return self._message(dict(out, tool_calls=calls) if calls else out)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise ModelError('invalid model response: ' + str(e)) from e
