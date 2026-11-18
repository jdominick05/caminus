"""Model transports: OpenAI-compatible chat completions (the local server) and Anthropic's Messages API.
Standard library only; no engine, GPU or agent-CLI dependency."""
import base64
import email.utils
import hashlib
import http.client
import json
import math
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from ... import __version__
from . import catalog, effort


# a speed is measured only over a window this long with this many tokens: a reply that arrived in one burst (Haiku
# with thinking) divided its tokens by microseconds and the status line read 31,746,037 tok/s
SPEED_MIN_S = 0.25
SPEED_MIN_TOKENS = 16
SHORT_CALL = 4096          # a call with less output room than this runs without thinking (see _anthropic)
# models that refuse thinking off (current Opus and Sonnet; the advisor runs on them): short calls keep thinking there
THINKING_REQUIRED = frozenset(('claude-opus-5-5', 'claude-opus-5', 'claude-sonnet-5-5', 'claude-sonnet-5',
                               'claude-fable-5-2', 'claude-fable-5-1', 'claude-fable-5'))


def estimated_speed(tokens, t_first, t_last):
    """A reply's decode speed measured here: its usage token count over the client's window from the first piece to
    the last, prefill and queueing excluded; None when the window is too short to mean anything. Only for a brain whose
    server reports no speed of its own (engine_speed)."""
    if isinstance(tokens, int) and tokens >= SPEED_MIN_TOKENS and t_first and t_last - t_first >= SPEED_MIN_S:
        return dict(source='estimate', tokens=tokens, seconds=round(t_last - t_first, 3),
                    tps=round(tokens / (t_last - t_first), 1))
    return None


def engine_speed(timings, server):
    """A reply's speeds as a Caminus server measured them (its `timings`, as sent: predicted_per_second for writing,
    prompt_per_second for reading the prompt's uncached part); None when it reports neither. A 0 is not a speed (a
    prompt read wholly from the cache reads nothing) and is left out rather than shown as 0 tok/s."""
    if not isinstance(timings, dict):
        return None
    def rate(key):
        v = timings.get(key)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else None
    tps, prompt_tps = rate('predicted_per_second'), rate('prompt_per_second')
    if tps is None and prompt_tps is None:
        return None
    return dict(source='engine', server=server, tps=tps, prompt_tps=prompt_tps, tokens=timings.get('predicted_n'),
                prompt_n=timings.get('prompt_n'), cache_n=timings.get('cache_n'))


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


# how long a local brain's window is kept before its server is asked again (Model.context_size)
WINDOW_TTL = 60


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
# What an OpenCode plan is sent (catalog.OPENCODE): Ember's own name, not a library's (OpenCode Go asks clients to
# identify themselves), and the Messages API version its gateway was checked with (2026-11-18)
USER_AGENT = 'ember/' + __version__
OPENCODE_ANTHROPIC_VERSION = '2023-06-01'

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
# The Meta Model API's Muse chat models, in the PRICES row form; Meta bills no cache writes (the fourth rate never
# applies). Muse Ember's rates triple past 200K input tokens. Standard mode only: Contributor mode (the -contributor ids)
# is cheaper because Meta trains on its prompts, files and replies, so it has no row and the relay refuses it. From
# dev.meta.ai/docs/pricing-rate-limits as read 2026-11-17: check live pricing before budgeting on these numbers.
_SPARK = ((1.25, 4.25, 0.15, 1.25),) * 2 + (10**9,)
META_PRICES = {'muse-ember-1.0': ((0.10, 0.50, 0.004, 0.10), (0.30, 1.50, 0.012, 0.30), 200_000),
               'muse-spark-1.3': _SPARK, 'muse-spark-1.2': _SPARK, 'muse-spark-1.1': _SPARK,
               'muse-wonder-1.0': ((4.00, 20.00, 0.20, 4.00),) * 2 + (10**9,)}


# Prices a provider published (the model catalog: DeepInfra's per-model pricing), in the PRICES row form. Filled from
# a config's `prices` when the runtime starts, so a catalog model is metered like a built-in one.
CATALOG_PRICES = {}


def flat_row(inp, out, read):
    """one price at every prompt length (USD per million: input, output, cache read), in the PRICES row form"""
    return ((inp, out, read, inp),) * 2 + (10**9,)


def remember_prices(model, prices):
    """prices: {input, output, cache_read, cache_write?} in USD per million tokens, as the catalog stores them, with a
    long-context tier under `long` ({above, input, output, ...}: an OpenCode model's) where the provider lists one."""
    def rates(p):
        inp = float(p['input'])
        return (inp, float(p['output']), float(p['cache_read']) if p.get('cache_read') is not None else inp,
                float(p['cache_write']) if p.get('cache_write') is not None else inp)
    if model and prices and prices.get('input') is not None and prices.get('output') is not None:
        low, long = rates(prices), prices.get('long')
        try:
            CATALOG_PRICES[model] = (low, rates(long), int(long['above'])) if isinstance(long, dict) else (low, low, 10**9)
        except (KeyError, TypeError, ValueError):
            CATALOG_PRICES[model] = (low, low, 10**9)


def price_row(model):
    """the model's PRICES-form row (input, output, cache read, write; below and above a prompt-length line), or None"""
    return PRICES.get(model) or META_PRICES.get(model) or CATALOG_PRICES.get(model)


def price(model, usage, served=None):
    """the USD cost of one call's usage on a priced model, or None when the model's price is not known. When a fallback
    served the call (served differs from model), it is priced at the serving model's rates, or at the highest known
    rates when that model is not in the table: an estimate that errs high. Meta's published table, with Muse Ember's
    long-prompt tier, comes before the catalog's flat copy of its first tier.
    A reply that says what it charged (an OpenCode plan's usage.cost beside usage.opencode: 0 inside a Go allowance,
    dollars from the balance past it) costs that, whatever the price table says."""
    charged = (usage or {}).get('cost')
    if isinstance((usage or {}).get('opencode'), dict) and isinstance(charged, (int, float)) and not isinstance(charged, bool):
        return float(charged)
    row = price_row(model)
    if row is None:
        return None
    if served and served != model:
        row = PRICES.get(served) or max(PRICES.values(), key=lambda r: r[0][1])
    return row_cost(row, usage)


def row_cost(row, usage):
    """One call's USD cost by a PRICES-form row, from either usage style; the key relay's ledger prices with it too."""
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

# Images (files.read): a tool message names its image (store.action_result; a user message can too: the finish judge's),
# and each request sends the file itself, so the store keeps no bytes and a brain switched mid-goal gets what it can
# take: the image where its config says vision (the catalog's flag, copied at init and /model), a line saying it can't
# see where not, nothing more where the eyes model already described it in the result (advisor.Eyes).
IMAGES_PER_REQUEST = 20    # newest image results sent as images; 20 x 5 MB stays inside the API's 128 MB request
IMAGE_CACHE = 32           # encoded files kept per brain: compaction counts the same request many times
NO_VISION = ("this model can't see images; {name} is a {w}x{h} {fmt}; describe-by-tool options: discover topic images "
             "lists the image tools installed here (identify, exiftool, tesseract for text in it, ...) to run in the "
             "terminal, or ask your human")


def image_tokens(width, height):
    """An image's input tokens, estimated by Claude's rule (platform.claude.com vision): one token per 28x28-pixel patch,
    after a downscale to its high-resolution tier (2576 px long edge, 4784 patches). It stands in where a provider counts
    no tokens (Qwen-VL patches are 28 px too) and errs high for Claude's standard tier (1568 at most)."""
    s = min(1.0, 2576 / max(width, height, 1))
    return min(4784, math.ceil(width * s / 28) * math.ceil(height * s / 28))


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
    err.retry_after = retry_after(e.headers) or None
    return err


def retry_after(headers):
    """the seconds a Retry-After header asks for (a number or an HTTP date), else None"""
    value = (headers or {}).get('retry-after')
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        return max(0.0, email.utils.parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def _recording_opener(held):
    """urlopen that hands each connection's socket to held(sock) as it connects (Model.abort closes them)."""
    class Conn(http.client.HTTPConnection):
        def connect(self):
            super().connect()
            held(self.sock)

    class Conns(http.client.HTTPSConnection):
        def connect(self):
            super().connect()
            held(self.sock)

    class Plain(urllib.request.HTTPHandler):
        def http_open(self, req):
            return self.do_open(Conn, req)

    class Secure(urllib.request.HTTPSHandler):
        def https_open(self, req):
            return self.do_open(Conns, req, context=self._context)

    return urllib.request.build_opener(Plain, Secure).open


class Cancelled(Exception):
    """the goal was cancelled while the model was generating; nothing of this reply is stored"""


class Model:
    def __init__(self, config, opener=urllib.request.urlopen):
        self.config, self.opener = config, opener
        self._sockets, self._sockets_lock = set(), threading.Lock()
        if opener is urllib.request.urlopen:
            self.opener = _recording_opener(self._held)   # (abort() can then close an open call: see abort)

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
        if self.api == 'anthropic' and config.get('context') is None:
            raise ValueError('the anthropic api needs an explicit context window (--context)')
        self.context, self._context_at = None, 0.0   # the window context_size resolved, and when
        # an OpenCode plan: each request names its conversation (x-opencode-session: Go refuses a client without one)
        # and Ember (USER_AGENT); its errors are never retried in a loop (runtime) and nothing is asked that it may
        # not serve (no count_tokens call: the estimate counts)
        self.opencode = config.get('provider_kind') in catalog.OPENCODE
        self.counter = not self.opencode
        self.last_usage = {}
        self.last_speed = None   # the last reply's speed: engine_speed on a Caminus server, else estimated_speed
        self.caminus = None     # whether the endpoint is a Caminus server, once asked (is_caminus)
        self.last_call = {}      # the newest Messages API call: requested and serving model, fallback, stop, request id
        self._images = {}        # (path, mtime_ns, size) -> (sha256, base64): see _image_data

    def _held(self, sock):
        with self._sockets_lock:
            self._sockets = {s for s in self._sockets if s.fileno() != -1}   # (closed ones out)
            self._sockets.add(sock)

    def abort(self):
        """Close every open call's connection. A cancelled or interrupted turn stops here and on the server: abandoning
        the call left the server generating the reply (up to the reply limit) and holding its slot; caminus serve stops
        a job when its client's socket closes, also during prefill."""
        with self._sockets_lock:
            held, self._sockets = self._sockets, set()
        for s in held:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass                      # (already closed)
    def headers(self, stream=False, beta=None, session=None):
        """session: the conversation's id for an OpenCode plan (a goal's, kept with it); a call without one is a
        conversation of its own (a side call: the judge, a summary) and gets a fresh id"""
        name = self.config.get('api_key_env', 'CAMINUS_AGENT_API_KEY')
        key = os.environ.get(name)
        headers = {'Content-Type': 'application/json'}
        if stream:
            headers['Accept'] = 'text/event-stream'
        if self.opencode:
            headers['x-opencode-session'] = session or uuid.uuid4().hex
            headers['User-Agent'] = USER_AGENT
            if self.api == 'anthropic':
                headers['anthropic-version'] = OPENCODE_ANTHROPIC_VERSION
            headers['Authorization'] = 'Bearer ' + (key or 'caminus-relay')   # (Bearer on both APIs; the relay replaces it)
        elif self.api == 'anthropic':
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

    def request(self, url, body=None, session=None):
        req = urllib.request.Request(url, None if body is None else json.dumps(body).encode(), self.headers(session=session))
        try:
            with self.opener(req, timeout=float(self.config.get('timeout', 900))) as r:
                raw = r.read(16 * 1024 * 1024 + 1)
                if len(raw) > 16 * 1024 * 1024:
                    raise ModelError('model response exceeds16MiB')
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            raise http_error(e) from e
        except (OSError, ValueError) as e:
            raise self._failed(e) from e

    def probe(self, timeout=3.0):
        """whether the server answers at all (a local server's /props, a short timeout); a remote API is assumed up"""
        if self.remote or self.api == 'anthropic':
            return True
        try:
            with self.opener(urllib.request.Request(self.root + '/props', None, self.headers()), timeout=timeout) as r:
                r.read(65536)
            return True
        except (OSError, ValueError, urllib.error.HTTPError) as e:
            self._failed(e)
            return False

    def _failed(self, e):
        """A call's transport failure as a ModelError. A connection that failed may come back as another server on the
        same port (llama-server where caminus serve was): whether the brain is Caminus is then asked again."""
        if isinstance(e, OSError) and not isinstance(e, urllib.error.HTTPError):
            self.caminus = None
            self._context_at = 0.0   # (and its window is asked again: the server that comes back may run another mode)
        return ModelError(str(e))

    def is_caminus(self):
        """Whether this brain is a local Caminus server (catalog.caminus_props: its /props names the engine). Asked once
        and kept until a call fails to connect (_failed); a server that cannot be reached is asked again next time,
        never remembered as another engine. A paid brain (remote, or through the key relay) is never asked. Any failure
        is "not now": the reply it follows is already read, and a speed lookup must never fail it."""
        if self.caminus is None:
            if self.remote:
                self.caminus = False
            else:
                from . import catalog
                try:
                    self.caminus = catalog.caminus_props(self.base, self.opener, 3) is not None
                except Exception:
                    return False
        return self.caminus

    def _speed(self, timings, tokens=None, t_first=None, t_last=None):
        """The reply's speed: a Caminus server's own numbers whenever the brain is one (never an estimate there, even
        when it sent none), else the client's estimate."""
        if timings is not None and self.is_caminus():
            return engine_speed(timings, self.root)
        if self.caminus:
            return None
        return estimated_speed(tokens, t_first, t_last)

    def context_size(self):
        """The window (catalog.window): a local brain's from its server now, asked again every WINDOW_TTL seconds and
        after a failed connection, so a server restarted in another mode (262K, 1M) is measured as it runs; a paid
        brain's from its config. Nothing known: the endpoint's /props, whatever its engine (llama-server, a fixture)."""
        if self.context is None or not self.remote and time.monotonic() - self._context_at > WINDOW_TTL:
            from . import catalog
            served = None
            # (only a Caminus server, or provider_kind local, is asked each time: is_caminus remembers the answer, so a
            # key relay on loopback is not asked every minute)
            if not self.remote and catalog.window_candidate(self.config) and \
                    (self.config.get('provider_kind') == 'local' or self.is_caminus()):
                try:
                    served = catalog.local_window(self.config, self.opener, 3)
                except Exception:
                    served = None   # (a window lookup never fails a turn)
            if served is None and self.context is not None and not self.remote:
                window = self.context   # a lookup that learns nothing keeps what the server said: never the stale config
            else:
                window = catalog.window(self.config, served, not self.remote)
            if window is None:
                window = catalog.props_window(self.request(self.root + '/props'))
            self.context = window
            if not self.remote:
                self._context_at = time.monotonic()   # (a paid brain's window is its config's: never asked again)
        if isinstance(self.context, bool) or not isinstance(self.context, int) or self.context < 1024:
            self.context = None
            raise ModelError('backend must report a positive context window, or set --context explicitly')
        return self.context

    def reply_limit(self):
        """A reply's token limit (effort.reply_limit): config max_tokens; else a local brain the card's maximum output up
        to 64K, a paid one 4096 (its catalog maximum, up to 64K, is set when it is picked)."""
        return effort.reply_limit(self.config, self.remote)

    def complete(self, messages, tools=None, max_tokens=None, model=None, on_delta=None, session=None):
        """One reply. With on_delta(kind, value) the reply streams: kind is 'reasoning', 'content' or 'tool'
        (a tool call's name starts), value is the text or the name. on_delta returning False cancels: the
        stream is closed and Cancelled is raised, so a cancelled turn stores no reply and runs no tool.
        session: the conversation's id, sent to an OpenCode plan (headers)."""
        remaining = self.context_size() - self.tokens(messages, tools or []) - 256
        if remaining < 1:
            raise ModelContextError('request exceeds the model context; compact or shorten the goal')
        if self.api == 'anthropic':
            return self._anthropic(messages, tools, min(max_tokens or self.reply_limit(), remaining), model, on_delta,
                                   session)
        limit = min(max_tokens or self.reply_limit(), remaining)
        # the effort in force (effort.py): a local model takes it as the server's model id for the level, a provider
        # with reasoning_effort as that field; a short side call (the judge's) runs with thinking off
        choice = effort.resolve(self.config, short=limit < SHORT_CALL)
        body = dict(model=model or choice['model'] or self.config['model'], messages=self._wire(messages),
                    stream=on_delta is not None, max_tokens=limit)
        # sampling: none for a local brain (its server applies the model's own defaults: a fixed 0.6 overrode the
        # Qwen3.8 cards' 1.0), a paid model's card values when known, the config's own when set
        body.update(effort.sampling(self.config, self.remote))
        if choice['reasoning_effort']:
            body['reasoning_effort'] = choice['reasoning_effort']   # (DeepInfra, Mistral; 'none' turns it off)
        if choice['thinking_off']:
            body['chat_template_kwargs'] = dict(enable_thinking=False)   # a local server with no -nothink id
        if self.config.get('provider_kind') in ('mistral', 'meta'):
            # Mistral routes requests to its prompt cache by this key (a hint, not a promise): one stable key per agent,
            # since its turns share the system prompt and history as their prefix. Without it the cache hit 0 %. Meta
            # caches prefixes by itself and takes the same field (dev.meta.ai/docs/protocols/chat-completions).
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
            # usage in the final chunk gives the generated token count for a client-measured decode speed (a Caminus
            # server's own timings replace it: _speed)
            body['stream_options'] = dict(include_usage=True)
            return self._stream(body, on_delta, session)
        result = self.request(self.base + '/chat/completions', body, **({'session': session} if session else {}))
        self.last_usage = result.get('usage') or {}
        # no decode window without streamed pieces: only a server's own timings give a speed here
        self.last_speed = self._speed(result.get('timings'))
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

    def _stream(self, body, on_delta, session=None):
        """read the SSE reply, hand each piece to on_delta, assemble the message complete() would return.
        A backend that answers with one whole JSON body instead of a stream is accepted and reported once."""
        req = urllib.request.Request(self.base + '/chat/completions', json.dumps(body).encode(),
                                     self.headers(stream=True, session=session))
        content, reasoning, calls, finish, usage, timings = [], [], {}, None, {}, None
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
                        self.last_speed = self._speed(whole.get('timings'))
                        if whole['choices'][0].get('finish_reason') == 'length':
                            raise ModelTruncated(m)
                        return self._message(m)
                    data = line[5:].strip()
                    if data == '[DONE]':
                        done_seen = True
                        break
                    chunk = json.loads(data)
                    now = time.monotonic()
                    # the window opens at the first piece with text or a call in it: a role-only chunk first, then
                    # reasoning a provider does not stream (Meta), would count that hidden time as writing
                    if t_first is None and any(((c or {}).get('delta') or {}).get(k) for c in chunk.get('choices') or []
                                               for k in ('content', 'reasoning_content', 'tool_calls')):
                        t_first = now
                    t_last = now
                    if chunk.get('usage'):
                        usage = chunk['usage']
                    if isinstance(chunk.get('timings'), dict):
                        timings = chunk['timings']    # (caminus serve: on the finish chunk, before the usage chunk)
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
            raise self._failed(e) from e
        self.last_usage = usage
        written = usage.get('completion_tokens') if isinstance(usage, dict) else None
        hidden = ((usage.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0) if isinstance(usage, dict) else 0
        if isinstance(written, int) and not reasoning and isinstance(hidden, int) and 0 < hidden <= written:
            written -= hidden      # reasoning billed but not streamed (Meta): not written inside the window
        self.last_speed = self._speed(timings, written, t_first, t_last)
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
                # (without the image blocks: compaction counts one request many times, and each count would upload
                # every image again; their cost is added once, by the same patch rule the other counters use)
                body = self._to_anthropic([{k: v for k, v in m.items() if k != 'image'} for m in messages], tools)
                body.pop('max_tokens', None)
                value = self.request(self.base + '/messages/count_tokens', body)['input_tokens']
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError('invalid token count')
                return value + self._image_cost(messages)
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
                return value + self._image_cost(messages)   # (the converted messages carry the images' text only)
            except (ModelError, KeyError, ValueError):
                # Other compatible providers need not implement Caminus's counter.
                # UTF8 bytes plus framing is conservative for byte-based tokenizers.
                self.counter = False
        return (len(json.dumps(dict(messages=messages, tools=tools), ensure_ascii=False).encode()) + 256 * len(messages)
                + self._image_cost(messages))

    # ---- images ------------------------------------------------------------------------------------------------

    def _image_data(self, img):
        """the image file as base64 while it still is the file that was read (its sha256), else None"""
        try:
            st = os.stat(img['path'])
            key = (img['path'], st.st_mtime_ns, st.st_size)
            if key not in self._images:
                with open(img['path'], 'rb') as f:
                    raw = f.read()
                if len(self._images) >= IMAGE_CACHE:
                    self._images.clear()
                self._images[key] = (hashlib.sha256(raw).hexdigest(), base64.b64encode(raw).decode())
            sha, data = self._images[key]
            return data if sha == img.get('sha256') else None
        except (OSError, KeyError, TypeError):
            return None

    def _image_plan(self, messages):
        """{index: (base64 data or None, a line for the text or None)} for each tool or user message naming an image"""
        found = [i for i, m in enumerate(messages)
                 if m.get('role') in ('tool', 'user') and isinstance(m.get('image'), dict)]
        recent, plan = set(found[-IMAGES_PER_REQUEST:]), {}
        for i in found:
            img = messages[i]['image']
            name = os.path.basename(str(img.get('path') or '')) or 'the file'
            if not self.config.get('vision'):
                plan[i] = (None, None if img.get('described_by') else NO_VISION.format(
                    name=name, w=img.get('width'), h=img.get('height'), fmt=str(img.get('format') or '').upper()))
            elif 'sha256' not in img:
                plan[i] = (None, None)                       # past read's caps: the result's note says so
            elif i not in recent:
                plan[i] = (None, f'[{name}: not sent again; a request carries the newest {IMAGES_PER_REQUEST} images: '
                                 'read it again to see it]')
            else:
                data = self._image_data(img)
                plan[i] = (data, None) if data else (None, f'[{name} changed or is gone since it was read: no image '
                                                           'sent; read it again]')
        return plan

    def _image_cost(self, messages):
        return sum(image_tokens(int(messages[i]['image']['width']), int(messages[i]['image']['height']))
                   for i, (data, _) in self._image_plan(messages).items() if data)

    def _wire(self, messages):
        """The messages as an OpenAI-compatible request sends them. A Chat Completions tool message carries text only
        (OpenAI's spec: ChatCompletionToolMessageParam takes a string or text parts), so an image goes in a user message
        after its turn's last tool result, never between parallel results (a user message's image joins its own text); a
        message without an image is sent as stored."""
        plan = self._image_plan(messages)
        if not plan:
            return self._system_as_user(messages)
        out, parts = [], []
        for i, m in enumerate(messages):
            if parts and m.get('role') != 'tool':
                out.append(dict(role='user', content=parts))
                parts = []
            if i in plan:
                img, (data, line) = m['image'], plan[i]
                m = {k: v for k, v in m.items() if k != 'image'}
                if data and m['role'] == 'user':
                    m['content'] = [dict(type='text', text=m.get('content') or ''), dict(
                        type='image_url', image_url=dict(url=f"data:image/{img['format']};base64,{data}"))]
                elif data:
                    parts += [dict(type='text', text=f"{img['path']} (the read in call {m.get('tool_call_id')}):"),
                              dict(type='image_url', image_url=dict(url=f"data:image/{img['format']};base64,{data}"))]
                elif line:
                    m['content'] = (m.get('content') or '') + '\n' + line
            out.append(m)
        if parts:
            out.append(dict(role='user', content=parts))
        return self._system_as_user(out)

    def _system_as_user(self, messages):
        """Meta merges every system message into the front instructions, so a trailing one (the runtime block, new each
        turn) changed the prompt's start and voided its prefix cache on every turn. For Meta, each system message after
        the leading ones goes as a user message marked [system], as the Messages API transport sends it, and after the
        tool results its assistant turn still waits for (Meta refuses a message between a call and its results). Every
        other provider gets the messages as they are."""
        if self.config.get('provider_kind') != 'meta':
            return messages
        out, held, waiting, lead = [], [], set(), True
        for m in messages:
            lead = lead and m['role'] == 'system'
            if lead:
                out.append(m)
                continue
            if m['role'] == 'system':
                (held if waiting else out).append(dict(role='user', content='[system] ' + str(m.get('content') or '')))
                continue
            if m['role'] == 'assistant':
                waiting = {c['id'] for c in m.get('tool_calls') or []}
            elif m['role'] == 'tool':
                waiting.discard(m.get('tool_call_id'))
            out.append(m)
            if held and not waiting:
                out, held = out + held, []
        return out + held

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
        out, images = [], self._image_plan(rest)

        def with_image(i, text):
            """a message's text, as blocks with its image after it, or with the line that says why none is sent"""
            data, line = images.get(i, (None, None))
            if data:
                return [dict(type='text', text=text), dict(type='image', source=dict(
                    type='base64', media_type='image/' + rest[i]['image']['format'], data=data))]
            return text + '\n' + line if line else text

        def add(role, blocks):
            if not blocks:
                return
            if out and out[-1]['role'] == role:      # steers and tool results arrive as separate messages
                out[-1]['content'].extend(blocks)
            else:
                out.append(dict(role=role, content=list(blocks)))

        for i, m in enumerate(rest):
            role = m['role']
            if role == 'system':                     # a mid-history system message: kept as an operator note
                add('user', [dict(type='text', text='[system] ' + m['content'])])
            elif role == 'tool':                     # (a tool_result's content may hold image blocks)
                add('user', [dict(type='tool_result', tool_use_id=m['tool_call_id'],
                                  content=with_image(i, m['content'] or ''))])
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
                blocks = with_image(i, m['content'] or ' ')
                add('user', blocks if isinstance(blocks, list) else [dict(type='text', text=blocks)])
        if out and out[0]['role'] != 'user':
            out.insert(0, dict(role='user', content=[dict(type='text', text='(conversation resumed)')]))
        if out:
            if self.config.get('cache', True):
                out[-1]['content'][-1]['cache_control'] = dict(type='ephemeral')
        if tail:
            if out and out[-1]['role'] == 'user' and not self.opencode:
                # (not to an OpenCode plan: its Messages API is checked at the 2023-06-01 version, which has no
                #  mid-conversation system message; there the block joins the last user turn as text)
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
        choice = effort.resolve(self.config)
        body['output_config'] = dict(effort=choice['anthropic_effort'] or 'medium')
        if choice['thinking_off'] or self.config.get('thinking', 'adaptive') == 'disabled':
            effort.anthropic_thinking_off(body)   # each model's own way (effort.THINKING_OFF): a wrong one is a 400
        if self.config.get('fallbacks') and body['model'] in FALLBACK_MODELS:
            # A safety classifier can decline a legitimate request (cyber work on one's own machine trips it); the
            # server then re-runs it on a fallback model in the same call. Off for models that do not accept it.
            body['fallbacks'] = self.config['fallbacks']
        return body

    def _anthropic(self, messages, tools, max_tokens, model, on_delta, session=None):
        body = self._to_anthropic(messages, tools, model)
        body['max_tokens'] = max_tokens
        body['stream'] = True
        # A short side call (the finish judge's 600 tokens, a fresh look, a theme) answers without thinking: thinking
        # shares the output budget, and with it on the judge's verdict came back empty in 20 of 54 calls on a CTF
        # benchmark run (each one a finish let through unchecked). Main turns keep their thinking.
        if (max_tokens or 0) < SHORT_CALL:
            effort.anthropic_thinking_off(body)   # (Opus 5.5 and Fable cannot: they answer at effort low)
        have = effort.efforts(self.config) if self.opencode else None
        if have is not None and 'effort' in (body.get('output_config') or {}):
            # an OpenCode model on Messages takes only the levels its list names (Qwen3.8: low, medium, xhigh), and a
            # model that lists none (MiniMax M3) is sent none
            if have:
                body['output_config']['effort'] = effort.nearest(body['output_config']['effort'], have)
            else:
                body.pop('output_config')
        return self._anthropic_stream(body, on_delta or (lambda kind, value: None), session)

    def _anthropic_stream(self, body, on_delta, session=None):
        """read a Messages API event stream into the reply shape complete() returns"""
        req = urllib.request.Request(self.base + '/messages', json.dumps(body).encode(),
                                     self.headers(stream=True, beta=FALLBACK_BETA if 'fallbacks' in body else None,
                                                  session=session))
        text, blocks, stop, usage, details, timings = [], {}, None, {}, None, None
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
                        elif delta.get('type') == 'thinking_delta' and delta.get('thinking'):
                            # shown live like a local model's reasoning (it is not kept in the reply); each piece
                            # is also a point where a cancel or an Esc stops the call
                            if on_delta('reasoning', delta['thinking']) is False:
                                raise Cancelled()
                        elif delta.get('type') == 'input_json_delta':
                            blocks[ev['index']]['json'] += delta.get('partial_json', '')
                    elif kind == 'message_delta':
                        stop = (ev.get('delta') or {}).get('stop_reason') or stop
                        details = (ev.get('delta') or {}).get('stop_details') or details
                        usage.update(ev.get('usage') or {})
                        if isinstance(ev.get('timings'), dict):
                            timings = ev['timings']   # (caminus serve's Anthropic route: on message_delta)
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
            raise self._failed(e) from e
        self.last_call = dict(model=body.get('model'), served=served, fallback_ran=fallback_ran,
                              declined_by=[m for m in declined if m], stop_reason=stop,
                              stop_category=(details or {}).get('category'), request_id=request_id)
        read = usage.get('cache_read_input_tokens') or 0
        wrote = usage.get('cache_creation_input_tokens') or 0
        out_tokens = usage.get('output_tokens') or 0
        self.last_usage = dict(prompt_tokens=(usage.get('input_tokens') or 0) + read + wrote,
                               completion_tokens=out_tokens, cache_read_input_tokens=read,
                               cache_creation_input_tokens=wrote)
        for k in ('cost', 'opencode'):      # what an OpenCode plan charged (price() meters it)
            if k in usage:
                self.last_usage[k] = usage[k]
        self.last_speed = self._speed(timings, out_tokens, t_first, t_last)
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
