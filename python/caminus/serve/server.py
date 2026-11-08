"""K3 HTTP server (server spec section 2, step S4): stdlib ThreadingHTTPServer, SSE, OpenAI and Anthropic routes.

Routes
  POST /v1/chat/completions          OpenAI Chat Completions, stream and not: tools, tool_choice, parallel_tool_calls,
                                     stop, max_tokens / max_completion_tokens, temperature, top_p, top_k, min_p,
                                     penalties, seed, stream_options.include_usage, reasoning_content,
                                     chat_template_kwargs, reasoning_effort
  POST /v1/completions               raw prompt (string or token array; 248044 is put first when missing)
  GET  /v1/models                    the model ids (MODEL_PROFILES)
  POST /v1/messages                  Anthropic Messages, stream and not: thinking blocks, tool_use / tool_result,
                                     tool_choice, stop_sequences, stop_reason, usage, pings
  POST /v1/messages/count_tokens     {"input_tokens": n}
  GET  /health, /props, /slots, /metrics
                                     llama-server's monitoring surface, its names and shapes (llm-top reads it):
                                     /health 200 {"status": "ok"} or 503 'Loading model' while loading; /props with
                                     an engine marker ("engine": "caminus"); /metrics with the llamacpp: series and
                                     the caminus: ones (docs/specs/spec-metrics.md)
  GET  /caminus/status               the versioned JSON status: every metric group, histograms, per-card data
(the same paths without /v1 are accepted for the OpenAI routes, as llama-server does.)

Flow. The HTTP thread validates and normalizes the request, renders and tokenizes the prompt (template.PromptBuilder;
Rust tokenization releases the GIL), and submits a worker.Job. The engine thread (worker.EngineWorker) runs one job
at a time, FIFO, and sends back released text; the HTTP thread runs the output parser (toolparse) and writes the
OpenAI / Anthropic shapes (deltas.py). Stop strings are cut on the engine thread so the generation ends at the exact
token (generate.py).

Choices
  * Tool calls are streamed whole by default: a call's deltas are held until the call closes, and a call the
    generation cut (max_tokens or a stop string ended it before the call's </function>, whether inside a value,
    between parameters or right after the name) is never sent, in stream or non-stream responses, so a client
    cannot run truncated arguments or a call missing the parameters the model was still writing
    (_withhold_unclosed; an unclosed call at the model's own EOS is S2's case: reported when its values are
    whole). `incremental_tool_args=True` streams argument pieces as they are parsed
    (llama-server's behaviour); a cut call's deltas are then already out, but the finish / stop reason is still not
    tool_calls / tool_use.
  * tool_choice none: tools still render (the system prompt stays cacheable), tool-call text is content.
    required / named (Anthropic any / tool): after </think> the server writes '\\n\\n<tool_call>\\n' (plus
    '<function=NAME>\\n' when named) as the model's own tokens; with thinking off they end the prompt instead.
  * Anthropic usage: input_tokens = prompt - cached, cache_read_input_tokens = cached (llama-server's convention;
    clients add the two). OpenAI usage.prompt_tokens_details.cached_tokens = cached.
  * timings (llama.cpp names plus queue_ms, ttft_ms, passes, drafted, accepted, seed) are on the final OpenAI chunk /
    response and on the Anthropic message_delta / response; with log_dir every request also writes a JSONL row there (an
    access log, off by default: caminus serve's per-request log is the flight recorder's line).
  * Context limit: prompt tokens + 1 > max_ctx is a 400 ('prompt is too long' for Anthropic, code
    context_length_exceeded for OpenAI); max_tokens is clamped to what fits.
  * Engine errors (EngineError): the session is reset; 503 (OpenAI) / 529 overloaded_error (Anthropic) before the
    headers, an SSE error event after them. A full queue answers the same way.
  * Internal errors (any other exception on the engine thread, a bug): 500 'internal error: ...'. The session is
    not reset: EngineSession.abort() rolls back an unaccepted pass and keeps the committed state and every
    checkpoint; only when abort() fails is the session reset.
  * Request validation: the JSON body may not contain NaN / Infinity (nor numbers that overflow to them) or lone
    UTF-16 surrogates; nesting too deep for the parser is a 400. Sampling fields are type- and range-checked here
    (SAMPLING_FIELDS), so a request cannot make the sampler fail on the engine thread.
  * Client disconnect: a failed write, or a closed socket seen while waiting (polled every POLL_S = 0.05 s), sets
    the job's cancel flag; the engine stops at the next pass boundary. The row in the log says client_gone.
    A peer that half-closes (shutdown(SHUT_WR) after its request) is treated as gone: on TCP a close() and a
    half-close both arrive as a FIN, and they only differ once a write draws an RST, so the server cannot tell
    them apart while it has nothing to write. nginx treats a client EOF the same way. HTTP clients do not
    half-close in practice; cancelling abandoned requests early is worth more.
  * Anthropic streams: message_start is always the first event. If a ping is due before the job reached the engine
    (queued), message_start goes out with provisional usage (input_tokens = the prompt, cache_read 0); the
    message_delta at the end carries the real input_tokens / cache_read_input_tokens, which the SDK applies.
  * Not supported (400): n > 1, logprobs, response_format json_object / json_schema, echo, best_of, suffix, images.
  * SSE responses close the connection at the end (Connection: close); JSON responses keep it alive.
  * A client that resets its connection (a keep-alive socket closed while the handler waits for its next request, a write
    to a peer that left) is not an error: the HTTP server drops it quietly (_HTTPServer.handle_error); any other exception
    in a handler still prints its traceback.
  * Process name: start() names the process `caminus` (prctl PR_SET_NAME on the main thread; /proc/<pid>/comm),
    so llm-top finds it for per-card VRAM.
  * Metrics never keep request text: /slots has no prompt or generated text, the loop detector keeps line hashes,
    harnesses are a fixed set of User-Agent classes.
"""
import json
import math
import os
import queue
import select
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import deltas as D
from . import metrics as MX
from .agentmon import LoopDetector, harness_of, tool_call_report
from .normalize import RequestError, anthropic_to_template, openai_chat_to_template
from .sampler import SamplingParams
from .stats import ENGINE, STATUS_VERSION, EngineStats, build_info
from .template import MODEL_PROFILES, PromptBuilder
from .toolparse import OutputParser, ToolSet
from .flight import new_request_id
from .worker import BundleJob, EngineWorker, Job, QueueFull

THINK_END = '</think>'
TOOL_OPEN = '<tool_call>'
MAX_BODY = 64 << 20
POLL_S = 0.05
PR_SET_NAME = 15
GEN_ROUTES = ('chat', 'completions', 'messages')


def set_process_name(name='caminus'):
    """the name /proc/<pid>/comm shows (llm-top finds the server by it). prctl(PR_SET_NAME) names the calling thread,
    so it names the process when called on the main thread (the launcher does so before any thread starts); from
    another thread /proc/self/comm (the thread group leader's) is written instead. Returns what /proc/self/comm
    reads back, or None where neither works."""
    b = name.encode()[:15]
    try:
        if threading.current_thread() is threading.main_thread():
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(PR_SET_NAME, ctypes.c_char_p(b), 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), 'prctl(PR_SET_NAME) failed')
        else:
            with open('/proc/self/comm', 'wb') as f:
                f.write(b)
        with open('/proc/self/comm', 'rb') as f:
            return f.read().strip().decode()
    except (OSError, AttributeError):
        return None


class HttpError(Exception):
    def __init__(self, status, message, etype='invalid_request_error', code=None):
        super().__init__(message)
        self.status, self.message, self.etype, self.code = status, message, etype, code


class ClientGone(Exception):
    pass


# ------------------------------------------------------------------------------------------------ error shapes
_ANTHROPIC_TYPES = {400: 'invalid_request_error', 404: 'not_found_error', 405: 'invalid_request_error',
                    411: 'invalid_request_error', 413: 'request_too_large', 500: 'api_error', 503: 'overloaded_error',
                    529: 'overloaded_error'}


def _err_text(ev):
    """the message of a worker ('error', kind, msg) event"""
    return f"{'engine' if ev[1] == 'engine' else 'internal'} error: {ev[2]}"


def openai_error(status, message, etype=None, code=None):
    if etype is None:
        etype = {404: 'not_found_error', 500: 'server_error', 503: 'server_error'}.get(status, 'invalid_request_error')
    return {'error': {'message': message, 'type': etype, 'param': None, 'code': code}}


def anthropic_error(status, message, etype=None):
    return {'type': 'error', 'error': {'type': etype or _ANTHROPIC_TYPES.get(status, 'api_error'), 'message': message}}


# ------------------------------------------------------------------------------------------------ request fields
def _reject_constant(name):
    raise ValueError(f'{name} is not valid JSON')


def _finite_float(text):
    v = float(text)
    if not math.isfinite(v):
        raise ValueError(f'number {text[:40]} is out of range')
    return v


def _num(req, key, lo=None, hi=None, integer=False):
    v = req.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not math.isfinite(v)) \
            or (integer and isinstance(v, float) and not v.is_integer()):
        raise HttpError(400, f"'{key}' must be {'an integer' if integer else 'a finite number'}")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise HttpError(400, f"'{key}' is out of range")
    return int(v) if integer else float(v)


# sampling fields (request name -> integer?, lo, hi): checked before SamplingParams.with_request. The bounds keep the
# float32 chain finite (a positive logit / repeat_penalty, logit / temperature, count * penalty).
SAMPLING_FIELDS = {
    'temperature': (False, 0.0, 100.0),
    'top_p': (False, 0.0, 1.0),
    'min_p': (False, 0.0, 1.0),
    'top_k': (True, 0, 1 << 31),
    'repeat_penalty': (False, 0.01, 100.0),
    'frequency_penalty': (False, -100.0, 100.0),
    'presence_penalty': (False, -100.0, 100.0),
    'repeat_last_n': (True, -1, 1 << 31),
    'penalty_last_n': (True, -1, 1 << 31),
    'min_keep': (True, 0, 1 << 31),
    'seed': (True, -(1 << 63), (1 << 64) - 1),
}
MAX_TOKENS_HI = 1 << 62     # larger max_tokens are clamped to the context room anyway; beyond this they are absurd
MIN_TEMP = 1e-4             # 0 < temperature < MIN_TEMP is greedy (its limit); smaller ones overflow float32 logits


def _bool(req, key, default):
    v = req.get(key)
    if v is None:
        return default
    if not isinstance(v, bool):
        raise HttpError(400, f"'{key}' must be a boolean")
    return v


def _stops(v, key):
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list) or not all(isinstance(s, str) for s in v):
        raise HttpError(400, f"'{key}' must be a string or a list of strings")
    if len(v) > 16:
        raise HttpError(400, f"'{key}': at most 16 stop sequences")
    return [s for s in v if s]


@dataclass
class GenSpec:
    api: str                    # 'openai' | 'anthropic' | 'completions'
    model: str
    stream: bool
    prompt_ids: list
    params: SamplingParams
    max_new: int
    stops: list = field(default_factory=list)
    force: tuple = None         # (trigger id, forced ids)
    prefix_text: str = ''       # forced text that ends the prompt (thinking off)
    tools: list = None
    thinking: bool = False
    parse_tools: bool = True
    parallel: bool = True
    include_usage: bool = False
    profile: str = ''
    n_tools: int = 0
    n_predict: int = -1         # the client's requested max tokens (-1: none), never clamped (/slots params)
    marks: dict = None          # the prompt cache's message boundaries (sessions with wants_marks)
    harness: str = ''           # the User-Agent's class (agentmon.harness_of; the flight recorder's line)


def _withhold_unclosed(msg, finish):
    """a generation cut from outside (finish 'length' or 'stop') leaves a call that its tags did not close unfinished:
    it may lack parameters the model was still going to write. Such calls are marked cut, so they are never reported
    as calls (deltas.py and CallGate leave cut calls out). Call it after parser.finish(), before its events are
    filtered."""
    if finish in ('length', 'stop'):
        for c in msg.tool_calls:
            if not c.complete:
                c.cut = True


class CallGate:
    """holds a tool call's events until the call closes; a cut call is dropped (see the module docstring)"""

    def __init__(self, msg, incremental=False):
        self.msg = msg
        self.incremental = incremental
        self.held = []

    def filter(self, events):
        if self.incremental:
            return events
        out = []
        for ev in events:
            k = ev[0]
            if k in ('tool_start', 'tool_args'):
                self.held.append(ev)
            elif k == 'tool_end':
                if not self.msg.tool_calls[ev[1]].cut:
                    for h in self.held:         # the held argument pieces go out as one
                        if h[0] == 'tool_args' and out and out[-1][0] == 'tool_args' and out[-1][1] == h[1]:
                            out[-1] = ('tool_args', h[1], out[-1][2] + h[2])
                        else:
                            out.append(h)
                    out.append(ev)
                self.held = []
            else:
                out.append(ev)
        return out


class RequestLog:
    def __init__(self, log_dir):
        self.dir = log_dir
        self.lock = threading.Lock()
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    @property
    def path(self):
        return os.path.join(self.dir, 'k3-server-' + time.strftime('%Y%m%d') + '.jsonl') if self.dir else None

    def write(self, row):
        if not self.dir:
            return
        line = json.dumps(row, ensure_ascii=False) + '\n'
        with self.lock:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(line)


# ------------------------------------------------------------------------------------------------ the server
class _HTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that does not print a traceback for a client that went away (socketserver's handle_error prints
    every handler exception: a keep-alive connection the client resets, e.g. at a client's or this server's shutdown, showed
    up as a ConnectionResetError traceback in the log)"""
    QUIET = (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], self.QUIET):
            return
        super().handle_error(request, client_address)


class K3Server:
    """session: an engine.EngineSession; tokenizer: tokenizer.QwenTokenizer. The HTTP server is created by start().

    Metrics: stats (stats.EngineStats; default: the session's, else a new one, given to the session), physics
    (stats.PhysicsConfig for a new one), hw (hw.HwReader or None: no hardware readings; hw_interval > 0 runs its
    sampler thread from start()), proc_name (the process name start() sets; None: unchanged), loading (True: every
    route answers 503 'Loading model', as llama-server does until its model is loaded; set_loading() changes it),
    model_alias (/props' model_alias: the default model id)."""

    def __init__(self, session, tokenizer, host='127.0.0.1', port=8092, log_dir=None, ping_interval=10.0,
                 max_queue=16, incremental_tool_args=False, drafter=None, sampling=None, model_path=None,
                 builder=None, stats=None, physics=None, hw=None, hw_interval=1.0, proc_name='caminus',
                 loading=False, model_alias='k3-coder', flight=None):
        self.sess = session
        self.flight = flight        # serve.flight.FlightRecorder or None (the per-request JSONL line, replay bundles)
        self.tok = tokenizer
        self.stats = stats or session.stats or EngineStats(physics=physics)
        session.stats = self.stats
        self.hw = hw
        self.hw_interval = hw_interval
        if hw is not None:
            hw.phase = lambda: self.stats.phase
            if getattr(hw, 'on_host', None) is None:
                hw.on_host = self.stats.host_sample        # KFD evictions / page migrations, from the sampler thread
        self.clock_check = None     # hw.clock_check()'s result at startup (caminus serve), reported in /caminus/status
        self.proc_name = proc_name
        self.loading = bool(loading)
        self.model_alias = model_alias
        self.build_info = build_info()
        self.spec_types = getattr(drafter, 'spec_type', 'draft-dflash') if drafter is not None else 'none'
        self.builder = builder or PromptBuilder.from_tokenizer(tokenizer)
        self.host, self.port = host, port
        self.ping_interval = ping_interval
        self.incremental_tool_args = incremental_tool_args
        self.drafter = drafter
        self.model_path = model_path
        self.defaults = sampling or SamplingParams.from_gguf_meta(getattr(tokenizer.vocab, 'meta', {}) or {})
        self.think_end_id = tokenizer.special.get(THINK_END)
        self.tool_open_id = tokenizer.special.get(TOOL_OPEN)
        self.worker = EngineWorker(session, tokenizer, max_queue=max_queue, tool_open_id=self.tool_open_id,
                                   stats=self.stats, think_end_id=self.think_end_id, flight=flight)
        self.worker.model_path = model_path
        self.log = RequestLog(log_dir)
        self.counts = {}
        self.lock = threading.Lock()
        self.httpd = None
        self.t_start = time.time()

    @property
    def max_ctx(self):
        return self.sess.max_ctx

    # ------------------------------------------------------------------------------------------- lifecycle
    def start(self, background=True):
        srv = self
        if self.proc_name:
            set_process_name(self.proc_name)
        if self.hw is not None and self.hw_interval > 0:
            self.hw.start(self.hw_interval)

        class Handler(K3Handler):
            server_obj = srv
        self.httpd = _HTTPServer((self.host, self.port), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        if background:
            t = threading.Thread(target=self.httpd.serve_forever, kwargs={'poll_interval': 0.1}, daemon=True,
                                 name='k3-http')
            t.start()
        else:
            self.httpd.serve_forever()
        return self

    def stop(self):
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        self.worker.shutdown()
        if self.hw is not None:
            self.hw.stop()

    def drain(self, timeout=60.0):
        """a clean shutdown's first half: the HTTP server stops accepting, the running job is cancelled at its next pass
        boundary (its client gets what was generated, finish 'cancel'), queued jobs end 'cancel', and the engine thread
        exits (waited for at most timeout seconds). True when the engine thread exited: the session may then be used from
        the caller's thread (shutdown_park)"""
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
        return self.worker.stop(timeout)

    def set_loading(self, loading):
        self.loading = bool(loading)

    def set_session(self, session, drafter=None):
        """swap in the engine session built while the server answered 503 'Loading model' (caminus serve: the HTTP surface
        is up while the engine loads); only while loading, so no job can be on the engine"""
        if not self.loading:
            raise RuntimeError('K3Server.set_session: only while loading')
        session.stats = self.stats
        self.sess = self.worker.sess = session
        self.worker.idle_s = session.idle_s
        self.drafter = drafter
        self.spec_types = getattr(drafter, 'spec_type', 'draft-dflash') if drafter is not None else 'none'

    def count(self, route, status):
        with self.lock:
            k = (route, status)
            self.counts[k] = self.counts.get(k, 0) + 1

    # ------------------------------------------------------------------------------------------- request -> spec
    def _sampling(self, req):
        for key, (integer, lo, hi) in SAMPLING_FIELDS.items():
            _num(req, key, lo, hi, integer)
        try:
            p = self.defaults.with_request(req)
        except (TypeError, ValueError, OverflowError):
            raise HttpError(400, 'invalid sampling parameter') from None
        if not (0.0 < p.top_p <= 1.0):
            raise HttpError(400, "'top_p' is out of range")
        if 0.0 < p.temperature < MIN_TEMP:
            p = replace(p, temperature=0.0)
        return p.resolve()

    def _max_new(self, n_prompt, max_tokens):
        room = self.max_ctx - n_prompt
        if room < 1:
            self.stats.refusal('context')           # a host refusal (refusals_total{kind="context"}), before the engine
            raise HttpError(400, f'prompt is too long: {n_prompt} tokens > {self.max_ctx - 1} maximum',
                            code='context_length_exceeded')
        return room if max_tokens is None else min(max_tokens, room)

    def _force(self, name, thinking):
        text = TOOL_OPEN + '\n' + (f'<function={name}>\n' if name else '')
        if thinking:
            return (self.think_end_id, self.tok.encode('\n\n' + text)), ''
        return None, text

    def _build(self, norm, model):
        try:
            return self.builder.build(norm['messages'], norm['tools'], model=model, client_kwargs=norm['client_kwargs'],
                                      marks=bool(self.sess.wants_marks))
        except (ValueError, TypeError) as e:          # TemplateRequestError is a ValueError
            raise HttpError(400, f'cannot render the prompt: {e}') from None
        except Exception as e:                       # noqa: BLE001 - other jinja errors are request errors too
            raise HttpError(400, f'cannot render the prompt: {type(e).__name__}: {e}') from None

    def chat_spec(self, req, api):
        if not isinstance(req, dict):
            raise HttpError(400, 'body must be a JSON object')
        model = req.get('model') if isinstance(req.get('model'), str) else 'k3-coder'
        stream = _bool(req, 'stream', False)
        try:
            norm = openai_chat_to_template(req) if api == 'openai' else anthropic_to_template(req)
        except RequestError as e:
            raise HttpError(400, str(e)) from None
        tools = norm['tools']
        names = [t['function']['name'] for t in tools or ()]
        parse_tools, force_name, required, parallel = True, None, False, True
        if api == 'openai':
            for k in ('n', 'best_of'):
                if req.get(k) not in (None, 1):
                    raise HttpError(400, f"'{k}' other than 1 is not supported")
            if req.get('logprobs') or req.get('top_logprobs'):
                raise HttpError(400, 'logprobs are not supported')
            rf = req.get('response_format')
            if rf is not None:
                if not isinstance(rf, dict) or rf.get('type') not in ('text',):
                    raise HttpError(400, 'response_format json_object / json_schema is not supported (no grammar '
                                         'sampler in v1)')
            mt = _num(req, 'max_completion_tokens', 1, MAX_TOKENS_HI, integer=True)
            if mt is None:
                mt = _num(req, 'max_tokens', 1, MAX_TOKENS_HI, integer=True)
            stops = _stops(req.get('stop'), 'stop')
            parallel = _bool(req, 'parallel_tool_calls', True)
            tc = req.get('tool_choice')
            if tc is None or tc == 'auto':
                pass
            elif tc == 'none':
                parse_tools = False
            elif tc == 'required':
                required = True
            elif isinstance(tc, dict) and tc.get('type') == 'function' and isinstance(tc.get('function'), dict):
                force_name = tc['function'].get('name')
                if not isinstance(force_name, str) or not force_name:
                    raise HttpError(400, 'tool_choice.function.name must be a non-empty string')
            else:
                raise HttpError(400, "tool_choice must be 'none', 'auto', 'required' or a named function")
            so = req.get('stream_options')
            include_usage = isinstance(so, dict) and bool(so.get('include_usage'))
        else:
            mt = _num(req, 'max_tokens', 1, MAX_TOKENS_HI, integer=True)
            if mt is None:
                raise HttpError(400, "'max_tokens' is required")
            stops = _stops(req.get('stop_sequences'), 'stop_sequences')
            tc = req.get('tool_choice')
            if tc is not None:
                if not isinstance(tc, dict) or tc.get('type') not in ('auto', 'any', 'tool', 'none'):
                    raise HttpError(400, 'tool_choice.type must be auto, any, tool or none')
                if tc['type'] == 'none':
                    parse_tools = False
                elif tc['type'] == 'any':
                    required = True
                elif tc['type'] == 'tool':
                    force_name = tc.get('name')
                    if not isinstance(force_name, str) or not force_name:
                        raise HttpError(400, "tool_choice of type 'tool' needs a non-empty string 'name'")
                parallel = not _bool(tc, 'disable_parallel_tool_use', False)
            include_usage = True
        if (required or force_name is not None) and not tools:
            raise HttpError(400, 'tool_choice requires tools')
        if force_name is not None and force_name not in names:
            raise HttpError(400, f'tool_choice names an unknown tool {force_name!r}')
        params = self._sampling(req)
        prompt = self._build(norm, model)
        thinking = bool(prompt.kwargs.get('enable_thinking', True))
        force, prefix_text = None, ''
        if required or force_name is not None:
            force, prefix_text = self._force(force_name, thinking)
        ids = list(prompt.ids)
        if prefix_text:
            ids += self.tok.encode(prefix_text)
        return GenSpec(api=api, model=model, stream=stream, prompt_ids=ids, params=params,
                       max_new=self._max_new(len(ids), mt), stops=stops, force=force, prefix_text=prefix_text,
                       tools=tools, thinking=thinking, parse_tools=parse_tools, parallel=parallel,
                       include_usage=include_usage, profile=prompt.profile, n_tools=len(names),
                       n_predict=-1 if mt is None else mt, marks=prompt.marks)

    def completion_spec(self, req):
        if not isinstance(req, dict):
            raise HttpError(400, 'body must be a JSON object')
        for k in ('echo', 'logprobs', 'suffix'):
            if req.get(k):
                raise HttpError(400, f"'{k}' is not supported")
        if req.get('n') not in (None, 1) or req.get('best_of') not in (None, 1):
            raise HttpError(400, "'n' / 'best_of' other than 1 are not supported")
        p = req.get('prompt')
        if isinstance(p, list) and len(p) == 1 and isinstance(p[0], (str, list)):
            p = p[0]
        bos = self.tok.bos_id                       # (None: the vocabulary has no BOS token, nothing is prepended)
        head = [] if bos is None else [bos]
        if isinstance(p, str):
            if not p:
                raise HttpError(400, "'prompt' is empty")
            ids = head + self.tok.encode(p)
        elif isinstance(p, list) and p and all(isinstance(t, int) and not isinstance(t, bool) for t in p):
            if not all(0 <= t < self.tok.n_vocab for t in p):
                raise HttpError(400, 'token id out of range')
            ids = list(p) if bos is None or p[0] == bos else [bos] + list(p)
        else:
            raise HttpError(400, "'prompt' must be a string or a list of token ids (one prompt)")
        mt = _num(req, 'max_tokens', 1, MAX_TOKENS_HI, integer=True)
        model = req.get('model') if isinstance(req.get('model'), str) else 'k3-coder'
        so = req.get('stream_options')
        return GenSpec(api='completions', model=model, stream=_bool(req, 'stream', False), prompt_ids=ids,
                       params=self._sampling(req), max_new=self._max_new(len(ids), mt if mt is not None else 16),
                       stops=_stops(req.get('stop'), 'stop'),
                       include_usage=isinstance(so, dict) and bool(so.get('include_usage')),
                       n_predict=mt if mt is not None else 16)

    def count_tokens(self, req):
        if not isinstance(req, dict):
            raise HttpError(400, 'body must be a JSON object')
        req = dict(req)
        req.setdefault('max_tokens', 1)
        try:
            norm = anthropic_to_template(req)
        except RequestError as e:
            raise HttpError(400, str(e)) from None
        model = req.get('model') if isinstance(req.get('model'), str) else 'k3-coder'
        return {'input_tokens': len(self._build(norm, model).ids)}

    def submit(self, spec):
        job = Job(spec.prompt_ids, spec.params, spec.max_new, spec.stops, spec.force, self.drafter, tag=spec.api,
                  n_predict=spec.n_predict, thinking=spec.thinking, marks=spec.marks,
                  slot_params=MX.task_params(spec.params, spec.n_predict, spec.stream, self.spec_types),
                  request_id=new_request_id(), tools=spec.tools, parse_tools=spec.parse_tools, parallel=spec.parallel,
                  prefix_text=spec.prefix_text, profile=spec.profile, harness=spec.harness)
        try:
            return self.worker.submit(job)
        except QueueFull as e:
            raise HttpError(503, f'overloaded: {e}', etype='overloaded_error') from None

    # ------------------------------------------------------------------------------------------- reporting
    def timings(self, job, t_recv):
        r = job.result
        pre_n = r.prompt_n - r.cached_n
        dec_n = len(r.gen_ids)
        t_end = r.prefill_end_t + r.decode_ms / 1e3
        t = {'cache_n': r.cached_n, 'prompt_n': pre_n, 'prompt_ms': round(r.prefill_ms, 3),
             'prompt_per_token_ms': round(r.prefill_ms / pre_n, 4) if pre_n else 0.0,
             'prompt_per_second': round(pre_n / (r.prefill_ms / 1e3), 2) if pre_n and r.prefill_ms else 0.0,
             'predicted_n': dec_n, 'predicted_ms': round(r.decode_ms, 3),
             'predicted_per_token_ms': round(r.decode_ms / dec_n, 4) if dec_n else 0.0,
             'predicted_per_second': round(dec_n / (r.decode_ms / 1e3), 2) if dec_n and r.decode_ms else 0.0,
             'decode_tps': round((dec_n - 1) / (t_end - r.first_token_t), 2)
             if dec_n > 1 and r.first_token_t and t_end > r.first_token_t else None,
             'queue_ms': round((job.t_start - job.t_submit) * 1e3, 3) if job.t_start else None,
             'ttft_ms': round((r.first_token_t - t_recv) * 1e3, 3) if r.first_token_t else None,
             'passes': r.passes, 'prefill_passes': r.prefill_passes, 'drafted': r.drafted, 'accepted': r.accepted,
             'restore': r.restore, 'seed': job.params.seed}
        return t

    def log_row(self, route, spec, job, t_recv, status, error=None, client_gone=False, finish=None):
        row = {'ts': round(time.time(), 3), 'route': route, 'status': status,
               'model': spec.model if spec else None, 'profile': spec.profile if spec else None,
               'stream': spec.stream if spec else None, 'tools': spec.n_tools if spec else None,
               'client_gone': client_gone, 'error': error,
               'total_ms': round((time.perf_counter() - t_recv) * 1e3, 3)}
        if job is not None and job.result is not None:
            r = job.result
            row.update(self.timings(job, t_recv))
            row.update({'prompt_tokens': r.prompt_n, 'cached_tokens': r.cached_n, 'completion_tokens': len(r.gen_ids),
                        'finish': finish or r.finish, 'forced': r.forced_n, 'stop_in_draft': r.stop_in_draft})
        elif job is not None:
            row['queue_ms'] = round(((job.t_start or time.perf_counter()) - job.t_submit) * 1e3, 3)
        self.log.write(row)
        self.count(route, status)
        return row

    def flight_row(self, row, spec, job, harness=None, loop=None, tools=None):
        """the flight recorder's line of a request (M7 ops; the plan's fields; no request text): the access-log row plus the
        request id, harness class, model profile, sampling parameters with the resolved seed, the cache source and restore
        ms, the TTFT split, decode ms per token, DFlash acceptance, bails / recoveries, the stop reason, tool-call parse
        failures and the loop-detector flag"""
        out = dict(row)
        out['request_id'] = getattr(job, 'request_id', None)
        out['harness'] = harness if harness is not None else (spec.harness if spec else None)
        if spec is not None:
            p = spec.params
            out['sampling'] = {k: getattr(p, k) for k in ('temperature', 'top_k', 'top_p', 'min_p', 'presence_penalty',
                                                            'frequency_penalty', 'repeat_penalty', 'penalty_last_n', 'seed')
                               if hasattr(p, k)}
            out['max_new'] = spec.max_new
        r = getattr(job, 'result', None)
        if r is not None:
            pf = getattr(r, 'prefill', None) or {}
            out['cache'] = dict(source=pf.get('source'), restore=r.restore, cached=r.cached_n, restore_ms=pf.get('restore_ms'),
                                exact=pf.get('exact'), lcp=pf.get('lcp'), slot=pf.get('slot', 'main'))
            out['ttft_split_ms'] = dict(queue=out.get('queue_ms'), restore=pf.get('restore_ms'), batched=pf.get('batched_ms'),
                                        handoff=pf.get('handoff_ms'), passes=pf.get('passes_ms'), total=out.get('ttft_ms'))
            n = len(r.gen_ids)
            out['decode_ms_per_token'] = round(r.decode_ms / n, 4) if n else None
            out['dflash'] = dict(drafted=r.drafted, accepted=r.accepted, acceptance=round(r.accepted / r.drafted, 4) if r.drafted else None,
                                 passes=r.passes)
            out['engine'] = getattr(r, 'engine_counts', None)
        out['tool_calls'] = None if tools is None else dict(ok=tools[0], failures=tools[1], lenient=tools[2])
        out['loop'] = None if loop is None else dict(flagged=bool(loop[0]), repeated_lines=int(loop[1]))
        rec = self.worker.recent.get(out['request_id']) if out['request_id'] else None
        out['verdict'] = rec.get('verdict') if rec else None
        out['bundles'] = [b[2] and os.path.basename(b[2]) for b in self.worker.bundles if b[0] == out['request_id']] or None
        return out

    def flight_api(self, what, req):
        """the flight recorder's routes: GET /caminus/flight (the directory's entries), POST /caminus/flight/bundle
        {"request_id": ID} (default: the most recent request; saved on the engine thread in FIFO order with the requests),
        POST /caminus/flight/purge {"older_than_days": D} (default: everything). -> (status, body)"""
        fl = self.flight
        if fl is None:
            return 404, {'error': {'message': 'the flight recorder is off (caminus serve --flight-dir)', 'type': 'not_found_error', 'code': 404}}
        if what == 'flight_list':
            ents = fl.entries()
            return 200, {'root': fl.root, 'bytes': sum(e[2] for e in ents), 'cap_bytes': fl.cap,
                         'entries': [{'kind': k, 'name': os.path.basename(p), 'bytes': nb, 'mtime': round(mt, 3)} for k, p, nb, mt in ents]}
        if what == 'flight_purge':
            od = req.get('older_than_days')
            if od is not None and (isinstance(od, bool) or not isinstance(od, (int, float)) or od < 0):
                return 400, {'error': {'message': "'older_than_days' must be a number >= 0", 'type': 'invalid_request_error', 'code': 400}}
            return 200, {'removed': fl.purge(all=od is None, older_than_days=od)}
        rid = req.get('request_id')
        if rid is not None and not isinstance(rid, str):
            return 400, {'error': {'message': "'request_id' must be a string", 'type': 'invalid_request_error', 'code': 400}}
        bj = BundleJob(rid, 'api')
        self.worker.q.put(bj)
        try:
            ev = bj.events.get(timeout=900)
        except queue.Empty:
            return 504, {'error': {'message': 'the bundle did not finish in 900 s', 'type': 'api_error', 'code': 504}}
        if ev[0] == 'done':
            return 200, ev[1]
        code = 404 if ev[1] == 'not_found' else 500
        return code, {'error': {'message': ev[2], 'type': 'not_found_error' if code == 404 else 'api_error', 'code': code}}

    def server_view(self):
        """the server-side values the metrics renderers need (metrics.prometheus / status)"""
        w = self.worker
        with self.lock:
            http = dict(self.counts)
        return {'requests_processing': 1 if w.busy else 0, 'requests_deferred': w.deferred, 'queue_depth': w.depth,
                'build_info': self.build_info, 'model_alias': self.model_alias, 'model_path': self.model_path or '',
                'n_ctx': self.max_ctx, 'loading': self.loading, 'spec_types': self.spec_types, 'http': http,
                'worker': dict(w.stats), 'clock_check': self.clock_check}

    def hw_view(self):
        if self.hw is None:
            return [], []
        try:
            return self.hw.read(), self.hw.energy()
        except Exception:       # noqa: BLE001 - a hardware reading never fails a scrape
            return [], []

    # the scrapes: one snapshot under the stats lock (a copy, ~0.1 ms), then quantiles and text on this HTTP thread with the
    # lock released (M7 ops: keep the ~1.5 ms of formatting off the engine thread)
    def metrics(self):
        snap = self.stats.snapshot(take_bucket=True)
        cards, energy = self.hw_view()
        return MX.prometheus(snap, self.server_view(), cards, energy)

    def status(self):
        snap = self.stats.snapshot()
        cards, energy = self.hw_view()
        return MX.status(snap, self.server_view(), cards, energy)

    def slots(self):
        return [MX.slot_json(self.stats.snapshot(), self.max_ctx, self.drafter is not None)]

    def props(self):
        """llama-server's get_res_props shape (llm-top reads default_generation_settings.n_ctx / .params.n_predict,
        total_slots, model_alias, model_path, build_info, is_sleeping), plus the engine marker and Caminus' own keys
        (n_ctx, model, models)"""
        p = self.defaults
        tok = self.tok
        eos = getattr(tok, 'eos_id', None)
        return {
            'default_generation_settings': {'params': MX.task_params(p, -1, False, self.spec_types),
                                            'n_ctx': self.max_ctx},
            'total_slots': 1, 'model_alias': self.model_alias, 'model_ftype': '',
            'model_path': self.model_path or '',
            'modalities': {'vision': False, 'video': False, 'audio': False}, 'media_marker': '<__media__>',
            'endpoint_slots': True, 'endpoint_props': False, 'endpoint_metrics': True, 'ui': False, 'ui_settings': {},
            'chat_template': tok.vocab.chat_template,
            'chat_template_caps': {'supports_object_arguments': True, 'supports_parallel_tool_calls': True,
                                   'supports_preserve_reasoning': True, 'supports_reasoning_effort': True,
                                   'supports_string_content': True, 'supports_system_role': True,
                                   'supports_tool_calls': True, 'supports_tools': True,
                                   'supports_typed_content': True},
            'bos_token': tok.vocab.tokens[tok.bos_id] if tok.bos_id is not None else '',
            'eos_token': tok.vocab.tokens[eos] if eos is not None else '',
            'build_info': self.build_info, 'is_sleeping': False, 'cors_proxy_enabled': False,
            'engine': ENGINE,
            'caminus': {'build_info': self.build_info, 'status': '/caminus/status', 'status_version': STATUS_VERSION,
                        'metrics_prefix': 'caminus:', 'speculative_types': self.spec_types,
                        'stats_enabled': self.stats.enabled},
            'n_ctx': self.max_ctx,
            'model': os.path.basename(self.model_path) if self.model_path else 'mock',
            'models': list(MODEL_PROFILES)}


# ------------------------------------------------------------------------------------------------ the handler
class K3Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'k3-server/0.1'
    server_obj = None               # set by K3Server.start

    def log_message(self, fmt, *args):     # quiet; the JSONL log has the requests
        pass

    # ------------------------------------------------------------------------------------------- io
    def _read_body(self):
        te = (self.headers.get('Transfer-Encoding') or '').lower()
        if 'chunked' in te:
            parts, total = [], 0
            while True:
                line = self.rfile.readline(65537)
                try:
                    n = int(line.split(b';')[0].strip(), 16)
                except ValueError:
                    raise HttpError(400, 'bad chunked encoding') from None
                if n == 0:
                    while self.rfile.readline(65537) not in (b'\r\n', b'\n', b''):
                        pass
                    break
                total += n
                if total > MAX_BODY:
                    self.close_connection = True
                    raise HttpError(413, 'request body too large')
                parts.append(self.rfile.read(n))
                self.rfile.readline(3)
            return b''.join(parts)
        cl = self.headers.get('Content-Length')
        if cl is None:
            if self.command == 'POST':
                self.close_connection = True
                raise HttpError(411, 'Content-Length required')
            return b''
        try:
            n = int(cl)
        except ValueError:
            self.close_connection = True
            raise HttpError(400, 'bad Content-Length') from None
        if n < 0 or n > MAX_BODY:
            self.close_connection = True
            raise HttpError(413, 'request body too large')
        return self.rfile.read(n)

    def _json_body(self):
        raw = self._read_body()
        try:
            obj = json.loads(raw.decode('utf-8'), parse_constant=_reject_constant, parse_float=_finite_float)
        except (UnicodeDecodeError, ValueError) as e:
            raise HttpError(400, f'invalid JSON body: {e}') from None
        except RecursionError:
            raise HttpError(400, 'invalid JSON body: nested too deeply') from None
        if b'\\ud' in raw.lower():           # a \uD800-\uDFFF escape: reject lone surrogates (not UTF-8 encodable)
            try:
                json.dumps(obj, ensure_ascii=False).encode('utf-8')
            except UnicodeEncodeError:
                raise HttpError(400, 'invalid JSON body: lone UTF-16 surrogate in a string') from None
            except RecursionError:
                raise HttpError(400, 'invalid JSON body: nested too deeply') from None
        return obj

    def _send(self, status, body, ctype='application/json', headers=None, raw_ctype=False):
        data = body if isinstance(body, bytes) else (
            json.dumps(body, ensure_ascii=False).encode('utf-8') if ctype == 'application/json' else body.encode())
        self._hdr_sent = True
        try:
            self.send_response(status)
            self.send_header('Content-Type', ctype if raw_ctype else
                             ctype + ('; charset=utf-8' if ctype.startswith('text/') else ''))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.send_header('Content-Length', str(len(data)))
            if self.close_connection:
                self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()
        except OSError:
            self.close_connection = True

    def _sse_begin(self):
        self.close_connection = True
        self._hdr_sent = True
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'close')
            self.send_header('X-Accel-Buffering', 'no')
            self.end_headers()
            self.wfile.flush()
        except OSError:
            raise ClientGone() from None

    def _sse(self, data, event=None):
        msg = (f'event: {event}\n' if event else '') + 'data: ' + (
            data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)) + '\n\n'
        try:
            self.wfile.write(msg.encode('utf-8'))
            self.wfile.flush()
        except OSError:
            raise ClientGone() from None

    def _peer_closed(self):
        try:
            r, _, _ = select.select([self.connection], [], [], 0)
            if not r:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b''
        except (OSError, ValueError):
            return True

    def _events(self, job, on_idle=None):
        """yields the job's events; raises ClientGone (after setting cancel) when the peer closes. The socket is
        checked every POLL_S whether or not events arrive (a non-streaming request writes nothing until the end)."""
        idle_since = last_check = time.perf_counter()
        while True:
            try:
                ev = job.events.get(timeout=POLL_S)
            except queue.Empty:
                ev = None
            now = time.perf_counter()
            if now - last_check >= POLL_S:
                last_check = now
                if self._peer_closed():
                    job.cancel.set()
                    raise ClientGone() from None
            if ev is None:
                if on_idle is not None and now - idle_since >= self.server_obj.ping_interval:
                    on_idle()
                    idle_since = time.perf_counter()
                continue
            idle_since = now
            if ev[0] == 'text' and self._loop is not None:
                self._loop.feed(ev[1])
            yield ev
            if ev[0] in ('done', 'error'):
                return

    def _drain(self, job, timeout=60.0):
        """after a disconnect: wait for the worker to finish the job (it stops at the next pass boundary)"""
        job.cancel.set()
        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end:
            try:
                ev = job.events.get(timeout=0.5)
            except queue.Empty:
                continue
            if ev[0] in ('done', 'error'):
                return ev
        return None

    # ------------------------------------------------------------------------------------------- dispatch
    def _api_of(self, path):
        return 'anthropic' if path.startswith('/v1/messages') else 'openai'

    def _error(self, api, e):
        if api == 'anthropic':
            status = 529 if e.status == 503 else e.status
            et = e.etype if e.etype in ('overloaded_error', 'api_error', 'not_found_error') else None
            self._send(status, anthropic_error(status, e.message, et))
        else:
            et = None if e.etype in ('invalid_request_error',) else e.etype
            if et == 'overloaded_error':
                et = 'server_error'
            self._send(e.status, openai_error(e.status, e.message, et, e.code))

    _loop = None

    def _loading(self):
        """llama-server's answer to every route while its model loads"""
        self._send(503, {'error': {'message': 'Loading model', 'type': 'unavailable_error', 'code': 503}})

    def do_GET(self):
        srv = self.server_obj
        path, _, query = self.path.partition('?')
        try:
            self._read_body()
            if srv.loading:
                self._loading()
                srv.count(path, 503)
                return
            if path in ('/health', '/v1/health'):
                self._send(200, {'status': 'ok'})
            elif path in ('/v1/models', '/models'):
                self._send(200, {'object': 'list', 'data': [
                    {'id': m, 'object': 'model', 'created': int(srv.t_start), 'owned_by': 'k3',
                     'meta': {'n_ctx': srv.max_ctx}} for m in MODEL_PROFILES]})
            elif path == '/props':
                self._send(200, srv.props())
            elif path == '/slots':
                if 'fail_on_no_slot=' in query and srv.stats.slot.is_processing:
                    self._send(503, {'error': {'code': 503, 'message': 'no slot available',
                                               'type': 'unavailable_error'}})
                    srv.count(path, 503)
                    return
                self._send(200, srv.slots())
            elif path == '/metrics':
                self._send(200, srv.metrics(), ctype='text/plain; version=0.0.4', raw_ctype=True,
                           headers={'Process-Start-Time-Unix': str(srv.stats.t_start_unix)})
            elif path == '/caminus/status':
                self._send(200, srv.status())
            elif path == '/caminus/flight':
                st, body = srv.flight_api('flight_list', {})
                self._send(st, body)
                srv.count(path, st)
                return
            elif path in ('/v1/chat/completions', '/chat/completions', '/v1/completions', '/completions',
                          '/v1/messages', '/v1/messages/count_tokens'):
                raise HttpError(405, f'{path} needs POST')
            else:
                raise HttpError(404, f'no route {path}', etype='not_found_error')
            srv.count(path, 200)
        except HttpError as e:
            self._error(self._api_of(path), e)
            srv.count(path, e.status)

    def do_POST(self):
        self._row = self._spec = self._job = self._tools = self._loopres = self._harness = None
        try:
            self._do_post()
        finally:
            srv = self.server_obj
            if srv.flight is not None and self._row is not None and self._harness is not None:
                try:            # the flight recorder's line (generation routes, errors and 499s included)
                    srv.flight.record(srv.flight_row(self._row, self._spec, self._job, self._harness, self._loopres, self._tools))
                except Exception:   # noqa: BLE001 - the recorder never fails a request
                    traceback.print_exc()

    def _tool_rep(self, msg, parser):
        rep = tool_call_report(msg, parser.notes)
        self.server_obj.stats.tool_report(*rep)
        self._tools = rep

    def _do_post(self):
        srv = self.server_obj
        path = self.path.split('?', 1)[0]
        api = self._api_of(path)
        t_recv = time.perf_counter()
        spec = job = None
        self._hdr_sent = False
        try:
            if path in ('/v1/chat/completions', '/chat/completions'):
                route = 'chat'
            elif path in ('/v1/completions', '/completions'):
                route = 'completions'
            elif path == '/v1/messages':
                route = 'messages'
            elif path == '/v1/messages/count_tokens':
                route = 'count_tokens'
            elif path in ('/caminus/flight/bundle', '/caminus/flight/purge'):
                raw = self._read_body()
                try:
                    req = json.loads(raw.decode('utf-8')) if raw.strip() else {}
                    if not isinstance(req, dict): raise ValueError('not an object')
                except (UnicodeDecodeError, ValueError) as e:
                    raise HttpError(400, f'invalid JSON body: {e}') from None
                st, body = srv.flight_api('flight_bundle' if path.endswith('bundle') else 'flight_purge', req)
                self._send(st, body)
                srv.count(path, st)
                return
            elif path in ('/v1/models', '/models', '/health', '/props', '/metrics', '/slots', '/caminus/status', '/caminus/flight'):
                self._read_body()
                raise HttpError(405, f'{path} needs GET')
            else:
                self._read_body()
                raise HttpError(404, f'no route {path}', etype='not_found_error')
            if srv.loading:
                self._read_body()
                self._loading()
                srv.count(route, 503)
                return
            if route in GEN_ROUTES:
                self._harness = harness_of(self.headers.get('User-Agent'))
                srv.stats.http_request(self._harness)
            req = self._json_body()
            if route == 'count_tokens':
                self._send(200, srv.count_tokens(req))
                srv.count(route, 200)
                return
            if route == 'completions':
                spec = srv.completion_spec(req)
            else:
                spec = srv.chat_spec(req, 'openai' if route == 'chat' else 'anthropic')
            spec.harness = self._harness or ''
            self._spec = spec
            job = srv.submit(spec)
            self._job = job
            self._loop = LoopDetector(spec.thinking, srv.stats.loop_live_set) if spec.api != 'completions' else None
        except HttpError as e:
            self._error(api, e)
            self._row = srv.log_row(path, spec, job, t_recv, e.status, error=e.message)
            return
        except Exception as e:      # noqa: BLE001
            traceback.print_exc()
            self._error(api, HttpError(500, f'internal error: {type(e).__name__}: {e}', etype='api_error'))
            self._row = srv.log_row(path, spec, job, t_recv, 500, error=str(e))
            return
        try:
            if spec.api == 'anthropic':
                status = self._anthropic(spec, job, t_recv)
            elif spec.api == 'openai':
                status = self._openai(spec, job, t_recv)
            else:
                status = self._completions(spec, job, t_recv)
            err = None
            if isinstance(status, tuple):
                status, err = status
            self._row = srv.log_row(route, spec, job, t_recv, status, error=err)
        except ClientGone:
            self.close_connection = True
            self._drain(job)
            self._row = srv.log_row(route, spec, job, t_recv, 499, client_gone=True, finish='cancel')
        except Exception as e:      # noqa: BLE001 - a bug while answering: stop the job, answer if still possible
            traceback.print_exc()
            self.close_connection = True
            self._drain(job)
            if not self._hdr_sent:
                self._error(api, HttpError(500, f'internal error: {type(e).__name__}: {e}', etype='api_error'))
            self._row = srv.log_row(route, spec, job, t_recv, 500, error=f'{type(e).__name__}: {e}')
        finally:
            if self._loop is not None:
                self._loopres = self._loop.finish()
                srv.stats.loop_report(*self._loopres)
                self._loop = None

    # ------------------------------------------------------------------------------------------- generation
    def _engine_error_status(self, api):
        return 529 if api == 'anthropic' else 503

    def _wait_done(self, job):
        """non-streaming: collect the text; returns (text, done event)"""
        parts = []
        for ev in self._events(job):
            if ev[0] == 'text':
                parts.append(ev[1])
            elif ev[0] in ('done', 'error'):
                return ''.join(parts), ev
        return ''.join(parts), None

    def _parser(self, spec):
        p = OutputParser(ToolSet(spec.tools), thinking=spec.thinking, parse_tools=spec.parse_tools,
                         parallel=spec.parallel)
        pre = p.feed(spec.prefix_text) if spec.prefix_text else []
        return p, pre

    @staticmethod
    def _stop_kind(finish):
        return {'eos': 'eos', 'stop': 'stop', 'length': 'length'}.get(finish, 'eos')

    # ---- OpenAI chat
    def _openai(self, spec, job, t_recv):
        srv = self.server_obj
        cid = D.new_id('chatcmpl-')
        created = int(time.time())
        parser, pre = self._parser(spec)
        msg = parser.result()

        def usage(r):
            return {'prompt_tokens': r.prompt_n, 'completion_tokens': len(r.gen_ids),
                    'total_tokens': r.prompt_n + len(r.gen_ids),
                    'prompt_tokens_details': {'cached_tokens': r.cached_n}}

        if not spec.stream:
            text, ev = self._wait_done(job)
            if ev[0] == 'error':
                st = 503 if ev[1] == 'engine' else 500
                self._send(st, openai_error(st, _err_text(ev)))
                return st, ev[2]
            parser.feed(text)
            parser.finish()
            r = job.result
            _withhold_unclosed(msg, r.finish)
            self._tool_rep(msg, parser)
            fin = D.openai_finish_reason(msg, self._stop_kind(r.finish))
            self._send(200, {'id': cid, 'object': 'chat.completion', 'created': created, 'model': spec.model,
                             'choices': [{'index': 0, 'message': D.openai_message(msg), 'logprobs': None,
                                          'finish_reason': fin}],
                             'usage': usage(r), 'timings': srv.timings(job, t_recv)})
            return 200

        gate = CallGate(msg, srv.incremental_tool_args)
        ostream = D.OpenAIStream()

        def chunk(delta, finish=None):
            return {'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': spec.model,
                    'choices': [{'index': 0, 'delta': delta, 'logprobs': None, 'finish_reason': finish}]}

        def send_events(evs):
            for d in ostream.deltas(gate.filter(evs)):
                self._sse(chunk(d))

        began = []

        def begin():
            if not began:
                began.append(True)
                self._sse_begin()
                self._sse(chunk({'role': 'assistant', 'content': ''}))
                send_events(pre)

        for ev in self._events(job):
            if ev[0] == 'text':
                begin()
                send_events(parser.feed(ev[1]))
            elif ev[0] == 'error':
                st = 503 if ev[1] == 'engine' else 500
                if not began:
                    self._send(st, openai_error(st, _err_text(ev)))
                else:
                    self._sse({'error': {'message': _err_text(ev), 'type': 'server_error', 'param': None,
                                         'code': None}})
                return st, ev[2]
            elif ev[0] == 'done':
                r = ev[1]
                if r.finish == 'cancel':
                    raise ClientGone()
                begin()
                evs = parser.finish()
                _withhold_unclosed(msg, r.finish)
                self._tool_rep(msg, parser)
                send_events(evs)
                fin = D.openai_finish_reason(msg, self._stop_kind(r.finish))
                last = chunk({}, fin)
                last['timings'] = srv.timings(job, t_recv)
                if not spec.include_usage:
                    last['usage'] = usage(r)
                self._sse(last)
                if spec.include_usage:
                    u = chunk({})
                    u['choices'] = []
                    u['usage'] = usage(r)
                    self._sse(u)
                self._sse('[DONE]')
        return 200

    # ---- OpenAI completions
    def _completions(self, spec, job, t_recv):
        srv = self.server_obj
        cid = D.new_id('cmpl-')
        created = int(time.time())

        def usage(r):
            return {'prompt_tokens': r.prompt_n, 'completion_tokens': len(r.gen_ids),
                    'total_tokens': r.prompt_n + len(r.gen_ids),
                    'prompt_tokens_details': {'cached_tokens': r.cached_n}}

        def body(text, finish):
            return {'id': cid, 'object': 'text_completion', 'created': created, 'model': spec.model,
                    'choices': [{'index': 0, 'text': text, 'logprobs': None, 'finish_reason': finish}]}

        def fin(r):
            return 'length' if r.finish == 'length' else 'stop'

        if not spec.stream:
            text, ev = self._wait_done(job)
            if ev[0] == 'error':
                st = 503 if ev[1] == 'engine' else 500
                self._send(st, openai_error(st, _err_text(ev)))
                return st, ev[2]
            r = job.result
            out = body(text, fin(r))
            out['usage'] = usage(r)
            out['timings'] = srv.timings(job, t_recv)
            self._send(200, out)
            return 200
        began = []
        for ev in self._events(job):
            if ev[0] == 'text':
                if not began:
                    began.append(True)
                    self._sse_begin()
                self._sse(body(ev[1], None))
            elif ev[0] == 'error':
                st = 503 if ev[1] == 'engine' else 500
                if not began:
                    self._send(st, openai_error(st, _err_text(ev)))
                else:
                    self._sse({'error': {'message': _err_text(ev), 'type': 'server_error', 'param': None,
                                         'code': None}})
                return st, ev[2]
            elif ev[0] == 'done':
                r = ev[1]
                if r.finish == 'cancel':
                    raise ClientGone()
                if not began:
                    began.append(True)
                    self._sse_begin()
                last = body('', fin(r))
                last['timings'] = srv.timings(job, t_recv)
                if not spec.include_usage:
                    last['usage'] = usage(r)
                self._sse(last)
                if spec.include_usage:
                    u = body('', None)
                    u['choices'] = []
                    u['usage'] = usage(r)
                    self._sse(u)
                self._sse('[DONE]')
        return 200

    # ---- Anthropic messages
    def _anthropic(self, spec, job, t_recv):
        srv = self.server_obj
        mid = D.new_id('msg_')
        parser, pre = self._parser(spec)
        msg = parser.result()

        def usage(r, start=False):
            u = {'input_tokens': r['prompt_n'] - r['cached_n'], 'cache_creation_input_tokens': 0,
                 'cache_read_input_tokens': r['cached_n'], 'output_tokens': 0 if start else r['out']}
            return u

        def ustats(r):
            return {'prompt_n': r.prompt_n, 'cached_n': r.cached_n, 'out': len(r.gen_ids)}

        def stop_seq(r):
            return job.stop_matched if r.finish == 'stop' else None

        if not spec.stream:
            text, ev = self._wait_done(job)
            if ev[0] == 'error':
                st = 529 if ev[1] == 'engine' else 500
                self._send(st, anthropic_error(st, _err_text(ev)))
                return st, ev[2]
            parser.feed(text)
            parser.finish()
            r = job.result
            _withhold_unclosed(msg, r.finish)
            self._tool_rep(msg, parser)
            self._send(200, {'id': mid, 'type': 'message', 'role': 'assistant', 'model': spec.model,
                             'content': D.anthropic_content(msg),
                             'stop_reason': D.anthropic_stop_reason(msg, self._stop_kind(r.finish)),
                             'stop_sequence': stop_seq(r), 'usage': usage(ustats(r)),
                             'timings': srv.timings(job, t_recv)})
            return 200

        gate = CallGate(msg, srv.incremental_tool_args)
        astream = D.AnthropicStream()

        def send(evs):
            for e in astream.events(gate.filter(evs)):
                self._sse(e, e['type'])

        state = {'began': False, 'started': False, 'info': None}

        def begin(provisional=False):
            """headers on the first output or ping; message_start once the job reached the engine, or at a ping
            while it is still queued (provisional usage: the whole prompt as input, nothing cached yet; the final
            message_delta carries the real numbers)"""
            if not state['began']:
                state['began'] = True
                self._sse_begin()
            if not state['started'] and (state['info'] is not None or provisional):
                state['started'] = True
                info = state['info'] or {'prompt_n': len(spec.prompt_ids), 'cached_n': 0}
                self._sse({'type': 'message_start', 'message': {
                    'id': mid, 'type': 'message', 'role': 'assistant', 'model': spec.model, 'content': [],
                    'stop_reason': None, 'stop_sequence': None,
                    'usage': usage({'prompt_n': info['prompt_n'], 'cached_n': info['cached_n'], 'out': 0}, True)}},
                    'message_start')
                send(pre)

        def ping():
            begin(provisional=True)
            self._sse({'type': 'ping'}, 'ping')

        for ev in self._events(job, on_idle=ping):
            if ev[0] == 'start':
                state['info'] = ev[1]
                if state['began']:
                    begin()
            elif ev[0] == 'text':
                begin()
                send(parser.feed(ev[1]))
            elif ev[0] == 'error':
                st, et = (529, 'overloaded_error') if ev[1] == 'engine' else (500, 'api_error')
                if not state['began']:
                    self._send(st, anthropic_error(st, _err_text(ev), et))
                else:
                    self._sse({'type': 'error', 'error': {'type': et, 'message': _err_text(ev)}}, 'error')
                return st, ev[2]
            elif ev[0] == 'done':
                r = ev[1]
                if r.finish == 'cancel':
                    raise ClientGone()
                begin()
                evs = parser.finish()
                _withhold_unclosed(msg, r.finish)
                self._tool_rep(msg, parser)
                send(evs)
                for e in astream.close():
                    self._sse(e, e['type'])
                self._sse({'type': 'message_delta',
                           'delta': {'stop_reason': D.anthropic_stop_reason(msg, self._stop_kind(r.finish)),
                                     'stop_sequence': stop_seq(r)},
                           'usage': usage(ustats(r)), 'timings': srv.timings(job, t_recv)}, 'message_delta')
                self._sse({'type': 'message_stop'}, 'message_stop')
        return 200
