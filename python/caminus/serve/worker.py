"""The engine worker: one thread owns the EngineSession and runs one request at a time from a FIFO queue
(server spec section 2: one live sequence, batch 1; Concurrency, Cancellation).

A Job carries the prompt ids and the generation settings in, and an event queue out. Events, in order:
    ('start', info)        the job reached the engine: info = {'queue_ms', 'prompt_n', 'cached_n'}
    ('text', str)          released text (EOS skipped, UTF-8 complete, stop strings held back / cut)
    ('done', GenResult)    finished; GenResult.finish is 'eos' | 'length' | 'stop' | 'cancel'
    ('error', kind, msg)   kind 'engine' (EngineError: the session was reset; answer 503 / 529) or 'internal'
                           (any other exception, a bug: the session is aborted, not reset, so the committed state
                           and the checkpoints survive; reset only if abort() fails; answer 500)
A job cancelled before it starts gets ('done', result with finish 'cancel') without touching the engine.
The HTTP thread sets job.cancel on a client disconnect; the worker sees it at the next pass boundary.

Metrics (docs/specs/spec-metrics.md): with `stats` (stats.EngineStats) the worker reports each job's begin / end
(or error), counts the generated ids as reasoning (up to and including </think> when the request thinks) or answer,
and keeps the content class of the text being generated (agentmon.ContentClass) for DFlash's per-class acceptance.
Job carries what /slots shows: id_task (a process-wide counter), n_predict (the client's requested max tokens, -1
when none, never the context clamp) and slot_params (llama-server's per-request params)."""
import collections
import dataclasses
import itertools
import queue
import threading
import time
import traceback

from .agentmon import ContentClass, LoopDetector, tool_call_report
from .engine import EngineError
from .generate import GenResult, TokenText, run_generation
from .sampler import Sampler

_TASK_IDS = itertools.count()


class QueueFull(RuntimeError):
    pass


class Job:
    def __init__(self, prompt_ids, params, max_new, stops=(), force=None, drafter=None, tag='', n_predict=None,
                 slot_params=None, thinking=False, marks=None, request_id=None, tools=None, parse_tools=True, parallel=True,
                 prefix_text='', profile='', harness=''):
        self.prompt_ids = list(prompt_ids)
        # the flight recorder's (M7 ops): the request id, and what the engine thread needs to judge the reply (the loop flag,
        # tool-call parse failures) and to save a replay bundle
        self.request_id = request_id
        self.tools, self.parse_tools, self.parallel, self.prefix_text = tools, parse_tools, parallel, prefix_text
        self.profile, self.harness = profile, harness
        self.marks = marks              # the prompt cache's message boundaries (template marks) or None
        self.params = params            # SamplingParams (resolved: the seed is fixed)
        self.max_new = int(max_new)
        self.stops = list(stops or ())
        self.force = force              # (trigger_id, forced ids) or None
        self.drafter = drafter
        self.tag = tag
        self.id_task = next(_TASK_IDS)
        self.n_predict = self.max_new if n_predict is None else int(n_predict)
        self.slot_params = slot_params
        self.thinking = bool(thinking)  # the prompt opened a think block: ids up to </think> are reasoning
        self.cancel = threading.Event()
        self.events = queue.Queue()
        self.t_submit = time.perf_counter()
        self.t_start = None
        self.t_end = None
        self.c_submit = self.c_start = None     # the stats clock's stamps (queue time in the metrics)
        self.result = None
        self.stop_matched = None

    def queue_ms(self):
        if self.c_submit is None or self.c_start is None:
            return 0.0
        return (self.c_start - self.c_submit) * 1e3


class BundleJob:
    """a replay bundle saved on the engine thread (the flight recorder's on-demand save: POST /caminus/flight/bundle), in FIFO
    order with the requests: request_id None = the most recent request. events: ('done', meta) or ('error', kind, msg)"""
    def __init__(self, request_id=None, reason='api'):
        self.request_id, self.reason = request_id, reason
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.result = None
        self.t_submit = time.perf_counter()
        self.c_submit = None


class EngineWorker:
    """idle_s: when no job arrived for this many seconds after the last one, the session's idle() runs once on the engine
    thread (the K3 session parks the live conversation in the background); default: the session's idle_s, None = never.

    flight (serve.flight.FlightRecorder or None): replay bundles, saved on the engine thread (the single owner of the engine):
    on demand (a BundleJob), automatically when a pass failed past its retries (before the session is reset), and when a reply's
    reasoning loops (agentmon.LoopDetector) or its tool calls fail to parse (unclosed / undeclared / malformed: judged here on
    the reply's text, the same parser the HTTP thread runs, so no state moves before the save); at most auto_per_hour automatic
    bundles. recent: the last RECENT requests' records (token log, parameters, the prompt-end state) a bundle is made from."""
    RECENT = 8
    def __init__(self, session, tokenizer, max_queue=16, tool_open_id=None, stats=None, think_end_id=None, idle_s='session',
                 flight=None, auto_per_hour=12):
        self.sess = session
        self.flight = flight
        self.auto_per_hour = auto_per_hour
        self._auto_t = collections.deque()
        self.recent = collections.OrderedDict()
        self.model_path = None
        self.bundles = []               # (request id, reason, path or None, note) of every bundle tried
        self.idle_s = session.idle_s if idle_s == 'session' else idle_s
        self._idle_due = False
        self.tok = tokenizer
        self.max_queue = max_queue
        self.tool_open_id = tool_open_id
        self.stats_obj = stats
        self.think_end_id = think_end_id
        self.q = queue.Queue()
        self.busy = False
        self.stats = {'jobs': 0, 'engine_errors': 0, 'internal_errors': 0, 'cancelled': 0, 'prompt_tokens': 0,
                      'cached_tokens': 0, 'generated_tokens': 0, 'passes': 0, 'drafted': 0, 'accepted': 0,
                      'aborts': 0, 'resets': 0}
        self.history = []               # (tag, t_submit, t_start, t_end) of the jobs run (tests read it)
        self.current = None             # the job on the engine (cancel_all)
        self._stop = False
        self.thread = threading.Thread(target=self._loop, name='k3-engine', daemon=True)
        self.thread.start()

    @property
    def depth(self):
        return self.q.qsize() + (1 if self.busy else 0)

    @property
    def deferred(self):
        """jobs waiting for the engine (llama-server's requests_deferred)"""
        return self.q.qsize()

    def submit(self, job):
        if self.q.qsize() >= self.max_queue:
            raise QueueFull(f'{self.q.qsize()} requests queued')
        if self.stats_obj is not None:
            job.c_submit = self.stats_obj.clock()
        self.q.put(job)
        return job

    def shutdown(self):
        self._stop = True
        self.q.put(None)

    def cancel_all(self):
        """a clean shutdown: the job on the engine stops at its next pass boundary (or chunk boundary of its prompt), the
        queued ones end 'cancel' without touching the engine"""
        j = self.current
        if j is not None:
            j.cancel.set()
        with self.q.mutex:
            for j in list(self.q.queue):
                if j is not None:
                    j.cancel.set()

    def stop(self, timeout=None):
        """cancel_all, then end the loop and wait for the engine thread (bounded by timeout); True when it exited"""
        self.cancel_all()
        self.shutdown()
        self.thread.join(timeout)
        return not self.thread.is_alive()

    def _loop(self):
        while True:
            idle = self.sess.idle if self.idle_s else None
            if idle is not None and self._idle_due:
                try:
                    job = self.q.get(timeout=self.idle_s)
                except queue.Empty:
                    self._idle_due = False
                    try:
                        idle()
                    except Exception:   # noqa: BLE001 - an idle park that fails is skipped
                        traceback.print_exc()
                    continue
            else:
                job = self.q.get()
            if job is None:
                return
            if isinstance(job, BundleJob):
                self.busy = True
                try:
                    self._bundle_job(job)
                finally:
                    self.busy = False
                continue
            if self._stop:
                job.cancel.set()        # (shutting down: a job still queued ends 'cancel' without touching the engine)
            self._idle_due = True
            self.busy = True
            self.current = job
            try:
                self._run(job)
            finally:
                self.busy = False
                self.current = None

    def _run(self, job):
        job.t_start = time.perf_counter()
        so = self.stats_obj
        if so is not None:
            job.c_start = so.clock()
        if job.cancel.is_set():
            job.result = GenResult(finish='cancel', prompt_n=len(job.prompt_ids))
            self.stats['cancelled'] += 1
            job.t_end = time.perf_counter()
            job.events.put(('done', job.result))
            return
        self.stats['jobs'] += 1
        try:
            cached = self.sess.plan(job.prompt_ids, max_new=job.max_new)
            job.events.put(('start', {'queue_ms': (job.t_start - job.t_submit) * 1e3,
                                      'prompt_n': len(job.prompt_ids), 'cached_n': cached}))
            tt = TokenText(self.tok, job.stops, self.sess.eos_ids)
            es = so if so is not None and so.enabled else None
            if es is not None:
                es.begin(job, self.sess.max_ctx)
            cc = ContentClass()
            think = [job.thinking and self.think_end_id is not None]
            te = self.think_end_id
            texts = [] if self.flight is not None else None
            marks0 = self._marks()

            def on_tokens(ids):
                text, n_keep = tt.feed(ids)
                if text:
                    job.events.put(('text', text))
                    if texts is not None:
                        texts.append(text)
                if es is not None:
                    kept = ids if n_keep is None else ids[:n_keep]
                    if think[0]:
                        if te in kept:
                            j = kept.index(te) + 1
                            think[0] = False
                            es.tokens(j, len(kept) - j)
                        else:
                            es.tokens(len(kept), 0)
                    else:
                        es.tokens(0, len(kept))
                    if text:
                        es.cls = cc.feed(text)
                return n_keep

            res = run_generation(self.sess, job.prompt_ids, Sampler(job.params), job.max_new, on_tokens,
                                 cancel=job.cancel, drafter=job.drafter, force=job.force,
                                 tool_open_id=self.tool_open_id, stats=es, marks=job.marks)
            if res.finish != 'stop':
                tail = tt.finish()
                if tail:
                    job.events.put(('text', tail))
                    if texts is not None:
                        texts.append(tail)
            job.stop_matched = tt.stop.matched
            job.result = res
            self._counts(res, marks0)
            st = self.stats
            st['prompt_tokens'] += res.prompt_n
            st['cached_tokens'] += res.cached_n
            st['generated_tokens'] += len(res.gen_ids)
            st['passes'] += res.passes + res.prefill_passes
            st['drafted'] += res.drafted
            st['accepted'] += res.accepted
            st['cancelled'] += res.finish == 'cancel'
            job.t_end = time.perf_counter()
            self.history.append((job.tag, job.t_submit, job.t_start, job.t_end))
            if es is not None:
                es.end(job, res)
            rec = self._record(job, res) if self.flight is not None else None
            if rec is not None:
                rec['verdict'] = self._verdict(job, ''.join(texts), res)     # (before 'done': the HTTP thread's line reads it)
            job.events.put(('done', res))
            if rec is not None and rec['verdict'] is not None and self._auto_ok():
                try:
                    self._bundle(rec, rec['verdict'])
                except Exception:   # noqa: BLE001 - a bundle never fails the request (already answered)
                    traceback.print_exc()
        except EngineError as e:
            self.stats['engine_errors'] += 1
            self.stats['resets'] += 1
            if so is not None:
                so.error(job, 'engine')
                so.bail(False)
            if self.flight is not None:             # (before the reset drops the VRAM tier: the prompt-end state is still there)
                try:
                    rec = self._record(job, None)
                    if self._auto_ok():
                        self._bundle(rec, 'retry_exhausted')
                except Exception:   # noqa: BLE001 - a bundle never changes the error path
                    traceback.print_exc()
            self.sess.reset()
            job.t_end = time.perf_counter()
            job.events.put(('error', 'engine', str(e)))
        except Exception as e:      # noqa: BLE001 - the worker must survive anything a request triggers
            self.stats['internal_errors'] += 1
            if so is not None:
                so.error(job, 'internal')
            traceback.print_exc()
            try:
                self.sess.abort()
                self.stats['aborts'] += 1
            except Exception:       # noqa: BLE001 - the state is not consistent: drop it
                traceback.print_exc()
                self.stats['resets'] += 1
                try:
                    self.sess.reset()
                except Exception:   # noqa: BLE001
                    pass
            job.t_end = time.perf_counter()
            job.events.put(('error', 'internal', f'{type(e).__name__}: {e}'))

    # ------------------------------------------------------------------------------------------------ the flight recorder
    def _marks(self):
        try:
            return self.sess.request_marks()
        except Exception:   # noqa: BLE001
            return {}

    def _counts(self, res, marks0):
        """the request's own bails / recoveries (the session's counters moved during it)"""
        m1 = self._marks()
        res.engine_counts = {k: m1[k] - marks0.get(k, 0) for k in m1}

    def _record(self, job, res):
        """the request's record for a bundle (kept for the last RECENT requests)"""
        import numpy as np
        s = self.sess
        st = dict(s.bundle_state(job.prompt_ids))
        sgen = st.pop('generated', None) or []
        gen = list(res.gen_ids) if res is not None else list(sgen)
        rec = dict(request_id=job.request_id, prompt=np.asarray(job.prompt_ids, dtype=np.int32), generated=[int(t) for t in gen],
                   finish=res.finish if res is not None else 'error', params=dataclasses.asdict(job.params), max_new=job.max_new,
                   stops=list(job.stops), force=None if job.force is None else [job.force[0], list(job.force[1])], marks=job.marks,
                   thinking=job.thinking, profile=job.profile, route=job.tag, n_predict=job.n_predict, engine_counts=getattr(res, 'engine_counts', None),
                   **st)
        if job.request_id is not None:
            self.recent[job.request_id] = rec
            self.recent.move_to_end(job.request_id)
            while len(self.recent) > self.RECENT:
                self.recent.popitem(last=False)
        return rec

    def _auto_ok(self):
        now = time.time()
        q = self._auto_t
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= self.auto_per_hour:
            return False
        q.append(now)
        return True

    def _verdict(self, job, text, res):
        """the engine thread's verdict on a finished reply: 'loop' (the reasoning repeated a line LoopDetector.REPEAT times),
        'tool_parse' (a tool call unclosed, undeclared or malformed; a call cut by max_tokens is not a parse failure), or None"""
        try:
            reason = None
            if job.thinking:
                ld = LoopDetector(True)
                ld.feed(text)
                if ld.finish()[0]:
                    reason = 'loop'
            if reason is None and job.tools and job.parse_tools:
                from .toolparse import OutputParser, ToolSet
                p = OutputParser(ToolSet(job.tools), thinking=job.thinking, parse_tools=True, parallel=job.parallel)
                if job.prefix_text:
                    p.feed(job.prefix_text)
                p.feed(text)
                p.finish()
                if res.finish in ('length', 'stop'):
                    for c in p.msg.tool_calls:
                        if not c.complete:
                            c.cut = True
                _, fails, _ = tool_call_report(p.msg, p.notes)
                if fails['unclosed'] or fails['undeclared'] or fails['malformed']:
                    reason = 'tool_parse'
            return reason
        except Exception:   # noqa: BLE001 - a verdict never fails the request
            traceback.print_exc()
            return None

    def _bundle(self, rec, reason):
        """a replay bundle of rec (the engine thread): the session writes the start state and the first logits"""
        fl = self.flight
        path = fl.new_bundle(rec['request_id'] or 'none', reason)
        meta = {k: v for k, v in rec.items() if k not in ('prompt', 'first', 'pe', 'source', 'cached', 'prompt_sha')}
        meta = dict(request_id=rec['request_id'], reason=reason, model_path=self.model_path,
                    request=dict(meta, prompt=[int(t) for t in rec['prompt']]))
        meta.update(self.sess.export_bundle(path, rec, fl))
        out = fl.finish_bundle(path, meta, first_logits=rec.get('first'))
        self.bundles.append((rec['request_id'], reason, path, (out.get('cache_ref') or {}).get('note')))
        return path, out

    def _bundle_job(self, job):
        try:
            if self.flight is None:
                raise RuntimeError('the flight recorder is off (caminus serve --flight-dir)')
            rid = job.request_id
            if rid is None:
                if not self.recent:
                    raise LookupError('no request to save yet')
                rid = next(reversed(self.recent))
            rec = self.recent.get(rid)
            if rec is None:
                raise LookupError(f'request {rid} is not among the last {self.RECENT} requests')
            path, meta = self._bundle(rec, job.reason)
            job.result = dict(path=path, request_id=rid, reason=job.reason, cache_ref=meta.get('cache_ref'))
            job.events.put(('done', job.result))
        except LookupError as e:
            job.events.put(('error', 'not_found', str(e)))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            job.events.put(('error', 'internal', f'{type(e).__name__}: {e}'))
