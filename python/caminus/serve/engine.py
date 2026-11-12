"""The engine session interface the server drives (server spec section 2 and 5, steps S0 / S4; AGENTIC-PLAN M6/M7),
and MockSession, its CPU implementation over mock_engine.MockEngine.

EngineSession is the live sequence of a batch-1 engine. The engine worker thread (worker.py) is its only caller.
A session keeps a TOKEN LOG: the ids whose state the live sequence holds, all of them processed. Generated tokens
are appended as they are fed back; the last sampled token of a completion (EOS, or the token a stop ended on) is not
fed, so after a completion the log is prompt + reply[:-1].

    plan(ids, max_new=None)    -> cached_n: how many leading ids prefill() would reuse (no side effect). max_new: the
                                  request's generation limit (the server always passes it; the K3 session's side-slot routing
                                  uses it, MockSession ignores it)
    prefill(ids, cancel, marks=None, max_new=None)
                               -> logits [V] for the token after ids. marks: the prompt cache's message boundaries
                                  (template.PromptBuilder.build(marks=True).marks; sessions that take none ignore them;
                                  `wants_marks` tells the server to compute them). The live sequence becomes ids: the session
                                  restores the deepest state it holds that is a prefix of ids[:-1] (the live state, or
                                  a checkpoint), then processes the rest in passes of at most ntok tokens. Returns None
                                  when cancel is set between passes (the log then holds the part already processed).
                                  last_prefill: PrefillStats.
    verify(feed)               one pass appending feed (1 .. ntok tokens) provisionally, its logits not returned: the
                                  server's forced tokens beyond one pass (processed like a prompt). Returns None.
    accept(n)                  keep the first n tokens of the last pass's feed (the rest are rolled back). Every
                                  verify / sample_pass must be followed by an accept before the next pass / end.
    sample_pass(feed, drafts, sampler, history, stop_ids)
                               -> the emitted ids of one pass over feed + drafts. The base class samples on the host:
                                  pass_rows(feed + drafts) -> logits [len, V] (row j predicts the token after
                                  log + body[:j + 1]), then sample-and-match on the rows from the last feed token on. An
                                  engine with a device sampler (M7) overrides sample_pass and needs no pass_rows.
    end()                      completion end: the state is made exact and checkpointed (checkpoint c)
    reset()                    drop the live state and every checkpoint (after an EngineError)
    abort()                    after a host-side exception (not an EngineError) between calls: roll back an
                                  unaccepted verify, make the state exact at the committed log, keep every
                                  checkpoint. Raises if the state cannot be made consistent (the caller then resets).
                                  The base class resets (always safe); an engine that can roll back overrides it.
    tokens, ntok, max_ctx, n_vocab, eos_ids, last_prefill
  Declared with a default on EngineSession (a session overrides what it has; the server never probes for them):
    stats, wants_marks (prefill uses marks: the server computes them), drafter (a session-bound drafter, else the
    server's own), drafts_real (False: the coming pass's drafts are filler, not counted), idle_s / idle() (the worker's
    idle timer: idle() after idle_s seconds without a request; None: never), shutdown_park() (the clean shutdown's park:
    None = nothing parked), request_marks() (the session's bail / recovery counters, for a request's flight line),
    bundle_state(prompt) / export_bundle(path, rec, flight) (the flight recorder's replay bundles: the start state).
  contract_problems(sess) lists what a session lacks (tests/serve/test_session.py; K3Session in
  tests/cache/test_prompt_cache.py cpu).

Errors: EngineError (a pass failed: bail retries exhausted). The live state is then undefined; the caller resets the
session and answers 503 / 529.

Metrics (docs/specs/spec-metrics.md): `stats` is the server's stats.EngineStats (None: nothing recorded). A session
reports through it from the engine thread only, with host timestamps and counters: prefill_progress(n) per prompt
pass, cached_known(n) once the reused prefix is known, pass_split(k3_ms, sampler_ms, cards) for each sample_pass (the base
class times its pass_rows and its host sampling; an engine with a device sampler reports its own; cards: per-card pass and
exchange-wait ms, link bytes, for engines that know them), and the engine-only hooks (bail, retry, refusal,
evicted, cache_update with the prompt cache's own counter names). last_prefill carries the TTFT split (restore,
batched, hand-off, K3 passes), the cache source in the prompt cache's names and the LCP.

MockSession models a recurrent state strictly: MockEngine raises on any pass that does not continue its committed
state, so a wrong accept, a wrong restore or a stale cache entry fails loudly instead of returning a plausible
answer. Its checkpoints are snapshots of the committed token list, taken at (b) each prompt end and (c) each
completion end, LRU-bounded; a restore picks the longest checkpoint that is a prefix of the new prompt. There is no
free rewind: when the live state is not a prefix of the prompt and no checkpoint is, the prompt is processed cold.
Checkpoint (a, the system + tools root), parking and saved prefix files are step S5's."""
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from .mock_engine import EngineStateError, MockEngine

CACHE_SOURCE = {'live': 'live', 'checkpoint': 'vram', 'disk': 'disk', 'cold': 'cold'}


class EngineError(RuntimeError):
    """a pass failed (retries exhausted); the live state is lost"""


@dataclass
class PrefillStats:
    prompt_n: int = 0           # prompt tokens
    cached_n: int = 0           # reused from the live state or a checkpoint
    processed_n: int = 0        # processed by this prefill
    passes: int = 0
    ms: float = 0.0             # wall time of prefill(), restore included
    restore: str = 'cold'       # 'live' | 'checkpoint' | 'cold' ('disk': MockSession's opt-in disk tier)
    restore_ms: float = 0.0
    cancelled: bool = False
    source: str = 'cold'        # the prompt cache's name: 'live' | 'vram' | 'spill' | 'disk' | 'cold'
    exact: bool = False         # an exact hit (no new token past the cached state)
    lcp: int = 0                # longest common prefix with the live token log
    batched_ms: float = 0.0     # TTFT split: the batched prefill, its hand-off, the K3 prompt passes
    handoff_ms: float = 0.0
    passes_ms: float = 0.0
    slot: str = 'main'          # 'side': the request ran in the prompt cache's side slot (K3Session's side-slot routing)


def common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class EngineSession:
    """the interface (see the module docstring): every member the server and the worker use, declared here with its
    default; sample_pass has the host default (over pass_rows)"""
    ntok = 1
    max_ctx = 0
    n_vocab = 0
    eos_ids = ()
    last_prefill = None     # PrefillStats of the last prefill()
    stats = None            # serve.stats.EngineStats (the server sets it), or None: nothing recorded
    wants_marks = False     # True: prefill() uses the prompt cache's marks, so the server computes them
    drafter = None          # a session-bound drafter(tokens, k) (the K3 session's), or None: the server's own
    drafts_real = True      # False: the coming pass's drafts are filler (run_generation counts no drafts for it)
    idle_s = None           # seconds without a request before the worker calls idle() (None: never)

    @property
    def tokens(self):
        raise NotImplementedError

    def plan(self, ids, max_new=None, hint=None):
        raise NotImplementedError

    def prefill(self, ids, cancel=None, marks=None, max_new=None, hint=None):
        raise NotImplementedError

    def verify(self, feed):
        raise NotImplementedError

    def pass_rows(self, feed):
        """host sampling's pass (the base sample_pass): logits [len(feed), V]; a device-sampler session needs none"""
        raise NotImplementedError

    def accept(self, n):
        raise NotImplementedError

    def end(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def abort(self):
        self.reset()

    def sample_pass(self, feed, drafts, sampler, history, stop_ids=()):
        """history: every token before the one being sampled at the first row used (it ends with feed)"""
        st = self.stats
        if st is None or not st.enabled:
            rows = self.pass_rows(list(feed) + list(drafts))
            return sampler.sample_and_match(rows[len(feed) - 1:], list(drafts), history, stop_ids)
        clock = st.clock
        t0 = clock()
        rows = self.pass_rows(list(feed) + list(drafts))
        t1 = clock()
        out = sampler.sample_and_match(rows[len(feed) - 1:], list(drafts), history, stop_ids)
        st.pass_split((t1 - t0) * 1e3, (clock() - t1) * 1e3)
        return out

    # -- optional parts, each with the default of a session that has no such state -------------------------------------------
    def idle(self):
        """no request for idle_s seconds (the worker's idle timer): nothing to park"""
        return None

    def shutdown_park(self):
        """a clean shutdown's park (caminus serve): None, nothing parked"""
        return None

    def request_marks(self):
        """the session's own counters a request's flight line reports as deltas (bails, recoveries): none"""
        return {}

    def bundle_state(self, prompt=None):
        """the flight recorder's record of the request's start state: none"""
        return {}

    def export_bundle(self, path, rec, flight):
        """a replay bundle's start state written under path: none (the bundle replays from cold)"""
        return dict(cache_ref=dict(source=rec.get('source'), state=None, note='this session keeps no engine state (MockSession)'))


CONTRACT_ATTRS = ('tokens', 'ntok', 'max_ctx', 'n_vocab', 'eos_ids', 'last_prefill', 'stats', 'wants_marks', 'drafter',
                  'drafts_real', 'idle_s')
CONTRACT_CALLS = {'plan': ('ids', 'max_new'), 'prefill': ('ids', 'cancel', 'marks', 'max_new'), 'verify': ('feed',),
                  'accept': ('n',), 'end': (), 'reset': (), 'abort': (), 'idle': (), 'shutdown_park': (), 'request_marks': (),
                  'sample_pass': ('feed', 'drafts', 'sampler', 'history', 'stop_ids'), 'bundle_state': ('prompt',),
                  'export_bundle': ('path', 'rec', 'flight')}


def contract_problems(sess):
    """what sess lacks of the session contract (the module docstring), as strings: [] for a conforming session. sess: an
    instance (every attribute readable, the sizes positive) or a class (K3Session where no engine can be built: the calls
    and the sampling route only). Every call takes its parameters by name; a session samples by overriding sample_pass (a
    device sampler) or by pass_rows (the base class's host sampling)"""
    import inspect
    cls = sess if isinstance(sess, type) else type(sess)
    out = [] if issubclass(cls, EngineSession) else ['not an EngineSession']
    if not isinstance(sess, type):
        for a in CONTRACT_ATTRS:
            try: getattr(sess, a)
            except Exception as e:   # noqa: BLE001
                out.append(f'{a}: {type(e).__name__}: {e}')
        for a in ('ntok', 'max_ctx', 'n_vocab'):
            v = getattr(sess, a, None)
            if not isinstance(v, int) or v < 1: out.append(f'{a}: {v!r} (a positive int)')
        if not isinstance(getattr(sess, 'eos_ids', None), tuple): out.append('eos_ids: not a tuple')
    for name, params in CONTRACT_CALLS.items():
        fn = getattr(sess, name, None)
        if not callable(fn):
            out.append(f'{name}: not callable'); continue
        have = inspect.signature(fn).parameters
        out += [f'{name}: no parameter {p!r}' for p in params if p not in have]
    if getattr(cls, 'sample_pass', None) is EngineSession.sample_pass and getattr(cls, 'pass_rows', None) is EngineSession.pass_rows:
        out.append('neither sample_pass (a device sampler) nor pass_rows (host sampling) is implemented')
    return out


class MockSession(EngineSession):
    """EngineSession over a MockEngine (or a subclass such as ScriptedEngine).

    pass_sleep: seconds slept per pass (to make cancels, queues and pings observable); fail_at: a set of pass numbers
    (1-based, counted over the session's life) at which verify raises EngineError, or a callable(pass_no) -> bool;
    checkpoints: LRU size.

    Metrics: stats (stats.EngineStats or None, usually set by the server); synth (mock_engine.SynthProfile or None):
    the engine-internal numbers (per-card split, link bytes, batched prefill and hand-off, restore times, cache tier
    sizes, occasional bails) and, with synth.clock, the deterministic test mode. disk_slots: an opt-in second
    checkpoint tier standing in for the prompt cache's disk tier (a VRAM checkpoint the LRU drops goes there; 0: off,
    the behaviour the server tests were written against)."""

    def __init__(self, engine=None, max_ctx=2051, eos_ids=(248046, 248044), pass_sleep=0.0, fail_at=None,
                 checkpoints=8, stats=None, synth=None, disk_slots=0):
        self.m = engine if engine is not None else MockEngine()
        self.ntok = self.m.ntok
        self.n_vocab = self.m.V
        self.max_ctx = max_ctx
        self.eos_ids = tuple(eos_ids)
        self.pass_sleep = pass_sleep
        self.fail_at = fail_at
        self.n_checkpoints = checkpoints
        self.disk_slots = disk_slots
        self.stats = stats
        self.synth = synth
        self.total_passes = 0
        self.decode_passes = 0
        self.last_prefill = PrefillStats()
        self.cache_stats = dict.fromkeys(('requests', 'prompt_tokens', 'reused', 'source_live', 'source_vram',
                                          'source_spill', 'source_disk', 'source_cold', 'checkpoints', 'evicted',
                                          'spilled', 'unparked', 'refused'), 0)
        self._lock = threading.Lock()       # only to catch misuse from two threads
        self.reset()

    # ------------------------------------------------------------------------------------------------ state
    def reset(self):
        self.m.reset()
        self._log = []              # accepted tokens (state)
        self._feed = None           # the last verify's feed, until accept()
        self._commit = 0            # MockEngine lazy commit owed by the next pass / flush
        self._cps = OrderedDict()   # tuple(tokens) -> True (LRU)
        self._disk = OrderedDict()  # the opt-in disk tier (LRU)

    @property
    def tokens(self):
        return tuple(self._log)

    @property
    def checkpoints(self):
        return [len(k) for k in self._cps]

    def _clock(self):
        st = self.stats
        return st.clock if st is not None else time.perf_counter

    def _spend(self, seconds):
        """synthetic engine time: advances synth.clock, or sleeps on the real clock when synth.sleep"""
        sy = self.synth
        if sy is None or seconds <= 0:
            return
        if sy.clock is not None:
            sy.clock.advance(seconds)
        elif sy.sleep:
            time.sleep(seconds)

    def _flush(self):
        if self._feed is not None:
            raise EngineStateError('verify without accept')
        self.m.flush(self._commit)
        self._commit = 0
        if list(self.m.state) != self._log:
            raise EngineStateError('token log and engine state disagree')

    def _insert(self, key):
        self._cps[key] = True
        self._cps.move_to_end(key)
        while len(self._cps) > self.n_checkpoints:
            old, _ = self._cps.popitem(last=False)
            self.cache_stats['evicted'] += 1
            if self.disk_slots:
                self._disk[old] = True
                self._disk.move_to_end(old)
                self.cache_stats['spilled'] += 1
                while len(self._disk) > self.disk_slots:
                    self._disk.popitem(last=False)

    def _checkpoint(self):
        self._flush()
        key = tuple(self._log)
        if not key:
            return
        if key not in self._cps:
            self.cache_stats['checkpoints'] += 1
        self._insert(key)

    def _best(self, ids):
        """(start, source): the deepest reusable state for ids (it must leave the last prompt token to process)"""
        limit = len(ids) - 1
        n_live = len(self._log)
        if 0 < n_live <= limit and common_prefix(self._log, ids) == n_live:
            best, src = n_live, 'live'
        else:
            best, src = 0, 'cold'
        for k in self._cps:
            if best < len(k) <= limit and tuple(ids[:len(k)]) == k:
                best, src = len(k), 'checkpoint'
        for k in self._disk:
            if best < len(k) <= limit and tuple(ids[:len(k)]) == k:
                best, src = len(k), 'disk'
        return best, src

    def plan(self, ids, max_new=None, hint=None):   # (hint: no side slot here)
        return self._best(ids)[0]

    def cache_gauges(self):
        sy = self.synth
        ck = sy.ckpt_mib if sy is not None else None
        return {'n_slots': self.n_checkpoints, 'slots_used': len(self._cps),
                'tier_mib': None if ck is None else round(self.n_checkpoints * ck, 1),
                'headroom_mib': None if sy is None else sy.headroom_mib,
                'disk_bytes': None if ck is None else int(len(self._disk) * ck * (1 << 20)),
                'disk_cap_bytes': None if ck is None or not self.disk_slots else int(30e9)}

    def _report_cache(self):
        if self.stats is not None:
            self.stats.cache_update(self.cache_stats, self.cache_gauges())

    # ------------------------------------------------------------------------------------------------ passes
    def _pass(self, feed, batched=False):
        self.total_passes += 1
        fa = self.fail_at
        if fa is not None and (fa(self.total_passes) if callable(fa) else self.total_passes in fa):
            raise EngineError(f'mock bail at pass {self.total_passes}')
        sy = self.synth
        if sy is not None and batched:
            self._spend(len(feed) / sy.batched_tok_s)
        elif sy is not None and sy.clock is not None:
            self._spend(sy.pass_ms(len(feed)) / 1e3)
        elif self.pass_sleep:
            time.sleep(self.pass_sleep)
        seq = self._log + list(feed)
        return self.m.verify(seq, len(self._log), self._commit)

    def _verify(self, feed, batched=False):
        if self._feed is not None:
            raise EngineStateError('verify without accept')
        feed = [int(t) for t in feed]
        if not 1 <= len(feed) <= self.ntok:
            raise EngineStateError(f'pass of {len(feed)} tokens with ntok={self.ntok}')
        if len(self._log) + len(feed) > self.max_ctx:
            raise EngineStateError(f'pass beyond max_ctx ({len(self._log) + len(feed)} > {self.max_ctx})')
        rows = self._pass(feed, batched)
        self._commit = 0            # applied by the pass
        self._feed = feed
        return rows

    def verify(self, feed):
        self._verify(feed)

    def pass_rows(self, feed):
        return self._verify(feed)

    def _accept(self, n):
        if self._feed is None:
            raise EngineStateError('accept without verify')
        if not 0 <= n <= len(self._feed):
            raise EngineStateError(f'accept {n} of a {len(self._feed)}-token pass')
        self._log += self._feed[:n]
        self._commit = n
        self._feed = None

    def accept(self, n):
        self._accept(n)
        sy = self.synth
        if sy is not None and sy.clock is not None:
            sy.clock.advance(sy.host_ms / 1e3)      # the generation loop's host share of a pass (synthetic)

    def sample_pass(self, feed, drafts, sampler, history, stop_ids=()):
        out = super().sample_pass(feed, drafts, sampler, history, stop_ids)
        sy, st = self.synth, self.stats
        if sy is not None and st is not None and st.enabled:
            smp = st.pass_sampler_ms
            if sy.clock is not None:
                sy.clock.advance(sy.sampler_ms / 1e3)
                smp = sy.sampler_ms
            k3 = st.pass_k3_ms
            n = len(feed) + len(drafts)
            st.pass_split(k3, smp, (tuple(k3 * f for f in sy.card_split), tuple(k3 * f for f in sy.wait_split), sy.link_bytes_per_row * n))
            self.decode_passes += 1
            if sy.bail_every and self.decode_passes % sy.bail_every == 0:
                st.bail(True)
                st.retry()
                st.evicted(sy.evicted_ms)
        return out

    def prefill(self, ids, cancel=None, marks=None, max_new=None, hint=None):
        with self._lock:
            clock = self._clock()
            st_ = self.stats
            sy = self.synth
            t0 = clock()
            ids = [int(t) for t in ids]
            if not ids:
                raise ValueError('empty prompt')
            if len(ids) > self.max_ctx:
                raise EngineStateError(f'prompt of {len(ids)} tokens > max_ctx {self.max_ctx}')
            st = PrefillStats(prompt_n=len(ids))
            st.lcp = common_prefix(self._log, ids)
            start, src = self._best(ids)
            if src == 'live':
                if self._feed is not None:
                    raise EngineStateError('verify without accept')
            else:
                tr = clock()
                self.m.reset()
                self._feed, self._commit = None, 0
                self._log = list(ids[:start])
                if start:
                    self.m._state = list(self._log)          # restore the snapshot (mock: the token list)
                    key = tuple(self._log)
                    if src == 'disk':
                        del self._disk[key]
                        self.cache_stats['unparked'] += 1
                        self._insert(key)
                    else:
                        self._cps.move_to_end(key)
                if sy is not None:
                    self._spend(sy.restore_ms.get(CACHE_SOURCE[src], 0.0) / 1e3)
                st.restore_ms = (clock() - tr) * 1e3
            st.restore, st.cached_n, st.source = src, start, CACHE_SOURCE[src]
            cs = self.cache_stats
            cs['requests'] += 1
            cs['prompt_tokens'] += len(ids)
            cs['reused'] += start
            cs['source_' + st.source] += 1
            if st_ is not None:
                st_.cached_known(start)
            # the synthetic batched prefill: every prompt pass but the last when the new tokens reach T*
            n_new = len(ids) - start
            end_b = start
            if sy is not None and n_new >= sy.crossover:
                end_b = len(ids) - ((n_new - 1) % self.ntok + 1)
            rows = None
            i = start
            tb = clock()
            while i < end_b and not st.cancelled:
                if cancel is not None and cancel.is_set():
                    st.cancelled = True
                    break
                feed = ids[i:i + self.ntok]
                self._verify(feed, batched=True)
                self._accept(len(feed))
                st.passes += 1
                i += len(feed)
                if st_ is not None:
                    st_.prefill_progress(i - start)
            if end_b > start and not st.cancelled:
                st.batched_ms = (clock() - tb) * 1e3
                th = clock()
                self._spend(sy.handoff_ms / 1e3)
                st.handoff_ms = (clock() - th) * 1e3
            tp = clock()
            while i < len(ids) and not st.cancelled:
                if cancel is not None and cancel.is_set():
                    st.cancelled = True
                    break
                feed = ids[i:i + self.ntok]
                rows = self._verify(feed)
                self._accept(len(feed))
                st.passes += 1
                i += len(feed)
                if st_ is not None:
                    st_.prefill_progress(i - start)
            st.passes_ms = (clock() - tp) * 1e3
            st.processed_n = i - start
            if not st.cancelled:
                self._checkpoint()      # (b) the prompt end
            st.ms = (clock() - t0) * 1e3
            self.last_prefill = st
            self._report_cache()
            return None if st.cancelled else rows[-1]

    def abort(self):
        with self._lock:
            if self._feed is not None:
                self._accept(0)         # drop the provisional pass
            self._flush()               # strict: raises EngineStateError if the log and the state disagree

    def end(self):
        with self._lock:
            self._checkpoint()          # (c) the completion end
            self._report_cache()
