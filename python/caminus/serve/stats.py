"""The engine stats hook and the metric recorder behind /metrics, /slots and /caminus/status
(docs/specs/spec-metrics.md; AGENTIC-PLAN 2026-11-04 'llm-top support' and 'the full Caminus metric set').

EngineStats is one object per server. The engine thread (worker.py, generate.py, the EngineSession) is the only writer
of its engine-side fields and calls the hook methods below at points the loop already has: host timestamps from
`clock` and integer counters, never a device sync. HTTP threads write only the agent-side fields (User-Agent classes,
tool calls, the loop detector), the hardware sampler thread only the host counters (host_sample: KFD evictions, page
migrations). Every writer holds `lock` for its few adds (M7 ops: a scrape used to read the rings while the engine wrote
them, and take_bucket's swap lost the engine's concurrent adds). Readers (GET /metrics, /slots, /caminus/status) take
snapshot() under the same lock (a copy of the counters and the rings, ~0.1 ms) and compute quantiles and format the text
on their own thread after it is released, so a scrape never holds the engine thread for more than the copy (tests/serve/
test_stats.py: the cost per pass and per request; tests/e2e/test_serve_k3.py scrape: decode with and without a 10 Hz scraper).

The hook (engine thread; every method is a no-op when `enabled` is False, so the M7 <= 3 % gate can A/B it):
    begin(job)                         a request reached the engine (job: worker.Job)
    prefill_progress(processed)        prompt tokens processed so far by this request (live /slots progress)
    prefill_done(pf)                   the prompt is processed (pf: engine.PrefillStats, with its TTFT split)
    first_token()                      the first token of the reply was emitted (ends TTFT)
    pass_done(t0, draft_ms, fed, drafted, accepted, emitted)
                                       one decode pass, from its drafter call (t0) to its emitted tokens' bookkeeping;
                                       the session's K3 / sampler split and per-card numbers: its pass_split()
    pass_split(k3_ms, sampler_ms, cards)
                                       the session's report of the pass in flight (EngineSession.sample_pass)
    tokens(n_reasoning, n_answer)      generated ids by kind (the worker counts them around </think>)
    forced(n)                          n forced tool-call ids emitted (n_decoded / n_remain / position, no pass record)
    end(job, res) / error(job, kind)   the request is over
  engine-only (M7; MockSession synthesizes them):
    bail(recovered), retry(), refusal(kind, n), evicted(ms), pgmigrate(n),
    cache_update(counters, gauges)     the prompt cache's own names (docs/specs/spec-cache-api.md, Metrics)
    route(decision)                    the side-slot routing decision of a request (PromptCache.route's dict: route 'main' /
                                       'side', reason, shared, root, need, main_n, side_rows; impl:m7-finish)
    plain_estimate(ms)                 a plain pass's ms per token estimated by the session (DraftPolicy: a DFlash pass less
                                       the drafter step's device time) while no plain pass is measured
  the hardware sampler thread (hw.HwReader, 1 Hz; M7 ops):
    host_sample(evicted_ms, pgmigrate, busy)
                                       this process's KFD queue-eviction ms since the last sample (the largest per-node
                                       rise: one eviction stops every node at once) and the system's page migrations
                                       (/proc/vmstat pgmigrate_success), counted in pgmigrate when the engine was busy

VirtualClock: a deterministic clock for tests (MockSession(synth=...) advances it by synthetic durations), so a
fixed request sequence gives byte-exact metrics."""
import bisect
import math
import os
import threading
import time
from dataclasses import dataclass, field

STATUS_VERSION = 1
ENGINE = 'caminus'
PREFIX = 'caminus:'

CLASSES = ('prose', 'code', 'shell', 'tool')                     # DFlash content classes (agentmon.ContentClass)
SOURCES = ('live', 'vram', 'spill', 'disk', 'cold')               # PromptCache.plan / prepare 'source'
TIER_OF = {'live': 'vram', 'vram': 'vram', 'spill': 'disk', 'disk': 'disk'}
FINISHES = ('eos', 'stop', 'length', 'cancel', 'error')
REFUSALS = ('routing_flag', 'bounds', 'context', 'prefill', 'cache', 'capacity')
PASS_PARTS = ('k3', 'drafter', 'sampler', 'host', 'total')
TTFT_PARTS = ('queue', 'restore', 'batched', 'handoff', 'passes', 'host', 'total')
# the prompt cache's counters (PromptCache.stats keys, spec-cache-api.md 'Metrics'), exported as events
CACHE_EVENTS = ('checkpoints', 'evicted', 'spilled', 'parked', 'unparked', 'park_failed', 'refused', 'side_requests',
                'side_ends', 'main_lost', 'external_reset')
# side-slot routing (PromptCache.route): the reasons of each route
ROUTE_REASONS = {'main': ('match', 'no_main', 'no_fit', 'main_shorter', 'parked', 'off'), 'side': ('unrelated',)}
PHASES = ('idle', 'prefill', 'decode')

# bucket edges: milliseconds (~x2, 0.05 ms .. 65 s) and small counts (tokens per pass)
MS_EDGES = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
            65536)
COUNT_EDGES = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 64)
TOKEN_EDGES = (16, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
RATE_EDGES = (10, 30, 100, 300, 1000, 2000, 3000, 5000, 10000, 30000)


class VirtualClock:
    """a clock for tests: time moves only by advance(seconds)"""

    def __init__(self, t=0.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@dataclass
class PhysicsConfig:
    """the bandwidth floor of a decode token. Per card: the bytes one token streams (weights at the engine's TP split)
    and the card's measured streaming bandwidth. mode 'max': the floor is the slowest card's bytes / bandwidth (the
    cards stream in parallel, a pass waits for both); 'sum': all bytes / summed bandwidth (only a perfectly balanced
    split reaches it). Defaults: the Coder IQ1_M at TP=2 (docs/specs/spec-gsq-port.md section 2: 3.60 / 2.87 GB per
    token at 881 / 774 GB/s -> 4.09 ms, mode 'max'; 'sum' gives 3.91 ms).
    kv_bytes_per_token: per card, KV + indexer rows (AGENTIC-PLAN 0.5: 13,056 B of KV across both cards, 6,528 each,
    plus 2,040 B of indexer replicated). qsa_from: the first position QSA attention serves (capacity.QSA_EXACT 2051:
    dense == QSA below it). guard_mib: tools/guard.sh's kill lines per card; reserve_mib: the XTX's kept 0.5 GB.
    plain_ms_per_token: a plain (undrafted, T=1) token's time when no such pass was measured yet (None: unknown)."""
    card_bytes: tuple = (3.60e9, 2.87e9)
    card_bw: tuple = (881e9, 774e9)
    mode: str = 'max'
    kv_bytes_per_token: tuple = (8568, 8568)
    qsa_from: int = 2051
    guard_mib: tuple = (23800, 19850)
    reserve_mib: tuple = (512, 0)
    card_names: tuple = ('XTX', 'XT')
    plain_ms_per_token: float = None
    source: str = 'spec'        # 'spec' (the defaults above) or 'engine' (K3Session: card bytes from the engine's weights)

    def floor_ms(self):
        if self.mode == 'sum':
            return sum(self.card_bytes) / sum(self.card_bw) * 1e3
        return max(b / w for b, w in zip(self.card_bytes, self.card_bw)) * 1e3


def build_info():
    """'caminus-<short sha>' from the repository's .git (read directly, no subprocess), else 'caminus-unknown'"""
    here = os.path.dirname(os.path.abspath(__file__))
    git = os.path.join(here, '..', '..', '..', '.git')
    try:
        with open(os.path.join(git, 'HEAD')) as f:
            head = f.read().strip()
        if head.startswith('ref: '):
            ref = head[5:]
            p = os.path.join(git, ref)
            if os.path.exists(p):
                with open(p) as f:
                    head = f.read().strip()
            else:
                head = ''
                with open(os.path.join(git, 'packed-refs')) as f:
                    for line in f:
                        if line.rstrip().endswith(' ' + ref):
                            head = line.split()[0]
        if len(head) >= 7 and all(c in '0123456789abcdef' for c in head[:7]):
            return f'{ENGINE}-{head[:7]}'
    except OSError:
        pass
    return f'{ENGINE}-unknown'


class Series:
    """one series: a preallocated ring of the last `window` values (quantiles on read) and, since start, counts per
    bucket, sum, count and max. add() is the only writer (one thread)."""
    __slots__ = ('name', 'unit', 'edges', 'ring', 'i', 'n', 'counts', 'sum', 'count', 'max')

    def __init__(self, name, unit='ms', window=1024, edges=MS_EDGES):
        self.name, self.unit, self.edges = name, unit, tuple(edges)
        self.ring = [0.0] * window
        self.i = self.n = 0
        self.counts = [0] * (len(self.edges) + 1)
        self.sum = 0.0
        self.count = 0
        self.max = 0.0

    def add(self, v):
        r, i = self.ring, self.i
        r[i] = v
        i += 1
        self.i = 0 if i == len(r) else i
        if self.n < len(r):
            self.n += 1
        self.counts[bisect.bisect_left(self.edges, v)] += 1
        self.sum += v
        self.count += 1
        if v > self.max:
            self.max = v

    def window(self):
        """the last values, oldest first"""
        r, n, i = list(self.ring), self.n, self.i
        if n < len(r):
            return r[:n]
        return r[i:] + r[:i]

    def quantiles(self, qs=(0.5, 0.95)):
        w = sorted(self.window())
        if not w:
            return [None] * len(qs)
        return [w[max(0, min(len(w) - 1, math.ceil(q * len(w)) - 1))] for q in qs]

    def copy(self):
        """a frozen copy (the readers' snapshot): the ring copied, the counts too"""
        c = Series.__new__(Series)
        c.name, c.unit, c.edges = self.name, self.unit, self.edges
        c.ring, c.i, c.n = list(self.ring), self.i, self.n
        c.counts, c.sum, c.count, c.max = list(self.counts), self.sum, self.count, self.max
        return c

    def snapshot(self, hist=False):
        w = self.window()
        p50, p95 = self.quantiles()
        out = {'p50': _r(p50), 'p95': _r(p95), 'mean': _r(sum(w) / len(w)) if w else None, 'window': len(w),
               'max': _r(self.max) if self.count else None, 'count': self.count, 'sum': _r(self.sum), 'unit': self.unit}
        if hist:
            out['hist'] = {'le': list(self.edges) + ['+Inf'], 'counts': list(self.counts)}
        return out


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def _frozen(v):
    """snapshot(): a value copied so the engine thread's later writes do not reach the reader"""
    if isinstance(v, Series):
        return v.copy()
    if isinstance(v, SlotState):
        c = SlotState()
        for k in SlotState.__slots__:
            setattr(c, k, getattr(v, k))
        return c
    if isinstance(v, dict):
        return {k: _frozen(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_frozen(x) for x in v]
    return v


class SlotState:
    """llama-server's /slots entry for the one slot, written by the engine thread, read by HTTP threads. The fields of
    the last request stay after it ends (llama-server's task_prev), so a client sees its final numbers."""
    __slots__ = ('id_task', 'is_processing', 'n_prompt', 'n_cache', 'n_processed', 'n_decoded', 'n_remain',
                 'has_next_token', 'has_new_line', 'params', 'started')

    def __init__(self):
        self.id_task = None
        self.is_processing = False
        self.n_prompt = self.n_cache = self.n_processed = self.n_decoded = 0
        self.n_remain = -1
        self.has_next_token = False
        self.has_new_line = False
        self.params = None
        self.started = False


class EngineStats:
    def __init__(self, clock=time.perf_counter, physics=None, window=1024, enabled=True):
        self.clock = clock
        self.enabled = enabled
        self.physics = physics or PhysicsConfig()
        self.window = window
        self.t0 = clock()
        self.t_start_unix = int(time.time())
        self.lock = threading.Lock()           # the HTTP threads' writers
        self.n_cards = len(self.physics.card_bytes)
        self.reset()

    def reset(self):
        W = self.window
        nc = self.n_cards
        # ---- llama-server's counters (llamacpp: names)
        self.prompt_tokens = 0          # processed, cached excluded
        self.prompt_us = 0
        self.prompt_cached = 0
        self.predicted_tokens = 0
        self.predicted_steps = 0
        self.predicted_us = 0
        self.n_decode = 0               # engine passes: prompt passes + decode passes
        self.n_busy = 0
        self.n_tokens_max = 0
        self.spec_drafts = 0            # verification steps with >= 1 draft
        self.spec_draft_tokens = 0
        self.spec_accepted = 0
        self.accepted_per_pos = []
        self.bucket = [0, 0, 0, 0]      # since the last scrape: prompt tokens, prompt us, predicted steps, predicted us
        self.processing = 0
        self.slot = SlotState()
        self.phase = 'idle'
        # ---- decode passes
        self.pass_series = {p: Series('decode_pass_' + p, 'ms', W) for p in PASS_PARTS}
        self.emitted = Series('tokens_per_pass', 'tokens', W, COUNT_EDGES)
        self.pass_k3_ms = 0.0           # the pass in flight, from the session's pass_split() (pass_done reads them)
        self.pass_sampler_ms = 0.0
        self.pass_cards = None          # scratch: (card ms tuple, exchange-wait ms tuple or None, link bytes) or None
        self.plain_ms = None            # EMA of plain (undrafted) passes' ms per emitted token (measured)
        self.plain_est = None           # the session's estimate of it (plain_estimate) while none is measured
        self.card_pass = [Series(f'card{c}_pass', 'ms', W) for c in range(nc)]
        self.card_wait = [Series(f'card{c}_exchange_wait', 'ms', W) for c in range(nc)]
        self.target_pass = Series('target_pass', 'ms', W)    # the target's own pass span (the slower card's), from pass_cards
        self.exchange_wait_note = getattr(self, 'exchange_wait_note', None)   # the engine's reason when it has no wait times
        self.link_bytes = 0
        self.link_ring = Series('link_bytes_per_pass', 'bytes', W, (1e4, 1e5, 1e6, 1e7, 1e8))
        # ---- DFlash per content class
        self.cls = 'prose'              # the class at the pass start (agentmon.ContentClass, set by the worker)
        self.cls_drafts = dict.fromkeys(CLASSES, 0)
        self.cls_draft_tokens = dict.fromkeys(CLASSES, 0)
        self.cls_accepted = dict.fromkeys(CLASSES, 0)
        self.cls_pos = {c: [] for c in CLASSES}
        # ---- requests / TTFT / prefill
        self.ttft = {p: Series('ttft_' + p, 'ms', W) for p in TTFT_PARTS}
        self.prefill_rate = Series('prefill_tok_s', 'tok/s', W, RATE_EDGES)
        self.prefill_rate_ema = None    # batched-or-not prompt tok/s (for the seconds-saved estimate)
        self.req = None                 # the request in flight: {t_begin, queue_ms, parts, first}
        self.finish = dict.fromkeys(FINISHES, 0)
        self.requests = 0
        self.tokens_reasoning = 0
        self.tokens_answer = 0
        # ---- prompt cache
        self.cache_tokens = {'vram': 0, 'disk': 0, 'exact': 0}
        self.cache_sources = dict.fromkeys(SOURCES, 0)
        self.cache_restore = Series('cache_restore', 'ms', W)
        self.cache_lcp = Series('cache_lcp', 'tokens', W, TOKEN_EDGES)
        self.cache_counters = dict.fromkeys(CACHE_EVENTS, 0)
        self.cache_gauges = {'n_slots': None, 'slots_used': None, 'tier_mib': None, 'headroom_mib': None,
                             'disk_bytes': None, 'disk_cap_bytes': None, 'side_active': None, 'side_rows': None}
        self.cache_route = {r: dict.fromkeys(rs, 0) for r, rs in ROUTE_REASONS.items()}
        self.cache_route_last = None    # the last request's decision: {route, reason, shared, root, need, main_n, side_rows}
        self.saved_s = 0.0              # prefill seconds saved (estimate: cached tokens / prefill rate)
        self.saved_tokens = 0           # cached tokens the estimate covers
        # ---- context
        self.position = 0
        self.max_ctx = 0
        # ---- reliability
        self.bails = {'recovered': 0, 'failed': 0}
        self.retries = 0
        self.evicted_ms = 0
        self.pgmigrate = 0              # page migrations in samples while the engine was busy
        self.pgmigrate_all = 0          # every page migration since the first sample
        self.host_samples = 0           # host_sample calls (0: no sampler feeds the host counters)
        self.refusals = dict.fromkeys(REFUSALS, 0)
        self.engine_errors = 0
        self.internal_errors = 0
        # ---- agent side (HTTP threads, under lock)
        self.harness = {}
        self.tool_calls = 0
        self.tool_failures = {'cut': 0, 'unclosed': 0, 'undeclared': 0, 'malformed': 0}
        self.tool_lenient = 0
        self.loop_flagged = 0
        self.loop_repeated = 0
        self.loop_live = 0              # the current request's highest repeat count of one reasoning line
        self.max_tokens_hits = 0

    # ============================================================================ engine thread
    # (every writer holds self.lock for its adds: a reader's snapshot() copies under it; uncontended ~0.1 us)
    def begin(self, job, max_ctx=None):
        if not self.enabled:
            return
        now = self.clock()
        with self.lock:
            s = self.slot
            s.id_task = job.id_task
            s.is_processing = True
            s.n_prompt = len(job.prompt_ids)
            s.n_cache = s.n_processed = s.n_decoded = 0
            s.n_remain = job.n_predict
            s.has_next_token = True
            s.has_new_line = False
            s.params = job.slot_params
            s.started = True
            self.processing = 1
            self.phase = 'prefill'
            self.cls = 'prose'
            if max_ctx:
                self.max_ctx = max_ctx
            self.req = {'t_begin': now, 'queue_ms': job.queue_ms(), 'first': None, 'parts': None}

    def prefill_progress(self, processed):
        """prompt tokens processed so far by this request (llama-server's n_prompt_tokens_processed)"""
        if self.enabled:
            self.slot.n_processed = processed

    def cached_known(self, cached):
        """the reused prefix is known (before processing starts)"""
        if self.enabled:
            self.slot.n_cache = cached

    def prefill_done(self, pf):
        """pf: engine.PrefillStats of this request's prompt"""
        if not self.enabled:
            return
        now = self.clock()
        with self.lock:
            self.n_decode += pf.passes
            self.n_busy += pf.passes
            s = self.slot
            s.n_cache, s.n_processed = pf.cached_n, pf.processed_n
            us = int(round(pf.ms * 1e3))
            self.prompt_tokens += pf.processed_n
            self.prompt_us += us
            self.prompt_cached += pf.cached_n
            b = self.bucket
            b[0] += pf.processed_n
            b[1] += us
            if pf.processed_n and pf.ms > 0:
                rate = pf.processed_n / (pf.ms / 1e3)
                self.prefill_rate.add(rate)
                e = self.prefill_rate_ema
                self.prefill_rate_ema = rate if e is None else 0.8 * e + 0.2 * rate
            src = pf.source if pf.source in self.cache_sources else 'cold'
            self.cache_sources[src] += 1
            if pf.cached_n:
                tier = 'exact' if pf.exact else TIER_OF.get(src)
                if tier:
                    self.cache_tokens[tier] += pf.cached_n
                if src != 'live':
                    self.cache_restore.add(pf.restore_ms)
                if self.prefill_rate_ema:
                    self.saved_s += pf.cached_n / self.prefill_rate_ema
                    self.saved_tokens += pf.cached_n
            self.cache_lcp.add(pf.lcp)
            if self.req is not None:
                self.req['parts'] = (pf.restore_ms, pf.batched_ms, pf.handoff_ms, pf.passes_ms)
                self.req['t_decode'] = now
            self.position = pf.prompt_n

    def first_token(self, n=1):
        """the first token(s) of the reply were emitted (sampled from the prompt's last pass): ends TTFT"""
        if not self.enabled or self.req is None or self.req['first'] is not None:
            return
        now = self.clock()
        with self.lock:
            r = self.req
            r['first'] = now
            s = self.slot
            s.n_decoded += n
            if s.n_remain > 0:
                s.n_remain = max(0, s.n_remain - n)
            total = (now - r['t_begin']) * 1e3
            parts = r['parts'] or (0.0, 0.0, 0.0, 0.0)
            t = self.ttft
            t['queue'].add(r['queue_ms'])
            t['restore'].add(parts[0])
            t['batched'].add(parts[1])
            t['handoff'].add(parts[2])
            t['passes'].add(parts[3])
            t['host'].add(max(0.0, total - sum(parts)))
            t['total'].add(total + r['queue_ms'])
            self.phase = 'decode'

    def pass_split(self, k3_ms, sampler_ms, cards=None):
        """the session's report of the pass in flight (EngineSession.sample_pass, before it returns): k3_ms the engine's
        pass, sampler_ms the sampling, cards (per-card pass ms tuple, exchange-wait ms tuple or None, link bytes) or None.
        pass_done reads and clears it (the engine thread only: no lock)"""
        self.pass_k3_ms, self.pass_sampler_ms, self.pass_cards = float(k3_ms), float(sampler_ms), cards

    def pass_done(self, t0, draft_ms, fed, drafted, accepted, emitted, cls=None):
        """one decode pass: t0 = clock() before its drafter call; accepted = drafts accepted (a prefix of the drafts:
        positions 0 .. accepted - 1); drafted = the pass's real drafts (0 for a plain pass: filler drafts are not drafts,
        run_generation asks the session); cls = the content class when the pass started (default: the current one). The
        K3 / sampler split and the per-card numbers come from the session's pass_split() for the pass"""
        if not self.enabled:
            return
        total = (self.clock() - t0) * 1e3
        k3, smp = self.pass_k3_ms, self.pass_sampler_ms
        pc = self.pass_cards
        self.pass_cards = None
        with self.lock:
            ps = self.pass_series
            ps['k3'].add(k3)
            ps['drafter'].add(draft_ms)
            ps['sampler'].add(smp)
            ps['host'].add(max(0.0, total - k3 - draft_ms - smp))
            ps['total'].add(total)
            self.emitted.add(emitted)
            self.n_decode += 1
            self.n_busy += 1
            self.predicted_steps += 1
            if self.req is not None:
                self.req['steps'] = self.req.get('steps', 0) + 1
            if drafted:
                self.spec_drafts += 1
                self.spec_draft_tokens += drafted
                self.spec_accepted += accepted
                c = cls if cls in self.cls_drafts else self.cls
                self.cls_drafts[c] += 1
                self.cls_draft_tokens[c] += drafted
                self.cls_accepted[c] += accepted
                app, cpp = self.accepted_per_pos, self.cls_pos[c]
                while len(app) < drafted:
                    app.append(0)
                while len(cpp) < drafted:
                    cpp.append(0)
                for i in range(accepted):
                    app[i] += 1
                    cpp[i] += 1
            elif fed == 1:
                e = self.plain_ms
                v = total / max(1, emitted)
                self.plain_ms = v if e is None else 0.9 * e + 0.1 * v
            if pc is not None:
                cms, waits, nbytes = pc
                if cms is not None:
                    for c in range(min(len(cms), self.n_cards)):
                        self.card_pass[c].add(cms[c])
                    if cms:
                        self.target_pass.add(max(cms))
                if waits is not None:
                    for c in range(min(len(waits), self.n_cards)):
                        self.card_wait[c].add(waits[c])
                if nbytes:
                    self.link_bytes += nbytes
                    self.link_ring.add(nbytes)
            s = self.slot
            s.n_decoded += emitted
            if s.n_remain > 0:
                s.n_remain = max(0, s.n_remain - emitted)
            self.position += emitted

    def other_pass(self, n=1):
        """passes that are neither prompt nor decode records (forced tool-call tokens beyond one pass)"""
        if self.enabled:
            with self.lock:
                self.n_decode += n
                self.n_busy += n

    def forced(self, n):
        """n forced tool-call ids (tool_choice required / named) emitted as if generated: they count in /slots' n_decoded and
        n_remain and in the context position like sampled ids (M7: the metrics verifier's defect, they used to reach gen_ids
        only); they are not a decode pass, so no pass record"""
        if not self.enabled or n <= 0:
            return
        with self.lock:
            s = self.slot
            s.n_decoded += n
            if s.n_remain > 0:
                s.n_remain = max(0, s.n_remain - n)
            self.position += n

    def tokens(self, n_reasoning, n_answer):
        if self.enabled:
            with self.lock:
                self.tokens_reasoning += n_reasoning
                self.tokens_answer += n_answer

    def end(self, job, res):
        if not self.enabled:
            return
        now = self.clock()
        with self.lock:
            n = len(res.gen_ids)
            r = self.req
            if r is not None and r.get('t_decode') is not None:    # the prompt processed -> the end (llama.cpp's)
                us = int(round((now - r['t_decode']) * 1e6))
            else:
                us = int(round(res.decode_ms * 1e3))
            self.predicted_tokens += n
            self.predicted_us += us
            self.bucket[2] += r.get('steps', 0) if r is not None else 0     # llama-server adds both at the request end
            self.bucket[3] += us
            self.n_tokens_max = max(self.n_tokens_max, res.prompt_n + n)
            self.finish[res.finish if res.finish in self.finish else 'eos'] += 1
            if res.finish == 'length':
                self.max_tokens_hits += 1
            self.requests += 1
            self._idle()

    def error(self, job, kind):
        if not self.enabled:
            return
        with self.lock:
            if kind == 'engine':
                self.engine_errors += 1
            else:
                self.internal_errors += 1
            self.finish['error'] += 1
            self.requests += 1
            self._idle()

    def _idle(self):
        s = self.slot
        s.is_processing = False
        s.has_next_token = False
        self.processing = 0
        self.phase = 'idle'
        self.req = None

    # ---- engine-only hooks (M7)
    def bail(self, recovered):
        if self.enabled:
            with self.lock:
                self.bails['recovered' if recovered else 'failed'] += 1

    def retry(self, n=1):
        if self.enabled:
            with self.lock:
                self.retries += n

    def refusal(self, kind, n=1):
        if self.enabled and n > 0:
            with self.lock:
                k = kind if kind in self.refusals else 'bounds'
                self.refusals[k] += n

    def evicted(self, ms):
        if self.enabled:
            with self.lock:
                self.evicted_ms += ms

    def pgmigrate_add(self, n):
        if self.enabled:
            with self.lock:
                self.pgmigrate += n

    def host_sample(self, evicted_ms=None, pgmigrate=None, busy=False):
        """the hardware sampler thread's host counters since its previous sample (None: unreadable): this process's KFD
        queue eviction ms (the largest per-node rise) and the system's page migrations (counted in pgmigrate when the
        engine was busy at either end of the interval, in pgmigrate_all always)"""
        if not self.enabled:
            return
        with self.lock:
            self.host_samples += 1
            if evicted_ms:
                self.evicted_ms += evicted_ms
            if pgmigrate:
                self.pgmigrate_all += pgmigrate
                if busy:
                    self.pgmigrate += pgmigrate

    def plain_estimate(self, ms_per_token):
        """the session's estimate of a plain pass's ms per token (used while no plain pass was measured)"""
        if self.enabled and ms_per_token is not None and ms_per_token > 0:
            self.plain_est = float(ms_per_token)

    def route(self, decision):
        """the side-slot routing decision of the request in flight (PromptCache.route's dict); an unknown reason counts as
        'off' on main"""
        if not self.enabled or not decision:
            return
        r = decision.get('route') if decision.get('route') in ROUTE_REASONS else 'main'
        why = decision.get('reason') if decision.get('reason') in ROUTE_REASONS[r] else ROUTE_REASONS[r][-1]
        with self.lock:
            self.cache_route[r][why] += 1
            self.cache_route_last = {k: decision.get(k) for k in ('route', 'reason', 'shared', 'root', 'need', 'main_n', 'side_rows')}

    def cache_update(self, counters=None, gauges=None):
        """counters: the prompt cache's own Counter (PromptCache.stats) - its CACHE_EVENTS keys are copied as totals;
        gauges: n_slots, slots_used, tier_mib, headroom_mib, disk_bytes, disk_cap_bytes, side_active (a side call holds the
        engine), side_rows (None = unknown)"""
        if not self.enabled:
            return
        with self.lock:
            if counters:
                cc = self.cache_counters
                for k in CACHE_EVENTS:
                    if k in counters:
                        cc[k] = counters[k]
            if gauges:
                g = self.cache_gauges
                for k in g:
                    if k in gauges:
                        g[k] = gauges[k]

    # ============================================================================ the readers' snapshot
    def snapshot(self, take_bucket=False):
        """a frozen copy for one scrape, taken under the lock (the rings and dicts copied, ~0.1 ms); the renderers
        (metrics.prometheus / status / slot_json) run on it after the lock is released. take_bucket: llama-server's
        prompt / predicted rate bucket is taken (and reset) atomically with the copy (only /metrics takes it)"""
        snap = EngineStats.__new__(EngineStats)
        with self.lock:
            d = dict(self.__dict__)
            out = {}
            for k, v in d.items():
                out[k] = _frozen(v)
            if take_bucket:
                rates = self.take_bucket()
            else:
                rates = None
        snap.__dict__.update(out)
        snap.lock = threading.Lock()            # (the snapshot's own: the renderers' 'with st.lock' reads stay valid)
        snap._rates = rates
        snap._frozen = True
        return snap

    # ============================================================================ HTTP threads
    def http_request(self, harness):
        with self.lock:
            self.harness[harness] = self.harness.get(harness, 0) + 1

    def tool_report(self, n_ok, failures, lenient):
        with self.lock:
            self.tool_calls += n_ok
            for k, v in failures.items():
                self.tool_failures[k] = self.tool_failures.get(k, 0) + v
            self.tool_lenient += lenient

    def loop_live_set(self, repeats):
        self.loop_live = repeats

    def loop_report(self, flagged, repeated):
        with self.lock:
            self.loop_flagged += bool(flagged)
            self.loop_repeated += repeated
            self.loop_live = 0

    # ============================================================================ readers
    def take_bucket(self):
        """llama-server's prompt / predicted gauges: the rates over the window since the last scrape, then reset (on a
        snapshot: the rates its snapshot(take_bucket=True) took, else zeros)"""
        if getattr(self, '_frozen', False):
            return self._rates if self._rates is not None else (0.0, 0.0)
        b = self.bucket
        self.bucket = [0, 0, 0, 0]
        prompt = b[0] / b[1] * 1e6 if b[1] > 0 else 0.0
        pred = b[2] / b[3] * 1e6 if b[3] > 0 else 0.0
        return prompt, pred

    def ms_per_token(self):
        """decode ms per emitted token over the window (None before any pass)"""
        tot = self.pass_series['total'].window()
        em = self.emitted.window()
        n = min(len(tot), len(em))
        tok = sum(em[len(em) - n:])
        return sum(tot[len(tot) - n:]) / tok if tok else None

    def physics_view(self):
        """pass_pct: the floor / the median target pass. The target's own span is the per-card pass time (pass_cards: the
        slower card's, device events on the real engine) when the engine reports it; else pass_k3_ms, which on the
        device-accept path also holds the drafter's step queued behind the verify (pass_source says which)"""
        floor = self.physics.floor_ms()
        k3_p50 = self.pass_series['k3'].quantiles((0.5,))[0]
        tgt = self.target_pass.quantiles((0.5,))[0] if self.target_pass.count else None
        den = tgt if tgt else k3_p50
        mpt = self.ms_per_token()
        return {'floor_ms': _r(floor), 'mode': self.physics.mode, 'source': self.physics.source,
                'pass_pct': _r(floor / den * 100, 2) if den else None,
                'pass_source': 'card_events' if tgt else ('k3_host' if k3_p50 else None),
                'token_pct': _r(floor / mpt * 100, 2) if mpt else None,
                'k3_pass_p50_ms': _r(k3_p50), 'target_pass_p50_ms': _r(tgt), 'ms_per_token': _r(mpt)}

    def spec_view(self):
        def per(acc, tok):
            return _r(acc / tok, 4) if tok else None
        classes = {}
        for c in CLASSES:
            d = self.cls_drafts[c]
            pos = list(self.cls_pos[c])
            classes[c] = {'drafts': d, 'draft_tokens': self.cls_draft_tokens[c], 'accepted': self.cls_accepted[c],
                          'acceptance': per(self.cls_accepted[c], self.cls_draft_tokens[c]),
                          'per_pos': pos, 'per_pos_rate': [_r(x / d, 4) for x in pos] if d else []}
        em = self.emitted.window()
        tpp = sum(em) / len(em) if em else None
        if self.plain_ms is not None:
            plain, src = self.plain_ms, 'measured'
        elif self.plain_est is not None:
            plain, src = self.plain_est, 'estimate'
        elif self.physics.plain_ms_per_token is not None:
            plain, src = self.physics.plain_ms_per_token, 'configured'
        else:
            plain, src = None, None
        mpt = self.ms_per_token()
        d = self.spec_drafts
        app = list(self.accepted_per_pos)
        return {'drafts': d, 'draft_tokens': self.spec_draft_tokens, 'accepted': self.spec_accepted,
                'acceptance': per(self.spec_accepted, self.spec_draft_tokens), 'per_pos': app,
                'per_pos_rate': [_r(x / d, 4) for x in app] if d else [], 'classes': classes,
                'tokens_per_pass': _r(tpp), 'plain_ms_per_token': _r(plain), 'plain_measured': self.plain_ms is not None,
                'plain_source': src, 'speedup_vs_plain': _r(plain / mpt, 3) if plain and mpt else None}
