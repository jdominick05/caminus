"""The real EngineSession (M7): the server's front end on the K3 engine through the prompt cache.

    K3Session(m, pf, dr, pc)   m: FlashNextTP (T = 4, max_ctx 262,144 by default: the dual-tape QSA engine), pf: its
                               Prefill, dr: DFlashK3Drafter(df) (device accept) or None, pc: PromptCache(m, pf, dr)
                               (build it last: the VRAM tier is sized from what the others left)

The mapping (docs/specs/spec-cache-api.md, "M7 dependencies"):
  plan(ids, max_new)     -> pc.plan(ids, side='auto', max_new=max_new)['cached']
  prefill(ids, cancel, marks, max_new)
                         -> pc.prepare(ids, marks, side='auto', cancel=cancel, max_new=max_new) (side-slot routing, then restore
                            the best state, the batched prefill / K3 passes with the policy's checkpoints), then
                            longctx.first_logits (the prompt's last pass, its features injected into the drafter), then
                            pc.extend. A cancel stops at the next chunk boundary of the batched prefill (or K3 pass) and leaves
                            the processed prefix as the live state (commit 0): None is returned and the next request continues it.
                            Side-slot routing (impl:m7-finish, the lead's ruling of 11-05; PromptCache.route): a request that does
                            not match the main line (shares less than its root mark, or than promptcache.SIDE_MATCH_MIN tokens with
                            no root known), fits the side slot with its max_new (prompt + max_new + T <= side_rows: the server
                            gives the session max_new, wants_max_new) and is shorter than the main line runs in the side slot, so
                            harness title / summary / next-speaker calls never displace the conversation; the next main request is
                            served as if it had not happened. side_route=False sends every request to main. The decision is in
                            last_route, the stats (route(): caminus:cache_route_total, /caminus/status cache.route) and
                            last_prefill.slot.
  sample_pass(feed, drafts, sampler, history, stop_ids)
                         -> one T-token pass at i0 = len(log) (the feed, the drafts, filler to T), three forms:
                            * device accept (greedy, penalties off, the drafter in step, the policy says draft): the target's
                              verify and the drafter's 'step' (accept + inject + the next drafts) in one host round trip:
                              DFlashK3Drafter.device_step, _loop's own pass. Emits d[:j] + [bonus] (cut at a stop id).
                            * greedy without the step (no drafter, plain mode, near max_pos / a window gap): verify_tokens
                              (each card's argmax, no logits copied) and the host match.
                            * T > 0 or penalties: the verify, then the device sampler (each card's top-k' and the penalty ids'
                              raw logits, caminus.serve.candsample: bitwise the host reference on the full rows), sample and
                              match with Philox keyed on (seed, position); the drafts come from the generic DFlash path
                              (DFlashK3Drafter.__call__: the pass's inject fused into the next draft).
  accept(n)              -> the log grows by n fed tokens; the next pass commits exactly n rows (lazy commit)
  verify(feed)           -> a pass whose logits nobody reads (forced tool-call tokens beyond one pass): verify_tokens,
                            its features injected (the drafter stays in step)
  end()                  -> pc.commit(log, state): the completion's accepted tokens reported, the live state kept
  abort()                -> (a host-side exception between calls, or inside a call) whatever is in flight drained, then ONE
                            report to the cache (pc.settle(returned, drained)), which decides what the device holds (it owns the
                            live state and ran the prompt phase and the recoveries, so it knows when the state is not known):
                            a pass that ran or was launched and was not accepted is dropped (the next pass at its position
                            commits 0), a refusal before the launch keeps the commit, an unknown state or a failed drain drops
                            the live state (every checkpoint kept), an engine reset since the last completed operation drops
                            it too; reset only when the engine is poisoned
  reset()                -> pc.reset()
  idle()                 -> pc.park(background=True) (the worker calls it when no request came for idle_s seconds: 8 hours by
                            default, the user's ruling of 11-05), only when the live log gained >= pc.auto_park_min tokens
                            since this session's last park of the line (a short or unchanged conversation is not written); an
                            active side call is ended first (pc.end_side: the main line back; park() is refused during one)
  shutdown_park()        -> on a clean shutdown (caminus serve: SIGTERM / SIGINT / a systemd stop), after the engine worker
                            stopped: pc.park() of the live conversation when it changed since its last park, so a restart
                            within the idle time loses nothing (the caller bounds its time); an active side call is ended first

Bail recovery: when a pass fails after the engine's bitwise retries (FlashNextTP's, or the drafter step's), the live state is
undefined. The session drains anything in flight; the cache (pc.recover_pass; in the prompt phase pc.prompt itself) restores
the request's prompt-end checkpoint (every checkpoint kept; in a side call, which takes no checkpoints, it ends the side call
with the main line intact and processes the side request's prompt again to the same state), replays the request's processed tokens teacher-forced with K3
passes (bitwise the passes it lost: a T-token pass equals one-token steps) and runs the failed pass again: the same logits, so
the same tokens (greedy, or the same Philox draws). The client sees a delay, not an error. A failure while the prompt is
processed drops the live state (checkpoints kept; in a side call only the side call's state: the cache ends it with the
main line intact) and processes the prompt again (pc.prompt). Counted in stats (bail(True), pc.stats['recovered']); after
`max_recoveries` failures in one request, or without the prompt-end checkpoint, EngineError (the worker resets: 503 / 529).
Test hook: K3_TEST_HOOKS=police-controls with K3_TEST_BAIL=<n> makes this session's n-th target pass (0-based) bail on
every attempt (a 1 us wait bound on every tape: the designed bail path, tools/gates/forcebail.py's 'tiny' mode), so the
engine's retries are exhausted and the recovery runs; never a real invalid launch.

Draft only when it pays: caminus.serve.draftpolicy.DraftPolicy (per content class and depth; plain = the T = 4 filler pass
with the drafter's inject only, so it stays in step). The session's `drafter` is the callable the server's generation loop
asks for drafts (run_generation's drafter(tokens, k)).

Metrics (docs/specs/spec-metrics.md; M7 ops): cached_known before processing, prefill_progress per chunk of the batched
prefill and per K3 prompt pass (the cancel object passed to prepare() carries a progress(position) the prefill and the cache
call); the pass split pass_k3_ms (the verify, or verify + step) / pass_sampler_ms (the device sampler's wait + host
sampling); pass_cards per decode pass: each card's span of the target's pass (FlashNextTP.enable_card_timing: device events
read after the pass's own wait, no added sync) and the exchange's payload bytes (2 cards x 2 exchanges per layer x T x D x 4);
the exchange wait inside the span is not measured (stats.exchange_wait_note says why); drafts_real tells run_generation which
passes drafted (filler drafts of a plain pass or of a window gap are not drafts: the DFlash counters and the plain pass time);
plain_estimate from the policy while no plain pass was measured; bails and retries from FlashNextTP.bails; refusals by kind
(refusal_kind); cache_update with the prompt cache's own counters after each request (its 'refused' rises as refusals
{cache}); the physics floor's per-card bytes and the KV bytes per token from the engine (engine_physics). Host timestamps
only, no device sync is added (the passes sync anyway)."""
import os
import time

import numpy as np
import torch

from .. import arch as _arch
from ..device import capacity, topk
from ..decode import longctx
from ..engine import passpolice
from ..engine.k3 import ContextBoundsError
from ..prefill.driver import PrefillRefused   # (its home: the Coder's prefill module re-exports it but loads the vendored MMQ and gguf-py)
from ..prefill.glue import RoutingError
from . import candsample as CS
from .draftpolicy import DraftPolicy
from .engine import EngineError, EngineSession, PrefillStats, common_prefix
from .mock_engine import EngineStateError
from ..cache.livestate import is_bail
from ..cache.prompt import RecoveryFailed

RESTORE_OF = {'live': 'live', 'vram': 'checkpoint', 'spill': 'checkpoint', 'disk': 'disk', 'cold': 'cold'}
IDLE_PARK_S = 28800.0       # 8 hours (decision 11-05: an idle park after hours, not minutes, as servers usually do)
EXCHANGE_WAIT_NOTE = ('not measured on this engine: the TP exchange waits inside the persistent kernel (OP_XCHG spins on the '
                      "peer's flag), so the wait is part of each card's pass span; separating it needs device timestamps inside "
                      'the kernel (the *_trace builds, benchmarks/k3_trace.py), which the serving code objects do not have')


def refusal_kind(e):
    """the metrics' refusal kind of a host refusal (refusals_total{kind}), or None for anything else"""
    from ..cache.prompt import CacheError
    if isinstance(e, RoutingError): return 'routing_flag'
    if isinstance(e, PrefillRefused): return 'prefill'
    if isinstance(e, ContextBoundsError): return 'bounds'
    if isinstance(e, capacity.CapacityError): return 'capacity'
    if isinstance(e, CacheError): return 'cache'
    return None


def engine_pci(m):
    """the engine devices' PCI slots ('0000:03:00.0', ...) in engine order (hw.HwReader(cards=)), from torch's device
    properties; None when they cannot be read (the reader then picks the cards by VRAM size)"""
    import torch as _t
    out = []
    try:
        for d in m.devs:
            p = _t.cuda.get_device_properties(int(d))
            out.append(f'{int(p.pci_domain_id):04x}:{int(p.pci_bus_id):02x}:{int(p.pci_device_id):02x}.0')
    except Exception:                                   # noqa: BLE001
        return None
    return out


def engine_physics(m, pc=None):
    """(per-card bytes one decode token streams, per-card KV + indexer bytes per position) of the engine: every weight the
    tapes read on each card (FlashNextTP.shards: the split tensors' shards; FlashNextTP.weights: the replicated ones, a split
    tensor counted once), the routed experts at k_used / n_exp (a token reads k_used of each layer's experts); the KV bytes
    from the prompt cache's layout (KV rows of every attention layer + IDXRAW + its pools). (None, None) parts when they
    cannot be read (the defaults stay)"""
    import torch as _t
    card = None
    if hasattr(m, 'physics_bytes'):                    # (a one-card engine counts its own bytes: Qwen35One.physics_bytes)
        try: card = tuple(float(x) for x in m.physics_bytes())
        except Exception:                               # noqa: BLE001
            card = None
    try:
        if card is not None: raise StopIteration
        frac = float(m.k_used) / float(m.n_exp)
        per = [0.0, 0.0]
        def nb(x):
            if isinstance(x, dict): x = x.get('w')
            return x.numel() * x.element_size() if isinstance(x, _t.Tensor) else 0
        for name, sh in m.shards.items():
            f = frac if sh['kind'].startswith('exp') else 1.0
            for c in (0, 1): per[c] += nb(sh['infos'][c]) * f
        seen = set(m.shards)
        for (loader, name), v in m.weights.items():
            if name in seen or not isinstance(v, (list, tuple)) or len(v) != 2: continue
            seen.add(name)
            for c in (0, 1): per[c] += nb(v[c])
        card = tuple(per) if min(per) > 0 else None
    except StopIteration:
        pass
    except Exception:                                   # noqa: BLE001 - the spec defaults stay
        card = None
    kv = None
    try:
        L = pc.layout
        kv = tuple(len(L.kv[c]) * L.pb[c] + (L.na * L.rb * 1.25 if L.idx else 0.0) for c in range(len(L.devs)))
    except Exception:                                   # noqa: BLE001
        kv = None
    return card, kv


class NullPolice:
    """longctx.teacher_forced (first_logits) polices its passes through passpolice.of(m), which attaches a recorder of every
    launch attempt to the engine: right for harness runs, a leak in a long-lived server. The server's engine gets this
    instead (no records, no test hooks: the session's own K3_TEST_BAIL hook is the server's)"""
    def begin(self, where, i0):
        return None

    def end(self, p, max_pass_ms=None):
        return dict(raw=0.0, net=0.0, ev=0)

    def mode(self):
        return 'off (server)'

    def unexplained_bails(self):
        return []


_bail = is_bail     # (a pass that failed after the engine's bitwise retries: caminus.cache.livestate)


class _Hooks:
    """what the prompt cache calls back while it runs a request phase for the session (PromptCache.prompt / recover_pass):
    arm(on) before / after each pass attempt (the test bail hook), event(kind, side) after a recovery (counters, stats)"""
    __slots__ = ('s',)

    def __init__(self, sess):
        self.s = sess

    def arm(self, on):
        if on:
            self.s._arm()
        else:
            self.s._armed = False

    def event(self, kind, side=False):
        self.s._event(kind, side)


class _Progress:
    """the cancel object K3Session hands to prepare(): is_set() is the client's cancel; progress(position), called by the
    batched prefill after each chunk and by the cache after each K3 prompt pass, feeds /slots' live prompt progress"""
    __slots__ = ('cancel', 'stats', 'start')

    def __init__(self, cancel, stats, start):
        self.cancel, self.stats, self.start = cancel, stats, int(start)

    def is_set(self):
        c = self.cancel
        if c is None:
            return False
        return bool(c.is_set() if hasattr(c, 'is_set') else c())

    def progress(self, pos):
        st = self.stats
        if st is not None:
            st.prefill_progress(max(0, int(pos) - self.start))


class SessionDrafter:
    """the server generation loop's drafter(tokens, k) for a K3Session: the drafts of the pass about to run. After a device
    step they are the step's own (drafted at the accepted position, anchored on the bonus token); else the generic DFlash path
    (DFlashK3Drafter.__call__: a 'fused' / 'draft' launch, filler where it cannot draft); in plain mode filler"""
    spec_type = 'draft-dflash'

    def __init__(self, sess):
        self.s = sess

    def __call__(self, toks, k):
        return self.s._drafts(toks, k)


class K3Session(EngineSession):
    _dk = None      # the drafter kind ('dflash' | 'mtp' | None); the test rigs build the session with __new__
    _ver = None     # the verify tape of an MTP session (m.ver)

    def __init__(self, m, pf, dr, pc, stats=None, eos_ids=(248046, 248044), policy=None, max_recoveries=2, idle_s=IDLE_PARK_S,
                 marks=True, side_route=True):
        self.m, self.pf, self.dr, self.pc = m, pf, dr, pc
        self.df = getattr(dr, 'df', None)
        if pc.draft is not dr: raise ValueError('K3Session: the prompt cache must own the same drafter')
        self._dk = getattr(dr, 'kind', None) if dr is not None else None
        if dr is not None and not getattr(dr, 'device_accept', False) and self._dk != 'mtp':
            raise ValueError('K3Session: the drafter must be DFlashK3Drafter(df) with device_accept (the default), '
                             f'or an MTP drafter (kind mtp); got {type(dr).__name__}')
        self._ver = None
        if self._dk == 'mtp':
            # an MTP session's draft passes are the T = 4 verify passes on the decode engine's states (bitwise its
            # decode steps, engine/q35_dense_verify.py); plain passes run the engine's own T = 1 tape
            self._ver = getattr(m, 'ver', None)
            if self._ver is None or int(getattr(m, 'spec_T', 1)) < 4:
                raise ValueError('K3Session: kind mtp needs the engine built with spec_T 4 and its verify tape (m.ver)')
        self.request_params = None   # the worker sets it per request: sampled requests on an MTP session run plain
        if getattr(m, '_pass_police', None) is None:
            m._pass_police = NullPolice()
        self.T = self.ntok = 4 if self._dk == 'mtp' else int(m.ntok)
        # a generation of n_new tokens after n prompt tokens needs n + n_new + T - 2 positions (spec.positions_needed): the server
        # keeps prompt + max_new <= max_ctx, so the session's limit is the engine's less T - 2 (never above the engine's rows: a
        # T = 1 engine, the one-card Qwen35One, keeps its max_ctx; arch.usable_ctx, which the loading server reports too)
        self.max_ctx = _arch.usable_ctx(m.max_ctx, self.T)
        self.n_vocab = int(m.T['output.weight'].shape[1])
        self.vh = int(m.vh)
        self.eos_ids = tuple(eos_ids)
        self.stats = stats
        self.policy = policy if policy is not None else DraftPolicy()
        self.max_recoveries = int(max_recoveries)
        self.idle_s = idle_s
        self.wants_marks = bool(marks)
        # side-slot routing (PromptCache.route): on by default (caminus serve --no-side-route turns it off); the server passes
        # max_new to plan / prefill, so a side call's generation is known to fit the side slot
        self.side_route = bool(side_route) and bool(getattr(pc, 'side_rows', 0))
        self.last_route = None      # the last request's routing decision (PromptCache.route's dict)
        self._side = False          # the request in flight runs in the side slot
        self.drafter = SessionDrafter(self) if dr is not None else None
        if self.drafter is not None:
            self.drafter.spec_type = getattr(dr, 'spec_type', 'draft-dflash')   # ('draft-mtp' for an MTP drafter)
        self.last_prefill = PrefillStats()
        self._planned = None
        self._pass = None           # the last pass awaiting accept: dict(i0, F, seq)
        self._dev = None            # the last device step's next drafts: (anchor position, anchor, drafts)
        self._kind = None           # what _drafts() gave for the coming pass: 'dflash' | 'device' | 'filler' | 'plain'
        self._mode = 'draft'
        self._hooks = _Hooks(self)
        self._inflight = None       # 'step': a device step under way (abort drains the drafter's step when it was launched)
        self._step_ep = None        # the drafter's launch epoch (DFlashK3.token) before that step
        self._bails_seen = len(getattr(m, 'bails', ()))
        self.counters = dict(own_topk=0, device_passes=0, host_greedy_passes=0, sampled_passes=0, full_rows=0, wide_rows=0, plain_passes=0,
                             recoveries=0, recover_failed=0, prefill_recoveries=0, cancelled_prefills=0, aborts=0, aborts_commit0=0,
                             dropped_live=0, parks=0, filler_passes=0, idle_skipped=0, side_calls=0, side_recoveries=0,
                             side_ended=0, side_dropped=0)
        self.drafts_real = True     # the coming pass's drafts are DFlash's (False: filler; run_generation reads it)
        self._refused_seen = int(pc.stats.get('refused', 0))
        self.last_first_logits = None   # the last request's first logits row (the flight recorder's bundle)
        self._park_mark = None      # (positions, tok_sha of the log's first positions) of this session's last park of a line
        self._park_pending = None   # the same for a background park not yet confirmed (pc.parked says when it landed)
        # metrics from the engine (M7 ops): per-card pass spans (device events, opt-in on the engine), the exchange payload,
        # the floor's per-card bytes and the KV bytes per position
        on = stats is not None and stats.enabled
        self.card_timing = bool(on and hasattr(m, 'enable_card_timing'))
        if self.card_timing:
            m.enable_card_timing(True)
        self.link_bytes_per_pass = 2 * int(getattr(m, 'n_ex', 0)) * self.T * int(getattr(m, 'D', 0)) * 4
        self.physics_source = 'spec'
        if stats is not None:
            stats.exchange_wait_note = EXCHANGE_WAIT_NOTE
            card, kv = engine_physics(m, pc)
            ph = stats.physics
            if card is not None:
                ph.card_bytes, self.physics_source = card, 'engine'
            if kv is not None:
                ph.kv_bytes_per_token = kv
            q = getattr(m, 'qsa_from', capacity.QSA_EXACT)   # (None: dense attention at every depth, the one-card engines)
            ph.qsa_from = int(q) if q is not None else 1 << 40
            ph.source = self.physics_source
            try:   # the guard's lines and the display card's reserve from this machine (caminus.device.machine; here (23800, 19850) / (512, 0))
                devs = list(getattr(m, 'devs', ()))
                if len(devs) == len(ph.guard_mib):
                    ph.guard_mib = tuple(int(capacity.guard_line_mib(d)) for d in devs)
                    ph.reserve_mib = tuple(capacity.DISPLAY_RESERVE_MIB if capacity.is_display(d) else 0 for d in devs)
                    cs = [capacity._machine.by_card(capacity.drm_card(d)) for d in devs]
                    if all(cs):
                        ph.card_names = tuple(c['label'] for c in cs)
                        from ..arch import CARD_BW     # (each card's bandwidth by its label, as the names: not by position)
                        ph.card_bw = tuple(CARD_BW.get(c['label'], CARD_BW['XT']) for c in cs)
            except Exception:   # noqa: BLE001 - a stand-in engine without cards keeps the defaults
                pass
        # test hook (passpolice's positive control, here on this session's own passes)
        hooks = passpolice.parse_hooks(os.environ)
        self._bail_at = hooks.get('bail')
        self._n_pass = 0
        self._armed = False
        if self._bail_at is not None:
            print(f'K3Session: TEST HOOK ARMED: pass {self._bail_at} bails on every attempt (K3_TEST_BAIL)', flush=True)
            self._wrap_launch()
        # the device sampler's vocabulary shards: FlashNextTP's two cards, or a one-card engine's two column views of its row on
        # one card (Qwen35One.shard_devs: candsample merges two shards' candidates exactly either way)
        dev = [torch.device(f'cuda:{d}') for d in getattr(m, 'shard_devs', m.devs)]
        # the pinned candidate buffers' width: top_k <= 64 + the default penalty window (64 ids) + the pass's drafts + the
        # tie margin; a wider k' (a longer penalty window) takes the full rows (_sampled_pass): no large pinned buffers
        kmax = self.kmax = CS.MAX_TOP_K + 64 + 2 * self.T + CS.TIE_MARGIN + 8
        self._dev_t = dev
        self._h_val = [torch.zeros(self.T, kmax, dtype=torch.float32, pin_memory=True) for _ in (0, 1)]
        self._h_idx = [torch.zeros(self.T, kmax, dtype=torch.int64, pin_memory=True) for _ in (0, 1)]
        self._h_pen = [torch.zeros(self.T, kmax, dtype=torch.float32, pin_memory=True) for _ in (0, 1)]
        self._ev = [None, None]
        # the device sampler's own kernel and its buffers, one per shard, allocated (and checked against each card's cap)
        # here, when the session is built: a request never meets an allocation
        self._cand = [topk.Candidates(dev[c], self.T, kmax, cols=w) for c, w in ((0, self.vh), (1, self.n_vocab - self.vh))]
        import collections
        self.samp_t = collections.deque(maxlen=4096)    # per sampled pass: (rows wait ms, host sampling ms, rows emitted)

    # ------------------------------------------------------------------------------------------------ helpers
    @property
    def toks(self):
        """the live tokens: the prompt cache's (it owns the live state; caminus.cache.livestate). The cache's own list while the
        state is live: read it, never modify it"""
        return self.pc.live_tokens()

    @property
    def st(self):
        """the live state dict(i0, commit): the prompt cache's (read only)"""
        return self.pc.st

    @property
    def tokens(self):
        return tuple(self.toks)

    @property
    def _pe(self):
        """the request's prompt-end state (the bail recovery's checkpoint): the cache's"""
        return self.pc.pe

    @property
    def _prompt(self):
        return self.pc.prompt_ids

    def _event(self, kind, side):
        """a recovery the cache ran for this session's request: its counters and stats (as before, at the same moments)"""
        st_ = self.stats
        on = st_ is not None and st_.enabled
        if kind == 'prefill_recovery':
            self.counters['prefill_recoveries'] += 1
            if on: st_.bail(True)
            if side: self.counters['side_recoveries'] += 1
        elif kind == 'recover_failed':
            self.counters['recover_failed'] += 1
            if st_ is not None: st_.bail(False)
        elif kind == 'recovered':
            nb = len(getattr(self.m, 'bails', ()))
            if on:
                st_.retry(max(0, nb - self._bails_seen))   # (the failed pass's attempts and any in the replay: pass_retries_total)
                st_.bail(True)
            self._bails_seen = nb
            self.counters['recoveries'] += 1
            if side: self.counters['side_recoveries'] += 1

    def _wrap_launch(self):
        m, sess = self.m, self
        launch = m._pass_launch

        def _pass_launch(i0, commit):
            if not sess._armed:
                return launch(i0, commit)
            eng = passpolice.Police._all_engines(m)
            saved = [e.limit_us for e in eng]
            try:
                for e in eng: e.limit_us = 1          # K3_TEST_BAIL: this attempt bails at its first wait
                return launch(i0, commit)
            finally:
                for e, lu in zip(eng, saved): e.limit_us = lu
        m._pass_launch = _pass_launch

    def _arm(self):
        """count this session's target passes; arm the test bail on the chosen one (every attempt of it)"""
        if self._bail_at is not None:
            self._armed = self._n_pass == self._bail_at
        self._n_pass += 1

    def _bail_stats(self):
        st = self.stats
        b = getattr(self.m, 'bails', ())
        new = len(b) - self._bails_seen
        self._bails_seen = len(b)
        if new > 0 and st is not None and st.enabled:
            st.retry(new)
            st.bail(True)

    def _cls(self):
        st = self.stats
        return st.cls if st is not None and st.enabled else 'prose'

    def _sync(self):
        for d in self.m.devs: torch.cuda.synchronize(d)

    # ------------------------------------------------------------------------------------------------ plan / prefill
    def _route_arg(self):
        return 'auto' if self.side_route else False

    def side_question(self, ids, marks, max_new):
        """PromptCache.side_question_ids: the side question asked against the stored conversation, or None (as sent)"""
        return self.pc.side_question_ids([int(t) for t in ids], marks, int(max_new))

    def plan(self, ids, max_new=None, hint=None):
        ids = [int(t) for t in ids]
        n = int(self.pc.plan(ids, side=self._route_arg(), max_new=max_new, hint=hint)['cached'])
        self._planned = (ids, n, max_new, hint)   # (the worker plans the job first; prefill reuses it for cached_known)
        return n

    def prefill(self, ids, cancel=None, marks=None, max_new=None, hint=None):
        ids = [int(t) for t in ids]
        if not ids:
            raise ValueError('empty prompt')
        clock = self.stats.clock if self.stats is not None else time.perf_counter
        t0 = clock()
        self.last_first_logits = None
        pfs = PrefillStats(prompt_n=len(ids))
        pfs.lcp = common_prefix(self.toks, ids)
        self._pass, self._dev = None, None
        n_plan = 0
        if self.stats is not None:
            pl = self._planned
            n_plan = pl[1] if pl is not None and pl[0] == ids and pl[2] == max_new and pl[3] == hint else self.plan(ids, max_new, hint)
            self.stats.cached_known(n_plan)
        self._planned = None
        last_before = getattr(self.pf, 'last', None)
        mk = None
        if marks:
            mk = {k: v for k, v in marks.items() if isinstance(v, int) and 0 < v < len(ids)}
        # the cache runs the prompt phase (prepare, its bail retry, the prompt's last pass and its recovery): it knows when the
        # device state is not known, and abort() asks it (PromptCache.settle)
        self._side = False
        on = self.stats is not None and self.stats.enabled
        prog = _Progress(cancel, self.stats, n_plan) if on else cancel
        pc = self.pc
        try:
            r = pc.prompt(ids, mk or None, side=self._route_arg(), cancel=prog, max_new=max_new, hint=hint,
                          max_recoveries=self.max_recoveries, hooks=self._hooks)
        except RecoveryFailed as f:
            raise EngineError(str(f)) from None
        except (ContextBoundsError, PrefillRefused) as e:
            if pc.phase != 'prepare':
                self._refused(e)
                raise
            if self.stats is not None:
                self.stats.refusal('prefill' if isinstance(e, PrefillRefused) else 'context')
            raise EngineStateError(str(e)) from None
        except RuntimeError as e:
            if pc.phase != 'prepare':
                self._refused(e)
                raise
            if isinstance(e, RoutingError) and self.stats is not None:
                self.stats.refusal('routing_flag')
            raise EngineError(f'prefill: {e}') from None
        except Exception as e:                           # noqa: BLE001 - re-raised
            if pc.phase == 'first':
                self._refused(e)
            raise
        st = r['st']
        self._side = bool(r.get('side'))
        rt = r.get('route') or dict(route='side' if self._side else 'main', reason='off')
        self.last_route = rt
        if self._side: self.counters['side_calls'] += 1
        if self.stats is not None: self.stats.route(rt)
        pfs.cached_n, pfs.source, pfs.restore = int(r['cached']), r['source'], RESTORE_OF.get(r['source'], 'cold')
        pfs.restore_ms = float(r['ms'].get('restore', 0.0) + r['ms'].get('park', 0.0))
        pfs.slot = 'side' if self._side else 'main'
        pfs.tail = rt.get('tail')   # (a side question's history against the stored conversation: PromptCache.tail_match)
        last = getattr(self.pf, 'last', None)
        batched = last is not None and last is not last_before
        last = last or {}
        if r.get('cancelled'):
            pfs.cancelled = True
            pfs.processed_n = int(st['i0']) - pfs.cached_n
            pfs.ms = (clock() - t0) * 1e3
            self.last_prefill = pfs
            self.counters['cancelled_prefills'] += 1
            self._report_cache()
            return None
        pfs.exact = int(st['i0']) == len(ids)
        self._armed = False
        self._bail_stats()
        lg = r['logits'].numpy()
        self.last_first_logits = lg
        pfs.passes_ms = float(r['ms']['first']) + max(0.0, r['ms'].get('process', 0.0) - (last.get('prefill_ms', 0.0) + last.get('inject_ms', 0.0) if batched else 0.0))
        if batched:
            pfs.batched_ms = float(last.get('prefill_ms', 0.0))
            pfs.handoff_ms = float(last.get('inject_ms', 0.0))
        pfs.processed_n = len(ids) - pfs.cached_n
        pfs.passes = 1 + (len(ids) - 1 - int(st['i0'])) // self.T if int(st['i0']) < len(ids) else 1
        pfs.ms = (clock() - t0) * 1e3
        self.last_prefill = pfs
        self._report_cache()
        return lg

    # ------------------------------------------------------------------------------------------------ recovery
    def _recover(self, err):
        """a pass of the generation failed after the engine's retries: the cache restores the request's prompt-end checkpoint
        and replays the processed tokens (PromptCache.recover_pass); the failed pass then runs again from the returned state.
        EngineError when it cannot (no checkpoint, too many failures, poisoned). (A drafter step queued behind a failed verify
        was drained by DFlashK3Drafter.device_step.)"""
        self._inflight = None
        try:
            st = self.pc.recover_pass(err, list(self.toks), max_recoveries=self.max_recoveries, hooks=self._hooks)
        except RecoveryFailed as f:
            raise EngineError(str(f)) from None
        self._dev = None
        return st

    # ------------------------------------------------------------------------------------------------ drafts
    def _drafts(self, toks, k):
        """the drafts of the coming pass (SessionDrafter); records which kind for sample_pass, and in drafts_real whether
        they are DFlash's (plain mode and the drafter's filler, a window gap or a block past max_pos, are not drafts)"""
        self.drafts_real = True
        if k <= 0 or self.dr is None:
            self._kind = 'filler'
            return []
        pos = len(toks) - 1
        self._mode = self.policy.decide(self._cls(), pos)
        if self._dk == 'mtp':
            prm = self.request_params
            if prm is not None and not (prm.greedy and not prm.penalties_on):
                self._mode = 'plain'       # (a sampled request runs the T = 1 tape: no draft round)
        if self._mode == 'plain':
            self._kind = 'plain'
            self.drafts_real = False
            return [int(toks[-1])] * k
        if self._dk == 'mtp':
            self._kind = 'mtp'
            self._dev = None
            while True:
                try:
                    out = list(self.dr(list(toks), k))
                except RuntimeError as e:
                    if not _bail(e):
                        raise
                    self._recover(e)
                    continue
                self.drafts_real = not getattr(self.dr, 'filler', False)
                if not self.drafts_real:
                    self.counters['filler_passes'] += 1
                return out
        d = self._dev
        if d is not None and d[0] == pos and d[1] == int(toks[-1]) and k == self.T - 1:
            self._kind = 'device'
            return list(d[2])
        self._dev = None
        self._kind = 'dflash'
        while True:
            df_st = self.df.stats
            n0 = df_st['draft_launches'] + df_st['fused']
            try:
                out = list(self.dr(list(toks), k))
            except RuntimeError as e:
                if not _bail(e):
                    raise
                self._recover(e)
                continue
            self.drafts_real = df_st['draft_launches'] + df_st['fused'] > n0     # (no draft launch: the drafter's filler)
            if not self.drafts_real:
                self.counters['filler_passes'] += 1
            return out

    # ------------------------------------------------------------------------------------------------ passes
    def _take_h(self, i0, n, toks):
        """the trunk's h rows of the pass just run into the MTP drafter (its ring, and the block rows i0 + 1 .. i0 + n
        filled from them): the verify tape's buf['h'] for a T-wide pass, the decode tape's buf['h'] for a T = 1 one"""
        if self._dk != 'mtp' or self.dr is None: return
        h = self._ver.buf['h'] if (self._ver is not None and n == self.T) else self.m.buf['h']
        self.dr.take_h(i0, h[:n * self.m.D], toks)

    def _seq(self, feed, drafts):
        toks = self.toks
        i0 = len(toks)
        body = [int(t) for t in feed] + [int(t) for t in drafts]
        if not 1 <= len(feed) <= self.T or len(body) > self.T:
            raise EngineStateError(f'K3Session: a pass of {len(feed)} fed + {len(drafts)} drafts with T = {self.T}')
        return i0, toks + body + [body[-1]] * (self.T - len(body))

    def verify(self, feed):
        """a pass whose logits are not read (forced tool-call tokens beyond one pass): its features injected; returns None"""
        if self._pass is not None:
            raise EngineStateError('verify without accept')
        i0, seq = self._seq(feed, [])
        while True:
            self._arm()
            try:
                if self.df is not None: self.df.flush()
                if self._dk == 'mtp':
                    self._ver.run(list(seq[i0:i0 + self.T]), i0, int(self.st['commit']), argmax=False)
                    self._take_h(i0, self.T, [int(x) for x in seq[i0 + 1:i0 + self.T]] + [int(seq[i0 + self.T - 1])])
                else:
                    self.m.verify_tokens(seq, i0, int(self.st['commit']))
                    if self.dr is not None: self.dr.inject(i0, self.m.features); self.df.flush()
                break
            except Exception as e:
                self._refused(e)
                if not _bail(e):
                    raise
                self._recover(e)
        self._armed = False
        self._bail_stats()
        self._pass = dict(i0=i0, F=len(feed), seq=seq)
        self._dev = None
        return None

    def accept(self, n):
        p = self._pass
        if p is None:
            raise EngineStateError('accept without verify')
        if not 0 <= n <= p['F'] + (self.T - p['F']):
            raise EngineStateError(f'accept {n} of a {self.T}-token pass')
        i0 = p['i0']
        self.pc.append(p['seq'][i0:i0 + n], dict(i0=i0 + n, commit=n))   # (the pass's accepted tokens: the cache's live state moves)
        self._pass = None

    def sample_pass(self, feed, drafts, sampler, history, stop_ids=()):
        if self._pass is not None:
            raise EngineStateError('verify without accept')
        st = self.stats
        on = st is not None and st.enabled
        clock = st.clock if on else time.perf_counter
        prm = sampler.params
        stop = frozenset(stop_ids)
        F, k = len(feed), len(drafts)
        kind = self._kind
        self._kind = None
        greedy_plain = prm.greedy and not prm.penalties_on
        if self._dk == 'mtp' and not greedy_plain and k > 0:
            drafts, k = [], 0          # (a sampled pass on an MTP session runs the T = 1 tape, no drafts)
            if kind == 'mtp': kind = 'plain'
        t0 = clock()
        tw = None
        step_ms = None
        while True:
            i0, seq = self._seq(feed, drafts)
            commit = int(self.st['commit'])
            try:
                if greedy_plain and self.dr is not None and kind in ('device', 'dflash') and F + k == self.T and self.dr.step_ok(i0):
                    self._arm()
                    if self.df.queued is not None: self.df.flush()
                    self._step_ep = int(self.df.token)
                    self._inflight = 'step'
                    r, nd, tm = self.dr.device_step(self.m, seq, i0, commit, F, timed=(self.policy.n % 16 == 1))
                    self._inflight = None
                    j, bonus = int(r['j']), int(r['anchor'])
                    out = seq[i0 + F:i0 + F + j] + [bonus]
                    self._dev = (i0 + F + j, bonus, list(nd))
                    if tm:
                        try: step_ms = self.df.step_ms()
                        except Exception: step_ms = None       # noqa: BLE001 - a timing read only
                    self.counters['device_passes'] += 1
                    tw = clock()
                elif greedy_plain:
                    self._arm()
                    if self.df is not None: self.df.flush()
                    if self._dk == 'mtp' and kind == 'mtp' and F + k == self.T:
                        # the T = 4 verify pass on the engine's states; its h rows go to the drafter with the pass's
                        # tokens (the last row's bonus as the last token: a rejected one is rewritten by the pass
                        # that verifies that position again)
                        a = self._ver.run(list(seq[i0:i0 + self.T]), i0, commit, argmax=True)
                        self._take_h(i0, self.T, [int(x) for x in seq[i0 + 1:i0 + self.T]] + [int(a[self.T - 1])])
                    else:
                        a = self.m.verify_tokens(seq, i0, commit)
                        if self.dr is not None: self.dr.inject(i0, self.m.features)
                        if self._dk == 'mtp': self._take_h(i0, 1, [int(a[0])])
                    tw = clock()
                    if len(a) == self.T and k > 0:
                        j = 0
                        while j < k and drafts[j] == a[F - 1 + j]: j += 1
                        out = [int(x) for x in drafts[:j]] + [int(a[F - 1 + j])]
                    else:
                        out = [int(a[0])]
                    self._dev = None
                    self.counters['host_greedy_passes'] += 1
                else:
                    self._arm()
                    if self.df is not None and kind != 'dflash': self.df.flush()
                    tw, out = self._sampled_pass(seq, i0, commit, F, k, drafts, sampler, history, stop, clock,
                                                 plain=kind not in ('dflash', 'device'))
                    if self.dr is not None: self.dr.inject(i0, self.m.features)
                    self._dev = None
                    self.counters['sampled_passes'] += 1
                break
            except Exception as e:
                self._refused(e)
                if not _bail(e):
                    raise
                self._recover(e)
                if kind == 'device':
                    kind = 'dflash'             # (the device drafts are host ints: the same pass runs again)
        self._armed = False
        if kind == 'plain':
            self.counters['plain_passes'] += 1
        for jj, t in enumerate(out):
            if t in stop:
                out = out[:jj + 1]
                break
        self._pass = dict(i0=i0, F=F, seq=seq)
        self._bail_stats()
        t1 = clock()
        if on:
            cm = getattr(self.m, 'card_ms', None) if self.card_timing else None
            st.pass_split(((tw or t1) - t0) * 1e3, (t1 - (tw or t1)) * 1e3,
                          (cm, None, self.link_bytes_per_pass) if (cm is not None or self.link_bytes_per_pass) else None)
        if self.dr is not None and kind in ('device', 'dflash', 'plain', 'mtp'):
            self.policy.record('plain' if kind == 'plain' else 'draft', self._cls(), i0, (t1 - t0) * 1e3, len(out), step_ms)
            if on:
                st.plain_estimate(self.policy.bucket(self._cls(), i0).plain_cost())
        return out

    def _refused(self, e):
        """a host refusal raised out of a pass: counted by kind (refusals_total)"""
        k = refusal_kind(e)
        if k is not None and self.stats is not None:
            self.stats.refusal(k)

    def queue_rows(self, r0, R, pid, kp):
        """the device sampler's device half for rows r0 .. r0 + R - 1 of the last verify (the logits shards each card holds):
        each card's top-kp raw logits with their ids and the raw logits of the penalty ids pid in its range, queued on each
        card's current stream behind the pass, copied into pinned host buffers (rows_wait() collects them)"""
        if kp is None:
            return
        if not 0 < kp <= self.kmax or len(pid) > self.kmax:
            raise ValueError(f'K3Session.queue_rows: k\' {kp} / {len(pid)} penalty ids past the candidate buffers ({self.kmax} wide)')
        m = self.m
        for c in (0, 1):
            cand = self._cand[c]
            with torch.cuda.device(self._dev_t[c]):
                L = m.logits_half[c][r0:r0 + R]
                lo, hi = (0, self.vh) if c == 0 else (self.vh, self.n_vocab)
                sel = pid[(pid >= lo) & (pid < hi)]
                cand.run(L, kp, sel - lo)          # (Caminus's own kernel: top-kp values and ids, the penalty ids' raw values)
                # whole rows of the buffers (contiguous blocks: one copy each, no copy kernel); rows_wait reads the first kp
                self._h_val[c][:R].copy_(cand.val[:R], non_blocking=True)
                self._h_idx[c][:R].copy_(cand.idx[:R], non_blocking=True)
                if len(sel):
                    self._h_pen[c][:R].copy_(cand.pen[:R], non_blocking=True)
                e = torch.cuda.Event(); e.record(); self._ev[c] = e
        self.counters['own_topk'] += 1

    def cards_wait(self, R, kp):
        """queue_rows' candidates as arrays (V0, I0, V1, I1) [R, k'] (values descending, global ids; views of the pinned
        buffers, valid until the next queue_rows)"""
        for c in (0, 1):
            self._ev[c].synchronize()
        kk = min(kp, self.vh, self.n_vocab - self.vh)
        return (self._h_val[0][:R, :kk].numpy(), self._h_idx[0][:R, :kk].numpy(), self._h_val[1][:R, :kk].numpy(),
                self._h_idx[1][:R, :kk].numpy() + self.vh)

    def rows_wait(self, R, pid, kp):
        """[candsample.Row] of queue_rows' rows (the cards' candidates; Row() without them when kp is None)"""
        if kp is None:
            return [CS.Row() for _ in range(R)]
        for c in (0, 1):
            self._ev[c].synchronize()
        kk = min(kp, self.vh, self.n_vocab - self.vh)
        n0 = int((pid < self.vh).sum())
        rows = []
        for j in range(R):
            c0 = (self._h_val[0][j, :kk].numpy().copy(), self._h_idx[0][j, :kk].numpy().copy())
            c1 = (self._h_val[1][j, :kk].numpy().copy(), self._h_idx[1][j, :kk].numpy() + self.vh)
            pv = np.concatenate((self._h_pen[0][j, :n0].numpy(), self._h_pen[1][j, :len(pid) - n0].numpy())) if len(pid) else np.zeros(0, np.float32)
            rows.append(CS.Row(cards=[c0, c1], pid=pid, pvals=pv))
        return rows

    def full_row(self, r):
        """row r of the last verify, whole (float32 [V] on the host: both cards' shards)"""
        parts = []
        for c in (0, 1):
            with torch.cuda.device(self._dev_t[c]):
                parts.append(self.m.logits_half[c][r].float().cpu().numpy())
        return np.concatenate(parts)

    def _sampled_pass(self, seq, i0, commit, F, k, drafts, sampler, history, stop, clock, plain=False):
        """the verify with the device sampler: each card's top-k' of rows F - 1 .. F - 1 + k and the penalty ids' raw logits
        (queued behind the pass on each card's stream, again after a retried pass), then candsample's sample and match"""
        m, prm = self.m, sampler.params
        R = k + 1
        pid = CS.penalty_ids(prm, history, drafts)
        kp = CS.k_prime(prm, len(pid))
        if kp is not None and kp > self.kmax:
            kp = None               # (a penalty window wider than the pinned candidate buffers: every row whole, the reference)
            self.counters['wide_rows'] += 1
        m.launch_tokens(seq, i0, commit)
        self.queue_rows(F - 1, R, pid, kp)
        if m.finish_pass(i0, commit):
            self.queue_rows(F - 1, R, pid, kp)
        tw = clock()
        ta = time.perf_counter()

        def fetch(j):
            self.counters['full_rows'] += 1
            return self.full_row(F - 1 + j)
        if kp is not None and CS.batched_ok(prm) and plain is False:   # (penalties off, DFlash drafts: every row drawn at once)
            base = len(history)
            us = [sampler.uniform(base + j) for j in range(R)]          # (host work while the cards' top-k finishes)
            V0, I0, V1, I1 = self.cards_wait(R, kp)
            tb = time.perf_counter()
            out = CS.sample_and_match_batched(V0, I0, V1, I1, sampler, list(drafts), list(history), stop, fetch=fetch, us=us)
        else:
            rows = self.rows_wait(R, pid, kp)
            tb = time.perf_counter()
            out = CS.sample_and_match(rows, sampler, list(drafts), list(history), stop, fetch=fetch)
        self.samp_t.append(((tb - ta) * 1e3, (time.perf_counter() - tb) * 1e3, len(out)))
        return tw, out

    # ------------------------------------------------------------------------------------------------ end / abort / reset
    def end(self):
        if self._pass is not None:
            raise EngineStateError('end without accept')
        if self.toks:
            self.pc.commit(self.toks, self.st)   # (the passes were appended as they ran: what end() adds is extend()'s whole-log check)
        self._report_cache()

    def abort(self):
        """after a host-side exception (the worker's 'internal' error; between calls or inside one): drain what is in flight
        (the target pass launched since the cache last vouched for the state: its status re-read; the drafter's step behind
        it), then report once to the cache, which decides (PromptCache.settle; live state step 3, 11-07: before, the session
        decided with its own copy of the state and its own epochs):
          * a pass that returned and was not accepted: dropped, (its i0, commit 0);
          * a target pass launched and not returned (an exception inside a pass after the launch, e.g. in the drafter's step,
            the device sampler or the inject): (st's i0, commit 0): the launch applied st's commit on the device; committing
            it again in the next pass would apply it twice;
          * no launch since: st as it was (a refusal before the launch keeps its commit);
          * not known (a failure while the prompt was processed or during a bail recovery, a drain that failed, a launch at
            another position): the live state is dropped (every checkpoint kept, the next request restores one);
          * the engine was reset since the last completed operation (e.g. inside end(): the cache resets its slot when it
            refuses a state): dropped too (final round 4, the verifier's Y2).
        The cache resets only when the engine is poisoned (or it refuses the state); the session then forgets what it had in
        flight"""
        self.counters['aborts'] += 1
        m, df, pc = self.m, self.df, self.pc
        inf, self._inflight = self._inflight, None
        launched = pc.launched()
        drained = True
        if launched:
            try: m.finish_pass(int(self.st['i0']), int(self.st['commit']))   # (a pass already waited for: its status re-read)
            except BaseException: drained = False      # noqa: BLE001
        if inf == 'step' and df is not None and int(df.token) != self._step_ep:
            try: drained &= not df.step_wait()          # (the drafter's step behind it: its result taken, or its bail reported)
            except BaseException: drained = False      # noqa: BLE001
        self._sync()
        p, self._pass, self._dev = self._pass, None, None
        try:
            out = pc.settle(returned=p is not None, drained=drained)
        except Exception:                               # noqa: BLE001 - the cache refused the state and reset: so does the session
            self._forget()
            raise
        if out == 'reset':
            self._forget()
        elif out in ('dropped', 'dropped_side'):
            self.counters['dropped_live'] += 1
            if out == 'dropped_side': self.counters['side_dropped'] += 1
        elif out == 'commit0':
            self.counters['aborts_commit0'] += 1

    def _forget(self):
        """the session's own part of a reset (the cache's live state went cold): nothing in flight"""
        self._pass, self._dev, self._inflight = None, None, None
        self._bails_seen = len(getattr(self.m, 'bails', ()))

    def reset(self):
        self.pc.reset()
        self._forget()

    def _resolve_park(self):
        """a background park requested earlier: its mark counts once pc.parked shows it landed (a refused / failed one never)"""
        p, self._park_pending = self._park_pending, None
        pk = self.pc.parked
        if p is not None and pk is not None and pk[0][0] == p[1] and int(pk[0][1]) == p[0]:
            self._park_mark = p

    def park_gain(self):
        """positions the live log holds beyond this session's last park of the same line (the whole log for another line, or
        when nothing was parked); 0 for an unchanged slot"""
        from ..cache.prompt import tok_sha
        log = list(self.pc.log)
        mk = self._park_mark
        if mk is not None and mk[0] <= len(log) and tok_sha(log[:mk[0]]) == mk[1]:
            return len(log) - mk[0]
        return len(log)

    def _end_side(self):
        """an active side call ended now (idle / shutdown park: park() is refused during one): the main line back; the session's
        log and state follow the cache's (the main line's processed tokens)"""
        if self.pc.side is None:
            return False
        self.pc.end_side()
        self._pass, self._dev = None, None
        self.counters['side_ended'] += 1
        return True

    def _park(self, background):
        from ..cache.prompt import tok_sha
        log = list(self.pc.log)
        mk = (len(log), tok_sha(log))
        nm = self.pc.park(background=background)
        if background:
            self._park_pending = mk
        elif nm is not None:
            self._park_mark = mk
        return nm

    def idle(self):
        """no request for idle_s seconds: park the live conversation on the disk tier in the background, when it gained at
        least pc.auto_park_min positions since this session's last park of the line (the lead's rule, 11-05: a park in
        flight makes the next request wait for it; a short or unchanged conversation is not worth a write)"""
        if self.pc.store is None or not self.toks:
            return
        try:
            self.pc.fence()
            self._end_side()
            self._resolve_park()
            if self.park_gain() < int(self.pc.auto_park_min):
                self.counters['idle_skipped'] += 1
                return
            self._park(True)
            self.counters['parks'] += 1
        except Exception:                               # noqa: BLE001 - a park that cannot run is skipped
            pass

    def shutdown_park(self):
        """a clean shutdown (the engine worker stopped: this thread is the only caller): park the live conversation when it
        changed since this session's last park, in the foreground. Returns the entry name, 'unchanged', or None (nothing to
        park, no disk tier, refused: pc.park_error)"""
        if self.pc.store is None or not self.toks:
            return None
        self.pc.fence()
        self._end_side()
        if int(self.pc.st['i0']) < 1:
            return None
        self._resolve_park()
        if self.park_gain() == 0:
            return 'unchanged'
        nm = self._park(False)
        if nm is not None:
            self.counters['parks'] += 1
        return nm

    # ------------------------------------------------------------------------------------------------ the flight recorder
    def request_marks(self):
        """counters the worker diffs around a request (the flight recorder's per-request bails / recoveries)"""
        c = self.counters
        return dict(bails=len(getattr(self.m, 'bails', ())), recoveries=c['recoveries'], recover_failed=c['recover_failed'],
                    prefill_recoveries=c['prefill_recoveries'])

    def bundle_state(self, prompt=None):
        """what a replay bundle needs of the last request, read right after it (the worker's record): the prompt-end state, the
        cache source, the first logits, the tokens generated so far (an aborted request's). Only when the session's last
        processed prompt is `prompt` (a failure before its prompt end leaves none: never another request's state)"""
        pf = self.last_prefill
        if self._prompt is None or (prompt is not None and list(self._prompt) != list(prompt)):
            return dict(pe=None, source=None, cached=None, first=None, generated=[])
        gen = self.toks[len(self._prompt):] if self.toks[:len(self._prompt)] == list(self._prompt) else []
        return dict(pe=self._pe, source=pf.source, cached=int(pf.cached_n), first=self.last_first_logits, generated=gen)

    def export_bundle(self, path, rec, flight):
        """the engine part of a replay bundle (engine thread): the engine key and the cache-state reference; the request's
        start state (its prompt-end checkpoint with the rows below it) written under path/state when the request did not start
        cold (pc.export_checkpoint: a cold start is reproduced by the same prefill)"""
        pc, m = self.pc, self.m
        eng = dict(key_sha=pc.key_sha, key=pc.key, max_ctx=int(m.max_ctx), T=self.T, drafter=self.dr is not None,
                   drafter_path=getattr(self.df, 'path', None))
        pe, src = rec.get('pe'), rec.get('source')
        ref = dict(source=src, cached=rec.get('cached'), prompt_end=list(pe) if pe else None, state=None)
        if pe is None:
            ref['note'] = 'no prompt-end state (the prompt was not processed): the replay starts cold'
        elif src == 'cold':
            ref['note'] = 'the request started cold: the replay runs the same prefill from an empty engine (bitwise)'
        else:
            L = pc.layout
            need = int(sum(L.rows_bytes(int(pe[0]))) + sum(L.nbytes) + (64 << 20))
            if not flight.state_room(need):
                ref['note'] = (f'no room for the {need / 1e9:.2f} GB start state (the flight cap or the free-space floor): a replay '
                               f'from cold is bitwise only for a prefill-built prefix')
            else:
                r = pc.export_checkpoint(pe, os.path.join(path, 'state'))
                if r is None:
                    ref['note'] = f'start state not saved: {pc.export_error}'
                else:
                    ref.update(state=os.path.join('state', r['name']), state_bytes=r['bytes'], state_ms=round(r['ms'], 1))
        return dict(engine=eng, cache_ref=ref)

    def _report_cache(self):
        st = self.stats
        if st is None or not st.enabled:
            return
        pc = self.pc
        nref = int(pc.stats.get('refused', 0))
        if nref > self._refused_seen:
            st.refusal('cache', nref - self._refused_seen)      # (disk entries refused: served cold)
        self._refused_seen = nref
        st.cache_update(pc.stats, {'n_slots': pc.n_slots, 'slots_used': len([c for c in pc.ckpts if c.slot is not None]),
                                   'tier_mib': round(float(pc.tier_mib[0]), 1), 'headroom_mib': None if pc.headroom is None else round(float(pc.headroom[0]), 1),
                                   'disk_bytes': pc.store.total_bytes() if pc.store is not None else None,
                                   'disk_cap_bytes': pc.store.cap if pc.store is not None else None,
                                   'side_active': pc.side is not None, 'side_rows': int(getattr(pc, 'side_rows', 0) or 0)})
