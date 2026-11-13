"""The batched prefill driver (docs/specs/spec-prefill-cache.md M5 S6, milestone A: correct up to capacity.QSA_EXACT =
2051 positions; S7: on a QSA engine up to its max_ctx, rows from 2051 on through the QSA prompt path): the chunk loop
over caminus.prefill.tp.ChunkStack (S2 GEMMs, S3 glue, S4 GDN, S5 exchange, S6 attention core, S7 the QSA rows) on a
FlashNextTP engine's own weights and states, the hand-off to K3, the routing of short prompts to K3 T-passes, and the
DFlash window's features.

Validation up front (validate()): before ANY reset, KV or state write, the whole prompt is checked on the host: every token
id inside the token embedding's rows (and the drafter's, if one is given), the batched positions inside the engine's KV
rows and below the engine's limit (Prefill.limit: its max_ctx on a QSA engine, else QSA_EXACT), the engine not poisoned, the drafter able to take a prefill (DFlash: features; MTP needs the
target's per-pass h rows, which a batched prefill does not produce: refused). The feed alone refuses bad tokens one chunk
at a time, after earlier chunks were written; here nothing runs.

The hand-off: the prefill writes COMMITTED state (FlashNextTP.state in place: the GDN state, conv state, PLE history, the
q8_0 KV rows and, on a QSA engine, the indexer's raw key rows IDXRAW and its pools IDXPOOL from token 0, so the QSA tape
backfills nothing after a prefill), so the state after run() is (i0 = p1, commit = 0): the first K3 pass after it
commits nothing (lazy commit), and FlashNextTP.fresh is set so step() does the same. The last prompt token always goes
through a K3 pass (spec.generate / DFlashK3Drafter.generate run it as their first pass; longctx.fill leaves it to the
caller's teacher_forced / generate), so the first logits come from decode numerics. After the chunks, both cards are
synchronized and every flag word checked (the routing flags, the feed's clamps, the attention core's row refusals);
a set flag POISONS the engine (FlashNextTP.poisoned: passes refused until reset()) and raises.

Appends: run(toks, i0, commit, p1) continues a sequence whose last K3 pass left `commit` rows uncommitted: the prefill
restarts at i0 - commit (the recurrent states are committed only to there), so those rows are re-run through the batched
path (their KV rows rewritten) instead of replaying the lazy commit (S3's conv / PLE kernels have no replay).

Routing (route()): a prompt part shorter than the measured crossover CROSSOVER (T*) runs as K3 T-passes (bitwise the
one-token decode), a longer one through the batched prefill. T* is the shortest length at which the batched prefill
(including its DFlash inject) took less time than the K3 passes, measured interleaved (tests/prefill/test_prefill_s6.py speed).

DFlash: the prefill exports OP_HC_AVG features (ChunkStack's export, on the feature card, feat_rows = 2052 positions:
the whole prompt when it ends within them, else its last feat_rows positions: ChunkStack.feat_base); run(draft=) injects
them into the drafter's ring T rows at a time through the drafter's own inject launch (the same path a K3 fill takes: copy
into the target's feature buffer, inject, flush), contiguous from max(the prefill's first position, feat_base). Positions
below that are given up first (the drafter's skip_inject: never injected, their ring slots tagged with no row of theirs,
so a window that reads one is refused, never wrong); the drafter's window is 2048 positions, inside the last feat_rows.
So a prompt within feat_rows keeps the drafter's injected count and ring slot tags what a K3-filled context gives; feature
rows past the prefill's end in the last group are zeros (those ring rows are rewritten by the next pass's inject and draft
block before any draft reads them, as a K3 fill's filler rows)."""
import concurrent.futures as cf
import time
import torch
from ..engine import k3
from ..device import capacity
from .tp import ChunkStack, bf16_wanted
from .driver import PrefillDriver, PrefillRefused, drafters   # noqa: F401  (PrefillRefused: its home since 11-07, imported from here too)

# T*: measured 11-04 (tests/prefill/test_prefill_s6.py tstar, the DFlash inject included, interleaved medians of 3, compaction_proactiveness 0):
# time to the first token for an append of L tokens, batched vs K3 T = 4 passes, from empty / at depth 1024 (ms):
# L 32: 195.9 vs 140.1 / 193.8 vs 144.2; L 48: 198.2 vs 209.5 / 196.1 vs 214.7; L 64: 198.5 vs 278.1 / 196.5 vs 285.2 (the K3 path
# costs ~4.4 ms per token, the batched one ~190 ms flat at these sizes: its host enqueue of ~4,800 launches, ~160 ms per chunk)
CROSSOVER = 48
# micro-chunks per chunk (ChunkStack.run m): 1 below MICRO2_FROM tokens, 2 from there. Measured 11-04 (chunk alone, enqueue +
# GPU, medians of 3): 256 tokens m 1 183.8 / m 2 323.2 ms; 512: 302.3 / 327.7; 1024: 505.6 / 495.3; 2048: 999.5 / 765.4 (m 2's
# second piece doubles the host enqueue, ~160 -> ~320 ms, which only the exchange overlap of a long chunk pays back). The
# results are the same bits at any m (ChunkStack G2b)
MICRO2_FROM = 1024
LIMIT = capacity.QSA_EXACT   # positions the batched prefill may write on an engine without QSA: [0, LIMIT) (Prefill.limit)


_drafters = drafters


class Prefill(PrefillDriver):
    """the batched prefill of a FlashNextTP engine m: its two-card feed and the ChunkStack for chunks of up to `chunk`
    tokens (attention core, the DFlash feature export when m has feature layers). micro: the micro-chunks per chunk;
    None: 1 below MICRO2_FROM tokens a chunk, 2 from there (measured); crossover: T* (None: CROSSOVER); router_logits /
    wmma_out: prefill_ops.Glue's modes (None: prefill_ops.ROUTER_LOGITS / WMMA_OUT); moe: ChunkStack's routed-expert path
    (None: prefill_tp.MOE_PATH); q6k / q6k_buffers: its dense Q6_K path (None: prefill_tp.Q6K_PATH; ChunkStack); graphs /
    max_graphs: HIP-graph chunks (caminus.prefill.graph; None: off, the default; needs prefill_graph.ENV in the process)"""
    def __init__(self, m, chunk=2048, micro=None, router_logits=None, crossover=None, path=None, wmma_out=None, moe=None, q6k=None,
                 q6k_buffers=None, graphs=None, max_graphs=8):
        from .feed import PrefillFeed
        from .plan import PrefillPlan, check_counted
        self.m, self.chunk, self.micro = m, int(chunk), (None if micro is None else int(micro))
        if not 1 <= self.chunk <= 2048: raise ValueError(f'Prefill: chunk {chunk} (1 .. 2048)')
        devs = [torch.device(f'cuda:{d}') for d in m.devs]
        T, bf = capacity.PREFILL_TABLE_CARD, bf16_wanted(q6k, q6k_buffers)
        counted = getattr(m, 'prefill_plan', None)
        if counted is not None:        # (no more than the engine counted before it loaded: refused before anything is allocated)
            check_counted(m, PrefillPlan.for_engine(m, self.chunk, bf16_dense=bf, feed=counted.feed), 'Prefill')
        self.feed = PrefillFeed(path or m.path, devices=(devs[T], devs[1 - T]), max_chunk=self.chunk)   # (its own checks of both caps)
        # the plan every buffer of this prefill is sized by (caminus.prefill.plan: the feed's shapes as its GGUF gives them)
        plan = PrefillPlan.for_engine(m, self.chunk, bf16_dense=bf, feed=self.feed.shape)
        assert plan.table_card == T and self.feed.lay2 == [plan.feed_layout(T), plan.feed_layout(1 - T)], 'Prefill: the feed is not the plan\'s'
        if counted is not None and counted.feed != plan.feed:
            raise ValueError(f'Prefill: the feed\'s GGUF ({path or m.path}) is not the one its engine counted ({counted.feed} vs {plan.feed})')
        self.S = ChunkStack(m, self.feed, self.chunk, router_logits=router_logits, attn='core', wmma_out=wmma_out, moe=moe, q6k=q6k,
                            q6k_buffers=q6k_buffers, plan=plan)
        self.n_vocab = self.feed.n_vocab
        self.crossover = CROSSOVER if crossover is None else int(crossover)
        self.last = None    # the last run's numbers (chunks, ms per part)
        # HIP-graph chunks (caminus.prefill.graph; M6): None / False = every chunk eager (the default); True = capture every chunk
        # structure on first use; 'never' = replay only structures already held; a set of chunk sizes = capture chunks of those
        from .graph import Graphs
        self.graphs = None if graphs is None or graphs is False else Graphs(self.S, max_graphs=max_graphs, policy='always' if graphs is True else graphs)
        # the serving configuration is complete here (target, drafter, prefill: built in that order): the staging buffers of
        # their construction (the weights' full-tensor reads, ...) sit in torch's free cache, counted by the device but by
        # no projection: returned to the driver (11-04, max_ctx 33024 + DFlash: 721 / 62.5 MiB free in torch's cache on the XTX /
        # XT, of which the fully free segments, 252 / 14 MiB, went back; the rest lies in partly used segments)
        self.released_mib = capacity.release_cache(m.devs)

    @property
    def plan(self):
        """the prefill plan its buffers are sized by (ChunkStack's: caminus.prefill.plan)"""
        return self.S.plan

    @property
    def limit(self):
        """the positions [0, limit) the batched prefill may write: a QSA engine's max_ctx (S7: rows from QSA_EXACT on through
        the QSA prompt path), else QSA_EXACT (dense attention is the model's only below it)"""
        return int(self.m.max_ctx) if getattr(self.m, 'qsa', False) else min(LIMIT, int(self.m.max_ctx))

    # -- checks -------------------------------------------------------------------------------------------------------------
    def check_drafter(self, draft):
        for d in _drafters(draft):
            if hasattr(d, 'prefill') and not hasattr(d, 'flush'):
                raise PrefillRefused(f'prefill: drafter {type(d).__name__} needs the target\'s per-pass rows (MTP: h_nextn), which the '
                                     f'batched prefill does not produce (route this prompt to K3 passes, or load DFlash only)')

    def feat_window(self, p0, p1):
        """(feat_base, the first injected position) for a prefill of [p0, p1): the export holds [feat_base, + feat_rows); a
        prompt ending within feat_rows keeps feat_base 0 (positions as they are)"""
        from .attn import feat_rows
        fr = feat_rows(self.m.ntok)
        fb = 0 if p1 <= fr else int(p1) - fr
        return fb, max(int(p0), fb)

    def _limit_reason(self):
        return ('the engine\'s max_ctx' if getattr(self.m, 'qsa', False) else f'the batched prefill\'s dense attention equals the '
                f"model's QSA only below {LIMIT} and this engine has no QSA (route the rest to K3 passes)")

    def _validate_more(self, p0, p1, draft):
        """the feature export's window and the drafter's rows (after the driver's own refusals: caminus.prefill.driver)"""
        m = self.m
        if draft is not None and self.feat_window(p0, p1)[1] > p0:
            from .attn import feat_rows
            for d in _drafters(draft):
                if hasattr(d, 'inject') and not hasattr(d, 'skip_inject'):
                    raise PrefillRefused(f'prefill: positions {p0} .. {p1 - 1} exceed the feature export\'s {feat_rows(m.ntok)} rows and drafter '
                                         f'{type(d).__name__} cannot skip the positions before them (skip_inject)')
        dp = min([getattr(d, 'max_pos', None) for d in _drafters(draft) if getattr(d, 'max_pos', None)] or [None]) if draft else None
        if dp is not None and p1 > dp: raise PrefillRefused(f'prefill: positions to {p1 - 1} past the drafter\'s {dp} rows')

    def numerics(self):
        """the engine key's prefill part: the live glue modes (both cards' must agree) and ChunkStack's GEMM paths (.moe /
        .q6k may change between chunks)"""
        S = self.S
        g = sorted({(gl.wmma_out, gl.router_mode) for gl in S.Gl})
        if len(g) != 1: raise ValueError(f'Prefill: the cards\' glue modes differ: {g}')
        out = dict(wmma_out=g[0][0], router_logits=g[0][1], moe=S.moe, q6k=S.q6k)
        if not getattr(self.m, 'q8', True):
            # Earlier F16 engines still ran a Q8 prefill. Their parks cannot be reused after the matching F16 path.
            out.update(target_kv='f16', kv_prefill=1)
        return out

    # -- the prefill -----------------------------------------------------------------------------------------------------------
    # (run(): caminus.prefill.driver.PrefillDriver.run; these are its two-card parts)
    def _run_start(self, toks, bounds):
        self._fut = self.feed.host_pool.submit(self.feed.host_chunk, toks, *bounds[0][1:])

    def _run_chunk(self, toks, k, j, a, b, segs, bounds):
        S = self.S
        if a == segs[j][0]: S.feat_base = self.feat_window(*segs[j])[0]   # (each segment exports its own window)
        h = self._fut.result()                                           # (the next chunk's host feed overlaps this chunk)
        self._fut = self.feed.host_pool.submit(self.feed.host_chunk, toks, *bounds[k + 1][1:]) if k + 1 < len(bounds) else None
        S.run(h, m=self.micro or (1 if b - a < MICRO2_FROM else 2), mode='overlap', graphs=self.graphs)

    def _run_settle(self):
        S = self.S
        S.sync()
        S.check_routing('prefill'); self.feed.check2(); S.check_attn('prefill')

    def _run_drain(self):
        if self._fut is not None: self._fut.result()   # (the next chunk's host feed, in flight: drained, unused)

    def _sync_devices(self):
        for d in self.m.devs: torch.cuda.synchronize(d)

    def _cut_inject_from(self, s0, b):
        return min(b, max(s0, int(self.S.feat_base)))   # (the cut segment's export window was set from the segment end)

    def _handoff(self):
        """a run's end (or a stop): committed state; the next K3 pass commits nothing"""
        m = self.m
        m.fresh = True; m._during = None
        if m.ple_layers and m.ple._direct(): m.ple._drain()   # (a prefetch of an earlier pass: content-keyed rows, dropped)

    def inject(self, draft, p0, p1, g0=None):
        """the DFlash inject of positions [p0, p1) from the prefill's feature export, T rows per inject launch; returns the
        launches (0 for a drafter without inject). g0: the first position to inject (default: the window of a run ending at
        p1, feat_window; a cancelled run passes its export's own start); positions [p0, g0) are given up (skip_inject)"""
        ds = [d for d in _drafters(draft) if hasattr(d, 'inject')]
        if not ds: return 0
        m, T = self.m, self.m.ntok
        F = m.features
        n = 0
        g0 = self.feat_window(p0, p1)[1] if g0 is None else int(g0)
        if g0 > p0: draft.skip_inject(g0)     # (positions [p0, g0) never injected: outside every later window)
        for g in range(g0, p1, T):
            r = min(T, p1 - g)
            with torch.cuda.device(F.device):
                F[:, :r].copy_(self.S.features(g, g + r))
                if r < T: F[:, r:].zero_()
            draft.inject(g, F)
            if hasattr(draft, 'flush'): draft.flush()
            n += 1
        if hasattr(draft, 'sync'): draft.sync()
        return n

    def close(self):
        if self.graphs is not None: self.graphs.clear()
        self.feed.close()
