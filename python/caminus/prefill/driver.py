"""The batched prefill drivers' common shell, and the one routing rule.

A driver is what the prompt cache (caminus.cache.prompt), the session (caminus.serve.k3_session), the speculative loop
(caminus.decode.spec.prefill_span) and the long fills (caminus.decode.longctx.fill) call for a prompt:

  route(n) / span(p0, end)   the routing rule (batched_span, one function): positions [p0, min(end, limit)) go through the
                             batched prefill when that span reaches the crossover T* (shorter spans run as K3 passes,
                             bitwise the one-token decode); crossover None: a driver with no batched prefill (KPassOnly)
  check_tokens / check_drafter / validate
                             the refusals (PrefillRefused) before anything is reset or written
  run(...)                   the batched prefill of [i0 - commit, p1): segments at the cache's stops (each a chunk
                             boundary with its own hand-off, the drafter's inject and at_stop), chunks of up to `chunk`
                             tokens, a client's cancel at a chunk boundary, the engine poisoned when a failure follows
                             written state; returns dict(i0, commit=0, the run's numbers)
  fill(toks, n, draft)       reset and prefill [0, n) (longctx.fill's batched form)
  numerics()                 the engine key's prefill part (caminus.cache.prompt.numerics): the glue modes and GEMM paths
  limit, chunk, crossover, n_vocab, last, close()

The drivers fill in a chunk (_run_chunk), the cards' sync and flag checks (_run_settle), the hand-off (_handoff), the
drafter's inject (inject, _cut_inject_from) and their own refusals (_validate_more, _limit_reason):
caminus.prefill.coder.Prefill (the Coder on both cards), caminus.prefill.q35.Prefill35 (one card) and
caminus.prefill.kpass.KPassOnly (no batched prefill: every prompt as K3 passes). Before 11-07 each kept its own copy of
validate / check_tokens / run's shell, the prompt cache's test stand-in a third, the routing rule was written four ways
(plan_request compared the crossover itself, so KPassOnly carried crossover = 1 << 62) and the numerics key was read from
a ChunkStack's attributes that the other drivers faked."""
import time

import numpy as np
from ..errors import NumericalFailure, NonFiniteLogits

from ..engine import k3


class PrefillRefused(k3.ContextBoundsError):
    """a prompt the batched prefill refuses before anything is reset or written (a token out of range, a span past the
    engine's rows or QSA_EXACT, a drafter that cannot take a prefill)"""


def drafters(draft):
    """the drafters behind `draft` (PolicyDrafter: its members)"""
    if draft is None: return []
    d = getattr(draft, 'd', None)
    return list(d.values()) if isinstance(d, dict) and hasattr(draft, 'select') else [draft]


def batched_span(p0, end, crossover, limit=None):
    """the one routing rule: (p0, P) with P = min(end, limit) when the batched prefill takes positions [p0, P): P > p0 and
    the span reaches the crossover T* (shorter spans run as K3 passes). crossover None: no batched prefill, never. None
    otherwise. caminus.cache.prompt.plan_request (pure: tools/eval/cache_replay.py runs it on numbers), Driver.route /
    span, spec.prefill_span and longctx.fill all decide by it"""
    P = int(end) if limit is None else min(int(end), int(limit))
    if crossover is None or P <= int(p0) or P - int(p0) < int(crossover): return None
    return int(p0), P


class PrefillDriver:
    crossover = None     # T*: spans shorter than it run as K3 passes; None: no batched prefill
    chunk = 0            # the largest chunk (tokens) the driver's buffers take
    last = None          # the last run's numbers
    n_vocab = None       # the token embedding's rows (None: no token check)

    @property
    def limit(self):
        """the positions [0, limit) the batched prefill may write"""
        return int(self.m.max_ctx)

    # -- routing -------------------------------------------------------------------------------------------------------
    def route(self, n):
        """True: n new positions from 0 go through the batched prefill (batched_span); False: K3 passes"""
        return batched_span(0, int(n), self.crossover) is not None

    def span(self, p0, end):
        """(p0, P): the batched part of positions [p0, min(end, limit)), or None (batched_span)"""
        return batched_span(p0, end, self.crossover, self.limit)

    # -- refusals ------------------------------------------------------------------------------------------------------
    def _check_terminal(self):
        if getattr(self.m, 'broken', None):
            cls = getattr(self.m, '_broken_error', NonFiniteLogits)
            raise cls(f'prefill: terminal engine state: {self.m.broken}')

    def check_drafter(self, draft):
        """refuse (PrefillRefused) a drafter this prefill cannot feed (the default: any drafter is fine)"""

    def check_tokens(self, toks, draft=None):
        """refuse (PrefillRefused) a prompt with any token id outside the token embedding's rows (and every drafter's): the
        whole prompt, before anything runs (also for a prompt routed to K3 passes, which would check it pass by pass)"""
        vocab = ([self.n_vocab] if self.n_vocab is not None else []) + [
            int(getattr(getattr(d, 'df', d), 'emb_vocab')) for d in drafters(draft) if hasattr(getattr(d, 'df', d), 'emb_vocab')]
        if not vocab: return
        t = np.asarray(toks, dtype=np.int64)
        if len(t) and (int(t.min()) < 0 or int(t.max()) >= min(vocab)):
            bad = int(np.flatnonzero((t < 0) | (t >= min(vocab)))[0])
            raise PrefillRefused(f'prefill: token {int(t[bad])} at position {bad} outside [0, {min(vocab)}) (refused before any write)')

    def _limit_reason(self):
        return 'the engine\'s max_ctx'

    def _validate_more(self, p0, p1, draft):
        """a driver's own refusals after the common ones (the Coder's: the feature export and the drafter's rows)"""

    def validate(self, toks, p0, p1, draft=None, n_total=None):
        """refuse (PrefillRefused) before anything runs: the batched positions [p0, p1), every token of toks[:n_total]
        (default: all of toks) in range, the engine's state, the drafter"""
        m = self.m
        self._check_terminal()
        if m.poisoned: raise PrefillRefused(f'prefill: the engine is poisoned ({m.poisoned}): reset it first')
        n_total = len(toks) if n_total is None else int(n_total)
        p0, p1 = int(p0), int(p1)
        if not 0 <= p0 < p1 <= n_total <= len(toks):
            raise PrefillRefused(f'prefill: positions {p0} .. {p1 - 1} of a prompt of {n_total} tokens ({len(toks)} given)')
        try: k3.check_span(p0, p1 - p0, m.max_ctx, what='prefill (the KV rows)')
        except k3.ContextBoundsError as e: raise PrefillRefused(str(e)) from None
        if p1 > self.limit:
            raise PrefillRefused(f'prefill: positions {p0} .. {p1 - 1} reach {self.limit}: {self._limit_reason()}')
        self.check_tokens(toks[:n_total], draft)
        self.check_drafter(draft)
        self._validate_more(p0, p1, draft)

    # -- the run -------------------------------------------------------------------------------------------------------
    def _run_start(self, toks, bounds):
        """before the first chunk (the Coder's: the first chunk's host feed submitted)"""

    def _run_chunk(self, toks, k, j, a, b, segs, bounds):
        """chunk k: positions [a, b) of segment j (queued; nothing synchronized)"""
        raise NotImplementedError

    def _run_settle(self):
        """the cards synchronized and every flag word checked (a stop, the run's end)"""

    def _run_drain(self):
        """after a cancel: what the run had in flight beyond the cut (the Coder's: the next chunk's host feed)"""

    def _sync_devices(self):
        """a failure: every card synchronized before the engine is poisoned"""

    def _run_failure_check(self):
        """After a failure drain, report terminal device flags before ordinary recovery."""

    def _handoff(self):
        """a run's end (or a stop): committed state; the next K3 pass commits nothing"""
        self.m.fresh = True

    def inject(self, draft, p0, p1, g0=None):
        """the drafter's inject of positions [p0, p1) (g0: the first one; positions [p0, g0) given up); returns the launches"""
        return 0

    def _cut_inject_from(self, s0, b):
        """a cancelled segment [s0, ...) cut at b: the first position its exported features hold below b"""
        return b

    def run(self, toks, i0, commit, p1, draft=None, validated=False, chunk=None, stops=(), at_stop=None, cancel=None):
        """the batched prefill of positions [i0 - commit, p1) of toks (the state left by K3 passes: positions < i0 run, the
        last pass's first `commit` rows uncommitted; (0, 0) after a reset), then the drafter's inject of those positions
        (draft: the drafter, or None). chunk: tokens per chunk (<= self.chunk; G2b's test of chunk geometry). Returns the
        state dict(i0=p1, commit=0) with the run's numbers.
        stops (the prompt cache's checkpoints, caminus.cache.prompt; M6): positions strictly inside (i0 - commit, p1) where
        the run is split into segments: a chunk boundary at each stop (G2b: chunk geometry changes no bit), and after each
        segment the hand-off of a run ending there (the cards synchronized, every flag word checked, m.fresh, the drafter's
        inject of the segment's own feature window), then at_stop(position, dict(i0=position, commit=0)) with the engine
        in exactly the state a separate run ending there leaves.
        cancel (M7: a client's cancel; a callable -> bool, or an object with is_set()): checked after each chunk but the run's
        last; when set the run ends at that chunk boundary b with the hand-off of a run ending there and returns dict(i0=b,
        commit=0, cancelled=True): the target's state is bitwise a run of [i0 - commit, b) (a chunk boundary: G2b). The
        drafter's inject covers the positions of the segment's feature export below b (_cut_inject_from); positions below it
        are given up (skip_inject: never wrong, the drafter drafts filler while a window reads them). A cancel object's
        progress(position), when it has one, is called after each chunk (the server's live prompt progress)"""
        m, name = self.m, type(self).__name__
        self._check_terminal()   # validated=True also refuses an engine that failed since validation
        p0, p1 = int(i0) - int(commit), int(p1)
        C = self.chunk if chunk is None else int(chunk)
        if not 1 <= C <= self.chunk: raise ValueError(f'{name}.run: chunk {chunk} (1 .. {self.chunk})')
        stops = sorted(set(int(s) for s in stops))
        if any(not p0 < s < p1 for s in stops): raise ValueError(f'{name}.run: stops {stops} outside ({p0}, {p1})')
        if stops and at_stop is None: raise ValueError(f'{name}.run: stops without at_stop')
        if not validated: self.validate(toks, p0, p1, draft)
        else: self.check_drafter(draft)
        t0 = time.perf_counter()
        edges = [p0] + stops + [p1]
        segs = list(zip(edges[:-1], edges[1:]))
        bounds = [(j, a, min(a + C, b)) for j, (s0, b) in enumerate(segs) for a in range(s0, b, C)]
        self._run_start(toks, bounds)
        inj = 0; inj_s = 0.0; stop_s = 0.0; cut = None; k = -1
        is_set = (cancel.is_set if hasattr(cancel, 'is_set') else cancel) if cancel is not None else None
        prog = getattr(cancel, 'progress', None)
        try:
            for k, (j, a, b) in enumerate(bounds):
                self._run_chunk(toks, k, j, a, b, segs, bounds)
                if b == segs[j][1] and j + 1 < len(segs):          # a stop: the hand-off of a run ending at b
                    self._run_settle()
                    self._handoff()
                    ti = time.perf_counter()
                    inj += self.inject(draft, *segs[j]) if draft is not None else 0
                    tj = time.perf_counter(); inj_s += tj - ti
                    at_stop(b, dict(i0=b, commit=0))
                    stop_s += time.perf_counter() - tj
                if prog is not None: prog(b)                     # (the chunk's launches are queued: progress, not completion)
                if is_set is not None and k + 1 < len(bounds) and is_set():   # a cancel: the run ends at this chunk boundary
                    cut = (j, b)
                    break
            if cut is not None: self._run_drain()
            self._run_settle()
        except BaseException as e:
            self._sync_devices()
            if isinstance(e, NumericalFailure):
                raise
            self._run_failure_check()
            self._check_terminal()
            if not m.poisoned: m.poisoned = f'the batched prefill of positions {p0} .. {p1 - 1} failed after writing state: {type(e).__name__}: {e}'
            raise
        t1 = time.perf_counter()
        # the hand-off: committed state at p1 (or at a cancel's chunk boundary); the next K3 pass commits nothing
        self._handoff()
        if cut is None:
            inj += self.inject(draft, *segs[-1]) if draft is not None else 0
        elif draft is not None:   # the cut segment's exported features below the cut
            s0, b = segs[cut[0]][0], cut[1]
            inj += self.inject(draft, s0, b, g0=self._cut_inject_from(s0, b))
        t2 = time.perf_counter()
        end = p1 if cut is None else cut[1]
        self.last = dict(p0=p0, p1=end, chunks=len(bounds) if cut is None else k + 1, prefill_ms=(t1 - t0 - inj_s - stop_s) * 1e3,
                         inject_ms=(t2 - t1 + inj_s) * 1e3, inject_launches=inj, stops=[x for x in stops if x < end], stop_ms=stop_s * 1e3,
                         cancelled=cut is not None)
        return dict(i0=end, commit=0, **self.last)

    def fill(self, toks, n=None, draft=None):
        """longctx.fill's batched form: reset m (and draft), prefill positions [0, n) (n = len(toks)); the state (i0 = n,
        commit = 0); the K3 passes past limit are the caller's (longctx.fill)"""
        n = len(toks) if n is None else int(n)
        self.validate(toks, 0, n, draft, n_total=n)
        self.m.reset()
        if draft is not None and hasattr(draft, 'reset'): draft.reset()
        return self.run(toks, 0, 0, n, draft, validated=True)

    def numerics(self):
        """the engine key's prefill part (caminus.cache.prompt.numerics): dict(wmma_out, router_logits, moe, q6k) as this
        driver runs now"""
        raise NotImplementedError

    def close(self):
        pass
