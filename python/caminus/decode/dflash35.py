"""The dense qwen35 line's DFlash 2 drafter (engine/q35_dflash.py Dflash35) beside its one-card target (engine/q35_dense.py
Dense35Own), wired the way the serve session and the prompt cache consume a host-accept drafter (caminus.serve.k3_session,
caminus.cache.prompt; the same seat as decode/mtp35.py's MTP drafter: the session's draft passes are the target's T = 4
verify passes, engine/q35_dense_verify.py, and the host compares their greedy rows with the drafts).

The drafter's state is its ring: per drafter layer, the K / V of the target's features at every injected position
(row p & (RING - 1)). The host keeps the one contiguous interval [lo, hi) of positions whose rows hold injected features;
a draft at position n0 (the block [last, MASK x 3] at n0 .. n0 + 3) reads the window max(0, n0 + 3 - W + 1) .. n0 - 1 of
them, so a round runs only when the interval covers that window (hi >= n0 and lo <= its start), and returns filler
otherwise (the verify decides every token: only acceptance changes). Rows at n0 and past it are the block's own (written
by the draft before its attention reads them), so an inject past n0 (a verify pass's rejected rows) is clamped there on
the next round. Every pass that runs on the target feeds the drafter: the verify passes (their feature export), the one-
token passes (Dense35Own(feature_layers=)) and the batched prefill (Q35OwnPrefill(feature_layers=), per chunk).

The cache's window protocol (export / import of the ring rows [lo, hi) of every drafter layer into a checkpoint's win
region, W + T rows at most) and its bookkeeping (injected_to, skip_inject, give_up) are this interval's."""
import contextlib

import torch

from ..engine import k3
from ..engine.q35_dflash import RING, T, Dflash35


class Dflash35Drafter:
    """Dflash35 with the serve session's and the prompt cache's drafter interface (a host-accept drafter: kind dflash35)"""
    kind = 'dflash35'
    spec_type = 'draft-dflash'
    device_accept = False

    def __init__(self, d):
        if not isinstance(d, Dflash35): raise ValueError(f'Dflash35Drafter: {type(d).__name__} is not a Dflash35 drafter')
        self.d = d
        self.df = self               # (the cache reads the drafter's df: Layout, engine_key, the window protocol)
        self.tgt = d.eng
        self.queued = None           # (nothing is ever queued: every inject runs at once)
        self.lo = self.hi = 0
        self.filler = False
        self.stats = dict(rounds=0, filler=0, injected=0)

    # the engine-level attributes the cache's Layout / engine_key read
    path = property(lambda s: s.d.path)
    window = property(lambda s: s.d.window)
    ring = property(lambda s: RING)
    nl = property(lambda s: s.d.nl)
    nkv = property(lambda s: s.d.nkv)
    hd = property(lambda s: s.d.hd)
    row_bytes = property(lambda s: s.d.row_bytes)
    dx = property(lambda s: s.d.dx)
    T = property(lambda s: T)
    mib = property(lambda s: s.d.mib)
    token = property(lambda s: int(s.d.token))   # (the launch epoch the cache's live state reads: every inject / draft)

    # -- feeding ---------------------------------------------------------------------------------------------------------
    def inject(self, i0, feats):
        """the target's features [n_layers, n, D] of positions i0 .. i0 + n - 1 (a pass's export, or a prefill chunk's).
        A start past hi is a gap: the interval restarts at i0 (the rows below it are stale for the rounds that would
        read them); a start below hi rewrites from i0 (a rejected tail, or a pass run again)"""
        if feats is None: return
        i0, n = int(i0), int(feats.shape[1])
        if n <= 0: return
        self.d.inject(i0, feats)
        if i0 > self.hi or i0 < self.lo: self.lo = i0
        self.hi = i0 + n
        self.stats['injected'] += n

    def take_pass(self, i0, n, toks, src):
        """the session's / the cache's hook after a T-wide verify pass on src (Dense35Verify): its exported rows"""
        self.inject(i0, src.features[:, :int(n)])

    def fill_chunk(self, h, toks, p0):
        """(the MTP drafter's prefill hook: this drafter takes the chunk's features through inject_chunk)"""

    def inject_chunk(self, src, a, n):
        """the batched prefill's per-chunk hook (positions a .. a + n - 1): src the prefill itself (Q35OwnPrefill: its
        features through the drafter's batched inject, Dflash35.inject_rows, when the drafter was built for its chunk), or
        the chunk's feature rows [n_layers, n, D] (the 4-row inject tape)"""
        a, n = int(a), int(n)
        if n <= 0: return
        if isinstance(src, torch.Tensor): self.inject(a, src[:, :n]); return
        if self.d.chunk >= n: self.d.inject_rows(src, a, n)
        else: self.d.inject(a, src.features[:, :n])
        if a > self.hi or a < self.lo: self.lo = a
        self.hi = a + n
        self.stats['injected'] += n

    def prefill(self, toks, n_h):
        """every pass and prefill chunk injected its features already: nothing here"""

    # -- the cache's drafter bookkeeping (caminus.cache.prompt._follow / _skip_gap) ------------------------------------
    def injected_to(self):
        return int(self.hi)

    def skip_inject(self, p):
        """positions below p the drafter never saw: the interval restarts at p"""
        p = int(p)
        if p > self.hi: self.lo = self.hi = p

    def followed(self):
        return True

    def drop_stale(self):
        return None

    def drop_pending(self):
        """(nothing is ever pending)"""

    def give_up(self, lo, hi):
        self.hi = min(self.hi, int(lo)); self.lo = min(self.lo, self.hi)

    def flush(self):
        """nothing queued"""

    def reset(self):
        self.lo = self.hi = 0
        self.filler = False

    def snapshot(self):
        return dict(lo=int(self.lo), hi=int(self.hi))

    def restore(self, s):
        self.lo, self.hi = int(s['lo']), int(s['hi'])
        self.filler = False

    # -- the session's drafter call --------------------------------------------------------------------------------------
    def _covered(self, n0):
        return self.hi >= n0 and self.lo <= max(0, n0 + T - self.d.window)

    def __call__(self, toks, k):
        """the k drafts for the coming pass (its last fed token toks[-1] at position len(toks) - 1), or filler (the last token
        k times) when the injected interval does not cover the round's window"""
        toks = [int(t) for t in toks]
        n0, k = len(toks) - 1, int(k)
        self.filler = False
        if k <= 0: return []
        self.hi = min(self.hi, n0) if self.hi > n0 else self.hi   # (rows at n0 and past: the block's own)
        self.lo = min(self.lo, self.hi)
        if n0 < 0 or not self._covered(n0) or n0 + T > self.d.max_pos:
            self.filler = True; self.stats['filler'] += 1
            return [toks[-1]] * k
        self.stats['rounds'] += 1
        return self.d.draft(toks[-1], n0)[:k]

    # -- the cache's window protocol -------------------------------------------------------------------------------------
    def _on(self):
        """the drafter's card as the current device (a CPU stand-in's rings: nothing)"""
        return torch.cuda.device(self.d.dx) if self.d.L[0]['ring'].is_cuda else contextlib.nullcontext()

    def wrows(self):
        return int(self.d.window) + T

    def window_state(self, lo):
        """the rows readable from lo: dict(lo, hi), at most W + T of them (the layout's win region)"""
        hi = int(self.hi)
        lo = min(max(int(lo), int(self.lo), hi - self.wrows()), hi)
        return dict(lo=lo, hi=hi)

    def _rows(self, lo, hi):
        return torch.arange(lo, hi, device=f'cuda:{self.d.dx}') & (RING - 1)

    def export_window(self, win, dm):
        """ring rows of positions dm['lo'] .. dm['hi'] - 1 of every drafter layer into the slot's win region
        (win: [nl, wrows, row_bytes] uint8)"""
        lo, hi = int(dm['lo']), int(dm['hi'])
        if hi <= lo: return
        if hi - lo > win.shape[1]: raise ValueError(f'Dflash35Drafter export_window: {hi - lo} rows past the slot window {win.shape[1]}')
        with self._on():
            idx = self._rows(lo, hi)
            for il, L in enumerate(self.d.L):
                src = L['ring'].view(torch.uint8).view(RING, self.d.row_bytes)
                torch.index_select(src, 0, idx, out=win[il, :hi - lo])

    def import_window(self, win, dm):
        if dm is None: return
        lo, hi = int(dm['lo']), int(dm['hi'])
        if hi <= lo: return
        with self._on():
            idx = self._rows(lo, hi)
            for il, L in enumerate(self.d.L):
                L['ring'].view(torch.uint8).view(RING, self.d.row_bytes).index_copy_(0, idx, win[il, :hi - lo])

    def set_window_state(self, dm):
        """the restored interval: rows dm['lo'] .. dm['hi'] - 1 hold their saved features"""
        if dm is None: self.lo = self.hi = 0
        else: self.lo, self.hi = int(dm['lo']), int(dm['hi'])
