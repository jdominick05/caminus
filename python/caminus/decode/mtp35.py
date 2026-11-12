"""The dense qwen35 line's MTP drafter (engine/q35_mtp.py Mtp35) beside its one-card trunk (engine/q35_dense.py
Dense35Own), wired the way the serve session and the prompt cache consume a drafter (caminus.serve.k3_session,
caminus.cache.prompt): the cache-facing window protocol lives on Mtp35 itself; this adapter adds the trunk-h
bookkeeping and the session's drafter call.

State and the round protocol. The block's KV rows are by position: row p pairs (the trunk's h of position p - 1,
the token at position p). Mtp35.fill / draft write those rows from the trunk's h; `filled` is the high-water mark
(rows 1 .. filled - 1 hold pairs; a round refuses a gap past it). The trunk's h per position is NOT kept per
position (that grows with max_ctx): only a ring of the last HR positions (the last passes), because a round's catch
tape rewrites its own rows from the current h and reads the older rows from the block KV. After every trunk pass
the session / the cache's pass loop calls take_h(i0, h, toks): the pass's h rows (positions i0 .. i0 + n - 1) go into
the ring, and the block rows i0 + 1 .. i0 + n are filled from them with the tokens at those positions (the pass's
seq, plus the bonus token of its last row: a row whose token is rejected is rewritten by the pass that verifies that
position again).

A draft round for the coming pass (at i0 = len(log) = len(toks) - 1, the last token of toks fed by it) is
Mtp35.draft with p0 = len(toks) - T: catch rows p0 .. len(toks) - 1 from (h of p0 - 1 .., toks[p0 ..]), then the
chained steps; the returned drafts are the tokens at positions len(toks) .., exactly the pass's draft slots. The
h rows it needs (positions len(toks) - T - 1 .. len(toks) - 2) are in the ring whenever the last two passes went
through take_h (HR = 2 * MAX_T covers them); after a restore or a bypassed pass they may not be, and the round falls
back to filler drafts (the verify decides every token, so the output never changes; only acceptance sags until the
ring is back in step - 'doubt drops to plain until back in step').

Greedy only: the drafts are the block's argmax. The cache checkpoint protocol is Mtp35's (window_state /
export_window / import_window / set_window_state: the block KV window [lo, filled) into the slot's win region);
snapshot / restore carry the pointers (filled, h_valid), the KV rows themselves go through the window export."""
import torch

from ..engine import k3
from ..engine.q35_mtp import MAX_T, Mtp35

HR = 2 * MAX_T          # the h ring's rows: the last two passes' h (T rows each) cover any round's needs


class Mtp35Drafter:
    """Mtp35 with the serve session's and the prompt cache's drafter interface (branch on kind == 'mtp')"""
    kind = 'mtp'
    spec_type = 'draft-mtp'
    device_accept = False          # (the accept is the host's: the verify pass's argmax rows vs the drafts)

    def __init__(self, mtp):
        if not isinstance(mtp, Mtp35) and getattr(mtp, 'kind', None) != 'mtp':
            raise ValueError(f'Mtp35Drafter: {type(mtp).__name__} is not an Mtp35 drafter')
        self.m = mtp
        self.df = self              # the cache reads the drafter's df (Layout, engine_key, LiveState.counters, the
                                    # window protocol): the engine-level attributes come through __getattr__
        self.max_pos = int(mtp.eng.max_ctx)
        self.H = None               # the h ring [HR, D] fp32 on the trunk's card, row p % HR = h of position p
        self.h_valid = 0            # h of positions 0 .. h_valid - 1 are in the ring (when within HR of it)
        self.synced = 0             # the position the drafter is in step with (the last take_h's end)
        self.filler = False         # the last __call__ gave filler drafts (the session's drafts_real=False)

    def __getattr__(self, name):
        """the cache-facing attributes of the engine-level drafter (path, dx, tgt, nl, nkv, hd, T, window, ring,
        row_bytes, queued, token, filled, cache, the window protocol): read through to the Mtp35"""
        if name == 'm': raise AttributeError(name)
        return getattr(self.m, name)

    # -- the trunk's h -----------------------------------------------------------------------------------------------------
    def take_h(self, i0, h, toks):
        """after the trunk's pass at positions i0 .. i0 + n - 1 (n = len(toks)): its h rows (fp32 [n, D] on the
        trunk's card, a view of the pass engine's buf['h']) into the ring, and the block rows i0 + 1 .. i0 + n filled
        from them with toks (the tokens at those positions: the pass's seq rows, then the bonus). Nothing when the
        fill would cross a gap (p0 > filled): the round falls back to filler until a restore hands the window over"""
        i0, n = int(i0), len(toks)
        if n <= 0: return
        D = self.m.eng.D
        if h.dtype != torch.float32 or h.numel() < n * D:
            raise ValueError(f'Mtp35Drafter.take_h: h must be fp32 [{n}, {D}] (got {h.dtype} {tuple(h.shape)})')
        if self.H is None:
            self.H = torch.zeros(HR, D, dtype=torch.float32, device=h.device)
        k3.check_span(i0, n, self.max_pos + 1, what='Mtp35Drafter take_h')
        hv = h.reshape(-1)[:n * D].view(n, D)
        for r in range(n):
            self.H[(i0 + r) % HR].copy_(hv[r])
        self.h_valid = max(self.h_valid, i0 + n)
        self.synced = max(self.synced, i0 + n)
        p0 = i0 + 1
        if p0 <= self.m.filled and len(toks) == n:
            self.m.fill(self._h(i0, n), [int(t) for t in toks], p0)

    def _h(self, p0, n):
        """the ring's h rows of positions p0 .. p0 + n - 1 as fp32 [n, D], or None when they left the ring"""
        if self.H is None or p0 < 0 or p0 + n > self.h_valid or p0 < self.h_valid - HR:
            return None
        idx = torch.arange(p0, p0 + n, device=self.H.device) % HR
        return self.H[idx]

    def inject(self, i0, feats):
        """the cache's K3-pass loop calls this per pass; for the MTP drafter the h came with the pass (take_h), so
        there is nothing to inject"""

    def fill_chunk(self, h, toks, p0):
        """the batched prefill's per-chunk fill: block rows p0 .. p0 + len(toks) - 1 from the prefill's exported h
        (prefill numerics, not the decode tape's: the served gate is acceptance + token identity). A gap (p0 past
        filled) is skipped: the draft rounds fall back to filler until a restore hands the window over"""
        p0 = int(p0)
        if p0 < 1 or p0 > self.m.filled: return
        self.m.fill(h, [int(t) for t in toks], p0)

    def prefill(self, toks, n_h):
        """the batched prefill fills the block rows per chunk itself (its h export); the K3 tail passes filled theirs
        through take_h: nothing here"""

    def skip_inject(self, p):
        """the cache's gap guard: positions below p the drafter never saw. For the MTP drafter there is no pending
        inject to refuse - the round's own guards (p0 <= filled, the h rows in the ring) already fall back to filler
        over an uncovered range - so this only records the mark"""
        self.given_up = max(getattr(self, 'given_up', 0), int(p))

    # -- the cache's drafter bookkeeping (caminus.cache.prompt._follow / _skip_gap) ------------------------------------
    def injected_to(self):
        return int(self.m.filled)

    def followed(self):
        return self.synced >= self.m.filled - 1

    def drop_stale(self):
        return None

    def give_up(self, lo, hi):
        self.m.filled = min(int(self.m.filled), int(lo))
        self.h_valid = min(self.h_valid, int(lo))
        self.synced = min(self.synced, int(lo))

    def flush(self):
        """nothing queued (no pending inject)"""

    def reset(self):
        self.m.reset()
        self.h_valid = 0
        self.synced = 0
        self.filler = False

    def snapshot(self):
        return dict(filled=int(self.m.filled), h_valid=int(self.h_valid))

    def restore(self, s):
        self.m.filled = int(s['filled'])
        self.h_valid = int(s['h_valid'])
        self.synced = self.h_valid
        self.filler = False

    # -- the session's drafter call --------------------------------------------------------------------------------------
    def __call__(self, toks, k):
        """the k drafts for the coming pass (at i0 = len(toks) - 1): the round at p0 = len(toks) - T, or filler
        (the last token k times) when the round cannot run (a gap past filled, the h rows out of the ring, a too
        short sequence): the verify decides every token either way"""
        toks = [int(t) for t in toks]
        L, k = len(toks), int(k)
        T = MAX_T
        self.filler = False
        p0 = L - T
        h = self._h(p0 - 1, T) if p0 >= 1 else None
        if k <= 0 or p0 < 1 or h is None or p0 > self.m.filled:
            self.filler = True
            return [toks[-1]] * k if k > 0 else []
        try:
            return self.m.draft(h, toks[p0:], p0, k=min(k, self.m.k))
        except (ValueError, k3.ContextBoundsError):
            # a gap / span refusal: drop to filler for this round (the output never changes; the next take_h
            # refills from the trunk's h)
            self.filler = True
            return [toks[-1]] * k

    # -- the cache's window protocol (on Mtp35) --------------------------------------------------------------------------
    def window_state(self, lo):
        return self.m.window_state(lo)

    def export_window(self, win, dm):
        self.m.export_window(win, dm)

    def import_window(self, win, dm):
        self.m.import_window(win, dm)

    def set_window_state(self, dm):
        self.m.set_window_state(dm)
        if dm is not None:
            self.synced = max(self.synced, int(dm['hi']))
