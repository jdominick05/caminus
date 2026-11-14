"""Draft only when it pays (M7, the lead's pre-M7 review item 3): a rolling DFlash acceptance per content class and depth
bucket, and plain decode when drafting is below break-even.

The cost of a pass is measured, not assumed: a DFlash pass (the target's T-token verify + the drafter's step) costs ms_d and
emits tok_d tokens on average; a plain pass (the same engine's T-token verify with filler drafts, the drafter's inject only,
so it stays in step for a later switch back) costs ms_p and emits tok_p (filler drafts are accepted now and then on repeats).
Drafting pays while ms_d / tok_d <= ms_p / tok_p. Both are EMAs per (content class, depth bucket); a bucket switches only
after MIN_SAMPLES DFlash passes, with HYSTERESIS on the comparison. In plain mode a DFlash pass every PROBE_EVERY passes keeps
the acceptance current. Before any plain pass was measured in a bucket, ms_p is estimated from the DFlash pass itself:
its wall time less the drafter step's device time (DFlashK3.step_ms(), sampled), the cheaper side of a plain pass (it also
runs an inject).

Plain decode on this engine is the T = 4 engine's own filler pass: the server has one engine (two FlashNextTP instances would
need two copies of the weights), and FlashNextTP's tapes are built for one ntok (the T = 1 plain decode of the 262K sweep,
15.26 ms/token at 261,632, is a separate engine; T = 1 tapes over the same states would need the lazy-commit records of a
T = 4 pass replayed by a T = 1 tape: an engine item, not wiring). A plain pass costs about the verify alone, so plain wins
only when DFlash's acceptance is near zero; the policy is measured and switches on its own when it does.

The choice never changes a token: a plain pass verifies filler drafts with the target, exactly as plain greedy decoding on
a T-token engine (spec == plain bitwise at T = 0; at T > 0 sample-and-match with Philox by position)."""
import bisect

DEPTH_EDGES = (8192, 32768, 65536, 131072, 196608)     # depth buckets (positions)
CLASSES = ('prose', 'code', 'shell', 'tool')
MIN_SAMPLES = 8
HYSTERESIS = 0.05
PROBE_EVERY = 16
ALPHA = 0.1


class _Ema:
    __slots__ = ('v', 'n')

    def __init__(self):
        self.v, self.n = None, 0

    def add(self, x, alpha=ALPHA):
        self.v = x if self.v is None else (1 - alpha) * self.v + alpha * x
        self.n += 1


class Bucket:
    __slots__ = ('ms_d', 'tok_d', 'ms_p', 'tok_p', 'ms_p_est', 'plain', 'since', 'switches')

    def __init__(self):
        self.ms_d, self.tok_d, self.ms_p, self.tok_p, self.ms_p_est = _Ema(), _Ema(), _Ema(), _Ema(), _Ema()
        self.plain, self.since, self.switches = False, 0, 0

    def plain_cost(self):
        """ms per token of a plain pass (measured, else the estimate from DFlash passes), or None"""
        if self.ms_p.n:
            return self.ms_p.v / max(1.0, self.tok_p.v)
        return self.ms_p_est.v

    def draft_cost(self):
        return None if self.tok_d.v is None else self.ms_d.v / max(1e-6, self.tok_d.v)


class DraftPolicy:
    """decide(cls, pos) -> 'draft' | 'plain'; record(mode, cls, pos, ms, emitted, step_ms=None) after each decode pass.
    enabled=False: always 'draft'. force: a callable(pass_no) -> 'draft' | 'plain' | None (tests: forced schedules).
    probe_plain: a bucket with min_samples drafted passes and no plain cost yet runs one plain pass to measure it (the
    host-accept drafters, MTP and the 27B DFlash2: their drafted passes carry no drafter step time, so no estimate ever
    formed and a bucket where drafting loses never switched: ThinkingCap MTP at 40K served 0.84x of plain)."""

    def __init__(self, enabled=True, min_samples=MIN_SAMPLES, hysteresis=HYSTERESIS, probe_every=PROBE_EVERY, force=None,
                 probe_plain=False):
        self.enabled, self.min_samples, self.hysteresis, self.probe_every, self.force = enabled, min_samples, hysteresis, probe_every, force
        self.probe_plain = bool(probe_plain)
        self.b = {}
        self.n = 0
        self.counts = {'draft': 0, 'plain': 0, 'probe': 0, 'plain_probe': 0}

    @staticmethod
    def depth(pos):
        return bisect.bisect_right(DEPTH_EDGES, int(pos))

    def bucket(self, cls, pos):
        k = (cls if cls in CLASSES else 'prose', self.depth(pos))
        b = self.b.get(k)
        if b is None:
            b = self.b[k] = Bucket()
        return b

    def decide(self, cls, pos):
        self.n += 1
        if self.force is not None:
            f = self.force(self.n)
            if f is not None:
                self.counts[f] += 1
                return f
        if not self.enabled:
            self.counts['draft'] += 1
            return 'draft'
        b = self.bucket(cls, pos)
        if b.plain:
            b.since += 1
            if b.since % self.probe_every == 0:
                self.counts['probe'] += 1
                return 'draft'
            self.counts['plain'] += 1
            return 'plain'
        if self.probe_plain and b.ms_d.n >= self.min_samples and b.plain_cost() is None:
            self.counts['plain_probe'] += 1   # (one plain pass: the bucket's plain cost, for record()'s comparison)
            return 'plain'
        self.counts['draft'] += 1
        return 'draft'

    def record(self, mode, cls, pos, ms, emitted, step_ms=None):
        """one decode pass: its wall ms and emitted tokens; step_ms: the drafter step's device ms when it was timed"""
        if not self.enabled:
            return
        b = self.bucket(cls, pos)
        if mode == 'draft':
            b.ms_d.add(ms)
            b.tok_d.add(float(emitted))
            if step_ms is not None:
                b.ms_p_est.add(max(0.0, ms - step_ms))
        else:
            b.ms_p.add(ms)
            b.tok_p.add(float(emitted))
        if b.ms_d.n < self.min_samples:
            return
        cd, cp = b.draft_cost(), b.plain_cost()
        if cd is None or cp is None:
            return
        if b.plain and cd < cp * (1 - self.hysteresis):
            b.plain, b.since, b.switches = False, 0, b.switches + 1
        elif not b.plain and cd > cp * (1 + self.hysteresis):
            b.plain, b.since, b.switches = True, 0, b.switches + 1

    def view(self):
        """per bucket: (class, depth bucket) -> dict(mode, draft / plain ms per token, tokens per DFlash pass, passes, switches)"""
        out = {}
        for (c, d), b in sorted(self.b.items()):
            out[f'{c}/{d}'] = dict(mode='plain' if b.plain else 'draft', draft_ms_per_token=b.draft_cost(), plain_ms_per_token=b.plain_cost(),
                                   tokens_per_draft_pass=b.tok_d.v, draft_passes=b.ms_d.n, plain_passes=b.ms_p.n, switches=b.switches)
        return out
