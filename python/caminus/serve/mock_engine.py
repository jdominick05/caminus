"""MockEngine: a CPU stand-in for the K3 engine (FlashNextTP) with the same pass interface, for the server tests.

Interface (mirrors caminus.engine.flashnext_tp.FlashNextTP and caminus.decode.spec.generate):
    ntok                      tokens per pass (T)
    reset()                   zeroed state
    verify(seq, i0, commit)   one pass over positions i0 .. len(seq) - 1 (at most T). `commit` first makes the
                              previous pass's first `commit` tokens permanent (the lazy commit). Returns float32
                              logits [len(seq) - i0, V]; row j predicts the token after seq[i0 + j].
    flush(commit)             apply a final lazy commit without a new pass (end of a completion)
    state                     the committed token tuple (stands for the recurrent state)
It models the recurrent state strictly: a pass whose i0 or prefix disagrees with the committed state raises, so a
wrong lazy commit cannot go unnoticed. Logits are a deterministic function of (the last 4 tokens, the position),
peaked on a `mode` token so that drafts of the mode are often accepted at temperature > 0."""
import hashlib
from dataclasses import dataclass, field

import numpy as np


class EngineStateError(RuntimeError):
    pass


def _h(*xs):
    return int.from_bytes(hashlib.blake2b(repr(xs).encode(), digest_size=8).digest(), 'little')


class MockEngine:
    def __init__(self, ntok=1, vocab=512, eos_id=None, eos_every=0, seed=0, peak=(1.0, 9.0)):
        """vocab: V; eos_id / eos_every: the mode is eos_id on about 1 in eos_every positions; peak: the range of
        the mode's logit boost (a uniform pick per position; low boosts give rejections, high ones acceptances)"""
        self.ntok, self.V, self.eos_id, self.eos_every, self.seed, self.peak = ntok, vocab, eos_id, eos_every, seed, peak
        self.reset()

    def reset(self):
        self._state = []
        self._pending = []
        self.passes = 0

    @property
    def state(self):
        return tuple(self._state)

    def mode(self, prefix):
        """the peaked token after prefix (what a good drafter guesses)"""
        key = _h(self.seed, 'mode', tuple(prefix[-4:]), len(prefix))
        if self.eos_id is not None and self.eos_every and key % self.eos_every == 0:
            return self.eos_id
        return key % self.V

    def logits(self, prefix):
        """the target logits after prefix (float32 [V])"""
        key = _h(self.seed, 'row', tuple(prefix[-4:]), len(prefix))
        rng = np.random.default_rng(key)
        l = rng.standard_normal(self.V).astype(np.float32) * np.float32(1.5)
        lo, hi = self.peak
        l[self.mode(prefix)] += np.float32(lo + (hi - lo) * rng.random())
        return l

    def _apply(self, commit):
        if commit < 0 or commit > len(self._pending):
            raise EngineStateError(f'commit {commit} outside the previous pass ({len(self._pending)} tokens)')
        self._state += self._pending[:commit]
        self._pending = []

    def verify(self, seq, i0, commit):
        self._apply(commit)
        if i0 != len(self._state):
            raise EngineStateError(f'pass at i0={i0} but the state holds {len(self._state)} tokens')
        if list(seq[:i0]) != self._state:
            raise EngineStateError('pass prefix differs from the committed state')
        n = len(seq) - i0
        if not 1 <= n <= self.ntok:
            raise EngineStateError(f'pass of {n} tokens with ntok={self.ntok}')
        self._pending = list(seq[i0:])
        self.passes += 1
        return np.stack([self.logits(list(seq[:i0 + j + 1])) for j in range(n)])

    def flush(self, commit):
        self._apply(commit)


class MockDrafter:
    """deterministic drafts (a point mass, like DFlash's draft_graph): the engine's mode chain, each step replaced by
    a random token with probability 1 - quality (decided by a hash of the context, so drafts are reproducible).
    It stands in for DFlash (spec_type: llama-server's speculative.types name). clock / ms: a stats.VirtualClock
    advanced by ms per call (the deterministic metrics mode)."""
    spec_type = 'draft-dflash'

    def __init__(self, engine, quality=0.8, seed=1, clock=None, ms=2.0):
        self.m, self.quality, self.seed = engine, quality, seed
        self.clock, self.ms = clock, ms

    def __call__(self, toks, k):
        if self.clock is not None:
            self.clock.advance(self.ms / 1e3)
        out, ctx = [], list(toks)
        for j in range(k):
            r = _h(self.seed, tuple(ctx[-4:]), len(ctx))
            t = self.m.mode(ctx) if (r % 1000) < self.quality * 1000 else (r >> 10) % self.m.V
            out.append(t)
            ctx.append(t)
        return out


@dataclass
class SynthProfile:
    """plausible engine-internal numbers for MockSession (docs/specs/spec-metrics.md): the values only the real
    engine can measure (per-card pass and exchange-wait split, link bytes, the batched prefill and its hand-off,
    restore times by source, cache tier sizes, occasional recovered bails and KFD evictions).

    clock: a stats.VirtualClock makes every duration synthetic (the deterministic test mode: verify advances k3_ms,
    sampling sampler_ms, accept host_ms, a batched prompt pass tokens / batched_tok_s, a restore restore_ms; the
    MockDrafter its own ms). With clock None the durations are measured on the real clock; the batched passes, the
    hand-off and restores then sleep their synthetic time when sleep is True (the mock server), and plain passes
    sleep MockSession.pass_sleep as before.
    Shapes: K3 pass times of T = 1..4 tokens and the 2,900 tok/s prefill are REAP-308 / Coder measurements of the
    plan (AGENTIC-PLAN, spec-server.md section 4); link bytes per token row are 96 all-reduces x 2560 fp32 x 2
    directions (INFERRED)."""
    k3_ms: tuple = (10.45, 13.2, 15.0, 16.5)
    sampler_ms: float = 0.12
    host_ms: float = 0.35
    batched_tok_s: float = 2900.0
    handoff_ms: float = 2.0
    crossover: int = 48
    restore_ms: dict = field(default_factory=lambda: {'live': 0.0, 'vram': 0.6, 'disk': 260.0, 'cold': 0.0})
    card_split: tuple = (0.97, 0.91)
    wait_split: tuple = (0.05, 0.11)
    link_bytes_per_row: int = 96 * 2560 * 4 * 2
    ckpt_mib: float = 160.0
    headroom_mib: float = 2400.0
    bail_every: int = 0             # a recovered bail every N decode passes (0: never)
    evicted_ms: int = 0             # KFD eviction ms reported with each synthetic bail
    clock: object = None
    sleep: bool = True

    def pass_ms(self, n):
        return self.k3_ms[min(max(n, 1), len(self.k3_ms)) - 1]


class MockQDrafter:
    """q-sampled drafts: draft j is drawn from a drafter distribution q (a softened, perturbed copy of the target's
    logits) with its own RNG; returns (tokens, [(ids, probs)] per draft) for sampler.accept_q"""

    def __init__(self, engine, temp=1.3, noise=1.0, seed=2):
        self.m, self.temp, self.noise, self.seed = engine, temp, noise, seed
        self.calls = 0

    def q(self, ctx):
        rng = np.random.default_rng(_h(self.seed, 'q', tuple(ctx[-4:]), len(ctx)))
        l = self.m.logits(ctx).astype(np.float64) / self.temp + rng.standard_normal(self.m.V) * self.noise
        p = np.exp(l - l.max())
        return np.arange(self.m.V), p / p.sum()

    def __call__(self, toks, k):
        self.calls += 1
        rng = np.random.default_rng(_h(self.seed, 'draw', len(toks), self.calls))
        out, qs, ctx = [], [], list(toks)
        for _ in range(k):
            ids, p = self.q(ctx)
            t = int(rng.choice(ids, p=p))
            out.append(t); qs.append((ids, p)); ctx.append(t)
        return out, qs


class ScriptedEngine(MockEngine):
    """A MockEngine whose logits put (practically) all mass on a scripted continuation, for end-to-end server tests:
    register(prompt_ids, reply_ids) makes every prefix prompt + reply[:k] predict reply[k]. Lookups go by token
    match (the longest registered prompt that the prefix extends), so the order in which requests arrive does not
    matter. Prefixes with no script get the base mock's random logits (the text is then garbage, which a test sees).
    logits of a scripted row: 0 everywhere, `boost` on the scripted token (exp(-boost) of mass elsewhere)."""

    def __init__(self, vocab=248320, boost=60.0, **kw):
        super().__init__(vocab=vocab, **kw)
        self.boost = boost
        self.scripts = []           # (prompt tuple, reply tuple), longest prompt first
        self.misses = 0

    def register(self, prompt_ids, reply_ids):
        """a prompt registered again must keep its reply (two different replies to one prompt are a test bug)"""
        p, r = tuple(prompt_ids), tuple(reply_ids)
        for q, r0 in self.scripts:
            if q == p:
                if r0 != r:
                    raise ValueError('prompt already scripted with a different reply')
                return
        self.scripts.append((p, r))
        self.scripts.sort(key=lambda s: -len(s[0]))

    def scripted(self, prefix):
        n = len(prefix)
        for p, r in self.scripts:
            lp = len(p)
            k = n - lp
            if 0 <= k < len(r) and prefix[lp - 1] == p[-1] and tuple(prefix[lp:]) == r[:k] \
                    and tuple(prefix[:lp]) == p:
                return r[k]
        return None

    def mode(self, prefix):
        t = self.scripted(prefix)
        return super().mode(prefix) if t is None else t

    def logits(self, prefix):
        t = self.scripted(prefix)
        l = np.zeros(self.V, np.float32)
        if t is None:               # cheap unscripted row: a hash-picked mode over a flat floor
            self.misses += 1
            l[super().mode(prefix)] = 4.0
            return l
        l[t] = self.boost
        return l
