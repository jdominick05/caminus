"""Host reference sampler (server spec section 4, step S3). numpy only, CPU only; the device version is M7.

Filter chain, in llama.cpp's default order (common/common.h `samplers`; src/llama-sampler.cpp):
    penalties (repeat, frequency, presence over the last penalty_last_n tokens, prompt included)
    -> top_k -> top_p -> min_p -> temperature -> draw
Temperature comes last, so top_p and min_p see untempered logits, as in llama.cpp. Each filter keeps llama.cpp's
float32 arithmetic and the order of its sums (sequential in the current candidate order), so the surviving set and
its probabilities equal llama.cpp's (gate: tests/serve/test_sampler.py against tools/reference/sampler_ref.cpp).
Not ported: DRY, top-n-sigma, typical-p, XTC, dynamic temperature, mirostat, logit bias, grammar (all off in the
harness configurations).

RNG: Philox4x32-10 (Salmon et al. 2011; the same rounds as torch's at::philox_engine), key = the 64-bit request seed,
counter = (position lo, position hi, stream, 0). One uniform per (seed, absolute position, stream), so the draw for
the token at position i is the same however the passes are split: a seeded run is reproducible with or without
drafting. The uniform is a 53-bit double in [0, 1): ((w0 >> 5) * 2^26 + (w1 >> 6)) * 2^-53.
Streams: 0 = the token draw, 1 = the accept test of a q-sampled draft, 2 = its residual draw.

Draw: inverse CDF over the surviving candidates in ascending token id order. Weights are llama.cpp's
w_i = expf(l_i - l_max) (float32), summed sequentially in float64; the pick is the first i with cum_i > u * sum.
llama.cpp draws with mt19937 instead, so only the filtered distribution is shared with llama.cpp, not the draws.

Speculative acceptance:
  * deterministic drafts (DFlash returns tokens, a point mass): sample and match, as llama.cpp's
    common_sampler_sample_and_accept_n. Row j draws y_j from the filtered target distribution with history + drafts[:j]
    and the uniform of its own position; drafts are accepted while y_j == d_j, and the first mismatch emits y_j.
    The output is token-for-token the plain run's (same uniforms, same histories), hence exactly distributed.
  * q-sampled drafts: accept_q(), the standard rule (accept with min(1, p/q), else draw from norm(max(0, p - q)))
    on streams 1 and 2. Exact in distribution, but NOT token-equal to the plain run; never use it for
    deterministic drafts.
Greedy (temperature <= 0): argmax of the penalized logits, first max wins; with penalties off, a plain argmax."""
import os
from dataclasses import dataclass, replace

import numpy as np

# ---------------------------------------------------------------- Philox4x32-10

_MASK = 0xFFFFFFFF
_PHILOX_M0, _PHILOX_M1 = 0xD2511F53, 0xCD9E8D57          # multipliers (torch kPhiloxSA, kPhiloxSB)
_PHILOX_W0, _PHILOX_W1 = 0x9E3779B9, 0xBB67AE85          # Weyl key increments (kPhilox10A, kPhilox10B)
STREAM_DRAW, STREAM_ACCEPT, STREAM_RESIDUAL = 0, 1, 2


def philox4x32(ctr, key, rounds=10):
    """one Philox4x32 block: ctr = 4 uint32, key = 2 uint32 -> 4 uint32"""
    c0, c1, c2, c3 = (int(x) & _MASK for x in ctr)
    k0, k1 = (int(x) & _MASK for x in key)
    for r in range(rounds):
        if r:
            k0 = (k0 + _PHILOX_W0) & _MASK
            k1 = (k1 + _PHILOX_W1) & _MASK
        p0 = _PHILOX_M0 * c0
        p1 = _PHILOX_M1 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ k0) & _MASK, p1 & _MASK, ((p0 >> 32) ^ c3 ^ k1) & _MASK, p0 & _MASK
    return c0, c1, c2, c3


def philox4x32_np(c0, c1, c2, c3, k0, k1, rounds=10):
    """vectorized philox4x32 over numpy arrays (broadcast); returns 4 uint64 arrays holding uint32 values"""
    m = np.uint64(_MASK)
    c0, c1, c2, c3 = (np.asarray(x, dtype=np.uint64) & m for x in (c0, c1, c2, c3))
    k0, k1 = (np.asarray(x, dtype=np.uint64) & m for x in (k0, k1))
    c0, c1, c2, c3, k0, k1 = np.broadcast_arrays(c0, c1, c2, c3, k0, k1)
    k0, k1 = k0.copy(), k1.copy()
    s32 = np.uint64(32)
    for r in range(rounds):
        if r:
            k0 = (k0 + np.uint64(_PHILOX_W0)) & m
            k1 = (k1 + np.uint64(_PHILOX_W1)) & m
        p0 = np.uint64(_PHILOX_M0) * c0
        p1 = np.uint64(_PHILOX_M1) * c2
        c0, c1, c2, c3 = ((p1 >> s32) ^ c1 ^ k0) & m, p1 & m, ((p0 >> s32) ^ c3 ^ k1) & m, p0 & m
    return c0, c1, c2, c3


def seed_key(seed):
    s = int(seed) & 0xFFFFFFFFFFFFFFFF
    return s & _MASK, s >> 32


def uniform(seed, pos, stream=STREAM_DRAW):
    """the uniform double in [0, 1) for (seed, absolute position, stream)"""
    p = int(pos)
    w = philox4x32((p & _MASK, (p >> 32) & _MASK, stream, 0), seed_key(seed))
    return ((w[0] >> 5) * 67108864 + (w[1] >> 6)) * (1.0 / 9007199254740992.0)


def uniform_np(seed, pos, stream=STREAM_DRAW):
    p = np.asarray(pos, dtype=np.uint64)
    k0, k1 = seed_key(seed)
    w0, w1, _, _ = philox4x32_np(p & np.uint64(_MASK), p >> np.uint64(32), stream, 0, k0, k1)
    return ((w0 >> np.uint64(5)).astype(np.float64) * 67108864.0 + (w1 >> np.uint64(6)).astype(np.float64)) \
        * (1.0 / 9007199254740992.0)


def random_seed():
    return int.from_bytes(os.urandom(8), 'little')


# ---------------------------------------------------------------- parameters

LLAMA_DEFAULT_SEED = 0xFFFFFFFF     # llama.cpp's "pick a random seed"


@dataclass(frozen=True)
class SamplingParams:
    """Defaults: the GGUF's general.sampling.* (top_k 20, top_p 0.95, temp 1.0) and the llama-server reference runs' min_p 0; the
    penalties are llama.cpp's defaults (off, window 64). penalty_last_n: 0 = off, -1 = the whole history.
    seed None (or -1 / LLAMA_DEFAULT_SEED) = a random seed, fixed per request by resolve()."""
    temperature: float = 1.0
    top_k: int = 20
    top_p: float = 0.95
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    penalty_last_n: int = 64
    min_keep: int = 0
    seed: int = None

    @classmethod
    def from_gguf_meta(cls, meta, **over):
        """defaults from GGUF metadata (caminus.serve.vocab.Vocab.meta), then keyword overrides"""
        kw = {}
        for gk, name, typ in (('general.sampling.temp', 'temperature', float), ('general.sampling.top_k', 'top_k', int),
                              ('general.sampling.top_p', 'top_p', float), ('general.sampling.min_p', 'min_p', float)):
            if gk in meta:
                kw[name] = typ(meta[gk])
        kw.update(over)
        return cls(**kw)

    def with_request(self, req):
        """apply a request's sampling fields (OpenAI, Anthropic and llama.cpp names; absent or None = keep)"""
        names = {'temperature': 'temperature', 'top_k': 'top_k', 'top_p': 'top_p', 'min_p': 'min_p',
                 'repeat_penalty': 'repeat_penalty', 'frequency_penalty': 'frequency_penalty',
                 'presence_penalty': 'presence_penalty', 'repeat_last_n': 'penalty_last_n',
                 'penalty_last_n': 'penalty_last_n', 'min_keep': 'min_keep', 'seed': 'seed'}
        kw = {}
        for rk, name in names.items():
            v = req.get(rk)
            if v is None or isinstance(v, bool):
                continue
            kw[name] = int(v) if name in ('top_k', 'penalty_last_n', 'min_keep', 'seed') else float(v)
        return replace(self, **kw) if kw else self

    def resolve(self):
        """fix a random seed when none was given (the seed is reported back so the run can be repeated)"""
        if self.seed is None or self.seed < 0 or self.seed == LLAMA_DEFAULT_SEED:
            return replace(self, seed=random_seed())
        return self

    @property
    def greedy(self):
        return self.temperature <= 0.0

    @property
    def penalties_on(self):
        return self.penalty_last_n != 0 and not (self.repeat_penalty == 1.0 and self.frequency_penalty == 0.0
                                                 and self.presence_penalty == 0.0)


# ---------------------------------------------------------------- filter chain

_F32 = np.float32


@dataclass
class Candidates:
    """the filtered distribution: ids ascending, the final (tempered) logits, and the normalized probabilities.
    weights = expf(l - l_max) unnormalized (float32), the draw's weights."""
    ids: np.ndarray       # int64, ascending
    logits: np.ndarray    # float32
    probs: np.ndarray     # float32
    weights: np.ndarray   # float32


def _expf(x):
    # expf of a float32 array, correctly rounded via float64 (closer to glibc expf than numpy's float32 SIMD exp)
    return np.exp(x.astype(np.float64)).astype(_F32)


def _seqsum32(x):
    # sequential float32 sum (llama.cpp's `cum_sum += p` loop); np.sum is pairwise
    return np.add.accumulate(x, dtype=_F32)[-1] if len(x) else _F32(0)


def penalty_counts(history, last_n):
    """(ids, counts) of the penalty window: the last last_n tokens of history (-1 = all, 0 = none)"""
    if last_n == 0 or not len(history):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    h = np.asarray(history, dtype=np.int64)
    if last_n > 0:
        h = h[-last_n:]
    return np.unique(h, return_counts=True)


def apply_penalties(ids, logits, history, prm):
    """llama.cpp llama_sampler_penalties_apply on candidates (ids, logits); ids None = a full row (id == index).
    Returns new float32 logits (the input is not modified)."""
    out = np.array(logits, dtype=_F32, copy=True)
    if not prm.penalties_on or not len(out):
        return out
    pid, cnt = penalty_counts(history, prm.penalty_last_n)
    if not len(pid):
        return out
    if ids is None:                                              # full row: id == index
        pos = pid[pid < len(out)]; cnt = cnt[pid < len(out)]
    else:
        ids = np.asarray(ids)
        order = np.argsort(ids, kind='stable')
        loc = np.searchsorted(ids, pid, sorter=order)
        loc = np.minimum(loc, len(ids) - 1)
        hit = ids[order[loc]] == pid
        pos, cnt = order[loc[hit]], cnt[hit]
    l = out[pos]
    rep = _F32(prm.repeat_penalty)
    l = np.where(l <= 0, l * rep, l / rep).astype(_F32)
    sub = (cnt.astype(_F32) * _F32(prm.frequency_penalty) + (cnt > 0).astype(_F32) * _F32(prm.presence_penalty))
    out[pos] = (l - sub.astype(_F32)).astype(_F32)
    return out


def _sort_desc(ids, logits):
    # descending logit, ties by ascending id (llama.cpp's partial_sort leaves tie order unspecified)
    o = np.lexsort((ids, -logits.astype(np.float64)))
    return ids[o], logits[o]


def _topk_sorted(ids, logits, k):
    """the k largest, sorted; a tie at the k-th logit keeps the lowest ids (deterministic)"""
    if k < len(logits):
        v = logits[np.argpartition(-logits, k - 1)[k - 1]]       # the k-th largest value
        sel = np.nonzero(logits >= v)[0]                          # everything above it plus all of its ties
        i, l = _sort_desc(ids[sel], logits[sel])
        return i[:k], l[:k]
    return _sort_desc(ids, logits)


def filter_chain(logits, prm, history=(), ids=None):
    """llama.cpp's chain on one row. logits: float32 [V] (or candidates with their ids). Returns Candidates."""
    l = np.asarray(logits, dtype=_F32)
    l = apply_penalties(ids, l, history, prm)
    ids = np.arange(len(l), dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
    is_sorted = False
    # top_k
    if prm.top_k > 0:
        ids, l = _topk_sorted(ids, l, min(prm.top_k, len(l)))
        is_sorted = True
    # top_p (softmax over the current candidates in their current order, then cumulative in sorted order)
    p = _F32(prm.top_p)
    if p < _F32(1.0) and len(l):
        mx = l[0] if is_sorted else l.max()
        e = _expf((l - mx).astype(_F32))
        pr = (e / _seqsum32(e)).astype(_F32)
        if not is_sorted:
            o = np.lexsort((ids, -l.astype(np.float64)))
            ids, l, pr = ids[o], l[o], pr[o]
            is_sorted = True
        cum = np.add.accumulate(pr, dtype=_F32)
        i = np.arange(len(cum))
        hit = np.nonzero((cum >= p) & (i + 1 >= prm.min_keep))[0]
        last = int(hit[0]) + 1 if len(hit) else len(cum)
        ids, l = ids[:last], l[:last]
    # min_p
    mp = _F32(prm.min_p)
    if mp > _F32(0.0) and len(l):
        done = False
        if not is_sorted:
            min_logit = _F32(l.max() + _F32(np.log(np.float64(mp))))
            keep = l >= min_logit
            nk = int(keep.sum())
            if nk and nk >= prm.min_keep:
                ids, l = ids[keep], l[keep]
                done = True
        if not done:
            if not is_sorted:
                ids, l = _sort_desc(ids, l)
                is_sorted = True
            min_logit = _F32(l[0] + _F32(np.log(np.float64(mp))))
            i = np.arange(len(l))
            stop = np.nonzero((l < min_logit) & (i >= prm.min_keep) & (i >= 1))[0]
            n = int(stop[0]) if len(stop) else len(l)
            ids, l = ids[:n], l[:n]
    # temperature (llama_sampler_temp_impl): t <= 0 keeps the first max only (in the current order)
    t = _F32(prm.temperature)
    if t <= _F32(0.0):
        j = int(np.argmax(l))
        ids, l = ids[j:j + 1], l[j:j + 1]
    else:
        l = (l / t).astype(_F32)
    # dist: weights expf(l - max), probs normalized by the float64 sum
    o = np.argsort(ids, kind='stable')
    ids, l = ids[o], l[o]
    w = _expf((l - l.max()).astype(_F32)) if len(l) else l
    s = float(np.add.accumulate(w.astype(np.float64))[-1]) if len(w) else 0.0
    probs = (w.astype(np.float64) / s).astype(_F32) if s > 0 else w
    return Candidates(ids=ids, logits=l, probs=probs, weights=w)


def draw(c, u):
    """inverse CDF over Candidates c (ascending id order) with the uniform u in [0, 1)"""
    cum = np.add.accumulate(c.weights.astype(np.float64))
    i = int(np.searchsorted(cum, u * cum[-1], side='right'))
    if i >= len(cum):                                    # u * sum rounded up to sum: the last positive weight
        i = int(np.nonzero(c.weights > 0)[0][-1])
    return int(c.ids[i])


def draw_np(c, u):
    """vectorized draw() over an array of uniforms"""
    cum = np.add.accumulate(c.weights.astype(np.float64))
    i = np.searchsorted(cum, np.asarray(u) * cum[-1], side='right')
    last = int(np.nonzero(c.weights > 0)[0][-1])
    return c.ids[np.where(i >= len(cum), last, np.minimum(i, len(cum) - 1))]


def greedy_token(logits, prm, history=()):
    """temperature <= 0: argmax of the penalized logits, first max wins (top_k / top_p / min_p keep the max)"""
    l = np.asarray(logits, dtype=_F32)
    if prm.penalties_on:
        l = apply_penalties(None, l, history, prm)
    return int(np.argmax(l))


# ---------------------------------------------------------------- per-request sampler

class Sampler:
    """one request's sampler. history = every token before the one being sampled (prompt included); the token's
    absolute position is len(history)."""

    def __init__(self, params):
        self.params = params.resolve()
        self.seed = self.params.seed

    def uniform(self, pos, stream=STREAM_DRAW):
        return uniform(self.seed, pos, stream)

    def distribution(self, logits, history):
        return filter_chain(logits, self.params, history)

    def sample(self, logits, history):
        if self.params.greedy:
            return greedy_token(logits, self.params, history)
        return draw(filter_chain(logits, self.params, history), self.uniform(len(history)))

    def sample_and_match(self, rows, drafts, history, stop_ids=()):
        """rows[j] = target logits for the token after history + drafts[:j] (len(rows) = len(drafts) + 1).
        Returns the emitted tokens: the accepted drafts, then the first mismatching (or bonus) draw. Emission stops at
        the first stop id (EOS), even inside accepted drafts (that stop token is then an accepted draft itself)."""
        h = list(history)
        out = []
        for j in range(len(drafts) + 1):
            y = self.sample(rows[j], h)
            out.append(y)
            if y in stop_ids:
                break
            if j == len(drafts) or y != drafts[j]:
                break
            h.append(y)
        return out

    def accept_q(self, logits, history, x, q_ids, q_probs):
        """one q-sampled draft x with drafter distribution q (q_ids, q_probs) at position len(history).
        Returns (token, accepted). accepted: u1 < min(1, p(x)/q(x)); otherwise a draw from norm(max(0, p - q)).
        Greedy targets accept iff x is the argmax."""
        pos = len(history)
        if self.params.greedy:
            y = greedy_token(logits, self.params, history)
            return y, y == x
        c = filter_chain(logits, self.params, history)
        p = dict(zip(c.ids.tolist(), c.probs.astype(np.float64).tolist()))
        q = dict(zip((int(i) for i in q_ids), (float(v) for v in q_probs)))
        px, qx = p.get(int(x), 0.0), q.get(int(x), 0.0)
        if qx > 0.0 and self.uniform(pos, STREAM_ACCEPT) * qx < px:     # u < p/q without dividing
            return int(x), True
        ids = np.array(sorted(set(p) | set(q)), dtype=np.int64)
        r = np.array([max(0.0, p.get(int(i), 0.0) - q.get(int(i), 0.0)) for i in ids])
        if not (r.sum() > 0.0):                                           # p == q numerically: fall back to p
            return draw(c, self.uniform(pos, STREAM_RESIDUAL)), False
        cum = np.add.accumulate(r)
        i = int(np.searchsorted(cum, self.uniform(pos, STREAM_RESIDUAL) * cum[-1], side='right'))
        if i >= len(cum):
            i = int(np.nonzero(r > 0)[0][-1])
        return int(ids[i]), False


def sampled_generate(m, sampler, prompt, n_new, draft=None, stop_ids=(), stats=None, q_draft=False):
    """caminus.decode.spec.generate with sampling. m: an engine with ntok = T and verify(seq, i0, commit) -> [T, V] rows
    (row j predicts the token after seq[i0 + j]). draft(tokens, k) -> k tokens (or, with q_draft, (tokens, q) where
    q[j] = (ids, probs) is the drafter's distribution for draft j). Returns (tokens, commit): prompt + generated tokens,
    and the lazy commit that leaves the engine state at every token but the last (as a plain run stopped there).
    Generation ends after n_new tokens or at a stop id."""
    T = m.ntok
    toks = list(prompt)
    i0 = commit = passes = acc = drafted = multi = stop_in_draft = boundary = 0
    m.reset()
    done = False
    while not done:
        forced = min(T, len(toks) - i0)
        k = T - forced if draft is not None else 0
        if k and len(toks) - len(prompt) + 1 >= n_new:
            k = 0                                           # one token left: no point drafting
        qd = None
        if k:
            d = draft(toks, k)
            if q_draft:
                d, qd = d
            d = list(d)[:k]
        else:
            d = []
        seq = toks[:i0 + forced] + d
        lg = m.verify(seq, i0, commit)
        passes += 1
        if i0 + forced < len(toks):                         # still in the prompt
            commit = forced; i0 += forced
            continue
        rows = lg[forced - 1:forced + len(d)]
        if qd is None:
            out = sampler.sample_and_match(rows, d, toks, stop_ids)
        else:
            out, h = [], list(toks)
            for jj in range(len(d) + 1):
                if jj < len(d):
                    y, ok = sampler.accept_q(rows[jj], h, d[jj], *qd[jj])
                else:
                    y, ok = sampler.sample(rows[jj], h), False
                out.append(y)
                if y in stop_ids or not ok:
                    break
                h.append(y)
        drafted += len(d)
        boundary += forced > 1 and len(d) > 0             # the prompt's last pass also carries drafts
        left = len(prompt) + n_new - len(toks)
        if len(out) >= left:
            out = out[:left]; done = True
        if any(t in stop_ids for t in out):
            out = out[:next(i for i, t in enumerate(out) if t in stop_ids) + 1]; done = True
        a = len(out) - 1                                    # out[:-1] are accepted drafts; out[-1] is the new draw
        # the last token can equal its draft only when emission ended there (a stop, or n_new reached)
        last_is_draft = done and len(out) <= len(d) and out[-1] == d[len(out) - 1]
        if done and out[-1] in stop_ids and last_is_draft:
            stop_in_draft += 1                              # a drafted stop token was accepted
        acc += a + last_is_draft
        multi += a >= 2
        toks += out
        if done:
            commit = forced + len(out) - 1
            break
        commit = forced + a; i0 += forced + a
    if stats is not None:
        stats.update(passes=passes, accepted=acc, drafted=drafted, multi_accept_passes=multi, stop_in_draft=stop_in_draft,
                     boundary_draft_passes=boundary,
                     gen_tokens=len(toks) - len(prompt))
    return toks, commit
