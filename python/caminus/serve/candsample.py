"""The device sampler's host half (server spec section 4 'Two-card logits'; M7): sampling from per-card top-k candidates
instead of whole logits rows, bitwise the host reference sampler (sampler.py) on the full rows.

The TP engine leaves each verify row's logits split by vocabulary on its two cards (card 0: ids [0, vh), card 1: [vh, V)).
Copying a whole row to the host is ~1 MB (the XT's half over the x4 link). Instead each card returns, per row:
  * its k' largest raw logits with their ids (torch.topk, sorted; the order among equal values unspecified), and
  * the raw logits of the penalty ids in its range (every id of the rows' penalty windows: the history's last
    penalty_last_n tokens plus the pass's drafts),
and the host merges them here: the penalties applied to the candidates with sampler.apply_penalties' float32 arithmetic,
then sampler.filter_chain on the candidates (ids given). k' = top_k + (penalty ids) + TIE_MARGIN.

Why this is exact. A token outside the candidates is unpenalized and its raw logit is <= v_c, its card's k'-th value.
Each card alone holds >= k' - n_pen = top_k + TIE_MARGIN unpenalized candidates >= v_c, so the k-th largest penalized
candidate s_k is >= max(v_c). When s_k > v_c on both cards, no token outside the candidates can enter top_k or tie its
last member, so top_k selects the same tokens in the same order (value descending, ids ascending on ties) as on the full
row; top_p, min_p and temperature then act on those survivors only (llama.cpp's chain), so the Candidates are equal.
When s_k == v_c on a card (a tie at the cut; torch.topk may have left out a lower id of equal value), or any value is
not finite, or top_k is 0 or above MAX_TOP_K, the row falls back to the full row (`need_full`). Greedy (temperature
<= 0) needs only the maximum: the same argument with k = 1 (the first maximum: the lowest id).

Sample and match (sampler.Sampler.sample_and_match) runs over the rows as there: row j's history is history +
drafts[:j], its uniform is Philox(seed, its absolute position)."""
from dataclasses import replace

import numpy as np

from . import sampler as S

MAX_TOP_K = 64          # the candidate path's top_k (the spec's top-64); above it (or top_k 0) the full row
TIE_MARGIN = 8          # extra candidates per card: a tie at the cut then needs 8 + 1 equal values to force the full row


def penalty_ids(prm, history, drafts):
    """the ids whose raw logits the device must return: every id of every row's penalty window (sorted, unique); empty with
    the penalties off"""
    if not prm.penalties_on:
        return np.zeros(0, np.int64)
    n = prm.penalty_last_n
    h = list(history) + list(drafts)
    if n > 0:
        h = h[-(n + len(drafts)):]
    return np.unique(np.asarray(h, dtype=np.int64)) if h else np.zeros(0, np.int64)


def k_prime(prm, n_pen):
    """the candidates each card returns per row (None: the full row is needed)"""
    if prm.greedy:
        return 1 + n_pen + TIE_MARGIN
    if not 0 < prm.top_k <= MAX_TOP_K:
        return None
    return prm.top_k + n_pen + TIE_MARGIN


class Row:
    """one verify row's device output: per card (vals [k'] float32 descending, ids [k'] int64 global), and the raw logits of
    the penalty ids (pid sorted, pvals float32); full: the whole row (float32 [V]) when the candidate path cannot be exact"""
    __slots__ = ('cards', 'pid', 'pvals', 'full')

    def __init__(self, cards=None, pid=None, pvals=None, full=None):
        self.cards, self.pid, self.pvals, self.full = cards, pid, pvals, full


def _nopen(prm):
    return replace(prm, repeat_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0) if prm.penalties_on else prm


def _counts(prm, history):
    """(ids, counts) of the row's penalty window (sampler.penalty_counts)"""
    return S.penalty_counts(history, prm.penalty_last_n) if prm.penalties_on else (None, None)


def _penalize(u_ids, u_val, cid, cnt, prm):
    """sampler.apply_penalties on candidates u_ids (ascending) holding every id of cid: the same float32 expressions"""
    out = np.array(u_val, dtype=S._F32, copy=True)
    if cid is None or not len(cid):
        return out
    pos = np.searchsorted(u_ids, cid)
    l = out[pos]
    rep = S._F32(prm.repeat_penalty)
    l = np.where(l <= 0, l * rep, l / rep).astype(S._F32)
    sub = (cnt.astype(S._F32) * S._F32(prm.frequency_penalty) + (cnt > 0).astype(S._F32) * S._F32(prm.presence_penalty))
    out[pos] = (l - sub.astype(S._F32)).astype(S._F32)
    return out


def candidates(row, prm, history, counts=None):
    """(ids ascending, penalized logits) of the row's candidates that can survive top_k (every one >= the k-th largest), or
    None when the full row is needed (see the module notes). counts: the row's penalty window (_counts), if known"""
    (v0, i0), (v1, i1) = row.cards
    pv = row.pvals
    if not (np.isfinite(v0).all() and np.isfinite(v1).all()) or (pv is not None and len(pv) and not np.isfinite(pv).all()):
        return None
    u_ids = np.concatenate((i0, i1))
    u_val = np.concatenate((v0, v1))
    if row.pid is not None and len(row.pid):
        have = set(u_ids.tolist())
        extra = [j for j, p in enumerate(row.pid.tolist()) if p not in have]
        if extra:
            u_ids = np.concatenate((u_ids, row.pid[extra]))
            u_val = np.concatenate((u_val, pv[extra]))
    o = np.argsort(u_ids, kind='stable')
    u_ids, u_val = u_ids[o], u_val[o].astype(S._F32, copy=False)
    cid, cnt = counts if counts is not None else _counts(prm, history)
    pen = _penalize(u_ids, u_val, cid, cnt, prm)
    k = 1 if prm.greedy else prm.top_k
    if k > len(pen):
        return None
    s_k = pen[np.argpartition(-pen, k - 1)[k - 1]]          # the k-th largest penalized candidate
    if not (s_k > v0[-1] and s_k > v1[-1]):                 # a tie at a card's cut: not provably exact
        return None
    keep = pen >= s_k                                       # top_k's survivors (ties at s_k included: top_k cuts them)
    return u_ids[keep], pen[keep]


def _expf(x):
    return np.exp(x.astype(np.float64)).astype(S._F32)


def draw_fast(ids, l, prm, u):
    """S.draw(S.filter_chain(l, prm with the penalties off, ids=ids), u) for candidates (ids, penalized logits) that hold every
    survivor of top_k (0 < top_k <= len) and nothing below the k-th value but its ties (candidates() gives exactly that), with
    temperature > 0: the same float32 expressions in the same order (top_k's sort, top_p's softmax and cumulative sum, min_p,
    the temperature, the weights expf(l - max) and the float64 inverse CDF), without the steps the draw does not read
    (sampler.Candidates' probabilities). Bitwise filter_chain + draw (tests/serve/test_candsample.py)"""
    f32 = S._F32
    o = np.lexsort((ids, -l.astype(np.float64)))            # top_k: value descending, ids ascending on ties (_sort_desc)
    k = min(prm.top_k, len(l))
    ids, l = ids[o[:k]], l[o[:k]]
    p = f32(prm.top_p)
    if p < f32(1.0):
        e = _expf((l - l[0]).astype(f32))
        pr = (e / np.add.accumulate(e, dtype=f32)[-1]).astype(f32)
        cum = np.add.accumulate(pr, dtype=f32)
        hit = np.nonzero((cum >= p) & (np.arange(1, len(cum) + 1) >= prm.min_keep))[0]
        if len(hit):
            n = int(hit[0]) + 1
            ids, l = ids[:n], l[:n]
    mp = f32(prm.min_p)
    if mp > f32(0.0):
        min_logit = f32(l[0] + f32(np.log(np.float64(mp))))
        i = np.arange(len(l))
        stop = np.nonzero((l < min_logit) & (i >= prm.min_keep) & (i >= 1))[0]
        if len(stop):
            n = int(stop[0])
            ids, l = ids[:n], l[:n]
    l = (l / f32(prm.temperature)).astype(f32)
    o = np.argsort(ids, kind='stable')
    ids, l = ids[o], l[o]
    w = _expf((l - l.max()).astype(f32))
    cum = np.add.accumulate(w.astype(np.float64))
    j = int(np.searchsorted(cum, u * cum[-1], side='right'))
    if j >= len(cum):
        j = int(np.nonzero(w > 0)[0][-1])
    return int(ids[j])


def distribution(row, prm, history, counts=None):
    """sampler.filter_chain of the row (Candidates), from the candidates when exact, else from the full row"""
    if row.cards is not None:
        c = candidates(row, prm, history, counts)
        if c is not None:
            return S.filter_chain(c[1], _nopen(prm), history, ids=c[0])
    if row.full is None:
        raise LookupError('candsample: the full row is needed')
    return S.filter_chain(row.full, prm, history)


def sample(row, sampler, history, counts=None):
    """sampler.Sampler.sample on one Row (the same token, bitwise the same arithmetic)"""
    prm = sampler.params
    if prm.greedy:
        if row.cards is not None:
            c = candidates(row, prm, history, counts)
            if c is not None:
                return int(c[0][int(np.argmax(c[1]))])       # (ascending ids: the first maximum is the lowest id)
        if row.full is None:
            raise LookupError('candsample: the full row is needed')
        return S.greedy_token(row.full, prm, history)
    if row.cards is not None and prm.temperature > 0:
        c = candidates(row, prm, history, counts)
        if c is not None:
            return draw_fast(c[0], c[1], prm, sampler.uniform(len(history)))
    return S.draw(distribution(row, prm, history, counts), sampler.uniform(len(history)))


def needs_full(row, prm, history):
    """True when this row cannot be sampled from its candidates (the caller fetches row.full and samples again)"""
    return row.cards is None or candidates(row, prm, history) is None


def sample_and_match(rows, sampler, drafts, history, stop_ids=(), fetch=None):
    """sampler.Sampler.sample_and_match over Rows: rows[j] predicts the token after history + drafts[:j]. fetch(j) -> float32
    [V]: the full row j, called only for a row whose candidates cannot be exact (it fills row.full). Returns the emitted ids"""
    prm = sampler.params
    h = list(history)
    out = []
    for j in range(len(drafts) + 1):
        r = rows[j]
        cn = _counts(prm, h)
        if r.full is None and (r.cards is None or candidates(r, prm, h, cn) is None):
            if fetch is None:
                raise LookupError(f'candsample: row {j} needs the full row and no fetch was given')
            r.full = fetch(j)
        y = sample(r, sampler, h, cn)
        out.append(y)
        if y in stop_ids:
            break
        if j == len(drafts) or y != drafts[j]:
            break
        h.append(y)
    return out


def batched_ok(prm):
    """the batched path (draws_batched) applies: T > 0, penalties off (every row's distribution is then independent of the
    history), 0 < top_k <= MAX_TOP_K"""
    return prm.temperature > 0 and not prm.penalties_on and 0 < prm.top_k <= MAX_TOP_K


def draws_batched(V0, I0, V1, I1, prm, us):
    """the tokens sample() would draw on R rows at once (penalties off, batched_ok): V0 / I0, V1 / I1 [R, k'] the cards'
    candidates (values descending, global ids), us [R] the rows' uniforms. The same float32 expressions as draw_fast per row,
    vectorized over the rows (variable survivor counts by masks; padding never enters a sum: zero weights at the end of the
    float64 prefix sums, the reference's inverse-CDF fallback kept). Returns (tokens [R] int64, ok [R] bool): rows with ok
    False (a tie at a card's cut, a non-finite value) must be drawn from the full row"""
    f32 = S._F32
    R = V0.shape[0]
    V = np.concatenate((V0, V1), axis=1)
    I = np.concatenate((I0, I1), axis=1)
    k = min(prm.top_k, V.shape[1])
    fin = np.isfinite(V).all(axis=1)
    Vs = V                                                    # (a row with a non-finite value is flagged; its numbers unused)
    rr = np.arange(R)[:, None]
    # top_k per row: value descending, ids ascending on ties, as one int64 key per candidate (the float32 bits mapped to a
    # signed order, -0.0 as +0.0 so equal values tie as the float compare has it; ids < 2^20)
    b = (Vs + f32(0.0)).view(np.int32).astype(np.int64)
    key = ((-(b ^ ((b >> 31) & 0x7FFFFFFF))) << 20) | I
    o = np.argsort(key, axis=1)[:, :k]
    l = Vs[rr, o]
    ids = I[rr, o]
    ok = fin & (l[:, -1] > V0[:, -1]) & (l[:, -1] > V1[:, -1])    # the k-th value strictly above both cards' cuts
    n = np.full(R, k, dtype=np.int64)
    col = np.arange(k)
    p = f32(prm.top_p)
    if p < f32(1.0):
        e = _expf((l - l[:, :1]).astype(f32))
        pr = (e / np.add.accumulate(e, axis=1, dtype=f32)[:, -1:]).astype(f32)
        cum = np.add.accumulate(pr, axis=1, dtype=f32)
        hit = (cum >= p) & (col[None, :] + 1 >= prm.min_keep)
        n = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, n)
    mp = f32(prm.min_p)
    if mp > f32(0.0):
        min_logit = (l[:, 0] + f32(np.log(np.float64(mp)))).astype(f32)
        stop = (l < min_logit[:, None]) & (col[None, :] >= prm.min_keep) & (col[None, :] >= 1) & (col[None, :] < n[:, None])
        n = np.where(stop.any(axis=1), stop.argmax(axis=1), n)
    valid = col[None, :] < n[:, None]
    l = (l / f32(prm.temperature)).astype(f32)
    big = np.iinfo(np.int64).max
    o2 = np.argsort(np.where(valid, ids, big), axis=1, kind='stable')
    ids, l, valid = ids[rr, o2], l[rr, o2], valid[rr, o2]
    mx = np.where(valid, l, -np.inf).max(axis=1).astype(f32)
    w = np.where(valid, _expf((l - mx[:, None]).astype(f32)), f32(0.0)).astype(f32)
    cum = np.add.accumulate(w.astype(np.float64), axis=1)
    tot = cum[np.arange(R), n - 1]
    j = (cum <= (np.asarray(us, dtype=np.float64) * tot)[:, None]).sum(axis=1)
    over = j >= n
    if over.any():                                            # u * sum rounded up to sum: the last positive weight
        lastpos = np.where(valid & (w > 0), col[None, :], -1).max(axis=1)
        j = np.where(over, lastpos, j)
    return ids[np.arange(R), j], ok


def sample_and_match_batched(V0, I0, V1, I1, sampler, drafts, history, stop_ids=(), fetch=None, us=None):
    """sample_and_match for batched_ok parameters: every row's token drawn at once (draws_batched), then the match; a row
    that needs its full row is drawn from fetch(j) (sampler.Sampler.sample: the reference). us: the rows' uniforms when the
    caller computed them already (Philox(seed, len(history) + j))"""
    R = len(drafts) + 1
    base = len(history)
    if us is None:
        us = [sampler.uniform(base + j) for j in range(R)]
    toks, ok = draws_batched(V0[:R], I0[:R], V1[:R], I1[:R], sampler.params, us)
    out = []
    h = None
    for j in range(R):
        if ok[j]:
            y = int(toks[j])
        else:
            if fetch is None:
                raise LookupError(f'candsample: row {j} needs the full row and no fetch was given')
            if h is None: h = list(history)
            y = sampler.sample(fetch(j), list(history) + list(drafts[:j]))
        out.append(y)
        if y in stop_ids or j == len(drafts) or y != drafts[j]:
            break
    return out


def emulate_rows(full_rows, vh, prm, history, drafts, tie_order='high'):
    """CPU emulation of the device half (tests): Rows from full logits rows [R, V] split at vh, each card's top-k' with
    ties at the cut resolved adversarially (tie_order 'high': the highest ids first, the opposite of the host's rule;
    'low': the lowest first), plus the penalty ids' raw values"""
    full_rows = np.asarray(full_rows, dtype=np.float32)
    pid = penalty_ids(prm, history, drafts)
    kp = k_prime(prm, len(pid))
    out = []
    for r in full_rows:
        if kp is None:
            out.append(Row(full=None))
            continue
        cards = []
        for lo, hi in ((0, vh), (vh, len(r))):
            v = r[lo:hi]
            key = (np.arange(hi - lo)[::-1] if tie_order == 'high' else np.arange(hi - lo))
            vv = np.where(np.isnan(v), np.inf, v).astype(np.float64)
            o = np.lexsort((key, -vv, ~np.isnan(v)))[:kp]           # NaN first (torch.topk's rule), value descending, ties
            cards.append((v[o].astype(np.float32), (o + lo).astype(np.int64)))
        out.append(Row(cards=cards, pid=pid, pvals=r[pid] if len(pid) else np.zeros(0, np.float32)))
    return out
