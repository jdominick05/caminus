"""The device sampler's host half (caminus.serve.candsample, M7), CPU only: sampling from each card's top-k' candidates
plus the penalty ids' raw logits gives bitwise the host reference sampler's result on the full rows (sampler.py):
the same Candidates (ids, logits, probs, weights) and the same sample-and-match tokens, under adversarial tie orders at
the cards' cuts, heavy ties, penalties of both signs, both vocabulary halves, NaN / inf rows (the full-row fallback),
greedy with and without penalties, top_k 0 / above 64 (the full row). Also the host time per row.
Run:  HIP_VISIBLE_DEVICES= python -I tests/serve/run_all.py"""
import time
import unittest

import numpy as np

import common  # noqa: F401
from caminus.serve import candsample as CS
from caminus.serve import sampler as S


def rows_for(rng, R, V, kind):
    if kind == 'normal':
        return (rng.standard_normal((R, V)) * 3).astype(np.float32)
    if kind == 'ties':                                   # few distinct values: ties everywhere, at every cut
        return rng.integers(-4, 6, (R, V)).astype(np.float32)
    if kind == 'peaky':
        x = (rng.standard_normal((R, V)) * 1).astype(np.float32)
        x[:, rng.integers(0, V, 3)] += 12
        return x
    if kind == 'nonfinite':
        x = (rng.standard_normal((R, V)) * 3).astype(np.float32)
        x[0, rng.integers(0, V)] = np.nan
        x[-1, rng.integers(0, V)] = np.inf
        return x
    raise ValueError(kind)


def outcome(f):
    """f()'s value, or ('raised', the exception type)"""
    try:
        return f()
    except Exception as e:              # noqa: BLE001
        return ('raised', type(e).__name__)


def cand_eq(a, b):
    return (np.array_equal(a.ids, b.ids) and a.logits.tobytes() == b.logits.tobytes() and a.probs.tobytes() == b.probs.tobytes()
            and a.weights.tobytes() == b.weights.tobytes())


class TestCandidates(unittest.TestCase):
    def test_distribution_equal(self):
        rng = np.random.default_rng(0)
        V, vh = 3000, 1400
        n = cand = 0
        for it in range(600):
            kind = ('normal', 'ties', 'peaky', 'nonfinite')[it % 4]
            prm = S.SamplingParams(temperature=float(rng.choice([0.0, 0.6, 1.0, 1.7])), top_k=int(rng.choice([1, 5, 20, 40, 64])),
                                   top_p=float(rng.choice([1.0, 0.95, 0.5])), min_p=float(rng.choice([0.0, 0.05, 0.2])),
                                   repeat_penalty=float(rng.choice([1.0, 1.1, 0.9])), presence_penalty=float(rng.choice([0.0, 1.5, -1.0])),
                                   frequency_penalty=float(rng.choice([0.0, 0.5, -0.4])), penalty_last_n=int(rng.choice([0, 64, -1, 8])),
                                   seed=int(rng.integers(0, 2 ** 63))).resolve()
            hist = [int(x) for x in rng.integers(0, V, int(rng.integers(1, 120)))]
            if it % 3 == 0:                              # history ids that sit at the cards' cuts
                hist += [vh - 1, vh, 0, V - 1]
            drafts = [int(x) for x in rng.integers(0, V, 3)]
            full = rows_for(rng, 4, V, kind)
            for order in ('high', 'low'):
                rows = CS.emulate_rows(full, vh, prm, hist, drafts, tie_order=order)
                for j in range(4):
                    h = hist + drafts[:j]
                    ref_tok = outcome(lambda: S.Sampler(prm).sample(full[j], h))
                    r = rows[j]
                    exact = not CS.needs_full(r, prm, h)
                    cand += exact
                    n += 1
                    if not exact:
                        r.full = full[j]
                    if not prm.greedy and not isinstance(ref_tok, tuple):
                        self.assertTrue(cand_eq(CS.distribution(r, prm, h), S.filter_chain(full[j], prm, h)), (it, kind, order, j))
                    # (the reference raises on some non-finite rows: the candidate path then took the full row and raises the same)
                    self.assertEqual(outcome(lambda: CS.sample(r, S.Sampler(prm), h)), ref_tok, (it, kind, order, j))
        self.assertGreater(cand, n // 3)
        print(f'\n[candsample] {n} rows: the candidates == the full row bitwise (distribution and token); {cand} from the '
              f'candidates, {n - cand} took the full-row fallback (ties at a cut, non-finite values)')

    def test_draw_fast(self):
        """draw_fast == sampler.draw(filter_chain(...)) on candidate sets (ties, top_p / min_p / min_keep, temperatures)"""
        rng = np.random.default_rng(9)
        n = 0
        for it in range(3000):
            m = int(rng.integers(1, 90))
            ids = rng.choice(5000, m, replace=False).astype(np.int64)
            l = (rng.integers(-6, 6, m) if it % 3 == 0 else rng.standard_normal(m) * 3).astype(np.float32)
            prm = S.SamplingParams(temperature=float(rng.choice([0.3, 0.6, 1.0, 1.7])), top_k=int(rng.integers(1, 65)),
                                   top_p=float(rng.choice([1.0, 0.95, 0.5, 0.1])), min_p=float(rng.choice([0.0, 0.05, 0.3])),
                                   min_keep=int(rng.choice([0, 0, 1, 3])), seed=1).resolve()
            k = min(prm.top_k, m)
            s_k = np.sort(l)[::-1][k - 1]
            keep = l >= s_k                               # (what candidates() hands over: top_k's survivors and their ties)
            ci, cl = ids[keep], l[keep]
            o = np.argsort(ci, kind='stable'); ci, cl = ci[o], cl[o]
            for u in (0.0, 0.3, 0.999999, float(rng.random())):
                ref = S.draw(S.filter_chain(cl, prm, (), ids=ci), u)
                self.assertEqual(CS.draw_fast(ci, cl, prm, u), ref, (it, u))
                n += 1
        print(f'\n[candsample] draw_fast == draw(filter_chain) on {n} draws')

    def test_batched(self):
        """draws_batched / sample_and_match_batched == the per-row path (penalties off; ties at the cuts and NaN rows take the
        full row through fetch)"""
        rng = np.random.default_rng(11)
        V, vh = 3000, 1400
        n = fb = 0
        for it in range(500):
            prm = S.SamplingParams(temperature=float(rng.choice([0.3, 0.7, 1.0, 1.6])), top_k=int(rng.choice([1, 5, 20, 40, 64])),
                                   top_p=float(rng.choice([1.0, 0.95, 0.5])), min_p=float(rng.choice([0.0, 0.05, 0.2])),
                                   min_keep=int(rng.choice([0, 1, 3])), seed=int(rng.integers(0, 2 ** 62))).resolve()
            hist = [int(x) for x in rng.integers(0, V, 40)]
            full = rows_for(rng, 4, V, ('normal', 'ties', 'peaky', 'nonfinite')[it % 4])
            smp = S.Sampler(prm)
            drafts = [smp.sample(full[0], hist)] if it % 2 else [int(rng.integers(0, V))]
            drafts += [int(rng.integers(0, V)) for _ in range(2)]
            rows = CS.emulate_rows(full, vh, prm, hist, drafts, tie_order='high')
            V0 = np.stack([r.cards[0][0] for r in rows]); I0 = np.stack([r.cards[0][1] for r in rows])
            V1 = np.stack([r.cards[1][0] for r in rows]); I1 = np.stack([r.cards[1][1] for r in rows])
            box = []
            def fetch(j):
                box.append(j); return full[j]
            ref = outcome(lambda: smp.sample_and_match(full, drafts, hist, ()))
            got = outcome(lambda: CS.sample_and_match_batched(V0, I0, V1, I1, smp, drafts, hist, (), fetch=fetch))
            self.assertEqual(got, ref, it)
            us = [smp.uniform(len(hist) + j) for j in range(4)]
            toks, ok = CS.draws_batched(V0, I0, V1, I1, prm, us)
            for j in range(4):
                if ok[j]:
                    self.assertEqual(int(toks[j]), smp.sample(full[j], hist + drafts[:j]), (it, j))
                    n += 1
                else: fb += 1
        print(f'\n[candsample] batched draws == the per-row reference: {n} rows from the candidates, {fb} flagged for the full row; '
              f'sample-and-match on 500 passes equal')

    def test_full_row_needed(self):
        prm = S.SamplingParams(temperature=1.0, top_k=0).resolve()
        self.assertIsNone(CS.k_prime(prm, 0))
        prm = S.SamplingParams(temperature=1.0, top_k=65).resolve()
        self.assertIsNone(CS.k_prime(prm, 0))
        rows = CS.emulate_rows(np.zeros((2, 100), np.float32), 50, prm, [1], [])
        self.assertTrue(CS.needs_full(rows[0], prm, [1]))
        with self.assertRaises(LookupError):
            CS.sample(rows[0], S.Sampler(prm), [1])

    def test_sample_and_match_equal(self):
        rng = np.random.default_rng(1)
        V, vh = 2048, 1024
        fetched = 0
        for it in range(300):
            prm = S.SamplingParams(temperature=float(rng.choice([0.0, 0.7, 1.0])), top_k=20, top_p=0.95,
                                   presence_penalty=float(rng.choice([0.0, 1.5])), seed=int(rng.integers(0, 2 ** 63))).resolve()
            hist = [int(x) for x in rng.integers(0, V, 50)]
            full = rows_for(rng, 4, V, ('peaky', 'ties', 'normal')[it % 3])
            # drafts the target likes (so several get accepted): the reference's own tokens, one corrupted at random
            smp = S.Sampler(prm)
            ref_free = smp.sample_and_match(full, [0, 0, 0], hist, ())
            drafts = [smp.sample(full[0], hist)]
            for j in range(1, 3):
                drafts.append(smp.sample(full[j], hist + drafts))
            if it % 2:
                drafts[int(rng.integers(0, 3))] = int(rng.integers(0, V))
            ref = smp.sample_and_match(full, drafts, hist, stop_ids=(5,))
            rows = CS.emulate_rows(full, vh, prm, hist, drafts)
            box = []

            def fetch(j):
                box.append(j)
                return full[j]
            got = CS.sample_and_match(rows, smp, drafts, hist, stop_ids=(5,), fetch=fetch)
            fetched += len(box)
            self.assertEqual(got, ref, it)
            self.assertIsNotNone(ref_free)
        print(f'\n[candsample] sample-and-match == the full-row reference on 300 passes ({fetched} rows fetched whole)')

    def test_host_time(self):
        rng = np.random.default_rng(2)
        V, vh = 248320, 124160
        prm = S.SamplingParams(temperature=0.6, top_k=20, top_p=0.95, seed=7).resolve()
        hist = [int(x) for x in rng.integers(0, V, 300)]
        full = (rng.standard_normal((4, V)) * 3).astype(np.float32)
        rows = CS.emulate_rows(full, vh, prm, hist, [1, 2, 3])
        smp = S.Sampler(prm)
        best = 1e9
        for _ in range(7):
            t0 = time.perf_counter()
            for _ in range(200):
                for j in range(4):
                    CS.sample(rows[j], smp, hist)
            best = min(best, (time.perf_counter() - t0) / 800)
        prm_p = S.SamplingParams(temperature=0.6, top_k=20, top_p=0.95, presence_penalty=1.5, seed=7).resolve()
        rows_p = CS.emulate_rows(full, vh, prm_p, hist, [1, 2, 3])
        smp_p = S.Sampler(prm_p)
        best_p = 1e9
        for _ in range(7):
            t0 = time.perf_counter()
            for _ in range(200):
                for j in range(4):
                    CS.sample(rows_p[j], smp_p, hist)
            best_p = min(best_p, (time.perf_counter() - t0) / 800)
        V0 = np.stack([r.cards[0][0] for r in rows]); I0 = np.stack([r.cards[0][1] for r in rows])
        V1 = np.stack([r.cards[1][0] for r in rows]); I1 = np.stack([r.cards[1][1] for r in rows])
        best_b = 1e9
        for _ in range(7):
            t0 = time.perf_counter()
            for _ in range(200):
                CS.draws_batched(V0, I0, V1, I1, prm, [0.1, 0.2, 0.3, 0.4])
            best_b = min(best_b, (time.perf_counter() - t0) / 200)
        print(f'\n[candsample] batched draws of 4 rows (penalties off): {best_b * 1e6:.1f} us per pass')
        print(f'\n[candsample] host time per row (top_k 20, T = 0.6, best of 7): {best * 1e6:.1f} us; with presence_penalty 1.5 '
              f'(64-id window): {best_p * 1e6:.1f} us')


def _old_window(history, drafts, last_n):
    """the penalty window as the code before the tail took it (the whole history copied and converted, then cut): the oracle"""
    h = list(history) + list(drafts)
    if last_n > 0:
        h = h[-(last_n + len(drafts)):]
    return h


# the tokens K3Session's sampled passes emitted in TestSessionTail on the code before the penalty tail (the whole history
# copied into every sampler call): sha256 of the emitted ids, per case
SESSION_TAIL_SHA = {
    'presence+frequency, window 64, T 4': '1700bdc4bfe49d36',
    'presence+frequency, whole history (-1), T 4': 'd094f3bd3b1f2d32',
    'window past the history, repeat 1.1, T 1': '58e8458d95205f58',
    'greedy with penalties, window 8, T 4': '44e7a32aee443dfb',
    'penalties off, batched, ties, T 4': '313df32e6516f208',
    'presence, top_k 0 (whole rows), T 4': '335656c433886b40',
}


class TestSessionTail(unittest.TestCase):
    """K3Session's sampled pass at a long history (its own _seq, _penalty_rows and _sample_rows on a stand-in whose cards
    are CS.emulate_rows of the full rows): the samplers take the penalty window's tail, the token's position travels apart
    from it, and the emitted tokens stay the full-row reference's (sampler.Sampler on the whole history) and the tokens the
    code before the tail emitted (SESSION_TAIL_SHA)"""
    V, VH = 3000, 1400

    def _row(self, pos, hist, kind):
        r = np.random.default_rng(pos * 7919 + 13)
        if kind == 'ties' and pos % 5 == 0:                 # (ties at the cards' cuts: the full-row fallback)
            x = r.integers(-3, 4, self.V).astype(np.float32)
        else:
            x = (r.standard_normal(self.V) * 2).astype(np.float32)
        x[np.asarray(hist[-24:], dtype=np.int64)] += 3.0     # (recent ids favoured: the penalties decide draws)
        return x

    def _run(self, prm, T, n0, steps, kind, seed):
        import collections
        from types import SimpleNamespace
        from caminus.serve.k3_session import K3Session
        rng = np.random.default_rng(seed)
        hist = [int(x) for x in rng.integers(0, self.V, n0)]
        hist[-200:] = [int(x) for x in rng.integers(0, 40, min(200, n0))]   # (a repetitive tail: real counts in the window)
        smp = S.Sampler(prm)
        s = K3Session.__new__(K3Session)
        s.T, s.kmax, s.samp_t = T, 512, collections.deque(maxlen=8)
        s.counters = collections.Counter()
        box = {}
        s.pc = SimpleNamespace(live_tokens=lambda: box['toks'])
        s.rows_wait = lambda R, pid, kp: box['rows'][:R]
        s.cards_wait = lambda R, kp: tuple(np.stack([r.cards[c][i] for r in box['rows'][:R]]) for c in (0, 1) for i in (0, 1))
        s.full_row = lambda r, src=None: box['full'][r - box['F'] + 1]
        out_all = []
        for step in range(steps):
            F = 1
            k = min(T - F, 3) if step % 3 else 0
            pos = len(hist)
            ref_free = [smp.sample(self._row(pos, hist, kind), hist)] if k else []
            drafts = []
            for j in range(k):                               # (drafts the target likes, one corrupted now and then)
                drafts.append(int(ref_free[-1]) if j == 0 else int(smp.sample(self._row(pos + j, hist + drafts, kind), hist + drafts)))
            if k and step % 4 == 1:
                drafts[int(rng.integers(0, k))] = int(rng.integers(0, self.V))
            full = np.stack([self._row(pos + j, hist + drafts[:j], kind) for j in range(k + 1)])
            toks, feed = hist[:-F], hist[-F:]
            box.update(toks=toks, full=full, F=F, rows=CS.emulate_rows(full, self.VH, prm, hist, drafts, tie_order='high'))
            i0, seq = s._seq(feed, drafts)
            body = feed + drafts
            old = toks + body + [body[-1]] * (T - len(body))
            self.assertEqual(len(seq), len(old))
            self.assertEqual(list(seq[i0:i0 + T]), old[i0:i0 + T])
            self.assertEqual(list(seq[max(0, i0 - 5):i0 + T]), old[max(0, i0 - 5):i0 + T])
            self.assertEqual([seq[i] for i in (0, i0 - 1, i0, i0 + T - 1)], [old[i] for i in (0, i0 - 1, i0, i0 + T - 1)])
            pid, kp = s._penalty_rows(prm, hist, drafts)
            if prm.penalties_on:
                self.assertTrue(np.array_equal(pid, np.unique(np.asarray(_old_window(hist, drafts, prm.penalty_last_n), np.int64))))
            got = s._sample_rows(F, k, drafts, smp, hist, frozenset(), pid, kp, False)
            ref = smp.sample_and_match(full, drafts, list(hist), ())
            self.assertEqual(got, ref, (step, k))
            out_all += got
            hist += got
        return out_all

    def test_tail_equal(self):
        import hashlib
        cases = {
            'presence+frequency, window 64, T 4': (S.SamplingParams(temperature=0.6, top_k=20, top_p=0.95, presence_penalty=1.5,
                                                                    frequency_penalty=0.5, seed=7), 4, 3000, 'normal'),
            'presence+frequency, whole history (-1), T 4': (S.SamplingParams(temperature=0.8, top_k=40, presence_penalty=1.0,
                                                                             frequency_penalty=0.3, penalty_last_n=-1, seed=8), 4, 1500, 'normal'),
            'window past the history, repeat 1.1, T 1': (S.SamplingParams(temperature=1.0, top_k=20, repeat_penalty=1.1,
                                                                          presence_penalty=0.5, penalty_last_n=64, seed=9), 1, 30, 'normal'),
            'greedy with penalties, window 8, T 4': (S.SamplingParams(temperature=0.0, presence_penalty=1.5, frequency_penalty=0.5,
                                                                      penalty_last_n=8, seed=10), 4, 3000, 'ties'),
            'penalties off, batched, ties, T 4': (S.SamplingParams(temperature=0.7, top_k=20, top_p=0.95, seed=11), 4, 3000, 'ties'),
            'presence, top_k 0 (whole rows), T 4': (S.SamplingParams(temperature=0.7, top_k=0, presence_penalty=1.5, seed=12),
                                                    4, 3000, 'normal'),
        }
        for name, (prm, T, n0, kind) in cases.items():
            toks = self._run(prm.resolve(), T, n0, 60, kind, seed=len(name))
            sha = hashlib.sha256(repr(toks).encode()).hexdigest()
            print(f'\n[candsample] session tail: {name}: {len(toks)} tokens == the full-row reference, sha {sha[:16]}')
            self.assertEqual(sha[:16], SESSION_TAIL_SHA[name], name)

    def test_window_oracle(self):
        """sampler.penalty_counts and candsample.penalty_ids == the whole-history conversion cut afterwards (lists, tuples,
        arrays; every window width against the history's length)"""
        rng = np.random.default_rng(5)
        for it in range(400):
            n = int(rng.integers(0, 300))
            h = [int(x) for x in rng.integers(0, 50, n)]
            hist = (h, tuple(h), np.asarray(h, dtype=np.int64))[it % 3]
            drafts = [int(x) for x in rng.integers(0, 50, int(rng.integers(0, 4)))]
            for last_n in (-1, 0, 1, 8, 64, n, n + 5):
                w = _old_window(hist, (), last_n) if last_n != 0 else []
                ri, rc = np.unique(np.asarray(w, dtype=np.int64), return_counts=True)
                gi, gc = S.penalty_counts(hist, last_n)
                self.assertTrue(np.array_equal(gi, ri) and np.array_equal(gc, rc), (it, last_n))
                prm = S.SamplingParams(presence_penalty=1.0, penalty_last_n=last_n).resolve()
                pid = CS.penalty_ids(prm, hist, drafts)
                want = np.unique(np.asarray(_old_window(hist, drafts, last_n), np.int64)) if prm.penalties_on else np.zeros(0, np.int64)
                self.assertTrue(np.array_equal(pid, want), (it, last_n))


if __name__ == '__main__':
    unittest.main()
