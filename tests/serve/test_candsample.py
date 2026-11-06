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


if __name__ == '__main__':
    unittest.main()
