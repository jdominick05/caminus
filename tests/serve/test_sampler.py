"""S3 gates (CPU only, MockEngine): Philox known answers, filter-chain parity with llama.cpp's sampler on fixed logits,
chi-square of sampled frequencies against the filtered distribution, seeded determinism, and output with drafting ==
without drafting. Needs build/sampler_ref and build/philox_ref (tools/build_sampler_ref.sh; built on demand).
K3_SAMPLER_ROWS sets the parity row count (default 1000, full 248320-token vocab)."""
import math
import os
import struct
import subprocess
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from caminus.serve import sampler as S  # noqa: E402
from caminus.serve.mock_engine import EngineStateError, MockDrafter, MockEngine, MockQDrafter  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SAMPLER_REF = os.path.join(ROOT, 'build', 'sampler_ref')
PHILOX_REF = os.path.join(ROOT, 'build', 'philox_ref')
N_VOCAB = 248320
ROWS = int(os.environ.get('K3_SAMPLER_ROWS', '1000'))


def _need_refs(test):
    """skip `test` when the references cannot be built here: tools/build_sampler_ref.sh links sampler_ref against the
    ignite-llama fork's CPU libllama ($IGNITE/build-cpu/bin; absent on a machine without the fork, e.g. CI)"""
    if os.path.exists(SAMPLER_REF) and os.path.exists(PHILOX_REF):
        return
    lib = os.path.join(os.environ.get('IGNITE', os.path.expanduser('~/src/ignite-llama')), 'build-cpu', 'bin', 'libllama.so')
    if not os.path.exists(lib):
        test.skipTest(f'no build/sampler_ref + build/philox_ref and no {lib} to build them from (tools/build_sampler_ref.sh)')


def _ensure_refs():
    if not (os.path.exists(SAMPLER_REF) and os.path.exists(PHILOX_REF)):
        subprocess.run([os.path.join(ROOT, 'tools', 'build_sampler_ref.sh')], check=True, capture_output=True)


# ---------------------------------------------------------------- chi-square without scipy

def _gammq(a, x):
    """regularized upper incomplete gamma Q(a, x) (series / Lentz continued fraction)"""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        s = term = 1.0 / a
        ap = a
        for _ in range(10000):
            ap += 1; term *= x / ap; s += term
            if abs(term) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - gln))
    b = x + 1 - a; c = 1e300; d = 1 / b; h = d
    for i in range(1, 10000):
        an = -i * (i - a); b += 2
        d = an * d + b; d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c; c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d; de = d * c; h *= de
        if abs(de - 1) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi2_test(counts, probs, min_expected=5.0):
    """Pearson chi-square of counts against probs; bins with expected < min_expected are pooled (tail bin).
    Returns (stat, dof, p_value)."""
    counts = np.asarray(counts, np.float64); probs = np.asarray(probs, np.float64)
    n = counts.sum()
    e = probs / probs.sum() * n
    big = e >= min_expected
    oc, ec = list(counts[big]), list(e[big])
    if (~big).any():
        oc.append(counts[~big].sum()); ec.append(e[~big].sum())
    oc, ec = np.array(oc), np.array(ec)
    stat = float(((oc - ec) ** 2 / ec).sum())
    dof = len(oc) - 1
    return stat, dof, _gammq(dof / 2.0, stat / 2.0)


# ---------------------------------------------------------------- fixtures

def llm_like_logits(rng, V, ties=False, shape=None):
    """LM-like rows. shape 'peaked': N(0, 2.5) body plus ~60 boosted tokens with an exponential spread (one or two
    tokens dominate, like code); 'flat_head': 60 head tokens within 4 nats (prose); 'diffuse': a body of N(0, 1) only
    (thousands of live tokens, exercises top_p's partial-sort extension with top_k = 0)"""
    shape = shape or ('peaked', 'flat_head', 'diffuse')[int(rng.integers(0, 3))]
    if shape == 'diffuse':
        l = rng.standard_normal(V).astype(np.float32)
    else:
        l = rng.standard_normal(V).astype(np.float32) * np.float32(2.5)
        head = rng.choice(V, 60, replace=False)
        if shape == 'peaked':
            l[head] += (rng.exponential(4.0, 60) + rng.uniform(4, 12)).astype(np.float32)
        else:
            l[head] = rng.uniform(8, 12, 60).astype(np.float32)
    if ties:
        l = (np.round(l * 4) / 4).astype(np.float32)
    return l


CONFIGS = [
    dict(name='defaults'),
    dict(name='ctf_temp0.6', temperature=0.6),
    dict(name='topk0_topp', top_k=0, top_p=0.95),
    dict(name='topk0_minp_unsorted', top_k=0, top_p=1.0, min_p=0.05),
    dict(name='k40_p0.9_minp', top_k=40, top_p=0.9, min_p=0.05, temperature=0.8),
    dict(name='k200_p0.99', top_k=200, top_p=0.99, temperature=1.2),
    dict(name='min_keep', top_k=50, top_p=0.1, min_p=0.5, min_keep=5),
    dict(name='repeat', repeat_penalty=1.3, penalty_last_n=64),
    dict(name='freq_pres', frequency_penalty=0.5, presence_penalty=0.8, penalty_last_n=16),
    dict(name='all_pen_whole_hist', repeat_penalty=1.1, frequency_penalty=0.2, presence_penalty=0.3,
         penalty_last_n=-1, top_k=0, top_p=0.9),
    dict(name='greedy', temperature=0.0),
    dict(name='greedy_pen', temperature=0.0, repeat_penalty=1.5, presence_penalty=1.0),
    dict(name='ties', ties=True),
    dict(name='ties_k0', ties=True, top_k=0, top_p=0.8),
]


def _params(cfg):
    kw = {k: v for k, v in cfg.items() if k not in ('name', 'ties')}
    return S.SamplingParams(seed=1, **kw)


def _history(rng, logits, n=200):
    """a history weighted toward the head (so the penalties hit candidates), including negative-logit tokens"""
    top = np.argsort(-logits)[:80]
    neg = np.nonzero(logits < 0)[0][:50]
    pool = np.concatenate([top, top, neg, rng.integers(0, len(logits), 40)])
    return rng.choice(pool, n).astype(np.int64)


class SamplerRef:
    def __init__(self):
        _ensure_refs()
        self.p = subprocess.Popen([SAMPLER_REF], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def run(self, logits, prm, history):
        last_n = prm.penalty_last_n if prm.penalty_last_n >= 0 else len(history)  # common maps -1 to n_ctx
        hdr = struct.pack('<iififffffii', len(logits), len(history), prm.temperature, prm.top_k, prm.top_p,
                          prm.min_p, prm.repeat_penalty, prm.frequency_penalty, prm.presence_penalty, last_n,
                          prm.min_keep)
        self.p.stdin.write(hdr + np.asarray(history, '<i4').tobytes() + np.asarray(logits, '<f4').tobytes())
        self.p.stdin.flush()
        n = struct.unpack('<i', self.p.stdout.read(4))[0]
        ids = np.frombuffer(self.p.stdout.read(4 * n), '<i4').astype(np.int64)
        p = np.frombuffer(self.p.stdout.read(4 * n), '<f4')
        return ids, p

    def close(self):
        self.p.stdin.write(struct.pack('<i', 0)); self.p.stdin.close(); self.p.wait(10); self.p.stdout.close()


# struct format above: '<i i f i f f f f f i i' = n_vocab, n_hist, temp, top_k, top_p, min_p, rep, freq, pres,
# last_n, min_keep; the pack string must spell exactly that.
assert struct.calcsize('<iififffffii') == 4 * 11


# ---------------------------------------------------------------- tests

class TestPhilox(unittest.TestCase):
    def test_known_answers(self):
        # Random123 kat_vectors, philox4x32_10
        kat = [((0, 0, 0, 0), (0, 0), (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)),
               ((0xffffffff,) * 4, (0xffffffff,) * 2, (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)),
               ((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0),
                (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1))]
        for ctr, key, want in kat:
            self.assertEqual(S.philox4x32(ctr, key), want)
            got = S.philox4x32_np(*ctr, *key)
            self.assertEqual(tuple(int(x) for x in got), want)

    def test_torch_engine(self):
        """2000 blocks equal torch's at::philox_engine(seed, subsequence=stream, offset=position)"""
        _need_refs(self)
        _ensure_refs()
        rng = np.random.default_rng(7)
        trip = [(int(rng.integers(0, 2**63)) * 2 + int(rng.integers(0, 2)), int(rng.integers(0, 4)),
                 int(rng.integers(0, 2**40)) if i % 3 else i) for i in range(2000)]
        trip[0] = (2**64 - 1, 3, 2**64 - 1)
        out = subprocess.run([PHILOX_REF], input=''.join(f'{a} {b} {c}\n' for a, b, c in trip), text=True,
                             capture_output=True, check=True).stdout.split('\n')
        for (seed, sub, off), line in zip(trip, out):
            k0, k1 = S.seed_key(seed)
            got = S.philox4x32((off & 0xFFFFFFFF, off >> 32, sub, 0), (k0, k1))
            self.assertEqual(' '.join('%08x' % w for w in got), line)

    def test_uniform(self):
        pos = np.arange(0, 50000, 7, dtype=np.uint64) + np.uint64(2**33)
        un = S.uniform_np(12345, pos)
        us = np.array([S.uniform(12345, int(p)) for p in pos[:500]])
        self.assertTrue(np.array_equal(un[:500], us))
        self.assertTrue((un >= 0).all() and (un < 1).all())
        stat, dof, pv = chi2_test(np.histogram(un, 64, (0, 1))[0], np.ones(64))
        self.assertGreater(pv, 1e-3, (stat, dof))
        self.assertNotEqual(S.uniform(1, 5, 0), S.uniform(1, 5, 1))     # streams differ
        self.assertNotEqual(S.uniform(1, 5), S.uniform(2, 5))           # seeds differ


class TestChainParity(unittest.TestCase):
    """the filtered set equals llama.cpp's and |dp| <= 1e-6. A set difference passes only when the sets have the same
    size and every differing token's penalized logit equals the cut logit exactly (llama.cpp's partial_sort leaves
    tie order unspecified; ours keeps the lowest ids)"""

    def test_llama_cpp_parity(self):
        _need_refs(self)
        ref = SamplerRef()
        rng = np.random.default_rng(2026)
        n_eq = n_tie = n_tie_chance = 0
        logic, maxdp, per_cfg, ncand = [], 0.0, {}, []
        t_ours = 0.0
        try:
            for r in range(ROWS):
                cfg = CONFIGS[r % len(CONFIGS)]
                prm = _params(cfg)
                l = llm_like_logits(rng, N_VOCAB, ties=cfg.get('ties', False))
                hist = _history(rng, l, int(rng.integers(0, 300)))
                rid, rp = ref.run(l, prm, hist)
                keep = rp > 0
                rid, rp = rid[keep], rp[keep]
                t0 = time.perf_counter()
                c = S.filter_chain(l, prm, hist)
                t_ours += time.perf_counter() - t0
                ncand.append(len(rid))
                if set(rid.tolist()) == set(c.ids.tolist()):
                    n_eq += 1
                    o = np.argsort(rid)
                    dp = float(np.abs(rp[o] - c.probs).max())
                    maxdp = max(maxdp, dp)
                    if dp > 1e-6:
                        logic.append((r, cfg['name'], f'dp={dp:.2e}'))
                    per_cfg[cfg['name']] = per_cfg.get(cfg['name'], 0) + 1
                    continue
                # a set difference is acceptable only when every differing token ties with the cut logit
                pl = S.apply_penalties(np.arange(N_VOCAB), l, hist, prm)
                diff = set(rid.tolist()) ^ set(c.ids.tolist())
                cut = min(pl[c.ids].min(), pl[rid].min())
                # (exact float32 ties also occur by chance in a 248K-token row: birthday collisions in the body)
                if len(rid) == len(c.ids) and all(pl[t] == cut for t in diff):
                    dp = float(np.abs(np.sort(rp) - np.sort(c.probs)).max())    # tied tokens carry equal p
                    maxdp = max(maxdp, dp)
                    if dp > 1e-6:
                        logic.append((r, cfg['name'], f'tie row dp={dp:.2e}'))
                    n_tie += 1
                    n_tie_chance += not cfg.get('ties')
                else:
                    logic.append((r, cfg['name'], f'|ref|={len(rid)} |ours|={len(c.ids)} diff={sorted(diff)[:6]}'))
        finally:
            ref.close()
        print(f'\n[S3 parity] rows={ROWS} V={N_VOCAB}: equal sets {n_eq}, tie-only diffs {n_tie} ({n_tie_chance} chance ties), '
              f'logic misses {len(logic)}, max |dp| {maxdp:.2e}, ours {1e3 * t_ours / ROWS:.2f} ms/row (host numpy); '
              f'surviving candidates: median {int(np.median(ncand))}, >=10 in {sum(n >= 10 for n in ncand)} rows, '
              f'>=1000 in {sum(n >= 1000 for n in ncand)} rows, max {max(ncand)}')
        self.assertEqual(logic, [])
        self.assertEqual(n_eq + n_tie, ROWS)
        self.assertGreaterEqual(len(per_cfg), len(CONFIGS) - 2)

    def test_candidate_subset_penalties(self):
        """penalties applied to a candidate subset (the split top-k path) equal the full-row result on those ids"""
        rng = np.random.default_rng(3)
        l = llm_like_logits(rng, 5000)
        hist = _history(rng, l)
        prm = S.SamplingParams(repeat_penalty=1.3, frequency_penalty=0.4, presence_penalty=0.2)
        full = S.apply_penalties(np.arange(5000), l, hist, prm)
        sub = rng.choice(5000, 700, replace=False)
        self.assertTrue(np.array_equal(S.apply_penalties(sub, l[sub], hist, prm), full[sub]))


class TestChiSquare(unittest.TestCase):
    """10^5 seeded draws per configuration against the filtered distribution"""
    N = 100000

    def _draws(self, c, seed):
        u = S.uniform_np(seed, np.arange(self.N, dtype=np.uint64))
        return S.draw_np(c, u), u

    def test_frequencies(self):
        rng = np.random.default_rng(11)
        rows = []
        for cfg in (dict(), dict(temperature=0.6), dict(top_k=0, top_p=0.9), dict(top_k=40, min_p=0.05),
                    dict(repeat_penalty=1.3, presence_penalty=0.5)):
            prm = S.SamplingParams(seed=5, **cfg)
            l = llm_like_logits(rng, N_VOCAB, shape='flat_head')            # many live candidates
            hist = _history(rng, l, 100)
            c = S.filter_chain(l, prm, hist)
            y, u = self._draws(c, 1000 + len(rows))
            ys = np.array([S.draw(c, float(x)) for x in u[:2000]])
            self.assertTrue(np.array_equal(ys, y[:2000]))                  # scalar draw == vectorized
            cnt = np.array([(y == t).sum() for t in c.ids])
            self.assertEqual(int(cnt.sum()), self.N)                       # never outside the filtered set
            stat, dof, pv = chi2_test(cnt, c.probs)
            rows.append((cfg, len(c.ids), stat, dof, pv))
        for cfg, n, stat, dof, pv in rows:
            print(f'[S3 chi2] {cfg or "defaults"}: candidates={n} chi2={stat:.1f} dof={dof} p={pv:.3f}')
            self.assertGreater(pv, 1e-3)
            self.assertGreaterEqual(dof, 9)                                 # not a vacuous one- or two-bin test

    def test_power(self):
        """the test rejects a wrong distribution: draws at temperature 1.0 vs the temperature 0.8 filtered probs"""
        rng = np.random.default_rng(12)
        l = llm_like_logits(rng, 4000, shape='flat_head')
        c1 = S.filter_chain(l, S.SamplingParams(temperature=1.0))
        c2 = S.filter_chain(l, S.SamplingParams(temperature=0.8))
        self.assertTrue(np.array_equal(c1.ids, c2.ids))
        y, _ = self._draws(c1, 77)
        cnt = np.array([(y == t).sum() for t in c1.ids])
        stat, dof, pv = chi2_test(cnt, c2.probs)
        print(f'[S3 chi2 power] T=1.0 draws vs T=0.8 probs: chi2={stat:.1f} dof={dof} p={pv:.2e}')
        self.assertLess(pv, 1e-6)

    def test_gammq(self):
        self.assertAlmostEqual(_gammq(0.5, 3.841459 / 2), 0.05, places=5)      # chi2 sf(3.841, 1)
        self.assertAlmostEqual(_gammq(5.0, 18.307038 / 2), 0.05, places=5)     # chi2 sf(18.307, 10)
        self.assertAlmostEqual(_gammq(50.0, 124.342113 / 2), 0.05, places=5)   # chi2 sf(124.342, 100)
        self.assertAlmostEqual(_gammq(2.0, 13.276704 / 2), 0.01, places=5)     # chi2 sf(13.277, 4)


class TestDeterminism(unittest.TestCase):
    def _run(self, seed, T=1, draft_q=None, temp=1.0, prompt_len=40, **kw):
        m = MockEngine(ntok=T, vocab=512, eos_id=7, eos_every=60, seed=3)
        sp = S.Sampler(S.SamplingParams(seed=seed, temperature=temp, **kw))
        d = MockDrafter(m, draft_q) if draft_q is not None else None
        st = {}
        toks, commit = S.sampled_generate(m, sp, list(range(100, 100 + prompt_len)), 300, draft=d, stop_ids=(7,),
                                          stats=st)
        m.flush(commit)
        return toks, m.state, st, sp

    def test_seeded_repeatable(self):
        a = self._run(42)[0]
        self.assertEqual(a, self._run(42)[0])
        self.assertNotEqual(a, self._run(43)[0])

    def test_random_seed_reported(self):
        sp = S.Sampler(S.SamplingParams())
        self.assertIsNotNone(sp.seed)
        self.assertNotEqual(sp.seed, S.Sampler(S.SamplingParams(seed=-1)).seed)
        toks = self._run(sp.seed)[0]
        self.assertEqual(toks, self._run(sp.seed)[0])

    def test_greedy_fast_path(self):
        rng = np.random.default_rng(5)
        l = llm_like_logits(rng, 3000)
        prm = S.SamplingParams(temperature=0.0)
        self.assertEqual(S.greedy_token(l, prm), int(np.argmax(l)))
        hist = [int(np.argmax(l))] * 3
        pen = S.SamplingParams(temperature=0.0, repeat_penalty=100.0)
        self.assertNotEqual(S.greedy_token(l, pen, hist), int(np.argmax(l)))
        self.assertEqual(S.greedy_token(l, pen, hist), int(S.filter_chain(l, pen, hist).ids[0]))


class TestDraftEquality(unittest.TestCase):
    """seeded output with drafting == without, token for token, across T, draft quality, temperature and penalties;
    the engine state after the final lazy commit equals the plain run's"""

    def test_sweep(self):
        keys = ('accepted', 'multi_accept_passes', 'stop_in_draft', 'gen_tokens', 'drafted', 'boundary_draft_passes')
        tot = dict.fromkeys(keys + ('runs',), 0)
        # prompt lengths 37, 42, 43, 47 leave 1-3 tokens (T=4) and 2, 3, 5, 7 tokens (T=8) for the prompt's last pass,
        # so that pass also carries drafts (forced > 1, rows start at forced - 1)
        for temp, pen in ((1.0, {}), (0.6, {}), (0.0, {}), (1.0, dict(repeat_penalty=1.2, presence_penalty=0.4))):
            for seed, plen in ((1, 37), (2, 42), (3, 43), (4, 47)):
                pen = dict(pen, prompt_len=plen)
                base = TestDeterminism()._run(seed, T=1, temp=temp, **pen)
                self.assertEqual(base[1], tuple(base[0][:-1]))              # plain state = all but the last token
                for T in (2, 4, 8):
                    for q in (0.0, 0.5, 0.9, 1.0):
                        toks, state, st, _ = TestDeterminism()._run(seed, T=T, draft_q=q, temp=temp, **pen)
                        self.assertEqual(toks, base[0], (temp, pen, seed, T, q))
                        self.assertEqual(state, base[1], (temp, pen, seed, T, q))
                        for k in keys:
                            tot[k] += st[k]
                        tot['runs'] += 1
        print(f'\n[S3 draft==plain] runs={tot["runs"]} gen_tokens={tot["gen_tokens"]} drafted={tot["drafted"]} '
              f'accepted={tot["accepted"]} multi-accept passes={tot["multi_accept_passes"]} '
              f'drafted stops accepted={tot["stop_in_draft"]} prompt-boundary passes with drafts='
              f'{tot["boundary_draft_passes"]}')
        self.assertGreater(tot['accepted'], 1000)
        self.assertGreater(tot['multi_accept_passes'], 100)
        self.assertGreater(tot['stop_in_draft'], 0)
        self.assertGreater(tot['boundary_draft_passes'], 0)

    def test_engine_catches_bad_commit(self):
        m = MockEngine(ntok=4, vocab=64)
        m.verify([1, 2, 3, 4], 0, 0)
        with self.assertRaises(EngineStateError):
            m.verify([1, 2, 3, 4, 5, 6], 3, 2)


class TestQSampled(unittest.TestCase):
    """the min(1, p/q) hook: the emitted token is distributed as the filtered target p, for any drafter q"""

    def _check(self, q_fn, n=40000, name=''):
        rng = np.random.default_rng(21)
        V = 64
        l = (rng.standard_normal(V) * 1.5).astype(np.float32)
        prm = S.SamplingParams(seed=9, top_k=0, top_p=1.0)
        c = S.filter_chain(l, prm)
        ids, q = q_fn(rng, V, l)
        hist = [0]
        out = np.empty(n, np.int64)
        acc = 0
        for i in range(n):
            sp = S.Sampler(S.SamplingParams(seed=1000 + i, top_k=0, top_p=1.0))
            x = int(rng.choice(ids, p=q))
            out[i], ok = sp.accept_q(l, hist, x, ids, q)
            acc += ok
        cnt = np.array([(out == t).sum() for t in c.ids])
        stat, dof, pv = chi2_test(cnt, c.probs)
        print(f'[S3 q-hook {name}] n={n} accept={acc / n:.3f} chi2={stat:.1f} dof={dof} p={pv:.3f}')
        self.assertGreater(pv, 1e-3)
        return acc / n

    def test_arbitrary_q(self):
        def q_fn(rng, V, l):
            q = rng.dirichlet(np.ones(V) * 0.5)
            return np.arange(V), q
        self._check(q_fn, name='dirichlet q')

    def test_point_mass_q(self):
        """one-hot q (a deterministic draft) through the q rule: still distributed as p (but not token-equal)"""
        self._check(lambda rng, V, l: (np.array([int(np.argmax(l))]), np.array([1.0])), name='one-hot q (mode)')
        self._check(lambda rng, V, l: (np.array([int(np.argsort(-l)[5])]), np.array([1.0])), n=20000,
                    name='one-hot q (6th)')

    def test_generation_runs(self):
        """q-sampled drafting through the loop keeps the engine state consistent; greedy equals the plain run"""
        for temp in (1.0, 0.0):
            m = MockEngine(ntok=4, vocab=256, eos_id=7, eos_every=80, seed=4)
            sp = S.Sampler(S.SamplingParams(seed=3, temperature=temp))
            st = {}
            toks, commit = S.sampled_generate(m, sp, list(range(10, 33)), 150, draft=MockQDrafter(m),
                                              stop_ids=(7,), stats=st, q_draft=True)
            m.flush(commit)
            self.assertEqual(m.state, tuple(toks[:-1]))
            self.assertGreater(st['accepted'], 0)
            self.assertEqual(st['boundary_draft_passes'], 1)               # 23 tokens: the last prompt pass has 3
            if temp == 0.0:
                m1 = MockEngine(ntok=1, vocab=256, eos_id=7, eos_every=80, seed=4)
                plain, _ = S.sampled_generate(m1, S.Sampler(S.SamplingParams(seed=3, temperature=0.0)),
                                              list(range(10, 33)), 150, stop_ids=(7,))
                self.assertEqual(toks, plain)


class TestParams(unittest.TestCase):
    def test_gguf_defaults(self):
        tk = common.tokenizer()
        prm = S.SamplingParams.from_gguf_meta(tk.vocab.meta)
        self.assertEqual((prm.top_k, prm.temperature), (20, 1.0))
        self.assertAlmostEqual(prm.top_p, 0.95, places=6)
        self.assertEqual((prm.min_p, prm.repeat_penalty, prm.penalty_last_n), (0.0, 1.0, 64))

    def test_request_fields(self):
        prm = S.SamplingParams().with_request({'temperature': 0.6, 'seed': 77, 'top_k': None, 'repeat_last_n': 32,
                                               'presence_penalty': 0.5, 'stream': True})
        self.assertEqual((prm.temperature, prm.seed, prm.top_k, prm.penalty_last_n, prm.presence_penalty),
                         (0.6, 77, 20, 32, 0.5))
        self.assertTrue(prm.penalties_on)
        self.assertFalse(S.SamplingParams().penalties_on)


if __name__ == '__main__':
    unittest.main()
