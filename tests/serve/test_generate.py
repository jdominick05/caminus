"""S4 unit gates (CPU only): stop-string holdback under random chunking, UTF-8 streaming through TokenText, the
server's generation loop against S3's sampled_generate (plain and drafted, seeded), exact state after stops inside
accepted drafts, and MockSession prefix reuse / checkpoints / strictness."""
import os
import random
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from caminus.serve import sampler as S  # noqa: E402
from caminus.serve.engine import EngineError, MockSession  # noqa: E402
from caminus.serve.generate import StopMatcher, TokenText, run_generation  # noqa: E402
from caminus.serve.mock_engine import EngineStateError, MockDrafter, MockEngine  # noqa: E402

EOS = 7


def chunks(rng, s):
    out, i = [], 0
    while i < len(s):
        n = rng.choice([1, 1, 2, 3, 5, 8, 13])
        out.append(s[i:i + n])
        i += n
    return out


class TestStopMatcher(unittest.TestCase):
    def test_random_chunkings(self):
        rng = random.Random(0)
        alphabet = 'ab<>/\n é世🎉'
        n = 0
        for _ in range(3000):
            text = ''.join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
            stops = [''.join(rng.choice(alphabet) for _ in range(rng.randint(1, 4))) for _ in range(rng.randint(1, 3))]
            first = min([(text.find(s) + len(s), text.find(s), s) for s in stops if s in text], default=None)
            first = first and first[1:]                 # the match that completes first, then the earliest start
            m = StopMatcher(stops)
            out, hit = [], False
            for c in chunks(rng, text):
                r, hit = m.feed(c)
                out.append(r)
                self.assertLessEqual(m.held, max(map(len, stops)) - 1)
                if hit:
                    break
            if not hit:
                out.append(m.flush())
            got = ''.join(out)
            if first is None:
                self.assertFalse(hit)
                self.assertEqual(got, text)
            else:
                self.assertTrue(hit)
                self.assertEqual(got, text[:first[0]])
                self.assertIn(m.matched, stops)
                self.assertEqual(m.matched, first[1])
            n += 1
        print(f'\n[stop] {n} random texts x chunkings: released text == text before the first stop')

    def test_no_stops_passthrough(self):
        m = StopMatcher([])
        self.assertEqual(m.feed('abc'), ('abc', False))
        self.assertEqual(m.flush(), '')


class TestTokenText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = common.tokenizer()

    def test_utf8_split_tokens(self):
        """multi-byte characters split across byte-level tokens are released whole, never as U+FFFD"""
        tok = self.tok
        texts = ['héllo 世界 🎉🎉 Ωμέγα — 完成。✅', '🧑‍💻 emoji ZWJ, 𝔘𝔫𝔦𝔠𝔬𝔡𝔢, 한국어 텍스트', 'plain ascii']
        split = 0
        for t in texts:
            ids = tok.encode(t)
            split += sum(1 for i in ids if tok.token_bytes(i) and not _valid_utf8(tok.token_bytes(i)))
            tt = TokenText(tok)
            out = ''.join(tt.feed([i])[0] for i in ids) + tt.finish()
            self.assertEqual(out, t)
            self.assertNotIn('�', out)
        self.assertGreater(split, 0, 'no token ends inside a character: the test does not test anything')
        print(f'\n[utf8] {split} tokens end inside a multi-byte character; streamed text equal')

    def test_stop_across_tokens_and_eos(self):
        tok = self.tok
        text = 'The answer is 42.\nEND_OF_ANSWER and more'
        ids = tok.encode(text) + [tok.eos_id]
        tt = TokenText(tok, ['END_OF'], eos_ids=[tok.eos_id])
        got, keep_at = [], None
        for j, i in enumerate(ids):
            s, k = tt.feed([i])
            got.append(s)
            if k is not None:
                keep_at = j
                break
        self.assertEqual(''.join(got), 'The answer is 42.\n')
        self.assertEqual(tok.decode(ids[:keep_at + 1]).find('END_OF') >= 0, True)
        self.assertEqual(tok.decode(ids[:keep_at]).find('END_OF'), -1)
        tt2 = TokenText(tok, [], eos_ids=[tok.eos_id])
        self.assertEqual(tt2.feed(ids)[0] + tt2.finish(), text)      # EOS not decoded

    def test_specials_are_text(self):
        tok = self.tok
        ids = tok.encode('<think>\nr\n</think>\n\n<tool_call>\n<function=f>\n')
        tt = TokenText(tok, eos_ids=[248046, 248044])
        self.assertEqual(tt.feed(ids)[0], '<think>\nr\n</think>\n\n<tool_call>\n<function=f>\n')


def _valid_utf8(b):
    try:
        b.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


class TestGeneration(unittest.TestCase):
    def test_equals_sampled_generate(self):
        """the server loop emits exactly S3's reference tokens, with and without drafting, and leaves the state at
        prompt + generated[:-1]"""
        runs = 0
        for seed in range(24):
            prompt = [248044] + list(range(5, 22 + seed))
            prm = S.SamplingParams(temperature=[0.0, 0.7, 1.0][seed % 3], seed=1000 + seed)
            ref, _ = S.sampled_generate(MockEngine(ntok=1, vocab=512, eos_id=EOS, eos_every=97, seed=seed),
                                        S.Sampler(prm), prompt, 60, stop_ids=(EOS,))
            for T, q in ((1, None), (4, 0.9), (4, 1.0), (3, 0.5)):
                m = MockEngine(ntok=T, vocab=512, eos_id=EOS, eos_every=97, seed=seed)
                sess = MockSession(m, max_ctx=4096, eos_ids=(EOS,))
                r = run_generation(sess, prompt, S.Sampler(prm), 60, lambda ids: None,
                                   drafter=MockDrafter(m, quality=q) if q else None)
                self.assertEqual(prompt + r.gen_ids, ref, (seed, T, q))
                self.assertEqual(list(sess.tokens), ref[:-1])
                self.assertEqual(sess.plan(ref + [11, 12]), len(ref) - 1)
                runs += 1
        print(f'\n[gen] {runs} runs: server loop == sampled_generate, state == output[:-1]')

    def test_stop_inside_drafts_exact_state(self):
        """a stop (EOS or a stop string) on an accepted draft: emission ends there and the state equals a plain run
        stopped at the same token; the next turn reuses it"""
        hits = 0
        for seed in range(40):
            m = MockEngine(ntok=4, vocab=512, eos_id=EOS, eos_every=29, seed=seed, peak=(8.0, 9.0))
            sess = MockSession(m, max_ctx=4096, eos_ids=(EOS,))
            prompt = [248044 % 512] + list(range(30, 50))
            prm = S.SamplingParams(temperature=0.0)
            budget = random.Random(seed).randint(3, 40)
            seen = []

            def on_tokens(ids, seen=seen, budget=budget):
                seen.extend(ids)
                if len(seen) >= budget:          # a 'stop string' that completes at token `budget`
                    return len(ids) - (len(seen) - budget)
                return None
            r = run_generation(sess, prompt, S.Sampler(prm), 200, on_tokens, drafter=MockDrafter(m, quality=1.0))
            hits += r.stop_in_draft
            out = prompt + r.gen_ids
            self.assertEqual(list(sess.tokens), out[:-1])
            plain = MockEngine(ntok=1, vocab=512, eos_id=EOS, eos_every=29, seed=seed, peak=(8.0, 9.0))
            ps = MockSession(plain, max_ctx=4096, eos_ids=(EOS,))
            r2 = run_generation(ps, prompt, S.Sampler(prm), len(r.gen_ids), lambda ids: None)
            self.assertEqual(r2.gen_ids, r.gen_ids)
            self.assertEqual(ps.tokens, sess.tokens)
            nxt = out + [3, 4, 5]
            row = sess.prefill(nxt)
            self.assertIsNotNone(row)
            self.assertEqual(sess.last_prefill.cached_n, len(out) - 1)
        self.assertGreater(hits, 5)
        print(f'\n[gen] stops on accepted drafts: {hits} of 40 runs; state == plain run, next turn reuses it')

    def test_forced_tokens(self):
        m = MockEngine(ntok=4, vocab=512, eos_id=None, seed=3, peak=(30.0, 31.0))
        sess = MockSession(m, max_ctx=4096, eos_ids=(EOS,))
        trig = m.mode([1, 2, 3])          # make the first generated token the trigger
        r = run_generation(sess, [1, 2, 3], S.Sampler(S.SamplingParams(temperature=0.0)), 20, lambda ids: None,
                           force=(trig, [100, 101, 102, 103, 104, 105]), tool_open_id=999)
        self.assertEqual(r.gen_ids[:7], [trig, 100, 101, 102, 103, 104, 105])
        self.assertEqual(r.forced_n, 6)
        self.assertEqual(list(sess.tokens), [1, 2, 3] + r.gen_ids[:-1])

    def test_forced_tokens_in_stats(self):
        """M7 (the metrics verifier's defect): forced tool-call ids count in /slots' n_decoded / n_remain and in the context
        position exactly like sampled ids, also when a stop string cuts inside them"""
        from caminus.serve.stats import EngineStats
        forced = [100, 101, 102, 103, 104, 105]
        for cut in (None, 3):
            m = MockEngine(ntok=4, vocab=512, eos_id=None, seed=3, peak=(30.0, 31.0))
            st = EngineStats()
            sess = MockSession(m, max_ctx=4096, eos_ids=(EOS,), stats=st)
            trig = m.mode([1, 2, 3])

            class J:            # the worker's Job fields begin() reads
                id_task, prompt_ids, n_predict, slot_params = 0, [1, 2, 3], 20, None
                def queue_ms(self): return 0.0
            st.begin(J(), 4096)
            seen = []

            def on_tokens(ids):
                seen.extend(ids)
                if cut is not None and len(seen) > cut:
                    return len(ids) - (len(seen) - cut - 1)      # a 'stop string' completing at seen[cut]
                return None
            r = run_generation(sess, [1, 2, 3], S.Sampler(S.SamplingParams(temperature=0.0)), 20, on_tokens,
                               force=(trig, forced), tool_open_id=999)
            n = len(r.gen_ids)
            self.assertEqual(st.slot.n_decoded, n, cut)
            self.assertEqual(st.slot.n_remain, 20 - n, cut)
            self.assertEqual(st.position, 3 + n - 1, cut)      # (the first emitted token is n_decoded's, as without forcing)
            if cut is not None:
                self.assertEqual(r.finish, 'stop')
                self.assertEqual(r.gen_ids, [trig] + forced[:cut])
        print('\n[gen] forced tool-call ids in n_decoded / n_remain / position (whole, and cut by a stop string)')

    def test_forced_tokens_clipped_to_max_new(self):
        """the forced ids count against max_new: never more generated tokens than asked for (fix round: a forced
        call used to overrun max_tokens), and the strict session stays consistent for the next turn"""
        forced = [100, 101, 102, 103, 104, 105]
        for max_new in range(1, len(forced) + 4):
            for ntok in (1, 2, 4):
                m = MockEngine(ntok=ntok, vocab=512, eos_id=None, seed=3, peak=(30.0, 31.0))
                sess = MockSession(m, max_ctx=4096, eos_ids=(EOS,))
                trig = m.mode([1, 2, 3])
                seen = []

                def on_tokens(ids):
                    seen.extend(ids)

                r = run_generation(sess, [1, 2, 3], S.Sampler(S.SamplingParams(temperature=0.0)), max_new, on_tokens,
                                   force=(trig, forced), tool_open_id=999)
                self.assertLessEqual(len(r.gen_ids), max_new, (max_new, ntok))
                self.assertEqual(seen, r.gen_ids)
                self.assertEqual(r.finish, 'length')
                self.assertEqual(len(r.gen_ids), max_new)
                self.assertEqual(r.gen_ids, ([trig] + forced + r.gen_ids[len(forced) + 1:])[:max_new])
                self.assertEqual(r.forced_n, min(len(forced), max_new - 1))
                n_log = len(sess.tokens)
                self.assertEqual(list(sess.tokens), ([1, 2, 3] + r.gen_ids)[:n_log])
                sess.prefill([1, 2, 3] + r.gen_ids + [7])          # strict: raises if the state were off
        print('\n[gen] forced ids clipped to max_new: 1..9 x ntok 1/2/4, never over, state consistent')

    def test_forced_tokens_context_room(self):
        """max_new clamped by the context room below the forced length: prompt + generated <= max_ctx"""
        forced = list(range(100, 112))
        for room in range(1, 8):
            m = MockEngine(ntok=4, vocab=512, eos_id=None, seed=3, peak=(30.0, 31.0))
            sess = MockSession(m, max_ctx=3 + room, eos_ids=(EOS,))
            trig = m.mode([1, 2, 3])
            r = run_generation(sess, [1, 2, 3], S.Sampler(S.SamplingParams(temperature=0.0)), room, lambda ids: None,
                               force=(trig, forced), tool_open_id=999)
            self.assertEqual(len(r.gen_ids), room)
            self.assertLessEqual(3 + len(r.gen_ids), sess.max_ctx)


class TestMockSession(unittest.TestCase):
    def test_checkpoints_and_cold(self):
        m = MockEngine(ntok=4, vocab=512, seed=1)
        s = MockSession(m, max_ctx=4096, eos_ids=(EOS,))
        a = list(range(1, 50))
        s.prefill(a)
        self.assertEqual(s.last_prefill.restore, 'cold')
        s.verify([60]); s.accept(1); s.end()               # noqa: E702 - a one-token completion
        b = a + [60, 61, 62]
        s.prefill(b)
        self.assertEqual((s.last_prefill.restore, s.last_prefill.cached_n), ('live', len(a) + 1))
        c = a + [70, 71]                                   # diverges after a: checkpoint (b) at len(a)
        s.prefill(c)
        self.assertEqual((s.last_prefill.restore, s.last_prefill.cached_n), ('checkpoint', len(a)))
        d = [1, 2, 3, 99, 98]                              # nothing reusable: cold, no free rewind
        s.prefill(d)
        self.assertEqual((s.last_prefill.restore, s.last_prefill.cached_n), ('cold', 0))
        s.prefill(b)        # b's own prompt-end checkpoint is too long (the last token is processed for its logits)
        self.assertEqual((s.last_prefill.restore, s.last_prefill.cached_n), ('checkpoint', len(a) + 1))

    def test_strict(self):
        s = MockSession(MockEngine(ntok=4, vocab=512), max_ctx=64)
        s.prefill([1, 2, 3])
        with self.assertRaises(EngineStateError):
            s.accept(1)                                    # accept without verify
        s.verify([4, 5])
        with self.assertRaises(EngineStateError):
            s.verify([6])                                  # verify without accept
        with self.assertRaises(EngineStateError):
            s.accept(3)
        s.accept(1)
        with self.assertRaises(EngineStateError):
            s.prefill(list(range(100)))                    # over max_ctx

    def test_abort_keeps_state_and_checkpoints(self):
        """abort() after a host-side exception: an unaccepted pass is rolled back, the committed state and the
        checkpoints stay, and the next prompt that extends them reuses them"""
        s = MockSession(MockEngine(ntok=4, vocab=512, seed=2), max_ctx=4096, eos_ids=(EOS,))
        a = list(range(1, 40))
        s.prefill(a)
        s.verify([50, 51, 52]); s.accept(2)                # noqa: E702
        s.verify([52, 53])                                 # pending: the exception came before accept
        cps = s.checkpoints
        s.abort()
        self.assertEqual(list(s.tokens), a + [50, 51])
        self.assertEqual(s.checkpoints, cps)
        s.prefill(a + [50, 51, 60, 61])
        self.assertEqual((s.last_prefill.restore, s.last_prefill.cached_n), ('live', len(a) + 2))
        s.prefill([1, 2] + a)                              # elsewhere, then back: the prompt-end checkpoint of a
        s.prefill(a + [70])
        self.assertEqual((s.last_prefill.restore, s.last_prefill.cached_n), ('checkpoint', len(a)))
        s.abort()                                          # nothing pending: a no-op
        self.assertEqual(list(s.tokens), a + [70])

    def test_fault_and_cancel(self):
        s = MockSession(MockEngine(ntok=2, vocab=512), max_ctx=4096, fail_at={3})
        with self.assertRaises(EngineError):
            s.prefill(list(range(1, 20)))
        s.reset()
        ev = threading.Event()
        ev.set()
        self.assertIsNone(s.prefill(list(range(1, 20)), ev))
        self.assertTrue(s.last_prefill.cancelled)
        self.assertEqual(s.prefill(list(range(1, 20))).shape, (512,))


if __name__ == '__main__':
    unittest.main()
