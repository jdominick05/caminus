"""The engine stats hook and recorder (caminus.serve.stats, agentmon) on MockSession, CPU only:
  * the recorder's pieces: Series windows / quantiles / buckets, the bandwidth floor (max vs sum), the content
    classifier, the User-Agent classes, the tool-call report, the loop detector;
  * the deterministic test mode (stats.VirtualClock + mock_engine.SynthProfile): a fixed request sequence through
    the engine worker gives exact synthetic timings, consistent counters (llama-server's totals == the requests'
    own numbers, per-position and per-class acceptance summing to the totals), every cache source (live, vram
    checkpoint, the stand-in disk tier, cold) and byte-identical /metrics text on a second run;
  * the cost: the recording a decode pass and a request add (microbenchmark, printed; the M7 <= 3 % gate is
    measured on the real engine).
Run:  HIP_VISIBLE_DEVICES= python -I tests/serve/run_all.py"""
import time
import unittest

import common
import srvutil as U
import synth
from caminus.serve import metrics as MX
from caminus.serve.agentmon import ContentClass, LoopDetector, harness_of, tool_call_report
from caminus.serve.engine import MockSession
from caminus.serve.generate import run_generation
from caminus.serve.mock_engine import MockDrafter, MockEngine, ScriptedEngine, SynthProfile
from caminus.serve.sampler import Sampler, SamplingParams
from caminus.serve.stats import (CLASSES, PASS_PARTS, EngineStats, PhysicsConfig, Series, VirtualClock)
from caminus.serve.toolparse import OutputParser, ToolSet
from caminus.serve.worker import EngineWorker, Job

SYS = {'role': 'system', 'content': 'You are a coding agent.'}


# ==================================================================================================== pieces
class TestPieces(unittest.TestCase):
    def test_series(self):
        s = Series('x', 'ms', window=4, edges=(1, 10))
        for v in (0.5, 5, 50, 7, 8, 9):
            s.add(v)
        self.assertEqual(s.window(), [50, 7, 8, 9])                  # the last 4, oldest first
        self.assertEqual(s.quantiles(), [8, 50])                    # nearest rank: ceil(0.5 * 4) = 2nd, 4th
        self.assertEqual(s.counts, [1, 4, 1])                       # <= 1, <= 10, > 10 since start
        self.assertEqual((s.count, s.sum, s.max), (6, 79.5, 50))
        snap = s.snapshot(hist=True)
        self.assertEqual((snap['p50'], snap['p95'], snap['window'], snap['hist']['le']), (8, 50, 4, [1, 10, '+Inf']))
        self.assertEqual(Series('e').quantiles(), [None, None])

    def test_floor(self):
        self.assertAlmostEqual(PhysicsConfig().floor_ms(), 3.60 / 881 * 1e3, places=9)     # 4.086: the XTX's
        self.assertAlmostEqual(PhysicsConfig(mode='sum').floor_ms(), 6.47 / 1655 * 1e3, places=9)  # 3.909
        self.assertEqual(round(PhysicsConfig().floor_ms(), 2), 4.09)

    def test_content_class(self):
        cc = ContentClass()
        seq = [('Let me think about it.\n', 'prose'), ('```py', 'code'), ('thon\nx = 1\n', 'code'),
               ('``', 'code'), ('`\nDone. ', 'prose'), ('```ba', 'code'), ('sh\nls -la\n', 'shell'), ('```', 'prose'),
               ('<tool_call>', 'tool'), ('\n<function=run_shell', 'tool'), ('_command>\n', 'shell'),
               ('<parameter=command>\nls\n```\n', 'shell'), ('</parameter>\n</function>\n</tool_call>', 'prose'),
               ('<tool_call>\n<function=edit_file>\n', 'tool'), ('</function>\n</tool_call>\n', 'prose')]
        for text, want in seq:
            self.assertEqual(cc.feed(text), want, repr(text))
        self.assertEqual(ContentClass().feed(''), 'prose')

    def test_harness(self):
        for ua, want in (('QwenCode/0.0.14 (linux; x64)', 'qwen-code'), ('claude-cli/2.1.3 (external, cli)', 'claude-code'),
                         ('Hermes-Agent/1.0', 'hermes'), ('opencode/0.3', 'opencode'), ('Anthropic/Python 0.71.0',
                                                                                          'anthropic-sdk'),
                         ('OpenAI/JS 5.1.0', 'openai-sdk'), ('curl/8.5.0', 'curl'), ('python-requests/2.32', 'python'),
                         ('Mozilla/5.0 "quoted"\n', 'other'), ('', 'none'), (None, 'none')):
            self.assertEqual(harness_of(ua), want, ua)

    def test_tool_call_report(self):
        tools = ToolSet(synth.TOOLS)
        p = OutputParser(tools, thinking=False)
        p.feed('<tool_call>\n<function=read_file>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>\n'
               '<tool_call>\n<function=nope>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n'
               '<tool_call>\n<function=edit_file>\n<parameter=path>\nb.py\n</parameter>\n<parameter=new_str>\nab')
        p.finish()
        msg = p.result()
        ok, fails, lenient = tool_call_report(msg, p.notes)
        cut = sum(c.cut for c in msg.tool_calls)
        self.assertEqual(cut, 1)
        self.assertEqual(fails['cut'], 1)
        self.assertEqual(ok, len(msg.tool_calls) - 1)
        self.assertGreaterEqual(fails['undeclared'] + fails['malformed'], 0)

    def test_loop_detector(self):
        live = []
        d = LoopDetector(True, on_live=live.append)
        line = 'I should check the file again before editing.'
        for piece in ('Plan.\n' + line[:10], line[10:] + '\n' + line + '\n', 'ok\n', line + '\n</think>\n',
                      line + '\n' + line + '\n'):          # after </think>: answer text, not counted
            d.feed(piece)
        flagged, repeated = d.finish()
        self.assertEqual((flagged, repeated, d.max_repeat, live), (True, 2, 3, [2, 3]))
        d2 = LoopDetector(False)
        d2.feed(line + '\n' + line + '\n' + line + '\n')
        self.assertEqual(d2.finish(), (False, 0))
        d3 = LoopDetector(True)
        d3.feed('short\n' * 10)                             # lines under MIN_LINE are ignored
        self.assertEqual(d3.finish(), (False, 0))


# ==================================================================================================== deterministic
def det_rig(checkpoints=2, disk_slots=4, quality=0.8):
    tok = common.tokenizer()
    clock = VirtualClock()
    stats = EngineStats(clock=clock)
    eng = ScriptedEngine(ntok=4)
    sy = SynthProfile(clock=clock)
    sess = MockSession(eng, max_ctx=32768, stats=stats, synth=sy, disk_slots=disk_slots, checkpoints=checkpoints)
    drafter = MockDrafter(eng, quality, clock=clock, ms=2.0)
    worker = EngineWorker(sess, tok, stats=stats, think_end_id=tok.special['</think>'],
                          tool_open_id=tok.special['<tool_call>'])
    return clock, stats, eng, sess, drafter, worker


def run_job(worker, ids, drafter, max_new=4096, thinking=True):
    job = Job(ids, SamplingParams(temperature=0.0).resolve(), max_new, drafter=drafter, thinking=thinking,
              slot_params={'n_predict': max_new})
    worker.submit(job)
    while True:
        ev = job.events.get(timeout=60)
        if ev[0] in ('done', 'error'):
            return job, ev


LONG = ('Here is the failing test output. ' + 'assert add(2, 2) == 4 failed with 5; ' * 20).strip()


def scenario(rig):
    """four requests: a cold one (batched prefill), its follow-up (live), another conversation (cold), then the
    follow-up's prompt again (its checkpoint was spilled to the stand-in disk tier)"""
    clock, stats, eng, sess, drafter, worker = rig
    b = common.builder()
    m1 = [SYS, {'role': 'user', 'content': LONG}]
    r1 = {'role': 'assistant', 'reasoning_content': 'The add function is off by one.\nFix it.',
          'content': 'Fixed:\n```python\ndef add(a, b):\n    return a + b\n```\nRun:\n```bash\npytest -q\n```\n'}
    p1, ids1 = U.script(eng, m1, r1, None)
    m2 = m1 + [r1, {'role': 'user', 'content': 'Now add a docstring.'}]
    r2 = {'role': 'assistant', 'reasoning_content': 'One line.', 'content': 'Done.'}
    p2, ids2 = U.script(eng, m2, r2, None)
    m3 = [SYS, {'role': 'user', 'content': 'Unrelated: list the files.'}]
    r3 = {'role': 'assistant', 'reasoning_content': 'Use ls.', 'content': '```sh\nls\n```'}
    p3, ids3 = U.script(eng, m3, r3, None)
    out = []
    for p in (p1, p2, p3, p2):
        job, ev = run_job(worker, p, drafter)
        assert ev[0] == 'done', ev
        out.append((p, job.result, sess.last_prefill))
    del b
    return out


class TestDeterministic(unittest.TestCase):
    def test_scenario(self):
        rig = det_rig()
        clock, stats, eng, sess, drafter, worker = rig
        try:
            runs = scenario(rig)
        finally:
            worker.shutdown()
        sy = sess.synth
        srcs = [pf.source for _, _, pf in runs]
        self.assertEqual(srcs, ['cold', 'live', 'cold', 'disk'])
        # llama-server's totals == the requests' own numbers
        gen = sum(len(r.gen_ids) for _, r, _ in runs)
        self.assertEqual(stats.predicted_tokens, gen)
        self.assertEqual(stats.prompt_tokens, sum(pf.processed_n for _, _, pf in runs))
        self.assertEqual(stats.prompt_cached, sum(pf.cached_n for _, _, pf in runs))
        self.assertEqual(stats.n_decode, sum(r.passes + r.prefill_passes for _, r, _ in runs))
        self.assertEqual(stats.n_tokens_max, max(len(p) + len(r.gen_ids) for p, r, _ in runs))
        self.assertEqual(stats.spec_draft_tokens, sum(r.drafted for _, r, _ in runs))
        self.assertEqual(stats.spec_accepted, sum(r.accepted for _, r, _ in runs))
        self.assertEqual(sum(stats.accepted_per_pos), stats.spec_accepted)
        self.assertEqual(stats.accepted_per_pos, sorted(stats.accepted_per_pos, reverse=True))
        self.assertLessEqual(stats.accepted_per_pos[0], stats.spec_drafts)
        for tot, per in ((stats.spec_draft_tokens, stats.cls_draft_tokens), (stats.spec_accepted, stats.cls_accepted),
                         (stats.spec_drafts, stats.cls_drafts)):
            self.assertEqual(sum(per.values()), tot)
        for c in CLASSES:
            self.assertEqual(sum(stats.cls_pos[c]), stats.cls_accepted[c])
        self.assertGreater(stats.cls_drafts['code'], 0)
        self.assertGreater(stats.cls_drafts['shell'], 0)
        self.assertEqual(stats.tokens_reasoning + stats.tokens_answer, gen)
        think_end = common.tokenizer().special['</think>']
        self.assertEqual(stats.tokens_reasoning, sum(r.gen_ids.index(think_end) + 1 for _, r, _ in runs))
        self.assertEqual(stats.finish['eos'], 4)
        # exact synthetic timings
        ps = stats.pass_series
        self.assertAlmostEqual(ps['k3'].quantiles((0.5,))[0], sy.pass_ms(4), places=6)
        self.assertAlmostEqual(ps['sampler'].quantiles((0.5,))[0], sy.sampler_ms, places=6)
        self.assertAlmostEqual(ps['host'].quantiles((0.5,))[0], sy.host_ms, places=6)
        self.assertAlmostEqual(ps['drafter'].quantiles((0.5,))[0], 2.0, places=6)
        self.assertAlmostEqual(ps['total'].quantiles((0.5,))[0], sy.pass_ms(4) + sy.sampler_ms + sy.host_ms + 2.0,
                               places=6)
        p1, r1, pf1 = runs[0]
        n_b = pf1.processed_n - ((pf1.processed_n - 1) % 4 + 1)
        self.assertAlmostEqual(pf1.batched_ms, n_b / sy.batched_tok_s * 1e3, places=6)
        self.assertAlmostEqual(pf1.handoff_ms, sy.handoff_ms, places=6)
        self.assertAlmostEqual(pf1.passes_ms, sy.pass_ms(pf1.processed_n - n_b), places=6)
        t = stats.ttft
        self.assertAlmostEqual(t['total'].window()[0], pf1.batched_ms + pf1.handoff_ms + pf1.passes_ms, places=6)
        self.assertEqual(t['queue'].window(), [0.0] * 4)
        self.assertAlmostEqual(t['restore'].window()[3], sy.restore_ms['disk'], places=6)
        self.assertEqual(runs[1][2].batched_ms, 0.0)           # a short append: K3 passes only
        # the cache
        self.assertEqual(stats.cache_sources, {'live': 1, 'vram': 0, 'spill': 0, 'disk': 1, 'cold': 2})
        self.assertEqual(stats.cache_tokens['vram'], runs[1][2].cached_n)
        self.assertEqual(stats.cache_tokens['disk'], runs[3][2].cached_n)
        self.assertEqual(runs[1][2].cached_n, len(runs[0][0]) + len(runs[0][1].gen_ids) - 1)
        self.assertEqual(runs[1][2].lcp, runs[1][2].cached_n)
        ev = stats.cache_counters
        self.assertGreater(ev['spilled'], 0)
        self.assertEqual(ev['unparked'], 1)
        self.assertEqual(stats.cache_gauges['slots_used'], 2)
        # physics: the K3 p50 is T = 4's synthetic 16.5 ms; pass_pct is over the target's own span, the slower card's pass
        # time from pass_cards (the mock's synthetic card split: 0.97 of the K3 pass)
        ph = stats.physics_view()
        self.assertEqual(ph['pass_source'], 'card_events')
        self.assertAlmostEqual(ph['pass_pct'], round(PhysicsConfig().floor_ms() / (sy.pass_ms(4) * max(sy.card_split)) * 100, 2),
                               places=6)
        self.assertGreater(ph['token_pct'], ph['pass_pct'])    # accepted drafts: more than one token per pass
        sv = stats.spec_view()
        self.assertGreater(sv['tokens_per_pass'], 1.0)
        print(f"\n[metrics, deterministic] {len(runs)} requests, {gen} tokens, {stats.spec_drafts} drafted passes: "
              f"acceptance {sv['acceptance']}, per position {sv['per_pos_rate']}, tokens/pass {sv['tokens_per_pass']}; "
              f"K3 p50 {ps['k3'].quantiles((0.5,))[0]} ms = {ph['pass_pct']} % of the {ph['floor_ms']} ms floor, "
              f"{ph['token_pct']} % per token; TTFT {[round(x, 3) for x in t['total'].window()]} ms; "
              f"cache sources {stats.cache_sources}")

    def test_byte_identical(self):
        texts = []
        for _ in range(2):
            rig = det_rig()
            try:
                scenario(rig)
            finally:
                rig[5].shutdown()
            st = rig[1]
            view = {'requests_processing': 0, 'requests_deferred': 0, 'queue_depth': 0, 'build_info': 'caminus-test',
                    'model_alias': 'k3-coder', 'model_path': '', 'n_ctx': 32768, 'loading': False,
                    'spec_types': 'draft-dflash', 'http': {}, 'worker': {}}
            texts.append(MX.prometheus(st, view, [], []))
        self.assertEqual(texts[0], texts[1])
        self.assertIn('caminus:decode_pass_ms{part="k3",quantile="0.5"} 16.5', texts[0])

    def test_disabled(self):
        clock, stats, eng, sess, drafter, worker = det_rig()
        stats.enabled = False
        try:
            scenario((clock, stats, eng, sess, drafter, worker))
        finally:
            worker.shutdown()
        self.assertEqual((stats.prompt_tokens, stats.predicted_tokens, stats.n_decode, stats.requests), (0, 0, 0, 0))
        self.assertEqual(stats.pass_series['total'].count, 0)


# ==================================================================================================== cost
def _best(fn, reps=7):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        n = fn()
        best = min(best, (time.perf_counter() - t0) / n)
    return best * 1e6


class TestOverhead(unittest.TestCase):
    """the recording a decode pass / a request adds (host only; the M7 gate is <= 3 % of decode ms/token on the real
    engine, measured in M7)"""

    def test_cost(self):
        st = EngineStats()
        st.begin(Job([1] * 100, SamplingParams(), 100, slot_params={}), 32768)
        cc = ContentClass()
        clock = st.clock
        N = 20000

        def per_pass():
            for i in range(N):
                t_pass = clock()                    # generate: the pass start, the class
                cls = st.cls
                d = (clock() - t_pass) * 1e3        # the drafter's ms
                t0 = clock()                        # sample_pass: the K3 / sampler split
                t1 = clock()
                st.pass_split((t1 - t0) * 1e3, (clock() - t1) * 1e3, ((16.0, 15.0), (0.8, 1.7), 7864320))
                st.tokens(0, 3)                     # worker: ids by kind, the content class
                st.cls = cc.feed(' x = y + 1\n')
                st.pass_done(t_pass, d, 1, 3, 2, 3, cls)
            return N
        us_pass = _best(per_pass)

        def per_request():
            for i in range(N // 10):
                job = Job([1] * 100, SamplingParams(), 100, slot_params={})
                st.begin(job, 32768)
                st.cached_known(50)
                for k in range(4):
                    st.prefill_progress(12 * k)
                st.prefill_done(U_PF)
                st.first_token()
                st.cache_update({'checkpoints': i, 'evicted': 0}, {'n_slots': 8, 'slots_used': 2})
                st.end(job, U_RES)
            return N // 10
        us_req = _best(per_request)
        st.enabled = False

        def off_pass():
            for i in range(N):
                st.pass_done(0.0, 0.0, 1, 3, 2, 3, 'prose')
                st.tokens(0, 3)
            return N
        us_off = _best(off_pass)
        # end to end: run_generation on a CPU mock, stats on vs off (includes everything the loop adds)
        e2e = {}
        for mode in ('none', 'off', 'on'):
            def gen():
                eng = MockEngine(ntok=4, vocab=512, seed=3)
                s = EngineStats(enabled=(mode == 'on')) if mode != 'none' else None
                sess = MockSession(eng, max_ctx=1 << 20, stats=s)
                dr = MockDrafter(eng, 0.8)
                res = run_generation(sess, [5, 6, 7, 8], Sampler(SamplingParams(temperature=0.0)), 3000,
                                     lambda ids: None, drafter=dr)
                return res.passes
            e2e[mode] = _best(gen, reps=5)
        print(f'\n[metrics cost] per decode pass {us_pass:.2f} us recorded (the hooks the loop, the session and the '
              f'worker add: clock reads, pass_done, tokens, ContentClass.feed), {us_off:.2f} us with the hook '
              f'disabled; per request {us_req:.2f} us (begin, progress x4, prefill_done, first_token, cache_update, '
              f'end); end to end on the CPU mock (4-token passes, 3000 tokens): {e2e["none"]:.1f} / {e2e["off"]:.1f} / '
              f'{e2e["on"]:.1f} us per pass with no stats / disabled / on (the mock pass itself dominates)')
        self.assertLess(us_pass, 50.0)
        self.assertLess(us_req, 500.0)


class _PF:
    prompt_n, cached_n, processed_n, passes, ms = 100, 50, 50, 4, 30.0
    source, exact, lcp, restore_ms, batched_ms, handoff_ms, passes_ms = 'vram', False, 50, 0.6, 20.0, 2.0, 7.0


class _RES:
    gen_ids = [1] * 40
    decode_ms, prompt_n, finish = 500.0, 100, 'eos'


U_PF, U_RES = _PF(), _RES()


# ==================================================================================================== M7 ops
class _PlainSession(MockSession):
    """a MockSession that reports its drafts as filler (a plain pass on the real engine: drafts_real False)"""
    drafts_real = False


class TestM7Ops(unittest.TestCase):
    """impl:m7-ops: the scrape's snapshot under the stats lock, filler drafts not counted as drafts, the plain pass time
    (measured / estimate), per-card spans without exchange waits, the host counters, the refusal kinds"""

    def test_snapshot_consistent_under_writes(self):
        """an engine thread recording passes and requests while a scraper takes snapshots (with the llama-server bucket):
        every snapshot is internally consistent (the pass record's series move together) and no bucket add is lost"""
        import threading
        st = EngineStats()
        N, stop, snaps, taken = 4000, threading.Event(), [], [0, 0]

        def engine():
            for i in range(N):
                job = Job([1] * 10, SamplingParams(), 10, slot_params={})
                st.begin(job, 4096)
                st.prefill_done(U_PF)
                st.first_token()
                st.pass_split(0.0, 0.0, ((16.0, 15.0), None, 7864320))
                st.pass_done(st.clock(), 0.5, 1, 3, 2, 3, 'prose')
                st.end(job, U_RES)
            stop.set()

        def scraper():
            while not stop.is_set():
                sn = st.snapshot(take_bucket=True)
                taken[0] += sn.bucket[0]
                taken[1] += sn.bucket[2]
                snaps.append((sn.pass_series['total'].count, sn.emitted.count, sn.card_pass[0].count, sn.target_pass.count,
                              sn.predicted_steps, sn.requests))
                MX.prometheus(sn, _VIEW, [], [])            # the renderer on the snapshot (its take_bucket: the snapshot's rates)
        te, ts = threading.Thread(target=engine), threading.Thread(target=scraper)
        ts.start(); te.start(); te.join(); ts.join()
        final = st.snapshot(take_bucket=True)
        taken[0] += final.bucket[0]; taken[1] += final.bucket[2]
        self.assertGreater(len(snaps), 10)
        for tot, em, cp, tp, steps, req in snaps:
            self.assertEqual((tot, em, cp, tp), (steps, steps, steps, steps))
        self.assertEqual(taken[0], N * U_PF.processed_n)        # the prompt tokens: none lost to the bucket swap
        self.assertEqual(taken[1], N)                            # the predicted steps likewise
        self.assertEqual(final.pass_series['total'].count, N)
        # the snapshot is a copy: later writes do not reach it
        c0 = final.pass_series['total'].count
        st.pass_done(st.clock(), 0.0, 1, 0, 0, 1)
        self.assertEqual(final.pass_series['total'].count, c0)

    def test_filler_is_not_drafted(self):
        """a session whose drafts are filler (drafts_real False): the passes verify them but the DFlash counters stay at 0,
        and the plain pass time is measured; the same run with real drafts counts them"""
        out = {}
        for cls in (_PlainSession, MockSession):
            eng = MockEngine(ntok=4, vocab=512, seed=3)
            st = EngineStats()
            sess = cls(eng, max_ctx=1 << 16, stats=st)
            res = run_generation(sess, [5, 6, 7, 8], Sampler(SamplingParams(temperature=0.0)), 200, lambda ids: None,
                                 drafter=MockDrafter(eng, 0.8))
            out[cls.__name__] = (res, st)
        res, st = out['_PlainSession']
        sv = st.spec_view()
        self.assertEqual((res.drafted, res.accepted, st.spec_drafts, st.spec_draft_tokens, st.spec_accepted), (0, 0, 0, 0, 0))
        self.assertEqual(sv['per_pos'], [])
        self.assertEqual(sv['plain_source'], 'measured')
        self.assertIsNotNone(sv['speedup_vs_plain'])
        res2, st2 = out['MockSession']
        self.assertGreater(res2.drafted, 0)
        self.assertEqual(st2.spec_draft_tokens, res2.drafted)
        self.assertEqual(res.gen_ids, res2.gen_ids)                 # (greedy: drafting never changes a token)

    def test_plain_estimate_and_cards(self):
        st = EngineStats()
        st.exchange_wait_note = 'not measured here'
        self.assertIsNone(st.spec_view()['plain_source'])
        st.plain_estimate(12.5)
        self.assertEqual((st.spec_view()['plain_source'], st.spec_view()['plain_ms_per_token']), ('estimate', 12.5))
        st.pass_split(0.0, 0.0, ((16.0, 15.5), None, 7864320))
        st.pass_done(st.clock(), 1.0, 1, 3, 1, 2)
        self.assertEqual((st.card_pass[0].count, st.card_pass[1].count, st.card_wait[0].count), (1, 1, 0))
        self.assertEqual((st.target_pass.window(), st.link_bytes), ([16.0], 7864320))
        ph = st.physics_view()
        self.assertEqual(ph['pass_source'], 'card_events')
        self.assertAlmostEqual(ph['pass_pct'], round(st.physics.floor_ms() / 16.0 * 100, 2), places=6)
        s = MX.status(st.snapshot(), dict(_VIEW, clock_check={'ok': True, 'cards': [], 'summary': 'x'}), [], [])
        self.assertEqual([c['exchange_wait_note'] for c in s['cards']], ['not measured here'] * 2)
        self.assertEqual(s['clock_check']['ok'], True)
        text = MX.prometheus(st.snapshot(take_bucket=True), dict(_VIEW, clock_check={'ok': False}), [], [])
        self.assertIn('caminus:clock_check_ok 0', text)
        self.assertNotIn('caminus:exchange_wait_ms', text)          # absent, never a made-up value
        self.assertIn('caminus:card_pass_ms{card="0",quantile="0.5"} 16', text)

    def test_host_counters_and_refusals(self):
        st = EngineStats()
        st.host_sample(5, 100, busy=True)
        st.host_sample(3, 50, busy=False)
        st.host_sample(None, None, busy=True)
        self.assertEqual((st.evicted_ms, st.pgmigrate, st.pgmigrate_all, st.host_samples), (8, 100, 150, 3))
        st.refusal('cache', 3); st.refusal('routing_flag'); st.refusal('nonsense')
        self.assertEqual((st.refusals['cache'], st.refusals['routing_flag'], st.refusals['bounds']), (3, 1, 1))
        text = MX.prometheus(st.snapshot(take_bucket=True), _VIEW, [], [])
        self.assertIn('caminus:kfd_evicted_ms_total 8', text)
        self.assertIn('caminus:pgmigrate_total 100', text)
        self.assertIn('caminus:pgmigrate_all_total 150', text)
        # the real session's refusal kinds (caminus.serve.k3_session.refusal_kind; CPU: nothing is launched)
        from caminus.device.capacity import CapacityError
        from caminus.engine.k3 import ContextBoundsError
        from caminus.prefill.coder import PrefillRefused
        from caminus.prefill.glue import RoutingError
        from caminus.cache.prompt import CacheRefused
        from caminus.serve.k3_session import refusal_kind
        self.assertEqual([refusal_kind(e) for e in (RoutingError('x'), PrefillRefused('x'), ContextBoundsError('x'), CapacityError('x'),
                                                    CacheRefused('x'), RuntimeError('a wait gave up'), ValueError('x'))],
                         ['routing_flag', 'prefill', 'bounds', 'capacity', 'cache', None, None])


_VIEW = {'requests_processing': 0, 'requests_deferred': 0, 'queue_depth': 0, 'build_info': 'caminus-test', 'model_alias': 'k3-coder',
         'model_path': '', 'n_ctx': 32768, 'loading': False, 'spec_types': 'draft-dflash', 'http': {}, 'worker': {}}


if __name__ == '__main__':
    unittest.main()
