"""CPU-only regression tests for sustained answer repetition; no model, GPU or external service required."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
from caminus.serve.agentmon import AnswerLoopGuard
from caminus.serve.generate import TokenText

LINE = 'Got it. Thanks for the additional context!'


class TestAnswerLoop(unittest.TestCase):
    def test_cli_opt_out(self):
        from caminus.serve.cli import parse
        self.assertFalse(parse([])['no_loop_guard'])
        self.assertTrue(parse(['--no-loop-guard'])['no_loop_guard'])

    def test_answer_after_split_think_end(self):
        text = LINE + '\n</think>\n' + (LINE + '\n\n') * 78
        for width in (1, 2, 7, 64, len(text)):
            guard = AnswerLoopGuard(True)
            for i in range(0, len(text), width):
                if guard.feed(text[i:i + width]):
                    break
            self.assertTrue(guard.flagged)
            self.assertEqual(guard.max_repeat, guard.REPEAT)

    def test_non_thinking_and_final_line(self):
        guard = AnswerLoopGuard(False)
        guard.feed('\n\n'.join([LINE] * guard.REPEAT))
        self.assertFalse(guard.flagged)
        self.assertTrue(guard.finish())

    def test_optional_think_open_matches_visible_answer_policy(self):
        for prefix in ('', ' \t\n', '<thinx q\n', '<think>reasoning\n</think>\n'):
            text = prefix + (LINE + '\n') * AnswerLoopGuard.REPEAT
            for width in (1, 2, 5, 64, len(text)):
                guard = AnswerLoopGuard(True, think_open=False)
                for start in range(0, len(text), width):
                    guard.feed(text[start:start + width])
                self.assertTrue(guard.finish(), (prefix, width))

    def test_optional_opening_holds_only_a_bounded_prefix(self):
        guard = AnswerLoopGuard(True, think_open=False)
        for _ in range(20):
            guard.feed(' ' * 10000)
            self.assertEqual(guard.open_pending, '')
        for ch in '<think':
            guard.feed(ch)
            self.assertLessEqual(len(guard.open_pending), 6)
        self.assertFalse(guard.finish())
        for width in (1, 7, 1000):
            guard = AnswerLoopGuard(True, think_open=False)
            text = '<think>\n' + (LINE + '\n') * 80
            for start in range(0, len(text), width):
                self.assertFalse(guard.feed(text[start:start + width]))
                self.assertLessEqual(len(guard.tail), 10)
            self.assertFalse(guard.finish())   # still reasoning: no closing marker

    def test_tool_open_ends_reasoning_without_counting_payload(self):
        text = ('<think>reasoning\n<tool_call>\n<function=write_file>\n<parameter=content>\n'
                + (LINE + '\n') * 80 + '</parameter>\n</function>\n</tool_call>\n')
        for width in (1, 3, 7, len(text)):
            guard = AnswerLoopGuard(True, think_open=False)
            for start in range(0, len(text), width):
                self.assertFalse(guard.feed(text[start:start + width]))
            self.assertFalse(guard.finish())
            self.assertTrue(guard.feed((LINE + '\n') * guard.REPEAT))

    def test_worker_passes_the_prompt_opening_policy(self):
        import collections
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from caminus.serve.generate import GenResult
        from caminus.serve.sampler import SamplingParams
        from caminus.serve.worker import EngineWorker, Job

        class BytesTokenizer:
            def decode_bytes(self, ids, skip_special=False):
                return bytes(ids)

        ids = list(((LINE + '\n') * 80).encode())

        def generate(sess, prompt, sampler, max_new, on_tokens, **kw):
            keep = on_tokens(ids)
            return GenResult(finish='stop' if keep is not None else 'eos',
                             gen_ids=ids if keep is None else ids[:keep], prompt_n=len(prompt))

        for think_open in (False, True):
            worker = EngineWorker.__new__(EngineWorker)   # run the real request method without a thread or socket
            worker.sess = SimpleNamespace(plan=lambda *a, **kw: 0, eos_ids=(), request_marks=lambda: {}, abort=Mock())
            worker.tok, worker.stats_obj, worker.flight = BytesTokenizer(), None, None
            worker.think_end_id = worker.tool_open_id = None
            worker.stats, worker.history = collections.defaultdict(int), []
            job = Job([1, 2], SamplingParams(temperature=0).resolve(), len(ids), thinking=True,
                      think_open=think_open, loop_guard=True)
            with patch('caminus.serve.worker.run_generation', generate):
                worker._run(job)
            events = list(job.events.queue)
            self.assertEqual(job.loop_stopped, not think_open)
            self.assertEqual(events[-1][0], 'error' if not think_open else 'done')
            if not think_open:
                self.assertEqual(events[-1][1], 'loop')
                self.assertLess(len(job.result.gen_ids), len(ids) // 2)
            worker.sess.abort.assert_not_called()

    def test_code_and_tool_payloads_are_excluded(self):
        for text in ('```text\n' + (LINE + '\n') * 100 + '```\n',
                     '<tool_call>\n<function=write_file>\n<parameter=content>\n' + (LINE + '\n') * 100 +
                     '</parameter>\n</function>\n</tool_call>\n'):
            guard = AnswerLoopGuard(False)
            for ch in text:
                guard.feed(ch)
            self.assertFalse(guard.finish())
            guard.feed((LINE + '\n') * guard.REPEAT)
            self.assertTrue(guard.flagged)

    def test_different_prose_and_short_runs_reset(self):
        guard = AnswerLoopGuard(False)
        guard.feed((LINE + '\n') * 5 + 'A different sentence changes the run.\n' + (LINE + '\n') * 5)
        self.assertFalse(guard.finish())
        self.assertEqual(guard.max_repeat, 5)

    def test_long_lines_are_bounded(self):
        guard = AnswerLoopGuard(False)
        for _ in range(100):
            guard.feed('x' * 1000)
            self.assertLessEqual(len(guard.buf), guard.MAX_LINE)
        guard.feed('\n' + (LINE + '\n') * guard.REPEAT)
        self.assertTrue(guard.flagged)

    def test_exact_token_cut_is_independent_of_batching(self):
        class BytesTokenizer:
            def decode_bytes(self, ids, skip_special=False):
                return bytes(ids)
        text = ('Check.\n</think>\n' + (LINE + '\n\n') * 78).encode()
        results = []
        for width in (1, 4, 23, len(text)):
            tt = TokenText(BytesTokenizer(), loop_guard=AnswerLoopGuard(True))
            out, kept = [], 0
            for i in range(0, len(text), width):
                batch = list(text[i:i + width])
                released, n_keep = tt.feed(batch)
                out.append(released)
                kept += len(batch) if n_keep is None else n_keep
                if n_keep is not None:
                    break
            self.assertTrue(tt.loop_stopped)
            self.assertLess(kept, len(text) // 2)
            results.append((kept, ''.join(out)))
        self.assertTrue(all(result == results[0] for result in results))

    def test_optional_opening_token_cut_is_independent_of_batching(self):
        class BytesTokenizer:
            def decode_bytes(self, ids, skip_special=False):
                return bytes(ids)
        for prefix in ('', ' \n', '<thinx>\n', '<think>reasoning\n</think>\n'):
            text = (prefix + (LINE + '\n\n') * 80).encode()
            results = []
            for width in (1, 4, 23, len(text)):
                tt = TokenText(BytesTokenizer(), loop_guard=AnswerLoopGuard(True, think_open=False))
                out, kept = [], 0
                for start in range(0, len(text), width):
                    batch = list(text[start:start + width])
                    released, n_keep = tt.feed(batch)
                    out.append(released)
                    kept += len(batch) if n_keep is None else n_keep
                    if n_keep is not None:
                        break
                self.assertTrue(tt.loop_stopped)
                self.assertLess(kept, len(text) // 2)
                results.append((kept, ''.join(out)))
            self.assertTrue(all(result == results[0] for result in results), prefix)


if __name__ == '__main__':
    unittest.main()
