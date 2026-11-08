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


if __name__ == '__main__':
    unittest.main()
