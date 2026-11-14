"""Bounded content classification across arbitrary decoded-text chunk boundaries; no model or socket."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
from caminus.serve.agentmon import ContentClass


class TestContentClass(unittest.TestCase):
    def test_split_markers_and_function_names(self):
        cases = [
            ('<tool_call>', 'tool'),
            ('<tool_call><function=run_shell_command>', 'shell'),
            ('<tool_call><function=read_file>', 'tool'),
            ('<tool_call><function=bash>echo ok</function></tool_call>', 'prose'),
            ('```bash\necho ok\n', 'shell'),
            ('```python\nx = 1\n```', 'prose'),
            ('```bash\necho ok\n```<tool_call><function=run_shell_command>', 'shell'),
        ]
        for text, want in cases:
            self.assertEqual(ContentClass().feed(text), want)
            for cut in range(len(text) + 1):
                cc = ContentClass()
                cc.feed(text[:cut])
                self.assertEqual(cc.feed(text[cut:]), want, (text, cut))

    def test_every_prefix_has_the_same_class(self):
        text = ('Plan.\n```bash\necho ok\n```\n<tool_call>\n<function=run_shell_command>\n'
                '<parameter=command>\necho done\n</parameter>\n</function>\n</tool_call>\n'
                'Then code.\n```python\nprint(1)\n```\n<tool_call><function=read_file>')
        for width in (1, 2, 3, 5, 7, 11, 16, len(text)):
            cc = ContentClass()
            for start in range(0, len(text), width):
                end = min(start + width, len(text))
                cc.feed(text[start:end])
                self.assertEqual(cc.cls, ContentClass().feed(text[:end]), (width, end))

    def test_completed_markers_are_not_replayed(self):
        cc = ContentClass()
        self.assertEqual(cc.feed('```'), 'code')
        self.assertEqual(cc.feed('bash\n'), 'shell')
        self.assertEqual(cc.feed('```'), 'prose')
        self.assertEqual(cc.feed(' '), 'prose')
        self.assertEqual(cc.feed('<tool_'), 'prose')
        self.assertEqual(cc.feed('call>'), 'tool')
        self.assertEqual(cc.feed('</tool_'), 'tool')
        self.assertEqual(cc.feed('call>'), 'prose')

    def test_partial_state_is_bounded_even_for_long_headers(self):
        for opening, ending, want in (('<tool_call><function=', '>', 'tool'), ('```', '\n', 'code')):
            text = opening + 'x' * 10000 + ending
            for width in (1, 127, len(text)):
                cc = ContentClass()
                for start in range(0, len(text), width):
                    cc.feed(text[start:start + width])
                    self.assertLessEqual(len(cc.tail), cc.MAX_TAIL)
                    self.assertLessEqual(len(cc.fn_pending or ''), cc.MAX_FN)
                    self.assertLessEqual(len(cc.lang_pending or ''), cc.MAX_LANG)
                self.assertEqual(cc.cls, want)


if __name__ == '__main__':
    unittest.main()
