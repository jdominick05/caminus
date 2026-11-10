"""OutputParser(think_open=False): the Qwen3 template leaves a thinking prompt at 'assistant\n' and the model writes '<think>'
itself; one leading '<think>' of the output is consumed (held while it is partial across chunks), the rest as with a prompt
that opened the block; output that does not start with the tag is content (llama.cpp's parse: an answer without tags must
not come back with empty content). think_open=True (Qwen3.6's template: the prompt ends '<think>\n') keeps a literal tag as reasoning."""
import itertools
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
from caminus.serve.toolparse import OutputParser   # noqa: E402


def parse(text, cuts, **kw):
    p = OutputParser(None, thinking=True, **kw)
    pos = 0
    for c in list(cuts) + [len(text)]:
        p.feed(text[pos:c]); pos = c
    p.finish()
    m = p.result()
    return m.reasoning_content, m.content


class ThinkOpen(unittest.TestCase):
    def test_leading_tag_consumed_any_chunking(self):
        text = '<think>\nLet me add them.\n</think>\n\n17 * 23 = 391.'
        for k in (0, 1, 2):
            for cuts in itertools.combinations(range(1, len(text)), k):
                self.assertEqual(parse(text, cuts, think_open=False), ('Let me add them.\n', '17 * 23 = 391.'), cuts)

    def test_no_tag_is_content(self):
        for cuts in ((), (1,), (3, 9)):
            self.assertEqual(parse('The answer is 391.', cuts, think_open=False), ('', 'The answer is 391.'), cuts)
            self.assertEqual(parse('\n\nThe answer is 391.', cuts, think_open=False), ('', 'The answer is 391.'), cuts)

    def test_partial_lookalike_is_content(self):
        self.assertEqual(parse('<thinx q', (3,), think_open=False), ('', '<thinx q'))
        self.assertEqual(parse('<thi', (), think_open=False), ('', '<thi'))

    def test_only_one_tag(self):
        self.assertEqual(parse('<think><think>x</think>y', (), think_open=False), ('<think>x', 'y'))

    def test_prompt_opened_block_unchanged(self):
        self.assertEqual(parse('<think>\nx</think>y', ()), ('<think>\nx', 'y'))


if __name__ == '__main__':
    unittest.main()
