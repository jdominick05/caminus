"""DraftPolicy(probe_plain=True): a host-accept drafter's bucket measures one plain pass, then switches where drafting
loses (ThinkingCap MTP at 40K served 0.84x of plain: drafted passes carry no drafter step time, so without the probe the
bucket had no plain cost and never switched)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
from caminus.serve.draftpolicy import DraftPolicy, MIN_SAMPLES   # noqa: E402

DRAFT_MS, DRAFT_TOK, PLAIN_MS = 52.0, 2.0, 33.0   # (a drafted pass at 26 ms / token against 33 ms per plain token: loses)


def run(policy, passes, draft_ms=DRAFT_MS, draft_tok=DRAFT_TOK, plain_ms=PLAIN_MS, pos=40000):
    modes = []
    for _ in range(passes):
        m = policy.decide('prose', pos)
        modes.append(m)
        if m == 'draft': policy.record('draft', 'prose', pos, draft_ms, draft_tok)
        else: policy.record('plain', 'prose', pos, plain_ms, 1)
    return modes


class ProbePlain(unittest.TestCase):
    def test_without_probe_never_switches(self):
        p = DraftPolicy()
        modes = run(p, 64, draft_ms=90.0)          # (drafting loses badly, but no plain cost ever forms)
        self.assertTrue(all(m == 'draft' for m in modes))

    def test_probe_measures_then_switches_where_drafting_loses(self):
        p = DraftPolicy(probe_plain=True)
        modes = run(p, 64, draft_ms=90.0)          # 45 ms per drafted token against 33 per plain token
        self.assertEqual(modes[:MIN_SAMPLES], ['draft'] * MIN_SAMPLES)
        self.assertEqual(modes[MIN_SAMPLES], 'plain')                # the probe
        self.assertEqual(p.counts['plain_probe'], 1)
        self.assertGreater(modes.count('plain'), 40)                  # then plain, with a draft probe every PROBE_EVERY
        self.assertEqual(p.view()[f'prose/{DraftPolicy.depth(40000)}']['mode'], 'plain')

    def test_probe_keeps_drafting_where_it_pays(self):
        p = DraftPolicy(probe_plain=True)
        modes = run(p, 64)                         # 26 ms per drafted token against 33 per plain token
        self.assertEqual(modes.count('plain'), 1)  # the one probe, then drafting stays
        self.assertEqual(p.view()[f'prose/{DraftPolicy.depth(40000)}']['mode'], 'draft')


if __name__ == '__main__':
    unittest.main()
