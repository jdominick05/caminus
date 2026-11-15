"""Terminal numerical state cannot be vouched for, parked or restored (CPU only)."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from caminus.cache.prompt import PromptCache
from caminus.errors import InvalidKVEncoding, NonFiniteLogits


class NumericalState(unittest.TestCase):
    def cache(self, cls=InvalidKVEncoding):
        pc = PromptCache.__new__(PromptCache)
        pc.m = NS(broken='synthetic refused cache block', poisoned=True)
        if cls is not None:
            pc.m._broken_error = cls
        pc.live = NS(known=True)
        pc.fence = Mock()
        pc.reset = Mock(side_effect=AssertionError('terminal state must not reset'))
        pc._cold = Mock(side_effect=AssertionError('terminal state must not restore or reset'))
        return pc

    def test_terminal_state_refuses_mutation_before_other_cache_state_is_touched(self):
        # No store, state tensors or layout exist: these paths must refuse before
        # looking them up, and before the ordinary poisoned-state reset branch.
        entries = (
            lambda pc: pc._check_engine(),
            lambda pc: pc.prepare([1, 2]),
            lambda pc: pc.extend([1], {'i0': 1, 'commit': 0}),
            lambda pc: pc.append([1], {'i0': 1, 'commit': 0}),
            lambda pc: pc.commit([1], {'i0': 1, 'commit': 0}, checkpoint=True),
            lambda pc: pc.settle(returned=True, drained=False),
            lambda pc: pc.park(),
            lambda pc: pc.park(background=True),
            lambda pc: pc.load_entry('synthetic.ckv'),
            lambda pc: pc.export_checkpoint((1, 0), 'unused'),
        )
        for cls in (InvalidKVEncoding, NonFiniteLogits, None):
            for i, entry in enumerate(entries):
                with self.subTest(cls=cls, entry=i):
                    pc = self.cache(cls)
                    with self.assertRaises(cls or NonFiniteLogits):
                        entry(pc)
                    self.assertFalse(pc.live.known)
                    pc.reset.assert_not_called()
                    pc._cold.assert_not_called()

    def test_numerical_error_during_settle_commit_never_enters_reset_fallback(self):
        for cls in (InvalidKVEncoding, NonFiniteLogits):
            with self.subTest(cls=cls):
                pc = self.cache()
                pc.m = NS(broken=None, poisoned=False, seq=1, token=0)
                pc.live = NS(known=True, vouched=(1, 0), st={'i0': 1, 'commit': 0}, vouch=Mock())
                pc.live_tokens = Mock(return_value=[1])
                error = cls('synthetic terminal commit failure')
                pc.commit = Mock(side_effect=error)
                with self.assertRaises(cls) as caught:
                    pc.settle(returned=False, drained=True)
                self.assertIs(caught.exception, error)
                self.assertFalse(pc.live.known)
                pc.live.vouch.assert_not_called()
                pc.reset.assert_not_called()


if __name__ == '__main__':
    result = unittest.main(exit=False).result
    print('RESULT', 'PASS' if result.wasSuccessful() else 'FAIL')
    raise SystemExit(0 if result.wasSuccessful() else 1)
