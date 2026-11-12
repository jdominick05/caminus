"""decode/dflash35.py's host bookkeeping on the CPU, with a stand-in drafter (no GPU, no model file): the injected interval
[lo, hi), when a round runs or falls back to filler, the clamp of rejected rows, gaps, give_up / skip_inject, snapshot /
restore, and the cache's window protocol (export / import of ring rows by position, wrapping the ring).
  python -I tests/decode/test_dflash35_adapter.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import torch
from caminus.engine import q35_dflash
from caminus.engine.q35_dflash import RING, T
from caminus.decode.dflash35 import Dflash35Drafter

ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


class Stub(q35_dflash.Dflash35):
    """the device side recorded: inject writes each layer's ring row p & (RING - 1) = p (as bytes); draft returns fixed ids"""
    def __init__(self, W=16, nl=2):
        self.window, self.nl, self.nkv, self.hd, self.dx, self.max_pos, self.path, self.mib = W, nl, 1, 2, 0, 1 << 20, 'stub', 0
        self.row_bytes = 2 * self.nkv * self.hd * 2
        self.L = [dict(ring=torch.zeros(RING * self.row_bytes // 2, dtype=torch.float16)) for _ in range(nl)]
        self.drafts, self.eng = [], None

    def inject(self, pos0, feats):
        for r in range(feats.shape[1]):
            for L in self.L: L['ring'].view(torch.uint8).view(RING, self.row_bytes)[(pos0 + r) & (RING - 1)] = (pos0 + r) % 251

    def draft(self, last, n0):
        self.drafts.append(n0)
        return [7, 8, 9]


# (the window protocol indexes a card's tensors: on the CPU the device string is 'cpu')
Dflash35Drafter._rows = lambda self, lo, hi: torch.arange(lo, hi) & (RING - 1)
d = Stub(); a = Dflash35Drafter(d)
F = lambda n: torch.zeros(2, n, 4)
check(a([1, 2, 3], 3) == [3, 3, 3] and a.filler, 'nothing injected: filler')
a.inject(0, F(2)); check(a([1, 2, 3], 3) == [7, 8, 9] and d.drafts == [2], 'positions 0 .. 1 injected: the round at 2 runs')
a.inject(2, F(4))                                   # (a verify pass: 4 rows, 1 accepted -> next n0 = 4)
check(a([0] * 5, 3) == [7, 8, 9] and a.hi == 4, 'a verify pass\'s rejected rows: clamped to the round\'s position (hi 6 -> 4)')
a.inject(9, F(1)); check(a.lo == 9 and a.hi == 10, 'an inject past hi: a gap restarts the interval there')
check(a([0] * 11, 2) == [0, 0] and a.filler, 'a gap inside the round\'s window: filler (no stale row read)')
a.inject(10, F(16)); check(a([0] * 27, 3) == [7, 8, 9], f'the window covered again after W positions (lo {a.lo}, hi {a.hi})')
a.give_up(20, 30); check(a.hi == 20, 'give_up: the interval ends at lo')
a.skip_inject(25); check(a.lo == a.hi == 25, 'skip_inject past hi: an empty interval at p')
s = a.snapshot(); a.reset(); check(a.hi == 0, 'reset'); a.restore(s); check((a.lo, a.hi) == (25, 25), 'restore')
# window protocol across the ring's wrap
a.reset(); a.inject(RING - 6, F(12))
dm = a.window_state(0); check(dm == dict(lo=RING - 6, hi=RING + 6), f'window_state: the interval, <= W + T rows ({dm})')
win = torch.zeros(d.nl, a.wrows(), d.row_bytes, dtype=torch.uint8)
dm = a.window_state(RING - 4)
a.export_window(win, dm)
want = torch.tensor([(p % 251) for p in range(RING - 4, RING + 6)], dtype=torch.uint8)
check(all(torch.equal(win[l, :10, 0], want) for l in range(d.nl)), 'export_window: ring rows by position, across the wrap')
for L in d.L: L['ring'].zero_()
a.import_window(win, dm); a.set_window_state(dm)
got = d.L[1]['ring'].view(torch.uint8).view(RING, d.row_bytes)
check(torch.equal(got[(torch.arange(RING - 4, RING + 6)) & (RING - 1), 0], want) and (a.lo, a.hi) == (RING - 4, RING + 6),
      'import_window + set_window_state: the rows back at their positions, the interval restored')
big = a.window_state(0); a.inject(RING + 6, F(40)); big = a.window_state(0)
check(big['hi'] - big['lo'] == a.wrows(), f'window_state caps at W + T rows ({big})')
print('RESULT ' + ('PASS' if ok else 'FAIL'), flush=True)
sys.exit(0 if ok else 1)
