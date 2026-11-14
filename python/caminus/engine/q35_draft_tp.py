"""One-card drafter weights beside a two-card dense trunk.

The embedding is already replicated. The head is sharded: keep a complete copy on the drafter's card so each chained
draft needs no peer launch. This copy is projected separately from both trunk shards and checked before it is loaded.
The trunk's dimensions remain global here (the drafter's attention block is not sharded).
"""
import numpy as np
import torch

from .. import ggufio
from ..device import capacity
from .own_layout import q6k_planes
from .q35_dense import Dense35Own


def head_mib(path):
    F = ggufio.GGUFFile(path)
    try: return F.T['output.weight'].n_bytes / capacity.MIB
    finally: F.close()


class DraftTarget35TP:
    """The existing MTP / DFlash engines' view of one TP card, with the full head and shared embedding."""
    def __init__(self, trunk, card=0, check=True):
        if card not in (0, 1): raise ValueError(f'DraftTarget35TP: card {card}: 0 or 1')
        self.trunk, self.card = trunk, int(card)
        self.dev = int(trunk.devs[self.card])
        self.per_wgp = trunk.per_wgp_c[self.card]
        self.emb_dev = trunk.emb_dev[self.card]
        t = trunk.F.T['output.weight']
        want = 'Q8_0' if trunk.tcap else 'Q6_K'
        if t.type != want or tuple(t.shape) != (trunk.D, trunk.vocab):
            raise ValueError(f'DraftTarget35TP: head {t.type} {tuple(t.shape)}, expected {want} {(trunk.D, trunk.vocab)}')
        self.mib = t.n_bytes / capacity.MIB
        if check: capacity.check([self.dev], [self.mib], f'DraftTarget35TP full {want} head ({self.mib:.1f} MiB)')
        with torch.cuda.device(self.dev):
            raw = t.pread(0, t.n_bytes)
            if t.type == 'Q6_K': raw = q6k_planes(raw, trunk.vocab, trunk.D)
            self.head = dict(name=t.name, fmt=t.type, rows=trunk.vocab, K=trunk.D,
                             w=torch.from_numpy(np.ascontiguousarray(raw)).to('cuda'))

    def __getattr__(self, name):
        if name == 'trunk': raise AttributeError(name)
        return getattr(self.trunk, name)

    def _f32(self, name, n):
        with torch.cuda.device(self.dev): return Dense35Own._f32(self, name, n)


def pass_h(src):
    """The MTP block stays on TP card 0; its input is that card's replicated final h, or the one-card pass's h."""
    return (src.buf[0] if isinstance(src.buf, list) else src.buf)['h']


def verify_shards(src, vh):
    """A verify pass's vocabulary halves on their existing cards (no full-head gather for sampled acceptance)."""
    L = src.logits2
    return L if isinstance(L, (list, tuple)) else (L[:, :vh], L[:, vh:])
