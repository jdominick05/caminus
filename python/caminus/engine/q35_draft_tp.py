"""One-card drafter weights beside a two-card dense trunk (engine/q35_dense_tp.py Dense35TP).

The MTP block (engine/q35_mtp.py) and the DFlash2 drafter (engine/q35_dflash.py) run on one card and read their target's
head and embedding in place. On two cards the embedding is already replicated, but the head is split by vocabulary rows:
DraftTarget35TP keeps a complete copy on the drafter's card so each chained draft needs no peer launch. That copy is
projected separately from both trunk shards (head_mib) and checked before it is loaded; it goes through the trunk's
uploader (engine/upload.py: the two-card default stages through pinned memory, and CAMINUS_UPLOAD_LOG records it).

DraftTarget35TP is the drafters' compute view only: the attributes below are the ones Mtp35 and Dflash35 read, set
explicitly (no attribute falls through to the trunk). The prompt cache's identity stays the trunk (the drafters' tgt).
The trunk's dimensions stay global here: the drafter's own block is not split.
"""
import numpy as np
import torch

from .. import ggufio
from ..device import capacity
from .own_layout import q6k_planes


def head_mib(path):
    """MiB of the full output head: the copy DraftTarget35TP keeps on the drafter's card"""
    F = ggufio.GGUFFile(path)
    try: return F.T['output.weight'].n_bytes / capacity.MIB
    finally: F.close()


class CardUploads:
    """the trunk's uploader seen from one card: up(a, name, 0, role) uploads to that card (the drafters' one-card index),
    loading(what) frames a drafter's load in the upload log"""
    def __init__(self, h2d, card):
        self.h2d, self.card = h2d, int(card)

    def up(self, a, name, card, role):
        if card != 0: raise ValueError(f'CardUploads: a one-card drafter names card 0, not {card}')
        return self.h2d.up(a, name, self.card, role)

    def loading(self, engine, path):
        return self.h2d.loading(engine, path)


class DraftTarget35TP:
    """the MTP / DFlash engines' view of one TP card: that card's device, grid and embedding, a complete head on it, and
    the trunk's global shapes and launch settings"""
    def __init__(self, trunk, card=0, check=True):
        if card not in (0, 1): raise ValueError(f'DraftTarget35TP: card {card}: 0 or 1')
        self.trunk, self.card = trunk, int(card)
        self.dev = int(trunk.devs[self.card])
        self.per_wgp = trunk.per_wgp_c[self.card]
        self.emb, self.emb_dev = trunk.emb, trunk.emb_dev[self.card]
        self.path, self.F, self.module, self.tcap = trunk.path, trunk.F, trunk.module, trunk.tcap
        self.D, self.hd, self.nh, self.nkv, self.ffn, self.vocab = trunk.D, trunk.hd, trunk.nh, trunk.nkv, trunk.ffn, trunk.vocab
        self.max_ctx, self.eps, self.nrot, self.theta_scale = trunk.max_ctx, trunk.eps, trunk.nrot, trunk.theta_scale
        self.rpt, self.rpt16, self.limit_us, self.max_retries = trunk.rpt, trunk.rpt16, trunk.limit_us, trunk.max_retries
        self.h2d = CardUploads(trunk.h2d, self.card)
        t = trunk.F.T['output.weight']
        want = 'Q8_0' if trunk.tcap else 'Q6_K'
        if t.type != want or tuple(t.shape) != (trunk.D, trunk.vocab):
            raise ValueError(f'DraftTarget35TP: head {t.type} {tuple(t.shape)}, expected {want} {(trunk.D, trunk.vocab)}')
        self.mib = t.n_bytes / capacity.MIB
        if check: capacity.check([self.dev], [self.mib], f'DraftTarget35TP full {want} head ({self.mib:.1f} MiB)')
        raw = t.pread(0, t.n_bytes)
        if t.type == 'Q6_K': raw = q6k_planes(raw, trunk.vocab, trunk.D)
        # (under the drafter card's device: the pageable arm copies to the current device, which the trunk left at card 0)
        with torch.cuda.device(self.dev), self.h2d.loading('DraftTarget35TP', trunk.path):
            w = self.h2d.up(np.ascontiguousarray(raw), t.name, 0, 'mat')
        self.head = dict(name=t.name, fmt=t.type, rows=trunk.vocab, K=trunk.D, w=w)

    def _f32(self, name, n):
        """an F32 vector of the file on the drafter's card (Dense35Own._f32's checks; the MTP block's norms)"""
        t = self.F.T[name]
        if t.type != 'F32' or t.n_values != n: raise ValueError(f'DraftTarget35TP: {name} is {t.type} {list(t.shape)}, the tape needs F32 [{n}]')
        with torch.cuda.device(self.dev): return self.h2d.up(t.pread(0, t.n_bytes).view(np.float32), name, 0, 'vec')


def pass_h(src):
    """the trunk's final h of the pass just run: the one-card engine's buf['h'], or TP card 0's (replicated on both cards;
    the MTP block runs on card 0)"""
    return (src.buf[0] if isinstance(src.buf, list) else src.buf)['h']


def verify_shards(src, vh):
    """a verify pass's two vocabulary shards: the two-card tape's per-card logits as they are (no gather of the full head
    for sampled acceptance), or the one-card tape's row split at vh"""
    L = src.logits2
    return tuple(L) if isinstance(L, (list, tuple)) else (L[:, :vh], L[:, vh:])
