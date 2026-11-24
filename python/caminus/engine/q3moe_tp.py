"""qwen3moe (Qwen3-30B-A3B) decoding on TWO cards with tensor parallelism, on Caminus's own code from the GGUF file to the
logits: engine/q3moe_one.py's reader, weight layouts and fused tape (Q3MoEOne._tape_fused, layer for layer), each card
running the two-card build of the lean decode module (dec_q3m_tp: the lean ops plus a push-only OP_XCHG and
OP_OWN_SLOT_SUM, kernels/ops_q3m.h under K3_Q3M_TP). The host shell is engine/q35_dense_tp.py's (the uploader, the
capacity check before any allocation, the exchange slots of engine/tp_slots.py, the server contract). No gguf-py, no
tokenizers. Served by `caminus serve --cards 2` (caminus.arch.Q3MoETP, the 'qwen3moe-tp' entry; one card stays the model's
default).

The split (card c of 2; card 0 first in every sum; every cut on a whole quantization block of the matrix it cuts):
- Attention: card 0 takes kv heads 0 .. kv0 - 1 and their query heads (8 per kv head: OP_OWN_ATTS's ratio), card 1 the
  rest: rows of attn_q, attn_k, attn_v, and attn_output's columns of its query heads; its kv heads' f16 KV cache (q8=True:
  Q8_0 rows, below). The attention splits over positions follow the card (q3moe_one.splits_for of its kv heads), never the
  grid.
- Experts: every expert's intermediate rows 0 .. exp_rows - 1 on card 0, exp_rows .. ffn - 1 on card 1 (ffn 768 = 3 x
  256, so the cuts are 512 / 256 or 256 / 512): ffn_gate_exps / ffn_up_exps rows, ffn_down_exps columns (a K split). The
  grouped tiles take the cut tensors as whole expert tensors of rows_c rows each (total_rows = experts x rows_c).
- Head: the vocabulary rows split at vh (vocab_share, rounded to 128 rows); each card's logits are its rows.
- Replicated on both cards: the embedding (decoded on each card), the norms, the q / k norms, the router (F32) and the
  residual. Both cards pick the same experts from the same bits: check_replicas compares the residual AND every layer's
  router ids.
Per layer and card (15 segments; one card's fused tape has 12):
  add_rms16(x += the previous FFN exchange) -> v, k, nop-join, q -> own_atts (its kv heads) -> attn_output (its columns:
  a partial) -> XCHG 2 il -> add_rms16(x += it) -> router f32_mv, router -> gate | up (its rows, grouped) -> down (its
  K, per slot) -> OWN_SLOT_SUM (the partial weighted over the top-k slots on the card: D floats to exchange, not k_used
  x D) -> XCHG 2 il + 1
then add_rms16 (the output norm) and the head's rows. Exchanges per token: 2 nl (96 for the 48-layer file), D floats each.
No link keep-alive segment (Q3MoEOne's): the exchanges keep both links busy (an A/B of it on the XTX is a later measurement).
Not bitwise to the one-card engine by design (the K-split down, the attn_output column split and the split count each
change a summation order); bitwise to itself across grids, and its replicas equal after every pass.

q8=True: each card's KV cache as Q8_0 rows on dec_q3m_tp_kvq8 (the two-card module with the one-card Q8 module's
OP_OWN_ATTS: kernels/ops_own_atts.h under K3_Q3M_KV_Q8), the one-card engine's format (q3moe_one.KV_FORMAT_Q8: each block
the encode of the fp16 values the F16 cache holds, K and q through the head's Walsh-Hadamard transform) over that card's
2 kv heads: 544 bytes per position and layer on each card instead of 1,024. A refused block (a non-finite K / V value)
sets that card's error word after Sync; both words come back in each card's one status copy and are read before the
bound and bail paths (which zero Sync): the engine is then broken (InvalidKVEncoding, reset() keeps it). On a bailed pass
the segments after the expired wait still run, on the earlier pass's partial in the slot (finite values), and the retry
rewrites the position's rows; a refused block there is as terminal as anywhere else. Served by caminus serve --cards 2
--kv q8 (arch.Q3MoETP); its two-card prefill is Q3MPrefillTP's Q8 path.

The server contract is Dense35TP's (the device sampler's two logits shards on two cards: logits_half, shard_devs, vh;
step / step_token / verify / launch_tokens / finish_pass; reset; state {(card, layer, 'kv')}; tp_key)."""
import ctypes
import os
import sys

import numpy as np
import torch

from .. import ggufio
from ..device import capacity
from . import k3, upload
from .own_layout import q6k_planes
from .q35_dense_tp import cut, vocab_cut
from .q3moe_one import card_per_wgp, splits_for, ATS_SPLITS, ATS_LMIN, KV_FORMAT_Q8, _rope_mode

SPLITS_CAP = 128         # the most attention splits per kv head on a card (splits_cap); one card's engine caps at 64. A card holds half
                          # the kv heads, so 64 splits fill 128 of the XTX's 192 / the XT's 210 workgroups; the two-card
                          # modules' merge takes up to k3.ATS_MAXS_TP (128: XTX 96, XT 105 splits of 2 kv heads), the
                          # depth benchmark's arm option ':cap=128' (benchmarks/q3moe_depth.py). Measured 2026-11-15, XTX + XT: 128 gives 1.10 / 1.12 / 1.12x cap 64's decode at 16K / 32K / 40.9K, the same at 8 and 4K
from .tp_slots import Slots, _hip

TAPE_MIB = 4      # an upper bound of one card's tape buffers (checked in _build_tapes), counted in the projection
MODULE = 'dec_q3m_tp'
EXP_ROWS, VOCAB_SHARE = 512, 0.532   # the design defaults: the XTX (card 0) takes 512 of each expert's 768 rows; the
                                     # head at the pair's bandwidth share (XTX 881 vs XT 774 GB/s)


def shapes(F):
    """the header facts the split and the projection read"""
    hp = F.hp
    return dict(nl=int(hp('block_count')), D=int(hp('embedding_length')), nh=int(hp('attention.head_count')),
                nkv=int(hp('attention.head_count_kv')), hd=int(hp('attention.key_length')), n_exp=int(hp('expert_count')),
                k_used=int(hp('expert_used_count')), ffn=int(hp('expert_feed_forward_length')),
                vocab=int(F.T['output.weight'].shape[1]))


def split_plan(nh, nkv, hd, n_exp, ffn, D, vocab, kv0, r0, vh):
    """the split as index ranges per card: {role: [card 0's, card 1's]}, each a list of (start, end): rows of the row-split
    matrices, columns (values) of the column-split ones. 'exp' is every expert's gate / up rows in the whole tensor
    (expert e's rows at e * ffn), 'exp_k' the down's columns"""
    g = nh // nkv; kvs = [(0, kv0), (kv0, nkv)]
    P = dict(kv_heads=[b - a for a, b in kvs])
    P['q'] = [[(a * g * hd, b * g * hd)] for a, b in kvs]
    P['kv'] = [[(a * hd, b * hd)] for a, b in kvs]                     # attn_k / attn_v rows
    P['attn_output'] = [[(a * g * hd, b * g * hd)] for a, b in kvs]    # columns
    cuts = [(0, r0), (r0, ffn)]
    P['exp'] = [[(e * ffn + a, e * ffn + b) for e in range(n_exp)] for a, b in cuts]
    P['exp_k'] = [[c] for c in cuts]
    P['exp_rows'] = [b - a for a, b in cuts]
    P['head'] = [[(0, vh)], [(vh, vocab)]]
    return P


def _check_split(c, kv0, r0):
    if c['hd'] != 128 or c['nh'] != 8 * c['nkv']: raise ValueError(f'Q3MoETP: head dim {c["hd"]}, {c["nh"]} / {c["nkv"]} heads (128, 8 query heads per kv head: OP_OWN_ATTS)')
    if not 1 <= kv0 < c['nkv']: raise ValueError(f'Q3MoETP: kv_heads {kv0} on card 0 (1 .. {c["nkv"] - 1})')
    if r0 % 256 or not 0 < r0 < c['ffn']: raise ValueError(f'Q3MoETP: exp_rows {r0} (a multiple of 256 inside the experts\' {c["ffn"]}: the down\'s K cut must be whole blocks)')


def _role(name):
    """how a tensor is split: 'rep' (both cards whole), 'q' / 'kv' (rows by kv heads), 'o' (columns by kv heads), 'exp'
    (expert rows), 'exp_k' (expert columns), 'head' (vocabulary rows)"""
    if name == 'output.weight': return 'head'
    for k, r in (('.attn_q.weight', 'q'), ('.attn_k.weight', 'kv'), ('.attn_v.weight', 'kv'), ('.attn_output.weight', 'o'),
                 ('.ffn_gate_exps.weight', 'exp'), ('.ffn_up_exps.weight', 'exp'), ('.ffn_down_exps.weight', 'exp_k')):
        if name.endswith(k): return r
    return 'rep'   # (the embedding, every norm, the q / k norms, the router)


def projection_terms(F, max_ctx, exp_rows=EXP_ROWS, kv_heads=None, vocab_share=VOCAB_SHARE, splits_cap=SPLITS_CAP, q8=False, yarn=False):
    """bytes per card by term, from the header alone: [dict(split, rep, kv, ats, slots, small, tape) for card 0, card 1].
    split: its slices of the split matrices (exact: the cuts are whole blocks); rep: the replicated tensors (the embedding,
    the norms, the router); kv: its kv heads' KV cache (f16, or q8: Q8_0 rows), max_ctx + ATS_PAD rows; ats: OP_OWN_ATTS's scratch at splits_cap
    splits (the most a card takes); slots: the exchanges' receive slots (16 words + D floats, 2 nl of them, uncached VRAM outside
    torch); small: the per-layer q / k norm blocks (yarn: with the YaRN table) and the router choices; tape: TAPE_MIB. max_ctx is not bounded by the
    file's context_length here (a projection for a longer context is arithmetic only)"""
    c = shapes(F); kv0 = c['nkv'] // 2 if kv_heads is None else int(kv_heads)
    _check_split(c, kv0, int(exp_rows))
    vh = vocab_cut(c['vocab'], vocab_share); kvs = [kv0, c['nkv'] - kv0]
    frac = dict(q=[k / c['nkv'] for k in kvs], kv=[k / c['nkv'] for k in kvs], o=[k / c['nkv'] for k in kvs],
                exp=[exp_rows / c['ffn'], 1 - exp_rows / c['ffn']], exp_k=[exp_rows / c['ffn'], 1 - exp_rows / c['ffn']],
                head=[vh / c['vocab'], 1 - vh / c['vocab']])
    out = [dict(split=0.0, rep=0.0) for _ in (0, 1)]
    for t in F.tensors:
        r = _role(t.name)
        for k in (0, 1):
            if r == 'rep': out[k]['rep'] += t.n_bytes
            else: out[k]['split'] += t.n_bytes * frac[r][k]
    for k in (0, 1):
        out[k].update(kv=c['nl'] * (int(max_ctx) + k3.Engine.ATS_PAD) * k3.kv_pos_bytes(kvs[k] * c['hd'], bool(q8)),
                      ats=4 * k3.Engine.atts_scratch_floats(kvs[k], splits_cap),   # (a card's splits are at most the cap)
                      slots=2 * c['nl'] * (16 + c['D']) * 4,
                      small=c['nl'] * ((2 * c['hd'] + 2 + (k3.YARN_TABLE_FLOATS if yarn else 0)) * 4 + c['k_used'] * 8),
                      tape=TAPE_MIB * 2 ** 20)
    return out


class Q3MoETP:
    TAPE_MIB = TAPE_MIB

    def __init__(self, path, devices=(0, 1), max_ctx=4096, limit_us=200_000, module=None, per_wgp='card', rpt=16, rpt_down=None,
                 rpt_head=64, exp_rows=EXP_ROWS, kv_heads=None, vocab_share=VOCAB_SHARE, lmin=ATS_LMIN, local_only=False,
                 vram_check=True, vram_extra_mib=(0.0, 0.0), max_retries=8, splits_cap=SPLITS_CAP, q8=False, rope=None):
        """path: a qwen3moe GGUF with Q4_K / Q6_K matrices (the lean module's formats). devices: the two torch / HIP device
        indices (card 0 first: its partials come first in every sum). exp_rows: card 0's rows of every expert's
        intermediate (a multiple of 256); kv_heads: card 0's kv heads (None: half); vocab_share: card 0's part of the
        head rows. module None: dec_q3m_tp (or its one-card base / the _trace builds, by name). per_wgp: workgroups
        per WGP ('card': q3moe_one.card_per_wgp on each card, or an int for both; the logits are the same bits at every
        grid). rpt: rows per matvec task (an int, or one per card); rpt_down: the expert down's (per card; None: rpt;
        the second card's down is K = ffn - exp_rows wide, so larger tasks there are a measured option); rpt_head: the
        head's. local_only: TEST-ONLY, every exchange copies the card's own partial (no peer: each card's time alone; the
        logits are not the model's). vram_extra_mib: per card, what the caller allocates right after (checked with the
        engine's projection: another tape set is TAPE_MIB). splits_cap: the most attention splits per kv head on a card
        (1 .. k3.atts_max_splits of the module: 128 on the two-card modules; default SPLITS_CAP = 128); each card takes
        q3moe_one.splits_for(its kv heads, splits_cap), and the projection counts the cap's scratch. q8: the KV caches as Q8_0
        rows on dec_q3m_tp_kvq8 (KV_FORMAT_Q8; module None or that module only; not with the _trace build); a refused
        encoding raises caminus.errors.InvalidKVEncoding and leaves the engine broken (reset() keeps it). rope: None or a
        caminus.rope.YarnMode (the long-context mode: the two-card module's YaRN build, k3.Q3M_YARN_OF, the table after eps in
        every layer's qk_par on both cards, max_ctx up to the mode's positions; Q3MoEOne's)"""
        if type(q8) is not bool: raise ValueError(f'Q3MoETP: q8 must be a bool, not {q8!r}')
        self.q8 = q8
        self.rope = _rope_mode(rope, 'Q3MoETP')
        self.F = F = ggufio.GGUFFile(path); self.path = path
        if F.arch != 'qwen3moe': raise ValueError(f'Q3MoETP: {path} is a {F.arch!r} GGUF, not qwen3moe')
        hp = F.hp
        c = shapes(F)
        for k, v in c.items(): setattr(self, k, v)
        self.eps = float(hp('attention.layer_norm_rms_epsilon'))
        assert self.D % 256 == 0 and self.ffn % 256 == 0 and self.k_used <= 64, (self.D, self.ffn, self.k_used)
        assert int(hp('attention.value_length', self.hd)) == self.hd
        self.kv0 = self.nkv // 2 if kv_heads is None else int(kv_heads)
        self.exp_rows0 = int(exp_rows)
        _check_split(c, self.kv0, self.exp_rows0)
        libm = ctypes.CDLL('libm.so.6'); libm.powf.restype = ctypes.c_float; libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]
        self.theta_scale = libm.powf(float(hp('rope.freq_base')), -2.0 / self.hd)
        n_ctx = int(hp('context_length')) if self.rope is None else self.rope.max_ctx   # (YaRN: the mode's positions)
        if isinstance(max_ctx, bool) or not isinstance(max_ctx, (int, np.integer)) or not 1 <= max_ctx <= n_ctx:
            raise ValueError(f'max_ctx {max_ctx!r}: 1 .. {n_ctx}' + ('' if self.rope is not None else ' (the long-context mode, rope=YarnMode, serves more)'))
        if self.rope is not None:
            self.rope_key = self.rope.key()
            self.rope_table = self.rope.table(self.hd, float(hp('rope.freq_base')))
        devices = tuple(int(d) for d in devices)
        if len(devices) != 2 or devices[0] == devices[1]: raise ValueError(f'Q3MoETP: two different devices, not {devices}')
        self.devs, self.max_ctx, self.limit_us = list(devices), int(max_ctx), limit_us
        self.cache_rows = self.max_ctx + k3.Engine.ATS_PAD   # (OP_OWN_ATTS's chunks read up to ATS_PAD - 1 rows past the position)
        one = module or ('dec_q3m_kvq8' if q8 else 'dec_q3m')
        one = {v: k for k, v in k3.Q3M_TP_OF.items()}.get(one, one)          # (a two-card module named: its one-card base)
        if one in k3.Q3M_YARN_MODULES and self.rope is None: raise ValueError(f'Q3MoETP: {one} is a YaRN build: it needs rope=YarnMode')
        if self.rope is not None:   # (the long-context mode: the YaRN build of the module the rest of these checks pick)
            if one not in k3.Q3M_YARN_OF and one not in k3.Q3M_YARN_MODULES:
                raise ValueError(f'Q3MoETP: the long-context mode (YaRN) has no build of {one} ({sorted(k3.Q3M_YARN_OF)})')
            one = k3.Q3M_YARN_OF.get(one, one)
        if one not in k3.Q3M_TP_OF: raise ValueError(f'Q3MoETP: module {one} has no two-card build ({sorted(k3.Q3M_TP_OF)})')
        if (k3.Q3M_TP_OF[one] in k3.Q3M_KV_Q8_MODULES) != q8:   # (before anything is read: the module decides the cache format)
            raise ValueError(f'Q3MoETP: module {k3.Q3M_TP_OF[one]} ' + ('needs q8=True' if not q8 else
                             f'reads F16 rows (the Q8 KV cache runs on {k3.Q3M_TP_OF[k3.Q3M_KV_Q8_OF["dec_q3m"]]} only, or its YaRN build)'))
        self.module = k3.Q3M_TP_OF[one]
        if q8: self.kv_format_id = KV_FORMAT_Q8   # (the engine key carries it: Q8 rows never restore into F16 ones)
        self.emb = F.T['token_embd.weight']
        if self.emb.type != 'Q4_K': raise ValueError(f'Q3MoETP: token_embd is {self.emb.type} (Q4_K only)')
        for t in F.tensors:
            if len(t.shape) >= 2 and t.type not in ('Q4_K', 'Q6_K', 'F32'):
                raise ValueError(f'Q3MoETP: {t.name} is {t.type} (Q4_K / Q6_K matrices: {self.module}\'s formats)')
        self.per_wgp = per_wgp
        pair = lambda v: [int(v[0]), int(v[1])] if isinstance(v, (tuple, list)) else [int(v), int(v)]
        self.rpt_c = pair(rpt); self.rpt_down_c = pair(rpt if rpt_down is None else rpt_down); self.rpt_head = int(rpt_head)
        self.lmin = int(lmin)
        self.splits_cap = int(splits_cap)   # (splits per kv head: the card's grid / its kv heads, at most this)
        if not 1 <= self.splits_cap <= k3.atts_max_splits(self.module):   # (before any allocation: the projection counts the cap's scratch)
            raise ValueError(f'Q3MoETP: splits_cap {splits_cap}: 1 .. {k3.atts_max_splits(self.module)} on {self.module}')
        self.vocab_share, self.local_only = float(vocab_share), bool(local_only)
        assert 0.0 < self.vocab_share < 1.0, vocab_share
        self.vh = vocab_cut(self.vocab, self.vocab_share)
        self.plan = split_plan(self.nh, self.nkv, self.hd, self.n_exp, self.ffn, self.D, self.vocab, self.kv0, self.exp_rows0, self.vh)
        self.kvc = self.plan['kv_heads']; self.exp_rows = self.plan['exp_rows']
        # (the prompt cache's key: the split changes the partial sums' bits)
        self.tp_key = dict(exp_rows=self.exp_rows0, kv_heads=self.kv0, vocab_rows=self.vh)
        # VRAM, every term before the first allocation (the uploader's pinned staging is host memory, checked by it)
        terms = projection_terms(F, self.max_ctx, self.exp_rows0, self.kv0, self.vocab_share, self.splits_cap, q8=q8, yarn=self.rope is not None)
        h2d = upload.mode_of(default=upload.TWO_CARD_DEFAULT)                     # (CAMINUS_H2D: how the weights reach the cards; engine/upload.py)
        self.vram_mib = [sum(t.values()) / 2 ** 20 + 256 + e for t, e in zip(terms, upload.extra_mib(h2d, 2))]
        ex = [float(v) for v in vram_extra_mib]
        if vram_check:
            capacity.check(self.devs, [v + e for v, e in zip(self.vram_mib, ex)],
                           f'Q3MoETP {os.path.basename(path)} (max_ctx {self.max_ctx}: '
                           f'{self.vram_mib[0]:.0f} / {self.vram_mib[1]:.0f} MiB + {ex[0]:.0f} / {ex[1]:.0f} MiB after it)')
        self.h2d = upload.Uploader(self.devs, h2d, upload.UploadLog.from_env(), what=f'Q3MoETP {os.path.basename(path)}')
        hip = _hip()
        for a, b in ((0, 1), (1, 0)):
            hip.hipSetDevice(self.devs[a]); hip.hipDeviceEnablePeerAccess(self.devs[b], 0)
            hip.hipGetLastError()                       # (already enabled: not an error here)
        torch.cuda.set_device(self.devs[0])
        # the server's engine contract (Dense35TP's names): two cards, T = 1, no recurrent state (a checkpoint is KV rows
        # only), dense attention at every depth, no drafter
        self.ntok, self.max_retries, self.bails = 1, int(max_retries), []
        self.fresh, self.poisoned, self.seq, self.token = True, None, 0, 0
        self.broken = None          # (non-finite logits or a refused Q8 KV block: sticky, reset() keeps it - corrupt weights
                                    #  or rows outlive a reset; _broken_error names which)
        self.pos_host = 0
        self.T = F.T
        self.qsa, self.qsa_limit, self.idx_cache, self.dual = False, None, False, False
        self.qsa_from = None
        self.feature_layers, self.feature_card, self.features, self.ple_layers = [], 0, None, None
        self.gdn_keys = 0; self.moe_blocks = 0
        self._during, self._tok = None, False
        self.model_tag = 'qwen3moe'
        self.n_ex = 2 * self.nl
        self._card_ev, self.card_ms = None, None
        self.slots = None
        self._load_logged(); self._build_tapes()

    @staticmethod
    def projection_mib(path, max_ctx, exp_rows=EXP_ROWS, kv_heads=None, vocab_share=VOCAB_SHARE, splits_cap=SPLITS_CAP, q8=False, yarn=False):
        """per card, MiB the engine allocates at max_ctx, from the file's header alone (nothing on a card): __init__'s own
        check counts the same terms (with the CAMINUS_H2D arm's buffer: upload.extra_mib, 0 unless peer). max_ctx past the
        file's context_length is computed, not refused (the engine refuses it outside the long-context mode). q8: the Q8_0
        rows; yarn: the long-context mode's table in each layer's block"""
        F = ggufio.GGUFFile(path)
        try:
            return [sum(t.values()) / 2 ** 20 + 256 + e for t, e in zip(projection_terms(F, max_ctx, exp_rows, kv_heads, vocab_share, splits_cap, q8, yarn),
                                                                       upload.extra_mib(upload.mode_of(default=upload.TWO_CARD_DEFAULT), 2))]
        finally: F.close()

    @staticmethod
    def kv_pos_mib(path, kv_heads=None, q8=False):
        """per card, MiB of KV cache per position (its kv heads of every layer; q8: Q8_0 rows)"""
        F = ggufio.GGUFFile(path)
        try:
            c = shapes(F); kv0 = c['nkv'] // 2 if kv_heads is None else int(kv_heads)
            return [c['nl'] * k3.kv_pos_bytes(k * c['hd'], bool(q8)) / 2 ** 20 for k in (kv0, c['nkv'] - kv0)]
        finally: F.close()

    # -- weights -----------------------------------------------------------------------------------------------------------
    def _split(self, name, rows, cols, R_=None, K_=None):
        """the tensor's two cards' slices in the layout their tiles read: rows[c] / cols[c] = lists of (start, end) ranges
        of rows / columns (values) of card c, None = all; one read of the file per tensor. Q6_K: the planes of the cut
        (engine.own_layout.q6k_planes over its rows: the grouped tiles' layout for experts x rows_c rows). Returns
        [dict(name, fmt, rows, K, w) on card 0, the same on card 1]"""
        t = self.F.T[name]; K = int(t.shape[0]); R, rb = t.rows()
        if (R_ is not None and R != R_) or (K_ is not None and K != K_):
            raise ValueError(f'Q3MoETP: {name} is [{R}, {K}], the tape needs [{R_}, {K_}]')
        _, bs, ts = ggufio.TYPES[t.type_id]
        b = t.pread(0, t.n_bytes).reshape(R, rb)
        out = []
        for c in (0, 1):
            x, Rl, Kl = cut(b, rows[c], cols[c], bs, ts)
            with torch.cuda.device(self.devs[c]):
                if t.type == 'F32':
                    out.append(dict(name=name, fmt='F32', rows=Rl, K=Kl, w=self.h2d.up(np.ascontiguousarray(x.view(np.float32)).reshape(Rl, Kl), name, c, 'cut')))
                else:
                    if t.type not in ('Q4_K', 'Q6_K'): raise ValueError(f'Q3MoETP: {name} is {t.type} (Q4_K / Q6_K)')
                    y = x.reshape(-1)
                    if t.type == 'Q6_K': y = q6k_planes(y, Rl, Kl)
                    out.append(dict(name=name, fmt=t.type, rows=Rl, K=Kl, w=self.h2d.up(np.ascontiguousarray(y), name, c, 'cut')))
            self._kept(name, c, 'cut', out[-1]['w'], (rows, cols))
        return out

    def _vec(self, name, n):
        """an F32 vector replicated on both cards"""
        t = self.F.T[name]
        if t.type != 'F32' or t.n_values != n: raise ValueError(f'Q3MoETP: {name} is {t.type} {list(t.shape)}, the tape needs F32 [{n}]')
        v = t.pread(0, t.n_bytes).view(np.float32)
        out = []
        for c in (0, 1):
            with torch.cuda.device(self.devs[c]): out.append(self.h2d.up(np.ascontiguousarray(v), name, c, 'vec'))
            self._kept(name, c, 'vec', out[-1], ())
        return out

    def _qk_par(self, il):
        """OP_OWN_ATTS's block of layer il on the host: q_norm | k_norm | theta_scale, eps (fp32; Q3MoEOne's), then the YaRN
        table in the long-context mode"""
        p = f'blk.{il}.'
        f = lambda n: self.F.T[n].pread(0, self.F.T[n].n_bytes).view(np.float32)
        for n in (p + 'attn_q_norm.weight', p + 'attn_k_norm.weight'):
            t = self.F.T[n]
            if t.type != 'F32' or t.n_values != self.hd: raise ValueError(f'Q3MoETP: {n} is {t.type} {list(t.shape)}, the tape needs F32 [{self.hd}]')
        return np.ascontiguousarray(np.concatenate([f(p + 'attn_q_norm.weight'), f(p + 'attn_k_norm.weight'),
                                                    np.array([self.theta_scale, self.eps], dtype=np.float32)]
                                                   + ([self.rope_table] if self.rope is not None else [])))   # (YaRN: + freq | mscale)

    def _load(self):
        D, hd, nh, nkv, E, FF = self.D, self.hd, self.nh, self.nkv, self.n_exp, self.ffn
        P = self.plan; both = [None, None]
        self.L, self.state = [], {}
        for il in range(self.nl):
            p = f'blk.{il}.'
            an, fn = self._vec(p + 'attn_norm.weight', D), self._vec(p + 'ffn_norm.weight', D)
            qk = self._qk_par(il); qkp = []
            for c in (0, 1):
                with torch.cuda.device(self.devs[c]): qkp.append(self.h2d.up(qk, p + 'attn_qk_par', c, 'vec'))
                self._kept(p + 'attn_qk_par', c, 'qkpar', qkp[-1], (il,))
            q = self._split(p + 'attn_q.weight', P['q'], both, nh * hd, D)
            k = self._split(p + 'attn_k.weight', P['kv'], both, nkv * hd, D)
            v = self._split(p + 'attn_v.weight', P['kv'], both, nkv * hd, D)
            o = self._split(p + 'attn_output.weight', both, P['attn_output'], D, nh * hd)
            ro = self._split(p + 'ffn_gate_inp.weight', both, both, E, D)
            g = self._split(p + 'ffn_gate_exps.weight', P['exp'], both, E * FF, D)
            u = self._split(p + 'ffn_up_exps.weight', P['exp'], both, E * FF, D)
            dn = self._split(p + 'ffn_down_exps.weight', both, P['exp_k'], E * D, FF)
            dd = []
            for c in (0, 1):
                if (g[c]['fmt'], g[c]['rows'], g[c]['K']) != (u[c]['fmt'], u[c]['rows'], u[c]['K']):
                    raise ValueError(f'Q3MoETP: {p}ffn_gate_exps and ffn_up_exps differ in format or shape (the fused gate | up tile takes one)')
                if ro[c]['fmt'] != 'F32': raise ValueError(f'Q3MoETP: {p}ffn_gate_inp is {ro[c]["fmt"]} (F32: the router\'s matvec)')
                # (q8: uint8 Q8_0 rows of that card's kv heads, kv_pos_bytes; else f16 halves)
                row = k3.kv_pos_bytes(self.kvc[c] * hd, True) if self.q8 else 2 * self.kvc[c] * hd   # (elements per position)
                with torch.cuda.device(self.devs[c]):
                    cache = torch.zeros(self.cache_rows * row, dtype=torch.uint8 if self.q8 else torch.float16, device=f'cuda:{self.devs[c]}')
                self.state[(c, il, 'kv')] = cache[:self.max_ctx * row]   # (the first max_ctx rows: the padding rows are never a position)
                dd.append(dict(attn_norm=an[c], ffn_norm=fn[c], qk_par=qkp[c], q=q[c], k=k[c], v=v[c], o=o[c], router=ro[c],
                               gate=g[c], up=u[c], down=dn[c], cache=cache))
            self.L.append(dd)
        self.out_norm = self._vec('output_norm.weight', D)
        self.head = self._split('output.weight', P['head'], both, self.vocab, D)
        eb = self.emb.pread(0, self.emb.n_bytes)
        self.emb_dev = []
        for c in (0, 1):
            with torch.cuda.device(self.devs[c]): self.emb_dev.append(self.h2d.up(eb, 'token_embd.weight', c, 'emb'))
            self._kept('token_embd.weight', c, 'emb', self.emb_dev[-1], ())
        del eb

    def _load_logged(self):
        """_load in the upload log's frame (engine/upload.py): its start, environment and maps; on an exception its type and
        the maps, then the exception; at the end the arms' buffers released and the maps again"""
        self.uploads = []
        self.h2d.begin('Q3MoETP', self.path)
        try: self._load()
        except BaseException as e:
            self.h2d.failed(e); raise
        self.h2d.finish()

    def _kept(self, name, c, kind, w, args):
        """records a card tensor the engine keeps, with what host_bytes needs to rebuild its bytes from the file
        (tools/gates/tp_load_only.py reads every one back after the load)"""
        self.uploads.append((name, c, kind, w, args))

    def host_bytes(self, name, c, kind, args):
        """card c's tensor `name` (a _kept record) rebuilt on the host from the file, the same steps as the load: the bytes
        it must hold on the card"""
        if kind == 'qkpar': return self._qk_par(args[0])
        t = self.F.T[name]
        if kind == 'emb': return t.pread(0, t.n_bytes)
        if kind == 'vec': return np.ascontiguousarray(t.pread(0, t.n_bytes).view(np.float32))
        rows, cols = args
        R, rb = t.rows(); _, bs, ts = ggufio.TYPES[t.type_id]
        x, Rl, Kl = cut(t.pread(0, t.n_bytes).reshape(R, rb), rows[c], cols[c], bs, ts)
        if t.type == 'F32': return np.ascontiguousarray(x.view(np.float32)).reshape(Rl, Kl)
        y = x.reshape(-1)
        if t.type == 'Q6_K': y = q6k_planes(y, Rl, Kl)
        return np.ascontiguousarray(y)

    # -- the tapes ---------------------------------------------------------------------------------------------------------
    def _build_tapes(self):
        """both cards' buffers and tapes at the current per_wgp / rpt settings; callable again (a grid test, a bench's arms):
        the receive slots are allocated once (the exchange flags only grow: self.token never restarts), each rebuild adds
        TAPE_MIB per card (the caller's vram_extra_mib)"""
        D, hd, KU = self.D, self.hd, self.k_used
        f16 = torch.float16
        if self.slots is None:
            self.slots = [Slots(self.devs[c], self.n_ex, D) for c in (0, 1)]
        # the per-pass inputs, one block of int32 per card (one copy from the shared pinned stage per token): [0] the token,
        # [1] its position, [2] the exchanges' epoch base (token * n_ex: every exchange flag only grows)
        self.hstage = torch.zeros(4, dtype=torch.int32, pin_memory=True)
        self.E, self.buf, self.dstage, self.tok, self.pos, self.epoch, self.splits, self.per_wgp_c = [], [], [], [], [], [], [], []
        for c in (0, 1):
            with torch.cuda.device(self.devs[c]):
                pw = card_per_wgp() if self.per_wgp == 'card' else int(self.per_wgp)
                self.per_wgp_c.append(pw)
                self.splits.append(splits_for(self.kvc[c], self.splits_cap))   # (the card's, never the grid's: the same bits at every grid)
                B = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt, device='cuda')
                QH, M, rc = self.kvc[c] * 8 * hd, self.kvc[c] * hd, self.exp_rows[c]
                nlog = self.vh if c == 0 else self.vocab - self.vh
                qkv = B(QH + 2 * M)   # q | k | v: the matvecs' outputs, one buffer (OP_OWN_ATTS reads it whole)
                s = dict(x=B(D), h=B(D), h16=B(D, f16), qkv=qkv, q=qkv[:QH], kv=qkv[QH:],
                         ats=B(k3.Engine.atts_scratch_floats(self.kvc[c], self.splits[c])), att16=B(QH, f16),
                         op=B(D), o=B(D), logit=B(self.n_exp), ids=B(self.nl * KU, torch.int32), wts=B(self.nl * KU),
                         a16=B(KU * rc, f16), d=B(KU * D), fp=B(D), f=B(D), logits=B(nlog))
                assert sum(v.numel() * v.element_size() for k_, v in s.items() if k_ not in ('q', 'kv')) <= TAPE_MIB * 2 ** 20, 'TAPE_MIB'
                self.buf.append(s)
                ds = B(4, torch.int32); self.dstage.append(ds)
                self.tok.append(ds[0:1]); self.pos.append(ds[1:2]); self.epoch.append(ds[2:3])
                self.E.append(k3.Engine(module=self.module, grid=k3.resident_grid(self.module, pw), T=1, limit_us=self.limit_us))
        self.seg_names, self.layer_seg = [{}, {}], [[], []]
        for c in (0, 1):
            with torch.cuda.device(self.devs[c]): self._tape(c)
        self._server_buffers()

    def _tape(self, c):
        D, hd, KU, E_, FF = self.D, self.hd, self.k_used, self.n_exp, self.ffn
        kvc, rc = self.kvc[c], self.exp_rows[c]
        QH, M = kvc * 8 * hd, kvc * hd
        E, s, pos = self.E[c], self.buf[c], self.pos[c]
        names = self.seg_names[c]
        rpt, rpd = self.rpt_c[c], self.rpt_down_c[c]

        def named(si, name):
            names[si] = name; return si

        def mv(W, x16, out, dep, rows_per_task=rpt, **kw):
            return named(E.own_matvec_1rt(W['w'], W['fmt'], W['rows'] if 'ids' not in kw else W['rows'] // E_, W['K'], x16, out,
                                          rows_per_task=rows_per_task, dep=dep, **kw), W['name'])

        def xchg(partial, out, x, dep):
            return named(E.xchg(partial, out, self.slots[1 - c].slot(x), self.slots[c].slot(x), self.epoch[c], x, c,
                                local_only=self.local_only, dep=dep), f'xchg {x}')

        prev = E.own_emb(self.emb_dev[c], 'Q4_K', int(self.emb.shape[1]), D, self.tok[c], s['x'])
        for il, dd in enumerate(self.L):
            d = dd[c]
            ids, wts = s['ids'][il * KU:(il + 1) * KU], s['wts'][il * KU:(il + 1) * KU]   # (each layer's choice kept: check_replicas)
            self.layer_seg[c].append((il, len(E.segs)))
            # the attention's norm, after the previous layer's FFN exchange is added to the residual (layer 0: no add)
            n1 = E.own_add_rms16(s['x'], d['attn_norm'], s['h'], s['h16'], self.eps, d=s['f'] if il else None, slots=1 if il else 0, dep=prev)
            # v and k before q, k | v behind a no-op join (Q3MoEOne._tape_fused's order and its reason)
            v_ = mv(d['v'], s['h16'], s['kv'][M:], n1)
            k_ = mv(d['k'], s['h16'], s['kv'][:M], n1)
            jn = E.also(E.nop(1, 0, dep=k_), v_)
            q_ = mv(d['q'], s['h16'], s['q'], n1)
            at = E.also(E.own_atts(s['qkv'], s['ats'], d['qk_par'], d['cache'], pos, s['att16'], 8 * kvc, kvc, hd, 1.0 / np.sqrt(hd),
                                   splits=self.splits[c], lmin=self.lmin, dep=q_), jn)
            o_ = mv(d['o'], s['att16'], s['op'], at)
            x1 = xchg(s['op'], s['o'], 2 * il, o_)
            n2 = E.own_add_rms16(s['x'], d['ffn_norm'], s['h'], s['h16'], self.eps, d=s['o'], slots=1, dep=x1)
            rl = named(E.f32_mv(d['router']['w'], s['h'], s['logit'], dep=n2), d['router']['name'])
            ro = E.router(s['logit'], KU, ids, wts, dep=rl)
            sw = mv(d['gate'], s['h16'], s['a16'], ro, ids=ids, total_rows=E_ * rc, up=d['up']['w'])
            dn = mv(d['down'], s['a16'], s['d'], sw, rows_per_task=rpd, ids=ids, total_rows=E_ * D, x_per_slot=True)
            ss = E.own_slot_sum(s['d'], s['fp'], KU, wts, dep=dn)
            prev = xchg(s['fp'], s['f'], 2 * il + 1, ss)
        self.layer_seg[c].append(('head', len(E.segs)))
        nf = E.own_add_rms16(s['x'], self.out_norm[c], s['h'], s['h16'], self.eps, d=s['f'], slots=1, dep=prev)
        mv(self.head[c], s['h16'], s['logits'], nf, rows_per_task=self.rpt_head)
        E.build()

    def _server_buffers(self):
        """the server contract's views and staging (Dense35TP's names): logits_half[c] = card c's rows [1, rows], the argmax
        read-backs, the status words, the done events"""
        self.logits_half = [self.buf[0]['logits'].view(1, self.vh), self.buf[1]['logits'].view(1, self.vocab - self.vh)]
        self.logits = None
        self.shard_devs = list(self.devs)
        self.d_max, self.h_max, self.h_st, self.ev_done = [], [], [], []
        for c in (0, 1):
            with torch.cuda.device(self.devs[c]):
                self.d_max.append((torch.zeros(1, device='cuda'), torch.zeros(1, dtype=torch.int64, device='cuda')))
                self.h_max.append((torch.zeros(1, pin_memory=True), torch.zeros(1, dtype=torch.int64, pin_memory=True)))
                self.h_st.append(torch.zeros(k3.STATUS_WORDS + int(self.q8), dtype=torch.int32, pin_memory=True))   # (+ the Q8 KV error word: one copy)
                self.ev_done.append(torch.cuda.Event(enable_timing=self._card_ev is not None))
        self.engines = list(self.E)
        self.res = [self.buf[c]['x'] for c in (0, 1)]

    # -- decode ------------------------------------------------------------------------------------------------------------
    @property
    def staged(self):
        """the position the staging block holds for the next launch"""
        return int(self.hstage[1])

    def reset(self):
        """a new sequence: positions restart at 0 (every KV row is written before any pass reads it); clears poisoned"""
        self.pos_host = 0
        self.seq += 1; self.fresh = True; self.poisoned = None

    def commit_max(self):
        """the most the next pass may commit: 1 (T = 1, no recurrent state to replay)"""
        return 1

    def mark_broken(self, why, error_cls=k3.NonFiniteLogits):
        """a terminal numerical failure: sticky (reset() keeps it); the passes refuse with error_cls. Non-finite logits (the
        default) are reported; a refused Q8 KV encoding (error_cls InvalidKVEncoding) raises it, as Q3MoEOne's does"""
        if self.broken is None: self.broken, self._broken_error = why, error_cls   # (the cache's terminal check names the class)
        print(f'k3 qwen3moe TP: {why}', file=sys.stderr, flush=True)
        if issubclass(error_cls, k3.InvalidKVEncoding): raise error_cls(f'Q3MoETP: {why}')

    def prefetch(self, tokens, positions):
        """nothing to read ahead: the speculative loop's hook"""

    def logits_full(self):
        """the full logits [1, vocab] on card 0 (a fresh tensor: card 1's rows copied over)"""
        return torch.cat([self.logits_half[0], self.logits_half[1].to(f'cuda:{self.devs[0]}')], dim=1)

    def step(self, tok, trace=None):
        """one token at the next position: the full logits (fp32 [vocab] on card 0, a fresh tensor); trace: [card 0's, card
        1's] int64 buffers for the _trace module's timestamps"""
        self._run_tok(int(tok), self.pos_host, 0 if self.fresh else 1, False, trace)
        return self.logits_full()[0]

    def _check(self, tokens, i0, commit, what):
        if self.broken: raise getattr(self, '_broken_error', k3.NonFiniteLogits)(f'{what}: refused: {self.broken}')
        if self.poisoned: raise k3.ContextBoundsError(f'{what}: refused until reset(): {self.poisoned}')
        k3.check_span(i0, 1, self.max_ctx, commit=commit, ntok=1, what=what)
        if tokens is not None and len(tokens) < i0 + 1:
            raise k3.ContextBoundsError(f'{what}: position {i0} but {len(tokens)} tokens')

    def _stage(self, tok, i0):
        if not 0 <= tok < self.vocab: raise ValueError(f'Q3MoETP: token {tok} outside the vocabulary ({self.vocab})')
        self.hstage[0] = tok; self.hstage[1] = int(i0)

    def step_token(self, tokens, i):
        """position i; returns only the greedy next token (each card's max of its rows; card 0's on a tie)"""
        return self._run(tokens, i, 0 if self.fresh else 1, True)[0]

    def verify(self, tokens, i0, commit):
        """position i0 in one pass (commit: the previous pass's tokens to commit, nothing to replay here); returns the logits
        [1, vocab] on card 0 (a fresh tensor)"""
        self._run(tokens, i0, commit, False)
        return self.logits_full()

    def verify_tokens(self, tokens, i0, commit):
        return self._run(tokens, i0, commit, True)

    def launch_tokens(self, tokens, i0, commit):
        """verify()'s pass launched without waiting or reading anything back; then finish_pass(i0, commit)"""
        self._check(tokens, i0, commit, 'Q3MoETP pass')
        self._stage(int(tokens[i0]), i0); self._tok = False; self.fresh = False
        self._pass_launch(i0, commit)

    def finish_pass(self, i0, commit):
        """wait for launch_tokens' pass; a bailed pass runs again before returning. True when it was rerun"""
        bad = self._pass_wait(i0)
        if not bad:
            self.pos_host = i0 + 1; return False
        for attempt in range(1, self.max_retries + 1):
            self.bails.append((i0, attempt - 1, bad))
            print(f'k3 qwen3moe TP: a wait gave up at position {i0} (attempt {attempt}), running the pass again: ' + '; '.join(bad),
                  file=sys.stderr, flush=True)
            bad = self._pass(i0, 0)
            if not bad:
                self.pos_host = i0 + 1; return True
        self.bails.append((i0, self.max_retries, bad))
        raise RuntimeError(f'k3 qwen3moe TP: a wait gave up at position {i0} in {self.max_retries + 1} attempts: ' + '; '.join(bad))

    def _run(self, tokens, i0, commit, tok):
        self._check(tokens, i0, commit, 'Q3MoETP pass')
        return self._run_tok(int(tokens[i0]), i0, commit, tok)

    def _run_tok(self, token, i0, commit, tok, trace=None):
        self._check(None, i0, commit, 'Q3MoETP pass')
        self._stage(token, i0); self.fresh = False; self._tok = tok
        for attempt in range(self.max_retries + 1):
            bad = self._pass(i0, commit if attempt == 0 else 0, trace)
            if not bad: break
            self.bails.append((i0, attempt, bad))
            print(f'k3 qwen3moe TP: a wait gave up at position {i0} (attempt {attempt + 1}), running the pass again: ' + '; '.join(bad),
                  file=sys.stderr, flush=True)
        else:
            raise RuntimeError(f'k3 qwen3moe TP: a wait gave up at position {i0} in {self.max_retries + 1} attempts: ' + '; '.join(bad))
        self.pos_host = i0 + 1
        if not tok: return None
        (v0, x0), (v1, x1) = [(v.numpy(), x.numpy()) for v, x in self.h_max]
        # the first maximum over [card 0's rows | card 1's rows]: card 0's on a tie; a NaN counts as the maximum (argmax)
        return [int(x0[0]) if (v0[0] >= v1[0] or v0[0] != v0[0]) else self.vh + int(x1[0])]

    def _pass(self, i0, commit, trace=None):
        self._pass_launch(i0, commit, trace)
        return self._pass_wait(i0)

    def _pass_launch(self, i0, commit, trace=None):
        """the launches on both cards from the staged block (and each card's max / argmax read back when self._tok)"""
        self._check(None, i0, commit, 'Q3MoETP._pass')
        if int(self.hstage[1]) != i0: raise k3.ContextBoundsError(f'Q3MoETP._pass: staged position {int(self.hstage[1])} != {i0}')
        base = self.token * self.n_ex; self.token += 1           # a fresh epoch per launch (a retry too): exchange flags only grow
        if base + self.n_ex >= 2 ** 31: raise RuntimeError('Q3MoETP: the exchange epoch would pass int32')
        self.hstage[2] = base
        ct = self._card_ev
        for c in (0, 1):
            with torch.cuda.device(self.devs[c]):
                if ct is not None: ct[c].record()
                self.dstage[c].copy_(self.hstage, non_blocking=True)   # (the pinned block is rewritten only after the waits)
                if trace is None: self.E[c].run(check=False)
                else: self.E[c].run(check=False, trace=trace[c])
        for c in (0, 1):
            with torch.cuda.device(self.devs[c]):
                if self._tok:
                    torch.max(self.logits_half[c], dim=1, out=self.d_max[c])
                    self.h_max[c][0].copy_(self.d_max[c][0], non_blocking=True); self.h_max[c][1].copy_(self.d_max[c][1], non_blocking=True)
                self.h_st[c].copy_(self.E[c].status_words(), non_blocking=True)   # (a Q8 engine: the error word in the same copy)
                self.ev_done[c].record()

    def enable_card_timing(self, on=True):
        """per-card pass spans (card_ms after each wait: each card's launch to its done event), as Dense35TP's"""
        if on and self._card_ev is None:
            self._card_ev = []
            for c in (0, 1):
                with torch.cuda.device(self.devs[c]):
                    self.ev_done[c] = torch.cuda.Event(enable_timing=True); self._card_ev.append(torch.cuda.Event(enable_timing=True))
        elif not on and self._card_ev is not None:
            for c in (0, 1):
                with torch.cuda.device(self.devs[c]): self.ev_done[c] = torch.cuda.Event()
            self._card_ev = None
        self.card_ms = None

    def _clear(self, c):
        """after a bail or a bounds report (both cards synced): the sync block and OP_OWN_ATTS's merge counters (a bailed
        pass leaves them wherever its tasks got; every task still ran once, so they are back at zero, but they are cleared
        before the retry, as Q3MoEOne does)"""
        with torch.cuda.device(self.devs[c]):
            torch.cuda.synchronize(); self.E[c].sync.zero_(); self.buf[c]['ats'][:k3.Engine.ATS_CNT].zero_()

    def _pass_wait(self, i0):
        """wait for both cards, read their status; the bounds guard poisons and raises, a wait that gave up is returned"""
        for c in (0, 1): self.ev_done[c].synchronize()
        if self._card_ev is not None:
            self.card_ms = tuple(self._card_ev[c].elapsed_time(self.ev_done[c]) for c in (0, 1))
        if self.q8:   # (first: the bound and bail paths below zero Sync, the error words with it; read with the status words)
            bad_kv = []
            for c in (0, 1):
                try: self.E[c].check_kv_code(int(self.h_st[c][k3.STATUS_WORDS]))
                except k3.InvalidKVEncoding as e: bad_kv.append(f'card {c} (device {self.devs[c]}): {e}')
            if bad_kv: self.mark_broken(f'a refused KV block at pass position {i0}: ' + '; '.join(bad_kv), error_cls=k3.InvalidKVEncoding)
        oob = []
        for c in (0, 1):
            b = k3.bounds_status(self.h_st[c].tolist())
            if b: oob.append(f'card {c} (device {self.devs[c]}): {b}')
        if oob:
            for c in (0, 1): self._clear(c)
            self.poisoned = f'the device bound guard caught an out-of-range value at pass position {i0}: ' + '; '.join(oob)
            raise k3.ContextBoundsError(f'k3 qwen3moe TP: {self.poisoned}')
        bad = []
        for c in (0, 1):
            st = self.h_st[c].tolist()
            if st[1]:
                with torch.cuda.device(self.devs[c]):
                    torch.cuda.synchronize()
                    names = {v: k for k, v in vars(k3).items() if k.startswith('OP_')}
                    bad.append(f'card {c} (device {self.devs[c]}): segment {st[2]} = {names.get(self.E[c].segs[st[2]].op)} '
                               f'{self.seg_names[c].get(st[2], "")} (seen {st[3]}); {self.E[c].stall_report()}')
        if bad:
            for c in (0, 1): self._clear(c)   # (the card that did not bail ran on the other's garbage partial: it reruns too)
        return bad

    def physics_bytes(self):
        """per card: the weight bytes one decode pass reads (its slices, the routed experts at k_used / n_exp of their bytes:
        the average token; the replicated vectors and router; the embedding one row)"""
        out = []
        nb = lambda t: t.numel() * t.element_size()
        for c in (0, 1):
            n = 0.0
            for dd in self.L:
                d = dd[c]
                n += sum(nb(d[k]['w']) for k in ('q', 'k', 'v', 'o', 'router'))
                n += sum(nb(d[k]['w']) for k in ('gate', 'up', 'down')) * self.k_used / self.n_exp
                n += nb(d['attn_norm']) + nb(d['ffn_norm']) + nb(d['qk_par'])
            n += nb(self.head[c]['w']) + nb(self.out_norm[c]) + self.D * 144 // 256
            out.append(float(n))
        return tuple(out)

    def check_replicas(self):
        """the replicated state must be the same bits on both cards: (the residual equal, every layer's last router ids
        equal)"""
        r0, r1 = self.buf[0]['x'].cpu(), self.buf[1]['x'].cpu()
        i0, i1 = self.buf[0]['ids'].cpu(), self.buf[1]['ids'].cpu()
        return bool(torch.equal(r0, r1)), bool(torch.equal(i0, i1))
