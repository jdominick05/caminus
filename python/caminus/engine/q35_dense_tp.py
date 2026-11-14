"""The dense qwen35 line (Swift 1.5 Q4_K_M, ThinkingCap RCO 4.2 bpw) decoding on TWO cards with tensor parallelism, on
Caminus's own code from the GGUF file to the logits: engine/q35_dense.py's reader, weight layouts, tiles and ops (the
one-card engine's tape, block for block), each card running the two-card build of the file's module (k3_q35_swift_tp /
k3_q35_tcap_tp: the same ops plus OP_XCHG, kernels/k3_xchg.h). No gguf-py: the exchange's host side is engine/tp_slots.py.

The split (card c of 2; every cut on a quantization-block boundary of the matrix it cuts):
- Gated DeltaNet: card c takes the key heads ks .. ks + kc - 1 (half of them) and the value heads that read them. The
  value heads are TILED over the key heads (value head h reads key head h % nk, kernels/k3_q35_ops.h OP_GDN_STEP), so
  card c's value heads are {r nk + k : r < nv / nk, ks <= k < ks + kc}, in that order: local value head r kc + (k - ks)
  reads local key head k - ks, the same rule with nk = kc. Its rows of attn_qkv (q, k of its key heads, v of its value
  heads), the conv channels, attn_gate, ssm_alpha / ssm_beta, dt / a, its heads' states, and ssm_out's columns of its value
  heads (nv / nk runs of kc S columns: block-aligned when ks S and kc S are multiples of 256).
- Attention: card c takes kv heads c nkv / 2 .. and their query heads (the GQA ratio kept), rows of attn_q ([q | gate] per
  head), attn_k, attn_v, attn_output's columns of its query heads; its kv heads' f16 KV cache.
- Feed-forward: ffn_gate / ffn_up rows and ffn_down columns f0 .. f1 of card c, in units of 256 (ffn_share: card 0's part).
- Head: the vocabulary rows split at vh (vocab_share, rounded to 128 rows); each card's logits are its rows.
- Replicated on both cards: the embedding (decoded on each card), the norms, the residual.
Each card's attention / Gated DeltaNet output and feed-forward output are partial sums over its heads / rows: two OP_XCHG
per block sum them, card 0's partial first on both cards, so the replicated residual stays the same bits on both
(check_replicas). Exchanges per token: 2 nl (128 for the 64-block models).

The server contract is the one-card engine's and FlashNextTP's (the device sampler's two logits shards on two cards:
logits_half, shard_devs, vh; step / step_token / verify / launch_tokens / finish_pass; reset; state {(card, block, kind)})."""
import ctypes
import os
import sys

import numpy as np
import torch

from .. import ggufio
from ..device import capacity
from . import k3
from .own_layout import q6k_planes, iq_records, IQ_REC, IQ_PAD
from .q35_dense import module_of, kv_module, trunk_blocks, block_kinds, _shapes, RECURRENT, attn_part_elems
from .q3moe_one import card_per_wgp
from .tp_slots import Slots, _hip

TAPE_MIB = 8      # an upper bound of one card's tape buffers


def ffn_cut(ffn, share):
    """card 0's feed-forward rows: share of ffn in units of 256 (at least one unit on each card)"""
    u = ffn // 256
    return 256 * min(u - 1, max(1, int(round(share * u))))


def vocab_cut(vocab, share):
    """card 0's head rows: share of the vocabulary rounded to 128 rows"""
    return min(vocab - 128, max(128, int(round(share * vocab / 128)) * 128))


def gdn_split(NK, NV, S):
    """per card: (first key head, key heads, its value heads in local order); key heads halved"""
    kc = NK // 2; R = NV // NK
    out = []
    for c in (0, 1):
        ks = c * kc
        out.append((ks, kc, [r * NK + k for r in range(R) for k in range(ks, ks + kc)]))
    return out


def split_plan(NK, NV, S, nh, nkv, hd, ffn, f0, vocab, vh):
    """the split as index ranges per card: {matrix role: [card 0's, card 1's]}, each a list of (start, end); rows of the
    row-split matrices, columns (values) of the column-split ones. Also 'vh' (the value heads, local order) and 'ch' (the
    qkv / conv channels: the rows of attn_qkv, in order)"""
    KD = NK * S
    G = gdn_split(NK, NV, S); nhc, nkvc = nh // 2, nkv // 2
    rng = lambda idx, w: [(i * w, (i + 1) * w) for i in idx]
    P = dict(vh=[VH for _, _, VH in G], gdn=G)
    P['qkv'] = [rng(range(ks, ks + kc), S) + [(KD + a, KD + b) for a, b in rng(range(ks, ks + kc), S)] +
                [(2 * KD + a, 2 * KD + b) for a, b in rng(VH, S)] for ks, kc, VH in G]
    P['ch'] = [np.concatenate([np.arange(a, b) for a, b in P['qkv'][c]]) for c in (0, 1)]
    P['gate'] = [rng(VH, S) for _, _, VH in G]                                    # attn_gate rows
    P['ab'] = [rng(VH, 1) for _, _, VH in G]                                      # ssm_alpha / ssm_beta rows
    P['ssm_out'] = [[((r * NK + ks) * S, (r * NK + ks + kc) * S) for r in range(NV // NK)] for ks, kc, _ in G]   # columns
    P['q'] = [rng(range(c * nhc, (c + 1) * nhc), 2 * hd) for c in (0, 1)]
    P['kv'] = [rng(range(c * nkvc, (c + 1) * nkvc), hd) for c in (0, 1)]
    P['attn_output'] = [[(c * nhc * hd, (c + 1) * nhc * hd)] for c in (0, 1)]     # columns
    P['ffn'] = [[(0, f0)], [(f0, ffn)]]                                           # gate / up rows, down columns
    P['head'] = [[(0, vh)], [(vh, vocab)]]
    return P


def cut(b, ranges_r, ranges_c, bs, ts):
    """a matrix's raw GGUF bytes b [rows, row bytes] -> the rows ranges_r (None: all) and columns ranges_c (values; None:
    all) as raw bytes [rows', row bytes']: whole blocks of bs values (ts bytes) only. Returns (bytes 2-D, rows', K')"""
    R = b.shape[0]; K = b.shape[1] // ts * bs
    rr = ranges_r or [(0, R)]; cc = ranges_c or [(0, K)]
    for c0, c1 in cc:
        if c0 % bs or c1 % bs or not 0 <= c0 < c1 <= K: raise ValueError(f'cut: columns {c0} .. {c1} not whole blocks of {bs} in {K}')
    x = np.concatenate([b[r0:r1] for r0, r1 in rr], axis=0)
    x = np.ascontiguousarray(np.concatenate([x[:, c0 // bs * ts:c1 // bs * ts] for c0, c1 in cc], axis=1))
    return x, x.shape[0], sum(c1 - c0 for c0, c1 in cc)


def _projection_bytes(F, max_ctx, ffn_share, vocab_share, spec_T=1, q8=False):
    """bytes per card: its slices of every trunk matrix (exact for the split above), the replicated vectors and embedding,
    its kv heads' f16 KV cache (spec_T - 1 rows past max_ctx: the guard's span), its Gated DeltaNet states (with spec_T
    tokens' saved inputs), the exchange slots (spec_T * D floats each); [card 0, card 1]"""
    c = _shapes(F); nl = c['nl']; kinds = block_kinds(F)
    if q8 and (c['hd'] != 256 or c['nh'] != 6 * c['nkv']):
        raise ValueError('Dense35TP: Q8 requires head_dim 256 and six query heads per KV head')
    mtp = tuple(f'blk.{i}.' for i in range(nl, int(F.hp('block_count'))))
    D, ffn, S, NK, NV = c['D'], c['ffn'], c['S'], c['NK'], c['NV']
    f0 = ffn_cut(ffn, ffn_share); vocab = int(F.T['output.weight'].shape[1]); v0 = vocab_cut(vocab, vocab_share)
    out = [0.0, 0.0]
    for t in F.tensors:
        if t.name.startswith(mtp): continue
        nb = t.n_bytes * (2 if t.type == 'BF16' else 1) + (IQ_PAD if t.type in IQ_REC else 0)
        if len(t.shape) < 2 or t.name in ('token_embd.weight',) or t.name.endswith('_norm.weight') or t.name.endswith('ssm_dt.bias') \
                or t.name.endswith('ssm_a'):
            sh = [nb, nb]                                      # replicated (vectors, norms, the embedding)
        elif t.name == 'output.weight':
            sh = [nb * v0 / vocab, nb * (vocab - v0) / vocab]
        elif '.ffn_' in t.name:
            sh = [nb * f0 / ffn, nb * (ffn - f0) / ffn]
        else:
            sh = [nb / 2, nb / 2]                              # the heads' halves (attention, Gated DeltaNet)
        out[0] += sh[0]; out[1] += sh[1]
    n_attn = sum(1 for g in kinds if not g)
    CHl = 2 * S * (NK // 2) + S * (NV // 2)
    gdn = (nl - n_attn) * (k3.gdn_block_size(NV // 2, S, NK // 2, spec_T) + CHl * 3 + spec_T * CHl) * 4
    kv = n_attn * (int(max_ctx) + spec_T - 1) * k3.kv_pos_bytes((c['nkv'] // 2) * c['hd'], q8)
    slots = 2 * nl * (16 + spec_T * D) * 4
    part = 4 * attn_part_elems(1, c['nh'] // 2, int(max_ctx) + spec_T - 1)   # (the split attention pair's records, per card)
    return [o + gdn + kv + slots + part + (4 if q8 else 0) for o in out]


class Dense35TP:
    TAPE_MIB = TAPE_MIB

    def __init__(self, path, devices=(0, 1), max_ctx=4096, limit_us=200_000, module=None, per_wgp='card', rpt=16, rpt16=32,
                 ffn_share=0.5, vocab_share=0.5, local_only=False, vram_check=True, vram_extra_mib=(0.0, 0.0), max_retries=8,
                 debug_layers=None, spec_T=1, feature_layers=(), feature_card=0, q8=False):
        """path: a dense qwen35 GGUF (engine/q35_dense.py Dense35Own's formats; module None: the two-card build of the module
        module_of picks). devices: the two torch / HIP device indices (card 0 first: its partials come first in every
        sum). ffn_share / vocab_share: card 0's part of the feed-forward rows / the head rows. local_only: TEST-ONLY, every
        exchange copies the card's own partial (no peer: timing and grids without the link; the logits are not the model's).
        vram_extra_mib: per card, what the caller allocates right after (checked with the engine's projection).
        debug_layers: TEST-ONLY, the first n blocks, then the output norm and the head. spec_T: the most tokens a two-card
        verify pass on this engine's state runs (engine/q35_dense_verify_tp.py; 1 .. 4): each Gated DeltaNet block keeps
        that many tokens' saved inputs, the KV caches grow spec_T - 1 rows past max_ctx, the exchange slots hold spec_T * D
        floats, and a pass after a verify pass commits up to spec_T of them (1: the decode engine as served)."""
        self.F = F = ggufio.GGUFFile(path); self.path = path
        if F.arch != 'qwen35': raise ValueError(f'Dense35TP: {path} is a {F.arch!r} GGUF, not qwen35 (dense)')
        if any(n.endswith('ffn_gate_inp.weight') for n in F.T): raise ValueError(f'Dense35TP: {path} has a router tensor: not the dense line')
        hp = F.hp
        for k, v in _shapes(F).items(): setattr(self, k, v)
        self.gdn = block_kinds(F)
        self.eps = float(hp('attention.layer_norm_rms_epsilon'))
        self.vocab = int(F.T['output.weight'].shape[1])
        assert (self.d_conv, self.S) == (4, 128) and self.NV % self.NK == 0 and self.d_inner == self.NV * self.S, _shapes(F)
        assert self.hd % 128 == 0 and self.hd <= 512 and self.nh % self.nkv == 0, (self.nh, self.nkv, self.hd)
        assert self.nrot % 2 == 0 and self.nrot <= self.hd and int(hp('attention.value_length', self.hd)) == self.hd
        assert self.D % 256 == 0 and self.ffn % 512 == 0 and (self.nh * self.hd) % 256 == 0 and self.d_inner % 256 == 0
        # the split's own shape conditions (block-aligned column cuts, whole heads per card)
        assert self.NK % 2 == 0 and ((self.NK // 2) * self.S) % 256 == 0, (self.NK, self.S)
        assert self.nkv % 2 == 0 and ((self.nh // 2) * self.hd) % 256 == 0, (self.nh, self.nkv)
        sec = hp('rope.dimension_sections')
        assert sec is None or sum(int(v) for v in sec) * 2 == self.nrot, (sec, self.nrot)
        libm = ctypes.CDLL('libm.so.6'); libm.powf.restype = ctypes.c_float; libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]
        self.theta_scale = libm.powf(float(hp('rope.freq_base')), -2.0 / self.nrot)
        n_ctx = int(hp('context_length'))
        if isinstance(max_ctx, bool) or not isinstance(max_ctx, (int, np.integer)) or not 1 <= max_ctx <= n_ctx:
            raise ValueError(f'max_ctx {max_ctx!r}: 1 .. {n_ctx}')
        devices = tuple(int(d) for d in devices)
        if len(devices) != 2 or devices[0] == devices[1]: raise ValueError(f'Dense35TP: two different devices, not {devices}')
        if isinstance(spec_T, bool) or not isinstance(spec_T, (int, np.integer)) or not 1 <= spec_T <= 4:
            raise ValueError(f'spec_T {spec_T!r}: 1 .. 4')
        self.devs, self.max_ctx, self.limit_us, self.spec_T = list(devices), int(max_ctx), limit_us, int(spec_T)
        self.kv_rows = self.max_ctx + self.spec_T - 1
        one = module or module_of(F)
        one = {v: k for k, v in k3.Q35_TP_OF.items()}.get(one, one)          # (a two-card module named: its one-card base)
        one = kv_module(F, one, q8)
        if one not in k3.Q35_TP_OF: raise ValueError(f'Dense35TP: module {one} has no two-card build ({sorted(k3.Q35_TP_OF)})')
        self.module = k3.Q35_TP_OF[one]
        self.tcap = self.module in k3.Q35T_MODULES
        self.q8 = bool(q8)
        if self.q8 and (self.hd != 256 or self.nh != 6 * self.nkv):
            raise ValueError('Dense35TP: Q8 requires head_dim 256 and six query heads per KV head')
        if self.q8: self.kv_format_id = k3.Q35_KV_Q8_FORMAT_ID
        self.per_wgp, self.rpt, self.rpt16 = per_wgp, int(rpt), int(rpt16)
        self.ffn_share, self.vocab_share, self.local_only = float(ffn_share), float(vocab_share), bool(local_only)
        assert 0.0 < self.ffn_share < 1.0 and 0.0 < self.vocab_share < 1.0, (ffn_share, vocab_share)
        self.f0 = ffn_cut(self.ffn, self.ffn_share); self.vh = vocab_cut(self.vocab, self.vocab_share)
        self.tp_key = dict(ffn_rows=self.f0, vocab_rows=self.vh)   # (the prompt cache's key: the split changes the partial sums' bits)
        self.nl_build = self.nl if debug_layers is None else int(debug_layers)
        assert 1 <= self.nl_build <= self.nl, debug_layers
        fl = [int(l) for l in feature_layers]
        if any(not 1 <= l < self.nl_build for l in fl) or len(set(fl)) != len(fl):
            raise ValueError(f'Dense35TP: feature_layers {fl} (distinct, 1 .. {self.nl_build - 1})')
        if feature_card not in (0, 1): raise ValueError(f'Dense35TP: feature_card {feature_card}: 0 or 1')
        self.feature_layers, self.feature_card, self.features, self.ple_layers = fl, int(feature_card), None, None
        self.emb = F.T['token_embd.weight']
        if self.emb.type not in (('Q4_K', 'IQ2_S') if self.tcap else ('Q4_K',)):
            raise ValueError(f'Dense35TP: token_embd is {self.emb.type} on module {self.module}')
        self.vram_mib = [b / 2 ** 20 + 256 for b in _projection_bytes(F, self.max_ctx, self.ffn_share, self.vocab_share, self.spec_T, self.q8)]
        self.vram_mib[self.feature_card] += self.feature_mib(self.D, 1, len(fl))
        ex = [float(v) for v in vram_extra_mib]
        if vram_check:
            capacity.check(self.devs, [v + e for v, e in zip(self.vram_mib, ex)],
                           f'Dense35TP {os.path.basename(path)} (max_ctx {self.max_ctx}: '
                           f'{self.vram_mib[0]:.0f} / {self.vram_mib[1]:.0f} MiB + {ex[0]:.0f} / {ex[1]:.0f} MiB after it)')
        hip = _hip()
        for a, b in ((0, 1), (1, 0)):
            hip.hipSetDevice(self.devs[a]); hip.hipDeviceEnablePeerAccess(self.devs[b], 0)
            hip.hipGetLastError()                       # (already enabled: not an error here)
        torch.cuda.set_device(self.devs[0])
        if self.per_wgp == 'card':
            self.per_wgp_c = []
            for d in self.devs:
                with torch.cuda.device(d): self.per_wgp_c.append(card_per_wgp())
        else:
            self.per_wgp_c = [self.per_wgp, self.per_wgp]
        # the server's engine contract (FlashNextTP's names): two cards, T = 1, dense attention at every depth
        self.ntok, self.max_retries, self.bails = 1, int(max_retries), []
        self.fresh, self.poisoned, self.seq, self.token = True, None, 0, 0
        self.broken = None          # (non-finite logits: sticky, reset() keeps it - corrupt weights outlive a reset)
        # spec_T > 1: the tokens of the last pass launched on these states (a decode step 1, a verify pass its T, 0 after a
        # reset or a prefill): the most the next pass may commit
        self.last_T = 0
        self.pos_host = 0
        self.T = F.T
        self.qsa, self.qsa_limit, self.idx_cache, self.dual = False, None, False, False
        self.qsa_from = None
        self.gdn_keys = self.NK // 2; self.moe_blocks = 0
        self._during, self._tok = None, False
        self.model_tag = 'qwen35'
        self.n_ex = 2 * self.nl_build
        self._card_ev, self.card_ms = None, None
        self._load(); self._build_tapes()

    @staticmethod
    def feature_mib(D, T, nfeat):
        """The selected card's exported residuals and the copy segments' fp32 / fp16 norm scratch."""
        return (int(nfeat) * T * D * 4 + T * D * 6) / 2 ** 20 if nfeat else 0.0

    @staticmethod
    def projection_mib(path, max_ctx, ffn_share=0.5, vocab_share=0.5, spec_T=1, nfeat=0, feature_card=0, q8=False):
        """per card, MiB the engine allocates at max_ctx, from the file's header alone (nothing on a card): __init__'s own
        check counts the same terms"""
        F = ggufio.GGUFFile(path)
        try:
            out = [b / 2 ** 20 + 256 for b in _projection_bytes(F, max_ctx, ffn_share, vocab_share, spec_T, q8)]
            out[feature_card] += Dense35TP.feature_mib(_shapes(F)['D'], 1, nfeat)
            return out
        finally: F.close()

    @staticmethod
    def recurrent_mib(path, spec_T=1):
        """per card, MiB of its Gated DeltaNet blocks' states (half the heads, with spec_T tokens' saved inputs): a
        prompt-cache checkpoint's part beside the KV rows"""
        F = ggufio.GGUFFile(path)
        try:
            c = _shapes(F); n_gdn = sum(1 for g in block_kinds(F) if g)
            CHl = 2 * c['S'] * (c['NK'] // 2) + c['S'] * (c['NV'] // 2)
            return n_gdn * (k3.gdn_block_size(c['NV'] // 2, c['S'], c['NK'] // 2, spec_T) + CHl * 3 + spec_T * CHl) * 4 / 2 ** 20
        finally: F.close()

    @staticmethod
    def kv_pos_mib(path, q8=False):
        """per card, MiB of KV cache per position (its kv heads of every attention block)"""
        F = ggufio.GGUFFile(path)
        try:
            c = _shapes(F); n_attn = sum(1 for g in block_kinds(F) if not g)
            return n_attn * k3.kv_pos_bytes((c['nkv'] // 2) * c['hd'], q8) / 2 ** 20
        finally: F.close()

    # -- weights -----------------------------------------------------------------------------------------------------------
    def _split(self, name, rows, cols, R_=None, K_=None):
        """the tensor's two cards' slices in the layout their tiles read: rows[c] / cols[c] = lists of (start, end) ranges
        of rows / columns (values) of card c, None = all; one read of the file per tensor. Returns [dict(name, fmt, rows,
        K, w) on card 0, the same on card 1]"""
        t = self.F.T[name]; K = int(t.shape[0]); R, rb = t.rows()
        if (R_ is not None and R != R_) or (K_ is not None and K != K_):
            raise ValueError(f'Dense35TP: {name} is [{R}, {K}], the tape needs [{R_}, {K_}]')
        _, bs, ts = ggufio.TYPES[t.type_id]
        b = t.pread(0, t.n_bytes).reshape(R, rb)
        out = []
        for c in (0, 1):
            x, Rl, Kl = cut(b, rows[c], cols[c], bs, ts)
            with torch.cuda.device(self.devs[c]):
                if t.type in ('F32', 'BF16'):   # (BF16: widened to F32, exactly)
                    f = x.view(np.float32) if t.type == 'F32' else (x.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
                    out.append(dict(name=name, fmt='F32', rows=Rl, K=Kl, w=torch.from_numpy(np.ascontiguousarray(f).reshape(Rl, Kl)).to('cuda')))
                    continue
                ok = k3.OWN_Q35T_1RT + ('Q8_0',) if self.tcap else ('Q4_K', 'Q8_0', 'Q5_K', 'Q6_K')
                if t.type not in ok: raise ValueError(f'Dense35TP: {name} is {t.type} ({ok} on module {self.module}, or F32 / BF16 alpha / beta)')
                y = x.reshape(-1)
                if t.type == 'Q6_K': y = q6k_planes(y, Rl, Kl)
                elif t.type in IQ_REC: y = iq_records(y, Rl, Kl, t.type)
                out.append(dict(name=name, fmt=t.type, rows=Rl, K=Kl, w=torch.from_numpy(np.ascontiguousarray(y)).to('cuda')))
        return out

    def _vec(self, name, n, idx=None):
        """an F32 vector on both cards (idx: the entries each card takes, [c] lists; None: replicated)"""
        t = self.F.T[name]
        if t.type != 'F32' or t.n_values != n: raise ValueError(f'Dense35TP: {name} is {t.type} {list(t.shape)}, the tape needs F32 [{n}]')
        v = t.pread(0, t.n_bytes).view(np.float32)
        out = []
        for c in (0, 1):
            x = v if idx is None else v[np.asarray(idx[c])]
            with torch.cuda.device(self.devs[c]): out.append(torch.from_numpy(np.ascontiguousarray(x)).to('cuda'))
        return out

    def _load(self):
        D, hd, nh, nkv, S, NK, NV, ffn = self.D, self.hd, self.nh, self.nkv, self.S, self.NK, self.NV, self.ffn
        CH = 2 * S * NK + S * NV
        P = self.plan = split_plan(NK, NV, S, nh, nkv, hd, ffn, self.f0, self.vocab, self.vh)
        G = self.gsplit = P['gdn']; nkvc = nkv // 2
        qkv_rows, vh_rows, vh_one, so_cols, ch_idx = P['qkv'], P['gate'], P['ab'], P['ssm_out'], P['ch']
        q_rows, kv_rows, o_cols, f_rows = P['q'], P['kv'], P['attn_output'], P['ffn']
        self.L, self.state = [], {}
        Z = lambda c, n, dt=torch.float32: torch.zeros(n, dtype=dt, device=f'cuda:{self.devs[c]}')
        rep = lambda name, n: self._vec(name, n)
        for il in range(self.nl_build):
            p = f'blk.{il}.'
            an, pn = rep(p + 'attn_norm.weight', D), rep(p + 'post_attention_norm.weight', D)
            fg = self._split(p + 'ffn_gate.weight', f_rows, [None, None], ffn, D)
            fu = self._split(p + 'ffn_up.weight', f_rows, [None, None], ffn, D)
            fd = self._split(p + 'ffn_down.weight', [None, None], f_rows, D, ffn)
            dd = [dict(attn_norm=an[c], post_norm=pn[c], fg=fg[c], fu=fu[c], fd=fd[c]) for c in (0, 1)]
            if self.gdn[il]:
                qkv = self._split(p + 'attn_qkv.weight', qkv_rows, [None, None], CH, D)
                z = self._split(p + 'attn_gate.weight', vh_rows, [None, None], NV * S, D)
                al = self._split(p + 'ssm_alpha.weight', vh_one, [None, None], NV, D)
                be = self._split(p + 'ssm_beta.weight', vh_one, [None, None], NV, D)
                so = self._split(p + 'ssm_out.weight', [None, None], so_cols, D, NV * S)
                cw_all = self._vec(p + 'ssm_conv1d.weight', CH * 4)
                dt = self._vec(p + 'ssm_dt.bias', NV, [G[c][2] for c in (0, 1)])
                a = self._vec(p + 'ssm_a', NV, [G[c][2] for c in (0, 1)])
                nw = rep(p + 'ssm_norm.weight', S)
                for c in (0, 1):
                    ks, kc, VH = G[c]; CHl = len(ch_idx[c]); NVl = len(VH)
                    with torch.cuda.device(self.devs[c]):
                        cw = cw_all[c].view(CH, 4)[torch.from_numpy(ch_idx[c]).to('cuda')].contiguous()
                    dd[c].update(kind='gdn', qkv=qkv[c], z=z[c], al=al[c], be=be[c], so=so[c], conv_w=cw, dt=dt[c], a=a[c], nw=nw[c],
                                 conv_st=Z(c, CHl * 3).view(CHl, 3), convx=Z(c, self.spec_T * CHl),
                                 gdn=Z(c, k3.gdn_block_size(NVl, S, kc, self.spec_T)), CH=CHl, NV=NVl, NK=kc)
                    self.state.update({(c, il, 'conv'): dd[c]['conv_st'], (c, il, 'convx'): dd[c]['convx'], (c, il, 'gdn'): dd[c]['gdn']})
                del cw_all
            else:
                q = self._split(p + 'attn_q.weight', q_rows, [None, None], nh * 2 * hd, D)
                k = self._split(p + 'attn_k.weight', kv_rows, [None, None], nkv * hd, D)
                v = self._split(p + 'attn_v.weight', kv_rows, [None, None], nkv * hd, D)
                o = self._split(p + 'attn_output.weight', [None, None], o_cols, D, nh * hd)
                qn, kn = rep(p + 'attn_q_norm.weight', hd), rep(p + 'attn_k_norm.weight', hd)
                for c in (0, 1):
                    dd[c].update(kind='attn', q=q[c], k=k[c], v=v[c], o=o[c], qn=qn[c], kn=kn[c],
                                 cache=Z(c, self.kv_rows * k3.kv_pos_bytes(nkvc * hd, self.q8) // (1 if self.q8 else 2),
                                         torch.uint8 if self.q8 else torch.float16))
                    self.state[(c, il, 'kv')] = dd[c]['cache']
            self.L.append(dd)
        self.out_norm = rep('output_norm.weight', D)
        self.head = self._split('output.weight', P['head'], [None, None], self.vocab, D)
        eb = self.emb.pread(0, self.emb.n_bytes)
        self.emb_dev = []
        for c in (0, 1):
            with torch.cuda.device(self.devs[c]): self.emb_dev.append(torch.from_numpy(eb).to('cuda'))
        del eb

    # -- the tapes ---------------------------------------------------------------------------------------------------------
    def _build_tapes(self):
        D, hd, S = self.D, self.hd, self.S
        nhc, nkvc = self.nh // 2, self.nkv // 2
        QHc = nhc * hd
        f16 = torch.float16
        # the per-pass inputs, one block of int32 per card (one copy from the shared pinned stage per token): [0 .. 15] the
        # engine's pass parameters ([1] = the previous pass's tokens to commit), [16] the token, [17] its position, [18] the
        # exchanges' epoch base (token * n_ex: every exchange flag only grows)
        self.hstage = torch.zeros(32, dtype=torch.int32, pin_memory=True)
        self.E, self.buf, self.dstage, self.tok, self.pos, self.epoch, self.slots = [], [], [], [], [], [], []
        for c in (0, 1):
            with torch.cuda.device(self.devs[c]):
                B = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt, device='cuda')
                gl = [d[c] for d in self.L if d[c]['kind'] == 'gdn']
                CHm = max([d['CH'] for d in gl], default=4); NVm = max([d['NV'] for d in gl], default=4)
                KDm = max([d['NK'] * S for d in gl], default=4)
                fc = self.f0 if c == 0 else self.ffn - self.f0
                nlog = self.vh if c == 0 else self.vocab - self.vh
                s = dict(x=B(D), h=B(D), h16=B(D, f16), blkp=B(D), blk=B(D), ffpp=B(D), ffp=B(D), g=B(fc), u=B(fc), a16=B(fc, f16),
                         al=B(NVm), be=B(NVm), gates=B(2 * NVm), qkv=B(CHm), conv=B(CHm), qk=B(2 * KDm), z=B(NVm * S), o=B(NVm * S),
                         fin16=B(NVm * S, f16), qf=B(2 * QHc), kr=B(nkvc * hd), vr=B(nkvc * hd), qo=B(QHc), ko=B(nkvc * hd),
                         att=B(QHc), gat16=B(QHc, f16), logits=B(nlog))
                assert sum(v.numel() * v.element_size() for v in s.values()) <= TAPE_MIB * 2 ** 20, 'TAPE_MIB'
                if self.module in k3.ATTN_SPLIT_MODULES:   # (the split attention pair's records over this card's heads)
                    s['part'] = B(attn_part_elems(1, QHc // hd, self.max_ctx + self.spec_T - 1))
                self.buf.append(s)
                if self.feature_layers and c == self.feature_card:
                    s.update(feat=B(len(self.feature_layers) * D), fh=B(D), fh16=B(D, torch.float16))
                    self.features = s['feat'].view(len(self.feature_layers), 1, D)
                ds = B(32, torch.int32); self.dstage.append(ds)
                self.tok.append(ds[16:17]); self.pos.append(ds[17:18]); self.epoch.append(ds[18:19])
                E = k3.Engine(module=self.module, grid=k3.resident_grid(self.module, self.per_wgp_c[c]), T=1, limit_us=self.limit_us)
                E.params = ds[0:16]
                if self.spec_T > 1: E.guard_T = self.spec_T   # (the commit after a verify pass: up to spec_T tokens)
                self.E.append(E)
            self.slots.append(Slots(self.devs[c], self.n_ex, max(D, self.spec_T * D)))   # (a verify pass exchanges T * D floats)
        self.seg_names, self.layer_seg = [{}, {}], [[], []]
        for c in (0, 1):
            with torch.cuda.device(self.devs[c]): self._tape(c)
        self._server_buffers()

    def _tape(self, c):
        D, hd, S = self.D, self.hd, self.S
        nhc, nkvc = self.nh // 2, self.nkv // 2
        E, s, pos = self.E[c], self.buf[c], self.pos[c]
        q8 = self.q8
        names = self.seg_names[c]
        fslot = {l: k for k, l in enumerate(self.feature_layers)} if c == self.feature_card else {}

        def named(si, name):
            names[si] = name; return si

        def mv(W, x16, out, dep, x32=None):
            if W['fmt'] == 'F32':
                return named(E.f32_mv(W['w'], x32, out, rows_per_task=4, dep=dep), W['name'])
            if W['fmt'] in (k3.OWN_Q35T_1RT if self.tcap else k3.OWN_1RT_FMT):
                return named(E.own_matvec_1rt(W['w'], W['fmt'], W['rows'], W['K'], x16, out, rows_per_task=self.rpt, dep=dep), W['name'])
            return named(E.own_matvec16(W['w'], W['fmt'], W['rows'], W['K'], x16, out, rows_per_task=self.rpt16, dep=dep), W['name'])

        def xchg(partial, out, x, dep):
            return named(E.xchg(partial, out, self.slots[1 - c].slot(x), self.slots[c].slot(x), self.epoch[c], x, c,
                                local_only=self.local_only, dep=dep), f'xchg {x}')

        prev = E.own_emb(self.emb_dev[c], self.emb.type, self.vocab, D, self.tok[c], s['x'])
        for il, dd in enumerate(self.L):
            d = dd[c]
            self.layer_seg[c].append((il, len(E.segs)))
            n1 = E.own_add_rms16(s['x'], d['attn_norm'], s['h'], s['h16'], self.eps, d=s['ffp'] if il else None,
                                 slots=1 if il else 0, dep=prev)
            fc = None
            if il in fslot:
                k = fslot[il]
                fc = E.own_add_rms16(s['feat'][k * D:(k + 1) * D], d['attn_norm'], s['fh'], s['fh16'], self.eps,
                                     d=s['x'], slots=1, dep=n1)
            if d['kind'] == 'gdn':
                NV, NK, CH = d['NV'], d['NK'], d['CH']; KD = NK * S
                a_ = mv(d['al'], s['h16'], s['al'], n1, x32=s['h']); b_ = mv(d['be'], s['h16'], s['be'], n1, x32=s['h'])
                gt = E.also(E.gdn_gates(s['al'][:NV], s['be'][:NV], d['dt'], d['a'], s['gates'][:2 * NV], dep=a_), b_)
                q_ = mv(d['qkv'], s['h16'], s['qkv'], n1)
                c_ = E.gdn_conv(s['qkv'][:CH], d['conv_w'], d['conv_st'], s['conv'][:CH], d['convx'], dep=q_)
                l_ = E.gdn_l2(s['conv'], s['qk'], 2 * NK, S, self.eps, x_stride=CH, dep=c_)
                z_ = mv(d['z'], s['h16'], s['z'], n1)
                st = E.also(E.gdn_step(s['qk'], s['conv'][2 * KD:], s['gates'], d['gdn'], s['o'], NV, S, NK, CH, dep=l_), gt)
                sv = E.gdn_save(s['qk'], s['conv'][2 * KD:], s['gates'], d['gdn'], NV, S, NK, CH, dep=st)
                nm = E.also(E.gdn_norm16(s['o'][:NV * S], d['nw'], s['z'][:NV * S], s['fin16'][:NV * S], NV, S, self.eps, dep=st), z_)
                o_ = E.also(mv(d['so'], s['fin16'], s['blkp'], nm), sv)
            else:
                k_ = mv(d['k'], s['h16'], s['kr'], n1); v_ = mv(d['v'], s['h16'], s['vr'], n1)
                kd = E.qk_nr(s['kr'], hd, nkvc, hd, d['kn'], pos, s['ko'], self.nrot, self.theta_scale, self.eps, dep=k_)
                if q8:
                    kd = E.hadamard(s['ko'], 256, dep=kd)
                    v_ = E.hadamard(s['vr'], 64, dep=v_)
                ap = E.also(E.kv_append(s['ko'], s['vr'], d['cache'], pos, q8=q8, dep=kd), v_)
                q_ = mv(d['q'], s['h16'], s['qf'], n1)
                qd = E.qk_nr(s['qf'], 2 * hd, nhc, hd, d['qn'], pos, s['qo'], self.nrot, self.theta_scale, self.eps, dep=q_)
                if q8: qd = E.hadamard(s['qo'], 256, dep=qd)
                akw = dict(max_cells=self.max_ctx + self.spec_T - 1, part=s['part']) if E.split_attn() else dict(batched=True)
                a_ = E.also(E.attn_dec(s['qo'], d['cache'], pos, s['att'], nhc, nkvc, hd, 1.0 / np.sqrt(hd), q8=q8, dep=ap, **akw), qd)
                if q8: a_ = E.hadamard(s['att'], 64, dep=a_)
                g_ = E.gate_mul16(s['att'], s['qf'], 2 * hd, hd, s['gat16'], hd, dep=a_)
                o_ = mv(d['o'], s['gat16'], s['blkp'], g_)
            x1 = xchg(s['blkp'], s['blk'], 2 * il, o_)
            n2 = E.own_add_rms16(s['x'], d['post_norm'], s['h'], s['h16'], self.eps, d=s['blk'], slots=1, dep=x1)
            if fc is not None: E.also(n2, fc)   # the feature copy must finish before the next writer of x
            g_ = mv(d['fg'], s['h16'], s['g'], n2); u_ = mv(d['fu'], s['h16'], s['u'], n2)
            sw = E.also(E.own_swiglu16(s['g'], s['u'], s['a16'], dep=u_), g_)
            f_ = mv(d['fd'], s['a16'], s['ffpp'], sw)
            prev = xchg(s['ffpp'], s['ffp'], 2 * il + 1, f_)
        self.layer_seg[c].append(('head', len(E.segs)))
        nf = E.own_add_rms16(s['x'], self.out_norm[c], s['h'], s['h16'], self.eps, d=s['ffp'], slots=1, dep=prev)
        mv(self.head[c], s['h16'], s['logits'], nf)
        E.build()

    def _server_buffers(self):
        """the server contract's views and staging (FlashNextTP's names): logits_half[c] = card c's rows [1, rows], the
        argmax read-backs, the status words, the done events"""
        self.logits_half = [self.buf[0]['logits'].view(1, self.vh), self.buf[1]['logits'].view(1, self.vocab - self.vh)]
        self.logits = None
        self.shard_devs = list(self.devs)
        self.d_max, self.h_max, self.h_st, self.ev_done = [], [], [], []
        for c in (0, 1):
            with torch.cuda.device(self.devs[c]):
                self.d_max.append((torch.zeros(1, device='cuda'), torch.zeros(1, dtype=torch.int64, device='cuda')))
                self.h_max.append((torch.zeros(1, pin_memory=True), torch.zeros(1, dtype=torch.int64, pin_memory=True)))
                self.h_st.append(torch.zeros(k3.STATUS_WORDS, dtype=torch.int32, pin_memory=True))
                self.ev_done.append(torch.cuda.Event())
        self.engines = list(self.E)
        self.res = [self.buf[c]['x'] for c in (0, 1)]

    # -- decode ------------------------------------------------------------------------------------------------------------
    @property
    def staged(self):
        """the position the staging block holds for the next launch"""
        return int(self.hstage[17])

    def reset(self):
        """a new sequence: zero every recurrent state on both cards (positions restart at 0); clears poisoned"""
        for (c, _, kind), t in self.state.items():
            if kind in RECURRENT:
                with torch.cuda.device(self.devs[c]): t.zero_()
        self.pos_host = 0; self.last_T = 0
        self.seq += 1; self.fresh = True; self.poisoned = None

    def commit_max(self):
        """the most the next pass on these states may commit: 1 on the served engine (spec_T 1, as before verify tapes
        existed); with spec_T > 1 the tokens of the last pass launched on them (0 after a reset or a prefill)"""
        if self.spec_T == 1: return 1
        return 0 if self.fresh else self.last_T

    def mark_broken(self, why, error_cls=None):
        """non-finite logits: sticky (reset() keeps it); the passes refuse"""
        if self.broken is None: self.broken = why
        if error_cls is not None:
            self._broken_error = error_cls
            raise error_cls(f'Dense35TP: {self.broken}')
        print(f'k3 qwen35 TP: {why}', file=sys.stderr, flush=True)

    def prefetch(self, tokens, positions):
        """nothing to read ahead: the speculative loop's hook"""

    def logits_full(self):
        """the full logits [1, vocab] on card 0 (a fresh tensor: card 1's rows copied over)"""
        return torch.cat([self.logits_half[0], self.logits_half[1].to(f'cuda:{self.devs[0]}')], dim=1)

    def step(self, tok, trace=None):
        """one token at the next position (the previous token committed): the full logits (fp32 [vocab] on card 0, a fresh
        tensor); trace: [card 0's, card 1's] int64 buffers for the _trace modules' timestamps"""
        self._run_tok(int(tok), self.pos_host, 0 if self.fresh else 1, False, trace)
        return self.logits_full()[0]

    def _check(self, tokens, i0, commit, what):
        if self.broken: raise getattr(self, '_broken_error', k3.NonFiniteLogits)(f'{what}: refused: {self.broken}')
        if self.poisoned: raise k3.ContextBoundsError(f'{what}: refused until reset(): {self.poisoned}')
        k3.check_span(i0, 1, self.max_ctx, commit=commit, ntok=self.commit_max(), what=what)
        if tokens is not None and len(tokens) < i0 + 1:
            raise k3.ContextBoundsError(f'{what}: position {i0} but {len(tokens)} tokens')

    def _stage(self, tok, i0):
        if not 0 <= tok < self.vocab: raise ValueError(f'Dense35TP: token {tok} outside the vocabulary ({self.vocab})')
        self.hstage[16] = tok; self.hstage[17] = int(i0)

    def step_token(self, tokens, i):
        """position i; returns only the greedy next token (each card's max of its rows; card 0's on a tie)"""
        return self._run(tokens, i, 0 if self.fresh else 1, True)[0]

    def verify(self, tokens, i0, commit):
        """position i0 in one pass, after committing the previous pass's first `commit` tokens (0 or 1); returns the logits
        [1, vocab] on card 0 (a fresh tensor)"""
        self._run(tokens, i0, commit, False)
        return self.logits_full()

    def verify_tokens(self, tokens, i0, commit):
        return self._run(tokens, i0, commit, True)

    def launch_tokens(self, tokens, i0, commit):
        """verify()'s pass launched without waiting or reading anything back; then finish_pass(i0, commit)"""
        self._check(tokens, i0, commit, 'Dense35TP pass')
        self._stage(int(tokens[i0]), i0); self._tok = False; self.fresh = False
        self._pass_launch(i0, commit)

    def finish_pass(self, i0, commit):
        """wait for launch_tokens' pass; a bailed pass runs again before returning. True when it was rerun"""
        bad = self._pass_wait(i0)
        if not bad:
            self.pos_host = i0 + 1; self.last_T = 1; return False
        for attempt in range(1, self.max_retries + 1):
            self.bails.append((i0, attempt - 1, bad))
            print(f'k3 qwen35 TP: a wait gave up at position {i0} (attempt {attempt}), running the pass again: ' + '; '.join(bad),
                  file=sys.stderr, flush=True)
            bad = self._pass(i0, 0)
            if not bad:
                self.pos_host = i0 + 1; self.last_T = 1; return True
        self.bails.append((i0, self.max_retries, bad))
        raise RuntimeError(f'k3 qwen35 TP: a wait gave up at position {i0} in {self.max_retries + 1} attempts: ' + '; '.join(bad))

    def _run(self, tokens, i0, commit, tok):
        self._check(tokens, i0, commit, 'Dense35TP pass')
        return self._run_tok(int(tokens[i0]), i0, commit, tok)

    def _run_tok(self, token, i0, commit, tok, trace=None):
        self._check(None, i0, commit, 'Dense35TP pass')
        self._stage(token, i0); self.fresh = False; self._tok = tok
        for attempt in range(self.max_retries + 1):
            bad = self._pass(i0, commit if attempt == 0 else 0, trace)   # (a bailed pass committed already)
            if not bad: break
            self.bails.append((i0, attempt, bad))
            print(f'k3 qwen35 TP: a wait gave up at position {i0} (attempt {attempt + 1}), running the pass again: ' + '; '.join(bad),
                  file=sys.stderr, flush=True)
        else:
            raise RuntimeError(f'k3 qwen35 TP: a wait gave up at position {i0} in {self.max_retries + 1} attempts: ' + '; '.join(bad))
        self.pos_host = i0 + 1; self.last_T = 1
        if not tok: return None
        (v0, x0), (v1, x1) = [(v.numpy(), x.numpy()) for v, x in self.h_max]
        # the first maximum over [card 0's rows | card 1's rows]: card 0's on a tie; a NaN counts as the maximum (argmax)
        return [int(x0[0]) if (v0[0] >= v1[0] or v0[0] != v0[0]) else self.vh + int(x1[0])]

    def _pass(self, i0, commit, trace=None):
        self._pass_launch(i0, commit, trace)
        return self._pass_wait(i0)

    def _pass_launch(self, i0, commit, trace=None):
        """the launches on both cards from the staged block (and each card's max / argmax read back when self._tok)"""
        self._check(None, i0, commit, 'Dense35TP._pass')
        if int(self.hstage[17]) != i0: raise k3.ContextBoundsError(f'Dense35TP._pass: staged position {int(self.hstage[17])} != {i0}')
        base = self.token * self.n_ex; self.token += 1           # a fresh epoch per launch (a retry too): exchange flags only grow
        if base + self.n_ex >= 2 ** 31: raise RuntimeError('Dense35TP: the exchange epoch would pass int32')
        self.hstage[1] = int(commit); self.hstage[18] = base
        ct = self._card_ev
        for c in (0, 1):
            with torch.cuda.device(self.devs[c]):
                if ct is not None: ct[c].record()
                self.dstage[c].copy_(self.hstage, non_blocking=True)   # (the pinned block is rewritten only after the waits)
                if self.features is not None and c == self.feature_card: self.features.zero_()
                if trace is None: self.E[c].run(check=False)
                else: self.E[c].run(check=False, trace=trace[c])
        for c in (0, 1):
            with torch.cuda.device(self.devs[c]):
                if self._tok:
                    torch.max(self.logits_half[c], dim=1, out=self.d_max[c])
                    self.h_max[c][0].copy_(self.d_max[c][0], non_blocking=True); self.h_max[c][1].copy_(self.d_max[c][1], non_blocking=True)
                self.h_st[c].copy_(self.E[c].sync[2 * k3.MAXSEG:2 * k3.MAXSEG + k3.STATUS_WORDS], non_blocking=True)
                self.ev_done[c].record()

    def enable_card_timing(self, on=True):
        """per-card pass spans (card_ms after each wait), as FlashNextTP.enable_card_timing"""
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

    def _pass_wait(self, i0):
        """wait for both cards, read their status; the bounds guard poisons and raises, a wait that gave up is returned"""
        for c in (0, 1): self.ev_done[c].synchronize()
        if getattr(self, 'q8', False):
            for c in (0, 1):
                try:
                    self.E[c].check_kv_error()
                except k3.InvalidKVEncoding as exc:
                    self.mark_broken(f'card {c}: {exc}', error_cls=k3.InvalidKVEncoding)
        if self._card_ev is not None:
            self.card_ms = tuple(self._card_ev[c].elapsed_time(self.ev_done[c]) for c in (0, 1))
        oob = []
        for c in (0, 1):
            b = k3.bounds_status(self.h_st[c].tolist())
            if b: oob.append(f'card {c} (device {self.devs[c]}): {b}')
        if oob:
            for c in (0, 1):
                with torch.cuda.device(self.devs[c]): torch.cuda.synchronize(); self.E[c].sync.zero_()
            self.poisoned = f'the device bound guard caught an out-of-range value at pass position {i0}: ' + '; '.join(oob)
            raise k3.ContextBoundsError(f'k3 qwen35 TP: {self.poisoned}')
        bad = []
        for c in (0, 1):
            st = self.h_st[c].tolist()
            if st[1]:
                with torch.cuda.device(self.devs[c]):
                    torch.cuda.synchronize()
                    names = {v: k for k, v in vars(k3).items() if k.startswith('OP_')}
                    bad.append(f'card {c} (device {self.devs[c]}): segment {st[2]} = {names.get(self.E[c].segs[st[2]].op)} '
                               f'{self.seg_names[c].get(st[2], "")} (seen {st[3]}); {self.E[c].stall_report()}')
                    self.E[c].sync.zero_()
        return bad

    def physics_bytes(self):
        """per card: the weight bytes one decode pass reads (its slices, the replicated vectors; the embedding one row)"""
        out = []
        for c in (0, 1):
            n = 0
            for dd in self.L:
                for v in dd[c].values():
                    if isinstance(v, dict) and 'w' in v: n += v['w'].numel() * v['w'].element_size()
                    elif torch.is_tensor(v) and v.dtype == torch.float32 and v.dim() == 1 and v.numel() <= self.D: n += v.numel() * 4
            n += self.head[c]['w'].numel() * self.head[c]['w'].element_size() + self.D * 144 // 256
            out.append(float(n))
        return tuple(out)

    def check_replicas(self):
        """the replicated residual must be the same bits on both cards: (residual equal, True)"""
        r0, r1 = self.buf[0]['x'].cpu(), self.buf[1]['x'].cpu()
        return bool(torch.equal(r0, r1)), True
