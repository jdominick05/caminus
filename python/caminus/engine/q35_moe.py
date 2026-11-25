"""The qwen35moe line (Qwen3.6-35B-A3B: 40 blocks, 256 routed experts top-8 plus a gated shared expert) decoding on one
card on Caminus's own code from the GGUF file to the logits: our reader (caminus.ggufio), our matvec tiles on fp16
activations (the grouped one-round-trip Q2_0 expert tile, the general rows tile for the trunk's Q8_0), the BF16 embedding
table decoded on the device, and the K3 engine's own Gated DeltaNet and gated-attention ops built without llama.cpp's
headers (kernels/ops_q35.h; module dec_q35m). One token per pass (T = 1). The correctness baseline is the ggml
engine (caminus.engine.qwen35_one) on the same file: one token's argmax and top-10 agree, twelve greedy tokens are
identical.

Per block (fp32 residual x; h = rms(x) * norm, h16 = fp16(h): the matvecs' input):
  Gated DeltaNet (three blocks of four):
    alpha, beta = F32 matvecs of h;  gates = softplus(alpha + dt) * a, sigmoid(beta)
    q | k | v = W_qkv h16 -> causal conv over the last 4 inputs, SiLU -> q, k L2-normed per key head
    the gated delta rule per value head (value head h reads key head h % nk; state [128, 128], committed lazily)
    y16 = fp16(rms(o) * ssm_norm * silu(W_gate h16));  x += W_out y16
  gated attention (every fourth block):
    k, v = W h16;  k: per-head rms * k_norm + RoPE (the first n_rot dims);  KV append (f16 cache, [K | V] per position)
    q rows per head [q | gate] = W_q h16;  q: rms * q_norm + RoPE;  attention over 0 .. pos (6 query heads per kv head)
    y16 = fp16(att * sigmoid(gate));  x += W_o y16
  the mixture of experts (every block):
    logits = W_router h;  the top 8 experts (softmax over all 256, renormalised), and the shared expert's weight
    = sigmoid(W_shared_gate h) written into slot 8 of the 9-slot add
    routed: g = W_gate[e] h16, u = W_up[e] h16 (grouped by the router's ids), a16 = fp16(silu(g) * u),
            x[slot e] = W_down[e] a16   (each expert its own input)
    shared: g, u, a16, then x[slot 8] = W_shared_down a16
    h16 = fp16(rms(x) * post_attention_norm) reads all 9 slots, weighted by their router weights
Then rms(x) * output_norm -> fp16 -> the head -> logits (fp32 [vocab]).

The file's blocks past the trunk (block_count less nextn_predict_layers: a multi-token-prediction block) are not read.
Nothing here is measured for speed yet: the grid is the qwen3moe engine's per-card choice and the task sizes are that
engine's defaults."""
import ctypes
import os
import sys

import math

import numpy as np
import torch

from .. import ggufio
from ..device import capacity
from . import k3, upload, dec_q35m_attn
from .own_layout import q6k_planes, iq_records, IQ_REC, IQ_PAD, q2k_records, iq1m_records
from .q3moe_one import card_per_wgp
from .q35_dense import RECURRENT, Q35Host, _wsum, _wsum_np, block_kinds, trunk_blocks   # (the dense line's host pass lifecycle, weight checks and block layout)

MODULE = 'dec_q35m'


def module_of(F):
    """the decode module a qwen35moe file needs: always dec_q35m (kernels/dec_q35m.hip). The GSQ file's routed
    experts are Q2_0, its trunk matrices Q8_0, its router F32 and its alpha / beta BF16: one module reads all of them,
    so there is nothing to pick between."""
    return MODULE


def kv_module(F, module=None, q8=False):
    """Resolve staged cache-specialized objects separately from weight-format selection."""
    if type(q8) is not bool:
        raise ValueError('Moe35Own: q8 must be a bool')
    selected = module or module_of(F)
    if q8:
        if selected in k3.Q35MOE_KV_Q8_MODULES:
            return selected
        if selected not in k3.Q35MOE_KV_Q8_OF:
            raise ValueError(f'Moe35Own: no Q8 KV specialization for {selected}')
        return k3.Q35MOE_KV_Q8_OF[selected]
    if selected in k3.Q35MOE_KV_Q8_MODULES or selected in k3.Q35_KV_Q8_MODULES:
        raise ValueError(f'Moe35Own: {selected} requires q8=True')
    return selected


WSUM_WORDS = 1 << 20   # int32 words per checksum chunk: its int64 cast is the only scratch (8 MiB, inside the 256 MiB working allowance)


def yarn_table(theta_scale, n_rot, factor, n_ctx=None):
    """The rope op's per-dim multipliers and magnitude scale, from caminus.rope's one YaRN recipe.

    `rope.yarn_freqs` gives each pair's frequency (the angle at position p is p * freq_j); this op computes
    theta_scale**j = base**(-2j/n_rot) itself, so the multiplier it multiplies in is freq_j divided by that. Returns
    (m fp32 [n_rot / 2], mscale): the op multiplies its angle by m(j) and cos / sin by mscale.
    """
    from ..rope import yarn_freqs, mscale
    base = theta_scale ** (-n_rot / 2.0)   # (theta_scale = base ** (-2 / n_rot))
    j = np.arange(n_rot // 2, dtype=np.float64)
    inv = base ** (-2.0 * j / n_rot)
    return (yarn_freqs(n_rot, base, factor, n_ctx) / inv).astype(np.float32), mscale(factor)


def _shapes(F):
    """the header's shapes the engine and its projection read (qwen35moe: no dense feed_forward_length; the experts'
    and the shared expert's widths are separate keys)"""
    hp = F.hp
    return dict(nl=trunk_blocks(F), D=int(hp('embedding_length')), ffn=int(hp('expert_feed_forward_length')),
                ffn_shexp=int(hp('expert_shared_feed_forward_length')), n_exp=int(hp('expert_count')),
                k_used=int(hp('expert_used_count')), nh=int(hp('attention.head_count')),
                nkv=int(hp('attention.head_count_kv')), hd=int(hp('attention.key_length')), nrot=int(hp('rope.dimension_count')),
                S=int(hp('ssm.state_size')), NK=int(hp('ssm.group_count')), NV=int(hp('ssm.time_step_rank')),
                d_conv=int(hp('ssm.conv_kernel')), d_inner=int(hp('ssm.inner_size')))


def attn_part_elems(T, heads, cells, module='dec_q35m'):
    """floats of the split attention pair's records (kernels/ops_attn_split.h, the own split modules: k3.Q35_SPLIT_MODULES)
    for T pass rows of `heads` query heads over lists of up to `cells` cache rows: one record of k3.ATTN_RS floats per
    (row, head, ATTN_SS-cell split); one buffer shared by a tape's attention blocks (they run one after another)"""
    SS = k3.ATTN_SS_OF.get(module, k3.ATTN_SS)   # (this line is 128: kernels/dec_q35m.hip sets ATTN_SS)
    return int(T) * int(heads) * ((int(cells) + SS - 1) // SS) * k3.ATTN_RS


def _projection_bytes(F, max_ctx, spec_T=1, q8=False):
    """bytes the engine allocates on its card: every trunk tensor (the embedding and the head too), the selected KV cache of
    every attention block, the Gated DeltaNet states (with spec_T tokens' saved inputs: Moe35Own's spec_T), the decode
    tape's split attention records (attn_part_elems)"""
    c = _shapes(F); nl = c['nl']
    mtp = tuple(f'blk.{i}.' for i in range(nl, int(F.hp('block_count'))))
    w = sum(t.n_bytes for t in F.tensors if not t.name.startswith(mtp))
    w += sum(t.n_bytes for t in F.tensors if not t.name.startswith(mtp) and t.type == 'BF16'
             and t.name != 'token_embd.weight')   # (token_embd goes up as BF16: never widened)   # (widened to F32 on load)
    w += IQ_PAD * sum(1 for t in F.tensors if not t.name.startswith(mtp) and (t.type in IQ_REC or t.type in ('Q2_K', 'IQ1_M')))   # (the record layout's tail)
    w += sum(t.n_bytes // 56 * 10 for t in F.tensors if not t.name.startswith(mtp) and t.type == 'IQ1_M')   # (its records: 66 bytes per 56-byte block)
    n_attn = sum(1 for g in block_kinds(F) if not g)
    CH = 2 * c['S'] * c['NK'] + c['S'] * c['NV']
    gdn = (nl - n_attn) * (k3.gdn_block_size(c['NV'], c['S'], c['NK'], spec_T) + CH * 3 + spec_T * CH) * 4
    part = 4 * attn_part_elems(1, c['nh'], int(max_ctx) + spec_T - 1,
                               module=k3.Q35MOE_KV_Q8_OF['dec_q35m'] if q8 else 'dec_q35m')
    return (w + n_attn * (int(max_ctx) + spec_T - 1) * k3.kv_pos_bytes(c['nkv'] * c['hd'], q8) + gdn + part + (4 if q8 else 0)
            + sliced_bytes(n_attn))


def sliced_bytes(n_attn):
    """bytes the sliced pass adds beside the single-launch tape (Moe35Own._build_slices): the 2 n_attn + 1 launches' sync
    rows (k3.Engine.split), their copy of the tape (at most MAXSEG segments in all) and the attention kernel's workspace
    (engine/dec_q35m_attn.py, NS_MAX splits); its error word lies in the first sync row"""
    return ((2 * n_attn + 1) * (k3.SYNC_WORDS + 8) * 4 + k3.MAXSEG * ctypes.sizeof(k3.Seg)
            + dec_q35m_attn.workspace_floats(dec_q35m_attn.NS_MAX) * 4)


# From this position on a pass runs as slices around each attention layer with the attention as its own kernel
# (Moe35Own._build_slices, each attention block's prefetch behind its pair); below it, as one launch. Keyed by Q8 KV.
# Measured with benchmarks/q3moe_depth.py --engine q35moe (XT, 210 workgroups, interleaved, median of 3 rounds): the
# sliced pass costs 0.26 ms/token more at depth 0 (21 k3_run launches + 10 attention launches for one; a bare launch
# boundary is ~5 us, plus the tasks one launch overlaps with the next segments) and gains with depth (its slope 0.029
# us per position per token F16, 0.020 Q8, against the single launch's 0.046 / 0.045). F16 sliced - one: +0.24 ms at
# 8,192, +0.02 at 16,384, -0.10 at 24,576, -0.39 at 32,768, -2.01 at 131,072 (8.83 vs 10.83 ms). Q8: +0.13 at 12,288,
# -0.03 at 16,384, -0.06 at 20,480, -0.48 at 32,768, -3.02 at 131,072 (7.74 vs 10.76). The switch sits at each format's
# first depth with a gain past 0.05 ms.
ATTN_FROM = {False: 24576, True: 20480}


class Moe35Own(Q35Host):
    TAPE_MIB = 8   # an upper bound of one tape's buffers (S, the stage, the logits): what a further tape on the same engine adds

    def __init__(self, path, device=0, max_ctx=4096, limit_us=200_000, module=None, per_wgp='card', rpt=32,
                 rpt16=32, vram_check=True, vram_extra_mib=0.0, max_retries=8, debug_layers=None, spec_T=1,
                 feature_layers=(), q8=False, yarn=None, prefetch=16, prefetch_threads=16, attn_from=None, attn_launch='kernel',
                 attn_splits=None):
        """path: a dense qwen35 GGUF whose matrices are Q4_K / Q6_K / Q8_0 / Q5_K (alpha / beta may be F32; the Swift module)
        or IQ4_XS / IQ3_XXS / IQ3_S / IQ2_S / Q5_K / Q6_K / Q8_0 (alpha / beta BF16; the ThinkingCap module): module None
        picks it from the file's formats (module_of). device: the
        torch / HIP device index. max_ctx: KV cache positions. per_wgp: workgroups per WGP ('card': the qwen3moe engine's
        per-card grid; a number for the grid tests). rpt / rpt16: rows per task of the one-round-trip (the routed experts:
        32 = 4 rows per wave, the module's OWN_K3_RW) / the general rows matvecs (the wave-per-row Q8_0 tile sizes its
        own). debug_layers: TEST-ONLY, the first n blocks, then the output norm and
        the head; after a pass buf['x'] is block n - 1's output residual. spec_T: the most tokens a verify pass on this
        engine's state runs (engine/q35_dense_verify.py; 1 .. 4): each Gated DeltaNet block keeps that many tokens' saved
        inputs, and a pass after a verify pass commits up to spec_T of them (1: the decode engine as served).
        feature_layers: layers whose entering residual each pass copies to self.features [len, 1, D] fp32 (a DFlash
        drafter's target layers, as Dense35Verify(feature_layers=) exports them, so a one-token pass feeds the drafter
        too); the copy changes no other bit. q8: staged internal Q8 target-cache path, with Q/K Hadamard256 and
        V/output Hadamard64; no served option or matching batched prefill yet (docs/dev/q8-kv.md).
        attn_from: the first position whose pass runs sliced (None: ATTN_FROM for the KV format): the tape cut at each attention layer's
        split pair, the layer's attention as its own kernel between the slices (engine/dec_q35m_attn.py; past the
        switch the pair's in-launch attention costs more than the launch boundaries). attn_launch / attn_splits are
        TEST-ONLY: 'pair' runs the pair's own two segments as the launch between the slices (the single launch's bits:
        tests/engine/test_q35m_slices.py), attn_splits the kernel's splits (None: dec_q35m_attn.splits of the depth)."""
        self.F = F = ggufio.GGUFFile(path); self.path = path
        if F.arch != 'qwen35moe': raise ValueError(f'Moe35Own: {path} is a {F.arch!r} GGUF, not qwen35moe')
        if not any(n.endswith('ffn_gate_inp.weight') for n in F.T): raise ValueError(f'Moe35Own: {path} has no router tensor: not qwen35moe')
        hp = F.hp
        for k, v in _shapes(F).items(): setattr(self, k, v)
        self.gdn = block_kinds(F)
        self.eps = float(hp('attention.layer_norm_rms_epsilon'))
        self.vocab = int(F.T['output.weight'].shape[1])
        # the shapes the tape and the module's ops are written for (checked, never assumed)
        assert (self.d_conv, self.S) == (4, 128) and self.NV % self.NK == 0 and self.d_inner == self.NV * self.S, _shapes(F)
        assert self.hd % 128 == 0 and self.hd <= 512 and self.nh % self.nkv == 0, (self.nh, self.nkv, self.hd)
        assert self.nrot % 2 == 0 and self.nrot <= self.hd and int(hp('attention.value_length', self.hd)) == self.hd
        assert self.D % 256 == 0 and self.ffn % 256 == 0 and (self.nh * self.hd) % 256 == 0 and self.d_inner % 256 == 0
        sec = hp('rope.dimension_sections')
        # (the sectioned RoPE with text-only positions, every component equal, is plain NEOX RoPE over n_rot dims)
        assert sec is None or sum(int(v) for v in sec) * 2 == self.nrot, (sec, self.nrot)
        libm = ctypes.CDLL('libm.so.6'); libm.powf.restype = ctypes.c_float; libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]
        self.theta_scale = libm.powf(float(hp('rope.freq_base')), -2.0 / self.nrot)
        n_ctx = int(hp('context_length'))
        # (the module always reads a table: all-ones and mscale 1.0 is the plain rope bit for bit, so the op needs no
        #  NULL branch and keeps its register pin)
        self.yarn = yarn_table(self.theta_scale, self.nrot, float(yarn), n_ctx=n_ctx) if yarn is not None \
            else (np.ones(self.nrot // 2, dtype=np.float32), 1.0)
        self.yarn_factor = yarn
        # (the engine key must tell a native park from a YaRN one: the module name is the same in both modes, so without
        #  this the disk tier could restore KV rows rotated by the other rope into a live slot)
        self.rope_key = None if yarn is None else f'yarn{float(yarn):g}'
        if yarn is not None: n_ctx = int(n_ctx * float(yarn))   # (YaRN stretches the window by its factor)
        if isinstance(max_ctx, bool) or not isinstance(max_ctx, (int, np.integer)) or not 1 <= max_ctx <= n_ctx:
            raise ValueError(f'max_ctx {max_ctx!r}: 1 .. {n_ctx}')
        if isinstance(spec_T, bool) or not isinstance(spec_T, (int, np.integer)) or not 1 <= spec_T <= 4:
            raise ValueError(f'spec_T {spec_T!r}: 1 .. 4')
        self.dev, self.max_ctx, self.limit_us, self.spec_T = int(device), int(max_ctx), limit_us, int(spec_T)
        # Rows in each KV cache: max_ctx plus the verify tail of spec_T - 1 rows.
        # The prompt cache's checkpoint layout divides by this, not by max_ctx.
        self.kv_rows = self.max_ctx + self.spec_T - 1
        self.module = kv_module(F, module, q8)
        self.q8 = q8
        if q8 and (self.hd != 256 or self.nh != 8 * self.nkv):
            raise ValueError('Moe35Own: staged Q8 kernels require head_dim 256 and eight query heads per KV head')
        if q8:
            self.kv_format_id = 'q8_0-v1:qk-hadamard256:v-hadamard64-v1'
        self.tcap = False
        # the formats whose matvecs run on the one-round-trip tiles (own_matvec_1rt); the others on own_matvec16
        self.one_rt = k3.OWN_Q35MOE_1RT
        if self.module not in k3.Q35MOE_MODULES: raise ValueError(f'Moe35Own: module {self.module} ({k3.Q35MOE_MODULES})')
        # (no gate | up fusion on the ThinkingCap module: its feed-forward's gate and up differ in format in most blocks)
        self.per_wgp, self.rpt, self.rpt16 = per_wgp, int(rpt), int(rpt16)
        # prefetch: workgroups that stream weights into the Infinity Cache during each block (OP_PREFETCH, 0 = off), each
        # with prefetch_threads loading threads: from the block's norm on, the block's router and shared expert, then the
        # next block's trunk matrices (the last block: the first block's, for the next token). XT, F16 KV, short context:
        # 5.87 -> 5.03 ms per token at 16 x 16 (4 x 64 5.35, 2 x 256 5.62; 16 x 16 is about 256 misses in flight: more
        # slow the latency-bound ops beside them), the same bits at every setting
        self.prefetch, self.prefetch_threads = int(prefetch), int(prefetch_threads)
        self.attn_from = ATTN_FROM[bool(q8)] if attn_from is None else int(attn_from)
        if attn_launch not in ('kernel', 'pair'): raise ValueError(f"Moe35Own: attn_launch {attn_launch!r} ('kernel' or 'pair')")
        self.attn_launch, self.attn_splits = attn_launch, attn_splits
        self.nl_build = self.nl if debug_layers is None else int(debug_layers)
        assert 1 <= self.nl_build <= self.nl, debug_layers
        self.emb = F.T['token_embd.weight']
        if self.emb.type != 'BF16':
            raise ValueError(f'Moe35Own: token_embd is {self.emb.type}, the tape needs BF16')
        # VRAM, checked against the card's cap before a byte is read (vram_extra_mib: what the caller adds right after)
        self.vram_mib = _projection_bytes(F, self.max_ctx, self.spec_T, q8) / 2 ** 20 + 256
        # (CAMINUS_H2D: how the weights reach the card, engine/upload.py; peer is refused on one card)
        self.h2d = upload.Uploader([self.dev], upload.mode_of(), upload.UploadLog.from_env(), what=f'Moe35Own {os.path.basename(path)}')
        if vram_check:
            capacity.check([self.dev], [self.vram_mib + float(vram_extra_mib)],
                           f'Moe35Own {os.path.basename(path)} (max_ctx {self.max_ctx}: {self.vram_mib:.0f} MiB + {float(vram_extra_mib):.0f} MiB after it)')
        torch.cuda.set_device(self.dev)
        if self.per_wgp == 'card': self.per_wgp = card_per_wgp()
        self.pos_host = 0
        # spec_T > 1: the tokens of the last pass launched on these states (a decode step 1, a verify pass its T, 0 after a
        # reset): the most the next pass may commit, since the records past it are an older pass's
        self.last_T = 0
        # the server's engine contract (Qwen35One's / Q3MoEOne's names): one card, T = 1, dense attention at every depth
        self.ntok, self.max_retries, self.bails = 1, int(max_retries), []
        self.fresh, self.poisoned, self.seq, self.token = True, None, 0, 0
        self.broken = None          # terminal numerical failure; reset keeps it
        self.devs = [self.dev]
        self.T = F.T
        self.qsa, self.qsa_limit, self.idx_cache, self.dual, self.local_only = False, None, False, False, True
        self.qsa_from = None
        fl = [int(l) for l in feature_layers]
        if any(not 1 <= l < self.nl_build for l in fl) or len(set(fl)) != len(fl):
            raise ValueError(f'Moe35Own: feature_layers {fl} (distinct, 1 .. {self.nl_build - 1})')
        self.feature_layers, self.feature_card, self.features, self.ple_layers = fl, 0, None, None
        self.gdn_keys = self.moe_blocks = 0; self.vocab_share = 1.0
        self._during = None
        self.model_tag = 'qwen35'
        with torch.cuda.device(self.dev): self._load_logged(); self._build_tape()

    @staticmethod
    def projection_mib(path, max_ctx, spec_T=1, q8=False):
        """MiB the engine allocates on its card at max_ctx, from the file's header alone (nothing on a card): __init__'s
        own check counts the same terms"""
        F = ggufio.GGUFFile(path)
        try: return _projection_bytes(F, max_ctx, spec_T, q8) / 2 ** 20 + 256
        finally: F.close()

    @staticmethod
    def recurrent_mib(path, spec_T=1):
        """MiB of every Gated DeltaNet block's state (the recurrent state with its saved-input record, the conv state and
        its previous inputs): what a prompt-cache checkpoint holds beside the KV rows"""
        F = ggufio.GGUFFile(path)
        try:
            c = _shapes(F); n_gdn = sum(1 for g in block_kinds(F) if g)
            CH = 2 * c['S'] * c['NK'] + c['S'] * c['NV']
            return n_gdn * (k3.gdn_block_size(c['NV'], c['S'], c['NK'], spec_T) + CH * 3 + spec_T * CH) * 4 / 2 ** 20
        finally: F.close()

    @staticmethod
    def kv_pos_mib(path, q8=False):
        """MiB of KV cache per position (every attention block)"""
        F = ggufio.GGUFFile(path)
        try:   # (the KV row itself: a projection difference would also count the split records' steps of ATTN_SS rows)
            c = _shapes(F)
            return sum(1 for g in block_kinds(F) if not g) * k3.kv_pos_bytes(c['nkv'] * c['hd'], q8) / 2 ** 20
        finally: F.close()

    # -- weights -----------------------------------------------------------------------------------------------------------
    def _w(self, name, rows=None, cols=None):
        """a matrix on the card in the layout its tile reads: dict(name, fmt, rows, K, w). Q4_K / Q8_0 / Q5_K: the GGUF's
        blocks; Q6_K: its planes; F32: a float32 [rows, K]. The bytes come through ordinary memory (ggufio.Tensor.pread),
        never the file's mapping"""
        t = self.F.T[name]; K = int(t.shape[0]); R, _ = t.rows()
        if (rows is not None and R != rows) or (cols is not None and K != cols):
            raise ValueError(f'Moe35Own: {name} is [{R}, {K}], the tape needs [{rows}, {cols}]')
        fmt, b = self._w_host(name)
        return dict(name=name, fmt=fmt, rows=R, K=K, w=self.h2d.up(b, name, 0, 'mat'))

    def _w_host(self, name):
        """(format the tile reads, the bytes _w uploads: a host array in that layout)"""
        t = self.F.T[name]; K = int(t.shape[0]); R, _ = t.rows()
        b = t.pread(0, t.n_bytes)
        if t.type in ('F32', 'BF16'):   # (BF16: widened to F32, exactly: the high half of a float32)
            f = b.view(np.float32) if t.type == 'F32' else (b.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            return 'F32', f.reshape(R, K).copy()
        ok = self.one_rt + ('Q8_0',)
        if t.type not in ok: raise ValueError(f'Moe35Own: {name} is {t.type} ({ok} on module {self.module}, or F32 / BF16 alpha / beta)')
        if t.type == 'Q6_K': b = q6k_planes(b, R, K)
        elif t.type in IQ_REC: b = iq_records(b, R, K, t.type)
        elif t.type == 'Q2_K': b = q2k_records(b, R, K)
        elif t.type == 'IQ1_M': b = iq1m_records(b, R, K)
        return t.type, np.ascontiguousarray(b)

    def _f32(self, name, n):
        t = self.F.T[name]
        if t.type != 'F32' or t.n_values != n: raise ValueError(f'Moe35Own: {name} is {t.type} {list(t.shape)}, the tape needs F32 [{n}]')
        return self.h2d.up(t.pread(0, t.n_bytes).view(np.float32), name, 0, 'vec')

    def _f32m(self, name, rows, cols):
        """an F32 matrix [rows, cols] on the card: f32_mv reads w.shape, so the router and the shared expert's gate
        logit cannot go through _f32 (a flat vector)"""
        t = self.F.T[name]
        if t.type != 'F32' or t.n_values != rows * cols:
            raise ValueError(f'Moe35Own: {name} is {t.type} {list(t.shape)}, the tape needs F32 [{rows}, {cols}]')
        a = t.pread(0, t.n_bytes).view(np.float32).reshape(rows, cols)
        return self.h2d.up(np.ascontiguousarray(a), name, 0, 'mat')

    def _wexp(self, name, rows, cols):
        """a routed expert matrix: the whole tensor on the card (experts x rows x cols), the tile told rows = one
        expert's row count (total_rows = experts x rows). The grouped tile indexes ids[e] * rows + r, which is exactly
        the GGUF layout ([cols, rows, experts] read as rows x cols)."""
        t = self.F.T[name]
        R, _ = t.rows()
        if int(t.shape[0]) != cols or R % rows or R != rows * int(t.shape[2]):
            raise ValueError(f'Moe35Own: {name} is {list(t.shape)}, the tape needs [{cols}, {rows}, {R // rows}]')
        fmt, b = self._w_host(name)
        return dict(name=name, fmt=fmt, rows=rows, K=cols, w=self.h2d.up(b, name, 0, 'mat'))

    def _load(self):
        D, hd, nh, nkv, S, NK, NV, ffn = self.D, self.hd, self.nh, self.nkv, self.S, self.NK, self.NV, self.ffn
        CH = 2 * S * NK + S * NV
        Z = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt, device='cuda')
        self.L, self.state = [], {}
        for il in range(self.nl_build):
            p = f'blk.{il}.'
            d = dict(attn_norm=self._f32(p + 'attn_norm.weight', D), post_norm=self._f32(p + 'post_attention_norm.weight', D),
                     router=self._f32m(p + 'ffn_gate_inp.weight', self.n_exp, D),
                     sh_inp=self._f32m(p + 'ffn_gate_inp_shexp.weight', 1, D),
                     sh_gate=self._w(p + 'ffn_gate_shexp.weight', self.ffn_shexp, D),
                     sh_up=self._w(p + 'ffn_up_shexp.weight', self.ffn_shexp, D),
                     sh_down=self._w(p + 'ffn_down_shexp.weight', D, self.ffn_shexp),
                     ffn_gate_exps=self._wexp(p + 'ffn_gate_exps.weight', self.ffn, D),
                     ffn_up_exps=self._wexp(p + 'ffn_up_exps.weight', self.ffn, D),
                     ffn_down_exps=self._wexp(p + 'ffn_down_exps.weight', D, self.ffn))
            if self.gdn[il]:
                d.update(kind='gdn', qkv=self._w(p + 'attn_qkv.weight', CH, D), z=self._w(p + 'attn_gate.weight', NV * S, D),
                         al=self._w(p + 'ssm_alpha.weight', NV, D), be=self._w(p + 'ssm_beta.weight', NV, D),
                         so=self._w(p + 'ssm_out.weight', D, NV * S), conv_w=self._f32(p + 'ssm_conv1d.weight', CH * 4).view(CH, 4),
                         dt=self._f32(p + 'ssm_dt.bias', NV), a=self._f32(p + 'ssm_a', NV), nw=self._f32(p + 'ssm_norm.weight', S),
                         conv_st=Z(CH * 3).view(CH, 3), convx=Z(self.spec_T * CH), gdn=Z(k3.gdn_block_size(NV, S, NK, self.spec_T)))
                self.state.update({(0, il, 'conv'): d['conv_st'], (0, il, 'convx'): d['convx'], (0, il, 'gdn'): d['gdn']})
            else:
                d.update(kind='attn', q=self._w(p + 'attn_q.weight', nh * 2 * hd, D), k=self._w(p + 'attn_k.weight', nkv * hd, D),
                         v=self._w(p + 'attn_v.weight', nkv * hd, D), o=self._w(p + 'attn_output.weight', D, nh * hd),
                         qn=self._f32(p + 'attn_q_norm.weight', hd), kn=self._f32(p + 'attn_k_norm.weight', hd),
                         cache=(Z(self.kv_rows * k3.kv_pos_bytes(nkv * hd, True), torch.uint8) if self.q8 else
                                Z(self.kv_rows * 2 * nkv * hd, torch.float16)))
                self.state[(0, il, 'kv')] = d['cache']
            self.L.append(d)
        self.out_norm = self._f32('output_norm.weight', D)
        self.head = self._w('output.weight', self.vocab, D)
        self.emb_dev = self.h2d.up(self.emb.pread(0, self.emb.n_bytes), 'token_embd.weight', 0, 'emb')
        self.wsum0 = self.weight_sums()   # (the weights as uploaded: mark_broken compares against these)

    def _load_logged(self):
        """_load in the upload log's frame (engine/upload.py; Dense35TP._load_logged)"""
        self.h2d.begin('Moe35Own', self.path)
        try: self._load()
        except BaseException as e:
            self.h2d.failed(e); raise
        self.h2d.finish()

    # -- the tape ----------------------------------------------------------------------------------------------------------
    def _build_tape(self):
        q8 = bool(getattr(self, 'q8', False))
        D, hd, nh, nkv, S, NK, NV, ffn = self.D, self.hd, self.nh, self.nkv, self.S, self.NK, self.NV, self.ffn
        CH, KD, VD, QH = 2 * S * NK + S * NV, NK * S, NV * S, nh * hd
        B = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt, device='cuda')
        # (the YaRN table on the card once: the rope ops read it by pointer, Seg.aux)
        yy = (torch.tensor(self.yarn[0], dtype=torch.float32, device='cuda'), self.yarn[1])
        f16 = torch.float16
        KU, FF, FS = self.k_used, self.ffn, self.ffn_shexp
        s = self.buf = dict(x=B(D), h=B(D), h16=B(D, f16), blk=B(D),
                          d=B((KU + 1) * D), wts=B(KU + 1), ids=B(KU, torch.int32), logit=B(self.n_exp), sgl=B(1),
                          a16=B(KU * FF, f16),
                          shg=B(FS), shu=B(FS), sha16=B(FS, f16),
                          al=B(NV), be=B(NV), gates=B(2 * NV), qkv=B(CH), conv=B(CH), qk=B(2 * KD), z=B(VD), o=B(VD),
                          fin16=B(VD, f16), qf=B(2 * QH), kr=B(nkv * hd), vr=B(nkv * hd), qo=B(QH), ko=B(nkv * hd),
                          att=B(QH), gat16=B(QH, f16), logits=B(self.vocab))
        nfe = len(self.feature_layers)
        if nfe: s.update(feat=B(nfe * D), fh=B(D), fh16=B(D, f16))   # (the export: feat [nfe, D]; fh / fh16 the copy's scratch norm)
        assert sum(v.numel() * v.element_size() for v in s.values()) <= self.TAPE_MIB * 2 ** 20, 'TAPE_MIB'
        self.features = s['feat'].view(nfe, 1, D) if nfe else None
        fslot = {l: k for k, l in enumerate(self.feature_layers)}
        # the per-pass inputs, one block of int32 (one copy from the pinned stage per token): [0 .. 15] the engine's pass
        # parameters ([1] = the previous pass's tokens to commit into the recurrent states), [16] the token, [17] its position
        self.hstage = torch.zeros(32, dtype=torch.int32, pin_memory=True)
        self.dstage = B(32, torch.int32); self.tok, self.pos = self.dstage[16:17], self.dstage[17:18]
        E = self.E = k3.Engine(module=self.module, grid=k3.resident_grid(self.module, self.per_wgp), T=1, limit_us=self.limit_us)
        E.params = self.dstage[0:16]
        if self.spec_T > 1: E.guard_T = self.spec_T   # (the commit after a verify pass: up to spec_T tokens)
        self.seg_names, self.layer_seg = {}, []

        def named(si, name):
            self.seg_names[si] = name; return si

        def mv(W, x16, out, dep, x32=None, up=None, ids=None, total_rows=None, x_per_slot=False):
            """W's matvec on the fp16 activations x16 (an F32 matrix: on the fp32 x32). ids: the routed experts' grouped
            tile, total_rows = experts x rows, x_per_slot for the down pass (each expert its own input)"""
            if W['fmt'] == 'F32':
                return named(E.f32_mv(W['w'], x32, out, rows_per_task=4, dep=dep), W['name'])
            if ids is not None:
                kw = {} if up is None else dict(up=up['w'])
                return named(E.own_matvec_1rt(W['w'], W['fmt'], W['rows'], W['K'], x16, out, ids=ids, total_rows=total_rows,
                                              x_per_slot=x_per_slot, rows_per_task=self.rpt, dep=dep, **kw), W['name'])
            if W['fmt'] in self.one_rt:
                kw = {} if up is None else dict(up=up['w'])
                return named(E.own_matvec_1rt(W['w'], W['fmt'], W['rows'], W['K'], x16, out, rows_per_task=self.rpt, dep=dep, **kw),
                             W['name'])
            assert up is None
            # Q8_0 at K = 1024 n: the module's wave-per-row tile (kernels/quant.h own_row_q8_0_w), its tasks sized to about
            # one per workgroup in rows of 8 (a wave per row); the 4-rows tile (rpt16) for the rest (the shared expert's down, K 512)
            rpt, row = self.rpt16, W['fmt'] == 'Q8_0' and W['K'] % 1024 == 0
            if row: rpt = -(-(-(-W['rows'] // E.grid)) // 8) * 8
            return named(E.own_matvec16(W['w'], W['fmt'], W['rows'], W['K'], x16, out, rows_per_task=rpt, dep=dep, row=row), W['name'])

        rows = self.max_ctx + self.spec_T - 1   # (the cache rows: the split pair's longest list)

        def akw():
            """the attention segments' arguments: the split pair's records (one buffer for every block of the tape) on the
            split modules, else the one-head op's batched loads"""
            if not E.split_attn(): return dict(batched=True)
            if self._attn_part is None: self._attn_part = B(attn_part_elems(1, nh, rows, module=self.module))
            return dict(max_cells=rows, part=self._attn_part)

        self._attn_part = None
        self.attn_caches = []   # (each attention block's cache, in tape order: the sliced pass's kernel launches read them)
        prev = E.own_emb(self.emb_dev, self.emb.type, self.vocab, D, self.tok, s['x'])
        sv_prev = None   # (the last GDN block's gdn_save)
        for il, d in enumerate(self.L):
            self.layer_seg.append((il, len(E.segs)))
            # the block's norm, after the previous block's feed-forward is added to the residual (block 0: no add)
            n1 = E.own_add_rms16(s['x'], d['attn_norm'], s['h'], s['h16'], self.eps, d=s['d'] if il else None,
                                 slots=(KU + 1) if il else 0, w=s['wts'] if il else None, dep=prev)
            if self.prefetch:
                wt = lambda m: m['w'] if isinstance(m, dict) else m
                # (the last block: the first block's, for the next token; the head's row-tile loads allocate no cache lines)
                nxt = self.L[(il + 1) % self.nl_build]
                ws = [wt(d[k]) for k in ('router', 'sh_inp', 'sh_gate', 'sh_up', 'sh_down')]
                ws += [wt(nxt[k]) for k in (('al', 'be', 'qkv', 'z', 'so') if nxt['kind'] == 'gdn' else ('q', 'k', 'v', 'o'))]
                pf = E.prefetch(ws, self.prefetch, threads=self.prefetch_threads, dep=n1)
                if pf is not None: named(pf, f'prefetch.{il}')
            fc = None
            if il in fslot:   # (feat (zeroed per pass) += x: the residual entering block il, exactly; ordered before n2 rewrites x)
                k = fslot[il]
                fc = E.own_add_rms16(s['feat'][k * D:(k + 1) * D], d['attn_norm'], s['fh'], s['fh16'], self.eps, d=s['x'], slots=1, dep=n1)
            if d['kind'] == 'gdn':
                # (the previous GDN block's save reads gates / conv / qk: their writers here wait for it, gates through beta's
                #  matvec, l2 through conv; the output matvec of that block does not, so the save is off the critical path)
                a_ = mv(d['al'], s['h16'], s['al'], n1, x32=s['h']); b_ = mv(d['be'], s['h16'], s['be'], n1, x32=s['h'])
                if sv_prev is not None: b_ = E.also(b_, sv_prev)
                gt = E.also(E.gdn_gates(s['al'], s['be'], d['dt'], d['a'], s['gates'], dep=a_), b_)
                q_ = mv(d['qkv'], s['h16'], s['qkv'], n1)
                # (T = 1: the conv also L2-norms the q | k heads in place, so they are read from conv: k3 gdn_conv l2)
                fl2 = E.T == 1
                c_ = E.gdn_conv(s['qkv'], d['conv_w'], d['conv_st'], s['conv'], d['convx'], dep=q_,
                                l2=(2 * KD, S, self.eps) if fl2 else None)
                if sv_prev is not None: c_ = E.also(c_, sv_prev)
                l_ = c_ if fl2 else E.gdn_l2(s['conv'], s['qk'], 2 * NK, S, self.eps, x_stride=CH, dep=c_)
                qk = s['conv'] if fl2 else s['qk']
                z_ = mv(d['z'], s['h16'], s['z'], n1)
                st = E.also(E.gdn_step(qk, s['conv'][2 * KD:], s['gates'], d['gdn'], s['o'], NV, S, NK, CH, dep=l_), gt)
                sv = E.gdn_save(qk, s['conv'][2 * KD:], s['gates'], d['gdn'], NV, S, NK, CH, dep=st)
                nm = E.also(E.gdn_norm16(s['o'], d['nw'], s['z'], s['fin16'], NV, S, self.eps, dep=st), z_)
                o_ = mv(d['so'], s['fin16'], s['blk'], nm); sv_prev = sv
            else:
                k_ = mv(d['k'], s['h16'], s['kr'], n1); v_ = mv(d['v'], s['h16'], s['vr'], n1)
                kd = E.qk_nr(s['kr'], hd, nkv, hd, d['kn'], self.pos, s['ko'], self.nrot, self.theta_scale, self.eps, dep=k_, yarn=yy)
                if q8:
                    kd = E.hadamard(s['ko'], 256, dep=kd)
                    v_ = E.hadamard(s['vr'], 64, dep=v_)
                ap = E.also(E.kv_append(s['ko'], s['vr'], d['cache'], self.pos, q8=q8, dep=kd), v_)
                q_ = mv(d['q'], s['h16'], s['qf'], n1)
                qd = E.qk_nr(s['qf'], 2 * hd, nh, hd, d['qn'], self.pos, s['qo'], self.nrot, self.theta_scale, self.eps, dep=q_, yarn=yy)
                if q8: qd = E.hadamard(s['qo'], 256, dep=qd)
                a_ = E.also(E.attn_dec(s['qo'], d['cache'], self.pos, s['att'], nh, nkv, hd, 1.0 / np.sqrt(hd), dep=ap,
                                       q8=q8, **akw()), qd)
                self.attn_caches.append(d['cache'])
                if q8: a_ = E.hadamard(s['att'], 64, dep=a_)
                g_ = E.gate_mul16(s['att'], s['qf'], 2 * hd, hd, s['gat16'], hd, dep=a_)
                o_ = mv(d['o'], s['gat16'], s['blk'], g_)
            n2 = E.own_add_rms16(s['x'], d['post_norm'], s['h'], s['h16'], self.eps, d=s['blk'], slots=1, dep=o_)
            if fc is not None: E.also(n2, fc)
            # the router: F32 logits, softmax top-8 (renormalised) with the gated shared expert's weight as slot KU
            rl = E.f32_mv(d['router'], s['h'], s['logit'], dep=n2)
            sl = E.f32_mv(d['sh_inp'], s['h'], s['sgl'], dep=n2)
            ro = E.also(E.router_shexp(s['logit'], KU, s['ids'], s['wts'], s['sgl'], dep=rl), sl)
            # the gated shared expert, its down pass into slot KU (the routed experts take 0 .. KU - 1)
            sg_ = mv(d['sh_gate'], s['h16'], s['shg'], n2); su_ = mv(d['sh_up'], s['h16'], s['shu'], n2)
            ssw = E.also(E.own_swiglu16(s['shg'], s['shu'], s['sha16'], dep=su_), sg_)
            sh_ = mv(d['sh_down'], s['sha16'], s['d'][KU * D:(KU + 1) * D], ssw)
            # the routed experts: gate | up and their SwiGLU as one segment grouped by the router's ids (OWN_GU: the up's
            # rows no longer queue behind the gate's, no SwiGLU hand-off), the down pass each slot its own input
            sw = mv(d['ffn_gate_exps'], s['h16'], s['a16'], ro, up=d['ffn_up_exps'], ids=s['ids'], total_rows=self.n_exp * FF)
            prev = mv(d['ffn_down_exps'], s['a16'], s['d'][:KU * D], sw, ids=s['ids'], total_rows=self.n_exp * D,
                      x_per_slot=True)
            E.also(prev, sh_)   # (the next block's add reads every slot: it waits for the shared expert's too)
        self.layer_seg.append(('head', len(E.segs)))
        nf = E.own_add_rms16(s['x'], self.out_norm, s['h'], s['h16'], self.eps, d=s['d'], slots=KU + 1, w=s['wts'], dep=prev)
        mv(self.head, s['h16'], s['logits'], nf)
        E.build(); self._server_buffers(); self._build_slices()

    def _build_slices(self):
        """the sliced pass: the tape cut before each attention block's split pair (OP_ATTN_PART, then its OP_ATTN_COMB)
        and after it (k3.Engine.split), so a pass is 2 n + 1 launches: the trunk slices, and between them the pair's
        own launch, in whose place the pass runs the attention kernel (same inputs: q after its rope (and the Q8 tape's
        rotation), the block's cache, the position word; same output). One sync row per launch; the kernel's error word
        sits in the first row's free words, so the pass's one status copy reads it too"""
        E = self.E
        parts = [i for i, sg in enumerate(E.segs) if sg.op == k3.OP_ATTN_PART]
        assert len(parts) == len(self.attn_caches) and all(E.segs[i + 1].op == k3.OP_ATTN_COMB and E.segs[i + 1].dep == i
                                                            for i in parts), parts
        # each attention block's prefetch segment (OP_PREFETCH, it writes nothing) runs behind the block's pair: in one
        # launch its tasks stream the weights while the attention runs, but a launch ends only when every task has, so in
        # front of the cut the attention waited for them. XT, Q8 KV, prefetch 16 x 16, ms per token sliced as is / moved:
        # 5.55 / 5.10 at position 0, 5.95 / 5.53 at 16K, 6.39 / 5.86 at 32K, the same bits (the single launch keeps its
        # order: 4.87 as is, 4.92 moved, at position 0)
        order = list(range(len(E.segs)))
        pfs = [i for i, sg in enumerate(E.segs) if sg.op == k3.OP_PREFETCH]
        for i in parts:
            mine = [p for p in pfs if p < i]
            if not mine: continue
            order.remove(mine[-1]); order.insert(order.index(i + 1) + 1, mine[-1])
        at = {o: k for k, o in enumerate(order)}
        launches, self.slice_block = E.split([c for i in parts for c in (at[i], at[i] + 2)], order=order)
        self.slices, self.pair_slices = launches[0::2], launches[1::2]
        assert all(e.segs[0].op == k3.OP_ATTN_PART and len(e.segs) == 2 for e in self.pair_slices)
        self._tapes = {'one': [E], 'sliced': launches}   # (every engine of the pass's two forms: forcebail, the pass police)
        self.slice_names = [{k: self.seg_names[i] for k, i in enumerate(e.orig) if i in self.seg_names} for e in launches]
        self.h_sts = torch.zeros(len(launches), k3.STATUS_WORDS + 2, dtype=torch.int32, pin_memory=True)
        self.attn = None
        if torch.cuda.is_available():   # (a CPU stand-in builds the tape only: no code object to load)
            self.attn = dec_q35m_attn.DecQ35mAttn(self.q8, err=self.slice_block[0, k3.SYNC_WORDS + 1:k3.SYNC_WORDS + 2])
        self._wgps = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self.sliced = False   # (the last pass's form: _pass_runs reads its launches)

    # -- decode ------------------------------------------------------------------------------------------------------------
    def _stage(self, tok, i0):
        if not 0 <= tok < self.vocab: raise ValueError(f'Moe35Own: token {tok} outside the vocabulary ({self.vocab})')
        self.hstage[16] = tok; self.hstage[17] = int(i0)

    def launch_tokens(self, tokens, i0, commit):
        """verify()'s pass launched without waiting or reading anything back (the device sampler queues behind it); then
        finish_pass(i0, commit)"""
        self._check(tokens, i0, commit, 'Moe35Own pass')
        self._stage(int(tokens[i0]), i0); self._tok = False; self.fresh = False
        self._pass_launch(i0, commit)

    def _run(self, tokens, i0, commit, tok):
        self._check(tokens, i0, commit, 'Moe35Own pass')
        return self._run_tok(int(tokens[i0]), i0, commit, tok)

    def _run_tok(self, token, i0, commit, tok, trace=None):
        self._check(None, i0, commit, 'Moe35Own pass')
        self._stage(token, i0); self.fresh = False; self._tok = tok
        for attempt in range(self.max_retries + 1):
            bad = self._pass(i0, commit if attempt == 0 else 0, trace)   # (a bailed pass committed already: its rerun commits nothing)
            if not bad: break
            self.bails.append((i0, attempt, bad))
            print(f'k3 qwen35: a wait gave up at position {i0} (attempt {attempt + 1}), running the pass again: ' + '; '.join(bad),
                  file=sys.stderr, flush=True)
        else:
            raise RuntimeError(f'k3 qwen35: a wait gave up at position {i0} in {self.max_retries + 1} attempts: ' + '; '.join(bad))
        self.pos_host = i0 + 1
        if not tok: return None
        return [int(self.h_max[1][0])]

    def _pass_launch(self, i0, commit, trace=None):
        """the launch from the staged block (and the logits' argmax read back when self._tok)"""
        self._check(None, i0, commit, 'Moe35Own._pass')
        if int(self.hstage[17]) != i0: raise k3.ContextBoundsError(f'Moe35Own._pass: staged position {int(self.hstage[17])} != {i0}')
        self.token += 1; self.last_T = 1
        self.hstage[1] = int(commit)
        self.sliced = trace is None and i0 >= self.attn_from
        with torch.cuda.device(self.dev):
            self.dstage.copy_(self.hstage, non_blocking=True)   # (the pinned block is rewritten only after the wait below)
            if self.features is not None: self.features.zero_()   # (the copy segments add into it)
            if self.sliced:
                ns = dec_q35m_attn.splits(i0 + 1, self._wgps) if self.attn_splits is None else self.attn_splits
                s, scale = self.buf, 1.0 / np.sqrt(self.hd)
                for j, e in enumerate(self.slices):
                    e.run(check=False)
                    if j == len(self.pair_slices): break
                    if self.attn_launch == 'pair': self.pair_slices[j].run(check=False)
                    else: self.attn(s['qo'], self.attn_caches[j], self.pos, s['att'], scale, ns=ns)
            else:
                self.E.run(check=False) if trace is None else self.E.run(check=False, trace=trace)
            if self._tok:
                torch.max(self.logits, dim=1, out=self.d_max)
                self.h_max[0].copy_(self.d_max[0], non_blocking=True); self.h_max[1].copy_(self.d_max[1], non_blocking=True)
            if self.sliced:   # (every launch's status words and the kernel's error word: one copy of the rows' tails)
                self.h_sts.copy_(self.slice_block[:, 2 * k3.MAXSEG:k3.SYNC_WORDS + 2], non_blocking=True)
            else:
                self.h_st.copy_(self.E.sync[2 * k3.MAXSEG:2 * k3.MAXSEG + k3.STATUS_WORDS], non_blocking=True)
            self.ev_done.record()

    def _pass_runs(self):
        """the launches of the last pass with their status words (Q35Host._pass_wait): the tape, or every slice and pair
        launch of a sliced one (a pair launch that did not run reads zeros). The attention kernel's error word (its
        position past the cache's rows) reads as the bound guard's report on the first launch: k3_run's own guard sees
        the same position and cache rows first, so this is a third check behind the host's and the guard's"""
        if not self.sliced: return super()._pass_runs()
        st = self.h_sts.tolist()
        launches = self._tapes['sliced']
        runs = [(e, row[:k3.STATUS_WORDS], nm) for e, row, nm in zip(launches, st, self.slice_names)]
        if st[0][k3.STATUS_WORDS + 1] and not k3.bounds_status(runs[0][1]):
            runs[0] = (launches[0], runs[0][1][:4] + [1, -1, self.staged, self.kv_rows], runs[0][2])
        return runs

    # -- weight integrity -------------------------------------------------------------------------------------------------
    def mark_broken(self, why, error_cls=k3.NonFiniteLogits):
        """terminal numerical failure: refuse every later pass (reset() keeps this). For non-finite logits, report which
        weights changed since load (corrupted on the card) or, when none did, which differ from the file (corrupted by
        the upload). A refused KV encoding skips that check: the refusal names the cache, and the check reads every weight."""
        self.broken, self._broken_error = why, error_cls   # sticky even if diagnostics fail
        rep = why
        if issubclass(error_cls, k3.InvalidKVEncoding):
            raise error_cls(f'Moe35Own: {rep}')
        try:
            now = self.weight_sums()
            changed = [n for n, v in now.items() if self.wsum0.get(n) != v]
            if changed:
                rep += f'; weights changed on the card since load: {changed[:8]}{" ..." if len(changed) > 8 else ""}'
            else:
                hs = self.host_sums(list(now))
                bad = [n for n, v in hs.items() if v != now[n]]
                rep += (f'; device weights differ from the file (corrupted by the upload): {bad[:8]}' if bad else
                        '; the weights match the load and the file (not a weight corruption)')
        except Exception as e:                       # noqa: BLE001 - the report must not hide the failure
            rep += f'; the weight check failed ({type(e).__name__}: {e})'
        self.broken = rep
        raise error_cls(f'Moe35Own: {rep}')
