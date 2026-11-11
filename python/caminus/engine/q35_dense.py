"""The dense qwen35 line (Qwen3.8-27B finetunes: Swift 1.5 Q4_K_M first) decoding on one card on Caminus's own code from
the GGUF file to the logits: our reader (caminus.ggufio), our weight layout (engine/own_layout.py: Q6_K planes), our
matvec tiles on fp16 activations (the one-round-trip tiles for Q4_K / Q6_K, the general rows tiles for Q8_0 / Q5_K), the
embedding row decoded on the device, and the K3 engine's own Gated DeltaNet and gated-attention ops built without
llama.cpp's headers (kernels/k3_q35_ops.h; module k3_q35_swift). One token per pass (T = 1). The test oracle is
caminus.ref_q35 (float64, the same reading of the model).

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
  h16 = fp16(rms(x) * post_attention_norm);  a16 = fp16(silu(W_gate h16) * (W_up h16));  x += W_down a16
Then rms(x) * output_norm -> fp16 -> the head -> logits (fp32 [vocab]).

The file's blocks past the trunk (block_count less nextn_predict_layers: a multi-token-prediction block) are not read.
Nothing here is measured for speed yet: the grid is the qwen3moe engine's per-card choice and the task sizes are that
engine's defaults."""
import ctypes
import os
import sys

import numpy as np
import torch

from .. import ggufio
from ..device import capacity
from . import k3
from .own_layout import q6k_planes
from .q3moe_one import card_per_wgp

MODULE = 'k3_q35_swift'
RECURRENT = ('conv', 'convx', 'gdn')   # the state kinds reset() zeroes (the KV rows are written before any pass reads them)


def trunk_blocks(F):
    """the number of blocks of the model's forward: block_count less the multi-token-prediction blocks after them"""
    return int(F.hp('block_count')) - int(F.hp('nextn_predict_layers') or 0)


def block_kinds(F):
    """per trunk block: True = Gated DeltaNet, False = gated attention (attention.recurrent_layers where the file has
    it; without that key every block but each full_attention_interval-th)"""
    rec = F.hp('attention.recurrent_layers'); fai = int(F.hp('full_attention_interval', 4))
    return [bool(rec[i]) if rec is not None else (i + 1) % fai != 0 for i in range(trunk_blocks(F))]


def _shapes(F):
    """the header's shapes the engine and its projection read"""
    hp = F.hp
    return dict(nl=trunk_blocks(F), D=int(hp('embedding_length')), ffn=int(hp('feed_forward_length')), nh=int(hp('attention.head_count')),
                nkv=int(hp('attention.head_count_kv')), hd=int(hp('attention.key_length')), nrot=int(hp('rope.dimension_count')),
                S=int(hp('ssm.state_size')), NK=int(hp('ssm.group_count')), NV=int(hp('ssm.time_step_rank')),
                d_conv=int(hp('ssm.conv_kernel')), d_inner=int(hp('ssm.inner_size')))


def _projection_bytes(F, max_ctx):
    """bytes the engine allocates on its card: every trunk tensor (the embedding and the head too), the f16 KV cache of
    every attention block, the Gated DeltaNet states"""
    c = _shapes(F); nl = c['nl']
    mtp = tuple(f'blk.{i}.' for i in range(nl, int(F.hp('block_count'))))
    w = sum(t.n_bytes for t in F.tensors if not t.name.startswith(mtp))
    n_attn = sum(1 for g in block_kinds(F) if not g)
    CH = 2 * c['S'] * c['NK'] + c['S'] * c['NV']
    gdn = (nl - n_attn) * (k3.gdn_block_size(c['NV'], c['S'], c['NK'], 1) + CH * 3 + CH) * 4
    return w + n_attn * int(max_ctx) * k3.kv_pos_bytes(c['nkv'] * c['hd'], False) + gdn


class Dense35Own:
    TAPE_MIB = 8   # an upper bound of one tape's buffers (S, the stage, the logits): what a further tape on the same engine adds

    def __init__(self, path, device=0, max_ctx=4096, limit_us=200_000, module=None, per_wgp='card', gu=True, rpt=16,
                 rpt16=32, vram_check=True, vram_extra_mib=0.0, max_retries=8, debug_layers=None):
        """path: a dense qwen35 GGUF whose matrices are Q4_K / Q6_K / Q8_0 / Q5_K (alpha / beta may be F32). device: the
        torch / HIP device index. max_ctx: KV cache positions. per_wgp: workgroups per WGP ('card': the qwen3moe engine's
        per-card grid; a number for the grid tests). gu: the feed-forward's gate, up and SwiGLU as one segment where the
        two matrices share a one-round-trip format (False: three segments everywhere). rpt / rpt16: rows per task of the
        one-round-trip / the general rows matvecs. debug_layers: TEST-ONLY, the first n blocks, then the output norm and
        the head; after a pass buf['x'] is block n - 1's output residual."""
        self.F = F = ggufio.GGUFFile(path); self.path = path
        if F.arch != 'qwen35': raise ValueError(f'Dense35Own: {path} is a {F.arch!r} GGUF, not qwen35 (dense)')
        if any(n.endswith('ffn_gate_inp.weight') for n in F.T): raise ValueError(f'Dense35Own: {path} has a router tensor: not the dense line')
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
        if isinstance(max_ctx, bool) or not isinstance(max_ctx, (int, np.integer)) or not 1 <= max_ctx <= n_ctx:
            raise ValueError(f'max_ctx {max_ctx!r}: 1 .. {n_ctx}')
        self.dev, self.max_ctx, self.limit_us = int(device), int(max_ctx), limit_us
        self.module = module or MODULE
        self.per_wgp, self.gu, self.rpt, self.rpt16 = per_wgp, bool(gu), int(rpt), int(rpt16)
        self.nl_build = self.nl if debug_layers is None else int(debug_layers)
        assert 1 <= self.nl_build <= self.nl, debug_layers
        self.emb = F.T['token_embd.weight']
        if self.emb.type != 'Q4_K': raise ValueError(f'Dense35Own: token_embd is {self.emb.type} (Q4_K only)')
        # VRAM, checked against the card's cap before a byte is read (vram_extra_mib: what the caller adds right after)
        self.vram_mib = _projection_bytes(F, self.max_ctx) / 2 ** 20 + 256
        if vram_check:
            capacity.check([self.dev], [self.vram_mib + float(vram_extra_mib)],
                           f'Dense35Own {os.path.basename(path)} (max_ctx {self.max_ctx}: {self.vram_mib:.0f} MiB + {float(vram_extra_mib):.0f} MiB after it)')
        torch.cuda.set_device(self.dev)
        if self.per_wgp == 'card': self.per_wgp = card_per_wgp()
        self.pos_host = 0
        # the server's engine contract (Qwen35One's / Q3MoEOne's names): one card, T = 1, dense attention at every depth
        self.ntok, self.max_retries, self.bails = 1, int(max_retries), []
        self.fresh, self.poisoned, self.seq, self.token = True, None, 0, 0
        self.devs = [self.dev]
        self.T = F.T
        self.qsa, self.qsa_limit, self.idx_cache, self.dual, self.local_only = False, None, False, False, True
        self.qsa_from = None
        self.feature_layers, self.feature_card, self.features, self.ple_layers = [], 0, None, None
        self.gdn_keys = self.moe_blocks = 0; self.vocab_share = 1.0
        self._during = None
        self.model_tag = 'qwen35'
        with torch.cuda.device(self.dev): self._load(); self._build_tape()

    @staticmethod
    def projection_mib(path, max_ctx):
        """MiB the engine allocates on its card at max_ctx, from the file's header alone (nothing on a card): __init__'s
        own check counts the same terms"""
        F = ggufio.GGUFFile(path)
        try: return _projection_bytes(F, max_ctx) / 2 ** 20 + 256
        finally: F.close()

    @staticmethod
    def recurrent_mib(path):
        """MiB of every Gated DeltaNet block's state (the recurrent state with its saved-input record, the conv state and
        its previous inputs): what a prompt-cache checkpoint holds beside the KV rows"""
        F = ggufio.GGUFFile(path)
        try:
            c = _shapes(F); n_gdn = sum(1 for g in block_kinds(F) if g)
            CH = 2 * c['S'] * c['NK'] + c['S'] * c['NV']
            return n_gdn * (k3.gdn_block_size(c['NV'], c['S'], c['NK'], 1) + CH * 3 + CH) * 4 / 2 ** 20
        finally: F.close()

    @staticmethod
    def kv_pos_mib(path):
        """MiB of KV cache per position (every attention block)"""
        F = ggufio.GGUFFile(path)
        try: return (_projection_bytes(F, 1) - _projection_bytes(F, 0)) / 2 ** 20
        finally: F.close()

    # -- weights -----------------------------------------------------------------------------------------------------------
    def _w(self, name, rows=None, cols=None):
        """a matrix on the card in the layout its tile reads: dict(name, fmt, rows, K, w). Q4_K / Q8_0 / Q5_K: the GGUF's
        blocks; Q6_K: its planes; F32: a float32 [rows, K]. The bytes come through ordinary memory (ggufio.Tensor.pread),
        never the file's mapping"""
        t = self.F.T[name]; K = int(t.shape[0]); R, _ = t.rows()
        if (rows is not None and R != rows) or (cols is not None and K != cols):
            raise ValueError(f'Dense35Own: {name} is [{R}, {K}], the tape needs [{rows}, {cols}]')
        b = t.pread(0, t.n_bytes)
        if t.type == 'F32':
            return dict(name=name, fmt='F32', rows=R, K=K, w=torch.from_numpy(b.view(np.float32).reshape(R, K)).to('cuda'))
        if t.type == 'Q6_K': b = q6k_planes(b, R, K)
        elif t.type not in ('Q4_K', 'Q8_0', 'Q5_K'):
            raise ValueError(f'Dense35Own: {name} is {t.type} (Q4_K / Q6_K / Q8_0 / Q5_K, or F32 for alpha / beta)')
        return dict(name=name, fmt=t.type, rows=R, K=K, w=torch.from_numpy(np.ascontiguousarray(b)).to('cuda'))

    def _f32(self, name, n):
        t = self.F.T[name]
        if t.type != 'F32' or t.n_values != n: raise ValueError(f'Dense35Own: {name} is {t.type} {list(t.shape)}, the tape needs F32 [{n}]')
        return torch.from_numpy(t.pread(0, t.n_bytes).view(np.float32)).to('cuda')

    def _load(self):
        D, hd, nh, nkv, S, NK, NV, ffn = self.D, self.hd, self.nh, self.nkv, self.S, self.NK, self.NV, self.ffn
        CH = 2 * S * NK + S * NV
        Z = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt, device='cuda')
        self.L, self.state = [], {}
        for il in range(self.nl_build):
            p = f'blk.{il}.'
            d = dict(attn_norm=self._f32(p + 'attn_norm.weight', D), post_norm=self._f32(p + 'post_attention_norm.weight', D),
                     fg=self._w(p + 'ffn_gate.weight', ffn, D), fu=self._w(p + 'ffn_up.weight', ffn, D),
                     fd=self._w(p + 'ffn_down.weight', D, ffn))
            if self.gdn[il]:
                d.update(kind='gdn', qkv=self._w(p + 'attn_qkv.weight', CH, D), z=self._w(p + 'attn_gate.weight', NV * S, D),
                         al=self._w(p + 'ssm_alpha.weight', NV, D), be=self._w(p + 'ssm_beta.weight', NV, D),
                         so=self._w(p + 'ssm_out.weight', D, NV * S), conv_w=self._f32(p + 'ssm_conv1d.weight', CH * 4).view(CH, 4),
                         dt=self._f32(p + 'ssm_dt.bias', NV), a=self._f32(p + 'ssm_a', NV), nw=self._f32(p + 'ssm_norm.weight', S),
                         conv_st=Z(CH * 3).view(CH, 3), convx=Z(CH), gdn=Z(k3.gdn_block_size(NV, S, NK, 1)))
                self.state.update({(0, il, 'conv'): d['conv_st'], (0, il, 'convx'): d['convx'], (0, il, 'gdn'): d['gdn']})
            else:
                d.update(kind='attn', q=self._w(p + 'attn_q.weight', nh * 2 * hd, D), k=self._w(p + 'attn_k.weight', nkv * hd, D),
                         v=self._w(p + 'attn_v.weight', nkv * hd, D), o=self._w(p + 'attn_output.weight', D, nh * hd),
                         qn=self._f32(p + 'attn_q_norm.weight', hd), kn=self._f32(p + 'attn_k_norm.weight', hd),
                         cache=Z(self.max_ctx * 2 * nkv * hd, torch.float16))
                self.state[(0, il, 'kv')] = d['cache']
            self.L.append(d)
        self.out_norm = self._f32('output_norm.weight', D)
        self.head = self._w('output.weight', self.vocab, D)
        self.emb_dev = torch.from_numpy(self.emb.pread(0, self.emb.n_bytes)).to('cuda')

    # -- the tape ----------------------------------------------------------------------------------------------------------
    def _build_tape(self):
        D, hd, nh, nkv, S, NK, NV, ffn = self.D, self.hd, self.nh, self.nkv, self.S, self.NK, self.NV, self.ffn
        CH, KD, VD, QH = 2 * S * NK + S * NV, NK * S, NV * S, nh * hd
        B = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt, device='cuda')
        f16 = torch.float16
        s = self.buf = dict(x=B(D), h=B(D), h16=B(D, f16), blk=B(D), ffp=B(D), g=B(ffn), u=B(ffn), a16=B(ffn, f16),
                          al=B(NV), be=B(NV), gates=B(2 * NV), qkv=B(CH), conv=B(CH), qk=B(2 * KD), z=B(VD), o=B(VD),
                          fin16=B(VD, f16), qf=B(2 * QH), kr=B(nkv * hd), vr=B(nkv * hd), qo=B(QH), ko=B(nkv * hd),
                          att=B(QH), gat16=B(QH, f16), logits=B(self.vocab))
        assert sum(v.numel() * v.element_size() for v in s.values()) <= self.TAPE_MIB * 2 ** 20, 'TAPE_MIB'
        # the per-pass inputs, one block of int32 (one copy from the pinned stage per token): [0 .. 15] the engine's pass
        # parameters ([1] = the previous pass's tokens to commit into the recurrent states), [16] the token, [17] its position
        self.hstage = torch.zeros(32, dtype=torch.int32, pin_memory=True)
        self.dstage = B(32, torch.int32); self.tok, self.pos = self.dstage[16:17], self.dstage[17:18]
        E = self.E = k3.Engine(module=self.module, grid=k3.resident_grid(self.module, self.per_wgp), T=1, limit_us=self.limit_us)
        E.params = self.dstage[0:16]
        self.seg_names, self.layer_seg = {}, []

        def named(si, name):
            self.seg_names[si] = name; return si

        def mv(W, x16, out, dep, x32=None, up=None):
            """W's matvec on the fp16 activations x16 (an F32 matrix: on the fp32 x32)"""
            if W['fmt'] == 'F32':
                return named(E.f32_mv(W['w'], x32, out, rows_per_task=4, dep=dep), W['name'])
            if W['fmt'] in k3.OWN_1RT_FMT:
                kw = {} if up is None else dict(up=up['w'])
                return named(E.own_matvec_1rt(W['w'], W['fmt'], W['rows'], W['K'], x16, out, rows_per_task=self.rpt, dep=dep, **kw),
                             W['name'])
            assert up is None
            return named(E.own_matvec16(W['w'], W['fmt'], W['rows'], W['K'], x16, out, rows_per_task=self.rpt16, dep=dep), W['name'])

        prev = E.own_emb(self.emb_dev, 'Q4_K', self.vocab, D, self.tok, s['x'])
        for il, d in enumerate(self.L):
            self.layer_seg.append((il, len(E.segs)))
            # the block's norm, after the previous block's feed-forward is added to the residual (block 0: no add)
            n1 = E.own_add_rms16(s['x'], d['attn_norm'], s['h'], s['h16'], self.eps, d=s['ffp'] if il else None,
                                 slots=1 if il else 0, dep=prev)
            if d['kind'] == 'gdn':
                a_ = mv(d['al'], s['h16'], s['al'], n1, x32=s['h']); b_ = mv(d['be'], s['h16'], s['be'], n1, x32=s['h'])
                gt = E.also(E.gdn_gates(s['al'], s['be'], d['dt'], d['a'], s['gates'], dep=a_), b_)
                q_ = mv(d['qkv'], s['h16'], s['qkv'], n1)
                c_ = E.gdn_conv(s['qkv'], d['conv_w'], d['conv_st'], s['conv'], d['convx'], dep=q_)
                l_ = E.gdn_l2(s['conv'], s['qk'], 2 * NK, S, self.eps, x_stride=CH, dep=c_)
                z_ = mv(d['z'], s['h16'], s['z'], n1)
                st = E.also(E.gdn_step(s['qk'], s['conv'][2 * KD:], s['gates'], d['gdn'], s['o'], NV, S, NK, CH, dep=l_), gt)
                sv = E.gdn_save(s['qk'], s['conv'][2 * KD:], s['gates'], d['gdn'], NV, S, NK, CH, dep=st)
                nm = E.also(E.gdn_norm16(s['o'], d['nw'], s['z'], s['fin16'], NV, S, self.eps, dep=st), z_)
                # (the output matvec also waits for the save: the next block rewrites the buffers the save reads)
                o_ = E.also(mv(d['so'], s['fin16'], s['blk'], nm), sv)
            else:
                k_ = mv(d['k'], s['h16'], s['kr'], n1); v_ = mv(d['v'], s['h16'], s['vr'], n1)
                kd = E.qk_nr(s['kr'], hd, nkv, hd, d['kn'], self.pos, s['ko'], self.nrot, self.theta_scale, self.eps, dep=k_)
                ap = E.also(E.kv_append(s['ko'], s['vr'], d['cache'], self.pos, dep=kd), v_)
                q_ = mv(d['q'], s['h16'], s['qf'], n1)
                qd = E.qk_nr(s['qf'], 2 * hd, nh, hd, d['qn'], self.pos, s['qo'], self.nrot, self.theta_scale, self.eps, dep=q_)
                a_ = E.also(E.attn_dec(s['qo'], d['cache'], self.pos, s['att'], nh, nkv, hd, 1.0 / np.sqrt(hd), dep=ap,
                                       batched=True), qd)
                g_ = E.gate_mul16(s['att'], s['qf'], 2 * hd, hd, s['gat16'], hd, dep=a_)
                o_ = mv(d['o'], s['gat16'], s['blk'], g_)
            n2 = E.own_add_rms16(s['x'], d['post_norm'], s['h'], s['h16'], self.eps, d=s['blk'], slots=1, dep=o_)
            fg, fu = d['fg'], d['fu']
            if self.gu and fg['fmt'] == fu['fmt'] and fg['fmt'] in k3.OWN_1RT_FMT and self.rpt % 16 == 0 and self.rpt <= 64:
                sw = mv(fg, s['h16'], s['a16'], n2, up=fu)
            else:   # (the two matrices in different formats, or one without a one-round-trip tile: three segments)
                g_ = mv(fg, s['h16'], s['g'], n2); u_ = mv(fu, s['h16'], s['u'], n2)
                sw = E.also(E.own_swiglu16(s['g'], s['u'], s['a16'], dep=u_), g_)
            prev = mv(d['fd'], s['a16'], s['ffp'], sw)
        self.layer_seg.append(('head', len(E.segs)))
        nf = E.own_add_rms16(s['x'], self.out_norm, s['h'], s['h16'], self.eps, d=s['ffp'], slots=1, dep=prev)
        mv(self.head, s['h16'], s['logits'], nf)
        E.build(); self._server_buffers()

    def _server_buffers(self):
        """the server contract's views and staging: logits [1, vocab] and its two column halves (the device sampler's
        shards, on this card), the argmax read-back, the status words, the done event"""
        self.logits = self.buf['logits'].view(1, self.vocab)
        self.vh = self.vocab // 2
        self.logits_half = [self.logits[:, :self.vh], self.logits[:, self.vh:]]
        self.shard_devs = [self.dev, self.dev]
        self.d_max = (torch.zeros(1, device='cuda'), torch.zeros(1, dtype=torch.int64, device='cuda'))
        self.h_max = (torch.zeros(1, pin_memory=True), torch.zeros(1, dtype=torch.int64, pin_memory=True))
        self.h_st = torch.zeros(k3.STATUS_WORDS, dtype=torch.int32, pin_memory=True)
        self.ev_done = torch.cuda.Event()
        self.engines = [self.E]

    # -- decode ------------------------------------------------------------------------------------------------------------
    @property
    def staged(self):
        """the position the staging block holds for the next launch"""
        return int(self.hstage[17])

    def reset(self):
        """a new sequence: zero every recurrent state (positions restart at 0; every KV row is written before any pass
        reads it); clears poisoned"""
        with torch.cuda.device(self.dev):
            for (_, _, kind), t in self.state.items():
                if kind in RECURRENT: t.zero_()
        self.pos_host = 0
        self.seq += 1; self.fresh = True; self.poisoned = None

    def prefetch(self, tokens, positions):
        """nothing to read ahead: the speculative loop's hook"""

    def step(self, tok):
        """one token at the next position (the previous token committed into the recurrent states): its logits (fp32
        [vocab], on the device; valid until the next step)"""
        i0 = self.pos_host
        self._run_tok(int(tok), i0, 0 if self.fresh else 1, False)
        return self.buf['logits']

    def _check(self, tokens, i0, commit, what):
        if self.poisoned: raise k3.ContextBoundsError(f'{what}: refused until reset(): {self.poisoned}')
        k3.check_span(i0, 1, self.max_ctx, commit=commit, ntok=1, what=what)
        if tokens is not None and len(tokens) < i0 + 1:
            raise k3.ContextBoundsError(f'{what}: position {i0} but {len(tokens)} tokens')

    def _stage(self, tok, i0):
        if not 0 <= tok < self.vocab: raise ValueError(f'Dense35Own: token {tok} outside the vocabulary ({self.vocab})')
        self.hstage[16] = tok; self.hstage[17] = int(i0)

    def step_token(self, tokens, i):
        """position i; returns only the greedy next token (argmax on the card, the first index on ties)"""
        return self._run(tokens, i, 0 if self.fresh else 1, True)[0]

    def verify(self, tokens, i0, commit):
        """position i0 in one pass, after committing the previous pass's first `commit` tokens (0 or 1) into the recurrent
        states; returns logits [1, vocab] (a fresh tensor per pass)"""
        self._run(tokens, i0, commit, False)
        return self.logits.clone()

    def verify_tokens(self, tokens, i0, commit):
        return self._run(tokens, i0, commit, True)

    def _run(self, tokens, i0, commit, tok):
        self._check(tokens, i0, commit, 'Dense35Own pass')
        return self._run_tok(int(tokens[i0]), i0, commit, tok)

    def _run_tok(self, token, i0, commit, tok):
        self._check(None, i0, commit, 'Dense35Own pass')
        self._stage(token, i0); self.fresh = False; self._tok = tok
        for attempt in range(self.max_retries + 1):
            bad = self._pass(i0, commit if attempt == 0 else 0)   # (a bailed pass committed already: its rerun commits nothing)
            if not bad: break
            self.bails.append((i0, attempt, bad))
            print(f'k3 qwen35: a wait gave up at position {i0} (attempt {attempt + 1}), running the pass again: ' + '; '.join(bad),
                  file=sys.stderr, flush=True)
        else:
            raise RuntimeError(f'k3 qwen35: a wait gave up at position {i0} in {self.max_retries + 1} attempts: ' + '; '.join(bad))
        self.pos_host = i0 + 1
        if not tok: return None
        return [int(self.h_max[1][0])]

    def _pass(self, i0, commit):
        self._pass_launch(i0, commit)
        return self._pass_wait(i0)

    def _pass_launch(self, i0, commit):
        """the launch from the staged block (and the logits' argmax read back when self._tok)"""
        self._check(None, i0, commit, 'Dense35Own._pass')
        if int(self.hstage[17]) != i0: raise k3.ContextBoundsError(f'Dense35Own._pass: staged position {int(self.hstage[17])} != {i0}')
        self.token += 1
        self.hstage[1] = int(commit)
        with torch.cuda.device(self.dev):
            self.dstage.copy_(self.hstage, non_blocking=True)   # (the pinned block is rewritten only after the wait below)
            self.E.run(check=False)
            if self._tok:
                torch.max(self.logits, dim=1, out=self.d_max)
                self.h_max[0].copy_(self.d_max[0], non_blocking=True); self.h_max[1].copy_(self.d_max[1], non_blocking=True)
            self.h_st.copy_(self.E.sync[2 * k3.MAXSEG:2 * k3.MAXSEG + k3.STATUS_WORDS], non_blocking=True)
            self.ev_done.record()

    def _pass_wait(self, i0):
        """wait for the launch, read its status; the bounds guard poisons and raises, a wait that gave up is returned as
        its report"""
        self.ev_done.synchronize()
        st = self.h_st.tolist()
        b = k3.bounds_status(st)
        if b:
            with torch.cuda.device(self.dev): torch.cuda.synchronize(); self.E.sync.zero_()
            self.poisoned = f'the device bound guard caught an out-of-range value at pass position {i0}: {b}'
            raise k3.ContextBoundsError(f'k3 qwen35: {self.poisoned}')
        if st[1]:
            with torch.cuda.device(self.dev):
                torch.cuda.synchronize()
                names = {v: k for k, v in vars(k3).items() if k.startswith('OP_')}
                rep = (f'segment {st[2]} = {names.get(self.E.segs[st[2]].op)} {self.seg_names.get(st[2], "")} (seen {st[3]}); '
                       f'{self.E.stall_report()}')
                self.E.sync.zero_()
            return [rep]
        return []

    def physics_bytes(self):
        """per card (one): the weight bytes one decode pass reads (every trunk matrix and norm once; the embedding is one
        row), for the bandwidth bound"""
        mtp = tuple(f'blk.{i}.' for i in range(self.nl, int(self.F.hp('block_count'))))
        tot = sum(t.n_bytes for t in self.F.tensors if not t.name.startswith(mtp) and t.name != 'token_embd.weight')
        return (float(tot + self.D * 144 // 256),)

    def check_replicas(self):
        """one card: no replicas: always (True, True)"""
        return True, True
