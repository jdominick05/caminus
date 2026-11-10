"""qwen3moe (Qwen3-30B-A3B) decode on one card, on Caminus's own code from the GGUF file to the logits: our reader
(caminus.ggufio), our weight layout (engine/own_layout.py: Q6_K planes), our matvec tiles (OP_OWN_MV one-round-trip,
the 8 experts of a layer grouped by device ids), the embedding row decoded on the device (OP_OWN_EMB, bitwise
caminus.own_dequant), and the K3 engine's own norm / q-k norm + RoPE / KV append / decode attention / router ops; module k3_gguf_q3m_own. One token per
pass (T = 1). The test oracle is caminus.ref_q3moe (float64, the same code path's math).

Per layer (fp32 residual x; h16 = fp16 activations for the matvecs):
  rms(x) * attn_norm -> h -> h16;  q, k, v = W h16;  qk_nr: per-head rms * q_norm / k_norm + NEOX RoPE (n_rot 128);
  KV append (f16 cache, [K | V] per position);  attention over 0..pos (8 query heads per kv head) -> fp16;  x += Wo att
  rms(x) * ffn_norm -> h -> h16;  router: F32 matvec on h, softmax, top 8, weights renormalized;
  g, u = W_gate / W_up [ids] h16 (grouped);  a16 = fp16(silu(g) * u);  d = W_down [ids] a16 (grouped, per-slot input);
  x += sum_s w_s d_s
Then rms(x) * output_norm -> fp16 -> the head (Q6_K planes) -> logits (fp32 [vocab])."""
import ctypes
import os
import sys

import numpy as np
import torch

from .. import ggufio
from ..device import capacity
from . import k3
from .own_layout import q6k_planes

MODULE = 'k3_gguf_q3m_own'         # every tape form (the unfused ones, the five-segment attention: tests and A/Bs)
MODULE_LEAN = 'k3_q3m_lean'   # the default tape's ops only: the module the engine decodes with (k3.Q3M_LEAN_OPS)
ATS_SPLITS, ATS_LMIN = 64, 64   # OP_OWN_ATTS: splits per kv head, fewest positions per split. XT, a chain of 48 attention
                                # segments (us per layer at positions 8 / 1K / 4K / 8K): one task per query head 10.4 / 86 /
                                # 322 / 631; S 64 Lmin 64: 10.3 / 14.0 / 24.9 / 35.0 (Lmin 128: 10.8 / 16.1 / 30.5 / 35.2).
                                # Measured and dropped: copies of the tape with fewer splits and smaller head groups for
                                # shallow positions (slower at every depth tried: XT 4.79 -> 4.99 ms at positions 8 .. 48)
PER_WGP = 6   # workgroups per WGP (12 waves per SIMD of the 16 that fit): interleaved at the same positions, 5 rounds,
              # the lean module, positions 8 .. 72 / 104: XT 8: 4.776, 7: 4.591, 6: 4.513, 5: 4.547 ms per token; XTX 8:
              # 4.143, 7: 4.172, 6: 4.093, 5: 4.056. A segment's hop costs each active wave its share of the SIMD


class Q3MoEOne:
    def __init__(self, path, device=0, max_ctx=4096, limit_us=200_000, module=None, fused=True, rpt=16, per_wgp=PER_WGP, gu=True, attn_batched=True, attn_one=True,
                 attn_split=True, splits=ATS_SPLITS, lmin=ATS_LMIN, vram_check=True, vram_extra_mib=0.0, max_retries=8):
        self.F = F = ggufio.GGUFFile(path); self.path = path
        if F.arch != 'qwen3moe': raise ValueError(f'Q3MoEOne: {path} is a {F.arch!r} GGUF, not qwen3moe')
        hp = F.hp
        self.nl, self.D = int(hp('block_count')), int(hp('embedding_length'))
        self.nh, self.nkv, self.hd = int(hp('attention.head_count')), int(hp('attention.head_count_kv')), int(hp('attention.key_length'))
        self.n_exp, self.k_used, self.ffn = int(hp('expert_count')), int(hp('expert_used_count')), int(hp('expert_feed_forward_length'))
        self.eps = float(hp('attention.layer_norm_rms_epsilon'))
        self.vocab = int(F.T['output.weight'].shape[1])
        # the shapes the tape is written for (checked, never assumed)
        assert self.hd == 128 and self.nh % self.nkv == 0 and self.D % 256 == 0 and self.ffn % 256 == 0, (self.hd, self.D, self.ffn)
        assert int(hp('attention.value_length', self.hd)) == self.hd and self.k_used <= 64
        libm = ctypes.CDLL('libm.so.6'); libm.powf.restype = ctypes.c_float; libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]
        self.theta_scale = libm.powf(float(hp('rope.freq_base')), -2.0 / self.hd)
        n_ctx = int(hp('context_length'))
        if not isinstance(max_ctx, int) or not 1 <= max_ctx <= n_ctx: raise ValueError(f'max_ctx {max_ctx!r}: 1 .. {n_ctx}')
        self.dev, self.max_ctx, self.limit_us, self.module, self.fused = int(device), max_ctx, limit_us, module, fused   # (module None: by the tape form)
        self.rpt = rpt   # rows per matvec task: 'auto' = each stage's tasks fit the resident grid once, or a number
        self.per_wgp = per_wgp   # workgroups per WGP (None: the module's resident grid, 8; the grid tests set 3 / 7 / 8)
        self.gu = gu             # the fused tape: gate, up and the SwiGLU as one segment (False: three)
        self.attn_batched = attn_batched   # OP_ATTN_DEC's loads in batches (the same bits; False: one round trip each)
        self.attn_one = attn_one   # the fused tape: q / k norm + RoPE, cache row, attention, fp16 out as one segment
        self.attn_split = attn_split   # ... split over positions (OP_OWN_ATTS: speed at depth); False: one task per query head
        self.splits, self.lmin = splits, lmin
        self.emb = F.T['token_embd.weight']
        # VRAM: every tensor (same bytes in our layouts; the embedding too: its row is decoded on the device), the KV
        # cache, the buffers (+ slack)
        if self.emb.type != 'Q4_K': raise ValueError(f'Q3MoEOne: token_embd is {self.emb.type} (Q4_K only)')
        wbytes = sum(t.n_bytes for t in F.tensors)
        self.cache_rows = max_ctx + k3.Engine.ATS_PAD   # (OP_OWN_ATTS's chunks read up to ATS_PAD - 1 rows past the position)
        kv = self.nl * self.cache_rows * k3.kv_pos_bytes(self.nkv * self.hd, False)
        self.vram_mib = (wbytes + kv + 4 * k3.Engine.atts_scratch_floats(self.nkv, splits)) / 2 ** 20 + 256
        if vram_check:   # (vram_extra_mib: what the caller adds right after: a prefill's buffers, a drafter)
            capacity.check([self.dev], [self.vram_mib + float(vram_extra_mib)],
                           f'Q3MoEOne {os.path.basename(path)} (max_ctx {max_ctx}: {self.vram_mib:.0f} MiB + {float(vram_extra_mib):.0f} MiB after it)')
        torch.cuda.set_device(self.dev)
        self.pos_host = 0
        # the server's engine contract (Qwen35One's / FlashNextTP's names; caminus.serve.k3_session and caminus.cache.prompt
        # read them): one card, T = 1, no recurrent state (a checkpoint is KV rows only; the lazy commit commits nothing),
        # dense attention at every depth, no drafter features yet
        self.ntok, self.max_retries, self.bails = 1, int(max_retries), []
        self.fresh, self.poisoned, self.seq, self.token = True, None, 0, 0
        self.devs = [self.dev]
        self.T = F.T                                          # (the GGUF's tensors by name: K3Session reads output.weight's shape)
        self.qsa, self.qsa_limit, self.idx_cache, self.dual, self.local_only = False, None, False, False, True
        self.qsa_from = None
        self.feature_layers, self.feature_card, self.features, self.ple_layers = [], 0, None, None
        self.gdn_keys = self.moe_blocks = 0; self.vocab_share = 1.0
        self._during = None
        self.model_tag = 'qwen3moe'
        with torch.cuda.device(self.dev): self._load(); self._build_tape()

    @staticmethod
    def projection_mib(path, max_ctx, splits=ATS_SPLITS):
        """MiB the engine allocates on its card at max_ctx, from the file's header alone (nothing on a card): __init__'s own
        check counts the same terms"""
        F = ggufio.GGUFFile(path)
        try:
            nl, nkv, hd = int(F.hp('block_count')), int(F.hp('attention.head_count_kv')), int(F.hp('attention.key_length'))
            w = sum(t.n_bytes for t in F.tensors)
        finally:
            F.close()
        kv = nl * (int(max_ctx) + k3.Engine.ATS_PAD) * 4 * nkv * hd
        return (w + kv + 4 * k3.Engine.atts_scratch_floats(nkv, splits)) / 2 ** 20 + 256

    @staticmethod
    def kv_pos_mib(path):
        """MiB of KV cache per position (every layer)"""
        F = ggufio.GGUFFile(path)
        try: return int(F.hp('block_count')) * 4 * int(F.hp('attention.head_count_kv')) * int(F.hp('attention.key_length')) / 2 ** 20
        finally: F.close()

    # -- weights -----------------------------------------------------------------------------------------------------------
    @property
    def module(self):
        """the code object: the one set, else the lean decode module for the default tape form and the full one for the others"""
        if self._module is not None: return self._module
        return MODULE_LEAN if self.fused and self.gu and self.attn_one and self.attn_split else MODULE

    @module.setter
    def module(self, m): self._module = m

    def _w(self, name):
        """(format, rows per matrix, K, device bytes in our layout) of a Q4_K / Q6_K matrix (experts: the whole tensor)"""
        t = self.F.T[name]; K = int(t.shape[0]); R, _ = t.rows()
        b = np.asarray(t.data())
        if t.type == 'Q6_K': b = q6k_planes(b, R, K)
        elif t.type != 'Q4_K': raise ValueError(f'Q3MoEOne: {name} is {t.type} (Q4_K / Q6_K only)')
        return t.type, int(t.shape[1]), K, torch.from_numpy(np.ascontiguousarray(b)).to('cuda')

    def _f32(self, name, shape=None):
        t = self.F.T[name]; assert t.type == 'F32', (name, t.type)
        a = np.asarray(t.data()).view(np.float32).copy()
        return torch.from_numpy(a.reshape(shape) if shape else a).to('cuda')

    def _load(self):
        self.L = []
        for il in range(self.nl):
            p = f'blk.{il}.'
            self.L.append(dict({r: self._w(p + r + '.weight') for r in ('attn_q', 'attn_k', 'attn_v', 'attn_output',
                                                                      'ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps')},
                               attn_norm=self._f32(p + 'attn_norm.weight'), ffn_norm=self._f32(p + 'ffn_norm.weight'),
                               q_norm=self._f32(p + 'attn_q_norm.weight'), k_norm=self._f32(p + 'attn_k_norm.weight'),
                               qk_par=None,   # (OP_OWN_ATTN's block: q_norm | k_norm | theta_scale, eps; built below)
                               router=self._f32(p + 'ffn_gate_inp.weight', (self.n_exp, self.D)),
                               cache=torch.zeros(self.cache_rows * 2 * self.nkv * self.hd, dtype=torch.float16, device='cuda')))
        for d in self.L:
            d['qk_par'] = torch.cat([d['q_norm'], d['k_norm'], torch.tensor([self.theta_scale, self.eps], dtype=torch.float32,
                                                                              device='cuda')]).contiguous()
        self.out_norm = self._f32('output_norm.weight')
        self.head = self._w('output.weight')
        self.emb_dev = torch.from_numpy(np.ascontiguousarray(np.asarray(self.emb.data()))).to('cuda')

    # -- the tape ----------------------------------------------------------------------------------------------------------
    def _build_tape(self):
        D, hd, nh, nkv, KU, FF = self.D, self.hd, self.nh, self.nkv, self.k_used, self.ffn
        B = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt, device='cuda')
        qkv = B(nh * hd + 2 * nkv * hd)   # q | k | v: the matvecs' outputs, one buffer (OP_OWN_ATTS reads it whole)
        s = self.S = dict(x=B(D), h=B(D), h16=B(D, torch.float16), q=qkv[:nh * hd], k=B(nkv * hd), v=B(nkv * hd),
                          kv=qkv[nh * hd:], qkv=qkv, ats=B(k3.Engine.atts_scratch_floats(nkv, self.splits)),
                          qo=B(nh * hd), ko=B(nkv * hd), att=B(nh * hd), att16=B(nh * hd, torch.float16), o=B(D),
                          logit=B(self.n_exp), ids=B(KU, torch.int32), wts=B(KU), g=B(KU * FF), u=B(KU * FF),
                          a16=B(KU * FF, torch.float16), d=B(KU * D), logits=B(self.vocab))
        # the per-pass inputs, one block: [0] the token, [1] its position (one copy from the pinned stage per token)
        self.hstage = torch.zeros(2, dtype=torch.int32, pin_memory=True)
        self.dstage = B(2, torch.int32); self.tok, self.pos = self.dstage[0:1], self.dstage[1:2]
        E = self.E = k3.Engine(module=self.module, grid=k3.resident_grid(self.module, self.per_wgp), T=1, limit_us=self.limit_us)
        # (a touch of each workgroup's first task's weights before its dependency wait measured no gain: XT, interleaved,
        #  48 tokens: whole task 8.873, 16 KB 8.820, off 8.609 ms/token)
        # rows per task. Every workgroup claims every segment once, and a workgroup's second task of a segment costs a
        # publish and a claim (~1 us) and starts only when its first is done: 'auto' sizes a stage's tasks (the matvecs
        # that run together share the grid) so they fit the resident grid once, with at least one row pair per wave.
        G = k3.resident_grid(self.module, self.per_wgp)
        def rpt_of(rows):
            if self.rpt != 'auto': return int(self.rpt)
            r = -(-rows // G); return max(16, r + (r & 1))
        self.RPT = dict(qkv=rpt_of((nh + 2 * nkv) * hd), o=rpt_of(D), gu=rpt_of(2 * KU * FF), down=rpt_of(KU * D),
                        head=64 if self.rpt != 'auto' else rpt_of(self.vocab))
        mv = lambda w, x16, out, dep, rpt='qkv', **kw: E.own_matvec_1rt(w[3], w[0], w[1], w[2], x16, out, dep=dep,
                                                                       rows_per_task=self.RPT[rpt], **kw)
        prev = E.own_emb(self.emb_dev, 'Q4_K', int(self.emb.shape[1]), D, self.tok, s['x'])
        if self.fused:
            return self._tape_fused(E, s, mv, prev)
        for d in self.L:
            n1 = E.rms_mul(s['x'].view(1, D), d['attn_norm'].view(1, D), s['h'].view(1, D), self.eps, dep=prev)
            c1 = E.own_f16(s['h'], s['h16'], dep=n1)
            q_ = mv(d['attn_q'], s['h16'], s['q'], c1)
            k_ = mv(d['attn_k'], s['h16'], s['k'], c1)
            v_ = mv(d['attn_v'], s['h16'], s['v'], c1)
            kd = E.qk_nr(s['k'], hd, nkv, hd, d['k_norm'], self.pos, s['ko'], hd, self.theta_scale, self.eps, dep=k_)
            ap = E.also(E.kv_append(s['ko'], s['v'], d['cache'], self.pos, dep=kd), v_)
            qd = E.qk_nr(s['q'], hd, nh, hd, d['q_norm'], self.pos, s['qo'], hd, self.theta_scale, self.eps, dep=q_)
            a_ = E.also(E.attn_dec(s['qo'], d['cache'], self.pos, s['att'], nh, nkv, hd, 1.0 / np.sqrt(hd), dep=ap,
                                   batched=self.attn_batched), qd)
            c2 = E.own_f16(s['att'], s['att16'], dep=a_)
            o_ = mv(d['attn_output'], s['att16'], s['o'], c2, rpt='o')
            r1 = E.own_moe_sum(s['o'], s['x'], dep=o_)
            n2 = E.rms_mul(s['x'].view(1, D), d['ffn_norm'].view(1, D), s['h'].view(1, D), self.eps, dep=r1)
            c3 = E.own_f16(s['h'], s['h16'], dep=n2)
            rl = E.f32_mv(d['router'], s['h'], s['logit'], dep=n2)
            ro = E.router(s['logit'], KU, s['ids'], s['wts'], dep=rl)
            g_ = E.also(mv(d['ffn_gate_exps'], s['h16'], s['g'], ro, rpt='gu', ids=s['ids'], total_rows=self.n_exp * FF), c3)
            u_ = E.also(mv(d['ffn_up_exps'], s['h16'], s['u'], ro, rpt='gu', ids=s['ids'], total_rows=self.n_exp * FF), c3)
            sw = E.also(E.own_swiglu16(s['g'], s['u'], s['a16'], dep=u_), g_)
            d_ = mv(d['ffn_down_exps'], s['a16'], s['d'], sw, rpt='down', ids=s['ids'], total_rows=self.n_exp * D, x_per_slot=True)
            prev = E.own_moe_sum(s['d'], s['x'], slots=KU, w=s['wts'], dep=d_)
        nf = E.rms_mul(s['x'].view(1, D), self.out_norm.view(1, D), s['h'].view(1, D), self.eps, dep=prev)
        cf = E.own_f16(s['h'], s['h16'], dep=nf)
        mv(self.head, s['h16'], s['logits'], cf, rpt='head')
        E.build(); self._server_buffers()

    def _tape_fused(self, E, s, mv, prev):
        """the tape with the residual add, the RMS norm and the fp16 copy as one segment (OP_OWN_ADD_RMS16; the three-stage
        form's math, equal to rounding). Per layer 12 segments with the options' defaults: gate | up | SwiGLU as one
        (gu) and the attention as one behind a no-op join of k and v (attn_one: OP_OWN_ATTN, the five segments' bits);
        17 with both off (the unfused tape: 21)"""
        D, hd, nh, nkv, KU, FF = self.D, self.hd, self.nh, self.nkv, self.k_used, self.ffn
        for il, d in enumerate(self.L):
            # the attention's norm, after the previous layer's experts are added to the residual (layer 0: no add)
            n1 = E.own_add_rms16(s['x'], d['attn_norm'], s['h'], s['h16'], self.eps, d=s['d'] if il else None,
                                 slots=KU if il else 0, w=s['wts'] if il else None, dep=prev)
            if self.attn_one:   # k | v into one buffer; the attention waits for q and (through a no-op join) k and v
                # v and k before q: workgroups claim in tape order, the first done with the previous stage first. With q
                # first the last (slowest) ones got v, and the join's publish (2.3 us after v, + a wake-up) sat on the
                # critical path; this way q's tail ends the stage and the join is long done (XT, interleaved, 3 rounds x
                # 48 tokens, the same bits: 5.561 -> 5.495 ms per token).
                M = nkv * hd
                v_ = mv(d['attn_v'], s['h16'], s['kv'][M:], n1)
                k_ = mv(d['attn_k'], s['h16'], s['kv'][:M], n1)
                jn = E.also(E.nop(1, 0, dep=k_), v_)
                q_ = mv(d['attn_q'], s['h16'], s['q'], n1)
                if self.attn_split:
                    c2 = E.also(E.own_atts(s['qkv'], s['ats'], d['qk_par'], d['cache'], self.pos, s['att16'], nh, nkv, hd,
                                           1.0 / np.sqrt(hd), splits=self.splits, lmin=self.lmin, dep=q_), jn)
                else:
                    c2 = E.also(E.own_attn(s['q'], s['kv'], d['qk_par'], d['cache'], self.pos, s['att16'], nh, nkv, hd,
                                           1.0 / np.sqrt(hd), dep=q_), jn)
            else:
                q_ = mv(d['attn_q'], s['h16'], s['q'], n1)
                k_ = mv(d['attn_k'], s['h16'], s['k'], n1)
                v_ = mv(d['attn_v'], s['h16'], s['v'], n1)
                kd = E.qk_nr(s['k'], hd, nkv, hd, d['k_norm'], self.pos, s['ko'], hd, self.theta_scale, self.eps, dep=k_)
                ap = E.also(E.kv_append(s['ko'], s['v'], d['cache'], self.pos, dep=kd), v_)
                qd = E.qk_nr(s['q'], hd, nh, hd, d['q_norm'], self.pos, s['qo'], hd, self.theta_scale, self.eps, dep=q_)
                a_ = E.also(E.attn_dec(s['qo'], d['cache'], self.pos, s['att'], nh, nkv, hd, 1.0 / np.sqrt(hd), dep=ap,
                                       batched=self.attn_batched), qd)
                c2 = E.own_f16(s['att'], s['att16'], dep=a_)
            o_ = mv(d['attn_output'], s['att16'], s['o'], c2, rpt='o')
            n2 = E.own_add_rms16(s['x'], d['ffn_norm'], s['h'], s['h16'], self.eps, d=s['o'], slots=1, dep=o_)
            rl = E.f32_mv(d['router'], s['h'], s['logit'], dep=n2)
            ro = E.router(s['logit'], KU, s['ids'], s['wts'], dep=rl)
            if self.gu:
                assert d['ffn_gate_exps'][:3] == d['ffn_up_exps'][:3], 'gate and up: one shape and format'
                sw = mv(d['ffn_gate_exps'], s['h16'], s['a16'], ro, rpt='gu', ids=s['ids'], total_rows=self.n_exp * FF,
                        up=d['ffn_up_exps'][3])
            else:
                g_ = mv(d['ffn_gate_exps'], s['h16'], s['g'], ro, rpt='gu', ids=s['ids'], total_rows=self.n_exp * FF)
                u_ = mv(d['ffn_up_exps'], s['h16'], s['u'], ro, rpt='gu', ids=s['ids'], total_rows=self.n_exp * FF)
                sw = E.also(E.own_swiglu16(s['g'], s['u'], s['a16'], dep=u_), g_)
            prev = mv(d['ffn_down_exps'], s['a16'], s['d'], sw, rpt='down', ids=s['ids'], total_rows=self.n_exp * D, x_per_slot=True)
        nf = E.own_add_rms16(s['x'], self.out_norm, s['h'], s['h16'], self.eps, d=s['d'], slots=KU, w=s['wts'], dep=prev)
        mv(self.head, s['h16'], s['logits'], nf, rpt='head')
        E.build(); self._server_buffers()

    def _server_buffers(self):
        """the server contract's views and staging (K3Session, the prompt cache): logits [1, vocab] and its two column halves
        (the device sampler's shards, on this card), the argmax read-back, the status words, the done event, the KV caches
        as positional states (the first max_ctx rows: the padding rows past them are never a position)"""
        self.logits = self.S['logits'].view(1, self.vocab)
        self.vh = self.vocab // 2
        self.logits_half = [self.logits[:, :self.vh], self.logits[:, self.vh:]]
        self.shard_devs = [self.dev, self.dev]
        self.d_max = (torch.zeros(1, device='cuda'), torch.zeros(1, dtype=torch.int64, device='cuda'))
        self.h_max = (torch.zeros(1, pin_memory=True), torch.zeros(1, dtype=torch.int64, pin_memory=True))
        self.h_st = torch.zeros(k3.STATUS_WORDS, dtype=torch.int32, pin_memory=True)
        self.ev_done = torch.cuda.Event()
        self.engines = [self.E]
        row = 2 * self.nkv * self.hd
        self.state = {(0, il, 'kv'): d['cache'][:self.max_ctx * row] for il, d in enumerate(self.L)}

    # -- decode ------------------------------------------------------------------------------------------------------------
    def reset(self):
        self.pos_host = 0

    def step(self, tok, trace=None):
        """one token at the next position: its logits (fp32 [vocab], on the device; valid until the next step); trace: an
        int64 buffer [grid * segments * 8] for the _trace module's timestamps"""
        if self.poisoned: raise k3.ContextBoundsError(f'Q3MoEOne step: refused until reset(): {self.poisoned}')
        k3.check_span(self.pos_host, 1, self.max_ctx, what='Q3MoEOne step')   # (before the staging block is written)
        if not 0 <= int(tok) < self.vocab: raise ValueError(f'Q3MoEOne: token {tok} outside the vocabulary ({self.vocab})')
        with torch.cuda.device(self.dev):
            self.hstage[0] = int(tok); self.hstage[1] = self.pos_host
            self.dstage.copy_(self.hstage, non_blocking=True)    # (the stage is rewritten only after the sync below)
            self.E.run() if trace is None else self.E.run(trace=trace)
            torch.cuda.synchronize()
        self.pos_host += 1
        return self.S['logits']

    # -- the server's engine contract (caminus.serve.k3_session, caminus.cache.prompt: Qwen35One's names, T = 1) -----------
    @property
    def staged(self):
        """the position the staging block holds for the next launch (the prompt cache and the session read it)"""
        return int(self.hstage[1])

    def reset(self):
        """a new sequence: positions restart at 0 (every KV row is written before any pass reads it); clears poisoned"""
        self.pos_host = 0
        self.seq += 1; self.fresh = True; self.poisoned = None

    def prefetch(self, tokens, positions):
        """nothing to read ahead: the speculative loop's hook"""

    def _check(self, tokens, i0, commit, what):
        if self.poisoned: raise k3.ContextBoundsError(f'{what}: refused until reset(): {self.poisoned}')
        k3.check_span(i0, 1, self.max_ctx, commit=commit, ntok=1, what=what)
        if tokens is not None and len(tokens) < i0 + 1:
            raise k3.ContextBoundsError(f'{what}: position {i0} but {len(tokens)} tokens')

    def _stage(self, tokens, i0):
        tok = int(tokens[i0])
        if not 0 <= tok < self.vocab: raise ValueError(f'Q3MoEOne: token {tok} outside the vocabulary ({self.vocab})')
        self.hstage[0] = tok; self.hstage[1] = int(i0)

    def step_token(self, tokens, i):
        """position i; returns only the greedy next token (argmax on the card, the first index on ties)"""
        return self._run(tokens, i, 0 if self.fresh else 1, True)[0]

    def verify(self, tokens, i0, commit):
        """position i0 in one pass (commit: the previous pass's tokens to commit, nothing to do here: no recurrent state);
        returns logits [1, vocab] (a fresh tensor per pass)"""
        self._run(tokens, i0, commit, False)
        return self.logits.clone()

    def verify_tokens(self, tokens, i0, commit):
        return self._run(tokens, i0, commit, True)

    def _run(self, tokens, i0, commit, tok):
        self._check(tokens, i0, commit, 'Q3MoEOne pass')
        self._stage(tokens, i0); self.fresh = False; self._tok = tok
        for attempt in range(self.max_retries + 1):
            bad = self._pass(i0, commit if attempt == 0 else 0)
            if not bad: break
            self.bails.append((i0, attempt, bad))
            print(f'k3 qwen3moe: a wait gave up at position {i0} (attempt {attempt + 1}), running the pass again: ' + '; '.join(bad),
                  file=sys.stderr, flush=True)
        else:
            raise RuntimeError(f'k3 qwen3moe: a wait gave up at position {i0} in {self.max_retries + 1} attempts: ' + '; '.join(bad))
        self.pos_host = i0 + 1
        if not tok: return None
        return [int(self.h_max[1][0])]

    def _pass(self, i0, commit):
        self._pass_launch(i0, commit)
        return self._pass_wait(i0)

    def _pass_launch(self, i0, commit):
        """the launch from the staged block (and the logits' argmax read back when self._tok); K3Session's test hook wraps it"""
        self._check(None, i0, commit, 'Q3MoEOne._pass')
        if int(self.hstage[1]) != i0: raise k3.ContextBoundsError(f'Q3MoEOne._pass: staged position {int(self.hstage[1])} != {i0}')
        self.token += 1
        with torch.cuda.device(self.dev):
            self.dstage.copy_(self.hstage, non_blocking=True)   # (the pinned block is rewritten only after the wait below)
            self.E.run(check=False)
            if self._tok:
                torch.max(self.logits, dim=1, out=self.d_max)
                self.h_max[0].copy_(self.d_max[0], non_blocking=True); self.h_max[1].copy_(self.d_max[1], non_blocking=True)
            self.h_st.copy_(self.E.sync[2 * k3.MAXSEG:2 * k3.MAXSEG + k3.STATUS_WORDS], non_blocking=True)
            self.ev_done.record()

    def _pass_wait(self, i0):
        """wait for the launch, read its status; the bounds guard poisons and raises, a wait that gave up is returned as its
        report. A bailed pass leaves OP_OWN_ATTS's merge counters wherever its tasks got: every task still ran once, so
        they are back at zero, but they are cleared here too before the retry"""
        self.ev_done.synchronize()
        st = self.h_st.tolist()
        b = k3.bounds_status(st)
        if b:
            with torch.cuda.device(self.dev): torch.cuda.synchronize(); self.E.sync.zero_(); self.S['ats'][:k3.Engine.ATS_CNT].zero_()
            self.poisoned = f'the device bound guard caught an out-of-range value at pass position {i0}: {b}'
            raise k3.ContextBoundsError(f'k3 qwen3moe: {self.poisoned}')
        if st[1]:
            with torch.cuda.device(self.dev):
                torch.cuda.synchronize()
                names = {v: k for k, v in vars(k3).items() if k.startswith('OP_')}
                rep = f'segment {st[2]} = {names.get(self.E.segs[st[2]].op)} (seen {st[3]}); {self.E.stall_report()}'
                self.E.sync.zero_(); self.S['ats'][:k3.Engine.ATS_CNT].zero_()
            return [rep]
        return []

    def launch_tokens(self, tokens, i0, commit):
        """verify()'s pass launched without waiting or reading anything back (the device sampler queues behind it); then
        finish_pass(i0, commit)"""
        self._check(tokens, i0, commit, 'Q3MoEOne pass')
        self._stage(tokens, i0); self._tok = False; self.fresh = False
        self._pass_launch(i0, commit)

    def finish_pass(self, i0, commit):
        """wait for launch_tokens' pass; a bailed pass runs again before returning. True when it was rerun (whatever was
        queued behind the first launch read a bailed pass's outputs and must run again)"""
        bad = self._pass_wait(i0)
        if not bad:
            self.pos_host = i0 + 1; return False
        for attempt in range(1, self.max_retries + 1):
            self.bails.append((i0, attempt - 1, bad))
            print(f'k3 qwen3moe: a wait gave up at position {i0} (attempt {attempt}), running the pass again: ' + '; '.join(bad),
                  file=sys.stderr, flush=True)
            bad = self._pass(i0, 0)
            if not bad:
                self.pos_host = i0 + 1; return True
        self.bails.append((i0, self.max_retries, bad))
        raise RuntimeError(f'k3 qwen3moe: a wait gave up at position {i0} in {self.max_retries + 1} attempts: ' + '; '.join(bad))

    def physics_bytes(self):
        """per card (one): the weight bytes one decode pass reads (every tensor once, the routed experts at k_used / n_exp
        of their bytes: the average token), for the server's physics floor"""
        tot = 0.0
        for t in self.F.tensors:
            tot += t.n_bytes * (self.k_used / self.n_exp if '_exps.' in t.name else 1.0) if t.name != 'token_embd.weight' else self.D * 144 / 256
        return (float(tot),)

    def check_replicas(self):
        """one card: no replicas: always (True, True)"""
        return True, True
