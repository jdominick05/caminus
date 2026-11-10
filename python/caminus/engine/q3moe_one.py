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

import numpy as np
import torch

from .. import ggufio
from ..device import capacity
from . import k3
from .own_layout import q6k_planes

MODULE = 'k3_gguf_q3m_own'


class Q3MoEOne:
    def __init__(self, path, device=0, max_ctx=4096, limit_us=200_000, module=MODULE, fused=True, rpt=16, per_wgp=None, gu=True):
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
        self.dev, self.max_ctx, self.limit_us, self.module, self.fused = int(device), max_ctx, limit_us, module, fused
        self.rpt = rpt   # rows per matvec task: 'auto' = each stage's tasks fit the resident grid once, or a number
        self.per_wgp = per_wgp   # workgroups per WGP (None: the module's resident grid; the grid tests set 3 / 6 / 7)
        self.gu = gu             # the fused tape: gate, up and the SwiGLU as one segment (False: three)
        self.emb = F.T['token_embd.weight']
        # VRAM: every tensor (same bytes in our layouts; the embedding too: its row is decoded on the device), the KV
        # cache, the buffers (+ slack)
        if self.emb.type != 'Q4_K': raise ValueError(f'Q3MoEOne: token_embd is {self.emb.type} (Q4_K only)')
        wbytes = sum(t.n_bytes for t in F.tensors)
        kv = self.nl * max_ctx * k3.kv_pos_bytes(self.nkv * self.hd, False)
        self.vram_mib = (wbytes + kv) / 2 ** 20 + 256
        capacity.check([self.dev], [self.vram_mib], f'Q3MoEOne {os.path.basename(path)} (max_ctx {max_ctx}: {self.vram_mib:.0f} MiB)')
        torch.cuda.set_device(self.dev)
        self.pos_host = 0
        with torch.cuda.device(self.dev): self._load(); self._build_tape()

    # -- weights -----------------------------------------------------------------------------------------------------------
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
                               router=self._f32(p + 'ffn_gate_inp.weight', (self.n_exp, self.D)),
                               cache=torch.zeros(self.max_ctx * 2 * self.nkv * self.hd, dtype=torch.float16, device='cuda')))
        self.out_norm = self._f32('output_norm.weight')
        self.head = self._w('output.weight')
        self.emb_dev = torch.from_numpy(np.ascontiguousarray(np.asarray(self.emb.data()))).to('cuda')

    # -- the tape ----------------------------------------------------------------------------------------------------------
    def _build_tape(self):
        D, hd, nh, nkv, KU, FF = self.D, self.hd, self.nh, self.nkv, self.k_used, self.ffn
        B = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt, device='cuda')
        s = self.S = dict(x=B(D), h=B(D), h16=B(D, torch.float16), q=B(nh * hd), k=B(nkv * hd), v=B(nkv * hd),
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
        if self.fused: return self._tape_fused(E, s, mv, prev)
        for d in self.L:
            n1 = E.rms_mul(s['x'].view(1, D), d['attn_norm'].view(1, D), s['h'].view(1, D), self.eps, dep=prev)
            c1 = E.own_f16(s['h'], s['h16'], dep=n1)
            q_ = mv(d['attn_q'], s['h16'], s['q'], c1)
            k_ = mv(d['attn_k'], s['h16'], s['k'], c1)
            v_ = mv(d['attn_v'], s['h16'], s['v'], c1)
            kd = E.qk_nr(s['k'], hd, nkv, hd, d['k_norm'], self.pos, s['ko'], hd, self.theta_scale, self.eps, dep=k_)
            ap = E.also(E.kv_append(s['ko'], s['v'], d['cache'], self.pos, dep=kd), v_)
            qd = E.qk_nr(s['q'], hd, nh, hd, d['q_norm'], self.pos, s['qo'], hd, self.theta_scale, self.eps, dep=q_)
            a_ = E.also(E.attn_dec(s['qo'], d['cache'], self.pos, s['att'], nh, nkv, hd, 1.0 / np.sqrt(hd), dep=ap), qd)
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
        E.build()

    def _tape_fused(self, E, s, mv, prev):
        """the tape with the residual add, the RMS norm and the fp16 copy as one segment (OP_OWN_ADD_RMS16; the three-stage
        form's math, equal to rounding): per layer 17 segments, 13 dependent stages (was 21 / 16)"""
        D, hd, nh, nkv, KU, FF = self.D, self.hd, self.nh, self.nkv, self.k_used, self.ffn
        for il, d in enumerate(self.L):
            # the attention's norm, after the previous layer's experts are added to the residual (layer 0: no add)
            n1 = E.own_add_rms16(s['x'], d['attn_norm'], s['h'], s['h16'], self.eps, d=s['d'] if il else None,
                                 slots=KU if il else 0, w=s['wts'] if il else None, dep=prev)
            q_ = mv(d['attn_q'], s['h16'], s['q'], n1)
            k_ = mv(d['attn_k'], s['h16'], s['k'], n1)
            v_ = mv(d['attn_v'], s['h16'], s['v'], n1)
            kd = E.qk_nr(s['k'], hd, nkv, hd, d['k_norm'], self.pos, s['ko'], hd, self.theta_scale, self.eps, dep=k_)
            ap = E.also(E.kv_append(s['ko'], s['v'], d['cache'], self.pos, dep=kd), v_)
            qd = E.qk_nr(s['q'], hd, nh, hd, d['q_norm'], self.pos, s['qo'], hd, self.theta_scale, self.eps, dep=q_)
            a_ = E.also(E.attn_dec(s['qo'], d['cache'], self.pos, s['att'], nh, nkv, hd, 1.0 / np.sqrt(hd), dep=ap), qd)
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
        E.build()

    # -- decode ------------------------------------------------------------------------------------------------------------
    def reset(self):
        self.pos_host = 0

    def step(self, tok, trace=None):
        """one token at the next position: its logits (fp32 [vocab], on the device; valid until the next step); trace: an
        int64 buffer [grid * segments * 8] for the _trace module's timestamps"""
        if self.pos_host >= self.max_ctx: raise k3.ContextBoundsError(f'Q3MoEOne: position {self.pos_host} past max_ctx {self.max_ctx}')
        if not 0 <= int(tok) < self.vocab: raise ValueError(f'Q3MoEOne: token {tok} outside the vocabulary ({self.vocab})')
        with torch.cuda.device(self.dev):
            self.hstage[0] = int(tok); self.hstage[1] = self.pos_host
            self.dstage.copy_(self.hstage, non_blocking=True)    # (the stage is rewritten only after the sync below)
            self.E.run() if trace is None else self.E.run(trace=trace)
            torch.cuda.synchronize()
        self.pos_host += 1
        return self.S['logits']
