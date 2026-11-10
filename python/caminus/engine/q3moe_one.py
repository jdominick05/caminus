"""qwen3moe (Qwen3-30B-A3B) decode on one card, on Caminus's own code from the GGUF file to the logits: our reader
(caminus.ggufio), our weight layout (engine/own_layout.py: Q6_K planes), our matvec tiles (OP_OWN_MV one-round-trip,
the 8 experts of a layer grouped by device ids), our host decoders for the embedding row (caminus.own_dequant), and the
K3 engine's own norm / q-k norm + RoPE / KV append / decode attention / router ops; module k3_gguf_q3m_own. One token per
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

from .. import ggufio, own_dequant
from ..device import capacity
from . import k3
from .own_layout import q6k_planes

MODULE = 'k3_gguf_q3m_own'


class Q3MoEOne:
    def __init__(self, path, device=0, max_ctx=4096, limit_us=200_000):
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
        self.dev, self.max_ctx, self.limit_us = int(device), max_ctx, limit_us
        self.emb = F.T['token_embd.weight']
        # VRAM: every tensor but the embedding (same bytes in our layouts), the KV cache, the buffers (+ slack)
        wbytes = sum(t.n_bytes for t in F.tensors if t.name != 'token_embd.weight')
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

    # -- the tape ----------------------------------------------------------------------------------------------------------
    def _build_tape(self):
        D, hd, nh, nkv, KU, FF = self.D, self.hd, self.nh, self.nkv, self.k_used, self.ffn
        B = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt, device='cuda')
        s = self.S = dict(x=B(D), h=B(D), h16=B(D, torch.float16), q=B(nh * hd), k=B(nkv * hd), v=B(nkv * hd),
                          qo=B(nh * hd), ko=B(nkv * hd), att=B(nh * hd), att16=B(nh * hd, torch.float16), o=B(D),
                          logit=B(self.n_exp), ids=B(KU, torch.int32), wts=B(KU), g=B(KU * FF), u=B(KU * FF),
                          a16=B(KU * FF, torch.float16), d=B(KU * D), logits=B(self.vocab))
        self.pos = B(1, torch.int32)
        E = self.E = k3.Engine(module=MODULE, grid=k3.resident_grid(MODULE), T=1, limit_us=self.limit_us)
        mv = lambda w, x16, out, dep, **kw: E.own_matvec_1rt(w[3], w[0], w[1], w[2], x16, out, dep=dep, **kw)
        prev = -1
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
            o_ = mv(d['attn_output'], s['att16'], s['o'], c2)
            r1 = E.own_moe_sum(s['o'], s['x'], dep=o_)
            n2 = E.rms_mul(s['x'].view(1, D), d['ffn_norm'].view(1, D), s['h'].view(1, D), self.eps, dep=r1)
            c3 = E.own_f16(s['h'], s['h16'], dep=n2)
            rl = E.f32_mv(d['router'], s['h'], s['logit'], dep=n2)
            ro = E.router(s['logit'], KU, s['ids'], s['wts'], dep=rl)
            g_ = E.also(mv(d['ffn_gate_exps'], s['h16'], s['g'], ro, ids=s['ids'], total_rows=self.n_exp * FF), c3)
            u_ = E.also(mv(d['ffn_up_exps'], s['h16'], s['u'], ro, ids=s['ids'], total_rows=self.n_exp * FF), c3)
            sw = E.also(E.own_swiglu16(s['g'], s['u'], s['a16'], dep=u_), g_)
            d_ = mv(d['ffn_down_exps'], s['a16'], s['d'], sw, ids=s['ids'], total_rows=self.n_exp * D, x_per_slot=True)
            prev = E.own_moe_sum(s['d'], s['x'], slots=KU, w=s['wts'], dep=d_)
        nf = E.rms_mul(s['x'].view(1, D), self.out_norm.view(1, D), s['h'].view(1, D), self.eps, dep=prev)
        cf = E.own_f16(s['h'], s['h16'], dep=nf)
        mv(self.head, s['h16'], s['logits'], cf, rows_per_task=64)
        E.build()

    # -- decode ------------------------------------------------------------------------------------------------------------
    def reset(self):
        self.pos_host = 0

    def step(self, tok):
        """one token at the next position: its logits (fp32 [vocab], on the device; valid until the next step)"""
        if self.pos_host >= self.max_ctx: raise k3.ContextBoundsError(f'Q3MoEOne: position {self.pos_host} past max_ctx {self.max_ctx}')
        x = own_dequant.rows(self.emb, int(tok), int(tok) + 1).reshape(-1)
        with torch.cuda.device(self.dev):
            self.S['x'].copy_(torch.from_numpy(x))
            self.pos.fill_(self.pos_host)
            self.E.run()
            torch.cuda.synchronize()
        self.pos_host += 1
        return self.S['logits']
