# Parts of this file (the DFlash drafter's forward and the draft-dflash selector walk) are ported from llama.cpp's
# src/models/dflash.cpp and common/speculative.cpp (https://github.com/ggml-org/llama.cpp, through the ignite-llama fork),
# by way of caminus/decode/dflash_torch.py, which is distributed under the following license (THIRD_PARTY.md):
#
# MIT License
#
# Copyright (c) 2023-2026 The ggml authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""A float64 reference of a DFlash 2 drafter on a single-stream target (the dense Qwen3.8-27B line: Swift 1.5,
ThinkingCap; incoai/Qwen3.8-27B-DFlash2), CPU only: the oracle the own drafter module's ops are gated against
(docs/specs/spec-own-quant.md: G-acc / G-model take a float64 reference).

The forward is caminus/decode/dflash_torch.py's (llama.cpp's dflash.cpp / draft-dflash), in float64, with every weight
read through caminus.ggufio and decoded by caminus.own_dequant (no gguf-py, no torch):
  inject(pos0, feats): feats [n_target_layers, n, D], the residual entering each target layer at positions pos0 .. +n
    (Dense35Verify / Q35OwnPrefill feature_layers; a single stream, so no stream mean) -> fc -> RMSNorm -> per drafter
    layer K (RMSNorm per head, NeoX RoPE over the whole head) and V into the drafter's cache at those positions.
  draft(last, n0, k): the block [last, MASK x k] at positions n0 .. n0 + k through the drafter's layers (dynamic short
    convs around attention and the FFN; non-causal attention over the window before n0 and the block), the target's
    output norm-free head on the drafter's output norm, then the selector: top-k candidates per position, the low-rank
    predecessor -> successor scores, the greedy walk. Returns the k drafted tokens and the block's logits.
Weights are decoded per use (the target's head in row chunks), so the host never holds a float64 copy of a 1 GB
matrix; slow by design (seconds per draft)."""
import numpy as np

from . import ggufio
from .own_dequant import dequant, rows


def _rms(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


class RefDFlash:
    def __init__(self, path, target_path, max_pos=4096):
        self.F, self.TF = ggufio.GGUFFile(path), ggufio.GGUFFile(target_path)
        m = self.F.meta
        hp = lambda k: m[f'dflash.{k}']
        self.nl, self.D, self.ffn = int(hp('block_count')), int(hp('embedding_length')), int(hp('feed_forward_length'))
        self.nh, self.nkv, self.hd = int(hp('attention.head_count')), int(hp('attention.head_count_kv')), int(hp('attention.key_length'))
        self.eps, self.base = float(hp('attention.layer_norm_rms_epsilon')), float(hp('rope.freq_base'))
        self.block, self.ks, self.gs = int(hp('block_size')), int(hp('conv_kernel_size')), int(hp('conv_group_size'))
        self.rank, self.topk, self.window = int(hp('selector_rank')), int(hp('selector_top_k')), int(hp('attention.sliding_window'))
        self.layers_tgt = [int(v) for v in hp('target_layers')]
        self.mask = int(m['tokenizer.ggml.mask_token_id'])
        if bool(hp('attention.causal')): raise ValueError('RefDFlash: a causal drafter (DFlash 2 drafts the block non-causally)')
        self.T, self.TT = ({t.name: t for t in f.tensors} for f in (self.F, self.TF))
        if self.TT['token_embd.weight'].shape[0] != self.D: raise ValueError('RefDFlash: the target\'s n_embd differs')
        self.vocab = self.TT['output.weight'].rows()[0]
        self.K = np.zeros((self.nl, max_pos, self.nkv, self.hd))
        self.V = np.zeros((self.nl, max_pos, self.nkv, self.hd))
        self.inv_freq = self.base ** (-np.arange(0, self.hd // 2, dtype=np.float64) * 2 / self.hd)
        self.mask_emb = self._embed([self.mask])

    # weights, decoded per use
    def _vec(self, name):
        t = self.T[name]
        return dequant(t.data(), t.type, t.n_values).astype(np.float64).reshape(t.shape[::-1])

    def _mat(self, name):
        t = self.T[name]
        return rows(t, 0, t.rows()[0]).astype(np.float64)

    def _mm(self, x, name):
        """x [n, in] @ W^T for the drafter's matrix `name` [out, in]"""
        return x @ self._mat(name).T

    def _embed(self, ids):
        t = self.TT['token_embd.weight']
        return np.concatenate([rows(t, int(i), int(i) + 1) for i in ids]).astype(np.float64)

    def _rope(self, x, pos):
        """NeoX RoPE over the whole head: pairs (j, j + hd/2); x [n, heads, hd], pos [n]"""
        ang = np.asarray(pos, dtype=np.float64)[:, None] * self.inv_freq[None, :]
        c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
        a, b = x[..., :self.hd // 2], x[..., self.hd // 2:]
        return np.concatenate([a * c - b * s, a * s + b * c], axis=-1)

    def _conv(self, h, dyn, base, side):
        """the dynamic depthwise conv over the block positions, groups of gs channels (dflash_torch._conv)"""
        n, C = h.shape; ng = C // self.gs
        coeff = dyn.reshape(n, 2, self.ks, ng)[:, side]                  # [n, tap, group]
        w = np.repeat(coeff, self.gs, axis=-1) + base[side][None]         # [n, tap, C]; base [side, tap, C]
        out = np.zeros_like(h)
        for tap in range(min(self.ks, n)):
            vals = h if tap == 0 else np.concatenate([np.zeros((tap, C)), h[:n - tap]])
            out = out + w[:, tap] * vals
        return out

    def _check(self, p0, n):
        if p0 < 0 or p0 + n > self.K.shape[1]: raise ValueError(f'RefDFlash: positions {p0} .. {p0 + n - 1} outside 0 .. {self.K.shape[1] - 1}')

    def inject(self, pos0, feats):
        feats = np.asarray(feats, dtype=np.float64)
        nt, n, D = feats.shape
        if nt != len(self.layers_tgt) or D != self.D: raise ValueError(f'RefDFlash.inject: feats {feats.shape}, expect [{len(self.layers_tgt)}, n, {self.D}]')
        self._check(pos0, n)
        x = feats.transpose(1, 0, 2).reshape(n, -1)                      # [n, n_target_layers * D], target-layer order
        g = _rms(self._mm(x, 'fc.weight'), self._vec('enc.output_norm.weight'), self.eps)
        pos = np.arange(pos0, pos0 + n)
        for il in range(self.nl):
            b = f'blk.{il}.'
            k = _rms(self._mm(g, b + 'attn_k.weight').reshape(n, self.nkv, self.hd), self._vec(b + 'attn_k_norm.weight'), self.eps)
            self.K[il, pos0:pos0 + n] = self._rope(k, pos)
            self.V[il, pos0:pos0 + n] = self._mm(g, b + 'attn_v.weight').reshape(n, self.nkv, self.hd)

    def inject_g(self, pos0, g):
        """inject() from g [n, D] = rms(fc(features)) * enc_norm already (llama.cpp's inp_g_embeddings: the anchor of the
        layers' K / V without the target features)"""
        g = np.asarray(g, dtype=np.float64); n = g.shape[0]
        self._check(pos0, n)
        pos = np.arange(pos0, pos0 + n)
        for il in range(self.nl):
            b = f'blk.{il}.'
            k = _rms(self._mm(g, b + 'attn_k.weight').reshape(n, self.nkv, self.hd), self._vec(b + 'attn_k_norm.weight'), self.eps)
            self.K[il, pos0:pos0 + n] = self._rope(k, pos)
            self.V[il, pos0:pos0 + n] = self._mm(g, b + 'attn_v.weight').reshape(n, self.nkv, self.hd)

    def head(self, emb, chunk=8192):
        """logits [n, vocab] = emb @ output^T, the target's head decoded in row chunks"""
        t = self.TT['output.weight']
        out = np.empty((emb.shape[0], self.vocab))
        for r0 in range(0, self.vocab, chunk):
            r1 = min(self.vocab, r0 + chunk)
            out[:, r0:r1] = emb @ rows(t, r0, r1).astype(np.float64).T
        return out

    def _topk(self, logits, k):
        """the k largest per row, equal values ordered by the lower token id (dflash_torch._topk's order)"""
        idx = np.lexsort((np.arange(logits.shape[-1])[None, :].repeat(logits.shape[0], 0), -logits), axis=-1)[:, :k]
        return np.take_along_axis(logits, idx, -1), idx

    def forward(self, last, n0, k):
        """the block's drafter output emb [k + 1, D] (after the output norm) at positions n0 .. n0 + k; writes the block's
        K / V into the cache as draft() does"""
        nb = k + 1
        self._check(n0, nb)
        x = np.concatenate([self._embed([int(last)]), np.repeat(self.mask_emb, k, axis=0)])
        pos = np.arange(n0, n0 + nb)
        # the sliding window per query, as llama.cpp masks it (is_masked_swa STANDARD, also non-causal): row p sees
        # positions [max(0, p - (W - 1)), n0 + k], the whole block (dflash_torch took one lower bound for the block; the same
        # below W positions)
        lo = max(0, n0 - (self.window - 1))
        m = n0 + nb - lo
        mask = np.zeros((nb, m), dtype=bool)
        for r in range(nb): mask[r, :max(0, n0 + r - (self.window - 1) - lo)] = True
        rep = self.nh // self.nkv
        self._last_layers = []                                           # (each layer's output: the anchor's l_out-L)
        for il in range(self.nl):
            b = f'blk.{il}.'
            ca_base, cf_base = self._vec(b + 'attn_conv_base'), self._vec(b + 'ffn_conv_base')
            nn = _rms(x, self._vec(b + 'attn_norm.weight'), self.eps)
            dyn_a = self._mm(nn, b + 'attn_conv_proj.weight')
            nn = self._conv(nn, dyn_a, ca_base, 0)
            q = self._rope(_rms(self._mm(nn, b + 'attn_q.weight').reshape(nb, self.nh, self.hd), self._vec(b + 'attn_q_norm.weight'), self.eps), pos)
            kb = self._rope(_rms(self._mm(nn, b + 'attn_k.weight').reshape(nb, self.nkv, self.hd), self._vec(b + 'attn_k_norm.weight'), self.eps), pos)
            vb = self._mm(nn, b + 'attn_v.weight').reshape(nb, self.nkv, self.hd)
            self.K[il, n0:n0 + nb], self.V[il, n0:n0 + nb] = kb, vb
            Kh = np.repeat(self.K[il, lo:n0 + nb], rep, axis=1)            # [m, nh, hd]
            Vh = np.repeat(self.V[il, lo:n0 + nb], rep, axis=1)
            s = np.einsum('qhd,khd->hqk', q, Kh) / np.sqrt(self.hd)
            s[:, mask] = -np.inf
            s = np.exp(s - s.max(-1, keepdims=True)); att = s / s.sum(-1, keepdims=True)
            o = np.einsum('hqk,khd->qhd', att, Vh).reshape(nb, -1)
            cur = self._conv(self._mm(o, b + 'attn_output.weight'), dyn_a, ca_base, 1)
            ffn_inp = cur + x
            h = _rms(ffn_inp, self._vec(b + 'ffn_norm.weight'), self.eps)
            dyn_f = self._mm(h, b + 'ffn_conv_proj.weight')
            h = self._conv(h, dyn_f, cf_base, 0)
            g = self._mm(h, b + 'ffn_gate.weight')
            h = self._mm(g / (1 + np.exp(-g)) * self._mm(h, b + 'ffn_up.weight'), b + 'ffn_down.weight')
            x = self._conv(h, dyn_f, cf_base, 1) + ffn_inp
            self._last_layers.append(x.copy())
        return _rms(x, self._vec('output_norm.weight'), self.eps)

    def draft(self, last, n0, k):
        """(the k drafted tokens, the block's logits [k + 1, vocab], the selector's per-position picks)"""
        emb = self.forward(last, n0, k)
        logits = self.head(emb)
        unary, cand = self._topk(logits, self.topk)
        gate = self._mm(emb, 'selector_hidden.weight')                     # [nb, rank]
        sp, sn = self.T['selector_predecessor.weight'], self.T['selector_successor.weight']
        row = lambda t, i: rows(t, int(i), int(i) + 1)[0].astype(np.float64)
        succ = np.stack([[row(sn, c) for c in cand[i]] for i in range(1, k + 1)])                    # [k, topk, rank]
        prev = np.stack([[row(sp, last)] * self.topk] + [[row(sp, c) for c in cand[i]] for i in range(1, k)])   # [k, pred, rank]
        score = np.einsum('ipr,isr->ips', prev * gate[1:, None, :], succ) + unary[1:, None, :]       # [k, pred, succ]
        pick, pr = [], 0
        for i in range(k):                                                 # the walk (pred row 0 at position 1)
            pr = int(np.argmax(score[i, pr])); pick.append(pr)
        return [int(cand[i + 1, p]) for i, p in enumerate(pick)], logits, pick
