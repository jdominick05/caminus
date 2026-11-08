# Parts of this file (the DFlash drafter's forward and the draft-dflash loop) are ported to PyTorch from llama.cpp's
# src/models/dflash.cpp and common/speculative.cpp (https://github.com/ggml-org/llama.cpp, through the ignite-llama fork),
# which is distributed under the following license (THIRD_PARTY.md):
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
"""DFlash 2 drafter (incoai/Qwen3.8-Flash-Next-DFlash2) in torch, following ignite-llama's src/models/dflash.cpp and
common/speculative.cpp (draft-dflash):
- inject(): for every position the target processed, the stream mean of its residual entering the target layers
  (dflash.target_layers) -> fc -> RMSNorm -> per layer K (RMSNorm per head, NeoX RoPE) and V, into the drafter's own
  cache at that position (later writes at a position overwrite earlier ones, so rejected drafts are replaced).
- draft(): the block [last token, MASK x (block - 1)] at positions n .. through 5 layers (dynamic short convs around
  attention and FFN, non-causal attention over the cached context before n and the block), the target's output head,
  then the DFlash 2 selector: top-k candidates per position, low-rank predecessor -> successor scores, a greedy walk.
Weights: Q8_0 from the GGUF, dequantized once (fp32); the target's token embeddings (rows dequantized per call) and
output head (fp16) come from the target GGUF."""
import contextlib
import numpy as np
import torch
from ..engine.flashnext import quants          # (puts gguf-py on the path)
from gguf import GGUFReader            # noqa: E402
from ..engine.k3 import check_span             # noqa: E402  (host bound checks: k3.ContextBoundsError)


def _rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


class _deterministic:
    """torch's deterministic mode for a span of the drafter's work, the previous mode restored after. On ROCm it makes
    rocBLAS run without atomics: the attention output (softmax @ V, a K = max_pos reduction) took split-K kernels whose
    atomic sums made the same replay on the same inputs differ in the last bits (62 distinct block K / V states in 150
    replays), and the drafts flipped at near-ties (REAP-308 prose: 2 draft positions; tmp/m2/s7/det_test.py,
    det_graph.py). The captured graph keeps the kernels picked at capture. Cost at T = 4: the attention products alone
    +0.76 ms per draft (8.04 -> 8.80), the whole draft +2.0 ms; inject +0.13 ms."""
    def __enter__(self):
        self.prev = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
        torch.use_deterministic_algorithms(True)

    def __exit__(self, *exc):
        torch.use_deterministic_algorithms(self.prev[0], warn_only=self.prev[1])


# where the drafter runs deterministic: 'attn' (the captured draft's attention products only, and inject), 'all' (the
# whole capture, and inject), 'none' (test only)
DET_SCOPE = 'attn'


def _det(scope):
    return _deterministic() if DET_SCOPE in scope else contextlib.nullcontext()


class DFlash2:
    def __init__(self, path, target, device='cuda:1', max_pos=4096, wdtype=torch.float16):
        """target: the FlashNextK3 / FlashNextTP whose GGUF holds token_embd and output (shared with the drafter)"""
        rd = GGUFReader(path, 'r'); T = {t.name: t for t in rd.tensors}
        f = {x.name: x for x in rd.fields.values()}
        hp = lambda k: f[f'dflash.{k}'].contents()
        self.dev, self.tgt = device, target
        self.nl, self.D, self.ffn = int(hp('block_count')), int(hp('embedding_length')), int(hp('feed_forward_length'))
        self.nh, self.nkv, self.hd = int(hp('attention.head_count')), int(hp('attention.head_count_kv')), int(hp('attention.key_length'))
        self.eps, self.base = float(hp('attention.layer_norm_rms_epsilon')), float(hp('rope.freq_base'))
        self.block, self.ks, self.gs = int(hp('block_size')), int(hp('conv_kernel_size')), int(hp('conv_group_size'))
        self.rank, self.topk, self.window = int(hp('selector_rank')), int(hp('selector_top_k')), int(hp('attention.sliding_window'))
        self.layers_tgt = [int(v) for v in hp('target_layers')]
        self.mask = int(f['tokenizer.ggml.mask_token_id'].contents())
        dq = lambda n, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(
            quants.dequantize(np.asarray(T[n].data), T[n].tensor_type).astype(np.float32))).to(device, dt)
        W = lambda n: dq(n, wdtype)                                         # matrices in fp16 (half the bytes), norms fp32
        self.fc, self.enc_norm, self.out_norm = W('fc.weight'), dq('enc.output_norm.weight'), dq('output_norm.weight')
        self.sel_hidden = W('selector_hidden.weight')
        self.sel_prev, self.sel_next = dq('selector_predecessor.weight', torch.float16), dq('selector_successor.weight', torch.float16)
        self.L = []
        for il in range(self.nl):
            g = lambda n, dt=torch.float32: dq(f'blk.{il}.{n}', dt)
            w = lambda n: g(n, wdtype)
            self.L.append(dict(attn_norm=g('attn_norm.weight'), ffn_norm=g('ffn_norm.weight'), q_norm=g('attn_q_norm.weight'),
                               k_norm=g('attn_k_norm.weight'), wq=w('attn_q.weight'), wk=w('attn_k.weight'), wv=w('attn_v.weight'),
                               wo=w('attn_output.weight'), wg=w('ffn_gate.weight'), wu=w('ffn_up.weight'), wd=w('ffn_down.weight'),
                               ca_base=g('attn_conv_base'), ca_proj=w('attn_conv_proj.weight'),
                               cf_base=g('ffn_conv_base'), cf_proj=w('ffn_conv_proj.weight')))
        # the target's output head, fp16 (dequantized in row chunks to bound host memory)
        ot = target.T['output.weight']; od = np.asarray(ot.data); rows = od.shape[0]
        self.head = torch.empty(rows, self.D, dtype=torch.float16, device=device)
        for r0 in range(0, rows, 16384):
            self.head[r0:r0 + 16384] = torch.from_numpy(quants.dequantize(np.ascontiguousarray(od[r0:r0 + 16384]), ot.tensor_type).astype(np.float16)).to(device)
        self.K = torch.zeros(self.nl, max_pos, self.nkv, self.hd, device=device)
        self.V = torch.zeros(self.nl, max_pos, self.nkv, self.hd, device=device)
        inv = self.base ** (-torch.arange(0, self.hd // 2, device=device, dtype=torch.float64) * 2 / self.hd)
        self.inv_freq = inv
        self.mask_emb = self._embed([self.mask])                            # the mask token's embedding, once

    def _embed(self, ids):
        te = self.tgt.T['token_embd.weight']
        return torch.from_numpy(quants.dequantize(np.ascontiguousarray(np.asarray(te.data)[ids]), te.tensor_type).astype(np.float32)).to(self.dev)

    def _topk(self, logits, k):
        """torch.topk(logits, k) with equal values ordered by the lower token id: the fp16 head's logits tie often, and
        torch.topk does not define the order (or the choice at the k-th place) among equal values. Unique int64 keys:
        the float's order-preserving int32 bits, then the reversed id; the values are gathered back (the same as topk's).
        A guarantee, not the measured cause of the non-reproducible drafts (that was the attention: _deterministic);
        ~0.2 ms per draft."""
        n = logits.shape[-1]; assert n <= 1 << 18, n
        v = logits.float().contiguous().view(torch.int32).to(torch.int64)
        key = torch.where(v >= 0, v, v ^ 0x7FFFFFFF) * (1 << 18) + ((1 << 18) - 1 - torch.arange(n, device=logits.device))
        idx = torch.topk(key, k, dim=-1).indices
        return logits.gather(-1, idx), idx

    def _mm(self, x, w):
        """x [n, in] fp32 @ w [out, in] (fp16 weights: fp16 matmul, fp32 result)"""
        return (x.to(w.dtype) @ w.T).float()

    def _rope(self, x, pos):
        """NeoX RoPE over the whole head: pairs (j, j + hd/2); x [n, heads, hd], pos [n]"""
        ang = (pos.to(torch.float64)[:, None] * self.inv_freq[None, :]).float()     # [n, hd/2]
        c, s = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
        a, b = x[..., :self.hd // 2], x[..., self.hd // 2:]
        return torch.cat([a * c - b * s, a * s + b * c], dim=-1)

    def _conv(self, h, dyn, base, side):
        """build_dflash2_conv: per-token dynamic depthwise conv over the block positions, groups of gs channels"""
        n, C = h.shape; ng = C // self.gs
        coeff = dyn.view(n, 2, self.ks, ng)[:, side]                      # [n, tap, group] (ne order: group, tap, side)
        w = coeff.repeat_interleave(self.gs, dim=-1) + base[side][None]    # [n, tap, C]; base [side, tap, C]
        out = None
        for tap in range(min(self.ks, n)):
            vals = h if tap == 0 else torch.cat([torch.zeros(tap, C, device=h.device), h[:n - tap]])
            term = w[:, tap] * vals
            out = term if out is None else out + term
        return out

    @torch.no_grad()
    def inject(self, pos0, feats):
        """feats [n_layers_tgt, n, D] (the target's exported features for positions pos0 .. pos0 + n - 1). Through a captured
        graph (inject_graph, the default: ~50 small launches per call otherwise) or eagerly; the same result bitwise
        (tmp/m3/loop/inject_check.py)"""
        n = feats.shape[1]
        check_span(pos0, n, self.K.shape[1], what='DFlash2.inject')   # host check of the cache rows written
        if getattr(self, 'inject_graph', True):
            if getattr(self, 'gi_n', None) != n: self._capture_inject(n)
            assert feats.shape == self.gi_f.shape, (feats.shape, self.gi_f.shape)
            self.gi_f.copy_(feats); self.gi_pos0.fill_(int(pos0)); self.gi_graph.replay()
            return
        with _det(('attn', 'all')):
            x = feats.to(self.dev).permute(1, 0, 2).reshape(n, -1)       # [n, 5 D] in target-layer order
            g = _rms(self._mm(x, self.fc), self.enc_norm, self.eps)
            pos = torch.arange(pos0, pos0 + n, device=self.dev)
            for il, L in enumerate(self.L):
                k = _rms(self._mm(g, L['wk']).view(n, self.nkv, self.hd), L['k_norm'], self.eps)
                self.K[il, pos0:pos0 + n] = self._rope(k, pos)
                self.V[il, pos0:pos0 + n] = self._mm(g, L['wv']).view(n, self.nkv, self.hd)

    def _inject_static(self, n):
        """inject() on the static buffers: features gi_f [n_layers_tgt, n, D], first position gi_pos0 (a device scalar).
        The same ops on the same values as the eager inject, with the shared pieces computed once (g's fp16 cast for the
        10 K / V products, RoPE's cos / sin for the 5 layers): elementwise, so the same bits. Deterministic mode only
        around the matrix products (rocBLAS without atomics, as eager); index_copy_ outside it (its deterministic form
        sorts the indices: slower, and the rows are distinct anyway)"""
        hd = self.hd
        x = self.gi_f.permute(1, 0, 2).reshape(n, -1)                    # [n, 5 D] in target-layer order
        with _det(('attn', 'all')): y = self._mm(x, self.fc)
        g = _rms(y, self.enc_norm, self.eps); gh = g.to(self.fc.dtype)  # (_mm's own cast, once)
        pos = self.gi_pos0 + torch.arange(n, device=self.dev)
        ang = (pos.to(torch.float64)[:, None] * self.inv_freq[None, :]).float()     # as _rope
        c, s = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
        for il, L in enumerate(self.L):
            with _det(('attn', 'all')): kk = self._mm(gh, L['wk']); vv = self._mm(gh, L['wv'])
            k = _rms(kk.view(n, self.nkv, hd), L['k_norm'], self.eps)
            a, b = k[..., :hd // 2], k[..., hd // 2:]
            self.K[il].index_copy_(0, pos, torch.cat([a * c - b * s, a * s + b * c], dim=-1))
            self.V[il].index_copy_(0, pos, vv.view(n, self.nkv, hd))

    @torch.no_grad()
    def _capture_inject(self, n):
        """capture _inject_static(n) as a graph (its products under inject's deterministic scope, so the same kernels as the
        eager inject). The warm-ups execute and write K / V rows gi_pos0 .. + n - 1: they run at the scratch rows max_pos - n,
        saved before and restored after, so the live K / V is never touched (as _capture)"""
        dev = self.dev; max_pos = self.K.shape[1]; s0 = max_pos - n
        assert 1 <= n <= max_pos, (n, max_pos)
        self.gi_f = torch.zeros(len(self.layers_tgt), n, self.D, device=dev)
        self.gi_pos0 = torch.full((), s0, dtype=torch.long, device=dev)
        torch.cuda.synchronize(dev)
        keep = (self.K[:, s0:].clone(), self.V[:, s0:].clone())
        snap = (self.K[:, :s0].clone(), self.V[:, :s0].clone()) if getattr(self, 'check_capture', False) else None
        st = torch.cuda.Stream(device=dev)
        with torch.cuda.device(dev), torch.cuda.stream(st):
            for _ in range(3): self._inject_static(n)
            torch.cuda.synchronize(dev)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=st): self._inject_static(n)
        torch.cuda.synchronize(dev)
        self.K[:, s0:].copy_(keep[0]); self.V[:, s0:].copy_(keep[1]); torch.cuda.synchronize(dev)
        if snap is not None:
            assert torch.equal(snap[0], self.K[:, :s0]) and torch.equal(snap[1], self.V[:, :s0]), 'inject capture wrote the live K / V'
        del keep, snap
        self.gi_graph, self.gi_n = g, n

    # -- graph-captured drafting: static shapes (attention over the whole cache with a validity mask), the position and
    # the anchor in device tensors, the selector walk on the device; one replay per draft --------------------------------
    @torch.no_grad()
    def _draft_static(self, nb):
        dev = self.dev; max_pos = self.K.shape[1]
        x = torch.cat([self.g_anchor_emb, self.mask_emb.expand(nb - 1, -1)])
        pos = self.g_n0 + torch.arange(nb, device=dev)
        kpos = torch.arange(max_pos, device=dev)
        valid = (kpos < self.g_n0 + nb) & (kpos >= self.g_n0 + nb - self.window)       # the context and the block
        rep = self.nh // self.nkv
        for il, L in enumerate(self.L):
            nn = _rms(x, L['attn_norm'], self.eps)
            dyn_a = self._mm(nn, L['ca_proj'])
            nn = self._conv(nn, dyn_a, L['ca_base'], 0)
            q = self._rope(_rms(self._mm(nn, L['wq']).view(nb, self.nh, self.hd), L['q_norm'], self.eps), pos)
            kb = self._rope(_rms(self._mm(nn, L['wk']).view(nb, self.nkv, self.hd), L['k_norm'], self.eps), pos)
            vb = self._mm(nn, L['wv']).view(nb, self.nkv, self.hd)
            self.K[il].index_copy_(0, pos, kb); self.V[il].index_copy_(0, pos, vb)
            qg = q.view(nb, self.nkv, rep, self.hd).permute(1, 2, 0, 3).reshape(self.nkv, rep * nb, self.hd)   # [kv, rep*nb, hd]
            with _det(('attn',)):   # (softmax @ V reduces over max_pos: split-K with atomics otherwise)
                sc = torch.einsum('gqd,pgd->gqp', qg, self.K[il]) / self.hd ** 0.5
                sc = sc.masked_fill(~valid[None, None, :], float('-inf'))
                o = torch.einsum('gqp,pgd->gqd', torch.softmax(sc, dim=-1), self.V[il])                   # [kv, rep*nb, hd]
            o = o.view(self.nkv, rep, nb, self.hd).permute(2, 0, 1, 3).reshape(nb, -1)
            cur = self._conv(self._mm(o, L['wo']), dyn_a, L['ca_base'], 1)
            ffn_inp = cur + x
            h = _rms(ffn_inp, L['ffn_norm'], self.eps)
            dyn_f = self._mm(h, L['cf_proj'])
            h = self._conv(h, dyn_f, L['cf_base'], 0)
            h = self._mm(torch.nn.functional.silu(self._mm(h, L['wg'])) * self._mm(h, L['wu']), L['wd'])
            x = self._conv(h, dyn_f, L['cf_base'], 1) + ffn_inp
        emb = _rms(x, self.out_norm, self.eps)
        logits = (self.head @ emb.half().T).T.float()                     # weight-major: 676 vs 591 GB/s on the XT
        unary, cand = self._topk(logits, self.topk)
        gate = self._mm(emb, self.sel_hidden)
        succ = self.sel_next[cand[1:]].float()
        prev = torch.cat([self.sel_prev[self.g_anchor_id.expand(1, self.topk)], self.sel_prev[cand[1:-1]]]).float()
        score = torch.einsum('ipr,isr->ips', prev * gate[1:, None, :], succ) + unary[1:, None, :]
        pr = torch.zeros(1, dtype=torch.long, device=dev); pick = []
        for i in range(nb - 1):                                          # index_select: no host sync inside the capture
            pr = torch.argmax(score[i].index_select(0, pr), dim=-1); pick.append(pr)
        self.g_out.copy_(cand[1:].gather(1, torch.stack(pick)).squeeze(1))

    def compile(self):
        """fuse the many small elementwise ops of the static draft with torch.compile (graph capture alone gained
        nothing: the time is GPU time, not launches)"""
        self._draft_static = torch.compile(self._draft_static, dynamic=False)

    @torch.no_grad()
    def _capture(self, nb):
        """warm the static draft and capture it as a graph. The warm-up runs execute (the capture only records), and each
        writes the block's K / V at positions g_n0 .. g_n0 + nb - 1: they run at the scratch position max_pos - nb, and
        those cache rows are saved before and restored after, so the live (injected) K / V is never touched"""
        dev = self.dev; max_pos = self.K.shape[1]; s0 = max_pos - nb
        assert 2 <= nb <= max_pos, (nb, max_pos)
        self.g_anchor_emb = torch.zeros(1, self.D, device=dev); self.g_anchor_id = torch.zeros((), dtype=torch.long, device=dev)
        self.g_n0 = torch.full((), s0, dtype=torch.long, device=dev); self.g_out = torch.zeros(nb - 1, dtype=torch.long, device=dev)
        torch.cuda.synchronize(dev)
        keep = (self.K[:, s0:].clone(), self.V[:, s0:].clone())
        snap = (self.K[:, :s0].clone(), self.V[:, :s0].clone()) if getattr(self, 'check_capture', False) else None
        st = torch.cuda.Stream(device=dev)
        with torch.cuda.device(dev), torch.cuda.stream(st), _det(('all',)):
            for _ in range(3): self._draft_static(nb)                       # warm up (allocator, BLAS handles)
            torch.cuda.synchronize(dev)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=st): self._draft_static(nb)
        torch.cuda.synchronize(dev)
        self.K[:, s0:].copy_(keep[0]); self.V[:, s0:].copy_(keep[1]); torch.cuda.synchronize(dev)
        if snap is not None:   # (test hook) every live row outside the scratch is bitwise untouched
            assert torch.equal(snap[0], self.K[:, :s0]) and torch.equal(snap[1], self.V[:, :s0]), 'capture wrote the live K / V'
        del keep, snap
        self.graph, self.graph_nb = g, nb

    def warm(self, k):
        """capture the k-draft graph (and the inject graph for k + 1 positions, a T = k + 1 verify pass) now (call after
        loading, before anything is timed; leaves the K / V cache as it was)"""
        if getattr(self, 'graph_nb', None) != k + 1: self._capture(k + 1)
        if getattr(self, 'inject_graph', True) and getattr(self, 'gi_n', None) != k + 1: self._capture_inject(k + 1)

    @torch.no_grad()
    def draft_graph(self, last, n0, k):
        """draft() through a captured graph (same result; the K / V block writes happen inside the replay)"""
        nb = k + 1
        check_span(n0, nb, self.K.shape[1], what='DFlash2.draft_graph')   # the replay's index_copy_ rows
        if getattr(self, 'graph_nb', None) != nb: self._capture(nb)
        er = getattr(self.tgt, '_emb_row', None)          # the target's cached row: the same dequantize, no host re-decode
        self.g_anchor_emb.copy_(torch.from_numpy(er(int(last)))[None] if er is not None else self._embed([int(last)]))
        self.g_anchor_id.fill_(int(last)); self.g_n0.fill_(int(n0))
        self.graph.replay()
        return self.g_out.tolist()

    def draft(self, last, n0, k):
        """k draft tokens after token `last` at position n0 (the context before n0 must have been injected)"""
        with _deterministic(): return self._draft_eager(last, n0, k)

    @torch.no_grad()
    def _draft_eager(self, last, n0, k):
        nb = k + 1
        check_span(n0, nb, self.K.shape[1], what='DFlash2.draft')
        x = torch.cat([self._embed([int(last)]), self.mask_emb.expand(k, -1)])
        pos = torch.arange(n0, n0 + nb, device=self.dev)
        lo = max(0, n0 + nb - self.window)                                # the sliding window (non-causal block)
        for il, L in enumerate(self.L):
            nn = _rms(x, L['attn_norm'], self.eps)
            dyn_a = self._mm(nn, L['ca_proj'])
            nn = self._conv(nn, dyn_a, L['ca_base'], 0)
            q = self._rope(_rms(self._mm(nn, L['wq']).view(nb, self.nh, self.hd), L['q_norm'], self.eps), pos)
            kb = self._rope(_rms(self._mm(nn, L['wk']).view(nb, self.nkv, self.hd), L['k_norm'], self.eps), pos)
            vb = self._mm(nn, L['wv']).view(nb, self.nkv, self.hd)
            self.K[il, n0:n0 + nb], self.V[il, n0:n0 + nb] = kb, vb
            K, V = self.K[il, lo:n0 + nb], self.V[il, lo:n0 + nb]           # context before n0, then the block
            rep = self.nh // self.nkv
            Kh, Vh = K.repeat_interleave(rep, dim=1), V.repeat_interleave(rep, dim=1)   # [m, nh, hd]
            att = torch.softmax(torch.einsum('qhd,khd->hqk', q, Kh) / self.hd ** 0.5, dim=-1)
            o = torch.einsum('hqk,khd->qhd', att, Vh).reshape(nb, -1)
            cur = self._conv(self._mm(o, L['wo']), dyn_a, L['ca_base'], 1)
            ffn_inp = cur + x
            h = _rms(ffn_inp, L['ffn_norm'], self.eps)
            dyn_f = self._mm(h, L['cf_proj'])
            h = self._conv(h, dyn_f, L['cf_base'], 0)
            h = self._mm(torch.nn.functional.silu(self._mm(h, L['wg'])) * self._mm(h, L['wu']), L['wd'])
            x = self._conv(h, dyn_f, L['cf_base'], 1) + ffn_inp
        emb = _rms(x, self.out_norm, self.eps)
        logits = (emb.half() @ self.head.T).float()                      # [nb, vocab]
        # DFlash 2 selector and the greedy lattice walk
        unary, cand = self._topk(logits, self.topk)
        gate = self._mm(emb, self.sel_hidden)                                     # [nb, rank]
        # every position's scores in one go: position 1 from the anchor, position i >= 2 from position i-1's candidates
        succ = self.sel_next[cand[1:]].float()                                       # [nb-1, topk, rank]
        prev = torch.cat([self.sel_prev[torch.full((1, self.topk), int(last), device=self.dev)],
                          self.sel_prev[cand[1:-1]]]).float()                          # [nb-1, topk(pred), rank]
        cond = prev * gate[1:, None, :]
        score = torch.einsum('ipr,isr->ips', cond, succ) + unary[1:, None, :]          # [nb-1, pred, succ]
        pick, pr = [], torch.zeros((), dtype=torch.long, device=self.dev)
        for i in range(nb - 1):                                                      # the walk (pred row 0 at position 1)
            pr = torch.argmax(score[i, pr]); pick.append(pr)
        sel = torch.stack(pick)
        return cand[1:].gather(1, sel[:, None]).squeeze(1).tolist()
