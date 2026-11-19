"""A float64 host forward of the Qwen3.8-Flash-Next GSQ-RCO Coder (architecture qwen4exp) on Caminus's own reader
(caminus.ggufio) and decoders (caminus.own_dequant): the oracle of the Coder's own-kernel gates (exact weights, float64
activations: the 'fp64 reference mode' of docs/specs/spec-own-quant.md). The statements follow the ignite-llama fork's
graph (src/models/qwen4exp.cpp); tests/engine/test_coder_golden_cpu.py checks each block against llama.cpp's own tensors.

It is caminus.ref_q35 (Q35Ref) with the Coder's parts around it: the Gated DeltaNet block, the gated attention block and
the MoE FFN are Q35Ref's (layer-major over a chunk, one matrix decoded at a time, an expert decoded only for the tokens
that picked it); the GDN output gate is a sigmoid here (Qwen3.5's is a SiLU). The rest is this module's:
  - hyper-connections: the residual is 4 streams [N, 4, D] (hyper_connection.count), each block reads a mix of them and
    writes back into all four;
  - the PLE n-gram embedding (ple.layers: one Gated DeltaNet layer), added to the streams before that layer's mixer;
  - the head: a hyper-connection mix (output_hc_*) is the output norm, then output.weight.
A split GGUF is given by its first shard: the other shards' tensors are read from their own files (the PLE table is shard
2's only tensor; its rows are read with pread, 16 rows of 90 B per token, never through the mapping).

Scope: dense attention. The model's attention layers (attention.compress_ratios > 0) select cells with the QSA indexer
(top_k 2048 cells in pools of 4, plus a 3-cell tail); while a query sees at most top_k + ratio - 1 = 2051 cells the
selection is every cell, so dense attention IS the model's (capacity.QSA_EXACT). feed() refuses positions past that
(self.limit): a longer reference needs the indexer (the fp32 depth reference at 32K+, a later change; its selection goes
where _attn masks the cells: Q35Ref._attn's causal mask would become causal & selected).

The math (x the streams [N, 4, D], float64; rms over D per stream, eps the file's; Wd·y = the matrix applied):
  x = 4 copies of token_embd[t]
  mix(x; norm, down, up, inject):  xn = rms(x) * norm[4, D];  f = xn as [N, 4 D];  lo = silu((Wdown f) / 4)
      gate = Wup lo [N, 4 D];  mixed = mean over the streams of xn * sigmoid(gate);  inj = Winject f  [N, 4]
  combine(x, y, inj):  x_s += y * 2 sigmoid(inj_s / 4)   (each stream s; y the block's output [N, D])
  per layer:
    PLE layer: emb = the 16 table rows of the token's 2- and 3-grams (heads slowest) [N, 2560]
      key = rms(Wkey emb as [N, 4, D]) * norm_key;  query = rms(x) * norm_query;  s = sum_D(key * query) / sqrt(D)  [N, 4]
      gated = (Wvalue emb) * sigmoid(sign(s) sqrt(max(|s|, 1e-6)))  [N, 4, D]
      c = rms(gated) * norm_conv as [N, 4 D];  conv_t = sum_k w[k] c_{t - (3 - k) 3}  (4 taps, 3 apart: ple.ngram_size;
      the 9 earlier rows carry across feeds, zeros before the start);  x += gated + silu(conv)
    mixed, inj = mix(x; hc_attn_*);  x = combine(x, GDN or attention block (mixed), inj)
    mixed, inj = mix(x; hc_ffn_*);   x = combine(x, MoE FFN (mixed), inj)
      MoE FFN: softmax router over 256 experts, the 10 largest (ties: the lower id), weights renormalized;
      + sigmoid(Wsg . mixed) * the shared expert (Q35Ref._moe)
  logits = Woutput mix(x; output_hc_*, no inject)
  n-gram ids (PLE, caminus.prefill.feed.NgramIds): context token s back = EOS (ple.eos_token_id) from the first one before
  the sequence start or equal to EOS on; mixed = XOR_j (ctx_j * mult_j) mod 2^64; row = mixed % vocab_h + offset_h.

  python -I -m caminus.ref_coder GGUF "prompt" [--new 2] [--threads 16]"""
import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from caminus import ggufio, own_dequant
from caminus.ref_q35 import HEAD_ROWS, Q35Ref, sigmoid, silu


def shards(path):
    """[path] for a single-file GGUF, else every shard's path of a split one (llama.cpp's -%05d-of-%05d naming), first first"""
    m = re.match(r'(.*)-(\d{5})-of-(\d{5})\.gguf$', os.fspath(path))
    if not m: return [os.fspath(path)]
    n = int(m.group(3))
    if int(m.group(2)) != 1: raise ValueError(f'ref_coder: give the first shard of a split GGUF, not {path}')
    return [f'{m.group(1)}-{k:05d}-of-{n:05d}.gguf' for k in range(1, n + 1)]


class CoderRef(Q35Ref):
    gdn_gate = staticmethod(sigmoid)     # build_norm_gated: the Coder's one numerical difference from Qwen3.5's GDN

    def __init__(self, path, threads=None):
        paths = shards(path)
        self.F = F = ggufio.GGUFFile(paths[0])
        if F.arch != 'qwen4exp': raise ValueError(f'ref_coder: architecture {F.arch!r} (qwen4exp only)')
        n = int(F.meta.get('split.count', 1))
        if n != len(paths): raise ValueError(f'ref_coder: split.count {n}, {len(paths)} shard names from {path}')
        self.files = [F] + [ggufio.GGUFFile(p) for p in paths[1:]]
        for G in self.files[1:]:   # (one table of every shard's tensors: each Tensor reads its own file)
            dup = set(G.T) & set(F.T)
            if dup: raise ValueError(f'ref_coder: tensors in two shards: {sorted(dup)[:4]}')
            F.T.update(G.T)
        if int(F.meta.get('split.tensors.count', len(F.T))) != len(F.T):
            raise ValueError(f'ref_coder: {len(F.T)} tensors, the header says {F.meta["split.tensors.count"]}')
        hp = F.hp
        self.L, self.D = int(hp('block_count')), int(hp('embedding_length'))
        if int(hp('nextn_predict_layers') or 0) or any(n.startswith(f'blk.{self.L}.') for n in F.T):
            raise ValueError('ref_coder: a multi-token-prediction block (this reference reads the trunk only)')
        self.H, self.HKV, self.HD = int(hp('attention.head_count')), int(hp('attention.head_count_kv')), int(hp('attention.key_length'))
        self.NK, self.NV, self.S, self.DC = int(hp('ssm.group_count')), int(hp('ssm.time_step_rank')), int(hp('ssm.state_size')), int(hp('ssm.conv_kernel'))
        self.eps, self.base, self.nrot = float(hp('attention.layer_norm_rms_epsilon')), float(hp('rope.freq_base')), int(hp('rope.dimension_count'))
        rec = hp('attention.recurrent_layers'); fai = int(hp('full_attention_interval', 4))
        self.gdn = [bool(rec[i]) if rec is not None else (i + 1) % fai != 0 for i in range(self.L)]
        if int(hp('ssm.inner_size')) != self.NV * self.S or self.NV % self.NK or self.H % self.HKV or self.nrot > self.HD or self.nrot % 2:
            raise ValueError('ref_coder: head counts / sizes this reference does not read')
        if int(hp('attention.value_length', self.HD)) != self.HD: raise ValueError('ref_coder: value_length != key_length')
        sec = hp('rope.dimension_sections')
        if sec is not None and 2 * sum(int(v) for v in sec) != self.nrot:
            raise ValueError(f'ref_coder: rope sections {list(sec)} do not cover the {self.nrot} rotated dims')
        if any(k.startswith(f'{F.arch}.rope.scaling') for k in F.meta): raise ValueError('ref_coder: rope.scaling keys (not read)')
        unread = sorted(n for n in F.T if re.search(r'(\.scale|_s(\.weight)?|ffn_gate_up_exps(\.weight)?|output_norm\.weight)$', n))
        if unread: raise ValueError(f'ref_coder: tensors this reference does not read: {unread[:4]}')
        # the MoE FFN (Q35Ref._moe reads E, KU, FF, D)
        if any(np.ndim(hp(k)) for k in ('expert_count', 'expert_used_count', 'expert_feed_forward_length', 'expert_shared_feed_forward_length')):
            raise ValueError('ref_coder: per-block expert counts or widths (an array-valued expert key)')
        self.moe = True
        self.E, self.KU = int(hp('expert_count')), int(hp('expert_used_count'))
        self.FF, self.FFS = int(hp('expert_feed_forward_length')), int(hp('expert_shared_feed_forward_length'))
        if not 0 < self.KU <= self.E or any(f'blk.{i}.ffn_gate_inp.weight' not in F.T for i in range(self.L)):
            raise ValueError(f'ref_coder: {self.KU} of {self.E} experts, or a block without its router')
        # hyper-connections
        self.HC, self.HLR = int(hp('hyper_connection.count')), int(hp('hyper_connection.low_rank'))
        if self.HC < 2: raise ValueError(f'ref_coder: hyper_connection.count {self.HC}')
        # QSA: dense attention is the model's while a query sees at most top_k + ratio - 1 cells
        cr = hp('attention.compress_ratios')
        ratios = sorted({int(r) for r in (cr if cr is not None else [])} - {0})
        if len(ratios) > 1: raise ValueError(f'ref_coder: compress ratios {ratios} (the model shares one)')
        self.limit = int(hp('attention.indexer.top_k')) + ratios[0] - 1 if ratios else None
        # PLE
        pl = hp('ple.layers')
        self.ple_layers = [int(v) for v in (pl if pl is not None else [])]
        if len(self.ple_layers) > 1 or any(not self.gdn[il] for il in self.ple_layers):
            raise ValueError(f'ref_coder: PLE layers {self.ple_layers} (one, a Gated DeltaNet layer)')
        if self.ple_layers:
            from caminus.prefill.feed import NgramIds   # (the engine's n-gram hash: torch and gguf-py come with the module)
            self.ngram = NgramIds(hp('ple.layer_multipliers'), hp('ple.head_offsets'), hp('ple.head_vocab_sizes'),
                                  hp('ple.ngram_size'), hp('ple.heads_per_ngram'), hp('ple.eos_token_id'))
            self.ple_dim, self.ple_kern = int(hp('embedding_length_per_layer_input')), int(hp('ple.conv_kernel'))
            self.ple_hist = (self.ple_kern - 1) * self.ngram.n_gram          # the conv's taps are ngram_size apart
            t = F.T['per_layer_token_embd.weight']
            if t.shape[0] != self.ple_dim or self.ngram.n_heads * self.ple_dim != self.D:
                raise ValueError(f'ref_coder: PLE table {list(t.shape)}: rows of {self.ple_dim} (grouped tables are not read), '
                                 f'{self.ngram.n_heads} heads x {self.ple_dim} != {self.D}')
        self.pool = ThreadPoolExecutor(threads or min(16, os.cpu_count() or 1))
        self.census = None
        self.reset()

    def reset(self):
        super().reset()
        self.toks = []                                                        # every token fed (the n-gram context)
        self.ple_c = np.zeros((self.ple_hist, self.HC * self.D)) if self.ple_layers else None   # the PLE conv's earlier rows

    # -- hyper-connections -----------------------------------------------------------------------------------------------
    def hc_rms(self, x, w):
        """x [N, HC, D] normed over D per stream, times the [HC, D] gamma (w: the tensor name)"""
        return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + self.eps) * self.vec(w).reshape(self.HC, self.D)

    def hc_mix(self, x, p, inject=True):
        """(the block input [N, D], the stream weights' logits [N, HC] or None) from the streams x [N, HC, D]; p: the tensor
        prefix ('blk.3.hc_attn', 'blk.3.hc_ffn', 'output_hc')"""
        N, HC = x.shape[0], self.HC
        xn = self.hc_rms(x, p + '_norm.weight'); f = xn.reshape(N, HC * self.D)
        self._see('hc down / inject input (hc_norm)', f)
        lo = silu((f @ self.W(p + '_down.weight').T) / HC)
        gate = (lo @ self.W(p + '_up.weight').T).reshape(N, HC, self.D)
        mixed = (xn * sigmoid(gate)).mean(axis=1)
        return mixed, (f @ self.W(p + '_inject.weight').T if inject else None)

    def hc_combine(self, x, y, inj):
        return x + y[:, None, :] * (2.0 * sigmoid(inj / self.HC))[:, :, None]

    # -- PLE -------------------------------------------------------------------------------------------------------------
    def ple_rows(self, p0, p1):
        """float64 [p1 - p0, n_heads * ple_dim]: the table rows of positions p0 .. p1 - 1 (heads slowest), read with pread"""
        ids = self.ngram.ids(self.toks, p0, p1)                              # [n, n_heads]
        t = self.F.T['per_layer_token_embd.weight']; _, rb = t.rows()
        u, inv = np.unique(ids, return_inverse=True)
        raw = np.concatenate([t.pread(int(r) * rb, (int(r) + 1) * rb) for r in u])
        rows = own_dequant.dequant(raw, t.type, len(u) * self.ple_dim).reshape(len(u), self.ple_dim)
        return rows[inv.reshape(-1)].reshape(ids.shape[0], -1).astype(np.float64)

    def _ple(self, il, x, p0):
        """the streams x [N, HC, D] of positions p0 .. p0 + N - 1 with layer il's PLE added"""
        gated, conv = self.ple_parts(il, x, p0)
        return x + gated + conv

    def ple_parts(self, il, x, p0):
        """(the gated value, silu(conv)), each [N, HC, D]; the conv's history moves on by N rows"""
        p, N, HC, D = f'blk.{il}.', x.shape[0], self.HC, self.D
        emb = self.ple_rows(p0, p0 + N)
        key = self.hc_rms((emb @ self.W(p + 'ple_key.weight').T).reshape(N, HC, D), p + 'ple_norm_key.weight')
        query = self.hc_rms(x, p + 'ple_norm_query.weight')
        s = (key * query).sum(axis=-1) / np.sqrt(D)                          # [N, HC]
        gated = (emb @ self.W(p + 'ple_value.weight').T)[:, None, :] * sigmoid(np.sign(s) * np.sqrt(np.maximum(np.abs(s), 1e-6)))[:, :, None]
        c = self.hc_rms(gated, p + 'ple_norm_conv.weight').reshape(N, HC * D)
        win = np.concatenate([self.ple_c, c])                                # the earlier rows, then this chunk's
        w = self.W(p + 'ple_conv1d.weight')                                  # [channels, taps]
        G, K = self.ngram.n_gram, self.ple_kern
        conv = sum(win[self.ple_hist - (K - 1 - k) * G:self.ple_hist - (K - 1 - k) * G + N] * w[:, k] for k in range(K))
        self.ple_c = win[N:].copy()
        return gated, silu(conv).reshape(N, HC, D)

    # -- a chunk of tokens -----------------------------------------------------------------------------------------------
    def feed(self, ids, last=None, trace=None, log=None, routes=None):
        """the float64 logits [M, vocab] after the last M (default: all) of these tokens, which follow every token fed
        before; trace: a list that gets each layer's output streams [N, HC, D]; routes: a list that gets each layer's
        (chosen experts [N, KU], their weights [N, KU])"""
        ids = [int(t) for t in ids]; N = len(ids); p0 = self.pos
        if self.limit is not None and p0 + N > self.limit:
            raise ValueError(f'ref_coder: positions {p0} .. {p0 + N - 1}: dense attention is the model\'s QSA only below '
                             f'{self.limit} (top_k + ratio - 1); past it the reference needs the indexer (the depth reference)')
        self.toks += ids
        emb = self.F.T['token_embd.weight']
        x = np.concatenate([own_dequant.rows(emb, t, t + 1) for t in ids]).astype(np.float64)
        x = np.repeat(x[:, None, :], self.HC, axis=1)                       # [N, HC, D]
        pos = p0 + np.arange(N, dtype=np.float64)
        for il in range(self.L):
            p = f'blk.{il}.'
            if il in self.ple_layers: x = self._ple(il, x, p0)
            h, inj = self.hc_mix(x, p + 'hc_attn'); self._see('block input (hc_mixed)', h)
            x = self.hc_combine(x, self._gdn(il, h) if self.gdn[il] else self._attn(il, h, pos), inj)
            h, inj = self.hc_mix(x, p + 'hc_ffn'); self._see('FFN input (hc_mixed)', h)
            y, top, w = self._moe(il, h)
            if routes is not None: routes.append((top, w))
            x = self.hc_combine(x, y, inj); self._see('residual streams', x)
            if trace is not None: trace.append(x.copy())
            if log: log(il)
        self.pos += N
        M = N if last is None else min(N, int(last))
        hN, _ = self.hc_mix(x[N - M:], 'output_hc', inject=False); self._see('head input (output_hc mix)', hN)
        R, _ = self.F.T['output.weight'].rows()
        return np.concatenate([hN @ self.W('output.weight', r0, min(R, r0 + HEAD_ROWS)).T for r0 in range(0, R, HEAD_ROWS)], axis=1)


def main(argv=None):
    from caminus.bpe import BPE
    ap = argparse.ArgumentParser(description="float64 host forward of the Coder (qwen4exp GGUF) on Caminus's own reader and decoders")
    ap.add_argument('gguf'); ap.add_argument('prompt'); ap.add_argument('--new', type=int, default=2)
    ap.add_argument('--threads', type=int, default=None)
    A = ap.parse_args(argv)
    m = CoderRef(A.gguf, threads=A.threads); tk = BPE(m.F.meta)
    ids = tk.encode(A.prompt)
    print(f'prompt {A.prompt!r} -> {len(ids)} tokens {ids}; {m.L} layers ({sum(m.gdn)} Gated DeltaNet, {m.L - sum(m.gdn)} attention)', flush=True)
    t0 = time.time(); out = []; chunk = ids
    for i in range(A.new):
        lg = m.feed(chunk, last=1)[0]
        pr = np.exp(lg - lg.max()); pr /= pr.sum()
        top = np.argsort(-lg)[:5]
        print(f'  next token {i}: ' + ', '.join(f'{tk.decode([int(j)])!r} {pr[j]:.3f}' for j in top) +
              f'   ({time.time() - t0:.0f} s)', flush=True)
        nxt = int(top[0]); out.append(nxt); chunk = [nxt]
        if nxt == tk.eos: break
    print(f'greedy: {A.prompt!r} + {tk.decode(out)!r}')


if __name__ == '__main__':
    sys.exit(main())
