"""A float64 host forward of qwen3moe (Qwen3-30B-A3B) on Caminus's own reader (caminus.ggufio), decoders
(caminus.own_dequant) and tokenizer (caminus.bpe): the GPU engine's test oracle (exact weights, float64 activations: the
'fp64 reference mode' of docs/specs/spec-own-quant.md) and the first token from the GGUF file on agent-written code only.
Slow by design (weights decoded per use; ~10 s per token on the CPU) and light on memory (one layer's matrices at a time,
the head in 16K-row chunks).

The math (per layer; x the residual stream, float64):
  h = rms(x) * attn_norm;  q = Wq h (32 heads x 128), k = Wk h, v = Wv h (4 heads x 128)
  q, k: per head rms(.) * q_norm / k_norm;  RoPE NEOX (pairs i, i + 64; theta_i = base^(-2 i / 128)), base 1e6
  attention: softmax(q . k / sqrt(128)) over positions <= the token's, 8 query heads per kv head;  x += Wo (heads)
  h = rms(x) * ffn_norm;  p = softmax(Wr h) over 128 experts;  top 8 by p, weights p / sum(p of the 8)
  x += sum_e w_e Wdown_e (silu(Wgate_e h) * Wup_e h)
  logits = Whead (rms(x) * output_norm)        rms(y) = y / sqrt(mean(y^2) + eps)
yarn=(factor, n_ctx_orig): the long-context mode's RoPE, written from llama.cpp's statements (ggml rope_yarn and
ggml_rope_yarn_corr_dims, beta_fast 32, beta_slow 1) in float64, on purpose not through caminus.rope (the engine's table
builder), so the gate can catch a wrong table: theta_extrap = pos base^(-2 j / 128), theta_interp = theta_extrap / factor,
theta = theta_interp (1 - ramp_j) + theta_extrap ramp_j, cos and sin times 1 + 0.1 ln(factor).

  python -I -m caminus.ref_q3moe GGUF "prompt" [--new 4] [--yarn 4:32768]"""
import argparse
import math
import sys
import time

import numpy as np

from caminus import ggufio, own_dequant
from caminus.bpe import BPE


class Q3MoERef:
    def __init__(self, path, yarn=None):
        self.yarn = None if yarn is None else (float(yarn[0]), int(yarn[1]))
        self.F = F = ggufio.GGUFFile(path)
        if F.arch != 'qwen3moe': raise ValueError(f'ref_q3moe: architecture {F.arch!r} (qwen3moe only)')
        self.L, self.D = int(F.hp('block_count')), int(F.hp('embedding_length'))
        self.H, self.HKV, self.HD = int(F.hp('attention.head_count')), int(F.hp('attention.head_count_kv')), int(F.hp('attention.key_length'))
        self.E, self.KU, self.FF = int(F.hp('expert_count')), int(F.hp('expert_used_count')), int(F.hp('expert_feed_forward_length'))
        self.eps, self.base = float(F.hp('attention.layer_norm_rms_epsilon')), float(F.hp('rope.freq_base'))
        self.k_cache = [[] for _ in range(self.L)]; self.v_cache = [[] for _ in range(self.L)]
        self.pos = 0

    # -- weights ---------------------------------------------------------------------------------------------------------
    def W(self, name, r0=None, r1=None):
        t = self.F.T[name]; R, _ = t.rows()
        return own_dequant.rows(t, r0 or 0, R if r1 is None else r1).astype(np.float64)

    def vec(self, name):
        return self.W(name).reshape(-1)

    def rms(self, x, w):
        return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + self.eps) * w

    def yarn_angles(self, pos):
        """(theta [64], mscale) of llama.cpp's rope_yarn at position pos (float64; the statement form, see the module doc)"""
        factor, n_orig = self.yarn; n_dims, base = self.HD, self.base
        corr_dim = lambda n_rot: n_dims * math.log(n_orig / (n_rot * 2 * math.pi)) / (2 * math.log(base))
        lo, hi = max(0.0, math.floor(corr_dim(32.0))), min(n_dims - 1.0, math.ceil(corr_dim(1.0)))
        i0 = np.arange(0, n_dims, 2)                                   # (llama.cpp's i0; i0 / 2 is the pair)
        theta_extrap = pos * base ** (-i0 / n_dims)
        freq_scale = 1.0 / factor
        theta_interp = freq_scale * theta_extrap
        ramp_mix = 1.0 - np.minimum(1.0, np.maximum(0.0, (i0 // 2 - lo) / max(0.001, hi - lo)))
        theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix
        return theta, 1.0 + 0.1 * math.log(1.0 / freq_scale)

    def rope(self, x, pos):
        """x [..., heads, 128] at position pos: NEOX pairs (i, i + 64); YaRN's angles and factor in the long-context mode"""
        half = self.HD // 2
        if self.yarn is None:
            th = pos * self.base ** (-2.0 * np.arange(half) / self.HD)
            c, s = np.cos(th), np.sin(th)
        else:
            th, ms = self.yarn_angles(pos)
            c, s = np.cos(th) * ms, np.sin(th) * ms
        a, b = x[..., :half], x[..., half:]
        return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)

    # -- one token ---------------------------------------------------------------------------------------------------------
    def step(self, tok, trace=None):
        """the next position's float64 logits [vocab] for token tok (the KV cache grows by one)"""
        x = self.W('token_embd.weight', tok, tok + 1).reshape(-1)
        pos = self.pos
        for il in range(self.L):
            p = f'blk.{il}.'
            h = self.rms(x, self.vec(p + 'attn_norm.weight'))
            q = (self.W(p + 'attn_q.weight') @ h).reshape(self.H, self.HD)
            k = (self.W(p + 'attn_k.weight') @ h).reshape(self.HKV, self.HD)
            v = (self.W(p + 'attn_v.weight') @ h).reshape(self.HKV, self.HD)
            q = self.rope(self.rms(q, self.vec(p + 'attn_q_norm.weight')), pos)
            k = self.rope(self.rms(k, self.vec(p + 'attn_k_norm.weight')), pos)
            self.k_cache[il].append(k); self.v_cache[il].append(v)
            K = np.stack(self.k_cache[il]); V = np.stack(self.v_cache[il])            # [pos + 1, HKV, HD]
            G = self.H // self.HKV
            att = np.empty((self.H, self.HD))
            for hh in range(self.H):
                kv = hh // G
                sc = K[:, kv] @ q[hh] / np.sqrt(self.HD)
                pr = np.exp(sc - sc.max()); pr /= pr.sum()
                att[hh] = pr @ V[:, kv]
            x = x + self.W(p + 'attn_output.weight') @ att.reshape(-1)
            h = self.rms(x, self.vec(p + 'ffn_norm.weight'))
            lg = self.W(p + 'ffn_gate_inp.weight') @ h
            pr = np.exp(lg - lg.max()); pr /= pr.sum()
            top = np.argsort(-pr, kind='stable')[:self.KU]
            w = pr[top] / pr[top].sum()
            moe = np.zeros(self.D)
            for e, we in zip(top, w):
                r0, r1 = e * self.FF, (e + 1) * self.FF
                g = self.W(p + 'ffn_gate_exps.weight', r0, r1) @ h
                u = self.W(p + 'ffn_up_exps.weight', r0, r1) @ h
                a = g / (1.0 + np.exp(-g)) * u
                moe += we * (self.W(p + 'ffn_down_exps.weight', e * self.D, (e + 1) * self.D) @ a)
            x = x + moe
            if trace is not None: trace.append((x.copy(), top.copy(), w.copy()))
        self.pos += 1
        hN = self.rms(x, self.vec('output_norm.weight'))
        t = self.F.T['output.weight']; R, _ = t.rows()
        return np.concatenate([self.W('output.weight', r0, min(R, r0 + 16384)) @ hN for r0 in range(0, R, 16384)])


def parse_yarn(v):
    """'FACTOR:ORIG_CTX' -> (factor, orig) or None"""
    if not v: return None
    f, o = v.split(':')
    return float(f), int(o)


def main(argv=None):
    ap = argparse.ArgumentParser(description='float64 host forward of qwen3moe on Caminus\'s own reader, decoders and tokenizer')
    ap.add_argument('gguf'); ap.add_argument('prompt'); ap.add_argument('--new', type=int, default=4)
    ap.add_argument('--yarn', default=None, help='FACTOR:ORIG_CTX (the long-context mode: 4:32768)')
    A = ap.parse_args(argv)
    m = Q3MoERef(A.gguf, yarn=parse_yarn(A.yarn)); tk = BPE(m.F.meta)
    ids = tk.encode(A.prompt)
    print(f'prompt {A.prompt!r} -> {len(ids)} tokens {ids}', flush=True)
    t0 = time.time()
    for t in ids: lg = m.step(t)
    out = []
    for i in range(A.new):
        pr = np.exp(lg - lg.max()); pr /= pr.sum()
        top = np.argsort(-lg)[:5]
        print(f'  next token {i}: ' + ', '.join(f'{tk.decode([int(j)])!r} {pr[j]:.3f}' for j in top) +
              f'   ({time.time() - t0:.0f} s)', flush=True)
        nxt = int(top[0]); out.append(nxt)
        if nxt == tk.eos: break
        lg = m.step(nxt)
    print(f'greedy: {A.prompt!r} + {tk.decode(out)!r}')


if __name__ == '__main__':
    sys.exit(main())
