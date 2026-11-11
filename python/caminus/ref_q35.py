"""A float64 host forward of the dense qwen35 line (Qwen3.8-27B finetunes: Swift 1.5, ThinkingCap, AREX-2) on Caminus's own
reader (caminus.ggufio), decoders (caminus.own_dequant) and tokenizer (caminus.bpe): the test oracle of the own-kernel
engine and prefill for these models (exact weights, float64 activations: the 'fp64 reference mode' of
docs/specs/spec-own-quant.md).

Layer-major over a chunk of tokens: a dense model reads every weight for every token (16.5 GB for Swift's Q4_K_M), so a
token at a time would decode the file once per token. feed(ids) decodes each matrix once and applies it to the whole chunk;
the per-layer state (the KV cache of an attention block; the conv window and the delta state of a Gated DeltaNet block)
carries to the next feed. One matrix is in memory at a time (float64: 0.7 GB for a 17408 x 5120 FFN matrix), the head in
16K-row pieces. About 5 minutes per feed on 16 threads, whatever the chunk's length up to a few hundred tokens.

The trunk is blocks 0 .. block_count - nextn_predict_layers - 1; a multi-token-prediction block after it (Swift's blk.64)
is not part of the model's forward and is not read here. A block is Gated DeltaNet where attention.recurrent_layers says
so (without that key: every block but each full_attention_interval-th).

The math (x the residual stream [N, D], float64; rms(y) = y / sqrt(mean(y^2) + eps); per block):
  h = rms(x) * attn_norm
  Gated DeltaNet block (16 key heads, 48 value heads, 128 wide):
    c = silu(conv4(Wqkv h))                 a causal 4-tap filter per channel over the last 4 tokens (tap 3 = this token)
    q | k | v = c's channels [0, 2048) | [2048, 4096) | [4096, 10240);  q, k: per head y / sqrt(sum(y^2) + eps)
    beta = sigmoid(Wbeta h);  g = softplus(Walpha h + dt) * a          (one of each per value head)
    value head j reads key head j % 16:  S_j = S_j * exp(g_j);  S_j += outer((v_j - S_j k) * beta_j, k);
    o_j = S_j (q / sqrt(128));  x += Wout ((rms(o_j) * ssm_norm) * silu(z_j)),  z = Wgate h
  attention block (24 query heads, 4 KV heads, 256 wide; a query head's rows of Wq are [q (256) | gate (256)]):
    q, k: per head rms(.) * q_norm / k_norm, then RoPE on the first rope.dimension_count dims (pairs i, i + n/2;
    theta_i = pos * base^(-2 i / n));  softmax(q . k / sqrt(256)) over the positions so far, 6 query heads per KV head
    x += Wo (attention * sigmoid(gate))
  h = rms(x) * post_attention_norm;  x += Wdown (silu(Wgate h) * (Wup h))
  logits = Whead (rms(x) * output_norm)

  python -I -m caminus.ref_q35 GGUF "prompt" [--new 2] [--threads 16]"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from caminus import ggufio, own_dequant
from caminus.bpe import BPE

HEAD_ROWS = 16384            # the head is decoded and applied in pieces of this many rows
DECODE_VALUES = 1 << 22      # a decode task: about this many values (numpy releases the lock: tasks run in parallel)


def silu(y):
    return y / (1.0 + np.exp(-y))


def sigmoid(y):
    return 1.0 / (1.0 + np.exp(-y))


def softplus(y):
    return np.logaddexp(0.0, y)


def trunk_blocks(F):
    """the number of blocks of the model's forward: block_count less the multi-token-prediction blocks after them"""
    return int(F.hp('block_count')) - int(F.hp('nextn_predict_layers') or 0)


class Q35Ref:
    def __init__(self, path, threads=None):
        self.F = F = ggufio.GGUFFile(path)
        if F.arch != 'qwen35': raise ValueError(f'ref_q35: architecture {F.arch!r} (qwen35 only)')
        if any(n.endswith('ffn_gate_inp.weight') for n in F.T): raise ValueError('ref_q35: a router tensor: not the dense line')
        self.L, self.D = trunk_blocks(F), int(F.hp('embedding_length'))
        self.H, self.HKV, self.HD = int(F.hp('attention.head_count')), int(F.hp('attention.head_count_kv')), int(F.hp('attention.key_length'))
        self.NK, self.NV, self.S, self.DC = int(F.hp('ssm.group_count')), int(F.hp('ssm.time_step_rank')), int(F.hp('ssm.state_size')), int(F.hp('ssm.conv_kernel'))
        self.eps, self.base, self.nrot = float(F.hp('attention.layer_norm_rms_epsilon')), float(F.hp('rope.freq_base')), int(F.hp('rope.dimension_count'))
        rec = F.hp('attention.recurrent_layers'); fai = int(F.hp('full_attention_interval', 4))
        self.gdn = [bool(rec[i]) if rec is not None else (i + 1) % fai != 0 for i in range(self.L)]
        if int(F.hp('ssm.inner_size')) != self.NV * self.S or self.NV % self.NK or self.H % self.HKV or self.nrot > self.HD or self.nrot % 2:
            raise ValueError('ref_q35: head counts / sizes this reference does not read')
        sec = F.hp('rope.dimension_sections')
        if sec is not None and 2 * sum(int(v) for v in sec) != self.nrot:
            raise ValueError(f'ref_q35: rope sections {list(sec)} do not cover the {self.nrot} rotated dims')
        self.pool = ThreadPoolExecutor(threads or min(16, os.cpu_count() or 1))
        self.census = None   # a dict: gets the largest |value| of each matvec input kind (and of the residual stream)
        self.reset()

    def reset(self):
        self.pos = 0
        self.kc = [None] * self.L; self.vc = [None] * self.L                    # attention: [pos, HKV, HD]
        self.conv = [np.zeros((self.DC - 1, 2 * self.NK * self.S + self.NV * self.S)) if g else None for g in self.gdn]
        self.state = [np.zeros((self.NV, self.S, self.S)) if g else None for g in self.gdn]   # S[j][value dim][key dim]

    # -- weights ---------------------------------------------------------------------------------------------------------
    def W(self, name, r0=0, r1=None):
        """float64 [rows, K] of a matrix's rows r0 .. r1 - 1 (decoded in parallel pieces)"""
        t = self.F.T[name]; R, _ = t.rows(); K = t.shape[0]
        r1 = R if r1 is None else r1
        out = np.empty((r1 - r0, K))
        step = max(1, DECODE_VALUES // K)

        def piece(a):
            out[a - r0:min(r1, a + step) - r0] = own_dequant.rows(t, a, min(r1, a + step))
        list(self.pool.map(piece, range(r0, r1, step)))
        return out

    def _see(self, kind, a):
        if self.census is not None: self.census[kind] = max(self.census.get(kind, 0.0), float(np.abs(a).max()))

    def vec(self, name):
        return self.W(name).reshape(-1)

    def rms(self, x, w):
        return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + self.eps) * w

    def rope(self, x, pos):
        """x [N, heads, HD] at positions pos [N]: the first nrot dims rotate (pairs i, i + nrot / 2), the rest stay"""
        half = self.nrot // 2
        th = pos[:, None, None] * self.base ** (-2.0 * np.arange(half) / self.nrot)
        c, s = np.cos(th), np.sin(th)
        a, b = x[..., :half], x[..., half:self.nrot]
        return np.concatenate([a * c - b * s, a * s + b * c, x[..., self.nrot:]], axis=-1)

    # -- the two block kinds ---------------------------------------------------------------------------------------------
    def _gdn(self, il, h):
        p, N, NK, NV, S = f'blk.{il}.', h.shape[0], self.NK, self.NV, self.S
        mix = h @ self.W(p + 'attn_qkv.weight').T                                # [N, 2 NK S + NV S]
        win = np.concatenate([self.conv[il], mix])                               # the last DC - 1 inputs, then this chunk's
        cw = self.W(p + 'ssm_conv1d.weight')                                     # [channels, DC]
        c = silu(sum(win[j:j + N] * cw[:, j] for j in range(self.DC)))
        self.conv[il] = win[N:].copy()
        l2 = lambda y: y / np.sqrt((y * y).sum(axis=-1, keepdims=True) + self.eps)
        q = l2(c[:, :NK * S].reshape(N, NK, S)); k = l2(c[:, NK * S:2 * NK * S].reshape(N, NK, S))
        v = c[:, 2 * NK * S:].reshape(N, NV, S)
        beta = sigmoid(h @ self.W(p + 'ssm_beta.weight').T)                      # [N, NV]
        g = softplus(h @ self.W(p + 'ssm_alpha.weight').T + self.vec(p + 'ssm_dt.bias')) * self.vec(p + 'ssm_a')
        kh = np.arange(NV) % NK                                                  # value head j reads key head j % NK
        St = self.state[il]; o = np.empty((N, NV, S))
        for n in range(N):
            kd, qd = k[n, kh], q[n, kh] / np.sqrt(S)
            St = St * np.exp(g[n])[:, None, None]
            d = (v[n] - np.einsum('jvk,jk->jv', St, kd)) * beta[n][:, None]
            St = St + np.einsum('jv,jk->jvk', d, kd)
            o[n] = np.einsum('jvk,jk->jv', St, qd)
        self.state[il] = St
        z = (h @ self.W(p + 'attn_gate.weight').T).reshape(N, NV, S)
        fin = self.rms(o, self.vec(p + 'ssm_norm.weight')) * silu(z)
        self._see('ssm_out input', fin)
        return fin.reshape(N, NV * S) @ self.W(p + 'ssm_out.weight').T

    def _attn(self, il, h, pos):
        p, N, H, HKV, HD = f'blk.{il}.', h.shape[0], self.H, self.HKV, self.HD
        qf = (h @ self.W(p + 'attn_q.weight').T).reshape(N, H, 2 * HD)           # a head's rows: [q | gate]
        q = self.rope(self.rms(qf[..., :HD], self.vec(p + 'attn_q_norm.weight')), pos)
        gate = qf[..., HD:]
        k = self.rope(self.rms((h @ self.W(p + 'attn_k.weight').T).reshape(N, HKV, HD), self.vec(p + 'attn_k_norm.weight')), pos)
        v = (h @ self.W(p + 'attn_v.weight').T).reshape(N, HKV, HD)
        K = k if self.kc[il] is None else np.concatenate([self.kc[il], k])
        V = v if self.vc[il] is None else np.concatenate([self.vc[il], v])
        self.kc[il], self.vc[il] = K, V
        P = K.shape[0] - N                                                       # positions before this chunk
        seen = np.arange(K.shape[0])[None, :] <= (P + np.arange(N))[:, None]     # [N, positions]: causal
        G = H // HKV; att = np.empty((N, H, HD))
        for kv in range(HKV):
            sc = np.einsum('nhd,td->nht', q[:, kv * G:(kv + 1) * G], K[:, kv]) / np.sqrt(HD)
            sc = np.where(seen[:, None, :], sc, -np.inf)
            pr = np.exp(sc - sc.max(axis=-1, keepdims=True)); pr /= pr.sum(axis=-1, keepdims=True)
            att[:, kv * G:(kv + 1) * G] = np.einsum('nht,td->nhd', pr, V[:, kv])
        att *= sigmoid(gate); self._see('attn_output input', att)
        return att.reshape(N, H * HD) @ self.W(p + 'attn_output.weight').T

    # -- a chunk of tokens -----------------------------------------------------------------------------------------------
    def feed(self, ids, last=None, trace=None, log=None):
        """the float64 logits [M, vocab] after the last M (default: all) of these tokens, which follow every token fed
        before; trace: a list that gets each block's output residual [N, D]"""
        ids = [int(t) for t in ids]; N = len(ids)
        emb = self.F.T['token_embd.weight']
        x = np.concatenate([own_dequant.rows(emb, t, t + 1) for t in ids]).astype(np.float64)
        pos = self.pos + np.arange(N, dtype=np.float64)
        for il in range(self.L):
            p = f'blk.{il}.'
            h = self.rms(x, self.vec(p + 'attn_norm.weight')); self._see('block input (attn_norm)', h)
            x = x + (self._gdn(il, h) if self.gdn[il] else self._attn(il, h, pos))
            h = self.rms(x, self.vec(p + 'post_attention_norm.weight')); self._see('FFN input (post_attention_norm)', h)
            a = silu(h @ self.W(p + 'ffn_gate.weight').T)
            a *= h @ self.W(p + 'ffn_up.weight').T; self._see('ffn_down input', a)
            x = x + a @ self.W(p + 'ffn_down.weight').T; self._see('residual stream', x)
            if trace is not None: trace.append(x.copy())
            if log: log(il)
        self.pos += N
        M = N if last is None else min(N, int(last))
        hN = self.rms(x[N - M:], self.vec('output_norm.weight')); self._see('head input (output_norm)', hN)
        R, _ = self.F.T['output.weight'].rows()
        return np.concatenate([hN @ self.W('output.weight', r0, min(R, r0 + HEAD_ROWS)).T for r0 in range(0, R, HEAD_ROWS)], axis=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description="float64 host forward of a dense qwen35 GGUF on Caminus's own reader, decoders and tokenizer")
    ap.add_argument('gguf'); ap.add_argument('prompt'); ap.add_argument('--new', type=int, default=2)
    ap.add_argument('--threads', type=int, default=None)
    A = ap.parse_args(argv)
    m = Q35Ref(A.gguf, threads=A.threads); tk = BPE(m.F.meta)
    ids = tk.encode(A.prompt)
    print(f'prompt {A.prompt!r} -> {len(ids)} tokens {ids}; {m.L} blocks ({sum(m.gdn)} Gated DeltaNet, {m.L - sum(m.gdn)} attention)', flush=True)
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
