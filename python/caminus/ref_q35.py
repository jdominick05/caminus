"""A float64 host forward of the qwen35 line on Caminus's own reader (caminus.ggufio), decoders (caminus.own_dequant) and
tokenizer (caminus.bpe): the dense models (architecture qwen35: the Qwen3.8-27B finetunes Swift 1.5, ThinkingCap, AREX-2)
and the MoE one (qwen35moe: Qwen3.6-35B-A3B). The test oracle of the own-kernel engine and prefill for these models (exact
weights, float64 activations: the 'fp64 reference mode' of docs/specs/spec-own-quant.md).

Layer-major over a chunk of tokens: a dense model reads every weight for every token (16.5 GB for Swift's Q4_K_M), so a
token at a time would decode the file once per token. feed(ids) decodes each matrix once and applies it to the whole chunk;
the per-layer state (the KV cache of an attention block; the conv window and the delta state of a Gated DeltaNet block)
carries to the next feed. One matrix is in memory at a time (float64: 0.7 GB for a 17408 x 5120 FFN matrix), the head in
16K-row pieces. About 5 minutes per feed on 16 threads, whatever the chunk's length up to a few hundred tokens. The MoE
FFN never decodes a whole expert tensor (256 experts: 2.1 GB as float64): it decodes one expert's rows at a time, for the
experts the chunk's tokens picked, and applies them to those tokens only (Qwen3.6-35B-A3B's 2-bit GSQ file: 128 positions
in 30 s, under 1 GB of anonymous memory).

The trunk is blocks 0 .. block_count - nextn_predict_layers - 1; a multi-token-prediction block after it (Swift's blk.64)
is not part of the model's forward and is not read here. A block is Gated DeltaNet where attention.recurrent_layers says
so (without that key: every block but each full_attention_interval-th).

The math (x the residual stream [N, D], float64; rms(y) = y / sqrt(mean(y^2) + eps); per block; the head counts and
channel ranges are Swift's, the code reads them from the header: Qwen3.6-35B-A3B has D 2048, 32 value heads and 16
query heads on 2 KV heads, 8 per KV head):
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
    qwen35moe (256 experts, 8 used, one shared expert):  p = softmax(Wr h) over the experts;  the 8 largest p (ties: the
    lower id), weights w = p / sum(p of the 8);  x += sum_e w_e Wdown_e (silu(Wgate_e h) * (Wup_e h))
      + sigmoid(Wsg . h) Wdown_s (silu(Wgate_s h) * (Wup_s h))     (Wsg: ffn_gate_inp_shexp, one row)
  logits = Whead (rms(x) * output_norm)

  python -I -m caminus.ref_q35 GGUF "prompt" [--new 2] [--threads 16]     (a qwen4exp file: caminus.ref_coder's forward)"""
import argparse
import os
import re
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


def rope_sections_ok(sec, nrot):
    """does IMRoPE with these sections rotate every rotated pair of a text position? llama.cpp's rope_multi (imrope) gives
    pair s (s < nrot / 2) section s % 3 only while s < 3 * sections[s % 3], else the fourth position component, which a text
    token holds as 0 (pair s then does not turn); the three sections' positions are a text token's own position, so a file
    whose sections cover every pair is NEOX RoPE on text (what rope() computes), and one that leaves a pair out is not"""
    sec = [int(v) for v in sec]
    return 2 * sum(sec) == nrot and all(s < 3 * sec[s % 3] for s in range(nrot // 2))


class Q35Ref:
    gdn_gate = staticmethod(silu)   # the Gated DeltaNet output gate on z (the Coder's reference, caminus.ref_coder: sigmoid)
    UNREAD = r'(\.scale|_s(\.weight)?|ffn_gate_up_exps(\.weight)?)$'   # tensor names of forms no reference here reads
    yarn_m, yarn_s = None, 1.0      # (the native rope; __init__ sets YaRN's per-pair multipliers and cos / sin magnitude)

    def __init__(self, path, threads=None, yarn=None):
        self.yarn = yarn          # (the long-context factor; _trunk builds the rope table from it)
        self.F = F = ggufio.GGUFFile(path)
        if F.arch not in ('qwen35', 'qwen35moe'): raise ValueError(f'ref_q35: architecture {F.arch!r} (qwen35 and qwen35moe only)')
        self.moe = F.arch == 'qwen35moe'
        if not self.moe and any(n.endswith('ffn_gate_inp.weight') for n in F.T):
            raise ValueError('ref_q35: a router tensor in a qwen35 file: not the dense line')
        self._trunk('ref_q35', trunk_blocks(F))
        if self.moe: self._experts('ref_q35')
        self.pool = ThreadPoolExecutor(threads or min(16, os.cpu_count() or 1))
        self.census = None   # a dict: gets the largest |value| of each matvec input kind (and of the residual stream)
        self.reset()

    def _trunk(self, tag, L):
        """the trunk's header (L blocks): the sizes, the block kinds and the RoPE; refuses forms this reference does not read
        (tag: the error messages' module)"""
        F, hp = self.F, self.F.hp
        unread = sorted(n for n in F.T if re.search(self.UNREAD, n))
        if unread:   # (scale tensors, fused gate|up experts; the Coder's subclass adds output_norm)
            raise ValueError(f'{tag}: tensors this reference does not read: {unread[:4]}{" ..." if len(unread) > 4 else ""}')
        self.L, self.D = L, int(hp('embedding_length'))
        self.H, self.HKV, self.HD = int(hp('attention.head_count')), int(hp('attention.head_count_kv')), int(hp('attention.key_length'))
        self.NK, self.NV, self.S, self.DC = int(hp('ssm.group_count')), int(hp('ssm.time_step_rank')), int(hp('ssm.state_size')), int(hp('ssm.conv_kernel'))
        self.eps, self.base, self.nrot = float(hp('attention.layer_norm_rms_epsilon')), float(hp('rope.freq_base')), int(hp('rope.dimension_count'))
        # (YaRN: the same recipe and the same native context the engine uses, so both sides of a gate row rotate alike.
        #  getattr: a subclass (the Coder's CoderRef) sets up its own trunk and need not take the factor at all)
        yarn = getattr(self, 'yarn', None)
        if yarn:
            from caminus.rope import mscale, yarn_freqs
            self.yarn_s = mscale(float(yarn))   # (the magnitude on cos / sin of the rotated dims: the engine's op scales both)
            j = np.arange(self.nrot // 2, dtype=np.float64)
            n_ctx = int(hp('context_length'))
            self.yarn_m = (yarn_freqs(self.nrot, self.base, float(yarn), n_ctx)
                           / self.base ** (-2.0 * j / self.nrot)).astype(np.float64)
        rec = hp('attention.recurrent_layers'); fai = int(hp('full_attention_interval', 4))
        self.gdn = [bool(rec[i]) if rec is not None else (i + 1) % fai != 0 for i in range(self.L)]
        if int(hp('ssm.inner_size')) != self.NV * self.S or self.NV % self.NK or self.H % self.HKV or self.nrot > self.HD or self.nrot % 2:
            raise ValueError(f'{tag}: head counts / sizes this reference does not read')
        if int(hp('attention.value_length', self.HD)) != self.HD: raise ValueError(f'{tag}: value_length != key_length')
        sec = hp('rope.dimension_sections')
        if sec is not None and not rope_sections_ok(sec, self.nrot):
            raise ValueError(f'{tag}: rope sections {[int(v) for v in sec]} do not rotate all {self.nrot // 2} pairs of a text position')
        if any(k.startswith(f'{F.arch}.rope.scaling') for k in F.meta): raise ValueError(f'{tag}: rope.scaling keys (not read)')

    def _experts(self, tag):
        """the MoE FFN's header (Q35Ref._moe reads E, KU, FF, FFS): one count and width for every block, a router in each"""
        F, hp = self.F, self.F.hp
        if any(np.ndim(hp(k)) for k in ('expert_count', 'expert_used_count', 'expert_feed_forward_length', 'expert_shared_feed_forward_length')):
            raise ValueError(f'{tag}: per-block expert counts or widths (an array-valued expert key)')
        routers = sum(f'blk.{i}.ffn_gate_inp.weight' in F.T for i in range(self.L))   # (the trunk's: not an MTP block's)
        self.E, self.KU = int(hp('expert_count')), int(hp('expert_used_count'))
        self.FF, self.FFS = int(hp('expert_feed_forward_length')), int(hp('expert_shared_feed_forward_length'))
        if routers != self.L or not 0 < self.KU <= self.E:
            raise ValueError(f'{tag}: {routers} router tensors in {self.L} trunk blocks, {self.KU} of {self.E} experts')

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
        if self.yarn_m is not None: th = th * self.yarn_m
        c, s = np.cos(th) * self.yarn_s, np.sin(th) * self.yarn_s
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
        fin = self.rms(o, self.vec(p + 'ssm_norm.weight')) * self.gdn_gate(z)
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

    # -- the two FFN kinds ------------------------------------------------------------------------------------------------
    def _ffn(self, il, h):
        p = f'blk.{il}.'
        a = silu(h @ self.W(p + 'ffn_gate.weight').T)
        a *= h @ self.W(p + 'ffn_up.weight').T; self._see('ffn_down input', a)
        return a @ self.W(p + 'ffn_down.weight').T

    def _moe(self, il, h):
        """(the MoE FFN's output [N, D], the chosen experts [N, KU], their weights [N, KU])"""
        p, N, FF, D = f'blk.{il}.', h.shape[0], self.FF, self.D
        lg = h @ self.W(p + 'ffn_gate_inp.weight').T                              # [N, E]
        pr = np.exp(lg - lg.max(axis=1, keepdims=True)); pr /= pr.sum(axis=1, keepdims=True)
        top = np.argsort(-pr, axis=1, kind='stable')[:, :self.KU]                 # (stable: a tie keeps the lower id)
        w = np.take_along_axis(pr, top, axis=1); w /= w.sum(axis=1, keepdims=True)
        T = {k: self.F.T[p + f'ffn_{k}_exps.weight'] for k in ('gate', 'up', 'down')}

        def expert(e):          # one expert's rows (8 MB of float64 a matrix), on the tokens that picked it
            n, slot = np.nonzero(top == e)
            g = h[n] @ own_dequant.rows(T['gate'], e * FF, (e + 1) * FF).astype(np.float64).T
            a = silu(g) * (h[n] @ own_dequant.rows(T['up'], e * FF, (e + 1) * FF).astype(np.float64).T)
            y = a @ own_dequant.rows(T['down'], e * D, (e + 1) * D).astype(np.float64).T
            return n, w[n, slot][:, None] * y, float(np.abs(a).max())

        out = np.zeros((N, D))
        for n, y, amax in self.pool.map(expert, np.unique(top)):                  # (in expert order: the same sum every run)
            out[n] += y
            if self.census is not None: self._see('expert ffn_down input', np.array(amax))
        a = silu(h @ self.W(p + 'ffn_gate_shexp.weight').T)
        a *= h @ self.W(p + 'ffn_up_shexp.weight').T; self._see('shared ffn_down input', a)
        out += sigmoid(h @ self.vec(p + 'ffn_gate_inp_shexp.weight'))[:, None] * (a @ self.W(p + 'ffn_down_shexp.weight').T)
        return out, top, w

    # -- a chunk of tokens -----------------------------------------------------------------------------------------------
    def feed(self, ids, last=None, trace=None, log=None, routes=None):
        """the float64 logits [M, vocab] after the last M (default: all) of these tokens, which follow every token fed
        before; trace: a list that gets each block's output residual [N, D]; routes (qwen35moe): a list that gets each
        block's (chosen experts [N, KU], their weights [N, KU])"""
        ids = [int(t) for t in ids]; N = len(ids)
        emb = self.F.T['token_embd.weight']
        x = np.concatenate([own_dequant.rows(emb, t, t + 1) for t in ids]).astype(np.float64)
        pos = self.pos + np.arange(N, dtype=np.float64)
        for il in range(self.L):
            p = f'blk.{il}.'
            h = self.rms(x, self.vec(p + 'attn_norm.weight')); self._see('block input (attn_norm)', h)
            x = x + (self._gdn(il, h) if self.gdn[il] else self._attn(il, h, pos))
            h = self.rms(x, self.vec(p + 'post_attention_norm.weight')); self._see('FFN input (post_attention_norm)', h)
            if self.moe:
                y, top, w = self._moe(il, h)
                if routes is not None: routes.append((top, w))
            else:
                y = self._ffn(il, h)
            x = x + y; self._see('residual stream', x)
            if trace is not None: trace.append(x.copy())
            if log: log(il)
        self.pos += N
        M = N if last is None else min(N, int(last))
        hN = self.rms(x[N - M:], self.vec('output_norm.weight')); self._see('head input (output_norm)', hN)
        return self.head(hN)

    def head(self, hN):
        """the logits [M, vocab] of the head's inputs hN [M, D] (output.weight in HEAD_ROWS pieces)"""
        R, _ = self.F.T['output.weight'].rows()
        return np.concatenate([hN @ self.W('output.weight', r0, min(R, r0 + HEAD_ROWS)).T for r0 in range(0, R, HEAD_ROWS)], axis=1)


def reference(path, threads=None, yarn=None):
    """the float64 reference of a GGUF, by its architecture: Q35Ref (qwen35, qwen35moe) or caminus.ref_coder.CoderRef
    (qwen4exp, the Coder)"""
    F = ggufio.GGUFFile(path); arch = F.arch; F.close()
    if arch == 'qwen4exp':
        from caminus.ref_coder import CoderRef
        return CoderRef(path, threads=threads)
    return Q35Ref(path, threads=threads, yarn=yarn)


def main(argv=None):
    ap = argparse.ArgumentParser(description="float64 host forward of a qwen35 / qwen35moe / qwen4exp GGUF on Caminus's own reader, decoders and tokenizer")
    ap.add_argument('gguf'); ap.add_argument('prompt'); ap.add_argument('--new', type=int, default=2)
    ap.add_argument('--threads', type=int, default=None)
    A = ap.parse_args(argv)
    m = reference(A.gguf, threads=A.threads); tk = BPE(m.F.meta)
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
