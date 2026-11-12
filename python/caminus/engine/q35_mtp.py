"""The dense qwen35 line's multi-token-prediction drafter (Swift 1.5: blk.<n_layer>, all Q4_0) on the drafter module
(kernels/k3_q35_swift_mtp.hip), beside a Dense35Own trunk on the same card.

The block (llama.cpp's pairing, src/models/qwen35.cpp): MTP position j reads the trunk's normed final hidden state of
position j - 1 (h_{j-1}: output_norm of the last block's residual, Dense35Own's buf['h'] / Dense35Verify's buf['h'] rows)
and the token at position j:
    x  = eh_proj . [rms(embed(token_j)) * enorm ; rms(h_{j-1}) * hnorm]
    x += attention(rms(x) * attn_norm)          (gated, q / k norms, NEOX RoPE, the block's own KV cache)
    x += ffn(rms(x) * post_attention_norm)      (SwiGLU)
    hn = rms(x) * shared_head_norm;   logits = output . hn  (the trunk's head);   draft = argmax
and a chained draft step feeds hn as the next h with the draft as the next token.

Every block row written into the KV cache must come from the trunk's h and the accepted token, so a round is:
    rows p0 .. p0 + T - 1 (T = 1 .. 4: the rows accepted in the last verify pass, then the row of the newest token),
    each from (trunk h of the row before, token), in one launch (the catch tape), the last row's head and argmax giving
    draft 1; then k - 1 chained steps (one launch each, the token read on the device from the previous argmax).
A chained step writes its KV row too; the next round's catch tape writes those positions again from the trunk's h.
fill() writes rows from the trunk's h without drafting (the prompt), 4 rows per launch.

The T-row tapes run the trunk's T-token forms (the verify module's), so a row's bits are the same in a tape of any T.
Each launch is waited for in run order at the end of the round; a wait that gave up reruns the whole round."""
import os
import sys

import numpy as np
import torch

from . import k3
from .own_layout import q4_0_planes, q6k_planes
from ..device import capacity

MODULE = 'k3_q35_swift_mtp'
MAX_T = 4
NEXTN = ('nextn.eh_proj', 'attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down')


def block_of(F):
    """the multi-token-prediction block's index in a dense qwen35 GGUF (blk.<n_layer>), or None when the file has none"""
    n = int(F.hp('block_count'))
    nn = int(F.hp('nextn_predict_layers', 0) or 0)
    il = n - nn if nn else n
    return il if f'blk.{il}.nextn.eh_proj.weight' in F.T else None


class Mtp35:
    @staticmethod
    def need_mib(eng, k=3):
        """MiB this drafter allocates on the trunk's card: the block's weights, its KV cache at the trunk's max_ctx, the
        tapes' buffers"""
        F = eng.F; il = block_of(F)
        if il is None: raise ValueError(f'Mtp35: {os.path.basename(eng.path)} has no multi-token-prediction block')
        p = f'blk.{il}.'
        w = sum(t.n_bytes for n, t in F.T.items() if n.startswith(p))
        kv = (eng.max_ctx + MAX_T + k) * 2 * eng.nkv * eng.hd * 2
        D, QH, ffn = eng.D, eng.nh * eng.hd, eng.ffn
        buf = 4 * MAX_T * (6 * D + 2 * D + 3 * QH + 2 * QH + 4 * eng.nkv * eng.hd + 2 * ffn) + 2 * MAX_T * (2 * D + 2 * D + QH + ffn)
        return (w + kv + buf + 4 * eng.vocab) / 2 ** 20 + 4 * (k + 2 * MAX_T) + 8.0

    def __init__(self, eng, k=3, module=MODULE, per_wgp=None, vram_check=True):
        """eng: a Dense35Own on the Swift module (its card, max_ctx, head and embedding); k: drafts per round (1 .. 4)"""
        if module not in k3.Q35M_MODULES: raise ValueError(f'Mtp35: module {module!r} ({k3.Q35M_MODULES})')
        if eng.tcap or eng.module not in k3.Q35S_MODULES: raise ValueError(f'Mtp35: the Swift module only (the engine runs {eng.module})')
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 4: raise ValueError(f'Mtp35: k {k!r}: 1 .. 4')
        self.il = block_of(eng.F)
        if self.il is None: raise ValueError(f'Mtp35: {os.path.basename(eng.path)} has no multi-token-prediction block')
        for n in NEXTN:
            t = eng.F.T[f'blk.{self.il}.{n}.weight']
            if t.type != 'Q4_0': raise ValueError(f'Mtp35: blk.{self.il}.{n} is {t.type} (the drafter module runs Q4_0)')
        if eng.head['fmt'] != 'Q6_K': raise ValueError(f"Mtp35: the head is {eng.head['fmt']} (the drafter module runs Q6_K)")
        self.eng, self.k, self.module = eng, int(k), module
        self.mib = self.need_mib(eng, self.k)
        if vram_check: capacity.check([eng.dev], [self.mib], f'Mtp35 (k {self.k}, max_ctx {eng.max_ctx}: {self.mib:.0f} MiB)')
        self.per_wgp = int(per_wgp) if per_wgp else eng.per_wgp
        self.filled = 0      # block KV rows 1 .. filled - 1 hold rows from the trunk's h (row 0: zero)
        self.bails = []
        with torch.cuda.device(eng.dev): self._load(); self._build()

    # -- weights -----------------------------------------------------------------------------------------------------------
    def _q4(self, name, rows, cols):
        t = self.eng.F.T[name]; K = int(t.shape[0]); R, _ = t.rows()
        if (R, K) != (rows, cols): raise ValueError(f'Mtp35: {name} is [{R}, {K}], the tape needs [{rows}, {cols}]')
        b = q4_0_planes(t.pread(0, t.n_bytes), R, K)
        return dict(name=name, fmt='Q4_0', rows=R, K=K, w=torch.from_numpy(np.ascontiguousarray(b)).to('cuda'))

    def _load(self):
        e, p = self.eng, f'blk.{self.il}.'
        D, hd, nh, nkv, ffn = e.D, e.hd, e.nh, e.nkv, e.ffn
        self.W = dict(eh=self._q4(p + 'nextn.eh_proj.weight', D, 2 * D), q=self._q4(p + 'attn_q.weight', nh * 2 * hd, D),
                      k=self._q4(p + 'attn_k.weight', nkv * hd, D), v=self._q4(p + 'attn_v.weight', nkv * hd, D),
                      o=self._q4(p + 'attn_output.weight', D, nh * hd), fg=self._q4(p + 'ffn_gate.weight', ffn, D),
                      fu=self._q4(p + 'ffn_up.weight', ffn, D), fd=self._q4(p + 'ffn_down.weight', D, ffn))
        f = e._f32
        self.N = dict(enorm=f(p + 'nextn.enorm.weight', D), hnorm=f(p + 'nextn.hnorm.weight', D),
                      shn=f(p + 'nextn.shared_head_norm.weight', D), an=f(p + 'attn_norm.weight', D),
                      pn=f(p + 'post_attention_norm.weight', D), qn=f(p + 'attn_q_norm.weight', hd), kn=f(p + 'attn_k_norm.weight', hd))
        self.rows = e.max_ctx + MAX_T + self.k          # (the guard's span: a round's catch rows and chained rows)
        self.cache = torch.zeros(self.rows * 2 * nkv * hd, dtype=torch.float16, device='cuda')

    # -- tapes -------------------------------------------------------------------------------------------------------------
    def _build(self):
        e, T4, k = self.eng, MAX_T, self.k
        D, hd, nh, nkv, ffn, QH = e.D, e.hd, e.nh, e.nkv, e.ffn, e.nh * e.hd
        f32, f16 = torch.float32, torch.float16
        Z = lambda n, dt=f32: torch.zeros(n, dtype=dt, device='cuda')
        self.b = b = dict(hin=Z(T4 * D), e=Z(T4 * D), scr=Z(T4 * 2 * D), xe16=Z(T4 * 2 * D, f16), x=Z(T4 * D), h=Z(T4 * D),
                          h16=Z(T4 * D, f16), qf=Z(T4 * 2 * QH), kr=Z(T4 * nkv * hd), vr=Z(T4 * nkv * hd), qo=Z(T4 * QH),
                          ko=Z(T4 * nkv * hd), att=Z(T4 * QH), gat16=Z(T4 * QH, f16), blk=Z(T4 * D), g=Z(T4 * ffn),
                          u=Z(T4 * ffn), a16=Z(T4 * ffn, f16), ffp=Z(T4 * D), hn=Z(T4 * D), hn16=Z(T4 * D, f16),
                          logits=Z(e.vocab))
        # the stage, one block for every tape (one copy per round): tape slot i at [32 i, 32 i + 32): [0 .. 16) the
        # engine's pass parameters, [16 .. 20) the tokens, [20] the first position; then the k argmax keys (int64)
        self.slots = {}
        tapes = [('catch', T) for T in range(1, T4 + 1)] + [('fill', T) for T in range(1, T4 + 1)] + [('chain', i) for i in range(1, k)]
        for i, t in enumerate(tapes): self.slots[t] = i
        nst = 32 * len(tapes)
        self.key_at = (nst + 1) // 2 * 2
        self.hstage = torch.zeros(self.key_at + 2 * k, dtype=torch.int32, pin_memory=True)
        self.dstage = Z(self.key_at + 2 * k, torch.int32)
        self.keys = self.dstage[self.key_at:].view(torch.int64)
        self.h_keys = torch.zeros(k, dtype=torch.int64, pin_memory=True)
        self.E, self.names = {}, {}
        for t in tapes: self.E[t] = self._tape(*t)
        self.ev = torch.cuda.Event()
        self.h_st = {t: torch.zeros(k3.STATUS_WORDS, dtype=torch.int32, pin_memory=True) for t in tapes}

    def _tape(self, kind, n):
        """kind 'catch' / 'fill': n rows (1 .. 4) of the block from the trunk's h (hin rows 0 .. n - 1) and the staged
        tokens, at the staged positions; 'catch' then the last row's head and argmax (key 0). 'chain' n (1 .. k - 1): one
        row from the previous step's hn (hn row 3) and key n - 1's token, its head and argmax into key n"""
        e, b, W, N = self.eng, self.b, self.W, self.N
        D, hd, nh, nkv, ffn, QH = e.D, e.hd, e.nh, e.nkv, e.ffn, e.nh * e.hd
        T = n if kind != 'chain' else 1
        sl = self.slots[(kind, n)]
        st = self.dstage[32 * sl:32 * sl + 32]
        toks, pos = st[16:16 + T], st[20:21]
        E = k3.Engine(module=self.module, grid=k3.resident_grid(self.module, self.per_wgp), T=T, limit_us=e.limit_us)
        E.params = st[0:16]
        names = self.names[(kind, n)] = {}

        def mv(w, x16, out, dep, tokens=T, xs=None, os_=None):
            kw = dict(tokens=tokens, x_stride=xs or w['K'], out_stride=os_ or w['rows']) if tokens > 1 else {}
            si = E.own_matvec_1rt(w['w'], w['fmt'], w['rows'], w['K'], x16, out, rows_per_task=e.rpt, dep=dep, **kw)
            names[si] = w['name']; return si

        r0 = 4 - T   # (this tape's rows of hn / hn16: the last is always row 3, the chained step's input)
        # every per-row buffer as a view of exactly T rows (the ops size their tasks from the tensors they are given)
        V = lambda name, per: b[name][:T * per]
        if kind == 'chain':
            em = E.own_emb(e.emb_dev, e.emb.type, e.vocab, D, self.keys[n - 1:n], V('e', D), key=True)
            hsrc = b['hn'][3 * D:4 * D]
        else:
            em = E.own_emb(e.emb_dev, e.emb.type, e.vocab, D, toks, V('e', D), tokens=T)
            hsrc = V('hin', D)
        n_e = E.own_add_rms16(V('e', D), N['enorm'], b['scr'], b['xe16'], e.eps, dep=em, tokens=T, h16_stride=2 * D)
        n_h = E.own_add_rms16(hsrc, N['hnorm'], b['scr'][T * D:], b['xe16'][D:], e.eps, tokens=T, h16_stride=2 * D)
        x, h, h16 = V('x', D), V('h', D), V('h16', D)
        kr, vr, ko, qf, qo = V('kr', nkv * hd), V('vr', nkv * hd), V('ko', nkv * hd), V('qf', 2 * QH), V('qo', QH)
        att, gat16, blk, ffp = V('att', QH), V('gat16', QH), V('blk', D), V('ffp', D)
        eh = E.also(mv(W['eh'], V('xe16', 2 * D), x, n_e, xs=2 * D, os_=D), n_h)
        n1 = E.own_add_rms16(x, N['an'], h, h16, e.eps, dep=eh, tokens=T)
        k_ = mv(W['k'], h16, kr, n1); v_ = mv(W['v'], h16, vr, n1)
        kd = E.qk_nr(kr, hd, nkv, hd, N['kn'], pos, ko, e.nrot, e.theta_scale, e.eps, dep=k_)
        ap = E.also(E.kv_append(ko, vr, self.cache, pos, dep=kd), v_)
        q_ = mv(W['q'], h16, qf, n1)
        qd = E.qk_nr(qf, 2 * hd, nh, hd, N['qn'], pos, qo, e.nrot, e.theta_scale, e.eps, dep=q_)
        a_ = E.also(E.attn_dec(qo, self.cache, pos, att, nh, nkv, hd, 1.0 / np.sqrt(hd), dep=ap, batched=True), qd)
        g_ = E.gate_mul16(att, qf, 2 * hd, hd, gat16, hd, dep=a_)
        o_ = mv(W['o'], gat16, blk, g_)
        n2 = E.own_add_rms16(x, N['pn'], h, h16, e.eps, d=blk, slots=1, dep=o_, tokens=T)
        gg = mv(W['fg'], h16, V('g', ffn), n2); uu = mv(W['fu'], h16, V('u', ffn), n2)
        sw = E.also(E.own_swiglu16(V('g', ffn), V('u', ffn), V('a16', ffn), dep=uu), gg)
        dd = mv(W['fd'], V('a16', ffn), ffp, sw)
        nf = E.own_add_rms16(x, N['shn'], b['hn'][r0 * D:], b['hn16'][r0 * D:], e.eps, d=ffp, slots=1, dep=dd, tokens=T)
        if kind != 'fill':
            hd_ = mv(e.head, b['hn16'][3 * D:], b['logits'], nf, tokens=1)
            E.own_argmax(b['logits'], e.vocab, self.keys[0:1] if kind == 'catch' else self.keys[n:n + 1], dep=hd_)
        E.build()
        return E

    # -- passes ------------------------------------------------------------------------------------------------------------
    def reset(self):
        """a new sequence: no block row is filled (row 0 is zero: MTP position 0 has no h before it)"""
        with torch.cuda.device(self.eng.dev): self.cache[:2 * self.eng.nkv * self.eng.hd].zero_()
        self.filled = 1

    def _rows(self, h, n):
        if h.dtype != torch.float32 or h.device.index != self.eng.dev or h.numel() < n * self.eng.D:
            raise ValueError(f'Mtp35: h must be fp32 [{n}, {self.eng.D}] on card {self.eng.dev} (got {h.dtype} {tuple(h.shape)} on {h.device})')
        self.b['hin'][:n * self.eng.D].copy_(h.reshape(-1)[:n * self.eng.D])

    def _stage(self, tape, toks, p0):
        o = 32 * self.slots[tape]
        for t, v in enumerate(toks): self.hstage[o + 16 + t] = int(v)
        self.hstage[o + 20] = int(p0)

    def _check(self, toks, p0, n, what):
        e = self.eng
        if self.filled < 1: raise RuntimeError('Mtp35: reset() before the first row')
        if not 1 <= p0 <= self.filled:
            raise ValueError(f'Mtp35 {what}: rows from {p0}, but only rows 1 .. {self.filled - 1} hold the trunk\'s h (a gap)')
        k3.check_span(p0, n, e.max_ctx + MAX_T + self.k, what=f'Mtp35 {what}')
        if any(not 0 <= int(t) < e.vocab for t in toks): raise ValueError(f'Mtp35 {what}: a token of {list(toks)} outside the vocabulary')

    def _launch(self, order):
        """run the tapes in order (one stage copy before), wait once, and return the reports of the waits that gave up"""
        self.dstage.copy_(self.hstage, non_blocking=True)
        for t in order: self.E[t].run(check=False)
        for t in order:
            S = self.E[t].sync
            self.h_st[t].copy_(S[2 * k3.MAXSEG:2 * k3.MAXSEG + k3.STATUS_WORDS], non_blocking=True)
        self.h_keys.copy_(self.keys, non_blocking=True)
        self.ev.record(); self.ev.synchronize()
        bad = []
        for t in order:
            st = self.h_st[t].tolist()
            b = k3.bounds_status(st)
            if b:
                torch.cuda.synchronize(); self.E[t].sync.zero_()
                raise k3.ContextBoundsError(f'Mtp35: the device bound guard caught an out-of-range value ({t}): {b}')
            if st[1]:
                torch.cuda.synchronize(); bad.append(f'{t} segment {st[2]} {self.names[t].get(st[2], "")}'); self.E[t].sync.zero_()
        return bad

    def _run(self, order, what):
        with torch.cuda.device(self.eng.dev):
            for attempt in range(self.eng.max_retries + 1):
                # (the keys start at 0 for every attempt: the stage copy writes them)
                bad = self._launch(order)
                if not bad: return
                self.bails.append((what, attempt, bad))
                print(f'k3 qwen35 mtp: a wait gave up ({what}, attempt {attempt + 1}), running it again: ' + '; '.join(bad), file=sys.stderr, flush=True)
            raise RuntimeError(f'k3 qwen35 mtp: a wait gave up ({what}) in {self.eng.max_retries + 1} attempts: ' + '; '.join(bad))

    def fill(self, h, toks, p0):
        """block rows p0 .. p0 + n - 1 from the trunk's h (fp32 [n, D] on the card: row r = h of position p0 + r - 1) and
        the tokens at those positions, no draft (the prompt). p0 >= 1 and at most the first unfilled row"""
        n = len(toks)
        self._check(toks, p0, n, 'fill')
        if h.numel() < n * self.eng.D: raise ValueError(f'Mtp35 fill: h holds {h.numel() // self.eng.D} rows, {n} tokens')
        hv = h.reshape(-1)
        for a in range(0, n, MAX_T):
            T = min(MAX_T, n - a)
            self._rows(hv[a * self.eng.D:], T)
            self._stage(('fill', T), toks[a:a + T], p0 + a)
            self._run([('fill', T)], f'fill at {p0 + a}')
        self.filled = max(self.filled, p0 + n)

    def draft(self, h, toks, p0, k=None):
        """one round: block rows p0 .. p0 + T - 1 (T = len(toks), 1 .. 4) from the trunk's h (fp32 [T, D]: row r = h of
        position p0 + r - 1) and toks (the tokens at those positions: the accepted drafts, then the newest token), then
        k - 1 chained steps; returns the k drafts (the tokens after position p0 + T - 1). p0 at most the first unfilled
        row; afterwards rows up to p0 + T - 1 count as filled (the chained rows are rewritten by the next round)"""
        k = self.k if k is None else int(k)
        T = len(toks)
        if not 1 <= T <= MAX_T or not 1 <= k <= self.k: raise ValueError(f'Mtp35 draft: {T} rows (1 .. {MAX_T}), k {k} (1 .. {self.k})')
        self._check(toks, p0, T + k - 1, 'draft')
        self._rows(h, T)
        self._stage(('catch', T), toks, p0)
        for i in range(1, k): self._stage(('chain', i), [], p0 + T - 1 + i)
        self._run([('catch', T)] + [('chain', i) for i in range(1, k)], f'draft at {p0}')
        self.filled = max(self.filled, p0 + T)
        return [k3.key_token(v) for v in self.h_keys.tolist()[:k]]
