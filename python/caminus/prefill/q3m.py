"""The batched prefill of a one-card qwen3moe engine (caminus.engine.q3moe_one.Q3MoEOne: Qwen3-30B-A3B) on Caminus's own
kernels (kernels/q3m_prefill.hip through caminus.prefill.q3m_ops; no ggml-cuda code). It reads the decode engine's own
weights (Q4_K raw, Q6_K in the decode's planes) and writes its KV caches in place: nothing copied.

Per layer of a chunk of n tokens at positions p .. p + n - 1, the decode tape's ops in its order:
  h16 = add_rms16(x (+ the previous layer's 8 weighted expert rows), attn_norm)
  q | k | v = the Q4_K / Q6_K GEMMs (WMMA, the weights exact, fp16 x, fp32 sums) into one row per token
  q / k RMS norm + NEOX RoPE (OP_OWN_ATTS's statements), k / v -> the cache rows p .. p + n - 1 (fp16)
  attention: causal over cache rows 0 .. p + t (q3p_attn: fp16 q / k / v / p, fp32 scores and softmax state), split
             into launches of at most ATTN_TP token-positions (each launch well under the 200 ms bound at any depth)
  o = attn_output GEMM; x += o; h, h16 = norm(x, ffn_norm)
  router logits (fp32), the top 8 (OP_ROUTER's gating), the (token, slot) pairs grouped by expert
  gate / up GEMMs over the pairs, fp16 silu(g) u, the down GEMM -> 8 rows per token, added at the next layer's norm
The last layer stops after its cache rows: its attention output and FFN feed only the prompt's logits, which a prefill
never returns (the last prompt token goes through a K3 pass: the first logits come from decode numerics).
Not bitwise the decode (the GEMMs sum in WMMA order): the gate is the G-model one, prefill-then-decode against the
float64 reference with llama.cpp's own prefill-then-decode as the bound (tests/prefill/test_q3m_prefill.py).

The hand-off is the shared driver's (caminus.prefill.driver.PrefillDriver): run(toks, i0, commit, p1) prefills
[i0 - commit, p1) (the KV rows written in place; no recurrent state), sets m.fresh and returns (i0 = p1, commit = 0).
No drafter yet (draft must be None)."""
import numpy as np
import torch

from ..device import capacity
from ..engine import k3
from . import q3m_ops
from .driver import PrefillDriver, PrefillRefused

# T*: below it a prompt part runs as K3 one-token passes (bitwise the decode). Measured 11-10 (benchmarks/q3moe_prefill.py,
# XTX, chunk 2048, from empty, medians of 5; K3 ~4.35 ms a token there): span 4: 28.6 ms (K3 17.4); 6: 29.4 (26.1); 8: 34.0
# (34.8); 16: 37.6 (69.6); 32: 43.4 (139)
CROSSOVER = 8
ATTN_TP = 1 << 24          # token-positions per attention launch (tokens x the deepest position they read)
ROUTER_FLOOR = 6.103515625e-5


def layout(m, T):
    """(f32 counts, f16 counts, i32 counts) of Q3MPrefill's device buffers for engine m at chunk T, from its shapes"""
    D, nh, nkv, hd, k, E, ff = m.D, m.nh, m.nkv, m.hd, m.k_used, m.n_exp, m.ffn
    P = T * k
    f32 = dict(x=T * D, h=T * D, qkv=T * (nh * hd + 2 * nkv * hd), o=T * D, logits=T * E, wts=P,
               big=max(2 * P * ff, P * D))                   # (g | u, then the down rows over them: g, u are dead by then)
    f16 = dict(h16=T * D, q16=T * nh * hd, att16=T * nh * hd, a16=P * ff)
    i32 = dict(toks=T, ids=P, xs=P, yd=P, items=4 * q3m_ops.items_bound(P, E), nitems=1, err=1)
    return f32, f16, i32


def buffer_need_mib(m, T):
    """MiB of Q3MPrefill's device buffers at chunk T (m: the engine, or shapes_of(path))"""
    f32, f16, i32 = layout(m, T)
    return (4 * sum(f32.values()) + 2 * sum(f16.values()) + 4 * sum(i32.values())) / capacity.MIB


def shapes_of(path):
    """the shapes layout() reads, from a qwen3moe GGUF's header alone (caminus serve's projection, before any load)"""
    from types import SimpleNamespace
    from .. import ggufio
    F = ggufio.GGUFFile(path)
    try:
        hp = lambda k: int(F.hp(k))
        return SimpleNamespace(D=hp('embedding_length'), nh=hp('attention.head_count'), nkv=hp('attention.head_count_kv'),
                               hd=hp('attention.key_length'), k_used=hp('expert_used_count'), n_exp=hp('expert_count'),
                               ffn=hp('expert_feed_forward_length'))
    finally:
        F.close()


MM_KEYS = ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps')


class Q3MPrefill(PrefillDriver):
    """the batched prefill of a Q3MoEOne engine m, chunks of up to `chunk` tokens; buffers preallocated (checked against the
    card's guard line first: capacity.check)"""

    def __init__(self, m, chunk=1024, crossover=None, check=True, module=None):
        self.m, self.chunk = m, int(chunk)
        if not 16 <= self.chunk <= 4096: raise ValueError(f'Q3MPrefill: chunk {chunk} (16 .. 4096)')
        self.dev = dev = torch.device(f'cuda:{m.dev}')
        self.crossover = CROSSOVER if crossover is None else int(crossover)
        fmts = {L[key][0] for L in m.L for key in MM_KEYS}
        if module is None:   # (a file with codebook-format tensors: the study object, as the engine picks its decode module)
            module = q3m_ops.IQ_MODULE if fmts & set(q3m_ops.IQ_KERNEL) else 'q3m_prefill'
        for il, L in enumerate(m.L):
            for key in MM_KEYS:
                if L[key][0] not in q3m_ops.formats(module): raise ValueError(f'Q3MPrefill: blk.{il} {key} is {L[key][0]} ({module})')
        T = self.chunk
        self.need_mib = buffer_need_mib(m, T)
        if check: capacity.check([dev.index], [self.need_mib], f'Q3MPrefill buffers (chunk {T})')
        f32, f16, i32 = layout(m, T)
        with torch.cuda.device(dev):
            self.ops = q3m_ops.Ops(dev, module)
            for n, c in f32.items(): setattr(self, n, torch.zeros(c, dtype=torch.float32, device=dev))
            for n, c in f16.items(): setattr(self, n, torch.zeros(c, dtype=torch.float16, device=dev))
            for n, c in i32.items(): setattr(self, n, torch.zeros(c, dtype=torch.int32, device=dev))
        self.items = self.items.view(-1, 4)
        self.host_toks = torch.zeros(T, dtype=torch.int32, pin_memory=True)
        self.host_ev = None
        self.n_vocab = int(m.emb.shape[1])
        self.emb_rb = int(m.emb.rows()[1])
        self.last = None

    # -- checks (the shared driver's: limit = max_ctx, check_tokens, validate, route) -----------------------------------------
    def check_drafter(self, draft):
        if draft is not None: raise PrefillRefused('Q3MPrefill: no drafter yet')

    def numerics(self):
        """the engine key's prefill part: own WMMA GEMMs (fp16 x, exact weights, fp32 sums), the fp32 router"""
        return dict(wmma_out='f32', router_logits='f32', moe='own-wmma', q6k='own-wmma')

    # -- one chunk ---------------------------------------------------------------------------------------------------------
    def _chunk(self, toks, a, b):
        m, O = self.m, self.ops
        n, p = b - a, a
        D, nh, nkv, hd, k, E, ff, nl = m.D, m.nh, m.nkv, m.hd, m.k_used, m.n_exp, m.ffn, m.nl
        Q, M = nh * hd, nkv * hd
        ldq = Q + 2 * M
        if self.host_ev is not None: self.host_ev.synchronize()   # (the stage is rewritten after its last copy finished)
        self.host_toks[:n].numpy()[:] = np.asarray(toks[a:b], dtype=np.int32)
        with torch.cuda.device(self.dev):
            self.toks[:n].copy_(self.host_toks[:n], non_blocking=True)
            ev = torch.cuda.Event(); ev.record(); self.host_ev = ev
        x, h, h16, qkv, q16, att16, o = (self.x[:n * D], self.h[:n * D], self.h16[:n * D], self.qkv[:n * ldq], self.q16[:n * Q],
                                         self.att16[:n * Q], self.o[:n * D])
        P = n * k
        ids, wts, xs, yd = self.ids[:P], self.wts[:P], self.xs[:P], self.yd[:P]
        g, u, dn, a16 = self.big[:P * ff], self.big[P * ff:2 * P * ff], self.big[:P * D], self.a16[:P * ff]
        work_gu = (self.items, self.nitems, q3m_ops.items_bound(P, E), xs, yd)
        work_d = (self.items, self.nitems, q3m_ops.items_bound(P, E), yd, yd)
        O.emb(m.emb_dev, self.emb_rb, self.n_vocab, self.toks, x, n, D)
        scale = 1.0 / float(np.sqrt(hd))
        for il, L in enumerate(m.L):
            if il == 0: O.add_rms16(x, L['attn_norm'], m.eps, h16, n, D)
            else: O.add_rms16(x, L['attn_norm'], m.eps, h16, n, D, d=dn, w=wts, S=k)
            O.mm(L['attn_q'], h16, D, qkv, ldq, n=n)
            O.mm(L['attn_k'], h16, D, qkv[Q:], ldq, n=n)
            O.mm(L['attn_v'], h16, D, qkv[Q + M:], ldq, n=n)
            O.qk_rope(qkv, ldq, nh, nkv, L['qk_par'], p, q16, L['cache'], n, rows=m.max_ctx)
            if il == nl - 1: break                         # (the last layer: only its cache rows are needed)
            _, nb = O.attn_blocks(p, n)
            step = max(1, ATTN_TP // (p + n) // q3m_ops.AT_T)           # (16-position blocks per launch)
            for i0 in range(0, nb, step):
                O.attn(q16, L['cache'], p, n, nkv, scale, att16, i0, min(nb, i0 + step), rows=m.max_ctx)
            O.mm(L['attn_output'], att16, Q, o, D, n=n)
            O.add_rms16(x, L['ffn_norm'], m.eps, h16, n, D, d=o, S=1, h=h)
            O.router(h, L['router'], E, D, self.logits, n)
            O.topk(self.logits, E, k, ROUTER_FLOOR, ids, wts, self.err, n)
            O.group(ids, P, k, E, xs, yd, self.items, self.nitems, self.err)
            O.mm(L['ffn_gate_exps'], h16, D, g, ff, work=work_gu)
            O.mm(L['ffn_up_exps'], h16, D, u, ff, work=work_gu)
            O.swiglu(g, u, a16, P * ff)
            O.mm(L['ffn_down_exps'], a16, ff, dn, D, work=work_d)

    def check_flags(self, what='prefill'):
        """raise if a chunk set the flag word (1: a router logit not finite; 2: an expert id out of range); the caller
        poisons the engine"""
        e = int(self.err.item())
        if e:
            self.err.zero_()
            raise k3.ContextBoundsError(f'{what}: the routing flag word is {e:#x} (1: a router logit not finite, 2: an id out of range)')

    # -- the prefill (run(), fill(), the hand-off: PrefillDriver's) -----------------------------------------------------------
    def _run_chunk(self, toks, k, j, a, b, segs, bounds):
        self._chunk(toks, a, b)

    def _run_settle(self):
        torch.cuda.synchronize(self.dev); self.check_flags('prefill')

    def _sync_devices(self):
        torch.cuda.synchronize(self.dev)
