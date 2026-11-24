"""The batched prefill of the qwen35moe line (Qwen3.6-35B-A3B; caminus.engine.q35_moe.Moe35Own) on Caminus's own code: the
layer loop of a chunk over the decode engine's own weights and states (Moe35Own.L: references, nothing copied). One card,
F16 KV, native rope; no ggml-cuda code. The trunk is the dense qwen35 line's own prefill (caminus.prefill.q35own: its WMMA
GEMMs, GDN glue and WMMA attention tiles, kernels/q35_prefill.hip); the feed-forward is a mixture of experts
(kernels/q35m_prefill.hip, which includes that object).

Per layer of a chunk of n tokens at positions p .. p + n - 1, the decode tape's ops in its order:
  h, h16 = add_rms16(x (+ the previous layer's 9 weighted slots), attn_norm)         q3p_add_rms16 (OP_OWN_ADD_RMS16)
  GDN       alpha / beta (F32: pf_f32_mv_k2048, OP_F32_MV's order), gates, attn_qkv (GEMM), the conv + its state, l2,
            attn_gate (GEMM), the recurrence (the decode's state in place), the output norm with the SiLU gate as fp16,
            ssm_out (GEMM)
  attention attn_q / k / v (GEMMs); k norm + RoPE, the f16 KV append, q norm + RoPE, the attention over positions
            0 .. pos (q35mp_attn: WMMA tiles; 8 query heads per kv head, head dim 256), the output gate as fp16,
            attn_output (GEMM)
  h, h16 = add_rms16(x += block, post_attention_norm)
  MoE       router logits (F32 [256], pf_f32_mv_k2048) and the shared expert's gate logit (F32 [1]); the top 8 with the
            gated shared weight as slot 8 (q35mp_router_sh: OP_ROUTER on the qwen35moe module); the shared expert (Q8_0
            gate / up GEMMs, fp16 swiglu, down GEMM into slot 8); the (token, slot) pairs grouped by expert (q3p_group with its y9 index),
            the routed experts' Q2_0 gate / up (q35mp_mm_q2_0, grouped), fp16 swiglu, down into slots 0 .. 7
            -> the 9 slots are summed, weighted, at the next layer's norm
The last layer stops after its cache rows: its output feeds only the prompt's logits, which a prefill never returns (the
last prompt token goes through a K3 pass). Every op but the GEMMs and the attention is its K3 decode op's statements on
the same inputs; the GEMMs and the attention sum in WMMA order, so the prefill is G-model gated (prefill-then-decode
against the float64 reference). Chunk geometry changes no bit (a row's ops see only its own row and the rows before it).

The hand-off is the shared driver's (caminus.prefill.driver.PrefillDriver): run(toks, i0, commit, p1) prefills
[i0 - commit, p1), sets m.fresh and returns (i0 = p1, commit = 0). No drafter (draft must be None)."""
import ctypes

import numpy as np
import torch

from ..device import capacity, hip
from ..engine import k3
from . import q3m_ops, q35own
from .attn import Attn
from .driver import PrefillDriver, PrefillRefused
from .gdn import Gdn
from .glue import Glue

I, F = ctypes.c_int, ctypes.c_float
NULL = ctypes.c_void_p(0)
MODULE = 'q35m_prefill'
KVQ8_MODULE = 'q35m_prefill_kvq8'   # the Q8 KV cache's append and attention (kernels/q35m_prefill_kvq8.hip)
ROUTER_FLOOR = 6.103515625e-5   # (the router's clamp on the top-k sum: Moe35Own's router_shexp default)
# T*: below it a prompt part runs as K3 one-token passes (bitwise the decode). Unmeasured: q35own's value until a speed
# test sets it
CROSSOVER = q35own.CROSSOVER


class Ops(q35own.Ops):
    """kernels/q35m_prefill.hip's launchers: q35_prefill's (included as they are) + the MoE line's own"""

    # the tiled attention: q35mp_attn (kernels/q35m_prefill.hip: rocKE's gfx11 pipeline on q35p_attn_wm's absolute-position
    # tiles; attn_wm's contract), q35mp_attn_q8 from the Q8 object
    WM_KERNEL = ('q35mp_attn', 'q35mp_attn_q8')

    def __init__(self, dev, module=MODULE, q8=False):
        """q8: the KV cache is Q8_0 rows (kernels/own_kv_codec.h): kv_append_q8 and attn_wm's Q8 kernel run KVQ8_MODULE's
        kernels, every other kernel stays on `module`"""
        super().__init__(dev, module)
        self.q8 = bool(q8)
        if self.q8:
            with torch.cuda.device(dev): self.kvmod = hip.Module(KVQ8_MODULE)

    def kv_append_q8(self, k, ldk, v, ldv, cache, M, pos0, n, rows, err):
        """cache rows pos0 .. pos0 + n - 1 = the Q8_0 blocks of (k, v), fp32 (the decode's own_kv_q8_encode); a refused block
        ORs 2 into err, a row past `rows` 1 (k3.check_span refuses such chunks first)"""
        if not self.q8: raise ValueError('q35moe kv_append_q8: an F16 Ops')
        alloc = k3.kv_rows(cache, int(M), True)
        if not 0 < int(rows) <= alloc: raise ValueError(f'q35moe kv_append_q8: rows {rows} of a cache of {alloc}')
        k3.check_span(pos0, n, int(rows), what='q35moe kv_append_q8')
        self.kvmod.q35p_kv_append_q8((int(n),), (256,), k, I(ldk), v, I(ldv), cache, I(M), I(pos0), I(rows), err)

    def add_rms16(self, x, nw, eps, h16, n, N, d=None, w=None, S=0, h=None):
        """q3p_add_rms16: per row x += the S slots of d weighted by w (w None: summed), then h = rms_norm(x) * nw and
        h16 = fp16(h) (the decode's own_op_add_rms16, the 9-slot form)"""
        return q3m_ops.Ops.add_rms16(self, x, nw, eps, h16, n, N, d=d, w=w, S=S, h=h)

    def mm_q2_0(self, W, x16, ldx, y, ldy, work):
        """y[yd] = W[e] x16[xs] in fp32 for the grouped (token, slot) pairs: work = (items, nitems, bound, xs, yd) from
        group9() (bound = the grid's work-item count; the device's *nitems decides). W: Moe35Own's expert dict (fmt Q2_0,
        rows = one expert's rows, K)"""
        if W['fmt'] != 'Q2_0': raise ValueError(f'q35moe mm_q2_0: {W["fmt"]}')
        R, K = int(W['rows']), int(W['K'])
        if K % 256: raise ValueError(f'q35moe mm_q2_0: K {K} is not a multiple of 256')
        items, nitems, bound, xs, yd = work
        self.mod.q35mp_mm_q2_0((int(bound), -(-R // q3m_ops.MM_N)), (256,), W['w'], I(R), I(K), x16, I(ldx), xs, y, I(ldy),
                               yd, items, nitems, I(0))

    def router_sh(self, logits, sgl, E, k, floor_, ids, wts, err, n):
        """ids [n][k], wts [n][k + 1]: the softmax top k of each row's E logits (weights over their sum clamped at floor_),
        then slot k = 1 / (1 + exp(-sgl)) (OP_ROUTER on the qwen35moe module); a logit not finite sets err bit 1"""
        if not 1 <= E <= 1024 or not 1 <= k <= 32: raise ValueError(f'q35moe router_sh: E {E} (<= 1024), k {k} (<= 32)')
        self.mod.q35mp_router_sh((int(n),), (256,), logits, sgl, I(E), I(k), F(floor_), ids, wts, err)

    def group9(self, ids, npairs, k, E, xs, yp, y9, items, nitems, err):
        if E > 1024: raise ValueError(f'q35moe group9: E {E} (<= 1024)')
        self.mod.q3p_group((1,), (1024,), ids, I(npairs), I(k), I(E), xs, yp, y9, items, nitems, err)   # (Q3P_GROUP_Y9's form)

    def emb_bf16(self, W, vocab, toks, x, n, N):
        self.mod.q35mp_emb_bf16((int(n),), (256,), W, I(vocab), toks, x, I(N))


def layout(m, T):
    """(f32 counts, f16 counts, i32 counts) of Q35MoEPrefill's buffers for engine m (Moe35Own, or shapes_of) at chunk T.
    `work` is shared by the two layer kinds (each is dead before the next one writes)"""
    D, nh, nkv, hd, S, NK, NV = m.D, m.nh, m.nkv, m.hd, m.S, m.NK, m.NV
    ff, fs, E, k = m.ffn, m.ffn_shexp, m.n_exp, m.k_used
    CH, KD, VD, QH = 2 * S * NK + S * NV, NK * S, NV * S, nh * hd
    gdn = 2 * CH + 2 * KD + 2 * VD                       # qkv, conv, qk, z, o
    att = 2 * QH + 3 * nkv * hd + 2 * QH                 # qf, kr, vr, ko, qo, att
    P = T * k
    f32 = dict(x=T * D, h=T * D, blk=T * D, al=T * NV, be=T * NV, gates=T * 2 * NV, work=T * max(gdn, att),
               logits=T * E, sgl=T, wts=T * (k + 1), dn=T * (k + 1) * D, shg=T * fs, shu=T * fs, g=P * ff, u=P * ff)
    f16 = dict(h16=T * D, x16=T * max(VD, QH, fs), a16=P * ff)
    i32 = dict(toks=T, err=1, rerr=1, ids=P, xs=P, yp=P, y9=P, items=4 * q3m_ops.items_bound(P, E), nitems=1)
    return f32, f16, i32


def buffer_need_mib(m, T):
    f32, f16, i32 = layout(m, T)
    return (4 * sum(f32.values()) + 2 * sum(f16.values()) + 4 * sum(i32.values())) / capacity.MIB


def shapes_of(path):
    """the shapes layout() reads, from a qwen35moe GGUF's header alone (before any load)"""
    from types import SimpleNamespace
    from .. import ggufio
    from ..engine.q35_moe import _shapes
    F_ = ggufio.GGUFFile(path)
    try: return SimpleNamespace(**_shapes(F_))
    finally: F_.close()


class Q35MoEPrefill(PrefillDriver):
    """the batched prefill of a Moe35Own engine m (F16 or Q8 KV, native rope or YaRN), chunks of up to `chunk` tokens; buffers
    preallocated (checked against the card's guard line first: capacity.check)"""

    def __init__(self, m, chunk=2048, crossover=None, check=True):
        self.m, self.chunk = m, int(chunk)
        self.yarn = getattr(m, 'yarn_factor', None) is not None   # (the long-context mode: pf_qk_nr's YaRN form)
        self.q8 = bool(getattr(m, 'q8', False))   # (the engine's cache: Q8_0 rows, its Hadamard transforms; never F16 kernels)
        if self.q8 and (m.hd != 256 or m.nh != 8 * m.nkv):
            raise ValueError(f'Q35MoEPrefill: the Q8 cache needs head dim 256 and 8 query heads per kv head ({m.hd}, {m.nh} / {m.nkv})')
        if not 16 <= self.chunk <= 4096: raise ValueError(f'Q35MoEPrefill: chunk {chunk} (16 .. 4096)')
        if m.nl_build != m.nl: raise ValueError('Q35MoEPrefill: a debug_layers engine is refused')
        if m.D != 2048: raise ValueError(f'Q35MoEPrefill: n_embd {m.D} (the F32 matvec kernel is pf_f32_mv_k2048)')
        if not Ops.attn_wm_ok(m.nh, m.nkv, m.hd): raise ValueError(f'Q35MoEPrefill: {m.nh} / {m.nkv} heads of {m.hd}: no tiled attention')
        if m.k_used > 32 or m.n_exp > 1024: raise ValueError(f'Q35MoEPrefill: top {m.k_used} of {m.n_exp} experts')
        bad = [f"blk.{il} {k}: {L[k]['fmt']}" for il, L in enumerate(m.L) for k in self.MM_KEYS[L['kind']] + self.SH_KEYS
               if L[k]['fmt'] != 'Q8_0']
        bad += [f"blk.{il} {k}: {L[k]['fmt']}" for il, L in enumerate(m.L) for k in self.EXP_KEYS if L[k]['fmt'] != 'Q2_0']
        bad += [f"blk.{il} {k}: {L[k]['fmt']}" for il, L in enumerate(m.L) if L['kind'] == 'gdn' for k in ('al', 'be')
                if L[k]['fmt'] != 'F32']
        if m.emb.type != 'BF16': bad.append(f'token_embd: {m.emb.type}')
        if bad: raise ValueError(f'Q35MoEPrefill: {len(bad)} tensors without a prefill kernel (first: {"; ".join(bad[:4])})')
        self.dev = dev = torch.device(f'cuda:{m.dev}')
        self.crossover = CROSSOVER if crossover is None else int(crossover)
        T = self.chunk
        self.need_mib = buffer_need_mib(m, T)
        if check: capacity.check([dev.index], [self.need_mib], f'Q35MoEPrefill buffers (chunk {T})')
        f32, f16, i32 = layout(m, T)
        with torch.cuda.device(dev):
            self.O = Ops(dev, q8=self.q8)
            self.Gl = Glue(dev, module='prefill_glue_q35')
            self.Gd = Gdn(dev, module='prefill_gdn_q35')
            self.A = Attn(dev, name='prefill_attn_q35_yarn' if self.yarn else 'prefill_attn_q35')
            # (the YaRN mode: the decode tape's table and magnitude scale, the same fp32 values: Moe35Own.yarn)
            self.ytab = None if not self.yarn else (torch.tensor(m.yarn[0], dtype=torch.float32, device=dev), float(m.yarn[1]))
            for n, c in f32.items(): setattr(self, n, torch.zeros(c, dtype=torch.float32, device=dev))
            for n, c in f16.items(): setattr(self, n, torch.zeros(c, dtype=torch.float16, device=dev))
            for n, c in i32.items(): setattr(self, n, torch.zeros(c, dtype=torch.int32, device=dev))
        self.host_toks = torch.zeros(T, dtype=torch.int32, pin_memory=True)
        self.host_ev = None
        self.n_vocab = int(m.vocab)
        self.last = None

    MM_KEYS = dict(gdn=('qkv', 'z', 'so'), attn=('q', 'k', 'v', 'o'))
    SH_KEYS = ('sh_gate', 'sh_up', 'sh_down')
    EXP_KEYS = ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps')

    def check_drafter(self, draft):
        if draft is not None: raise PrefillRefused('Q35MoEPrefill: no drafter for this line yet')

    def numerics(self):
        """the engine key's prefill part: own WMMA GEMMs (fp16 x, exact weights, fp32 sums), the F32 router; a Q8 engine's
        rows (the decode's encode of the rotated k / v); a YaRN engine's rope key"""
        return dict(wmma_out='f32', router_logits='f32', moe='own-wmma', q6k='none', dense='own-wmma',
                    **({'kv': self.m.kv_format_id} if self.q8 else {}), **({'rope': self.m.rope_key} if self.yarn else {}))

    def _chunk(self, toks, a, b):
        m, O, Gl, Gd, A = self.m, self.O, self.Gl, self.Gd, self.A
        if getattr(m, 'broken', None): raise k3.InvalidKVEncoding(f'Q35MoEPrefill: refused: {m.broken}')
        n, p = b - a, a
        D, nh, nkv, hd, S, NK, NV, eps = m.D, m.nh, m.nkv, m.hd, m.S, m.NK, m.NV, m.eps
        ff, fs, E, k = m.ffn, m.ffn_shexp, m.n_exp, m.k_used
        CH, KD, VD, QH, M = 2 * S * NK + S * NV, NK * S, NV * S, nh * hd, nkv * hd
        if self.host_ev is not None: self.host_ev.synchronize()   # (the stage is rewritten after its last copy finished)
        self.host_toks[:n].numpy()[:] = np.asarray(toks[a:b], dtype=np.int32)
        with torch.cuda.device(self.dev):
            self.toks[:n].copy_(self.host_toks[:n], non_blocking=True)
            ev = torch.cuda.Event(); ev.record(); self.host_ev = ev
        x, h, blk, h16, x16 = self.x[:n * D], self.h[:n * D], self.blk[:n * D], self.h16[:n * D], self.x16
        P = n * k
        lg, sgl, ids, wts, dn = self.logits[:n * E], self.sgl[:n], self.ids[:P], self.wts[:n * (k + 1)], self.dn[:n * (k + 1) * D]
        shg, shu, g, u, a16 = self.shg[:n * fs], self.shu[:n * fs], self.g[:P * ff], self.u[:P * ff], self.a16[:P * ff]
        xs, yp, y9 = self.xs[:P], self.yp[:P], self.y9[:P]
        bound = q3m_ops.items_bound(P, E)
        work_gu = (self.items, self.nitems, bound, xs, yp)
        work_d = (self.items, self.nitems, bound, yp, y9)
        W = self.work
        O.emb_bf16(m.emb_dev, self.n_vocab, self.toks, x, n, D)
        scale = 1.0 / float(np.sqrt(hd))
        last = m.nl - 1
        for il, L in enumerate(m.L):
            if il: O.add_rms16(x, L['attn_norm'], eps, h16, n, D, d=dn, w=wts, S=k + 1, h=h)
            else: O.add_rms16(x, L['attn_norm'], eps, h16, n, D, h=h)
            if L['kind'] == 'gdn':
                al, be, gt = self.al[:n * NV], self.be[:n * NV], self.gates[:n * 2 * NV]
                o0 = 0
                qkv = W[o0:o0 + n * CH]; o0 += n * CH
                conv = W[o0:o0 + n * CH]; o0 += n * CH
                qk = W[o0:o0 + n * 2 * KD]; o0 += n * 2 * KD
                z = W[o0:o0 + n * VD]; o0 += n * VD
                att = W[o0:o0 + n * VD]
                Gl.f32_mv(L['al']['w'], NV, D, h, al, n); Gl.f32_mv(L['be']['w'], NV, D, h, be, n)
                Gd.gates(al, be, L['dt'], L['a'], gt, n, NV)
                O.mm(L['qkv'], h16, D, qkv, CH, n=n)
                Gl.gdn_conv(qkv, L['conv_w'], L['conv_st'], conv, CH, n)
                Gd.l2(conv, qk, n, 2 * NK, eps, ldx=CH)
                Gd.step(qk, conv[2 * KD:], gt, L['gdn'], att, n, NV, NK, CH)
                if il == last: break                       # (the last layer: only its recurrent states are needed)
                O.mm(L['z'], h16, D, z, VD, n=n)
                O.gdn_norm16(att, L['nw'], z, x16, n, NV, S, eps)
                O.mm(L['so'], x16, VD, blk, D, n=n)
            else:
                o0 = 0
                qf = W[o0:o0 + n * 2 * QH]; o0 += n * 2 * QH
                kr = W[o0:o0 + n * M]; o0 += n * M
                vr = W[o0:o0 + n * M]; o0 += n * M
                ko = W[o0:o0 + n * M]; o0 += n * M
                qo = W[o0:o0 + n * QH]; o0 += n * QH
                att = W[o0:o0 + n * QH]
                O.mm(L['k'], h16, D, kr, M, n=n); O.mm(L['v'], h16, D, vr, M, n=n)
                A.qk_nr(kr, hd, nkv, hd, L['kn'], m.nrot, m.theta_scale, eps, p, ko, n, yarn=self.ytab)
                if self.q8:   # (the decode Q8 tape's transforms: K and q by Hadamard256, V and the output by Hadamard64)
                    A.hadamard(ko, 256, n * M // 256); A.hadamard(vr, 64, n * M // 64)
                    O.kv_append_q8(ko, M, vr, M, L['cache'], M, p, n, m.max_ctx, self.err)
                else: O.kv_append16(ko, M, vr, M, L['cache'], M, p, n, m.max_ctx, self.err)
                if il == last: break                       # (the last layer: only its cache rows are needed)
                O.mm(L['q'], h16, D, qf, 2 * QH, n=n)
                A.qk_nr(qf, 2 * hd, nh, hd, L['qn'], m.nrot, m.theta_scale, eps, p, qo, n, yarn=self.ytab)
                if self.q8: A.hadamard(qo, 256, n * QH // 256)
                for b0, b1 in O.attn_blocks(p, n):
                    O.attn_wm(qo, L['cache'], p, n, nh, nkv, hd, scale, att, m.max_ctx, b0, b1)
                if self.q8: A.hadamard(att, 64, n * QH // 64)
                O.gate_mul16(att, qf, x16, n * QH, hd, 2 * hd, hd)
                O.mm(L['o'], x16, QH, blk, D, n=n)
            O.add_rms16(x, L['post_norm'], eps, h16, n, D, d=blk, S=1, h=h)
            # the router and the shared expert's gate (F32 on the fp32 h), the top k with the shared weight as slot k
            Gl.f32_mv(L['router'], E, D, h, lg, n); Gl.f32_mv(L['sh_inp'], 1, D, h, sgl, n)
            O.router_sh(lg, sgl, E, k, ROUTER_FLOOR, ids, wts, self.rerr, n)
            # the shared expert into slot k of each token's k + 1 down rows
            O.mm(L['sh_gate'], h16, D, shg, fs, n=n); O.mm(L['sh_up'], h16, D, shu, fs, n=n)
            O.swiglu(shg, shu, x16, n * fs)
            O.mm(L['sh_down'], x16, fs, dn[k * D:], (k + 1) * D, n=n)
            # the routed experts: the pairs grouped by expert, gate / up, swiglu, down into slots 0 .. k - 1
            O.group9(ids, P, k, E, xs, yp, y9, self.items, self.nitems, self.rerr)
            O.mm_q2_0(L['ffn_gate_exps'], h16, D, g, ff, work_gu)
            O.mm_q2_0(L['ffn_up_exps'], h16, D, u, ff, work_gu)
            O.swiglu(g, u, a16, P * ff)
            O.mm_q2_0(L['ffn_down_exps'], a16, ff, dn, D, work_d)

    def check_flags(self, what='prefill'):
        e, r = int(self.err.item()), int(self.rerr.item())
        if e & 2:   # (a Q8 block refused by the encode: rows may already be written; the engine is unusable)
            self.err.zero_(); self.rerr.zero_()
            why = f'{what}: a KV block was refused (error word {e:#x}); the cache rows are unusable'
            mark = getattr(self.m, 'mark_broken', None)
            if mark is not None: mark(why, error_cls=k3.InvalidKVEncoding)
            raise k3.InvalidKVEncoding(why)
        if e or r:
            self.err.zero_(); self.rerr.zero_()
            raise k3.ContextBoundsError(f'{what}: the KV append refused rows (word {e:#x}) / the routing flag word is {r:#x} '
                                        f'(1: a router logit not finite, 2: an id out of range)')

    def _run_chunk(self, toks, k, j, a, b, segs, bounds):
        self._chunk(toks, a, b)

    def _run_settle(self):
        torch.cuda.synchronize(self.dev); self.check_flags('prefill')

    def _sync_devices(self):
        torch.cuda.synchronize(self.dev)
