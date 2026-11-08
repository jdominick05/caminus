"""The batched prefill of a one-card qwen35moe engine (caminus.engine.qwen35_one.Qwen35One: Qwen3.6-35B-A3B; the 2-bit GSQ
checkpoint first): the layer loop of a chunk over the decode tape's own weights and states (Qwen35One.pw: references,
nothing copied), on the prefill kernels' qwen35moe code objects (prefill_glue_q35, prefill_gdn_q35, prefill_attn_q35,
prefill_mmq_glue8) and the vendored MMQ. One card: no exchange, no hyper-connections, no PLE, no QSA, no micro-chunks.

Per layer of a chunk of n tokens at positions p .. p + n - 1, the decode tape's ops in its order:
  mixed = rms_mul(res, attn_norm)                         pf_rms_mul at 2048 (OP_RMS_MUL's general loop: bitwise)
  GDN       alpha / beta (BF16: pf_bf16_mv_n4, bitwise OP_BF16_MV), gates (pf_gdn_gates), attn_qkv (MMQ), the conv + its
            state (pf_gdn_conv / pf_conv_state), l2, attn_gate (MMQ), the recurrence (pf_gdn_step: K3's state in place,
            read once, written once), the output norm with the SiLU gate (prefill_gdn_q35), ssm_out (MMQ)
  attention attn_q / k / v (MMQ); k: q/k norm + RoPE, Hadamard; v: Hadamard; the q8_0 KV append; q: norm + RoPE,
            Hadamard; the split pair over cells 0 .. pos at any depth (Attn.attend_full: 32-cell splits, 8 query heads per
            kv head, bitwise the decode's), Hadamard, the output gate, attn_output (MMQ)
  res += block (pf_add_sum with a zero row: OP_ADD_SUM's expression); mixed = rms_mul(res, post_attention_norm)
  FFN       router logits (F32: pf_f32_mv_k2048, bitwise OP_F32_MV), the top 8 (pf_router: OP_ROUTER), the shared-expert
            gate logit (pf_f32_mv), the shared expert (MMQ gate / up, swiglu, MMQ down), the routed experts (mul_mat_id
            MMQ behind the id clamp, mm_ids_helper<8>), the slot-order combine + gated shared expert (pf_moe_combine), then
            res += ffn
The GEMMs are llama.cpp's own batched kernels (bitwise llama.cpp's MMQ), so a prefill is not bitwise K3's decode: its gate
is G4 (prefill-then-decode against llama.cpp's own floor). Every other op is bitwise its decode op on the same inputs.

The hand-off is caminus.prefill.coder.Prefill's: run(toks, i0, commit, p1) prefills [i0 - commit, p1) (committed state: the GDN
state, the conv state and the KV rows, written in place), sets m.fresh and returns (i0 = p1, commit = 0): the next K3
pass commits nothing, and the last prompt token goes through a K3 pass (the first logits come from decode numerics).
No drafter: Qwen3.6 has no DFlash / MTP drafter in K3 yet (draft must be None)."""
import time
import numpy as np
import torch
from ..engine import k3
from ..device import capacity
from . import mmq
from ..engine.flashnext import EPS
from .driver import PrefillDriver, PrefillRefused
from .glue import Glue, check_routing
from .gdn import Gdn
from .attn import Attn, ATTN_SS, ATTN_RS, ROWS_PER_BLOCK
from .tp import Tp
from gguf import quants

# T*: below it a prompt part runs as K3 one-token passes (bitwise the decode). Measured 11-07 (tests/prefill/test_q36_prefill.py
# speed, XTX alone, GSQ, medians of 3, L positions from empty, K3 passes vs the batched prefill of L - 1 + one pass, ms):
# L 8: 45.5 vs 47.8; 16: 90.4 vs 47.4; 32: 180.2 vs 48.0; 128: 728.6 vs 53.0 (K3 ~5.65 ms a token, a chunk ~47 ms flat)
CROSSOVER = 12
REC_MIB = 128          # the attention records of one block of rows (attend_full sizes its blocks to them; at least one row at max_ctx)


def layout(m, T):
    """Prefill35's device buffers for engine m (a Qwen35One, or its shape_only twin) at chunk T: (f32 counts, i32 counts,
    the mmq.Workspace dims), from the model's shapes alone"""
    S, NK, NV = m.ssm['S'], m.ssm['nk'], m.ssm['nv']
    D, nh, nkv, hd = m.D, m.nh, m.nkv, m.hd
    CH, VD, KD = 2 * S * NK + S * NV, S * NV, S * NK
    kk, ne = max(1, m.k_used), max(1, m.n_exp)   # (dense: one 'expert' row of the MoE buffers, never used)
    wd = dict(D=D, qkv=CH, z=VD, ssm_K=VD, q=2 * nh * hd, kv=nkv * hd, ao_K=nh * hd, ffc=m.ffn, k=kk, n_exp=ne,
              kmax=m.ffn if m.dense else 0)   # (the dense FFN's down reads K = ffn through the dense activation buffers)
    rec_f = max(REC_MIB * capacity.MIB // 4, nh * ((m.max_ctx + ATTN_SS - 1) // ATTN_SS) * ATTN_RS)
    f32 = dict(res=T * D, mixed=T * D, zero=T * D, al=T * NV, be=T * NV, gates=T * 2 * NV, conv=T * CH, qk=T * 2 * KD,
               att=T * max(VD, nh * hd), ko=T * nkv * hd, qo=T * nh * hd, logits=T * ne, wts=T * kk, sgl=T, rec=rec_f)
    i32 = dict(ids=T * kk, aerr=1)
    return f32, i32, wd


def buffer_need_mib(m, T):
    """MiB of Prefill35's device buffers at chunk T (what __init__ allocates and checks, before the padded alpha / beta
    copies of a non-BF16 checkpoint: a few MiB): the counted f32 / i32 buffers and the GEMM workspace. Measured 11-07: 906
    MiB on the GSQ-hybrid at 2048 (the formula's own number); AREX-2's dense FFN takes 1,864 (the flat 1,000 MiB caminus
    serve counted before left its prompt cache one slot at max_ctx auto)"""
    f32, i32, wd = layout(m, T)
    return (4 * (sum(f32.values()) + sum(i32.values())) + mmq.workspace_bytes(T, wd)) / capacity.MIB


class Prefill35(PrefillDriver):
    """the batched prefill of a Qwen35One engine m, chunks of up to `chunk` tokens; buffers preallocated (checked against the
    card's guard line first: capacity.check)"""

    def __init__(self, m, chunk=2048, crossover=None, check=True):
        self.m, self.chunk = m, int(chunk)
        if not 1 <= self.chunk <= 2048: raise ValueError(f'Prefill35: chunk {chunk} (1 .. 2048)')
        if not getattr(m, 'pw', None) or len(m.pw) != m.nl:
            raise ValueError('Prefill35: the engine has no per-layer tensors (Qwen35One.pw; a debug_layers engine is refused)')
        self.dev = dev = torch.device(f'cuda:{m.dev}')
        self.crossover = CROSSOVER if crossover is None else int(crossover)
        D, nh, nkv, hd = m.D, m.nh, m.nkv, m.hd
        NK, NV, S = m.ssm['nk'], m.ssm['nv'], m.ssm['S']
        self.dims = dict(CH=2 * S * NK + S * NV, KD=NK * S, VD=NV * S, NK=NK, NV=NV, S=S)
        self.ffn_keys = self.MMQ_KEYS['dense'] if m.dense else self.MMQ_KEYS['ffn']
        self._check_formats()
        fmts = sorted({W['fmt'] for L in m.pw for k in self.MMQ_KEYS[L['kind']] + self.ffn_keys for W in [L[k]]} |
                      {L[k]['fmt'] for L in m.pw if L['kind'] == 'gdn' for k in ('al', 'be') if L[k]['fmt'] in mmq.FORMATS_ALL})
        glue = {2048: 'prefill_glue_q35', 5120: 'prefill_glue_q35d'}.get(D)
        attn = {8: 'prefill_attn_q35', 6: 'prefill_attn_q35d'}.get(nh // nkv)
        if glue is None or attn is None: raise ValueError(f'Prefill35: no prefill objects for n_embd {D} / {nh // nkv} query heads per kv head')
        with torch.cuda.device(dev):
            self.Gl = Glue(dev, module=glue)
            self.Gd = Gdn(dev, module='prefill_gdn_q35')
            self.A = Attn(dev, name=attn)
            self.P = Tp(dev)
            self.M = mmq.MMQ(dev, formats=fmts, glue='prefill_mmq_glue8')
        T = self.chunk
        assert nkv * self.A.G == nh, (nkv, self.A.G, nh)
        f32, i32, wd = layout(m, T)
        ab = self._ab_layout()
        # (the padded alpha / beta copies and their two output buffers in the same check: before, they were added to need_mib
        # after it and never checked against the guard's line)
        mib = buffer_need_mib(m, T) + (sum(n for *_, n in ab) + (2 * 4 * T * 128 if ab else 0)) / capacity.MIB
        if check: capacity.check([dev.index], [mib], f'Prefill35 buffers (chunk {T})')
        self.need_mib = mib
        with torch.cuda.device(dev):
            self.W = mmq.Workspace(dev, T, wd, check=False)
            for n, c in f32.items(): setattr(self, n, torch.zeros(c, dtype=torch.float32, device=dev))
            for n, c in i32.items(): setattr(self, n, torch.zeros(c, dtype=torch.int32, device=dev))
        # the host feed: token_embd rows dequantized by gguf-py (Qwen35One._emb_row's call, for a chunk's distinct tokens at
        # once), into two pinned buffers in turn (an event per buffer: rewritten only after its copy to the card finished)
        te = m.T['token_embd.weight']
        self.n_vocab, self._te = int(te.shape[1]), te
        self.host = [torch.zeros(T, D, dtype=torch.float32, pin_memory=True) for _ in range(2)]
        self.host_ev = [None, None]; self._hb = 0
        self.abq = self._ab_mmq(ab)
        if self.abq:
            self.ab_out = [torch.zeros(T * 128, dtype=torch.float32, device=dev) for _ in range(2)]
        self.last = None
        self.released_mib = capacity.release_cache([m.dev])

    # the matrices each layer kind runs on MMQ (their formats must be MMQ's, rows a multiple of 128)
    MMQ_KEYS = dict(gdn=('qkv', 'z', 'so'), attn=('q', 'k', 'v', 'o'), ffn=('sg', 'su', 'sd', 'g', 'u', 'd'), dense=('fg', 'fu', 'fd'))

    def _check_formats(self):
        """refuse (ValueError) a checkpoint whose matrices the prefill kernels cannot take, before any buffer exists"""
        bad = []
        for il, L in enumerate(self.m.pw):
            for k in self.MMQ_KEYS[L['kind']] + self.ffn_keys:
                W = L[k]
                if W['fmt'] not in mmq.FORMATS_ALL or W['rows'] % 128: bad.append(f"blk.{il} {k}: {W['fmt']} x {W['rows']} rows")
            if not self.m.dense and (L['router'].dtype != torch.float32 or L['sgate'].dtype != torch.float32):
                bad.append(f'blk.{il}: the router / shared gate are not F32')
        if bad:
            raise ValueError(f'Prefill35: {len(bad)} matrices without a prefill kernel (first: {"; ".join(bad[:4])}): this checkpoint '
                             f'decodes on K3 but has no batched prefill yet')

    def _ab_layout(self):
        """[(layer, 'al' | 'be', the tensor, bytes)] of the ssm alpha / beta _ab_mmq pads, sized before anything is allocated"""
        out = []
        for il, L in enumerate(self.m.pw):
            if L['kind'] != 'gdn': continue
            for k in ('al', 'be'):
                W = L[k]
                if W['fmt'] in ('BF16', 'F32'): continue   # (BF16: pf_bf16_mv; F32: pf_f32_mv, both bitwise the decode's)
                if W['fmt'] not in mmq.FORMATS_ALL: raise ValueError(f"Prefill35: blk.{il} ssm {k}: {W['fmt']} has no MMQ kernel")
                out.append((il, k, W, 128 * W['rb'] + mmq.pad_bytes(W['fmt'], W['cols'])))
        return out

    def _ab_mmq(self, ab):
        """ssm alpha / beta that are not BF16 (HauhauCS: Q3_K / Q5_K, 32 rows, below MMQ's 128-row tiles): their raw rows
        copied once into zero-padded [128 rows] matrices for MMQ (llama.cpp's own batched numerics, from the same q8_1
        activations as the decode's matvec; 6.6 MiB for 30 layers); the gates read rows 0 .. 31 of each token's 128 (lda 128).
        BF16 ones (GSQ) stay on pf_bf16_mv_n4, bitwise the decode's OP_BF16_MV. ab: _ab_layout()'s list (counted in the check)"""
        out = {}
        for il, k, W, need in ab:
            rb = W['rb']
            wp = torch.zeros(need, dtype=torch.uint8, device=self.dev)
            wp[:W['rows'] * rb].copy_(W['w'][:W['rows'] * rb])
            out[(il, k)] = dict(name=W['name'] + ' (padded to 128 rows)', w=wp[:128 * rb], fmt=W['fmt'], rows=128, cols=W['cols'], rb=rb)
        return out

    # -- checks (caminus.prefill.driver.PrefillDriver's: limit = max_ctx, check_tokens, validate, route) -----------------------------
    def check_drafter(self, draft):
        if draft is not None: raise PrefillRefused('Prefill35: no drafter (Qwen3.6 has no DFlash / MTP drafter in K3 yet)')

    def numerics(self):
        """the engine key's prefill part: the one Glue's modes (its router mode and WMMA output are unused here: no BF16 WMMA
        GEMMs, the F32 router on pf_f32_mv) and the MMQ GEMMs"""
        return dict(wmma_out=self.Gl.wmma_out, router_logits=self.Gl.router_mode, moe='mmq', q6k='mmq')

    # -- the host feed -----------------------------------------------------------------------------------------------------------
    def _emb(self, toks, a, b):
        """token_embd rows of toks[a:b] into the next pinned buffer (fp32, Qwen35One._emb_row's values); returns it"""
        t = np.asarray(toks[a:b], dtype=np.int64)
        u, inv = np.unique(t, return_inverse=True)
        te = self._te
        rows = quants.dequantize(np.ascontiguousarray(np.asarray(te.data)[u]), te.tensor_type).astype(np.float32).reshape(len(u), self.m.D)
        i = self._hb; self._hb ^= 1
        if self.host_ev[i] is not None: self.host_ev[i].synchronize()
        h = self.host[i][:b - a]
        h.numpy()[:] = rows[inv]
        return i, h

    # -- one chunk -------------------------------------------------------------------------------------------------------------
    def _dense(self, W, x, K, n, out, yc, reuse=True):
        """out [n, rows] = W x (MMQ) for the n rows of x (fp32 [n, K]); x quantized into the workspace buffer of W's layout
        unless yc says it holds this x already (the D-wide mixed rows: quantized once per layout per half)"""
        M, WS = self.M, self.W; fmt = W['fmt']; lay = M.layout(fmt)
        yb = WS.y_d4 if lay == mmq.D4 else WS.y_ds4
        key = (x.data_ptr(), int(K), int(n))
        if not (reuse and yc.get(lay) == key): M.quantize(x, K, n, yb, lay)
        yc[lay] = key if reuse else None
        M.mm(fmt, W['w'], W['rows'], K, yb, n, out)

    def _chunk(self, toks, a, b):
        m, Gl, Gd, A, P, M, WS = self.m, self.Gl, self.Gd, self.A, self.P, self.M, self.W
        n, p = b - a, a
        D, nh, nkv, hd, k, E, ffn = m.D, m.nh, m.nkv, m.hd, m.k_used, m.n_exp, m.ffn
        d = self.dims; CH, KD, VD, NK, NV, S = d['CH'], d['KD'], d['VD'], d['NK'], d['NV'], d['S']
        i, h = self._emb(toks, a, b)
        res, mixed, zero = self.res[:n * D], self.mixed[:n * D], self.zero[:n * D]
        with torch.cuda.device(self.dev):
            res.view(n, D).copy_(h, non_blocking=True)
            ev = torch.cuda.Event(); ev.record(); self.host_ev[i] = ev
        for il, L in enumerate(m.pw):
            Gl.rms_mul(res, L['attn_norm'], mixed, n, EPS)
            yc = {}
            blk = WS.o_blk[:n * D]
            if L['kind'] == 'gdn':
                al, be, gt = self.al[:n * NV], self.be[:n * NV], self.gates[:n * 2 * NV]
                if (il, 'al') in self.abq:     # (MMQ on the padded rows; the gates read 32 of each token's 128)
                    a128, b128 = self.ab_out[0][:n * 128], self.ab_out[1][:n * 128]
                    self._dense(self.abq[(il, 'al')], mixed, D, n, a128, yc); self._dense(self.abq[(il, 'be')], mixed, D, n, b128, yc)
                    Gd.gates(a128, b128, L['dt'], L['a'], gt, n, NV, lda=128, ldb=128)
                elif L['al']['fmt'] == 'F32':   # (AREX-2: OP_F32_MV's order, bitwise the decode's)
                    Gl.f32_mv(L['al']['w'].view(torch.float32), NV, D, mixed, al, n); Gl.f32_mv(L['be']['w'].view(torch.float32), NV, D, mixed, be, n)
                    Gd.gates(al, be, L['dt'], L['a'], gt, n, NV)
                else:
                    Gl.bf16_mv(L['al']['w'], NV, D, mixed, al, n); Gl.bf16_mv(L['be']['w'], NV, D, mixed, be, n)
                    Gd.gates(al, be, L['dt'], L['a'], gt, n, NV)
                oq = WS.o_qkv[:n * CH]
                self._dense(L['qkv'], mixed, D, n, oq, yc)
                conv = self.conv[:n * CH]
                Gl.gdn_conv(oq, L['conv_w'], L['conv_st'], conv, CH, n)
                qk = self.qk[:n * 2 * KD]
                Gd.l2(conv, qk, n, 2 * NK, EPS, ldx=CH)
                oz = WS.o_z[:n * VD]
                self._dense(L['z'], mixed, D, n, oz, yc)
                att = self.att[:n * VD]
                Gd.step(qk, conv[2 * KD:], gt, L['gdn'], att, n, NV, NK, CH)
                xs = WS.x_ssm_out[:n * VD]
                Gd.norm(att, L['nw'], oz, xs, n, NV, EPS)
                self._dense(L['so'], xs, VD, n, blk, yc, reuse=False)
            else:
                oqf, okr, ovr = WS.o_q[:n * 2 * nh * hd], WS.o_k[:n * nkv * hd], WS.o_v[:n * nkv * hd]
                self._dense(L['q'], mixed, D, n, oqf, yc); self._dense(L['k'], mixed, D, n, okr, yc); self._dense(L['v'], mixed, D, n, ovr, yc)
                ko, qo = self.ko[:n * nkv * hd], self.qo[:n * nh * hd]
                A.qk_nr(okr, hd, nkv, hd, L['kn'], m.nrot, m.theta_scale, EPS, p, ko, n); A.hadamard(ko, hd, n * nkv)
                A.hadamard(ovr, 64, n * nkv * hd // 64)
                A.kv_append(ko, nkv * hd, ovr, nkv * hd, nkv * hd, L['kv'], p, n, self.aerr)
                A.qk_nr(oqf, 2 * hd, nh, hd, L['qn'], m.nrot, m.theta_scale, EPS, p, qo, n); A.hadamard(qo, hd, n * nh)
                xa = WS.x_attn_out[:n * nh * hd]
                A.attend_full(qo, L['kv'], p, n, nkv, 1.0 / np.sqrt(hd), xa, self.rec, self.aerr)
                A.hadamard(xa, 64, n * nh * hd // 64, 4); A.gate_mul(xa, oqf, 2 * hd, hd, xa, n * nh * hd, hd)
                self._dense(L['o'], xa, nh * hd, n, blk, yc, reuse=False)
            Gl.add_sum(blk, zero, res, n * D)
            Gl.rms_mul(res, L['post_norm'], mixed, n, EPS)
            yc = {}
            if m.dense:                                    # the dense FFN: gate / up (MMQ), swiglu, down (MMQ), res += it
                og2, ou2, xd2, od2 = WS.o_sh_gate[:n * ffn], WS.o_sh_up[:n * ffn], WS.x_sh_down[:n * ffn], WS.o_sh_down[:n * D]
                self._dense(L['fg'], mixed, D, n, og2, yc); self._dense(L['fu'], mixed, D, n, ou2, yc)
                P.swiglu(og2, ou2, xd2, n * ffn)
                self._dense(L['fd'], xd2, ffn, n, od2, yc, reuse=False)
                Gl.add_sum(od2, zero, res, n * D)
                continue
            lg, ids, wts, sgl = self.logits[:n * E], self.ids[:n * k], self.wts[:n * k], self.sgl[:n]
            Gl.f32_mv(L['router'].view(-1), E, D, mixed, lg, n)
            Gl.router(lg, n, E, k, ids, wts, WS.err)
            Gl.f32_mv(L['sgate'].view(-1), 1, D, mixed, sgl, n)
            osg, osu, xsd, osd = WS.o_sh_gate[:n * ffn], WS.o_sh_up[:n * ffn], WS.x_sh_down[:n * ffn], WS.o_sh_down[:n * D]
            self._dense(L['sg'], mixed, D, n, osg, yc); self._dense(L['su'], mixed, D, n, osu, yc)
            P.swiglu(osg, osu, xsd, n * ffn)
            self._dense(L['sd'], xsd, ffn, n, osd, yc, reuse=False)
            g, u, dn = L['g'], L['u'], L['d']
            og, ou, xmd, od = WS.o_gate[:n * k * ffn], WS.o_up[:n * k * ffn], WS.x_moe_down[:n * k * ffn], WS.o_down[:n * k * D]
            M.group(ids, n, k, E, WS.ids_inv, WS.ids_dst, WS.bounds, False, WS.err)
            M.quantize_scatter(mixed, D, n, WS.ids_inv, k, WS.y_moe, M.layout(g['fmt']))
            M.mm_id(g['fmt'], g['w'], E, g['rows'], D, WS.y_moe, n, k, WS.ids_dst, WS.bounds, og)
            M.mm_id(u['fmt'], u['w'], E, u['rows'], D, WS.y_moe, n, k, WS.ids_dst, WS.bounds, ou)
            P.swiglu(og, ou, xmd, n * k * ffn)
            M.group(ids, n, k, E, WS.ids_fwd, WS.ids_dst, WS.bounds, True, WS.err)
            M.quantize(xmd, ffn, n * k, WS.y_down, M.layout(dn['fmt']), ids=WS.ids_fwd)
            M.mm_id(dn['fmt'], dn['w'], E, dn['rows'], ffn, WS.y_down, n, k, WS.ids_dst, WS.bounds, od)
            P.moe_combine(od, wts, osd, sgl, osd, n, k, D)
            Gl.add_sum(osd, zero, res, n * D)

    def check_flags(self, what='prefill'):
        """raise if a chunk set the routing flag word (the id clamp replaced an id, a NaN / inf logit) or the attention's
        row refusal; the caller poisons the engine"""
        check_routing(self.W.err, what=what)
        e = int(self.aerr.item())
        if e: raise k3.ContextBoundsError(f'{what}: the attention core refused rows (error word {e:#x})')

    # -- the prefill -------------------------------------------------------------------------------------------------------------
    # (run(), fill(), the hand-off, inject (none), close: caminus.prefill.driver.PrefillDriver's; these are its one-card
    # parts. draft must be None: check_drafter)
    def _run_chunk(self, toks, k, j, a, b, segs, bounds):
        self._chunk(toks, a, b)

    def _run_settle(self):
        torch.cuda.synchronize(self.dev); self.check_flags('prefill')

    def _sync_devices(self):
        torch.cuda.synchronize(self.dev)
