"""The model's MTP block (qwen4exp nextn) as a K3 drafter beside the TP target (caminus.engine.flashnext_tp.FlashNextTP).

The block comes from the sidecar GGUF (blk.48 = one QSA full-attention MoE layer with 512 experts, + nextn.{eh_proj, enorm,
hnorm, hc_head_*}); its token_embd / output equal the Coder's, so the target's embedding rows and head shards are reused.
The graph is llama.cpp's graph_mtp (ignite-llama src/models/qwen4exp.cpp):
    h_norm = rms(h per hc stream) * hnorm[s];  e_norm = rms(embed(tok)) * enorm, repeated over the 4 streams
    res = eh_proj . concat(e_norm, h_norm)      (one K = 5120 dot per stream)
    hc_attn mix, attention, combine, hc_ffn mix, MoE + shared expert, combine -> res (= the next step's h)
    hc_head mix (the output norm), the shared output head -> logits; greedy: argmax
Row p pairs the token at position p with the target's h of position p - 1 (position 0: zeros).

On the device (TP like the trunk: card c = kv head c and its 12 query heads; experts split by intermediate rows; the shared
expert likewise; hc / router / eh_proj replicated; the head by vocabulary, the target's own shards):
- rms_mul over X [rows * 8, D] (rows: e, h_0, e, h_1, ... ; weights [enorm, hnorm_0, enorm, hnorm_1, ...]) writes the
  concatenation [rows * 4, 2 * D] directly; eh_proj is one BF16 segment over the rows * 4 stream rows (bf16_mv ntok)
- KV f16 without rotation (llama.cpp's draft context: -ctkd default f16, no ctx_other for qwen4exp, so no Hadamard)
- two engine sets: the catch-up (T = the target's ntok rows: the verified pairs of a round, ending at the bonus token,
  whose row gives draft 1) and the chain (T = 1: draft i + the block's own res -> draft i + 1)
- the shared expert is BF16: gate / up as BF16 matvecs, the down with the fp32 swiglu LDS stage (k3_gguf_gsq_mtp*,
  OP_BF16_MV work 2); its split must keep <= 320 columns per card (the LDS stage)

The MTP layer has no recurrent state: its KV rows are rewritten by position, so rollback is a pointer (n_done: the rows
paired with the target's final h) and a bailed launch is simply run again. Long context: the target runs the model's QSA
past 2051 positions (FlashNextTP with max_ctx > 2051), so the output stays exact (the verify decides every token). The MTP
layer itself (blk.48, a QSA layer in the model: its indexer tensors are in the sidecar) keeps DENSE attention over every
position here: up to 2051 positions that is its QSA exactly; past it the drafts are the dense layer's, not llama.cpp's
MTP (draft quality only), and each MTP launch walks all positions (its cost grows with the context). A target limited to
2051 positions (the dense tape) limits the drafter the same way."""
import os
import sys
import numpy as np
import torch
from ..engine import k3
from ..engine.flashnext import EPS, DirectReader
from ..engine.flashnext_tp import FlashNextTP, Slots
from gguf import GGUFReader                      # (caminus.engine.flashnext puts gguf-py on the path)
from gguf.constants import GGML_QUANT_SIZES      # noqa: F401  (FlashNextTP's slicing methods use it)

from .. import config as _config   # noqa: E402
SIDECAR = _config.mtp()   # the MTP sidecar GGUF (not distributed: CAMINUS_MTP / config.toml; None when unset)
from ..device.capacity import QSA_EXACT   # noqa: E402  (2051: dense attention == the QSA layer's selection up to here)
from ..device import capacity   # noqa: E402
MODULE = 'k3_gguf_gsq_mtp'
P = 'blk.48'


class _Sidecar:
    """the sidecar's tensors with FlashNextTP's slicing methods (the same per-card pieces as the trunk's)"""
    _dev, _full, _info, _rep, _rep_f32, _router_w = (FlashNextTP._dev, FlashNextTP._full, FlashNextTP._info, FlashNextTP._rep,
                                                    FlashNextTP._rep_f32, FlashNextTP._router_w)
    _rows, _cols, _exp_rows, _exp_cols = FlashNextTP._rows, FlashNextTP._cols, FlashNextTP._exp_rows, FlashNextTP._exp_cols
    _padded, _shard, ksplit_pad = FlashNextTP._padded, FlashNextTP._shard, FlashNextTP.__dict__['ksplit_pad']

    def __init__(self, path, devs, D):
        self.shards = {}
        self.rd = GGUFReader(path, 'r'); self.T = {t.name: t for t in self.rd.tensors}
        self.f = {x.name: x for x in self.rd.fields.values()}
        self.dr, self.devs, self.D = DirectReader(path), devs, D

    def hp(self, k):
        return self.f[f'qwen4exp.{k}'].contents()


class _Set:
    """one engine pair (one per card) for `rows` rows per launch, with its staging and buffers"""
    pass


class MTPK3:
    def __init__(self, tgt, path=SIDECAR, moe_blocks=12, shexp_blocks=10, chain=True, limit_us=200_000, max_retries=8, solo=None,
                 vram_check=True):
        """tgt: the FlashNextTP target (its head shards, embedding rows and per-pass residual are used). moe_blocks /
        shexp_blocks: card 0's share of the routed / shared experts' 640-wide intermediate, in 32s (the shared expert's
        <= 10 per card: the fp32 swiglu stage holds <= 320 columns). chain=False: no T = 1 set (one draft per round).
        solo = c: build card c's half alone with its exchanges local (a partial, wrong sums: an indexing test of the tapes on
        one card, e.g. the XT alone with both 'cards' on one device).
        vram_check: before loading or allocating anything, refuse (capacity.CapacityError) when the MTP layer's VRAM (its
        measured fixed part + the H rows and KV of max_ctx + T positions, capacity.drafter_need_mib) would push a card past
        its cap given its use now (at 128K the H rows alone are ~5 GiB per card)"""
        assert isinstance(tgt, FlashNextTP) and 'gsq' in tgt.module, 'MTPK3: the GSQ-RCO Coder target in K3 TP'
        if path is None:
            raise ValueError('MTPK3: no MTP sidecar GGUF: pass path= or set CAMINUS_MTP / mtp in config.toml (the sidecar is not distributed)')
        lim = getattr(tgt, 'qsa_limit', QSA_EXACT)
        if lim is not None and tgt.max_ctx > lim:   # (a target whose passes stop at QSA_EXACT: the dense tape)
            raise ValueError(f'MTPK3: target max_ctx {tgt.max_ctx} > {lim}: its passes are exact only up to {lim} positions')
        self.tgt, self.devs, self.D, self.hc = tgt, tgt.devs, tgt.D, tgt.hc
        self.cards, self.local = ((0, 1), False) if solo is None else ((solo,), True)
        self.limit_us, self.max_retries, self.bails = limit_us, max_retries, []
        W = self.W = _Sidecar(path, tgt.devs, tgt.D)
        hp = W.hp
        assert int(hp('embedding_length')) == tgt.D and int(hp('hyper_connection.count')) == tgt.hc
        assert int(hp('nextn_predict_layers')) == 1 and int(hp('block_count')) == tgt.nl + 1
        self.n_exp, self.k_used, self.ffn = int(hp('expert_count')), int(hp('expert_used_count')), int(hp('expert_feed_forward_length'))
        self.nh, self.nkv, self.hd, self.nrot = (int(hp('attention.head_count')), int(hp('attention.head_count_kv')),
                                                 int(hp('attention.key_length')), int(hp('rope.dimension_count')))
        assert (self.nh, self.nkv, self.hd, self.nrot) == (tgt.nh, tgt.nkv, tgt.hd, tgt.nrot) and self.nkv == 2
        assert float(hp('rope.freq_base')) == float(tgt.rd.fields['qwen4exp.rope.freq_base'].contents())
        assert int(hp('expert_shared_feed_forward_length')) == self.ffn
        self.theta_scale = tgt.theta_scale
        for n in ('token_embd.weight', 'output.weight'):   # reused from the target: the same tensors (type, shape, bytes)
            a, b = W.T[n], tgt.T[n]
            assert a.tensor_type == b.tensor_type and list(a.shape) == list(b.shape) and int(a.n_bytes) == int(b.n_bytes), n
            for off in (0, int(a.n_bytes) // 2, int(a.n_bytes) - 4096):
                assert bytes(np.asarray(a.data).reshape(-1)[off:off + 4096]) == bytes(np.asarray(b.data).reshape(-1)[off:off + 4096]), f'{n}: sidecar bytes differ'
        nbt = self.ffn // 32
        assert 0 < moe_blocks < nbt and 0 < shexp_blocks < nbt and max(shexp_blocks, nbt - shexp_blocks) * 32 <= 320
        self.mb = [(0, moe_blocks), (moe_blocks, nbt)]; self.sb = [(0, shexp_blocks), (shexp_blocks, nbt)]
        self.ffc = [32 * (b1 - b0) for b0, b1 in self.mb]; self.ffs = [32 * (b1 - b0) for b0, b1 in self.sb]
        self.Tc = tgt.ntok
        self.max_ctx = tgt.max_ctx
        self.kv_rows = self.max_ctx + self.Tc         # + the catch-up's padding rows past the last position
        if vram_check:   # (before _load and the H / KV allocations)
            need = capacity.drafter_need_mib('mtp', tgt.geom, self.max_ctx, self.Tc)
            self.vram_rows = capacity.check([self.devs[c] for c in self.cards], [need[c] for c in self.cards],
                                            f'MTPK3(max_ctx={self.max_ctx}, T={self.Tc})')
        self._load()
        # h rows by position + 1 (row 0 = zeros: position 0's pair): the target's h_nextn of every verified position
        self.H, self.cache = {}, {}
        for c in self.cards:
            with torch.cuda.device(self.devs[c]):
                self.H[c] = torch.zeros(self.kv_rows + 1, self.hc * self.D, device='cuda')
                self.cache[c] = torch.zeros(self.kv_rows * k3.kv_pos_bytes(self.hd, False), dtype=torch.uint8, device='cuda')
        self.sets = {'catch': self._build_set(self.Tc, 'full')}
        if chain and self.Tc > 2: self.sets['chain'] = self._build_set(1, 'full')
        torch.cuda.set_device(self.devs[0])
        self.reset()

    # -- weights (one copy, shared by the engine sets) ---------------------------------------------------------------
    def _load(self):
        W, D, HC, hd = self.W, self.D, self.hc, self.hd
        nh = self.nh // 2
        L = self.w = {}
        en = W._full(f'{P}.nextn.enorm.weight').view(torch.float32)
        hn = W._full(f'{P}.nextn.hnorm.weight').view(torch.float32).view(HC, D)
        w8 = torch.stack([v for s in range(HC) for v in (en, hn[s])]).contiguous()   # [enorm, hnorm_0, enorm, hnorm_1, ...]
        L['w8'] = [w8, w8.to(W._dev(1))]
        L['eh'] = W._rep(f'{P}.nextn.eh_proj.weight')
        for pre in ('hc_attn', 'hc_ffn', 'nextn.hc_head'):
            L[pre] = dict(norm=W._rep_f32(f'{P}.{pre}_norm.weight', (HC, D)), down=W._rep(f'{P}.{pre}_down.weight'),
                          up=W._rep(f'{P}.{pre}_up.weight'), inject=W._rep(f'{P}.{pre}_inject.weight') if pre != 'nextn.hc_head' else None)
        L['q'] = W._rows(f'{P}.attn_q.weight', [[(nh * 2 * hd * c, nh * 2 * hd * (c + 1))] for c in (0, 1)])
        L['k'] = W._rows(f'{P}.attn_k.weight', [[(hd * c, hd * (c + 1))] for c in (0, 1)])
        L['v'] = W._rows(f'{P}.attn_v.weight', [[(hd * c, hd * (c + 1))] for c in (0, 1)])
        L['qn'], L['kn'] = W._rep_f32(f'{P}.attn_q_norm.weight'), W._rep_f32(f'{P}.attn_k_norm.weight')
        bs = GGML_QUANT_SIZES[W.T[f'{P}.attn_output.weight'].tensor_type][0]; bph = hd // bs; assert bph * bs == hd
        L['o'] = W._cols(f'{P}.attn_output.weight', [[(nh * bph * c, nh * bph * (c + 1))] for c in (0, 1)])
        L['router'] = W._router_w(f'{P}.ffn_gate_inp.weight', self.n_exp)
        L['sgate'] = W._router_w(f'{P}.ffn_gate_inp_shexp.weight', 1)
        rs = [[(32 * b0, 32 * b1)] for b0, b1 in self.sb]
        L['sg'], L['su'] = W._rows(f'{P}.ffn_gate_shexp.weight', rs), W._rows(f'{P}.ffn_up_shexp.weight', rs)
        qk = GGML_QUANT_SIZES[W.T[f'{P}.ffn_down_shexp.weight'].tensor_type][0]
        L['sd'] = W._cols(f'{P}.ffn_down_shexp.weight', [[(32 * b0 // qk, 32 * b1 // qk)] for b0, b1 in self.sb])
        rr = [(32 * b0, 32 * b1) for b0, b1 in self.mb]
        L['g'], L['u'] = W._exp_rows(f'{P}.ffn_gate_exps.weight', rr), W._exp_rows(f'{P}.ffn_up_exps.weight', rr)
        qk = GGML_QUANT_SIZES[W.T[f'{P}.ffn_down_exps.weight'].tensor_type][0]
        assert all(32 * v % qk == 0 for r in self.mb for v in r)
        L['d'] = W._exp_cols(f'{P}.ffn_down_exps.weight', [(32 * b0 // qk, 32 * b1 // qk) for b0, b1 in self.mb])
        for k in ('eh', 'q', 'k', 'v', 'o', 'sg', 'su', 'sd'):
            assert L[k][0]['fmt'] == 'BF16', (k, L[k][0]['fmt'])
        for pre in ('hc_attn', 'hc_ffn', 'nextn.hc_head'):
            assert all(L[pre][n] is None or L[pre][n][0]['fmt'] == 'BF16' for n in ('down', 'up', 'inject'))

    # -- tapes -------------------------------------------------------------------------------------------------------
    def _build_set(self, Tm, part):
        """part: 'full' (the whole step), 'pre' (rms + eh_proj only: refdump checks), 'head' (hc_head mix + head from res)"""
        D, HC, hd, L, tgt = self.D, self.hc, self.hd, self.w, self.tgt
        nh, nkv = self.nh // 2, self.nkv // 2
        st = _Set(); st.rows, st.part = Tm, part
        W0 = 64
        st.hstage = torch.zeros(W0 + Tm * D, dtype=torch.float32, pin_memory=True)
        hs = st.hstage.numpy(); st.h_i32 = hs[:W0].view(np.int32); st.h_emb = hs[W0:].reshape(Tm, D)
        st.E, st.blk, st.pos, st.epoch, st.erow, st.X, st.CAT, st.res, st.S, st.lg = ({} for _ in range(10))
        st.slots = {c: Slots(self.devs[c], 2, Tm * D) for c in self.cards}
        CC = self.cards
        st.token = 0
        RPT = dict(hc_down=1, hc_up=32, head_up=64, router=4, attn_o=16, sdown=32, eh=4) if Tm == 1 else \
            dict(hc_down=2, hc_up=24, head_up=24, router=2, attn_o=8, sdown=64, eh=4)
        for c in CC:
            with torch.cuda.device(self.devs[c]):
                B = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device='cuda')
                q8b = lambda n: B((n + 31) // 32 * 36, dt=torch.uint8)
                E = k3.Engine(module=MODULE, grid=k3.resident_grid(MODULE), T=Tm, limit_us=self.limit_us)
                blk = B(W0 + Tm * D); E.params = blk[0:16].view(torch.int32)
                pos, epoch = blk[16:17].view(torch.int32), blk[17:18].view(torch.int32)
                erow = blk[W0:].view(Tm, D); assert erow.data_ptr() % 256 == 0
                X, CAT, res = B(Tm * 8, D), B(Tm * HC, 2 * D), B(Tm * HC, D)
                fc, fs = self.ffc[c], self.ffs[c]
                s = dict(xn=B(Tm * HC, D), lo=B(Tm * 320), inj=B(Tm * HC), gate=B(Tm * HC, D), mixed=B(Tm * D), mq=q8b(Tm * D),
                         blkp=B(Tm * D), blk=B(Tm * D), kr=B(Tm * nkv * hd), vr=B(Tm * nkv * hd), ko=B(Tm * nkv * hd),
                         qf=B(Tm * nh * 2 * hd), qo=B(Tm * nh * hd), att=B(Tm * nh * hd), gat=B(Tm * nh * hd),
                         logit=B(Tm * self.n_exp), ids=B(Tm * self.k_used, dt=torch.int32), wts=B(Tm * self.k_used),
                         h=B(Tm * self.k_used * fc), hq=q8b(Tm * self.k_used * fc), moe=B(Tm * D), shh=B(Tm * fs), shu=B(Tm * fs),
                         shx=B(Tm * D), sgl=B(Tm), ffp=B(Tm * D), ffo=B(Tm * D))
                st.E[c], st.blk[c], st.pos[c], st.epoch[c], st.erow[c] = E, blk, pos, epoch, erow
                st.X[c], st.CAT[c], st.res[c], st.S[c] = X, CAT, res, s
        E, S = st.E, st.S
        ON = lambda c: torch.cuda.device(self.devs[c])

        def bmv(c, Wi, x, out, dep, rpt, **kw):
            return E[c].bf16_mv(Wi['w'], Wi['rows'], Wi['cols'], x, out, rows_per_task=rpt, dep=dep, **kw)

        def xchg(c, partial, out, x, dep):
            peer = st.slots[c if self.local else 1 - c]
            return E[c].xchg(partial, out, peer.slot(x), st.slots[c].slot(x), st.epoch[c], x, c, local_only=self.local, dep=dep)

        def hc_mix(pre, dep, inject=True, post=None, up_rpt=RPT['hc_up']):
            M, IJ = {}, {}
            for c in CC:
                with ON(c):
                    s = S[c]; w = L[pre]
                    if post is None: r = E[c].rms_mul(st.res[c], w['norm'][c], s['xn'], EPS, dep=dep[c])
                    else: r = E[c].also(E[c].hc_post_rms(s[post[0]], s['inj'], st.res[c], w['norm'][c], s['xn'], EPS, dep=post[1][c]), post[2][c])
                    a = bmv(c, w['down'][c], s['xn'], s['lo'], r, RPT['hc_down'])
                    ij = bmv(c, w['inject'][c], s['xn'], s['inj'], r, 8) if inject else r
                    a = bmv(c, w['up'][c], s['lo'], s['gate'].reshape(-1), a, up_rpt, silu_scale=1.0 / HC)
                    M[c] = E[c].hc_mean(s['xn'], s['gate'], s['mixed'], dep=a, q8=s['mq']); IJ[c] = ij
            return M, IJ

        dep = {c: -1 for c in CC}
        if part in ('full', 'pre'):
            for c in CC:
                with ON(c):
                    a = E[c].rms_mul(st.X[c], L['w8'][c], st.CAT[c].view(Tm * 8, D), EPS)
                    dep[c] = bmv(c, L['eh'][c], st.CAT[c], st.res[c], a, RPT['eh'], ntok=Tm * HC)
        if part == 'full':
            M, inj = hc_mix('hc_attn', dep)
            BLK = {}
            for c in CC:
                with ON(c):
                    s = S[c]; cache = self.cache[c]
                    k_ = bmv(c, L['k'][c], s['mixed'], s['kr'], M[c], 16); v_ = bmv(c, L['v'][c], s['mixed'], s['vr'], M[c], 16)
                    kd = E[c].qk_nr(s['kr'], hd, nkv, hd, L['kn'][c], st.pos[c], s['ko'], self.nrot, self.theta_scale, EPS, dep=k_)
                    ap = E[c].also(E[c].kv_append(s['ko'], s['vr'], cache, st.pos[c], q8=False, dep=kd), v_)
                    q_ = bmv(c, L['q'][c], s['mixed'], s['qf'], M[c], 16)
                    qd = E[c].qk_nr(s['qf'], 2 * hd, nh, hd, L['qn'][c], st.pos[c], s['qo'], self.nrot, self.theta_scale, EPS, dep=q_)
                    a_ = E[c].also(E[c].attn_dec(s['qo'], cache, st.pos[c], s['att'], nh, nkv, hd, 1.0 / np.sqrt(hd), q8=False, dep=ap), qd)
                    a_ = E[c].gate_mul(s['att'], s['qf'], 2 * hd, hd, s['gat'], hd, dep=a_)
                    BLK[c] = bmv(c, L['o'][c], s['gat'], s['blkp'], a_, RPT['attn_o'])
            XB = {}
            for c in CC:
                with ON(c): XB[c] = xchg(c, S[c]['blkp'], S[c]['blk'], 0, BLK[c])
            M, inj = hc_mix('hc_ffn', None, post=('blk', XB, inj))
            FF = {}
            for c in CC:
                with ON(c):
                    s = S[c]
                    rw, sw = L['router'][c], L['sgate'][c]; assert rw['fmt'] == sw['fmt'] == 'F32'
                    r_ = E[c].f32_mv(rw['w'], s['mixed'], s['logit'], rows_per_task=RPT['router'], dep=M[c])
                    r_ = E[c].router(s['logit'], self.k_used, s['ids'], s['wts'], dep=r_)
                    sl = E[c].f32_mv(sw['w'], s['mixed'], s['sgl'], rows_per_task=8, dep=M[c])
                    sg_ = bmv(c, L['sg'][c], s['mixed'], s['shh'], M[c], 16)
                    su_ = bmv(c, L['su'][c], s['mixed'], s['shu'], sl, 16)      # (after the shared gate logit: one chain)
                    s_ = E[c].also(bmv(c, L['sd'][c], s['shh'], s['shx'], sg_, RPT['sdown'], swiglu_mul=s['shu']), su_)
                    g, u, d = L['g'][c], L['u'][c], L['d'][c]
                    r_ = E[c].gg_moe_gu(g['w'], u['w'], g['fmt'], g['slab'], self.ffc[c], D, s['ids'], s['mq'], s['h'], rows_per_task=8, dep=r_, up_fmt=u['fmt'])
                    r_ = E[c].gg_q8_1(s['h'], s['hq'], dep=r_)
                    r_ = E[c].gg_moe_down(d['w'], d['fmt'], d['slab'], self.ffc[c], D, s['ids'], s['wts'], s['hq'], s['moe'], rows_per_task=8, dep=r_)
                    f_ = E[c].also(E[c].shexp_add(s['moe'], s['shx'], s['sgl'], s['ffp'], dep=r_), s_)
                    FF[c] = xchg(c, s['ffp'], s['ffo'], 1, f_)
            for c in CC:
                with ON(c): dep[c] = E[c].also(E[c].hc_post(S[c]['ffo'], S[c]['inj'], st.res[c], st.res[c], dep=FF[c]), inj[c])
        if part in ('full', 'head'):
            M, _ = hc_mix('nextn.hc_head', dep, inject=False, up_rpt=RPT['head_up'])
            # the head: every row for T = 1 / the 'head' test tape; the catch-up only its last row (the draft's), one token's
            # matvec on q8_1 row Tm - 1 (catch_up() ends the final launch at the bonus token)
            st.head_rows = Tm if (Tm == 1 or part == 'head') else 1
            for c in CC:
                with ON(c):
                    hw = tgt.head_w[c]
                    lg = torch.zeros(st.head_rows, hw['rows'], device='cuda'); st.lg[c] = lg
                    if st.head_rows == Tm:
                        E[c].gg_matvec(hw['w'], hw['fmt'], hw['rows'], hw['cols'], hw['rb'], S[c]['mq'], lg, rows_per_task=64, dep=M[c])
                    else:
                        rb = D // 32 * 36; mq1 = S[c]['mq'][(Tm - 1) * rb:Tm * rb]
                        E[c].gg_matvec(hw['w'], hw['fmt'], hw['rows'], hw['cols'], hw['rb'], mq1, lg, rows_per_task=64, dep=M[c], ntok=1)
        st.d_max, st.h_max, st.h_st, st.ev = {}, {}, {}, {}
        for c in CC:
            with ON(c):
                if Tm == 1:   # the trunk's one-token prefetch cap on every matvec (k3_extra_prefetch)
                    for sg in E[c].segs:
                        if sg.op in (k3.OP_GG_MV, k3.OP_BF16_MV): sg.aux = 16384 + 1
                E[c].build()
                hr = getattr(st, 'head_rows', Tm)
                st.d_max[c] = (torch.zeros(hr, device='cuda'), torch.zeros(hr, dtype=torch.int64, device='cuda'))
                st.ev[c] = torch.cuda.Event()
            st.h_max[c] = (torch.zeros(hr, pin_memory=True), torch.zeros(hr, dtype=torch.int64, pin_memory=True))
            st.h_st[c] = torch.zeros(k3.STATUS_WORDS, dtype=torch.int32, pin_memory=True)
        g = [(sorted(cap for _, cap in E[j].guard_caps.values()), E[j].guard_commit, E[j].T) for j in CC]
        assert all(x == g[0] for x in g), f'MTP: the cards guard different bounds {g}'
        return st

    # -- state -------------------------------------------------------------------------------------------------------
    def reset(self):
        """a new sequence: no MTP row is paired yet (KV rows are rewritten by position, nothing else to clear)"""
        self.n_done = 0       # rows 0 .. n_done - 1 hold (token p, target h p - 1) with final values
        self.h_valid = 0      # H rows 1 .. h_valid hold target h of positions 0 .. h_valid - 1 (from take_h)
        self.poisoned = None
        self._force_bail = None
        for c in self.cards: self.H[c][0].zero_()
        self.stats = dict(launches=0, catch_rows=0, chain_steps=0)

    def snapshot(self):
        """caminus.decode.longctx: the host state of a sequence (H / KV rows are by position: rows below n_done / h_valid stay as
        written, later ones are rewritten before they are read)"""
        if self.poisoned: raise k3.ContextBoundsError(f'MTP snapshot: poisoned ({self.poisoned})')
        return dict(n_done=self.n_done, h_valid=self.h_valid)

    def restore(self, s):
        if self.poisoned: raise k3.ContextBoundsError(f'MTP restore: poisoned ({self.poisoned})')
        self.n_done, self.h_valid = s['n_done'], s['h_valid']
        self.last = None   # (draft() runs a catch-up before any chain step, which sets it)

    def take_h(self, i0, n=None):
        """after the target's pass at positions i0 .. i0 + n - 1: its h_nextn (the hc residual after the last layer, before the
        output mixer: the target's res) into H rows i0 + 1 .. i0 + n, on each card's stream (before the next pass's staging
        copy rewrites res). Rows of rejected drafts are rewritten by the pass that verifies those positions again."""
        n = self.tgt.ntok if n is None else n
        k3.check_span(i0 + 1, n, self.H[self.cards[0]].shape[0], what='MTP take_h (H rows)')
        for c in self.cards:
            with torch.cuda.device(self.devs[c]):
                self.H[c][i0 + 1:i0 + 1 + n].copy_(self.tgt.res[c].view(self.tgt.ntok, -1)[:n], non_blocking=True)
        self.h_valid = i0 + n

    def set_h(self, p0, h):
        """test hook: H rows p0 + 1 .. (target h of positions p0 ..) from a host array [n, hc * D] (the llama.cpp dump)"""
        n = h.shape[0]; k3.check_span(p0 + 1, n, self.H[self.cards[0]].shape[0], what='MTP set_h')
        t = torch.from_numpy(np.ascontiguousarray(h, dtype=np.float32))
        for c in self.cards: self.H[c][p0 + 1:p0 + 1 + n].copy_(t.to(self._d(c)))
        self.h_valid = max(self.h_valid, p0 + n)

    def _d(self, c):
        return f'cuda:{self.devs[c]}'

    # -- launches ----------------------------------------------------------------------------------------------------
    def _launch(self, st, p0, toks, hsrc, nvalid=None):
        """one launch of set st: rows at positions p0 .. p0 + rows - 1 with tokens toks (nvalid of them; the rest repeat the
        last valid row, written past the last position and rewritten before any read) and h rows hsrc(c) -> [nvalid, hc, D]
        on card c. Returns the greedy token of each head row (a list: every row, or the catch-up's last row only), or None
        for a set without the head."""
        Tm = st.rows; nvalid = Tm if nvalid is None else nvalid
        assert 1 <= nvalid <= Tm and len(toks) == nvalid, (nvalid, Tm, len(toks))
        if self.poisoned: raise k3.ContextBoundsError(f'MTP: refused until reset(): {self.poisoned}')
        last = p0 + nvalid - 1
        if last >= self.max_ctx or p0 < 0:
            raise k3.ContextBoundsError(f'MTP: positions {p0} .. {last} outside [0, {self.max_ctx})')
        k3.check_span(p0, Tm, self.kv_rows, what='MTP launch (KV rows incl. padding)')
        for t in range(Tm): st.h_emb[t] = self.tgt._emb_row(int(toks[min(t, nvalid - 1)]))
        st.h_i32[16] = p0; st.h_i32[1] = 0
        D, HC = self.D, self.hc
        fb = self._force_bail
        if fb is not None and self.sets.get(fb[0]) is st:   # test hook: arm the tiny wait bound on the nth launch of this set
            fb[1] -= 1
            if fb[1] == 0:
                for c in self.cards: st.E[c].limit_us = 1
        for attempt in range(self.max_retries + 1):
            st.h_i32[17] = st.token * 2; st.token += 1
            for c in self.cards:
                with torch.cuda.device(self.devs[c]):
                    st.blk[c].copy_(st.hstage, non_blocking=True)   # every attempt: it carries the new epoch
                    if attempt == 0:
                        # the X inputs once, before the first attempt: the tape never writes X, but hsrc may be this set's own
                        # res (chain step >= 2), which an attempt overwrites (eh_proj, hc_post) before a bail is seen
                        X4 = st.X[c].view(Tm, HC, 2, D)
                        X4[:, :, 0, :].copy_(st.erow[c].view(Tm, 1, D).expand(Tm, HC, D))
                        h = hsrc(c).view(nvalid, HC, D)
                        X4[:nvalid, :, 1, :].copy_(h)
                        if nvalid < Tm: X4[nvalid:, :, 1, :].copy_(h[nvalid - 1:nvalid].expand(Tm - nvalid, HC, D))
                    st.E[c].run(check=False)
            for c in self.cards:
                with torch.cuda.device(self.devs[c]):
                    if st.lg:
                        torch.max(st.lg[c], dim=1, out=st.d_max[c])
                        st.h_max[c][0].copy_(st.d_max[c][0], non_blocking=True); st.h_max[c][1].copy_(st.d_max[c][1], non_blocking=True)
                    st.h_st[c].copy_(st.E[c].sync[2 * k3.MAXSEG:2 * k3.MAXSEG + k3.STATUS_WORDS], non_blocking=True)
                    st.ev[c].record()
            for c in self.cards: st.ev[c].synchronize()
            self.stats['launches'] += 1
            oob = [f'card {c}: {b}' for c in self.cards for b in [k3.bounds_status(st.h_st[c].tolist())] if b]
            if oob:
                for c in self.cards:
                    with torch.cuda.device(self.devs[c]): torch.cuda.synchronize(); st.E[c].sync.zero_()
                self.poisoned = f'the device bound guard caught a position at MTP launch {p0}: ' + '; '.join(oob)
                raise k3.ContextBoundsError(self.poisoned)
            bad = []
            for c in self.cards:
                s_ = st.h_st[c].tolist()
                if s_[1]:
                    with torch.cuda.device(self.devs[c]):
                        torch.cuda.synchronize(); bad.append(f'card {c}: segment {s_[2]} (seen {s_[3]}); {st.E[c].stall_report()}'); st.E[c].sync.zero_()
            if not bad: break
            self.bails.append((p0, attempt, bad))
            print(f'MTP: a wait gave up at position {p0} (attempt {attempt + 1}), running the launch again: ' + '; '.join(bad), file=sys.stderr, flush=True)
            if fb is not None and fb[1] == 0:   # test hook: the forced bail's tiny wait bound only on the first attempt
                for c in self.cards: st.E[c].limit_us = self.limit_us
        else:
            raise RuntimeError(f'MTP: a wait gave up at position {p0} in {self.max_retries + 1} attempts')
        if fb is not None and fb[1] == 0:        # (also when the armed attempt did not bail)
            for c in self.cards: st.E[c].limit_us = self.limit_us
            self._force_bail = None
        if not st.lg: return None
        hr = st.head_rows
        if self.local: return [int(st.h_max[c][1][t]) for c in self.cards for t in range(hr)]   # (solo: card's own rows)
        (v0, x0), (v1, x1) = [(v.numpy(), x.numpy()) for v, x in (st.h_max[0], st.h_max[1])]
        vh = self.tgt.vh
        return [int(x0[t]) if (v0[t] >= v1[t] or v0[t] != v0[t]) else vh + int(x1[t]) for t in range(hr)]

    def force_bail(self, which='catch', nth=1):
        """test hook: the nth launch of set `which` from now (chain: nth = i is chain step i of the next draft) bails (a 1 us
        wait bound on its first attempt) and is run again"""
        assert which in self.sets and nth >= 1, (which, nth)
        self._force_bail = [which, nth]

    def catch_up(self, toks, p_end, draft=False):
        """rows n_done .. p_end - 1 (token p with target h p - 1) in launches of the catch-up set. draft: the final launch
        ends at row p_end - 1 (rows p_end - Tm .. p_end - 1: rows below n_done are recomputed from the same inputs, the same
        bytes) and its head row gives the greedy token after it (returned: the next draft); else launches start at n_done
        (the last one padded) and nothing is returned"""
        st = self.sets['catch']; Tm = st.rows
        if p_end - 1 > self.h_valid:
            raise RuntimeError(f'MTP catch_up: row {p_end - 1} needs the target h of position {p_end - 2}, only {self.h_valid} known')
        assert p_end <= len(toks), (p_end, len(toks))
        stop = max(p_end - Tm, 0) if draft else p_end
        while self.n_done < stop:
            p0 = self.n_done; n = min(Tm, stop - p0)
            self._launch(st, p0, toks[p0:p0 + n], lambda c, p0=p0, n=n: self.H[c][p0:p0 + n], n)
            self.n_done = p0 + n; self.stats['catch_rows'] += n
        if not draft: return None
        if p_end < Tm: raise RuntimeError(f'MTP catch_up: a draft needs at least {Tm} rows ({p_end})')
        p0 = p_end - Tm
        out = self._launch(st, p0, toks[p0:p_end], lambda c: self.H[c][p0:p_end], Tm)
        self.stats['catch_rows'] += p_end - max(self.n_done, p0)
        self.n_done = p_end; self.last = (st, Tm - 1)
        return out[0]

    def prefill(self, toks, n_h):
        """after a prompt pass: the target h of positions < n_h is known; pair every prompt row but the last (draft() runs it
        and takes draft 1 from it)"""
        p_end = min(n_h + 1, len(toks) - 1)
        if p_end > self.n_done: self.catch_up(toks, p_end)

    def draft(self, toks, k, sampler=None):
        """k drafts after toks (len L: the bonus token at L - 1): the catch-up of rows n_done .. L - 1 gives draft 1, then
        k - 1 chain steps (draft i at L - 1 + i with the block's own res). sampler: None = greedy argmax (the hook for
        sampled drafting later: it would take the logits rows)"""
        assert sampler is None, 'MTP draft: only greedy drafting so far (the sampling hook is reserved)'
        L = len(toks)
        assert self.n_done <= L - 1, (self.n_done, L)
        if L - 1 + max(k - 1, 0) >= self.max_ctx:
            raise k3.ContextBoundsError(f'MTP draft: positions up to {L - 1 + k - 1} past max_ctx {self.max_ctx}')
        d = [self.catch_up(toks, L, draft=True)]
        if k > 1:
            ch = self.sets.get('chain')
            if ch is None: raise RuntimeError('MTP draft: k > 1 needs the chain set (chain=True and target ntok > 2)')
            for i in range(1, k):
                st, r = self.last
                d.append(self._launch(ch, L - 1 + i, [d[-1]], lambda c, st=st, r=r: st.res[c].view(st.rows, -1)[r:r + 1])[0])
                self.last = (ch, 0); self.stats['chain_steps'] += 1
        return d[:k]


class MTPDrafter:
    """MTPK3 with caminus.decode.spec.generate's drafter interface"""
    def __init__(self, mtp):
        self.m = mtp
        self.max_pos = mtp.max_ctx

    def inject(self, i0, feats):
        self.m.take_h(i0)            # the target's h_nextn of the pass just verified

    def prefill(self, toks, n_h):
        self.m.prefill(toks, n_h)

    def reset(self):
        self.m.reset()

    def snapshot(self):
        return self.m.snapshot()

    def restore(self, s):
        self.m.restore(s)

    def __call__(self, toks, k):
        L = len(toks)
        if L - 2 >= self.m.h_valid or L < self.m.Tc:   # no target h of position L - 2 yet (a prompt tail shorter than T), or
            # fewer rows than one catch-up launch (a prompt shorter than T)
            if L - 1 > self.m.n_done: self.m.catch_up(toks, min(L - 1, self.m.h_valid + 1))
            return [int(toks[-1])] * k   # filler drafts (verification rejects what is wrong)
        return self.m.draft(toks, k)


# -- per-content policy hook --------------------------------------------------------------------------------------------
# which method a generation uses, by content class: 'plain' (one-token decoding, no drafter), 'dflash', 'mtp'. The default is
# DFlash everywhere (impl:dflash-ring, the user's choice): it runs at the model's full context (a fixed 4096-row ring, 262,144
# + DFlash fits both cards) and drafts exactly there, while MTP is parked: its H rows and KV grow with max_ctx (refused at
# 131,072: the XTX over its cap) and past 2051 its own layer attends densely (drafts inexact, output exact). MTP stays
# available by an explicit table entry. Measured on the Coder at short context (smoke 20261103-120414, T = 4, 128 greedy
# tokens, max_ctx 2048, rep 1 / rep 2): DFlash 155.6-155.8 / 202.2-202.5 / 122.6-123.0 tok/s (prose / code / shell), MTP
# 147.4-148.0 / 188.7-188.9 / 134.7-136.0: MTP is faster on shell (+12-13 tok/s) at short context. A caller passes its own table; the per-pass EMA policy
# (drafter spec S5) plugs in here later. A pick whose drafter is not loaded falls back (FALLBACK, then 'plain'), for the
# default table and a caller's alike: a request never fails on a drafter that was not loaded (MTP parked, a drafter refused
# by the VRAM check); PolicyDrafter.fallback records the substitution.
DEFAULT_POLICY = {'prose': 'dflash', 'code': 'dflash', 'shell': 'dflash', None: 'dflash'}
FALLBACK = ('dflash', 'mtp')   # a pick that is not loaded: the first of these that is, else 'plain'


def choose(content, policy=None, loaded=None):
    """the method for content class `content`: the table's entry for it, else the table's None entry, else 'plain' (policy
    None: DEFAULT_POLICY). loaded: the names of the drafters available (None: not checked); a pick that is not among them
    falls back to the first of FALLBACK that is, else 'plain' (never a KeyError)"""
    p = DEFAULT_POLICY if policy is None else policy
    m = p.get(content, p.get(None, 'plain'))
    assert m in ('plain', 'dflash', 'mtp'), m
    if loaded is not None and m != 'plain' and m not in loaded:
        m = next((x for x in FALLBACK if x in loaded), 'plain')
    return m


class PolicyDrafter:
    """one spec drafter over the loaded drafters (drafters: {'dflash': DFlashK3Drafter, 'mtp': MTPDrafter}, any subset,
    none too) that delegates each generation to the method chosen for its content class (select(); until the first
    select(), the table's default, content None). Every drafter still sees the target's per-pass data (inject / prefill),
    and in each pass the idle ones take what they need, so any of them can take over at the next select(): MTP pairs its
    rows up to the last known target h, DFlash flushes its pending inject (one inject launch per pass while another method
    drafts; without it the next pass's inject is refused, 'still pending').
    Forwarded to every drafter: reset, inject, prefill, flush (spec.loop's k = 0 passes), snapshot / restore
    (caminus.decode.longctx: one state per drafter; the active method is the request's choice, not part of the sequence state,
    and restore leaves it as it is).
    device_accept / generate: when the active drafter runs its own loop (DFlashK3Drafter with device_accept), spec.generate
    hands the generation to it; the idle drafters are reset first (that loop resets the engine and feeds only its own
    drafter), so afterwards they hold nothing of that sequence: harmless, since spec.generate returns no state to continue
    from, and the next spec.generate / longctx.fill resets them again.
    'plain' (picked, or the fallback with nothing loaded): filler drafts, the last token k times (as caminus.decode.longctx.Repeat):
    spec over the T-token engine is then plain greedy decoding (exact; a one-token engine is faster per token, if the
    caller has one, which is what select() returning 'plain' tells it)"""
    def __init__(self, drafters, policy=None):
        self.d, self.policy = dict(drafters), policy
        self.max_pos = min((getattr(v, 'max_pos', 1 << 30) for v in self.d.values()), default=None)   # (None: no drafter rows)
        self.fallback = None
        self.select(None)

    def select(self, content):
        """the method for this generation's content class (choose(); a pick that is not loaded falls back to a loaded
        drafter or 'plain'); self.fallback = (the table's pick, the method used) when they differ, else None"""
        want = choose(content, self.policy)
        self.active = choose(content, self.policy, loaded=self.d)
        self.fallback = (want, self.active) if want != self.active else None
        return self.active

    def reset(self):
        for v in self.d.values():
            if hasattr(v, 'reset'): v.reset()

    def inject(self, i0, feats):
        for v in self.d.values():
            if hasattr(v, 'inject'): v.inject(i0, feats)

    def skip_inject(self, p):
        for v in self.d.values():
            if hasattr(v, 'skip_inject'): v.skip_inject(p)

    def prefill(self, toks, n_h):
        for v in self.d.values():
            if hasattr(v, 'prefill'): v.prefill(toks, n_h)

    def flush(self):
        for v in self.d.values():
            if hasattr(v, 'flush'): v.flush()

    def snapshot(self):
        return {n: v.snapshot() for n, v in self.d.items() if hasattr(v, 'snapshot')}

    def restore(self, s):
        have = {n for n, v in self.d.items() if hasattr(v, 'snapshot')}
        if set(s) != have: raise RuntimeError(f'PolicyDrafter.restore: a snapshot of drafters {sorted(s)}, these are {sorted(have)}')
        for n, x in s.items(): self.d[n].restore(x)

    @property
    def device_accept(self):
        v = self.d.get(self.active)
        return bool(getattr(v, 'device_accept', False)) and hasattr(v, 'generate')

    def generate(self, m, prompt, n_new, stats=None, prefill=None):
        """spec.generate's hand-off: the active drafter's own loop (device_accept), the idle drafters reset first; without one,
        spec.generate's generic loop over this wrapper. prefill (caminus.prefill.coder.Prefill) forwarded: a batched prefill
        refuses an MTP drafter among the loaded ones (it needs per-pass target rows a prefill does not produce)"""
        if not self.device_accept:
            from . import spec
            return spec.generate(m, prompt, n_new, self, stats, prefill=prefill)   # (device_accept false: the generic loop, no recursion)
        v = self.d[self.active]
        if prefill is not None: prefill.check_drafter(self)   # (before any reset: an idle MTP could not take over afterwards)
        for o in self.d.values():
            if o is not v and hasattr(o, 'reset'): o.reset()
        return v.generate(m, prompt, n_new, stats) if prefill is None else v.generate(m, prompt, n_new, stats, prefill=prefill)

    def __call__(self, toks, k):
        for n, v in self.d.items():
            if n == self.active: continue
            if isinstance(v, MTPDrafter):
                # the idle MTP keeps its rows paired up to the last row whose target h is known (row p needs h of p - 1)
                p_end = min(len(toks), v.m.h_valid + 1)
                if v.m.n_done < p_end: v.m.catch_up(toks, p_end)
            elif hasattr(v, 'flush'): v.flush()   # (the idle DFlash: its pending inject, before the next target pass)
        v = self.d.get(self.active)
        if v is None: return [int(toks[-1])] * k   # 'plain': filler (longctx.Repeat), the verify decides every token
        return v(toks, k)


def smoke_factory(cfg, T, max_ctx=2048):
    """tools/k3_smoke.py's spec:mtp stage: the Coder target with ntok = T and its MTP drafter (T - 1 drafts per pass)"""
    # the MTP layer's VRAM reserved in the target's check: a config over the caps is refused before the target loads
    m = FlashNextTP(cfg['gguf'], ntok=T, max_ctx=max_ctx, vram_drafters=('mtp',))
    mt = MTPK3(m, cfg.get('mtp', SIDECAR))
    return m, MTPDrafter(mt), dict(sidecar=cfg.get('mtp', SIDECAR), T=T, catch_rows=mt.Tc, chain='chain' in mt.sets)
