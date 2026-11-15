"""The DFlash 2 drafter of the dense qwen35 line on Caminus's own code (incoai's Qwen3.8-27B-DFlash2 Q8_0 on Swift 1.5:
kernels/k3_q35_dflash.hip; the float64 oracle: caminus.ref_dflash). One card, the target engine's card: the drafter reads
the target's Q6_K head and Q4_K embedding where the target engine holds them (Dense35Own.head / emb_dev), nothing copied.

Two tapes of T = 4 rows on the drafter module:
  inject (the target's features for 4 positions: Dense35Verify / Q35OwnPrefill feature_layers, [5, 4, D] -> [4, 5 D] fp16
    on the host stream): fc (Q8_0, K 25,600) -> rms * enc_norm -> per drafter layer k (Q8_0) with its norm + NeoX RoPE over
    the whole head and v (Q8_0) into the layer's f16 ring at those positions (row p & 4095).
  draft (the block [last, MASK x 3] at positions n0 .. n0 + 3): per layer rms * attn_norm -> dyn_a (attn_conv_proj) ->
    conv side 0 -> q / k / v -> q / k norm + RoPE -> the block's k / v into the ring -> attention (non-causal, window 2048
    per query, over the ring) -> o -> conv side 1 + the residual -> rms * ffn_norm -> dyn_f -> conv side 0 -> gate / up ->
    SwiGLU -> down -> conv side 1 + the residual; then rms * output_norm -> the target's head [4, vocab] -> top-16 of rows
    1 .. 3 in two vocabulary halves (OP_DF_TOPK scan + merge) -> selector_hidden gate -> OP_DF_SELECT: the 3 drafts.
The ring holds 4096 rows (a power of two >= window + block): position p in row p & 4095, so any position addresses inside it.
An inject of fewer than 4 rows pads with copies of its last row at the next positions; the next draft (at the position after
the last real row) rewrites those rows before it reads them."""
import numpy as np
import torch

from .. import ggufio
from ..device import capacity
from . import k3

MODULE = 'k3_q35_dflash'
MODULE_TCAP = 'k3_q35_tcap_dflash'   # (ThinkingCap's: its IQ2_S embedding and Q8_0 head, kernels/k3_q35_tcap_dflash.hip)
T = 4                 # rows per tape: the drafter's block (1 anchor + 3 drafts) and the inject chunk (OWN_MT)
RING = 4096           # f16 ring rows per drafter layer (>= window 2048 + T)
ROWS = 512            # the batched prefill inject's rows per pass (inject_rows: a chunk in passes of this many; ~56 MiB of
                      # buffers, against ~224 MiB for a whole 2,048-position chunk on a card the drafter already crowds)
CHUNK = 4096          # = DF_CHUNK in kernels/k3_df_own_ops.h (OP_DF_TOPK's scan)
MERGE_MAX = 640       # OP_DF_TOPK merge: pairs per row held in LDS


def _seg(E, op, ntask, dep, rpt, W=None, gs=None, a=None, asum=None, y=None, rs=0.0, N=0, K=0, out_f32=1, work=0, pad1=0, pad2=0,
         aux=None, keep=()):
    p = lambda t: None if t is None else (t if isinstance(t, int) else t.data_ptr())
    sg = k3.Seg(op, ntask, dep, rpt, p(W), p(gs), p(a), p(asum), p(y), float(rs), N, K, out_f32, work, -1, pad1, pad2)
    if aux is not None: sg.aux = p(aux)
    E.keep += [t for t in keep if t is not None and not isinstance(t, int)]
    E.segs.append(sg)
    return len(E.segs) - 1


def df_conv(E, h, dyn, base, out, C, side, ks, gs, res=None, out16=None, dep=-1, rpt=1024):
    """OP_DF_CONV over h [T, C] (flat buffers: C given)"""
    assert h.numel() == E.T * C and out.numel() == E.T * C and dyn.numel() == E.T * 2 * ks * (C // gs) and base.numel() == 2 * ks * C
    assert C % gs == 0 and rpt % 256 == 0 and (res is None or res.numel() == E.T * C) and (out16 is None or out16.numel() == E.T * C)
    return _seg(E, k3.OP_DF_CONV, (E.T * C + rpt - 1) // rpt, dep, rpt, W=base, gs=dyn, a=h, asum=out16, y=out, N=E.T * C, K=C,
                work=ks, pad1=side, pad2=gs, aux=res, keep=(h, dyn, base, out, res, out16))


def df_topk_scan(E, logits, row_len, col0, ncols, row0, out, K, dep=-1):
    """the top K of columns [col0, col0 + ncols) of rows row0 .. T-1 of logits [T, row_len], ids offset by col0"""
    nch = (ncols + CHUNK - 1) // CHUNK; nr = E.T - row0
    assert out.numel() >= nr * nch * K * 2 and 1 <= K <= 16 and nch * K <= MERGE_MAX and col0 + ncols <= row_len
    # (Seg.a = row 0's slice; the scan reads row r at a + r * out_f32 (the full row) and ncols ids of it)
    return _seg(E, k3.OP_DF_TOPK, nr * nch, dep, 1, a=logits[col0:], y=out, N=ncols, K=K, out_f32=row_len, work=0, pad1=nch,
                pad2=row0, aux=int(col0), keep=(logits, out)), nch


def df_topk_merge(E, pairs, nrows, n, out, K, dep=-1, parts=4):
    assert pairs.numel() >= nrows * n * 2 and out.numel() >= nrows * K * 2 and n <= MERGE_MAX and parts >= 1
    return _seg(E, k3.OP_DF_TOPK, nrows * parts, dep, 1, a=pairs, y=out, N=n, K=K, work=1, pad1=parts, keep=(pairs, out))


def df_select(E, cand, halves, sel_prev, sel_next, gate, anchor, out, rank, kc, row_bytes, vocab, dep=-1):
    k = E.T - 1
    assert cand.numel() >= halves * k * kc * 2 and gate.numel() == E.T * rank and out.numel() >= k + 3 * k * kc
    assert rank % 32 == 0 and rank <= 768 and kc <= 16 and k <= 8 and 512 + 2 * k * halves * kc <= 1280 and 512 + rank <= 1280
    assert sel_prev.numel() == sel_next.numel() == vocab * row_bytes and 0 < vocab < (1 << 31)
    return _seg(E, k3.OP_DF_SELECT, 1, dep, 1, W=sel_prev, gs=sel_next, a=cand, asum=anchor, y=out, N=k, K=rank, work=row_bytes,
                pad1=halves, pad2=kc, out_f32=vocab, aux=gate, keep=(cand, sel_prev, sel_next, gate, anchor, out))


def kv_ring(E, k, v, cache, pos, dep=-1):
    M = k.numel() // E.T
    assert cache.numel() == RING * 2 * M and cache.dtype == torch.float16 and v.numel() == k.numel()
    return _seg(E, k3.OP_KV_APPEND, E.T, dep, 1, W=v, a=k, asum=pos, y=cache, N=M, aux=RING - 1, keep=(k, v, cache, pos))


def attn_df(E, q, cache, pos, out, out16, heads, kv_heads, dim, scale, window, dep=-1):
    assert dim <= 512 and q.numel() == E.T * heads * dim and out.numel() == q.numel() and out16.numel() == q.numel()
    assert cache.numel() == RING * 2 * kv_heads * dim and RING >= window + E.T
    return _seg(E, k3.OP_ATTN_DEC, E.T * heads, dep, 1, W=cache, gs=out16, a=q, asum=pos, y=out, rs=scale, N=E.T * heads, K=dim,
                pad1=kv_heads, pad2=3, aux=int(window) | ((RING - 1) << 32), keep=(q, cache, pos, out, out16))


class Dflash35:
    @staticmethod
    def need_mib(path, vocab, chunk=0):
        """MiB the drafter adds on the target's card: its weights (the file's tensors), the rings and the tapes' buffers;
        chunk > 0: the batched prefill inject's buffers for chunks of that many positions (inject_rows)"""
        F = ggufio.GGUFFile(path)
        try:
            m = F.meta; hp = lambda k: m[f'dflash.{k}']
            D, ffn, nl = int(hp('embedding_length')), int(hp('feed_forward_length')), int(hp('block_count'))
            M = int(hp('attention.head_count_kv')) * int(hp('attention.key_length')); QH = int(hp('attention.head_count')) * int(hp('attention.key_length'))
            wb = sum(t.n_bytes for t in F.tensors)
            ring = nl * RING * 2 * M * 2
            vocab = int(vocab or 0)
            if vocab <= 0: raise ValueError('Dflash35.need_mib: the target\'s vocabulary size')
            rank, kc = int(hp('selector_rank')), int(hp('selector_top_k'))
            nt = len(hp('target_layers')); CP = 2 * int(hp('conv_kernel_size')) * (D // int(hp('conv_group_size')))
            nch = (vocab // 2 + CHUNK - 1) // CHUNK
            buf = (T * (6 * D + 3 * QH + 3 * M + 2 * ffn + CP + rank + 3 * nl * M) * 4 + T * (2 * D + QH + ffn + nt * D) * 2
                   + T * vocab * 4 + 32 * 4 + (2 * (T - 1) * nch * kc * 2 + 2 * (T - 1) * kc * 2 + (T - 1) * (1 + 3 * kc)) * 4)
            # inject_rows: the chunk's features as fp16 [chunk, nt D], fc's output and its norm (fp32 + fp16), k / v / k-roped
            # (fp32) shared by the layers in turn
            ib = min(int(chunk), ROWS) * (nt * D * 2 + D * 4 * 2 + D * 2 + 3 * M * 4) if chunk else 0
            return (wb + ring + buf + ib) / 2 ** 20 + 8
        finally: F.close()

    def __init__(self, path, eng, vram_check=True, per_wgp=None, chunk=0):
        """eng: the target Dense35Own, whose head and embedding the drafter reads in place: Swift 1.5 (Q6_K head, Q4_K
        embedding: kernels/k3_q35_dflash.hip) or ThinkingCap (Q8_0 head, IQ2_S embedding: kernels/k3_q35_tcap_dflash.hip)"""
        gsq = eng.module in k3.Q35R_MODULES   # (the plain Qwen3.8-27B GSQ-RCO files: Q4_K head, IQ2_S embedding)
        want = ('Q4_K', 'IQ2_S') if gsq else ('Q8_0', 'IQ2_S') if eng.tcap else ('Q6_K', 'Q4_K')
        if (eng.head['fmt'], eng.emb.type) != want:
            line = 'GSQ' if gsq else 'ThinkingCap' if eng.tcap else 'Swift'
            raise ValueError(f'Dflash35: head {eng.head["fmt"]}, embedding {eng.emb.type} (the {line} line runs {want})')
        self.module = k3.Q35RDF_MODULES[0] if gsq else MODULE_TCAP if eng.tcap else MODULE
        self.F = F = ggufio.GGUFFile(path); self.eng = eng
        m = F.meta; hp = lambda k: m[f'dflash.{k}']
        self.nl, self.D, self.ffn = int(hp('block_count')), int(hp('embedding_length')), int(hp('feed_forward_length'))
        self.nh, self.nkv, self.hd = int(hp('attention.head_count')), int(hp('attention.head_count_kv')), int(hp('attention.key_length'))
        self.eps, self.base = float(hp('attention.layer_norm_rms_epsilon')), float(hp('rope.freq_base'))
        self.ks, self.gs = int(hp('conv_kernel_size')), int(hp('conv_group_size'))
        self.rank, self.topk, self.window = int(hp('selector_rank')), int(hp('selector_top_k')), int(hp('attention.sliding_window'))
        self.layers_tgt = [int(v) for v in hp('target_layers')]
        self.mask = int(m['tokenizer.ggml.mask_token_id'])
        if bool(hp('attention.causal')) or int(hp('block_size')) < T or self.D != eng.D:
            raise ValueError('Dflash35: a non-causal DFlash 2 drafter with a block of >= 4 rows on the target\'s n_embd')
        self.vocab = eng.vocab
        self.TT = {t.name: t for t in F.tensors}
        self.path, self.T, self.ring, self.dx = path, T, RING, eng.dev
        self.token = 0
        self.max_pos = int(eng.max_ctx) + T   # (a draft block reaches T - 1 past the target's last position)
        self.row_bytes = 2 * self.nkv * self.hd * 2   # (one layer's ring row: K | V f16)
        self.dev = eng.dev
        self.chunk = int(chunk)   # (the batched prefill inject's chunks: inject_rows, in passes of min(chunk, ROWS) rows; 0: none)
        self.mib = self.need_mib(path, eng.vocab, self.chunk)
        if vram_check: capacity.check([eng.dev], [self.mib], f'Dflash35 drafter ({self.mib:.0f} MiB)')
        self.per_wgp = int(per_wgp) if per_wgp else eng.per_wgp
        self.theta_scale = float(np.float32(self.base) ** np.float32(-2.0 / self.hd))
        with torch.cuda.device(self.dev):
            self._load(); self._build()
            if self.chunk: self._rows_buffers()

    # -- weights -------------------------------------------------------------------------------------------------------
    def _w(self, name, rows, cols):
        t = self.TT[name]; R, _ = t.rows()
        if t.type != 'Q8_0' or R != rows or int(t.shape[0]) != cols: raise ValueError(f'Dflash35: {name} is {t.type} [{R}, {t.shape[0]}], expect Q8_0 [{rows}, {cols}]')
        return dict(name=name, fmt='Q8_0', rows=R, K=cols, w=torch.from_numpy(np.ascontiguousarray(t.pread(0, t.n_bytes))).to('cuda'))

    def _f32(self, name, n):
        t = self.TT[name]
        if t.type != 'F32' or t.n_values != n: raise ValueError(f'Dflash35: {name} is {t.type} {list(t.shape)}, expect F32 [{n}]')
        return torch.from_numpy(t.pread(0, t.n_bytes).view(np.float32).copy()).to('cuda')

    def _load(self):
        D, ffn, QH, M, nl = self.D, self.ffn, self.nh * self.hd, self.nkv * self.hd, len(self.layers_tgt)
        CP = 2 * self.ks * (D // self.gs)
        self.fc = self._w('fc.weight', D, nl * D)
        self.enc_norm, self.out_norm = self._f32('enc.output_norm.weight', D), self._f32('output_norm.weight', D)
        self.sel_hidden = self._w('selector_hidden.weight', self.rank, D)
        sp, sn = self.TT['selector_predecessor.weight'], self.TT['selector_successor.weight']
        self.sel_rb = self.rank // 32 * 34
        self.sel_prev = torch.from_numpy(np.ascontiguousarray(sp.pread(0, sp.n_bytes))).to('cuda')
        self.sel_next = torch.from_numpy(np.ascontiguousarray(sn.pread(0, sn.n_bytes))).to('cuda')
        self.L = []
        for il in range(self.nl):
            b = f'blk.{il}.'
            self.L.append(dict(attn_norm=self._f32(b + 'attn_norm.weight', D), ffn_norm=self._f32(b + 'ffn_norm.weight', D),
                               qn=self._f32(b + 'attn_q_norm.weight', self.hd), kn=self._f32(b + 'attn_k_norm.weight', self.hd),
                               q=self._w(b + 'attn_q.weight', QH, D), k=self._w(b + 'attn_k.weight', M, D),
                               v=self._w(b + 'attn_v.weight', M, D), o=self._w(b + 'attn_output.weight', D, QH),
                               fg=self._w(b + 'ffn_gate.weight', ffn, D), fu=self._w(b + 'ffn_up.weight', ffn, D),
                               fd=self._w(b + 'ffn_down.weight', D, ffn),
                               ca_proj=self._w(b + 'attn_conv_proj.weight', CP, D), cf_proj=self._w(b + 'ffn_conv_proj.weight', CP, D),
                               ca_base=self._f32(b + 'attn_conv_base', 2 * self.ks * D), cf_base=self._f32(b + 'ffn_conv_base', 2 * self.ks * D),
                               ring=torch.zeros(RING * 2 * M, dtype=torch.float16, device='cuda')))

    # -- tapes ---------------------------------------------------------------------------------------------------------
    def _engine(self):
        E = k3.Engine(module=self.module, grid=k3.resident_grid(self.module, self.per_wgp), T=T, limit_us=self.eng.limit_us)
        return E

    def _mv(self, E, W, x16, out, dep):
        return E.own_matvec16(W['w'], 'Q8_0', W['rows'], W['K'], x16, out, rows_per_task=self.eng.rpt16, dep=dep, tokens=T)

    def _build(self):
        eng, D, ffn, QH, M, nl = self.eng, self.D, self.ffn, self.nh * self.hd, self.nkv * self.hd, len(self.layers_tgt)
        CP, V = 2 * self.ks * (D // self.gs), self.vocab
        B = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt, device='cuda')
        f16 = torch.float16
        s = self.buf = dict(x=B(T * D), h=B(T * D), h16=B(T * D, f16), c=B(T * D), c16=B(T * D, f16), dyn=B(T * CP),
                            q=B(T * QH), k=B(T * M), v=B(T * M), qo=B(T * QH), ko=B(T * M), att=B(T * QH), att16=B(T * QH, f16),
                            blk=B(T * D), g=B(T * ffn), u=B(T * ffn), a16=B(T * ffn, f16), gate=B(T * self.rank),
                            logits=B(T * V), feat16=B(T * nl * D, f16), y=B(T * D), stage=B(32, torch.int32))
        s['ik'], s['iv'], s['iko'] = ([B(T * M) for _ in range(self.nl)] for _ in range(3))
        self.hstage = torch.zeros(32, dtype=torch.int32, pin_memory=True)
        st = s['stage']; self.toks, self.pos, self.anchor = st[16:16 + T], st[16 + T:17 + T], st[16:17]
        scale = 1.0 / float(np.sqrt(self.hd))
        # inject
        E = self.Ei = self._engine()
        prev = self._mv(E, self.fc, s['feat16'], s['y'], -1)
        prev = E.own_add_rms16(s['y'], self.enc_norm, s['h'], s['h16'], self.eps, dep=prev, tokens=T)
        # (the layers' k / v run side by side, all after the norm: each layer has its own scratch rows, or they overwrite
        #  each other's before the ring append reads them)
        for il, d in enumerate(self.L):
            ki, vi, koi = s['ik'][il], s['iv'][il], s['iko'][il]
            k_ = self._mv(E, d['k'], s['h16'], ki, prev); v_ = self._mv(E, d['v'], s['h16'], vi, prev)
            kn = E.qk_nr(ki, self.hd, self.nkv, self.hd, d['kn'], self.pos, koi, self.hd, self.theta_scale, self.eps, dep=k_)
            E.also(kv_ring(E, koi, vi, d['ring'], self.pos, dep=kn), v_)
        E.build()
        # draft
        E = self.Ed = self._engine()
        prev = E.own_emb(eng.emb_dev, eng.emb.type, V, D, self.toks, s['x'], tokens=T)
        for d in self.L:
            n1 = E.own_add_rms16(s['x'], d['attn_norm'], s['h'], s['h16'], self.eps, dep=prev, tokens=T)
            dy = self._mv(E, d['ca_proj'], s['h16'], s['dyn'], n1)
            c0 = df_conv(E, s['h'], s['dyn'], d['ca_base'], s['c'], D, 0, self.ks, self.gs, out16=s['c16'], dep=dy)
            q_ = self._mv(E, d['q'], s['c16'], s['q'], c0); k_ = self._mv(E, d['k'], s['c16'], s['k'], c0)
            v_ = self._mv(E, d['v'], s['c16'], s['v'], c0)
            qd = E.qk_nr(s['q'], self.hd, self.nh, self.hd, d['qn'], self.pos, s['qo'], self.hd, self.theta_scale, self.eps, dep=q_)
            kd = E.qk_nr(s['k'], self.hd, self.nkv, self.hd, d['kn'], self.pos, s['ko'], self.hd, self.theta_scale, self.eps, dep=k_)
            ap = E.also(kv_ring(E, s['ko'], s['v'], d['ring'], self.pos, dep=kd), v_)
            at = E.also(attn_df(E, s['qo'], d['ring'], self.pos, s['att'], s['att16'], self.nh, self.nkv, self.hd, scale,
                                self.window, dep=ap), qd)
            o_ = self._mv(E, d['o'], s['att16'], s['blk'], at)
            c1 = df_conv(E, s['blk'], s['dyn'], d['ca_base'], s['x'], D, 1, self.ks, self.gs, res=s['x'], dep=o_)   # x = ffn_inp
            n2 = E.own_add_rms16(s['x'], d['ffn_norm'], s['h'], s['h16'], self.eps, dep=c1, tokens=T)
            dy = self._mv(E, d['cf_proj'], s['h16'], s['dyn'], n2)
            c2 = df_conv(E, s['h'], s['dyn'], d['cf_base'], s['c'], D, 0, self.ks, self.gs, out16=s['c16'], dep=dy)
            g_ = self._mv(E, d['fg'], s['c16'], s['g'], c2); u_ = self._mv(E, d['fu'], s['c16'], s['u'], c2)
            sw = E.also(E.own_swiglu16(s['g'], s['u'], s['a16'], dep=u_), g_)
            f_ = self._mv(E, d['fd'], s['a16'], s['blk'], sw)
            prev = df_conv(E, s['blk'], s['dyn'], d['cf_base'], s['x'], D, 1, self.ks, self.gs, res=s['x'], dep=f_)
        nf = E.own_add_rms16(s['x'], self.out_norm, s['h'], s['h16'], self.eps, dep=prev, tokens=T)
        if eng.head['fmt'] in ('Q6_K', 'Q4_K'):   # (Swift's Q6_K / the GSQ files' Q4_K: one-round-trip T-token tiles; ThinkingCap's Q8_0: the rows tile)
            hd_ = E.own_matvec_1rt(eng.head['w'], eng.head['fmt'], V, D, s['h16'], s['logits'], rows_per_task=eng.rpt, dep=nf, tokens=T,
                                   x_stride=D, out_stride=V)
        else:
            hd_ = E.own_matvec16(eng.head['w'], 'Q8_0', V, D, s['h16'], s['logits'], rows_per_task=eng.rpt16, dep=nf, tokens=T)
        ga = self._mv(E, self.sel_hidden, s['h16'], s['gate'], nf)
        half = V // 2; kc = self.topk; k = T - 1
        nch = (half + CHUNK - 1) // CHUNK
        s['pairs'] = B(2 * k * nch * kc * 2, torch.int32); s['cand'] = B(2 * k * kc * 2); s['out'] = B(k + 3 * k * kc, torch.int32)
        merges = []
        for hh in range(2):
            sc, _ = df_topk_scan(E, s['logits'], V, hh * half, half if hh == 0 else V - half, 1,
                                 s['pairs'][hh * k * nch * kc * 2:], kc, dep=hd_)
            merges.append(df_topk_merge(E, s['pairs'][hh * k * nch * kc * 2:], k, nch * kc, s['cand'][hh * k * kc * 2:], kc, dep=sc))
        sl = df_select(E, s['cand'], 2, self.sel_prev, self.sel_next, s['gate'], self.anchor, s['out'], self.rank, kc, self.sel_rb,
                       V, dep=merges[0])
        E.also(merges[1], ga)                    # (the gate is done before the select: merge 1 waits for it, the select for merge 1)
        E.also(sl, merges[1])
        E.build()

    # -- running -------------------------------------------------------------------------------------------------------
    def _launch(self, E):
        self.token += 1   # (the launch epoch: the prompt cache's live state reads it through the adapter)
        self.buf['stage'].copy_(self.hstage, non_blocking=True)
        E.run(check=True)

    def _rows_buffers(self):
        n, nt, D, M = min(self.chunk, ROWS), len(self.layers_tgt), self.D, self.nkv * self.hd
        Z = lambda k, dt=torch.float32: torch.zeros(k, dtype=dt, device='cuda')
        self.rb = dict(x16=Z(n * nt * D, torch.float16), y=Z(n * D), h=Z(n * D), h16=Z(n * D, torch.float16),
                       k=Z(n * M), v=Z(n * M), ko=Z(n * M))

    def inject_rows(self, pf, pos0, n):
        """the batched inject of a prefill chunk: pf (Q35OwnPrefill(feature_layers=) of the target) just ran positions
        pos0 .. pos0 + n - 1 and holds their features (pf.features [n_layers, chunk, D]); the drafter's inject over all n
        rows at once on the prefill's own launchers: fc and each layer's k / v as WMMA GEMMs (Q8_0), the norms, the k norm
        with RoPE, the f16 ring append (split where the ring wraps). Prefill numerics (WMMA sums, not the K3 inject tape's
        bits): the served gates are acceptance and token identity"""
        n, pos0 = int(n), int(pos0)
        if not 1 <= n <= self.chunk: raise ValueError(f'Dflash35.inject_rows: {n} rows (1 .. the projected chunk {self.chunk})')
        k3.check_span(pos0, n, self.max_pos, what='Dflash35.inject_rows')
        step = min(self.chunk, ROWS)
        for t0 in range(0, n, step): self._rows(pf, pos0 + t0, t0, min(step, n - t0))
        self.token += 1

    def _rows(self, pf, pos0, f0, n):
        """one pass of inject_rows: n <= ROWS rows, positions pos0 .., the prefill's feature rows f0 .."""
        if [int(l) for l in pf.feature_layers] != [int(l) for l in self.layers_tgt]:
            raise ValueError(f'Dflash35.inject_rows: the prefill exports {pf.feature_layers}, the drafter reads {self.layers_tgt}')
        O, A, b = pf.O, pf.A, self.rb
        nt, D, M, hd = len(self.layers_tgt), self.D, self.nkv * self.hd, self.hd
        with torch.cuda.device(self.dev):
            x16 = b['x16'][:n * nt * D]
            x16.view(n, nt, D).copy_(pf.features[:, f0:f0 + n].permute(1, 0, 2))   # (row t = the n_layers features of position t)
            O.mm(self.fc, x16, nt * D, b['y'], D, n=n)
            O.add_rms16(b['y'], self.enc_norm, self.eps, b['h16'], n, D, h=b['h'])
            for d in self.L:
                O.mm(d['k'], b['h16'], D, b['k'], M, n=n); O.mm(d['v'], b['h16'], D, b['v'], M, n=n)
                A.qk_nr(b['k'], hd, self.nkv, hd, d['kn'], hd, self.theta_scale, self.eps, pos0, b['ko'], n)
                t = 0
                while t < n:                                    # (ring rows (pos0 + t) & (RING - 1) .. up to the wrap)
                    r0 = (pos0 + t) & (RING - 1); c = min(n - t, RING - r0)
                    O.kv_append16(b['ko'][t * M:], M, b['v'][t * M:], M, d['ring'], M, r0, c, RING, pf.err)
                    t += c

    def inject(self, pos0, feats):
        """feats [n_target_layers, n, D] fp32 on the card or in host memory (copied T rows a launch into feat16) (the target's export for positions pos0 .. pos0 + n - 1), in chunks
        of T rows (the last padded with copies of its last row at the following positions)"""
        nl, n, D = feats.shape
        if nl != len(self.layers_tgt) or D != self.D: raise ValueError(f'Dflash35.inject: feats {tuple(feats.shape)}')
        k3.check_span(int(pos0), int(n), self.max_pos, what='Dflash35.inject')   # (the ring addresses any position as p & (RING - 1); bounded by the target's)
        with torch.cuda.device(self.dev):
            for c0 in range(0, n, T):
                rows = [min(c0 + r, n - 1) for r in range(T)]
                self.buf['feat16'].view(T, nl, D).copy_(feats[:, rows].permute(1, 0, 2))
                self.hstage[16 + T] = pos0 + c0
                self._launch(self.Ei)

    def draft(self, last, n0):
        """the 3 drafts for the positions after `last`: the block [last, MASK x 3] at positions n0 .. n0 + 3, `last` the token at
        n0 (the next one the target reads), the features of positions < n0 injected"""
        if not 0 <= int(last) < self.vocab: raise ValueError(f'Dflash35.draft: token {last}')
        k3.check_span(int(n0), T, self.max_pos, what='Dflash35.draft')
        self.hstage[16] = int(last)
        for r in range(1, T): self.hstage[16 + r] = self.mask
        self.hstage[16 + T] = int(n0)
        with torch.cuda.device(self.dev):
            self._launch(self.Ed); torch.cuda.synchronize()
        return self.buf['out'][:T - 1].tolist()
