"""The batched prefill's attention core (docs/agentic/spec-prefill-cache.md M5 S6, milestone A; kernels/prefill_attn.hip):
q / k norm + RoPE (OP_QK_NR), the Hadamard rotations (OP_HADAMARD), the q8_0 KV rows (OP_KV_APPEND), dense attention over
every earlier position as the Coder's split pair (OP_ATTN_PART / OP_ATTN_COMB over the identity list), the output gate
(OP_GATE_MUL), and the DFlash features (OP_HC_AVG): each kernel is its K3 decode op's expressions in its order over many
rows, so BITWISE that op on the same inputs (tests/test_prefill_attn.py g1, against K3 itself one token per pass).

Dense attention equals the model's QSA attention only while a row sees at most capacity.QSA_EXACT (2051) cells: the
dense launchers refuse a position at or past QSA_EXACT and every position outside the cache's rows. The device also
refuses a row past the cache's rows (the error word, bit 0 KV append / bit 1 attention / bit 2 a pool; check() raises),
never writes or reads past them.

The QSA prompt path (M5 S7, rows at positions >= QSA_EXACT; the decode's QSA tape ops in kernels/k3_qsa_ops.h and
k3_attn_split.h, each kernel its op's expressions in its order, so BITWISE the op on the same inputs: tests/test_prefill_qsa.py
g1 against K3's QSA module one token per pass):
  idx_pool    OP_IDX_APPEND's pooled keys, IDXPOOL[j] from the IDXRAW rows 4j .. 4j + 3 (which the prefill writes through
              kv_append(M = 64), the dense tape's bytes); pools from token 0, so the QSA tape never backfills after a prefill
  qsa_score   OP_QSA_SCORE for a query sub-block of rows into the bounded score scratch (AttnWorkspace.score: score_rows rows
              of the engine's pool rows, <= SCORE_MIB; counted in capacity.prefill_need_mib(qsa_ctx=))
  qsa_select  OP_QSA_SELECT per row: the top 512 pools, the lowest index first among equal scores, ascending, + the tail
  attend(sel=) the split pair over each row's SEL list (OP_ATTN_PART / COMB with Seg.gs = SEL)

Records: the split pair keeps a partial record per (row, query head, split) [260 floats]; K3's layout for 2048 rows at
65 splits would be ~1.7 GB per card, so attend() runs a chunk's rows in blocks of `rows_per_block` (AttnWorkspace: the
records for one block, 52 MiB at 64 rows). A row's splits depend only on its own list (its position), so the blocks, the
chunk size and the split points change no bit (G2b).

Memory (AttnWorkspace, counted in capacity.prefill_need_mib): per card the rotated q [T, 12, 256], the rotated k [T, 256],
the records of one row block, the indexer's raw key rows [T, 128] (QSA engines: IDXRAW), the error word; on the feature card
the DFlash features [layers, FEAT_ROWS, D] by absolute position (the prefill covers at most QSA_EXACT positions). The
attention output itself is written into mmq.Workspace.x_attn_out (the attn_output GEMM's input: the Hadamard and the gate
run in place there)."""
import ctypes
import torch
from .hip import Module
from .bufcheck import disjoint, view_bytes

ALIGN = 256
ATTN_SS, ATTN_G, ATTN_D, ATTN_RS = 32, 12, 256, 260        # = kernels/prefill_attn.hip / k3_attn_split.h
ROWS_PER_BLOCK = 64                                        # attention rows per record block (52 MiB of records at 65 splits)
ERR_KV, ERR_ATTN, ERR_POOL = 1, 2, 4
QSA_ROWB, QSA_TOPK, QSA_SELW, QSA_D, QSA_H = 136, 512, 516, 128, 4   # = kernels/k3_qsa_ops.h / caminus.k3
SCORE_MIB = 64          # the score scratch's cap per card: query sub-blocks of score_rows(T, pool rows) rows (64 at least)
SCORE_RPG = 64          # rows per pf_qsa_score block (its 4 pools per wave loaded once for these rows)
# the score kernels (the same bits: tests/test_prefill_qsa.py): 'rs' the reduce-scatter (default), 'v0' op_qsa_score's butterflies
SCORE_KERNELS = {'rs': 'pf_qsa_score', 'v0': 'pf_qsa_score_v0'}
SCORE_KERNEL = 'rs'


def score_rows(T, prow):
    """rows of the QSA score scratch (one query sub-block): a multiple of 64, at least 64, at most T rounded up to 64, and at
    most SCORE_MIB of fp32 rows of prow pools (max_ctx 32768: 2048 rows, 64 MiB; 262144: 256 rows)"""
    cap = (SCORE_MIB * 2 ** 20) // (4 * int(prow)) // 64 * 64
    return int(max(64, min(-(-int(T) // 64) * 64, cap)))


def pool_rows(qsa_ctx):
    """IDXPOOL rows of an engine of max_ctx qsa_ctx (FlashNextTP._alloc_idx: (max_ctx + 3) / 4)"""
    return (int(qsa_ctx) + 3) // 4


def qsa_exact():
    from .capacity import QSA_EXACT
    return QSA_EXACT


def max_splits():
    """the most splits a dense row can have: positions < QSA_EXACT see at most QSA_EXACT cells"""
    return (qsa_exact() + ATTN_SS - 1) // ATTN_SS


def feat_rows(T=4):
    """rows of the DFlash feature export: every position the prefill can cover (< QSA_EXACT), rounded up to the target's
    pass rows T (the drafter injects T rows at a time)"""
    q = qsa_exact(); return (q + T - 1) // T * T


def attn_layout(T, nh, hd, idx_dim, rows_per_block=ROWS_PER_BLOCK, n_feat=0, D=0, T_pass=4, qsa_ctx=0):
    """name -> bytes of one card's attention workspace for chunks of up to T tokens (n_feat > 0: the feature card's export;
    qsa_ctx > 0: a QSA engine of that max_ctx: the indexer query rows iq / iqo [T, 4, 128], the SEL rows [T, QSA_SELW] and the
    score scratch [score_rows, pool rows])"""
    T = int(T)
    L = dict(qo=4 * T * nh * hd, ko=4 * T * hd, rec=4 * int(rows_per_block) * nh * max_splits() * ATTN_RS, ikr=4 * T * idx_dim, err=4)
    if n_feat: L['feats'] = 4 * int(n_feat) * feat_rows(T_pass) * int(D)
    if qsa_ctx:
        pr = pool_rows(qsa_ctx)
        L.update(iq=4 * T * QSA_H * QSA_D, iqo=4 * T * QSA_H * QSA_D, sel=4 * T * QSA_SELW, score=4 * score_rows(T, pr) * pr)
    return {n: (b + ALIGN - 1) // ALIGN * ALIGN for n, b in L.items()}


def attn_bytes(T, nh, hd, idx_dim, rows_per_block=ROWS_PER_BLOCK, n_feat=0, D=0, T_pass=4, qsa_ctx=0):
    return sum(attn_layout(T, nh, hd, idx_dim, rows_per_block, n_feat, D, T_pass, qsa_ctx).values())


class AttnWorkspace:
    """one card's preallocated attention-core buffers (one allocation, carved by attn_layout; checked against the card's cap
    first); n_feat > 0 on the feature card: .feats [n_feat, feat_rows(T_pass), D]; qsa_ctx > 0 (a QSA engine's max_ctx): .iq /
    .iqo [T * 512], .sel int32 [T * QSA_SELW], .score [srows * prow] (srows = score_rows(T, prow), prow = pool_rows(qsa_ctx))"""
    def __init__(self, device, T, nh, hd, idx_dim, rows_per_block=ROWS_PER_BLOCK, n_feat=0, D=0, T_pass=4, check=True, qsa_ctx=0):
        from . import capacity
        self.dev, self.T, self.nh, self.hd, self.rpb = torch.device(device), int(T), int(nh), int(hd), int(rows_per_block)
        self.qsa_ctx = int(qsa_ctx); self.prow = pool_rows(qsa_ctx) if qsa_ctx else 0
        self.srows = score_rows(T, self.prow) if qsa_ctx else 0
        self.layout = attn_layout(T, nh, hd, idx_dim, rows_per_block, n_feat, D, T_pass, qsa_ctx); self.nbytes = sum(self.layout.values())
        if check: capacity.check([self.dev.index], [self.nbytes / capacity.MIB], f'prefill attention workspace (T={self.T})')
        self.block = torch.zeros(self.nbytes, dtype=torch.uint8, device=self.dev)
        off = 0
        for n, b in self.layout.items():
            v = self.block[off:off + b]
            setattr(self, n, v.view(torch.int32) if n in ('err', 'sel') else v.view(torch.float32)); off += b
        self.err = self.err[:1]
        if qsa_ctx:
            self.iq, self.iqo = self.iq[:self.T * QSA_H * QSA_D], self.iqo[:self.T * QSA_H * QSA_D]
            self.sel, self.score = self.sel[:self.T * QSA_SELW], self.score[:self.srows * self.prow]
        if n_feat: self.feats = self.feats[:int(n_feat) * feat_rows(T_pass) * int(D)].view(int(n_feat), feat_rows(T_pass), int(D))

    def check(self, what='prefill attention'):
        """raise k3.ContextBoundsError if the device refused a row (waits for the device)"""
        from . import k3
        e = int(self.err.item())
        if e:
            raise k3.ContextBoundsError(f'{what}: the device refused a row past the cache (err {e:#x}: '
                                        f'{"KV append " if e & ERR_KV else ""}{"attention " if e & ERR_ATTN else ""}'
                                        f'{"indexer pool" if e & ERR_POOL else ""})')


def _b(t, n):
    return (t, int(n) * t.element_size()) if t is not None else (None, 0)


class Attn:
    """the prefill_attn kernels on one device (every launch with that device current, on its current stream)"""
    def __init__(self, device, module=None):
        self.dev = torch.device(device)
        if module is not None: self.m = module
        else:
            with torch.cuda.device(self.dev): self.m = Module('prefill_attn')

    def _t(self, t, what, dtype, n, align=4):
        if not isinstance(t, torch.Tensor) or t.device != self.dev or not t.is_contiguous():
            raise ValueError(f'prefill_attn: {what} must be a contiguous tensor on {self.dev}')
        if t.dtype != dtype: raise ValueError(f'prefill_attn: {what} must be {dtype} (is {t.dtype})')
        if t.numel() < int(n): raise ValueError(f'prefill_attn: {what} holds {t.numel()} < {n} elements (its view; {view_bytes(t)} B)')
        if t.data_ptr() % align: raise ValueError(f'prefill_attn: {what} must be {align}-byte aligned')
        return t

    @staticmethod
    def _pos(pos0, T, rows, what):
        """positions pos0 .. pos0 + T - 1 inside the cache's rows and below QSA_EXACT (dense attention is the model's only there)"""
        from . import k3
        pos0, T = int(pos0), int(T)
        k3.check_span(pos0, T, int(rows), what=what)
        if pos0 + T > qsa_exact():
            raise k3.ContextBoundsError(f'{what}: positions {pos0} .. {pos0 + T - 1} reach {qsa_exact()}: dense attention equals the '
                                        f"model's QSA only below {qsa_exact()} positions (the QSA prompt path, S7)")

    def _go(self, fn, grid, *args):
        g = tuple(int(x) for x in grid) if isinstance(grid, tuple) else (int(grid),)
        with torch.cuda.device(self.dev):
            getattr(self.m, fn)(g, (256,), *args)

    def qk_nr(self, x, head_stride, heads, D, w, n_rot, theta_scale, eps, pos0, out, T):
        """out [T, heads, D] = RoPE(rms_norm(x head) * w) per (row, head) at position pos0 + row (OP_QK_NR); x [T, heads]
        heads at head_stride floats"""
        T, heads, D, hs = int(T), int(heads), int(D), int(head_stride)
        if T < 1 or heads < 1 or not 1 <= D <= 512 or hs < D or not 0 < int(n_rot) <= D or int(n_rot) % 2:
            raise ValueError(f'qk_nr: T {T}, heads {heads}, D {D}, head_stride {hs}, n_rot {n_rot}')
        nx = (T * heads - 1) * hs + D
        self._t(x, 'x', torch.float32, nx); self._t(w, 'w', torch.float32, D); self._t(out, 'out', torch.float32, T * heads * D)
        disjoint('qk_nr', dict(x=_b(x, nx), w=_b(w, D), out=_b(out, T * heads * D)), ('out',))
        self._go('pf_qk_nr', T * heads, x, ctypes.c_int(hs), ctypes.c_int(heads), w, ctypes.c_int(D), ctypes.c_int(int(n_rot)),
                 ctypes.c_float(theta_scale), ctypes.c_float(eps), ctypes.c_int(int(pos0)), out)

    def hadamard(self, x, n, C, rpb=1):
        """in place: the orthonormal Walsh-Hadamard transform of C chunks of n values (n 64 / 128 / 256)"""
        n, C, rpb = int(n), int(C), int(rpb)
        if n not in (64, 128, 256) or C < 1 or rpb < 1: raise ValueError(f'hadamard: n {n}, C {C}, rpb {rpb}')
        self._t(x, 'x', torch.float32, C * n)
        self._go('pf_hadamard', (C + rpb - 1) // rpb, x, ctypes.c_int(C), ctypes.c_int(n), ctypes.c_int(rpb))

    def kv_append(self, k, ldk, v, ldv, M, cache, pos0, T, err):
        """cache rows pos0 .. pos0 + T - 1 = q8_0 [K | V] of the T rows of k / v (M values each, row strides ldk / ldv); any
        rows inside the cache (the KV and IDXRAW rows of QSA positions too: S7)"""
        from . import k3
        M, T, ldk, ldv = int(M), int(T), int(ldk), int(ldv)
        if M < 32 or M % 32 or T < 1 or ldk < M or ldv < M: raise ValueError(f'kv_append: M {M}, T {T}, ldk {ldk}, ldv {ldv}')
        pb = 2 * (M // 32) * 34
        if cache.dtype != torch.uint8 or (view_bytes(cache) % pb): raise ValueError(f'kv_append: the cache must be uint8 rows of {pb} B')
        rows = view_bytes(cache) // pb
        k3.check_span(int(pos0), T, rows, what='prefill kv_append')
        nk, nv = (T - 1) * ldk + M, (T - 1) * ldv + M
        self._t(k, 'k', torch.float32, nk); self._t(v, 'v', torch.float32, nv); self._t(cache, 'cache', torch.uint8, rows * pb, 1)
        self._t(err, 'err', torch.int32, 1)
        disjoint('kv_append', dict(k=_b(k, nk), v=_b(v, nv), cache=(cache, rows * pb), err=_b(err, 1)), ('cache', 'err'))
        self._go('pf_kv_append', T, k, ctypes.c_int(ldk), v, ctypes.c_int(ldv), ctypes.c_int(M), cache, ctypes.c_int(int(pos0)),
                 ctypes.c_int(rows), err)

    def attend(self, q, cache, pos0, T, nkv, scale, out, rec, err, rows_per_block=ROWS_PER_BLOCK, sel=None):
        """out [T, nkv * 12, 256] = attention of rows pos0 .. pos0 + T - 1 over the q8_0 cache: sel None, dense over the cells
        0 .. pos (the split pair over the identity list, OP_ATTN_PART / OP_ATTN_COMB; positions below QSA_EXACT only); sel
        int32 [T, QSA_SELW] (qsa_select's rows), each row over its selected pools' cells then its tail (the split pair with
        Seg.gs = SEL; any position inside the cache: a list is at most QSA_EXACT cells). rows_per_block rows per launch pair
        (rec: one block's records [rows_per_block, heads, max_splits(), 260])"""
        T, nkv, R = int(T), int(nkv), int(rows_per_block)
        heads = nkv * ATTN_G; pb = 2 * (nkv * ATTN_D // 32) * 34
        if T < 1 or nkv < 1 or R < 1: raise ValueError(f'attend: T {T}, nkv {nkv}, rows_per_block {R}')
        if cache.dtype != torch.uint8 or view_bytes(cache) % pb: raise ValueError(f'attend: the cache must be uint8 rows of {pb} B')
        rows = view_bytes(cache) // pb
        if sel is None: self._pos(pos0, T, rows, 'prefill attention')
        else:
            from . import k3
            k3.check_span(int(pos0), T, rows, what='prefill attention (QSA rows)')
            self._t(sel, 'sel', torch.int32, T * QSA_SELW)
        NSM = max_splits(); nrec = R * heads * NSM * ATTN_RS
        self._t(q, 'q', torch.float32, T * heads * ATTN_D, 16); self._t(out, 'out', torch.float32, T * heads * ATTN_D)
        self._t(rec, 'rec', torch.float32, nrec); self._t(cache, 'cache', torch.uint8, rows * pb, 1); self._t(err, 'err', torch.int32, 1)
        disjoint('attend', dict(q=_b(q, T * heads * ATTN_D), cache=(cache, rows * pb), rec=_b(rec, nrec), out=_b(out, T * heads * ATTN_D),
                                err=_b(err, 1), sel=_b(sel, T * QSA_SELW)), ('rec', 'out', 'err'))
        for b0 in range(0, T, R):
            Rb = min(R, T - b0); p_first = int(pos0) + b0
            # dense: the block's last row sees p_first + Rb cells; SEL rows: at most QSA_EXACT (512 pools + a 3-cell tail)
            NS = (p_first + Rb + ATTN_SS - 1) // ATTN_SS if sel is None else NSM
            assert NS <= NSM, (NS, NSM)
            qb = q.view(-1)[b0 * heads * ATTN_D:]; ob = out.view(-1)[b0 * heads * ATTN_D:]
            if sel is None:
                self._go('pf_attn_part', NS * Rb * nkv, qb, cache, ctypes.c_int(p_first), ctypes.c_int(Rb), ctypes.c_int(nkv), ctypes.c_int(NS),
                         ctypes.c_float(scale), ctypes.c_int(rows), rec, err)
                self._go('pf_attn_comb', Rb * heads, rec, ctypes.c_int(p_first), ctypes.c_int(heads), ctypes.c_int(NS), ctypes.c_int(rows), ob)
            else:
                sb = sel.view(-1)[b0 * QSA_SELW:]
                self._go('pf_attn_part_sel', NS * Rb * nkv, qb, cache, ctypes.c_int(p_first), ctypes.c_int(Rb), ctypes.c_int(nkv),
                         ctypes.c_int(NS), ctypes.c_float(scale), ctypes.c_int(rows), rec, err, sb)
                self._go('pf_attn_comb_sel', Rb * heads, rec, ctypes.c_int(p_first), ctypes.c_int(heads), ctypes.c_int(NS), ctypes.c_int(rows),
                         ob, sb)

    # ---- the QSA indexer (M5 S7) -------------------------------------------------------------------------------------------
    @staticmethod
    def _rows136(t, what):
        if not isinstance(t, torch.Tensor) or t.dtype != torch.uint8 or not t.is_contiguous() or view_bytes(t) % QSA_ROWB or not view_bytes(t):
            raise ValueError(f'prefill_attn: {what} must be contiguous uint8 rows of {QSA_ROWB} B')
        return view_bytes(t) // QSA_ROWB

    def idx_pool(self, raw, pool, j0, j1, kn, theta_scale, eps, err, n_rot=64):
        """IDXPOOL pools j0 .. j1 - 1 from the IDXRAW rows 4 j0 .. 4 j1 - 1 (OP_IDX_APPEND's pooling: mean, RMS norm * kn, NEOX
        RoPE of 64 dims at 4j, q8_0); every pool's rows inside IDXRAW and the pool inside IDXPOOL (refused on the host;
        the device sets err bit 2 instead of reading or writing past them)"""
        from . import k3
        j0, j1 = int(j0), int(j1)
        if int(n_rot) != 64: raise ValueError(f'idx_pool: n_rot {n_rot} (the indexer ropes 64 dims)')
        rows, prow = self._rows136(raw, 'IDXRAW'), self._rows136(pool, 'IDXPOOL')
        if not 0 <= j0 <= j1 or 4 * j1 > rows or j1 > prow:
            raise k3.ContextBoundsError(f'idx_pool: pools {j0} .. {j1 - 1} (IDXRAW rows {4 * j0} .. {4 * j1 - 1}) outside IDXRAW {rows} / IDXPOOL {prow} rows')
        if j1 == j0: return
        self._t(raw, 'IDXRAW', torch.uint8, rows * QSA_ROWB, 1); self._t(pool, 'IDXPOOL', torch.uint8, prow * QSA_ROWB, 1)
        self._t(kn, 'k_norm', torch.float32, QSA_D, 16); self._t(err, 'err', torch.int32, 1)
        disjoint('idx_pool', dict(raw=(raw, rows * QSA_ROWB), pool=(pool, prow * QSA_ROWB), kn=_b(kn, QSA_D), err=_b(err, 1)), ('pool', 'err'))
        self._go('pf_idx_pool', (j1 - j0 + 7) // 8, raw, ctypes.c_int(rows), pool, ctypes.c_int(prow), ctypes.c_int(j0), ctypes.c_int(j1 - j0),
                 kn, ctypes.c_float(theta_scale), ctypes.c_float(eps), err)

    @staticmethod
    def qsa_w():
        """the indexer's score weight 1 / sqrt(128) (fp32: caminus.k3.Engine.qsa_score's default)"""
        import numpy as np
        return float(np.float32(1.0) / np.sqrt(np.float32(QSA_D)))

    def qsa_score(self, q, pool, pos0, T, score, ld, w=None, rpg=SCORE_RPG, kernel=None):
        """score[r * ld + j] = OP_QSA_SCORE of row r (position pos0 + r, r < T) and pool j < nv_r = min((pos + 1) / 4, pool
        rows); q fp32 [T, 4, 128] (normed, RoPE'd; 16-byte aligned); entries past nv_r are not written"""
        from . import k3
        T, ld, rpg = int(T), int(ld), int(rpg)
        prow = self._rows136(pool, 'IDXPOOL')
        if T < 1 or rpg < 1: raise ValueError(f'qsa_score: T {T}, rows per group {rpg}')
        k3.check_span(int(pos0), T, 4 * prow, what='prefill qsa_score (positions vs the pools)')
        nvmax = min((int(pos0) + T) >> 2, prow)
        if nvmax < 1: raise ValueError(f'qsa_score: rows {pos0} .. {int(pos0) + T - 1} see no pool')
        if ld < nvmax: raise ValueError(f'qsa_score: row stride {ld} < the last row\'s {nvmax} pools')
        ns = (T - 1) * ld + nvmax
        self._t(q, 'q', torch.float32, T * QSA_H * QSA_D, 16); self._t(pool, 'IDXPOOL', torch.uint8, prow * QSA_ROWB, 1)
        self._t(score, 'score', torch.float32, ns)
        disjoint('qsa_score', dict(q=_b(q, T * QSA_H * QSA_D), pool=(pool, prow * QSA_ROWB), score=_b(score, ns)), ('score',))
        w = self.qsa_w() if w is None else float(w)
        kn = SCORE_KERNEL if kernel is None else kernel
        if kn not in SCORE_KERNELS: raise ValueError(f'qsa_score: kernel {kn!r} (one of {SCORE_KERNELS})')
        self._go(SCORE_KERNELS[kn], ((nvmax + 31) // 32, (T + rpg - 1) // rpg), q, pool, ctypes.c_int(prow), ctypes.c_int(int(pos0)), ctypes.c_int(T),
                 ctypes.c_int(rpg), ctypes.c_float(w), score, ctypes.c_long(ld))

    def qsa_select(self, score, ld, pos0, T, prow, sel):
        """sel[r] = OP_QSA_SELECT of row r (position pos0 + r, r < T) over score[r * ld + j], j < nv_r = min((pos + 1) / 4, prow):
        [0] count, [1] tail cells, [2] nv, [3] the threshold key, then the pool ids ascending"""
        from . import k3
        T, ld, prow = int(T), int(ld), int(prow)
        if T < 1 or prow < 1: raise ValueError(f'qsa_select: T {T}, pool rows {prow}')
        k3.check_span(int(pos0), T, 4 * prow, what='prefill qsa_select (positions vs the pools)')
        nvmax = min((int(pos0) + T) >> 2, prow)
        if ld < nvmax: raise ValueError(f'qsa_select: row stride {ld} < the last row\'s {nvmax} pools')
        ns = (T - 1) * ld + nvmax
        self._t(score, 'score', torch.float32, ns); self._t(sel, 'sel', torch.int32, T * QSA_SELW)
        disjoint('qsa_select', dict(score=_b(score, ns), sel=_b(sel, T * QSA_SELW)), ('sel',))
        self._go('pf_qsa_select', T, score, ctypes.c_long(ld), ctypes.c_int(prow), ctypes.c_int(int(pos0)), sel)

    def gate_mul(self, a, gate, stride, off, y, N, D):
        """y[i] = a[i] * sigmoid(gate[(i / D) * stride + off + i % D]) over N elements (y may be a exactly)"""
        N, D, stride, off = int(N), int(D), int(stride), int(off)
        if N < 1 or D < 1 or N % D or off + D > stride: raise ValueError(f'gate_mul: N {N}, D {D}, stride {stride}, off {off}')
        ng = (N // D - 1) * stride + off + D
        self._t(a, 'a', torch.float32, N); self._t(gate, 'gate', torch.float32, ng); self._t(y, 'y', torch.float32, N)
        disjoint('gate_mul', dict(a=_b(a, N), gate=_b(gate, ng), y=_b(y, N)), ('y',), same_ok=[('y', 'a')])
        self._go('pf_gate_mul', (N + 255) // 256, a, gate, y, ctypes.c_long(N), ctypes.c_int(D), ctypes.c_int(stride), ctypes.c_int(off))

    def hc_avg(self, res, y, T, hc, D, ldo=None):
        """y[t * ldo + i] = the mean over the hc streams of res [T, hc, D] (OP_HC_AVG)"""
        T, hc, D = int(T), int(hc), int(D); ldo = D if ldo is None else int(ldo)
        if T < 1 or hc < 1 or D < 1 or ldo < D: raise ValueError(f'hc_avg: T {T}, hc {hc}, D {D}, ldo {ldo}')
        ny = (T - 1) * ldo + D
        self._t(res, 'res', torch.float32, T * hc * D); self._t(y, 'y', torch.float32, ny)
        disjoint('hc_avg', dict(res=_b(res, T * hc * D), y=_b(y, ny)), ('y',))
        self._go('pf_hc_avg', (T * D + 255) // 256, res, y, ctypes.c_int(T), ctypes.c_int(hc), ctypes.c_int(D), ctypes.c_int(ldo))
