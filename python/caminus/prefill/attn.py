"""The batched prefill's attention core (docs/specs/spec-prefill-cache.md M5 S6, milestone A; kernels/prefill_attn.hip):
q / k norm + RoPE (OP_QK_NR), the Hadamard rotations (OP_HADAMARD), the q8_0 KV rows (OP_KV_APPEND), dense attention over
every earlier position as the Coder's split pair (OP_ATTN_PART / OP_ATTN_COMB over the identity list), the output gate
(OP_GATE_MUL), and the DFlash features (OP_HC_AVG): each kernel is its K3 decode op's expressions in its order over many
rows, so BITWISE that op on the same inputs (tests/prefill/test_prefill_attn.py g1, against K3 itself one token per pass).

Dense attention equals the model's QSA attention only while a row sees at most capacity.QSA_EXACT (2051) cells: the
dense launchers refuse a position at or past QSA_EXACT and every position outside the cache's rows. The device also
refuses a row past the cache's rows (the error word, bit 0 KV append / bit 1 attention / bit 2 a pool; check() raises),
never writes or reads past them.

The QSA prompt path (M5 S7, rows at positions >= QSA_EXACT; the decode's QSA tape ops in kernels/k3_qsa_ops.h and
k3_attn_split.h, each kernel its op's expressions in its order, so BITWISE the op on the same inputs: tests/prefill/test_prefill_qsa.py
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
from ..device.hip import Module
from ..device.bufcheck import ALIGN, LaunchCheck, aligned, view_bytes   # noqa: F401  (ALIGN: attn.ALIGN, the layout's block)

F32, F16, I32, U8 = torch.float32, torch.float16, torch.int32, torch.uint8
_check = LaunchCheck('prefill_attn', align=4)   # every launch: each buffer stated once (fp32 / int32 pointers 4-byte aligned by default)
ATTN_SS, ATTN_G, ATTN_D, ATTN_RS = 32, 12, 256, 260        # = kernels/prefill_attn.hip / k3_attn_split.h
ATTN_G_OF = {'prefill_attn': 12, 'prefill_attn_q35': 8, 'prefill_attn_q35d': 6}    # query heads per kv head of each code object (prefill_attn_q35.hip)
ROWS_PER_BLOCK = 64                                        # attention rows per record block (52 MiB of records at 65 splits)
ERR_KV, ERR_ATTN, ERR_POOL = 1, 2, 4
QSA_ROWB, QSA_TOPK, QSA_SELW, QSA_D, QSA_H = 136, 512, 516, 128, 4   # = kernels/k3_qsa_ops.h / caminus.engine.k3
SCORE_MIB = 64          # the score scratch's cap per card: query sub-blocks of score_rows(T, pool rows) rows (64 at least)
SCORE_RPG = 64          # rows per pf_qsa_score block (its 4 pools per wave loaded once for these rows)
# the score kernels (the same bits: tests/prefill/test_prefill_qsa.py): 'rs' the reduce-scatter (default), 'v0' op_qsa_score's butterflies
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
    from ..device.capacity import QSA_EXACT
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
    return aligned(L)


def attn_bytes(T, nh, hd, idx_dim, rows_per_block=ROWS_PER_BLOCK, n_feat=0, D=0, T_pass=4, qsa_ctx=0):
    return sum(attn_layout(T, nh, hd, idx_dim, rows_per_block, n_feat, D, T_pass, qsa_ctx).values())


class AttnWorkspace:
    """one card's preallocated attention-core buffers (one allocation, carved by attn_layout; checked against the card's cap
    first); n_feat > 0 on the feature card: .feats [n_feat, feat_rows(T_pass), D]; qsa_ctx > 0 (a QSA engine's max_ctx): .iq /
    .iqo [T * 512], .sel int32 [T * QSA_SELW], .score [srows * prow] (srows = score_rows(T, prow), prow = pool_rows(qsa_ctx))"""
    def __init__(self, device, T, nh, hd, idx_dim, rows_per_block=ROWS_PER_BLOCK, n_feat=0, D=0, T_pass=4, check=True, qsa_ctx=0):
        from ..device import capacity
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
        from ..engine import k3
        e = int(self.err.item())
        if e:
            raise k3.ContextBoundsError(f'{what}: the device refused a row past the cache (err {e:#x}: '
                                        f'{"KV append " if e & ERR_KV else ""}{"attention " if e & ERR_ATTN else ""}'
                                        f'{"indexer pool" if e & ERR_POOL else ""})')


class Attn:
    """the prefill_attn kernels on one device (every launch with that device current, on its current stream)"""
    def __init__(self, device, module=None, name='prefill_attn', module16=None):
        """module: an already loaded prefill_attn Module (the Coder's callers), else the code object `name`: prefill_attn
        (12 query heads per kv head, the Coder) or prefill_attn_q35 (8: qwen35moe, caminus.prefill.q35; attend_full only)"""
        if name not in ATTN_G_OF: raise ValueError(f'Attn: name {name!r} ({sorted(ATTN_G_OF)})')
        self.dev = torch.device(device); self.name, self.G = name, ATTN_G_OF[name]
        self.m16 = module16   # loaded only for an F16 target cache; the QSA indexer keeps the original Q8 module
        if module is not None: self.m = module
        else:
            with torch.cuda.device(self.dev): self.m = Module(name)

    @staticmethod
    def _pos(pos0, T, rows, what):
        """positions pos0 .. pos0 + T - 1 inside the cache's rows and below QSA_EXACT (dense attention is the model's only there)"""
        from ..engine import k3
        pos0, T = int(pos0), int(T)
        k3.check_span(pos0, T, int(rows), what=what)
        if pos0 + T > qsa_exact():
            raise k3.ContextBoundsError(f'{what}: positions {pos0} .. {pos0 + T - 1} reach {qsa_exact()}: dense attention equals the '
                                        f"model's QSA only below {qsa_exact()} positions (the QSA prompt path, S7)")

    def prepare_f16(self):
        """Load the target-cache variant before graph capture; keep QSA indexer kernels on self.m."""
        if self.G != ATTN_G: raise ValueError('F16 prefill split attention requires the Coder GQA geometry')
        if self.m16 is None:
            with torch.cuda.device(self.dev): self.m16 = Module('prefill_attn_f16')
        return self.m16

    def _go(self, fn, grid, *args, cache_f16=False):
        g = tuple(int(x) for x in grid) if isinstance(grid, tuple) else (int(grid),)
        with torch.cuda.device(self.dev):
            m = self.prepare_f16() if cache_f16 else self.m
            getattr(m, fn)(g, (256,), *args)

    def _cache(self, cache, M, q8, what):
        """The cache's explicit wire format, before any module load or launch; indexer callers default to Q8."""
        from ..engine import k3
        if not isinstance(q8, bool): raise ValueError(f'{what}: q8 must be a bool')
        if not q8 and self.G != ATTN_G: raise ValueError(f'{what}: F16 cache requires the Coder GQA geometry')
        pb, dt = k3.kv_pos_bytes(M, q8), U8 if q8 else F16
        if cache.dtype != dt or view_bytes(cache) % pb:
            raise ValueError(f'{what}: the cache must be {dt} rows of {pb} B')
        return view_bytes(cache) // pb, pb, dt

    def qk_nr(self, x, head_stride, heads, D, w, n_rot, theta_scale, eps, pos0, out, T):
        """out [T, heads, D] = RoPE(rms_norm(x head) * w) per (row, head) at position pos0 + row (OP_QK_NR); x [T, heads]
        heads at head_stride floats"""
        T, heads, D, hs = int(T), int(heads), int(D), int(head_stride)
        if T < 1 or heads < 1 or not 1 <= D <= 512 or hs < D or not 0 < int(n_rot) <= D or int(n_rot) % 2:
            raise ValueError(f'qk_nr: T {T}, heads {heads}, D {D}, head_stride {hs}, n_rot {n_rot}')
        nx = (T * heads - 1) * hs + D
        _check(self.dev, 'qk_nr', dict(x=(x, F32, nx), w=(w, F32, D), out=(out, F32, T * heads * D)), ('out',))
        self._go('pf_qk_nr', T * heads, x, ctypes.c_int(hs), ctypes.c_int(heads), w, ctypes.c_int(D), ctypes.c_int(int(n_rot)),
                 ctypes.c_float(theta_scale), ctypes.c_float(eps), ctypes.c_int(int(pos0)), out)

    def hadamard(self, x, n, C, rpb=1):
        """in place: the orthonormal Walsh-Hadamard transform of C chunks of n values (n 64 / 128 / 256)"""
        n, C, rpb = int(n), int(C), int(rpb)
        if n not in (64, 128, 256) or C < 1 or rpb < 1: raise ValueError(f'hadamard: n {n}, C {C}, rpb {rpb}')
        _check(self.dev, 'hadamard', dict(x=(x, F32, C * n)), ('x',))
        self._go('pf_hadamard', (C + rpb - 1) // rpb, x, ctypes.c_int(C), ctypes.c_int(n), ctypes.c_int(rpb))

    def kv_append(self, k, ldk, v, ldv, M, cache, pos0, T, err, q8=True):
        """cache rows = Q8_0 (default) or F16 [K | V] of k/v. Indexer rows always use the default Q8 format."""
        from ..engine import k3
        M, T, ldk, ldv = int(M), int(T), int(ldk), int(ldv)
        if M < 32 or M % 32 or T < 1 or ldk < M or ldv < M: raise ValueError(f'kv_append: M {M}, T {T}, ldk {ldk}, ldv {ldv}')
        rows, pb, dt = self._cache(cache, M, q8, 'kv_append')
        k3.check_span(int(pos0), T, rows, what='prefill kv_append')
        nk, nv = (T - 1) * ldk + M, (T - 1) * ldv + M
        _check(self.dev, 'kv_append', dict(k=(k, F32, nk), v=(v, F32, nv), cache=(cache, dt, rows * pb // (1 if q8 else 2), 1 if q8 else 2),
                                         err=(err, I32, 1)), ('cache', 'err'))
        self._go('pf_kv_append', T, k, ctypes.c_int(ldk), v, ctypes.c_int(ldv), ctypes.c_int(M), cache, ctypes.c_int(int(pos0)),
                 ctypes.c_int(rows), err, cache_f16=not q8)

    def attend(self, q, cache, pos0, T, nkv, scale, out, rec, err, rows_per_block=ROWS_PER_BLOCK, sel=None, q8=True):
        """out [T, nkv * 12, 256] = attention of rows pos0 .. pos0 + T - 1 over the selected Q8/F16 cache: sel None, dense over the cells
        0 .. pos (the split pair over the identity list, OP_ATTN_PART / OP_ATTN_COMB; positions below QSA_EXACT only); sel
        int32 [T, QSA_SELW] (qsa_select's rows), each row over its selected pools' cells then its tail (the split pair with
        Seg.gs = SEL; any position inside the cache: a list is at most QSA_EXACT cells). rows_per_block rows per launch pair
        (rec: one block's records [rows_per_block, heads, max_splits(), 260])"""
        T, nkv, R = int(T), int(nkv), int(rows_per_block)
        heads = nkv * ATTN_G
        if T < 1 or nkv < 1 or R < 1: raise ValueError(f'attend: T {T}, nkv {nkv}, rows_per_block {R}')
        rows, pb, dt = self._cache(cache, nkv * ATTN_D, q8, 'attend')
        if sel is None: self._pos(pos0, T, rows, 'prefill attention')
        else:
            from ..engine import k3
            k3.check_span(int(pos0), T, rows, what='prefill attention (QSA rows)')
        NSM = max_splits(); nrec = R * heads * NSM * ATTN_RS
        _check(self.dev, 'attend', dict(sel=(sel, I32, T * QSA_SELW), q=(q, F32, T * heads * ATTN_D, 16), out=(out, F32, T * heads * ATTN_D),
                                        rec=(rec, F32, nrec), cache=(cache, dt, rows * pb // (1 if q8 else 2), 1 if q8 else 16), err=(err, I32, 1)),
               ('rec', 'out', 'err'), opt=('sel',))
        for b0 in range(0, T, R):
            Rb = min(R, T - b0); p_first = int(pos0) + b0
            # dense: the block's last row sees p_first + Rb cells; SEL rows: at most QSA_EXACT (512 pools + a 3-cell tail)
            NS = (p_first + Rb + ATTN_SS - 1) // ATTN_SS if sel is None else NSM
            assert NS <= NSM, (NS, NSM)
            qb = q.view(-1)[b0 * heads * ATTN_D:]; ob = out.view(-1)[b0 * heads * ATTN_D:]
            if sel is None:
                self._go('pf_attn_part', NS * Rb * nkv, qb, cache, ctypes.c_int(p_first), ctypes.c_int(Rb), ctypes.c_int(nkv), ctypes.c_int(NS),
                         ctypes.c_float(scale), ctypes.c_int(rows), rec, err, cache_f16=not q8)
                self._go('pf_attn_comb', Rb * heads, rec, ctypes.c_int(p_first), ctypes.c_int(heads), ctypes.c_int(NS), ctypes.c_int(rows), ob,
                         cache_f16=not q8)
            else:
                sb = sel.view(-1)[b0 * QSA_SELW:]
                self._go('pf_attn_part_sel', NS * Rb * nkv, qb, cache, ctypes.c_int(p_first), ctypes.c_int(Rb), ctypes.c_int(nkv),
                         ctypes.c_int(NS), ctypes.c_float(scale), ctypes.c_int(rows), rec, err, sb, cache_f16=not q8)
                self._go('pf_attn_comb_sel', Rb * heads, rec, ctypes.c_int(p_first), ctypes.c_int(heads), ctypes.c_int(NS), ctypes.c_int(rows),
                         ob, sb, cache_f16=not q8)

    def attend_full(self, q, cache, pos0, T, nkv, scale, out, rec, err, max_rows=ROWS_PER_BLOCK):
        """out [T, nkv * G, 256] = dense attention of rows pos0 .. pos0 + T - 1 over the cells 0 .. pos of the q8_0 cache, at
        ANY position inside it (a model whose attention is dense at every depth: qwen35moe; attend() keeps the Coder's QSA
        limit). The split pair over the identity list as attend(), bitwise its decode op (k3_gguf_q35*: 32-cell splits, G 8),
        in blocks of rows sized so each block's records fit rec: block rows = min(max_rows, rec floats // (heads x the
        block's splits x 260)) (a row's splits depend only on its own position, so the blocking changes no bit: G2b)"""
        T, nkv, G = int(T), int(nkv), self.G
        heads = nkv * G; pb = 2 * (nkv * ATTN_D // 32) * 34
        if T < 1 or nkv < 1 or int(max_rows) < 1: raise ValueError(f'attend_full: T {T}, nkv {nkv}, max_rows {max_rows}')
        if cache.dtype != torch.uint8 or view_bytes(cache) % pb: raise ValueError(f'attend_full: the cache must be uint8 rows of {pb} B')
        rows = view_bytes(cache) // pb
        from ..engine import k3
        k3.check_span(int(pos0), T, rows, what='prefill attention (attend_full)')
        # rec: any size holding one row's records at the last position (the block rows follow from it), no alignment rule
        if not isinstance(rec, torch.Tensor) or rec.dtype != torch.float32 or not rec.is_contiguous() or rec.device != self.dev:
            raise ValueError('attend_full: rec fp32 on the device')
        nrec = rec.numel(); per_row_last = heads * ((int(pos0) + T + ATTN_SS - 1) // ATTN_SS) * ATTN_RS
        if nrec < per_row_last: raise ValueError(f'attend_full: rec holds {nrec} floats < one row\'s records at position {int(pos0) + T - 1} ({per_row_last})')
        _check(self.dev, 'attend_full', dict(q=(q, F32, T * heads * ATTN_D, 16), cache=(cache, U8, rows * pb, 1), rec=(rec, F32, nrec, 1),
                                             out=(out, F32, T * heads * ATTN_D), err=(err, I32, 1)), ('rec', 'out', 'err'))
        b0 = 0
        while b0 < T:
            p_first = int(pos0) + b0
            Rb = min(int(max_rows), T - b0)
            while Rb > 1 and Rb * heads * ((p_first + Rb + ATTN_SS - 1) // ATTN_SS) * ATTN_RS > nrec: Rb -= 1
            NS = (p_first + Rb + ATTN_SS - 1) // ATTN_SS
            assert Rb * heads * NS * ATTN_RS <= nrec, (Rb, NS, nrec)
            qb = q.view(-1)[b0 * heads * ATTN_D:]; ob = out.view(-1)[b0 * heads * ATTN_D:]
            self._go('pf_attn_part', NS * Rb * nkv, qb, cache, ctypes.c_int(p_first), ctypes.c_int(Rb), ctypes.c_int(nkv), ctypes.c_int(NS),
                     ctypes.c_float(scale), ctypes.c_int(rows), rec, err)
            self._go('pf_attn_comb', Rb * heads, rec, ctypes.c_int(p_first), ctypes.c_int(heads), ctypes.c_int(NS), ctypes.c_int(rows), ob)
            b0 += Rb

    # ---- the QSA indexer (M5 S7) -------------------------------------------------------------------------------------------
    _IDX_NAMES = dict(raw='IDXRAW', pool='IDXPOOL', kn='k_norm')   # (the refusals name the cache regions)

    @staticmethod
    def _rows136(t, what):
        if not isinstance(t, torch.Tensor) or t.dtype != torch.uint8 or not t.is_contiguous() or view_bytes(t) % QSA_ROWB or not view_bytes(t):
            raise ValueError(f'prefill_attn: {what} must be contiguous uint8 rows of {QSA_ROWB} B')
        return view_bytes(t) // QSA_ROWB

    def idx_pool(self, raw, pool, j0, j1, kn, theta_scale, eps, err, n_rot=64):
        """IDXPOOL pools j0 .. j1 - 1 from the IDXRAW rows 4 j0 .. 4 j1 - 1 (OP_IDX_APPEND's pooling: mean, RMS norm * kn, NEOX
        RoPE of 64 dims at 4j, q8_0); every pool's rows inside IDXRAW and the pool inside IDXPOOL (refused on the host;
        the device sets err bit 2 instead of reading or writing past them)"""
        from ..engine import k3
        j0, j1 = int(j0), int(j1)
        if int(n_rot) != 64: raise ValueError(f'idx_pool: n_rot {n_rot} (the indexer ropes 64 dims)')
        rows, prow = self._rows136(raw, 'IDXRAW'), self._rows136(pool, 'IDXPOOL')
        if not 0 <= j0 <= j1 or 4 * j1 > rows or j1 > prow:
            raise k3.ContextBoundsError(f'idx_pool: pools {j0} .. {j1 - 1} (IDXRAW rows {4 * j0} .. {4 * j1 - 1}) outside IDXRAW {rows} / IDXPOOL {prow} rows')
        if j1 == j0: return
        _check(self.dev, 'idx_pool', dict(raw=(raw, U8, rows * QSA_ROWB, 1), pool=(pool, U8, prow * QSA_ROWB, 1), kn=(kn, F32, QSA_D, 16), err=(err, I32, 1)),
               ('pool', 'err'), names=self._IDX_NAMES)
        self._go('pf_idx_pool', (j1 - j0 + 7) // 8, raw, ctypes.c_int(rows), pool, ctypes.c_int(prow), ctypes.c_int(j0), ctypes.c_int(j1 - j0),
                 kn, ctypes.c_float(theta_scale), ctypes.c_float(eps), err)

    @staticmethod
    def qsa_w():
        """the indexer's score weight 1 / sqrt(128) (fp32: caminus.engine.k3.Engine.qsa_score's default)"""
        import numpy as np
        return float(np.float32(1.0) / np.sqrt(np.float32(QSA_D)))

    def qsa_score(self, q, pool, pos0, T, score, ld, w=None, rpg=SCORE_RPG, kernel=None):
        """score[r * ld + j] = OP_QSA_SCORE of row r (position pos0 + r, r < T) and pool j < nv_r = min((pos + 1) / 4, pool
        rows); q fp32 [T, 4, 128] (normed, RoPE'd; 16-byte aligned); entries past nv_r are not written"""
        from ..engine import k3
        T, ld, rpg = int(T), int(ld), int(rpg)
        prow = self._rows136(pool, 'IDXPOOL')
        if T < 1 or rpg < 1: raise ValueError(f'qsa_score: T {T}, rows per group {rpg}')
        k3.check_span(int(pos0), T, 4 * prow, what='prefill qsa_score (positions vs the pools)')
        nvmax = min((int(pos0) + T) >> 2, prow)
        if nvmax < 1: raise ValueError(f'qsa_score: rows {pos0} .. {int(pos0) + T - 1} see no pool')
        if ld < nvmax: raise ValueError(f'qsa_score: row stride {ld} < the last row\'s {nvmax} pools')
        ns = (T - 1) * ld + nvmax
        _check(self.dev, 'qsa_score', dict(q=(q, F32, T * QSA_H * QSA_D, 16), pool=(pool, U8, prow * QSA_ROWB, 1), score=(score, F32, ns)), ('score',),
               names=self._IDX_NAMES)
        w = self.qsa_w() if w is None else float(w)
        kn = SCORE_KERNEL if kernel is None else kernel
        if kn not in SCORE_KERNELS: raise ValueError(f'qsa_score: kernel {kn!r} (one of {SCORE_KERNELS})')
        self._go(SCORE_KERNELS[kn], ((nvmax + 31) // 32, (T + rpg - 1) // rpg), q, pool, ctypes.c_int(prow), ctypes.c_int(int(pos0)), ctypes.c_int(T),
                 ctypes.c_int(rpg), ctypes.c_float(w), score, ctypes.c_long(ld))

    def qsa_select(self, score, ld, pos0, T, prow, sel):
        """sel[r] = OP_QSA_SELECT of row r (position pos0 + r, r < T) over score[r * ld + j], j < nv_r = min((pos + 1) / 4, prow):
        [0] count, [1] tail cells, [2] nv, [3] the threshold key, then the pool ids ascending"""
        from ..engine import k3
        T, ld, prow = int(T), int(ld), int(prow)
        if T < 1 or prow < 1: raise ValueError(f'qsa_select: T {T}, pool rows {prow}')
        k3.check_span(int(pos0), T, 4 * prow, what='prefill qsa_select (positions vs the pools)')
        nvmax = min((int(pos0) + T) >> 2, prow)
        if ld < nvmax: raise ValueError(f'qsa_select: row stride {ld} < the last row\'s {nvmax} pools')
        ns = (T - 1) * ld + nvmax
        _check(self.dev, 'qsa_select', dict(score=(score, F32, ns), sel=(sel, I32, T * QSA_SELW)), ('sel',))
        self._go('pf_qsa_select', T, score, ctypes.c_long(ld), ctypes.c_int(prow), ctypes.c_int(int(pos0)), sel)

    def gate_mul(self, a, gate, stride, off, y, N, D):
        """y[i] = a[i] * sigmoid(gate[(i / D) * stride + off + i % D]) over N elements (y may be a exactly)"""
        N, D, stride, off = int(N), int(D), int(stride), int(off)
        if N < 1 or D < 1 or N % D or off + D > stride: raise ValueError(f'gate_mul: N {N}, D {D}, stride {stride}, off {off}')
        ng = (N // D - 1) * stride + off + D
        _check(self.dev, 'gate_mul', dict(a=(a, F32, N), gate=(gate, F32, ng), y=(y, F32, N)), ('y',), same_ok=[('y', 'a')])
        self._go('pf_gate_mul', (N + 255) // 256, a, gate, y, ctypes.c_long(N), ctypes.c_int(D), ctypes.c_int(stride), ctypes.c_int(off))

    def hc_avg(self, res, y, T, hc, D, ldo=None):
        """y[t * ldo + i] = the mean over the hc streams of res [T, hc, D] (OP_HC_AVG)"""
        T, hc, D = int(T), int(hc), int(D); ldo = D if ldo is None else int(ldo)
        if T < 1 or hc < 1 or D < 1 or ldo < D: raise ValueError(f'hc_avg: T {T}, hc {hc}, D {D}, ldo {ldo}')
        ny = (T - 1) * ldo + D
        _check(self.dev, 'hc_avg', dict(res=(res, F32, T * hc * D), y=(y, F32, ny)), ('y',))
        self._go('pf_hc_avg', (T * D + 255) // 256, res, y, ctypes.c_int(T), ctypes.c_int(hc), ctypes.c_int(D), ctypes.c_int(ldo))
