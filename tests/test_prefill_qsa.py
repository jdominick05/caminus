"""Gates of the batched prefill's QSA prompt path, per op (docs/agentic/spec-prefill-cache.md M5 S7; caminus.prefill_attn's
idx_pool / qsa_score / qsa_select / attend(sel=), kernels/prefill_attn.hip pf_idx_pool / pf_qsa_score / pf_qsa_select /
pf_attn_part_sel / pf_attn_comb_sel).

  python -I tests/test_prefill_qsa.py CODER STAGE... [--small] [--tmax N] [--json OUT]

Stages:
  cpu    CPU only, nothing launched (launch recorders stand in for the kernels): the new launchers' refusals (pools past
         IDXRAW / IDXPOOL, positions past the pools, a short score row stride, short views, written buffers overlapping
         another, n_rot) and their positive controls; the QSA buffers in capacity.prefill_need_mib(qsa_ctx=)
  g1     GPU, ONE card that must be the XT (HIP_VISIBLE_DEVICES=1; refused on an XTX unless --allow-xtx): one attention
         layer's QSA core on one card's shapes (12 query heads, 1 kv head, head dim 256; the indexer's 4 heads of 128),
         the Coder's real attention and indexer norm weights (layer 3), seeded random rows (some indexer queries zero: every
         score 0, the lowest pools win; repeated keys: tied scores at the threshold), positions 0 .. tmax - 1 (crossing
         2051: nv 512 -> 513 .. tmax / 4). Reference: K3's own decode ops on the QSA module (k3_gguf_gsq_qsa, the QSA tape's
         ops: IDX_APPEND, QK_NR of the indexer query, QSA_SCORE, QSA_SELECT, the split pair with SEL, the KV chain), one
         token per pass. The prefill's chain over chunks (the IDXRAW rows by kv_append M = 64 as ChunkStack writes them, the
         pools by idx_pool, rows < 2051 dense, rows >= 2051 through scores in query sub-blocks, select, attend(sel=)):
         G1 IDXRAW / IDXPOOL bytes, every score of every row with nv > 512, every SEL row (header and ids), the attention
         output, bitwise. Canaries (the run1 method): every buffer the new kernels write (IDXPOOL, the score scratch, SEL,
         the records, the attention output, the error word) sits between 1 MiB sentinel guards; before each new kernel's
         launch the allocation is snapshotted, after it the guards hold the sentinel and no byte outside the launch's
         intended range changed; the inputs are unchanged. G2b: other chunkings (2051 / 4096 boundaries, 1000 + 3000 +
         ..., 777-row chunks), score sub-blocks of 64 rows, record blocks of 17 rows == the first run, bitwise. Every
         launch timed (events); the longest per kernel reported (< 50 ms). --small: tmax 2120, two chunkings
  speed  GPU, the XT: the new kernels at a 32K-depth chunk on synthetic pools (2048 query rows at positions 30720 ..
         32767, 8192 pools; the score scratch's 1984-row sub-block at max_ctx 33024): per-launch ms of idx_pool (512 pools),
         qsa_score, qsa_select, attend(sel=) per 64-row block, and the per-layer QSA core
Prints PASS / FAIL per check and RESULT PASS / FAIL; exit 0 = PASS."""
import argparse, contextlib, ctypes, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'python')); sys.path.insert(0, HERE)
from caminus import flashnext   # noqa: E402,F401  (puts gguf-py on the path)

EPS = 9.999999974752427e-07
FAILS, NUMBERS = [], {}
NH, HD, QD, QH, ROWB, SELW = 12, 256, 128, 4, 136, 516
QSA_EXACT = 2051
SENT = -5913691                  # the sentinel word 0xFFA5C3A5 (as tests/test_prefill_moe_canary.py): a NaN as fp32, a negative int32
GUARD = 1 << 20


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)
    return ok


def comp():
    return open('/proc/sys/vm/compaction_proactiveness').read().strip()


def meminfo():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable'): return l.split()[1] + ' kB'


def bits(a):
    a = a.detach().cpu().numpy() if hasattr(a, 'cpu') else a
    return np.ascontiguousarray(a).view(np.uint8)


def same(a, b):
    return np.array_equal(bits(a), bits(b))


class Rec:
    def __init__(self, log): self.log = log

    def __getattr__(self, name):
        if name.startswith('__'): raise AttributeError(name)
        return lambda *a, **kw: self.log.append(name)


def refused(log, fn, what, exc=(ValueError,)):
    n0 = len(log)
    try:
        fn()
    except exc as e:
        return check(len(log) == n0, f'refused, nothing launched: {what} ({str(e)[:110]})')
    except Exception as e:
        return check(False, f'{what}: raised {type(e).__name__} {e} instead of {exc}')
    return check(False, f'{what}: NOT refused ({len(log) - n0} launches recorded)')


def accepted(log, fn, what):
    n0 = len(log)
    try:
        fn()
    except Exception as e:
        return check(False, f'positive control {what}: refused ({type(e).__name__}: {str(e)[:160]})')
    return check(len(log) > n0, f'positive control: {what} accepted ({len(log) - n0} launches recorded, none run)')


def model_params(coder, il=3):
    """the Coder's RoPE parameters and layer il's attention / indexer norm weights (fp32 numpy)"""
    from gguf import GGUFReader
    rd = GGUFReader(coder, 'r'); f = {x.name: x for x in rd.fields.values()}
    hp = lambda k: f[f'qwen4exp.{k}'].contents()
    nrot = int(hp('rope.dimension_count'))
    libm = ctypes.CDLL('libm.so.6'); libm.powf.restype = ctypes.c_float; libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]
    theta = libm.powf(float(hp('rope.freq_base')), -2.0 / nrot)
    T = {t.name: t for t in rd.tensors}
    w = lambda n: np.array(T[n].data, dtype=np.float32).reshape(-1).copy()
    return dict(nrot=nrot, theta=theta, qn=w(f'blk.{il}.attn_q_norm.weight'), kn=w(f'blk.{il}.attn_k_norm.weight'),
                iqn=w(f'blk.{il}.indexer.q_norm.weight'), ikn=w(f'blk.{il}.indexer.k_norm.weight'))


# ---- stage: cpu ---------------------------------------------------------------------------------------------------------
def stage_cpu(coder):
    import torch
    from caminus import prefill_attn as pa, capacity, k3
    real = torch.cuda.device
    torch.cuda.device = lambda d: contextlib.nullcontext()
    try:
        log = []; A = pa.Attn('cpu', module=Rec(log))
        z = lambda n, dt=torch.float32: torch.zeros(int(n), dtype=dt)
        rows = 8192; prow = rows // 4
        raw, pool, err, kn = z(rows * ROWB, torch.uint8), z(prow * ROWB, torch.uint8), z(1, torch.int32), z(QD)
        accepted(log, lambda: A.idx_pool(raw, pool, 0, prow, kn, 0.5, EPS, err), 'idx_pool of every pool')
        accepted(log, lambda: A.idx_pool(raw, pool, 600, 1100, kn, 0.5, EPS, err), 'idx_pool of pools 600 .. 1099')
        refused(log, lambda: A.idx_pool(raw, pool, 0, prow + 1, kn, 0.5, EPS, err), 'idx_pool: a pool past IDXPOOL / IDXRAW', (k3.ContextBoundsError,))
        refused(log, lambda: A.idx_pool(raw[:4000 * ROWB], pool, 0, 1001, kn, 0.5, EPS, err), 'idx_pool: a pool\'s rows past IDXRAW (its view)', (k3.ContextBoundsError,))
        refused(log, lambda: A.idx_pool(raw, pool, 5, 4, kn, 0.5, EPS, err), 'idx_pool: j1 < j0', (k3.ContextBoundsError,))
        refused(log, lambda: A.idx_pool(raw, pool, -1, 4, kn, 0.5, EPS, err), 'idx_pool: a negative pool', (k3.ContextBoundsError,))
        refused(log, lambda: A.idx_pool(raw, pool, 0, 4, kn[:-1], 0.5, EPS, err), 'idx_pool: k_norm short')
        refused(log, lambda: A.idx_pool(raw, pool, 0, 4, kn, 0.5, EPS, err, n_rot=32), 'idx_pool: n_rot 32')
        refused(log, lambda: A.idx_pool(raw[:-1], pool, 0, 4, kn, 0.5, EPS, err), 'idx_pool: IDXRAW not whole rows')
        both = z((rows + prow) * ROWB, torch.uint8)
        refused(log, lambda: A.idx_pool(both[:rows * ROWB], both[(rows - 10) * ROWB:(rows - 10 + prow) * ROWB], 0, 4, kn, 0.5, EPS, err),
                'idx_pool: IDXPOOL overlapping IDXRAW')
        T = 256; p0 = 6000; nv = (p0 + T) // 4
        q, score, sel = z(T * QH * QD), z(T * prow), z(T * SELW, torch.int32)
        accepted(log, lambda: A.qsa_score(q, pool, p0, T, score, prow), 'qsa_score (ld = the pool rows)')
        accepted(log, lambda: A.qsa_score(q, pool, p0, T, score[:(T - 1) * nv + nv], nv), 'qsa_score with the exact row stride and score bytes')
        refused(log, lambda: A.qsa_score(q, pool, p0, T, score[:(T - 1) * nv + nv - 1], nv), 'qsa_score: the score view one short')
        refused(log, lambda: A.qsa_score(q, pool, p0, T, score, nv - 1), 'qsa_score: row stride below the last row\'s pools')
        refused(log, lambda: A.qsa_score(q, pool, rows - 10, T, score, prow), 'qsa_score: positions past the pools\' rows', (k3.ContextBoundsError,))
        refused(log, lambda: A.qsa_score(q[:-1], pool, p0, T, score, prow), 'qsa_score: q short')
        qq = z(T * QH * QD + 4)
        refused(log, lambda: A.qsa_score(qq[1:T * QH * QD + 1], pool, p0, T, score, prow), 'qsa_score: q not 16-byte aligned (float4 loads)')
        refused(log, lambda: A.qsa_score(score[:T * QH * QD], pool, p0, T, score, prow), 'qsa_score: score overlapping q')
        accepted(log, lambda: A.qsa_select(score, prow, p0, T, prow, sel), 'qsa_select')
        refused(log, lambda: A.qsa_select(score, prow, p0, T, prow, sel[:-1]), 'qsa_select: sel short')
        refused(log, lambda: A.qsa_select(score, nv - 1, p0, T, prow, sel), 'qsa_select: row stride below the last row\'s pools')
        refused(log, lambda: A.qsa_select(score, prow, rows, T, prow, sel), 'qsa_select: positions past the pools\' rows', (k3.ContextBoundsError,))
        refused(log, lambda: A.qsa_select(score, prow, p0, T, prow, score.view(torch.int32)[:T * SELW]), 'qsa_select: sel on the scores')
        pb = 2 * (HD // 32) * 34; cache = z(rows * pb, torch.uint8)
        qa, out = z(T * NH * HD), z(T * NH * HD); rec = z(pa.ROWS_PER_BLOCK * NH * pa.max_splits() * pa.ATTN_RS)
        accepted(log, lambda: A.attend(qa, cache, p0, T, 1, 0.0625, out, rec, err, sel=sel), 'attend(sel=) past QSA_EXACT')
        refused(log, lambda: A.attend(qa, cache, p0, T, 1, 0.0625, out, rec, err), 'attend without sel past QSA_EXACT (dense)', (k3.ContextBoundsError,))
        refused(log, lambda: A.attend(qa, cache[:6100 * pb], p0, T, 1, 0.0625, out, rec, err, sel=sel), 'attend(sel=): rows past the cache', (k3.ContextBoundsError,))
        refused(log, lambda: A.attend(qa, cache, p0, T, 1, 0.0625, out, rec, err, sel=sel[:-1]), 'attend(sel=): sel short')
        refused(log, lambda: A.attend(qa, cache, p0, T, 1, 0.0625, out, rec, err, sel=rec.view(torch.int32)[:T * SELW]), 'attend(sel=): the records overlapping sel')
    finally:
        torch.cuda.device = real
    g = capacity.geometry(coder)
    for ctx in (4096, 33024, 262144):
        a = capacity.prefill_attn_bytes(g, 2048, n_feat=5, qsa_ctx=ctx); b = capacity.prefill_attn_bytes(g, 2048, n_feat=5)
        need = capacity.prefill_need_mib(g, 2048, gdn_keys=10, moe_blocks=12, n_feat=5, qsa_ctx=ctx)
        base = capacity.prefill_need_mib(g, 2048, gdn_keys=10, moe_blocks=12, n_feat=5)
        lay = pa.attn_layout(2048, NH, HD, g['idx_dim'], qsa_ctx=ctx); pr = pa.pool_rows(ctx); sr = pa.score_rows(2048, pr)
        want = (2 * 4 * 2048 * QH * QD + 4 * 2048 * SELW + 4 * sr * pr)
        check(all(abs(need[c] - base[c] - (a[c] - b[c]) / capacity.MIB) < 1e-6 for c in (0, 1)) and a[0] - b[0] >= want and a[1] - b[1] >= want
              and lay['score'] >= 4 * sr * pr and 4 * sr * pr <= pa.SCORE_MIB * 2 ** 20 + 4 * 64 * pr,
              f'vram: max_ctx {ctx}: the QSA buffers (iq / iqo / sel / score scratch {sr} rows x {pr} pools = {4 * sr * pr / 2 ** 20:.1f} MiB) in '
              f'capacity.prefill_need_mib(qsa_ctx=): +{(a[0] - b[0]) / capacity.MIB:.1f} MiB per card')
        NUMBERS[f'vram_{ctx}'] = dict(qsa_mib=(a[0] - b[0]) / capacity.MIB, score_rows=sr, prefill_need_mib=need)


# ---- guarded allocations (the run1 canary method) -------------------------------------------------------------------------
class Guarded:
    """[GUARD sentinel | body | GUARD sentinel], one allocation, filled with the sentinel; take() snapshots it, changed()
    returns the int32 words that changed outside the allowed body byte ranges"""
    def __init__(self, torch, dev, nbytes, name):
        nbytes = (int(nbytes) + 255) // 256 * 256
        self.torch, self.name, self.nb = torch, name, nbytes
        self.a = torch.empty(GUARD + nbytes + GUARD, dtype=torch.uint8, device=dev)
        self.a32 = self.a.view(torch.int32); self.a32.fill_(SENT)
        self.body = self.a[GUARD:GUARD + nbytes]
        self.snap = torch.empty_like(self.a); self.d = torch.empty(self.a32.numel(), dtype=torch.bool, device=dev)

    def take(self):
        self.snap.copy_(self.a)

    def changed(self, allowed):
        torch = self.torch
        torch.ne(self.a32, self.snap.view(torch.int32), out=self.d)
        for lo, hi in allowed:
            lo, hi = int(lo), int(hi)
            assert 0 <= lo <= hi <= self.nb and lo % 4 == 0, (self.name, lo, hi)
            self.d[(GUARD + lo) // 4:(GUARD + hi + 3) // 4] = False
        n = int(self.d.sum().item())
        return n, ([] if not n else [int(v) * 4 - GUARD for v in self.d.nonzero()[:6].view(-1).tolist()])

    def guards_ok(self):
        return bool((self.a32[:GUARD // 4] == SENT).all().item()) and bool((self.a32[(GUARD + self.nb) // 4:] == SENT).all().item())


class K3Qsa:
    """the QSA tape's attention-layer core on one card's shapes as a K3 tape (k3_gguf_gsq_qsa, T = 1), one token per pass"""
    def __init__(self, torch, dev, P, rows):
        from caminus import k3
        self.torch, self.k3, self.P, self.rows = torch, k3, P, rows
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
        self.qf, self.kr, self.vr, self.kraw, self.iq = z(NH * 2 * HD), z(HD), z(HD), z(QD), z(QH * QD)
        self.qo, self.ko, self.att, self.iqo = z(NH * HD), z(HD), z(NH * HD), z(QH * QD)
        self.pos = z(1, dt=torch.int32); self.prow = (rows + 3) // 4
        self.cache = z(rows * 2 * (HD // 32) * 34, dt=torch.uint8)
        self.raw, self.pool = z(rows * ROWB, dt=torch.uint8), z(self.prow * ROWB, dt=torch.uint8)
        self.score, self.sel = z(1, self.prow), z(SELW, dt=torch.int32)
        t = lambda a: torch.tensor(a, device=dev)
        self.qn, self.kn, self.iqn, self.ikn = t(P['qn']), t(P['kn']), t(P['iqn']), t(P['ikn'])
        e = k3.Engine(module='k3_gguf_gsq', grid=k3.resident_grid('k3_gguf_gsq'), T=1)
        kd = e.qk_nr(self.kr, HD, 1, HD, self.kn, self.pos, self.ko, P['nrot'], P['theta'], EPS)
        kd = e.hadamard(self.ko, HD, dep=kd); vd = e.hadamard(self.vr, 64)
        ap = e.also(e.kv_append(self.ko, self.vr, self.cache, self.pos, q8=True, dep=kd), vd)
        ia = e.idx_append(self.kraw, self.raw.view(rows, ROWB), self.pool.view(self.prow, ROWB), self.pos, self.ikn, P['theta'], EPS, n_rot=P['nrot'])
        iqd = e.qk_nr(self.iq, QD, QH, QD, self.iqn, self.pos, self.iqo, P['nrot'], P['theta'], EPS)
        qd = e.qk_nr(self.qf, 2 * HD, NH, HD, self.qn, self.pos, self.qo, P['nrot'], P['theta'], EPS)
        qd = e.hadamard(self.qo, HD, dep=qd)
        sc = e.also(e.qsa_score(self.iqo, self.pool.view(self.prow, ROWB), self.pos, self.score, dep=iqd), ia)
        sl = e.also(e.qsa_select(self.score, self.pos, self.sel, self.prow, dep=sc), qd)
        a_ = e.also(e.attn_sel(self.qo, self.cache, self.pos, self.sel, self.att, NH, 1, HD, 1.0 / np.sqrt(HD), q8=True, dep=sl,
                               max_cells=min(rows, QSA_EXACT)), ap)
        e.hadamard(self.att, 64, rows_per_task=4, dep=a_)
        self.e = e.build()
        self.module = e.module

    def run(self, qf, k, v, kraw, iq, n):
        """positions 0 .. n - 1; returns (att [n, NH * HD], iqo [n, 512], scores {pos: row[:nv]} for nv > 512, sel [n, SELW])"""
        torch = self.torch
        att, iqo = torch.zeros(n, NH * HD, device=qf.device), torch.zeros(n, QH * QD, device=qf.device)
        sel = torch.zeros(n, SELW, dtype=torch.int32, device=qf.device); scores = {}
        for t in range(n):
            self.qf.copy_(qf[t].view(-1)); self.kr.copy_(k[t]); self.vr.copy_(v[t]); self.kraw.copy_(kraw[t]); self.iq.copy_(iq[t].view(-1))
            self.pos.fill_(t); self.e.run()
            att[t].copy_(self.att); iqo[t].copy_(self.iqo); sel[t].copy_(self.sel)
            nv = (t + 1) // 4
            if nv > 512: scores[t] = self.score[0, :nv].clone()
        return att, iqo, scores, sel


class PfQsa:
    """the same chain on the prefill's kernels, chunk by chunk, every buffer the new kernels write guarded"""
    def __init__(self, torch, dev, P, rows, srows=None, rpb=None, score_kernel=None):
        from caminus import prefill_attn as pa
        self.torch, self.pa, self.P, self.dev, self.rows = torch, pa, P, dev, rows
        self.A = pa.Attn(dev); self.prow = (rows + 3) // 4
        self.srows = srows or pa.score_rows(2048, self.prow); self.rpb = rpb or pa.ROWS_PER_BLOCK
        t = lambda a: torch.tensor(a, device=dev)
        self.qn, self.kn, self.iqn, self.ikn = t(P['qn']), t(P['kn']), t(P['iqn']), t(P['ikn'])
        self.cache = torch.zeros(rows * 2 * (HD // 32) * 34, dtype=torch.uint8, device=dev)
        self.raw = torch.zeros(rows * ROWB, dtype=torch.uint8, device=dev)
        G = lambda n, nm: Guarded(torch, dev, n, nm)
        self.gpool, self.gscore, self.gsel = G(self.prow * ROWB, 'IDXPOOL'), G(4 * self.srows * self.prow, 'score'), G(4 * rows * SELW, 'SEL')
        self.grec = G(4 * 128 * NH * pa.max_splits() * pa.ATTN_RS, 'records'); self.gout = G(4 * rows * NH * HD, 'attention out')
        self.gerr = G(4, 'err')
        self.pool = self.gpool.body[:self.prow * ROWB]; self.pool.zero_()
        self.score = self.gscore.body.view(torch.float32)[:self.srows * self.prow]
        self.sel = self.gsel.body.view(torch.int32)[:rows * SELW]
        self.rec = self.grec.body.view(torch.float32)[:128 * NH * pa.max_splits() * pa.ATTN_RS]
        self.att = self.gout.body.view(torch.float32)[:rows * NH * HD]
        self.err = self.gerr.body.view(torch.int32)[:1]; self.err.zero_()
        self.longest, self.canary_fail = {}, []
        self.sk = score_kernel

    def _guarded(self, name, fn, regions, inputs):
        """run one new-kernel launch with the canary checks: regions [(Guarded, allowed byte ranges)], inputs: tensors that
        must stay unchanged"""
        torch = self.torch
        for G_, _ in regions: G_.take()
        ins = [x.clone() for x in inputs]
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize(self.dev)
        self.longest[name] = max(self.longest.get(name, 0.0), e0.elapsed_time(e1))
        for G_, allowed in regions:
            n, where = G_.changed(allowed)
            if not G_.guards_ok() or n:
                self.canary_fail.append(f'{name}: {G_.name}: guards {"ok" if G_.guards_ok() else "CHANGED"}, {n} words changed outside its range (at {where})')
        u8 = lambda x: x.reshape(-1).view(torch.uint8)            # (bytes: the scratch's unwritten entries are the NaN sentinel)
        if not all(torch.equal(u8(a), u8(b)) for a, b in zip(ins, inputs)): self.canary_fail.append(f'{name}: an input changed')

    def chunk(self, qf, k, v, kraw, iq, p, n):
        """positions p .. p + n - 1 (rows of the inputs at those positions)"""
        torch, A, P, pa = self.torch, self.A, self.P, self.pa
        sl = slice(p, p + n)
        ko = torch.zeros(n, HD, device=self.dev); qo = torch.zeros(n, NH * HD, device=self.dev); vv = v[sl].clone()
        A.qk_nr(k[sl].contiguous(), HD, 1, HD, self.kn, P['nrot'], P['theta'], EPS, p, ko, n); A.hadamard(ko, HD, n)
        A.hadamard(vv, 64, n * HD // 64)
        A.kv_append(ko, HD, vv, HD, HD, self.cache, p, n, self.err)
        ik = kraw[sl].contiguous()
        A.kv_append(ik, QD, ik.view(-1)[64:], QD, 64, self.raw, p, n, self.err)     # IDXRAW as ChunkStack writes it
        j0, j1 = p // 4, (p + n) // 4
        if j1 > j0:
            self._guarded('idx_pool', lambda: A.idx_pool(self.raw, self.pool, j0, j1, self.ikn, P['theta'], EPS, self.err),
                          [(self.gpool, [(j0 * ROWB, j1 * ROWB)]), (self.gerr, [])], [self.raw])
        qfs = qf[sl].contiguous()
        A.qk_nr(qfs, 2 * HD, NH, HD, self.qn, P['nrot'], P['theta'], EPS, p, qo, n); A.hadamard(qo, HD, n * NH)
        b = max(p, min(p + n, QSA_EXACT))
        att = self.att.view(self.rows, NH * HD)
        if b > p:   # dense rows
            o = att[p:b]
            self._guarded('attend dense', lambda: A.attend(qo[:b - p], self.cache, p, b - p, 1, 1.0 / np.sqrt(HD), o, self.rec, self.err, rows_per_block=self.rpb),
                          [(self.gout, [(4 * p * NH * HD, 4 * b * NH * HD)]), (self.grec, [(0, self.grec.nb)]), (self.gerr, [])], [qo, self.cache])
        if p + n > b:   # QSA rows
            m = p + n - b
            iqo = torch.zeros(m, QH * QD, device=self.dev)
            A.qk_nr(iq[b:p + n].contiguous(), QD, QH, QD, self.iqn, P['nrot'], P['theta'], EPS, b, iqo, m)
            self.iqo_rows = getattr(self, 'iqo_rows', {}); self.iqo_rows[b] = iqo
            sel = self.sel.view(self.rows, SELW)
            self.scores = getattr(self, 'scores', {})
            for s0 in range(b, p + n, self.srows):
                s1 = min(p + n, s0 + self.srows); R = s1 - s0
                allowed = [(4 * (r * self.prow), 4 * (r * self.prow + min((s0 + r + 1) // 4, self.prow))) for r in range(R)]
                self._guarded('qsa_score', lambda: A.qsa_score(iqo[s0 - b:], self.pool, s0, R, self.score, self.prow, kernel=self.sk),
                              [(self.gscore, allowed)], [iqo, self.pool])
                for r in range(R):
                    nv = (s0 + r + 1) // 4
                    self.scores[s0 + r] = self.score[r * self.prow:r * self.prow + nv].clone()
                sallow = [(4 * (s0 + r) * SELW, 4 * ((s0 + r) * SELW + 4 + min(512, (s0 + r + 1) // 4))) for r in range(R)]
                sv = sel[s0:s1].reshape(-1)
                self._guarded('qsa_select', lambda: A.qsa_select(self.score, self.prow, s0, R, self.prow, sv),
                              [(self.gsel, sallow)], [self.score])
            o = att[b:p + n]
            self._guarded('attend sel', lambda: A.attend(qo[b - p:], self.cache, b, m, 1, 1.0 / np.sqrt(HD), o, self.rec, self.err, rows_per_block=self.rpb,
                                                         sel=sel[b:p + n].reshape(-1)),
                          [(self.gout, [(4 * b * NH * HD, 4 * (p + n) * NH * HD)]), (self.grec, [(0, self.grec.nb)]), (self.gerr, [])],
                          [qo, self.cache, sel[b:p + n]])
        A.hadamard(att[p:p + n].reshape(-1), 64, n * NH * HD // 64, 4)

    def run(self, qf, k, v, kraw, iq, bounds):
        for a, b in bounds: self.chunk(qf, k, v, kraw, iq, a, b - a)
        self.torch.cuda.synchronize(self.dev)
        return self


def inputs(torch, dev, n, seed=7):
    rng = np.random.default_rng(seed)
    f = lambda *s, sc=1.0: torch.tensor((rng.standard_normal(s) * sc).astype(np.float32), device=dev)
    qf, k, v = f(n, NH, 2 * HD, sc=1.5), f(n, HD, sc=1.5), f(n, HD)
    kraw, iq = f(n, QD, sc=2.0), f(n, QH, QD, sc=1.0)
    for t in range(3, n, 97): k[t, 7] = 40.0
    for t in range(11, n, 89): iq[t] = 0.0                                  # every score 0: the lowest pools win
    for t in range(40, n, 50): kraw[t] = kraw[t - 4]                         # repeated keys: pools with tied scores
    for t in range(2300, n, 311): iq[t] = iq[t] * -1.0                       # (a sign flip: many relu zeros)
    return qf, k, v, kraw, iq


def stage_g1(torch, dev, coder, small, tmax):
    from caminus import prefill_attn as pa
    P = model_params(coder)
    name = torch.cuda.get_device_name(dev)
    print(f'g1: device {name}; RoPE n_rot {P["nrot"]}, theta_scale {P["theta"]!r}; compaction_proactiveness {comp()}; MemAvailable {meminfo()}', flush=True)
    N = 2120 if small else tmax
    rows = (N + 63) // 64 * 64 + 64
    qf, k, v, kraw, iq = inputs(torch, dev, N)
    K = K3Qsa(torch, dev, P, rows)
    t0 = time.time(); att_r, iqo_r, sc_r, sel_r = K.run(qf, k, v, kraw, iq, N); torch.cuda.synchronize(dev)
    raw_r, pool_r = K.raw.clone(), K.pool.clone()
    print(f'  K3 reference ({K.module}): {N} positions one token per pass, {time.time() - t0:.1f} s', flush=True)
    nq = sum(1 for t in range(N) if (t + 1) // 4 > 512)
    def compare(F, label):
        okr = same(F.raw[:N * ROWB], raw_r[:N * ROWB])
        np_ = N // 4; okp = same(F.pool[:np_ * ROWB], pool_r[:np_ * ROWB])
        bad_sc = [t for t, r in sc_r.items() if not same(F.scores[t], r)]
        selF = F.sel.view(rows, SELW)[:N].cpu().numpy(); selR = sel_r.cpu().numpy()
        bad_sel = [t for t in range(N) if t >= QSA_EXACT and not np.array_equal(selF[t, :4 + selR[t, 0]], selR[t, :4 + selR[t, 0]])]
        attF = F.att.view(rows, NH * HD)[:N]
        nd = int((bits(attF).reshape(N, -1) != bits(att_r).reshape(N, -1)).any(1).sum())
        okq = all(same(F.iqo_rows[b], iqo_r[b:b + F.iqo_rows[b].shape[0]]) for b in F.iqo_rows)
        oke = int(F.err.item()) == 0
        return check(okr and okp and not bad_sc and not bad_sel and nd == 0 and okq and oke,
                     f'g1 {label}: positions 0 .. {N - 1} == K3 one token per pass ({K.module}), bitwise: IDXRAW {"==" if okr else "!="}, '
                     f'IDXPOOL ({np_} pools) {"==" if okp else "!="}, indexer q {"==" if okq else "!="}, scores of {len(sc_r)} rows '
                     f'{"==" if not bad_sc else f"!= at {bad_sc[:4]}"}, SEL of {nq} QSA rows {"==" if not bad_sel else f"!= at {bad_sel[:4]}"}, '
                     f'attention output {"==" if nd == 0 else f"!= at {nd} rows"}, err {int(F.err.item())}')
    def chunks(step, start=0, first=None):
        b = [] ; a = start
        if first: b.append((0, first)); a = first
        while a < N: b.append((a, min(N, a + step))); a = min(N, a + step)
        return b
    t0 = time.time(); F0 = PfQsa(torch, dev, P, rows).run(qf, k, v, kraw, iq, chunks(2048)); t1 = time.time()
    compare(F0, f'chunks of 2048 ({t1 - t0:.1f} s with the per-launch canaries)')
    check(not F0.canary_fail, f'canaries (chunks of 2048): every new kernel launch left its guards and every byte outside its range unchanged, '
          f'inputs unchanged' + (f'; {F0.canary_fail[:3]}' if F0.canary_fail else ''))
    base = dict(raw=F0.raw[:N * ROWB].clone(), pool=F0.pool[:(N // 4) * ROWB].clone(), sel=F0.sel.view(rows, SELW)[:N].clone(),
                att=F0.att.view(rows, NH * HD)[:N].clone())
    variants = ([('chunks 2051 + rest', chunks(2048, first=QSA_EXACT), {})] if small else
                [('chunks 2051 + 2048 ...', chunks(2048, first=QSA_EXACT), {}), ('chunks 1000 + 3000 ...', chunks(3000, first=1000), {}),
                 ('chunks of 777', chunks(777), {}), ('score sub-blocks of 64 rows', chunks(2048), dict(srows=64)),
                 ('the score kernel v0 (op_qsa_score\'s butterflies)', chunks(2048), dict(score_kernel='v0')),
                 ('record blocks of 17 rows', chunks(2048), dict(rpb=17))])
    for label, bnd, kw in variants:
        F = PfQsa(torch, dev, P, rows, **kw).run(qf, k, v, kraw, iq, bnd)
        selF = F.sel.view(rows, SELW)[:N]
        hdr = selF[:, 0].cpu().numpy()
        ok_sel = all(np.array_equal(selF[t, :4 + hdr[t]].cpu().numpy(), base['sel'][t, :4 + hdr[t]].cpu().numpy()) for t in range(QSA_EXACT, N))
        ok = same(F.raw[:N * ROWB], base['raw']) and same(F.pool[:(N // 4) * ROWB], base['pool']) and ok_sel and same(F.att.view(rows, NH * HD)[:N], base['att'])
        check(ok and not F.canary_fail and int(F.err.item()) == 0, f'g2b: {label} == chunks of 2048, bitwise (IDXRAW, IDXPOOL, SEL, attention); canaries '
              f'{"clean" if not F.canary_fail else F.canary_fail[:2]}')
        for kk, vv in F.longest.items(): F0.longest[kk] = max(F0.longest.get(kk, 0.0), vv)
    lg = {kk: round(vv, 3) for kk, vv in F0.longest.items()}
    check(all(v < 50.0 for v in F0.longest.values()), f'launch time: the longest launch per new kernel (ms, events): {lg} < 50 ms')
    NUMBERS['g1'] = dict(N=N, longest_ms=lg, device=name)


def stage_speed(torch, dev, coder):
    """the new kernels at a 32K-depth chunk on synthetic pools"""
    from caminus import prefill_attn as pa
    P = model_params(coder); A = pa.Attn(dev)
    ctx = 33024; prow = pa.pool_rows(ctx); srows = pa.score_rows(2048, prow); p0, T = 30720, 2048
    rng = np.random.default_rng(3)
    raw = torch.tensor(rng.integers(-127, 128, (ctx, ROWB), dtype=np.int8).view(np.uint8), device=dev)
    raw.view(ctx, 4, 34)[:, :, :2].copy_(torch.tensor(np.full((ctx, 4), 0.01, np.float16).view(np.uint8).reshape(ctx, 4, 2), device=dev))
    pool = torch.zeros(prow * ROWB, dtype=torch.uint8, device=dev); err = torch.zeros(1, dtype=torch.int32, device=dev)
    ikn = torch.tensor(P['ikn'], device=dev); A.idx_pool(raw.view(-1), pool, 0, prow, ikn, P['theta'], EPS, err)
    q = torch.tensor(rng.standard_normal((T, QH, QD)).astype(np.float32), device=dev)
    score = torch.zeros(srows * prow, device=dev); sel = torch.zeros(T * SELW, dtype=torch.int32, device=dev)
    pb = 2 * (HD // 32) * 34; cache = torch.tensor(rng.integers(0, 256, ctx * pb, dtype=np.uint8), device=dev)
    cache.view(ctx, 16, 34)[:, :, :2].copy_(torch.tensor(np.full((ctx, 16), 0.02, np.float16).view(np.uint8).reshape(ctx, 16, 2), device=dev))
    qa = torch.tensor(rng.standard_normal((T, NH * HD)).astype(np.float32), device=dev); out = torch.zeros(T, NH * HD, device=dev)
    rec = torch.zeros(pa.ROWS_PER_BLOCK * NH * pa.max_splits() * pa.ATTN_RS, device=dev)
    def timed(fn, reps=5):
        ts = []
        for _ in range(reps):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); fn(); e1.record(); torch.cuda.synchronize(dev); ts.append(e0.elapsed_time(e1))
        return sorted(ts)[len(ts) // 2]
    ms = {}
    ms['idx_pool 512 pools'] = timed(lambda: A.idx_pool(raw.view(-1), pool, p0 // 4, (p0 + T) // 4, ikn, P['theta'], EPS, err))
    nsb = -(-T // srows)
    def scores_and_select():
        for s0 in range(0, T, srows):
            R = min(srows, T - s0)
            A.qsa_score(q[s0:], pool, p0 + s0, R, score, prow); A.qsa_select(score, prow, p0 + s0, R, prow, sel[s0 * SELW:])
    ms[f'qsa_score v0 sub-block {srows} rows'] = timed(lambda: A.qsa_score(q, pool, p0, srows, score, prow, kernel='v0'))
    s_v0 = score[:srows * prow].clone(); score.fill_(0.0)
    ms[f'qsa_score sub-block {srows} rows'] = timed(lambda: A.qsa_score(q, pool, p0, srows, score, prow))
    tri = torch.zeros(srows, prow, dtype=torch.bool, device=dev)
    for r in range(srows): tri[r, :min((p0 + r + 1) // 4, prow)] = True
    check(torch.equal(score.view(srows, prow)[tri].view(torch.int32), s_v0.view(srows, prow)[tri].view(torch.int32)),
          f'speed: the reduce-scatter score kernel == v0 (op_qsa_score\'s butterflies), bitwise, {int(tri.sum())} scores of {srows} rows at nv up to {prow}')
    ms[f'qsa_select {srows} rows'] = timed(lambda: A.qsa_select(score, prow, p0, srows, prow, sel))
    ms[f'score + select, {T} rows ({nsb} sub-blocks)'] = timed(scores_and_select)
    scores_and_select(); torch.cuda.synchronize(dev)
    ms[f'attend(sel=) {T} rows'] = timed(lambda: A.attend(qa, cache, p0, T, 1, 0.0625, out, rec, err, sel=sel))
    ms['attend(sel=) one 64-row block'] = timed(lambda: A.attend(qa[:64], cache, p0, 64, 1, 0.0625, out[:64], rec, err, sel=sel[:64 * SELW]))
    core = ms['idx_pool 512 pools'] + ms[f'score + select, {T} rows ({nsb} sub-blocks)'] + ms[f'attend(sel=) {T} rows']
    print(f'  speed ({torch.cuda.get_device_name(dev)}, positions {p0} .. {p0 + T - 1}, {prow} pool rows, compaction {comp()}): ' +
          ', '.join(f'{n} {v:.3f}' for n, v in ms.items()) + f'; the QSA-specific core per layer {core:.2f} ms, x 12 layers {12 * core:.1f} ms per chunk', flush=True)
    check(int(err.item()) == 0 and max(ms[f'qsa_score sub-block {srows} rows'], ms[f'qsa_select {srows} rows'], ms['attend(sel=) one 64-row block']) < 50.0,
          f'speed: err 0; the longest single launches (score {ms[f"qsa_score sub-block {srows} rows"]:.2f}, select {ms[f"qsa_select {srows} rows"]:.2f}, '
          f'one attention block {ms["attend(sel=) one 64-row block"]:.2f} ms) < 50 ms')
    NUMBERS['speed_ms'] = dict(ms, core=core, chunk_12_layers=12 * core)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('coder'); ap.add_argument('stages', nargs='+')
    ap.add_argument('--small', action='store_true'); ap.add_argument('--tmax', type=int, default=6200)
    ap.add_argument('--allow-xtx', action='store_true'); ap.add_argument('--json', default=None)
    a = ap.parse_args()
    if 'cpu' in a.stages: stage_cpu(a.coder)
    gpu = [s for s in a.stages if s != 'cpu']
    if gpu:
        import torch
        dev = torch.device('cuda:0'); torch.cuda.set_device(dev)
        nm = torch.cuda.get_device_name(dev)
        if 'XTX' in nm and not a.allow_xtx:
            check(False, f'the GPU stages run on the XT alone first (HIP_VISIBLE_DEVICES=1); cuda:0 is {nm} (--allow-xtx after the XT passed)')
        else:
            torch.cuda.set_per_process_memory_fraction(0.9, dev)
            for s in gpu:
                {'g1': lambda: stage_g1(torch, dev, a.coder, a.small, a.tmax), 'speed': lambda: stage_speed(torch, dev, a.coder)}[s]()
    if a.json: json.dump(dict(fails=FAILS, numbers=NUMBERS), open(a.json, 'w'), indent=1, default=str)
    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'))
    sys.exit(0 if not FAILS else 1)


if __name__ == '__main__':
    main()
