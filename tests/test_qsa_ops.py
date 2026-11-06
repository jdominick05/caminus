"""The QSA indexer ops (kernels/k3_qsa_ops.h: OP_IDX_APPEND, OP_QSA_SCORE, OP_QSA_SELECT) in the GSQ-RCO Coder's QSA
modules (k3_gguf_gsq_qsa T = 1, k3_gguf_gsq_qsa_mt T = 4: the trunk modules plus these ops), on ONE card (run it on the XT alone: HIP_VISIBLE_DEVICES=1).
  ref DIR [DIR..] : vs llama.cpp's own tensors from one-token decodes (build/gg_golden --lc 'qsa' jobs: indexer_k_raw,
                    indexer_pool_k_new, indexer_q, indexer_k, indexer_score, indexer_top_k of the dumped layers):
                    IDX_APPEND fed llama.cpp's raw keys -> its pooled keys before quantization (indexer_pool_k_new) and the
                    dequantized pooled rows (indexer_k) bitwise, the raw rows bitwise vs numpy's q8_0; SCORE fed its q and
                    its pooled keys -> indexer_score bitwise (T = 1, and T = 4 rows from 4 consecutive steps); SELECT on those
                    scores bitwise vs caminus.qsa.select_ref and set-equal to indexer_top_k except ties at the threshold
                    (counted); with two dumps of the same job (d8192a / d8192b) llama.cpp's own run-to-run selection drift
  synth [per_wgp] : seeded synthetic keys / queries over positions 0 .. 2111 (crossing nv 512 -> 513 at 2051 / 2052): T = 4
                    passes advancing 1 / 2 / 3 / 4 tokens (every pool phase; the rejected rows carry garbage that the next
                    pass overwrites) bitwise vs one-token steps (SCORE rows, SEL rows, IDXRAW, IDXPOOL); SEL vs select_ref;
                    IDXRAW vs numpy's q8_0; SELECT on adversarial scores (ties, zeros, -0, inf, nv 513 .. 65536) vs select_ref
  grids           : synth's T = 4 schedule at 3 / 6 / 7 / 8 workgroups per WGP, outputs bitwise equal across grids
  attn [w..]      : OP_ATTN_SEL vs OP_ATTN_DEC bitwise: the identity selection at <= 2051 cells, random selections past
                    2051 vs ATTN_DEC on the compacted cells, adversarial SEL rows (clamped, in bounds), other grids (w)
  time            : SCORE + SELECT per call at nv 512 / 2048 / 8192 / 65536 (T = 1 and 4)
usage: HIP_VISIBLE_DEVICES=1 guard.sh LOG SECS python -I tests/test_qsa_ops.py synth"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from caminus import k3, qsa

MOD = 'k3_gguf_gsq'
EPS = np.float32(1e-6)
TS = None   # theta_scale: libm powf(1e7, -2 / 64) as caminus.flashnext (= llama.cpp's host value)
RES = []
MODS = set()   # the code objects the tests ran on (the QSA modules: k3.QSA_MODULES)


def theta_scale():
    import ctypes
    libm = ctypes.CDLL('libm.so.6'); libm.powf.restype = ctypes.c_float; libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]
    return libm.powf(1e7, -2.0 / 64)


def ok(name, cond, info=''):
    RES.append((name, bool(cond)))
    print(f'{"ok  " if cond else "FAIL"} {name} {info}', flush=True)


def beq(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def ndiff(a, b):
    a = np.ascontiguousarray(a).view(np.uint8); b = np.ascontiguousarray(b).view(np.uint8)
    return int((a != b).sum()) if a.shape == b.shape else -1


class Ops:
    """one engine per (T, op set): buffers for a QSA layer of `rows` positions; tapes [append], [score], [select] or all"""
    def __init__(self, T, rows, ops=('append', 'score', 'select'), per_wgp=None, min_nv=0, kraw_dev=None):
        self.T, self.rows, self.prow = T, rows, (rows + 3) // 4
        cu = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device='cuda')
        self.k = cu(T, 128); self.q = cu(T, 4, 128); self.knorm = cu(128); self.pooled = cu(T, 128)
        self.raw = cu(rows, qsa.ROWB, dt=torch.uint8); self.pool = cu(self.prow, qsa.ROWB, dt=torch.uint8)
        self.score = cu(T, self.prow); self.sel = cu(T, qsa.SELW, dt=torch.int32); self.pos = cu(1, dt=torch.int32)
        self.E = k3.Engine(module=MOD, grid=k3.resident_grid(MOD, per_wgp), T=T, limit_us=200_000)
        d = -1
        if 'append' in ops: d = self.E.idx_append(self.k, self.raw, self.pool, self.pos, self.knorm, TS, float(EPS), pooled=self.pooled, dep=d)
        if 'score' in ops: d = self.E.qsa_score(self.q, self.pool, self.pos, self.score, min_nv=min_nv, dep=d)
        if 'select' in ops: d = self.E.qsa_select(self.score, self.pos, self.sel, self.prow, dep=d)
        self.E.build()
        MODS.add(self.E.module)

    def run(self, p):
        k3.check_span(p, self.T, self.rows, what='qsa test pass')
        self.pos.fill_(p); self.E.run(check=True)


# ---------------------------------------------------------------- llama.cpp references
class Ref:
    def __init__(self, d):
        self.d, self.ix = d, {}
        for l in open(os.path.join(d, 'index.tsv')):
            st, seq, name, ty, n0, n1, n2, n3, f = l.rstrip('\n').split('\t')
            self.ix[(int(st), name)] = (ty, (int(n0), int(n1), int(n2), int(n3)), f)
        self.steps = sorted({s for s, _ in self.ix})
        self.layers = sorted({int(n.rsplit('-', 1)[1]) for _, n in self.ix})

    def get(self, step, name):
        ty, ne, f = self.ix[(step, name)]
        a = np.fromfile(os.path.join(self.d, f), dtype=np.int32 if ty == 'i32' else np.float32)
        assert a.size == ne[0] * ne[1] * ne[2] * ne[3], (name, a.size, ne)
        return a, ne


def ref_layer(R, L, knorm_w):
    """the gates for layer L of one dump; returns per-step (score, ours sel, llama top_k) for the drift comparison"""
    steps = R.steps; S0, S1 = steps[0], steps[-1]
    rows = 4 * ((S1 + 4) // 4) + 4
    tag = f'{os.path.basename(R.d)} L{L}'
    # -- IDX_APPEND, T = 1: llama.cpp's raw keys step by step (pools ending at a dumped step with all members dumped)
    o1 = Ops(1, rows, ('append',)); o1.knorm.copy_(torch.from_numpy(knorm_w))
    kr = {s: R.get(s, f'indexer_k_raw-{L}')[0] for s in steps}
    raw_ok = pool_ok = dq_ok = True; npool = 0; pooled1 = {}
    for s in steps:
        o1.k.copy_(torch.from_numpy(kr[s]).view(1, 128)); o1.run(s)
        if not beq(o1.raw[s].cpu().numpy(), qsa.q8_0_rows(kr[s])[0]): raw_ok = False
        if s % 4 == 3 and s - 3 >= S0:
            new, ne = R.get(s, f'indexer_pool_k_new-{L}')
            mine = o1.pooled[0].cpu().numpy(); pooled1[s] = (mine.copy(), o1.pool[s // 4].cpu().numpy().copy())
            if not beq(mine, new[:128]): pool_ok = False; print(f'  {tag} pos {s}: pooled f32 differs in {ndiff(mine, new[:128]) // 4} values')
            nxt = [t for t in steps if t > s]
            if nxt:
                kd, kne = R.get(nxt[0], f'indexer_k-{L}')
                ref_row = kd.reshape(-1, 128)[s // 4]
                if not beq(qsa.dq_rows(o1.pool[s // 4].cpu().numpy())[0], ref_row): dq_ok = False
            npool += 1
    ok(f'{tag} IDX_APPEND T=1 raw rows == numpy q8_0', raw_ok, f'({len(steps)} positions)')
    ok(f'{tag} IDX_APPEND T=1 pooled f32 == indexer_pool_k_new', pool_ok and npool > 0, f'({npool} pools)')
    ok(f'{tag} IDX_APPEND T=1 IDXPOOL dequant == indexer_k', dq_ok and npool > 0)
    # -- IDX_APPEND, T = 4 from every phase: the same pooled keys / rows as T = 1
    t4_ok = True
    for ph in range(4):
        o4 = Ops(4, rows, ('append',)); o4.knorm.copy_(torch.from_numpy(knorm_w))
        p = S0 + ph
        if ph:   # positions S0 .. p - 1 one by one first (T = 1 engine's rows copied in)
            o4.raw.copy_(o1.raw)
        while p + 4 <= S1 + 1:
            o4.k.copy_(torch.from_numpy(np.stack([kr[p + r] for r in range(4)])))
            o4.run(p)
            for r in range(4):
                s = p + r
                if s in pooled1:
                    if not beq(o4.pooled[r].cpu().numpy(), pooled1[s][0]) or not beq(o4.pool[s // 4].cpu().numpy(), pooled1[s][1]): t4_ok = False
            p += 4
    ok(f'{tag} IDX_APPEND T=4 (4 phases) == T=1 pooled + IDXPOOL', t4_ok)
    # -- SCORE fed llama.cpp's q and pooled keys; SELECT on the scores
    sc_ok = inv_ok = True; sel_ok = True; sets = []; drift = {}
    o1s = Ops(1, rows, ('score', 'select'))
    for s in steps:
        q, _ = R.get(s, f'indexer_q-{L}'); kd, kne = R.get(s, f'indexer_k-{L}'); sc, sne = R.get(s, f'indexer_score-{L}')
        tk, _ = R.get(s, f'indexer_top_k-{L}')
        nv = qsa.nv_of(s); npl = kne[2]
        kr_ = kd.reshape(npl, 128)
        o1s.pool[:nv].copy_(torch.from_numpy(qsa.requant_rows(kr_[:nv])))
        o1s.q.copy_(torch.from_numpy(q).view(1, 4, 128)); o1s.run(s)
        mine = o1s.score[0, :nv].cpu().numpy()
        if not beq(mine, sc[:nv]): sc_ok = False; print(f'  {tag} pos {s}: score differs in {ndiff(mine, sc[:nv]) // 4} of {nv}')
        if not np.all(np.isneginf(sc[nv:])): inv_ok = False
        sel = o1s.sel[0].cpu().numpy(); ref_ids, thr = qsa.select_ref(sc, nv)
        cnt = int(sel[0])
        if not (cnt == len(ref_ids) and beq(sel[4:4 + cnt], ref_ids) and sel[1] == qsa.ntail_of(s) and sel[2] == nv and (nv <= 512 or (int(sel[3]) & 0xFFFFFFFF) == thr)):
            sel_ok = False
        ll = tk[tk < nv]   # live picks (invisible pools score -inf and are killed by llama.cpp's live mask)
        rep = qsa.tie_report(sc, nv, sel[4:4 + cnt], ll); rep['pos'] = s; rep['nv'] = nv; rep['ll_live'] = len(ll); sets.append(rep)
        drift[s] = (sc[:nv].copy(), np.sort(ll))
    ok(f'{tag} SCORE T=1 == indexer_score', sc_ok, f'({len(steps)} steps, nv {qsa.nv_of(S0)}..{qsa.nv_of(S1)})')
    ok(f'{tag} llama.cpp scores invisible pools -inf', inv_ok)
    ok(f'{tag} SELECT == select_ref (bitwise, header incl.)', sel_ok)
    nd = sum(r['ndiff'] for r in sets); bad = [r for r in sets if not r['diff_at_threshold']]
    ok(f'{tag} SELECT set == indexer_top_k live picks outside threshold ties', not bad,
       f'(steps with differences {sum(1 for r in sets if r["ndiff"])}, pools differing {nd}; max ties at threshold '
       f'{max(r["ties_at_threshold"] for r in sets)}; zero scores visible max {max(r["zero_scores"] for r in sets)}, zero picks max '
       f'{max(r["zero_picks"] for r in sets)})')
    for r in bad[:4]: print('   ', r)
    # -- SCORE / SELECT T = 4: q of 4 consecutive steps, pooled keys of the last one
    o4s = Ops(4, rows, ('score', 'select')); s4_ok = True; n4 = 0
    for i in range(0, len(steps) - 3):
        p = steps[i]
        if steps[i + 3] != p + 3: continue
        kd, kne = R.get(p + 3, f'indexer_k-{L}'); nvl = qsa.nv_of(p + 3)
        o4s.pool[:nvl].copy_(torch.from_numpy(qsa.requant_rows(kd.reshape(kne[2], 128)[:nvl])))
        o4s.q.copy_(torch.from_numpy(np.stack([R.get(p + r, f'indexer_q-{L}')[0].reshape(4, 128) for r in range(4)])))
        o4s.run(p); n4 += 1
        for r in range(4):
            nv = qsa.nv_of(p + r); sc, _ = R.get(p + r, f'indexer_score-{L}')
            if not beq(o4s.score[r, :nv].cpu().numpy(), sc[:nv]): s4_ok = False
            ref_ids, _ = qsa.select_ref(sc, nv); sel = o4s.sel[r].cpu().numpy()
            if not (sel[0] == len(ref_ids) and beq(sel[4:4 + sel[0]], ref_ids)): s4_ok = False
    ok(f'{tag} SCORE + SELECT T=4 rows == indexer_score / select_ref', s4_ok and n4 > 0, f'({n4} passes)')
    return drift, sets


def cmd_ref(dirs):
    from gguf import GGUFReader
    rd = GGUFReader(CODER, 'r'); T = {t.name: t for t in rd.tensors}
    drifts = {}
    for d in dirs:
        R = Ref(d)
        for L in R.layers:
            t = T[f'blk.{L}.indexer.k_norm.weight']
            kw = np.array(t.data, dtype=np.float32).reshape(-1).copy()
            drifts[(os.path.basename(d), L)] = ref_layer(R, L, kw)
    # llama.cpp's run-to-run drift: the same job dumped twice (names ending a / b)
    for (n, L), (dr, _) in drifts.items():
        if not n.endswith('a') or (n[:-1] + 'b', L) not in drifts: continue
        db, _ = drifts[(n[:-1] + 'b', L)]
        same_sc = all(beq(dr[s][0], db[s][0]) for s in dr if s in db)
        nset = sum(len(set(dr[s][1]) ^ set(db[s][1])) for s in dr if s in db)
        print(f'drift {n[:-1]} L{L}: scores bitwise equal across llama.cpp runs {same_sc}; top-k picks differing {nset} '
              f'over {len(dr)} steps (llama.cpp vs itself)')


# ---------------------------------------------------------------- synthetic
P = 2112


def synth_data(seed=1):
    g = np.random.default_rng(seed)
    kr = (g.standard_normal((P, 128)) * 2.0).astype(np.float32)
    q = g.standard_normal((P, 4, 128)).astype(np.float32)
    kn = (0.5 + g.random(128)).astype(np.float32)
    junk = (g.standard_normal((P + 8, 128)) * 3.0).astype(np.float32), g.standard_normal((P + 8, 4, 128)).astype(np.float32)
    return kr, q, kn, junk


def run_t1(kr, q, kn, per_wgp=None):
    o = Ops(1, P, per_wgp=per_wgp); o.knorm.copy_(torch.from_numpy(kn))
    scores, sels = [], []
    for p in range(P):
        o.k.copy_(torch.from_numpy(kr[p:p + 1])); o.q.copy_(torch.from_numpy(q[p:p + 1])); o.run(p)
        nv = qsa.nv_of(p)
        scores.append(o.score[0, :nv].cpu().numpy().copy()); sels.append(o.sel[0].cpu().numpy().copy())
    return scores, sels, o.raw.cpu().numpy(), o.pool.cpu().numpy()


def schedule(kind):
    """advances per T = 4 pass (accepted tokens 1..4)"""
    if kind == 0: return [4]
    if kind == 1: return [1, 2, 3, 4, 4, 3, 1, 2]
    if kind == 2: return [3]
    return [2, 1, 4, 3, 1, 1, 2, 4]


def run_t4(kr, q, kn, junk, kind, per_wgp=None):
    o = Ops(4, P, per_wgp=per_wgp); o.knorm.copy_(torch.from_numpy(kn))
    sch = schedule(kind); p = 0; i = 0; out = {}; starts = []
    while p + 4 <= P:
        a = sch[i % len(sch)]; i += 1
        k = np.concatenate([kr[p:p + a], junk[0][p + a:p + 4]]); qq = np.concatenate([q[p:p + a], junk[1][p + a:p + 4]])
        o.k.copy_(torch.from_numpy(k)); o.q.copy_(torch.from_numpy(qq)); o.run(p); starts.append(p)
        sc = o.score.cpu().numpy(); sl = o.sel.cpu().numpy()
        for r in range(a): out[p + r] = (sc[r, :qsa.nv_of(p + r)].copy(), sl[r].copy())
        p += a
    return out, o.raw.cpu().numpy(), o.pool.cpu().numpy(), p, starts


def cmd_synth(per_wgp=None):
    kr, q, kn, junk = synth_data()
    s1, l1, raw1, pool1 = run_t1(kr, q, kn, per_wgp)
    ok('synth IDXRAW (T=1) == numpy q8_0', beq(raw1, qsa.q8_0_rows(kr)))
    sel_ok = True
    for p in range(P):
        ids, thr = qsa.select_ref(s1[p], qsa.nv_of(p)); sl = l1[p]
        if not (sl[0] == len(ids) and beq(sl[4:4 + sl[0]], ids) and sl[1] == qsa.ntail_of(p) and sl[2] == qsa.nv_of(p)): sel_ok = False
    ok('synth SEL (T=1, 0..2111) == select_ref', sel_ok, f'(nv up to {qsa.nv_of(P - 1)})')
    # scores vs a float64 recomputation from the device's own pooled rows (sanity, not bitwise)
    kd = qsa.dq_rows(pool1); err = 0.0
    for p in (2050, 2051, 2052, P - 1):
        nv = qsa.nv_of(p); ref = (np.maximum(np.einsum('hd,jd->hj', q[p].astype(np.float64), kd[:nv].astype(np.float64)), 0).sum(0) / np.sqrt(128))
        err = max(err, float(np.abs(ref - s1[p]).max() / max(1e-30, np.abs(ref).max())))
    ok('synth SCORE close to float64 (sanity)', err < 1e-5, f'(max rel {err:.2e})')
    for kind in range(4):
        out, raw4, pool4, pend, starts = run_t4(kr, q, kn, junk, kind, per_wgp)
        bad = [p for p, (sc, sl) in out.items() if not (beq(sc, s1[p]) and beq(sl, l1[p]))]
        ph = sorted({s % 4 for s in starts}); cross = [s for s in starts if s <= 2051 < s + 4 or s <= 2052 < s + 4]
        ok(f'synth T=4 schedule {kind} rows == T=1 (SCORE + SEL)', not bad and len(out) == pend,
           f'({len(out)} rows, start phases {ph}, passes over 2051/2052 at {cross[:4]}; first bad {bad[:3]})')
        ok(f'synth T=4 schedule {kind} IDXRAW / IDXPOOL == T=1', beq(raw4[:pend], raw1[:pend]) and beq(pool4[:pend // 4], pool1[:pend // 4]))
    cmd_select_adv()


def cmd_select_adv():
    """SELECT on chosen scores (no SCORE op): ties, zeros, -0, inf, denormals, nv 513 .. 65536, T = 1 and 4"""
    g = np.random.default_rng(7)
    cases = []
    def add(name, s): cases.append((name, np.asarray(s, np.float32)))
    add('513 zeros', np.zeros(513)); add('1000 three levels', g.integers(0, 3, 1000) * 0.5)
    add('65536 uniform', g.random(65536)); add('65536 all equal', np.full(65536, 1.25))
    t = np.zeros(65536, np.float32); t[g.choice(65536, 400, replace=False)] = 3.0; t[g.choice(np.where(t == 0)[0], 600, replace=False)] = 2.0
    add('65536 400 top + 600 tied at the threshold', t)
    m = g.random(4096).astype(np.float32); m[::7] = -0.0; m[::11] = 0.0; m[5] = np.inf; m[9] = 1e-45; m[13] = -1.0
    add('4096 -0 / +0 / inf / denormal / negative', m)
    add('514 descending', np.arange(514, 0, -1)); add('2048 ascending', np.arange(2048))
    allok = True
    for T in (1, 4):
        for name, s in cases:
            nv = len(s); pos = 4 * nv - 1 - (T - 1)   # the last row sees nv pools
            rows = 4 * nv
            o = Ops(T, rows, ('select',))
            sc = np.zeros((T, o.prow), np.float32); sc[:, :nv] = s
            o.score.copy_(torch.from_numpy(sc)); o.run(pos)
            sl = o.sel.cpu().numpy()
            for r in range(T):
                nvr = qsa.nv_of(pos + r); ids, thr = qsa.select_ref(s, nvr)
                good = sl[r, 0] == len(ids) and beq(sl[r, 4:4 + sl[r, 0]], ids) and sl[r, 2] == nvr
                if not good: allok = False; print(f'  select {name} T={T} row {r}: count {sl[r, 0]} vs {len(ids)}')
    ok('SELECT adversarial cases == select_ref (T=1, T=4)', allok, f'({len(cases)} cases)')


def cmd_grids():
    kr, q, kn, junk = synth_data(3)
    base = None
    for w in (8, 3, 6, 7):
        out, raw4, pool4, pend, _ = run_t4(kr, q, kn, junk, 1, per_wgp=w)
        flat = (b''.join(out[p][0].tobytes() + out[p][1].tobytes() for p in sorted(out)), raw4[:pend].tobytes(), pool4[:pend // 4].tobytes())
        if base is None: base = flat; continue
        ok(f'grids per_wgp {w} == per_wgp 8 (SCORE, SEL, IDXRAW, IDXPOOL)', flat == base)


def cmd_time():
    g = np.random.default_rng(5)
    for nv in (512, 2048, 8192, 65536):
        for T in (1, 4):
            rows = 4 * nv; pos = rows - T; us = {}
            pool = torch.from_numpy(qsa.q8_0_rows(g.standard_normal(((rows + 3) // 4, 128)).astype(np.float32)))
            qv = torch.from_numpy(g.standard_normal((T, 4, 128)).astype(np.float32))
            for ops in (('score',), ('select',), ('score', 'select')):
                o = Ops(T, rows, ops, min_nv=0); o.pool.copy_(pool); o.q.copy_(qv)
                if ops == ('select',): o.score.normal_()
                for _ in range(3): o.run(pos)
                ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]; n = 20
                k3.check_span(pos, T, rows); o.pos.fill_(pos)
                ev[0].record()
                for _ in range(n): o.E.run(check=False)
                ev[1].record(); torch.cuda.synchronize()
                st = o.E.sync[2 * k3.MAXSEG:].tolist(); assert not st[1] and not st[4], st
                us['+'.join(ops)] = ev[0].elapsed_time(ev[1]) / n * 1e3
            print(f'time nv {nv:6d} T={T}: launch us ' + ', '.join(f'{k} {v:7.1f}' for k, v in us.items()), flush=True)
    o = Ops(1, 64, ('append',)); ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    o.run(3); ev[0].record()
    for _ in range(20): o.E.run(check=False)
    ev[1].record(); torch.cuda.synchronize()
    print(f'time append T=1 (pool completing): launch {ev[0].elapsed_time(ev[1]) / 20 * 1e3:.1f} us', flush=True)


# ---------------------------------------------------------------- OP_ATTN_SEL (attention over the selected cells)
AH, AD = 12, 256   # a TP card's attention: 12 query heads over its one kv head of 256


def kv_rand(g, rows, q8):
    """a random KV cache [rows][K 256 | V 256]: q8_0 (544 B per position, block_q8_0 as llama.cpp's) or f16 (1024 B)"""
    x = (g.standard_normal((rows, 2 * AD)) * 1.5).astype(np.float32)
    if q8: return qsa.q8_0_rows(x.reshape(-1, 128)).reshape(rows, -1)
    return x.astype(np.float16).view(np.uint8).reshape(rows, -1)


class Attn:
    """one engine: OP_ATTN_DEC on the trunk module (sel None) or OP_ATTN_SEL on the QSA module, T rows, a cache of rows"""
    def __init__(self, T, rows, q8, sel=False, per_wgp=None):
        cu = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device='cuda')
        self.T, self.rows = T, rows
        self.cache = cu(rows * (544 if q8 else 1024), dt=torch.uint8); self.q = cu(T, AH, AD); self.out = cu(T, AH, AD)
        self.pos = cu(1, dt=torch.int32); self.sel = cu(T, qsa.SELW, dt=torch.int32)
        self.E = k3.Engine(module=MOD, grid=k3.resident_grid(MOD, per_wgp), T=T, limit_us=200_000)
        args = (self.q, self.cache, self.pos)
        if sel: self.E.attn_sel(*args, self.sel, self.out, AH, 1, AD, 1.0 / 16, q8=q8)
        else: self.E.attn_dec(*args, self.out, AH, 1, AD, 1.0 / 16, q8=q8)
        self.E.build(); MODS.add(self.E.module)

    def run(self, p, q, cache=None, sel=None):
        k3.check_span(p, self.T, self.rows, what='attn test pass')
        if cache is not None: self.cache.copy_(torch.from_numpy(np.ascontiguousarray(cache).reshape(-1)))
        if sel is not None: self.sel.copy_(torch.from_numpy(np.ascontiguousarray(sel, np.int32)))
        self.q.copy_(torch.from_numpy(q)); self.pos.fill_(p); self.E.run(check=True)
        return self.out.cpu().numpy().copy()


def sel_row(ids, pos):
    """a SEL row: [count, tail, nv, 0, ids ...]"""
    r = np.zeros(qsa.SELW, np.int32); r[0] = len(ids); r[1] = qsa.ntail_of(pos); r[2] = qsa.nv_of(pos); r[4:4 + len(ids)] = ids
    return r


def cells_of(row, pos):
    """the cell list OP_ATTN_SEL walks for a SEL row (with its clamps)"""
    nv = qsa.nv_of(pos); n = min(max(int(row[0]), 0), min(nv, qsa.TOPK))
    if n == nv: return np.arange(pos + 1, dtype=np.int64)   # every pool: the identity (the ids are not read)
    ids = np.clip(row[4:4 + n].astype(np.int64), 0, max(nv - 1, 0))
    c = (4 * ids[:, None] + np.arange(4)[None]).reshape(-1)
    return np.concatenate([c, np.arange(4 * nv, pos + 1)]).astype(np.int64)


def cmd_attn(per_wgps=(None,)):
    """OP_ATTN_SEL vs OP_ATTN_DEC (trunk module), bitwise: (1) the identity selection at positions 0 .. 2050 (T = 1 and 4,
    q8_0 and f16) == ATTN_DEC on the same cache; (2) random selections past 2051 (nv 513 .. 2048, incl. empty tails) ==
    ATTN_DEC run on a compacted cache holding exactly the listed cells in list order; (3) adversarial SEL rows (counts
    < 0 / > 512, ids < 0 / >= nv, an empty list) stay inside the cache and equal the clamped list's ATTN_DEC (empty: zeros);
    (4) per_wgps: the same outputs at other grids"""
    g = np.random.default_rng(11)
    R = 8200
    for q8 in (True, False):
        cache = kv_rand(g, R, q8); tag = 'q8_0' if q8 else 'f16'
        for T in (1, 4):
            dec = Attn(T, R, q8); sel = Attn(T, R, q8, sel=True)
            idp = [0, 1, 2, 3, 4, 7, 510, 511, 512, 513, 1023, 1024, 2043, 2047 - T + 1, 2050 - T + 1]
            bad = []
            for p in idp:
                q = g.standard_normal((T, AH, AD)).astype(np.float32)
                rows = np.stack([sel_row(np.arange(qsa.nv_of(p + r)), p + r) for r in range(T)])
                a = dec.run(p, q, cache); b = sel.run(p, q, cache, rows)
                if not beq(a, b): bad.append(p)
            ok(f'ATTN_SEL identity == ATTN_DEC ({tag}, T={T})', not bad, f'({len(idp)} positions up to {idp[-1] + T - 1}; bad {bad[:4]})')
            # random selections, compacted reference (one-token ATTN_DEC per row on its own compacted cache)
            ref1 = Attn(1, 2052, q8); bad = []; npass = 0
            for p in [2051, 2052, 2053, 2054, 2055, 3000, 4095, 4096, 6143, 8191 - T + 1]:
                q = g.standard_normal((T, AH, AD)).astype(np.float32)
                rows = []
                for r in range(T):
                    nv = qsa.nv_of(p + r); rows.append(sel_row(np.sort(g.choice(nv, qsa.TOPK, replace=False)), p + r))
                rows = np.stack(rows); b = sel.run(p, q, cache, rows); npass += 1
                for r in range(T):
                    cl = cells_of(rows[r], p + r); comp = np.zeros((2052, cache.shape[1]), np.uint8); comp[:len(cl)] = cache[cl]
                    a = ref1.run(len(cl) - 1, q[r:r + 1], comp)
                    if not beq(a[0], b[r]): bad.append((p, r))
            ok(f'ATTN_SEL random selections == ATTN_DEC on the compacted cells ({tag}, T={T})', not bad, f'({npass} passes; bad {bad[:4]})')
            # adversarial rows (garbage counts / ids): in bounds, == the clamped list; an empty list -> zeros
            bad = []
            for case in range(6):
                p = [2051, 4095, 8191 - T + 1, 2051, 4099, 2054][case]
                q = g.standard_normal((T, AH, AD)).astype(np.float32)
                rows = np.stack([sel_row(np.sort(g.choice(qsa.nv_of(p + r), qsa.TOPK, replace=False)), p + r) for r in range(T)])
                if case == 0: rows[:, 0] = -5
                if case == 1: rows[:, 0] = 10 ** 9
                if case == 2: rows[:, 4:10] = [-7, -1, 2 ** 31 - 1, 99999, -2 ** 31, 5]
                if case == 3: rows[:, 4:] = 2 ** 30
                if case == 4: rows[:, 0] = 0          # (pos + 1) % 4 == 0 at row 0 of 4099: an empty list there
                if case == 5: rows[:, 4:] = -(2 ** 30)
                b = sel.run(p, q, cache, rows)
                for r in range(T):
                    cl = cells_of(rows[r], p + r)
                    assert len(cl) == 0 or (cl.min() >= 0 and cl.max() <= p + r < R), (case, cl)
                    if len(cl) == 0:
                        if not (b[r] == 0).all(): bad.append((case, r, 'nonzero'))
                        continue
                    comp = np.zeros((2052, cache.shape[1]), np.uint8); comp[:len(cl)] = cache[cl]
                    a = ref1.run(len(cl) - 1, q[r:r + 1], comp)
                    if not beq(a[0], b[r]): bad.append((case, r))
            ok(f'ATTN_SEL adversarial SEL rows == clamped list / zeros ({tag}, T={T})', not bad, f'(6 cases; bad {bad[:4]})')
    # grids: the T = 4 random-selection pass at other grids
    for w in per_wgps:
        if w is None: continue
        g2 = np.random.default_rng(5); cache = kv_rand(g2, R, True)
        outs = []
        for ww in (8, w):
            e = Attn(4, R, True, sel=True, per_wgp=ww); o = []
            for p in (2051, 5000, 8188):
                q = np.random.default_rng(p).standard_normal((4, AH, AD)).astype(np.float32)
                rows = np.stack([sel_row(np.sort(np.random.default_rng(p + r).choice(qsa.nv_of(p + r), qsa.TOPK, replace=False)), p + r) for r in range(4)])
                o.append(e.run(p, q, cache, rows))
            outs.append(np.stack(o))
        ok(f'ATTN_SEL grids per_wgp {w} == per_wgp 8', beq(outs[0], outs[1]))


from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
CODER = devpaths.coder()

if __name__ == '__main__':
    sys.path.insert(0, os.path.expanduser('~/src/ignite-llama/gguf-py'))
    TS = theta_scale()
    torch.cuda.set_per_process_memory_fraction(0.9, 0)
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'synth'
    if cmd == 'ref': cmd_ref(sys.argv[2:])
    elif cmd == 'synth': cmd_synth(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == 'grids': cmd_grids()
    elif cmd == 'time': cmd_time()
    elif cmd == 'attn': cmd_attn(tuple(int(v) for v in sys.argv[2:]) or (None,))
    else: raise SystemExit(__doc__)
    nf = sum(1 for _, g in RES if not g)
    print(f'RESULT {cmd}: {len(RES) - nf} ok, {nf} FAIL (modules {sorted(MODS)})')
    sys.exit(1 if nf else 0)
