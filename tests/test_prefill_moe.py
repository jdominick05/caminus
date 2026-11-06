"""Gates of the batched prefill's own MoE GEMMs (caminus.moe, kernels/prefill_moe.hip): the grouping, the fused gate / up +
swiglu + down-input quantization, and the down projection over flat tile lists, against the vendored MMQ path they replace
(caminus.mmq: mm_ids_helper, quantize_scatter, mul_mat_id gate / up, pf_swiglu, the per-slot quantize, mul_mat_id down).

  python -I tests/test_prefill_moe.py CODER STAGE... [--real DIR] [--json OUT] [--layers L,..] [--js J,..]

Stages (one card, cuda:0: HIP_VISIBLE_DEVICES picks it; the XT first for a changed kernel):
  cpu    CPU only, nothing launched: moe.MoE's host refusals (short views, overlaps, misaligned tile lists, bad shapes,
         a weight allocation without MMQ's over-read tail, no fused kernel for a config) through a launch recorder, with
         the positive controls on the same buffers
  g1     real routed inputs (--real: the FFN half's `mixed` rows and the router's ids of a two-card Coder prefill, captured
         per layer, one layer per (gate / up, down) format pair) on each card's real weight shards (gate / up rows,
         down K slice with the zero tail), at n = 2048 and its micro-chunk pieces (1024 + 1024, 512): the grouping's tables
         == mm_ids_helper's (bounds, ids_dst, the inverse map; the down's forward map == ids_dst); every compiled J of the
         fused kernel: the down input bytes == MMQ's (quantize of pf_swiglu(mul_mat_id gate, up)); every J of the down
         kernel: the per-slot outputs == mul_mat_id down's, bitwise; the single-GEMM kernel on gate and up (K = 2560) ==
         mul_mat_id's; the routing flag word clear
  edge   synthetic routing on the same weights: an expert taking every token, empty experts, a token count of 1 / 3 / 777,
         bitwise as g1; dst prefilled with NaN: every (token, slot) row written (--small: the 1- and 3-token cases only,
         a changed kernel's first run)
  g2b    the same 2048 tokens as one chunk and split (1024 + 1024, 777 + 1271, 512 + 788 + 748, 1 + 2047), each piece grouped
         on its own: every (token, slot) output of the down projection and every down input row bitwise the one-chunk run
  speed  per layer at n = 512 / 1024 / 2048 (real ids, the layers of --layers): the MMQ path (ChunkStack's sequence before
         this kernel: two groupings, quantize_scatter, mul_mat_id gate + up, pf_swiglu, the per-slot quantize, mul_mat_id
         down) against ours (one grouping, quantize_scatter, the fused kernel, the down kernel) per J, interleaved, medians
         of --reps; the longest single launch
Prints PASS / FAIL per check and RESULT PASS / FAIL; exit 0 = PASS."""
import argparse, json, os, re, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'python')); sys.path.insert(0, HERE)
FAILS, NUMBERS = [], {}
from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
ART = devpaths.artifacts()
REAL = os.path.join(ART, 'impl-moe-gemm', 'real')
NEXP, KU, D = 256, 10, 2560
MBR = [(0, 12), (12, 20)]        # FlashNextTP's moe_blocks = 12: card 0 ffc 384, card 1 256 (32-column blocks)


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)
    return ok


def bits(t):
    a = t.cpu().numpy() if hasattr(t, 'cpu') else t
    return np.ascontiguousarray(a).view(np.uint8)


def same(a, b):
    return a.shape == b.shape and np.array_equal(bits(a), bits(b))


# ---- weights ------------------------------------------------------------------------------------------------------------
class Layer:
    """one layer's MoE shards for card c on the device: gate / up rows [n_exp][ffc][2560], down [n_exp][2560][ffc] (+ a zeroed
    tail: MMQ's over-read, as FlashNextTP.ksplit_pad)"""
    def __init__(self, torch, dev, rd, il, c):
        from caminus import mmq
        from gguf.constants import GGML_QUANT_SIZES
        tens = {t.name: t for t in rd.tensors}
        a, b = MBR[c]; self.ffc = 32 * (b - a); self.il, self.c = il, c
        def put(raw):
            nb = raw.size; t = torch.zeros(nb + 512, dtype=torch.uint8, device=dev)
            t[:nb] = torch.from_numpy(np.ascontiguousarray(raw).reshape(-1)).to(dev); return t[:nb]
        out = {}
        for kind in ('gate', 'up', 'down'):
            t = tens[f'blk.{il}.ffn_{kind}_exps.weight']; fmt = t.tensor_type.name; K = int(t.shape[0]); E, rows = int(t.shape[2]), int(t.shape[1])
            raw = np.asarray(t.data).reshape(-1).view(np.uint8)
            if kind != 'down':
                rb = mmq.row_bytes(fmt, K); sh = raw.reshape(E, rows, rb)[:, 32 * a:32 * b]
            else:
                bs, ts = GGML_QUANT_SIZES[t.tensor_type]; sh = raw.reshape(E, rows, K // bs, ts)[:, :, 32 * a // bs:32 * b // bs]
            out[kind] = (fmt, put(sh))
        (self.fg, self.wg), (self.fu, self.wu), (self.fd, self.wd) = out['gate'], out['up'], out['down']
        assert self.fg == self.fu


# ---- the two paths ------------------------------------------------------------------------------------------------------
class Paths:
    """buffers for chunks of up to T tokens and both paths' launch sequences on one device"""
    def __init__(self, torch, dev, T, ffc_max=384):
        from caminus import mmq, moe, prefill_tp
        self.torch, self.dev, self.T = torch, dev, T
        self.M = mmq.MMQ(dev, formats=moe.FORMATS); self.O = moe.MoE(dev, self.M); self.P = prefill_tp.Tp(dev)
        N = T * KU; z = lambda n, dt: torch.zeros(n, dtype=dt, device=dev)
        I32, U8, F32 = torch.int32, torch.uint8, torch.float32
        self.ids = z(N, I32); self.err = z(1, I32)
        self.s1, self.sd, self.bd, self.sf = z(N, I32), z(N + mmq.J_MAX - 1, I32), z(NEXP + 1, I32), z(N, I32)
        self.y = z(mmq.y_bytes(D, N), U8); self.yd_ref = z(mmq.y_bytes(ffc_max, N), U8); self.yd = z(mmq.y_bytes(ffc_max, N), U8)
        self.og, self.ou, self.xmd = z(N * ffc_max, F32), z(N * ffc_max, F32), z(N * ffc_max, F32)
        self.od_ref, self.od = z(N * D, F32), z(N * D, F32)
        self.ob, self.oi, self.oinv = z(NEXP + 1, I32), z(N, I32), z(N, I32)
        mt = moe.max_tiles(N, NEXP, moe.J_MIN)
        self.ta, self.tb, self.nt = z(4 * mt, I32), z(4 * mt, I32), z(2, I32)
        self.mmq, self.moe = mmq, moe

    def set_ids(self, ids):
        n = ids.shape[0]; self.ids[:n * KU].copy_(self.torch.from_numpy(np.ascontiguousarray(ids, dtype=np.int32).reshape(-1)).to(self.dev))
        return n

    def ref(self, L, x, n, ids_np, groups=True):
        """the MMQ path (ChunkStack's sequence before this step) for n tokens: x fp32 [n, 2560] on the device"""
        M, P, mmq = self.M, self.P, self.mmq; N = n * KU; ffc = L.ffc
        if groups: self.set_ids(ids_np)
        M.group(self.ids[:N], n, KU, NEXP, self.s1, self.sd, self.bd, False, self.err)
        M.quantize_scatter(x, D, n, self.s1, KU, self.y, M.layout(L.fg))
        M.mm_id(L.fg, L.wg, NEXP, ffc, D, self.y, n, KU, self.sd, self.bd, self.og)
        M.mm_id(L.fu, L.wu, NEXP, ffc, D, self.y, n, KU, self.sd, self.bd, self.ou)
        P.swiglu(self.og, self.ou, self.xmd, N * ffc)
        M.group(self.ids[:N], n, KU, NEXP, self.sf, self.sd, self.bd, True, self.err)
        M.quantize(self.xmd, ffc, N, self.yd_ref, M.layout(L.fd), ids=self.sf)
        M.mm_id(L.fd, L.wd, NEXP, D, ffc, self.yd_ref, n, KU, self.sd, self.bd, self.od_ref)

    def own(self, L, x, n, ids_np, jg, jd, groups=True):
        """ours: one grouping, quantize_scatter, the fused gate / up kernel, the down kernel"""
        M, O = self.M, self.O; N = n * KU; ffc = L.ffc
        if groups: self.set_ids(ids_np)
        O.group(self.ids[:N], n, KU, NEXP, self.ob, self.oi, self.oinv, self.ta, jg, self.tb, jd, self.nt, self.err)
        M.quantize_scatter(x, D, n, self.oinv, KU, self.y, M.layout(L.fg))
        O.gu(L.fg, L.wg, L.wu, NEXP, ffc, D, self.y, N, self.ta, self.nt[0:1], self.yd, self.err, jg, L.fd)
        O.mm(L.fd, L.wd, NEXP, D, ffc, self.yd, N, self.oi, self.tb, self.nt[1:2], self.od, self.err, jd)


def yd_valid(P, ffc, N):
    """the down input's blocks that hold values (k < ffc): the first ffc / 128 block rows of N columns"""
    return P.torch.cat([P.yd[:0]] + [P.yd[kb * N * 144:(kb + 1) * N * 144] for kb in range(ffc // 128)]), \
           P.torch.cat([P.yd_ref[:0]] + [P.yd_ref[kb * N * 144:(kb + 1) * N * 144] for kb in range(ffc // 128)])


def compare(P, L, n, tag, jg, jd, fills=False):
    torch = P.torch; N = n * KU; ffc = L.ffc; ok = True
    a, b = yd_valid(P, ffc, N)
    ok &= check(same(a, b), f'{tag} J {jg}/{jd}: the down input (fused gate / up + swiglu + q8_1) == MMQ\'s bytes ({ffc // 128} x {N} blocks)')
    o1, o0 = P.od[:N * D], P.od_ref[:N * D]
    ok &= check(same(o1, o0), f'{tag} J {jg}/{jd}: the down projection\'s {N} x {D} per-slot outputs == mul_mat_id\'s, bitwise')
    if fills: ok &= check(not bool(torch.isnan(o1).any().item()), f'{tag}: every per-slot row written (dst prefilled with NaN)')
    return ok


def tables_equal(P, n, tag):
    torch = P.torch; N = n * KU
    ok = check(torch.equal(P.ob, P.bd) and torch.equal(P.oi[:N], P.sd[:N]) and torch.equal(P.oinv[:N], P.s1[:N]) and torch.equal(P.sf[:N], P.sd[:N]),
               f'{tag}: the grouping == mm_ids_helper (bounds, ids_dst, the inverse map; the down\'s forward map == ids_dst)')
    nt = P.nt.cpu().numpy(); cnt = np.diff(P.bd.cpu().numpy())
    return ok, cnt, nt


def stage_cpu():
    import torch
    from caminus import moe, mmq
    from caminus.hip import Kernel
    calls = []
    class Rec:
        def __init__(self, name): self.name = name
        def __call__(self, grid, block, *a, **kw): calls.append(self.name)
    O = moe.MoE.__new__(moe.MoE); O.dev = torch.device('cpu'); O.kclamp, O.kgroup = Rec('clamp'), Rec('group')
    O.cfg, O.kmm, O.kgu = {}, {}, {}
    for f in moe.FORMATS:
        for J in moe.JS:
            O.cfg[(f, J)] = dict(nthreads=128, I=64, nbytes_mm=40000, nbytes_gu=60000 if f in moe.FORMATS_GU and J != 128 else 0)
            O.kmm[(f, J)] = Rec(f'mm {f} {J}')
            if O.cfg[(f, J)]['nbytes_gu']: O.kgu[(f, J)] = Rec(f'gu {f} {J}')
    O._launch = lambda k, *a, **kw: k(*a, **kw)
    T = 64; N = T * KU; z = lambda n, dt=torch.int32: torch.zeros(n, dtype=dt)
    mt = moe.max_tiles(N, NEXP, 16)
    bufs = dict(ids=z(N), bounds=z(NEXP + 1), ids_dst=z(N), ids_inv=z(N), ta=z(4 * mt), tb=z(4 * mt), nt=z(2), err=z(1))
    def grp(**over):
        b = dict(bufs); b.update(over)
        return lambda: O.group(b['ids'], T, KU, NEXP, b['bounds'], b['ids_dst'], b['ids_inv'], b['ta'], 16, b['tb'], 16, b['nt'], b['err'])
    def expect(fn, refused, what):
        n0 = len(calls)
        try: fn(); got = False
        except ValueError: got = True
        check(got == refused and (len(calls) == n0) == refused, f'cpu: {what} {"refused, nothing launched" if refused else "accepted, launched"}')
    expect(grp(), False, 'group on whole buffers')
    blk = z(4 * N + 4 * N)
    expect(grp(ids_dst=blk[:N - 1]), True, 'group: a short view of ids_dst')
    expect(grp(ids_inv=bufs['ids_dst']), True, 'group: ids_inv the same buffer as ids_dst')
    expect(grp(ids_inv=blk[N // 2:N // 2 + N], ids_dst=blk[:N]), True, 'group: ids_inv partially overlapping ids_dst')
    expect(grp(ta=z(4 * mt - 1)), True, 'group: a tile list shorter than its bound')
    expect(grp(nt=z(1)), True, 'group: one tile count word (two lists)')
    expect(lambda: O.group(bufs['ids'], T, KU, NEXP, bufs['bounds'], bufs['ids_dst'], bufs['ids_inv'], bufs['ta'], 24, bufs['tb'], 16, bufs['nt'], bufs['err']),
           True, 'group: a column tile that no kernel has (24)')
    expect(lambda: O.group(bufs['ids'], T, 17, NEXP, bufs['bounds'], bufs['ids_dst'], bufs['ids_inv'], bufs['ta'], 16, bufs['tb'], 16, bufs['nt'], bufs['err']),
           True, 'group: k 17 (> 16 slots)')
    expect(lambda: O.group(bufs['ids'], T, KU, 600, bufs['bounds'], bufs['ids_dst'], bufs['ids_inv'], bufs['ta'], 16, bufs['tb'], 16, bufs['nt'], bufs['err']),
           True, 'group: n_exp 600 (> the kernel\'s 512)')
    # the GEMMs, at the card-0 down / gate shapes, on CPU stand-ins (uint8 weights with the over-read tail)
    fmt, ffc = 'IQ4_NL', 384
    wn = NEXP * D * mmq.row_bytes(fmt, ffc); wfull = z(wn + 512, torch.uint8); w = wfull[:wn]
    y, od, oi = z(mmq.y_bytes(ffc, N), torch.uint8), z(N * D, torch.float32), z(N)
    def mmc(**over):
        b = dict(w=w, y=y, oi=oi, tb=bufs['tb'], nt=bufs['nt'][1:2], od=od, err=bufs['err']); b.update(over)
        return lambda: O.mm(fmt, b['w'], NEXP, D, ffc, b['y'], N, b['oi'], b['tb'], b['nt'], b['od'], b['err'], 16)
    expect(mmc(), False, 'mm (down IQ4_NL, K 384) on whole buffers')
    expect(mmc(w=torch.zeros(wn, dtype=torch.uint8)), True, 'mm: weights without the over-read tail (K 384)')
    expect(mmc(od=od[:N * D - 1]), True, 'mm: a short view of dst')
    expect(mmc(y=y[:mmq.y_bytes(ffc, N) - 1]), True, 'mm: a short activation view')
    expect(mmc(oi=oi[:N - 1]), True, 'mm: a short ids_dst view')
    expect(mmc(od=wfull.view(torch.float32)[:0].new_zeros(0)), True, 'mm: an empty dst')
    expect(mmc(err=od[:1].view(torch.int32)), True, 'mm: err inside dst')
    expect(lambda: O.mm(fmt, w, NEXP, D, 320, y, N, oi, bufs['tb'], bufs['nt'][1:2], od, bufs['err'], 16), True, 'mm: K 320 (not whole 128-value halves)')
    expect(lambda: O.mm(fmt, w, NEXP, 2500, ffc, y, N, oi, bufs['tb'], bufs['nt'][1:2], od, bufs['err'], 16), True, 'mm: rows 2500 (not a multiple of the row block)')
    gf = 'IQ2_S'; gn = NEXP * ffc * mmq.row_bytes(gf, D)
    wg, wu = z(gn + 512, torch.uint8)[:gn], z(gn + 512, torch.uint8)[:gn]
    yg, yd = z(mmq.y_bytes(D, N), torch.uint8), z(mmq.y_bytes(ffc, N), torch.uint8)
    def guc(J=16, **over):
        b = dict(wg=wg, wu=wu, y=yg, yd=yd, ta=bufs['ta'], err=bufs['err']); b.update(over)
        return lambda: O.gu(gf, b['wg'], b['wu'], NEXP, ffc, D, b['y'], N, b['ta'], bufs['nt'][0:1], b['yd'], b['err'], J, 'IQ4_NL')
    expect(guc(), False, 'gu (IQ2_S, ffc 384) on whole buffers')
    expect(guc(yd=yd[:mmq.y_bytes(ffc, N) - 16]), True, 'gu: a short down-input view')
    expect(guc(yd=yg[:mmq.y_bytes(ffc, N)]), True, 'gu: the down input inside the gate / up activations')
    expect(guc(J=128), True, 'gu: a config without the fused kernel (J 128 here)')
    expect(lambda: O.gu(gf, wg, wu, NEXP, ffc, D, yg, N, bufs['ta'], bufs['nt'][0:1], yd, bufs['err'], 16, 'IQ2_S'), True, 'gu: a down format whose layout is not D4-only (IQ2_S)')
    expect(lambda: O.gu(gf, wg, wu, NEXP, ffc, D, yg, N, bufs['ta'][1:], bufs['nt'][0:1], yd, bufs['err'], 16, 'IQ4_NL'), True, 'gu: a misaligned / short tile list')
    check(moe.max_tiles(N, NEXP, 16) >= NEXP + N // 16, 'cpu: the tile bound covers every expert\'s partial tile plus the full ones')


def load_real(name, il, n0=0, n=2048):
    x = np.load(os.path.join(REAL, f'{name}_L{il}_mixed.npy'))[n0:n0 + n]
    ids = np.load(os.path.join(REAL, f'{name}_ids_all.npy'))[il][n0:n0 + n]
    return np.ascontiguousarray(x), np.ascontiguousarray(ids)


def stage_g1(torch, rd, layers, js):
    dev = torch.device('cuda:0'); P = Paths(torch, dev, 2048)
    for il in layers:
        for c in (0, 1):
            L = Layer(torch, dev, rd, il, c); tagL = f'L{il} card {c} ({L.fg} / {L.fd}, ffc {L.ffc})'
            for name in ('d8192', 'd1792'):
                for n0, n in ((0, 2048), (0, 1024), (1024, 1024), (512, 512)):
                    xn, ids = load_real(name, il, n0, n); x = torch.from_numpy(xn).to(dev); tag = f'{tagL} {name} [{n0}, {n0 + n})'
                    P.err.zero_(); P.ref(L, x, n, ids)
                    jlist = [(J, J) for J in js] if (name, n0, n) == ('d8192', 0, 2048) else [(48, 48)]
                    for jg, jd in jlist:
                        if (L.fg, jg) not in P.O.kgu: continue
                        P.od.fill_(float('nan')); P.own(L, x, n, ids, jg, jd)
                        torch.cuda.synchronize(dev)
                        ok, cnt, nt = tables_equal(P, n, tag)
                        check(int(nt[0]) == int(np.ceil(cnt / jg).sum()) and int(nt[1]) == int(np.ceil(cnt / jd).sum()),
                              f'{tag}: the tile lists hold sum ceil(count / J) entries ({nt[0]} / {nt[1]})')
                        compare(P, L, n, tag, jg, jd, fills=True)
                    check(int(P.err.item()) == 0, f'{tag}: the routing flag word clear')
                    # the single-GEMM kernel at K = 2560 (gate, up) == mul_mat_id's outputs (on the J 48 lists of own())
                    if (name, n0, n) == ('d8192', 0, 2048):
                        P.own(L, x, n, ids, 48, 48)
                        for kind, fmt, w, ref in (('gate', L.fg, L.wg, P.og), ('up', L.fu, L.wu, P.ou)):
                            out = torch.full((n * KU * L.ffc,), float('nan'), device=dev)
                            P.O.mm(fmt, w, NEXP, L.ffc, D, P.y, n * KU, P.oi, P.ta, P.nt[0:1], out, P.err, 48)
                            torch.cuda.synchronize(dev)
                            check(same(out, ref[:n * KU * L.ffc]), f'{tagL}: the single-GEMM kernel on {kind} (K 2560, J 48) == mul_mat_id\'s, bitwise')
            del L
    NUMBERS['g1'] = 'ok' if not FAILS else 'fail'


def stage_edge(torch, rd, layers, small=False):
    dev = torch.device('cuda:0'); P = Paths(torch, dev, 2048); rng = np.random.default_rng(7)
    il = layers[0]
    for c in (0, 1):
        L = Layer(torch, dev, rd, il, c)
        xall, _ = load_real('d8192', il)
        cases = []
        others = np.array([e for e in range(NEXP) if e != 5])
        ids = np.stack([np.insert(rng.permutation(others)[:KU - 1], 3, 5) for _ in range(777)]).astype(np.int32)   # expert 5 in every token
        cases.append(('777 tokens, expert 5 in every token', ids))
        ids2 = np.tile(np.arange(KU, dtype=np.int32), (2048, 1)); cases.append(('2048 tokens, all on experts 0..9 (246 empty)', ids2))
        cases.append(('1 token', np.arange(KU, dtype=np.int32)[None, :] * 7))
        cases.append(('3 tokens', np.stack([rng.permutation(NEXP)[:KU] for _ in range(3)]).astype(np.int32)))
        if small: cases = [cs for cs in cases if cs[1].shape[0] <= 3]
        for what, ids in cases:
            assert all(len(set(r)) == KU for r in ids.tolist())
            n = ids.shape[0]; x = torch.from_numpy(np.ascontiguousarray(xall[:n])).to(dev); tag = f'edge L{il} card {c}: {what}'
            P.err.zero_(); P.ref(L, x, n, ids)
            for jg, jd in ((16, 16), (48, 48), (96, 32)):
                if (L.fg, jg) not in P.O.kgu: continue
                P.od.fill_(float('nan')); P.own(L, x, n, ids, jg, jd); torch.cuda.synchronize(dev)
                tables_equal(P, n, tag); compare(P, L, n, tag, jg, jd, fills=True)
            check(int(P.err.item()) == 0, f'{tag}: the routing flag word clear')
        # bad ids only into the CLAMPED path (k3_mmq_ids_clamp runs before both groupings, on the same stream): out of
        # range (300, -5, INT_MIN) and repeated within a token; both paths see the same clamped ids
        bad = np.stack([rng.permutation(NEXP)[:KU] for _ in range(64)]).astype(np.int32)
        bad[0, 3] = 300; bad[1, 2] = -5; bad[2, 5] = bad[2, 4]; bad[3, :] = 7; bad[4, 9] = -2 ** 31
        n = bad.shape[0]; x = torch.from_numpy(np.ascontiguousarray(xall[:n])).to(dev); tag = f'edge L{il} card {c}: 64 tokens with bad ids (clamped path)'
        P.err.zero_(); P.ref(L, x, n, bad); f_ref = int(P.err.item())
        P.err.zero_(); P.od.fill_(float('nan')); P.own(L, x, n, bad, 48, 32); torch.cuda.synchronize(dev); f_own = int(P.err.item())
        tables_equal(P, n, tag); compare(P, L, n, tag, 48, 32, fills=True)
        clamped = P.ids[:n * KU].view(n, KU).cpu().numpy()
        check(all(len(set(r)) == KU for r in clamped.tolist()) and clamped.min() >= 0 and clamped.max() < NEXP and
              np.array_equal(clamped[5:], bad[5:]), f'{tag}: the ids reaching the grouping are in range and distinct, valid tokens untouched')
        check(f_own == f_ref == 6, f'{tag}: the flag word = ROUTE_RANGE | ROUTE_DUP on both paths ({f_own:#x} / {f_ref:#x}), no ROUTE_TILES')
        del L


def stage_g2b(torch, rd, layers):
    dev = torch.device('cuda:0'); P = Paths(torch, dev, 2048)
    for il in layers[:2]:
        for c in (0, 1):
            L = Layer(torch, dev, rd, il, c); xn, ids = load_real('d8192', il); x = torch.from_numpy(xn).to(dev)
            N = 2048 * KU
            P.own(L, x, 2048, ids, 48, 48); torch.cuda.synchronize(dev)
            od1 = P.od[:N * D].clone(); yd1 = [P.yd[kb * N * 144:(kb + 1) * N * 144].view(N, 144).clone() for kb in range(L.ffc // 128)]
            oinv1 = P.oinv[:N].clone().long()
            for split in ((1024, 1024), (777, 1271), (512, 788, 748), (1, 2047)):
                okd = oky = True; t0 = 0
                for n in split:
                    xs = x[t0:t0 + n].contiguous(); Nn = n * KU
                    P.own(L, xs, n, ids[t0:t0 + n], 48, 48); torch.cuda.synchronize(dev)
                    okd &= same(P.od[:Nn * D], od1[t0 * KU * D:(t0 + n) * KU * D])
                    inv = P.oinv[:Nn].long()
                    for kb in range(L.ffc // 128):
                        mine = P.yd[kb * Nn * 144:(kb + 1) * Nn * 144].view(Nn, 144)[inv]
                        ref = yd1[kb][oinv1[t0 * KU:(t0 + n) * KU]]
                        oky &= same(mine, ref)
                    t0 += n
                check(okd and oky, f'g2b L{il} card {c}: split {split} == one 2048 chunk: every (token, slot) down output and down-input block, bitwise')
            del L


def stage_speed(torch, rd, layers, js, reps):
    dev = torch.device('cuda:0'); P = Paths(torch, dev, 2048); res = {}; longest = [0.0, '']
    def tm(fn):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize(); return a.elapsed_time(b)
    for il in layers:
        for c in (0, 1):
            L = Layer(torch, dev, rd, il, c)
            for n in (512, 1024, 2048):
                xn, ids = load_real('d8192', il, 0, n); x = torch.from_numpy(xn).to(dev); P.set_ids(ids)
                variants = [('mmq', None)] + [('own', (jg, jd)) for jg in js for jd in js if (L.fg, jg) in P.O.kgu]
                fns = {v: (lambda: P.ref(L, x, n, ids, groups=False)) if v[0] == 'mmq' else
                          (lambda jj=v[1]: P.own(L, x, n, ids, jj[0], jj[1], groups=False)) for v in variants}
                for f in fns.values(): f(); f()
                t = {v: [] for v in variants}
                for _ in range(reps):
                    for v in variants: t[v].append(tm(fns[v]))
                med = {v: float(np.median(t[v])) for v in variants}
                best = min((v for v in variants if v[0] == 'own'), key=lambda v: med[v])
                key = f'L{il} card{c} {L.fg}/{L.fd} n{n}'
                res[key] = dict(mmq_ms=med[('mmq', None)], own_ms={f'{v[1][0]}/{v[1][1]}': med[v] for v in variants if v[0] == 'own'},
                                best=f'{best[1][0]}/{best[1][1]}', best_ms=med[best])
                print(f'SPEED {key}: MMQ path {med[("mmq", None)]:.3f} ms, ours best J {best[1][0]}/{best[1][1]} {med[best]:.3f} ms '
                      f'({med[("mmq", None)] / med[best]:.2f}x); J 48/48 {med.get(("own", (48, 48)), float("nan")):.3f}', flush=True)
                # the per-kernel split of ours at J 48 / 48 and the longest launch
                if ('own', (48, 48)) in med:
                    N = n * KU; O, M = P.O, P.M
                    parts = dict(group=lambda: O.group(P.ids[:N], n, KU, NEXP, P.ob, P.oi, P.oinv, P.ta, 48, P.tb, 48, P.nt, P.err),
                                 quant=lambda: M.quantize_scatter(x, D, n, P.oinv, KU, P.y, M.layout(L.fg)),
                                 gu=lambda: O.gu(L.fg, L.wg, L.wu, NEXP, L.ffc, D, P.y, N, P.ta, P.nt[0:1], P.yd, P.err, 48, L.fd),
                                 down=lambda: O.mm(L.fd, L.wd, NEXP, D, L.ffc, P.yd, N, P.oi, P.tb, P.nt[1:2], P.od, P.err, 48))
                    pt = {}
                    for nm, f in parts.items():
                        ts = [tm(f) for _ in range(max(3, reps))]; pt[nm] = float(np.median(ts))
                        if max(ts) > longest[0]: longest[:] = [max(ts), f'{key} {nm}']
                    res[key]['own_parts_ms'] = pt
                    print(f'       ours J 48/48 parts: ' + ', '.join(f'{k} {v:.3f}' for k, v in pt.items()), flush=True)
            del L
    NUMBERS['speed'] = res; NUMBERS['longest_launch_ms'] = longest
    print(f'longest single launch timed: {longest[0]:.3f} ms ({longest[1]})', flush=True)


def main():
    global REAL
    ap = argparse.ArgumentParser()
    ap.add_argument('coder'); ap.add_argument('stages', nargs='+')
    ap.add_argument('--real', default=REAL); ap.add_argument('--json', default=None)
    ap.add_argument('--layers', default='0,1,2,12,17,21,47'); ap.add_argument('--js', default='16,32,48,64,96,128')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--small', action='store_true', help='edge: only the 1- and 3-token cases (a changed kernel\'s first run)')
    a = ap.parse_args()
    REAL = a.real
    layers = [int(v) for v in a.layers.split(',')]; js = [int(v) for v in a.js.split(',')]
    for s in a.stages:
        if s == 'cpu': stage_cpu(); continue
        import torch
        from caminus import flashnext   # noqa: F401
        from gguf import GGUFReader
        rd = GGUFReader(a.coder)
        print('device', torch.cuda.get_device_name(0), 'compaction_proactiveness', open('/proc/sys/vm/compaction_proactiveness').read().strip(), flush=True)
        if s == 'g1': stage_g1(torch, rd, layers, js)
        elif s == 'edge': stage_edge(torch, rd, layers, a.small)
        elif s == 'g2b': stage_g2b(torch, rd, layers)
        elif s == 'speed': stage_speed(torch, rd, layers, js, a.reps)
        else: raise SystemExit(f'unknown stage {s}')
    if a.json: json.dump(dict(fails=FAILS, numbers=NUMBERS), open(a.json, 'w'), indent=1)
    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'), flush=True)
    sys.exit(0 if not FAILS else 1)


if __name__ == '__main__':
    main()
