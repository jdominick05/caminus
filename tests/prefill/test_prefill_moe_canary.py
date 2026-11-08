"""Canary checks of the batched prefill's own MoE path (caminus.prefill.moe, kernels/prefill_moe.hip), investigation run 1 after the
11-04 09:24 SDMA page fault on the XTX (docs/AGENTIC-PLAN.md, "MoE grouped GEMM ... halted on a GPU fault"). One card:
cuda:0, which must be the XT (HIP_VISIBLE_DEVICES=1; the GPU stages refuse an XTX).

  python -I tests/prefill/test_prefill_moe_canary.py CODER STAGE... [--layers L,..] [--cards 0,1] [--routings r,..]
                                                     [--pieces all|tiny] [--js all|prod] [--rounds N] [--json OUT]

Stages:
  review  CPU only, nothing launched: for every piece the stack runs (prefill_tp.micro_bounds, m 1-4, of the chunks 512, 1024,
          2048, 777 + 1271, 1000 + 1048, 1 + 2047 and 512 + 788 + 748), both cards' shapes and every J: the byte extent every
          store of ChunkStack._ffn's own-path sequence may reach (the kernels' index expressions; the tile lists at the
          worst routing), against the view ChunkStack passes (mmq.Workspace's layout at chunk 2048, the piece's o_down
          rows) and the host check's minimum; the reads past a tile's columns (moe_load_y) and the weights' over-read
          beside them; the workspace views in layout order and disjoint, and this script's copy of the layout (carve) ==
          mmq.Workspace's views
  canary  ChunkStack._ffn's own-path sequence: group (k3_mmq_ids_clamp + k3_moe_group), quantize_scatter, gu (k3_moe_gu),
          mm (k3_moe_mm) on a copy of mmq.Workspace's layout (chunk 2048, the card's dims) carved out of ONE allocation
          with 1 MiB sentinel guards before and after it; the router ids and the `mixed` rows (GlueWorkspace stand-ins, the
          piece's rows at its offset r0 as ChunkStack's _v views) and each weight shard ([weights | MMQ's 512-byte zero
          tail]) in guarded allocations too. The block, ids and mixed start as the sentinel for every configuration (any
          read of a byte no launch wrote shows). Before every launch they are snapshotted; after it: the guards hold the
          sentinel, every byte outside the launch's intended output range is unchanged (the ranges from an independent
          numpy model of the routing: bounds, ids_dst / ids_inv [0, N), the tile entries [0, ntiles), y_moe 20 x N
          blocks, y_down ffc / 128 x N blocks, o_down the piece's rows), the weights equal their load-time copy, the flag
          word stays 0, and the outputs equal the numpy model and the MMQ path (its own buffers) bitwise. A launch runs only
          after every check of the one before it passed; the stage stops at the first write outside its range. Routings:
          the real routed inputs ($ARTIFACTS/impl-moe-gemm/real: d8192's `mixed` rows and router ids, each piece at its
          absolute position) and synthetic ones on the same rows: one expert in every token (hot), every token on 10
          experts (ten: 246 empty), Zipf, uniform, near-maximum tile counts for J 16 / 32 / 64 (counts = 1 mod J). J: the
          production pair (moe.J_GU, moe.j_down(n)) and the diagonal (J, J) over every compiled J (gu where it has a
          kernel, else J_GU). Each launch timed (events); the longest per kernel reported
  d2h     the D2H pattern the fault surfaced in (tests/prefill/test_prefill_tp.py stack: prompt()'s and after_chunk()'s .cpu()
          copies, then states(): each state tensor copied with .cpu() into pageable host memory and hashed with that
          file's own sha()), both cards' tensor sizes on the one device, after own-path launches (no checks in between) at
          the m 3 / m 4 pieces of a 2048-token chunk and 777 + 1271 at m 1, --rounds times: every round's hashes == round
          0's; the host destinations' address range logged per round
Prints PASS / FAIL per check and RESULT PASS / FAIL; exit 0 = PASS."""
import argparse, json, os, sys, time, types, zlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'python')); sys.path.insert(0, HERE)
import test_prefill_moe as tpm   # noqa: E402  (its Layer slicing, the MMQ reference path, load_real)

FAILS, NUMBERS = [], {}
NEXP, KU, D = tpm.NEXP, tpm.KU, tpm.D
T_CHUNK = 2048
SPLIT = (10, 12)                 # (gdn_keys, moe_blocks): FlashNextTP's split; --tp-split GK,MB (the even split 8,10)
SEQS = ((512,), (1024,), (2048,), (777, 1271), (1000, 1048), (1, 2047), (512, 788, 748))
SENT = -5913691                  # the sentinel word 0xFFA5C3A5: a NaN as fp32, a negative id as int32
GUARD = 1 << 20                  # bytes of sentinel before and after every guarded allocation
WTAIL = 512                      # the weights' zeroed tail (as tests/test_prefill_moe.Layer / FlashNextTP.ksplit_pad)
ROUTINGS = ('real', 'hot', 'ten', 'zipf', 'uniform', 'max16', 'max32', 'max64')


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)
    return ok


def meminfo():
    m = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo')}
    return dict(MemAvailable_MiB=m['MemAvailable'] // 1024, compaction_proactiveness=open('/proc/sys/vm/compaction_proactiveness').read().strip())


def pieces():
    """(absolute position, offset in its chunk, tokens) of every micro-chunk piece the stack runs (SEQS, m 1-4), deduplicated"""
    from caminus.prefill.tp import micro_bounds
    out = []
    for s in SEQS:
        p0 = 0
        for T in s:
            for m in (1, 2, 3, 4):
                for r0, r1 in micro_bounds(T, m):
                    if (p0 + r0, r0, r1 - r0) not in out: out.append((p0 + r0, r0, r1 - r0))
            p0 += T
    return out


def carve(torch, block, layout):
    """mmq.Workspace's views over `block` (its __init__'s loop: the same names, offsets, dtypes, err[:1])"""
    W = types.SimpleNamespace(off={}, layout=dict(layout))
    off = 0
    for n, b in layout.items():
        v = block[off:off + b]
        if n.startswith('ids') or n.startswith('moe_') or n in ('bounds', 'err'): v = v.view(torch.int32)
        elif n.startswith('o_') or n.startswith('x_'): v = v.view(torch.float32)
        setattr(W, n, v); W.off[n] = off; off += b
    W.err = W.err[:1]
    W.nbytes = off
    return W


def carve_equals_workspace(torch, mmq, d, T=16):
    """the carve's views == mmq.Workspace's (offsets from the block, element counts, dtypes) on a small CPU workspace"""
    ws = mmq.Workspace(torch.device('cpu'), T, d, check=False)
    blk = torch.zeros(ws.nbytes, dtype=torch.uint8)
    W = carve(torch, blk, mmq.workspace_layout(T, d))
    bad = []
    for n in ws.layout:
        a, b = getattr(ws, n), getattr(W, n)
        if (a.data_ptr() - ws.block.data_ptr(), a.numel(), a.dtype) != (b.data_ptr() - blk.data_ptr(), b.numel(), b.dtype): bad.append(n)
    return not bad and W.nbytes == ws.nbytes, bad


# ---- the numpy model of a routing ----------------------------------------------------------------------------------------
def model(ids, js):
    """ids [n, k] (valid, distinct per token) -> mm_ids_helper's tables and the tile lists per J, independently of the kernels"""
    n, k = ids.shape; N = n * k; flat = ids.reshape(-1).astype(np.int64)
    cnt = np.bincount(flat, minlength=NEXP)
    bounds = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int32)
    order = np.lexsort((np.arange(N), flat))           # experts ascending, (t * k + slot) ascending within an expert
    ids_dst = order.astype(np.int32); ids_inv = np.empty(N, np.int32); ids_inv[order] = np.arange(N, dtype=np.int32)
    tiles = {}
    for J in set(js):
        tl = [(e, int(bounds[e]) + j * J, min(J, int(cnt[e]) - j * J), 0) for e in range(NEXP) for j in range(-(-int(cnt[e]) // J))]
        tiles[J] = np.array(tl, dtype=np.int32).reshape(-1, 4)
    return dict(cnt=cnt, bounds=bounds, ids_dst=ids_dst, ids_inv=ids_inv, tiles=tiles)


def worst_tiles(N, n, J):
    """the near-maximum tile count of N compact columns over NEXP experts with at most n per expert (counts = 1 mod J):
    the counts of the max<J> routing"""
    c = np.zeros(NEXP, np.int64)
    if N <= NEXP: c[:N] = 1
    else:
        c[:] = 1; rem = N - NEXP
        while rem >= J:
            prog = False
            for e in range(NEXP):
                if rem < J: break
                if c[e] + J <= n: c[e] += J; rem -= J; prog = True
            if not prog: break
        while rem > 0:
            for e in range(NEXP):
                if rem == 0: break
                if c[e] < n: c[e] += 1; rem -= 1
    assert c.sum() == N and c.max() <= n, (N, n, J)
    return c


def routing(kind, n, rng, real_ids=None):
    """[n, KU] int32 expert ids, distinct per token, every id in [0, NEXP)"""
    k = KU
    if kind == 'real': ids = np.asarray(real_ids, dtype=np.int32)
    elif kind == 'hot':
        others = np.array([e for e in range(NEXP) if e != 5])
        ids = np.stack([np.insert(rng.permutation(others)[:k - 1], 3, 5) for _ in range(n)]).astype(np.int32)
    elif kind == 'ten': ids = np.tile(np.arange(k, dtype=np.int32), (n, 1))
    elif kind == 'zipf':
        perm = rng.permutation(NEXP); p = 1.0 / (np.arange(NEXP) + 1.0) ** 1.2; p /= p.sum()
        ids = np.stack([perm[rng.choice(NEXP, k, replace=False, p=p)] for _ in range(n)]).astype(np.int32)
    elif kind == 'uniform': ids = np.argsort(rng.random((n, NEXP)), axis=1)[:, :k].astype(np.int32)
    elif kind.startswith('max'):
        J = int(kind[3:]); c = worst_tiles(n * k, n, J)
        perm = rng.permutation(NEXP); lst = np.repeat(perm, c[perm])
        ids = lst.reshape(k, n).T.copy()                 # position p -> (token p % n, slot p // n): runs of <= n land on distinct tokens
        ids = ids[rng.permutation(n)]
        ids = np.take_along_axis(ids, np.argsort(rng.random((n, k)), axis=1), axis=1).astype(np.int32)
    else: raise ValueError(kind)
    assert ids.shape == (n, k) and ids.min() >= 0 and ids.max() < NEXP and all(len(set(r)) == k for r in ids.tolist()), kind
    return np.ascontiguousarray(ids)


# ---- stage review (CPU) --------------------------------------------------------------------------------------------------
def stage_review(rd):
    import torch
    from caminus.device import capacity
    from caminus.prefill import mmq, moe
    g = capacity.geometry(rd)
    E, k = g['n_exp'], g['k_used']
    assert (E, k, g['D']) == (NEXP, KU, D), (E, k, g['D'])
    PC = pieces(); sizes = sorted({n for _, _, n in PC})
    print(f'review: {len(PC)} pieces (absolute position, offset, tokens), {len(sizes)} sizes: {sizes}', flush=True)
    worst = {}
    for c in (0, 1):
        d = mmq.card_dims(g, c, *SPLIT); ffc = d['ffc']; lay = mmq.workspace_layout(T_CHUNK, d)
        ok, bad = carve_equals_workspace(torch, mmq, d)
        check(ok, f'review card {c}: this script\'s carve == mmq.Workspace\'s views (names, offsets, counts, dtypes){" " + str(bad) if bad else ""}')
        offs, off = {}, 0
        for nme, b in lay.items(): offs[nme] = (off, off + b); off += b
        names = list(lay)
        check(names[-3:] == ['moe_ta', 'moe_tb', 'moe_nt'] and all(offs[a][1] == offs[b][0] for a, b in zip(names, names[1:])),
              f'review card {c}: the layout\'s views are consecutive and disjoint ({off} B = {off / 2**20:.1f} MiB); moe_ta / moe_tb / moe_nt '
              f'are its last three (an overrun past them leaves the block)')
        view = {nme: lay[nme] for nme in lay}; nf0 = len(FAILS)
        for J in moe.JS:
            for _, r0, n in PC:
                N = n * k; mt = moe.max_tiles(N, E, J)
                wt = int(np.ceil(worst_tiles(N, n, J) / J).sum())        # the max<J> routing's tile count
                worst[(c, J, n)] = (wt, mt)
                odv = 4 * N * D                                           # ChunkStack's od: _v(W.o_down, r0, n, k * D)
                ext = {   # store extents (bytes from the view's start), the view it lands in (bytes), the host check's minimum
                    'group bounds[e], e <= n_exp': (4 * (E + 1), view['bounds'], 4 * (E + 1)),
                    'group ids_dst[pos], 0 <= pos < N (device-checked)': (4 * N, view['ids_dst'], 4 * N),
                    'group ids_inv[t*k+s], t < n, s < k': (4 * N, view['ids_inv'], 4 * N),
                    'group tiles_a[o], o < max_a (cut there)': (16 * mt, view['moe_ta'], 16 * mt),
                    'group tiles_b[o], o < max_b (cut there)': (16 * mt, view['moe_tb'], 16 * mt),
                    'group ntiles[0..1]': (8, view['moe_nt'], 8),
                    'quantize_scatter y[kb*N + ids_inv], kb < 20, ids_inv < N': (20 * N * 144, view['y_moe'], mmq.y_bytes(D, N)),
                    'gu yd[(it*I+i4)/128 * N + c0 + j], j < nc, c0 + nc <= N, it*I + I <= ffc': ((ffc // 128) * N * 144, view['y_down'], mmq.y_bytes(ffc, N)),
                    'mm dst[ids_sh[j]*D + it*I + i], ids_sh < N (device-checked), it*I + I <= D': (4 * N * D, odv, 4 * D * N),
                }
                for what, (e_, vb, hmin) in ext.items():
                    if not e_ <= hmin <= vb:
                        check(False, f'review card {c} J {J} piece r0 {r0} n {n}: {what}: extent {e_} B, host minimum {hmin}, view {vb}')
                if not 4 * (r0 + n) * k * D <= view['o_down']:
                    check(False, f'review card {c} piece r0 {r0} n {n}: the od view ends past o_down')
                # the tile counts: the worst routing's within the bound the host sizes the lists to
                if not (wt <= mt and mt * 16 <= view['moe_ta'] and mt * 16 <= view['moe_tb']):
                    check(False, f'review card {c} J {J} n {n}: tile count {wt} vs the bound {mt} (views {view["moe_ta"] // 16} entries)')
                # reads (not stores): gu's moe_load_y J columns from c0 in the last block row (into Y_TAIL); mm's in y_down
                rg = (20 * N + J - 1) * 144; rm = ((-(-ffc // 128)) * N + J - 1) * 144
                if not (rg <= mmq.y_bytes(D, N) <= view['y_moe'] and rm <= mmq.y_bytes(ffc, N) <= view['y_down']):
                    check(False, f'review card {c} J {J} n {n}: a read past its view (y_moe {rg} vs {mmq.y_bytes(D, N)}, y_down {rm} vs {mmq.y_bytes(ffc, N)})')
        over = mmq.overread_bytes('IQ4_NL', ffc), mmq.overread_bytes('Q2_0', ffc)
        check(len(FAILS) == nf0, f'review card {c} (ffc {ffc}): every store extent <= the host check\'s minimum <= the view ChunkStack passes, at every J '
              f'{moe.JS} and all {len(PC)} pieces; tile counts at the worst routing <= max_tiles; moe_load_y\'s reads past nc stay in Y_TAIL; '
              f'the weights\' over-read (IQ4_NL / Q2_0 down, K {ffc}) {over[0]} / {over[1]} B into the 512-byte zero tail')
    NUMBERS['review_worst_tiles'] = {f'card{c} J{J} n{n}': v for (c, J, n), v in worst.items() if n in (1, 9, 512, 2048)}


# ---- guarded allocations -------------------------------------------------------------------------------------------------
class Guarded:
    """[GUARD sentinel | body (nbytes) | GUARD sentinel], one allocation; snap / d: its snapshot and diff buffers"""
    def __init__(self, torch, dev, nbytes, name, snap=True):
        assert nbytes % 4 == 0 and GUARD % 256 == 0, (name, nbytes)
        self.torch, self.name, self.nb, self.g = torch, name, int(nbytes), GUARD
        self.a = torch.empty(GUARD + self.nb + GUARD, dtype=torch.uint8, device=dev)
        self.a32 = self.a.view(torch.int32); self.a32.fill_(SENT)
        self.body = self.a[GUARD:GUARD + self.nb]
        self.snap = torch.empty_like(self.a) if snap else None
        self.d = torch.empty(self.a32.numel(), dtype=torch.bool, device=dev) if snap else None

    def fill(self):
        self.a32.fill_(SENT)

    def take(self):
        self.snap.copy_(self.a)

    def outside(self, allowed):
        """(int32 words changed since take() outside the allowed byte ranges [lo, hi) of the body, their first offsets)"""
        torch = self.torch
        torch.ne(self.a32, self.snap.view(torch.int32), out=self.d)
        for lo, hi in allowed:
            assert lo % 4 == 0 and hi % 4 == 0 and 0 <= lo <= hi <= self.nb, (self.name, lo, hi)
            self.d[(GUARD + lo) // 4:(GUARD + hi) // 4] = False
        n = int(self.d.sum().item())
        where = [] if n == 0 else [int(v) * 4 - GUARD for v in self.d.nonzero()[:8].view(-1).tolist()]
        return n, where

    def guards_ok(self):
        torch = self.torch
        return bool((self.a32[:GUARD // 4] == SENT).all().item()) and bool((self.a32[(GUARD + self.nb) // 4:] == SENT).all().item())


def where_in(W, offs):
    """block offsets -> 'view+offset'"""
    out = []
    for o in offs:
        if o < 0: out.append(f'guard-before{o}'); continue
        nm = [n for n, (a, b) in W.spans.items() if a <= o < b]
        out.append(f'{nm[0]}+{o - W.spans[nm[0]][0]}' if nm else f'guard-after+{o - W.nbytes}')
    return out


class GLayer:
    """a layer's card-c MoE shards (tests/test_prefill_moe.Layer's slicing) re-homed into guarded allocations
    [GUARD | weights | WTAIL zeros | GUARD]; the load-time snapshot is the reference for 'unchanged'"""
    def __init__(self, torch, dev, rd, il, c):
        L = tpm.Layer(torch, dev, rd, il, c)
        self.il, self.c, self.ffc, self.fg, self.fu, self.fd = il, c, L.ffc, L.fg, L.fu, L.fd
        self.regions = []
        for nm in ('wg', 'wu', 'wd'):
            w = getattr(L, nm); nb = w.numel()
            G = Guarded(torch, dev, nb + WTAIL, f'{nm} L{il} c{c}', snap=False)
            G.body[:nb].copy_(w); G.body[nb:].zero_()
            G.snap = G.a.clone()
            setattr(self, nm, G.body[:nb]); self.regions.append(G)
        del L
        torch.cuda.synchronize(dev)

    def unchanged(self):
        return all(self.torch_equal(G) for G in self.regions)

    @staticmethod
    def torch_equal(G):
        return bool(G.torch.equal(G.a, G.snap))


# ---- stage canary (GPU, the XT) ------------------------------------------------------------------------------------------
def stage_canary(torch, rd, layers, cards, rkinds, piece_set, js_mode):
    from caminus.device import capacity
    from caminus.prefill import mmq, moe
    dev = torch.device('cuda:0')
    g = capacity.geometry(rd); E, k = NEXP, KU
    P = tpm.Paths(torch, dev, T_CHUNK)                     # the MMQ reference path's own buffers (outside every guarded region)
    O, M = P.O, P.M
    cfg = {f'{f} J{J}': dict(v, gu=(f, J) in O.kgu) for (f, J), v in O.cfg.items()}
    NUMBERS['configs'] = cfg
    PC = pieces()
    if piece_set == 'tiny': PC = [p for p in PC if p[2] in (1, 9, 20)] or PC[:1]
    gids = Guarded(torch, dev, 4 * T_CHUNK * k, 'ids (GlueWorkspace.ids)')
    gmix = Guarded(torch, dev, 4 * T_CHUNK * D, 'mixed (GlueWorkspace.mixed)')
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    longest = {}
    totals = dict(configs=0, launches=0, refs=0)
    tile_max = {}
    abort = [False]

    def timed(name, fn, tag):
        ev[0].record(); fn(); ev[1].record(); ev[1].synchronize()
        ms = ev[0].elapsed_time(ev[1])
        if ms > longest.get(name, (0.0, ''))[0]: longest[name] = (ms, tag)
        totals['launches'] += 1

    for c in cards:
        d = mmq.card_dims(g, c, *SPLIT); lay = mmq.workspace_layout(T_CHUNK, d)
        ok, bad = carve_equals_workspace(torch, mmq, d)
        if not check(ok, f'canary card {c}: the carve == mmq.Workspace\'s views {bad if bad else ""}'): return
        gws = Guarded(torch, dev, sum(lay.values()), f'workspace card {c}')
        W = carve(torch, gws.body, lay)
        W.spans = {n: (W.off[n], W.off[n] + b) for n, b in lay.items()}
        print(f'canary card {c}: workspace block {W.nbytes / 2**20:.1f} MiB (chunk {T_CHUNK}, ffc {d["ffc"]}) between {GUARD >> 20} MiB guards; '
              f'{meminfo()}', flush=True)
        for il in layers:
            t0 = time.time(); L = GLayer(torch, dev, rd, il, c); ffc = L.ffc
            assert ffc == d['ffc'], (ffc, d['ffc'])
            xn, idsr = tpm.load_real('d8192', il, 0, T_CHUNK)
            xdev = torch.from_numpy(xn).to(dev)
            tagL = f'L{il} card {c} ({L.fg} / {L.fd}, ffc {ffc})'
            print(f'  {tagL}: loaded in {time.time() - t0:.1f} s', flush=True)
            nconf0 = totals['configs']; t0 = time.time()
            seen_synth = set()
            for p, r0, n in PC:
                for rk in rkinds:
                    if rk != 'real':
                        if (rk, r0, n) in seen_synth: continue
                        seen_synth.add((rk, r0, n))
                    rng = np.random.default_rng(zlib.crc32(f'{rk} {n} {r0} {il} {c}'.encode()))
                    ids = routing(rk, n, rng, idsr[p:p + n] if rk == 'real' else None)
                    N = n * k
                    combos = [(moe.J_GU, moe.j_down(n))]
                    if js_mode == 'all':
                        for J in moe.JS:
                            cb = (J if (L.fg, J) in O.kgu else moe.J_GU, J)
                            if cb not in combos: combos.append(cb)
                    mdl = model(ids, [j for cb in combos for j in cb])
                    for J, tl in mdl['tiles'].items():
                        key = f'card{c} J{J}'; bound = moe.max_tiles(N, E, J)
                        if len(tl) > bound: check(False, f'{tagL} {rk} n {n}: the model\'s tile count {len(tl)} > max_tiles {bound}')
                        if len(tl) > tile_max.get(key, (0, 0, 0))[0]: tile_max[key] = (len(tl), bound, n)
                    x = xdev[p:p + n]
                    P.err.zero_(); P.ref(L, x, n, ids); torch.cuda.synchronize(dev); totals['refs'] += 1
                    if int(P.err.item()) != 0:
                        check(False, f'{tagL} {rk} p {p} n {n}: the MMQ reference\'s flag word {int(P.err.item()):#x}'); continue
                    dv = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
                    m_b, m_dst, m_inv = dv(mdl['bounds']), dv(mdl['ids_dst']), dv(mdl['ids_inv'])
                    m_tl = {J: dv(t) for J, t in mdl['tiles'].items()}
                    idsd = dv(ids.reshape(-1))
                    for jg, jd in combos:
                        tag = f'{tagL} {rk} p {p} r0 {r0} n {n} J {jg}/{jd}'
                        totals['configs'] += 1
                        gws.fill(); gids.fill(); gmix.fill()
                        idv = gids.body.view(torch.int32)[r0 * k:(r0 + n) * k]; idv.copy_(idsd)
                        mxv = gmix.body.view(torch.float32)[r0 * D:(r0 + n) * D]; mxv.copy_(x.reshape(-1))
                        W.err.zero_()
                        nta, ntb = len(mdl['tiles'][jg]), len(mdl['tiles'][jd])
                        od = W.o_down.view(-1)[r0 * k * D:(r0 + n) * k * D]
                        ob = W.off['o_down'] + 4 * r0 * k * D
                        steps = [
                            ('group', lambda: O.group(idv, n, k, E, W.bounds, W.ids_dst, W.ids_inv, W.moe_ta, jg, W.moe_tb, jd, W.moe_nt, W.err),
                             [(W.off['bounds'], W.off['bounds'] + 4 * (E + 1)), (W.off['ids_dst'], W.off['ids_dst'] + 4 * N),
                              (W.off['ids_inv'], W.off['ids_inv'] + 4 * N), (W.off['moe_ta'], W.off['moe_ta'] + 16 * nta),
                              (W.off['moe_tb'], W.off['moe_tb'] + 16 * ntb), (W.off['moe_nt'], W.off['moe_nt'] + 8)],
                             lambda: (torch.equal(W.bounds[:E + 1], m_b) and torch.equal(W.ids_dst[:N], m_dst) and torch.equal(W.ids_inv[:N], m_inv)
                                      and torch.equal(W.moe_ta[:4 * nta], m_tl[jg].view(-1)) and torch.equal(W.moe_tb[:4 * ntb], m_tl[jd].view(-1))
                                      and [int(v) for v in W.moe_nt[:2].tolist()] == [nta, ntb]
                                      and torch.equal(W.bounds[:E + 1], P.bd) and torch.equal(W.ids_dst[:N], P.sd[:N]) and torch.equal(W.ids_inv[:N], P.s1[:N]))),
                            ('quantize_scatter', lambda: M.quantize_scatter(mxv, D, n, W.ids_inv, k, W.y_moe, M.layout(L.fg)),
                             [(W.off['y_moe'], W.off['y_moe'] + 20 * N * 144)],
                             lambda: torch.equal(W.y_moe[:20 * N * 144], P.y[:20 * N * 144])),
                            ('gu', lambda: O.gu(L.fg, L.wg, L.wu, E, ffc, D, W.y_moe, N, W.moe_ta, W.moe_nt[0:1], W.y_down, W.err, jg, L.fd),
                             [(W.off['y_down'], W.off['y_down'] + (ffc // 128) * N * 144)],
                             lambda: torch.equal(W.y_down[:(ffc // 128) * N * 144], P.yd_ref[:(ffc // 128) * N * 144])),
                            ('mm', lambda: O.mm(L.fd, L.wd, E, D, ffc, W.y_down, N, W.ids_dst, W.moe_tb, W.moe_nt[1:2], od, W.err, jd),
                             [(ob, ob + 4 * N * D)],
                             lambda: torch.equal(od, P.od_ref[:N * D])),
                        ]
                        for name, fn, allowed, content in steps:
                            for G in (gws, gids, gmix): G.take()
                            timed(name if name != 'gu' and name != 'mm' else f'{name} {L.fg if name == "gu" else L.fd} J{jg if name == "gu" else jd}', fn, tag)
                            nb_, wb_ = gws.outside(allowed)
                            ni_, wi_ = gids.outside([]); nx_, wx_ = gmix.outside([])
                            err = int(W.err.item()); wok = L.unchanged()
                            cok = bool(content())
                            if nb_ or ni_ or nx_ or not wok:
                                check(False, f'{tag} {name}: WRITE OUTSIDE ITS RANGE: workspace {nb_} words at {where_in(W, wb_)}, ids {ni_} at {wi_}, '
                                             f'mixed {nx_} at {wx_}, weights {"unchanged" if wok else "CHANGED"}')
                                abort[0] = True
                            if err: check(False, f'{tag} {name}: the routing flag word {err:#x} (expected 0)')
                            if not cok: check(False, f'{tag} {name}: the output != the model / the MMQ path, bitwise')
                            if abort[0] or err or not cok: break
                        if abort[0]: break
                        if not (gws.guards_ok() and gids.guards_ok() and gmix.guards_ok()):
                            check(False, f'{tag}: a guard region lost its sentinel'); abort[0] = True; break
                    if abort[0]: break
                if abort[0]: break
            ok_l = not abort[0] and L.unchanged() and all(G.torch.equal(G.a[:GUARD], G.snap[:GUARD]) and G.torch.equal(G.a[-GUARD:], G.snap[-GUARD:]) for G in L.regions)
            check(ok_l, f'canary {tagL}: {totals["configs"] - nconf0} configurations ({len(PC)} pieces x routings {",".join(rkinds)} x J pairs), '
                        f'4 launches each checked (guards, every byte outside its range unchanged, weights unchanged, flag word 0, outputs '
                        f'== model / MMQ bitwise) in {time.time() - t0:.0f} s')
            del L, xdev; torch.cuda.synchronize(dev)
            if abort[0]: break
        del gws, W; torch.cuda.synchronize(dev)
        if abort[0]: break
    NUMBERS['canary'] = dict(totals, longest_launch_ms={kk: v for kk, v in sorted(longest.items())}, tile_max=tile_max, mem=meminfo())
    lm = max(longest.items(), key=lambda kv: kv[1][0]) if longest else ('', (0.0, ''))
    print(f'canary: {totals["configs"]} configurations, {totals["launches"]} launches checked, {totals["refs"]} MMQ references; '
          f'longest launch {lm[1][0]:.3f} ms ({lm[0]}: {lm[1][1]}); {meminfo()}', flush=True)
    for kk, v in sorted(longest.items()): print(f'  longest {kk}: {v[0]:.3f} ms ({v[1]})', flush=True)
    for kk, v in sorted(tile_max.items()): print(f'  largest tile list {kk}: {v[0]} entries (bound {v[1]}, n {v[2]})', flush=True)
    check(lm[1][0] < 50.0, f'canary: every launch under 50 ms (longest {lm[1][0]:.3f} ms)')


# ---- stage d2h (GPU, the XT) ---------------------------------------------------------------------------------------------
def stage_d2h(torch, rd, coder, rounds, layers):
    from caminus.device import capacity
    from caminus.prefill import mmq, moe
    from caminus.engine import k3
    from caminus.prefill.tp import micro_bounds
    import test_prefill_tp as ttp   # its sha() (the stack's D2H expression), unchanged
    dev = torch.device('cuda:0')
    g = capacity.geometry(rd); E, k = NEXP, KU
    f = {x.name: x for x in rd.fields.values()}
    nl = int(f['qwen4exp.block_count'].contents()); fai = int(f['qwen4exp.full_attention_interval'].contents())
    ple_layers = sorted(int(v) for v in f['qwen4exp.ple.layers'].contents())
    HC, S = g['hc'], g['ssm_state']; nvk = g['ssm_heads'] // g['ssm_groups']
    kc = [SPLIT[0], g['ssm_groups'] - SPLIT[0]]; H = [nvk * x for x in kc]; CH = [(2 + nvk) * S * x for x in kc]
    M = mmq.MMQ(dev, formats=moe.FORMATS); O = moe.MoE(dev, M)
    Ws, Ls = [], []
    for c in (0, 1):
        d = mmq.card_dims(g, c, *SPLIT); lay = mmq.workspace_layout(T_CHUNK, d)
        blk = torch.zeros(sum(lay.values()), dtype=torch.uint8, device=dev); W = carve(torch, blk, lay); W.blk = blk; Ws.append(W)
        for il in layers: Ls.append((c, GLayer(torch, dev, rd, il, c)))
    xr = {il: torch.from_numpy(tpm.load_real('d8192', il, 0, T_CHUNK)[0]).to(dev) for il in layers}
    ir = {il: tpm.load_real('d8192', il, 0, T_CHUNK)[1] for il in layers}
    ids_dev = torch.zeros(T_CHUNK * k, dtype=torch.int32, device=dev)
    # the prompts the fault followed: 2048 at m 3 and m 4 (overlap / serial: the same launches on one card), 777 + 1271 at m 1
    prompts = [('2048 m3', [(0, T_CHUNK, 3)]), ('2048 m4', [(0, T_CHUNK, 4)]), ('777+1271 m1', [(0, 777, 1), (777, T_CHUNK, 1)])]
    # the stack's tensors, both cards' sizes, on this one device (FlashNextTP.state key order: per layer, card)
    state = {}
    for il in range(nl):
        for c in (0, 1):
            if il in ple_layers: state[(c, il, 'ple')] = torch.zeros(HC * D, 9, device=dev)
            if il % fai != fai - 1:
                state[(c, il, 'conv')] = torch.zeros(CH[c], 3, device=dev)
                state[(c, il, 'gdn')] = torch.zeros(k3.gdn_block_size(H[c], S, kc[c], 4), device=dev)
    resid = [torch.zeros(T_CHUNK * HC * D, device=dev) for _ in (0, 1)]
    hist = dict(ids=[[torch.zeros(T_CHUNK, k, dtype=torch.int32, device=dev) for _ in range(nl)] for _ in (0, 1)],
                wts=[[torch.zeros(T_CHUNK, k, device=dev) for _ in range(nl)] for _ in (0, 1)])
    sizes = sorted({(kk[2], t[:H[kk[0]] * S * S].numel() * 4 if kk[2] == 'gdn' else t.numel() * 4) for kk, t in state.items()})
    print(f'd2h: {len(state)} state tensors (copied sizes {sizes} B), residual {resid[0].numel() * 4} B x 2, hist [{T_CHUNK}, {k}] x {nl} x 2 x 2; '
          f'ple layers {len(ple_layers)}; {meminfo()}', flush=True)

    def own(c, L, x, ids_np, r0, n):
        W = Ws[c]; N = n * k
        idv = ids_dev[r0 * k:(r0 + n) * k]; idv.copy_(torch.from_numpy(np.ascontiguousarray(ids_np).reshape(-1)).to(dev))
        od = W.o_down.view(-1)[r0 * k * D:(r0 + n) * k * D]
        jg, jd = moe.J_GU, moe.j_down(n)
        O.group(idv, n, k, E, W.bounds, W.ids_dst, W.ids_inv, W.moe_ta, jg, W.moe_tb, jd, W.moe_nt, W.err)
        M.quantize_scatter(x, D, n, W.ids_inv, k, W.y_moe, M.layout(L.fg))
        O.gu(L.fg, L.wg, L.wu, E, L.ffc, D, W.y_moe, N, W.moe_ta, W.moe_nt[0:1], W.y_down, W.err, jg, L.fd)
        O.mm(L.fd, L.wd, E, D, L.ffc, W.y_down, N, W.ids_dst, W.moe_tb, W.moe_nt[1:2], od, W.err, jd)

    def chunk(p0, p1, m):
        """the own-path launches of one chunk at m pieces, every loaded layer on its card's workspace; then the chunk's
        outputs into the stand-in tensors (device copies)"""
        for c, L in Ls:
            il = L.il
            for r0, r1 in micro_bounds(p1 - p0, m):
                own(c, L, xr[il][p0 + r0:p0 + r1], ir[il][p0 + r0:p0 + r1], r0, r1 - r0)
        T = p1 - p0
        for c in (0, 1):
            src = Ws[c].o_down.view(-1)
            resid[c][:T * HC * D].copy_(src[:T * HC * D])
            for il in range(nl):
                hist['ids'][c][il][:T].copy_(torch.from_numpy(ir[layers[il % len(layers)]][p0:p1]).to(dev))
                hist['wts'][c][il][:T].copy_(src[il * T * k:(il + 1) * T * k].view(T, k))
        for i, (key, t) in enumerate(state.items()):
            c = key[0]; src = Ws[c].o_down.view(-1); o = (i * 9973) % (src.numel() - t.numel())
            t.copy_(src[o:o + t.numel()].view_as(t))
        for c in (0, 1):
            if int(Ws[c].err.item()) != 0: raise RuntimeError(f'd2h: card {c} shapes: the routing flag word {int(Ws[c].err.item()):#x}')

    def states():
        out = {}
        for key, t in state.items():
            c, il, kind = key
            if kind == 'gdn': out[key] = ttp.sha(t[:H[c] * S * S])
            elif kind in ('conv', 'ple'): out[key] = ttp.sha(t)
        return out

    addrs = []
    def note(t):
        addrs.append((t.data_ptr(), t.numel() * t.element_size())); return t

    def prompt(bounds):
        """test_prefill_tp stack's prompt() / after_chunk() host copies for the chunks, then states()"""
        res, ids, wts = [], [[], []], [[], []]
        for p0, p1, m in bounds:
            T = p1 - p0
            torch.cuda.synchronize(dev)
            chunk(p0, p1, m); torch.cuda.synchronize(dev)
            r_ok = torch.equal(note(resid[0][:T * HC * D].cpu()), note(resid[1][:T * HC * D].cpu()))
            rt_ok = all(torch.equal(note(hist['ids'][0][il][:T].cpu()), note(hist['ids'][1][il][:T].cpu())) and
                        torch.equal(note(hist['wts'][0][il][:T].cpu()), note(hist['wts'][1][il][:T].cpu())) for il in range(nl))
            ple_ok = all(torch.equal(note(state[(0, il, 'ple')].cpu()), note(state[(1, il, 'ple')].cpu())) for il in ple_layers)
            res.append(note(resid[0][:T * HC * D].cpu()))
            for c in (0, 1):
                ids[c].append(torch.stack([note(hist['ids'][c][il][:T].cpu()) for il in range(nl)]))
                wts[c].append(torch.stack([note(hist['wts'][c][il][:T].cpu()) for il in range(nl)]))
            del r_ok, rt_ok, ple_ok
        return dict(res=torch.cat(res), ids=[torch.cat(x, 1) for x in ids], wts=[torch.cat(x, 1) for x in wts], st=states())

    import hashlib
    ref, t0, lo_hi = None, time.time(), [None, None]
    for rnd in range(rounds):
        addrs.clear(); hs = []
        for tagp, bounds in prompts:
            r = prompt(bounds)
            h = hashlib.sha256(np.ascontiguousarray(r['res'].numpy()).tobytes()).hexdigest()[:16]
            hs.append((tagp, h, r['st']))
            del r
        lo = min(a for a, _ in addrs); hi = max(a + b for a, b in addrs)
        lo_hi = [lo if lo_hi[0] is None else min(lo, lo_hi[0]), hi if lo_hi[1] is None else max(hi, lo_hi[1])]
        if ref is None: ref = hs
        same = all(a[1] == b[1] and a[2] == b[2] for a, b in zip(hs, ref))
        if not same: check(False, f'd2h round {rnd}: the hashes differ from round 0\'s')
        if rnd % 10 == 0 or rnd == rounds - 1 or not same:
            print(f'  d2h round {rnd}: {len(addrs)} host copies, destinations [{lo:#x}, {hi:#x}), hashes {"== round 0" if same else "DIFFER"}; '
                  f'{time.time() - t0:.0f} s; {meminfo()}', flush=True)
    check(not FAILS, f'd2h: {rounds} rounds of own-path launches (2048 at m 3 / m 4, 777 + 1271 at m 1; layers {layers}, both cards\' shapes) '
                     f'followed by the stack\'s .cpu() + sha() pattern: no fault, every round\'s hashes == round 0\'s')
    NUMBERS['d2h'] = dict(rounds=rounds, host_range=[hex(v) for v in lo_hi], seconds=time.time() - t0, mem=meminfo())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('coder'); ap.add_argument('stages', nargs='+')
    ap.add_argument('--layers', default='0,1,2,12,17,21,47'); ap.add_argument('--cards', default='0,1')
    ap.add_argument('--routings', default=','.join(ROUTINGS)); ap.add_argument('--pieces', default='all', choices=('all', 'tiny'))
    ap.add_argument('--js', default='all', choices=('all', 'prod')); ap.add_argument('--rounds', type=int, default=200)
    ap.add_argument('--d2h-layers', default='1,17'); ap.add_argument('--json', default=None)
    ap.add_argument('--tp-split', default=None, help='GK,MB: the split whose card shapes are checked (default 10,12, the '
                    'measured XTX + XT split; 8,10: the even split, both cards\' down shards 320 wide with a K-split tail)')
    a = ap.parse_args()
    layers = [int(v) for v in a.layers.split(',')]; cards = [int(v) for v in a.cards.split(',')]
    if a.tp_split:
        global SPLIT
        SPLIT = tuple(int(v) for v in a.tp_split.split(','))
        nbt = tpm.MBR[-1][1]; tpm.MBR[:] = [(0, SPLIT[1]), (SPLIT[1], nbt)]     # (the shards' block ranges: tests/test_prefill_moe.Layer)
        print(f'split: gdn_keys {SPLIT[0]}, moe_blocks {SPLIT[1]} (card 0 ffc {32 * SPLIT[1]}, card 1 {32 * (nbt - SPLIT[1])})', flush=True)
    rkinds = [v for v in a.routings.split(',') if v]
    assert all(r in ROUTINGS for r in rkinds), rkinds
    from caminus.engine import flashnext   # noqa: F401  (gguf-py on the path)
    from gguf import GGUFReader
    rd = GGUFReader(a.coder)
    print('start', meminfo(), flush=True)
    for s in a.stages:
        if s == 'review': stage_review(rd); continue
        import torch
        name = torch.cuda.get_device_name(0)
        if torch.cuda.device_count() != 1 or 'XTX' in name or '7900 XT' not in name:
            raise SystemExit(f'{s}: this stage runs on the XT alone (HIP_VISIBLE_DEVICES=1); cuda:0 is {name!r}, {torch.cuda.device_count()} devices')
        print('device', name, meminfo(), flush=True)
        if s == 'canary': stage_canary(torch, rd, layers, cards, rkinds, a.pieces, a.js)
        elif s == 'd2h': stage_d2h(torch, rd, a.coder, a.rounds, [int(v) for v in a.d2h_layers.split(',')])
        else: raise SystemExit(f'unknown stage {s}')
    print('end', meminfo(), flush=True)
    if a.json: json.dump(dict(fails=FAILS, numbers=NUMBERS), open(a.json, 'w'), indent=1, default=str)
    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'), flush=True)
    sys.exit(0 if not FAILS else 1)


if __name__ == '__main__':
    main()
