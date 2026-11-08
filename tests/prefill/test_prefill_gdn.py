"""Gates of the batched prefill's GDN ops (docs/specs/spec-prefill-cache.md M5 S4; caminus.prefill.gdn,
kernels/prefill_gdn.hip): the l2 norm, the gates, the per-token recurrence with state snapshots, and the output norm.

  python -I tests/prefill/test_prefill_gdn.py CODER STAGE... [--small] [--json OUT]

Stages (GPU stages on cuda:0: HIP_VISIBLE_DEVICES picks the card; 'gpu' = g1 g2 speed):
  cpu    CPU only, nothing launched (launch recorders stand in for the kernels): every refusal of the launchers (short
         views, misaligned float4 buffers, overlapping written buffers, snapshot positions outside [1, T] / not increasing
         / too many, a snapshot buffer too short, mprev without records or with too few, H not a multiple of nk, a bad
         stride or workgroup size), each next to the positive control on the same buffers (the K3 layer block's state and
         records side by side are accepted: they do not overlap)
  g1     each op against K3's own decode op (caminus.engine.k3.Engine, module k3_gguf_gsq: the GDN tape of FlashNextTP,
         gdn_l2 -> gdn_gates -> gdn_step (+ gdn_save) -> gdn_norm, one token per pass with the lazy commit) on the same
         inputs, both TP cards' shapes (card 0: 10 key / 30 value heads; card 1: 6 / 18), real ssm_dt / ssm_a / ssm_norm
         weights of layer 0, random conv outputs, alpha / beta (with forced g ~ 1, g ~ 0 and beta ~ 1 tokens), z and a
         nonzero initial state; for T = 1, 4, 512, 777, 2048 bitwise: the q / k l2 rows, the gates, the recurrence's
         output at every token, the state after T tokens, the snapshots at positions inside the chunk (== the decode's
         committed state after that many tokens) and the norm's output. Also: the T = 4 verify-pass module
         (k3_gguf_gsq_qsa_mt, 4 tokens per pass) over the same 2048 tokens == the prefill, and the lazy-commit replay
         (a prefill started on a decode block with 1 / 4 accepted tokens pending == the decode continuing)
  g2     G2a: the chain twice at T = 2048, bitwise, and every recurrence kernel (prefill_gdn.STEP_KERNELS: 8 or 32 lanes
         per state column, 2 / 4 / 8 waves per workgroup, LDS blocks of 8 / 16 tokens or none) on the same inputs.
         G2b: the same 2048 tokens as chunks 1024 + 1024 / 777 + 1271 / 512 + 788 + 748 / 5 + 3 + 2040 / 1 + 4 + 2043
         (state carried) == one T = 2048 pass, bitwise per position and in the final state, and snapshots at the same
         absolute positions (on and across the split points) equal
  speed  ms per op at T = 2048 on both cards' shapes (median of 20 launches each, 3 interleaved rounds, the median
         round reported), every recurrence kernel interleaved, the GDN half of one layer, x 36 layers per
         chunk, and the longest single launch measured
  xcard  both cards: the same inputs on the XTX and the XT give the same bits (every op, the state, the snapshots)
  vram   CPU: the glue workspace with the GDN buffers (prefill_ops.glue_layout) and capacity.prefill_need_mib with it
--small: g1 / g2 at T <= 8 only (the first run of a new kernel on the XT).
Prints PASS / FAIL per check and RESULT PASS / FAIL; exit 0 = PASS."""
import argparse, contextlib, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'python'))
from caminus.engine import flashnext   # noqa: E402,F401  (puts gguf-py on the path)
from gguf import GGUFReader   # noqa: E402

FAILS, NUMBERS = [], {}
EPS = 9.999999974752427e-07
S, NG = 128, 16                                      # state width; GDN key heads (ssm groups)
GK = 10; KC, KS = [GK, NG - GK], [0, GK]             # the TP split (flashnext_tp._build_tapes, gdn_keys 10)
TS = (1, 4, 512, 777, 2048)
SPLITS = ((1024, 1024), (777, 1271), (512, 788, 748), (5, 3, 2040), (1, 4, 2043))
NL_GDN = 36


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)
    return ok


def bits(a):
    a = a.detach().cpu().numpy() if hasattr(a, 'cpu') else a
    return np.ascontiguousarray(a).view(np.int32)


def same(a, b):
    return tuple(a.shape) == tuple(b.shape) and np.array_equal(bits(a), bits(b))


def ndiff(a, b):
    return int((bits(a) != bits(b)).sum()) if tuple(a.shape) == tuple(b.shape) else -1


def acts(rng, n, scale=1.0, outl=0.002):
    x = rng.standard_normal(n).astype(np.float32) * scale
    x[rng.random(n) < outl] *= 30.0
    return x


def snaps_for(T):
    """snapshot positions inside a chunk of T tokens (at most 8, increasing, in [1, T])"""
    c = sorted({p for p in (1, 2, 3, 5, T // 3, T // 2 + 1, T - 1, T) if 1 <= p <= T})
    return c[:8] if len(c) <= 8 else c[:4] + c[-4:]


class W:
    def __init__(self, rd):
        self.T = {t.name: t for t in rd.tensors}

    def f32(self, name):
        t = self.T[name]; assert t.tensor_type.name == 'F32', name
        return np.asarray(t.data).reshape(-1).view(np.float32)


# ---- one card's shapes and inputs -----------------------------------------------------------------------------------------
class Case:
    def __init__(self, torch, dev, wt, card, T0=2048, seed=21):
        rng = np.random.default_rng(seed + card); self.T0, self.card = T0, card
        kc = KC[card]; self.kc, self.H = kc, 3 * kc; self.CH = 640 * kc; H = self.H
        put = lambda a: torch.from_numpy(np.array(a, dtype=np.float32)).to(dev)      # (a copy: gguf-py's arrays are read-only)
        KH = list(range(KS[card], KS[card] + kc)); VH = [h for h in range(48) if h % NG in KH]
        self.dt = put(wt.f32('blk.0.ssm_dt.bias')[VH]); self.a = put(wt.f32('blk.0.ssm_a')[VH])
        self.wn = put(wt.f32('blk.0.ssm_norm.weight')[:S])
        self.x = put(acts(rng, T0 * self.CH, 0.5).reshape(T0, self.CH))        # the conv output rows (q, k, v)
        al = rng.standard_normal((T0, H)).astype(np.float32) * 2.0; be = rng.standard_normal((T0, H)).astype(np.float32) * 2.0
        for t in range(7, T0, 97): al[t] = -40.0                                 # softplus ~ 0: g = exp(~0) ~ 1 (no decay)
        for t in range(11, T0, 89): al[t] = 25.0                                 # the ab > 20 branch: g = exp(25 a) ~ 0
        for t in range(13, T0, 61): be[t] = 30.0                                 # beta_sig = 1
        self.al, self.be = put(al), put(be)
        self.zg = put(acts(rng, T0 * H * S).reshape(T0, H * S))
        self.st0 = put(acts(rng, H * S * S, 0.05).reshape(H, S, S))

    def v(self, x):
        """the value rows of conv output rows x (contiguous [n, CH]): a flat view from the first value head on"""
        return x.reshape(-1)[256 * self.kc:]


# ---- K3's decode ops (the reference) --------------------------------------------------------------------------------------
def k3_ref(torch, dev, cs, ntok=1, keep=(), block_at=None):
    """the GDN tape (FlashNextTP's) over the case's T0 tokens, ntok per pass, lazy commit (params[1] = ntok after the first
    pass, one more pass commits the last tokens): per token qk, gates, att, fin; the committed state after n tokens for n
    in keep (multiples of ntok); the layer block (state + records) after the pass that ends at token block_at"""
    from caminus.engine import k3
    e = k3.Engine(module='k3_gguf_gsq', grid=k3.resident_grid('k3_gguf_gsq'), T=ntok)
    H, kc, CH, T0 = cs.H, cs.kc, cs.CH, cs.T0; z = lambda *s: torch.zeros(*s, device=dev)
    xb, alb, beb, zb = z(ntok, CH), z(ntok, H), z(ntok, H), z(ntok, H * S)
    qkb, gtb, attb, finb = z(ntok, 2 * kc * S), z(ntok, 2 * H), z(ntok, H * S), z(ntok, H * S)
    blk = z(k3.gdn_block_size(H, S, kc, ntok))
    l_ = e.gdn_l2(xb, qkb, 2 * kc, S, EPS, x_stride=CH)
    g_ = e.gdn_gates(alb, beb, cs.dt, cs.a, gtb)
    st = e.also(e.gdn_step(qkb, xb.reshape(-1)[256 * kc:], gtb, blk, attb, H, S, kc, CH, dep=l_), g_)
    e.gdn_save(qkb, xb.reshape(-1)[256 * kc:], gtb, blk, H, S, kc, CH, dep=st)
    e.gdn_norm(attb, cs.wn, zb, finb, H, S, EPS, dep=st)
    e.build()
    blk[:H * S * S].copy_(cs.st0.reshape(-1))
    out = dict(qk=z(T0, 2 * kc * S), gates=z(T0, 2 * H), att=z(T0, H * S), fin=z(T0, H * S)); states = {}; blocks = {}
    npass = T0 // ntok
    for p in range(npass + 1):
        e.params[1] = 0 if p == 0 else ntok
        if p < npass:
            sl = slice(p * ntok, (p + 1) * ntok)
            xb.copy_(cs.x[sl]); alb.copy_(cs.al[sl]); beb.copy_(cs.be[sl]); zb.copy_(cs.zg[sl])
        else:
            xb.zero_(); alb.zero_(); beb.zero_(); zb.zero_()
        e.run()
        if p * ntok in keep: states[p * ntok] = blk[:H * S * S].clone().reshape(H, S, S)
        if p < npass:
            for k, b in (('qk', qkb), ('gates', gtb), ('att', attb), ('fin', finb)): out[k][sl].copy_(b)
            if block_at is not None and (p + 1) * ntok == block_at: blocks[block_at] = blk.clone()
    torch.cuda.synchronize(dev)
    return out, states, blocks


# ---- the prefill chain on tokens [lo, hi) -------------------------------------------------------------------------------
def chain(torch, G, cs, lo, hi, state, snap_pos=(), kernel=None, rec=None, mprev=0):
    """l2 -> gates -> step -> norm on the case's tokens [lo, hi); state updated in place; returns outputs + snapshots"""
    dev = cs.x.device; T = hi - lo; H, kc, CH = cs.H, cs.kc, cs.CH
    z = lambda *s: torch.zeros(*s, device=dev)
    x = cs.x[lo:hi]; qk, gt, att, fin = z(T, 2 * kc * S), z(T, 2 * H), z(T, H * S), z(T, H * S)
    G.l2(x, qk, T, 2 * kc, EPS, ldx=CH)
    G.gates(cs.al[lo:hi], cs.be[lo:hi], cs.dt, cs.a, gt, T, H)
    sn = z(max(1, len(snap_pos)), H, S, S)
    G.step(qk, cs.v(x), gt, state, att, T, H, kc, CH, rec=rec, mprev=mprev, snap=sn if snap_pos else None, snap_pos=snap_pos, kernel=kernel)
    G.norm(att, cs.wn, cs.zg[lo:hi], fin, T, H, EPS)
    return dict(qk=qk, gates=gt, att=att, fin=fin), [sn[i] for i in range(len(snap_pos))]


# ---- stage: g1 ------------------------------------------------------------------------------------------------------------
def stage_g1(torch, dev, wt, small):
    from caminus.prefill.gdn import Gdn
    G = Gdn(dev); t0 = time.time()
    T0 = 8 if small else 2048; ts = (1, 4, 5, 8) if small else TS
    for card in (0, 1):
        cs = Case(torch, dev, wt, card, T0=T0); H = cs.H
        keep = set(ts) | {p for T in ts for p in snaps_for(T)}
        n_rep = 5 if small else 600                                            # the lazy-commit replay start
        ref, st_ref, blk1 = k3_ref(torch, dev, cs, 1, keep | {n_rep + (3 if small else 200)}, block_at=n_rep)
        degenerate = [k for k, v in ref.items() if not (torch.isfinite(v).all().item() and (v != 0).float().mean().item() > 0.5)]
        check(not degenerate, f'G1 card-{card} shapes: the K3 reference outputs are finite and nonzero ({degenerate or "all"})')
        bad = []
        for T in ts:
            st = cs.st0.clone(); pos = snaps_for(T)
            o, sn = chain(torch, G, cs, 0, T, st, pos)
            for k in o:
                if not same(o[k], ref[k][:T]): bad.append(f'T={T} {k}: {ndiff(o[k], ref[k][:T])} differ')
            if not same(st, st_ref[T]): bad.append(f'T={T} state: {ndiff(st, st_ref[T])} differ')
            for p, s_ in zip(pos, sn):
                if not same(s_, st_ref[p]): bad.append(f'T={T} snapshot at {p}: {ndiff(s_, st_ref[p])} differ')
        check(not bad, f'G1 card-{card} shapes (H {H}, nk {cs.kc}): l2, gates, the recurrence (every token), the state after T, the '
              f'snapshots inside the chunk and the norm == K3 decode ops one token per pass, bitwise, T = {"/".join(map(str, ts))}'
              + (f' ({bad[:4]})' if bad else ''))
        # the lazy-commit replay: a prefill on the decode block of pass n_rep - 1 (token n_rep - 1 accepted, not committed)
        L = 3 if small else 200; blk = blk1[n_rep]; st = blk[:H * S * S].clone().reshape(H, S, S)
        o, _ = chain(torch, G, cs, n_rep, n_rep + L, st, rec=blk[H * S * S:], mprev=1)
        ok = same(o['att'], ref['att'][n_rep:n_rep + L]) and same(st, st_ref[n_rep + L])
        check(ok, f'G1 card-{card}: prefill of tokens {n_rep}..{n_rep + L - 1} on the decode block with 1 accepted token pending '
              f'(mprev 1, its OP_GDN_SAVE record replayed) == the decode continuing (outputs, state), bitwise')
        if not small:
            # the T = 4 verify-pass module over the same tokens
            ref4, st4, blk4 = k3_ref(torch, dev, cs, 4, {T0, 1024, 604 + 200}, block_at=604)
            ok = all(same(ref4[k], ref[k]) for k in ref) and same(st4[T0], st_ref[T0])
            check(ok, f'G1 card-{card}: K3 T = 4 passes ({"k3_gguf_gsq_qsa_mt"}) == one-token passes over {T0} tokens (outputs, '
                  'state): the reference holds for both decode modules')
            st = cs.st0.clone(); o, sn = chain(torch, G, cs, 0, T0, st, (1024, T0))
            ok = all(same(o[k], ref4[k]) for k in o) and same(st, st4[T0]) and same(sn[0], st4[1024])
            check(ok, f'G1 card-{card}: prefill T = {T0} == K3 T = 4 passes, bitwise (outputs, state, snapshot at 1024)')
            b = blk4[604]; st = b[:H * S * S].clone().reshape(H, S, S)
            o, _ = chain(torch, G, cs, 604, 804, st, rec=b[H * S * S:], mprev=4)
            check(same(o['att'], ref4['att'][604:804]) and same(st, st4[804]),
                  f'G1 card-{card}: prefill on the T = 4 decode block with 4 accepted tokens pending (mprev 4) == the decode '
                  'continuing, bitwise')
        del ref, st_ref
    print(f'(g1 in {time.time() - t0:.0f} s)', flush=True)


# ---- stage: g2 ------------------------------------------------------------------------------------------------------------
def stage_g2(torch, dev, wt, small):
    from caminus.prefill.gdn import Gdn, STEP_KERNELS
    G = Gdn(dev)
    T0 = 8 if small else 2048
    splits = ((4, 4), (1, 2, 5), (3, 5)) if small else SPLITS
    absp = (1, 3, 4, 5, 8) if small else (1, 5, 512, 777, 1024, 1300, 2043, 2048)
    for card in (0, 1):
        cs = Case(torch, dev, wt, card, T0=T0, seed=31)
        st1 = cs.st0.clone(); a, sa = chain(torch, G, cs, 0, T0, st1, absp)
        st2 = cs.st0.clone(); b, sb = chain(torch, G, cs, 0, T0, st2, absp)
        ok = all(same(a[k], b[k]) for k in a) and same(st1, st2) and all(same(x, y) for x, y in zip(sa, sb))
        check(ok, f'G2a card-{card}: two runs of the chain at T = {T0}, bitwise (outputs, state, {len(absp)} snapshots)')
        bad = []
        for w in STEP_KERNELS:
            st = cs.st0.clone(); o, sn = chain(torch, G, cs, 0, T0, st, absp, kernel=w)
            if not (same(o['att'], a['att']) and same(st, st1) and all(same(x, y) for x, y in zip(sn, sa))): bad.append(w)
        check(not bad, f'G2a card-{card}: every recurrence kernel ({"/".join(STEP_KERNELS)}: 8 or 32 lanes per column, workgroups of 2 / 4 / 8 waves, LDS blocks of 8 / 16 tokens or none) gives the same bits '
              f'(outputs, state, snapshots)' + (f' (differ: {bad})' if bad else ''))
        for sp in splits:
            st = cs.st0.clone(); parts = {}; snp = {}; lo = 0
            for n in sp:
                hi = lo + n; mine = [p for p in absp if lo < p <= hi]
                o, sn = chain(torch, G, cs, lo, hi, st, [p - lo for p in mine])
                for k, v in o.items(): parts.setdefault(k, []).append(v)
                snp.update(zip(mine, sn)); lo = hi
            bad = [k for k in a if not same(torch.cat(parts[k]), a[k])]
            if not same(st, st1): bad.append('state')
            bad += [f'snapshot {p}' for p, s_ in zip(absp, sa) if not same(snp[p], s_)]
            check(not bad, f'G2b card-{card}: chunks {" + ".join(map(str, sp))} == one T = {T0} pass, bitwise per position (l2, '
                  f'gates, recurrence, norm), the final state and the snapshots at {absp}' + (f' (differ: {bad})' if bad else ''))


# ---- stage: xcard -----------------------------------------------------------------------------------------------------------
def stage_xcard(torch, wt):
    from caminus.prefill.gdn import Gdn
    res = {}
    for d in (1, 0):                                                         # (the XT first)
        dev = torch.device(f'cuda:{d}'); torch.cuda.set_device(dev); G = Gdn(dev); r = {}
        for card in (0, 1):
            cs = Case(torch, dev, wt, card, seed=41); st = cs.st0.clone()
            o, sn = chain(torch, G, cs, 0, cs.T0, st, snaps_for(cs.T0))
            r[card] = ({k: v.cpu() for k, v in o.items()}, st.cpu(), [s_.cpu() for s_ in sn])
            del cs, o, sn, st
        res[d] = (r, torch.cuda.get_device_name(dev)); torch.cuda.synchronize(dev); torch.cuda.empty_cache()
    (a, na), (b, nb) = res[0], res[1]
    bad = []
    for card in (0, 1):
        oa, sa, pa = a[card]; ob, sb, pb = b[card]
        bad += [f'card-{card} {k}' for k in oa if not same(oa[k], ob[k])]
        if not same(sa, sb): bad.append(f'card-{card} state')
        bad += [f'card-{card} snapshot {i}' for i, (x, y) in enumerate(zip(pa, pb)) if not same(x, y)]
    check(not bad, f'xcard: {na} and {nb} give the same bits on the same inputs (both cards\' shapes: l2, gates, recurrence, norm, '
          f'state, snapshots) at T = 2048' + (f' (differ: {bad})' if bad else ''))


# ---- stage: speed -----------------------------------------------------------------------------------------------------------
def stage_speed(torch, dev, wt, reps=20, rounds=3):
    from caminus.prefill.gdn import Gdn, STEP_KERNELS, STEP_KERNEL
    G = Gdn(dev); T = 2048
    comp = open('/proc/sys/vm/compaction_proactiveness').read().strip()
    print('device', torch.cuda.get_device_name(dev), '| compaction_proactiveness', comp, flush=True)
    longest = [0.0, '']

    def tm(name, fn):
        fn(); torch.cuda.synchronize(dev); ts = []
        for _ in range(reps):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); fn(); e1.record(); torch.cuda.synchronize(dev); ts.append(e0.elapsed_time(e1))
        if max(ts) > longest[0]: longest[:] = [max(ts), name]
        return float(np.median(ts))
    out = {}
    for card in (0, 1):
        cs = Case(torch, dev, wt, card, seed=51); H, kc, CH = cs.H, cs.kc, cs.CH
        z = lambda *s: torch.zeros(*s, device=dev)
        qk, gt, att, fin = z(T, 2 * kc * S), z(T, 2 * H), z(T, H * S), z(T, H * S); st = cs.st0.clone()
        G.l2(cs.x, qk, T, 2 * kc, EPS, ldx=CH); G.gates(cs.al, cs.be, cs.dt, cs.a, gt, T, H)
        sn = z(4, H, S, S); pos = (512, 1024, 1536, 2048)
        ops = {f'l2 ({2 * kc} heads)': lambda: G.l2(cs.x, qk, T, 2 * kc, EPS, ldx=CH),
               'gates': lambda: G.gates(cs.al, cs.be, cs.dt, cs.a, gt, T, H),
               'norm': lambda: G.norm(att, cs.wn, cs.zg, fin, T, H, EPS)}
        for w in STEP_KERNELS:
            ops[f'step {w}'] = (lambda w=w: G.step(qk, cs.v(cs.x), gt, st, att, T, H, kc, CH, kernel=w))
        ops[f'step {STEP_KERNEL} + 4 snapshots'] = lambda: G.step(qk, cs.v(cs.x), gt, st, att, T, H, kc, CH, snap=sn, snap_pos=pos)

        def layer():
            G.l2(cs.x, qk, T, 2 * kc, EPS, ldx=CH); G.gates(cs.al, cs.be, cs.dt, cs.a, gt, T, H)
            G.step(qk, cs.v(cs.x), gt, st, att, T, H, kc, CH); G.norm(att, cs.wn, cs.zg, fin, T, H, EPS)
        ops['GDN layer (l2 + gates + step + norm)'] = layer
        rr = {k: [] for k in ops}
        for _ in range(rounds):                                              # interleaved rounds
            for k, fn in ops.items(): rr[k].append(tm(f'card-{card} {k}', fn))
        res = {k: float(np.median(v)) for k, v in rr.items()}
        for k, v in res.items():
            print(f'  SPEED card-{card} shapes (H {H}): {k:40s} {v:8.4f} ms  (T = {T}; rounds {", ".join(f"{x:.4f}" for x in rr[k])})', flush=True)
        lay = res['GDN layer (l2 + gates + step + norm)']
        print(f'  CHUNK card-{card} shapes: GDN half x {NL_GDN} layers at T = {T}: {NL_GDN * lay:.1f} ms '
              f'(the recurrence alone {NL_GDN * res[f"step {STEP_KERNEL}"]:.1f} ms)', flush=True)
        out[card] = dict(ops_ms=res, rounds_ms=rr, layer_ms=lay, chunk_ms=NL_GDN * lay)
        del cs
    print(f'  LONGEST single launch timed: {longest[0]:.3f} ms ({longest[1]})', flush=True)
    check(longest[0] < 50.0, f'speed: every timed launch under 50 ms (longest {longest[0]:.3f} ms: {longest[1]})')
    NUMBERS['speed'] = dict(device=torch.cuda.get_device_name(dev), compaction=comp, cards=out, longest_ms=longest[0], longest=longest[1])


# ---- stage: cpu (refusals, nothing launched) --------------------------------------------------------------------------------
class Rec:
    def __init__(self, log): self.log = log

    def __getattr__(self, name):
        if name.startswith('__'): raise AttributeError(name)
        return lambda *a, **kw: self.log.append(name)


def refused(log, fn, what):
    n0 = len(log)
    try:
        fn()
    except ValueError as e:
        return check(len(log) == n0, f'refused, nothing launched: {what} ({str(e)[:110]})')
    except Exception as e:
        return check(False, f'{what}: raised {type(e).__name__} {e} instead of ValueError')
    return check(False, f'{what}: NOT refused ({len(log) - n0} launches recorded)')


def accepted(log, fn, what):
    n0 = len(log)
    try:
        fn()
    except Exception as e:
        return check(False, f'positive control {what}: refused ({type(e).__name__}: {str(e)[:160]})')
    return check(len(log) > n0, f'positive control: {what} accepted ({len(log) - n0} launch recorded, none run)')


def stage_cpu():
    import torch
    from caminus.engine import k3
    from caminus.prefill import gdn as pg
    real = torch.cuda.device
    torch.cuda.device = lambda d: contextlib.nullcontext()                 # (the recorders stand in; nothing is launched)
    try:
        cpu = torch.device('cpu'); log = []
        G = pg.Gdn.__new__(pg.Gdn); G.dev = cpu; G.m = Rec(log)
        T, kc = 64, 10; H, CH = 3 * kc, 640 * kc; z = lambda n: torch.zeros(int(n), dtype=torch.float32)
        x = z(T * CH); qk = z(T * 2 * kc * S); gt = z(T * 2 * H); st = z(H * S * S); att = z(T * H * S); fin = z(T * H * S)
        al, be, dt, a, wn, zg = z(T * H), z(T * H), z(H), z(H), z(S), z(T * H * S)
        v = x[256 * kc:]
        blk = z(k3.gdn_block_size(H, S, kc, 4)); bst, brec = blk[:H * S * S], blk[H * S * S:]
        sn = z(3 * H * S * S)
        step = lambda **kw: G.step(**{**dict(qk=qk, v=v, gates=gt, state=st, out=att, T=T, H=H, nk=kc, ldv=CH), **kw})
        accepted(log, lambda: step(), 'step')
        accepted(log, lambda: step(snap=sn, snap_pos=(1, 33, T)), 'step with 3 snapshots (1, 33, T)')
        accepted(log, lambda: step(state=bst, rec=brec, mprev=4), 'step on a K3 layer block: state and records side by side, mprev 4')
        for w in pg.STEP_KERNELS: accepted(log, lambda w=w: step(kernel=w), f'step, kernel {w}')
        refused(log, lambda: step(qk=qk[:-1]), 'step: qk one float short (its view)')
        refused(log, lambda: step(v=x[256 * kc:-1]), 'step: v short of (T - 1) ldv + H * 128')
        xw = z(T * CH + 8); refused(log, lambda: step(v=xw[256 * kc + 1:]), 'step: v long enough but not 16-byte aligned (float4 loads)')
        refused(log, lambda: step(state=st[:-4]), 'step: state short')
        refused(log, lambda: step(state=z(H * S * S + 1)[1:]), 'step: state misaligned')
        refused(log, lambda: step(out=att[:-1]), 'step: out short')
        refused(log, lambda: step(out=x[256 * kc:256 * kc + T * H * S]), 'step: out the same bytes as v (no in-place form)')
        refused(log, lambda: step(out=x[4:4 + T * H * S]), 'step: out partially overlapping v')
        refused(log, lambda: step(state=blk[:H * S * S], rec=blk[H * S * S - 4:], mprev=1), 'step: state partially overlapping the records')
        refused(log, lambda: step(snap=sn, snap_pos=(0,)), 'step: snapshot position 0')
        refused(log, lambda: step(snap=sn, snap_pos=(T + 1,)), 'step: snapshot position T + 1')
        refused(log, lambda: step(snap=sn, snap_pos=(5, 5)), 'step: snapshot positions repeated')
        refused(log, lambda: step(snap=sn, snap_pos=(9, 5)), 'step: snapshot positions decreasing')
        refused(log, lambda: step(snap=z(9 * H * S * S), snap_pos=tuple(range(1, 10))), 'step: 9 snapshot positions (> MAX_SNAP)')
        refused(log, lambda: step(snap_pos=(3,)), 'step: snapshot positions without a buffer')
        refused(log, lambda: step(snap=sn[:-1], snap_pos=(1, 2, 3)), 'step: snapshot buffer one float short of 3 states')
        refused(log, lambda: step(snap=st, snap_pos=(3,)), 'step: snapshot buffer is the state')
        refused(log, lambda: step(snap=sn, snap_pos=(1, 2), snap_stride=H * S * S - 4), 'step: snapshot stride below one state')
        refused(log, lambda: step(mprev=1), 'step: mprev 1 without records')
        refused(log, lambda: step(rec=brec, mprev=5), 'step: mprev 5 with records for 4')
        refused(log, lambda: step(rec=brec, mprev=-1), 'step: mprev -1')
        refused(log, lambda: step(nk=7), 'step: H 30 not a multiple of nk 7')
        refused(log, lambda: step(ldv=H * S - 4), 'step: ldv below H * 128')
        refused(log, lambda: step(ldv=CH + 1), 'step: ldv not a multiple of 4')
        refused(log, lambda: step(kernel='w3'), 'step: an unknown kernel')
        refused(log, lambda: step(qk=torch.zeros(T * 2 * kc * S, dtype=torch.float16)), 'step: qk float16')
        accepted(log, lambda: G.l2(x, qk, T, 2 * kc, EPS, ldx=CH), 'l2')
        refused(log, lambda: G.l2(x[:(T - 1) * CH + 2 * kc * S - 1], qk, T, 2 * kc, EPS, ldx=CH), 'l2: x one float short')
        refused(log, lambda: G.l2(x, qk[:-1], T, 2 * kc, EPS, ldx=CH), 'l2: y short')
        refused(log, lambda: G.l2(x, x[CH:CH + T * 2 * kc * S], T, 2 * kc, EPS, ldx=CH), 'l2: y overlapping x')
        refused(log, lambda: G.l2(x, qk, T, 2 * kc, EPS, ldx=2 * kc * S - 1), 'l2: ldx below heads * K')
        accepted(log, lambda: G.gates(al, be, dt, a, gt, T, H), 'gates')
        refused(log, lambda: G.gates(al[:-1], be, dt, a, gt, T, H), 'gates: alpha short')
        refused(log, lambda: G.gates(al, be, dt, a, gt[:-1], T, H), 'gates: out short')
        gb = z(3 * T * H); refused(log, lambda: G.gates(gb[:T * H], be, dt, a, gb[T * H // 2:T * H // 2 + 2 * T * H], T, H), 'gates: out partially overlapping alpha')
        accepted(log, lambda: G.norm(att, wn, zg, fin, T, H, EPS), 'norm')
        refused(log, lambda: G.norm(att, wn, zg[:-1], fin, T, H, EPS), 'norm: z short')
        refused(log, lambda: G.norm(att, wn, zg, att, T, H, EPS), 'norm: y is o')
        refused(log, lambda: G.norm(att, wn, zg, zg, T, H, EPS), 'norm: y is z')
    finally:
        torch.cuda.device = real


# ---- stage: vram (CPU) ----------------------------------------------------------------------------------------------------
def stage_vram(coder):
    from caminus.device import capacity
    from caminus.prefill import glue as po
    g = capacity.geometry(coder)
    for T in (1024, 2048):
        d = [po.glue_dims(g, c, 10) for c in (0, 1)]; L = [po.glue_layout(T, d[c]) for c in (0, 1)]
        gd = [sum(v for k, v in L[c].items() if k.startswith('gdn_')) / capacity.MIB for c in (0, 1)]
        b = [po.glue_bytes(T, d[c]) / capacity.MIB for c in (0, 1)]
        need = capacity.prefill_need_mib(g, T, gdn_keys=10, moe_blocks=12)
        print(f'  chunk {T}: GDN buffers (gdn_qk + gdn_gates + gdn_att) XTX {gd[0]:.1f} / XT {gd[1]:.1f} MiB; glue workspace with them '
              f'{b[0]:.1f} / {b[1]:.1f} MiB; prefill_need_mib {need[0]:.1f} / {need[1]:.1f} MiB', flush=True)
        if T == 2048: NUMBERS['vram'] = dict(gdn_mib=gd, glue_mib=b, prefill_need_mib=need)
        kc = [10, 6]
        want = [4 * T * (2 * kc[c] * S + 2 * 3 * kc[c] + 3 * kc[c] * S) for c in (0, 1)]
        got = [sum(v for k, v in L[c].items() if k.startswith('gdn_')) for c in (0, 1)]
        check(all(got[c] >= want[c] and got[c] - want[c] < 3 * po.ALIGN for c in (0, 1)),
              f'vram: the glue layout holds the GDN buffers at chunk {T} (qk [T, 2 nk, 128], gates [T, 2H], att [T, H, 128]): '
              f'{got} B for {want} B')
    s2 = capacity.prefill_workspace_bytes(g, 2048, gdn_keys=10, moe_blocks=12); f = capacity.prefill_feed_bytes(g, 2048)
    gl = capacity.prefill_glue_bytes(g, 2048, gdn_keys=10); need = capacity.prefill_need_mib(g, 2048, gdn_keys=10, moe_blocks=12)
    xb, ab = capacity.prefill_xchg_bytes(g, 2048), capacity.prefill_attn_bytes(g, 2048)
    check(all(abs(need[c] - (s2[c] + f[c] + gl[c] + xb[c] + ab[c]) / capacity.MIB - capacity.PREFILL_CODE_MIB[c] - capacity.PREFILL_STREAMS_MIB[c]) < 1e-6 for c in (0, 1)),
          'vram: capacity.prefill_need_mib = S2 GEMM workspace + feed + glue workspace (with the GDN buffers) + S5 exchange receive buffer + S6 attention workspace + the prefill modules and streams')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('coder'); ap.add_argument('stages', nargs='+')
    ap.add_argument('--small', action='store_true'); ap.add_argument('--json', default=None)
    a = ap.parse_args()
    stages = []
    for s in a.stages: stages += ['g1', 'g2', 'speed'] if s == 'gpu' else [s]
    if 'cpu' in stages: stage_cpu()
    if 'vram' in stages: stage_vram(a.coder)
    need_w = any(s in stages for s in ('g1', 'g2', 'speed', 'xcard'))
    wt = W(GGUFReader(a.coder, 'r')) if need_w else None
    if 'xcard' in stages:
        import torch
        stage_xcard(torch, wt)
    gpu = [s for s in stages if s in ('g1', 'g2', 'speed')]
    if gpu:
        import torch
        dev = torch.device('cuda:0'); torch.cuda.set_device(dev)
        print('device', torch.cuda.get_device_name(dev), flush=True)
        if 'g1' in gpu: stage_g1(torch, dev, wt, a.small)
        if 'g2' in gpu: stage_g2(torch, dev, wt, a.small)
        if 'speed' in gpu: stage_speed(torch, dev, wt)
    if a.json: json.dump(dict(fails=FAILS, numbers=NUMBERS), open(a.json, 'w'), indent=1, default=str)
    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'))
    sys.exit(0 if not FAILS else 1)


if __name__ == '__main__':
    main()
