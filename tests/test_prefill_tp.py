"""Gates of the batched prefill on both TP cards (docs/agentic/spec-prefill-cache.md M5 S5; caminus.prefill_tp,
kernels/prefill_tp.hip): the exchange with micro-chunk overlap, measured inside a whole 48-layer chunk.

  python -I tests/test_prefill_tp.py CODER STAGE... [--small] [--chunk 2048] [--reps 4] [--ms 1,2,3,4] [--moe own|mmq] [--no-moe-ab] [--json OUT]

Stages:
  cpu      CPU only, nothing launched (launch recorders stand in for the kernels): every refusal of the prefill_tp
           launchers (short views, misaligned float4 buffers, overlapping written buffers) next to the positive controls
           (the in-place forms: the exchange sum over either partial, the combine over the shared expert's rows);
           micro_bounds (covers the chunk, every piece but the last on the 128-row grid, none empty); the exchange's
           receive buffer in capacity.prefill_need_mib
  kernels  GPU (cuda:0; HIP_VISIBLE_DEVICES=1 = the XT first): pf_xchg_add == the fp32 sum bitwise (out of place and both
           in-place forms); pf_moe_combine's slot-order sum == its float32 emulation bitwise, and the gated shared-expert
           add == K3's OP_SHEXP_ADD per token bitwise; pf_attn_standin == K3's OP_GATE_MUL on the v rows per token
           bitwise; pf_swiglu against fp64 (ggml's silu with expf: not bitwise anything, reported); T = 1 / 4 / 512 / 777 /
           2048 (--small: 4 / 8)
  stack    both cards (cuda:0 = XTX, cuda:1 = XT), the loaded Coder (FlashNextTP, T = 4, --max-ctx, the prefill counted in
           its VRAM check): FlashNextTP.weights are the tapes' own tensors; the two-card feed; the 48-layer chunk
           (caminus.prefill_tp.ChunkStack: S2 + S3 + S4 ops, the exchange, the safety step's router: WMMA logits, the id
           clamp, the routing flag words) on real text (qsa/lc/d8192/ctx.ids). Exchange refusals (nothing queued). A
           256-token chunk first. Then at --chunk: G2b (the residual streams of every position, every layer's routing on
           both cards, the final GDN / conv / PLE states) bitwise across micro-chunks 1 / 2 / 3 / 4 (serial and overlap)
           and chunk splits 1024 + 1024 / 777 + 1271; after every chunk the replicas (residual, routing, PLE history: card 0
           == card 1), the routing flag words and the feed's bound flags. Timing, interleaved rounds (round 0 = cold, the
           median of the rest = warm): local (no exchange, each card its own partial: the floor), serial and overlap per
           micro-chunk count, the exchange's event release scope (all / ready / none) and half duplex at m 2, and overlap with a
           concurrent feed-sized P2P copy (the link's contention); end to end per
           chunk (the device feed included; the host feed's part prepared beforehand, measured separately), the host's
           enqueue time beside it. 8K from empty: 4 chunks back to back (the next chunk's host feed on the feed's worker
           thread). The breakdown from event-instrumented runs (GEMMs by kind and format, WMMA, glue, GDN, exchange wait
           and add, feed). --small: the 256-token chunk only. The routed experts' paths (caminus.moe, ChunkStack(moe=):
           --moe own | mmq, default prefill_tp.MOE_PATH): the other path on the same prompts (one chunk m 1 serial, m 2
           overlap, 777 + 1271) == the reference bitwise (residual, routing, states), and both paths timed interleaved
           (overlap m 2 and local m 2, warm medians) with each one's MoE + glue time from the breakdown. --no-moe-ab: the
           other path never runs (those three parts skipped, everything else unchanged; --moe alone still runs both paths).
           The dense Q6_K paths (caminus.dense, ChunkStack(q6k=): --q6k own | wmma | mmq, default prefill_tp.Q6K_PATH): each
           other path on the same three prompts: 'own' / 'mmq' == the reference bitwise; 'wmma' (a tolerance) == its own one
           chunk m 1 serial bitwise (G2b, replicas) and its distance from the reference reported; all paths timed interleaved
           (overlap / local m 2) with each one's dense Q6_K time from the breakdown. --no-q6k-ab: the other paths never run
Prints PASS / FAIL per check and RESULT PASS / FAIL; exit 0 = PASS."""
import argparse, contextlib, hashlib, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'python'))
from caminus import flashnext   # noqa: E402,F401  (puts gguf-py on the path)

from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
ART = devpaths.artifacts()
TOKENS = os.path.join(ART, 'qsa', 'lc', 'd8192', 'ctx.ids')
FAILS, NUMBERS = [], {}


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)
    return ok


def bits(a):
    a = a.detach().cpu().numpy() if hasattr(a, 'cpu') else a
    return np.ascontiguousarray(a).view(np.int32)


def same(a, b):
    return np.array_equal(bits(a), bits(b))


def ndiff(a, b):
    return int((bits(a) != bits(b)).sum())


def comp():
    return open('/proc/sys/vm/compaction_proactiveness').read().strip()


# ---- stage: cpu ---------------------------------------------------------------------------------------------------------
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


def stage_cpu(coder):
    import torch
    from caminus import prefill_tp as pt, capacity
    real = torch.cuda.device
    torch.cuda.device = lambda d: contextlib.nullcontext()                 # (the recorders stand in; nothing is launched)
    try:
        log = []; P = pt.Tp('cpu', module=Rec(log))
        z = lambda n: torch.zeros(int(n), dtype=torch.float32)
        n = 4096; a, b, y = z(n), z(n), z(n)
        accepted(log, lambda: P.xchg_add(a, b, y, n), 'xchg_add out of place')
        accepted(log, lambda: P.xchg_add(a, b, a, n), 'xchg_add in place over a (card 0: own partial + received)')
        accepted(log, lambda: P.xchg_add(a, b, b, n), 'xchg_add in place over b (card 1: received + own partial)')
        refused(log, lambda: P.xchg_add(a[:-4], b, y, n), 'xchg_add: a short (its view)')
        refused(log, lambda: P.xchg_add(a, b, y[:-1], n), 'xchg_add: y short')
        refused(log, lambda: P.xchg_add(a, b, y, n - 2), 'xchg_add: n not a multiple of 4 (float4)')
        big = z(n + 8)
        refused(log, lambda: P.xchg_add(big[1:n + 1], b, y, n), 'xchg_add: a not 16-byte aligned')
        refused(log, lambda: P.xchg_add(big[:n], big[4:n + 4], big[4:n + 4], n), 'xchg_add: y = b in place but b partially overlaps a')
        refused(log, lambda: P.xchg_add(big[:n], b, big[4:n + 4], n), 'xchg_add: y partially overlapping a')
        g, u, h = z(n), z(n), z(n)
        accepted(log, lambda: P.swiglu(g, u, h, n), 'swiglu')
        refused(log, lambda: P.swiglu(g, u, h[:-1], n), 'swiglu: y short')
        refused(log, lambda: P.swiglu(g, u, g, n), 'swiglu: y the same bytes as g (no in-place form)')
        refused(log, lambda: P.swiglu(big[:n], u, big[8:n + 8], n), 'swiglu: y partially overlapping g')
        T, k, D = 16, 10, 2560
        dn, w, shx, sg, out = z(T * k * D), z(T * k), z(T * D), z(T), z(T * D)
        accepted(log, lambda: P.moe_combine(dn, w, shx, sg, out, T, k, D), 'moe_combine')
        accepted(log, lambda: P.moe_combine(dn, w, shx, sg, shx, T, k, D), 'moe_combine in place over the shared expert\'s rows')
        refused(log, lambda: P.moe_combine(dn[:-1], w, shx, sg, out, T, k, D), 'moe_combine: down short')
        refused(log, lambda: P.moe_combine(dn, w[:-1], shx, sg, out, T, k, D), 'moe_combine: router weights short')
        refused(log, lambda: P.moe_combine(dn, w, shx, sg[:-1], out, T, k, D), 'moe_combine: gate logits short')
        refused(log, lambda: P.moe_combine(dn, w, shx, sg, out[:-1], T, k, D), 'moe_combine: y short')
        refused(log, lambda: P.moe_combine(dn, w, shx, sg, dn[:T * D], T, k, D), 'moe_combine: y on the per-slot outputs')
        refused(log, lambda: P.moe_combine(dn, w, shx, sg, out, T, 65, D), 'moe_combine: k > 64')
        nh, hd = 12, 256
        v, qf, ya = z(T * hd), z(T * nh * 2 * hd), z(T * nh * hd)
        accepted(log, lambda: P.attn_standin(v, qf, ya, T, nh, hd), 'attn_standin')
        refused(log, lambda: P.attn_standin(v[:-1], qf, ya, T, nh, hd), 'attn_standin: v short')
        refused(log, lambda: P.attn_standin(v, qf[:-1], ya, T, nh, hd), 'attn_standin: qf short')
        refused(log, lambda: P.attn_standin(v, qf, qf[:T * nh * hd], T, nh, hd), 'attn_standin: y on qf')
    finally:
        torch.cuda.device = real
    bad = []
    for T in (1, 5, 127, 128, 129, 256, 777, 1024, 1271, 2047, 2048):
        for m in range(1, 7):
            B = pt.micro_bounds(T, m)
            ok = B[0][0] == 0 and B[-1][1] == T and all(a < b for a, b in B) and all(B[i][1] == B[i + 1][0] for i in range(len(B) - 1))
            ok &= all(a % 128 == 0 for a, _ in B) and all((b - a) % 128 == 0 for a, b in B[:-1]) and len(B) <= m
            if not ok: bad.append((T, m, B))
    check(not bad, f'micro_bounds: every chunk 1 .. 2048 x m 1 .. 6 covered in order, pieces on the 128-row grid but the last, none empty' + (f' {bad[:3]}' if bad else ''))
    g = capacity.geometry(coder)
    for T in (1024, 2048):
        w = capacity.prefill_workspace_bytes(g, T, gdn_keys=10, moe_blocks=12); f = capacity.prefill_feed_bytes(g, T)
        gl = capacity.prefill_glue_bytes(g, T, gdn_keys=10); xb = capacity.prefill_xchg_bytes(g, T); ab = capacity.prefill_attn_bytes(g, T)
        need = capacity.prefill_need_mib(g, T, gdn_keys=10, moe_blocks=12)
        check(all(abs(need[c] - (w[c] + f[c] + gl[c] + xb[c] + ab[c]) / capacity.MIB - capacity.PREFILL_CODE_MIB[c] - capacity.PREFILL_STREAMS_MIB[c]) < 1e-6 for c in (0, 1)) and
              all(xb[c] >= 4 * T * g['D'] for c in (0, 1)),
              f'vram: chunk {T}: the exchange receive buffer ({xb[0] / capacity.MIB:.1f} MiB per card) in capacity.prefill_need_mib '
              f'({need[0]:.1f} / {need[1]:.1f} MiB)')
        if T == 2048: NUMBERS['vram'] = dict(xchg_mib=[x / capacity.MIB for x in xb], prefill_need_mib=need)


# ---- stage: kernels (one card) ----------------------------------------------------------------------------------------------
def acts(rng, n, scale=1.0):
    x = (rng.standard_normal(n) * scale).astype(np.float32)
    x[::997] = 0.0; x[1::991] = -0.0; x[2::983] *= 40.0; x[3::977] = np.float32(1e-39)   # zeros of both signs, outliers, subnormals
    return x


def stage_kernels(torch, dev, small):
    from caminus import prefill_tp as pt, k3
    P = pt.Tp(dev); rng = np.random.default_rng(5)
    put = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    D, k, nh, hd = 2560, 10, 12, 256
    TS = (4, 8) if small else (1, 4, 512, 777, 2048)
    for T in TS:
        n = T * D
        a, b = acts(rng, n), acts(rng, n)
        ga, gb = put(a), put(b); y = torch.zeros(n, device=dev)
        ref = (a + b).astype(np.float32)                                    # numpy float32 adds: IEEE round to nearest
        P.xchg_add(ga, gb, y, n); r0 = y.cpu().numpy()
        ia = ga.clone(); P.xchg_add(ia, gb, ia, n); r1 = ia.cpu().numpy()
        ib = gb.clone(); P.xchg_add(ga, ib, ib, n); r2 = ib.cpu().numpy()
        check(same(r0, ref) and same(r1, ref) and same(r2, ref), f'kernels T={T}: xchg_add == p0 + p1 bitwise (out of place, in place over p0, over p1)')
        # combine: slot-order sum emulated in float32 (no fma), shared expert 0 so y = acc + 0 * s
        dn = acts(rng, T * k * D, 0.3).reshape(T, k, D); w = rng.random((T, k)).astype(np.float32); w /= w.sum(1, keepdims=True)
        sg = (rng.standard_normal(T) * 3).astype(np.float32); sg[::7] = 30.0; sg[1::11] = -30.0
        acc = dn[:, 0, :] * w[:, 0:1]
        for j in range(1, k): acc = acc + dn[:, j, :] * w[:, j:j + 1]
        gdn, gw, gsg = put(dn.reshape(-1)), put(w.reshape(-1)), put(sg)
        zsh = torch.zeros(T * D, device=dev); yc = torch.zeros(T * D, device=dev)
        P.moe_combine(gdn, gw, zsh, gsg, yc, T, k, D)
        check(same(yc.cpu().numpy().reshape(T, D), (acc + np.float32(0.0)).astype(np.float32)),
              f'kernels T={T}: moe_combine\'s routed sum (slot order, separate roundings) == its float32 emulation bitwise')
        # the gated shared expert: == K3's OP_SHEXP_ADD per token on the same moe / shexp / gate logit
        shx = acts(rng, T * D, 0.5); gshx = put(shx)
        P.moe_combine(gdn, gw, gshx, gsg, yc, T, k, D)
        ip = gshx.clone(); P.moe_combine(gdn, gw, ip, gsg, ip, T, k, D)
        e = k3.Engine(module='k3_gguf_gsq', grid=k3.resident_grid('k3_gguf_gsq'), T=1)
        mb, sb, gb1, ob = (torch.zeros(D, device=dev) for _ in range(4))
        gb1 = torch.zeros(1, device=dev); e.shexp_add(mb, sb, gb1, ob); e.build()
        nt = min(T, 512); refk = torch.zeros(nt, D, device=dev); gacc = put(acc.reshape(T, D))
        for t in range(nt):
            mb.copy_(gacc[t]); sb.copy_(gshx[t * D:(t + 1) * D]); gb1.copy_(gsg[t:t + 1]); e.run(); refk[t].copy_(ob)
        check(same(yc[:nt * D], refk.reshape(-1)) and same(ip[:nt * D], refk.reshape(-1)),
              f'kernels T={T}: moe_combine (routed sum + gated shared expert, out of place and in place) == K3 OP_SHEXP_ADD per token bitwise ({nt} tokens)')
        # the attention stand-in == OP_GATE_MUL on each token's v row repeated over its 12 heads
        v = acts(rng, T * hd, 0.5); qf = acts(rng, T * nh * 2 * hd, 2.0)
        gv, gq = put(v), put(qf); ya = torch.zeros(T * nh * hd, device=dev)
        P.attn_standin(gv, gq, ya, T, nh, hd)
        e2 = k3.Engine(module='k3_gguf_gsq', grid=k3.resident_grid('k3_gguf_gsq'), T=1)
        ab, qb, gob = torch.zeros(nh * hd, device=dev), torch.zeros(nh * 2 * hd, device=dev), torch.zeros(nh * hd, device=dev)
        e2.gate_mul(ab, qb, 2 * hd, hd, gob, hd); e2.build()
        refa = torch.zeros(nt, nh * hd, device=dev)
        for t in range(nt):
            ab.copy_(gv[t * hd:(t + 1) * hd].repeat(nh)); qb.copy_(gq[t * nh * 2 * hd:(t + 1) * nh * 2 * hd]); e2.run(); refa[t].copy_(gob)
        check(same(ya[:nt * nh * hd], refa.reshape(-1)), f'kernels T={T}: attn_standin == K3 OP_GATE_MUL of the token\'s own v per head, bitwise ({nt} tokens)')
        # swiglu: ggml's silu with expf, against fp64 (no decode op writes it in fp32)
        g_, u_ = acts(rng, T * 640 * k // 4, 2.0), acts(rng, T * 640 * k // 4)
        gg, gu = put(g_), put(u_); ys = torch.zeros(g_.size, device=dev); P.swiglu(gg, gu, ys, g_.size)
        r64 = (g_.astype(np.float64) / (1.0 + np.exp(-g_.astype(np.float64)))) * u_.astype(np.float64)
        ysn = ys.cpu().numpy().astype(np.float64)
        rel = np.abs(ysn - r64) / np.maximum(np.abs(r64), 1e-30)
        rel[np.abs(r64) < 1e-30] = 0.0
        emu = (g_ / (np.float32(1.0) + np.exp(-g_)) * u_).astype(np.float32)
        nex = int((ys.cpu().numpy().view(np.int32) == emu.view(np.int32)).sum())
        check(float(rel.max()) < 4e-7, f'kernels T={T}: swiglu within 4e-7 relative of fp64 (max {rel.max():.2e}; {nex} / {g_.size} equal to numpy\'s float32 '
              f'expression: expf differs from numpy\'s in the last bit)')
    torch.cuda.synchronize(dev)


# ---- stage: stack (both cards) --------------------------------------------------------------------------------------------
def sha(t):
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()[:16]


def stage_stack(torch, coder, chunk, max_ctx, reps, ms, small, moe=None, moe_ab=True, q6k=None, q6k_ab=True):
    from caminus import capacity, prefill_tp as pt
    from caminus.flashnext_tp import FlashNextTP
    from caminus.prefill_feed import PrefillFeed
    from caminus.prefill_ops import RoutingError
    global Exchange_EV
    Exchange_EV = pt.Exchange.EVENTS
    SYSFS = ('/sys/class/drm/card1/device', '/sys/class/drm/card2/device')
    used = lambda: [int(open(f'{d}/mem_info_vram_used').read()) / capacity.MIB for d in SYSFS]
    print('compaction_proactiveness', comp(), flush=True)
    assert 'XTX' in torch.cuda.get_device_name(0) and 'XTX' not in torch.cuda.get_device_name(1), 'cuda:0 must be the XTX, cuda:1 the XT'
    vram = dict(idle=used())
    t0 = time.time()
    m = FlashNextTP(coder, ntok=4, max_ctx=max_ctx, vram_prefill=chunk)
    print(f'loaded in {time.time() - t0:.1f} s; vram plan (MiB per card):', {k: v for k, v in m.vram_plan.items() if k != 'rows'}, flush=True)
    for d in (0, 1): torch.cuda.synchronize(d)
    vram['loaded'] = used()
    # FlashNextTP.weights: the tapes' own tensors (references, nothing copied)
    nchk, missing = 0, []
    for c in (0, 1):
        kept = {t.data_ptr() for t in m.engines[c].keep if isinstance(t, torch.Tensor)}
        for (kind, name), v in m.weights.items():
            x = v[c]; t = x['w'] if isinstance(x, dict) else x
            nchk += 1
            if t.data_ptr() not in kept: missing.append(f'{kind} {name} card {c}')
    check(not missing, f'FlashNextTP.weights: {nchk} per-card entries ({len(m.weights)} tensors x 2 cards), each the tensor the card\'s tape reads '
          f'(its data_ptr in the engine\'s kept tensors)' + (f' (not found: {missing[:4]})' if missing else ''))
    devs = [torch.device('cuda:0'), torch.device('cuda:1')]
    feed = PrefillFeed(coder, devices=(devs[capacity.PREFILL_TABLE_CARD], devs[1 - capacity.PREFILL_TABLE_CARD]), max_chunk=chunk)
    S = pt.ChunkStack(m, feed, chunk, attn='standin', moe=moe, q6k=q6k)   # (S5's stack: the attention stand-in, so 8K from empty runs; the core is S6's)
    print(f'  routed experts: {S.moe} (ChunkStack.moe); dense Q6_K: {S.q6k} (ChunkStack.q6k)', flush=True)
    for d in (0, 1): torch.cuda.synchronize(d)
    vram['prefill'] = used()
    print(f'prefill buffers + modules: device-level +{vram["prefill"][0] - vram["loaded"][0]:.0f} / +{vram["prefill"][1] - vram["loaded"][1]:.0f} MiB '
          f'(capacity.prefill_need_mib: {m.vram_plan["prefill_mib"][0]:.0f} / {m.vram_plan["prefill_mib"][1]:.0f})', flush=True)
    toks = [int(v) for v in open(TOKENS).read().split()]
    nl, k = m.nl, m.k_used
    # the projection against the measurement BEFORE any test-only buffer exists (S2's tp check): one 256-token chunk in each
    # mode and one full chunk overlapped, no routing record; every buffer the prefill uses is then allocated and every stream
    # used (construction already used them once: capacity.PREFILL_STREAMS_MIB)
    t0 = time.perf_counter()
    for mode, mm, T_ in (('local', 1, 256), ('serial', 2, 256), ('overlap', 2, 256), ('overlap', 2, chunk)):
        m.reset(); S.run(feed.host_chunk(toks, 0, T_), m=mm, mode=mode); S.sync(); S.check_routing(f'vram run {mode}')
    print(f'  first chunks of the process (256 tokens local / serial / overlap, then {chunk} overlap m 2): {(time.perf_counter() - t0) * 1e3:.0f} ms', flush=True)
    vram['runs'] = used()
    vp = m.vram_plan; proj = [vp['fixed_mib'][c] + vp['ctx_mib'] + vp['reserve_mib'][c] + vp['drafters_mib'][c] + vp['prefill_mib'][c] for c in (0, 1)]
    peak = [max(vram[kk][c] for kk in ('loaded', 'prefill', 'runs')) for c in (0, 1)]
    for c, nm in ((0, 'XTX'), (1, 'XT')):
        check(vram['idle'][c] + proj[c] >= peak[c], f'stack VRAM projection >= measurement, {nm}: idle {vram["idle"][c]:.0f} + projected {proj[c]:.0f} '
              f'(fixed {vp["fixed_mib"][c]:.0f}, ctx {vp["ctx_mib"]:.0f}, prefill {vp["prefill_mib"][c]:.0f}) = {vram["idle"][c] + proj[c]:.0f} MiB vs the '
              f'measured device-level use {peak[c]:.0f} MiB after the prefill ran (margin {vram["idle"][c] + proj[c] - peak[c]:+.0f})')
    NUMBERS['vram_projection'] = dict(projected=proj, idle=vram['idle'], measured=peak)
    # the exchange's refusals (nothing queued)
    X = S.X; live0 = len(X._live)
    bad_calls = [('rows past the chunk', lambda: X.post([S.W[0].o_blk, S.W[1].o_blk], 0, chunk + 1)),
                 ('empty rows', lambda: X.post([S.W[0].o_blk, S.W[1].o_blk], 5, 5)),
                 ('a short partial view', lambda: X.post([S.W[0].o_blk[:chunk * m.D - 1], S.W[1].o_blk], 0, chunk)),
                 ('the partials on the wrong cards', lambda: X.post([S.W[1].o_blk, S.W[0].o_blk], 0, 16))]
    for what, fn in bad_calls:
        try: fn(); ok = False
        except ValueError: ok = len(X._live) == live0
        check(ok, f'stack: Exchange.post refuses {what}, nothing queued')
    hist = dict(ids=[[torch.zeros(chunk, k, dtype=torch.int32, device=devs[c]) for _ in range(nl)] for c in (0, 1)],
                wts=[[torch.zeros(chunk, k, device=devs[c]) for _ in range(nl)] for c in (0, 1)])

    def states():
        out = {}
        for key, t in m.state.items():
            if len(key) != 3: continue
            c, il, kind = key
            if kind == 'gdn': out[(c, il, kind)] = sha(t[:S.H[c] * S.S * S.S])
            elif kind in ('conv', 'ple'): out[(c, il, kind)] = sha(t)
        return out

    def after_chunk(tag, T):
        """the checks after every chunk (both cards synchronized): routing flags, feed flags, replicas"""
        S.sync()
        try: S.check_routing(tag); okr = True
        except RoutingError as e: okr = False; print('   ', e, flush=True)
        try: feed.check2(); okf = True
        except Exception as e: okf = False; print('   ', e, flush=True)
        r_ok = torch.equal(S.residual(0, T).cpu(), S.residual(1, T).cpu())
        rt_ok = all(torch.equal(hist['ids'][0][il][:T].cpu(), hist['ids'][1][il][:T].cpu()) and
                    torch.equal(hist['wts'][0][il][:T].cpu(), hist['wts'][1][il][:T].cpu()) for il in range(nl))
        ple_ok = all(torch.equal(m.state[(0, il, 'ple')].cpu(), m.state[(1, il, 'ple')].cpu()) for il in m.ple_layers)
        return okr and okf and r_ok and rt_ok and ple_ok, dict(routing=okr, feed=okf, residual=r_ok, router=rt_ok, ple=ple_ok)

    def prompt(bounds, mm, mode, check_each=True, tag=''):
        """reset, then the chunks [(p0, p1)] in mode / mm: per position the residual (card 0) and routing (both cards) on
        the host, the final states' hashes, the per-chunk check results"""
        m.reset()
        res, ids, wts, rep, tms = [], [[], []], [[], []], [], []
        for p0, p1 in bounds:
            hc = feed.host_chunk(toks, p0, p1); T = p1 - p0
            for d in (0, 1): torch.cuda.synchronize(d)
            t0 = time.perf_counter(); S.run(hc, m=mm, mode=mode, hist=hist); S.sync(); tms.append((time.perf_counter() - t0) * 1e3)
            ok, why = after_chunk(f'{tag} chunk {p0}..{p1}', T) if check_each else (True, {})
            rep.append((ok, why))
            res.append(S.residual(0, T).cpu())
            for c in (0, 1):
                ids[c].append(torch.stack([hist['ids'][c][il][:T].cpu() for il in range(nl)]))
                wts[c].append(torch.stack([hist['wts'][c][il][:T].cpu() for il in range(nl)]))
        return dict(res=torch.cat(res), ids=[torch.cat(x, 1) for x in ids], wts=[torch.cat(x, 1) for x in wts], st=states(), rep=rep, ms=tms)

    # -- a small chunk first ------------------------------------------------------------------------------------------------
    t0 = time.perf_counter(); first = prompt([(0, 256)], 1, 'serial', tag='first 256')
    print(f'  first chunk of the process (256 tokens, m 1 serial, with its checks): {(time.perf_counter() - t0) * 1e3:.0f} ms', flush=True)
    check(first['rep'][0][0], f'stack 256: replicas equal on both cards, routing / feed flags clear ({first["rep"][0][1]})')
    fin = bool(torch.isfinite(first['res']).all())
    check(fin, 'stack 256: the residual streams are finite after 48 layers')
    for mm, mode in ((2, 'overlap'), (2, 'serial')):
        r = prompt([(0, 256)], mm, mode, tag=f'256 m{mm} {mode}')
        check(r['rep'][0][0] and same(r['res'], first['res']) and r['st'] == first['st'] and all(same(r['ids'][c], first['ids'][c]) for c in (0, 1)),
              f'stack 256: m {mm} {mode} == m 1 serial bitwise (residual, routing, states), replicas equal')
    if small: return
    # -- G2b + replicas at the full chunk -----------------------------------------------------------------------------------
    C = chunk
    ref = prompt([(0, C)], 1, 'serial', tag='ref')
    print(f'  COLD: the first {C}-token chunk of the process (m 1 serial, the routing record on): {ref["ms"][0]:.1f} ms', flush=True)
    NUMBERS['cold_first_chunk_ms'] = ref['ms'][0]
    check(ref['rep'][0][0], f'stack {C}: m 1 serial: replicas equal, flags clear ({ref["rep"][0][1]})')
    check(bool(torch.isfinite(ref['res']).all()), f'stack {C}: the residual streams are finite after 48 layers')
    variants = [([(0, C)], mm, mode) for mm in ms for mode in ('overlap', 'serial') if (mm, mode) != (1, 'serial')]
    variants += [([(0, C // 2), (C // 2, C)], 2, 'overlap'), ([(0, 777), (777, C)], 2, 'overlap'), ([(0, 777), (777, C)], 1, 'serial')]
    g2b = {}
    for bounds, mm, mode in variants:
        tag = '+'.join(str(b - a) for a, b in bounds) + f' m{mm} {mode}'
        r = prompt(bounds, mm, mode, tag=tag)
        reps_ok = all(x[0] for x in r['rep'])
        dpos = int((bits(r['res'].view(C, -1)) != bits(ref['res'].view(C, -1))).any(1).sum())
        rt = all(same(r['ids'][c], ref['ids'][c]) and same(r['wts'][c], ref['wts'][c]) for c in (0, 1))
        st = r['st'] == ref['st']; g2b[tag] = dict(replicas=reps_ok, positions_differ=dpos, routing=rt, states=st)
        check(reps_ok and dpos == 0 and rt and st, f'stack {C} G2b {tag} == one chunk m 1 serial: residual at {C - dpos} / {C} positions, '
              f'routing {"==" if rt else "!="}, states {"==" if st else "!="}; replicas after each chunk {"equal" if reps_ok else "DIFFER"}')
    NUMBERS['g2b'] = g2b
    # -- the routed experts' other path on the same prompts: bitwise the reference ------------------------------------------
    # (moe_ab False, --no-moe-ab: the other path never runs in this process: this A/B, the A/B timing and the other path's
    # breakdown are skipped; everything else as with it. For the 11-04 SDMA fault's control runs: --moe mmq alone would still
    # launch the own path here, on both cards)
    base = S.moe; other = 'mmq' if base == 'own' else 'own'; mab = {}
    if not moe_ab: print(f'  routed experts: the other path ({other}) is not run (--no-moe-ab): its A/B, A/B timing and breakdown skipped', flush=True)
    for bounds, mm, mode in (([(0, C)], 1, 'serial'), ([(0, C)], 2, 'overlap'), ([(0, 777), (777, C)], 2, 'overlap')) if moe_ab else ():
        tag = f'moe {other} ' + '+'.join(str(b - a) for a, b in bounds) + f' m{mm} {mode}'
        S.moe = other
        try: r = prompt(bounds, mm, mode, tag=tag)
        finally: S.moe = base
        reps_ok = all(x[0] for x in r['rep'])
        dpos = int((bits(r['res'].view(C, -1)) != bits(ref['res'].view(C, -1))).any(1).sum())
        rt = all(same(r['ids'][c], ref['ids'][c]) and same(r['wts'][c], ref['wts'][c]) for c in (0, 1))
        st = r['st'] == ref['st']; mab[tag] = dict(replicas=reps_ok, positions_differ=dpos, routing=rt, states=st)
        check(reps_ok and dpos == 0 and rt and st, f'stack {C} {tag} == the {base} path\'s one chunk m 1 serial: residual at {C - dpos} / {C} positions, '
              f'routing {"==" if rt else "!="}, states {"==" if st else "!="}; replicas {"equal" if reps_ok else "DIFFER"}')
    NUMBERS['moe_paths'] = mab if moe_ab else f'skipped (--no-moe-ab: {other} not run)'
    # -- the dense Q6_K GEMMs' other paths (ChunkStack.q6k): 'own' and 'mmq' bitwise the reference; 'wmma' (a tolerance) bitwise
    # its own one chunk m 1 serial across micro-chunks and splits (G2b), with replicas, and its distance from the reference
    qbase = S.q6k; qothers = [p for p in pt.Q6K_PATHS if p != qbase] if q6k_ab else []
    if 'wmma' in qothers: S.alloc_q6k_buffers()      # (after the VRAM projection check above: the default path's buffers only there)
    qab = {}
    for p in qothers:
        exact = 'wmma' not in (p, qbase); own_ref = None
        for bounds, mm, mode in (([(0, C)], 1, 'serial'), ([(0, C)], 2, 'overlap'), ([(0, 777), (777, C)], 2, 'overlap')):
            tag = f'q6k {p} ' + '+'.join(str(b - a) for a, b in bounds) + f' m{mm} {mode}'
            S.q6k = p
            try: r = prompt(bounds, mm, mode, tag=tag)
            finally: S.q6k = qbase
            reps_ok = all(x[0] for x in r['rep'])
            cmp_ = ref if exact else (own_ref or r)
            dpos = int((bits(r['res'].view(C, -1)) != bits(cmp_['res'].view(C, -1))).any(1).sum())
            rt = all(same(r['ids'][c], cmp_['ids'][c]) and same(r['wts'][c], cmp_['wts'][c]) for c in (0, 1))
            st = r['st'] == cmp_['st']
            row = dict(replicas=reps_ok, positions_differ=dpos, routing=rt, states=st)
            if not exact:
                d = (r['res'].double() - ref['res'].double()).abs(); rel = float(d.max() / ref['res'].double().abs().max())
                ideq = float(np.mean([np.mean(r['ids'][c].numpy() == ref['ids'][c].numpy()) for c in (0, 1)]))
                row.update(vs_ref_rel_max=rel, vs_ref_mean_abs=float(d.mean()), vs_ref_ids_equal=ideq)
            qab[tag] = row
            what = f'the {qbase} path\'s one chunk m 1 serial' if exact else f'its own one chunk m 1 serial'
            check(reps_ok and dpos == 0 and rt and st, f'stack {C} {tag} == {what}: residual at {C - dpos} / {C} positions, routing {"==" if rt else "!="}, '
                  f'states {"==" if st else "!="}; replicas {"equal" if reps_ok else "DIFFER"}' +
                  ('' if exact else f'; vs the {qbase} reference: residual max |diff| / max |ref| {row["vs_ref_rel_max"]:.2e}, mean |diff| '
                                   f'{row["vs_ref_mean_abs"]:.2e}, router ids equal {100 * row["vs_ref_ids_equal"]:.2f} % (reported)'))
            if own_ref is None: own_ref = r
    NUMBERS['q6k_paths'] = qab if q6k_ab else 'skipped (--no-q6k-ab)'
    # -- timing ------------------------------------------------------------------------------------------------------------
    hc = feed.host_chunk(toks, 0, C)
    th = []
    for p0 in range(0, 4 * C, C):
        t0 = time.perf_counter(); feed.host_chunk(toks, p0, p0 + C); th.append((time.perf_counter() - t0) * 1e3)
    print(f'  host feed part (n-gram ids, dedupe, O_DIRECT PLE reads) per {C}-token chunk: {", ".join(f"{x:.1f}" for x in th)} ms '
          f'(runs on the feed\'s worker thread while the previous chunk computes; not in the timings below)', flush=True)
    NUMBERS['host_feed_ms'] = th
    from caminus.hip import hip
    fsrc = feed.buf2[0]['te_rows']; fbytes = C * feed.te_rb
    fdst = torch.zeros(fbytes, dtype=torch.uint8, device=devs[0]); fstream = torch.cuda.Stream(device=devs[1])
    def side_copy():
        """a feed-sized P2P copy (the next chunk's raw token_embd rows, XT -> XTX) issued when the XT's compute stream gets here"""
        e = torch.cuda.Event(); e.record(S.ks[1]); fstream.wait_event(e)
        pt._peer_copy(fdst.data_ptr(), 0, fsrc.data_ptr(), 1, fbytes, fstream)
    tv = [('local', mm, 'all') for mm in ms] + [('serial', mm, 'all') for mm in ms] + [('overlap', mm, 'all') for mm in ms if mm > 1]
    tv += [('overlap+feedcopy', mm, 'all') for mm in ms if mm > 1]
    tv += [(mode, mm, xe) for xe in ('ready', 'none') for mode, mm in (('serial', 1), ('overlap', 2))]
    tv += [('overlap-halfduplex', 2, 'ready')]
    times = {v: [] for v in tv}; enq = {v: [] for v in tv}; rep_bad = []
    def one(v):
        mode, mm, xe = v; m.reset(); S.X.events = xe; S.X.duplex = mode != 'overlap-halfduplex'
        for d in (0, 1): torch.cuda.synchronize(d)
        t0 = time.perf_counter()
        S.run(hc, m=mm, mode='overlap' if mode.startswith('overlap') else mode); t1 = time.perf_counter()
        S.sync(); t2 = time.perf_counter(); S.X.duplex = True
        return (t2 - t0) * 1e3, (t1 - t0) * 1e3
    orig_half = S._half
    def run_with_copy(v):
        mode, mm, xe = v; cnt = [0]
        half_n = nl * 2 * len(pt.micro_bounds(C, mm))      # _half calls per chunk: halves x pieces x 2 cards; the copy half way
        def hooked(c, kind, il, r0, n, inputs, hist_):
            cnt[0] += 1
            if cnt[0] == half_n: side_copy()
            return orig_half(c, kind, il, r0, n, inputs, hist_)
        S._half = hooked
        try: return one(v)
        finally: S._half = orig_half
    for r in range(reps):
        for v in tv:
            ms_, eq = run_with_copy(v) if v[0] == 'overlap+feedcopy' else one(v)
            times[v].append(ms_); enq[v].append(eq)
            ok, why = after_chunk(f'timing {v}', C)
            if v[0] != 'local' and not ok: rep_bad.append((v, why))
            elif v[0] == 'local' and not why['routing']: rep_bad.append((v, why))
    S.X.events = 'all'
    check(not rep_bad, f'stack timing runs: after each (outside the timed window) the routing / feed flags clear and, every run but local, the replicas '
          f'equal ({sum(len(x) for x in times.values())} runs)' + (f' {rep_bad[:3]}' if rep_bad else ''))
    tres = {}
    for v in tv:
        cold, warm = times[v][0], float(np.median(times[v][1:])) if len(times[v]) > 1 else times[v][0]
        tres[f'{v[0]} m{v[1]} ev:{v[2]}'] = dict(cold_ms=cold, warm_ms=warm, all_ms=times[v], enqueue_ms=enq[v], tok_s_warm=C / warm * 1e3)
        print(f'  TIME chunk {C} {v[0]:17s} m {v[1]} events {v[2]:5s}: cold {cold:7.1f} ms, warm (median of {len(times[v]) - 1}) {warm:7.1f} ms = '
              f'{C / warm * 1e3:6.0f} tok/s; host enqueue {np.median(enq[v]):6.1f} ms; runs {", ".join(f"{x:.1f}" for x in times[v])}', flush=True)
    NUMBERS['timing'] = tres
    # -- the routed experts' paths timed interleaved (overlap m 2: the production setting; local m 2: compute only) ------------
    paths = (base, other) if moe_ab else (base,)
    tab = {(p, mode): [] for p in paths for mode in ('overlap', 'local')}; bad_ab = []
    for r in range(reps + 1):
        for p, mode in tab:
            S.moe = p
            try: tab[(p, mode)].append(one((mode, 2, 'ready'))[0])
            finally: S.moe = base
            ok, why = after_chunk(f'moe A/B {p} {mode}', C)
            if not (ok if mode != 'local' else why['routing'] and why['feed']): bad_ab.append((p, mode, why))
    S.X.events = 'all'
    check(not bad_ab, f'stack MoE A/B timing runs: flags clear and (but local) replicas equal after each' + (f' {bad_ab[:3]}' if bad_ab else ''))
    tab_res = {}
    for (p, mode), v in tab.items():
        warm = float(np.median(v[1:])); tab_res[f'{p} {mode} m2'] = dict(warm_ms=warm, all_ms=v, tok_s=C / warm * 1e3)
        print(f'  TIME MoE path {p:4s} chunk {C} {mode:7s} m 2: warm (median of {len(v) - 1}) {warm:7.1f} ms = {C / warm * 1e3:6.0f} tok/s; '
              f'runs {", ".join(f"{x:.1f}" for x in v)}', flush=True)
    NUMBERS['moe_ab_timing'] = tab_res
    # -- the dense Q6_K paths timed interleaved (overlap m 2, local m 2), the routed experts at the stack's path ------------------
    qpaths = [qbase] + qothers
    tq = {(p, mode): [] for p in qpaths for mode in ('overlap', 'local')}; bad_q = []
    for r in range(reps + 1):
        for p, mode in tq:
            S.q6k = p
            try: tq[(p, mode)].append(one((mode, 2, 'ready'))[0])
            finally: S.q6k = qbase
            ok, why = after_chunk(f'q6k A/B {p} {mode}', C)
            if not (ok if mode != 'local' else why['routing'] and why['feed']): bad_q.append((p, mode, why))
    S.X.events = 'all'
    check(not bad_q, f'stack dense Q6_K A/B timing runs: flags clear and (but local) replicas equal after each' + (f' {bad_q[:3]}' if bad_q else ''))
    tq_res = {}
    for (p, mode), v in tq.items():
        warm = float(np.median(v[1:])); tq_res[f'{p} {mode} m2'] = dict(warm_ms=warm, all_ms=v, tok_s=C / warm * 1e3)
        print(f'  TIME dense Q6_K path {p:4s} chunk {C} {mode:7s} m 2: warm (median of {len(v) - 1}) {warm:7.1f} ms = {C / warm * 1e3:6.0f} tok/s; '
              f'runs {", ".join(f"{x:.1f}" for x in v)}', flush=True)
    NUMBERS['q6k_ab_timing'] = tq_res
    best_m = min((mm for mm in ms if mm > 1), key=lambda mm: tres[f'overlap m{mm} ev:all']['warm_ms'])
    best_ev = min(Exchange_EV, key=lambda e: tres.get(f'overlap m2 ev:{e}', {'warm_ms': 1e9})['warm_ms'])
    print(f'  best micro-chunk count (overlap, warm, events all): m {best_m}; best event policy at m 2: {best_ev}', flush=True)
    NUMBERS['best_m'] = best_m; NUMBERS['best_ev'] = best_ev
    S.X.events = 'ready'                                                # (the default; the policies are within noise)
    # -- 8K from empty: 4 chunks back to back ----------------------------------------------------------------------------
    h0 = feed.host_chunk(toks, 0, C)
    def eight(mm, mode, first_ready):
        """4 chunks back to back from a reset (the next chunk's host part on the feed's thread); first_ready: chunk 0's host
        part prepared before the window, as the single-chunk timings (the difference: that host part on the critical path)"""
        import concurrent.futures as cf
        m.reset()
        for d in (0, 1): torch.cuda.synchronize(d)
        t0 = time.perf_counter()
        if first_ready: fut = cf.Future(); fut.set_result(h0)
        else: fut = feed.host_pool.submit(feed.host_chunk, toks, 0, C)
        for p0 in range(0, 4 * C, C):
            h = fut.result()
            if p0 + C < 4 * C: fut = feed.host_pool.submit(feed.host_chunk, toks, p0 + C, p0 + 2 * C)
            S.run(h, m=mm, mode=mode)
        S.sync(); return (time.perf_counter() - t0) * 1e3
    ev = [(1, 'serial', False), (best_m, 'overlap', False), (best_m, 'overlap', True), (best_m, 'local', False)]
    t8 = {v: [] for v in ev}; bad8 = []
    for r in range(max(3, reps - 1)):
        for v in ev:
            t8[v].append(eight(*v))
            ok, why = after_chunk(f'8K {v}', C)                         # (outside the timed window)
            if not (ok if v[1] != 'local' else why['routing'] and why['feed']): bad8.append((v, why))
    check(not bad8, f'stack 8K (4 x {C}) timing runs: after each the routing / feed flags clear and (but local) the replicas equal after the last chunk'
          + (f' {bad8[:3]}' if bad8 else ''))
    r8 = {}
    for v in ev:
        med = float(np.median(t8[v])); tag = f'{v[1]} m{v[0]}' + (' chunk-0 host part prepared' if v[2] else '')
        r8[tag] = dict(ms=t8[v], median_ms=med, tok_s=4 * C / med * 1e3)
        print(f'  TIME 8K from empty (4 x {C}) {tag:40s} events {S.X.events}: median {med:7.1f} ms = {4 * C / med * 1e3:6.0f} tok/s; '
              f'runs {", ".join(f"{x:.1f}" for x in t8[v])}', flush=True)
    NUMBERS['8k'] = r8
    # one untimed 8K run with the checks after every chunk
    m.reset(); okc = []
    for p0 in range(0, 4 * C, C):
        S.run(feed.host_chunk(toks, p0, p0 + C), m=best_m, mode='overlap', hist=hist); okc.append(after_chunk(f'8K chunk {p0}', C))
    check(all(o for o, _ in okc), f'stack 8K (4 x {C}, m {best_m} overlap): after every chunk the replicas equal and the flags clear ({[w for _, w in okc]})')
    # -- breakdown ---------------------------------------------------------------------------------------------------------
    bd = {}
    for mode, mm in (('serial', 1), ('overlap', best_m), ('local', best_m), ('local', 1)):
        m.reset()
        for d in (0, 1): torch.cuda.synchronize(d)
        tl = pt.Timeline(S.ks); t0 = time.perf_counter(); S.run(hc, m=mm, mode=mode, timeline=tl); S.sync(); wall = (time.perf_counter() - t0) * 1e3
        tot = tl.totals(); bd[f'{mode} m{mm}'] = dict(wall_ms=wall, cards=tot)
        for c, nm in ((0, 'XTX'), (1, 'XT')):
            parts = tot[c]['parts']; groups = {}
            for lab, v in parts.items():
                if lab.startswith('mmq:'):
                    _, kind, fmt = lab.split(':'); key = f'dense MMQ {fmt}' if not (fmt == 'Q6_K' and S.q6k == 'own') else 'dense Q6_K (caminus.dense)'
                elif lab.startswith('q6k:'): key = 'dense Q6_K (bf16 copy + dequant + WMMA)'
                else: key = {'moe': 'MoE (group + quantize + mul_mat_id)', 'wmma': 'BF16 WMMA (hc down / up, PLE key / value)',
                             'bf16_mv': 'thin BF16 (inject, alpha / beta, shared gate)', 'glue': 'glue (norms, hc mean, conv, l2 / gates / norm, swiglu, combine, PLE)',
                             'gdn_step': 'GDN recurrence', 'router': 'router (WMMA logits + top-k)', 'attn_standin': 'attention stand-in',
                             'xchg_wait': 'exchange wait (exposed)', 'xchg_add': 'exchange sum', 'feed': 'device feed (+ its cross-card barrier)',
                             'hist': 'routing record (test)'}.get(lab, lab)
                groups[key] = groups.get(key, 0.0) + v
            print(f'  BREAKDOWN {mode} m {mm} events {S.X.events} {nm}: stream span {tot[c]["span"]:.1f} ms (host wall {wall:.1f} ms, instrumented: '
                  f'{len(tl.ev[c])} timing events on this card)', flush=True)
            for key, v in sorted(groups.items(), key=lambda kv: -kv[1]):
                print(f'      {key:62s} {v:8.2f} ms ({100 * v / max(tot[c]["span"], 1e-9):5.1f}%)', flush=True)
            bd[f'{mode} m{mm}'][f'groups_{nm}'] = groups
    # the exchange's share without the fine labels' event cost: coarse timelines (compute / exchange wait / exchange sum)
    for mode, mm in (('overlap', best_m), ('serial', 1), ('local', best_m)):
        m.reset()
        for d in (0, 1): torch.cuda.synchronize(d)
        tl = pt.Timeline(S.ks, coarse=True); t0 = time.perf_counter(); S.run(hc, m=mm, mode=mode, timeline=tl); S.sync(); wall = (time.perf_counter() - t0) * 1e3
        tot = tl.totals(); bd[f'coarse {mode} m{mm}'] = dict(wall_ms=wall, cards=tot)
        for c, nm in ((0, 'XTX'), (1, 'XT')):
            p_ = tot[c]['parts']
            print(f'  COARSE {mode:7s} m {mm} events {S.X.events} {nm}: span {tot[c]["span"]:6.1f} ms (host wall {wall:.1f}, {len(tl.ev[c])} events): '
                  f'compute {p_.get("compute", 0):6.1f}, exchange wait {p_.get("xchg_wait", 0):6.1f}, exchange sum {p_.get("xchg_add", 0):5.1f}, '
                  f'feed {p_.get("feed", 0):4.1f} ms', flush=True)
    NUMBERS['breakdown'] = bd
    # the MoE + glue time of each routed-expert path (local m 2, instrumented; the MMQ path's routed swiglu is in its glue)
    mg = {}
    for p in paths:
        S.moe = p
        try:
            m.reset()
            for d in (0, 1): torch.cuda.synchronize(d)
            tl = pt.Timeline(S.ks); S.run(hc, m=best_m, mode='local', timeline=tl); S.sync()
        finally: S.moe = base
        tot = tl.totals(); mg[p] = [dict(moe=tot[c]['parts'].get('moe', 0.0), glue=tot[c]['parts'].get('glue', 0.0), span=tot[c]['span']) for c in (0, 1)]
        for c, nm in ((0, 'XTX'), (1, 'XT')):
            x = mg[p][c]
            print(f'  MOE PATH {p:4s} local m {best_m} {nm}: MoE {x["moe"]:7.1f} ms + glue {x["glue"]:6.1f} ms = {x["moe"] + x["glue"]:7.1f} ms of a '
                  f'{x["span"]:.1f} ms stream span', flush=True)
    NUMBERS['moe_breakdown'] = mg
    # the dense Q6_K time of each path (local m 2, instrumented: its quantize or bf16 copy, dequantization and GEMM)
    qg = {}
    for p in qpaths:
        S.q6k = p
        try:
            m.reset()
            for d in (0, 1): torch.cuda.synchronize(d)
            tl = pt.Timeline(S.ks); S.run(hc, m=best_m, mode='local', timeline=tl); S.sync()
        finally: S.q6k = qbase
        tot = tl.totals(); qg[p] = [dict(q6k=sum(v for lab, v in tot[c]['parts'].items() if lab.endswith(':Q6_K') or lab.startswith('q6k:')),
                                         span=tot[c]['span']) for c in (0, 1)]
        for c, nm in ((0, 'XTX'), (1, 'XT')):
            print(f'  Q6K PATH {p:4s} local m {best_m} {nm}: dense Q6_K {qg[p][c]["q6k"]:7.1f} ms of a {qg[p][c]["span"]:.1f} ms stream span', flush=True)
    NUMBERS['q6k_breakdown'] = qg
    for d in (0, 1): torch.cuda.synchronize(d)
    vram['end'] = used(); vram['torch_peak_reserved'] = [torch.cuda.max_memory_reserved(d) / capacity.MIB for d in (0, 1)]
    print('VRAM (device-level MiB [XTX, XT]):', {kk: [round(x, 0) for x in vv] for kk, vv in vram.items()}, flush=True)
    NUMBERS['vram_stack'] = vram; NUMBERS['compaction'] = comp(); NUMBERS['chunk'] = C
    feed.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('coder'); ap.add_argument('stages', nargs='+')
    ap.add_argument('--small', action='store_true'); ap.add_argument('--chunk', type=int, default=2048)
    ap.add_argument('--max-ctx', type=int, default=2048); ap.add_argument('--reps', type=int, default=4)
    ap.add_argument('--ms', default='1,2,3,4'); ap.add_argument('--json', default=None)
    ap.add_argument('--moe', default=None, help='own | mmq: ChunkStack\'s routed-expert path (default prefill_tp.MOE_PATH); the other is the A/B')
    ap.add_argument('--q6k', default=None, help='own | wmma | mmq: ChunkStack\'s dense Q6_K path (default prefill_tp.Q6K_PATH); the others are the A/B')
    ap.add_argument('--no-q6k-ab', dest='q6k_ab', action='store_false', help='stack: never run the other dense Q6_K paths')
    ap.add_argument('--no-moe-ab', dest='moe_ab', action='store_false', help='stack: never run the other MoE path (its A/B, A/B timing and '
                    'breakdown skipped; everything else unchanged)')
    a = ap.parse_args()
    if 'cpu' in a.stages: stage_cpu(a.coder)
    if 'kernels' in a.stages:
        import torch
        dev = torch.device('cuda:0'); torch.cuda.set_device(dev)
        print('device', torch.cuda.get_device_name(dev), '| compaction_proactiveness', comp(), flush=True)
        stage_kernels(torch, dev, a.small)
    if 'stack' in a.stages:
        import torch
        stage_stack(torch, a.coder, a.chunk, a.max_ctx, a.reps, [int(x) for x in a.ms.split(',')], a.small, a.moe, a.moe_ab, a.q6k, a.q6k_ab)
    if a.json:
        json.dump(dict(fails=FAILS, numbers=NUMBERS), open(a.json, 'w'), indent=1, default=str)
    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'))
    sys.exit(0 if not FAILS else 1)


if __name__ == '__main__':
    main()
