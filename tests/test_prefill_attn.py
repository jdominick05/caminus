"""Gates of the batched prefill's attention core (docs/agentic/spec-prefill-cache.md M5 S6; caminus.prefill_attn,
kernels/prefill_attn.hip).

  python -I tests/test_prefill_attn.py CODER STAGE... [--small] [--json OUT]

Stages:
  cpu    CPU only, nothing launched (launch recorders stand in for the kernels): the launchers' refusals (dense attention
         at or past QSA_EXACT, rows outside the cache, short views, misaligned q, written buffers overlapping another; the
         in-place gate accepted) and the attention workspace in capacity.prefill_need_mib
  g1     GPU (cuda:0; HIP_VISIBLE_DEVICES=1 = the XT first), one card's attention shapes (12 query heads, 1 kv head, head dim
         256), the Coder's real q / k norm weights and RoPE parameters (layer 3), random q / gate / k / v rows: the chain
         k norm + RoPE -> Hadamard 256, v Hadamard 64, the q8_0 KV append, q norm + RoPE -> Hadamard 256, dense attention
         (the split pair), the output Hadamard 64 and the gate, against K3's own decode ops (caminus.k3.Engine on the
         Coder's trunk module k3_gguf_gsq, the same tape as FlashNextTP's attention layer, one token per pass, positions 0
         .. T - 1) bitwise: the rotated k / q rows, the KV cache bytes, the attention output, the gated output; OP_HC_AVG
         (the DFlash features) bitwise per token. T = 1 / 4 / 512 / 777 / 2048 (positions up to 2047; plus a chunk ending at
         2050, the last dense position). G2b: the same rows as chunks 1 + 4 + 507, 777 + 1271, 512 x 4, 1000 + 1048 and
         record blocks of 17 / 64 / 128 rows == one chunk, bitwise. Speed: the core of one attention layer at T = 2048 (the
         ops' ms, the longest single launch). --small: T = 4 / 37 only
Prints PASS / FAIL per check and RESULT PASS / FAIL; exit 0 = PASS."""
import argparse, contextlib, ctypes, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'python'))
from caminus import flashnext   # noqa: E402,F401  (puts gguf-py on the path)

EPS = 9.999999974752427e-07
FAILS, NUMBERS = [], {}
NH, HD = 12, 256


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)
    return ok


def bits(a):
    a = a.detach().cpu().numpy() if hasattr(a, 'cpu') else a
    return np.ascontiguousarray(a).view(np.uint8)


def same(a, b):
    return np.array_equal(bits(a), bits(b))


def comp():
    return open('/proc/sys/vm/compaction_proactiveness').read().strip()


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


def model_params(coder):
    """the Coder's RoPE parameters and layer 3's attention norm weights (fp32 numpy)"""
    from gguf import GGUFReader
    rd = GGUFReader(coder, 'r'); f = {x.name: x for x in rd.fields.values()}
    hp = lambda k: f[f'qwen4exp.{k}'].contents()
    nrot = int(hp('rope.dimension_count'))
    libm = ctypes.CDLL('libm.so.6'); libm.powf.restype = ctypes.c_float; libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]
    theta = libm.powf(float(hp('rope.freq_base')), -2.0 / nrot)
    T = {t.name: t for t in rd.tensors}
    w = lambda n: np.array(T[n].data, dtype=np.float32).reshape(-1).copy()
    return dict(nrot=nrot, theta=theta, qn=w('blk.3.attn_q_norm.weight'), kn=w('blk.3.attn_k_norm.weight'))


# ---- stage: cpu ---------------------------------------------------------------------------------------------------------
def stage_cpu(coder):
    import torch
    from caminus import prefill_attn as pa, capacity, k3
    real = torch.cuda.device
    torch.cuda.device = lambda d: contextlib.nullcontext()
    try:
        log = []; A = pa.Attn('cpu', module=Rec(log))
        z = lambda n, dt=torch.float32: torch.zeros(int(n), dtype=dt)
        T = 16; rows = 2560; pb = 2 * (HD // 32) * 34
        cache, err = z(rows * pb, torch.uint8), z(1, torch.int32)
        k, v = z(T * HD), z(T * HD)
        accepted(log, lambda: A.kv_append(k, HD, v, HD, HD, cache, 0, T, err), 'kv_append')
        accepted(log, lambda: A.kv_append(k, HD, v, HD, HD, cache, 2048, T, err), 'kv_append: rows 2048 .. 2063 past QSA_EXACT (the QSA prompt path writes them, S7)')
        refused(log, lambda: A.kv_append(k, HD, v, HD, HD, cache[:1000 * pb], 990, T, err), 'kv_append: rows past the cache (its view)', (k3.ContextBoundsError,))
        refused(log, lambda: A.kv_append(k, HD, v, HD, HD, cache, -1, T, err), 'kv_append: a negative position', (k3.ContextBoundsError,))
        refused(log, lambda: A.kv_append(k[:-1], HD, v, HD, HD, cache, 0, T, err), 'kv_append: k short (its view)')
        refused(log, lambda: A.kv_append(k, HD, v, HD, HD, cache[:-1], 0, T, err), 'kv_append: the cache not whole rows')
        big = z(2 * T * HD)
        refused(log, lambda: A.kv_append(big[:T * HD], HD, v, HD, HD, big.view(torch.uint8)[:T * pb], 0, T, err), 'kv_append: the cache rows overlapping k')
        refused(log, lambda: A.kv_append(k, HD, v, HD, HD, big.view(torch.uint8)[:T * pb], 0, T, big.view(torch.int32)[:1]), 'kv_append: err inside the cache')
        q, out = z(T * NH * HD), z(T * NH * HD); rec = z(pa.ROWS_PER_BLOCK * NH * pa.max_splits() * pa.ATTN_RS)
        accepted(log, lambda: A.attend(q, cache, 100, T, 1, 0.0625, out, rec, err), 'attend')
        refused(log, lambda: A.attend(q, cache, 2040, T, 1, 0.0625, out, rec, err), 'attend: rows reach QSA_EXACT', (k3.ContextBoundsError,))
        refused(log, lambda: A.attend(q, cache[:50 * pb], 40, T, 1, 0.0625, out, rec, err), 'attend: rows past the cache', (k3.ContextBoundsError,))
        refused(log, lambda: A.attend(q, cache, 0, T, 1, 0.0625, out, rec[:-1], err), 'attend: the records short of one block')
        refused(log, lambda: A.attend(q, cache, 0, T, 1, 0.0625, q, rec, err), 'attend: out the same bytes as q')
        refused(log, lambda: A.attend(q, cache, 0, T, 1, 0.0625, out[:-1], rec, err), 'attend: out short')
        qq = z(T * NH * HD + 4)
        refused(log, lambda: A.attend(qq[1:T * NH * HD + 1], cache, 0, T, 1, 0.0625, out, rec, err), 'attend: q not 16-byte aligned (float4 loads)')
        x = z(T * NH * 2 * HD); w = z(HD); y = z(T * NH * HD)
        accepted(log, lambda: A.qk_nr(x, 2 * HD, NH, HD, w, 64, 0.5, EPS, 0, y, T), 'qk_nr (q heads in [q | gate] pairs)')
        accepted(log, lambda: A.qk_nr(x[:(T * NH - 1) * 2 * HD + HD], 2 * HD, NH, HD, w, 64, 0.5, EPS, 0, y, T), 'qk_nr: x ending at the last head\'s q half (what it reads)')
        refused(log, lambda: A.qk_nr(x[:(T * NH - 1) * 2 * HD + HD - 1], 2 * HD, NH, HD, w, 64, 0.5, EPS, 0, y, T), 'qk_nr: x one short of the last head\'s q half')
        refused(log, lambda: A.qk_nr(x, 2 * HD, NH, HD, w, 64, 0.5, EPS, 0, x[:T * NH * HD], T), 'qk_nr: out on x (no in-place form)')
        refused(log, lambda: A.qk_nr(x, 2 * HD, NH, 600, w, 64, 0.5, EPS, 0, y, T), 'qk_nr: D > 512')
        accepted(log, lambda: A.gate_mul(out, x, 2 * HD, HD, out, T * NH * HD, HD), 'gate_mul in place (y = a)')
        refused(log, lambda: A.gate_mul(out, x, 2 * HD, HD, x[:T * NH * HD], T * NH * HD, HD), 'gate_mul: y on the gate')
        refused(log, lambda: A.gate_mul(big[:T * NH * HD // 2], x, 2 * HD, HD, big[4:4 + T * NH * HD // 2], T * NH * HD // 2, HD), 'gate_mul: y partially overlapping a')
        refused(log, lambda: A.gate_mul(out, x[:-1], 2 * HD, HD, out, T * NH * HD, HD), 'gate_mul: the gate short')
        res, f = z(T * 4 * 2560), z(T * 2560)
        accepted(log, lambda: A.hc_avg(res, f, T, 4, 2560), 'hc_avg')
        refused(log, lambda: A.hc_avg(res, f[:-1], T, 4, 2560), 'hc_avg: y short')
        refused(log, lambda: A.hc_avg(res, res[:T * 2560], T, 4, 2560), 'hc_avg: y on the residual')
        accepted(log, lambda: A.hadamard(out, 64, T * NH * HD // 64, 4), 'hadamard')
        refused(log, lambda: A.hadamard(out[:-1], 64, T * NH * HD // 64), 'hadamard: x short')
        refused(log, lambda: A.hadamard(out, 32, T * NH * HD // 32), 'hadamard: n 32')
    finally:
        torch.cuda.device = real
    g = capacity.geometry(coder)
    for T in (1024, 2048):
        a = capacity.prefill_attn_bytes(g, T, n_feat=5)
        need = capacity.prefill_need_mib(g, T, gdn_keys=10, moe_blocks=12, n_feat=5)
        base = capacity.prefill_need_mib(g, T, gdn_keys=10, moe_blocks=12, n_feat=0)
        lay = pa.attn_layout(T, NH, HD, g['idx_dim'])
        check(all(abs(need[c] - base[c] - (a[c] - capacity.prefill_attn_bytes(g, T, n_feat=0)[c]) / capacity.MIB) < 1e-6 for c in (0, 1))
              and a[0] > a[1] and lay['rec'] >= pa.ROWS_PER_BLOCK * NH * pa.max_splits() * pa.ATTN_RS * 4,
              f'vram: chunk {T}: the attention workspace in capacity.prefill_need_mib ({a[0] / capacity.MIB:.1f} / {a[1] / capacity.MIB:.1f} MiB '
              f'with the features on card {capacity.FEATURE_CARD}; records {lay["rec"] / capacity.MIB:.1f} MiB per card)')
        if T == 2048: NUMBERS['vram'] = dict(attn_mib=[x / capacity.MIB for x in a], prefill_need_mib=need)


# ---- stage: g1 (one card) -------------------------------------------------------------------------------------------------
class K3Chain:
    """FlashNextTP's attention-layer core on one card's shapes as a K3 tape (module k3_gguf_gsq, T = 1), one token per pass"""
    def __init__(self, torch, dev, P, rows):
        from caminus import k3
        self.torch, self.k3, self.P = torch, k3, P
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
        self.qf, self.kr, self.vr = z(NH * 2 * HD), z(HD), z(HD)
        self.qo, self.ko, self.att, self.gat = z(NH * HD), z(HD), z(NH * HD), z(NH * HD)
        self.pos = z(1, dt=torch.int32); self.rows = rows
        self.cache = z(rows * 2 * (HD // 32) * 34, dt=torch.uint8)
        self.qn, self.kn = torch.tensor(P['qn'], device=dev), torch.tensor(P['kn'], device=dev)
        e = k3.Engine(module='k3_gguf_gsq', grid=k3.resident_grid('k3_gguf_gsq'), T=1)
        kd = e.qk_nr(self.kr, HD, 1, HD, self.kn, self.pos, self.ko, P['nrot'], P['theta'], EPS)
        kd = e.hadamard(self.ko, HD, dep=kd); vd = e.hadamard(self.vr, 64)
        ap = e.also(e.kv_append(self.ko, self.vr, self.cache, self.pos, q8=True, dep=kd), vd)
        qd = e.qk_nr(self.qf, 2 * HD, NH, HD, self.qn, self.pos, self.qo, P['nrot'], P['theta'], EPS)
        qd = e.hadamard(self.qo, HD, dep=qd)
        a_ = e.also(e.attn_dec(self.qo, self.cache, self.pos, self.att, NH, 1, HD, 1.0 / np.sqrt(HD), q8=True, dep=ap,
                               max_cells=min(rows, 2051)), qd)
        a_ = e.hadamard(self.att, 64, rows_per_task=4, dep=a_)
        e.gate_mul(self.att, self.qf, 2 * HD, HD, self.gat, HD, dep=a_)
        self.e = e.build()
        self.hres, self.havg = z(4, 2560), z(2560)
        h = k3.Engine(module='k3_gguf_gsq', grid=k3.resident_grid('k3_gguf_gsq'), T=1); h.hc_avg(self.hres, self.havg); self.h = h.build()

    def run(self, qf, k, v, p0=0):
        """positions p0 .. p0 + T - 1 (the cache's rows below p0 as they are); returns ko, qo, att, gat [T, ...] (device)"""
        torch = self.torch; T = qf.shape[0]
        out = [torch.zeros(T, n, device=qf.device) for n in (HD, NH * HD, NH * HD, NH * HD)]
        for t in range(T):
            self.qf.copy_(qf[t].view(-1)); self.kr.copy_(k[t]); self.vr.copy_(v[t]); self.pos.fill_(p0 + t)
            self.e.run()
            for o, b in zip(out, (self.ko, self.qo, self.att, self.gat)): o[t].copy_(b)
        return out

    def hc_avg(self, res):
        torch = self.torch; T = res.shape[0]; out = torch.zeros(T, 2560, device=res.device)
        for t in range(T):
            self.hres.copy_(res[t]); self.h.run(); out[t].copy_(self.havg)
        return out


class PfChain:
    """the same chain on the prefill's kernels over a chunk of rows"""
    def __init__(self, torch, dev, P, rows, rpb=None):
        from caminus import prefill_attn as pa
        self.torch, self.pa, self.P, self.dev = torch, pa, P, dev
        self.A = pa.Attn(dev); self.rows = rows
        self.qn, self.kn = torch.tensor(P['qn'], device=dev), torch.tensor(P['kn'], device=dev)
        self.cache = torch.zeros(rows * 2 * (HD // 32) * 34, dtype=torch.uint8, device=dev)
        self.err = torch.zeros(1, dtype=torch.int32, device=dev)
        self.rec = torch.zeros(pa.ROWS_PER_BLOCK * 2 * NH * pa.max_splits() * pa.ATTN_RS, device=dev)   # (room for 128-row blocks)

    def run(self, qf, k, v, p0=0, rpb=None):
        torch, A, P = self.torch, self.A, self.P; T = qf.shape[0]; rpb = rpb or self.pa.ROWS_PER_BLOCK
        ko, qo, att = torch.zeros(T, HD, device=self.dev), torch.zeros(T, NH * HD, device=self.dev), torch.zeros(T, NH * HD, device=self.dev)
        vv = v.clone()
        A.qk_nr(k, HD, 1, HD, self.kn, P['nrot'], P['theta'], EPS, p0, ko, T); A.hadamard(ko, HD, T)
        A.hadamard(vv, 64, T * HD // 64)
        A.kv_append(ko, HD, vv, HD, HD, self.cache, p0, T, self.err)
        A.qk_nr(qf, 2 * HD, NH, HD, self.qn, P['nrot'], P['theta'], EPS, p0, qo, T); A.hadamard(qo, HD, T * NH)
        A.attend(qo, self.cache, p0, T, 1, 1.0 / np.sqrt(HD), att, self.rec, self.err, rows_per_block=rpb)
        A.hadamard(att, 64, T * NH * HD // 64, 4)
        gat = att.clone(); A.gate_mul(gat, qf, 2 * HD, HD, gat, T * NH * HD, HD)
        return [ko, qo, att, gat]


def stage_g1(torch, dev, coder, small):
    P = model_params(coder)
    print(f'RoPE n_rot {P["nrot"]}, theta_scale {P["theta"]!r}; device {torch.cuda.get_device_name(dev)}; compaction_proactiveness {comp()}', flush=True)
    rows = 2052
    rng = np.random.default_rng(5)
    TMAX = 37 if small else 2051
    qf = torch.tensor((rng.standard_normal((TMAX, NH, 2 * HD)) * 1.5).astype(np.float32), device=dev)
    k = torch.tensor((rng.standard_normal((TMAX, HD)) * 1.5).astype(np.float32), device=dev)
    v = torch.tensor(rng.standard_normal((TMAX, HD)).astype(np.float32), device=dev)
    # a few rows with one large component (a peaked softmax) and a few all-equal keys (ties in the scores)
    for t in range(3, TMAX, 97): k[t, 7] = 40.0
    for t in range(5, TMAX, 211): k[t] = 0.5
    K = K3Chain(torch, dev, P, rows)
    t0 = time.time(); ref = K.run(qf, k, v); ref_cache = K.cache.clone()
    print(f'  K3 reference: {TMAX} positions one token per pass, {time.time() - t0:.1f} s', flush=True)
    names = ('rotated k', 'rotated q', 'attention output (after its Hadamard)', 'gated output')
    pb = 2 * (HD // 32) * 34
    for T in ((4, 37) if small else (1, 4, 512, 777, 2048, 2051)):
        F = PfChain(torch, dev, P, rows)
        out = F.run(qf[:T], k[:T], v[:T]); torch.cuda.synchronize(dev)
        ok = [same(o, r[:T]) for o, r in zip(out, ref)]
        okc = same(F.cache[:T * pb], ref_cache[:T * pb]); oke = int(F.err.item()) == 0
        nd = [int((bits(o.view(T, -1)).reshape(T, -1) != bits(r[:T].view(T, -1)).reshape(T, -1)).any(1).sum()) for o, r in zip(out, ref)]
        check(all(ok) and okc and oke, f'g1 T={T}: positions 0 .. {T - 1} == K3 one token per pass, bitwise: ' +
              ', '.join(f'{n} {"==" if o else f"!= at {d} rows"}' for n, o, d in zip(names, ok, nd)) + f', KV cache rows {"==" if okc else "!="}, err {int(F.err.item())}')
    # G2b: chunk splits and record blocks
    T = 37 if small else 2048
    base = PfChain(torch, dev, P, rows); ob = base.run(qf[:T], k[:T], v[:T])
    splits = [[(0, 4), (4, 37)]] if small else [[(0, 1), (1, 5), (5, 512), (512, 2048)], [(0, 777), (777, 2048)], [(0, 512), (512, 1024), (1024, 1536), (1536, 2048)], [(0, 1000), (1000, 2048)]]
    for sp in splits:
        F = PfChain(torch, dev, P, rows); parts = [F.run(qf[a:b], k[a:b], v[a:b], p0=a) for a, b in sp]
        cat = [torch.cat([p[i] for p in parts]) for i in range(4)]
        ok = all(same(c, o) for c, o in zip(cat, ob)) and same(F.cache, base.cache)
        check(ok, f'g2b: chunks {" + ".join(str(b - a) for a, b in sp)} == one chunk of {T}, bitwise (every output, the KV cache)')
    for rpb in ((5, 17) if small else (17, 64, 128)):
        F = PfChain(torch, dev, P, rows); o = F.run(qf[:T], k[:T], v[:T], rpb=rpb)
        check(all(same(a, b) for a, b in zip(o, ob)), f'g2b: record blocks of {rpb} rows == {base.pa.ROWS_PER_BLOCK}, bitwise ({T} rows)')
    # OP_HC_AVG
    res = torch.tensor(rng.standard_normal((64, 4, 2560)).astype(np.float32), device=dev)
    ra = K.hc_avg(res); pf = torch.zeros(64, 3000, device=dev); base.A.hc_avg(res, pf, 64, 4, 2560, ldo=3000)
    check(same(pf[:, :2560], ra), 'g1 hc_avg (the DFlash features): 64 tokens == K3 OP_HC_AVG per token, bitwise (row stride 3000)')
    if small: return
    # speed: one attention layer's core at T = 2048 from position 0 (and at 1024 .. 2047, a second chunk)
    F = PfChain(torch, dev, P, rows)
    def timed(fn, reps=5):
        ts = []
        for _ in range(reps):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); fn(); e1.record(); torch.cuda.synchronize(dev); ts.append(e0.elapsed_time(e1))
        return sorted(ts)[len(ts) // 2]
    A = F.A; T = 2048
    ko, qo, att = torch.zeros(T, HD, device=dev), torch.zeros(T, NH * HD, device=dev), torch.zeros(T, NH * HD, device=dev)
    vv = v[:T].clone()
    ms = dict(k_nr=timed(lambda: A.qk_nr(k[:T], HD, 1, HD, F.kn, P['nrot'], P['theta'], EPS, 0, ko, T)),
              q_nr=timed(lambda: A.qk_nr(qf[:T], 2 * HD, NH, HD, F.qn, P['nrot'], P['theta'], EPS, 0, qo, T)),
              had=timed(lambda: (A.hadamard(ko, HD, T), A.hadamard(qo, HD, T * NH), A.hadamard(vv, 64, T * HD // 64))),
              kv_append=timed(lambda: A.kv_append(ko, HD, vv, HD, HD, F.cache, 0, T, F.err)),
              attend=timed(lambda: A.attend(qo, F.cache, 0, T, 1, 0.0625, att, F.rec, F.err)),
              attend_128=timed(lambda: A.attend(qo, F.cache, 0, T, 1, 0.0625, att, F.rec, F.err, rows_per_block=128)),
              out_had_gate=timed(lambda: (A.hadamard(att, 64, T * NH * HD // 64, 4), A.gate_mul(att, qf[:T], 2 * HD, HD, att, T * NH * HD, HD))))
    # the longest single launch: the last record block's part (65 splits x 64 rows)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    longest = []
    for _ in range(5):
        e0.record(); A.attend(qo[-64:], F.cache, T - 64, 64, 1, 0.0625, att[-64:], F.rec, F.err); e1.record(); torch.cuda.synchronize(dev)
        longest.append(e0.elapsed_time(e1))
    core = ms['k_nr'] + ms['q_nr'] + ms['had'] + ms['kv_append'] + ms['attend'] + ms['out_had_gate']
    print(f'  speed T=2048 (ms, median of 5): ' + ', '.join(f'{n} {v:.3f}' for n, v in ms.items()) + f'; core per layer {core:.2f}, '
          f'x 12 attention layers {12 * core:.1f} ms per chunk; the last 64-row block (part + comb, 2 launches) {sorted(longest)[2]:.3f} ms', flush=True)
    check(sorted(longest)[2] < 50.0, f'launch time: the largest attention launch pair {sorted(longest)[2]:.3f} ms < 50 ms')
    NUMBERS['speed_ms'] = dict(ms, core=core, chunk_12_layers=12 * core, last_block=sorted(longest)[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('coder'); ap.add_argument('stages', nargs='+')
    ap.add_argument('--small', action='store_true'); ap.add_argument('--json', default=None)
    a = ap.parse_args()
    if 'cpu' in a.stages: stage_cpu(a.coder)
    if 'g1' in a.stages:
        import torch
        dev = torch.device('cuda:0'); torch.cuda.set_device(dev)
        stage_g1(torch, dev, a.coder, a.small)
    if a.json: json.dump(dict(fails=FAILS, numbers=NUMBERS), open(a.json, 'w'), indent=1, default=str)
    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'))
    sys.exit(0 if not FAILS else 1)


if __name__ == '__main__':
    main()
