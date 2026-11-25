"""The qwen35moe decode attention kernel (kernels/dec_q35m_attn.hip through caminus.engine.dec_q35m_attn) on one attention
layer's shapes (16 query heads over 2 kv heads, head dim 256), random q and cache rows, at list lengths n = pos + 1 that
sit on and beside every edge of its layout (a 16-key tile, a 64-key run, the last run of the default splits ns * 64,
1, ...), each with a flat (q ~ N(0, 0.1)) and a peaked (q ~ N(0, 8)) softmax, F16 and Q8_0 caches (--kv):
  - against float64 attention over the values the kernel reads (F16: the cache's halves; Q8: fp16(code * d)): max
    |difference| <= 1e-5 x max |V|, the bound tests/engine/test_q35_split_attn.py holds the split pair to. Its margin:
    hi + lo fp16 halves of Q and P carry ~22 bits, fp16 x fp16 products are exact and the sums fp32
  - Q8 against float64 over kv_get's fp32 values (code * d; what the split pair reads): within the per-case bound of
    the one deliberate rounding, each K / V value to fp16 (u = 2^-11 of it): u max |V| for V, and for K every score
    moved by at most delta = scale u max_k sum_d |q_d k_d| (natural units), so every p by a factor in
    [e^-2delta, e^2delta]: (e^2delta - 1) max |V|; plus the 1e-5 x max |V| above. The measured error is printed beside it
  - against the split pair itself (OP_ATTN_PART + OP_ATTN_COMB on dec_q35m / dec_q35m_kvq8, a tape of just that
    pair): F16 within 2e-5 x max |V| (both within 1e-5 of float64), Q8 within the bound above + 1e-5 x max |V|
  - the rows past the last cell filled with NaN (64 rows, every length): the output unchanged (no read past pos)
  - an exactly sized cache (F16 16384 rows = 32 MiB, Q8 32768 rows = 34 MiB, whole 2 MiB pages as the benchmark's
    caches are): its own hipMalloc of exactly rows x PB bytes, past torch's caching allocator (which can place a block
    inside a bigger segment it holds), wrapped as a tensor through DLPack; hipMemGetAddressRange must report that
    pointer and size. Read at n = rows and through the clamp (pos past its rows) with several split counts: the last
    row's last byte is the allocation's last byte, so a read past the last position faults here (a Q8 V over-read did,
    in a benchmark) when the next page is unmapped, as it was there
  - two launches bitwise equal (the counters re-armed, the merge deterministic); other split counts (1, 2, 3, 7,
    NS_MAX) within the float64 bound; pos = -1 writes zeros; pos past the cache's rows sets the error word and reads
    the clamped list (n = rows)
  tools/guard.sh LOG 600 python -I tests/engine/test_dec_q35m_attn.py [--kv f16,q8] [--lens ...]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>). Prints PASS / FAIL and RESULT."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np, torch
from caminus.device import capacity
from caminus.engine import k3
from caminus.engine.dec_q35m_attn import DecQ35mAttn, splits, NS_MAX
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # tests/exact_alloc.py
from exact_alloc import exact_alloc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--kv', default='f16,q8', help='cache formats, comma separated')
ap.add_argument('--lens', default=None, help='list lengths (default: the layout edges, then 1000 / 4097 / 16389 / 40000)')
A = ap.parse_args()
H, NKV, D, G = 16, 2, 256, 8
SCALE = 1.0 / 16.0
U16 = 2.0 ** -11
ns0 = splits(10 ** 9)                                       # (the most splits the default picks on this card)
LENS = ([int(v) for v in A.lens.split(',')] if A.lens else
        sorted({1, 2, 15, 16, 17, 63, 64, 65, 127, 128, 129, 1000, 4096, 4097, 64 * ns0 - 1, 64 * ns0, 64 * ns0 + 1,
                16389, 40000}))
ROWS = max(LENS) + 64
EXACT = {'f16': 16384, 'q8': 32768}                         # rows x PB = 32 / 34 MiB: a whole number of 2 MiB pages
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


def ref(q, n, kvv):
    """float64 attention of the 16 heads over cells 0 .. n - 1 of kvv [rows][K | V][kv head][D]: [H][D]"""
    qq = q.reshape(H, D); o = np.zeros((H, D))
    if n <= 0: return o
    for gk in range(NKV):
        s = kvv[:n, 0, gk] @ qq[gk * G:(gk + 1) * G].T * SCALE        # [n][G]
        p = np.exp(s - s.max(axis=0)); p /= p.sum(axis=0)
        o[gk * G:(gk + 1) * G] = p.T @ kvv[:n, 1, gk]
    return o


def k_bound(q, n, kv32):
    """the Q8 K rounding's output bound / max |V|: (e^2delta - 1), delta = scale u max_k sum_d |q_d k_d|"""
    qa = np.abs(q.reshape(H, D)); dl = 0.0
    for gk in range(NKV):
        dl = max(dl, float((np.abs(kv32[:n, 0, gk]) @ qa[gk * G:(gk + 1) * G].T).max()) * SCALE * U16)
    return float(np.expm1(2 * dl))


def make_cache(rng, rows, Q8):
    """random cache rows [rows][PB] as host bytes, with the float64 values the kernel reads ([rows][K | V][kv head][D];
    Q8: fp16(code * d)) and, for Q8, kv_get's fp32 code * d (else None)"""
    if Q8:
        from caminus import ref_kv
        lay = ref_kv.Q8KVLayout(NKV, D)
        raw = np.empty((rows, lay.row_bytes), dtype=np.uint8)
        for r0 in range(0, rows, 4096):
            r1 = min(rows, r0 + 4096)
            raw[r0:r1] = ref_kv.encode_q8_kv(rng.normal(0.0, 0.5, (r1 - r0, NKV, D)).astype(np.float32),
                                             rng.normal(0.0, 0.5, (r1 - r0, NKV, D)).astype(np.float32), lay)
        kd, vd = ref_kv.decode_q8_kv(raw, lay)
        kv32 = np.stack((kd, vd), axis=1); del kd, vd                    # [rows][K | V][kv head][D] fp32: code * d
        return raw.reshape(-1), kv32.astype(np.float16).astype(np.float64), kv32
    h = rng.normal(0.0, 0.5, rows * 2 * NKV * D).astype(np.float16)
    return h.view(np.uint8), h.reshape(rows, 2, NKV, D).astype(np.float64), None


for kv in A.kv.split(','):
    Q8 = kv == 'q8'
    pb = k3.kv_pos_bytes(NKV * D, Q8)
    need = ((2 * ROWS + EXACT[kv]) * pb + 4 * H * ((ROWS + 127) // 128) * k3.ATTN_RS * 4 + 64 * H * D * 4) / 2 ** 20 + 64
    capacity.check([0], [need], f'test_dec_q35m_attn ({need:.0f} MiB)')
    print(torch.cuda.get_device_name(0), f'| {kv} cache, {H} q heads / {NKV} kv heads x {D}, rows {ROWS}, default splits '
          f'{splits(ROWS)}, lengths {LENS}', flush=True)
    rng = np.random.default_rng(11)
    raw, kvk, kv32 = make_cache(rng, ROWS, Q8)
    kv64 = kv32.astype(np.float64) if Q8 else None
    cache = torch.from_numpy(raw).to('cuda')
    module = 'dec_q35m_kvq8' if Q8 else 'dec_q35m'
    vmax = float(np.abs(kvk[:, 1]).max())
    rowsv = cache.view(torch.uint8).view(ROWS, pb)
    att = DecQ35mAttn(Q8)
    out = torch.zeros(H * D, device='cuda'); pos_t = torch.zeros(1, dtype=torch.int32, device='cuda')

    def run_new(q, pos, ns=None):
        pos_t.fill_(pos); out.fill_(float('nan'))
        att(q, cache, pos_t, out, SCALE, ns=ns); torch.cuda.synchronize()
        return out.clone()

    def run_pair(q, pos):
        E = k3.Engine(module=module, grid=k3.resident_grid(module), T=1, limit_us=500_000)
        o = torch.zeros(H * D, device='cuda'); p = torch.full((1,), pos, dtype=torch.int32, device='cuda')
        E.attn_dec(q, cache, p, o, H, NKV, D, SCALE, q8=Q8, max_cells=ROWS)
        E.build(); E.run(check=True); torch.cuda.synchronize()
        return o

    e_k, e_32, e_pair, b_worst, pois, rep = 0.0, 0.0, 0.0, True, True, True
    for n in LENS:
        for sig in (0.1, 8.0):
            qh = (rng.standard_normal(H * D) * sig).astype(np.float32)
            q = torch.from_numpy(qh).to('cuda')
            pair = run_pair(q, n - 1).cpu().double().numpy().reshape(H, D)
            # the rows past the last cell poisoned (NaN scales / halves), restored after
            r1 = min(ROWS, n + 64); keep = rowsv[n:r1].clone(); rowsv[n:r1] = 0xff
            a = run_new(q, n - 1); a2 = run_new(q, n - 1)
            rowsv[n:r1] = keep
            rep &= torch.equal(a, a2)
            an = a.cpu().double().numpy().reshape(H, D)
            pois &= bool(np.isfinite(an).all())
            ek = float(np.abs(an - ref(qh.astype(np.float64), n, kvk)).max()) / vmax
            e_k = max(e_k, ek)
            if Q8:
                bnd = U16 + k_bound(qh.astype(np.float64), n, kv32) + 1e-5
                e32 = float(np.abs(an - ref(qh.astype(np.float64), n, kv64)).max()) / vmax
                ep = float(np.abs(an - pair).max()) / vmax
                e_32 = max(e_32, e32); e_pair = max(e_pair, ep)
                b_worst &= e32 <= bnd and ep <= bnd + 1e-5
                print(f'  n {n:6d} q ~ N(0, {sig}): vs float64 (kernel values) {ek:.2e}, vs float64 (code * d) {e32:.2e}, '
                      f'vs the pair {ep:.2e} (bound {bnd:.2e})', flush=True)
            else:
                ep = float(np.abs(an - pair).max()) / vmax
                e_pair = max(e_pair, ep)
                print(f'  n {n:6d} q ~ N(0, {sig}): vs float64 {ek:.2e}, vs the pair {ep:.2e}', flush=True)
    check(e_k <= 1e-5, f'{kv}: against float64 over the values the kernel reads: max |difference| {e_k:.2e} x max |V| '
                       f'(<= 1e-5), every length and both q scales')
    if Q8:
        check(b_worst, f'q8: against float64 over code * d (max {e_32:.2e}) and against the split pair (max {e_pair:.2e}) '
                       f'within each case\'s fp16-rounding bound (u + (e^2delta - 1) + 1e-5, x max |V|)')
    else:
        check(e_pair <= 2e-5, f'f16: against the split pair (dec_q35m): max |difference| {e_pair:.2e} x max |V| (<= 2e-5)')
    check(pois, f'{kv}: NaN in the 64 rows past the last cell never reaches the output (every length)')
    check(rep, f'{kv}: two launches bitwise equal (every length and q)')
    # other split counts
    nsw = [v for v in (1, 2, 3, 7, NS_MAX) if v <= NS_MAX]
    ens = 0.0
    for n in (4097, 16389):
        if n > ROWS: continue
        qh = (rng.standard_normal(H * D) * 8.0).astype(np.float32); q = torch.from_numpy(qh).to('cuda')
        rf = ref(qh.astype(np.float64), n, kvk)
        for v in nsw: ens = max(ens, float(np.abs(run_new(q, n - 1, ns=v).cpu().double().numpy().reshape(H, D) - rf).max()) / vmax)
    check(ens <= 1e-5, f'{kv}: split counts {nsw} at lengths 4097 / 16389: max |difference| to float64 {ens:.2e} x max |V| (<= 1e-5)')
    # the empty list and a position past the rows
    q = torch.from_numpy((rng.standard_normal(H * D)).astype(np.float32)).to('cuda')
    z = run_new(q, -1)
    check(bool((z == 0).all()) and not att.error(), f'{kv}: pos = -1 (no cell) writes zeros, no error')
    att.reset()
    a = run_new(q, ROWS + 5)
    e = float(np.abs(a.cpu().double().numpy().reshape(H, D) - ref(q.cpu().double().numpy(), ROWS, kvk)).max()) / vmax
    check(att.error() and e <= 1e-5, f'{kv}: pos past the cache\'s {ROWS} rows sets the error word and reads the clamped '
                                     f'list ({e:.2e} x max |V| to float64 at n = rows)')
    att.reset()
    del cache, rowsv, raw, kvk, kv32, kv64
    # the exactly sized cache: its own hipMalloc of rows x PB bytes, its last position's last byte the allocation's last
    rx = EXACT[kv]
    raw, kvk, _ = make_cache(rng, rx, Q8)
    cache, (base, size) = exact_alloc(rx * pb)
    cache.copy_(torch.from_numpy(raw)); torch.cuda.synchronize()
    exact = base == cache.data_ptr() and size == rx * pb == cache.numel() and cache.is_cuda
    check(exact, f'{kv}: the exact cache ({rx} rows x {pb} B = {rx * pb / 2 ** 20:g} MiB) is one allocation of exactly that size '
                 f'(hipMemGetAddressRange: {size} B from offset {base - cache.data_ptr()})')
    vmax = float(np.abs(kvk[:, 1]).max())
    ex = 0.0
    for v in (None, 1, 7, NS_MAX):
        qh = (rng.standard_normal(H * D) * 8.0).astype(np.float32); q = torch.from_numpy(qh).to('cuda')
        rf = ref(qh.astype(np.float64), rx, kvk)
        ex = max(ex, float(np.abs(run_new(q, rx - 1, ns=v).cpu().double().numpy().reshape(H, D) - rf).max()) / vmax)
        ex = max(ex, float(np.abs(run_new(q, rx + 5, ns=v).cpu().double().numpy().reshape(H, D) - rf).max()) / vmax)
    err_set = att.error()
    check(ex <= 1e-5 and err_set, f'{kv}: the exact cache read at n = rows and clamped from pos = rows + 5, splits default '
                                  f'({splits(rx)}) / 1 / 7 / {NS_MAX}: max |difference| to float64 {ex:.2e} x max |V| '
                                  f'(<= 1e-5), the clamp sets the error word, no fault')
    torch.cuda.synchronize(); del att, cache, raw, kvk; torch.cuda.empty_cache()

print('RESULT', 'PASS' if ok else 'FAIL', flush=True)
sys.exit(0 if ok else 1)
