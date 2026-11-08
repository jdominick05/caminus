"""Safety gates of the batched prefill (docs/AGENTIC-PLAN.md wf_af0b9af2-d20 "Open safety items"; impl:prefill-safety):
the routing ids that reach ggml's mm_ids_helper, the buffers' view extents and overlaps, and the VRAM projection's split.

  python -I tests/prefill/test_prefill_safety.py CODER STAGE...      (CODER: the Coder GGUF, metadata only)

Stages:
  cpu   CPU only, nothing launched (HIP_VISIBLE_DEVICES= is fine): the launchers' host refusals on CPU tensors through
        launch recorders (a refused call records no launch; the positive controls on the same buffers record one):
        - mmq / prefill_ops: a short VIEW inside a Workspace / GlueWorkspace block is refused (the old storage-end
          check, mmq.extent, would have let it through: shown on the same buffers);
        - prefill_ops: partially aliased buffers refused for every glue op (rms_mul, ple_gate, hc_mean, add_sum, ...),
          the exact in-place forms each kernel is written for accepted;
        - capacity: prefill_need_mib / budget take the engine's TP split (no default), FlashNextTP._check_vram passes its
          own gdn_keys / moe_blocks;
        - the old path was unsafe, by CPU EMULATION ONLY of ggml's mm_ids_helper<10> (kernels/ggml_mmq/mmid.cuh): ids out
          of range or repeated leave table entries unwritten, and the index each later kernel would compute from such a
          stale word lands outside its buffer (host arithmetic, nothing launched); the clamp's rule (emulated) makes
          every case complete.
  gpu   one card (cuda:0; run with HIP_VISIBLE_DEVICES=1 = the XT): the clamp from its DEVICE OUTPUTS. Bad values go
        only into the clamped path: NaN / inf logits into pf_router, out-of-range / duplicate ids into mmq.group (whose
        k3_mmq_ids_clamp runs before mm_ids_helper on the same stream; the unclamped helper is never launched on bad ids).
        Checked: every output id in range and distinct per token, untouched tokens bitwise unchanged, the clamp's choices
        == its CPU rule, the flag bits, the helper's tables complete and == the CPU emulation on the clamped ids, MMQ
        mul_mat_id on the clamped groups writes every output column (dst prefilled with NaN) and equals the same run on
        ids that were valid from the start, and check_routing / the workspaces refuse on the flag.
Prints PASS / FAIL per check and RESULT PASS / FAIL; exit 0 = PASS."""
import argparse, contextlib, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'python'))
FAILS = []
INT_MAX, INT_MIN = 2 ** 31 - 1, -2 ** 31
NEXP, KU = 256, 10


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)
    return ok


# ---- CPU emulations (the reference for the device outputs, and the only place bad ids meet mm_ids_helper) ---------------
def emu_mm_ids_helper(ids, n_exp, per_slot, fill):
    """ggml's mm_ids_helper<10> as mmq.group launches it (kernels/ggml_mmq/mmid.cuh, the n_expert_used_template = 10 path:
    one 32-lane wave per expert, 16 lanes per token, 2 tokens per step; nchannels_y = sis1 = k (per_slot) or 1,
    write_inverse = not per_slot), on host int32 ids [ntok][k]. The output tables start filled with `fill` (what the
    buffers held before: torch.empty garbage, or 0 after a zeroing); entries the kernel never writes keep it. Per expert e:
    nex_prev = the number of slots of ALL tokens whose id < e (warp sum of every lane's count), the tokens holding e
    compacted in token order (one column per token even when two of its slots hold e: both lanes store to the same LDS
    word; here the higher slot wins, the hardware's order is unspecified, either way one slot is lost). Returns
    (ids_src1, ids_dst, bounds) as int64 arrays"""
    ids = np.asarray(ids, dtype=np.int64); ntok, k = ids.shape
    assert k == 10
    src1 = np.full(ntok * k, fill, np.int64); dst = np.full(ntok * k + 127, fill, np.int64); bounds = np.full(n_exp + 1, fill, np.int64)
    for e in range(n_exp):
        nex_prev = int((ids < e).sum())
        hit = ids == e
        toks = np.nonzero(hit.any(axis=1))[0]                                    # token order
        slot = np.array([np.nonzero(hit[t])[0].max() for t in toks], dtype=np.int64)   # (two lanes, one LDS word: the higher kept)
        col = nex_prev + np.arange(len(toks))
        dst[col] = toks * k + slot
        if per_slot: src1[col] = toks * k + slot % k
        else: src1[toks * k + slot] = col
        bounds[e] = nex_prev
        if e == n_exp - 1: bounds[n_exp] = nex_prev + len(toks)
    return src1, dst, bounds


def emu_clamp(ids, n_exp):
    """k3_mmq_ids_clamp's rule on host ids [ntok][k]: slot by slot, an id kept if in [0, n_exp) and not held by an earlier
    slot of its token, else replaced by the lowest expert the token does not hold yet; flags 2 (range) / 4 (duplicate of
    an earlier in-range slot). Returns (clamped ids, flags, tokens changed)"""
    ids = np.asarray(ids, dtype=np.int64); out = ids.copy(); ntok, k = ids.shape; flags = 0
    rng_bad = (ids < 0) | (ids >= n_exp)
    srt = np.sort(np.where(rng_bad, -1 - np.arange(k)[None, :], ids), axis=1)
    dup = (srt[:, 1:] == srt[:, :-1]).any(axis=1)
    cand = np.nonzero(rng_bad.any(axis=1) | dup)[0]
    changed = []
    for t in cand:
        v = []
        for j in range(k):
            x = int(ids[t, j]); ok = 0 <= x < n_exp
            if not ok: flags |= 2
            elif x in v: ok = False; flags |= 4
            if not ok: x = min(e for e in range(j + 1) if e not in v)
            v.append(x)
        out[t] = v
        if list(ids[t]) != v: changed.append(int(t))
    return out.astype(np.int32), flags, changed


def valid_routing(rng, T, nexp=NEXP, k=KU, skew=1.1):
    p = 1.0 / np.arange(1, nexp + 1) ** skew; p = p[rng.permutation(nexp)]; p /= p.sum()
    g = -np.log(-np.log(rng.random((T, nexp))))
    return np.argsort(-(np.log(p)[None, :] + g), axis=1)[:, :k].astype(np.int32)


def bad_routing(rng, T, kinds=('range', 'dup')):
    """valid ids with injected faults: {token: description}"""
    ids = valid_routing(rng, T); what = {}
    def at(t, j, v, d): ids[t, j] = v; what[t] = d
    if 'range' in kinds:
        at(3, 2, NEXP, 'id = n_exp'); at(7, 0, -1, 'id = -1'); at(11, 9, INT_MIN, 'id = INT_MIN'); at(12, 5, INT_MAX, 'id = INT_MAX')
        at(13, 4, NEXP + 44, 'id = n_exp + 44')
        ids[T - 1, :] = -5; what[T - 1] = 'all 10 ids = -5'
    if 'dup' in kinds:
        at(5, 4, ids[5, 1], 'slots 1 and 4 equal')
        ids[17, :] = ids[17, 0]; what[17] = 'all 10 ids equal'
        ids[19, 2] = ids[19, 6]; ids[19, 7] = ids[19, 6]; what[19] = 'three slots equal'
        ids[23, 3] = 0; ids[23, 8] = 0; what[23] = 'two slots = expert 0'
    if 'range' in kinds and 'dup' in kinds:
        ids[29, 1] = ids[29, 0]; ids[29, 2] = NEXP + 1; what[29] = 'a duplicate and an out-of-range id'
    return ids, what


# ---- stage: cpu --------------------------------------------------------------------------------------------------------
class Rec:
    """a module / kernel stand-in: attribute access returns a recording callable (no device, nothing runs)"""
    def __init__(self, log): self.log = log

    def __getattr__(self, name):
        if name.startswith('__'): raise AttributeError(name)
        return lambda *a, **kw: self.log.append(name)


def refused(log, fn, what, expect=ValueError):
    """fn() must raise expect before any launch is recorded"""
    n0 = len(log)
    try:
        fn()
    except expect as e:
        return check(len(log) == n0, f'refused, nothing launched: {what} ({str(e)[:110]})')
    except Exception as e:
        return check(False, f'{what}: raised {type(e).__name__} {e} instead of {expect.__name__}')
    return check(False, f'{what}: NOT refused ({len(log) - n0} launches recorded)')


def accepted(log, fn, what):
    n0 = len(log)
    try:
        fn()
    except Exception as e:
        return check(False, f'positive control {what}: refused ({type(e).__name__}: {str(e)[:160]})')
    return check(len(log) > n0, f'positive control (same buffers, valid use): {what} accepted ({len(log) - n0} launch(es) recorded, none run)')


def stage_cpu(coder):
    import torch
    real = torch.cuda.device
    torch.cuda.device = lambda d: contextlib.nullcontext()        # (nothing is launched here: the recorders stand in)
    try:
        _stage_cpu(coder)
    finally:
        torch.cuda.device = real                                  # (a gpu stage after this one gets the real context manager)


def _stage_cpu(coder):
    import torch
    from caminus.device import capacity
    from caminus.prefill import mmq, glue as po
    cpu = torch.device('cpu'); log = []
    g = capacity.geometry(coder)
    T = 512
    # -- the launchers on CPU tensors, kernels replaced by recorders --------------------------------------------------------
    M = mmq.MMQ.__new__(mmq.MMQ); M.dev = cpu
    M.cfg = {('Q8_0', J): dict(nthreads=256, I=128, nbytes=4096, layout=mmq.D4) for J in (16, 32, 48, 64, 96, 128)}
    M.kmm = {k: Rec(log).mm for k in M.cfg}; M.kq = {(l, s): Rec(log).quantize for l in (0, 1) for s in (False, True)}
    M.kids = {10: Rec(log).mm_ids_helper}; M.kclamp = Rec(log).ids_clamp
    M._launch = lambda k, *a, **kw: k(*a, **kw)
    G = po.Glue.__new__(po.Glue); G.dev = cpu; G.router_mode = po.ROUTER_LOGITS; G.wmma_out = po.WMMA_OUT; G.m = Rec(log)
    G._go = lambda fn, n, *a, threads=256: log.append(fn)
    d0 = mmq.card_dims(g, 0); W = mmq.Workspace(cpu, T, d0, check=False)
    GW = po.GlueWorkspace(cpu, T, po.glue_dims(g, 0), check=False)
    print(f'(CPU workspaces: GEMM {W.nbytes / 2**20:.1f} MiB, glue {GW.nbytes / 2**20:.1f} MiB at T = {T}, card 0 shapes; the views '
          'are slices of one block each, as on the device)', flush=True)
    rows, K = 256, 2560; fmt = 'Q8_0'; rb = mmq.row_bytes(fmt, K)
    w = torch.zeros(rows * rb, dtype=torch.uint8)
    # item 2: short views inside the Workspace block (the old check measured to the end of the block's storage)
    dst_short = W.o_qkv[:rows * T - 1]
    old_ok = mmq.extent(dst_short) >= 4 * rows * T
    check(old_ok, f'(before) the old storage-end check would pass a dst view 1 element short inside the Workspace block '
          f'(extent {mmq.extent(dst_short)} B >= {4 * rows * T} B needed)')
    refused(log, lambda: M.mm(fmt, w, rows, K, W.y_d4, T, dst_short), 'mmq.mm with a dst view 1 element short (o_qkv[:rows*T-1])')
    y_short = W.y_d4[:mmq.y_bytes(K, T) - 1]
    check(mmq.extent(y_short) >= mmq.y_bytes(K, T), '(before) the old check would pass a y view 1 byte short inside the block')
    refused(log, lambda: M.mm(fmt, w, rows, K, y_short, T, W.o_qkv), 'mmq.mm with a y view 1 byte short')
    refused(log, lambda: M.quantize(W.o_blk[:T * K - 1], K, T, W.y_d4, mmq.D4), 'mmq.quantize with an x view 1 element short')
    refused(log, lambda: M.quantize(W.x_moe_down, d0['ffc'], T * KU, W.y_down[:100], mmq.D4, ids=W.ids_fwd), 'mmq.quantize (ids) with a y view of 100 B')
    refused(log, lambda: M.quantize_scatter(W.o_blk, K, T, W.ids_inv, KU, W.y_moe[:mmq.y_bytes(K, T * KU) - 1], mmq.D4),
            'mmq.quantize_scatter with a y view 1 byte short')
    we = torch.zeros(NEXP * 128 * mmq.row_bytes(fmt, K), dtype=torch.uint8)
    refused(log, lambda: M.mm_id(fmt, we, NEXP, 128, K, W.y_moe, T, KU, W.ids_dst, W.bounds, W.o_gate[:128 * T * KU - 1]),
            'mmq.mm_id with a dst view 1 element short')
    refused(log, lambda: M.group(W.ids_inv[:T * KU - 1], T, KU, NEXP, W.ids_fwd, W.ids_dst, W.bounds, True, W.err),
            'mmq.group with an ids view 1 element short')
    # positive controls on the same buffers
    accepted(log, lambda: M.mm(fmt, w, rows, K, W.y_d4, T, W.o_qkv), 'mmq.mm (full views)')
    accepted(log, lambda: M.quantize(W.o_blk, K, T, W.y_d4, mmq.D4), 'mmq.quantize')
    accepted(log, lambda: M.mm_id(fmt, we, NEXP, 128, K, W.y_moe, T, KU, W.ids_dst, W.bounds, W.o_gate), 'mmq.mm_id')
    # the weights: a view one row short, and MMQ's over-read past the last row of a K = 384 shard (mmq.weight_bytes, the
    # one check against the allocation rather than the view; caminus.prefill.moe calls the same function)
    refused(log, lambda: M.mm(fmt, w[:(rows - 1) * rb], rows, K, W.y_d4, T, W.o_qkv), 'mmq.mm with a w view 1 row short')
    K3c = 384; rb3 = mmq.row_bytes(fmt, K3c)
    w3 = torch.zeros(rows * rb3, dtype=torch.uint8)
    refused(log, lambda: M.mm(fmt, w3, rows, K3c, W.y_d4, T, W.o_qkv), f'mmq.mm at K {K3c} with no over-read tail ({mmq.overread_bytes(fmt, K3c)} B)')
    w3p = torch.zeros(rows * rb3 + mmq.pad_bytes(fmt, K3c), dtype=torch.uint8)[:rows * rb3]
    accepted(log, lambda: M.mm(fmt, w3p, rows, K3c, W.y_d4, T, W.o_qkv), f'mmq.mm at K {K3c} with the zeroed tail past its view (pad_bytes)')
    ids_t = torch.from_numpy(valid_routing(np.random.default_rng(1), T))
    n0 = len(log); M.group(ids_t, T, KU, NEXP, W.ids_fwd, W.ids_dst, W.bounds, True, W.err)
    check(log[n0:] == ['ids_clamp', 'mm_ids_helper'], f'mmq.group launches the clamp before mm_ids_helper, every call (recorded {log[n0:]})')
    # overlaps in mmq
    refused(log, lambda: M.mm(fmt, w, rows, K, W.y_d4, T, W.y_d4.view(torch.float32)), 'mmq.mm with dst on y')
    refused(log, lambda: M.group(ids_t, T, KU, NEXP, ids_t.view(-1), W.ids_dst, W.bounds, True, W.err), 'mmq.group with ids_src1 on ids')
    refused(log, lambda: M.group(ids_t, T, KU, NEXP, W.ids_fwd, W.ids_dst, W.bounds, True, W.ids_dst[5:6]), 'mmq.group with err inside ids_dst')
    refused(log, lambda: M.group(ids_t, T, KU, 8, W.ids_fwd, W.ids_dst, W.bounds, True, W.err), 'mmq.group with n_exp 8 < k 10')
    # the glue: short views
    f32 = lambda n, off=0: GW.block[off:off + 4 * n].view(torch.float32)
    R = T * 4
    refused(log, lambda: G.rms_mul(GW.res, torch.zeros(4, 2560), GW.gate[:R * 2560 - 1], R, 1e-6), 'rms_mul with y 1 element short (inside the glue block)')
    refused(log, lambda: G.router(GW.logits, T, NEXP, KU, GW.ids[:T * KU - 1], GW.wts, GW.err), 'router with ids 1 element short')
    refused(log, lambda: G.bf16_gemm(torch.zeros(NEXP, 2560, dtype=torch.bfloat16), NEXP, 2560, GW.mixed_bf, GW.logits[:po.tpad(T) * NEXP - 1], T),
            'bf16_gemm (WMMA router logits) with out 1 element short')
    refused(log, lambda: G.bf16_gemm(torch.zeros(NEXP, 2560, dtype=torch.bfloat16), NEXP, 2560, GW.mixed_bf, GW.logits[:po.tpad(T) * NEXP - 1], T,
                                     wmma_out='bf16'), 'bf16_gemm wmma_out bf16 (the *_rbf kernels) with out 1 element short')
    refused(log, lambda: G.bf16_gemm(torch.zeros(NEXP, 2560, dtype=torch.bfloat16), NEXP, 2560, GW.mixed_bf, GW.logits, T, wmma_out='fp16'),
            'bf16_gemm with a wmma_out other than fp32 / bf16')
    # item 3: partial aliasing refused, exact in-place accepted
    D, HC = 2560, 4; n = T * HC * D; base = torch.zeros(4 * (n + 3 * n + 4096), dtype=torch.uint8)
    fv = lambda off_el, cnt: base[4 * off_el:4 * (off_el + cnt)].view(torch.float32)
    x = fv(0, n); w4 = torch.zeros(HC, D)
    refused(log, lambda: G.rms_mul(x, w4, fv(D, n), R, 1e-6), 'rms_mul with y = x shifted by one row (partial overlap)')
    refused(log, lambda: G.rms_mul(x, w4, None, R, 1e-6, ybf=base[4 * 7:4 * 7 + 2 * n].view(torch.bfloat16)), 'rms_mul with ybf overlapping x')
    refused(log, lambda: G.rms_mul(x, w4, None, R, 1e-6, scale=w4.view(-1)[:R]), 'rms_mul with scale inside w')
    accepted(log, lambda: G.rms_mul(x, w4, x, R, 1e-6), 'rms_mul in place (y is x exactly)')
    kn, qn = fv(0, n), fv(n, n); val, gate = fv(2 * n, T * D), fv(2 * n + T * D, T * HC)
    refused(log, lambda: G.ple_gate(kn, qn, val, fv(D, n), gate, T, HC), 'ple_gate with gated = kn shifted by one row')
    refused(log, lambda: G.ple_gate(kn, qn, val, fv(n + 1, n), gate, T, HC), 'ple_gate with gated overlapping qn')
    refused(log, lambda: G.ple_gate(kn, qn, val, kn, fv(2 * n + 5, T * HC), T, HC), 'ple_gate with gate overlapping val')
    accepted(log, lambda: G.ple_gate(kn, qn, val, kn, gate, T, HC), 'ple_gate in place (gated is kn exactly)')
    xn, gt = fv(0, n), fv(n, n)
    refused(log, lambda: G.hc_mean(xn, gt, fv(0, T * D), T, HC), 'hc_mean with out = the start of xn (partial: [T, D] in [T, HC, D])')
    refused(log, lambda: G.hc_mean(xn, gt, fv(n + 3 * D, T * D), T, HC), 'hc_mean with out inside gate')
    refused(log, lambda: G.hc_mean(xn, gt, fv(2 * n, T * D), T, HC, scale=fv(2 * n + 10, T * HC), w=w4), 'hc_mean with scale inside out')
    accepted(log, lambda: G.hc_mean(xn, gt, fv(2 * n, T * D), T, HC), 'hc_mean, disjoint out')
    a, b = fv(0, n), fv(n, n)
    refused(log, lambda: G.add_sum(a, b, fv(1, n), n), 'add_sum with y = a shifted by one element')
    refused(log, lambda: G.add_sum(a, b, fv(n - 7, n), n), 'add_sum with y straddling a and b')
    accepted(log, lambda: G.add_sum(a, b, a, n), 'add_sum in place (y is a exactly)')
    accepted(log, lambda: G.add_sum(a, b, b, n), 'add_sum in place (y is b exactly)')
    accepted(log, lambda: G.add_sum(a, a, fv(2 * n, n), n), 'add_sum with a is b (both read only)')
    blk, inj, res = fv(0, T * D), fv(T * D, T * HC), fv(T * D + T * HC, n)
    refused(log, lambda: G.hc_post_rms(blk, inj, res, w4, T, HC, 1e-6, scale=fv(T * D + T * HC + 9, T * HC)), 'hc_post_rms with scale inside res')
    refused(log, lambda: G.hc_post(fv(T * D + T * HC, T * D), inj, res, T, HC), 'hc_post with block inside res')
    C = 6400; cx = fv(0, T * C)
    refused(log, lambda: G.gdn_conv(cx, torch.zeros(C, 4), fv(T * C - 3 * C, 3 * C), fv(T * C, T * C), C, T), 'gdn_conv with the state inside x')
    pC = HC * D; px = fv(0, T * pC)
    refused(log, lambda: G.ple_conv(px, torch.zeros(pC * 8, dtype=torch.uint8), fv(100, pC * 9), pC, T, y=fv(T * pC, T * pC)), 'ple_conv with hist inside x')
    refused(log, lambda: G.bf16_mv(torch.zeros(NEXP, 2560, dtype=torch.bfloat16), NEXP, 2560, fv(0, T * D), fv(T * D - 64, T * NEXP), T), 'bf16_mv with out overlapping x')
    refused(log, lambda: G.to_bf16(fv(0, T * D), base[4 * 100:4 * 100 + 2 * po.tpad(T) * D].view(torch.bfloat16), T, D), 'to_bf16 with y overlapping x')
    refused(log, lambda: G.router(GW.logits, T, NEXP, KU, GW.ids, GW.ids.view(torch.float32), GW.err), 'router with wts on ids')
    refused(log, lambda: G.router(GW.logits, T, NEXP, KU, GW.ids, GW.wts, GW.ids[3:4]), 'router with err inside ids')
    accepted(log, lambda: G.router(GW.logits, T, NEXP, KU, GW.ids, GW.wts, GW.err), 'router on the glue workspace buffers')
    # item 5: the router logits mode
    check(po.ROUTER_LOGITS == 'wmma' and G.router_mode == 'wmma', f'router logits default: {po.ROUTER_LOGITS} (lead decision 11-03); exact one switch away')
    try:
        po.Glue.__init__(po.Glue.__new__(po.Glue), cpu, router_logits='fast'); bad_mode = False
    except ValueError: bad_mode = True
    check(bad_mode, 'Glue(router_logits=...) refuses a mode other than wmma / exact')
    check(po.WMMA_OUT == 'bf16' and G.wmma_out == 'bf16', f'WMMA output default: {po.WMMA_OUT} (user 11-04: llama.cpp\'s batched numerics; fp32 one switch away)')
    try:
        po.Glue.__init__(po.Glue.__new__(po.Glue), cpu, wmma_out='fp16'); bad_out = False
    except ValueError: bad_out = True
    check(bad_out, 'Glue(wmma_out=...) refuses an output other than fp32 / bf16')
    G2 = po.Glue.__new__(po.Glue); G2.dev = cpu; G2.m = Rec(log); G2.router_mode = 'exact'
    n0 = len(log); G2.router_logits(torch.zeros(NEXP, D, dtype=torch.bfloat16), NEXP, GW.mixed[:T * D], GW.logits, T)
    check(log[n0:] == ['pf_bf16_mv_n5'], f'router_logits mode exact = bf16_mv (OP_BF16_MV order) (recorded {log[n0:]})')
    refused(log, lambda: G.router_logits(torch.zeros(NEXP, D, dtype=torch.bfloat16), NEXP, GW.mixed[:T * D], GW.logits, T),
            'router_logits mode wmma without the bf16 input buffer')
    # -- capacity: the split ------------------------------------------------------------------------------------------------
    for fn, nm in ((lambda: capacity.prefill_need_mib(g, 2048), 'prefill_need_mib without the split'),
                   (lambda: capacity.budget(g, 32768, 4, 'coder', ('dflash',), prefill=2048), 'budget(prefill=) without the split')):
        try: fn(); ok = False
        except (TypeError, ValueError): ok = True
        check(ok, f'capacity: {nm} is refused (no default split)')
    a10, a8 = capacity.prefill_need_mib(g, 2048, gdn_keys=10, moe_blocks=12), capacity.prefill_need_mib(g, 2048, gdn_keys=8, moe_blocks=10)
    check(a10 != a8 and abs(sum(a10) - sum(a8)) < 64, f'capacity.prefill_need_mib follows the split: 10 / 12 -> {a10[0]:.1f} / {a10[1]:.1f} MiB, '
          f'8 / 10 -> {a8[0]:.1f} / {a8[1]:.1f} MiB')
    from caminus.engine.flashnext_tp import FlashNextTP
    got = {}
    class FakeRd:
        tensors = [type('t', (), dict(n_bytes=capacity.MODEL_BYTES['coder']))()]
    fake = type('E', (), {})(); fake.rd = FakeRd(); fake.model_tag = 'coder'; fake.ntok = 4; fake.geom = g; fake.max_ctx = 32768
    fake.idx_cache = True; fake.qsa = True; fake.devs = [0, 1]; fake.gdn_keys, fake.moe_blocks = 8, 10
    real_check, real_split = capacity.check, capacity.MEASURED_SPLIT
    capacity.check = lambda devs, need, what: got.setdefault('need', need) and []
    capacity.MEASURED_SPLIT = (8, 10)      # (the fixed part is measured at 10 / 12 only: this fake's split stands in for it here)
    try: FlashNextTP._check_vram(fake, (0, 0), (), 2048)
    finally: capacity.check, capacity.MEASURED_SPLIT = real_check, real_split
    # a QSA engine's prefill also holds the QSA prompt path's buffers (S7: indexer query rows, SEL rows, the score scratch),
    # counted as prefill_need_mib(qsa_ctx=max_ctx); this fake is a QSA engine at max_ctx 32768
    pf = capacity.prefill_need_mib(g, 2048, gdn_keys=8, moe_blocks=10, qsa_ctx=fake.max_ctx)
    pf0 = capacity.prefill_need_mib(g, 2048, gdn_keys=8, moe_blocks=10)
    if not check('need' in got, 'FlashNextTP._check_vram ran on a fake engine (gdn_keys 8, moe_blocks 10; capacity.check captured)'): return
    fx = capacity.fixed_mib('coder', 4, capacity.MODEL_BYTES['coder']); cb = capacity.tp_ctx_bytes(g, 32768); cb.update(capacity.tp_qsa_bytes(g, 32768, 4))
    want = [fx[c] + sum(cb.values()) / capacity.MIB + pf[c] for c in (0, 1)]
    check(all(abs(got['need'][c] - want[c]) < 1e-6 for c in (0, 1)), f'FlashNextTP._check_vram counts prefill_need_mib at the ENGINE\'s split '
          f'with qsa_ctx = max_ctx on a QSA engine (gdn_keys 8 / moe_blocks 10: {got["need"][0]:.1f} / {got["need"][1]:.1f} MiB = fixed + ctx + '
          f'{pf[0]:.1f} / {pf[1]:.1f}; the QSA prompt path +{pf[0] - pf0[0]:.2f} / +{pf[1] - pf0[1]:.2f})')
    # the same engine without QSA: qsa_ctx is not counted
    got0 = {}; fake0 = type('E', (), {})(); fake0.__dict__.update(fake.__dict__); fake0.qsa = False
    capacity.check = lambda devs, need, what: got0.setdefault('need', need) and []
    capacity.MEASURED_SPLIT = (8, 10)      # (as above)
    try: FlashNextTP._check_vram(fake0, (0, 0), (), 2048)
    finally: capacity.check, capacity.MEASURED_SPLIT = real_check, real_split
    want0 = [fx[c] + sum(capacity.tp_ctx_bytes(g, 32768).values()) / capacity.MIB + pf0[c] for c in (0, 1)]
    check('need' in got0 and all(abs(got0['need'][c] - want0[c]) < 1e-6 for c in (0, 1)),
          f'FlashNextTP._check_vram on the same engine without QSA counts no QSA prompt path buffers '
          f'({got0.get("need", [0, 0])[0]:.1f} / {got0.get("need", [0, 0])[1]:.1f} MiB = fixed + ctx + {pf0[0]:.1f} / {pf0[1]:.1f})')
    # -- the prefill plan (caminus.prefill.plan): one set of arguments for the projection and the allocations ----------------
    from caminus.prefill.plan import PrefillPlan, CODER_FEED
    from caminus.prefill import feed as pfd, glue as pgl, attn as pat
    from gguf import GGUFReader
    pl = PrefillPlan.of(g, 2048, gdn_keys=8, moe_blocks=10, qsa_ctx=32768)
    check(pl.need_mib() == pf, f'plan: capacity.prefill_need_mib is the plan\'s need_mib ({pl.need_mib()[0]:.3f} / {pl.need_mib()[1]:.3f})')
    rd = GGUFReader(coder, 'r'); fs = pfd.feed_shape(rd, coder)
    check(fs._replace(table_bytes=CODER_FEED.table_bytes) == CODER_FEED and fs.table_bytes == 248320 * 1360,
          f'plan: the Coder file\'s feed shapes = CODER_FEED but the table ({fs.table_bytes:,} B in the file, {CODER_FEED.table_bytes:,} counted)')
    ef = type('E', (), {})(); ef.__dict__.update(fake.__dict__); ef.ntok = 1; ef.feature_layers = [3, 7]; ef.feature_card = 1
    ep = PrefillPlan.for_engine(ef, 512, feed=fs)
    TC = capacity.PREFILL_TABLE_CARD
    lay2 = [pfd.feed_layout(512, fs.D, fs.ple_heads, fs.ple_dim, fs.ple_rb, fs.te_rb, fs.table_bytes if c == 0 else 0) for c in (0, 1)]
    same = (all(ep.card_dims(c) == mmq.card_dims(g, c, 8, 10) and ep.glue_dims(c) == pgl.glue_dims(g, c, 8) for c in (0, 1))
            and ep.attn_args(1) == dict(T=512, nh=g['nh'] // g['nkv'], hd=g['hd'], idx_dim=g['idx_dim'], rows_per_block=pat.ROWS_PER_BLOCK,
                                        n_feat=2, D=g['D'], T_pass=1, qsa_ctx=32768) and ep.attn_args(0)['n_feat'] == 0
            and [ep.feed_layout(TC), ep.feed_layout(1 - TC)] == lay2)
    check(same, 'plan: an engine\'s plan gives the arguments ChunkStack / PrefillFeed derived themselves before (split dims, glue dims, '
                'the feature export on its card only, T_pass = the engine\'s ntok, the feed\'s table on PREFILL_TABLE_CARD = the feed\'s devices[0])')
    # the engine counts the plan of the prefill it will be given (FlashNextTP.plan_prefill: its own ntok, its file's feed shapes,
    # the declared dense path), and a prefill needing more is refused before it allocates (plan.check_counted)
    from caminus.prefill.plan import check_counted
    from caminus.prefill import coder as pcd, tp as ptp
    em = type('E', (), {})(); em.__dict__.update(ef.__dict__); em.rd, em.path, em.devs = rd, coder, [0, 1]; em.qsa = False
    own = FlashNextTP.plan_prefill(em, 2048, 2, 1)
    wm = FlashNextTP.plan_prefill(em, dict(chunk=2048, q6k='wmma'), 2, 1)
    check(own.T_pass == 1 and own.feed == fs and not own.bf16_dense and own.n_feat == 2 and own.feature_card == 1 and wm.bf16_dense
          and all(wm.need_mib()[c] > own.need_mib()[c] for c in (0, 1)),
          f'plan: FlashNextTP.plan_prefill counts its own ntok (T_pass {own.T_pass}), its file\'s feed shapes and the declared dense path '
          f'(wmma +{wm.need_mib()[0] - own.need_mib()[0]:.1f} / +{wm.need_mib()[1] - own.need_mib()[1]:.1f} MiB)')
    for bad in (dict(chunk=2048, q6k='wmma', x=1), dict(q6k='wmma'), True):
        try: FlashNextTP.plan_prefill(em, bad); ok = False
        except ValueError: ok = True
        check(ok, f'plan: vram_prefill {bad!r} is refused')
    got1 = {}; capacity.check = lambda devs, need, what: got1.setdefault('need', need) and []
    real_split = capacity.MEASURED_SPLIT; capacity.MEASURED_SPLIT = (em.gdn_keys, em.moe_blocks)   # (this stand-in's split, as above)
    try: FlashNextTP._check_vram(em, (0, 0), (), wm, 2, 1)
    finally: capacity.check, capacity.MEASURED_SPLIT = real_check, real_split
    fx1 = capacity.fixed_mib('coder', 1, capacity.MODEL_BYTES['coder']); ctx1 = sum(capacity.tp_ctx_bytes(g, 32768).values()) / capacity.MIB
    want1 = [fx1[c] + ctx1 + wm.need_mib()[c] for c in (0, 1)]
    check('need' in got1 and all(abs(got1['need'][c] - want1[c]) < 1e-6 for c in (0, 1)),
          f'plan: FlashNextTP._check_vram counts the plan it is given (fixed + ctx + the wmma plan: {got1.get("need", [0, 0])[0]:.1f} / '
          f'{got1.get("need", [0, 0])[1]:.1f} MiB)')
    em.prefill_plan = own
    def refused_v(fn, what):
        try: fn(); ok, msg = False, ''
        except ValueError as e: ok, msg = True, str(e)
        check(ok, f'plan: {what} is refused before it allocates ({msg[:150]})')
    refused_v(lambda: pcd.Prefill(em, chunk=2048, q6k='wmma'), 'Prefill(q6k=\'wmma\') on an engine that counted the default path')
    refused_v(lambda: ptp.ChunkStack(em, None, 2048, q6k='wmma'), 'ChunkStack(q6k=\'wmma\') on the same engine')
    Sx = ptp.ChunkStack.__new__(ptp.ChunkStack); Sx.m, Sx.plan, Sx.WB = em, PrefillPlan.for_engine(em, 2048, feed=fs), [None, None]
    refused_v(Sx.alloc_q6k_buffers, 'ChunkStack.alloc_q6k_buffers() on the same engine')
    try: check_counted(em, PrefillPlan.for_engine(em, 1024), 'x'); ok = True
    except ValueError: ok = False
    em2 = type('E', (), {})(); em2.__dict__.update(em.__dict__); em2.prefill_plan = None
    try: check_counted(em2, PrefillPlan.for_engine(em2, 2048, bf16_dense=True), 'x'); ok2 = True
    except ValueError: ok2 = False
    check(ok and ok2, 'plan: a smaller chunk than counted, and any prefill on an engine that counted none, pass check_counted')
    em.prefill_plan = wm
    try: check_counted(em, PrefillPlan.for_engine(em, 2048, bf16_dense=True), 'x'); ok = True
    except ValueError: ok = False
    check(ok, 'plan: the wmma buffers pass on an engine told vram_prefill=dict(chunk=2048, q6k=\'wmma\')')
    del rd
    b1 = capacity.budget(g, 32768, 1, 'coder', ('dflash',), dflash_geom=(2, 4, 128, 2048), prefill=2048, split=(10, 12))
    a1 = PrefillPlan.of(g, 2048, gdn_keys=10, moe_blocks=12, T_pass=1, n_feat=capacity.DFLASH_FEATURE_LAYERS, qsa_ctx=32768).attn_bytes()
    ra = next(r for r in b1['rows'] if r[0].startswith('prefill attention'))
    check(abs(ra[1] - a1[0] / capacity.MIB) < 1e-9 and abs(ra[2] - a1[1] / capacity.MIB) < 1e-9,
          f'plan: capacity.budget at T=1 counts the attention workspace at T_pass 1 ({ra[1]:.2f} / {ra[2]:.2f} MiB)')
    # -- the old path, by CPU emulation of mm_ids_helper only -----------------------------------------------------------------
    rng = np.random.default_rng(5); T2 = 64
    good = valid_routing(rng, T2)
    for per_slot in (False, True):
        s1, dd, bd = emu_mm_ids_helper(good, NEXP, per_slot, fill=-7777)
        full = (s1 != -7777).all() and (dd[:T2 * KU] != -7777).all() and bd[NEXP] == T2 * KU and sorted(dd[:T2 * KU]) == list(range(T2 * KU))
        check(full, f'emulated mm_ids_helper, valid ids, per_slot {per_slot}: every ids_src1 / ids_dst entry written, bounds[n_exp] = T * k')
    GARBAGE = 0x7EEEEEEE                                                  # what torch.empty left in a table (any value)
    Ty, K_ = T2 * KU, 2560
    for desc, mk in (('an id = n_exp (256)', lambda a: a.__setitem__((3, 2), NEXP)),
                     ('an id = -1', lambda a: a.__setitem__((7, 0), -1)),
                     ('an id = INT_MIN (a NaN cast)', lambda a: a.__setitem__((9, 5), INT_MIN)),
                     ('two slots of one token = the same expert', lambda a: a.__setitem__((5, 4), a[5, 1])),
                     ('all 10 slots of one token equal', lambda a: a.__setitem__(17, a[17, 0]))):
        ids = good.copy(); mk(ids)
        inv, dd, bd = emu_mm_ids_helper(ids, NEXP, False, fill=GARBAGE)
        fwd, dd2, bd2 = emu_mm_ids_helper(ids, NEXP, True, fill=GARBAGE)
        un_inv = int((inv == GARBAGE).sum()); un_dst = int((dd[:Ty] == GARBAGE).sum())
        # host arithmetic only: the addresses the later kernels would derive from the unwritten words
        oob = []
        if un_inv:   # quantize_scatter writes token t's row to compact column ids_inv[t * k + slot] of y ([K/128][T*k] blocks of 144 B)
            c = int(inv[inv == GARBAGE][0]); byte = c * 144; oob.append(f'quantize_scatter would write y column {c} (byte {byte:,} > y\'s {mmq.y_bytes(K_, Ty):,})')
        ex = [e for e in range(NEXP) if any(dd[j] == GARBAGE for j in range(bd[e], bd[e + 1]) if 0 <= bd[e] <= bd[e + 1] <= Ty + 127)]
        if ex:       # MMQ writes expert e's column j to dst row ids_dst[j] ([T*k][rows] fp32)
            j = next(j for j in range(bd[ex[0]], bd[ex[0] + 1]) if dd[j] == GARBAGE)
            oob.append(f'MMQ (expert {ex[0]}, column {j}) would write dst row {int(dd[j])} (byte {int(dd[j]) * 4 * 128:,} > dst\'s {4 * 128 * Ty:,})')
        uf = [j for j in range(Ty) if fwd[j] == GARBAGE]
        if uf: oob.append(f'the down quantizer would read x row ids_fwd[{uf[0]}] = {int(fwd[uf[0]])} (> {Ty} rows)')
        nb = int(bd[NEXP]) if bd[NEXP] != GARBAGE else None
        check(bool(oob), f'old path UNSAFE (CPU emulation, nothing launched), {desc}: {un_inv} ids_inv entries and {un_dst} ids_dst entries '
              f'never written, bounds[n_exp] = {nb} (T * k = {Ty}); ' + '; '.join(oob))
        # zeroed tables alone: every stale word is 0 (in range: no garbage address), but slots still get no group column
        # (their output rows are never computed: wrong data the host cannot see): why the clamp and the flag are needed
        inv0, dd0, bd0 = emu_mm_ids_helper(ids, NEXP, False, fill=0)
        lo, hi = int(bd0[0]), int(bd0[NEXP])
        covered = {int(dd0[j]) for e in range(NEXP) for j in range(int(bd0[e]), int(bd0[e + 1]))}
        missing = sorted(set(range(Ty)) - covered)
        inr = all(0 <= int(v) < Ty for v in inv0) and all(0 <= int(v) < Ty for v in dd0[lo:hi])
        check(inr and len(missing) >= 1, f'  with zeroed tables only ({desc}): every word in range (no out-of-bounds address), but {len(missing)} '
              f'slot(s) {missing[:4]} get no group column (their outputs never computed): the clamp and the flag are still needed')
        cl, fl, ch = emu_clamp(ids, NEXP)
        s1c, ddc, bdc = emu_mm_ids_helper(cl, NEXP, False, fill=GARBAGE)
        ok = (s1c != GARBAGE).all() and (ddc[:Ty] != GARBAGE).all() and bdc[NEXP] == Ty and sorted(ddc[:Ty]) == list(range(Ty))
        check(ok and fl and ch, f'  after the clamp\'s rule ({desc}): flags {fl:#x}, tokens {ch} changed, every table entry written, '
              f'bounds[n_exp] = {int(bdc[NEXP])}')


# ---- stage: gpu (one card) ----------------------------------------------------------------------------------------------
def stage_gpu():
    import torch
    from caminus.prefill import mmq, glue as po
    dev = torch.device('cuda:0'); torch.cuda.set_device(dev)
    print('device', torch.cuda.get_device_name(dev), flush=True)
    M = mmq.MMQ(dev, formats=('Q8_0',)); G = po.Glue(dev)
    rng = np.random.default_rng(21)
    z = lambda *s, dt=torch.int32: torch.zeros(*s, dtype=dt, device=dev)
    def tables(ids_dev, T, per_slot, err):
        s1, dd, bd = z(T * KU), z(T * KU + mmq.J_MAX - 1), z(NEXP + 1)
        M.group(ids_dev, T, KU, NEXP, s1, dd, bd, per_slot, err); torch.cuda.synchronize(dev)
        return s1.cpu().numpy(), dd.cpu().numpy(), bd.cpu().numpy()
    # -- clean ids: no flag, unchanged, tables == the emulation (which validates the emulation) -------------------------------
    for T in (1, 2, 777, 2048):
        good = valid_routing(rng, T); err = z(1)
        for per_slot in (False, True):
            idd = torch.from_numpy(good.copy()).to(dev)
            s1, dd, bd = tables(idd, T, per_slot, err)
            e1, e2, e3 = emu_mm_ids_helper(good, NEXP, per_slot, fill=0)
            check(np.array_equal(idd.cpu().numpy(), good) and int(err.item()) == 0 and np.array_equal(s1, e1) and np.array_equal(dd[:T * KU], e2[:T * KU])
                  and np.array_equal(bd, e3), f'gpu clean ids T={T} per_slot {per_slot}: ids unchanged, flag 0, the device tables == the CPU emulation of '
                  'mm_ids_helper (bitwise)')
    # the clamp's own cost at T = 2048 on valid ids (a pure check: nothing stored), one launch per mmq.group call
    import ctypes
    idv = torch.from_numpy(valid_routing(rng, 2048)).to(dev); err = z(1); ts = []
    for _ in range(23):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); M._launch(M.kclamp, (8,), (256,), idv, ctypes.c_int(2048), ctypes.c_int(KU), ctypes.c_int(NEXP), err); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b))
    print(f'SPEED k3_mmq_ids_clamp, T = 2048 valid ids: median {np.median(ts[3:]) * 1e3:.1f} us of 20 (flag {int(err.item())}; two calls per MoE layer: '
          f'gate / up and down)', flush=True)
    # -- out-of-range / duplicate ids into the clamped path -----------------------------------------------------------------
    T = 2048
    for kinds, want in ((('range',), po.ROUTE_RANGE), (('dup',), po.ROUTE_DUP), (('range', 'dup'), po.ROUTE_RANGE | po.ROUTE_DUP)):
        ids, what = bad_routing(rng, T, kinds)
        cl, fl, ch = emu_clamp(ids, NEXP)
        for per_slot in (False, True):
            err = z(1); idd = torch.from_numpy(ids.copy()).to(dev)
            s1, dd, bd = tables(idd, T, per_slot, err)
            out = idd.cpu().numpy(); f = int(err.item())
            inr = bool(((out >= 0) & (out < NEXP)).all()); dist = all(len(set(r)) == KU for r in out.tolist())
            untouched = [t for t in range(T) if t not in what]
            same_rest = np.array_equal(out[untouched], ids[untouched])
            e1, e2, e3 = emu_mm_ids_helper(out, NEXP, per_slot, fill=0)
            full = bd[NEXP] == T * KU and sorted(dd[:T * KU].tolist()) == list(range(T * KU))
            inv_ok = (not per_slot and all(s1[dd[j]] == j for j in range(T * KU))) or (per_slot and np.array_equal(s1, dd[:T * KU]))
            check(inr and dist and same_rest and np.array_equal(out, cl) and f == want == fl,
                  f'gpu clamp ({"+".join(kinds)}, {len(what)} bad tokens, per_slot {per_slot}): device ids all in [0, {NEXP}) and distinct per token, '
                  f'the {len(untouched)} other tokens bitwise unchanged, the replacements == the CPU rule, flag {f:#x} (want {want:#x})')
            check(full and inv_ok and np.array_equal(s1, e1) and np.array_equal(dd[:T * KU], e2[:T * KU]) and np.array_equal(bd, e3),
                  f'  mm_ids_helper after the clamp: bounds[n_exp] = {int(bd[NEXP])} = T * k, ids_dst a permutation of the {T * KU} slots, the '
                  f'{"forward" if per_slot else "inverse"} map consistent, == the CPU emulation on the clamped ids')
            try: po.check_routing(err); refused_ = False
            except po.RoutingError as e: refused_ = True; msg = str(e)
            check(refused_, f'  host refuses on the flag: check_routing raised RoutingError ({msg[:100] if refused_ else ""})')
    # MMQ mul_mat_id on the clamped groups: every output column written; == the same run on ids valid from the start
    K, rows = 256, 128; rb = mmq.row_bytes('Q8_0', K)
    blocks = np.zeros((NEXP * rows * K // 32, 34), np.uint8)
    blocks[:, :2] = np.frombuffer(np.float16(0.02).tobytes(), np.uint8)
    blocks[:, 2:] = rng.integers(-127, 128, (blocks.shape[0], 32), dtype=np.int8).view(np.uint8)
    w = torch.from_numpy(blocks.reshape(-1)).to(dev); x = torch.from_numpy(rng.standard_normal((T, K)).astype(np.float32)).to(dev)
    ids, what = bad_routing(rng, T)
    def moe(idd):
        err = z(1); s1, dd, bd = z(T * KU), z(T * KU + mmq.J_MAX - 1), z(NEXP + 1)
        y = torch.zeros(mmq.y_bytes(K, T * KU), dtype=torch.uint8, device=dev)
        out = torch.full((T * KU * rows,), float('nan'), device=dev)
        M.group(idd, T, KU, NEXP, s1, dd, bd, False, err)
        M.quantize_scatter(x, K, T, s1, KU, y, M.layout('Q8_0'))
        J = M.mm_id('Q8_0', w, NEXP, rows, K, y, T, KU, dd, bd, out); torch.cuda.synchronize(dev)
        return out.cpu().numpy(), int(err.item()), J
    idb = torch.from_numpy(ids.copy()).to(dev)
    o1, f1, J = moe(idb)
    o2, f2, _ = moe(torch.from_numpy(idb.cpu().numpy().copy()).to(dev))
    check(f1 == po.ROUTE_RANGE | po.ROUTE_DUP and f2 == 0 and bool(np.isfinite(o1).all()) and np.array_equal(o1.view(np.int32), o2.view(np.int32)),
          f'gpu clamp -> mm_ids_helper -> quantize_scatter -> MMQ mul_mat_id (Q8_0, {NEXP} experts x {rows} x {K}, T={T}, J={J}): every one of the '
          f'{T * KU} x {rows} outputs written (no NaN left), == the run on the clamped ids given as valid input (bitwise; its flag {f2})')
    # -- NaN / inf logits into pf_router ------------------------------------------------------------------------------------
    for k in (KU, 40):
        lg = rng.standard_normal((T, NEXP)).astype(np.float32) * 2.0; clean = lg.copy()
        bad = {3: 'one NaN', 10: '+inf', 11: '-inf', 20: 'all NaN', 21: 'all -inf', 22: 'all +inf', 40: 'NaN and +inf'}
        lg[3, 17] = np.nan; lg[10, 5] = np.inf; lg[11, 0] = -np.inf; lg[20, :] = np.nan; lg[21, :] = -np.inf; lg[22, :] = np.inf
        lg[40, 1] = np.nan; lg[40, 2] = np.inf
        res = {}
        for tag, arr in (('bad', lg), ('clean', clean)):
            err = z(1); ids_ = z(T, k); wts_ = z(T, k, dt=torch.float32)
            G.router(torch.from_numpy(arr).to(dev), T, NEXP, k, ids_, wts_, err); torch.cuda.synchronize(dev)
            res[tag] = (ids_.cpu().numpy(), wts_.cpu().numpy(), int(err.item()))
        (ib, wb, fb), (ic, wc, fc) = res['bad'], res['clean']
        rest = [t for t in range(T) if t not in bad]
        inr = bool(((ib >= 0) & (ib < NEXP)).all()); dist = all(len(set(r)) == k for r in ib.tolist())
        check(fb == po.ROUTE_NONFINITE and fc == 0 and inr and dist and np.array_equal(ib[rest], ic[rest]) and np.array_equal(wb[rest].view(np.int32), wc[rest].view(np.int32)),
              f'gpu router k={k} with NaN / inf logits ({len(bad)} tokens: {", ".join(bad.values())}): flag {fb:#x} (want {po.ROUTE_NONFINITE:#x}; '
              f'clean run {fc}), device ids of ALL {T} tokens in range and distinct, the other {len(rest)} tokens\' ids / weights bitwise the clean run')
        if k == KU:   # the router's ids from NaN logits through the clamped grouping: valid already (no new bit), the tables complete
            err = z(1); err.fill_(fb); idd = torch.from_numpy(ib.copy()).to(dev)
            s1, dd, bd = tables(idd, T, False, err)
            check(int(err.item()) == fb and np.array_equal(idd.cpu().numpy(), ib) and bd[NEXP] == T * KU,
                  f'  the NaN tokens\' ids through mmq.group: no id replaced (flag stays {int(err.item()):#x}), bounds[n_exp] = {int(bd[NEXP])}')
    # -- the workspaces' flag words -----------------------------------------------------------------------------------------
    from caminus.device import capacity
    gws = po.GlueWorkspace(dev, 128, dict(D=2560, HC=4, lr=320, n_exp=NEXP, k=KU, ab=30, conv=6400, gk=10, gs=128), check=False)
    lg = torch.zeros(po.tpad(128), NEXP, device=dev); lg[7, 3] = float('nan')
    G.router(lg, 128, NEXP, KU, gws.ids, gws.wts, gws.err)
    try: gws.check_routing(); r1 = False
    except po.RoutingError: r1 = True
    po.clear_routing(gws.err); G.router(lg[8:].contiguous(), 120, NEXP, KU, gws.ids, gws.wts, gws.err)
    try: gws.check_routing(); r2 = True
    except po.RoutingError: r2 = False
    check(r1 and r2, 'GlueWorkspace.check_routing: refuses after a NaN logit, passes after clear_routing and a clean chunk')
    del gws; torch.cuda.synchronize(dev); torch.cuda.empty_cache()
    _ = capacity


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('coder'); ap.add_argument('stages', nargs='+')
    a = ap.parse_args()
    if 'cpu' in a.stages: stage_cpu(a.coder)
    if 'gpu' in a.stages: stage_gpu()
    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'))
    sys.exit(0 if not FAILS else 1)


if __name__ == '__main__':
    main()
