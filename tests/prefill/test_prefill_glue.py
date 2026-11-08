"""Gates of the batched prefill's BF16 GEMMs and glue (docs/specs/spec-prefill-cache.md M5 S3; caminus.prefill.glue,
kernels/prefill_glue.hip, kernels/gemm.hip's WMMA).

  python -I tests/prefill/test_prefill_glue.py CODER STAGE... [--scratch DIR] [--ref-stock BIN] [--golden DIR] [--json OUT]

Stages (GPU stages on cuda:0: HIP_VISIBLE_DEVICES picks the card; 'gpu' = g1 g2 wmma speed):
  g1     every bitwise op against K3's own decode op (caminus.engine.k3.Engine, module k3_gguf_gsq, one token per pass) on the same
         inputs, real Coder weights, over the first T of 2048 tokens for T = 1, 4, 512, 777, 2048, bitwise:
         rms_mul, hc_post_rms, hc_post, hc_mean, router (forced exact ties and near ties included), gdn_conv + its state,
         ple_gate, ple_conv + its history (and fused with add_sum), add_sum, bf16_mv (router logits, shared-expert gate,
         ssm alpha / beta on both cards' rows, hc inject); the bf16 copies (to_bf16 == torch's round to nearest even,
         silu_bf16 within one bf16 step of fp64's silu); router_logits mode 'exact' == the decode's logits; the router's
         routing flag word stays 0 on every one of these inputs (ties, near ties, an all-equal token)
  g2     G2a: every op twice at T = 2048, bitwise. G2b: the same 2048 tokens as chunks 1024 + 1024 / 777 + 1271 /
         512 + 788 + 748 / 5 + 3 + 2040 / 1 + 4 + 2043 (state and history carried between chunks; chunks shorter than the
         3- and 9-token windows) == one T = 2048 pass, bitwise per position and in the final state; the WMMA GEMMs at the
         same splits (every token at another row of its 128-row tile)
  wmma   the wide BF16 GEMMs on gemm.hip's WMMA (hc down / up, output_hc down / up, ple key / value, indexer k / q, and
         the router logits: router_logits' default mode, lead decision 11-03) on
         recorded activations (--golden: hc_norm / hc_mixed rows dumped by gg_golden, 768 of each; the default dump is
         REAP-308's, the same architecture, NOT the Coder's own activations; the hc up input from them through an fp64 hc
         down) and random ones (T = 2048): max |err| / max |ref| and mean |err| / mean |ref| against fp64, next to
         K3's decode op (fp32 activations) and llama.cpp's own batched op (--ref-stock gg_mmq_ref: ggml_mul_mat, which runs
         hipBLAS with BF16 compute above 8 columns on RDNA3); gate: WMMA's error <= llama.cpp's batched op's, both measures.
         The router logits (added 11-04): gate = the bf16-activation floor (fp64 on the bf16-rounded activations) plus the
         WMMA's fp32 accumulation only (<= 1e-5 of max |floor|); llama.cpp's op is reported next to it, not gated (it
         rounds its output to bf16, which scatters its error around the floor); and the share of tokens whose top-10
         experts from the WMMA logits equal those from the exact logits and from fp64. wmma_out 'bf16' (fix:prefill-s6:
         gemm.hip's *_rbf kernels): every element bitwise RNE to bf16 of the fp32 mode's (the gate), its agreement with
         llama.cpp's batched op (which rounds its output to bf16 above 8 columns) and the router top-10 sets vs llama.cpp's
         logits reported. G2 / xcard run both output modes
  xcard  both cards: the same inputs on the XTX and the XT give the same bits, every op (WMMA included) and the states
  speed  per op at T = 2048 on both cards' shapes: ms (median of 20) and the projected glue + BF16 time of one chunk of
         the 48 layers (op counts of the Coder's layer stack), against S2's measured quantized GEMM stack
  vram   CPU: the glue workspace per card (prefill_ops.glue_layout) and capacity.prefill_need_mib with it (FlashNextTP's
         default split, gdn_keys 10 / moe_blocks 12)
Prints PASS / FAIL per check and RESULT PASS / FAIL; exit 0 = PASS."""
import argparse, json, os, re, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'python'))
from caminus.engine import flashnext   # noqa: E402,F401  (puts gguf-py on the path)
from gguf import GGUFReader   # noqa: E402

from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
ART = devpaths.artifacts()
FAILS, NUMBERS = [], {}
EPS = 9.999999974752427e-07
HC, D, LR, NEXP, KU = 4, 2560, 320, 256, 10
TS = (1, 4, 512, 777, 2048)
SPLITS = ((1024, 1024), (777, 1271), (512, 788, 748), (5, 3, 2040), (1, 4, 2043))
# the TP split's GDN rows (flashnext_tp._build_tapes, gdn_keys 10): card c's ssm alpha / beta rows and qkv / conv channels
def _tp_split(default=(10, 12)):
    """--tp-split GK,MB (taken out of sys.argv before argparse): the TP split whose card shapes are run (default the
    measured XTX + XT split 10 / 12; 8,10: the even split of equal pairs)"""
    if '--tp-split' not in sys.argv: return default
    i = sys.argv.index('--tp-split'); v = tuple(int(x) for x in sys.argv[i + 1].split(',')); del sys.argv[i:i + 2]
    print(f'split: gdn_keys {v[0]}, moe_blocks {v[1]}', flush=True)
    return v


GK = _tp_split()[0]; KC, KS = [GK, 16 - GK], [0, GK]   # (--tp-split)
VRUNS = [[(16 * g + KS[c], 16 * g + KS[c] + KC[c]) for g in range(3)] for c in (0, 1)]
QKV_R = [[(128 * KS[c], 128 * (KS[c] + KC[c])), (2048 + 128 * KS[c], 2048 + 128 * (KS[c] + KC[c]))] +
         [(4096 + 128 * a, 4096 + 128 * b) for a, b in VRUNS[c]] for c in (0, 1)]


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)
    return ok


def rows_of(ranges):
    return np.concatenate([np.arange(a, b) for a, b in ranges])


class W:
    """real Coder tensors (shard 1, memory-mapped through gguf-py; small tensors only)"""
    def __init__(self, rd):
        self.T = {t.name: t for t in rd.tensors}

    def raw(self, name):
        return np.asarray(self.T[name].data).reshape(-1).view(np.uint8)

    def f32(self, name):
        t = self.T[name]; assert t.tensor_type.name == 'F32', name
        return np.asarray(t.data).reshape(-1).view(np.float32)

    def bf16(self, name):
        """raw bf16 rows [rows, K] as uint16"""
        t = self.T[name]; assert t.tensor_type.name == 'BF16', name
        K = int(t.shape[0]); return self.raw(name).view(np.uint16).reshape(-1, K)


def _np(a):
    """a tensor (bf16 through its int16 view) or an array -> numpy"""
    if hasattr(a, 'cpu'):
        import torch
        if a.dtype == torch.bfloat16: a = a.view(torch.int16)
        return a.detach().cpu().numpy()
    return a


def bits(a):
    a = _np(a)
    return np.ascontiguousarray(a).view({4: np.int32, 2: np.int16, 1: np.int8}[a.dtype.itemsize])


def same(a, b):
    a, b = _np(a), _np(b)
    return a.shape == b.shape and np.array_equal(bits(a), bits(b))


def ndiff(a, b):
    a, b = _np(a), _np(b)
    return int((bits(a) != bits(b)).sum()) if a.shape == b.shape else -1


def bf_dev(torch, dev, u16):
    """raw bf16 rows (numpy uint16) -> a device bfloat16 tensor of the same shape"""
    return torch.from_numpy(np.ascontiguousarray(u16).view(np.int16)).to(dev).view(torch.bfloat16)


def acts(rng, n, scale=1.0, outl=0.002):
    x = rng.standard_normal(n).astype(np.float32) * scale
    x[rng.random(n) < outl] *= 30.0
    return x


# ---- the inputs / weights of one test case (a T0 = 2048-token chunk) -----------------------------------------------------
class Case:
    def __init__(self, torch, dev, wt, T0=2048, seed=11):
        rng = np.random.default_rng(seed); self.T0 = T0
        put = lambda a, dt=None: (lambda t: t if dt is None else t.to(dt))(torch.from_numpy(np.ascontiguousarray(a)).to(dev))
        self.put = put
        self.hcn = put(wt.f32('blk.0.hc_attn_norm.weight').reshape(HC, D))
        self.res = put(acts(rng, T0 * HC * D).reshape(T0, HC, D))
        self.blk = put(acts(rng, T0 * D, 0.3).reshape(T0, D))
        self.inj = put(acts(rng, T0 * HC, 2.0).reshape(T0, HC))
        self.gate = put(acts(rng, T0 * HC * D, 3.0).reshape(T0, HC, D))
        self.xn = put(acts(rng, T0 * HC * D, 0.1).reshape(T0, HC, D))
        lg = rng.standard_normal((T0, NEXP)).astype(np.float32) * 2.0
        for t in range(0, T0, 3):                                          # forced ties among the likely winners
            top = np.argsort(-lg[t])[:12]; i, j = rng.choice(top, 2, replace=False)
            if t % 2: lg[t, j] = lg[t, i]                                    # exact tie of logits (and so of probabilities)
            else: lg[t, j] = np.nextafter(lg[t, i], np.float32(-np.inf))     # a near tie (one ulp apart)
        lg[5, :] = 1.25                                                      # a token whose logits are all equal
        self.logits = put(lg)
        self.mixed = put(acts(rng, T0 * D, 0.5).reshape(T0, D))
        self.xbig = put(acts(rng, T0 * HC * D, 0.05).reshape(T0, HC * D))   # hc inject's input (xn rows of 10240)
        # GDN conv: card 0's channels (the qkv rows of 10 key heads)
        cw = wt.f32('blk.0.ssm_conv1d.weight').reshape(-1, 4); sel = rows_of(QKV_R[0])
        self.C = len(sel); self.cw = put(cw[sel]); self.cx = put(acts(rng, T0 * self.C).reshape(T0, self.C)); self.cst = put(acts(rng, self.C * 3).reshape(self.C, 3))
        # PLE (layer 1)
        self.pw = put(wt.raw('blk.1.ple_conv1d.weight')); self.pC = HC * D
        self.px = put(acts(rng, T0 * self.pC, 0.5).reshape(T0, self.pC)); self.phist = put(acts(rng, self.pC * 9, 0.5).reshape(self.pC, 9))
        self.pa = put(acts(rng, T0 * self.pC, 0.5).reshape(T0, self.pC))
        self.kn = put(acts(rng, T0 * HC * D, 0.2).reshape(T0, HC, D)); self.qn = put(acts(rng, T0 * HC * D, 0.2).reshape(T0, HC, D))
        self.val = put(acts(rng, T0 * D).reshape(T0, D))
        # thin BF16 matrices: (name, weights [N, K] raw bf16, x [T0, K])
        al = wt.bf16('blk.0.ssm_alpha.weight'); be = wt.bf16('blk.0.ssm_beta.weight')
        bf = lambda a: bf_dev(torch, dev, a)
        self.thin = [('router', bf(wt.bf16('blk.0.ffn_gate_inp.weight')), self.mixed),
                     ('shexp_gate', bf(wt.bf16('blk.0.ffn_gate_inp_shexp.weight')), self.mixed),
                     ('alpha_c0', bf(al[rows_of(VRUNS[0])]), self.mixed), ('beta_c0', bf(be[rows_of(VRUNS[0])]), self.mixed),
                     ('alpha_c1', bf(al[rows_of(VRUNS[1])]), self.mixed), ('beta_c1', bf(be[rows_of(VRUNS[1])]), self.mixed),
                     ('hc_inject', bf(wt.bf16('blk.0.hc_attn_inject.weight')), self.xbig)]


# ---- K3's decode ops, one token per pass (the reference) ---------------------------------------------------------------
class Dec:
    def __init__(self, torch, dev):
        from caminus.engine import k3
        self.torch, self.dev, self.k3 = torch, dev, k3

    def eng(self):
        return self.k3.Engine(module='k3_gguf_gsq', grid=self.k3.resident_grid('k3_gguf_gsq'), T=1)

    def z(self, *s, dt=None):
        return self.torch.zeros(*s, dtype=dt or self.torch.float32, device=self.dev)

    def per_token(self, e, T, ins, outs):
        """ins: [(buf, src [T, ...])], outs: [(buf, dst [T, ...])]; one pass per token"""
        e.build()
        for t in range(T):
            for b, s in ins: b.copy_(s[t].reshape(b.shape))
            e.run()
            for b, d in outs: d[t].copy_(b.reshape(d[t].shape))

    def rms_mul(self, x, w, T):
        e = self.eng(); xb, ob = self.z(HC, D), self.z(HC, D); e.rms_mul(xb, w, ob, EPS)
        out = self.z(T, HC, D); self.per_token(e, T, [(xb, x)], [(ob, out)]); return out

    def hc_post_rms(self, blk, inj, res, w, T):
        e = self.eng(); bb, ib, rb, xb = self.z(D), self.z(HC), self.z(HC, D), self.z(HC, D)
        e.hc_post_rms(bb, ib, rb, w, xb, EPS, inject_scale=0.25)
        ro, xo = self.z(T, HC, D), self.z(T, HC, D); self.per_token(e, T, [(bb, blk), (ib, inj), (rb, res)], [(rb, ro), (xb, xo)]); return ro, xo

    def hc_post(self, blk, inj, res, T):
        e = self.eng(); bb, ib, rb, ro = self.z(D), self.z(HC), self.z(HC, D), self.z(HC, D); e.hc_post(bb, ib, rb, ro)
        out = self.z(T, HC, D); self.per_token(e, T, [(bb, blk), (ib, inj), (rb, res)], [(ro, out)]); return out

    def hc_mean(self, xn, gate, T):
        e = self.eng(); xb, gb, ob = self.z(HC, D), self.z(HC, D), self.z(D); e.hc_mean(xb, gb, ob)
        out = self.z(T, D); self.per_token(e, T, [(xb, xn), (gb, gate)], [(ob, out)]); return out

    def router(self, lg, T):
        e = self.eng(); lb, ib, wb = self.z(NEXP), self.z(KU, dt=self.torch.int32), self.z(KU); e.router(lb, KU, ib, wb)
        ids, wts = self.z(T, KU, dt=self.torch.int32), self.z(T, KU); self.per_token(e, T, [(lb, lg)], [(ib, ids), (wb, wts)]); return ids, wts

    def ple_gate(self, kn, qn, val, T):
        e = self.eng(); kb, qb, vb, gb, pb = self.z(HC, D), self.z(HC, D), self.z(D), self.z(HC, D), self.z(HC)
        e.ple_gate(kb, qb, vb, gb, pb)
        g, p = self.z(T, HC, D), self.z(T, HC); self.per_token(e, T, [(kb, kn), (qb, qn), (vb, val)], [(gb, g), (pb, p)]); return g, p

    def add_sum(self, a, b, y, T):
        e = self.eng(); n = a[0].numel(); ab, bb, yb = self.z(n), self.z(n), self.z(n); e.add_sum(ab, bb, yb)
        out = self.z(T, n); self.per_token(e, T, [(ab, a), (bb, b), (yb, y)], [(yb, out)]); return out

    def bf16_mv(self, w, x, T):
        N, K = w.shape; e = self.eng(); xb, ob = self.z(K), self.z(N); e.bf16_mv(w, N, K, xb, ob)
        out = self.z(T, N); self.per_token(e, T, [(xb, x)], [(ob, out)]); return out

    def stateful(self, kind, x, w, st0, T, keep):
        """gdn_conv / ple_conv over T tokens with the lazy commit (params[1] = 1 from the second pass): outputs [T, C] and the
        committed state after each n in keep (one more pass commits the last token)"""
        torch = self.torch; C = x.shape[1]; e = self.eng()
        xb, ob, xp, st = self.z(C), self.z(C), self.z(C), st0.clone()
        if kind == 'conv': e.gdn_conv(xb, w, st, ob, xp)
        else: e.ple_conv(xb, w, st, ob, xp, dil=3)
        e.build(); out = self.z(T, C); states = {}
        for t in range(T + 1):
            e.params[1] = 0 if t == 0 else 1
            xb.copy_(x[t] if t < T else torch.zeros_like(xb)); e.run()
            if t in keep: states[t] = st.clone()
            if t < T: out[t].copy_(ob)
        return out, states


# ---- stage: g1 ------------------------------------------------------------------------------------------------------------
def stage_g1(torch, dev, wt):
    from caminus.prefill import glue as po
    G = po.Glue(dev); dec = Dec(torch, dev); cs = Case(torch, dev, wt); T0 = cs.T0
    t0 = time.time()
    z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
    def over_T(name, ref, run):
        """run(T) -> the batched outputs (tuple); ref: the decode's (tuple, full T0 rows) or a callable(T) -> tuple"""
        bad = []
        full = ref(T0) if callable(ref) else ref
        for i, r in enumerate(full):                                         # (a reference that never ran would be zeros)
            rf = r.float() if r.dtype != torch.int32 else r.float() + 1
            if not (torch.isfinite(rf).all().item() and (rf != 0).float().mean().item() > 0.5): bad.append(f'reference out{i} is degenerate')
        for T in TS:
            got = run(T); want = ref(T) if callable(ref) else tuple(r[:T] for r in ref)
            for i, (g, w) in enumerate(zip(got, want)):
                if not same(g, w): bad.append(f'T={T} out{i}: {ndiff(g, w)} elements differ')
        what = name if ' == ' in name else f'{name} == K3 decode op'
        check(not bad, f'G1 {what}, bitwise, T = {"/".join(map(str, TS))}' + (f' ({bad[:3]})' if bad else ''))
    # rms_mul (+ its bf16 copy)
    ref = dec.rms_mul(cs.res, cs.hcn, T0)
    def run_rms(T):
        y, yb = z(T, HC, D), z(T, HC, D, dt=torch.bfloat16); G.rms_mul(cs.res, cs.hcn, y, T * HC, EPS, ybf=yb)
        check(same(yb, y.to(torch.bfloat16)), f'G1 rms_mul bf16 copy == torch bf16 (round to nearest even) of its output, T={T}') if T == T0 else None
        return (y,)
    over_T('rms_mul (rows of 2560, the hc norm)', (ref,), run_rms)
    # in place
    y = cs.res.clone(); G.rms_mul(y, cs.hcn, y, T0 * HC, EPS); check(same(y, ref), 'G1 rms_mul in place == K3 decode op, bitwise')
    # hc_post_rms
    ro, xo = dec.hc_post_rms(cs.blk, cs.inj, cs.res, cs.hcn, T0)
    def run_hpr(T):
        r, x, xb = cs.res[:T].clone(), z(T, HC, D), z(T, HC, D, dt=torch.bfloat16)
        G.hc_post_rms(cs.blk, cs.inj, r, cs.hcn, T, HC, EPS, xn=x, xnbf=xb)
        if T == T0: check(same(xb, x.to(torch.bfloat16)), 'G1 hc_post_rms bf16 copy == torch bf16 of its xn')
        return r, x
    over_T('hc_post_rms (residual + next norm)', (ro, xo), run_hpr)
    # the prefill's path: no fp32 xn stored, its consumers recompute it from the residual and the row scale
    hm_ref = dec.hc_mean(xo, cs.gate, T0)
    ij_w = next(w for n, w, _ in cs.thin if n == 'hc_inject'); ij_ref = dec.bf16_mv(ij_w, xo.reshape(T0, -1), T0)
    def run_hpr_rs(T):
        r, xb, sc = cs.res[:T].clone(), z(T, HC, D, dt=torch.bfloat16), z(T, HC)
        G.hc_post_rms(cs.blk, cs.inj, r, cs.hcn, T, HC, EPS, xnbf=xb, scale=sc)
        m, ij = z(T, D), z(T, HC)
        G.hc_mean(r, cs.gate, m, T, HC, scale=sc, w=cs.hcn)
        G.bf16_mv(ij_w, HC, HC * D, r.reshape(T, -1), ij, T, xscale=sc, xw=cs.hcn)
        return r, m, ij, xb
    over_T('hc_post_rms (residual, scale, bf16 xn; no fp32 xn) -> hc_mean and hc inject recomputing xn from the residual '
           '== K3 decode hc_post_rms -> hc_mean / bf16_mv', (ro, hm_ref, ij_ref, xo.to(torch.bfloat16)), run_hpr_rs)
    xr = dec.rms_mul(cs.res, cs.hcn, T0); hm2 = dec.hc_mean(xr, cs.gate, T0)
    def run_rms_rs(T):
        sc, m = z(T, HC), z(T, D); G.rms_mul(cs.res, cs.hcn, None, T * HC, EPS, scale=sc); G.hc_mean(cs.res, cs.gate, m, T, HC, scale=sc, w=cs.hcn)
        return (m,)
    over_T('rms_mul (scale only) -> hc_mean recomputing xn == K3 decode rms_mul -> hc_mean', (hm2,), run_rms_rs)
    ref = dec.hc_post(cs.blk, cs.inj, cs.res, T0)
    def run_hp(T):
        r = cs.res[:T].clone(); G.hc_post(cs.blk, cs.inj, r, T, HC); return (r,)
    over_T('hc_post', (ref,), run_hp)
    ref = dec.hc_mean(cs.xn, cs.gate, T0)
    def run_hm(T):
        o = z(T, D); G.hc_mean(cs.xn, cs.gate, o, T, HC); return (o,)
    over_T('hc_mean', (ref,), run_hm)
    # router
    ids, wts = dec.router(cs.logits, T0)
    rerr = z(1, dt=torch.int32)
    def run_rt(T):
        i, w = z(T, KU, dt=torch.int32), z(T, KU); G.router(cs.logits, T, NEXP, KU, i, w, rerr); return i, w
    over_T('router (ids, weights; exact ties, one-ulp near ties, an all-equal token)', (ids, wts), run_rt)
    check(int(rerr.item()) == 0, f'G1 router: the routing flag word stays 0 on every T (ties, near ties, an all-equal token) (got {int(rerr.item()):#x})')
    # the tie rule, stated against a host re-derivation: on equal probability the lower expert index ranks first
    check(ids[5].tolist() == list(range(KU)), f'G1 router tie rule: a token with all logits equal picks experts 0..9 in order (got {ids[5].tolist()})')
    lgn = cs.logits.cpu().numpy(); ex = sum(1 for t in range(T0) if len(np.unique(lgn[t])) < NEXP)
    print(f'  (router inputs: {ex} tokens with an exact logit tie, {len(range(0, T0, 6))} with a one-ulp near tie among their top 12)', flush=True)
    # GDN conv (+ state)
    out, sts = dec.stateful('conv', cs.cx, cs.cw, cs.cst, T0, set(TS))
    def run_cv(T):
        st, y = cs.cst.clone(), z(T, cs.C); G.gdn_conv(cs.cx, cs.cw, st, y, cs.C, T); return y, st
    over_T(f'gdn_conv (C = {cs.C}, card 0\'s channels) + its state after T tokens', lambda T: (out[:T], sts[T]), run_cv)
    # PLE
    g, pg = dec.ple_gate(cs.kn, cs.qn, cs.val, T0)
    def run_pg(T):
        o, q = z(T, HC, D), z(T, HC); G.ple_gate(cs.kn, cs.qn, cs.val, o, q, T, HC); return o, q
    over_T('ple_gate (gated value, gate)', (g, pg), run_pg)
    k2 = cs.kn.clone(); q2 = z(T0, HC); G.ple_gate(k2, cs.qn, cs.val, k2, q2, T0, HC)
    check(same(k2, g) and same(q2, pg), 'G1 ple_gate in place (gated over the key) == K3 decode op, bitwise')
    # the prefill's PLE: the query norm recomputed from the residual (decode: rms_mul(res, ple_norm_query) -> ple_gate)
    qnorm = cs.put(wt.f32('blk.1.ple_norm_query.weight').reshape(HC, D)); qn_dec = dec.rms_mul(cs.res, qnorm, T0)
    g3, p3 = dec.ple_gate(cs.kn, qn_dec, cs.val, T0)
    def run_pg_rs(T):
        sq, o, q = z(T, HC), cs.kn[:T].clone(), z(T, HC)
        G.rms_mul(cs.res, qnorm, None, T * HC, EPS, scale=sq)
        G.ple_gate(o, cs.res, cs.val, o, q, T, HC, qscale=sq, qw=qnorm); return o, q
    over_T('rms_mul (scale only) -> ple_gate in place recomputing the query norm == K3 decode rms_mul -> ple_gate', (g3, p3), run_pg_rs)
    out, sts = dec.stateful('ple', cs.px, cs.pw, cs.phist, T0, set(TS))
    def run_pc(T):
        h, y = cs.phist.clone(), z(T, cs.pC); G.ple_conv(cs.px, cs.pw, h, cs.pC, T, dil=3, y=y); return y, h
    over_T('ple_conv (C = 10240, dilation 3) + its history after T tokens', lambda T: (out[:T], sts[T]), run_pc)
    ref = dec.add_sum(cs.pa, out, cs.res.reshape(T0, -1), T0)
    def run_as(T):
        y = cs.res.reshape(T0, -1)[:T].clone(); G.add_sum(cs.pa, out, y, T * cs.pC); return (y,)
    over_T('add_sum', (ref,), run_as)
    def run_pf(T):
        h, r = cs.phist.clone(), cs.res.reshape(T0, -1)[:T].clone(); G.ple_conv(cs.px, cs.pw, h, cs.pC, T, dil=3, a=cs.pa, res=r); return r, h
    over_T('ple_conv fused with add_sum (res += a + conv) == ple_conv then add_sum', lambda T: (ref[:T], sts[T]), run_pf)
    # the prefill's PLE conv: its input norm recomputed from the gated rows (decode: rms_mul(gated, ple_norm_conv) -> ple_conv
    # -> add_sum(gated, conv, res))
    cnorm = cs.put(wt.f32('blk.1.ple_norm_conv.weight').reshape(HC, D)); xc_dec = dec.rms_mul(cs.pa.reshape(T0, HC, D), cnorm, T0)
    out4, sts4 = dec.stateful('ple', xc_dec.reshape(T0, -1), cs.pw, cs.phist, T0, set(TS))
    ref4 = dec.add_sum(cs.pa, out4, cs.res.reshape(T0, -1), T0)
    def run_pf_rs(T):
        sc, h, r = z(T, HC), cs.phist.clone(), cs.res.reshape(T0, -1)[:T].clone()
        G.rms_mul(cs.pa, cnorm, None, T * HC, EPS, scale=sc)
        G.ple_conv(cs.pa, cs.pw, h, cs.pC, T, dil=3, a=cs.pa, res=r, xscale=sc, xw=cnorm); return r, h
    over_T('rms_mul (scale only) -> ple_conv recomputing its input norm, fused add_sum, history == K3 decode rms_mul -> ple_conv '
           '-> add_sum', lambda T: (ref4[:T], sts4[T]), run_pf_rs)
    # thin BF16 matrices
    for name, w, x in cs.thin:
        N, K = w.shape; ref = dec.bf16_mv(w, x, T0)
        def run_mv(T, w=w, x=x, N=N, K=K):
            o = z(T, N); G.bf16_mv(w, N, K, x, o, T); return (o,)
        over_T(f'bf16_mv {name} [{N} x {K}]', (ref,), run_mv)
        o = z(T0, N); G.bf16_mv(w, N, K, x, o, T0, rpb=1)
        check(same(o, ref), f'G1 bf16_mv {name}: one row per workgroup == the default grid, bitwise')
        if name == 'router':
            def run_rl(T, w=w, x=x, N=N):
                o = z(T, N); md = G.router_logits(w, N, x, o, T, mode='exact'); assert md == 'exact'; return (o,)
            over_T(f'router_logits mode exact [{N} x {K}] (the switch away from the WMMA default)', (ref,), run_rl)
    # the bf16 operands of the WMMA GEMMs
    x = cs.mixed; yb = z(po.tpad(777), D, dt=torch.bfloat16)
    G.to_bf16(x, yb, 777, D)
    check(same(yb[:777], x[:777].to(torch.bfloat16)) and not yb[777:].float().abs().sum().item(), 'G1 to_bf16 == torch bf16 (RNE), padding rows zero')
    lo = cs.mixed[:, :LR].contiguous(); sb = z(po.tpad(777), LR, dt=torch.bfloat16); G.silu_bf16(lo, sb, 777, LR, 0.25)
    l64 = lo[:777].double() * 0.25; s64 = l64 / (1 + torch.exp(-l64))
    tol = torch.abs(s64) * 2.0 ** -7                                       # >= one bf16 step at the value
    check(bool((torch.abs(sb[:777].double() - s64) <= tol).all().item()) and not sb[777:].float().abs().sum().item(),
          'G1 silu_bf16 within one bf16 step of fp64 silu(x / 4), padding rows zero')
    print(f'(g1 in {time.time() - t0:.0f} s)', flush=True)


# ---- stage: g2 ------------------------------------------------------------------------------------------------------------
class Ops:
    """every op of this step on one device (a Case's inputs), as a function of a chunk [lo, hi): the G2 / xcard runs"""
    def __init__(self, torch, dev, wt, seed=12):
        from caminus.prefill import glue as po
        self.torch, self.po, self.G, self.cs = torch, po, po.Glue(dev, wmma_out='fp32'), Case(torch, dev, wt, seed=seed)
        self.Gb = po.Glue(dev, wmma_out='bf16')   # the WMMA GEMMs with bf16-rounded output (fix:prefill-s6)
        self.z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
        bf = lambda name: bf_dev(torch, dev, wt.bf16(name))
        self.wd, self.wu, self.wk, self.wv = bf('blk.0.hc_attn_down.weight'), bf('blk.0.hc_attn_up.weight'), bf('blk.1.ple_key.weight'), bf('blk.1.ple_value.weight')
        self.qnorm = self.cs.put(wt.f32('blk.1.ple_norm_query.weight').reshape(HC, D)); self.cnorm = self.cs.put(wt.f32('blk.1.ple_norm_conv.weight').reshape(HC, D))

    def fresh(self):
        cs = self.cs
        return dict(conv=cs.cst.clone(), ple=cs.phist.clone(), ple_rs=cs.phist.clone())

    def __call__(self, lo, hi, carry):
        """every op on tokens [lo, hi) (one chunk); carry: the states between chunks (updated); {name: tuple of outputs}"""
        torch, po, G, cs, z = self.torch, self.po, self.G, self.cs, self.z
        T = hi - lo; T0 = cs.T0; o = {}
        sl = lambda t: t[lo:hi].contiguous()
        y, yb = z(T, HC, D), z(T, HC, D, dt=torch.bfloat16); G.rms_mul(sl(cs.res), cs.hcn, y, T * HC, EPS, ybf=yb); o['rms_mul'] = (y, yb)
        r, x = sl(cs.res).clone(), z(T, HC, D); G.hc_post_rms(sl(cs.blk), sl(cs.inj), r, cs.hcn, T, HC, EPS, xn=x); o['hc_post_rms'] = (r, x)
        r, xb2, sc = sl(cs.res).clone(), z(T, HC, D, dt=torch.bfloat16), z(T, HC)
        G.hc_post_rms(sl(cs.blk), sl(cs.inj), r, cs.hcn, T, HC, EPS, xnbf=xb2, scale=sc)
        m2, ij = z(T, D), z(T, HC); G.hc_mean(r, sl(cs.gate), m2, T, HC, scale=sc, w=cs.hcn)
        G.bf16_mv(cs.thin[-1][1], HC, HC * D, r.reshape(T, -1), ij, T, xscale=sc, xw=cs.hcn); o['hc_post_rms -> recomputing hc_mean / inject'] = (r, xb2, sc, m2, ij)
        r = sl(cs.res).clone(); G.hc_post(sl(cs.blk), sl(cs.inj), r, T, HC); o['hc_post'] = (r,)
        m = z(T, D); G.hc_mean(sl(cs.xn), sl(cs.gate), m, T, HC); o['hc_mean'] = (m,)
        i, w, er = z(T, KU, dt=torch.int32), z(T, KU), z(1, dt=torch.int32); G.router(sl(cs.logits), T, NEXP, KU, i, w, er); o['router'] = (i, w)
        assert int(er.item()) == 0, f'router flagged the G2 logits ({int(er.item()):#x})'
        y = z(T, cs.C); G.gdn_conv(sl(cs.cx), cs.cw, carry['conv'], y, cs.C, T); o['gdn_conv'] = (y,)
        g, p = z(T, HC, D), z(T, HC); G.ple_gate(sl(cs.kn), sl(cs.qn), sl(cs.val), g, p, T, HC); o['ple_gate'] = (g, p)
        sq, g2, p2 = z(T, HC), sl(cs.kn).clone(), z(T, HC); G.rms_mul(sl(cs.res), self.qnorm, None, T * HC, EPS, scale=sq)
        G.ple_gate(g2, sl(cs.res), sl(cs.val), g2, p2, T, HC, qscale=sq, qw=self.qnorm); o['ple_gate (query norm recomputed)'] = (g2, p2)
        y = z(T, cs.pC); G.ple_conv(sl(cs.px), cs.pw, carry['ple'], cs.pC, T, dil=3, y=y); o['ple_conv'] = (y,)
        scn, r2 = z(T, HC), cs.res.reshape(T0, -1)[lo:hi].clone(); G.rms_mul(sl(cs.pa), self.cnorm, None, T * HC, EPS, scale=scn)
        G.ple_conv(sl(cs.pa), cs.pw, carry['ple_rs'], cs.pC, T, dil=3, a=sl(cs.pa), res=r2, xscale=scn, xw=self.cnorm)
        o['ple_conv + add_sum (input norm recomputed)'] = (r2,)
        r = cs.res.reshape(T0, -1)[lo:hi].clone(); G.add_sum(sl(cs.pa), sl(cs.px), r, T * cs.pC); o['add_sum'] = (r,)
        for name, w, xx in cs.thin:
            N, K = w.shape; out = z(T, N); G.bf16_mv(w, N, K, sl(xx), out, T); o[f'bf16_mv {name}'] = (out,)
        # WMMA: hc down (A = the bf16 copy of xn rows), hc up (A = silu(lo / 4)), ple key / value (A = bf16 of a 2560 row)
        Tp = po.tpad(T); xb = z(Tp, HC * D, dt=torch.bfloat16); G.to_bf16(sl(cs.xbig), xb, T, HC * D)
        lo_ = z(Tp, LR); G.bf16_gemm(self.wd, LR, HC * D, xb, lo_, T); sb = z(Tp, LR, dt=torch.bfloat16); G.silu_bf16(lo_, sb, T, LR, 0.25)
        gt = z(Tp, HC * D); G.bf16_gemm(self.wu, HC * D, LR, sb, gt, T)
        mb = z(Tp, D, dt=torch.bfloat16); G.to_bf16(sl(cs.mixed), mb, T, D)
        pk = z(Tp, HC * D); G.bf16_gemm(self.wk, HC * D, D, mb, pk, T); pv = z(Tp, D); G.bf16_gemm(self.wv, D, D, mb, pv, T)
        o['wmma hc_down'] = (lo_[:T],); o['wmma hc_up'] = (gt[:T],); o['wmma ple_key'] = (pk[:T],); o['wmma ple_value'] = (pv[:T],)
        rl, rb_ = z(Tp, NEXP), z(Tp, D, dt=torch.bfloat16)
        md = G.router_logits(cs.thin[0][1], NEXP, sl(cs.mixed), rl, T, xbf=rb_); assert md == 'wmma', md
        o['router_logits (wmma)'] = (rl[:T], rb_[:T])
        # the same WMMA GEMMs with wmma_out 'bf16' (each result rounded to bf16): the hc up's input from this mode's hc down
        Gb = self.Gb
        lob = z(Tp, LR); Gb.bf16_gemm(self.wd, LR, HC * D, xb, lob, T); sbb = z(Tp, LR, dt=torch.bfloat16); Gb.silu_bf16(lob, sbb, T, LR, 0.25)
        gtb = z(Tp, HC * D); Gb.bf16_gemm(self.wu, HC * D, LR, sbb, gtb, T)
        pkb = z(Tp, HC * D); Gb.bf16_gemm(self.wk, HC * D, D, mb, pkb, T); pvb = z(Tp, D); Gb.bf16_gemm(self.wv, D, D, mb, pvb, T)
        o['wmma bf16-out hc_down'] = (lob[:T],); o['wmma bf16-out hc_up'] = (gtb[:T],); o['wmma bf16-out ple_key'] = (pkb[:T],)
        o['wmma bf16-out ple_value'] = (pvb[:T],)
        rlb, rbb = z(Tp, NEXP), z(Tp, D, dt=torch.bfloat16)
        md = Gb.router_logits(cs.thin[0][1], NEXP, sl(cs.mixed), rlb, T, xbf=rbb); assert md == 'wmma', md
        o['router_logits (wmma, bf16 out)'] = (rlb[:T], rbb[:T])
        return o


# ---- stage: g2 ------------------------------------------------------------------------------------------------------------
def stage_g2(torch, dev, wt):
    ops = Ops(torch, dev, wt); T0 = ops.cs.T0
    c1 = ops.fresh(); a = ops(0, T0, c1); c2 = ops.fresh(); b = ops(0, T0, c2)
    bad = [k for k in a if not all(same(x, y) for x, y in zip(a[k], b[k]))]
    bad += [k for k in c1 if not same(c1[k], c2[k])]
    check(not bad, f'G2a two runs bitwise: all {len(a)} ops + the conv state / PLE histories at T = {T0}' + (f' (differ: {bad})' if bad else ''))
    for sp in SPLITS:
        c = ops.fresh(); parts = {}; lo = 0
        for n in sp:
            o = ops(lo, lo + n, c); lo += n
            for k, v in o.items(): parts.setdefault(k, []).append(v)
        bad = [k for k in a if not all(same(torch.cat([p[i] for p in parts[k]]), a[k][i]) for i in range(len(a[k])))]
        bad += [k for k in c1 if not same(c[k], c1[k])]
        check(not bad, f'G2b chunks {" + ".join(map(str, sp))} == one T = {T0} pass, bitwise per position (all {len(a)} ops, WMMA included) '
              f'and the final conv state / PLE histories' + (f' (differ: {bad})' if bad else ''))


# ---- stage: xcard (both cards) ----------------------------------------------------------------------------------------
def stage_xcard(torch, wt):
    """the same inputs on the XTX (cuda:0) and the XT (cuda:1): every op gives the same bits (the replicated ops of TP: the
    routers of both cards must pick the same experts)"""
    res = {}
    for d in (0, 1):
        dev = torch.device(f'cuda:{d}'); torch.cuda.set_device(dev)
        ops = Ops(torch, dev, wt); c = ops.fresh(); o = ops(0, ops.cs.T0, c)
        res[d] = ({k: tuple(x.cpu() for x in v) for k, v in o.items()}, {k: v.cpu() for k, v in c.items()},
                  torch.cuda.get_device_name(dev))
        del ops, o, c; torch.cuda.synchronize(dev); torch.cuda.empty_cache()
    (a, ca, na), (b, cb, nb) = res[0], res[1]
    bad = [k for k in a if not all(same(x, y) for x, y in zip(a[k], b[k]))] + [k for k in ca if not same(ca[k], cb[k])]
    check(not bad, f'xcard: {na} and {nb} give the same bits on the same inputs, all {len(a)} ops (WMMA included) + states at T = 2048'
          + (f' (differ: {bad})' if bad else ''))


# ---- stage: wmma (tolerance against fp64 and llama.cpp's batched op) ------------------------------------------------------
class Ref:
    """gg_mmq_ref jobs (ggml_mul_mat on files) under dir, run at once; the files are emptied after the run (fixed names)"""
    def __init__(self, binary, d, tag):
        self.bin, self.d, self.tag, self.jobs, self.outs = binary, d, tag, [], []
        os.makedirs(d, exist_ok=True)

    def mm(self, key, w_u16, x, reps=0):
        n = len(self.jobs); o = os.path.join(self.d, f'{self.tag}_{n}.out'); rows, K = w_u16.shape
        wp, xp = os.path.join(self.d, f'{self.tag}_{n}.w'), os.path.join(self.d, f'{self.tag}_{n}.x')
        np.ascontiguousarray(w_u16).tofile(wp); np.ascontiguousarray(x, dtype=np.float32).tofile(xp)
        self.jobs.append(f'mm bf16 {rows} {K} {x.shape[0]} {wp} {xp} {o} {reps}'); self.outs.append((key, o, (x.shape[0], rows)))

    def run(self):
        jf = os.path.join(self.d, f'{self.tag}.jobs'); open(jf, 'w').write('\n'.join(self.jobs) + '\n')
        r = subprocess.run([self.bin, jf], capture_output=True, text=True, timeout=1800)
        print(r.stdout.splitlines()[0] if r.stdout else '', flush=True)
        if r.returncode != 0: raise RuntimeError(f'gg_mmq_ref failed ({r.returncode}): {r.stderr[-2000:]}')
        lines = [l for l in r.stdout.splitlines() if l.startswith('JOB ')]; assert len(lines) == len(self.outs), (len(lines), len(self.outs))
        res, times = {}, {}
        for (key, o, shape), l in zip(self.outs, lines):
            res[key] = np.fromfile(o, dtype=np.float32).reshape(shape)
            m = re.search(r'ms median ([\d.]+) min ([\d.]+)', l)
            if m: times[key] = (float(m.group(1)), float(m.group(2)))
        for f in os.listdir(self.d):
            if f.startswith(self.tag + '_') or f == self.tag + '.jobs': open(os.path.join(self.d, f), 'w').close()
        self.jobs, self.outs = [], []
        return res, times


def golden_rows(gdir, kind, n_steps=8):
    """the rows of one dumped tensor kind ('hc_norm' 10240, 'hc_mixed' 2560) over every layer and step of a gg_golden dump"""
    out = []
    for line in open(os.path.join(gdir, 'index.tsv')):
        f = line.rstrip('\n').split('\t')
        if int(f[0]) < n_steps and re.fullmatch(kind + r'-\d+', f[2]): out.append(np.fromfile(os.path.join(gdir, f[8]), dtype=np.float32))
    return np.stack(out)


def errs(y, ref):
    d = np.abs(y.astype(np.float64) - ref)
    return float(d.max() / max(np.abs(ref).max(), 1e-300)), float(d.mean() / max(np.abs(ref).mean(), 1e-300))


def stage_wmma(torch, dev, wt, ref_bin, gdir, scratch):
    from caminus.prefill import glue as po
    G = po.Glue(dev, wmma_out='fp32'); Gb = po.Glue(dev, wmma_out='bf16')
    dec = Dec(torch, dev); R = Ref(ref_bin, os.path.join(scratch, 'wmma'), 'wmma'); rng = np.random.default_rng(13)
    xn_real = golden_rows(gdir, 'hc_norm'); mx_real = golden_rows(gdir, 'hc_mixed')
    print(f'recorded activations: {xn_real.shape[0]} hc_norm rows, {mx_real.shape[0]} hc_mixed rows ({gdir}; the default dump is '
          f'REAP-308\'s: the same architecture, not the Coder\'s own activations)', flush=True)
    mats = [('hc_attn_down', 'blk.0.hc_attn_down.weight'), ('hc_ffn_down', 'blk.30.hc_ffn_down.weight'),
            ('output_hc_down', 'output_hc_down.weight'), ('hc_attn_up', 'blk.0.hc_attn_up.weight'),
            ('hc_ffn_up', 'blk.30.hc_ffn_up.weight'), ('output_hc_up', 'output_hc_up.weight'),
            ('ple_key', 'blk.1.ple_key.weight'), ('ple_value', 'blk.1.ple_value.weight'),
            ('indexer_k', 'blk.3.indexer.k_proj.weight'), ('indexer_q', 'blk.3.indexer.q_proj.weight'),
            ('router_logits', 'blk.0.ffn_gate_inp.weight')]
    down_of = {'hc_attn_up': 'blk.0.hc_attn_down.weight', 'hc_ffn_up': 'blk.30.hc_ffn_down.weight', 'output_hc_up': 'output_hc_down.weight'}
    cases = {}
    for tag, name in mats:
        w = wt.bf16(name); N, K = w.shape; wf = w.astype(np.uint32) << 16; wd = wf.view(np.float32).astype(np.float64)
        for src in ('real', 'random'):
            if tag in down_of:                                    # the hc up's input: silu(lo / 4), lo = an fp64 hc down of xn rows
                xin = xn_real if src == 'real' else acts(rng, 2048 * 10240, 0.05).reshape(2048, 10240)
                dw = wt.bf16(down_of[tag]); dwd = (dw.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
                lo = (xin.astype(np.float64) @ dwd.T).astype(np.float32); l = lo.astype(np.float64) * 0.25
                x = (l / (1 + np.exp(-l))).astype(np.float32)
            elif K == 10240: x = xn_real if src == 'real' else acts(rng, 2048 * K, 0.05).reshape(2048, K)
            else: x = mx_real if src == 'real' else acts(rng, 2048 * K, 0.5).reshape(2048, K)
            cases[(tag, src)] = (w, x, wd)
            R.mm((tag, src), w, x)
    gg, _ = R.run()
    rows = []; worse = []; rrows = []; rbad = []; brows = []; bbad = []
    for (tag, src), (w, x, wd) in cases.items():
        N, K = w.shape; T = x.shape[0]; Tp = po.tpad(T)
        ref = x.astype(np.float64) @ wd.T
        wt_ = bf_dev(torch, dev, w); xt = torch.from_numpy(np.ascontiguousarray(x)).to(dev)
        xb = torch.zeros(Tp, K, dtype=torch.bfloat16, device=dev); out = torch.zeros(Tp, N, device=dev)
        outb = torch.zeros(Tp, N, device=dev)
        if tag == 'router_logits':   # the prefill's router logits path itself (to_bf16 + the WMMA GEMM)
            assert G.router_logits(wt_, N, xt, out, T, xbf=xb) == 'wmma'
            assert Gb.router_logits(wt_, N, xt, outb, T, xbf=torch.zeros_like(xb)) == 'wmma'
            ex = torch.zeros(T, N, device=dev); G.router_logits(wt_, N, xt, ex, T, mode='exact')
            sets = {}
            for nm, lgt in (('wmma', out[:T].contiguous()), ('exact', ex), ('wmma_bf16out', outb[:T].contiguous()),
                            ('llama', torch.from_numpy(np.ascontiguousarray(gg[(tag, src)])).to(dev))):
                ii, ww, ee = torch.zeros(T, KU, dtype=torch.int32, device=dev), torch.zeros(T, KU, device=dev), torch.zeros(1, dtype=torch.int32, device=dev)
                G.router(lgt, T, NEXP, KU, ii, ww, ee); sets[nm] = np.sort(ii.cpu().numpy(), axis=1); assert int(ee.item()) == 0
            sets['fp64'] = np.sort(np.argsort(-ref, axis=1, kind='stable')[:, :KU], axis=1)
            agree = {f'{a}=={b}': float((sets[a] == sets[b]).all(axis=1).mean()) for a, b in (('wmma', 'exact'), ('wmma', 'fp64'), ('exact', 'fp64'),
                     ('wmma', 'llama'), ('wmma_bf16out', 'llama'), ('wmma_bf16out', 'fp64'))}
            print(f'  router top-{KU} sets ({src}, T={T}): share of tokens with the same experts: ' +
                  ', '.join(f'{k} {v * 100:.2f}%' for k, v in agree.items()), flush=True)
            NUMBERS.setdefault('router_topk_agreement', {})[src] = agree
        else:
            G.to_bf16(xt, xb, T, K); G.bf16_gemm(wt_, N, K, xb, out, T); Gb.bf16_gemm(wt_, N, K, xb, outb, T)
        mine = out[:T].cpu().numpy()
        # wmma_out 'bf16': each result of the fp32 mode rounded to bf16 (RNE), bitwise, on the bf16 grid; next to llama.cpp's
        # own batched op, which rounds its output the same way above 8 columns (gemm.hip RBF)
        mb_ = outb[:T].cpu().numpy(); u = mine.view(np.uint32).astype(np.uint64)
        want = ((((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16) & 0xFFFFFFFF).astype(np.uint32).view(np.float32)
        b_def = bool(np.array_equal(mb_.view(np.uint32), want.view(np.uint32)))
        b_eq = float((mb_.view(np.uint32) == gg[(tag, src)].view(np.uint32)).mean()); e_b = errs(mb_, ref)
        brows.append(dict(op=tag, src=src, T=T, rne_of_fp32=b_def, eq_llama=b_eq, err=e_b))
        if not b_def: bbad.append((tag, src))
        e_m, e_g = errs(mine, ref), errs(gg[(tag, src)], ref)
        e_d = errs(dec.bf16_mv(wt_, xt, min(T, 256)).cpu().numpy(), ref[:min(T, 256)])
        print(f'  {tag:15s} {src:6s} T={T:4d} [{N:5d} x {K:5d}]  max|err|/max|ref|, mean|err|/mean|ref| vs fp64:  WMMA {e_m[0]:.2e} {e_m[1]:.2e} | '
              f'llama.cpp batched {e_g[0]:.2e} {e_g[1]:.2e} | K3 decode (fp32 x) ' + (f'{e_d[0]:.2e} {e_d[1]:.2e}' if e_d[0] is not None else '-'), flush=True)
        if tag == 'router_logits':
            # the router logits (the prefill's default since 11-03) are reported against the bf16-activation floor: fp64 on the
            # bf16-rounded activations (torch's RNE == to_bf16, G1). Every bf16-activation GEMM, llama.cpp's included, has at
            # least this error; the WMMA must add only its fp32 accumulation to it. llama.cpp's op also rounds its OUTPUT to
            # bf16, which moves its error around the floor (either way): a comparison of maxima against it is noise at the
            # 10-20% level, so it is reported here, not gated (G4, end to end, decides the routing tier)
            flo = torch.from_numpy(np.ascontiguousarray(x)).to(torch.bfloat16).to(torch.float64).numpy() @ wd.T
            e_f, d_f = errs(flo, ref), errs(mine, flo)
            rrows.append(dict(op=tag, src=src, T=T, N=N, K=K, wmma=e_m, llama_batched=e_g, k3_decode=e_d, bf16_floor=e_f, wmma_vs_floor=d_f))
            print(f'  {"":15s} {src:6s} bf16-activation floor vs fp64 {e_f[0]:.2e} {e_f[1]:.2e} | WMMA vs the floor (its fp32 accumulation) '
                  f'{d_f[0]:.2e} {d_f[1]:.2e} | llama.cpp batched vs WMMA: max {e_g[0] / e_m[0]:.2f}x, mean {e_g[1] / e_m[1]:.2f}x', flush=True)
            print(f'  {"":15s} {src:6s} wmma_out bf16: == RNE(fp32 mode) {b_def}, == llama.cpp batched {b_eq * 100:.2f}% of elements, vs fp64 '
                  f'{e_b[0]:.2e} {e_b[1]:.2e}', flush=True)
            if not d_f[0] <= 1e-5: rbad.append(src)
            continue
        ok = e_m[0] <= e_g[0] and e_m[1] <= e_g[1]
        if not ok: worse.append((tag, src))
        rows.append(dict(op=tag, src=src, T=T, N=N, K=K, wmma=e_m, llama_batched=e_g, k3_decode=e_d))
        print(f'  {"":15s} {src:6s} wmma_out bf16: == RNE(fp32 mode) {b_def}, == llama.cpp batched {b_eq * 100:.2f}% of elements, vs fp64 '
              f'{e_b[0]:.2e} {e_b[1]:.2e} (llama.cpp {e_g[0]:.2e} {e_g[1]:.2e})', flush=True)
    NUMBERS['wmma_tolerance'] = rows; NUMBERS['wmma_router_logits'] = rrows; NUMBERS['wmma_bf16_out'] = brows
    check(not worse, f'WMMA BF16 GEMMs: error vs fp64 <= llama.cpp\'s own batched op (hipBLAS BF16) on the same inputs, both measures, '
          f'{len(rows)} cases' + (f' (worse: {worse})' if worse else ''))
    check(rrows and not rbad, f'WMMA router logits (wmma_out fp32): the bf16-activation floor plus fp32 accumulation only '
          f'(max |WMMA - floor| / max |floor| <= 1e-5: ' + ', '.join(f'{r["src"]} {r["wmma_vs_floor"][0]:.1e}' for r in rrows) + ')' +
          (f' (over: {rbad})' if rbad else ''))
    # the bf16-output mode is the fp32 mode (gated above against llama.cpp's op) followed by llama.cpp's own output rounding,
    # so it is gated by that definition, bitwise; its agreement with llama.cpp's op and its error vs fp64 are reported
    check(brows and not bbad, f'WMMA wmma_out bf16: every element == RNE to bf16 of the fp32 mode\'s, bitwise, {len(brows)} cases (router '
          f'logits included); == llama.cpp\'s batched op on ' + ', '.join(f'{r["op"]}/{r["src"]} {r["eq_llama"] * 100:.2f}%' for r in brows)
          + ' of elements' + (f' (not RNE: {bbad})' if bbad else ''))


# ---- stage: speed ---------------------------------------------------------------------------------------------------------
def stage_speed(torch, dev, wt, reps=20):
    from caminus.prefill import glue as po
    G = po.Glue(dev); T = 2048; Tp = po.tpad(T); rng = np.random.default_rng(14)
    print('device', torch.cuda.get_device_name(dev), '| compaction_proactiveness', open('/proc/sys/vm/compaction_proactiveness').read().strip(), flush=True)
    z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
    put = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    def tm(fn):
        fn(); torch.cuda.synchronize(dev); ts = []
        for _ in range(reps):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); fn(); e1.record(); torch.cuda.synchronize(dev); ts.append(e0.elapsed_time(e1))
        return float(np.median(ts))
    res = put(acts(rng, T * HC * D).reshape(T, HC, D)); xn = z(T, HC, D); xb = z(Tp, HC * D, dt=torch.bfloat16)
    hcn = put(wt.f32('blk.0.hc_attn_norm.weight')); blk = put(acts(rng, T * D, 0.3)); inj = put(acts(rng, T * HC))
    gate = z(Tp, HC * D); mixed = put(acts(rng, T * D, 0.5).reshape(T, D)); lo = z(Tp, LR); lob = z(Tp, LR, dt=torch.bfloat16)
    lg = put(rng.standard_normal((T, NEXP)).astype(np.float32)); ids, wts, rerr = z(T, KU, dt=torch.int32), z(T, KU), z(1, dt=torch.int32)
    bf = lambda name: bf_dev(torch, dev, wt.bf16(name))
    wd, wu, wi, wr = bf('blk.0.hc_attn_down.weight'), bf('blk.0.hc_attn_up.weight'), bf('blk.0.hc_attn_inject.weight'), bf('blk.0.ffn_gate_inp.weight')
    ws, al = bf('blk.0.ffn_gate_inp_shexp.weight'), wt.bf16('blk.0.ssm_alpha.weight')
    wk, wv, wiq, wik = bf('blk.1.ple_key.weight'), bf('blk.1.ple_value.weight'), bf('blk.3.indexer.q_proj.weight'), bf('blk.3.indexer.k_proj.weight')
    cw = put(wt.f32('blk.0.ssm_conv1d.weight').reshape(-1, 4)); pw = put(wt.raw('blk.1.ple_conv1d.weight'))
    out = {}
    sc = z(T, HC)
    out['rms_mul (+bf16 copy, scale)'] = tm(lambda: G.rms_mul(res, hcn, None, T * HC, EPS, ybf=xb, scale=sc))
    out['hc_post_rms (+bf16 copy, scale)'] = tm(lambda: G.hc_post_rms(blk, inj, res, hcn, T, HC, EPS, xnbf=xb, scale=sc))
    out['(hc_post_rms storing fp32 xn too)'] = tm(lambda: G.hc_post_rms(blk, inj, res, hcn, T, HC, EPS, xn=xn, xnbf=xb, scale=sc))
    out['hc_post'] = tm(lambda: G.hc_post(blk, inj, res, T, HC))
    out['wmma hc_down'] = tm(lambda: G.bf16_gemm(wd, LR, HC * D, xb, lo, T))
    io = z(T, HC); out['bf16_mv hc_inject [4 x 10240] (xn recomputed)'] = tm(lambda: G.bf16_mv(wi, HC, HC * D, res.reshape(T, -1), io, T, xscale=sc, xw=hcn))
    out['silu_bf16 (hc up input)'] = tm(lambda: G.silu_bf16(lo, lob, T, LR, 0.25))
    out['wmma hc_up'] = tm(lambda: G.bf16_gemm(wu, HC * D, LR, lob, gate, T))
    out['hc_mean (xn recomputed)'] = tm(lambda: G.hc_mean(res, gate, mixed, T, HC, scale=sc, w=hcn))
    # the A side of the A/B for storing no fp32 xn: the consumers reading a stored xn (interleaved in the same run)
    out['(hc_mean reading a stored fp32 xn)'] = tm(lambda: G.hc_mean(xn, gate, mixed, T, HC))
    out['(hc inject reading a stored fp32 xn)'] = tm(lambda: G.bf16_mv(wi, HC, HC * D, xn.reshape(T, -1), io, T))
    lgo = z(Tp, NEXP); out['router logits exact (bf16_mv) [256 x 2560]'] = tm(lambda: G.router_logits(wr, NEXP, mixed, lgo, T, mode='exact'))
    mb = z(Tp, D, dt=torch.bfloat16); G.to_bf16(mixed, mb, T, D)
    out['router logits default (wmma: to_bf16 + GEMM)'] = tm(lambda: G.router_logits(wr, NEXP, mixed, lgo, T, xbf=mb))
    out['(of it: the WMMA GEMM alone)'] = tm(lambda: G.bf16_gemm(wr, NEXP, D, mb, lgo, T))
    out['router (softmax + top-10)'] = tm(lambda: G.router(lg, T, NEXP, KU, ids, wts, rerr))
    sg = z(T, 1); out['bf16_mv shexp gate [1 x 2560]'] = tm(lambda: G.bf16_mv(ws, 1, D, mixed, sg, T))
    for c in (0, 1):
        a = bf_dev(torch, dev, al[rows_of(VRUNS[c])]); ao = z(T, a.shape[0])
        out[f'bf16_mv ssm alpha card {c} [{a.shape[0]} x 2560]'] = tm(lambda a=a, ao=ao: G.bf16_mv(a, a.shape[0], D, mixed, ao, T))
        sel = rows_of(QKV_R[c]); C = len(sel); cwc = cw[torch.from_numpy(sel).to(dev)].contiguous()
        cx = put(acts(rng, T * C).reshape(T, C)); cy = z(T, C); st = z(C, 3)
        out[f'gdn_conv card {c} (C {C}, + state)'] = tm(lambda cx=cx, cwc=cwc, st=st, cy=cy, C=C: G.gdn_conv(cx, cwc, st, cy, C, T))
    # PLE (layer 1 only)
    out['to_bf16 (PLE input)'] = tm(lambda: G.to_bf16(mixed, mb, T, D))
    out['wmma ple_key'] = tm(lambda: G.bf16_gemm(wk, HC * D, D, mb, gate, T))
    pv = z(Tp, D); out['wmma ple_value'] = tm(lambda: G.bf16_gemm(wv, D, D, mb, pv, T))
    pgo = z(T, HC); sq = z(T, HC)
    out['ple_gate (query norm recomputed)'] = tm(lambda: G.ple_gate(gate, res, pv, gate, pgo, T, HC, qscale=sq, qw=hcn))
    hist = z(HC * D, 9); res2 = res.clone()
    out['ple_conv + add_sum (+ history; input norm recomputed)'] = tm(lambda: G.ple_conv(gate, pw, hist, HC * D, T, dil=3, a=gate, res=res2, xscale=sq, xw=hcn))
    out['rms_mul (scale only)'] = tm(lambda: G.rms_mul(res, hcn, None, T * HC, EPS, scale=sq))
    out['rms_mul (in place, the PLE key)'] = tm(lambda: G.rms_mul(gate, hcn, gate, T * HC, EPS))
    # the indexer (S7 uses them: QSA layers only)
    iq = z(Tp, 512); out['wmma indexer q [512 x 2560]'] = tm(lambda: G.bf16_gemm(wiq, 512, D, mb, iq, T))
    ik = z(Tp, 128); out['wmma indexer k [128 x 2560]'] = tm(lambda: G.bf16_gemm(wik, 128, D, mb, ik, T))
    for k, v in out.items(): print(f'  SPEED {k:40s} {v:8.3f} ms  (T = {T})', flush=True)
    # one chunk of the 48 layers (the Coder: 36 GDN + 12 attention layers, PLE at layer 1, router + shared gate per layer)
    nl, ngdn = 48, 36
    per = {c: {
        'hc norms (2 rms_mul + 94 hc_post_rms)': 2 * out['rms_mul (+bf16 copy, scale)'] + 94 * out['hc_post_rms (+bf16 copy, scale)'],
        'hc_post (layer 0 before PLE, layer 47)': 2 * out['hc_post'],
        'hc down WMMA x 96': 96 * out['wmma hc_down'], 'hc inject x 96': 96 * out['bf16_mv hc_inject [4 x 10240] (xn recomputed)'],
        'silu_bf16 x 96': 96 * out['silu_bf16 (hc up input)'], 'hc up WMMA x 96': 96 * out['wmma hc_up'], 'hc_mean x 96': 96 * out['hc_mean (xn recomputed)'],
        'router logits x 48 (default: wmma)': nl * out['router logits default (wmma: to_bf16 + GEMM)'], 'router x 48': nl * out['router (softmax + top-10)'],
        'shared gate x 48': nl * out['bf16_mv shexp gate [1 x 2560]'],
        'ssm alpha + beta x 36': 2 * ngdn * out[f'bf16_mv ssm alpha card {c} [{30 if c == 0 else 18} x 2560]'],
        'gdn_conv x 36': ngdn * out[f'gdn_conv card {c} (C {6400 if c == 0 else 3840}, + state)'],
        'PLE layer (to_bf16, key, value, 3 rms, gate, conv + add)': out['to_bf16 (PLE input)'] + out['wmma ple_key'] + out['wmma ple_value'] +
            out['rms_mul (in place, the PLE key)'] + 2 * out['rms_mul (scale only)'] + out['ple_gate (query norm recomputed)'] +
            out['ple_conv + add_sum (+ history; input norm recomputed)']} for c in (0, 1)}
    xn_cost = 94 * (out['(hc_post_rms storing fp32 xn too)'] - out['hc_post_rms (+bf16 copy, scale)']) + \
        96 * (out['(hc_mean reading a stored fp32 xn)'] - out['hc_mean (xn recomputed)']) + \
        96 * (out['(hc inject reading a stored fp32 xn)'] - out['bf16_mv hc_inject [4 x 10240] (xn recomputed)'])
    print(f'  A/B storing fp32 xn (hc_post_rms + hc_mean + hc inject, same run): {xn_cost:+.1f} ms per chunk', flush=True)
    rl_ab = nl * (out['router logits exact (bf16_mv) [256 x 2560]'] - out['router logits default (wmma: to_bf16 + GEMM)'])
    print(f'  router logits, exact (bf16_mv) instead of the wmma default (same run): {rl_ab:+.1f} ms per chunk', flush=True)
    for c in (0, 1):
        tot = sum(per[c].values())
        print(f'  CHUNK card-{c} shapes on this device: S3 ops of the 48 layers at T = {T}: {tot:.1f} ms', flush=True)
        for k, v in per[c].items(): print(f'      {k:58s} {v:7.2f} ms ({100 * v / tot:4.1f}%)', flush=True)
    NUMBERS['speed'] = dict(device=torch.cuda.get_device_name(dev), ops_ms=out, chunk_ms={c: sum(per[c].values()) for c in (0, 1)}, xn_store_ms=xn_cost, router_exact_ms=rl_ab,
                            chunk_parts={c: per[c] for c in (0, 1)}, compaction=open('/proc/sys/vm/compaction_proactiveness').read().strip())


# ---- stage: vram (CPU) ----------------------------------------------------------------------------------------------------
def stage_vram(coder):
    from caminus.device import capacity
    from caminus.prefill import glue as po
    g = capacity.geometry(coder)
    for T in (1024, 2048):
        b = [po.glue_bytes(T, po.glue_dims(g, c)) / capacity.MIB for c in (0, 1)]
        need = capacity.prefill_need_mib(g, T, gdn_keys=10, moe_blocks=12)
        print(f'  glue workspace at chunk {T}: XTX {b[0]:.1f} / XT {b[1]:.1f} MiB; prefill_need_mib (S2 workspace + feed + glue) '
              f'{need[0]:.1f} / {need[1]:.1f} MiB', flush=True)
        if T == 2048: NUMBERS['vram'] = dict(glue_mib=b, prefill_need_mib=need, layout={k: v / capacity.MIB for k, v in po.glue_layout(T, po.glue_dims(g, 0)).items()})
    s2 = capacity.prefill_workspace_bytes(g, 2048, gdn_keys=10, moe_blocks=12); f = capacity.prefill_feed_bytes(g, 2048)
    gl = capacity.prefill_glue_bytes(g, 2048, gdn_keys=10); need = capacity.prefill_need_mib(g, 2048, gdn_keys=10, moe_blocks=12)
    xb, ab = capacity.prefill_xchg_bytes(g, 2048), capacity.prefill_attn_bytes(g, 2048)
    check(all(abs(need[c] - (s2[c] + f[c] + gl[c] + xb[c] + ab[c]) / capacity.MIB - capacity.PREFILL_CODE_MIB[c] - capacity.PREFILL_STREAMS_MIB[c]) < 1e-6 for c in (0, 1)),
          'vram: capacity.prefill_need_mib = S2 GEMM workspace + feed + S3 glue workspace + S5 exchange receive buffer + S6 attention workspace + the prefill modules and streams, per card')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('coder'); ap.add_argument('stages', nargs='+')
    ap.add_argument('--scratch', default=None)   # default $ARTIFACTS/impl-prefill-s3/scratch
    ap.add_argument('--ref-stock', default=os.path.join(ROOT, 'build', 'gg_mmq_ref'))
    ap.add_argument('--golden', default=os.path.join(ART, 'golden_layers'))
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    stages = []
    for s in a.stages: stages += ['g1', 'g2', 'wmma', 'speed'] if s == 'gpu' else [s]
    rd = GGUFReader(a.coder, 'r'); wt = W(rd)
    if 'vram' in stages: stage_vram(a.coder)
    if 'xcard' in stages:
        import torch
        stage_xcard(torch, wt)
    gpu = [s for s in stages if s in ('g1', 'g2', 'wmma', 'speed')]
    if gpu:
        import torch
        dev = torch.device('cuda:0'); torch.cuda.set_device(dev)
        print('device', torch.cuda.get_device_name(dev), flush=True)
        if 'g1' in gpu: stage_g1(torch, dev, wt)
        if 'g2' in gpu: stage_g2(torch, dev, wt)
        if 'wmma' in gpu: stage_wmma(torch, dev, wt, a.ref_stock, a.golden, a.scratch or devpaths.need('ARTIFACTS', 'impl-prefill-s3', 'scratch'))
        if 'speed' in gpu: stage_speed(torch, dev, wt)
    if a.json:
        json.dump(dict(fails=FAILS, numbers=NUMBERS), open(a.json, 'w'), indent=1, default=str)
    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'))
    sys.exit(0 if not FAILS else 1)


if __name__ == '__main__':
    main()
