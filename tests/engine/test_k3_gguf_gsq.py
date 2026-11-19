"""Gate G1 of the GSQ port (docs/specs/spec-gsq-port.md S1): the GSQ-RCO Coder's (op, format) instances in the K3 engine
(build/k3_gguf_gsq{,_mt}{,_trace}.hsaco, kernels/k3_gguf_gsq.hip), on real Coder tensors read with O_DIRECT (shard 1 only):
  MV       Q4_K Q5_K Q6_K IQ4_XS IQ4_NL Q8_0 (full matrices and the TP column splits), q8_1 and fp32 (AQ) activations
  MOE_GU   IQ2_S IQ3_XXS IQ3_S IQ4_XS (640 rows per expert and the TP row splits 384 / 256), h fp32 and h q8_1
  MOE_DOWN IQ4_NL Q2_0 (K 640 and the TP column splits 384 / 256)
checks
  (a) the engine op is bitwise the standalone kernels (gg_mv_*; MoE: per-expert gg_mv_* composed with the engine's own
      silu (gg_silu_mul) or summed in llama.cpp's mul-then-add order) at 3 / 5 / 6 / 7 / 8 workgroups per WGP, every card;
      Every case also checks build/gg_mv_ref.hsaco (llama.cpp's matvec as a test object: the G-acc yardstick, which stays
      when the engine modules stop including the ggml headers) bitwise the module's standalone gg_mv_* and quantizer.
  (b) T = 2 / 3 / 4 (the _mt module, named explicitly: k3.module_for sends k3_gguf_gsq at T > 1 to k3_gguf_gsq_qsa_mt; T = 2
      and 3 leave partial token groups) rows are bitwise the T = 1 rows (different
      expert ids per token, some repeated);
  (c) float64 (gguf-py's dequantizer, the same q8_1 activations), over EVERY row of the MV matrices and EVERY expert of the
      MoE tensors at full width: rel = max |out - ref| / sum |w| |xq| (as tests/engine/test_gguf_mv.py)
      - as written in the spec: rel <= 1e-6 against the exact dequantized dot. Passes for 10 of the 12 (op, format) pairs.
        It cannot pass for IQ2_S and IQ3_XXS: llama.cpp's MMVQ vec_dots apply the sub-block scale (0.5 + ls) with integer
        truncation (vecdotq.cuh: IQ2_S sumi = (s0*ls0 + s1*ls1 + (s0+s1)/2)/4; IQ3_XXS sumi = (ls*s + s/2)/2), and K3
        must stay bitwise with llama.cpp (G3), so for those two the exact-dot rel is reported (with the number of
        (expert, tensor) slices over 1e-6), not gated;
      - PROPOSED for IQ2_S / IQ3_XXS (needs the lead's sign-off): (c1) rel <= 1e-6 against a float64 reference with that
        integer step emulated; (c2) per row, |out - exact| <= B + 1e-6 sum |w| |xq|, with B the analytic truncation bound
        sum over 32-value sub-blocks of c |d d8| (c = 7/8 IQ2_S: the /2 truncation <= 1/2, divided by 4, plus the /4
        truncation <= 3/4; c = 3/4 IQ3_XXS: (1/2)/2 + 1/2). The derivation is checked per sub-block in float64.
  (d) unlisted (op, format) pairs: build() raises, run() refuses, and a forced launch (refuse cleared) leaves a sentinel
      output untouched, for MV, MOE_GU and MOE_DOWN at T = 1 and 4.
  (e) docs/specs/spec-own-quant.md's G-acc on recorded activations (run_gacc; --ops GACC --acts FILE): llama.cpp's error
      per tensor and pooled per (op, format), on sampled real rows of every (tensor class, format) and the vectors
      recorded at that op site: the bar each pair's own kernel must meet when it moves (--gacc-out writes it as JSON).
      The recording (record(); --record FILE) runs FlashNextTP on both cards under K3_TEST_HOOKS=gacc-sites.
  tools/guard.sh LOG 900 python -I tests/engine/test_k3_gguf_gsq.py [CODER-00001-of-00002.gguf] [--cards 0,1] [--wgp 3,5,6,7,8]
      [--module k3_gguf_gsq | k3_gguf_gsq_trace] [--ops MV,MOE_GU,MOE_DOWN,REFUSE] [--c-cards 0,1]
  K3_TEST_HOOKS=gacc-sites tools/guard.sh LOG 900 python -I tests/engine/test_k3_gguf_gsq.py --record ACTS.pt [--positions 24]
      [--tokens FILE,FILE,...]   (both cards; refuses an ACTS.pt that exists)
  HIP_VISIBLE_DEVICES=<one card> tools/guard.sh LOG 900 python -I tests/engine/test_k3_gguf_gsq.py --ops GACC --acts ACTS.pt
      --cards 0 [--gacc-out GACC.json] [--gacc-layers ends | all]"""
import ctypes, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tools', 'gates'))
import numpy as np
import torch
from caminus.engine import gguf as G, k3
from caminus.engine.flashnext import DirectReader
from gguf import quants, GGML_QUANT_SIZES
import gacc  # noqa: E402  (tools/gates/gacc.py: G-acc's shared pieces)

a = sys.argv[1:]
def opt(name, default):
    if name in a:
        i = a.index(name); v = a[i + 1]; del a[i:i + 2]; return v
    return default
CARDS = [int(v) for v in opt('--cards', '0,1').split(',')]
C_CARDS = [int(v) for v in opt('--c-cards', ','.join(map(str, CARDS))).split(',')]   # cards that run the float64 check (c)
WGP = [int(v) for v in opt('--wgp', '3,5,6,7,8').split(',')]
ONLY = opt('--only', None)   # debugging: only cases whose label contains this
OPS = opt('--ops', 'MV,MOE_GU,MOE_DOWN,REFUSE').split(',')
M1 = opt('--module', 'k3_gguf_gsq'); assert M1 in ('k3_gguf_gsq', 'k3_gguf_gsq_trace'), M1
MT = M1.replace('k3_gguf_gsq', 'k3_gguf_gsq_mt')
RECORD = opt('--record', None)        # (e)'s recording: write the recorded activations here (both cards), then exit
ACTS = opt('--acts', None)            # (e): the recording to read
POSITIONS = int(opt('--positions', '24'))   # teacher-forced positions per text (3 texts: 72 vectors per site)
TOKENS = opt('--tokens', None)        # comma-separated token files (default: the three short decode texts of the KLD floor gate, in $ARTIFACTS)
GACC_OUT = opt('--gacc-out', None)    # (e): write llama.cpp's G-acc per tensor and per pair here (JSON; the stored baseline)
GACC_LAYERS = opt('--gacc-layers', 'ends')   # (e): per (tensor class, format) its first and last layer ('ends') or 'all'
from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
PATH = a[0] if a else devpaths.coder()
RD = G.GGUFReader(PATH, 'r'); TT = {t.name: t for t in RD.tensors}
DR = DirectReader(PATH)
D, K_USED = 2560, 10
TS = (2, 3, 4)
fails = []


def check(cond, what):
    if not cond: fails.append(what); print('  FAIL ' + what, flush=True)
    return cond


def full(name):
    """the whole tensor's raw bytes on cuda:0 (O_DIRECT), shaped [..., row bytes]"""
    t = TT[name]; w = DR.read(int(t.data_offset), int(t.n_bytes))
    return w.view(*t.data.shape), t.tensor_type


def ncols(w, qt):
    bs, ts = GGML_QUANT_SIZES[qt]; return w.shape[-1] // ts * bs


def _tdiv(a, b):
    """C integer division (truncates toward zero)"""
    return np.sign(a) * (np.abs(a) // b)


INT_ROUNDED = {'IQ2_S': 7 / 8, 'IQ3_XXS': 3 / 4}   # format -> c, the per-32-value truncation bound (in sumi units)


def int_ref(raw, qt, W, q, d8):
    """float64 reference with llama.cpp's integer scale step emulated (the integer grid values recovered from gguf-py's
    dequantized weights), and the analytic bound B [n] of |integer result - exact| per row. Checks the derivation: per
    32-value sub-block, |sumi_int - sumi_exact| <= c."""
    n, K = W.shape; nb = K // 256; bs = GGML_QUANT_SIZES[qt][1]; blk = raw.reshape(n, nb, bs)
    d = blk[..., :2].copy().view(np.float16)[..., 0].astype(np.float64)                     # [n, nb]
    q = q.reshape(nb, 8, 32); d8 = d8.reshape(nb, 8)
    if qt.name == 'IQ2_S':   # block_iq2_s: d, qs[64], qh[8], scales[8] (low nibble: first 16 values, high: second 16)
        sc = blk[..., 74:82].astype(np.int64); ls = np.stack([sc & 15, sc >> 4], -1)         # [n, nb, 8, 2]
        s = d[..., None, None] * (0.5 + ls) * 0.25
        g = W.reshape(n, nb, 8, 2, 16) / np.where(s == 0, 1, s)[..., None]
        gi = np.rint(g).astype(np.int64); assert np.abs(g - gi).max() < 1e-6, 'IQ2_S: grid values not integral'
        p = (gi * q.reshape(nb, 8, 2, 16)[None]).sum(-1)                                       # [n, nb, 8, 2]
        sumi = _tdiv(p[..., 0] * ls[..., 0] + p[..., 1] * ls[..., 1] + _tdiv(p[..., 0] + p[..., 1], 2), 4)
        exact = (p[..., 0] * (ls[..., 0] + 0.5) + p[..., 1] * (ls[..., 1] + 0.5)) / 4
    else:                    # block_iq3_xxs: d, 64 grid bytes, 8 uint32 (signs + the 4-bit scale in the top bits)
        aux = blk[..., 66:98].copy().view(np.uint32).astype(np.int64); ls = aux >> 28        # [n, nb, 8]
        s = d[..., None] * (0.5 + ls) * 0.5
        g = W.reshape(n, nb, 8, 32) / np.where(s == 0, 1, s)[..., None]
        gi = np.rint(g).astype(np.int64); assert np.abs(g - gi).max() < 1e-6, 'IQ3_XXS: grid values not integral'
        si = (gi * q[None]).sum(-1)
        sumi = _tdiv(ls * si + _tdiv(si, 2), 2)
        exact = si * (ls + 0.5) / 2
    c = INT_ROUNDED[qt.name]
    dev = np.abs(sumi - exact).max()
    assert dev <= c + 1e-12, f'{qt.name}: truncation {dev} > analytic bound {c}'
    ref = (d[..., None] * d8[None] * sumi).reshape(n, -1).sum(-1)
    B = c * (np.abs(d)[..., None] * np.abs(d8)[None]).reshape(n, -1).sum(-1)
    return ref, B, float(dev)


def fp64(raw, qt, xq_bytes, out, chunk=2048):
    """(c) over every row of raw [n, rb] (uint8 numpy) with out [n] (float64 numpy):
    {'rel': exact rel, and for INT_ROUNDED 'rel_int', 'bound' (max |out - exact| / (B + 1e-6 S)), 'trunc' (max sub-block
    truncation, sumi units)}"""
    r = {'rel': 0.0}
    for i in range(0, raw.shape[0], chunk):
        rw = raw[i:i + chunk]; o = out[i:i + chunk]; n = rw.shape[0]
        W = quants.dequantize(rw, qt).astype(np.float64).reshape(n, -1); K = W.shape[1]
        yb = xq_bytes.reshape(-1, 36)[:K // 32]
        q = yb[:, 4:].view(np.int8).astype(np.int64); d8 = yb[:, :2].copy().view(np.float16)[:, 0].astype(np.float64)
        x = (q * d8[:, None]).reshape(-1)
        S = np.maximum(np.abs(W) @ np.abs(x), 1e-30); err = np.abs(o - W @ x)
        r['rel'] = max(r['rel'], float(np.max(err / S)))
        if qt.name in INT_ROUNDED:
            ref, B, dev = int_ref(rw, qt, W, q, d8)
            r['rel_int'] = max(r.get('rel_int', 0.0), float(np.max(np.abs(o - ref) / S)))
            r['bound'] = max(r.get('bound', 0.0), float(np.max(err / (B + 1e-6 * S))))
            r['trunc'] = max(r.get('trunc', 0.0), dev)
    return r


class Worst:
    def __init__(self): self.v = {}; self.slices = {}
    def add(self, op, fmt, r, card, what):
        """record and gate one slice's fp64() result"""
        for k, x in r.items():
            key = (op, fmt if k == 'rel' else f'{fmt} {k}'); self.v[key] = max(self.v.get(key, 0.0), x)
        n, over = self.slices.get((op, fmt), (0, 0)); self.slices[(op, fmt)] = (n + 1, over + (r['rel'] > 1e-6))
        if fmt in INT_ROUNDED:   # as written: reported only (see the docstring); gated on the proposed (c1) + (c2)
            check(r['rel_int'] <= 1e-6, f'card {card} {op} {fmt} {what}: standalone vs float64 int-emulated {r["rel_int"]:.2e} > 1e-6')
            check(r['bound'] <= 1.0, f'card {card} {op} {fmt} {what}: |out - exact| over the analytic bound ({r["bound"]:.3f})')
        else:
            check(r['rel'] <= 1e-6, f'card {card} {op} {fmt} {what}: standalone vs float64 {r["rel"]:.2e} > 1e-6')


WORST = Worst()


def grid(module, n):
    return k3.resident_grid(module, n)


def np_(t):
    return t.cpu().numpy()


def ref_q8(x):
    """x fp32 [n] (cuda) -> q8_1 blocks with build/gg_mv_ref.hsaco's quantizer (tools/gates/gacc.py)"""
    return gacc.ref_q8_1(G._m('gg_mv_ref'), x)


def ref_mv(w, qt, n, K, yq):
    """llama.cpp's matvec from build/gg_mv_ref.hsaco (the yardstick that stays when the engine modules drop their gg_mv_*):
    w raw blocks of n contiguous rows (uint8, cuda), yq q8_1 -> fp32 [n]"""
    out = torch.empty(n, dtype=torch.float32, device=w.device)
    getattr(G._m('gg_mv_ref'), 'gg_ref_mv_' + qt.name.lower())(((n + 3) // 4,), (32, 4), w, yq, out, ctypes.c_int(n), ctypes.c_int(K))
    return out


# ---- MV ----
MV_CASES = [  # (name, column-block splits (None = whole rows)); one per format, plus the TP K splits
    ('blk.0.attn_gate.weight', None), ('blk.5.ssm_out.weight', [(0, 15), (15, 24)]), ('blk.3.attn_output.weight', [(0, 12), (12, 24)]),   # Q4_K
    ('blk.1.attn_qkv.weight', None), ('blk.18.ssm_out.weight', [(0, 15), (15, 24)]),                                            # Q5_K
    ('blk.0.attn_qkv.weight', None), ('blk.0.ssm_out.weight', [(0, 15), (15, 24)]),                                             # Q6_K
    ('blk.2.attn_gate.weight', None), ('blk.7.attn_k.weight', None), ('blk.0.ffn_up_shexp.weight', None),                      # IQ4_XS
    ('blk.0.ffn_down_shexp.weight', None), ('blk.0.ffn_down_shexp.weight', [(0, 12), (12, 20)]),                              # IQ4_NL
    ('blk.47.ffn_down_shexp.weight', None), ('blk.47.ffn_down_shexp.weight', [(0, 12), (12, 20)]),                           # Q8_0
]


def mv_cases():
    for name, splits in MV_CASES:
        w, qt = full(name)
        if splits is None: cs = [(name, w)]
        else: cs = [(f'{name}[blocks {b0}:{b1}]', gacc.block_cols(w, GGML_QUANT_SIZES[qt][1], b0, b1)) for b0, b1 in splits]
        for label, wc in cs:
            if ONLY is None or ONLY in label: yield label, wc, qt


def run_mv(card):
    dev = f'cuda:{card}'; n_ok = n_all = 0
    for label, w0, qt in mv_cases():
        w = w0.to(dev); N, rb = w.shape; K = ncols(w, qt); fmt = qt.name
        gen = torch.Generator(device=dev).manual_seed(1000 + N + K)
        X = torch.randn(4, K, device=dev, generator=gen) * 0.5
        xq = [G.quantize_q8_1(X[t]) for t in range(4)]
        ref = [G.matvec(w, qt, N, K, xq[t]) for t in range(4)]
        n_all += 1; n_ok += check(torch.equal(ref_q8(X[0]), xq[0]) and torch.equal(ref_mv(w, qt, N, K, xq[0]), ref[0]),
                                  f'card {card} MV {fmt} {label}: gg_mv_ref != the module\'s standalone gg_mv / quantizer')
        rel = None
        if card in C_CARDS:   # (c) standalone vs float64, every row, tokens 0 and 3
            raw = np_(w)
            for t in (0, 3):
                r = fp64(raw, qt, np_(xq[t]), np_(ref[t].double())); WORST.add('MV', fmt, r, card, f'{label} token {t}')
                rel = max(rel or 0.0, r['rel'])
        modes = ['q8'] + (['f32'] if K <= 32 * k3.AQ_BLOCKS else [])
        for nw in WGP:
            for mode in modes:     # (a) T = 1, each per-WGP grid, both activation sources
                e = k3.Engine(module=M1, grid=grid(M1, nw)); out = torch.full((N,), float('nan'), device=dev)
                e.gg_matvec(w.reshape(-1), fmt, N, K, rb, xq[0] if mode == 'q8' else X[0].contiguous(), out, rows_per_task=16)
                e.build().run()
                n_all += 1; n_ok += check(torch.equal(out, ref[0]), f'card {card} MV {fmt} {label} wgp {nw} {mode}: engine != standalone')
            for T in TS:           # (b) T tokens (multi-token module) vs T = 1 rows
                e = k3.Engine(module=MT, grid=grid(MT, nw), T=T); assert e.module == MT
                out4 = torch.full((T * N,), float('nan'), device=dev)
                e.gg_matvec(w.reshape(-1), fmt, N, K, rb, torch.cat(xq[:T]), out4, rows_per_task=16)
                e.build().run()
                n_all += 1; n_ok += check(all(torch.equal(out4[t * N:(t + 1) * N], ref[t]) for t in range(T)),
                                          f'card {card} MV {fmt} {label} wgp {nw} T={T}: rows != T=1')
        print(f'  card {card} MV {fmt:6s} {label:42s} N {N:5d} K {K:5d}: float64 rel {"-" if rel is None else f"{rel:.1e}"} (all rows)', flush=True)
    return n_ok, n_all


# ---- MoE ----
GU_LAYERS = {'IQ2_S': 1, 'IQ3_XXS': 0, 'IQ3_S': 17, 'IQ4_XS': 47}
DOWN_LAYERS = {'IQ4_NL': 0, 'Q2_0': 1}
IDS = [[0, 255, 17, 42, 99, 150, 151, 200, 7, 128], [255, 3, 17, 64, 0, 77, 190, 201, 33, 128],
       [5, 6, 7, 8, 9, 250, 251, 252, 253, 254], [17, 0, 255, 42, 99, 1, 2, 150, 200, 128]]   # T = 4: repeats across tokens


def standalone_slab(stack, e, rows, qt, xq, K):
    return G.matvec(stack[e].reshape(-1), qt, rows, K, xq)


def run_moe_gu(card):
    dev = f'cuda:{card}'; n_ok = n_all = 0
    for fmt, il in GU_LAYERS.items():
        g0, qt = full(f'blk.{il}.ffn_gate_exps.weight'); u0, qt2 = full(f'blk.{il}.ffn_up_exps.weight'); assert qt == qt2
        for r0, r1 in ((0, 640), (0, 384), (384, 640)):          # whole experts, then the TP row split (moe_blocks 12)
            g = g0[:, r0:r1].contiguous().to(dev); u = u0[:, r0:r1].contiguous().to(dev)
            E, F, rb = g.shape; slab = F * rb
            gen = torch.Generator(device=dev).manual_seed(2000 + F)
            X = torch.randn(4, D, device=dev, generator=gen) * 0.5
            xq = [G.quantize_q8_1(X[t]) for t in range(4)]
            ids = [torch.tensor(v, dtype=torch.int32, device=dev) for v in IDS]
            gs = [[standalone_slab(g, int(e), F, qt, xq[t], D) for e in IDS[t]] for t in range(4)]
            us = [[standalone_slab(u, int(e), F, qt, xq[t], D) for e in IDS[t]] for t in range(4)]
            href = [G.silu_mul(torch.cat(gs[t]), torch.cat(us[t])) for t in range(4)]
            e0 = IDS[0][0]
            n_all += 1; n_ok += check(torch.equal(ref_mv(g[e0].reshape(-1), qt, F, D, xq[0]), gs[0][0]),
                                      f'card {card} MOE_GU {fmt} rows {r0}:{r1}: gg_mv_ref != the module\'s standalone gg_mv')
            if (r0, r1) == (0, 640) and card in C_CARDS:   # (c) every expert of gate and up, every row, against float64
                x0 = np_(xq[0])
                for kind, st in (('gate', g), ('up', u)):
                    for e in range(E):
                        o = np_(standalone_slab(st, e, F, qt, xq[0], D).double())
                        WORST.add('MOE_GU', fmt, fp64(np_(st[e]), qt, x0, o), card, f'layer {il} {kind} expert {e}')
                print(f'  card {card} MOE_GU {fmt:7s} layer {il:2d}: float64 over {E} experts x gate/up x {F} rows done', flush=True)
            for nw in WGP:
                for mode in ('q8', 'f32'):
                    for hq8 in (False, True):
                        e = k3.Engine(module=M1, grid=grid(M1, nw))
                        h = torch.full((K_USED * F // 32 * 36,), 0, dtype=torch.uint8, device=dev) if hq8 else torch.full((K_USED * F,), float('nan'), device=dev)
                        e.gg_moe_gu(g.reshape(-1), u.reshape(-1), fmt, slab, F, D, ids[0], xq[0] if mode == 'q8' else X[0].contiguous(), h,
                                    rows_per_task=32 if hq8 else 8, h_q8=hq8, up_fmt=fmt)
                        e.build().run()
                        good = torch.equal(h, G.quantize_q8_1(href[0])) if hq8 else torch.equal(h, href[0])
                        n_all += 1; n_ok += check(good, f'card {card} MOE_GU {fmt} rows {r0}:{r1} wgp {nw} {mode} h_q8 {hq8}: engine != standalone')
                for T in TS:
                    e = k3.Engine(module=MT, grid=grid(MT, nw), T=T); assert e.module == MT
                    h4 = torch.full((T * K_USED * F,), float('nan'), device=dev)
                    e.gg_moe_gu(g.reshape(-1), u.reshape(-1), fmt, slab, F, D, torch.cat(ids[:T]), torch.cat(xq[:T]), h4, rows_per_task=8, up_fmt=fmt)
                    e.build().run()
                    n_all += 1; n_ok += check(all(torch.equal(h4[t * K_USED * F:(t + 1) * K_USED * F], href[t]) for t in range(T)),
                                              f'card {card} MOE_GU {fmt} rows {r0}:{r1} wgp {nw} T={T}: rows != T=1')
            print(f'  card {card} MOE_GU {fmt:7s} layer {il:2d} rows {r0}:{r1} (F {F}): done', flush=True)
            del g, u
    return n_ok, n_all


def run_moe_down(card):
    dev = f'cuda:{card}'; n_ok = n_all = 0
    for fmt, il in DOWN_LAYERS.items():
        d0, qt = full(f'blk.{il}.ffn_down_exps.weight')
        qk = GGML_QUANT_SIZES[qt][0]
        for c0, c1 in ((0, 640), (0, 384), (384, 640)):          # whole rows, then the TP K split (moe_blocks 12)
            dw = gacc.block_cols(d0, GGML_QUANT_SIZES[qt][1], c0 // qk, c1 // qk).to(dev); E, R, rb = dw.shape; K = c1 - c0; slab = R * rb
            gen = torch.Generator(device=dev).manual_seed(3000 + K)
            Hx = torch.randn(4, K_USED * K, device=dev, generator=gen) * 0.5
            hq = [G.quantize_q8_1(Hx[t]) for t in range(4)]
            wts = [torch.softmax(torch.randn(K_USED, device=dev, generator=gen), 0).contiguous() for _ in range(4)]
            ids = [torch.tensor(v, dtype=torch.int32, device=dev) for v in IDS]
            kb = K // 32 * 36
            ref = []
            for t in range(4):
                v = [standalone_slab(dw, IDS[t][j], R, qt, hq[t][j * kb:(j + 1) * kb], K).cpu().numpy() for j in range(K_USED)]
                w_ = wts[t].cpu().numpy(); acc = (v[0] * w_[0]).astype(np.float32)
                for j in range(1, K_USED): acc = (acc + (v[j] * w_[j]).astype(np.float32)).astype(np.float32)
                ref.append(torch.from_numpy(acc).to(dev))
            e0 = IDS[0][0]
            n_all += 1; n_ok += check(torch.equal(ref_mv(dw[e0].reshape(-1), qt, R, K, hq[0][:kb]), standalone_slab(dw, e0, R, qt, hq[0][:kb], K)),
                                      f'card {card} MOE_DOWN {fmt} K {K}: gg_mv_ref != the module\'s standalone gg_mv')
            if card in C_CARDS:   # (c) every expert, every row (expert e against slot e % 10's activations)
                for ex in range(E):
                    xs = hq[0][(ex % K_USED) * kb:(ex % K_USED + 1) * kb]
                    o = np_(standalone_slab(dw, ex, R, qt, xs, K).double())
                    WORST.add('MOE_DOWN', fmt, fp64(np_(dw[ex]), qt, np_(xs), o), card, f'layer {il} K {K} expert {ex}')
            modes = ['q8'] + (['f32'] if K_USED * K <= 32 * k3.AQ_BLOCKS else [])
            for nw in WGP:
                for mode in modes:
                    e = k3.Engine(module=M1, grid=grid(M1, nw)); out = torch.full((D,), float('nan'), device=dev)
                    e.gg_moe_down(dw.reshape(-1), fmt, slab, K, D, ids[0], wts[0], hq[0] if mode == 'q8' else Hx[0].contiguous(), out, rows_per_task=8)
                    e.build().run()
                    n_all += 1; n_ok += check(torch.equal(out, ref[0]), f'card {card} MOE_DOWN {fmt} K {K} wgp {nw} {mode}: engine != standalone')
                for T in TS:
                    e = k3.Engine(module=MT, grid=grid(MT, nw), T=T); assert e.module == MT
                    out4 = torch.full((T * D,), float('nan'), device=dev)
                    e.gg_moe_down(dw.reshape(-1), fmt, slab, K, D, torch.cat(ids[:T]), torch.cat(wts[:T]), torch.cat(hq[:T]), out4, rows_per_task=8)
                    e.build().run()
                    n_all += 1; n_ok += check(all(torch.equal(out4[t * D:(t + 1) * D], ref[t]) for t in range(T)),
                                              f'card {card} MOE_DOWN {fmt} K {K} wgp {nw} T={T}: rows != T=1')
            print(f'  card {card} MOE_DOWN {fmt:6s} layer {il} K {K} (cols {c0}:{c1}): done', flush=True)
            del dw
    return n_ok, n_all


# ---- (d) unlisted pairs ----
UNLISTED = [('MV', 'IQ3_XXS'), ('MV', 'IQ2_S'), ('MV', 'Q2_0'), ('MOE_GU', 'Q4_K'), ('MOE_GU', 'IQ4_NL'), ('MOE_GU', 'Q8_0'),
            ('MOE_DOWN', 'IQ4_XS'), ('MOE_DOWN', 'Q6_K'), ('MOE_DOWN', 'Q8_0')]
SENT = 1234.5


def run_refuse(card):
    """build() raises, run() refuses, and a forced launch (refuse cleared) writes nothing: zero weights would give 0, so
    the sentinel shows any write. Buffers are sized for the widest format, so even a wrongly running pair stays in bounds"""
    dev = f'cuda:{card}'; n_ok = n_all = 0
    E, F, Kd = 256, 64, 640
    w = torch.zeros(E * F * (D // 32 * 36) + (1 << 20), dtype=torch.uint8, device=dev)
    xq = G.quantize_q8_1(torch.randn(4 * K_USED * max(D, Kd), device=dev))
    for T in (1, 4):
        ids = torch.tensor(sum(IDS[:T], []), dtype=torch.int32, device=dev); wts = torch.full((T * K_USED,), 0.1, device=dev)
        for op, fmt in UNLISTED:
            def mk():
                e = k3.Engine(module=MT if T > 1 else M1, grid=grid(MT if T > 1 else M1, 8), T=T)
                if op == 'MV':
                    out = torch.full((T * F,), SENT, device=dev); e.gg_matvec(w, fmt, F, D, D // 32 * 36, xq, out, rows_per_task=16)
                elif op == 'MOE_GU':
                    out = torch.full((T * K_USED * F,), SENT, device=dev)
                    e.gg_moe_gu(w, w, fmt, F * (D // 32 * 36), F, D, ids, xq, out, rows_per_task=8, up_fmt=fmt)
                else:
                    out = torch.full((T * D,), SENT, device=dev)
                    e.gg_moe_down(w, fmt, D * (Kd // 32 * 36), Kd, D, ids, wts, xq, out, rows_per_task=8)
                return e, out
            e, out = mk()
            try:
                e.build(); ok = False
            except ValueError as ex:
                ok = 'does not implement' in str(ex)
            n_all += 1; n_ok += check(ok, f'card {card} T={T} {op} {fmt}: build() did not refuse')
            e, out = mk(); e.build(strict=False)
            try:
                e.run(); ok = False
            except RuntimeError:
                ok = True
            torch.cuda.synchronize()
            n_all += 1; n_ok += check(ok and bool((out == SENT).all()), f'card {card} T={T} {op} {fmt}: run() did not refuse / out touched')
            e, out = mk(); e.build(strict=False); assert e.refuse; e.refuse = None
            e.run(); torch.cuda.synchronize()
            n_all += 1; n_ok += check(bool((out == SENT).all()), f'card {card} T={T} {op} {fmt}: forced launch wrote {(out != SENT).sum().item()} values')
    print(f'  card {card} unlisted pairs {[f"{o} {f}" for o, f in UNLISTED]} at T=1,4: refused at build and run, forced launch writes nothing', flush=True)
    return n_ok, n_all


# ---- (e) G-acc on recorded activations ----
def record():
    """--record OUT: the activations of (e), recorded at every quantized op site of a teacher-forced FlashNextTP decode
    (one token per pass, both cards, the trunk tape) over POSITIONS positions of each text: per layer the attention half's
    mixed (qkv / z / q / k / v), fin (ssm_out) or gat (attn_output) per card, the FFN half's mixed (router-free shared gate
    / up and the experts' gate / up), the shared down's input per card (shh when the module fuses the shared gate / up,
    else the engine's own silu(shh) * shu), each slot's h per card (the experts' down) and the router's ids; the head's
    mixed. Checked while recording: each fp32 row quantizes to the q8_1 copy the op read (so the rows are the op's
    input), and the replicated rows (mixed, ids) are the same bits on both cards"""
    from caminus.engine.flashnext_tp import FlashNextTP, GACC_SITE_HOOKS
    if os.environ.get('K3_TEST_HOOKS') != GACC_SITE_HOOKS:
        sys.exit(f'--record: run it with K3_TEST_HOOKS={GACC_SITE_HOOKS} (every layer its own site rows; test-only)')
    if os.path.exists(RECORD): sys.exit(f'--record: {RECORD} exists: refusing to overwrite (a recording is a reference)')
    files = TOKENS.split(',') if TOKENS else [devpaths.env('ARTIFACTS', 'refs', 'd281-short', f'e2e_{t}_decode', 'tokens.txt')
                                              for t in ('code', 'prose', 'shell')]
    texts = [[int(v) for v in open(f).read().split()] for f in files]
    if min(len(t) for t in texts) < POSITIONS or len(texts) * POSITIONS < 64:
        sys.exit(f'--record: {len(texts)} texts x {POSITIONS} positions (texts of {[len(t) for t in texts]} tokens): need at '
                 f'least 64 vectors per site (docs/specs/spec-own-quant.md, G-acc)')
    t0 = time.time(); m = FlashNextTP(PATH)
    assert m.gacc_sites and not m.qsa and m.ntok == 1 and D == m.D and K_USED == m.k_used, 'the trunk tape, one token, the Coder'
    print(f'FlashNextTP {[e.module for e in m.engines]} loaded in {time.time() - t0:.0f} s', flush=True)
    gdn = [il % m.fai != m.fai - 1 for il in range(m.nl)]
    fmt_ = lambda il, n: TT[f'blk.{il}.{n}.weight'].tensor_type.name
    fused = [fmt_(il, 'ffn_gate_shexp') == fmt_(il, 'ffn_up_shexp') and ('GU_DENSE', fmt_(il, 'ffn_gate_shexp')) in (m.engines[0].caps or ())
             for il in range(m.nl)]
    rec, bad = {}, {}

    def keep(k, t): rec.setdefault(k, []).append(t.cpu().clone())

    def same(kind, ok): bad[kind] = bad.get(kind, 0) + (not ok)

    def q8(c, kind, x, q):   # the op's q8_1 copy is the quantized fp32 row
        with torch.cuda.device(m.devs[c]):
            y = G.quantize_q8_1(x); same(kind, torch.equal(y, q[:y.numel()]))

    t0 = time.time()
    for toks in texts:
        m.reset()
        for i in range(POSITIONS):
            m.step(toks, i)
            for key in [(il, part) for il in range(m.nl) for part in ('attn', 'ffn')] + [('head',)]:
                b = [m.site_bufs[c][key] for c in (0, 1)]; tag = 'head' if key == ('head',) else f'L{key[0]}.{key[1]}'
                same('mixed on both cards', torch.equal(b[0]['mixed'].cpu(), b[1]['mixed'].cpu())); keep(f'{tag}.mixed', b[0]['mixed'])
                for c in (0, 1): q8(c, 'mq = q8_1(mixed)', b[c]['mixed'], b[c]['mq'])
                if key == ('head',): continue
                il, part = key
                if part == 'attn':
                    x = 'fin' if gdn[il] else 'gat'
                    for c in (0, 1): q8(c, f'fq = q8_1({x})', b[c][x], b[c]['fq']); keep(f'{tag}.{x}.c{c}', b[c][x])
                    continue
                same('ids on both cards', torch.equal(b[0]['ids'].cpu(), b[1]['ids'].cpu())); keep(f'{tag}.ids', b[0]['ids'])
                for c in (0, 1):
                    q8(c, 'hq = q8_1(h)', b[c]['h'], b[c]['hq']); keep(f'{tag}.h.c{c}', b[c]['h'])
                    with torch.cuda.device(m.devs[c]): shd = b[c]['shh'] if fused[il] else G.silu_mul(b[c]['shh'], b[c]['shu'])
                    q8(c, 'shq = q8_1(shh), fused shared gate / up' if fused[il] else 'shq = q8_1(gg_silu_mul(shh, shu)), unfused',
                       shd, b[c]['shq']); keep(f'{tag}.shd.c{c}', shd)
    print(f'recorded {len(texts)} texts x {POSITIONS} positions in {time.time() - t0:.0f} s', flush=True)
    n_ok = n_all = 0
    for kind, nb in sorted(bad.items()):
        n_all += 1; n_ok += check(nb == 0, f'record: {kind}: {nb} rows differ')
    if n_ok < n_all:   # (a recording is a reference: one whose rows are not the ops' inputs is never written)
        print(f'--record: {n_all - n_ok} recording checks failed: nothing saved', flush=True); return n_ok, n_all
    out = dict(meta=dict(model=os.path.basename(PATH), tokens=[os.path.join(*f.split(os.sep)[-2:]) for f in files], positions=POSITIONS,
                         nl=int(m.nl), gdn=[bool(v) for v in gdn], fused=[bool(v) for v in fused], modules=[str(e.module) for e in m.engines],
                         tool='tests/engine/test_k3_gguf_gsq.py --record'),
               shards={n: dict(kind=v['kind'], ranges=[[(int(a), int(b)) for a, b in r] for r in v['ranges']], fmt=v['fmt'])
                       for n, v in m.shards.items()},
               sites={k: torch.stack(v) for k, v in rec.items()})
    del m; torch.cuda.empty_cache()
    if os.path.exists(RECORD): sys.exit(f'--record: {RECORD} appeared during the run: refusing to overwrite')
    torch.save(out, RECORD)
    mib = sum(t.numel() * t.element_size() for t in out['sites'].values()) / 2 ** 20
    print(f'saved {RECORD}: {len(out["sites"])} sites x {len(texts) * POSITIONS} vectors, {mib:.0f} MiB; recording checks {n_ok}/{n_all}', flush=True)
    return n_ok, n_all


GACC_CLASSES = [   # (tensor class, op, its input (layer part, name), how K3 splits it over the cards)
    ('attn_qkv', 'MV', ('attn', 'mixed'), 'rows'), ('attn_gate', 'MV', ('attn', 'mixed'), 'rows'),
    ('attn_q', 'MV', ('attn', 'mixed'), 'rows'), ('attn_k', 'MV', ('attn', 'mixed'), 'rows'), ('attn_v', 'MV', ('attn', 'mixed'), 'rows'),
    ('ssm_out', 'MV', ('attn', 'fin'), 'cols'), ('attn_output', 'MV', ('attn', 'gat'), 'cols'),
    ('ffn_gate_shexp', 'MV', ('ffn', 'mixed'), 'rows'), ('ffn_up_shexp', 'MV', ('ffn', 'mixed'), 'rows'),
    ('ffn_down_shexp', 'MV', ('ffn', 'shd'), 'cols'),
    ('ffn_gate_exps', 'MOE_GU', ('ffn', 'mixed'), 'experts'), ('ffn_up_exps', 'MOE_GU', ('ffn', 'mixed'), 'experts'),
    ('ffn_down_exps', 'MOE_DOWN', ('ffn', 'h'), 'exp_cols')]
GACC_ROWS, GACC_EXP_ROWS = 256, gacc.EXP_ROWS   # rows sampled per matrix, and per routed expert


def gacc_cases(nl):
    """[(tensor name, op, site key or None for the per-card sites, split)]: per class and format its first and last layer
    (GACC_LAYERS 'ends') or every layer ('all'), then the head"""
    out = []
    for cls, op, (part, x), split in GACC_CLASSES:
        by = {}
        for il in range(nl):
            n = f'blk.{il}.{cls}.weight'
            if n in TT: by.setdefault(TT[n].tensor_type.name, []).append(il)
        for fmt, ls in sorted(by.items()):
            for il in (ls if GACC_LAYERS == 'all' else sorted({ls[0], ls[-1]})):
                out.append((f'blk.{il}.{cls}.weight', op, f'L{il}.{part}.{x}', split))
    return out + [('output.weight', 'MV', 'head.mixed', 'rows')]


def run_gacc(card):
    """(e) docs/specs/spec-own-quant.md's G-acc on the recorded activations (--acts): per tensor of gacc_cases, on sampled
    real rows (GACC_ROWS of each matrix; GACC_EXP_ROWS of each routed expert, with the vectors that routed to it), the
    error |y - y_ref| / rms(y_ref) of llama.cpp's matvec (build/gg_mv_ref.hsaco on q8_1 of the recorded fp32 row), y_ref
    the float64 dot of gguf-py's dequantized rows with that fp32 row; mean and p99 per tensor and pooled per (op, format):
    the bar a moved pair's own kernel must meet. Checks: >= 64 vectors and >= 256 distinct real rows per tensor, finite
    errors, and gg_mv_ref bitwise the module's standalone gg_mv_* (and quantizer) on the first vector of every row set"""
    if ACTS is None: sys.exit('--ops GACC: name the recording with --acts FILE (written by --record)')
    dev = f'cuda:{card}'; n_ok = n_all = 0
    R = torch.load(ACTS); meta, shards, sites = R['meta'], R['shards'], R['sites']
    pooled, tensors = {}, []
    for name, op, site, split in gacc_cases(meta['nl']):
        t = TT[name]; qt = t.tensor_type; fmt = qt.name; data = np.asarray(t.data)
        rng = np.random.default_rng(sum(map(ord, name)))
        sets = []   # (raw rows [r, rb] uint8, K, vectors [v, K] fp32, row ids)
        if split in ('rows', 'cols'):
            rows = np.sort(rng.choice(data.shape[0], min(GACC_ROWS, data.shape[0]), replace=False)); raw = data[rows]
            if split == 'rows': sets.append((raw, int(t.shape[0]), sites[site], rows))
            else:
                for c in (0, 1): sets.append((gacc.kcols(raw, shards[name]['ranges'][c], *GGML_QUANT_SIZES[qt]), None, sites[f'{site}.c{c}'], rows))
        else:
            il = int(name.split('.')[1]); ids = sites[f'L{il}.ffn.ids'].numpy()
            if split == 'experts': sets = [(raw, int(t.shape[0]), X, rr) for raw, X, rr in gacc.expert_sets(data, ids, sites[site], False)]
            else:   # (per expert, card 0's shard then card 1's: the order the stored baselines were computed in)
                for c0, c1 in zip(*[gacc.expert_sets(data, ids, sites[f'{site}.c{c}'], True, ranges=shards[name]['ranges'][c],
                                                     bs=GGML_QUANT_SIZES[qt][0], ts=GGML_QUANT_SIZES[qt][1]) for c in (0, 1)]):
                    sets += [(c0[0], None, c0[1], c0[2]), (c1[0], None, c1[1], c1[2])]
        errs, seen, xbad, nd = [], set(), 0, 0
        for raw, K, X, rows in sets:
            K = K or X.shape[1]; r = raw.shape[0]
            W = quants.dequantize(raw, qt).astype(np.float64).reshape(r, -1)
            assert W.shape[1] == K, (name, W.shape, K)
            w = torch.from_numpy(np.ascontiguousarray(raw)).to(dev).reshape(-1); Xd = X.to(dev)
            Y = W @ X.double().numpy().T                                   # [r, v]
            y = np.empty_like(Y)
            for v in range(X.shape[0]):
                x = Xd[v].contiguous(); yq = ref_q8(x); y[:, v] = np_(ref_mv(w, qt, r, K, yq).double())
                if v == 0:
                    xq = G.quantize_q8_1(x); xbad += not (torch.equal(xq, yq) and torch.equal(G.matvec(w, qt, r, K, xq), ref_mv(w, qt, r, K, yq)))
            e_, z_ = gacc.rel_errors(Y, y); errs.append(e_); nd += z_
            seen.update(int(q) for q in rows)
        err = np.concatenate(errs); mean, p99 = float(err.mean()), float(np.percentile(err, 99))
        n_vec_site = sites[site if split in ('rows', 'experts') else f'{site}.c0'].shape[0]
        n_all += 1; n_ok += check(n_vec_site >= 64 and len(seen) >= min(GACC_ROWS, data.shape[0] * (data.shape[1] if data.ndim == 3 else 1)),
                                  f'card {card} G-acc {op} {fmt} {name}: {n_vec_site} vectors, {len(seen)} distinct rows (need 64, 256)')
        n_all += 1; n_ok += check(bool(np.isfinite(err).all()) and nd == 0, f'card {card} G-acc {op} {fmt} {name}: non-finite errors or {nd} zero references')
        n_all += 1; n_ok += check(xbad == 0, f'card {card} G-acc {op} {fmt} {name}: gg_mv_ref != the module\'s gg_mv / quantizer in {xbad} row sets')
        pooled.setdefault((op, fmt), []).append(err)
        tensors.append(dict(tensor=name, op=op, fmt=fmt, split=split, rows=len(seen), vectors=n_vec_site, dots=int(err.size), mean=mean, p99=p99))
        print(f'  card {card} G-acc {op:8s} {fmt:7s} {name:30s} {split:8s} {len(seen):6d} rows, {n_vec_site} vectors, {err.size:7d} dots: '
              f'llama.cpp mean {mean:.2e} p99 {p99:.2e}', flush=True)
    pairs = {}
    for (op, fmt), es in sorted(pooled.items()):
        e = np.concatenate(es); pairs[f'{op} {fmt}'] = dict(mean=float(e.mean()), p99=float(np.percentile(e, 99)), dots=int(e.size), tensors=len(es))
        print(f' G-acc {op:8s} {fmt:7s} pooled over {len(es)} tensors ({e.size} dots): llama.cpp mean {e.mean():.2e} p99 {np.percentile(e, 99):.2e}', flush=True)
    if GACC_OUT:
        if os.path.exists(GACC_OUT): sys.exit(f'--gacc-out: {GACC_OUT} exists: refusing to overwrite (a stored baseline)')
        with open(GACC_OUT, 'w') as f:
            json.dump(dict(acts=os.path.basename(ACTS), acts_meta=meta, layers=GACC_LAYERS, rows=GACC_ROWS, exp_rows=GACC_EXP_ROWS,
                           arm='llama.cpp (build/gg_mv_ref.hsaco)', pairs=pairs, tensors=tensors), f, indent=1)
        print(f' wrote {GACC_OUT}', flush=True)
    return n_ok, n_all


if RECORD:
    ok_, all_ = record()
    print(f'checks: {ok_}/{all_}; failures: {len(fails)}')
    print('PASS' if not fails else 'FAIL'); sys.exit(0 if not fails else 1)


tot_ok = tot = 0
print(f'modules {M1} / {MT}; float64 (c) on cards {C_CARDS}', flush=True)
for card in CARDS:
    torch.cuda.set_device(card)
    print(f'card {card}: {torch.cuda.get_device_name(card)}, {torch.cuda.get_device_properties(card).multi_processor_count} WGPs, '
          f'grids {[grid(M1, n) for n in WGP]}', flush=True)
    for fn, op in ((run_mv, 'MV'), (run_moe_gu, 'MOE_GU'), (run_moe_down, 'MOE_DOWN'), (run_refuse, 'REFUSE'), (run_gacc, 'GACC')):
        if op not in OPS: continue
        ok_, all_ = fn(card); tot_ok += ok_; tot += all_
        print(f' card {card} {op}: {ok_}/{all_} bitwise / refusal checks pass', flush=True)
    torch.cuda.empty_cache()
print('worst standalone vs float64 (max |out - ref| / sum|w||xq|; "int" = integer step emulated, "bound" = |out - exact| / '
      '(B + 1e-6 S), "trunc" = max sub-block truncation in sumi units): '
      + ', '.join(f'{op} {f} {v:.3g}' for (op, f), v in sorted(WORST.v.items())))
asw = [f'{op} {f}: {o}/{n} slices > 1e-6' for (op, f), (n, o) in sorted(WORST.slices.items()) if o]
print('G1(c) as written (exact dot <= 1e-6): ' + ('PASS for every pair' if not asw else 'FAILS for ' + '; '.join(asw))
      + ' (IQ2_S / IQ3_XXS gated on the proposed int-emulated + analytic-bound criterion instead)')
print(f'checks: {tot_ok}/{tot}; failures: {len(fails)}')
print('PASS' if not fails else 'FAIL'); sys.exit(0 if not fails else 1)
