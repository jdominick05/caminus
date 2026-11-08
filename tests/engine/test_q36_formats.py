"""The Qwen3.6-35B-A3B one-card engine's (op, format) instances (build/k3_gguf_q35h{,_mt}.hsaco, kernels/k3_gguf_q35h.hip),
on real HauhauCS Q3_K_P tensors read with O_DIRECT, as tests/engine/test_k3_gguf_gsq.py checks the Coder's:
  MV       Q3_K Q4_K Q5_K Q6_K (whole matrices: one card, no splits), q8_1 and fp32 (AQ) activations
  MOE_GU   Q3_K (layer 0) Q5_K (layer 20), 512 rows per expert, h fp32 and h q8_1
  MOE_DOWN Q4_K (layer 2) Q5_K (layer 0), K 512
checks (a) the engine op is bitwise the standalone kernels (gg_mv_*; gg_mv_q3_k is built into k3_gguf_q35h only, K3_MV_Q3K)
at each --wgp grid; (b) T = 2 / 3 / 4 rows (the _mt module) are bitwise the T = 1 rows; (c) float64 (gguf-py's dequantizer,
the same q8_1 activations) over every row / every expert: rel <= 1e-6; (d) unlisted (op, format) pairs refused, a forced
launch writes nothing. New kernel code: run it on the XT alone first (HIP_VISIBLE_DEVICES=1, --cards 0).
  tools/guard.sh LOG SECS python -I tests/engine/test_q36_formats.py GGUF [--module k3_gguf_q35h | k3_gguf_q35g] [--cards 0] [--wgp 3,6,7,8]
      [--ops MV,MOE_GU,MOE_DOWN,BF16,REFUSE]
      [--small] (--small: the first case of each op only, grids 8 only)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
from caminus import devpaths  # noqa: E402  (catalog models and ARTIFACTS from the environment: CLAUDE.md, "Models and data")
import numpy as np
import torch
from caminus.engine import gguf as G, k3
from caminus.engine.flashnext import DirectReader
from gguf import quants, GGML_QUANT_SIZES

a = sys.argv[1:]
def opt(name, default):
    if name in a:
        i = a.index(name); v = a[i + 1]; del a[i:i + 2]; return v
    return default
CARDS = [int(v) for v in opt('--cards', '0').split(',')]
C_CARDS = [int(v) for v in opt('--c-cards', ','.join(map(str, CARDS))).split(',')]   # cards that run the float64 check (c)
WGP = [int(v) for v in opt('--wgp', '3,6,7,8').split(',')]
SMALL = '--small' in a
if SMALL: a.remove('--small'); WGP = [8]
ONLY = opt('--only', None)   # debugging: only cases whose label contains this
OPS = opt('--ops', 'MV,MOE_GU,MOE_DOWN,BF16,REFUSE').split(',')
M1 = opt('--module', 'k3_gguf_q35h'); assert M1 in ('k3_gguf_q35h', 'k3_gguf_q35g'), M1
MT = M1 + '_mt'
PATH = a[0] if a else devpaths.catalog('qwen3.6-hauhau', 'HAUHAU')
RD = G.GGUFReader(PATH, 'r'); TT = {t.name: t for t in RD.tensors}
DR = DirectReader(PATH)
D, K_USED = 2048, 8
TS = (2, 3, 4)
fails = []


def check(cond, what):
    if not cond: fails.append(what); print('  FAIL ' + what, flush=True)
    return cond


def full(name):
    """the whole tensor's raw bytes on cuda:0 (O_DIRECT), shaped [..., row bytes]"""
    t = TT[name]; w = DR.read(int(t.data_offset), int(t.n_bytes))
    return w.view(*t.data.shape), t.tensor_type


def cols(w, qt, b0, b1):
    """block columns [b0, b1) of every row of raw rows w [..., rb] (a K split)"""
    ts = GGML_QUANT_SIZES[qt][1]; sh = w.shape
    return w.view(*sh[:-1], sh[-1] // ts, ts)[..., b0:b1, :].reshape(*sh[:-1], (b1 - b0) * ts).contiguous()


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


# ---- MV ----
MV_CASES = {  # (name, None): whole matrices, every format the module lists
    'k3_gguf_q35h': [
        ('blk.1.attn_gate.weight', None), ('blk.0.ssm_out.weight', None), ('blk.11.attn_q.weight', None), ('blk.1.ssm_alpha.weight', None),   # Q3_K
        ('blk.3.attn_output.weight', None), ('blk.4.attn_qkv.weight', None), ('blk.2.ffn_down_shexp.weight', None),                       # Q4_K
        ('blk.0.attn_gate.weight', None), ('blk.0.ssm_alpha.weight', None), ('blk.0.ffn_down_shexp.weight', None), ('blk.20.ffn_gate_shexp.weight', None),  # Q5_K
        ('output.weight', None)],                                                                                                         # Q6_K
    'k3_gguf_q35g': [   # Q8_0 everywhere dense (the GSQ checkpoint's INT8 weights)
        ('blk.0.attn_qkv.weight', None), ('blk.0.ssm_out.weight', None), ('blk.3.attn_q.weight', None), ('blk.3.attn_k.weight', None),
        ('blk.0.ffn_down_shexp.weight', None), ('blk.0.ffn_gate_shexp.weight', None), ('output.weight', None)],
}[M1]
if SMALL: MV_CASES = MV_CASES[:3]


def mv_cases():
    for name, splits in MV_CASES:
        w, qt = full(name)
        if splits is None: cs = [(name, w)]
        else: cs = [(f'{name}[blocks {b0}:{b1}]', cols(w, qt, b0, b1)) for b0, b1 in splits]
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
                e = k3.Engine(module=M1, grid=grid(MT, nw), T=T); assert e.module == MT
                out4 = torch.full((T * N,), float('nan'), device=dev)
                e.gg_matvec(w.reshape(-1), fmt, N, K, rb, torch.cat(xq[:T]), out4, rows_per_task=16)
                e.build().run()
                n_all += 1; n_ok += check(all(torch.equal(out4[t * N:(t + 1) * N], ref[t]) for t in range(T)),
                                          f'card {card} MV {fmt} {label} wgp {nw} T={T}: rows != T=1')
        print(f'  card {card} MV {fmt:6s} {label:42s} N {N:5d} K {K:5d}: float64 rel {"-" if rel is None else f"{rel:.1e}"} (all rows)', flush=True)
    return n_ok, n_all


# ---- MoE ----
GU_LAYERS, DOWN_LAYERS = {'k3_gguf_q35h': ({'Q3_K': 0, 'Q5_K': 20}, {'Q4_K': 2, 'Q5_K': 0}),
                          'k3_gguf_q35g': ({'Q2_0': 0}, {'Q2_0': 0})}[M1]
if SMALL: GU_LAYERS = dict(list(GU_LAYERS.items())[:1]); DOWN_LAYERS = dict(list(DOWN_LAYERS.items())[:1])
IDS = [[0, 255, 17, 42, 99, 150, 200, 7], [255, 3, 17, 64, 0, 77, 190, 128],
       [5, 6, 7, 8, 250, 251, 252, 253], [17, 0, 255, 42, 99, 1, 2, 128]]   # T = 4: repeats across tokens


def standalone_slab(stack, e, rows, qt, xq, K):
    return G.matvec(stack[e].reshape(-1), qt, rows, K, xq)


def run_moe_gu(card):
    dev = f'cuda:{card}'; n_ok = n_all = 0
    for fmt, il in GU_LAYERS.items():
        g0, qt = full(f'blk.{il}.ffn_gate_exps.weight'); u0, qt2 = full(f'blk.{il}.ffn_up_exps.weight'); assert qt == qt2
        for r0, r1 in ((0, 512),):                               # whole experts (one card: no row split)
            g = g0[:, r0:r1].contiguous().to(dev); u = u0[:, r0:r1].contiguous().to(dev)
            E, F, rb = g.shape; slab = F * rb
            gen = torch.Generator(device=dev).manual_seed(2000 + F)
            X = torch.randn(4, D, device=dev, generator=gen) * 0.5
            xq = [G.quantize_q8_1(X[t]) for t in range(4)]
            ids = [torch.tensor(v, dtype=torch.int32, device=dev) for v in IDS]
            gs = [[standalone_slab(g, int(e), F, qt, xq[t], D) for e in IDS[t]] for t in range(4)]
            us = [[standalone_slab(u, int(e), F, qt, xq[t], D) for e in IDS[t]] for t in range(4)]
            href = [G.silu_mul(torch.cat(gs[t]), torch.cat(us[t])) for t in range(4)]
            if (r0, r1) == (0, 512) and card in C_CARDS:   # (c) every expert of gate and up, every row, against float64
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
                    e = k3.Engine(module=M1, grid=grid(MT, nw), T=T); assert e.module == MT
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
        for c0, c1 in ((0, 512),):                               # whole rows (one card: no K split)
            dw = cols(d0, qt, c0 // qk, c1 // qk).to(dev); E, R, rb = dw.shape; K = c1 - c0; slab = R * rb
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
            if card in C_CARDS:   # (c) every expert, every row (expert e against slot e % 8's activations)
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
                    e = k3.Engine(module=M1, grid=grid(MT, nw), T=T); assert e.module == MT
                    out4 = torch.full((T * D,), float('nan'), device=dev)
                    e.gg_moe_down(dw.reshape(-1), fmt, slab, K, D, torch.cat(ids[:T]), torch.cat(wts[:T]), torch.cat(hq[:T]), out4, rows_per_task=8)
                    e.build().run()
                    n_all += 1; n_ok += check(all(torch.equal(out4[t * D:(t + 1) * D], ref[t]) for t in range(T)),
                                              f'card {card} MOE_DOWN {fmt} K {K} wgp {nw} T={T}: rows != T=1')
            print(f'  card {card} MOE_DOWN {fmt:6s} layer {il} K {K} (cols {c0}:{c1}): done', flush=True)
            del dw
    return n_ok, n_all


# ---- BF16 (k3_gguf_q35g: ssm_alpha / ssm_beta) ----
def run_bf16(card):
    """OP_BF16_MV on the GSQ checkpoint's BF16 matrices vs float64 (bf16 values are exact in float64): rel <= 1e-6, T = 4 rows ==
    T = 1, every grid. The Coder's tests/engine/test_k3_bf16_mv.py checks the same op bitwise against llama.cpp's mmvf"""
    dev = f'cuda:{card}'; n_ok = n_all = 0
    for name in ('blk.0.ssm_alpha.weight', 'blk.0.ssm_beta.weight', 'blk.38.ssm_alpha.weight'):
        t = TT[name]; assert t.tensor_type.name == 'BF16', (name, t.tensor_type.name)
        w = DR.read(int(t.data_offset), int(t.n_bytes)).to(dev); N, K = int(t.shape[1]), int(t.shape[0])
        X = torch.randn(4, K, device=dev, generator=torch.Generator(device=dev).manual_seed(4000 + N)) * 0.5
        Wf = torch.from_numpy(np.asarray(t.data).view(np.uint16).astype(np.uint32) << 16).view(torch.float32).double().reshape(N, K)
        ref = (Wf @ X.double().cpu().T).T                                   # [4, N]
        outs = {}
        for nw in WGP:
            for T in (1, 4):
                e = k3.Engine(module=M1 if T == 1 else MT, grid=grid(M1 if T == 1 else MT, nw), T=T)
                out = torch.full((T * N,), float('nan'), device=dev)
                e.bf16_mv(w, N, K, X[:T].contiguous(), out, rows_per_task=4); e.build().run()
                outs[(nw, T)] = out.view(T, N).cpu()
        o1 = outs[(WGP[0], 1)][0].double()
        rel = float(((o1 - ref[0]).abs() / (Wf.abs() @ X[0].double().cpu().abs()).clamp_min(1e-30)).max())
        n_all += 1; n_ok += check(rel <= 1e-6, f'card {card} BF16_MV {name}: vs float64 {rel:.2e} > 1e-6')
        for (nw, T), o in outs.items():
            n_all += 1; n_ok += check(torch.equal(o[0], outs[(WGP[0], 1)][0]) and (T == 1 or all(torch.equal(o[t], outs[(nw, 4)][t]) for t in range(T))),
                                      f'card {card} BF16_MV {name} wgp {nw} T={T}: row 0 != the T = 1 row')
        print(f'  card {card} BF16_MV {name:28s} N {N:4d} K {K:5d}: float64 rel {rel:.1e}', flush=True)
    return n_ok, n_all


# ---- (d) unlisted pairs ----
UNLISTED = {
    'k3_gguf_q35h': [('MV', 'IQ3_S'), ('MV', 'IQ4_XS'), ('MV', 'Q8_0'), ('MV', 'IQ4_NL'), ('MOE_GU', 'Q4_K'), ('MOE_GU', 'IQ3_XXS'),
                     ('MOE_GU', 'Q6_K'), ('MOE_DOWN', 'Q3_K'), ('MOE_DOWN', 'Q6_K'), ('MOE_DOWN', 'IQ4_XS')],
    'k3_gguf_q35g': [('MV', 'Q4_K'), ('MV', 'Q6_K'), ('MV', 'IQ4_XS'), ('MV', 'Q3_K'), ('MOE_GU', 'Q4_K'), ('MOE_GU', 'IQ3_XXS'),
                     ('MOE_GU', 'Q3_K'), ('MOE_DOWN', 'IQ4_NL'), ('MOE_DOWN', 'Q4_K'), ('MOE_DOWN', 'Q3_K')],
}[M1]
SENT = 1234.5


def run_refuse(card):
    """build() raises, run() refuses, and a forced launch (refuse cleared) writes nothing: zero weights would give 0, so
    the sentinel shows any write. Buffers are sized for the widest format, so even a wrongly running pair stays in bounds"""
    dev = f'cuda:{card}'; n_ok = n_all = 0
    E, F, Kd = 256, 64, 512
    w = torch.zeros(E * F * (D // 32 * 36) + (1 << 20), dtype=torch.uint8, device=dev)
    xq = G.quantize_q8_1(torch.randn(4 * K_USED * max(D, Kd), device=dev))
    for T in (1, 4):
        ids = torch.tensor(sum(IDS[:T], []), dtype=torch.int32, device=dev); wts = torch.full((T * K_USED,), 0.1, device=dev)
        for op, fmt in UNLISTED:
            def mk():
                e = k3.Engine(module=M1, grid=grid(MT if T > 1 else M1, 8), T=T)
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


tot_ok = tot = 0
print(f'modules {M1} / {MT}; float64 (c) on cards {C_CARDS}', flush=True)
for card in CARDS:
    torch.cuda.set_device(card)
    print(f'card {card}: {torch.cuda.get_device_name(card)}, {torch.cuda.get_device_properties(card).multi_processor_count} WGPs, '
          f'grids {[grid(M1, n) for n in WGP]}', flush=True)
    for fn, op in ((run_mv, 'MV'), (run_moe_gu, 'MOE_GU'), (run_moe_down, 'MOE_DOWN'), (run_bf16, 'BF16'), (run_refuse, 'REFUSE')):
        if op == 'BF16' and M1 != 'k3_gguf_q35g': continue
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
