"""Qwen3-30B-A3B's decode-token matvecs as ONE persistent K3 launch (docs/specs/spec-own-quant.md; the decode-floor
paper's token: 48 layers x (q, k, v, o, router, 8 x gate, 8 x up, 8 x down) + the vocab head = 1,393 matvecs), on the
real GGUF (read with caminus.ggufio), module k3_gguf_q35d_own:
  own    Caminus's one-round-trip tiles (OP_OWN_MV a16_1rt), the 8 experts of a layer grouped into one segment by device
         ids: 48 x 8 + 1 = 385 segments; fp16 activations; Q6_K in our plane layout (engine/own_layout.py); one arm per
         --rpt (rows per task). (A 16 KB touch of each workgroup's first task before the dependency wait, as the GGUF
         matvecs have, measured no gain here: chain 4.504 ms with and without, XT)
  llama  llama.cpp's matvec (OP_GG_MV, q8_1 activations precomputed: their quantize segments are not timed), one segment
         per matrix: 1,393 segments
  both   the router as OP_F32_MV (the same in both arms)
Checks first, per distinct shape: the op is bitwise the standalone kernel (own_mv_q4_k_1rt_k* / own_mv_q6_k_pl_k*, per
expert for the grouped roles) at K3_PER_WGP 3 / 6 / 7 / 8; error vs fp64 (gguf-py's dequant, the test oracle) on 256 rows.
Timing: --experts of each layer's 128 are loaded (default 32: 4.6 GiB of experts, far past the Infinity Cache); every
round draws new expert ids (own: the ids buffer; llama: a tape built per round with the same picks); 'chain' = each
segment waits on the one before (stricter than the real token), 'free' = no waits (throughput). Median of --rounds.
  tools/guard.sh LOG 1800 python -I benchmarks/own_mv_k3_token.py GGUF [--rounds 5] [--experts 32]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, ctypes, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from caminus import ggufio
from caminus.device import capacity
from caminus.device.hip import Module
from caminus.engine import gguf as G, k3     # (gg_matvec / quantize_q8_1: the llama.cpp yardstick arm; puts gguf-py on the path)
from caminus.engine.own_layout import q6k_planes
from gguf import quants, GGMLQuantizationType   # (the test oracle only)

ap = argparse.ArgumentParser()
ap.add_argument('gguf')
ap.add_argument('--rounds', type=int, default=5)
ap.add_argument('--experts', type=int, default=32)
ap.add_argument('--log', default=None)
ap.add_argument('--rpt', nargs='+', default=['16'], help="own arms: rows per task (multiples of 16; the head keeps 64), or "
                'auto<k>: about k tasks per workgroup in every segment')
ap.add_argument('--timing-only', action='store_true', help='skip the bitwise / error checks and the llama.cpp arm '
                '(a measurement build of the module, e.g. -DOWN_READONLY: the tiles with free arithmetic)')
A = ap.parse_args()
dev = torch.device('cuda:0')
MOD = 'k3_gguf_q35d_own'
OWN = Module('own_mv')
rng = np.random.default_rng(0)
F = ggufio.GGUFFile(A.gguf)
NL, KU, NE = int(F.hp('block_count')), int(F.hp('expert_used_count')), int(F.hp('expert_count'))
EL = min(A.experts, NE)
ok_all, OUT = True, []


def say(s):
    print(s, flush=True); OUT.append(s)


def check(c, what):
    global ok_all
    say(('PASS ' if c else 'FAIL ') + what); ok_all &= bool(c)


def dims(t):
    """(rows per matrix, K, experts)"""
    return (int(t.shape[1]), int(t.shape[0]), int(t.shape[2]) if len(t.shape) == 3 else 1)


def raw_of(t, ne=None):
    """the tensor's bytes (experts: the first ne experts)"""
    rows, K, E = dims(t)
    if E == 1: return np.asarray(t.data())
    per = t.n_bytes // E
    return np.asarray(t.data()[:(ne or E) * per])


def dev_u8(a):
    return torch.from_numpy(np.ascontiguousarray(a).copy()).to(dev)


def own_w(t, ne=None):
    rows, K, E = dims(t)
    b = raw_of(t, ne)
    return dev_u8(q6k_planes(b, rows * (ne or 1 if E > 1 else 1), K) if t.type == 'Q6_K' else b)


def std_kernel(fmt, K):
    return getattr(OWN, f'own_mv_q4_k_1rt_k{K}' if fmt == 'Q4_K' else f'own_mv_q6_k_pl_k{K}')


def heavy_x(K, n=1):
    v = rng.standard_t(3, (n, K)); v[:, rng.choice(K, 8, replace=False)] *= 20
    return (v / np.sqrt((v ** 2).mean(axis=1, keepdims=True))).astype(np.float32)


# ---- projection ---------------------------------------------------------------------------------------------------------
def tok_roles(il):
    return {r: F.T[f'blk.{il}.{r}.weight'] for r in ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate_exps',
                                                    'ffn_up_exps', 'ffn_down_exps', 'ffn_gate_inp')}


need = 0
for il in range(NL):
    for r, t in tok_roles(il).items():
        b = t.n_bytes * (EL / NE if '_exps' in r else 1)
        need += b * (2 if t.type == 'Q6_K' else 1)              # (Q6_K: planes for ours + raw for llama.cpp)
need += 2 * F.T['output.weight'].n_bytes
need_mib = need / 2 ** 20 + 1024                                # + activations, outputs, check buffers
capacity.check([0], [need_mib], f'own_mv_k3_token ({EL} experts per layer)')
say(f'{torch.cuda.get_device_name(0)}  {os.path.basename(A.gguf)}  {NL} layers, {KU} of {NE} experts ({EL} loaded per layer), '
    f'~{need_mib / 1024:.1f} GiB on the card')

# ---- checks: bitwise vs the standalone kernels across grids, error vs fp64 ------------------------------------------------
def first_layer(role, fmt):
    for il in range(NL):
        if F.T[f'blk.{il}.{role}.weight'].type == fmt: return il
    return None


cases = []
for role in ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps'):
    for fmt in ('Q4_K', 'Q6_K'):
        il = first_layer(role, fmt)
        if il is not None: cases.append((role, fmt, F.T[f'blk.{il}.{role}.weight']))
cases.append(('head', 'Q6_K', F.T['output.weight']))
if A.timing_only: cases = []
for role, fmt, t in cases:
    rows, K, E = dims(t); qt = GGMLQuantizationType[fmt]
    grouped = E > 1
    ne = EL if grouped else None
    W = own_w(t, ne)
    S = KU if grouped else 1
    ids_np = rng.choice(EL, KU, replace=False).astype(np.int32) if grouped else None
    per_slot = role == 'ffn_down_exps'
    xs = heavy_x(K, S if per_slot else 1)
    x16 = torch.tensor(xs, dtype=torch.float16, device=dev).reshape(-1)
    # the standalone kernel per slot (Q6_K grouped: that expert's own planes)
    ref = torch.zeros(S * rows, device=dev)
    rb = t.n_bytes // E // rows
    for s in range(S):
        if grouped:
            e = int(ids_np[s]); sl = raw_of(t, ne)[e * rows * rb:(e + 1) * rows * rb]
            Ws = dev_u8(q6k_planes(sl, rows, K) if fmt == 'Q6_K' else sl)
        else:
            Ws = W
        xo = x16[s * K:(s + 1) * K] if per_slot else x16[:K]
        std_kernel(fmt, K)(((rows + 15) // 16,), (256,), Ws, ctypes.c_int(rows), xo, ref[s * rows:(s + 1) * rows])
    torch.cuda.synchronize()
    outs = []
    ids = torch.tensor(ids_np, device=dev) if grouped else None
    for nw in (3, 6, 7, 8):
        e = k3.Engine(module=MOD, grid=k3.resident_grid(MOD, nw)); out = torch.full((S * rows,), float('nan'), device=dev)
        e.own_matvec_1rt(W, fmt, rows, K, x16, out, ids=ids, total_rows=rows * EL if grouped else None, x_per_slot=per_slot,
                         rows_per_task=64 if role == 'head' else 16)
        e.build().run(); outs.append(out)
    check(all(torch.equal(o, ref) for o in outs) and not torch.isnan(ref).any(),
          f'{role} {fmt} [{rows} x {K}]{" x %d slots" % S if grouped else ""}: K3 op bitwise the standalone kernel at K3_PER_WGP 3/6/7/8')
    # error vs fp64 on slot 0's sampled rows
    e0 = int(ids_np[0]) if grouped else 0
    srows = np.sort(rng.choice(rows, min(256, rows), replace=False))
    raw = raw_of(t, ne)[e0 * rows * rb:(e0 + 1) * rows * rb].reshape(rows, rb)[srows] if grouped else raw_of(t).reshape(rows, rb)[srows]
    W64 = quants.dequantize(raw, qt).astype(np.float64).reshape(len(srows), K)
    yref = W64 @ xs[0].astype(np.float64); sc = np.sqrt((yref ** 2).mean())
    err = np.abs(ref[:rows].double().cpu().numpy()[srows] - yref) / sc
    say(f'      {role} {fmt}: error vs fp64 mean {err.mean():.2e} p99 {np.percentile(err, 99):.2e}')
    del W, ref, outs; torch.cuda.empty_cache()

# ---- the token -----------------------------------------------------------------------------------------------------------
say('loading the token\'s weights ...')
L = []
for il in range(NL):
    R = tok_roles(il); d = {}
    for r in ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps'):
        t = R[r]; rows, K, E = dims(t); ne = EL if E > 1 else None
        wo = own_w(t, ne)
        wl = wo if t.type == 'Q4_K' else dev_u8(raw_of(t, ne))
        d[r] = (t.type, rows, K, wo, wl, t.n_bytes // E // rows)
    gi = R['ffn_gate_inp']
    d['router'] = torch.from_numpy(np.asarray(gi.data()).view(np.float32).reshape(int(gi.shape[1]), int(gi.shape[0])).copy()).to(dev)
    L.append(d)
th = F.T['output.weight']; HR, HK, _ = dims(th)
head_o, head_l = own_w(th), dev_u8(raw_of(th))
D = int(F.hp('embedding_length')); QD = int(L[0]['attn_q'][1]); KVD = int(L[0]['attn_k'][1]); FF = int(L[0]['ffn_gate_exps'][1])
x32 = torch.tensor(heavy_x(D)[0], device=dev); x16 = x32.half()
xo32 = torch.tensor(heavy_x(QD)[0], device=dev); xo16 = xo32.half()
xd32 = torch.tensor(heavy_x(FF, KU), device=dev); xd16 = xd32.half().reshape(-1)
yq, yq_o = G.quantize_q8_1(x32), G.quantize_q8_1(xo32)
yq_d = [G.quantize_q8_1(xd32[s].contiguous()) for s in range(KU)]
o_q, o_k, o_v, o_o = (torch.zeros(n, device=dev) for n in (QD, KVD, KVD, D))
o_r = torch.zeros(NE, device=dev); o_g, o_u = torch.zeros(KU * FF, device=dev), torch.zeros(KU * FF, device=dev)
o_d, o_h = torch.zeros(KU * D, device=dev), torch.zeros(HR, device=dev)
ids = torch.zeros((NL, KU), dtype=torch.int32, device=dev)
picks = [np.stack([rng.choice(EL, KU, replace=False) for _ in range(NL)]).astype(np.int32) for _ in range(A.rounds + 1)]
tok_bytes = sum(d[r][1] * d[r][5] * (KU if '_exps' in r else 1) for d in L for r in d if r != 'router') \
    + sum(d['router'].numel() * 4 for d in L) + th.n_bytes


NWG = k3.resident_grid(MOD)


def pick_rpt(rpt, n):
    """rows per task: a number, or 'auto<k>' = about k tasks per workgroup of the grid (a multiple of 16, at least 16)"""
    if isinstance(rpt, int): return min(rpt, n)
    k = int(rpt[4:] or 1)
    return max(16, -(-n // (NWG * k * 16)) * 16)


def tape_own(struct, rpt):
    e = k3.Engine(module=MOD); prev = -1
    dp = lambda: prev if struct == 'chain' else -1
    for il, d in enumerate(L):
        for r, x, o in (('attn_q', x16, o_q), ('attn_k', x16, o_k), ('attn_v', x16, o_v), ('attn_output', xo16, o_o)):
            f, rows, K, wo, _, _ = d[r]; prev = e.own_matvec_1rt(wo, f, rows, K, x, o, dep=dp(), rows_per_task=pick_rpt(rpt, rows))
        prev = e.f32_mv(d['router'], x32, o_r, dep=dp())
        for r, x, o, ps in (('ffn_gate_exps', x16, o_g, False), ('ffn_up_exps', x16, o_u, False), ('ffn_down_exps', xd16, o_d, True)):
            f, rows, K, wo, _, _ = d[r]
            prev = e.own_matvec_1rt(wo, f, rows, K, x, o, ids=ids[il], total_rows=rows * EL, x_per_slot=ps, dep=dp(),
                                    rows_per_task=pick_rpt(rpt, KU * rows) if isinstance(rpt, str) else rpt)
    e.own_matvec_1rt(head_o, 'Q6_K', HR, HK, x16, o_h, rows_per_task=64 if isinstance(rpt, int) else pick_rpt(rpt, HR), dep=dp())
    return e.build()


def tape_llama(struct, pk):
    e = k3.Engine(module=MOD); prev = -1
    dp = lambda: prev if struct == 'chain' else -1
    for il, d in enumerate(L):
        for r, y, o in (('attn_q', yq, o_q), ('attn_k', yq, o_k), ('attn_v', yq, o_v), ('attn_output', yq_o, o_o)):
            f, rows, K, _, wl, rb = d[r]; prev = e.gg_matvec(wl, f, rows, K, rb, y, o, dep=dp())
        prev = e.f32_mv(d['router'], x32, o_r, dep=dp())
        for r, o in (('ffn_gate_exps', o_g), ('ffn_up_exps', o_u), ('ffn_down_exps', o_d)):
            f, rows, K, _, wl, rb = d[r]
            for s in range(KU):
                ex = int(pk[il, s]); y = yq_d[s] if r == 'ffn_down_exps' else yq
                prev = e.gg_matvec(wl[ex * rows * rb:(ex + 1) * rows * rb], f, rows, K, rb, y, o[s * rows:(s + 1) * rows], dep=dp())
    e.gg_matvec(head_l, 'Q6_K', HR, HK, th.n_bytes // HR, yq, o_h, dep=dp())
    return e.build()


say(f'token: {tok_bytes / 2**30:.3f} GiB of weights (router included); own {NL * 8 + 1} segments, llama.cpp {NL * 29 + 1}')
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
res = {}
for struct in ('chain', 'free'):
    tapes = {f'own-{rpt}': tape_own(struct, int(rpt) if rpt.isdigit() else rpt) for rpt in A.rpt}
    tl = [None if A.timing_only else tape_llama(struct, picks[r]) for r in range(A.rounds + 1)]
    tm = {k: [] for k in list(tapes) + ([] if A.timing_only else ['llama'])}
    for r in range(A.rounds + 1):
        ids.copy_(torch.from_numpy(picks[r]))
        for arm, tp in list(tapes.items()) + ([] if A.timing_only else [('llama', tl[r])]):
            torch.cuda.synchronize(); ev[0].record(); tp.run(); ev[1].record(); torch.cuda.synchronize()
            if r: tm[arm].append(ev[0].elapsed_time(ev[1]))
    for arm in tm:
        ms = float(np.median(tm[arm])); res[(struct, arm)] = ms
        say(f'  {struct:5s} {arm:8s} {ms:7.3f} ms/token  {tok_bytes / ms / 1e6:6.1f} GB/s  (min {min(tm[arm]):.3f}, max {max(tm[arm]):.3f})')
    del tapes, tl
say('RESULT ' + ('PASS' if ok_all else 'FAIL'))
if A.log:
    with open(A.log, 'a') as f: f.write('\n'.join(OUT) + '\n')
