"""The codebook formats' matvec tiles inside K3 (OP_OWN_MV on module k3_q3m_iq: IQ4_XS, IQ3_XXS, IQ3_S, IQ2_XXS, beside
Q4_K and Q6_K) against their standalone kernels (own_mv), on a file's real expert tensors (docs/dev/iq-quant-study.md):
  host    the engine's refusals, no launch: a grid format on the shipped module; a tape with grid formats that does not
          start with the embedding (its segment carries the grids to stage); IQ2_XXS and an IQ3 format on one tape;
          the stage bits Engine.build derives from the tape's own segments; a router over more experts than the
          module's local memory holds
  bitwise every (role, format) of the file's experts as ONE tape (so one launch stages the grids this file needs
          together): each segment's rows are the standalone kernel's, per expert slot, at 3 / 5 / 7 / 8 workgroups per WGP
  gate|up the fused segment (fp16(silu(gate . x) * (up . x))) is the same bits at every grid and equals the expression
          on the standalone kernels' sums to fp16 rounding
  error   against float64 (caminus.own_dequant, the host decoders, on the fp16 activations the tile reads) on every row of
          every slot: what is left is the fp32 sums' rounding, so a wrong decode of any block shows
--layers all: a case per layer, role and format (every expert tensor of the file, 8 drawn experts of --experts each)
instead of the first layer of each (role, format).
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  tools/guard.sh LOG 900 python -I tests/engine/test_own_iq_ops.py GGUF [--experts 16] [--layers all]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, ctypes, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
import torch
from caminus import ggufio, own_dequant
from caminus.device import capacity
from caminus.device.hip import Module
from caminus.engine import k3
from caminus.engine.own_layout import q6k_planes, iq_records, IQ_REC

ap = argparse.ArgumentParser()
ap.add_argument('gguf')
ap.add_argument('--experts', type=int, default=16, help='experts of each tensor on the device')
ap.add_argument('--layers', default='first', choices=('first', 'all'))
A = ap.parse_args()
MOD, LEAN = 'k3_q3m_iq', 'k3_q3m_lean'
F = ggufio.GGUFFile(A.gguf)
NL, KU, NE = int(F.hp('block_count')), int(F.hp('expert_used_count')), int(F.hp('expert_count'))
EL = min(A.experts, NE)
ROLES = ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps')
STD = {'Q4_K': 'own_mv_q4_k_1rt_k{K}', 'Q6_K': 'own_mv_q6_k_pl_k{K}', 'IQ4_XS': 'own_mv_iq4xs_1rt_k{K}_r2',
       'IQ3_XXS': 'own_mv_iq3xxs_1rt_k{K}_r2', 'IQ3_S': 'own_mv_iq3s_1rt_k{K}_r2', 'IQ2_XXS': 'own_mv_iq2xxs_1rt_k{K}_r2'}
ok_all = True
rng = np.random.default_rng(0)


def check(c, what):
    global ok_all
    print(('PASS ' if c else 'FAIL ') + what, flush=True); ok_all &= bool(c)


def layout(raw, fmt, rows, K):
    """the device layout of rows rows of a matrix (the engine's: Q3MoEOne._w)"""
    return q6k_planes(raw, rows, K) if fmt == 'Q6_K' else iq_records(raw, rows, K, fmt) if fmt in IQ_REC else raw


def raises(fn):
    try: fn()
    except ValueError as e: return str(e)
    return None


# ---- the cases: the first layer of each (role, format) the file has ---------------------------------------------------------
cases = {}
for il in range(NL):
    for role in ROLES:
        t = F.T[f'blk.{il}.{role}.weight']
        if t.type in STD: cases.setdefault((role, t.type, il if A.layers == 'all' else -1), il)
fmts = sorted({f for _, f, _ in cases})
grid_fmts = [f for f in fmts if f in IQ_REC]
print(f'{os.path.basename(A.gguf)}: expert formats {fmts}; {len(cases)} cases, {EL} experts each on the device', flush=True)

# ---- host refusals (no launch; zero buffers of the right sizes) -------------------------------------------------------------
dev = torch.device('cuda:0')
need = sum(2 * EL * F.T[f'blk.{il}.{role}.weight'].n_bytes / NE for (role, _, _), il in cases.items()) / 2 ** 20 + 256
capacity.check([0], [need], f'test_own_iq_ops ({EL} experts of {len(cases)} tensors)')
z16, zo = torch.zeros(2048, dtype=torch.float16, device=dev), torch.zeros(4096, device=dev)
embw, tok, embx = torch.zeros(4 * 144, dtype=torch.uint8, device=dev), torch.zeros(1, dtype=torch.int32, device=dev), torch.zeros(256, device=dev)
zw = lambda fmt: torch.zeros(16 * 8 * k3.OWN_BLOCK[fmt] + 16, dtype=torch.uint8, device=dev)   # 16 rows of K 2048
mv0 = lambda e, fmt: e.own_matvec_1rt(zw(fmt), fmt, 16, 2048, z16, zo)
for fmt in ('IQ4_XS', 'IQ3_XXS', 'IQ3_S', 'IQ2_XXS'):
    msg = raises(lambda: mv0(k3.Engine(module=LEAN, grid=8), fmt))
    check(msg is not None and fmt in msg, f'host: {fmt} on {LEAN} is refused')
e = k3.Engine(module=MOD, grid=8); mv0(e, 'IQ3_S')
msg = raises(lambda: e.build())
check(msg is not None and 'OP_OWN_EMB' in msg, 'host: a tape with a grid format that does not start with the embedding is refused')
e = k3.Engine(module=MOD, grid=8); e.own_emb(embw, 'Q4_K', 4, 256, tok, embx); mv0(e, 'IQ2_XXS'); mv0(e, 'IQ3_XXS')
msg = raises(lambda: e.build())
check(msg is not None and 'IQ2_XXS' in msg, 'host: IQ2_XXS and an IQ3 format on one tape are refused')
for want, fs in ((0, ('Q4_K', 'IQ4_XS')), (1, ('IQ3_XXS', 'Q6_K')), (3, ('IQ3_S', 'IQ3_XXS', 'IQ4_XS')), (4, ('IQ2_XXS', 'Q4_K'))):
    e = k3.Engine(module=MOD, grid=8); e.own_emb(embw, 'Q4_K', 4, 256, tok, embx)
    for f in fs: mv0(e, f)
    e.build()
    check(e.segs[0].out_f32 == want, f'host: stage bits {want} for a tape of {fs}')

lg, ri, rw = torch.zeros(257, device=dev), torch.zeros(8, dtype=torch.int32, device=dev), torch.zeros(8, device=dev)
msg = raises(lambda: k3.Engine(module=MOD, grid=8).router(lg, 8, ri, rw))
check(msg is not None and '257 experts' in msg, f'host: a router over 257 experts is refused on {MOD} (its router ranks {k3.ROUTER_NE[MOD]})')
check(raises(lambda: k3.Engine(module=MOD, grid=8).router(lg[:256], 8, ri, rw)) is None and
      raises(lambda: k3.Engine(module=LEAN, grid=8).router(lg, 8, ri, rw)) is None, f'host: 256 experts are accepted on {MOD}, 257 on {LEAN}')

# ---- device: each case's buffers and its standalone reference ----------------------------------------------------------------
OWN = Module('own_mv')
C = []
for (role, fmt, _), il in sorted(cases.items()):
    t = F.T[f'blk.{il}.{role}.weight']; rows, K = int(t.shape[1]), int(t.shape[0])
    rb = K // 256 * k3.OWN_BLOCK[fmt]
    raw = np.asarray(t.data())[:EL * rows * rb]
    W = torch.from_numpy(np.ascontiguousarray(layout(raw, fmt, EL * rows, K))).to(dev)
    ids_np = rng.choice(EL, KU, replace=False).astype(np.int32)
    per_slot = role == 'ffn_down_exps'
    nx = KU if per_slot else 1
    v = rng.standard_t(3, (nx, K)); v[:, rng.choice(K, 8, replace=False)] *= 20; v = v / np.sqrt((v ** 2).mean())
    x16 = torch.tensor(v, dtype=torch.float16, device=dev).reshape(-1)
    ref = torch.zeros(KU * rows, device=dev)
    kern = getattr(OWN, STD[fmt].format(K=K))
    for s in range(KU):
        e0 = int(ids_np[s]); sl = raw[e0 * rows * rb:(e0 + 1) * rows * rb]
        Ws = torch.from_numpy(np.ascontiguousarray(layout(sl, fmt, rows, K))).to(dev)
        xo = x16[s * K:(s + 1) * K] if per_slot else x16[:K]
        kern(((rows + 15) // 16,), (256,), Ws, ctypes.c_int(rows), xo, ref[s * rows:(s + 1) * rows])
        torch.cuda.synchronize(); del Ws
    C.append(dict(role=role, fmt=fmt, il=il, rows=rows, K=K, rb=rb, raw=raw, W=W, ids=torch.tensor(ids_np, device=dev), ids_np=ids_np,
                  per_slot=per_slot, x16=x16, ref=ref))
check(all(not torch.isnan(c['ref']).any() and float(c['ref'].abs().max()) > 0 for c in C), 'the standalone kernels return finite, non-zero rows')

# ---- one tape with every case, per grid ---------------------------------------------------------------------------------
gu_pairs = [(g, u) for g in C for u in C if g['role'] == 'ffn_gate_exps' and u['role'] == 'ffn_up_exps' and g['fmt'] == u['fmt']
            and g['il'] == u['il']]
runs = []
for nw in (3, 5, 7, 8):
    e = k3.Engine(module=MOD, grid=k3.resident_grid(MOD, nw))
    e.own_emb(embw, 'Q4_K', 4, 256, tok, embx)
    outs, gus = [], []
    for c in C:
        out = torch.full((KU * c['rows'],), float('nan'), device=dev); outs.append(out)
        e.own_matvec_1rt(c['W'], c['fmt'], c['rows'], c['K'], c['x16'], out, ids=c['ids'], total_rows=EL * c['rows'],
                         x_per_slot=c['per_slot'], rows_per_task=16)
    for g, u in gu_pairs:   # the fused gate | up segment: the gate's slots and input, the up tensor of the same layer
        y = torch.full((KU * g['rows'],), float('nan'), dtype=torch.float16, device=dev); gus.append(y)
        e.own_matvec_1rt(g['W'], g['fmt'], g['rows'], g['K'], g['x16'], y, ids=g['ids'], total_rows=EL * g['rows'],
                         rows_per_task=16, up=u['W'])
    e.build().run(); torch.cuda.synchronize()
    runs.append((outs, gus))
want_bits = 0
for f in grid_fmts: want_bits |= k3.OWN_IQ_STAGE[k3.OWN_IQ_FMT[f]]
check(e.segs[0].out_f32 == want_bits, f'the tape stages the grids of {grid_fmts or "no grid format"} (bits {want_bits})')
worst_mean = worst_max = 0.0
for i, c in enumerate(C):
    check(all(torch.equal(r[0][i], c['ref']) for r in runs),
          f"{c['role']} {c['fmt']} layer {c['il']} [{c['rows']} x {c['K']}] x {KU} slots: K3 op bitwise the standalone kernel at 3 / 5 / 7 / 8 per WGP")
    errs = []
    for sl in range(KU):
        e0 = int(c['ids_np'][sl]); blk = c['raw'][e0 * c['rows'] * c['rb']:(e0 + 1) * c['rows'] * c['rb']]
        W64 = own_dequant.dequant(blk, c['fmt'], c['rows'] * c['K']).astype(np.float64).reshape(c['rows'], c['K'])
        xo = c['x16'][sl * c['K']:(sl + 1) * c['K']] if c['per_slot'] else c['x16'][:c['K']]
        yref = W64 @ xo.double().cpu().numpy(); sc = np.sqrt((yref ** 2).mean())
        errs.append(np.abs(runs[0][0][i][sl * c['rows']:(sl + 1) * c['rows']].double().cpu().numpy() - yref) / sc)
    err = np.concatenate(errs); worst_mean, worst_max = max(worst_mean, err.mean()), max(worst_max, err.max())
    check(err.mean() < 1e-6 and err.max() < 2e-5,
          f"{c['role']} {c['fmt']} layer {c['il']}: error vs float64, all {len(err)} rows: mean {err.mean():.1e}, max {err.max():.1e} of the rms (< 1e-6, < 2e-5)")
for j, (g, u) in enumerate(gu_pairs):
    ys = [r[1][j] for r in runs]
    same = all(torch.equal(y.view(torch.int16), ys[0].view(torch.int16)) for y in ys) and not torch.isnan(ys[0]).any()
    # the up tensor's standalone sums on the GATE's slots and input: computed here (the up case above has its own ids)
    uref = torch.zeros(KU * u['rows'], device=dev)
    kern = getattr(OWN, STD[u['fmt']].format(K=u['K']))
    for s in range(KU):
        e0 = int(g['ids_np'][s]); sl = u['raw'][e0 * u['rows'] * u['rb']:(e0 + 1) * u['rows'] * u['rb']]
        Ws = torch.from_numpy(np.ascontiguousarray(layout(sl, u['fmt'], u['rows'], u['K']))).to(dev)
        kern(((u['rows'] + 15) // 16,), (256,), Ws, ctypes.c_int(u['rows']), g['x16'][:g['K']], uref[s * u['rows']:(s + 1) * u['rows']])
        torch.cuda.synchronize(); del Ws
    gd, ud = g['ref'].double().cpu().numpy(), uref.double().cpu().numpy()
    want = gd / (1.0 + np.exp(-gd)) * ud
    got = ys[0].double().cpu().numpy()
    tol = 2e-3 * np.abs(want) + 1e-3 * np.sqrt((want ** 2).mean())
    check(same and bool((np.abs(got - want) <= tol).all()),
          f"gate | up {g['fmt']} (layer {g['il']}): one segment, the same bits at every grid, = silu(gate) * up of the standalone "
          f"sums to fp16 rounding (max |diff| {np.abs(got - want).max():.2e}, rms {np.sqrt((want ** 2).mean()):.2e})")
if not gu_pairs: check(False, 'no gate | up pair of one layer and format in this file')
print(f'{len(C)} cases, {len(gu_pairs)} gate | up pairs; worst error vs float64: mean {worst_mean:.1e}, max {worst_max:.1e} of the rms')
print('RESULT', 'PASS' if ok_all else 'FAIL')
