"""The row-scale study's matvec tile inside K3 (OP_OWN_MV on module k3_q3m_rs4 / k3_q3m_rs2: fixed-width codes with one
scale per row, docs/dev/row-scale-study.md) against its standalone kernels (own_mv_rs) and float64, on a file's real
expert tensors (a file whose experts are Q4_0 or Q2_0 with one scale per row: tools/eval/requant_rows.py --scale row):
  host    the engine's refusals, no launch: a row-scale format on the shipped module and on the other width's module;
          a row-scale segment without its scales, scales on another format
  error   the standalone kernel against float64 (caminus.own_dequant on the CONTAINER's bytes, on the fp16 activations
          the tile reads) on every row of every slot: what is left is the fp32 sums' rounding, so a wrong read of the
          planes (engine.own_layout.rs_planes) or of a row's scale shows
  bitwise each role's segment in K3 is the standalone kernel's rows, per expert slot, at 3 / 5 / 7 / 8 workgroups per WGP
  gate|up the fused segment (fp16(silu(gate . x) * (up . x))) is the same bits at every grid and equals the expression
          on the standalone kernels' sums to fp16 rounding
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  tools/guard.sh LOG 900 python -I tests/engine/test_own_rs_ops.py GGUF [--experts 16]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, ctypes, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
import torch
from caminus import ggufio, own_dequant
from caminus.device import capacity
from caminus.device.hip import Module
from caminus.engine import k3
from caminus.engine.own_layout import rs_planes, RS_CONT

ap = argparse.ArgumentParser()
ap.add_argument('gguf')
ap.add_argument('--experts', type=int, default=16, help='experts of each tensor on the device')
A = ap.parse_args()
F = ggufio.GGUFFile(A.gguf)
NL, KU, NE = int(F.hp('block_count')), int(F.hp('expert_used_count')), int(F.hp('expert_count'))
EL = min(A.experts, NE)
ROLES = ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps')
CONT = F.T['blk.0.ffn_gate_exps.weight'].type
if CONT not in RS_CONT: sys.exit(f'{A.gguf}: the experts are {CONT} ({sorted(RS_CONT)} with one scale per row needed)')
BITS, BV, BS = RS_CONT[CONT]
FMT = f'RS{BITS}'; MOD = next(m for m, f in k3.Q3M_RS_MODULES.items() if f == FMT)
OTHER = next(m for m, f in k3.Q3M_RS_MODULES.items() if f != FMT); LEAN = 'k3_q3m_lean'
ok_all = True
rng = np.random.default_rng(0)


def check(c, what):
    global ok_all
    print(('PASS ' if c else 'FAIL ') + what, flush=True); ok_all &= bool(c)


def raises(fn):
    try: fn()
    except ValueError as e: return str(e)
    return None


cases = [(role, 0) for role in ROLES]
T0 = [F.T[f'blk.{il}.{role}.weight'] for role, il in cases]
if any(t.type != CONT for t in T0): sys.exit(f'{A.gguf}: layer 0\'s experts are not all {CONT}')
print(f'{os.path.basename(A.gguf)}: experts {CONT} -> {FMT} on {MOD}; {len(cases)} cases, {EL} experts each on the device', flush=True)

# ---- host refusals (no launch; zero buffers of the right sizes) -------------------------------------------------------------
dev = torch.device('cuda:0')
need = sum(2 * EL * t.n_bytes / NE for t in T0) / 2 ** 20 + 256
capacity.check([0], [need], f'test_own_rs_ops ({EL} experts of {len(cases)} tensors, {need:.0f} MiB projected)')
print(torch.cuda.get_device_name(0), f'projected {need:.0f} MiB', flush=True)
z16, zo = torch.zeros(2048, dtype=torch.float16, device=dev), torch.zeros(4096, device=dev)
zw = lambda fmt: torch.zeros(16 * 8 * k3.OWN_BLOCK[fmt] + 16, dtype=torch.uint8, device=dev)   # 16 rows of K 2048
zs = torch.zeros(32, device=dev)
mv0 = lambda e, fmt, **kw: e.own_matvec_1rt(zw(fmt), fmt, 16, 2048, z16, zo, **kw)
msg = raises(lambda: mv0(k3.Engine(module=LEAN, grid=8), FMT, scales=zs))
check(msg is not None and FMT in msg, f'host: {FMT} on {LEAN} is refused')
msg = raises(lambda: mv0(k3.Engine(module=OTHER, grid=8), FMT, scales=zs))
check(msg is not None and FMT in msg, f'host: {FMT} on {OTHER} (the other width\'s module) is refused')
msg = raises(lambda: mv0(k3.Engine(module=MOD, grid=8), FMT))
check(msg is not None and 'scales' in msg, f'host: {FMT} without its row scales is refused')
msg = raises(lambda: mv0(k3.Engine(module=MOD, grid=8), 'Q4_K', scales=zs))
check(msg is not None and 'scales' in msg, 'host: row scales on a Q4_K segment are refused')
check(raises(lambda: mv0(k3.Engine(module=MOD, grid=8), FMT, scales=zs)) is None and
      raises(lambda: mv0(k3.Engine(module=MOD, grid=8), 'Q4_K')) is None, f'host: {FMT} with scales and Q4_K are accepted on {MOD}')

# ---- device: each case's buffers, its standalone sums and their error against float64 ----------------------------------------
OWN = Module('own_mv_rs')
C = []
worst_mean = worst_max = 0.0
for (role, il), t in zip(cases, T0):
    rows, K = int(t.shape[1]), int(t.shape[0])
    rb = K // BV * BS                                            # the container's bytes per row
    raw = np.asarray(t.data())[:EL * rows * rb]
    codes, sc = rs_planes(raw, EL * rows, K, CONT)
    cb = K * BITS // 8                                           # the planes' bytes per row
    W, SC = torch.from_numpy(codes).to(dev), torch.from_numpy(sc).to(dev)
    ids_np = rng.choice(EL, KU, replace=False).astype(np.int32)
    per_slot = role == 'ffn_down_exps'
    nx = KU if per_slot else 1
    v = rng.standard_t(3, (nx, K)); v[:, rng.choice(K, 8, replace=False)] *= 20; v = v / np.sqrt((v ** 2).mean())
    x16 = torch.tensor(v, dtype=torch.float16, device=dev).reshape(-1)
    ref = torch.full((KU * rows,), float('nan'), device=dev)
    kern = getattr(OWN, f'own_mv_rs{BITS}_1rt_k{K}')
    errs = []
    for s in range(KU):
        e0 = int(ids_np[s])
        Ws = torch.from_numpy(np.ascontiguousarray(codes[e0 * rows * cb:(e0 + 1) * rows * cb])).to(dev)
        Ss = torch.from_numpy(np.ascontiguousarray(sc[e0 * rows:(e0 + 1) * rows])).to(dev)
        xo = x16[s * K:(s + 1) * K] if per_slot else x16[:K]
        kern(((rows + 15) // 16,), (256,), Ws, Ss, ctypes.c_int(rows), xo, ref[s * rows:(s + 1) * rows])
        torch.cuda.synchronize(); del Ws, Ss
        W64 = own_dequant.dequant(raw[e0 * rows * rb:(e0 + 1) * rows * rb], CONT, rows * K).astype(np.float64).reshape(rows, K)
        yref = W64 @ xo.double().cpu().numpy(); rms = np.sqrt((yref ** 2).mean())
        errs.append(np.abs(ref[s * rows:(s + 1) * rows].double().cpu().numpy() - yref) / rms)
    err = np.concatenate(errs); worst_mean, worst_max = max(worst_mean, float(np.nanmean(err))), max(worst_max, float(np.nanmax(err)))
    check(bool(np.isfinite(err).all()) and err.mean() < 1e-6 and err.max() < 2e-5,
          f'{role} {FMT} layer {il} [{rows} x {K}] x {KU} slots: the standalone kernel against float64, all {len(err)} rows: '
          f'mean {err.mean():.1e}, max {err.max():.1e} of the rms (< 1e-6, < 2e-5)')
    C.append(dict(role=role, il=il, rows=rows, K=K, cb=cb, codes=codes, sc=sc, W=W, SC=SC, ids=torch.tensor(ids_np, device=dev),
                  ids_np=ids_np, per_slot=per_slot, x16=x16, ref=ref))

# ---- one tape with every case, per grid ---------------------------------------------------------------------------------
g, u = C[0], C[1]
assert (g['role'], u['role']) == ('ffn_gate_exps', 'ffn_up_exps') and (g['rows'], g['K']) == (u['rows'], u['K'])
GU = torch.cat([g['SC'], u['SC']]).contiguous()
runs = []
for nw in (3, 5, 7, 8):
    e = k3.Engine(module=MOD, grid=k3.resident_grid(MOD, nw))
    outs = []
    for c in C:
        out = torch.full((KU * c['rows'],), float('nan'), device=dev); outs.append(out)
        e.own_matvec_1rt(c['W'], FMT, c['rows'], c['K'], c['x16'], out, ids=c['ids'], total_rows=EL * c['rows'],
                         x_per_slot=c['per_slot'], rows_per_task=16, scales=c['SC'])
    y = torch.full((KU * g['rows'],), float('nan'), dtype=torch.float16, device=dev)   # the fused gate | up: the gate's slots and input
    e.own_matvec_1rt(g['W'], FMT, g['rows'], g['K'], g['x16'], y, ids=g['ids'], total_rows=EL * g['rows'], rows_per_task=16,
                     up=u['W'], scales=GU)
    e.build().run(); torch.cuda.synchronize()
    runs.append((outs, y))
for i, c in enumerate(C):
    check(all(torch.equal(r[0][i], c['ref']) for r in runs),
          f"{c['role']} {FMT} layer {c['il']}: the K3 op is bitwise the standalone kernel at 3 / 5 / 7 / 8 per WGP")
ys = [r[1] for r in runs]
same = all(torch.equal(yy.view(torch.int16), ys[0].view(torch.int16)) for yy in ys) and not torch.isnan(ys[0]).any()
uref = torch.zeros(KU * u['rows'], device=dev)                  # the up tensor's standalone sums on the GATE's slots and input
kern = getattr(OWN, f"own_mv_rs{BITS}_1rt_k{u['K']}")
for s in range(KU):
    e0 = int(g['ids_np'][s]); rows, cb = u['rows'], u['cb']
    Ws = torch.from_numpy(np.ascontiguousarray(u['codes'][e0 * rows * cb:(e0 + 1) * rows * cb])).to(dev)
    Ss = torch.from_numpy(np.ascontiguousarray(u['sc'][e0 * rows:(e0 + 1) * rows])).to(dev)
    kern(((rows + 15) // 16,), (256,), Ws, Ss, ctypes.c_int(rows), g['x16'][:g['K']], uref[s * rows:(s + 1) * rows])
    torch.cuda.synchronize(); del Ws, Ss
gd, ud = g['ref'].double().cpu().numpy(), uref.double().cpu().numpy()
want = gd / (1.0 + np.exp(-gd)) * ud
got = ys[0].double().cpu().numpy()
tol = 2e-3 * np.abs(want) + 1e-3 * np.sqrt((want ** 2).mean())
check(same and bool((np.abs(got - want) <= tol).all()),
      f'gate | up {FMT} (layer 0): one segment, the same bits at every grid, = silu(gate) * up of the standalone sums to fp16 '
      f'rounding (max |diff| {np.abs(got - want).max():.2e}, rms {np.sqrt((want ** 2).mean()):.2e})')
print(f'{len(C)} cases; worst error vs float64: mean {worst_mean:.1e}, max {worst_max:.1e} of the rms')
print('RESULT', 'PASS' if ok_all else 'FAIL')
sys.exit(0 if ok_all else 1)
