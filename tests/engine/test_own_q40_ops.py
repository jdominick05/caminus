"""The Q4_0 family's matvec tile inside K3 (OP_OWN_MV with K3_Q3M_Q40 on module k3_q3m_q40 and its verify twin
k3_q3m_q40_ver: kernels/own_quant_q4_0.h in place of the Q4_K tile, for the official GPTQ-Int4 checkpoint converted by
tools/eval/gptq_to_gguf.py) against its standalone kernel (own_mv_q4_0_k3, build/own_mv_q35.hsaco) and float64, on the
file's real rows (layer 0 and the last layer: attn_q, attn_k, attn_v, attn_output, and --experts experts of
ffn_gate_exps, ffn_up_exps, ffn_down_exps; the Q6_K head on the same module):
  host    the engine's refusals, no launch: Q4_0 on the shipped lean module, Q4_K on the twin
  error   the standalone kernel against float64 (caminus.own_dequant on the file's bytes, on the fp16 activations the tile
          reads) on every row of every slot: what is left is the fp32 sums' rounding, so a wrong read of the planes
          (engine.own_layout.q4_0_planes) shows (the bars are test_own_rs_ops.py's: mean < 1e-6, max < 2e-5 of the rms)
  bitwise each case's segment in K3 is the standalone kernel's rows (per expert slot for the grouped roles; the head
          against own_mv_q6_k_pl), at 3 / 6 / 7 / 8 workgroups per WGP
  gate|up the fused segment (fp16(silu(gate . x) * (up . x))) is the same bits at every grid and equals the expression
          on the standalone kernels' sums to fp16 rounding
  verify  on k3_q3m_q40_ver, a dense case's 4 tokens in one segment (tokens = 4) are the 4 one-token segments' bits
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  tools/guard.sh LOG 900 python -I tests/engine/test_own_q40_ops.py GGUF [--experts 32]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, ctypes, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
import torch
from caminus import ggufio, own_dequant
from caminus.device import capacity
from caminus.device.hip import Module
from caminus.engine import k3
from caminus.engine.own_layout import q4_0_planes, q6k_planes

ap = argparse.ArgumentParser()
ap.add_argument('gguf')
ap.add_argument('--experts', type=int, default=32, help='experts of each expert tensor on the device')
A = ap.parse_args()
F = ggufio.GGUFFile(A.gguf)
NL, KU, NE = int(F.hp('block_count')), int(F.hp('expert_used_count')), int(F.hp('expert_count'))
EL = min(A.experts, NE)
LEAN = 'k3_q3m_lean'; MOD = k3.Q3M_Q40_OF[LEAN]; VER = k3.Q3M_Q40_OF['k3_q3m_ver']
DENSE = ('attn_q', 'attn_k', 'attn_v', 'attn_output'); EXPS = ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps')
ok_all = True
rng = np.random.default_rng(0)


def check(c, what):
    global ok_all
    print(('PASS ' if c else 'FAIL ') + what, flush=True); ok_all &= bool(c)


def raises(fn):
    try: fn()
    except ValueError as e: return str(e)
    return None


def heavy_x(n, K):
    v = rng.standard_t(3, (n, K)); v[:, rng.choice(K, 8, replace=False)] *= 20
    return v / np.sqrt((v ** 2).mean(axis=1, keepdims=True))


cases = [(role, il) for il in (0, NL - 1) for role in DENSE + EXPS]
T0 = [F.T[f'blk.{il}.{role}.weight'] for role, il in cases]
bad = [(r, il, t.type) for (r, il), t in zip(cases, T0) if t.type != 'Q4_0']
if bad: sys.exit(f'{A.gguf}: not Q4_0: {bad} (this test is for the Q4_0 family: tools/eval/gptq_to_gguf.py)')
TH = F.T['output.weight']
if TH.type != 'Q6_K': sys.exit(f'{A.gguf}: output.weight is {TH.type} (Q6_K expected: the spliced head)')
print(f'{os.path.basename(A.gguf)}: {len(cases)} Q4_0 cases + the Q6_K head on {MOD}; {EL} experts each on the device', flush=True)

dev = torch.device('cuda:0')
need = sum(2 * (EL / NE if role in EXPS else 1) * t.n_bytes for (role, _), t in zip(cases, T0)) / 2 ** 20 + 2 * TH.n_bytes / 2 ** 20 + 512
capacity.check([0], [need], f'test_own_q40_ops ({len(cases)} cases, {need:.0f} MiB projected)')
print(torch.cuda.get_device_name(0), f'projected {need:.0f} MiB', flush=True)

# ---- host refusals (no launch; zero buffers of the right sizes) -------------------------------------------------------------
z16, zo = torch.zeros(2048, dtype=torch.float16, device=dev), torch.zeros(4096, device=dev)
zw = lambda fmt: torch.zeros(16 * 8 * k3.OWN_BLOCK[fmt] + 16, dtype=torch.uint8, device=dev)   # 16 rows of K 2048
mv0 = lambda e, fmt: e.own_matvec_1rt(zw(fmt), fmt, 16, 2048, z16, zo)
msg = raises(lambda: mv0(k3.Engine(module=LEAN, grid=8), 'Q4_0'))
check(msg is not None and 'Q4_0' in msg, f'host: Q4_0 on {LEAN} is refused')
msg = raises(lambda: mv0(k3.Engine(module=MOD, grid=8), 'Q4_K'))
check(msg is not None and 'Q4_K' in msg, f'host: Q4_K on {MOD} is refused (the twin compiles no Q4_K tile)')
check(raises(lambda: mv0(k3.Engine(module=MOD, grid=8), 'Q4_0')) is None and
      raises(lambda: mv0(k3.Engine(module=MOD, grid=8), 'Q6_K')) is None, f'host: Q4_0 and Q6_K are accepted on {MOD}')

# ---- device: each case's buffers, its standalone sums and their error against float64 ----------------------------------------
OWN = Module('own_mv_q35'); OWN6 = Module('own_mv')


def std_q40(Wd, rows, K, x16, out, nt=1):
    OWN.own_mv_q4_0_k3(((rows + 15) // 16,), (256,), Wd, ctypes.c_int(rows), ctypes.c_int(K), x16, out, ctypes.c_int(nt))


C = []
worst_mean = worst_max = 0.0
for (role, il), t in zip(cases, T0):
    rows, K = int(t.shape[1]), int(t.shape[0])
    rb = K // 32 * 18
    grouped = role in EXPS
    ne = EL if grouped else 1
    raw = np.asarray(t.data())[:ne * rows * rb]
    W = torch.from_numpy(q4_0_planes(raw, ne * rows, K)).to(dev)
    S = KU if grouped else 1
    ids_np = rng.choice(EL, KU, replace=False).astype(np.int32) if grouped else None
    per_slot = role == 'ffn_down_exps'
    x16 = torch.tensor(heavy_x(S if per_slot else 1, K), dtype=torch.float16, device=dev).reshape(-1)
    ref = torch.full((S * rows,), float('nan'), device=dev)
    errs = []
    for s in range(S):
        e0 = int(ids_np[s]) if grouped else 0
        sl = raw[e0 * rows * rb:(e0 + 1) * rows * rb]
        Ws = torch.from_numpy(q4_0_planes(sl, rows, K)).to(dev) if grouped else W
        xo = x16[s * K:(s + 1) * K] if per_slot else x16[:K]
        std_q40(Ws, rows, K, xo, ref[s * rows:(s + 1) * rows])
        torch.cuda.synchronize()
        W64 = own_dequant.dequant(sl, 'Q4_0', rows * K).astype(np.float64).reshape(rows, K)
        yref = W64 @ xo.double().cpu().numpy(); rms = np.sqrt((yref ** 2).mean())
        errs.append(np.abs(ref[s * rows:(s + 1) * rows].double().cpu().numpy() - yref) / rms)
        if grouped: del Ws
    err = np.concatenate(errs); worst_mean, worst_max = max(worst_mean, float(np.nanmean(err))), max(worst_max, float(np.nanmax(err)))
    check(bool(np.isfinite(err).all()) and err.mean() < 1e-6 and err.max() < 2e-5,
          f'{role} Q4_0 layer {il} [{rows} x {K}]{" x %d slots" % S if grouped else ""}: the standalone kernel against float64, '
          f'all {len(err)} rows: mean {err.mean():.1e}, max {err.max():.1e} of the rms (< 1e-6, < 2e-5)')
    C.append(dict(role=role, il=il, rows=rows, K=K, rb=rb, raw=raw, W=W, grouped=grouped, S=S, per_slot=per_slot, x16=x16, ref=ref,
                  ids_np=ids_np, ids=torch.tensor(ids_np, device=dev) if grouped else None))

# the head (Q6_K planes, the standalone own_mv_q6_k_pl kernel)
HR, HK = int(TH.shape[1]), int(TH.shape[0])
Wh = torch.from_numpy(q6k_planes(np.asarray(TH.data()), HR, HK)).to(dev)
xh = torch.tensor(heavy_x(1, HK), dtype=torch.float16, device=dev).reshape(-1)
href = torch.full((HR,), float('nan'), device=dev)
getattr(OWN6, f'own_mv_q6_k_pl_k{HK}')(((HR + 15) // 16,), (256,), Wh, ctypes.c_int(HR), xh, href)
torch.cuda.synchronize()

# ---- one tape with every case, per grid ---------------------------------------------------------------------------------
runs = []
for nw in (3, 6, 7, 8):
    e = k3.Engine(module=MOD, grid=k3.resident_grid(MOD, nw))
    outs, ys = [], []
    for c in C:
        out = torch.full((c['S'] * c['rows'],), float('nan'), device=dev); outs.append(out)
        e.own_matvec_1rt(c['W'], 'Q4_0', c['rows'], c['K'], c['x16'], out, ids=c['ids'],
                         total_rows=EL * c['rows'] if c['grouped'] else None, x_per_slot=c['per_slot'], rows_per_task=16)
    for il in (0, NL - 1):   # the fused gate | up: the gate's slots and input
        g, u = (next(c for c in C if c['role'] == r and c['il'] == il) for r in EXPS[:2])
        y = torch.full((KU * g['rows'],), float('nan'), dtype=torch.float16, device=dev); ys.append(y)
        e.own_matvec_1rt(g['W'], 'Q4_0', g['rows'], g['K'], g['x16'], y, ids=g['ids'], total_rows=EL * g['rows'], rows_per_task=16,
                         up=u['W'])
    ho = torch.full((HR,), float('nan'), device=dev)
    e.own_matvec_1rt(Wh, 'Q6_K', HR, HK, xh, ho, rows_per_task=64)
    e.build().run(); torch.cuda.synchronize()
    runs.append((outs, ys, ho))
for i, c in enumerate(C):
    check(all(torch.equal(r[0][i], c['ref']) for r in runs),
          f"{c['role']} Q4_0 layer {c['il']}: the K3 op is bitwise the standalone kernel at 3 / 6 / 7 / 8 per WGP")
check(all(torch.equal(r[2], href) for r in runs) and not torch.isnan(href).any(),
      f'output.weight Q6_K [{HR} x {HK}] on {MOD}: the K3 op is bitwise own_mv_q6_k_pl at 3 / 6 / 7 / 8 per WGP')
for j, il in enumerate((0, NL - 1)):
    g, u = (next(c for c in C if c['role'] == r and c['il'] == il) for r in EXPS[:2])
    ys = [r[1][j] for r in runs]
    same = all(torch.equal(yy.view(torch.int16), ys[0].view(torch.int16)) for yy in ys) and not torch.isnan(ys[0]).any()
    uref = torch.zeros(KU * u['rows'], device=dev)            # the up tensor's standalone sums on the GATE's slots and input
    for s in range(KU):
        e0, rows, rb = int(g['ids_np'][s]), u['rows'], u['rb']
        Ws = torch.from_numpy(q4_0_planes(u['raw'][e0 * rows * rb:(e0 + 1) * rows * rb], rows, u['K'])).to(dev)
        std_q40(Ws, rows, u['K'], g['x16'][:g['K']], uref[s * rows:(s + 1) * rows]); torch.cuda.synchronize(); del Ws
    gd, ud = g['ref'].double().cpu().numpy(), uref.double().cpu().numpy()
    want = gd / (1.0 + np.exp(-gd)) * ud
    got = ys[0].double().cpu().numpy()
    tol = 2e-3 * np.abs(want) + 1e-3 * np.sqrt((want ** 2).mean())
    check(same and bool((np.abs(got - want) <= tol).all()),
          f'gate | up Q4_0 (layer {il}): one segment, the same bits at every grid, = silu(gate) * up of the standalone sums to '
          f'fp16 rounding (max |diff| {np.abs(got - want).max():.2e}, rms {np.sqrt((want ** 2).mean()):.2e})')

# ---- the verify twin: 4 tokens in one segment = 4 one-token segments -------------------------------------------------------
for c in (C[0], C[3]):                                         # attn_q and attn_output of layer 0
    rows, K = c['rows'], c['K']
    x4 = torch.tensor(heavy_x(4, K), dtype=torch.float16, device=dev).reshape(-1)
    o4 = torch.full((4 * rows,), float('nan'), device=dev); o1 = torch.full((4 * rows,), float('nan'), device=dev)
    e = k3.Engine(module=VER, grid=k3.resident_grid(VER))   # (T = 1: the qwen3moe verify ops take tokens per segment)
    e.own_matvec_1rt(c['W'], 'Q4_0', rows, K, x4, o4, rows_per_task=16, tokens=4, x_stride=K, out_stride=rows)
    e.build().run()
    for j in range(4):
        e = k3.Engine(module=VER, grid=k3.resident_grid(VER))
        e.own_matvec_1rt(c['W'], 'Q4_0', rows, K, x4[j * K:(j + 1) * K], o1[j * rows:(j + 1) * rows], rows_per_task=16)
        e.build().run()
    s1 = torch.full((4 * rows,), float('nan'), device=dev)
    for j in range(4): std_q40(c['W'], rows, K, x4[j * K:(j + 1) * K], s1[j * rows:(j + 1) * rows])
    torch.cuda.synchronize()
    check(torch.equal(o4, o1) and torch.equal(o1, s1) and not torch.isnan(o4).any(),
          f"{c['role']} Q4_0 layer {c['il']} on {VER}: 4 tokens in one segment = 4 one-token segments = the standalone kernel, bit for bit")

print(f'{len(C)} cases; worst error vs float64: mean {worst_mean:.1e}, max {worst_max:.1e} of the rms')
print('RESULT', 'PASS' if ok_all else 'FAIL')
sys.exit(0 if ok_all else 1)
