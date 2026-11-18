"""The long-context mode's RoPE (YaRN x4, caminus.rope.yarn_table) in the kernels, alone, at Qwen3-30B-A3B's shapes (32 query /
4 kv heads, head dim 128) on known inputs; no model file. The decode op (OP_OWN_ATTS on the YaRN build of k3_q3m_lean,
kernels/k3_own_atts.h under K3_Q3M_YARN) and the prefill row kernel (q3p_qk_rope from q3m_prefill_yarn, Q3P_YARN) read the
same par block (q_norm | k_norm | theta_scale, eps | freq [64] | mscale):
  table     the block's freq / mscale are caminus.rope.yarn_table (x4 from 32,768, base 1e6): correction dims [23, 40],
            pairs 0 .. 23 at their trained frequency, 40 .. 63 at a quarter, mscale fp32(1 + 0.1 ln 4) = 1.1386294
  row       at positions 0, 1, 63, 64, 1,000, 32,767, 32,768, 40,960, 65,536, 100,000 and 131,071 the decode op's cache row
            (its K and V halves) is bit for bit q3p_qk_rope's for the same q | k | v (one prompt token at that position)
  oracle    K against float64 norm + NEOX RoPE with the kernel's fp32 angle (fp32(pos) x fp32 freq, rounded to fp32: at
            131K that rounding is larger than fp16's, and llama.cpp's fp32 angle has it too) and float64 cos / sin x mscale:
            |K - ref| <= 1 fp16 ulp of ref + 1e-5 x max |ref| per element
  read      the op's fp16 attention output against float64 attention of the oracle's q over the rows 0 .. p (prior rows
            known fp16 values, the own row the op's): <= 2 fp16 ulps + 1e-5 x max |V|, at 0, 1, 1,000 and 40,960
  native    the same inputs through the native module and q3p_qk_rope at position 1,000 give another row (the table is used)
  grids     the rows and outputs the same bits at 3 / 6 / 7 workgroups per WGP
--kv q8: the Q8_0 builds (k3_q3m_lean_kvq8_yarn, q3p_qk_rope_q8 from q3m_prefill_kvq8_yarn): the decode row and the prefill
row are both byte for byte caminus.ref_kv.encode_q8_kv(hadamard128(K), V) of the F16 YaRN kernel's halves.
--tp: the two-card module's YaRN build (k3_q3m_lean_tp_yarn, or k3_q3m_lean_tp_kvq8_yarn with --kv q8) at one card's shapes of
the two-card engine (16 query / 2 kv heads, up to 128 splits), on one card alone.
--q40: the same builds of the Q4_0 family (k3.Q3M_Q40_OF: k3_q3m_q40_yarn, ..._kvq8_yarn, ..._tp_yarn, ..._tp_kvq8_yarn), whose
attention statements are their bases'; the prefill row kernel is the same object.
  tools/guard.sh LOG 600 python -I tests/engine/test_q3m_yarn_rope.py [--kv q8] [--tp] [--q40]
New kernel code: a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>). Prints PASS / FAIL and RESULT."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
import torch
from caminus import ref_kv, rope
from caminus.device import capacity
from caminus.engine import k3, q3moe_one
from caminus.prefill import q3m_ops

ap = argparse.ArgumentParser(); ap.add_argument('--kv', default='f16', choices=('f16', 'q8')); ap.add_argument('--tp', action='store_true')
ap.add_argument('--q40', action='store_true', help='the Q4_0 family\'s YaRN twins (k3.Q3M_Q40_OF)')
A = ap.parse_args()
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


Q8 = A.kv == 'q8'
NH, NKV, HD = (16, 2, 128) if A.tp else (32, 4, 128)
M = NKV * HD
base = 'k3_q3m_lean_tp' if A.tp else 'k3_q3m_lean'
NATIVE = k3.Q3M_KV_Q8_OF[base] if Q8 else base
if A.q40: NATIVE = k3.Q3M_Q40_OF[NATIVE]
MOD = k3.Q3M_YARN_OF[NATIVE]
LAY = ref_kv.Q8KVLayout(NKV, HD)
RB = LAY.row_bytes if Q8 else 2 * M * 2
POS = (0, 1, 63, 64, 1000, 32767, 32768, 40960, 65536, 100000, 131071)
READ = (0, 1, 1000, 40960)
ROWS = 131072 + k3.Engine.ATS_PAD
dev = torch.device('cuda:0')
print(torch.cuda.get_device_name(0), f'module {MOD} (native {NATIVE}), {NH} / {NKV} heads', flush=True)
capacity.check([0], [(2 * ROWS * RB + ROWS * 2 * M * 2) / 2 ** 20 + 32], 'test_q3m_yarn_rope: the caches and buffers')

# -- the table -----------------------------------------------------------------------------------------------------------
tab = rope.yarn_table(HD, 1e6, 4.0, 32768)
lo, hi = rope.corr_dims(HD, 32768, 1e6)
inv = 1e6 ** (-2.0 * np.arange(HD // 2) / HD)
check((lo, hi) == (23.0, 40.0) and np.array_equal(tab[:24], inv[:24].astype(np.float32))
      and np.array_equal(tab[40:64], (inv[40:] / 4).astype(np.float32)) and tab[64] == np.float32(1 + 0.1 * np.log(4.0)),
      f'table: correction dims [{lo:g}, {hi:g}], pairs 0 .. 23 trained, 40 .. 63 a quarter, mscale {tab[64]!r}')

rng = np.random.default_rng(23)
theta_scale = float(np.float32(np.float32(1e6) ** np.float32(-2.0 / HD)))
eps = 1e-6
qn = (0.5 + rng.random(HD)).astype(np.float32); kn = (0.5 + rng.random(HD)).astype(np.float32)
par_n = np.concatenate([qn, kn, np.array([theta_scale, eps], np.float32)])
par = torch.tensor(np.concatenate([par_n, tab]), device=dev)
par_native = torch.tensor(par_n, device=dev)
# prior rows: fp16-valued K / V (as after the norm), F16 halves; the Q8 build's cache holds their encode
Kh = (rng.standard_normal((ROWS, NKV, HD)) * 0.8).astype(np.float16)
Vh = (rng.standard_normal((ROWS, NKV, HD)) * 0.5).astype(np.float16)
f16_prior = np.concatenate([Kh.reshape(ROWS, M), Vh.reshape(ROWS, M)], 1)
if Q8:
    prior = ref_kv.encode_q8_kv(ref_kv.hadamard128(Kh.astype(np.float32)), Vh.astype(np.float32), LAY).reshape(-1)
    cache = torch.from_numpy(prior.copy()).to(dev); pcache = torch.from_numpy(prior.copy()).to(dev)
else:
    cache = torch.from_numpy(f16_prior.reshape(-1).copy()).to(dev); pcache = cache.clone()
f16c = torch.zeros(ROWS * 2 * M, dtype=torch.float16, device=dev)     # (the F16 YaRN kernel's rows: the oracle of both forms)
qkv = torch.zeros(NH * HD + 2 * M, device=dev)
pos_d = torch.zeros(1, dtype=torch.int32, device=dev)
out16 = torch.zeros(NH * HD, dtype=torch.float16, device=dev)
q16 = torch.zeros(NH * HD, dtype=torch.float16, device=dev)
kverr = torch.zeros(1, dtype=torch.int32, device=dev)
S = q3moe_one.splits_for(NKV, k3.atts_max_splits(MOD))
scratch = torch.zeros(k3.Engine.atts_scratch_floats(NKV, S), device=dev)
ops_f16 = q3m_ops.Ops(dev, 'q3m_prefill', yarn=True)               # (the F16 YaRN row kernel)
ops = q3m_ops.Ops(dev, 'q3m_prefill', q8=Q8, yarn=True)             # (the form under test)
scale = 1.0 / np.sqrt(HD)
H128 = np.array([[1.0]])
for _ in range(7): H128 = np.block([[H128, H128], [H128, -H128]])
H128 /= np.sqrt(HD)   # (the orthonormal Sylvester Hadamard128, float64, as tests/engine/test_q3m_kvq8_atts.py)


def tape(mod, p_, per_wgp=None):
    E = k3.Engine(module=mod, grid=k3.resident_grid(mod, per_wgp), T=1, limit_us=200_000)
    E.own_atts(qkv, scratch, p_, cache, pos_d, out16, NH, NKV, HD, scale, splits=S, lmin=q3moe_one.ATS_LMIN)
    return E.build()


def host_rope(x, w, p):
    """float64 norm + NEOX RoPE of rows x [h][128] at position p with the kernel's fp32 angle and the table's mscale"""
    x = x.astype(np.float64)
    y = x / np.sqrt((x * x).mean(1, keepdims=True) + eps) * w
    th = (np.float32(p) * tab[:64]).astype(np.float64)
    c, s = np.cos(th) * float(tab[64]), np.sin(th) * float(tab[64])
    return np.concatenate([y[:, :64] * c - y[:, 64:] * s, y[:, 64:] * c + y[:, :64] * s], 1)


def row(c, p):
    return c[p * RB:(p + 1) * RB].cpu().numpy() if Q8 else c[p * 2 * M:(p + 1) * 2 * M].view(torch.int16).cpu().numpy()


E = tape(MOD, par)
same, orc, worst, read_ok = True, True, -np.inf, True
outs = {}
for p in POS:
    x = (rng.standard_normal(NH * HD + 2 * M) * 1.3).astype(np.float32)
    qkv.copy_(torch.from_numpy(x)); pos_d.fill_(p)
    E.run()
    ops_f16.qk_rope(qkv, NH * HD + 2 * M, NH, NKV, par, p, q16, f16c, 1)
    ops.qk_rope(qkv, NH * HD + 2 * M, NH, NKV, par, p, q16, pcache, 1, err=kverr if Q8 else None)
    torch.cuda.synchronize()
    h = f16c[p * 2 * M:(p + 1) * 2 * M].float().cpu().numpy()
    if Q8:
        want = ref_kv.encode_q8_kv(ref_kv.hadamard128(h[:M].reshape(1, NKV, HD)), h[M:].reshape(1, NKV, HD), LAY)[0]
        same &= np.array_equal(row(cache, p), want) and np.array_equal(row(pcache, p), want)
    else:
        same &= np.array_equal(row(cache, p), row(pcache, p))
    rk = host_rope(x[NH * HD:NH * HD + M].reshape(NKV, HD), kn, p)
    ulp = np.abs(np.spacing(np.abs(rk).astype(np.float16))).astype(np.float64)
    d = np.abs(h[:M].reshape(NKV, HD) - rk)
    worst = max(worst, float((d / (ulp + 1e-5 * np.abs(rk).max())).max()))
    orc &= bool((d <= ulp + 1e-5 * np.abs(rk).max()).all())
    if p in READ:   # (float64 attention of the oracle q over the rows the op read: the prior rows and its own)
        if Q8:
            dk, dv = ref_kv.decode_q8_kv(cache[:(p + 1) * RB].cpu().numpy().reshape(p + 1, RB), LAY)
            q = host_rope(x[:NH * HD].reshape(NH, HD), qn, p) @ H128   # (rotated q against rotated K)
        else:
            rows_ = cache[:(p + 1) * 2 * M].float().cpu().numpy().reshape(p + 1, 2, NKV, HD)
            dk, dv = rows_[:, 0], rows_[:, 1]
            q = host_rope(x[:NH * HD].reshape(NH, HD), qn, p)
        o = out16.float().cpu().numpy().reshape(NH, HD); r = np.zeros((NH, HD))
        for hh in range(NH):
            sc = dk[:, hh // 8].astype(np.float64) @ q[hh] * scale
            w = np.exp(sc - sc.max()); w /= w.sum(); r[hh] = w @ dv[:, hh // 8].astype(np.float64)
        u = np.abs(np.spacing(np.abs(r).astype(np.float16))).astype(np.float64)
        read_ok &= bool((np.abs(o - r) <= 2 * u + 1e-5 * float(np.abs(dv).max())).all())
    outs[p] = (out16.float().cpu().numpy().copy(), row(cache, p).copy(), x)
check(same, f'row: the decode op\'s row is bit for bit the prefill kernel\'s{" (both encode_q8_kv(hadamard128(K), V) of the F16 YaRN halves)" if Q8 else ""} at {POS}')
check(orc, f'oracle: K within 1 fp16 ulp + 1e-5 x max of float64 RoPE on the kernel\'s fp32 angle (worst {worst:.2f} of the bound)')
check(read_ok, f'read: the attention output within 2 fp16 ulps + 1e-5 x max |V| of float64 at {READ}')
check(int(kverr.item()) == 0, 'no refused block in the prefill\'s error word')

# -- native differs ------------------------------------------------------------------------------------------------------
p = 1000; x = outs[p][2]
qkv.copy_(torch.from_numpy(x)); pos_d.fill_(p)
tape(NATIVE, par_native).run(); torch.cuda.synchronize()
check(not np.array_equal(row(cache, p), outs[p][1]), 'native: the native module writes another row at position 1,000 (the YaRN table is read)')

# -- grids ---------------------------------------------------------------------------------------------------------------
gs = True
for g in (3, 6, 7):
    Eg = tape(MOD, par, g)
    for p in (1000, 65536, 131071):
        qkv.copy_(torch.from_numpy(outs[p][2])); pos_d.fill_(p); Eg.run(); torch.cuda.synchronize()
        gs &= np.array_equal(out16.float().cpu().numpy(), outs[p][0]) and np.array_equal(row(cache, p), outs[p][1])
check(gs, 'grids: rows and outputs the same bits at 3 / 6 / 7 workgroups per WGP (positions 1,000, 65,536, 131,071)')
print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
