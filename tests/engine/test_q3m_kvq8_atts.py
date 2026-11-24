"""The Q8 KV decode attention (OP_OWN_ATTS on k3_q3m_lean_kvq8, kernels/k3_own_atts.h under K3_Q3M_KV_Q8) alone, at Qwen3-30B-A3B's
shapes (32 query / 4 kv heads, head dim 128), on known inputs; no model file. One segment per launch, on a cache whose rows
0 .. p - 1 hold caminus.ref_kv.encode_q8_kv(hadamard128(K), V) of known fp16 values (K rotated, as the op writes them):
  row       the position's row the op writes is byte for byte encode_q8_kv(hadamard128(K), V) of the K / V halves the F16
            path stores for the same q | k | v (the F16 prefill kernel q3p_qk_rope, OP_OWN_ATTS's norm + RoPE statements, on
            the same inputs; caminus.ref_kv.hadamard128 is kernels/own_kv_rot128.h's fp32 transform in its order)
  read      the fp16 output against float64 attention of the float64-rotated q (H q, H the orthonormal Sylvester
            Hadamard128) over the decoded rows 0 .. p (the own row decoded too), per element
            |out - ref| <= 2 fp16 ulps of ref + 1e-5 x max |V|, at positions across one / many splits and chunk edges (0, 1,
            63, 64, 65, 127, 1000, 3071, 4000). The decoded values are exact in fp32, so the bound is the F16 arm's
            1e-5 x max |V| plus the output's fp16 rounding. A CPU fp32 emulation of the op's order (splits, 64-position chunks,
            online softmax, split merge) on these inputs stayed within one ulp + 1e-7 x max |V|. Faults injected into it at
            depth (one chunk's V on the neighbouring block's scale, chunk 1 rows shifted by one, one V row from its
            neighbour: 6e-4 .. 4e-3 x max |V|) all fail this bound. The earlier bound (4e-3 x max |V|) passed every one
  grids     the outputs and rows the same bits at 3 / 6 / 7 workgroups per WGP
  edges     fixed fp16 V blocks through the op: +-65,504, exact ties (x / d = k + 0.5: roundf away from zero), a block whose
            stored half scale rounds to zero (+-2^-20), a zero block: the row byte for byte the reference
  refusals  a NaN in v, and a v that overflows fp16 (1e5): InvalidKVEncoding from the error word after Sync; the refused
            block's bytes untouched, the other blocks of the row written. A NaN in k: its head's k norm makes the whole
            head NaN (as without the rotation), so all four K blocks of that kv head are refused (untouched), the other
            heads' blocks written. One RoPE'd k value past fp16 (dims 5 / 69, blocks 0 / 2, from a large k norm weight):
            the transform spreads it, so all four K blocks of each kv head are refused, blocks 1 / 3 included
  prefill   q3p_qk_rope_q8 (75 tokens at position 1,000, with the edge blocks) writes encode_q8_kv(hadamard128(K), V) of
            q3p_qk_rope's halves, byte for byte, and its q16 is fp16(hadamard128(q3p_qk_rope's q16)), bit for bit; it refuses
            a NaN block into its error word; q3mp_attn_q8 over Q8 rows equals q3mp_attn over an F16 cache holding
            fp16(decoded rows), bit for bit (the Q8 staging's values are exactly those halves)
  tools/guard.sh LOG 600 python -I tests/engine/test_q3m_kvq8_atts.py [--tp]
--tp: the same checks on the two-card module's Q8 build (k3_q3m_lean_tp_kvq8) at one card's shapes of the two-card engine
(16 query / 2 kv heads), launched on one card alone: its attention takes up to 128 splits per kv head (the card's own
q3moe_one.splits_for(2, 128): XTX 96, XT 105), so the positions run past 64 splits x 64 positions (4,096 .. 6,100: more
than 64 active splits, the two-card merge's second round of m | l loads and the V scales' LDS beside it) to 8,000, on a
cache of 8,192 rows.
New kernel code: a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>). Prints PASS / FAIL and RESULT."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
import torch
from caminus import errors, ref_kv
from caminus.device import capacity
from caminus.engine import k3, q3moe_one
from caminus.prefill import q3m_ops

ap = argparse.ArgumentParser(); ap.add_argument('--tp', action='store_true', help='the two-card module (2 kv heads, 128 splits)')
A = ap.parse_args()
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


NH, NKV, HD = (16, 2, 128) if A.tp else (32, 4, 128)   # (--tp: one card's heads of the two-card engine)
M = NKV * HD
MOD = k3.Q3M_KV_Q8_OF['k3_q3m_lean_tp' if A.tp else 'k3_q3m_lean']
LAY = ref_kv.Q8KVLayout(NKV, HD)
RB = LAY.row_bytes
ROWS = (8192 if A.tp else 4096) + k3.Engine.ATS_PAD
POS = (0, 1, 63, 64, 65, 127, 1000, 3071, 4000) + ((4095, 4096, 4160, 5000, 6000, 6100, 8000) if A.tp else ())
dev = torch.device('cuda:0')
print(torch.cuda.get_device_name(0), f'module {MOD}, {NH} / {NKV} heads', flush=True)
capacity.check([0], [(ROWS * RB + ROWS * 2 * M * 2) / 2 ** 20 + 16], 'test_q3m_kvq8_atts: the Q8 cache, the F16 oracle cache, buffers')
rng = np.random.default_rng(11)
theta_scale = float(np.float32(np.float32(1e6) ** np.float32(-2.0 / HD)))
eps = 1e-6
qn = (0.5 + rng.random(HD)).astype(np.float32); kn = (0.5 + rng.random(HD)).astype(np.float32)
par = torch.tensor(np.concatenate([qn, kn, np.array([theta_scale, eps], np.float32)]), device=dev)
# the prior rows: fp16-valued K (unit-ish, as after k_norm) and V
Kh = (rng.standard_normal((ROWS, NKV, HD)) * 0.8).astype(np.float16).astype(np.float32)
Vh = (rng.standard_normal((ROWS, NKV, HD)) * 0.5).astype(np.float16).astype(np.float32)
prior = ref_kv.encode_q8_kv(ref_kv.hadamard128(Kh), Vh, LAY)
cache = torch.from_numpy(prior.reshape(-1).copy()).to(dev)
f16c = torch.zeros(ROWS * 2 * M, dtype=torch.float16, device=dev)
qkv = torch.zeros(NH * HD + 2 * M, device=dev)
pos_d = torch.zeros(1, dtype=torch.int32, device=dev)
out16 = torch.zeros(NH * HD, dtype=torch.float16, device=dev)
q16 = torch.zeros(NH * HD, dtype=torch.float16, device=dev)
S = q3moe_one.splits_for(NKV, k3.atts_max_splits(MOD))   # (the engines' count: 64 cap on one card, 128 on the two-card module)
scratch = torch.zeros(k3.Engine.atts_scratch_floats(NKV, S), device=dev)
ops = q3m_ops.Ops(dev, 'q3m_prefill')   # (the F16 kernels: the row oracle)
scale = 1.0 / np.sqrt(HD)
H128 = np.array([[1.0]])
for _ in range(7): H128 = np.block([[H128, H128], [H128, -H128]])
H128 /= np.sqrt(HD)   # (the orthonormal Sylvester Hadamard128, float64; symmetric)


def oracle(h, n=1):
    """the Q8 rows for n positions of F16 halves h ([n][2 M] or flat): encode_q8_kv(hadamard128(K), V)"""
    h = np.asarray(h, np.float32).reshape(n, 2, NKV, HD)
    return ref_kv.encode_q8_kv(ref_kv.hadamard128(np.ascontiguousarray(h[:, 0])), np.ascontiguousarray(h[:, 1]), LAY)


def tape(per_wgp):
    E = k3.Engine(module=MOD, grid=k3.resident_grid(MOD, per_wgp), T=1, limit_us=200_000)
    E.own_atts(qkv, scratch, par, cache, pos_d, out16, NH, NKV, HD, scale, splits=S, lmin=q3moe_one.ATS_LMIN)
    return E.build()


def host_q(x, p):
    """float64 q norm + NEOX RoPE (the op's math; the device runs it in fp32)"""
    x = x.astype(np.float64).reshape(NH, HD)
    y = x / np.sqrt((x * x).mean(1, keepdims=True) + eps) * qn
    j = np.arange(HD // 2); th = p * np.float64(theta_scale) ** j
    c, s = np.cos(th), np.sin(th)
    return np.concatenate([y[:, :64] * c - y[:, 64:] * s, y[:, 64:] * c + y[:, :64] * s], 1)


def ref_attn(q, rows_k, rows_v):
    out = np.zeros((NH, HD))
    for h in range(NH):
        g = h // 8
        sc = rows_k[:, g] @ q[h] * scale
        w = np.exp(sc - sc.max()); w /= w.sum()
        out[h] = w @ rows_v[:, g]
    return out


E = tape(None)
rows_ok, read_ok, worst, most = True, True, -np.inf, 0
outs = {}
for p in POS:
    x = (rng.standard_normal(NH * HD + 2 * M) * 1.3).astype(np.float32)
    qkv.copy_(torch.from_numpy(x)); pos_d.fill_(p)
    E.run()
    # the oracle row: the F16 kernel's halves for the same inputs, encoded
    ops.qk_rope(qkv, NH * HD + 2 * M, NH, NKV, par, p, q16, f16c, 1)
    torch.cuda.synchronize()
    h = f16c[p * 2 * M:(p + 1) * 2 * M].float().cpu().numpy()
    want = oracle(h)[0]
    got = cache[p * RB:(p + 1) * RB].cpu().numpy()
    rows_ok &= np.array_equal(got, want)
    dk, dv = ref_kv.decode_q8_kv(cache[:(p + 1) * RB].cpu().numpy().reshape(p + 1, RB), LAY)
    r = ref_attn(host_q(x[:NH * HD], p) @ H128, dk.astype(np.float64), dv.astype(np.float64))   # (rotated q against rotated K)
    o = out16.float().cpu().numpy().reshape(NH, HD)
    vmax = float(np.abs(dv).max()); d = np.abs(o - r)
    ulp = np.abs(np.spacing(np.abs(r).astype(np.float16))).astype(np.float64)   # (fp16 ulp at each reference value)
    e = float(((d - 2 * ulp) / vmax).max()); worst = max(worst, e)             # (<= 1e-5: inside the bound)
    read_ok &= bool((d <= 2 * ulp + 1e-5 * vmax).all())
    outs[p] = (o.copy(), got.copy(), x)
    Lp = -(-(p + 1) // S); Lp = max(-(-Lp // 64) * 64, q3moe_one.ATS_LMIN); nact = -(-(p + 1) // Lp)   # (kernels/k3_own_atts.h ats_layout)
    most = max(most, nact)
    print(f'  pos {p} ({nact} active splits of {Lp}): row {"=" if np.array_equal(got, want) else "!="} encode(F16 row); max |out - float64| / max|V| {float(d.max()) / vmax:.2e}, '
          f'past 2 fp16 ulps {e:.2e} (bound 1e-5)', flush=True)
check(rows_ok, f'the row written at each of {len(POS)} positions is byte for byte encode_q8_kv(hadamard128(K), V) of the F16 kernel\'s halves')
check(read_ok, f'outputs within 2 fp16 ulps + 1e-5 x max |V| of float64 attention over the decoded rows, every element '
               f'(worst past 2 ulps {worst:.2e} x max |V|; splits {S})')
if A.tp:
    check(most > 64, f'--tp: the positions reached {most} active splits (more than 64: the two-card merge\'s second round of loads; splits {S})')
# grids: the same bits
gr_ok = True
for pw in (3, 6, 7):
    Eg = tape(pw)
    for p in POS:
        o0, row0, x = outs[p]
        qkv.copy_(torch.from_numpy(x)); pos_d.fill_(p); Eg.run(); torch.cuda.synchronize()
        gr_ok &= np.array_equal(out16.float().cpu().numpy().reshape(NH, HD), o0) and np.array_equal(cache[p * RB:(p + 1) * RB].cpu().numpy(), row0)
check(gr_ok, 'outputs and rows the same bits at 3 / 6 / 7 workgroups per WGP')
# refusals: a NaN v, an fp16-overflowing v; a zero block
for what, val in (('NaN', np.nan), ('1e5 (fp16 inf)', 1e5)):
    p = 2000
    x = (rng.standard_normal(NH * HD + 2 * M)).astype(np.float32)
    x[NH * HD + M + 3] = val                       # (V of kv head 0, block 0)
    cache[p * RB:(p + 1) * RB] = 0xA5
    qkv.copy_(torch.from_numpy(x)); pos_d.fill_(p)
    try: E.run(); raised = False
    except errors.InvalidKVEncoding: raised = True
    torch.cuda.synchronize()
    row = cache[p * RB:(p + 1) * RB].cpu().numpy()
    vb0 = (M // 32) * 34
    untouched = bool((row[vb0:vb0 + 34] == 0xA5).all()); written = not bool((row[:vb0] == 0xA5).all())
    check(raised and int(E.kv_error[0]) == 2 and untouched and written,
          f'a {what} in v: InvalidKVEncoding (error word 2), the refused block untouched, the K blocks written')
    E.sync.zero_(); scratch[:k3.Engine.ATS_CNT].zero_()
# a NaN in k (kv head 1): the head's k norm makes all its 128 values NaN, so its four K blocks are refused
p = 2050
x = (rng.standard_normal(NH * HD + 2 * M)).astype(np.float32)
x[NH * HD + HD + 7] = np.nan
cache[p * RB:(p + 1) * RB] = 0xA5
qkv.copy_(torch.from_numpy(x)); pos_d.fill_(p)
try: E.run(); raised = False
except errors.InvalidKVEncoding: raised = True
torch.cuda.synchronize()
row = cache[p * RB:(p + 1) * RB].cpu().numpy()
blk = lambda part, g, b: row[(part * (M // 32) + 4 * g + b) * 34:(part * (M // 32) + 4 * g + b + 1) * 34]
check(raised and int(E.kv_error[0]) == 2 and all((blk(0, 1, b) == 0xA5).all() for b in range(4))
      and not any((blk(0, g, 0) == 0xA5).all() for g in range(NKV) if g != 1) and not any((blk(1, g, b) == 0xA5).all() for g in range(NKV) for b in range(4)),
      'a NaN in k: InvalidKVEncoding (error word 2), its kv head\'s four K blocks untouched, the other heads\' K and every V block written')
E.sync.zero_(); scratch[:k3.Engine.ATS_CNT].zero_()
# one RoPE'd k value past fp16 (a k norm weight of 1e9 at dim 5, the input 3.0 there): finite through the norm and the
# RoPE in fp32, inf once rounded to fp16. Dims 5 and 69 (RoPE's pair) sit in blocks 0 and 2; without the transform blocks 1
# and 3 would be written. The transform spreads the inf over every dim of the head, so all four K blocks of every kv
# head (the weight is shared) are refused; every V block is written
p = 2060
x = (rng.standard_normal(NH * HD + 2 * M)).astype(np.float32)
for g in range(NKV): x[NH * HD + g * HD + 5] = 3.0
kn5 = float(par[HD + 5]); par[HD + 5] = 1e9
cache[p * RB:(p + 1) * RB] = 0xA5
qkv.copy_(torch.from_numpy(x)); pos_d.fill_(p)
try: E.run(); raised = False
except errors.InvalidKVEncoding: raised = True
torch.cuda.synchronize(); par[HD + 5] = kn5
row = cache[p * RB:(p + 1) * RB].cpu().numpy()
check(raised and int(E.kv_error[0]) == 2 and all((blk(0, g, b) == 0xA5).all() for g in range(NKV) for b in range(4))
      and not any((blk(1, g, b) == 0xA5).all() for g in range(NKV) for b in range(4)),
      'a RoPE\'d k value past fp16 (dims 5 / 69 only, blocks 0 / 2): InvalidKVEncoding (error word 2), all four K blocks of '
      'each kv head untouched (the transform spreads it), every V block written')
E.sync.zero_(); scratch[:k3.Engine.ATS_CNT].zero_()
# edges: fixed fp16 V blocks of kv head 0 (blocks 0 .. 3)
EDGE = np.zeros(128, np.float32)
EDGE[0:32] = np.where(np.arange(32) % 2, 65504.0, -65504.0)
EDGE[32] = 127.0; EDGE[33:64] = np.array([k + 0.5 for k in range(-15, 16)], np.float32)   # (d = 1: x / d = k + 0.5 exactly)
EDGE[64:96] = np.where(np.arange(32) % 2, 2.0 ** -20, -(2.0 ** -20))                     # (d = 2^-20 / 127: half(d) = 0)
EDGE = EDGE.astype(np.float16).astype(np.float32)
p = 2100
x = (rng.standard_normal(NH * HD + 2 * M)).astype(np.float32); x[NH * HD + M:NH * HD + M + 128] = EDGE
qkv.copy_(torch.from_numpy(x)); pos_d.fill_(p); E.run()
ops.qk_rope(qkv, NH * HD + 2 * M, NH, NKV, par, p, q16, f16c, 1); torch.cuda.synchronize()
h = f16c[p * 2 * M:(p + 1) * 2 * M].float().cpu().numpy()
want = oracle(h)[0]
vb = (M // 32) * 34
tie_codes = want[vb + 34 + 3:vb + 68].view(np.int8)
check(np.array_equal(cache[p * RB:(p + 1) * RB].cpu().numpy(), want) and int(E.kv_error[0]) == 0
      and np.array_equal(tie_codes, np.array([k + (1 if k >= 0 else 0) if k >= 0 else k for k in range(-15, 16)], np.int8))
      and want[vb + 68:vb + 70].tobytes() == b'\x00\x00',
      'edge V blocks (+-65,504, exact ties, a stored scale that rounds to zero, zeros): the row byte for byte the reference; no error')

# ---- the prefill's Q8 kernels -------------------------------------------------------------------------------------------------
oq8 = q3m_ops.Ops(dev, 'q3m_prefill', q8=True)
kverr = torch.zeros(1, dtype=torch.int32, device=dev)
pos0, n = 1000, 75
xs = (rng.standard_normal((n, NH * HD + 2 * M)) * 1.3).astype(np.float32)
xs[7, NH * HD + M:NH * HD + M + 128] = EDGE
qd = torch.from_numpy(xs).to(dev)
q16a = torch.zeros(n * NH * HD, dtype=torch.float16, device=dev); q16b = torch.zeros_like(q16a)
oq8.qk_rope(qd, NH * HD + 2 * M, NH, NKV, par, pos0, q16a, cache, n, err=kverr)
ops.qk_rope(qd, NH * HD + 2 * M, NH, NKV, par, pos0, q16b, f16c, n); torch.cuda.synchronize()
want = oracle(f16c[pos0 * 2 * M:(pos0 + n) * 2 * M].float().cpu().numpy(), n)
qwant = ref_kv.hadamard128(q16b.float().cpu().numpy().reshape(-1, HD)).astype(np.float16).reshape(-1)
check(np.array_equal(cache[pos0 * RB:(pos0 + n) * RB].cpu().numpy().reshape(n, RB), want) and int(kverr[0]) == 0
      and np.array_equal(q16a.cpu().numpy().view(np.uint16), qwant.view(np.uint16)),
      f'q3p_qk_rope_q8: {n} rows at position {pos0} byte for byte encode_q8_kv(hadamard128(K), V) of q3p_qk_rope\'s halves (edge blocks '
      f'included); its q16 bit for bit fp16(hadamard128(q3p_qk_rope\'s q16))')
# the attention: Q8 rows against an F16 cache of fp16(decoded rows)
rows = cache[:(pos0 + n) * RB].cpu().numpy().reshape(pos0 + n, RB)
dk, dv = ref_kv.decode_q8_kv(rows, LAY)
f16c[:(pos0 + n) * 2 * M] = torch.from_numpy(np.concatenate([dk.reshape(pos0 + n, M), dv.reshape(pos0 + n, M)], 1).astype(np.float16).reshape(-1)).to(dev)
qa = torch.from_numpy((rng.standard_normal(n * NH * HD) * 0.5).astype(np.float16)).to(dev)
oa = torch.zeros(n * NH * HD, dtype=torch.float16, device=dev); ob = torch.zeros_like(oa)
oq8.attn(qa, cache, pos0, n, NKV, scale, oa); ops.attn(qa, f16c, pos0, n, NKV, scale, ob); torch.cuda.synchronize()
check(torch.equal(oa.view(torch.int16), ob.view(torch.int16)),
      f'q3mp_attn_q8 over Q8 rows = q3mp_attn over fp16(decoded rows), bit for bit ({n} queries at position {pos0})')
# a refused block in the prefill
xs2 = xs.copy(); xs2[3, NH * HD + M + 40] = np.nan
cache[(pos0 + 3) * RB:(pos0 + 4) * RB] = 0xA5
oq8.qk_rope(torch.from_numpy(xs2).to(dev), NH * HD + 2 * M, NH, NKV, par, pos0, q16a, cache, n, err=kverr); torch.cuda.synchronize()
r3 = cache[(pos0 + 3) * RB:(pos0 + 4) * RB].cpu().numpy()
check(int(kverr[0]) == 2 and bool((r3[vb + 34:vb + 68] == 0xA5).all()) and not bool((r3[:vb] == 0xA5).all()),
      'q3p_qk_rope_q8: a NaN in v sets its error word (2), the refused block untouched, the row\'s K blocks written')
print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
