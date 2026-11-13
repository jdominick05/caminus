"""The dense qwen35 own modules' decode attention (the split pair of kernels/k3_attn_split.h, built into k3_q35_swift /
k3_q35_swift_ver / k3_q35_tcap by kernels/k3_q35_ops.h under K3_Q35_ATTN_SPLIT) on one attention layer's shapes (24 query
heads over 4 kv heads, head dim 256, f16 cache), a tape of just the pair on random q and cache rows:
  - bitwise the GGUF module's pair on the same inputs (k3_gguf_q35d for T = 1, k3_gguf_q35d_mt for T = 2, 3, 4: the same
    source, its PART unpacked), at every list length tried: the verify module k3_q35_swift_ver runs the packed PART (one
    task per (split, kv head) for every pass row: kernels/k3_attn_split.h op_attn_part_packed) and must give its bits;
    lengths 34 / 35 / 4098 put a split under only some of a pass's rows
  - a T-row pass's row r bitwise the one-row pass at position pos + r (the split layout depends only on the list length)
  - the packed PART refuses T above k3.ATTN_PACK_T and a QSA selection (identity lists only)
  - against float64 attention: max |difference| <= 1e-5 x max |V|
  - the q8_0 cache refused on the own modules (their device code reads f16 only)
  tools/guard.sh LOG 600 python -I tests/engine/test_q35_split_attn.py [--lens 1,2,31,32,33,34,35,1000,4096,4098,16389]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>). Prints PASS / FAIL and RESULT."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np, torch
from caminus.device import capacity
from caminus.engine import k3

ap = argparse.ArgumentParser(); ap.add_argument('--lens', default='1,2,31,32,33,34,35,1000,4096,4098,16389')
A = ap.parse_args()
H, NKV, D, G = 24, 4, 256, 6
LENS = [int(v) for v in A.lens.split(',')]
ROWS = max(LENS) + 4
SCALE = 1.0 / 16.0
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


need = (ROWS * k3.kv_pos_bytes(NKV * D, False) + 4 * H * ((ROWS + 31) // 32) * k3.ATTN_RS * 4 * 2 + 64 * H * D * 4) / 2 ** 20 + 64
capacity.check([0], [need], f'test_q35_split_attn ({need:.0f} MiB)')
print(torch.cuda.get_device_name(0), f'| {H} q heads / {NKV} kv heads x {D}, cache rows {ROWS}', flush=True)
g = torch.Generator(device='cuda').manual_seed(7)
cache = torch.empty(ROWS * 2 * NKV * D, dtype=torch.float16, device='cuda').normal_(0.0, 0.5, generator=g)   # (in place: no fp32 temporaries)
kv = cache.view(ROWS, 2, NKV, D).cpu().double().numpy()   # (to the host first: no float64 copy on the card)
vmax = float(np.abs(kv[:, 1]).max())


def run(module, T, pos, q):
    """the pair on `module` with T rows from position pos: out fp32 [T][H][D] (a fresh engine and records)"""
    E = k3.Engine(module=module, grid=k3.resident_grid(module), T=T, limit_us=500_000)
    out = torch.zeros(T * H * D, device='cuda'); p = torch.full((1,), pos, dtype=torch.int32, device='cuda')
    E.attn_dec(q, cache, p, out, H, NKV, D, SCALE, max_cells=ROWS)
    E.build(); E.run(check=True); torch.cuda.synchronize()
    return out.view(T, H, D).clone()


def ref(qrow, pos):
    """float64 attention of one row at position pos: [H][D]"""
    qq = qrow.double().cpu().numpy().reshape(H, D); o = np.zeros((H, D))
    for h in range(H):
        gk = h // G
        s = kv[:pos + 1, 0, gk] @ qq[h] * SCALE; s = np.exp(s - s.max()); o[h] = (s / s.sum()) @ kv[:pos + 1, 1, gk]
    return o


bit, rows_ok, err = True, True, 0.0
for n in LENS:
    pos = n - 1
    q1 = torch.randn(H * D, generator=g, device='cuda') * 0.1
    a = run('k3_q35_swift', 1, pos, q1); b = run('k3_gguf_q35d', 1, pos, q1); c = run('k3_q35_tcap', 1, pos, q1)
    bit &= torch.equal(a, b) and torch.equal(a, c)
    err = max(err, float(np.abs(a[0].double().cpu().numpy() - ref(q1, pos)).max()))
    for T in (2, 3, 4):
        if n < T: continue
        qT = torch.randn(T * H * D, generator=g, device='cuda') * 0.1
        p0 = n - T
        aT = run('k3_q35_swift_ver', T, p0, qT); bT = run('k3_gguf_q35d_mt', T, p0, qT)
        bit &= torch.equal(aT, bT)
        for r in range(T):
            rows_ok &= torch.equal(aT[r], run('k3_q35_swift', 1, p0 + r, qT[r * H * D:(r + 1) * H * D].contiguous())[0])
    print(f'  list length {n}: max |own - float64| so far {err:.2e}', flush=True)
check(bit, f'the own modules\' pair (packed PART at T 2 / 3 / 4) is bitwise k3_gguf_q35d\'s (T 1) and k3_gguf_q35d_mt\'s (unpacked) at list lengths {LENS}')
check(rows_ok, 'a T = 2 / 3 / 4 pass\'s rows are bitwise the one-row passes at their positions')
check(err <= 1e-5 * vmax, f'against float64 attention: max |difference| {err:.2e} <= 1e-5 x max |V| ({vmax:.2f})')
E = k3.Engine(module='k3_q35_swift', grid=k3.resident_grid('k3_q35_swift'), T=1, limit_us=500_000)
try:
    q8c = torch.zeros(64 * k3.kv_pos_bytes(NKV * D, True), dtype=torch.uint8, device='cuda')
    E.attn_dec(torch.zeros(H * D, device='cuda'), q8c, torch.zeros(1, dtype=torch.int32, device='cuda'), torch.zeros(H * D, device='cuda'),
               H, NKV, D, SCALE, q8=True, max_cells=64)
    E.build(); refused = False
except ValueError:
    refused = True
check(refused, 'the q8_0 cache is refused on the own modules (f16 only)')
for what, kw in (('T 5', dict(T=5)), ('a QSA selection', dict(T=4, sel=True)), ('a q8_0 cache', dict(T=4, q8=True))):
    try:
        Ex = k3.Engine(module='k3_q35_swift_ver', grid=k3.resident_grid('k3_q35_swift_ver'), T=kw['T'], limit_us=500_000)
        T_ = kw['T']; sel = torch.zeros(T_ * k3.QSA_SELW, dtype=torch.int32, device='cuda') if kw.get('sel') else None
        q8 = bool(kw.get('q8'))
        Ex._attn_split(torch.zeros(T_ * H * D, device='cuda'), q8c if q8 else cache, torch.zeros(1, dtype=torch.int32, device='cuda'), sel,
                       torch.zeros(T_ * H * D, device='cuda'), H, NKV, D, SCALE, q8, -1, 64, None)
        refused = False
    except ValueError as ex:
        refused = 'the packed PART' in str(ex)   # (the packed PART's own refusal, not another ValueError on the way)
        if not refused: print(f'  ({what}: a different ValueError: {ex})', flush=True)
    check(refused, f'the packed PART refuses {what}')
print('RESULT', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
