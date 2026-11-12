"""The dense qwen35 own modules' decode attention (the split pair of kernels/k3_attn_split.h, built into k3_q35_swift /
k3_q35_swift_ver / k3_q35_tcap by kernels/k3_q35_ops.h under K3_Q35_ATTN_SPLIT) on one attention layer's shapes (24 query
heads over 4 kv heads, head dim 256, f16 cache), a tape of just the pair on random q and cache rows:
  - bitwise the GGUF module's pair on the same inputs (k3_gguf_q35d for T = 1, k3_gguf_q35d_mt for T = 4: the same source
    and settings), at every list length tried
  - a T = 4 pass's row r bitwise the one-row pass at position pos + r (the split layout depends only on the list length)
  - against float64 attention: max |difference| <= 1e-5 x max |V|
  - the q8_0 cache refused on the own modules (their device code reads f16 only)
  tools/guard.sh LOG 600 python -I tests/engine/test_q35_split_attn.py [--lens 1,2,31,32,33,1000,4096,16389]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>). Prints PASS / FAIL and RESULT."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np, torch
from caminus.device import capacity
from caminus.engine import k3

ap = argparse.ArgumentParser(); ap.add_argument('--lens', default='1,2,31,32,33,1000,4096,16389')
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
cache = (torch.randn(ROWS * 2 * NKV * D, generator=g, device='cuda') * 0.5).half()
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
    if n >= 4:
        q4 = torch.randn(4 * H * D, generator=g, device='cuda') * 0.1
        p0 = n - 4
        a4 = run('k3_q35_swift_ver', 4, p0, q4); b4 = run('k3_gguf_q35d_mt', 4, p0, q4)
        bit &= torch.equal(a4, b4)
        for r in range(4):
            rows_ok &= torch.equal(a4[r], run('k3_q35_swift', 1, p0 + r, q4[r * H * D:(r + 1) * H * D].contiguous())[0])
    print(f'  list length {n}: max |own - float64| so far {err:.2e}', flush=True)
check(bit, f'the own modules\' pair is bitwise k3_gguf_q35d\'s (T 1) and k3_gguf_q35d_mt\'s (T 4) at list lengths {LENS}')
check(rows_ok, 'a T = 4 pass\'s rows are bitwise the one-row passes at their positions')
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
print('RESULT', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
