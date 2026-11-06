"""The MTP module pair's device paths (kernels/k3_gguf_gsq_mtp*.hip, python/caminus/mtp.py), on one card:
  (a) bf16_mv(ntok=R): R rows through one segment (the MTP eh_proj: 4 hc streams per token) == R one-row segments, bitwise,
      in the one-token module (T = 1 engine) and the multi-token one (T = 4 engine, team kept by build()); the trunk's
      k3_gguf_gsq gives the same rows bitwise for the plain segments
  (b) bf16_mv(swiglu_mul=m) (Seg.work 2: LDS stage silu(x) * m): with m = 1 bitwise the silu_scale = 1 path (silu(fma(1, x, 0))
      = silu(x)), and against a float64 reference of W . (x / (1 + exp(-x)) * m) within 1e-5 relative
  (c) the trunk modules refuse swiglu_mul
  tools/guard.sh LOG 900 python -I tests/test_mtp_ops.py [--card 1]   (run with HIP_VISIBLE_DEVICES=1 for the XT alone)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import torch
from caminus import k3

a = sys.argv[1:]
card = int(a[a.index('--card') + 1]) if '--card' in a else 0
torch.cuda.set_device(card); torch.manual_seed(0)
dev = 'cuda'
fails = []


def bf16(rows, cols, scale=0.02):
    w = (torch.randn(rows, cols, device=dev) * scale).to(torch.bfloat16)
    return w.view(torch.int16).contiguous(), w.float()


def eng(module, T):
    return k3.Engine(module=module, grid=k3.resident_grid(module), T=T, limit_us=200_000)


def chk(name, ok):
    print(f'{name}: {"ok" if ok else "FAIL"}', flush=True)
    if not ok: fails.append(name)


# (a) eh_proj shapes: W [2560, 5120], R = 4 (T = 1) and 16 (T = 4)
W, Wf = bf16(2560, 5120)
for module, T in (('k3_gguf_gsq_mtp', 1), ('k3_gguf_gsq_mtp', 4)):
    R = 4 * T
    x = torch.randn(R, 5120, device=dev)
    e = eng(module, T)
    o_all = torch.zeros(R, 2560, device=dev)
    e.bf16_mv(W, 2560, 5120, x, o_all, rows_per_task=8, ntok=R)
    o_rows = torch.zeros(R, 2560, device=dev)
    for r0 in range(0, R, T):   # T rows per plain segment
        e.bf16_mv(W, 2560, 5120, x[r0:r0 + T], o_rows[r0:r0 + T], rows_per_task=8)
    e.build(); e.run()
    chk(f'(a) {e.module} T={T}: ntok={R} == {R // T} plain T={T} segments', torch.equal(o_all, o_rows))
    ref = x.double() @ Wf.double().t()
    chk(f'(a) {e.module} T={T}: vs float64 (rel {float((o_all - ref).abs().max() / ref.abs().max()):.2e})',
        float((o_all - ref).abs().max() / ref.abs().max()) < 1e-5)
    if T == 1:
        x1 = x; o1 = o_all
    else:
        o_t1 = torch.zeros(R, 2560, device=dev); e1 = eng('k3_gguf_gsq_mtp', 1); e1.bf16_mv(W, 2560, 5120, x, o_t1, ntok=R); e1.build(); e1.run()
        chk('(a) T=4 module rows == T=1 module rows', torch.equal(o_all, o_t1))
        g = eng('k3_gguf_gsq', 1); o_g = torch.zeros(4, 2560, device=dev)
        for r in range(4): g.bf16_mv(W, 2560, 5120, x[r:r + 1], o_g[r:r + 1])
        g.build(); g.run()
        chk('(a) trunk k3_gguf_gsq one-row segments == MTP module rows', torch.equal(o_g, o_t1[:4]))

# (b) shared-expert down shapes: W [2560, 320] (K 320 per card at a 10 / 10 split)
W2, W2f = bf16(2560, 320, 0.05)
for module, T in (('k3_gguf_gsq_mtp', 1), ('k3_gguf_gsq_mtp', 4)):
    x = torch.randn(T, 320, device=dev) * 2; m = torch.randn(T, 320, device=dev)
    one = torch.ones(T, 320, device=dev)
    e = eng(module, T)
    o_sw1, o_si, o_sw = (torch.zeros(T, 2560, device=dev) for _ in range(3))
    e.bf16_mv(W2, 2560, 320, x, o_sw1, swiglu_mul=one)
    e.bf16_mv(W2, 2560, 320, x, o_si, silu_scale=1.0)
    e.bf16_mv(W2, 2560, 320, x, o_sw, rows_per_task=3, swiglu_mul=m)
    e.build(); e.run()
    chk(f'(b) {e.module} T={T}: swiglu(m = 1) == silu_scale 1 bitwise', torch.equal(o_sw1, o_si))
    xd = x.double(); ref = (xd / (1 + torch.exp(-xd)) * m.double()) @ W2f.double().t()
    rel = float((o_sw - ref).abs().max() / ref.abs().max())
    chk(f'(b) {e.module} T={T}: swiglu vs float64 (rel {rel:.2e})', rel < 1e-5)

# (c)
for module in ('k3_gguf_gsq', 'k3_gguf_gsq_mt'):
    e = k3.Engine(module=module, T=1 if module == 'k3_gguf_gsq' else 4)
    try:
        e.bf16_mv(W2, 2560, 320, torch.zeros(e.T, 320, device=dev), torch.zeros(e.T, 2560, device=dev), swiglu_mul=torch.zeros(e.T, 320, device=dev))
        chk(f'(c) {e.module} refuses swiglu_mul', False)
    except AssertionError:
        chk(f'(c) {e.module} refuses swiglu_mul', True)

print('RESULT', 'PASS' if not fails else f'FAIL {fails}')
sys.exit(1 if fails else 0)
