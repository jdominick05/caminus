"""Expert-indexed MoE ops in the K3 engine (OP_GG_MOE_GU / OP_GG_MOE_DOWN, build/k3_gguf.hsaco) on the real REAP-308-UI
layer-0 expert stacks (IQ3_XXS gate / up, IQ4_NL down, 308 experts, top-10, ffn 640, d 2560), with the expert ids and
router weights on the device:
  gate/up: h = silu(gate . x) * (up . x) equals a host-side composition of the standalone matvecs (g and u bitwise, h
           within 1 ulp: torch's exp vs HIP's expf);
  down:    out = sum_j w_j (down . h_j) is bitwise equal to the standalone matvecs on the engine's own h, summed in order.
Then the speed of a chained tape of 30 pseudo-layers [q8_1 -> MOE_GU -> q8_1 -> MOE_DOWN] with distinct experts.
  HIP_VISIBLE_DEVICES=0 tools/guard.sh LOG 600 python -I tests/engine/test_k3_gguf_moe.py [model.gguf]"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
import torch
from caminus.engine import gguf as G, k3

from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
PATH = sys.argv[1] if len(sys.argv) > 1 else devpaths.reap()
r = G.GGUFReader(PATH, 'r'); T = {t.name: t for t in r.tensors}
torch.manual_seed(0); ok = True
D, FFN, K = 2560, 640, 10


def stack(name):
    t = T[name]; a = np.asarray(t.data)
    return torch.from_numpy(np.ascontiguousarray(a)).cuda().reshape(-1), a.shape[1] * a.shape[2], t.tensor_type.name   # flat; bytes per slab


gate, gslab, gfmt = stack('blk.0.ffn_gate_exps.weight'); up, _, _ = stack('blk.0.ffn_up_exps.weight')
down, dslab, dfmt = stack('blk.0.ffn_down_exps.weight')
ids = torch.tensor([3, 17, 42, 99, 150, 151, 200, 255, 300, 307], dtype=torch.int32, device='cuda')
wts = torch.softmax(torch.randn(K, device='cuda'), 0).contiguous()
x = torch.randn(D, device='cuda') * 0.5
xq = torch.empty(D // 32 * 36, dtype=torch.uint8, device='cuda'); h = torch.empty(K * FFN, device='cuda')
hq = torch.empty(K * FFN // 32 * 36, dtype=torch.uint8, device='cuda'); out = torch.empty(D, device='cuda')
e = k3.Engine(module='k3_gguf')
a = e.gg_q8_1(x, xq); b = e.gg_moe_gu(gate, up, gfmt, gslab, FFN, D, ids, xq, h, dep=a)
c = e.gg_q8_1(h, hq, dep=b); e.gg_moe_down(down, dfmt, dslab, FFN, D, ids, wts, hq, out, dep=c)
e.build().run()
# reference: standalone matvecs per expert
xq_ref = G.quantize_q8_1(x); ok &= torch.equal(xq, xq_ref)
g = torch.stack([G.matvec(gate[gslab * int(i):gslab * (int(i) + 1)], G.QT[gfmt], FFN, D, xq_ref) for i in ids])
u = torch.stack([G.matvec(up[gslab * int(i):gslab * (int(i) + 1)], G.QT[gfmt], FFN, D, xq_ref) for i in ids])
h_ref = (g / (1.0 + torch.exp(-g)) * u).reshape(-1)
ulp = (h.view(torch.int32) - h_ref.view(torch.int32)).abs().max().item()
print(f'gate/up: h vs host composition: max {ulp} ulp')
ok &= ulp <= 1
hq_ref = G.quantize_q8_1(h); ok &= torch.equal(hq, hq_ref)
v = torch.stack([G.matvec(down[dslab * int(i):dslab * (int(i) + 1)], G.QT[dfmt], D, FFN, hq_ref[j * FFN // 32 * 36:(j + 1) * FFN // 32 * 36])
                 for j, i in enumerate(ids)]).cpu().numpy()
w = wts.cpu().numpy(); acc = (v[0] * w[0]).astype(np.float32)
for j in range(1, K): acc = (acc + (v[j] * w[j]).astype(np.float32)).astype(np.float32)
o = out.cpu().numpy()
accf = (v[0].astype(np.float64) * w[0]).astype(np.float32)          # FMA-contracted variant, for diagnosis
for j in range(1, K): accf = (v[j].astype(np.float64) * w[j] + accf.astype(np.float64)).astype(np.float32)
print(f'  diag: hq bytes equal {torch.equal(hq, hq_ref)}; out == mul-then-add {np.array_equal(o, acc)}, == fma chain {np.array_equal(o, accf)}; differing rows {int((o != acc).sum())}/{D}')
dok = np.array_equal(o, acc); ok &= dok
print(f'down: out vs standalone matvecs summed in order: {"bitwise" if dok else "DIFF, max " + str(np.abs(out.cpu().numpy() - acc).max())}')

# speed
for rgu, rdn in ((16, 8), (8, 8)):
    L = 30; e = k3.Engine(module='k3_gguf'); prev = -1; byts = 0
    for l in range(L):
        il = torch.tensor([(l * 10 + j) % 308 for j in range(K)], dtype=torch.int32, device='cuda')
        q1 = e.gg_q8_1(x, torch.empty_like(xq), dep=prev)
        hl = torch.empty(K * FFN, device='cuda'); hql = torch.empty_like(hq)
        gu = e.gg_moe_gu(gate, up, gfmt, gslab, FFN, D, il, e.keep[-1], hl, rows_per_task=rgu, dep=q1)
        q2 = e.gg_q8_1(hl, hql, dep=gu)
        prev = e.gg_moe_down(down, dfmt, dslab, FFN, D, il, wts, hql, torch.empty(D, device='cuda'), rows_per_task=rdn, dep=q2)
        byts += K * (2 * gslab + dslab)
    e.build(); e.run(); torch.cuda.synchronize()
    t0 = time.perf_counter(); n = 10
    for _ in range(n): e.run(check=False)
    torch.cuda.synchronize(); us = (time.perf_counter() - t0) / n * 1e6; e.run()
    print(f'{torch.cuda.get_device_name(0)} rows/task gu {rgu} down {rdn}: {us / L:.1f} us per MoE layer ({byts / L / 1e6:.1f} MB), {byts / us / 1e3:.0f} GB/s')
print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
