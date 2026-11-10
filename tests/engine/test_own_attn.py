"""OP_OWN_ATTN (a query head's whole decode attention as one task, kernels/k3_own_ops.h) against the five segments it
replaces (OP_QK_NR on k and q, OP_KV_APPEND, OP_ATTN_DEC with its loads in batches, OP_OWN_F16) on the same random inputs,
position by position, each form with its own cache: the fp16 outputs and the cache rows are bitwise equal at every
position, across the 512-position chunk boundary of the softmax. Needs no model. Prints PASS / FAIL and RESULT PASS | FAIL.
The positions are enough for the rounding cases to occur: an exact fp16 tie of the fp32 output is about one value in
8000 (4096 values per position), and a merged multiply-add in the fused op shows on the first position.
  tools/guard.sh LOG 300 python -I tests/engine/test_own_attn.py [--positions 660] [--module k3_gguf_q3m_own]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np, torch
from caminus.device import capacity
from caminus.engine import k3
ap = argparse.ArgumentParser()
ap.add_argument('--positions', type=int, default=660); ap.add_argument('--module', default='k3_gguf_q3m_own')
A = ap.parse_args()
nh, nkv, hd = 32, 4, 128                       # qwen3moe's attention: 32 query heads, 4 kv heads, head dim 128
M, CTX = nkv * hd, A.positions
g = torch.Generator().manual_seed(7)
# two forms, each with its own cache (f16 rows, 2 KiB per position) and under 1 MiB of rows, and the HIP context
need_mib = 2 * (CTX * 2 * M * 2 / 2 ** 20 + 1) + 256
capacity.check([0], [need_mib], f'test_own_attn ({CTX} positions: {need_mib:.0f} MiB projected)')
B = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt, device='cuda')
qn = (torch.rand(hd, generator=g) * 2 + 0.5).cuda(); kn = (torch.rand(hd, generator=g) * 2 + 0.5).cuda()
ts = float(np.float32(1e6) ** np.float32(-2.0 / hd)); eps = 1e-6; scale = 1.0 / np.sqrt(hd)
par = torch.cat([qn, kn, torch.tensor([ts, eps], dtype=torch.float32, device='cuda')]).contiguous()
G = k3.resident_grid(A.module)
# the five segments
C = dict(q=B(nh * hd), k=B(M), v=B(M), qo=B(nh * hd), ko=B(M), att=B(nh * hd), att16=B(nh * hd, torch.float16),
         cache=B(CTX * 2 * M, torch.float16), pos=B(1, torch.int32))
EC = k3.Engine(module=A.module, grid=G, T=1, limit_us=200_000)
kd = EC.qk_nr(C['k'], hd, nkv, hd, kn, C['pos'], C['ko'], hd, ts, eps)
apd = EC.kv_append(C['ko'], C['v'], C['cache'], C['pos'], dep=kd)
qd = EC.qk_nr(C['q'], hd, nh, hd, qn, C['pos'], C['qo'], hd, ts, eps)
a_ = EC.also(EC.attn_dec(C['qo'], C['cache'], C['pos'], C['att'], nh, nkv, hd, scale, dep=apd, batched=True), qd)
EC.own_f16(C['att'], C['att16'], dep=a_); EC.build()
# the one segment
F = dict(q=B(nh * hd), kv=B(2 * M), att16=B(nh * hd, torch.float16), cache=B(CTX * 2 * M, torch.float16), pos=B(1, torch.int32))
EF = k3.Engine(module=A.module, grid=G, T=1, limit_us=200_000)
EF.own_attn(F['q'], F['kv'], par, F['cache'], F['pos'], F['att16'], nh, nkv, hd, scale); EF.build()
u16 = lambda t: t.cpu().numpy().view(np.uint16)
bad_out, bad_row, first, ties = 0, 0, None, 0
for p in range(A.positions):
    q = torch.randn(nh * hd, generator=g) * 3; k = torch.randn(M, generator=g) * 3; v = torch.randn(M, generator=g)
    C['q'].copy_(q); C['k'].copy_(k); C['v'].copy_(v); C['pos'].fill_(p)
    F['q'].copy_(q); F['kv'][:M].copy_(k); F['kv'][M:].copy_(v); F['pos'].fill_(p)
    EC.run(); torch.cuda.synchronize(); EF.run(); torch.cuda.synchronize()
    do = int((u16(C['att16']) != u16(F['att16'])).sum())
    dr = int((u16(C['cache'][p * 2 * M:(p + 1) * 2 * M]) != u16(F['cache'][p * 2 * M:(p + 1) * 2 * M])).sum())
    ties += int(((C['att'].cpu().numpy().view(np.uint32) & 0x1fff) == 0x1000).sum())   # (fp32 outputs that are exact fp16 ties)
    if (do or dr) and first is None: first = p
    bad_out += do; bad_row += dr
ok = True
def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)
check(bad_out == 0, f'fp16 outputs bitwise the five segments\' at {A.positions} positions ({bad_out} of {A.positions * nh * hd} values differ'
                    + (f', first at position {first}' if first is not None else '') + f'; {ties} exact fp16 ties among them)')
check(bad_row == 0, f'cache rows (K | V, f16) bitwise the five segments\' ({bad_row} of {A.positions * 2 * M} values differ)')
check(A.positions <= 512 or ties > 0, 'the run reached the softmax\'s second chunk and met fp16 ties')
print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
