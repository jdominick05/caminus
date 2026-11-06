"""K3 M1 step 1: the decode matvec (kernels/k3_mv.hip) is bit-identical to the training GEMM gemm_tn_iu4g for the row
it shares, on every S1 linear shape: random int4 codes and g128 scales, the token as row r of an M = 64 batch (the
other rows random), bf16 and fp32 outputs. Includes extreme codes (-8 x -8, 7 x -8) and real activation codes from
actq on a real-scale row.
  python tests/test_k3_matvec.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import torch
from caminus import gemm, k3
from caminus.actq import codes as actq_codes

torch.manual_seed(0)
ok = True
SHAPES = [(7680, 896, 'GDN block q|k|v|g|gu'), (5248, 896, 'attention q|kv|gu'), (896, 896, 'o_proj'),
          (896, 2048, 'mlp.down'), (1536, 896, 'slice gu (768)'), (896, 768, 'slice down (768)'), (896, 384, 'slice down (384)')]
for N, K, what in SHAPES:
    qw = torch.randint(-8, 8, (N, K), device='cuda'); qw[0] = -8; qw[1] = 7
    G = K // 128; gs = (torch.rand(N, G, device='cuda') + 0.05) / 50
    xa = torch.randint(-8, 8, (64, K), device='cuda'); xa[5] = -8                        # row 5 all -8: extremes
    rows = {'random row': 3, 'extreme row': 5}
    h = torch.randn(1, K, device='cuda') * torch.rand(1, K, device='cuda') * 4          # a real actq row
    cx, sx = actq_codes(h, 4, 1.0); xa[7] = k3.unpack_codes(cx)[0].to(xa.dtype); rows['actq row'] = 7
    rs = torch.rand(64, device='cuda') + 0.5; rs[7] = sx[0]
    ref32 = gemm.gemm_tn_iu4g(gemm.pack_int4(xa), gemm.pack_int4(qw), gs.contiguous(), rs, torch.float32, tile=(64, 128))
    ref16 = gemm.gemm_tn_iu4g(gemm.pack_int4(xa), gemm.pack_int4(qw), gs.contiguous(), rs, torch.bfloat16, tile=(64, 128))
    wmv = k3.repack_mv(qw)
    res = []
    for name, r in rows.items():
        a = xa[r].to(torch.int8)
        y32 = k3.mv_g128(wmv, gs, a, float(rs[r]), torch.float32); y16 = k3.mv_g128(wmv, gs, a, float(rs[r]), torch.bfloat16)
        e = torch.equal(y32, ref32[r]) and torch.equal(y16, ref16[r]); ok &= e
        res.append(f'{name} {"bitwise" if e else "DIFF (%d fp32 differ)" % (y32 != ref32[r]).sum().item()}')
    print(f'{what:24s} {N}x{K}: ' + ', '.join(res))
print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
