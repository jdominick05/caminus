"""The attention block of a Flash-Next attention layer in the K3 engine vs llama.cpp's own tensors (tools/gg_golden
dumps), for REAP-308 and the GSQ-RCO Coder, as the tape runs it (caminus.engine.flashnext): from the layer's hc_mixed, with the
cache filled from llama.cpp's own K / V of the earlier positions, the engine computes the Q (+ gate), K, V projections
(REAP Q6_K / Q8_0; Coder Q6_K / IQ4_XS / ...), q / k RMSNorm + RoPE (64 of 256 dims), and, for a q8_0 cache, llama.cpp's
Walsh-Hadamard rotation (q / k over the head, v in 64s, the output rotated back), appends K / V, decode attention (24 q
heads on 2 kv heads), the sigmoid gate and attn_output.
Gates (tests/engine/golden_util.py): the projections fed llama.cpp's input (Qcur_full, raw K / V from hc_mixed; attn_output
also from llama.cpp's attn_gated) bitwise for the one-warp formats, else within 1e-6; Q / K after norm + RoPE within
1e-6; the engine's attention within 1e-6 of float64 over its own cache. llama.cpp's flash attention holds V in f16, so
its attention output (and the chained attn_output) is only compared within 2e-2 (q8_0 cache: steps after the first
reach ~3e-3 - 6e-3, inferred to be its quantized K . Q path; position 0, V only, is ~1e-4) / 2e-3 (f16).
  HIP_VISIBLE_DEVICES=0 python tests/engine/test_golden_attn.py GOLDEN_DIR [layer=3] [model.gguf] [kv=q8_0|f16]
The KV cache type must be the one the dumps were recorded with (tools/gg_golden: q8_0 by default)."""
import ctypes, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # golden_util (python -I drops the script's directory)
import numpy as np
import torch
from golden_util import Golden, Model, Report, REAP, mv, cu, q8b, buf

GD = sys.argv[1]; L = int(sys.argv[2]) if len(sys.argv) > 2 else 3
m = Model(sys.argv[3] if len(sys.argv) > 3 else REAP); golden = Golden(GD)
Q8 = (sys.argv[4] if len(sys.argv) > 4 else 'q8_0') == 'q8_0'
libm = ctypes.CDLL('libm.so.6'); libm.powf.restype = ctypes.c_float; libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]
D, NH, NKV, HD, NROT, EPS = 2560, 24, 2, 256, 64, 9.999999974752427e-07
theta_scale = libm.powf(10000000.0, -2.0 / NROT)
W = {n: m.raw(f'blk.{L}.{n}.weight') for n in ('attn_q', 'attn_k', 'attn_v', 'attn_output')}
qn, kn = m.f32(f'blk.{L}.attn_q_norm.weight'), m.f32(f'blk.{L}.attn_k_norm.weight')
PB = (2 * NKV * HD // 32 * 34) if Q8 else 4 * NKV * HD            # bytes per cached position
ATOL = 2e-2 if Q8 else 2e-3     # llama.cpp's q8_0 attention: ~3e-3 - 6e-3 once K . Q matters (inferred: its quantized K . Q path; V in f16)
rep = Report()
print(f'module {m.module}, layer {L}, kv {"q8_0 (rotated)" if Q8 else "f16"}: ' + ', '.join(f'{n} {w["fmt"]}' for n, w in W.items()))
for step in golden.steps:
    x = cu(golden(step, f'hc_mixed-{L}', 0))
    cache = torch.zeros(64 * PB, dtype=torch.uint8, device='cuda')
    for p in range(step):                       # earlier positions: llama.cpp's own roped K and V, stored as the tape does
        pe = m.engine(); k_, v_ = cu(golden(p, f'Kcur-{L}', 1)).reshape(-1), cu(golden(p, f'Vcur-{L}', 1)).reshape(-1)
        a = -1
        if Q8: a = pe.hadamard(k_, HD); a = pe.also(pe.hadamard(v_, 64, dep=-1), a)
        pe.kv_append(k_, v_, cache, torch.tensor([p], dtype=torch.int32, device='cuda'), q8=Q8, dep=a)
        pe.build().run()
    pos = torch.tensor([step], dtype=torch.int32, device='cuda')
    xq, qf, kr, vr, qo, ko = q8b(D), buf(NH * 2 * HD), buf(NKV * HD), buf(NKV * HD), buf(NH * HD), buf(NKV * HD)
    att, gat, gq, out = buf(NH * HD), buf(NH * HD), q8b(NH * HD), buf(D)
    gat_g, gq_g, out_g = cu(golden(step, f'attn_gated-{L}')), q8b(NH * HD), buf(D)
    # 1: projections, q / k norm + RoPE (compared before the rotation); attn_output from llama.cpp's attn_gated
    e = m.engine()
    a = e.gg_q8_1(x, xq)
    a = mv(e, W['attn_q'], qf, a, xq=xq); a = mv(e, W['attn_k'], kr, a, xq=xq); a = mv(e, W['attn_v'], vr, a, xq=xq)
    a = e.qk_nr(qf, 2 * HD, NH, HD, qn, pos, qo, NROT, theta_scale, EPS, dep=a)
    a = e.qk_nr(kr, HD, NKV, HD, kn, pos, ko, NROT, theta_scale, EPS, dep=a)
    a = e.gg_q8_1(gat_g, gq_g, dep=a); mv(e, W['attn_output'], out_g, a, xq=gq_g)
    e.build().run()
    rep.check('Qfull', qf, golden(step, f'Qcur_full-{L}'), fmt=W['attn_q']['fmt'])
    rep.check('Kraw', kr, golden(step, f'Kcur-{L}', 0), fmt=W['attn_k']['fmt']); rep.check('Vraw', vr, golden(step, f'Vcur-{L}', 0), fmt=W['attn_v']['fmt'])
    rep.check('Q', qo, golden(step, f'Qcur-{L}')); rep.check('K', ko, golden(step, f'Kcur-{L}', 1)); rep.check('V', vr, golden(step, f'Vcur-{L}', 1))
    rep.check('out.iso', out_g, golden(step, f'attn_output-{L}'), fmt=W['attn_output']['fmt'])
    # 2: rotation, append, attention (compared with float64 over the engine's own cache, in the rotated basis)
    e = m.engine(); a = -1
    if Q8: a = e.hadamard(qo, HD); a = e.also(e.hadamard(ko, HD), a); a = e.also(e.hadamard(vr, 64), a)
    a = e.kv_append(ko, vr, cache, pos, q8=Q8, dep=a)
    e.attn_dec(qo, cache, pos, att, NH, NKV, HD, 1.0 / 16.0, q8=Q8, dep=a)
    e.build().run()
    cb = cache.cpu().numpy().reshape(64, PB)
    if Q8:      # dequantize the q8_0 blocks
        blk = cb.reshape(64, -1, 34); cq = (blk[:, :, :2].copy().view(np.float16).astype(np.float64) * blk[:, :, 2:].view(np.int8).astype(np.float64)).reshape(64, 2, NKV, HD)
    else:
        cq = cb.view(np.float16).astype(np.float64).reshape(64, 2, NKV, HD)
    Q = qo.cpu().numpy().astype(np.float64).reshape(NH, HD); ref = np.zeros((NH, HD))
    for h in range(NH):
        g = h // (NH // NKV); sc = cq[:step + 1, 0, g] @ Q[h] / 16.0; pr = np.exp(sc - sc.max()); pr /= pr.sum()
        ref[h] = pr @ cq[:step + 1, 1, g]
    rep.check('attn_vs_f64', att, ref)
    # 3: rotate back, gate, attn_output (chained: carries llama.cpp's f16-V difference)
    e = m.engine(); a = e.hadamard(att, 64, rows_per_task=4) if Q8 else -1
    a = e.gate_mul(att, qf, 2 * HD, HD, gat, HD, dep=a, q8=gq); mv(e, W['attn_output'], out, a, xq=gq)
    e.build().run()
    rep.check('pregate', att, golden(step, f'attn_pregate-{L}'), tol=ATOL)
    rep.check('gated', gat, golden(step, f'attn_gated-{L}'), tol=ATOL); rep.check('out', out, golden(step, f'attn_output-{L}'), tol=ATOL)
    rep.flush(f'step {step}:')
print(f'(pregate / gated / out: chained through llama.cpp\'s f16-V flash attention, gate {ATOL:.0e})')
sys.exit(rep.summary())
