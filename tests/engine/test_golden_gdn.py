"""The GDN block of a Flash-Next layer in the K3 engine vs llama.cpp's own tensors (tools/gg_golden dumps), for REAP-308
and the GSQ-RCO Coder: from the layer's attention-side hc_mixed and llama.cpp's conv / recurrent states before the step,
the engine computes qkv_mixed and z (REAP Q6_K; Coder Q6_K / Q5_K / Q4_K), alpha / beta (REAP Q8_0; Coder BF16), the
gates, the short conv + silu, the q / k L2 norms, the gated-delta-rule step, the gated RMS norm and ssm_out; a second
pass commits the token (lazy commit, as the tapes run it) and gives the new recurrent state and the shifted conv state.
Gate (tests/engine/golden_util.py): the projections fed llama.cpp's input (qkv, z, alpha, beta from hc_mixed; ssm_out also
from llama.cpp's final_output: out.iso) bitwise for BF16 and the one-warp formats; everything within 1e-6 relative, but
the chained ssm_out (out), which crosses the q8_1 of the engine's final_output: there the q8_1 bytes are gated (equal, or
rounding flips the float difference explains), out is bitwise out.iso when they are equal, and out vs llama.cpp is a
diagnostic (one flipped code moved it to 1.6e-4 on a code text).
  HIP_VISIBLE_DEVICES=0 python tests/engine/test_golden_gdn.py GOLDEN_DIR [layer=0] [model.gguf]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # golden_util (python -I drops the script's directory)
import torch
from golden_util import Golden, Model, Report, REAP, mv, cu, q8b, buf
from caminus.engine import k3

GD = sys.argv[1]; L = int(sys.argv[2]) if len(sys.argv) > 2 else 0
m = Model(sys.argv[3] if len(sys.argv) > 3 else REAP); golden = Golden(GD)
D, H, S, NK, C, EPS = 2560, 48, 128, 16, 10240, 9.999999974752427e-07
W = {n: m.raw(f'blk.{L}.{n}.weight') for n in ('attn_qkv', 'attn_gate', 'ssm_alpha', 'ssm_beta', 'ssm_out')}
conv_w, dt, A, nw = m.f32(f'blk.{L}.ssm_conv1d.weight'), m.f32(f'blk.{L}.ssm_dt.bias'), m.f32(f'blk.{L}.ssm_a'), m.f32(f'blk.{L}.ssm_norm.weight')
steps = golden.steps; rep = Report()
# the GDN conv state: on a PLE layer the first conv_state_at is the PLE history [C, 9], the GDN one [C, 3] comes after
CNTH = [int(r[4]) for r in golden.idx if int(r[0]) == steps[0] and r[2] == f'conv_state_at-{L}'].index(3)
print(f'module {m.module}, layer {L}: ' + ', '.join(f'{n} {w["fmt"]}' for n, w in W.items()))
for step in steps:
    x = cu(golden(step, f'hc_mixed-{L}', 0))
    xq, qkv, z, al, be, gates = q8b(D), buf(C), buf(H * S), buf(H), buf(H), buf(2 * H)
    cst, cxp = cu(golden(step, f'conv_state_at-{L}', CNTH)), buf(C)                # [C, 3]: ne0 = 3 taps per channel
    blk = torch.zeros(k3.gdn_block_size(H, S, NK, 1), device='cuda'); blk[:H * S * S] = cu(golden(step, f'state_predelta-{L}')).reshape(-1)
    conv_out, qk, attn, fin, fq, out = buf(C), buf(2 * NK * S), buf(H * S), buf(H * S), q8b(H * S), buf(D)
    fin_g, fq_g, out_g = cu(golden(step, f'final_output-{L}')), q8b(H * S), buf(D)
    e = m.engine()
    a = e.gg_q8_1(x, xq)
    pm = lambda n, o, dep: mv(e, W[n], o, dep, xq=xq, xf=x)
    a = pm('attn_qkv', qkv, a); a = pm('attn_gate', z, a); a = pm('ssm_alpha', al, a); a = pm('ssm_beta', be, a)
    a = e.gdn_gates(al, be, dt, A, gates, dep=a)
    a = e.gdn_conv(qkv, conv_w, cst, conv_out, cxp, dep=a)
    a = e.gdn_l2(conv_out, qk, 2 * NK, S, EPS, dep=a)
    a = e.gdn_step(qk, conv_out[2 * NK * S:], gates, blk, attn, H, S, NK, C, dep=a)
    a = e.gdn_save(qk, conv_out[2 * NK * S:], gates, blk, H, S, NK, C, dep=a)
    a = e.gdn_norm(attn, nw, z, fin, H, S, EPS, dep=a, q8=fq)
    a = mv(e, W['ssm_out'], out, a, xq=fq)
    a = e.gg_q8_1(fin_g, fq_g, dep=a); mv(e, W['ssm_out'], out_g, a, xq=fq_g)          # isolated: llama.cpp's final_output in
    e.build().run()
    rep.check('qkv', qkv, golden(step, f'linear_attn_qkv_mixed-{L}'), fmt=W['attn_qkv']['fmt'])
    rep.check('z', z, golden(step, f'z-{L}'), fmt=W['attn_gate']['fmt'])
    rep.check('alpha', al, golden(step, f'alpha-{L}'), fmt=W['ssm_alpha']['fmt'])
    rep.check('beta', be, golden(step, f'beta-{L}'), fmt=W['ssm_beta']['fmt'])
    rep.check('gate', gates[:H], golden(step, f'gate-{L}')); rep.check('beta_sig', gates[H:], golden(step, f'beta_sigmoid-{L}'))
    rep.check('conv', conv_out, golden(step, f'conv_output_silu-{L}'))
    rep.check('q_l2', qk[:NK * S], golden(step, f'q_conv_predelta-{L}')); rep.check('k_l2', qk[NK * S:], golden(step, f'k_conv_predelta-{L}'))
    rep.check('attn', attn, golden(step, f'attn_output-{L}')); rep.check('final', fin, golden(step, f'final_output-{L}'))
    rep.check('out.iso', out_g, golden(step, f'linear_attn_out-{L}'), fmt=W['ssm_out']['fmt'])
    eq = rep.q8x('fq', fq, fq_g, fin, golden(step, f'final_output-{L}'))      # the gated-norm producer's q8_1 vs llama.cpp's
    rep.chain('out', eq, out, out_g, golden(step, f'linear_attn_out-{L}'))
    # the commit pass (params[1] = 1): the conv state shifts in the token, the step replays it from the saved record
    e2 = m.engine(); e2.params[1] = 1
    c2 = e2.gdn_conv(qkv, conv_w, cst, buf(C), cxp)
    e2.gdn_step(qk, conv_out[2 * NK * S:], gates, blk, buf(H * S), H, S, NK, C, dep=c2)
    e2.build().run()
    rep.check('state', blk[:H * S * S], golden(step, f'new_state-{L}'))
    if step + 1 in steps: rep.check('conv_state', cst, golden(step + 1, f'conv_state_at-{L}', CNTH))
    rep.flush(f'step {step}:')
sys.exit(rep.summary())
