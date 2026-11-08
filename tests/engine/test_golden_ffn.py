"""The whole FFN of a Flash-Next layer in the K3 engine vs llama.cpp's own tensors (tools/gg_golden dumps), for REAP-308
and the GSQ-RCO Coder, as the tape runs it (caminus.engine.flashnext): from llama.cpp's ffn input (the layer's 2nd hc_mixed),
the engine computes router logits (REAP F32; Coder BF16), softmax top-10 and normalised weights, the 10 routed experts,
the shared expert with its sigmoid gate, and ffn_out. The shared expert runs as the MoE ops with one expert of weight 1
when one format decodes its gate and up (REAP-308), else as llama.cpp's unfused path (gate and up matvecs, the SWIGLU
q8_1 glue, the down matvec: the Coder, whose gate and up formats differ).
Gate (tests/engine/golden_util.py): the same expert ids; router logits, the shared-expert gate logit and the shared gate / up
matvecs (fed llama.cpp's input) bitwise for BF16 / the one-warp formats; the shared down from llama.cpp's ffn_swiglu
bitwise for a one-warp format; everything within 1e-6 relative, but the chained ops after a q8_1 crossing of the
engine's own float (moe_out after the routed h, shexp after the shared swiglu, ffn_out after both): there the q8_1
bytes are gated (equal, or rounding flips the float difference explains), the op is bitwise its isolated twin (fed
llama.cpp's swiglu; the routed twin with the engine's ids / weights) when they are equal, and its error vs llama.cpp is
a diagnostic; the twins carry the 1e-6 / bitwise gates vs llama.cpp (moe_out.iso, shexp.iso, ffn_out.iso).
  HIP_VISIBLE_DEVICES=0 python tests/engine/test_golden_ffn.py GOLDEN_DIR [layer=0] [model.gguf]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # golden_util (python -I drops the script's directory)
import numpy as np
import torch
from golden_util import Golden, Model, Report, REAP, mv, small_mv, cu, q8b, buf

GD = sys.argv[1]; L = int(sys.argv[2]) if len(sys.argv) > 2 else 0
m = Model(sys.argv[3] if len(sys.argv) > 3 else REAP); golden = Golden(GD)
D, FFN, K = 2560, 640, 10
p = f'blk.{L}'
E = int(m.T[f'{p}.ffn_gate_inp.weight'].shape[1])
gate, up, down = (m.raw(f'{p}.ffn_{n}_exps.weight') for n in ('gate', 'up', 'down'))
sg, su, sd = (m.raw(f'{p}.ffn_{n}_shexp.weight') for n in ('gate', 'up', 'down'))
fused = sg['fmt'] == su['fmt'] and ('GU_DENSE', sg['fmt']) in (m.engine().caps or ())   # = caminus.engine.flashnext's condition
rf, sf = m.fmt(f'{p}.ffn_gate_inp.weight'), m.fmt(f'{p}.ffn_gate_inp_shexp.weight')
rep = Report()
print(f'module {m.module}, layer {L}: router {rf} [{E}], shared gate {sf}, experts {gate["fmt"]}/{down["fmt"]}, '
      f'shared {sg["fmt"]}/{su["fmt"]}/{sd["fmt"]} ({"MoE ops, one expert" if fused else "unfused: 2 MV + SWIGLU glue + MV"})')
for step in golden.steps:
    x = cu(golden(step, f'hc_mixed-{L}', 1))
    logits, ids, wts = buf(E), buf(K, dt=torch.int32), buf(K)
    xq, h, hq, moe = q8b(D), buf(K * FFN), q8b(K * FFN), buf(D)
    sh_g, sh_u, sh_h, sh_hq, shexp, sglog, out = buf(FFN), buf(FFN), buf(FFN), q8b(FFN), buf(D), buf(1), buf(D)
    shq_g, shexp_g = q8b(FFN), buf(D)
    sw_g, hq_g, moe_g, out_g = golden(step, f'ffn_moe_swiglu-{L}'), q8b(K * FFN), buf(D), buf(D)
    one_id = torch.zeros(1, dtype=torch.int32, device='cuda'); one_w = torch.ones(1, device='cuda')
    e = m.engine()
    q = e.gg_q8_1(x, xq)
    r = small_mv(m, e, f'{p}.ffn_gate_inp.weight', E, x, logits, rpt=2); r = e.router(logits, K, ids, wts, dep=r)
    g1 = e.also(e.gg_moe_gu(gate['w'], up['w'], gate['fmt'], gate['slab'], FFN, D, ids, xq, h, rows_per_task=8, dep=r, up_fmt=up['fmt']), q)
    q2 = e.gg_q8_1(h, hq, dep=g1); mo = e.gg_moe_down(down['w'], down['fmt'], down['slab'], FFN, D, ids, wts, hq, moe, rows_per_task=8, dep=q2)
    if fused:
        s1 = e.gg_moe_gu(sg['w'], su['w'], sg['fmt'], sg['slab'], FFN, D, one_id, xq, sh_h, rows_per_task=8, dep=q, up_fmt=su['fmt'])
        s2 = e.gg_q8_1(sh_h, sh_hq, dep=s1)
        s3 = e.gg_moe_down(sd['w'], sd['fmt'], sd['slab'], FFN, D, one_id, one_w, sh_hq, shexp, rows_per_task=8, dep=s2)
    else:
        s1 = mv(e, sg, sh_g, q, xq=xq); s2 = mv(e, su, sh_u, q, xq=xq)
        s2 = e.also(e.gg_q8_1(sh_g, sh_hq, dep=s1, mul=sh_u), s2)
        s3 = mv(e, sd, shexp, s2, xq=sh_hq)
    s4 = small_mv(m, e, f'{p}.ffn_gate_inp_shexp.weight', 1, x, sglog, dep=s3)
    e.also(e.shexp_add(moe, shexp, sglog, out, dep=mo), s4)
    # isolated twins of the ops after the q8_1 crossings: llama.cpp's shared swiglu in, and llama.cpp's routed swiglu in
    # with the engine's own ids and weights (llama.cpp's weights differ by an ulp: the twin would not be bitwise), then
    # ffn_out from the two
    a = e.gg_q8_1(cu(golden(step, f'ffn_swiglu-{L}')), shq_g, dep=s4)
    if fused: si = e.gg_moe_down(sd['w'], sd['fmt'], sd['slab'], FFN, D, one_id, one_w, shq_g, shexp_g, rows_per_task=8, dep=a)
    else: si = mv(e, sd, shexp_g, a, xq=shq_g)
    a2 = e.gg_q8_1(cu(sw_g).reshape(-1), hq_g, dep=r)
    mi = e.gg_moe_down(down['w'], down['fmt'], down['slab'], FFN, D, ids, wts, hq_g, moe_g, rows_per_task=8, dep=a2)
    e.also(e.shexp_add(moe_g, shexp_g, sglog, out_g, dep=mi), si)
    e.build().run()
    gi = golden(step, f'ffn_moe_topk-{L}')
    same = ids.cpu().numpy().tolist() == gi.tolist(); rep.ok &= same
    rep.line.append('ids ' + ('same' if same else f'DIFFER {ids.tolist()} vs {gi.tolist()} FAIL'))
    rep.check('logits', logits, golden(step, f'ffn_moe_logits-{L}'), fmt=rf)
    rep.check('weights', wts, golden(step, f'ffn_moe_weights_norm-{L}'))
    rep.check('moe_out.iso', moe_g, golden(step, f'ffn_moe_out-{L}'))
    eq_m = rep.q8x('hq', hq, hq_g, h, sw_g)
    rep.chain('moe_out', eq_m, moe, moe_g, golden(step, f'ffn_moe_out-{L}'))
    sw_sh = golden(step, f'ffn_swiglu-{L}')
    if fused:
        rep.check('sh_swiglu', sh_h, sw_sh); eq_s = rep.q8x('shq', sh_hq, shq_g, sh_h, sw_sh)
    else:
        rep.check('sh_gate', sh_g, golden(step, f'ffn_gate-{L}'), fmt=sg['fmt']); rep.check('sh_up', sh_u, golden(step, f'ffn_up-{L}'), fmt=su['fmt'])
        g_, u_ = sh_g.cpu().numpy(), sh_u.cpu().numpy()   # the SWIGLU glue quantizes silu(g) * u without storing it
        eq_s = rep.q8x('shq', sh_hq, shq_g, (g_ / (np.float32(1) + np.exp(-g_))) * u_, sw_sh, exact=False)
    rep.check('shexp.iso', shexp_g, golden(step, f'ffn_shexp-{L}'), fmt=sd['fmt'])
    rep.chain('shexp', eq_s, shexp, shexp_g, golden(step, f'ffn_shexp-{L}'))
    rep.check('sh_gate_logit', sglog, golden(step, f'shared_expert_gate-{L}'), fmt=sf)
    rep.check('ffn_out.iso', out_g, golden(step, f'ffn_out-{L}'))
    rep.chain('ffn_out', eq_m and eq_s, out, out_g, golden(step, f'ffn_out-{L}'))
    rep.flush(f'step {step}:')
sys.exit(rep.summary())
