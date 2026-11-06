"""K3 MoE ops vs llama.cpp itself (tools/gg_golden dumps), for REAP-308 and the GSQ-RCO Coder: for each dumped decode step,
feed the engine llama.cpp's layer-L MoE input (the ffn hc_mixed), its top-k expert ids and normalized weights, and
compare
  swiglu   the engine's h = silu(gate . x) * (up . x) of the k experts with ffn_moe_swiglu
  down     each expert's down matvec of llama.cpp's own swiglu row (k = 1, weight 1) with ffn_moe_down
  out.iso  the weighted sum over the experts of llama.cpp's swiglu with ffn_moe_out
  out      the chained MoE output (from the engine's h) with ffn_moe_out
Gate (tests/golden_util.py): swiglu bitwise for the one-warp gate / up formats (IQ2_S, IQ3_S, IQ4_XS), down and
out.iso bitwise for a one-warp down format (Q2_0); everything within 1e-6 relative, but the chained out, which crosses
the q8_1 of the engine's h: the q8_1 bytes are gated (equal, or rounding flips the float difference explains), out is
bitwise out.iso when they are equal, and out vs llama.cpp is a diagnostic.
  HIP_VISIBLE_DEVICES=0 python tests/test_golden_moe.py GOLDEN_DIR [layer=0] [model.gguf]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # golden_util (python -I drops the script's directory)
import numpy as np
import torch
from golden_util import Golden, Model, Report, REAP, cu, q8b, buf

GD = sys.argv[1]; L = int(sys.argv[2]) if len(sys.argv) > 2 else 0
m = Model(sys.argv[3] if len(sys.argv) > 3 else REAP); golden = Golden(GD)
gate, up, down = (m.raw(f'blk.{L}.ffn_{n}_exps.weight') for n in ('gate', 'up', 'down'))
D, FFN = 2560, 640
QB = FFN // 32 * 36                                      # q8_1 bytes of one expert's h
rep = Report()
print(f'module {m.module}, layer {L}: gate/up {gate["fmt"]}/{up["fmt"]}, down {down["fmt"]}')
for step in golden.steps:
    x = cu(golden(step, f'hc_mixed-{L}', 1))             # 2nd hc_mixed of the layer = ffn input
    ids = torch.from_numpy(golden(step, f'ffn_moe_topk-{L}').astype(np.int32).copy()).cuda()
    w = cu(golden(step, f'ffn_moe_weights_norm-{L}')); K = ids.numel()
    sw_g = golden(step, f'ffn_moe_swiglu-{L}')           # [k, ffn]
    xq, h, hq, out = q8b(D), buf(K * FFN), q8b(K * FFN), buf(D)
    hq_g, out_g, downs = q8b(K * FFN), buf(D), buf(K, D)
    one = torch.ones(1, device='cuda')
    e = m.engine()
    a = e.gg_q8_1(x, xq); b = e.gg_moe_gu(gate['w'], up['w'], gate['fmt'], gate['slab'], FFN, D, ids, xq, h, rows_per_task=8, dep=a, up_fmt=up['fmt'])
    c = e.gg_q8_1(h, hq, dep=b); e.gg_moe_down(down['w'], down['fmt'], down['slab'], FFN, D, ids, w, hq, out, rows_per_task=8, dep=c)
    a = e.gg_q8_1(cu(sw_g).reshape(-1), hq_g)            # isolated: llama.cpp's swiglu in
    e.gg_moe_down(down['w'], down['fmt'], down['slab'], FFN, D, ids, w, hq_g, out_g, rows_per_task=8, dep=a)
    for j in range(K):
        e.gg_moe_down(down['w'], down['fmt'], down['slab'], FFN, D, ids[j:j + 1], one, hq_g[j * QB:(j + 1) * QB], downs[j], rows_per_task=8, dep=a)
    e.build().run()
    rep.check('swiglu', h, sw_g, fmt=gate['fmt'] if gate['fmt'] == up['fmt'] else None)
    rep.check('down', downs, golden(step, f'ffn_moe_down-{L}'), fmt=down['fmt'])
    rep.check('out.iso', out_g, golden(step, f'ffn_moe_out-{L}'), fmt=down['fmt'])
    eq = rep.q8x('hq', hq, hq_g, h, sw_g)                 # the q8_1 of the engine's swiglu vs of llama.cpp's
    rep.chain('out', eq, out, out_g, golden(step, f'ffn_moe_out-{L}'))
    rep.flush(f'step {step}: experts {ids.tolist()}')
sys.exit(rep.summary())
