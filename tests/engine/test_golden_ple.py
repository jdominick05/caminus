"""The PLE n-gram injection (layer 1) in the K3 engine vs llama.cpp's own tensors (tools/gg_golden dumps), for REAP-308
and the GSQ-RCO Coder (whose table is in shard 2 of the split GGUF, read with O_DIRECT): the host hashes the n-gram
context and gathers the 16 table rows (caminus.engine.flashnext.PLE), the engine computes key / value as the tape does (REAP
Q8_0 with per-workgroup quantization; Coder BF16, OP_BF16_MV), the grouped norms, the per-stream gate, the gated value,
the dilated conv over the 9-position history and the residual update; compared with ple_gate, ple_gated_value,
ple_conv_out, the layer's first hc_norm (which reads the updated residual) and, after a commit pass (lazy commit, as
the tapes run it), the next step's PLE conv history. Gate: within 1e-6 relative (key / value are not dumped).
  HIP_VISIBLE_DEVICES=0 python tests/engine/test_golden_ple.py GOLDEN_DIR [model.gguf]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # golden_util (python -I drops the script's directory)
import numpy as np
import torch
from golden_util import Golden, Model, Report, REAP, mv, cu, buf
from caminus.engine.flashnext import PLE, split_tensors

GD = sys.argv[1]; L = 1
m = Model(sys.argv[2] if len(sys.argv) > 2 else REAP); golden = Golden(GD)
D, HC, EPS = 2560, 4, 9.999999974752427e-07
ple = PLE(m.rd, split_tensors(m.path, m.rd)); tokens = golden.tokens()
kw, vw = m.raw(f'blk.{L}.ple_key.weight'), m.raw(f'blk.{L}.ple_value.weight')
nk, nq, nc = (m.f32(f'blk.{L}.ple_norm_{n}.weight', (HC, D)) for n in ('key', 'query', 'conv'))
convw = torch.from_numpy(np.array(m.T[f'blk.{L}.ple_conv1d.weight'].data)).cuda().reshape(-1)   # f16 [C, 4]
hcn = m.f32(f'blk.{L}.hc_attn_norm.weight', (HC, D))
steps = golden.steps; rep = Report()
print(f'module {m.module}: ple key / value {kw["fmt"]} / {vw["fmt"]}, table {ple.qtype.name}' + (' (shard 2, O_DIRECT)' if m.T.get('per_layer_token_embd.weight') is None else ''))
for step in steps:
    emb = cu(ple.embedding(tokens, step))
    hidden = cu(golden(step, f'l_last-{L - 1}'))                       # [hc, D]: the residual entering layer 1
    hist, pxp = cu(golden(step, f'conv_state_at-{L}', 0)), buf(HC * D)  # [C, 9]
    key, val, keyn, qn, gated, gate, nrm, co, xn = buf(HC, D), buf(D), buf(HC, D), buf(HC, D), buf(HC, D), buf(HC), buf(HC, D), buf(HC * D), buf(HC, D)
    e = m.engine()
    a = mv(e, kw, key.reshape(-1), xf=emb); a = mv(e, vw, val, a, xf=emb)
    a = e.rms_mul(key, nk, keyn, EPS, dep=a); a = e.rms_mul(hidden, nq, qn, EPS, dep=a)
    a = e.ple_gate(keyn, qn, val, gated, gate, dep=a)
    a = e.rms_mul(gated, nc, nrm, EPS, dep=a)
    a = e.ple_conv(nrm.reshape(-1), convw, hist, co, pxp, dep=a)
    a = e.add_sum(gated.reshape(-1), co, hidden.reshape(-1), dep=a)
    e.rms_mul(hidden, hcn, xn, EPS, dep=a)
    e.build().run()
    rep.check('gate', gate, golden(step, f'ple_gate-{L}')); rep.check('gated', gated, golden(step, f'ple_gated_value-{L}'))
    rep.check('conv_out', co, golden(step, f'ple_conv_out-{L}')); rep.check('hc_norm', xn, golden(step, f'hc_norm-{L}', 0))
    if step + 1 in steps:      # the commit pass (params[1] = 1) shifts this token into the history
        e2 = m.engine(); e2.params[1] = 1
        e2.ple_conv(nrm.reshape(-1), convw, hist, buf(HC * D), pxp); e2.build().run()
        rep.check('history', hist, golden(step + 1, f'conv_state_at-{L}', 0))
    rep.flush(f'step {step} (rows {ple.rows(tokens, step)[:3]}...):')
sys.exit(rep.summary())
