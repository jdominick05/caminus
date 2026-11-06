"""Hyper-connection ops of a Flash-Next layer in the K3 engine vs llama.cpp's own tensors (tools/gg_golden dumps), for
REAP-308 (Q8_0 hc) and the GSQ-RCO Coder (BF16 hc: OP_BF16_MV, llama.cpp's mul_mat_vec_f order). Per mixer (attention,
ffn) and step:
  isolated  from llama.cpp's hc_norm: hc_inject (inject matvec) and hc_gate (down -> silu(x / hc) -> up); bitwise for BF16
            (gate G3, tests/golden_util.py), within 1e-6 for Q8_0
  chain     from the mixer's input residual (hc_init on layer 0, l_last of the layer before, hc_combine for the ffn mixer):
            hc_norm, hc_inject, hc_gate, hc_mixed and the post-block combine (hc_combine / l_last), within 1e-6. A Q8_0 hc
            (REAP-308) crosses the q8_1 of the engine's xn: the q8_1 bytes are gated (equal, or rounding flips the float
            difference explains), inject / gate are bitwise their isolated twins when they are equal, and the chain's
            errors vs llama.cpp after a flip are diagnostics (tests/golden_util.py Report.q8x). BF16 (the Coder) has no
            crossing (the matvecs read the fp32 xn).
The attention mixer's chain needs its input residual: not on a PLE layer (tests/test_golden_ple.py covers that hc_norm)
nor when the layer before was not dumped.
  HIP_VISIBLE_DEVICES=0 python tests/test_golden_hc.py GOLDEN_DIR [layer=0] [model.gguf]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # golden_util (python -I drops the script's directory)
from golden_util import Golden, Model, Report, REAP, mv, cu, q8b, buf

GD = sys.argv[1]; L = int(sys.argv[2]) if len(sys.argv) > 2 else 0
m = Model(sys.argv[3] if len(sys.argv) > 3 else REAP); golden = Golden(GD)
f = {x.name: x for x in m.rd.fields.values()}
PLE_LAYERS = {int(v) for v in f['qwen4exp.ple.layers'].contents()}
D, HC, EPS = 2560, 4, 9.999999974752427e-07
names = {r[2] for r in golden.idx}
rep = Report()
print(f'module {m.module}, layer {L}')
for step in golden.steps:
    prev = 'hc_init' if L == 0 else f'l_last-{L - 1}'
    res_in = {'attn': golden(step, prev) if L not in PLE_LAYERS and prev in names else None, 'ffn': golden(step, f'hc_combine-{L}', 0)}
    blkname = f'linear_attn_out-{L}' if f'linear_attn_out-{L}' in names else f'attn_output-{L}'
    block = {'attn': golden(step, blkname), 'ffn': golden(step, f'ffn_out-{L}')}
    after = {'attn': golden(step, f'hc_combine-{L}', 0), 'ffn': golden(step, f'l_last-{L}')}
    for kind, nth in (('attn', 0), ('ffn', 1)):
        p = f'blk.{L}.hc_{kind}'
        wn = m.f32(f'{p}_norm.weight', (HC, D))
        dn, ij, up = m.raw(f'{p}_down.weight'), m.raw(f'{p}_inject.weight'), m.raw(f'{p}_up.weight')
        assert (dn['fmt'] == 'BF16') == (ij['fmt'] == 'BF16'), 'hc down / inject: one BF16, one GG format (the chain gates assume both or neither read xq)'
        def gq(e, xn, dep=-1):   # the q8_1 copy of xn, only for GGUF-format down / inject (BF16 reads the fp32 xn)
            if dn['fmt'] == 'BF16' and ij['fmt'] == 'BF16': return None, dep
            q = q8b(HC * D); return q, e.gg_q8_1(xn.reshape(-1), q, dep=dep)
        # isolated: llama.cpp's hc_norm in
        xn_g = cu(golden(step, f'hc_norm-{L}', nth))
        lo, inj, gate = buf(dn['rows']), buf(ij['rows']), buf(HC, D)
        e = m.engine()
        xq, a = gq(e, xn_g)
        b = mv(e, dn, lo, a, xq=xq, xf=xn_g, rpt=8); c = mv(e, ij, inj, a, xq=xq, xf=xn_g, rpt=8)
        mv(e, up, gate.reshape(-1), b, xf=lo, rpt=16, silu_scale=1.0 / HC)
        e.build().run()
        rep.check(f'{kind}.inject', inj, golden(step, f'hc_inject-{L}', nth), fmt=ij['fmt'])
        rep.check(f'{kind}.gate', gate, golden(step, f'hc_gate-{L}', nth), fmt=dn['fmt'] if dn['fmt'] == up['fmt'] else None)
        xq_g, inj_g, gate_g = xq, inj, gate
        if res_in[kind] is None: continue
        # chain: the residual in
        x = cu(res_in[kind]).reshape(HC, D)
        xn, lo, inj, gate, mixed, res_out = buf(HC, D), buf(dn['rows']), buf(ij['rows']), buf(HC, D), buf(D), buf(HC, D)
        blk = cu(block[kind])
        e = m.engine()
        a = e.rms_mul(x, wn, xn, EPS)
        xq, a = gq(e, xn, a)
        b = mv(e, dn, lo, a, xq=xq, xf=xn, rpt=8); c = mv(e, ij, inj, a, xq=xq, xf=xn, rpt=8)
        d = mv(e, up, gate.reshape(-1), b, xf=lo, rpt=16, silu_scale=1.0 / HC)
        g = e.hc_mean(xn, gate, mixed, dep=d)
        e.also(e.hc_post(blk, inj, x, res_out, dep=g), c)
        e.build().run()
        rep.check(f'{kind}.chain.norm', xn, golden(step, f'hc_norm-{L}', nth))
        if xq is None:      # BF16 down / inject read the fp32 xn: no q8_1 crossing, the chain is float only (1e-6)
            rep.check(f'{kind}.chain.inject', inj, golden(step, f'hc_inject-{L}', nth))
            rep.check(f'{kind}.chain.gate', gate, golden(step, f'hc_gate-{L}', nth))
            eq = True
        else:               # the q8_1 of the engine's xn vs of llama.cpp's hc_norm; the up's per-workgroup q8_1 of
            # silu(lo / hc) is downstream of it (equal xq bytes: the same lo bits, the same up activation)
            eq = rep.q8x(f'{kind}.xq', xq, xq_g, xn, golden(step, f'hc_norm-{L}', nth))
            rep.chain(f'{kind}.chain.inject', eq, inj, inj_g, golden(step, f'hc_inject-{L}', nth))
            rep.chain(f'{kind}.chain.gate', eq, gate, gate_g, golden(step, f'hc_gate-{L}', nth))
        rep.check(f'{kind}.chain.mixed', mixed, golden(step, f'hc_mixed-{L}', nth), diag=not eq)
        rep.check(f'{kind}.chain.post', res_out, after[kind], diag=not eq)
    rep.flush(f'step {step}:')
sys.exit(rep.summary())
