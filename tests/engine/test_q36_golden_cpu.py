"""CPU cross-check of the qwen35moe (Qwen3.6-35B-A3B) reading for the one-card engine, against llama.cpp's own
intermediates (tools/reference/q3n_refs.sh's golden_layers with Q3N_LAYERS=0,1,2,3,19,38,39: layers 0 (GDN) and 3 (attention),
positions 0 .. 7 of the code text), as tests/engine/test_q3n_golden_cpu.py does for qwen35moe. The differences it pins down:
the fused gated-delta-net op maps value head h to key head h % nk (TILED, ggml-cuda gated_delta_net.cu fastmodulo; the
qwen3next interleave h // 2 must NOT match), separate ssm_alpha / ssm_beta tensors, the SiLU output gate, top-8 of 256,
IMRoPE on 64 dims with equal text positions. It checks the reading, not K3. Nothing runs on a GPU; reads only small F32
tensors of the GGUF.
usage: HIP_VISIBLE_DEVICES= python -I tests/engine/test_q36_golden_cpu.py GOLDEN_DIR GGUF"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import caminus.engine.flashnext   # noqa: E402,F401  (puts gguf-py on the path)
from gguf import GGUFReader   # noqa: E402

if len(sys.argv) < 2: raise SystemExit(__doc__)
G = sys.argv[1]
if len(sys.argv) < 3: raise SystemExit(__doc__)
M = sys.argv[2]
rows = [l.rstrip('\n').split('\t') for l in open(f'{G}/index.tsv')]
RES = []


def ok(name, cond, info=''):
    RES.append(bool(cond)); print(f'{"ok  " if cond else "FAIL"} {name} {info}', flush=True)


def get(step, name, k=0):
    """the k-th tensor named `name` at `step`, shaped as numpy [ne3, ne2, ne1, ne0] without the unit dims"""
    hits = [r for r in rows if int(r[0]) == step and r[2] == name]
    r = hits[k]; ne = [int(v) for v in r[4:8]]
    if r[8].startswith('skipped'): return None          # a non-contiguous view gg_golden did not dump
    a = np.fromfile(f'{G}/{r[8]}', dtype=np.int32 if r[3] == 'i32' else np.float32)
    shape = [n for n in ne[::-1]]
    return a.reshape(shape).squeeze() if a.size == np.prod(ne) else a


def rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(1e-30, np.abs(b).max()))


rd = GGUFReader(M, 'r'); T = {t.name: t for t in rd.tensors}
f32 = lambda n: np.asarray(T[n].data, dtype=np.float32)
S, NK, NV, R = 128, 16, 32, 2
KU = 8   # experts per token
TOL = 2e-5

# ---- GDN layer 0 ----
for st in (0, 5):
    cos = get(st, 'conv_output_silu-0'); q = get(st, 'q_conv-0'); k = get(st, 'k_conv-0')
    l2 = lambda x: x / np.sqrt(np.maximum((x.astype(np.float64) ** 2).sum(-1, keepdims=True), 1e-6 ** 2))
    eps = float(rd.fields['qwen35moe.attention.layer_norm_rms_epsilon'].contents())
    l2e = lambda x: x / np.maximum(np.sqrt((x.astype(np.float64) ** 2).sum(-1, keepdims=True)), eps)
    qc = cos[:NK * S].reshape(NK, S); kc = cos[NK * S:2 * NK * S].reshape(NK, S); vc = cos[2 * NK * S:].reshape(NV, S)
    e1 = rel(qc, q); e2 = rel(kc, k)
    ok(f'step {st}: attn_qkv / conv channels are q [0, 2048) | k [2048, 4096) | v [4096, 8192)', max(e1, e2) == 0, f'({e1:.1e} / {e2:.1e})')
    qp, kp, vp = get(st, 'q_conv_predelta-0'), get(st, 'k_conv_predelta-0'), get(st, 'v_conv_predelta-0')
    e = max(rel(l2e(q), qp), rel(l2e(k), kp))
    ok(f'step {st}: q / k l2-normed per key head (16 heads; the fused delta-net op repeats them itself)', e < 2e-4, f'({e:.1e})')
    ok(f'step {st}: v rows are the value heads in order', rel(vc, vp) == 0.0, f'({rel(vc, vp):.1e})')
    al = get(st, 'alpha-0'); bs = get(st, 'beta_sigmoid-0').reshape(-1)
    dt = f32('blk.0.ssm_dt.bias'); sa = f32('blk.0.ssm_a')
    sp = np.log1p(np.exp((al.astype(np.float32) + dt).astype(np.float64)))
    asp = get(st, 'a_softplus-0'); gate = get(st, 'gate-0')
    ok(f'step {st}: a_softplus = softplus(alpha + dt) (separate ssm_alpha tensor)', rel(sp, asp) < 1e-5, f'({rel(sp, asp):.1e})')
    ok(f'step {st}: gate[h] = softplus(..) * ssm_a[h]', rel(asp.astype(np.float64) * sa, gate) < 1e-6, f'({rel(asp.astype(np.float64) * sa, gate):.1e})')
    bsr = 1.0 / (1.0 + np.exp(-get(st, 'beta-0').reshape(-1).astype(np.float64)))
    ok(f'step {st}: beta_sigmoid = sigmoid(beta) (separate ssm_beta tensor)', rel(bsr, bs) < 1e-6, f'({rel(bsr, bs):.1e})')
    beta = bs.astype(np.float64)
    # the delta rule, per value head, from the golden previous state (state_predelta: [H, S, S] numpy = ggml [S, S, H])
    s0 = get(st, 'state_predelta-0').astype(np.float64); s1 = get(st, 'new_state-0').astype(np.float64)
    out = get(st, 'attn_output-0').astype(np.float64)
    g = gate.astype(np.float64); res_ = {}
    for mname, kh in (('tiled h % 16', np.arange(NV) % NK), ('interleaved h // 2', np.arange(NV) // R)):
        qd = qp.astype(np.float64)[kh] / np.sqrt(S); kd = kp.astype(np.float64)[kh]; vd = vp.astype(np.float64)
        M_ = s0 * np.exp(g)[:, None, None]                         # state[h][v][k] (qwen3next's check found this layout)
        kv = np.einsum('hvk,hk->hv', M_, kd); delta = (vd - kv) * beta[:, None]
        M_ = M_ + np.einsum('hv,hk->hvk', delta, kd); o = np.einsum('hvk,hk->hv', M_, qd)
        res_[mname] = max(rel(o, out), rel(M_, s1))
    ok(f'step {st}: gated delta rule with value head h -> key head h % 16 (tiled); the interleave does NOT match',
       res_['tiled h % 16'] < 1e-4 and res_['interleaved h // 2'] > 1e-2, f'({res_})')
    nrm = get(st, 'norm-0', 1).astype(np.float64); z = get(st, 'z-0').astype(np.float64).reshape(NV, S); fo = get(st, 'final_output-0').reshape(NV, S)
    w = f32('blk.0.ssm_norm.weight').astype(np.float64)
    o2 = out / np.sqrt((out ** 2).mean(-1, keepdims=True) + eps)                # llama.cpp's "norm": before the weight
    ok(f'step {st}: ssm_norm: rms over each head\'s 128 values', rel(o2, nrm) < 1e-5, f'({rel(o2, nrm):.1e})')
    nw = nrm * w
    es = rel(nw * (z / (1 + np.exp(-z))), fo); eg = rel(nw * (1 / (1 + np.exp(-z))), fo)
    ok(f'step {st}: final = (rms(o) * w) * SiLU(z), not sigmoid(z)', es < 1e-5 and eg > 1e-3, f'(silu {es:.1e}, sigmoid {eg:.1e})')

# ---- attention layer 3 ----
fb = float(rd.fields['qwen35moe.rope.freq_base'].contents()); nrot = int(rd.fields['qwen35moe.rope.dimension_count'].contents())
toks = [int(v) for v in open(f'{G}/tokens.txt').read().split()]
for st in (0, 6):
    qf = get(st, 'Qcur_full-3').reshape(16, 512).astype(np.float64); gt = qf[:, 256:]
    qnw = f32('blk.3.attn_q_norm.weight').astype(np.float64); qn = get(st, 'Qcur_normed-3').astype(np.float64)
    q0 = qf[:, :256]; alt = qf.reshape(-1)[:4096].reshape(16, 256)       # rejected: q heads contiguous, gates after
    e = rel(q0 / np.sqrt((q0 ** 2).mean(-1, keepdims=True) + eps) * qnw, qn); ea = rel(alt / np.sqrt((alt ** 2).mean(-1, keepdims=True) + eps) * qnw, qn)
    ok(f'step {st}: attn_q rows per head are [q(256) | gate(256)]; q_norm = rms per head * w', e < 1e-5 and ea > 1e-3, f'({e:.1e}; contiguous q {ea:.1e})'); qr = get(st, 'Qcur-3').astype(np.float64)   # after RoPE
    p = st
    j = np.arange(nrot // 2); th = p * fb ** (-2.0 * j / nrot)
    x0, x1 = qn[:, j], qn[:, j + nrot // 2]
    ex = qn.copy(); ex[:, j] = x0 * np.cos(th) - x1 * np.sin(th); ex[:, j + nrot // 2] = x0 * np.sin(th) + x1 * np.cos(th)
    ok(f'step {st}: IMRoPE (equal text positions) == NEOX on the first {nrot} dims, base {fb:g}, position {p}', rel(ex, qr) < 1e-5, f'({rel(ex, qr):.1e})')
    ag = get(st, 'attn_gated-3').astype(np.float64); ap = get(st, 'attn_pregate-3').astype(np.float64)
    ok(f'step {st}: attention output * sigmoid(gate) per head', rel(ap * (1 / (1 + np.exp(-gt.reshape(-1).astype(np.float64)))), ag) < 1e-6)

# ---- MoE + shared expert, layers 0 and 3 ----
for L in (0, 3):
    for st in (0, 7):
        lg = get(st, f'ffn_moe_logits-{L}').astype(np.float64); pr = get(st, f'ffn_moe_probs-{L}').astype(np.float64)
        sm = np.exp(lg - lg.max()); sm /= sm.sum()
        tk = get(st, f'ffn_moe_topk-{L}'); wn = get(st, f'ffn_moe_weights_norm-{L}').astype(np.float64)
        top = np.argsort(-pr, kind='stable')[:KU]
        w = pr[tk]; w = w / max(w.sum(), 6.103515625e-5)
        ok(f'layer {L} step {st}: softmax router, top-8 of 256, weights renormalized', rel(sm, pr) < 1e-5 and set(top) == set(tk) and rel(w, wn) < 1e-5,
           f'(softmax {rel(sm, pr):.1e}, w {rel(w, wn):.1e})')
        dn = get(st, f'ffn_moe_down-{L}').astype(np.float64); mo = get(st, f'ffn_moe_out-{L}').astype(np.float64)
        ok(f'layer {L} step {st}: moe_out = sum_j w_j down_j', rel((dn * wn[:, None]).sum(0), mo) < 1e-5)
        sh = get(st, f'ffn_shexp-{L}').astype(np.float64); sg = get(st, f'shared_expert_gate-{L}').astype(np.float64)
        fo = get(st, f'ffn_out-{L}').astype(np.float64)
        ok(f'layer {L} step {st}: ffn_out = moe_out + shexp * sigmoid(gate)', rel(mo + sh / (1 + np.exp(-sg)), fo) < 1e-5)
        ar = get(st, f'attn_residual-{L}').astype(np.float64); lo = get(st, f'l_out-{L}').astype(np.float64)
        ok(f'layer {L} step {st}: l_out = attn_residual + ffn_out', rel(ar + fo, lo) < 1e-6)
        an = get(st, f'attn_post_norm-{L}').astype(np.float64); pw = f32(f'blk.{L}.post_attention_norm.weight').astype(np.float64)
        ok(f'layer {L} step {st}: post_attention_norm = rms(attn_residual) * w', rel(ar / np.sqrt((ar ** 2).mean() + eps) * pw, an) < 1e-5)
print(f'RESULT {sum(RES)} ok, {len(RES) - sum(RES)} FAIL')
sys.exit(0 if all(RES) else 1)
