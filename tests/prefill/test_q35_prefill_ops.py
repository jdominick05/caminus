"""G1 of the dense qwen35 line's own prefill (caminus.prefill.q35own): every op but the GEMMs, over a chunk's rows, BITWISE
its K3 decode op on the file's decode module (Swift's k3_q35_swift; ThinkingCap's k3_q35_tcap with the q35_prefill_tcap
object) on the same inputs, one token per pass.

  tools/guard.sh LOG 900 env HIP_VISIBLE_DEVICES=<a non-display card> python -I tests/prefill/test_q35_prefill_ops.py GGUF

Real norm / conv / gate weights of the file (Tensor.pread: ordinary memory), random activations (fixed seeds):
  attention  k norm + RoPE, the f16 KV append, q norm + RoPE, the fp16 output gate: bitwise; the attention over positions
             0 .. pos (the prefill's WMMA tiles, the decode's split pair: within 2e-2 / 2e-3 of rms of each other):
             37 rows at p0 = 0 and at p0 = 600 (rows below p0: the same random f16 cache rows in both), the prefill's
             attention in two launches
  GDN        gates (incl. the softplus branches), the conv + its state, l2, the recurrence, the fp16 output norm: 37 rows from
             random conv / recurrent states; the decode's lazy commit (one more pass commits the last token) against the
             prefill's states written once
  norm       the residual add + RMS norm at 5120 (with and without the add), the F32 alpha matvec (BF16: widened, as the
             engine loads it), the embedding rows (Q4_K, or ThinkingCap's IQ2_S),
             the fp16 swiglu
Prints PASS / FAIL lines, the largest attention launch's time, and RESULT PASS | FAIL."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import torch  # noqa: E402

from caminus import ggufio  # noqa: E402
from caminus.device import capacity  # noqa: E402
from caminus.engine import k3  # noqa: E402
from caminus.engine.q35_dense import _shapes, block_kinds  # noqa: E402

FAILS = []


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)


def bits_equal(a, b):
    a, b = a.contiguous().view(-1), b.contiguous().view(-1)
    if a.numel() != b.numel(): return False, -1
    ia = a.view(torch.int16 if a.element_size() == 2 else torch.int32)
    ib = b.view(torch.int16 if b.element_size() == 2 else torch.int32)
    ne = int((ia != ib).sum())
    return ne == 0, ne


def same(name, a, b):
    ok, ne = bits_equal(a, b)
    check(ok, f'{name}: bitwise ({ne} of {a.numel()} values differ)' if not ok else f'{name}: bitwise ({a.numel()} values)')


def main():
    path = sys.argv[1]
    F = ggufio.GGUFFile(path)
    sh = _shapes(F); kinds = block_kinds(F)
    D, nh, nkv, hd, nrot, S, NK, NV, ffn = (sh[k] for k in ('D', 'nh', 'nkv', 'hd', 'nrot', 'S', 'NK', 'NV', 'ffn'))
    eps = float(F.hp('attention.layer_norm_rms_epsilon'))
    import ctypes
    libm = ctypes.CDLL('libm.so.6'); libm.powf.restype = ctypes.c_float; libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]
    theta = libm.powf(float(F.hp('rope.freq_base')), -2.0 / nrot)
    dev = torch.device('cuda', 0)
    print(f'NOTE device: {torch.cuda.get_device_name(dev)} (HIP_VISIBLE_DEVICES={os.environ.get("HIP_VISIBLE_DEVICES")}); '
          f'D {D}, heads {nh} / {nkv} x {hd}, n_rot {nrot}, GDN {NV} / {NK} x {S}', flush=True)
    ia, ig = kinds.index(False), kinds.index(True)
    f32 = lambda name: torch.from_numpy(np.array(F.T[name].pread(0, F.T[name].n_bytes).view(np.float32))).to(dev)
    ROWS, T, V = 1024, 37, 4096
    CH, KD, VD, QH, M = 2 * S * NK + S * NV, NK * S, NV * S, nh * hd, nkv * hd
    TE = F.T['token_embd.weight'].type
    TC = TE == 'IQ2_S'
    MOD = 'k3_q35_tcap' if TC else 'k3_q35_swift'
    emb_rb = D // 256 * (82 if TC else 144)
    print(f'NOTE decode module {MOD}, prefill object {"q35_prefill_tcap" if TC else "q35_prefill"}; embedding {TE}', flush=True)
    mib = (2 * ROWS * 2 * M * 2 + 16 * T * (2 * QH + CH + 2 * D + ffn) * 4 + 2 * NV * S * S * 4 + V * emb_rb + NV * D * 4) / capacity.MIB
    capacity.check([dev.index], [mib * 1.5 + 64], 'q35 prefill G1 buffers')   # (before anything is allocated)
    from caminus.prefill.q35own import Ops
    from caminus.prefill.attn import Attn
    from caminus.prefill.gdn import Gdn
    from caminus.prefill.glue import Glue
    O, A, Gd, Gl = Ops(dev, module='q35_prefill_tcap' if TC else 'q35_prefill'), Attn(dev, name='prefill_attn_q35d'), Gdn(dev, module='prefill_gdn_q35'), Glue(dev, module='prefill_glue_q35d')
    rng = np.random.default_rng(5)
    put = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)

    # ---- attention ------------------------------------------------------------------------------------------------------
    qn, kn = f32(f'blk.{ia}.attn_q_norm.weight'), f32(f'blk.{ia}.attn_k_norm.weight')
    E = k3.Engine(module=MOD, T=1)
    qf1, kr1, vr1, qo1, ko1, att1, gat1 = z(2 * QH), z(M), z(M), z(QH), z(M), z(QH), z(QH, dt=torch.float16)
    pos1 = z(1, dt=torch.int32); cache1 = z(ROWS * 2 * M, dt=torch.float16)
    kd = E.qk_nr(kr1, hd, nkv, hd, kn, pos1, ko1, nrot, theta, eps)
    ap = E.kv_append(ko1, vr1, cache1, pos1, dep=kd)
    qd = E.qk_nr(qf1, 2 * hd, nh, hd, qn, pos1, qo1, nrot, theta, eps)
    a_ = E.also(E.attn_dec(qo1, cache1, pos1, att1, nh, nkv, hd, 1.0 / np.sqrt(hd), dep=ap, batched=True), qd)
    E.gate_mul16(att1, qf1, 2 * hd, hd, gat1, hd, dep=a_)
    E.build()
    worst_ms = 0.0
    for p0 in (0, 600):
        qf = put((rng.standard_normal((T, 2 * QH)) * 2.0).astype(np.float32))
        kr = put((rng.standard_normal((T, M)) * 2.0).astype(np.float32))
        vr = put((rng.standard_normal((T, M)) * 1.0).astype(np.float32))
        pre = put((rng.standard_normal((ROWS, 2 * M)) * 0.7).astype(np.float16)).view(-1)
        cache1.copy_(pre)
        out1 = dict(ko=z(T, M), qo=z(T, QH), att=z(T, QH), gat=z(T, QH, dt=torch.float16))
        for t in range(T):
            qf1.copy_(qf[t]); kr1.copy_(kr[t]); vr1.copy_(vr[t]); pos1.fill_(p0 + t)
            E.run()
            for k_, b in (('ko', ko1), ('qo', qo1), ('att', att1), ('gat', gat1)): out1[k_][t].copy_(b)
        torch.cuda.synchronize(dev)
        cache2 = pre.clone(); err = z(1, dt=torch.int32)
        ko2, qo2, att2, gat2 = z(T * M), z(T * QH), z(T * QH), z(T * QH, dt=torch.float16)
        A.qk_nr(kr.view(-1), hd, nkv, hd, kn, nrot, theta, eps, p0, ko2, T)
        O.kv_append16(ko2, M, vr.view(-1), M, cache2, M, p0, T, ROWS, err)
        A.qk_nr(qf.view(-1), 2 * hd, nh, hd, qn, nrot, theta, eps, p0, qo2, T)
        torch.cuda.synchronize(dev)
        B0, B1 = O._wm_span(p0, T)
        for b0, b1 in ((B0, B0 + 2), (B0 + 2, B1)):     # (two launches)
            t0 = time.perf_counter()
            O.attn_wm(qo2, cache2, p0, T, nh, nkv, hd, 1.0 / np.sqrt(hd), att2, ROWS, b0, b1)
            torch.cuda.synchronize(dev); worst_ms = max(worst_ms, 1e3 * (time.perf_counter() - t0))
        O.gate_mul16(att2, qf.view(-1), gat2, T * QH, hd, 2 * hd, hd)
        torch.cuda.synchronize(dev)
        check(int(err.item()) == 0, f'p0 {p0}: the KV append wrote every row (error word {int(err.item())})')
        same(f'attention p0 {p0}: k norm + RoPE', ko2, out1['ko']); same(f'attention p0 {p0}: q norm + RoPE', qo2, out1['qo'])
        same(f'attention p0 {p0}: the f16 cache', cache2, cache1)
        # the decode's attention is the split pair (fp32 q), the prefill's WMMA tiles (fp16 q, k, v, p): both against
        # float64 in tests/engine/test_q35_split_attn.py / tests/prefill/test_q35_prefill_attn.py; here against each other
        a2, a1 = att2.view(T, -1).double(), out1['att'].double()
        rms = float(a1.pow(2).mean().sqrt()); dmax = float((a2 - a1).abs().max()) / rms; dmean = float((a2 - a1).abs().mean()) / rms
        check(dmax < 2e-2 and dmean < 2e-3, f'attention p0 {p0}: the attention (tiles vs the decode\'s split pair): max {dmax:.2e}, mean {dmean:.2e} of rms')
        g2, g1 = gat2.view(T, -1).double(), out1['gat'].double()
        gr = float(g1.pow(2).mean().sqrt()); gmax = float((g2 - g1).abs().max()) / gr
        check(gmax < 2e-2, f'attention p0 {p0}: the fp16 gate on it: max {gmax:.2e} of rms (bitwise when the attention is: the same op)')
    print(f'NOTE the slowest attention launch: {worst_ms:.2f} ms (rows x positions <= {T} x {600 + T})', flush=True)
    # a row past the cache is refused on the host before any launch
    try:
        O.kv_append16(ko2, M, vr.view(-1), M, cache2, M, ROWS - 3, T, ROWS, err); refused = False
    except k3.ContextBoundsError: refused = True
    check(refused, 'kv_append16: a chunk past the cache rows is refused on the host')
    del E, cache1, cache2, pre

    # ---- GDN --------------------------------------------------------------------------------------------------------------
    p = f'blk.{ig}.'
    dt_, a_w, conv_w, nw = f32(p + 'ssm_dt.bias'), f32(p + 'ssm_a'), f32(p + 'ssm_conv1d.weight').view(CH, 4), f32(p + 'ssm_norm.weight')
    al = (rng.standard_normal((T, NV)) * 2.0).astype(np.float32); be = (rng.standard_normal((T, NV)) * 2.0).astype(np.float32)
    al[3] = -40.0; al[7] = 25.0; be[11] = 30.0                               # (softplus ~ 0, the ab > 20 branch, beta_sig = 1)
    al, be = put(al), put(be)
    qkv = put((rng.standard_normal((T, CH)) * 0.8).astype(np.float32))
    zz = put((rng.standard_normal((T, VD)) * 1.0).astype(np.float32))
    st0 = put((rng.standard_normal((CH, 3)) * 0.5).astype(np.float32))
    g0 = put((rng.standard_normal(NV * S * S) * 0.05).astype(np.float32))
    E = k3.Engine(module=MOD, T=1)
    al1, be1, gt1, x1, cv1, qk1, z1, o1, fin1 = z(NV), z(NV), z(2 * NV), z(CH), z(CH), z(2 * KD), z(VD), z(VD), z(VD, dt=torch.float16)
    cst1, cx1, blk1 = st0.clone(), z(CH), z(k3.gdn_block_size(NV, S, NK, 1))
    blk1[:NV * S * S].copy_(g0)
    g_ = E.gdn_gates(al1, be1, dt_, a_w, gt1)
    c_ = E.gdn_conv(x1, conv_w, cst1, cv1, cx1)
    l_ = E.gdn_l2(cv1, qk1, 2 * NK, S, eps, x_stride=CH, dep=c_)
    st = E.also(E.gdn_step(qk1, cv1[2 * KD:], gt1, blk1, o1, NV, S, NK, CH, dep=l_), g_)
    E.gdn_save(qk1, cv1[2 * KD:], gt1, blk1, NV, S, NK, CH, dep=st)
    E.gdn_norm16(o1, nw, z1, fin1, NV, S, eps, dep=st)
    E.build()
    out1 = dict(gt=z(T, 2 * NV), cv=z(T, CH), qk=z(T, 2 * KD), o=z(T, VD), fin=z(T, VD, dt=torch.float16))
    for t in range(T + 1):
        E.params[1] = 0 if t == 0 else 1
        if t < T: al1.copy_(al[t]); be1.copy_(be[t]); x1.copy_(qkv[t]); z1.copy_(zz[t])
        else: al1.zero_(); be1.zero_(); x1.zero_(); z1.zero_()
        E.run()
        if t < T:
            for k_, b in (('gt', gt1), ('cv', cv1), ('qk', qk1), ('o', o1), ('fin', fin1)): out1[k_][t].copy_(b)
    torch.cuda.synchronize(dev)
    gt2, cv2, qk2, o2, fin2 = z(T * 2 * NV), z(T * CH), z(T * 2 * KD), z(T * VD), z(T * VD, dt=torch.float16)
    cst2, blk2 = st0.clone(), z(k3.gdn_block_size(NV, S, NK, 1)); blk2[:NV * S * S].copy_(g0)
    Gd.gates(al.view(-1), be.view(-1), dt_, a_w, gt2, T, NV)
    Gl.gdn_conv(qkv.view(-1), conv_w, cst2, cv2, CH, T)
    Gd.l2(cv2, qk2, T, 2 * NK, eps, ldx=CH)
    Gd.step(qk2, cv2[2 * KD:], gt2, blk2, o2, T, NV, NK, CH)
    O.gdn_norm16(o2, nw, zz.view(-1), fin2, T, NV, S, eps)
    torch.cuda.synchronize(dev)
    same('GDN: gates', gt2, out1['gt']); same('GDN: conv', cv2, out1['cv']); same('GDN: l2', qk2, out1['qk'])
    same('GDN: recurrence output', o2, out1['o']); same('GDN: fp16 output norm', fin2, out1['fin'])
    same('GDN: conv state after the chunk (the decode: after the commit pass)', cst2, cst1)
    same('GDN: recurrent state after the chunk (the decode: after the commit pass)', blk2[:NV * S * S], blk1[:NV * S * S])
    del E

    # ---- norm, alpha, embedding, swiglu -----------------------------------------------------------------------------------
    anw = f32(f'blk.{ig}.attn_norm.weight')
    x = put((rng.standard_normal((T, D)) * 3.0).astype(np.float32)); d = put((rng.standard_normal((T, D)) * 1.5).astype(np.float32))
    for with_add in (False, True):
        E = k3.Engine(module=MOD, T=1)
        xx, dd, hh, h16 = z(D), z(D), z(D), z(D, dt=torch.float16)
        E.own_add_rms16(xx, anw, hh, h16, eps, d=dd if with_add else None, slots=1 if with_add else 0); E.build()
        ox, oh, oh16 = z(T, D), z(T, D), z(T, D, dt=torch.float16)
        for t in range(T):
            xx.copy_(x[t]); dd.copy_(d[t]); E.run(); ox[t].copy_(xx); oh[t].copy_(hh); oh16[t].copy_(h16)
        torch.cuda.synchronize(dev)
        x2, h2, h162 = x.clone().view(-1), z(T * D), z(T * D, dt=torch.float16)
        O.add_rms16(x2, anw, eps, h162, T, D, d=d.view(-1) if with_add else None, h=h2)
        torch.cuda.synchronize(dev)
        w = 'add + norm' if with_add else 'norm'
        same(f'{w} at {D}: x', x2, ox); same(f'{w} at {D}: h', h2, oh); same(f'{w} at {D}: h16', h162, oh16)
        del E
    if F.T[f'blk.{ig}.ssm_alpha.weight'].type in ('F32', 'BF16'):
        ta = F.T[f'blk.{ig}.ssm_alpha.weight']; ra = ta.pread(0, ta.n_bytes)
        wa = ra.view(np.float32) if ta.type == 'F32' else (ra.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        Wa = torch.from_numpy(np.array(wa)).to(dev).view(NV, D)
        E = k3.Engine(module=MOD, T=1); hh, oa = z(D), z(NV)
        E.f32_mv(Wa, hh, oa, rows_per_task=4); E.build()
        o1 = z(T, NV)
        for t in range(T): hh.copy_(x[t]); E.run(); o1[t].copy_(oa)
        o2 = z(T * NV); Gl.f32_mv(Wa, NV, D, x.view(-1), o2, T); torch.cuda.synchronize(dev)
        same('F32 alpha matvec', o2, o1); del E
    else:
        print(f"NOTE ssm_alpha is {F.T[f'blk.{ig}.ssm_alpha.weight'].type}: the prefill takes the own GEMM (not part of G1)", flush=True)
    te = F.T['token_embd.weight']
    embw = put(te.pread(0, V * emb_rb))
    toks = [0, 1, 17, 1234, V - 1, 5, 4000]
    E = k3.Engine(module=MOD, T=1); tk, ex = z(1, dt=torch.int32), z(D)
    E.own_emb(embw, TE, V, D, tk, ex); E.build()
    e1 = z(len(toks), D)
    for i, t in enumerate(toks): tk.fill_(t); E.run(); e1[i].copy_(ex)
    tk2 = put(np.array(toks, dtype=np.int32)); e2 = z(len(toks) * D)
    (O.emb_iq2s if TC else O.emb)(embw, emb_rb, V, tk2, e2, len(toks), D); torch.cuda.synchronize(dev)
    same(f'{TE} embedding rows', e2, e1); del E
    g = put((rng.standard_normal((T, ffn)) * 3.0).astype(np.float32)); u = put((rng.standard_normal((T, ffn)) * 2.0).astype(np.float32))
    E = k3.Engine(module=MOD, T=1); gg, uu, y16 = z(ffn), z(ffn), z(ffn, dt=torch.float16)
    E.own_swiglu16(gg, uu, y16); E.build()
    s1 = z(T, ffn, dt=torch.float16)
    for t in range(T): gg.copy_(g[t]); uu.copy_(u[t]); E.run(); s1[t].copy_(y16)
    s2 = z(T * ffn, dt=torch.float16); O.swiglu(g.view(-1), u.view(-1), s2, T * ffn); torch.cuda.synchronize(dev)
    same('fp16 swiglu (the decode fuses it into the gate / up matvec; this is the separate op)', s2, s1)
    print('RESULT', 'FAIL' if FAILS else 'PASS')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
