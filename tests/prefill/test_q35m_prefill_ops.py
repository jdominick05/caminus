"""G1 of the qwen35moe line's own prefill (caminus.prefill.q35moe, kernels/q35m_prefill.hip): the ops it adds to the dense
line's (those are tests/prefill/test_q35_prefill_ops.py's), over a chunk's rows, against Qwen3.6-35B-A3B's own weights:

  router     q35mp_router_sh == the decode module's OP_ROUTER (k3_q35_moe: Engine.router_shexp), BITWISE: ids [8] and
             weights [9] (the gated shared expert's in slot 8), one token per pass; the real router and shexp gate of a block
             on random h, plus rows with a tie and a logit spread past fp32's exp range
  norm       q3p_add_rms16 over the 9 weighted slots == OP_OWN_ADD_RMS16 (Engine.own_add_rms16, slots 9, w), BITWISE
  embedding  q35mp_emb_bf16 == OP_OWN_EMB on the BF16 table (Engine.own_emb), BITWISE
  kv q8      q35p_kv_append_q8 (the Q8 KV object, kernels/q35m_prefill_kvq8.hip) == OP_KV_APPEND on the Q8 decode module
             (k3_q35_moe_kvq8), BITWISE: a chunk's rows of rotated-looking fp32 K / V (an all-zero block among them) at a
             position offset; a non-finite value refuses its block (error bit 2), a row past the cache sets bit 1
  experts    q35mp_mm_q2_0 grouped by q3p_group's Q3P_GROUP_Y9 form (gate K 2048 and down K 512 of a block's experts): every (token, slot)
             row against float64 of the exact weights (the blocks decoded on the host) and the fp16 inputs: max |error|
             within 1e-4 of the row's rms (fp32 sums; a layout or scatter bug is O(1)); the pairs' rows land where
             group9's three indices say (yp, y9), and an id out of range sets the routing flag
  tools/guard.sh LOG 600 env HIP_VISIBLE_DEVICES=<a non-display card> python -I tests/prefill/test_q35m_prefill_ops.py GGUF
Prints PASS / FAIL lines and RESULT PASS | FAIL."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import torch  # noqa: E402

from caminus import ggufio  # noqa: E402
from caminus.device import capacity  # noqa: E402
from caminus.engine import k3  # noqa: E402
from caminus.engine.q35_moe import _shapes  # noqa: E402

FAILS = []


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)


def same(name, a, b):
    a, b = a.contiguous().view(-1), b.contiguous().view(-1)
    ia = a.view(torch.int16 if a.element_size() == 2 else torch.int32)
    ib = b.view(torch.int16 if b.element_size() == 2 else torch.int32)
    ne = int((ia != ib).sum()) if a.numel() == b.numel() else -1
    check(ne == 0, f'{name}: bitwise ({a.numel()} values)' if ne == 0 else f'{name}: {ne} of {a.numel()} values differ')


def q2_0_rows(raw, R, K):
    """float64 [R, K] of Q2_0 rows (18-byte blocks: fp16 d, 16 code bytes; value j = bits 2 (j % 4) of byte j / 4,
    w = d (q - 1)): kernels/own_quant_q2_0.h's format, decoded exactly"""
    b = np.frombuffer(raw, dtype=np.uint8).reshape(R, K // 64, 18)
    d = b[:, :, :2].copy().view(np.float16).astype(np.float64)[:, :, 0]
    c = b[:, :, 2:]
    q = np.stack([(c >> (2 * s)) & 3 for s in range(4)], axis=-1).reshape(R, K // 64, 64).astype(np.float64)
    return (d[:, :, None] * (q - 1.0)).reshape(R, K)


def main():
    path = sys.argv[1]
    F = ggufio.GGUFFile(path)
    if F.arch != 'qwen35moe': sys.exit(f'test_q35m_prefill_ops: {path} is {F.arch!r}, not qwen35moe')
    sh = _shapes(F)
    D, E, KU, FF = sh['D'], sh['n_exp'], sh['k_used'], sh['ffn']
    eps = float(F.hp('attention.layer_norm_rms_epsilon'))
    dev = torch.device('cuda', 0)
    print(f'NOTE device: {torch.cuda.get_device_name(dev)} (HIP_VISIBLE_DEVICES={os.environ.get("HIP_VISIBLE_DEVICES")}); '
          f'D {D}, {KU} of {E} experts of {FF}', flush=True)
    T, NE, V = 37, 8, 4096                         # rows; experts uploaded (ids drawn from them); embedding rows read
    RB_G, RB_D = D // 64 * 18, FF // 64 * 18
    mib = (NE * FF * (RB_G * 2) + NE * D * RB_D + V * D * 2 + T * (E + 4 * (KU + 1) * D + 4 * KU * FF) * 4 + E * D * 4) / capacity.MIB
    capacity.check([dev.index], [mib * 1.5 + 64], 'q35m prefill G1 buffers')    # (before anything is allocated)
    from caminus.prefill.q35moe import Ops, ROUTER_FLOOR
    from caminus.prefill import q3m_ops
    O = Ops(dev)
    rng = np.random.default_rng(11)
    put = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
    MOD = 'k3_q35_moe'
    il = 0

    # ---- router + the gated shared weight ---------------------------------------------------------------------------------
    lg = (rng.standard_normal((T, E)) * 2.0).astype(np.float32)
    lg[3, 17] = lg[3, 40] = lg[3].max() + 1.0          # (a tie at the top: the rank's index order decides)
    lg[5] *= 40.0                                       # (exp underflows for most: the softmax's sum from a few)
    sg = (rng.standard_normal(T) * 3.0).astype(np.float32); sg[7] = -90.0; sg[8] = 90.0
    lg, sg = put(lg), put(sg)
    Eg = k3.Engine(module=MOD, T=1)
    l1, s1, i1, w1 = z(E), z(1), z(KU, dt=torch.int32), z(KU + 1)
    Eg.router_shexp(l1, KU, i1, w1, s1); Eg.build()
    oi, ow = z(T, KU, dt=torch.int32), z(T, KU + 1)
    for t in range(T):
        l1.copy_(lg[t]); s1.copy_(sg[t:t + 1]); Eg.run(); oi[t].copy_(i1); ow[t].copy_(w1)
    torch.cuda.synchronize(dev)
    ids2, wts2, err = z(T * KU, dt=torch.int32), z(T * (KU + 1)), z(1, dt=torch.int32)
    O.router_sh(lg.view(-1), sg, E, KU, ROUTER_FLOOR, ids2, wts2, err, T); torch.cuda.synchronize(dev)
    same('router: top-8 ids', ids2, oi); same('router: 8 weights + the gated shared weight', wts2, ow)
    check(int(err.item()) == 0, 'router: finite logits leave the flag clear')
    bad = lg.clone(); bad[2, 9] = float('nan')
    O.router_sh(bad.view(-1), sg, E, KU, ROUTER_FLOOR, ids2, wts2, err, T); torch.cuda.synchronize(dev)
    check(int(err.item()) & 1 == 1, 'router: a NaN logit sets the routing flag (bit 1)')
    del Eg

    # ---- the 9-slot weighted residual add + norm --------------------------------------------------------------------------
    nw = torch.from_numpy(np.array(F.T[f'blk.{il}.post_attention_norm.weight'].pread(0, D * 4).view(np.float32))).to(dev)
    x = put((rng.standard_normal((T, D)) * 3.0).astype(np.float32))
    dn = put((rng.standard_normal((T, KU + 1, D)) * 1.5).astype(np.float32))
    En = k3.Engine(module=MOD, T=1)
    xx, dd, ww, hh, h16 = z(D), z((KU + 1) * D), z(KU + 1), z(D), z(D, dt=torch.float16)
    En.own_add_rms16(xx, nw, hh, h16, eps, d=dd, slots=KU + 1, w=ww); En.build()
    ox, oh, oh16 = z(T, D), z(T, D), z(T, D, dt=torch.float16)
    for t in range(T):
        xx.copy_(x[t]); dd.copy_(dn[t].view(-1)); ww.copy_(ow[t]); En.run(); ox[t].copy_(xx); oh[t].copy_(hh); oh16[t].copy_(h16)
    torch.cuda.synchronize(dev)
    x2, h2, h162 = x.clone().view(-1), z(T * D), z(T * D, dt=torch.float16)
    O.add_rms16(x2, nw, eps, h162, T, D, d=dn.view(-1), w=ow.view(-1), S=KU + 1, h=h2); torch.cuda.synchronize(dev)
    same('9-slot add + norm: x', x2, ox); same('9-slot add + norm: h', h2, oh); same('9-slot add + norm: h16', h162, oh16)
    rn = oh.view(-1).half()                              # (fp16 of the fp32 h, round to nearest even: which side is it?)
    for nm, v in (('prefill', h162), ('decode', oh16.view(-1))):
        bad = torch.nonzero(v.view(torch.int16) != rn.view(torch.int16)).view(-1)[:4].tolist()
        print(f'NOTE h16 of the {nm} op: {int((v.view(torch.int16) != rn.view(torch.int16)).sum())} values not fp16_rn(h)'
              + ''.join(f'; [{i}] h {float(oh.view(-1)[i]):.9g} -> {float(v[i]):.9g} (rn {float(rn[i]):.9g})' for i in bad), flush=True)
    del En

    # ---- the BF16 embedding -----------------------------------------------------------------------------------------------
    te = F.T['token_embd.weight']
    check(te.type == 'BF16', f'token_embd is {te.type} (the prefill reads BF16)')
    embw = put(te.pread(0, V * D * 2))                 # (raw bytes, as Moe35Own uploads the table)
    toks = [0, 1, 17, 1234, V - 1, 5, 4000]
    Ee = k3.Engine(module=MOD, T=1); tk, ex = z(1, dt=torch.int32), z(D)
    Ee.own_emb(embw, 'BF16', V, D, tk, ex); Ee.build()
    e1 = z(len(toks), D)
    for i, t in enumerate(toks): tk.fill_(t); Ee.run(); e1[i].copy_(ex)
    e2 = z(len(toks) * D)
    O.emb_bf16(embw, V, put(np.array(toks, dtype=np.int32)), e2, len(toks), D); torch.cuda.synchronize(dev)
    same('BF16 embedding rows', e2, e1); del Ee

    # ---- the Q8 KV rows: the prefill's append == the decode module's --------------------------------------------------------
    nkv, hd = sh['nkv'], sh['hd']; M = nkv * hd; TK, P0, ROWS = 21, 5, 32
    kq = (rng.standard_normal((TK, M)) * 2.0).astype(np.float32); vq = (rng.standard_normal((TK, M)) * 0.7).astype(np.float32)
    kq[4, 64:96] = 0.0                                  # (an all-zero block: scale 0, codes 0)
    kq, vq = put(kq), put(vq)
    pb = k3.kv_pos_bytes(M, True)
    Eq = k3.Engine(module=k3.Q35MOE_KV_Q8_OF[MOD], T=1)
    k1, v1, c1, ps = z(M), z(M), z(ROWS * pb, dt=torch.uint8), z(1, dt=torch.int32)
    Eq.kv_append(k1, v1, c1, ps, q8=True); Eq.build()
    for t in range(TK):
        k1.copy_(kq[t]); v1.copy_(vq[t]); ps.fill_(P0 + t); Eq.run()
    torch.cuda.synchronize(dev)
    check(int(Eq.kv_error.item()) == 0, 'kv q8: the decode module refused no block')
    Oq = Ops(dev, q8=True)
    c2, kerr = z(ROWS * pb, dt=torch.uint8), z(1, dt=torch.int32)
    Oq.kv_append_q8(kq, M, vq, M, c2, M, P0, TK, ROWS, kerr); torch.cuda.synchronize(dev)
    same(f'kv q8: {TK} rows at positions {P0} .. {P0 + TK - 1} (every byte of the cache)', c2, c1)
    check(int(kerr.item()) == 0, 'kv q8: finite rows leave the error word clear')
    bad = kq.clone(); bad[2, 300] = float('inf'); c3 = c2.clone()
    Oq.kv_append_q8(bad, M, vq, M, c3, M, P0, TK, ROWS, kerr); torch.cuda.synchronize(dev)
    blk = (P0 + 2) * pb + (300 // 32) * 34              # (the refused block's bytes: kept as they were)
    check(int(kerr.item()) & 2 == 2 and bool((c3[blk:blk + 34] == c2[blk:blk + 34]).all()),
          'kv q8: an inf refuses its block (error bit 2) and leaves its bytes alone')
    try:
        Oq.kv_append_q8(kq, M, vq, M, c3, M, ROWS - 3, TK, ROWS, kerr); refused = False
    except (ValueError, k3.ContextBoundsError): refused = True
    check(refused, 'kv q8: a chunk past the cache is refused on the host (k3.check_span)')
    del Eq, Oq

    # ---- the routed experts' Q2_0 GEMMs, grouped ------------------------------------------------------------------------
    def experts(name, R, K):
        t_ = F.T[f'blk.{il}.{name}.weight']
        if t_.type != 'Q2_0': sys.exit(f'{name}: {t_.type}, not Q2_0')
        rb = K // 64 * 18
        raw = t_.pread(0, NE * R * rb)                  # (experts 0 .. NE - 1: [E][R][K] blocks, expert e at e R rb)
        return put(raw), raw, rb
    ids = rng.integers(0, NE, size=(T, KU)).astype(np.int32)
    ids[4] = 3                                          # (one token's 8 slots on one expert: 8 pairs of the same row)
    P = T * KU
    xs, yp, y9 = z(P, dt=torch.int32), z(P, dt=torch.int32), z(P, dt=torch.int32)
    items, nitems = z(4 * q3m_ops.items_bound(P, NE), dt=torch.int32), z(1, dt=torch.int32)
    err.zero_()
    O.group9(put(ids.reshape(-1)), P, KU, NE, xs, yp, y9, items, nitems, err); torch.cuda.synchronize(dev)
    xs_, yp_, y9_ = xs.cpu().numpy(), yp.cpu().numpy(), y9.cpu().numpy()
    check(int(err.item()) == 0 and sorted(yp_.tolist()) == list(range(P)) and np.array_equal(xs_, yp_ // KU)
          and np.array_equal(y9_, (yp_ // KU) * (KU + 1) + yp_ % KU),
          f'group9: every pair once, x row = token, down row = token * {KU + 1} + slot ({int(nitems.item())} work items)')
    bound = q3m_ops.items_bound(P, NE)
    for name, R, K, work, ldy in (('ffn_gate_exps', FF, D, (items, nitems, bound, xs, yp), FF),
                                  ('ffn_down_exps', D, FF, (items, nitems, bound, yp, y9), D)):
        Wd, raw, rb = experts(name, R, K)
        nrow_in = T if name == 'ffn_gate_exps' else P
        x16 = put((rng.standard_normal((nrow_in, K)) * 1.0).astype(np.float16))
        nout = P if name == 'ffn_gate_exps' else T * (KU + 1)
        y = torch.full((nout * ldy,), float('nan'), device=dev)
        O.mm_q2_0(dict(fmt='Q2_0', rows=R, K=K, w=Wd), x16.view(-1), K, y, ldy, work); torch.cuda.synchronize(dev)
        Wf = [q2_0_rows(raw[e * R * rb:(e + 1) * R * rb], R, K) for e in range(NE)]
        X = x16.cpu().numpy().astype(np.float64); Y = y.view(nout, ldy).cpu().numpy().astype(np.float64)
        worst, missing = 0.0, 0
        for p in range(P):
            e, tkn = ids.reshape(-1)[p], p // KU
            xi = tkn if name == 'ffn_gate_exps' else p
            yi = p if name == 'ffn_gate_exps' else tkn * (KU + 1) + p % KU
            ref = Wf[e] @ X[xi]; got = Y[yi, :R]
            if not np.all(np.isfinite(got)): missing += 1; continue
            worst = max(worst, float(np.abs(got - ref).max() / max(np.sqrt(np.mean(ref ** 2)), 1e-30)))
        check(missing == 0 and worst < 1e-4, f'{name} (Q2_0, {R} x {K}, {P} pairs over {NE} experts): every pair row written '
              f'({missing} missing), max |error| {worst:.2e} of the row rms vs float64 (< 1e-4)')
        if name == 'ffn_down_exps':                     # (slot 8 of every token is the shared expert's: untouched)
            check(bool(torch.isnan(y.view(T, KU + 1, D)[:, KU]).all()), 'down: slot 8 of every token untouched (the shared expert\'s row)')
    bad_ids = ids.copy().reshape(-1); bad_ids[5] = NE + 7
    err.zero_(); O.group9(put(bad_ids), P, KU, NE, xs, yp, y9, items, nitems, err); torch.cuda.synchronize(dev)
    check(int(err.item()) & 2 == 2, 'group9: an id out of range sets the routing flag (bit 2)')
    F.close()
    print('RESULT', 'FAIL' if FAILS else 'PASS')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
