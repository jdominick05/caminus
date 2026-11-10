"""kernels/q3m_prefill.hip (the qwen3moe batched prefill's own kernels) against float64 on the model's real rows and random
activations; needs the Qwen3-30B-A3B Q4_K_M GGUF and one card (under 1 GB: projected from the file's header and checked
against the card's guard line before the first transfer).

  GEMMs     Q4_K dense (blk.0 attn_q), Q6_K dense (blk.0 attn_v: the decode's plane layout), Q4_K experts (blk.0 gate) and
            Q6_K experts (blk.0 down) through q3p_group's work items, token counts that leave partial tiles: max and
            mean |y - ref| / rms(ref) within fp32-accumulation bounds (the weights are exact in the B operand)
  glue      q3p_emb bitwise caminus.own_dequant's rows; add_rms16, router, swiglu vs float64; topk's ids equal numpy's
            ranking of the same float32 probabilities and its weights within float32 rounding
  attention q3p_qk_rope vs float64 (norm + RoPE; the cache rows) and q3p_attn vs float64 causal attention over random
            fp16 rows below the chunk, at a depth with partial chunks; and the same output when launched in token splits;
            a span outside the cache is refused by the launch wrappers before any launch
  tools/guard.sh LOG 600 python -I tests/prefill/test_q3m_ops.py GGUF
New kernel code: a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np, torch
from caminus import ggufio, own_dequant
from caminus.device import capacity
from caminus.engine import k3
from caminus.engine.own_layout import q6k_planes
from caminus.prefill import q3m_ops
from caminus.prefill.q3m_ops import Ops

path = sys.argv[1]
F = ggufio.GGUFFile(path)
dev = torch.device('cuda:0')
print(torch.cuda.get_device_name(0), flush=True)
# every model tensor this test puts on the card (each stays cached by the allocator after its section), the test's own
# buffers (the largest: the experts' fp32 output, 1,200 x 2,048) and the HIP context
TENSORS = ('blk.0.attn_q.weight', 'blk.0.attn_v.weight', 'blk.1.attn_k.weight', 'blk.0.ffn_gate_exps.weight',
           'blk.0.ffn_down_exps.weight', 'token_embd.weight', 'blk.3.ffn_gate_inp.weight')
WORK_MIB, CONTEXT_MIB = 96, 256
need_mib = sum(F.T[name].n_bytes for name in TENSORS) / 2 ** 20 + WORK_MIB + CONTEXT_MIB
capacity.check([0], [need_mib], f'test_q3m_ops ({need_mib:.0f} MiB projected)')
print(f'projected {need_mib:.0f} MiB', flush=True)
O = Ops(dev)
g = np.random.default_rng(5)
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


def W(name):
    t = F.T[name]; K = int(t.shape[0]); R, _ = t.rows(); b = np.asarray(t.data())
    if t.type == 'Q6_K': b = q6k_planes(b, R, K)
    return (t.type, int(t.shape[1]), K, torch.from_numpy(np.ascontiguousarray(b)).to(dev)), t


def rel(y, ref):
    r = np.sqrt((ref ** 2).mean()); e = np.abs(y - ref) / r
    return float(e.max()), float(e.mean())


# ---- dense GEMMs
for name, n in (('blk.0.attn_q.weight', 100), ('blk.0.attn_v.weight', 77), ('blk.1.attn_k.weight', 64)):
    w, t = W(name); fmt, R, K, _ = w
    x = (g.standard_normal((n, K)) * 0.7).astype(np.float16)
    y = torch.full((n, R), np.nan, dtype=torch.float32, device=dev)
    O.mm(w, torch.from_numpy(x).to(dev), K, y, R, n=n); torch.cuda.synchronize()
    ref = x.astype(np.float64) @ own_dequant.rows(t, 0, R).astype(np.float64).T
    mx, mn = rel(y.cpu().numpy(), ref)
    check(np.isfinite(y.cpu().numpy()).all() and mx < 2e-5 and mn < 2e-6, f'mm {fmt} dense {name} n {n}: max {mx:.2e}, mean {mn:.2e} of rms')

# ---- expert GEMMs through q3p_group
n, k, E = 150, 8, 128
use = g.choice(E, 12, replace=False)
ids_np = np.stack([g.choice(use, k, replace=False) for _ in range(n)]).astype(np.int32)
ids = torch.from_numpy(ids_np.reshape(-1)).to(dev)
bound = q3m_ops.items_bound(n * k, E)
xs, yd = (torch.zeros(n * k, dtype=torch.int32, device=dev) for _ in range(2))
items = torch.zeros(bound, 4, dtype=torch.int32, device=dev); nitems = torch.zeros(1, dtype=torch.int32, device=dev)
err = torch.zeros(1, dtype=torch.int32, device=dev)
O.group(ids, n * k, k, E, xs, yd, items, nitems, err); torch.cuda.synchronize()
it = items[:int(nitems.item())].cpu().numpy()
cnt = np.bincount(ids_np.reshape(-1), minlength=E)
check(int(err.item()) == 0 and int(nitems.item()) == sum(-(-c // 64) for c in cnt) and
      sorted(yd.cpu().numpy().tolist()) == list(range(n * k)) and all(ids_np.reshape(-1)[yd[i0:i0 + c].cpu().numpy()].tolist() == [e] * c for e, i0, c, _ in it) and
      np.array_equal(xs.cpu().numpy(), yd.cpu().numpy() // k), f'group: {int(nitems.item())} items, every pair once under its expert')
work = (items, nitems, bound, xs, yd)
for name in ('blk.0.ffn_gate_exps.weight', 'blk.0.ffn_down_exps.weight'):
    w, t = W(name); fmt, R, K, _ = w
    x = (g.standard_normal((n * k if 'down' in name else n, K)) * 0.7).astype(np.float16)
    y = torch.full((n * k, R), np.nan, dtype=torch.float32, device=dev)
    wk = work if 'down' not in name else (items, nitems, bound, yd, yd)
    O.mm(w, torch.from_numpy(x).to(dev), K, y, R, work=wk); torch.cuda.synchronize()
    yn = y.cpu().numpy(); ref = np.zeros_like(yn, dtype=np.float64)
    deq = {int(e): own_dequant.rows(t, int(e) * R, int(e) * R + R).astype(np.float64) for e in use}
    for p in range(n * k):
        xr = x[p] if 'down' in name else x[p // k]
        ref[p] = deq[int(ids_np.reshape(-1)[p])] @ xr.astype(np.float64)
    mx, mn = rel(yn, ref)
    check(np.isfinite(yn).all() and mx < 2e-5 and mn < 2e-6, f'mm {fmt} experts {name} ({n} tokens x {k}): max {mx:.2e}, mean {mn:.2e} of rms')

# ---- glue
te = F.T['token_embd.weight']; R_, rb = te.rows(); D = int(te.shape[0])
toks_np = np.array([0, 1, 785, 151935, 99999, -3, 200000], dtype=np.int32)
xe = torch.zeros(len(toks_np), D, device=dev)
O.emb(torch.from_numpy(np.asarray(te.data())).to(dev), rb, R_, torch.from_numpy(toks_np).to(dev), xe, len(toks_np), D)
ref = np.stack([own_dequant.rows(te, int(min(max(t_, 0), R_ - 1)), int(min(max(t_, 0), R_ - 1)) + 1)[0] for t_ in toks_np])
check(np.array_equal(xe.cpu().numpy().view(np.uint32), ref.view(np.uint32)), 'emb: bitwise own_dequant (ids clamped to the vocabulary)')

n, S = 37, 8
x0 = (g.standard_normal((n, D)) * 3).astype(np.float32); dd = g.standard_normal((n, S, D)).astype(np.float32)
ww = g.random((n, S)).astype(np.float32); nw = (g.random(D) + 0.5).astype(np.float32)
xt = torch.from_numpy(x0.copy()).to(dev); h = torch.zeros(n, D, device=dev); h16 = torch.zeros(n, D, dtype=torch.float16, device=dev)
O.add_rms16(xt, torch.from_numpy(nw).to(dev), 1e-6, h16, n, D, d=torch.from_numpy(dd).to(dev), w=torch.from_numpy(ww).to(dev), S=S, h=h)
torch.cuda.synchronize()
xr = x0.astype(np.float64) + np.einsum('ns,nsd->nd', ww.astype(np.float64), dd.astype(np.float64))
hr = xr / np.sqrt((xr ** 2).mean(-1, keepdims=True) + 1e-6) * nw
check(rel(xt.cpu().numpy(), xr)[0] < 1e-6 and rel(h.cpu().numpy(), hr)[0] < 2e-6 and rel(h16.float().cpu().numpy(), hr)[0] < 2e-3,
      f'add_rms16 (8 weighted slots): x {rel(xt.cpu().numpy(), xr)[0]:.1e}, h {rel(h.cpu().numpy(), hr)[0]:.1e}, h16 {rel(h16.float().cpu().numpy(), hr)[0]:.1e}')

wr_ = F.T['blk.3.ffn_gate_inp.weight']; Wr = np.asarray(wr_.data()).view(np.float32).reshape(128, D)
hh = (g.standard_normal((n, D))).astype(np.float32)
lg = torch.zeros(n, 128, device=dev)
O.router(torch.from_numpy(hh).to(dev), torch.from_numpy(Wr.copy()).to(dev), 128, D, lg, n); torch.cuda.synchronize()
check(rel(lg.cpu().numpy(), hh.astype(np.float64) @ Wr.astype(np.float64).T)[0] < 1e-5, 'router logits vs float64')
lgn = lg.cpu().numpy()
lgn[3, 7] = lgn[3, 9] = lgn[3].max() + 1.0   # (a tie: the lower index first)
lg.copy_(torch.from_numpy(lgn))
ids8 = torch.zeros(n * 8, dtype=torch.int32, device=dev); wts8 = torch.zeros(n * 8, device=dev); err.zero_()
O.topk(lg, 128, 8, 6.103515625e-5, ids8, wts8, err, n); torch.cuda.synchronize()
p32 = np.exp(lgn.astype(np.float32) - lgn.max(-1, keepdims=True)); p32 = p32 / p32.sum(-1, keepdims=True)
rid = np.stack([sorted(range(128), key=lambda e: (-p32[t_, e], e))[:8] for t_ in range(n)])
rw = np.take_along_axis(p32, rid, 1); rw = rw / rw.sum(-1, keepdims=True)
check(int(err.item()) == 0 and np.array_equal(ids8.cpu().numpy().reshape(n, 8), rid) and np.abs(wts8.cpu().numpy().reshape(n, 8) - rw).max() < 1e-6,
      f'topk: ids = the ranking (a tie lower index first: {ids8.cpu().numpy().reshape(n, 8)[3, :2].tolist()}), weights within 1e-6')
lgn[5, 2] = np.nan; lg.copy_(torch.from_numpy(lgn)); err.zero_()
O.topk(lg, 128, 8, 6.103515625e-5, ids8, wts8, err, n); torch.cuda.synchronize()
check(int(err.item()) & 1, 'topk: a NaN logit sets the flag')

gg = (g.standard_normal(5000) * 4).astype(np.float32); uu = g.standard_normal(5000).astype(np.float32); y16 = torch.zeros(5000, dtype=torch.float16, device=dev)
O.swiglu(torch.from_numpy(gg).to(dev), torch.from_numpy(uu).to(dev), y16, 5000); torch.cuda.synchronize()
sr = gg.astype(np.float64) / (1 + np.exp(-gg.astype(np.float64))) * uu
check(np.abs(y16.float().cpu().numpy() - sr).max() <= np.abs(sr).max() * 2 ** -10, 'swiglu within fp16 rounding')

# ---- attention: qk_rope and the causal core
nh, nkv, hd = 32, 4, 128
M = nkv * hd
pos0, n = 1000, 75
ctx = pos0 + n
qn = (g.random(hd) + 0.5).astype(np.float32); kn = (g.random(hd) + 0.5).astype(np.float32)
ts = float(np.float32(1e6) ** np.float32(-2.0 / hd))
par = torch.from_numpy(np.concatenate([qn, kn, np.array([ts, 1e-6], np.float32)])).to(dev)
cache_np = (g.standard_normal((ctx, 2, M)) * 1.5).astype(np.float16)
cache = torch.from_numpy(cache_np.reshape(-1).copy()).to(dev)
qkv = (g.standard_normal((n, nh * hd + 2 * M)) * 3).astype(np.float32)
q16 = torch.zeros(n, nh, hd, dtype=torch.float16, device=dev)
O.qk_rope(torch.from_numpy(qkv).to(dev), nh * hd + 2 * M, nh, nkv, par, pos0, q16, cache, n); torch.cuda.synchronize()


def nr(x, w, p):
    x = x.astype(np.float64); x = x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * w.astype(np.float64)
    j = np.arange(hd // 2); th = p * (np.float64(1e6) ** (-2.0 * j / hd)); c, s = np.cos(th), np.sin(th)
    x0, x1 = x[..., :hd // 2], x[..., hd // 2:]
    return np.concatenate([x0 * c - x1 * s, x1 * c + x0 * s], -1)


cn = cache.cpu().numpy().reshape(ctx, 2, M)
qr = np.stack([nr(qkv[t_, :nh * hd].reshape(nh, hd), qn, pos0 + t_) for t_ in range(n)])
kr = np.stack([nr(qkv[t_, nh * hd:nh * hd + M].reshape(nkv, hd), kn, pos0 + t_) for t_ in range(n)])
eq = np.abs(q16.float().cpu().numpy() - qr).max() / np.abs(qr).max(); ek = np.abs(cn[pos0:, 0].astype(np.float64).reshape(n, nkv, hd) - kr).max() / np.abs(kr).max()
ev = np.array_equal(cn[pos0:, 1], qkv[:, nh * hd + M:].astype(np.float16))
check(eq < 1e-3 and ek < 1e-3 and ev and np.array_equal(cn[:pos0], cache_np[:pos0]),
      f'qk_rope: q16 {eq:.1e}, cache K {ek:.1e} of max (fp16), V rows exact, rows below the chunk untouched')
out = torch.zeros(n, nh, hd, dtype=torch.float16, device=dev)
O.attn(q16, cache, pos0, n, nkv, 1.0 / np.sqrt(hd), out); torch.cuda.synchronize()
qf = q16.float().cpu().numpy().astype(np.float64); Kc = cn[:, 0].astype(np.float64).reshape(ctx, nkv, hd); Vc = cn[:, 1].astype(np.float64).reshape(ctx, nkv, hd)
ref = np.zeros((n, nh, hd))
for t_ in range(n):
    P = pos0 + t_ + 1
    for hh_ in range(nh):
        gq = hh_ // 8; sc = Kc[:P, gq] @ qf[t_, hh_] / np.sqrt(hd); e = np.exp(sc - sc.max()); ref[t_, hh_] = (e[:, None] * Vc[:P, gq]).sum(0) / e.sum()
on = out.float().cpu().numpy()
mx, mn = rel(on, ref)
check(np.isfinite(on).all() and mx < 2e-2 and mn < 2e-3, f'attn at positions {pos0} .. {ctx - 1} vs float64: max {mx:.2e}, mean {mn:.2e} of rms (fp16 p)')
out2 = torch.zeros_like(out)
_, nb = O.attn_blocks(pos0, n)
O.attn(q16, cache, pos0, n, nkv, 1.0 / np.sqrt(hd), out2, 0, 2); O.attn(q16, cache, pos0, n, nkv, 1.0 / np.sqrt(hd), out2, 2, nb)
torch.cuda.synchronize()
check(torch.equal(out.view(torch.int16), out2.view(torch.int16)), f'attn: launched in block splits (0 .. 1, 2 .. {nb - 1}), the same bits')
# chunk geometry on the same rows: the chunk cut at an odd position (a pair split across chunks), outputs of the shared tokens equal
cut = 41 - pos0 % 2   # (position pos0 + cut odd)
o3 = torch.zeros_like(out)
O.attn(q16, cache, pos0, cut, nkv, 1.0 / np.sqrt(hd), o3); O.attn(q16[cut:], cache, pos0 + cut, n - cut, nkv, 1.0 / np.sqrt(hd), o3[cut:])
torch.cuda.synchronize()
check(torch.equal(out.view(torch.int16), o3.view(torch.int16)), f'attn: the chunk cut at token {cut} (position {pos0 + cut}) gives the same bits')

# ---- a span outside the cache: refused on the host, nothing launched, the cache and the outputs keep their bytes
qd = torch.from_numpy(qkv).to(dev)
keep = (cache.clone(), q16.clone(), out.clone())
sc_ = 1.0 / np.sqrt(hd)
for what, fn in (('qk_rope at position -1', lambda: O.qk_rope(qd, nh * hd + 2 * M, nh, nkv, par, -1, q16, cache, n)),
                 ('qk_rope one row past the cache', lambda: O.qk_rope(qd, nh * hd + 2 * M, nh, nkv, par, ctx - n + 1, q16, cache, n)),
                 ('attn at position -1', lambda: O.attn(q16, cache, -1, n, nkv, sc_, out)),
                 ('attn one row past the cache', lambda: O.attn(q16, cache, ctx - n + 1, n, nkv, sc_, out)),
                 ('attn at a position that is not an int', lambda: O.attn(q16, cache, 1000.5, n, nkv, sc_, out))):
    try: fn(); r = False
    except k3.ContextBoundsError: r = True
    torch.cuda.synchronize()
    same = all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip((cache, q16, out), keep))
    check(r and same, f'{what}: refused before the launch, cache / q16 / output unchanged')
torch.cuda.synchronize()
print(f'peak {torch.cuda.max_memory_reserved(0) / 2 ** 20:.0f} MiB reserved by the allocator (projected {need_mib:.0f})', flush=True)
check(torch.cuda.max_memory_reserved(0) / 2 ** 20 <= need_mib, 'the allocator stayed under the projection')
print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
