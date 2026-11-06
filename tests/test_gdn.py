"""caminus.gdn.fwd_h / bwd_dhu vs fla's chunk_gated_delta_rule_fwd_h / _bwd_dhu on identical inputs, at S1 (B 4, T 2048, H 7) and
N-track-like shapes, with and without dht / h0; plus speed.
  python tests/test_gdn.py
Pass: relative error < 1e-2 for every output (h, v_new, final state; dh, dv2, dh0)."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import torch
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu as fla_dhu, chunk_gated_delta_rule_fwd_h as fla_fwd
from fla.ops.utils.cumsum import chunk_local_cumsum
from caminus.gdn import bwd_dhu, bwd_dhu_supported, fwd_h, fwd_h_supported

torch.manual_seed(0)


def inputs(B, T, H, K=128, V=128, states=False, vfirst=False):
    bf = dict(device='cuda', dtype=torch.bfloat16)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K, device='cuda'), dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, device='cuda'), dim=-1).to(torch.bfloat16)
    w = (torch.randn(B, T, H, K, device='cuda') * 0.1).to(torch.bfloat16)
    do = torch.randn(B, T, H, V, **bf); dv = torch.randn(B, T, H, V, **bf)
    # per-token log gates in (-0.4, 0): chunk-local cumulative sum in log2 space, as fla's forward produces
    g = chunk_local_cumsum(-torch.rand(B, T, H, device='cuda') * 0.4 / 0.6931, chunk_size=64)
    kw = dict(q=q, k=k, w=w, do=do, dv=dv, g=g, scale=K ** -0.5, chunk_size=64, state_v_first=vfirst)
    if states:
        shp = (B, H, V, K) if vfirst else (B, H, K, V)
        kw['h0'] = torch.randn(*shp, device='cuda'); kw['dht'] = torch.randn(*shp, device='cuda')
    return kw


def rel(a, b): return ((a.float() - b.float()).norm() / b.float().norm()).item()


def timeit(fn, n=10):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3


ok = True
CASES = [(4, 2048, 7, False, True), (4, 2048, 7, True, True), (1, 1000, 7, False, True), (4, 2048, 7, False, False), (4, 2048, 7, True, False), (2, 4096, 16, False, False)]
for B, T, H, states, vf in CASES:
    kw = inputs(B, T, H, states=states, vfirst=vf)
    assert bwd_dhu_supported(**kw)
    ref = fla_dhu(**kw); new = bwd_dhu(**kw)
    errs = {'dh': rel(new[0], ref[0]), 'dv2': rel(new[2], ref[2])}
    if states: errs['dh0'] = rel(new[1], ref[1])
    good = all(e < 1e-2 for e in errs.values()); ok &= good
    t_ref, t_new = timeit(lambda: fla_dhu(**kw)), timeit(lambda: bwd_dhu(**kw))
    print(f'bwd_dhu B {B} T {T:4d} H {H:2d}{" +states" if states else ""}{" vfirst" if vf else ""}: ' + ', '.join(f'{k} {v:.1e}' for k, v in errs.items()) +
          f' | {t_ref:.2f} ms -> {t_new:.2f} ms ({t_ref / t_new:.1f}x)  {"ok" if good else "FAIL"}')
for B, T, H, states, vf in CASES:
    kw = inputs(B, T, H, states=states, vfirst=vf)
    fk = dict(k=kw['k'], w=kw['w'], u=kw['dv'], g=kw['g'], chunk_size=64, state_v_first=vf,
              initial_state=kw.get('h0'), output_final_state=states)
    assert fwd_h_supported(**fk)
    ref = fla_fwd(**fk); new = fwd_h(**fk)
    errs = {'h': rel(new[0], ref[0]), 'v_new': rel(new[1], ref[1])}
    if states: errs['ht'] = rel(new[2], ref[2])
    good = all(e < 1e-2 for e in errs.values()); ok &= good
    t_ref, t_new = timeit(lambda: fla_fwd(**fk)), timeit(lambda: fwd_h(**fk))
    print(f'fwd_h B {B} T {T:4d} H {H:2d}{" +states" if states else ""}{" vfirst" if vf else ""}: ' + ', '.join(f'{k} {v:.1e}' for k, v in errs.items()) +
          f' | {t_ref:.2f} ms -> {t_new:.2f} ms ({t_ref / t_new:.1f}x)  {"ok" if good else "FAIL"}')
print('PASS' if ok else 'FAIL')
