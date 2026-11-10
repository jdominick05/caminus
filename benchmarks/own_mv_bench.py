"""Caminus's own quantized matvec (kernels/own_mv.hip) against llama.cpp's (the gg_mv_* standalone kernels, ggml-cuda's
vec_dot), standalone, on real tensors (docs/specs/spec-own-quant.md, order of work step 3):
  accuracy  per variant, |y - y_ref| / rms(y_ref) over 256 sampled rows of the first tensor of each set, y_ref the fp64
            dot of gguf-py's dequantized rows with the fp32 activation (mean and p99 over rows x vectors); activations:
            the model's real layer-0 input (token_embd rows, RMS-normed, times blk.0.attn_norm) where K = n_embd, else
            heavy-tailed synthetic vectors (Student-t 3, 8 channels x 20)
  speed     every variant over the whole set (all tensors of that format and shape, at least 512 MiB: past the
            Infinity Cache), interleaved round by round; median GB/s of the weight bytes. Standalone numbers only: the
            shipped decision is measured inside the K3 module (G-speed)
  tools/guard.sh LOG 900 python -I benchmarks/own_mv_bench.py --arex AREX2.gguf [--q36 QWEN36_GSQ.gguf] [--rounds 7]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, ctypes, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from caminus.engine import gguf as G   # (puts gguf-py on the path; gg_mv_* launchers)
from caminus.device import capacity
from caminus.device.hip import Module
from gguf import GGUFReader, quants

ap = argparse.ArgumentParser()
ap.add_argument('--arex', required=True)
ap.add_argument('--q36', default=None)
ap.add_argument('--rounds', type=int, default=7)
ap.add_argument('--set-mib', type=int, default=512)
ap.add_argument('--nvec', type=int, default=8)
A = ap.parse_args()
dev = torch.device('cuda:0')
rng = np.random.default_rng(0)
OWN = Module('own_mv')
FMTS = {'Q6_K': 'q6_k', 'Q4_K': 'q4_k', 'Q8_0': 'q8_0'}


def tensor_set(rd, fmt):
    """the tensors of one format and one shape (the most common one), in file order, up to set_mib"""
    ts = [t for t in rd.tensors if t.tensor_type.name == fmt and len(t.shape) == 2 and t.name != 'token_embd.weight']
    shapes = {}
    for t in ts: shapes.setdefault(tuple(int(s) for s in t.shape), []).append(t)
    best = max(shapes.values(), key=lambda v: sum(int(t.n_bytes) for t in v))
    out, tot = [], 0
    for t in best:
        out.append(t); tot += int(t.n_bytes)
        if tot >= A.set_mib * 2 ** 20: break
    return out


def layer0_input(rd, K, n):
    T = {t.name: t for t in rd.tensors}
    te = T['token_embd.weight']
    if int(te.shape[0]) != K: return []
    ids = rng.integers(0, int(te.shape[1]), n)
    e = quants.dequantize(np.asarray(te.data)[ids], te.tensor_type).astype(np.float64).reshape(n, -1)
    w = quants.dequantize(np.asarray(T['blk.0.attn_norm.weight'].data), T['blk.0.attn_norm.weight'].tensor_type).astype(np.float64).reshape(-1)
    return list(e / np.sqrt((e ** 2).mean(1, keepdims=True) + 1e-6) * w)


def variants(fmt, K):
    """name -> (prepare(x32) -> args, run(w, nrows, args, out)); prepare runs once per activation vector"""
    nm = FMTS[fmt]
    xd = torch.zeros(K // 32, dtype=torch.float32, device=dev); xs = torch.zeros_like(xd)
    xq = torch.zeros(K, dtype=torch.int8, device=dev); x16 = torch.zeros(K, dtype=torch.float16, device=dev)
    g = lambda n: ((n + 31) // 32,)
    def own(mode):
        k = getattr(OWN, f'own_mv_{nm}_{mode}')
        def prep(x32):
            if mode == 'a8': OWN.own_q8((K // 256 + (K % 256 > 0),), (256,), x32, ctypes.c_int(K), xq, xd, xs); return xq
            if mode == 'a16': OWN.own_f16(((K + 255) // 256,), (256,), x32, ctypes.c_int(K), x16); return x16
            return x32   # (a32, and a16f: fp32 x converted to fp16 in registers)
        def run(w, n, xa, out): k(g(n), (256,), w, ctypes.c_int(n), ctypes.c_int(K), xa, xd, xs, out)
        return prep, run
    def llama():
        qt = quants.GGMLQuantizationType[fmt]
        def prep(x32): return G.quantize_q8_1(x32)
        def run(w, n, yq, out):
            getattr(G._m('k3_gguf_q35h' if qt in G.MV_Q35H else 'k3_gguf_gsq' if qt in G.MV_GSQ else 'k3_gguf'), G.MV[qt])(
                ((n + 3) // 4,), (32, 4), w, yq, out, ctypes.c_int(n), ctypes.c_int(K))
        return prep, run
    return {'llama.cpp q8_1': llama(), 'own a8': own('a8'), 'own a16': own('a16'), 'own a16f': own('a16f'), 'own a32': own('a32')}


def bench(path, fmt):
    rd = GGUFReader(path, 'r')
    ts = tensor_set(rd, fmt)
    K, N = int(ts[0].shape[0]), int(ts[0].shape[1])
    mib = sum(int(t.n_bytes) for t in ts) / 2 ** 20 + 64
    capacity.check([0], [mib], f'own_mv_bench {fmt} set ({len(ts)} tensors)')
    W = [torch.from_numpy(np.ascontiguousarray(np.asarray(t.data)).reshape(-1)).to(dev) for t in ts]
    nbytes = sum(w.numel() for w in W)
    V = variants(fmt, K)
    # accuracy on the first tensor's sampled rows
    rows = np.sort(rng.choice(N, min(256, N), replace=False))
    raw = np.asarray(ts[0].data).reshape(N, -1)[rows]
    W64 = quants.dequantize(raw, ts[0].tensor_type).astype(np.float64).reshape(len(rows), K)
    X = layer0_input(rd, K, A.nvec // 2)
    while len(X) < A.nvec:
        v = rng.standard_t(3, K); v[rng.choice(K, 8, replace=False)] *= 20; X.append(v / np.sqrt((v ** 2).mean()))
    out = torch.zeros(N, dtype=torch.float32, device=dev)
    err = {k: [] for k in V}
    for x in X:
        x32 = torch.tensor(x, dtype=torch.float32, device=dev)
        yref = W64 @ x32.double().cpu().numpy(); sc = np.sqrt((yref ** 2).mean())
        for k, (prep, run) in V.items():
            xa = prep(x32); out.zero_(); run(W[0], N, xa, out); torch.cuda.synchronize()
            err[k].append(np.abs(out.double().cpu().numpy()[rows] - yref) / sc)
    # speed: whole set per variant, interleaved by round
    x32 = torch.tensor(X[0], dtype=torch.float32, device=dev)
    prepped = {k: p(x32) for k, (p, _) in V.items()}
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    gbs = {k: [] for k in V}
    for rd_ in range(A.rounds + 1):
        for k, (_, run) in V.items():
            torch.cuda.synchronize(); ev[0].record()
            for w in W: run(w, N, prepped[k], out)
            ev[1].record(); torch.cuda.synchronize()
            if rd_: gbs[k].append(nbytes / (ev[0].elapsed_time(ev[1]) * 1e-3) / 1e9)
    print(f'\n{os.path.basename(path)}  {fmt}  [{N} x {K}] x {len(ts)} tensors, {nbytes / 2**20:.0f} MiB  ({ts[0].name} for accuracy)')
    for k in V:
        e = np.concatenate(err[k])
        print(f'  {k:16s} err mean {e.mean():.2e}  p99 {np.percentile(e, 99):.2e}   {np.median(gbs[k]):6.1f} GB/s  '
              f'(min {min(gbs[k]):.1f}, max {max(gbs[k]):.1f})', flush=True)
    del W; torch.cuda.empty_cache()


print(torch.cuda.get_device_name(0), flush=True)
bench(A.arex, 'Q6_K'); bench(A.arex, 'Q4_K'); bench(A.arex, 'Q8_0')
if A.q36: bench(A.q36, 'Q8_0')
print('RESULT DONE')
