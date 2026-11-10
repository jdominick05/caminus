"""Caminus's own matvec inside the K3 module (OP_OWN_MV, build/k3_gguf_q35d_own.hsaco) against llama.cpp's (OP_GG_MV, the
same module), on real tensors (docs/specs/spec-own-quant.md: G-speed's in-module A/B, G-self's grid check):
  checks  OP_OWN_MV's rows are bitwise the standalone own_mv_<fmt>_<act> kernel's (build/own_mv.hsaco), and bitwise the
          same at K3_PER_WGP 3 / 6 / 7 / 8 (the order depends on no grid); error vs fp64 on 256 sampled rows
  speed   tapes over a set of real matrices of one format (>= 512 MiB, past the Infinity Cache), each a segment:
          'free' = independent segments (throughput), 'chain' = each segment waits on the one before (the decode's
          dependency structure); OP_GG_MV gets its q8_1 activation precomputed (its quantize segment not timed), OP_OWN_MV
          reads the fp32 activation. Interleaved by round, median of --rounds; GB/s of weight bytes and ms per tape
  tools/guard.sh LOG 900 python -I benchmarks/own_mv_k3.py --arex AREX2.gguf [--rounds 7]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, ctypes, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from caminus.engine import gguf as G, k3
from caminus.device import capacity
from caminus.device.hip import Module
from gguf import GGUFReader, quants

ap = argparse.ArgumentParser()
ap.add_argument('--arex', required=True)
ap.add_argument('--rounds', type=int, default=7)
ap.add_argument('--set-mib', type=int, default=512)
A = ap.parse_args()
dev = torch.device('cuda:0')
MOD = 'k3_gguf_q35d_own'
OWN = Module('own_mv')
rng = np.random.default_rng(0)
ok_all = True


def check(cond, what):
    global ok_all
    print(('PASS ' if cond else 'FAIL ') + what, flush=True)
    ok_all &= bool(cond)


def tensor_set(rd, fmt):
    ts = [t for t in rd.tensors if t.tensor_type.name == fmt and len(t.shape) == 2 and t.name != 'token_embd.weight']
    shapes = {}
    for t in ts: shapes.setdefault(tuple(int(s) for s in t.shape), []).append(t)
    best = max(shapes.values(), key=lambda v: sum(int(t.n_bytes) for t in v))
    out, tot = [], 0
    for t in best:
        out.append(t); tot += int(t.n_bytes)
        if tot >= A.set_mib * 2 ** 20: break
    return out


def bench(rd, fmt):
    ts = tensor_set(rd, fmt)
    K, N = int(ts[0].shape[0]), int(ts[0].shape[1])
    capacity.check([0], [sum(int(t.n_bytes) for t in ts) / 2 ** 20 + 4 * N * len(ts) * 4 / 2 ** 20 + 64], f'own_mv_k3 {fmt} set')
    W = [torch.from_numpy(np.ascontiguousarray(np.asarray(t.data)).reshape(-1)).to(dev) for t in ts]
    nbytes = sum(w.numel() for w in W)
    v = rng.standard_t(3, K); v[rng.choice(K, 8, replace=False)] *= 20
    x = torch.tensor(v / np.sqrt((v ** 2).mean()), dtype=torch.float32, device=dev)
    yq = G.quantize_q8_1(x)
    rb = W[0].numel() // N
    print(f'\n{fmt} [{N} x {K}] x {len(ts)} tensors, {nbytes / 2**20:.0f} MiB', flush=True)
    # (1) bitwise: the op vs the standalone kernel, across grids; error vs fp64
    rows = np.sort(rng.choice(N, min(256, N), replace=False))
    W64 = quants.dequantize(np.asarray(ts[0].data).reshape(N, -1)[rows], ts[0].tensor_type).astype(np.float64).reshape(len(rows), K)
    yref = W64 @ x.double().cpu().numpy(); sc = np.sqrt((yref ** 2).mean())
    for act in ('a32', 'a16f'):
        ref = torch.zeros(N, device=dev)
        getattr(OWN, f'own_mv_{fmt.lower()}_{act}')(((N + 31) // 32,), (256,), W[0], ctypes.c_int(N), ctypes.c_int(K), x,
                                                    ctypes.c_void_p(0), ctypes.c_void_p(0), ref)
        outs = []
        for nw in (3, 6, 7, 8):
            e = k3.Engine(module=MOD, grid=k3.resident_grid(MOD, nw)); out = torch.full((N,), float('nan'), device=dev)
            e.own_matvec(W[0], fmt, N, K, x, out, act=act); e.build().run(); outs.append(out)
        check(all(torch.equal(o, ref) for o in outs), f'{fmt} {act}: OP_OWN_MV bitwise the standalone kernel at K3_PER_WGP 3/6/7/8')
        e64 = np.abs(ref.double().cpu().numpy()[rows] - yref) / sc
        print(f'      {fmt} {act}: error vs fp64 mean {e64.mean():.2e} p99 {np.percentile(e64, 99):.2e}', flush=True)
    ll = torch.zeros(N, device=dev)
    e = k3.Engine(module=MOD); e.gg_matvec(W[0], fmt, N, K, rb, yq, ll, rows_per_task=16); e.build().run()
    el = np.abs(ll.double().cpu().numpy()[rows] - yref) / sc
    print(f'      {fmt} llama.cpp q8_1 (OP_GG_MV): error vs fp64 mean {el.mean():.2e} p99 {np.percentile(el, 99):.2e}', flush=True)
    # (2) speed: one tape per (variant, structure)
    outs = [torch.zeros(N, device=dev) for _ in W]
    tapes = {}
    for struct in ('free', 'chain'):
        for name in ('llama.cpp OP_GG_MV', 'own a32', 'own a16f'):
            e = k3.Engine(module=MOD)
            prev = -1
            for w, o in zip(W, outs):
                dep = prev if struct == 'chain' else -1
                if name.startswith('llama'): prev = e.gg_matvec(w, fmt, N, K, rb, yq, o, rows_per_task=16, dep=dep)
                else: prev = e.own_matvec(w, fmt, N, K, x, o, act=name.split()[1], rows_per_task=32, dep=dep)
            tapes[(struct, name)] = e.build()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    t = {k: [] for k in tapes}
    for r in range(A.rounds + 1):
        for k, e in tapes.items():
            torch.cuda.synchronize(); ev[0].record(); e.run(); ev[1].record(); torch.cuda.synchronize()
            if r: t[k].append(ev[0].elapsed_time(ev[1]))
    for k in tapes:
        ms = float(np.median(t[k]))
        print(f'  {k[0]:5s} {k[1]:20s} {ms:7.3f} ms  {nbytes / (ms * 1e-3) / 1e9:6.1f} GB/s  (min {min(t[k]):.3f}, max {max(t[k]):.3f})', flush=True)
    del W, outs, tapes; torch.cuda.empty_cache()


print(torch.cuda.get_device_name(0), flush=True)
rd = GGUFReader(A.arex, 'r')
for fmt in ('Q6_K', 'Q4_K', 'Q8_0'): bench(rd, fmt)
print('RESULT', 'PASS' if ok_all else 'FAIL')
