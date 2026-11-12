"""The dense qwen35 prefill's attention for one chunk at depth (Swift 1.5 / ThinkingCap / AREX-2 geometry: 24 query heads
over 4 kv heads, head dim 256, f16 KV cache): a chunk of N rows at positions P0 .. P0 + N - 1 attends to the cache rows
0 .. its own position. Ours: prefill/q35own.py Ops.attn16 (kernels/q35_prefill.hip q35p_attn16: one workgroup per (row,
query head) walks every position up to its own; split into launches of at most ATTN_TP token-positions, as the driver
runs it). Yardstick: torch.nn.functional.scaled_dot_product_attention on the same shapes (fp16 q / k / v, GQA, an explicit
causal-with-offset mask; PyTorch picks the backend), which reads K / V once per tile of rows instead of once per row.
Prints each one's median time over REPS runs and TFLOPS (4 N (P0 + N / 2) heads D: the causal half counted once).
  tools/guard.sh LOG SECS python -I benchmarks/q35_prefill_attn.py [--n 2048] [--p0 0,8192,30720] [--reps 5]
Run it on the XT alone first."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np, torch
import torch.nn.functional as Fn
from caminus.device import capacity
from caminus.prefill import q35own

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=2048); ap.add_argument('--p0', default='0,8192,30720'); ap.add_argument('--reps', type=int, default=5)
A = ap.parse_args()
H, NKV, D = 24, 4, 256; N = A.n
dev = torch.device('cuda:0'); ops = q35own.Ops(dev)
print(torch.cuda.get_device_name(dev), f'| chunk {N} rows, {H} q heads / {NKV} kv heads x {D}', flush=True)


def bench(fn):
    fn(); torch.cuda.synchronize(dev); ts = []
    for _ in range(A.reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3)
    return float(np.median(ts))


for P0 in [int(v) for v in A.p0.split(',')]:
    L = P0 + N
    need = (L * 2 * NKV * D * 2 * 2 + 2 * N * H * D * 4 * 2 + N * L * 2) / 2 ** 20 + 64
    capacity.check([0], [need], f'q35_prefill_attn P0 {P0} ({need:.0f} MiB)')
    cache = (torch.randn(L * 2 * NKV * D, device=dev) * 0.5).half()
    q = torch.randn(N * H * D, device=dev); out = torch.zeros(N * H * D, device=dev)
    scale = 1.0 / np.sqrt(D)
    runs = ops.attn_rows(P0, N)

    def ours():
        for r0, r1 in runs: ops.attn16(q, cache, P0, N, H, NKV, D, scale, out, L, r0, r1)

    kv = cache.view(L, 2, NKV, D)
    k = kv[:, 0].permute(1, 0, 2).unsqueeze(0).contiguous(); v = kv[:, 1].permute(1, 0, 2).unsqueeze(0).contiguous()
    qh = q.view(N, H, D).permute(1, 0, 2).unsqueeze(0).half().contiguous()
    rows = torch.arange(N, device=dev).unsqueeze(1) + P0; cols = torch.arange(L, device=dev).unsqueeze(0)
    mask = (cols <= rows).view(1, 1, N, L)

    def sdpa():
        return Fn.scaled_dot_product_attention(qh, k, v, attn_mask=mask, scale=scale, enable_gqa=True)

    t_o = bench(ours); t_s = bench(sdpa)
    ref = sdpa()[0].permute(1, 0, 2).float().reshape(-1)
    err = float((out - ref).abs().max())
    fl = 4.0 * N * (P0 + N / 2) * H * D
    print(f'P0 {P0:6d} (rows to {L:6d}): ours {t_o / 1e3:8.2f} ms ({fl / t_o / 1e6:5.1f} TF, {len(runs)} launches) | sdpa {t_s / 1e3:8.2f} ms '
          f'({fl / t_s / 1e6:5.1f} TF) | ours / sdpa {t_o / t_s:5.2f} | max |ours - sdpa| {err:.2e}', flush=True)
    del cache, q, out, k, v, qh, mask; torch.cuda.empty_cache()
print('RESULT PASS (ran to the end; a measurement, not a gate)')
