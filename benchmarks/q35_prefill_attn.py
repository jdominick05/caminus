"""The own prefill's attention for one chunk at depth: a chunk of N rows at positions P0 .. P0 + N - 1 attends to the cache
rows 0 .. its own position. Geometries: the dense qwen35 line by default (Swift 1.5 / ThinkingCap / AREX-2: 24 query heads
over 4 kv heads, head dim 256, f16 KV cache); --heads 16 --kv-heads 2 --module q35m_prefill is Qwen3.6-35B-A3B's (its
prefill object's q35mp_attn; --kv q8 its Q8_0 rows, q35mp_attn_q8); --hd 128 --heads 32 --kv-heads 4 is Qwen3-30B-A3B's (q3m_prefill q3p_attn).
Ours: the kernels the prefill drivers run, in their launch splits:
  wm      prefill/q35own.py Ops.attn_wm (kernels/q35_prefill.hip q35p_attn_wm; the 35B's object: kernels/q35m_prefill.hip
          q35mp_attn, q35mp_attn_q8 over Q8 rows: WMMA tiles, a workgroup = 8 positions x one kv head's query heads, wave
          pairs over the 256 dims; launches of at most ATTN_WM_TP row-positions)
  attn16  Ops.attn16 (q35p_attn16: one workgroup per (row, query head) walks every position up to its own; the kernel wm
          replaced; f16 only, slow at depth: --skip attn16)
  q3p     (head dim 128) prefill/q3m_ops.py Ops.attn (q3p_attn: WMMA tiles, 16 positions x 8 query heads; launches of
          prefill/q3m.py's ATTN_TP token-positions)
Yardstick: torch.nn.functional.scaled_dot_product_attention (fp16, GQA; f16 KV only) restricted to its tiled backends
(flash / memory-efficient; never the math path, which would materialize the N x L x heads scores: 17.5 GB at L 32K). An
explicit causal-with-offset mask has no tiled kernel here, so the chunk runs as the two tiled calls it splits into: the
prefix (every row sees cache rows 0 .. P0 - 1: no mask) and the chunk's own causal square (is_causal); their time is
summed, the log-sum-exp merge of the two halves is not counted. Output checked against ours where there is no prefix
(P0 0). rocKE's kernel is benchmarks/rocke_attn.py (its own process: no torch there).
Prints each one's median time over REPS runs and TFLOPS (4 N (P0 + N / 2) heads D: the causal half counted once).
A Q8 cache holds random codes under one small scale per block (valid rows; the time does not depend on the values).
  tools/guard.sh LOG SECS python -I benchmarks/q35_prefill_attn.py [--n 2048] [--p0 0,8192,30720] [--reps 5]
                 [--heads 24 --kv-heads 4 --hd 256] [--module q35_prefill] [--kv f16|q8] [--skip attn16,sdpa]
Run it on the XT alone first."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np, torch
import torch.nn.functional as Fn
from torch.nn.attention import SDPBackend, sdpa_kernel
from caminus.device import capacity
from caminus.engine import k3
from caminus.prefill import q35own, q35moe, q3m, q3m_ops

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=2048); ap.add_argument('--p0', default='0,8192,30720'); ap.add_argument('--reps', type=int, default=5)
ap.add_argument('--heads', type=int, default=24); ap.add_argument('--kv-heads', type=int, default=4)
ap.add_argument('--hd', type=int, default=256, choices=(128, 256)); ap.add_argument('--module', default=None)
ap.add_argument('--kv', default='f16', choices=('f16', 'q8')); ap.add_argument('--skip', default='')
A = ap.parse_args()
H, NKV, D, N, Q8 = A.heads, A.kv_heads, A.hd, A.n, A.kv == 'q8'
SKIP = set(v for v in A.skip.split(',') if v)
if D == 128 and (Q8 or H != 8 * NKV): sys.exit('q35_prefill_attn: head dim 128 runs q3p_attn (f16, 8 query heads per kv head)')
dev = torch.device('cuda:0')
if D == 128: ops = q3m_ops.Ops(dev, module=A.module or 'q3m_prefill')
elif Q8 or A.module == q35moe.MODULE: ops = q35moe.Ops(dev, q8=Q8)
else: ops = q35own.Ops(dev, module=A.module or q35own.MODULE)
WM = ops.WM_KERNEL[Q8] if D == 256 else 'q3p_attn'
print(torch.cuda.get_device_name(dev), f'| chunk {N} rows, {H} q heads / {NKV} kv heads x {D}, {A.kv} KV, {ops.module} {WM}', flush=True)


def bench(fn):
    fn(); torch.cuda.synchronize(dev); ts = []
    for _ in range(A.reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3)
    return float(np.median(ts))


def q8_cache(L):
    """L rows of valid Q8_0 blocks: random codes, every block's fp16 scale 1/256 (values within +-0.5)"""
    c = torch.randint(0, 256, (L * k3.kv_pos_bytes(NKV * D, True),), dtype=torch.uint8, device=dev)
    b = c.view(-1, 34); b[:, 0] = 0x00; b[:, 1] = 0x1c   # (fp16 0x1c00 = 2^-8)
    return c


for P0 in [int(v) for v in A.p0.split(',')]:
    L = P0 + N
    need = (L * k3.kv_pos_bytes(NKV * D, Q8) * 3 + 2 * N * H * D * 4 * 2 + 4 * N * H * D * 4) / 2 ** 20 + 64   # (cache and its
    # k / v copies; q / out; the fp16 q copies, SDPA's output and the fp32 reference copy)
    capacity.check([0], [need], f'q35_prefill_attn P0 {P0} ({need:.0f} MiB)')
    cache = q8_cache(L) if Q8 else torch.empty(L * 2 * NKV * D, dtype=torch.float16, device=dev).normal_(0.0, 0.5)   # (in place)
    q = torch.randn(N * H * D, device=dev); out = torch.zeros(N * H * D, device=dev)
    scale = 1.0 / np.sqrt(D)
    fl = 4.0 * N * (P0 + N / 2) * H * D
    times = {}
    if D == 128:
        q16, out16 = q.half(), torch.zeros(N * H * D, dtype=torch.float16, device=dev)
        _, nb = ops.attn_blocks(P0, N); step = max(1, q3m.ATTN_TP // L // q3m_ops.AT_T)
        launches = len(range(0, nb, step))

        def ours():
            for i0 in range(0, nb, step): ops.attn(q16, cache, P0, N, NKV, scale, out16, i0, min(nb, i0 + step), rows=L)
        times['q3p'] = bench(ours); out = out16.float()
        name = 'q3p'
    else:
        blocks = ops.attn_blocks(P0, N); launches = len(blocks)

        def ours():
            for b0, b1 in blocks: ops.attn_wm(q, cache, P0, N, H, NKV, D, scale, out, L, b0, b1)
        times['wm'] = bench(ours)
        name = 'wm'
        if not Q8 and 'attn16' not in SKIP:
            runs = ops.attn_rows(P0, N)

            def old():
                for r0, r1 in runs: ops.attn16(q, cache, P0, N, H, NKV, D, scale, out, L, r0, r1)
            times['attn16'] = bench(old)
            ours()   # (out back to the tiles' result for the check below)
    t_o = times[name]
    line = f'P0 {P0:6d} (rows to {L:6d}): ours ({name}) {t_o / 1e3:8.2f} ms ({fl / t_o / 1e6:5.1f} TF, {launches} launches)'
    if 'attn16' in times:
        t_p = times['attn16']; line += f' | per-(row, head) {t_p / 1e3:8.2f} ms ({fl / t_p / 1e6:5.1f} TF) | {t_p / t_o:5.1f}x'
    if not Q8 and 'sdpa' not in SKIP:
        kv = cache.view(L, 2, NKV, D)
        k = kv[:, 0].permute(1, 0, 2).unsqueeze(0).contiguous(); v = kv[:, 1].permute(1, 0, 2).unsqueeze(0).contiguous()
        qh = q.view(N, H, D).permute(1, 0, 2).unsqueeze(0).half().contiguous()
        kp, vp = k[:, :, :P0], v[:, :, :P0]; ks, vs = k[:, :, P0:], v[:, :, P0:]

        def tiled(fn):
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]): return fn()

        def sq():
            return tiled(lambda: Fn.scaled_dot_product_attention(qh, ks, vs, is_causal=True, scale=scale, enable_gqa=True))

        def pre():
            return tiled(lambda: Fn.scaled_dot_product_attention(qh, kp, vp, scale=scale, enable_gqa=True))
        torch.cuda.reset_peak_memory_stats(dev); base = torch.cuda.memory_allocated(dev)
        try:
            t_s = bench(sq) + (bench(pre) if P0 else 0.0)
            extra = (torch.cuda.max_memory_allocated(dev) - base) / 2 ** 20
            line += (f' | sdpa tiled (prefix + causal square) {t_s / 1e3:8.2f} ms ({fl / t_s / 1e6:5.1f} TF, +{extra:.0f} MiB peak)'
                     f' | ours / sdpa {t_o / t_s:5.2f}')
            if P0 == 0:
                ref = sq()[0].permute(1, 0, 2).float().reshape(-1)
                line += f' | max |ours - sdpa| {float((out - ref).abs().max()):.2e}'
        except RuntimeError as ex:
            line += f' | sdpa NOT RUN (no tiled backend: {str(ex)[:120]})'
        del k, v, qh, kp, vp, ks, vs
    print(line, flush=True)
    del cache, q, out; torch.cuda.empty_cache()
print('RESULT PASS (ran to the end; a measurement, not a gate)')
