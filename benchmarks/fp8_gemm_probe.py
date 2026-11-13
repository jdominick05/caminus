"""Emulated-fp8 GEMM probe on gfx1100 (RDNA3): is there anything to gain over fp16/bf16 WMMA?

RDNA3 has no fp8/fp4 matrix or conversion hardware (native fp8 arrived with RDNA4 / CDNA4), so any fp8 GEMM here is
software emulation. This probe measures, per prefill-GEMM shape of the dense qwen35 line (Swift 1.5: D 5120,
ffn 17408, chunk rows M 2048), on one card:
  fp16      torch.matmul fp16 (rocBLAS) - the recorded bar (torch fp16 1.96-1.99 s per 2048 chunk)
  bf16      torch.matmul bf16 (rocBLAS/WMMA)
  fp8_bf16  fp8 e4m3 weights cast to bf16 (exact: every e4m3 value is a bf16) then bf16 matmul - the AITER-style
            software-emulated path; also its accuracy vs fp32
  t_bf16    a Triton tl.dot bf16 kernel (the WMMA path our own kernels use)
  t_fp8     Triton with fp8 inputs (upcast in-kernel, as emulation does) - or 'unsupported' if the backend refuses
  t_int8    Triton tl.dot int8 (v_dot-class) - or 'unsupported' on RDNA3
and prints the static decode-side note: fp8 weights are ~8 bpw vs Q4_K's ~4.5, and decode is bandwidth-bound
(834 of ~888 GB/s measured), so fp8 decode would be slower, not faster.

  tools/guard.sh LOG SECS python -I benchmarks/fp8_gemm_probe.py [--iters 20] [--shapes 2048x5120x17408,...]
Run it on the XT alone first (a peer window may hold a card; check first).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import torch
from caminus.device import capacity

ap = argparse.ArgumentParser()
ap.add_argument('--iters', type=int, default=20)
ap.add_argument('--shapes', default='2048x5120x17408,2048x17408x5120,2048x5120x5120,2048x5120x20480')
A = ap.parse_args()
SHAPES = [tuple(int(v) for v in s.split('x')) for s in A.shapes.split(',')]

print(torch.cuda.get_device_name(0), flush=True)
# project the peak live bytes of one shape (fp32 operands + fp32 ref + fp16/bf16/fp8 copies + one output) and check
# through capacity before any of them are allocated
peak = max((M * K + K * N) * 4 + M * N * 4 + (M * K + K * N) * 4 + (M * K + K * N) + M * N * 2 for (M, K, N) in SHAPES)
capacity.check([0], [peak / 2**20], f'fp8_gemm_probe, {len(SHAPES)} shapes, peak {peak >> 20} MiB')
torch.manual_seed(0)


def bench(fn, iters):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
    t0.record()
    for _ in range(iters): fn()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters          # ms


import triton, triton.language as tl


@triton.jit
def _gemm(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DT: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(N, BN)
    pid_m, pid_n = pid % tl.cdiv(M, BM), pid // tl.cdiv(M, BM)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        kk = k + offs_k
        a = tl.load(A + offs_m[:, None] * K + kk[None, :], mask=(offs_m[:, None] < M) & (kk[None, :] < K), other=0).to(DT)
        b = tl.load(B + kk[:, None] * N + offs_n[None, :], mask=(kk[:, None] < K) & (offs_n[None, :] < N), other=0).to(DT)
        acc += tl.dot(a, b, acc)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_gemm(a, b, dt):
    M, K = a.shape; K2, N = b.shape
    c = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    BM, BN, BK = 128, 128, 32
    grid = (tl.cdiv(M, BM) * tl.cdiv(N, BN),)
    _gemm[grid](a, b, c, M, N, K, BM, BN, BK, dt, num_warps=8)
    return c


rows = []
for (M, K, N) in SHAPES:
    a32 = torch.randn(M, K, dtype=torch.float32, device='cuda') * 0.1
    b32 = torch.randn(K, N, dtype=torch.float32, device='cuda') * 0.1
    ref = a32 @ b32
    out = {'shape': f'{M}x{K}x{N}'}
    a16, b16 = a32.half(), b32.half()
    a_bf, b_bf = a32.bfloat16(), b32.bfloat16()
    # fp8 e4m3 weights, kept as fp8, cast at use
    b8 = b32.to(torch.float8_e4m3fn)
    a8 = a32.to(torch.float8_e4m3fn)

    ms = bench(lambda: a16 @ b16, A.iters);        out['fp16'] = round(2 * M * N * K / (ms * 1e-3) / 1e12, 1)
    ms = bench(lambda: a_bf @ b_bf, A.iters);      out['bf16'] = round(2 * M * N * K / (ms * 1e-3) / 1e12, 1)
    ms = bench(lambda: a_bf @ b8.to(torch.bfloat16), A.iters)
    out['fp8_as_bf16'] = round(2 * M * N * K / (ms * 1e-3) / 1e12, 1)
    err = ((a_bf @ b8.to(torch.bfloat16)).float() - ref).abs().max().item()
    out['fp8_as_bf16_maxerr_vs_fp32'] = round(err, 4)
    import traceback
    def arm(name, fn, iters=None):
        try:
            ms = bench(fn, iters or A.iters)
            out[name] = round(2 * M * N * K / (ms * 1e-3) / 1e12, 1)
        except Exception as e:
            out[name] = f'unsupported ({type(e).__name__})'
            if not out.get('_tb'):
                out['_tb'] = ''.join(traceback.format_exception_only(type(e), e))[:200]
    arm('triton_bf16', lambda: triton_gemm(a_bf, b_bf, tl.bfloat16))
    arm('triton_fp8_upcast', lambda: triton_gemm(a8.to(torch.bfloat16), b8.to(torch.bfloat16), tl.bfloat16))
    arm('triton_int8', lambda: triton_gemm(a32.to(torch.int8), b32.to(torch.int8), tl.int8))
    arm('torch_fp8_matmul', lambda: a8 @ b8, 3)
    rows.append(out)
    print('PROBE ' + json.dumps(out), flush=True)

# decode-side static note: bytes per token for the weights
gb = 27e9
print(f'NOTE decode: Q4_K_M ~4.5 bpw -> {gb * 4.5 / 8 / 1e9:.1f} GB per token read vs fp8 8 bpw -> {gb * 8 / 8 / 1e9:.1f} GB; '
      f'measured read 834 of ~888 GB/s ceiling: fp8 weights would cost ~1.8x the bytes at the same rate', flush=True)
print('RESULT PASS (ran to the end; a measurement, not a gate)')
