# Yardsticks: outside RDNA3 kernels at Caminus's shapes

Caminus's math is its own. Outside projects that ship gfx1100 kernels are measured here as yardsticks, at the shapes
Caminus actually runs, and read as sources of ideas or code to borrow (with their notices, THIRD_PARTY.md). None of
them is a dependency of the package. Rows here are not release rows: docs/dev/benchmarks.md holds what every release
reruns; a yardstick needs outside installs, so it is rerun when the kernel it measures changes.

The projects, with the versions measured:

| Project | Licence | What runs on gfx1100 | Measured at |
|---|---|---|---|
| FlyDSL (ROCm/FlyDSL) | Apache-2.0 | RDNA3 WMMA GEMMs: fp16 (`kernels/gemm/rdna3_f16_gemm.py`) and int8 -> int32 (`rdna3_int8_gemm.py`), autotuned tiles | wheel 0.5.0 + source tag v0.5.0 (226cde8) |
| rocKE (ROCm/rocm-libraries, hip-kernel-provider) | MIT | RDNA WMMA attention: split-KV GQA decode (`wmma_fmha_decode`), prefill (`wmma_fmha_fwd_v2`), gfx11-generic | not yet measured |
| Composable Kernel (ROCm/rocm-libraries) | MIT | ck_tile FMHA, a gfx1100 batch-prefill pipeline policy | not yet measured |
| AITER (ROCm/aiter) | MIT | gfx1100 "Experimental": mostly Triton / FlyDSL kernels | not measured (Triton) |
| int8-aiter (a downstream AITER fork) | MIT | none: gfx908 (MI100) MFMA kernels; design reference only (int8 KV decode attention with per-token-head scales, Hadamard KV rotation, verify positions packed per KV pass) | - |

Every speed row states each side's harness, input format, KV type, whether the launch is included, and a bandwidth
floor (bytes / 952 GB/s, the XTX's measured read peak; the XT's memory is about 20 % slower, so x-floor figures
measured on the XT overstate the distance to that card's own floor by up to ~20 %).

## Decode and verify attention at depth (dense qwen35 geometry)

`benchmarks/attn_depth.py`, RX 7900 XT alone, testing 737df92. One attention layer of Swift 1.5 / ThinkingCap /
AREX-2 (24 query heads over 4 kv heads, head dim 256): a K3 tape of only the attention op(s) on a zeroed KV cache of P
positions, median of 30 host-timed persistent launches (launch included; an empty tape is 12-21 us). x16 = the 16
attention blocks of a token. Floor = the KV bytes read once for all T rows.

| P | own (k3_q35_swift), T 1 | split f16 (k3_gguf_q35d), T 1 | split q8_0, T 1 | own, T 4 | split f16, T 4 | split q8_0, T 4 | floor f16 / q8_0 |
|---|---|---|---|---|---|---|---|
| 1,024 | 130.8 us | 33.6 | 34.8 | 145.2 | 55.9 | 58.2 | 4.4 / 2.3 |
| 4,096 | 474.6 | 58.5 | 60.4 | 529.3 | 149.3 | 159.6 | 17.6 / 9.4 |
| 16,384 | 1,864.2 | 193.6 | 176.9 | 2,087.6 | 628.6 | 615.2 | 70.5 / 37.4 |
| 32,768 | 4,354.4 | 412.3 | 381.9 | 4,448.2 | 1,297.2 | 1,324.3 | 141.0 / 74.9 |
| 49,152 | 6,517.2 | 624.7 | 579.0 | 6,655.2 | 1,930.8 | 1,975.2 | 211.5 / 112.3 |
| x16 at 32K | **69.7 ms** | **6.6 ms** | 6.1 ms | 71.2 ms | 20.8 ms | 21.2 ms | 2.3 / 1.2 ms |

What it says:
- Swift's and ThinkingCap's own modules run decode attention as one workgroup per query head walking every position
  (`own_op_attn_dec`): 26-31x the bandwidth floor at every depth. At 32K that is 69.7 ms of a token; the measured
  Swift decode step there is 84.4 ms (24 ms at short context), so this one op is nearly all of the depth cost.
- Our own split pair (flash decoding, `kernels/k3_attn_split.h`, already in the AREX-2 / Coder / Qwen3.6 modules) runs
  the same geometry at 2.7-3.3x the floor: 10.6x faster at 32K. Generalizing it into the Swift / ThinkingCap modules is
  a step of the dense-27B plan that had not been built; it is the next change (its own pull request, accuracy-gated:
  the split pair is not bitwise the one-head op).
- A q8_0 cache halves the bytes but saves only 7-9 % here (the split pair is not at the floor yet); an int8 KV pays only
  once the kernel is near the bandwidth floor.
- T = 4 on the split pair costs ~3.1x T = 1 at depth: each verify row reads the cache again. Packing the T rows into
  one KV pass (int8-aiter's verify packing; rocKE's GQA-packed decode) is the lead for verify passes at depth.
- Outside yardsticks for this table (rocKE split-KV decode, CK) are pending.

## Prefill GEMMs (dense qwen35, Swift 1.5 Q4_K_M shapes)

`benchmarks/q35_prefill_gemm.py`, RX 7900 XT alone, testing 737df92. Each GEMM of one GDN block (blk.0) and one
attention block (blk.3) at its real format and shape, M = 2048 tokens:
- ours reads the GGUF blocks and dequantizes inside the WMMA GEMM (`kernels/q35_prefill.hip`), eager launches;
- torch.matmul: dense fp16 operands (PyTorch's rocBLAS / hipBLASLt pick), eager;
- FlyDSL (0.5.0): its RDNA3 fp16 GEMM (fp32 out) and int8 GEMM (int32 out) on their autotuned tiles, timed by replaying
  the launches from a captured HIP graph (its own tuner's method; through its Python dispatch an eager call costs
  ~30 ms of host time, so eager timing measures that, not the kernel).
The yardsticks read 2 (fp16) or 1 (int8) bytes per weight and do no dequantization. Median of 7 groups of 10 launches.

| Shape (R x K) | format | ours | torch fp16 | FlyDSL fp16 | FlyDSL int8 |
|---|---|---|---|---|---|
| attn_qkv 10240 x 5120 | Q8_0 | 5,718 us (37.6 TF) | 4,259 (50.4) | 2,890 (74.3) | 2,772 (77.5) |
| attn_gate 6144 x 5120 | Q8_0 | 3,454 (37.3) | 2,630 (49.0) | 1,781 (72.4) | 1,695 (76.0) |
| ssm_out 5120 x 6144 | Q8_0 | 3,366 (38.3) | 2,699 (47.7) | 1,755 (73.4) | 1,696 (76.0) |
| ffn_gate 17408 x 5120 | Q4_K | 7,271 (50.2) | 7,234 (50.5) | 5,051 (72.3) | 4,732 (77.2) |
| ffn_up 17408 x 5120 | Q6_K | 8,149 (44.8) | 7,184 (50.8) | 5,064 (72.1) | 4,732 (77.2) |
| ffn_down 5120 x 17408 | Q6_K | 8,256 (44.2) | 7,366 (49.6) | 5,033 (72.5) | 4,771 (76.5) |
| attn_q 12288 x 5120 | Q4_K | 5,179 (49.8) | 5,115 (50.4) | 3,458 (74.5) | 3,350 (76.9) |
| attn_k / attn_v 1024 x 5120 | Q8_0 | 702 (30.6) | 419 (51.2) | 411 (52.2) | 335 (64.1) |
| attn_output 5120 x 6144 | Q8_0 | 3,370 (38.2) | 2,705 (47.6) | 1,756 (73.4) | 1,696 (76.0) |
| ffn_down (blk.3) 5120 x 17408 | Q4_K | 7,400 (49.3) | 7,303 (50.0) | 5,034 (72.5) | 4,771 (76.5) |
| **all GEMMs of a 2048-token chunk** (48 GDN + 16 attention blocks) | | **2,264 ms** | 1,995 ms | 1,374 ms | 1,298 ms |
| same at M 512 | | 595 ms | 523 ms | 392 ms | 342 ms |

What it says:
- FlyDSL's RDNA3 fp16 GEMM runs these shapes at 72-75 TF on the XT: 1.65x ours per chunk and 1.45x rocBLAS. Its int8
  GEMM is only ~5 % faster than its fp16 one: on gfx1100 iu8 WMMA has fp16's rate, so int8 buys bytes, not arithmetic.
- Ours: Q4_K at rocBLAS's ~50 TF, Q6_K 12-14 % behind it, Q8_0 25-34 % behind, the 1024-row k / v projections 67 %
  behind. Its tile reads quantized blocks and dequantizes in the loop, which a dense GEMM does not; FlyDSL's numbers are
  the ceiling for a dense fp16 tile on this card at these shapes, not a drop-in.
- The borrowable part is FlyDSL's tile design (4 waves, double-buffered LDS ping-pong, 128 x 128 x 32 default tile,
  swizzled grid, shape-autotuned tile ladder; Apache-2.0): our GEMMs keep their in-tile dequantization and adopt the
  pipeline. Upper bound if our tiles reached FlyDSL's fp16 rate: a chunk's GEMMs 2.26 -> ~1.4 s.
- GEMMs are most of a prefill chunk: 2.26 s of GEMMs per 2048 tokens on the XT.
- FlyDSL's fp16 output matches torch's on the same operands to <= 3.6e-4 relative (every shape above).

## Prefill attention at depth (dense qwen35 geometry)

`benchmarks/q35_prefill_attn.py`, RX 7900 XT alone, testing 737df92. One 2048-row chunk at positions P0 .. P0 + 2047
attends to the f16 cache rows up to its own (24 query heads over 4 kv heads, head dim 256). Ours:
`q35p_attn16`, one workgroup per (row, query head) walking every position up to its own, in launches of at most 2^20
token-positions, as the prefill driver runs it. Yardstick: torch's `scaled_dot_product_attention` (fp16, GQA, an
explicit causal-with-offset mask; PyTorch picks the backend). TFLOPS count the causal half once. Median of 5.

| P0 | ours | torch SDPA | ours / SDPA | max abs difference |
|---|---|---|---|---|
| 0 | 25.7 ms (2.0 TF) | 11.8 ms (4.4 TF) | 2.18 | 5.2e-4 |
| 8,192 | 242.9 ms (1.9 TF) | 59.2 ms (7.8 TF) | 4.10 | 1.0e-5 |
| 30,720 | 963.4 ms (1.7 TF) | 182.2 ms (8.8 TF) | 5.29 | 5.0e-6 |

What it says:
- Our prefill attention is not tiled: each (row, head) workgroup re-reads K and V, so it runs at ~2 TF at every depth
  and its cost grows ~31 ms per 1K of depth per chunk: the chunk at 30K spends 0.96 s in attention against 2.26 s in
  all its GEMMs (43 %); at this rate attention would equal the GEMMs near 70K.
- SDPA (a tiled kernel) is 2-5x faster and still far from the card's WMMA rate (8.8 TF); rocKE's RDNA WMMA prefill
  (`wmma_fmha_fwd_v2`) and CK's gfx1100 FMHA policy are the yardsticks that would show the ceiling (pending).
- The fix is our own WMMA flash-attention tile for prefill (K / V tiles in LDS shared by a block of rows, online
  softmax), with SDPA / rocKE as the bar.

## Summary: where the outside kernels say we stand (XT)

| Op | ours | best yardstick measured | gap |
|---|---|---|---|
| Decode attention, Swift / ThinkingCap modules, 32K | 4.35 ms / layer | our own split pair: 0.41 ms | 10.6x (own fix, not outside code) |
| Verify attention (T 4), split pair, 32K | 1.30 ms / layer (3.1x T 1) | - (rocKE GQA-packed decode pending) | rows re-read the cache |
| Prefill GEMMs, Swift chunk of 2048 | 2.26 s | FlyDSL fp16: 1.37 s | 1.65x |
| Prefill attention, chunk at 30K | 963 ms | torch SDPA: 182 ms | 5.3x |
