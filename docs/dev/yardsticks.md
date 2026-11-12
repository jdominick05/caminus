# Yardsticks: outside RDNA3 kernels at Caminus's shapes

Caminus's math is its own. Outside projects that ship gfx1100 kernels are measured here as yardsticks, at the shapes
Caminus actually runs, and read as sources of ideas or code to borrow (with their notices, THIRD_PARTY.md). None of
them is a dependency of the package. Rows here are not release rows: docs/dev/benchmarks.md holds what every release
reruns; a yardstick needs outside installs, so it is rerun when the kernel it measures changes.

The projects, with the versions measured:

| Project | Licence | What runs on gfx1100 | Measured at |
|---|---|---|---|
| FlyDSL (ROCm/FlyDSL) | Apache-2.0 | RDNA3 WMMA GEMMs: fp16 (`kernels/gemm/rdna3_f16_gemm.py`) and int8 -> int32 (`rdna3_int8_gemm.py`), autotuned tiles | wheel 0.5.0 + source tag v0.5.0 (226cde8) |
| rocKE (ROCm/rocm-libraries, hip-kernel-provider) | MIT | RDNA WMMA attention: split-KV GQA decode (`wmma_fmha_decode`), prefill (`wmma_fmha_fwd_v2`), gfx11-generic | rocm-libraries 9e91949 (editable install, its Python lowerer) |
| Composable Kernel (ROCm/rocm-libraries) | MIT | ck_tile FMHA, a gfx1100 batch-prefill pipeline policy | not yet measured |
| AITER (ROCm/aiter) | MIT | gfx1100 "Experimental": mostly Triton / FlyDSL kernels | not measured (Triton) |
| int8-aiter (a downstream AITER fork) | MIT | none: gfx908 (MI100) MFMA kernels; design reference only (int8 KV decode attention with per-token-head scales, Hadamard KV rotation, verify positions packed per KV pass) | - |

Every speed row states each side's harness, input format, KV type, whether the launch is included, and a bandwidth
floor (bytes / 952 GB/s, the XTX's measured read peak; the XT's memory is about 20 % slower, so x-floor figures
measured on the XT overstate the distance to that card's own floor by up to ~20 %).

## Decode and verify attention at depth (dense qwen35 geometry)

`benchmarks/attn_depth.py`, testing 737df92. One attention layer of Swift 1.5 / ThinkingCap / AREX-2 (24 query heads
over 4 kv heads, head dim 256): a K3 tape of only the attention op(s) on a zeroed KV cache of P positions, every arm on
the dense engine's own grid (`card_per_wgp`: 5 per WGP, 210 workgroups on the XT; 4, 192 on the XTX), median of 30
host-timed persistent launches (launch included; an empty tape is 11-13 us). x16 = the 16 attention blocks of a token.
Floor = the KV bytes read once for all T rows at 952 GB/s.

RX 7900 XT alone:

| P | own (k3_q35_swift), T 1 | split f16 (k3_gguf_q35d), T 1 | split q8_0, T 1 | own, T 4 | split f16, T 4 | split q8_0, T 4 | floor f16 / q8_0 |
|---|---|---|---|---|---|---|---|
| 1,024 | 130.9 us | 33.6 | 34.5 | 145.8 | 66.5 | 68.8 | 4.4 / 2.3 |
| 4,096 | 474.6 | 69.5 | 70.9 | 530.0 | 199.3 | 204.9 | 17.6 / 9.4 |
| 16,384 | 1,863.6 | 229.1 | 218.9 | 2,094.2 | 874.1 | 798.0 | 70.5 / 37.4 |
| 32,768 | 4,353.4 | 517.3 | 496.8 | 4,454.2 | 1,762.0 | 1,786.9 | 141.0 / 74.9 |
| 49,152 | 6,517.0 | 772.8 | 758.0 | 6,658.9 | 2,625.4 | 2,698.2 | 211.5 / 112.3 |
| x16 at 32K | **69.7 ms** | **8.3 ms** | 7.9 ms | 71.3 ms | 28.2 ms | 28.6 ms | 2.3 / 1.2 ms |

RX 7900 XTX (the card the Swift decode numbers come from), own op, T 1: 111.6 / 400.9 / 1,795 / 4,016 / 5,891 us at
1K / 4K / 16K / 32K / 49K: **64.3 ms of a token at 32K**, 94.3 ms at 49K.

What it says:
- Swift's and ThinkingCap's own modules run decode attention as one workgroup per query head walking every position
  (`own_op_attn_dec`): 23-31x the bandwidth floor at every depth. On the XTX that is 64.3 ms of a token at 32K; Swift's
  measured decode step there is 84.4 ms against 24 ms at short context, so this one op is nearly all of the depth
  cost (same card).
- Our own split pair (flash decoding, `kernels/k3_attn_split.h`, already in the AREX-2 / Coder / Qwen3.6 modules) runs
  the same geometry on the same grid at 3.3-3.9x the floor from 4K up: 8.4x faster at 32K. Generalizing it into the Swift /
  ThinkingCap modules is a step of the dense-27B plan that had not been built; it is the next change (its own pull
  request, accuracy-gated: the split pair is not bitwise the one-head op).
- A q8_0 cache halves the bytes but saves only 2-5 % here (the split pair is not at the floor); an int8 KV pays only
  once the kernel is near the bandwidth floor.
- T = 4 on the split pair costs 3.4x T = 1 at depth: each verify row reads the cache again. Packing the T rows into one
  KV pass (int8-aiter's verify packing; rocKE's GQA-packed decode) is the lead for verify passes at depth.
- At 8 workgroups per WGP (the GGUF modules' resident grid, 672 workgroups) the split pair was 412 us at 32K instead of
  517: its slices want more workgroups than the dense engine's grid; worth a look in the fix.
- rocKE's split-KV, GQA-packed decode on the same geometry (`benchmarks/rocke_attn.py`, its own harness: HIP graph
  replay, launch not counted; output checked against its float64 reference at every row):

  | P | rocKE T 1 | rocKE T 4 (bottom-right causal) | ours split T 1 / T 4 | ours own op T 1 |
  |---|---|---|---|---|
  | 1,024 | 21.1 us | 21.0 | 33.6 / 66.5 | 130.9 |
  | 4,096 | 26.7 | 36.7 | 69.5 / 199.3 | 474.6 |
  | 16,384 | 62.2 | 114.5 | 229.1 / 874.1 | 1,863.6 |
  | 32,768 | 199.1 | 224.8 | 517.3 / 1,762.0 | 4,353.4 |
  | 49,152 | 291.4 | 327.4 | 772.8 / 2,625.4 | 6,517.0 |

  At 32K and 49K (the cache is larger than the XT's 80 MB Infinity Cache) rocKE reads ~0.67 TB/s, about 84 % of the
  XT's memory bandwidth; at 16K and below the whole cache stays in the Infinity Cache between repeated launches in both
  harnesses, so those rows are cache-warm. rocKE's T 4 costs 1.1x its T 1: its rows share each K / V tile (GQA packing
  of query heads and rows); our split pair's T 4 costs 3.4x. That is the design to take into our split pair after the
  generalization: 2.6x at 32K for T 1, 7.8x for T 4.

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
- FlyDSL's tile (4 waves, double-buffered LDS ping-pong, 128 x 128 x 32 default tile, swizzled grid, shape-autotuned
  tile ladder; Apache-2.0) is the bar. An earlier dense bf16 WMMA GEMM of ours (the training stack, not in this tree)
  already reached 70-90 TF on the XTX with the same kind of pipeline, so the prefill tiles start from our own and keep
  their in-tile dequantization. Upper bound if they reached FlyDSL's fp16 rate: a chunk's GEMMs 2.26 -> ~1.4 s.
- GEMMs are most of a prefill chunk: 2.26 s of GEMMs per 2048 tokens on the XT.
- FlyDSL's fp16 output matches torch's on the same operands to <= 3.6e-4 relative (every shape above, checked at M 2048).
- The per-chunk totals assume blk.0's formats for all 48 GDN blocks and blk.3's for all 16 attention blocks.

## Prefill attention at depth (dense qwen35 geometry)

`benchmarks/q35_prefill_attn.py`, RX 7900 XT alone. One 2048-row chunk at positions P0 .. P0 + 2047 attends to the
f16 cache rows up to its own (24 query heads over 4 kv heads, head dim 256). Ours, now: `q35p_attn_wm`, WMMA tiles (a
workgroup = 8 positions x one kv head's 6 query heads, K / V staged once per 32 positions for its 8 waves, a wave pair
per 16 query rows over the 256 dims, fp16 q / k / v / p, fp32 scores and sums), in launches of at most 2^25
row-positions, as the prefill driver runs it. Before (testing 737df92): `q35p_attn16`, one workgroup per (row, query
head) walking every position up to its own. Yardstick: torch's `scaled_dot_product_attention` (fp16, GQA) on its tiled
backends only (flash / memory-efficient). A causal-with-offset mask has no tiled kernel here (with the mask PyTorch falls
back to the math path, which materializes the scores: 17.5 GB at 32K), so the chunk runs as its two tiled calls, the
prefix (no mask) and the causal square, their times summed (the log-sum-exp merge not counted). TFLOPS count the causal
half once. Median of 5.

| P0 | ours, tiles | ours before, per (row, head) | torch SDPA, tiled | tiles / SDPA |
|---|---|---|---|---|
| 0 | 2.85 ms (18.1 TF) | 25.8 ms (2.0 TF) | 2.36 ms (21.8 TF) | 1.21 (outputs agree to 6.8e-4) |
| 8,192 | 23.2 ms (20.0 TF) | 243.0 ms (1.9 TF) | 12.6 ms (36.8 TF) | 1.84 |
| 30,720 | 79.7 ms (20.0 TF) | 963.6 ms (1.7 TF) | 40.6 ms (39.4 TF) | 1.96 |

What it says:
- The per-(row, head) kernel re-read K and V for every row and head: ~2 TF at every depth, ~31 ms per 1K of depth per
  chunk (0.96 s at 30K against 2.26 s in all the chunk's GEMMs). The tiles run at ~20 TF from 8K up: 9-12x faster, the
  chunk at 30K in 80 ms. Accuracy: within 9.4e-4 of rms of float64 at every depth tested; the same bits at every launch
  split and chunk geometry (tests/prefill/test_q35_prefill_attn.py).
- End to end (tests/prefill/test_q35_prefill.py --speed 16384, Swift 1.5 Q4_K_M, XT alone, chunk 2048, median of 3):
  16,384 positions from empty in 44.3 s (370 tok/s) before (testing 352918c), 20.6 s (794 tok/s) with the tiles: 2.15x.
  What remains is the GEMMs (~2.26 s a chunk) and the GDN; attention is now ~13 % of a 16K prefill.
- rocKE's RDNA WMMA prefill (`wmma_fmha_fwd_v2`, d256 causal 64 x 32 tile; `benchmarks/rocke_attn.py`, graph replay,
  checked at P0 0) does the same chunk in 0.94 / 8.28 / 28.5 ms at P0 0 / 8K / 30K: 54.8-56.0 TF, 1.4x tiled SDPA and
  2.8x our tiles at 30K (34x the per-(row, head) kernel).
- What the tiles leave on the table, against rocKE's design: one 16-column WMMA tile holds 2 positions x 6 heads (12 of
  16 columns used: GQA packing by absolute position pairs, which keeps the chunk geometry bit-exact); V^T is staged by
  scalar LDS writes; one LDS buffer (two barriers per 32 positions) where rocKE double-buffers and prefetches the next
  tile into registers; scores exchanged through LDS between the pair. CK's gfx1100 FMHA is not measured.

## Summary: where the outside kernels say we stand (XT)

| Op | ours | best yardstick measured | gap |
|---|---|---|---|
| Decode attention, Swift / ThinkingCap modules, 32K (XT, engine grid) | 4.35 ms / layer | our own split pair: 0.52 ms | 8.4x (own fix, not outside code) |
| Decode attention after the fix, split pair, 32K | 0.52 ms / layer | rocKE split-KV decode: 0.20 ms | 2.6x |
| Verify attention (T 4), split pair, 32K | 1.29 ms / layer with the packed PART (1.52 ms before; T 1: 0.48 ms) | rocKE, GQA-packed: 0.22 ms | 5.9x (each row's arithmetic is still its own: decode's bits) |
| Prefill GEMMs, Swift chunk of 2048 | 2.26 s | FlyDSL fp16: 1.37 s | 1.65x |
| Prefill attention, chunk at 30K | 79.7 ms (964 ms before the tiles) | rocKE WMMA prefill: 28.5 ms (torch SDPA tiled: 40.6 ms) | 2.8x (34x before) |
