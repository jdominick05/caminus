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

| P | own (dec_q35_swift), T 1 | split f16 (dec_q35_arex), T 1 | split q8_0, T 1 | own, T 4 | split f16, T 4 | split q8_0, T 4 | floor f16 / q8_0 |
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
- Our own split pair (flash decoding, `kernels/ops_attn_split.h`, already in the AREX-2 / Coder / Qwen3.6 modules) runs
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
- ours reads the GGUF blocks and dequantizes inside the WMMA GEMM (`kernels/pf_q35_swift.hip`), eager launches;
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

### Where the tile's time goes (ablation builds)

Measurement builds of `kernels/pf_q35_swift.hip` with `-DQ3P_ABL=N` (not built by build.sh since the study ended; wrong
outputs by design) remove one part of the tiles each (kernels/pf_q3m.hip `Q3P_ABL`): 1 the weight loads and decode (a
constant B), 2 the X staging, 3 the per-block / per-sub-block scale step. `benchmarks/q35_prefill_gemm.py --module OBJECT`
on such a build, Swift's shapes, M 2048, XT:

| removed | chunk of 2048 | Q8_0 attn_qkv | Q6_K ffn_up | Q4_K ffn_gate |
|---|---|---|---|---|
| nothing | 2,268 ms | 37.5 TF | 44.5 TF | 50.1 TF |
| weight loads + decode | 1,907 ms | 54.8 | 52.1 | 51.9 |
| X staging | 2,024 ms | 42.3 | 49.2 | 56.6 |
| scale step | 2,004 ms | 41.5 | 53.2 | 54.3 |

Q8_0's weight path was its largest cost: its blocks' loads waited right before their WMMAs. Now prefetched a block pair
ahead as 17 aligned dword loads (bitwise the previous kernel on every launch of tests/prefill/test_q35_prefill_mm.py
--ref-module): attn_qkv 37.6 -> 49.1 TF, ssm_out / attn_output 38.3 -> 48.1, attn_k 30.6 -> 41.2; the chunk 2,266 ->
2,116 ms (XT, interleaved twice). With every part removed in turn the tile still stops near 52-57 TF: four WMMAs per A
load from LDS and one LDS buffer with two barriers per 256 values; FlyDSL's tile (16 WMMAs per 8 LDS loads, double
buffered) is the bar there.

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
  tile into registers; scores exchanged through LDS between the pair. CK's gfx1100 FMHA is not measured. Qwen3.6-35B-A3B's
  `q35mp_attn` (next section) takes the pipeline items; run at this geometry from its object it is 1.39 / 11.9 / 41.1 ms
  (37-39 TF, 2.0x these tiles; the 12-of-16 columns stay). The dense line keeps `q35p_attn_wm` until its own gates rerun.

## Prefill attention at depth (Qwen3.6-35B-A3B and Qwen3-30B-A3B geometry)

`benchmarks/q35_prefill_attn.py` (`--heads 16 --kv-heads 2 --module pf_q35m`, `--kv q8`; `--hd 128 --heads 32
--kv-heads 4` for the 30B) and `benchmarks/rocke_attn.py --heads 16 --kv-heads 2 --prefill-only` (`--hd 128 --heads 32
--kv-heads 4`), RX 7900 XT alone, one 2048-row chunk, the tables above's conventions. The 35B: 16 query heads over 2 kv
heads, head dim 256, 8 query heads per kv head (all 16 WMMA columns used); the 30B: 32 over 4, head dim 128 (`q3mp_attn`;
`tests/prefill/test_q35_prefill_attn.py --kernel q3m,q3m-q8 --speed`, interleaved against the old `q3p_attn`).

| P0 | 35B before: `q35p_attn_wm` (F16 / Q8) | 35B now: `q35mp_attn` (F16 / Q8) | rocKE `wmma_fmha_fwd_v2` d256 | torch SDPA tiled |
|---|---|---|---|---|
| 0 | 1.49 / 1.57 ms (23.0 / 21.9 TF) | 0.80 / 0.78 ms (43.2 / 44.1 TF) | 0.64 ms (53.6 TF) | 1.63 ms (21.0 TF) |
| 8,192 | 11.95 / 12.45 ms (25.9 / 24.8 TF) | 6.27 / 6.39 ms (49.4 / 48.4 TF) | 5.68 ms (54.4 TF) | 9.07 ms (34.1 TF) |
| 30,720 | 41.30 / 43.00 ms (25.8 / 24.8 TF) | 20.95 / 22.10 ms (50.8 / 48.2 TF) | 19.60 ms (54.3 TF) | 29.08 ms (36.6 TF) |

| P0 | 30B before: `q3p_attn` (F16 / Q8, 1 << 24 launches) | 30B now: `q3mp_attn` (F16 / Q8, one launch) | rocKE d128 (m128 n32) | torch SDPA tiled |
|---|---|---|---|---|
| 0 | 0.96 / 0.95 ms (35.8 / 36.1 TF) | 0.64 / 0.67 ms (53.5 / 51.4 TF) | 0.63 ms (54.9 TF) | 1.16 ms |
| 8,192 | 6.98 / 7.19 ms (44.3 / 43.0 TF, 2 launches) | 5.11 / 5.33 ms (60.6 / 58.0 TF) | 5.41 ms (57.1 TF) | 6.46 ms |
| 30,720 | 30.02 / 30.24 ms (35.5 / 35.2 TF, 4 launches) | 17.38 / 18.18 ms (61.3 / 58.6 TF) | 18.62 ms (57.2 TF) | 20.82 ms |

What it says:
- `q35mp_attn` (kernels/pf_q35m.hip) is `q35p_attn_wm`'s absolute-position tiles with rocKE's pipeline: the next
  tile's K / V in registers during the current one, two LDS-only barriers per tile (`__syncthreads` would wait for that
  prefetch), V^T staged by 4-position runs, each wave reading only its partner's score partial, per-column lazy rescale.
  1.9-2.0x the old kernel and 0.93x rocKE at 30K (F16), the same bits at every chunk geometry and launch split; 230 / 233
  VGPRs (F16 / Q8), 0 spills, 53,760 B of LDS. Q8 decodes codes with packed fp16 math (the same bits as fp32
  conversions; 44.8 -> 48.2 TF at 30K, interleaved).
- The 35B end to end (prefill from empty, chunk 2048, median of 3, interleaved against the old attention):
  F16 4,096 positions 787 -> 746 ms (5,203 -> 5,488 tok/s, 1.05x), 16,384 3.73 -> 3.30 s (1.13x), 32,768 9.03 -> 7.46 s
  (3,630 -> 4,393 tok/s, 1.21x); Q8 4,096 811 -> 768 ms (1.06x), 16,384 3.85 -> 3.37 s (1.14x), 32,768 9.32 -> 7.58 s
  (3,516 -> 4,322 tok/s, 1.23x). At 32K attention was ~37 % of the prefill, now ~23 %.
- `q3mp_attn` (kernels/pf_q3m.hip) is `q35mp_attn`'s source (`kernels/pf_attn_g8.h`) at head dim 128: each
  wave holds its own query pair (no partner exchange), 16 positions a workgroup, 64-position tiles (the same 4 + 4
  staging chunks a thread as the 35B's 32 x 256); 227 / 239 VGPRs (F16 / Q8), 0 spills, 35,840 B of LDS (three
  workgroups per WGP by both). The old kernel run in one launch is 23.1 ms at 30K: the rest of the gain is the launch
  split (`q3m.ATTN_TP` 1 << 24 -> 1 << 27, one launch per chunk up to 64K positions): 4 launches of 128 workgroups each
  left a second, nearly empty round on the XT's 126 resident slots (the new kernel 22.6 ms at 4 launches, 19.0 at 2, 17.3
  at 1). The slowest launch: 17.4 ms at 30K; 40.7 ms at 131,072 (Q8, two launches of 64 blocks).
- The 30B end to end (prefill from empty, chunk 2048, median of 3, interleaved against the old attention at its old
  split, XT): F16 4,096 839 -> 792 ms (4,882 -> 5,175 tok/s, 1.06x), 12,288 3.40 -> 3.01 s (3,618 -> 4,089 tok/s,
  1.13x); Q8 4,096 850 -> 805 ms (1.06x), 12,288 3.45 -> 3.07 s (1.12x), 24,576 9.76 -> 8.10 s (2,518 -> 3,034 tok/s,
  1.21x). At 12K (F16, `benchmarks/q3moe_prefill.py --ops`) attention is now 31.8 % of the prefill.

## Decode attention as its own kernel (qwen35moe geometry)

`kernels/dec_q35m_attn.hip` (`engine/dec_q35m_attn.py`) is Qwen3.6-35B-A3B's T 1 decode attention (16 query heads over
2 kv heads, head dim 256, F16 or Q8_0 KV) launched on its own, outside k3_run's register budget, on rocKE's gfx11
split-KV decode design (THIRD_PARTY.md): workgroup (split, kv head, half of the output dims), 4 waves, each split taking
every ns-th run of 4 16-key tiles; scores S^T = K Q^T and O^T = V^T P on WMMA, Q and P as fp16 hi + lo halves in the 8
unused columns (fp32-grade sums from fp16 operands); Q8_0 rows decoded to fp16(code x d) in registers; the merge done by
the last workgroup of each group (an atomic counter, re-armed by that workgroup). Splits: `splits(cells)`, one per 512
keys up to 3 workgroups per WGP over the 4 groups (at most 64).

- Accuracy (`tests/engine/test_dec_q35m_attn.py`): against float64 over the values it reads within 1e-5 x max |V| (F16
  and Q8), against the split pair within 2e-5 (F16) or the Q8 fp16-rounding bound the test prints per case; rows past
  the last cell NaN-poisoned; an exactly sized cache (its own allocation) read to its last byte.
- Speed: `benchmarks/attn_bench.py --new [--kv q8] [--cold]`, the kernel and the split pair (`OP_ATTN_PART` +
  `OP_ATTN_COMB` on `dec_q35m` / `dec_q35m_kvq8` at the engine's grid) interleaved, a queued batch of 8 launches per
  round. rocKE's split-KV decode measured on the XT at this geometry: 21.6 us (4K), 61.9 us (32K), 385 us (128K).

  XT, us per layer, batch 8, median of 15 rounds, the kernel and the pair interleaved; warm = the same rows each launch
  (as rocKE's graph replay), cold = rotating copies so each launch's rows are not in the Infinity Cache. Splits 8 at 4K,
  31 at 32K and 128K (the XT's 42 WGPs x 3 / 4); untuned:

  | KV | depth | new warm | pair warm | pair / new | new cold | pair cold | pair / new | rocKE (warm) |
  |---|---|---|---|---|---|---|---|---|
  | F16 | 4K | 26.5 | 40.3 | 1.52x | 32.2 | 44.9 | 1.40x | 21.6 |
  | F16 | 32K | 81.9 (819 GB/s) | 153.8 | 1.88x | 119.1 | 163.7 | 1.37x | 61.9 |
  | F16 | 128K | 389.0 (690 GB/s) | 571.2 | 1.47x | 389.5 | 570.8 | 1.47x | 385 |
  | Q8 | 4K | 26.2 | 40.2 | 1.54x | 32.2 | 45.9 | 1.42x | |
  | Q8 | 32K | 84.1 (424 GB/s) | 159.8 | 1.90x | 90.7 | 162.1 | 1.79x | |
  | Q8 | 128K | 270.6 (527 GB/s) | 584.9 | 2.16x | 271.0 | 589.9 | 2.18x | |

  Not tuned yet: the Q staging loop, no K / V prefetch (V is loaded after the scores), few splits at 4K. Q8 reads 0.42-0.53 TB/s at
  depth against F16's 0.69-0.82: its in-register decode is the likely limit (not profiled).

- In the engine (`Moe35Own`, `benchmarks/q3moe_depth.py --engine q35moe`, XT, ms per token, interleaved, median of 3): the
  tape in slices around the kernel pays for its launch boundaries. A bare boundary is ~5 us; the rest is the work one
  launch overlaps. At first it paid about 0.7 ms a token (~35 us per k3_run boundary): each attention block's prefetch
  segment (OP_PREFETCH, 16 tasks streaming weights into the Infinity Cache) ran in front of the cut, so the trunk's
  launch could not end before it had, and the attention waited (the gap after a trunk launch 19 us, after a no-op one 5).
  With that segment behind the block's pair (the same bits) the sliced pass pays 0.26 ms at depth 0 and wins from
  24,576 (F16) / 20,480 (Q8) on: 131,072 8.83 against 10.83 F16, 7.74 against 10.76 Q8. The switch and its numbers:
  `q35_moe.ATTN_FROM`.

## Prefill GEMMs (Qwen3-30B-A3B and Qwen3.6-35B-A3B, grouped experts)

`benchmarks/q35_prefill_gemm.py GGUF --engine --tokens REF.npz --M 2048`, RX 7900 XT alone. It records every GEMM of one
real 2048-token chunk of the engine's own prefill: the 30B on `ref32_16384.npz`'s text, the 35B on the same text
re-encoded (`--tokens-from`). It then times each group of launches by format, shape and role, replaying the recorded
launches from a HIP graph.

The arms:
- ours (the grouped expert GEMMs at their real routing);
- torch fp16 and FlyDSL 0.5.0 fp16 / int8 on dense operands at the group's total M (as if grouping were free);
- per expert (`/exp`): torch and FlyDSL fp16 once per routed expert at its own M, from the middle layer's routing,
  padded to 32 for FlyDSL.

Routing at 2048 tokens, top 8 (16,384 pairs a layer):

| | experts | pairs per expert (median / p90 / max) | experts unused | work items a layer | fill |
|---|---|---|---|---|---|
| 30B | 128 | 61 / 291 / 1,814 | 11.9 % | 324 | 79 % |
| 35B | 256 | 23 / 156 / 1,378 | 10.7 % | 404 | 63 % |

The 30B before this change, ms per chunk (TF):

| Group (R x K) | launches | ours | torch | FlyDSL fp16 | FlyDSL int8 | torch / exp | FlyDSL / exp |
|---|---|---|---|---|---|---|---|
| attn_q Q4_K 4096 x 2048 | 48 | 34.98 (47.1) | 33.66 | 24.07 (68.5) | 22.72 | | |
| attn_k Q4_K 512 x 2048 | 72 | 9.43 (32.8) | 9.81 | 6.50 (47.6) | 5.96 | | |
| attn_v Q6_K 512 x 2048 | 24 | 3.60 (28.6) | 3.33 | 2.16 (47.7) | 1.98 | | |
| attn_output Q4_K 2048 x 4096 | 47 | 35.58 (45.4) | 41.91 | 25.25 (63.9) | 22.75 | | |
| gate + up experts Q4_K 768 x 2048 | 94 | 129.48 (37.4) | 100.43 | 68.69 (70.5) | 65.54 | 246.37 | 244.27 |
| down experts Q6_K 2048 x 768 | 23 | 35.91 (33.0) | 18.81 | 16.69 (71.0) | 16.15 | 43.46 | 45.98 |
| down experts Q4_K 2048 x 768 | 24 | 34.05 (36.3) | 19.64 | 17.42 (71.0) | 16.85 | 45.29 | 47.97 |
| **all GEMMs of a chunk** | 332 | **283.0** | 227.6 | 160.8 | 151.9 | 335.1 | 338.2 |

The 35B, ms per chunk:

| Group | launches | ours | torch | FlyDSL fp16 | FlyDSL int8 |
|---|---|---|---|---|---|
| Q8_0 8192 x 2048 | 39 | 51.31 (52.2) | 46.75 | 36.80 (72.8) | 35.49 |
| Q8_0 4096 x 2048 | 30 | 20.80 | 20.71 | 15.06 | 14.23 |
| Q8_0 2048 x 4096 | 39 | 28.82 | 34.89 | 21.09 | 18.88 |
| Q8_0 512 x 2048 | 98 | 13.10 | 13.27 | 8.88 | 8.11 |
| Q8_0 2048 x 512 | 39 | 4.41 | 5.97 | 3.20 | 2.89 |
| gate / up experts Q2_0 512 x 2048 | 78 | 86.74 (30.9) | 58.27 | 39.27 (68.3) | 37.00 |
| down experts Q2_0 2048 x 512 | 39 | 45.24 (29.6) | 26.72 | 19.47 (68.8) | 18.45 |
| **all GEMMs of a chunk** | 362 | **250.4** | 206.6 | 143.8 | 135.1 |

Per expert, the 35B's experts take 367.7 ms (torch) and 396.5 ms (FlyDSL).

What it says:
- Our grouped expert GEMMs already beat a dense GEMM run once per expert, torch's or FlyDSL's (30B 283 vs 335 ms, 35B
  250 vs 368 ms). Most experts get 20-60 tokens, too few for a dense tile.
- The bar is FlyDSL at the aggregate M: 1.76x ours on the 30B and 1.74x on the 35B. The experts are where ours trails
  most: 1.9-2.2x on the 30B's down GEMMs, 2.2-2.3x on the 35B's Q2_0.

The 30B's wide tiles (`kernels/pf_q3m_mm.hip`). Every step was measured interleaved against the one before, every
output bitwise:

| Step | GEMMs per chunk |
|---|---|
| narrow tiles; gate, up, swiglu apart (the swiglu launch counted) | 292.1 ms |
| the dense line's wide tile (two 16-row tiles per wave, 256 rows a workgroup) on every Q4_K / Q6_K GEMM, dense and grouped; the experts' gate and up in one launch with the swiglu epilogue (`q3p_mm_q4k_gu`) | 265.4 ms |
| a partial item's empty 16-token tiles skip their WMMAs | 256.5 ms |
| Q4_K weights one chunk ahead in a two-chunk register ring (48 registers where the whole block took 72) | 253.6 ms |
| the three kernels in their own object under LLVM's max-ilp scheduler (`pf_q3m` keeps its code: its `q3p_mm_q4k` and attention spill under it) | **243.4 ms** |

Registers (VGPRs, no spills; LDS): `q3p_mm_q4k_w` and `q3p_mm_q4k_gu` 239, 36,100 B; `q3p_mm_q6k_w` 253, 33,792 B (the
narrow `q3p_mm_q4k` 191, `q3p_mm_q6k` 154).

Tried and dropped:
- weight prefetch of the next block's half (13-15 spills);
- an A-operand ping-pong across token tiles and k-steps (48 spills by default, 51 under max-ilp);
- max-ilp on the whole `pf_q3m` object (`q3p_mm_q4k` 57 spills, `q3mp_attn` 198);
- the max-memory-clause scheduler (no change);
- the 30B's body on the dense line's objects (Swift's GEMMs 0.8 % slower: the dense line keeps its own tiles).

After, per group: attn_q 32.91 ms (50.1 TF), attn_k 8.69, attn_v 3.43, attn_output 34.30, gate | up with swiglu 101.30
(47.8 TF; 139.00 before), down Q6_K 34.03, down Q4_K 28.75 (43.0 TF).

Still behind FlyDSL fp16 at the aggregate M: 1.51x per chunk.

| Group | how far behind |
|---|---|
| attn_q, attn_k, attn_output | 1.34-1.37x |
| attn_v | 1.59x |
| gate \| up | 1.47x (with its swiglu) |
| down Q4_K | 1.65x |
| down Q6_K | 2.04x |

End to end (`benchmarks/q3moe_prefill.py --chunk 2048 --ab`: prefill from empty, wide and narrow interleaved, median of 3,
the last layer's KV rows bitwise equal):

| Length | KV | wide | narrow | speed-up |
|---|---|---|---|---|
| 4,096 | F16 | 5,893 tok/s | 5,182 tok/s | 1.14x |
| 12,288 | F16 | 4,517 tok/s | 4,090 tok/s | 1.10x |
| 24,576 | Q8 | 3,259 tok/s | 3,031 tok/s | 1.08x |

Short prompts, the same way:

| Length | KV | wide | narrow | speed-up |
|---|---|---|---|---|
| 128 | F16 | 2,480 tok/s | 1,719 tok/s | 1.44x |
| 512 | F16 | 5,009 tok/s | 3,876 tok/s | 1.29x |
| 1,024 | F16 | 5,846 tok/s | 4,914 tok/s | 1.19x |

A dense GEMM goes wide only from half the card's compute units in workgroups (`q3m_ops.Ops.wide_min_wg`; 42 on the
XT). At 512 tokens the k / v projections have 16 wide workgroups and ran 1.6-1.8x slower wide (attn_k 4.72 vs 3.01 ms
a chunk); attn_output's 64 ran faster (10.58 vs 11.51). With the rule, a 512-token chunk's GEMMs take 82.0 ms against
110.8 narrow.

At 12K F16 (`--ops`), attention is 35.1 % of the prefill and the fused gate | up 22.0 %.

The 35B's wide tiles (`kernels/pf_q35m_mm.hip`, the 30B's skeleton from `kernels/pf_mm_wide.h` with the Q2_0
statements of `q35mp_mm_q2_0` per row). Each step was measured interleaved against the one before, every output bitwise;
the times are the routed experts' GEMMs (gate | up and down) of one 2048-token chunk on the XT:

| Step | expert GEMMs per chunk |
|---|---|
| narrow tiles; gate, up, swiglu apart (the swiglu launch counted) | 136.8 ms |
| wide tiles + fused gate \| up, every token tile's WMMAs (56 VGPRs spilled) | 141.9 ms |
| a partial item's empty 16-token tiles skip their WMMAs (no spills) | 107.6 ms |
| the weights half a step ahead in a two-half register ring (36 registers where a step and the next took 72) | 101.9 ms |
| its own object under max-ilp | 101.6 ms |
| the Q2_0 B operand by v_perm, a mask and one packed fma (about half the VALU instructions) | **91.3 ms** |

- Fusing gate | up is worth 8.5 ms of it (the final tile with `gu` off: 66.8 against 58.3 ms for gate and up).
- The final tile under the default scheduler: 92.3 ms (max-ilp 91.6 in the same run; 36.1 against 35.7 at 512 tokens).
- Registers: `q35mp_mm_q2_0_w` and `q35mp_mm_q2_0_gu` 211 VGPRs, 0 spills, 33,792 B of LDS (the narrow tile: 256 under max-ilp, 44 spilled).
- The dense Q8_0 GEMMs go wide from half the card's compute units in workgroups, the 30B's rule (`q35moe.Ops.q8_wide_min_wg`).
  At 2048 tokens that moves the shared expert's gate / up and the k / v projections (512 x 2048): 12.97 -> 11.84 ms (Q8_0-only runs; 13.22 -> 12.11 in the full run below).
  At 512 tokens it moves ssm_out (9.91 -> 8.92) and the shared expert's down (1.47 -> 1.39) and leaves the 512 x 2048
  GEMMs narrow (16 workgroups: 4.23 narrow, 6.35 wide). The dense Q8_0 tile under max-ilp spills 18 VGPRs, so it stays
  in `pf_q35m`.

The 35B after, ms per chunk (one run on the final build: before = `Ops(wide=False)`, the narrow tiles and the dense line's
rule, interleaved with the wide tiles, every output bitwise):

| Group (R x K) | before | after | FlyDSL fp16 | after / FlyDSL |
|---|---|---|---|---|
| Q8_0 8192 x 2048 | 51.22 | 51.22 | 36.78 | 1.39x |
| Q8_0 4096 x 2048 | 20.81 | 20.81 | 15.03 | 1.38x |
| Q8_0 2048 x 4096 | 28.83 | 28.83 | 21.03 | 1.37x |
| Q8_0 512 x 2048 | 13.22 | 12.11 | 9.76 | 1.24x |
| Q8_0 2048 x 512 | 4.23 | 4.23 | 3.19 | 1.33x |
| gate \| up experts Q2_0 512 x 2048 (before: two GEMMs + swiglu) | 91.54 | 58.17 (46.1 TF) | 39.21 (no swiglu) | 1.48x |
| down experts Q2_0 2048 x 512 | 45.21 | 33.01 (40.6 TF) | 19.53 | 1.69x |
| **per chunk** | **255.1** | **208.4** | **144.5** | **1.44x** |

End to end, the 35B (`benchmarks/q3moe_prefill.py GGUF35 --chunk 2048 --ab`: prefill from empty, the wide arm against
`Ops(wide=False)` interleaved, median of 3, the last attention layer's KV rows bitwise equal at every length):

| Length | KV | after | before | speed-up |
|---|---|---|---|---|
| 128 | F16 | 2,767 tok/s | 1,858 tok/s | 1.49x |
| 512 | F16 | 4,984 tok/s | 3,779 tok/s | 1.32x |
| 4,096 | F16 | 6,235 tok/s | 5,461 tok/s | 1.14x |
| 16,384 | F16 | 5,552 tok/s | 4,936 tok/s | 1.12x |
| 16,384 | Q8 | 5,423 tok/s | 4,833 tok/s | 1.12x |
| 32,768 | Q8 | 4,764 tok/s | 4,301 tok/s | 1.11x |

## Summary: where the outside kernels say we stand (XT)

| Op | ours | best yardstick measured | gap |
|---|---|---|---|
| Decode attention, Swift / ThinkingCap modules, 32K (XT, engine grid) | 4.35 ms / layer | our own split pair: 0.52 ms | 8.4x (own fix, not outside code) |
| Decode attention after the fix, split pair, 32K | 0.52 ms / layer | rocKE split-KV decode: 0.20 ms | 2.6x |
| Decode attention, qwen35moe, own kernel (`dec_q35m_attn`), 32K F16 warm | 81.9 us / layer (split pair: 153.8 us) | rocKE split-KV decode: 61.9 us | 1.32x (128K: 389.0 vs 385 us, 1.01x) |
| Verify attention (T 4), split pair, 32K | 1.29 ms / layer with the packed PART (1.52 ms before; T 1: 0.48 ms) | rocKE, GQA-packed: 0.22 ms | 5.9x (each row's arithmetic is still its own: decode's bits) |
| Prefill GEMMs, Swift chunk of 2048 | 2.26 s | FlyDSL fp16: 1.37 s | 1.65x |
| Prefill GEMMs, 30B chunk of 2048 (grouped experts) | 243.4 ms, swiglu inside (283.0 ms before, 292.1 with its swiglu launch) | FlyDSL fp16 at the aggregate M: 160.8 ms | 1.51x (1.76x before) |
| Prefill GEMMs, 35B chunk of 2048 (grouped experts) | 208.4 ms, swiglu inside (250.4 ms before, 255.1 with its swiglu launch) | FlyDSL fp16 at the aggregate M: 144.5 ms | 1.44x (1.74x before) |
| Prefill attention, chunk at 30K | 79.7 ms (964 ms before the tiles) | rocKE WMMA prefill: 28.5 ms (torch SDPA tiled: 40.6 ms) | 2.8x (34x before) |
| Prefill attention, 35B chunk at 30K | 20.95 ms (`q35mp_attn`; 41.3 ms before) | rocKE WMMA prefill: 19.60 ms | 1.07x (2.1x before) |
| Prefill attention, 30B chunk at 30K | 17.38 ms (`q3mp_attn`; 30.0 ms before) | rocKE d128: 18.62 ms | 0.93x (1.6x before) |
