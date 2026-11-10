# Own quantized math: Caminus's dequant and dot kernels, and the gate that replaces "bitwise llama.cpp"

Status: draft (2026-11-09). Owner: the K3 engine.

## Goal

Every quantized product K3 computes (the decode's matvecs, the MoE gate / up / down, the batched prefill's GEMMs) runs on
Caminus's own kernels, designed for RDNA3. llama.cpp stops being the source of the math and stays the yardstick.

Today the decode calls ggml-cuda's `vec_dot_<type>_q8_1` (vecdotq.cuh) for every GGUF format, quantizes activations with
llama.cpp's `quantize_q8_1` formula, and orders the BF16 matvec like `mul_mat_vec_f`. The prefill runs ggml-cuda's MMQ,
vendored. Correctness was defined as bitwise equality with llama.cpp, which made our own math impossible by definition.

End state, checkable: no K3 decode module and no prefill object includes a ggml-cuda header (`vecdotq.cuh`, `mmq.cuh`,
`common.cuh`) or compiles `kernels/ggml_mmq/`. `build.sh` no longer needs the ggml include path for them.

## What stays third-party

- The GGUF block layouts and the IQ codebooks and sign tables are the file format: the bytes on disk mean what ggml
  defines. Our kernels read them; the tables keep their MIT notice and a THIRD_PARTY.md entry.
- gguf-py's `dequantize` is the format oracle in the tests (it covers all 16 formats K3 meets: BF16, IQ1_M, IQ2_XXS,
  IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q2_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0).
- Kernels are written from the layouts, not by editing ggml's functions.

## Where the bytes are

Bytes read per decoded token, by (op, format), from the GGUF headers (routed experts at k / n_experts; token_embd and the
Coder's per-layer table are read on the host):

| Model | Largest pairs |
|---|---|
| Qwen3.8-Flash-Next Coder | MV Q6_K 37%, MV BF16 28%, MoE IQ4_NL 7%, MV Q5_K 7%, MoE IQ3_XXS / IQ2_S / IQ3_S 3-4% each |
| Qwen3-Coder-Next | MV Q6_K 75%, MoE IQ2_S 15%, MoE IQ3_S 10% |
| Qwen3.6 GSQ | MV Q8_0 87%, MoE Q2_0 12% |
| Qwen3.6 HauhauCS | MV Q6_K 24%, MV Q3_K 16%, MV Q4_K 16%, MoE Q5_K 12%, MV Q5_K 11%, MoE Q4_K 11%, MoE Q3_K 10% |
| AREX-2 | MV Q4_K 66%, MV Q6_K 25%, MV Q8_0 9% |

The dense matvecs in Q6_K, Q4_K and Q8_0 decide most of the decode time across the catalog; they go first.

## Measured starting point (CPU, real weights)

Mean (p99) of |y - y_ref| / rms(y_ref), y_ref the fp64 dot of the dequantized rows with the fp32 activation; 256 real rows
per tensor; activations: layer 0's real input and heavy-tailed synthetic vectors (recorded deep-layer activations replace
these in the gate).

| Tensor | Format | int8 per 32 (llama.cpp q8_1) | int8, fp32 scale and sum | fp16 activation | bf16 | fp32 |
|---|---|---|---|---|---|---|
| AREX-2 blk.0.ffn_down | Q6_K | 7.2e-3 (2.4e-2) | 7.2e-3 | 1.7e-4 (5.4e-4) | 1.4e-3 | 8.8e-8 |
| AREX-2 blk.3.attn_q | Q4_K | 1.6e-2 (5.9e-2) | 6.1e-3 | 1.6e-4 (5.4e-4) | 1.3e-3 | 8.3e-8 |
| AREX-2 blk.0.attn_qkv | Q8_0 | 4.9e-3 (2.3e-2) | 4.9e-3 | 1.3e-4 (5.0e-4) | 1.0e-3 | 6.7e-8 |
| Qwen3.6 GSQ output | Q8_0 | 6.4e-3 (2.6e-2) | 6.4e-3 | 1.7e-4 (5.6e-4) | 1.3e-3 | 7.8e-8 |
| Swift 1.5 blk.2.ffn_gate | IQ3_XXS | 5.9e-3 (2.3e-2) | 5.9e-3 | 1.7e-4 (5.5e-4) | 1.3e-3 | 8.3e-8 |

- The error of llama.cpp's matvec is the int8 activation; its summation order contributes about 1e-7.
- fp16 activations cut it about 40x on every format.
- Q4_K under q8_1 is 2.7x worse than int8 allows: vec_dot takes the min term from the activation sum stored as fp16.

## The gate

Applies to every change of quantized math. Pairs still on llama.cpp's math keep the bitwise gate until they move.

**G-acc, per op.** For each (op, format) pair of every catalog model: at least 256 real rows per tensor and at least 64
activation vectors recorded at that op site during a fixed decode (prompts in the gate tool), the activation exactly as
the op receives it. Mean and p99 of |y - y_ref| / rms(y_ref) must not exceed llama.cpp's on the same inputs, measured
with its own kernel on the GPU. Both numbers go in the pull request.

**G-model, end to end.** A reference mode of the engine runs every quantized op with fp32 activations (per-op error
about 1e-7 above). On the gate prompts: mean KLD of the candidate against the reference mode must not exceed llama.cpp's
KLD against the reference mode, and top-1 agreement with the reference mode must be at least llama.cpp's. KLD against
llama.cpp itself stays as a divergence bound that catches gross bugs; it is not an accuracy measure (llama.cpp's own
error counts as divergence there).

**G-self, bitwise against our own previous output.** T-token passes equal one-token steps; `K3_PER_WGP` 3 / 6 / 7 give
the same bits; where a prefill op is designed to match its decode op, every row matches bitwise. The gate tool stores
our outputs as the next reference; references are never regenerated to make a gate pass.

**G-speed.** Interleaved A/B in the real module on a real model's tapes, median of at least 3 runs, decode labelled.
Pre-registered: a difference under 0.05 ms per token is a tie, and an own kernel wins a tie. An own kernel that passes
G-acc but is slower is reported with its cost and ships as the default only when the maintainer accepts that cost in
the pull request. Per-op times from the trace module, against bytes over bandwidth, go in the pull request.

**G-safety (unchanged).** `k3_run` at most 96 VGPRs and no spills; first runs on a non-display card under the guard;
existing modules keep identical code (`.text` / `.rodata`) until a pair moves.

## Design space, decode matvec

- Activation precision. int8: exact integer sums, `v_dot4_i32_iu8`, accuracy at best tied with llama.cpp (fp32 scale and
  sum fix Q4_K / Q5_K's extra loss). fp16: `v_dot2_f32_f16`, about 40x more accurate, more VALU work per weight byte.
  fp32: the reference mode. Decode is bound by memory: the choice is whether the decode of each format's weights to
  fp16 fits in the VALU time one byte of bandwidth buys. Measured per format, not assumed.
- Summation order. Fixed per op and independent of T and of the grid (G-self): each lane sums its 32-value blocks in K
  order in fp32, then a fixed tree across lanes. Chosen so a prefill GEMM tile can reproduce it per row, letting the
  prefill match the decode bitwise later.
- Weight layout. GGUF blocks read in place first. Repacking at load into a lane-friendly layout (scales apart, 16-byte
  aligned per lane) is a later option, VRAM-neutral, paid in load time.
- Codebooks (IQ formats): in LDS or registers, measured per format.
- New pairs land as new modules. The persistent kernel shares one register allocation across all its ops, so a format's
  cost is measured in the module that will ship it.

## Order of work

1. This spec; the gate tool (G-acc recording and comparison, the reference mode for G-model).
2. GPU baseline on a non-display card: per-op time against bytes over bandwidth for the current llama.cpp-math modules.
3. First target: Qwen3-30B-A3B (Q4_K_M: Q4_K and Q6_K only), every step from the GGUF file to the output token on
   Caminus's own code, with the decode's matvecs at the bytes / bandwidth limit: the one-round-trip tiles (Q4_K in
   place, Q6_K in a plane layout made at load), the 8 experts of a layer grouped into one op by device ids, fp16
   activations. Then Swift 1.5's formats (IQ3_S, IQ4_XS, IQ3_XXS, Q4_K), then the other served models' pairs.
   int8 and fp16 activations were measured side by side first: fp16 is 40x more accurate at the same speed.
4. BF16 matvec in our own order.
5. Prefill GEMMs, designed to reproduce the decode's per-row order where the hardware allows.
6. Remove the ggml-cuda includes from the decode modules and prefill objects; update THIRD_PARTY.md.
