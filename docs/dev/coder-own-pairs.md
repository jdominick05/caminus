# The Coder's (op, format) pairs on own math: state (issue #87)

Branch state: the dense Q4_K, Q5_K and IQ4_XS matvecs of the decode are moved to own tiles. Not yet a pull request:
G-speed (the interleaved A/B in the real module) has not run.

## Gates run on this branch (two cards, XTX + XT)

- G-acc (`tests/engine/test_k3_gguf_gsq.py` on the XT, every row of 11 real matrices with the TP splits, Gaussian and
  Student-t activations): own mean 1.6e-4, p99 5.5e-4; llama.cpp mean 4.3e-3 to 6.9e-3, p99 1.5e-2 to 2.4e-2. 930/930
  checks pass.
- G-self: every grid (3 / 6 / 8 per WGP) and task size gives the same bits; T = 2 / 3 / 4 rows equal T = 1 on
  `k3_gguf_gsq_mt` and on the served `k3_gguf_gsq_qsa_mt`; `tools/gates/tp_multi.py 4 24`: 24/24 bitwise; refdump at
  K3_PER_WGP 3 / 6 / 7 bitwise the default (and, as designed, not the stored pre-change reference).
- The KLD floor gate, short (`tests/engine/test_flashnext_e2e.py`, 768 positions): mean KLD 0.0057 (before 0.0057), top-1 97.4 %
  (before 96.2 %); floor 0.0076 / 95.9 %.
- objcheck: only the six `k3_gguf_gsq` trunk / QSA / mt / trace objects change; VGPRs within 96, the mt forms at their
  one-spill exemption; spilled SGPRs go up (gsq 143 -> 145, qsa 169 -> 181, qsa_mt 195 -> 198, trace 155 -> 156): this
  needs the maintainer's acceptance next to the G-speed numbers.
- Not run: G-speed (interleaved base vs new, T = 1 and T = 4, 3 rounds), `tests/prefill/test_prefill_s6.py` g4a (the
  prefill rows of these formats are no longer bitwise the decode's), the lc / lc32 stages.

## Pairs

Bytes per decoded token from the GGUF header (routed experts at 10 / 256; 4.81 GB per token over both cards).

Decode (`k3_gguf_gsq`, `_qsa`, `_qsa_mt` served at T = 4, `_mt`, `_trace`; drafter `k3_gguf_draft`; MTP pair
`k3_gguf_gsq_mtp*`):

| Pair | Bytes | State | Own tile elsewhere |
|---|---|---|---|
| MV Q6_K (qkv, gate, q / k / v, o, ssm_out, shared gate / up, head) | 37.2 % | llama.cpp vec_dot | plane layout `own_rows_q6k_planes`; the in-place general tile is slow |
| BF16_MV (hc, router, PLE, alpha / beta, output_hc) | 28.3 % | Caminus kernel in llama.cpp's mul_mat_vec_f order | - |
| MOE_DOWN IQ4_NL | 7.1 % | llama.cpp | none (18-byte blocks need a layout) |
| MV Q5_K | 6.9 % | own (`own_rows_q5k_1rt`, in place) | - |
| MOE_GU IQ3_XXS / IQ2_S / IQ3_S / IQ4_XS | 4.2 / 4.2 / 2.8 / 0.3 % | llama.cpp | dense tiles; grouped experts only in the qwen3moe IQ study |
| MV Q4_K | 3.9 % | own (`own_rows_q4k_1rt`, in place) | - |
| MV IQ4_XS | 3.4 % | own (`own_rows_iq4xs_1rt`, in place) | - |
| MV IQ4_NL (shared down) | 0.9 % | llama.cpp | none |
| MOE_DOWN Q2_0 | 0.8 % | llama.cpp | none |
| MV Q8_0 (layer 47 shared down) | 0.04 % | llama.cpp | `own_rows_q8_0`, `OP_OWN_MV16` |
| Q8_1 glue (PLAIN / SILU / SWIGLU) | - | llama.cpp's quantize_q8_1 (feeds the GG pairs) | goes with the last GG consumer |
| Glue ops (rms, attention, GDN, QSA, exchange) | - | Caminus ops; some call ggml's `warp_reduce_sum<32>` | - |
| Drafter: MV Q8_0 (its layers), Q6_K (target head) | - | llama.cpp | - |
| MTP pair: the Coder's formats | - | llama.cpp (code objects unchanged here) | - |

Prefill: dense Q6_K `prefill_dense.hip` (Caminus kernel on MMQ's int8 numerics, includes `mmq.cuh`); dense Q4_K / Q5_K /
IQ4_XS / IQ4_NL / Q8_0 the vendored MMQ; MoE gate / up and down `prefill_moe.hip` (Caminus kernels on MMQ's vec_dot and
tile loaders); the activation quantizer and `mm_ids_helper` vendored; BF16 GEMMs, attention, GDN and glue own.
Host: token_embd IQ4_XS and the PLE table IQ4_NL through gguf-py's dequantize (`caminus.own_dequant` has no IQ4_NL yet).

## Next

1. This branch: G-speed (base vs new, T = 1 and T = 4), then the pull request with the SGPR-spill cost for acceptance.
2. MV Q6_K (37 %): the plane-layout tile exists; the prefill's dense Q6_K kernel and the DFlash drafter's head read the
   same GGUF blocks, so the layout change moves all three in one pull request.
3. MOE_DOWN IQ4_NL (needs a tile), then the MoE gate / up IQ formats.
