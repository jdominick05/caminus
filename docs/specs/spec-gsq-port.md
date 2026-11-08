K3 port to the GSQ-RCO Coder (IQ1_M pack). I read the real shard headers and only read code; I did not build or run anything.

**1. What the GGUF holds** (gguf-py header dump of both shards; the REAP-308-UI file was dumped the same way for the diff)
- **Metadata.** qwen4exp, 48 layers (36 GDN + 12 attention), expert_count 256, top-10, expert ffn 640, shared ffn 640, d 2560, hc 4, hc_low_rank 320. Indexer: 4 heads × 128, top_k 2048, compress 4 on the attention layers. PLE on layer 1 only (3-gram, 8 heads, 16 tables).
- **Shards.** Shard 1 (29.6 GB) has 1223 tensors. Shard 2 (28.8 GB) has only `per_layer_token_embd` (IQ4_NL [160, 320001536]). So all GPU tensors are in shard 1.
- **Routed experts, 25.15 GB.** Gate and up always share one type per layer:
  - IQ2_S on 20 layers, IQ3_XXS on 17, IQ3_S on 10, IQ4_XS on layer 47.
  - Down: IQ4_NL on 39 layers, Q2_0 on 9 (layers 1, 8, 10, 12, 13, 20, 21, 30, 36). Q2_0 is QK=64, 18 B per block (ggml-common.h:187).
- **Dense projections.** Types vary per layer:
  - attn_qkv: Q6_K ×22, Q5_K ×10, Q4_K ×2, IQ4_XS ×2.
  - attn_gate: Q6_K ×14, Q4_K ×10, IQ4_XS ×8, Q5_K ×4.
  - ssm_out: Q6_K ×29, Q5_K ×4, Q4_K ×3.
  - attn_q: Q6_K / Q5_K / IQ4_XS / Q4_K. attn_k: Q6_K, plus IQ4_XS on layer 7. attn_v: Q6_K / Q5_K. attn_output: Q6_K, plus Q4_K on layer 3.
- **Shared expert.** Gate and up are each IQ4_XS / Q4_K / Q5_K / Q6_K, and **gate ≠ up type on most layers** (layer 0 is Q4_K / IQ4_XS). Down is IQ4_NL ×47 and Q8_0 ×1 (layer 47).
- **BF16 tensors, 1.45 GB:**
  - hc_{attn,ffn}_{down [10240→320], up [320→10240], inject [10240→4]}: 48 layers each, 1.26 GB.
  - output_hc down/up; ffn_gate_inp [2560→256]; ffn_gate_inp_shexp [2560→1]; ssm_alpha/beta [2560→48].
  - ple_key [2560→10240], ple_value [2560→2560].
  - indexer q/k (39 MB). These stay unused until QSA lands.
- **Other.** F32 norms, ssm_a, dt, conv. ple_conv1d is F16. token_embd is **IQ4_XS** (it was Q6_K). The output head is Q6_K.
- **What changed from REAP-308.** hc, alpha/beta, ple k/v, attn k/v and the shared expert were Q8_0; the router was F32; the experts were IQ3_XXS / IQ4_NL.
  - New formats: Q4_K, Q5_K, IQ4_XS, IQ2_S, IQ3_S, Q2_0, plus BF16.
  - Q8_0 is left on one tensor.

**2. Bytes per token** (my calculation from the header)
- Total about 5.06 GB per token:
  - dense projections 1.97 GB
  - hc BF16 1.27 GB (replicated)
  - experts 0.98 GB (10 of 256)
  - head 0.52 GB
  - shared expert 0.15 GB
  - ple, router and alpha/beta 0.15 GB
- Per card at today's split (XTX 60% of the split weights, experts 384/256): **3.60 / 2.87 GB**, against REAP's 3.43 / 2.58. The bandwidth floor is 4.09 ms per token (881 / 774 GB/s).
- Resident weights: **18.1 GB on the XTX, 12.6 GB on the XT.**
  - KV (q8_0, one kv head per card) is 1.71 GB per card at 262k.
  - The DFlash drafter on the XT is about 2.5 GB, of which the fp16 head is 1.27 GB (dflash.py:51-54).
  - That leaves about 4 GB on each card for the indexer cache and a prompt cache.

**3. What llama.cpp (ignite 842653396 (published as e4e3f8eda; identical ggml/, common/, src/), HIP, gfx1100) runs**
- **Quantized dense weights:** MMVQ for ne11 ≤ 5 on RDNA3 (mmvq.cu:59-61), so MMVQ at both batch 1 and batch 4.
  - All new formats have vec_dots: vecdotq.cuh Q2_0 :754, Q4_K :973, Q5_K :1023, IQ2_S :1187, IQ3_S :1274, IQ4_XS :1420.
  - Every vec_dot reads only `__low2float(ds)` (Q4_K and Q5_K get the q sum from dp4a, vecdotq.cuh:509-525). So aq_fill's reordered ds.y (k3_gguf_ops.h:35-38) stays bitwise for every format.
- **Warps per row:** RDNA3 uses 1 warp per row at ncols=1 for Q4_K, Q5_K, IQ4_XS, IQ2_S, IQ3_S and Q2_0 (mmvq-impl.cuh:368-385). So K3's one-wave `gg_row` should be bitwise to llama.cpp for these (INFERRED, gate G3). Q6_K (2 warps), IQ4_NL and Q8_0 (8) stay tolerance-only, as today.
- **MoE (mul_mat_id) max batch for MMVQ** (mmvq-impl.cuh:245-261): IQ2_S and IQ3_S 4, IQ4_XS 6, IQ3_XXS and IQ4_NL 8, Q2_0 8. So T ≤ 4 is MMVQ for every expert format; at T ≥ 5 llama.cpp moves the IQ2_S / IQ3_S layers to MMQ.
- **Fusions that do not apply:** the paired MMVQ (hc down+inject, qkv+z, alpha+beta) needs quantized weights of the same type (ggml-cuda.cu:4395-4422). So BF16 hc does not pair, and qkv+z pairs only where the types match. Shared gate+up fusion needs the same type (ggml-cuda.cu:1829), so most layers run the shared expert as separate MMVQs plus swiglu.
- **BF16 dispatch** (ggml-cuda.cu:1984-2010; should_use_mmvf mmvf.cu:858-880; should_use_mmf mmf.cu:133-191):
  - On RDNA3, MMF qualifies when ne00 % 64 == 0 and ne01 % 32 == 0; then MMVF is used only for ne11 ≤ 3. Otherwise MMVF covers ne11 ≤ 8. Above that it falls to hipBLAS with compute type **BF16** (fast_bf16 is true on RDNA3, ggml-cuda.cu:1680-1694).
  - **ne11 = 1:** every BF16 matmul runs MMVF (`mul_mat_vec_f<nv_bfloat16>`).
  - **ne11 = 4, MMF (WMMA):** hc down/up, output_hc, router [2560,256], ple_key/value, indexer q/k. MMF **rounds the f32 activations to bf16** (mmf.cuh:208-209).
  - **ne11 = 4, still MMVF:** hc_inject (4 rows), ssm_alpha/beta (48 rows), gate_inp_shexp (1 row), because they fail ne01 % 32.
  - **Prompt batches (ne11 > 8):** hipBLAS on bf16-rounded activations. So llama.cpp's own batch-vs-decode floor on GSQ will probably be larger than on REAP (INFERRED; it must be re-measured).
- **MMVF summation order at ne11 = 1** (the target for bitwise matching):
  - Block size (mmvf.cu:439-450, the first strictly smaller niter): **256 for K = 2560 and K = 10240, 160 for K = 320** (hc up, output_hc up).
  - Thread v loops col2 = v, v+bs, …; per step it does `acc += x0*y.x; acc += x1*y.y` (HIP int path, mmvf.cu:237-272; ggml_cuda_mad is common.cuh:752). Under HIP's default contraction these are two FMAs (INFERRED; gate G2).
  - Then warp_reduce_sum<32> (xor 16..1, common.cuh:469-475), then buf_iw[warp] with slots zeroed to 32, then a second 32-lane butterfly (mmvf.cu:309-341).
  - bf16 → f32 conversion is exact.

**4. K3 design**
- **(a) Formats** (k3_gguf_ops.h:8 enum, k3.py:49 GG_FMT).
  - Add Q4_K=4, Q5_K=5, IQ4_XS=6, IQ2_S=7, IQ3_S=8, Q2_0=9. BF16 gets its own op, not a GG format.
  - Replace the four if-chains in gg_row / gg_row_tg / gg_row_n (k3_gguf_ops.h:160-176, 180-209, 418-434) with one `gg_vdot<TYPE>` constexpr switch.
- **(b) Code size:** stop instantiating every format in every op.
  - GG_DISPATCH (k3_gguf_ops.h:1356-1362) currently puts all 4 formats into each of MV, MOE_GU, MOE_DOWN and GU_DENSE.
  - Use per-op whitelists set by -D from a tape inventory script (header → (op, fmt) set):
    - MV: Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL, Q8_0.
    - MOE_GU: IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS.
    - MOE_DOWN: IQ4_NL, Q2_0.
  - That is 12 instantiations against 16 today. Build a per-model module (k3_gguf_gsq / k3_gguf_gsq_mt) beside the REAP one.
  - Bar: k3_run ≤ about 130 KB, ≤ 96 VGPRs, 0 scratch. IQ2_S and IQ3_S bring grid tables (iq2s_grid 8 KB, iq3s_grid 2 KB) and more VGPRs; one heavy instance sets occupancy for the whole kernel.
- **(c) New OP_BF16_MV.** W bf16 [N, K] row-major, x f32 [T, K], y f32 [T, N]. It replaces OP_F32_MV for the router and the shared gate; do not retype op_f32_mv, whose expression tree (k3_gguf_ops.h:579-600) differs from MMVF.
  - **One wave per row emulates MMVF:** W = bs/32 virtual warps (8 for K = 2560 / 10240, 5 for K = 320), so each lane keeps W accumulators.
  - Per pair: explicit `__builtin_fmaf(x0,y0,acc)` then `__builtin_fmaf(x1,y1,acc)`, starting from acc0(). Then W wsum butterflies, then `wsum(lane < W ? p[lane] : 0.f)`. That is the bitwise MMVF tree.
  - Float add is commutative, so only the tree shape has to match. The load layout is free: 128-bit loads, with the xor 1/2 levels done in registers.
  - **K = 320 (hc up):** stage silu(lo · 1/HC) in LDS as f32, once per workgroup (≤ T × 1.25 KB). That is ggml_scale then ggml_silu with separate roundings, as today's GG_ACT_F32_SILU does. Eight rows per workgroup.
  - **T > 1:** each token uses the same per-row order as T = 1, which keeps the internal T=4-vs-one-token gate. If VGPRs run short (8 × 4 accumulators), process the virtual warps in passes. Do not follow llama.cpp's MMF / bf16 batch path.
  - Loads-first from the start, and k3_extra_prefetch extended to the new op (k3_gguf_ops.h:1474).
- **(d) Producers that change** (flashnext_tp.py):
  - rms_mul / hc_post_rms stop emitting `q8=s['xq']` (lines 232-233): BF16 down/inject read xn as f32, which saves glue work.
  - The T>1 `gg_q8_1(lo)` (line 241) and `gg_q8_1(px)` for PLE (line 259) are dropped.
  - alpha/beta, router and shared gate read f32 `mixed`. hc_mean keeps both mixed and mq; mq still feeds the quantized projections and the MoE.
  - gdn_norm and gate_mul keep their q8 emission.
- **(e) Shared expert:** `gg_gu_dense(..., sgu[c]['fmt'], ...)` (flashnext_tp.py:348, op k3_gguf_ops.h:557) would silently decode up with gate's format. Run two OP_GG_MV segments (gate, up; independent) and a new glue mode of OP_GG_Q8_1 that computes silu(g)·u → q8_1. That matches llama.cpp's unfused path. It costs one extra hand-off, off the critical path.
- **(f) MoE:**
  - Relax the block-32 assert (flashnext_tp.py:332). `_exp_cols` converts the 32-unit ranges to type blocks (b·32/qk); require moe_blocks to be even (the default 12 gives 6/4 Q2_0 blocks).
  - Add unrolled instances: `gg_moe_down_n<Q2_0, NB 6/4>` and `gg_moe_gu_n<IQ2_S, 10>` (k3_gguf_ops.h:1404-1424). OP_ROUTER is unchanged for E = 256.
  - Expected: 20 / 17 / 10 / 1 layers of the four gate/up types; T = 4 unions are about 32 of 40 experts, as on REAP.
- **(g) Host:**
  - `_rep_f32` / `_f32` assert on F32 for gate_inp and gate_inp_shexp; add `_rep_bf16`. BF16 row slicing via `_rows` works unchanged (gguf-py gives uint8 rows).
  - Split GGUF: the PLE table at flashnext.py:18 needs its own reader on shard 2.
  - token_embd is IQ4_XS, dequantized on the host with gguf-py.
  - The K-quant ssm_out / attn_output column splits already hold for even gdn_keys (flashnext_tp.py:271-272, bph = 1).
- **(h) TP:** keep hc replicated.
  - Splitting hc up by rows needs a 40 KB all-gather per mixer. At about 32 µs per 20 KB (plan I0) that is about 6 ms per token: not viable.
  - Splitting hc down and inject by K saves about 0.32 GB per card per token (about 0.4 ms), but adds about 96 exchanges (about 0.34 ms plus the XT lag) and gives up the MMVF order: a wash.
  - Retune RPT (flashnext_tp.py:169-172) and gdn_keys / moe_blocks / vocab_share for the new 3.60 / 2.87 GB profile.

**5. Steps, gates, effort**
- **S0 (0.5 d).** Tape inventory script; GG_FMT and enum; split-GGUF PLE; BF16 loaders; Q2_0 split; shared-expert format fix (asserts first).
  - Gate: GSQ loads with `local_only`; every (op, fmt) in the tape is on the whitelist.
- **S1 (1 d).** Six new vec_dot instantiations; standalone GG_MV kernels (k3_gguf.hip:55-58 macro) on real GSQ tensors.
  - G1: engine op bitwise equal to the standalone kernel at 3, 5, 6, 7 and 8 workgroups per WGP; T=4 row bitwise equal to T=1; within 1e-6 of gguf-py float64; ≤ 96 VGPRs, 0 scratch.
- **S2 (1.5 d).** OP_BF16_MV. Compile ignite's own `mul_mat_vec_f<nv_bfloat16>` as a standalone reference (MIT, included like vecdotq.cuh).
  - G2: disassemble ignite's libggml-hip for v_fma / v_fmac, read-only.
  - G2b: bitwise equal to the reference for K = 320, 2560 and 10240, at T = 1 and per token at T = 4.
- **S3 (1 d).** Tape edits (4d-4g) in flashnext.py and flashnext_tp.py.
  - Gate: 0 replica mismatches; TP output equal to the layer-split output.
- **S4 (1 d).** `tools/gg_golden` dumps on GSQ (public text, 4 tokens) for layers {0, 1, 3, 17, 47}, which together cover all 12 types:
  - 0: IQ3_XXS, Q4_K
  - 1: IQ2_S, Q2_0, Q5_K, PLE
  - 3: attention, IQ4_XS q, Q4_K o
  - 17: IQ3_S
  - 47: IQ4_XS experts, Q8_0 shared down

  Generalize the tests/test_golden_{hc,ffn,gdn,attn,ple,moe,layers}.py checks (they still assume Q8_0 hc and F32 router).
  - G3: bitwise where expected (BF16 MMVF ops; the one-warp formats) and within 1e-6 otherwise; same 10 experts; inp_embd from IQ4_XS bitwise.
- **S5 (1 d).** Re-measure llama.cpp's own batched-vs-one-token KLD on GSQ (the D-281 floor). Then K3 TP on 768 positions over 3 texts.
  - G4 (D-281): mean KLD ≤ that floor, top-1 within 1 point. Identical greedy tokens; 3000-token soak with every bail recovered.
- **S6 (1-2 d).** Speed: RPT and split sweep, unrolled MoE instances, loads-first BF16, the k3_gguf_gsq_mt module (T=4), I-cache check.
  - Gate: every step bitwise against the previous one. Targets ≤ 11 ms plain and ≤ 17.5 ms for T=4 (INFERRED, from REAP's 10.45 / 16.5 plus the extra replicated bytes).
- **S7 (0.5 d).** DFlash on GSQ (token_embd now IQ4_XS); re-measure acceptance on code, shell and prose.

Total: about 7.5-9 days.

**6. Risks**
- **BF16 hc costs bytes.** It is +0.6 GB per card per token against REAP's Q8_0. The hc matvecs already run at about 4× their byte time (latency-bound), so the real cost is perhaps +0.3-0.7 ms (INFERRED).
- **IQ2_S / IQ3_S speed and registers.** Their table lookups may slow the MoE GU (about 30% of bandwidth today) and push it past the 96-VGPR cap.
- **The floor may move.** llama.cpp's floor on GSQ could be noisier (bf16 activations in hipBLAS / MMF). D-281 is relative, so this is fine, but report it.
- **Code-size cliff** if the whitelists are not used.
- **Host RAM.** It stays tight (29 GiB): the same 28.8 GB PLE mmap, and the loader reads 29.3 GB with O_DIRECT (about 25 s).

**7. Open questions for the user**
1. Requantize hc (and alpha/beta, ple k/v) from BF16 to Q8_0 as an opt-in K3-native variant?
   - It saves about 0.6 GB per card per token, about 0.4-0.7 ms.
   - Strata has the same option, HC_Q8 (borrow.md:40).
   - It changes the model and loses parity with llama.cpp on this file, so it would need its own KLD gate against the BF16 reference.
2. Keep the REAP-308 module building beside the GSQ one, or make it GSQ-only (code size)?
3. Was the DFlash 2 drafter trained on Flash-Next features good enough on the pruned Coder? This needs an acceptance re-measure.
