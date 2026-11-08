DRAFTERS INSIDE K3: spec for DFlash 2 and the model's MTP layer. Nothing was built or run, and no repo was changed. The only network use was one safetensors-header read at de4b8e4d. Projections are marked INFERRED.

**0. Facts this spec rests on (checked)**
- **DFlash2 GGUF** (Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf, read with gguf-py):
  - 5 layers, D 2560, FFN 7680; 32 q heads / 8 kv heads at hd 128.
  - Non-causal, SWA 2048 on all layers, block_size 8, conv k=2 with groups of 16.
  - Selector rank 256, top_k 16; target_layers [6,16,26,36,46]; mask id 248077.
  - Bytes, Q8_0: 94.0 MB per layer (q/o 11.1, k/v 2.8, conv_proj 1.7 x2, gate/up/down 20.9 each), so 470 MB of layers. fc 12800->2560 is 34.8 MB. selector_pred and selector_succ are 67.5 MB each (row gathers only). 641 MB in all; the head and embedding come from the target.
- **MTP tensors** (HfApi.get_safetensors_metadata, Qwen/Qwen3.8-Flash-Next@de4b8e4d): 31 `mtp.*` tensors, 5,214,301,696 bytes BF16.
  - fc_embedding and fc_hidden: [2560,2560] each. pre_fc_norm_embedding [2560], pre_fc_norm_hidden [10240].
  - hyper_connection_mixer (final, for the head): hc_norm [10240], down [320,10240], up [10240,320].
  - layers.0 has attn_ and mlp_hyper_connection (down/up/norm and block_inject [4,10240]) and self_attn: q_proj [12288,2560] (24 heads x 256, gated), k/v [512,2560] (2 kv heads), o [2560,6144], q/k_norm [256].
  - It also has a QSA indexer: index_qk_proj [640,2560] plus q/k layernorm [128].
  - MoE: router gate [512,2560]; experts.gate_up_proj [512,1280,2560] (3.36 GB) and down_proj [512,2560,640] (1.68 GB); a shared expert at 640 with shared_expert_gate [1,2560].
  - So MTP is a full-attention QSA MoE decoder layer with all 512 experts (the Coder keeps 256), a shared expert, 4-stream hc, and no PLE. It matches main layer 47 tensor for tensor, plus the eh projection and the head mixer.
- **How llama.cpp runs MTP** (ignite-llama src/models/qwen4exp.cpp:550-640):
  - Input h is the hc-wide residual after layer 47, before the output mixer, [T, 4x2560] (:526-527). It is not the OP_HC_AVG mean that DFlash uses.
  - hnorm is applied per stream (:607). e = enorm(embed(tok)) is repeated over the 4 streams.
  - eh_proj is one K=5120 dot over [e; h_s] per stream. The converter concatenates fc_embedding|fc_hidden (conversion/qwen4exp.py:145-151).
  - Then hc_attn mix, attention, combine, hc_ffn mix, MoE, combine. The output residual feeds the next draft step (:618-622). Then the hc head mix and the shared `output` (:624-634).
  - Generic driver: common/speculative.cpp:1433-1453 (draft-mtp, one head, chained).
- **How Strata runs MTP** (untrusted, read only):
  - Experts: all 512 resident as Q2_0 (708 MB) with an MSE scale search (mtp_pack.py:6-14, mtp.hpp:19). Q4_0 and Q8_0 are comparison arms.
  - Dense projections are Q8_0; hc and router stay BF16 (mtp_rt.py:5-11). About 0.9 GB of VRAM in total (mtp.hpp:47).
  - Attention is dense over a 32K window: exact up to 2051 cells, draft-quality only beyond that (mtp.hpp:15-19, :49).
  - Its default normalises over all 10240 at once (vLLM). `--mtp-hnorm stream` follows llama.cpp: acceptance 0.60 -> 0.63, decode +4.3% (DGX_SPARK_LINUX.md:45).
  - **The "draft subset" is a vocabulary subset for the draft head, not an expert subset.** rt/draft_vocab.bin holds 106,299 ids (en + code + CJK). en = 40,525, cyrillic = 58,963 (mtp.cpp:676-677). Chosen by script and corpus coverage at 0.99 (tools/draft_vocab.py:1-21). 133-348 MiB at IQ3_S (setup.py:7310).
  - Chain: `--spec 4` (3 drafts) with `--spec-min-p 0.5`. The host spins per draft (borrow.md:89). Measured 5.8 ms per draft round on its separate launches (docs/QUANTS.md:113).
- **MTP acceptance as reported.** Not comparable to our DFlash numbers: min-p 0.5 offers fewer drafts and inflates the rate (A770_SPEED_SPEC.md:245-247: 52% at min-p 0 vs 68% at 0.5).
  - Code 77-90%, prose/story 55-72% (INTEL.md:216-217, :442); en 0.61-0.63, code 0.77-0.78 (DETAILS.md:1345).
  - **Coder IQ1_M: 68-70%** (DETAILS.md:1425). Q2_0 draft layer: 88.7% (NVFP4.md:201). vLLM probe: 0.89/0.86/0.85 per step (mtp.hpp:4).
  - sh0wie card: 44-68%. Our unsloth MTP Q8 sidecar: 52% (D-042). Its local file was not reachable, so a fresh 5.2 GB download is the weight source.
- **K3 today:**
  - GGUF types: IQ3_XXS, IQ4_NL, Q6_K, Q8_0 (k3_gguf_ops.h:8). The Coder port adds IQ2_S, IQ3_S, IQ4_XS, Q4_K, Q5_K, Q2_0 and BF16.
  - OP_ATTN_DEC handles any D <= 512 and any kv_heads, causal from position 0 (k3_gguf_ops.h:1027-1036). It has no window start and no non-causal mode.
  - OP_QK_NR does NeoX RoPE on n_rot dims with theta_scale (:961-965).
  - The head is split by vocab 0.6/0.4 across the cards (flashnext_tp.py:53). The host argmaxes [T, 248320] logits (spec.py:237).
  - Embedding and PLE rows are built on the host from the tokens (flashnext_tp.py:388-390). **So the host stays in the loop every pass**, drafters on-device or not.
- **Pass overhead.** With tokens per pass E = 1 + 3a and DFlash acceptance a = 91/50/28%: code 3.73/142 = 26.3 ms, minus 16.5 verify and 8 draft, leaves h = 1.8 ms. Prose: 1.84/54 = 34.1, minus 16.5 and 15.6, leaves h = 2.0. So h is about 1.5-2.0 ms per pass of host and sync time.
- **The 8 vs 15.6 ms draft gap is unexplained.** `_draft_static` has a fixed shape (nb = 4, attention over all of max_pos with a mask, dflash.py:104-121), so its cost cannot depend on content. Source: task text. Re-profile it.

**1. DFlash 2 in K3**

Op list per pass (k = 3, nb = 4):
- **Inject**, over the T verify rows, using features already exported by OP_HC_AVG (flashnext_tp.py:232):
  - fc 12800->2560 as Q8_0 GG_MV, T=4.
  - RMS (OP_RMS_MUL).
  - 5 x (k and v GG_MV; QK_NR on k with n_rot 128 and base 1e4; KV_APPEND f16 into the ring (4096 rows at every context since 7d2f0a2; the attention window is 2048)). That is 62.7 MB.
- **Draft, per layer x5, T=4 rows** = [anchor embedding, MASK x3] at positions n0..n0+3:
  1. RMS, then conv_proj GG_MV (2560->640), then **OP_DF_CONV** side 0. NEW: a per-token dynamic depthwise conv, taps t and t-1 inside the block only, coefficients = dyn[group] + base[side][tap][c].
  2. q/k/v GG_MV, QK_NR on q and k, KV_APPEND at the block positions.
  3. **ATTN_DEC with a window start lo = n0+nb-2048 and a non-causal flag** (every row sees up to n0+nb-1). Two new Seg fields: about 20 lines and no new registers.
  4. o GG_MV, then OP_DF_CONV side 1, then ADD.
  5. RMS, conv_proj, OP_DF_CONV, GG_GU_DENSE (gate/up 7680, SiLU), GG_MV down, OP_DF_CONV side 1, ADD.
- **Tail:**
  - out_norm.
  - Head on rows 1-3, reusing the target's Q6_K head shards on both cards.
  - **OP_TOPK16** (NEW): partial top-16 per workgroup, then a merge; one XCHG of 16 (value, id) pairs x 3 rows across the cards.
  - sel_hidden GG_MV.
  - **OP_DF_SELECT** (NEW): one workgroup gathers 48 pred/succ Q8_0 rows, builds the [3,16,16] scores in LDS and does the greedy walk.
- **Device accept + embed**, before the draft:
  - **OP_ARGMAX_ACCEPT** (NEW): vocab argmax per verify row, merged across cards; lowest index wins a tie, like torch.argmax. Compares with the stored drafts, gives a and the bonus token.
  - **OP_EMBED** (NEW): token_embd row gather plus dequant (IQ4_XS on the Coder).
  - Drops the 4x248K logits copy, saving an INFERRED 0.3-0.5 ms of h.
- **Bytes per pass:**

  | Part | MB |
  |---|---|
  | Layers | 470 |
  | Inject | 63 |
  | KV (at most 2048 x 8 x 128 x 2 x 2 B x 5) | 42 |
  | Head (Coder output.weight Q6_K, 248320 x 2560) | 521 |
  | **Total** | **~1.10 GB** |

- **Where it runs:**
  - v1: the drafter layers on the XT, the head shards on both cards (the XTX's share is 0.6), 2 XCHGs.
  - v2: split the layers like the target if v1 is over 2 ms (kv heads 8 -> 4/4, FFN by intermediate; about 11 XCHGs at 3.5 us).
- **Expected time:**
  - v1: XT reads about 0.78 GB at about 650 GB/s, so 1.2 ms. Add the chain, 5 layers x ~18 serial segments x 5-8 us = 0.45-0.7 ms. **About 1.6-1.9 ms**, against 8-15.6 ms in torch.
  - v2: about 1.4 ms. All INFERRED.
- **Registers and code size:**
  - New code is about 8-12 KB: conv, top-k, select, argmax, embed, plus the attention flags. All are lean glue ops with LDS scratch.
  - Build them as a **separate module and launch (k3_draft)** with its own VGPR budget. That leaves k3_run (125 KB, 96-VGPR cap, 10.41 ms decode) untouched.
  - Fuse into the verify tape only if the measured launch gap is over 0.15 ms.
- **Retry-safe:** drafter state is per-position KV that gets overwritten, so a bail retry recomputes it bit for bit.

**2. MTP in K3**
- **Per round:** a catch-up pass over the T verify rows, pairing R_{p+t} with the token at p+t+1. Row a gives draft 1. Then k-1 chained single-row steps (argmax, OP_EMBED, next step) in the same launch.
- **Ops:** nearly all exist from the target's full-attention layer: hc mix, QK_NR, KV_APPEND (+Hadamard), ATTN_DEC, router, MoE gate/up and down, shexp, combine, and the output hc mixer.
- **New:**
  - **OP_HC_EXPORT**: keep R_final [T,10240] after layer 47. It is replicated bit for bit on both cards under TP.
  - **eh_proj**: GG_MV with a two-pointer K=5120 input [e_norm; h_norm_s]. Keep the concatenation: splitting it into two dots changes the summation order. 4 streams x T rows means four T=4 segments in the catch-up.
  - Indirect row select (row a read from device memory) for the head.
  - A head-subset tensor with an id map.
  - ATTN_DEC window start, so dense attention stays bounded beyond 2051 cells (draft quality only).
- **Split:** the same as the target. One kv head per card, MoE by intermediate, head by vocab. The tape builder is reused.
- **Bytes per step:**

  | Part | Size |
  |---|---|
  | Dense: eh 13.9 + hc attn/ffn 26.2 (BF16) + q/k/v/o 55.7 + router 2.6 + shexp 5.2 | 104 MB |
  | hc head mixer | 13 MB |
  | Routed, T=1 (10 experts) | Q8_0 52 / IQ4_XS 26 / Q2_0 14 MB |
  | Routed, T=4 catch-up (about 32 distinct experts) | 166 / 83 / 44 MB |
  | Head | full Q6_K 521 MB, en subset 85 MB (40.5K rows) |
  | KV per cell | 2 KB |

- **Round time (k=3, both cards), INFERRED.** A step's fixed chain is about 0.25-0.3 ms: one layer plus eh, hc head, head and argmax XCHG.

  | Configuration | Round |
  |---|---|
  | Q2_0 experts + subset head | ~1.3 ms |
  | Q8_0 experts + subset head | ~1.4 ms |
  | Full head | ~2.0-2.5 ms |

  **The head decides the speed, not the expert quant.** Expert quant is a VRAM choice.
- **VRAM:**
  - Dense 0.12 GB; experts 0.71 (Q2_0), 1.34 (IQ4_XS) or 2.67 GB (Q8_0); subset head 0.06-0.13 GB; KV 64 MB for a 32K window (0.54 GB for all 262K).
  - Start with IQ4_XS or Q2_0 (both come with the Coder port). If VRAM is short, keep 256 of the 512 experts by MTP routing frequency on agent traffic, with a router mask, behind an acceptance gate.

**3. Expected tok/s** (INFERRED; pass = verify 16.5 + draft + h 1.5; plain T=1 = 96; llama.cpp + DFlash code/shell/prose = 96/62/58)

| | Code | Shell | Prose |
|---|---|---|---|
| Today, K3 + DFlash in torch | 142 | 96 | 54 |
| DFlash in K3 (1.6 ms; a = 91/50/28) | 3.73/19.6 = **190** | 2.5/19.6 = **128** | 1.84/19.6 = 94 (below plain) |
| MTP in K3 (1.4 ms; assumed a = .80/.65/.55, to re-measure on the Coder) | 3.4/19.4 = 175 | 2.95/19.4 = **152** | 2.65/19.4 = **137** |
| Both: pick per pass, the idle drafter kept warm with a KV-only catch-up (~0.2 ms) | 188 | 151 | 136 |

- **Break-even** at pass 19.6 ms: E > 1.88, i.e. a > 0.29. Prose DFlash loses even with a free drafter.
- **Adaptive T.** Add T=1 and T=2 tapes and pick T per pass. Under a T=2 tape (about 13 ms, INFERRED) a single draft pays when first-draft acceptance is at least 0.48.
- **No tree verify.** The GDN recurrent state rules out tree-verifying both drafters' proposals, so "both" means picking one per pass by an acceptance EMA.
- **Agent harnesses sample (temperature > 0).** MTP gives a real softmax q, which exact speculative sampling needs. DFlash's selector does not. That is a strong reason to do MTP even where DFlash wins greedy code.
- **What's left after this.** Once a drafter costs about 1.5 ms it is under 10% of a pass. The remaining levers are the T=4 verify cost (16.5 vs 10.45 ms; routed experts at T>1) and acceptance, not more drafter work.

**4. Steps, gates, effort**
- **S0 (0.5 d, needs the GPUs; torch only):**
  - Re-measure DFlash acceptance per depth on the Coder: code/shell/prose plus an agent replay.
  - Profile draft_graph to explain the 8 vs 15.6 ms gap.
  - Test the vocab subset by masking DFlash logits outside it (en / en+code / 64K) and comparing acceptance.
  - Gate: subset loss no more than 1 point on code/shell.
- **S1 (1.5 d):**
  - Fetch the weights with our own fetcher (approved; on ntfs3, converted one expert at a time to fit the 29 GiB RAM).
  - Write a torch reference, caminus/mtp.py, following qwen4exp.cpp graph_mtp. K3 exports R_final.
  - Measure acceptance for: hnorm per stream vs over all 10240, RoPE position i vs i+1, Q8_0/IQ4_XS/Q2_0 experts, full head vs subset, and acceptance vs context (2K/16K/60K, windowed).
  - **Go gate:** with the section 3 formula, MTP-K3 beats DFlash-K3 on at least 2 of 3 types or on the agent replay.
- **S2 (3 d): DFlash in K3 (k3_draft).**
  - Gates:
    - Per op, bitwise vs a llama.cpp dflash.cpp refdump (cb names Kcur_injected etc.). The bitwise reference is llama.cpp, not dflash.py, whose fp16 torch matmuls can't match a Q8_0 x q8_1 kernel bit for bit.
    - Race check at 3 and 5-7 WG/WGP.
    - Gate (c): spec greedy equals plain greedy, with perfect drafts and with a corrupted draft at each position.
    - Draft tokens equal to llama.cpp DFlash for the same features (ties allowed in TOPK only).
    - Acceptance within 1 point of torch.
    - Forced-bail retry is bitwise.
    - 2.0 ms per pass or less.
- **S3 (1 d): device accept + embed.** Gates: bitwise vs the host argmax, gate (c), h down by at least 0.3 ms.
- **S4 (3 d, if S1 passes): MTP in K3.** Gates:
  - Bitwise vs a llama.cpp graph_mtp refdump (mtp_hnorm, mtp_eh_proj, h_nextn).
  - Acceptance within 1 point of the S1 reference.
  - 1.6 ms per round or less.
  - Gate (c) and the race check.
- **S5 (1 d): policy.** T in {1,2,4}, drafter choice by EMA over 16 passes, KV-only catch-up for the idle drafter. Gate: tok/s at least max(plain, each drafter alone) minus 2% on each type and on the agent replay.
- **Total about 10 days.** S2/S3 depend only on Q8_0 (on the 308 now). S4 needs the Coder port's Q2_0/IQ4_XS/BF16.

**5. Risks**
- The IQ1_M 256-expert target drifts from the BF16 states both drafters were trained on. Strata saw 68-70% on the Coder.
- MTP beyond 2051 cells is approximate, and agent prompts run 10-60K tokens. Acceptance may sag (S1 measures it).
- The head subset can never draft tokens outside it (rare identifiers, CJK). That costs acceptance only, never output.
- Fusing into k3_run would risk the 96-VGPR cap and I-cache fetch contention, which is why the drafter gets its own module.
- MTP weights are under the Qwen license: local use, never redistributed.

**6. Open questions**
- Which convention did Qwen train with: hnorm per stream (llama.cpp, and Strata measured +3 points) or over all 10240 (vLLM)? And RoPE position i or i+1?
- What does T=8 verify cost now (49.7 ms in the old build)? DFlash block 8 gives k=7 at the same draft cost, but it only pays if T=8 costs no more than about 1.6x T=4.
- For the draft head: a full-vocab Q4_0 copy (+0.36 GB, no excluded tokens) or a subset?
- Is a separate drafter launch cheap enough (under 0.15 ms) to stay unfused?

Files: python/caminus/{dflash.py,spec.py,flashnext_tp.py}, kernels/k3_gguf_ops.h; the ignite-llama fork's src/models/{qwen4exp.cpp,dflash.cpp}, conversion/qwen4exp.py, common/speculative.cpp; Strata's include/strata/core/mtp.hpp, tools/{mtp_pack.py,mtp_rt.py,draft_vocab.py}.