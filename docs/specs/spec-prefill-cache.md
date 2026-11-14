SPEC: PREFILL + PROMPT/CONVERSATION CACHE for the GSQ-RCO Coder on K3 (XTX + XT, TP=2). Read only: nothing was built or run, no GPU was used.

**0. Measured numbers**

**S0, measured 2026-11-03 (step impl:prefill-s0-s1; caminus 7d2f0a2; vm.compaction_proactiveness 0 for every run; logs in $ARTIFACTS/impl-prefill-s0-s1/logs)**

*VRAM map of the loaded Coder* (FlashNextTP T=4 + DFlashK3, max_ctx 32768, ring 4096 rows; after the load, 3 passes and a 32-token DFlash generation; torch's allocation history attributes every live block; script and bench driver kept in $ARTIFACTS/impl-prefill-s0-s1/: `vram_map.py`, `lbench.sh`). MiB per card:

| Item | XTX | XT |
|---|---|---|
| Weights as placed (TP shards + replicated BF16 hc / router / norms; of it MoE experts 5,872.5 / 3,915.0) | 17,245.5 | 12,048.8 |
| KV, q8_0 (6,528 B per position per card) | 204.0 | 204.0 |
| QSA indexer cache (raw 1,632 + pooled 408 B per position) | 63.8 | 63.8 |
| Recurrent state (GDN 70.3 / 42.2, conv 6.1 / 3.7, PLE history 0.6 / 0.6) | 77.0 | 46.5 |
| DFlash drafter, weights and buffers (XTX: the remote inject copy and head-shift rows) | 112.0 | 939.2 |
| DFlash K/V ring, 4096 rows | 0 | 80.0 |
| QSA scratch 12.2 + split-attention records 3.1 | 15.3 | 15.3 |
| Tapes, staging, logits shards, features | 5.1 | 4.7 |
| Allocator rounding (allocated minus requested) | 101.3 | 100.9 |
| torch's cache (reserved, not allocated; reusable by later torch allocations) | 1,156.0 | 59.5 |
| Not torch: HIP context, code objects, hipMalloc'd exchange slots | 253.6 | 237.9 |
| Idle before the run (desktop and other processes) | 109.7 | 260.7 |
| **Used** | **19,343** | **14,061** |
| Free to the guard's kill line (23,800 / 19,850) | 4,457 | 5,789 |
| Free to capacity.py's cap (0.97 x card - 256) | 4,224 | 5,533 |

- Context-sized buffers: 8,568 B per position per card (KV 6,528 + indexer 2,040), 268 MiB per card at 32K. At 262K they add 1,874 MiB per card more, leaving about 2.6 / 3.9 GB to the kill line.
- Totals agree with verify-dflash-ring's measure at the same configuration (19,343 / 14,061).
- The budget section's "about 10 GB free" (INFERRED, 200K) compares with 10.2 GB measured across both cards at 32K.
- The S1 feed's device token_embd table adds 322 MiB (337.7 MB) on whichever card holds it.

*llama.cpp Coder prefill*:
- **Config:** ignite-llama build 77b7e9083 (published as 719ac1c30; identical ggml/, common/, src/) (`llama-bench` from build-hip-therock), M0's flags: `-dev ROCm0/ROCm1 -sm layer -ts 55/45 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -lm dio -t 8 -b 2048 -n 0 -r 3`, `GGML_CUDA_MMVQ_MAX_BATCH=8`.
- **Reading the table:** tok/s, mean ± sd of 3 runs. "d32K" is the rate of the p tokens appended after a 32,768-token depth.
- **Peaks:** XTX 17,089 / XT 17,236 MiB at depth 0, and 18,009 / 18,114 at 32K (host MemAvailable never below 12.5 GB).

| `-ub` | pp512 | pp2048 | pp8192 | pp512 @ d32K | pp2048 @ d32K | pp8192 @ d32K |
|---|---|---|---|---|---|---|
| 256 | 1,538.8 ± 4.0 | 2,053.6 ± 6.8 | 2,188.6 ± 1.5 | 1,096.3 ± 2.5 | 1,475.4 ± 0.8 | 1,568.9 ± 0.9 |
| 512 | 1,354.9 ± 25.1 | **2,200.4 ± 5.3** | 2,540.8 ± 4.0 | 950.3 ± 8.8 | **1,574.1 ± 16.2** | 1,845.0 ± 2.1 |
| 1024 | 1,368.3 ± 15.6 | 2,172.8 ± 2.6 | **2,884.1 ± 42.9** | 952.1 ± 9.3 | 1,497.3 ± 0.9 | **1,994.1 ± 0.5** |
| 2048 | 1,369.9 ± 11.2 | 1,596.4 ± 4.4 | 2,590.5 ± 1.7 | 942.5 ± 15.9 | 1,120.7 ± 2.0 | 1,816.5 ± 20.1 |

- **Reproducibility:** pp2048 matches M0's run of the same build (2,041.6 / 2,189.3 / 2,147.1 / 1,582.7 at depth 0; 1,472.1 / 1,523.4 / 1,481.7 / 1,119.3 at 32K) within 1-3%.
- **Best of the grid:**
  - **2,884 tok/s** at 8K from empty (`-ub 1024`) and **1,994** for 8K appended at 32K;
  - for 2K prompts, 2,200 (`-ub 512`) and 1,574.
- **The `-ub 2048` cliff:** a single ubatch per `-b 2048` batch is the slowest at 2K. Smaller ubatches let the layer split overlap ubatches across the two cards (INFERRED from the pattern; not profiled).
- **Where llama.cpp's prefill time goes:** not shown. llama-bench prints only t/s, and the fork has no per-op breakdown in its output. No profiler was run.
- **Not run in S0:**
  - the 128K depth (long context is tested last);
  - the QSA selection overlap per 64-query tile, from dumps (that part of S0 in section 4 is still open).

*Consequences for the gates (for the lead to settle; the gate text below is unchanged):*
- **G5 is now inconsistent.** It asks for ≥2.5x llama.cpp's S0 number, and 2.5 x 2,884 = 7.2K (5.5K against the 2K figure). That is above this spec's own inferred ceiling of about 4.8K. The 2.5x was written against REAP's ~980 tok/s.
- **The absolute target is near parity.** 2,500 tok/s at ≤32K is about llama.cpp's own 8K number, so beating llama.cpp means more than 2,884 from empty and more than 1,994 at 32K.
- **Superseded:** section 1's "Expected: about 2.5K tok/s against about 1K for llama.cpp" now reads as roughly parity.

**Before S0 (kept for the record)**
- **llama.cpp, REAP-308, layer split 550/450, `-ub 256`:** prefill about 980 tok/s. Source: the development machine's llama-server notes (its `-ub 256` setting); not measured in this repository.
- **llama.cpp, short prompts:** 595-626 tok/s on 200-token prompts (tmp/lcpp/server-none.log). About 105-125 tok/s on 50-157-token prompts (tmp/srv308.log).
- **No llama.cpp prefill number existed for the GSQ Coder** when this spec was written (M0 then measured pp2048; S0 above has the full grid).
- **K3 T=4 verify pass:** 16.5 ms (task brief; 20.4 ms in INFERENCE-PLAN.md:681). That is about 240 tok/s, so K3 already beats llama.cpp on appends under about 200 tokens.
- **Strata, one XTX:** 900-966 tok/s on 4-9K-token prompts (Strata docs/AMD_HIP_PERFORMANCE.md:80-85). 2,583 tok/s at 74K, but with `--resident-experts` and 50 GB of RAM (AMD_HIP.md:744-745). That setup cannot be reproduced on our 29 GiB host.
- **Compute ceiling (INFERRED):**
  - About 5.9B active parameters per prompt token (shapes from INFERENCE-PLAN.md:136-146; the head is only needed for the last token), so about 11.8 GFLOP per token.
  - WMMA dense peak is about 123 TFLOP/s (XTX) + 103 (XT).
  - MMQ-class kernels reach about 25% of that, which gives a ceiling of about 4.8K tok/s.
  - **Target: 2,500 tok/s at ≤32K and ≥1,500 at 128K.**

**1. Options and recommendation**

**(b) llama.cpp prefill with a state hand-off: rejected, because the weights don't fit twice.**
- The resident shard is 29.6 GB. K3's TP layout replicates the BF16 hc weights, so K3 holds about 30.9 GB. Usable VRAM is 46.6 GB (INFERENCE-PLAN.md:176). A second llama.cpp copy (29.6 GB) does not fit.
- Time-sharing the cards is not an option: a reload takes about 30 s (INFERENCE-PLAN.md:439).
- llama.cpp cannot use K3's shards either. K3 splits MoE down by block-columns and splits GDN and attention by heads (flashnext_tp.py:144-153); llama.cpp's layer split needs whole tensors.
- The hand-off itself would be easy and a bitwise byte copy:
  - llama.cpp's KV is q8_0 with the same Hadamard rotation K3 already reproduces (INFERENCE-PLAN.md:485-490).
  - Its GDN state uses the transposed layout K3 already uses (:355).
  - Conv and PLE history are a re-layout.
  - It saves only the recurrent state, never the KV (`PARTIAL_ONLY`, server-context.cpp:2615).
- llama.cpp is also slow on our cards for two further reasons:
  - Its chunked GDN kernel runs only on RDNA3.5 with T≥2048 (gated-delta-net-mma.cu:394). On gfx1100 it falls back to a per-token loop (gated_delta_net.cu:75).
  - Its QSA prefill masks attention densely over all n_kv cells (qwen4exp.cpp:862). So prefill cost grows as O(T·n_kv) at 200K (INFERRED).
- **Keep llama.cpp as the gate reference** (gg_golden in batched mode), not as the runtime.

**(c) Large-T K3 passes: only for short appends.**
- The scalar dot-product matvecs become compute-bound at T≥8: 49.7 ms at T=8 (INFERENCE-PLAN.md:627).
- A 100K-token prompt at about 250 tok/s takes about 7 min.
- Use K3 T-passes for appends below a measured crossover T* (expected about 128-256 tokens), and for the last token.

**(a) Our own batched prefill: RECOMMENDED.**
- A conventional multi-launch path. At T=2048 a layer-chunk takes milliseconds, so launch overhead (about 1,500 launches × 5 µs per chunk) is about 1%.
- It runs in-process on K3's own TP shards and writes K3's state buffers in place.
- Its parts:
  - **Quantized GEMMs:** vendored ggml-cuda MMQ (MIT). It already covers all 10 Coder types on RDNA3 (IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_0, Q4_K, Q5_K, Q6_K, Q8_0; mmq-config-rdna3.cuh) with int8 WMMA. **stream_k is false on RDNA3** (e.g. :137), so the accumulation order does not depend on CU count or T (INFERRED; gate G2b).
  - **MoE:** mul_mat_id with tokens grouped by expert. At T=2048 that is about 80 tokens per expert, against about 10 at llama.cpp's `-ub 256`.
  - **The 484 BF16 tensors** (hc, router, ssm alpha/beta): kernels/gemm.hip. It is bf16 WMMA with a fixed K order (gemm.hip:1-3), so the replicated ops give the same bits on both cards.
  - **GDN: the per-token fp32 recurrence, not a chunked scan.** One workgroup per head-slice keeps its state resident and loops over the T tokens, with the same math order as OP_GDN_STEP. It is bitwise equal to decode for equal inputs. The cost is about 2 ms per layer per 2,048 tokens, about 9% of a chunk (INFERRED). That is about 1 day of work, against 2-3 days for a non-bitwise chunked port. Keep llama.cpp's bf16 hi/lo kernel (gated-delta-net-mma.cu) as the fallback if profiling says the recurrence costs more than 15%.
  - **QSA prompt path:**
    - Indexer key projection, writing the raw and pooled key cache (qwen4exp.cpp:81) in the format the decode-QSA spec owns.
    - Scores into a VRAM scratch: 2048 × n/4 × 4 B, 410 MB per layer at 200K, or 512-query sub-blocks.
    - Per-row radix top-512 pools plus the 3-cell tail (n_sel 2051, qwen4exp.cpp:754).
    - Gathered attention, grouped per kv head so each cell is loaded once for its 12 query heads. Per-head math order stays as in OP_ATTN_DEC, so it is bitwise equal to decode attention.
    - Cost about 0.6 GFLOP per token for indexer and attention at 128-200K (INFERRED, about 5-10%).
    - A WMMA union-tile kernel comes only if profiling demands it.
  - **TP exchange:** two all-reduces per layer of T×2560 fp32 (Slots(…, Tn*D), flashnext_tp.py:193). That is 21 MB each at T=2048: 96 × 21 MB / 7.1 GB/s = 0.29 s per chunk, a cap of about 7K tok/s and a **30-40% tax at 2.5K** unless hidden. Fix: split each chunk into 2-4 micro-chunks so one exchanges while the next computes. bf16 partials are a gated fallback, since both cards would sum the same bf16 bits.
  - **Host feed:** today 0.55 ms per token (INFERENCE-PLAN.md:602), a cap of about 1.8K tok/s. Fix it with:
    - a vectorized numpy n-gram hash per chunk and row de-duplication;
    - batched O_DIRECT gathers of the PLE rows (not mmap fancy-indexing on ntfs3) on a thread pool, overlapped with the previous chunk;
    - raw IQ4_NL rows uploaded and dequantized on the GPU (no gguf-py);
    - the token-embedding gather on the device.
- **Expected:** about 2.5K tok/s against about 1K for llama.cpp (INFERRED).
  - Append 4K at 100K context: TTFT about 1.6 s, against 4.1 s for llama.cpp.
  - Cold 100K: about 40 s, against about 100 s or more.

**Hand-off to K3 (within the engine)**
- Prefill writes committed state: GDN state block, conv [C,3], PLE history [HC·D,9], q8_0 rotated KV per position as [K|V] (flashnext.py:216, :233-241), plus the indexer cache.
- The first K3 pass after it runs with commit=0 (lazy commit, INFERENCE-PLAN.md:612).
- The last prompt token goes through a K3 pass, so the first logits come from decode numerics.
- DFlash needs OP_HC_AVG features only for the last W positions (the drafter's sliding window, dflash.py:32, :108). Prefill exports those features.

**Determinism**
- If G2b holds, position p depends only on tokens ≤p, whatever the chunk geometry: the GEMMs are per column, GDN and attention go token by token, and the exchange is per token.
- So a checkpoint at any position is reproducible.
- If G2b fails, fall back to chunk boundaries on an absolute grid (multiples of 64, which are also multiples of kpool=4), with checkpoints only there.

**2. Prompt / conversation cache**

**Structure**
- One live sequence (slot): the token list, plus KV and indexer cache by position (append-only).
- KV and indexer entries below a match point stay valid in place. Only the recurrent state needs to be rewound.

**Checkpoint contents**
- GDN state: 36×48×128²×4 B = 113.2 MB, split 30/18 value heads between the cards (gdn_keys=10).
- Conv state: 4.4 MB.
- PLE history: n_ple × 368 KB, replicated.
- The drafter's K/V window.
- The token-prefix hash.
- About 118 MB in total, roughly 73 MB on the XTX and 45 MB on the XT. llama.cpp shows the same size: RS 113 MiB per sequence, with no drafter.
- Checkpoints are written by the prefill kernels at listed positions (snapshot-on-the-fly), so no chunk has to be split.

**Policy** (the Strata DETAILS.md:687-691 and ignite server-context.cpp:2566-2590 patterns)
- **Root:** end of the system prompt plus tools, pinned.
- **Every request's prompt end:** the next agent turn diverges right after `<|im_start|>assistant\n` when the template strips earlier reasoning.
- **Start of the last user or tool message:** covers an edited last message.
- **Every 8K freshly prefilled tokens.**
- **Eviction:** keep root and newest, drop the checkpoint whose removal leaves the smallest gap.

**Budget**
- **VRAM free after load (INFERRED):** 46.6 GB minus weights 30.9, KV+indexer at 200K q8 3.8 (15.9 KB per token), drafter 0.6, prefill scratch about 1.0, leaves about 10 GB. Measure it in S0.
- **VRAM tier:** default 3.5 GB, about 30 checkpoints. A save or restore is a device-to-device copy, under 1 ms.
- **Host tier:** optional, pinned, ≤2 GB. The 28.8 GB PLE table's page cache and the torch process compete for the 29 GiB.
- **Disk tier: REQUIRED (user, 11-04).** It replaces the earlier "no disk tier", which assumed too little free space on the development machine's root disk (since freed).
  - **Where:** ~/.cache/caminus/kv/ on root (never $ARTIFACTS / NTFS, never the repo), files 0600, local only, never uploaded; a purge command.
  - **Size:** a cap (default 30 GB, configurable) with LRU eviction; retention default 7 days (configurable).
  - **Key and integrity:** the token-prefix hash + the model fingerprint + the numerics configuration (WMMA_OUT, MOE_PATH, engine version). Every file is checksummed; a truncated, corrupt or mismatched file is refused and the prompt re-prefilled.
  - **I/O:** O_DIRECT reads and writes through pinned buffers (as the PLE table does), so the page cache never grows. Page-cache pressure is what made KFD evict the GPU queues (db8bdfe). Parking writes in the background; writes are atomic (a temp file, then rename).
  - **Contents:** the token list, KV and indexer rows [0, n), and the checkpoints (GDN state, conv, PLE history, the drafter window and the DFlash ring).
  - **Gates:** a restore from disk == from VRAM == a cold prefill, bitwise (state and logits); refusal of corrupt / truncated / mismatched files; the cap and LRU enforced; park / reload times measured (target ~1-2 s to reload a 100K-token conversation, vs ~33 s to re-prefill it).
- **Parking a second conversation** (sub-agents, side calls) in VRAM: KV costs 15.9 KB per token (1.6 GB per 100K tokens). Parking and unparking is a device-to-device copy of [root, n), about 4 ms.
- **Side slot:** a small one (≤8K tokens) for short title or summary calls, so they never evict the main slot.

**On a mismatch**
1. Find the longest common prefix m.
2. Drop checkpoints above m and take the deepest checkpoint p ≤ m.
3. Restore the recurrent state, set pos=p, and truncate the indexer pools to p/4.
4. KV above p is overwritten as prefill runs [p, n).
- Example: compaction at 60K in a 120K conversation re-prefills about 60K tokens, about 24 s, against 48 s cold.

**Prompt normalization (server spec; the hit rate depends on it)**
- Pin Claude Code's changing billing-header stamp (Strata DETAILS.md:690-691).
- Use the template's reasoning-strip rules consistently.
- Log every divergence position and its cause.

**3. Gates**
- **G1 per op, against llama.cpp batched dumps at the same T:**
  - MMQ outputs are bitwise equal to llama.cpp's MMQ (vendored kernel) or within 1e-6 relative.
  - GDN state and outputs are bitwise equal to K3's OP_GDN_STEP over T steps.
  - QSA top-k sets agree ≥99.9% (ties allowed).
  - Attention is within 1e-6 of fp64.
  - KV bytes are equal to K3 decode's KV for equal K/V inputs.
- **G2 determinism:**
  - (a) Running twice gives bitwise-equal results, including the race check at 3/5-7 workgroups per WGP.
  - (b) Chunk geometry: T=512/1024/2048 and split points give bitwise-equal state and logits.
  - (c) Replicas are equal on both cards every chunk (check_replicas).
- **G3 the cache's core gate:** a cold prompt and checkpoint-restore-then-resume of the same prompt give bitwise-equal logits and state. Resuming from the live post-decode state is statistical, so it goes under G4.
- **G4 D-281, end to end:** prefill(N) then K3 decode, against llama.cpp's one-token decode on the same positions. Mean KLD must be ≤ llama.cpp's OWN prefill-then-decode floor on the same protocol at our chunk size (`-ub 2048`, below), and top-1 within 1 point of that floor's. Run it at 2K, 32K and 128K (needs decode QSA).
  - **Why this floor and not D-281's 0.0076:** the D-281 floor is llama.cpp's batch-vs-decode floor (every position's logits from one batched pass, default settings). Prefill state cannot equal decode state (S3 note: MMQ never equals the decode matvec bitwise), so S6's run is "llama.cpp-style prefill, then decode", and the bar is what llama.cpp itself scores on exactly that.
  - **Measured 11-04 (impl:prefill-s4; harness tools/gates/g4_floor.py + gg_golden `--lc` jobs `ubatch` / `tfx` / `slide`, commit 29d2141).** Setup:
    - ignite-llama fork HEAD 842653396 (published as e4e3f8eda; identical ggml/, common/, src/), build-hip-therock (the libllama of 10-28 that made the stored D-281 dumps and qsa/lc refs; the controls reproduce both bitwise);
    - layer split 0.55 / 0.45, `-fa on`, q8_0 K / V, n_batch 2048, n_ctx 2560, no GGML_* / HIP_* environment;
    - both cards, llama.cpp only, under tools/guard.sh; compaction_proactiveness 0;
    - outputs in $ARTIFACTS/impl-prefill-s4/g4 (`g4_floor.json`, `g4_floor_extra.json`, `cmp.log`, `extra.log`, the logs).
    - KLD is floor.py's math: float64, full vocabulary, KL(decode ‖ other); top-1 is equal argmax.

    | Protocol | Positions | Reference | `-ub 2048` (our chunk) | `-ub 512` (llama.cpp's default) |
    |---|---|---|---|---|
    | **short:** memory cleared, tokens [0, p) in one prompt pass, then token p alone, for every p = 1 .. 255 of the three D-281 texts | 765 | the stored one-token decode dumps, refs/d281-short/e2e_*_decode | **0.0068, 97.6 %** | the same bits (a prompt of ≤ 255 tokens is one ubatch at either setting) |
    | per text: prose / code / shell | 255 each | | 0.0099, 95.7 % / 0.0011, 100 % / 0.0095, 97.3 % | the same |
    | **2K, N = 1792:** [0, 1792) in one llama_decode (1 ubatch at 2048, 4 at 512), then 1792 .. 2047 one per llama_decode (qsa/lc/d1792/ctx.ids; all below 2051: the dense regime, S6's milestone A) | 256 | one-token decode of the same tokens from position 0, same process (`tfx` B = 1) | **0.0011, 100 %** | 0.0010, 100 % |
    | **2K, N = 2048:** one full chunk, then 2048 .. 2303 (the first 2304 tokens of qsa/lc/d8192/ctx.ids; decode past 2051, where the indexer selects: S7's regime) | 256 | the same | **0.0006, 100 %** | 0.0004, 100 % |

  - **The bar S6 must meet** (our chunk size, `-ub 2048`; top-1 = the floor's minus 1 point):
    - short: mean KLD ≤ 0.0068 and top-1 ≥ 96.6 %;
    - 2K at N = 1792: mean KLD ≤ 0.0011 and top-1 ≥ 99.0 %;
    - 2K at N = 2048 (S7: needs the prefill to write the indexer cache): mean KLD ≤ 0.0006 and top-1 ≥ 99.0 %.
    - The default-ub floors are lower at 2K (0.0010 / 0.0004). If the lead wants the stricter bar, use them.
  - **How S6 reproduces the protocol:**
    - Short: for each p, reset the state, prefill [0, p) in one chunk, then run token p through K3 (the last prompt token goes through K3 anyway) and compare its logits with dump row p.
    - 2K: prefill N tokens (N = 1792 is one chunk of 1792), then K3 teacher-forced decode of the next 256 tokens. Compare with the reference rows.
    - The reference logits are `ub512/ref1792.bin` and `ub512/ref2048.bin` ([256][248320] f32, with .tsv: pos, token, argmax, FNV-1a), tokens `qsa/lc/d1792/ctx.ids` and `impl-prefill-s4/g4/ctx2304.ids`. They are not in $ARTIFACTS/refs; promoting them (MANIFEST.md + sha256) is the lead's call.
  - **Controls (all PASS):**
    - the lc-mode one-token decode (n_ctx 2560) equals the stored main-mode dumps (n_ctx 512) on all 768 rows, by FNV-1a, so the stored dumps are the short protocol's reference;
    - prefill(1792) then decode at ub 512 equals qsa/lc/ref/tf1792_r1 on all 256 rows;
    - the one-token references at ub 2048 equal those at ub 512;
    - the short protocol at ub 2048 equals ub 512;
    - every prefill run equals its repeat (0 of 256 rows differ: llama.cpp is deterministic here).
  - **Caveat: the 2K means hang on single positions.**
    - At N = 1792, position 1811 carries 74 % of the KLD sum at ub 2048 (max 0.2121; the mean without it is 0.00029) and 55 % at ub 512 (0.1368; 0.00044 without it).
    - At N = 2048, position 2070 carries 48 % / 43 % (0.0742 / 0.0451; 0.00031 / 0.00023 without it).
    - Medians are ≤ 0.00003.
    - 95 % bootstrap intervals of the mean (positions resampled): N = 1792 ub 2048 0.00015 .. 0.00288; N = 2048 ub 2048 0.00007 .. 0.00137; short 0.00570 .. 0.00810.
    - So at 2K, a K3 run whose one outlier lands elsewhere, or is a little larger, can pass or fail on that one token. Report S6's per-position max and its position next to the mean, and the mean without the max. Consider more 2K contexts before treating the 2K bar as decisive (lead's call). The short protocol (765 positions; its max is 5 % of the sum) is the robust one.
  - **Caveat: the short bar may not be meetable as written (lead's call).**
    - G4's S6 side has two deviation sources: the prefill-made state, and K3's own decode numerics. The floor has one: llama.cpp's prefill-made state, with llama.cpp's decode on both sides.
    - K3's decode alone on the same texts scores mean KLD 0.0057 and top-1 96.2 % against llama.cpp's one-token decode (D-281 short, refs/current/e2e_tp.log, 768 positions). All 29 of its flips are at positions ≥ 5, so it is also 96.2 % on the short protocol's 765 positions.
    - That top-1 is already below the short bar's 96.6 %, and that KLD is 84 % of the 0.0068 budget, before any prefill deviation.
    - Options, not chosen here:
      - (a) judge the prefill's own contribution: S6's prefill-then-K3-decode against K3's own one-token decode on the same positions, bar = this floor;
      - (b) keep llama.cpp's decode as the reference with a combined bar (e.g. D-281's decode-only result plus this floor);
      - (c) the bar as written.
    - The 2K rows have no K3-decode-only counterpart at these positions: D-281 long compares K3 with llama.cpp's prefill-then-decode refs, a different comparison. So nothing is claimed either way there.
  - **Not measured here:** the floors at 32K and 128K (G4's other depths, with S7), and REAP-308's.
- **G5 speed (lead's decision 11-03, after S0; replaces "≥2.5× llama.cpp", which was written against REAP's ~980 tok/s and is above the ~4.8K inferred ceiling):** beat llama.cpp's best measured point at each length and depth:
  - ≥3,000 tok/s on an 8K prompt from empty (llama.cpp best 2,884, verifier rerun 2,912, ub 1024);
  - ≥2,000 tok/s on 8K at 32K depth (llama.cpp 1,994); ≥2,300 on 2K from empty (2,200);
  - stretch: ~4K tok/s at 8K;
  - TTFT ≤1.5 s for a 4K append at 100K;
  - host feed ≥3× the GPU rate.
- **G6 cache:**
  - ≥95% reused tokens on replayed public agent sessions (Qwen Code, OpenCode, Claude Code);
  - mismatch, eviction and park tests;
  - the 0.5 GB XTX reserve never breached;
  - every kernel launch under 50 ms, with no multi-second kernels.

**4. Steps** (about 20-24 days in total)
- **S0 (0.5 d, needs the GPUs):**
  - VRAM map of the loaded Coder.
  - llama-bench on the Coder: `-p` with `-ub` 256/512/1024/2048, at depths 0/32K/128K.
  - QSA selection overlap per 64-query tile, from dumps.
- **S1 (1 d):** batched host feed. Gate: host feed rate.
  - **Done 11-03** (python/caminus/prefill/feed.py, kernels/prefill_feed.hip). Bitwise equal to the per-token path on 4 prompts × 3 chunk sizes: ids, PLE vectors, embedding rows, and the whole token_embd table (tests/prefill/test_prefill_feed.py, 105 checks on the XT, BatchRows fault paths included).
  - **Feed rate:** 54,021 tok/s on text and 37,324 on random tokens, pipelined; median of 3 runs at 32K tokens, 2048-token chunks, fresh rows each run (benchmarks/prefill_feed_bench.py, the XT). The O_DIRECT reads bound it: 51.8K / 38.2K tok/s alone.
  - **Before, measured interleaved:** the per-token path's inputs (PLE.start + embeddings in T=4 passes, then _emb_row) run 8,004 / 5,786 tok/s, so the batched feed is 6.7x / 6.5x faster. The 0.55 ms/token figure (about 1.8K tok/s) is T=1 and also counts staging, status and logits.
  - **Per part, text:** hash 9.1M, dedupe 1.6M, upload 4.8M, PLE dequant 19.5M, embedding 29.0M tok/s.
  - **VRAM:** the device token_embd table costs 322 MiB on its card.
- **S2 (3-4 d):** vendor MMQ plus quantize_mmq_q8_1, with expert-grouped mul_mat_id on TP shards. Check that the down types' block size (IQ4_NL 32 / Q2_0 64) divides K3's 32-column MoE split. Gates G1, G2b.
  - **Done 11-03** (8ce055f, 1fa0dde, de428b3, 7346c38, 77ee6af): kernels/ggml_mmq (vendored, tools/release/vendor_mmq.py), kernels/prefill_mmq.hip, python/caminus/prefill/mmq.py, the two-card feed, mmq.Workspace + capacity; tests/prefill/test_prefill_mmq.py.
  - **Shards (STEP 0):** every K-split shard starts and ends on its format's blocks; every per-card row count is a multiple of 128. Only card 0's MoE down / shared-expert down width (384 of 640) is not a multiple of MMQ's 256-value K step: MMQ reads 128 values past each row, as llama.cpp's own full-K down (640) does, so those shards carry a zeroed tail of exactly that over-read (72 / 36 / 136 B). No repack.
  - **G1:** bitwise llama.cpp's MMQ (a GGML_CUDA_FORCE_MMQ fork build) on every column-split shard; K-split partials bitwise llama.cpp's MMQ on the same K shard, and the card-0-first sum bitwise the same sum of llama.cpp's (vs llama.cpp's full-K MMQ: max |diff| / max |ref| 1.3e-7 .. 6.6e-7); mul_mat_id bitwise llama.cpp's and each expert through the plain MMQ. The kernels' instructions equal ignite's shipped objects; the XTX and the XT give the same bits.
  - **G2b holds, measured:** chunk splits (1024 / 512 / 777 + 1271 / 1000 + 1048 / 512 + 788 + 748) and every J give bitwise the one-pass outputs, mul_mat_id included (groups rebuilt per chunk). So the absolute-64-grid fallback is not needed for the GEMMs.
  - **Speed at T = 2048:** the 48 layers' quantized GEMMs on K3's shards, both cards at once, take 464.7 ms per chunk (XTX's share alone 467.1, XT's 406.0), 4,407 tok/s from the GEMMs alone. On the XTX's shapes: MoE gate + up 39%, GDN qkv / z / ssm_out 28%, MoE down 23%, attention 6%, shared expert 4%. Dense Q6_K runs at 25-27 TFLOPS in MMQ (132.7 ms of the XTX's 511.8 ms per-op sum); llama.cpp's stock path sends it to hipBLAS above 128 columns (69.4 ms at the same shapes). Per op, MMQ is llama.cpp's speed within 1%.
  - **Feed:** the token_embd table only on the XT; raw rows over P2P (0.44 ms per 2K chunk; fp32 rows 3.0 ms; a second table: +322 MiB on the XTX).
  - **VRAM (chunk 2048):** workspace 524 / 465 MiB + feed 46 / 368 MiB per card, counted by FlashNextTP(vram_prefill=). Measured at 262,144 + DFlash with them allocated: XTX 21,771 / XT 16,842 MiB (2.0 / 3.0 GB to the guard lines).
- **S3 (2 d):** BF16 GEMMs and batched glue (rms, hc mean/post, router softmax and top-k, conv, PLE gate/conv). Gate G1.
  - **Done 11-03** (kernels/prefill_glue.hip, python/caminus/prefill/glue.py, tests/prefill/test_prefill_glue.py; the wide GEMMs on kernels/gemm.hip's WMMA). Measured on both cards alone (XT first), compaction_proactiveness 0.
  - **Reference per op:**
    - *Bitwise K3's decode op on the same inputs* (G1: the K3 op itself through k3.Engine, one token per pass, over the first T of 2048 tokens, T = 1 / 4 / 512 / 777 / 2048): rms_mul, hc_post_rms, hc_post, hc_mean, router (exact ties, one-ulp near ties, an all-equal token), GDN conv + its state, PLE gate, PLE conv + its history (+ the fused add_sum), add_sum, and the thin BF16 matrices through `bf16_mv` in OP_BF16_MV's split-K order (router logits 256 x 2560, shared-expert gate, ssm alpha / beta 30 / 18 rows, hc inject 4 x 10240).
    - *The norms store no fp32 xn:* they write the row scale (+ the bf16 copy the WMMA reads); hc_mean, the hc inject, the PLE gate and the PLE conv recompute xn = (x * scale) * w, the norm's own expression on the same bits. G1: bitwise the decode op chain (norm, then consumer).
    - *Router tie rule:* top-k by rank on the fp32 bit patterns of the normalized probabilities; equal probabilities: the lower expert index first; weights summed in slot order, clamped at 6.103515625e-5. Bitwise the decode router, so both TP cards and decode pick the same experts from the same logits.
    - *Measured tolerance: the wide BF16 GEMMs* (hc down / up, output_hc, ple key / value, indexer k / q) on gemm.hip's WMMA (activation rounded to bf16, fixed K order, fp32 out). The decode keeps the activation in fp32, so its order cannot be matched on WMMA; llama.cpp's own batched op (hipBLAS BF16 above 8 columns on RDNA3) rounds the activation and the output. Gate: our error vs fp64 <= llama.cpp's batched op's, both max |err| / max |ref| and mean |err| / mean |ref|, 10 matrices x (768 recorded hc_norm / hc_mixed rows, 2048 random rows): PASS 20 / 20 on each card. The recorded rows come from $ARTIFACTS/golden_layers, a REAP-308 gg_golden dump (the same architecture), not the Coder's own activations. The tolerance is measured at T = 768 and 2048; it holds at any T because G2b shows the WMMA outputs bitwise T-invariant (chunks of 1, 3, 4, 5, 512, 777 ...). Ours 4.9e-4 .. 2.7e-3 / 5.9e-4 .. 1.7e-3; llama.cpp's 1.5e-3 .. 4.3e-3 / 1.6e-3 .. 2.3e-3; K3 decode (fp32 activation) 7e-8 .. 1.8e-7.
    - *Prefill state cannot equal decode state end to end:* S2's MMQ against K3's decode matvec on the same input differs in almost every output (Q4_K attn_gate max |diff| / max |ref| 1.24e-2, Q6_K attn_qkv 2.27e-4, IQ4_XS attn_q 2.41e-4). That is llama.cpp's batch-vs-decode difference; G4 judges the whole.
  - **G2:** G2a two runs bitwise; G2b chunks 1024 + 1024, 777 + 1271, 512 + 788 + 748, 5 + 3 + 2040 and 1 + 4 + 2043 (state and history carried; chunks shorter than the 3- and 9-token windows; every WMMA token at another tile row) == one T = 2048 pass, bitwise, all 23 ops; xcard: the XTX and the XT give the same bits on the same inputs.
  - **Hand-off (S6):** the conv state and the PLE history the prefill writes are committed state, so the first K3 pass after it runs with commit 0.
  - **Speed at T = 2048** (ms per op, XTX / XT; median of 3 runs, each op the median of 20): hc_post_rms (+bf16 copy, scale) 0.466 / 0.560 (0.766 / 0.942 if it also stores fp32 xn); hc down WMMA 0.321 / 0.366; hc inject 0.105 / 0.143; hc up WMMA 0.211 / 0.280; hc_mean 0.267 / 0.314; router logits `bf16_mv` 0.457 / 0.631 (WMMA instead: 0.103 / 0.122); router 0.057 / 0.072; ssm alpha 30 rows 0.091 / 0.110; GDN conv card 0 0.187 / 0.251; ple key WMMA 1.458 / 1.698. Storing fp32 xn as well (hc_post_rms + the consumers reading it, A/B in the same run): +22.7 / +27.6 ms per chunk.
  - **One chunk of the 48 layers, S3 ops** (per-op times x the layer stack's counts): XTX at card 0's shapes 175.7 ms, XT at card 1's 217.3 ms. On the XT: hc norms 24%, hc down 16%, router logits 14%, hc_mean 14%, hc up 12%, hc inject 6%, alpha + beta 3%, GDN conv 3%.
  - **Projection against G5:** S2's stack alone (XTX share 469.6 ms, XT share 406.3 ms) + S3 = XTX 645.3 / XT 623.6 ms per 2048-token chunk, about 3,170 tok/s from S2 + S3 alone. It is a sum of per-op medians measured in isolation, each op repeated on the same buffers (warm caches, Infinity Cache 96 / 80 MB), not a measured stack; S2's stack ran with the model loaded, S3's ops on idle cards. It leaves out output_hc (the last token goes through K3), the indexer projections (S7) and the glue not built yet. G5's 3,000 tok/s at 8K allows 683 ms, which leaves about 37 ms for S4's GDN recurrence (inferred about 72 ms), attention, the glue not built yet and the exchange. **So G5 at 8K is out of reach without cuts.** The 2K-from-empty (890 ms) and 32K-depth (1,024 ms) targets leave room.
  - **Levers:**
    - router logits on WMMA: measured -0.35 / -0.51 ms per layer, about -17 / -24 ms per chunk, plus the bf16 copy of the mixed row (0.046 / 0.059 ms per layer unless hc_mean writes it); routing then takes the tolerance tier;
    - hc_mean fused into the hc up GEMM's epilogue: the 84 MB gate would never be stored, inferred -15 to -25 ms;
    - hc down reading the fp32 residual and its scale while loading, so no bf16 copy: inferred -10 to -15 ms;
    - hc down's grid: only 80 workgroups at T = 2048.
  - **Not built here:** the remaining glue (GDN gates / l2 / norm: S4; MoE and shared-expert swiglu, shexp_add, attention glue, hc_avg features: S6 / S7).
  - **VRAM (chunk 2048):** glue workspace 227 / 226 MiB per card (306 with fp32 xn); `prefill_need_mib` 796 / 1,059 MiB. Measured at 262,144 + DFlash with every prefill buffer allocated: XTX peak 21,773 MiB (2,027 MiB to the guard line), XT 17,076 MiB (2,774).
- **S4 (1-1.5 d):** GDN recurrence kernel with state snapshots. Gate G1, bitwise against K3.
  - **Done 11-04** (2c23f1d: kernels/prefill_gdn.hip, python/caminus/prefill/gdn.py, tests/prefill/test_prefill_gdn.py). The GDN half of a layer for a chunk: l2 (OP_GDN_L2), gates (OP_GDN_GATES), the recurrence (OP_GDN_STEP) and the output norm (OP_GDN_NORM; fp32 out into mmq.Workspace.x_ssm_out, no q8 copy). Each is bitwise its decode op. All runs had the XT alone first (`--small` first for every kernel change), then the XTX alone, then both; compaction_proactiveness 0.
  - **The recurrence:**
    - A per-token fp32 loop, not a chunked scan.
    - A workgroup keeps its state columns of one value head in registers and walks the chunk's tokens with OP_GDN_STEP's update code (gdn_dot / gdn_upd verbatim) and its partial sums and butterfly adds.
    - The state is K3's own layout (the first H·128² floats of a layer's gdn block), read once and written once. So the first K3 pass after a prefill runs with commit 0.
    - `rec` / `mprev` replay the decode's pending lazy commit first (OP_GDN_SAVE's records), so a prefill can follow K3 passes with accepted tokens not yet committed. Tested for 1 and 4 pending tokens.
  - **Default kernel `g8`:** a column on 8 lanes. Lane j holds K3's lanes j, j+8, j+16, j+24's values, computes their partials, and does their first two butterfly levels in registers ((p0 + p2) + (p1 + p3): K3's own operands). The last three levels are DPP. The 32-lane variants (OP_GDN_STEP's map, with and without LDS staging) are kept and give the same bits.
  - **Snapshots:** up to 8 chunk positions per launch. The state after p tokens goes into the caller's buffers on the fly, so no chunk has to be split at a checkpoint. Cost: +0.00 ms within noise for 4 snapshots.
    - The other checkpoint parts at a position p inside a chunk are the last inputs before p: S3's pf_conv_state / pf_ple_hist write them from the chunk's rows if given T = p and a copy of the chunk-start state (S8).
    - Snapshot memory is the cache tier's (S8), not counted here.
  - **G1:** both TP cards' shapes (30 value / 10 key heads, 18 / 6), real layer-0 ssm_dt / ssm_a / ssm_norm, random conv rows, alpha / beta with forced g ≈ 1, g ≈ 0 and beta ≈ 1 tokens, a nonzero initial state.
    - Every op, the recurrence at every token, the state after T, the snapshots and the norm equal K3's decode ops one token per pass (k3_gguf_gsq, FlashNextTP's GDN tape, lazy commit), bitwise, at T = 1 / 4 / 512 / 777 / 2048.
    - They also equal K3 T = 4 passes (k3_gguf_gsq_qsa_mt) at 2048.
    - PASS on the XT and on the XTX: 27 / 27 checks each.
  - **G2:**
    - G2a: two runs bitwise, and every step kernel gives the same bits.
    - G2b: chunks 1024 + 1024, 777 + 1271, 512 + 788 + 748, 5 + 3 + 2040 and 1 + 4 + 2043 (state carried) equal one pass, per position, in the final state, and in snapshots on and across the split points.
    - xcard: the XTX and the XT give the same bits.
  - **Launches:** the longest single launch timed was 1.60 ms (the XT, the slowest variant). The default g8 is ~1.06 ms at its largest shape.
  - **Speed at T = 2048** (ms per layer, median of 3 interleaved rounds of 20 launches):

    | | XTX, card 0's shapes (30 heads) | XT, card 1's shapes (18 heads) |
    |---|---|---|
    | l2 / gates / norm | 0.098 / 0.025 / 0.137 | 0.088 / 0.025 / 0.109 |
    | recurrence g8 | 0.868 | 0.796 |
    | the GDN half of a layer (all four) | **1.094** | **0.962** |
    | × 36 layers per chunk | **39.4 ms** | **34.6 ms** |

    - That is about half the spec's inferred 2 ms per layer (72 ms per chunk), and 5.8 % of the S2 + S3 + S4 sum on the XTX (6.1 % of S2 + S3 alone). It is under the 15 % line, so llama.cpp's chunked kernel is not needed as a fallback.
    - **How it got there** (the same run, card 0's shapes on the XT): first version (32 lanes, one token of register prefetch) 1.87-1.94 ms; folding the DPP moves into the adds gave direct 1.53; LDS-staged blocks gave w8b16 1.48; 8 lanes per column (g8) 1.06.
    - **Why:** one head alone runs at 0.24 µs per token and 60 heads at 1.42, so at the real shapes the loop is VALU-issue bound. Fewer instructions per token moved it; hiding latency through LDS staging did not.
  - **Projection** (sums of isolated per-op medians, as S3's): S2 + S3 + S4 = XTX 684.7 / XT 658.2 ms per 2048-token chunk. G5's 3,000 tok/s at 8K allows 683 ms, so the attention, the remaining glue and the exchange must come out of the planned GEMM work (the MoE grouped GEMM, the Q6_K path) and S3's fusion levers.
  - **VRAM:** the chunk's qk / gates / att buffers are in the glue workspace (`gdn_qk` / `gdn_gates` / `gdn_att`): +50.5 / +30.3 MiB at chunk 2048. `prefill_need_mib` is now 947.4 / 1,167.6 MiB per card. att could later alias onto mmq.Workspace.o_qkv, which is dead after the conv (S6's call).
  - **Open for S6:**
    - S3's conv and PLE kernels have no lazy-commit replay. A prefill that follows decode passes with pending accepted tokens must commit those first; the GDN state alone replays here.
- **S5 (1.5 d):** TP exchange with micro-chunk overlap. Gate G2c and speed.
  - **Done 11-04** (37b676d: python/caminus/prefill/tp.py, kernels/prefill_tp.hip, tests/prefill/test_prefill_tp.py, benchmarks/prefill_link_bench.py, `FlashNextTP.weights`). Both cards, compaction_proactiveness 0 for every run; logs and JSON in $ARTIFACTS/impl-prefill-s5 (`stack_commit.*`, `link5.*`).
  - **The link** (benchmarks/prefill_link_bench.py, hipMemcpyPeerAsync):
    - 7.10 / 7.11 GB/s one way; 14.0 GB/s both directions pushed (12.6 pulled).
    - A 21 MB exchange (2048 x 2560 fp32) takes 2.99 ms duplex, so 96 per chunk is ~287 ms if exposed (the spec's 0.29 s). Pieces scale linearly (1/2: 1.51, 1/4: 0.76 ms).
    - Beside an MMQ GEMM on each card, the GEMM goes 7.32 -> 7.38 ms (XTX) and 10.27 -> 10.61 (XT).
    - **Feed contention:** the two-card feed's 2.79 MB copy takes 0.41 ms alone and adds 0.40 ms to an exchange it shares the link with (2.99 -> 3.39). Inside a chunk, a feed-sized copy issued mid-chunk is not measurable (695.0 vs 695.5 ms).
  - **The exchange** (`prefill_tp.Exchange`):
    - Each card pushes its partial rows on its own copy stream into the peer's ordinary receive buffer (20 MiB per card at chunk 2048, in `prefill_need_mib`).
    - **VRAM:** each new HIP stream takes device memory at its first use: the compute streams +51 / +45 MiB (XTX / XT) at their first kernel, the copy streams +44 / +38 MiB at their first copy, nothing more over later chunks. They were outside every count at 37b676d: the XT's measured use passed idle + its projection by ~114 MiB, the test's own side-copy stream included. Since the follow-up commit they are counted (`capacity.PREFILL_STREAMS_MIB`, 104 / 96 MiB) and taken at construction. `prefill_need_mib` at chunk 2048 is now 1,071.4 / 1,283.6 MiB. The stack stage checks idle + projection >= measured use after the prefill ran, before any test-only buffer exists: XTX 20,148 vs 19,704 MiB (+444), XT 14,059 vs 14,012 (+47).
    - `pf_xchg_add` then sums p0 + p1 in place, card 0's partial first on both cards.
    - Ordering is HIP events only:
      - the partial-ready event carries a system-scope release;
      - every push waits the peer's last read of its receive rows;
      - every sum waits both pushes.
    - Not torch's cross-device `copy_`, and no spin.
    - The release scope is within noise (all / ready / none: 695.5 / 694.3 / 694.4 ms). Half duplex is worse (709.5).
  - **The measured chunk** (`prefill_tp.ChunkStack`):
    - What runs: all 48 layers with S2's MMQ, S3's glue and WMMA, S4's GDN half, the exchange and the safety step's router (WMMA logits, id clamp, flag words), on K3's own weights and states.
    - Inputs: real text (qsa/lc/d8192); the two-card feed's device part included. The host part (30-33 ms per chunk) was prepared beforehand, as the pipelined feed would have it.
    - **Not in it:**
      - the attention core (`pf_attn_standin`: each token attends to itself; the q / k / v / output GEMMs and the exchange run on real inputs);
      - DFlash features, the head, the hand-off.
    - Results, ms per 2048-token chunk, warm (median of 3 after a cold round, interleaved; stack_commit.log):

    | m (micro-chunks) | no exchange (each card its own partial: the floor) | exchange serialized | exchange overlapped |
    |---|---|---|---|
    | 1 | 618.1 | 911.3 | - |
    | 2 | 633.7 | 954.8 | **694.3** (2,950 tok/s) |
    | 3 | 751.1 | 1,090.2 | 848.8 |
    | 4 | 814.9 | 1,172.8 | 939.2 |

    - **The overlap's gain:** -217.0 ms against the serialized exchange at m 1, -260.5 against it at the same m 2.
    - **m 2 is the best:** above it, the MoE GEMMs (~20 tokens per expert at 512-token pieces) and the WMMA GEMMs lose more than the overlap hides.
    - Exposed at m 2: 60.6 ms over the floor.
    - **Cold:** the process's first 2048 chunk takes 910.8 ms (m 1 serial). Every variant's cold round is within 2 ms of warm but one (overlap+feedcopy m 4: 958.0 vs 940.1).
    - **Host enqueue:** ~320 ms at overlap m 2, against ~694 ms of GPU time, so not host-bound. It grows ~140 ms per extra micro-chunk.
  - **8K from empty** (4 x 2048 back to back, the next chunk's host feed on the feed's thread; stack_commit2.log):
    - overlap m 2: 2,818.8 ms = **2,906 tok/s**;
    - the same with chunk 0's host part prepared before the window: 2,776.1 = 2,951, i.e. 4 x the single chunk. So no idle at the chunk boundaries; the 42.7 ms difference is chunk 0's host feed part on the critical path (30-33 ms measured alone; the rest is inferred to be the next chunk's host part contending for the host while chunk 0 is enqueued);
    - serial m 1: 3,676.8 = 2,228;
    - no exchange m 2: 2,558.3 = 3,202.
  - **Against G5:**
    - 3,000 tok/s at 8K allows 683 ms per chunk. Measured: 704.7 ms per chunk with chunk 0's host feed in the window (2,906 tok/s, 3 % short), 694.0 without it (2,951, 1.6 % short), before the attention core, the DFlash features and the hand-off.
    - ≥ 2,000 at 32K depth: not measurable without the attention core, which is the depth-dependent part. Everything measured here is depth-independent.
  - **Gates:**
    - check_replicas after every chunk (the residual, every layer's router ids and weights, the PLE history: card 0 == card 1), in every non-local run (76 timed runs, the G2b runs, an untimed 8K run); the routing and feed flag words clear on every chunk.
    - G2b, bitwise against one 2048 chunk at m 1 serial: the residual at 2048 / 2048 positions, the routing of all 48 layers on both cards, and the final GDN / conv / PLE states. Checked at m 1 / 2 / 3 / 4 (serial and overlap), 1024 + 1024 and 777 + 1271 (m 2; 777 + 1271 also m 1).
    - The new kernels on the XT alone (small T first), then the XTX alone:
      - the exchange sum: bitwise the fp32 sum;
      - the MoE combine: its float32 emulation, and K3's OP_SHEXP_ADD per token, bitwise;
      - the attention stand-in: OP_GATE_MUL bitwise;
      - swiglu: within 1.96e-7 of fp64.
    - Decode objects byte-identical; gates.sh trunk grids qsa multi 19 / 19; k3_smoke PASS, tokens 12 / 12 lists equal to the previous run.
  - **Where the chunk goes:**
    - Compute without the exchange, local m 2 (instrumented, XTX / XT ms):

      | Part | XTX | XT |
      |---|---|---|
      | MoE (group + quantize + mul_mat_id) | 263.1 | 249.1 |
      | dense Q6_K | 115.6 | 111.8 |
      | other dense MMQ | 59.3 | 58.7 |
      | glue (norms, hc mean, conv, l2 / gates / norm, swiglu, combine, PLE) | 79.1 | 89.7 |
      | BF16 WMMA (hc down / up, PLE) | 58.3 | 72.7 |
      | GDN recurrence | 30.7 | 27.9 |
      | thin BF16 | 19.7 | 25.9 |
      | router | 9.1 | 11.3 |

    - The exchange at overlap m 2 (coarse timelines):
      - XTX: compute 678.9, exchange wait 6.9, sum 27.6;
      - XT: compute 653.4, wait 44.8, sum 14.5;
      - local m 2 compute: XTX 620.7, XT 632.2.
      - So the 61 ms are: the XTX computing ~58 ms slower while the copy engines run both ways (paced pushes alone, no sums or waits: 620 -> 656 ms; one direction alone: 0), its sums, and the XT waiting for it.
    - Idle: host enqueue stays below the GPU time (~320 ms at overlap m 2), and back-to-back chunks show no gap at the boundaries (the 8K split above). What 8K from empty spends besides chunks is chunk 0's host feed part (42.7 ms).
  - **Levers (not built):**
    - the MoE grouped GEMM and dense Q6_K (the plan's next GEMM work: 263 + 116 ms on the XTX);
    - the exchange sum fused into the next hc_post_rms (two block inputs, the same bits: ~27 ms on the XTX);
    - bf16 partials (half the copy traffic and its interference; gated fallback).
- **S6 (1.5 d):** chunk loop, hand-off, tail through K3, DFlash window features.
  - **F16 target-cache implementation (GPU gates pending):** the attention core now follows `FlashNextTP.q8`.
    Q8 keeps the measured rotation and byte layout below. F16 omits target K/Q/output Hadamards and stores unrotated
    half K then V, using the separate `prefill_attn_f16` append and split-attention kernels. The indexer remains Q8.
    The F16 module is loaded at stack construction, before graph capture. F16 park numerical keys include the new
    prefill version so caches made by the former Q8-only prefill path cannot be restored into it. Focused CPU tests
    cover dispatch, views, bounds and keys; they do not prove device parity. On the non-display card first, run
    `tools/guard.sh LOG 1800 python -I tests/prefill/test_prefill_attn.py CODER g1 --kv f16 --small` and
    `tools/guard.sh LOG 1800 python -I tests/prefill/test_prefill_qsa.py CODER g1 --kv f16 --small`, then full gates
    and the Q8 regression arms, before claiming support or exposing a user switch.
  - **Milestone A:** usable at ≤2,051 tokens. Measure G5 early.
  - **Done 11-04** (224008d, c1c4068, 166ec51: kernels/prefill_attn.hip, python/caminus/prefill/attn.py, python/caminus/prefill/coder.py, `ChunkStack(attn='core')`; tests/prefill/test_prefill_attn.py, tests/prefill/test_prefill_s6.py). Both cards, FlashNextTP T = 4 at max_ctx 4096 (QSA dual) + DFlashK3 + `Prefill(chunk=2048)`, compaction_proactiveness 0; logs in $ARTIFACTS/impl-prefill-s6/logs.
    - **The attention core:** q / k norm + RoPE, the Hadamard rotations, the q8_0 KV append into K3's own cache, dense attention as the Coder's split pair over the identity list, the output rotation and gate, and OP_HC_AVG for the DFlash features. Each is bitwise its K3 decode op, one token per pass, at T = 1 / 4 / 512 / 777 / 2048 / 2051 on the XT and on the XTX; G2b holds over chunk splits and record blocks. K3's record layout would take ~1.7 GB per card at 2048 rows, so rows run in 64-row blocks (49.5 MiB of records). attend at T = 2048: 4.74 / 6.60 ms per layer (XTX / XT), the core 67 / 93 ms per 12-layer chunk; the largest launch pair is 0.30 / 0.41 ms.
    - **IDXRAW:** on a QSA engine the prefill writes the indexer's raw key rows: k_proj through `Glue.bf16_mv` (bitwise OP_BF16_MV), then q8_0, as the dense tape writes them. The pools past the prefill's first position are marked stale for the QSA tape's backfill. The N = 2048 window (decode past 2051) runs on those rows.
    - **The hand-off** (`caminus.prefill.coder.Prefill`):
      - the whole prompt is validated before anything is reset or written (`PrefillRefused`); three refused prompts on the loaded engine left every state, KV / IDXRAW row and feature byte unchanged;
      - the state is committed and the next K3 pass commits 0; the last prompt token goes through a K3 pass;
      - an append restarts at the last committed position (no conv / PLE replay needed);
      - after the chunks every flag word is checked, and a set one poisons the engine.
    - **Entry points** (opt-in `prefill=`): `longctx.fill`, `spec.generate`, `DFlashK3Drafter.generate`, `PolicyDrafter.generate`. A prefill with an MTP drafter is refused (it produces no per-pass h rows).
    - **T\* = 48** (the batched path including its DFlash inject vs K3 T = 4 passes, to the first token, from empty / at depth 1024, ms):

      | L | batched | K3 passes |
      |---|---|---|
      | 32 | 195.9 / 193.8 | 140.1 / 144.2 |
      | 48 | 198.2 / 196.1 | 209.5 / 214.7 |
      | 64 | 198.5 / 196.5 | 278.1 / 285.2 |
      | 128 | 206.4 / 203.5 | 555.6 / 568.3 |

      The batched path's ~190 ms floor is host enqueue: 4,847 launches per chunk, ~160 ms; the ctypes launch wrapper is about half of it, the buffer checks ~22 ms.
    - **Micro-chunks by size:** m 1 below 1024 tokens, m 2 from 1024 (chunk alone, ms):

      | Chunk | m 1 | m 2 |
      |---|---|---|
      | 256 | 183.8 | 323.2 |
      | 512 | 302.3 | 327.7 |
      | 1024 | 505.6 | 495.3 |
      | 2048 | 999.5 | 765.4 |

    - **Speed from empty** (TTFT: reset, prefill of n - 1, the hand-off, the last token's K3 pass, the first token on the host; medians of 3):

      | Prompt | TTFT | tok/s | with the DFlash inject | llama-bench (S0 best) |
      |---|---|---|---|---|
      | 512 | 336.6 ms | 1,521 | 367.1 ms = 1,395 | 1,539 (ub 256) |
      | 1024 | 549.6 ms | 1,863 | 607.3 ms = 1,686 | no cell measured |
      | 2048 | 820.1 ms | 2,497 | 929.7 ms = 2,203 | 2,200 (ub 512) |

      - K3 passes alone at 512: 2,192.8 ms = 233 tok/s.
      - The inject is T rows per launch through the drafter's own inject: 448 / 500 launches take 91 / 102 ms at 1792 / 2000 tokens.
      - The 8K-from-empty number is not an S6 measurement: attention past 2051 is S7. S5's stand-in stack: 2,906 tok/s.
    - **G4a** (the prefill's own error vs K3's own one-token decode):
      - short, the batched path forced for every p (765 positions): KLD 0.00619 (bar 0.0068), top-1 97.25 % (bar 96.6);
      - short, routed: 0.00538 / 97.91 %; the 141 positions below T\* are bitwise K3;
      - 2K N = 1792: 0.00031 (bar 0.00288), 100 %; max 0.0296 at 2010, 0.00020 without it;
      - 2K N = 2048: 0.00045 (bar 0.00137), 100 %; max 0.0638 at 2291, 0.00020 without it.
    - **G4b** (vs llama.cpp's own prefill-then-decode):
      - **short: 0.00769, bar 0.0076: FAIL.** Top-1 98.17 % passes. Per text: prose 0.0106, code 0.0007, shell 0.0118. The value is inside the short floor's own 95 % bootstrap interval (0.0057 .. 0.0081);
      - 2K: 0.00040 / 0.00064, 100 %;
      - reported, not gated: short vs llama.cpp's one-token decode 0.00706 / 97.12 %.
    - **Router A/B** (the exact logits, bitwise the decode's): G4a short 0.00639 / 96.73 %, G4b short 0.00830. Worse on both, so WMMA stays the default.
    - **WMMA output A/B** (fix:prefill-s6, a569eb8; `Glue(wmma_out='bf16')`, `--wmma-out bf16`). Every WMMA result, router logits included, is rounded to bf16 by RNE. That is llama.cpp's own batched BF16 mul_mat on RDNA3 above 8 columns: hipBLAS with a bf16 C.
      - **On the XT:** llama.cpp's output has its low 16 bits zero on 100 % of elements at T 9 .. 768, and on 0 % at T <= 8.
      - **Against llama.cpp:** our rounded output equals its elements on 99.7-100 %, and its router top-10 sets on 100 %. With fp32 logits the sets match on 96-98 %.
      - **Results, run once:**

        | | G4a short forced | G4a routed | G4b short | G4a 2K (1792 / 2048) | G4b 2K (1792 / 2048) |
        |---|---|---|---|---|---|
        | fp32 (default) | 0.00619 / 97.25 % PASS | 0.00538 / 97.91 % | 0.00769 / 98.17 % FAIL | 0.00031 / 0.00045 | 0.00040 / 0.00064 |
        | bf16 | 0.00739 / 96.47 % FAIL | 0.00640 / 97.39 % | 0.00645 / 98.17 % PASS | 0.00065 (99.61 %) / 0.00021 | 0.00071 / 0.00023 |

      - **Why no single choice passes both:** moving the prefill state toward llama.cpp's prefill fixes G4b and breaks G4a by about the same amount. G4a forced plus G4b short sums to 0.0139 in both rows.
        - K3's own decode, with no prefill, already scores 0.00737 against llama.cpp's prefill-then-decode (verify:prefill-s6). That is above G4a's 0.0068 and just under G4b's 0.0076.
        - So the two short gates measure distances to two references that are themselves about 0.0074 apart.
      - **Decision (user, 11-04): bf16 is the default.** The prefill reproduces llama.cpp's own batched numerics. The router then picks llama.cpp's top-10 sets on 100 % of tokens, a discrete agreement, while every KLD difference here is inside the short floor's bootstrap interval (0.0057 .. 0.0081). fp32 stays one switch away (`wmma_out='fp32'`) for later A/Bs on agent tasks.
        - **Binding from now on:** G4b short and 2K (D-281's bar); G4a routed short (the serving path) and the G4a 2K windows (chunk boundaries vs K3's own decode).
        - **Reported, not gated:** G4a short with the batched path forced (llama.cpp's own seam 0.0068 / 97.6 % for scale), and the score vs llama.cpp's one-token decode (the spec's original G4: 0.00701 / 97.52 % in bf16 mode, inside D-281's bar).
        - **Re-measured with bf16 as the default (the lead, 11-04):** G4b short 0.00645 / 98.17 %; G4a routed 0.00640 / 97.39 %; G4a 2K 0.00065 / 0.00021; G4b 2K 0.00071 / 0.00023; G4a forced 0.00739 / 96.47 % (reported). Glue gpu, g2b / g3 / dflash / long, gates trunk multi short (9 / 9) and the smoke pass.
    - **G2b and G3-lite:**
      - [0, 2048) in chunks of 1024 / 512 / 777 and as 1000 + 1048 runs equals one chunk bitwise: every state, the KV / IDXRAW rows, the features and the next 8 QSA-tape logits;
      - a prefill to 512 / 1001, a snapshot, a probe that rewrites the rows past it, a restore and a resume equal the cold prefill, bitwise.
    - **DFlash after a prefill:** spec == plain greedy after the same prefill, bitwise, on 8 prompts x 96 tokens. Accepted drafts per pass: 2.258 vs 2.219 on a K3-filled context (paired +0.039, 2 SE 0.213).
    - **VRAM after a prefill:** XTX idle 110 + projected 20,401 ≥ measured 19,715 MiB; XT 279 + 14,963 ≥ 15,167. `prefill_need_mib` at chunk 2048 with 5 feature layers: 1,248 / 1,360 MiB.
    - **Levers:**
      - the host enqueue (a cheaper launch path, or fewer launches): T\* and short prompts;
      - a batched DFlash inject: ~100 ms at 2K;
      - the planned MoE / Q6_K GEMMs;
      - the attention core's scalar split pair, which a WMMA version would make non-bitwise.
- **S7 (3-4 d):** QSA prompt path. It shares the cache format with the decode-QSA spec and depends on it.
  - **Milestone B:** correct up to 200K; G4.
  - **Done 11-04 (impl:prefill-s7; 6cfd91c, 123acc0, 9abe219 + the docs commit).** On a QSA engine the batched prefill covers positions up to max_ctx; rows from 2,051 on take the QSA prompt path instead of K3 T = 4 passes (~230 tok/s before). Both cards unless noted, compaction_proactiveness 0, MemAvailable 14.3-14.9 GB at every start; logs and JSON in $ARTIFACTS/impl-prefill-s7.
    - **Ops** (kernels/prefill_attn.hip, caminus.prefill.attn), each its QSA-module decode op's expressions in its order: pf_idx_pool (OP_IDX_APPEND's pooling, from the IDXRAW rows the prefill writes by kv_append M = 64; pools from token 0), pf_qsa_score (OP_QSA_SCORE for a query sub-block into AttnWorkspace.score: score_rows rows of (max_ctx + 3) / 4 pools, <= 64 MiB per card; 1,984 rows at max_ctx 33,024, 256 at 262,144; a reduce-scatter instead of 16 butterflies per row and wave, the same tree: 5.68 -> 2.13 ms per 1,984-row sub-block on the XT, bitwise the direct copy pf_qsa_score_v0), pf_qsa_select (OP_QSA_SELECT: the top 512 pools, the lowest index first among ties, ascending, + the tail), pf_attn_part_sel / pf_attn_comb_sel (the split pair over SEL lists; one template with the dense pair). The indexer query is q_proj through Glue.bf16_mv (bitwise OP_BF16_MV, like k_proj), then OP_QK_NR's q norm + RoPE.
    - **ChunkStack:** rows below 2,051 dense as in S6, rows from 2,051 through the QSA path; every chunk pools its rows (and at its first piece every pool left unpooled before it), so after a chunk FlashNextTP._pool_ok = p1 / 4 and the QSA tape backfills nothing. Prefill.limit = max_ctx on a QSA engine (QSA_EXACT otherwise); the entry points batch up to it.
    - **DFlash:** the feature export stays 2,052 rows: it holds the last 2,052 positions of a longer prompt (ChunkStack.feat_base); the inject starts at max(p0, feat_base) after skip_inject (DFlashK3.skip: n_inj advanced, the skipped ring slots carry no tag of theirs, so any window reading one is refused). Prompts within 2,052 positions inject as before.
    - **G1** (tests/prefill/test_prefill_qsa.py g1, the XT alone, then the XTX alone, each launch between 1 MiB sentinel guards): positions 0 .. 6,199 == K3's QSA module one token per pass, bitwise: IDXRAW, IDXPOOL (1,550 pools), the indexer q, the scores of 4,149 rows, SEL of 4,149 QSA rows, the attention output; no byte outside any launch's range changed; G2b over chunkings, score sub-blocks of 64, record blocks of 17, the v0 score kernel. Longest launches (XT): score 1.05 ms, select 0.24, attention pair 0.42; at a 32K-depth sub-block score 2.13, select 0.58.
    - **G2b across 2,051 and at 8K** (test_prefill_s6 g2b7, max_ctx 33,024): [0, 8192) in chunks of 1,024 / 777 and as runs [0, 2051) + rest, [0, 3000) + rest == chunks of 2,048, bitwise (every state, KV / IDXRAW / IDXPOOL rows, the export window's features from 3,000, the next 8 logits). Also at max_ctx 4,096 (N = 4,000) for the chunkings; the two-run variants there compared the export's rows before the last run's start, which the export does not keep (fixed in the test).
    - **G3-lite past 2,051** (g3q): prefill to 2,500 / 3,001 / 7,192, snapshot, a probe rewriting the rows after, restore, resume to 8,192 (and to 4,000 at max_ctx 4,096) == the cold prefill, bitwise.
    - **The depth floors** (tools/gates/g4_floor.py depth-jobs / depth-cmp; llama.cpp prefill-then-decode at ub 2048 vs its one-token decode, positions N .. N + 255 of qsa/lc/d{N}/ctx.ids): N = 8,192 mean KLD 0.00214 (95 % bootstrap 0.00075 .. 0.00403), top-1 99.61 %; N = 32,768 0.00113 (0.00009 .. 0.00293), 100 %. llama.cpp past 2,051 is not run-to-run deterministic: the stored qsa/lc tf8192 / tf32768 r1 vs r2 differ in every row's FNV and 2 argmaxes each; our pd32768 r1 vs r2 in every row, 0 argmaxes (pd8192 r1 == r2).
    - **G4 at depth** (g4d; G4a bar = the floor's bootstrap upper bound and its top-1 minus 1 point, as the 2K windows; G4b = D-281's bar):

      | | G4a vs K3's one-token decode | G4b vs llama.cpp prefill-then-decode ub 2048 | reported: vs llama.cpp one-token | reported: K3 decode alone vs llama.cpp pd |
      |---|---|---|---|---|
      | N = 8,192 | 0.00156 / 100 % (bar 0.00403 / 98.61; floor mean 0.00214) PASS | 0.00397 / 99.61 % PASS | 0.0025 / 100 % | 0.0033 / 99.6 % |
      | N = 32,768 | 0.00128 / 100 % (bar 0.00293 / 99.00; floor mean 0.00113) PASS | 0.00127 / 100 % PASS | 0.0016 / 100 % | 0.0026 / 100 % |

      At 32K the G4a mean is 0.00015 above the floor's own mean (inside its bootstrap interval); max 0.0913 at 32,919, the mean without it 0.00093. Replicas equal after every window.
    - **Copy probes** (probes: qsa/lc needle + rare token at 10 / 50 / 90 % of 8K and 32K, the whole probe prompt batched): 12 of 12 answered as llama.cpp answers them (10 found by both, q_rare_50 at 8K and q_needle_50 at 32K missed by both); 3 continuations token-identical to llama.cpp's, the others part after the answer (first divergence 10-15 tokens in).
    - **Speed** (speed7, TTFT from empty: the prompt's n - 1 positions batched, the hand-off, the last token's K3 pass, the first token; medians of 3 interleaved; llama-bench beside it at the S0 flags, ub 1024 best):

      | Prompt | TTFT | tok/s | with the DFlash inject | llama-bench (11-04) | G5 |
      |---|---|---|---|---|---|
      | 4,096 | 1,408.1 ms | 2,909 | 2,696 | 2,628 (ub 1024), 2,164 (ub 2048) | - |
      | 8,192 | 2,851.1 ms | 2,873 | 2,768 | 2,884 (S0, ub 1024) | >= 3,000: **not met** |
      | 32,768 | 11,676.1 ms | 2,806 | 2,779 | 2,791 (ub 1024), 2,699 (ub 2048) | - |
      | 8,192 appended at depth 24,576 (with the inject) | 3,147.6 ms | 2,603 | (included) | 8K at 32K: 1,994 (S0) | >= 2,000 at 32K depth: 8K at 32K reaches 40,960 positions and was not run (no fills beyond 32K) |

      Per-layer QSA core at a 32K-depth chunk (synthetic, test_prefill_qsa speed): XT 15.2 ms (score + select 2.9, attend(sel) 12.2); XTX 13.7 measured with the v0 score kernel (score + select 5.0, attend(sel) 8.7); the attention over ~2,051 selected cells per row is the larger part.
    - **VRAM:** in a fresh process after a 32,768-token prompt with DFlash at max_ctx 33,024 (vram7): idle + projected >= measured, XTX 20,823 >= 20,212 (+611), XT 15,556 >= 15,503 (+54). In-process after earlier stages the check FAILS on the XT: after g2b7 .. probes 15,577 measured (-21 vs the same projection); after g3q alone 15,565 (-11). Measured cause: torch's allocated memory is the same as in the fresh process (14,838 MiB on the XT) while its reserved-but-free cache grew from 62 to 126 MiB (reserved 14,900 -> 14,964) through the test's snapshots; capacity does not count torch's free cache, so with S8 keeping snapshots on the device the XT margin (+54 fresh) is a real gap. 262,144 + DFlash + prefill, allocation only (alloc): XTX 22,189 measured / 22,698 projected vs the 23,800 line, XT 17,361 / 17,431 vs 19,850.
    - **S6 gates after S7** (test_prefill_s6 at max_ctx 4,096): smoke, g2b, g3, g4a (every number equal to the previous run: G4b short 0.00645 / 98.17 %, G4a routed 0.00640 / 97.39 %, 2K 0.00065 / 0.00021, G4b 2K 0.00071 / 0.00023), dflash (2.110 vs 2.219 accepted per pass, as before) PASS. **long: FAIL on its top-1 bar** (the 3,000-token prompt is now batched whole: mean KLD 0.00021 vs 0.0068, top-1 98.44 % vs 99 %: one flip in 64 positions, at 3,017, K3 decode's top-1 / top-2 gap 0.10); its DFlash and snapshot / restore checks PASS.
    - **Decode untouched:** only build/prefill_attn.hsaco changed; gates.sh trunk multi short 9 / 9; k3_smoke RESULT PASS, its 9 token lists equal the 11-04 09:21 run's.
- **S8 (3 d):** cache (checkpoint radix, tiers, policy, mismatch handling, side slot, VRAM park). Gates G3, G6.
- **S9 (2 d):** soak, a scoreboard row, and the bail path: a prefill chunk is idempotent from its input state, so a retry is just a re-run.

**5. Risks**
- **TP exchange over OCuLink x4** may cap prefill near 7K tok/s raw, or cost 30-40% if the overlap fails.
- **VRAM:** scratch plus checkpoints plus a second parked conversation, against the 0.5 GB reserve. The tiers must size themselves from measured free memory.
- **Cold PLE reads on ntfs3:** 32K random rows per 2K chunk.
- **MMQ T-invariance** is assumed (G2b). If it fails, the absolute grid costs up to 63 extra tokens per turn.
- **Noise at the IQ1_M bit level:** prefill-against-decode noise on this cut is unknown and may exceed REAP-308's floor.
- **Vendored ggml code drift and size:** it lives in a separate module, so k3_run's 125 KB of code is untouched.

**6. Open questions**
- What n_ple is in this GGUF (`ple.layers`), and what is the drafter's window W?
- Does the decode-QSA spec store the indexer cache as q8_0 (272 B per layer per token, inferred from the 816 MiB buffer in tmp/measure-155-262k-q8_0-dflash-ot18.txt) or f16?
- Do the user's harnesses keep reasoning (Qwen `preserve_thinking`)? That sets where the per-turn checkpoint goes.
- Is a second VRAM KV slot for sub-agents worth about 1.6 GB per 100K tokens of checkpoint budget?
