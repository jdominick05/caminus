Spec: QSA long-context attention in K3 (decode T=1 and verify T<=8, up to 262K context)

Status: read-only study, nothing built or run. Every timing projection is INFERRED. GSQ metadata was read from the GGUF header: 48 layers, QSA on layers 3,7,...,47 (compress_ratios=4), 24 q / 2 kv heads of 256, indexer 4 heads x 128, top_k 2048, rope n_rot 64, freq_base 1e7, eps 1e-6. The indexer tensors are BF16 (q_proj 2560x512, k_proj 2560x128, 39.3 MB in total). Shard 1 holds 29.6 GB of weights; shard 2 holds only the 28.8 GB PLE table.

**1. The reference algorithm (ignite-llama)**

The indexer cache:
- It is a second llama_kv_cache with one key head. Each 256-wide row is `[raw 128 | pooled 128]` (llama-memory-hybrid-idx.cpp:58).
- Its type is **type_k**, so q8_0 in the daily config (:68-71). No V is stored, and the raw keys are never RoPE-shifted (:62).
- No Hadamard: qwen4exp is not in the rotation list (llama-kv-cache.cpp:325-336), and cpy_k does not rotate.
- At 262K this gives exactly 816 MiB, matching the earlier estimate.

Per QSA layer and token (qwen4exp.cpp:771-900):
1. **Raw key.** `k_raw = k_proj(cur)` (BF16 matvec), stored unnormed and unrotated as q8_0 in the raw half of its cell (:790-795).
2. **Pooling.** This runs for each pool the ubatch completes ("new" pools: one every 4 decode tokens, at most T/4+1 per pass, :714):
   - gather the 4 member rows → dequant(q8_0) to fp32;
   - sum the slices in member order, ((k0+k1)+k2)+k3, then `*0.25f`;
   - RMSNorm with eps 1e-6 times k_norm;
   - rope_multi at the M-RoPE position of the pool's **first** member (:804-817; npos at memory-hybrid-idx.cpp:866-871). For text cells, ext.x and ext.y are set from the ubatch's pos dims 2 and 1 (llama-kv-cache.cpp:1155-1158). INFERRED: for text these equal pos, so this is plain text RoPE;
   - stored as q8_0 into the pooled half of the pool's **last** member row (:852-857, :822).
3. **Indexer query.** `q = q_proj(cur)` → [4,128] → RMSNorm(q_norm) → RoPE at the token's own position (:827-833).
4. **Score.** `score[p] = sum_h relu(q_h · pooled_p) * (1/sqrt(128)) + mask[p]`, computed by ggml_lightning_indexer (:837-838).
   - The pooled keys are gathered by get_rows, so they arrive as f32 = dequant(q8_0).
   - Pool p is visible to a token of rank r iff `p < nv = (r+1)/4` (:925). Invisible pools score -inf.
   - n_pool is padded to `max(64, pad64(n_real+1))` (:366-368).
5. **Top-k.** `ggml_top_k(score, min(n_pool, 512))`, output unordered (:842-843).
6. **Selected cells.** The 4 cells of each picked pool, plus a tail of `(r+1)%4` cells: rank r, r-1, ..., with sentinels for the rest (:948-972). n_sel = 4*512 + 3 = 2051 (:754).
   - Picked pools that are not visible are killed by the "live" trick (:872-888).
   - Quirk, which Strata reproduces too: when (r+1)%4 == 0 the tail is empty. The token's own block then has to win top-k for the token to attend to itself.
7. **Attention.** The selection becomes a dense -inf/0 mask over all n_kv cells, added to the causal mask (:890-896). Then normal FA over the whole cache with the Hadamard rotation: Q and K over 256, V in blocks of 64, output rotated back (:903-959).

llama.cpp kernels on HIP:
- **Indexer scoring.** T<8 uses the vec kernel (lightning-indexer.cu:255-395, dispatch :719-737):
  - lane i holds dims 4i..4i+3 as a float4;
  - `acc += q*k` four times (ggml_cuda_mad, common.cuh:752);
  - warp_reduce_sum over 32 lanes;
  - relu, then `score += sum*w` with the heads in order 0..3 (N_HEAD_INNER = 1);
  - then `+ half2float(mask)`.
  - T>=8 switches to the tile kernel (:700), which stages keys in **half**. That is different arithmetic.
- **Top-k.** HIP has no CUB (common.cuh:114). Up to 1024 columns it uses a bitonic argsort, deterministic but with tie order set by the network (top-k.cu:5-12, :309-315). Above that it uses a radix select, which places ties at the threshold in atomicAdd order: **nondeterministic** (:182-209).
- **FA.** The sparse FA path is NVIDIA-only (fattn.cu:131-135). KV_max trims only trailing masked tiles, and the tail is the newest cell. So **llama.cpp reads the whole KV every token**: 13,056 B x ctx across both cards, about 3.4 GB/token at 262K. INFERRED: its long-context decode slows linearly. K3's gather beats it there.

What "bitwise vs llama.cpp" can mean, in layers:
- **(a) Pooling and scoring arithmetic.** It can be reproduced exactly:
  - K3 includes the same common.cuh, so the same mad/fma, the same warp_reduce_sum and the same q8_0 rounding;
  - INFERRED: both builds use the HIP default fp-contract — verify that the build flags match;
  - gate: feed llama.cpp's own dumped `indexer_q-L` and `indexer_k-L` into OP_QSA_SCORE and require a bitwise match to `indexer_score-L`.
- **(b) The upstream inputs are not bitwise.** The BF16 matvec and the RMSNorm differ (~2.4e-7, INFERENCE-PLAN), so q8_0 codes can flip.
- **(c) The selection.** Compare it as a **set**, allowing differences only where a tie or near-tie straddles rank 512.
  - Exact-zero scores happen whenever all 4 head dots are negative. If at most 512 pools score above 0, the boundary falls among the zeros. On radix contexts (n_pool > 1024, ctx > ~4K) llama.cpp itself is then nondeterministic.
  - K3 rule: lowest pool index wins (deterministic, grid-independent).
- **(d) End to end.** D-281, re-measured at long context. llama.cpp's own batch-vs-decode floor will likely grow (tile kernel plus nondeterministic ties).
- **T>1.** K3 uses the vec order for every T, so the T=4/T=8 vs one-token bitwise gate holds even though llama.cpp's own T=8 does not.

**2. Strata as a performance reference (untrusted data, read only)**

- **qsa_block_scores** (qsa_select.hpp:9-12): one warp per (query, block), FP32, relu per head then summed, **no 1/sqrt(d)**. The tail is scored as a "dead" key with a +1e9 bias.
- **qsa_block_topk**: one workgroup per query, a 4-pass radix select over n_bid+1 blocks weighted by cell count, ascending emit, ties to the lowest index. The older per-cell select cost ~0.1-0.2 ms per query at 32K (qsa_select.hpp:4-6).
- **qsa_decode_attn** (qsa_decode_attn.hpp:3-12, CHUNK=64 at .cu:26):
  - split-K: grid (cap/64, n_kv_heads), with all 12 q heads of a kv head served from one read of the chunk;
  - then an LSE merge per head;
  - reads the KV through the page table and the ids, with no gather copy;
  - the old one-block-per-head kernel cost 0.3-0.5 ms per layer at 4K.
- **Indexer append**: f16-rounded raw tail, FP32 pooling, FP32 pooled keys. KV is fp16, int8 with an fp16 scale per 64, or q4_0. KV streaming keeps 32K cells resident, in pages of 4 cells.
- **Numbers**: block scores at 131K, T=1, are 36-73 us per call on a 2080 Ti / 3060 (QSAROWS.md:46-53). The gfx1100 result is the prefill score kernel: 131K, nq=256, 3.07 → 0.74 ms (bench/results/2026-10-11-gfx1100-qsa-scores/correctness.json).
- **Not a correctness reference**: FP32 keys instead of q8_0, int8 with per-64 scales instead of q8_0's per-32, the scale omitted, and +1e9 tail semantics. Its selection will differ from llama.cpp at the boundary.
- **What to borrow**: split-K attention sharing one KV read across the 12 heads, a device step record instead of per-token arguments (qsa.hpp:26-50), and ascending emit with lowest-index ties.

**3. K3 design**

Buffers per card (TP replicates the indexer, so no exchange is needed):
- `IDXRAW[L][cap][136 B]`: q8_0 raw keys for every position. The same bytes llama.cpp stores; kept so that truncation mid-block (prompt cache) can re-pool.
- `IDXPOOL[L][cap/4][136 B]`: q8_0 pooled keys, dense. The llama.cpp row layout would put them at a 1088 B stride and roughly double the lines fetched.
- `SCORE[T][cap/4]` fp32, `HIST[L][T][256]` int, `SEL[T][512]` int.
- A **step record** `[T]{pos, nv, n_tail}` written by the host. The tape is never rebuilt per token.

New ops. The q/k projections use #55's BF16 matvec (**a dependency**); OP_QK_NR is reused with D=128 and n_rot=64.

1. **OP_IDX_APPEND** (T tasks, then pooling): quantize the raw key to q8_0 (the same code as op_kv_append, k3_gguf_ops.h:1001-1025). Then for each row whose `(pos+1)%4 == 0`, pool exactly as in §1.2 and write IDXPOOL[pos/4].
   - It **always runs, from token 0, in prefill passes too**. Otherwise the early pools would not exist when the context crosses 2051.
   - It is off the critical path: a sibling of the q/k/v chain.
2. **OP_QSA_SCORE**:
   - fixed ntask (capacity bucket); each task loops over a stride of pools, with an early return past nv;
   - one wave per pool group, lanes as in the vec kernel, 4 pools in flight (16 VGPRs) plus q for 2 rows (32 VGPRs); for T>2, row passes re-read the keys from L2/MALL. That stays under the 96 cap;
   - writes `SCORE[t][p]` and adds each score's top byte to `HIST[t]` with integer atomics (order-independent, so deterministic);
   - skipped when nv <= 512.
3. **OP_QSA_SELECT** (one task per row):
   - nv <= 512: emit 0..nv-1 (the identity);
   - otherwise: find the threshold bin from HIST, refine the candidates in that bin with a 3-byte radix (1 KB of LDS, reusing MVP), then do one ascending emit pass over SCORE with block-prefix compaction: `>` threshold always, `==` threshold in index order up to the remaining count;
   - then zero HIST. A bail retry must memset it from the host.
4. **OP_ATTN_SEL**: op_attn_dec (:1027-1085) with the cell list `4*SEL[i]+j`, then the tail cells, in ascending order, with the same 512-chunk online softmax over list positions. **For ctx <= 2051 the list is the identity, so it is bitwise equal to today's OP_ATTN_DEC.**
   - Hadamard, KV append and the gate stay as they are.

Tape order per QSA layer:
- `{q,k,v matvec | idx_k matvec → IDX_APPEND | idx_q matvec → QK_NR(idx)}`
- → SCORE (dep: QK_NR idx; dep2: IDX_APPEND)
- → SELECT → ATTN_SEL (dep2: KV_APPEND).
- This adds about 2 serial segments per layer.

Verify T<=8 and lazy commit:
- Every row has its own nv and tail.
- All T raw keys are written and their pools pooled before any score is computed, so row i sees pools completed by rows < i.
- A pool completed by a rejected draft is never visible: visibility needs rank >= its last member. It is re-pooled when its block completes again. Accepted rows only ever saw pools made entirely of accepted tokens.

Code size: about 4 small ops (INFERRED ~6-10 KB on a ~125 KB k3_run). SELECT is cold (12 calls per token). Do not move it out of line: the router experiment shows calls cost registers (47 tok/s).

**4. Memory per card (TP, q8_0, 12 QSA layers)**

| ctx | KV, 1 kv head (6528 B/tok) | IDXPOOL (408 B/tok) | IDXRAW (1632 B/tok) | Total | llama.cpp-layout indexer instead |
|---|---|---|---|---|---|
| 32K | 204 MiB | 12.8 MiB | 51 MiB | 268 MiB | 102 MiB |
| 128K | 816 MiB | 51 MiB | 204 MiB | 1071 MiB | 408 MiB |
| 262K | 1632 MiB | 102 MiB | 408 MiB | 2142 MiB | 816 MiB |

- Scratch is about 2 MiB (SCORE at 262K, T=8).
- One card without TP: KV is 3264 MiB at 262K.
- Fallback if VRAM gets tight: keep raw keys only in a 16-slot ring and save the 3-key tail with each GDN checkpoint. This saves 408 MiB per card at 262K.

**5. Cost per token vs context (per card, T=1)**

- **Up to 2051 tokens**: only k_proj (7.9 MB) and the append are added. The attention is today's.
- **Above 2051**:
  - pooled keys: 408·ctx, so 13 / 53 / 107 MB at 32K / 128K / 262K;
  - selected KV: 2051 x 544 x 12 = 13.4 MB, constant;
  - indexer weights: 39.3 MB;
  - the SCORE array stays L2-resident.
- **Bandwidth**: about 66 / 106 / 160 MB, roughly +0.08 / +0.13 / +0.2 ms.
- **Chain**: +2 segments x 12 layers x 5-10 us = +0.12-0.24 ms. SELECT at 262K is about 5-15 us per layer.
- **Today's per-head attention at 2051 cells**: INFERRED ~15-25 us per layer (byte-wise q8_0 reads in 24 tasks) — measure it first.
- **Projection (INFERRED)**: 10.45 ms → ~10.7 (4K), ~10.8 (32K), ~11.0 (128K), ~11.3 ms (262K).
- **llama.cpp**: +~4 ms per token at 262K just for FA over all cells (INFERRED).

**6. Steps, gates, effort**

| Step | Work | Gate | Effort |
|---|---|---|---|
| S0 | Extend gg_golden: n_ctx argument (fixed at 512 now, tools/reference/gg_golden.cpp:76), a name filter for `indexer_*-L`/`kqv_out`, a start step, logits only for long runs. Build a numpy QSA reference in tests/. | The numpy reference matches the llama.cpp dumps on the same inputs | 1.5 d |
| S1 | Measure from llama.cpp dumps at 8K/32K: the zero-score rate, the ties at rank 512, the score gap at the boundary, and llama.cpp run-to-run selection drift | Report only; this decides the tolerance in S4 | 0.5 d (needs GPUs) |
| S2 | IDX_APPEND plus pooling (k_proj through the #55 BF16 matvec) | Bitwise vs numpy given the engine's own k_raw. With llama.cpp's `indexer_k_raw` fed in, `indexer_pool_k_new` matches bitwise | 1.5 d |
| S3 | QSA_SCORE | Fed with llama.cpp's indexer_q / indexer_k dumps: **bitwise** to `indexer_score`. Bitwise across grids (race check) | 1.5 d |
| S4 | QSA_SELECT | Bitwise to the numpy lexsort rule. Set-equal to `indexer_top_k` except at counted tie-straddles. Bitwise across grids | 2 d |
| S5 | ATTN_SEL | ctx <= 2051: bitwise = OP_ATTN_DEC. Above that: <= 2e-7 vs float64 on the same selection | 1 d |
| S6 | Tape wiring: single card plus TP, step record, T<=8, lazy commit | T=4/T=8 vs one-token bitwise at positions crossing 2051, 2052 and every pool phase. A rejected draft that crosses a block boundary. TP replica check extended to SEL (0 mismatches) | 2 d |
| S7 | Long-context end to end | D-281 at 4K/8K/32K (a llama.cpp one-token teacher-forced reference, logits only). Needle and rare-token copy recall at depths 10/50/90% of 8K/32K/128K/262K vs llama.cpp (copy collapse can stay invisible to KLD). ms/token curve 2K-262K. Soak run crossing 2051 and every bucket boundary, with 0 bails | 2 d |
| S8 | Speed (optional): split-K ATTN_SEL (chunk x 4-head group tasks, then a merge), skip q_proj below 2051 | Bitwise vs its own reference; D-281 unchanged | 2 d |

Total: about 12-14 days with the gates (core ops 7-9). This is above the Strata audit's 5-8 days because of the gates and tooling. Prefill (T=512 chunks using WMMA scoring, not bitwise) is a separate spec but must use the same IDXRAW/IDXPOOL layout.

**7. Risks**

- **Boundary ties make bitwise selection impossible.** If S1 shows many zero scores at the boundary, the gate becomes set agreement measured against llama.cpp's own run-to-run drift.
- **Upstream non-bitwise inputs** (BF16 matvec, RMSNorm) flip q8_0 codes of pooled keys, and so the selection, near the boundary. Only D-281 and needle tests can judge it.
- **SELECT latency** on one workgroup at 262K could reach 20-50 us per layer if L2 reads are slower than INFERRED. The fallbacks are histogram fusion (already planned) or a pool split across cards (exchange of 512 candidates, about 42 us of XCHG).
- **Register pressure** from multi-row scoring at T=8: keep 2-row passes.
- **A new segment kind with work that grows over time**: capacity buckets must keep the 200 ms wait bound and the KFD bail retry correct (HIST re-zeroing).
- **A daily server run at f16 KV**: with f16, the indexer cache is f16 too, and S2/S3 need an f16 variant.

**8. Open questions**

1. The S1 numbers (zero-score and tie rate). They set the S4 and S7 tolerances.
2. Keep all raw keys, or a ring plus checkpoint tails? This depends on how the prompt cache truncates.
3. Is a pool split across cards worth it above 128K, or is replication enough?
4. The re-measured D-281 floor at long context, and whether the user accepts it as the bar there.
5. Confirm the text M-RoPE ext positions and matching fp-contract flags with the S2/S3 dumps (both INFERRED above).
6. Does the HF reference share the "own block must win top-k" quirk? llama.cpp and Strata agree on it; llama.cpp is the bar either way.
**9. Depth investigation (11-05)**

Outcome: **no defect** (impl:qsa-depth, independently re-verified; evidence `$ARTIFACTS/impl-qsa-depth/report.json`, tool `tools/eval/qsa_depth.py`). The question was whether the G4 tail at depth (our prefill-then-decode vs our own one-token decode) comes from a fault in our QSA path or from the selection's normal near-tie sensitivity.
- **Selection disagreement matches llama.cpp's own** (the pre-registered test). Pools of the 512 that one selection picks and the other does not, per position and layer, ours (prefill-then-decode vs one-token decode) vs llama.cpp's own (the same pair):
  - 8K: 15.38 vs 15.79 (llama.cpp's 95 % bootstrap 15.13 .. 16.57);
  - 32K: 23.13 vs 23.25 (22.22 .. 24.40). Every layer within.
- **The inputs check clean:**
  - pooling and the backfill's re-pool are bitwise;
  - the boundary is a near-tie band (mean score gap at rank 512: 0.002 - 0.003, ours and llama.cpp's alike) with no exact ties on our decode and prefill-then-decode paths (2 of 3,072 rows at 32K on our in-batch path and on llama.cpp's decode);
  - the indexer query's perturbation between the two paths is llama.cpp's size (relative 0.047 ours vs 0.045 llama.cpp at 32K).
- **The KLD tail is position-intrinsic:** our worst positions are llama.cpp's own (4 of its top 5 at 32K), and the selection differences do not explain it (Spearman 0.24 at 32K, ~0 at 8K).
- **llama.cpp is not run-to-run reproducible at 32K** (every row's FNV differs between two runs). Its floor mean spans 0.00113 .. 0.00167 over the six run pairings measured then; the stored 0.00113 (`g4_depth.json`) is the low end. Ours: 0.00128 (bf16 WMMA outputs, the default) / 0.00118 (fp32).
- The rubric's verdict was "decision" only because its "tail explained by selection" clause failed.

**The user's ruling (11-05):**
1. "No defect" accepted; the binding depth rule stays: G4b at depth (vs llama.cpp's prefill-then-decode at ub 2048, the D-281 bar 0.0076 / 95.9 %) plus the G4a bootstrap rule (vs our own one-token decode, against llama.cpp's floor's 95 % bootstrap upper bound).
2. The long-context sweep's method: at 64K, the floor from 3 llama.cpp one-token runs, pooled over run pairings; at 128K and ~262K, G4b plus needle / rare-token copy recall vs llama.cpp bind; the one-token D-281 at 262K is reported from a single run.

**The pooled 32K floor** (sweep-262k step 1, CPU only, from the runs already on disk; `tools/gates/g4_floor.py pool`):
- Runs (positions 32,768 .. 33,023 of qsa/lc/d32768, all at ub 2048): 4 one-token (impl-prefill-s7's ref32768, impl-qsa-depth's dec32768, verify-qsa-depth's ref32768_r2 and dec32768_r2) x 3 prefill-then-decode (impl-prefill-s7's pd32768_r1, impl-qsa-depth's pd32768_r1 / r2): 12 pairings. The qsax dump runs are valid floor runs: at 8K (deterministic there) their rows equal the stored tfx runs by FNV.
- Method: per position the mean KLD over the pairings; its mean; the 95 % bootstrap interval resampling positions (the pairings of one position are not independent samples); top-1 the mean over pairings. The tool reproduces the six earlier pairing means to 1e-9 (control).
- Result: **mean 0.00140, 95 % bootstrap 0.00024 .. 0.00324, top-1 99.80 %**; per-pairing means 0.00113 .. 0.00167 (sd 0.00016); max 0.186 at 32,982 (every pairing's max is there). File: `$ARTIFACTS/impl-prefill-s7/g4/g4_depth_pooled_N32768.json` (new, beside `g4_depth.json`, which is unchanged and still what `tests/prefill/test_prefill_s6.py g4d` gates on).
- Our G4a at 32K (0.00128, top-1 100 %) lies below the pooled mean and well inside the pooled bar (0.00324 / 98.80 %).
