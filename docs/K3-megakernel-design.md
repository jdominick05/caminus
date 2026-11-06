# K3: a persistent decode engine for our models on gfx1100 (design, 2026-10-31)

K3 is a **model-agnostic persistent decode engine**. It is not an S1 kernel. One launch per token, or per verify block,
runs a tape of tasks that a host planner builds from a model description. The engine core (grid, sync, int4 g128
matvec, row ops, residency and staging, state and KV, verify, instrumentation) is meant to survive the architecture
changes under study: per-layer token tables, hashed n-gram memory, chunk-sticky MoE, GQA 16:1, NoPE global layers,
gated or bounded attention, conv-hybrid blocks, and the ~2B N track. A small op vocabulary plugs into the core.
Today's S1 (155M; 9 GDN + 3 SWA attention parallel blocks, host-routed slices, adaptive head) is the first instance
and the bit-exact test vehicle, because it is the only trained model. Tuning goes into the shared engine, not into
S1's shapes.

Sources: four investigations from 2026-10-31.
- ignite: FN-R1 megakernel archaeology, the ignite-llama fork (its development branch then; the file cited below is in the
  published fork, github.com/jdominick05/ignite-llama at e4e3f8eda).
- math: S1 decode numerics under the accepted stack.
- budget: traffic, sync and bandwidth model.
- prior: 2025-26 megakernel prior art.

Also used: the private research notes of the S1 work (the training-stack research, the MoE plan, two architecture surveys of
2026-10-31); they are not published, and the citations below that point into them are kept as written.
"(est)" marks unmeasured numbers. No GPU was run for this document.

---

## M1 measurements so far (2026-10-31, XTX) and the kill-rule call
| Constant | Measured | Test |
|---|---|---|
| Decode matvec numerics | **bitwise** equal to gemm_tn_iu4g on every S1 shape (random, all -8, real actq rows; fp32 and bf16) | tests/test_k3_matvec.py |
| Cache read bandwidth, b128 | **3.35-3.39 TB/s** up to 88 MiB at 192 WGs (3.25 at 92 MiB; 2.2 at 96 WGs); 1.8 at 96 MiB; VRAM 0.86-0.91 beyond 112 MiB; nontemporal loads same speed | tests/bench_bw.py |
| Hand-off, idle stages | 0.73-0.96 us | tests/bench_sync.py |
| Hand-off under streaming load | **0.36-0.62 us** when tasks >= WGs (waits overlap the stream tail); 1.34 us with half the WGs idle (96 tasks, 192 WGs) | tests/bench_sync_load.py |
| Claim round per WG | ~0.5 us (256 KB stage: 0.61 us at 96 tasks vs 1.37 us at 384, 192 WGs, no waits) | same |
| Chained stage, 3.4 MB (a GDN block projection) | ~2.3 us (192 tasks, 192 WGs); 1 MB stage ~1.6 us | same |
| Naive one-wave-per-row matvec | 0.10 TB/s (correctness form only) | tests/bench_bw.py |

Per token with naive streaming: 12 x (stage 1 2.3-3.3 + stage 2 1.6-1.9) + head ~5 + launch ~3 = **~56-68 us (15-18k tok/s)**,
before prefetch-before-wait, chunked claims and more rows per task. **Kill rule: inside the first band (<= 66.7 us): aim for
15k.** The hot set must stay <= ~90 MiB (the cache knee is between 92 and 96 MiB), so the head format decision (int4 per
row vs int8 per row; Sec. 3.9) is a real constraint: raised with the user 2026-10-31, needed before M4.

### M1 engine skeleton, measured (2026-10-31, XTX at gpu-high, commits 17f2518 -> 5d5ad7d)
`kernels/k3.hip` runs a tape of segments in one persistent launch, with claim/done counters, prefetch-before-wait, a
sticky status word and reset by the last workgroup. `tests/test_k3_engine.py`:
- The matvec op is **bitwise** equal to gemm_tn_iu4g on all 7 S1 shapes × 5 task sizes, including partial last tasks.
- A 400-segment chain and dependent matvecs pass at **1x, 3x and 5x oversubscribed** grids; the status word is never set.
- The token tape's outputs are bitwise too.

The synthetic S1 token tape has 25 dependent matvec segments: 12 × [block projection 7680 or 5248 × 896 -> o + down
896 × 2944] plus the 24,576-row head, 66 MB in all. Configuration: grid 192, 8 rows per wave in flight (109 VGPRs),
40 / 16 / 64 rows per task. **Result: 77.6 us (12.9k tok/s).** That passes the 6,960 gate (143.7 us), misses the 15k
target (66.7 us), and misses this doc's matvec-only estimate. Per segment, from `tools/k3_trace.py` (k3_trace build,
medians over 20 launches):

| Part | Measured | Doc assumption |
|---|---|---|
| Hop (dependency's last publish -> waiter released) | **0.3-0.4 us** | 0.40 central (0.8 idle): confirmed |
| Activation-code load after the wait (L2) | 0.5 us | in "entry" 0.30-0.45 |
| Dots + P_g to LDS + barrier | 0.45-0.65 us (head 1.3) | - |
| FMA chain + store | 0.45-0.55 us (G = 23: up to 1.1) | in "epilogue" 0.20 |
| Publish (release + add) | 0.2 us | - |
| Segment span, stage 1 / S2 / head | 3.1-3.6 / 2.6-3.0 / 5.2-6.0 us | 2.0 / 1.05 / 4.0 (overlap) |

**Diagnosis: weight delivery, not sync.** Weights reach a workgroup's registers about 2.2-2.8 us after it claims a
stage-1 task under load. Workgroups that claim within about 1 us of the dependency finishing stall on their own
prefetch: their activation phase takes 1.3-1.6 us instead of 0.3. This holds even when every task is claimed before
its dependency finishes (the 57 / 16 configuration: no late workgroups, same span). Each workgroup issues one task's
loads and then waits, so bytes in flight per CU stay well below what the 3.35 TB/s streaming measurement had. Per
segment, delivery is about 1.2-1.5 TB/s (0.85 TB/s over the whole tape); Sec. 5.2 assumed 3.0.

What did not help:
- Static first claims (the claim atomics are not the limit).
- The tape cached in LDS.
- Grids of 288-384, with 12 waves per SIMD.
- Bigger tasks.
- 16 rows per wave.

**Re-projection (records the cost-model miss, Sec. 3.12).** The real token adds, on top of the matvec tape:
- about 28 MB of bytes: GDN state read + write, KV, the slices;
- the entry row ops;
- the R / ATTN chains.

At today's delivery rate that projects to roughly 95-115 us (9-10.5k tok/s): above the gate, short of 15k. The kill
rule's first band (<= 66.7 us) is not reached by the skeleton as built.

Levers, in the order to try (M5, or earlier if M2 confirms the diagnosis):
1. **Deeper prefetch.** Claim one segment ahead and stage that task's weights in LDS. At 4 WGs per WGP that is 32 KB
   per WG; a stage-1 task is 18 KB and an S2 task 23.5 KB. Or use loader waves streaming into an LDS ring (wave
   specialization).
2. **Warm the next segment into L2.** Workgroups idle during the short S2 segments warm the next segment's weights into
   L2 (6 MB).
3. **Data-carrying flags.** Use NCCL-LL-style words (data + token epoch) so that hop + activation load becomes one
   round trip. Measure this in M2 with real producer outputs: the tape's activations are independent vectors, so its
   0.5 us load is not yet a true producer -> consumer read.

## 0. Conflicts between the investigations, and what this design adopts

| # | Topic | What the sources say | Adopted, and the evidence that decides |
|---|---|---|---|
| C1 | Sync primitive | All four reject arrival barriers. | Use **claim/done counters only**. Arrival barriers dropped 20 of 168 blocks on a display-driving card (ignite commit 1a188d150 (published as 0bc23d0e7)). The race showed up in 2 of 14,396 launches, and only the bitwise shadow compare caught it; KLD did not. Under load, Reeder measured barriers at 3-5 us against 0.5 us idle. The XTX also hosts the RDP compositor, so partial residency is the normal case on both cards. |
| C2 | Grid size | ignite ships BPW 8, i.e. 384 WGs. Budget recommends 192. | **192 WGs x 256 threads (4 per WGP)**. Measured: a sync costs 0.36-0.41 us at 1-4 blocks per WGP and 0.94 us at 8 (SCOREBOARD.md:81-82). More WGs only add sync cost. M1 re-measures this under load. |
| C3 | Serialized hand-offs per token | Plan / ignite: 48. Prior: ~39 (3 per layer, P->R->O). Budget: 24 (schedule A) or 36 (B). | **Design for schedule A and budget for B.** Only two waits per block are grid-wide (stage 1 -> stage 2 -> next stage 1). The mixer chain (R) runs inside stage 1 behind per-head counters. It costs nothing on the critical path only if it ends before stage 1's weight fetch does. Budget estimates 1.25 us for the chain vs 1.6-1.8 us for the fetch (unverified). M2 measures it, and that measurement decides A or B. Prior's "3 hops, none removable" counts the same chain; it is one hop more only when R does not hide. |
| C4 | Wait cost | 0.36-0.94 us idle (arrival barrier, mk-barrier.cpp). Claim/done wait: never timed. Reeder: about 1 us per hand-off under load. | **Measured 2026-10-31 (tests/bench_sync.py, commit b5bdb71):** claim/done 0.73-0.96 us per stage on the XTX at 1-4 WGs per WGP (idle stages), arrival 0.70-0.88 us, XT within ~10%. Use **0.8 us as the central value** (the 0.40 below is superseded): schedule A's 24 hops cost ~19 us, giving ~50-55 us/token with prefetch overlap and ~71 us serial. Under-load cost remains an M1 measurement. |
| C5 | Counter reset | ignite: the last block out resets every counter. Prior: monotonic epochs. | **Monotonic 64-bit counters, no reset.** The last WG out writes the next launch's per-counter bases into the control block in one flat pass (Sec. 3.3). That stays correct across tape variants with different task counts. The last-out reset makes every block walk every stage, which adds tail latency (ignite risk 4). mk-barrier.cpp's never-reset counter replays correctly in HIP graphs. |
| C6 | Spin loop | ignite's mk_wait does an acquire load on every poll. On gfx1100 that emits `buffer_gl1_inv; buffer_gl0_inv` on each iteration (prior, disassembly of tu0034.co). | **Relaxed agent-scope polls with s_sleep, then one acquire fence per wave after exit.** Keep ignite's release fence in every wave before `done`; this is needed in WGP mode, where a block spans two CUs with separate L0s. M1 A/B-tests the polling variants. |
| C7 | A4 vs "W4A8 stays" | Plan: kill test 2 keeps W4A8 (dot4_iu8 beats dot8_iu4). Math / prior: A4 codes are what training computed. | **These do not conflict.** The plan picked the instruction; the numerics are a separate choice. Feed **A4 codes sign-extended to int8** into `v_dot4_i32_iu8`. The int32 per-group dot then equals INTLINEAR's exactly, at dot4 speed. A8 would change the numerics and buy no speed. |
| C8 | K0 correctness | Plan: all three modes pass (rel err <= 9e-6). Log on disk: `FAIL max rel err 1.00e+00`. The .hip was modified one minute after the log. | **Unproven.** The throughput numbers stand. M1 re-proves exactness at the S1 layout before anything builds on it. |
| C9 | GDN port | Plan:59 says the FN-R1 GDN megakernel "only needs a silu flag". | **Wrong for S1** (ignite evidence). FN-R1 follows ggml, not fla, in six ways: decay order (kv on the undecayed state), L2 form (rsqrt(sum/S+eps)*x*scale vs x/sqrt(sum+1e-6)), where the q scale is applied, the fp32/bf16 rounding points, a sigmoid output gate where fla uses swish, and Q8_0 MMVQ alpha/beta where S1 has a bf16 Linear. Its conv state holds 3 values where fla keeps 4. Only the [v][k] state layout matches. Port the task structure; write the math new, from fla's order. |
| C10 | Head precision | prior: int8 (qat.py default). Budget: int4 per row. Trained head: bf16 (42 MiB). | **int4 per row (MSE clip) + fp32 row scale, 11.1 MB.** This is the recorded decision: int4 costs +0.005 to +0.019 nats (the research notes). It is a deliberate deviation from the trained function and has its own statistical gate (Sec. 2). A bf16 head pushes the hot set past the 96 MB cache. |
| C11 | Per-token traffic and hot set | Math: 69 MB (null route). Budget: 92.3 MB (q8_0 KV). Prior: 80-104 MB (int8 head). Plan hot set: 83.2 MiB (fp16 scales). | **Recomputed here** with an int4 head, fp32 g128 scales and bf16 KV: **80.6 MB per token on the null route, 93.7 MB with the 768-wide slice routed. Hot set 74.3 / 86.9 MiB.** fp32 scales are required by the exact g128 chain (+2.0 MiB over the plan's fp16 assumption). |
| C12 | KV format | Plan: q8_0. Training: bf16. | **bf16 in v1** (the training numerics; costs 1.4 MiB more). q8_0 is a priced cost-model entry, not a default. |
| C13 | Training stack | Math assumed stack I. The research notes: the accepted stack is now **F** (I + F1-F4). Gate 4 passed E (F + FUSEENTRY), but E stays opt-in. | **The forward of F equals the forward of I.** F1 WCACHE fake-quantizes once per step, which changes only gradients
(inferred: the source states only that gradients differ, and the core is frozen anyway); F2 and F3 are bitwise; F4 is backward-only. The decode contract targets the exported codes and scales, so the training fusion does not move it. The block entry follows **block_entry's (E) order**. Gate 4 accepted that order as model-equivalent: within 0.0036 nats on every section, coding +0.0009. Torch's internal order cannot be reproduced anyway. |
| C14 | "M0: decode equals prefill" | Task milestone text. | **Not bitwise for GDN.** The chunk path rounds to bf16 at six points (q-hat/k-hat, (I+A)^-1, w, u, v_new, state snapshots), decays with exp2 of an RCP_LN2-scaled cumsum, and runs bf16 WMMA dots, which are not exact on gfx1100. The recurrent step keeps fp32 state. So M0 is bitwise per op wherever a bitwise reference exists, and statistical for the whole model (Sec. 2, Sec. 6). |
| C15 | Split-K | Budget: never across WGs. ignite: allowed with int32 partials and an in-order owner. | **Both are exact.** Default: split within a WG (int32 partials in LDS; the owner lane runs the chain). Cross-WG int32 partials plus a last-finisher owner chain is an **opt-in** layout for narrow stages whose task count falls below the grid. A float split-K is never allowed. |
| C16 | Interpreter vs descriptor loop | Prior: a general opcode interpreter costs VGPRs and buys about 10%; a data-driven loop over layer descriptors is enough. The task asks for model agnosticism. | **A small op table:** dispatch on the op id only at task boundaries, and make op bodies compile-time variants generated from the model description (Sec. 3.4). This is not a general interpreter. Numerics are protected by contract(off) plus explicit fmaf, not by code shape (ignite's KLD trap: an untaken runtime branch changed results). |
| C17 | Escape check | Budget: at the start of the next launch. ignite pattern: last-block selection. | **Both.** In this launch, the hot argmax resolves through a last finisher with no wait. The tail runs at the start of the next launch, and only when the escape flag is set (Sec. 3.9). |
| C18 | Head row count | Task text: 24,585. Code: 24,577 (s1.py:49). Training GEMM pads to 24,584. | **24,577** (code). |
| C19 | Which card raced | The commit message says the XT. The race log shows dev0, which is probably the XTX. | **Not needed to settle.** Both cards run OVERSUB and shadow-compare tests. |

---

## 1. Goal, targets, scope

### 1.1 Targets (XTX, gfx1100, 96 MB Infinity Cache)
| Item | Target | Basis |
|---|---|---|
| S1-today batch 1 | **>= 15k tok/s (66.7 us/token)**; acceptance gate >= 10x K1 = 6,960 tok/s (143.7 us) | the MoE plan's K3 speed gate; K1 = 696 tok/s |
| Model estimate | central 45.3 us (22.1k) with prefetch overlap / 61.4 us (16.3k) serial; pessimistic 63-69 us | Sec. 5 |
| Verify blocks T = 4 / 8 / 16 (dot4) | <= 60 / 80 / 130 us per block (est 51 / 68 / 117 us = 78k / 117k / 136k positions/s) | budget model; iu4 WMMA at T = 16 (~84 us) stays K6 |
| Hot set | <= 88 MiB (plan); computed **86.9 MiB** with slice 768, bf16 KV, int4 head; 74.3 MiB null route | Sec. 5.1 |
| Host slice switch | <= 50 us (plan gate); est 14.5-19.7 us one-time cold read of 13.2 MB | VRAM 0.67-0.91 TB/s |
| Cost model | predicts S1 us/token within +-15% of measured (M4/M5) | Sec. 3.12 |

The **XT is a correctness target only in v1**. Its 72 MiB budget (80 MB cache) cannot hold the 74-87 MiB hot set. It
runs the OVERSUB and shadow-compare tests, since it drives a display.

### 1.2 Out of scope for v1
- Prefill: decode-only tests step through tokens; serving prefill is fla's chunk path or K5.
- Multi-sequence batching.
- Sampling: greedy only. Top-p needs the full LSE over 24,577 logits and fires the tail more often.
- Ship modes:
  - q8_0 KV;
  - an int8 or int4 tail;
  - an int8 embedding;
  - fp16 group scales;
  - the QuaRot fold.
  Each changes the trained numerics and needs its own gate.
- iu4 WMMA for verify (K6).
- Semantics past 2048 positions: RoPE positions stay absolute, and nothing in training covers this.
- The XT as a speed target.

### 1.3 Future features the engine must take without a rewrite
| Feature (source) | How it plugs in |
|---|---|
| Chunk-sticky MoE: experts chosen once per 64-token chunk, lag-1 or lag-2, staged into the cache (sparse Sec. 3B) | `EXPERT_MV` op reading an expert-set descriptor; `ROUTE_CHUNK` op on boundary tokens; `STAGE` tasks; residency class `staged` |
| Per-token routed experts (N-scale) | `ROUTE_TOKEN` (router GEMV + last-finisher top-k) + `EXPERT_MV` with an on-device descriptor; +1 hop per MoE layer |
| Per-layer token embeddings (PLE) and hashed n-gram memory (Engram-lite) (sparse Sec. 3A, frontier L8) | `TABLE_LOOKUP` (cold gather, no dependencies, issued at launch start) + `INJECT` rows in a fused row group; a device token ring for n-gram hashing |
| GQA 16:1, NoPE global layers, sliding or no window, sigmoid output gate, softcap, log-position q scaling (frontier L3, L5) | `ATTN` op parameters; the gate's rows concatenate onto the q projection; the epilogue applies them |
| Bounded numerics: static A4 scales, SiTU-GLU (frontier L6) | the `ENTRY` / `ACTQ` mode (dynamic per-row or static) and the MLP activation parameter |
| Conv-hybrid blocks | `SHORT_CONV` op (width and channels as parameters, ring state) |
| Attention Residuals (frontier L4) | an `ENTRY` pre-mix hook (depth softmax over stored block outputs) |
| Looped core (sparse Sec. 3D) | the tape repeats segments with per-recursion state buffers |
| N track (~2B, VRAM-bound) | the same engine with weight class `cold` (nontemporal streaming); hop costs become minor |
| Replay verify for GDN (frontier L9) | built in from M6 (Sec. 3.10) |

---

## 2. Numerics contract

The engine's bitwise target is the **M0 per-node decode reference**. That reference is built from the *same shared
device functions* (`kernels/common/*.h`, Sec. 4), and the engine's SPLIT mode, one launch per segment, must also match
it. The reference is in turn tied to training per op, as in the table below.

Test levels:
- **L1:** unit, per op, against the training kernels or the captured Triton code objects.
- **L2:** per step, the engine against the M0 reference: memcmp of every output, all state and the KV, every step.
- **L3:** whole model, statistically.
  - The measured noise band on the golden window: soft CE 1.7910-1.7969, top-1 64.50-65.09% (the research notes).
  - Teacher-forced decode: prefill t tokens in chunks, then one decode step, compared with a prefill of t+1.
  - The rare-token copy gate (>= 99%).

| Op | Reference | Class | Test |
|---|---|---|---|
| Embedding gather (fp32 row) | `embedding_f32` (core.hip:23-28) | bitwise | L1 memcmp |
| Block-entry RMSNorm (eps 1e-6) | `block_entry_fwd` order (actq.hip:239-273). `ss = fmaf(x,x,ss)` strided by 256, wave xor 16..1, 8 wave sums in order, `rsqrtf(ss/896+eps)`, `(x*rstd)*w` | bitwise vs E; <= 1 fp32 ulp vs torch (stack F) | L1 vs the FUSEENTRY row kernel |
| Activation quant (FWHT-128, row amax, int4 codes) | `actq_fwd_src` (actq.hip:97-132). bf16 input; butterflies at distance 1, 2, 4..64 in order; `u = bf16(fwht*0.08837890625)`; `s = max(amax/7, 1e-8)` by IEEE division; `rintf(u/s)` clamped to [-8, 7] | bitwise | L1 codes and scales memcmp on random and golden rows (896, 2048, 768) |
| GDN a/b projections (bf16 [7,896]) | block_entry's a/b order (actq.hip:256-273) | bitwise vs E; <= 1 bf16 ulp vs rocBLAS (F) | L1 |
| **int4 g128 matvec** | `gemm_tn_iu4g` (gemm.hip:244-350). Exact int32 `P_g` per 128-group; `acc = fmaf((float)P_g, gs[n*G+g], acc)` for g = 0..G-1 in order; `*s_x`; bf16 RNE | **bitwise for every element at any N, M, tile or concatenation** | L1: pad M to 64 with copies of the row and run the GEMM. Every S1 shape: N 7680/5248/896/1536, K 896/2048/768 |
| MLP silu*mul | `silu_mul_bf16` / `silu_mul_actq_fwd` (core.hip:48-55, actq.hip:307): `bf16(bf16(g/(1+exp(-g)))*u)` | bitwise | L1 |
| Residual | `residual_slices_fwd` (fuse.hip:19-32): `((x+m)+mlp)+(routed ? ys : 0)` in fp32 | bitwise | L1 |
| RoPE at position p | `rope_pair` (attn.hip:99-122); every product and sum rounded to bf16 | bitwise | L1 |
| Short conv + silu (W = 4, the 2688 q\|k\|v channels) | captured `causal_conv1d_fwd_kernel` code object, prefill path. The taps are probably FMA-contracted (74 v_fmac + 64 v_fma); this is **not** fla's `causal_conv1d_update`, which sums with tl.sum | target bitwise; fall back to <= 1 bf16 ulp, documented | L1 against the code object on [3 history + current] |
| L2 norm of q/k (eps 1e-6), beta = sigmoid(b), g = -exp(A_log)*softplus(a+dt_bias) | captured chunk-path Triton kernels. **q-hat and k-hat rounded to bf16, as training does** (math recommendation) | target bitwise; <= 1 ulp if the tl.sum or libm lowering differs | L1 against the code objects; record any ulp diff |
| **GDN recurrent step** | fla `fused_recurrent` order: `S *= exp(g)`; `v' = beta*(v - S k-hat)`; `S += v' k-hat^T`; `o = S (q-hat*128^-0.5)`; o stored bf16; fp32 state [v][k] | bitwise vs M0 (same device function). Within <= 4 fp32 ulp (est) of fla fused_recurrent **fed the same bf16 q-hat/k-hat with `use_qk_l2norm_in_kernel=False`** (fused_recurrent.py:197), so only the tl.sum order differs. **Statistical** vs the chunk path | L1 vs that oracle (state and o, 1 and 2048 steps). Report separately the drift against fused_recurrent with its own fp32 in-kernel L2 norm. L3 teacher-forced vs `gdn_fwd_h` with h0/ht |
| Gated output norm (eps 1e-5, swish gate) | captured `layer_norm_gated`: `y = (o*rstd)*w; y = (y*g)*sigmoid(g)`, bf16 | target bitwise; <= 1 ulp | L1 |
| Attention decode (14 q / 2 kv heads x 64, window 2048, scale 0.125) | an engine-defined split: 16 chunks of 128 keys per kv group give partials (m, l, o[64]) in fp32, merged in chunk order by the last finisher; o bf16 | bitwise vs M0 (same split). vs training (AOTriton rounds P to bf16): 1.1-1.3e-3 relative (measured for attn.hip) | L1 vs `attn_causal_bf16` within tolerance; L2 bitwise |
| Slices (gu, silu*mul, down) | training used bf16 proxy GEMMs whose order depends on the batch's row count (s1_patches.py:490-520). Decode uses **iu4g** | bitwise vs M0; statistical vs training | L3. Open: switch training to iu4g by padding slice rows to 64 |
| Final RMSNorm | `rmsnorm_f32` order (core.hip:30-46) | bitwise vs that kernel; <= 1 ulp vs torch | L1 |
| Hot head (int4 per row) | engine-defined: `L = bf16(rs_n * sum(code * bf16(h)))` with a fixed reduction order (W4A16 by `v_dot2_f32_bf16`; int4 to bf16 is exact) | bitwise vs M0. **Deviation from the trained bf16 head** | L3: delta soft CE <= 0.02 vs the bf16 head (M0 also runs a bf16 head); copy gate >= 99% |
| Argmax / top-1 | 64-bit key `orderable(L) << 32 \| ~idx`; ties go to the lower index (torch.max's documented rule; not verified on ROCm) | bitwise | L1 with forced ties |
| Escape and tail | `tail_lq = (L_esc + tv) - lse(z)`; tail wins only if `tail_lq > hv` (strict). Skip the tail when `L_esc < hv`. That is exact: `lse >= max`, and a bf16 gap far exceeds one fp32 rounding (math derivation) | bitwise vs M0; exact skip rule | L1 with forced escapes; count the fire rate |

Rules that keep this true:
1. **No implicit FMA contraction anywhere.** `k3.hip` and the shared headers build with `-ffp-contract=off`. Where a
   reference fuses (block_entry's `ss += x*x` under HIP's default contraction, the conv taps), the shared header writes
   `__builtin_fmaf` explicitly. M0 rebuilds the *training* kernels from the same headers, and they must give the same
   bits as today's code objects. This removes the class of failure behind ignite's untaken-branch KLD change and its
   1-ulp contraction trap (SCOREBOARD.md:183-190, 217-222).
2. **Optional paths are compile-time variants.** Slices on or off, the verify batch T, head mode, RoPE or NoPE, gate
   and softcap are template parameters. Every build passes L1 and L2.
3. **Weights come from the training process.** `dump_golden.py` exports the codes and fp32 group scales held in
   `b._ccat` (s1_patches.py:242-258), not the bf16 `_wq` it writes today. That sidesteps whether the rotated weights
   were computed under autocast.
4. **Same compiler and flags** as `build.sh` (TheRock amdclang++, -O3, code object v5). A toolchain bump re-runs L1 and L2.

---

## 3. Architecture

### 3.1 Shape
```
host:  ModelDesc --planner--> Tape (segments + arg arena) + residency map + cost-model prediction
                    | hipGraph: 1 launch per token / verify block (2.7 us); optional n-step device loop (M5)
device k3<Variant>: 192 WG x 256 thr, persistent
  prologue: read control block (counter bases, escape flag, pending-replay count) -> pick tape (normal | escape | flush)
  loop: i = claim()  -> seg = find(i) -> switch(seg.op): body<V>(task)
        body: prefetch weights (VGPRs) -> wait(seg.deps) -> entry (row recompute) -> compute -> epilogue -> signal
  exit when i >= ntask; status word (err, timeouts, token) written by the last finisher of the head
```

### 3.2 Grid, occupancy, registers, LDS
- **Grid:** 192 WGs x 256 threads (8 waves) on the XTX, 4 per WGP. The XT runs 168. The grid size is a knob.
  Occupancy comes from `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`, which is untested on module kernels in
  this repo (M1). OVERSUB mode launches 3-5x more WGs than occupancy allows.
- **WGP mode is the default**, with per-wave fences (C6). M5 A/B-tests CU mode (`-mcumode`), whose single L0 per WG
  could allow cheaper fences.
- **VGPRs:** at most 120 (plan rule). Prefetch budget per WG: about 28 KB of weights, i.e. 7 b128 loads per lane = 28
  VGPRs (budget mapping). Verify variants with T >= 8 get their own kernel instantiations, so their register pressure
  does not lower the T = 1 occupancy.
- **LDS:** at most 32 KB per WG, so 4 WGs fit per WGP (128 KB). At T = 1 a WG uses under 8 KB: activation codes as int8
  (at most 3,712 B), x-sums per group, int32 partials (8 rows x 29 groups x 4 B), reduction scratch. At T = 16, codes
  stay packed int4 in LDS (29.7 KB) and are unpacked in registers.
- gfx1100 has no direct-to-LDS global loads, so weights prefetch into VGPRs.

### 3.3 Synchronization

**Tape claim.**
- One 64-bit claim counter per launch. A WG claims the tape index `i` and maps it to a segment by binary search over
  the segment table, which is cached in LDS (about 150 segments for S1).
- Segments are laid out in a **topological claim order**.
- Each claim can take 1-4 tasks, a knob against atomic contention.
- No claim count per launch is assumed to be fixed. Chunked claims can straddle segment ends, and tape variants
  (normal, escape, per-T) have different task counts.

**Done counters and bases (no reset, no fixed epoch arithmetic).**
- One counter per segment or per sub-segment (per GDN head, per kv group, per row class), each on its own 128 B line.
- All counters, including the claim counter, are monotonic 64-bit.
- At launch start, each WG reads a **control block** holding a base value for every counter. A claim index is
  `atomicAdd(claim) - base_claim`. A wait targets `base_c + count_c(tape)`.
- Exit: each WG adds 1 to a monotonic exit counter. The last arriver (old value = `base_exit + grid - 1`) reads the
  final value of every counter and writes them into the control block as the next launch's bases. This is one flat
  pass over about 150 counters by one WG, not ignite's per-stage walk by every block. The kernel boundary publishes
  it.
- The design therefore needs no assumption that a launch advances a counter by a fixed amount. The escape tape's TAIL
  counters, the T variants and chunked claims all stay consistent.
- Fallback if the exit pass shows up in the profile: single-task claims plus a separate counter block per tape
  variant, with each variant's epoch in the control word.

**Wait.**
1. Thread 0 polls with relaxed agent-scope loads and `s_sleep(1)` until the target is reached, up to a bounded spin
   limit.
2. `__syncthreads`.
3. Every wave issues an agent-scope acquire fence.

On timeout the wait sets the sticky `err` bit and logs (block, segment, value seen, wall_clock64) to the diag ring. The
**status word is written into the token output buffer on every step**, so the host sees errors even inside graphs.
This fixes ignite's CHECK-only readback (megakernel.cu:787-798).

**Signal.**
1. Every wave issues an agent-scope release fence.
2. `__syncthreads`.
3. Thread 0 runs `__hip_atomic_fetch_add(done, n, RELEASE, AGENT)`.

This is ignite's mk_done (megakernel.cu:162-168).

**Last finisher (no spin).**
- `mk_last_of` (megakernel.cu:852-868): release fence, then an acq_rel add. The n-th arriver runs the continuation.
- The release also orders earlier *loads*, which is what makes write-after-read safe.
- Uses: the attention split-KV merge per kv group, the head's argmax and escape decision, the per-head GDN epilogue
  (variant A1), and exact cross-WG split-K owners.

**Progress invariant.** A WG waits only on segments that come earlier in claim order. Claims are monotonic, so every
earlier task is either done or owned by a running WG. The lowest unfinished task therefore has all of its dependencies
done, and it makes progress. One resident WG suffices, so the engine is correct under any partial residency: the
compositor on the XTX, the display on the XT, or other kernels. Within a task, a WG claims, prefetches, waits, computes
and signals. It never holds an unstarted claim on an earlier index while it waits.

**Why counters and not barriers** (C1-C4):
- Barrier atomics queue behind the weight stream: Reeder measured 0.5 us idle and 3-5 us under load.
- A barrier needs co-residency, which cannot be guaranteed here.
- Counters let independent branches overlap: the MLP gate/up rows run while the GDN recurrence runs.
- No source has a clean barrier-vs-counter ablation, so M1 measures the hop cost under load directly.

**Open (M1).**
- Polling through the scalar path (`s_load ... glc`, lgkmcnt) so a poll does not wait behind in-order vector weight
  loads.
- Whether naturally aligned 64-bit stores are single-copy atomic on gfx11. Reeder-style epoch-tagged data words depend
  on it, and they are the fallback if counter hops are slow.

### 3.4 Model description, planner and tape

**ModelDesc** is a C++ builder (`runtime/k3/model.h`) that reads the exported config and tensors. A text form is
used for review:
```
model s1  d=896 vocab=248320 pos=rope(base=10000)
embed     fp32 class=cold
block[i<12] parallel entry=rms(eps=1e-6,order=block_entry) actq=dynamic_row(fwht128,qmax=7)
  mixer = (i+1)%4==0 ? attn(q=14,kv=2,hd=64,rope,window=2048,scale=0.125,kv=bf16)
                     : gdn(heads=7,dk=128,dv=128,conv=4,l2eps=1e-6,gate=swishnorm(1e-5),ab=bf16)
  mlp   = gated(silu, 2048)
  slice = host_routed(ptr_table, widths=..., null=+0.0)
  resid = ((x + mixer) + mlp) + slice
head      adaptive(hot=24576+esc int4_row, tail=223744x112 bf16 cold, p=112 iu4g)
# future lines the same planner accepts:
#  mlp   = chunk_moe(shared=512, experts=32x256, k=4, chunk=64, lag=2, pool=cold)
#  inject= ple(rows=24576, dim=256, int8, class=cold) | engram(ngram=2..3, heads=8, dim=64)
#  mixer = attn(q=32,kv=2,hd=64,nope,window=none,gate=sigmoid,softcap=30,qscale=logpos)
#  mixer = conv(width=3, gated)
```

**Planner** (host, once per model and mode) emits:
- **Segments** of 64 B each: `op, variant, ntask, tape_base, deps[<=3] (counter id, per-epoch count), signal counter,
  args offset, residency class`.
- An **arg arena** holding weight pointers, layouts, scale pointers, state and ring pointers, and constants.
- A **residency map** of every tensor's class (Sec. 3.11).
- **Fusion choices**: a fused row group merges several matvecs that share output rows, plus their epilogue (Sec. 3.5).
- The **cost-model prediction** for the tape (Sec. 3.12).

**Dispatch.**
- A uniform scalar `switch(op)` at each task boundary.
- Op bodies are templates over compile-time parameters: groups held in registers, rows per wave-pass, T, and flags.
  The planner writes `build/k3_variants.h`, the instantiation list for the model, and `build.sh` compiles `k3.hip`
  against it.
- A new model rebuilds `k3.hsaco` in seconds. Hand-written device code is needed only for a new op.
- This is the prior investigation's "data-driven loop over layer descriptors", widened to a fixed op table (C16).

**S1 tape for one GDN block** (claim order; 64-row tasks = 8 waves x 8 rows):

| Seg | Op | Tasks | Waits | Signals |
|---|---|---|---|---|
| P.h0..h6 | `MV_ROWS` q_h\|k_h\|v_h\|g_h (rows repacked head-major) + epilogue conv+silu (ring) | 7 x 8 | x[l] ready | hrow[l,h] |
| R | `GDN_STEP` (7 heads x 8 tasks x 16 v-rows) | 56 | hrow[l,h] for its own head | rdone[l] |
| P.m | `MV_ROWS` mlp gate\|up interleaved + epilogue silu*mul (bf16 act) | 64 | x[l] | prow[l] |
| P.s | `MV_ROWS` slice gu + epilogue silu*mul | 24 | x[l] | prow[l] |
| S2 | `MV_ROWGROUP` {o_proj K=896 (entry: gated norm + actq), down K=2048 (entry: actq), slice down K=768 (entry: actq)} + epilogue residual | 112 (8 rows each) | rdone[l], prow[l] | x[l+1] ready |

- R is claimed right after the head rows. Its 56 WGs prefetch their 8 KB state slices and wait on fine-grained head
  counters, while the remaining WGs stream the MLP and slice rows. This is variant A2.
- Variant A1 has the last finisher of each head's rows run the whole head step: 64 KiB of state read and written by
  one WG, est 1-1.5 us, with no spin.
- Attention blocks replace P.h and R:
  - `MV_ROWS` q\|kv (18 tasks) with an epilogue that applies RoPE and appends K/V to the ring.
  - `ATTN_DECODE` (32 tasks = 2 kv groups x 16 chunks of 128 keys) with a last-finisher merge per group.
- The head tape: `MV_ROWS` hot rows (24,577 / 192 = 128 rows per task) with entry = final norm and an epilogue
  argmax, then a last finisher that writes the token and the status word.

### 3.5 The int4 g128 matvec engine
- **Layout** (offline repack from the exported codes):
  - `[row_tile][g][row_in_tile 8][chunk 4][16 B]`. One wave-instruction loads 512 contiguous bytes, which is 8 rows of
    one 128-group, b128 per lane, with all 32 lanes busy for any K that is a multiple of 128. The SoA tile is the M1
    candidate; others are tested as K0b variants.
  - Nibbles are permuted so that `w & 0x0F0F0F0F` gives k 0..3 and `(w>>4) & 0x0F0F0F0F` gives k 4..7 of a dword.
    Codes are stored offset-binary (XOR 0x88).
  - Scales: fp32 `[row_tile][g][8]`.
- **Inner loop (T = 1):**
  - Per 16 B of weights: 12 unpack ops plus 8 `v_dot4_i32_iu8` with sign-extended A4 activation codes from LDS. That is
    0.625 VALU per weight, with a ceiling of about 24.6 T weights/s, far above the cache bandwidth.
  - The zero point is corrected exactly: `P_g = sum(w_u*x) - 8*xsum_g`, where `xsum_g` is computed once per row by the
    entry.
  - A quad (4 lanes) reduction over DPP gives exact int32 `P_g` per row. The quad leader then runs
    `acc = fmaf((float)P_g, gs, acc)` with g in order, then `*s_x`, then bf16.
- **Splits:**
  - Within a group, splits are free (int32).
  - Across waves of a WG: int32 partials go to LDS and one owner lane runs the chain (S2 default).
  - Across WGs: int32 `[row][g]` partials go to scratch, and a last finisher owns the chain. Opt-in (C15).
- **Fused row groups:** one task computes rows n..n+7 of several matvecs, each with its own activation codes and K. A
  shared epilogue (residual, RoPE, conv, silu*mul) then consumes them in the reference's add order. This is how S2
  computes o_proj + down + slice-down + residual with no extra hop.
- **Loads:**
  - Weights: b128 `global_load_b128` with the residency class's cache policy (Sec. 3.11).
  - Activations: rows of at most 8 KB read from L2 by the entry.
  - Prefetch-before-wait (megakernel.cu:238-262, 320-324, 421-425): claim, issue weight loads into VGPRs, wait, then
    read activations.
  - Vector loads return in order, so activation loads issued after the prefetch wait for it. That is acceptable: the
    prefetch takes the stage's bandwidth time anyway. The scalar-path poll (Sec. 3.3) would decouple the two.
- **Head variant:** W4A16. Codes go to bf16 (exact), then `v_dot2_f32_bf16` with a fixed order, then `*rs_row`, then
  bf16. At T = 1 this is bandwidth-bound; K0's f16 path ran 2,359 G weights/s vs dot4's 2,635. Verify T >= 8 may move
  to int8 WMMA (K6-class).

### 3.6 Row ops: redundant recompute instead of syncs
A whole-row dependency (RMSNorm, the row amax of per-row A4) normally costs a hop. Here every consumer task recomputes
it from the L2-resident row instead (megakernel.cu:264-318, 905-914, 1228-1256). The row is 896 fp32 values, or at most
3,712 bf16 values for S2.

| Entry | Work per task | Est |
|---|---|---|
| P entry | read x (3.5 KB); RMSNorm (block_entry order); bf16; FWHT-128 x 7; amax; int4 codes + xsum | 0.30 us |
| R entry | rstd + a/b dots for its head (block_entry order, the same 256-thread layout); conv outputs from scratch; L2 norm (bf16 q-hat/k-hat); beta and g | inside the R chain |
| S2 entry | gated norm per head (7 x 128) of o + g_proj, then actq over 896; actq over act 2048; actq over slice act 768 | 0.45 us |
| Head entry | final norm (rmsnorm_f32 order), then bf16(h) | 0.30 us |

**Model-change lever** (hardware-first rule): static A4 scales (frontier L6) turn each entry into a
multiply-round-pack (est 0.45 -> 0.15 us). Per-128-group activation scales would let o_proj split-K by GDN head
exactly, which removes the R->O dependency. The cost model prices both before any tuning (Sec. 5.3).

### 3.7 Op vocabulary
| Op | Parameters | Tasks | S1 use |
|---|---|---|---|
| `EMBED` | table, dtype, class (cold NT), T | 1 | fp32 row |
| `ENTRY` (inlined into consumers) | norm type, eps, order, actq mode (dynamic_row / static), pre-mix hook | - | P, S2, head |
| `MV_ROWS` | K, G, rows per task, act source, epilogue id, out dtype, class | rows/64 | stage 1, head |
| `MV_ROWGROUP` | list of (K, act source, entry), epilogue (residual order) | rows/8 | S2 |
| `GDN_STEP` | heads, dk, dv, v-rows per task, gate kind, l2 eps, q/k rounding, T, replay | heads x 8 | 9 layers |
| `ATTN_DECODE` | q heads, kv heads, hd, pos (rope base / NoPE / logpos qscale), window or none, chunk keys, softcap, out gate, KV dtype, T | kv x chunks | 3 layers |
| `SHORT_CONV` | channels, width, activation, ring size | epilogue or tasks | GDN conv (as an epilogue) |
| `TABLE_LOOKUP` | table, ids source (token / n-gram hash of the device token ring), rows, dim, class | 1-8 | future |
| `EXPERT_MV` | expert-set descriptor, K, G, per token or per chunk | rows/64 | future (slices use the descriptor today) |
| `ROUTE_CHUNK` / `ROUTE_TOKEN` | router weights, k, lag, summary (last token / mean) | small + last finisher | future |
| `STAGE` | byte range to touch with cache-allocating loads | spread over a chunk | future |
| `ARGMAX` (epilogue + last finisher) | rows, key, ties | in head tasks | head |
| `TAIL` (escape tape only) | p GEMV (112 x 896), tail scan 223,744 x 112 bf16 (NT), LSE merge in order | 1 + 192 | head |
| `SLICE` | descriptor of the active section (pointer table), null = +0.0 | - | per turn |

### 3.8 State, KV and scratch layouts
| Buffer | Layout | Size (S1) | Hazard rule |
|---|---|---|---|
| GDN state | `[layer][head][v 128][k 128]` fp32 (fla state_v_first; also FN-R1's layout) | 448 KiB/layer, 3.94 MiB | single owner per 16 v-rows, updated in place |
| Conv ring | `[layer][slot R][2688]` bf16, slot = pos mod R; R = 4 (decode), >= T+3 (verify, 32) | 21 KB/layer (R = 4) | the row owner writes slot pos and reads 3 earlier slots, so the slots are disjoint and there is no write-after-read shift (FN-R1 needed last-of-readers counters to shift in place, :981-993) |
| KV ring | `[layer][K\|V][kvh 2][slot 2048+16][64]` bf16 | 1 MiB/layer | the slot written for position p+T-1 must not be inside any in-block query's window, hence 2048+T_max slots |
| Replay buffer (verify) | `[layer][T][head]{q-hat, k-hat, v, g, beta}` | ~1.5 MB at T = 16 | double-buffered across launches |
| Token ring | last 8 token ids + position | tiny | for n-gram hashing on the device |
| Scratch | per layer: x fp32 (T x 896), stage-1 out bf16 (T x 7680), act, o, attention partials, int32 partials; counters 128 B apart; status; prof; diag | ~1.4 MiB | no aliasing: each layer owns its buffers (FN-R1's gallocr race lesson) |

**Request start (serving convention, D-277, 2026-10-31).** Every request starts from a clean slate:
- GDN state is zeroed for every layer and head.
- The conv ring and the KV ring are emptied.
- The position counter is set to 0.
- The first input token is the end-of-text token (248044), not the first prompt token.

Training saw every document start right after an EOT, so this matches that start. The packed-document leak test
(research/doc-leak-2026-10-31.md §5.3) scored it best against the actual next token: domain hard CE 2.322 with EOT
first, against 2.436 alone without EOT and 2.428 packed. Nothing carries over between requests. A multi-turn
conversation is one request whose state persists across its turns; a "request" boundary and a "turn" boundary are
different things.

### 3.9 Head and next-token feedback
1. The hot-head tasks compute bf16 logits for their 128 rows. They reduce a local argmax over a DPP/permlanex16 64-bit
   key (topk-moe.cu:131-157), then post to a packed `atomicMax` (64-bit) and a last-finisher counter. The escape row's
   logit is stored apart.
2. The last finisher:
   - If `L_esc < hv`: token = `hot_ids[hi]`. This is exact (Sec. 2).
   - Otherwise: set the escape flag and save h.
   - Write the token, position, status and `err` into the output ring.
3. The next launch's prologue reads the control word, which the kernel boundary makes visible. With escape set, every
   WG takes the **escape tape**:
   1. `TAIL`: p(h); a scan of the 50.1 MB bf16 tail with NT loads; per-task (max, sum-exp) merged in task order by a
      last finisher.
   2. The decision, then the token.
   3. `EMBED` of the resolved token, then the normal token.
   This costs 2 hops plus 55-75 us, only on firing. The normal path pays 0 hops for it.
4. **Token feedback stays on the device.** `EMBED` reads the token written by the previous launch, so the host enqueues
   graph launches without syncing and reads tokens asynchronously.
5. **n-step device loop** (M5 option): the tape repeats with epoch+1, and EMBED waits on the head's last finisher. That
   is 1 hop instead of a 2.7 us launch. n <= 16 bounds how long the compositor waits.
6. **Future n-gram tables** then hash on the device from the token ring. The hash is integer arithmetic and must match
   the host and training hash bit for bit.

### 3.10 Verify blocks (T = 4, 8, 16)
- Each T is its own kernel instantiation (C16, rule 2).
- Weights are read once per block, and ALU work scales with T.
- The matvec is per element, so a verify block is bitwise equal to batch-1 decode at every row.
- **Attention:** per position, over the position's own window. The KV for all T positions is appended in P before the
  attention tasks wait.
- **GDN:** sequential within the block. Each R task applies T steps to its 16 v-rows held in VGPRs.
- **Rollback by deferred replay** (frontier L9; no 4 MiB per-position snapshots):
  - The R task reads S, which holds the state at the start of the previous block. It first replays the previous
    block's n accepted positions from the replay buffer, then runs this block's T positions in registers. It writes
    back only the post-replay state and logs this block's inputs.
  - State traffic per block equals one decode step. The replay is ALU only.
  - The replay applies the identical step function to identical inputs, so it is bitwise equal to sequential decode.
  - After a verify launch, the state in memory is one block stale. The control word therefore carries a
    **pending-replay count**, and **every** tape variant consumes it in its GDN tasks before stepping: T = 1 decode,
    the escape tape, and verify.
  - Anything that reads the state from outside (a turn-boundary checkpoint, a host readback, a prefill handoff) first
    runs a `FLUSH_REPLAY` tape: GDN tasks only, with no new positions.
- **Acceptance** is computed on the device by the head's last finisher. It compares the drafts with the argmax per
  position and stops at the first mismatch or the first escape-flagged position. It writes n for the next launch. A
  flagged position is resolved by the next launch's tail pass.
- **Rings:** slots are indexed by absolute position, so rejected positions are simply overwritten.
- **Gate (M6):**
  - With all drafts accepted, a block equals T sequential steps bitwise.
  - Random acceptance patterns followed by continuation equal the sequential path bitwise.

### 3.11 Residency and staging (96 MB Infinity Cache)
| Class | Load policy | Holds |
|---|---|---|
| `hot` | default (MALL allocate) | core weights, the active slice, the hot head, GDN state, rings, KV, scratch |
| `cold` | nontemporal (`slc dlc`): K0 measured 23.6 us vs 53.0 us after a normal 512 MB stream, i.e. no eviction | embedding table (890 MB), tail table, inactive slices, a cold expert pool, memory tables, the N track's weights |
| `staged` | cold in VRAM; touched ahead of use by `STAGE` tasks with normal loads | the next chunk's expert set (chunk-sticky MoE) |

- **Expert and slice selection** goes through a **descriptor table** in device memory. A host slice switch is a pointer
  write between launches. A chunk router's last finisher writes the descriptor for chunk c+1 (lag-1) or c+2 (lag-2).
- **Staging:** under lag-2 it is spread over the 64 tokens of the previous chunk, about 0.26 MB per token of `STAGE`
  tasks with no dependencies. Under lag-1 it stalls the boundary token.
- **Open:** the MALL replacement policy is not LRU (gradual falloff over 96-160 MB). Whether staging next to NT streams
  keeps the trunk resident is unmeasured (sparse Sec. 4 Q4), and M7 measures it.
- **Memory tables:** with ids known at launch start (token id; n-gram hash from the token ring), `TABLE_LOOKUP` has no
  dependencies and is claimed first. Its 1-6 KB cold gather overlaps blocks 0-1. The projection and gate join a later
  stage's fused row group as an extra (K, act) pair, which adds 0 hops. A whole-row context gate (Engram) adds 1 hop
  per injection layer.

### 3.12 Instrumentation and the cost model
**Profile** (ignite pattern, megakernel.cu:62-64, 222-224, 747-775):
- wall_clock64 (100 MHz) per task: claim, wait end, compute end, signal.
- Per segment: min start / max end, the wait histogram, and idle-WG counts.
- Read back outside graph capture.

**Diag:** timeouts, err, idle claims and epochs (megakernel.cu:66-82, 553-579).

**Cost model** (`tools/k3_price.py` + a `build/k3_bench` microbenchmark that writes `runs/k3-costmodel-<date>.json`).

| Constant | Measured by | Prior value |
|---|---|---|
| `t_launch` | graph launch of the persistent module kernel | 2.7 us XTX / 3.0 XT |
| `t_hop(grid, load)` | last signal -> waiter release; idle and under a b128 stream from the MALL; 1/2/4/8 WGs per WGP; poll variants | 0.36-0.94 idle; under load unknown (Reeder 1-5) |
| `t_claim`, claim throughput | many WGs on one counter | unknown |
| `t_lastof` | the continuation's atomic and fences | unknown |
| `BW_cache(ws)` | b128 matvec at a 32/64/80/88/92 MiB working set | 1.1-1.8 TB/s (narrow K0); 3.4 stream |
| `BW_vram_nt`, `BW_stage` | NT stream; cache-allocating staging beside an NT stream | 0.67-0.91 TB/s |
| `t_entry(K)` | redundant row recompute per task | 0.30 / 0.45 us (est) |
| `t_gdn(heads, tasks, T)` | GDN_STEP chain | 0.1-0.15 us per head-step (est) |
| `t_attn(ctx, kvh, q/kv, T)` | ATTN_DECODE vs context | - |
| `t_lookup(bytes)`, `t_switch(bytes, lag)`, `t_argmax(rows)` | microbenchmarks | - |

- **Price:** `t_launch + sum over stages [max(bytes/BW_class, t_hop + t_entry) + t_epi] + unhidden chains`. A
  list-scheduling simulation of the tape over the grid refines it.
- **Validation gate:** S1 predicted vs measured within +-15%.
- **Output:** a per-op table (us per barrier, per MB of matvec bytes, per expert switch, per lookup, per GDN or
  attention step vs context), so model candidates can be priced before training (Sec. 5.3).

---

## 4. Reuse

### From ignite-llama (github.com/jdominick05/ignite-llama at e4e3f8eda, `ggml/src/ggml-cuda/megakernel.cu` unless noted)
| What | Where | How in K3 |
|---|---|---|
| claim / done / wait / tasks | :150-158, :162-168, :171-189, :193-211 | port to HIP almost as is. Replace stage arrays with the segment table, the exit reset (:529-537) with epochs, and acquire-per-poll with relaxed polls plus a fence |
| Per-wave release / acquire fences (WGP mode) | :160-168, :186-188 | keep exactly |
| mk_last_of continuation (also orders loads, for write-after-read) | :852-868 (used at :982, :1051, :1091) | merges, argmax, A1 heads, split-K owners |
| Bounded spin, err, diag, OVERSUB | :99-100, :170-189, :603-635, :66-82, :553-579 | keep; err goes into the status word on every step |
| Prefetch-before-wait | :238-262, :320-324, :421-425 | the core overlap mechanism |
| Recompute row reductions | :264-318, :905-914, :1228-1256 | ENTRY |
| GDN task structure, single-owner in-place state, [v][k] layout | :871-1100, :995-1047 | the structure only; the math is new (C9); 56 tasks, not 28 |
| Profiler | :62-64, :222-224, :747-775 | stage ids from the segment table |
| Last-block selection; DPP argmax with 64-bit keys | `topk-moe.cu`:405-467, :131-157 | head argmax, ROUTE_* top-k |
| Barrier and launch microbenchmark | `ignite/perf/mk-barrier.cpp`:17-45, 47-163 | extend into `k3_bench`: claim/done hops under load |
| Bitwise shadow compare | the FN-R1 megakernel's shadow-compare script (it caught a race there) | the L2 harness, ported to C++ |
| Numerics rules | `SCOREBOARD.md`:183-190, 217-222 | contract(off) + explicit fmaf; compile-time variants |

Not reused: `dim3(32,8)` MMVQ shapes, q8_1 activations, the ggml graph matchers, FN-R1's GDN arithmetic, arrival
barriers (`mk-barrier.cpp` barrier mode), and the mk_seq single-block executor.

### From this repository's training stack and a private lab repository
| What | Where | How |
|---|---|---|
| `fwht128`, `actq_fwd_src`, `row_rstd`, `block_entry_fwd`, `silu_bf`, `silu_mul_actq_fwd` | `kernels/actq.hip`:40-53, 97-132, 239-273, 88, 307 | move into `kernels/common/actq.h`; training kernels and K3 both include it (rule 1) |
| iu4g epilogue order (reference) | `kernels/gemm.hip`:244-350 (chain at 294-336) | the matvec's chain; the L1 reference by M = 64 padding |
| `rope_pair`, attention math | `kernels/attn.hip`:99-122, 124-156 | `common/attn.h`; `attn_causal_bf16` is the tolerance reference |
| embedding, `rmsnorm_f32`, `silu_mul_bf16`, `residual3` | `kernels/core.hip`:23-61 | `common/core.h` |
| `residual_slices_fwd` add order | `kernels/fuse.hip`:19-32 | the S2 epilogue |
| `gdn_fwd_h` with h0 / ht | `kernels/gdn.hip`:1-20, 51 | the prefill-state handoff for teacher-forced L3 tests |
| Module / Args / launch; TritonManifest | `runtime/rt.h`:55-70; `runtime/rt.cpp`:80-124 | K3 is an `extern "C"` code-object kernel; captured Triton kernels are the L1 oracles |
| Golden and scoring tools | `tools/dump_golden.py`:88-96, `tools/score_hidden.py`, `tools/capture_triton.py`, `runtime/test_gdn.cpp` | export `b._ccat` codes and fp32 scales plus head buffers; L3 scoring; new captures (conv, l2norm, gate, gated norm, fwd_h with h0/ht) |
| dot4 unpack, x-sum zero point | the K0 matvec prototype (a private lab repository), lines 29-79 | grows into K0b (b128 SoA, S1 tiles, 80-92 MiB) |
| Reeder megakernels (MIT) | github.com/SamuelReeder/megakernels | ideas only: epoch-tagged words, the conv ring, a multi-step loop |

### New
`runtime/k3/` (ModelDesc, planner, tape, residency map, engine launcher, status reader), `kernels/k3.hip` (engine
loop plus op bodies), `kernels/common/*.h` (the shared numerics), `runtime/s1_decode.cpp` (the M0 per-node
reference), `runtime/test_k3.cpp` (L1/L2 harness with OVERSUB and multi-thousand-step shadow compare),
`tools/k3_price.py` (cost model), deferred-replay verify, the escape tape, and the b128 SoA weight repacker.

---

## 5. Budget

### 5.1 Bytes (S1-today, int4 g128 + fp32 group scales, int4 hot head, bf16 KV)
| Item | Hot set (MiB) | Traffic per token (MB) |
|---|---|---|
| Core weights (9 GDN + 3 attention blocks, MLPs, norms, a/b, conv) | 55.19 | 57.88 |
| Hot head 24,577 x 896 int4 + fp32 row scales | 10.59 | 11.11 |
| Slice 768 (12 x (gu 1536x896 + down 896x768)) | 12.55 | 13.16 (0 on the null route) |
| GDN state (read + write) | 3.94 | 8.26 |
| Conv ring (R = 4) | 0.18 | 0.19 |
| KV window 2048 (bf16) | 3.00 | 3.15 |
| Scratch, counters | ~1.4 (est) | small |
| **Total, slice 768 / null** | **86.9 / 74.3** (budget 88; physical 91.55) | **93.7 / 80.6** |

Bandwidth term for 93.7 MB:

| Bandwidth | 3.4 TB/s | 3.0 TB/s | 2.5 TB/s | 1.8 TB/s | 1.3 TB/s (K0 narrow) |
|---|---|---|---|---|---|
| Time | 27.6 us | 31.2 us | 37.5 us | 52.1 us | 72.1 us |

### 5.2 Per-stage critical path, batch 1, slice 768, schedule A
Assumptions (central): 3.0 TB/s effective; hop 0.40 us; entry 0.30 us (P, head) and 0.45 us (S2); epilogue 0.20 us;
launch + embedding 3.0 us. "Overlap" = prefetch-before-wait, `max(fetch, hop+entry) + epi`; "serial" = the sum.

| Stage (count) | Bytes | Fetch @3.0 | Overlap | Serial | Bound by |
|---|---|---|---|---|---|
| Launch + EMBED (1) | 3.5 KB cold | - | 3.0 | 3.0 | launch |
| GDN stage 1: P rows + R (9) | 5.40 MB (incl. state r+w) | 1.80 | **2.00** | 2.70 | bandwidth; R chain must hide (A vs B) |
| Attention stage 1: P rows + ATTN (3) | 4.28 MB (incl. 1 MiB KV) | 1.43 | **1.63** | 2.33 | bandwidth |
| S2: o\|down\|slice-down + residual (12) | 1.77 MB | 0.59 | **1.05** | 1.64 | latency (hop + entry) |
| Head + argmax (1) | 11.11 MB | 3.70 | **4.00** | 4.70 | bandwidth |
| Unhidden intra-stage chains | - | - | 2.8 (est) | 2.8 | R / ATTN chain tails |
| **Total** | 93.7 MB | | **45.3 us = 22.1k tok/s** | **61.4 us = 16.3k** | |

**Critical path:** launch -> EMBED -> for each block: [hop -> entry -> head-row P tasks -> R chain (or ATTN chain) ||
MLP and slice rows] -> hop -> S2 entry (gated norm and actq) -> S2 dot -> residual -> ... -> final norm -> hot head ->
last finisher -> token.

**Per op class at the central point (overlap):**
- matvec bytes: 82.1 MB = 27.4 us
- GDN state: 8.3 MB = 2.8 us
- KV: 1.05 us
- grid hops: 24 x 0.40 = 9.6 us, of which about 4.8 is exposed in S2
- entries: about 5.4 us exposed (S2)
- launch: 2.7 us
- argmax / last finisher: 0.3 us

**Scenarios:**

| Scenario | Bandwidth | Hop | Overlap | Serial |
|---|---|---|---|---|
| Optimistic | 3.4 TB/s | 0.36 us | 41.2 us (24.3k) | 56.0 us (17.9k) |
| Central, schedule A | 3.0 TB/s | 0.40 us | 45.3 us (22.1k) | 61.4 us (16.3k) |
| Central, schedule B (budget model) | 3.0 TB/s | 0.40 us | 52.5 us (19.0k) | 72.0 us (13.9k, miss) |
| Pessimistic, idle hops | 1.8 TB/s | 0.51 us | 63.0 us (15.9k) | 85.0 us (11.8k) |
| Pessimistic, hops under load | 1.8 TB/s | 0.9 us | 68.6 us (14.6k, miss) | 96.0 us (10.4k) |
| Reeder-like hops under load | 2.5 TB/s | 3 us | 99.5 us (10.0k) | 136 us (7.4k, at the gate) |
| K0 narrow loads | 1.3 TB/s | 0.40 us | 83.0 us (12.0k) | 102 us (9.8k) |

What 15k needs (budget bisection):
- At least 1.66 TB/s effective with overlap, or 2.55 TB/s serial (schedule A, central hop).
- Schedule B: 1.93 / 3.71 TB/s.

**Verify (A, central, dot4):**

| T | Time per block | Positions/s | Bound |
|---|---|---|---|
| 4 | 51 us | 78k | - |
| 8 | 68 us | 117k | ALU from about T = 8 |
| 16 | 117 us | 136k | ALU (iu4 WMMA: about 84 us, 191k) |

The GDN chain grows by T x 0.1-0.15 us per layer.

**Escape tail** (bf16 50.1 MB from VRAM): 55-75 us per firing.

| Escape rate | 3% | 10% | 100% (hash / base64 spans) |
|---|---|---|---|
| tok/s | 21.3k | 19.4k | 9.2k |

### 5.3 Cost-model entries for candidate features (est; M1 and M7 replace them with measured constants)
| Feature | Engine mechanism | Added bytes per token | Added hops | Est us/token at central |
|---|---|---|---|---|
| Host slice switch (per turn) | descriptor pointer write | 13.2 MB cold once | 0 | +14.5-19.7 once, then 0 |
| **Chunk-sticky expert switch** (64-token chunk; 12 layers x 4 x 256-wide = 33M params, about 16.8 MB) | `ROUTE_CHUNK` on the boundary token + `STAGE` + `EXPERT_MV` | 16.8 MB / 64 = 0.26 MB VRAM | +1 per MoE layer on boundary tokens only | lag-2: +0.3-0.4 (hidden staging; VRAM contention) + about 0.1 router; lag-1: a 18.5-25 us stall per chunk (+0.3-0.4 amortized). Hot set: trunk + 2 sets must fit 88 MiB (sparse Sec. 3B: 47 + 2 x 16 MiB) |
| Per-token routed experts (cache-resident pool) | `ROUTE_TOKEN` + `EXPERT_MV` | active experts only | +1 per MoE layer | +0.7 per MoE layer (hop + top-k) = +8.4 for 12 layers |
| PLE (hot-vocab 24k x 256 x 12, int8) | `TABLE_LOOKUP` at launch start + inject rows fused into S2 | 3 KB cold + 1.4 MB projections (hot) | 0 | +0.5 |
| Engram-lite (2-3-gram, 2 layers) | device hash from the token ring + lookup + context gate | about 2 KB cold + gate and projections | +1 per injection layer | +1.4 |
| Extra attention layer, GQA 16:1 (d 896, 14 q / 1 kv, window 2048) | `ATTN_DECODE` | q, kv and o rows + 0.5 MiB KV | 0 (fits the P/S2 stages) | +1.5-2 (est) |
| NoPE global layer, no window, 2 kv heads bf16 | `ATTN_DECODE(window=none)` | ctx x 512 B: 1 MB @2k, 16.8 MB @32k | 0 | +0.4 @2k; +5.6 @32k if cache-resident, +18-25 @32k from VRAM (q8_0: half) |
| Sigmoid output gate (rows concatenated onto q) | epilogue | +0.43 MB per layer | 0 | +0.15 per layer |
| Softcap, log-position q scale | epilogue | 0 | 0 | about 0 |
| Short-conv mixer (C channels, W 3-4) | `SHORT_CONV` in the P epilogue | projections + C x W x 2 B ring | 0 | bytes/BW + about 0.1 |
| **Static A4 scales** (model change) | `ENTRY` static mode | 0 | 0 | -3.6 (S2 entries); enables o_proj split per head |
| **Per-128-group activation scales** (model change) | exact split-K per head into o_proj | 0 | -1 per GDN block (R->O) | -4 to -6 if schedule B is what M2 measures |
| Looped core, R = 2 | tape repeats | x2 block bytes | x2 | +38 |
| q8_0 KV | `ATTN_DECODE(kv=q8_0)` | -1.5 MB | 0 | -0.5 (numerics change, gated) |

---

## 6. Milestones (engine first; each is 1-2 days; GPU work only when the cards are free, with every XTX test capped at 0.85 of VRAM because of the RDP compositor)

| M | Deliverable | Gate (bitwise or tolerance) | Measurement |
|---|---|---|---|
| **M0** decode reference (2 d) | (1) Shared numerics headers `kernels/common/*.h`; the training kernels rebuilt from them. (2) `dump_golden.py` exports `b._ccat` codes + fp32 g128 scales, the int4 head (+bf16 head), tail, conv, a/b, A_log, dt_bias. (3) New Triton captures: conv, l2norm, gate, gated norm, `fwd_h` with h0/ht. (4) `runtime/s1_decode.cpp`: per-node, stateful (GDN, conv ring, KV ring, positions, slices, adaptive head + escape), on the shared op bodies | (a) Rebuilt training kernels are **bitwise** equal to today's code objects. (b) L1 bitwise for embedding, entry, actq, iu4g (M = 64 pad), silu*mul, residual, RoPE, conv / l2 / gate / gated norm (or a documented <= 1 ulp). (c) GDN step within 4 fp32 ulp of fused_recurrent after 1 step; drift reported over 2,048 steps. (d) L3: with the bf16 head, the decode golden-window soft CE and top-1 lie inside the measured band, and teacher-forced decode agrees with chunked prefill statistically. (e) The int4 head is within +0.02 nats; copy gate >= 99% | reference us/token (per node, about 1 ms est); escape fire rate on the golden window and on hash/base64 spans |
| **M1** engine skeleton + sync + matvec + cost-model microbenchmark (2 d) | `kernels/k3.hip` loop, segment table, claim, epochs, wait/signal/last-of, status word, OVERSUB, SPLIT mode; the b128 SoA repacker; `MV_ROWS` / `MV_ROWGROUP`; `k3_bench`; K0 correctness rerun | **Bitwise** vs `gemm_tn_iu4g` for every S1 shape, the head variant vs M0. A 1,000-segment synthetic tape passes OVERSUB 3-5x on both cards. A forced timeout surfaces err through the status word inside a graph | `t_hop` idle and under load at 1/2/4/8 WGs per WGP; poll variants (acquire-per-poll, relaxed+fence, scalar); `t_claim` and throughput; `BW_cache` b128 at 32/64/80/88/92 MiB; `BW_vram_nt`; module-API occupancy; graph capture of the module launch; `v_dot2_f32_bf16` emitted for the head variant (disassembly). **Kill rule (Sec. 7)** |
| **M2** row ops + GDN block (2 d) | `ENTRY`, `GDN_STEP` (A2 and A1), the conv-ring epilogue, S2 rowgroup with gated norm + residual; one GDN block in the engine | L1 bitwise vs M0 for every op. L2: block output, state and ring **bitwise** vs M0 for 4,096 steps, normal and OVERSUB, both cards | R chain vs P fetch (**decides schedule A or B**); `t_gdn` vs tasks per head; `t_entry` |
| **M3** attention block (1-2 d) | q\|kv rows with a RoPE + KV-append epilogue; `ATTN_DECODE` with a last-finisher merge | L2 bitwise vs M0 for positions 0-4,095 (crossing the window wrap); tolerance vs `attn_causal_bf16` (<= 1.3e-3 relative) | `t_attn(ctx)` for ctx 128-2,048, plus no-window up to 32k (the NoPE price) |
| **M4** S1 end to end + head (2 d) | full tape, the head with argmax, the escape tape, device token feedback, slices via the descriptor | L2 **bitwise** vs M0 for >= 10k steps on the golden window, both cards, normal + OVERSUB, err = 0; forced-escape tests bitwise; L3 band (as in M0) | tok/s on the XTX vs the 15k target and the 6,960 gate; per-segment profile; **cost model within +-15%** of measured |
| **M5** tuning (2 d) | knobs: grid, claim order and chunking, tasks per stage (S2 granularity, cross-WG int32 split), prefetch depth, WGP vs CU mode, the n-step device loop. Ask the user before any op-level kernel optimization, and price the model levers (Sec. 5.3) first (hardware-first rule) | every change re-passes L2 bitwise (shadow compare) | tok/s, cost-model refit |
| **M6** verify blocks (2 d) | T = 4/8/16 instantiations, deferred GDN replay, verify-sized conv and KV rings, acceptance on the device | all-accepted block equals T sequential steps **bitwise**; random accept patterns plus continuation equal sequential bitwise over 10k blocks | us per block per T vs 60/80/130 us |
| **M7** first future-feature op (2 d) | priced by the cost model first, then built: `TABLE_LOOKUP` + inject (PLE) **or** `ROUTE_CHUNK` + `STAGE` + `EXPERT_MV` on a synthetic S1-shaped model with random weights | op bitwise vs a per-node reference; measured cost within +-20% of predicted. Staging test: trunk 47 MiB + 16 MiB staged + 100 MiB NT stream; trunk matvec time unchanged within 5% | `t_lookup`, `t_switch(lag)`, MALL behaviour under staging |

---

## 7. Risks and kill criteria

| # | Risk | Detect | Mitigation | Kill / re-plan criterion |
|---|---|---|---|---|
| K1 | b128 matvec bandwidth at an 80-92 MiB working set is below need (nothing has measured it; K0 narrow gave 1.1-1.8 TB/s at 32 MB) | M1 `BW_cache` | layout variants, more loads in flight, a null-route default (74 MiB) | the **single rule** below |
| K2 | Hand-off under load is slow (Reeder: 3-5 us barriers; claim/done never timed) | M1 `t_hop(load)` | relaxed polls, scalar polls, epoch-tagged data words, coarser stages, static A4 (fewer entries) | the **single rule** below |

**Single kill rule after M1** (K1 and K2 together). Plug the measured `BW_cache(88 MiB)`, `t_hop(load)`, `t_claim`
and `t_entry` into the cost model for S1, schedule B (the conservative case).
- Predicted **<= 66.7 us**: build toward 15k.
- **66.7-143.7 us**: drop the 15k target and build to the >= 6,960 tok/s gate. Price the model levers (static A4,
  per-group activation scales, a smaller hot set) before any kernel tuning.
- **> 143.7 us**: stop K3 and re-plan with model changes.

For reference, Sec. 5.2's Reeder-like row (3 us hops) still predicts 99.5 / 136 us, which passes the gate. Only both
constants going bad together kills it.
| K3 | A partial-residency deadlock or race (FN-R1: 1.4e-4 per launch, invisible to KLD) | OVERSUB + L2 shadow compare, both cards | claim-order invariant, bounded spin, status word | **any** L2 mismatch over 10k steps stops the ship |
| K4 | Bit drift from the compiler, flags or code shape | L1/L2 on every build | contract(off) + explicit fmaf; template variants; pinned toolchain | a mismatch blocks the merge |
| K5 | Decode vs training drift: GDN fp32 recurrence vs the bf16 chunk path; the int4 head | L3 band | bf16 q-hat/k-hat as in training; a bf16-head reference mode | L3 more than 0.01 nats outside the band: investigate rounding points before speed work |
| K6 | Hot set over 88 MiB (margin 1.1 MiB) | measured footprint; matvec time vs working set | null-route default; q8_0 KV (gated); slices <= 512 | measured warm matvec time more than 10% above cold-free |
| K7 | Too few tasks in narrow stages (S2: 112 tasks < 192 WGs) or claim-counter contention (about 4k claims per token) | M4 profile: idle WGs, claim latency | 4-row tasks, cross-WG int32 split, multi-task claims | - |
| K8 | VGPRs over 120 in the fused op switch | compile report | separate instantiations per T | occupancy below 4 WGs per WGP |
| K9 | Escape tail fires often on target content (55-75 us each) | M0 fire-rate measurement | int8 / int4 tail PTQ test (a model decision) | fire rate > 10% on coding spans: raise it with the user |
| K10 | Verify ALU-bound at T >= 8 | M6 | iu4 WMMA (K6) | - |
| K11 | Module-API occupancy and graph capture are untested here | M1 | hipcc fallback build for the launcher | - |
| K12 | S1's design changes mid-build | - | S1 specifics live only in ModelDesc and op parameters; the engine (M1) comes first | none: this is the point of the design |
| K13 | GPU contention: training on both cards, the RDP compositor on the XTX | - | run tests in free windows; 0.85 VRAM cap; the XT for correctness while the XTX trains (split-work rule) | - |
