# IQ quant study: codebook formats in Qwen3-30B-A3B's experts

A study in progress. This page holds its plan and its predictions, written down before the first timed run on the
RX 7900 XTX. Results will be added below the predictions and scored against them.

## The question

[Qwen3-30B-A3B on own code](qwen3-30b-a3b.md) found that decoding the quantized weights is the smallest part of a
token. That was Q4_K and Q6_K, which decode to small exact integers in a few vector instructions. The codebook
formats (IQ) look their values up in a table. Does the finding survive that, and do fewer bytes per weight buy speed
on a decode that is bound by the memory bus?

- **One thing changes: the format of the expert tensors.** The architecture (`qwen3moe`), the decode tape, the
  attention, router, norm and head tensors (the official Q4_K_M's), the card and the harness are held.
- **Why the experts:** they are 1.10 of the 1.92 GB a token reads (57 %).

## The arms

Four files that differ only in their 144 expert tensors. Arm A is the official Q4_K_M. B and C take their experts
from unsloth's imatrix quants of the same weights, D from one requantization. Everything outside the experts is the
official file's, byte for byte (`tools/eval/gguf_splice.py` writes each file and checks it against its sources).

| Arm | Experts from | Expert formats (layers of 48) | Expert bytes per token | Bytes per token | Bound at 858 GB/s |
|---|---|---|---|---|---|
| A | the official Q4_K_M | gate, up Q4_K; down Q4_K 24, Q6_K 24 | 1.097 GB | 1.920 GB | 2.24 ms, 447 tok/s |
| B | unsloth IQ4_XS | IQ4_XS everywhere | 0.963 GB | 1.785 GB | 2.08 ms, 481 tok/s |
| C | unsloth UD-IQ3_XXS | gate, up IQ3_XXS 32, IQ3_S 16; down IQ3_XXS 25, IQ4_XS 13, IQ3_S 10 | 0.743 GB | 1.565 GB | 1.82 ms, 548 tok/s |
| D | own requant | IQ2_XXS everywhere | 0.467 GB | 1.290 GB | 1.50 ms, 665 tok/s |

Bytes count 8 of 128 experts per layer at short context (`tools/eval/gguf_census.py`). In arm C the gate and the up
tensor of a layer always share a format, so the tape keeps its shape: one segment computes both.

- **Arm D:** `llama-quantize --allow-requantize` from unsloth's Q8_0 with unsloth's importance matrix, Q4_K_M as the
  base type and the three expert tensors overridden to IQ2_XXS; its experts are then spliced into the official file
  like B's and C's.
- **Why splice:** unsloth's IQ files also move the attention to other formats, which changes two things at once.

## The formats

All four are blocks of 256 weights with one scale per 32 and no offset term. What is new is the lookup.

| Format | Block | A weight is | Codebook |
|---|---|---|---|
| IQ4_XS | 136 B | d × (scale − 32) × kv[4-bit index] | 16 int8 values |
| IQ3_XXS | 98 B | d × (0.5 + s) × 0.5 × grid[index][k] × sign | 256 entries × 4 values (1 KB) |
| IQ3_S | 110 B | d × (1 + 2 s) × grid[9-bit index][k] × sign | 512 entries × 4 values (2 KB) |
| IQ2_XXS | 66 B | d × (0.5 + s) × 0.25 × grid[index][k] × sign | 256 entries × 8 values (2 KB) |

The host decoders (`caminus.own_dequant`) are written from these layouts and are bitwise gguf-py's on random blocks
and on real tensors. The codebooks are format constants that cannot be derived; they are ggml's data and live in
`caminus.iq_tables` with their notice. The 8 sign bits of a group are a 7-bit index plus its parity, so signs need no
table.

In the decode kernel a 16-value table fits in registers, and the grids go to the workgroup's local memory, staged at
the start of a launch by the arms that need them. A grid read depends on the index load, so a pass of the tile becomes
a memory round trip and then a local one: that second dependent step is what this study measures.

## Predictions

Written before any timed run of an IQ arm. The numbers of the plan's first draft, made before the decode engine took
its per-card grid, are kept beside the revised ones.

The model behind the revised numbers, from a traced token of arm A on the RX 7900 XTX: a stage costs its bytes at
858 GB/s plus a fixed 4 to 5 µs (expert gate and up: 16.5 + 4.2 = 20.7 µs per layer; expert down: 10.1 + 5.0 =
15.2 µs). The **reference line** of an arm is A's measured token (4.03 ms) with the expert stages' bytes replaced by
the arm's and nothing added for decoding: A 4.03, B 3.87, C 3.62, D 3.30 ms (248, 258, 276, 303 tok/s). The first
draft's line, from the older 4.32 ms token: 4.32, 4.16, 3.90, 3.58 ms.

| Hypothesis | Prediction | First draft | Measured by | Fails if |
|---|---|---|---|---|
| H1: decoding stays the smallest part | Decode time of a token on the XT (full build minus loads-only build): A 0.57 ms (measured earlier), B 0.55–0.75, C 0.75–1.2, D 0.7–1.1 ms | B 0.6–0.8, C 0.9–1.5, D 0.8–1.4 ms | the measurement builds, XT | an arm's decode time exceeds its own loads time |
| H2: fewer bytes buy less than their share | XTX, depth 8: B 3.90–4.00, C 3.70–3.95, D 3.40–3.70 ms. Every arm sits above its reference line by its decode cost | B 4.20–4.40, C 3.95–4.30, D 3.80–4.20 ms | `benchmarks/q3moe_depth.py`, interleaved, 5 rounds | an arm lands on or below its reference line |
| H2, per stage | XTX trace, µs per layer, gate and up / down: A 20.7 / 15.2 (measured); IQ4_XS 19.7–21.0 / 12.8–14.0; IQ3_XXS 15.4–20 / 10.7–13.5; IQ3_S 16.8–21 / 11.3–14; IQ2_XXS 11.7–17 / 8.8–12. The low end is bytes plus the fixed cost | (none) | `benchmarks/q3moe_trace.py` | a stage lands below its bytes plus A's fixed cost |
| H3: codebook decode slows the prefill | at 4,096 tokens: B within 10 % of A, C and D 10–40 % slower than A | the same | `benchmarks/q3moe_prefill.py` | C or D at or above A's prompt speed |
| H4: own math stays closer to the reference | llama.cpp's mean KLD over ours ≥ 3 for every arm, top-1 ≥ llama.cpp's, each arm against a reference computed from its own decoded weights | the same | the G-model gates at 128 tokens and at 16,384 | the ratio drops below 3 |
| H5: the splice changes nothing outside the experts | the official file spliced with itself comes out byte for byte | the same | the splice tool's own check | any tensor differs |

H5 is already in: the official file spliced with itself is the official file (579 tensors, 18.557 GB, equal sha256).

## Stop rules

The run scripts stop on any of these.

1. A host decoder is not bitwise gguf-py's dequantize on every expert tensor of its arm.
2. An own tile's mean or p99 error against fp64 is above llama.cpp's on the same rows.
3. The arms do not share one code object and one grid per card, or that code object goes above 96 VGPRs or spills a
   VGPR. (The first draft said "falls below 6 workgroups per WGP"; the engine has since taken 4 on the XTX and 5 on
   the XT for every model file, so the rule is worded by its intent: nothing but the format may differ between arms.)
4. An own standalone tile is more than 2× slower than llama.cpp's matvec on the same shape: the comparison would
   measure implementations, not formats.
5. A tensor outside the experts differs from the official file's.

Rules 2 and 4 name llama.cpp's matvec kernel, and the build of it used here has none for IQ2_XXS. For arm D they
read, written down before the IQ2_XXS tile's first run:

- **Rule 2 for IQ2_XXS:** the tile's mean and p99 error against fp64 stay within the worst of the three other own
  codebook tiles on their own matrices (mean 2.0e-4, p99 5.8e-4 of the output's rms). All four share the activation
  format and the summation, so a larger error would be a decode bug, not rounding.
- **Rule 4 for IQ2_XXS:** the tile is not slower per matvec than the own IQ3_XXS tile on the same shape (it reads
  fewer bytes and does the same arithmetic).

## How it is run

- **A study module, not the shipped one.** All four arms run one new decode module that holds every format's tile.
  The shipped `k3_q3m_lean` stays as it is; arm A on the shipped module against arm A on the study module is reported
  as what carrying the formats costs.
- **Grids are staged only by the tapes that use them,** so arm A does not pay for them, and the staging is timed on
  its own.
- **Order:** IQ4_XS first (no grid in local memory, and arm C needs it too), then IQ3_XXS, IQ3_S, IQ2_XXS; standalone
  tiles with their accuracy against fp64 before the K3 ops; the XT before the XTX; every GPU command under
  `tools/guard.sh`.
- **Baselines:** `llama-bench` on the four spliced files, and on unsloth's files as they are for the real-world
  numbers.

## Where this departs from the first draft of the plan

- Arm D uses unsloth's importance matrix (the one behind B and C), not one computed here.
- The splice writes the file itself (`tools/eval/gguf_splice.py`); the vendored gguf-py has no writer.
- All three grids need 5 KB of local memory, not 3. The module holds 3 KB: see the next section.
- The draft called the 2.29 ms a token spends outside loads and arithmetic "hand-off". A trace has since shown that
  time is inside the stages: the hand-offs are 0.09 ms of a token.
- The decode engine took a per-card grid after the draft was written; the reference line and stop rule 3 follow it.

## Fitting four formats into one module

Stop rule 3 was tested at build time, before any arm ran in the decode kernel. The first build of a module with all
four tiles had 21 spilled VGPRs. The module the arms run has none: `k3_run` at 96 VGPRs, 0 spilled, 66 spilled scalars
(the shipped module: 96, 0, 53), and the shipped module's code is unchanged, instruction for instruction. Four things
got it there, each a finding about this compiler and card.

| Step | k3_run | What it showed |
|---|---|---|
| Four tile bodies beside Q4_K and Q6_K | 21 spilled VGPRs | every body's loop state is live across the dispatch |
| One body for the three grid formats, the format a runtime value | 7 spilled | their code differs in offsets and one shift, not in shape |
| The lane number through an opaque copy inside each format's branch | 0 spilled, but 97 VGPRs | from one shared lane value the compiler hoisted every tile's addresses out of the row loop, where they stayed live across whichever tile ran |
| The module's local memory kept at the shipped size | 96 VGPRs, 1 spilled | with the grids' 5 KB the local memory crossed 16 KB, and k3_run came out at 97 VGPRs under every register cap tried; at the shipped size it is 96 |
| The grid formats read from a record layout | 96 VGPRs, 0 spilled | in the GGUF block a unit of 32 weights needs five loads at five offsets; as one record it is one 16-byte load, plus one 8-byte load of its block's header |

- **Local memory at the shipped size:** this module's router ranks 256 experts, not 1,024 (the model has 128), and the
  IQ2_XXS grid shares its words with the two IQ3 grids. A launch stages the IQ3 grids or the IQ2_XXS grid; the engine
  refuses a tape that needs both. No arm does.
- **The record layout** holds the same bytes as the GGUF blocks, rearranged once at load
  (`caminus.engine.own_layout.iq_records`, with its inverse and a test), as the engine already does for Q6_K. So the
  arms' bytes per token, and the bandwidth bound, are unchanged.
- **Both changes came after the predictions were written,** forced by stop rule 3. No number in the predictions table
  was touched.

## Results so far

### The tiles alone (RX 7900 XT)

One launch per matvec on each arm's real expert tensors, 8 of 128 experts per layer, cold between matvecs
(`benchmarks/own_mv_census.py`). A launch costs about 12 µs here, so these times say which kernel is faster on the
same matrix, not how fast the format is in a token. Error is against float64 on 256 sampled rows, as a fraction of the
output's rms: mean / 99th percentile.

| Format | Shape | Read only | llama.cpp | Own tile | Error, llama.cpp | Error, own |
|---|---|---|---|---|---|---|
| IQ4_XS | 2048 × 768 | 9.23 µs | 10.84 µs | 10.05 µs | 9.7e-3 / 2.9e-2 | 1.8e-4 / 5.1e-4 |
| IQ3_XXS | 768 × 2048 | 12.35 µs | 14.90 µs | 13.10 µs | 8.1e-3 / 2.8e-2 | 1.8e-4 / 5.8e-4 |
| IQ3_XXS | 2048 × 768 | 9.00 µs | 10.62 µs | 9.82 µs | 8.2e-3 / 2.8e-2 | 1.7e-4 / 5.7e-4 |
| IQ3_S | 768 × 2048 | 12.50 µs | 14.20 µs | 13.04 µs | 7.9e-3 / 2.5e-2 | 2.0e-4 / 5.5e-4 |
| IQ3_S | 2048 × 768 | 9.06 µs | 10.93 µs | 9.52 µs | 8.8e-3 / 2.7e-2 | 1.6e-4 / 4.6e-4 |
| IQ2_XXS | 768 × 2048 | 12.49 µs | (no kernel) | 13.26 µs | | 1.5e-4 / 4.8e-4 |
| IQ2_XXS | 2048 × 768 | 9.02 µs | (no kernel) | 9.68 µs | | 1.7e-4 / 5.2e-4 |

- **Stop rules 2 and 4 hold** for IQ4_XS, IQ3_XXS and IQ3_S: every own tile is faster than llama.cpp's kernel on the
  same matrix and 40 to 60 times closer to float64. The own tiles' error is the fp16 activations' rounding: against
  float64 on the rounded activations it is 5e-8 to 1e-7 (`tests/engine/test_own_iq_ops.py`).
- **IQ2_XXS, rule 2 as written for it:** holds (1.7e-4 / 5.2e-4 at worst against 2.0e-4 / 5.8e-4).
- **IQ2_XXS, rule 4 as written for it: not settled by this instrument.** The first two runs: 1.4 % faster than the
  IQ3_XXS tile on the 2048 × 768 shape, 1.2 % slower on 768 × 2048 (13.26 against 13.10 µs), where the two runs'
  read-only kernels differ by as much. Repeated with the two files alternated three times (7 rounds each), µs per
  matvec, own tile and (own minus read-only):

  | Shape (role) | IQ3_XXS | IQ2_XXS |
  |---|---|---|
  | 768 × 2048 (gate) | 9.54, 9.57, 9.50 (0.26, 0.46, –) | 9.62, 12.15, 9.82 (0.39, 0.33, 0.59) |
  | 768 × 2048 (up) | 13.04, 12.99, 12.98 (0.74, 0.72, 0.82) | 13.01, 13.33, 13.17 (0.69, 0.97, 0.89) |
  | 2048 × 768 (down) | 9.54, 9.50, 9.55 (0.59, 0.65, 0.67) | 9.67, 9.59, 9.82 (0.57, 0.49, 0.64) |

  By the medians of the own tile, IQ2_XXS is 1 to 3 % slower on every shape, which trips the rule by its letter. By
  own minus read-only it is faster on down, overlaps on gate and is slower on up. The same kernel on the same matrix
  moves between 9.3 and 12 µs across processes here (the gate rows; one read-only run took longer than its own
  tile, shown as –), and a launch-bound time cannot see what a format costs inside a token (below). The rule is
  left unruled and arm D's times below are reported but not scored.

### The tiles in the decode kernel (RX 7900 XT)

`tests/engine/test_own_iq_ops.py` on arms B, C and D: RESULT PASS on each.

- Every expert role and format of an arm as one tape (one launch stages the grids that arm needs together): each
  segment's rows are bitwise the standalone kernel's, per expert slot, at 3, 5, 7 and 8 workgroups per WGP.
- The fused gate and up segment gives the same bits at every grid and equals silu(gate) × up of the standalone sums
  to fp16 rounding.
- The engine refuses, before any launch: a grid format on the shipped module, a tape with grid formats that does not
  start with the embedding (its segment carries the grids to stage), and IQ2_XXS with an IQ3 format on one tape.

### The arms in the decode engine (RX 7900 XT, then RX 7900 XTX)

`tests/engine/test_own_iq_ops.py --layers all` on arms B, C and D (XT): all 144 expert tensors of each arm, 16 drawn
experts each, every row of every slot against float64 on the activations the tile reads: worst mean 1.2e-7, worst
row 2.1e-6 of the rms; bitwise the standalone kernels at every grid. RESULT PASS on each.

`tests/engine/test_q3moe_one.py` per arm against a float64 reference computed from that arm's own decoded weights
(seven positions), on both cards:

| Arm | Module | Top-1 | Mean KLD (max) | Grids 3 / 7 / 8 |
|---|---|---|---|---|
| A | shipped, and the study module | equal | 3.99e-06 (2.59e-05) | same bits; the two modules' logits differ in 0 words |
| B | study | equal | 4.45e-07 (1.69e-06) | same bits |
| C | study | equal | 3.23e-06 (1.99e-05) | same bits |
| D | study | equal | 2.65e-04 (1.83e-03) | same bits |

D's one high position is a router tie, not arithmetic: at position 4, layer 37, the reference's 8th and 9th expert
have probabilities 0.01993560 and 0.01993237, and the engine picks the other one. The float64 reference with that one
pick swapped agrees with the engine to a KLD of 1.3e-06 at that position (1.83e-03 without the swap) and 1.9e-07 at
the next (1.4e-05).

### Whole tokens (RX 7900 XTX), and the predictions scored

`benchmarks/q3moe_depth.py --tokens 32 --rounds 3`, five rounds of processes in the order A, B, C, D (one arm's file
in the page cache at a time), every arm on the study module `k3_q3m_iq` at the card's grid; A also on the shipped
module in the same process. ms per token, median of the five rounds (lowest .. highest):

| Arm | Depth 8 | 1,024 | 4,096 | 16,384 | Predicted at depth 8 | Reference line |
|---|---|---|---|---|---|---|
| A, shipped module | 4.041 (4.028 .. 4.046) | 4.235 | 4.700 | 6.221 | | 4.03 |
| A, study module | 4.141 (4.140 .. 4.142) | 4.350 | 4.870 | 6.408 | | |
| B | 4.122 (4.118 .. 4.139) | 4.353 | 4.888 | 6.427 | 3.90–4.00 | 3.87 |
| C | 4.807 (4.798 .. 4.824) | 5.092 | 5.677 | 7.194 | 3.70–3.95 | 3.62 |
| D (not scored: rule 4) | 4.688 (4.673 .. 4.696) | 4.970 | 5.568 | 7.077 | 3.40–3.70 | 3.30 |

Against A on the same module in the same round, at depth 8: B −0.003 to −0.023 ms, C +0.656 to +0.683, D +0.532 to
+0.556; at 4,096: B +0.003 to +0.032, C +0.788 to +0.814, D +0.694 to +0.722.

- **H2's direction holds and its numbers do not.** Every arm sits above its reference line, as predicted, but the
  ranges were too low: B by 0.12 ms and C by 0.86 (D, not scored, sits 0.99 above its range). The model behind them
  (a stage costs its bytes at the bus's speed plus a fixed 4 to 5 µs) is wrong for these tiles in this module:
  **arms C and D are slower than A**, with 18 % and 33 % fewer bytes per token. How much of that is the formats and
  how much the module's one shared grid body is taken apart under "After scoring" below.
- **B ties A.** Its 7 % fewer bytes buy 0.00 to 0.02 ms at depth 8 and nothing at 4,096.
- **Carrying the four formats costs A 0.10 ms** per token at depth 8 and 0.17 at 4,096 (same bits, shipped module
  against study module in one process): the study module spills 66 scalars against 53.

One traced token per arm (`benchmarks/q3moe_trace.py --modules k3_q3m_iq_trace`), µs per layer on the critical path:

| Arm | Expert gate and up | Expert down | Predicted | Everything else in the token |
|---|---|---|---|---|
| A (Q4_K) | 20.73 | 15.46 | (20.7 / 15.2 measured before) | 2.58 ms |
| B (IQ4_XS) | 21.29 | 15.09 | 19.7–21.0 / 12.8–14.0 | 2.59 ms |
| C (IQ3_XXS and IQ3_S; down also IQ4_XS) | 28.79 | 20.16 | 15.4–21 / 10.7–14 | 2.64 ms |
| C, its IQ3_XXS layers (32 gate and up, 25 down) | 28.87 | 21.48 | 15.4–20 / 10.7–13.5 | |
| C, its IQ3_S layers (16, 10) | 28.35 | 22.52 | 16.8–21 / 11.3–14 | |
| C, its IQ4_XS layers (down only, 13) | | 15.36 | 12.8–14.0 | |
| D (IQ2_XXS, not scored) | 26.15 | 20.29 | 11.7–17 / 8.8–12 | 2.61 ms |

- **The cost is inside the expert stages.** The stages outside them agree across arms to 0.06 ms. Staging the grids
  into shared memory happens at the launch, before the trace's first timestamp, so the trace cannot see it; the wall
  clock can. C is 0.667 ms slower than A on the wall and 0.669 ms in the trace, of which the expert stages are
  0.613; D is 0.547 and 0.515: staging costs at most the 0.03 ms between them.
- **In this module the grid formats are bound by their decode, not by the bus.** A layer's gate and up are 16 matvecs of 768 × 2048
  on 192 workgroups at once, 25.2 M weights: 20.7 µs as Q4_K (1.21 × 10^12 weights/s), 28.8 µs on C (0.87), 26.2 µs
  as IQ2_XXS (0.96). An IQ3_XXS layer's 9.6 MB of gate and up arrive at 334 GB/s of the bus's 858. Every "per stage" range is
  missed on the high side, IQ4_XS by 0.3 and 1.1 µs.
- **Why the standalone table above did not show it:** one matvec per launch is bound by the launch (12 µs), and the
  whole card decodes 1.6 M weights. A token's stage gives the same card sixteen times the weights in the same wall
  time.

### Loads against decode (RX 7900 XT)

`benchmarks/q3moe_depth.py --depths 8,4096`, three rounds of processes; each arm on the study module and on its
loads-only build (the same tapes with every expert weight read and no arithmetic). ms per token at depth 8, median:

| Arm | Full | Loads only | Decode (full − loads only, per round) | Predicted |
|---|---|---|---|---|
| A, shipped module | 4.833 | 4.519 | 0.310 – 0.315 | (0.57 measured earlier) |
| A, study module | 4.932 | 4.530 | 0.402 – 0.403 | |
| B | 4.945 | 4.374 | 0.565 – 0.600 | 0.55–0.75 |
| C | 5.769 | 4.189 | 1.580 – 1.596 | 0.75–1.2 |
| D (not scored) | 5.633 | 4.017 | 1.616 – 1.621 | 0.7–1.1 |

- **H1, read two ways.** Its "fails if" clause compares an arm's decode time with its loads time, and the table's
  loads-only token is everything but the decode (the attention, the head and every other stage included): by that
  reading it holds (C: 1.59 against 4.19 ms), with B inside its range and **C's range missed** by 0.4 ms. By its
  title ("decoding stays the smallest part") and per stage it fails for the grid formats: an IQ3_XXS layer's gate and
  up take 28.9 µs on the RX 7900 XTX for 11.2 µs of bytes at the bus's speed, so the decode is the larger part of
  the stage.
- **The bytes do buy their share when nothing is decoded:** loads-only tokens fall from 4.53 ms (A) to 4.37, 4.19 and
  4.02 ms. The decode then takes back 0.2 ms (B), 1.2 ms (C) and 1.2 ms (D) more than Q4_K's.
- A's decode on the shipped module is 0.31 ms here, not the 0.57 ms the predictions quote: that figure was measured
  before the engine took its per-card grid.

### Accuracy per arm at 128 tokens (RX 7900 XT)

`tools/gates/q3moe_gmodel.py` per arm: the engine's logits and llama.cpp's one-token decode (f16 KV, one card)
against an fp32 reference computed from that arm's own decoded weights (`tools/reference/q3moe_ref32.py`, which
gained the four codebook decoders: bitwise the host decoders on random blocks, `tests/io/test_own_dequant.py`),
teacher-forced over the same 128 tokens. The fp32 reference against the float64 one: mean KLD 6.9e-11 over arm A's
128 positions, at most 1.9e-10 over seven positions of B, C and D, top-1 equal everywhere.

| Arm | Caminus mean KLD (max) | llama.cpp mean KLD (max) | llama.cpp over Caminus | Top-1, Caminus / llama.cpp |
|---|---|---|---|---|
| A (study module) | 4.66e-04 (9.7e-03) | 2.83e-03 (4.0e-02) | 6.1 | 128 / 127 |
| B | 1.45e-04 (4.2e-03) | 1.82e-03 (1.9e-02) | 12.6 | 128 / 125 |
| C | 4.22e-04 (1.5e-02) | 2.03e-03 (1.4e-02) | 4.8 | 127 / 128 |
| D | 4.34e-05 (2.2e-03) | 4.14e-03 (1.2e-01) | 95 | 128 / 126 |

- **H4's ratio holds on every arm** (at least 3; measured 4.8 to 95).
- **H4's top-1 clause fails on C by one token:** the engine misses one of 128 where llama.cpp misses none, so the
  gate script ends RESULT FAIL on C. On A, B and D the engine misses none and llama.cpp misses 1, 3 and 2. The miss
  is position 37, where the reference's two best tokens are 1.8 % apart (0.018 in log probability; the engine has
  them 0.076 the other way, llama.cpp 0.117 the reference's way). The RX 7900 XTX gives the same numbers (mean KLD
  4.221e-04, 127 of 128).
- The engine's means are set by a few positions (medians 2e-06 to 3e-05; C has eight positions above 1e-03). A
  router pick flipping where two experts tie is shown for arm D at one position above; for the positions here it is
  inferred, not tested.

### llama.cpp on the same files (RX 7900 XTX)

`llama-bench -ngl 99 -fa 1 -p 512,4096 -n 64 -d 0,4096 -r 3` (under the guard, one file at a time), tok/s; Caminus
beside it from the table above (decode, the study module, depth 8 and 4,096):

| File | llama.cpp decode, empty / at 4,096 | Caminus decode, 8 / 4,096 | llama.cpp prefill of 4,096 |
|---|---|---|---|
| A, the official Q4_K_M | 149.8 / 135.3 | 241.5 / 205.3 (shipped module 247.5 / 212.8) | 3,885 |
| B (IQ4_XS experts) | 152.0 / 137.0 | 242.6 / 204.6 | 3,511 |
| C (UD-IQ3_XXS experts) | 148.8 / 134.4 | 208.0 / 176.2 | 3,484 |
| D (IQ2_XXS experts) | 155.1 / 137.5 | 213.3 / 179.6 | 3,837 |
| unsloth IQ4_XS, as published | 154.7 / 139.3 | (not run: the study measures the spliced arms only) | 3,552 |
| unsloth UD-IQ3_XXS, as published | 150.1 / 134.9 | (not run) | 3,533 |

llama.cpp's decode moves by 4 % across the four arms (148.8 to 155.1 tok/s) for a third fewer bytes: there too the
bytes are not what a token waits for. Its prefill is 10 % slower on B and C than on A, and 1 % on D.

## After scoring: the shared grid body taken apart (RX 7900 XT)

Nothing above is rescored: every table above is the study module as it was when the predictions were scored. This
section asks what the scored result leaves open: arms C and D decode slower than A with fewer bytes; is that the
formats, or the one shared body the 96-register fit forced on the three grid formats?

Two experiment builds answer it. Neither is an arm and neither ships: the module with the grid body's format fixed at
compile time to arm D's (`OWN_K3_IQG_ONE`), and the shared body without its opaque lane copy (`OWN_K3_IQG_NOCOPY`).
`benchmarks/q3moe_depth.py --depths 8,4096 --tokens 32 --rounds 5`, the builds of an arm interleaved as separate
processes, ms per token at depth 8 / 4,096 (decode):

| Arm | Shared body as scored | Shared body reworked | Format fixed at compile time | Loads only |
|---|---|---|---|---|
| A | 4.935 / 5.756 | 4.936 / 5.736 | | 4.555 / 5.323 |
| B | 4.964 / 5.799 | 4.964 / 5.771 | | 4.406 / 5.171 |
| C | 5.785 / 6.695 | 5.097 / 5.943 | (three formats in one file: no such build) | 4.197 / 4.976 |
| D | 5.639 / 6.540 | 5.121 / 5.983 | 4.957 / 5.837 | 4.028 / 4.801 |

- **Most of the scored gap was the shared body's form, not the formats.** As scored, the body chose the format inside
  each of a unit's eight grid lookups and eight sign nibbles (both forms computed, then a select). Reworked, what
  differs between the formats is settled once per unit and the lookups and dot products are the same statements for
  all three: IQ2_XXS's grid is staged as two half-tables, so a lookup's address is an index byte plus one bit × 256
  in every format; the sign codes are expanded to 32 sign bits once; the scale is (1 + 2 s) times a constant. C gains
  0.69 ms per token and D 0.52; A and B do not move.
- **Same bits.** Logits of 128 positions on B, C and D are bitwise the scored module's, A's are bitwise the shipped
  module's, and `tests/engine/test_own_iq_ops.py --layers all` passes on C and D (worst row 2.1e-6 of the rms against
  float64). So the accuracy table above holds for the reworked body as it stands.
- **D, taken apart:** with its format fixed at compile time arm D ties A on the same module (4.957 against 4.936 ms).
  Its bytes save 0.53 ms of loads and its decode takes 0.93 ms against Q4_K's 0.38: IQ2_XXS's decode costs what its
  bytes save. One code object for three formats costs another 0.16 ms (5.121 against 4.957). The opaque lane copy is
  not that cost: without it the token is 0.01 to 0.02 ms faster, below what this benchmark resolves.
- **C's floor is unknown.** Only its gain is measured (5.785 to 5.097 ms); it is still 0.16 ms slower than A. No
  single-format build can run a file with three expert formats.
- **Decode time (full minus loads only) with the reworked body:** A 0.38 ms, B 0.56, C 0.90 (1.59 as scored),
  D 1.09 (1.61 as scored; 0.93 with the format fixed).
- **k3_run:** 96 VGPRs, 0 spilled, 64 spilled scalars (66 as scored); the trace build 96, 0, 79. The shipped
  modules' code is unchanged, instruction for instruction; of the 67 kernel sources, 64 preprocess to the same text
  as before the rework (the study module, its trace build and the standalone matvec kernels differ).

The standalone tiles share the body, so the alternated census of stop rule 4 was run again on the reworked one
(`benchmarks/own_mv_census.py`, every role of the token, 5 rounds, the two files alternated three times; µs per
matvec, own tile and (own minus read-only)):

| Shape (role) | IQ3_XXS | IQ2_XXS |
|---|---|---|
| 768 × 2048 (gate) | 13.22, 13.12, 13.18 (0.74, 0.68, 0.84) | 12.95, 13.27, 13.34 (0.47, 0.73, 0.53) |
| 768 × 2048 (up) | 13.22, 13.25, 13.21 (0.78, 0.71, 0.84) | 13.29, 13.34, 13.17 (0.85, 0.75, 0.52) |
| 2048 × 768 (down) | 9.74, 9.46, 9.54 (0.54, 0.57, 0.55) | 9.72, 9.39, 9.40 (0.53, 0.50, 0.49) |

The two tiles are within this instrument's spread of each other on every shape. The rule stays unruled.
