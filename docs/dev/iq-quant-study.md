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
| D | own requant | IQ2_XXS everywhere | 0.467 GB (estimate) | 1.290 GB (estimate) | 1.50 ms, 665 tok/s |

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

In the decode kernel a 16-value table fits in registers, and the three grids go to the workgroup's local memory (5 KB
for all three), staged at the start of a launch by the arms that need them. A grid read depends on the index load, so
a pass of the tile becomes a memory round trip and then a local one: that second dependent step is what this study
measures.

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
- All three grids need 5 KB of local memory, not 3.
- The draft called the 2.29 ms a token spends outside loads and arithmetic "hand-off". A trace has since shown that
  time is inside the stages: the hand-offs are 0.09 ms of a token.
- The decode engine took a per-card grid after the draft was written; the reference line and stop rule 3 follow it.
