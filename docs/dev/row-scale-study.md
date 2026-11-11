# Row-scaled integer codes on Qwen3-30B-A3B's experts: a test

Status: both halves and the int8-activation probe are measured and scored (Results: quality, Results: speed, The
int8-activation probe). Follows [the IQ quant study](iq-quant-study.md), which found that
the codebook formats' smaller files buy no decode speed on these cards: their decode costs what their bytes save.

## The question

Would a format built for the kernel do better: plain fixed-width integer codes (4-bit or 2-bit, unpacked with a shift
and a mask), **one scale per output row** applied once after the dot product, and 8-bit activations? Two halves:

1. **Quality.** What does it cost against the same model with Q8_0 experts, when it is made by conversion alone (no
   training, no calibration passes through the model)?
2. **Speed.** What does a decode tile for it cost on the RX 7900 XT and the RX 7900 XTX, against Q4_K's and the
   codebook formats' in the same module? Written and predicted while the quality half's window ran: before any
   timing, and before any accuracy number was read.

Only the expert tensors change (144 of the file's 579), as in the IQ study, so the result is about the experts'
format and nothing else.

## The arms

Every file is the official Q4_K_M file with its expert tensors replaced (`tools/eval/gguf_splice.py`,
`tools/eval/requant_rows.py`), and every requantized arm starts from the same Q8_0 experts (unsloth's Q8_0 file).

| Arm | Experts | Bits per weight | Made by |
|---|---|---|---|
| R | Q8_0 | 8.5 | the reference: the donor's experts as they are |
| A | Q4_K gate and up, Q6_K / Q4_K down | 4.5 to 6.6 | the official file |
| B | IQ4_XS | 4.25 | the IQ study's arm |
| C | IQ3_XXS / IQ3_S / IQ4_XS | 3.1 to 4.25 | the IQ study's arm |
| D | IQ2_XXS | 2.06 | the IQ study's arm |
| W4r | 4-bit codes, one scale per row | 4.0 + 16 bits per row (stored here in Q4_0 blocks that repeat the row's scale) | `requant_rows.py --bits 4 --scale row` |
| W4b | 4-bit codes, one scale per 32 values (Q4_0) | 4.5 | `requant_rows.py --bits 4 --scale block` |
| W4L | Q4_0 | 4.5 | llama-quantize with the same imatrix |
| W2r | 2-bit codes, one scale per row | 2.0 + 16 bits per row (stored in Q2_0 blocks) | `requant_rows.py --bits 2 --scale row` |
| W2b | 2-bit codes, one scale per 64 values (Q2_0) | 2.25 | `requant_rows.py --bits 2 --scale block` |

- **The 4-bit level set** is Q4_0's: -8 .. 7 times the scale. **The 2-bit level set** is the Q2_0 container's:
  -1, 0, +1, +2 times the scale, which is not symmetric. The fit logs, per tensor, what a set shifted by half a
  step would cost (codes ±0.5, ±1.5: symmetric, no zero), so the container's share of the error is known; that set
  is not written to a file.
- **The fit** is the same for the row and the block scale: per group, the scale and codes that minimize the squared
  weight error weighted by unsloth's imatrix (the mean squared activation of that expert's input column), by a grid
  of starting scales, three rounds of assign-and-refit each, the scale rounded to fp16 and the codes assigned again.
  W4b against W4L tells whether this fitter is as good as llama.cpp's for the same container; W4r against W4b is
  then what the row scale costs.
- **8-bit activations** are measured in the reference tool, not as files: `tools/reference/q3moe_ref32.py --act8`
  rounds every expert matvec's input vector to 8-bit integers with one scale per vector (the largest magnitude maps
  to 127). No calibration data: the scale is computed from the vector at run time. Run on R (what the activations
  alone cost), W4r and W2r.

## What is measured (quality)

`tools/reference/q3moe_ref32.py` computes each arm's logits in fp32 from its exactly decoded weights, so no engine's
rounding is in the comparison: the differences are the formats'. One token sequence for every arm (the tree's own
docs and sources as tokenized for the engine's 16,384-token depth gate, its first 2,048 tokens), the last 1,024
positions scored against arm R:

- mean, median and 99th percentile of KLD(R ‖ arm),
- top-1 agreement with R.

Arm R is itself a quantized model (Q8_0 experts, the official file's other tensors); the numbers are distances to
that model, not to the unquantized one.

## Predictions

Written before the first requantized file exists.

| | Prediction | Fails if |
|---|---|---|
| P1: the row scale costs accuracy at 4 bits | W4r's mean KLD is 1.5 to 4 times arm A's; W4b's is between A's and W4r's | W4r at or below A (the row scale is free), or above 4 times A |
| P2: the fitter is not the handicap | W4b's mean KLD is within 30 % of W4L's | W4b more than 30 % above W4L: then W4r's number is the fitter's, not the format's |
| P3: 2 bits by rounding does not hold up | W2r's mean KLD is at least 3 times arm D's and its top-1 agreement with R is below 85 %; W2b is between D and W2r | W2r within 1.5 times D |
| P4: 8-bit activations with one scale per vector are not free | R with `--act8` has a mean KLD between 0.1 and 1 times arm A's (a vector's largest entry sets the step for all 2,048) | above arm A's: then the activations need a scale per block, or a rotation, before any integer tile is worth writing |
| P5: the shifted 2-bit level set is the better one | its weighted fit error is at least 20 % below the container's on every role | less than 5 % below: the Q2_0 level set is not what limits W2 |

Bytes read per token, from the tensor sizes (8 of 128 experts per layer; the other tensors read 0.82 GB): A 1.920 GB,
B 1.785, C 1.565, D 1.290, W4r 1.728 (10 % under A), W2r 1.275 (34 % under A). On the RX 7900 XT a token of arm D
with its decode removed is 0.50 ms shorter than A's of 4.5 ms (the IQ study's loads-only builds), so **a 2-bit tile
with a free unpack could gain about 11 % of a token there, and a 4-bit one about 3 %**; the decode half asks how
much of that a real tile keeps.

## Changes to the tools after the predictions were committed

Both were made after the fitter's first run had printed its first three tensors and before any file was complete;
no prediction was reworded.

- **The starting scales** were multiples of the group's largest weight (as a 32-value block wants). A row of 2,048
  weights has its best step at a small fraction of its largest weight, outside that grid, so the grid is now a
  geometric range of multiples of the group's rms (4-bit 0.10 to 1.0, 2-bit 0.35 to 3.0, 16 values, both signs).
  On layer 0 this moved the 2-bit row fit's weighted error from 0.1437 to 0.1419 (down), 0.1782 to 0.1729 (gate).
- **The shifted level set's logged error** clamped its codes to the container's range around the half step, which
  gave the levels -0.5, +0.5, +1.5, +2.5: not the symmetric set the plan describes. It now clamps to ±0.5, ±1.5.
  Only the logged comparison for P5 changed; the codes written to the files never used it. This fix came after
  the wrong set's first output, which contradicted P5 (layer 0's down: 0.2400 for the "shifted" set against
  0.1437 for the container's), so it is checked independently: the fitter's statements on unit-Gaussian rows of
  2,048 values on the CPU give 0.1461 for the container's set, 0.1189 for ±0.5, ±1.5 (the known optimum of a
  uniform 4-level quantizer on a Gaussian is 0.1188) and 0.2547 for the first run's clamp, with exactly those
  level sets.
- The fitter writes `OUT.part` and renames it when the file is complete, so an interrupted run leaves no file under
  an arm's name.
- **How the tools read a file.** The fitter and the reference tool copied tensor bytes to the card straight from
  the memory-mapped file. The donor is 32.5 GB on a host with 29 GiB of RAM: the first complete fit slowed when the
  page cache filled (23 GB in) and then stopped advancing, the card idle and no fault recorded; it was stopped by
  process id at 116 of 144 tensors and its partial file discarded. Both tools now read tensor bytes into ordinary
  memory (`ggufio.Tensor.pread`), and the fitter drops the donor's and the output's pages from the page cache as it
  goes. The fit is the same: the restarted run printed all 116 of the first run's per-tensor errors digit for
  digit, and ran on past the tensor where the first had stopped at the same 8 s per tensor as before it.
- **Both cards at once.** The window was planned for the RX 7900 XT alone. After the restart the 2-bit files and
  the 8-bit-activation runs are made on the XT and the 4-bit files and the Q4_0 control on the RX 7900 XTX, each
  under its own guard; the reference tool's first run is the XT's. Arm A's logits are computed on both cards, and
  their distance is reported with the table: what the card itself adds to a distance.
  Written before any logits exist: every distance is to arm R as computed on the XT. Where a prediction compares
  an arm computed on the XTX with arm A (P1), A's number is the XTX's; P2, P3 and P4 each compare arms of one card.

- **The Q4_0 control is two files.** llama-quantize does not write a pure Q4_0 file when it is given an imatrix:
  by a rule of its own ("guard against craziness in the first few ffn_down layers") it stores `ffn_down` of the first
  eighth of the layers, here 6 tensors, as Q4_1 (a scale and an offset per 32 values, 5.0 bits per weight). The
  reference tool had no Q4_1 decoder, so that file's logits failed to compute. Decided before any logits were
  compared: P2 is about the same container, so its control (**W4L**) is llama-quantize with the same imatrix and
  `--tensor-type ffn_down_exps=q4_0`, all 144 expert tensors Q4_0. The file as llama-quantize makes it by default
  is reported beside it as **W4Lg** (it needed a Q4_1 decoder in `caminus.own_dequant` and the reference tool,
  checked against gguf-py like the others). W4b and W4r keep Q4_0 on those six tensors.

## Results: quality

Every file's fit ended RESULT PASS (stop rule 2). Distances to arm R over the last 1,024 of 2,048 positions
(`tools/eval/logits_kld.py`); the 2-bit files, R, A, D, the controls and the 8-bit-activation runs were computed on the
RX 7900 XT, B, C, W4r and W4b on the RX 7900 XTX. **The card adds nothing:** arm A computed on the XTX is arm A
computed on the XT at every position (KLD 0, 1,024 of 1,024 top-1), so no column below depends on the card.

| Arm | Experts | Mean KLD | Median | p99 | Top-1 agreement with R |
|---|---|---|---|---|---|
| A | the official file's Q4_K / Q6_K | 0.0273 | 0.0133 | 0.190 | 92.0 % |
| B | IQ4_XS | 0.0140 | 0.0060 | 0.105 | 94.8 % |
| C | IQ3_XXS / IQ3_S / IQ4_XS | 0.0379 | 0.0207 | 0.273 | 89.6 % |
| D | IQ2_XXS | 0.1735 | 0.0856 | 1.463 | 82.6 % |
| W4b | 4-bit, a scale per 32 values, this fitter | 0.0144 | 0.0067 | 0.116 | 93.8 % |
| W4L | Q4_0 by llama-quantize, all 144 tensors Q4_0 | 0.0181 | 0.0085 | 0.164 | 93.5 % |
| W4Lg | llama-quantize's default (6 tensors Q4_1) | 0.0153 | 0.0073 | 0.141 | 93.8 % |
| **W4r** | **4-bit, one scale per row** | **0.0325** | 0.0142 | 0.289 | **92.7 %** |
| W2b | 2-bit, a scale per 64 values | 0.3389 | 0.1696 | 2.261 | 76.2 % |
| **W2r** | **2-bit, one scale per row** | **0.4732** | 0.2503 | 3.511 | **72.3 %** |
| R + 8-bit activations | Q8_0 | 0.0079 | 0.0036 | 0.053 | 96.1 % |
| W4r + 8-bit activations | | 0.0365 | 0.0177 | 0.311 | 92.1 % |
| W2r + 8-bit activations | | 0.4729 | 0.2519 | 3.496 | 72.0 % |

| | Predicted | Measured | |
|---|---|---|---|
| P1 | W4r 1.5 to 4 times A; W4b between A and W4r | W4r 1.19 times A; W4b 0.53 times A | **Missed, on the cheap side.** The row scale costs less against the official file than predicted, and W4b is not between: it is closer to R than A is. The fail line (W4r at or below A) is not met. |
| P2 | W4b within 30 % of W4L | W4b 20 % **below** W4L (and 6 % below W4Lg) | **Holds.** The fitter is not the handicap: on the same container it is closer to R than llama-quantize's. |
| P3 | W2r at least 3 times D, top-1 below 85 %; W2b between | W2r 2.73 times D, top-1 72.3 %; W2b 1.95 times D | **Missed by the ratio (2.73 against 3), not failed** (the fail line is 1.5). The substance stands: 2 bits by rounding does not hold up. |
| P4 | R with 8-bit activations 0.1 to 1 times A | 0.29 times A | **Holds.** |
| P5 | the shifted 2-bit set's fit error at least 20 % below the container's on every role | W2r, by the fitter's per-role lines: down 1.7 % **above**, gate 12.3 % below, up 13.3 % below | **Fails** by its letter on down, and no role reaches 20 %. Of W2r's 144 tensors, 123 have a per-tensor line on record (layers 0 to 38 and 45 to 47; the rest scrolled out of the kept log): the shifted set is 4 to 17 % better on 122 of them (median 15 %) and 19 % worse on one, the last layer's `ffn_down`. For that one tensor to turn the down role's line from about 15 % below to 1.7 % above, it has to carry about half of the role's weighted error (arithmetic on the two figures, not a measurement). With a scale per 64 values (W2b, all 144 lines on record) the shifted set gains less: 0 to 8 % on 141 tensors, worse on three (the last `ffn_down` by 17 %, layer 0's gate and up); per role down 5.2 % above, gate 3.2 % and up 3.9 % below. |

What the table says:

- **The row scale is what costs at 4 bits, and it costs 2.3 times** (W4r 0.0325 against W4b 0.0144). It still lands
  at 1.19 times the official file's distance with the same top-1 agreement (92.7 % against 92.0 %: 7 positions of
  1,024, inside the noise). The tail is heavier: p99 0.289 against 0.190. On the device its experts are 4.0 bits per
  weight plus one fp32 per row; the study file keeps the Q4_0 container and is W4b's size (17.3 GB).
- **Arm A is not a strong baseline at its size.** Plain Q4_0 with unsloth's imatrix (W4b, W4L) and IQ4_XS (B)
  are all about half A's distance. The official file carries no `quantize.imatrix` metadata (llama-quantize writes
  those keys when it is given one: unsloth's Q8_0 and this study's llama-quantize files have them), and its name
  metadata reads "Qwen3 30B Gptq Fp16": its experts may also come from other source weights than unsloth's Q8_0, so
  part of A's distance to R may be the source and not the quantization. That split is not measured here. A row-scale file competes with A, not
  with the best 4.5-bit file.
- **llama.cpp's own rule about the first `ffn_down` layers is measurable** (W4L 0.0181 with them at Q4_0, W4Lg 0.0153
  with its six Q4_1 tensors), and this fitter's Q4_0 beats both without it.
- **2 bits by rounding is not usable:** W2r agrees with R on 72 % of positions, IQ2_XXS at 2.06 bits on 83 %. The
  codebook buys a factor of 2.7 in KLD at the same size; the row scale costs another 1.4 times over a block scale.
  A usable plain 2-bit file needs training (or at least calibration through the model), which this test excludes.
- **8-bit activations with one scale per vector, computed at run time, cost 0.0079** on Q8_0 experts: under a third
  of what the official file's expert quantization costs. On W4r they add 0.004 (12 %); on W2r nothing measurable.
  No calibration data was needed. So an int8-activation tile was worth measuring; the probe below bounds what it
  wins in decode (at most 0.08 ms of a token), and it is not built.

## The speed half: the tile, and its predictions

Written while the accuracy window runs: no row-scale tile has run on a card yet, and nothing below was timed.

**The tile** (`kernels/own_quant_rs.h`, modules `k3_q3m_rs4` and `k3_q3m_rs2`: the lean decode module plus one code
width each, so the shipped module's object does not change). The engine makes its planes at load
(`engine.own_layout.rs_planes`): the codes alone, K x bits / 8 bytes per row, and one fp32 scale per row in a buffer
of its own. A lane owns 32 values per pass, as in the Q4_K tile: one 16-byte load (4 bits) or 8-byte load (2 bits)
per row against Q4_K's two, a shift and a mask per four codes, the codes into fp16 exactly, `v_dot2_f32_f16` per
two weights, and the row's scale applied once to the wave's sum. No sub-block scale, no min term, no table. The
activations stay fp16: this tile is the weights' half only; an int8-activation form waits for P4.

**The register budget decided the layout.** With the row scales stored behind the codes (or in a header in front of
each row) and read through the weight pointer, `k3_run` spilled 6 vector registers in every form tried; read from a
buffer of their own, 0. The cost moved into scalars: both modules spill 66 scalars against the lean module's 53, a
price every segment hop of every wave pays. So each prediction below is against the official file **on the same
module**, and the module's own cost is its own row. The loads-only builds spill fewer: 54 on the row-scale modules,
48 on the lean one. So "full minus loads-only" carries 12 spilled scalars on a row-scale module against 5 on the lean
module, where S2's range was anchored (the official file there: 4.855 - 4.545 = 0.31 ms, of which the experts'
share by bytes is about 0.18 and the attention's and the head's matvecs, which this tile does not touch, 0.13).
The range stands as written: a tile at about 60 % of Q4_K's arithmetic gives 0.24, and the 7 extra scalars are
expected inside the range's width.

**Method.** `benchmarks/q3moe_depth.py --depths 8,4096 --tokens 32 --rounds 5`, an arm's builds (full and loads-only)
interleaved in one process, the arms as separate processes in turn, three passes, medians; the RX 7900 XT first,
then the RX 7900 XTX. Anchors from the IQ study on the XT at depth 8: A 4.855 ms on the lean module, 4.545 with
its expert weights read and nothing computed; arm D (1.290 GB per token) 4.03 loads-only, so 0.82 ms per GB read.
From the bytes, a loads-only token of W4r should be about 4.39 ms and of W2r about 4.02.

| | Prediction (RX 7900 XT, depth 8, ms per token) | Fails if |
|---|---|---|
| S1: the modules' own cost | the official file is 0.03 to 0.15 slower on `k3_q3m_rs4` / `_rs2` than on the lean module | more than 0.15 slower: the 13 extra spilled scalars cost more than a 4-bit tile can win back |
| S2: the tile decodes for less than Q4_K | W4r's and W2r's full minus loads-only is 0.20 to 0.40, the same for both widths (the arithmetic is per weight, not per byte), and below the official file's on the same module | either above the official file's |
| S3: 4 bits, net | W4r is 0.10 to 0.30 faster than the official file on the same module | less than 0.05 faster (below what this repository calls a speed difference) |
| S4: 2 bits, net | W2r is 0.45 to 0.70 faster than the official file on the same module, and at least 0.30 faster than the official file on the lean module | less than 0.30 faster on the same module |
| S5: depth | each arm's gain in ms at 4,096 positions is its gain at 8 within 0.05 (only the experts' stage changes) | a gain that shrinks by more than 0.05 |
| S6: the RX 7900 XTX | W4r 0.08 to 0.25 faster, W2r 0.35 to 0.60 faster than the official file on the same module | W2r less than 0.25 faster |

Correctness before any timing: `tests/engine/test_own_rs_ops.py` on each file (the tile against float64 on every row of
8 experts per role, and the K3 op bitwise its standalone kernel at four grids), on the XT alone first.

## Stop rules

1. If P2 fails, the fitter is fixed (or llama.cpp's weighting adopted) before W4r or W2r is read as the format's cost.
2. A file is only scored if `requant_rows.py` ends RESULT PASS on it (metadata and every other tensor are the base
   file's bytes; with a row scale, one scale per row in every expert tensor).
3. Every GPU command runs under `tools/guard.sh`, first on the RX 7900 XT.

## Results: speed

Correctness first, on each card before its timings: `tests/engine/test_own_rs_ops.py` on both files (RX 7900 XT, then
RX 7900 XTX: RESULT PASS on each; the tile against float64 on every row, mean 5e-08 and max 1e-06 of the rms; the
K3 op bitwise its standalone kernel at four grids), and on the XT `tests/engine/test_q3moe_one.py` on both files
against their fp32 references (mean KLD 1.5e-06 and 2.4e-07, top-1 equal at every position) and on the official file
on each row-scale module (logits bitwise the lean module's). Every object of the comparison was built from one source
tree into one directory; `k3_run` is 96 VGPRs with 0 VGPR spills in each.

Decode, ms per token (tokens per second), medians of three passes of five rounds of 32 tokens; each arm's full and
loads-only builds interleaved in one process. The widest spread of one build across its passes is about 0.05 ms.

| | XT, depth 8 | XT, 4,096 | XTX, depth 8 | XTX, 4,096 |
|---|---|---|---|---|
| Official file, lean module (as shipped) | 4.840 (206.6) | 5.660 (176.7) | 4.053 (246.7) | 4.705 (212.5) |
| Official file, row-scale module | 4.914 / 4.920 | 5.734 / 5.746 | 4.125 / 4.125 | 4.802 / 4.815 |
| **W4r** (`k3_q3m_rs4`) | **4.697 (212.9)** | 5.551 (180.1) | **3.972 (251.8)** | 4.653 (214.9) |
| **W2r** (`k3_q3m_rs2`) | **4.394 (227.6)** | 5.257 (190.2) | **3.716 (269.1)** | 4.408 (226.9) |
| B, IQ4_XS (`k3_q3m_iq`) | 4.925 (203.0) | 5.750 (173.9) | 4.129 (242.2) | 4.864 (205.6) |
| D, IQ2_XXS (`k3_q3m_iq`) | 5.115 (195.5) | 5.978 (167.3) | 4.128 (242.2) | 4.878 (205.0) |
| Loads only: official / W4r / W2r | 4.527 / 4.368 / 3.841 | 5.303 / 5.183 / 4.650 | 3.848 / 3.759 / 3.322 | 4.439 / 4.318 / 3.934 |
| Loads only: B / D | 4.369 / 4.004 | 5.161 / 4.792 | 3.744 / 3.454 | 4.329 / 4.048 |

Full minus loads-only (what the arithmetic adds to a token whose expert bytes are read anyway), paired inside each
process:

| | XT, depth 8 | XT, 4,096 | XTX, depth 8 | XTX, 4,096 |
|---|---|---|---|---|
| Official file on the lean module | 0.317 | 0.359 | 0.205 | 0.266 |
| Official file on a row-scale module | 0.397 | 0.425 / 0.444 | 0.260 / 0.266 | 0.354 / 0.375 |
| W4r | 0.329 | 0.368 | 0.220 | 0.329 |
| W2r | 0.552 | 0.607 | 0.392 | 0.471 |
| B, IQ4_XS | 0.559 | 0.593 | 0.392 | 0.535 |
| D, IQ2_XXS | 1.106 | 1.184 | 0.674 | 0.830 |

| | Predicted | Measured | |
|---|---|---|---|
| S1 | the official file 0.03 to 0.15 slower on the row-scale modules | 0.075 (`rs4`), 0.077 (`rs2`), paired per process | **Holds.** 13 more spilled scalars cost 0.08 ms. |
| S2 | W4r's and W2r's full minus loads-only 0.20 to 0.40, equal, below the official file's on the same module | W4r 0.329 (official 0.397); W2r 0.552 | **Fails on W2r**: outside the range, not W4r's, and above the official file's. W4r's part holds. |
| S3 | W4r 0.10 to 0.30 faster than the official file on the same module | 0.217 | **Holds.** Against the shipped lean module the gain is 0.143. |
| S4 | W2r 0.45 to 0.70 faster on the same module, at least 0.30 against the lean module | 0.526; 0.446 | **Holds.** |
| S5 | each gain at 4,096 positions within 0.05 of its gain at 8 | W4r 0.217 -> 0.183; W2r 0.526 -> 0.489 | **Holds**, with both gains 0.03 to 0.04 smaller at depth: close to the spread between processes. On the XTX the gains do not move (0.153 -> 0.149, 0.409 -> 0.407). |
| S6 | on the XTX W4r 0.08 to 0.25, W2r 0.35 to 0.60 faster on the same module | 0.153; 0.409 | **Holds.** |

What the tables say:

- **A plain 4-bit tile with one scale per row is the cheapest arithmetic measured here, and the gain is small.**
  W4r is 0.14 ms (XT) and 0.08 ms (XTX) faster than the shipped engine on the official file at depth 8: 3 % and 2 %.
  It reads 0.16 ms fewer bytes and computes for the same 0.33 ms the lean module's Q4_K tile takes. At 4,096
  positions on the XTX the gain against the shipped module is 0.05 ms: at the edge of what counts as a difference.
- **2 bits read 0.69 ms less than the official file on the XT and give back 0.24 ms of it in arithmetic.** The S2
  prediction (the arithmetic is per weight, so the same at both widths) is wrong: W2r's full minus loads-only is
  0.55 ms against W4r's 0.33. The tile does the same number of multiply-adds for half the bytes, so the faster the
  loads finish, the more of the arithmetic is left uncovered by them: at 2 bits the experts' stage is no longer
  bound by the bytes alone. The quality half rules W2r out, so its tile is not tuned further; the number stands as
  what plain 2-bit codes would have to beat.
- **The codebook formats pay for their bytes in arithmetic; the plain codes do not.** D reads 0.52 ms less than the
  official file on the XT and computes 0.79 ms more (5.115 against 4.840); W2r reads 0.69 ms less and computes
  0.24 ms more. At the same expert bytes per token (IQ4_XS and W4r both load in 4.37 ms on the XT), the plain tile
  is 0.23 ms faster than the codebook's (4.697 against 4.925).
- **The loads-only rows are the ceiling of any weight format on this engine**: with free arithmetic, 4-bit experts
  give 4.37 ms (229 tok/s) on the XT and 3.76 ms (266 tok/s) on the XTX at depth 8; 2-bit experts 3.84 ms (260) and
  3.32 ms (301). Everything else in a token (attention, the router, the output head, the hand-offs) is in those
  figures and does not shrink with the experts.
- **What an int8-activation form of the 4-bit tile could win is bounded by W4r's 0.33 ms (XT) and 0.22 ms (XTX)**
  of arithmetic, at a quality cost the quality half measured (KLD 0.0325 -> 0.0365 on W4r).

## The int8-activation probe

P4 held (8-bit activations cost a KLD of 0.0079 on Q8_0 experts), and the speed results bound what such a tile can
win: the arithmetic W4r adds to its loads, 0.33 ms on the RX 7900 XT. Before anything is built for it, one
measurement build says how much of that an integer tile would remove: `OWN_A8_PROBE` in `kernels/own_quant_rs.h`
reads a unit's 32 activations as 32 bytes of the activation buffer (half the bytes of fp16), multiplies them with the
unit's codes through `sudot4` (8 instructions per unit and row where the fp16 form takes 16 `fdot2` and builds an
fp16 pair for each), and scales the unit's integer sum once. Its sums are not the model's (the bytes are whatever the
buffer holds), as with the loads-only builds: it is timed, never scored.

What it leaves out, so that it is an upper bound: producing the int8 activations (a quantizing step inside the ops
that write the experts' inputs, or segments of their own, with their hops), and reading a scale per unit if the
format has one.

Written before the probe is built or run (RX 7900 XT, depth 8, the file's full, loads-only and probe builds
interleaved in one process, three passes):

| | Prediction | Fails if |
|---|---|---|
| A1: 4 bits | the probe token is 0.13 to 0.25 ms faster than W4r's full tile (probe minus loads-only 0.08 to 0.20) | less than 0.05 faster: the 0.33 ms is not the fp16 arithmetic's instructions, and int8 activations buy nothing here |
| A2: 2 bits | the probe token is 0.25 to 0.43 faster than W2r's full tile (probe minus loads-only 0.12 to 0.30) | less than 0.10 faster |
| A3: the widths | probe minus loads-only is 0.03 to 0.12 higher on W2r than on W4r (the same instructions over shorter loads; S2 predicted equal and failed) | equal within 0.03, or more than 0.15 apart |

**Result** (RX 7900 XT; every object built from one tree into one directory; the probe objects are 96 VGPRs, 0 VGPR
spills, 66 spilled scalars as the full ones, with 32 `v_dot2acc_f32_f16` fewer and 24 `v_dot4` more than the full
object; medians of three passes, differences paired inside each process):

| ms per token | W4r, depth 8 | W4r, 4,096 | W2r, depth 8 | W2r, 4,096 |
|---|---|---|---|---|
| Full tile | 4.696 | 5.532 | 4.406 | 5.260 |
| Probe (int8 activations in memory) | 4.616 | 5.435 | 4.263 | 5.091 |
| Loads only | 4.388 | 5.178 | 3.856 | 4.648 |
| Full minus probe | 0.081 | 0.100 | 0.148 | 0.169 |
| Probe minus loads-only | 0.223 | 0.257 | 0.407 | 0.443 |

| | Predicted | Measured | |
|---|---|---|---|
| A1 | the probe 0.13 to 0.25 faster than W4r's full tile | 0.081 | **Missed low** (the fail line is 0.05). |
| A2 | the probe 0.25 to 0.43 faster than W2r's full tile | 0.148 | **Missed low** (the fail line is 0.10). |
| A3 | probe minus loads-only 0.03 to 0.12 higher on W2r than on W4r | 0.184 higher | **Fails** (more than 0.15 apart). |

- **Int8 activations are worth at most 0.08 ms of a 4.70 ms token here (1.7 %) in this tile's form**, and that is
  before the activations are produced: the probe reads bytes that are already there. Per unit of 32 weights and row
  the probe runs 8 `v_dot4` where the full tile runs 16 `v_dot2acc_f32_f16` and builds an fp16 pair of codes for each
  (plus 8 `v_dot4` per unit for the activations' sum, shared by its rows), and reads 32 activation bytes where the
  full tile reads 64. That removed a quarter of what the tile adds to its loads (0.08 of 0.31 ms). The other 0.22 ms
  is not attributed by this probe: it still holds the probe's own integer multiply-adds, the activation reads (the
  loads-only build reads none) and the waiting of every sum on its loads.
- The RX 7900 XTX was not run. Its full minus loads-only is smaller (0.22 ms against the XT's 0.33), so a smaller
  bound is expected there; that is an expectation, not a measurement.
- **So the int8-activation tile is not built.** 8-bit activations are nearly free in accuracy (P4), and in this
  engine's decode they are bounded at 1.7 % of a token before their own cost. Expected, not measured here: int8
  activations pay where the activations are the large operand, as in batched prefill and verify passes (a GEMM over
  many tokens); a decode matvec over 8 active experts reads megabytes of weights against a few KB of activations that
  every row shares.

**The measurement builds.** A loads-only or probe object is a two-line source, `#define OWN_READONLY 1` (or
`#define OWN_A8_PROBE 1`) followed by `#include "k3_q3m_rs4.hip"` (`_rs2`, `_lean`, `_iq`), compiled with its
module's own command from `build.sh` under the name `<module>_ro` (`<module>_a8p`).
`benchmarks/q3moe_depth.py --arms NAME=<object>` gives an object named after a module that module's ops. Every
object of one comparison is built from one tree into one directory (`CAMINUS_KERNEL_DIR`).
