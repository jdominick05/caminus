# Row-scaled integer codes on Qwen3-30B-A3B's experts: a test

Status: the quality half is measured and scored (Results: quality); the decode tile is written and predicted, not yet run
on a card. Follows [the IQ quant study](iq-quant-study.md), which found that
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
| P5 | the shifted 2-bit set's fit error at least 20 % below the container's on every role | by the fitter's per-role lines: down 1.7 % **above**, gate 12.3 % below, up 13.3 % below | **Fails** by its letter on down, and no role reaches 20 %. Per tensor the shifted set is 9 to 16 % better on every one of the first 116 tensors logged (a 4 % exception on one gate), so the per-role line for down, a sum weighted by each tensor's activation energy, is carried by the last layer (blk.47: 19 % worse shifted). The level set is worth about 13 % of the fit error, not 20, and not on the tensor that weighs most. |

What the table says:

- **The row scale is what costs at 4 bits, and it costs 2.3 times** (W4r 0.0325 against W4b 0.0144). It still lands
  at 1.19 times the official file's distance and above its top-1 agreement (92.7 % against 92.0 %), at 4.0 bits per
  expert weight against 4.5 to 6.6. The tail is heavier: p99 0.289 against 0.190.
- **Arm A is not a strong baseline at its size.** The official file's experts were made without this imatrix; plain
  Q4_0 with it (W4b, W4L) and IQ4_XS (B) are all about half A's distance. A row-scale file competes with A, not
  with the best 4.5-bit file.
- **llama.cpp's own rule about the first `ffn_down` layers is measurable** (W4L 0.0181 with them at Q4_0, W4Lg 0.0153
  with its six Q4_1 tensors), and this fitter's Q4_0 beats both without it.
- **2 bits by rounding is not usable:** W2r agrees with R on 72 % of positions, IQ2_XXS at 2.06 bits on 83 %. The
  codebook buys a factor of 2.7 in KLD at the same size; the row scale costs another 1.4 times over a block scale.
  A usable plain 2-bit file needs training (or at least calibration through the model), which this test excludes.
- **8-bit activations with one scale per vector, computed at run time, cost 0.0079** on Q8_0 experts: under a third
  of what the official file's expert quantization costs. On W4r they add 0.004 (12 %); on W2r nothing measurable.
  No calibration data was needed. The int8-activation tile is worth writing.

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
