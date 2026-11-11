# Row-scaled integer codes on Qwen3-30B-A3B's experts: a test

Status: predictions written; nothing measured yet. Follows [the IQ quant study](iq-quant-study.md), which found that
the codebook formats' smaller files buy no decode speed on these cards: their decode costs what their bytes save.

## The question

Would a format built for the kernel do better: plain fixed-width integer codes (4-bit or 2-bit, unpacked with a shift
and a mask), **one scale per output row** applied once after the dot product, and 8-bit activations? Two halves:

1. **Quality.** What does it cost against the same model with Q8_0 experts, when it is made by conversion alone (no
   training, no calibration passes through the model)?
2. **Speed.** What does a decode tile for it cost on the RX 7900 XT and the RX 7900 XTX, against Q4_K's and the
   codebook formats' in the same module? Written and predicted after the quality half, before any timing.

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

## Stop rules

1. If P2 fails, the fitter is fixed (or llama.cpp's weighting adopted) before W4r or W2r is read as the format's cost.
2. A file is only scored if `requant_rows.py` ends RESULT PASS on it (metadata and every other tensor are the base
   file's bytes; with a row scale, one scale per row in every expert tensor).
3. Every GPU command runs under `tools/guard.sh`, first on the RX 7900 XT.
