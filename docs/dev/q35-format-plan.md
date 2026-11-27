# Per-tensor format planning for the dense qwen35 line (CPU)

`tools/q35_format_plan.py` chooses a quantization format per tensor of a dense qwen35 GGUF (the Qwen3.8-27B line: Swift 1.5,
ThinkingCap, the plain GSQ-RCO IQ3_S file) to minimize the **predicted** decode time per token on one card, under a VRAM
cap and a quality-loss cap — instead of minimizing file size. It is CPU-only and reads the GGUF header alone: no model
download, no card. Every time it prints is a cost-model prediction and is labelled as one; the measured yardsticks are
printed separately and never mixed in.

```
python -I tools/q35_format_plan.py GGUF [--card XTX] [--depth 128] [--max-ctx 32768] [--vram-cap GiB]
    [--quality-cap 1e-2] [--costs FILE] [--sensitivity FILE] [--set TENSOR=FORMAT ...] [--module dec_q35_gsq] [--json OUT]
```

The planner lives in `python/caminus/engine/q35_plan.py`; the file-level facts it shares with the engine are in
`python/caminus/engine/q35_file.py` (moved out of `engine/q35_dense.py`, which re-exports them). Tests:
`tests/tools/test_q35_format_plan.py` (the `q35_format_plan` suite of `tests/ci_cpu.py`, which needs `torch` — CI installs
the CPU wheel), on synthetic headers built with `tests/tiny_gguf.py`: the solver is checked against brute force on
cases small enough to enumerate (320 random instances across binding, non-binding and infeasible VRAM and quality
caps, with tied optima), and the refusals (unsupported pair, split fusion group, infeasible budget) are tested.

## What it takes in

- **The tensor inventory**, from the GGUF header via `caminus.ggufio`: per tensor its name, shape, type, bytes, role, the
  decode stage that reads it, how many times a token reads it, and its fusion group. Sources:
  - the decode module: `engine/q35_file.py module_of` (moved from `engine/q35_dense.py`), i.e. the same choice
    `Dense35Own` makes from the file's formats;
  - block kinds (Gated DeltaNet vs gated attention): `q35_file.block_kinds` (`attention.recurrent_layers`, else every
    block but each `full_attention_interval`-th);
  - the stage skeleton: `q35_dense.py _build_tape` (the tape the engine runs), re-expressed header-only in
    `q35_plan.stages_of` — per block a norm, the block's matvecs, its glue ops, the attention in an attention block, the
    feed-forward, the down matvec, the post norm; then the head's norm and matvec; the embedding's one row first;
  - reads per token: every trunk matrix and the head once per token, the embedding one row
    (docs/dev/qwen38-27b.md: "One decode token reads 16.48 GB of weights (every trunk matrix once; the embedding is one
    row)"); the multi-token-prediction blocks past the trunk are not read (q35_dense.py's docstring);
  - the fusion groups: **the feed-forward's `ffn_gate` | `ffn_up` of a block, fused into one segment (one launch) when
    `q35_file.gu_fused` says so** — the engine's own test, moved from `q35_dense.py _build_tape`
    (`gu and fg['fmt'] == fu['fmt'] and fg['fmt'] in k3.OWN_1RT_FMT and rpt % 16 == 0 and rpt <= 64`), i.e. only on the
    Swift module (`gu` is off on the ThinkingCap / GSQ modules, `Dense35Own.__init__`), only when both matrices share a
    one-round-trip format (Q4_K / Q6_K), at the served task size (rpt 16). No other fusion exists in the dense qwen35
    decode tape: in a GDN block `ssm_alpha` / `ssm_beta` / `attn_qkv` / `attn_gate` / `ssm_out` are separate segments, and
    in an attention block q, k, v, output are separate segments (q35_dense.py `_build_tape`). Tensors fused in one launch
    share a format; the planner refuses a request that splits a group.
- **The menu**: exactly the `(op, format)` pairs of the chosen module's `k3.MODULE_CAPS` (`engine/k3.py`), mapped to
  tensors by the op the tape runs them with (`q35_plan.tape_pair`: F32 matrices on `F32_MV`, the embedding on
  `OWN_EMB`, a one-round-trip format on `OWN_MV` (`..._gu` when fused), any other block format on `OWN_MV16`). Anything
  else is refused with the reason.
- **The cost table** (`tools/q35_format_plan.costs.json`, schema `q35-format-plan.costs/1`): per card, per format, per
  shape class, a stage's time as a fixed part plus a per-byte part (`fixed_us`, `per_byte_us`, or `us` + `bytes` of one
  measured matvec). Shape classes are the exact `rows x K` then a rows bucket (`wide` ≥ 16384, `mid` ≥ 4096, else
  `narrow`); the most specific entry wins, per-byte defaults to the card's published read rate. **Every number in the
  default table cites the published figure it comes from** (the `source` field, checked by the tests):
  - XTX read rate 858 GB/s — docs/dev/qwen38-27b.md:32 ("the 858 GB/s the RX 7900 XTX's memory delivers to this kind of
    read"; the decode-floor study's sustained rate). The read peak 952 GB/s (docs/dev/yardsticks.md:476) is deliberately
    not used. XT 774 GB/s — `arch.py:46 CARD_BW`, with the assumption noted that the XT's decode-read rate is not
    published;
  - mv stage fixed part 5 µs — docs/dev/qwen3-30b-a3b.md "Where the remaining time is" ("every weight stage of a layer
    costs 3 to 5 µs on top of its bytes, whatever its size"): the top of the published range;
  - XT matvec entries per format and exact shape, measured inside the decode kernel (`K3_TRACE`, DPM high, short
    context) on the four dense 27B files — docs/dev/benchmarks.md "Dense 27B: decode time against file size": the
    stage's time as `us` + `bytes` with the fixed part in `us`. A stage alone in its group is `measured: true`; a share
    of a sibling group's wall split by bytes (attn_qkv / attn_gate, unfused ffn_gate / ffn_up) is `measured:
    "group-split"` and flagged. A (format, shape) that runs two tiles on different modules (Q5_K 5120x17408) has no
    entry; the table has no module key;
  - XT add + norm 7.29 µs (one workgroup, 129 a token), XT embedding row, and XT attention (fixed 15.8 µs a layer + the
    KV at 295 GB/s: the line through depths 512 and 32,768 of the verify tape at T = 1) — the same section;
  - XTX norm 3.6 µs, attention 11.9 µs at 8 positions — docs/dev/qwen3-30b-a3b.md "Where the remaining time is",
    marked `other-model` (measured on Qwen3-30B-A3B's module, not the qwen35 line's);
  - glue (GDN conv/l2/step/save/norm, qk_nr, kv_append, gate_mul, SwiGLU) and the XTX embedding segment:
    **unmeasured** (`fixed_us: null`, `measured: false`, counted as 0 and flagged in every output that uses them). The
    XT glue ops leave 0.74-0.82 ms a token with no weight read in flight; that is a remainder, not a per-op
    measurement, so it is not an entry.
  - card VRAM caps: the guard lines and the display reserve — docs/dev/benchmarks.md:97, `serve/stats.py
    PhysicsConfig` defaults.
- **The sensitivity table** (`tools/q35_format_plan.sensitivity.example.json`, schema
  `q35-format-plan.sensitivity/1`): per tensor and format, the quality loss of that tensor in that format; the file's own
  format is the zero baseline; `default` and a `'*'` entry are fallbacks, and a loss the table does not give is refused,
  never guessed. **The shipped file is a synthetic example** (`"synthetic": true`) — the real one needs the model and a
  GPU: it would be filled by a **per-tensor arm of G-model** (docs/specs/spec-own-quant.md; `tools/gates/q35_gmodel.py`):
  the gate prompts, the engine's fp32 reference mode's logits, one tensor requantized per run, the mean KLD of one decode
  against the reference. The unit is stated in the table's `unit` field.
- **The constraints**: the format menu (above), a VRAM cap (default: the card's guard line less the display reserve; the
  check is the engine's own VRAM projection `q35_file._projection_terms` — weights at the chosen formats plus the KV
  cache, GDN states and split-attention records at `--max-ctx`), and a quality-loss cap (the sum of the sensitivity
  losses of the chosen formats).

## The model and the solver

Predicted time per token = Σ over stages (fixed part + bytes read × per-byte part), with the attention stage's KV bytes
at the read rate (`kv_pos_bytes × depth`) and the fusion deciding whether gate | up is one stage or two plus the SwiGLU.
The solver (`q35_plan.solve`) minimizes that exactly over the menu under both caps: branch and bound over the variables
(fusion groups and singleton tensors), each option's cost / bytes / quality precomputed, dominated options dropped,
bounded by suffix minima and a suffix dynamic program over the quality cap, with state dominance memoization. It is
checked against brute-force enumeration in the tests on small synthetic files, at several cap settings (both slack,
quality-binding, VRAM-binding).

## What it puts out

The chosen format per tensor, the total bytes on the card, the predicted ms per token with a per-stage breakdown —
always labelled a model — and the same breakdown for the file's current formats, so the two compare directly; which
constraint was binding (VRAM cap / quality cap / none, i.e. the menu alone); and the list of cost entries used that are
not plain measurements, each with its source. `--json` writes the same machine-readable. `--set TENSOR=FORMAT ...` checks
a requested assignment instead of solving and refuses (exit 1, `REFUSED: <reason>`) an unsupported `(op, format)` pair, a
split fusion group, or an assignment outside the caps; the solver likewise refuses an infeasible budget, naming the cap
and the menu's minimum.

## Measured time against size

Measured on the XT and the XTX for the four dense 27B files; the rows, commands and per-stage times are in
[benchmarks.md](benchmarks.md#dense-27b-decode-time-against-file-size). **On this engine, decode time and file size
are different targets.** Inside the decode kernel, a stage's time stops falling with its bytes below about 3.4 bits a
weight, and some 2-bit stages take longer than 3-bit ones with more bytes. A file is faster only where its bytes are on
the stages that convert bytes to time.

**Time-equivalent bits**: the bits a weight would need at the card's 774 GB/s to take the stage's measured time
(bits x 774 GB/s / the stage's achieved rate). XT, short context, inside the decode kernel, **high / auto** clocks;
a format the kernel converted at the full rate would show its own bits:

| format | bits | ffn_gate / ffn_up [17408 x 5120], each | ffn_down [5120 x 17408] | attn_qkv + attn_gate [10240 + 6144 x 5120], group | ssm_out / attn_output [5120 x 6144] | output [248320 x 5120] |
|---|---:|---:|---:|---:|---:|---:|
| Q8_0 | 8.50 |  | 9.69 / 9.67 | 12.04 / 11.90 | 10.66 / 10.62 | 8.90 / 9.01 |
| Q6_K | 6.56 |  | 7.36 / 7.38 | 7.81 / 7.77 | 8.82 / 8.70 | 6.76 / 6.81 |
| Q5_K | 5.50 |  |  |  | 7.39 / 7.28 |  |
| Q4_K | 4.50 | 4.89 / 4.87 | 5.13 / 5.02 | 5.69 / 5.53 | 6.39 / 6.24 | 4.63 / 4.65 |
| IQ4_XS | 4.25 | 4.73 / 4.65 | 5.13 / 5.11 | 5.56 / 5.39 | 6.40 / 6.22 |  |
| IQ3_S | 3.44 | 4.32 / 3.94 | 4.56 / 4.35 | 5.27 / 4.84 | 5.78 / 5.45 |  |
| IQ3_XXS | 3.06 | 4.27 / 3.76 | 4.48 / 4.03 | 5.28 / 4.63 | 5.76 / 5.60 |  |
| Q2_K | 2.62 |  | 4.08 / 3.77 |  |  |  |
| IQ2_S | 2.56 | 4.28 / 3.90 | 4.30 / 3.90 | 5.15 / 4.79 | 5.66 / 5.31 |  |
| IQ2_XS | 2.31 |  | 4.46 / 3.68 |  |  |  |

- Everything at or below IQ3_S lands at 3.7-4.6 time-equivalent bits on the large shapes, whatever its own bits:
  ffn_down at auto is IQ3_S 4.35, IQ3_XXS 4.03, Q2_K 3.77, IQ2_S 3.90, IQ2_XS 3.68. IQ3_S -> IQ2_S on ffn_down cuts 25 %
  of the bytes and, at auto, 10 % of the time (6 % at high).
- Mixed groups follow the same order: an IQ2_XS + IQ2_XXS gate / up pair (48.7 MB) takes 119.8 us at auto against
  106.0 us for an IQ3_XXS pair (68.2 MB).
- The scalar formats are nearer the bus at the large shapes (Q4_K 4.9-5.1 on the ffn, Q6_K 7.4, Q8_0 9.7) and the head
  is near it (4.63 for Q4_K's 4.50). The 5120 x 6144 shape costs every format 1.0-1.6 bits more than ffn_down.
- The same name can be two tiles: Q5_K's ffn_down is 90 us on ThinkingCap's module and 385 us on Swift's (its general
  rows tile), so a Q5_K entry needs the module (the table has no module key, so that shape stays on the flagged `'*'`
  entry).
- At auto the codebook formats gain the most (ffn_down: IQ2_XS 0.83x the high time, Q2_K 0.92x, IQ3_S 0.95x) and the
  scalar ones the least (Q4_K 0.98x, Q6_K 1.00x).

**Predicted against measured**, the plain decode step (`benchmarks/q35_verify.py`, ms per token), the shipped table
against this one:

| file | card | depth | measured, high / auto | shipped table (error vs high / auto) | this table (error vs high / auto) |
|---|---|---:|---:|---:|---:|
| Swift 1.5 Q4_K_M | XT | 512 | 26.36 / 26.16 | 24.22 (-8.1 / -7.4 %) | 25.59 (-2.9 / -2.2 %) |
| Swift 1.5 Q4_K_M | XT | 4,096 | 27.08 / 26.82 | 24.53 (-9.4 / -8.5 %) | 26.38 (-2.6 / -1.6 %) |
| ThinkingCap RCO 4.2 | XT | 512 | 23.49 / 23.03 | 21.27 (-9.5 / -7.6 %) | 23.22 (-1.1 / +0.8 %) |
| ThinkingCap RCO 4.2 | XT | 4,096 | 24.30 / 23.83 | 21.58 (-11.2 / -9.4 %) | 24.02 (-1.2 / +0.8 %) |
| ThinkingCap RCO 4.2 | XT | 32,768 | 30.70 / 29.92 | 24.00 (-21.8 / -19.8 %) | 30.39 (-1.0 / +1.6 %) |
| Qwen3.8-27B GSQ-RCO | XT | 512 | 21.41 / 20.68 | 18.22 (-14.9 / -11.9 %) | 20.73 (-3.2 / +0.2 %) |
| Qwen3.8-27B GSQ-RCO | XT | 4,096 | 22.19 / 21.41 | 18.52 (-16.5 / -13.5 %) | 21.53 (-3.0 / +0.6 %) |
| Qwen3.8-27B GSQ-RCO | XT | 32,768 | 28.61 / 27.37 | 20.95 (-26.8 / -23.5 %) | 27.90 (-2.5 / +1.9 %) |
| Swift 1.5 GSQ-RCO | XT | 512 | 21.41 / 20.63 | 18.22 (-14.9 / -11.7 %) | 20.73 (-3.2 / +0.5 %) |
| Swift 1.5 GSQ-RCO | XT | 4,096 | 22.20 / 21.39 | 18.52 (-16.6 / -13.4 %) | 21.53 (-3.0 / +0.7 %) |
| Swift 1.5 GSQ-RCO | XT | 32,768 | 28.53 / 27.52 | 20.95 (-26.6 / -23.9 %) | 27.90 (-2.2 / +1.4 %) |
| Swift 1.5 Q4_K_M | XTX | 512 | 22.30 / 22.35 | 22.14 (-0.7 / -0.9 %) | (no XTX entries) |
| Swift 1.5 Q4_K_M | XTX | 4,096 | 22.97 / 23.02 | 22.41 (-2.4 / -2.6 %) | (no XTX entries) |
| Swift 1.5 Q4_K_M | XTX | 32,768 | 28.82 / 28.78 | 24.60 (-14.6 / -14.5 %) | (no XTX entries) |

The shipped table's error came from three stage classes, each measured:

1. **Attention at depth.** Its entry took the KV bytes at the card's read rate: +2.74 ms from 512 to 32,768 positions,
   against a measured +7.2 ms. The KV is read at 295 GB/s (high) over the split reads and their combine (benchmarks.md,
   "Attention at depth"), and the combine grows 19x with the depth.
2. **The time with no weight read in flight**: 0.69 ms predicted, 1.93-2.01 ms measured. The 129 add + norm stages
   run on one workgroup at 7.3 us each (0.94 ms; the table had the 30B's 3.6 us), and the other glue ops left 0.74-0.82
   ms a token, which the table counted as 0.
3. **The matvecs**: 1.3 ms (Swift, ThinkingCap) to 2.2 ms (the GSQ-RCO files) above the bus-rate prediction, the
   gradient by bits above.

What the table here changed, and only where a measurement of that entry exists: the XT matvec entries per format and
shape (replacing the census's five IQ2_S totals), the XT add + norm and embedding entries, and the XT attention entry
(the line through 512 and 32,768 positions). Its rows above are not an independent check: the entries come from traces
of the same four files, so the 512 rows agree in part by construction. Its independent checks are the 4,096 rows (the
attention line between its two depths) and the depth rows of the files whose attention was not traced (the GSQ-RCO and
Swift files: the attention entry comes from ThinkingCap's verify tape at T = 1). The remaining error at high is the glue
class, still unmeasured and counted as 0 (0.74-0.82 ms short), less the verify tape's slower attention at 512 (0.37 ms
against the plain tape's 0.24), plus the stages still on the flagged `'*'` entry (Swift's Q5_K ffn_down on its rows
tile, 0.6 ms short; the GSQ-RCO files' mixed-format groups). At auto the errors are -2.2 to +1.9 %: the table is
the high clock's, and the engine serves at auto. The XTX has no entries of its own yet; its rows are the shipped table's.

## What is unmeasured, and what GPU work fills it

- The glue stages' fixed costs (GDN conv / l2 / step / save / norm, qk_nr, kv_append, gate_mul, SwiGLU): the traces
  have each op's span, but many run beside a matvec, so a span is not its cost on the critical path. Filling them
  needs each op's share of the critical path from the `--dump` segment times, per op.
- The XTX's entries (add + norm, attention, embedding, matvecs; the table carries the 30B's figures there, flagged
  `other-model`): the same traces on the XTX.
- The auto clock: the entries are the high clock's; the engine serves at auto, where the codebook formats run up to
  17 % faster (benchmarks.md has both).
- Attention on the plain decode tape at depth (the entry is the verify tape's at T = 1: 0.37 ms a token at 512
  positions, against 0.24 ms on the plain tape's trace at short context): a trace build of the plain decode modules
  run at depth.
- A module key on entries, for a format whose tile differs by module (Q5_K), and the formats the files do not hold
  (IQ2_XXS and IQ1_M alone in a group, Q5_K / Q8_0 at the ffn shapes in a ThinkingCap-like layout).
- The XT's decode-read rate (the table uses its streaming 774 GB/s, flagged): the decode-floor read on the XT.
- The sensitivity table: the per-tensor G-model arm above — the only GPU measurement the quality side needs.

Nothing here runs on a GPU: the tool and its tests are CPU-only; the measurements above are what a GPU window would
produce to replace the flagged entries.
