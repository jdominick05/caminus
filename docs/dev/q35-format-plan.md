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
  - IQ2_S per-shape per-byte entries (XT, 0.4.0 census) — docs/dev/benchmarks.md:517 (ffn_gate/up 63.2 / 61.5 µs on
    17408x5120, ffn_down 66.7 on 5120x17408, attn_qkv 42.5 on 10240x5120, attn_gate 29.9 on 6144x5120, ssm_out 30.7 on
    5120x6144), marked `total-only`: the fixed / per-byte split at those shapes is not measured;
  - norm 3.6 µs, attention 11.9 µs at 8 positions — the same trace, marked `other-model` (measured on Qwen3-30B-A3B's
    module, not the qwen35 line's);
  - glue (GDN conv/l2/step/save/norm, qk_nr, kv_append, gate_mul, SwiGLU) and the embedding segment: **unmeasured**
    (`fixed_us: null`, `measured: false`, counted as 0 and flagged in every output that uses them).
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

## What is unmeasured, and what GPU work fills it

- The glue stages' fixed costs (GDN conv / l2 / step / save / norm, qk_nr, kv_append, gate_mul, SwiGLU) and the
  embedding segment's fixed cost: **not published anywhere**. Fill with the trace module's per-segment times
  (`benchmarks/q35_verify_trace.py` / `benchmarks/q3moe_trace.py --engine dense35` style, on the real 27B file, one card,
  under `tools/guard.sh`) — the per-op times of one token, as docs/dev/qwen3-30b-a3b.md "Where the remaining time is"
  does for the 30B.
- The qwen35 line's add+norm and attention fixed costs (the table carries the 30B's figures, flagged `other-model`): the
  same trace on a qwen35 file, per stage kind.
- Per-format, per-shape-class matvec times beyond IQ2_S's five shapes: `benchmarks/own_mv_census.py` on the file
  (`--formats ...`), own arm, which already prints us per matvec per (role, format, rows, K); each row becomes a table
  entry with its doc source.
- The XT's decode-read rate (the table uses its streaming 774 GB/s, flagged): the decode-floor read on the XT.
- The sensitivity table: the per-tensor G-model arm above — the only GPU measurement the quality side needs.

Nothing here runs on a GPU: the tool and its tests are CPU-only; the measurements above are what a GPU window would
produce to replace the flagged entries.
