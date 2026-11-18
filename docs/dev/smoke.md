# K3 smoke benchmark

One command checks that the K3 TP engine is still correct and shows how fast it is. It takes about 95 s,
model loads included, and runs through guard.sh like every GPU command.

```
# in a checkout, with CODER, DFLASH and ARTIFACTS set (AGENTS.md, "Models and data"); REAP and REAP_DUMPS for --model reap
tools/guard.sh $ARTIFACTS/smoke/logs/smoke.log 600 \
  python -I tools/k3_smoke.py --model coder     # or --model reap
```

- The exit code is 0 when every check passes and 1 on any correctness failure. The tool refuses to start
  (exit 3) unless guard.sh is one of its parent processes.
- guard.sh's own exit codes (99, 98) take precedence: [GPU safety](gpu-safety.md#every-gpu-process-runs-under-the-guard).
- The table and a `RESULT PASS|FAIL` line go to the log. The full results (tokens, per-position KLD,
  both reps, stage times, VRAM peaks) go to `$ARTIFACTS/smoke/<time>.json` (or `--out DIR`).
- To compare two runs (CPU only, no guard needed), with a tolerance of ±2 % by default:
  `python -I tools/k3_smoke.py --compare A.json B.json`.

## What it runs

Each GPU stage runs in its own child process, one after the other, so two engines never share VRAM.

1. **Floor (CPU).** The D-281 floor is measured on the same positions the engine is checked on: llama.cpp batch vs
   llama.cpp one-token decode, positions 0..63 of each e2e text, read from the `_batch` dumps.
2. **tp1: `FlashNextTP(ntok=1)`.**
   - **Teacher-forced check.** Positions 0..63 of prose, code and shell are compared with llama.cpp's decode
     dump: mean KLD (float64, full vocabulary) and top-1. The bar is the D-281 rule against the floor above:
     mean KLD at or below the floor, and top-1 no more than 1 point under it. `check_replicas()` runs after
     every position.
   - **Plain decode in the real generation loop.** This is `step_token()` per token: `step()` plus argmax with
     only the token read back. It gives the same tokens as `step().argmax()` in m2/s7 `spec_k3.py plain`. The engine prefills the 200-token prompt, then generates 128 greedy tokens. There is
     a 16-token warm-up, then 2 reps; rep 2 is reported and rep 1 must give the same tokens.
3. **spec:dflash.** `FlashNextTP(ntok=4)` runs with DFlash 2 inside K3 (`caminus.decode.dflash.smoke_factory`: the
   drafter's Q8_0 layers on the XT, the inject on the XTX from the target's features there, the target's head shards on
   both cards plus an XTX copy of part of the XT's). `caminus.decode.spec.generate` hands the generation to
   `DFlashK3Drafter.generate`, the device-accept loop (S3): each pass is the target's verify and the drafter's 'step'
   launch (accept + inject + next drafts) queued behind it, then one wait. Its "draft" ms is the step launch's device
   time on the XT. `--spec dflash_hostacc` adds the same drafter in the generic loop (host accept; one summary line with
   both loops' ms/pass: the same drafts pass for pass, so the difference is the host time saved). `--dflash-impl torch`
   runs the torch drafter (`caminus.decode.dflash_torch`) in this column instead, and `--spec dflash_torch` adds it as its own stage.
   After a warm-up generation, it runs the same prompts and 128 tokens, 2 reps. It reports tok/s (generation tokens
   after the prompt pass divided by generation wall time), tokens per pass and acceptance. The gate: the
   output is identical to the tp1 plain tokens.
4. **spec:mtp.** `FlashNextTP(ntok=4)` with the model's MTP block in K3 (`caminus.decode.mtp.smoke_factory`: the sidecar GGUF's
   blk.48 + nextn on both cards, the target's head shards and embedding rows), 3 drafts per pass, the same prompts, reps
   and gate as spec:dflash. It runs by default only when the sidecar is configured (`CAMINUS_MTP`, or `mtp` in config.toml):
   the sidecar is not distributed, so without it `--spec` defaults to `dflash` and the smoke prints a note. Other methods plug
   into `DRAFTERS` in the same way (`factory(cfg, T) -> (engine, drafter, info)`, the drafter with the `caminus.decode.spec.generate`
   interface).

The llama.cpp numbers in the table come from `tools/smoke_ref.json`. They come from M0 (plain, DFlash) and the
MTP reference run (MTP1, MTP3), and each entry names its source log and N. llama.cpp measured 256 tokens per
prompt and K3 measures 128, so treat the comparison as indicative rather than exact.

## Results

The smoke's numbers by date, with the llama.cpp columns from `tools/smoke_ref.json`: [benchmarks](benchmarks.md#decode)
(decode rows) and [benchmarks](benchmarks.md#accuracy) (the teacher-forced check).
