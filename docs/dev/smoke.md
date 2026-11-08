# K3 smoke benchmark

One command checks that the K3 TP engine is still correct and shows how fast it is. It takes about 95 s,
model loads included, and runs through guard.sh like every GPU command.

```
# in a checkout, with CODER, DFLASH and ARTIFACTS set (CLAUDE.md, "Models and data"); REAP and REAP_DUMPS for --model reap
tools/guard.sh $ARTIFACTS/smoke/logs/smoke.log 600 \
  python -I tools/k3_smoke.py --model coder     # or --model reap
```

- The exit code is 0 when every check passes and 1 on any correctness failure. The tool refuses to start
  (exit 3) unless guard.sh is one of its parent processes.
- guard.sh's own exit codes take precedence. 99 means a GPU fault: stop all GPU work and re-run nothing.
  98 means guard refused to start the run.
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

## DFlash 2 in K3 with the device accept (2026-11-02, 7e547c7)

`tools/k3_smoke.py --model coder --spec dflash,dflash_hostacc,dflash_torch,mtp` (20261102-133254.json, RESULT PASS, all
== plain, bails 0). 128 tokens after a 200-token prompt, rep 2:

| | prose | code | shell |
|---|---|---|---|
| K3 DFlash, device accept: tok/s (acceptance) | 167.3 (77.8 %) | 185.9 (90.5 %) | 112.2 (40.9 %) |
| K3 DFlash, host accept (same drafts) | 166.4 | 184.6 | 111.5 |
| ms/pass device / host accept | 19.92 / 20.04 | 19.98 / 20.12 | 19.85 / 19.99 |
| step launch device ms (accept + inject + draft) | 1.85 | 1.83 | 1.82 |
| torch DFlash tok/s (acceptance) | 121.8 (80.7 %) | 131.9 (90.5 %) | 80.9 (42.3 %) |
| MTP in K3 tok/s | 163.6 | 179.5 | 124.5 |
| plain | 92.7 | 92.5 | 91.6 |
| llama.cpp + DFlash (N = 256) | 92.4 | 89.1 | 65.1 |

- The device accept saves 0.12-0.14 ms per pass, not the 0.3 ms the spec projected (the logits copy it counted on was
  already gone: `verify_tokens` reads back only each card's max / argmax).
- Long context (prompt 1700, generation to 2051): step launch median 1.90 ms per 100-position bucket from 1700 to 2051,
  mean 1.95, max 2.73 (88 passes); the old loop's draft call 2.07 ms host wall (was 2.58).
- Acceptance vs torch on the same trajectory (each pass's drafts of both drafters scored against plain greedy): 128
  tokens prose +0.9, code 0, shell -1.2 points (one pass of 57), total -0.26; 512 tokens +0.25 / 0 / -0.17, total 0.00.
  The free-running gap above (prose 77.8 vs 80.7 %) comes from the trajectories splitting after one differing pass.

## DFlash 2 in K3 (2026-11-02, 79cd308)

`--spec dflash,dflash_torch,mtp`, smoke 20261102-115508.json (RESULT PASS, guard rc 0, 167 s):

| | prose | code | shell |
|---|---|---|---|
| K3 DFlash tok/s (acceptance) | 165.5 (77.8 %) | 184.4 (90.5 %) | 111.2 (40.9 %) |
| torch DFlash tok/s (acceptance), same run | 121.7 (80.7 %) | 131.9 (90.5 %) | 80.9 (42.3 %) |
| K3 MTP3 tok/s (acceptance), same run | 163.7 (86.1 %) | 179.7 (98.0 %) | 124.9 (57.4 %) |
| Plain tok/s, same run | 92.9 | 92.4 | 91.8 |
| llama.cpp + DFlash (N = 256) | 92.4 | 89.1 | 65.1 |

- Every speculative output == plain on all three texts, rep 1 == rep 2, replicas equal, 0 bails.
- Draft 2.00 / 1.99 / 1.99 ms per pass in the loop (host wall to the drafts), against 8.76 ms for the torch drafter. One
  fused launch on the XT (the pending inject + the draft, 99 segments) and a small one on the XTX (its head shard + top-k).
- Acceptance at 128 tokens differs from the torch drafter by -2.9 / 0 / -1.4 points (about 120 drafts per text). Over
  512 tokens on the same target (acc_cmp.py, one process): 93.6 vs 94.5 % / 93.8 = 93.8 % / 52.0 = 52.0 %, total 75.91 vs
  76.14 %; the K3 drafter's numerics are llama.cpp's (Q8_0 x q8_1 dots, f16 KV), the torch drafter's fp16 weights / fp32 KV.
- VRAM peak (whole smoke) XTX 19.9 GB, XT 15.6 GB (the torch stage; the K3 drafter's stage peaks at XT 13.5 GB).

## MTP in K3 (2026-11-02, bb550db)

| | prose | code | shell |
|---|---|---|---|
| Coder MTP3 tok/s (acceptance) | 163.8 (86.1 %) | 179.1 (98.0 %) | 124.6 (57.4 %) |
| Coder DFlash tok/s, same run | 121.9 | 132.0 | 80.9 |
| Coder plain tok/s, same run | 93.0 | 92.5 | 91.9 |
| llama.cpp MTP3 / MTP1 (N = 256) | 85.9 / 65.1 | 79.8 / 65.5 | 67.6 / 60.2 |

- MTP spec output == plain on all three texts, rep 1 == rep 2, replicas equal, 0 bails (smoke 20261102-092831.json).
- MTP pass: 21.9 ms = draft 3.7 + verify 18.1. The draft is one catch-up launch (the verified rows, 4-row tape, head on the
  last row only: 1.57 ms) + 2 one-row chain launches (1.03 ms each: device 0.86 = head ~0.38, eh_proj ~0.1, layer ~0.4).
- MTP1 / MTP2 (separate runs, T = 2 / 3): 118.9 / 121.6 / 110.5 and 142.3 / 150.7 / 114.4 tok/s; MTP1 acceptance
  95.4 / 100 / 81.4 % (llama.cpp MTP1 94.7 / 97.7 / 81.4 %).

## Current numbers (2026-11-02, after the generation-loop host work)

The host path of a generated token was cut down: inputs staged in one pinned block, per-card argmax with one
readback per card, event waits, cached launch arguments and embedding rows, PLE rows prefetched for guessed next
tokens, and the DFlash inject run as a captured graph. Same tokens and logits as before (bitwise).

| | prose | code | shell |
|---|---|---|---|
| Coder plain tok/s | 93.2 / 93.2 | 92.9 / 92.8 | 92.2 / 92.3 |
| Coder DFlash tok/s (tokens/pass) | 121.0 / 121.0 (3.42) | 131.0 (3.71) | 80.4 (2.27) |
| REAP-308 plain tok/s | 94.4 | 94.4 | 94.4 |
| REAP-308 DFlash tok/s (tokens/pass) | 70.4 (1.84) | 142.3 (3.74) | 105.2 (2.74) |

- Coder DFlash pass: 28.3 ms = draft 8.75 + verify 18.5 + inject 0.97 (was 29.9 = 8.85 + 19.7 + 1.29).
- Teacher-forced KLD / top-1: unchanged (Coder 0.0043 / 96.4%, REAP 0.0060 / 96.4%). Two Coder runs agreed on
  every metric within 2 %.
- **Context length matters.** The per-token time grows by about 0.6 ms per 100 positions of context: 9.5 ms at
  positions 5-54, 10.7 ms at 205-254 and 12.4 ms at 505-554 (Coder). This is device time. The most likely cause
  is the attention decode op, which runs one task per head over all positions (12 tasks per card); this has not
  been confirmed per segment yet. So fn_bench's 6-token prompt (about 9.8 ms/token)
  is not comparable with this tool's 200-token prompts (about 10.8 ms/token). It also explains why the 256-token
  m2/s7 runs were slower (11.69 ms) than the 128-token runs here.

## Baseline (2026-11-02, HEAD b3aebd4: the engine code of 201b00b)

| | prose | code | shell |
|---|---|---|---|
| Coder plain tok/s | 88.8 / 88.9 | 88.4 / 88.9 | 88.4 / 88.9 |
| Coder DFlash tok/s (tokens/pass) | 114.3 / 114.6 (3.42) | 124.1 (3.71) | 76.1 (2.27) |
| REAP-308 plain tok/s | 90.7 | 90.6 | 90.7 |
| REAP-308 DFlash tok/s (tokens/pass) | 67.2 (1.84) | 135.6 (3.74) | 100.3 (2.74) |

Where a cell shows two numbers, they are runs 1 and 2. The two Coder runs agreed within 0.6 % on every metric.

- **Coder teacher-forced:** mean KLD 0.0043 against a floor of 0.0063, top-1 96.4% against a bar of 95.4%.
  Pass.
- **REAP teacher-forced:** mean KLD 0.0060 against 0.0090, top-1 96.4% against 93.8%. Pass.
- **Wall time:** 92-95 s per run.
- **Plain vs the m2/s7 runs:** plain at 128 tokens runs 11.3 ms/token, against 11.69 ms in the 256-token
  m2/s7 runs. The cause is the shorter context; this was measured later (see Current numbers). The tokens equal
  the first 128 of `m2/s7/plain_coder.json`.
