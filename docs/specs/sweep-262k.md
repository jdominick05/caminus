# The long-context sweep to 262,144 (sweep-262k, 11-05)

The step that takes the engine from "gated up to 32K" to the model's cap: QSA spec S7 plus the items M6 deferred to it, with the
accuracy method decided on 11-05. Tools: `tools/eval/sweep_262k.py`,
`tools/gates/g4_floor.py pool` / `depth-jobs --ctx --runs-1tok --runs-pd`, `tools/eval/lc_probes.py build --min-chars --seq`.
Evidence: `$ARTIFACTS/sweep-262k/` (logs/, g4/, k3/, lc/). No engine or kernel code changed in this step: the decode code
objects are byte-identical to the baseline hashed before it (`sweep-262k/hsaco_baseline.sha256`).

## Results

**Item 4, the one-token D-281 at ~262K (the single llama.cpp one-token run to 261,632 and K3's one-token run): skipped by user
decision 11-05.** The llama.cpp job file exists (`sweep-262k/g4/jobs_c_1tok262k_ub512.txt`) and was never run. K3's one-token
rows at 261,632 exist only as a by-product of item 6's soak; nothing compares them with a llama.cpp one-token run.

### 1. The depth note and the pooled 32K floor (CPU)

spec-qsa.md section 9 (commit dc65031). Pooled 32K floor from the runs already on disk, 4 one-token x 3 prefill-then-decode runs =
12 pairings: **mean 0.00140, 95 % bootstrap 0.00024 .. 0.00324, top-1 99.80 %**; per-pairing means 0.00113 .. 0.00167. It
reproduces the six earlier pairing means to 1e-9. File: `$ARTIFACTS/impl-prefill-s7/g4/g4_depth_pooled_N32768.json` (new;
`g4_depth.json` unchanged).

### 2. 64K (binding): PASS

Window: positions 65,536 .. 65,791 of seq.ids. llama.cpp at ub 2048, n_ctx 65,792: 3 one-token runs (fills 1,610 / 1,606 /
1,605 s, decode 29.8 ms/token) and 2 prefill-then-decode runs (fills 28.9 / 28.2 s). Every run differs from every other in all
256 rows' FNV, with 0 argmax differences (not run-to-run reproducible, as at 32K).

| llama.cpp pairing (one-token : pd) | mean KLD | 95 % bootstrap | top-1 | max (at) |
|---|---|---|---|---|
| r1 : pd1 | 0.00036 | 0.00008 .. 0.00077 | 99.61 % | 0.0344 (65,630) |
| r1 : pd2 | 0.00071 | 0.00013 .. 0.00157 | 99.61 % | 0.0801 (65,645) |
| r2 : pd1 | 0.00064 | 0.00018 .. 0.00125 | 99.61 % | 0.0477 (65,630) |
| r2 : pd2 | 0.00047 | 0.00009 .. 0.00111 | 99.61 % | 0.0662 (65,630) |
| r3 : pd1 | 0.00035 | 0.00010 .. 0.00078 | 99.61 % | 0.0447 (65,736) |
| r3 : pd2 | 0.00029 | 0.00012 .. 0.00052 | 99.61 % | 0.0187 (65,630) |
| **pooled (6)** | **0.00047** | **0.00015 .. 0.00090** | **99.61 %** | 0.0381 (65,630) |

The per-pairing spread is 0.00029 .. 0.00071 (sd 0.00017): a factor of 2.4 between pairings of the same window. File:
`sweep-262k/g4/g4_pooled_N65536.json`.

| Gate (64K) | mean KLD | top-1 | bar | verdict |
|---|---|---|---|---|
| **G4a** (K3 prefill-then-decode vs K3 one-token) | **0.00050** (bootstrap 0.00012 .. 0.00119) | 99.61 % (1 flip at 65,630, gap 0.14) | <= 0.00090 (pooled upper bound), >= 98.61 % | PASS |
| **G4b** vs llama.cpp pd r1 (ub 2048) | **0.00092** | 99.61 % | <= 0.0076, >= 95.9 % | PASS |
| G4b vs llama.cpp pd r2 (reported) | 0.00082 | 99.61 % | | |
| D-281 long, K3 one-token vs llama.cpp one-token r1 / r2 / r3 (reported) | 0.00068 / 0.00026 / 0.00020 | 99.61 % each | | |
| K3 prefill-then-decode vs llama.cpp one-token r1 / r2 / r3 (reported) | 0.00055 / 0.00024 / 0.00031 | 100 % each | | |

G4a's mean (0.00050) is within the pooled floor's spread, just above its pooled mean (0.00047). Its only flip, at 65,630, is the
floor's own worst position, where every llama.cpp pairing has its max or close to it.

### 3. 128K and ~262K (binding): G4b PASS at every window

llama.cpp's memory past 64K (ub 2048, the M0 split): XT peak 18,774 MiB at n_ctx 65,792 and 19,242 at 82,176, against the guard's
19,850 line (+0.0286 MiB per cell between the two; the next 16K step, n_ctx 98,560, would peak at ~19,710, within 140 MiB of
the line). So ub 2048 binds at **81,920**, the deepest window run there with a safe margin. At 131,072 and 261,632, llama.cpp runs at **ub 512**
(n_ctx 261,888: XTX 20,281 / XT 18,849 MiB peak; compute buffers 2,659 / 2,728 MiB, KV 1,630 + indexer 408 MiB per card). The
ub control at 64K: llama.cpp's pd at ub 512 vs at ub 2048 r1 gives mean KLD 0.00056 / top-1 99.61 %, beside its own ub-2048
run-to-run 0.00031 / 100 %. The ub moves llama.cpp's prefill by about the size of its run-to-run noise, far below the bar.

| Window | G4b vs llama.cpp pd | mean KLD | top-1 | max (at) | bar | verdict |
|---|---|---|---|---|---|---|
| 81,920 | r1, ub 2048 (r2 reported: 0.00158; llama.cpp r2 vs r1 0.00033) | **0.00127** | 100 % | 0.1296 (82,000) | 0.0076 / 95.9 % | PASS |
| 131,072 | r1, ub 512 | **0.00321** | 99.22 % (flips 131,145 gap 0.67, 131,271 gap 0.01) | 0.1004 (131,279) | 0.0076 / 95.9 % | PASS |
| 261,632 | r1, ub 512 | **0.00020** | 100 % | 0.0134 (261,688) | 0.0076 / 95.9 % | PASS |

Reported beside them (no llama.cpp floor at these depths, by the ruling):

| Window | K3 pd vs K3 one-token | llama.cpp pd vs K3 one-token | K3 one-token mean entropy (nats) |
|---|---|---|---|
| 65,536 | 0.00050 / 99.61 % | (G4a above) | 0.023 |
| 81,920 | 0.00083 / 100 % | n/a (ub 2048: 0.00127 is G4b) | 0.015 |
| 131,072 | 0.00535 / 99.22 % | 0.00492 / 99.61 % | 0.210 |
| 261,632 | 0.00020 / 100 % | 0.00023 / 100 % | 0.023 |

The 131,072 window's larger distance is intrinsic to the window, not to our prefill. llama.cpp's own prefill-then-decode is as
far from our one-token decode (0.00492) as our prefill-then-decode is (0.00535). That window's text is ten times less
predictable (mean next-token entropy 0.210 vs 0.015 .. 0.023), and 56 positions over 0.001 are spread across it.

**Needle and rare-token copy recall (binding at 128K and ~262K): no probe that llama.cpp passes fails on ours.** 18 probes, each
answered after the batched prefill of the whole context + question (65,546 .. 261,641 positions); llama.cpp's references at ub 512.

| Depth | needle 10 / 50 / 90 % | rare 10 / 50 / 90 % | ours | llama.cpp | token-identical 24-token continuations |
|---|---|---|---|---|---|
| 65,536 | Y / N / Y (llama.cpp Y / N / Y) | Y / Y / Y (Y / Y / Y) | 5 / 6 | 5 / 6 | 3 of 6 |
| 131,072 | Y / Y / Y (Y / Y / Y) | Y / **Y** / Y (Y / **N** / Y) | 6 / 6 | 5 / 6 | 3 of 6 |
| 261,632 | Y / Y / Y (Y / Y / Y) | Y / Y / Y (Y / Y / Y) | 6 / 6 | 6 / 6 | 2 of 6 |

- The one miss we share (needle 50 % at 64K) is token-identical: both answer with the 10 % needle's code (`2086-7179`).
- llama.cpp's extra miss (rare 50 % at 128K) answers with the 10 % rare string (`d7rxzpHtQui3pG`) from its first token; ours copies
  the right one.
- Every other divergence comes after the answer: at token 11 .. 21 of 24, in what follows the closing quote.
- At 8K / 32K (S7, unchanged): 12 / 12 as llama.cpp.
- Files: `sweep-262k/k3/probes.json` (ours), `sweep-262k/lc/ref/*.tokens` / `.tsv` (llama.cpp).
- llama.cpp's probe fills at ub 512: 1,890 tok/s at 64K, 1,433 at 128K, 947 .. 959 at 261,632 (one-token decodes ~32 / ~38 / ~60 ms).
- Ours, the probe prompt through the batched prefill + 24 greedy tokens: 24.7 / 51.2 / 110.5 s.

### 5. Curves (timing; nothing else running; compaction_proactiveness 0, MemAvailable 12.8 - 13.1 GB)

Prompt = seq[:D] + a 200-token smoke text (prose for the TTFT). TTFT medians of 3 up to 32K, one run deeper. Decode: 128
DFlash tokens (T = 4, device accept), plain 128 tokens on the T = 1 engine. Every DFlash generation equals plain greedy on the
T = 1 engine after the same prefill (18 / 18 depth x text, 128 tokens each).

| Depth (+200) | TTFT from empty | prefill tok/s | TTFT with the DFlash inject | plain ms/token (prose / code / shell) | DFlash ms/token (prose / code / shell) | DFlash tokens per pass |
|---|---|---|---|---|---|---|
| 2,048 | 860 ms | 2,614 | 951 ms | 10.42 / 10.48 / 10.56 | 5.31 / 5.55 / 8.81 | 3.88 / 3.69 / 2.31 |
| 8,192 | 3,025 ms | 2,775 | 3,111 ms | 10.73 / 10.69 / 10.57 | 8.73 / 7.87 / 8.19 | 2.35 / 2.61 / 2.50 |
| 32,768 | 11.90 s | 2,771 | 11.98 s | 10.81 / 10.93 / 10.94 | 6.67 / 5.73 / 8.79 | 3.26 / 3.79 / 2.46 |
| 65,536 | 24.21 s | 2,715 | 24.29 s | 11.68 / 11.56 / 11.58 | 9.79 / 8.10 / 9.32 | 2.37 / 2.87 / 2.49 |
| 131,072 | 50.96 s | 2,576 | 50.59 s | 12.86 / 12.74 / 12.70 | 11.45 / 7.38 / 8.83 | 2.30 / 3.58 / 3.00 |
| 261,632 | 109.4 s | 2,393 | 109.8 s | 15.26 / 15.40 / 15.34 | **18.25** / 12.34 / 12.10 | 1.77 / 2.63 / 2.69 |

- **Plain decode** grows from 10.4 to 15.3 ms/token between 2K and 261,632 (+47 %). spec-qsa.md section 5's projection was ~11.3
  ms at 262K (INFERRED). Measured, the T = 4 pass grows from 17.7 ms (2,048) to 23.8 ms (196,608) and 25.7 .. 35.9 ms at 261,632.
  llama.cpp's one-token decode in the same jobs: 29.8 ms at 64K, ~38 at 128K, ~60 at 261,632 (it reads the whole KV).
- **DFlash at ~262K loses to plain on prose** (18.25 vs 15.26 ms/token: 1.77 tokens per pass, 0.78 drafts accepted, while the T = 4
  pass costs ~26+ ms there); code and shell still win (12.3 / 12.1 vs 15.4 / 15.3). Its acceptance depends on the text after the
  context as much as on depth (prose 3.88 / 2.35 / 3.26 tokens per pass at 2K / 8K / 32K): these are curve points of one
  generation per cell, not speed claims. A finding for M7's policy: a DFlash pass costs ms/token x tokens per pass = ~26 ms at 131,072 and ~32 ms at 261,632, so there
  DFlash beats plain decode only above ~2.1 tokens per pass.
- **Prefill** stays within 2,393 .. 2,775 tok/s from 8K to the cap (llama.cpp's own probe fills at ub 512: 1,890 / 1,433 / 950
  tok/s at 64K / 128K / 262K; its ub-2048 G4 fills: 2,264 tok/s at 64K, 2,106 at 80K).
- **8K appends at depth** (with the DFlash inject, one run each): 131,072 .. 139,263 in 3,626 ms (2,259 tok/s); 250,000 .. 258,191 in
  4,163 ms (1,968 tok/s). (S7: 2,599 tok/s at 24,576.)
- **The longest launch at depth** (`launch`: one 2048-token chunk at 259,584 .. 261,631 with both cards synced around each of its
  10,698 prefill launches): 2.30 ms (pf_qsa_score), p99 1.89, median 0.078 ms; K3 T = 4 passes at 261,632: 25.7 .. 35.9 ms. Every
  launch is far under 50 ms. In the soak the longest pass was 48.7 ms net of evictions (322.6 ms raw, one KFD eviction).

### 6. The soak: 0 unexplained bails

One policed run of T = 4 passes from 0 to 261,888 over seq.ids (65,472 passes, 1,468 s, 178 tok/s), the DFlash inject after every
pass, the pass limit 150 ms net of evictions:
- **POLICE: eviction-aware; 67 evicted passes, 2,365 ms evicted in all (largest node per pass); longest raw pass 322.6 ms at
  220,156, 48.7 ms net; over the limit raw 1, forgiven 1, violations 0; bails 1, explained by an eviction 1, unexplained 0**
  (the bail at 220,156: a 297.8 ms attempt with 273.9 ms of a 275 ms KFD eviction inside it; the bitwise retry ran).
  compaction_proactiveness 0 at the start and the end; replicas equal at the end.
- Pass times around the boundaries (raw ms, the passes ending around each position): 2,048 17.7; **2,052 39.1 once** (the first
  QSA-tape pass with the one-time pool backfill), then 18.3 (4,096), 18.4 (8,192), 18.7 (16,384), 19.0 (32,768), 19.9 (65,536),
  21.7 (131,072), 23.8 (196,608).
- **There are no capacity buckets in the engine** (see Method). The soak crosses every boundary that exists: 2051 (the tape
  switch and the backfill), nv 512, 63 DFlash ring wraps (4,096 rows), and QSA_SCORE's fixed 64 tasks over 65,472 pools (32 chunk
  loops per task at the end).
- From the soak's end state (261,888): 96 tokens with DFlash (device accept, to 261,983) == 96 plain tokens, bitwise (2.03 drafts
  accepted per pass).

### 7. VRAM at the real 262K fill

| Configuration (max_ctx 262,144, DFlash, prefill chunk 2048) | XTX measured | XTX idle + projection | XT measured | XT idle + projection |
|---|---|---|---|---|
| Engine + drafter + prefill, after load (k3 rig) | 22,189 | 22,698 | 17,381 | 17,451 |
| Same, guard peak over the soak + G4 + probes (a ~262K fill each) | 22,189 | 22,698 | 17,483 (*) | 17,451 |
| + the full cache tier (15 slots, 1,377 / 1,538 MiB), after a 261,832-token request with 33 checkpoints (17 spilled) + 16 DFlash tokens | **23,032** (+512 = 23,544 <= 23,800) | 24,074 | **18,919** (peak 18,938) | 18,991 |
| T = 1, no drafter (plain curve), after load | 21,925 | 22,477 | 16,345 | 16,412 |

- The XTX reserve was never breached: every sweep run's guard peak + 512 <= 23,800. Peaks: cache 23,032; K3 runs <= 22,189;
  llama.cpp <= 20,281.
- (*) The XT's 17,483 peak, 32 MiB over its idle + projection, came at the soak's very end in both the full run and the --small
  run. That is when the tool takes a `longctx.snapshot` (device copies of every recurrent state and the DFlash ring) for its
  DFlash-vs-plain check at the cap. Snapshots are counted by no projection (the prompt cache's checkpoints, which the serving path
  uses, are). It is test-side and released at once: the XT was back at 17,384 after the stage. The serving configuration with
  the tier keeps a +72 MiB XT margin at the 262K fill.

### 8. The cache at depth (the full tier; disk tier in the job scratch dir, purged after)

- **G3 (a) at a 200K checkpoint: PASS.** The prompt-end checkpoint at 199,999 (27 checkpoints on the way: root, 24 x 8K, the
  user mark, the prompt end) restored from VRAM after a probe (16 ms) == restored from disk after a reset (1,224 ms from a 7.19 GiB
  entry parked in 3.2 s) == the state when taken, bitwise (251 items: every recurrent state, KV / IDXRAW rows, pools, the DFlash
  window + tags). The entry's checkpoint bytes == the VRAM slot's.
- **G3 (b) at depth: PASS.** A cold 200,000 + 48 decoded (DFlash) + 700 appended, after a probe, restored from VRAM (2.2 ms restore,
  20.3 ms total) and from disk (1,228 ms) == resumed without a restore: every state, the first logits bitwise, 32 tokens.
- **Park / reload, measured** (not extrapolated):

  | Conversation | entry | checkpoints | park | reload (rows + one checkpoint) | cold re-prefill | reload == parked | page cache |
  |---|---|---|---|---|---|---|---|
  | 100,000 | 3.83 GiB | 14 | 1.44 s (2.86 GB/s) | 0.62 s (1.76 GiB, 3.04 GB/s) | 39.3 s | bitwise | +1.5 MiB |
  | 261,632 | 9.29 GiB | 32 (15 VRAM + 17 spilled) | 4.32 s (2.31 GB/s) | 1.53 s (4.34 GiB, 3.04 GB/s) | 114.7 s | bitwise | **+73.3 MiB** |

  **One check FAILED: the 261,632 park's page-cache criterion** (my tool's, copied from test_prompt_cache park: Cached may grow
  < 64 MiB over the park + reload). The reload came from disk and is bitwise the parked state. Every data path of the disk tier
  (writes, reads, the spill files' file-to-file copies) is O_DIRECT, and +73 MiB is 0.5 % of the 13.6 GiB moved, so it is not
  the entry's data in the cache. Its source is not attributed (other host activity in the ~6 s window, the directory's
  metadata). Reported as measured, not re-run, the threshold not changed. The 100K park: +1.5 MiB.
- **DFlash after a restore at depth: PASS.** A 200,000 base + 24 decoded (the 'end' checkpoint), a probe at the reply's end,
  restore (VRAM, 200,023 reused) + 400 appended: spec == plain greedy bitwise, 96 tokens. Acceptance 2.31 drafts per pass restored
  vs 2.31 from a cold fill of the same prompt; text == cold.
- vram: the cache's own device work at 262K (the stage's 261,832-token cold request): 33 checkpoints in 158 ms in all, 17 spills in
  1,027 ms, 1 evicted.

### No regression: PASS

One chain, nothing else on the cards (logs `sweep-262k/logs/regress3.out`, `r3_*.log`; outputs `sweep-262k/regress3/`). A first
attempt at 12:46 overlapped a duplicate chain on the cards: its gates and smoke were refused by `capacity.check` (device 0
already held 18,996 MiB) before anything was allocated, so they are void. No GPU fault, no STOP file. It was re-run alone from 13:10.
- **Decode code objects byte-identical:** `sha256sum -c sweep-262k/hsaco_baseline.sha256`, every build/*.hsaco (58 files) unchanged.
- **`tools/gates/gates.sh regress3/gates trunk multi short`: SUMMARY 9 passed, 0 failed**. Refdump g == sp_g.pt; T = 4 ==
  one-token on trunk 24 / 200 positions and QSA dual / always 200 positions; D-281 short 0.0057 / 96.2 %, every result line ==
  refs/current/e2e_tp.log.
- **k3_smoke --model coder: RESULT PASS.** All 9 token lists (plain, DFlash and MTP, each on prose / code / shell) equal the latest
  reference run (smoke/20261105-051743.json), every token.
- **tests/prefill/test_prefill_s6.py g4a: RESULT PASS.** G4b short 0.00645 / 98.17 %, G4a routed 0.00640 / 97.39 %, G4a 2K 0.00065 /
  0.00021, G4b 2K 0.00071 / 0.00023 (CLAUDE.md's numbers).
- **tests/prefill/test_prefill_s6.py g2b g3 dflash long: RESULT PASS.** The long stage: 0.00038 / 99.61 %.
- **tests/cache/test_prompt_cache.py vram g3 evict park side dflash resume bypass: RESULT PASS** (42 PASS, 0 FAIL; 38 slots at
  max_ctx 33,024; XTX peak 23,046 MiB). Its scratch tier was purged afterwards (0 entries).
- Disk: the sweep wrote 5.3 GB to `$ARTIFACTS/sweep-262k`; 30 GB is free there.

## Setup

- **Engine** (`sweep_262k.py k3`): FlashNextTP T = 4 at max_ctx 262,144 (the dual-tape QSA engine), the DFlash drafter (device
  accept, features on the XTX), the batched prefill at chunk 2048 with the defaults (`MOE_PATH` own, `Q6K_PATH` own,
  `WMMA_OUT` bf16, `ROUTER_LOGITS` wmma). The accuracy and timing stages build no prompt-cache tier (they never use the cache;
  the full tier leaves ~265 MiB above the XTX reserve, and snapshots are not counted anywhere). The cache stages
  (`sweep_262k.py cache`, tests/cache/test_prompt_cache.py's Rig) build the full tier sized from the measured headroom, as served.
  The plain-decode curve uses a T = 1 engine (`sweep_262k.py plain`), the engine plain decode has always been measured on.
- **Text** (mostly public, see below): `tools/eval/lc_probes.py build $ARTIFACTS/sweep-262k/lc --depths 65536,131072,261632 --min-chars 1300000 --seq`:
  CPython 3.12 standard-library modules (PSF licence), 23 files (`_collections_abc.py` .. `doctest.py`, sorted, each >= 20 KB),
  1,389,196 chars, 358,143 tokens, cut at newlines into ~400-char pieces tokenized alone (gg_golden --tokenize).
  **Not all public:** two of the 23 files, `_sysconfigdata__linux_x86_64-linux-gnu.py` and `_sysconfigdata_x86_64_conda_linux_gnu.py`,
  are generated by the local conda env and hold its build and install paths (184 lines with that machine's absolute paths in the first). The
  131,072 window falls inside them. So seq.ids, the probe contexts and every logits / token file under `$ARTIFACTS/sweep-262k`
  must not be published as they are. qsa/lc's smaller haystack (5 files) does not reach them.
  - `seq.ids`: the needle-free haystack. Its prefixes give every G4 window (positions N .. N + 255 for N = 65,536 / 81,920 /
    131,072 / 261,632), the soak, the curves' depths and the cache stages' conversations.
  - `d<D>/ctx.ids` (D = 65,536 / 131,072 / 261,632): the haystack with a two-line block (a needle `VAULT_<NAME>_CODE =
    'dddd-dddd'` and a rare string `TOKEN_<TAG>_KEY = '<14 random letters and digits>'`) at 10 / 50 / 90 % of D; the probes ask
    `assert VAULT_<NAME>_CODE == '` (or the TOKEN) after the context, 24 greedy tokens.
  - The smoke texts: the first 200 tokens of refs/d281-short/e2e_{prose,code,shell}_decode (k3_smoke's prompts).
- **~262K = 261,632**: the deepest window that leaves room for 256 teacher-forced positions (to 261,887), a probe question and
  24 generated tokens, or a 200-token prompt and 128 generated tokens, below 262,144.
- **Host**: compaction_proactiveness 0 for every run (recorded with each); MemAvailable recorded with each GPU run; every GPU run
  under tools/guard.sh (the .mon peaks in the results).

## Method

**The 64K floor, pooled over run pairings** (`g4_floor.py pool`, committed with the 32K result before any K3 64K number existed).
llama.cpp past ~8K is not run-to-run reproducible (every row's FNV differs between two runs at 32K and 64K), so one run pairing
is not the floor. Per position, the mean KLD over every (one-token run, prefill-then-decode run) pairing; the floor's mean is its
mean; its 95 % interval resamples POSITIONS (`boot()`, seeded, 10,000 resamples: the pairings of one position are not
independent samples); top-1 is the mean over pairings. The per-pairing table is kept beside it. Bars: **G4a** (K3
prefill-then-decode vs K3's own one-token decode) mean KLD <= the pooled upper bound and top-1 >= the pooled top-1 - 1 point
(g4d's rule on the pooled floor); **G4b** (K3 prefill-then-decode vs llama.cpp's prefill-then-decode at ub 2048, run 1; run 2
reported) <= 0.0076 and >= 95.9 % (the D-281 bar); **D-281 long** (K3 one-token vs each llama.cpp one-token run) reported per run.
At 64K: 3 one-token runs x 2 prefill-then-decode runs = 6 pairings, all at ub 2048 (the 32K floors' configuration).

**llama.cpp's memory past 64K** (measured, not changed: the M0 layer split 0.55 / 0.45 stays). At ub 2048 the XT holds the
largest compute buffer: XT peak 17,855 MiB at n_ctx 33,024 and 18,774 MiB at 65,792 (compute buffer 3,492 -> 4,134 MiB; KV +
indexer 257 -> 512 MiB per card), +0.028 MiB per cell. Projected XT peaks at ub 2048: ~20.5 GB at 131,072 and more at 262,144,
against the guard's 19,850 line. So:
- **G4b at ub 2048 binds at 81,920** (measured XT peak 19,242 MiB, ~600 MiB of margin);
- **at 131,072 and 261,632 llama.cpp's prefill-then-decode runs at ub 512** (measured at n_ctx 261,888: XTX 20,281 / XT 18,849
  MiB); a control at 64K measures what the ub alone changes (llama.cpp's pd at ub 512 vs ub 2048, beside its own run-to-run spread);
- the probes run at ub 512 as the stored qsa/lc probe references did. (The skipped item 4's one-token run was prepared at ub 512:
  a one-token fill is one token per llama_decode, so its ub only sizes the reserve; the 2K control: the one-token references at
  ub 2048 == at ub 512.)

**K3's one-token decode at every window comes from one policed soak** (`sweep_262k.py k3 soak`): T = 4 passes from position 0
over seq.ids to 261,888 (T = 4 == one-token decode bitwise: tools/gates/tp_multi.py), each pass's features injected into the
DFlash drafter, the rows at the windows kept. The same run is item 6's soak. (It would also have been K3's side of the
one-token D-281 at ~262K, item 4, skipped by user decision 11-05.)

**Capacity buckets.** The engine has none: `git log -S bucket` and a grep of kernels/ and the tape builder find the word only in
the serve metrics and in the QSA spec's design text. The QSA tape is built once at max_ctx with fixed task counts; the
boundaries the soak crosses are the ones the code has: 2051 (the dense tape -> the QSA tape, the one-time pool backfill of
QSA_BACKFILL = 2048 positions per launch), nv 512 (QSA_SCORE starts scoring, SELECT stops being the identity), every DFlash
ring wrap (4096 rows) and the drafter's 2048-position window, and QSA_SCORE's fixed 64 tasks of 32-pool chunks over a pool count
that grows with depth (each task's loop grows by one chunk every 8,192 positions).

**Probes** (S7's protocol, `sweep_262k.py k3 probes`): the prompt (context + question) through the batched prefill
(spec.generate(prefill=)), 24 greedy tokens; llama.cpp (gg_golden --lc greedy, ub 512): the context batched, the question one
token per decode, 24 greedy tokens. An answer counts as found when the generated text starts with it (lc_baseline's rule).

**Curves** (timing; nothing else running; no file I/O while timing): TTFT from empty for the prompt seq[:D] + a smoke text
(the prefill of n - 1, the hand-off, the last token's pass, the first token) without the DFlash inject (spec.generate) and with
it (DFlashK3Drafter.generate: seconds - gen_seconds); DFlash decode 128 tokens (ms/token = gen_seconds / (gen_tokens - 1));
plain decode on the T = 1 engine (144 greedy tokens through step_token, the last 128 timed); an 8K append at depth 131,072 and
250,000 with the DFlash inject (speed7's protocol). Medians of 3 up to 32K, one run deeper (each deep run is one 2-3 minute
fill): curve points, not speed claims. compaction_proactiveness 0 and MemAvailable are printed with every timing line.
