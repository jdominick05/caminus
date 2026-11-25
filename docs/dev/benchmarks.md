# Measurements

Every number in the README, with the date it was measured, the tool that measured it and the conditions. All of them come from
one machine: an RX 7900 XTX (device 0, driving the monitors) and an RX 7900 XT (device 1) on a PCIe Gen4 x4 link, 29 GiB of
host RAM, Linux, PyTorch 2.14.1+rocm7.2's HIP runtime, the code objects built with TheRock ROCm 10.2.0a20261011, the clocks held
at DPM `high` by the guard. Most sections below are the Coder (GSQ-RCO IQ1_M, two shards, 58.4 GB, with the DFlash2 Q8_0
drafter); [the other models](#the-other-models) have their own section.

The raw logs and result files stay in the development history, which is private; what is here is what they say. The
reference dumps the accuracy gates compare against (llama.cpp's own tensors and logits) are not distributed either: the
tools that produce and compare them are in this repository (tools/reference/gg_golden.cpp, tools/gates/, tools/k3_smoke.py,
tools/eval/sweep_262k.py), so the numbers can be reproduced on the same hardware with references you build yourself.

**The comparison:** llama.cpp at the fork the gates use (ignite-llama at e4e3f8eda, [github.com/jdominick05/ignite-llama](https://github.com/jdominick05/ignite-llama): docs/dev/building.md; a HIP build with the same TheRock toolchain), on the same file and the same two cards: llama-server with `-sm layer -ts 550,450`, flash attention on, a q8_0 KV
cache, DFlash with `--spec-draft-n-max 3`; llama-bench for prompt processing. llama.cpp's numbers were measured on 2026-11-01
unless a row says otherwise.

**0.3.0 (2026-11-17):** the release rows were rerun on the release commit (006130e) with tools/release/bench_runner.py:
45 of them measured, 54 marked not rerun (their inputs or references were lost, the Swift features test's second engine
does not fit beside the first on one card, the two-card byte comparison needs a step outside the runner, or the runner's
command for the ThinkingCap MTP gate failed to start). One more ran but does not print its value (the Coder server's
next-turn first token), so it keeps its 11-06 number. The llama.cpp yardstick rows and the diagnostic rows keep their dated
numbers (not rerun for 0.3.0).

## Decode

| What | Caminus | llama.cpp | Date | Tool and conditions |
|---|---|---|---|---|
| Plain decode, short context | **9.87 ms/token** (101 tok/s) | | 11-17, 0.3.0 | `benchmarks/fn_bench.py 96 $CODER --tp`: 96 greedy tokens, the two-card engine, one token per pass |
| Plain decode in the generation loop | **104.6 / 104.9 / 103.9 tok/s** (prose / code / shell) | **44.2 / 44.2 / 38.0 tok/s** | 11-06; not rerun for 0.3.0 (references lost) | `tools/k3_smoke.py --model coder`: 128 greedy tokens after a 200-token prompt, per text; llama.cpp: 256 tokens through llama-server, the same prompts (its shell run is a low outlier: 43.8 in another repetition) |
| Decode with DFlash | **156.0 / 202.5 / 123.3 tok/s** | **92.4 / 89.1 / 65.1 tok/s** | 11-06; not rerun for 0.3.0 (references lost) | the same smoke run: the DFlash2 drafter inside the engine, 4-token verify passes; the output equals plain greedy decoding (checked in the run) |
| The same smoke on the release package | plain 103.8-105.1, DFlash 156.0 / 202.8 / 123.2 tok/s | | 11-06 | `tools/k3_smoke.py` against the installed wheel (`CAMINUS_SMOKE_INSTALLED=1`) |
| Plain decode at depth | 10.4 / 10.9 / 11.6 / 12.8 / 15.3 ms/token at 2K / 32K / 64K / 128K / 262K | ~30 / ~38 / ~60 ms/token at 64K / 128K / 262K | 11-05; not rerun for 0.3.0 (references lost) | `tools/eval/sweep_262k.py plain` and `curves`: 128 tokens after a prompt of that depth; llama.cpp: its one-token decodes in the sweep's reference jobs (docs/specs/sweep-262k.md, section 5) |
| Plain decode at depth, parallel pool selection | 10.69 -> 10.57 / 11.17 -> 10.72 / 15.58 -> 11.39 ms/token at 2,048 / 32,768 / 261,632 (before -> after) | | 11-08 | the change that made the indexer's pool selection parallel (OP_QSA_PSEL): the sweep's plain-stage timing (tokens 16 .. 143 of 144 greedy tokens after a batched prefill of that depth, one text), old and new interleaved, medians of 3; the same greedy tokens. The row above predates it; the sweep has not been rerun |
| Decode at 32K before / after split attention | 25.00 -> 11.17 ms/token | 25.8 ms/token | 11-03 | one-token decode passes at that depth without a fill, measured in the split-attention step |

How many drafted tokens are accepted depends on the exact greedy text. A change of the attention's summation order (split
attention, 11-03) shifted near-tied tokens and with them the acceptance, without any change to the drafter. At ~262K DFlash
costs more than plain decoding on prose (18.25 vs 15.26 ms/token; measured 11-05, not rerun for 0.3.0: references lost), so the server drafts only when its rolling measurements say
it pays (`serve/draftpolicy.py`).

## Accuracy

| What | Result | Bar | Date | Tool |
|---|---|---|---|---|
| End to end against llama.cpp, short context (the D-281 gate) | mean KLD **0.0057**, top-1 **96.2 %** over 768 positions (3 texts x 256) | llama.cpp's own batch-vs-decode floor: KLD 0.0076, top-1 95.9 % | 11-03; not rerun for 0.3.0 (references lost) | `tests/engine/test_flashnext_e2e.py` against llama.cpp's one-token decode dumps of the same tokens (teacher forced) |
| The smoke's quick check | mean KLD 0.0044 against a floor of 0.0063 (64 positions x 3 texts) | the floor on the same positions | 11-06; not rerun for 0.3.0 (references lost) | `tools/k3_smoke.py` on the release package (a subset of the gate above, recomputed each run) |
| Long context with the QSA indexer | mean KLD 0.0021 / 0.00052, top-1 99.6 / 100 % at 8K / 32K | the long-context floors | 11-03; not rerun for 0.3.0 (references lost) | `tools/eval/lc_baseline.py` on `tools/eval/lc_probes.py`'s contexts; copy probes answered as llama.cpp answers them |
| The batched prefill against llama.cpp's prefill-then-decode (G4b) | 0.00092 / 0.00127 / 0.00321 / 0.00020 at 65,536 / 81,920 / 131,072 / 261,632 | 0.0076 / 95.9 % | 11-05; not rerun for 0.3.0 (references lost) | `tools/eval/sweep_262k.py k3 ... g4` and `cmp`; llama.cpp at ub 2048 to 81,920, ub 512 beyond (its memory) |
| Needle and rare-string recall, 64K to ~262K | **17 / 18** probes | llama.cpp **16 / 18** on the same probes | 11-05 | `tools/eval/sweep_262k.py k3 ... probes`; no probe llama.cpp passes fails here (docs/specs/sweep-262k.md, section 3) |
| A 262K soak | one policed run of 65,472 four-token passes from 0 to 261,888, no unexplained bail | | 11-05 | `tools/eval/sweep_262k.py k3 ... soak` |

The engine's weight-format and BF16 kernels match llama.cpp's bit for bit (tests/engine/test_k3_gguf_gsq.py, tests/engine/test_k3_bf16_mv.py
and the golden-layer tests); most speed changes kept the previous build's logits bitwise across grid sizes and 4-token verify
passes (tools/gates). Greedy text can still split from llama.cpp's at near-tied tokens, as llama.cpp's own runs split from each
other.

## Reading prompts (the batched prefill)

| What | Caminus | llama.cpp | Date | Tool and conditions |
|---|---|---|---|---|
| From empty, including the hand-off to decode | **2,909 / 2,873 / 2,806 tok/s** at 4K / 8K / 32K | llama-bench pp4096 2,628, pp8192 2,884-2,901, pp32768 2,791 | 11-04; not rerun for 0.3.0 (references lost) | `tests/prefill/test_prefill_s6.py $CODER speed7 --max-ctx 33024` (TTFT medians); llama-bench at its best ubatch |
| Deep prompts | 2,775 / 2,715 / 2,576 / 2,393 tok/s at 8K / 64K / 128K / 262K | 1,890 / 1,433 / ~950 tok/s at 64K / 128K / 262K (ub 512, what fits) | 11-05; not rerun for 0.3.0 (references lost) | `tools/eval/sweep_262k.py k3 ... curves`; llama.cpp: its probe fills in the sweep |
| Through the server | 11,285 tokens in 4.4 s (2,557 tok/s); 101,525 tokens in 40.1 s (2,538 tok/s); 2,537 tokens: first token in 1.2 s | | 11-06; the 101,525-token read rerun 11-17, 0.3.0 (same within 2 %; the 0.3.0 run prints no time for the other two) | `tools/serve_smoke.py --stages server`: the real `caminus serve` at its default configuration, synthetic prompts |

The prefill misses its own target of 3,000 tok/s at 8K (about 1 % under llama.cpp's best there).

### Prefill-GEMM arms on one RX 7900 XT (WMMA yardsticks, not the engine's quant kernels)

| 2048x5120xN | bf16 WMMA | W8A8 int8 WMMA | software fp8 | Date | Tool and conditions |
|---|---|---|---|---|---|
| 5120 | 55.8 TF | **62.0 TF** | 13.5 TF | 11-13 | `benchmarks/int8_gemm_probe.hip --shapes ... --reps 20 --check` under `tools/guard.sh` on the XT (device 1), gpu-window, no fault; fp64 check on the 64x64 C corner per arm: 7.97e-06 / 6.47e-08 / 8.74e-07, 0 nan |
| 17408 | 39.3 TF | **64.7 TF** | 13.7 TF | 11-13 | same |
| 20480 | 38.8 TF | **58.7 TF** | 13.8 TF | 11-13 | same |

torch bf16 matmul (rocBLAS) on the same card at the same shapes: 76-80 TF (`benchmarks/fp8_gemm_probe.py`, 11-12); fp8-as-bf16 through torch: 63-67 TF, maxerr 0.10-0.18 vs fp32; triton fp8/int8 and torch fp8 matmul: unsupported on gfx1100.

Reading: gfx1100 has no fp8 WMMA (the builtins are gfx12-only), and its one 8-bit WMMA (`i32_16x16x16_iu8_w32`) ties the engine's Q4_K wide tile at prefill (both ~59-64 TF) while costing 8 bpw vs Q4_K's ~4.5 - a ~1.8x byte loss at decode, where the card is read-bound. Software fp8 loses ~4x to the per-byte e4m3 decode. So the low-bit lever on these cards is the 4-bit formats already in use; the remaining prefill headroom over our tiles is unpack throughput (rocBLAS bf16's 76-80 TF is the tile-scheduling ceiling), not bit width.

## Serving

All from `tools/serve_smoke.py` on 11-06, rerun 11-17 on 0.3.0 where a row says so (the real server at its default 262,144-token configuration, started twice; scratch
cache directories; synthetic conversations):

| What | Measured |
|---|---|
| Start to ready | 67-68 s (the engine build 46 s; 11-17, 0.3.0) |
| VRAM | XTX 23,071 MiB at ready, 23,091 at the peak (its guard line 23,800, with a 512 MiB reserve kept); XT 19,019 / 19,037 (line 19,850); rerun 11-17, 0.3.0: same within 2 % |
| The next turn of an 11K-token conversation | first token in 0.21-0.23 s (its state live in VRAM; 11-06: the 0.3.0 rerun passed but prints no time for this turn) |
| After a restart | the conversation back from the disk tier, first token in 0.21 s at 11.6K tokens; the stop's park took 0.90 s (1.8 GiB), 1.76 s at 101K (3.9 GiB) (11-17, 0.3.0) |
| Decode through the server | 5.8-5.9 ms/token on tool-call code with DFlash at 0.6K and 32K (most drafts accepted; 11-17, 0.3.0); prose and thinking text accept fewer drafts and run slower |
| Server overhead | the SSE chat route + tool parser over the engine alone: +0.83 % (OpenAI) / +1.39 % (Anthropic) at 0.6K, -0.13 % / +0.09 % at 32K (11-17, 0.3.0) |

## The other models

> [!NOTE]
> Qwen3.6-35B-A3B's HauhauCS file, Qwen3-Coder-Next and AREX-2 are **paused** for the next release (`caminus.models.PAUSED`). Their
> rows below and in the two-card yardstick are from 2026-11-07 / 11-11 and are not rerun until they come back. The
> Qwen3.6-35B-A3B GSQ file is served again, on own code: [its section](#qwen36-35b-a3b-gsq-hybrid-own-code-one-card).

Measured 2026-11-07 on one card each (the RX 7900 XTX unless noted) or on both cards for Qwen3-Coder-Next; llama.cpp on
the same card(s) with llama-bench (decode tg128, prompt pp2048). Speeds are decode (writing) or prefill (reading a
prompt), in tok/s.

| Model | Caminus decode | llama.cpp decode | Caminus prefill 2K | llama.cpp prefill 2K | Accuracy gate |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B HauhauCS Q3_K_P | 192.5; 140.7 at 32K | 122.1; 107.7 at 32K | 4,136 | 2,923 | G4 0.00171 / 0.00181 |
| AREX-2 Q4_K_M (dense 27B) | 41.0 | 36.3 | 928 | 910 | short-context KLD 0.0003, top-1 99.1 % (floor 0.0033 / 97.9 %); G4 0.00053 / 0.00048 |
| Qwen3-Coder-Next UD-IQ3_S (both cards) | 136.5; 67.9 at 32K | 81.8; 72.6 at 32K | ~135 (one token per pass) | 2,787 | bitwise gates on the engine; no batched prefill yet |

Qwen3-Coder-Next decodes slower than llama.cpp at 32K and reads prompts about 20 times slower: its batched prefill
isn't written yet. Tools: `tests/prefill/test_q36_prefill.py` (ops, handoff, e2e, speed) and `tools/serve_smoke1.py`.

### Qwen3.6-35B-A3B GSQ-hybrid (own code, one card)

`Moe35Own` (`engine/q35_moe.py`) with its batched prefill (`prefill/q35moe.py`): own code from the GGUF file to the token
(the served smoke's own-code check below). Measured 2026-11-24 on the RX 7900 XT alone; the RX 7900 XTX: pending the
release window. llama.cpp on the same card: 90.9 tok/s decode (2026-11-07, not rerun).

| | F16 KV | Q8 KV (`--kv q8`) |
|---|---|---|
| Decode, short context (positions 128 .. 191) | 210 tok/s (4.75 ms) | 202 tok/s (4.96 ms) |
| Prefill, 2,048 positions from empty (chunk 2048 / 1024) | 6,571 / 6,013 tok/s | 6,346 / 5,829 tok/s |

- Decode: `tools/guard.sh LOG 300 python -I tools/gates/q35_gmodel.py GGUF --ref REF.npz --speed 64 [--kv q8]` (the median
  of 64 greedy passes after the 128 reference rows; two F16 runs agree to 0.01 ms).
- Prefill: `tools/guard.sh LOG 300 python -I tests/prefill/test_q35_prefill.py GGUF --ref REF.npz --speed 2048 [--kv q8]`
  (medians of 3).
- Accuracy, decode (`tools/gates/q35_gmodel.py`, 128 positions against the float64 reference `tools/reference/q35_ref.py`;
  printed, not gated: no llama.cpp one-token rows of these tokens): mean KLD 3.43e-4, top-1 128/128 (F16); 3.66e-4,
  128/128 (Q8); YaRN x2 (`--yarn 2` against a YaRN x2 reference) 9.79e-5, 128/128 (F16) and 4.91e-4, 127/128 (Q8).
- Accuracy, prefill then decode (`tests/prefill/test_q35_prefill.py --llama LL.bin`: positions 0 .. 63 prefilled, 64 .. 127
  decoded): mean KLD 3.43e-4, top-1 64/64, against llama.cpp's own prefill-then-decode 9.29e-4, 63/64 (F16; PASS); Q8
  3.41e-4, 64/64 (decode-only 2.69e-4). Chunk geometry and prompt-cache stops change no bit.
- Served (`tools/guard.sh LOG 900 python -I tools/serve_smoke1.py --model GGUF --scratch DIR --card 1 --max-ctx 262144
  --needle 250000 --block-imports gguf,tokenizers`, F16 KV, its full 262,144 context on the XT, 25 VRAM cache slots):
  greedy decode 211 tok/s; a 6,760-token prompt at 5,935 tok/s (first token 1.15 s); the needle at 37 % of a
  249,005-token prompt found, read at 1,501 tok/s (first token 166 s), then 89 tok/s decode at that depth (at 131,072:
  119,378 tokens at 2,699 tok/s, 44 s, then 129 tok/s). The tool round trip, Anthropic route and abort pass; the server
  loads only own-code objects and never imports gguf-py or tokenizers.

### Swift 1.5 Q4_K_M (dense 27B, own code)

Measured 2026-11-11; llama.cpp with `llama-bench -fa 1` on the same card. Untuned: decode short context only. The Caminus rows are not rerun for 0.3.0 (references lost).

| | RX 7900 XTX | RX 7900 XT |
|---|---|---|
| Caminus decode (short context) | 42 tok/s (23.8 ms) | 36 tok/s (27.5 ms) |
| llama.cpp decode (tg128) | 37.1 | 31.8 |
| Caminus prefill, 2,048 positions from empty | — | 771 (chunk 2048), 760 (chunk 1024) |
| llama.cpp prefill (pp2048 / pp512) | — | 763 / 770 |

- Accuracy: prefill 64 positions, then 64 decode passes against the float64 reference: mean KLD 3.0e-7, top-1 64/64.
  llama.cpp's own prefill-then-decode on the same rows: 3.2e-4, 64/64 (`tools/guard.sh LOG 1800 python -I tests/prefill/test_q35_prefill.py GGUF --ref REF.npz --llama LL.bin`) (measured 11-11; not rerun for 0.3.0: inputs lost).
- The served path on the XTX (rerun 11-17 on 0.3.0; `tools/guard.sh LOG 2400 python -I tools/serve_smoke1.py --model GGUF --scratch DIR --card 0
  --max-ctx 32768 --needle 28000 --restart --block-imports gguf,tokenizers`):
  - greedy decode 46 tok/s;
  - a 6,760-token prompt at 1,168 tok/s, then decode at 43 tok/s;
  - a 27,712-token prompt at 979 tok/s, then decode at 39 tok/s.

  The attention reads every earlier position per query head, so both the prompt rate and decode fall with depth.
- Swift 1.5 needs a 24 GB card. With its prompt cache it projects 19.1 GiB at 12K, past an XT's room.

#### Verify passes (the base for speculative decoding)

`engine/q35_dense_verify.py` runs T tokens in one pass on the decode engine's weights and states (module
`dec_q35_swift_ver`). Measured 2026-11-11, the table rerun 2026-11-17 on 0.3.0 (its faster decode step raises the ratios), after an own prefill of real text: ms per pass against one decode step at the
same depth, with the pass committing its own accepted tokens as it would in use (the benchmark's walk: medians of 9
interleaved rounds; `tools/guard.sh LOG 1800 python -I benchmarks/q35_verify.py GGUF --depths 512,4096 --per-wgp 5,6,7`):

| card | depth | decode step | T = 2 | T = 3 | T = 4 |
|---|---|---|---|---|---|
| RX 7900 XTX | 512 | 22.4 ms | 27.1 (1.21x) | 31.4 (1.40x) | 35.1 (1.57x) |
| RX 7900 XTX | 4,096 | 23.0 ms | 28.1 (1.22x) | 32.6 (1.42x) | 36.9 (1.61x) |
| RX 7900 XT | 512 | 26.5 ms | 31.1 (1.18x) | 35.9 (1.35x) | 40.7 (1.54x) |
| RX 7900 XT | 4,096 | 27.3 ms | 32.4 (1.19x) | 37.5 (1.37x) | 42.5 (1.56x) |

The T-token tiles unpack each weight word once per pass and run every token's sums on it (the same bits). Before
(the tiles unpacking again per token, the rows tiles running whole per token), T = 4 cost 1.92x / 1.70x (XTX) and
1.98x / 1.79x (XT). Where a T = 4 pass's time went, by op on the critical path (`K3_TRACE`, XTX, depth 512): the Q4_K
matvecs 2x a decode step's time for the same rows, the Q8_0 / Q5_K rows tiles 2.3 .. 3.1x; after, the rows tiles
~1.0x and the Q4_K matvecs ~1.5x, the gate and up still two segments.

- A pass of T tokens pays for itself when the drafter's tokens are accepted often enough to average more than that ratio
  in output tokens per pass (plus the drafter's own time).
- The commit's replay (the state taking the accepted tokens first) costs little: the same passes at commit 0 are at most
  1.3 ms (2.4 %) faster. The grid makes little difference: at T = 4, 5 to 7 workgroups per WGP are within 2.1 ms of the decode
  engine's grid (XTX at 4,096: 6 per WGP 51.9 ms against 53.9).
- The gate (rerun 11-17 on 0.3.0 on the XTX: passes; `tools/guard.sh LOG 1800 python -I tests/engine/test_q35_verify.py GGUF --depths 0,512`): for T = 1 .. 4 a
  walk of passes with rejected rows holding wrong tokens, decode steps between them, the lazily committed states handed
  both ways: every accepted row and every decode step bitwise the decode steps; passes of different T on one state (a pass
  of 2 committing 3 of a pass of 4) bitwise; a commit past the last pass's tokens refused on the host; T = 4 at 3 / 6 / 7
  workgroups per WGP bitwise. On both cards. Forced bails in the verify tape: `tools/guard.sh LOG 1200 python -I tools/gates/forcebail.py 4 40 tiny
  --engine dense35ver --model GGUF --ctx 512 --tokens FILE`.

#### The multi-token-prediction block's Q4_0 matvec (the MTP drafter's weights)

Swift 1.5 keeps its prediction block (`blk.64`: eh_proj, attention, feed-forward) in Q4_0, which no other model here
uses. `kernels/quant_q4_0.h` reads it in a plane layout made at load (`engine/own_layout.py q4_0_planes`); one
launch takes 1 to 4 tokens. `tools/guard.sh LOG 1800 python -I benchmarks/q35_mtp_gacc.py GGUF` records 116 activation
vectors at each of the block's op sites while it runs teacher-forced over two texts of this tree, then on 256 rows of
each tensor (RX 7900 XT):

| tensor | rows x K | error vs float64, mean / p99: own | llama.cpp | µs: read | llama.cpp | own | own, 4 tokens |
|---|---|---|---|---|---|---|---|
| eh_proj | 5,120 x 10,240 | 1.9e-4 / 7.3e-4 | 3.9e-2 / 1.9e-1 | 41.8 | 44.2 | 52.6 | 69.4 |
| attn_q | 12,288 x 5,120 | 5.0e-5 / 2.3e-4 | 6.1e-3 / 2.8e-2 | 49.8 | 52.2 | 54.2 | 73.2 |
| attn_k, attn_v | 1,024 x 5,120 | 0.9-1.2e-4 / 3.2-4.6e-4 | 1.3-1.4e-2 / 4.9-5.8e-2 | 12.3-12.4 | 14.8-15.0 | 16.2 | 17.5-17.6 |
| attn_output | 5,120 x 6,144 | 2.0e-4 / 6.7e-4 | 2.7e-2 / 9.2e-2 | 26.7 | 28.0 | 35.5 | 44.7 |
| ffn_gate, ffn_up | 17,408 x 5,120 | 0.8-1.0e-4 / 3.3-3.7e-4 | 1.0-1.5e-2 / 3.9-5.5e-2 | 73.1-73.8 | 75.5-76.3 | 77.1 | 93.5-93.6 |
| ffn_down | 5,120 x 17,408 | 8.7e-5 / 3.9e-4 | 1.6e-2 / 6.8e-2 | 73.4 | 76.4 | 77.6 | 91.5 |
| the block's 8 matvecs | | | | 363 | 382 | 406 | 501 |

Own is 120 to 265 times closer to float64 than llama.cpp on every tensor (llama.cpp rounds the activations to 8 bits),
and a 4-token launch gives each token the bits of its own one-token launch. Standalone, one launch per matvec, copies
rotated past the Infinity Cache.

#### Speculative decoding with the prediction block (engine level)

Measured in the engine: `benchmarks/q35_mtp_spec.py` drives it directly, with no server, HTTP or sampler in the loop.
The llama.cpp column is `llama-server`'s own decode timing (`timings.predicted_per_second`), with `-ctk q8_0 -ctv q8_0
-c 1024`, MTP drafting 3 tokens; its plain decode in that setup is 35.6 tok/s. Compare the speed-ups: llama.cpp gets
2.19x / 2.18x / 2.00x from its MTP, Caminus 2.15x / 2.14x / 1.97x. The tok/s difference is the plain decode's.
Both are measured through their servers below. llama.cpp's MTP output differed from its plain
output on the code prompt from token 42; Caminus's is the plain loop's token for token.

`engine/q35_mtp.py` runs the block on its own module (`kernels/objects.tsv dec_q35_swift_mtp`). A round writes the block's rows
for the last round's accepted drafts and the newest token from the verify pass's h rows, then chains 3 drafts on the
card. The verify pass of 4 tokens keeps the drafts its own greedy tokens equal. `tools/guard.sh LOG 1800 python -I
benchmarks/q35_mtp_spec.py GGUF --prompt 48 --n 128 --ids IDS.json` runs plain greedy decoding and the speculative loop
on the same prompts. Every generated token is the plain loop's. Decode tok/s over the 127 tokens after the first (measured 11-12; not rerun for 0.3.0: inputs lost):

| prompt | RX 7900 XTX: plain | speculative | tokens per round | llama-server + its MTP (plain 35.6), same prompts |
|---|---|---|---|---|
| code | 42.5 | 91.3 (2.15x) | 3.63 | 78 |
| prose | 42.6 | 91.1 (2.14x) | 3.63 | 78 |
| shell | 42.7 | 84.0 (1.97x) | 3.34 | 71 |

The prompts are the first 48 tokens of the e2e reference texts (code / prose / shell), the inputs llama.cpp's MTP was
measured on. A round's parts on the XTX: the drafter 5.4 ms (its 3 heads are most of it: 1.4 ms each), the verify pass
34.4 ms. On this tree's own texts (256-token prompts of `python/caminus/arch.py`, `docs/dev/architecture.md` and
`build.sh`, no `--ids`): XTX 81.1 / 56.5 / 74.8 tok/s (plain 44.6), XT 70.6 / 49.1 / 65.4 (plain 37.7; both rerun 11-17 on 0.3.0). The prose
page is mostly tables, where the block's drafts are kept less often (2.27 tokens per round).
- The drafter's gate (`tools/guard.sh LOG 1800 python -I tests/engine/test_q35_mtp.py GGUF`), XT:
  - Rows filled in tapes of 4, of 3 + 1 and of 1 give the same bits, and a round of 1 to 4 rows gives the KV rows, hn
    and draft of one-row rounds.
  - The device argmax is the first maximum, and a NaN loses.
  - Drafts agree with the block in fp32 torch at 63 of 63 positions, including both chained drafts. The first draft's
    logits are at mean KLD 2.9e-7 (rerun 11-17 on 0.3.0).

Served (`caminus serve --draft mtp`): `tools/guard.sh LOG 3600 python -I benchmarks/q35_served_ab.py --model GGUF --ids
IDS.json --scratch DIR --card 0 --draft mtp --max-ctx 12288`, the server's own decode timing for 128 greedy tokens,
interleaved over 3 rounds, the same prompts as the DFlash table below, measured 2026-11-12 on testing `db7b6d9` with
nothing else on the cards. The replies are the plain server's token for token. The llama.cpp columns are stock llama.cpp's
`llama-server` with its own MTP on the same card, prompts and settings (f16 KV, `-c 12288`, 128 greedy tokens, median of 3
interleaved runs, measured 2026-11-13): `tools/guard.sh LOG 1800 python -I benchmarks/llamacpp_spec_ab.py --bin
LLAMA_BIN --model GGUF --ids IDS.json --dev ROCm0 --out DIR` (`--spec-type draft-mtp --spec-draft-n-max 3`; two cards:
`--dev ROCm0,ROCm1 --server-args '-sm layer -ts 550,450 -lm dio'`, flags both arms get; `--dry-run` prints both command lines). The Caminus columns are not rerun for 0.3.0 (inputs lost).

| prompt | RX 7900 XTX: `--draft off` | `--draft mtp` | llama-server: plain | + its MTP (accepted) |
|---|---|---|---|---|
| code | 42.8 | 91.3 (2.13x) | 36.4 | 71.4 (1.96x, 78 %) |
| prose | 42.7 | 95.9 (2.24x) | 36.5 | 82.5 (2.26x, 91 %) |
| shell | 42.8 | 73.5 (1.72x) | 36.2 | 74.2 (2.05x, 78 %) |

- llama.cpp's drafted replies equalled its plain ones on all three prompts, in every run.
- Shell: the engine-level run above keeps the same share of drafts (3.34 tokens a round, about 78 %) at 1.97x, but the
  served run gets 1.72x while code holds (2.15x engine, 2.13x served); the served path's cost per round on that prompt is
  not explained yet.

Served at depth on rendered real sessions (a 40,885-token agent session and a 22,700-token code session, rendered to
ChatML ids from saved Qwen Code chats, 11-13; `benchmarks/q35_served_ab.py`, median of 3 rounds, 128 greedy tokens,
one card, testing's objects; not rerun for 0.3.0: inputs lost; ThinkingCap's agent 40K row on one XT now needs `--auto-cache-slots 3` to place: with four checkpoint slots placement projects 18,890 MiB against 18,800 MiB of room on an idle card, with three 18,717, projected, not rerun):

| model, card | prompt | max-ctx | off | mtp | x |
|---|---|---|---|---|---|
| ThinkingCap, XT | agent 40K | 40960 | 29.9 | 41.3 | 1.38 |
| ThinkingCap, XT | code 22K | 32768 | 34.1 | 42.6 | 1.25 |
| Swift 1.5, XTX | agent 40K | 40960 | 32.7 | 47.4 | 1.45 |
| Swift 1.5, XTX | code 22K | 32768 | 36.9 | 47.4 | 1.29 |

code 22K: every mtp round's reply equalled the first plain round's, token for token. agent 40K: the replies agree for the
first 73 tokens, then the mtp arm stops (EOS) while plain continues 2 tokens - the same on testing without the
draft-policy change. Whether that is a near-tie at the end token or a verify difference is not measured yet (no margin
was recorded), so `--draft mtp` stays opt-in until it is explained (the default is `--draft dflash`, the DFlash2 drafter, kept when its file is present and the cards still hold the context: `arch.Dense35.DRAFT_DEFAULT`).

#### Speculative decoding with DFlash 2 (served)

The base model's DFlash 2 drafter (incoai's Qwen3.8-27B-DFlash2 Q8_0: 5 non-causal blocks reading 5 of the target's
layers) on Caminus's own code: `kernels/objects.tsv dec_q35_swift_dflash` (the Swift module's T-token matvecs plus
`kernels/ops_q35_dflash.h`: the drafter's conv, top-k, selector, windowed attention and ring), `engine/q35_dflash.py`,
`decode/dflash35.py`. The verify pass of 4 tokens checks its 3 drafts on the host. Every pass on the target feeds the
drafter its features: the verify passes, the one-token passes and the batched prefill's chunks.

Served: `tools/guard.sh LOG 3600 python -I benchmarks/q35_served_ab.py --model GGUF --ids IDS.json --scratch DIR --card 0
--max-ctx 12288` starts `caminus serve` once with `--draft off` and once with `--draft dflash`, interleaved over 3
rounds, and reads the server's own decode timing (`timings.predicted_per_second`) for 128 greedy tokens. The prompts are
the 48-token inputs llama.cpp's own drafter was measured on (the server puts 248044 first: one prompt token more).
Measured 2026-11-12 with nothing else on the cards. Re-measured on testing `db7b6d9` (the 27B docs' release check, one window, the four A/Bs one card at a time). The Caminus columns are not rerun for 0.3.0 (inputs lost).
The llama.cpp column is `llama-server --spec-type draft-dflash --spec-draft-n-max 3 -c 1024 -ctk q8_0 -ctv q8_0`; its
plain decode in that setup is 35.4-36.0 tok/s.

| prompt | RX 7900 XTX: `--draft off` | `--draft dflash` | llama-server + DFlash 2 (n-max 3), same prompts |
|---|---|---|---|
| code | 42.6 | 95.8 (2.25x) | 74.7 (80 % accepted) |
| prose | 42.7 | 94.6 (2.21x) | 74.8 (80 %) |
| shell | 42.7 | 71.1 (1.67x) | 67.3 (67 %) |

- The replies are the plain server's token for token on all three prompts. llama.cpp's drafted output also equalled its
  plain output on Swift.
- Context: the drafter's ~2 GB leaves Swift ~12-16K tokens of context on a 24 GB card with 4 cache slots (32K needs
  ~23.2 GB of ~22.3 GB room, projected), so `--draft dflash` is placed from 8K and `max_ctx auto` takes the largest that
  fits. `--draft mtp` and `--draft off` keep the 32K floor and more.
- `tools/guard.sh LOG 2400 python -I tools/serve_smoke1.py --model GGUF --scratch DIR --card 0 --draft dflash --max-ctx
  12288 --block-imports gguf,tokenizers` passes (rerun 11-17 on 0.3.0): greedy 58.1 tok/s; a 6,760-token prompt at 1,154 tok/s, then decode
  at 44 tok/s.
- Prompt reading with the drafter: every prefill chunk's features go through the drafter's batched inject
  (`Dflash35.inject_rows`: fc and k / v as the prefill's own WMMA GEMMs, passes of 512 rows). The same smoke's 6,760-token
  prompt: 1,154 tok/s with `--draft dflash` (TTFT 5.88 s) against 1,163 with `--draft off` (5.84 s; 11-17, 0.3.0); the first build's
  4-row inject tape read it at 678.
- The drafter's gate (`tools/guard.sh LOG 1800 python -I tests/engine/test_q35_dflash.py GGUF DFLASH_GGUF`), XT: against
  the float64 reference (`caminus/ref_dflash.py`, itself equal to llama.cpp's drafts at 10 of 10 positions), the injected
  K / V within 5e-4, the drafts' logits within 1.7e-3 and the drafts equal, at 20 positions and after 4,200 (past the
  2,048-position window and the 4,096-row ring's wrap); ring rows, logits and drafts bitwise at 3, 6 and 7 workgroups per
  WGP. `dec_q35_swift_dflash`: 88 VGPRs, 0 spills; every other object unchanged. Rerun 11-17 on 0.3.0 (XT): passes within the same bounds.
- The one-token passes' feature export (`Dense35Own(feature_layers=)`) changes no logit and exports the verify pass's rows
  bit for bit (`tests/engine/test_q35_features.py`) (measured 11-12; not rerun for 0.3.0: the test's second engine does not fit beside the first on one card, a test fix after 0.3.0).

### ThinkingCap RCO 4.2 bpw (dense 27B, own code end to end)

Measured 2026-11-11; llama.cpp with `llama-bench -fa 1` on the same card. The grid and task sizes are untuned. The Caminus rows are not rerun for 0.3.0 (references and inputs lost). Caminus's figure is the median of positions 64 .. 127 with each step synchronized
(`tests/engine/test_q35_dense.py --model GGUF --ref REF.npz`).

| | RX 7900 XTX | RX 7900 XT |
|---|---|---|
| Caminus decode (short context) | 46.9 tok/s (21.33 ms) | 39.4 tok/s (25.37 ms) |
| llama.cpp decode (tg128) | 40.5 | 34.1 |
| Caminus prefill, 2,048 positions from empty | 942 (chunk 2048), 909 (chunk 1024) | — |
| llama.cpp prefill (pp2048 / pp512) | 1,043 / 1,117 | — |

- Accuracy, against the float64 reference (`tools/reference/q35_ref.py`), 128 positions: mean KLD 2.4e-7, top-1 128/128.
  llama.cpp's one-token decode of the same rows: 2.7e-4, 128/128
  (`tools/guard.sh LOG 1800 python -I tools/gates/q35_gmodel.py GGUF --ref REF.npz --llama LLAMA.bin`) (measured 11-11; not rerun for 0.3.0: inputs lost).
- Per-matvec error against float64 on real rows: 1.0-1.9e-4 mean on every shape and format, against llama.cpp's 6.2-9.6e-3
  (`tools/guard.sh LOG 1800 python -I benchmarks/own_mv_census.py GGUF`).
- Prefill 64 positions, then 64 decode passes against the float64 reference: mean KLD 2.6e-7, top-1 64/64. llama.cpp's
  own prefill-then-decode of the same rows: 4.5e-4, 64/64 (`tools/guard.sh LOG 1800 python -I
  tests/prefill/test_q35_prefill.py GGUF --ref REF.npz --llama LL.bin --speed 2048`, which also prints the prefill speed
  above) (measured 11-11; not rerun for 0.3.0: inputs lost).
- The served path (rerun 11-17 on 0.3.0; `tools/guard.sh LOG 2400 python -I tools/serve_smoke1.py --model GGUF --scratch DIR --card N
  --block-imports gguf,tokenizers`; the XTX with `--max-ctx 32768 --needle 28000 --restart`):
  - XTX: greedy decode 52.6 tok/s; a 6,760-token prompt at 1,011 tok/s, then decode at 49 tok/s; a 27,712-token prompt at
    887 tok/s (the needle found), then decode at 44 tok/s;
  - XT: the server picks 53,248 tokens of context by itself; greedy decode 43.5 tok/s; a 6,760-token prompt at 766 tok/s.

#### Speculative decoding with its own prediction block (served)

ThinkingCap's multi-token-prediction block (`blk.64`, all Q6_K) on `kernels/objects.tsv dec_q35_tcap_mtp` (the planes' T-token
tiles, its Q8_0 head as a rows tile, the IQ2_S embedding reading the argmax key on the card for the chained steps),
`engine/q35_mtp.py`, 3 drafts a round, verified by the 4-token pass. `tools/guard.sh LOG 3600 python -I
benchmarks/q35_served_ab.py --model GGUF --ids IDS.json --scratch DIR --card 1 --draft mtp --max-ctx 12288`, measured
2026-11-12 with nothing else on the cards, the same prompts as the DFlash table below. Re-measured on testing `db7b6d9` (the 27B docs' release check, one window, the four A/Bs one card at a time). Not rerun for 0.3.0 (inputs lost):

| prompt | RX 7900 XT: `--draft off` | `--draft mtp` |
|---|---|---|
| code | 39.8 | 68.3 (1.72x) |
| prose | 39.7 | 73.6 (1.85x) |
| shell | 39.7 | 52.7 (1.33x) |

Stock llama.cpp's `llama-server` with its own MTP, the same card, prompts and settings as Swift's MTP table (f16 KV,
`-c 12288`, `benchmarks/llamacpp_spec_ab.py`, measured 2026-11-13): plain 33.6 tok/s; MTP 65.7 / 73.7 / 64.7 (1.95x /
2.19x / 1.92x; 72 / 87 / 72 % of its drafts accepted). Its drafted replies differed from its plain ones on prose (from
token 27) and shell (from token 66), in every run.

In the engine (`tools/guard.sh LOG 1200 python -I benchmarks/q35_mtp_spec.py GGUF --prompt 48 --n 128 --ids IDS.json`,
XT, 2026-11-13, testing with the two-step Q8_0 rows tile; not rerun for 0.3.0: inputs lost): 1.52x / 1.73x / 1.39x at 3.10 / 3.53 / 2.82 tokens a round.
A round is the drafter 7.5 ms (9.1 before that tile: its three head passes) and the verify pass 43.0 ms, 1.74x a decode
step (24.8 ms). On shell Caminus keeps 61 % of its drafts ((2.82 - 1) / 3)
against llama.cpp's 72 %; at 72 % our round would give about 1.55x, so most of the shell gap is the round's fixed cost
and the rest acceptance.

- The replies are the plain server's token for token. The block needs no extra file and little VRAM, so it keeps the
  card's full context; `--draft dflash` is faster where the drafter fits (below).
- The drafter's gate (`tools/guard.sh LOG 1800 python -I tests/engine/test_q35_mtp.py GGUF`, XT): rows in tapes of 4,
  3 + 1 and 1 give the same bits; a round of T rows gives T one-row rounds' KV rows, hn and draft; drafts agree with the
  block in fp32 torch at 100 % (first and chained), mean KLD 2.3e-7, the same at
  `K3_PER_WGP=3/6/7` (every check passes at each grid). A round (k = 3, the catch rows and two chained steps) takes 8.9 ms on the XT; the split
  per segment is not measured yet (the three passes of the Q8_0 head on its rows tile are the likely part). (Measured 11-12; not rerun for 0.3.0: the runner's command for this gate failed to start.)

#### Speculative decoding with DFlash 2 (served)

The same drafter on ThinkingCap's module (`kernels/objects.tsv dec_q35_tcap_dflash`: its IQ2_S embedding, its Q8_0 head) and its
verify module (`kernels/objects.tsv dec_q35_tcap_ver`, the T-token tiles of `kernels/quant_tcap_ver.h`). `tools/guard.sh LOG
3600 python -I benchmarks/q35_served_ab.py --model GGUF --ids IDS.json --scratch DIR --card 1 --max-ctx 12288 --auto-cache-slots 3`, measured
2026-11-12 with nothing else on the cards; the prompts are the 48-token inputs llama.cpp's own drafter was measured on;
llama-server with `--spec-type draft-dflash --spec-draft-n-max 3 -c 1024 -ctk q8_0 -ctv q8_0`, plain 33.2 tok/s. Re-measured on testing `db7b6d9` (the 27B docs' release check, one window, the four A/Bs one card at a time). The Caminus columns are not rerun for 0.3.0 (inputs lost). One XT places the drafter at 12,288 positions only with `--auto-cache-slots 3` (since the placement and auto-context room became one calculation): with four checkpoint slots it projects 18,803 MiB against 18,800 MiB of room on an idle card (the guard's 19,850 MiB line less the allocator margin, the cache's reserve and work and the projection's slack); four slots place 8,192. The server still sizes its cache from the card's real headroom: at 12,288, 5 slots (1,425 MiB), a 19,337 MiB peak through a 9,661-token prompt (11-25, 0.4.0's release commit).

| prompt | RX 7900 XT: `--draft off` | `--draft dflash` | llama-server + DFlash 2 (n-max 3), same prompts |
|---|---|---|---|
| code | 39.7 | 75.9 (1.91x) | 70.9 (79 % accepted) |
| prose | 39.6 | 77.0 (1.95x) | 74.8 (84 %) |
| shell | 39.5 | 60.5 (1.53x) | 48.1 (63 %) |

- The replies are the plain server's token for token. llama.cpp's drafted output differed from its plain output on prose
  and shell.
- IQ2_S's matvec (18 a token) reads at about 60 % of the read ceiling. Its 10-byte unit records are read with one
  16-byte load each at a 10-byte stride; `benchmarks/iq2s_load_probe.hip` (run through `tools/guard.sh LOG 180 python -I
  benchmarks/iq2s_load_probe.py --probe PROBE --card N`, which checks its VRAM first; the tile's lane mapping, the ffn's 17408 x
  5120, every launch on another of 8 copies so it reads memory, not the Infinity Cache, the arms interleaved) times that
  against aligned 8 + 2-byte planes and aligned 16-byte loads over exactly the same bytes on the XT: 48.4 / 48.2 /
  49.2 us, the same. The load shape is not the gap; the grid
  lookups and the unpack are what is left.
- The tile was ALU-bound: its unit loop issued ~483 instructions (+113 `s_delay_alu`) per pass of 20 weight bytes, ~24
  per byte read (IQ4_XS: ~9), most of them selecting a packed grid entry's half and spreading its 2-bit codes to bytes.
  The grid now sits in LDS as its magnitude bytes (8 KB, one 8-byte read per 8 weights): 436 instructions a pass (333 +
  103), the same bytes into the same sign and dot statements, so every sum keeps its bits. `tools/guard.sh LOG 1800
  python -I benchmarks/own_mv_census.py GGUF --formats IQ2_S` on the XT, interleaved (old, new, old, new), us per matvec:

  | shape | old | new | llama.cpp |
  |---|---|---|---|
  | ffn_gate / ffn_up [17408 x 5120] | 79.4 / 79.5 | 59.0 / 58.4 | 74.5 / 73.2 |
  | ffn_down [5120 x 17408] | 85.1 | 61.7 | 75.8 |
  | attn_qkv [10240 x 5120] | 50.9 | 40.1 | 43.9 |
  | attn_gate [6144 x 5120] / ssm_out [5120 x 6144] | 34.2 / 34.8 | 27.3 / 27.6 | 29.8 / 31.5 |

  Plain decode in the engine (`benchmarks/q35_mtp_spec.py`, XT, interleaved): 40.4 / 40.5 -> 40.9 / 41.2 tok/s (+1.6 %) (measured 11-13; not rerun for 0.3.0: inputs lost).
  LDS per workgroup: 8.2 -> 14.4 KB (8 per WGP still fit in 128 KB); the verify module 10.6 -> 16.7 KB (7 per WGP: the
  engines' 4 / 5 and the gates' 3 / 6 / 7 fit); the DFlash module drops its matvec op and both grids (16.1 -> 11.0 KB).
- The verify pass of 4 tokens: 43.0 ms against a 25.3 ms decode step at depth 512 (XT; `benchmarks/q35_verify.py`), 1.63x;
  Swift's was 1.45x (11-11; 1.57x on 0.3.0, against its faster decode step). `benchmarks/q35_verify_trace.py` puts the rest in the IQ3 / IQ2_S / Q5_K tiles (1.7-1.9x a token).

### Qwen3.8-27B GSQ-RCO IQ3_S (dense 27B, own code end to end)

ISTA-DASLab's plain Qwen3.8-27B in GSQ-RCO IQ3_S (12.1 GB; formats IQ3_S, IQ2_S, IQ2_XS, IQ2_XXS, Q2_K, one IQ1_M
matrix, Q4_K head, Q6_K MTP block). Decode module `dec_q35_gsq`, verify modules `dec_q35_gsq_ver` + `dec_q35_gsq_ver_m`
(the IQ1_M matrix in its own launch), drafters `dec_q35_gsq_mtp` and `dec_q35_gsq_dflash`.

Served: `tools/guard.sh LOG 3600 python -I benchmarks/q35_served_ab.py --model GGUF --ids IDS.json --scratch DIR --card 0
--draft mtp --max-ctx 12288` (and `--draft dflash`), one RX 7900 XTX, 128 greedy tokens, median of 3 interleaved
rounds, three short chat prompts of 35-39 tokens (code: an ISO 8601 parser with tests; prose: how a refrigerator moves
heat; shell: the ten largest files under /var/log), measured 2026-11-15 on the commit that added the drafter modules; not rerun for 0.3.0 (inputs lost).
Every drafted reply equals the plain reply token for token.

| prompt | `--draft off` | `--draft mtp` | `--draft dflash` |
|---|---:|---:|---:|
| code | 55.6 | 70.0 (1.26x) | 74.7 (1.37x) |
| prose | 55.4 | 59.8 (1.08x) | 66.2 (1.23x) |
| shell | 55.3 | 61.0 (1.10x) | 64.8 (1.17x) |

The DFlash run's own plain arm measured 54.6 / 54.0 / 55.5. llama.cpp on this file, the served A/B at depth and two
cards are not measured yet.

### Swift 1.5 and ThinkingCap on two cards (tensor parallel, own code end to end)

Measured 2026-11-11 on an RX 7900 XTX + RX 7900 XT (`caminus serve --cards 2`; today that serve drafts by default, so a rerun of these plain rows passes `--draft off`). Each card holds half of every layer's heads
and part of its feed-forward rows (by the cards' bandwidth: 53 % on the XTX), and the two cards add their partial results
128 times per token. The grid and task sizes are the one-card engines'; nothing is tuned for two cards yet.

| | Swift 1.5 Q4_K_M | ThinkingCap RCO 4.2 bpw |
|---|---|---|
| Decode, two cards (short context, the engine; 11-11, not rerun for 0.3.0: inputs lost) | 59.8 tok/s (even split) | 65.4 tok/s (16.73 -> 15.29 ms at the bandwidth split) |
| Decode, two cards (served, greedy; 11-17, 0.3.0) | 63.6 tok/s (63.2 on 0.3.0, same within 1 %) | 73.0 tok/s at `--max-ctx 32768`; 69.2 at the context the server picks (262,144) |
| Decode, one XTX (for comparison) | 42 tok/s | 46.9 tok/s |
| Prefill, 2,048 positions from empty (11-11, not rerun for 0.3.0: inputs lost) | 982 tok/s | 936 tok/s |
| Context the server picks by itself (11-17, 0.3.0) | 262,144 | (262,144 fits: 16.6 / 16.1 GB projected) |

- Accuracy against the float64 reference, 128 positions: mean KLD 2.5e-7 (Swift) and 2.2e-7 (ThinkingCap), top-1 128/128;
  llama.cpp's one-card decode 2.7e-4 (`tools/guard.sh LOG 1800 python -I tools/gates/q35_gmodel.py GGUF --ref REF.npz
  --llama LLAMA.bin --cards 2 --speed 64`, which also checks the replicated residual is the same bits on both cards after
  every token) (measured 11-11; not rerun for 0.3.0: inputs lost).
- Prefill 64 positions on both cards, then 64 two-card decode passes: mean KLD 2.4e-7 (Swift) and 4.5e-7 (ThinkingCap),
  top-1 64/64; llama.cpp's own prefill-then-decode 3.2e-4 / 4.5e-4. Chunk geometry and the prompt cache's stops change no bit
  (`tests/prefill/test_q35_prefill_tp.py GGUF --ref REF.npz --llama LL.bin --speed 2048`) (measured 11-11; not rerun for 0.3.0: inputs lost).
- The exchange alone: 128 exchanges of a 5,120-float partial cost 0.82 ms per token (6.7 us each) on this machine's link
  (`benchmarks/q35_tp_xchg.py`).
- The feed-forward split, ThinkingCap decode: 15.75 ms at 50 / 50, 15.29 ms at 53 / 47, 15.13 ms at 56 / 44.
- The prefill does not gain from the second card yet: its exchanges run between the layers' halves without overlap (a
  2,048-token chunk moves 2 x 64 x 2,048 x 5,120 floats each way).
- The served path, plain (`tools/guard.sh LOG 2400 python -I tools/serve_smoke1.py --model GGUF --scratch DIR --card 0,1
  --cards 2 --draft off --block-imports gguf,tokenizers`, measured before drafting was the two-card default; ThinkingCap
  with `--max-ctx 32768 --needle 28000 --restart`): ThinkingCap reads a
  6,760-token prompt at 975 tok/s and a 27,712-token one at 869 tok/s (the needle found), decode 63 tok/s at that depth;
  Swift reads 6,760 tokens at 1,064 tok/s (both rerun 11-17 on 0.3.0 with `--draft off`).
- llama.cpp on the same two cards (`llama-bench -dev ROCm0/ROCm1 -ngl 99 -fa 1 -p 2048 -n 128 -r 3`):

  | | Swift 1.5 decode / prefill | ThinkingCap decode / prefill |
  |---|---|---|
  | `-sm tensor` (each layer split across the cards, as here) | 42.1 / 798 tok/s | 44.6 / 924 tok/s |
  | `-sm tensor -ts 53/47` | 43.6 / 910 tok/s | 46.3 / 969 tok/s |
  | `-sm layer -ts 53/47` (half the layers on each card) | 33.7 / 1,366 tok/s | 36.5 / 1,508 tok/s |

  Tensor mode is the like-for-like comparison: decode here is 1.37x / 1.41x its best, prefill 1.08x / 0.97x. Layer mode
  reads prompts faster (no exchanges between the cards) and writes slower. `-sm row` does not run on this fork ("device
  ROCm0 does not support split buffers").

#### Speculative decoding on two cards (served)

`caminus serve --cards 2` with `--draft mtp` or `--draft dflash`: the two-card verify tape (`kernels/objects.tsv dec_q35_swift_tp_ver`,
`kernels/objects.tsv dec_q35_tcap_tp_ver`) on both cards, the one-card drafter and a full copy of the output head on one of them (MTP
on card 0; DFlash on the card with more room left). DFlash is the two-card default for both models. Measured 2026-11-15
on an RX 7900 XTX + RX 7900 XT with `tools/guard.sh LOG 3600 python -I benchmarks/q35_served_ab.py --model GGUF --ids IDS
--scratch DIR --card 0,1 --cards 2 --draft KIND --max-ctx 40960`: three interleaved rounds, 128 greedy tokens. Every
drafted round was token-identical to plain. Not rerun for 0.3.0 (inputs lost).

Public prompts of 9,661 and 34,444 tokens, built from this repository's source:

| Two cards (XTX + XT), served decode tok/s | off | MTP | x | DFlash | x |
|---|---|---|---|---|---|
| Swift 1.5, 10K | 60.7 | 88.9 | 1.46 | 82.2 | 1.36 |
| Swift 1.5, 34K | 50.3 | 65.8 | 1.31 | 72.1 | 1.44 |
| ThinkingCap, 10K | 65.6 | 83.7 | 1.28 | 74.5 | 1.13 |
| ThinkingCap, 34K | 53.7 | 59.5 | 1.11 | 63.7 | 1.18 |

(Each drafter's off column comes from its own interleaved run.)

Agent-style prompts, DFlash against off, same command (not rerun for 0.3.0: inputs lost): a coding-agent system prompt, 7 tool schemas and tool-call turns
whose results are this repository's source files, ending on a tool result; they render to 11,465 and 28,902 tokens.

| two cards | off | DFlash | x |
|---|---|---|---|
| Swift 1.5, agent 11.5K | 59.7 | 71.6 | 1.20 |
| Swift 1.5, agent 28.9K | 51.8 | 75.2 | 1.45 |
| ThinkingCap, agent 11.5K | 64.8 | 82.0 | 1.26 |
| ThinkingCap, agent 28.9K | 55.9 | 70.8 | 1.27 |

- The served end to end (`tools/serve_smoke1.py --cards 2 --draft off|mtp|dflash --block-imports gguf,tokenizers
  --restart`, per model): greedy and sampled replies byte-identical to `--draft off`; own code objects only; the restart
  served from the disk tier (measured 11-15; not rerun for 0.3.0: the byte comparison across arms needs a step outside the runner).
- Seen, not blockers: a server started with `--draft mtp` runs its plain passes slower than a `--draft off` server
  (ThinkingCap 15.0 vs 14.1 ms per pass greedy, 19.0 vs 14.5 sampled) while its policy has drafting switched off. The
  first drafted request after a load is slower than later ones (28.8 ms per pass, then 15-18).

### Qwen3-30B-A3B Q4_K_M

Measured 2026-11-10 on the RX 7900 XTX, the Caminus column rerun 2026-11-17 on 0.3.0 (006130e; numbers within 2 % kept); llama.cpp on the same card with `llama-bench -ngl 99 -fa 1`.
Every Qwen3-30B-A3B number on this page (this section, the two-card section and the yardstick row) was measured on Qwen's
official Q4_K_M file; the catalog now serves unsloth's Q4_K_M, made with an importance matrix, which has the same tensors
in the same formats ([which file](qwen3-30b-a3b.md#which-file)). The model runs on
Caminus's own kernels, so its accuracy gate is not llama.cpp but a floating-point reference of the model, with llama.cpp
measured against the same reference. [Qwen3-30B-A3B on Caminus's own code](qwen3-30b-a3b.md) has the method.

| What | Caminus | llama.cpp | Tool and conditions |
|---|---|---|---|
| Decode at depth 8 / 1,024 / 4,096 / 16,384 / 32,768 / 40,900 | **248 / 235 / 213 / 161 / 124 / 111 tok/s** | 148 / - / 134 / 110 / 90 / - tok/s | `benchmarks/q3moe_depth.py --tokens 32 --rounds 5`: one token per pass, medians; llama.cpp: tg64 at `-d` 0, 4096, 16384, 32768 |
| Prefill of 512 / 4,096 / 32,768 tokens | **5,166 / 6,432 / 2,791 tok/s** (rerun 11-17 on 0.3.0, with the Q4_K overflow guard: within 2 %) | 4,330 / 3,853 / 1,794 tok/s | `benchmarks/q3moe_prefill.py --chunk 2048 --rounds 3` from empty; llama.cpp: pp512, pp4096, pp32768 |
| Through the server | decode 240 tok/s short, 197 after 6,587 prompt tokens, 123 after 37,659; that prompt read at 2,254 tok/s, first token in 16.7 s | | `tools/serve_smoke1.py --max-ctx 40960 --needle 38000 --block-imports gguf,tokenizers`: one request each |
| Mean KLD against the reference, 128 tokens | **4.66e-04**, top-1 128 / 128 | 2.83e-03, 127 / 128 | `tools/gates/q3moe_gmodel.py` (float64 reference) |
| The same for the last 64 of 16,384 / 40,896 tokens | **8.80e-04**, 62 / 64; **6.64e-04**, 64 / 64 | 5.06e-03, 60 / 64; 5.73e-03, 63 / 64 | `tools/gates/q3moe_depth_gmodel.py` (fp32 reference; prefill, then decode) |

### Qwen3-30B-A3B on two cards

`caminus serve --cards 2` (the `qwen3moe-tp` entry) runs the same file on an RX 7900 XTX + RX 7900 XT; one card stays the
model's default. Each card holds 2 of the 4 kv heads with their query heads, and 512 (XTX) or 256 (XT) of each expert's
768 rows; the two cards add their partial results 96 times per token.

Served, 2026-11-17 on 0.3.0 (006130e): `benchmarks/served_cards_ab.py` (under `tools/guard.sh`), `caminus serve` on one XTX against the
XTX + XT, both `--draft off` and `--no-disk-cache`, own code (`--block-imports gguf,tokenizers`). Three rounds, each
round a fresh one-card server then a fresh two-card server; one prompt per depth built from this repository's own source,
read from empty (0 cached tokens on every request), then 128 greedy tokens. Medians of the three rounds (tok/s):

| Depth (prompt tokens) | 8 | 4,096 | 16,384 | 32,768 | 40,768 |
|---|---|---|---|---|---|
| Decode, two cards | 213 | 188 | 158 | 129 | 121 |
| Decode, one XTX | 245 | 211 | 159 | 122 | 109 |
| Decode, two / one | 0.87x | 0.89x | 1.00x | 1.06x | 1.12x |
| Prefill, two cards | 207 | 6,928 | 5,117 | 3,539 | 2,997 |
| Prefill, one XTX | 242 | 6,304 | 4,176 | 2,795 | 2,323 |
| Prefill, two / one | 0.85x | 1.10x | 1.23x | 1.27x | 1.29x |

- Every range sits within 5 % of its median. The 8-token prefill is one short pass, not a throughput figure.
- Two cards read every prompt from 4K tokens on faster, and decode faster from about 32K tokens on. Short turns decode
  faster on one card, which is why one card stays the default.
- The two-card engine is not bitwise to the one-card engine by design (the summation order differs). Its accuracy is
  gated against an fp32 reference at depth instead (`tools/gates/q3moe_depth_gmodel.py --cards 2`, the two-card
  prefill then decode, 2026-11-16; rerun 11-17 on 0.3.0: the same): mean KLD 1.02e-03 over the last 64 of 16,384 positions and 8.0e-04
  over the last 64 of 40,896, both below llama.cpp's on the same file (5.06e-03 and 5.73e-03).
- Engine level (no server), for reference: decode 216 / 189 / 161 / 132 / 125 tok/s at 8 / 4,096 / 16,384 / 32,768 /
  40,900 against one XTX's 249 / 213 / 161 / 124 / 111 (2026-11-17 on 0.3.0, `benchmarks/q3moe_depth.py --cards 2`); prefill
  from empty 7,041 against 6,503 tok/s at 4,096 and 3,556 against 2,788 at 32,768 (2026-11-16, rerun 11-17 on 0.3.0: within 2 %;
  `benchmarks/q3moe_prefill.py --cards 2`, interleaved, medians of three).

## Two-card yardstick: stock llama.cpp, tensor and layer split

Measured 2026-11-11 on an RX 7900 XTX + RX 7900 XT with stock llama.cpp (ggml-org master 7bcbc04, built for gfx1100):
`llama-bench -dev ROCm0/ROCm1 -sm tensor|layer -ts 53/47 -ngl 99 -fa 1 -p 2048 -n 128 -r 2`, decode (tg128) / prefill
(pp2048) in tok/s. `-sm tensor` splits each layer across the cards, as Caminus's two-card engines do; `-sm layer` gives
each card whole layers.

| | Caminus | llama.cpp `-sm tensor` | llama.cpp `-sm layer` |
|---|---|---|---|
| Swift 1.5 Q4_K_M | 59.8 / 982 | 43.5 / 859 | 33.5 / 1,400 |
| ThinkingCap RCO 4.2 bpw | 65.4 / 936 | 46.0 / 978 | 36.3 / 1,509 |
| Qwen3-Coder-Next UD-IQ3_S | 136.5 / ~135 (no batched prefill yet) | 72.5 / 2,046 (even split) | 80.8 / 3,625 |
| The Coder (IQ1_M) | 104; 123-203 with its drafter / ~2,800 | 39.6 / 1,336 | 40.5 / 2,328 |
| Qwen3-30B-A3B Q4_K_M | 213 served (216 engine level) / no 2,048-token row (4,096: 6,928 served) | 112.4 / 2,909 | 126.2 / 5,780 |

- Qwen3-30B-A3B's row was measured later, 2026-11-15, with the same build (7bcbc04) and command (`-ts 53/47 -fa 1`);
  its Caminus column is the served two-card section above (engine level in brackets). Both two-card modes decode below llama.cpp's
  one-card row above (148 tok/s, measured on the ignite-llama fork).
- Files larger than the host's page cache (Qwen3-Coder-Next, the Coder on a 29 GiB host) need `-lm dio`: with the
  default memory-mapped loading llama-bench stalls after loading the weights, both cards idle.
- Qwen3-Coder-Next with `-sm tensor -ts 53/47` stops on `GGML_ASSERT(split_state.ne[j] % div == 0)`; the even split runs.
- The ignite-llama fork (842653396) gives the same dense-model numbers within noise (`-sm tensor -ts 53/47`: 43.6 / 910
  and 46.3 / 969); `-sm row` does not run on it.

## Not measured

- Any other pair of GPUs, any other host, kernel or driver version.
- Real harness sessions (Qwen Code, Hermes Agent, Claude Code) against the server: the wrappers in packaging/harness follow
  each harness's documentation; `tools/serve_smoke.py` exercised the same request shapes.
- A head-to-head with other engines than llama.cpp.
