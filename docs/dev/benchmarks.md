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

**The comparison:** llama.cpp at the fork the gates use (ignite-llama, commit 842653396, published as e4e3f8eda on
[github.com/jdominick05/ignite-llama](https://github.com/jdominick05/ignite-llama) with identical `ggml/`, `common/` and
`src/`: docs/dev/building.md; a HIP build with the same TheRock toolchain), on the same file and the same two cards: llama-server with `-sm layer -ts 550,450`, flash attention on, a q8_0 KV
cache, DFlash with `--spec-draft-n-max 3`; llama-bench for prompt processing. llama.cpp's numbers were measured on 2026-11-01
unless a row says otherwise.

## Decode

| What | Caminus | llama.cpp | Date | Tool and conditions |
|---|---|---|---|---|
| Plain decode, short context | **9.87 ms/token** (101 tok/s) | | 11-03 | `benchmarks/fn_bench.py 96 $CODER --tp`: 96 greedy tokens, the two-card engine, one token per pass |
| Plain decode in the generation loop | **104.6 / 104.9 / 103.9 tok/s** (prose / code / shell) | **44.2 / 44.2 / 38.0 tok/s** | 11-06 | `tools/k3_smoke.py --model coder`: 128 greedy tokens after a 200-token prompt, per text; llama.cpp: 256 tokens through llama-server, the same prompts (its shell run is a low outlier: 43.8 in another repetition) |
| Decode with DFlash | **156.0 / 202.5 / 123.3 tok/s** | **92.4 / 89.1 / 65.1 tok/s** | 11-06 | the same smoke run: the DFlash2 drafter inside the engine, 4-token verify passes; the output equals plain greedy decoding (checked in the run) |
| The same smoke on the release package | plain 103.8-105.1, DFlash 156.0 / 202.8 / 123.2 tok/s | | 11-06 | `tools/k3_smoke.py` against the installed wheel (`CAMINUS_SMOKE_INSTALLED=1`) |
| Plain decode at depth | 10.4 / 10.9 / 11.6 / 12.8 / 15.3 ms/token at 2K / 32K / 64K / 128K / 262K | ~30 / ~38 / ~60 ms/token at 64K / 128K / 262K | 11-05 | `tools/eval/sweep_262k.py plain` and `curves`: 128 tokens after a prompt of that depth; llama.cpp: its one-token decodes in the sweep's reference jobs (docs/specs/sweep-262k.md, section 5) |
| Plain decode at depth, parallel pool selection | 10.69 -> 10.57 / 11.17 -> 10.72 / 15.58 -> 11.39 ms/token at 2,048 / 32,768 / 261,632 (before -> after) | | 11-08 | the change that made the indexer's pool selection parallel (OP_QSA_PSEL): the sweep's plain-stage timing (tokens 16 .. 143 of 144 greedy tokens after a batched prefill of that depth, one text), old and new interleaved, medians of 3; the same greedy tokens. The row above predates it; the sweep has not been rerun |
| Decode at 32K before / after split attention | 25.00 -> 11.17 ms/token | 25.8 ms/token | 11-03 | one-token decode passes at that depth without a fill, measured in the split-attention step |

How many drafted tokens are accepted depends on the exact greedy text. A change of the attention's summation order (split
attention, 11-03) shifted near-tied tokens and with them the acceptance, without any change to the drafter. At ~262K DFlash
costs more than plain decoding on prose (18.25 vs 15.26 ms/token), so the server drafts only when its rolling measurements say
it pays (`serve/draftpolicy.py`).

## Accuracy

| What | Result | Bar | Date | Tool |
|---|---|---|---|---|
| End to end against llama.cpp, short context (the D-281 gate) | mean KLD **0.0057**, top-1 **96.2 %** over 768 positions (3 texts x 256) | llama.cpp's own batch-vs-decode floor: KLD 0.0076, top-1 95.9 % | 11-03 | `tests/engine/test_flashnext_e2e.py` against llama.cpp's one-token decode dumps of the same tokens (teacher forced) |
| The smoke's quick check | mean KLD 0.0044 against a floor of 0.0063 (64 positions x 3 texts) | the floor on the same positions | 11-06 | `tools/k3_smoke.py` on the release package (a subset of the gate above, recomputed each run) |
| Long context with the QSA indexer | mean KLD 0.0021 / 0.00052, top-1 99.6 / 100 % at 8K / 32K | the long-context floors | 11-03 | `tools/eval/lc_baseline.py` on `tools/eval/lc_probes.py`'s contexts; copy probes answered as llama.cpp answers them |
| The batched prefill against llama.cpp's prefill-then-decode (G4b) | 0.00092 / 0.00127 / 0.00321 / 0.00020 at 65,536 / 81,920 / 131,072 / 261,632 | 0.0076 / 95.9 % | 11-05 | `tools/eval/sweep_262k.py k3 ... g4` and `cmp`; llama.cpp at ub 2048 to 81,920, ub 512 beyond (its memory) |
| Needle and rare-string recall, 64K to ~262K | **17 / 18** probes | llama.cpp **16 / 18** on the same probes | 11-05 | `tools/eval/sweep_262k.py k3 ... probes`; no probe llama.cpp passes fails here (docs/specs/sweep-262k.md, section 3) |
| A 262K soak | one policed run of 65,472 four-token passes from 0 to 261,888, no unexplained bail | | 11-05 | `tools/eval/sweep_262k.py k3 ... soak` |

The engine's weight-format and BF16 kernels match llama.cpp's bit for bit (tests/engine/test_k3_gguf_gsq.py, tests/engine/test_k3_bf16_mv.py
and the golden-layer tests); most speed changes kept the previous build's logits bitwise across grid sizes and 4-token verify
passes (tools/gates). Greedy text can still split from llama.cpp's at near-tied tokens, as llama.cpp's own runs split from each
other.

## Reading prompts (the batched prefill)

| What | Caminus | llama.cpp | Date | Tool and conditions |
|---|---|---|---|---|
| From empty, including the hand-off to decode | **2,909 / 2,873 / 2,806 tok/s** at 4K / 8K / 32K | llama-bench pp4096 2,628, pp8192 2,884-2,901, pp32768 2,791 | 11-04 | `tests/prefill/test_prefill_s6.py $CODER speed7 --max-ctx 33024` (TTFT medians); llama-bench at its best ubatch |
| Deep prompts | 2,775 / 2,715 / 2,576 / 2,393 tok/s at 8K / 64K / 128K / 262K | 1,890 / 1,433 / ~950 tok/s at 64K / 128K / 262K (ub 512, what fits) | 11-05 | `tools/eval/sweep_262k.py k3 ... curves`; llama.cpp: its probe fills in the sweep |
| Through the server | 11,285 tokens in 4.4 s (2,557 tok/s); 101,525 tokens in 40.1 s (2,538 tok/s); 2,537 tokens: first token in 1.2 s | | 11-06 | `tools/serve_smoke.py --stages server`: the real `caminus serve` at its default configuration, synthetic prompts |

The prefill misses its own target of 3,000 tok/s at 8K (about 1 % under llama.cpp's best there).

## Serving

All from `tools/serve_smoke.py` on 11-06 (the real server at its default 262,144-token configuration, started twice; scratch
cache directories; synthetic conversations):

| What | Measured |
|---|---|
| Start to ready | 49-51 s (the engine build 41 s) |
| VRAM | XTX 23,071 MiB at ready, 23,091 at the peak (its guard line 23,800, with a 512 MiB reserve kept); XT 19,019 / 19,037 (line 19,850) |
| The next turn of an 11K-token conversation | first token in 0.21-0.23 s (its state live in VRAM) |
| After a restart | the conversation back from the disk tier, first token in 0.21 s at 11.6K tokens; the stop's park took 0.65 s (1.8 GiB), 1.45 s at 101K (3.9 GiB) |
| Decode through the server | 5.7-5.9 ms/token on tool-call code with DFlash at 0.6K and 32K (most drafts accepted); prose and thinking text accept fewer drafts and run slower |
| Server overhead | the SSE chat route + tool parser over the engine alone: +0.73 % (OpenAI) / +1.19 % (Anthropic) at 0.6K, +0.26 % / +0.22 % at 32K |

## The other models

Measured 2026-11-07 on one card each (the RX 7900 XTX unless noted) or on both cards for Qwen3-Coder-Next; llama.cpp on
the same card(s) with llama-bench (decode tg128, prompt pp2048). Speeds are decode (writing) or prefill (reading a
prompt), in tok/s.

| Model | Caminus decode | llama.cpp decode | Caminus prefill 2K | llama.cpp prefill 2K | Accuracy gate |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B GSQ 2-bit | 174.9 (XT alone: 148.5) | 106.8 (XT: 90.9) | 5,227 | 3,239 | within llama.cpp's floor: prefill-then-decode G4 0.00077 / 0.00072 |
| Qwen3.6-35B-A3B HauhauCS Q3_K_P | 192.5; 140.7 at 32K | 122.1; 107.7 at 32K | 4,136 | 2,923 | G4 0.00171 / 0.00181 |
| AREX-2 Q4_K_M (dense 27B) | 41.0 | 36.3 | 928 | 910 | short-context KLD 0.0003, top-1 99.1 % (floor 0.0033 / 97.9 %); G4 0.00053 / 0.00048 |
| Qwen3-Coder-Next UD-IQ3_S (both cards) | 136.5; 67.9 at 32K | 81.8; 72.6 at 32K | ~135 (one token per pass) | 2,787 | bitwise gates on the engine; no batched prefill yet |

Qwen3-Coder-Next decodes slower than llama.cpp at 32K and reads prompts about 20 times slower: its batched prefill
isn't written yet. Tools: `tests/prefill/test_q36_prefill.py` (ops, handoff, e2e, speed) and `tools/serve_smoke1.py`.

### Swift 1.5 Q4_K_M (dense 27B, own code)

Measured 2026-11-11; llama.cpp with `llama-bench -fa 1` on the same card. Untuned: decode short context only.

| | RX 7900 XTX | RX 7900 XT |
|---|---|---|
| Caminus decode (short context) | 42 tok/s (23.8 ms) | 36 tok/s (27.5 ms) |
| llama.cpp decode (tg128) | 37.1 | 31.8 |
| Caminus prefill, 2,048 positions from empty | — | 771 (chunk 2048), 760 (chunk 1024) |
| llama.cpp prefill (pp2048 / pp512) | — | 763 / 770 |

- Accuracy: prefill 64 positions, then 64 decode passes against the float64 reference: mean KLD 3.0e-7, top-1 64/64.
  llama.cpp's own prefill-then-decode on the same rows: 3.2e-4, 64/64 (`tools/guard.sh LOG 1800 python -I tests/prefill/test_q35_prefill.py GGUF --ref REF.npz --llama LL.bin`).
- The served path on the XTX (`tools/guard.sh LOG 2400 python -I tools/serve_smoke1.py --model GGUF --scratch DIR --card 0
  --max-ctx 32768 --needle 28000 --restart --block-imports gguf,tokenizers`):
  - greedy decode 44 tok/s;
  - a 6,760-token prompt at 748 tok/s, then decode at 28 tok/s;
  - a 27,712-token prompt at 335 tok/s, then decode at 15 tok/s.

  The attention reads every earlier position per query head, so both the prompt rate and decode fall with depth.
- Swift 1.5 needs a 24 GB card. With its prompt cache it projects 19.1 GiB at 12K, past an XT's room.

#### Verify passes (the base for speculative decoding; no drafter yet)

`engine/q35_dense_verify.py` runs T tokens in one pass on the decode engine's weights and states (module
`k3_q35_swift_ver`). Measured 2026-11-11 after an own prefill of real text: ms per pass against one decode step at the
same depth, with the pass committing its own accepted tokens as it would in use (the benchmark's walk: medians of 9
interleaved rounds; `tools/guard.sh LOG 1800 python -I benchmarks/q35_verify.py GGUF --depths 512,4096 --per-wgp 5,6,7`):

| card | depth | decode step | T = 2 | T = 3 | T = 4 |
|---|---|---|---|---|---|
| RX 7900 XTX | 512 | 24.5 ms | 29.7 (1.21x) | 38.4 (1.57x) | 47.2 (1.92x) |
| RX 7900 XTX | 4,096 | 31.7 ms | 36.4 (1.15x) | 44.0 (1.39x) | 53.9 (1.70x) |
| RX 7900 XT | 512 | 28.6 ms | 34.2 (1.20x) | 45.0 (1.57x) | 56.7 (1.98x) |
| RX 7900 XT | 4,096 | 36.3 ms | 42.0 (1.16x) | 53.6 (1.48x) | 65.1 (1.79x) |

- A pass of T tokens pays for itself when the drafter's tokens are accepted often enough to average more than that ratio
  in output tokens per pass (plus the drafter's own time).
- The commit's replay (the state taking the accepted tokens first) costs little: the same passes at commit 0 are at most
  1.3 ms (2.4 %) faster. The grid makes little difference: at T = 4, 5 to 7 workgroups per WGP are within 2.1 ms of the decode
  engine's grid (XTX at 4,096: 6 per WGP 51.9 ms against 53.9).
- The gate (`tools/guard.sh LOG 1800 python -I tests/engine/test_q35_verify.py GGUF --depths 0,512`): for T = 1 .. 4 a
  walk of passes with rejected rows holding wrong tokens, decode steps between them, the lazily committed states handed
  both ways: every accepted row and every decode step bitwise the decode steps; passes of different T on one state (a pass
  of 2 committing 3 of a pass of 4) bitwise; a commit past the last pass's tokens refused on the host; T = 4 at 3 / 6 / 7
  workgroups per WGP bitwise. On both cards. Forced bails in the verify tape: `tools/guard.sh LOG 1200 python -I tools/gates/forcebail.py 4 40 tiny
  --engine dense35ver --model GGUF --ctx 512 --tokens FILE`.

### ThinkingCap RCO 4.2 bpw (dense 27B, own code end to end)

Measured 2026-11-11; llama.cpp with `llama-bench -fa 1` on the same card. The grid and task sizes are untuned. Caminus's figure is the median of positions 64 .. 127 with each step synchronized
(`tests/engine/test_q35_dense.py --model GGUF --ref REF.npz`).

| | RX 7900 XTX | RX 7900 XT |
|---|---|---|
| Caminus decode (short context) | 46.9 tok/s (21.33 ms) | 39.4 tok/s (25.37 ms) |
| llama.cpp decode (tg128) | 40.5 | 34.1 |
| Caminus prefill, 2,048 positions from empty | 942 (chunk 2048), 909 (chunk 1024) | — |
| llama.cpp prefill (pp2048 / pp512) | 1,043 / 1,117 | — |

- Accuracy, against the float64 reference (`tools/reference/q35_ref.py`), 128 positions: mean KLD 2.4e-7, top-1 128/128.
  llama.cpp's one-token decode of the same rows: 2.7e-4, 128/128
  (`tools/guard.sh LOG 1800 python -I tools/gates/q35_gmodel.py GGUF --ref REF.npz --llama LLAMA.bin`).
- Per-matvec error against float64 on real rows: 1.0-1.9e-4 mean on every shape and format, against llama.cpp's 6.2-9.6e-3
  (`tools/guard.sh LOG 1800 python -I benchmarks/own_mv_census.py GGUF`).
- Prefill 64 positions, then 64 decode passes against the float64 reference: mean KLD 2.6e-7, top-1 64/64. llama.cpp's
  own prefill-then-decode of the same rows: 4.5e-4, 64/64 (`tools/guard.sh LOG 1800 python -I
  tests/prefill/test_q35_prefill.py GGUF --ref REF.npz --llama LL.bin --speed 2048`, which also prints the prefill speed
  above).
- The served path (`tools/guard.sh LOG 2400 python -I tools/serve_smoke1.py --model GGUF --scratch DIR --card N
  --block-imports gguf,tokenizers`; the XTX with `--max-ctx 32768 --needle 28000 --restart`):
  - XTX: greedy decode 49.5 tok/s; a 6,760-token prompt at 707 tok/s, then decode at 30 tok/s; a 27,712-token prompt at
    326 tok/s (the needle found), then decode at 16 tok/s;
  - XT: the server picks 53,248 tokens of context by itself; greedy decode 40.9 tok/s; a 6,760-token prompt at 532 tok/s.

### Swift 1.5 and ThinkingCap on two cards (tensor parallel, own code end to end)

Measured 2026-11-11 on an RX 7900 XTX + RX 7900 XT (`caminus serve --cards 2`). Each card holds half of every layer's heads
and part of its feed-forward rows (by the cards' bandwidth: 53 % on the XTX), and the two cards add their partial results
128 times per token. The grid and task sizes are the one-card engines'; nothing is tuned for two cards yet.

| | Swift 1.5 Q4_K_M | ThinkingCap RCO 4.2 bpw |
|---|---|---|
| Decode, two cards (short context, the engine) | 59.8 tok/s (even split) | 65.4 tok/s (16.73 -> 15.29 ms at the bandwidth split) |
| Decode, two cards (served, greedy) | 63.6 tok/s | 67.8 tok/s |
| Decode, one XTX (for comparison) | 42 tok/s | 46.9 tok/s |
| Prefill, 2,048 positions from empty | 982 tok/s | 936 tok/s |
| Context the server picks by itself | 262,144 | (262,144 fits: 16.6 / 16.1 GB projected) |

- Accuracy against the float64 reference, 128 positions: mean KLD 2.5e-7 (Swift) and 2.2e-7 (ThinkingCap), top-1 128/128;
  llama.cpp's one-card decode 2.7e-4 (`tools/guard.sh LOG 1800 python -I tools/gates/q35_gmodel.py GGUF --ref REF.npz
  --llama LLAMA.bin --cards 2 --speed 64`, which also checks the replicated residual is the same bits on both cards after
  every token).
- Prefill 64 positions on both cards, then 64 two-card decode passes: mean KLD 2.4e-7 (Swift) and 4.5e-7 (ThinkingCap),
  top-1 64/64; llama.cpp's own prefill-then-decode 3.2e-4 / 4.5e-4. Chunk geometry and the prompt cache's stops change no bit
  (`tests/prefill/test_q35_prefill_tp.py GGUF --ref REF.npz --llama LL.bin --speed 2048`).
- The exchange alone: 128 exchanges of a 5,120-float partial cost 0.82 ms per token (6.7 us each) on this machine's link
  (`benchmarks/q35_tp_xchg.py`).
- The feed-forward split, ThinkingCap decode: 15.75 ms at 50 / 50, 15.29 ms at 53 / 47, 15.13 ms at 56 / 44.
- The prefill does not gain from the second card yet: its exchanges run between the layers' halves without overlap (a
  2,048-token chunk moves 2 x 64 x 2,048 x 5,120 floats each way).
- The served path (`tools/guard.sh LOG 2400 python -I tools/serve_smoke1.py --model GGUF --scratch DIR --card 0,1 --cards 2
  --block-imports gguf,tokenizers`; ThinkingCap with `--max-ctx 32768 --needle 28000 --restart`): ThinkingCap reads a
  6,760-token prompt at 759 tok/s and a 27,712-token one at 403 tok/s (the needle found), decode 15 tok/s at that depth;
  Swift reads 6,760 tokens at 790 tok/s.
- llama.cpp on the same two cards (`llama-bench -dev ROCm0/ROCm1 -ngl 99 -fa 1 -p 2048 -n 128 -r 3`):

  | | Swift 1.5 decode / prefill | ThinkingCap decode / prefill |
  |---|---|---|
  | `-sm tensor` (each layer split across the cards, as here) | 42.1 / 798 tok/s | 44.6 / 924 tok/s |
  | `-sm tensor -ts 53/47` | 43.6 / 910 tok/s | 46.3 / 969 tok/s |
  | `-sm layer -ts 53/47` (half the layers on each card) | 33.7 / 1,366 tok/s | 36.5 / 1,508 tok/s |

  Tensor mode is the like-for-like comparison: decode here is 1.37x / 1.41x its best, prefill 1.08x / 0.97x. Layer mode
  reads prompts faster (no exchanges between the cards) and writes slower. `-sm row` does not run on this fork ("device
  ROCm0 does not support split buffers").

### Qwen3-30B-A3B Q4_K_M

Measured 2026-11-10 on the RX 7900 XTX; llama.cpp on the same card with `llama-bench -ngl 99 -fa 1`. The model runs on
Caminus's own kernels, so its accuracy gate is not llama.cpp but a floating-point reference of the model, with llama.cpp
measured against the same reference. [Qwen3-30B-A3B on Caminus's own code](qwen3-30b-a3b.md) has the method.

| What | Caminus | llama.cpp | Tool and conditions |
|---|---|---|---|
| Decode at depth 8 / 1,024 / 4,096 / 16,384 / 32,768 / 40,900 | **248 / 235 / 213 / 161 / 124 / 111 tok/s** | 148 / - / 134 / 110 / 90 / - tok/s | `benchmarks/q3moe_depth.py --tokens 32 --rounds 5`: one token per pass, medians; llama.cpp: tg64 at `-d` 0, 4096, 16384, 32768 |
| Prefill of 512 / 4,096 / 32,768 tokens | **5,166 / 6,432 / 2,791 tok/s** (32,768: before the Q4_K overflow guard, which costs 0.9 % at 512 and 4,096) | 4,330 / 3,853 / 1,794 tok/s | `benchmarks/q3moe_prefill.py --chunk 2048 --rounds 3` from empty; llama.cpp: pp512, pp4096, pp32768 |
| Through the server | decode 238 tok/s short, 195 after 6,587 prompt tokens, 123 after 37,659; that prompt read at 2,473 tok/s, first token in 15.3 s | | `tools/serve_smoke1.py --max-ctx 40960 --needle 38000 --block-imports gguf,tokenizers`: one request each |
| Mean KLD against the reference, 128 tokens | **4.66e-04**, top-1 128 / 128 | 2.83e-03, 127 / 128 | `tools/gates/q3moe_gmodel.py` (float64 reference) |
| The same for the last 64 of 16,384 / 40,896 tokens | **8.80e-04**, 62 / 64; **6.64e-04**, 64 / 64 | 5.06e-03, 60 / 64; 5.73e-03, 63 / 64 | `tools/gates/q3moe_depth_gmodel.py` (fp32 reference; prefill, then decode) |

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
