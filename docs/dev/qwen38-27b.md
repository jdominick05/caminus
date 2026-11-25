# The dense Qwen3.8-27B line on Caminus's own code

Swift 1.5 (Q4_K_M), ThinkingCap (RCO, 4.2 bits per weight) and the plain GSQ-RCO IQ3_S run on code written for this
project, the way [Qwen3-30B-A3B](qwen3-30b-a3b.md) does. They are dense models of GGUF architecture `qwen35`: 64
blocks, three Gated DeltaNet blocks then one gated-attention block, each followed by a SwiGLU feed-forward of width
17,408 on a residual stream of 5,120.

**Status: served.** One card each; Swift 1.5 and ThinkingCap also on two, with MTP and DFlash2 drafting.
`caminus pull swift-1.5|thinkingcap|qwen3.8-27b` fetches them (the DFlash2 drafter is shared). Speeds and gates:
[benchmarks](benchmarks.md) (the Swift 1.5, ThinkingCap and two-card sections); the models: [models](../models.md).
The rest of this page is the bring-up record of 2026-11-11 (Swift's decode module on one card), kept as measured then.

## The files

Files, sizes and licences: [models](../models.md). Read each licence before use: the two finetunes are not under an
open-source licence. The catalog (`python/caminus/models.py`) pins each file's revision and SHA-256.

## What Swift 1.5 Q4_K_M holds

The header says 65 blocks with `nextn_predict_layers = 1`: blocks 0 to 63 are the model, block 64 is a
multi-token-prediction block (a draft head, read only when drafting with it). The trunk's matrices by format:

| Format | Tensors | Share of the bytes one token reads |
|---|---|---|
| Q4_K | 257 | 65.1 % |
| Q6_K | 79 | 25.0 % |
| Q8_0 | 64 | 8.5 % |
| Q5_K | 2 | 0.7 % |
| F32 (norms, the DeltaNet's small tensors) | 456 | 0.6 % |

The multi-token-prediction block is Q4_0. One decode token reads 16.48 GB of weights (every trunk matrix once; the
embedding is one row). At the 858 GB/s the RX 7900 XTX's memory delivers to this kind of read
([rdna3-decode-floor](https://github.com/jdominick05/rdna3-decode-floor)), that is 19.2 ms: **52 tokens per second is
the bound from the weights alone**, before attention over the context.

## What runs on own code today

- **The reader:** `caminus.ggufio`, as for every model.
- **The tokenizer:** `caminus.bpe` reads the `qwen35` pre-tokenizer. It differs from `qwen2` in two character classes:
  a run of letters takes combining marks with it, and a run of symbols stops at one.
- **The decoders:** `caminus.own_dequant` decodes Q4_K, Q5_K, Q6_K, Q8_0 and Q4_0 (every format in the file).
- **The reference:** `caminus.ref_q35` is the forward in float64, every quantized weight decoded exactly. It works
  layer by layer over a chunk of tokens, so each matrix is decoded once: 128 positions take 71 s on a 16-thread CPU.
  `tools/reference/q35_ref.py` writes its logits for a fixed public text.

```bash
python -I -m caminus.ref_q35 MODEL.gguf "The capital of France is" --new 2
python -I tools/reference/q35_ref.py MODEL.gguf OUT.npz 128 --census
```

## Checks so far

| Check | Result |
|---|---|
| Tokenizer against the Hugging Face tokenizer of the model (`tests/io/test_bpe.py`) | the same ids on 1,815 texts, including 42 stress texts (19 with combining marks) |
| Decoders against gguf-py (`tests/io/test_own_dequant.py`) | bitwise on 95 tensors, 84 M values, every format |
| Decoders against a per-value definition (`tests/io/test_own_formats.py`, in CI) | bitwise |
| The reference on a tiny synthetic model (`tests/test_ref_q35.py`, in CI) | chunked = token by token; a DeltaNet block and an attention block equal a per-head definition to 3e-15 |
| The reference on Swift 1.5 | "The capital of France is" continues " Paris." (p 0.68, then 0.77); on 128 tokens of plain text the top token is the text's next token 74 times of 127 |

The reference's two block tests use definitions written from the same reading of the model as the reference itself.
The comparison that can catch a wrong reading is its logits against llama.cpp's on the real file (one-token decodes,
128 positions, RX 7900 XT): mean KLD 2.7e-04 (max 1.5e-03), the same top token at 127 of 128 positions. llama.cpp
computes in fp16 / fp32 and the reference in float64, so this is the distance expected between a correct reading and a
working engine; a wrong reading of a block would not land there. The reference is the truth of the gates below.

## The decode module

`kernels/dec_q35_swift.hip` is the persistent decode kernel with the 16 ops Swift's tape runs and no llama.cpp header:
the matvec tiles of the Qwen3-30B-A3B line for Q4_K and Q6_K, a plain rows tile for Q8_0 and Q5_K (9 % of the bytes),
the embedding row, the add-and-norm for a 5,120-wide residual, and the K3 engine's own Gated DeltaNet and
gated-attention ops with fp16 outputs. `engine/q35_dense.py` (`Dense35Own`) builds the tape: one token per pass, an f16
KV cache, the multi-token-prediction block left unread. Decode attention is the split pair of `kernels/ops_attn_split.h`
(flash decoding, the AREX-2 module's settings: 32-cell splits, one read of a split's K / V rows for its kv head's 6 query
heads); until that change it ran one workgroup per query head over every position, 26-31x the bandwidth floor at every
depth (docs/dev/yardsticks.md). A feed-forward whose gate and up matrices share Q4_K or Q6_K
runs them as one segment (47 of 64 blocks); the others run two matvecs and a SwiGLU segment.

| Check | Result |
|---|---|
| Register budget of `k3_run` (`llvm-readelf`) | 88 of 96 VGPRs, 0 spills |
| The Qwen3-30B-A3B modules after the shared-header edit | machine code byte-identical (`dec_q3m`, `dec_q3m_ver`, `dec_q3m_iq`, `dec_q3m_rs4`, `dec_q3m_rs2`, `dec_q3m_gg`) |
| The tape on a tiny synthetic quantized model, without a card (`tests/engine/test_q35_tape_cpu.py`, in CI) | every tile present, only ops the module dispatches, the host's op list equal to the header's |
| Tiny synthetic model on a card (`tests/engine/test_q35_dense.py`; the same figures on the RX 7900 XT and the RX 7900 XTX) | 26 matvecs within 4e-08 of float64 (relative to sum \|w x\|); residuals per block within 5e-03; logits mean KLD 6.5e-06; a second run bitwise; grids 3, 6, 7 bitwise the default |
| Swift 1.5 on a card, 128 positions against the float64 reference (same test; the same figures on both cards) | mean KLD 2.2e-07 (max 2.1e-06); the same top token at 128 of 128; residuals after blocks 0 and 3 within 2.2e-04; embedding rows bitwise the host decoder; no wait gave up; grids 3, 6, 7 bitwise the default |
| G-model against llama.cpp (`tools/gates/q35_gmodel.py`, each card) | PASS: mean KLD 2.2e-07 against llama.cpp's 2.7e-04; top token 128 of 128 against 127 of 128 |

Nothing is tuned: the grid is the Qwen3-30B-A3B engine's per-card choice and the task sizes are that engine's. For
orientation only, at bring-up (before tuning), at under 130 positions of context, every step synchronized by the test, next to llama.cpp's
`llama-bench` on the same file and card:

| Card | This module | llama.cpp |
|---|---|---|
| RX 7900 XT | 27.5 ms per token (36 tok/s) | 31.8 tok/s |
| RX 7900 XTX | 23.8 ms per token (42 tok/s) | 37.1 tok/s |

The RX 7900 XT's `llama-bench` figure is from an earlier run the same day (same build and file); the other three
are from one run. One token reads 16.48 GB of weights. Depth curves and drafted speeds: [benchmarks](benchmarks.md).

### Activation range

The decode kernels of the Qwen3-30B-A3B line read their activations as fp16. Whether that holds here depends on how
large the inputs of this model's matrices get. Over the same 128 positions (`--census`):

| Input of | Largest magnitude |
|---|---|
| a block's matrices (after `attn_norm`) | 104 |
| the feed-forward's gate and up (after `post_attention_norm`) | 55 |
| the feed-forward's down (17,408 values) | 274 |
| the DeltaNet's output matrix | 193 |
| the attention's output matrix | 114 |
| the output head | 37 |
| the residual stream itself (kept in fp32) | 577 |

fp16 holds values up to 65,504, so range is not a concern on this text. It is one text and 128 positions; the model
gates measure what fp16's 11 bits of precision cost.
