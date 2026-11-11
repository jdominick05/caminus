# The dense Qwen3.8-27B line on Caminus's own code (in progress)

Two finetunes of Qwen3.8-27B are being brought up on code written for this project, the way
[Qwen3-30B-A3B](qwen3-30b-a3b.md) was: Swift 1.5 (Q4_K_M) first, then ThinkingCap (RCO, 4.2 bits per weight). Both are
dense models of GGUF architecture `qwen35`: 64 blocks, three Gated DeltaNet blocks then one gated-attention block,
each followed by a SwiGLU feed-forward of width 17,408 on a residual stream of 5,120.

**Status: the host side only.** Nothing here runs on a card yet and `caminus serve` does not serve these files. This
page is updated as each part lands.

| Part | State |
|---|---|
| Files in the catalog (`python/caminus/models.py`, files only) | done |
| Reading the GGUF, the tokenizer, the host decoders | done, on own code |
| float64 reference of the forward (`caminus.ref_q35`) | done; not yet compared with llama.cpp |
| Decode kernels and engine, prefill, serving, two cards, drafting | not started |

## The files

| Model | File | Size | Licence |
|---|---|---|---|
| Swift 1.5 | `ukisai/Swift-1.5-Qwen3.8-27B-GGUF`, Q4_K_M | 17.4 GB | Swift Open License 1.0 |
| ThinkingCap | `bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF-RCO-4.2bpw` | 14.3 GB | PolyForm Small Business 1.0.0 |
| DFlash2 drafter of the base model | `incoai/Qwen3.8-27B-DFlash2-GGUF`, Q8_0 | 2.1 GB | Apache-2.0 |

Read each licence before use: the two finetunes are not under an open-source licence. The catalog pins each file's
revision and SHA-256. No named model downloads them yet.

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
The comparison that can catch a wrong reading, its logits against llama.cpp's on the real file, needs a card and has
not run. Until it has, the reference is not the truth of any gate.

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
