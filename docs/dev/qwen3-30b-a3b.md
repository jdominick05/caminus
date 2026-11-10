# Qwen3-30B-A3B on Caminus's own code

Caminus serves the official Qwen3-30B-A3B Q4_K_M GGUF on one RX 7900 XTX with code written for this project at every
step, from reading the file to the output token. No llama.cpp code runs or is compiled into that path. This page says
what "own code" covers, how fast and how accurate it is, where the remaining time goes, and how to reproduce every
number.

It is the follow-up to [rdna3-decode-floor](https://github.com/jdominick05/rdna3-decode-floor), which measured what
the matrix-vector products of one decode token of this model cost on the same card. That study timed the token's 1,393 matvecs
in isolation. This is the whole model: attention over the context, the tokenizer, the sampler and the HTTP server.

## Results

All on one RX 7900 XTX (24 GB, driving the monitors), clocks held at DPM `high`, 2026-11-10. llama.cpp is the
[ignite-llama](https://github.com/jdominick05/ignite-llama) fork at e4e3f8eda, a HIP build with the same toolchain,
on the same file and card (`llama-bench -ngl 99 -fa 1`).

### Decode (how fast it writes)

| Context depth | Caminus | llama.cpp | Bandwidth bound | Share of the bound |
|---|---|---|---|---|
| 8 | **231 tok/s** (4.32 ms) | 148 tok/s | 447 tok/s | 52 % |
| 1,024 | **219 tok/s** (4.57 ms) | | 425 tok/s | 52 % |
| 4,096 | **187 tok/s** (5.34 ms) | 134 tok/s | 369 tok/s | 51 % |
| 16,384 | **144 tok/s** (6.93 ms) | 110 tok/s | 243 tok/s | 59 % |
| 32,768 | **113 tok/s** (8.84 ms) | 90 tok/s | 167 tok/s | 68 % |
| 40,900 | **104 tok/s** (9.63 ms) | | 144 tok/s | 72 % |

- **Caminus:** `benchmarks/q3moe_depth.py --depths 8,1024,4096,16384,32768,40900 --tokens 32 --rounds 5`. One token per
  pass, the median of 5 rounds of 32 tokens at each depth; the rounds differ by at most 0.03 ms.
- **llama.cpp:** `tg64` at depths 0, 4,096, 16,384 and 32,768 (`-d`), 2 repetitions.
- **Bandwidth bound:** the bytes one token must read, divided by the 858 GB/s this card sustained in the decode-floor
  study. A token reads 1.92 GB of weights (every dense matrix, 8 of the 128 experts per layer, the output head) plus
  98,304 bytes of KV cache per position of context. Nothing that reads every byte once per token can beat it.

Through the HTTP server (`tools/serve_smoke1.py`, one request each): 225 tok/s on a short chat, 182 tok/s after a
6,587-token prompt, 113 tok/s after a 37,659-token prompt.

### Prompt reading (the batched prefill)

| Prompt length | Caminus | llama.cpp |
|---|---|---|
| 512 | **5,230 tok/s** | 4,330 tok/s |
| 4,096 | **6,493 tok/s** | 3,853 tok/s |
| 32,768 | **2,791 tok/s** | 1,794 tok/s |

`benchmarks/q3moe_prefill.py --lengths 512,4096,32768 --chunk 2048 --rounds 3` from an empty context; llama.cpp:
`llama-bench` `pp512`, `pp4096`, `pp32768`. Through the server, a 37,659-token prompt takes 15.3 s to the first token.

### Accuracy

The reference is the same model computed in floating point with exactly decoded weights: float64 on the CPU at short
context (`tools/reference/q3moe_ref.py`), fp32 on a card at depth (`tools/reference/q3moe_ref32.py`). Both engines are
compared with it on the same tokens.

| Where | Caminus: mean KLD, top-1 | llama.cpp: mean KLD, top-1 | Tool |
|---|---|---|---|
| 128 tokens, decode | 4.66e-04, 128 / 128 | 2.83e-03, 127 / 128 | `tools/gates/q3moe_gmodel.py` |
| The last 64 of 16,384 (prefill, then decode) | 9.75e-04, 62 / 64 | 5.16e-03, 60 / 64 | `tools/gates/q3moe_depth_gmodel.py` |
| The last 64 of 40,896 (prefill, then decode) | 9.17e-04, 64 / 64 | 5.35e-03, 63 / 64 | `tools/gates/q3moe_depth_gmodel.py` |

Caminus is 5 to 6 times closer to the reference than llama.cpp. The reason is the activation precision: these kernels
multiply the weights by fp16 activations, llama.cpp's by int8.

How exact the reference is: at 128 tokens the fp32 reference differs from the float64 one by a mean KLD of 6.5e-11. At
depth an fp32 reference is less sharp. Two fp32 evaluations whose attention products have different shapes (so their
sums run in a different order) differ by 7e-05 at 16,384 tokens and 1.8e-04 at 40,896, with top-1 equal at all 64
positions. Differences below about 2e-04 mean nothing there. Both engines sit well above that.

Through the server at the full 40,960-token context, a passphrase placed at 37 % of a 37,659-token prompt is found.

## What "own code" covers

| Step | Code |
|---|---|
| Reading the GGUF file | `python/caminus/ggufio.py` |
| Tokenizer (byte-level BPE from the file's metadata) | `python/caminus/bpe.py`; equal to the Hugging Face `tokenizers` library on 3.8M tokens of test text |
| Weight decoding, Q4_K and Q6_K | `kernels/own_quant.h`, written from the block layouts; host copy `python/caminus/own_dequant.py` |
| Decode: one GPU launch per token | `kernels/k3_q3m_lean.hip` (the persistent kernel `kernels/k3.hip` with 7 ops: `k3_own_ops.h`, `k3_own_atts.h`, `k3_q3m_ops.h`), driven by `python/caminus/engine/q3moe_one.py` |
| Prompt reading | `kernels/q3m_prefill.hip` (quantized matrix products on RDNA3's WMMA instructions), driven by `python/caminus/prefill/q3m.py` |
| Prompt cache, sampler, chat template, tool-call parser, HTTP and SSE | `python/caminus/cache/`, `python/caminus/serve/` |

The platform underneath is the compiler (LLVM in ROCm), the HIP runtime, and PyTorch, which Caminus uses to allocate
device memory, launch its kernels and pick the largest logits for the sampler (`torch.topk`). llama.cpp is the
yardstick for speed and accuracy and a test oracle. Its GGUF reader and the `tokenizers` library stay in the
repository for the other models and for the tests that compare against them.

Two checks hold this, and both run without trusting a list written by hand:

- `tests/test_own_code.py` (CPU): the decode module and the prefill object have no ggml or llama.cpp header in their
  include closure, `build.sh` gives them no include path to one, and the serve modules import with `gguf` and
  `tokenizers` refused.
- `tools/serve_smoke1.py --block-imports gguf,tokenizers` (GPU): the real server runs under an import hook that refuses
  both packages, answers chat, tool-call, Anthropic, long-prompt and needle requests, and reports the GPU code
  objects it loaded (`/props`, `caminus.code_objects`). The smoke then audits the include closure of each one. The
  run on 2026-11-10 loaded `q3m_prefill` and `k3_q3m_lean`, and the hook refused nothing.

## How it works, briefly

- **One launch per token.** A persistent kernel walks a tape of tasks. Workgroups claim tasks, wait for the
  segment they depend on, compute, and publish. A token is about 580 segments: 12 per layer, plus the embedding and the head.
- **Matvec tiles.** A task is 16 rows of a matrix. A wave issues all of a tile's weight loads before it uses any of
  them, so a tile costs one memory round trip. Weights are read in place in GGUF block order (Q6_K rearranged into
  planes once at load) and decoded in registers; activations are fp16, sums fp32.
- **Experts.** The router's 8 experts per layer run as one grouped matvec; gate, up and the SwiGLU are one segment.
- **Attention.** Head dimension 128, an fp16 KV cache. Past the first 64 positions the context is split across
  workgroups that would otherwise idle; each split's rows are read once for all 8 query heads of a KV head, and the
  splits are merged with an online softmax. This is what holds speed at depth: 0.124 µs per position per token.
- **A decode module holds only the ops its tape runs.** Every op inlined into the persistent kernel adds live
  scalars, and past the scalar registers every segment hop of every wave reloads them. Trimming the kernel to the
  tape's 7 ops gave 0.5 ms per token with the same bits.
- **Prompt reading.** The weights stay exact in fp16 in one WMMA operand (Q4_K: scale × quant is an integer up to 945;
  Q6_K: quant − 32), the activations are fp16, the accumulation fp32. Chunks of 2,048 tokens; a prompt's last token
  runs as a decode pass.

## Where the remaining time is

At short context a token takes 4.32 ms against a 2.24 ms bound. Measured on the XT with builds that drop the loads or
the arithmetic (an earlier build, 4.38 ms per token there):

| Part | Time |
|---|---|
| Task structure: claims, dependency waits, publishes, with no loads and no arithmetic | 2.29 ms |
| Weight loads | 1.51 ms |
| Arithmetic and activations | 0.57 ms |

So the bytes are already read at close to the bandwidth; about half of a token is the hand-off between segments (432
dependent stages in series, about 5 µs each). Decoding the quantized formats is the smallest part. At depth the attention's reads
dominate and the share of the bound rises to 72 %.

## Limits

- One card with 24 GB: the full 40,960-token context needs 22.4 GB. A 20 GB card can't hold 32,768 tokens of it, which
  is the server's minimum.
- One model file is tested: `Qwen/Qwen3-30B-A3B-GGUF`, `Qwen3-30B-A3B-Q4_K_M.gguf`. The kernels cover Q4_K and Q6_K.
- No speculative decoding yet.
- One machine. Numbers from other cards are welcome.

## Reproduce

Kernel builds need ROCm (TheRock 10.2.0a20261011 here); see [building](building.md). Every GPU command runs under the
guard ([gpu-safety](gpu-safety.md)).

```bash
caminus pull qwen3-30b-a3b
M=~/.local/share/caminus/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf
./build.sh

python -I tests/test_own_code.py --model $M                       # the own-code checks (CPU)
tools/guard.sh LOG 900  python -I tests/engine/test_q3moe_one.py $M
tools/guard.sh LOG 1800 python -I benchmarks/q3moe_depth.py $M --depths 8,1024,4096,16384,32768,40900 --tokens 32 --rounds 5
tools/guard.sh LOG 1800 python -I benchmarks/q3moe_prefill.py $M --lengths 512,4096,32768 --chunk 2048 --rounds 3
tools/guard.sh LOG 2400 python -I tools/serve_smoke1.py --model $M --scratch DIR --card 0 --max-ctx 40960 \
    --needle 38000 --block-imports gguf,tokenizers

caminus start qwen3-30b-a3b                                        # serve it
```

The accuracy gates need references you build yourself: `tools/reference/q3moe_ref.py` and `q3moe_ref32.py` for the
floating-point logits, `tools/reference/gg_golden.cpp` for llama.cpp's.
