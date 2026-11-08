# Caminus

**Fast local coding models on your AMD Radeon RX 7900 cards.** Caminus is an inference engine written from the GPU kernels
up for Radeon RDNA3 cards. It serves coding models to tools like Qwen Code through OpenAI- and Anthropic-compatible APIs.
On the same cards it writes 1.1 to 2.4 times as fast as llama.cpp at short context.

> **Caminus is early testing software, and things will break.** It has been tested on exactly one machine: an RX 7900 XTX
> and an RX 7900 XT on Linux. If something goes wrong, please
> [open an issue](https://github.com/jdominick05/caminus/issues/new/choose) with the output of `caminus doctor`.

## Get started

You need Linux, Python 3.12 and at least one RX 7900 XTX or XT.

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2   # PyTorch for AMD cards (not on PyPI)
pip install caminus
caminus doctor     # is this machine ready?
caminus panel      # download a model, start it, open Qwen Code on it
```

The [quickstart](https://github.com/jdominick05/caminus/blob/main/docs/quickstart.md) walks you through it.

## Models

| Model | Cards | Decode |
|---|---|---|
| Qwen3.6-35B-A3B, 2-bit | 1 | 175 tok/s |
| Qwen3.6-35B-A3B HauhauCS | 1 (24 GB) | 192 tok/s |
| AREX-2 27B | 1 (24 GB) | 41 tok/s |
| Qwen3.8-Flash-Next Coder (262K context) | 2 (XTX + XT) | 104 tok/s; 123-203 with its drafter |
| Qwen3-Coder-Next 80B | 2 | 136 tok/s |

Details: [models](https://github.com/jdominick05/caminus/blob/main/docs/models.md) ·
[hardware](https://github.com/jdominick05/caminus/blob/main/docs/hardware.md) ·
[benchmarks](https://github.com/jdominick05/caminus/blob/main/docs/dev/benchmarks.md).

No model weights are in this package. `caminus pull` downloads each model from its own repository, under its own licence.

## Safety

Every GPU process Caminus starts runs under a guard. It stops the process when a card passes 97% of its memory, and stops
all GPU work after a GPU fault, so a crash can't repeat and take your desktop down.

Caminus is Copyright (C) 2026 jdominick05, licensed under the GNU AGPL v3.0 or later. Source and third-party notices:
https://github.com/jdominick05/caminus.
