# Models

By the end of this page you'll know which model fits your cards and what to expect from it. Caminus downloads each model
from its Hugging Face repository and checks every file before using it.

## The models

| Name | Model | Cards | Download | Decode | Prompt reading |
|---|---|---|---|---|---|
| `qwen3.6` | Qwen3.6-35B-A3B, 2-bit (GSQ) | 1, any | 12.2 GB | 175 tok/s | 5,227 tok/s |
| `qwen3.6-hauhau` | Qwen3.6-35B-A3B, HauhauCS Q3_K_P | 1, 24 GB | 19.0 GB | 192 tok/s | 4,136 tok/s |
| `arex-2` | AREX-2 27B (a dense model) | 1, 24 GB | 17.2 GB | 41 tok/s | 928 tok/s |
| `coder` | Qwen3.8-Flash-Next Coder + its DFlash drafter | 2: an XTX + XT or XTX | 59.1 GB | 104 tok/s, 123-203 with the drafter | 2,800 tok/s |
| `qwen3-coder-next` | Qwen3-Coder-Next 80B-A3B, IQ3_S | 2 | 29.7 GB | 136 tok/s | 135 tok/s |
| `qwen3-30b-a3b` | Qwen3-30B-A3B, Q4_K_M | 1, 24 GB | 18.6 GB | 231 tok/s | 6,493 tok/s |

"Decode" is how fast a model writes. "Prompt reading" is how fast it reads what you send it. The one-card numbers were
measured on an RX 7900 XTX; the two-card numbers on an XTX and an XT together. [Benchmarks](dev/benchmarks.md) lists
where each number comes from and how llama.cpp does on the same cards.

## Which one to pick

- **One card:** start with `qwen3.6`. It fits either card with a 262K-token context. With a 24 GB card,
  `qwen3.6-hauhau` writes a little faster.
- **An XTX + XT or two XTX:** `coder` handles the longest work: 262K tokens of context, a prompt cache that keeps whole
  conversations, and a drafter that speeds up code. It was tested with 29 GB of system RAM: it reads part of the model from
  disk. Two XT cards are too small for it.
- **Planning:** `arex-2` is a dense 27B model, so it's slower. It's meant to plan while `qwen3.6` does the work, in the
  [brain / hands team](panel.md#the-brain--hands-team).
- **`qwen3-30b-a3b`** is the fastest writer on one 24 GB card and the most exact (4-bit weights, no llama.cpp code on its
  path). Its context is 40,960 tokens, the model's own limit, so it suits shorter work: 187 tok/s with 4K tokens of
  context, 104 tok/s when full. [The details](dev/qwen3-30b-a3b.md).
- **`qwen3-coder-next`** writes fast but reads prompts slowly for now (a 32K-token prompt takes about six minutes). Use it
  for short prompts until its fast prompt reading lands.

## Download and start

In the panel: `p` to download, `enter` to start. From a terminal:

```bash
caminus pull qwen3.6       # downloads into ~/.local/share/caminus/models; resumes if interrupted
caminus start qwen3.6      # picks the card and the largest context that fits
```

Caminus sizes the context to your cards: up to 262,144 tokens (40,960 for `qwen3-30b-a3b`). It refuses to start a model
that can't hold at least 32,768 tokens on your cards, rather than run out of memory later.

## The names your tools use

Each server answers to its model id and to the same id with `-nothink` (no thinking before answering): `qwen3.6` (both
Qwen3.6 files), `arex-2`, `qwen3-coder-next`, `qwen3-30b-a3b`. The Coder answers to `k3-coder`, `k3-coder-nothink`, `k3-coder-low` and `k3-coder-xhigh`
(the last two set how long it thinks).

## Licences

Caminus doesn't include any model weights. Each model has its own licence; read it before you use the model:

| Model | Licence |
|---|---|
| Qwen3.6-35B-A3B (both files), AREX-2, Qwen3-Coder-Next, Qwen3-30B-A3B | Apache-2.0 |
| The DFlash drafter | Apache-2.0 |
| Qwen3.8-Flash-Next Coder | treat it as `qwen-community-1.0`, the base model's licence (its card says Apache-2.0 in one place and the base licence in another) |

## Other model files

`caminus serve --model FILE.gguf` serves other GGUF files of the same model families. They're untested, and the server
refuses weight formats it has no kernels for.
