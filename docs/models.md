# Models

By the end of this page you'll know which model fits your cards and what to expect from it. Caminus downloads each model
from its Hugging Face repository and checks every file before using it.

## The models

| Name | Model | Cards | Download | Decode | Prompt reading |
|---|---|---|---|---|---|
| `arex-2` | AREX-2 27B (a dense model) | 1, 24 GB | 17.2 GB | 41 tok/s | 928 tok/s |
| `coder` | Qwen3.8-Flash-Next Coder + its DFlash drafter | 2: an XTX + XT or XTX | 59.1 GB | 104 tok/s, 123-203 with the drafter | 2,800 tok/s |
| `qwen3-30b-a3b` | Qwen3-30B-A3B, Q4_K_M | 1, 24 GB | 18.6 GB | 248 tok/s | 6,432 tok/s |
| `swift-1.5` | Swift 1.5 (Qwen3.8 27B, a dense model), Q4_K_M | 1, 24 GB; or 2 | 19.5 GB (2.1 GB of it the DFlash2 drafter, shared with `thinkingcap`) | 42 tok/s; 64 on two cards | 771 tok/s (measured on an XT) |
| `thinkingcap` | ThinkingCap (Qwen3.8 27B, a dense model), RCO 4.2 bpw | 1, any; or 2 | 16.4 GB (2.1 GB of it the DFlash2 drafter, shared with `swift-1.5`) | 47 tok/s; 68 on two cards | 942 tok/s |

"Decode" is how fast a model writes. "Prompt reading" is how fast it reads what you send it. The one-card numbers were
measured on an RX 7900 XTX; the two-card numbers on an XTX and an XT together. [Benchmarks](dev/benchmarks.md) lists
where each number comes from and how llama.cpp does on the same cards.

**Paused for this release:** Qwen3.6-35B-A3B (both files) and Qwen3-Coder-Next. They aren't measured on the current
code, so `caminus pull`, `caminus start` and `caminus serve` refuse them for now; `CAMINUS_ALLOW_PAUSED=1` still serves them, unmeasured. They come back once they are measured
again.

## Which one to pick

- **One card:** with a 24 GB card, start with `qwen3-30b-a3b`, the fastest writer (below). On a 20 GB card, `thinkingcap`
  is the model that fits.
- **An XTX + XT or two XTX:** `coder` handles the longest work: 262K tokens of context, a prompt cache that keeps whole
  conversations, and a drafter that speeds up code. It was tested with 29 GB of system RAM: it reads part of the model from
  disk. Two XT cards are too small for it.
- **Planning:** `arex-2` is a dense 27B model, so it's slower. It plans in the
  [brain / hands team](panel.md#the-brain--hands-team), which is paused with Qwen3.6, its other half.
- **`qwen3-30b-a3b`** is the fastest writer on one 24 GB card and the most exact (4-bit weights, no llama.cpp code on its
  path). Its context is 40,960 tokens, the model's own limit, so it suits shorter work: 213 tok/s with 4K tokens of
  context, 111 tok/s when full. [The details](dev/qwen3-30b-a3b.md).
- **`swift-1.5`** is a dense 27B model on Caminus's own code end to end, like `qwen3-30b-a3b` (no llama.cpp code on its
  path). It writes at about `arex-2`'s speed. `caminus serve --model ... --draft mtp` (one card) drafts with the model's
  own prediction block and verifies 4 tokens a pass (about 2.1-2.2x on code and prose, 1.7x on shell, served on an
  XTX; no extra file); `--draft dflash` drafts with its base model's DFlash2 drafter
  instead (about 2.2x on code and prose, 1.7x on shell, served on an XTX; the drafter's 2 GB leaves ~12-16K tokens of
  context on a 24 GB card, so with it the server starts from 8K instead of 32K). Either way the output is identical to plain decoding. `caminus pull swift-1.5` (or
  `thinkingcap`) fetches the 2 GB drafter file with the model, once for both. Its speed drops at depth: about
  15 tok/s with 28K tokens of context.
- **`thinkingcap`** is the other dense 27B model on Caminus's own code end to end: a finetune of the same base, quantized
  to 4.2 bits per weight. It is the smaller download and writes a little faster than `swift-1.5`. It fits a 20 GB card
  too: on an XT the server picks about 53K tokens of context by itself. `--draft mtp` (one card) drafts with its own
  prediction block (about 1.8x on code and prose, served on an XT; no extra file); `--draft dflash` (one card) drafts with the base
  model's DFlash2 drafter (output identical to plain decoding; served on an XT about 1.9x on code and prose, 1.5x on
  shell). It slows with depth (about 16 tok/s with 28K tokens of context). Its licence is PolyForm Small Business 1.0.0: read its terms
  before any commercial use.

## Download and start

In the panel: `p` to download, `enter` to start. From a terminal:

```bash
caminus pull thinkingcap       # downloads into ~/.local/share/caminus/models; resumes if interrupted
caminus start thinkingcap      # picks the card and the largest context that fits
```

Caminus sizes the context to your cards: up to 262,144 tokens (40,960 for `qwen3-30b-a3b`). It refuses to start a model
that can't hold at least 32,768 tokens on your cards, rather than run out of memory later (8,192 with `--draft dflash` on
the 27B models: the drafter's memory comes out of the context).

`swift-1.5` and `thinkingcap` also run on two cards at once (`caminus start swift-1.5 --cards 2`, or `caminus serve
--cards 2`). Each card holds half of every layer, so the model writes about 1.4 times as fast and the full 262,144-token
context fits. Reading a prompt is about as fast as on one XTX for now.

## The names your tools use

Each server answers to its model id and to the same id with `-nothink` (no thinking before answering): `arex-2`,
`qwen3-30b-a3b`, `swift-1.5`, `thinkingcap` (and, while paused, `qwen3.6` and `qwen3-coder-next`). The Coder answers to `k3-coder`, `k3-coder-nothink`, `k3-coder-low` and `k3-coder-xhigh`
(the last two set how long it thinks).

## Licences

Caminus doesn't include any model weights. Each model has its own licence; read it before you use the model:

| Model | Licence |
|---|---|
| AREX-2, Qwen3-30B-A3B (and the paused Qwen3.6-35B-A3B and Qwen3-Coder-Next) | Apache-2.0 |
| Swift 1.5 | the Swift Open License 1.0 |
| ThinkingCap | PolyForm Small Business 1.0.0 (read its terms before any commercial use) |
| The DFlash2 drafters (the Coder's, and the 27B models' shared one) | Apache-2.0 |
| Qwen3.8-Flash-Next Coder | treat it as `qwen-community-1.0`, the base model's licence (its card says Apache-2.0 in one place and the base licence in another) |

## Other model files

`caminus serve --model FILE.gguf` serves other GGUF files of the same model families. They're untested, and the server
refuses weight formats it has no kernels for.
