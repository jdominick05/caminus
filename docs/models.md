# Models

By the end of this page you'll know which model fits your cards and what to expect from it. Caminus downloads each model
from its Hugging Face repository and checks every file before using it.

## The models

| Name | Model | Cards | Download | Decode | Prompt reading |
|---|---|---|---|---|---|
| `coder` | Qwen3.8-Flash-Next Coder + its DFlash drafter | 2: an XTX + XT or XTX | 59.1 GB | 104 tok/s, 123-203 with the drafter | 2,800 tok/s |
| `qwen3-30b-a3b` | Qwen3-30B-A3B, Q4_K_M | 1, 24 GB; or 2 | 18.6 GB | 248 tok/s | 6,432 tok/s |
| `swift-1.5` | Swift 1.5 (Qwen3.8 27B, a dense model), Q4_K_M | 1, 24 GB; or 2 | 19.5 GB (2.1 GB of it the DFlash2 drafter, shared with `thinkingcap`) | 42 tok/s; 64 on two cards | 771 tok/s (measured on an XT) |
| `qwen3.8-27b` | The plain Qwen3.8 27B (a dense model), GSQ-RCO IQ3_S | 1, any | 14.2 GB (2.1 GB of it the DFlash2 drafter, shared with the other 27B models) | 55 tok/s | 800 tok/s (measured on an XT) |
| `thinkingcap` | ThinkingCap (Qwen3.8 27B, a dense model), RCO 4.2 bpw | 1, any; or 2 | 16.4 GB (2.1 GB of it the DFlash2 drafter, shared with `swift-1.5`) | 47 tok/s; 69 on two cards | 942 tok/s |

"Decode" is how fast a model writes. "Prompt reading" is how fast it reads what you send it. The one-card numbers were
measured on an RX 7900 XTX; the two-card numbers on an XTX and an XT together. [Benchmarks](dev/benchmarks.md) lists
where each number comes from and how llama.cpp does on the same cards.

**Paused for this release:** Qwen3.6-35B-A3B (both files), Qwen3-Coder-Next and AREX-2. They aren't measured on the current
code, so `caminus pull`, `caminus start` and `caminus serve` refuse them for now; `CAMINUS_ALLOW_PAUSED=1` still serves them, unmeasured. They come back once they are measured
again.

## Which one to pick

- **One card:** with a 24 GB card, start with `qwen3-30b-a3b`, the fastest writer (below). On a 20 GB card, `thinkingcap`
  is the model that fits.
- **An XTX + XT or two XTX:** `coder` handles the longest work: 262K tokens of context, a prompt cache that keeps whole
  conversations, and a drafter that speeds up code. It was tested with 29 GB of system RAM: it reads part of the model from
  disk. Two XT cards are too small for it.
- **`qwen3-30b-a3b`** is the fastest writer on one 24 GB card and the most exact (4-bit weights, no llama.cpp code on its
  path). Its context is 40,960 tokens by default, so it suits shorter work: 213 tok/s with 4K tokens of context, 111
  tok/s when full. For longer work, its opt-in long-context mode (`--long-context`, below) reaches about 80K tokens on
  one XTX and 131,072 on two cards. [The details](dev/qwen3-30b-a3b.md).
- **`swift-1.5`** is a dense 27B model on Caminus's own code end to end, like `qwen3-30b-a3b` (no llama.cpp code on its
  path). `caminus serve --model ... --draft mtp` drafts with the model's
  own prediction block and verifies 4 tokens a pass (about 2.1-2.2x on code and prose, 1.7x on shell, served on an
  XTX; no extra file); `--draft dflash` drafts with its base model's DFlash2 drafter
  instead (about 2.2x on code and prose, 1.7x on shell, served on an XTX; the drafter's 2 GB leaves ~12-16K tokens of
  context on a 24 GB card, so with it the server starts from 8K instead of 32K). Either way the output is identical to plain decoding. `caminus pull swift-1.5` (or
  `thinkingcap`) fetches the 2 GB drafter file with the model, once for both. Its speed drops at depth: about
  15 tok/s with 28K tokens of context.
- **`thinkingcap`** is the other dense 27B model on Caminus's own code end to end: a finetune of the same base, quantized
  to 4.2 bits per weight. It is the smaller download and writes a little faster than `swift-1.5`. It fits a 20 GB card
  too: on an XT the server picks about 53K tokens of context by itself. `--draft mtp` drafts with its own
  prediction block (about 1.8x on code and prose, served on an XT; no extra file); `--draft dflash` drafts with the base
  model's DFlash2 drafter (output identical to plain decoding; served on an XT about 1.9x on code and prose, 1.5x on
  shell). It slows with depth (about 16 tok/s with 28K tokens of context). Its licence is PolyForm Small Business 1.0.0: read its terms
  before any commercial use.
- **`qwen3.8-27b`** is the plain Qwen3.8 27B, not a finetune, quantized to about 3.6 bits per weight by ISTA-DASLab
  (GSQ-RCO, Apache-2.0). It is the smallest 27B download, fits either card and writes fastest of the three at short
  context (55 tok/s on an XTX). `--draft dflash` (about 1.2-1.4x, served on an XTX) and `--draft mtp` (about
  1.1-1.3x) give the same output as plain decoding; on this file the drafters gain less than on the other two.

## Download and start

In the panel: `p` to download, `enter` to start. From a terminal:

```bash
caminus pull thinkingcap       # downloads into ~/.local/share/caminus/models; resumes if interrupted
caminus start thinkingcap      # picks the card and the largest context that fits
```

Caminus sizes the context to your cards: up to 262,144 tokens (40,960 for `qwen3-30b-a3b`, up to 131,072 in its
long-context mode). It refuses to start a model
that can't hold at least 32,768 tokens on your cards, rather than run out of memory later (8,192 with `--draft dflash` on
the 27B models: the drafter's memory comes out of the context).

`swift-1.5` and `thinkingcap` also run on two cards at once (`caminus start swift-1.5 --cards 2`, or `caminus serve
--cards 2`). Each card holds half of every layer, so the model writes about 1.4 times as fast and the full 262,144-token
context fits with `--draft off` (with the default drafter it is projected to fit too on an XTX + XT with free memory, a
little less when the desktop holds memory on the XTX). Reading a prompt is about as fast as on one XTX for now. Both
drafters work on two cards too, with the same output as plain decoding, and `--draft dflash` is the default there when
the drafter file is present and 32,768 tokens of context still fit with it. On an XTX + XT with 10K-34K-token prompts, Swift 1.5 drafts at 1.36-1.44x and
ThinkingCap at 1.13-1.18x; on 11K-29K-token agent prompts the range is 1.20-1.45x
([benchmarks](dev/benchmarks.md#speculative-decoding-on-two-cards-served)). `--draft off` turns it off.

`qwen3-30b-a3b` runs on two cards too (`caminus start qwen3-30b-a3b --cards 2`, or `caminus serve --cards 2`), but one
card stays its default. Each card holds half of every layer's attention and part of every expert (two thirds on the
bigger card, which is card 0). Served, two cards decode more slowly than one XTX on short turns (213 against 245 tok/s)
and faster from about 32K tokens on (121 against 109 tok/s at 40,768), and read every prompt from 4K tokens on faster
(1.10x at 4K, 1.29x at 40,768); see [benchmarks](dev/benchmarks.md#qwen3-30b-a3b-on-two-cards). By default the context
stays 40,960 tokens, so the second card buys speed with a long context; in the long-context mode two cards hold the
full 131,072. No drafter on one card or two.

### Long context for `qwen3-30b-a3b` (opt-in)

`caminus start qwen3-30b-a3b --long-context` (or `caminus serve --long-context`, or a `--max-ctx` above 40,960) turns on
the model card's recipe for long texts: YaRN RoPE scaling, factor 4 from its native 32,768 tokens, up to 131,072 tokens.

- One XTX: as much as the card holds, about 80K tokens. Where the context only fits that way, the server stores the
  attention cache as 8-bit rows (Q8_0) and says so in its log.
- Two cards (`--cards 2 --long-context`): the full 131,072 tokens with the normal 16-bit cache.
- The trade-off: YaRN changes every position, not only the far ones, so short conversations read differently than in
  the default mode (in an exact reference of the model on a 128-token text, the most likely next token changed at 6 of
  128 positions). The model card advises it only for texts past 32,768 tokens; the default stays
  the native 40,960 without YaRN.
- Saved conversations are kept per mode: a conversation saved in one mode is never restored into the other.
- Other models refuse `--long-context` (the Coder has its own long-context mode: a `--max-ctx` above 262,144). So does
  another file of the same architecture whose header is not Qwen3-30B-A3B's (another context length or RoPE base, such as
  the 2507 releases), and a file with codebook (IQ) or row-scaled tensors.

### The 8-bit KV cache (`qwen3-30b-a3b`)

`qwen3-30b-a3b` also runs with an 8-bit attention cache: `caminus start qwen3-30b-a3b --kv q8` (or `caminus serve --kv
q8`), on one card or two (`--cards 2 --kv q8`). F16 stays the default. The cache holds Q8_0 rows (the keys rotated
first, which makes them easier to quantize) and takes 0.050 MiB per token of context instead of 0.094 (on two cards
half of that on each card). The context stays 40,960 tokens, the model's own limit, but the full context fits with
more of the card in use: on an XTX with the desktop holding 1 GiB, F16 fits 32,768 tokens and the 8-bit cache 40,960
(projected). Decode runs at about the same speed: on one XTX 0.97x on short turns and 1.06x with a full context,
where reading half the bytes pays off; on two cards within 2 % at every depth (each card already reads only its half
of the cache). Both engine-level; served numbers come with the release benchmarks. By mean KL divergence it changes
the output less than llama.cpp's own 8-bit cache changes llama.cpp's: measured on one XTX and on an XTX + XT at 128,
16,384 and 40,896 tokens, the mean KL divergence from the F16 cache is 1.2e-3 to 3.6e-3, llama.cpp's from its F16
cache 4.2e-3 to 6.6e-3 ([details](dev/q8-kv.md)). Top-1 agreement with the F16 cache is within one row of
llama.cpp's: on one card at 16,384 tokens 60 of 64 rows against its 61. A conversation the server saved in one format is never restored by a server running
the other; switching reads it again. The other models refuse `--kv q8`: the dense 27B models keep their F16 cache and
the Coder its own 8-bit one.

## Sampling defaults

Each model is served with the sampling its model card recommends for agentic work. These are the defaults a request
gets; any field the request sends itself (`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`,
`repeat_penalty` and the rest) wins.

| Model | temperature | top_p | top_k | min_p | presence penalty | repeat penalty | recommended output |
|---|---|---|---|---|---|---|---|
| `qwen3-30b-a3b` (thinking) | 0.6 | 0.95 | 20 | 0 | 1.5 | 1.0 (not stated) | 32,768 tokens |
| `qwen3-30b-a3b` without thinking | 0.7 | 0.8 | 20 | 0 | 1.5 | 1.0 (not stated) | 32,768 tokens |
| `coder`, `swift-1.5`, `thinkingcap`, `qwen3.8-27b` | 1.0 | 0.95 | 20 | 0 | 0 | 1.0 | 393,216 tokens (131,072 answer + 262,144 thinking) |

- Qwen3-30B-A3B's card recommends a presence penalty of 1.5 for quantized models, to suppress repetition. Its GGUF file
  carries no sampling settings, so before this table it was served at temperature 1.0 with no penalty. The penalty
  counts the last 64 tokens, as in llama.cpp.
- "Without thinking" means a request on `qwen3-30b-a3b-nothink`, or one that turns thinking off itself
  (`chat_template_kwargs: {"enable_thinking": false}`). The Qwen3.8 models' cards give one set for both, so they have one.
- A greedy request (`temperature: 0`) gets no default penalty, only the ones it sends itself, so greedy test and
  benchmark runs read the same as before. The cards advise against greedy decoding for real use.
- The Qwen3.8 models' GGUF files already carry 1.0 / 20 / 0.95, so nothing changes for them.
- The recommended output length is a hint only. `/props` reports it as `default_generation_settings.max_output` and
  `/v1/models` in each model's `meta`. A reply still ends at the request's `max_tokens` or when the context is full.
- `/props` reports the defaults in effect with thinking on under `default_generation_settings.params`.
- For another GGUF file, the server uses the file's own `general.sampling.*` settings (temperature, top_k, top_p, min_p
  and its repeat penalty), then the built-in defaults for the rest (temperature 1.0, top_p 0.95, top_k 20, min_p 0,
  repeat penalty 1.0, no presence or frequency penalty).

## The names your tools use

Each server answers to its model id and to the same id with `-nothink` (no thinking before answering):
`qwen3-30b-a3b`, `swift-1.5`, `thinkingcap` (and, while paused, `arex-2`, `qwen3.6` and `qwen3-coder-next`). The Coder answers to `k3-coder`, `k3-coder-nothink`, `k3-coder-low` and `k3-coder-xhigh`
(the last two set how long it thinks).

## Licences

Caminus doesn't include any model weights. Each model has its own licence; read it before you use the model:

| Model | Licence |
|---|---|
| AREX-2, Qwen3-30B-A3B, Qwen3.8-27B GSQ-RCO, and the paused Qwen3.6-35B-A3B and Qwen3-Coder-Next | Apache-2.0 |
| Swift 1.5 | the Swift Open License 1.0 |
| ThinkingCap | PolyForm Small Business 1.0.0 (read its terms before any commercial use) |
| The DFlash2 drafters (the Coder's, and the 27B models' shared one) | Apache-2.0 |
| Qwen3.8-Flash-Next Coder | treat it as `qwen-community-1.0`, the base model's licence (its card says Apache-2.0 in one place and the base licence in another) |

## Other model files

`caminus serve --model FILE.gguf` serves other GGUF files of the same model families. They're untested, and the server
refuses weight formats it has no kernels for.
