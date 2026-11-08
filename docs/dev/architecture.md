# How Caminus works

By the end of this page you'll know the main pieces of Caminus, where each lives in the code, and the rules that hold them
together. The detailed contracts are in [docs/specs/](../specs/); the engine's full design history is in
[engine-design.md](engine-design.md).

## The pieces

```
caminus panel / caminus start          (cmd/manage.py, panel/: start servers, pick cards)
          |
caminus serve                          (serve/: OpenAI + Anthropic HTTP, templates, tool calls, sampling)
          |
the session                            (serve/k3_session.py: one request at a time, drafting, recovery)
          |
the prompt cache                       (cache/: the live conversation, checkpoints in VRAM and on disk)
     |                 |
the prompt path        the decode engine
(prefill/: reads a     (engine/: one persistent GPU launch per pass,
 prompt in chunks)      decode/: the drafters)
          |
the GPU kernels                        (kernels/*.hip, built by build.sh)
```

## The decode engine

The engine runs a whole model pass, every layer, in **one persistent kernel launch**. The host builds a *tape*: a list of
tasks (matrix-vector products, attention, norms, routing) with their dependencies. Workgroups claim tasks from it and wait
on each other through counters in memory, each wait bounded in time. One launch per pass removes the launch overhead
that dominates small-batch decoding.

- **One model, many builds:** the `kernels/k3_gguf*.hip` files are one engine compiled with different sets of operations
  and weight formats. A persistent kernel has one register budget for every operation it contains (96 registers, no
  spills), so each model loads only what it needs. `MODULE_CAPS` in `engine/k3.py` says what each build contains, and the
  host refuses anything else.
- **Two cards:** a two-card model splits each layer across both cards and exchanges partial sums twice per layer over
  PCIe. Card 0's partial is always added first, so both cards hold the same bits.
- **Bitwise discipline:** the engine's weight-format kernels match llama.cpp's bit for bit. A speed change must leave the
  logits bitwise equal to the previous build's; changes that can't are held to llama.cpp's own noise instead
  ([testing](testing.md)).

## Drafting

The Coder has a small drafter model (DFlash) that guesses several tokens ahead; the engine checks four tokens per pass and
keeps the ones that match. The output equals plain greedy decoding. The server measures whether drafting pays for each
kind of text and turns it off when it doesn't (`serve/draftpolicy.py`).

## The prompt path

Long prompts don't go through the decode engine one token at a time. The batched prefill (`prefill/`) reads them in chunks
of 2,048 tokens with matrix kernels (llama.cpp's own, vendored, plus Caminus's), then hands the state to the decode engine.
Each prefill operation gives the same bits as the decode engine's operation on the same input.

## The prompt cache

The cache (`cache/prompt.py`) owns the conversation's live state: what's on the GPU and which tokens it represents. It
keeps checkpoints of that state in VRAM and on disk, so a new turn only reads its new tokens, a short side request runs
without losing the conversation, and a restart brings it back from disk. GLOSSARY.md defines its terms.

## Models

`caminus serve` reads the model file's architecture and looks it up in a registry (`arch.py`). Each entry says which
engine, prompt path and drafter to use and how many cards it needs. Adding a model family means adding an entry:
[adding a model](adding-a-model.md).

## Cards

`device/host.py` finds the cards from sysfs, without starting the GPU runtime, by PCI address. It decides which cards a
model runs on, refuses unsupported cards and slow slots, and sets `HIP_VISIBLE_DEVICES` before the runtime starts.
`device/capacity.py` projects every allocation against the guard's line before anything is allocated.
