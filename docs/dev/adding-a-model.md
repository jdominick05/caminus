# Adding a model

By the end of this page you'll know what a new model family needs before Caminus can serve it. Start with an issue: say
which model, which file, and what you measured with llama.cpp on your cards.

## 1. The registry entry

`caminus serve` picks everything from the GGUF's `general.architecture` through `ARCHS` in `python/caminus/arch.py`. An
entry says:

- the engine class and how many cards it needs,
- the tokens per pass and the drafter, if any,
- `build()`: the engine, its prompt path and drafter, with every VRAM check done before any weight is read,
- `auto_need()`: the VRAM projection that sizes the context to the cards,
- the model ids the server answers to.

Nothing else in the server, the CLI or replay branches on the architecture. `tests/test_arch.py` checks the registry.

## 2. The engine

If the model reuses known layer types, its engine is mostly a new tape layout over existing operations
(`engine/qwen35_one.py` and `engine/qwen3next_tp.py` are examples). New weight formats or operations need kernel work in
`kernels/`, with a new build entry in `MODULE_CAPS` (`engine/k3.py`).

- Every new kernel runs first on a non-display card with small inputs ([GPU safety](gpu-safety.md)).
- Each operation must match llama.cpp's on the same inputs, bit for bit where llama.cpp's own kernel is deterministic.
- The persistent kernel stays at 96 registers or fewer with no spills: check with `llvm-readelf` on the built object.

## 3. The prompt path

A batched prefill is optional but makes a big difference: without one, prompts are read one token per pass
(`prefill/kpass.py`, as Qwen3-Coder-Next does today). A prefill driver subclasses `PrefillDriver` (`prefill/driver.py`)
and must give the same bits as the decode engine's operations.

## 4. The catalog

To make the model downloadable by name, add its files (repository, pinned revision, size, sha256) to `ITEMS` and a
`CATALOG` entry in `python/caminus/models.py`, with measured decode and prefill speeds labelled as such.

## 5. The gates

Before a model ships:

- the end-to-end gate against llama.cpp: within llama.cpp's own noise, measured on the same texts;
- the prefill's handoff and its accuracy against llama.cpp's prefill-then-decode;
- `tools/serve_smoke1.py --model FILE` (or `serve_smoke.py` for two-card models): the real server end to end;
- its VRAM projection checked against measured use.

Record the numbers in [benchmarks](benchmarks.md) and the model's licence in THIRD_PARTY.md and [models](../models.md).
