<p align="center">
  <img src="docs/banner.svg" alt="CAMINUS" width="100%">
</p>

<p align="center">
  <b>An inference engine written for two AMD Radeon™ RDNA3 GPUs from the kernel up. It serves one mixture-of-experts coding
  model, the Qwen3.8-Flash-Next GSQ-RCO Coder, at about 104 tok/s, and at 123–203 tok/s with DFlash drafting: about twice
  llama.cpp on the same two cards. Its core kernels match llama.cpp bit for bit; end to end, the output stays within
  llama.cpp's own noise.</b><br>
  One persistent GPU launch per token, both cards working as one, an OpenAI / Anthropic API for coding harnesses.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/GPU-RX%207900%20XTX%20%2B%207900%20XT-b4561f?style=flat&labelColor=222a31" alt="GPU: RX 7900 XTX + 7900 XT">
  <img src="https://img.shields.io/badge/arch-RDNA3%20%C2%B7%20gfx1100-4b5b6b?style=flat&labelColor=222a31" alt="RDNA3, gfx1100">
  <img src="https://img.shields.io/badge/platform-Linux-4b5b6b?style=flat&labelColor=222a31&logo=linux&logoColor=white" alt="Linux">
  <img src="https://img.shields.io/badge/runtime-PyTorch%20ROCm%207.2-4b5b6b?style=flat&labelColor=222a31" alt="runtime: PyTorch ROCm 7.2">
  <img src="https://img.shields.io/badge/python-3.12-4b5b6b?style=flat&labelColor=222a31&logo=python&logoColor=white" alt="Python 3.12">
  <br>
  <img src="https://img.shields.io/badge/model-Qwen3.8--Flash--Next%20GSQ--RCO%20Coder-4b5b6b?style=flat&labelColor=222a31" alt="Qwen3.8-Flash-Next GSQ-RCO Coder">
  <a href="docs/BENCHMARKS.md"><img src="https://img.shields.io/badge/evidence-every%20number%20sourced-b4561f?style=flat&labelColor=222a31" alt="every number sourced"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0--or--later-4b5b6b?style=flat&labelColor=222a31" alt="AGPL-3.0-or-later"></a>
</p>

---

**Caminus** (*KAM-ih-nus*, Latin for forge or furnace) is the RDNA3 member of the ignite family, alongside
[ignite-xdna](https://github.com/jdominick05/ignite-xdna) and [ignite-llama](https://github.com/jdominick05/ignite-llama).

## What it is, and what it is not

- **One model.** The [Qwen3.8-Flash-Next GSQ-RCO Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
  (ISTA-DASLab's IQ1_M pack, 256 of 512 experts: 29.6 GB of weights plus a 28.8 GB n-gram table read from disk), with the
  [DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) drafter. Other models need a port.
- **Two gfx1100 cards.** Tested on exactly one pair: an RX 7900 XTX (24 GB, device 0, driving the monitors) and an RX 7900 XT
  (20 GB) with PCIe peer-to-peer. The code objects are built for gfx1100 only. Another pair of gfx1100 cards may work, or be
  refused by the capacity check, whose fixed VRAM parts were measured on this pair.
- **A local server for one user.** `caminus serve` speaks the OpenAI (`/v1/chat/completions`, `/v1/completions`) and
  Anthropic (`/v1/messages`) APIs on 127.0.0.1:8092, runs one conversation at a time at the model's full 262,144-token
  context, and keeps earlier turns in a prompt cache (VRAM checkpoints, a side slot for a harness's title calls, a disk tier).
- **Not** a general inference server: no batching across users, no other GPUs, no Windows, no CUDA.
- **Research code that is now usable.** It was built and measured on the machine above, then packaged; the project is paused
  there (2026-11-06) while it is used as an end user would.

## At a glance

Measured on that machine; [docs/BENCHMARKS.md](docs/BENCHMARKS.md) gives each number's date, tool and conditions. llama.cpp
is the reference and the bar throughout: the same model file on the same two cards.

| | Caminus | llama.cpp |
|---|---|---|
| **Decode** (short context; 11-03 / 11-06) | **9.87 ms/token** in the decode benchmark; **104 tok/s** in the generation loop | 44.2 tok/s |
| **Decode with DFlash** (prose / code / shell; 11-06) | **156 / 203 / 123 tok/s**, the output identical to plain decoding | 92 / 89 / 65 tok/s |
| **Accuracy** (11-03) | mean KLD **0.0057**, top-1 **96.2 %** against llama.cpp's one-token decode over 768 positions | its own batch-vs-decode floor: 0.0076 |
| **Long context** (11-05) | to the model's 262,144 cap: every accuracy check against llama.cpp passes; recall probes 17 / 18 | 16 / 18 |
| **Decode at depth** (11-05) | 11.6 / 12.8 / 15.3 ms/token at 64K / 128K / 262K | ~30 / ~38 / ~60 ms/token |
| **Reading a prompt** (11-04 / 11-05) | 2,806–2,909 tok/s at 4K–32K; 2,393 tok/s at 262K | best 2,884 tok/s (8K); 950 tok/s at 262K |
| **Through the server** (11-06) | 101,525 tokens read in 40.1 s; the next turn of an 11K conversation starts in 0.21 s; +0.7–1.2 % over the engine alone | |

**Where it is not faster:** the prefill is about 1 % under llama.cpp's best at 8K; at ~262K DFlash costs more than plain
decoding on prose (18.3 vs 15.3 ms/token), so the server drafts only when it pays; plain decode at 262K (15.3 ms/token) is
above the design's estimate (~11.3).

## Install

Linux, Python 3.12, the two cards, ~29 GiB of RAM and ~60 GB of disk for the model. The full guide, with the guard, the
harness configs and troubleshooting: **[docs/INSTALL.md](docs/INSTALL.md)**.

```bash
python3.12 -m venv ~/caminus-venv && . ~/caminus-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2   # PyTorch's ROCm build (not on PyPI)
pip install caminus
caminus doctor            # is this machine ready? (CPU only)
caminus download          # the model files from Hugging Face (59 GB; resumable, sha256-checked)
caminus serve             # under the guard; ready in ~50 s on http://127.0.0.1:8092
```

Then point a harness at it: example wrappers for Qwen Code, Hermes Agent and Claude Code are in
[packaging/harness/](packaging/harness/); [llm-top](https://github.com/jdominick05/llm-top) shows the engine live.

## Built not to take the desktop down

One of the cards drives the monitors. Every GPU run goes through a guard (shipped in the package; `caminus serve` starts
itself under it) that kills the run when a card passes 0.97 of its VRAM or host memory runs low, and at the first amdgpu page
fault, reset or timeout in the kernel log; after a fault it refuses every later run until someone has looked. Device waits are
time-bounded, contexts beyond the cache are refused, never written, and the prompt cache keeps a reserve on the display card.

## How it works

- **One launch per token.** K3 is a persistent kernel that walks a tape of segments. Workgroups claim tasks, and claim / done
  counters order the dependencies, so a whole forward pass needs no host round trip.
- **Two cards as one.** Tensor parallel over PCIe peer-to-peer: GDN by key heads, attention by kv head, experts by
  intermediate columns. Every exchange sums card 0's partial first, so both replicas stay bitwise equal.
- **One register budget for every op.** 96 VGPRs, 8 workgroups per WGP. Engine modules are compiled per op and format
  whitelist, so no heavy op costs the others occupancy.
- **Long context the model's own way.** Qwen's QSA indexer attends to the top 2,048 positions per token, so attention at 32K
  costs what it costs at 2K.
- **Speculative decoding inside the engine.** A 4-token verify pass with lazy-committed recurrent state, and the DFlash 2
  drafter running as its own K3 module.
- **Its own batched prefill.** llama.cpp's MMQ kernels (vendored) plus its own routed-expert and dense GEMMs, each bitwise
  llama.cpp's MMQ per output, with the two cards' exchange ordered by HIP events only.
- **Verify, then speed up.** Each optimization step was checked against the previous build (bitwise where the math allows,
  otherwise against llama.cpp's own noise floor) and by an independent review before it was kept.

Design notes: [the engine](docs/K3-megakernel-design.md), [the specs](docs/agentic/) (drafters, QSA, prefill and cache,
metrics), [the 262K sweep](docs/agentic/sweep-262k.md), [the package](docs/RELEASE-PLAN.md), [the smoke benchmark](docs/SMOKE.md).

## Status

Done: the engine on both cards, DFlash at full context, the QSA indexer to 262K, the batched prefill, the prompt cache, the
server with metrics and a flight recorder, the package. Next, after the pause: a harness bake-off (Hermes Agent vs Qwen Code
on the NYU CTF Bench) and a head-to-head with [Strata](https://github.com/Niko1221/Strata). Parked: MTP drafting (it works,
but isn't used) and 1M context through YaRN (it needs more host memory).

## Working on it

Building from source needs TheRock ROCm 10.2 and the ggml headers of the ignite-llama fork at its commit e4e3f8eda
([github.com/jdominick05/ignite-llama](https://github.com/jdominick05/ignite-llama); docs/INSTALL.md, "Building from
source"); the installed package does not need it. [CLAUDE.md](CLAUDE.md) and [AGENTS.md](AGENTS.md) hold the layout, the commands, the test suites and the invariants
(GPU safety, bitwise discipline) for people and coding agents alike. The CPU suites run without the GPUs; the accuracy gates
need llama.cpp reference dumps, which are not distributed.

This repository starts from a squashed snapshot; the development history is private. Comments cite it (commit hashes,
decision ids such as D-281, step names such as `impl:package`, plan documents not in this tree): treat those as labels.

## License

Copyright (C) 2026 jdominick05. Licensed under [AGPL-3.0-or-later](LICENSE). Third-party code (llama.cpp / ggml, gguf-py,
Strata, flash-linear-attention, transformers) and its notices: [THIRD_PARTY.md](THIRD_PARTY.md). Outside contributions need a
contributor licence agreement (the copyright holder keeps the option to relicense): open an issue first.

**Model licences.** No model weight is in this repository or the package; `caminus download` fetches them from their own
repositories, under their own terms, which you must read before you use them:
- the base model, Qwen/Qwen3.8-Flash-Next: `qwen-community-1.0`;
- the GSQ-RCO Coder pack (ISTA-DASLab): its card's tag says Apache-2.0 while its text says the weights inherit the base model's
  licence (read it as qwen-community-1.0);
- the DFlash2 drafter (Inco AI): Apache-2.0.
