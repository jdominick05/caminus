# CLAUDE.md

Guidance for Claude Code and other coding agents working in this repository. AGENTS.md is the short version and defers to
this file.

## What this is

**Caminus** (product, CLI and Python package `caminus`) is an inference engine written for AMD RDNA3 from the kernel up. **K3**,
its core, is a persistent decode megakernel: one GPU launch per token (or per speculative verify block) walks a tape of
segments, and two gfx1100 cards work as one through tensor parallelism over PCIe peer-to-peer. It serves one model, the
**Qwen3.8-Flash-Next GSQ-RCO Coder** (ISTA-DASLab's IQ1_M pack, 256 of 512 experts), with the **DFlash2** drafter, behind an
OpenAI / Anthropic API (`caminus serve`) for coding harnesses.

- **Tested on exactly one machine:** an RX 7900 XTX (device 0, it drives the monitors) and an RX 7900 XT (device 1) behind a
  PCIe Gen4 x4 link, 29 GiB of host RAM, Linux. Other pairs of gfx1100 cards may be refused by the capacity check, whose fixed
  VRAM parts were measured on that pair.
- **Status:** paused after end-user readiness (2026-11-06). README.md has the measured numbers, docs/INSTALL.md the user guide,
  docs/BENCHMARKS.md where each number comes from.
- **Design:** docs/K3-megakernel-design.md (the engine), docs/agentic/spec-*.md (the drafters, QSA, the prefill and cache, the
  metrics surface), docs/agentic/sweep-262k.md (the long-context sweep), docs/RELEASE-PLAN.md (the package), docs/SMOKE.md.
- **The bitwise reference is llama.cpp**, through the ignite-llama fork at commit e4e3f8eda
  (github.com/jdominick05/ignite-llama; its ggml headers are what build.sh compiles against; its HIP build is the reference
  binary of the gates; docs/INSTALL.md, "Building from source"): kernel builds and the gates need it, the installed package
  does not. Notes and the vendored kernels/ggml_mmq headers name the development commit 842653396, published as e4e3f8eda
  (identical ggml/, common/, src/). tools/gg_golden.cpp, built by tools/build_gg_golden.sh, dumps its tensors and logits.

**This repository is a squashed snapshot.** The development history is private. Comments and docs cite it: commit hashes,
decision ids (`D-277`, `D-281`: the D-281 gate is the KLD floor rule below), milestone and step names (`M5 S6`,
`impl:package`, `fix:m7`), workflow ids and plan documents that are not in this tree (docs/AGENTIC-PLAN.md,
docs/INFERENCE-PLAN.md, docs/agentic/spec-server.md). Treat them as labels; the code and the tests are the authority.

## Layout

### kernels/

`kernels/*.hip` are device code only. `./build.sh` compiles each to `build/<name>.hsaco` (gfx1100 code object v5) and records
build/BUILD-INFO + build/BUILD-CMDS.tsv (the toolchain, the sources' commit / dirty flag / content hash, each object's command).
`python/caminus/hip.py` loads code objects into the HIP runtime PyTorch already started, from `caminus.kernels.kernel_dir()`:
`CAMINUS_KERNEL_DIR`, else a checkout's `build/`, else the installed package's `hsaco/gfx1100` (the runtime set, staged by
`tools/stage_kernels.py` with provenance.json).

- **The engine core:** `k3.hip` (the tape runner: workgroups claim tasks from segments and wait on claim/done counters with
  time-bounded waits, a sticky `err` and a prologue bound guard), `k3_gguf_ops.h` (the GGUF ops), `k3_qsa_ops.h` (the QSA
  indexer and selected-cell attention), `k3_attn_split.h` (split attention), `k3_draft_ops.h` (the DFlash drafter).
- **The k3_gguf*.hip files are one engine compiled with different op / format whitelists**, because a persistent kernel has
  one register allocation for every op: one heavy op would lower occupancy for all of them (the budget is 96 VGPRs, 0 spills,
  8 workgroups per WGP), and k3_run is ~120-160 KB against a 32 KB I-cache. `k3_gguf` (REAP-308), `_gsq` (the Coder),
  `_gsq_qsa` (the QSA tape), `_gsq_mtp` (the MTP layer), `k3_gguf_draft` (DFlash), `*_mt` (T > 1 verify passes), `*_trace`
  (profiling), `*_swiglu` (test builds, `K3_TEST_BUILDS=1` only).
- **The batched prefill** (not K3): `prefill_mmq.hip` (llama.cpp's MMQ, vendored in `kernels/ggml_mmq` by tools/vendor_mmq.py;
  build.sh compiles it the way llama.cpp does, one object per format, so the instructions are llama.cpp's: tests/test_prefill_mmq.py
  codegen; do not move it to the `--offload-device-only` path or merge the formats), `prefill_moe.hip` and `prefill_dense.hip`
  (its own routed-expert and Q6_K GEMMs, bitwise MMQ per output), `prefill_glue.hip`, `prefill_gdn.hip`, `prefill_tp.hip`,
  `prefill_attn.hip`, `prefill_feed.hip`, and `gemm.hip` (WMMA BF16 GEMMs).
- **Capability checks are mandatory.** `MODULE_CAPS` in python/caminus/k3.py says which (op, format) pairs each module has, and
  the host refuses anything else: the device dispatch's default case silently decodes an unknown format as Q8_0.
- The dormant training kernels (actq, attn, core, fuse, gdn, headloss, wquant, the S1 `k3*.hip`): docs/TRAINING-STACK.md.

### python/caminus/

- `k3.py` (the tape builder, `check_span` / `ContextBoundsError`, refusals), `flashnext_tp.py` (`FlashNextTP`, the production
  two-card engine: GDN split by key heads, attention by kv head, MoE by intermediate; `OP_XCHG` sums card 0's partial first so
  both replicas stay bitwise equal; a dense tape below 2051 positions, the QSA tape from there; lazy commit for speculative
  verify and a bitwise pass retry after a bail), `flashnext.py` (GGUF loaders; the 28.8 GB PLE table through O_DIRECT /
  native AIO, `AioRows`), `spec.py` (the speculative loop), `dflash_k3.py` (the DFlash drafter inside K3, its 4096-row ring and
  slot tags), `mtp.py` (MTP and `PolicyDrafter`), `longctx.py`, `passpolice.py` (eviction-aware pass limits), `capacity.py`
  (every VRAM budget, read as the guard reads it), `qsa.py` (the numpy reference of the QSA ops).
- The batched prefill: `prefill.py` (the driver: whole-prompt validation before anything is written), `prefill_feed.py`,
  `mmq.py`, `moe.py`, `dense.py`, `prefill_ops.py`, `prefill_gdn.py`, `prefill_tp.py` (the two-card exchange),
  `prefill_attn.py`, `prefill_graph.py` (opt-in HIP-graph chunks), `bufcheck.py`.
- `promptcache.py`: the prompt / conversation cache (VRAM checkpoints of the recurrent state, a side slot for harness side
  calls, a disk tier; the M7 contract is docs/agentic/spec-cache-api.md).
- `serve/`: the OpenAI / Anthropic front end (stdlib HTTP + SSE, template, tokenizer, tool-call parser, sampler, metrics, the
  flight recorder); `serve/k3_session.py` wires it to the engine through the prompt cache; `MockSession` drives the CPU suite.
- The package: `__main__.py` and `cmd/` (the console commands), `config.py` (every path: environment, then
  `~/.config/caminus/config.toml`, then the XDG defaults), `models.py` (the download manifest), `kernels.py` (code objects and
  provenance), `machine.py` (the guard's cards and lines), `torchcheck.py`, `_vendor/gguf` (llama.cpp's gguf-py reader,
  vendored: the Coder's Q2_0 tensors postdate gguf-py 0.19.0 on PyPI), `share/guard.sh` (the guard).
- `devpaths.py`: the environment variables the tests and dev tools read (below); nothing the server runs imports it.

### tools/, tests/, packaging/

- `tools/`: launchers from a checkout (`caminus_serve.py`, `replay.py`, `flight.py`, `kv_cache.py`), the end-to-end checks
  (`k3_smoke.py`, `serve_smoke.py`), benchmarks (`fn_bench.py`, `prefill_*_bench.py`), the long-context tools (`lc_probes.py`,
  `lc_baseline.py`, `sweep_262k.py`, `qsa_depth.py`), the reference builders (`gg_golden.cpp`, `gg_mmq_ref.cpp`,
  `dflash_dump.cpp`, `qwen_parse_ref.cpp`, `sampler_ref.cpp`, `philox_ref.cpp` and their `build_*.sh`), packaging
  (`stage_kernels.py`, `vendor_gguf.py`, `vendor_mmq.py`, `release_build.sh`, `repro_dist.py`, `verify_release.py`,
  `privacy_scan.py`), `gates/` (the bitwise and D-281 gates), `guard.sh` (a link to the
  shipped guard).
- `tests/`: plain scripts, no pytest. Each prints PASS / FAIL (most end with `RESULT PASS`) and takes the GGUF path as an
  argument where it needs one; read the script header for its stages. `tests/serve/` is unittest, run through `run_all.py`.
- `packaging/harness/`: example configs and wrappers for Qwen Code, Hermes Agent, Claude Code and llm-top
  (docs/INSTALL.md). `packaging/systemd/`: a user unit for a checkout (its README). `packaging/pypi/README.md`: the PyPI page.

## Models and data

Nothing here is distributed with the repository. `caminus download` fetches the Coder and the drafter. The gates also need
reference dumps from llama.cpp and stored results (`$ARTIFACTS`), which are not published: the CPU suites run without them;
the GPU gates and `tools/k3_smoke.py` need them (or references you build yourself with tools/gg_golden.cpp and
tools/gates/refdump.py on a known-good tree).

```
export CODER=~/.local/share/caminus/models/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
export DFLASH=~/.local/share/caminus/models/Qwen3.8-Flash-Next-DFlash2-GGUF/Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf
export ARTIFACTS=/path/to/a/big/disk/k3-artifacts   # reference dumps, stored results, smoke JSON, big outputs
export REAP=/path/to/Qwen3.8-Flash-Next-reap-308-UI.gguf   # optional: the REAP-308 regression model (not distributed)
export MTP=/path/to/mtp-sidecar.gguf                         # optional: the MTP sidecar (parked; not distributed)
export HF_TOKENIZER_JSON=/path/to/tokenizer.json             # optional: two tokenizer cross-checks in tests/serve
```

Tests and tools read these through `caminus.devpaths`: CODER / DFLASH / MTP fall back to `caminus.config` (CAMINUS_CODER ...,
config.toml, the download location); REAP and ARTIFACTS have no fallback (an unset one reads as the placeholder `$NAME`, so the
error names it; a scratch or output default that would be created exits with a message instead). tools/smoke_ref.json's paths
are `$VARIABLES` (k3_smoke's `load_ref()` expands them; `REAP_DUMPS` for `--model reap`). Shipped code takes its paths from
`caminus.config` only; no machine path is a default anywhere.

## Commands

```
./build.sh                                   # all kernels -> build/*.hsaco (+ runtime tests); ROCM_PATH, ARCH, IGNITE, TORCHLIB, PATCHELF override
python -I tools/stage_kernels.py             # the runtime set + provenance.json -> python/caminus/hsaco/gfx1100 (refuses objects not built from the current kernels/ + build.sh, and a dirty kernel tree without --allow-dirty)
python -I -m pip wheel . --no-deps --no-build-isolation -w dist   # the py3-none-any wheel (setup.py refuses one without the staged set)
python -I tools/vendor_gguf.py --check; python -I tools/vendor_mmq.py --check   # the vendored files equal their sources (--src: a clone of the fork at e4e3f8eda, default $IGNITE; vendor_mmq's headers carry the label 842653396, its development commit: --label)
tools/release_build.sh [--untagged | --write-hashes]   # the release files from a clean clone at a tag vX.Y.Z (docs/RELEASING.md): build, stage, the objects == packaging/gfx1100-kernels.sha256, dist/ (wheel, sdist, SHA256SUMS; reproducible: SOURCE_DATE_EPOCH defaults to the commit's committer time, tools/repro_dist.py normalizes the archives), then tools/verify_release.py dist --tag vX.Y.Z (what CI's release.yml runs on the Release's files)
python -I tools/privacy_scan.py .            # the privacy / secret scan CI runs on every push (private names by salted digest: --digest TERM)

# CPU only (no guard needed); HIP_VISIBLE_DEVICES= hides the GPUs
python -I tests/ci_cpu.py --scratch DIR       # every suite below that needs no GPU, as CI runs them; each is skipped with the reason when torch, CODER or the Coder's weights are missing (--list)
HIP_VISIBLE_DEVICES= python -I tests/serve/run_all.py       # the server suite on MockSession; needs CODER (the tokenizer and chat template only: a vocab-only GGUF from tools/gguf_vocab_only.py does), transformers and the anthropic SDK; the llama.cpp parity tests need the fork's CPU build (llama-tokenize: K3_LLAMA_TOKENIZE, default ~/src/ignite-llama/build-cpu/bin/llama-tokenize; libllama for the sampler reference) and are skipped without it; the parser parity test builds tools/qwen_parse_ref.cpp against the fork on first use (skipped if it cannot); test_cancel_latency is timing-sensitive: not beside a build or a GPU job
HIP_VISIBLE_DEVICES= python -I tests/test_packaging.py --scratch DIR   # the package's parts (config, guard lines, kernels / provenance, download on a local fake server, doctor, the console commands, the ROCm-PyTorch refusals without torch)
HIP_VISIBLE_DEVICES= python -I tests/test_bounds_host.py
HIP_VISIBLE_DEVICES= python -I tests/test_ple_aio.py
HIP_VISIBLE_DEVICES= python -I tests/test_pass_police.py
HIP_VISIBLE_DEVICES= python -I tests/test_spec_policy.py cpu
HIP_VISIBLE_DEVICES= python -I tests/test_dflash_ring_fuzz.py
HIP_VISIBLE_DEVICES= python -I tests/test_prefill_feed.py $CODER --cpu
HIP_VISIBLE_DEVICES= python -I tests/test_prefill_safety.py $CODER cpu
HIP_VISIBLE_DEVICES= python -I tests/test_prefill_gdn.py $CODER cpu vram
HIP_VISIBLE_DEVICES= python -I tests/test_prefill_mmq.py $CODER codegen shards --scratch DIR   # needs build/ and the fork's HIP build objects
HIP_VISIBLE_DEVICES= python -I tests/test_prompt_cache.py cpu --cache-dir DIR

# GPU: always through the guard (tools/guard.sh LOG SECS CMD..., or `caminus guard`)
tools/guard.sh LOG 600 python -I tools/k3_smoke.py --model coder     # needs $ARTIFACTS's references; ends RESULT PASS (the MTP stage only with CAMINUS_MTP set: docs/SMOKE.md)
tools/guard.sh LOG 600 python -I tools/fn_bench.py 96 $CODER --tp    # decode ms/token
HIP_VISIBLE_DEVICES=1 tools/guard.sh LOG 900 python -I tests/test_k3_bf16_mv.py $CODER --cards 1   # a kernel test on the second card alone
tools/guard.sh LOG 3600 python -I tools/serve_smoke.py --scratch DIR  # the real server end to end (both cards; no other engine running)
tools/gates/gates.sh $ARTIFACTS/<step>/gates all                      # the bitwise / D-281 gates against stored references (~14 min)
```

After a kernel change: check registers and spills with TheRock's `llvm-readelf` on build/<module>.hsaco (k3_run <= 96 VGPRs,
0 spills), rerun the gates at `K3_PER_WGP=3/6/7` (`gates.sh OUT grids qsa`: results stay bitwise), and run new or modified
kernel code on the non-display card first (`HIP_VISIBLE_DEVICES=1`).

## Invariants

**GPU safety** (the display card drives the desktop: a fault or a long launch can take it down):
- **Every GPU process runs under the guard** (tools/guard.sh, a link to python/caminus/share/guard.sh). It watches every gfx1100
  card, kills the run when a card passes 0.97 of its VRAM (in 50 MiB steps: 23,800 / 19,850 MiB on a 7900 XTX / XT) or host
  MemAvailable falls below 1,500 MiB, and on any amdgpu page fault, reset, wedge or timeout in the kernel log.
  - **Exit 99:** a GPU fault. Stop all GPU work and re-run nothing.
  - **Exit 98:** a fault was recorded earlier in the STOP file (`~/.local/state/k3/GPU_FAULT_STOP`); remove it only after the
    fault has been inspected.
  - Why: re-running a faulting experimental kernel once reset both GPUs nine times and lost the desktop session.
- **The display-driving card** keeps a reserve below its line (512 MiB for the cache tier); every device wait stays
  time-bounded (`limit_us`; TP uses 200 ms plus a bitwise retry) and every launch stays well under 200 ms. Never build a
  spin-forever or no-timeout variant: a 30 s spin once crashed the desktop.

**Correctness:**
- **Bitwise discipline.** A K3 change must leave logits, router ids and replicas bitwise equal to the previous reference,
  across `K3_PER_WGP` grids, with T = 4 passes equal to one-token steps (tools/gates). A change that cannot be bitwise by design
  passes the **D-281 gate** instead: mean KLD against llama.cpp's one-token decode no worse than llama.cpp's own
  batch-vs-decode floor (0.0076 on the Coder), top-1 within 1 point.
- **Bounds.** Every position-indexed host entry calls `k3.check_span`; device position-indexed accesses clamp and set `err`.
  Contexts beyond the allocation are refused, never written. The prefill's expert ids are clamped on the device before every
  mm_ids_helper launch and a chunk whose routing flag word is set is refused (`prefill_ops.check_routing`). Never launch an
  unclamped path with bad inputs to show it is unsafe: show it by CPU emulation (tests/test_prefill_safety.py).
- **The prefill's TP exchange** moves data only with hipMemcpyPeerAsync on its copy streams, ordered only by HIP events: never
  a kernel that spins on a peer flag, never torch's cross-device `copy_`. Keep both waits (every push waits the peer's last
  read of its receive rows; every in-place sum waits both pushes); card 0's partial goes first on both cards. HIP graphs never
  hold the exchange.
- **The DFlash ring is 4096 rows at every max_ctx**, and its slot tags refuse any launch whose window reads a slot that does
  not hold that position's row (`StaleRingError`).

**Host limits:** host page migration makes KFD evict the process's GPU queues (the cause of "a wait gave up" bails), so
never mmap the PLE shard in a long-lived process (use `AioRows`) and never do big file I/O while timing. Long-context per-pass
limits are eviction-aware (passpolice.py): keep the POLICE summary in every long-context result.

**Process:**
- Speed claims need interleaved A/B, the median of >= 3 runs, and a difference above 0.05 ms.
- Never regenerate or refresh a reference to make a gate pass: a gate that stops reproducing its reference is a finding.
- Unattended agents never run commands that need a permission prompt (deletes, writes to /tmp, sudo, installs).

## Conventions

- **Commits:** `<area> : <what>` (for example `tests : cover attention bounds`), with the gates run and the measured
  before / after in the body. AI assistance is disclosed with an `Assisted-by: <model>` trailer; no session links.
- **Scope:** MoE / low-active models only (no dense models); drafters are DFlash and MTP only; 1M context (YaRN) waits for
  more host RAM. Long term the stack may grow into a general low-level LLM stack for AMD targets: where it costs nothing, keep
  architecture constants and model shapes out of shared kernels and host code, but never generalize speculatively or at a
  speed cost before a second target or model exists.
- **Kernel comments** carry the measured reason for each choice (see build.sh and k3_gguf_ops.h); keep that density.
- **Python:** run with `-I` (scripts put `python/` on sys.path themselves). gguf-py is vendored (`caminus.config.ensure_gguf`;
  `CAMINUS_GGUF_PY` points at another tree).
- **License:** AGPL-3.0-or-later (LICENSE), Copyright (C) 2026 jdominick05. The copyright holder keeps the option to relicense,
  so outside contributions need a contributor licence agreement: open an issue before a pull request. Third-party code is
  listed in THIRD_PARTY.md; anything new from outside gets an entry there and keeps its notice in the file header, in the same
  commit.
- **Maintenance:** when an invariant or command here changes, update this file and AGENTS.md in the same commit.
