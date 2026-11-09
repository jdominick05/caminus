# AGENTS.md

The guide for AI coding agents working in this repository (any tool). Humans: start at README.md and CONTRIBUTING.md.
CLAUDE.md imports this file.

## What this is

- Caminus: an inference engine for AMD RDNA3 (gfx1100) written from the kernels up; CLI and Python package `caminus`.
- It serves five models through an OpenAI / Anthropic API (`caminus serve`): the Qwen3.8-Flash-Next GSQ-RCO Coder (two
  cards, with the DFlash2 drafter), Qwen3.6-35B-A3B GSQ and HauhauCS (one card), AREX-2 27B (one card, the only dense
  model), Qwen3-Coder-Next (two cards). `python/caminus/models.py` is the catalog.
- K3, the core: a persistent decode megakernel; one GPU launch per pass walks a tape of tasks; two cards work as one
  through tensor parallelism over PCIe peer-to-peer.
- Tested on one machine only: an RX 7900 XTX (drives the monitors) + an RX 7900 XT on a PCIe 4.0 x4 link, 29 GiB RAM, Linux.
- The Coder runs at its measured split (10 / 12) on every pair (`arch.Coder.split`), so it needs a 24 GB card 0: XTX + XT
  (tested) or XTX + XTX (same shapes, unmeasured). The even split (8 / 10) can't prefill: each card's expert slice is 320
  rows and the vendored MMQ only takes whole 128-row tiles (the shared expert's non-Q6_K gate / up go through `mmq.mm`).
- The bitwise reference is llama.cpp: the ignite-llama fork at e4e3f8eda (github.com/jdominick05/ignite-llama). Kernel
  builds and the gates need it; the installed package doesn't. Older notes name its development commit 842653396.
- Comments cite a private history: commit hashes, decision ids (D-281 = the KLD floor gate below), step names
  (`impl:package`), plan documents not in this tree. Treat them as labels; code and tests are the authority.

## Where to look

| Need | Read |
|---|---|
| How the pieces fit | docs/dev/architecture.md |
| Terms (live state, checkpoints, gates) | GLOSSARY.md |
| Contracts of the drafters, prefill, cache, metrics | docs/specs/spec-*.md |
| The engine's design and measured history | docs/dev/engine-design.md |
| Building, testing, GPU rules | docs/dev/building.md, testing.md, gpu-safety.md |
| Adding a model family | docs/dev/adding-a-model.md |
| Where each published number comes from | docs/dev/benchmarks.md |
| User-facing behaviour | docs/*.md (quickstart, panel, models, hardware, harnesses, troubleshooting, cli, configuration) |

## Layout (the parts that need a reason)

- `kernels/*.hip`: device code only; `./build.sh` writes `build/*.hsaco` and records provenance.
- The `k3_gguf*.hip` files are ONE engine compiled with different op / format whitelists: a persistent kernel has one
  register allocation for all its ops (budget 96 VGPRs, 0 spills) and a ~120-160 KB body against a 32 KB I-cache.
- `MODULE_CAPS` in `engine/k3.py` lists each module's (op, format) pairs; the host refuses anything else (the device's
  default case would silently decode an unknown format as Q8_0).
- `kernels/ggml_mmq/`: llama.cpp's MMQ, vendored (`tools/release/vendor_mmq.py --check`); build.sh compiles it the way the
  fork does, one object per format. Don't move it to the `--offload-device-only` path or merge formats: the code changes.
- `python/caminus/arch.py`: the architecture registry (`ARCHS`); serve, replay and the CLI never branch on the architecture.
- `engine/`: `flashnext_tp.py` (the Coder, two cards), `qwen35_one.py` (one-card Qwen3.6 / AREX-2), `qwen3next_tp.py`.
- `prefill/`: the batched prefill; every op is bitwise its K3 decode op on the same inputs; `driver.py` is the shared shell.
- `cache/prompt.py` + `cache/livestate.py`: the prompt cache owns the only live state (token log, (i0, commit), counters).
- `serve/`: HTTP + SSE, template, tool parser, sampler; `EngineSession` declares the session contract
  (`contract_problems(sess)`); `k3_session.py` wires it to the engine through the cache.
- `device/host.py`: the host's cards from sysfs + KFD topology, never torch; `caminus serve` places itself and sets
  `HIP_VISIBLE_DEVICES` before HIP starts. `device/capacity.py`: every VRAM projection and check.
- `cmd/`: console commands; `panel/`: the Go control panel and monitor (`caminus panel`, `caminus top`).
- `panel/openrgb.go`: optional agent-state lighting over backwards-compatible SDK v3. Active per-LED modes need no
  mode change; `openrgb.switch_modes` opts into temporary Direct/Custom/Static mode changes. Restore the first-seen
  colors and complete mode payload on exit, using freshly discovered identities and matching layouts after reconnects.
  Never resize zones, save modes/profiles or write through stale device IDs. Test against mock SDK servers.
- `panel/openrgb_effects.go`: clocked, hardware-free RGB renderer. Detect eight-LED motherboard JARGB 2/3 rings or use
  explicit zone layouts; keep other zones linear. Phase transitions start from the displayed frame, including when
  interrupted. Brightness, frame rate and animation speed are bounded; tests use a fake clock and mock devices.
- `cmd/manage.py:top_config` carries monitor preferences into each server's generated config. Preserve the source file,
  resolve its relative paths before merging, and let the selected server determine its address and agent sessions.
- `share/guard.sh` (linked as `tools/guard.sh`): the GPU guard, shipped in the package.
- `tests/`: plain scripts printing PASS / FAIL and `RESULT PASS` (no pytest); `tests/serve/` is unittest via `run_all.py`.
- `tools/gates/`: the bitwise and D-281 gates; `tools/reference/`: llama.cpp reference builders.

## Models and data

Nothing is distributed with the repository. Export these before any test, tool or gate:

```
export CODER=~/.local/share/caminus/models/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
export DFLASH=~/.local/share/caminus/models/Qwen3.8-Flash-Next-DFlash2-GGUF/Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf
export ARTIFACTS=/path/to/a/big/disk/k3-artifacts   # reference dumps and stored results (not published; build your own)
export HF_TOKENIZER_JSON=/path/to/tokenizer.json   # optional: two tokenizer cross-checks in tests/serve
```

- Tests and tools read them through `caminus.devpaths`; an unset ARTIFACTS reads as the placeholder `$ARTIFACTS` and fails
  naming it.
- Shipped code takes paths from `caminus.config` only (env, then `~/.config/caminus/config.toml`, then XDG defaults). No
  machine path is a default anywhere.
- Gate references are never regenerated to make a gate pass.

## Commands

```
./build.sh                                          # kernels -> build/*.hsaco (+ C++ runtime tests); ROCM_PATH, IGNITE, TORCHLIB, PATCHELF
(cd panel && go build -o caminus-panel . && go test ./...)
python -I tests/ci_cpu.py --scratch DIR             # every CPU suite, as CI runs them (a FRESH DIR; skips with the reason what's missing)
python -I tools/release/privacy_scan.py .           # no secrets, machine paths or session links in the tree (CI runs it)
python -I tools/release/vendor_gguf.py --check; python -I tools/release/vendor_mmq.py --check
tools/release/release_build.sh                      # reproducible release files from a clean clone at a tag (docs/dev/releasing.md)

# GPU: always under the guard
tools/guard.sh LOG 600 python -I tools/k3_smoke.py --model coder          # ends RESULT PASS (needs $ARTIFACTS)
tools/guard.sh LOG 600 python -I benchmarks/fn_bench.py 96 $CODER --tp    # decode ms/token
tools/guard.sh LOG 3600 python -I tools/serve_smoke.py --scratch DIR      # the real server end to end (two cards)
tools/guard.sh LOG 2400 python -I tools/serve_smoke1.py --model GGUF --scratch DIR --card N   # a one-card model end to end
tools/gates/gates.sh $ARTIFACTS/<step>/gates all                          # bitwise / D-281 gates (~16 min)
```

After a kernel change: `llvm-readelf` on the module (k3_run <= 96 VGPRs, 0 spills); rerun the gates at `K3_PER_WGP=3/6/7`
(`gates.sh OUT grids qsa`: still bitwise); first runs on a non-display card (`caminus ps` lists the cards).

## Invariants

GPU safety (a fault on the display card can take the desktop down):
- Every GPU process runs under `tools/guard.sh LOG SECS CMD...` (or `caminus guard`).
- Guard exit 99 = a GPU fault: stop all GPU work, rerun nothing. Exit 98 = a fault on record
  (`~/.local/state/k3/GPU_FAULT_STOP`): a person inspects it before anything runs.
- New or changed kernel code runs first on a non-display card alone (`HIP_VISIBLE_DEVICES=<its index>`), small inputs.
- Every device wait is time-bounded (`limit_us`; TP 200 ms + a bitwise retry); every launch well under 200 ms. Never a
  spin-forever or no-timeout variant.
- Every allocation is projected and checked against the guard's line before it happens (`capacity`); a display card keeps
  512 MiB free.

Correctness:
- Bitwise: a K3 change leaves logits, router ids and replicas bitwise equal to the previous reference, across `K3_PER_WGP`
  grids, with T = 4 passes equal to one-token steps.
- D-281: a change that can't be bitwise by design keeps mean KLD vs llama.cpp's one-token decode within llama.cpp's own
  batch-vs-decode floor (0.0076 on the Coder) and top-1 within 1 point.
- Bounds: every position-indexed host entry calls `k3.check_span`; device accesses clamp and set `err`; contexts past the
  allocation are refused, never written.
- Routing ids are clamped on the device before every mm_ids_helper launch; a chunk with its routing flag set is refused.
- Unsafe paths are shown unsafe by CPU emulation only (tests/prefill/test_prefill_safety.py), never by launching them.
- The prefill's TP exchange uses hipMemcpyPeerAsync on copy streams ordered by HIP events only; card 0's partial is added
  first on both cards; HIP graphs never hold the exchange.
- The DFlash ring is 4096 rows at every max_ctx; its slot tags refuse a window reading a stale slot.
- Host page migration makes KFD evict GPU queues: never mmap the PLE shard in a long-lived process (use `AioRows`), never
  do big file I/O while timing.

Process:
- Speed claims: interleaved A/B, median of >= 3 runs, a difference above 0.05 ms; label every speed decode or prefill.
- Unattended agents never run commands that need a permission prompt (deletes, writes to /tmp, sudo, installs, network).
- When an invariant or command here changes, update this file in the same commit.

## Conventions

- Workflow: topic branch -> pull request into `testing` -> `main` for releases (tags). Never open pull requests against
  `main`.
- Commits: `<area> : <what>`, with the gates run and measured before / after in the body; disclose AI assistance with an
  `Assisted-by: <model>` trailer; no session links.
- Scope: MoE / low-active models (AREX-2 is the one dense exception); drafters are DFlash and MTP only.
- Keep architecture constants and model shapes out of shared kernels where it costs nothing; never generalize
  speculatively or at a speed cost before a second target exists.
- Kernel comments carry the measured reason for each choice; keep that density.
- Python runs with `-I`; gguf-py is vendored (`caminus.config.ensure_gguf`).
- Licence: AGPL-3.0-or-later. Outside contributions need the CLA (CLA.md; a bot asks on the first pull request). New third-party code gets a THIRD_PARTY.md
  entry and keeps its notice in the file header, in the same commit.
- Issues: GitHub Issues on this repository; the triage labels are needs-triage, needs-info, ready-for-agent,
  ready-for-human, wontfix.
- Review: Codex reviews every pull request into `testing` (its rules: Code Review Rules below). A P0 / P1 finding is
  resolved before the merge: fixed, or answered in its thread with the reason it does not apply.

## Code Review Rules

Codex flags P0 / P1 issues only. In this repository these are P1:

### GPU safety and correctness
- A change under `kernels/`, or to code that builds a K3 tape (`python/caminus/engine/`, `decode/`, `prefill/`), lists its
  GPU gates and their results in the pull request body: bitwise against the previous reference (T-token verify passes
  equal to one-token steps, and the `K3_PER_WGP=3/6/7` grids), or, where a change cannot be bitwise by design, the KLD
  floor (mean KLD against llama.cpp's one-token decode within llama.cpp's own batch-versus-decode floor, top-1 within one
  point). Flag one that lists none, or that reports k3_run above 96 VGPRs or any VGPR spill.
- Flag a device wait without a time bound (`limit_us`), a GPU allocation that is not projected and checked through
  `caminus.device.capacity` before it happens, and a position-indexed host entry that does not call `k3.check_span`.
- Flag a GPU command in a script, test or doc that does not run under `tools/guard.sh`.

### Public repository
- Flag newly added machine-specific paths (home directories, mount points), credentials or tokens, session links, and
  decision ids (`D-` and a number) in code, comments, docs, commit messages or the pull request body. Comments state
  the reason itself, not a private reference.
- Flag third-party code added without its licence notice in the file header and a THIRD_PARTY.md entry in the same
  pull request.

### Not findings
- Formatting, naming and comment style. The CPU suites, the vendored-code checks and the privacy scan run in CI.
