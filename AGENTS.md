# Repository Guidelines

Caminus is an inference engine for two AMD gfx1100 GPUs (K3, a persistent decode megakernel) serving the Qwen3.8-Flash-Next
GSQ-RCO Coder behind an OpenAI / Anthropic API. **CLAUDE.md is the full guide** (layout, paths, commands, invariants); this file
is the short version for other agents and defers to it.

## Structure

- `kernels/`: HIP device code; `./build.sh` writes `build/*.hsaco` (ignored) and the C++ runtime tests.
- `python/caminus/`: the host engine, the batched prefill, the prompt cache, the server (`serve/`), the console commands
  (`cmd/`, `__main__.py`), the config layer (`config.py`) and the shipped guard (`share/guard.sh`).
- `tests/`: standalone scripts (PASS / FAIL, most end with `RESULT PASS`); `tests/serve/`: unittest, through `run_all.py`.
- `tools/`: launchers from a checkout, benchmarks, reference builders, packaging helpers, `gates/`.
- `docs/`: INSTALL.md (users), BENCHMARKS.md (where each number comes from), the engine design and specs.

This repository is a squashed snapshot of a private development history: commit hashes, D-numbers, step names and plan
documents cited in comments refer to it and are not in this tree.

## Build and test

Use a Python 3.12 environment with ROCm PyTorch, TheRock ROCm 10.2 (`ROCM_PATH`) and the ignite-llama fork at e4e3f8eda
(`IGNITE`, for its ggml headers: github.com/jdominick05/ignite-llama; docs/INSTALL.md) to build. Export the model paths first (CLAUDE.md, "Models and data": `CODER`, `DFLASH`,
`ARTIFACTS`, optional `REAP`, `MTP`, `HF_TOKENIZER_JSON`); tests and tools read nothing else.

```bash
./build.sh
HIP_VISIBLE_DEVICES= python -I tests/serve/run_all.py        # CPU server suite (not beside a build or a GPU job: one test is timing-sensitive)
HIP_VISIBLE_DEVICES= python -I tests/test_packaging.py --scratch DIR
python -I tests/ci_cpu.py --scratch DIR      # all CPU suites as CI runs them (skips what this machine lacks, with the reason)
python -I tools/privacy_scan.py .           # the scan CI runs: no secrets, machine paths or session links in the tree
HIP_VISIBLE_DEVICES= python -I tests/test_bounds_host.py
tools/guard.sh LOG 600 python -I tools/k3_smoke.py --model coder   # GPU: always under the guard
```

## Safety rules

- Every GPU command runs under `tools/guard.sh LOG SECS CMD...` (or `caminus guard`). Exit 99 is a GPU fault: stop all GPU work
  and re-run nothing. Exit 98: a fault was recorded earlier (`~/.local/state/k3/GPU_FAULT_STOP`); a person inspects it first.
- Device 0 drives the monitors: keep its VRAM under the guard's line, run new kernel code on device 1 alone first
  (`HIP_VISIBLE_DEVICES=1`), keep every device wait time-bounded and every launch well under 200 ms.
- K3 changes stay bitwise against the stored references (tools/gates); non-bitwise-by-design changes pass the D-281 KLD floor
  gate. Never refresh a reference to make a gate pass.
- Unattended runs never need a permission prompt: no deletes, no writes under `/tmp`, no sudo, no installs, no network.

## Commits

`<area> : <change>` (for example `tests : cover attention bounds`), with validation results and measured before / after
numbers in the body; disclose AI assistance with an `Assisted-by: <model>` trailer, no session links. The project is
AGPL-3.0-or-later; third-party code gets an entry in `THIRD_PARTY.md` and keeps its notice in the file header. Outside
contributions need a contributor licence agreement (CLAUDE.md, Conventions). Keep this file and `CLAUDE.md` in step.
