# tests/

Plain scripts, no pytest: each prints `PASS` / `FAIL` lines and ends with `RESULT ...`; read a script's docstring for its
arguments (most take the GGUF path). The directories mirror the package (`python/caminus/<area>`):

| Directory | What it checks | Runs on |
|---|---|---|
| `engine/` | the K3 decode engines: ops and formats vs llama.cpp's tensors, bitwise gates, host and device bounds, the pass police | CPU (bounds, police, golden_cpu, ple_aio) and GPU |
| `prefill/` | the batched prefill: each launcher's refusals (`cpu` stages), each op bitwise its decode op, the drivers' G2b / G3 / G4 gates | CPU and GPU |
| `decode/` | drafters and the speculative loop: DFlash ring fuzz, PolicyDrafter, long-context fills | CPU and GPU |
| `cache/` | the prompt cache: a CPU stand-in engine (`cpu`, every shape via `--cards 1\|2 --t 1\|4`) and the GPU tiers | CPU and GPU |
| `serve/` | the server front end (unittest, `run_all.py`): template, tokenizer, tool parser, sampler, metrics, worker, the session contract (`test_session.py`: `contract_problems`, the monitoring surface pinned in `surface_keys.json`) | CPU |
| `e2e/` | the real server on the real engine (`test_serve_k3.py`) | GPU |
| `device/` | the toolchain smoke, the WMMA GEMM, the sampler's own top-k (`test_topk.py`) | GPU |
| `cmd/`, `packaging/`, `tools/` | the management CLI, the installable package, tool self-tests | CPU |
| `test_arch.py` (top level, as `caminus/arch.py` is) | the architecture registry and its readers: the catalog, caminus serve's options, caminus start's flags | CPU |
| `docs/test_docs.py` | the docs check: every relative link and #anchor in README / CONTRIBUTING / GLOSSARY / docs/*.md / docs/dev/*.md resolves, every `caminus COMMAND` the user docs show exists, every script the public AGENTS.md names exists | CPU (stdlib) |
| `device/test_host.py` | caminus.device.host: the card inventory (status, the PCIe link's slowest hop, HIP order), placement (pair, split, one, place: a given HIP_VISIBLE_DEVICES rewritten into engine order) and the hardware lines, on fixture trees of every layout (one card, any pair, three cards, a slow link, unsupported cards, headless, two displays) | CPU |
| `train/` | the dormant S1 training stack | GPU, needs the private trainer |

`ci_cpu.py` runs every CPU suite at once, as CI does. GPU scripts run only under `tools/guard.sh` (CLAUDE.md, GPU safety).
