# tools/

| Where | What |
|---|---|
| `tools/*` | running Caminus from a checkout: `caminus.py` (every `caminus` command), `caminus_serve.py`, `replay.py`, `flight.py`, `kv_cache.py`; the GPU guard `guard.sh` (every GPU command runs under it); the end-to-end smokes `serve_smoke.py` (the Coder), `serve_smoke1.py` (one-card models), `k3_smoke.py`; capacity and inventory (`k3_capacity.py`, `gg_inventory.py`); a mock server and a prompt renderer (`k3_serve.py`, `serve_render.py`) |
| `gates/` | the bitwise and D-281 gates against the stored references (`gates.sh`) |
| `eval/` | investigations that produce evidence: the long-context sweep and probes, the QSA depth study, the cache-reuse proxy |
| `reference/` | llama.cpp reference builds and dumps (gg_golden, MMQ, sampler, parser, DFlash) the gates compare against |
| `release/` | packaging and the public tree: staging code objects, vendoring, reproducible dist, the public export and privacy scan |
| `windows/` | the GPU steps of a milestone, batched for one user-announced GPU window |

Entry points other files call by path (`guard.sh`, `caminus.py`, `caminus_serve.py`, `serve_smoke*.py`, `replay.py`) stay at
`tools/` itself.
