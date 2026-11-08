# Configuration

By the end of this page you'll know where Caminus keeps its files and how to change any of it. You don't need any of this
to get started: the defaults work.

## Where things go

| What | Default | Change it with |
|---|---|---|
| Model files | `~/.local/share/caminus/models` | `model_dir` / `CAMINUS_MODEL_DIR` |
| More folders to look for model files | none | `model_search` / `CAMINUS_MODEL_SEARCH` (folders separated by `:`) |
| The prompt cache on disk (up to 30 GB, 7 days) | `~/.cache/caminus/kv` | `kv_dir` / `CAMINUS_KV_DIR` |
| The request log and replay files (up to 10 GB, 7 days) | `~/.cache/caminus/flight` | `flight_dir` / `CAMINUS_FLIGHT_DIR` |
| Server logs and state | `~/.local/state/caminus` | `state_dir` / `CAMINUS_STATE_DIR` |
| Qwen Code's home for Caminus | `~/.config/caminus/qwen-home` | `CAMINUS_QWEN_HOME` |
| The panel's settings | `~/.config/caminus/panel.json` | |
| The live monitor's settings, including OpenRGB | `~/.config/caminus/top.json` | `LLMTOP_CONFIG` |

`caminus doctor` prints the paths in effect.

## The config file

Every key is optional. Environment variables win over the file.

```toml
# ~/.config/caminus/config.toml
model_dir = "/data/models/caminus"
model_search = "/data/models:/mnt/big/gguf"
kv_dir = "/fast-ssd/caminus-kv"
```

## Server options

Each model starts with good defaults. To change one, use the panel's advanced layout (`a`, then **OPTIONS**), or pass it to
`caminus serve` (`caminus serve --help` lists them all). A few you may want:

| Option | What it does |
|---|---|
| `--max-ctx N` | A smaller context than the largest that fits. |
| `--no-drafter` | `coder` only: turn off the drafter. |
| `--no-disk-cache` | Don't save conversations to disk. |
| `--idle-park-s N` | Save an idle conversation to disk after N seconds (default 8 hours). |
| `--config FILE` | Read options from a JSON or TOML file (the same names with underscores). |

Serve and replay disable optional ROCm profiler registration by default to avoid a
[runtime crash](troubleshooting.md#while-it-runs). An explicit `HSA_TOOLS_DISABLE_REGISTER=0` enables registration for GPU
profiling; use a runtime containing the upstream fix. An attached `ROCP_TOOL_LIBRARIES` or `HSA_TOOLS_LIB` is also respected.

## Monitoring a server

Each server serves `/health`, `/metrics` (Prometheus), `/slots` and `/caminus/status` (JSON). None of them include the text
of your requests. `caminus top` and the panel's monitor read them.
