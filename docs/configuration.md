# Configuration

By the end of this page you'll know where Caminus keeps its files and how to change any of it. You don't need any of this
to get started: the defaults work.

## Where things go

| What | Default | Change it with |
|---|---|---|
| Model files | `~/.local/share/caminus/models` | `model_dir` / `CAMINUS_MODEL_DIR` |
| More folders to look for model files | none | `model_search` / `CAMINUS_MODEL_SEARCH` (folders separated by `:`) |
| The prompt cache on disk (up to 64 GB for the whole folder, 7 days) | `~/.cache/caminus/kv` | `kv_dir` / `CAMINUS_KV_DIR` |
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
| `--max-ctx N` | A smaller context than the largest that fits. Above the model's own context it turns on the long-context mode where the model has one (below). |
| `--long-context` | `qwen3-30b-a3b`: YaRN x4, up to 131,072 tokens (about 80K on one XTX, with 8-bit cache rows where only they fit; 131,072 on two cards). Short texts can read slightly differently, so it is off by default. `coder`: needs a `--max-ctx` above 262,144. Other models refuse it. |
| `--cards 2` | Run a one-card model across two cards (tensor parallel): `swift-1.5`, `thinkingcap` or `qwen3-30b-a3b`. Without it each model runs on its own card count. For `qwen3-30b-a3b` one card stays the default: two cards write more slowly below about 16K tokens of context and faster past it, and the context stays 40,960 tokens either way (the model's own limit). The bigger card is card 0: it holds two thirds of every expert. `caminus fit --cards 2` reports the context two cards hold. |
| `--kv q8` | `qwen3-30b-a3b` only, on one card or two: an 8-bit attention cache instead of F16, the default. It takes about half the memory per token of context (0.050 MiB instead of 0.094), so the full 40,960 tokens still fit when the desktop or another program holds part of the card; decode runs at about the same speed. A conversation saved by a server with one format is never restored by a server with the other: it is read again. Every other model refuses it (the dense 27B models keep F16, the Coder its own 8-bit cache). `caminus start --kv q8` takes it too, and `caminus fit` reports the context that fits with each cache; `caminus ps` shows the cache a server holds. See [models](models.md#the-8-bit-kv-cache-qwen3-30b-a3b). |
| `--kv-host` | `coder` only: keep the attention cache in system memory (the cards read it directly). Same results, less VRAM. |
| `--no-drafter` | `coder` only: turn off the drafter. |
| `--draft KIND` | Speculative decoding: `off` (plain), `dflash` (the DFlash2 drafter: the Coder's sidecar, needs `--drafter`; on `swift-1.5` and `thinkingcap`, on one card or two, their base model's drafter file from the model directory, `Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q8_0.gguf`, or `--drafter`), or `mtp` (the model's own prediction block: `swift-1.5` or `thinkingcap`, on one card or two). Default: `dflash` for the Coder when its file is there, and for `swift-1.5`, `thinkingcap` and `qwen3.8-27b` on one card when the shared DFlash2 file is there and 32,768 tokens of context (or your `--max-ctx`) still fit with it; otherwise `off`, with the full context. On a 24 GB card that means `qwen3.8-27b` and `thinkingcap` draft by default and `swift-1.5` does not (the drafter would leave it about 12-16K tokens); `--draft dflash` runs it anyway. On two cards: `dflash` for `swift-1.5` and `thinkingcap` under the same rule (the drafter's memory comes out of the two-card context). `off` for every other model. |
| `--no-disk-cache` | Don't save conversations to disk. |
| `--cache-cap-gb N` | The most the saved conversations may take on disk, every model's together (default 64 GB; at most a quarter of the disk). The oldest are removed first, another model's too while its server is stopped; a running server's saves never. If you switch models often, raise it. |
| `--cache-keep-per-root N` | Saved copies kept per conversation (default 3): a long conversation whose start changed (compaction) keeps only its newest copies. `0` keeps all. |
| `--no-loop-guard` | Allow sustained repeated answer prose when repetition is intentional. |
| `--idle-park-s N` | Save an idle conversation to disk after N seconds (default 8 hours). |
| `--config FILE` | Read options from a JSON or TOML file (the same names with underscores). |

### Long context (`qwen3-30b-a3b`, up to 131,072 tokens)

`--long-context` (or `--max-ctx` above 40,960) serves the model card's YaRN recipe: positions scaled by 4 from the
native 32,768 tokens, up to 131,072. It is opt-in: without it the server is unchanged (40,960 tokens, no scaling).

- The context: one XTX holds about 80K tokens, with the attention cache as 8-bit rows (the server picks them only where
  the 16-bit cache would hold less, and logs it); two cards hold 131,072 with the 16-bit cache. `caminus fit
  --long-context` reports what your cards hold.
- YaRN applies at every position, so short texts can read slightly differently than in the default mode (the model card
  advises it only past 32,768 tokens).
- A conversation saved in one mode is never restored into the other.
- `caminus start qwen3-30b-a3b --long-context` passes it to the server.
- The recipe is for Qwen3-30B-A3B's own file (its header says 40,960 tokens and RoPE base 1,000,000). Another file of the
  same architecture, such as the 2507 releases or Qwen3-Coder-30B-A3B (262,144 tokens natively, another base), is refused
  in this mode, as is a file with codebook (IQ) or row-scaled tensors; without the flag those files serve as before.

### Long context (`coder`, up to 1M tokens)

`--max-ctx` above the model's own 262,144 tokens turns on the long-context mode, for example `caminus serve --max-ctx
1048576`. It is opt-in, and a server at 262,144 or less is unchanged. What changes:

- The attention cache moves to pinned system memory: about 13 GB at 1,048,576 tokens, split over the two cards. The server
  checks free memory and the GPU's system-memory limit before it allocates, and refuses to start when they're short.
- Positions are stretched (NTK RoPE scaling, factor 4) so the model can recall past its training length. In our tests it
  found 3 of 3 planted values at 1,040,384 tokens, including one 936K tokens back; without the scaling it missed that one.
  The stretch costs a little at short range: KL divergence 0.0077 against llama.cpp, versus 0.0057 without it.
- Conversations are still saved to disk (since 0.2.2; 0.2.1 turned this off in this mode). Saving one writes its whole
  attention cache, about 12.4 KB per token (5 GB at 400K tokens), so the disk cap is 64 GB here unless you set
  `--cache-cap-gb`. When another conversation pushes yours out (a second session, or a harness's background job too
  big for the side slot), yours comes back from disk in seconds instead of being read again: about 1,100 tokens/s at
  350K and 770 tokens/s at 1M, so a full 1M prompt read from scratch takes about 22 minutes. A harness timeout shorter
  than that ends a cold request at that size. A conversation that grows turn by turn only processes its new tokens.
- Decoding is slower at depth: about 34 tokens/s at 1M without the drafter. With it, short tests ran at 50 to 85
  tokens/s at 350K to 1M, depending on how predictable the reply is.

Serve and replay disable optional ROCm profiler registration by default to avoid a
[runtime crash](troubleshooting.md#while-it-runs). An explicit `HSA_TOOLS_DISABLE_REGISTER=0` enables registration for GPU
profiling; use a runtime containing the upstream fix. An attached `ROCP_TOOL_LIBRARIES` or `HSA_TOOLS_LIB` is also respected.

## Monitoring a server

Each server serves `/health`, `/metrics` (Prometheus), `/slots` and `/caminus/status` (JSON). None of them include the text
of your requests. `caminus top` and the panel's monitor read them.
