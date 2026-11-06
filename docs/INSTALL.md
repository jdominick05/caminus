# Installing and running Caminus

Caminus serves one model, the **Qwen3.8-Flash-Next GSQ-RCO Coder**, from its own inference engine on **two AMD gfx1100 GPUs**,
behind the OpenAI and Anthropic APIs on `127.0.0.1:8092`, for coding harnesses on the same machine. This guide takes a
machine from nothing to a running server. Everything in it was done on one machine (RX 7900 XTX + RX 7900 XT, Linux, 29 GiB
of RAM); what was not tried is marked.

## 1. What you need

| | |
|---|---|
| GPUs | two gfx1100 cards (RDNA3, Navi 31) with PCIe peer-to-peer between them. Tested: an RX 7900 XTX (24 GB) as device 0 and an RX 7900 XT (20 GB) as device 1. Device 0 must be the larger card; the VRAM budget's fixed parts were measured on that pair, so another pair may be refused at startup rather than guessed |
| VRAM in use | at the default 262,144-token context both cards are full while it runs: ~23.1 GB on the XTX, ~19.0 GB on the XT. Nothing else can use the GPUs meanwhile |
| OS | Linux with the amdgpu kernel driver and `/dev/kfd` (ROCm's kernel side); your user in the `render` and `video` groups |
| Python | 3.12 (the only version tested), in a virtual environment |
| PyTorch | the **ROCm build** (tested: 2.14.1+rocm7.2). It is not on PyPI, which is why `caminus` does not declare it |
| Host RAM | ~29 GiB is enough: the 28.8 GB n-gram table is read from disk with O_DIRECT, never mapped |
| Disk | 59 GB for the model files; up to 30 GB for the prompt cache's disk tier and 10 GB for the flight recorder (under `~/.cache/caminus`) |
| Kernel log | readable by your user, for the guard's fault watch (section 6): the `systemd-journal` or `adm` group, or `kernel.dmesg_restrict = 0` |

## 2. Install

```bash
python3.12 -m venv ~/caminus-venv
. ~/caminus-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2    # PyTorch's ROCm build
pip install caminus
caminus version          # the package version and where its code objects come from
caminus doctor           # PASS / WARN / FAIL for this machine (CPU only; safe beside a running server)
```

`caminus doctor` checks Python, the package and every code object against its provenance, PyTorch (a ROCm build, or the
command to get one), the gfx1100 cards (count, VRAM in use, other GPU processes, the link), the clocks, the guard (its lines,
gpu-high, the STOP file, whether it can read the kernel log), host RAM, the model files, disk space and port 8092. Fix every
FAIL before you start the server; read every WARN. `caminus doctor --gpu` also runs one small kernel on each card and checks
peer access (under the guard).

The package ships prebuilt gfx1100 code objects (`caminus/hsaco/gfx1100`, with `provenance.json`: the toolchain, every
object's compile command, the source tree's content hash). Nothing is compiled at install. To build them yourself, see
section 10.

## 3. The model files

```bash
caminus download --dry-run      # what would be fetched, and where
caminus download                # 58.4 GB (the Coder, two shards) + 0.65 GB (the DFlash2 drafter)
caminus download --verify       # later: hash the files against the manifest (~1 min per 30 GB)
```

| File | Repository (pinned revision) | Size |
|---|---|---|
| The Coder, shard 1 (weights, held in VRAM) | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`, `IQ1_M/...-00001-of-00002.gguf` | 29.6 GB |
| The Coder, shard 2 (the n-gram table, read from disk) | the same repository, `...-00002-of-00002.gguf` | 28.8 GB |
| The DFlash2 drafter, Q8_0 | `incoai/Qwen3.8-Flash-Next-DFlash2-GGUF`, `Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf` | 0.65 GB |

- Downloads resume (`.part` files and HTTP Range), and only a file whose size and sha256 match the manifest is renamed into
  place. `HF_TOKEN` and `HF_ENDPOINT` are read as huggingface_hub reads them.
- The files go to `~/.local/share/caminus/models` (`CAMINUS_MODEL_DIR` or `--dir` for another place); `caminus serve` finds
  them there by default.
- **Pinned and checked:** each entry names a revision, a size and a sha256 (python/caminus/models.py); on 2026-11-06 Hugging
  Face's API listed all three files at those revisions with exactly those sizes and LFS sha256 (both repositories public, not
  gated). Without the drafter, `caminus serve --no-drafter` serves plain decoding (about 104 tok/s instead of 123-203).
- **Licences:** read each model's terms before you use it (the base model's `qwen-community-1.0`; the Coder pack's card says
  Apache-2.0 in its tag and the base model's licence in its text; the drafter is Apache-2.0). THIRD_PARTY.md, last section.

## 4. Configuration

Paths come from the environment first, then `~/.config/caminus/config.toml`, then the defaults. `caminus doctor` prints the
ones in effect. A config file holds string values for these keys:

```toml
# ~/.config/caminus/config.toml (all optional)
model_dir  = "~/.local/share/caminus/models"   # CAMINUS_MODEL_DIR: where caminus download puts the files
coder      = "/path/to/...-00001-of-00002.gguf"   # CAMINUS_CODER: the Coder's first shard (shard 2 is found beside it)
dflash     = "/path/to/Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf"   # CAMINUS_DFLASH
kv_dir     = "~/.cache/caminus/kv"             # CAMINUS_KV_DIR: the prompt cache's disk tier
flight_dir = "~/.cache/caminus/flight"         # CAMINUS_FLIGHT_DIR: the flight recorder
state_dir  = "~/.local/state/caminus"          # CAMINUS_STATE_DIR: the guard's logs
```

The server's own options (`caminus serve --help`) can also come from a JSON or TOML file (`caminus serve --config FILE`, the
same names with underscores); flags override it.

## 5. Run the server

```bash
caminus serve &                                  # starts itself under the guard; the server's output goes to its log
tail -f ~/.local/state/caminus/serve.log         # ~50 s: "loading", then "ready on http://127.0.0.1:8092"
```

```bash
curl -s 127.0.0.1:8092/health                    # {"status": "ok"}; 503 "Loading model" while the engine builds
curl -s 127.0.0.1:8092/v1/models                 # k3-coder, k3-coder-nothink, k3-coder-xhigh, k3-coder-low
curl -s 127.0.0.1:8092/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model": "k3-coder-nothink", "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 32}'
```

**Stop it with SIGTERM to the server** (the process names itself `caminus`):

```bash
kill -TERM $(pgrep -x caminus)
```

It finishes the running request at its next pass (at most 60 s), parks the live conversation to disk (at most 90 s), then
exits; the guard then prints its last line (the exit code and the VRAM peaks) where it was started and exits too. **Ctrl-C
and closing the terminal do not stop the server:** the guard runs it in a session of its own, so they end the guard and leave
the server running without its watch. Always stop it with the `kill -TERM` above.

**What it serves:**
- `/v1/chat/completions`, `/v1/completions` (OpenAI) and `/v1/messages`, `/v1/messages/count_tokens` (Anthropic), streaming
  or not, with tool calls.
- **Model profiles** (the `model` field): `k3-coder` thinks at medium effort, `k3-coder-nothink` never thinks,
  `k3-coder-xhigh` and `k3-coder-low` at other efforts. The effort is fixed by the model id because it is the first line of the
  prompt (changing it would drop the cache). On the Anthropic route `k3-coder` thinks only when the request carries `thinking`.
- **Context: 262,144 positions** (`--max-ctx`), the model's own cap. A prompt that does not fit gets HTTP 400 ("prompt is too
  long"), so harnesses compact. A cold 262K prompt takes ~110 s before its first token.
- **One conversation at a time.** The prompt cache keeps it in VRAM checkpoints, so the next turn reads only its new tokens
  (first token in ~0.2 s for an 11K-token conversation). Short requests unrelated to the conversation (a harness's title or
  summary call that sets `max_tokens`) run in a side slot and never displace it. A conversation you switch away from, one
  idle for 8 hours, and the live one at a clean stop are parked to disk and come back in about a second, also after a restart.
- **DFlash drafting** when it pays (the server measures it per content class and depth); `--no-drafter` turns it off.
- **Monitoring:** `/health`, `/props`, `/slots` (prompt progress), `/metrics` (Prometheus, with llama-server's `llamacpp:`
  names plus `caminus:`), `/caminus/status` (JSON: the engine, both cards, the cache, the drafter, reliability). None of them
  carries request text.
- **No API key is checked.** The server listens on 127.0.0.1 only; do not expose it.

## 6. The guard (GPU safety)

One of your cards most likely drives the monitors. A GPU fault there can take the desktop down, and re-running a faulting job
can make it worse, so every GPU run of Caminus goes through a guard: `caminus serve`, `caminus replay` and `caminus doctor --gpu`
start themselves under it, and `caminus guard LOG SECS CMD...` runs any other command under it (`SECS` 0: no time limit).

- **What it watches:** every gfx1100 card. It kills the run when a card passes 0.97 of its VRAM (in 50 MiB steps: 23,800 MiB on
  a 24 GB card, 19,850 on a 20 GB one; `caminus guard --lines` prints them for yours) or host MemAvailable falls below 1,500 MiB,
  and on any amdgpu page fault, GPU reset, wedge or timeout in the kernel log.
- **The kernel log:** read with `journalctl -k` when your user may (the `systemd-journal` or `adm` group), else `dmesg`. If it
  can read neither, it says so at the start (`GUARD WARNING`): the fault watch is then off. Add yourself to one of those groups
  (and log in again) to turn it on.
- **Exit 99: a GPU fault.** Stop all GPU work and re-run nothing; read `journalctl -k` around that time. The guard records the
  kernel-log lines in the STOP file, `~/.local/state/k3/GPU_FAULT_STOP`.
- **Exit 98: refused.** A fault was recorded earlier. Every guarded run refuses until you have looked at the fault and removed
  the STOP file by hand. Exit 97: no gfx1100 card to watch.
- **Clocks:** decode is ~13 % slower at the default DPM level `auto` than at `high`. The guard runs the command under a
  `gpu-high` helper when it finds one (`CAMINUS_GPU_HIGH`, else `gpu-high` on `PATH`, else `~/.local/bin/gpu-high`); without one
  it says so and runs at the current clocks. No helper is shipped: it is any executable that sets each card's
  `power_dpm_force_performance_level` to `high` and `power/control` to `on`, runs the command it is given, and sets them back
  afterwards. Writing those sysfs files needs root, so such a helper needs its own narrow sudo rule or udev rule; that is your
  call. The server logs a clock check at startup (`/caminus/status` `clock_check`).
- **What the guard writes:** the server's output to `~/.local/state/caminus/serve.log` (rewritten at each start) and a 1 Hz VRAM
  / RAM line to `serve.log.mon` beside it. The `.mon` file grows ~2.5 MB a day while the server runs and is never rotated:
  truncate it while the server is stopped (`: > ~/.local/state/caminus/serve.log.mon`).

The server has its own limits too: everything it allocates is counted before it allocates, against the guard's lines, with a
512 MiB reserve on the display card for the desktop's own growth; every device wait is time-bounded with a bitwise retry; a
context beyond the allocation is refused, never written.

## 7. Harnesses

Example configs and wrappers are in the repository, not in the package: clone it (`git clone https://github.com/jdominick05/caminus`)
or copy the files from `packaging/harness/`. Each wrapper points its harness's documented home variable at a config home of its
own under `~/.config/caminus/`, created on the first run from the example beside it, so your own settings, sessions and
credentials stay where they are (and your own providers, hooks and MCP servers do not apply there: copy what you want). The
harness itself must be installed.

| Harness | Run | Profiles |
|---|---|---|
| Qwen Code | `packaging/harness/qwen-code/qwen-k3` (sets `QWEN_HOME`; `settings.json`: the OpenAI provider at `http://127.0.0.1:8092/v1`, a 262,000-token window, 15-minute timeouts) | main `k3-coder`; `/model k3-coder-nothink` (no thinking), `/model k3-coder-low` / `/model k3-coder-xhigh` (low / extra-high effort); the fast model `k3-coder-nothink` |
| Hermes Agent | `packaging/harness/hermes/hermes-k3` (sets `HERMES_HOME`; `config.yaml`: the named provider `custom:caminus`, context 262,144) | `k3-coder`; `/model custom:caminus:k3-coder-nothink` (no thinking), `custom:caminus:k3-coder-low` / `-xhigh` (low / extra-high effort); titles on `k3-coder-nothink` |
| Claude Code | `packaging/harness/claude-code/claude-k3` (sets `CLAUDE_CONFIG_DIR`, `ANTHROPIC_BASE_URL=http://127.0.0.1:8092`, a dummy token; unsets `ANTHROPIC_API_KEY` so a real key is never sent) | main `k3-coder` (`CAMINUS_MODEL=k3-coder-nothink` for no thinking), the small model `k3-coder-nothink`; `MAX_THINKING_TOKENS=16000` asks Claude Code to send `thinking` |

None of the harnesses has been run against the server yet: the wrappers follow each harness's own documentation, and
`tools/serve_smoke.py` exercised the same request shapes. If a harness's side calls come without `max_tokens`, they run in
the main slot and push the conversation to disk (it comes back in about a second): `/caminus/status` -> `cache.route` shows
where requests went.

## 8. Watching it: llm-top

[llm-top](https://github.com/jdominick05/llm-top) has a Caminus panel (speed against the bandwidth floor, the ms/token and
TTFT splits, DFlash acceptance per content class, the cache tiers and routes, both cards, reliability). The Caminus backend
is on llm-top's GitHub main branch (since a81d51f, 2026-11-06): build llm-top from main (its README's Caminus section; an
llm-top built before that has no Caminus panel), then:

```bash
llm-top -config packaging/harness/llm-top/llm-top-caminus.json
```

## 9. The prompt cache, the flight recorder, replay

```bash
caminus kv-cache list                       # what the disk tier holds (stop the server first: one owner at a time)
caminus kv-cache purge [--older-than-days 3]
caminus flight list                         # the flight recorder: one JSONL line per request (no request text) and bundles
caminus flight bundle                       # ask the running server for a replay bundle of the latest request
caminus replay ~/.cache/caminus/flight/bundles/<bundle>    # rerun it offline, bitwise (GPU, under the guard; server stopped)
caminus purge                               # lists both directories; --yes deletes
```

The disk tier caps itself at 30 GB with LRU, keeps 7 days and keeps 10 GB of the disk free; the flight recorder caps at 10 GB
and 7 days. Bundles hold the token ids of your conversations: they stay local, and the purge removes them. The server saves
one automatically after a pass that failed its retries, a reasoning loop or a tool-call parse failure. A bundle of another
build is refused (`RESULT KEY MISMATCH`): the cache key hashes the code objects and the package's sources, so upgrading the
package also invalidates older disk-cache entries.

## 10. Troubleshooting

- **The capacity check refuses at startup:** something else holds VRAM (another engine, a previous server still parking).
  `caminus doctor` lists the cards' VRAM in use and other GPU processes; both should be near idle before a start.
- **A VRAM kill during a session:** the cache sizes its VRAM tier at startup from what is free, keeping 512 MiB below the
  display card's line. If the desktop's own VRAM grows by more than that afterwards (a browser, a game, a remote-desktop
  session), the guard kills the server. Start the server after the desktop session is up, and restart it after such a kill. A
  smaller `--max-ctx` does not leave the desktop more room (the tier takes it), and there is no flag for a bigger reserve yet.
- **Slow decode:** the clocks (section 6). After a runtime suspend a card can report `high` while running at auto clocks: the
  startup clock check in the log says which; restart under the guard if it is not OK.
- **"Loading model" (503)** for the first ~50 s: the engine builds (faster when shard 1 is in the page cache).
- **503 / 529 after an engine error:** a pass failed its bounded retries and could not be recovered from the request's
  checkpoint; the server stays up and the next request starts from a cached state or cold. A replay bundle was saved.
- **Nothing arrives during a long prompt:** OpenAI streams send nothing before the first token (a 100K prompt reads for ~40 s);
  Anthropic streams send a ping every 10 s. `/slots` shows the progress.
- **Disk space:** `caminus purge --yes` empties the disk tier and the flight recorder (stop the server first).

## 11. Building from source

For contributors, or to rebuild the code objects:

- **TheRock ROCm 10.2** (tested: the nightly 10.2.0a20261011; `ROCM_PATH`, default `~/rocm-therock/10.2.0a20261011`):
  amdclang++ and the offload driver's device LTO. A different compiler does not give the same code.
- **The ignite-llama fork of llama.cpp at commit e4e3f8eda** ([github.com/jdominick05/ignite-llama](https://github.com/jdominick05/ignite-llama),
  `IGNITE`, default `~/src/ignite-llama`): build.sh compiles against its ggml headers (the K3 modules use helpers the fork
  adds to `ggml-cuda/common.cuh`; upstream llama.cpp's headers at the fork's base, 5aeac37a35, stop at `quantize_q8_1_lane`),
  the vendored MMQ in `kernels/ggml_mmq` comes from it (`tools/vendor_mmq.py --check` compares against it), and its HIP build
  is the reference of the bitwise gates. The development builds and gates ran on the fork's commit 842653396, published as
  e4e3f8eda (identical `ggml/`, `common/` and `src/`; only its README, docs and perf scripts differ): the vendored files'
  headers and older notes name 842653396. Installing and running the package does not need the fork (the wheel carries
  the built code objects).

  ```bash
  git clone https://github.com/jdominick05/ignite-llama && git -C ignite-llama checkout e4e3f8eda
  ```
- The C++ runtime tests link PyTorch's bundled HIP runtime and rocBLAS (`TORCHLIB`, `PATCHELF`: build.sh's head).

```bash
git clone https://github.com/jdominick05/caminus && cd caminus
export ROCM_PATH=/path/to/therock IGNITE=/path/to/ignite-llama PATCHELF=$(command -v patchelf)
export TORCHLIB=$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
./build.sh                                # every kernels/*.hip -> build/*.hsaco, then the C++ runtime tests
python -I tools/stage_kernels.py          # the runtime set + provenance.json into python/caminus/hsaco/gfx1100
python -I -m pip wheel . --no-deps --no-build-isolation -w dist
```

From a checkout without installing, `CAMINUS_KERNEL_DIR` is not needed: the package imported from `python/` finds `build/`;
`python -I tools/caminus_serve.py` runs the server (under `tools/guard.sh LOG 0 ...`). CLAUDE.md lists the test suites (the CPU
ones run without the GPUs) and the rules for GPU work.

## 12. Licences

Caminus is Copyright (C) 2026 jdominick05, licensed under the GNU AGPL v3.0 or later (LICENSE): if you run a modified version
as a network service, its users are entitled to its source. Third-party code and its notices: THIRD_PARTY.md. The model files
are not part of Caminus and stay under their own licences (section 3).
