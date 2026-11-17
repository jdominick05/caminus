# Troubleshooting

By the end of this page you'll know what an error means and what to do about it. Start with `caminus doctor`: it checks
everything below and prints PASS, WARN or FAIL for each.

If nothing here helps, please [open an issue](https://github.com/jdominick05/caminus/issues/new/choose) and paste the full
output of `caminus doctor`.

## Reading `caminus doctor`

`caminus doctor` reads your machine (the system files about your cards, the installed package, the model files) without
starting anything on a GPU. Every line starts with a level and an area:

- **PASS** fine, **INFO** a fact for a bug report, **WARN** works but slower or riskier, **FAIL** fix it before you start
  a model. The last line, `RESULT`, is the worst level it saw.
- `caminus doctor --verify` also hashes the model files (about a minute per 30 GB); `caminus doctor --gpu` also runs a
  tiny test kernel on each card, under the guard; `--json` prints the same as JSON.

| Area | What it checks | When it is not PASS |
|---|---|---|
| `python`, `platform` | Python 3.12 or newer, Linux | Install Python 3.12 and make the venv with it (`python3.12 -m venv`). |
| `package` (INFO) | The Caminus version and where it is installed | Paste it in bug reports. |
| `kernels` | The GPU code that ships with the package, against its own checksum list | FAIL "missing N of the runtime set" or "bad": a broken or partial install: reinstall (`pip install --force-reinstall caminus`, or rebuild the wheel). In a source checkout without a build, run `./build.sh`. WARN "uncommitted kernel sources": a development build. |
| `deps` | numpy, jinja2 (new enough for its sandbox), tokenizers | `pip install` the one it names (`pip install -U jinja2` for an old one). |
| `gguf` | The model-file reader | A broken install: reinstall Caminus. |
| `torch` | PyTorch's AMD (ROCm) build, and the version Caminus was tested with | Install the ROCm build first: `pip install torch --index-url https://download.pytorch.org/whl/rocm7.2`. PyPI's default PyTorch is for NVIDIA cards. |
| `gpu` | The supported cards, their memory and links; what runs on one card or on the pair; other GPU programs | FAIL "no supported card": see [Your cards](#your-cards). WARN "something holds VRAM" or "other process(es) hold a GPU context": another engine, a test or a game is using the card; Caminus wants it idle (`caminus ps` shows Caminus's own servers). WARN "the Coder needs a 24 GB RX 7900 XTX as card 0": your pair runs everything but the Coder. |
| `clocks` | Whether the cards run at high clocks for full speed | WARN: about 13% slower; see [getting full speed](hardware.md#getting-full-speed). |
| `guard` | The GPU guard script, its memory lines per card, a recorded fault, the kernel log | FAIL "a GPU fault was recorded": see [Exit 99](#while-it-runs) below. WARN "kernel log not readable": join the `systemd-journal` group and log in again, so the guard can watch for faults. |
| `host` | System memory | WARN below 28 GiB: Caminus was tested with 29 GiB, and the guard stops a run when less than 1,500 MiB is left available; close other programs, or pick a smaller model. |
| `models` | Each model file you have: present, the size the release was tested with (`--verify`: its hash) | FAIL "does not exist": download it (the panel's `p`, or `caminus pull MODEL`). FAIL on size or hash: a damaged or different file: download it again. |
| `disk` | Free space where the models, the prompt cache and the request log live | WARN: free some space, or point them elsewhere ([configuration](configuration.md)); `caminus purge` lists what the caches hold. |
| `port` | Whether port 8092 is free | WARN: a server already runs there (`caminus ps`), or start the next one with `--port`. |
| `config` (INFO) | Every setting and where it came from | Paste it in bug reports. |

## Installing

**"PyTorch is not installed" or "not a ROCm build"**: install PyTorch's AMD build before Caminus:
`pip install torch --index-url https://download.pytorch.org/whl/rocm7.2`. The default PyTorch from PyPI is built for
NVIDIA cards and won't work.

**"caminus-panel is not installed"**: your Caminus package came without the panel. Reinstall it with
`pip install --force-reinstall caminus`, or use the [commands](cli.md) instead.

**Qwen Code won't install**: `i` in the panel and `caminus qwen-code setup --install` need Node.js 22 or newer with npm.

## Your cards

**"no supported GPU"**: Caminus didn't find an RX 7900 XTX or XT. Check that `ls /dev/kfd` works and that you're in the
`render` and `video` groups (log out and back in after adding yourself). Other cards: see [hardware](hardware.md).

**"two-card models refused on this pair"**: one of the cards sits on a link slower than PCIe 4.0 x4, or the board doesn't
allow direct card-to-card access. Move the card to a faster slot. One-card models still run. When the graphics driver
reports that it has lowered an idle card's link to save power, the check judges the link at full speed, and
`caminus doctor` shows the idle reading as "idle now at".

**"the fault watch is off" (GUARD WARNING)**: the guard can't read the kernel log. Add yourself to the `systemd-journal`
group and log in again.

## Starting a model

**"would use ... MiB > its cap" when you start**: something else is using the card: another model, a game, or a server that's still
shutting down. `caminus ps` and `caminus doctor` show what's using each card. Stop it and try again.

**"Loading model" (HTTP 503)** for the first minute: the model is still loading. `caminus logs NAME` shows its progress.

**The port is in use**: another program took port 8092. `caminus start` and the panel take the next free port (`caminus ps`
shows which); `caminus serve --port N` picks one yourself.

## While it runs

**The guard stopped the server for using too much VRAM**: on the card that drives your monitors, the desktop grew by more
than the 512 MB Caminus keeps free for it (a browser or game can do that). Restart the model after closing what grabbed
the memory.

**Exit 99, or "GPU FAULT"**: the GPU reported a fault. Stop all GPU work and don't rerun the same thing. Caminus refuses to
start anything (exit 98) until you've looked at it: read `journalctl -k` around that time, then delete the stop file
`~/.local/state/k3/GPU_FAULT_STOP`. Please report it with the doctor output and those kernel log lines.

**Exit 75, or "ENGINE BROKEN" / "engine broken (restart the server)"**: the model produced NaN or infinite logits, which
means a weight or state in that server process is corrupt (seen once: every reply was the token `!` repeated). The server
stops answering (`/health` returns 503 with the reason), skips parking the conversation so the bad state never reaches the
disk cache, and exits 75. `caminus start` and the systemd unit reload it once by themselves; a fresh load uploads the
weights again. If it breaks again right away, the log line names the weight that changed (`changed on the card since load`
or `differ from the file`): please report it with that line and `caminus doctor`.

**"Segmentation fault" or "Fatal Python error"**: guarded Python commands enable fatal-error tracebacks. Save
`caminus logs NAME` before restarting the server (a restart replaces its log), and include the traceback and kernel log
lines from that time in your bug report. A native crash may leave no Python traceback if it happens on a runtime thread.

**A native traceback names `rocr::core::Runtime::AsyncEventsLoop`**: some ROCm 7.2 runtimes bundled with PyTorch have a
[queue-interception heap overflow](https://github.com/ROCm/rocm-systems/pull/4066) when many kernels are queued.
[ROCm issue #5071](https://github.com/ROCm/rocm-systems/issues/5071) reports this workaround for the crash, now the default
for Caminus's serve and replay launchers. For an older installed build:

```bash
HSA_TOOLS_DISABLE_REGISTER=1 caminus restart NAME
```

This disables optional ROCm profiler registration for that server. Explicit `HSA_TOOLS_DISABLE_REGISTER` values and
profilers attached through `ROCP_TOOL_LIBRARIES` or `HSA_TOOLS_LIB` are preserved. GPU profiling needs registration enabled
and a runtime with the upstream fix. The HIP/HSA runtime comes from PyTorch, separately from the kernel compiler version
printed at startup. Preserve the logs if a crash persists with the workaround.

**"GUARD REFUSED" after your monitors went to sleep**: in Caminus 0.2.0 the guard took a display timeout for a GPU fault.
If every line in `~/.local/state/k3/GPU_FAULT_STOP` says `REG_WAIT timeout`, it was your monitor, not the GPU: delete the
file and start again. Later versions only note these lines and keep running.

**The whole desktop lags, and the kernel log shows `MES ring buffer is full`** (`journalctl -k | grep MES`): a card's
scheduler firmware stopped responding, usually after it woke from power saving. Reboot; the guard refuses GPU work until
then. We have seen it with Linux kernel 7.0.0-42 on an RX 7900 XT and not with 7.0.0-38: if it comes back, boot the older
kernel from the GRUB menu (Advanced options) and report it with the log lines.

**Slower than the numbers in the docs**: the cards may be running at normal clocks (about 13% slower); see
[getting full speed](hardware.md#getting-full-speed). The server's log says at startup whether the clocks are high.

**Nothing arrives during a long prompt**: OpenAI-style streams send nothing until the first token; a 100K-token prompt
takes about 40 s with `coder`.

**HTTP 503 or 529 after an error**: a step failed its retries. The server stays up and the next request starts fresh from
the cache. It saved a replay file for a bug report: `caminus flight list` shows it.

**Repeated acknowledgements in an answer**: chat responses stop after eight consecutive identical nonempty prose lines
of at least 24 characters, ending in sentence punctuation. Fenced code and tool payloads are excluded. The request returns
HTTP 422 with `generation_loop` (OpenAI), or an error event if streaming already began; it is not reported as a successful
completion. The flight recorder attempts a `loop` replay bundle, subject to its usual size and rate limits. Use
`--no-loop-guard` for intentional repeated prose. This limits runaway output; it does not establish model accuracy over
long sessions. Raw `/v1/completions` output is unchanged.

## Ember

`caminus ember setup` ends with a PASS / FAIL line per check, like the doctor; the
[Ember guide](ember.md#troubleshooting) explains each one, and `caminus ember agent status NAME` shows an agent's goals
and the last error.

## Disk space

`caminus purge` lists what the prompt cache and the request log hold; `caminus purge --yes` deletes it. Stop your models
first.
