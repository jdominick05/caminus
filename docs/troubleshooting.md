# Troubleshooting

By the end of this page you'll know what an error means and what to do about it. Start with `caminus doctor`: it checks
everything below and prints PASS, WARN or FAIL for each.

If nothing here helps, please [open an issue](https://github.com/jdominick05/caminus/issues/new/choose) and paste the full
output of `caminus doctor`.

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
allow direct card-to-card access. Move the card to a faster slot. One-card models still run.

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
takes about 40 s with `coder`. `qwen3-coder-next` reads prompts much more slowly for now.

**HTTP 503 or 529 after an error**: a step failed its retries. The server stays up and the next request starts fresh from
the cache. It saved a replay file for a bug report: `caminus flight list` shows it.

**Repeated acknowledgements in an answer**: chat responses stop after eight consecutive identical nonempty prose lines
of at least 24 characters, ending in sentence punctuation. Fenced code and tool payloads are excluded. The request returns
HTTP 422 with `generation_loop` (OpenAI), or an error event if streaming already began; it is not reported as a successful
completion. The flight recorder attempts a `loop` replay bundle, subject to its usual size and rate limits. Use
`--no-loop-guard` for intentional repeated prose. This limits runaway output; it does not establish model accuracy over
long sessions. Raw `/v1/completions` output is unchanged.

## Disk space

`caminus purge` lists what the prompt cache and the request log hold; `caminus purge --yes` deletes it. Stop your models
first.
