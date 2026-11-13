# Caminus as a systemd user unit

`caminus.service` runs `caminus serve` (the K3 engine on both cards, the OpenAI / Anthropic API on 127.0.0.1:8092) as a
systemd **user** unit **from a checkout**: it assumes the checkout at `~/src/caminus` and a conda env named `drafter` with ROCm
PyTorch (edit `ExecStart=` for yours). It is not a unit for the pip-installed package (docs/quickstart.md runs `caminus serve`
from a terminal). Nothing in the repo installs or enables it; these are the steps to take by hand.

## Link and start it

```
systemctl --user link ~/src/caminus/packaging/systemd/caminus.service   # a symlink in ~/.config/systemd/user (systemctl --user daemon-reload after edits)
# stop any other engine that holds the GPUs first (a llama-server, a test): only one engine on the cards
systemctl --user start caminus
journalctl --user -u caminus -f                                          # the guard's own lines (its rc line at the end) and caminus-stop's
tail -f ~/.local/state/caminus/serve.log                                 # the server's output (the guard writes it there, not to the journal)
curl -s 127.0.0.1:8092/caminus/status | python3 -m json.tool             # clock_check, cards, cache, ...
systemctl --user stop caminus                                            # SIGTERM to the server alone: drains, parks the live conversation, exits
```

`systemctl --user enable caminus` would start it at login; that is the user's choice (it takes both GPUs and ~23 GB of the
XTX's VRAM for as long as it runs). `systemctl --user disable caminus` / `systemctl --user unlink caminus.service` undo it.

## What the unit does

- **Port 8092** (`--port 8092`); tests use `--port 0`.
- **`Conflicts=fn-server.service`.** On the development machine `fn-server.service` is the name for a llama-server user unit
  on the same GPUs: starting caminus stops it, and the other way round. Put your own other engine's unit name there, or drop
  the line. An engine started outside systemd is out of its reach: stop it first. If both run, the second engine's capacity
  check refuses to build (the VRAM is read as the guard reads it, against the guard's lines).
- **The guard** (`tools/guard.sh LOG 0 ...`, the same wrapper every GPU command in this repo runs under):
  - kills the server when the XTX's VRAM passes 23,800 MiB, the XT's 19,850 MiB, or host MemAvailable falls below 1,500 MiB;
  - kills it on an amdgpu page fault / reset / wedge / timeout in the kernel log and records the STOP file
    (`~/.local/state/k3/GPU_FAULT_STOP`);
  - runs it under `gpu-high` (`CAMINUS_GPU_HIGH`, else on `PATH` or in `~/.local/bin`; DPM `high`, runtime PM `on`; the clocks go
    back to `auto` when it exits);
  - writes its log (`serve.log`, the server's output) and a 1 Hz VRAM / RAM line (`serve.log.mon`) to
    `~/.local/state/caminus/`. The `.mon` file grows ~2.5 MB a day while the unit runs (~75 MB a month) and is never
    rotated: truncate it while the unit is stopped (`: > ~/.local/state/caminus/serve.log.mon`). `serve.log` is rewritten at
    each start;
  - prints its own lines to its stdout, which under systemd is the journal: a `GUARD KILL` / `GPU FAULT` line (also in
    `serve.log`) and, when the server has exited, its rc line (`rc=0 wall=... s peak XTX ... MiB peak XT ... MiB min avail ...
    MiB`): `journalctl --user -u caminus`.
- **`ConditionPathExists=!.../GPU_FAULT_STOP`:** after a recorded fault the unit does not start (systemd reports the condition
  as failed) until the fault has been inspected and the file removed by hand.
- **`Restart=no`:** never restarted after a fault. On 11-01 repeated runs after faults reset both GPUs 9 times and lost the
  desktop session. The one exception is **exit 75** (`RestartForceExitStatus=75`, `RestartSec=10`): `caminus serve` exits 75
  when its engine produced non-finite logits (a weight or state corrupt in this process; the startup self-check, or a request:
  `/health` answered 503 with the reason until the drain ended). A fresh load re-uploads the weights. `StartLimitBurst=3` in
  `StartLimitIntervalSec=3600` allows 2 such restarts an hour; a model that breaks on every load then stays down.
- **Stop: the server alone gets SIGTERM** (`ExecStop=caminus-stop 170 $MAINPID`, `KillMode=mixed`). `caminus-stop` (beside the
  unit) finds the server among the guard's processes (it names itself `caminus`), sends it SIGTERM by its PID, and waits for
  the unit's main process, the guard, to exit (at most 170 s). The server stops accepting, cancels the running request at its
  next pass boundary (its client gets what was generated), waits for the engine thread (`--drain-s`, 60 s), parks the live
  conversation when it changed since its last park (`--park-s`, 90 s: a 262K conversation writes ~7 GiB in a few seconds),
  releases the prompt cache and exits. Then gpu-high puts the clocks back to `auto`, and the guard, which kept watching the
  whole time (the VRAM / RAM lines, the fault watcher), prints its rc line and exits. Only if the guard still runs after that
  does systemd send its own SIGTERM, to the guard alone (`KillMode=mixed`). With `KillMode=mixed`, systemd SIGKILLs every
  remaining process as soon as the guard exits (systemd.kill(5)), not after `TimeoutStopSec`: a server still draining or
  parking past caminus-stop's 170 s wait (its own bounds total 150 s) loses the park and the clocks stay high. A stop in the
  few milliseconds while gpu-high runs `sudo perf high` finds no server yet; caminus-stop then waits its 170 s first. (The first version had `KillMode=control-group`: the stop's SIGTERM reached the guard, `timeout` and gpu-high together
  with the server, so the guard stopped watching while the server parked, its rc line was lost, and gpu-high died before
  `perf auto`, leaving the clocks high. A CPU replica of the unit's ExecStart / ExecStop with copies of the guard and gpu-high
  showed both: the old unit lost the rc line and `perf auto`, the new one keeps them; final round 2, 11-06. The unit itself has
  not been run under systemd by an agent.)
- **A stop after the guard has exited on its own** (a `GUARD KILL`, a GPU fault, a crash): systemd still runs `ExecStop=`, with
  `$MAINPID` unset; `caminus-stop` then has nothing to stop and returns at once (it never signals under a PID that is not
  tools/guard.sh).
- **A stop while the engine loads** (the first ~50 s): the server's SIGTERM handler is installed once it is ready, so the
  signal ends it at once (nothing to park yet); gpu-high and the guard finish as above (rc 143).
- **Idle:** the live conversation is parked in the background after 8 hours without a request (`--idle-park-s 28800`), when
  it gained at least 4,096 positions since its last park. A switch to another conversation before then parks it in the
  request's path, as before.
- **The clock check** at startup (read only, never changed): each engine card's DPM level and runtime PM, in the log and in
  `/caminus/status` (`clock_check`). Under the guard both cards read `high` / `on`.

## What the server itself refuses (with or without the unit)

- VRAM: the engine, the drafter, the prefill and the cache tier are counted before they allocate, against the guard's lines
  with the XTX's 0.5 GB reserve (`capacity.check`); a configuration that does not fit is refused at startup.
- Every device wait is time-bounded (200 ms) with a bitwise pass retry; a pass that fails its retries is recovered from the
  request's prompt-end checkpoint (bail recovery) or answered 503 / 529.
- Not in the server (only the guard has them): the kernel-log fault watcher and the STOP file. Run it under the guard.

## Options

Flags or a JSON / TOML config (`--config FILE`): `python -I tools/caminus_serve.py --help`. To change the unit's flags, copy
the unit to `~/.config/systemd/user/caminus.service` instead of linking it, edit `ExecStart=`, then
`systemctl --user daemon-reload`.
