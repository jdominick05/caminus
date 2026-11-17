# Ember, the agent environment (`caminus ember`, experimental)

An agent gets an environment instead of a fixed tool list: a second user on this machine, reached over SSH, where it
can do any IO that user can do. The boundary is Unix, not a sandbox: your home, your keys and your processes are
protected by file ownership and permissions, and root is not in the agent's reach at all — a root command waits in a
bridge until you approve it, and only your side executes it.

Status: experimental. The environment, resident loop and panel have CPU integration tests;
CI uses local HTTP fixtures and real PTYs, without GPU inference or live SSH.

**Code:**
- `python/caminus/ember/settings.py`: the names and paths (`CAMINUS_ENV_USER`, `CAMINUS_ENV_KEY`, `CAMINUS_ENV_BRIDGE`, `CAMINUS_ENV_AUDIT`, then defaults).
- `python/caminus/ember/runner.py`: the SSH transport.
- `python/caminus/ember/bridge.py`: the root request/result protocol.
- `python/caminus/ember/audit.py`: the audit log.
- `python/caminus/ember/agent_side.py`: the `ask-root` script installed into the agent user.
- `python/caminus/cmd/ember.py`: the `caminus ember` command.

**Tests:** tests/ember/test_env.py (suite `env` in tests/ci_cpu.py).

The native resident loop, scriptable oneshot contract, peer mail and comparison
benchmark are described in [Resident agents](../resident-agent.md). CPU suites
`resident` and `resident_compare` cover these additions. Existing environment
commands remain available for external CLIs and manual sessions.

## 1. The model

Two principals on one machine:

- **You** run the harness (`caminus ember ...`). It holds the SSH key, the bridge and the audit log.
- **The agent user** (default `caminus-agent`) runs all agent work: `caminus ember run CMD...` executes as that user,
  `caminus ember shell` opens an interactive session for any agent CLI, and anything started there keeps running after
  logout (`loginctl enable-linger`).

What the boundary gives you, concretely: the agent user cannot read your home (750), your `~/.ssh`, your shell
environment or your API keys; it cannot signal your processes; it has no sudo rights, so it cannot become root. It
does still have the network, every world-readable file, the GPU (it is in the `render`/`video` groups so Caminus and
benchmarks run there), and write access to anything group- or world-writable — this is an environment, not a jail.
Scope credentials deliberately: give the agent user only short-lived, low-scope tokens in its own environment.

## 2. Provisioning

`caminus ember setup` prints the exact sudo commands (they are run by you, never by an unattended agent): install and
start `openssh-server`, create the user with a locked password, create the `caminus-env` bridge group and put both
users in it, the `render`/`video` groups for the agent user, the bridge directory `/srv/caminus-env-bridge` (root:
group, 770, with `req/` and `res/`), enable linger, generate the harness key, and add one line to the agent user's
`authorized_keys`:

```
from="127.0.0.1,::1",no-agent-forwarding,no-X11-forwarding,no-port-forwarding <harness pubkey>
```

`caminus ember setup --agent-side` installs `~/bin/ask-root` as the agent user over SSH. `caminus ember status` checks
each piece: the user exists, the key exists, `ssh` login works (`id -un` answers the agent user), the bridge is
writable, `ask-root` is installed.

## 3. The SSH runner

One persistent ControlMaster connection to `127.0.0.1` with `BatchMode=yes`. Commands travel **on stdin**
(`ssh ... bash -s`), so agent-supplied text is never re-quoted across the wire: `run_argv` shlex-quotes an argv into the
script, `run_shell` sends one shell string verbatim. The ControlMaster socket lives in the state dir.

## 4. The root bridge

The agent user files a request; only the harness side answers and executes. Files are JSON, written atomically
(tmp + rename, umask 007 so the other user can read them):

- `req/<id>.json`: `{'id', 'pid', 'user', 'cwd', 'cmd', 'ts'}` — written by the agent side (`ask-root CMD...`, or
  `caminus ember ask-root CMD...` when the package is installed in that user).
- `res/<id>.json`: `{'id', 'decision', 'cmd', 'approved_by', 'rc', 'out', 'err', 'ts'}` — written by the harness side.

`caminus ember watch` drains `req/`, shows you each command with its origin, and asks: **a**pprove runs exactly that
string as `sudo sh -c <cmd>` (sudo asks for your password on your terminal), **e**dit lets you change it first,
**d**eny answers 126, **s**kip leaves it queued. The result (exit code, stdout, stderr) goes back to the waiting
agent process. A request nobody answers times out (agent side: exit 124). The harness never executes anything from a
request file other than the string it displayed and you approved.

## 5. The audit log

One JSON line per event in the state dir (`env-audit.jsonl`, writable only by you): every `env run`, every root
request, decision and result. `caminus ember audit --tail N` prints the last N. sudo's own log in
`/var/log/auth.log` is the second record of every root execution.

## 6. The agent's own instructions: ROBOTS.md

`share/harness/ember/ROBOTS.md` (package data) is the environment's system prompt, in the soul.md / QWEN.md tradition:
it tells the agent what it is (a dedicated user on the human's real machine, not a sandbox), what it can do (full
shell, network, GPU, its own home, persistent processes), what it cannot (the human's files and root), how to ask for
root (`~/bin/ask-root CMD...`), that everything is audit-logged, and where the local model servers listen.
`caminus ember prompt` prints it; `caminus ember setup --agent-side` installs it as the agent user's `~/ROBOTS.md`;
`caminus ember shell` points at it when opening a session.

## 7. The TUI (`caminus ember ui`) and the scriptable CLI

`caminus ember ui` execs `caminus-panel env --user ... --bridge ... --audit ... --key ...` (panel/envui.go, same binary
and build as the control panel). Seven live tabs (tab / 1-7), starting on **agent** (goals and activity): **requests** (ROOT: `a` approve — sudo runs attached to
the terminal via `tea.ExecProcess` so your password prompt works, output teed to a temp file; `e` edit, `d` deny,
`s` skip — ASK: `a` answer, a number picks a choice), **jobs** (detached processes in the agent user: `K` kill,
`l` tail the log, `n` new job), **procs** (the agent user's busiest processes), **notes** (steering inbox: `n` new,
`d` delete), **ctx** (servers and their `/props` windows), **audit** (log tail). Answering a question also appends
the Q/A to the agent's `~/MEMORY.md`. Answer files and audit lines use the same JSON schema as `bridge.py` /
`audit.py`, so `caminus ember watch` / `audit` and the TUI are interchangeable.

For scripts and tests, everything is callable without the TUI: `caminus ember reqs --json`, `caminus ember answer ID
approved|edited TEXT|denied|answered TEXT [--json] [--digest HASH]`, `jobs --json`, `notes --json`, `ctx --json`, `audit --json`,
`memory` (print the agent's ~/MEMORY.md), `remember TEXT` (append to it). Go tests: panel/envui_test.go.

Root `approved` and `edited` decisions require `--digest HASH`: copy the `_digest`
from the same `env reqs --json` object whose command you reviewed. A changed request
is rejected instead of executing a replacement. The interactive watch and panel
retain and validate that reviewed snapshot automatically.

## 8. Detached processes and multiple terminals

`caminus ember spawn --name N -- CMD...` starts CMD detached in the agent user (`nohup setsid`, log
`~/env-logs/N.log`, pid `~/env-jobs/N.pid`) and returns the pid; `caminus ember jobs` lists them (alive or dead, with
the log path); `caminus ember stop N` kills one. Job names are `[A-Za-z0-9_.-]+`. Everything survives logout (linger).
Inside its own session the agent uses the same conventions directly (`nohup ... &`, `tmux`): each `caminus ember run`
is a separate short-lived SSH exec (multiplexed over the one ControlMaster connection), and each `caminus ember shell`
is its own PTY. ROBOTS.md section "Long-running work and more terminals" tells the agent all of this.

## 9. Questions, steering, context

The bridge carries typed requests, not just root: `kind: "root"` (cmd) and `kind: "question"` (text, choices).
Agent side: `~/bin/env-ask TEXT CHOICE...` prints the human's answer (a number picks a choice); `caminus ember ask`
when the package is installed in that user. Harness side: `caminus ember watch` prompts for the answer, the TUI opens
an answer line on `a`; both write `decision: "answered"` with the `answer` field, same schema otherwise.

Steering runs the other way through the resident's own channel: a message, `/steer` or `caminus ember agent send
--steer` reaches the running goal at its next step (the old notes inbox, `env say` / `env-note`, is gone).

Memory: the agent's `~/MEMORY.md` (seeded by `setup --agent-side` from `env/prompt.py`'s MEMORY_TEMPLATE) is its
persistent, self-writable memory; ROBOTS.md tells it to read it every task and append learnings. `caminus ember
memory` / `remember TEXT` read/append it from the harness side, and every answered question is auto-appended as a
Q/A block (watch, TUI and `answer answered` all do this) — the learn loop.

Context: `caminus ember ctx` probes 127.0.0.1:8092-8101 `/props` and prints each running server's model and `n_ctx`
(the real window, sized to the cards). The agent reads the same `/props` from inside its user and compacts itself
before the window fills (ROBOTS.md "Your context window"); Caminus already refuses over-long requests, and tools
like Qwen Code compact automatically against this number.

## 10. Named sessions: how the human tells the agent to do something

tmux inside the agent user is the interactive channel. `caminus ember new NAME -- CMD...` starts a detached named
session running CMD (an agent CLI, a shell); `caminus ember sessions` lists them; `caminus ember send NAME TEXT` types
TEXT+Enter into the session (a live instruction — audited); `caminus ember attach NAME` execs `ssh -t ... tmux attach
-t NAME` so the human drops into the same terminal to watch and type. Names are `[A-Za-z0-9_.-]+` like job names.
One-shots stay `env run`, long detached work stays `env spawn`/`jobs`/`stop` (section 8); mid-task steering goes
through the resident (section 9).

## 11. Server jobs (`caminus ember serve`) — one job per agent, each with its own home

`caminus ember serve NAME MODEL [--placement auto|0|1|2] [--ctx N] [--port P] [--dry-run] [-- serve flags...]`
starts a model server as an agent-user job: placement is resolved with the same code as `caminus start`
(`manage.pick_cards` / `serve_argv`) — `auto` uses the model's own card count, `0`/`1` force one card (a model
that needs two gets no one-card option, refused with the reason; pair rules like "the Coder needs a 24 GB card 0"
apply; VRAM held by non-Caminus processes refuses with "stop it first"), `2` splits across both cards. The next
free port from 8092 is allocated. Every stateful path goes under the agent user's `~/homes/NAME/`
(state, kv, flight, qwen-home, XDG dirs — its own session home), model/drafter paths are remapped to the models
bind-mount (`CAMINUS_ENV_MODELS`), and the server starts in a named tmux session running a generated
`~/homes/NAME/run.sh` (guard + HIP_VISIBLE_DEVICES + `caminus serve ...`). `--dry-run` prints the launcher and the
session command. In the TUI, `S` in the jobs tab runs this flow (model, then placement). `env stop NAME` kills
the job and the session. Provisioning for agent-user servers: the bind-mount lines in `caminus ember setup`
(`/opt/caminus-runtime/{conda,src,models}` + `CAMINUS_ENV_PYTHON` / `CAMINUS_ENV_PKG` / `CAMINUS_ENV_MODELS`).

## 12. Security properties, honestly

- Protects: your files, credentials and processes from the agent user and from anything that injects into it; root
  entirely (the agent user has no sudo; elevation needs your password on a command you saw).
- Does not confine: network egress, world-readable files, GPU, disk, and any shared writable tree remain open. An
  agent with full network IO can exfiltrate whatever it can read — which is why it can only read its own scoped
  environment.
- The resident's `sandbox` tool confines what runs inside it (namespaces, no home, no network unless the call names
  hosts, limits; docs/resident-agent.md "Tools"). The agent chooses when to use it: its terminal stays unconfined.
  Its `browser` tool runs Chrome in such a box, with the proxy open to public addresses (logged) and closed to the
  LAN and this machine unless a call names the address.
- The bridge directory is writable by the agent user; treat request files as untrusted input (they are display-only).
