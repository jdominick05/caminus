# Ember for contributors

How Ember, the agent environment, is built, where each piece lives, and how to test a change. Users: [the guide](../ember.md).
The contract (users, transport, bridge, audit, jobs, sessions, servers) is [spec-ember.md](../specs/spec-ember.md).

## The shape

Two users, three processes, one shared folder:

```
owner (you)                                   caminus-agent (the resident's account)
-----------                                   --------------------------------------
caminus ember ... ── SSH (your key, localhost) ─▶ ~/bin/caminus-agent  (a stdlib-only zipapp)
caminus ember ui (Go panel) ─ reads/streams ────▶   └─ resident daemon (systemd --user): the loop
caminus ember relay (systemd --user) ◀── HTTP ───   the brain calls (paid APIs; the key stays with you)
        \                                          /
         └── /srv/caminus-env-bridge: req/ res/ notes/ leases/ STOP ──┘   (root:caminus-env)
```

- **Owner side** (`python/caminus/ember/`, `python/caminus/cmd/ember.py`): setup, the SSH transport, the root bridge, the
  audit log, jobs / sessions / servers in the agent's account, the key relay. Runs as you; may import anything in
  `caminus`.
- **Agent side** (`python/caminus/ember/resident/`): the resident. Shipped as a zipapp holding only this package, so it
  imports nothing outside it and the standard library (a test fails if it does). Runs as `caminus-agent`.
- **The panel** (`panel/env*.go`): the chat. It talks to the resident only through `caminus ember agent ...` over SSH
  (status, a streamed event feed, actions) and to the bridge for root requests.

## Module map

| Owner side | |
|---|---|
| `cmd/ember.py` | every `caminus ember` subcommand: setup, status, run / shell, the bridge (reqs, answer, watch), jobs, sessions, stop-all |
| `env/provision.py` | the root setup script (safe to rerun, checks itself) |
| `env/runner.py` | the SSH transport: one ControlMaster connection to `caminus-agent@127.0.0.1` |
| `env/bridge.py`, `approval.py` (resident) | root requests: the agent files, only the owner answers; decisions serialize with withdrawal |
| `env/audit.py` | the owner's audit log (one JSON object per line) |
| `env/agent_host.py` | builds the zipapp and addresses the resident over SSH (`caminus ember agent ...`) |
| `env/relay.py` | the spend-capped key relay; `relay install` runs a frozen copy of its code as a user service |
| `env/jobs.py`, `sessions.py`, `serve.py`, `ctx.py` | detached jobs, tmux sessions, model servers and context budgets in the agent's account |
| `env/prompt.py`, `share/harness/ember/ROBOTS.md` | the account's instructions (`~/ROBOTS.md`) |

| Agent side (`env/resident/`) | |
|---|---|
| `runtime.py` | the loop: ticks, tool dispatch, the per-turn gate, compaction, the watchdog, updates |
| `store.py` | SQLite: goals, the conversation, actions, the inbox, the event stream the panel reads |
| `prompt.py` | the system prompt (what the agent is told; keep it true to what `runtime.py` enforces) |
| `model.py`, `brain.py`, `catalog.py` | transports (OpenAI-compatible, Anthropic), primary + fallback brains, the model catalog and prices |
| `judge.py`, `planning.py` | the finish judge (met / not yet / wait; it never declares a goal impossible), plan-first goals |
| `terminal.py`, `files.py`, `web.py`, `mcp.py` | persistent PTY jobs, read / edit / write, the web tools, MCP servers |
| `memory.py`, `freetime.py`, `agenda.py` | notes, STATE.md and history search; free time and dreams; report / budget / schedule |
| `housemate.py`, `leases.py`, `mail.py` | waiting without tokens, machine facts, GPU / memory leases, peer mail |
| `btw.py`, `diff.py`, `export.py`, `outputs.py` | `/btw`, `/diff`, `/export`, spilling big outputs to files |
| `cli.py` | the entry point every mode uses (`once`, `run`, `send`, `status`, `stream`, ...) |

| Panel (`panel/`) | |
|---|---|
| `envui.go` | the program: flags, the model, update routing, the window title |
| `envchat.go`, `envgroup.go` | the transcript (compact view, cards), Ember's box and the side panel, the task / now line |
| `envcomposer.go`, `envsteer.go`, `envselect.go` | the composer, steering (interrupt, pullback, rewind), select to copy |
| `envapprovals.go`, `envquestion.go`, `envbridge.go` | approval cards and the overlay, questions, the root bridge and grants |
| `envresident.go` | talking to the resident (status, stream, actions; panel controls go to the audit log) |
| `envgoals.go`, `envmodels.go`, `envsessions.go`, `envstudio.go`, `envpages.go` | the overlays |

## Rules a change must keep

- The resident package stays standard library only and self-contained (the zipapp holds nothing else).
- The agent never holds root or a key: root goes through the bridge and runs under the owner's sudo; paid brains go
  through the owner's relay.
- Nothing waits on the human: a question times out to the recommended option, a permission or root request to no.
- The prompt says what the runtime does. When you change a refusal, a limit or a guard, change `resident/prompt.py`
  (and the docs) in the same commit.
- Agent freedom: guards stop spending without progress; they do not decide what the agent may want. Free time is
  never graded and never paused by a loop guard.
- No GPU in any env test; the env suites are CPU only.

## Testing

```sh
python -I tests/ci_cpu.py --scratch DIR --only env,resident,resident_discovery,resident_compare,env_setup_e2e
(cd panel && go vet ./... && go test ./...)
python -I tools/ember_demo.py            # the chat on a scripted model in a scratch home: no account, server or key
```

- `tests/ember/test_env.py`: the owner side (settings, the SSH argv, the bridge, the audit log, the setup script).
- `tests/ember/test_resident.py`: the resident against an in-process OpenAI-compatible fixture (`Endpoint`, scripted
  replies; its judge says met unless a test queues a verdict). Most behaviour tests live here: add one per fix, named
  for the behaviour.
- `tests/ember/test_setup_e2e.py`: setup from a clean account to the first reply against a stub brain, through the real
  SSH path. It creates a Unix account with sudo, so it runs only with passwordless sudo and `CAMINUS_E2E_ROOT=1` (CI);
  anywhere else it prints SKIP.
- `panel/env*_test.go`: frames at fixed sizes (they must fit the terminal), keys, overlays, the side panel, selection.
  The panel never sends desktop notifications from a test binary.
- A live change: `caminus ember agent install` then `caminus ember agent update NAME` installs this checkout's runtime
  into a running resident between goals (`--now` at the next step); its heartbeat names the runtime it runs.
