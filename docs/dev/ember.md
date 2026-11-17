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
- Nothing waits on the human: a question times out to the advisor's pick or the recommended option, a permission or
  root request to no. A question waits longer while the human is away (the panel's `presence.json` stamp,
  `approval.wait_left`, mirrored by `questionLeft` in `panel/emberquestion.go`), never past `question_away_max`; a
  root request never does.
- The prompt says what the runtime does. When you change a refusal, a limit or a guard, change `resident/prompt.py`
  (and the docs) in the same commit.
- Agent freedom: guards stop spending without progress; they do not decide what the agent may want. Free time is
  never graded and never paused by a loop guard.
- No GPU in any env test; the env suites are CPU only.

## Testing

```sh
python -I tests/ci_cpu.py --scratch DIR --only env,resident,resident_discovery,resident_compare,env_setup_e2e,emberlab
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

## The lab

`tools/emberlab/` develops and tests Ember the way a model meets it: a real resident in a fresh home per run, a real
brain or a stand-in, and a scorer that does not trust the agent. It measures a harness change, and it lets you work
inside the harness with an agent while you design it.

| Module | |
|---|---|
| `lab.py` | the command line: `run`, `report`, `compare`; `open`, `say`, `turn`, `reply`, `events`, `status`, `close` |
| `runner.py` | suites, arms, the run matrix, one run (fresh home, `once`, events, the check) |
| `brains.py` | `scripted:NAME`, `driven` (you answer each model call), `openai:URL[#MODEL]`, `openai+relay:URL#MODEL`, `anthropic:PORT#MODEL` |
| `hosts.py` | `local` (scratch folders as you) and `agent` (the agent account over Ember's SSH, under `~/lab/`) |
| `report.py` | tables per brain, arm and task; the A/B verdict |
| `sessions.py` | long-lived lab agents to talk with or drive |
| `tui.py` | a terminal program in a hidden tmux window: keys in, exact screens out (text, colour, SVG) |

The suites live in `benchmarks/ember/suites/` ([their format](../../benchmarks/ember/README.md)).

**Measure a harness change.** An arm is a version of the harness: a checkout plus config.json keys and environment
variables. Run both arms on the same suite and brains, then compare:

```sh
python -I tools/emberlab/lab.py run --suite basic --brain openai:http://127.0.0.1:8093/v1 \
  --arm base=../caminus-testing --arm candidate=. --repeats 3 --out DIR
python -I tools/emberlab/lab.py compare DIR --base base --arm candidate
```

- **The score.** The task's check decides each run. The resident's own verdict is recorded beside it, and a goal that
  finished while its check failed counts as a false completion.
- **The verdict.** An arm is better only if it is better across the board: no brain's pass rate drops, no brain's
  false completions rise, and at least one improves. Fewer than three repeats per cell are marked noise-prone.
- **Order.** Arms alternate their order every repeat, so neither always runs first.
- **Records.** Each record names its runtime's commit and whether its tree was dirty.
- **Resuming.** `run` appends to `results.jsonl` and skips the ids already there, so an interrupted matrix resumes.

**Work with the agent.** A lab session is a resident with its daemon running, in its own home:

```sh
python -I tools/emberlab/lab.py open s1 --brain openai:http://127.0.0.1:8093/v1 --fixture basic --out DIR
python -I tools/emberlab/lab.py say s1 "Try the tools in your work folder and tell me which ones got in your way" --out DIR
```

- **Ask it, or drive it.** Ask the agent what it can and cannot do from inside its account, have it try a tool, or
  steer a goal with `say --goal ID`.
- **Be the model.** With `--brain driven` you answer each model call yourself. `turn` prints the waiting request (the
  conversation tail, the tools offered, the newest result), and `reply --call TOOL --args JSON` or `reply --text`
  answers it. You see exactly what a model sees, gate refusals included.
- **The TUI.** `tui.py --session NAME --lab-out DIR` opens the chat on that session's home. It runs with its own
  config folders, so it never reads or writes your panel settings.

**Hosts and safety.**

- **Real models run in the agent account.** A real model runs only on the `agent` host: the local host would give it a
  shell as you, and the lab refuses that unless you pass `--unsafe-local`.
- **What the agent host leaves alone.** It works under `~/lab/` in the agent account and never touches `~/homes` (the
  live sessions), a systemd unit or the owner's bridge. Each run has its own bridge, which nobody answers, so a root
  request times out to no.
- **Paid brains.** They go through the owner's key relay (`caminus ember relay`), whose daily cap is the spend guard.
  The lab never reads or passes a key.
- **Local servers.** A local server answers one request stream at a time (`--parallel 1`, the default for `openai:`).
  Don't point the lab at a server someone is using without asking: lab goals evict that server's cached prompt.
- **The output folder stays private.** It holds model output and paths from your machine, so keep it out of the
  repository.

**Real accounts.** `tui.py --cmd 'caminus ember ui' --read-only` drives your own chat with look-around keys only:
navigation, Escape, shift+tab, ^P, and Enter after a few read-only slash commands. A step that would send a goal,
start, stop or destroy a session is refused before the run begins.
