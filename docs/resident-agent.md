# Resident agents (experimental)

The resident is a native, durable loop around an OpenAI-compatible model. It runs
as the agent Unix user and uses that account's Ubuntu environment through real
PTY terminals. It can run installed programs, write programs, keep commands
running, send terminal input, communicate with peers and ask the human for help.
It does not require Qwen Code or another agent CLI to perform its work.

The inference server and the resident loop are separate processes. Installing
the resident does not move a running inference server into the agent account or
start a GPU workload. Provision the runtime and model files separately before
starting an agent-owned server, using the normal GPU guard and an agreed window.
Graphical desktop attachment and a built-in vision/computer-use interface are
not implemented; available Ubuntu automation programs can be used through the
terminal under the account's normal permissions.

## Start and use a resident

New here: [the quickstart](ember-quickstart.md) goes from install to the first conversation, and [the guide](ember.md)
covers the security model, where things live and troubleshooting. This page is the reference. After `caminus ember
setup` (one root script, then the command again):

```sh
caminus ember agent install
caminus ember agent init resident --model MY_MODEL --endpoint http://127.0.0.1:8092/v1
caminus ember agent start resident --json
caminus ember agent send resident --stdin --json < goal.txt
caminus ember agent status resident --json
caminus ember ui
```

`install` uploads a stdlib-only, versioned zipapp over the existing SSH transport.
Goals and commands execute as the agent user. `start` enables an agent-owned
systemd user service; linger, provisioned by `env setup`, keeps it alive after
logout. `env setup` also creates the shared leases folder (`<bridge>/leases`,
setgid 2770) the `lease` tool and the human's GPU windows write. Stopping the
resident does not stop its inference server. Terminal workers
survive a resident restart, so long commands can be inspected rather than repeated.

### The panel: one chat, with overlays

`caminus ember ui` opens the conversation; there are no pages to switch between.
Everything else opens over it in the same bordered box (↑↓ move, Enter acts, Esc
closes; `q` never quits inside an overlay). The header names the agent, its
state and the goal; the footer shows the brain, the context in use against the
window (a bar that warms toward the compaction point), this goal's spend and
today's on a paid brain, the last reply's speed and anything waiting for you.

The composer sits below the transcript. Enter sends, Alt+Enter or Shift+Enter
adds a line, paste keeps Unicode and newlines. A plain message is delivered at
the agent's next step as a **nudge**; `/steer TEXT` redirects (see "The loop").
A message that starts with `/` is a command; `//` sends it as text instead.

| Key | Action |
|---|---|
| Esc | while the agent waits on a reply (also before its first word: a long prompt read, a paid model thinking), interrupt that turn: the call is closed, so the server stops generating, and nothing of it is kept. The activity line then says "turn interrupted" with no clock, and the goal waits up to two minutes for your message before it carries on by itself. Esc while a tool runs says there is nothing to interrupt. PgUp/PgDn and the wheel scroll the transcript |
| ↑ on an empty composer | pull back your newest message the agent has not read yet |
| Ctrl+R | earlier prompts containing the draft, newest first (as you typed them: `//`, `@` and `+` kept) |
| Home, End | with an empty draft: the top or the bottom of the history |
| Esc on the `/` or path menu | closes it; the text stays |
| Ctrl+N | a new goal; the draft stays and becomes its text, Esc goes back to the goal you were talking to |
| Ctrl+C | with a draft: clear it (Up brings it back); again, or with no draft: quit |
| Ctrl+X Ctrl+E | edit the draft in `$VISUAL` / `$EDITOR` |
| Ctrl+O | the layout: compact (a card per action, short previews) or everything (every card in full, the thinking, every gate run); your place in the history is kept |
| Ctrl+T, Alt+↑/↓ | hold or scroll the live reply while it is written |
| Ctrl+G, Shift+Tab, Tab | the goals overlay, the session switcher, and a waiting request (Tab answers it) |
| `!cmd`, `!!cmd` | `!` on an empty draft: a command in the agent's account (purple); `!` again: in your own shell on this machine (copper). The command and its output land in the draft (yours marked "my shell"); `!` on the draft's empty last line runs another, its output joining the first |
| `@path`, `+path` | at the start of a word: `@` names a path, sent as plain text; `+` attaches the file, read as the agent (up to 64 KB each, at most 8); a missing file, a folder or a `..` path stops the message with what is wrong and keeps the draft. `@` only drops from a word that reads as a path (`@alice` stays). Both complete from the agent's own folders (Tab, the same menu as commands), relative to the folder it works in. `/` is for commands only |

While a reply is written, a bounded live box shows its thinking, the tools it
starts and the text so far, with Ember, the Caminus mascot, flickering beside
it; the durable blocks replace it when the turn lands.

| Command | Action |
|---|---|
| `/new [goal]`, `/goals` | start a goal, or open the goals overlay to pick one: Enter targets it, space shows it (with its plan; `v` opens the whole plan), `p` pauses, `r` resumes, `x x` cancels, Shift+↑/↓ reorders the queue |
| `/plan`, `/plan TEXT`, `/plan approve`, `/plan revise NOTE` | alone, the whole plan in a scrolling view; TEXT starts a plan-first goal (it explores read-only, proposes a plan and waits; the plan shows as a card above the composer); approve or revise answers a waiting plan |
| `/steer TEXT`, `/pause`, `/resume`, `/cancel` | steer, pause at the next step, resume, cancel |
| `/btw QUESTION` | a side question with no tools; nothing enters the goal |
| `/diff` | the goal's changes in its git repositories (new files included) |
| `/rewind` | pick "undo from step N on": files the edit tools changed from that step go back and the transcript is cut there (two presses; terminal side effects are not undone) |
| `/model`, `/effort` | the models overlay (every provider's models, windows, prices) and the reasoning level; a running resident takes them at its next turn |
| `/context`, `/memory` | what the agent's context holds, its brain, window and where compaction starts; its memory file, in a scrolling view |
| `/studio` | what free time made (the shelf, the journal, the studio); the "while you were away" card shows on its own |
| `/panel` | the Caminus panel (models, servers, the cards) full screen inside Ember; Esc, q or e comes back to the chat as it was |
| `/compact`, `/remember TEXT` | compact now; add to memory |
| `/halt` | the emergency stop (`/halt clear` lifts it) |
| `/theme` | colours, with a live preview; `g` there cycles the symbol set (nerd, unicode, ascii) |
| `/sessions` (Shift+Tab), `/start`, `/quit` | the session switcher: a card per session (goal, time, cost, model, speed, GitHub account); Enter or a click switches; `n` or the "+ new session" card creates one on your setup's default local model (the one chosen for your cards; `/model` switches it); `x x` destroys one (its home is kept in `~/.local/share/caminus/removed-sessions`); `s` stops or starts one. `/start` runs this session's loop; `/quit` closes the UI (the agent keeps running) |

In the CLI only (rare, or about the machine rather than the conversation): `caminus ember agent export NAME GOAL`
(the goal as markdown), `caminus ember agent install` (a new runtime; running daemons take it between goals, and the
status line says "older runtime" until they do), `caminus ember ctx` (the model servers), `caminus ember audit` (the
audit log), `caminus ember jobs` (the agent's jobs), `caminus ember agent github NAME` (its GitHub account),
`caminus ember agent advisor NAME` (the advisor's model and budget) and `caminus ember agent heartbeat NAME now`
(free time now).

Custom commands: a markdown file `~/.config/caminus/commands/NAME.md` becomes
`/NAME ARGS` (front matter is dropped; `$ARGUMENTS` is replaced, else the
arguments are appended) and is sent as a message.

### Root requests and questions

A waiting request asks where you type: the composer becomes a prompt with the command (and why) or the question,
"1 of N" when more wait, and the time left. A root command or a GitHub hand-off offers Yes / No / "Tell it something
else", with No preselected; a question offers its choices (the recommended one marked and preselected) and "Answer in your own words". The
arrows and Enter, or a digit, pick; the last option opens a line for your text (for a root command: a denial that
tells the agent what to do instead, filed as a steer). Esc leaves it waiting: a line above the composer says so, and
Tab, or the next request, brings it back. Yes runs the command under your sudo; the resident never runs root. The
transcript keeps one line for each: `⚿ root: CMD` and then `· yes` / `· no: TEXT` / `· timed out (no)`. A granted
repeat (the same command in the same goal, within 12 hours of your approval) is honoured by `caminus ember watch`;
`caminus ember answer ID denied "WHAT INSTEAD"` and `answer ID approved --grant` work from the command line.

Nothing waits on you. A pending request parks only the step that needs it: the
agent is told it is pending, works on what does not need the answer, and gets
the answer as a message (`wait` with `until message` sleeps until it arrives);
it cannot finish while a request is open. Timeouts: see "When you are away".
Desktop notifications announce what needs you (`CAMINUS_PANEL_NOTIFY=0`
turns them off).

Question tool arguments are rendered as the interactive question rather than raw
JSON. The chosen human answer remains in conversation history. Normal PTY CRLF
newlines render cleanly; control bytes in reviewed commands remain escaped.

The daemon polls without inference while idle or waiting for human answers.
There is no production step limit. Backend outages retain the goal and retry
with backoff. `ROBOTS.md`, the resident's `IDENTITY.md` and `MEMORY.md` provide
persistent instructions and memory; a SQLite journal records goals, tool calls,
results and messages. Completed actions are checkpointed. An interrupted terminal
with uncertain effects is reported as interrupted rather than blindly relaunched.
Context compaction summarizes bounded transcript fragments before continuing.

Root commands are requests, never executed by the resident. The human reviews the
exact command and approves, edits or denies it through the existing bridge. The
agent's open-ended shell does not grant root or access to another user's protected
files. This is a Unix account, not a security sandbox.

## The loop

A goal runs as ticks. The runtime owns the boundaries a model tends to skip
under pressure, and the model owns the inside of a tick.

A message to a working goal is a **nudge** by default: it adds to the work in
hand and keeps the plan. `--steer` (or `/steer TEXT` in the panel) is the heavy
form: it supersedes conflicting instructions, voids the plan and may move the
goal, so it has to be asked for. Both are filed verbatim in a `steers` table
with their kind, as well as tagged in the transcript, so a compaction and the
finish report can quote the human rather than a summary of them. After a compaction it restates the goal and every
steer word for word, bounded at 8 KB with the newest steers kept whole.

`define_done` states what done looks like as something checkable, and `verify`
records what the check actually showed. `finish` is refused while the goal has
no check, while a steer is waiting, or when the check has not been exercised
since the last steer. `todo` keeps the goal's checklist (pending, one item in
progress, done) and what blocks it, pinned into every turn so it survives
compaction and shown above the composer; `finish` wants every item done or what
is left named. `checkpoint` tars named paths before a destructive or
milestone-crossing action and `rollback` puts them back.

Before a graded goal completes, `finish` runs two checks. The **gate** is an
optional shell command given to `define_done` (`gate`), run in the agent's home;
a non-zero exit refuses the finish and its output becomes the next prompt. The
**judge** is a fresh model call with no tools that sees the goal, the done check,
the human's steers and the recorded evidence, not the agent's reasoning, and
answers met, not yet or wait. Not yet refuses the finish with what is missing;
after three in a row the advisor is asked what the agent should try differently,
and its advice goes back with a fresh set of tries (without an advisor, or past
its per-goal calls or daily budget, the goal pauses for you); wait means the work is done but its proof is still
running (a build, CI, a service starting): the agent is pointed at its `wait` tool,
and six waits pause the goal. The judge never declares a goal impossible: whether to
stop is the agent's call, and a blocked agent asks its human; a model that answers
"impossible" is read as not yet. An unreadable reply is asked once more, then
accepted unchecked. The judge runs on the
agent's own model in a fresh context by default (free on the local server); set
`judge_model` to `"advisor"` in the resident's config to use the advisor's model.

A reply cut off at the output limit runs nothing; the agent is told at its next
step and tries again in smaller pieces (three cut-offs in a row pause the goal).
Every turn's context states the reply limit, which model answers, and the measured
context use and distance to compaction.

The goal's gate (the shell check `define_done` names) also runs after every turn
that changed something (an edit, a write, a terminal command that is not
read-only), before the next model call; the agent hears only a change (the gate
starting to fail, with its output, or passing again). A per-turn run gets two
minutes: a gate that takes longer is off per turn for that goal (it still runs at
finish), and one slower than 30 s runs per turn at most every ten minutes.
`gate_each_turn false` turns it off.

The failure ledger keys a call by tool and arguments: a call that has failed
twice is refused before it runs and the model is told to change approach, and two
refused repeats of it pause the goal. The watchdog pauses a goal that spends
without moving — no progress for `idle_timeout` seconds, a prompt over
`token_budget`, the same call failing three times, or three replies in a row cut
off at the output limit — and records why in the goal's error, where the panel
shows it.

Compaction has two tiers. When the context passes the threshold, old tool
outputs (outside the newest eight messages) are first replaced by a one-line stub
and the path of a file holding them, with no model call; a summary runs only if
that is not enough. A paid brain (remote, or through a key relay) compacts early,
at 38 % of its window down to 7.6 % (`paid_compact_threshold`,
`paid_compact_target_ratio`); a local one at 85 % down to 20 %.

A plan-first goal (`send --plan`, `/plan TEXT`) explores before it changes
anything: `write`, `edit`, `checkpoint`, `rollback` and `ask_root` are refused,
and a terminal command runs only when every part of it is a read-only program
(`ls`, `grep`, `git status`, ...; anything unclear counts as a change). The agent
files its plan with `propose_plan` and waits (`waiting_plan`, no timeout) until
`agent plan NAME GOAL approve|reject [--note TEXT]`.

The emergency stop: `caminus ember stop-all [--reason TEXT]` (`/halt`) writes a
STOP file in the bridge folder that only you can remove (`stop-all --clear`).
Every resident pauses its working goals after the current step, starts nothing
new, and ends the commands running in its terminals and any process its account
detached (TERM, then KILL ten seconds later); lifting the stop resumes nothing.

`--no-loop-gates` records `loop_gates false` for fixture agents that predate the
loop; real residents get the gates by default.

## Tools

Besides `terminal` (its `timeout`, 1 to 3600 seconds, kills a command that runs
longer and the result says so, as `sandbox`'s does; `yield_ms` only bounds how long
one call waits for output), `discover`, the loop's tools and the bridge
(`ask_user`, `ask_root`), a resident has:

| Tool | What it does |
|---|---|
| `read`, `edit`, `write` | Files without the terminal: numbered windows; exact, unique replacement (CRLF and mode kept, atomic); whole-file writes; edits return a diff |
| `todo` | The goal's checklist (see the loop) |
| `remember`, `history_search` | A one-fact note with its index line; full-text search over the agent's own past goals, results, journal and notes |
| `wait` | Block without model calls until a time, a file, a port, a process, a terminal session or a message; a message, cancellation or new work ends it early |
| `machine` | GPUs, memory, disk, load, the model servers, its sessions, peers and leases |
| `lease` | A visible, expiring claim on GPU cards and memory, checked against the human's GPU windows; a claim, not a lock |
| `report`, `budget`, `schedule` | fyi or need_you to the human; what the goal has used and has left; park the goal and wake it later |
| `web_search`, `web_fetch` | Search through [ddgs](https://pypi.org/project/ddgs/) (installed in the agent's account: `python3 -m venv ~/.venvs/ddgs && ~/.venvs/ddgs/bin/pip install ddgs`); a page as readable text |
| `sandbox` | Code the agent did not write or does not trust (exploit proofs of concept, CVE reproductions, unknown repositories, install scripts) in a [bubblewrap](https://github.com/containers/bubblewrap) box: nothing of the agent's home, `/usr` and `/etc` read-only, a private `/tmp`, no GPUs, no D-Bus, display or host services, no capabilities. `folder` is mounted read-only at `/work` (refused: a resident's workspace or its state, the account home, a hidden folder of the account, a folder holding the token); `/box` (`boxes/GOAL` in the workspace) is writable, kept across the goal's sandbox calls and removed when the goal completes or is cancelled (a paused or failed goal keeps it, since it can resume). No user namespace can be made inside (`--disable-userns`). No network unless the call names `hosts`: then a proxy outside the box allows HTTPS (port 443) to those names and their subdomains when every address is public, and each connection shows as an event and in the card's expanded view (one made after the last poll, when the goal ends). Limits per call: `timeout` (600 s, at most 3600), 4 GB, half the CPUs, 512 processes (config.json `sandbox`: `memory_mb`, `tasks`, `cpu_percent`, `limiter`) through a user systemd scope, else rlimits; `/box` at most 4 GB on disk (`disk_mb`, checked every 2 s) and the tmpfs `/tmp` and `/run` at most the memory limit; going over kills the box and the result names the limit. The connection log keeps 1000 connections per box and a result returns at most 50 new ones (`connections_more` counts the rest). Polled like a terminal session |
| `advisor` | A stronger model's advice, within the owner's limits (below) |
| `mcp__SERVER__TOOL` | Tools from the agent's MCP servers (below) |

A tool result past about 8K tokens has its large text saved under `~/outputs/`;
the model gets the head, the tail and the path. Web pages, search results and MCP
results are marked as data, never instructions.

## Memory

`STATE.md`, `MEMORY.md` and `notes/` are in [the Ember guide](ember.md#its-memory).
`STATE.md` is asked for near compaction, so it is what the agent reads after a
compaction or a restart. An older, longer `MEMORY.md` is copied whole into
`notes/memory-before-index.md` the first time it is seen. Answers to questions stay
in the goal's transcript; nothing is written to memory automatically.

## Free time

With no work queued, a resident has free time: it settles in, reads its journal and
notes, and follows what interests it (research, writing, making things in
`~/studio`, tending up to three projects), writing a short intention as a todo and a
journal entry before it stops. It is never graded. Work preempts it at once; a
nightly dream consolidates notes. On the local server it works in the full context
window: before a stretch overwrites the human's conversation, the prompt cache
parks it to disk and the human's next request restores it. `caminus ember agent
heartbeat NAME on|off|status|recap|now`; the recap is the "while you were away" line, and `now` (the panel's
`caminus ember agent heartbeat NAME now`) starts a stretch at once. A stretch ends at `--stretch-minutes` (30 by default); `--stretch-steps N` adds a
step limit (0, the default, means none); `--rest-minutes` sets the rest between stretches and `--budget-usd` the daily allowance.
On a remote model free time spends its own allowance (`free_time.budget_usd`,
0.50 USD a day by default).

## When you are away

Nothing waits on you. A question you have not answered within `question_timeout`
(900 seconds by default; free time uses its own shorter window) is withdrawn by the
resident itself, never answered in your name, and the agent carries on: with the
advisor's pick when an advisor is configured, else with the option it recommended,
recording the choice in `STATE.md` and its finish. Silence is never approval: a
permission request (an irreversible change, an action in your name, spending money;
`ask_user` with `approval`) that times out is a no, and an unanswered root request
is not run. Each one is recorded (`question_unanswered`, `root_unanswered`), and you
can steer the goal when you are back.

## Brains: providers, models and effort

A resident's brain is any model its providers serve. A provider is an endpoint the agent can reach: the local
Caminus server, or a paid API through your key relay (the agent never holds the key):

```sh
caminus ember relay install --provider anthropic --cap-usd 3 --model claude-haiku-5-5   # http://127.0.0.1:8765/v1
caminus ember relay install --provider deepinfra --cap-usd 3                              # http://127.0.0.1:8766/v1/openai
caminus ember agent provider NAME add anthropic --kind anthropic --endpoint http://127.0.0.1:8765/v1 --relay
caminus ember agent provider NAME add deepinfra --kind deepinfra --endpoint http://127.0.0.1:8766/v1/openai --relay
caminus ember agent models NAME --refresh        # what each provider serves: window, output cap, effort levels, prices
caminus ember agent model NAME zai-org/GLM-5.2   # pick one (provider/id when two serve it)
caminus ember agent effort NAME high             # among the levels that model offers
```

Each relay reads its key from a file only you can read (`~/.config/anthropic/api_key`,
`~/.config/caminus/deepinfra-key`; it refuses one others can read) and keeps a daily dollar cap. `relay install`
makes it a systemd user service with no expiry (the cap guards the spend) that runs a frozen copy of its code, so
editing or moving a checkout cannot stop it; `relay status` shows each relay and today's spend, `relay uninstall`
removes one. (`caminus ember relay --provider P --cap-usd N --ttl SECS` runs one in the foreground instead.)
The Anthropic relay allows the models you name; the DeepInfra relay allows any chat model DeepInfra publishes a price
for, priced from its own list. The catalog is what the providers report: Anthropic's model list gives each model's
window, output cap, thinking and effort levels; DeepInfra's gives window, prices and whether it takes
`reasoning_effort`. In the panel, `/model` opens the list (Enter picks) and `/effort LEVEL` sets the depth; a running
resident takes either at its next turn.

A paid brain resends its whole context every turn, so it compacts early: at 38 % of its window, keeping the newest
7.6 % (on a 1M window about 400K and 80K). Set `paid_compact_threshold` and `paid_compact_target_ratio` (fractions of
the window) in the resident's config to change it; the local server keeps `compact_threshold` / `compact_target_ratio`
(85 % / 20 %).

## The advisor and two brains

`caminus ember agent advisor NAME --model claude-opus-5-5 --endpoint URL` gives the
agent an `advisor` tool on a model the owner picks, through the owner's own key
relay, with a daily dollar cap and calls per goal. `caminus ember agent fallback NAME
--model ... --endpoint ...` adds a second brain used only while the local server is
down, with its own daily allowance. Every failed or refused remote call is recorded
in the resident's `failures.jsonl`.

## MCP servers

`caminus ember agent mcp NAME add SERVER [--env KEY=VALUE] -- COMMAND ARGS...` adds a
Model Context Protocol server (stdio); `mcp NAME` lists them (environment values are
never printed), `mcp NAME remove|off|on SERVER`. The client speaks the stateless
2026-07-28 revision and falls back to the `initialize` handshake for older servers.
A server starts on first use and stops with the agent; its log is
`~/mcp-logs/SERVER.log`. Example, a read-only GitHub server:

```sh
caminus ember agent mcp NAME add github --env GITHUB_PERSONAL_ACCESS_TOKEN=... -- \
  github-mcp-server stdio --read-only --toolsets=repos,issues,pull_requests
```

## Discover before building

The resident prompt authorizes routine reversible work with the account's shell,
web access, installed browsers, user-level software and scheduled jobs, subject
to the human's goal. It explains the root and question bridges, persistent PTYs,
memory and shared-resource rules. Read the authoritative prompt in
`python/caminus/ember/resident/prompt.py`, or print it with the current account and
workspace values:

```sh
caminus ember agent prompt resident
caminus ember agent discover resident --json
caminus ember agent discover resident --topic commands --query pdf --json
caminus ember agent discover resident --topic packages --query tshark --json
caminus ember agent discover resident --topic containers --json
```

These commands work before initialization and also accept `--local` for the
current account. They never call the model. Each new goal automatically receives
one discovery snapshot before its first model turn. That snapshot is saved in
the conversation and journal, so retries do not reinsert it and changing RAM or
disk readings do not continually alter the system-prompt prefix.

The model's `discover` tool provides focused hardware, command, web, image,
document, networking, development, Python, container and package information.
Overview inspects metadata without launching discovered programs, importing
third-party modules, reading user documents, contacting the internet or
initializing GPUs. Python results describe the current interpreter only.
Container queries explicitly use a local Docker socket or local podman; package
queries read installed status and cached apt policy. These queries have a
three-second timeout and a bounded returned output. No image is run or pulled,
and no package is installed or index updated by discovery.

The prompt requires checking these capabilities and a tool's documentation or a
small relevant smoke check before building or installing an alternative. Presence
is labeled as metadata, not proof of working functionality or an exhaustive scan
of every environment. The model may use apt; system installs and index updates
requiring root go through `ask_root` with exact commands and the effect and undo
plan. A task's explicit offline or installation restrictions still apply.

## Scriptable oneshot testing

Daemon and oneshot call the same runtime and tools. `once` submits a goal and
waits for its result. If a daemon already owns the loop, it queues and observes
that daemon. `--jsonl` streams goal events followed by a result record; `--json`
returns a result object. Goal text can be positional, `--goal TEXT` or stdin.

```sh
caminus ember agent once resident --stdin --jsonl --max-steps 12 --max-seconds 120 < goal.txt
caminus ember agent events resident --after 0 --json
caminus ember agent once resident --goal-id GOAL_ID --json
caminus ember agent cancel resident GOAL_ID --json
```

`caminus ember agent stream resident` prints NDJSON for the reply being generated:
one line per change, the first line of a turn carrying the whole text (`reset`)
and later lines only the suffix, then a `done` line when the turn lands. It is
what the panel's live tail reads; `--once` prints the current snapshots and
exits, and `--seconds N` bounds a wait. Nothing it prints is durable.

Test budgets are optional, persist with the goal, and pause further actions;
they are not a hard process timeout for already running terminal jobs. `cancel`
sends TERM to the goal's terminal sessions. Root and clarification waits survive
process exit. Resuming uses the same goal and pending action, without submitting
the original command again. Finished or cancelled goals cannot be replayed.

Cancellation withdraws the goal's pending `ask_root` and `ask_user` requests. The CLI and panel recheck the
exact reviewed request under a shared lock before approving it, then save an
execution claim before starting sudo. A claim that wins before cancellation is
already committed; cancellation reports its ID and does not erase its result.
An interrupted claim requires human inspection and is never automatically rerun.
For scriptable root approvals, pass the reviewed request's `_digest` from
`env reqs --json` as `env answer ID approved --digest HASH`. Approving by ID alone
is rejected. An edited command still requires its reviewed request digest.
Interactive approvals display terminal controls and Unicode formatting characters
as visible escapes in commands, reasons and request metadata. The displayed
escaping does not change the raw command bound to the reviewed digest.

| Exit | State |
|---|---|
| 0 | completed |
| 20 | waiting for human clarification |
| 21 | waiting for human root approval |
| 22 | paused by a test budget |
| 23 | cancelled |
| 24 | waiting for an unavailable model |
| 1 | error |

For CPU fixtures or agentic tests in an isolated local home, every command also
accepts `--local`; this selects the same implementation without SSH. Initialize
each scratch agent with an isolated `--home` and `--bus`. Remote operation remains
the default. Credentials, when needed, are read from the configured environment
variable, not saved in config.json. The user service reads an optional
`~/.config/caminus/agents/NAME.env` owned by the agent account.

## Peer communication

Initialize each named resident, normally with the same local mail bus:

```sh
caminus ember agent init reviewer --model MY_MODEL --endpoint http://127.0.0.1:8092/v1
caminus ember agent start reviewer --json
caminus ember agent peers resident --json
caminus ember agent message resident reviewer --text 'Inspect the test results.' --json
caminus ember agent mail reviewer --after 0 --json
```

The model has `peers` and `send_message` tools. Mail records the sender, recipient,
sender goal, optional recipient goal and reply linkage. Without a recipient goal,
delivery creates a new goal; with one, it queues steering input for that goal.
Offline mail persists. The recipient commits delivery before acknowledging it,
so recovery cannot create duplicate goals. Recovered sends reuse the action ID.
Messages to finished goals are retained with a delivery error; send a new goal
instead. Human waits stay paused when peer messages arrive.

`--reply-to MESSAGE_ID` links a reply. `--recipient-goal GOAL_ID` directs it to an
existing goal. Use `--stdin` instead of `--text` for scripts. `mail` exposes
delivery status and errors; `events` exposes received messages.

The default bus is `~/.local/share/caminus/agents/mail.sqlite3`; `init --bus PATH`
selects another. Names cannot silently register a different home on the same bus.
This is communication between residents sharing a Unix account. Sender names
provide attribution, not authentication against processes with the same UID.
Peer messages are data, cannot approve root, and do not override the human's goal.
Separate Unix accounts or remote-machine routing are not implemented.

## Compare Qwen Code and the resident

`benchmarks/resident_compare.py` gives both CLIs the exact same goal bytes and
copies the same seed files into a fresh workspace for every run. It alternates
order, records wall time for each agent invocation and runs an independent
verifier afterward. A zero exit code with wrong files fails. Timeouts, waiting
states and failed setup remain failures. Speedup is emitted only if every run
passes correctness; median times of passing runs and success counts are retained.
The default is three runs per harness.

The included `benchmarks/cases/resident-report` task checks a CSV-to-JSON program,
including exact decimal arithmetic, rounding, quoted Unicode fields, bad input
and preserving existing output on failure. The verifier runs outside the working
directory. Agents should not receive the verifier as part of their prompt.

Provide JSON argv arrays with `--qwen-argv` and `--resident-argv`. The goal travels
on stdin; the process working directory is the fresh workspace. `{workspace}`,
`{run_dir}` and `{goal_file}` are literal argv placeholders. Optional
`--qwen-prepare` and `--resident-prepare` perform setup outside the measured
interval. For the resident, initialize its home at `{workspace}` in preparation
and then run `once --stdin --json` against that home. Use a separate mail bus for
each run. Do not attach to a production daemon for this comparison.

`--profile FILE` records the model, endpoint, actual context, sampling, output
budget, reasoning mode, hardware and client versions/configuration used by both
CLIs. Configure both clients to match; the runner does not force provider flags
or certify a caller's profile. Disable unrelated advisors and fallback providers.
The harness system prompts and tool interfaces will differ by design. Report
those settings and cold/warm server-cache conditions with any timing claim.

```sh
python -I benchmarks/resident_compare.py --help
python -I tests/ci_cpu.py --scratch build/resident-cpu --only env,resident,resident_compare
```

Model-backed comparisons must run through the repository GPU guard in an agreed
window. The comparison runner never starts, stops or preempts the model server.
Its timeout terminates only the CLI's process group; durable resident terminal
workers may need an explicit goal cancellation afterward. CPU fixture tests
validate the comparison procedure, not live-model speed or correctness.

The model's output-limit stop reason pauses a goal instead of reporting it
completed or executing partial tool calls. Increase its output budget and resume
that goal. Malformed tool arguments become durable tool errors so the daemon can
continue working rather than restart on the same action.

Context management counts the full prompt, including the system and tool schemas.
`agent init --compact-threshold .85 --compact-target-ratio .2` starts compaction
at 85% of the configured context and requires the resulting prompt to fit within
20%. It may compact earlier to reserve the next response. Recent assistant/tool
groups are retained together when they fit; larger observations enter the summary.
The original goal remains pinned. An oversized goal or persistent memory pauses
the run instead of sending a request that cannot fit. Compaction events record
the before/after token counts and targets. Every generation also reserves output
room against the model window.
`--compaction-model` selects a summary alias on the same endpoint; a no-thinking
alias can summarize without using the summary budget for hidden reasoning.

## Spend, updates and the demo

The relay stops at its daily cap (HTTP 402) and refuses a model it cannot price; the agent then waits for the next
day or another brain. Every priced turn is metered per goal and per day, cached input at the provider's cache price
(`agent status` reports `spend_today` and `spend_goals`; the panel's status line shows this goal's and today's).

`caminus ember agent update NAME` (`/update`) installs this caminus's runtime: the resident applies it between goals
(`--now`: at the next step), restarts its own service and reports the new version in its heartbeat (`--wait SECS`
waits for it).

`tools/ember_demo.py` runs all of this on a scripted model in a scratch home (no agent account, GPU, server or key):
the quickest way to see the chat.
