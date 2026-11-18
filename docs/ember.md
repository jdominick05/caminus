# Ember: the anti-harness

<p align="center">
  <img src="ember-chat.svg" alt="Ember's chat: an agent reads the code, states what done looks like, fixes a bug, runs the tests, and the judge checks the finish" width="100%">
</p>

Ember (`caminus ember`; `caminus env` still works) gives an AI agent a place to live on your machine: its own Unix account, its own files, its own
long-running loop. You give it goals and steer it as it works; when there is nothing to do, it has free time. New
here: [the quickstart](ember-quickstart.md). Every command and panel view: [the reference](resident-agent.md).

## What it is

Most coding agents are a harness around your terminal: they run in your session, as you, for as long as the window
is open, and each tool is something the harness hands them. `caminus ember` turns that around (hence anti-harness):

- **The agent is a resident, not a guest.** It runs as `caminus-agent`, a normal user with its own home, crontab,
  user services and installed tools. The terminal is the whole operating system, not a list of approved apps. Beside
  it, file tools: `read` (a line range: `start`, `lines`), `search` (file names by glob, text by regex with context,
  following .gitignore, skipping hidden folders outside a repository, using ripgrep for plain text when installed),
  `edit` (exact text, with the diff) and `write`.
- **It keeps going.** A systemd user service runs its loop, so it outlives your terminal and your logout. Goals
  queue; long commands keep running between turns; an interrupted step is reported, not replayed blind.
- **You steer, it decides how.** A message reaches it at its next step; `/steer` redirects. It defines what done
  looks like before it changes anything, verifies against that, and a separate judge (a fresh model call that sees the
  evidence, not the agent's reasoning) checks the finish. The harness keeps its own record too: a verify needs
  something that looked after the last change, a refused edit blocks the finish until the change is made or the file
  is read, and the judge sees which files were actually changed.
- **Nothing waits on you.** Questions and root requests are cards you answer when you can. An unanswered question
  waits 15 minutes while you are at the screen and longer while you are away (until 15 minutes after you come back,
  at most 8 hours); then the advisor's pick or the agent's recommended option. An unanswered permission is a no, and a
  root request is a no after 15 minutes either way.
- **It has free time.** With no goal, it reads its journal, follows its own interests, keeps projects in its studio,
  and leaves you things it thinks you will like (`/studio`). Free time is never graded and spends its own
  small allowance.
- **Any brain.** A local `caminus serve` model, a paid API (Anthropic, DeepInfra, Mistral, Meta) or your OpenCode Go plan, through a key relay you run.
  On a `caminus serve` brain the chat's speeds are the engine's own: "read 1,240 · write 62 tok/s" is how fast it read
  the last reply's prompt and wrote the reply. While a reply streams, the write speed is the engine's live figure, its
  mean over its most recent passes, so it can step when the reply ends.
  Any other brain (llama-server included, whose timings Ember does not take as the engine's) gets Ember's own
  estimate, marked "est.".

Compared with Qwen Code and similar harnesses, the difference that matters most is what a harness cannot do: keep
working when you close the window. On speed and cost there is one small measurement so far: a single bug-fix task (a
buggy library, ten unit tests, the tests not to be edited), three alternating runs each, both harnesses on the same
model (DeepSeek-V4.1-Flash on DeepInfra) with the same relay and budget. All six runs passed; the medians were 16 s and
0.005 USD for the resident against 20 s and 0.006 USD for Qwen Code. That is too few runs on too small a task to call
either faster.

## The security model

- **A separate account, not a sandbox.** The agent has exactly the rights of a normal user named `caminus-agent`. It
  cannot read your home (mode 750 by default on Ubuntu) or other users' protected files, and it is not root. It can
  read anything world-readable, like any user can, and it can use the network. This keeps it out of your files by the
  ordinary Unix rules; it is not containment against a hostile program.
- **A box for untrusted code.** Code the agent did not write or does not trust (an exploit's proof of concept, an
  unknown repository, an install script) goes to its `sandbox` tool, not its terminal: a bubblewrap box that sees none
  of its home, no GPU and no network unless the call names hosts, with time, memory and process limits. Its prompt
  asks for this; nothing forces it. What the box sees: [the tool reference](resident-agent.md#tools).
- **A browser in a box.** Its `browser` tool drives a headless Chrome in the same kind of box, with a fresh profile
  per goal and no logins; public sites are open to it and logged, your LAN and this machine only when a call names
  the address ([the tool reference](resident-agent.md#tools)). It needs Google Chrome or a Chromium that is not a
  snap (bubblewrap cannot run snaps: Ubuntu's chromium and Firefox are snaps). Install Google Chrome's `.deb` from
  google.com/chrome (`sudo apt install ./google-chrome-stable_current_amd64.deb`); without one the tool is not
  offered. Its memory is bounded through a user systemd scope; an account without one runs the browser with no
  memory limit.
- **Root only with your approval.** The agent asks through `ask_root` with the exact command, the goal it unblocks,
  the expected effect and how to undo it. You approve, edit or deny it in the panel; approved commands run under your
  sudo, never the agent's. A grant (`g g`) lets that exact command repeat for the rest of its goal, at most 12 hours.
  An approved command runs without input (package tools ask nothing, no pager opens), and the chat comes back when it
  exits. After 30 minutes the chat comes back anyway, and the agent hears that the command is still running and its
  pid; nothing kills it.
- **You reach it; it does not reach you.** You log in to its account over SSH with your key, from this machine
  only, without forwarding. It has no way into your account.
- **Keys stay with you.** A paid brain goes through a relay: a service in your account that holds the API key, allows
  only priced models and stops at a daily dollar cap. The agent sees an endpoint, never the key.
- **The bridge.** `/srv/caminus-env-bridge` (root:caminus-env) is the only shared place: requests, answers, notes, GPU
  leases, and the emergency STOP file. The folder is sticky, so the agent cannot delete your STOP file.
- **The emergency stop.** `/halt` (or `caminus ember stop-all`) pauses every resident after its current step and ends
  the terminals it started; `/halt clear` lifts it.
- **GPUs.** The agent's account can use the GPUs (it is in `render` and `video`). Its prompt tells it to take a lease
  first and run GPU work under the repository's guard; nothing but the agent enforces that, so keep the GPU groups off
  its account (`sudo gpasswd -d caminus-agent render`) if you do not want it near them.
- **Everything is recorded.** Your audit log has every root request, answer, grant and panel control; its journal
  has every goal, tool call and result.

## Paid brains

A paid model is reached through a **key relay**: a small service in your own account that holds the API key, forwards
only the provider's chat endpoints, allows only priced models, and refuses requests once the day's dollar cap is spent
(HTTP 402; it resets at midnight). The agent is given the relay's local address, never the key.

| Provider | Key file (mode 600) | Relay address | Agent `--api` |
|---|---|---|---|
| Anthropic | `~/.config/anthropic/api_key` | `http://127.0.0.1:8765/v1` | `anthropic` |
| DeepInfra | `~/.config/caminus/deepinfra-key` | `http://127.0.0.1:8766/v1/openai` | `openai` (the default) |
| Mistral | `~/.config/caminus/mistral-key` (and its price file, `~/.config/caminus/mistral-prices.json`) | `http://127.0.0.1:8767/v1` | `openai` |
| Meta (the Muse models) | `~/.config/caminus/meta-key` | `http://127.0.0.1:8768/v1` | `openai` |
| OpenCode Go (your subscription) | `~/.config/caminus/opencode-key` | `http://127.0.0.1:8769/inference/go` (the plan's root) | per model: `openai` or `anthropic` (`/model` picks it) |

Anthropic, step by step (the others follow the same pattern; the quickstart shows DeepInfra):

```sh
mkdir -p ~/.config/anthropic
[ -e ~/.config/anthropic/api_key ] || install -m 600 /dev/null ~/.config/anthropic/api_key
$EDITOR ~/.config/anthropic/api_key                       # the key alone, on one line
caminus ember relay install --provider anthropic --cap-usd 3 --model claude-haiku-5-5
caminus ember agent install
caminus ember agent init resident --model claude-haiku-5-5 --api anthropic --endpoint http://127.0.0.1:8765/v1
caminus ember agent provider resident add anthropic --kind anthropic --endpoint http://127.0.0.1:8765/v1 --relay
```

- **Which models.** The Anthropic relay allows only the models named with `--model` (repeat it; Claude Haiku 5.5 when
  none is given). DeepInfra allows any chat model it publishes a price for. Mistral publishes no prices, so its relay
  allows only the models in your price file: JSON, USD per million tokens, for example
  `{"mistral-large-latest": [2.0, 6.0]}` as `[input, output]`, re-read when it changes; then
  `caminus ember relay install --provider mistral --cap-usd 3`.
- **Meta.** Meta's Model API serves the Muse models: Muse Ember (Meta) (`muse-ember-1.0`, the cheap one: $0.10 in,
  $0.50 out, $0.004 cached per million tokens, three times that past 200K input tokens; a 1M window, 524K output),
  Muse Spark (`muse-spark-1.3`, `-1.2`, `-1.1`) and Muse Wonder (`muse-wonder-1.0`). Create a key in Meta's Model API
  dashboard under API keys ([its guide](https://dev.meta.ai/docs/authentication)); the prices are
  [Meta's table](https://dev.meta.ai/docs/pricing-rate-limits), built into the relay, so it allows only those models
  and needs no price file. The Contributor-mode ids (cheaper because Meta trains on your prompts and replies) are
  refused. Then:

  ```sh
  mkdir -p ~/.config/caminus
  [ -e ~/.config/caminus/meta-key ] || install -m 600 /dev/null ~/.config/caminus/meta-key
  $EDITOR ~/.config/caminus/meta-key                        # the key alone, on one line
  caminus ember relay install --provider meta --cap-usd 3
  caminus ember agent provider NAME add meta --kind meta --endpoint http://127.0.0.1:8768/v1 --relay
  caminus ember agent model NAME muse-ember-1.0
  ```

  Pick the model with `agent model` (or `/model`); a session from `agent init --model muse-ember-1.0` knows the Muse
  levels too. Past 200K input tokens every Muse Ember (Meta) turn costs three times as much, so it compacts at 85 % of
  that line (about 170K, 16 % of its window) instead of a paid brain's usual 38 % (Compaction, below).
- **OpenCode Go.** Your Go subscription as the brain: a Console API key from the workspace that has the plan. The
  relay offers the plan's chat models that call tools, each through its own API (Chat Completions for DeepSeek, GLM,
  Kimi and the others; Messages for Claude Haiku 5.5, MiniMax M3 and the Qwen3.8 / Qwen4 models; the GPT and Grok
  models through Chat Completions, which OpenCode translates). Gemini (its own API) and the models whose prompts train
  a model (the Contributor ones) are left out. Prices, windows and levels come from the plan's own model list.

  ```sh
  mkdir -p ~/.config/caminus
  [ -e ~/.config/caminus/opencode-key ] || install -m 600 /dev/null ~/.config/caminus/opencode-key
  $EDITOR ~/.config/caminus/opencode-key                    # the key alone, on one line
  caminus ember relay install --provider opencode-go --cap-usd 3
  caminus ember agent provider NAME add opencode --kind opencode-go --endpoint http://127.0.0.1:8769/inference/go --relay
  caminus ember agent model NAME deepseek-v4.1-flash
  ```

  OpenCode watches plan traffic for abuse, so Ember behaves as a client the plan expects: each conversation (a goal)
  sends one random session id (`x-opencode-session`; the plan refuses requests without one, and so does the relay),
  every request says it is Ember (`ember/VERSION`), and a refusal from the plan (a usage limit: 429, or any other 4xx)
  stops the goal with its words instead of retrying. Free time stays off the plan: it runs on your fallback brain when
  that is not a plan, else rests and says so; `caminus ember agent heartbeat NAME --on-plan yes` lets it use the plan.
  On the plan, free time's daily dollar allowance (`--budget-usd`) does not limit it: a turn inside the Go allowance
  costs $0, so only the stretch's time and steps bound how much of the allowance it uses.
  Each reply says what it cost: $0 inside the Go allowance, dollars from your balance past it (with Extra Usage on).
  The daily cap counts those dollars (a charge that is not a finite amount of at least $0, a $0 billed to anything
  but the Go allowance or a free model, or one sent before the reply ended is booked at the list price instead);
  `relay status` shows the allowance used beside them. A reply cut before its cost
  arrived is booked at the list price times any peak-hour multiplier in force (DeepSeek costs twice as much on weekday
  peak hours), so an estimate errs high.
- **Changing a relay.** `relay install` rewrites the service from the options you give it and restarts it (today's
  spend carries over), so give the whole set again: `--cap-usd` is required, and every `--model` you still want.
- **Several providers.** Install one relay per provider and add each with `agent provider NAME add`; `caminus ember
  agent models NAME --refresh` lists what each serves (context windows, output caps, prices), and `/model` in the chat
  (or `caminus ember agent model NAME ID`) switches between them at the next turn.
- **Spend.** `caminus ember relay status` shows each relay's spend today against its cap. The chat's status line shows
  the agent's own running cost.
- **The advisor.** An agent can ask a stronger model for advice (`caminus ember agent advisor NAME --model
  claude-opus-5-5 --endpoint http://127.0.0.1:8765/v1 --budget-usd 1`; it always speaks the Anthropic API). It goes
  through the same relay, so allow its model there too, with the cap restated:
  `caminus ember relay install --provider anthropic --cap-usd 3 --model claude-haiku-5-5 --model claude-opus-5-5`.
  Its own daily budget is counted by the agent, inside the relay's cap. `agent advisor` warns when the relay it names
  would refuse the model, and `caminus ember relay status` lists the models each Anthropic relay allows. When the
  advisor cannot be called (the relay refuses its model, is not running, or has spent its cap), `/advisor` opens the
  reason and the exact command to run, the agent's advisor tool says the same, and the attempt does not count against
  the goal's advisor calls. A running agent picks up an advisor you add or remove without a restart. While the advisor
  is asked (`/advisor`, or the agent's advisor tool), a popup over the typing box (the one `/btw` uses) shows who asked,
  the advisor's model and the seconds it has taken; it goes when the advice lands as a card (rendered as Markdown, as
  the agent's replies are; Ctrl+Y shows its source), and says why when the call fails (a spent budget, no calls left).
- **Eyes.** A brain without vision (every local Caminus model) can still look at an image, once you turn eyes on:
  when it reads one, Ember sends that image to a vision model on your relay, and the description is the read's result.
  With the image go the agent's question (read's `question` argument, else "describe what matters in this image for
  the task") and the first 600 characters of its goal, so the eyes model knows what matters. The chat's read card says
  `image WxH · described by MODEL`. Eyes are off until you turn them on, whatever relays the agent has (a session
  made with `create NAME --model` starts without them, as without an advisor or fallback); until then the read tells
  the agent its image's size and format and how you turn them on. Turn them on with
  `caminus ember agent eyes NAME --auto` (ministral-8b-latest on a Mistral relay provider, else Claude Haiku 5.5 on an
  Anthropic one) or a model of your choice, `--provider mistral --model ministral-14b-latest`. `--off` turns them off
  again, and `--budget-usd` sets their daily budget ($0.25 by default, counted by the agent like the advisor's and
  inside the relay's cap). The candidates, with the prices the relays are set up with (USD per million tokens,
  input / output):

  | Model | Relay | Price | An 860x480 terminal screenshot | On four images (terminal, chart, form, diagram) |
  |---|---|---|---|---|
  | ministral-3b-latest | Mistral | 0.10 / 0.10 | $0.00008 | every fact found; misquoted one character of an assertion |
  | ministral-8b-latest (`--auto`'s pick) | Mistral | 0.15 / 0.15 | $0.00012 | every fact right, text quoted exactly |
  | ministral-14b-latest | Mistral | 0.20 / 0.20 | $0.00015 | every fact right |
  | claude-haiku-5-5 | Anthropic | 0.10 / 0.50 | $0.00014 | every fact right, the most complete (quotes notes and labels too) |

  The Mistral relay allows a model it has a price for, so add the one you pick to its price file. The Anthropic relay
  allows only its `--model` list: `relay install --provider anthropic --cap-usd 3 --model claude-haiku-5-5` plus every
  model it allows now. A refused model's error tells the agent the exact command or price-file line for you.
- **Removing one.** `caminus ember relay uninstall --provider anthropic` stops it; the key file stays where you put it.

## A second setup on one machine

The defaults (the `caminus-agent` account, the `caminus-env` group, the bridge in `/srv/caminus-env-bridge`) are one
setup per machine. A second person on the same machine, or a trial setup beside yours, needs its own of all three:
with a shared group, either side could write into the other's bridge. Set them in the second owner's shell (its
`~/.profile` keeps them) before the quickstart:

```sh
export CAMINUS_ENV_USER=ember-alex            # its agent account (created by the root script)
export CAMINUS_ENV_GROUP=ember-alex           # the group that owner and that agent share
export CAMINUS_ENV_BRIDGE=/srv/ember-alex-bridge
caminus ember setup                           # prints the root script for these names
```

`caminus ember agent init` passes that bridge to the agent. Each owner's SSH key (`~/.ssh/caminus_env_key`), audit
log and relays are in their own home already; relays listen on fixed local ports, so a second owner's relay takes
another one (`relay install ... --port 8775`, and the agent's `--endpoint` to match).

## The mouse: clicks and copying

The chat (`caminus ember ui`) takes the mouse: the wheel scrolls the history, and a click opens a folded card (only
that one, in place: Ctrl+O opens them all) or folds it again, picks a card in the session picker (Shift+Tab) or presses
the jump-to-newest pill. Taking the mouse turns the terminal's own selection off, so the chat selects text itself:

- **Drag to copy.** Drag over any text on the screen: the conversation and its cards (folded or open), the side pane
  (tasks, now), the header, the composer, the status line, the session picker and every dialog. Releasing the
  button copies the plain text to the clipboard (wl-copy on Wayland, xclip or xsel on X11, and OSC 52 for terminals that
  take it, such as a remote session). The copy has no colours or styling; its lines break where the screen breaks them.
  The status line says how many lines it copied. An open card drawn as Markdown (Ctrl+Y, docs/resident-agent.md) is
  copied as its source, whole, whatever part of its body the drag covers: the rendered table and rules are not text.
- **One area at a time.** A drag selects inside the area it began in, so the conversation never picks up the side pane
  beside it and the other way round. In the conversation, dragging past its top or bottom scrolls and keeps
  selecting. A box's border (the composer's, a dialog's, between the picker's cards) is left out of the copy; a │ or ─
  inside a dialog's content (a diff, a table) is copied as it is.
- **Click or drag.** Pressing the button does nothing yet. If the pointer gets two or more cells away before the
  release, it is a drag and selects; if it doesn't, or it comes back to the cell where you pressed, it is a click, and the click acts where you pressed. So a drag that starts
  on a card or a session never opens or switches it.
- **Clearing.** A click clears the selection; outside the conversation a key or a resize clears it too, since the
  screen under it changes.
- **The terminal's own selection.** Shift+drag still gives it, in terminals that offer one (VTE / GNOME Terminal do).
  `/panel` is the control panel's own screen and keeps its own mouse handling.

## Defaults

Every setting below works without being set. The resident's `config.json` lives in its home; a key in the table is a
key of that file.

| Setting | Default | What it does | Change it with |
|---|---|---|---|
| Compaction, local brain | at 85 % of the window, down to 20 % | old tool output is trimmed first; a summary only if needed | `compact_threshold`, `compact_target_ratio` |
| Window, local brain | the window the server reports now (`caminus serve`'s mode: 262K native, 1M long) | compaction and the status line measure against it | `context_cap` (a limit under it; `agent init --context N` sets it) |
| Compaction, paid brain | at 38 % of the window, down to 7.6 %; a model whose price rises past a prompt length (Muse Ember (Meta) past 200K, Claude Haiku 5.5 past 100K) at 85 % of that length when it comes first, down to a fifth of that | a paid brain resends its context every turn, so it compacts early | `paid_compact_threshold`, `paid_compact_target_ratio` |
| Free time | off | stretches of 30 min with 30 min rests; $0.50 a day on a priced model; 3 questions a day, each waiting 10 min; a dream pass from 03:00; never on an OpenCode plan unless you allow it | `caminus ember agent heartbeat NAME on`; `--on-plan yes`; `free_time.*` |
| Key relay | your key stays with you; `relay install` asks you to name a daily cap (`--cap-usd`); a relay run by hand defaults to $1 a day | the agent spends through it and never sees the key | `caminus ember relay install --provider P --cap-usd N` |
| Advisor | $1 a day, 5 calls per goal, effort medium | a stronger model the agent can consult | `caminus ember agent advisor NAME --budget-usd N --calls-per-goal N` |
| Fallback brain | takes over after 60 s down; $1 a day | a second brain while the first is unreachable | `caminus ember agent fallback NAME --budget-usd N --down-seconds N` |
| A request's wait | 15 min | a question then takes the advisor's pick or its recommended answer, a root request or permission is a no; in the prompt No is preselected for root and GitHub hand-offs | `question_timeout` (seconds) |
| A question's wait while you are away | until 15 min after you come back, at most 8 h | away = no key or mouse in the chat for 5 min; the prompt then says "waits while you're away"; root requests keep their 15 min ([the rule](specs/spec-ember.md#9-questions-steering-context)) | `question_away_max` (seconds) |
| The browser's idle close | 30 min | a browser nobody called for that long closes itself (a paused goal keeps its box, not a running Chrome); the next `open` starts a new one and says so | `sandbox.browser_idle_s` (seconds; `0`: never, it stays open until the goal ends) |
| Effort | the lab's recommendation for the model (below) | how hard the model thinks: none, low, medium, high, xhigh, max, each run as the model's nearest level | `/effort LEVEL`, `/effort default` |
| Generation | per model: the lab's settings (below) | sampling, the reply limit and thinking | `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `max_tokens` |
| Images | what the catalog says the model takes (Anthropic `image_input`, DeepInfra's `vision` tag, Mistral's `vision`, Meta's docs for the Muse models, a local server's `/props` `modalities.vision`); unknown is no | `read` of a PNG, JPEG, GIF or WebP file (up to 3000 px a side and 3.75 MB, else refused with how to shrink it) sends the image with the result; without vision the eyes model describes it once you turn eyes on (Eyes, above), else the agent is told it can't see it, its size and format, and which tools could describe it. Each request reads the file again: one changed since is not sent. The image counts in the context at about one token per 28 px square, at most 4,784. The finish judge gets the goal's newest four images the agent saw when its brain sees; otherwise it is told the agent's account of them is an observation. An image the eyes described is listed with the eyes model's name: the description is that model's observation. One read by a brain without vision and without eyes is listed as unseen: what the agent says it shows is not evidence | `agent init` (asking a local server or relay) and `/model` set `vision`; a new session copies it; a fallback brain: `fallback.vision` |
| Eyes | off; once on, $0.25 a day | a vision model describes an image for a brain that can't see it (Eyes, above) | `caminus ember agent eyes NAME --provider P --model M` or `--auto` (ministral-8b-latest on a Mistral relay, else Claude Haiku 5.5), `--off`, `--budget-usd N`; `eyes` |

Pick the effort level at the start of a chat: on the Qwen3.8 models a change re-reads the conversation once.
A local session finds its server's effort ids by asking the server (at creation, `/model` or `/effort`); a session
made before that keeps running at medium until you pick a level.

### Per model

| Model | Effort (default) | Sampling | One reply at most |
|---|---|---|---|
| Qwen3.8 line on Caminus (the Coder, Swift 1.5, ThinkingCap, Qwen3.8-27B) | xhigh; low, medium and xhigh are its levels (high and max run as xhigh, none turns thinking off) | the server's, from the model file (1.0 / top_p 0.95 / top_k 20) | 64K (the card allows 393K; capped so a runaway reply cannot hold the server) |
| Qwen3-30B-A3B on Caminus | thinking on (none turns it off; it has no other levels) | the server's | 32K (its card) |
| Anthropic | medium; low to max natively. none: thinking off at effort high on Haiku 5.5 and Opus 5, thinking only between tools on Sonnet 5.5; Opus 5.5, Fable and any other model always think, so none runs as low | none sent (the API refuses temperature, top_p and top_k) | the model's listed maximum, at most 64K |
| Mistral Large 4 | high; it has low, medium, high (none runs as low, xhigh and max as high) | 1.0 / top_p 0.95 | the listed maximum, at most 64K |
| Mistral Medium 3.5 | high; it has none and high (low runs as none, medium and above as high) | 0.7 / top_p 0.95 (top_p only at high) | the listed maximum, at most 64K |
| DeepSeek-V4.1-Flash | the provider's | 1.0 / top_p 1.0 | the listed maximum, at most 64K |
| Muse Ember (Meta), Muse Wonder | max (Meta's default); they have minimal, low, medium, high, xhigh and max, and always reason, so none runs as minimal | none sent (Meta's defaults) | 64K (Muse Ember lists 524K; Muse Wonder lists no maximum, and its reasoning counts inside the limit) |
| Muse Spark | high (Meta's default); levels as above | none sent | 64K (Meta lists no maximum) |
| OpenCode Go models | on Chat Completions the model's own, on Messages medium (Ember's default there); each has the levels the plan's list names (DeepSeek V4.1 Flash low, high, max; Qwen3.8 low, medium, xhigh), each level as the nearest; a model that lists none has no effort control | its card's where Ember knows the model, else none | the listed maximum, at most 64K |
| Any other model | the provider's (or the server's) | the provider's (or the server's) | local: 64K; paid: the listed maximum up to 64K, else 4K |

A side call (the judge's check of a finish) runs with thinking off where the model can turn it off, else at its lowest
level (Opus 5.5 and Fable low, the Muse models minimal); a paid model whose effort control Ember does not know (no
level list in the catalog, no default) gets no effort field, so it reasons at the provider's default.

### The web tools

- **web_search** asks the ddgs engines in turn (auto, Bing, Brave, Mojeek, Wikipedia) until one has results; each
  engine gets at most 25 s, the search 75 s. The result's `tried` names what each engine gave (results, empty, or the
  error), and so does the error when none had results. ddgs says "No results found" both for an empty answer and for
  an engine that refuses (a rate limit), so that is named apart from an empty list; when no engine failed outright but
  none had results, auto and Bing are asked once more after 2.5 s. It needs ddgs in the agent's account (the install
  line is in [the resident's tool table](resident-agent.md); `search_python` in `config.json` names another Python that
  has it).
- **web_fetch** makes up to three tries on HTTP 429 and 5xx, waiting what the site's Retry-After asks (else 2 s, then 4 s);
  a site that asks for a wait past 30 s fails at once, naming the wait. A URL fetched again within a goal comes from a
  cache for 10 minutes, marked `cached` (`fresh=true` fetches it again); the cache holds 4M characters in all. A PDF (by
  its content type or its first bytes, up to 30 MB) comes back as text through `pdftotext` when it is installed
  (poppler-utils), else (or when pdftotext fails on it) through a small built-in reader that marks its text partial,
  says what it skipped, and after 5 s returns what it has read so far. `paper=true` on an arXiv abstract URL fetches the
  paper's PDF.

## Its memory

An agent keeps what it learns across goals in its home:

- `STATE.md`: where it stands now (doing, verified, assumed, next). It loads first on every turn.
- `MEMORY.md`: its notes index, one line per note. It loads into every goal, capped at 4 KB, and stays fixed for the
  length of a goal.
- `notes/NAME.md`: one fact each, read when needed. The `remember` tool writes a note and its index line together.

Keeping the index small:

- **The warning.** From 3,277 bytes (80 % of the cap), the goal's context and `remember`'s result say how full the
  index is and how to tidy it.
- **Tidying.** `remember` with action `tidy` takes the whole rewritten index (duplicate lines merged, long ones
  shortened) and the old lines taken out, which move to `notes/index-archive.md`. A pinned line keeps its mark. It
  is refused, with nothing written, when:
  - a note would end up named in neither the index nor the archive;
  - a pinned note loses its line;
  - a new line names a note that does not exist;
  - a line is over 160 bytes, or the index is over its cap.
- **A full index.** A save that would pass the cap moves the least recently used unpinned lines to the archive and
  names them.

`history_search` searches the archive along with the agent's past goals, journal and notes. Archive hits come first
(up to half the limit), each with the note its line names. Every hit gives its source: past goal, index archive, old
index, journal or note. Hits from the running goal come last, marked "this goal", and its own prompt is left out.

## Where things live

| What | Where |
|---|---|
| The agent's account and home | `/home/caminus-agent` (`~/homes/NAME` per resident) |
| A resident's journal (goals, tool calls, events) | `~/homes/NAME/agent.sqlite3` in the agent's account |
| Its config, state, memory and notes | `~/homes/NAME/config.json`, `STATE.md`, `MEMORY.md`, `notes/` |
| Its free time | `~/homes/NAME/journal/`, `studio/` (with `studio/PROJECTS.md` and `studio/for-you/`, what it leaves you) |
| Its instructions | `~/ROBOTS.md` in the agent's account (refreshed by `caminus ember setup`) |
| Its Python | `~/.venvs/caminus-cpu` in the agent's account, filled by `caminus ember setup` and topped up by `caminus ember agent install` (both add only what is missing; a venv left without a working pip is repaired in place, and when it cannot be, setup fails and names the apt package, python3-venv): pillow, requests, beautifulsoup4, lxml, scipy, pandas, matplotlib, ddgs, weasyprint, pypdf (about 110 MB to download, 460 MB on disk), CPU torch and numpy with `--test-venv`. pandoc makes PDFs with `--pdf-engine=$HOME/.venvs/caminus-cpu/bin/weasyprint`; weasyprint needs pango (`sudo apt install libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b libharfbuzz-subset0`; setup says so when it is missing) |
| Your key for its account | `~/.ssh/caminus_env_key` |
| Your audit log | `~/.local/state/caminus/env-audit.jsonl` |
| The setup script | `~/.local/share/caminus/env-setup.sh` |
| Relay keys | `~/.config/caminus/deepinfra-key`, `~/.config/caminus/mistral-key`, `~/.config/caminus/meta-key`, `~/.config/caminus/opencode-key`, `~/.config/anthropic/api_key` (mode 600) |
| Relay spend | `caminus ember relay status` |
| The bridge | `/srv/caminus-env-bridge` (`req/`, `res/`, `notes/`, `leases/`, `STOP`) |

## Its GitHub account

Give an agent its own GitHub account and a token file in its account (the token never leaves it), then link them
with `caminus ember agent github NAME link --token-file PATH --repo OWNER/NAME`.
Its terminals then use that account for `gh` and `git push`, and its commits carry its identity (the noreply email
GitHub links to the account, so a CLA check can find its signature). The token's text never appears in what it sees.

What its token may not do on your repositories (open a pull request from its fork, comment, approve a CI run from its
fork) asks where you type, like a root request: Yes runs it with your own `gh`, exactly as filed; No denies; "Tell it
something else" says what to do instead. Nothing it files is a command: the card holds checked fields, and the command is built from
them on your side.

Your files never move into the agent's account unless you put them there or approve a command that does.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `caminus ember setup` says the privileged step is still to do | Run `sudo sh ~/.local/share/caminus/env-setup.sh`, then `caminus ember setup` again. |
| `FAIL bridge` after the root script passed | Your account joined `caminus-env` but this session predates it: log out and in, or `newgrp caminus-env`. |
| `FAIL ssh login as the agent user` | The SSH server is not running or your key is not authorized: rerun the root script; it fixes both and says which. |
| `caminus ember ui: caminus-panel is not installed` | A source checkout: `cd panel && go build -o caminus-panel .` (the wheel ships it). |
| The agent stopped (`stopped` in the header) | `/start` in the panel, or `caminus ember agent start NAME`. The service restarts by itself after any exit except `/stop`. |
| A goal is `paused` | Its reason is on the goal (`/goals`): the watchdog (no progress, a token budget, the same call failing three times or repeated seven times), three plain replies, or three cut-off replies. Steer it, then `/resume`. |
| HTTP 402 from the relay | The day's cap is spent: it resets at midnight, or raise it with `relay install --cap-usd N`. |
| HTTP 403 "no published price" | The model is not on the provider's price list (or a stale model name in the config): pick one with `/model`. |
| The thoughts box under the history says there are no thoughts | The model answers without reasoning on this provider: `/effort` turns it on, at a token cost. |
| GNOME asks for an admin password | You opened a file in the agent's home from the file manager (an `admin://` path). Read its files with `caminus ember run` or from the panel instead. |
| Too many desktop notifications | `CAMINUS_PANEL_NOTIFY=0 caminus ember ui`. |

Still stuck: `caminus ember status` (each line PASS / FAIL with a fix), `caminus ember agent status NAME --json`, and the
agent's own log: `caminus ember run -s 'journalctl --user -u caminus-agent-NAME -n 50'`.
