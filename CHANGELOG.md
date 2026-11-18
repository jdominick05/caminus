# Changelog

Every release of Caminus, newest first. Speeds are measured on an RX 7900 XTX (two-card: XTX + XT) unless a line says
otherwise; where each number comes from is in [docs/dev/benchmarks.md](docs/dev/benchmarks.md).

## 0.3.2 (2026-11-18)

Faster long conversations (measured on Qwen3-30B-A3B), Qwen Code as the recommended way in, and Ember (now listed under
Experiments) with a much bigger toolset. After upgrading, run `caminus ember agent install` if you use Ember, so its
agent picks up the new runtime.

### Changed

- **Qwen Code is the recommended setup.** The README's Get started now ends with Qwen Code; Ember moved to a new
  Experiments section, and [docs/ember.md](docs/ember.md) lists its known limits (it has been tested on Ubuntu with
  systemd only) (#301).
- **The wheel says what it is.** It carries an x86-64 Linux binary (the panel) and gfx1100 GPU code, so it is now a
  `manylinux_2_17_x86_64` wheel instead of `py3-none-any`: pip no longer installs it on machines it can't run on, and
  building a wheel from the source package on another machine stops with a one-line reason (#303).

### Faster

- **Long conversations answer sooner.** Each request used to copy the whole history for sampling, tokenize the whole
  conversation twice and convert the token log to arrays several times. Now the sampler reads only its penalty window,
  the prompt is encoded once with repeated fragments remembered, and prefix matches compare arrays converted once
  (#290, #291, #293). Measured through `caminus serve` on Qwen3-30B-A3B (Q4_K_M, F16 KV), same replies token for token:
  - a warm turn's time to first token (a turn that restores a cached checkpoint): 454 -> 274 ms at about 38K tokens on one card, 1,153 -> 594 ms at about 126K
    on two cards;
  - sampled decoding with a presence penalty: 92 -> 108 tok/s at about 39K (one card), 52 -> 71 tok/s at about 127K
    (two cards).

### New in Ember (experimental)

- **Use your OpenCode Go plan as the brain** (`--kind opencode-go`): the plan's chat models, each conversation with
  its own session id as the plan asks, spend counted from what each reply says it charged, and free time kept off the
  plan unless you allow it (#294). The advisor can run on one of the plan's models too (`caminus ember agent advisor
  NAME --kind opencode-go`), counted against its budget at no less than the model's list price (#300).
- **New tools:** a sandbox for untrusted code (a bubblewrap box, #258), a browser (Chrome in a box, #261), web research
  with search fallbacks, retries, a cache, PDFs and arXiv papers (#251), a search tool and line ranges for `read`
  (#254), images in `read` for brains that see and borrowed eyes for brains that don't (#256, #260), command discovery
  (#264), and memory housekeeping (#252).
- **Meta's Model API (the Muse models)** as a brain provider (#283).
- **The agent knows the time** from a short stamp on the newest message, without breaking the prompt cache (#296).
- **Rendered Markdown in open cards:** command output that is clearly Markdown renders when a card is open; Ctrl+Y
  shows the source (#295). Advice renders as Markdown too, and a popup over the typing box shows while the advisor is
  asked, with who asked and the seconds it has taken (#302).
- **Closed sessions come back:** closing a session in the switcher keeps it, and `r` (or `caminus ember agent
  restore`) restores it. A new session's `/model` lists the local server it runs on (#299).
- **A question waits while you are away,** then the advisor answers it (#255).
- **The panel:** tasks at the top right (#267), select to copy anywhere on the screen (#268), the engine's own read and
  write speeds (#266), the whole turn's token count (#288), reasoning lines while a reply streams on a wide terminal
  (#276).

### Fixed

- A click opens or folds only the card under it, not every card (#275).
- The `/theme` picker moves at once (#278).
- A `!` or `!!` command's output is a card in the chat, not text in the typing box (#277).
- A local brain's context window comes from its server, under your cap, never "31K/0" (#280, #287).
- Markdown tables in replies use the body's text colour (#289).
- The done-check gate runs with the terminal's environment (#270); `sed -n` and other print-only reads pass the
  read-only check (#273); `todo` takes the shapes models are trained on (#279).
- Stopping works: SIGTERM stops a daemon that is waiting, a stopped worker reads stopped, ending a goal kills a box whose
  worker is gone, and the lab's close stops its workers too (#271, #281, #282).
- The setup fills the agent's Python environment with a standard data and document package set (#265).

### For contributors

- The cancel-latency test holds the 4th best of five cancels to 30 ms instead of the worst (#274).
- Qwen Code's context note stays tokens only; the time stamp is Ember's alone (#298).
- The prefix cache's live-log rule lives in one place (#297).
- `verify_release.py` checks the wheel's tag and that the panel inside is a static x86-64 binary; the release build pins
  the panel to linux/amd64 (#303).

### Known issues

- Ember's setup assumes Debian or Ubuntu with systemd; see the known limits in docs/ember.md.
- In Ember, a `!` shell command starts in the agent account's home, not the session's workspace.
- `caminus serve` answers a request for a model id it doesn't serve with its own model instead of refusing it.
- When the agent's own advisor call fails at once (no calls left, a spent budget), no popup opens; the error card in the
  chat says why. `/advisor` shows the popup as described.

Benchmarks: a patch release, so no full rerun. No kernel or engine code changed since 0.3.0; the served speedups above
were measured A/B on 489f37a vs 3a3612e (see #290).

## 0.3.1 (2026-11-17)

The fixes to 0.3.0's known issues, and the first release of 0.3 on PyPI (0.3.0's upload step failed; it is on GitHub
only). After upgrading, run `caminus ember agent install` so a running agent picks up the fixes.

### Fixed

- **A root command you approve always comes back to Ember.** An approved `apt install` could hang at its end
  ("Processing triggers ...") and never return to the chat: the panel waited for every process holding the command's
  output. It now waits for the command itself, with a bound, and returns its exit code and output to the chat.
- **Esc stops the reply on the server too.** After an interrupt, the server could keep generating and the status line
  kept counting. The call is closed from its first moment, so the server stops the job, and the status line says
  interrupted.
- **The panel swaps models.** Starting a model while another runs asks once ("stop X and start Y?") and does both,
  and `s` stops the running model, saying what it does for a server started outside the panel.
- **The advisor says what to run when it cannot be called,** for example when the key relay doesn't allow its model,
  instead of a raw error that vanished after a few seconds.
- **`/effort` works on a local model session made before this release.** It asks the session's own server which effort
  levels it serves (the Coder's `-low`, `-xhigh` and `-nothink` ids) instead of saying "no effort control".

### For contributors

- **The release workflow uploads to PyPI.** twine attaches the PEP 740 attestations only when they are named beside the
  files; 0.3.0's upload stopped there.
- **Growth rules for a tree written by agents** (ENTROPY.md, `tools/entropy.py`, and `tools/pr_ready.py`'s size check):
  a pull request that adds more lines than it removes says what it reused and removed.

### Known issues

- Clicking a folded card in Ember opens every card (as Ctrl+O does), not only that one.
- The `/theme` picker is still slow to move.
- The output of a `!` or `!!` command lands in the typing box (to send with your words), not in the chat.

Benchmarks: no engine or kernel code changed since 0.3.0, so its numbers stand (docs/dev/benchmarks.md, measured on
0.3.0).

## 0.3.0 (2026-11-17)

### New

- **Ember, an agent that works on your machine in its own account.** `caminus ember` sets up an agent as a second user
  on this machine: it works toward a goal you give it, with your local model or a paid one, and asks before anything
  needs root or would post on GitHub under your name. Talk to it in `caminus ember ui`: steer it mid-task, interrupt it
  with Esc, pull back a message it hasn't read, rewind its file changes, queue goals, and answer its questions and
  approvals inline. One effort scale (none to max) works on every model, with each model's recommended settings by
  default. See [Ember](docs/ember.md) and [its quickstart](docs/ember-quickstart.md).
- **A new control panel.** `caminus panel` opens on one status-first screen: the running server, the models with what
  each needs, and the agents. It also opens inside Ember with `/panel`. A server started outside the panel is watched,
  never stopped or restarted by it. See [the panel](docs/panel.md).
- **Qwen3-30B-A3B reads longer texts, opt-in.** `caminus start qwen3-30b-a3b --long-context` (or `caminus serve
  --long-context`) uses the model card's recipe for long texts, YaRN x4, for up to 131,072 tokens: about 80K on one
  RX 7900 XTX by the memory projection (the server stores the cache as 8-bit rows there and says so) and the full 131,072 on two cards. The default
  stays 40,960 tokens without scaling: YaRN changes every position, so short texts can read slightly differently. Other
  models refuse the flag. See [models](docs/models.md).

- **Signed build provenance (SLSA Build Level 3).** Every release and nightly is now built again on GitHub's runners
  in the same build image the maintainer uses, published only when both builds are byte for byte equal, and signed:
  `gh attestation verify caminus-X.Y.Z-py3-none-any.whl --repo jdominick05/caminus --signer-workflow
  jdominick05/caminus/.github/workflows/build.yml --source-ref refs/tags/vX.Y.Z`. Files on PyPI carry PEP 740 attestations. See
  [verify your download](docs/verifying.md).
- **Qwen Code's `/btw` no longer disturbs the conversation.** A side question now runs in the server's side slot, so the
  conversation's next turn continues where it was instead of being read again. In a long conversation on Swift 1.5 or
  ThinkingCap, it's answered from the whole stored conversation, so only the question is read. See
  [harnesses](docs/harnesses.md).
- **Two dense 27B models on Caminus's own code.** `caminus pull swift-1.5` (Swift 1.5, Q4_K_M, a 24 GB card) and
  `caminus pull thinkingcap` (ThinkingCap, 4.2 bits per weight, either card) are served from the GGUF file to the output
  token with code written for Caminus. ThinkingCap decodes at 47 tok/s on an XTX and 39 on an XT (llama.cpp on the same
  cards: 40 and 34) and reads a 2K prompt at 942 tok/s; its output is about 1,000 times closer to an exact reference of
  the model than llama.cpp's. Both take opt-in drafters (`--draft mtp` and `--draft dflash`, below). See
  [models](docs/models.md).
- **The two dense 27B models on two cards.** `caminus start swift-1.5 --cards 2` (or `thinkingcap`, or `caminus serve
  --cards 2`) splits every layer across both cards: decode at 64 and 69 tok/s on an XTX + XT, and the full 262,144-token
  context fits. Still Caminus's own code end to end, with the same accuracy. Reading prompts is no faster than on one XTX
  yet. See [benchmarks](docs/dev/benchmarks.md).
- **Qwen3-30B-A3B on two cards, offered.** `caminus start qwen3-30b-a3b --cards 2` (or `caminus serve --cards 2`, or the
  panel's two-card mode) splits every layer across an XTX + XT: each card holds half the attention heads and part of every
  expert (two thirds on the bigger card, card 0). Still Caminus's own code end to end. One card stays the default: served, two cards decode more slowly on short
  turns (213 against 245 tok/s) and faster from about 32K tokens on (121 against 109 tok/s at 40,768), and read every
  prompt from 4K tokens on faster (1.10x at 4K, 1.29x at 40,768). The context stays 40,960
  tokens, the model's own limit. See [benchmarks](docs/dev/benchmarks.md).
- **An 8-bit KV cache for Qwen3-30B-A3B.** `caminus start qwen3-30b-a3b --kv q8` (or `caminus serve --kv q8`), on one
  card or two, stores the attention cache in 8-bit rows (the keys rotated first) at about half the memory per token of
  context: 0.050 MiB instead of 0.094. F16 stays the default. The context stays the model's 40,960 tokens, but the
  full context fits with more of the card in use: on an XTX with the desktop holding 1 GiB, 40,960 tokens with the
  8-bit cache, 32,768 with F16 (projected). Decode is about as fast (engine-level: on one XTX 0.97x on short turns,
  1.06x with a full context; on two cards within 2 %). By mean KL divergence it changes the output less than
  llama.cpp's 8-bit cache changes llama.cpp's; top-1 agreement with the F16 cache is within one row of llama.cpp's
  (one card at 16,384 tokens: 60 of 64 rows against its 61). A conversation
  saved in one format is never restored by a server running the other. `/props` reports the cache
  (`caminus.kv_cache`) and `caminus ps` shows it. `caminus fit --json` reports the context that fits with each cache
  (`max_ctx.q8` next to `max_ctx.f16`, `null` for models without the 8-bit cache) and the panel's model picker shows
  both; for the Coder, which keeps its own 8-bit cache (`kv_fixed`), it shows that cache's one figure. Every other
  model refuses `--kv q8`. See [models](docs/models.md#the-8-bit-kv-cache-qwen3-30b-a3b).
- **Speculative decoding for both dense 27B models, served.** Two drafters on one card, both on Caminus's own
  code; the output is identical to `--draft off` token for token. `--draft dflash` is the default (above). Served decode, `caminus serve`, short prompts (code / prose / shell):
  - `--draft mtp` drafts 3 tokens with the model's own multi-token-prediction layer and verifies 4 tokens a pass; it
    needs no extra file. Swift 1.5 on an XTX: 91 / 96 / 74 tok/s against 43 plain. ThinkingCap on an XT: 68 / 74 / 53
    against 40.
  - `--draft dflash` drafts with the base model's DFlash2 drafter (2 GB, shared by both models; `caminus pull` now
    fetches it with either model). Swift 1.5 on an XTX: 96 / 95 / 71 tok/s (llama.cpp with the same drafter: 75 / 75 /
    67). ThinkingCap on an XT: 76 / 77 / 61 (llama.cpp: 71 / 75 / 48). The drafter's memory leaves Swift about 12-16K
    tokens of context on a 24 GB card (it starts from 8K); `--draft mtp` keeps the full context.
  See [benchmarks](docs/dev/benchmarks.md).
- **The plain Qwen3.8 27B on Caminus's own code.** `caminus pull qwen3.8-27b` serves ISTA-DASLab's GSQ-RCO IQ3_S file
  (Apache-2.0) on one card of either size: 55 tok/s decode on an XTX at short context, about 800 tok/s reading a 2K
  prompt on an XT. `--draft dflash` and `--draft mtp` give the same output as plain decoding, at about 1.2-1.4x and
  1.1-1.3x served on an XTX. Five new matrix formats (IQ2_XS, Q2_K, IQ1_M, IQ2_XXS, Q4_K) have their own decode
  tiles, prefill GEMMs and T-token verify tiles. See [models](docs/models.md).
- **Qwen3-30B-A3B on Caminus's own code.** `caminus pull qwen3-30b-a3b` serves the official Q4_K_M file on one 24 GB
  card: 248 tok/s at short context, 213 with 4K tokens of context, 111 at its full 40,960 (llama.cpp on the same
  card: 148 at short context, 134 at 4K). It reads a 4K prompt at 6,432 tok/s (llama.cpp: 3,853). Every step from the
  GGUF file to the output token is code written for Caminus, and its output is about 6 times closer to an exact
  reference of the model than llama.cpp's. See [docs/dev/qwen3-30b-a3b.md](docs/dev/qwen3-30b-a3b.md).
- **What the experts' weight format buys, measured.** Two studies on Qwen3-30B-A3B: the IQ codebook formats decode in
  the time their smaller files save (no faster than Q4_K_M at any depth), and plain 4-bit codes with one scale per row
  gain 2 to 3 % at about the same accuracy. Study modules only; the served engine is unchanged. See
  [the IQ quant study](docs/dev/iq-quant-study.md) and [the row-scale study](docs/dev/row-scale-study.md).
- **Qwen Code sees how much context it has left.** After each batch of tool calls, one line in the tool result says how
  many tokens are used and how many are left before automatic compaction, so the model can plan the end of a long task.
  The prompt cache keeps working. On by default; `caminus qwen-code setup --context-note off` turns it off. See
  [harnesses](docs/harnesses.md#the-context-note).
- **The advisor, an add-on for Qwen Code.** When the model gets stuck (failing commands in a row, the same call over
  and over, a long run), a second model reads the task and the recent steps and sends a short hint, added to the newest
  message so the prompt cache keeps working. The second model can be local (another model on your other card, or any
  OpenAI-compatible server; `caminus advisor setup --local-only` installs nothing) or hosted (Anthropic or Mistral).
  `/advise` asks on demand. See [docs/advisor.md](docs/advisor.md).

- **The sampler's candidates come from Caminus's own kernel.** Each row's largest logits (the top-k the sampler draws
  from, and the raw logits of the penalty window) were picked with PyTorch's `torch.topk`; a Caminus kernel does it now
  (`kernels/own_topk.hip`), with a fixed order among equal values, so reruns pick the same candidates. PyTorch still
  allocates the GPU memory and loads the HIP runtime.

### Changed

- **Drafting on two cards for Swift 1.5 and ThinkingCap, DFlash2 by default.** `caminus start swift-1.5 --cards 2` (or
  `thinkingcap`) now drafts with `--draft dflash` when the shared drafter file is there and 32,768 tokens of context
  still fit with it; `--draft mtp` works on two cards too, and `--draft off` serves plain. The output is identical to
  plain decoding. On an XTX + XT, served: 1.36-1.44x (Swift 1.5) and 1.13-1.18x (ThinkingCap) on 10K-34K-token prompts,
  1.20-1.45x on 11K-29K-token agent prompts. The panel offers auto / off / MTP / DFlash for both on two cards. See
  [benchmarks](docs/dev/benchmarks.md#speculative-decoding-on-two-cards-served).
- **DFlash2 drafting is on by default for the three dense 27B models on one card.** `swift-1.5`, `thinkingcap` and
  `qwen3.8-27b` serve with `--draft dflash` when the shared 2 GB drafter file is there (`caminus pull` fetches it)
  and 32,768 tokens of context still fit with it, with the same output as plain decoding: about 1.1-1.4x on 10K and
  34K-token prompts and up to 2.2x on short ones. Swift 1.5 on a 24 GB card keeps plain decoding by default (the
  drafter would leave it about 12-16K tokens, too few for a coding harness); `--draft dflash` runs it anyway. Two
  cards: above.
- **Qwen3.6, Qwen3-Coder-Next and AREX-2 are paused for this release.** The panel and `caminus models` no longer list
  them, and `caminus pull`, `caminus start` and `caminus serve` refuse them with the reason. Qwen3.6 and Qwen3-Coder-Next
  aren't measured on the current code; AREX-2 is outside this release's model set. The brain / hands team (AREX-2 +
  Qwen3.6) is paused with them. This release ships the Qwen3.8 Coder, Swift 1.5, ThinkingCap and Qwen3-30B-A3B, and the
  plain Qwen3.8-27B once its drafting is built. `CAMINUS_ALLOW_PAUSED=1` still serves the paused models, unmeasured.

### Fixed

- **Qwen3-30B-A3B runs at its model card's sampling settings.** Its GGUF file carries no sampling settings, so it was
  served at temperature 1.0 with no presence penalty. Each served model now gets the sampling its card recommends for
  agentic work: Qwen3-30B-A3B temperature 0.6, top_p 0.95, top_k 20, min_p 0 and presence penalty 1.5 (0.7 / 0.8 /
  20 / 0 / 1.5 without thinking, as its card gives); the Qwen3.8 models 1.0 / 0.95 / 20 / 0, as before. A request's own
  fields still win, and a greedy request (temperature 0) gets no default penalty. `/props` reports the defaults in
  effect and the card's recommended output length. A GGUF file outside the catalog keeps its own
  `general.sampling.*` settings, now including its repeat penalty. See [models](docs/models.md#sampling-defaults).
- **The two-card check re-reads a PCIe link before refusing a card, and judges a link the driver has lowered at full
  speed.** On the test machine the check once refused the XTX as "on a PCIe 1.0 x1 link" while it sat in its x16 slot.
  The cause was not found and the reading has not come back. When the graphics driver reports that it has lowered an
  idle card's link to save power, the check now judges the link at the driver's full speed, and `caminus doctor` shows
  the idle reading too.
- **Qwen3-30B-A3B reads a prompt with very large activations.** If 32 activations in a row averaged over 2,047, the
  prompt reader's Q4_K product turned that token into NaN and the request failed. Such a token now takes a scaled
  path and is as accurate as any other; every other token gives the same bits as before. It costs 0.9 % of a prompt's
  time.
- **Background requests no longer push your conversation out.** Qwen Code sends two background memory jobs (about 30K
  and 36K tokens) every time you send a message, including a steer mid-task or a redirect after Ctrl+C. They were too big
  for the server's side slot, so they took the main one and your conversation had to be read again (one 494,882-token
  conversation took 587 s). The side slot now holds 98,304 positions and keeps its copy of your rows in system memory, not
  VRAM, copying only the rows a side request overwrites.
- **The long-context mode saves conversations to disk again** (`--max-ctx` past 262,144). A conversation pushed out by
  another one comes back in seconds instead of being read again. Saving writes about 12.4 KB per token, so the disk cap
  is 64 GB in this mode; `--cache-cap-gb` sets it.
- **The guard stops on an unresponsive GPU scheduler** (`MES failed to respond`, `MES ring buffer is full`), seen with
  Linux kernel 7.0.0-42 on an RX 7900 XT. See [docs/troubleshooting.md](docs/troubleshooting.md).
- **The panel's server list keeps a long server name apart from its model.** A name of 10 characters or more (a server
  takes its model's name, e.g. `qwen3-30b-a3b`) was cut and ran into the model column; the column now fits the longest
  name.

### For contributors

- **The README's picture of the control panel shows this release.** `python -I tools/release/panel_shot.py` draws the
  panel's real render of the tree's version, model catalog and Qwen Code pin in a GNOME terminal window; every release
  runs it (docs/dev/releasing.md), and tests/docs fails while the picture shows another version.
  `python -I tools/release/make_banner.py --social` draws the repository's social preview (docs/social-preview.png),
  and the README's banner now takes turns between CAMINUS and QWAMINUS (CAMINUS only for readers who ask for less
  motion).

### Known issues

Found in the release test of Ember; all four are fixed in 0.3.1.

- A command run as root through Ember (an `apt install`, for example) can hang at its end and never return to Ember.
- Interrupting a reply with Esc can leave the reply running: the status line keeps counting and the server keeps
  generating.
- The panel doesn't stop a running model before starting another, so starting a second model fails while the first is
  loaded: stop the first one, then start the other.
- Ember's advisor can fail to start.

## 0.2.1 (2026-11-09)

### New

- **A 1M-token context for the Coder (opt-in).** `caminus serve --max-ctx N` with N past the model's 262,144 turns it on,
  up to 1,048,576 tokens on an XTX + XT. The KV cache moves to pinned system memory (about 13 GB at 1M; the server checks
  free memory first and refuses otherwise) and RoPE is scaled (NTK, x4). With it, values planted across a 1,040,384-token
  prompt were all found (3 of 3, and one 936K tokens back); unscaled, 2 of 3. The scaling costs a little at short context
  (mean KLD 0.0077 against 0.0057), so every context at or below 262,144 is unchanged. See
  [docs/configuration.md](docs/configuration.md).
- **Faster decode deep into a conversation.** The indexer's pool selection now runs in parallel, bit for bit the same as
  before: plain decode at 262K went from 15.58 to 11.39 ms/token, with no cost at short context.
- **The Coder on two RX 7900 XTX.** It keeps its measured split on every pair, so two XTX run it too; two XT are refused
  with the reason (card 0 needs 24 GB). Not yet tried on real two-XTX hardware: a
  [hardware report](https://github.com/jdominick05/caminus/issues/new?template=hardware_report.yml) is welcome.
- **Panel lighting.** `caminus top` can light your RGB devices by agent state through OpenRGB (off unless the config
  enables it; the devices' own colours and modes come back on exit). Monitor preferences now carry into `caminus top`,
  and guard refusals show in the panel's startup errors.
- **A guard against runaway answers.** Eight identical sentences in a row end the chat reply with an explicit error
  instead of a stream that never stops; `--no-loop-guard` turns it off.

### Fixed

- **A sleeping monitor no longer stops Caminus.** In 0.2.0 the guard took the display driver's timeouts (logged when a
  monitor sleeps, wakes or changes mode) for a GPU fault and refused every later start until its stop file was removed
  by hand. Real faults still stop it. If 0.2.0 left you a stop file holding only `REG_WAIT timeout` lines, see
  [docs/troubleshooting.md](docs/troubleshooting.md).
- **Long AREX-2 prompts no longer crash the server.** A ROCm profiler hook that overflowed on long prefills is now off by
  default (unless you set it or attach a profiler).
- Guarded commands keep the traceback of a fatal error.

### Security

- **Jinja2 3.1.6 or newer is required.** The chat template comes from the model file and is rendered in Jinja2's sandbox;
  older Jinja2 versions have sandbox escapes (CVE-2024-56326, CVE-2025-27516). Caminus now refuses an older Jinja2 and
  `caminus doctor` reports it with the install command.

### For contributors

- **Nightly builds.** When `testing` changed, its head is built overnight with the release toolchain and published as a
  GitHub pre-release and as a dev version on PyPI: `pip install "caminus>=0.2.2.dev0"` takes the newest build. See
  [docs/dev/releasing.md](docs/dev/releasing.md#nightly-builds-nightlysh-on-a-timer-then-nightlyyml).
- **Faster builds and CPU tests.** `./build.sh` compiles up to 8 code objects at once (`BUILD_JOBS`; 174 s -> 30 s on
  16 threads, every output byte-identical to the serial build) and `tests/ci_cpu.py` runs up to 8 suites at once
  (`--jobs`; 157 s -> 76 s without the model files); the default takes at most one suite per 2 GiB of free memory.
- Codex reviews every pull request into `testing`; its rules are in AGENTS.md. Dependabot proposes weekly updates for the
  workflows' pinned actions and the panel's Go modules.
- The dense Qwen3.5 line (AREX-2) has a multi-token verify module, the groundwork for a drafter; the server does not use it
  yet.

## 0.2.0 (2026-11-08)

### New

- **Four more models.** On one card: Qwen3.6-35B-A3B in two quantizations (GSQ 2-bit, 175 tok/s decode; HauhauCS
  Q3_K_P, 192 tok/s) and AREX-2 27B (41 tok/s). On two cards: Qwen3-Coder-Next 80B-A3B (136 tok/s decode; prompts are
  read slowly until its batched prefill lands).
- **The Caminus panel** (`caminus panel`): download a model, start it, run Qwen Code on it, watch it live (`caminus top`).
- **Model management from the terminal:** `caminus models`, `pull`, `start`, `ps`, `logs`, `stop`, `restart`,
  `qwen-code setup`, `qwen`.
- **Your cards, found automatically:** no card is hardcoded; Caminus finds RX 7900 XTX / XT cards by PCI address, picks
  where each model runs and sizes its context to the card.
- **The brain / hands team:** AREX-2 plans on one card while Qwen3.6 works on the other
  (`caminus start --team brain-hands`).
- **New docs:** a quickstart, pages for the panel, models, hardware, harnesses, troubleshooting and configuration, a
  contributor guide and issue forms.

### Changed

- The Coder's context is sized from your cards (262,144 on an XTX + XT, less if your desktop uses more VRAM).
- Qwen Code's and Hermes's context window follows what the server actually holds, every time you launch them.
- Pull requests go to the `testing` branch, with a contributor licence agreement on the first one.

## 0.1.0 (2026-11-06)

First public release: a persistent-megakernel inference engine for RDNA3 that runs the Qwen3.8-Flash-Next GSQ-RCO Coder
on two Radeon cards with tensor parallelism and serves OpenAI- and Anthropic-compatible APIs for agent harnesses (Qwen
Code, Hermes Agent, Claude Code).

- Decode about 104 tok/s plain, and 156 / 203 / 123 tok/s (prose / code / shell) with DFlash drafting, with output
  identical to plain decoding.
- Prompts read at about 2,800 to 2,900 tok/s at 4K to 32K, about 2,400 tok/s at 262K; the model's full 262,144-token
  context.
- A prompt cache with VRAM checkpoints and a disk tier: a 200K-token conversation restores in 2 ms from VRAM or about
  1.2 s from disk.
