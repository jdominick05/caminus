# Changelog

Every release of Caminus, newest first. Speeds are measured on an RX 7900 XTX (two-card: XTX + XT) unless a line says
otherwise; where each number comes from is in [docs/dev/benchmarks.md](docs/dev/benchmarks.md).

## Unreleased

### New

- **Qwen3-30B-A3B on Caminus's own code.** `caminus pull qwen3-30b-a3b` serves the official Q4_K_M file on one 24 GB
  card: 231 tok/s at short context, 187 with 4K tokens of context, 104 at its full 40,960 (llama.cpp on the same
  card: 148 at short context, 134 at 4K). It reads a 4K prompt at 6,493 tok/s (llama.cpp: 3,853). Every step from the
  GGUF file to the output token is code written for Caminus, and its output is 5 to 6 times closer to an exact
  reference of the model than llama.cpp's. See [docs/dev/qwen3-30b-a3b.md](docs/dev/qwen3-30b-a3b.md).

### Fixed

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
