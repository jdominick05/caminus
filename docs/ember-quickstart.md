# Ember: quickstart

From an installed Caminus to your first conversation with a resident agent: an AI that lives in its own Unix account
on your machine and works toward goals you give it. What it is and how it is kept apart from you: [the guide](ember.md).

You need Ubuntu (or another systemd Linux), `sudo` once for the setup, and a brain: a model the agent thinks with,
either a local `caminus serve` or a paid API through a spend-capped key relay.

## 1. Install

```sh
python3.12 -m venv ~/caminus-venv && source ~/caminus-venv/bin/activate
pip install caminus
caminus ember --help
```

Ember is in the package from 0.3.0 on (on PyPI from 0.3.1). Ember itself needs no PyTorch; a local model does: [the quickstart](quickstart.md) installs it.

The panel (`caminus ember ui`) ships inside the package. From a source checkout, build it first:
`cd panel && go build -o caminus-panel .`

## 2. Set up the agent account

```sh
caminus ember setup
```

The first run makes your SSH key for the agent account (`~/.ssh/caminus_env_key`) and writes one root script. Read
it, then run it as it says:

```sh
sudo sh ~/.local/share/caminus/env-setup.sh
```

It creates the `caminus-agent` account (key login only, no password), the `caminus-env` group you and the agent
share, the bridge folders in `/srv/caminus-env-bridge` (requests, answers, notes, GPU leases), SSH for your key from
this machine only, and lingering so the agent's services outlive a logout. It changes only what is missing, so it is
safe to run again, and it ends by checking every item (PASS / FAIL). `DRY_RUN=1 sh …` lists what it would change.

You joined the `caminus-env` group: log out and back in (or run `newgrp caminus-env` in this terminal). Then:

```sh
caminus ember setup          # installs the agent's side; every line should say PASS
```

If the agent will commit to a repository or run its CPU tests, give it a git identity and a test interpreter (both
optional, both safe to rerun):

```sh
caminus ember setup --git-name LOGIN --git-email LOGIN@users.noreply.github.com   # its GitHub account, for commits a CLA check can link
caminus ember setup --test-venv                                                    # ~/.venvs/caminus-cpu: CPU torch and numpy (about 200 MB)
```

## 3. Give it a brain

**A local model** (no cost, your GPUs). Which one, by your cards (`caminus ember setup` names it at the end):

| Your cards | Ember's default |
|---|---|
| one RX 7900 XTX | Swift 1.5 (`swift-1.5`) |
| one RX 7900 XT | ThinkingCap (`thinkingcap`) |
| two cards, an XTX first | the Coder (`coder`, both cards) |
| two XTs | ThinkingCap (the Coder needs a 24 GB first card) |

Qwen3-30B-A3B (`qwen3-30b-a3b`) runs on an XTX or on two cards too; it is not the default because it finishes fewer agent tasks
for now. Start the model with `caminus panel` or `caminus pull MODEL && caminus start MODEL` (see
[the quickstart](quickstart.md)), then point the agent at it:

```sh
caminus ember agent install
caminus ember agent init resident --model MODEL_ID --endpoint http://127.0.0.1:8092/v1
```

`MODEL_ID` is what the server reports (`curl -s http://127.0.0.1:8092/v1/models`).

**A paid API** (DeepInfra here; Anthropic and Mistral, with their key files and addresses, are in [the guide](ember.md#paid-brains)). Put the key in a file only
you can read, then run a relay: a small service on your account that holds the key, so the agent never sees it, and
stops at a daily cap.

```sh
mkdir -p ~/.config/caminus
[ -e ~/.config/caminus/deepinfra-key ] || install -m 600 /dev/null ~/.config/caminus/deepinfra-key
$EDITOR ~/.config/caminus/deepinfra-key
caminus ember relay install --provider deepinfra --cap-usd 3            # serves http://127.0.0.1:8766/v1/openai
caminus ember agent install
caminus ember agent init resident --model deepseek-ai/DeepSeek-V4.1-Flash-1017 --endpoint http://127.0.0.1:8766/v1/openai
caminus ember agent provider resident add deepinfra --kind deepinfra --endpoint http://127.0.0.1:8766/v1/openai --relay
caminus ember agent models resident --refresh                          # windows, output caps, prices
```

## 4. Start it and talk

```sh
caminus ember agent start resident
caminus ember ui
```

Type a goal and press Enter: "Find out which of my Python projects have failing tests and write a short report."
The agent first works out what done looks like (it may look around first), then works step by step in its own
account. You see its replies and one dim line per batch of tool work. On a wide terminal (160 columns or more) the
reply it is writing streams into the history as its newest block, and the right column holds its task list at the top
and what it is running now. A narrower terminal shows the reply being written in a box under the history, and the
current task and what runs now in two lines above where you type.

- **Steer**: type while it works. A message reaches it at its next step; `/steer TEXT` redirects it.
- **Answer**: questions and root requests ask where you type: pick with the arrows and Enter (a root command is yes or
  no; Yes runs it under your sudo, never the agent's), or tell it something else in your own words. Esc leaves it
  waiting; Tab brings it back. Nothing waits on you: an
  unanswered question times out to the agent's recommended option, a permission question to no.
- **Stop**: `/pause`, `/cancel`, or `/halt` (every resident pauses; `/halt clear` lifts it).
- **Copy**: drag over the history; the release copies it.
- **Leave**: `/quit` closes the panel; the agent keeps working. `caminus ember ui` brings you back to the same
  conversation.

With nothing to do, the agent has free time: it reads its journal, follows its own interests and keeps a studio
(`/studio`). `caminus ember agent heartbeat resident now` starts a stretch now (after any running work);
`caminus ember agent heartbeat resident off` turns free time off. On a paid brain it spends its own allowance (0.50 USD a day by default). On a
narrow terminal its thoughts run in the box under the history (the latest stay up between replies): its reasoning when
the model shows it, else its words and the tools it calls. A wide terminal leaves the reasoning out of the history, as
the compact view always does; Ctrl+O shows it.

## Next

- [The guide](ember.md): what the anti-harness is, the security model, where everything lives, troubleshooting.
- [Every command and view](resident-agent.md): the panel's keys and commands, the loop, tools, memory, brains.
