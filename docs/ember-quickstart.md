# Ember: quickstart

From an installed Caminus to your first conversation with a resident agent: an AI that lives in its own Unix account
on your machine and works toward goals you give it. What it is and how it is kept apart from you: [the guide](ember.md).
Ember is an experiment: read [its known limits](ember.md#known-limits) first. The recommended setup for your models is
Qwen Code ([the quickstart](quickstart.md)).

You need Debian or Ubuntu with systemd (other systems: [known limits](ember.md#known-limits)), `sudo` once for the
setup, and a brain: a model the agent thinks with, either a local `caminus serve` or a paid API through a spend-capped
key relay.

## 1. Install

Install Caminus ([the quickstart](quickstart.md#1-install-caminus)), then `caminus ember --help`. Ember is in the package
from 0.3.0 on (on PyPI from 0.3.1) and needs no PyTorch; a local model does.

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

Setup also fills the agent's own Python with a standard package set ([the list](ember.md#where-things-live)). If the
agent will commit to a repository or run its CPU tests, give it a git identity and CPU torch (both optional, both safe
to rerun):

```sh
caminus ember setup --git-name LOGIN --git-email LOGIN@users.noreply.github.com   # its GitHub account, for commits a CLA check can link
caminus ember setup --test-venv                                                    # CPU torch and numpy in ~/.venvs/caminus-cpu (about 200 MB more)
```

## 3. Give it a brain

**A local model** (no cost, your GPUs). Ember's default is the model to start with on your cards, the last column of
[the setup table](../README.md#the-model-for-your-setup) (`caminus ember setup` names it at the end).

Qwen3-30B-A3B (`qwen3-30b-a3b`) runs on an XTX or on two cards too; it is not the default because it finishes fewer agent tasks
for now. Start the model with `caminus panel` or `caminus pull MODEL && caminus start MODEL` (see
[the quickstart](quickstart.md)), then point the agent at it:

```sh
caminus ember agent install
caminus ember agent init resident --model MODEL_ID --endpoint http://127.0.0.1:8092/v1
```

`MODEL_ID` is what the server reports (`curl -s http://127.0.0.1:8092/v1/models`).

**A paid API** (DeepInfra here; Anthropic, Mistral, Meta and an OpenCode Go plan, with their key files and addresses, are in [the guide](ember.md#paid-brains)). Put the key in a file only
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

Keys and commands (steer, answer, stop, copy, leave): [the reference](resident-agent.md#the-panel-one-chat-with-overlays).

With nothing to do, the agent has free time: it reads its journal, follows its own interests and keeps a studio
(`/studio`); it is off until you turn it on ([free time](resident-agent.md#free-time)). On a
narrow terminal its thoughts run in the box under the history (the latest stay up between replies): its reasoning when
the model shows it, else its words and the tools it calls. On a wide terminal, until the reply's words start, the
streaming block shows the reasoning's last four lines dimmed (or "preparing" and the tool it is writing); the words
replace them, and the finished reply keeps the reasoning out of the history, as the compact view always does (Ctrl+O
shows it).

## Next

- [The guide](ember.md): what the anti-harness is, the security model, where everything lives, troubleshooting.
- [Every command and view](resident-agent.md): the panel's keys and commands, the loop, tools, memory, brains.
