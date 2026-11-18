# Quickstart

By the end of this page a model will be running on your Radeon card and Qwen Code will be using it. It takes about ten
minutes, plus the model download (14 to 59 GB, by the model: [models](models.md)). Qwen Code is the recommended setup;
Ember, the agent environment, is an experiment ([its known limits](ember.md#known-limits)).

## What you need

- Linux on x86-64 with the standard `amdgpu` driver.
- At least one AMD Radeon RX 7900 XTX or RX 7900 XT. Other cards: see [hardware](hardware.md).
- Python 3.12.
- Your user in the `render` and `video` groups: `sudo usermod -aG render,video $USER`, then log out and back in.
- For Qwen Code: Node.js 22 or newer.
- To build from source (recommended): Docker your user can run.

## 1. Install Caminus

Build it from source (recommended) or install the prebuilt wheel: both are in the
[README](../README.md#get-started), with the nightly builds. PyTorch goes first because its AMD build isn't on PyPI:

```bash
python3.12 -m venv ~/caminus-venv
source ~/caminus-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
```

## 2. Check your machine

```bash
caminus doctor
```

Each line says PASS, INFO, WARN or FAIL, with the area it checked. Fix every FAIL before you go on;
[reading `caminus doctor`](troubleshooting.md#reading-caminus-doctor) explains each area and what to do.

## 3. Start a model in the panel

```bash
caminus panel
```

1. In the model list, move to the model for your cards with the arrow keys: `thinkingcap` on one XT (it fits any supported
   card), `swift-1.5` on one XTX, `coder` on an XTX + XT ([the setup table](../README.md#get-started)).
   On a 24 GB card, `qwen3-30b-a3b` writes about five times as fast as ThinkingCap, but finishes fewer agent tasks for now.
   Read each model's licence before any commercial use ([licences](models.md#licences)). The steps below use `thinkingcap`.
2. Press `p` to download it (`y` confirms).
3. Press `enter` to start it. The first start takes about a minute.
4. Once the running card at the top shows it, press `i` (then `y`) to install Qwen Code (only once), then `u` to set it up.
5. Press `c`. Qwen Code opens, talking to your model.

Press `q` to leave the panel. The model keeps running until you stop it: `s` in the panel, or `caminus stop thinkingcap`.

## Next steps

- [The panel](panel.md): every key, the options editor, the live monitor.
- [Models](models.md): which model to pick for your cards.
- [Harnesses](harnesses.md): Hermes Agent, Claude Code, or any OpenAI- or Anthropic-compatible tool.
- [Without the panel](cli.md): the same steps as commands.
- [Ember](ember-quickstart.md) (experimental): an AI agent with its own account on your machine, working on the model
  you started. Read [its known limits](ember.md#known-limits) first.
- [Troubleshooting](troubleshooting.md): what each `caminus doctor` line and each error means.
