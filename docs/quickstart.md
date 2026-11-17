# Quickstart

By the end of this page a model will be running on your Radeon card and Qwen Code will be using it. It takes about ten
minutes, plus the model download (14 to 59 GB, by the model: [models](models.md)). For Ember, the agent environment, go
on with [the Ember quickstart](ember-quickstart.md) once a model runs.

## What you need

- Linux with the standard `amdgpu` driver.
- At least one AMD Radeon RX 7900 XTX or RX 7900 XT. Other cards: see [hardware](hardware.md).
- Python 3.12.
- Your user in the `render` and `video` groups: `sudo usermod -aG render,video $USER`, then log out and back in.
- For Qwen Code: Node.js 22 or newer.
- To build from source instead of installing the prebuilt wheel: Docker your user can run.

## 1. Install Caminus

```bash
python3.12 -m venv ~/caminus-venv
source ~/caminus-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
pip install caminus
```

PyTorch goes first because its AMD build isn't on PyPI. Ember and the models on this page arrive on PyPI with 0.3.0;
until then, build from source (below) or install a nightly.

**Building from source instead.** If you'd rather not install a prebuilt wheel, build it yourself (Docker, CPU only;
built from a release tag, the result is byte for byte the published wheel): clone the repository and `cd` into it, check
out `testing` (from 0.3.0 on: the latest release's tag), run
`tools/release/in_image.sh tools/release/release_build.sh --untagged` (no `--untagged` on a release tag), then, in the
clone with the venv above active, `pip install dist/caminus-*.whl` in place of `pip install caminus`. The
[README](../README.md#get-started) has the commands.

**Nightly builds.** To run what is on the development branch before it is released, install with
`pip install "caminus>=0.3.0.dev0"` instead: that takes the newest build, nightly or release. A nightly is built
automatically from the `testing` branch whenever it changed; it passed the automatic tests, not a release's GPU tests.
Don't use `--pre` for this: it also installs test versions of the packages Caminus depends on.

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

1. In **MODELS**, move to the model for your cards with the arrow keys: `thinkingcap` on one XT (it fits any supported
   card), `swift-1.5` on one XTX, `coder` on an XTX + XT ([the setup table](../README.md#get-started)).
   These are Ember's defaults too. On a 24 GB card, `qwen3-30b-a3b` writes about five times as fast as ThinkingCap.
   Read each model's licence before any commercial use: ThinkingCap's is PolyForm Small Business 1.0.0
   ([licences](models.md#licences)); `qwen3-30b-a3b` is Apache-2.0. The steps below use `thinkingcap`.
2. Press `p` to download it.
3. Press `enter` to start it. The first start takes about a minute.
4. Press `tab` until you reach **QWEN CODE**. Press `i` to install Qwen Code (only once), then `u` to set it up.
5. Press `tab` to reach **SERVERS**, then `enter`. Qwen Code opens, talking to your model.

Press `q` to leave the panel. The model keeps running until you stop it: `x` in **SERVERS**, or `caminus stop thinkingcap`.

## Next steps

- [The panel](panel.md): every key, the advanced layout, the live monitor.
- [Models](models.md): which model to pick for your cards.
- [Harnesses](harnesses.md): Hermes Agent, Claude Code, or any OpenAI- or Anthropic-compatible tool.
- [Without the panel](cli.md): the same steps as commands.
- [Ember](ember-quickstart.md): an AI agent with its own account on your machine, working on the model you started.
- [Troubleshooting](troubleshooting.md): what each `caminus doctor` line and each error means.
