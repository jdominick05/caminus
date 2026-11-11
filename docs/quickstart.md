# Quickstart

By the end of this page a model will be running on your Radeon card and Qwen Code will be using it. It takes about ten
minutes, plus the model download (12 GB for the first model below).

## What you need

- Linux with the standard `amdgpu` driver.
- At least one AMD Radeon RX 7900 XTX or RX 7900 XT. Other cards: see [hardware](hardware.md).
- Python 3.12.
- Your user in the `render` and `video` groups: `sudo usermod -aG render,video $USER`, then log out and back in.
- For Qwen Code: Node.js 22 or newer.

## 1. Install Caminus

```bash
python3.12 -m venv ~/caminus-venv
source ~/caminus-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
pip install caminus
```

PyTorch goes first because its AMD build isn't on PyPI.

**Nightly builds.** To run what is on the development branch before it is released, install with
`pip install "caminus>=0.2.2.dev0"` instead: that takes the newest build, nightly or release. A nightly is built
automatically from the `testing` branch whenever it changed; it passed the automatic tests, not a release's GPU tests.
Don't use `--pre` for this: it also installs test versions of the packages Caminus depends on.

## 2. Check your machine

```bash
caminus doctor
```

Each line says PASS, WARN or FAIL. Fix every FAIL before you go on. [Troubleshooting](troubleshooting.md) explains each one.

## 3. Start a model in the panel

```bash
caminus panel
```

1. In **MODELS**, move to `qwen3.6` with the arrow keys. It fits any supported card.
2. Press `p` to download it.
3. Press `enter` to start it. The first start takes about a minute.
4. Press `tab` until you reach **QWEN CODE**. Press `i` to install Qwen Code (only once), then `u` to set it up.
5. Press `tab` to reach **SERVERS**, then `enter`. Qwen Code opens, talking to your model.

Press `q` to leave the panel. The model keeps running until you stop it: `x` in **SERVERS**, or `caminus stop qwen3.6`.

## Next steps

- [The panel](panel.md): every key, the advanced layout, the live monitor.
- [Models](models.md): which model to pick for your cards.
- [Harnesses](harnesses.md): Hermes Agent, Claude Code, or any OpenAI- or Anthropic-compatible tool.
- [Without the panel](cli.md): the same steps as commands.
