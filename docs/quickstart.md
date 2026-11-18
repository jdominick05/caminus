# Quickstart

By the end of this page a model will be running on your Radeon card and Qwen Code will be using it. It takes about ten
minutes, plus the model download (14 to 59 GB, by the model: [models](models.md)). This is the one page on setting
Caminus up; the other docs link here.

## What you need

- Linux on x86-64 with the standard `amdgpu` driver.
- At least one AMD Radeon RX 7900 XTX or RX 7900 XT. Other cards: see [hardware](hardware.md).
- Python 3.12.
- Your user in the `render` and `video` groups: `sudo usermod -aG render,video $USER`, then log out and back in.
- For Qwen Code: Node.js 22 or newer.
- To build from source (recommended): Docker your user can run.

## 1. Install Caminus

**Build it from source (recommended).** The build runs the release recipe inside the public build image: on the CPU, no
GPU needed, nothing installed on your system besides Docker. Built from a release tag, the wheel is byte for byte the
published one. Build the tag of the latest release:

```bash
git clone https://github.com/jdominick05/caminus.git && cd caminus
git checkout "$(git tag -l 'v*' --sort=-v:refname | grep -v dev | head -n1)"   # the latest release's tag (not a nightly's)
tools/release/in_image.sh tools/release/release_build.sh    # GPU code + wheel into dist/ (--untagged at any other commit)
python3.12 -m venv ~/caminus-venv && source ~/caminus-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2   # PyTorch for AMD cards (not on PyPI)
pip install dist/caminus-*.whl
```

The build needs Docker that your user can run (`docker run hello-world` works). It refuses a checkout with an old
`dist/`: build from a fresh clone. 0.2.1 and older predate the build image: build those with
[their own building notes](https://github.com/jdominick05/caminus/blob/v0.2.1/docs/dev/building.md).

**Or install the prebuilt package.** It holds the same files: every release is built with the same recipe and signed,
and its provenance can be checked against the source ([verifying a download](verifying.md); self-assessed against SLSA
Build L3).

```bash
python3.12 -m venv ~/caminus-venv && source ~/caminus-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2   # PyTorch for AMD cards (not on PyPI)
pip install caminus
```

**Nightly builds.** When the `testing` branch changes, it is built overnight and published as a
[pre-release](https://github.com/jdominick05/caminus/releases) (the newest one is at the top). It has what the next
release will have, tested less. Nightlies are on GitHub only for now, not on PyPI. To try one, copy the link of the `.whl`
file attached to the newest pre-release and install it:
`pip install https://github.com/jdominick05/caminus/releases/download/vX.Y.Z.devYYYYMMDD/caminus-X.Y.Z.devYYYYMMDD-py3-none-manylinux_2_17_x86_64.whl`.
Please report what breaks.

## 2. Run the wizard

```bash
caminus wizard --dry-run   # the plan: runs only the read-only commands (doctor, pull --dry-run, fit), writes only its log
caminus wizard
```

It takes six steps and prints each `caminus` command before it runs it, so you learn the commands as you go:

1. `caminus doctor` checks the machine, then `caminus doctor --gpu` runs a tiny test on each card (skipped while another
   program uses a card). Fix every FAIL line ([reading `caminus doctor`](troubleshooting.md#reading-caminus-doctor)),
   then run the wizard again.
2. Pick a model. The list says which models run on your cards, and the one to start with on your setup
   ([the table](../README.md#the-model-for-your-setup)) is preselected: press Enter to take it. While a server runs, its
   model is preselected; choosing another asks, before anything downloads, whether to keep the server or switch (the
   question names each server a switch stops). Read each model's licence before any commercial use
   ([licences](models.md#licences)).
3. `caminus pull MODEL` downloads it, after the wizard shows its size, the folder and the free space there.
4. `caminus fit` says how much context fits on your cards, then `caminus start MODEL` starts the model; the first start
   takes about a minute.
5. `caminus qwen-code setup --install` installs Qwen Code (once) and points it at your server. Your own Qwen Code
   settings in `~/.qwen` stay untouched. Without Node.js 22 the wizard says what to install and skips this step.
6. Open Qwen Code now? Enter opens it, talking to your model. Later, `caminus qwen` does the same.

Run it again any time: it skips what is done. It never runs `sudo`: a step that needs root prints the command for you
to run. `--yes` takes every default (and opens nothing at the end). Its log is `~/.local/state/caminus/wizard.log`. On a
gfx1100 card Caminus hasn't been tested on, it asks for a [hardware report](hardware.md#supported-cards).

The model keeps running until you stop it: `caminus stop MODEL`, or `s` in the panel.

## Or: the same steps in the panel

```bash
caminus doctor
caminus panel
```

1. In the model list, move to the model for your cards ([the table](../README.md#the-model-for-your-setup)) with the arrow
   keys.
2. Press `p` to download it (`y` confirms), then `enter` to start it.
3. Once the running card at the top shows it, press `i` (then `y`) to install Qwen Code (only once), then `u` to set it up.
4. Press `c`. Qwen Code opens, talking to your model.

Press `q` to leave the panel; the model keeps running.

## Next steps

- [The panel](panel.md): every key, the options editor, the live monitor.
- [Models](models.md): which model to pick for your cards.
- [Harnesses](harnesses.md): Hermes Agent, Claude Code, or any OpenAI- or Anthropic-compatible tool.
- [The advisor](advisor.md) (optional): a hosted model steers Qwen Code when it looks lost.
- [Without the panel](cli.md): every command.
- [Ember](ember-quickstart.md) (experimental): an AI agent with its own account on your machine; read
  [its known limits](ember.md#known-limits) first.
- [Troubleshooting](troubleshooting.md): what each `caminus doctor` line and each error means.
