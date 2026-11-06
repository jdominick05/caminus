# Caminus

**Caminus** (*KAM-ih-nus*, Latin for forge or furnace) is an inference engine written for AMD RDNA3 GPUs from the kernel up:
one persistent GPU launch per token (the K3 megakernel), two cards working as one over PCIe peer-to-peer. It serves one
model, the **Qwen3.8-Flash-Next GSQ-RCO Coder** (a mixture of experts, IQ1_M), with the **DFlash2** drafter, and speaks the
**OpenAI** (`/v1/chat/completions`, `/v1/completions`) and **Anthropic** (`/v1/messages`) APIs on `127.0.0.1:8092` for coding
harnesses. Its core kernels match llama.cpp bit for bit; end to end it stays within llama.cpp's own noise.

## What it needs

| | |
|---|---|
| GPUs | two gfx1100 cards with peer-to-peer (tested: RX 7900 XTX 24 GB + RX 7900 XT 20 GB); the code objects in this package are gfx1100 only |
| OS | Linux with the amdgpu driver and ROCm's kernel side (`/dev/kfd`) |
| Python | 3.12, and **PyTorch's ROCm build** (not on PyPI; see below) |
| Host | ~29 GiB RAM is enough (the 28.8 GB n-gram table is read from disk, never mapped) |
| Disk | 59 GB for the model files; up to 30 GB for the prompt cache's disk tier and 10 GB for the flight recorder |

## Install

```
python3.12 -m venv ~/caminus-venv && . ~/caminus-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2    # the ROCm build (tested: 2.14.1+rocm7.2)
pip install caminus
caminus doctor            # PASS / WARN / FAIL for this machine, the install and the model files (CPU only)
caminus download          # the model files from Hugging Face (58.4 GB + 0.65 GB; resumable, size and sha256 checked)
caminus serve             # runs under the guard; ready in ~50 s on 127.0.0.1:8092 (its log: ~/.local/state/caminus/serve.log)
```

Stop the server with `kill -TERM $(pgrep -x caminus)` (it parks the live conversation to disk first); Ctrl-C does not stop
it. To serve without the DFlash drafter (plain decoding), run `caminus serve --no-drafter`.

The full guide (requirements, the guard, harness configs for Qwen Code / Hermes Agent / Claude Code, llm-top,
troubleshooting, building from source): [docs/INSTALL.md](https://github.com/jdominick05/caminus/blob/main/docs/INSTALL.md).
Measurements and their sources: [docs/BENCHMARKS.md](https://github.com/jdominick05/caminus/blob/main/docs/BENCHMARKS.md).
Source: https://github.com/jdominick05/caminus.

`caminus --help` lists every command (`serve`, `download`, `doctor`, `purge`, `replay`, `flight`, `kv-cache`, `guard`).
Paths come from the environment (`CAMINUS_MODEL_DIR`, `CAMINUS_CODER`, `CAMINUS_DFLASH`, ...), then
`~/.config/caminus/config.toml`, then the XDG defaults; `caminus doctor` prints the ones in effect.

## The guard

Every GPU run goes through the guard shipped in the package (`caminus serve` starts itself under it): it kills the run when
a card passes 0.97 of its VRAM or host memory runs low, and on any amdgpu page fault, reset or timeout in the kernel log (it
needs to read the kernel log: the `systemd-journal` or `adm` group). After a fault it records a STOP file and refuses every
later run until someone has looked. One of the cards usually drives the monitors; the guard and the prompt cache keep room
for the desktop on it.

## Model licences

Nothing in this package is a model weight. The files `caminus download` fetches stay under their own licences: the base
model's is `qwen-community-1.0` (Qwen/Qwen3.8-Flash-Next), the DFlash2 drafter's Apache-2.0 (incoai). See THIRD_PARTY.md.

Caminus is Copyright (C) 2026 jdominick05, licensed under the GNU AGPL v3.0 or later; third-party notices (llama.cpp / ggml,
gguf-py, Strata, flash-linear-attention, transformers) are in THIRD_PARTY.md.
