# Release plan: the installable `caminus` package (impl:package, 2026-11-06)

The release decisions of 11-06: an installable release on PyPI under the name `caminus` (the maintainer uploads it), a public
repository that starts from one squashed commit of the cleaned tree (the development history stays private). This note records how the package is built and why. The work happens on
the branch `release`; `main` stays what the user runs.

## What ships

| Part | Choice | Why |
|---|---|---|
| Distribution | `caminus` **0.1.0** (the name reservation was 0.0.x), AGPL-3.0-or-later, Python >= 3.12 | 3.12 is the only version tested; `tomllib` alone would allow 3.11 |
| Wheel tag | **pure `py3-none-any`** | No host-native code ships. The engine reaches the HIP runtime through `ctypes` on the `libamdhip64.so` that PyTorch already loaded (`caminus.device.hip`), and libc / libm through `ctypes`; the GPU code objects are gfx1100 ELF images loaded with `hipModuleLoadData`, data rather than host binaries. PyPI rejects `linux_x86_64` wheels, and none is needed. The C++ runtime tests (`runtime/`), `tools/*.cpp` and the reference builds stay in the repository |
| Python package | `python/caminus` (with `caminus.serve`, `caminus.cmd`, the vendored `caminus._vendor.gguf`) | `python/rstack` is a compatibility alias for old scratch scripts and is not shipped |
| GPU code | the **prebuilt gfx1100 runtime set** as package data (`caminus/hsaco/gfx1100/*.hsaco`) plus `provenance.json` | see "Code objects" |
| Guard | `caminus/share/guard.sh` (`tools/guard.sh` is a symlink to it) | `caminus serve` runs under it; see "Guard" |
| Console script | `caminus` with subcommands `serve`, `download`, `doctor`, `purge`, `replay`, `flight`, `kv-cache`, `guard`, `version` | one entry point; the old `tools/replay.py`, `tools/flight.py`, `tools/kv_cache.py` stay as thin wrappers of the same code |

**Dependencies declared:** `numpy`, `jinja2`, `tokenizers` (lower bounds only; tested with numpy 2.5.4, Jinja2 3.1.6, tokenizers
0.23.3). Jinja2's floor is 3.1.6, a security floor: the chat template comes from the model file and renders in Jinja2's sandbox,
which has known escapes before 3.1.6 (CVE-2024-56326, CVE-2025-27516); `caminus.serve.template` refuses an older one and
`caminus doctor` reports it.

**PyTorch is not declared.** The ROCm build of PyTorch is not on PyPI: a declared `torch` makes pip pull a CUDA build. The user
installs it from PyTorch's ROCm index first (`pip install torch --index-url https://download.pytorch.org/whl/rocm7.2`; tested
with torch 2.14.1+rocm7.2). `caminus doctor` and the engine's first HIP call check for it and print that command.

**gguf-py is vendored, not declared.** The Coder holds `Q2_0` tensors (GGML type 42). llama.cpp added `Q2_0` to gguf-py on
2026-07-07 (`bec4772f6`), after gguf-py's last version bump (0.19.0, 2026-05-06, `a290ce626`), so a gguf from PyPI at or below
0.19.0 refuses the file (`GGUFReader` maps every tensor type through the enum). The fork's gguf-py equals upstream's at
`5aeac37a35` (no diff), and the engine always read it from the fork's tree by `sys.path`. The four modules the reader needs
(`constants.py`, `gguf_reader.py`, `quants.py`, `lazy.py`) are copied verbatim from that commit into `caminus/_vendor/gguf/` with
a reduced `__init__.py` (the reader's names only, so PyYAML / requests / sentencepiece are not needed), MIT notice kept,
listed in THIRD_PARTY.md, checked by `tools/release/vendor_gguf.py --check`. The same `sys.path` mechanism puts it in front;
`CAMINUS_GGUF_PY` (or the old `K3_GGUF_PY`) points at another gguf-py tree for development.

## Code objects

**Prebuilt, not built at install.** The build needs the TheRock nightly (10.2.0a20261011: amdclang++ and the offload
driver's device LTO), the fork's ggml headers at its commit, and build.sh's own per-module pipelines; the prefill's MMQ objects
must keep llama.cpp's instructions (tests/prefill/test_prefill_mmq.py codegen) and the K3 objects the bitwise numerics the gates hold.
A pip install cannot provide that toolchain, and a different compiler would not give the same code. A source build stays
possible: `./build.sh` in a checkout, then `CAMINUS_KERNEL_DIR=<dir>`.

**The runtime set** (what serving, replay and `doctor --gpu` load; staged by `tools/release/stage_kernels.py` from `build/`):

| Group | Objects |
|---|---|
| K3 decode | `k3_gguf_gsq`, `k3_gguf_gsq_mt`, `k3_gguf_gsq_qsa`, `k3_gguf_gsq_qsa_mt` (the Coder), `k3_gguf_gsq_mtp`, `k3_gguf_gsq_mtp_mt` (MTP), `k3_gguf_draft` (DFlash), `k3_gguf`, `k3_gguf_mt` (REAP-308), `k3_guard` (the separate bound check) |
| Batched prefill | `prefill_attn`, `prefill_feed`, `prefill_gdn`, `prefill_glue`, `prefill_tp`, `prefill_dense_q6_k`, `prefill_mmq_*` (11), `prefill_moe_*` (7), `gemm` (the WMMA BF16 GEMMs) |
| Toolchain check | `smoke` (`caminus doctor --gpu` launches its saxpy once per card) |

**Not shipped:** the `*_trace` profiling builds (5), the `*_swiglu` test builds, `gg_mmvf_ref` (a test reference), the
dormant S1 engine (`k3`, `k3_trace`, `k3_u16`, `k3_mv`), the dormant training kernels (`actq`, `attn`, `core`, `fuse`, `gdn`,
`headloss`, `wquant`) and the probes (`bw_bench`, `sync_bench`, `wmma_probe`). A missing object raises with the way to build it.

**Provenance** (`caminus/hsaco/gfx1100/provenance.json`, printed by `caminus serve` at startup and checked by `caminus doctor`):
the toolchain (name, version directory, `amdclang++ --version`), the architecture, each object's exact compile command as
build.sh ran it (paths normalized to `<ROCM_PATH>`, `<IGNITE>`, the repository root), the source tree's commit and dirty flag,
a content hash of the kernel sources and build.sh (history-independent: the public squashed tree gives the same value as the
development tree), the fork's commit for the ggml headers, each object's sha256 and size, and `built` (SOURCE_DATE_EPOCH when set,
so a release rebuild of the same commit writes the same file: tools/release/release_build.sh sets it to the commit's committer time;
`built_from` says which). build.sh records the toolchain, the sources' state and every command as it runs (build/BUILD-INFO,
build/BUILD-CMDS.tsv); the stage step refuses objects whose recorded kernel-tree hash differs from the tree's now (not built
from these sources) and a dirty build unless told otherwise.

**Where the runtime finds them** (`caminus.device.kernels.kernel_dir`): `CAMINUS_KERNEL_DIR`, else a checkout's `build/` (the package
imported from `python/` beside `build.sh`: every test and tool keeps its current behaviour), else the package's own
`hsaco/gfx1100`. The prompt cache's engine key hashes that directory's objects, as it hashed `build/` before.

## Paths: one config layer (`caminus.config`)

Order: environment, then `~/.config/caminus/config.toml` (`$XDG_CONFIG_HOME`; `CAMINUS_CONFIG` names another file), then the
XDG defaults. No personal path remains in shipped code.

| Key (config.toml) | Environment | Default |
|---|---|---|
| `model_dir` | `CAMINUS_MODEL_DIR` | `$XDG_DATA_HOME/caminus/models` (`~/.local/share/caminus/models`): where `caminus download` puts the files |
| `coder` | `CAMINUS_CODER` (old: `K3_GGUF`) | the Coder's first shard under `model_dir` (its repository layout) |
| `dflash` | `CAMINUS_DFLASH` | the DFlash Q8_0 GGUF under `model_dir` |
| `mtp` | `CAMINUS_MTP` | none (the MTP sidecar is not distributed; MTP is parked) |
| `kv_dir`, `flight_dir` | `CAMINUS_KV_DIR`, `CAMINUS_FLIGHT_DIR` | `$XDG_CACHE_HOME/caminus/kv`, `.../flight` (as before: `~/.cache/caminus/...`) |
| `state_dir` | `CAMINUS_STATE_DIR` | `$XDG_STATE_HOME/caminus` (the guard's log for `caminus serve`) |
| `kernel_dir` | `CAMINUS_KERNEL_DIR` | see above |

`caminus serve --model / --drafter / --cache-dir / --flight-dir` and `--config FILE` still override all of it. Tests and dev
tools keep their environment-driven paths (CLAUDE.md's `$CODER` etc.); on this machine the user's own models need either
`config.toml` (`coder = "..."`, `dflash = "..."`) or the environment, because the old defaults pointed at this machine's drives.

## Guard

`caminus/share/guard.sh` replaces the machine-specific `tools/guard.sh` and never weakens it.

- **Cards:** every amdgpu card whose KFD node is gfx1100 (`gfx_target_version` 110000, the target the package ships), in DRM
  order. The iGPU is not watched: its VRAM is a small carve-out the desktop uses (here 372 of 512 MiB), so a 0.97 line would kill
  for nothing.
- **Kill lines from each card's total:** `floor(floor(total_MiB * 97 / 100) / 50) * 50`, the 0.97 of CLAUDE.md rounded down to
  50 MiB. Here: 24,560 MiB -> **23,800** (XTX), 20,464 MiB -> **19,850** (XT), the old constants exactly. Host MemAvailable
  1,500 MiB as before.
- **The display-driving card:** a card with a connected connector (`/sys/class/drm/cardN-*/status`) or `boot_vga` = 1. It gets the
  prompt cache's 512 MiB reserve below its line (`capacity`; the others keep the usual 256 MiB margin): here the XTX, as the old
  constant `(512, 256)` by engine order. `guard.sh --lines` prints the cards, totals, lines, display flags and reserves;
  `caminus.device.machine` computes the same in Python for `capacity` (a test compares both on fixture trees and live).
- **The amdgpu fault watch without root:** `journalctl -k` when the user can read it (the `systemd-journal` / `adm` groups),
  else `dmesg` (when `kernel.dmesg_restrict` = 0), else a `GUARD WARNING` line at the start saying the fault watch is off and how
  to enable it. The STOP file and exit codes 98 / 99 are unchanged.
- **Clocks:** `CAMINUS_GPU_HIGH`, else `gpu-high` on `PATH`, else `~/.local/bin/gpu-high` (a systemd user manager's PATH may
  lack it); without one the command runs at the current clocks and the guard says so (`caminus serve`'s clock check reports
  the level).
- **Output:** the 1 Hz `.mon` line is `time used_MiB... avail_MiB` per watched card (here `time XTX XT avail`, as before), the
  kill line names each card by its model where known, the rc line keeps its form.
- **`caminus serve`** (the console script) re-executes itself under the shipped guard unless it already runs under one (the
  guard exports `CAMINUS_GUARD`), with the server's log in `state_dir/serve.log`. `tools/caminus_serve.py` and
  `python -m caminus.serve.cli` stay unwrapped for tests that run under an outer guard.

## caminus download

The model files come from their public Hugging Face repositories at pinned revisions, with sizes and sha256 in the package
(`caminus.models`):

| File | Repository @ revision | sha256 source |
|---|---|---|
| Coder IQ1_M shards 1 and 2 | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` @ `bba88856` | the LFS ids Hugging Face served when this machine downloaded them (their `hf download` metadata); the local files hash to the same values (`caminus doctor --verify`) |
| DFlash2 Q8_0 | `incoai/Qwen3.8-Flash-Next-DFlash2-GGUF` @ `4f5a39be` | this machine's file (hashed), then checked against Hugging Face's API (2026-11-06): the repository serves `Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf` at that revision with size 652,101,024 and the same LFS sha256 (Apache-2.0, public, not gated); the Coder's shards at `bba88856` match too |

`.part` files with HTTP Range resume, retries, a free-space check, size and sha256 verified before the rename, a `.sha256`
marker so later checks need not re-hash; `HF_ENDPOINT` and `HF_TOKEN` as huggingface_hub reads them. It could not run against
Hugging Face here; tests/packaging/test_packaging.py drives it against a local fake HTTP server (resume, a server that ignores Range,
a dropped connection, a corrupt or short file, 404).

## caminus doctor

PASS / WARN / FAIL lines and a RESULT, CPU-safe by default (it reads sysfs and KFD topology, never initializes HIP): Python and
platform, the package and every code object against its provenance, the vendored gguf-py, PyTorch (a ROCm build, or the
install command), the gfx1100 cards (count, VRAM total / used, guard lines, the display card, other KFD processes, the link),
clocks and runtime suspend, the guard (present, `gpu-high`, the STOP file, kernel-log readability), host RAM, the model files
(size; sha256 with `--verify`), disk space for the caches, port 8092. `--gpu` (under the guard) loads `smoke` on each gfx1100
card, runs one saxpy and checks P2P access.

## Verified in this step (2026-11-06)

- **Code objects:** ./build.sh in the worktree: all 56 objects byte-identical to main's build/*.hsaco, before and after the
  build.sh recording change and the gdn.hip notice. The staged runtime set: 36 objects (4.1 MiB) from 0ceb45c, clean.
- **Wheel:** `pip wheel . --no-deps --no-build-isolation`: caminus-0.1.0-py3-none-any.whl, 2.3 MB, 122 files (36 code objects
  matching provenance.json, the guard, the vendored reader, LICENSE / THIRD_PARTY / Apache-2.0); no machine path or
  session-link string in it. Installed into a fresh scratch venv (`--system-site-packages`, `--no-deps --no-index`): every
  console command runs; `caminus doctor` finds the package's 36 objects matching their provenance.
- **GPU, through the installed package, under the guard (the cards idle):** `caminus doctor --gpu` (the XT alone, then both:
  saxpy exact on each, peer access both ways, RESULT PASS); tools/k3_smoke.py (RESULT PASS: KLD 0.0044 vs floor 0.0063, plain
  103.8-105.1 tok/s, DFlash 156.0 / 202.8 / 123.2 tok/s, MTP == plain); tools/serve_smoke.py --stages server (RESULT PASS, 38
  checks: 262K fit XTX 23,071 / XT 19,021 MiB, guard peaks 23,091 / 19,038, 101,525 tokens cold at 2,538 tok/s, the parks).
- **CPU:** tests/serve/run_all.py (167 OK), tests/packaging/test_packaging.py (72 checks), CLAUDE.md's CPU stages, vendor checks.
- **Model files:** `caminus doctor --verify` hashed this machine's two Coder shards and the drafter (58.4 GB, 80 s): all three
  equal the manifest's sha256.
- **sdist:** setuptools' build_sdist carries the package, the staged objects, provenance.json, the guard, setup.py and the PyPI
  readme (MANIFEST.in leaves the tests out); a wheel built from it has the same files, byte for byte, as the tree's wheel.
- **Not run:** a download from Hugging Face (no network: a local fake server only; the manifest's three entries were checked
  against Hugging Face's API on 11-06 instead), `caminus replay` of a real bundle, the REAP-308 modules (shipped, not loaded), the guarded re-exec of
  `caminus serve` / `replay` on a GPU (checked on the CPU only), a VRAM or fault kill, the systemd unit, another machine.
- **Before the upload** (done in the next step, impl:public): the wheel is built from a clean clone of the public repository,
  so provenance.json names its commit; `[project.urls]` points at github.com/jdominick05/caminus. kernel_tree_sha256 is now
  8c85f2dcd6fe... (comment-only kernel edits since; the code objects are byte-identical).

## Limits found, not changed here

- The engine itself still assumes a 24 GB card 0: `cuda:0` must be the larger card (replay asserts an XTX at 0), the fixed
  VRAM parts in `caminus.device.capacity` were measured on the XTX + XT (another pair is checked against its own guard lines with
  these measured parts), and an unmeasured model is refused rather than guessed. The guard, the paths and the package are
  portable; the engine's capacity tables are not (yet).
- `packaging/systemd/caminus.service` runs a checkout (`%h/src/caminus/tools/guard.sh` and a conda env's python): an example for
  a checkout, not for the installed package (docs/quickstart.md).
- `tools/` and `tests/` take their paths from the environment (`caminus.devpaths`: CODER, DFLASH, MTP, REAP, ARTIFACTS); the
  reference dumps the gates read are not distributed; neither directory is shipped in the wheel.

## Not in this step

Done in the next step (impl:public): the README and docs/quickstart.md for strangers, docs/dev/benchmarks.md, the dev tools' paths
from the environment, and the public tree. The fork's reference commit is public since 11-06: 842653396, published as
e4e3f8eda on github.com/jdominick05/ignite-llama (identical ggml/, common/, src/). Still open: a systemd unit for the
installed package, and publishing the public repository (the maintainer's steps).
