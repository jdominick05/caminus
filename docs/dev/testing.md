# Testing

By the end of this page you'll know which tests to run for a change and what they prove. Tests are plain scripts that
print PASS or FAIL lines and end with `RESULT PASS` (no pytest). Run Python with `-I` so it ignores `PYTHONPATH`.

## Before every pull request: the CPU suites

```bash
python -I tests/ci_cpu.py --scratch <a fresh directory>
```

This runs every CPU-only suite, the same set CI runs: the server and its APIs, the prompt cache on a simulated engine, the
host-side refusals of every GPU launcher, the drafters' host code, packaging, card detection on fake sysfs trees, and the
docs check. Suites that need the model files or PyTorch skip with a reason when those are missing. Use a fresh scratch
directory each time: some suites refuse leftovers from a previous run.

Up to 8 suites run at once, and at most one per 2 GiB of free host memory (`--jobs N`; `--jobs 1` runs them one after
another). Each suite is its own process with its own scratch directory and log, the results print in the same order
either way, and the suites that load the Coder's tensors never overlap one another. A suite that fails prints its
failure reports (FAIL / ERROR lines and tracebacks) and the end of its log; the whole log is `<scratch>/<suite>.log`.

`cd panel && go test ./...` tests the panel.

The one-card server smoke's tool acceptance check has a small CPU regression suite, with synthetic SSE through the real
collector and no sockets or model: `python -I tests/tools/test_serve_smoke1.py` (also the `serve_smoke1` CPU suite).
Installed-mode command and preflight refusals have a separate CPU suite,
`python -I tests/tools/test_serve_smoke_installed.py --scratch DIR`. Its package/provenance fixtures never execute
GPU code; the CLI is replaced by a recorder.

### Staged own Q8 KV codec

The internal `device.kv_codec.Q8KVCodec` writer is a foundation for new cache paths; it does not enable a model's Q8 mode.
Its format and failure rules: [q8-kv.md](q8-kv.md#wire-format-and-rotations).

`tests/io/test_own_kv_codec.py --scratch DIR` compiles the actual own header as host C++ and compares its bytes with
the independent NumPy reference. It checks roundf boundaries, stored-scale rounding, half underflow/subnormals,
seeded finite inputs and numerical refusals. `tests/device/test_kv_codec.py` checks launch spans, view extents,
alignment, aliasing and dispatch using CPU recorders. CI runs both; the host compilation needs a C++ compiler.
These tests establish source and host behavior, not GPU byte parity or model quality.

The GPU gate is `HIP_VISIBLE_DEVICES=N tools/guard.sh LOG 180 python -I tests/device/test_kv_codec_gpu.py --json OUT`,
on the non-display card first (the script uses the first visible card). It compares the device's bytes with the
independent reference over head widths 32 / 128 / 256, several head counts, padded input strides, last valid rows,
chunk cuts and cache canaries; invalid spans and layouts stay on the CPU tests. The writer alone is not Q8 model
support: attention, rotation, verify, prefill, drafting, persistence, capacity and each model's accuracy and speed gates
are separate.

## GPU tests

GPU tests always run under the guard (`tools/guard.sh LOG SECS CMD...`) and follow [GPU safety](gpu-safety.md). Each
script's header says what it needs and what it checks. The main kinds:

| Kind | Example | What it proves |
|---|---|---|
| Per-operation | `tests/prefill/test_prefill_glue.py GGUF gpu` | Each kernel gives the same bits as the decode engine's operation, or as llama.cpp's |
| End to end | `tests/engine/test_flashnext_e2e.py ... --tp` | The whole model against llama.cpp's own output |
| Server | `tools/serve_smoke.py --scratch DIR` | The real `caminus serve`: tool loops on both APIs, the cache, restarts, cancel, a 100K prompt |
| One-card server | `tools/guard.sh LOG 2400 python -I tools/serve_smoke1.py --model GGUF --scratch DIR` | The selected model calls a weather tool, consumes every result and answers with their synthetic observation codes |
| Speed | `benchmarks/fn_bench.py 96 GGUF --tp` | Time per token |

The one-card smoke first lets the model choose a tool naturally. If it emits no call, the smoke reports one retry with
`tool_choice: required`, using the same model and thinking policy. Every call must use the declared name, a JSON object
with a nonempty city string and a unique nonempty ID. Every result is replayed, and the final `tool_choice: none` turn
must finish with weather text containing every result's observation code and the supplied temperature, without more
calls. Text alone, invalid arguments, truncated answers or ignoring the results fail. The greedy and seeded-sampling
prompts used for drafter hash comparisons are independent of this tool check. CPU tests prove the acceptance logic;
the real model's behavior still needs the guarded server smoke.

For a fresh wheel installed in a separate environment, select that interpreter and set `CAMINUS_SMOKE_INSTALLED=1`:

```bash
CAMINUS_SMOKE_INSTALLED=1 tools/guard.sh LOG 2400 "$WHEEL_PY" -I tools/serve_smoke1.py \
  --model GGUF --scratch DIR --draft off --block-imports gguf,tokenizers --restart
```

`WHEEL_PY` is the installed environment's Python executable. The child uses `-I`, verifies that the imported package
matches its installed distribution, refuses editable/source imports and kernel_dir overrides, and checks the exact
packaged object set against provenance before starting the server. The outer guard is required so an automatic
guard re-exec cannot discard the import-blocking hook. Already imported forbidden packages are refused because
Python's module cache would bypass that hook. Every restart repeats these checks.

With `--block-imports`, the installed kernel-tree digest must also match this checkout: its own-code include audit
then describes the same kernel sources as the installed objects. Use a checkout matching the wheel, and clear any
kernel_dir setting in the test environment/config. The harness prints the installed version, origin and verified
object count. CPU preflight success proves package admission only; the full model/tool/restore tests still run on
the GPU. Without the environment switch the existing checkout server path is used.

## Dense two-card drafting

`--cards 2 --draft mtp|dflash` for Swift 1.5 and ThinkingCap passed gates 1-4 below on both models (2026-11-15) and is
served; `--draft dflash` is the two-card default (docs/dev/benchmarks.md, "Speculative decoding on two cards"). A file
whose decode module has no two-card verify object (the plain 27B's) is refused two-card drafting. The CPU check covers
the feature-copy dependencies on either card, the drafter's full head and its uploads, the cache identity, the
checkpoint layout against its projection, the prefill hooks, the drafter card's choice, that refusal and the two-card
default with its 32K rule:

```bash
python -I tests/engine/test_q35_verify_tp_cpu.py --scratch SCRATCH
```

GPU gates, in order, each under the guard and with the upload log on (`CAMINUS_UPLOAD_LOG=ULOG`; a fault address goes
to `tools/gates/sdma_classify.py`). `L` is the DFlash2 file's `dflash.target_layers`, comma-separated; `KIND` is
`off`, `mtp` and `dflash` in turn (gate 4: `mtp` and `dflash`):

```bash
# 1. the TP verify module alone on the non-display card (one-card tape, the module's ops; features bitwise)
CAMINUS_UPLOAD_LOG=ULOG HIP_VISIBLE_DEVICES=<non-display card> tools/guard.sh LOG 600 python -I tests/engine/test_q35_tp_ver_probe.py --scratch SCRATCH
# 2. per model: two-card load with read-back, then verify bitwise the decode steps, features on each card in turn
CAMINUS_UPLOAD_LOG=ULOG tools/guard.sh LOG 900 python -I tools/gates/tp_load_only.py GGUF
CAMINUS_UPLOAD_LOG=ULOG tools/guard.sh LOG 900 python -I tests/engine/test_q35_verify_tp.py GGUF --feature-layers L --feature-card 0
CAMINUS_UPLOAD_LOG=ULOG tools/guard.sh LOG 900 python -I tests/engine/test_q35_verify_tp.py GGUF --feature-layers L --feature-card 1
# 3. per model and KIND: the served end to end (greedy and sampled identity against --draft off, own imports, restart)
CAMINUS_UPLOAD_LOG=ULOG tools/guard.sh LOG 2400 python -I tools/serve_smoke1.py --model GGUF \
  --scratch DIR --card 0,1 --cards 2 --draft KIND --block-imports gguf,tokenizers --restart
# 4. per model and drafter: the served gain at depth, at least three interleaved rounds
tools/guard.sh LOG 3600 python -I benchmarks/q35_served_ab.py --model GGUF --ids IDS \
  --scratch DIR --card 0,1 --cards 2 --draft KIND
# 5. per model: the default, with no --draft. Passes when the log has "--draft dflash (the default", "drafter on card N"
#    and the "two-card --draft dflash on card N" build line and no later "--draft off (" fallback line, and greedy_sha256 /
#    sampled_sha256 equal a --draft off run of the same smoke (record the "max_ctx auto ->" value)
CAMINUS_UPLOAD_LOG=ULOG tools/guard.sh LOG 2400 python -I tools/serve_smoke1.py --model GGUF \
  --scratch DIR --card 0,1 --cards 2 --block-imports gguf,tokenizers --restart
```

Gate 2 must hold at every accepted row, the rejected tails and grids 3/6/7 (the test runs them); gate 3 compares
`greedy_sha256` and `sampled_sha256` with the `--draft off` run. The two `_tp_ver` objects are in `device/kernels.py`'s
runtime set (`tests/packaging/test_runtime_kernels.py`). A plain two-card measurement passes `--draft off`.

## Qwen3-30B-A3B's 8-bit KV cache, served

`caminus serve --kv q8`: its CPU suites (`tests/engine/test_q3moe_kvq8_cpu.py`, `test_q3moe_tp_kvq8_cpu.py`) and the three
guarded smokes (one card, two cards, the format mismatch) are listed in
[q8-kv.md](q8-kv.md#serving-qwen3-30b-a3b-with---kv-q8).

## The accuracy rules

- **Bitwise:** a change to the engine must leave logits and routing bitwise equal to the stored references, across grid
  sizes and multi-token passes. `tools/gates/gates.sh OUT all` runs these gates.
- **Within llama.cpp's noise:** a change that can't be bitwise by design is held to llama.cpp's own batch-versus-decode
  floor: its mean KL divergence from llama.cpp's one-token decode may not exceed llama.cpp's own (0.0076 on the Coder),
  and its top-1 agreement must stay within a point.
- **Never refresh a reference to make a gate pass.** A gate that stops matching is a finding.
- **Speed claims** need interleaved A/B runs, the median of at least three, and a difference above 0.05 ms.

The references the gates compare against (llama.cpp's tensors and logits) are large and not distributed. The tools that
make them are in `tools/reference/` and `tools/gates/`, so you can build your own on the same hardware.
