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

## GPU tests

GPU tests always run under the guard (`tools/guard.sh LOG SECS CMD...`) and follow [GPU safety](gpu-safety.md). Each
script's header says what it needs and what it checks. The main kinds:

| Kind | Example | What it proves |
|---|---|---|
| Per-operation | `tests/prefill/test_prefill_glue.py GGUF gpu` | Each kernel gives the same bits as the decode engine's operation, or as llama.cpp's |
| End to end | `tests/engine/test_flashnext_e2e.py ... --tp` | The whole model against llama.cpp's own output |
| Server | `tools/serve_smoke.py --scratch DIR` | The real `caminus serve`: tool loops on both APIs, the cache, restarts, cancel, a 100K prompt |
| Speed | `benchmarks/fn_bench.py 96 GGUF --tp` | Time per token |

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
