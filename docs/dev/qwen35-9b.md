# Qwen3.5-9B text serving and F16 MTP

The one-card `qwen35-own-9b` registry entry selects files with embedding4096,
16 query heads and4 KV heads. It uses the existing own dense loader and prefill,
with separate `k3_q35_9b`, `_ver` and `_mtp` code objects. Full attention has four
query heads per KV head; the existing27B modules retain six. The verify object
uses unpacked attention at T=1..4.

The restored prediction block may contain raw F16 matrices and F16 norms.
`OP_OWN_MVF16` reads the original F16 matrices and F16 activation rows and sums
with FP32 dot2 and the usual wave reduction. Each row's arithmetic is independent
of the pass width. Norms widen exactly to FP32 on upload; both runtime and
metadata-only MTP projections count the widened bytes. The target's Q6_K head
is shared with the drafter. No model file is rewritten or requantized.

For a compatible local GGUF, select MTP explicitly with `--draft mtp`. Default
drafting stays off pending served measurements. Two-card plain decoding selects
`k3_q35_9b_tp` with `--cards 2 --draft off`, using the existing dense tensor
parallel loader, split planner, exchange and prefill. DFlash and two-card MTP
are not advertised. Native context comes from the file; context placement still
checks the target, verify records, MTP cache, adapter and prefill workspace before
allocation. Advertising a context does not prove useful at-depth generation.

Before enabling this model, run the guarded non-display op gate
`tests/engine/test_q35_9b_ops.py`, whole-model decode/prefill quality gates,
`test_q35_mtp.py` and `test_q35_verify.py`, served off/MTP greedy+sampled identity,
then actual near-native-limit requests and measured VRAM. The CPU suite
`q35_9b` reads a synthetic GGUF and constructs every actual tape but proves no
device arithmetic. Source status and measured receipts belong in the PR body.
