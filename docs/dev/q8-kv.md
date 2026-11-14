# Own Q8 KV cache path (staged)

The own-code dense line has separate `*_kvq8` persistent objects. They preserve the F16 objects and weight-format
selection. `Dense35Own(..., q8=True)` is an internal bring-up path; the server and panel expose no Q8 selector.
Matching one-card batched prefill is also staged. The high-level two-card engine still refuses Q8.
Device correctness, model accuracy and speed remain unverified for these objects.

## Wire format and rotations

Each row holds K then V, each in 32-element blocks of a half scale and 32 signed bytes (34 bytes per block).
`kernels/own_kv_codec.h` owns the codec; `caminus.ref_kv` defines its CPU reference. The scale calculation and rounding
are fp32, with only the stored scale rounded to half. Invalid numerical domains refuse the block before writing it;
other blocks in that append may already have been written.

The dense path applies normalized Sylvester Hadamard256 to K and Q after normalization/RoPE, and Hadamard64 to V
before encoding. Attention dequantizes the same blocks. Its output receives Hadamard64 **before** the channel gate.
The staged host path requires head dimension 256 and six query heads per KV head. Verify uses the unpacked attention
reader: the packed reader is F16-only and compilation rejects that combination for Q8.

Swift and ThinkingCap have one-token decode, T-token verify and low-level TP decode objects. GSQ has a decode object;
its verify and TP implementation are separate work. Drafter caches, complete two-card
integration, and Coder/Qwen3 format selection are still required for an end-to-end selectable cache format.

## Terminal append failures

The Q8 engine allocates one additional int32 error word after the existing Sync storage. Each append's `Seg.gs`
points to it. A refused block atomically ORs bit 2 there; dependency timeouts use the original status word, so they
cannot overwrite the numerical failure. The host validates the pointer, format, buffer extents and alignment.

`Engine.run(check=True)` waits and checks the error before timeout recovery or status clearing. Asynchronous callers
must wait and call `check_kv_error()` themselves. Dense decode and verify do so before their retry loops or logits
consumption. A refusal marks the target engine permanently broken and raises `InvalidKVEncoding`, a
`NumericalFailure`. Reset keeps that broken state. The worker rejects later requests and queued bundle exports;
startup and shutdown use exit 75 and skip parking. The prompt cache also rejects direct park, restore, append and
commit calls once its engine is broken. The existing `NonFiniteLogits` error shares this terminal policy.

## Memory and checkpoints

The target cache is a byte tensor with `max_ctx + spec_T - 1` physical rows per attention layer. Projections count
that exact storage, split-attention scratch and the additional four-byte error word. `cache.prompt.Layout` uses
physical state-view rows to calculate byte stride, including the verify tail. The checkpoint key adds
`target_kv=q8_0-v1:qk-hadamard256:v-hadamard64-v1`; F16 keys retain their existing fields. Cross-format reuse must
remain refused when the served path is integrated.

## Matching batched prefill

`Q35OwnPrefill` preserves each weight line's existing GEMM module and loads one shared `q35_prefill_kvq8` object
for Q8 append and tiled attention. Its Q/K and V/output transforms use the same order as decode. Buffer projections
retain the existing workspace: the Q8 specialization adds no device buffers. The prefill numerical key identifies
its WMMA/FP32 fallback arithmetic separately from the target wire codec.

WMMA stages dequantized K/V and Q in fp16. Valid Q8 blocks can decode above fp16's finite range, so simply casting
every value would lose valid inputs. The Q8 kernel records the earliest unsafe K/V position and unsafe query rows.
Safe causal rows keep WMMA; affected rows use original fp32 Q and fully decoded Q8 in an online-softmax fallback.
Temporary half operands which overflow are zeroed only in discarded or causally masked WMMA calculations. A row's
path depends on its own query and causal prefix, preserving that choice across chunk boundaries and launch splits.
The gate can force FP32 for comparison; no codec-domain narrowing or saturation is allowed.

After every Q8 chunk, the prefill waits and checks flags before supplying DFlash features or MTP final h. A
numerical flag wins over bounds handling and remains sticky. The common driver checks flags after draining an
ordinary host exception too, before setting a recoverable poison state. Already-broken engines refuse even a
previously validated run. F16 retains its existing numerical-key fields and ordinary error recovery.

Fallback timing at depth is unverified. The existing WMMA launch budget does not establish a safe fallback launch
duration; measure its slowest launch before enabling this path in serving. No speed or model-accuracy claim follows
from compiling it.

## Evidence and remaining gates

CPU checks construct real Swift/ThinkingCap tapes against synthetic GGUFs, validate format/pointer refusals,
allocation/projection/rotation ordering and checkpoint stride, and inject an encoding error at the event wait.
They verify decode and verify do not retry even when a timeout is also set. Worker tests cover 503, queued exports,
startup exit 75 and shutdown without parking. No CPU check proves device arithmetic.

```bash
python -I tests/ci_cpu.py --only q35_tape,dense_q8_tape,q35_kvq8_host,q35_q8_prefill,q35_q8_fixtures,numerical_failure,numerical_state,ref_kv,own_kv_codec,kv_codec,own_code,docs --jobs 1 --scratch build/q8-cpu
```

After a whole-tree build, compare sections against the F16 branch point with `tools/gates/objcheck.py`. Every new
Q8 persistent object must also be explicitly paired with its corresponding F16 object: at most 96 VGPRs, zero
vector spills and no increase in scalar spills. Local GDN replay addresses and the codec's single finite-input
check keep the additional operations within that budget; they do not establish a speed improvement.

`tests/prefill/test_q35_kvq8_ops.py --cpu-fixtures` checks the device gate's fixtures without importing torch. The
actual GPU form requires the outer guard and a single non-display card; it checks bytes, rotations, decoded-Q8
attention, wide-value fallback, chunk/split identity and canaries. Its accuracy bounds match the existing F16
prefill attention gate, applied separately to each token and query head so wide valid fallback outputs cannot hide
an omitted or incorrect normal WMMA vector. Reports contain the worst normalized max and mean across those
vectors. CPU fixtures exercise mixed output magnitudes, missing normal vectors and zero-RMS references; their
success is recorded separately from device execution.

The full-model `tests/prefill/test_q35_prefill.py` gate accepts test-only `--kv q8` (default F16). It compares every
byte of each committed cache row, including Q8 scales and final V blocks, and the live recurrent state across chunk
sizes and prompt-cache stops. Snapshot memory is projected from those exact byte views. The gate's timing branch
uses the selected format's attention launcher. A recorded llama comparison must declare a matching `--llama-kv`;
the legacy default is F16, so an F16 reference is refused for a Q8 run. Verify the saved run metadata before declaring
its format. This argument is not proof of provenance, and no reference is regenerated or tolerance relaxed here.
For example, after valid-input non-display operator validation:

```bash
tools/guard.sh LOG 1800 python -I tests/prefill/test_q35_prefill.py GGUF --ref REF.npz --kv q8 --llama LL-Q8.bin --llama-kv q8
```

`q35_prefill_state` in the CPU runner checks complete F16/Q8 snapshot coverage, speculative-tail exclusion,
recurrent bit patterns and mismatched reference refusals. These are harness checks, not executed model gates.
Q8 versus F16 model-quality acceptance and the wide-value fallback's latency at depth remain outstanding.

Before enabling model use, run guarded non-display GPU gates for codec byte parity, rotations and attention against
the float64 reference over the **same decoded Q8 rows**, verify versus sequential passes at allowed resident grids,
and model accuracy at depth. Prefill-to-decode, accepted/rejected draft tails, cache park/restore and both card counts
need their own gates. Installed runtime registration and fresh-wheel acceptance follow that integration.
