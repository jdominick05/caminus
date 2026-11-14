# Own Q8 KV cache path (staged)

The own-code dense line has separate `*_kvq8` persistent objects. They preserve the F16 objects and weight-format
selection. `Dense35Own(..., q8=True)` is an internal bring-up path; the server and panel expose no Q8 selector.
Batched prefill and the high-level two-card engine explicitly refuse it until their matching implementations land.
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
its verify and TP implementation are separate work. Drafter caches, matching batched prefill, complete two-card
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

## Evidence and remaining gates

CPU checks construct real Swift/ThinkingCap tapes against synthetic GGUFs, validate format/pointer refusals,
allocation/projection/rotation ordering and checkpoint stride, and inject an encoding error at the event wait.
They verify decode and verify do not retry even when a timeout is also set. Worker tests cover 503, queued exports,
startup exit 75 and shutdown without parking. No CPU check proves device arithmetic.

```bash
python -I tests/ci_cpu.py --only q35_tape,dense_q8_tape,numerical_failure,numerical_state,ref_kv,own_kv_codec,kv_codec,own_code,docs --jobs 1 --scratch build/q8-cpu
```

After a whole-tree build, compare sections against the F16 branch point with `tools/gates/objcheck.py`. Every new
Q8 persistent object must also be explicitly paired with its corresponding F16 object: at most 96 VGPRs, zero
vector spills and no increase in scalar spills. Local GDN replay addresses and the codec's single finite-input
check keep the additional operations within that budget; they do not establish a speed improvement.

Before enabling model use, run guarded non-display GPU gates for codec byte parity, rotations and attention against
the float64 reference over the **same decoded Q8 rows**, verify versus sequential passes at allowed resident grids,
and model accuracy at depth. Matching prefill, accepted/rejected draft tails, cache park/restore and both card counts
need their own gates. Installed runtime registration and fresh-wheel acceptance follow that integration.
