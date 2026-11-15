# Own Q8 KV cache path (staged)

The own-code dense line has separate `*_kvq8` persistent objects. They keep the F16 objects and the weight-format
selection unchanged. `Dense35Own(..., q8=True)` is an internal bring-up path; the server and panel expose no Q8
selector. Batched prefill refuses a Q8 engine until its matching implementation lands.
Device correctness, model accuracy and speed are not yet verified for these objects.

## Objects

| Object | Built from | Use |
|---|---|---|
| `k3_q35_swift_kvq8` | `k3_q35_swift.hip` | Swift decode |
| `k3_q35_swift_ver_kvq8` | `k3_q35_swift.hip` + verify defines | Swift verify, unpacked PART |
| `k3_q35_tcap_kvq8` | `k3_q35_tcap.hip` | ThinkingCap decode |
| `k3_q35_tcap_ver_kvq8` | `k3_q35_tcap.hip` + verify defines | ThinkingCap verify, unpacked PART |
| `k3_q35_gsq_kvq8` | `k3_q35_gsq.hip` | plain GSQ decode (its verify is separate work) |

The GSQ line's verify pass is two objects' launches (`k3_q35_gsq_ver`, `k3_q35_gsq_ver_m`); neither has a Q8 build,
so `Dense35Verify` refuses a Q8 GSQ engine. Each launch of a pass has its own error word; the pass checks every
launch's word before any bound or timeout handling.

`k3.Q35_KV_Q8_BASE` maps each object to its F16 peer; the module families and `MODULE_CAPS` follow the peer.
One card only: the two-card Q8 objects wait for their own fault investigation.

## Wire format and rotations

Each row holds K then V, each in 32-element blocks of a half scale and 32 signed bytes (34 bytes per block).
`kernels/own_kv_codec.h` owns the codec; `caminus.ref_kv` defines its CPU reference. The scale calculation and rounding
are fp32, with only the stored scale rounded to half. Invalid numerical domains refuse the block before writing it;
other blocks in that append may already have been written.

The dense path applies normalized Sylvester Hadamard256 to K and Q after normalization/RoPE, and Hadamard64 to V
before encoding. Attention dequantizes the same blocks. Its output gets Hadamard64 **before** the channel gate.
The staged host path requires head dimension 256 and six query heads per KV head. Verify uses the unpacked attention
reader: the packed reader is F16-only and compilation rejects that combination for Q8.

## Append failures

The Q8 engine allocates one more int32 error word after the existing Sync storage. Each append's `Seg.gs` points to
it. A refused block atomically ORs bit 2 there; dependency timeouts use the original status word, so they cannot
overwrite the numerical failure. The host checks the pointer, format, buffer extents and alignment.

`Engine.run(check=True)` waits and checks the error word before timeout recovery or status clearing. Asynchronous
callers must wait and call `check_kv_error()` themselves. Dense decode and verify do so before their retry loops or
logits use. A refusal marks the engine permanently broken and raises `caminus.errors.InvalidKVEncoding`; reset keeps
that state. `mark_broken` skips the weight checksum for it: the refusal names the cache, and the check reads every
weight. `InvalidKVEncoding` is a `NumericalFailure`, so the server's terminal policy applies (no retry, reset,
recovery or park; exit 75). No served option selects Q8 yet.

## Memory

The target cache is a byte tensor with `max_ctx + spec_T - 1` physical rows per attention layer (`Dense35Own.kv_rows`).
Projections count that exact storage, the split-attention scratch and the four-byte error word.
`Dense35Own.kv_format_id` names the format (`q8_0-v1:qk-hadamard256:v-hadamard64-v1`) for a checkpoint key; the
prompt cache does not read it yet. The checkpoint layout already divides by `kv_rows`.

`k3_run` of each Q8 object beside its F16 object, from one build of this tree (`llvm-readelf --notes` fields;
VGPR spills 0 and private segment 0 on all ten):

| F16 object | VGPRs | SGPR spills | LDS | Q8 object | VGPRs | SGPR spills | LDS |
|---|---|---|---|---|---|---|---|
| `k3_q35_swift` | 86 | 62 | 3376 | `k3_q35_swift_kvq8` | 86 | 53 | 4400 |
| `k3_q35_swift_ver` | 93 | 88 | 6464 | `k3_q35_swift_ver_kvq8` | 93 | 76 | 5168 |
| `k3_q35_tcap` | 84 | 58 | 14384 | `k3_q35_tcap_kvq8` | 83 | 48 | 15408 |
| `k3_q35_tcap_ver` | 96 | 74 | 16704 | `k3_q35_tcap_ver_kvq8` | 96 | 67 | 15408 |
| `k3_q35_gsq` | 86 | 65 | 20528 | `k3_q35_gsq_kvq8` | 85 | 54 | 21552 |

Decode objects add the Hadamard op's 1024-byte buffer. Verify objects add it too but drop the packed PART's scratch
(3220 to 900 bytes): 1296 bytes less in all. `objcheck.py check` against a build of the branch point: the 120 existing
objects keep their code and data sections; the five Q8 objects are the only new ones.

## Register pinning in the Q8 objects

`k3_q35_kvq8_ops.h` defines `K3_Q35_GDN_LOCAL_BASE`, which only the Q8 objects see. It pins three values in the
Gated DeltaNet step and save ops with empty `asm volatile` statements: the head counts (`nk`, `H`) as scalars and the
state base address as a vector value. The compiler then rebuilds them inside each op instead of keeping them live
across the persistent loop. Without it (the same tree, the define removed):

| Q8 object | VGPRs | VGPR spills | SGPR spills |
|---|---|---|---|
| `k3_q35_swift_kvq8` | 91 | 0 | 67 |
| `k3_q35_swift_ver_kvq8` | 96 | 0 | 94 |
| `k3_q35_tcap_kvq8` | 88 | 0 | 63 |
| `k3_q35_tcap_ver_kvq8` | 96 | 2 (16 B scratch) | 84 |
| `k3_q35_gsq_kvq8` | 90 | 0 | 71 |

So without the pin, ThinkingCap's Q8 verify object spills vector registers, and every Q8 object spills more scalars
than its F16 object. With it, none spills a vector register and each spills fewer scalars than its F16 object. The
pin moves no arithmetic; the device gates below still have to show the bits.

## Evidence and remaining gates

CPU checks build real Swift/ThinkingCap tapes against synthetic GGUFs, check format and pointer refusals,
allocation, projection and rotation order, and inject an encoding error at the event wait. They check that decode and
verify do not retry even when a timeout is also set. No CPU check proves device arithmetic.

```bash
python -I tests/ci_cpu.py --only q35_tape,dense_q8_tape,ref_kv,own_kv_codec,kv_codec,own_code,docs --jobs 1 --scratch build/q8-cpu
```

After a whole-tree build, compare sections against the F16 branch point with `tools/gates/objcheck.py`. Pair every
new Q8 object with its F16 object by hand: at most 96 VGPRs, zero vector spills and no increase in scalar spills.

Device gates, on a non-display card first (written, not yet run):

```bash
tools/guard.sh LOG 600 python -I tests/engine/test_q35_split_attn.py --kv q8
tools/guard.sh LOG 2400 python -I tests/engine/test_q35_verify.py GGUF --kv q8
```

`test_q35_split_attn.py --kv q8` also runs the Q8 objects' Hadamard and append ops on known K and V: the cache bytes
must equal `caminus.ref_kv.encode_q8_kv` of a host fp32 Sylvester transform, bitwise, at T 1 to 4. The output H64
must equal the host transform bitwise, once and twice. In fp32 on the host, twice is the input within 1.2e-7 x
max |x| (the check allows 1e-6). A NaN in K must raise `InvalidKVEncoding`.

The first runs the Q8 split attention bitwise against the GGUF module's pair on the same q8_0 bytes, and against
float64 attention over the same decoded rows (1e-5 x max |V|, the F16 bound: the decoded values are exact in fp32).
The second checks verify passes on a Q8 cache bitwise against Q8 one-token decode, depths 0 and 512, grids 3 / 6 / 7.
None of these has run on a GPU. Still missing after them: model accuracy at depth (KLD against the F16 cache),
speed, matching prefill, cache park/restore, the GSQ verify objects and two cards.
