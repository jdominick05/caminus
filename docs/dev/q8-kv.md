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

## Qwen3-30B-A3B (one card)

`Q3MoEOne(..., q8=True)` stores the KV cache as Q8_0 rows on `k3_q3m_lean_kvq8` (`k3.Q3M_KV_Q8_BASE` maps it to
`k3_q3m_lean`; the same ops, pairs and lean dispatch check). Its batched prefill `Q3MPrefill` follows the engine:
`q3p_qk_rope_q8` and `q3p_attn_q8` from `q3m_prefill_kvq8` write and read the rows, every other kernel stays on
`q3m_prefill`. Not served: no option selects it and neither object is in the installed runtime set. The registry's
projection takes `opt['kv'] = 'q8'` (`arch.kv_q8`) so the later `--kv q8` switch only has to set it.

Format: the same wire codec and row layout as the dense line (`kernels/own_kv_codec.h`, `caminus.ref_kv`), with no
rotation (`Q3MoEOne.kv_format_id` = `q8_0-v1:unrotated:f16-rounded-v1`). Each block encodes the fp16-rounded values the
F16 object stores, so a Q8 row is `encode_q8_kv` of the F16 row; the gates below check exactly that. The writers are
`own_kv_q8_encode_wave` (the codec's statements with one block per wave and the amax as a wave reduction, exact for the
finite blocks that are stored). The position's own K / V that the attention keeps in LDS are the decoded values, so the
current token reads what every later token reads. The prefill's attention rounds the decoded values to fp16 for its WMMA
operands, one rounding more than the decode reader; the model gate at depth judges it.

Refusals: a non-finite value (an fp16 overflow included) refuses its block. Decode sets bit 2 in the int32 word after
Sync (`Engine.kv_error`, the dense line's word; `own_op_atts` takes `sy` for it), the prefill in its own `kverr` word.
The word sits right after the status words, so a pass's one asynchronous status copy (`Engine.status_words`, 9 words
on a Q8 engine) brings it: no extra read back per token. `_pass_wait` checks it before the bound and timeout paths (they
zero Sync), `Engine.run(check=True)` likewise from its one status read, `step` marks the engine too, and the prefill's
settle checks its word before the routing flags. The engine stays broken across `reset()` and raises
`InvalidKVEncoding`. `Q3MoEVerify` and two cards refuse a Q8 engine, and every registry entry but the one-card
qwen3moe one refuses `opt['kv'] = 'q8'` (`Arch.kv_check`) instead of building its F16 cache.

Memory: 1,088 bytes per position and layer instead of 2,048 (`kv_pos_mib(q8=True)`: 0.0498 MiB per position over the
48 layers, F16 0.0938), counted by `projection_mib(q8=True)` and the registry's checkpoint slots. The engine key carries
`kv` (only when set: F16 keys are unchanged) and the module, so a Q8 park never restores into F16 rows. One-card
`max_ctx auto` from the projection (guard line, margin, display reserve, cache work and slack as `context_fit`
counts them, nothing else on the card):

| Card | F16 | Q8 | Room-limited, without the file's 40,960 cap (F16 / Q8) |
|---|---|---|---|
| RX 7900 XTX, display, idle | 40,960 | 40,960 | 45,056 / 81,920 |
| RX 7900 XTX, display, 1 GiB in use | 32,768 | 40,960 | 32,768 / 61,440 |
| RX 7900 XT, idle | 4,096 | 8,192 | 4,096 / 8,192 |

`k3_run` of `k3_q3m_lean_kvq8`: 96 VGPRs, 0 spilled, 31 spilled scalars, private segment 0, 15,748 bytes of LDS;
`k3_q3m_lean`: 96, 0, 53, 15,748 (the chunk's V scales use the scores' spare LDS). The Q8 attention alone made it 54.
Changes to the attention op did not remove that one scalar: the error word's address made in the op, 32-bit row offsets,
DPP or shuffle reductions, and selects or shifts for the half-dword parity were all measured.
The module therefore defines `K3_Q3M_ADD_RMS_PIN`, the dense Q8 objects' pinning applied to another op.
`OP_OWN_ADD_RMS16` takes its row length and slot count through an empty `asm volatile` scalar operand, so the op
rebuilds them instead of keeping derived values live across the segment loop. No arithmetic changes.

Other pins measured on the Q8 object, k3_run's spilled scalars:

| Pin | Spilled scalars |
|---|---|
| none | 54 |
| `OP_OWN_ADD_RMS16`'s N and slot count (shipped) | 31 |
| `OP_OWN_ATTS`'s kv heads, splits and heads | 45 |
| `OP_OWN_MV`'s rows, row bound and expert rows | 51 |
| `OP_ROUTER`'s experts and k | 53 |
| add_rms16 + atts | 19 |
| add_rms16 + atts + router + `OP_F32_MV`'s rows | 12 |

VGPRs stayed at 96, with 0 VGPR spills, in every row.
The same add_rms16 pin on the F16 object gives 33, measured but not applied: `k3_q3m_lean` keeps its code. So, like for
like, the Q8 attention adds no spilled scalar. The speed A/B below compares the shipped F16 object (53) with the Q8
object (31), so the pin is part of the difference it measures. The prefill kernels: `q3p_qk_rope_q8` 29 VGPRs,
`q3p_attn_q8` 226 (F16: 25 and 223), no spills.

Gates (a non-display card first):

```bash
tools/guard.sh LOG 600 python -I tests/engine/test_q3m_kvq8_atts.py                    # the op alone: rows, reads, grids, refusals
tools/guard.sh LOG 1800 python -I tests/engine/test_q3moe_one.py GGUF --ref-cache R.npz --kv q8
tools/guard.sh LOG 900 python -I tests/prefill/test_q3m_prefill.py GGUF --ref REF.npz --kv q8
tools/guard.sh LOG 900 python -I tools/gates/q3moe_gmodel.py GGUF --ref REF128.npz --save-logits F16.bin       # the F16 run
tools/guard.sh LOG 900 python -I tools/gates/q3moe_gmodel.py GGUF --ref REF128.npz --kv q8 --llama LLQ8_128.bin --llama-kv q8_0 \
    --against F16.bin --against-llama LL128.bin
tools/guard.sh LOG 1800 python -I tools/gates/q3moe_depth_gmodel.py GGUF --ref REF32.npz --kv q8 --llama LLQ8.bin --llama-kv q8_0 \
    --against F16.bin --against-llama LL.bin
```

Speed (decode and prefill; the two caches do not both fit one card at depth, so alternate runs, F16 then Q8, twice):

```bash
tools/guard.sh LOG 900 python -I benchmarks/q3moe_depth.py GGUF --depths 8,4096,16384,40900 [--kv q8]
tools/guard.sh LOG 900 python -I benchmarks/q3moe_prefill.py GGUF --lengths 4096,32768 --chunk 2048 [--kv q8]
```

The row checks compare against the F16 row kernel (`q3p_qk_rope`: OP_OWN_ATTS's norm + RoPE statements) on the same
q | k | v. `test_q3m_kvq8_atts.py` also compares the output with float64 attention over the decoded rows at positions
across one and many splits. The bound is per element: 2 fp16 ulps of the reference plus 1e-5 x max |V|, the dense line's
bound plus the output's fp16 rounding. A CPU fp32 emulation of the op's summation order on the test's inputs stays
within one ulp + 1e-7 x max |V|. Faults injected into it (a neighbouring block's V scale, chunk 1 shifted by a row, one
V row from its neighbour) fail the bound; a 4e-3 x max |V| bound passed all three. It also runs fixed fp16 edge blocks (+-65,504, exact ties, a scale that rounds to zero, zeros),
checks `q3p_qk_rope_q8`'s rows and refusal, and `q3p_attn_q8` bit for bit against `q3p_attn` over fp16(decoded rows). The G-model gates score the Q8 run against the fp32 reference with the usual rule against
the `--llama` rows given: llama.cpp's q8_0-KV rows (`kv q8_0` in gg_golden's job file, `--llama-kv q8_0`) are the like
for like bound. `--save-logits` on the F16 run and `--against` on the Q8 run gate the KLD and top-1 agreement between
the two caches (`tools/gates/kv_against.py`). The bar is the floor rule's shape: our F16 to Q8 mean KLD at most
llama.cpp's own F16-KV to q8_0-KV mean KLD on the same tokens (`--against-llama` names its F16-KV rows), and top-1
agreement at most one point below its own. Explicit `--against-kld` / `--against-top1` bars exist for rows without
llama.cpp references. An `--against` with no bound is refused. CPU: `tests/engine/test_q3moe_kvq8_cpu.py`. None of the
device gates has run yet.
