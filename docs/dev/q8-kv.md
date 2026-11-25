# Own Q8 KV cache path

Qwen3-30B-A3B's Q8 KV cache is served (`caminus serve --kv q8`, one card or two: [below](#serving-qwen3-30b-a3b-with---kv-q8)).
The dense line's is staged: the own-code dense line has separate `*_kvq8` persistent objects. They keep the F16 objects
and the weight-format selection unchanged. `Dense35Own(..., q8=True)` is an internal bring-up path: `caminus serve --kv
q8` refuses the dense models at startup, before any card is picked. The panel offers the KV cache choice only on the
models whose serve modes list it (`kv` in `caminus models --json`), so it never passes `--kv q8` to a dense model or the
Coder; started by hand with it, they refuse to start (nothing is ignored silently). Batched prefill refuses a Q8 engine until its matching implementation
lands. Device correctness, model accuracy and speed are not yet verified for these objects.

## Objects

| Object | Built from | Use |
|---|---|---|
| `dec_q35_swift_kvq8` | `dec_q35_swift.hip` | Swift decode |
| `dec_q35_swift_ver_kvq8` | `dec_q35_swift.hip` + verify defines | Swift verify, unpacked PART |
| `dec_q35_tcap_kvq8` | `dec_q35_tcap.hip` | ThinkingCap decode |
| `dec_q35_tcap_ver_kvq8` | `dec_q35_tcap.hip` + verify defines | ThinkingCap verify, unpacked PART |
| `dec_q35_gsq_kvq8` | `dec_q35_gsq.hip` | plain GSQ decode (its verify is separate work) |

The GSQ line's verify pass is two objects' launches (`dec_q35_gsq_ver`, `dec_q35_gsq_ver_m`); neither has a Q8 build,
so `Dense35Verify` refuses a Q8 GSQ engine. Each launch of a pass has its own error word; the pass checks every
launch's word before any bound or timeout handling.

`k3.Q35_KV_Q8_BASE` maps each object to its F16 peer; the module families and `MODULE_CAPS` follow the peer.
One card only: the two-card Q8 objects wait for their own fault investigation.

## Wire format and rotations

Each row holds K then V, each in 32-element blocks of a half scale and 32 signed bytes (34 bytes per block).
`kernels/kv_codec_ops.h` owns the codec; `caminus.ref_kv` defines its CPU reference. The scale calculation and rounding
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
recovery or park; exit 75). No served option selects the dense line's Q8 objects.

## Memory

The target cache is a byte tensor with `max_ctx + spec_T - 1` physical rows per attention layer (`Dense35Own.kv_rows`).
Projections count that exact storage, the split-attention scratch and the four-byte error word.
With `q8=True` only, `Dense35Own.kv_format_id` names the format (`q8_0-v1:qk-hadamard256:v-hadamard64-v1`); the prompt
cache's engine key carries any engine's `kv_format_id` (as for Qwen3-30B-A3B below). An engine holds a Q8 cache when its
id is one of `caminus.arch.KV_Q8_FORMATS` (`arch.kv_q8_built`: serve's startup check and `/props` read it), so an F16
engine that names its own format still reads as F16. The checkpoint layout already divides by `kv_rows`.

`k3_run` of each Q8 object beside its F16 object, from one build of this tree (`llvm-readelf --notes` fields;
VGPR spills 0 and private segment 0 on all ten):

| F16 object | VGPRs | SGPR spills | LDS | Q8 object | VGPRs | SGPR spills | LDS |
|---|---|---|---|---|---|---|---|
| `dec_q35_swift` | 86 | 62 | 3376 | `dec_q35_swift_kvq8` | 86 | 53 | 4400 |
| `dec_q35_swift_ver` | 93 | 88 | 6464 | `dec_q35_swift_ver_kvq8` | 93 | 76 | 5168 |
| `dec_q35_tcap` | 84 | 58 | 14384 | `dec_q35_tcap_kvq8` | 83 | 48 | 15408 |
| `dec_q35_tcap_ver` | 96 | 74 | 16704 | `dec_q35_tcap_ver_kvq8` | 96 | 67 | 15408 |
| `dec_q35_gsq` | 86 | 65 | 20528 | `dec_q35_gsq_kvq8` | 85 | 54 | 21552 |

Decode objects add the Hadamard op's 1024-byte buffer. Verify objects add it too but drop the packed PART's scratch
(3220 to 900 bytes): 1296 bytes less in all. `objcheck.py check` against a build of the branch point: the 120 existing
objects keep their code and data sections; the five Q8 objects are the only new ones.

## Register pinning in the Q8 objects

`ops_q35_kvq8.h` defines `K3_Q35_GDN_LOCAL_BASE`, which only the Q8 objects see. It pins three values in the
Gated DeltaNet step and save ops with empty `asm volatile` statements: the head counts (`nk`, `H`) as scalars and the
state base address as a vector value. The compiler then rebuilds them inside each op instead of keeping them live
across the persistent loop. Without it (the same tree, the define removed):

| Q8 object | VGPRs | VGPR spills | SGPR spills |
|---|---|---|---|
| `dec_q35_swift_kvq8` | 91 | 0 | 67 |
| `dec_q35_swift_ver_kvq8` | 96 | 0 | 94 |
| `dec_q35_tcap_kvq8` | 88 | 0 | 63 |
| `dec_q35_tcap_ver_kvq8` | 96 | 2 (16 B scratch) | 84 |
| `dec_q35_gsq_kvq8` | 90 | 0 | 71 |

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

## Qwen3.6-35B-A3B (one card)

`Moe35Own(..., q8=True)` stores the KV cache as Q8_0 rows on `dec_q35m_kvq8` (`k3.Q35MOE_KV_Q8_OF`), with the dense
line's format and rotations above (`kv_format_id` `q8_0-v1:qk-hadamard256:v-hadamard64-v1`; eight query heads per KV
head). Its batched prefill `Q35MoEPrefill` follows the engine: the rotations run as `pf_gg_attn_q35m`'s `pf_hadamard`
(the decode op's butterflies), `kernels/objects.tsv pf_q35m_kvq8` (`kernels/pf_q35_swift.hip` and `kernels/pf_q35m.hip` under `Q35P_KV_Q8`) adds the
append (`q35p_kv_append_q8`: the decode's `own_kv_q8_encode`, one thread per block) and the tiled attention over the rows
(`q35mp_attn_q8`: each staged value fp16(code x scale), one rounding more than the decode's reader; at F16's speed, 48 TF
at 30K: docs/dev/yardsticks.md). A
refused block sets error bit 2 and marks the engine broken (`k3.InvalidKVEncoding`). `caminus serve --kv q8` takes the
batched prefill in either rope mode (YaRN: `pf_gg_attn_q35m_yarn`, `PF_QK_YARN`: the decode op's table and magnitude
scale; prefill-then-decode against the float64 YaRN x2 reference 3.37e-4 / 64 of 64, decode-only 4.76e-4 on the 0.4.0
release commit, both cards).

Gates (RX 7900 XT): `tests/prefill/test_q35m_prefill_ops.py` (the append bitwise the decode module's, the refusals);
prefill of positions 0 .. 63 then decode of 64 .. 127 against the float64 reference (`tests/prefill/test_q35_prefill.py
--kv q8`): mean KLD 3.41e-4, top-1 64/64, decode-only on the same rows 2.69e-4, 64/64 (F16: 3.43e-4; 0.4.0, both cards;
the speeds: [benchmarks](benchmarks.md#qwen36-35b-a3b-gsq-hybrid-own-code-one-card)); chunk
geometry and prompt-cache stops change no logit and no cache byte; 16,384 tokens from empty in 3.37 s (4,856 tok/s; F16
3.30 s). `--yarn 2` runs the same test in the YaRN mode against a `tools/reference/q35_ref.py --yarn 2` reference; a
`q35_ref.py --last 64` reference of N tokens with `--from N-64` scores the hand-off at depth.

## Qwen3-30B-A3B (one card)

`Q3MoEOne(..., q8=True)` stores the KV cache as Q8_0 rows on `dec_q3m_kvq8` (`k3.Q3M_KV_Q8_BASE` maps it to
`dec_q3m`; the same ops, pairs and lean dispatch check). Its batched prefill `Q3MPrefill` follows the engine:
`q3p_qk_rope_q8` and `q3mp_attn_q8` from `pf_q3m_kvq8` write and read the rows, every other kernel stays on
`pf_q3m`. Served: `caminus serve --kv q8` (or `caminus start --kv q8`) sets `opt['kv']`, which the registry's
projection and build read (`Arch.kv_check`, `arch.kv_q8`); both objects are in the installed runtime set. The long-context mode
(YaRN) picks the Q8 rows by itself on one card where only they hold its context (no `--kv` given), on
`dec_q3m_kvq8_yarn` and `pf_q3m_kvq8_yarn`.

Format: the same wire codec and row layout as the dense line (`kernels/kv_codec_ops.h`, `caminus.ref_kv`), with K
rotated (`Q3MoEOne.kv_format_id` = `q8_0-v1:qk-hadamard128:f16-rounded-v1`). Each block encodes the fp16-rounded values
the F16 object stores, K after the head's orthonormal Walsh-Hadamard transform: a Q8 row is
`encode_q8_kv(hadamard128(K), V)` of the F16 row's K and V (`caminus.ref_kv.hadamard128`); the gates below check exactly
that. q gets the same transform after its RoPE, so the scores are q . k in exact arithmetic ((H q) . (H k) = q . k for
an orthonormal H). The transform is `kernels/kv_rot128.h`, the dense line's butterflies in its order (h = 1, 2 .. 64,
then x 1/sqrt(128) rounded to fp32: `0x1.6a09e6p-4`, one rounding more than the dense line's power-of-two scales). A
wave's lane holds a head's dims l, l + 32, l + 64, l + 96; the stages h = 1 .. 16 are shuffles, 32 and 64 stay in
registers, and every wave of a head computes all four quarters: no LDS and no barrier added (the decode op's 15,748 bytes
are what eight workgroups per WGP leave room for). The decode op rotates q per head in its RoPE step and the position's
own fp16-rounded k before its encode; the prefill's `q3p_qk_rope_q8` writes rotated K rows and writes q16 as
fp16(H fp16(q)), the F16 kernel's q16 rotated (q16 has no other reader), so `q3mp_attn_q8` is unchanged and
rotation-blind. The engine key carries the format id, so the unrotated rows of the first draft (never merged) never
restore into this engine.

Why K and not V: a CPU float64 model of this model on the 128-token reference prompt (mean KLD against the float64
reference, as the G-model gate scores; 21 valid draws) put the unrotated codec at 3.4e-3 .. 5.0e-3 (the device measured 5.40e-3), K and Q rotated by Hadamard128 at
1.65e-3 .. 3.0e-3 (mean 2.09e-3), and V rotated by Hadamard64 as well (llama.cpp's recipe) at 1.49e-3 .. 2.78e-3 (mean
1.87e-3); each layer's attention-output relative error fell to 0.60x (K and Q) and 0.55x (with V) on all 48 layers. K's
encode was most of the error (K alone 4.49e-3, V alone 2.15e-3). V's further 10 % sits inside the draws' spread and needs
the output rotated back (o = H^T sum p v') in the decode op's merge and in the prefill attention's WMMA output: not
taken. llama.cpp's q8_0 KV on the same rows: 3.68e-3 rotated, 3.22e-3 unrotated.

The writers are `own_kv_q8_encode_wave` (the codec's statements with one block per wave and the amax as a wave
reduction, exact for the finite blocks that are stored). The position's own K / V that the attention keeps in LDS are the
decoded values, so the current token reads what every later token reads. The prefill's attention rounds the decoded
values to fp16 for its WMMA operands, one rounding more than the decode reader; the model gate at depth judges it.

Refusals: a non-finite value (an fp16 overflow included) refuses its block. In K, a non-finite input was already the
whole head's (the k norm), and the transform also spreads one RoPE'd value that overflows fp16 over its head: all four K
blocks of that kv head are refused (`tests/engine/test_q3m_kvq8_atts.py` checks it in decode: one value past fp16 in
blocks 0 / 2 refuses blocks 1 / 3 too). Decode sets bit 2 in the int32 word after
Sync (`Engine.kv_error`, the dense line's word; `own_op_atts` takes `sy` for it), the prefill in its own `kverr` word.
The word sits right after the status words, so a pass's one asynchronous status copy (`Engine.status_words`, 9 words
on a Q8 engine) brings it: no extra read back per token. `_pass_wait` checks it before the bound and timeout paths (they
zero Sync), `Engine.run(check=True)` likewise from its one status read, `step` marks the engine too, and the prefill's
settle checks its word before the routing flags. The engine stays broken across `reset()` and raises
`InvalidKVEncoding`. `Q3MoEVerify` refuses a Q8 engine, the two-card engine has its own Q8 build (below), and every
registry entry but the two qwen3moe ones refuses `opt['kv'] = 'q8'` (`Arch.kv_check`) instead of building its F16 cache.

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

`k3_run` of `dec_q3m_kvq8`: 96 VGPRs, 0 spilled, 19 spilled scalars, private segment 0, 15,748 bytes of LDS;
`dec_q3m`: 96, 0, 53, 15,748 (the chunk's V scales use the scores' spare LDS). The rotation keeps the LDS and the
VGPRs; with the quarter index as a scalar (`wave & 3`; written `i >> 5` it spilled one more) it leaves 31 spilled
scalars, and `OP_OWN_ATTS`'s own pin (its kv heads, splits and heads through an empty `asm volatile` scalar operand,
under `K3_Q3M_KV_Q8` only) takes them to 19. Before the rotation, the unpinned Q8 attention alone made it 54.
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

VGPRs stayed at 96, with 0 VGPR spills, in every row (measured before the rotation; with it: add_rms16 31, add_rms16 +
atts 19, the shipped pair).
The same add_rms16 pin on the F16 object gives 33, measured but not applied: `dec_q3m` keeps its code. So, like for
like, the Q8 attention adds no spilled scalar. The speed A/B below compares the shipped F16 object (53) with the Q8
object (19), so the pins could be part of the difference it measures; the control below shows they are not.

Speed (decode, engine level, `benchmarks/q3moe_depth.py` on one RX 7900 XTX, tok/s at positions 8 / 4,096 / 16,384 /
40,900, three interleaved rounds, 2026-11-16):

| Object | 8 | 4,096 | 16,384 | 40,900 |
|---|---|---|---|---|
| `dec_q3m` (F16, shipped) | 246.4 | 210.5 | 159.3 | 110.0 |
| the F16 object with the same scalar pins (a measurement build) | 247.2 | 211.7 | 159.4 | 109.9 |
| `dec_q3m_kvq8` (Q8) | 239.2 | 210.5 | 164.1 | 117.0 |

Q8 against the shipped F16 object: 0.97x at 8, 1.00x at 4,096, 1.03x at 16,384, 1.06x at 40,900. The pinned F16 build
gave logits bit for bit the shipped object's and the same speed (at most 1.2 tok/s apart, at 4,096 positions), so the
gain at depth is the format's (half the KV bytes read per position), not the pins'; at 8 positions Q8 pays its encode,
rotation and decode with nothing to save yet. The pins were not shipped on the F16 object: it keeps its code.

The prefill kernels: `q3p_qk_rope_q8` 29 VGPRs, `q3mp_attn_q8` 239 (F16: 25 and 227), no spills; the rotation left the
rope kernel's count as it was (it was measured on the earlier attention, `q3p_attn_q8` 226 / 223).

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
q | k | v, its K halves through `ref_kv.hadamard128` before `encode_q8_kv`. `test_q3m_kvq8_atts.py` also compares the
output with float64 attention of the float64-rotated q over the decoded rows at positions across one and many splits,
checks the prefill's q16 bit for bit against fp16(hadamard128(the F16 kernel's q16)), and a NaN in k refusing its head's
four K blocks. The bound is per element: 2 fp16 ulps of the reference plus 1e-5 x max |V|, the dense line's
bound plus the output's fp16 rounding. A CPU fp32 emulation of the op's summation order on the test's inputs stays
within one ulp + 1e-7 x max |V|. Faults injected into it (a neighbouring block's V scale, chunk 1 shifted by a row, one
V row from its neighbour) fail the bound; a 4e-3 x max |V| bound passed all three. It also runs fixed fp16 edge blocks (+-65,504, exact ties, a scale that rounds to zero, zeros),
checks `q3p_qk_rope_q8`'s rows and refusal, and `q3mp_attn_q8` bit for bit against `q3mp_attn` over fp16(decoded rows). The G-model gates score the Q8 run against the fp32 reference with the usual rule against
the `--llama` rows given: llama.cpp's q8_0-KV rows (`kv q8_0` in gg_golden's job file, `--llama-kv q8_0`) are the like
for like bound. `--save-logits` on the F16 run and `--against` on the Q8 run gate the KLD and top-1 agreement between
the two caches (`tools/gates/kv_against.py`). The bar is the floor rule's shape: our F16 to Q8 mean KLD at most
llama.cpp's own F16-KV to q8_0-KV mean KLD on the same tokens (`--against-llama` names its F16-KV rows), and top-1
agreement at most one point or one row (whichever is more) below its own: on 64 rows one row is 1.56 points, and a bar
finer than one row is below the measurement's resolution. Explicit `--against-kld` / `--against-top1` bars exist for rows without
llama.cpp references. An `--against` with no bound is refused. `test_q3moe_one.py --kv q8` gates its mean KLD against the fp32 reference
below 2.08e-3, the unrotated draft's on that test (1e-3 is the F16 cache's bar). CPU: `tests/engine/test_q3moe_kvq8_cpu.py`, which also checks the
rotation: H128 / sqrt(128) orthonormal, (H q) . (H k) = q . k on float64 vectors, `hadamard128` within fp32 rounding of
it with the kernel header's scale literal, a Python lane-by-lane transcription of the kernel's statements bit for bit
`hadamard128` (it checks the reference, not the compiled code: the device rows checks do that), and the format id's key apart from the unrotated draft's.
Both RoPE sites round each fma to fp32 and then to fp16 through an opaque copy (`own_kv_f16r`), as the F16 objects do:
a plain `(_Float16)fmaf(...)` let the backend fuse it into one `v_fma_mix*_f16` (one rounding of the exact value), which
disagrees with the two-step rounding on about 6e-5 of values, and the transform spreads each such ulp over the head. The rotated objects passed on a device on 2026-11-16: the op test, the one-card decode and the prefill on the XT, and on
the XTX the G-model gates: mean KLD against the reference 1.15e-3 (128 rows), 3.66e-3 (16,384), 2.35e-3 (40,896), and
against our own F16 cache 1.19e-3 / 3.57e-3 / 1.87e-3, each below llama.cpp's own F16 to q8_0 cost (4.17e-3 / 6.63e-3 /
4.24e-3); top-1 agreement with our F16 cache 100 % / 93.8 % (60 of 64 rows) / 100 %, llama.cpp's with its own F16 cache
97.4 % / 95.3 % / 95.3 % (at 16,384 one row behind it: within the one-row bar). The unrotated draft passed the op test and the rows checks on the XT and both depth gates on the XTX (16,384 and
40,896 positions), and failed the 128-token G-model gate: mean KLD 5.40e-3 against float64 (llama.cpp's q8_0 KV
3.68e-3) and 5.53e-3 against our F16 cache (llama.cpp's own F16 to q8_0 cost 4.17e-3); the rotation is the answer to
that row.

## Qwen3-30B-A3B (two cards)

`Q3MoETP(..., q8=True)` stores each card's KV cache (its 2 of the 4 kv heads) as Q8_0 rows on `dec_q3m_tp_kvq8`:
`kernels/objects.tsv dec_q3m_tp` (the push-only `OP_XCHG`, `OP_OWN_SLOT_SUM`) with `K3_Q3M_KV_Q8` and `K3_Q3M_ADD_RMS_PIN`,
the one-card Q8 module's two defines. `k3.Q3M_KV_Q8_BASE` maps it to `dec_q3m_tp`, `k3.Q3M_TP_OF` maps
`dec_q3m_kvq8` to it, so it takes the two-card modules' ops, pairs, push-only check and 128 attention splits, and
the Q8 modules' error word and byte cache. Its batched prefill `Q3MPrefillTP` runs the one-card Q8 kernels
(`pf_q3m_kvq8`'s `q3p_qk_rope_q8` and `q3mp_attn_q8`, both of which take the kv head count) at each card's local
shapes; there is no new prefill object. Served: `caminus serve --cards 2 --kv q8`; the registry's `qwen3moe-tp` entry
takes `opt['kv'] = 'q8'` (`kv_formats`), so its projection, checkpoint slots and build follow it, and the module (with
`pf_q3m_kvq8`) is in the installed runtime set.

Format: the one-card engine's (`q8_0-v1:qk-hadamard128:f16-rounded-v1`, the engine key's `kv`), over each card's own
kv heads: a card's row is `encode_q8_kv(hadamard128(K), V)` of the halves the F16 two-card engine stores for its heads,
544 bytes per position and layer instead of 1,024. The attention is the one-card Q8 op's text at the two-card cap of
128 splits per kv head (`ATS_MAXS` 128 under `K3_Q3M_TP`). The chunk's V scales sit in the scores' spare LDS
(`sc[512 .. 768)`) only while the chunks run; the 128-split merge's m | l rows reuse q + scores (`qs[0 .. 2048)`) only
after the last chunk's barrier, so they never overlap in time. LDS stays at the F16 two-card object's 15,752 bytes.

Refusals: each card has its error word after Sync and the prefill its `kverr` word per card. Both cards' words come back
in each card's one status copy (`Engine.status_words`, 9 words per card on a Q8 engine), and `_pass_wait` reads them
before the bound and bail paths (those zero Sync). A set word raises `InvalidKVEncoding` naming the card, and the
engine stays broken across `reset()` (`_broken_error`; non-finite logits still mark it `NonFiniteLogits`). The prefill's
settle reads both cards' `kverr` words before the routing flags. On a bailed pass the segments after the expired wait
still run, on the partial a previous exchange left in the slot (finite values), and the retry rewrites the position's
rows. A refused block during a bail is still terminal, as on one card; the forced-bail run of `test_q3moe_tp.py --kv q8`
checks that bails alone never set the word.

Memory, and `max_ctx auto` for an RX 7900 XTX (display) + RX 7900 XT pair, nothing else on either card (the one-card
table's inputs; it reproduces that table). The engine refuses more than the file's 40,960 positions, so the uncapped
column is the projection only; it is what YaRN to 131,072 positions would need:

| Usage | F16 | Q8 | Without the file's 40,960 cap (F16 / Q8) |
|---|---|---|---|
| both idle | 40,960 | 40,960 | 212,992 / 405,504 |
| 1 GiB in use on the XTX | 40,960 | 40,960 | 192,512 / 364,544 |

Card 0 limits both. It holds 512 of each expert's 768 rows and the larger share of the head. On two cards, F16 already
covers 131,072 positions in this projection, and Q8 leaves about twice that.

`k3_run` of `dec_q3m_tp_kvq8`: 96 VGPRs, 0 spilled, 33 spilled scalars, private segment 64 bytes, 15,752 bytes of LDS;
`dec_q3m_tp`: 96, 0, 66, 64, 15,752. The pins are the one-card Q8 object's, and no others were added.

Gates (the op on the XT alone first, then both cards; `F16-TP.bin` is a two-card F16 run's saved logits, so the
`--against` comparison is two cards against two cards):

```bash
HIP_VISIBLE_DEVICES=<XT> tools/guard.sh LOG 900 python -I tests/engine/test_q3m_kvq8_atts.py --tp   # the op on the two-card module
HIP_VISIBLE_DEVICES=<XT> tools/guard.sh LOG 300 python -I tests/engine/test_q3m_tp_probe.py --tiny --scratch DIR --kv q8
HIP_VISIBLE_DEVICES=<XT> tools/guard.sh LOG 900 python -I tests/engine/test_q3m_tp_probe.py GGUF --scratch DIR --kv q8   # its whole k3_run, one card
CAMINUS_UPLOAD_LOG=ULOG tools/guard.sh LOG 1800 python -I tests/engine/test_q3moe_tp.py GGUF --npos 96 --scratch DIR --kv q8
CAMINUS_UPLOAD_LOG=ULOG tools/guard.sh LOG 1800 python -I tests/prefill/test_q3m_prefill_tp.py GGUF --ref REF.npz --kv q8
CAMINUS_UPLOAD_LOG=ULOG tools/guard.sh LOG 900 python -I tools/gates/q3moe_gmodel.py GGUF --ref REF128.npz --cards 2 --save-logits F16-TP.bin
CAMINUS_UPLOAD_LOG=ULOG tools/guard.sh LOG 900 python -I tools/gates/q3moe_gmodel.py GGUF --ref REF128.npz --cards 2 --kv q8 \
    --llama LLQ8_128.bin --llama-kv q8_0 --against F16-TP.bin --against-llama LL128.bin
CAMINUS_UPLOAD_LOG=ULOG tools/guard.sh LOG 1800 python -I tools/gates/q3moe_depth_gmodel.py GGUF --ref REF32.npz --cards 2 --kv q8 \
    --llama LLQ8.bin --llama-kv q8_0 --against F16-TP-DEPTH.bin --against-llama LL.bin
```

`test_q3m_tp_probe.py --kv q8` replays the one-card Q8 engine's tape on the two-card Q8 module, on one card. Each step's
logits must be bitwise `dec_q3m_kvq8`'s, and the local-only exchange and the slot sum run alone. That is the first
launch of its whole `k3_run`. `test_q3m_kvq8_atts.py --tp` is the one-card op test at one card's shapes of the two-card engine (16 query and 2 kv
heads, the card's own `splits_for(2, 128)`), with positions up to 8,000 on an 8,192-row cache. It also requires that more
than 64 splits were active (4,096 .. 6,100), which exercises the merge's second round of m | l loads beside the V
scales. `test_q3moe_tp.py --kv q8` adds, in its reference run, each card's last-layer row at every position compared byte
for byte with `encode_q8_kv(hadamard128(K), V)` of the F16 row kernel's halves for that card's q | k | v: the one-card
Q8 rows' encoding of the same values. Its grid, served-form and forced-bail checks run on the Q8 module unchanged.
`test_q3m_prefill_tp.py --kv q8` checks the same thing for each card's last-layer prefilled rows. Its geometry checks
(one piece, serial, chunks of 16, stops) compare the byte caches. Those caches start as 0xFF bytes, a NaN half scale,
so "every row written" means every block decodes finite. Speed (alternate runs, F16 then Q8, twice):

```bash
CAMINUS_UPLOAD_LOG=ULOG tools/guard.sh LOG 900 python -I benchmarks/q3moe_depth.py GGUF --cards 2 --depths 8,4096,16384,40900 [--kv q8]
CAMINUS_UPLOAD_LOG=ULOG tools/guard.sh LOG 900 python -I benchmarks/q3moe_prefill.py GGUF --cards 2 --lengths 4096,32768 --chunk 2048 [--kv q8]
```

CPU: `tests/engine/test_q3moe_tp_kvq8_cpu.py`. It covers the module tables, the engine's caches, tapes and projection,
the refusals, a refused block on one card with a stalled wait on the other, the prefill's launches and its per-card
words, the engine key and checkpoint layout, and the registry.

Device results, 2026-11-16 (XTX + XT, upload log on):
- **The new object on the XT alone.** The two-card op test passes: more than 64 splits, 16/2 heads, positions to 8,000. The whole k3_run on one card is bitwise identical to `dec_q3m_kvq8`.
- **Both cards, tiny model and real file.**
  - Logits are bitwise identical at K3_PER_WGP 3, 6 and 7 over 4,200 passes.
  - Replicas are equal after every pass.
  - Forced bails give the reference's bytes.
  - Prefill chunk geometry is bitwise identical on the byte caches.
- **Q8 rows against the oracle.** Each card's last-layer rows are checked against `encode_q8_kv(hadamard128(K), V)`. V matches exactly. K is within one code or one scale LSB at the few positions where the oracle's RoPE differs by an ulp:
  - the oracle runs the prefill row kernel;
  - on the F16 engine, 27 of 8,400 card rows at 4,200 positions differ from it, all in K, by at most 2 ulp.

| Mean KLD against the reference (top-1) | 128 decoded | last 64 of 16,384 | last 64 of 40,896 |
|---|---|---|---|
| two cards, Q8 | 1.29e-3 (128) | 2.48e-3 (62) | 1.42e-3 (63) |
| two cards, Q8 against two-card F16 | 1.37e-3 (100 %) | 2.55e-3 (95.3 %) | 1.56e-3 (100 %) |
| llama.cpp q8_0: against the reference / its own F16 to q8_0 cost | 3.68e-3 / 4.17e-3 | 5.68e-3 / 6.63e-3 | 3.86e-3 / 4.24e-3 |

Speed on two cards, three interleaved rounds, medians:
- Decode, F16 then Q8: 215.4 / 214.8 tok/s at 8, 189.5 / 191.8 at 4,096, 161.1 / 160.0 at 16,384, and 124.6 / 122.4 at 40,900.
- Prefill: 7,072 / 7,098 at 4,096 and 3,558 / 3,543 at 32,768.

On two cards, Q8 buys memory, not speed. Each card already reads only its own two kv heads' cache, so the cache is a smaller share of each pass than on one card.

## Serving Qwen3-30B-A3B with `--kv q8`

`caminus serve --kv {f16,q8}` (default `f16`; `caminus start --kv`, `caminus fit --kv` and a config file's `kv` key take
the same values) sets `opt['kv']`. `serve.cli.recipe` calls `Arch.kv_check` before any placement: only the two qwen3moe
entries list `q8` in `kv_formats` (one card: `Q3MoEOne(q8=True)`, two: `Q3MoETP(q8=True)` with `Q3MPrefillTP`'s Q8 path),
and every other model is refused with a message naming the model that takes it. `f16` selects nothing on an entry without
the choice: the dense own-code line keeps its F16 cache, the GGUF engines (the Coder, the paused ones) their own 8-bit
q8_0 cache (`Arch.kv_fixed`). After the build, the server checks that the engine holds what was asked (a `kv_format_id`
in `arch.KV_Q8_FORMATS` exactly when `--kv q8`: `arch.kv_q8_built`) and reports what it holds in `/props`
(`caminus.kv_cache = {type, format}`) and `/caminus/status` (`server.kv_cache`), from `arch.kv_cache_of`; `caminus ps`
shows the type.

Parks: the engine key carries the format (`tp.kv`) and the module, so the prompt cache's disk store lists a park under
one format's key only: a server with the other format on the same cache directory is never offered it (the prompt is read
again) and leaves it on disk for a server of its own format. `caminus replay` builds a Q8 engine for a one-card bundle
whose key names the format. A bundle from a `--cards 2` server (Q8 or F16) is refused with a message: replay does not
rebuild the two-card entry's split and placement yet, so it would build the one-card engine and fail the key check.

Capacity: `max_ctx auto` and the placement read the Q8 projection with `--kv q8`. `caminus fit` reports both formats
every time (`max_ctx = {f16, q8}`): each through its own drafter keep, placement and room (`serve.cli.fit_report`), `q8`
null for an entry whose `kv_formats` lacks it or when no cards hold that format, so `--kv` never changes a figure.
`--kv` picks the format whose cards, `need_mib` and `room_mib` the report gives, the one a serve with those options
starts with; when no cards hold it the query fails (`ok: false`) and keeps the other format's figure (a refusal before
placement, such as `--kv q8` on a model without it, gives none). Refusing `--kv` in fit would break the panel, which
passes its advanced options to fit unchanged. The panel's model picker shows both figures (`panel/control.go` `fitRow`:
the Q8 figure, "none fits", or "not offered", also next to a refusal; for an entry with `kv_fixed`, the Coder, the
one figure of its own 8-bit cache, with no Q8 part). One RX 7900 XTX
(display), nothing else on it: 40,960 positions with either cache (the file's context_length caps both; the room holds
45,056 at F16 and 81,920 at Q8); with 1 GiB in use, F16 32,768 and Q8 40,960 (the table above; the same projection,
recomputed on this branch). The cap stays until YaRN is applied.

The three Q8 objects (`dec_q3m_kvq8`, `dec_q3m_tp_kvq8`, `pf_q3m_kvq8`) are in the installed runtime set
(`caminus.device.kernels.Q3_OWN`; `tests/packaging/test_runtime_kernels.py` selects them through the real module
property and prefill constructors).

CPU: `tests/engine/test_q3moe_kvq8_cpu.py` and `tests/engine/test_q3moe_tp_kvq8_cpu.py` run the server's park path on
the Q8 engines' byte rows (`tests/engine/kv_park_rt.py`: a checkpoint slot, a park, an unpark after a restart, byte for
byte per card and layer; misdirected reads as negative controls) and the mismatch both ways (an F16 park not offered to a
Q8 server and the reverse, with the park's own format restoring it as the positive control);
`tests/test_context_fit.py`, `tests/test_arch.py` and `tests/cmd/test_manage.py` cover the switch, the refusals and the
projection.

Served gates (one card on the XTX, then both cards; each under the guard, each with a fresh scratch directory: the
prompts are fixed, so a park left by an earlier run would be restored and mask the result):

```bash
tools/guard.sh LOG 2400 python -I tools/serve_smoke1.py --model GGUF --scratch DIR1 --card 0 --kv q8 \
    --block-imports gguf,tokenizers --restart
CAMINUS_UPLOAD_LOG=ULOG tools/guard.sh LOG 2400 python -I tools/serve_smoke1.py --model GGUF --scratch DIR2 --card 0,1 \
    --cards 2 --kv q8 --block-imports gguf,tokenizers --restart
tools/guard.sh LOG 2400 python -I tools/serve_smoke1.py --model GGUF --scratch DIR3 --card 0 --restart --restart-kv q8
```

`--kv q8` checks `/props`' cache and the code objects the server loaded (`tools/serve_smoke1.py kv_objects`): a Q8
decode module and not its F16 base, and `pf_q3m_kvq8` beside `pf_q3m`; `--restart` restores the long prompt
from the Q8 park, and the restarted server's objects are checked the same way. The third run starts F16, parks, and
restarts with `--kv q8` on the same directories: the F16 parks must be on disk before the restart and still there after
it, the long prompt must come back from cold (the F16 park is not offered), and the second server's objects must be the
Q8 ones.

All three passed on 2026-11-16 (on this branch before its last rebase, which changed only the fit JSON, the panel row
and tests): one XTX with `--kv q8 --restart`, 20 / 20 (decode 228 tok/s short, 197 after a 6,587-token prompt read at
5,353 tok/s; the restart served the prompt from the Q8 park); XTX + XT with `--kv q8 --restart`, 20 / 20 (197 tok/s
short, 177 after the prompt read at 5,882 tok/s; the restart served it from the park); one XTX, F16 then `--restart-kv
q8`, 19 / 19 (the Q8 server read the long prompt cold, cache 0, and the F16 park was still on disk after it).
