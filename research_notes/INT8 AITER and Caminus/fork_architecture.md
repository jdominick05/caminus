# INT8 AITER fork lineage and hardware

## What fork is this and how does its lineage compare with upstream?

### Takeaway

This is a specialized AMD Instinct MI100 gfx908/CDNA1 downstream, with active low-level INT8 work and a companion vLLM engine. It is not an RX 7900, RDNA4 or Strix Halo fork. Current source is substantially broader than its stale README suggests.

### Cited Findings

- GitHub repository metadata identifies `curvedinf/int8-aiter` as a public fork of `ROCm/aiter`, default branch `main`, not archived. It was created 2026-06-07 and its latest push at inspection was 2026-11-12 16:29:54 UTC. — [Repository metadata](https://api.github.com/repos/curvedinf/int8-aiter)
- Pinned HEAD is `cd53c11c6d399cd79aee9500536dc5f8206f6ea3`, committed 2026-11-12 16:29:50 UTC. Research retrieval was 2026-11-12 approximately 17:18–17:23 UTC. The head commit changes fused SiLU/quant GEMM tile geometry and reports measurements on its own deployment, not Caminus. — [Pinned HEAD](https://github.com/curvedinf/int8-aiter/commit/cd53c11c6d399cd79aee9500536dc5f8206f6ea3)
- README explicitly targets a four-MI100 node, Qwen3.8-27B-class model, W8A8 GPTQ weight/per-token activation quantization, fp16 activations, DFlash2, tensor parallelism 4, CK GEMM, custom all-reduce and graph-captured vLLM decode. Its consumer is `curvedinf/int8-vllm`. — [Fork README](https://github.com/curvedinf/int8-aiter/blob/cd53c11c6d399cd79aee9500536dc5f8206f6ea3/README.md#serving-stack-this-fork-targets)
- Verified merge `83d97db5b830aee47b5641d96411d0534d85d867`, 2026-08-21, imports upstream parent `03624e356efb57a72c88776e2adc04ca31ca404c`. The merge description says it preserves fork-only per-token/head INT8 KV support, true INT8 QK dot, MI100 build compatibility and gfx908-specific tuning while taking upstream's other-architecture defaults. — [Upstream merge](https://github.com/curvedinf/int8-aiter/commit/83d97db5b830aee47b5641d96411d0534d85d867)
- Comparing that known upstream parent to pinned fork HEAD succeeds: 89 commits reachable ahead, zero behind that base, and 96 changed files. **These are counts against the known August 21 upstream parent, not ahead/behind counts against November upstream main.** — [Exact comparison](https://github.com/curvedinf/int8-aiter/compare/03624e356efb57a72c88776e2adc04ca31ca404c...cd53c11c6d399cd79aee9500536dc5f8206f6ea3)
- GitHub comparisons of current ROCm main against curvedinf main returned HTTP 404 “No common ancestor” through the connected API. This does not establish why GitHub could not produce a current comparison. Do not infer a force-push, independent root, current merge base or current behind count from it. The known-parent comparison above is the reproducible comparison available. — [Attempted upstream comparison](https://api.github.com/repos/ROCm/aiter/compare/main...curvedinf:main)
- README says fork changes touch about 16 files and describes its gfx908 build profile as an unimplemented plan with zero tuned gfx908 rows. Source contradicts those descriptions: PREBUILD_KERNELS=4 exists, the pruning/strict code exists, and the current tuning CSV contains 569 data lines containing gfx908. The CSV header is `gfx,cu_num,M,N,K,q_dtype_w,kernelId,splitK,us,kernelName,tflops,bw,errRatio`; the count is source inventory, not a successful retest of every entry. — [README](https://github.com/curvedinf/int8-aiter/blob/cd53c11c6d399cd79aee9500536dc5f8206f6ea3/README.md#gfx908-only-build-plan-in-progress); [implemented build profile](https://github.com/curvedinf/int8-aiter/blob/cd53c11c6d399cd79aee9500536dc5f8206f6ea3/setup.py#L267-L298); [tuned CSV](https://github.com/curvedinf/int8-aiter/blob/cd53c11c6d399cd79aee9500536dc5f8206f6ea3/aiter/configs/a8w8_tuned_gemm.csv)
- New source paths in the known-parent comparison include native packed IQ2/Q8/NVFP4/GS128 kernels and caches, INT8 MFMA primitives, native paged INT8 attention, fused RMSNorm/GS128 output quantization, fused SwiGLU/INT8 quantization and all-reduce correctness tests. Thus inherited upstream code must be separated from the 96-file changed set when describing the fork's contribution. — [Exact changed-file comparison](https://github.com/curvedinf/int8-aiter/compare/03624e356efb57a72c88776e2adc04ca31ca404c...cd53c11c6d399cd79aee9500536dc5f8206f6ea3)

### Inferences

- This fork is architecturally closer to Caminus than the upstream project's broad operator-library description might suggest: it optimizes an actual small-batch, quantized, speculative AMD serving workload, including model-format and state concerns. Its engine remains the companion vLLM fork rather than an independent full engine within AITER.
- The README is useful for project intent but unreliable for latest implemented scope, tuning coverage and build status. Current source and dated commit descriptions should drive the comparison.
- No claim that these downstream modifications have merged into current upstream was verified. The presence of generic attention, GDN, MoE or GEMM in both trees is not evidence that a particular MI100 implementation was upstreamed.

### Gaps

- Exact merge base/ahead/behind against current upstream main could not be established from the GitHub compare API response.
- No build, GPU run or author measurement was reproduced. Test and benchmark claims retain upstream-author provenance.

## What hardware and build assumptions are actually present?

### Takeaway

The new paths are explicitly CDNA1 INT8 implementations, and some emit gfx908 machine code unconditionally. Shared HIP/Triton tooling does not make those paths usable on gfx1100. An arch string substitution is not a valid port.

### Cited Findings

- Core JIT allow-list adds gfx908, while retaining inherited RDNA targets including gfx1100/1101/1102/1103, gfx115x and gfx120x. Build-target utilities register gfx908 with 120 CUs. These gates allow compilation requests; they do not prove every inherited kernel supports MI100 or that every fork kernel supports RDNA. — [JIT allow-list](https://github.com/curvedinf/int8-aiter/blob/cd53c11c6d399cd79aee9500536dc5f8206f6ea3/aiter/jit/core.py#L505-L529); [CU mapping](https://github.com/curvedinf/int8-aiter/blob/cd53c11c6d399cd79aee9500536dc5f8206f6ea3/aiter/jit/utils/build_targets.py#L33-L44)
- PREBUILD_KERNELS=4 insists `GPU_ARCHS` be exactly gfx908 and requires `AITER_MI100_MODULES` to name allowed modules. JIT additionally supports `AITER_JIT_ALLOWLIST` so a module excluded from deployment does not silently recompile on first use. — [Build profile](https://github.com/curvedinf/int8-aiter/blob/cd53c11c6d399cd79aee9500536dc5f8206f6ea3/setup.py#L267-L298); [JIT allow-list](https://github.com/curvedinf/int8-aiter/blob/cd53c11c6d399cd79aee9500536dc5f8206f6ea3/aiter/jit/core.py#L1155-L1178)
- CK generation supports instance IDs from `AITER_CK_INSTANCE_LIST` and dtype filters. When all targets are gfx908, it keeps INT8 A/B paths and both fp16 and bf16 epilogues; comments explicitly say the target serves fp16 but DFlash2 resolves bf16 from its checkpoint. Thus README's assertion that bf16 epilogues are dropped is stale. This does not establish native BF16 MFMA serving on MI100. — [Generator filtering](https://github.com/curvedinf/int8-aiter/blob/cd53c11c6d399cd79aee9500536dc5f8206f6ea3/csrc/ck_gemm_a8w8/gen_instances.py#L32-L91)
- `aiter/ops/int8_native.py` defaults `INT8_NATIVE_ARCH` to gfx908, runs hipcc on first use and caches by source/header/compiler/options/arch digest. The loaded module is a ctypes library. Compilation uses `-ffp-contract=off`, `-fno-strict-aliasing` and an offload target from that arch setting. — [Native loader](https://github.com/curvedinf/int8-aiter/blob/cd53c11c6d399cd79aee9500536dc5f8206f6ea3/aiter/ops/int8_native.py#L1-L81)
- Its fragment helper directly uses `__builtin_amdgcn_mfma_i32_16x16x16i8` with a documented gfx908 lane layout. The new small-M GEMM source is explicitly a gfx908/CDNA1 GS128 INT8 GEMM. These hardware instructions/layouts are not Caminus's RDNA3 WMMA implementation. — [INT8 MFMA primitive](https://github.com/curvedinf/int8-aiter/blob/cd53c11c6d399cd79aee9500536dc5f8206f6ea3/csrc/kernels/int8_mfma_lane.cuh#L1-L65); [small-M GEMM](https://github.com/curvedinf/int8-aiter/blob/cd53c11c6d399cd79aee9500536dc5f8206f6ea3/csrc/kernels/gemm_a8w8_m16_gfx908.hip)
- The separate packed-weight loader has literal `--offload-arch=gfx908` in its compiler command and selects `gemm_packed_g128_gfx908.hip`. This is stronger than merely an optimization hint; it produces machine code for another ISA target than RX 7900. — [Packed native loader](https://github.com/curvedinf/int8-aiter/blob/cd53c11c6d399cd79aee9500536dc5f8206f6ea3/aiter/ops/gemm_packed_native.py#L17-L52)
- Setup inherits normal AITER packaging and pins FlyDSL 0.3.1, whereas the upstream snapshot researched earlier pins 0.5.0. It lists Python >=3.10 and pybind11/ninja/pandas/einops/psutil/packaging dependencies; normal editable setup can install FlyDSL and select/install Triton. It is not a dependency-neutral overlay. — [Fork setup](https://github.com/curvedinf/int8-aiter/blob/cd53c11c6d399cd79aee9500536dc5f8206f6ea3/setup.py#L16-L25); [install behavior](https://github.com/curvedinf/int8-aiter/blob/cd53c11c6d399cd79aee9500536dc5f8206f6ea3/setup.py#L56-L140); [requirements](https://github.com/curvedinf/int8-aiter/blob/cd53c11c6d399cd79aee9500536dc5f8206f6ea3/setup.py#L513-L545)
- `scripts/gfx908_env.sh` pins GPU_ARCHS/PYTORCH_ROCM_ARCH to gfx908 and contains author-local checkout/JIT paths. These are example environment settings tied to the developer's MI100 checkout, not paths appropriate to source unchanged in Caminus. — [Environment script](https://github.com/curvedinf/int8-aiter/blob/cd53c11c6d399cd79aee9500536dc5f8206f6ea3/scripts/gfx908_env.sh)

### Inferences

- On RX 7900 XTX/XT, reuse primarily means studying and porting the algorithms, packed layouts, quantization contracts and fusion decisions. Native gfx908 kernels are not a drop-in performance option.
- Do not assume a generic Triton path provides a safe fallback for every wrapper. The separate operator research has identified native dispatch paths; every prospective entrypoint must be traced through its arch gate and defaults before execution.
- Any eventual experiment should isolate dependency changes and respect Caminus's GPU-window/guard requirements. No installation or GPU testing was needed for this research.

### Gaps

- No claim of supported RX7900, Strix Halo, RDNA4, arbitrary MI100 workloads, or every inherited upstream operator is justified by these build-target lists.
- Exact performance and capacity on a mixed XTX/XT PCIe pair remain unmeasured.

## Which concrete downstream commits establish the current scope and maturity?

### Takeaway

The fork is actively iterated through correctness failures, reversions and later fixes. It includes valuable implementation evidence, but recent commit-level validation should be read as author-reported and path-specific, not broad production certification.

### Cited Findings

- `7e0186f8` on August 25 implements the gfx908 build profile, instance pruning, strict guards and tuner dtype fixes. Later `fccd5c96` restores bf16 DFlash2 epilogue support. This chronology directly refutes the old build-plan-only narrative. — [Build implementation](https://github.com/curvedinf/int8-aiter/commit/7e0186f8934c5618fab0d1100a76a753f109a959); [DFlash2 dtype fix](https://github.com/curvedinf/int8-aiter/commit/fccd5c9672f4e7e75a67dad118ccfd16f90b3aa4)
- `55e06e3f` on October 15 adds packed IQ2, Q8 and NVFP4 GS128 INT8 kernels; subsequent commits add bounded dense/expert caches and cache-aware prefill work, including a reversion after an aggregate decode recheck. Source shows a development process with measured tradeoffs, rather than uniformly accepted optimizations. — [Packed-format addition](https://github.com/curvedinf/int8-aiter/commit/55e06e3fe2358a76689c103e66c60fdd480d03af); [cache-aware prefill reversion](https://github.com/curvedinf/int8-aiter/commit/15a3e419553d970169edbf386a4ff7166dcc2f2a)
- November 11 commit `0cfc9a99` explicitly records a failed serving gate and keeps a new native decode path opt-in. Later `2d5f5e0b` reports native INT8 decode validation while leaving KV rotation off by default. These are different milestones; neither should be omitted to imply all experiments passed from the outset. — [Failed gate](https://github.com/curvedinf/int8-aiter/commit/0cfc9a99426b1e2057edc0cd90969d25a941e73c); [follow-up validation](https://github.com/curvedinf/int8-aiter/commit/2d5f5e0b55325f0799538550b2a63722092deac5)
- November 12 commits evolve native small-M GEMM from a first version to larger N/split-K and multi-wave CTAs. HEAD then adjusts fused SiLU GEMM block geometry to address CTA concurrency. Performance numbers in these commit messages refer to its own MI100 setup and shapes; they cannot be compared directly with Caminus token latency. — [GEMM first version](https://github.com/curvedinf/int8-aiter/commit/2dbb5be4ee879a1fad091df258493fb459a19af8); [multi-wave version](https://github.com/curvedinf/int8-aiter/commit/2c4c312323e397cee08cb80284a31dab0a6c1ac4); [HEAD](https://github.com/curvedinf/int8-aiter/commit/cd53c11c6d399cd79aee9500536dc5f8206f6ea3)
- New native helper/loader files carry Apache-2.0 SPDX notices. Do not label every file in this MIT-rooted fork as MIT merely from the repository-level license badge. Any code reuse must preserve the particular file's notice and satisfy Caminus's third-party provenance rules. — [Native loader SPDX](https://github.com/curvedinf/int8-aiter/blob/cd53c11c6d399cd79aee9500536dc5f8206f6ea3/aiter/ops/int8_native.py#L1); [MFMA helper SPDX](https://github.com/curvedinf/int8-aiter/blob/cd53c11c6d399cd79aee9500536dc5f8206f6ea3/csrc/kernels/int8_mfma_lane.cuh#L1-L2)

### Inferences

- Useful lessons for Caminus include prioritizing actual decode/verify shapes, distinguishing scaling contracts, pruning compiled instances, testing graph-safe communication, and evaluating fusion across the whole served workload.
- Claims about gains should be framed as reasons to investigate a technique, not evidence the fork is faster than Caminus or portable without changes.

### Gaps

- Source and commit history reveal implementation intent and reported results, not independent reproduction.
- This investigation did not audit every correctness claim or each new kernel; dedicated operator and numerical analysis is covered by the other researchers.

