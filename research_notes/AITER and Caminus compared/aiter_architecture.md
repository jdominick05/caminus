# AITER architecture and hardware support

## What is AITER and how is it packaged?

### Takeaway

AITER is a reusable AI operator and kernel library that framework authors integrate into larger inference and training systems. Its scope overlaps Caminus's computational layer, but the standard library does not replace a complete engine's model loading, scheduling, conversation state and API service. Its current backend breadth also makes a blanket “one standalone launch per operation” description inaccurate.

### Cited Findings

- Snapshot: upstream main resolved to `37da1dbeb305e868dda2e52d0d084a30fd71a99c`, whose commit timestamp is 2026-11-12 16:54:25 UTC; fetched at 2026-11-12 17:09 UTC. This is current source research, rather than a conclusion based on older MI300-only descriptions. — [Pinned commit](https://github.com/ROCm/aiter/commit/37da1dbeb305e868dda2e52d0d084a30fd71a99c)
- README describes the project as an optimized operator library with C++ and Python APIs, inference and training scope, and integration into custom frameworks. Its ecosystem table distinguishes vLLM/SGLang integrations from ROCm ATOM, which is an engine built on AITER. — [README introduction and ecosystem](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/README.md)
- Backends now extend beyond the top-level feature list's Triton/CK/ASM: contribution rules separately identify Triton/Gluon, HIP, ASM, CK, OPUS and FlyDSL. FlyDSL is used for GEMM and MoE; OPUS supplies lightweight HIP C++ templates, layouts and MFMA wrappers. — [README backend and FlyDSL sections](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/README.md); [OPUS developer README](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/csrc/include/opus/README.md#4-warp-gemm-and-tiled-mma)
- Python entrypoint imports torch directly. This rules out describing the ordinary `import aiter` path as PyTorch-free. The README separately describes a JAX bridge without a PyTorch dependency, and Triton internal rules require shared helpers/config loaders/kernel modules to work without torch. Those are distinct interfaces and do not establish that the entire package's ordinary Python API is framework-free. — [Python entrypoint lines 1–8](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/aiter/__init__.py#L1-L8); [Triton source layout and rules](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/aiter/ops/triton/README.md)
- JIT core defaults GPU_ARCHS to native, supports runtime compilation with a per-module build lock, loads extension modules and caches imported module objects. Ahead-of-time builds are also available through PREBUILD_KERNELS. A cold compilation cost is not equivalent to compiling on every token. — [JIT cache directory and architecture parsing](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/aiter/jit/core.py#L935-L995); [module cache](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/aiter/jit/core.py#L1143-L1153); [build function](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/aiter/jit/core.py#L1414-L1436); [setup prebuild handling](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/setup.py#L298)
- `compile_ops` exposes pybind and ctypes FFI forms; there are torch-excluded build options and custom compilation guards. Actual host overhead depends on selected operator, execution/capture path and integration; source inspection does not supply a fair numerical overhead comparison with K3. — [compile_ops](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/aiter/jit/core.py#L2322); [build torch exclusion](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/aiter/jit/core.py#L1547)
- Current setup pins FlyDSL 0.5.0 on Linux in normal mode, uses Python >=3.10, and lists pybind11, ninja, pandas, einops, psutil and packaging. Setup may install/replace Triton unless an appropriate system-Triton setting is used. These are packaging observations, not instructions to install into Caminus's environment. — [setup dependency and installation logic](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/setup.py#L27-L154); [install requirements](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/setup.py#L506-L543)
- `AITER_TRITON_ONLY=1` disables CK and HIP loading/JIT in the entrypoint; setup disables CK and kernel prebuild in this mode. This is a way to isolate a backend subset, not proof all Triton operators work on every architecture. — [entrypoint mode](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/aiter/__init__.py#L60-L101); [setup mode](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/setup.py#L38-L60)

### Inferences

- Treat AITER as a potential source of selected implementations, layouts, references and benchmark competitors. A wholesale replacement would mean changing how Caminus executes operations and owns memory/state, rather than replacing one Python package.
- Warm up and distinguish compile time from execution time in any eventual comparison. Test the actual call path, including CUDA/HIP graph capture where applicable, before assigning CPU launch overhead.
- Upstream kernels fused at a larger granularity must be compared at that same granularity. The separate operators researcher is examining AITER's persistent/fused implementations; this research does not establish that K3 is uniquely persistent among all AITER paths.

### Gaps

- No installation, build or GPU execution was performed. No numerical cold-start, warm dispatch, decoding or prefill comparison with Caminus is available.
- Upstream README “production” and headline performance characterizations are upstream claims; this research does not independently validate those workloads.

## Does AITER support Caminus's gfx1100 hardware?

### Takeaway

There is now explicit experimental RDNA3 support, including the gfx1100 family used by RX 7900 XTX/XT. The right conclusion is “operator- and backend-specific compatibility,” not “AITER cannot run on RDNA” and not “all AITER kernels are supported on a 7900.”

### Cited Findings

- Current hardware table labels MI300X/MI325X gfx942 fully supported, MI350/MI355X gfx950 supported, and AMD Pro W7900 gfx1100 experimental. It also lists experimental gfx1151 and gfx1201. Its footnote says Triton, most FlyDSL and most HIP kernels run on RDNA; most CK/ASM kernels remain CDNA-only. The W7900 entry describes an architecture family, not validation of Caminus's exact mixed-card setup. — [Supported hardware](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/README.md#supported-hardware)
- Core JIT's explicit allow-list includes gfx1100/1101/1102/1103, gfx1150/1151/1152/1153 and gfx1200/1201 alongside gfx90a/gfx94x/gfx950/gfx1250. Thus gfx1100 is no longer rejected at this top-level build gate. — [JIT architecture allow-list](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/aiter/jit/core.py#L960-L995)
- Build-target utilities explicitly map gfx1100 to 96 CUs with a Radeon RX 7900 XTX comment, and handle RDNA's WGP-vs-CU processor-count reporting. The gfx1100 static list only includes 96, whereas native detection and explicit CU settings have other paths. This is not evidence that the RX 7900 XT's 84-CU variant has equally complete tuned configuration coverage. — [Target/CU mappings](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/aiter/jit/utils/build_targets.py#L33-L92)
- Triton capability checks distinguish native FP8 from software emulation. gfx1100 is in the software-emulated list, whose comment states FP8 is converted to FP16 in software and FP16 WMMA is used. It is absent from native FP8, native FP4, Gluon availability and MX-scale-preshuffling capability predicates in this helper. Capability and optimization claims must therefore name the path. — [Architecture capabilities](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/aiter/ops/triton/utils/_triton/arch_info.py#L30-L79)
- OPUS GEMM runtime architecture enum/dispatch recognizes gfx942, gfx950 and gfx1250; a comment explicitly places gfx1100 in future work. This is direct source evidence that at least one major current GEMM backend is not a drop-in gfx1100 path. — [OPUS GEMM architecture dispatch](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/csrc/opus_gemm/include/opus_gemm_arch.cuh#L13-L57)
- A separate compilation helper still has a narrower architecture allow-list containing native, gfx90a, gfx940/941/942, gfx950, gfx1151 and gfx1250, but not gfx1100. Not every build path inherits the expanded core allow-list. — [cpp_itfs compile helper](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/csrc/cpp_itfs/utils.py#L230-L253)
- Current tree contains 49 gfx1100 config files. Examples include BF16/FP16 GEMM default and shape-specific tuning, A16W4, emulated-FP8 blockscale GEMM, unified attention and paged attention decode. Presence is useful evidence of intentional work, but does not prove the exact Caminus shapes or kernels passed. — [gfx1100 configs](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/aiter/ops/triton/configs/gfx1100); [A16W16 default](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/aiter/ops/triton/configs/gfx1100/triton/gemm/gemm_a16w16/DEFAULT.json)
- Main AITER CI workflow builds gfx942/gfx950 and runs standard tests on MI35X with MI300X added on main or by label; its multi-GPU job uses an MI350X 8-GPU runner. No RDNA runner is present in this inspected workflow. Do not generalize that finding to every private or external CI installation. — [CI architecture list](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/.github/workflows/aiter-test.yaml#L22-L25); [runner matrix](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/.github/workflows/aiter-test.yaml#L540-L557); [multi-GPU matrix](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/.github/workflows/aiter-test.yaml#L1139-L1146)

### Inferences

- The most promising evaluation path on the current machine is the source-supported Triton/HIP subset, selected per operator and format. CDNA code objects, MFMA assumptions, native FP8/FP4 paths and CDNA-only persistent kernels should not be treated as executable on RDNA3.
- AITER's low-precision native-Instinct marketing cannot be transferred to gfx1100. In particular, native FP8 arithmetic speedups cannot be assumed for a software-emulated FP8 path.
- Caminus's persistent decode tape might borrow algorithmic techniques from AITER without calling its public launch wrappers. Direct invocation of a separately compiled kernel is an integration boundary; porting its device routine into K3 is another task with instruction, register, synchronization and numerical implications.
- The 7900 XT/XTX pair and PCIe topology still require specific validation. Shared gfx1100 ISA does not establish the same tuned configurations, memory budget, communication behavior or latency as W7900/Instinct deployments.

### Gaps

- No exact per-operator gfx1100 support matrix was found. README's “most” claims need checking against each intended entrypoint.
- No direct comparison of AITER and Caminus on the same model, quantization, hardware, batch, context and output-quality target was found in this hardware investigation.
- MI200 gfx90a appears in build allow-lists but not the current README support table; do not label MI200 fully supported based only on a compiler target.

## What does recent RDNA integration evidence establish?

### Takeaway

RDNA enablement is active and has advanced beyond older issue reports. Dated user reports are useful compatibility warnings, while current code and merged framework integrations provide stronger evidence for what is available now.

### Cited Findings

- AITER issue #6574, opened 2026-10-15 and still open at inspection, reports tests on RX 7900 XT gfx1100 and Radeon AI PRO R9700 gfx1201, ROCm 7.2.1, torch 2.9.1+rocm7.2.1 and Triton 3.5.1, using AITER main 6c4d83c8a in Triton-only mode. The reporter records passing fused KDA decode, unified decode, flash KDA, chunk delta attention, topk and fused RMSNorm/add tests. This is reporter evidence, not tests run for this research. — [Issue #6574](https://github.com/ROCm/aiter/issues/6574)
- That report also records missing gfx1100 GEMM configs/import problems and a >700-second hang during software-emulated FP8 blockscale GEMM on gfx1100. No GPU test was attempted here; the hang is particularly relevant when deciding how conservatively to stage an eventual guarded non-display-card probe. — [Issue #6574](https://github.com/ROCm/aiter/issues/6574)
- Parts of the report's missing-config observation have aged: pinned main now has a gfx1100 A16W16 DEFAULT.json and shape-specific configurations, among others. No reply or closure establishes that every reported defect or hang is fixed. — [Current gfx1100 A16W16 config](https://github.com/ROCm/aiter/blob/37da1dbeb305e868dda2e52d0d084a30fd71a99c/aiter/ops/triton/configs/gfx1100/triton/gemm/gemm_a16w16/DEFAULT.json); [issue status](https://github.com/ROCm/aiter/issues/6574)
- AITER issue #4604 from 2026-08-06 describes patched W7900 vLLM use for Qwen3.6-27B W8A8 with Triton GEMM, GDN, causal conv and sampling, and originally reports that vLLM's integration did not expose RDNA3. An Aug 8 response welcomes RDNA support work and says machine/CI resources and maintenance are being scoped. — [Issue #4604](https://github.com/ROCm/aiter/issues/4604); [maintainer reply](https://github.com/ROCm/aiter/issues/4604#issuecomment-5224138748)
- The important current update is vLLM PR #52970, merged 2026-10-27 at 17:01 UTC. It enables AITER Triton RDNA3 paths, GDN and unified attention, and widens separate RDNA gates. Its test report is on 8 W7900 cards with a DeepSeek-V4 model and particular settings; it is not Caminus's workload. — [Merged vLLM #52970](https://github.com/vllm-project/vllm/pull/52970)
- A subsequent Oct 31 maintainer comment explicitly explains why the global all-AITER gate must not include RDNA3: that gate also permits unsupported CK and assembly paths. The fix belongs in the separate RDNA support gate. — [Maintainer clarification](https://github.com/ROCm/aiter/issues/4604#issuecomment-6539644216)

### Inferences

- Avoid repeating the old claim that vLLM never enables AITER on gfx1100. The merged Oct 27 work changes that conclusion.
- Even after framework-level enablement, exact low-bit model formats, state conventions and performance targets remain the real integration test for Caminus.
- A safe future experiment should pin the AITER SHA and backend, inspect required configs and source guards first, and run only under Caminus's GPU-window/guard workflow. Installation and dependency changes should be isolated from the current server environment.

### Gaps

- Upstream issue reports and merged test descriptions do not validate the user's installed ROCm/Triton versions or two-card topology.
- No verified fix/status for every item in #6574 was found. Current config presence should be reported separately from unresolved runtime correctness or hang concerns.

