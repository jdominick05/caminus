# The training stack (dormant since the 10-31 inference-first pivot, D-279)

The repository began as a ground-up training stack for small hybrid models (S1 155M, an N-track 2B) on two RDNA3 GPUs
(gfx1100: 7900 XTX + 7900 XT), in C++ and HIP. It has been dormant since the project became the Caminus inference engine. The
S1 model, its trainer, data and design notes live in a private research repository; the tools that drove them are not part of
this repository. What remains here:

- `kernels/*.hip` (actq, attn, core, fuse, gdn, gemm, headloss, wquant, the S1 K3 engine `k3*.hip`): device code only,
  `extern "C"` kernels. `./build.sh` compiles each to `build/<name>.hsaco` (gfx1100 code object v5) with TheRock clang.
- `python/caminus/device/hip.py`: loads code objects through the HIP runtime PyTorch already loaded (one runtime per process) and
  launches on torch's streams. The inference engine uses the same loader.
- `python/caminus/{actq,fuse,gdn,gemm,headloss,linear,qlinear,wquant}.py` and their tests (`tests/test_gemm*.py`,
  `test_linear.py`, `test_gdn*.py`, `test_block_entry.py`, `test_silu_mul_actq.py`, `test_k3_engine.py`, `test_k3_matvec.py`):
  the kernels against PyTorch / fla references on the S1 shapes.
- `runtime/`: the C++ runtime (no PyTorch in the process). It links torch's bundled `libamdhip64` and `librocblas` by path
  (patchelf records torch's unversioned names), so its GEMMs are the binaries training ran: one `rocblas_gemm_ex` call
  reproduces `F.linear` bit for bit. build.sh builds its tests (`build/test_*`); they need golden tensors from the private
  trainer to run.

The Python modules ship in the `caminus` package; of the kernels, only `gemm` (its WMMA BF16 GEMMs also serve the batched
prefill) is in the package's runtime set.
