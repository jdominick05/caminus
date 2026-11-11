#!/usr/bin/env bash
# Build every kernels/*.hip into a gfx1100 code object (build/<name>.hsaco), then the C++ runtime tests.
# Code objects are runtime-neutral: PyTorch's HIP 7.2 runtime and TheRock's both load them via hipModuleLoadData.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
TR=${ROCM_PATH:-$HOME/rocm-therock/10.2.0a20261011}
ARCH=${ARCH:-gfx1100}
mkdir -p build
# k3_gguf.hip includes llama.cpp's ggml-cuda headers (vecdotq.cuh: the GGUF block formats) from the ignite-llama tree
IG=${IGNITE:-$HOME/src/ignite-llama}
GG=$IG/ggml
# provenance inputs for tools/stage_kernels.py (docs/dev/packaging.md): build/BUILD-INFO (the toolchain, the sources' commit,
# dirty flag and content hash, the ggml headers' commit) and build/BUILD-CMDS.tsv (each object's command as run below, with the
# toolchain and ignite paths normalized to <ROCM_PATH> / <IGNITE>). Recording changes no compile flag
: > build/BUILD-INFO; : > build/BUILD-CMDS.tsv
info() { printf '%s\t%s\n' "$1" "$2" >> build/BUILD-INFO; }
gitstate() {   # gitstate DIR PATHS...: "<commit> <dirty 0|1>" of the tracked PATHS, or "unknown unknown" outside a git tree
    local d=$1; shift
    if git -C "$d" rev-parse --git-dir > /dev/null 2>&1; then
        echo "$(git -C "$d" rev-parse HEAD) $([ -n "$(git -C "$d" status --porcelain --untracked-files=no -- "$@")" ] && echo 1 || echo 0)"
    else echo unknown unknown; fi
}
logcmd() {   # logcmd OUT COMMAND-TEXT
    local c=$2; c=${c//"$GG"/<IGNITE>/ggml}; c=${c//"$TR"/<ROCM_PATH>}
    printf '%s\t%s\n' "$(basename "$1")" "$c" >> build/BUILD-CMDS.tsv
}
read -r src_commit src_dirty <<< "$(gitstate . kernels build.sh)"
read -r gg_commit gg_dirty <<< "$(gitstate "$IG" ggml)"
info built "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
info arch "$ARCH"
info toolchain "TheRock ROCm"
info toolchain_version "$(basename "$TR")"
info compiler "$("$TR/bin/amdclang++" --version | head -1)"
info source_commit "$src_commit"
info source_dirty "$src_dirty"
info kernel_tree_sha256 "$(find kernels build.sh -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64)"
info ggml_commit "$gg_commit"
info ggml_dirty "$gg_dirty"
for src in kernels/*.hip; do
    out=build/$(basename "$src" .hip).hsaco
    # test builds (kernels/k3_gguf*_swiglu.hip: REAP-308 with the swiglu glue, the forced-unfused shared expert) only
    # with K3_TEST_BUILDS=1
    case "$src" in *_swiglu.hip) [ "${K3_TEST_BUILDS:-0}" = 1 ] || continue ;; esac
    case "$src" in kernels/prefill_mmq.hip|kernels/prefill_moe.hip|kernels/prefill_dense.hip) continue ;; esac   # below: their own pipeline

    extra=(); std=c++20
    # k3_gguf*: LLVM's max-ilp machine scheduler (not occupancy-driven): the dense matvec loops issue their weight loads in
    # batches instead of one wait per dependent pair (attn_qkv 29.7 -> 17.6 us, hc_*_down 15 -> 10), same instructions
    # otherwise (bitwise); TP decode 13.56 -> 12.19 ms/token. It does not trade registers for occupancy: register-heavy
    # code spills at the 96-VGPR cap sooner than under the default scheduler (check llvm-readelf after changes)
    case "$src" in kernels/k3_q3m*.hip|kernels/k3_q35_*.hip) extra=(-mllvm -amdgpu-sched-strategy=max-ilp); std=c++17 ;;   # (the own decode modules: k3_gguf's scheduler, no ggml headers)
    esac
    case "$src" in *k3_gguf*.hip) extra=(-DGGML_USE_HIP -D__HIP_PLATFORM_AMD__ -I"$GG/include" -I"$GG/src" -I"$GG/src/ggml-cuda" -mllvm -amdgpu-sched-strategy=max-ilp); std=c++17 ;;
    # gg_mmvf_ref: ignite's mmvf.cu as is, with ignite's compile flags (build-hip-therock compile_commands.json): the BF16
    # matvec reference of tests/test_k3_bf16_mv.py
    *gg_mmvf_ref.hip) extra=(-DNDEBUG -fPIC -DGGML_USE_HIP -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -DGGML_HIP_GRAPHS -DGGML_HIP_NO_VMM
                             -DGGML_SCHED_MAX_COPIES=4 -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SHARED -D_GNU_SOURCE -D_XOPEN_SOURCE=600
                             -I"$GG/include" -I"$GG/src" -I"$GG/src/ggml-cuda"); std=gnu++17 ;;
    esac
    cmd=("$TR/bin/amdclang++" -x hip --rocm-path="$TR" --offload-arch="$ARCH" --offload-device-only --no-gpu-bundle-output
         -mcode-object-version=5 -O3 -std=$std -Ikernels "${extra[@]}" -o "$out" "$src")
    logcmd "$out" "${cmd[*]}"
    "${cmd[@]}"
    echo "$out"
done
# kernels/prefill_mmq.hip (the batched prefill's quantized GEMMs: ggml-cuda's MMQ vendored in kernels/ggml_mmq, MIT): one
# code object per format and one for the glue, each built the way ignite builds the same kernels (its compile_commands
# flags; a full HIP compile, -c, whose device half goes through the offload driver's LTO; the gfx1100 object unbundled).
# That is what makes the kernels' instructions equal to ignite's shipped ones (tests/test_prefill_mmq.py codegen):
# --offload-device-only (no device LTO) differed in 27 of 68 kernels, and all formats in one object in IQ2_S / IQ3_S.
# Intermediates in build/obj (overwritten on every build)
PF=(-O3 -DNDEBUG -fPIC -std=gnu++17 -DGGML_USE_HIP -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -DGGML_HIP_GRAPHS -DGGML_HIP_NO_VMM
    -DGGML_SCHED_MAX_COPIES=4 -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SHARED -DUSE_PROF_API=1 -D_GNU_SOURCE
    -D_XOPEN_SOURCE=600 -Ikernels -I"$GG/include" -I"$GG/src" -I"$GG/src/ggml-cuda")
mkdir -p build/obj
pfbuild() {   # pfbuild OUT OBJ SRC DEFINE...: the pipeline above (a full HIP compile with device LTO, then the gfx1100 object unbundled)
    local out=$1 o=$2 src=$3; shift 3
    local -a c1=("$TR/llvm/bin/clang++" -x hip --offload-arch="$ARCH" "${PF[@]}" "$@" -c -o "$o.o" "$src")
    local -a c2=("$TR/lib/llvm/bin/llvm-objcopy" --dump-section=.hip_fatbin="$o.fatbin" "$o.o" /dev/null)
    local -a c3=("$TR/lib/llvm/bin/clang-offload-bundler" --unbundle --type=o --input="$o.fatbin"
                 --targets=hipv4-amdgcn-amd-amdhsa--"$ARCH" --output="$out")
    logcmd "$out" "${c1[*]} && ${c2[*]} && ${c3[*]}"
    "${c1[@]}"; "${c2[@]}"; "${c3[@]}"
    echo "$out"
}
for v in IQ2_S IQ3_S IQ3_XXS IQ4_NL IQ4_XS Q2_0 Q3_K Q4_K Q5_K Q6_K Q8_0 GLUE GLUE8; do
    n=$(echo "$v" | tr 'A-Z' 'a-z'); o=build/obj/prefill_mmq_$n; out=build/prefill_mmq_$n.hsaco
    case $v in GLUE) def=(-DK3_MMQ_GLUE) ;; GLUE8) def=(-DK3_MMQ_GLUE -DK3_MMQ_IDS_K=8) ;; *) def=(-DK3_MMQ_TYPE=GGML_TYPE_$v) ;; esac   # GLUE8: qwen35moe's top-8
    pfbuild "$out" "$o" kernels/prefill_mmq.hip "${def[@]}"
done
# kernels/prefill_moe.hip (the batched prefill's own MoE grouped GEMMs; it compiles the vendored MMQ's per-format load_tiles /
# vec_dot): the same pipeline and flags as prefill_mmq, one object per MoE format (the Coder's gate / up IQ2_S IQ3_S IQ3_XXS
# IQ4_XS, down IQ4_NL Q2_0) and one for the grouping
for v in IQ2_S IQ3_S IQ3_XXS IQ4_XS IQ4_NL Q2_0 GLUE; do
    n=$(echo "$v" | tr 'A-Z' 'a-z'); o=build/obj/prefill_moe_$n; out=build/prefill_moe_$n.hsaco
    if [ "$v" = GLUE ]; then def=-DK3_MOE_GLUE; else def=-DK3_MOE_TYPE=GGML_TYPE_$v; fi
    pfbuild "$out" "$o" kernels/prefill_moe.hip "$def"
done
# kernels/prefill_dense.hip (the batched prefill's own dense quantized GEMMs, impl:q6k-dense; it compiles the vendored MMQ's
# load_tiles / vec_dot / write_back for its control variant and the tile helpers for ours): the same pipeline and flags as
# prefill_mmq, one object per format (Q6_K)
for v in Q6_K; do
    n=$(echo "$v" | tr 'A-Z' 'a-z'); o=build/obj/prefill_dense_$n; out=build/prefill_dense_$n.hsaco
    pfbuild "$out" "$o" kernels/prefill_dense.hip -DK3_DENSE_TYPE=GGML_TYPE_$v
done
# the C++ runtime and its tests, linked to torch's bundled HIP runtime and rocBLAS (the binaries training runs)
TORCHLIB=${TORCHLIB:-$HOME/miniforge3/envs/drafter/lib/python3.12/site-packages/torch/lib}
PATCHELF=${PATCHELF:-$HOME/miniforge3/envs/drafter/bin/patchelf}
CXX="$TR/bin/amdclang++ -O2 -std=c++20 -I$TR/include -Iruntime -D__HIP_PLATFORM_AMD__"
LINK="$TORCHLIB/libamdhip64.so $TORCHLIB/librocblas.so -Wl,-rpath,$TORCHLIB"   # full paths: not TheRock's runtime
for t in runtime/test_*.cpp; do
    out="build/$(basename "$t" .cpp)"
    $CXX runtime/rt.cpp runtime/s1.cpp "$t" -o "$out" $LINK
    # torch ships these under unversioned names and its own libraries NEED those names; record the same, so the
    # process loads exactly one copy of each (a soname-versioned entry would not be found, or load a second copy)
    "$PATCHELF" --replace-needed libamdhip64.so.7 libamdhip64.so --replace-needed librocblas.so.5 librocblas.so "$out"
    echo "$out"
done
