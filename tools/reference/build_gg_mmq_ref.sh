#!/usr/bin/env bash
# gg_mmq_ref (tools/reference/gg_mmq_ref.cpp): links the ignite-llama fork's libggml without changing that tree.
#   tools/reference/build_gg_mmq_ref.sh                         -> build/gg_mmq_ref against build-hip-therock (the stock build: ggml's
#                                                        own kernel choice per T, as llama-bench runs)
#   GGML_LIB=DIR OUT=PATH tools/reference/build_gg_mmq_ref.sh  -> against another fork build, e.g. one with GGML_CUDA_FORCE_MMQ=ON (MMQ
#                                                        at every T, the G1 reference for the dense formats ggml sends to
#                                                        hipBLAS above 128 columns). Such a build, out of tree, ggml only:
#     cmake -S ~/src/ignite-llama -B DIR -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TOOLS=OFF
#       -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DGGML_HIP=ON
#       -DGGML_HIP_NO_VMM=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CCACHE=OFF
#       -DGGML_NATIVE=ON -DAMDGPU_TARGETS=gfx1100 -DGPU_TARGETS=gfx1100 -DCMAKE_{C,CXX,HIP}_COMPILER=$TR/llvm/bin/clang{,++,++}
#       -DCMAKE_PREFIX_PATH=$TR && ninja -C DIR ggml          (its compile flags equal build-hip-therock's + FORCE_MMQ)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
IG=${IGNITE:-$HOME/src/ignite-llama}; TR=${ROCM_PATH:-$HOME/rocm-therock/10.2.0a20261011}
B=${GGML_LIB:-$IG/build-hip-therock/bin}; OUT=${OUT:-build/gg_mmq_ref}
mkdir -p "$(dirname "$OUT")"
"$TR/lib/llvm/bin/clang++" -O2 -std=c++17 -I"$IG/ggml/include" tools/reference/gg_mmq_ref.cpp -o "$OUT" \
    -L"$B" -lggml -lggml-base -Wl,-rpath,"$B"
echo "$OUT"
