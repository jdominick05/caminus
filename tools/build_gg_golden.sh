#!/usr/bin/env bash
# build/gg_golden: links the ignite-llama fork's libllama (build-hip-therock) without changing that tree
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
IG=${IGNITE:-$HOME/src/ignite-llama}; TR=${ROCM_PATH:-$HOME/rocm-therock/10.2.0a20261011}; B=$IG/build-hip-therock/bin
mkdir -p build
"$TR/lib/llvm/bin/clang++" -O2 -std=c++17 -I"$IG/include" -I"$IG/ggml/include" tools/gg_golden.cpp -o build/gg_golden \
    -L"$B" -lllama -lggml -lggml-base -Wl,-rpath,"$B"
echo build/gg_golden
