#!/usr/bin/env bash
# S3 sampler references, CPU only, without changing either source tree:
#   build/sampler_ref  llama.cpp's sampler chain (ignite-llama build-cpu libllama)
#   build/philox_ref   torch's header-only Philox4x32-10 (drafter env headers; no libtorch link)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
IG=${IGNITE:-$HOME/src/ignite-llama}; B=$IG/build-cpu/bin
TORCH_INC=${TORCH_INC:-$HOME/miniforge3/envs/drafter/lib/python3.12/site-packages/torch/include}
mkdir -p build
c++ -O2 -std=c++17 -DLLAMA_SHARED -I"$IG/include" -I"$IG/ggml/include" tools/reference/sampler_ref.cpp -o build/sampler_ref \
    -L"$B" -lllama -lggml -lggml-base -Wl,-rpath,"$B"
c++ -O2 -std=c++17 -I"$TORCH_INC" tools/reference/philox_ref.cpp -o build/philox_ref
echo build/sampler_ref build/philox_ref
