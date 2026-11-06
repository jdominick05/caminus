#!/usr/bin/env bash
# qwen_parse_ref: llama.cpp's chat-output parser (common/chat*) behind a JSONL driver, linked against ignite-llama's
# CPU build (build-cpu) without changing that tree. CPU only. Output: $OUT (default build/qwen_parse_ref).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
IG=${IGNITE:-$HOME/src/ignite-llama}; B=$IG/build-cpu/bin
OUT=${OUT:-build/qwen_parse_ref}
mkdir -p "$(dirname "$OUT")"
c++ -O2 -std=c++17 -DGGML_SHARED -DLLAMA_SHARED -I"$IG/common" -I"$IG/vendor" -I"$IG/include" -I"$IG/ggml/include" \
    tools/qwen_parse_ref.cpp -o "$OUT" -L"$B" -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,"$B"
echo "$OUT"
