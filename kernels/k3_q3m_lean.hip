// The qwen3moe decode module (Qwen3-30B-A3B), built without llama.cpp's headers: the K3 core with only the decode tape's
// ops (kernels/k3_q3m_ops.h: Caminus's own ops and the engine's F32 router matvec / router), no ggml include path. Until
// 11-10 the same ops were picked by a lean dispatch inside the full GGUF op header (k3_gguf_q3m_lean.hip, which included
// ggml-cuda's common.cuh / vecdotq.cuh / quantize.cuh); k3_run came out instruction for instruction the same (8,998
// instructions, llvm-objdump), so the bits are the same. build.sh compiles it with the k3_gguf modules' max-ilp
// scheduler and C++17.
#define K3_OWN
#define K3_OWN_Q3M
#define K3_Q3M_LEAN
#define K3_XWARM 0
#define K3_NO_G128
#define K3_OPAQUE_TID
#ifndef K3_PER_WGP           // the XT's grid, 5 x 42 WGPs: VGPR cap 144 (kernels/k3.hip); build.sh adds a _w4 object
#define K3_PER_WGP 5        // (-DK3_PER_WGP=4, cap 192) for the XTX's 4 x 48
#endif
#define K3_EXTRA_OPS_HEADER "k3_q3m_ops.h"
#include "k3.hip"
