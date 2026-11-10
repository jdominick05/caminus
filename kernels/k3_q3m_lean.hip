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
#define K3_MAX_VGPR 96
#define K3_EXTRA_OPS_HEADER "k3_q3m_ops.h"
#include "k3.hip"
