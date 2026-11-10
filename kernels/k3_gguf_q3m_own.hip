// K3 module for qwen3moe (Qwen3-30B-A3B) on Caminus's own math (docs/specs/spec-own-quant.md): OP_OWN_MV's one-round-trip
// tiles (Q4_K, Q6_K planes; the 8 experts of a layer grouped by device ids), OP_OWN_F16 / SWIGLU16 / MOE_SUM, and the
// engine's own rms / q-k norm + RoPE / KV append / decode attention (OP_ATTN_DEC, not the split pair: head dim 128,
// 8 query heads per kv head) / F32 router matvec / router. No llama.cpp op runs in this model's tape; the GG matvec
// formats below stay compiled only as the yardstick arm of the benchmarks (until the module drops the ggml headers).
#define K3_OWN
#define K3_OWN_Q3M
#define K3_FMTS_MV GG_Q4_K, GG_Q6_K
#define K3_FMTS_MOE_GU
#define K3_FMTS_MOE_DOWN
#define K3_FMTS_GU_DENSE
#define K3_MOE_N 0
#define K3_MOE_P 0
#include "k3_gguf.hip"
