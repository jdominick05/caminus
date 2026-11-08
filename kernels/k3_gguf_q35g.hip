// the GGUF engine for Qwen3.6-35B-A3B 2-bit GSQ (ISTA-DASLab; the GGUF chtisgit/Qwen3.6-35B-A3B-2Bit-GSQ-hybrid-GGUF, an exact
// remap: experts Q2_0, the INT8 attention / linear attention / shared experts / head as Q8_0, ssm_alpha / ssm_beta BF16) on ONE
// card (python/caminus/qwen35_one.py): kernels/k3_gguf_q35h.hip's engine (K3_GSQ split attention + F32 router + SWIGLU glue,
// K3_GDN_SILU, tiled GDN heads, ATTN_G 8) with this checkpoint's formats: MV Q8_0 (the one-token loads-first path, K3_MV1),
// experts gate / up / down Q2_0 (the generic gg_moe_*_t: K = 512, not the Coder's lane-packed 384 / 256 shapes), BF16 through
// OP_BF16_MV (K3_GSQ's)
#define K3_GSQ
#define K3_GDN_SILU
#define K3_Q8_SWIGLU   // OP_GG_Q8_1 mode 2 (the shared expert's swiglu glue)
#define ATTN_G 8
#define ATTN_VH 8
#define K3_ATTN_COMB_FAST   // OP_ATTN_COMB: the parallel max and loads ahead (kernels/k3_attn_split.h; the same bits)
#define K3_FMTS_MV GG_Q8_0
#define K3_FMTS_MOE_GU GG_Q2_0
#define K3_FMTS_MOE_DOWN GG_Q2_0
#define K3_FMTS_GU_DENSE
#define K3_MOE_N 0
#define K3_MOE_P 0
#include "k3_gguf.hip"
