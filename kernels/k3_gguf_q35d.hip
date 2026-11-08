// K3 module for the Qwen3.5 dense line (GGUF arch qwen35: BAAI AREX-2 Q4_K_M; caminus.qwen35_one.Qwen35One, one card): the
// qwen35moe engine's ops (GDN with the SiLU output gate, tiled key heads; gated attention) with a dense FFN (two matvecs and
// OP_GG_Q8_1's fused swiglu, the shared expert's pattern) and no routed experts. 24 query / 4 kv heads: 6 query heads per
// kv head (ATTN_G 6, ATTN_VH 6). The checkpoint's matrices: Q4_K / Q5_K / Q6_K / Q8_0; F32 ssm alpha / beta (OP_F32_MV)
#define K3_GSQ
#define K3_GDN_SILU
#define K3_Q8_SWIGLU   // OP_GG_Q8_1 mode 2 (the dense FFN's swiglu glue)
#define ATTN_G 6
#define ATTN_VH 6
#define K3_ATTN_COMB_FAST   // OP_ATTN_COMB: the parallel max and loads ahead (kernels/k3_attn_split.h; the same bits)
#define K3_FMTS_MV GG_Q4_K, GG_Q5_K, GG_Q6_K, GG_Q8_0
#define K3_FMTS_MOE_GU
#define K3_FMTS_MOE_DOWN
#define K3_FMTS_GU_DENSE
#define K3_MOE_N 0
#define K3_MOE_P 0
#include "k3_gguf.hip"
