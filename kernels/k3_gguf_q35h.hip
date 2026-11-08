// the GGUF engine for Qwen3.6-35B-A3B (llama.cpp arch qwen35moe) on ONE card, HauhauCS Uncensored-Aggressive Q3_K_P's
// formats (python/caminus/qwen35_one.py; $ARTIFACTS/q36/BASELINE.md): the GSQ module's k3_run (K3_GSQ: split attention,
// the F32 router, the shared expert's SWIGLU glue) with qwen35moe's two choices: the gated RMSNorm's output gate is SiLU
// (K3_GDN_SILU, as qwen3next) and the GDN value heads read key head h % nk (tiled: the default mapping, as Flash-Next;
// tests/test_q36_golden_cpu.py). 16 query heads over 2 kv heads: ATTN_G 8. Formats: dense / attention / GDN projections
// Q3_K / Q4_K / Q5_K and the head Q6_K; experts gate / up Q3_K or Q5_K (per layer), down Q4_K or Q5_K. No fixed-shape MoE
// paths (those are the Coder's IQ4_NL / Q2_0 / IQ3_XXS shapes)
#define K3_GSQ
#define K3_GDN_SILU
#define K3_Q8_SWIGLU   // OP_GG_Q8_1 mode 2 (the shared expert's swiglu glue)
#define ATTN_G 8
#define ATTN_VH 8
#define K3_ATTN_COMB_FAST   // OP_ATTN_COMB: the parallel max and loads ahead (kernels/k3_attn_split.h; the same bits)
#define K3_FMTS_MV GG_Q3_K, GG_Q4_K, GG_Q5_K, GG_Q6_K
#define K3_FMTS_MOE_GU GG_Q3_K, GG_Q5_K
#define K3_FMTS_MOE_DOWN GG_Q4_K, GG_Q5_K
#define K3_FMTS_GU_DENSE
#define K3_MOE_N 0
#define K3_MOE_P 0
#define K3_MV_Q3K      // the standalone gg_mv_q3_k (tests' reference; a separate kernel, not in k3_run)
#define K3_TG_Q3_K 2   // Q3_K matvec rows in token groups of 2: 4 spilled 33 VGPRs (_mt 29) at the 96 cap; 2: 0 (11-07)
#include "k3_gguf.hip"
