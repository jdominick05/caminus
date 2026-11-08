// the GGUF engine for Qwen3-Coder-Next (qwen3next; python/caminus/qwen3next_tp.py): the GSQ module's k3_run (K3_GSQ: split
// attention, the F32 router, the shared expert's SWIGLU glue) with each GG op instantiated only for the UD-IQ3_S formats
// (dense Q6_K, ssm_ba IQ3_S, experts IQ2_S gate / up and IQ3_S down), plus qwen3next's two differences from qwen4exp behind
// K3_GDN_GROUPED / K3_GDN_SILU (kernels/k3_gguf_ops.h): value head h reads key head h / (H / nk) (repeat-interleave, not tiling), and the gated
// RMSNorm's output gate is SiLU, not sigmoid. 16 query heads over 2 kv heads: ATTN_G 8, one V pass of 8 heads.
#define K3_GSQ
#define K3_GDN_GROUPED   // value head h reads key head h / (H / nk) (qwen3next's repeat-interleave)
#define K3_GDN_SILU      // the gated RMSNorm's output gate is SiLU
#define K3_Q8_SWIGLU   // OP_GG_Q8_1 mode 2 (the shared expert's swiglu glue)
#define ATTN_G 8
#define ATTN_VH 8
#define K3_ATTN_COMB_FAST   // OP_ATTN_COMB: the parallel max and loads ahead (kernels/k3_attn_split.h; the same bits)
#define K3_FMTS_MV GG_IQ3_S, GG_Q6_K
#define K3_FMTS_MOE_GU GG_IQ2_S
#define K3_FMTS_MOE_DOWN GG_IQ3_S
#define K3_FMTS_GU_DENSE
#define K3_MOE_N 0      // the unrolled / lane-packed MoE paths are the Coder's IQ4_NL / Q2_0 downs and IQ3_XXS gate / up:
#define K3_MOE_P 0      // this model's IQ3_S down and IQ2_S gate / up take the generic gg_moe_*_t (a fixed-shape path: later)
#define K3_TG_IQ3_S 2   // IQ3_S matvec rows in token groups of 2: 4 spilled 6 VGPRs (_mt: 3) at the 96 cap, 2 / 1: 0 (11-06)
#include "k3_gguf.hip"
