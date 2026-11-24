// The plain Qwen3.8-27B GSQ-RCO files' verify module: kernels/dec_q35_gsq.hip with K3_Q3M_VER (its formats' T-token tiles:
// kernels/quant_tcap_ver.h and kernels/quant_gsq_ver.h, through kernels/ops_q35_tcap.h; the verify passes of
// engine/q35_dense_verify.py on the GSQ engine's weights and states). The decode module's .text is unchanged: every
// T-token statement sits under K3_Q3M_VER.
#define K3_Q3M_VER
#define K3_Q35_VER
#define K3_Q35_ATTN_PACK   // (the split pair's PART takes every pass row of a split in one task: kernels/ops_attn_split.h)
#include "dec_q35_gsq.hip"
