// ThinkingCap's verify module: kernels/k3_q35_tcap.hip with K3_Q3M_VER (its T-token tiles: kernels/own_quant_tcap_mt.h,
// kernels/k3_q35t_ops.h; the verify passes of engine/q35_dense_verify.py on ThinkingCap's weights and states). The decode
// module's .text is unchanged: every T-token statement sits under K3_Q3M_VER.
#define K3_Q3M_VER
#define K3_Q35_VER
#define K3_Q35_ATTN_PACK   // (the split pair's PART takes every pass row of a split in one task: kernels/k3_attn_split.h)
#include "k3_q35_tcap.hip"
