// The plain Qwen3.8-27B GSQ-RCO files' multi-token-prediction drafter (engine/q35_mtp.py): their prediction block (all
// Q6_K: the planes' T-token tiles, as ThinkingCap's), the Q4_K head (Swift's T-token form, one row), the IQ2_S embedding
// reading OP_OWN_ARGMAX's key on a chained step, on the GSQ module (kernels/k3_q35t_ops.h under K3_Q35_MTP + K3_Q35_GSQ);
// the decode and verify modules' .text unchanged.
#define K3_Q35_MTP
#define K3_Q3M_VER
#define K3_Q35_VER
#include "k3_q35_gsq.hip"
