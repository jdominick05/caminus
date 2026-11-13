// The DFlash 2 drafter of the dense qwen35 line on ThinkingCap's module (engine/q35_dflash.py): kernels/k3_q35_dflash.hip's
// tapes with ThinkingCap's IQ2_S embedding (kernels/k3_q35t_ops.h q35t_op_emb, its T-token form) and Q8_0 head (the rows
// tiles' T-token form, OP_OWN_MV16), plus kernels/k3_df_own_ops.h (K3_Q35_DF). ThinkingCap's decode module's .text is
// unchanged: everything here sits under K3_Q35_DF / K3_Q3M_VER.
#define K3_Q35_DF
#define K3_Q3M_VER
#define K3_Q35_VER
#include "k3_q35_tcap.hip"
