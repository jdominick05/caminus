// The DFlash 2 drafter of the dense qwen35 line on ThinkingCap's module (engine/q35_dflash.py): kernels/dec_q35_swift_dflash.hip's
// tapes with ThinkingCap's IQ2_S embedding (kernels/ops_q35_tcap.h q35t_op_emb, its T-token form) and Q8_0 head (the rows
// tiles' T-token form, OP_OWN_MV16), plus kernels/ops_q35_dflash.h (K3_Q35_DF). ThinkingCap's decode module's .text is
// unchanged: everything here sits under K3_Q35_DF / K3_Q3M_VER.
#define K3_Q35_DF
#define K3_Q3M_VER
#define K3_Q35_VER
#include "dec_q35_tcap.hip"
