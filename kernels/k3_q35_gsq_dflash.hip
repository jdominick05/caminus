// The DFlash 2 drafter of the dense qwen35 line on the plain Qwen3.8-27B GSQ-RCO files' module (engine/q35_dflash.py):
// kernels/k3_q35_tcap_dflash.hip's tapes with the target's IQ2_S embedding and its Q4_K head (Swift's T-token form; OP_OWN_MV
// with that format's loop alone, no grid staged), plus kernels/k3_df_own_ops.h (K3_Q35_DF). The GSQ decode module's .text is
// unchanged: everything here sits under K3_Q35_DF / K3_Q3M_VER.
#define K3_Q35_DF
#define K3_Q3M_VER
#define K3_Q35_VER
#include "k3_q35_gsq.hip"
