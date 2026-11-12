// The dense qwen35 line's DFlash 2 drafter module (incoai's Qwen3.8-27B-DFlash2 on Swift 1.5: engine/q35_dflash.py). The
// Swift module's T-token ops (K3_Q3M_VER / K3_Q35_VER: the Q8_0 and Q6_K matvecs over the draft block's rows, add + norm,
// the q / k norm with RoPE, SwiGLU, the Q4_K embedding) and the drafter's own fp32 ops (kernels/k3_df_own_ops.h: the
// dynamic short conv, top-k, the selector walk, the non-causal windowed attention over its f16 ring). Decode attention as
// OP_ATTN_DEC (the split pair is the trunk's). A module of its own: every other code object is unchanged.
#define K3_Q35_DF
#define K3_Q3M_VER
#define K3_Q35_VER
#include "k3_q35_swift.hip"
