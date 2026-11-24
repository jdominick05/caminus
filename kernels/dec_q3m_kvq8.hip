// The qwen3moe decode module (kernels/dec_q3m.hip) with its KV cache in Q8_0 rows (kernels/kv_codec_ops.h: the F16
// object's rows encoded, K after the head's Walsh-Hadamard transform, kernels/kv_rot128.h, and q rotated the same way).
// Only OP_OWN_ATTS changes (kernels/ops_own_atts.h under K3_Q3M_KV_Q8: the rotations, the row encode, the Q8 K / V reads,
// the sticky error word after Sync, its counts pinned as scalars); a separate object, so dec_q3m keeps its code. One
// card, decode only. K3_Q3M_ADD_RMS_PIN (kernels/ops_own.h: OP_OWN_ADD_RMS16's row length and slot count pinned as
// scalars, no arithmetic changed): without it k3_run spilled 54 scalars against dec_q3m's 53; with it 31 (before the
// rotation and the attention's pin; 19 with both).
#define K3_Q3M_KV_Q8
#define K3_Q3M_ADD_RMS_PIN
#include "dec_q3m.hip"
