// The qwen3moe two-card decode module (kernels/k3_q3m_lean_tp.hip: OP_XCHG push-only, OP_OWN_SLOT_SUM) with its KV cache
// in Q8_0 rows: the one-card Q8 module's format and statements (kernels/k3_q3m_lean_kvq8.hip; kernels/k3_own_atts.h under
// K3_Q3M_KV_Q8: K and q through the head's Walsh-Hadamard transform, kernels/own_kv_rot128.h, the row encode, the Q8 K / V
// reads, the sticky error word after Sync) at the two-card shapes (each card its 2 of the 4 kv heads, up to 128 attention
// splits per kv head: ATS_MAXS 128 under K3_Q3M_TP). A separate object, so k3_q3m_lean_tp and k3_q3m_lean_kvq8 keep their
// code. K3_Q3M_ADD_RMS_PIN as in the one-card Q8 module (OP_OWN_ADD_RMS16's row length and slot count pinned as scalars,
// no arithmetic changed).
#define K3_Q3M_KV_Q8
#define K3_Q3M_ADD_RMS_PIN
#include "k3_q3m_lean_tp.hip"
