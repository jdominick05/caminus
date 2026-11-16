// The qwen3moe decode module (kernels/k3_q3m_lean.hip) with its KV cache in Q8_0 rows (kernels/own_kv_codec.h, no rotation:
// the F16 object's rows encoded). Only OP_OWN_ATTS changes (kernels/k3_own_atts.h under K3_Q3M_KV_Q8: the row encode, the
// Q8 K / V reads, the sticky error word after Sync); a separate object, so k3_q3m_lean keeps its code. One card, decode only.
#define K3_Q3M_KV_Q8
#include "k3_q3m_lean.hip"
