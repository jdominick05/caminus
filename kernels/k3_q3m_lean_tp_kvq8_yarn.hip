// The qwen3moe two-card Q8_0 KV decode module (kernels/k3_q3m_lean_tp_kvq8.hip) for the long-context mode: OP_OWN_ATTS's
// RoPE from the host's YaRN table (kernels/k3_own_atts.h under K3_Q3M_YARN). A separate object, so k3_q3m_lean_tp_kvq8
// keeps its code.
#define K3_Q3M_YARN
#include "k3_q3m_lean_tp_kvq8.hip"
