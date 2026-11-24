// The qwen3moe two-card Q8_0 KV decode module (kernels/dec_q3m_tp_kvq8.hip) for the long-context mode: OP_OWN_ATTS's
// RoPE from the host's YaRN table (kernels/ops_own_atts.h under K3_Q3M_YARN). A separate object, so dec_q3m_tp_kvq8
// keeps its code.
#define K3_Q3M_YARN
#include "dec_q3m_tp_kvq8.hip"
