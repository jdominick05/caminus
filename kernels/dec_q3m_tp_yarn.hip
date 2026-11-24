// The qwen3moe two-card decode module (kernels/dec_q3m_tp.hip) for the long-context mode: OP_OWN_ATTS's RoPE from the
// host's YaRN table (kernels/ops_own_atts.h under K3_Q3M_YARN, as kernels/dec_q3m_yarn.hip). A separate object, so
// dec_q3m_tp keeps its code.
#define K3_Q3M_YARN
#include "dec_q3m_tp.hip"
