// The qwen3moe decode module (kernels/k3_q3m_lean.hip) for the long-context mode (YaRN x4, caminus serve --long-context):
// OP_OWN_ATTS's RoPE takes each pair's frequency and the attention factor from the host's table at the end of its par block
// (kernels/k3_own_atts.h under K3_Q3M_YARN; caminus.rope.yarn_table) instead of theta_scale^j. A separate object, so
// k3_q3m_lean keeps its code: a runtime switch in the shared op would change the default module's machine code, and no
// value of the existing par block turns pos * theta_scale^j into YaRN's per-pair blend.
#define K3_Q3M_YARN
#include "k3_q3m_lean.hip"
