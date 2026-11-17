// The qwen3moe batched prefill (kernels/q3m_prefill.hip) for the long-context mode: q3p_qk_rope takes each pair's frequency
// and the attention factor from the YaRN table in its par block (Q3P_YARN; the decode modules' K3_Q3M_YARN statements on the
// same block). A separate object, so q3m_prefill keeps its code; caminus.prefill.q3m runs only q3p_qk_rope from it.
#define Q3P_YARN
#include "q3m_prefill.hip"
