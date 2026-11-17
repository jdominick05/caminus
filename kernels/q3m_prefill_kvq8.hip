// The qwen3moe batched prefill's Q8 KV kernels (kernels/q3m_prefill.hip under Q3P_KV_Q8: q3p_qk_rope_q8, q3p_attn_q8), a
// separate object so q3m_prefill keeps its code; caminus.prefill.q3m runs every other kernel from q3m_prefill.
#define Q3P_KV_Q8
#include "q3m_prefill.hip"
