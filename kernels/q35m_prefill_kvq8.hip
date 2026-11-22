// The qwen35moe batched prefill's Q8 KV kernels (kernels/q35_prefill.hip under Q35P_KV_Q8: q35p_kv_append_q8,
// q35p_attn_wm_q8; the block decoder q3p_q8_stage from kernels/q3m_prefill.hip under Q3P_KV_Q8), a separate object so
// q35m_prefill keeps its code; caminus.prefill.q35moe runs every other kernel from q35m_prefill.
#define Q35P_KV_Q8
#define Q3P_KV_Q8
#include "q35m_prefill.hip"
