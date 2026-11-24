// The qwen35moe batched prefill's Q8 KV kernels (kernels/q35_prefill.hip under Q35P_KV_Q8: q35p_kv_append_q8;
// kernels/q35m_prefill.hip under it: q35mp_attn_q8; Q3P_KV_Q8 includes kernels/own_kv_codec.h, and q3m_prefill.hip's own
// Q8 forms come along unused), a separate object so q35m_prefill keeps its code; caminus.prefill.q35moe runs every other
// kernel from q35m_prefill.
#define Q35P_KV_Q8
#define Q3P_KV_Q8
#include "q35m_prefill.hip"
