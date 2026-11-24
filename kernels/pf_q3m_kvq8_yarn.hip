// The qwen3moe batched prefill's Q8 KV kernels (kernels/pf_q3m_kvq8.hip) for the long-context mode: q3p_qk_rope_q8 from
// the YaRN table (Q3P_YARN, as kernels/pf_q3m_yarn.hip). A separate object, so pf_q3m_kvq8 keeps its code;
// caminus.prefill.q3m runs q3p_qk_rope_q8 and q3p_attn_q8 from it (the attention's code is pf_q3m_kvq8's).
#define Q3P_YARN
#include "pf_q3m_kvq8.hip"
