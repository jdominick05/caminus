// The qwen3moe batched prefill's Q8 KV kernels (kernels/q3m_prefill_kvq8.hip) for the long-context mode: q3p_qk_rope_q8 from
// the YaRN table (Q3P_YARN, as kernels/q3m_prefill_yarn.hip). A separate object, so q3m_prefill_kvq8 keeps its code;
// caminus.prefill.q3m runs q3p_qk_rope_q8 and q3p_attn_q8 from it (the attention's code is q3m_prefill_kvq8's).
#define Q3P_YARN
#include "q3m_prefill_kvq8.hip"
