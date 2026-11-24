// The qwen3moe batched prefill's Q8 KV kernels (kernels/pf_q3m.hip under Q3P_KV_Q8: q3p_qk_rope_q8, q3p_attn_q8), a
// separate object so pf_q3m keeps its code; caminus.prefill.q3m runs every other kernel from pf_q3m.
#define Q3P_KV_Q8
#include "pf_q3m.hip"
