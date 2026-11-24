// The qwen35moe batched prefill's Q8 KV kernels (kernels/pf_q35_swift.hip under Q35P_KV_Q8: q35p_kv_append_q8;
// kernels/pf_q35m.hip under it: q35mp_attn_q8; Q3P_KV_Q8 includes kernels/kv_codec_ops.h, and pf_q3m.hip's own
// Q8 forms come along unused), a separate object so pf_q35m keeps its code; caminus.prefill.q35moe runs every other
// kernel from pf_q35m.
#define Q35P_KV_Q8
#define Q3P_KV_Q8
#include "pf_q35m.hip"
