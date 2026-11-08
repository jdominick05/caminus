// The batched prefill's glue for qwen35moe (Qwen3.6-35B-A3B, one card; python/caminus/prefill35.py): prefill_glue.hip with
// rows of n_embd 2048 (GLUE_NI 8: pf_rms_mul is OP_RMS_MUL's general loop at K = 2048, the same fma chain and sums),
// pf_bf16_mv_n4 (K 2048: the GSQ checkpoint's BF16 ssm alpha / beta) and pf_f32_mv_k2048 (the F32 router logits and
// shared-expert gate, OP_F32_MV's order). The Coder's prefill_glue.hsaco is unchanged (these defines only).
#define GLUE_NI 8
#define PF_Q35
#include "prefill_glue.hip"
