// The batched prefill's glue for qwen35 dense (the Qwen3.5 dense line: AREX-2, n_embd 5120; python/caminus/prefill35.py):
// pf_gg_glue.hip with rows of 5120 (GLUE_NI 20: pf_rms_mul is OP_RMS_MUL's general loop at K 5120) and pf_f32_mv_k5120 (its
// F32 ssm alpha / beta, OP_F32_MV's order). The Coder's pf_gg_glue.hsaco and pf_gg_glue_q35m.hsaco are unchanged.
#define GLUE_NI 20
#define PF_Q35D
#include "pf_gg_glue.hip"
