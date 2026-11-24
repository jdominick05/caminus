// pf_gg_attn_q35m.hip in the long-context mode (YaRN): pf_qk_nr takes the decode op's YaRN table and magnitude scale
// (PF_QK_YARN: ops_q35.h's K3_Q35_YARN lines); every other kernel as pf_gg_attn_q35m. Loaded by caminus.prefill.q35moe
// only when the engine runs YaRN, so the native-rope objects keep their code.
#define ATTN_G 8
#define ATTN_VH 8
#define PF_QK_YARN
#include "pf_gg_attn.hip"
