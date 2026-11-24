// prefill_attn_q35.hip in the long-context mode (YaRN): pf_qk_nr takes the decode op's YaRN table and magnitude scale
// (PF_QK_YARN: k3_q35_ops.h's K3_Q35_YARN lines); every other kernel as prefill_attn_q35. Loaded by caminus.prefill.q35moe
// only when the engine runs YaRN, so the native-rope objects keep their code.
#define ATTN_G 8
#define ATTN_VH 8
#define PF_QK_YARN
#include "prefill_attn.hip"
