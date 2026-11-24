// The Qwen3.5 dense line (AREX-2 Q4_K_M) multi-token GGUF engine (T > 1: verify passes for a drafter; see kernels/dec_q4x_gg_ver.hip
// and kernels/dec_q35_arex.hip)
#define K3_MT
#include "dec_q35_arex.hip"
