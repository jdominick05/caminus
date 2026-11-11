// The dense qwen35 decode module on Caminus's own code (Swift 1.5 Q4_K_M, a Qwen3.8-27B finetune): the K3 core with
// only this line's decode tape's ops (kernels/k3_q35_ops.h), no ggml include path. build.sh compiles it as it does the
// qwen3moe module (the max-ilp scheduler, C++17).
#define K3_OWN
#define K3_OWN_Q3M
#define K3_Q3M_LEAN
#define K3_XWARM 0
#define K3_NO_G128
#define K3_OPAQUE_TID
#define K3_MAX_VGPR 96
#define K3_EXTRA_OPS_HEADER "k3_q35_ops.h"
#include "k3.hip"
