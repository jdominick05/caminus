// The dense qwen35 decode module for ThinkingCap (Qwen3.8-27B RCO 4.2 bpw) on Caminus's own code: the K3 core with the
// ops of kernels/k3_q35t_ops.h (the Swift module's ops plus the IQ / Q5_K one-round-trip tiles and the IQ2_S
// embedding), no ggml include path. build.sh compiles it as the other own decode modules (max-ilp scheduler, C++17).
#define K3_OWN
#define K3_OWN_Q3M
#define K3_Q3M_LEAN
#define K3_XWARM 0
#define K3_NO_G128
#define K3_OPAQUE_TID
#define K3_MAX_VGPR 96
#if !defined(K3_Q35_DF)
#define K3_Q35_ATTN_SPLIT   // decode attention as the split pair (kernels/k3_q35_ops.h), the MTP drafter's module too; the DFlash module keeps OP_ATTN_DEC
#endif
#define K3_EXTRA_OPS_HEADER "k3_q35t_ops.h"
#include "k3.hip"
