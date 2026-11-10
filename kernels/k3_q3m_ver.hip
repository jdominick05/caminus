// The qwen3moe verify module (Qwen3-30B-A3B, speculative decoding): the lean decode module's ops (kernels/k3_q3m_lean.hip)
// with their T-token forms (K3_Q3M_VER, kernels/k3_own_ops.h): one matvec segment serves every token of the pass (the
// weights' bytes are read from memory once), the embedding and the add + norm take one task per token, the experts'
// gate | up reads each token's own input. A module of its own: the decode module's k3_run stays as it is. Each token's
// sums are the decode's statements, so a pass gives the decode's bits.
#define K3_OWN
#define K3_OWN_Q3M
#define K3_Q3M_LEAN
#define K3_Q3M_VER
#define K3_XWARM 0
#define K3_NO_G128
#define K3_OPAQUE_TID
#define K3_MAX_VGPR 96
#define K3_EXTRA_OPS_HEADER "k3_q3m_ops.h"
#include "k3.hip"
