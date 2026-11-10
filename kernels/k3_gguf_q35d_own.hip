// The dense qwen35 module (kernels/k3_gguf_q35d.hip, AREX-2) with Caminus's own matvec op OP_OWN_MV (kernels/k3_own_ops.h)
// compiled in: the in-module A/B of docs/specs/spec-own-quant.md (G-speed). k3_gguf_q35d itself is unchanged.
#define K3_OWN
#include "k3_gguf_q35d.hip"
