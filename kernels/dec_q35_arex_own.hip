// The dense qwen35 module (kernels/dec_q35_arex.hip, AREX-2) with Caminus's own matvec op OP_OWN_MV (kernels/ops_own.h)
// compiled in: the in-module A/B of docs/specs/spec-own-quant.md (G-speed). dec_q35_arex itself is unchanged.
#define K3_OWN
#include "dec_q35_arex.hip"
