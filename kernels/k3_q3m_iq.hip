// The qwen3moe decode module of the IQ quant study (docs/dev/iq-quant-study.md): kernels/k3_q3m_lean.hip's defines plus
// K3_Q3M_IQ, which adds the codebook formats' tiles (IQ4_XS, IQ3_XXS, IQ3_S, IQ2_XXS: kernels/own_quant_iq.h) to the
// own matvec op and a launch hook that stages their grids. A measurement module: every arm of the study runs this one
// code object, and the shipped decode module stays as it is.
#define K3_OWN
#define K3_OWN_Q3M
#define K3_Q3M_LEAN
#define K3_Q3M_IQ
#define K3_ROUTER_NE 256   // (128 experts; the router's other 3 KB of LDS go to the grids: the module keeps the shipped one's LDS size)
#define K3_XWARM 0
#define K3_NO_G128
#define K3_OPAQUE_TID
#define K3_MAX_VGPR 96
#define K3_EXTRA_OPS_HEADER "k3_q3m_ops.h"
#include "k3.hip"
