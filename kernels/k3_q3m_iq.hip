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
#ifndef K3_PER_WGP           // the XT's grid, 5 x 42 WGPs: VGPR cap 144 (kernels/k3.hip); build.sh adds a _w4 object
#define K3_PER_WGP 5        // (-DK3_PER_WGP=4, cap 192) for the XTX's 4 x 48
#endif
#define K3_EXTRA_OPS_HEADER "k3_q3m_ops.h"
#include "k3.hip"
