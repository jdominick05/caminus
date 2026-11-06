// the GGUF engine for the GSQ-RCO Coder (docs/agentic/spec-gsq-port.md 4b): the same k3_run with each GG op instantiated
// only for the formats the Coder's tapes use (tools/gg_inventory.py MODEL --module k3_gguf_gsq prints these lists), plus
// standalone gg_mv_* kernels for the six new formats. 12 format instances (REAP-308's module: 16); no OP_GG_GU_DENSE (the
// shared expert's gate and up differ in format: two GG_MV + the SWIGLU glue). -D K3_FMTS_<op>=... overrides (experiments)
#define K3_GSQ
#define K3_Q8_SWIGLU   // OP_GG_Q8_1 mode 2 (the shared expert's swiglu glue)
#ifndef K3_FMTS_MV
#define K3_FMTS_MV GG_IQ4_NL, GG_IQ4_XS, GG_Q4_K, GG_Q5_K, GG_Q6_K, GG_Q8_0
#endif
#ifndef K3_FMTS_MOE_GU
#define K3_FMTS_MOE_GU GG_IQ2_S, GG_IQ3_S, GG_IQ3_XXS, GG_IQ4_XS
#endif
#ifndef K3_FMTS_MOE_DOWN
#define K3_FMTS_MOE_DOWN GG_IQ4_NL, GG_Q2_0
#endif
#ifndef K3_FMTS_GU_DENSE
#define K3_FMTS_GU_DENSE
#endif
#include "k3_gguf.hip"
