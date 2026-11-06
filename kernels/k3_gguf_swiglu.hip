// test build of the REAP-308 GGUF engine with OP_GG_Q8_1 mode 2 (K3_Q8_SWIGLU, the swiglu glue) compiled in: the forced
// unfused shared expert (two GG_MV + the glue instead of OP_GG_GU_DENSE) on REAP-308 runs on this module (k3.Engine
// module 'k3_gguf_swiglu'; T > 1: k3_gguf_mt_swiglu). The production k3_gguf modules leave the glue out (its never-taken
// branch cost their k3_run ~0.04-0.07 ms per token)
#define K3_Q8_SWIGLU
#include "k3_gguf.hip"
