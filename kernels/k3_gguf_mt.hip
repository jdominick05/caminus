// the GGUF engine for multi-token passes (speculative verify, T > 1): the same k3_run with the T > 1 fast paths compiled
// in (K3_MT: kernels/k3_gguf_ops.h). A separate module, so the one-token k3_run keeps its code (adding them there cost
// the one-token pass 0.16 ms through layout / scheduling alone); k3.Engine picks it for T > 1 (k3.module_for)
#define K3_MT
#include "k3_gguf.hip"
