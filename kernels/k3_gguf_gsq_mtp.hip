// the MTP block's engine (python/caminus/mtp.py): the GSQ-RCO Coder's one-token module plus the fp32 swiglu stage of
// OP_BF16_MV (Seg.work = 2, kernels/k3_gguf_ops.h bx_fill_swiglu) for the sidecar's BF16 shared expert. A separate code
// object so the trunk's k3_gguf_gsq* stay byte for byte the same
#define K3_MTP_SWIGLU
#include "k3_gguf_gsq.hip"
