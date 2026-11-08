// The batched prefill's GDN half for qwen35moe (python/caminus/prefill35.py): prefill_gdn.hip with the SiLU output gate
// (pf_gdn_norm: nrm * silu(z), OP_GDN_NORM under K3_GDN_SILU). Value head h reads key head h % nk (tiled) in both.
#define PF_GDN_SILU
#include "prefill_gdn.hip"
