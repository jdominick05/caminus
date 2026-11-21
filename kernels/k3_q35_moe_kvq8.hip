// Staged Q8 KV one-card decode for qwen35moe (Qwen3.6-35B-A3B): the same tape and ops as kernels/k3_q35_moe.hip with
// the Q8 target cache instead of F16. K3_Q35_KV_Q8 swaps OP_KV_APPEND for q35_op_kv_append_q8 and adds OP_HADAMARD
// (kernels/k3_q35_kvq8_ops.h); the split decode attention reads the wire format through ATTN_KV_TYPE.
//
// Compile only for now: this object exists to answer the register question before the engine is written, as
// k3_q35_moe.hip did. The MoE tape is the trunk's ~18 ops plus OP_ROUTER and the 9-slot add, and the dense line's Q8
// objects already sit above their F16 ones (k3_q35_gsq_kvq8 vs k3_q35_gsq: SGPR spills 71 vs 65). Budget: 96 VGPRs,
// 0 spills (tools/gates/objcheck.py).
#define K3_Q35_KV_Q8
#include "k3_q35_moe.hip"
