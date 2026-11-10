// The qwen3moe decode module (kernels/k3_gguf_q3m_own.hip) with ONLY the decode tape's ops in k3_run's dispatch:
// OP_OWN_EMB, OP_OWN_ADD_RMS16, OP_OWN_MV, OP_OWN_ATTN, OP_F32_MV, OP_ROUTER (+ the no-op). The full module carries
// every GGUF-engine op (the trunk's, the llama.cpp-format matvecs kept as benchmark yardsticks): k3_run then keeps far
// more scalars live than it has SGPRs, and every segment hop of every wave pays the reloads.
#define K3_Q3M_LEAN
#define K3_XWARM 0   // (no cross-card exchange in this tape: no warm-up stores in the waits)
#include "k3_gguf_q3m_own.hip"
