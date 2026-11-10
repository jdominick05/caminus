// The lean qwen3moe decode module (kernels/k3_gguf_q3m_lean.hip) with K3_TRACE timestamps: measurement only
// (benchmarks/q3moe_trace.py: where a token's time goes, per segment, on the tape the engine decodes with).
#define K3_TRACE
#include "k3_gguf_q3m_lean.hip"
