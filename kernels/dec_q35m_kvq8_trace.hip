// kernels/dec_q35m_kvq8.hip with K3_TRACE (per-segment timestamps; measurement only, not in the runtime set:
// benchmarks/q3moe_trace.py --engine q35moe)
#define K3_TRACE
#include "dec_q35m_kvq8.hip"
