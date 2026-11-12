// Measurement only (benchmarks/q35_prefill_gemm.py --module q35_prefill_abl3): kernels/q35_prefill.hip with Q3P_ABL 3
// (kernels/q3m_prefill.hip: the WMMAs accumulate into acc directly: no scale step). Wrong outputs by design; not in the runtime set.
#define Q3P_ABL 3
#include "q35_prefill.hip"
