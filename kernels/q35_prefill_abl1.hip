// Measurement only (benchmarks/q35_prefill_gemm.py --module q35_prefill_abl1): kernels/q35_prefill.hip with Q3P_ABL 1
// (kernels/q3m_prefill.hip: a constant B: no weight loads or decode). Wrong outputs by design; not in the runtime set.
#define Q3P_ABL 1
#include "q35_prefill.hip"
