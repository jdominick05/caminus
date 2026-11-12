// Measurement only (benchmarks/q35_prefill_gemm.py --module q35_prefill_abl2): kernels/q35_prefill.hip with Q3P_ABL 2
// (kernels/q3m_prefill.hip: no X staging). Wrong outputs by design; not in the runtime set.
#define Q3P_ABL 2
#include "q35_prefill.hip"
