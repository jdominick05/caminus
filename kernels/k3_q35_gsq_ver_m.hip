// The plain Qwen3.8-27B GSQ-RCO files' IQ1_M verify module: kernels/k3_q35_gsq_ver.hip's build with K3_Q35_GSQ_M, whose
// T-token matvec (kernels/k3_q35t_ops.h q35t_op_mv) is the IQ1_M loop alone (kernels/own_quant_gsq_mt.h). The files hold one
// IQ1_M matrix; engine/q35_dense_verify.py runs it in its own launch between the verify tape's two halves, because with
// it in k3_q35_gsq_ver's dispatch k3_run took 97 to 102 VGPRs (budget 96). The host's caps give this module that format only.
#define K3_Q35_GSQ_M
#include "k3_q35_gsq_ver.hip"
