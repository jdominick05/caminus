// the GSQ-RCO Coder's one-token engine with the QSA indexer ops (kernels/ops_qsa.h: OP_IDX_APPEND, OP_QSA_SCORE,
// OP_QSA_SELECT; docs/specs/spec-qsa.md). A separate code object, like the drafter's and the MTP block's: inlined into the
// trunk's own dec_q4x_gsq they cost its tapes ~0.7 ms per token at T = 1 / 0.3 ms per T = 4 pass (k3_run's register
// allocation moved; as a call inside the segment loop its SGPR spills went to scratch memory: 18 ms per token), so the
// trunk modules stay byte for byte the same and caminus.k3.Engine runs tapes with QSA ops on this one (module_for / build)
#define K3_QSA
#include "dec_q4x_gsq.hip"
