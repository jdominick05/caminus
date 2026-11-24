// kernels/dec_q35_swift.hip on two cards (tensor parallelism: engine/q35_dense_tp.py): the same ops plus OP_XCHG
// (kernels/ops_xchg.h), its warm word and claim-time store. The one-card module dec_q35_swift is built without K3_TP: its
// code object is unchanged.
#define K3_TP
#include "dec_q35_swift.hip"
