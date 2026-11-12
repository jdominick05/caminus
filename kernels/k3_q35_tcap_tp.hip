// kernels/k3_q35_tcap.hip on two cards (tensor parallelism: engine/q35_dense_tp.py): the same ops plus OP_XCHG
// (kernels/k3_xchg.h), its warm word and claim-time store. The one-card module k3_q35_tcap is built without K3_TP: its
// code object is unchanged.
#define K3_TP
#include "k3_q35_tcap.hip"
