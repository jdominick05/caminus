// ThinkingCap's multi-token-prediction drafter (engine/q35_mtp.py): its prediction block (all Q6_K: the planes' T-token
// tiles), the Q8_0 head as a rows tile, the IQ2_S embedding (a chained step reads OP_OWN_ARGMAX's key on the device) on
// ThinkingCap's module (kernels/ops_q35_tcap.h); the decode and verify modules' .text unchanged: all under K3_Q35_MTP.
#define K3_Q35_MTP
#define K3_Q3M_VER
#define K3_Q35_VER
#include "dec_q35_tcap.hip"
