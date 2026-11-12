// The dense qwen35 line's drafter module (Swift 1.5's multi-token-prediction block: engine/q35_mtp.py). The Swift
// module's attention, feed-forward, embedding and add + norm ops in their T-token forms (K3_Q3M_VER / K3_Q35_VER: the
// catch-up passes run 1 .. 4 rows of the block), the Q4_0 one-round-trip tile (kernels/own_quant_q4_0.h) and the Q6_K
// head, and OP_OWN_ARGMAX, whose key the next draft step's embedding reads on the device. No Gated DeltaNet op and no
// rows tile in its register allocation (K3_Q35_MTP). A module of its own: every other code object is unchanged.
#define K3_Q35_MTP
#define K3_Q3M_VER
#define K3_Q35_VER
#include "k3_q35_swift.hip"
