// kernels/dec_q3m.hip on two cards (tensor parallelism, issue #86): the same ops plus OP_XCHG (kernels/ops_xchg.h,
// its warm word and claim-time store) and OP_OWN_SLOT_SUM (the FFN partial weighted over its slots on the card, before
// the exchange). K3_Q3M_TP selects them in kernels/ops_q3m.h; K3_TP is the shared exchange's own switch. The
// one-card module dec_q3m is built without either: its code object is unchanged.
#define K3_TP
#define K3_Q3M_TP
// push-only exchange (no pull selects; the host refuses pull_mode on these modules): with the pull, k3_run took 96 VGPRs
// and spilled 1 (63 spilled SGPRs); push-only, 96 VGPRs, 0 spilled, 66 spilled SGPRs (dec_q3m: 53). XCHG_R 2
// instead of 3 left the 66 unchanged
#define K3_XCHG_PUSH_ONLY
#include "dec_q3m.hip"
