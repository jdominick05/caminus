// kernels/k3_q3m_lean.hip on two cards (tensor parallelism, issue #86): the same ops plus OP_XCHG (kernels/k3_xchg.h,
// its warm word and claim-time store) and OP_OWN_SLOT_SUM (the FFN partial weighted over its slots on the card, before
// the exchange). K3_Q3M_TP selects them in kernels/k3_q3m_ops.h; K3_TP is the shared exchange's own switch. The
// one-card module k3_q3m_lean is built without either: its code object is unchanged.
#define K3_TP
#define K3_Q3M_TP
// push-only exchange (no pull selects; the host refuses pull_mode on these modules): with the pull, k3_run took 96 VGPRs
// and spilled 1 (63 spilled SGPRs); push-only, 96 VGPRs, 0 spilled, 66 spilled SGPRs (k3_q3m_lean: 53). XCHG_R 2
// instead of 3 left the 66 unchanged
#define K3_XCHG_PUSH_ONLY
#include "k3_q3m_lean.hip"
