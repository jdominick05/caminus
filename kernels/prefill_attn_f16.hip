// F16 target KV specialization of the Coder prefill attention core. The host
// selects only append and the identity/SEL attention pairs from this object;
// the QSA indexer's raw key append, pooling and scoring keep the Q8 object.
// No transform is implicit: Q/K/V and output rotations are the host's choice.
#define PREFILL_KV_F16 1
#include "prefill_attn.hip"
