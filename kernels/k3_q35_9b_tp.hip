// Existing dense tensor parallelism with Qwen3.5-9B's group4 attention.
// Plain decode only: the one-card and verify objects remain unchanged.
#define K3_TP
#include "k3_q35_9b.hip"
