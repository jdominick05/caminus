// Qwen3.5-9B: own dense tape with four query heads per KV head.
// Existing 27B objects retain their six-head attention geometry.
#define K3_Q35_9B
#include "k3_q35_swift.hip"
