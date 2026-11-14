// Dense27 Q8 KV prefill: the weight GEMMs remain in their existing objects.
// The host selects only this object's Q8 append and attention entry points.
#define Q35P_KV_Q8
#include <hip/hip_runtime.h>
#include "own_kv_codec.h"
#include "q35_prefill.hip"
