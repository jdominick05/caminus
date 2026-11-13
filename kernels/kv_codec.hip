// Guarded standalone append for the staged own-code Q8 KV path. This does not
// enable a model's cache format: attention / transforms / persistence must agree
// before a model may use it. The same block codec is compiled in the host gate.
#include <hip/hip_runtime.h>
#include "own_kv_codec.h"

// Error bits: 1 position span, 2 numerical domain, 4 layout. Any error makes the
// WHOLE append unusable; valid blocks may already be written. Invalid blocks and
// invalid rows remain untouched. Never use a failed cache in attention.
extern "C" __global__ void __launch_bounds__(256) own_kv_q8_append(
        const float * k, int ldk, const float * v, int ldv, int M,
        uint8_t * cache, int pos0, int rows, int * err) {
    if (M < 32 || M % 32 || ldk < M || ldv < M || rows < 1) {
        if (threadIdx.x == 0) atomicOr(err, 4);
        return;
    }
    // Widen before the sum: a malformed position must not wrap into a valid row.
    const int64_t pos = (int64_t)pos0 + (int64_t)blockIdx.x;
    if (pos < 0 || pos >= rows) {
        if (threadIdx.x == 0) atomicOr(err, 1);
        return;
    }
    const size_t tt = blockIdx.x;
    auto * dst = (OwnKvQ8Block *)(cache + (size_t)pos * own_kv_q8_row_bytes(M));
    const int nb = M / 32;
    for (int b = threadIdx.x; b < 2 * nb; b += blockDim.x) {
        const float * x = (b < nb ? k + tt * ldk : v + tt * ldv) + (size_t)(b % nb) * 32;
        if (!own_kv_q8_encode(x, dst + b)) atomicOr(err, 2);
    }
}
