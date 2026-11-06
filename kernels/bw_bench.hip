// K3 engine cost model, part 2: read bandwidth by working-set size on gfx1100 (does the hot set stay in the 96 MB
// Infinity Cache, and how fast can b128 loads drain it). Each thread streams 16-byte loads over its slice of a buffer
// `passes` times, xor-folding into a register (written only on a magic value). `nt` = nontemporal loads (slc dlc).
#include <hip/hip_runtime.h>
#include <stdint.h>
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

extern "C" __global__ void __launch_bounds__(256) bw_read(const u32x4 * __restrict__ buf, size_t n16, int passes, int nt, uint32_t * sink) {
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
    u32x4 acc = {0, 0, 0, 0};
    for (int p = 0; p < passes; p++) {
        for (size_t i = tid; i < n16; i += stride) {
            const u32x4 v = nt ? __builtin_nontemporal_load(buf + i) : buf[i];
            acc ^= v;
        }
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9E3779B9u) *sink = acc.x;
}
