// Caminus's own quantized matvec, standalone prototypes (docs/specs/spec-own-quant.md, order of work step 3): out[r] =
// W[r] . x for the GGUF formats that carry most of the catalog's decode bytes (Q6_K, Q4_K, Q8_0), each in three activation
// precisions, written from the block layouts (no ggml-cuda code; the layouts are the file format):
//   A32  x fp32: each weight decoded to an exact fp32 value, one fma per weight (error vs fp64 ~1e-7, measured on CPU)
//   A16  x fp16: weights decoded to exact fp16 through the 1024-magic, v_dot2_f32_f16 per two weights (~1e-4)
//   A8   x int8 per 32 values with an fp32 scale (own_q8, below) and the exact fp32 block sum for the min terms;
//        v_dot4_i32_iu8 per four weights (integer sums exact; ~5e-3, the int8 bound)
// Layout of the work: one wave32 computes R = 4 rows, so each activation value a lane loads serves 4 rows; a lane owns 4
// consecutive columns of a block's sub-layout (Q6_K: 2 blocks per wave step, Q4_K: 1, Q8_0: 4). Each lane sums its groups
// in K order in fp32, then the wave adds lanes with a fixed butterfly (xor 16 .. 1): the order depends on neither the grid
// nor how many rows run, so a T-token version is bitwise per token by construction.
// Weights are read in place (GGUF block order; 2-byte aligned: Q6_K blocks are 210 B, Q8_0 34 B), 16-bit loads.
#include <hip/hip_runtime.h>

#include "own_quant.h"

// ---- the activations each mode reads, from fp32 x [K] (K % 32 == 0) -----------------------------------------------------
// own_q8: int8 per 32 values: d = amax / 127 (fp32), q = round-to-nearest-even(x / d), and the exact fp32 sum of the block's x
// (a fixed butterfly): one wave per 32-block
extern "C" __global__ void __launch_bounds__(256) own_q8(const float * __restrict__ x, int K, int8_t * __restrict__ q,
                                                         float * __restrict__ xd, float * __restrict__ xs) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= K) return;
    const float v = x[i];
    float amax = fabsf(v), s = v;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { amax = fmaxf(amax, __shfl_xor(amax, o, 32)); s += __shfl_xor(s, o, 32); }
    const float d = amax / 127.0f;
    q[i] = (int8_t)(d > 0.0f ? __builtin_rintf(v / d) : 0.0f);
    if ((i & 31) == 0) { xd[i / 32] = d; xs[i / 32] = s; }
}
extern "C" __global__ void __launch_bounds__(256) own_f16(const float * __restrict__ x, int K, _Float16 * __restrict__ h) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < K) h[i] = (_Float16)x[i];
}

#define OWN_KERNEL(FMT, MODE, ACT)                                                                                          \
extern "C" __global__ void __launch_bounds__(256) own_mv_##FMT##_##MODE(const uint8_t * W, int nrows, int K, const void * x, \
        const float * xd, const float * xs, float * out) {                                                              \
    const int row0 = (blockIdx.x * OWN_WAVES + (threadIdx.x >> 5)) * OWN_R;                                              \
    if (row0 < nrows) own_rows_##FMT<ACT>(W, row0, nrows, K, x, xd, xs, out, threadIdx.x & 31);                          \
}
#define OWN_INST(FMT) OWN_KERNEL(FMT, a32, Act32) OWN_KERNEL(FMT, a16, Act16) OWN_KERNEL(FMT, a16f, Act16F) OWN_KERNEL(FMT, a8, Act8)
OWN_INST(q6_k)
OWN_INST(q4_k)
OWN_INST(q8_0)
