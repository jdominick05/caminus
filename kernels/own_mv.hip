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
#include "own_quant_iq.h"

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

// the one-round-trip Q4_K tile at this model's K values (Qwen3-30B-A3B: 2048 = n_embd, 4096 = q heads x 128, 768 = the
// experts' ffn), 2 rows per wave, fp16 activations
#define OWN_Q4K_1RT(KK)                                                                                                     \
extern "C" __global__ void __launch_bounds__(256) own_mv_q4_k_1rt_k##KK(const uint8_t * W, int nrows, const _Float16 * x16, \
        float * out) {                                                                                                  \
    const int row0 = (blockIdx.x * OWN_WAVES + (threadIdx.x >> 5)) * 2;                                                  \
    if (row0 < nrows) own_rows_q4k_1rt<KK, 2>(W, row0, nrows, x16, out, threadIdx.x & 31);                               \
}
OWN_Q4K_1RT(2048)
OWN_Q4K_1RT(4096)
OWN_Q4K_1RT(768)

// Q6_K in the plane layout (own_rows_q6k_planes), 2 rows per wave, fp16 activations; nrows_total = the matrix's rows (the
// planes' strides), nrows = the rows to compute (<= nrows_total)
#define OWN_Q6K_PL(KK)                                                                                                      \
extern "C" __global__ void __launch_bounds__(256) own_mv_q6_k_pl_k##KK(const uint8_t * W, int nrows, const _Float16 * x16,   \
        float * out) {                                                                                                  \
    const int row0 = (blockIdx.x * OWN_WAVES + (threadIdx.x >> 5)) * 2;                                                  \
    if (row0 < nrows) own_rows_q6k_planes<KK, 2>(W, nrows, row0, nrows, x16, out, threadIdx.x & 31);                     \
}
OWN_Q6K_PL(2048)
OWN_Q6K_PL(4096)
OWN_Q6K_PL(768)

// the codebook formats' tiles (kernels/own_quant_iq.h) at the experts' K values (2048: gate / up, 768: down), RW rows per
// wave, fp16 activations
#define OWN_IQ_1RT(FMT, KK, RW)                                                                                              \
extern "C" __global__ void __launch_bounds__(256) own_mv_##FMT##_1rt_k##KK##_r##RW(const uint8_t * W, int nrows,           \
        const _Float16 * x16, float * out) {                                                                            \
    const int row0 = (blockIdx.x * OWN_WAVES + (threadIdx.x >> 5)) * RW;                                                 \
    if (row0 < nrows) own_rows_##FMT##_1rt<KK, RW>(W, row0, nrows, x16, out, threadIdx.x & 31);                          \
}
OWN_IQ_1RT(iq4xs, 2048, 2)
OWN_IQ_1RT(iq4xs, 768, 2)
// the grid formats: the workgroup stages its grid in LDS first (iq*_stage)
#define OWN_IQG_1RT(NAME, FMT, KK, RW)                                                                                       \
extern "C" __global__ void __launch_bounds__(256) own_mv_##NAME##_1rt_k##KK##_r##RW(const uint8_t * W, int nrows,          \
        const _Float16 * x16, float * out) {                                                                            \
    NAME##_stage(threadIdx.x);                                                                                          \
    const int row0 = (blockIdx.x * OWN_WAVES + (threadIdx.x >> 5)) * RW;                                                 \
    if (row0 < nrows) own_rows_iqg_1rt<FMT, KK, RW>(W, row0, nrows, x16, out, threadIdx.x & 31);                         \
}
OWN_IQG_1RT(iq3xxs, 0, 2048, 2)
OWN_IQG_1RT(iq3xxs, 0, 768, 2)
OWN_IQG_1RT(iq3s, 1, 2048, 2)
OWN_IQG_1RT(iq3s, 1, 768, 2)
OWN_IQG_1RT(iq2xxs, 2, 2048, 2)
OWN_IQG_1RT(iq2xxs, 2, 768, 2)

// a plain read of n16 16-byte words (the bandwidth ceiling of the same bytes): grid-stride, XOR into one word per thread
extern "C" __global__ void __launch_bounds__(256) own_read(const u32x4_t * __restrict__ p, long n16, uint32_t * __restrict__ sink) {
    uint32_t a = 0;
    for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n16; i += (long)gridDim.x * 256) { const u32x4_t v = p[i]; a ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (a == 0x9e3779b9u) sink[0] = a;   // (never true in practice: keeps the loads)
}
