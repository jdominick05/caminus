// Standalone own matvecs at the dense qwen35 line's K values (Qwen3.8-27B: K 5120 = n_embd, 6144 = the Gated DeltaNet's
// value width and 24 query heads x 256, 17408 = the feed-forward), for ThinkingCap's formats: the gates and the census
// (benchmarks/own_mv_census.py) time and check them against float64 before any decode module is built on them. A new
// object, so kernels/own_mv.hip (the Qwen3-30B-A3B shapes) stays as it is. fp16 activations, 2 rows per wave.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "own_quant.h"
#include "own_quant_iq.h"
#include "own_quant_iq2s.h"
#include "own_quant_iq2xs.h"
#include "own_quant_q2k.h"
#include "own_quant_iq1m.h"
#include "own_quant_q5k.h"

// K3's form of each tile (kernels/k3_own_ops.h own_1rt_acc: runtime K, 2 rows per wave, one unit per lane per pass,
// UNR 1 behind the scheduling fences): the code a decode module runs, at any of the K values above. (The default
// form, fully unrolled at a fixed K, spills at K 17408: 256 VGPRs + 257-389 spilled.)
#define Q35_ROW0 const int row0 = (blockIdx.x * OWN_WAVES + (threadIdx.x >> 5)) * 2
extern "C" __global__ void __launch_bounds__(256) own_mv_iq4xs_k3(const uint8_t * W, int nrows, int K, const _Float16 * x16,
        float * out) {
    Q35_ROW0; if (row0 < nrows) own_rows_iq4xs_1rt<0, 2, 1, 1>(W, row0, nrows, x16, out, threadIdx.x & 31, K);
}
#define Q35_IQG(NAME, FMT) extern "C" __global__ void __launch_bounds__(256) own_mv_##NAME##_k3(const uint8_t * W,       \
        int nrows, int K, const _Float16 * x16, float * out) {                                                         \
    iq_stage(threadIdx.x, 1 << FMT);                                                                                    \
    Q35_ROW0; if (row0 < nrows) own_rows_iqg_1rt<FMT, 0, 2, 1, 1>(W, nrows, row0, nrows, x16, out, threadIdx.x & 31, K); \
}
Q35_IQG(iq3xxs, 0)
Q35_IQG(iq3s, 1)
extern "C" __global__ void __launch_bounds__(256) own_mv_iq2s_k3(const uint8_t * W, int nrows, int K, const _Float16 * x16,
        float * out) {
    iq2s_stage(threadIdx.x);
    Q35_ROW0; if (row0 < nrows) own_rows_iq2s_1rt<0, 2, 1, 1>(W, nrows, row0, nrows, x16, out, threadIdx.x & 31, K);
}
// IQ2_XS (the Qwen3.8-27B GSQ-RCO files): its own grid array (kernels/own_quant_iq2xs.h), the decode module's form
extern "C" __global__ void __launch_bounds__(256) own_mv_iq2xs_k3(const uint8_t * W, int nrows, int K, const _Float16 * x16,
        float * out) {
    iq2xs_stage(threadIdx.x);
    Q35_ROW0; if (row0 < nrows) own_rows_iq2xs_1rt<0, 2, 1, 1>(W, nrows, row0, nrows, x16, out, threadIdx.x & 31, K);
}
// Q2_K (the Qwen3.8-27B GSQ-RCO files): its record layout (caminus.engine.own_layout.q2k_records)
extern "C" __global__ void __launch_bounds__(256) own_mv_q2_k_k3(const uint8_t * W, int nrows, int K, const _Float16 * x16,
        float * out) {
    Q35_ROW0; if (row0 < nrows) own_rows_q2k_1rt<0, 2, 1, 1>(W, nrows, row0, nrows, x16, out, threadIdx.x & 31, K);
}
// IQ1_M (the Qwen3.8-27B GSQ-RCO files: one matrix): its record layout (caminus.engine.own_layout.iq1m_records)
extern "C" __global__ void __launch_bounds__(256) own_mv_iq1m_k3(const uint8_t * W, int nrows, int K, const _Float16 * x16,
        float * out) {
    Q35_ROW0; if (row0 < nrows) own_rows_iq1m_1rt<0, 2, 1, 1>(W, nrows, row0, nrows, x16, out, threadIdx.x & 31, K);
}
extern "C" __global__ void __launch_bounds__(256) own_mv_q5_k_k3(const uint8_t * W, int nrows, int K, const _Float16 * x16,
        float * out) {
    Q35_ROW0; if (row0 < nrows) own_rows_q5k_1rt<0, 2, 1, 1>(W, row0, nrows, x16, out, threadIdx.x & 31, K);
}
extern "C" __global__ void __launch_bounds__(256) own_mv_q6_k_pl_k3(const uint8_t * W, int nrows, int K, const _Float16 * x16,
        float * out) {
    Q35_ROW0; if (row0 < nrows) own_rows_q6k_planes<0, 2, 1, 1>(W, nrows, row0, nrows, x16, out, threadIdx.x & 31, K);
}

// the general rows tiles (runtime K, OWN_R rows per wave): Q5_K and Q8_0 as the decode module's OP_OWN_MV16 runs them
// the Q8_0 rows tile with UP block steps' loads per pass (own_quant.h own_rows_q8_0_1rt: bitwise own_mv_q8_0_a16)
#define Q35_Q8_1RT(UP) extern "C" __global__ void __launch_bounds__(256) own_mv_q8_0_1rt_u##UP(const uint8_t * W, int nrows,  \
        int K, const void * x, float * out) {                                                                            \
    const int row0 = (blockIdx.x * OWN_WAVES + (threadIdx.x >> 5)) * OWN_R;                                                    \
    if (row0 < nrows) own_rows_q8_0_1rt<UP>(W, row0, nrows, K, x, out, threadIdx.x & 31);                                \
}
Q35_Q8_1RT(2)
Q35_Q8_1RT(4)
Q35_Q8_1RT(8)

#define Q35_GEN(FMT) extern "C" __global__ void __launch_bounds__(256) own_mv_##FMT##_a16(const uint8_t * W, int nrows,    \
        int K, const void * x, float * out) {                                                                          \
    const int row0 = (blockIdx.x * OWN_WAVES + (threadIdx.x >> 5)) * OWN_R;                                             \
    if (row0 < nrows) own_rows_##FMT<Act16>(W, row0, nrows, K, x, nullptr, nullptr, out, threadIdx.x & 31);             \
}
Q35_GEN(q5_k)
Q35_GEN(q8_0)

// Q4_0 (the multi-token-prediction block; kernels/own_quant_q4_0.h, plane layout): nt tokens (1 .. OWN_MT) per launch,
// token j's activations at x16 + j * K, its outputs at out + j * nrows; the per-token bits are the same at every nt
// (benchmarks/q35_mtp_gacc.py checks)
#include "own_quant_q4_0.h"
struct XOStride { int s; __device__ __forceinline__ size_t operator()(int j) const { return (size_t)j * s; } };
extern "C" __global__ void __launch_bounds__(256) own_mv_q4_0_k3(const uint8_t * W, int nrows, int K, const _Float16 * x16,
        float * out, int nt) {
    Q35_ROW0;
    if (row0 >= nrows) return;
    float res[OWN_MT][2];
    own_rows_q4_0_1rt<2>(W, nrows, row0, nrows, x16, XOStride{K}, nt, threadIdx.x & 31, K, res);
    if ((threadIdx.x & 31) == 0)
        for (int j = 0; j < nt; j++)
            for (int r = 0; r < 2; r++)
                if (row0 + r < nrows) out[(size_t)j * nrows + row0 + r] = res[j][r];
}
