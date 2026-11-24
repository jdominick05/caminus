// Standalone own Q2_0 matvecs (kernels/quant_q2_0.h) in K3's form (<0, 2, 1, 1>: runtime K, 2 rows per wave, one
// unit per lane per pass, the math behind scheduling fences), for the gates and the census before a decode module is
// built on them (tests/engine/test_own_q2_0.py, benchmarks/q36_gacc.py). A new object, so no existing object changes.
// The grouped kernels are kernels/ops_own.h own_op_mv_1rt's loop as one launch: each wave takes a row pair of the
// output; slot s computes rows ids[s] * R + r (r < R) of the [E * R]-row matrix, ids clamped to [0, E); a pair never spans
// two slots (R even). fp16 activations, fp32 sums.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "quant_q2_0.h"

#define Q2_RW 2   // rows per wave (kernels/ops_own.h OWN_K3_RW)

// plain rows: out[r] = W[r] . x16 for r < nrows (any matrix of contiguous rows, K % 64 == 0)
extern "C" __global__ void __launch_bounds__(256) own_mv_q2_0_k3(const uint8_t * W, int nrows, int K, const _Float16 * x16,
        float * out) {
    const int row0 = (blockIdx.x * OWN_WAVES + (threadIdx.x >> 5)) * Q2_RW;
    if (row0 < nrows) own_rows_q2_0_1rt<0, Q2_RW, 1, 1>(W, row0, nrows, x16, out, threadIdx.x & 31, K);
}

// the row pair of the output a wave takes: its slot, the expert's first row grow and the bound lim (own_op_mv_1rt's)
__device__ __forceinline__ bool q2_grouped_rows(const int * ids, int nslots, int R, int E, int & row, int & grow, int & lim) {
    row = (blockIdx.x * OWN_WAVES + (int)__builtin_amdgcn_readfirstlane(threadIdx.x >> 5)) * Q2_RW;
    if (row >= nslots * R) return false;
    const int slot = row / R, r = row - slot * R;
    int e = (int)__builtin_amdgcn_readfirstlane(ids[slot]);
    e = min(max(e, 0), E - 1);                                    // (ids clamped to the matrix's experts)
    grow = e * R + r; lim = e * R + min(R, r + Q2_RW);
    return true;
}

// the experts' gate | up on one input x16 [K]: y16[s * R + r] = fp16(silu(gate . x) * (up . x)), the K3 OWN_GU expression;
// gs / us: the two tiles' fp32 sums as well, or both nullptr (one alone is not checked: the hosts pass both or neither;
// the op test compares them with the plain rows bit for bit)
extern "C" __global__ void __launch_bounds__(256) own_moe_gu_q2_0(const uint8_t * Wg, const uint8_t * Wu, const int * ids,
        int nslots, int R, int E, int K, const _Float16 * x16, _Float16 * y16, float * gs, float * us) {
    int row, grow, lim;
    if (!q2_grouped_rows(ids, nslots, R, E, row, grow, lim)) return;
    const int lane = threadIdx.x & 31;
    float g[Q2_RW], u[Q2_RW];
    own_rows_q2_0_1rt<0, Q2_RW, 1, 1>(Wg, grow, lim, x16, nullptr, lane, K, g);
    own_rows_q2_0_1rt<0, Q2_RW, 1, 1>(Wu, grow, lim, x16, nullptr, lane, K, u);
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < Q2_RW; i++) {
            if (grow + i >= lim) continue;
            y16[row + i] = (_Float16)(g[i] / (1.0f + expf(-g[i])) * u[i]);
            if (gs) { gs[row + i] = g[i]; us[row + i] = u[i]; }
        }
    }
}

// the experts' down, each slot on its own input x16 + s * K: y[s * R + r] (fp32; the weighted slot sum is the next op's)
extern "C" __global__ void __launch_bounds__(256) own_moe_down_q2_0(const uint8_t * W, const int * ids, int nslots, int R, int E,
        int K, const _Float16 * x16, float * y) {
    int row, grow, lim;
    if (!q2_grouped_rows(ids, nslots, R, E, row, grow, lim)) return;
    const int lane = threadIdx.x & 31;
    float s[Q2_RW];
    own_rows_q2_0_1rt<0, Q2_RW, 1, 1>(W, grow, lim, x16 + (size_t)(row / R) * K, nullptr, lane, K, s);
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < Q2_RW; i++)
            if (grow + i < lim) y[row + i] = s[i];
    }
}
