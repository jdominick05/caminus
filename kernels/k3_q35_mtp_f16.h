// Raw F16 weights/activations, FP32 dot2 accumulation. A wave computes two
// rows; its lanes own consecutive pairs at stride 64. T=1..4 reuse weights
// while retaining each token's arithmetic independently of the pass width.
#pragma once
enum { OP_OWN_MVF16 = 56 };
__device__ __forceinline__ void q35_op_mvf16(const Seg & sg, int task) {
    const int wave = K3WAVE, lane = K3TID % 32, NT = seg_T(sg);
    const int end = min(sg.N, (task + 1) * sg.rows_per_task);
    const _Float16 * W = (const _Float16 *)sg.W, * x = (const _Float16 *)sg.a;
    float * out = (float *)sg.y;
    for (int row = task * sg.rows_per_task + 2 * wave; row < end; row += 16) {
        float sums[4][2] = {};
        for (int i = 2 * lane; i < sg.K; i += 64) {
            const h2_t a = as_h2(*(const uint32_t *)(W + (size_t)row * sg.K + i));
            const h2_t b = row + 1 < end ? as_h2(*(const uint32_t *)(W + (size_t)(row + 1) * sg.K + i)) : h2_t{};
#pragma unroll
            for (int tk = 0; tk < 4; ++tk) if (tk < NT) {
                const h2_t v = as_h2(*(const uint32_t *)(x + (size_t)tk * sg.K + i));
                sums[tk][0] = __builtin_amdgcn_fdot2(a, v, sums[tk][0], false);
                sums[tk][1] = __builtin_amdgcn_fdot2(b, v, sums[tk][1], false);
            }
        }
#pragma unroll
        for (int tk = 0; tk < 4; ++tk) if (tk < NT) {
            const float a = wave_sum(sums[tk][0]), b = wave_sum(sums[tk][1]);
            if (lane == 0) {
                out[(size_t)tk * sg.N + row] = a;
                if (row + 1 < end) out[(size_t)tk * sg.N + row + 1] = b;
            }
        }
    }
}
