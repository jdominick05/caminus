// OP_OWN_MV: Caminus's own quantized matvec as a K3 op (docs/specs/spec-own-quant.md), compiled into the K3_OWN modules
// only (kernels/k3_gguf_*_own.hip). Seg.W = raw GGUF blocks of N contiguous rows, Seg.a = the fp32 activation [K] (read as
// is: no q8_1 segment before it), Seg.y = fp32 out [N], K = ncols; Seg.work = the format (OWN_Q6_K / OWN_Q4_K / OWN_Q8_0),
// Seg.pad1 = the activation precision (OWN_A32: fp32; OWN_A16F: converted to fp16 in registers). A task = rows_per_task
// rows (a multiple of OWN_R); wave w takes the row groups r0 + OWN_R w, r0 + OWN_R (w + 8), ... Each row's sum is
// kernels/own_quant.h's: the same as the standalone own_mv_* kernels, whatever the grid or the task size. T = 1 only
// (the host refuses T > 1 for now).
#pragma once
#include "own_quant.h"
enum { OP_OWN_MV = 44 };
enum { OWN_Q6_K = 2, OWN_Q4_K = 4, OWN_Q8_0 = 3 };   // (the GG_* numbers of the same formats: k3.py GG_FMT)
enum { OWN_A32 = 0, OWN_A16F = 1 };

template <class ACT>
__device__ __forceinline__ void own_op_mv_a(const Seg & sg, int r0, int r1, int wave, int lane) {
    const uint8_t * W = sg.W; const void * x = sg.a; float * y = (float *)sg.y;
    for (int row = r0 + wave * OWN_R; row < r1; row += 8 * OWN_R) {
        if (sg.work == OWN_Q6_K) own_rows_q6_k<ACT>(W, row, r1, sg.K, x, nullptr, nullptr, y, lane);
        else if (sg.work == OWN_Q4_K) own_rows_q4_k<ACT>(W, row, r1, sg.K, x, nullptr, nullptr, y, lane);
        else own_rows_q8_0<ACT>(W, row, r1, sg.K, x, nullptr, nullptr, y, lane);
    }
}
__device__ __forceinline__ void own_op_mv(const Seg & sg, int t) {
    const int wave = K3WAVE, lane = K3TID % 32;
    const int r0 = t * sg.rows_per_task, r1 = min(sg.N, r0 + sg.rows_per_task);
    if (sg.pad1 == OWN_A16F) own_op_mv_a<Act16F>(sg, r0, r1, wave, lane);
    else own_op_mv_a<Act32>(sg, r0, r1, wave, lane);
}
