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
// The one-round-trip tiles (kernels/own_quant.h own_rows_q4k_1rt / own_rows_q6k_planes), fp16 activations, for the
// Qwen3-30B-A3B shapes (K 2048 / 4096 / 768), with expert grouping. Seg.pad1 = OWN_A16_1RT | flags:
//   Seg.a      fp16 activations: [K], or [slots][K] with OWN_X_PER_SLOT (the experts' down: each expert its own input)
//   Seg.asum   int32 ids [slots] (OWN_GROUPED): slot s computes rows ids[s] * Seg.pad2 + r of the matrix, r < Seg.pad2,
//              written to y[s * Seg.pad2 + r]; the matrix is the whole [experts * pad2] row set (Q6_K: its plane layout over
//              all of them). Without OWN_GROUPED: rows r of N, y[r]
//   Seg.out_f32 the matrix's total rows (Q6_K planes' strides)
// A task = rows_per_task rows of the output (a multiple of 2 * 8: two rows per wave), never spanning two slots
// (rows_per_task divides pad2). Ids are clamped to [0, experts) on read (Seg.dep2p1 unused here; Seg.rs unused).
enum { OWN_A16_1RT = 2, OWN_GROUPED = 4, OWN_X_PER_SLOT = 8 };
#ifndef OWN_K3_RW
#define OWN_K3_RW 2   // rows per wave in K3 (2: the activation loads shared by two rows)
#endif
// rows grow .. lim - 1 of the matrix (total rows: the Q6_K planes' strides); row grow + i -> out[i]. One body per format
// for every K (the tiles' runtime-K form): three K-specialised bodies cost K3 69 spilled VGPRs.
__device__ __forceinline__ void own_1rt_rows(const Seg & sg, int grow, int lim, int total, const _Float16 * x, float * out,
                                             int lane) {
    // own_rows_* write out[row0 + i]: hand them out - grow so that row grow + i lands in out[i]
#ifndef OWN_K3_NO_Q4   // (OWN_K3_NO_*: register-budget experiments only)
    if (sg.work == OWN_Q4_K) own_rows_q4k_1rt<0, OWN_K3_RW, 1, 1>(sg.W, grow, lim, x, out - grow, lane, sg.K);
#endif
#ifndef OWN_K3_NO_Q6
    if (sg.work == OWN_Q6_K) own_rows_q6k_planes<0, OWN_K3_RW, 1, 1>(sg.W, total, grow, lim, x, out - grow, lane, sg.K);
#endif
}
__device__ __forceinline__ void own_op_mv_1rt(const Seg & sg, int t, int wave, int lane) {
    const bool grouped = sg.pad1 & OWN_GROUPED, per_slot = sg.pad1 & OWN_X_PER_SLOT;
    const int r0 = t * sg.rows_per_task, r1 = min(sg.N, r0 + sg.rows_per_task);
    const int re = grouped ? sg.pad2 : sg.N, total = sg.out_f32;
    for (int row = r0 + OWN_K3_RW * wave; row < r1; row += 8 * OWN_K3_RW) {   // (8 waves x OWN_K3_RW rows)
        const int slot = grouped ? row / re : 0, r = row - slot * re;
        int e = grouped ? (int)__builtin_amdgcn_readfirstlane(((const int *)sg.asum)[slot]) : 0;
        e = min(max(e, 0), total / re - 1);                           // (ids clamped to the matrix's experts)
        const int grow = e * re + r, lim = e * re + min(re, r + min(OWN_K3_RW, r1 - row));
        const _Float16 * x = (const _Float16 *)sg.a + (per_slot ? (size_t)slot * sg.K : 0);
        float * out = (float *)sg.y + row;                            // (= y[slot * re + r])
        own_1rt_rows(sg, grow, lim, total, x, out, lane);
    }
}
__device__ __forceinline__ void own_op_mv(const Seg & sg, int t) {
    const int wave = K3WAVE, lane = K3TID % 32;
    // (the row, slot, expert and base pointers are wave-uniform: readfirstlane keeps them in SGPRs)
    if (sg.pad1 & OWN_A16_1RT) { own_op_mv_1rt(sg, t, (int)__builtin_amdgcn_readfirstlane(K3TID / 32), lane); return; }
    const int r0 = t * sg.rows_per_task, r1 = min(sg.N, r0 + sg.rows_per_task);
    if (sg.pad1 == OWN_A16F) own_op_mv_a<Act16F>(sg, r0, r1, wave, lane);
    else own_op_mv_a<Act32>(sg, r0, r1, wave, lane);
}
// OP_OWN_F16: y fp16 [N] = x fp32 [N] (round to nearest even); a task = rows_per_task values
enum { OP_OWN_F16 = 45 };
__device__ __forceinline__ void own_op_f16(const Seg & sg, int t) {
    const int i0 = t * sg.rows_per_task, i1 = min(sg.N, i0 + sg.rows_per_task);
    for (int i = i0 + (int)K3TID; i < i1; i += blockDim.x) ((_Float16 *)sg.y)[i] = (_Float16)((const float *)sg.a)[i];
}
// OP_OWN_SWIGLU16: y16[i] = fp16(silu(g[i]) * u[i]), i < N; Seg.a = g, Seg.W = u (fp32: the gate / up matvecs' outputs),
// Seg.y = y16 (the down matvec's fp16 input). silu(g) = g / (1 + exp(-g)). A task = rows_per_task values.
// OP_OWN_MOE_SUM: x[i] += sum over slots s < pad1 of w[s] * d[s * N + i] (fma in slot order; Seg.gs = w, or nullptr:
// every weight 1, i.e. x += d for pad1 = 1: the attention's residual add). Seg.a = d, Seg.y = x. A task = rows_per_task.
enum { OP_OWN_SWIGLU16 = 46, OP_OWN_MOE_SUM = 47 };
__device__ __forceinline__ void own_op_swiglu16(const Seg & sg, int t) {
    const int i0 = t * sg.rows_per_task, i1 = min(sg.N, i0 + sg.rows_per_task);
    const float * g = (const float *)sg.a; const float * u = (const float *)sg.W; _Float16 * y = (_Float16 *)sg.y;
    for (int i = i0 + (int)K3TID; i < i1; i += blockDim.x) { const float gv = g[i]; y[i] = (_Float16)(gv / (1.0f + expf(-gv)) * u[i]); }
}
__device__ __forceinline__ void own_op_moe_sum(const Seg & sg, int t) {
    const int i0 = t * sg.rows_per_task, i1 = min(sg.N, i0 + sg.rows_per_task), S = sg.pad1;
    const float * d = (const float *)sg.a; const float * w = (const float *)sg.gs; float * x = (float *)sg.y;
    for (int i = i0 + (int)K3TID; i < i1; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < S; s++) acc = w ? __builtin_fmaf(w[s], d[(size_t)s * sg.N + i], acc) : acc + d[(size_t)s * sg.N + i];
        x[i] += acc;
    }
}
