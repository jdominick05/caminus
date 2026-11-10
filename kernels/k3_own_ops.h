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
enum { OWN_A16_1RT = 2, OWN_GROUPED = 4, OWN_X_PER_SLOT = 8, OWN_GU = 16 };
#ifndef OWN_K3_RW
#define OWN_K3_RW 2   // rows per wave in K3 (2: the activation loads shared by two rows)
#endif
// rows grow .. lim - 1 of the matrix W (total rows: the Q6_K planes' strides): res[i] = row grow + i's sum, in every lane.
// One body per format for every K (the tiles' runtime-K form): three K-specialised bodies cost K3 69 spilled VGPRs.
__device__ __forceinline__ void own_1rt_acc(const uint8_t * W, const Seg & sg, int grow, int lim, int total,
                                            const _Float16 * x, int lane, float * res) {
#ifndef OWN_K3_NO_Q4   // (OWN_K3_NO_*: register-budget experiments only)
    if (sg.work == OWN_Q4_K) own_rows_q4k_1rt<0, OWN_K3_RW, 1, 1>(W, grow, lim, x, nullptr, lane, sg.K, res);
#endif
#ifndef OWN_K3_NO_Q6
    if (sg.work == OWN_Q6_K) own_rows_q6k_planes<0, OWN_K3_RW, 1, 1>(W, total, grow, lim, x, nullptr, lane, sg.K, res);
#endif
}
// OWN_GU (the qwen3moe modules): the segment computes TWO matrices of the same shape and format on the same input, the
// experts' gate (Seg.W) and up (Seg.aux), and stores y16[row] = fp16(silu(gate . x) * (up . x)) (Seg.y fp16: the down
// matvec's input; OP_OWN_SWIGLU16's expression on the two tiles' sums). One segment instead of three: 'up' no longer
// queues behind 'gate' (every workgroup held a gate task: its first workgroup arrived 12-13 us after the router, XT
// trace) and the SwiGLU's hand-off is gone. A wave takes its row pairs of the gate, then the same pairs of the up: ONE
// loop over (matrix, pair) with one tile call. A matrix loop inside the row loop keeps the row's state live across the
// first tile: 2 spilled VGPRs in k3_run, whether the gate sums wait in registers (5), as scalars (2) or in LDS (2).
// The gate sums wait in LDS, indexed by the row mod OWN_GU_RPT (rows_per_task <= OWN_GU_RPT, a multiple of 16: host-checked).
enum { OWN_GU_RPT = 64 };
__shared__ float OWN_GU_G[OWN_GU_RPT];   // (lane 0 of the wave that owns a row writes and reads its slot)
__device__ __forceinline__ void own_op_mv_1rt(const Seg & sg, int t, int wave, int lane) {
    const bool grouped = sg.pad1 & OWN_GROUPED, per_slot = sg.pad1 & OWN_X_PER_SLOT;
#ifdef K3_OWN_Q3M
    const bool gu = sg.pad1 & OWN_GU;
#else
    const bool gu = false;
#endif
    const int rpt = sg.rows_per_task, r0 = t * rpt, r1 = min(sg.N, r0 + rpt);
    const int re = grouped ? sg.pad2 : sg.N, total = sg.out_f32;
    // it < 0: the gate's pairs (OWN_GU only), it >= 0: the up's, or the one matrix's. One bound for both forms: a
    // separate trip count (2 rpt | rpt) is one more value live across the tile (1 spilled VGPR in the _trace build)
    for (int it = OWN_K3_RW * wave - (gu ? rpt : 0); it < rpt; it += 8 * OWN_K3_RW) {   // (8 waves x OWN_K3_RW rows)
        const bool m = gu && it >= 0;                                  // (the second matrix)
        const int row = r0 + (it < 0 ? it + rpt : it);
        if (row >= r1) continue;
        const int slot = grouped ? row / re : 0, r = row - slot * re;
        int e = grouped ? (int)__builtin_amdgcn_readfirstlane(((const int *)sg.asum)[slot]) : 0;
        e = min(max(e, 0), total / re - 1);                           // (ids clamped to the matrix's experts)
        const int grow = e * re + r, lim = e * re + min(re, r + min(OWN_K3_RW, r1 - row));
        const _Float16 * x = (const _Float16 *)sg.a + (per_slot ? (size_t)slot * sg.K : 0);
        float res[OWN_K3_RW];
        own_1rt_acc(m ? (const uint8_t *)sg.aux : sg.W, sg, grow, lim, total, x, lane, res);
        if (lane == 0) {
#pragma unroll
            for (int i = 0; i < OWN_K3_RW; i++) {
                if (grow + i >= lim) continue;
                if (!gu) ((float *)sg.y)[row + i] = res[i];          // (= y[slot * re + r + i])
                else if (!m) OWN_GU_G[(row + i) & (OWN_GU_RPT - 1)] = res[i];   // (a task's rows: distinct mod OWN_GU_RPT)
                else {
                    const float gv = OWN_GU_G[(row + i) & (OWN_GU_RPT - 1)];
                    ((_Float16 *)sg.y)[row + i] = (_Float16)(gv / (1.0f + expf(-gv)) * res[i]);
                }
            }
        }
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
#ifdef K3_OWN_Q3M   // the qwen3moe engine's ops (kernels/k3_gguf_q3m_own.hip): out of the other own module's register allocation
// OP_OWN_EMB: x fp32 [N] = the Q4_K row tok of the embedding matrix W (raw GGUF blocks, Seg.K = row bytes), tok =
// ((const int *)Seg.a)[0] clamped to [0, Seg.pad2) (the vocabulary). The arithmetic is caminus.own_dequant's, rounded
// the same way (no fma: (d * sc) * q - dmin * m, each product and the difference rounded to fp32), so the device row is
// bitwise the host decoder's. One task.
enum { OP_OWN_EMB = 48 };
__device__ __forceinline__ void own_op_emb(const Seg & sg) {
    const int tok = min(max(((const int *)sg.a)[0], 0), sg.pad2 - 1);
    const uint8_t * row = sg.W + (size_t)tok * sg.K;
    float * x = (float *)sg.y;
    for (int i = (int)K3TID; i < sg.N; i += blockDim.x) {
        const uint8_t * blk = row + (size_t)(i >> 8) * 144;
        const int j = i & 255, sb = j >> 5, l = j & 31, is = sb >> 1, hi = sb & 1;
        const uint8_t * s = blk + 4;
        const uint32_t sc = sb < 4 ? (s[sb] & 63u) : ((s[sb + 4] & 0xfu) | ((s[sb - 4] >> 6) << 4));
        const uint32_t m = sb < 4 ? (s[sb + 4] & 63u) : ((s[sb + 4] >> 4) | ((s[sb] >> 6) << 4));
        const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(blk[0] | (blk[1] << 8)));
        const float dm = (float)__builtin_bit_cast(_Float16, (uint16_t)(blk[2] | (blk[3] << 8)));
        const float q = (float)((blk[16 + is * 32 + l] >> (4 * hi)) & 15);
        x[i] = __fsub_rn(__fmul_rn(__fmul_rn(d, (float)sc), q), __fmul_rn(dm, (float)m));
    }
}
// OP_OWN_ADD_RMS16: one task (one workgroup) = the residual add, the RMS norm and the fp16 copy in one segment:
//   x[i] += sum over slots s < pad1 of w[s] * d[s * N + i]   (fma in slot order from 0; pad1 = 0: no add; w nullptr: + d)
//   h[i] = (x[i] * rsqrt(sum x^2 / N + eps)) * nw[i];   h16[i] = fp16(h[i])
// Seg.a = d, Seg.gs = w (nullptr: weights 1), Seg.y = x (in / out), Seg.W = nw, Seg.asum = h (fp32), Seg.aux = h16,
// Seg.rs = eps, Seg.N = the row length = 256 * NI, NI <= 8 (host-checked): a thread owns elements tid + 256 j.
// Every load of a step is issued before its arithmetic (d: OWN_ADD_S slots x 8 elements per round trip, then x; nw
// during the reduction; a slot past pad1 rereads the last one and is not added). Measured (XT, trace): the three ops this replaces
// (OP_OWN_MOE_SUM, OP_RMS_MUL's general loop, OP_OWN_F16) ran one load per loop turn, ~0.35 us each in series: 6.4 +
// 6.2 + 2.3 us; the same loops in one segment were no faster (7.15 vs 7.07 ms per token).
enum { OP_OWN_ADD_RMS16 = 49 };
#ifndef OWN_ADD_S
#define OWN_ADD_S 1   // slots per round trip of the add (x 8 elements of loads in flight); 2 spilled k3_run (15 VGPRs), 4: 23
#endif
__device__ __forceinline__ void own_op_add_rms16(const Seg & sg) {
    __shared__ float part[8];
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, N = sg.N, S = sg.pad1, NI = N >> 8;
    const float * d = (const float *)sg.a; const float * w = (const float *)sg.gs; const float * nw = (const float *)sg.W;
    float * x = (float *)sg.y; float * h = (float *)sg.asum; _Float16 * h16 = (_Float16 *)sg.aux;
    float acc[8], xv[8];
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = 0.0f;
#pragma unroll
    for (int s0 = 0; s0 < 8; s0 += OWN_ADD_S) {
        if (s0 >= S) continue;
        float wv[OWN_ADD_S], dv[OWN_ADD_S][8];
#pragma unroll
        for (int c = 0; c < OWN_ADD_S; c++) {
            const int sl = min(s0 + c, S - 1);
            wv[c] = w ? w[sl] : 1.0f;
#pragma unroll
            for (int j = 0; j < 8; j++) dv[c][j] = d[(size_t)sl * N + tid + 256 * min(j, NI - 1)];
        }
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int c = 0; c < OWN_ADD_S; c++) {
            if (s0 + c >= S) continue;
#pragma unroll
            for (int j = 0; j < 8; j++) acc[j] = w ? __builtin_fmaf(wv[c], dv[c][j], acc[j]) : acc[j] + dv[c][j];
        }
        __builtin_amdgcn_sched_barrier(0);
    }
#pragma unroll
    for (int j = 0; j < 8; j++) xv[j] = x[tid + 256 * min(j, NI - 1)];
    if (S > 0) {
#pragma unroll
        for (int j = 0; j < 8; j++) { xv[j] += acc[j]; if (j < NI) x[tid + 256 * j] = xv[j]; }
    }
    __builtin_amdgcn_sched_barrier(0);
    float wn[8];
#pragma unroll
    for (int j = 0; j < 8; j++) wn[j] = nw[tid + 256 * min(j, NI - 1)];   // (in flight during the reduction below)
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; j++) if (j < NI) ss = __builtin_fmaf(xv[j], xv[j], ss);
    ss = warp_reduce_sum<32>(ss); if (lane == 0) part[wave] = ss; __syncthreads();
    float tot = 0.0f; for (int q = 0; q < 8; q++) tot += part[q];
    const float scale = rsqrtf(tot / N + sg.rs);
#pragma unroll
    for (int j = 0; j < 8; j++)
        if (j < NI) { const float v = (xv[j] * scale) * wn[j]; h[tid + 256 * j] = v; h16[tid + 256 * j] = (_Float16)v; }
    __syncthreads();
}
#endif
