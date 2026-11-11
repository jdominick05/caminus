// OP_OWN_MV: Caminus's own quantized matvec as a K3 op (docs/specs/spec-own-quant.md), compiled into the K3_OWN modules
// only (kernels/k3_gguf_*_own.hip). Seg.W = raw GGUF blocks of N contiguous rows, Seg.a = the fp32 activation [K] (read as
// is: no q8_1 segment before it), Seg.y = fp32 out [N], K = ncols; Seg.work = the format (OWN_Q6_K / OWN_Q4_K / OWN_Q8_0),
// Seg.pad1 = the activation precision (OWN_A32: fp32; OWN_A16F: converted to fp16 in registers). A task = rows_per_task
// rows (a multiple of OWN_R); wave w takes the row groups r0 + OWN_R w, r0 + OWN_R (w + 8), ... Each row's sum is
// kernels/own_quant.h's: the same as the standalone own_mv_* kernels, whatever the grid or the task size. T = 1 only
// (the host refuses T > 1 for now).
#pragma once
#include "own_quant.h"
#ifdef K3_Q3M_VER
#include "own_quant_mt.h"   // the T-token tiles: a pass's weight words held, each token's sums in turn
#endif
#ifdef K3_Q3M_IQ
#include "own_quant_iq.h"   // the codebook formats' tiles (the IQ study module, kernels/k3_q3m_iq.hip)
#endif
enum { OP_OWN_MV = 44 };
enum { OWN_IQ3_XXS = 0, OWN_IQ4_XS = 6, OWN_IQ3_S = 8, OWN_IQ2_XXS = 11 };   // (K3_Q3M_IQ only; GG_* numbers, IQ2_XXS: its own)
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
// The verify module (K3_Q3M_VER, kernels/k3_q3m_ver.hip: T tokens per pass) only:
//   Seg.pad1 bits 8 .. 11 = tokens NT (0: one), without OWN_GROUPED / OWN_GU: token tk reads its activations at
//     Seg.a + tk * xs (fp16 values) and writes its rows at Seg.y + tk * ys (fp32 values); xs / ys = the low / high half of
//     Seg.gs. One tile per row pair (own_quant_mt.h): its weight words are loaded once and every token's sums run on
//     them, each the decode's statements (the same bits).
//   OWN_X_PER_TOKEN (with OWN_GROUPED, without OWN_X_PER_SLOT): 8 slots per token, slot s reads the input of token s / 8
//     at Seg.a + (s / 8) * K (the experts' gate | up of every token as one segment).
//   OWN_UNION (with OWN_GROUPED, at most 32 slots): slots that hold the same expert id are computed together by the
//     first of them (one read of the expert's rows, one sum per member slot on the member's own input and into its own
//     rows); the other member slots' row pairs do nothing. A wave finds the members itself: lane l compares slot l's id.
// A task index is the token for the one-task ops below (OP_OWN_EMB, OP_OWN_ADD_RMS16).
enum { OWN_X_PER_TOKEN = 32, OWN_UNION = 64 };
// (every T-token statement sits under K3_Q3M_VER: the other modules compile the text they compiled before)
#ifdef K3_Q3M_VER
typedef __attribute__((address_space(3))) float AS3F;
#define OWN_TK_ARG , t
#else
#define OWN_TK_ARG
#endif
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
#ifdef K3_Q3M_IQ   // (IQ4_XS: blocks in place; the grid formats: the record layout, and the grids own_iq_stage put in LDS)
    // The lane through an empty asm inside each branch: what a tile derives from the lane (its unit, the offsets of its
    // loads) is then computed in the branch. Derived from the one `lane`, the compiler hoisted every tile's values out
    // of the row loop, where they were live across whichever tile ran: 7 spilled VGPRs in k3_run with the four tiles.
#ifndef OWN_K3_NO_IQ4
    if (sg.work == OWN_IQ4_XS) {
        int l = lane; asm volatile("" : "+v"(l));
        own_rows_iq4xs_1rt<0, OWN_K3_RW, 1, 1>(W, grow, lim, x, nullptr, l, sg.K, res);
    }
#endif
#ifdef OWN_K3_IQG_ONE   // (a timing experiment, never a shipped build: the grid body with ONE format fixed at compile time,
                        //  0 IQ3_XXS / 1 IQ3_S / 2 IQ2_XXS, OWN_K3_IQG_UP units per pass, no lane copy)
    if (sg.work == OWN_IQ3_XXS || sg.work == OWN_IQ3_S || sg.work == OWN_IQ2_XXS)
        own_rows_iqg_1rt<OWN_K3_IQG_ONE, 0, OWN_K3_RW, OWN_K3_IQG_UP, 1>(W, total, grow, lim, x, nullptr, lane, sg.K, res);
#define OWN_K3_NO_IQG
#endif
#ifndef OWN_K3_NO_IQG   // (one body for the three grid formats: three were 13 more spilled VGPRs in k3_run)
    if (sg.work == OWN_IQ3_XXS || sg.work == OWN_IQ3_S || sg.work == OWN_IQ2_XXS) {
        int l = lane;
#ifndef OWN_K3_IQG_NOCOPY   // (a timing experiment: the runtime-format body without the opaque lane copy)
        asm volatile("" : "+v"(l));
#endif
        own_rows_iqg_1rt<3, 0, OWN_K3_RW, 1, 1>(W, total, grow, lim, x, nullptr, l, sg.K, res,
                                                sg.work == OWN_IQ3_XXS ? 0 : sg.work == OWN_IQ3_S ? 1 : 2);
    }
#endif
#endif
}
#ifdef K3_Q3M_IQ
// k3_run's launch hook: every workgroup stages the grids the tape's first segment asks for (Seg.out_f32 of the tape's
// OP_OWN_EMB: OWN_STAGE_* bits, set by the host from the formats on the tape; 0: nothing is staged and nothing waits;
// IQ2_XXS never with an IQ3 grid: they share the LDS words, own_quant_iq.h)
enum { OWN_STAGE_IQ3_XXS = 1, OWN_STAGE_IQ3_S = 2, OWN_STAGE_IQ2_XXS = 4 };
__device__ __forceinline__ void own_iq_stage(const Seg * tape) {
    const int f = tape[0].op == 48 ? tape[0].out_f32 : 0;   // (48: OP_OWN_EMB)
    if (f) iq_stage((int)K3TID, f);
}
#define K3_LAUNCH_HOOK(tape) own_iq_stage(tape)
#endif
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
#ifdef K3_Q3M_VER
__shared__ float OWN_GU_GM[OWN_MT * OWN_GU_RPT];   // (the same per member of a pass: [member][row mod OWN_GU_RPT])
#endif
__device__ __forceinline__ void own_op_mv_1rt(const Seg & sg, int t, int wave, int lane) {
    const bool grouped = sg.pad1 & OWN_GROUPED, per_slot = sg.pad1 & OWN_X_PER_SLOT;
#ifdef K3_OWN_Q3M
    const bool gu = sg.pad1 & OWN_GU;
#else
    const bool gu = false;
#endif
    const int rpt = sg.rows_per_task, r0 = t * rpt, r1 = min(sg.N, r0 + rpt);
    const int re = grouped ? sg.pad2 : sg.N, total = sg.out_f32;
#ifdef K3_Q3M_VER
    const int NT = max(1, (int)__builtin_amdgcn_readfirstlane((sg.pad1 >> 8) & 15));
    const uint64_t st = (uint64_t)sg.gs;
    const size_t xs = (uint32_t)__builtin_amdgcn_readfirstlane((int)(uint32_t)st), ys = (uint32_t)__builtin_amdgcn_readfirstlane((int)(uint32_t)(st >> 32));
#endif
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
#ifdef K3_Q3M_VER
        (void)x; (void)res;
        const bool uni = sg.pad1 & OWN_UNION;
        uint32_t mask = 0;
        if (uni) {                                                     // the slots that hold this slot's expert
            const int S = sg.N / re;
            const int idl = ((const int *)sg.asum)[min(lane, S - 1)];
            const int er = (int)__builtin_amdgcn_readfirstlane(((const int *)sg.asum)[slot]);
            mask = (uint32_t)__builtin_amdgcn_uicmp((unsigned)(lane < S ? idl : ~er), (unsigned)er, 32);   // (32: ICMP_EQ)
            if (mask & ((1u << slot) - 1u)) continue;                  // (an earlier slot computes these rows for this one)
        }
        const int nt = uni ? min((int)OWN_MT, (int)__builtin_popcount(mask)) : grouped ? 1 : NT;
        const auto member = [&](int j) -> int {                        // the slot of member j (not OWN_UNION: this slot)
            if (!uni) return slot;
            uint32_t mk = mask;
            for (int k = 0; k < j; k++) mk &= mk - 1u;
            return mk ? (int)__builtin_ctz(mk) : slot;
        };
        const auto xoff = [&](int j) -> size_t {                       // member / token j's activations, from Seg.a
            if (!grouped) return (size_t)j * xs;
            const int s = member(j);
            return per_slot ? (size_t)s * sg.K : (sg.pad1 & OWN_X_PER_TOKEN) ? (size_t)(s >> 3) * sg.K : 0;
        };
        float rm[OWN_MT][OWN_K3_RW];
        const uint8_t * Wm = m ? (const uint8_t *)sg.aux : sg.W;
        if (sg.work == OWN_Q4_K) own_rows_q4k_1rt_mt<OWN_K3_RW>(Wm, grow, lim, (const _Float16 *)sg.a, xoff, nt, lane, sg.K, rm);
        if (sg.work == OWN_Q6_K) own_rows_q6k_planes_mt<OWN_K3_RW>(Wm, total, grow, lim, (const _Float16 *)sg.a, xoff, nt, lane, sg.K, rm);
        if (lane == 0) {
#pragma unroll
            for (int j = 0; j < OWN_MT; j++) {
                if (j >= nt) break;
                const size_t yo = grouped ? (size_t)member(j) * re + r : (size_t)j * ys + row;
#pragma unroll
                for (int i = 0; i < OWN_K3_RW; i++) {
                    if (grow + i >= lim) continue;
                    AS3F * gp = (AS3F *)OWN_GU_GM + j * OWN_GU_RPT + ((row + i) & (OWN_GU_RPT - 1));
                    if (!gu) ((float *)sg.y)[yo + i] = rm[j][i];
                    else if (!m) *gp = rm[j][i];
                    else { const float gv = *gp; ((_Float16 *)sg.y)[yo + i] = (_Float16)(gv / (1.0f + expf(-gv)) * rm[j][i]); }
                }
            }
        }
#else
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
#endif
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
#ifdef K3_Q3M_VER   // (task tk = token tk: its id at Seg.a[tk], its row at Seg.y + tk * N)
__device__ __forceinline__ void own_op_emb(const Seg & sg, int tk) {
    const int tok = min(max(((const int *)sg.a)[tk], 0), sg.pad2 - 1);
    const uint8_t * row = sg.W + (size_t)tok * sg.K;
    float * x = (float *)sg.y + (size_t)tk * sg.N;
#else
__device__ __forceinline__ void own_op_emb(const Seg & sg) {
    const int tok = min(max(((const int *)sg.a)[0], 0), sg.pad2 - 1);
    const uint8_t * row = sg.W + (size_t)tok * sg.K;
    float * x = (float *)sg.y;
#endif
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
#ifdef K3_Q3M_VER   // (task tk = token tk: its rows d [tk][S][N], w [tk][S], x / h / h16 [tk][N]; nw shared)
__device__ __forceinline__ void own_op_add_rms16(const Seg & sg, int tk) {
    __shared__ float part[8];
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, N = sg.N, S = sg.pad1, NI = N >> 8;
    const float * d = (const float *)sg.a + (size_t)tk * S * N; const float * nw = (const float *)sg.W;
    const float * w = sg.gs ? (const float *)sg.gs + (size_t)tk * S : nullptr;
    float * x = (float *)sg.y + (size_t)tk * N; float * h = (float *)sg.asum + (size_t)tk * N; _Float16 * h16 = (_Float16 *)sg.aux + (size_t)tk * N;
#else
__device__ __forceinline__ void own_op_add_rms16(const Seg & sg) {
    __shared__ float part[8];
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, N = sg.N, S = sg.pad1, NI = N >> 8;
    const float * d = (const float *)sg.a; const float * w = (const float *)sg.gs; const float * nw = (const float *)sg.W;
    float * x = (float *)sg.y; float * h = (float *)sg.asum; _Float16 * h16 = (_Float16 *)sg.aux;
#endif
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
    ss = wave_sum(ss); if (lane == 0) part[wave] = ss; __syncthreads();
    float tot = 0.0f; for (int q = 0; q < 8; q++) tot += part[q];
    const float scale = rsqrtf(tot / N + sg.rs);
#pragma unroll
    for (int j = 0; j < 8; j++)
        if (j < NI) { const float v = (xv[j] * scale) * wn[j]; h[tid + 256 * j] = v; h16[tid + 256 * j] = (_Float16)v; }
    __syncthreads();
}
// OP_ATTN_DEC in the qwen3moe modules: op_attn_dec's arithmetic in its order (the same bits; kernels/k3_gguf_ops.h), with
// Seg.aux set the loads issued in batches. The serial form takes one load round trip per (position, 32 dims) of a
// wave's scores and one per position of a thread's V sum: 9.8 us per layer at position 14, 110 us at position 226 (XT
// trace: 5.3 of 12.0 ms per token). Batched: a wave scores two positions per round with their 4 K loads each (and q's)
// in flight (head dim a multiple of 128: host-checked), a thread sums V over 8 positions per round, each sum in the
// same order as before. One workgroup per head still walks every position: the cost stays linear in the position.
// LDS of the two attention ops below (a workgroup runs one at a time): [0, 512) scores (OP_OWN_ATTN's norms first: their
// rows [0, 256), their sums [256, 264)), [512, 520) reductions, then OP_OWN_ATTN's q, k and v [128] each. k3.hip's MVP /
// MVS are not in these modules (no OP_MATVEC_G128 code): borrowing them grew the kernel's LDS from 12.9 to 21.1 KB.
enum { OWN_AT_CH = 512, OWN_AT_Q = OWN_AT_CH + 8 };
__shared__ float OWN_AT[OWN_AT_Q + 3 * 128];
__device__ __forceinline__ void own_op_attn_dec(const Seg & sg, int h) {
    constexpr int CH = OWN_AT_CH;
    __attribute__((address_space(3))) float * sc = (__attribute__((address_space(3))) float *)OWN_AT, * red = sc + CH;
    // (the batched form reads the f16 cache's halves itself: kv_get's branch on the cache type, and its conversion
    // next to each load, kept every load a round trip of its own)
    const bool fast = sg.aux != nullptr && sg.work == 0;
    const int heads = sg.N / seg_T(sg), tt = h / heads;          // task = token * heads + head
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, D = sg.K, nkv = sg.pad1, g = (h - tt * heads) / (heads / nkv);
    const int npos = ((const int *)sg.asum)[0] + tt + 1, M = nkv * D, type = sg.work;
    const size_t pb = kv_pos_bytes(M, type);
    const float * q = (const float *)sg.a + (size_t)h * D;
    const uint8_t * cache = sg.W;
    float run_m = -INFINITY, run_s = 0.0f, acc[2] = {0.0f, 0.0f};
#pragma unroll 1
    for (int c0 = 0; c0 < npos; c0 += CH) {
        const int n = min(CH, npos - c0);
        if (!fast) {
            for (int t = wave; t < n; t += 8) {
                float a = 0.0f;
                for (int d = lane; d < D; d += 32) a += q[d] * kv_get(cache + (size_t)(c0 + t) * pb, M, type, 0, g * D + d);
                a = wave_sum(a);
                if (lane == 0) sc[t] = a * sg.rs;
            }
        } else {
            for (int t = wave; t < n; t += 16) {                 // positions t and t + 8 (the second clamped, unused past n)
                const int t1 = min(t + 8, n - 1);
                float a0 = 0.0f, a1 = 0.0f;
                const _Float16 * kp0 = (const _Float16 *)(cache + (size_t)(c0 + t) * pb) + g * D;
                const _Float16 * kp1 = (const _Float16 *)(cache + (size_t)(c0 + t1) * pb) + g * D;
                for (int d = lane; d < D; d += 128) {
                    float qv[4]; _Float16 k0[4], k1[4];
#pragma unroll
                    for (int j = 0; j < 4; j++) { qv[j] = q[d + 32 * j]; k0[j] = kp0[d + 32 * j]; k1[j] = kp1[d + 32 * j]; }
                    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                    for (int j = 0; j < 4; j++) { a0 += qv[j] * (float)k0[j]; a1 += qv[j] * (float)k1[j]; }
                    __builtin_amdgcn_sched_barrier(0);
                }
                a0 = wave_sum(a0); a1 = wave_sum(a1);
                if (lane == 0) { sc[t] = a0 * sg.rs; if (t + 8 < n) sc[t + 8] = a1 * sg.rs; }
            }
        }
        __syncthreads();
        float m = -INFINITY;
        for (int t = tid; t < n; t += blockDim.x) m = fmaxf(m, sc[t]);
        m = wave_max(m); if (lane == 0) red[wave] = m; __syncthreads();
        m = red[0]; for (int w = 1; w < 8; w++) m = fmaxf(m, red[w]); __syncthreads();
        const float mn = fmaxf(run_m, m), corr = expf(run_m - mn);
        float sum = 0.0f;
        for (int t = tid; t < n; t += blockDim.x) { const float e = expf(sc[t] - mn); sc[t] = e; sum += e; }
        sum = wave_sum(sum); if (lane == 0) red[wave] = sum; __syncthreads();
        sum = 0.0f; for (int w = 0; w < 8; w++) sum += red[w];
        run_s = c0 == 0 ? sum : run_s * corr + sum;
        for (int j = 0; j < 2; j++) {
            const int d = tid + j * (int)blockDim.x;
            if (d >= D) break;
            float part = 0.0f;
            if (!fast) {
                for (int t = 0; t < n; t++) part += sc[t] * kv_get(cache + (size_t)(c0 + t) * pb, M, type, 1, g * D + d);
            } else {
                const _Float16 * vb = (const _Float16 *)(cache + (size_t)c0 * pb) + M + g * D + d;   // (V of position c0)
                for (int t = 0; t < n; t += 8) {
                    _Float16 v[8];
#pragma unroll
                    for (int i = 0; i < 8; i++) v[i] = vb[(size_t)min(t + i, n - 1) * (2 * M)];   // (2 M halves per position)
                    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                    for (int i = 0; i < 8; i++) { if (t + i < n) part += sc[t + i] * (float)v[i]; }
                    __builtin_amdgcn_sched_barrier(0);
                }
            }
            acc[j] = c0 == 0 ? part : acc[j] * corr + part;
        }
        run_m = mn;
        __syncthreads();
    }
    const float inv = 1.0f / run_s;
    for (int j = 0; j < 2; j++) {
        const int d = tid + j * (int)blockDim.x;
        if (d < D) ((float *)sg.y)[(size_t)h * D + d] = acc[j] * inv;
    }
}
// OP_OWN_ATTN: one query head's decode attention as one task: the q and k norms + RoPE (op_qk_nr), the position's cache
// row (op_kv_append, f16), the attention over positions 0 .. pos (own_op_attn_dec, batched) and its fp16 output
// (OP_OWN_F16): five segments' work in one, every sum in their order (the same bits). Head dim 128, n_rot = the head dim,
// f16 cache, T = 1 (host-checked).
//   Seg.a = q fp32 [heads][D] and Seg.gs = k | v fp32 [2][kv heads * D], the matvecs' outputs; Seg.aux = fp32 [2 D + 2]:
//   q_norm [D], k_norm [D], theta_scale, eps; Seg.W = the cache, Seg.asum = int32 pos [1], Seg.y = fp16 out [heads][D];
//   N = heads, K = D, out_f32 = n_rot, pad1 = kv heads, rs = the score scale.
// Threads 0 .. 127 run q's norm and rotation, 128 .. 255 its kv head's k (the same lanes and waves as op_qk_nr's one
// head per task, so the same partial sums); the position's own k and v are used from LDS as the f16 cache would return
// them, and the kv head's first query head writes them to the cache (later positions' tasks read it, never this one's).
enum { OP_OWN_ATTN = 50 };
#ifndef OWN_AT_KB
#define OWN_AT_KB 4    // positions of K a wave has in flight per round trip (4 two-byte loads each: 4 VGPRs a position)
#endif
#ifndef OWN_AT_VB
#define OWN_AT_VB 8    // positions of V a thread has in flight per round trip (one two-byte load each)
#endif
#ifndef OWN_AT_VF
#define OWN_AT_VF 8    // scores read from LDS together in the V sum (divides OWN_AT_VB)
#endif
__device__ __forceinline__ void own_op_attn(const Seg & sg, int h) {
// Every multiply-add below is written out, in the form the compiler gives the five segments' ops: left to it here, it
// merged the norm's square with the wave sum's first add (whole heads' scales moved by an ulp) and the output's
// multiply with its fp16 conversion (v_fma_mixlo_f16: one rounding where fp32 -> fp16 makes two).
#pragma clang fp contract(off)
    constexpr int CH = OWN_AT_CH;
    __attribute__((address_space(3))) float * sc = (__attribute__((address_space(3))) float *)OWN_AT, * qkv = sc + OWN_AT_Q;   // (q, k, v [128] each)
    // (the attention's loads below: a scalar base per position + the 32-bit lane / thread offset, global not flat: a flat
    //  load takes a 64-bit address per lane in two VGPRs, 32 positions in flight then cost k3_run 250 spilled VGPRs)
    typedef __attribute__((address_space(1))) const _Float16 OwnGH;
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, D = sg.K, nrot = sg.out_f32, heads = sg.N, nkv = sg.pad1;
    const uint32_t ulane = (uint32_t)tid & 31u, utid = (uint32_t)tid & 255u;
    // (the position through readfirstlane: a value loaded through a flat pointer counts as different per lane, and with
    //  it every loop bound and load address below: per-lane compares, selects and 64-bit vector adds for each load)
    const int G = heads / nkv, g = h / G, M = nkv * D, pos = (int)__builtin_amdgcn_readfirstlane(((const int *)sg.asum)[0]), npos = pos + 1;
    const size_t pb = kv_pos_bytes(M, 0);
    const float * par = (const float *)sg.aux; const float * kvr = (const float *)sg.gs;
    const uint8_t * cache = sg.W;
    const bool isk = tid >= D; const int i = isk ? tid - D : tid;
    // -- the two norms + rotations (op_qk_nr's statements), every global input in one round trip
    const float theta_scale = par[2 * D], eps = par[2 * D + 1];
    const float * xp = isk ? kvr + (size_t)g * D + i : (const float *)sg.a + (size_t)h * D + i;
    const float x = *xp, w = par[tid], vr = kvr[(size_t)M + (size_t)g * D + i];
    float ss = x * x;
    ss = wave_sum(ss); if (lane == 0) sc[256 + wave] = ss; __syncthreads();
    float tot = 0.0f; for (int q = 0; q < 4; q++) tot += sc[256 + (isk ? 4 : 0) + q];
    const float scale = rsqrtf(tot / D + eps);
    sc[tid] = (scale * x) * w;
    __syncthreads();
    {
        const float fpos = (float)pos;
        const int j = i < nrot / 2 ? i : i - nrot / 2, nb = isk ? D : 0;
        const float theta = fpos * powf(theta_scale, (float)j);
        const float c = cosf(theta), sn = sinf(theta);
        const float x0 = sc[nb + j], x1 = sc[nb + j + nrot / 2];
        const float y = i < nrot / 2 ? __builtin_fmaf(x0, c, -(x1 * sn)) : __builtin_fmaf(x1, c, x0 * sn);
        if (!isk) qkv[i] = y;
        else {
            const _Float16 kh = (_Float16)y, vh = (_Float16)vr;
            qkv[D + i] = (float)kh; qkv[2 * D + i] = (float)vh;
            if (h == g * G) {
                _Float16 * cw = (_Float16 *)((uint8_t *)sg.W + (size_t)pos * pb);
                cw[g * D + i] = kh; cw[M + g * D + i] = vh;
            }
        }
    }
    __syncthreads();
    // -- the attention (own_op_attn_dec's batched form; the last position from LDS)
    float qv[4];
#pragma unroll
    for (int j = 0; j < 4; j++) qv[j] = qkv[lane + 32 * j];
    float run_m = -INFINITY, run_s = 0.0f, acc = 0.0f;
#pragma unroll 1
    for (int c0 = 0; c0 < npos; c0 += CH) {
        const int n = min(CH, npos - c0), nc = c0 + n == npos ? n - 1 : n;   // nc: the chunk's positions in the cache
        for (int t = (int)__builtin_amdgcn_readfirstlane(wave); t < nc; t += 8 * OWN_AT_KB) {   // a wave's positions t, t + 8, ..: OWN_AT_KB per round trip
            _Float16 kk[OWN_AT_KB][4];                         // (positions past nc reread the last one and store nothing)
#pragma unroll
            for (int p = 0; p < OWN_AT_KB; p++) {
                const OwnGH * kp = (const OwnGH *)(cache + (size_t)(c0 + min(t + 8 * p, nc - 1)) * pb) + g * D;   // (wave-uniform)
#pragma unroll
                for (int j = 0; j < 4; j++) kk[p][j] = kp[ulane + 32u * j];
            }
            __builtin_amdgcn_sched_barrier(0);
            float a[OWN_AT_KB];
#pragma unroll
            for (int p = 0; p < OWN_AT_KB; p++) {
                a[p] = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; j++) a[p] = __builtin_fmaf(qv[j], (float)kk[p][j], a[p]);
            }
            // the positions' lane sums stage by stage (each position wave_sum's order): a stage's cross-lane permutes
            // are LDS-path instructions, issued together they share one round trip; one position after another paid
            // five each
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
                for (int p = 0; p < OWN_AT_KB; p++) a[p] += __shfl_xor(a[p], o, 32);
            }
#pragma unroll
            for (int p = 0; p < OWN_AT_KB; p++) { if (lane == 0 && t + 8 * p < nc) sc[t + 8 * p] = a[p] * sg.rs; }
            __builtin_amdgcn_sched_barrier(0);
        }
        if (nc < n && wave == 0) {                            // this position's k: from LDS
            float a = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; j++) a = __builtin_fmaf(qv[j], qkv[D + lane + 32 * j], a);
            a = wave_sum(a);
            if (lane == 0) sc[nc] = a * sg.rs;
        }
        __syncthreads();
        float m = -INFINITY;
        for (int t = tid; t < n; t += blockDim.x) m = fmaxf(m, sc[t]);
        m = wave_max(m); if (lane == 0) sc[CH + wave] = m; __syncthreads();
        m = sc[CH]; for (int q = 1; q < 8; q++) m = fmaxf(m, sc[CH + q]); __syncthreads();
        const float mn = fmaxf(run_m, m), corr = expf(run_m - mn);
        float sum = 0.0f;
        for (int t = tid; t < n; t += blockDim.x) { const float e = expf(sc[t] - mn); sc[t] = e; sum += e; }
        sum = wave_sum(sum); if (lane == 0) sc[CH + wave] = sum; __syncthreads();
        sum = 0.0f; for (int q = 0; q < 8; q++) sum += sc[CH + q];
        run_s = c0 == 0 ? sum : __builtin_fmaf(run_s, corr, sum);
        if (tid < D) {
            float part = 0.0f;
            // (V of a position: after its K, M halves in; a position past nc rereads the last one. One scalar address at a
            //  time, behind a fence: the scheduler otherwise computes a round trip's addresses first, all live at once)
            const uint8_t * const vlast = cache + (size_t)(c0 + nc - 1) * pb + (size_t)(M + g * D) * 2;
            for (int t = 0; t < nc; t += OWN_AT_VB) {          // OWN_AT_VB positions per round trip, summed in position order
                _Float16 v[OWN_AT_VB];
                const uint8_t * vrow = cache + (size_t)(c0 + t) * pb + (size_t)(M + g * D) * 2;
#pragma unroll
                for (int q = 0; q < OWN_AT_VB; q++) {
                    v[q] = ((const OwnGH *)(t + q < nc ? vrow : vlast))[utid];
                    vrow += pb;
                    __builtin_amdgcn_sched_barrier(0);
                }
                __builtin_amdgcn_sched_barrier(0);
                // the scores OWN_AT_VF at a time, read from LDS together and unconditionally (an index past the chunk
                // rereads its last slot, unused): one read inside each position's branch was an LDS round trip a position
#pragma unroll
                for (int q0 = 0; q0 < OWN_AT_VB; q0 += OWN_AT_VF) {
                    float w[OWN_AT_VF];
#pragma unroll
                    for (int q = 0; q < OWN_AT_VF; q++) w[q] = sc[min(t + q0 + q, CH - 1)];
                    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                    for (int q = 0; q < OWN_AT_VF; q++) { if (t + q0 + q < nc) part = __builtin_fmaf(w[q], (float)v[q0 + q], part); }
                    __builtin_amdgcn_sched_barrier(0);
                }
            }
            if (nc < n) part = __builtin_fmaf(sc[nc], qkv[2 * D + tid], part);
            acc = c0 == 0 ? part : __builtin_fmaf(acc, corr, part);
        }
        run_m = mn;
        __syncthreads();
    }
    const float inv = 1.0f / run_s;
    // acc * inv rounded to fp32, then to fp16, as the five segments' fp32 buffer makes it. The backend selects
    // v_fma_mixlo_f16 for the fp16 conversion of a multiply or a multiply-add (one rounding: exact fp16 ties land on
    // the other neighbour, about one value in 8000), whatever the contraction setting; the empty asm hides the fp32
    // product from that pattern at no instruction.
    float o32 = acc * inv;
    __asm__ volatile("" : "+v"(o32));
    if (tid < D) ((_Float16 *)sg.y)[(size_t)h * D + tid] = (_Float16)o32;
}
#include "k3_own_atts.h"   // OP_OWN_ATTS: the attention split over positions (speed at depth)
#endif
