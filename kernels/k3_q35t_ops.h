// The dense qwen35 line's decode ops for ThinkingCap (Qwen3.8-27B, RCO 4.2 bpw: IQ4_XS, IQ3_XXS, IQ3_S, IQ2_S, Q5_K, Q6_K,
// Q8_0 matrices, an IQ2_S embedding; kernels/k3_q35_tcap.hip). kernels/k3_q35_ops.h's ops (the glue, the Gated DeltaNet,
// attention, OP_OWN_MV16 for Q8_0), under this header's dispatch, with:
//   OP_OWN_MV    q35t_op_mv: the one-round-trip tiles in their K3 form (runtime K, 2 rows per wave, UNR 1) for every
//                matrix format of the file except Q8_0: IQ4_XS (own_quant_iq.h, blocks in place), IQ3_XXS / IQ3_S (one
//                body, the grids in LDS, the record layout), IQ2_S (own_quant_iq2s.h, its packed grid in LDS, the record
//                layout), Q5_K (own_quant_q5k.h, blocks in place), Q6_K (own_quant.h, the plane layout). No gate|up
//                fusion (the feed-forward's gate and up differ in format in most of this file's blocks).
//   OP_OWN_EMB   q35t_op_emb: the IQ2_S embedding row, decoded as caminus.own_dequant iq2_s (bitwise), or Q4_K (k3_own_ops).
//   the launch hook: every workgroup stages the grids the tape's matvecs read (bits in the tape's first segment, the
//                embedding's Seg.out_f32: 1 IQ3_XXS, 2 IQ3_S, 8 IQ2_S; set by the host, engine/k3.py).
// The Swift and qwen3moe modules don't include this header: their code objects are unchanged.
#pragma once
#include <stdint.h>

#define K3_Q35_NO_DISPATCH
#include "k3_q35_ops.h"
#include "own_quant_iq.h"
#include "own_quant_iq2s.h"
#include "own_quant_q5k.h"
#ifdef K3_Q3M_VER   // (the verify module, kernels/k3_q35_tcap_ver.hip: the T-token tiles; the decode module never sees them)
#include "own_quant_tcap_mt.h"
#endif

enum { OWN_IQ2_S = 7 };   // (Seg.work of an IQ2_S matrix or embedding: its GG number; engine/k3.py OWN_Q35T_FMT; the others: k3_own_ops.h / k3_q35_ops.h)
enum { Q35T_STAGE_IQ3_XXS = 1, Q35T_STAGE_IQ3_S = 2, Q35T_STAGE_IQ2_S = 8 };

__device__ __forceinline__ void q35t_stage(const Seg * tape) {
#ifndef K3_Q35_DF   // (the DFlash drafter module runs no OWN_MV (its caps: none), so neither grid is staged nor held in its
                    //  LDS: with the IQ2_S grid as bytes (8 KB) it would need 22 KB a workgroup, 6 per WGP past the 128 KB)
    const int f = tape[0].op == OP_OWN_EMB ? tape[0].out_f32 : 0;
    if (f & (Q35T_STAGE_IQ3_XXS | Q35T_STAGE_IQ3_S)) iq_stage((int)K3TID, f & (Q35T_STAGE_IQ3_XXS | Q35T_STAGE_IQ3_S));
    if (f & Q35T_STAGE_IQ2_S) iq2s_stage((int)K3TID);
#endif
}
#define K3_LAUNCH_HOOK(tape) q35t_stage(tape)

// OP_OWN_MV: out[r] = W[r] . x16 for rows r of a task (Seg.rows_per_task, even): wave w takes the row pairs r0 + 2 w,
// r0 + 2 (w + 8), ... Seg.a = x16 fp16 [K], Seg.W = the matrix (its format's layout), Seg.y = fp32 out [N], Seg.N = rows,
// Seg.K = K, Seg.out_f32 = the matrix's rows (the planes' and records' strides), Seg.work = the format. A row's sum does
// not depend on the grid or the task size. Each format's tile gets the lane through an empty asm in its own branch
// (k3_own_ops.h's reason: derived from one shared lane value, the compiler hoists every tile's offsets out of the row
// loop and keeps them all live).
__device__ __forceinline__ void q35t_op_mv(const Seg & sg, int t) {
    const int wave = (int)__builtin_amdgcn_readfirstlane(K3TID / 32), lane = K3TID % 32;
    const int rpt = sg.rows_per_task, r0 = t * rpt, r1 = min(sg.N, r0 + rpt), total = sg.out_f32, fmt = sg.work;
    const _Float16 * x = (const _Float16 *)sg.a;
#ifdef K3_Q3M_VER
    // the verify module: NT tokens a pass (Seg.pad1 bits 8 .. 11), token j's activations at x + j xs and its rows at
    // y + j ys (xs / ys: Seg.gs's low / high half), each token's sums the one-token tile's bits (own_quant_tcap_mt.h)
    const int NT = max(1, (int)__builtin_amdgcn_readfirstlane((sg.pad1 >> 8) & 15));
    const uint64_t st = (uint64_t)sg.gs;
    const size_t xs = (uint32_t)__builtin_amdgcn_readfirstlane((int)(uint32_t)st), ys = (uint32_t)__builtin_amdgcn_readfirstlane((int)(uint32_t)(st >> 32));
    const auto xo = [&](int j) -> size_t { return (size_t)j * xs; };
    // one row loop per format (the format chosen once, outside it: five tiles in one loop body spilled k3_run)
#define Q35T_MT_LOOP(CALL)                                                                                             \
    for (int row = r0 + 2 * wave; row < r1; row += 16) {                                                               \
        const int lim = min(row + 2, r1);                                                                              \
        float rm[OWN_MT][2];                                                                                           \
        int l = lane; asm volatile("" : "+v"(l));                                                                      \
        CALL;                                                                                                          \
        if (lane == 0) {                                                                                               \
            _Pragma("unroll") for (int j = 0; j < OWN_MT; j++) {                                                       \
                if (j >= NT) break;                                                                                    \
                ((float *)sg.y)[(size_t)j * ys + row] = rm[j][0];                                                      \
                if (row + 1 < lim) ((float *)sg.y)[(size_t)j * ys + row + 1] = rm[j][1];                               \
            }                                                                                                          \
        }                                                                                                              \
    }
#ifndef K3_Q35_MTP   // (ThinkingCap's MTP drafter module, kernels/k3_q35_tcap_mtp.hip: its block is all Q6_K, the head a rows tile:
                     //  the other formats' T-token loops out of its register allocation; the host refuses them there)
    if (fmt == OWN_IQ4_XS) { Q35T_MT_LOOP(own_rows_iq4xs_mt<2>(sg.W, row, lim, x, xo, NT, l, sg.K, rm)) }
    else if (fmt == OWN_IQ3_XXS || fmt == OWN_IQ3_S) { const int f = fmt == OWN_IQ3_XXS ? 0 : 1; Q35T_MT_LOOP(own_rows_iq3_mt<2>(sg.W, total, row, lim, x, xo, NT, l, sg.K, rm, f)) }
    else if (fmt == OWN_IQ2_S) { Q35T_MT_LOOP(own_rows_iq2s_mt<2>(sg.W, total, row, lim, x, xo, NT, l, sg.K, rm)) }
    else if (fmt == OWN_Q5_K) { Q35T_MT_LOOP(own_rows_q5k_mt<2>(sg.W, row, lim, x, xo, NT, l, sg.K, rm)) }
    else
#endif
    { Q35T_MT_LOOP(own_rows_q6k_planes_mt2<2>(sg.W, total, row, lim, x, xo, NT, l, sg.K, rm)) }   // (OWN_Q6_K: host-checked)
#undef Q35T_MT_LOOP
    return;
#endif
    for (int row = r0 + 2 * wave; row < r1; row += 16) {
        const int lim = min(row + 2, r1);
        float res[2] = {0.0f, 0.0f};
        int l = lane; asm volatile("" : "+v"(l));
        if (fmt == OWN_IQ4_XS) own_rows_iq4xs_1rt<0, 2, 1, 1>(sg.W, row, lim, x, nullptr, l, sg.K, res);
        // (the format fixed at compile time, one body per format: own_quant_iq.h measured the runtime-format body slower
        //  and gives the same bits in both forms)
        else if (fmt == OWN_IQ3_XXS) own_rows_iqg_1rt<0, 0, 2, 1, 1>(sg.W, total, row, lim, x, nullptr, l, sg.K, res);
        else if (fmt == OWN_IQ3_S) own_rows_iqg_1rt<1, 0, 2, 1, 1>(sg.W, total, row, lim, x, nullptr, l, sg.K, res);
        else if (fmt == OWN_IQ2_S) own_rows_iq2s_1rt<0, 2, 1, 1>(sg.W, total, row, lim, x, nullptr, l, sg.K, res);
        else if (fmt == OWN_Q5_K) own_rows_q5k_1rt<0, 2, 1, 1>(sg.W, row, lim, x, nullptr, l, sg.K, res);
        else own_rows_q6k_planes<0, 2, 1, 1>(sg.W, total, row, lim, x, nullptr, l, sg.K, res);   // (OWN_Q6_K: host-checked)
        if (lane == 0) {
            ((float *)sg.y)[row] = res[0];
            if (row + 1 < lim) ((float *)sg.y)[row + 1] = res[1];
        }
    }
}

// OP_OWN_EMB, IQ2_S (Seg.work = OWN_IQ2_S): x fp32 [N] = row tok of the raw GGUF blocks (Seg.K bytes per row, Seg.pad2 =
// the vocabulary; the id clamped to it): value j of a block = ((d * (0.5 + s)) * 0.25) * grid * sign, the host decoder's
// float32 operations in its order (caminus.own_dequant iq2_s: bitwise). The grid read from the packed constant (one
// embedding row per token: not worth the LDS staging). Q4_K: k3_own_ops.h's own_op_emb. One task.
#ifdef K3_Q3M_VER   // (task tk = token tk: its id at Seg.a[tk], its row at Seg.y + tk * N; own_op_emb's verify form)
__device__ __forceinline__ void q35t_op_emb(const Seg & sg, int tk) {
    if (sg.work != OWN_IQ2_S) { own_op_emb(sg, tk); return; }
#ifdef K3_Q35_MTP   // (Seg.pad1 = 1: the token is OP_OWN_ARGMAX's key at Seg.a, the low half ~index: a chained draft step)
    const int tok = min(max(sg.pad1 ? (int)~(uint32_t)((const unsigned long long *)sg.a)[0] : ((const int *)sg.a)[tk], 0), sg.pad2 - 1);
#else
    const int tok = min(max(((const int *)sg.a)[tk], 0), sg.pad2 - 1);
#endif
    const uint8_t * row = sg.W + (size_t)tok * sg.K;
    float * x = (float *)sg.y + (size_t)tk * sg.N;
#else
__device__ __forceinline__ void q35t_op_emb(const Seg & sg) {
    if (sg.work != OWN_IQ2_S) { own_op_emb(sg); return; }
    const int tok = min(max(((const int *)sg.a)[0], 0), sg.pad2 - 1);
    const uint8_t * row = sg.W + (size_t)tok * sg.K;
    float * x = (float *)sg.y;
#endif
    for (int i = (int)K3TID; i < sg.N; i += blockDim.x) {
        const uint8_t * blk = row + (size_t)(i >> 8) * 82;
        const int j = i & 255, ib = j >> 5, g = (j >> 3) & 3, k = j & 7;
        const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(blk[0] | (blk[1] << 8)));
        const uint32_t idx = blk[2 + 4 * ib + g] | (((uint32_t)(blk[66 + ib] >> (2 * g)) & 3u) << 8);
        const uint32_t code = ((IQ2S_GRID[idx >> 1] >> (16 * (idx & 1u))) >> (2 * k)) & 3u;
        const float mag = code == 0 ? 8.0f : code == 1 ? 25.0f : 43.0f;
        const float s = (float)((blk[74 + ib] >> (4 * ((j >> 4) & 1))) & 15);
        const float db = __fmul_rn(__fmul_rn(d, __fadd_rn(0.5f, s)), 0.25f);
        const float v = __fmul_rn(db, mag);
        x[i] = ((blk[34 + (j >> 3)] >> k) & 1) ? -v : v;
    }
}

__device__ __forceinline__ bool k3_extra_op(const Seg & sg, int t, bool fresh, Sync * sy, int limit_us, int seg,
                                            uint64_t * tr = nullptr) {
#ifndef K3_Q35_DF   // (the DFlash module: no OWN_MV, k3.MODULE_CAPS refuses it there; its Q8_0 matvecs are OP_OWN_MV16)
    if (sg.op == OP_OWN_MV) { q35t_op_mv(sg, t); return true; }
#endif
#ifdef K3_Q35_MTP   // (ThinkingCap's MTP drafter module: the greedy key, k3_q35_ops.h's op; the host refuses the trunk's other ops)
    if (sg.op == OP_OWN_ARGMAX) { q35_op_argmax(sg, t); return true; }
#endif
#ifdef K3_Q35_DF   // (ThinkingCap's DFlash drafter module, kernels/k3_q35_tcap_dflash.hip: k3_q35_ops.h's DF block; the host refuses the rest)
    if (sg.op == OP_DF_CONV) { op_df_conv(sg, t); return true; }
    if (sg.op == OP_DF_TOPK) { op_df_topk(sg, t); return true; }
    if (sg.op == OP_DF_SELECT) { op_df_select(sg, t); return true; }
    if (sg.op == OP_KV_APPEND && sg.aux) { df_op_kv_append_ring(sg, t); return true; }
    if (sg.op == OP_ATTN_DEC && sg.pad2) { df_op_attn(sg, t); return true; }
#endif
    if (sg.op == OP_OWN_MV16) { q35_op_mv16(sg, t); return true; }
    if (sg.op == OP_OWN_ADD_RMS16) { q35_op_add_rms16(sg OWN_TK_ARG); return true; }
    if (sg.op == OP_F32_MV) { op_f32_mv(sg, t); return true; }
    if (sg.op == OP_GDN_CONV) { q35_op_gdn_conv(sg, t); return true; }
    if (sg.op == OP_GDN_L2) { q35_op_gdn_l2(sg, t); return true; }
    if (sg.op == OP_GDN_GATES) { q35_op_gdn_gates(sg); return true; }
    if (sg.op == OP_GDN_STEP) { q35_op_gdn_step(sg, t); return true; }
    if (sg.op == OP_GDN_SAVE) { q35_gdn_save_guarded(sg, t, sy); return true; }
    if (sg.op == OP_GDN_NORM16) { q35_op_gdn_norm16(sg, t); return true; }
    if (sg.op == OP_QK_NR) { q35_op_qk_nr(sg, t); return true; }
    if (sg.op == OP_KV_APPEND) { q35_op_kv_append(sg, t); return true; }
#ifdef K3_Q35_ATTN_SPLIT   // (k3_q35_ops.h's split pair; this module always defines it)
    if (sg.op == OP_ATTN_PART) { ATTN_PART_FN(sg, t); return true; }   // (k3_attn_split.h: the packed PART in the verify modules)
    if (sg.op == OP_ATTN_COMB) { op_attn_comb(sg, t); return true; }
#else
    if (sg.op == OP_ATTN_DEC) { own_op_attn_dec(sg, t); return true; }
#endif
    if (sg.op == OP_GATE_MUL16) { q35_op_gate_mul16(sg, t); return true; }
    if (sg.op == OP_OWN_SWIGLU16) { own_op_swiglu16(sg, t); return true; }
    if (sg.op == OP_OWN_EMB) { q35t_op_emb(sg OWN_TK_ARG); return true; }
#ifdef K3_TP   // (the two-card module k3_q35_tcap_tp; k3_xchg.h through k3_q35_ops.h)
    if (sg.op == OP_XCHG) { op_xchg(sg, sy, limit_us, seg, tr); return true; }
#endif
    return false;
}
#ifdef K3_TP
__device__ __forceinline__ uint32_t * k3_extra_warm(const Seg & sg) { return xchg_warm(sg); }
__device__ __forceinline__ void k3_extra_prefetch(const Seg & sg, int) { xchg_prefetch(sg); }
#else
__device__ __forceinline__ uint32_t * k3_extra_warm(const Seg &) { return nullptr; }
__device__ __forceinline__ void k3_extra_prefetch(const Seg &, int) {}
#endif
