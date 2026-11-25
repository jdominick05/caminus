// The qwen35moe batched prefill's wide GEMM tiles (Qwen3.6-35B-A3B's routed experts: Q2_0): q35mp_mm_q2_0_w and the fused
// gate | up + swiglu q35mp_mm_q2_0_gu, each row bitwise q35mp_mm_q2_0's. A separate object, so pf_q35m keeps its code
// and these build under LLVM's max-ilp scheduler (build.sh). Q35M_WIDE_OBJECT: pf_q35m.hip's helpers (and those of
// the files it includes) without their kernels. caminus.prefill.q35moe launches them from this object. No ggml-cuda code.
#define Q35M_WIDE_OBJECT
#include "pf_q35m.hip"
#include "pf_mm_wide.h"   // (the wide tiles' work item, rows and stores: one source with the 30B's Q4_K / Q6_K tiles)

// ---- the wide Q2_0 tile: two 16-row tiles per wave ----------------------------------------------------------------------
// The 30B's wide tile (kernels/pf_q3m_mm.hip mm_q4k_w: work item, rows, stores and the fused gate | up from
// pf_mm_wide.h; a partial item's empty 16-token tiles skip their WMMAs; the weights in a register ring one piece ahead)
// with q35mp_mm_q2_0's statements per row: each 64-value block's 4 WMMA k-steps from a zero accumulator, then
// acc = fma(d, P, acc), so every output keeps its bits (tests/prefill/test_q35m_prefill_ops.py and
// benchmarks/q35_prefill_gemm.py --engine compare them bitwise). Rows past R read row 0 and are never written.
// Measured on the XT, the experts' GEMMs of one real 2048-token chunk, each step interleaved against the one before:
// narrow (gate, up, swiglu apart) 136.8 ms; wide + fused 141.9 (every token tile's WMMAs: 56 VGPRs spilled); empty
// token tiles skipped 107.6; the half-step ring 101.9; max-ilp 101.6; q2_0_bw 91.3 (the same tile under the default
// scheduler: 92.3). 211 VGPRs, 0 spills, 33,792 B of LDS.

// q2_0_b's values (q - 1 as fp16, exact) by v_perm: byte k of the code dword holds values 4 k .. 4 k + 3 (2 bits each);
// it goes to both halves under fp16's exponent byte (1024 + its bits), a mask keeps one value per half, at weight 1, 4,
// 16 or 64, and one packed fma scales it back and subtracts 1025 / 257 / 65 / 17 (exact: every value is a small
// integer). About 20 VALU instructions per B operand where q2_0_b's shifts and ORs took about twice that (101.6 -> 91.3 ms
// of expert GEMMs a chunk)
__device__ __forceinline__ h16_t q2_0_bw(uint32_t v) {
    const h2_t s0 = {(_Float16)1.0f, (_Float16)0.25f}, o0 = {(_Float16)(-1025.0f), (_Float16)(-257.0f)};
    const h2_t s1 = {(_Float16)0.0625f, (_Float16)0.015625f}, o1 = {(_Float16)(-65.0f), (_Float16)(-17.0f)};
    uint32_t o[8];
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const uint32_t p = __builtin_amdgcn_perm(0x64646464u, v, 0x04000400u | (k << 16) | k);   // bytes: c_k, 0x64, c_k, 0x64
        o[2 * k] = as_u(__builtin_elementwise_fma(as_h2(p & 0xff0cff03u), s0, o0));         // values 4 k, 4 k + 1
        o[2 * k + 1] = as_u(__builtin_elementwise_fma(as_h2(p & 0xffc0ff30u), s1, o1));     // values 4 k + 2, 4 k + 3
    }
    return __builtin_bit_cast(h16_t, o);
}

// A step's 72 bytes (4 blocks) in two halves of 9 dwords: half h holds blocks 2 h and 2 h + 1 in the same places (block 0
// of a half: d in the low half of dword 0, its code dwords two bytes into dwords 0 .. 4; block 1: d in the high half of
// dword 4, codes in dwords 5 .. 8; q35mp_mm_q2_0's q2_block)
__device__ __forceinline__ void q2w_half_load(uint32_t (&w)[9], const uint8_t * wrow, int half) {
    const uint32_t * src = (const uint32_t *)(wrow + (size_t)half * 36);
#pragma unroll
    for (int i = 0; i < 9; i++) w[i] = src[i];
}
// block J of both row tiles' halves at k values k0 .. k0 + 63 of the stage: per row tile and token tile, P from zero over
// the 4 k-steps, then acc = fma(d, P, acc) (q2_block's order); each A load feeds both row tiles' WMMAs
template <int J>
__device__ __forceinline__ void q2w_block(const uint32_t (&w0)[9], const uint32_t (&w1)[9], const bool (&rok)[2], int k0, int lane,
                                          int nmt, f8_t (&acc)[2][4]) {
    f8_t P[2][4];
#pragma unroll
    for (int rt = 0; rt < 2; rt++)
#pragma unroll
        for (int mt = 0; mt < 4; mt++) P[rt][mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int c = 0; c < 4; c++) {
        const h16_t B0 = q2_0_bw(J ? w0[5 + c] : __builtin_amdgcn_alignbyte(w0[c + 1], w0[c], 2));
        const h16_t B1 = q2_0_bw(J ? w1[5 + c] : __builtin_amdgcn_alignbyte(w1[c + 1], w1[c], 2));
#pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            if (mt >= nmt) break;
            const h16_t A = mm_a(mt, k0 + 16 * c, lane);
            P[0][mt] = wmma(A, B0, P[0][mt]);
            P[1][mt] = wmma(A, B1, P[1][mt]);
        }
    }
    const float d0 = rok[0] ? (float)__builtin_bit_cast(_Float16, (uint16_t)(J ? w0[4] >> 16 : w0[0] & 0xffffu)) : 0.0f;
    const float d1 = rok[1] ? (float)__builtin_bit_cast(_Float16, (uint16_t)(J ? w1[4] >> 16 : w1[0] & 0xffffu)) : 0.0f;
#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
        if (mt >= nmt) break;
#pragma unroll
        for (int v = 0; v < 8; v++) {
            acc[0][mt][v] = __builtin_fmaf(d0, P[0][mt][v], acc[0][mt][v]);
            acc[1][mt][v] = __builtin_fmaf(d1, P[1][mt][v], acc[1][mt][v]);
        }
    }
}

template <bool GU>
__device__ __forceinline__ void mm_q2_0_w(const uint8_t * __restrict__ W, const uint8_t * __restrict__ W1, int R, int K,
                                          const _Float16 * __restrict__ X, int ldx, const int * __restrict__ xsrc,
                                          float * __restrict__ Y, _Float16 * __restrict__ Y16, int ldy, const int * __restrict__ ydst,
                                          const MMItem * __restrict__ items, const int * __restrict__ nitems, int ntok) {
    MMItem it;
    MMW_ITEM(it, items, nitems, ntok)
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int NB = K / 256;
    const size_t RB = (size_t)(K / 64) * 18;
    // (mm_q4k_w's skip of a partial item's empty token tiles: 63 % of the 35B's pair slots are filled at 2048 tokens;
    // without it the tile also spilled 56 VGPRs: 141.9 -> 107.6 ms a chunk)
    const int nmt = (it.cnt + 15) >> 4;
    int r[2]; bool rok[2]; const uint8_t * wrow[2];
#pragma unroll
    for (int rt = 0; rt < 2; rt++) {
        r[rt] = MMW_ROW(GU, rt, wave, lane);
        rok[rt] = r[rt] < R;
        wrow[rt] = (GU && rt ? W1 : W) + mmw_grow(it, items, R, r[rt], rok[rt]) * RB;
    }
    f8_t acc[2][4];
#pragma unroll
    for (int rt = 0; rt < 2; rt++)
#pragma unroll
        for (int mt = 0; mt < 4; mt++) acc[rt][mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
    // the weights half a step ahead: a two-half ring holds the half being multiplied and the next one in flight (this
    // step's second half, or the next step's first across the stage), 36 registers where a step and the next took 72
    // (247 VGPRs -> 211; 107.6 -> 101.9 ms a chunk)
    uint32_t hw[2][2][9];   // [ring = half][row tile]
#pragma unroll
    for (int rt = 0; rt < 2; rt++) q2w_half_load(hw[0][rt], wrow[rt], 0);
    for (int b = 0; b < NB; b++) {
        __syncthreads();
        mm_stage<false>(X, ldx, xsrc, it.off, it.cnt, b, tid);
        __syncthreads();
#pragma unroll
        for (int h = 0; h < 2; h++) {
            __builtin_amdgcn_sched_barrier(0);   // (the next half's loads issued here, not hoisted: the ring's registers are the point)
            const int nh = h ? 2 * min(b + 1, NB - 1) : 2 * b + 1;   // (the last step reloads its own first half: no branch)
#pragma unroll
            for (int rt = 0; rt < 2; rt++) q2w_half_load(hw[h ^ 1][rt], wrow[rt], nh);
            q2w_block<0>(hw[h][0], hw[h][1], rok, 128 * h, lane, nmt, acc);
            q2w_block<1>(hw[h][0], hw[h][1], rok, 128 * h + 64, lane, nmt, acc);
        }
    }
    MMW_STORE(GU, acc, r, rok, it, Y, Y16, ldy, ydst, lane)
}

// ---- the kernels: q35mp_mm_q2_0's interface; grid (work items or tokens / 64, rows / 256). q35mp_mm_q2_0_gu: W the gate
// tensor, W1 the up tensor (the same rows and experts), Y16 fp16 [.][ldy], grid (work items, rows / 128)
extern "C" __global__ void __launch_bounds__(256) q35mp_mm_q2_0_w(
        const uint8_t * __restrict__ W, int R, int K, const _Float16 * __restrict__ X, int ldx, const int * __restrict__ xsrc,
        float * __restrict__ Y, int ldy, const int * __restrict__ ydst, const MMItem * __restrict__ items, const int * __restrict__ nitems,
        int ntok) {
    mm_q2_0_w<false>(W, W, R, K, X, ldx, xsrc, Y, nullptr, ldy, ydst, items, nitems, ntok);
}
extern "C" __global__ void __launch_bounds__(256) q35mp_mm_q2_0_gu(
        const uint8_t * __restrict__ W, const uint8_t * __restrict__ W1, int R, int K, const _Float16 * __restrict__ X, int ldx,
        const int * __restrict__ xsrc, _Float16 * __restrict__ Y16, int ldy, const int * __restrict__ ydst,
        const MMItem * __restrict__ items, const int * __restrict__ nitems) {
    mm_q2_0_w<true>(W, W1, R, K, X, ldx, xsrc, nullptr, Y16, ldy, ydst, items, nitems, 0);
}


// ---- q35mp_mm_q8_0_l: the dense Q8_0 GEMMs on an LDS tile --------------------------------------------------------------
// The pipeline follows FlyDSL's RDNA3 fp16 GEMM (ROCm/FlyDSL v0.5.0, kernels/gemm/rdna3_f16_gemm.py; Copyright (c) 2026
// FlyDSL Project Contributors, Apache-2.0: THIRD_PARTY.md; a design followed, no code copied): a k-tile of x and of the weights in two LDS buffers, the next k-tile's
// global loads issued before this one's WMMAs and stored into the other buffer after them, one barrier per k-tile, rows
// of 32 + 8 fp16 in LDS (its "pad" layout).
// Y[t][r] = X[t] . W[r] in fp32 for t < ntok, each output bitwise q35p_mm_q8_0_w's (and so q35p_mm_q8_0's): per Q8_0
// block (32 values) and token tile, P from zero over the block's two WMMA k-steps (values 0 .. 15, then 16 .. 31), then
// acc = fma(d, P, acc), the blocks in order. What changes is where the work is done: the wide tile's waves each turn
// their own rows' int8 into fp16 B operands for their 64 tokens; here a workgroup is 128 tokens x 64 rows, and each
// weight becomes fp16 once per workgroup (q8_wbw's statements, exact), by one thread, into LDS, where the two waves of
// its 32-row band read it.
// A k-tile is one block. Steps go in pairs, so a block's byte alignment (34 B blocks: even blocks start 4-byte aligned,
// odd ones 2 bytes past) and the buffer are compile-time constants.
// Stage roles (128 threads): thread i stages all 32 values of token i and half i % 2 (16 values) of row i / 2; the row's
// d (the half-0 thread) goes to LDS as fp32. Tokens past ntok read token ntok - 1, rows past R read row R - 1; neither
// is written.
// Waves: wave w multiplies tokens 64 (w & 1) .. + 63 by rows 32 (w >> 1) .. + 31: 4 token tiles x 2 row tiles, as the
// wide tile's wave. The scale step runs one token tile behind the WMMAs (q8l_mul).
// Measured on the XT, the 35B's dense Q8_0 GEMMs of one real 2048-token chunk (benchmarks/q35_prefill_gemm.py --engine,
// each step interleaved against the one before; the wide tile 117.0 ms): 128 x 128 workgroups of 8 waves 112.4; 128 x 64
// of 4 waves 110.1 (30.5 KB of LDS: more workgroups resident); the scale step one token tile behind 107.6 (the wide tile
// 117.0 in the same runs). Tried and slower: 128 x 256 (8 token tiles a wave, 113.6: 256 VGPRs, spills), 16 waves
// (124.4). The scale step itself is ~7 % of the chunk: a correct fold (fp16(d q) staged, no scale step, not bitwise)
// took 100.9 against 108.9, and was not kept: mean KLD +28 % (F16) / +102 % (Q8 KV) in tests/prefill/test_q35_prefill.py
// for 1.03x end to end at 4,096 tokens (an earlier 99.5 against 112.4 came from a build with a wrong scale on half its
// rows). FlyDSL's fp16 GEMM on dense operands runs these shapes in 85.8 ms with 64 x 64 a wave
// (128 accumulator VGPRs, 4 spilled, no scale step). 16 accumulator tiles a wave here (8 token tiles x 2 row tiles)
// took 256 VGPRs and spilled (2 in 8-wave workgroups, 18 in 4-wave ones); a 4 x 4 wave tile with the scale step's P and
// four row tiles' B operands is estimated past 256 (not built). 208 VGPRs, 0 spills, 31,232 B of LDS.
#define Q8L_T 128   // tokens per workgroup
#define Q8L_R 64    // rows per workgroup
#define Q8L_KP 40   // fp16 per LDS row (32 values + 8: FlyDSL's K_PAD)
// a thread's staged block: its token's 32 values of x, and its half of its row's 34 bytes (5 dwords)
struct Q8LStage { u32x4_t x[4]; uint32_t w[5]; };
// global loads of block s (parity P = s & 1): x of token row xrow; half hw of weight row wrow
template <int P>
__device__ __forceinline__ void q8l_load(Q8LStage & S, const _Float16 * xrow, const uint8_t * wrow, int s, int hw) {
    const u32x4_t * xs = (const u32x4_t *)(xrow + s * 32);
#pragma unroll
    for (int j = 0; j < 4; j++) S.x[j] = xs[j];
    const uint32_t * ws = (const uint32_t *)(wrow + s * 34 + hw * 16 - 2 * P);   // (4-byte aligned: 34 s + 16 h - 2 P)
#pragma unroll
    for (int k = 0; k < 5; k++) S.w[k] = ws[k];
}
// the staged block into one LDS buffer: x as it is (token row ix), q as fp16 (q8_wbw: exact) and d as fp32 (weight row iw)
template <int P>
__device__ __forceinline__ void q8l_store(const Q8LStage & S, _Float16 * lx, _Float16 * lw, float * ld, int ix, int iw, int hw) {
    u32x4_t * dx = (u32x4_t *)(lx + ix * Q8L_KP);
#pragma unroll
    for (int j = 0; j < 4; j++) dx[j] = S.x[j];
    uint32_t q[4];
#pragma unroll
    for (int k = 0; k < 4; k++) q[k] = P ? S.w[k + 1] : __builtin_amdgcn_alignbyte(S.w[k + 1], S.w[k], 2);   // (2 + 2 P in)
    const _Float16 d = __builtin_bit_cast(_Float16, (uint16_t)(P ? S.w[0] >> 16 : S.w[0] & 0xffffu));
    const U2 b2 = __builtin_bit_cast(U2, q8_wbw(q, true));
    u32x4_t * dw = (u32x4_t *)(lw + iw * Q8L_KP + hw * 16);
    dw[0] = b2.a; dw[1] = b2.b;
    if (hw == 0) ld[iw] = (float)d;
}
// the accumulators: 8 separate floats per tile, not one 8-register tuple (the scale step's fmas then stay two-address and
// pair into dual-issue instructions: 100 VALU instructions for a k-tile pair's 128 fmas -> 66)
typedef float Q8LAcc[8];
// a k-tile from one LDS buffer: the wave's 4 token tiles (from token tb) x 2 row tiles (from row rb). The scale step runs
// one token tile behind: tile mt's WMMAs, then tile mt - 1's fmas, whose P (Pp) the WMMAs before wrote, so no fma waits
// on the WMMA just issued; the k-tile's first tile scales the previous k-tile's last tile (its d: dp). Each accumulator
// still takes its blocks in order (the same bits). Pp starts at zero with dp zero (fma(0, 0, +0) = +0); the caller
// scales the last k-tile's last tile after the loop.
__device__ __forceinline__ void q8l_mul(const _Float16 * lx, const _Float16 * lw, const float * ld, int tb, int rb, int lane,
                                        Q8LAcc (&acc)[4][2], f8_t (&Pp)[2], float (&dp)[2][2]) {
    const int l = lane & 15;
    h16_t B[2][2];
    float d[2][2];   // (each row tile's d twice, in two registers: a dual-issue fma pair needs its two scales in two banks)
#pragma unroll
    for (int rt = 0; rt < 2; rt++) {
        const u32x4_t * p = (const u32x4_t *)(lw + (rb + rt * 16 + l) * Q8L_KP);
        B[rt][0] = h16of(p[0], p[1]); B[rt][1] = h16of(p[2], p[3]);
        d[rt][0] = d[rt][1] = ld[rb + rt * 16 + l];
        asm volatile("" : "+v"(d[rt][1]));
    }
    // the token tiles in order, each one's A operands read during the WMMAs of the one before (fences keep that order: the
    // scheduler otherwise hoists every LDS read of the k-tile, 96 VGPRs of operands)
    const u32x4_t * pa = (const u32x4_t *)(lx + (tb + l) * Q8L_KP);
    h16_t A0 = h16of(pa[0], pa[1]), A1 = h16of(pa[2], pa[3]);
#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
        h16_t N0 = A0, N1 = A1;
        if (mt + 1 < 4) {
            const u32x4_t * pn = (const u32x4_t *)(lx + (tb + (mt + 1) * 16 + l) * Q8L_KP);
            N0 = h16of(pn[0], pn[1]); N1 = h16of(pn[2], pn[3]);
        }
        __builtin_amdgcn_sched_barrier(0);
        const int pt = (mt + 3) % 4;   // (the tile Pp belongs to)
        f8_t P0 = (f8_t){0, 0, 0, 0, 0, 0, 0, 0}, P1 = P0;
        P0 = wmma(A0, B[0][0], P0); P1 = wmma(A0, B[1][0], P1);
#pragma unroll
        for (int v = 0; v < 8; v++) acc[pt][0][v] = __builtin_fmaf(mt ? d[0][v & 1] : dp[0][v & 1], Pp[0][v], acc[pt][0][v]);
        P0 = wmma(A1, B[0][1], P0); P1 = wmma(A1, B[1][1], P1);
#pragma unroll
        for (int v = 0; v < 8; v++) acc[pt][1][v] = __builtin_fmaf(mt ? d[1][v & 1] : dp[1][v & 1], Pp[1][v], acc[pt][1][v]);
        // (the scale step pinned here: sunk past the next token tiles' WMMAs, every tile's P stayed live, 64 VGPRs and spills)
#pragma unroll
        for (int rt = 0; rt < 2; rt++)
            asm volatile("" : "+v"(acc[pt][rt][0]), "+v"(acc[pt][rt][1]), "+v"(acc[pt][rt][2]), "+v"(acc[pt][rt][3]),
                              "+v"(acc[pt][rt][4]), "+v"(acc[pt][rt][5]), "+v"(acc[pt][rt][6]), "+v"(acc[pt][rt][7]));
        Pp[0] = P0; Pp[1] = P1;
        __builtin_amdgcn_sched_barrier(0);
        A0 = N0; A1 = N1;
    }
#pragma unroll
    for (int rt = 0; rt < 2; rt++) { dp[rt][0] = d[rt][0]; dp[rt][1] = d[rt][1]; }
}
// grid (tokens / 128, rows / 64), block 128; K a multiple of 64; X fp16 [ntok][ldx] (ldx a multiple of 8), Y fp32 [ntok][ldy]
extern "C" __global__ void __launch_bounds__(128) q35mp_mm_q8_0_l(const uint8_t * __restrict__ W, int R, int K,
                                                                  const _Float16 * __restrict__ X, int ldx,
                                                                  float * __restrict__ Y, int ldy, int ntok) {
    __shared__ _Float16 LX[2][Q8L_T * Q8L_KP], LW[2][Q8L_R * Q8L_KP];
    __shared__ float LD[2][Q8L_R];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int iw = tid >> 1, hw = tid & 1;                       // (stage: x of token tid, half hw of weight row iw)
    const int t0 = blockIdx.x * Q8L_T, r0 = blockIdx.y * Q8L_R;
    const int NB = K / 32;                                       // (k-tiles: one Q8_0 block each; even)
    const _Float16 * xrow = X + (size_t)min(t0 + tid, ntok - 1) * ldx;
    const uint8_t * wrow = W + (size_t)min(r0 + iw, R - 1) * (size_t)(K / 32 * 34);
    const int tb = (wave & 1) * 64, rb = (wave >> 1) * 32;       // (the wave's tokens and rows in the workgroup)
    Q8LAcc acc[4][2];
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int rt = 0; rt < 2; rt++)
#pragma unroll
            for (int v = 0; v < 8; v++) acc[mt][rt][v] = 0.0f;
    f8_t Pp[2] = {(f8_t){0, 0, 0, 0, 0, 0, 0, 0}, (f8_t){0, 0, 0, 0, 0, 0, 0, 0}};
    float dp[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    Q8LStage S;
    q8l_load<0>(S, xrow, wrow, 0, hw);
    q8l_store<0>(S, LX[0], LW[0], LD[0], tid, iw, hw);
    __syncthreads();
    for (int s = 0; s < NB; s += 2) {
        // even k-tile s from buffer 0; the odd k-tile s + 1 staged into buffer 1
        q8l_load<1>(S, xrow, wrow, s + 1, hw);
        __builtin_amdgcn_sched_barrier(0);   // (the loads first: their latency runs under the WMMAs)
        q8l_mul(LX[0], LW[0], LD[0], tb, rb, lane, acc, Pp, dp);
        __builtin_amdgcn_sched_barrier(0);
        q8l_store<1>(S, LX[1], LW[1], LD[1], tid, iw, hw);
        __syncthreads();
        // odd k-tile s + 1 from buffer 1; the next even k-tile staged into buffer 0 (the last pair restages s: no branch)
        const int sn = s + 2 < NB ? s + 2 : s;
        q8l_load<0>(S, xrow, wrow, sn, hw);
        __builtin_amdgcn_sched_barrier(0);
        q8l_mul(LX[1], LW[1], LD[1], tb, rb, lane, acc, Pp, dp);
        __builtin_amdgcn_sched_barrier(0);
        q8l_store<0>(S, LX[0], LW[0], LD[0], tid, iw, hw);
        __syncthreads();
    }
#pragma unroll
    for (int rt = 0; rt < 2; rt++)   // (the last k-tile's last token tile)
#pragma unroll
        for (int v = 0; v < 8; v++) acc[3][rt][v] = __builtin_fmaf(dp[rt][v & 1], Pp[rt][v], acc[3][rt][v]);
    const int l = lane & 15;
#pragma unroll
    for (int rt = 0; rt < 2; rt++) {
        const int r = r0 + rb + rt * 16 + l;
        if (r >= R) continue;
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int v = 0; v < 8; v++) {
                const int t = t0 + tb + mt * 16 + 2 * v + (lane >> 4);
                if (t < ntok) Y[(size_t)t * ldy + r] = acc[mt][rt][v];
            }
    }
}
