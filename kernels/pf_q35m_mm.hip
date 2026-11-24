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
