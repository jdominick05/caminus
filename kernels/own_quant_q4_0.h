// Caminus's own Q4_0 matvec (the dense qwen35 line's multi-token-prediction block: Swift 1.5 stores all of blk.64 as
// Q4_0), written from the block layout (python/caminus/own_dequant.py q4_0, bitwise gguf-py's); no ggml-cuda code.
// Q4_0: 32 values per 18-byte block: d (fp16), 16 bytes of nibbles; byte k's low nibble is value k, its high nibble
// value 16 + k; w = d * (q - 8). The GGUF block is 2-byte aligned (18 bytes), so the tile reads the plane layout made at
// load (python/caminus/engine/own_layout.py q4_0_planes): plane A = every block's 16 code bytes (one aligned 16-byte
// load), plane B = every block's d (fp16), each over the whole matrix, row-major. Same byte count as the GGUF tensor.
//
// One-round-trip form (kernels/own_quant.h own_rows_q4k_1rt): RW rows per wave, a lane owns units of one block (32
// values, one scale), every weight load of a pass issued before any arithmetic. fp16 activations, fp32 sums.
// The 8 offset goes into the code before the float conversion: nibble n as fp16 1024 + n (exponent 10, exact) minus
// 1032 is n - 8 exactly, so a block's sum is d times one dot of exact integers with fp16 x (v_dot2_f32_f16 rounds only
// the fp32 accumulation), with no separate 8 * sum(x) term to cancel.
// T-token form from the start (the verify passes and the drafter's catch-up rows, kernels/own_quant_mt.h's contract):
// a pass's code words are unpacked once and held while up to OWN_MT tokens' sums run in turn, each on its own
// activations (token j's at x16 + xo(j)); per token the statements and their order are the same at every nt, so a
// T-token pass gives each token the one-token pass's bits.
//   nt (wave-uniform): tokens of this pass, 1 .. OWN_MT;  res[j][r]: token j's sum of row row0 + r, in every lane.
// Per token and row and unit: s0 = the low nibbles' 8 dot2 (values 0 .. 15) in order, s1 = the high nibbles' (16 .. 31),
// acc = fma(d, s0 + s1, acc); units in lane order (u = lane + 32 i, i ascending), then the wave's xor-butterfly sum.
#pragma once
#include "own_quant.h"
#ifndef OWN_MT
#define OWN_MT 4
#endif

template <int RW, class XO>
__device__ __forceinline__ void own_rows_q4_0_1rt(const uint8_t * __restrict__ Wm, int nrows_total, int row0, int nrows,
                                                  const _Float16 * __restrict__ x16, const XO & xo, int nt, int lane,
                                                  int Kr, float (&res)[OWN_MT][RW]) {
    // Wm: the planes of a [nrows_total, Kr] matrix; rows row0 .. row0 + RW - 1 that are < nrows (one wave)
    const int NUNITS = Kr / 32, NU = (NUNITS + 31) / 32;
    const uint8_t * A = Wm;
    const uint16_t * B = (const uint16_t *)(Wm + (size_t)nrows_total * NUNITS * 16);
    const h2_t off = {(_Float16)1032.0f, (_Float16)1032.0f};   // fp16 (1024 + n) - 1032 = n - 8, exact
    float acc[OWN_MT][RW];
#pragma unroll
    for (int j = 0; j < OWN_MT; j++)
#pragma unroll
        for (int r = 0; r < RW; r++) acc[j][r] = 0.0f;
#pragma unroll 1
    for (int i0 = 0; i0 < NU; i0++) {
        const int u = lane + 32 * i0, uu = u < NUNITS ? u : 0;
        u32x4_t qs[RW];
        float d[RW];
#pragma unroll
        for (int r = 0; r < RW; r++) {                       // every weight load of the pass, unconditionally
            const size_t ub = (size_t)min(row0 + r, nrows - 1) * NUNITS + uu;
            qs[r] = *(const u32x4_t *)(A + ub * 16);
            d[r] = (float)__builtin_bit_cast(_Float16, B[ub]);
        }
        h2_t wv[RW][16];                                     // [0 .. 7]: values 0 .. 15 as (n - 8), [8 .. 15]: 16 .. 31
#pragma unroll
        for (int r = 0; r < RW; r++) {
            const uint32_t qw[4] = {qs[r].x, qs[r].y, qs[r].z, qs[r].w};
#pragma unroll
            for (int half = 0; half < 2; half++)
#pragma unroll
                for (int w = 0; w < 4; w++) {
                    const uint32_t n = (qw[w] >> (4 * half)) & 0x0f0f0f0fu;
                    wv[r][8 * half + 2 * w] = as_h2(__builtin_amdgcn_perm(0u, n, 0x0c010c00u) | 0x64006400u) - off;
                    wv[r][8 * half + 2 * w + 1] = as_h2(__builtin_amdgcn_perm(0u, n, 0x0c030c02u) | 0x64006400u) - off;
                }
        }
#pragma unroll
        for (int j = 0; j < OWN_MT; j++) {
            if (j >= nt) break;
            const _Float16 * xp = x16 + xo(j) + (size_t)uu * 32;
            const u32x4_t x0 = *(const u32x4_t *)xp, x1 = *(const u32x4_t *)(xp + 8);
            const u32x4_t x2 = *(const u32x4_t *)(xp + 16), x3 = *(const u32x4_t *)(xp + 24);
            __builtin_amdgcn_sched_barrier(0);
            if (u >= NUNITS) continue;
            const uint32_t xw[16] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w,
                                     x2.x, x2.y, x2.z, x2.w, x3.x, x3.y, x3.z, x3.w};
#pragma unroll
            for (int r = 0; r < RW; r++) {
                float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    s0 = __builtin_amdgcn_fdot2(wv[r][k], as_h2(xw[k]), s0, false);
                    s1 = __builtin_amdgcn_fdot2(wv[r][8 + k], as_h2(xw[8 + k]), s1, false);
                }
                acc[j][r] = __builtin_fmaf(d[r], s0 + s1, acc[j][r]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < OWN_MT; j++) {
        if (j >= nt) break;
#pragma unroll
        for (int r = 0; r < RW; r++) res[j][r] = wave_sum(acc[j][r]);
    }
}
