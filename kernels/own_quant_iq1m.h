// Caminus's own quantized matvec for GGUF's IQ1_M (the Qwen3.8-27B GSQ-RCO files: one matrix), in the one-round-trip
// form of kernels/own_quant_iq.h's tiles: fp16 activations, fp32 sums, RW rows per wave, a lane owns units of 32
// consecutive values, every load of a pass issued before any arithmetic. Written from the block layout (the host decoder:
// python/caminus/own_dequant.py iq1_m, bitwise gguf-py's); no ggml-cuda code.
//
// IQ1_M: 256 values per 56-byte block: qs[32] (the low 8 bits of an 11-bit grid index per 8 values), qh[16] (a nibble per
// 8 values: the index's 3 high bits | the delta's sign << 3), scales (four u16: a 3-bit scale per 16 values in bits 0-11,
// the block's fp16 d over the four top nibbles). weight = (d (2 s + 1)) * (g +- 0.125), g in -1 / 0 / +1. A group of 8
// then sums (d (2 s + 1)) * (sum(g x) +- 0.125 sum(x)).
// Read in Caminus's record layout (caminus.engine.own_layout.iq1m_records): plane A = an 8-byte record per unit of 32
// values (4 index bytes | 2 qh bytes | its two scales s0 | s1 << 3 | 0), plane B = the block's d (reassembled on the host),
// then 16 bytes of padding. The grid (2048 x 8 values, 16 KB, kernels/iq1_grid.h stores g + 1) is read from global
// memory: one matrix uses it, and 16 KB of local memory would cost the decode module its occupancy.
#pragma once
#include "own_quant.h"
#include "iq1_grid.h"

template <int K, int RW, int UP = 2, int UNR = 64>
__device__ __forceinline__ void own_rows_iq1m_1rt(const uint8_t * __restrict__ Wm, int nrows_total, int row0, int nrows,
                                                  const _Float16 * __restrict__ x16, float * __restrict__ out, int lane,
                                                  int Kr = K, float * res = nullptr) {
    constexpr int RS = 8, HS = 2;
    const int NB = (K ? K : Kr) / 256, NUNITS = NB * 8, NU = (NUNITS + 31) / 32;
    const uint8_t * A = Wm;
    const uint8_t * B = Wm + (size_t)nrows_total * NUNITS * RS;
    const h2_t k1025 = {(_Float16)1025.0f, (_Float16)1025.0f}, one = {(_Float16)1.0f, (_Float16)1.0f};
    float acc[RW];
#pragma unroll
    for (int r = 0; r < RW; r++) acc[r] = 0.0f;
#pragma unroll UNR
    for (int i0 = 0; i0 < NU; i0 += UP) {
        u32x2_t rc[RW][UP];
        uint32_t hd[RW][UP];
        u32x4_t xv[UP][4];
#pragma unroll
        for (int i = 0; i < UP; i++) {                       // every weight and activation load of the pass
            const int u = lane + 32 * (i0 + i), uu = (i0 + i < NU && u < NUNITS) ? u : 0;
#pragma unroll
            for (int r = 0; r < RW; r++) {
                const size_t rw = (size_t)min(row0 + r, nrows - 1);
                rc[r][i] = *(const u32x2_t *)(A + (rw * NUNITS + uu) * RS);
                hd[r][i] = *(const uint32_t *)(B + (rw * NB + (uu >> 3)) * HS);
            }
#pragma unroll
            for (int qd = 0; qd < 4; qd++) xv[i][qd] = *(const u32x4_t *)(x16 + 32 * uu + 8 * qd);
        }
#if defined(OWN_READONLY)
#pragma unroll
        for (int i = 0; i < UP; i++) {
            const u32x4_t xo = xv[i][0] ^ xv[i][1] ^ xv[i][2] ^ xv[i][3];
#pragma unroll
            for (int r = 0; r < RW; r++)
                acc[r] += (float)((rc[r][i].x ^ rc[r][i].y ^ hd[r][i] ^ xo.x ^ xo.y ^ xo.z ^ xo.w) & 0xffu);
        }
        continue;
#endif
#pragma unroll
        for (int i = 0; i < UP; i++) {
            const int u = lane + 32 * (i0 + i);
            if (i0 + i >= NU || u >= NUNITS) continue;
            const uint32_t xw[16] = {xv[i][0].x, xv[i][0].y, xv[i][0].z, xv[i][0].w, xv[i][1].x, xv[i][1].y, xv[i][1].z, xv[i][1].w,
                                     xv[i][2].x, xv[i][2].y, xv[i][2].z, xv[i][2].w, xv[i][3].x, xv[i][3].y, xv[i][3].z, xv[i][3].w};
            float sx[4];                                     // sum(x) per group of 8: activation-only, shared by the RW rows
#pragma unroll
            for (int k = 0; k < 4; k++) {
                float t = __builtin_amdgcn_fdot2(as_h2(xw[4 * k]), one, 0.0f, false);
                t = __builtin_amdgcn_fdot2(as_h2(xw[4 * k + 1]), one, t, false);
                t = __builtin_amdgcn_fdot2(as_h2(xw[4 * k + 2]), one, t, false);
                sx[k] = __builtin_amdgcn_fdot2(as_h2(xw[4 * k + 3]), one, t, false);
            }
#pragma unroll
            for (int r = 0; r < RW; r++) {
                if (UNR == 1) __builtin_amdgcn_sched_barrier(0);
                const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd[r][i] & 0xffffu));
                const uint32_t qs = rc[r][i].x, qh = rc[r][i].y & 0xffffu, sc = (rc[r][i].y >> 16) & 0xffu;
#pragma unroll
                for (int k = 0; k < 4; k++) {                // group k: values 8 k .. 8 k + 7
                    const uint32_t nib = (qh >> (4 * k)) & 0xfu;
                    const uint32_t e = ((qs >> (8 * k)) & 0xffu) | ((nib & 7u) << 8);
                    const uint64_t g = IQ1_GRID[e];          // 8 bytes g + 1
                    const uint32_t g0 = (uint32_t)g, g1 = (uint32_t)(g >> 32);
                    float sg = 0.0f;
                    sg = __builtin_amdgcn_fdot2(as_h2(__builtin_amdgcn_perm(0x64646464u, g0, 0x05010400u)) - k1025, as_h2(xw[4 * k]), sg, false);
                    sg = __builtin_amdgcn_fdot2(as_h2(__builtin_amdgcn_perm(0x64646464u, g0, 0x05030402u)) - k1025, as_h2(xw[4 * k + 1]), sg, false);
                    sg = __builtin_amdgcn_fdot2(as_h2(__builtin_amdgcn_perm(0x64646464u, g1, 0x05010400u)) - k1025, as_h2(xw[4 * k + 2]), sg, false);
                    sg = __builtin_amdgcn_fdot2(as_h2(__builtin_amdgcn_perm(0x64646464u, g1, 0x05030402u)) - k1025, as_h2(xw[4 * k + 3]), sg, false);
                    const float delta = (nib & 8u) ? -0.125f : 0.125f;
                    const float dl = d * (float)(2u * ((sc >> (3 * (k >> 1))) & 7u) + 1u);
                    acc[r] = __builtin_fmaf(dl, __builtin_fmaf(delta, sx[k], sg), acc[r]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RW; r++) {
        const float s = wave_sum(acc[r]);
        if (res) res[r] = s;
        else if (lane == 0 && row0 + r < nrows) out[row0 + r] = s;
    }
}
