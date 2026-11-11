// Caminus's own quantized matvec for GGUF's IQ2_S (ThinkingCap's 2.5-bit tensors and embedding), in the one-round-trip
// form of kernels/own_quant_iq.h's grid tile: fp16 activations, fp32 sums, RW rows per wave, a lane owns units of 32
// consecutive values, every load of a pass issued before any arithmetic. Written from the block layout (the host
// decoder: python/caminus/own_dequant.py iq2_s, bitwise gguf-py's); no ggml-cuda code.
// IQ2_S, 82-byte blocks of 256: d (fp16), qs[32] (the low 8 bits of a 10-bit grid index per 8 values), signs[32] (a
// bit per value), qh[8] (the indices' 2 high bits, 4 per byte: a byte per 32 values), scales[8] (a nibble per 16
// values); weight = ((d * (0.5 + s)) * 0.25) * +-grid. Read in Caminus's record layout (caminus.engine.own_layout
// iq_records): plane A = a 10-byte record per unit of 32 weights (4 index bytes | 4 sign bytes | qh | the scale byte),
// plane B = d per block, then 16 bytes of padding: a unit is ONE 16-byte load, its block's d one 2-byte load.
// The grid (1024 entries of 8 magnitudes, each 8, 25 or 43) sits in the workgroup's LDS packed to 2 bits a magnitude
// (kernels/iq2s_grid.h: 2 KB); a lookup is one LDS word, then a 4-weight group's magnitude bytes from its 8 code bits
// (spread to one per byte, then a byte permute out of the three magnitudes). Separate from own_quant_iq.h's grid body
// and its LDS array, so no existing code object changes.
#pragma once
#include "own_quant_iq.h"     // iq_signed, iq_dot4, wave_sum, the vector types
#include "iq2s_grid.h"        // IQ2S_GRID

__shared__ uint32_t IQ2S_G[512];
// all 256 threads of the workgroup call it (one barrier)
__device__ __forceinline__ void iq2s_stage(int tid) {
    IQ2S_G[tid] = IQ2S_GRID[tid]; IQ2S_G[256 + tid] = IQ2S_GRID[256 + tid];
    __syncthreads();
}
// 4 weights' 2-bit codes (bits 2k .. 2k + 1 of c: weight k) -> their 4 magnitude bytes (code 0: 8, 1: 25, 2: 43)
__device__ __forceinline__ uint32_t iq2s_mag4(uint32_t c) {
    const uint32_t t = (c | (c << 6) | (c << 12) | (c << 18)) & 0x03030303u;
    return __builtin_amdgcn_perm(0u, 0x002b1908u, t);
}

// Wm = the record layout of nrows_total rows; rows row0 .. min(row0 + RW, nrows) - 1; K = 0: the runtime Kr; res: the
// rows' sums are returned in res[RW] (every lane) and nothing is stored. The grid must be staged (iq2s_stage).
template <int K, int RW, int UP = 2, int UNR = 64>
__device__ __forceinline__ void own_rows_iq2s_1rt(const uint8_t * __restrict__ Wm, int nrows_total, int row0, int nrows,
                                                  const _Float16 * __restrict__ x16, float * __restrict__ out, int lane,
                                                  int Kr = K, float * res = nullptr) {
    constexpr int RS = 10, HS = 2;
    const int NB = (K ? K : Kr) / 256, NUNITS = NB * 8, NU = (NUNITS + 31) / 32;
    const uint8_t * A = Wm;
    const uint8_t * B = Wm + (size_t)nrows_total * NUNITS * RS;
    float acc[RW];
#pragma unroll
    for (int r = 0; r < RW; r++) acc[r] = 0.0f;
#pragma unroll UNR
    for (int i0 = 0; i0 < NU; i0 += UP) {
        u32x4_t rc[RW][UP];
        uint32_t hd[RW][UP];
        u32x4_t xv[UP][4];
#pragma unroll
        for (int i = 0; i < UP; i++) {                       // every weight and activation load of the pass
            const int u = lane + 32 * (i0 + i), uu = (i0 + i < NU && u < NUNITS) ? u : 0;
#pragma unroll
            for (int r = 0; r < RW; r++) {
                const size_t rw = (size_t)min(row0 + r, nrows - 1);
                rc[r][i] = *(const u32x4_t *)(A + (rw * NUNITS + uu) * RS);
                hd[r][i] = *(const uint16_t *)(B + (rw * NB + (uu >> 3)) * HS);
            }
#pragma unroll
            for (int qd = 0; qd < 4; qd++) xv[i][qd] = *(const u32x4_t *)(x16 + 32 * uu + 8 * qd);
        }
#pragma unroll
        for (int i = 0; i < UP; i++) {
            const int u = lane + 32 * (i0 + i);
            if (i0 + i >= NU || u >= NUNITS) continue;
            const uint32_t xw[16] = {xv[i][0].x, xv[i][0].y, xv[i][0].z, xv[i][0].w, xv[i][1].x, xv[i][1].y, xv[i][1].z, xv[i][1].w,
                                     xv[i][2].x, xv[i][2].y, xv[i][2].z, xv[i][2].w, xv[i][3].x, xv[i][3].y, xv[i][3].z, xv[i][3].w};
#pragma unroll
            for (int r = 0; r < RW; r++) {
                if (UNR == 1) __builtin_amdgcn_sched_barrier(0);
                const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd[r][i] & 0xffffu));
                const uint32_t iw = rc[r][i].x, sg = rc[r][i].y, qh = rc[r][i].z & 0xffu, scb = (rc[r][i].z >> 8) & 0xffu;
                uint32_t e[4];                               // the grid reads: one LDS word per 8 weights
#pragma unroll
                for (int g = 0; g < 4; g++) {
                    const uint32_t idx = ((iw >> (8 * g)) & 0xffu) | (((qh >> (2 * g)) & 3u) << 8);
                    e[g] = IQ2S_G[idx >> 1] >> (16 * (idx & 1u));
                }
                float s[2] = {0.0f, 0.0f};                   // the two 16-weight halves (a scale each)
#pragma unroll
                for (int g = 0; g < 4; g++) {
                    if (UNR == 1) __builtin_amdgcn_sched_barrier(0);
                    s[g >> 1] = iq_dot4(iq_signed(iq2s_mag4(e[g] & 0xffu), sg >> (8 * g)), xw[4 * g], xw[4 * g + 1], s[g >> 1]);
                    s[g >> 1] = iq_dot4(iq_signed(iq2s_mag4((e[g] >> 8) & 0xffu), sg >> (8 * g + 4)), xw[4 * g + 2], xw[4 * g + 3], s[g >> 1]);
                }
                // (0.5 + s) 0.25 = (1 + 2 s) 0.125, exactly
                acc[r] = __builtin_fmaf(d * ((float)(1u + 2u * (scb & 15u)) * 0.125f), s[0], acc[r]);
                acc[r] = __builtin_fmaf(d * ((float)(1u + 2u * (scb >> 4)) * 0.125f), s[1], acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RW; r++) {
        const float sm = wave_sum(acc[r]);
        if (res) res[r] = sm;
        else if (lane == 0 && row0 + r < nrows) out[row0 + r] = sm;
    }
}
