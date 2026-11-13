// Caminus's own quantized matvec for GGUF's IQ2_XS (the Qwen3.8-27B GSQ-RCO files), in the one-round-trip form of
// kernels/own_quant_iq.h's grid tile: fp16 activations, fp32 sums, RW rows per wave, a lane owns units of 32 consecutive
// values, every load of a pass issued before any arithmetic. Written from the block layout (the host decoder:
// python/caminus/own_dequant.py iq2_xs, bitwise gguf-py's); no ggml-cuda code.
//
// IQ2_XS: 256 values per 74-byte block: d (fp16), qs[32] (u16 each: a 9-bit grid index | a 7-bit sign code << 9, per 8
// values), scales[8] (a nibble per 16 values: a byte's low nibble, then its high one). weight = d (0.5 + s) 0.25 * +-grid
// = d (1 + 2 s) 0.125 * +-grid, exactly; the grid entry is 8 magnitudes in {8, 25, 43} (kernels/iq2xs_grid.h).
// Read in Caminus's record layout (caminus.engine.own_layout.iq_records, IQ_REC['IQ2_XS'] = (74, 9, 2)): plane A = a
// 9-byte record per unit of 32 values (its 4 u16 | its scale byte), plane B = the block's d, then 16 bytes of padding. A
// unit is ONE 16-byte load (it runs past its record into the next bytes) and its d one 4-byte load.
// The grid (512 entries, 4 KB) lives in the workgroup's local memory in its own array, staged once per launch by
// iq2xs_stage: entry e = words e (weights 0 .. 3) and 512 + e (4 .. 7), so a lookup's address is the index plus one bit
// times 512. Its own array, not own_quant_iq.h's IQ_G: the modules that include only IQ_G keep their LDS and their code.
#pragma once
#include "own_quant_iq.h"
#include "iq2xs_grid.h"

__shared__ uint32_t IQ2XS_G[1024];
// all 256 threads of the workgroup call it (one barrier)
__device__ __forceinline__ void iq2xs_stage(int tid) {
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int e = tid + 256 * h;
        const uint64_t g = IQ2XS_GRID[e];
        IQ2XS_G[e] = (uint32_t)g; IQ2XS_G[512 + e] = (uint32_t)(g >> 32);
    }
    __syncthreads();
}

// K = 0: the row length is the runtime Kr; res: the rows' sums are returned in res[RW] (every lane) and nothing is
// stored; UP units per lane per pass, UNR the pass loop's unroll count (own_rows_iqg_1rt's parameters).
// OWN_READONLY (measurement builds): the same loads folded by xor, no decode.
template <int K, int RW, int UP = 2, int UNR = 64>
__device__ __forceinline__ void own_rows_iq2xs_1rt(const uint8_t * __restrict__ Wm, int nrows_total, int row0, int nrows,
                                                   const _Float16 * __restrict__ x16, float * __restrict__ out, int lane,
                                                   int Kr = K, float * res = nullptr) {
    constexpr int RS = 9, HS = 2;
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
            for (int r = 0; r < RW; r++) {
                const u32x4_t v = rc[r][i] ^ xo;
                acc[r] += (float)((v.x ^ v.y ^ v.z ^ v.w ^ hd[r][i]) & 0xffu);
            }
        }
        continue;
#endif
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
                const uint32_t q[4] = {rc[r][i].x & 0xffffu, rc[r][i].x >> 16, rc[r][i].y & 0xffffu, rc[r][i].y >> 16};
                const uint32_t scb = rc[r][i].z & 0xffu;
                uint32_t m[8];                               // the grid reads: 2 words (4 weights each) per entry
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    const uint32_t e = q[k] & 511u;
                    m[2 * k] = IQ2XS_G[e]; m[2 * k + 1] = IQ2XS_G[512 + e];
                }
                float s[2] = {0.0f, 0.0f};                   // values 0 .. 15 (scale nibble 0), 16 .. 31 (nibble 1)
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    const uint32_t sg = iq_signs8(q[k] >> 9);
                    s[k >> 1] = iq_dot4(iq_signed(m[2 * k], sg), xw[4 * k], xw[4 * k + 1], s[k >> 1]);
                    s[k >> 1] = iq_dot4(iq_signed(m[2 * k + 1], sg >> 4), xw[4 * k + 2], xw[4 * k + 3], s[k >> 1]);
                }
                acc[r] = __builtin_fmaf(d * ((float)(1u + 2u * (scb & 0xfu)) * 0.125f), s[0], acc[r]);
                acc[r] = __builtin_fmaf(d * ((float)(1u + 2u * (scb >> 4)) * 0.125f), s[1], acc[r]);
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
