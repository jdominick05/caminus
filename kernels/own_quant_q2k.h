// Caminus's own quantized matvec for GGUF's Q2_K (the Qwen3.8-27B GSQ-RCO files), in the one-round-trip form of
// kernels/own_quant_iq.h's tiles: fp16 activations, fp32 sums, RW rows per wave, a lane owns units of 32 consecutive
// values, every load of a pass issued before any arithmetic. Written from the block layout (the host decoder:
// python/caminus/own_dequant.py q2_k, bitwise gguf-py's); no ggml-cuda code.
//
// Q2_K: 256 values per 84-byte block: scales[16] (a scale nibble | a min nibble << 4, one per 16 values), qs[64] (2-bit
// codes), d (fp16), dmin (fp16). weight = (d * sc) * q - (dmin * m). A group of 16 values then sums
// (d sc) * sum(q x) - (dmin m) * sum(x).
// Read in Caminus's record layout (caminus.engine.own_layout.q2k_records): plane A = a 10-byte record per unit of 32
// values (its codes packed four per byte, value 4 k + j at bits 2 j of byte k, then its two scale bytes), plane B = a
// 4-byte header per block (d, dmin), then 16 bytes of padding. A unit is ONE 16-byte load and its header one 4-byte load;
// in the GGUF block its codes are one bit pair of 32 bytes three other units share (4x the bytes per unit, read in place).
// A code pair becomes two fp16 1024 + q (exact) by one perm-free mask; minus 1024 is q, exact; the fdot2 sums in fp32.
#pragma once
#include "own_quant.h"

// two fp16 lanes 1024 + c0, 1024 + c1 from a byte's codes at bit offsets o0, o0 + 2 (c in 0 .. 3)
__device__ __forceinline__ uint32_t q2k_pair(uint32_t b, int o0) {
    return 0x64006400u | ((b >> o0) & 3u) | (((b >> (o0 + 2)) & 3u) << 16);
}

// K = 0: the row length is the runtime Kr; res: the rows' sums are returned in res[RW] (every lane) and nothing is
// stored; UP units per lane per pass, UNR the pass loop's unroll count (own_rows_iqg_1rt's parameters).
// OWN_READONLY (measurement builds): the same loads folded by xor, no decode.
template <int K, int RW, int UP = 2, int UNR = 64>
__device__ __forceinline__ void own_rows_q2k_1rt(const uint8_t * __restrict__ Wm, int nrows_total, int row0, int nrows,
                                                 const _Float16 * __restrict__ x16, float * __restrict__ out, int lane,
                                                 int Kr = K, float * res = nullptr) {
    constexpr int RS = 10, HS = 4;
    const int NB = (K ? K : Kr) / 256, NUNITS = NB * 8, NU = (NUNITS + 31) / 32;
    const uint8_t * A = Wm;
    const uint8_t * B = Wm + (size_t)nrows_total * NUNITS * RS;
    const h2_t k1024 = {(_Float16)1024.0f, (_Float16)1024.0f}, one = {(_Float16)1.0f, (_Float16)1.0f};
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
            // sum(x) of each 16-value group: activation-only, shared by the RW rows
            float sx[2] = {0.0f, 0.0f};
#pragma unroll
            for (int w = 0; w < 16; w++) sx[w >> 3] = __builtin_amdgcn_fdot2(as_h2(xw[w]), one, sx[w >> 3], false);
#pragma unroll
            for (int r = 0; r < RW; r++) {
                if (UNR == 1) __builtin_amdgcn_sched_barrier(0);
                const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd[r][i] & 0xffffu));
                const float dm = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd[r][i] >> 16));
                const uint32_t cw[2] = {rc[r][i].x, rc[r][i].y}, sb = rc[r][i].z & 0xffffu;
                float sq[2] = {0.0f, 0.0f};                  // sum(q x) of values 0 .. 15, 16 .. 31
#pragma unroll
                for (int k = 0; k < 8; k++) {                // byte k: values 4 k .. 4 k + 3
                    if (UNR == 1 && !(k & 1)) __builtin_amdgcn_sched_barrier(0);   // (K3 form: own_quant_iq.h's fence)
                    const uint32_t b = (cw[k >> 2] >> (8 * (k & 3))) & 0xffu;
                    sq[k >> 2] = __builtin_amdgcn_fdot2(as_h2(q2k_pair(b, 0)) - k1024, as_h2(xw[2 * k]), sq[k >> 2], false);
                    sq[k >> 2] = __builtin_amdgcn_fdot2(as_h2(q2k_pair(b, 4)) - k1024, as_h2(xw[2 * k + 1]), sq[k >> 2], false);
                }
#pragma unroll
                for (int g = 0; g < 2; g++) {
                    const uint32_t s = (sb >> (8 * g)) & 0xffu;
                    acc[r] = __builtin_fmaf(d * (float)(s & 0xfu), sq[g], acc[r]);
                    acc[r] = __builtin_fmaf(-(dm * (float)(s >> 4)), sx[g], acc[r]);
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
