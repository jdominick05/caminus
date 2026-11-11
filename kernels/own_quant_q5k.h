// Caminus's own Q5_K matvec in the one-round-trip form of kernels/own_quant.h's own_rows_q4k_1rt: fp16 activations,
// fp32 sums, RW rows per wave, a lane owns units of 32 values (16 of a sub-block pair's first sub-block, the same 16
// positions of its second), every load of a pass issued before any arithmetic. Written from the block layout (the host
// decoder: python/caminus/own_dequant.py q5_k, bitwise gguf-py's); no ggml-cuda code.
// Q5_K, 176-byte blocks of 256: d (fp16), dmin (fp16), scales[12] (Q4_K's 6-bit scales and mins), qh[32], qs[128];
// value i (sub-block s = i / 32, l = i % 32) = (d * sc_s) * q - (dmin * m_s), q = its nibble (Q4_K's place in qs) | bit s
// of qh[l] << 4. The block is Q4_K's plus the 32-byte qh plane between the header and the nibbles, and 176 = 11 x 16:
// the header, a unit's 16 qh bytes and its 16 nibble bytes are each ONE aligned 16-byte load (the qh bytes of a unit are
// the same positions l as its nibble bytes). The general rows tile (own_rows_q5_k) reads 4 values per lane per load:
// ThinkingCap's 61 Q5_K matvecs per token ran at 185-300 GB/s there (RX 7900 XTX, benchmarks/own_mv_census.py).
#pragma once
#include "own_quant.h"

// sum over 16 values (4 dwords: nibble at shift sh of q, high bit at bit hs of the matching qh byte) times 16 fp16 x
// values (2 u32x4) in fp32: q4k_dot16's arithmetic on 5-bit values (fp16 1024 + q is exact for q <= 31)
__device__ __forceinline__ float q5k_dot16(const u32x4_t & q, const u32x4_t & qh, int sh, int hs, const u32x4_t & xa,
                                           const u32x4_t & xb, bool fence = false) {
    const h2_t off = {(_Float16)1024.0f, (_Float16)1024.0f};
    const uint32_t qw[4] = {q.x, q.y, q.z, q.w}, hw[4] = {qh.x, qh.y, qh.z, qh.w};
    const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; w++) {
        if (fence) __builtin_amdgcn_sched_barrier(0);
        const uint32_t n = ((qw[w] >> sh) & 0x0f0f0f0fu) | (((hw[w] >> hs) & 0x01010101u) << 4);   // 4 values as bytes
        const uint32_t p01 = __builtin_amdgcn_perm(0u, n, 0x0c010c00u) | 0x64006400u;  // bytes 0, 1 -> fp16 1024 + q
        const uint32_t p23 = __builtin_amdgcn_perm(0u, n, 0x0c030c02u) | 0x64006400u;
        s = __builtin_amdgcn_fdot2(as_h2(p01) - off, as_h2(xw[2 * w]), s, false);
        s = __builtin_amdgcn_fdot2(as_h2(p23) - off, as_h2(xw[2 * w + 1]), s, false);
    }
    return s;
}

// own_rows_q4k_1rt's parameters and pass structure; W = the GGUF rows as they are (176-byte blocks)
template <int K, int RW, int UP = 2, int UNR = 64>
__device__ __forceinline__ void own_rows_q5k_1rt(const uint8_t * __restrict__ W, int row0, int nrows,
                                                 const _Float16 * __restrict__ x16, float * __restrict__ out, int lane,
                                                 int Kr = K, float * res = nullptr) {
    const int NB = (K ? K : Kr) / 256, NUNITS = NB * 8, NU = (NUNITS + 31) / 32;
    const size_t RB = (size_t)NB * 176;
    float acc[RW];
#pragma unroll
    for (int r = 0; r < RW; r++) acc[r] = 0.0f;
#pragma unroll UNR
    for (int i0 = 0; i0 < NU; i0 += UP) {
        u32x4_t hd[RW][UP], qh[RW][UP], qs[RW][UP], xl[UP][2], xh[UP][2];
#pragma unroll
        for (int i = 0; i < UP; i++) {                       // every load of the pass, unconditionally
            const int u = lane + 32 * (i0 + i), uu = (i0 + i < NU && u < NUNITS) ? u : 0;
            const int b = uu >> 3, j = (uu & 7) >> 1, h = uu & 1;
#pragma unroll
            for (int r = 0; r < RW; r++) {
                const uint8_t * blk = W + (size_t)min(row0 + r, nrows - 1) * RB + (size_t)b * 176;
                hd[r][i] = *(const u32x4_t *)blk;
                qh[r][i] = *(const u32x4_t *)(blk + 16 + 16 * h);
                qs[r][i] = *(const u32x4_t *)(blk + 48 + 32 * j + 16 * h);
            }
            const _Float16 * xp = x16 + b * 256 + 64 * j + 16 * h;
            xl[i][0] = *(const u32x4_t *)xp; xl[i][1] = *(const u32x4_t *)(xp + 8);
            xh[i][0] = *(const u32x4_t *)(xp + 32); xh[i][1] = *(const u32x4_t *)(xp + 40);
        }
#pragma unroll
        for (int i = 0; i < UP; i++) {
            const int u = lane + 32 * (i0 + i);
            if (i0 + i >= NU || u >= NUNITS) continue;
            const int j = (u & 7) >> 1;
            if (UNR == 1) __builtin_amdgcn_sched_barrier(0);
            const float sxl = sum16h(xl[i][0], xl[i][1]), sxh = sum16h(xh[i][0], xh[i][1]);
#pragma unroll
            for (int r = 0; r < RW; r++) {
                if (UNR == 1) __builtin_amdgcn_sched_barrier(0);
                const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd[r][i].x & 0xffffu));
                const float dm = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd[r][i].x >> 16));
                float sc0, m0, sc1, m1;
                q4k_sm2(hd[r][i], j, sc0, m0, sc1, m1);
                float a = acc[r];
                a = __builtin_fmaf(d * sc0, q5k_dot16(qs[r][i], qh[r][i], 0, 2 * j, xl[i][0], xl[i][1], UNR == 1), a);
                a = __builtin_fmaf(-dm * m0, sxl, a);
                a = __builtin_fmaf(d * sc1, q5k_dot16(qs[r][i], qh[r][i], 4, 2 * j + 1, xh[i][0], xh[i][1], UNR == 1), a);
                acc[r] = __builtin_fmaf(-dm * m1, sxh, a);
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
