// T-token forms of ThinkingCap's one-round-trip tiles (the dense qwen35 line's ThinkingCap verify module,
// kernels/k3_q35_tcap_ver.hip, through kernels/k3_q35t_ops.h q35t_op_mv under K3_Q3M_VER): IQ4_XS (own_quant_iq.h),
// IQ3_XXS / IQ3_S (own_rows_iqg_1rt, the runtime format), IQ2_S (own_quant_iq2s.h), Q5_K (own_quant_q5k.h). Q6_K has its
// own (own_quant_mt.h own_rows_q6k_planes_mt2).
// As own_quant_mt.h's mt2 forms: a pass's weight words are loaded once for the pass's tokens; per token j its activations
// are read at x16 + xo(j) and the one-token tile's float statements run in that tile's order (its K3 form <0, 2, 1, 1>:
// UP 1, UNR 1, so the dot helpers' fences on), into acc[j][r]; the wave sums at the end. Each token's sums are the
// one-token tile's bits. What is decoded once per pass and row is integer (the IQ grids' signed weight bytes) or the
// one-token tile's own products computed once (d * scale): neither changes a bit.
//   nt (wave-uniform): tokens of this pass, 1 .. OWN_MT;  res[j][r]: token j's sum of row row0 + r, in every lane.
#pragma once
#include "own_quant_mt.h"
#include "own_quant_iq.h"
#include "own_quant_iq2s.h"
#include "own_quant_q5k.h"

// IQ4_XS: each row's 32 nibbles looked up once per pass (iq4_lut: 8 words of kv + 128 bytes), then per token the
// one-token tile's iq4_dot16 statements on those words (the perms, the fp16 offsets, the fdot2s in its order: the same
// bits); d * (scale - 32) once per row. Looking the nibbles up again per token made T = 4 cost 2.3-2.7x one token
// (ThinkingCap's largest verify cost; benchmarks/q35_verify_trace.py, XT)
__device__ __forceinline__ float iq4_dot16u(const uint32_t (&u)[4], const u32x4_t & xa, const u32x4_t & xb, float s) {
    const h2_t off = {(_Float16)1152.0f, (_Float16)1152.0f};
    const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
    for (int w = 0; w < 4; w++) {
        const uint32_t p01 = __builtin_amdgcn_perm(0u, u[w], 0x0c010c00u) | 0x64006400u;
        const uint32_t p23 = __builtin_amdgcn_perm(0u, u[w], 0x0c030c02u) | 0x64006400u;
        s = __builtin_amdgcn_fdot2(as_h2(p01) - off, as_h2(xw[2 * w]), s, false);
        s = __builtin_amdgcn_fdot2(as_h2(p23) - off, as_h2(xw[2 * w + 1]), s, false);
    }
    return s;
}
template <int RW, class XO>
__device__ __forceinline__ void own_rows_iq4xs_mt(const uint8_t * __restrict__ W, int row0, int nrows,
                                                  const _Float16 * __restrict__ x16, const XO & xo, int nt, int lane,
                                                  int Kr, float (&res)[OWN_MT][RW]) {
    const int NB = Kr / 256, NUNITS = NB * 8, NU = (NUNITS + 31) / 32;
    const size_t RB = (size_t)NB * 136;
    float acc[OWN_MT][RW];
#pragma unroll
    for (int j = 0; j < OWN_MT; j++)
#pragma unroll
        for (int r = 0; r < RW; r++) acc[j][r] = 0.0f;
#pragma unroll 1
    for (int i0 = 0; i0 < NU; i0++) {
        const int u = lane + 32 * i0, uu = u < NUNITS ? u : 0;
        const int b = uu >> 3, ib = uu & 7;
#pragma unroll
        for (int r = 0; r < RW; r++) {                       // one row at a time (both rows' 16 words: 2 spilled VGPRs)
            uint32_t ul[4], uh[4];
            const uint8_t * blk = W + (size_t)min(row0 + r, nrows - 1) * RB + (size_t)b * 136;
            const u32x4_t qs = *(const u32x4_t *)(blk + 8 + 16 * ib);
            const u32x2_t hd = *(const u32x2_t *)blk;
            const uint32_t qw[4] = {qs.x, qs.y, qs.z, qs.w};
#pragma unroll
            for (int w = 0; w < 4; w++) { ul[w] = iq4_lut(qw[w] & 0x0f0f0f0fu); uh[w] = iq4_lut((qw[w] >> 4) & 0x0f0f0f0fu); }
            const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd.x & 0xffffu));
            const uint32_t ls = ((hd.y >> (4 * ib)) & 0xfu) | ((((hd.x >> 16) >> (2 * ib)) & 3u) << 4);
            const float ds = d * (float)((int)ls - 32);
#pragma unroll
            for (int j = 0; j < OWN_MT; j++) {
                if (j >= nt) break;
                __builtin_amdgcn_sched_barrier(0);   // (token j's loads after token j - 1's sums: own_rows_iq3_mt's reason)
                const _Float16 * xp = x16 + xo(j) + 32 * uu;
                const u32x4_t xl0 = *(const u32x4_t *)xp, xl1 = *(const u32x4_t *)(xp + 8);
                const u32x4_t xh0 = *(const u32x4_t *)(xp + 16), xh1 = *(const u32x4_t *)(xp + 24);
                __builtin_amdgcn_sched_barrier(0);
                if (u >= NUNITS) continue;
                float s = iq4_dot16u(ul, xl0, xl1, 0.0f);
                s = iq4_dot16u(uh, xh0, xh1, s);
                acc[j][r] = __builtin_fmaf(ds, s, acc[j][r]);
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

// IQ3_XXS (fmt 0) / IQ3_S (fmt 1), the record layout: the 8 grid reads and the signs settled once per row and pass into 8
// signed weight words (the one-token tile's iq_signed operands), d * scale once; per token 8 iq_dot4 in the tile's order
template <int RW, class XO>
__device__ __forceinline__ void own_rows_iq3_mt(const uint8_t * __restrict__ Wm, int nrows_total, int row0, int nrows,
                                                const _Float16 * __restrict__ x16, const XO & xo, int nt, int lane,
                                                int Kr, float (&res)[OWN_MT][RW], int fmt) {
    const bool f3s = fmt == 1;
    const int RS = f3s ? 13 : 12, HS = f3s ? 6 : 2;
    const int NB = Kr / 256, NUNITS = NB * 8, NU = (NUNITS + 31) / 32;
    const uint8_t * A = Wm;
    const uint8_t * B = Wm + (size_t)nrows_total * NUNITS * RS;
    float acc[OWN_MT][RW];
#pragma unroll
    for (int j = 0; j < OWN_MT; j++)
#pragma unroll
        for (int r = 0; r < RW; r++) acc[j][r] = 0.0f;
#pragma unroll 1
    for (int i0 = 0; i0 < NU; i0++) {
        const int u = lane + 32 * i0, uu = u < NUNITS ? u : 0, ib = uu & 7;
        // one row at a time (its 8 signed words live, the tokens' activations read again for the next row): two rows' words
        // at once spilled the module (k3_run 96 VGPRs at the cap with IQ3 alone, 34 spilled with every format). A row's
        // statements and their order are the one-token tile's either way
#pragma unroll
        for (int r = 0; r < RW; r++) {
            uint32_t wv[8];
            float dsc;
            const size_t rw = (size_t)min(row0 + r, nrows - 1);
            const u32x4_t rc = *(const u32x4_t *)(A + (rw * NUNITS + uu) * RS);
            const u32x2_t hd = *(const u32x2_t *)(B + (rw * NB + (uu >> 3)) * HS);
            const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd.x & 0xffffu));
            const uint32_t sw = rc.z, scn = (hd.x >> 16) | (hd.y << 16);
            uint32_t qh = 0u, gb = IQ_G3X, sg = sw, si = sw >> 28;
            if (f3s) { qh = rc.w; gb = IQ_G3S; si = (scn >> (4 * ib)) & 0xfu; }
            else sg = iq_signs8(sw & 127u) | (iq_signs8((sw >> 7) & 127u) << 8) |
                      (iq_signs8((sw >> 14) & 127u) << 16) | (iq_signs8((sw >> 21) & 127u) << 24);
            const uint32_t iw[2] = {rc.x, rc.y};
#pragma unroll
            for (int k = 0; k < 8; k++)
                wv[k] = iq_signed(IQ_G[gb + (((iw[k >> 2] >> (8 * (k & 3))) & 0xffu) | (((qh >> k) & 1u) << 8))], sg >> (4 * k));
            dsc = d * ((float)(1u + 2u * si) * (f3s ? 1.0f : 0.25f));
#pragma unroll
            for (int j = 0; j < OWN_MT; j++) {
                if (j >= nt) break;
                __builtin_amdgcn_sched_barrier(0);   // (token j's loads stay after token j - 1's sums: all 4 tokens' 16 words each hoisted spilled the module)
                u32x4_t xv[4];
#pragma unroll
                for (int qd = 0; qd < 4; qd++) xv[qd] = *(const u32x4_t *)(x16 + xo(j) + 32 * uu + 8 * qd);
                __builtin_amdgcn_sched_barrier(0);
                if (u >= NUNITS) continue;
                const uint32_t xw[16] = {xv[0].x, xv[0].y, xv[0].z, xv[0].w, xv[1].x, xv[1].y, xv[1].z, xv[1].w,
                                         xv[2].x, xv[2].y, xv[2].z, xv[2].w, xv[3].x, xv[3].y, xv[3].z, xv[3].w};
                float s = 0.0f;
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    if (!(k & 1)) __builtin_amdgcn_sched_barrier(0);
                    s = iq_dot4(wv[k], xw[2 * k], xw[2 * k + 1], s);
                }
                acc[j][r] = __builtin_fmaf(dsc, s, acc[j][r]);
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

// IQ2_S, the record layout: the 4 grid reads and the signs settled once per row and pass into 8 signed weight words, the
// two halves' d * scale once; per token the tile's 8 iq_dot4 into its two half sums, then its two fmas
template <int RW, class XO>
__device__ __forceinline__ void own_rows_iq2s_mt(const uint8_t * __restrict__ Wm, int nrows_total, int row0, int nrows,
                                                 const _Float16 * __restrict__ x16, const XO & xo, int nt, int lane,
                                                 int Kr, float (&res)[OWN_MT][RW]) {
    constexpr int RS = 10, HS = 2;
    const int NB = Kr / 256, NUNITS = NB * 8, NU = (NUNITS + 31) / 32;
    const uint8_t * A = Wm;
    const uint8_t * B = Wm + (size_t)nrows_total * NUNITS * RS;
    float acc[OWN_MT][RW];
#pragma unroll
    for (int j = 0; j < OWN_MT; j++)
#pragma unroll
        for (int r = 0; r < RW; r++) acc[j][r] = 0.0f;
#pragma unroll 1
    for (int i0 = 0; i0 < NU; i0++) {
        const int u = lane + 32 * i0, uu = u < NUNITS ? u : 0;
#pragma unroll
        for (int r = 0; r < RW; r++) {                       // one row at a time (own_rows_iq3_mt's reason: 27 spilled VGPRs with both)
            uint32_t wv[8];
            float d0, d1;
            const size_t rw = (size_t)min(row0 + r, nrows - 1);
            const u32x4_t rc = *(const u32x4_t *)(A + (rw * NUNITS + uu) * RS);
            const uint32_t hd = *(const uint16_t *)(B + (rw * NB + (uu >> 3)) * HS);
            const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd & 0xffffu));
            const uint32_t iw = rc.x, sg = rc.y, qh = rc.z & 0xffu, scb = (rc.z >> 8) & 0xffu;
#pragma unroll
            for (int g = 0; g < 4; g++) {
                const uint32_t idx = ((iw >> (8 * g)) & 0xffu) | (((qh >> (2 * g)) & 3u) << 8);
                const uint32_t e = IQ2S_G[idx >> 1] >> (16 * (idx & 1u));
                wv[2 * g] = iq_signed(iq2s_mag4(e & 0xffu), sg >> (8 * g));
                wv[2 * g + 1] = iq_signed(iq2s_mag4((e >> 8) & 0xffu), sg >> (8 * g + 4));
            }
            d0 = d * ((float)(1u + 2u * (scb & 15u)) * 0.125f);
            d1 = d * ((float)(1u + 2u * (scb >> 4)) * 0.125f);
#pragma unroll
            for (int j = 0; j < OWN_MT; j++) {
                if (j >= nt) break;
                __builtin_amdgcn_sched_barrier(0);   // (token j's loads stay after token j - 1's sums: all 4 tokens' 16 words each hoisted spilled the module)
                u32x4_t xv[4];
#pragma unroll
                for (int qd = 0; qd < 4; qd++) xv[qd] = *(const u32x4_t *)(x16 + xo(j) + 32 * uu + 8 * qd);
                __builtin_amdgcn_sched_barrier(0);
                if (u >= NUNITS) continue;
                const uint32_t xw[16] = {xv[0].x, xv[0].y, xv[0].z, xv[0].w, xv[1].x, xv[1].y, xv[1].z, xv[1].w,
                                         xv[2].x, xv[2].y, xv[2].z, xv[2].w, xv[3].x, xv[3].y, xv[3].z, xv[3].w};
                float s[2] = {0.0f, 0.0f};
#pragma unroll
                for (int g = 0; g < 4; g++) {
                    __builtin_amdgcn_sched_barrier(0);
                    s[g >> 1] = iq_dot4(wv[2 * g], xw[4 * g], xw[4 * g + 1], s[g >> 1]);
                    s[g >> 1] = iq_dot4(wv[2 * g + 1], xw[4 * g + 2], xw[4 * g + 3], s[g >> 1]);
                }
                acc[j][r] = __builtin_fmaf(d0, s[0], acc[j][r]);
                acc[j][r] = __builtin_fmaf(d1, s[1], acc[j][r]);
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

// Q5_K: the row's four scale products once (d * sc0, -dm * m0, d * sc1, -dm * m1: the tile's operands), the 5-bit values
// looked up again per token (q5k_dot16: integer); per token its activations' sums and the tile's four fmas in order
template <int RW, class XO>
__device__ __forceinline__ void own_rows_q5k_mt(const uint8_t * __restrict__ W, int row0, int nrows,
                                                const _Float16 * __restrict__ x16, const XO & xo, int nt, int lane,
                                                int Kr, float (&res)[OWN_MT][RW]) {
    const int NB = Kr / 256, NUNITS = NB * 8, NU = (NUNITS + 31) / 32;
    const size_t RB = (size_t)NB * 176;
    float acc[OWN_MT][RW];
#pragma unroll
    for (int j = 0; j < OWN_MT; j++)
#pragma unroll
        for (int r = 0; r < RW; r++) acc[j][r] = 0.0f;
#pragma unroll 1
    for (int i0 = 0; i0 < NU; i0++) {
        u32x4_t qh[RW], qs[RW];
        float f[RW][4];
        const int u = lane + 32 * i0, uu = u < NUNITS ? u : 0;
        const int b = uu >> 3, jj = (uu & 7) >> 1, h = uu & 1;
#pragma unroll
        for (int r = 0; r < RW; r++) {
            const uint8_t * blk = W + (size_t)min(row0 + r, nrows - 1) * RB + (size_t)b * 176;
            const u32x4_t hd = *(const u32x4_t *)blk;
            qh[r] = *(const u32x4_t *)(blk + 16 + 16 * h);
            qs[r] = *(const u32x4_t *)(blk + 48 + 32 * jj + 16 * h);
            const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd.x & 0xffffu));
            const float dm = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd.x >> 16));
            float sc0, m0, sc1, m1;
            q4k_sm2(hd, jj, sc0, m0, sc1, m1);
            f[r][0] = d * sc0; f[r][1] = -dm * m0; f[r][2] = d * sc1; f[r][3] = -dm * m1;
        }
#pragma unroll
        for (int j = 0; j < OWN_MT; j++) {
            if (j >= nt) break;
            __builtin_amdgcn_sched_barrier(0);   // (token j's loads after token j - 1's sums: own_rows_iq3_mt's reason)
            const _Float16 * xp = x16 + xo(j) + b * 256 + 64 * jj + 16 * h;
            const u32x4_t xl0 = *(const u32x4_t *)xp, xl1 = *(const u32x4_t *)(xp + 8);
            const u32x4_t xh0 = *(const u32x4_t *)(xp + 32), xh1 = *(const u32x4_t *)(xp + 40);
            __builtin_amdgcn_sched_barrier(0);
            if (u >= NUNITS) continue;
            __builtin_amdgcn_sched_barrier(0);
            const float sxl = sum16h(xl0, xl1), sxh = sum16h(xh0, xh1);
#pragma unroll
            for (int r = 0; r < RW; r++) {
                __builtin_amdgcn_sched_barrier(0);
                float a = acc[j][r];
                a = __builtin_fmaf(f[r][0], q5k_dot16(qs[r], qh[r], 0, 2 * jj, xl0, xl1, true), a);
                a = __builtin_fmaf(f[r][1], sxl, a);
                a = __builtin_fmaf(f[r][2], q5k_dot16(qs[r], qh[r], 4, 2 * jj + 1, xh0, xh1, true), a);
                acc[j][r] = __builtin_fmaf(f[r][3], sxh, a);
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
