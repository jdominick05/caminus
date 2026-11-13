// T-token forms of the plain Qwen3.8-27B GSQ-RCO files' one-round-trip tiles (the GSQ verify module,
// kernels/k3_q35_gsq_ver.hip, through kernels/k3_q35t_ops.h q35t_op_mv under K3_Q3M_VER + K3_Q35_GSQ): IQ2_XXS
// (own_quant_iq.h own_rows_iqg_1rt, format 2), IQ2_XS (own_quant_iq2xs.h), Q2_K (own_quant_q2k.h), IQ1_M
// (own_quant_iq1m.h). Q4_K has Swift's (own_quant_mt.h own_rows_q4k_1rt_mt2).
// As own_quant_tcap_mt.h's forms: a pass's weight words are loaded once for the pass's tokens, one row at a time; per
// token j its activations are read at x16 + xo(j) and the one-token tile's float statements run in that tile's order (its
// K3 form <0, 2, 1, 1>), into acc[j][r]; the wave sums at the end. What is settled once per row and pass is integer (signed
// grid words, codes) or the one-token tile's own products computed once (d * scale): each token's sums are the one-token
// tile's bits.
//   nt (wave-uniform): tokens of this pass, 1 .. OWN_MT;  res[j][r]: token j's sum of row row0 + r, in every lane.
#pragma once
#include "own_quant_mt.h"
#include "own_quant_iq.h"
#include "own_quant_iq2xs.h"
#include "own_quant_q2k.h"
#include "own_quant_iq1m.h"

#define GSQ_MT_ACC_INIT                                                                                                \
    float acc[OWN_MT][RW];                                                                                             \
    _Pragma("unroll") for (int j = 0; j < OWN_MT; j++)                                                                 \
    _Pragma("unroll") for (int r = 0; r < RW; r++) acc[j][r] = 0.0f;

#define GSQ_MT_RESULT                                                                                                  \
    _Pragma("unroll") for (int j = 0; j < OWN_MT; j++) {                                                               \
        if (j >= nt) break;                                                                                            \
        _Pragma("unroll") for (int r = 0; r < RW; r++) res[j][r] = wave_sum(acc[j][r]);                                \
    }

// token j's 16 activation words of unit uu (the loads kept after token j - 1's sums, as the ThinkingCap forms)
#define GSQ_MT_LOAD_X(j)                                                                                               \
    __builtin_amdgcn_sched_barrier(0);                                                                                 \
    u32x4_t xv[4];                                                                                                     \
    _Pragma("unroll") for (int qd = 0; qd < 4; qd++) xv[qd] = *(const u32x4_t *)(x16 + xo(j) + 32 * uu + 8 * qd);       \
    __builtin_amdgcn_sched_barrier(0);                                                                                 \
    if (u >= NUNITS) continue;                                                                                         \
    const uint32_t xw[16] = {xv[0].x, xv[0].y, xv[0].z, xv[0].w, xv[1].x, xv[1].y, xv[1].z, xv[1].w,                   \
                             xv[2].x, xv[2].y, xv[2].z, xv[2].w, xv[3].x, xv[3].y, xv[3].z, xv[3].w};

// IQ2_XXS (record 8: 4 index bytes | four 7-bit sign codes and the scale << 28; header d): the 8 grid reads (each index
// twice: its entry's two words in IQ_G at IQ_G2X) and the signs settled once per row into 8 signed words, d * scale once;
// per token the tile's 8 iq_dot4 and its fma
template <int RW, class XO>
__device__ __forceinline__ void own_rows_iq2xxs_mt(const uint8_t * __restrict__ Wm, int nrows_total, int row0, int nrows,
                                                   const _Float16 * __restrict__ x16, const XO & xo, int nt, int lane,
                                                   int Kr, float (&res)[OWN_MT][RW]) {
    constexpr int RS = 8, HS = 2;
    const int NB = Kr / 256, NUNITS = NB * 8, NU = (NUNITS + 31) / 32;
    const uint8_t * A = Wm;
    const uint8_t * B = Wm + (size_t)nrows_total * NUNITS * RS;
    GSQ_MT_ACC_INIT
#pragma unroll 1
    for (int i0 = 0; i0 < NU; i0++) {
        const int u = lane + 32 * i0, uu = u < NUNITS ? u : 0;
#pragma unroll
        for (int r = 0; r < RW; r++) {
            uint32_t wv[8];
            const size_t rw = (size_t)min(row0 + r, nrows - 1);
            const u32x4_t rc = *(const u32x4_t *)(A + (rw * NUNITS + uu) * RS);
            const u32x2_t hd = *(const u32x2_t *)(B + (rw * NB + (uu >> 3)) * HS);
            const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd.x & 0xffffu));
            const uint32_t sw = rc.y, si = sw >> 28;
            const uint32_t sg = iq_signs8(sw & 127u) | (iq_signs8((sw >> 7) & 127u) << 8) |
                                (iq_signs8((sw >> 14) & 127u) << 16) | (iq_signs8((sw >> 21) & 127u) << 24);
            const uint32_t iw[2] = {__builtin_amdgcn_perm(0u, rc.x, 0x01010000u), __builtin_amdgcn_perm(0u, rc.x, 0x03030202u)};
            const uint32_t qh = 0xaau;
#pragma unroll
            for (int k = 0; k < 8; k++)
                wv[k] = iq_signed(IQ_G[(uint32_t)IQ_G2X + (((iw[k >> 2] >> (8 * (k & 3))) & 0xffu) | (((qh >> k) & 1u) << 8))], sg >> (4 * k));
            const float dsc = d * ((float)(1u + 2u * si) * 0.125f);
#pragma unroll
            for (int j = 0; j < OWN_MT; j++) {
                if (j >= nt) break;
                GSQ_MT_LOAD_X(j)
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
    GSQ_MT_RESULT
}

// IQ2_XS (record 9: 4 u16 = a 9-bit index | a 7-bit sign code << 9 | the scale byte; header d): the 8 grid words (two an
// entry, IQ2XS_G) and the signs settled once per row into 8 signed words, the two halves' d * scale once; per token the
// tile's 8 iq_dot4 into its two half sums, then its two fmas
template <int RW, class XO>
__device__ __forceinline__ void own_rows_iq2xs_mt(const uint8_t * __restrict__ Wm, int nrows_total, int row0, int nrows,
                                                  const _Float16 * __restrict__ x16, const XO & xo, int nt, int lane,
                                                  int Kr, float (&res)[OWN_MT][RW]) {
    constexpr int RS = 9, HS = 2;
    const int NB = Kr / 256, NUNITS = NB * 8, NU = (NUNITS + 31) / 32;
    const uint8_t * A = Wm;
    const uint8_t * B = Wm + (size_t)nrows_total * NUNITS * RS;
    GSQ_MT_ACC_INIT
#pragma unroll 1
    for (int i0 = 0; i0 < NU; i0++) {
        const int u = lane + 32 * i0, uu = u < NUNITS ? u : 0;
#pragma unroll
        for (int r = 0; r < RW; r++) {
            uint32_t wv[8];
            const size_t rw = (size_t)min(row0 + r, nrows - 1);
            const u32x4_t rc = *(const u32x4_t *)(A + (rw * NUNITS + uu) * RS);
            const uint32_t hd = *(const uint32_t *)(B + (rw * NB + (uu >> 3)) * HS);
            const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd & 0xffffu));
            const uint32_t q[4] = {rc.x & 0xffffu, rc.x >> 16, rc.y & 0xffffu, rc.y >> 16};
            const uint32_t scb = rc.z & 0xffu;
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const uint32_t e = q[k] & 511u, sg = iq_signs8(q[k] >> 9);
                wv[2 * k] = iq_signed(IQ2XS_G[e], sg);
                wv[2 * k + 1] = iq_signed(IQ2XS_G[512 + e], sg >> 4);
            }
            const float d0 = d * ((float)(1u + 2u * (scb & 0xfu)) * 0.125f);
            const float d1 = d * ((float)(1u + 2u * (scb >> 4)) * 0.125f);
#pragma unroll
            for (int j = 0; j < OWN_MT; j++) {
                if (j >= nt) break;
                GSQ_MT_LOAD_X(j)
                float s[2] = {0.0f, 0.0f};
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    __builtin_amdgcn_sched_barrier(0);   // (own_rows_iq2s_mt's fence: without it 105 VGPRs, past the cap)
                    s[k >> 1] = iq_dot4(wv[2 * k], xw[4 * k], xw[4 * k + 1], s[k >> 1]);
                    s[k >> 1] = iq_dot4(wv[2 * k + 1], xw[4 * k + 2], xw[4 * k + 3], s[k >> 1]);
                }
                acc[j][r] = __builtin_fmaf(d0, s[0], acc[j][r]);
                acc[j][r] = __builtin_fmaf(d1, s[1], acc[j][r]);
            }
        }
    }
    GSQ_MT_RESULT
}

// Q2_K (record 10: the unit's codes four a byte | its two scale bytes; header 4: d, dmin): the row's codes and its four
// scale products (d sc0, -(dm m0), d sc1, -(dm m1): the tile's operands) once; per token the activations' two 16-value
// sums and the tile's sixteen fdot2 on the code pairs (integer unpacking, again per token), then its four fmas in order
template <int RW, class XO>
__device__ __forceinline__ void own_rows_q2k_mt(const uint8_t * __restrict__ Wm, int nrows_total, int row0, int nrows,
                                                const _Float16 * __restrict__ x16, const XO & xo, int nt, int lane,
                                                int Kr, float (&res)[OWN_MT][RW]) {
    constexpr int RS = 10, HS = 4;
    const int NB = Kr / 256, NUNITS = NB * 8, NU = (NUNITS + 31) / 32;
    const uint8_t * A = Wm;
    const uint8_t * B = Wm + (size_t)nrows_total * NUNITS * RS;
    const h2_t k1024 = {(_Float16)1024.0f, (_Float16)1024.0f}, one = {(_Float16)1.0f, (_Float16)1.0f};
    GSQ_MT_ACC_INIT
#pragma unroll 1
    for (int i0 = 0; i0 < NU; i0++) {
        const int u = lane + 32 * i0, uu = u < NUNITS ? u : 0;
#pragma unroll
        for (int r = 0; r < RW; r++) {
            const size_t rw = (size_t)min(row0 + r, nrows - 1);
            const u32x4_t rc = *(const u32x4_t *)(A + (rw * NUNITS + uu) * RS);
            const uint32_t hd = *(const uint32_t *)(B + (rw * NB + (uu >> 3)) * HS);
            const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd & 0xffffu));
            const float dm = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd >> 16));
            const uint32_t cw[2] = {rc.x, rc.y}, sb = rc.z & 0xffffu;
            float ca[2], cm[2];                              // d sc_g, -(dm m_g)
#pragma unroll
            for (int g = 0; g < 2; g++) {
                const uint32_t s = (sb >> (8 * g)) & 0xffu;
                ca[g] = d * (float)(s & 0xfu); cm[g] = -(dm * (float)(s >> 4));
            }
#pragma unroll
            for (int j = 0; j < OWN_MT; j++) {
                if (j >= nt) break;
                GSQ_MT_LOAD_X(j)
                float sx[2] = {0.0f, 0.0f};
#pragma unroll
                for (int w = 0; w < 16; w++) sx[w >> 3] = __builtin_amdgcn_fdot2(as_h2(xw[w]), one, sx[w >> 3], false);
                float sq[2] = {0.0f, 0.0f};
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    if (!(k & 1)) __builtin_amdgcn_sched_barrier(0);
                    const uint32_t b = (cw[k >> 2] >> (8 * (k & 3))) & 0xffu;
                    sq[k >> 2] = __builtin_amdgcn_fdot2(as_h2(q2k_pair(b, 0)) - k1024, as_h2(xw[2 * k]), sq[k >> 2], false);
                    sq[k >> 2] = __builtin_amdgcn_fdot2(as_h2(q2k_pair(b, 4)) - k1024, as_h2(xw[2 * k + 1]), sq[k >> 2], false);
                }
#pragma unroll
                for (int g = 0; g < 2; g++) {
                    acc[j][r] = __builtin_fmaf(ca[g], sq[g], acc[j][r]);
                    acc[j][r] = __builtin_fmaf(cm[g], sx[g], acc[j][r]);
                }
            }
        }
    }
    GSQ_MT_RESULT
}

// 8 grid bytes (each 0 .. 2: g + 1) -> 16 bits, 2 per byte; and back (the same bytes)
__device__ __forceinline__ uint32_t iq1m_pack8(uint64_t g) {
    uint32_t c = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) c |= (uint32_t)((g >> (8 * i)) & 3u) << (2 * i);
    return c;
}
__device__ __forceinline__ uint32_t iq1m_unpack4(uint32_t c) {   // 4 codes (8 bits) -> 4 bytes
    return (c | (c << 6) | (c << 12) | (c << 18)) & 0x03030303u;
}

// IQ1_M (record 8: 4 index bytes | 2 qh bytes | the two scales; header d): the row's 4 grid entries read once and held
// packed (2 bits a byte: 2 words), each token's group bytes unpacked from them (the same bytes the one-token tile reads);
// each group's delta, d (2 s + 1) and activation sum recomputed per token, the one-token tile's operations in its order:
// the same bits. Holding the 8 grid words and the 8 scale floats per row took k3_run to 120 VGPRs, the words alone 99,
// reading the grid per token 97, the one-token tile run once per token 102
template <int RW, class XO>
__device__ __forceinline__ void own_rows_iq1m_mt(const uint8_t * __restrict__ Wm, int nrows_total, int row0, int nrows,
                                                 const _Float16 * __restrict__ x16, const XO & xo, int nt, int lane,
                                                 int Kr, float (&res)[OWN_MT][RW]) {
    constexpr int RS = 8, HS = 2;
    const int NB = Kr / 256, NUNITS = NB * 8, NU = (NUNITS + 31) / 32;
    const uint8_t * A = Wm;
    const uint8_t * B = Wm + (size_t)nrows_total * NUNITS * RS;
    const h2_t k1025 = {(_Float16)1025.0f, (_Float16)1025.0f}, one = {(_Float16)1.0f, (_Float16)1.0f};
    GSQ_MT_ACC_INIT
#pragma unroll 1
    for (int i0 = 0; i0 < NU; i0++) {
        const int u = lane + 32 * i0, uu = u < NUNITS ? u : 0;
#pragma unroll
        for (int r = 0; r < RW; r++) {
            const size_t rw = (size_t)min(row0 + r, nrows - 1);
            const u32x2_t rc = *(const u32x2_t *)(A + (rw * NUNITS + uu) * RS);
            const uint32_t hd = *(const uint32_t *)(B + (rw * NB + (uu >> 3)) * HS);
            const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd & 0xffffu));
            const uint32_t qs = rc.x, qh = rc.y & 0xffffu, sc = (rc.y >> 16) & 0xffu;
            uint32_t pk[2] = {0u, 0u};
#pragma unroll
            for (int k = 0; k < 4; k++)
                pk[k >> 1] |= iq1m_pack8(IQ1_GRID[((qs >> (8 * k)) & 0xffu) | (((qh >> (4 * k)) & 7u) << 8)]) << (16 * (k & 1));
#pragma unroll
            for (int j = 0; j < OWN_MT; j++) {
                if (j >= nt) break;
                GSQ_MT_LOAD_X(j)
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    __builtin_amdgcn_sched_barrier(0);
                    float t = __builtin_amdgcn_fdot2(as_h2(xw[4 * k]), one, 0.0f, false);
                    t = __builtin_amdgcn_fdot2(as_h2(xw[4 * k + 1]), one, t, false);
                    t = __builtin_amdgcn_fdot2(as_h2(xw[4 * k + 2]), one, t, false);
                    const float sxk = __builtin_amdgcn_fdot2(as_h2(xw[4 * k + 3]), one, t, false);
                    const uint32_t nib = (qh >> (4 * k)) & 0xfu;
                    const uint32_t c16 = (pk[k >> 1] >> (16 * (k & 1))) & 0xffffu;
                    const uint32_t g0 = iq1m_unpack4(c16 & 0xffu), g1 = iq1m_unpack4(c16 >> 8);
                    const float delta = (nib & 8u) ? -0.125f : 0.125f;
                    const float dl = d * (float)(2u * ((sc >> (3 * (k >> 1))) & 7u) + 1u);
                    float sg = 0.0f;
                    sg = __builtin_amdgcn_fdot2(as_h2(__builtin_amdgcn_perm(0x64646464u, g0, 0x05010400u)) - k1025, as_h2(xw[4 * k]), sg, false);
                    sg = __builtin_amdgcn_fdot2(as_h2(__builtin_amdgcn_perm(0x64646464u, g0, 0x05030402u)) - k1025, as_h2(xw[4 * k + 1]), sg, false);
                    sg = __builtin_amdgcn_fdot2(as_h2(__builtin_amdgcn_perm(0x64646464u, g1, 0x05010400u)) - k1025, as_h2(xw[4 * k + 2]), sg, false);
                    sg = __builtin_amdgcn_fdot2(as_h2(__builtin_amdgcn_perm(0x64646464u, g1, 0x05030402u)) - k1025, as_h2(xw[4 * k + 3]), sg, false);
                    acc[j][r] = __builtin_fmaf(dl, __builtin_fmaf(delta, sxk, sg), acc[j][r]);
                }
            }
        }
    }
    GSQ_MT_RESULT
}
