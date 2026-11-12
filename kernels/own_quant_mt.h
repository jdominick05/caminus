// T-token forms of own_quant.h's one-round-trip tiles (the qwen3moe verify module, kernels/k3_q3m_ver.hip): a pass's
// weight words are loaded once and held in registers while up to OWN_MT tokens' sums run in turn, each on its own
// activations (token j's at x16 + xo(j) fp16 values). Per token the statements are the one-token tile's
// (own_rows_q4k_1rt / own_rows_q6k_planes with K = 0, UP = 1, UNR = 1) in its order, so every sum has that tile's bits.
//   nt (wave-uniform): tokens of this pass, 1 .. OWN_MT;  res[j][r]: token j's sum of row row0 + r, in every lane.
// What a token more costs is its arithmetic, not memory (XT, T = 4, real tokens, per layer: attn_output 10.8 us for one
// token, 32.9 for four, 18.0 in a build that skipped the sums of tokens 1 .. 3 and kept their loads; k | v 12.7 / 23.9 /
// 12.7; the head 350 / 516 / 381): the weights' bytes are read once whatever the token count, and issuing token j + 1's
// activation loads ahead of token j's sums changed nothing (11.46 vs 11.54 ms per pass). Each token decodes the quant
// words again (nibbles -> fp16). The experts (OWN_UNION, k3_own_ops.h): gate | up 24.7 us for one token; for four, 53.6
// with one token repeated (8 distinct experts, every one with 4 member slots: the rows read once, 4 sums each) and
// 64.9 on real tokens (20.1 distinct of 32 slots): the member sums and the distinct experts' bytes both count.
#pragma once
#include "own_quant.h"
#ifndef OWN_MT
#define OWN_MT 4
#endif

template <int RW, class XO>
__device__ __forceinline__ void own_rows_q4k_1rt_mt(const uint8_t * __restrict__ W, int row0, int nrows,
                                                    const _Float16 * __restrict__ x16, const XO & xo, int nt, int lane,
                                                    int Kr, float (&res)[OWN_MT][RW]) {
    const int NB = Kr / 256, NUNITS = NB * 8, NU = (NUNITS + 31) / 32;
    const size_t RB = (size_t)NB * 144;
    float acc[OWN_MT][RW];
#pragma unroll
    for (int j = 0; j < OWN_MT; j++)
#pragma unroll
        for (int r = 0; r < RW; r++) acc[j][r] = 0.0f;
#pragma unroll 1
    for (int i0 = 0; i0 < NU; i0++) {
        u32x4_t hd[RW], qs[RW], xl[2], xh[2];
        const int u = lane + 32 * i0, uu = u < NUNITS ? u : 0;
        const int b = uu >> 3, jq = (uu & 7) >> 1, h = uu & 1;
#pragma unroll
        for (int r = 0; r < RW; r++) {                       // every weight load of the pass, unconditionally
            const uint8_t * blk = W + (size_t)min(row0 + r, nrows - 1) * RB + (size_t)b * 144;
            qs[r] = *(const u32x4_t *)(blk + 16 + 32 * jq + 16 * h);
            hd[r] = *(const u32x4_t *)blk;
        }
        const size_t xu = (size_t)b * 256 + 64 * jq + 16 * h;
#pragma unroll
        for (int j = 0; j < OWN_MT; j++) {
            if (j >= nt) break;
            const _Float16 * xp = x16 + xo(j) + xu;
            xl[0] = *(const u32x4_t *)xp; xl[1] = *(const u32x4_t *)(xp + 8);
            xh[0] = *(const u32x4_t *)(xp + 32); xh[1] = *(const u32x4_t *)(xp + 40);
            if (u >= NUNITS) continue;
            const int ju = (u & 7) >> 1;
            __builtin_amdgcn_sched_barrier(0);
            const float sxl = sum16h(xl[0], xl[1]), sxh = sum16h(xh[0], xh[1]);
#pragma unroll
            for (int r = 0; r < RW; r++) {
                __builtin_amdgcn_sched_barrier(0);
                const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd[r].x & 0xffffu));
                const float dm = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd[r].x >> 16));
                float sc0, m0, sc1, m1;
                q4k_sm2(hd[r], ju, sc0, m0, sc1, m1);
                float a = acc[j][r];
                a = __builtin_fmaf(d * sc0, q4k_dot16(qs[r], 0, xl[0], xl[1], true), a);
                a = __builtin_fmaf(-dm * m0, sxl, a);
                a = __builtin_fmaf(d * sc1, q4k_dot16(qs[r], 4, xh[0], xh[1], true), a);
                acc[j][r] = __builtin_fmaf(-dm * m1, sxh, a);
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

template <int RW, class XO>
__device__ __forceinline__ void own_rows_q6k_planes_mt(const uint8_t * __restrict__ Wm, int nrows_total, int row0, int nrows,
                                                       const _Float16 * __restrict__ x16, const XO & xo, int nt, int lane,
                                                       int Kr, float (&res)[OWN_MT][RW]) {
    const int NB = Kr / 256, NUNITS = NB * 8, NU = (NUNITS + 31) / 32;
    const uint8_t * A = Wm;
    const uint8_t * B = Wm + (size_t)nrows_total * NB * 192;
    const uint16_t * C = (const uint16_t *)(Wm + (size_t)nrows_total * NB * 208);
    const h2_t off = {(_Float16)1056.0f, (_Float16)1056.0f};   // 1024 + 32: fp16 (1024 + q) - 1056 = q - 32, exact
    float acc[OWN_MT][RW];
#pragma unroll
    for (int j = 0; j < OWN_MT; j++)
#pragma unroll
        for (int r = 0; r < RW; r++) acc[j][r] = 0.0f;
#pragma unroll 1
    for (int i0 = 0; i0 < NU; i0++) {
        u32x2_t q0[RW], q1[RW], qh[RW];
        uint32_t sc[RW];
        u32x4_t xv[4];
        uint16_t dh[RW];
        const int u = lane + 32 * i0, uu = u < NUNITS ? u : 0;
        const int b = uu >> 3, n = (uu >> 2) & 1, c8 = (uu & 3) * 8;
#pragma unroll
        for (int r = 0; r < RW; r++) {                       // every weight load of the pass, unconditionally
            const size_t rb = (size_t)min(row0 + r, nrows - 1) * NB + b;
            const uint8_t * a = A + rb * 192;
            q0[r] = *(const u32x2_t *)(a + n * 64 + c8);
            q1[r] = *(const u32x2_t *)(a + n * 64 + 32 + c8);
            qh[r] = *(const u32x2_t *)(a + 128 + n * 32 + c8);
            sc[r] = *(const uint32_t *)(B + rb * 16 + n * 8 + 4 * ((uu & 3) >> 1));
            dh[r] = C[rb];
        }
        const size_t xu = (size_t)b * 256 + n * 128 + c8;
#pragma unroll
        for (int j = 0; j < OWN_MT; j++) {
            if (j >= nt) break;
            const _Float16 * xp = x16 + xo(j) + xu;
#pragma unroll
            for (int qd = 0; qd < 4; qd++) xv[qd] = *(const u32x4_t *)(xp + qd * 32);
            if (u >= NUNITS) continue;
#pragma unroll
            for (int r = 0; r < RW; r++) {
                __builtin_amdgcn_sched_barrier(0);
                const float d = (float)__builtin_bit_cast(_Float16, dh[r]);
                const uint32_t lw0[2] = {q0[r].x, q0[r].y}, lw1[2] = {q1[r].x, q1[r].y};
                const uint32_t hw[2] = {qh[r].x, qh[r].y}, scw = sc[r];
                float a = acc[j][r];
#pragma unroll
                for (int qd = 0; qd < 4; qd++) {
                    __builtin_amdgcn_sched_barrier(0);
                    const float s = (float)(int8_t)((scw >> (8 * qd)) & 0xffu);
                    const uint32_t xw[4] = {xv[qd].x, xv[qd].y, xv[qd].z, xv[qd].w};
                    float aq = 0.0f;
#pragma unroll
                    for (int w = 0; w < 2; w++) {
                        const uint32_t lo = ((qd & 1) ? lw1[w] : lw0[w]) >> ((qd & 2) ? 4 : 0);
                        const uint32_t q = (lo & 0x0f0f0f0fu) | (((hw[w] >> (2 * qd)) & 0x03030303u) << 4);   // 0 .. 63
                        const uint32_t p01 = __builtin_amdgcn_perm(0u, q, 0x0c010c00u) | 0x64006400u;
                        const uint32_t p23 = __builtin_amdgcn_perm(0u, q, 0x0c030c02u) | 0x64006400u;
                        aq = __builtin_amdgcn_fdot2(as_h2(p01) - off, as_h2(xw[2 * w]), aq, false);
                        aq = __builtin_amdgcn_fdot2(as_h2(p23) - off, as_h2(xw[2 * w + 1]), aq, false);
                    }
                    a = __builtin_fmaf(d * s, aq, a);
                }
                acc[j][r] = a;
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

// ---- the dense qwen35 verify module's T-token tiles (kernels/k3_q35_swift_ver.hip, K3_Q35_VER): the weights unpacked
// once per pass and reused by every token. Per token the statements and their order are the one-token tile's (the same
// products, the same dot2 chains from 0, the same fma chain per row), so every sum keeps that tile's bits; what goes is
// the repeat of the unpacking (nibbles / bytes -> fp16) and of the scale decode for tokens 1 .. T - 1. Measured on the
// trace of a T = 4 pass (XTX, depth 512) before: the Q4_K tiles 2.0x a decode step's time for the same rows, the rows
// tiles below 2.3 .. 3.1x (each token ran the whole tile again).
// Q4_K one round trip: the low and the high nibbles of a pass each unpacked once for both rows, every token's two dots on
// them, then the one-token tile's fma chain per token and row (d sc0 . lo, -dm m0 . sum lo, d sc1 . hi, -dm m1 . sum hi).
template <int RW, class XO>
__device__ __forceinline__ void own_rows_q4k_1rt_mt2(const uint8_t * __restrict__ W, int row0, int nrows,
                                                     const _Float16 * __restrict__ x16, const XO & xo, int nt, int lane,
                                                     int Kr, float (&res)[OWN_MT][RW]) {
    const int NB = Kr / 256, NUNITS = NB * 8, NU = (NUNITS + 31) / 32;
    const size_t RB = (size_t)NB * 144;
    const h2_t off = {(_Float16)1024.0f, (_Float16)1024.0f};
    float acc[OWN_MT][RW];
#pragma unroll
    for (int j = 0; j < OWN_MT; j++)
#pragma unroll
        for (int r = 0; r < RW; r++) acc[j][r] = 0.0f;
#pragma unroll 1
    for (int i0 = 0; i0 < NU; i0++) {
        u32x4_t hd[RW], qs[RW];
        const int u = lane + 32 * i0, uu = u < NUNITS ? u : 0;
        const int b = uu >> 3, jq = (uu & 7) >> 1, h = uu & 1;
#pragma unroll
        for (int r = 0; r < RW; r++) {                       // every weight load of the pass, unconditionally
            const uint8_t * blk = W + (size_t)min(row0 + r, nrows - 1) * RB + (size_t)b * 144;
            qs[r] = *(const u32x4_t *)(blk + 16 + 32 * jq + 16 * h);
            hd[r] = *(const u32x4_t *)blk;
        }
        const size_t xu = (size_t)b * 256 + 64 * jq + 16 * h;
        const int ju = (u & 7) >> 1;
        float ks0[RW], km0[RW], ks1[RW], km1[RW];            // the tile's products d sc0, -dm m0, d sc1, -dm m1
#pragma unroll
        for (int r = 0; r < RW; r++) {
            const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd[r].x & 0xffffu));
            const float dm = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd[r].x >> 16));
            float sc0, m0, sc1, m1;
            q4k_sm2(hd[r], ju, sc0, m0, sc1, m1);
            ks0[r] = d * sc0; km0[r] = -dm * m0; ks1[r] = d * sc1; km1[r] = -dm * m1;
        }
        // the low half's two fmas of every token's chain, then the high half's: per token and row the chain is the tile's
        // (d sc0 . lo, -dm m0 . sum lo, d sc1 . hi, -dm m1 . sum hi), the halves only interleaved across tokens
#pragma unroll
        for (int half = 0; half < 2; half++) {               // 0: the low nibbles with x[0 .. 15], 1: the high with x[32 .. 47]
            h2_t wv[RW][8];
#pragma unroll
            for (int r = 0; r < RW; r++) {
                const uint32_t qw[4] = {qs[r].x, qs[r].y, qs[r].z, qs[r].w};
#pragma unroll
                for (int w = 0; w < 4; w++) {
                    const uint32_t n = (qw[w] >> (4 * half)) & 0x0f0f0f0fu;
                    wv[r][2 * w] = as_h2(__builtin_amdgcn_perm(0u, n, 0x0c010c00u) | 0x64006400u) - off;
                    wv[r][2 * w + 1] = as_h2(__builtin_amdgcn_perm(0u, n, 0x0c030c02u) | 0x64006400u) - off;
                }
            }
#pragma unroll
            for (int j = 0; j < OWN_MT; j++) {
                if (j >= nt) break;
                const _Float16 * xp = x16 + xo(j) + xu + 32 * half;
                const u32x4_t xa = *(const u32x4_t *)xp, xb = *(const u32x4_t *)(xp + 8);
                __builtin_amdgcn_sched_barrier(0);
                if (u >= NUNITS) continue;
                const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
                const float sx = sum16h(xa, xb);
#pragma unroll
                for (int r = 0; r < RW; r++) {
                    float s = 0.0f;
#pragma unroll
                    for (int k = 0; k < 8; k++) s = __builtin_amdgcn_fdot2(wv[r][k], as_h2(xw[k]), s, false);
                    const float a = __builtin_fmaf(half ? ks1[r] : ks0[r], s, acc[j][r]);
                    acc[j][r] = __builtin_fmaf(half ? km1[r] : km0[r], sx, a);
                }
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

// Q6_K planes: each quarter's 6-bit values unpacked once for both rows, every token's quarter dot on them; per token and
// row the quarters' fmas run in the tile's order (0 .. 3)
template <int RW, class XO>
__device__ __forceinline__ void own_rows_q6k_planes_mt2(const uint8_t * __restrict__ Wm, int nrows_total, int row0, int nrows,
                                                        const _Float16 * __restrict__ x16, const XO & xo, int nt, int lane,
                                                        int Kr, float (&res)[OWN_MT][RW]) {
    const int NB = Kr / 256, NUNITS = NB * 8, NU = (NUNITS + 31) / 32;
    const uint8_t * A = Wm;
    const uint8_t * B = Wm + (size_t)nrows_total * NB * 192;
    const uint16_t * C = (const uint16_t *)(Wm + (size_t)nrows_total * NB * 208);
    const h2_t off = {(_Float16)1056.0f, (_Float16)1056.0f};   // 1024 + 32: fp16 (1024 + q) - 1056 = q - 32, exact
    float acc[OWN_MT][RW];
#pragma unroll
    for (int j = 0; j < OWN_MT; j++)
#pragma unroll
        for (int r = 0; r < RW; r++) acc[j][r] = 0.0f;
#pragma unroll 1
    for (int i0 = 0; i0 < NU; i0++) {
        u32x2_t q0[RW], q1[RW], qh[RW];
        uint32_t sc[RW];
        uint16_t dh[RW];
        const int u = lane + 32 * i0, uu = u < NUNITS ? u : 0;
        const int b = uu >> 3, n = (uu >> 2) & 1, c8 = (uu & 3) * 8;
#pragma unroll
        for (int r = 0; r < RW; r++) {                       // every weight load of the pass, unconditionally
            const size_t rb = (size_t)min(row0 + r, nrows - 1) * NB + b;
            const uint8_t * a = A + rb * 192;
            q0[r] = *(const u32x2_t *)(a + n * 64 + c8);
            q1[r] = *(const u32x2_t *)(a + n * 64 + 32 + c8);
            qh[r] = *(const u32x2_t *)(a + 128 + n * 32 + c8);
            sc[r] = *(const uint32_t *)(B + rb * 16 + n * 8 + 4 * ((uu & 3) >> 1));
            dh[r] = C[rb];
        }
        const size_t xu = (size_t)b * 256 + n * 128 + c8;
        float dd[RW];
#pragma unroll
        for (int r = 0; r < RW; r++) dd[r] = (float)__builtin_bit_cast(_Float16, dh[r]);
#pragma unroll
        for (int qd = 0; qd < 4; qd++) {
            h2_t wv[RW][4];
            float ds[RW];
#pragma unroll
            for (int r = 0; r < RW; r++) {
                const uint32_t lw0[2] = {q0[r].x, q0[r].y}, lw1[2] = {q1[r].x, q1[r].y}, hw[2] = {qh[r].x, qh[r].y};
                ds[r] = dd[r] * (float)(int8_t)((sc[r] >> (8 * qd)) & 0xffu);
#pragma unroll
                for (int w = 0; w < 2; w++) {
                    const uint32_t lo = ((qd & 1) ? lw1[w] : lw0[w]) >> ((qd & 2) ? 4 : 0);
                    const uint32_t q = (lo & 0x0f0f0f0fu) | (((hw[w] >> (2 * qd)) & 0x03030303u) << 4);   // 0 .. 63
                    wv[r][2 * w] = as_h2(__builtin_amdgcn_perm(0u, q, 0x0c010c00u) | 0x64006400u) - off;
                    wv[r][2 * w + 1] = as_h2(__builtin_amdgcn_perm(0u, q, 0x0c030c02u) | 0x64006400u) - off;
                }
            }
#pragma unroll
            for (int j = 0; j < OWN_MT; j++) {
                if (j >= nt) break;
                const u32x4_t xv = *(const u32x4_t *)(x16 + xo(j) + xu + qd * 32);
                __builtin_amdgcn_sched_barrier(0);
                if (u >= NUNITS) continue;
                const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
                for (int r = 0; r < RW; r++) {
                    float aq = 0.0f;
#pragma unroll
                    for (int w = 0; w < 2; w++) {
                        aq = __builtin_amdgcn_fdot2(wv[r][2 * w], as_h2(xw[2 * w]), aq, false);
                        aq = __builtin_amdgcn_fdot2(wv[r][2 * w + 1], as_h2(xw[2 * w + 1]), aq, false);
                    }
                    acc[j][r] = __builtin_fmaf(ds[r], aq, acc[j][r]);
                }
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

// the rows tiles (own_quant.h own_rows_q8_0 / own_rows_q5_k on fp16 activations) for nt tokens: a block's activations
// loaded for every token, each row's weight word unpacked once, then each token's statements in the tile's order. Token
// j reads x16 + j xs and writes out + j ys (fp32), rows row0 .. row0 + OWN_R - 1 that are < nrows
__device__ __forceinline__ void own_rows_q8_0_mt(const uint8_t * __restrict__ W, int row0, int nrows, int K,
                                                 const _Float16 * __restrict__ x16, size_t xs, int nt,
                                                 float * __restrict__ out, size_t ys, int lane) {
    const int nb = K / 32, rb = nb * 34, sub = lane >> 3, c4 = (lane & 7) * 4;
    const h2_t off = {(_Float16)1152.0f, (_Float16)1152.0f};
    float acc[OWN_MT][OWN_R];
#pragma unroll
    for (int j = 0; j < OWN_MT; j++)
#pragma unroll
        for (int r = 0; r < OWN_R; r++) acc[j][r] = 0.0f;
    for (int b = sub; b < nb; b += 4) {
        Act16 a[OWN_MT];
#pragma unroll
        for (int j = 0; j < OWN_MT; j++) { if (j >= nt) break; load_act(a[j], x16 + j * xs, nullptr, b * 32 + c4); }
#pragma unroll
        for (int r = 0; r < OWN_R; r++) {
            if (row0 + r >= nrows) continue;
            const uint8_t * blk = W + (size_t)(row0 + r) * rb + b * 34;
            const float d = f16f(*(const uint16_t *)blk);
            const uint32_t uq = ld32h(blk + 2 + c4) ^ 0x80808080u;       // (dot4s(Act16)'s unpacking, once)
            const h2_t w01 = as_h2(__builtin_amdgcn_perm(0u, uq, 0x0c010c00u) | 0x64006400u) - off;
            const h2_t w23 = as_h2(__builtin_amdgcn_perm(0u, uq, 0x0c030c02u) | 0x64006400u) - off;
#pragma unroll
            for (int j = 0; j < OWN_MT; j++) {
                if (j >= nt) break;
                const float s = __builtin_amdgcn_fdot2(w23, as_h2(a[j].hi), __builtin_amdgcn_fdot2(w01, as_h2(a[j].lo), 0.0f, false), false);
                acc[j][r] = __builtin_fmaf(d, s, acc[j][r]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < OWN_MT; j++) {
        if (j >= nt) break;
#pragma unroll
        for (int r = 0; r < OWN_R; r++) {
            const float s = wave_sum(acc[j][r]);
            if (lane == 0 && row0 + r < nrows) out[j * ys + row0 + r] = s;
        }
    }
}

__device__ __forceinline__ void own_rows_q5_k_mt(const uint8_t * __restrict__ W, int row0, int nrows, int K,
                                                 const _Float16 * __restrict__ x16, size_t xs, int nt,
                                                 float * __restrict__ out, size_t ys, int lane) {
    const int nb = K / 256, rb = nb * 176, j4 = lane >> 3, c4 = (lane & 7) * 4;
    const h2_t off = {(_Float16)1024.0f, (_Float16)1024.0f};
    float acc[OWN_MT][OWN_R];
#pragma unroll
    for (int j = 0; j < OWN_MT; j++)
#pragma unroll
        for (int r = 0; r < OWN_R; r++) acc[j][r] = 0.0f;
    for (int b = 0; b < nb; b++) {
        const int vlo = b * 256 + j4 * 64 + c4, vhi = vlo + 32;
        Act16 alo[OWN_MT], ahi[OWN_MT];
        float mlo[OWN_MT], mhi[OWN_MT];
#pragma unroll
        for (int j = 0; j < OWN_MT; j++) {
            if (j >= nt) break;
            load_act(alo[j], x16 + j * xs, nullptr, vlo); load_act(ahi[j], x16 + j * xs, nullptr, vhi);
            mlo[j] = sum4(alo[j]); mhi[j] = sum4(ahi[j]);
        }
#pragma unroll
        for (int r = 0; r < OWN_R; r++) {
            if (row0 + r >= nrows) continue;
            const uint8_t * blk = W + (size_t)(row0 + r) * rb + b * 176;
            const float d = f16f(*(const uint16_t *)blk), dmin = f16f(*(const uint16_t *)(blk + 2));
            int sc0, m0, sc1, m1;
            q4k_scale_min(blk + 4, 2 * j4, sc0, m0); q4k_scale_min(blk + 4, 2 * j4 + 1, sc1, m1);
            const uint32_t qh = *(const uint32_t *)(blk + 16 + c4);
            const uint32_t qs = *(const uint32_t *)(blk + 48 + j4 * 32 + c4);
            const uint32_t lo = (qs & 0x0f0f0f0fu) | (((qh >> (2 * j4)) & 0x01010101u) << 4);
            const uint32_t hi = ((qs >> 4) & 0x0f0f0f0fu) | (((qh >> (2 * j4 + 1)) & 0x01010101u) << 4);
            // (dot4u(Act16)'s unpacking, once per row)
            const h2_t l01 = as_h2(__builtin_amdgcn_perm(0u, lo, 0x0c010c00u) | 0x64006400u) - off;
            const h2_t l23 = as_h2(__builtin_amdgcn_perm(0u, lo, 0x0c030c02u) | 0x64006400u) - off;
            const h2_t h01 = as_h2(__builtin_amdgcn_perm(0u, hi, 0x0c010c00u) | 0x64006400u) - off;
            const h2_t h23 = as_h2(__builtin_amdgcn_perm(0u, hi, 0x0c030c02u) | 0x64006400u) - off;
            const float k0 = d * (float)sc0, n0 = -dmin * (float)m0, k1 = d * (float)sc1, n1 = -dmin * (float)m1;
#pragma unroll
            for (int j = 0; j < OWN_MT; j++) {
                if (j >= nt) break;
                const float dlo = __builtin_amdgcn_fdot2(l23, as_h2(alo[j].hi), __builtin_amdgcn_fdot2(l01, as_h2(alo[j].lo), 0.0f, false), false);
                const float dhi = __builtin_amdgcn_fdot2(h23, as_h2(ahi[j].hi), __builtin_amdgcn_fdot2(h01, as_h2(ahi[j].lo), 0.0f, false), false);
                float s = acc[j][r];
                s = __builtin_fmaf(k0, dlo, s);
                s = __builtin_fmaf(n0, mlo[j], s);
                s = __builtin_fmaf(k1, dhi, s);
                acc[j][r] = __builtin_fmaf(n1, mhi[j], s);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < OWN_MT; j++) {
        if (j >= nt) break;
#pragma unroll
        for (int r = 0; r < OWN_R; r++) {
            const float s = wave_sum(acc[j][r]);
            if (lane == 0 && row0 + r < nrows) out[j * ys + row0 + r] = s;
        }
    }
}
