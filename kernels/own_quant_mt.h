// T-token forms of own_quant.h's one-round-trip tiles (the qwen3moe verify module, kernels/k3_q3m_ver.hip): a pass's
// weight words are loaded once and held in registers while up to OWN_MT tokens' sums run in turn, each on its own
// activations (token j's at x16 + xo(j) fp16 values). Per token the statements are the one-token tile's
// (own_rows_q4k_1rt / own_rows_q6k_planes with K = 0, UP = 1, UNR = 1) in its order, so every sum has that tile's bits.
//   nt (wave-uniform): tokens of this pass, 1 .. OWN_MT;  res[j][r]: token j's sum of row row0 + r, in every lane.
// What a token more costs is its arithmetic, not memory (XT trace, T = 4, real tokens, per layer: attn_output 10.8 us
// for one token, 32.9 for four, 18.0 with OWN_MT_NOARITH; k | v 12.7 / 23.9 / 12.7; the head 350 / 516 / 381): the
// weights' bytes are read once whatever the token count, and issuing token j + 1's activation loads ahead of token j's
// sums changed nothing (11.46 vs 11.54 ms per pass). Each token decodes the quant words again (nibbles -> fp16).
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
#ifdef OWN_MT_NOARITH   // measurement only: tokens past the first keep their loads and skip their sums
            if (j) { acc[j][0] += (float)((xl[0].x ^ xl[1].x ^ xh[0].x ^ xh[1].x) & 0xffu); continue; }
#endif
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
#ifdef OWN_MT_NOARITH
            if (j) { acc[j][0] += (float)((xv[0].x ^ xv[1].x ^ xv[2].x ^ xv[3].x) & 0xffu); continue; }
#endif
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
