// The dense qwen35 line's batched prefill for the plain Qwen3.8-27B GSQ-RCO files (python/caminus/prefill/q35own.py):
// kernels/q35_prefill_tcap.hip as it is (Swift's object + the IQ GEMMs + ThinkingCap's IQ2_S GEMM and embedding; its
// q3p_mm_iq2xxs serves this file's IQ2_XXS matrices) and this file's GEMMs for the three formats ThinkingCap's files do
// not use: IQ2_XS, Q2_K, IQ1_M. Each reads the decode engine's copy in Caminus's record layout
// (caminus.engine.own_layout: iq_records / q2k_records / iq1m_records; the decode tiles kernels/own_quant_iq2xs.h,
// own_quant_q2k.h, own_quant_iq1m.h read the same buffers). Written from the block layouts (the host decoders:
// python/caminus/own_dequant.py); no ggml-cuda code. ThinkingCap's prefill object does not include this file: its code
// object is unchanged.
//
// The contract of the IQ study's GEMMs (kernels/q3m_prefill_iq.h iqp_mm): Y[ydst[i]][r] = X[xsrc[i]] . W_e[r] in fp32,
// fp16 activations through WMMA, the same work items and 64-token x 128-row tiles; items nullptr: dense rows; Rtot = the
// rows of the whole buffer (plane B starts after plane A of every row). Each B operand is an integer of at most 1,333 in
// magnitude (exact in fp16, iqp_b), so a block's products are exact up to the fp32 accumulation.
#include "q35_prefill_tcap.hip"
#include "iq2xs_grid.h"
#include "iq1_grid.h"

// 4 codes (2 bits each, code k at bits 2k) -> 4 bytes code + 128
__device__ __forceinline__ uint32_t q35g_code4(uint32_t c) {
    return ((c | (c << 6) | (c << 12) | (c << 18)) & 0x03030303u) | 0x80808080u;
}

#define Q35G_HEAD(RS_) \
    MMItem it; \
    if (items) { if ((int)blockIdx.x >= *nitems) return; it = items[blockIdx.x]; } \
    else { it.e = 0; it.off = blockIdx.x * MM_M; it.cnt = min(MM_M, ntok - it.off); } \
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5; \
    const int r = blockIdx.y * MM_N + wave * 16 + (lane & 15); \
    const bool rok = r < R; \
    const int NB = K / 256; \
    constexpr int RS = RS_; \
    const size_t g = (size_t)it.e * R + (rok ? r : 0); \
    const uint8_t * pa = W + g * NB * 8 * RS; \
    const uint8_t * pb = W + (size_t)Rtot * NB * 8 * RS + g * NB * HS; \
    f8_t acc[4]; \
    _Pragma("unroll") for (int mt = 0; mt < 4; mt++) acc[mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};

#define Q35G_STORE \
    if (!rok) return; \
    _Pragma("unroll") for (int mt = 0; mt < 4; mt++) \
    _Pragma("unroll") for (int v = 0; v < 8; v++) { \
        const int t = mt * 16 + 2 * v + (lane >> 4); \
        if (t < it.cnt) Y[(size_t)(ydst ? ydst[it.off + t] : it.off + t) * ldy + r] = acc[mt][v]; \
    }

#define Q35G_ARGS \
        const uint8_t * __restrict__ W, int R, int Rtot, int K, const _Float16 * __restrict__ X, int ldx, const int * __restrict__ xsrc, \
        float * __restrict__ Y, int ldy, const int * __restrict__ ydst, const MMItem * __restrict__ items, const int * __restrict__ nitems, \
        int ntok

// ---- q35g_mm_iq2xs: IQ2_XS. Record (9 bytes per unit of 32): 4 u16 (a 9-bit grid index | a 7-bit sign code << 9, per 8
// values) | the scale byte (a nibble per 16 values); header: d per block. w = d 0.125 (1 + 2 s) (+-grid), the grid's
// magnitudes in {8, 25, 43}: B = (1 + 2 s) (+-grid), |B| <= 31 * 43. The grid packed to 2 bits a magnitude (code =
// magnitude >> 4), two entries a word in the IQ study's LDS array IQP_G (256 words), staged before the first barrier.
extern "C" __global__ void __launch_bounds__(256) q35g_mm_iq2xs(Q35G_ARGS) {
    constexpr int HS = 2;
    Q35G_HEAD(9)
    {
        uint32_t w = 0;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint64_t e = IQ2XS_GRID[2 * tid + h];
#pragma unroll
            for (int k = 0; k < 8; k++) w |= (uint32_t)(((e >> (8 * k)) & 0xffu) >> 4) << (16 * h + 2 * k);
        }
        IQP_G[tid] = w;
    }
    for (int b = 0; b < NB; b++) {
        u32x4_t rc[8];
        const uint32_t hd = *(const uint16_t *)(pb + (size_t)b * HS);
#pragma unroll
        for (int i = 0; i < 8; i++) rc[i] = *(const u32x4_t *)(pa + ((size_t)b * 8 + i) * RS);
        __syncthreads();
        mm_stage<false>(X, ldx, xsrc, it.off, it.cnt, b, tid);
        __syncthreads();
        const float d = rok ? (float)__builtin_bit_cast(_Float16, (uint16_t)hd) : 0.0f;
        f8_t P[4];
#pragma unroll
        for (int mt = 0; mt < 4; mt++) P[mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int ib = 0; ib < 8; ib++) {
            const uint32_t q[4] = {rc[ib].x & 0xffffu, rc[ib].x >> 16, rc[ib].y & 0xffffu, rc[ib].y >> 16};
            const uint32_t scb = rc[ib].z & 0xffu;
            uint32_t u[8];                                   // the unit's 32 weights + 128, values 4 w .. 4 w + 3 in u[w]
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const uint32_t idx = q[k] & 511u, sg = iqp_signs8(q[k] >> 9);
                const uint32_t e = IQP_G[idx >> 1] >> (16 * (idx & 1u));
                u[2 * k] = iqp_signed(q35t_mag4(e & 0xffu), sg);
                u[2 * k + 1] = iqp_signed(q35t_mag4((e >> 8) & 0xffu), sg >> 4);
            }
            const _Float16 sc0 = rok ? (_Float16)(float)(1u + 2u * (scb & 15u)) : (_Float16)0.0f;
            const _Float16 sc1 = rok ? (_Float16)(float)(1u + 2u * (scb >> 4)) : (_Float16)0.0f;
            const h16_t B0 = iqp_b(u, sc0), B1 = iqp_b(u + 4, sc1);
#pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                P[mt] = wmma(mm_a(mt, 32 * ib, lane), B0, P[mt]);
                P[mt] = wmma(mm_a(mt, 32 * ib + 16, lane), B1, P[mt]);
            }
            __builtin_amdgcn_sched_barrier(0);   // (one unit at a time, as the IQ study's GEMMs)
        }
        const float dc = d * 0.125f;
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int v = 0; v < 8; v++) acc[mt][v] = __builtin_fmaf(dc, P[mt][v], acc[mt][v]);
    }
    Q35G_STORE
}

// ---- q35g_mm_q2k: Q2_K. Record (10 bytes per unit of 32): its 2-bit codes four a byte (value 4 k + j at bits 2 j of byte
// k) | its two scale bytes (a scale nibble | a min nibble << 4, per 16 values); header (4 bytes per block): d, dmin.
// w = d sc q - dmin m: P = sum(x (sc q)) and Pm = sum(x m) through WMMA (B = sc q <= 45; m <= 15), acc += d P - dmin Pm.
extern "C" __global__ void __launch_bounds__(256) q35g_mm_q2k(Q35G_ARGS) {
    constexpr int HS = 4;
    Q35G_HEAD(10)
    for (int b = 0; b < NB; b++) {
        u32x4_t rc[8];
        const uint32_t hd = *(const uint32_t *)(pb + (size_t)b * HS);
#pragma unroll
        for (int i = 0; i < 8; i++) rc[i] = *(const u32x4_t *)(pa + ((size_t)b * 8 + i) * RS);
        __syncthreads();
        mm_stage<false>(X, ldx, xsrc, it.off, it.cnt, b, tid);
        __syncthreads();
        const float d = rok ? (float)__builtin_bit_cast(_Float16, (uint16_t)(hd & 0xffffu)) : 0.0f;
        const float dm = rok ? (float)__builtin_bit_cast(_Float16, (uint16_t)(hd >> 16)) : 0.0f;
        f8_t P[4], Pm[4];
#pragma unroll
        for (int mt = 0; mt < 4; mt++) { P[mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0}; Pm[mt] = P[mt]; }
#pragma unroll
        for (int ib = 0; ib < 8; ib++) {
            const uint32_t cw[2] = {rc[ib].x, rc[ib].y}, sb = rc[ib].z & 0xffffu;
            uint32_t u[8];                                   // codes + 128, values 4 w .. 4 w + 3 in u[w]
#pragma unroll
            for (int w = 0; w < 8; w++) u[w] = q35g_code4((cw[w >> 2] >> (8 * (w & 3))) & 0xffu);
            const uint32_t s0 = sb & 0xffu, s1 = sb >> 8;
            const _Float16 sc0 = (_Float16)(float)(s0 & 15u), sc1 = (_Float16)(float)(s1 & 15u);
            const uint32_t w0 = 0x80808080u + (s0 >> 4) * 0x01010101u, w1 = 0x80808080u + (s1 >> 4) * 0x01010101u;
            const uint32_t um0[4] = {w0, w0, w0, w0}, um1[4] = {w1, w1, w1, w1};   // (the group's min, 16 times)
            const h16_t B0 = iqp_b(u, sc0), B1 = iqp_b(u + 4, sc1);
            const h16_t M0 = iqp_b(um0, (_Float16)1.0f), M1 = iqp_b(um1, (_Float16)1.0f);
#pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                const h16_t a0 = mm_a(mt, 32 * ib, lane), a1 = mm_a(mt, 32 * ib + 16, lane);
                P[mt] = wmma(a0, B0, P[mt]);
                P[mt] = wmma(a1, B1, P[mt]);
                Pm[mt] = wmma(a0, M0, Pm[mt]);
                Pm[mt] = wmma(a1, M1, Pm[mt]);
            }
            __builtin_amdgcn_sched_barrier(0);
        }
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int v = 0; v < 8; v++) acc[mt][v] = __builtin_fmaf(-dm, Pm[mt][v], __builtin_fmaf(d, P[mt][v], acc[mt][v]));
    }
    Q35G_STORE
}

// ---- q35g_mm_iq1m: IQ1_M. Record (8 bytes per unit of 32): 4 index bytes (an 11-bit grid index's low 8 bits per 8 values)
// | 2 qh bytes (a nibble per 8 values: the index's 3 high bits | the delta's sign << 3) | the two scales s0 | s1 << 3;
// header: the block's d (reassembled on the host). w = d (2 s + 1) (g +- 1/8), g in -1 / 0 / +1: B = (2 s + 1) (8 g +- 1)
// (|B| <= 15 * 9), acc += (d / 8) P. The grid (2048 entries of g + 1, 16 KB) read from global memory, as the decode tile.
extern "C" __global__ void __launch_bounds__(256) q35g_mm_iq1m(Q35G_ARGS) {
    constexpr int HS = 2;
    Q35G_HEAD(8)
    for (int b = 0; b < NB; b++) {
        u32x2p_t rc[8];
        const uint32_t hd = *(const uint16_t *)(pb + (size_t)b * HS);
#pragma unroll
        for (int i = 0; i < 8; i++) rc[i] = *(const u32x2p_t *)(pa + ((size_t)b * 8 + i) * RS);
        __syncthreads();
        mm_stage<false>(X, ldx, xsrc, it.off, it.cnt, b, tid);
        __syncthreads();
        const float d = rok ? (float)__builtin_bit_cast(_Float16, (uint16_t)hd) : 0.0f;
        f8_t P[4];
#pragma unroll
        for (int mt = 0; mt < 4; mt++) P[mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int ib = 0; ib < 8; ib++) {
            const uint32_t qs = rc[ib].x, qh = rc[ib].y & 0xffffu, sc = (rc[ib].y >> 16) & 0xffu;
            uint32_t u[8];                                   // 8 g + delta + 128 = 8 (g + 1) + 120 + delta, as bytes
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const uint32_t nib = (qh >> (4 * k)) & 0xfu;
                const uint64_t gg = IQ1_GRID[((qs >> (8 * k)) & 0xffu) | ((nib & 7u) << 8)];
                const uint32_t off = (nib & 8u) ? 0x77777777u : 0x79797979u;
                u[2 * k] = ((uint32_t)gg << 3) + off;
                u[2 * k + 1] = ((uint32_t)(gg >> 32) << 3) + off;
            }
            const _Float16 sc0 = rok ? (_Float16)(float)(2u * (sc & 7u) + 1u) : (_Float16)0.0f;
            const _Float16 sc1 = rok ? (_Float16)(float)(2u * ((sc >> 3) & 7u) + 1u) : (_Float16)0.0f;
            const h16_t B0 = iqp_b(u, sc0), B1 = iqp_b(u + 4, sc1);
#pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                P[mt] = wmma(mm_a(mt, 32 * ib, lane), B0, P[mt]);
                P[mt] = wmma(mm_a(mt, 32 * ib + 16, lane), B1, P[mt]);
            }
            __builtin_amdgcn_sched_barrier(0);
        }
        const float dc = d * 0.125f;
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int v = 0; v < 8; v++) acc[mt][v] = __builtin_fmaf(dc, P[mt][v], acc[mt][v]);
    }
    Q35G_STORE
}
