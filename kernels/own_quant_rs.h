// Row-scaled integer codes (a study format: docs/dev/row-scale-study.md): fixed-width codes, ONE scale per output row,
// applied once to the row's sum. Nothing is decoded per block: no header, no sub-block scale, no min term.
// What a kernel reads (caminus.engine.own_layout.rs_planes; made at load from a Q4_0 / Q2_0 container whose blocks
// repeat their row's scale), for a matrix of R rows of K values (K a multiple of 256):
//   codes   per row K * BITS / 8 bytes, rows contiguous (every load of a tile is inside its row: no pad)
//   scales  one fp32 per row, in a buffer of their own
// The scales are NOT behind the codes: read through the weight pointer at a row-dependent offset, the load cost
// k3_run 6 spilled VGPRs in every form tried (planes, a 16-byte row header, loaded before or after the passes, from
// wave-uniform scalars); read through any other pointer, 0 (llvm-readelf, the study modules). The price moved into
// scalars: k3_q3m_rs4 / _rs2 spill 66 against the lean module's 53 (every 0-VGPR-spill form sat at 62 to 66), which
// every segment hop of every wave pays, so the modules' own cost is measured (the official file on them against
// the lean module) and a tile's gain is read against the official file on the SAME module.
// 4 bits: a unit of 32 values is 16 bytes, byte b = value b | value (b + 16) << 4 (Q4_0's block bytes as they are);
//         value = d (q - 8).
// 2 bits: a unit of 32 values is 8 bytes, two dwords of 16 values: value 4 i + k of a dword sits at bits 8 k + 2 i, so
//         (dword >> 2 i) & 0x03030303 is values 4 i .. 4 i + 3 as four bytes (the Q2_0 container packs value j at bits
//         2 (j % 4) of byte j / 4: four shifts per byte to the same place); value = d (q - 1): -1, 0, +1, +2.
// A lane owns one unit per pass (K 2048: 2 passes, 768: 1), RW rows per wave on the same 32 activations, as the Q4_K
// one-round-trip tile (own_quant.h); a row's loads are ONE 16-byte (8-byte) word per pass against Q4_K's two.
#pragma once
#include <type_traits>
#include "own_quant.h"

struct Rs4 { u32x4_t q; };
struct Rs2 { u32x2_t q; };
__device__ __forceinline__ void rs_load(Rs4 & c, const uint8_t * p) { c.q = *(const u32x4_t *)p; }
__device__ __forceinline__ void rs_load(Rs2 & c, const uint8_t * p) { c.q = *(const u32x2_t *)p; }
__device__ __forceinline__ uint32_t rs_fold(const Rs4 & c) { return c.q.x ^ c.q.y ^ c.q.z ^ c.q.w; }
__device__ __forceinline__ uint32_t rs_fold(const Rs2 & c) { return c.q.x ^ c.q.y; }
// four code bytes n (each < 16) times four fp16 activations (two half2 words), added to s in fp32; a code enters as the
// exact fp16 (1024 + q) - o: o = 1024 + the level set's zero code, so the signed level is exact and nothing is left
// for a correction term
__device__ __forceinline__ float rs_dot4(uint32_t n, uint32_t x01, uint32_t x23, float s, float o) {
    const h2_t off = {(_Float16)o, (_Float16)o};
    const uint32_t p01 = __builtin_amdgcn_perm(0u, n, 0x0c010c00u) | 0x64006400u;
    const uint32_t p23 = __builtin_amdgcn_perm(0u, n, 0x0c030c02u) | 0x64006400u;
    s = __builtin_amdgcn_fdot2(as_h2(p01) - off, as_h2(x01), s, false);
    return __builtin_amdgcn_fdot2(as_h2(p23) - off, as_h2(x23), s, false);
}
// a unit's 32 values times its activations (xa: values 0 .. 15, xb: 16 .. 31, as 8 half2 words each), added to s
__device__ __forceinline__ float rs_dot32(const Rs4 & c, const uint32_t * xa, const uint32_t * xb, float s, bool fence) {
    const uint32_t qw[4] = {c.q.x, c.q.y, c.q.z, c.q.w};
#pragma unroll
    for (int h = 0; h < 2; h++) {
#pragma unroll
        for (int w = 0; w < 4; w++) {
            if (fence) __builtin_amdgcn_sched_barrier(0);
            s = rs_dot4((qw[w] >> (4 * h)) & 0x0f0f0f0fu, (h ? xb : xa)[2 * w], (h ? xb : xa)[2 * w + 1], s, 1032.0f);
        }
    }
    return s;
}
__device__ __forceinline__ float rs_dot32(const Rs2 & c, const uint32_t * xa, const uint32_t * xb, float s, bool fence) {
    const uint32_t cw[2] = {c.q.x, c.q.y};
#pragma unroll
    for (int h = 0; h < 2; h++) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (fence) __builtin_amdgcn_sched_barrier(0);
            s = rs_dot4((cw[h] >> (2 * i)) & 0x03030303u, (h ? xb : xa)[2 * i], (h ? xb : xa)[2 * i + 1], s, 1025.0f);
        }
    }
    return s;
}

// rows row0 .. row0 + RW - 1 (< nrows) of the matrix W, fp16 activations x16 [K]; sc: the matrix's row scales.
// K = 0: the row length is the runtime Kr (K3: one body for every K). UP / UNR / res: as own_rows_q4k_1rt.
template <int BITS, int K, int RW, int UP = 2, int UNR = 64>
__device__ __forceinline__ void own_rows_rs_1rt(const uint8_t * __restrict__ W, const float * __restrict__ sc, int row0, int nrows,
                                                const _Float16 * __restrict__ x16, float * __restrict__ out, int lane,
                                                int Kr = K, float * res = nullptr) {
    typedef typename std::conditional<BITS == 4, Rs4, Rs2>::type C;
    const int KK = K ? K : Kr, NUNITS = KK / 32, NU = (NUNITS + 31) / 32, UB = 4 * BITS;   // (UB: bytes per unit)
    const size_t RB = (size_t)KK * BITS / 8;
    float acc[RW];
#pragma unroll
    for (int r = 0; r < RW; r++) acc[r] = 0.0f;
#pragma unroll UNR
    for (int i0 = 0; i0 < NU; i0 += UP) {
        C c[RW][UP];
        u32x4_t xl[UP][2], xh[UP][2];
#pragma unroll
        for (int i = 0; i < UP; i++) {                       // every load of the pass, unconditionally
            const int u = lane + 32 * (i0 + i), uu = (i0 + i < NU && u < NUNITS) ? u : 0;
#pragma unroll
            for (int r = 0; r < RW; r++) rs_load(c[r][i], W + (size_t)min(row0 + r, nrows - 1) * RB + (size_t)uu * UB);
            const _Float16 * xp = x16 + 32 * uu;
#if defined(OWN_READONLY)   // (measurement: the codes only)
            xl[i][0] = xl[i][1] = xh[i][0] = xh[i][1] = (u32x4_t){0u, 0u, 0u, 0u}; (void)xp;
#else
            xl[i][0] = *(const u32x4_t *)xp; xl[i][1] = *(const u32x4_t *)(xp + 8);
            xh[i][0] = *(const u32x4_t *)(xp + 16); xh[i][1] = *(const u32x4_t *)(xp + 24);
#endif
        }
#if defined(OWN_READONLY)   // measurement only (what a token costs with the codes read and no arithmetic): folded by xor
#pragma unroll
        for (int i = 0; i < UP; i++) {
#pragma unroll
            for (int r = 0; r < RW; r++) acc[r] += (float)((rs_fold(c[r][i]) ^ xl[i][0].x) & 0xffu);
        }
        continue;
#endif
#pragma unroll
        for (int i = 0; i < UP; i++) {
            const int u = lane + 32 * (i0 + i);
            if (i0 + i >= NU || u >= NUNITS) continue;
            const uint32_t xa[8] = {xl[i][0].x, xl[i][0].y, xl[i][0].z, xl[i][0].w, xl[i][1].x, xl[i][1].y, xl[i][1].z, xl[i][1].w};
            const uint32_t xb[8] = {xh[i][0].x, xh[i][0].y, xh[i][0].z, xh[i][0].w, xh[i][1].x, xh[i][1].y, xh[i][1].z, xh[i][1].w};
#pragma unroll
            for (int r = 0; r < RW; r++) {
                if (UNR == 1) __builtin_amdgcn_sched_barrier(0);
                acc[r] = rs_dot32(c[r][i], xa, xb, acc[r], UNR == 1);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RW; r++) {
        const float s = sc[min(row0 + r, nrows - 1)] * wave_sum(acc[r]);
        if (res) res[r] = s;
        else if (lane == 0 && row0 + r < nrows) out[row0 + r] = s;
    }
}
