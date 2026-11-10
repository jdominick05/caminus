// Caminus's own quantized matvec: the format decoders and dot helpers shared by the standalone kernels
// (kernels/own_mv.hip) and the K3 op OP_OWN_MV (kernels/k3_gguf_ops.h, the K3_OWN modules). Written from the GGUF block
// layouts (docs/specs/spec-own-quant.md); no ggml-cuda code. Activation precisions:
//   Act32   x fp32: each weight decoded to an exact fp32 value, one fma per weight (error vs fp64 ~1e-7, CPU-measured)
//   Act16   x fp16 (a buffer of halves), Act16F the same from fp32 x converted in registers (round to nearest even):
//           weights decoded to exact fp16 through the 1024-magic, v_dot2_f32_f16 per two weights (~1e-4)
//   Act8    x int8 per 32 values with an fp32 scale (own_q8) and the exact fp32 block sum for min terms; v_dot4_i32_iu8
//           (integer sums exact; ~5e-3, the int8 bound)
// Work layout: one wave32 computes OWN_R = 4 rows, so each activation value a lane loads serves 4 rows; a lane owns 4
// consecutive columns of a block's sub-layout (Q6_K: 2 blocks per wave step, Q4_K: 1, Q8_0: 4). Each lane sums its groups in
// K order in fp32, then the wave adds lanes with a fixed butterfly (xor 16 .. 1): the order depends on neither the grid nor
// how many rows a task holds. Weights are read in place (GGUF block order; Q6_K blocks are 210 B, Q8_0 34 B: 16-bit loads).
#pragma once
#include <stdint.h>

#define OWN_R 4      // rows per wave
#define OWN_WAVES 8  // waves per workgroup (256 threads)

enum { A32 = 0, A16 = 1, A8 = 2 };

__device__ __forceinline__ float f16f(uint16_t h) { return (float)__builtin_bit_cast(_Float16, h); }
__device__ __forceinline__ uint32_t ld32h(const uint8_t * p) {   // 4 bytes at a 2-byte-aligned address
    const uint16_t * q = (const uint16_t *)p;
    return (uint32_t)q[0] | ((uint32_t)q[1] << 16);
}
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}
typedef _Float16 h2_t __attribute__((ext_vector_type(2)));
__device__ __forceinline__ h2_t as_h2(uint32_t u) { return __builtin_bit_cast(h2_t, u); }

// ---- the activation a lane holds for one group of 4 consecutive values (x index i, i % 4 == 0) --------------------------
struct Act32 { float v[4]; };
struct Act16 { uint32_t lo, hi; };              // two half2
struct Act8  { int32_t q; float d; };            // 4 int8 + their block's scale
struct Act16F : Act16 {};                          // fp16 from an fp32 buffer, converted on load

__device__ __forceinline__ void load_act(Act32 & a, const void * x, const float *, int i) {
    const float4 f = *(const float4 *)((const float *)x + i);
    a.v[0] = f.x; a.v[1] = f.y; a.v[2] = f.z; a.v[3] = f.w;
}
__device__ __forceinline__ void load_act(Act16 & a, const void * x, const float *, int i) {
    const uint2 u = *(const uint2 *)((const uint16_t *)x + i);
    a.lo = u.x; a.hi = u.y;
}
__device__ __forceinline__ void load_act(Act16F & a, const void * x, const float *, int i) {
    const float4 f = *(const float4 *)((const float *)x + i);
    const h2_t lo = {(_Float16)f.x, (_Float16)f.y}, hi = {(_Float16)f.z, (_Float16)f.w};
    a.lo = __builtin_bit_cast(uint32_t, lo); a.hi = __builtin_bit_cast(uint32_t, hi);
}
__device__ __forceinline__ void load_act(Act8 & a, const void * x, const float * xd, int i) {
    a.q = *(const int32_t *)((const int8_t *)x + i); a.d = xd[i / 32];
}

// sum_j w_j * x_j over 4 weights given as 4 signed bytes (packed, byte j = weight j), in fp32
__device__ __forceinline__ float dot4s(uint32_t w, const Act32 & a) {
    float s = (float)(int8_t)(w & 0xff) * a.v[0];
    s = __builtin_fmaf((float)(int8_t)((w >> 8) & 0xff), a.v[1], s);
    s = __builtin_fmaf((float)(int8_t)((w >> 16) & 0xff), a.v[2], s);
    return __builtin_fmaf((float)(int8_t)(w >> 24), a.v[3], s);
}
__device__ __forceinline__ float dot4s(uint32_t w, const Act16 & a) {
    // signed byte b -> fp16 1024 + (b + 128) exactly (exponent 10: integers 1024 .. 2047), minus 1152
    const uint32_t u = w ^ 0x80808080u;
    const uint32_t p01 = __builtin_amdgcn_perm(0u, u, 0x0c010c00u) | 0x64006400u;   // bytes 0, 1 -> halves
    const uint32_t p23 = __builtin_amdgcn_perm(0u, u, 0x0c030c02u) | 0x64006400u;
    const h2_t off = {(_Float16)1152.0f, (_Float16)1152.0f};
    const h2_t w01 = as_h2(p01) - off, w23 = as_h2(p23) - off;
    float s = __builtin_amdgcn_fdot2(w01, as_h2(a.lo), 0.0f, false);
    return __builtin_amdgcn_fdot2(w23, as_h2(a.hi), s, false);
}
__device__ __forceinline__ float dot4s(uint32_t w, const Act8 & a) {
    return (float)__builtin_amdgcn_sudot4(true, (int)w, true, a.q, 0, false) * a.d;
}
// the same over 4 unsigned bytes (0 .. 255)
__device__ __forceinline__ float dot4u(uint32_t w, const Act32 & a) {
    float s = (float)(w & 0xff) * a.v[0];   // (byte -> float: the compiler emits v_cvt_f32_ubyte0..3, one op per weight)
    s = __builtin_fmaf((float)((w >> 8) & 0xff), a.v[1], s);
    s = __builtin_fmaf((float)((w >> 16) & 0xff), a.v[2], s);
    return __builtin_fmaf((float)(w >> 24), a.v[3], s);
}
__device__ __forceinline__ float dot4u(uint32_t w, const Act16 & a) {
    const uint32_t p01 = __builtin_amdgcn_perm(0u, w, 0x0c010c00u) | 0x64006400u;
    const uint32_t p23 = __builtin_amdgcn_perm(0u, w, 0x0c030c02u) | 0x64006400u;
    const h2_t off = {(_Float16)1024.0f, (_Float16)1024.0f};
    const h2_t w01 = as_h2(p01) - off, w23 = as_h2(p23) - off;
    float s = __builtin_amdgcn_fdot2(w01, as_h2(a.lo), 0.0f, false);
    return __builtin_amdgcn_fdot2(w23, as_h2(a.hi), s, false);
}
__device__ __forceinline__ float dot4u(uint32_t w, const Act8 & a) {
    return (float)__builtin_amdgcn_sudot4(false, (int)w, true, a.q, 0, false) * a.d;
}
// the sum of the 4 activation values (Q4_K's min term, A32 / A16; A8 takes the exact block sum instead)
__device__ __forceinline__ float sum4(const Act32 & a) { return ((a.v[0] + a.v[1]) + a.v[2]) + a.v[3]; }
__device__ __forceinline__ float sum4(const Act16 & a) {
    const h2_t one = {(_Float16)1.0f, (_Float16)1.0f};
    return __builtin_amdgcn_fdot2(as_h2(a.hi), one, __builtin_amdgcn_fdot2(as_h2(a.lo), one, 0.0f, false), false);
}

// ---- Q6_K: 256 values per 210-byte block: ql[128] low 4 bits, qh[64] high 2 bits, int8 scales[16] (one per 16 values),
// fp16 d. Value v = n * 128 + qd * 32 + c (half n, quarter qd, column c < 32): low bits from ql[n * 64 + (qd & 1) * 32 + c]
// (low nibble for qd < 2, high for qd >= 2), high bits (qh[n * 32 + c] >> 2 qd) & 3, scale scales[n * 8 + qd * 2 + c / 16];
// w = d * scale * (q - 32). A lane takes half n, columns c4 .. c4 + 3 of one block: 16 values, 4 groups (the quarters)
template <class ACT>
__device__ __forceinline__ void q6k_block(const uint8_t * blk, const void * x, const float * xd, int v0, int n, int c4,
                                          const ACT (&a)[4], float & acc) {
    const uint32_t lo = ld32h(blk + n * 64 + c4), hi = ld32h(blk + n * 64 + 32 + c4), qh = ld32h(blk + 128 + n * 32 + c4);
    const float d = f16f(*(const uint16_t *)(blk + 208));
    const int8_t * sc = (const int8_t *)(blk + 192) + n * 8 + c4 / 16;
    const uint32_t q[4] = {(lo & 0x0f0f0f0fu) | ((qh & 0x03030303u) << 4), (hi & 0x0f0f0f0fu) | (((qh >> 2) & 0x03030303u) << 4),
                           ((lo >> 4) & 0x0f0f0f0fu) | (((qh >> 4) & 0x03030303u) << 4),
                           ((hi >> 4) & 0x0f0f0f0fu) | (((qh >> 6) & 0x03030303u) << 4)};
#pragma unroll
    for (int qd = 0; qd < 4; qd++) {
        const uint32_t b = q[qd] ^ 0x20202020u;                    // q - 32 as a 6-bit signed value ...
        const uint32_t s8 = b | ((b & 0x20202020u) * 6u);          // ... sign-extended to 8 bits (bit 5 -> bits 6, 7)
        acc = __builtin_fmaf(d * (float)sc[qd * 2], dot4s(s8, a[qd]), acc);
    }
}

template <class ACT>
__device__ __forceinline__ void own_rows_q6_k(const uint8_t * __restrict__ W, int row0, int nrows, int K,
                                                     const void * __restrict__ x, const float * __restrict__ xd,
                                                     const float * __restrict__ xs, float * __restrict__ out, int lane) {
    // rows row0 .. row0 + OWN_R - 1 that are < nrows (one wave; lane = 0 .. 31)
    const int nb = K / 256, rb = nb * 210, half = lane >> 4, n = (lane >> 3) & 1, c4 = (lane & 7) * 4;
    float acc[OWN_R] = {};
    for (int b = half; b < nb; b += 2) {
        const int v0 = b * 256 + n * 128 + c4;
        ACT a[4];
#pragma unroll
        for (int qd = 0; qd < 4; qd++) load_act(a[qd], x, xd, v0 + qd * 32);
#pragma unroll
        for (int r = 0; r < OWN_R; r++)
            if (row0 + r < nrows) q6k_block(W + (size_t)(row0 + r) * rb + b * 210, x, xd, v0, n, c4, a, acc[r]);
    }
#pragma unroll
    for (int r = 0; r < OWN_R; r++) {
        const float s = wave_sum(acc[r]);
        if (lane == 0 && row0 + r < nrows) out[row0 + r] = s;
    }
}

// ---- Q4_K: 256 values per 144-byte block: fp16 d, fp16 dmin, scales[12] (6-bit scale and min for 8 sub-blocks of 32),
// qs[128]. Chunk j < 4 holds 64 values in 32 bytes: low nibbles -> values j * 64 + c (sub-block 2j), high nibbles -> j * 64
// + 32 + c (sub-block 2j + 1). w = d * sc * q - dmin * m. A lane takes chunk j, columns c4 .. c4 + 3: 8 values, 2 groups
__device__ __forceinline__ void q4k_scale_min(const uint8_t * s, int j, int & sc, int & m) {
    if (j < 4) { sc = s[j] & 63; m = s[j + 4] & 63; }
    else { sc = (s[j + 4] & 0xf) | ((s[j - 4] >> 6) << 4); m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4); }
}
template <class ACT>
__device__ __forceinline__ float minsum(const ACT & a, const float * xs, int i, bool first) { return sum4(a); }
template <>
__device__ __forceinline__ float minsum<Act8>(const Act8 &, const float * xs, int i, bool first) {
    return first ? xs[i / 32] : 0.0f;   // the exact fp32 sum of the 32-block, added once (by the lane at column 0)
}

template <class ACT>
__device__ __forceinline__ void own_rows_q4_k(const uint8_t * __restrict__ W, int row0, int nrows, int K,
                                                     const void * __restrict__ x, const float * __restrict__ xd,
                                                     const float * __restrict__ xs, float * __restrict__ out, int lane) {
    // rows row0 .. row0 + OWN_R - 1 that are < nrows (one wave; lane = 0 .. 31)
    const int nb = K / 256, rb = nb * 144, j = lane >> 3, c4 = (lane & 7) * 4;
    float acc[OWN_R] = {};
    for (int b = 0; b < nb; b++) {
        const int vlo = b * 256 + j * 64 + c4, vhi = vlo + 32;
        ACT alo, ahi;
        load_act(alo, x, xd, vlo); load_act(ahi, x, xd, vhi);
        const float mlo = minsum(alo, xs, vlo, c4 == 0), mhi = minsum(ahi, xs, vhi, c4 == 0);
#pragma unroll
        for (int r = 0; r < OWN_R; r++) {
            if (row0 + r >= nrows) continue;
            const uint8_t * blk = W + (size_t)(row0 + r) * rb + b * 144;
            const float d = f16f(*(const uint16_t *)blk), dmin = f16f(*(const uint16_t *)(blk + 2));
            int sc0, m0, sc1, m1;
            q4k_scale_min(blk + 4, 2 * j, sc0, m0); q4k_scale_min(blk + 4, 2 * j + 1, sc1, m1);
            const uint32_t qs = *(const uint32_t *)(blk + 16 + j * 32 + c4);   // (144-byte blocks: 4-byte aligned)
            float s = acc[r];
            s = __builtin_fmaf(d * (float)sc0, dot4u(qs & 0x0f0f0f0fu, alo), s);
            s = __builtin_fmaf(-dmin * (float)m0, mlo, s);
            s = __builtin_fmaf(d * (float)sc1, dot4u((qs >> 4) & 0x0f0f0f0fu, ahi), s);
            acc[r] = __builtin_fmaf(-dmin * (float)m1, mhi, s);
        }
    }
#pragma unroll
    for (int r = 0; r < OWN_R; r++) {
        const float s = wave_sum(acc[r]);
        if (lane == 0 && row0 + r < nrows) out[row0 + r] = s;
    }
}

// ---- Q8_0: 32 values per 34-byte block: fp16 d, int8 qs[32]; w = d * q. A lane takes 4 values of one block (8 lanes per
// block, 4 blocks per wave step): 1 group
template <class ACT>
__device__ __forceinline__ void own_rows_q8_0(const uint8_t * __restrict__ W, int row0, int nrows, int K,
                                                     const void * __restrict__ x, const float * __restrict__ xd,
                                                     const float * __restrict__ xs, float * __restrict__ out, int lane) {
    // rows row0 .. row0 + OWN_R - 1 that are < nrows (one wave; lane = 0 .. 31)
    const int nb = K / 32, rb = nb * 34, sub = lane >> 3, c4 = (lane & 7) * 4;
    float acc[OWN_R] = {};
    for (int b = sub; b < nb; b += 4) {
        ACT a;
        load_act(a, x, xd, b * 32 + c4);
#pragma unroll
        for (int r = 0; r < OWN_R; r++) {
            if (row0 + r >= nrows) continue;
            const uint8_t * blk = W + (size_t)(row0 + r) * rb + b * 34;
            acc[r] = __builtin_fmaf(f16f(*(const uint16_t *)blk), dot4s(ld32h(blk + 2 + c4), a), acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < OWN_R; r++) {
        const float s = wave_sum(acc[r]);
        if (lane == 0 && row0 + r < nrows) out[row0 + r] = s;
    }
}


// ---- Q4_K, one dependent round trip per pass (the decode-floor tile's structure, from its measured lesson: loads that
// wait on each other in series cost ~0.5 us each). A lane unit is 16 bytes of one block's quants: unit u = lane + 32 i,
// block u / 8, part p = u % 8 -> chunk j = p / 2, half h = p % 2, quant bytes 16 + 32 j + 16 h of the block; its 16 low
// nibbles are values 64 j + 16 h + (0 .. 15) (sub-block 2 j), its high nibbles the same + 32 (sub-block 2 j + 1). Every
// load of a pass (each row's 16-byte header d | dmin | scales[12] and quant unit, the lane's fp16 activations) is issued
// unconditionally, from a valid address, before any arithmetic; the 6-bit scale / min decode is a select, not a branch.
// RW rows per wave, UP units per lane per pass (K = 2048: 2 units, one pass; 4096: two passes; 768: lanes 24-31 idle).
// Activations: fp16 [K] (x16); sums of x per 16-value run in fp32 (v_dot2 with ones) for the min terms, shared by the rows.
typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));
__device__ __forceinline__ uint32_t q4k_byte(const u32x4_t & hd, int k) {   // scale byte k (0 .. 11) of a header
    const uint32_t d = k < 4 ? hd.y : k < 8 ? hd.z : hd.w;                    // (selects: v_cndmask, no branch)
    return (d >> (8 * (k & 3))) & 0xffu;
}
__device__ __forceinline__ void q4k_sm(const u32x4_t & hd, int s, float & sc, float & m) {   // sub-block s: scale, min
    const uint32_t a = q4k_byte(hd, s & 7), b = q4k_byte(hd, s + 4), c = q4k_byte(hd, (s - 4) & 3);
    const bool lo = s < 4;
    sc = (float)(lo ? (a & 63u) : ((b & 0xfu) | ((c >> 6) << 4)));
    m = (float)(lo ? (b & 63u) : ((b >> 4) | ((a >> 6) << 4)));
}
// sum over 16 quant nibbles (4 dwords, nibble n of dword w at shift sh) times 16 fp16 x values (2 u32x4) in fp32
__device__ __forceinline__ float q4k_dot16(const u32x4_t & q, int sh, const u32x4_t & xa, const u32x4_t & xb) {
    const h2_t off = {(_Float16)1024.0f, (_Float16)1024.0f};
    const uint32_t qw[4] = {q.x, q.y, q.z, q.w}, xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; w++) {
        const uint32_t n = (qw[w] >> sh) & 0x0f0f0f0fu;                                // 4 nibbles as bytes
        const uint32_t p01 = __builtin_amdgcn_perm(0u, n, 0x0c010c00u) | 0x64006400u;  // bytes 0, 1 -> fp16 1024 + q
        const uint32_t p23 = __builtin_amdgcn_perm(0u, n, 0x0c030c02u) | 0x64006400u;
        s = __builtin_amdgcn_fdot2(as_h2(p01) - off, as_h2(xw[2 * w]), s, false);
        s = __builtin_amdgcn_fdot2(as_h2(p23) - off, as_h2(xw[2 * w + 1]), s, false);
    }
    return s;
}
__device__ __forceinline__ float sum16h(const u32x4_t & xa, const u32x4_t & xb) {
    const h2_t one = {(_Float16)1.0f, (_Float16)1.0f};
    const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; k++) s = __builtin_amdgcn_fdot2(as_h2(xw[k]), one, s, false);
    return s;
}
template <int K, int RW, int UP = 2>
__device__ __forceinline__ void own_rows_q4k_1rt(const uint8_t * __restrict__ W, int row0, int nrows,
                                                 const _Float16 * __restrict__ x16, float * __restrict__ out, int lane) {
    constexpr int NB = K / 256, NUNITS = NB * 8, NU = (NUNITS + 31) / 32;
    constexpr size_t RB = (size_t)NB * 144;
    float acc[RW];
#pragma unroll
    for (int r = 0; r < RW; r++) acc[r] = 0.0f;
#pragma unroll
    for (int i0 = 0; i0 < NU; i0 += UP) {
        u32x4_t hd[RW][UP], qs[RW][UP], xl[UP][2], xh[UP][2];
#pragma unroll
        for (int i = 0; i < UP; i++) {                       // every load of the pass, unconditionally
            const int u = lane + 32 * (i0 + i), uu = (i0 + i < NU && u < NUNITS) ? u : 0;
            const int b = uu >> 3, j = (uu & 7) >> 1, h = uu & 1;
#pragma unroll
            for (int r = 0; r < RW; r++) {
                const uint8_t * blk = W + (size_t)min(row0 + r, nrows - 1) * RB + (size_t)b * 144;
                hd[r][i] = *(const u32x4_t *)blk;
                qs[r][i] = *(const u32x4_t *)(blk + 16 + 32 * j + 16 * h);
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
            const float sxl = sum16h(xl[i][0], xl[i][1]), sxh = sum16h(xh[i][0], xh[i][1]);
#pragma unroll
            for (int r = 0; r < RW; r++) {
                const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd[r][i].x & 0xffffu));
                const float dm = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd[r][i].x >> 16));
                float sc0, m0, sc1, m1;
                q4k_sm(hd[r][i], 2 * j, sc0, m0); q4k_sm(hd[r][i], 2 * j + 1, sc1, m1);
                float a = acc[r];
                a = __builtin_fmaf(d * sc0, q4k_dot16(qs[r][i], 0, xl[i][0], xl[i][1]), a);
                a = __builtin_fmaf(-dm * m0, sxl, a);
                a = __builtin_fmaf(d * sc1, q4k_dot16(qs[r][i], 4, xh[i][0], xh[i][1]), a);
                acc[r] = __builtin_fmaf(-dm * m1, sxh, a);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RW; r++) {
        const float s = wave_sum(acc[r]);
        if (lane == 0 && row0 + r < nrows) out[row0 + r] = s;
    }
}

// ---- Q6_K in Caminus's plane layout, one round trip per pass. GGUF's Q6_K block (ql[128] | qh[64] | scales[16] | d) is
// 210 bytes, so in place its parts are not 16-byte aligned; at load the matrix is rearranged into three planes, same total
// bytes: A = per row, per block, ql | qh (192 B); B = per row, per block, the 16 int8 scales; C = per row, per block, d (fp16).
// Row r of an R-row matrix with NB blocks per row: A at r * NB * 192, B at R * NB * 192 + r * NB * 16, C at R * NB * 208 +
// r * NB * 2 (caminus.engine.own_layout.q6k_planes). A lane unit is 64 values: half n and 16 columns c16 of a block, all four
// quarters: ql bytes n * 64 + c16 (quarters 0 low, 2 high) and n * 64 + 32 + c16 (1 low, 3 high), qh bytes n * 32 + c16 (bits
// 2 qd), scales n * 8 + c16 / 16 + 2 qd; value n * 128 + 32 qd + c16 + (0 .. 15) = d * scale * (q - 32).
template <int K, int RW, int UP = 1>
__device__ __forceinline__ void own_rows_q6k_planes(const uint8_t * __restrict__ Wm, int nrows_total, int row0, int nrows,
                                                    const _Float16 * __restrict__ x16, float * __restrict__ out, int lane) {
    constexpr int NB = K / 256, NUNITS = NB * 4, NU = (NUNITS + 31) / 32;
    const uint8_t * A = Wm;
    const int8_t * B = (const int8_t *)(Wm + (size_t)nrows_total * NB * 192);
    const uint16_t * C = (const uint16_t *)(Wm + (size_t)nrows_total * NB * 208);
    const h2_t off = {(_Float16)1056.0f, (_Float16)1056.0f};   // 1024 + 32: fp16 (1024 + q) - 1056 = q - 32, exact
    float acc[RW];
#pragma unroll
    for (int r = 0; r < RW; r++) acc[r] = 0.0f;
#pragma unroll
    for (int i0 = 0; i0 < NU; i0 += UP) {
        u32x4_t q0[RW][UP], q1[RW][UP], qh[RW][UP], sc[RW][UP], xv[UP][4][2];
        uint16_t dh[RW][UP];
#pragma unroll
        for (int i = 0; i < UP; i++) {                       // every load of the pass, unconditionally
            const int u = lane + 32 * (i0 + i), uu = (i0 + i < NU && u < NUNITS) ? u : 0;
            const int b = uu >> 2, n = (uu >> 1) & 1, c16 = (uu & 1) * 16;
#pragma unroll
            for (int r = 0; r < RW; r++) {
                const size_t rr = (size_t)min(row0 + r, nrows - 1);
                const uint8_t * a = A + (rr * NB + b) * 192;
                q0[r][i] = *(const u32x4_t *)(a + n * 64 + c16);
                q1[r][i] = *(const u32x4_t *)(a + n * 64 + 32 + c16);
                qh[r][i] = *(const u32x4_t *)(a + 128 + n * 32 + c16);
                sc[r][i] = *(const u32x4_t *)(B + (rr * NB + b) * 16);
                dh[r][i] = C[rr * NB + b];
            }
#pragma unroll
            for (int qd = 0; qd < 4; qd++) {
                const _Float16 * xp = x16 + b * 256 + n * 128 + qd * 32 + c16;
                xv[i][qd][0] = *(const u32x4_t *)xp; xv[i][qd][1] = *(const u32x4_t *)(xp + 8);
            }
        }
#pragma unroll
        for (int i = 0; i < UP; i++) {
            const int u = lane + 32 * (i0 + i);
            if (i0 + i >= NU || u >= NUNITS) continue;
            const int n = (u >> 1) & 1, c16 = (u & 1) * 16;
#pragma unroll
            for (int r = 0; r < RW; r++) {
                const float d = (float)__builtin_bit_cast(_Float16, dh[r][i]);
                const uint32_t scw[4] = {sc[r][i].x, sc[r][i].y, sc[r][i].z, sc[r][i].w};
                const uint32_t lw0[4] = {q0[r][i].x, q0[r][i].y, q0[r][i].z, q0[r][i].w};
                const uint32_t lw1[4] = {q1[r][i].x, q1[r][i].y, q1[r][i].z, q1[r][i].w};
                const uint32_t hw[4] = {qh[r][i].x, qh[r][i].y, qh[r][i].z, qh[r][i].w};
                float a = acc[r];
#pragma unroll
                for (int qd = 0; qd < 4; qd++) {
                    const int si = n * 8 + c16 / 16 + 2 * qd;                       // scale index (0 .. 15), per lane
                    const float s = (float)(int8_t)((scw[si >> 2] >> (8 * (si & 3))) & 0xffu);
                    const uint32_t xw[8] = {xv[i][qd][0].x, xv[i][qd][0].y, xv[i][qd][0].z, xv[i][qd][0].w,
                                            xv[i][qd][1].x, xv[i][qd][1].y, xv[i][qd][1].z, xv[i][qd][1].w};
                    float aq = 0.0f;
#pragma unroll
                    for (int w = 0; w < 4; w++) {
                        const uint32_t lo = ((qd & 1) ? lw1[w] : lw0[w]) >> ((qd & 2) ? 4 : 0);
                        const uint32_t q = (lo & 0x0f0f0f0fu) | (((hw[w] >> (2 * qd)) & 0x03030303u) << 4);   // 0 .. 63
                        const uint32_t p01 = __builtin_amdgcn_perm(0u, q, 0x0c010c00u) | 0x64006400u;
                        const uint32_t p23 = __builtin_amdgcn_perm(0u, q, 0x0c030c02u) | 0x64006400u;
                        aq = __builtin_amdgcn_fdot2(as_h2(p01) - off, as_h2(xw[2 * w]), aq, false);
                        aq = __builtin_amdgcn_fdot2(as_h2(p23) - off, as_h2(xw[2 * w + 1]), aq, false);
                    }
                    a = __builtin_fmaf(d * s, aq, a);
                }
                acc[r] = a;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RW; r++) {
        const float s = wave_sum(acc[r]);
        if (lane == 0 && row0 + r < nrows) out[row0 + r] = s;
    }
}
