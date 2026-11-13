// Caminus's own quantized matvec for GGUF's codebook formats (IQ4_XS; IQ3_XXS, IQ3_S and IQ2_XXS follow), in the
// one-round-trip form of kernels/own_quant.h (own_rows_q4k_1rt): fp16 activations, fp32 sums, RW rows per wave, a lane
// owns units of 32 consecutive values (one scale each), every load of a pass issued before any arithmetic. Written from
// the block layouts (the host decoders: python/caminus/own_dequant.py, bitwise gguf-py's); no ggml-cuda code. The
// codebooks are format constants: kernels/iq_grids.h (ggml's data, MIT notice there).
#pragma once
#include "own_quant.h"

// ---- IQ4_XS: 256 values per 136-byte block: d (fp16), scales_h (u16: 2 high bits of each of the 8 scales), scales_l[4]
// (their low nibbles), qs[128]. Unit ib (32 values, ib < 8) = the 16 bytes qs + 16 ib: byte k's low nibble is value k,
// its high nibble value 16 + k; value = d * (ls - 32) * kv[nibble], ls = the unit's 6-bit scale, kv = 16 int8 values
// (kvalues_iq4nl). Read in place: the block is 8-byte aligned (136 = 17 * 8), a unit one 16-byte load, the header one
// 8-byte load. No offset term, so a unit is one scale times a 32-value dot.
// The lookup, 4 nibbles at a time: kv + 128 as bytes (1 .. 241) sits in four dwords; v_perm_b32 picks entries 0 .. 7
// from two of them by the nibble's low 3 bits, a second one entries 8 .. 15, and the nibble's bit 3 selects between
// them. The result is the weight + 128: own_quant.h dot4s' form after its xor.
#define IQ4_T0 0x3f2d1801u   // kv + 128, entries 0 .. 3 (byte k = entry k): 1, 24, 45, 63
#define IQ4_T1 0x766a5d4fu   // 4 .. 7: 79, 93, 106, 118
#define IQ4_T2 0xa6998d81u   // 8 .. 11: 129, 141, 153, 166
#define IQ4_T3 0xf1d9c5b5u   // 12 .. 15: 181, 197, 217, 241
__device__ __forceinline__ uint32_t iq4_lut(uint32_t n) {   // 4 nibbles as bytes -> 4 bytes kv + 128
    const uint32_t sel = n & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(IQ4_T1, IQ4_T0, sel), hi = __builtin_amdgcn_perm(IQ4_T3, IQ4_T2, sel);
    const uint32_t m = ((n >> 3) & 0x01010101u) * 0xffu;
    return lo ^ ((lo ^ hi) & m);
}
// s + sum over 16 nibbles (4 dwords, nibble n of dword w at shift sh) of kv[nibble] times 16 fp16 x values, in fp32
__device__ __forceinline__ float iq4_dot16(const u32x4_t & q, int sh, const u32x4_t & xa, const u32x4_t & xb, float s,
                                           bool fence = false) {
    const h2_t off = {(_Float16)1152.0f, (_Float16)1152.0f};   // fp16 (1024 + (kv + 128)) - 1152 = kv, exact
    const uint32_t qw[4] = {q.x, q.y, q.z, q.w}, xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
    for (int w = 0; w < 4; w++) {
        if (fence) __builtin_amdgcn_sched_barrier(0);
        const uint32_t u = iq4_lut((qw[w] >> sh) & 0x0f0f0f0fu);
        const uint32_t p01 = __builtin_amdgcn_perm(0u, u, 0x0c010c00u) | 0x64006400u;
        const uint32_t p23 = __builtin_amdgcn_perm(0u, u, 0x0c030c02u) | 0x64006400u;
        s = __builtin_amdgcn_fdot2(as_h2(p01) - off, as_h2(xw[2 * w]), s, false);
        s = __builtin_amdgcn_fdot2(as_h2(p23) - off, as_h2(xw[2 * w + 1]), s, false);
    }
    return s;
}
// K = 0: the row length is the runtime Kr; res: the rows' sums are returned in res[RW] (every lane) and nothing is
// stored; UP units per lane per pass, UNR the pass loop's unroll count (own_rows_q4k_1rt's parameters, K3: 1, 1).
// OWN_READONLY (measurement builds): the same loads folded by xor, no decode.
template <int K, int RW, int UP = 2, int UNR = 64>
__device__ __forceinline__ void own_rows_iq4xs_1rt(const uint8_t * __restrict__ W, int row0, int nrows,
                                                   const _Float16 * __restrict__ x16, float * __restrict__ out, int lane,
                                                   int Kr = K, float * res = nullptr) {
    const int NB = (K ? K : Kr) / 256, NUNITS = NB * 8, NU = (NUNITS + 31) / 32;
    const size_t RB = (size_t)NB * 136;
    float acc[RW];
#pragma unroll
    for (int r = 0; r < RW; r++) acc[r] = 0.0f;
#pragma unroll UNR
    for (int i0 = 0; i0 < NU; i0 += UP) {
        u32x2_t hd[RW][UP];
        u32x4_t qs[RW][UP], xl[UP][2], xh[UP][2];
#pragma unroll
        for (int i = 0; i < UP; i++) {                       // every load of the pass, unconditionally
            const int u = lane + 32 * (i0 + i), uu = (i0 + i < NU && u < NUNITS) ? u : 0;
            const int b = uu >> 3, ib = uu & 7;
#pragma unroll
            for (int r = 0; r < RW; r++) {
                const uint8_t * blk = W + (size_t)min(row0 + r, nrows - 1) * RB + (size_t)b * 136;
                qs[r][i] = *(const u32x4_t *)(blk + 8 + 16 * ib);
#if defined(OWN_READONLY) && OWN_READONLY >= 2
                hd[r][i] = (u32x2_t){qs[r][i].x, qs[r][i].y};
#else
                hd[r][i] = *(const u32x2_t *)blk;
#endif
            }
            const _Float16 * xp = x16 + 32 * uu;
#if defined(OWN_READONLY) && OWN_READONLY >= 2
            xl[i][0] = xl[i][1] = xh[i][0] = xh[i][1] = qs[0][i]; (void)xp;
#else
            xl[i][0] = *(const u32x4_t *)xp; xl[i][1] = *(const u32x4_t *)(xp + 8);
            xh[i][0] = *(const u32x4_t *)(xp + 16); xh[i][1] = *(const u32x4_t *)(xp + 24);
#endif
        }
#if defined(OWN_READONLY)
#pragma unroll
        for (int i = 0; i < UP; i++) {
            const u32x4_t xo = xl[i][0] ^ xl[i][1] ^ xh[i][0] ^ xh[i][1];
#pragma unroll
            for (int r = 0; r < RW; r++) {
                const u32x4_t v = qs[r][i] ^ xo;
                acc[r] += (float)((v.x ^ v.y ^ v.z ^ v.w ^ hd[r][i].x ^ hd[r][i].y) & 0xffu);
            }
        }
        continue;
#endif
#pragma unroll
        for (int i = 0; i < UP; i++) {
            const int u = lane + 32 * (i0 + i);
            if (i0 + i >= NU || u >= NUNITS) continue;
            const int ib = u & 7;
#pragma unroll
            for (int r = 0; r < RW; r++) {
                if (UNR == 1) __builtin_amdgcn_sched_barrier(0);
                const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd[r][i].x & 0xffffu));
                const uint32_t ls = ((hd[r][i].y >> (4 * ib)) & 0xfu) | ((((hd[r][i].x >> 16) >> (2 * ib)) & 3u) << 4);
                float s = iq4_dot16(qs[r][i], 0, xl[i][0], xl[i][1], 0.0f, UNR == 1);
                s = iq4_dot16(qs[r][i], 4, xh[i][0], xh[i][1], s, UNR == 1);
                acc[r] = __builtin_fmaf(d * (float)((int)ls - 32), s, acc[r]);
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

// ---- the grid formats: IQ3_XXS, IQ3_S, IQ2_XXS. A group of 4 (IQ3) or 8 (IQ2_XXS) weights is one index into a grid of
// magnitudes (kernels/iq_grids.h) plus a sign bit per weight; a unit of 32 weights has one scale. The grids live in the
// workgroup's local memory (LDS: 1 KB, 2 KB, 2 KB; one 3 KB array), staged once per launch by iq_stage (256 threads,
// then a barrier): a pass is the weight loads (one memory round trip, as own_rows_q4k_1rt), then the grid reads the
// index bytes name (a second, local round trip), then the arithmetic.
// The matrix is read in Caminus's record layout (caminus.engine.own_layout.iq_records; the same bytes as the GGUF
// blocks, which are 98 / 110 / 66 B and whose fields a unit needs sit at five different offsets): plane A = a record
// per unit of 32 weights, plane B = a header per block, then 16 bytes of padding. A unit is then ONE 16-byte load
// and its block's header ONE 8-byte load (both may run past their record into the next bytes of the buffer). In place,
// a row's unit took five loads at five lane-dependent addresses, and the addresses alone cost k3_run a spilled VGPR.
#include "iq_grids.h"
// One array for the three grids, IQ2_XXS's over the two IQ3 grids' words: 3 KB, not 5. A launch stages IQ3_XXS and / or
// IQ3_S, or IQ2_XXS (the host refuses a tape that needs both kinds). Why: the compiler sizes a kernel's register
// budget by the workgroups its LDS lets a CU hold; K3's decode module sits just under the size where 4 fit (15,748 of
// 16,384 B), and with 5 KB more k3_run was allowed, and took, 97 VGPRs.
// IQ2_XXS entry e (8 weights) is words e (weights 0 .. 3) and 256 + e (4 .. 7): a lookup's address is then an index byte
// plus one bit times 256 in every format (IQ3_S: its index's ninth bit), and the tile's eight lookups are one statement.
#ifdef K3_Q35_GSQ   // (the Qwen3.8-27B GSQ-RCO module: its files use IQ3_XXS, IQ3_S AND IQ2_XXS, so IQ2_XXS gets its own words)
enum { IQ_G3X = 0, IQ_G3S = 256, IQ_G2X = 768, IQ_GN = 1280 };
#else
enum { IQ_G3X = 0, IQ_G3S = 256, IQ_G2X = 0, IQ_GN = 768 };   // the grids' first words
#endif
__shared__ uint32_t IQ_G[IQ_GN];
// which: bit 0 IQ3_XXS, bit 1 IQ3_S, or bit 2 alone IQ2_XXS; all 256 threads of the workgroup call it (one barrier)
__device__ __forceinline__ void iq_stage(int tid, int which) {
    if (which & 1) IQ_G[IQ_G3X + tid] = IQ3XXS_GRID[tid];
    if (which & 2) { IQ_G[IQ_G3S + tid] = IQ3S_GRID[tid]; IQ_G[IQ_G3S + 256 + tid] = IQ3S_GRID[256 + tid]; }
    if (which & 4) { const uint64_t e = IQ2XXS_GRID[tid]; IQ_G[IQ_G2X + tid] = (uint32_t)e; IQ_G[IQ_G2X + 256 + tid] = (uint32_t)(e >> 32); }
    __syncthreads();
}
// 4 magnitude bytes m (1 .. 127 each) and 4 sign bits sb (bit k set: weight k is negative) -> 4 bytes weight + 128:
// the bits spread to one per byte (a multiply), -m = ~m + 1 per byte (m >= 1: no carry leaves a byte)
__device__ __forceinline__ uint32_t iq_signed(uint32_t m, uint32_t sb) {
    const uint32_t b = ((sb & 0xfu) * 0x00204081u) & 0x01010101u;
    return ((m ^ (b * 0xffu)) + b) ^ 0x80808080u;
}
// s + sum of 4 weights (bytes weight + 128) times 4 fp16 x values (two words), in fp32
__device__ __forceinline__ float iq_dot4(uint32_t u, uint32_t x0, uint32_t x1, float s) {
    const h2_t off = {(_Float16)1152.0f, (_Float16)1152.0f};
    const uint32_t p01 = __builtin_amdgcn_perm(0u, u, 0x0c010c00u) | 0x64006400u;
    const uint32_t p23 = __builtin_amdgcn_perm(0u, u, 0x0c030c02u) | 0x64006400u;
    s = __builtin_amdgcn_fdot2(as_h2(p01) - off, as_h2(x0), s, false);
    return __builtin_amdgcn_fdot2(as_h2(p23) - off, as_h2(x1), s, false);
}
// the 8 sign bits of 8 weights from their 7-bit code: the code and its parity (IQ3_XXS, IQ2_XXS)
__device__ __forceinline__ uint32_t iq_signs8(uint32_t s7) { return s7 | ((__builtin_popcount(s7) & 1u) << 7); }

// One body for the three formats (FMT 0 IQ3_XXS, 1 IQ3_S, 2 IQ2_XXS; FMT 3: the format is the runtime, wave-uniform
// fmt: K3's form, one body in k3_run instead of three). Wm = the record layout of nrows_total rows; other parameters as
// own_rows_iq4xs_1rt. Record (RS bytes) and header (HS bytes) of a unit ib of a block:
//   IQ3_XXS  record 12: 8 index bytes (4 weights each) | four 7-bit sign codes (8 weights each) and scale s << 28;
//            header 2: d (fp16);                     weight = d (0.5 + s) 0.5 * +-grid
//   IQ3_S    record 13: 8 index bytes | 32 sign bits | qh (bit k: index k's bit 8);
//            header 6: d | the block's 8 scale nibbles; weight = d (1 + 2 s) * +-grid
//   IQ2_XXS  record 8: 4 index bytes (8 weights each) | four 7-bit sign codes and scale s << 28;
//            header 2: d;                             weight = d (0.5 + s) 0.25 * +-grid
template <int FMT, int K, int RW, int UP = 2, int UNR = 64>
__device__ __forceinline__ void own_rows_iqg_1rt(const uint8_t * __restrict__ Wm, int nrows_total, int row0, int nrows,
                                                 const _Float16 * __restrict__ x16, float * __restrict__ out, int lane,
                                                 int Kr = K, float * res = nullptr, int fmt = FMT) {
    const int F = FMT < 3 ? FMT : fmt;
    const bool f3x = F == 0, f3s = F == 1, f2x = F == 2;
    const int RS = f3x ? 12 : f3s ? 13 : 8, HS = f3s ? 6 : 2;
    const int NB = (K ? K : Kr) / 256, NUNITS = NB * 8, NU = (NUNITS + 31) / 32;
    const uint8_t * A = Wm;
    const uint8_t * B = Wm + (size_t)nrows_total * NUNITS * RS;
    float acc[RW];
#pragma unroll
    for (int r = 0; r < RW; r++) acc[r] = 0.0f;
#pragma unroll UNR
    for (int i0 = 0; i0 < NU; i0 += UP) {
        u32x4_t rc[RW][UP];
        u32x2_t hd[RW][UP];
        u32x4_t xv[UP][4];
#pragma unroll
        for (int i = 0; i < UP; i++) {                       // every weight and activation load of the pass
            const int u = lane + 32 * (i0 + i), uu = (i0 + i < NU && u < NUNITS) ? u : 0;
#pragma unroll
            for (int r = 0; r < RW; r++) {
                const size_t rw = (size_t)min(row0 + r, nrows - 1);
                rc[r][i] = *(const u32x4_t *)(A + (rw * NUNITS + uu) * RS);
                hd[r][i] = *(const u32x2_t *)(B + (rw * NB + (uu >> 3)) * HS);
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
                acc[r] += (float)((v.x ^ v.y ^ v.z ^ v.w ^ hd[r][i].x ^ hd[r][i].y) & 0xffu);
            }
        }
        continue;
#endif
#pragma unroll
        for (int i = 0; i < UP; i++) {
            const int u = lane + 32 * (i0 + i);
            if (i0 + i >= NU || u >= NUNITS) continue;
            const int ib = u & 7;
            const uint32_t xw[16] = {xv[i][0].x, xv[i][0].y, xv[i][0].z, xv[i][0].w, xv[i][1].x, xv[i][1].y, xv[i][1].z, xv[i][1].w,
                                     xv[i][2].x, xv[i][2].y, xv[i][2].z, xv[i][2].w, xv[i][3].x, xv[i][3].y, xv[i][3].z, xv[i][3].w};
#pragma unroll
            for (int r = 0; r < RW; r++) {
                if (UNR == 1) __builtin_amdgcn_sched_barrier(0);
                const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(hd[r][i].x & 0xffffu));
                // What differs between the formats is settled here, once per unit, so that the eight lookups and the eight
                // dot4 below are the same statements in every format. With the format chosen inside each lookup and each
                // sign nibble, a token of the IQ2_XXS arm took 5.63 ms in the runtime-format body (RX 7900 XT, depth 8);
                // settled per unit 5.12 ms; with the format fixed at compile time 4.95 ms. Same bits in the three.
                //   index bytes: 8 (IQ3), or IQ2_XXS's 4, each twice (its entry's two words: bit 8 set for the second)
                //   signs: 32 bits (IQ3_S), or four 7-bit codes, each with its parity as the eighth bit
                const uint32_t sw = f2x ? rc[r][i].y : rc[r][i].z, scn = (hd[r][i].x >> 16) | (hd[r][i].y << 16);
                uint32_t iw[2] = {rc[r][i].x, rc[r][i].y}, qh = 0u, gb = f2x ? (uint32_t)IQ_G2X : 0u, sg = sw, si = sw >> 28;   // (IQ_G2X is 0 outside K3_Q35_GSQ: the same code)
                if (f3s) { qh = rc[r][i].w; gb = IQ_G3S; si = (scn >> (4 * ib)) & 0xfu; }
                else {     // (branches on the wave-uniform format: 0.02 to 0.05 ms per token under selects of both forms)
                    sg = iq_signs8(sw & 127u) | (iq_signs8((sw >> 7) & 127u) << 8) |
                         (iq_signs8((sw >> 14) & 127u) << 16) | (iq_signs8((sw >> 21) & 127u) << 24);
                    if (f2x) {
                        iw[1] = __builtin_amdgcn_perm(0u, iw[0], 0x03030202u); iw[0] = __builtin_amdgcn_perm(0u, iw[0], 0x01010000u);
                        qh = 0xaau;
                    }
                }
                uint32_t m[8];                               // the grid reads: 8 words (4 weights each)
#pragma unroll
                for (int k = 0; k < 8; k++)
                    m[k] = IQ_G[gb + (((iw[k >> 2] >> (8 * (k & 3))) & 0xffu) | (((qh >> k) & 1u) << 8))];
                float s = 0.0f;
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    if (UNR == 1 && !(k & 1)) __builtin_amdgcn_sched_barrier(0);
                    s = iq_dot4(iq_signed(m[k], sg >> (4 * k)), xw[2 * k], xw[2 * k + 1], s);
                }
                // (0.5 + s) 0.5 = (1 + 2 s) 0.25 and (0.5 + s) 0.25 = (1 + 2 s) 0.125, exactly: one form for the three
                const float sc = (float)(1u + 2u * si) * (f3s ? 1.0f : f3x ? 0.25f : 0.125f);
                acc[r] = __builtin_fmaf(d * sc, s, acc[r]);
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
