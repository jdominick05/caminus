// Caminus's own Q2_0 matvec (docs/specs/spec-own-quant.md): the routed experts of Qwen3.6-35B-A3B's GSQ file (gate / up
// K 2048, down K 512) and the GSQ-RCO Coder's Q2_0 expert down (K 640, cut 384 / 256 over two cards), written from the
// block layout (python/caminus/own_dequant.py q2_0, bitwise gguf-py's); no ggml-cuda code.
// Q2_0: 64 values per 18-byte block: d (fp16), 16 code bytes; value j is bits 2 (j % 4) of byte j / 4, w = d (q - 1),
// q = 0 .. 3 (-d, 0, d, 2d). Read as little-endian dwords, code dword c of a block (c = 0 .. 3) holds values
// 16 c .. 16 c + 15, value 16 c + j at bits 2 j.
// In place: the tile reads the GGUF blocks as they are, any K that is a whole number of blocks (K % 64 == 0). No copy at
// load: the batched prefill reads the same bytes, and a second copy of the experts does not fit (Qwen3.6: 9.1 GB of
// Q2_0). The blocks are only 2-byte aligned, so a unit's 8 code bytes are one unaligned 8-byte load: gfx1100 serves
// unaligned global loads, and amdclang already merges 16-bit pairs such as own_quant.h's ld32h into them.
//
// One-round-trip form (own_quant.h own_rows_q4k_1rt: the same signature and the same K3 form <0, RW, 1, 1>): RW rows
// per wave, every load of a pass issued before any arithmetic, fp16 activations, fp32 sums.
//   A lane unit is 32 values: half h of block b (unit u: b = u / 2, h = u % 2), the block's d (a 16-bit load) and its
//   code dwords 2 h, 2 h + 1 (one 8-byte load at byte 2 + 8 h). A 64-value unit would hold 32 VGPRs of activations
//   alone (own_quant.h: K3 spilled with that).
//   Lanes: a row has NUNITS = K / 32 units. NUNITS <= 16 (K <= 512): each half of the wave takes half the rows (lanes
//   0 .. 15 rows row0 .. row0 + RW / 2 - 1, lanes 16 .. 31 the rest), lane l & 15 owning unit l & 15, so K 512 keeps
//   every lane busy (16 units); NUNITS > 16: every lane takes every row, unit l + 32 i in pass i (K 2048: 2 passes, K
//   640: 20 of 32 lanes). Which form a row gets depends on K alone (RW must be even), so every caller of a K gets the
//   same bits for a row: the plain rows kernel, the grouped gate | up, the grouped down.
// Decode, per code dword v (16 values), four shifts s_i = v >> 2 i (i = 0 .. 3), each giving two exact fp16 pairs:
//   A_i = values (i, 8 + i) from bits (0, 16) of s_i: (s_i & 0x00030003) | 0x64006400 is fp16 (1024 + q) per half
//         (exponent 10: integers exact), minus 1025 = q - 1 exactly
//   B_i = values (4 + i, 12 + i) from bits (8, 24) of s_i: (s_i & 0x03000300) | 0x000c000c, as a v_perm selector, picks
//         from 0x403c00bc each half's high byte (fp16 -1, 0, 1, 2 = 0xbc00, 0, 0x3c00, 0x4000: low bytes zero) and
//         zeros the low bytes: q - 1 exactly, no subtraction
//   The lane's 32 activations (16 fp16 pairs as loaded) are re-paired the same way once per pass (v_perm), shared by
//   the RW rows: (x_i, x_8+i) and (x_4+i, x_12+i) of each 16 values.
// Order (the contract for the T-token and prefill forms): per row and unit, for code dword c = 0, 1 of the unit
//   s_c = 0; s_c = dot2(A_0, .) .. in the order A_0 B_0 A_1 B_1 A_2 B_2 A_3 B_3 (v_dot2_f32_f16: exact products, the
//   fp32 accumulation rounded); acc = fma(d, s_0 + s_1, acc); units in lane order (i ascending), then the butterfly
//   (xor 16 .. 1, or 8 .. 1 within a half when the halves split the rows).
#pragma once
#include "own_quant.h"

// code dword v -> its 8 fp16 pairs of (q - 1), in the order A_0 B_0 A_1 B_1 A_2 B_2 A_3 B_3
__device__ __forceinline__ void q2_0_pairs(uint32_t v, h2_t (&w)[8]) {
    const h2_t off = {(_Float16)1025.0f, (_Float16)1025.0f};   // fp16 (1024 + q) - 1025 = q - 1, exact
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t s = v >> (2 * i);
        w[2 * i] = as_h2((s & 0x00030003u) | 0x64006400u) - off;
        w[2 * i + 1] = as_h2(__builtin_amdgcn_perm(0u, 0x403c00bcu, (s & 0x03000300u) | 0x000c000cu));
    }
}
// 16 fp16 activations as loaded (8 words: (x_2m, x_2m+1)) -> the same pairs: (x_i, x_8+i), (x_4+i, x_12+i), i = 0 .. 3
__device__ __forceinline__ void q2_0_xpairs(const uint32_t * xw, uint32_t (&xp)[8]) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t sel = (i & 1) ? 0x07060302u : 0x05040100u;   // the high halves of two words, or the low halves
        xp[2 * i] = __builtin_amdgcn_perm(xw[4 + (i >> 1)], xw[i >> 1], sel);
        xp[2 * i + 1] = __builtin_amdgcn_perm(xw[6 + (i >> 1)], xw[2 + (i >> 1)], sel);
    }
}

// rows row0 .. row0 + RW - 1 that are < nrows (one wave; rows of nrows contiguous rows of K values). K = 0: the runtime
// Kr (K3: one body for every K). res: the rows' sums are returned in res[RW] (every lane) and nothing is stored; else
// lane 0 stores them at out[row0 + r]. UNR as own_rows_q4k_1rt (K3: UNR 1, the math behind scheduling fences).
template <int K, int RW, int UP = 1, int UNR = 64>
__device__ __forceinline__ void own_rows_q2_0_1rt(const uint8_t * __restrict__ W, int row0, int nrows,
                                                  const _Float16 * __restrict__ x16, float * __restrict__ out, int lane,
                                                  int Kr = K, float * res = nullptr) {
    static_assert(RW % 2 == 0, "own_rows_q2_0_1rt: RW even (the halves of a wave split the rows at K <= 512)");
    constexpr int H = RW / 2;
    const int NUNITS = (K ? K : Kr) / 32;
    const size_t RB = (size_t)(NUNITS / 2) * 18;
    const bool split = NUNITS <= 16;                          // (wave-uniform: from K alone)
    const int LG = split ? 16 : 32, gl = split ? (lane & 15) : lane;
    const int rb0 = split ? row0 + H * (lane >> 4) : row0;   // this lane's first row
    const int NU = (NUNITS + LG - 1) / LG;
    float acc[RW];
#pragma unroll
    for (int r = 0; r < RW; r++) acc[r] = 0.0f;
#pragma unroll UNR
    for (int i0 = 0; i0 < NU; i0 += UP) {
        u32x2_t q[RW][UP];
        uint16_t dh[RW][UP];
        u32x4_t xv[UP][4];
#pragma unroll
        for (int i = 0; i < UP; i++) {                       // every load of the pass, unconditionally
            const int u = gl + LG * (i0 + i), uu = (i0 + i < NU && u < NUNITS) ? u : 0;
            const int b = uu >> 1, h = uu & 1;
#pragma unroll
            for (int r = 0; r < RW; r++) {
                if (split && r >= H) continue;
                const uint8_t * blk = W + (size_t)min(rb0 + r, nrows - 1) * RB + (size_t)b * 18;
                dh[r][i] = *(const uint16_t *)blk;
                __builtin_memcpy(&q[r][i], blk + 2 + 8 * h, 8);
            }
#pragma unroll
            for (int k = 0; k < 4; k++) xv[i][k] = *(const u32x4_t *)(x16 + 32 * uu + 8 * k);
        }
#pragma unroll
        for (int i = 0; i < UP; i++) {
            const int u = gl + LG * (i0 + i);
            if (i0 + i >= NU || u >= NUNITS) continue;
            if (UNR == 1) __builtin_amdgcn_sched_barrier(0);
            const uint32_t xw[16] = {xv[i][0].x, xv[i][0].y, xv[i][0].z, xv[i][0].w, xv[i][1].x, xv[i][1].y, xv[i][1].z, xv[i][1].w,
                                     xv[i][2].x, xv[i][2].y, xv[i][2].z, xv[i][2].w, xv[i][3].x, xv[i][3].y, xv[i][3].z, xv[i][3].w};
            uint32_t xp[2][8];
            q2_0_xpairs(xw, xp[0]); q2_0_xpairs(xw + 8, xp[1]);
#pragma unroll
            for (int r = 0; r < RW; r++) {
                if (split && r >= H) continue;
                const uint32_t qw[2] = {q[r][i].x, q[r][i].y};
                float s[2];
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    if (UNR == 1) __builtin_amdgcn_sched_barrier(0);
                    h2_t w[8];
                    q2_0_pairs(qw[c], w);
                    float a = 0.0f;
#pragma unroll
                    for (int k = 0; k < 8; k++) a = __builtin_amdgcn_fdot2(w[k], as_h2(xp[c][k]), a, false);
                    s[c] = a;
                }
                acc[r] = __builtin_fmaf((float)__builtin_bit_cast(_Float16, dh[r][i]), s[0] + s[1], acc[r]);
            }
        }
    }
    float sum[RW];
#pragma unroll
    for (int r = 0; r < RW; r++) {
        if (split && r >= H) { sum[r] = 0.0f; continue; }
        float v = acc[r];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            if (split && o == 16) continue;                   // (each half sums its own rows)
            v += __shfl_xor(v, o, 32);
        }
        sum[r] = v;
    }
#pragma unroll
    for (int r = 0; r < RW; r++) {
        // split: row r's sum is in the half r / H, at its local row r % H (every lane of that half holds it)
        const float s = split ? __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, sum[r % H]), 16 * (r / H)))
                              : sum[r];
        if (res) res[r] = s;
        else if (lane == 0 && row0 + r < nrows) out[row0 + r] = s;
    }
}
