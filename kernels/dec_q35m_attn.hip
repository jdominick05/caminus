// dec_q35m_attn: the qwen35moe (Qwen3.6-35B-A3B) decode attention as its own kernel, outside the K3 megakernel: one launch
// per attention layer, T = 1, 16 query heads over 2 kv heads, head dim 256, the F16 or the Q8_0 cache. The inputs and the
// output are the split pair's (kernels/ops_attn_split.h OP_ATTN_PART + OP_ATTN_COMB, caminus.engine.k3 attn_dec): q fp32
// [16][256] as the tape's qk_nr leaves it, the cache rows 0 .. pos (pos: the int32 device word the tape's ops read), per
// position [K | V] in kv_get's layout (F16: 512 halves each; Q8: 16 blocks {fp16 d, int8 qs[32]} each), out fp32 [16][256].
// The Q8 tape's Hadamard rotations stay outside (q / k over 256 and v in blocks of 64 before, the output back after).
// Not bitwise the pair (another summation order and WMMA products): tests/engine/test_dec_q35m_attn.py holds its bounds.
//
// The design is rocKE's gfx11 split-KV decode (ROCm/rocm-libraries, dnn-providers/hip-kernel-provider/rocke,
// library/kernels/gfx1151/wmma_fmha_decode.py at 9e91949; MIT License, Copyright (c) Advanced Micro Devices, Inc.:
// THIRD_PARTY.md), written here from its description; no rocKE code:
//   - split-KV with the splits interleaved: split s of NS takes every NS-th run of DQ_NW 16-key tiles, so the splits of
//     a head sweep the keys as one DRAM front (not NS separate streams)
//   - S^T = K Q^T per 16-key tile and wave: K as the WMMA A operand (row = key), the query rows as B, so a lane holds
//     one query column's scores and the softmax is in-lane plus one lane ^ 16 exchange; O^T = V^T P with V transposed
//     through a wave-private LDS region; Q in LDS; the head's 256 dims of O split over 2 workgroups (dv halves)
//   - the splits' partials combined through a workspace by the last workgroup of each (kv head, dv half) to finish
// What is Caminus's own (the 35B's 8 query heads fill half of WMMA's 16 columns): the other 8 columns carry the low
// half of an fp16 hi + lo split, of Q (scores) and of P (sums), each lo scaled by 2^11 into fp16's normal range, so the
// products are fp32-grade on an F16 cache (fp16 K / V values are exact operands) instead of fp16-rounded Q and P.
// Q8: each value decoded as fp16(code * d) (kv_get's value, one rounding: half an fp16 ulp; pf_q3m's q3p_attn_q8
// does the same), the code turned into fp16 exactly by the 0x6400 bias.
//
// Launch: grid NS * 4 workgroups (workgroup b: dv half b & 1, kv head (b >> 1) & 1, split b >> 2: a split's four
// workgroups are adjacent, so the second dv half's K tile reads hit the first's L2 lines), DQ_NW * 32 threads, static
// LDS only. ws: DQ_CNT int32 counters (zero before the first launch; the last workgroup re-arms its own), then the
// records [4][NS][8][DQ_RS] fp32 (caminus.engine.dec_attn.workspace_floats). err (nullable): set to 1 when pos + 1 is
// outside [0, rows] (the cells are clamped to [0, rows]: no read past the cache's rows).
// Bounds (every load and store below carries its own): a K / V read is of position key <= n - 1 <= rows - 1 and lies
// inside that position's PB bytes, so the last byte read is < rows * PB, the cache's end (tests/engine/test_dec_q35m_attn.py
// runs n = rows on a cache whose last byte ends its allocation); q / out stay in [16][256]; the records and counters in
// workspace_floats(ns) = DQ_CNT + 4 ns 8 DQ_RS floats (the host checks the buffer holds that many).
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef _Float16 h2_t __attribute__((ext_vector_type(2)));
typedef _Float16 h16_t __attribute__((ext_vector_type(16)));
typedef float f8_t __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));

#define DQ_D 256              // head dim
#define DQ_G 8                // query heads per kv head
#define DQ_NKV 2              // kv heads
#ifndef DQ_NW
#define DQ_NW 4               // waves per workgroup
#endif
#define DQ_DVT 128            // dims of O per workgroup (the dv half)
#define DQ_NDC (DQ_DVT / 16)  // O^T tiles per wave
#define DQ_QP 264             // halves per Q column in LDS (256 + 8: the 16 columns' rows in different banks)
#define DQ_VP 12              // dwords per V^T row in LDS (8 key pairs + 4: rows in different banks)
#define DQ_RS 132             // floats per split record: m, l, 0, 0, O[128]
#define DQ_CNT 64             // int32 words before the records (counters [4] used)
#define DQ_LO 2048.0f         // the lo halves' scale (2^11)

struct DqU2 { u32x4_t a, b; };
__device__ __forceinline__ h16_t dq_h16(u32x4_t a, u32x4_t b) { return __builtin_bit_cast(h16_t, (DqU2){a, b}); }
__device__ __forceinline__ h2_t dq_h2(uint32_t u) { return __builtin_bit_cast(h2_t, u); }
__device__ __forceinline__ uint32_t dq_u(h2_t h) { return __builtin_bit_cast(uint32_t, h); }
__device__ __forceinline__ f8_t dq_wmma(const h16_t & a, const h16_t & b, const f8_t & c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
// lane ^ 16 (permlanex16) and lane ^ 8 within rows of 16 (DPP row_xmask:8)
__device__ __forceinline__ uint32_t dq_x16u(uint32_t u) {
    return __builtin_amdgcn_permlanex16(u, u, 0x76543210u, 0xfedcba98u, false, false);
}
__device__ __forceinline__ float dq_x16(float x) { return __builtin_bit_cast(float, dq_x16u(__builtin_bit_cast(uint32_t, x))); }
__device__ __forceinline__ float dq_x8(float x) {
    const int xi = __builtin_bit_cast(int, x);
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(xi, xi, 0x168, 0xf, 0xf, false));
}
// two Q8 codes to fp16(code * d): the bytes the selector picks (into bytes 0 and 2, zeros above) biased to fp16
// 1024 + (code ^ 0x80) = 1152 + code (exact), less 1152 (exact), times d2 = (d, d') (one rounding each)
__device__ __forceinline__ uint32_t dq_q8x2(uint32_t hi, uint32_t lo, uint32_t sel, uint32_t d2) {
    const h2_t x = dq_h2(__builtin_amdgcn_perm(hi, lo, sel) ^ 0x64806480u) - (h2_t){(_Float16)1152.0f, (_Float16)1152.0f};
    return dq_u(x * dq_h2(d2));
}
__device__ __forceinline__ constexpr uint32_t dq_sel2(int b) {   // bytes b, b + 1 of the low source dword into halves 0, 1
    return 0x0c000c00u | (uint32_t)(b + 1) << 16 | (uint32_t)b;
}
__device__ __forceinline__ constexpr uint32_t dq_seld(int b) {   // the fp16 at byte b of the low source, in both halves
    return (uint32_t)(b + 1) << 24 | (uint32_t)b << 16 | (uint32_t)(b + 1) << 8 | (uint32_t)b;
}

__shared__ u32x4_t DQ_QL[16 * DQ_QP / 8];                 // Q^T B rows: [16 columns][DQ_QP halves]
__shared__ u32x4_t DQ_VT[DQ_NW * DQ_DVT * DQ_VP / 4];     // per wave V^T [128 rows][DQ_VP dwords]; then the wave combine
__shared__ float DQ_ML[2 * DQ_NW * 32];                   // the wave combine's m [NW][32], then l [NW][32]
__shared__ int DQ_LAST;

template <bool Q8>
__device__ __forceinline__ void dq_attn(const float * __restrict__ q, const uint8_t * __restrict__ cache, const int * __restrict__ pos,
                                        int rows, float scale, int ns, float * __restrict__ ws, float * __restrict__ out,
                                        int * __restrict__ err) {
    constexpr int PB = Q8 ? 2 * (DQ_NKV * DQ_D / 32) * 34 : 4 * DQ_NKV * DQ_D;   // bytes per position (kv_pos_bytes)
    constexpr int KH = Q8 ? 8 * 34 : DQ_D * 2;                                  // bytes per kv head's K (or V) row
    constexpr int VO = Q8 ? DQ_NKV * KH : DQ_NKV * DQ_D * 2;                     // V's offset in a position
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, c = lane & 15, hf = lane >> 4;
    const int dvi = blockIdx.x & 1, g = (blockIdx.x >> 1) & 1, s = blockIdx.x >> 2, grp = g * 2 + dvi;
    const int P = *pos;                                           // (pos[0]: the host checks pos holds 1 int32)
    if (tid == 0 && err && (P < -1 || P >= rows)) *err = 1;
    const int n = min(max(P + 1, 0), rows);                       // cells 0 .. n - 1
    const int nruns = (n + 16 * DQ_NW - 1) / (16 * DQ_NW);        // runs of DQ_NW tiles holding a cell
    int * cnt = (int *)ws;
    float * rec = ws + DQ_CNT;
    if (s < nruns) {
        // Q^T into LDS: column cc < 8 = head 8 g + cc's hi half, cc >= 8 = head 8 g + cc - 8's lo half (x 2^11), of
        // q * scale * log2(e) (the scores in base 2). WMMA step j's 16 values: dims 8 j .. 8 j + 7, then 128 + 8 j ..
        // 128 + 8 j + 7 (K's A operand takes the same order: a lane's two 16-byte loads per step)
        {
            const float sl = scale * 1.4426950408889634f;
            _Float16 * ql = (_Float16 *)DQ_QL;
            for (int i = tid; i < 16 * DQ_D; i += DQ_NW * 32) {
                const int cc = i >> 8, j = (i >> 4) & 15, kk = i & 15;
                const int dim = kk < 8 ? 8 * j + kk : 128 + 8 * j + kk - 8;
                // q[h][dim]: h = 8 g + (cc & 7) <= 15, dim <= 128 + 8 * 15 + 7 = 255: inside [16][256]
                const float x = q[(size_t)(g * DQ_G + (cc & 7)) * DQ_D + dim] * sl;
                const _Float16 hi = (_Float16)x;
                ql[cc * DQ_QP + 16 * j + kk] = cc < 8 ? hi : (_Float16)((x - (float)hi) * DQ_LO);
            }
        }
        __syncthreads();
        f8_t O[DQ_NDC];
#pragma unroll
        for (int d = 0; d < DQ_NDC; d++) O[d] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
        float m = -INFINITY, l = 0.0f;
        const u32x4_t * qb = DQ_QL + c * (DQ_QP / 8);
        u32x4_t * vt = DQ_VT + wave * (DQ_DVT * DQ_VP / 4);
        const int h8 = lane & 1, vcg = lane >> 1;                   // V staging: keys 8 h8 .. 8 h8 + 7, dims 8 vcg .. + 7
        const uint8_t * kbase = cache + (size_t)g * KH;
        const uint8_t * vbase = cache + VO + (size_t)g * KH;
#pragma unroll 1
        for (int r = s; ; r += ns) {
            const int k0 = (r * DQ_NW + wave) * 16;
            if (k0 >= n) break;
            // ---- scores S^T (16 keys x 16 columns): lane = key k0 + c (the A row, both halves), column c (B)
            f8_t S = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
            {
                const int key = min(k0 + c, n - 1);                     // (keys past the end: a written row, masked)
                const uint8_t * kr = kbase + (size_t)key * PB;
                if constexpr (!Q8) {
                    // own-half loads (rocKE's gfx11 form): lane half hf loads step j's dims 8 j .. (hf 0) or 128 + 8 j ..
                    // (hf 1); lane ^ 16 (the same key) brings the other 8, so both halves hold the step's 16 values
                    const u32x4_t * kp = (const u32x4_t *)kr + 16 * hf;
                    u32x4_t ko[16];
#pragma unroll
                    for (int j = 0; j < 16; j++) ko[j] = kp[j];   // bytes g 512 + 256 hf + [0, 256): <= 1024 of PB 2048
#pragma unroll
                    for (int j = 0; j < 16; j++) {
                        const u32x4_t pr = {dq_x16u(ko[j].x), dq_x16u(ko[j].y), dq_x16u(ko[j].z), dq_x16u(ko[j].w)};
                        S = dq_wmma(hf ? dq_h16(pr, ko[j]) : dq_h16(ko[j], pr), dq_h16(qb[2 * j], qb[2 * j + 1]), S);
                    }
                } else {
                    uint32_t w[68];                                     // the key's 272-byte row (8 blocks)
#pragma unroll
                    for (int i = 0; i < 17; i++) {
                        const u32x4_t x = ((const u32x4_t *)kr)[i];   // bytes g 272 + [0, 272): <= 544 of PB 1088
                        w[4 * i] = x.x; w[4 * i + 1] = x.y; w[4 * i + 2] = x.z; w[4 * i + 3] = x.w;
                    }
#pragma unroll
                    for (int j = 0; j < 16; j++) {
                        uint32_t a[8];
#pragma unroll
                        for (int hh = 0; hh < 2; hh++) {
                            const int bk = 4 * hh + (j >> 2), b0 = 34 * bk;                 // block, its byte offset
                            const uint32_t d2 = __builtin_amdgcn_perm(0u, w[b0 >> 2], dq_seld(b0 & 3));
#pragma unroll
                            for (int i = 0; i < 4; i++) {
                                const int bb = b0 + 2 + 8 * (j & 3) + 2 * i;                 // the codes' bytes (even)
                                a[4 * hh + i] = dq_q8x2(0u, w[bb >> 2], dq_sel2(bb & 3), d2);
                            }
                        }
                        const DqU2 au = {(u32x4_t){a[0], a[1], a[2], a[3]}, (u32x4_t){a[4], a[5], a[6], a[7]}};
                        S = dq_wmma(__builtin_bit_cast(h16_t, au), dq_h16(qb[2 * j], qb[2 * j + 1]), S);
                    }
                }
            }
            // ---- online softmax (base 2): S[v] = key k0 + 2 v + hf, column c; hi + lo of head c & 7 in lanes c, c ^ 8
            float sv[8], tmax = -INFINITY;
#pragma unroll
            for (int v = 0; v < 8; v++) {
                const float x = S[v], y = dq_x8(x);
                const float sc = c < 8 ? __builtin_fmaf(y, 1.0f / DQ_LO, x) : __builtin_fmaf(x, 1.0f / DQ_LO, y);
                sv[v] = k0 + 2 * v + hf < n ? sc : -INFINITY;
                tmax = fmaxf(tmax, sv[v]);
            }
            tmax = fmaxf(tmax, dq_x16(tmax));
            const float mn = fmaxf(m, tmax), ms = mn == -INFINITY ? 0.0f : mn;
            const float alpha = exp2f(m - ms);
            uint32_t pw[4];                                             // B column c: p's hi (c < 8) or lo x 2^11 (c >= 8)
            float ps = 0.0f;
#pragma unroll
            for (int v = 0; v < 8; v += 2) {
                _Float16 e[2];
#pragma unroll
                for (int u = 0; u < 2; u++) {
                    const float p = exp2f(sv[v + u] - ms);
                    ps += p;
                    const _Float16 ph = (_Float16)p;
                    e[u] = c < 8 ? ph : (_Float16)((p - (float)ph) * DQ_LO);
                }
                pw[v >> 1] = dq_u((h2_t){e[0], e[1]});
            }
            l = l * alpha + ps; m = mn;
            if (__builtin_amdgcn_ballot_w32(alpha != 1.0f)) {
#pragma unroll
                for (int d = 0; d < DQ_NDC; d++)
#pragma unroll
                    for (int v = 0; v < 8; v++) O[d][v] *= alpha;
            }
            // P's B operand: keys k0 .. k0 + 15 of column c: own (2 v + hf) and lane ^ 16's (2 v + 1 - hf)
            uint32_t po[8];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const uint32_t mine = pw[i], other = dq_x16u(mine);      // mine: keys 4 i + hf, 4 i + 2 + hf
                const uint32_t ev = hf ? other : mine, od = hf ? mine : other;
                po[2 * i] = __builtin_amdgcn_perm(od, ev, 0x05040100u);  // keys 4 i, 4 i + 1
                po[2 * i + 1] = __builtin_amdgcn_perm(od, ev, 0x07060302u);   // keys 4 i + 2, 4 i + 3
            }
            const h16_t PB_ = __builtin_bit_cast(h16_t, po);
            // ---- V^T through the wave's LDS: lane (h8, vcg) stages keys k0 + 8 h8 + kk, dims dv0 + 8 vcg + e into row
            // 16 e + vcg, key pair dword 4 h8 + kk / 2 (each write instruction covers 16 consecutive rows); tile dc's A row
            // i is row 16 dc + i = dim 8 i + dc of the half
            {
                uint32_t hv[8][4];
#pragma unroll
                for (int kk = 0; kk < 8; kk++) {
                    const int key = min(k0 + 8 * h8 + kk, n - 1);
                    const uint8_t * vr = vbase + (size_t)key * PB;
                    if constexpr (!Q8) {
                        // bytes 1024 + g 512 + 2 (128 dvi + 8 vcg) + [0, 16): <= 1024 + 512 + 2 (128 + 120) + 16 = 2048 = PB
                        const u32x4_t x = *(const u32x4_t *)(vr + (size_t)(dvi * DQ_DVT + 8 * vcg) * 2);
                        hv[kk][0] = x.x; hv[kk][1] = x.y; hv[kk][2] = x.z; hv[kk][3] = x.w;
                    } else {
                        // block 4 dvi + vcg / 4 of the kv head's V row; codes 8 (vcg & 3) .. + 7. Every load stays
                        // inside the head's 272-byte row: the third dword is the one after the codes only when they
                        // start 2 bytes into a dword (an even block: it ends <= 36 bytes into the block, <= 240 of 272);
                        // from an aligned start (odd block) it rereads the second (unused): the dword after the last
                        // block's codes lies past the row (past the cache's last position for kv head 1: that over-read
                        // page-faulted on an exactly sized cache)
                        const int b0 = 34 * (4 * dvi + (vcg >> 2)), qo = b0 + 2 + 8 * (vcg & 3), qa = qo & ~3;
                        const uint32_t sh = qo & 3;
                        const uint32_t * wp = (const uint32_t *)(vr + qa);
                        // bytes 544 + g 272 + qa + [0, 8) (sh 0: odd block, qa <= 238 + 26 = 264: <= 272) or + [0, 12)
                        // (sh 2: even block, qa <= 204 + 24 = 228: <= 240); d: (b0 & ~3) + [0, 4) <= 236 + 4. So <= 1088 = PB
                        const uint32_t w0 = wp[0], w1 = wp[1], w2 = wp[sh ? 2 : 1];
                        const uint32_t dw = *(const uint32_t *)(vr + (b0 & ~3));
                        hv[kk][0] = __builtin_amdgcn_alignbyte(w1, w0, sh);    // codes 0 .. 3
                        hv[kk][1] = __builtin_amdgcn_alignbyte(w2, w1, sh);    // codes 4 .. 7
                        hv[kk][2] = (b0 & 2) ? dw >> 16 : dw & 0xffffu;        // d
                        hv[kk][3] = 0;
                    }
                }
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    uint32_t o4[4];
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        if constexpr (!Q8) {
                            o4[i] = __builtin_amdgcn_perm(hv[2 * i + 1][e >> 1], hv[2 * i][e >> 1], (e & 1) ? 0x07060302u : 0x05040100u);
                        } else {
                            const uint32_t d2 = hv[2 * i][2] | hv[2 * i + 1][2] << 16;
                            const uint32_t sel = 0x0c000c00u | (uint32_t)(4 + (e & 3)) << 16 | (uint32_t)(e & 3);
                            o4[i] = dq_q8x2(hv[2 * i + 1][e >> 2], hv[2 * i][e >> 2], sel, d2);
                        }
                    }
                    vt[((16 * e + vcg) * DQ_VP) / 4 + h8] = (u32x4_t){o4[0], o4[1], o4[2], o4[3]};
                }
            }
#pragma unroll
            for (int d = 0; d < DQ_NDC; d++) {
                const u32x4_t * ap = vt + ((16 * d + c) * DQ_VP) / 4;   // (LDS: row <= 127, dwords [0, 8) of DQ_VP)
                O[d] = dq_wmma(dq_h16(ap[0], ap[1]), PB_, O[d]);
            }
        }
        // ---- the wave combine: each lane's columns c and c ^ 8 (hi + lo / 2^11) -> head c & 7; lanes c < 8 hold the
        // result: tile d, slot v = dim 8 (2 v + hf) + d of the half. Waves 1 .. NW - 1 through LDS, wave 0 sums
        l = l + dq_x16(l);                                             // (lanes c and c ^ 16 hold the tile's other keys)
#pragma unroll
        for (int d = 0; d < DQ_NDC; d++)
#pragma unroll
            for (int v = 0; v < 8; v++) {
                const float x = O[d][v], y = dq_x8(x);
                O[d][v] = __builtin_fmaf(y, 1.0f / DQ_LO, x);
            }
        __syncthreads();                                               // (every wave's V^T reads done: the region is reused)
        float * cb = (float *)DQ_VT;                                   // [NW - 1][64][32]: wave w - 1, value i, lane
        if (wave > 0) {
#pragma unroll
            for (int d = 0; d < DQ_NDC; d++)
#pragma unroll
                for (int v = 0; v < 8; v++) cb[((wave - 1) * 64 + d * 8 + v) * 32 + lane] = O[d][v];
            DQ_ML[wave * 32 + lane] = m; DQ_ML[(DQ_NW + wave) * 32 + lane] = l;
        }
        __syncthreads();
        if (wave == 0) {
            float M = m;
#pragma unroll
            for (int w = 1; w < DQ_NW; w++) M = fmaxf(M, DQ_ML[w * 32 + lane]);
            const float Ms = M == -INFINITY ? 0.0f : M;
            const float f0 = exp2f(m - Ms);
            float L = l * f0;
#pragma unroll
            for (int d = 0; d < DQ_NDC; d++)
#pragma unroll
                for (int v = 0; v < 8; v++) O[d][v] *= f0;
#pragma unroll 1
            for (int w = 1; w < DQ_NW; w++) {
                const float f = exp2f(DQ_ML[w * 32 + lane] - Ms);
                L = __builtin_fmaf(DQ_ML[(DQ_NW + w) * 32 + lane], f, L);
#pragma unroll
                for (int d = 0; d < DQ_NDC; d++)
#pragma unroll
                    for (int v = 0; v < 8; v++) O[d][v] = __builtin_fmaf(cb[((w - 1) * 64 + d * 8 + v) * 32 + lane], f, O[d][v]);
            }
            if (c < 8) {   // the split's record of head c: m, l, then O over the half's 128 dims
                // record (grp, s, c): floats DQ_CNT + ((grp ns + s) 8 + c) DQ_RS + [0, 4 + 8 (2 * 7 + 1) + 8 = 132):
                // grp <= 3, s <= ns - 1, c <= 7, so < DQ_CNT + 4 ns 8 DQ_RS = workspace_floats(ns)
                float * rp = rec + ((size_t)(grp * ns + s) * DQ_G + c) * DQ_RS;
                if (hf == 0) { rp[0] = M; rp[1] = L; }
#pragma unroll
                for (int v = 0; v < 8; v++) {
                    float * o8 = rp + 4 + 8 * (2 * v + hf);
                    *(float4 *)o8 = make_float4(O[0][v], O[1][v], O[2][v], O[3][v]);
                    *(float4 *)(o8 + 4) = make_float4(O[4][v], O[5][v], O[6][v], O[7][v]);
                }
            }
        }
    }
    // ---- the last of the group's NS workgroups to finish merges the splits' records (a workgroup without cells too)
    __threadfence();
    __syncthreads();
    if (tid == 0) DQ_LAST = atomicAdd(cnt + grp, 1) == ns - 1;       // (counter grp <= 3 of DQ_CNT)
    __syncthreads();
    if (!DQ_LAST) return;
    __threadfence();
    if (tid == 0) cnt[grp] = 0;                                         // (re-armed for the next launch)
    const int nu = min(ns, nruns);
    for (int o = tid; o < DQ_G * DQ_DVT / 8; o += DQ_NW * 32) {
        const int hc = o >> 4, dv8 = (o & 15) * 8;
        // the records (grp, u, hc) for u < nu <= ns: floats [0, 2) and 4 + dv8 + [0, 8) <= 4 + 120 + 8 = 132 of each, all
        // inside the ones the group's workgroups wrote (u < nruns: every such split wrote all 8 heads' records)
        const float * rp = rec + ((size_t)grp * ns * DQ_G + hc) * DQ_RS;
        float M = -INFINITY;
        for (int u = 0; u < nu; u++) M = fmaxf(M, rp[(size_t)u * DQ_G * DQ_RS]);
        float L = 0.0f, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 4
        for (int u = 0; u < nu; u++) {
            const float * ru = rp + (size_t)u * DQ_G * DQ_RS;
            const float f = exp2f(ru[0] - M);
            const float4 a = *(const float4 *)(ru + 4 + dv8), b = *(const float4 *)(ru + 8 + dv8);
            L = __builtin_fmaf(ru[1], f, L);
            acc[0] = __builtin_fmaf(a.x, f, acc[0]); acc[1] = __builtin_fmaf(a.y, f, acc[1]);
            acc[2] = __builtin_fmaf(a.z, f, acc[2]); acc[3] = __builtin_fmaf(a.w, f, acc[3]);
            acc[4] = __builtin_fmaf(b.x, f, acc[4]); acc[5] = __builtin_fmaf(b.y, f, acc[5]);
            acc[6] = __builtin_fmaf(b.z, f, acc[6]); acc[7] = __builtin_fmaf(b.w, f, acc[7]);
        }
        const float il = nu > 0 ? 1.0f / L : 0.0f;                     // (no cell: zeros, the split pair's empty list)
        float * op = out + (size_t)(g * DQ_G + hc) * DQ_D + dvi * DQ_DVT + dv8;   // head <= 15, dims <= 128 + 120 + 7
        *(float4 *)op = make_float4(acc[0] * il, acc[1] * il, acc[2] * il, acc[3] * il);
        *(float4 *)(op + 4) = make_float4(acc[4] * il, acc[5] * il, acc[6] * il, acc[7] * il);
    }
}

extern "C" __global__ void __launch_bounds__(DQ_NW * 32) dec_q35m_attn_f16(const float * __restrict__ q, const uint8_t * __restrict__ cache,
        const int * __restrict__ pos, int rows, float scale, int ns, float * __restrict__ ws, float * __restrict__ out, int * __restrict__ err) {
    dq_attn<false>(q, cache, pos, rows, scale, ns, ws, out, err);
}
extern "C" __global__ void __launch_bounds__(DQ_NW * 32) dec_q35m_attn_q8(const float * __restrict__ q, const uint8_t * __restrict__ cache,
        const int * __restrict__ pos, int rows, float scale, int ns, float * __restrict__ ws, float * __restrict__ out, int * __restrict__ err) {
    dq_attn<true>(q, cache, pos, rows, scale, ns, ws, out, err);
}
