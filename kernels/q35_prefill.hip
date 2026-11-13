// The batched prefill of the dense qwen35 line (Qwen3.8-27B finetunes, Swift 1.5 Q4_K_M first) on Caminus's own kernels:
// no ggml-cuda code, written from the GGUF block layouts. This object holds q3m_prefill.hip's kernels (its Q4_K / Q6_K
// GEMMs and glue: included as they are, so that file's own object does not change) and the two formats Swift adds:
//
// ---- q35p_mm_q8_0 / q35p_mm_q5k: Y[ydst[i]][r] = X[xsrc[i]] . W[r] in fp32, q3p_mm_q4k's interface and tiling --------------
// The decode engine's raw GGUF blocks are read in place. Activations fp16, products and sums fp32 through RDNA3's WMMA,
// the weights EXACT in the B operand:
//   Q8_0  w = d q (32 values a block, 34 B: d then 32 int8): B = q (|q| <= 128: exact in fp16) over 16 values, the
//         block's two k-steps into P, then acc += d * P
//   Q5_K  w = d sc q - dmin m (256 values a block, 176 B: d, dmin, the 12 scale bytes as Q4_K's, qh[32], qs[128]; q =
//         the nibble | bit s of qh[l] << 4 for value l of sub-block s): B = sc * q (<= 63 x 31 = 1953: exact), then Q4_K's
//         acc += d * P - dmin * MS with the same sub-block sums and the same rare overflow branch
// so a product's only rounding is the fp32 accumulation, as in q3p_mm_q4k / q3p_mm_q6k.
#include "q3m_prefill.hip"

// ---- Q8_0 ------------------------------------------------------------------------------------------------------------
// 16 int8 (2-byte aligned: a block starts at an even offset) -> fp16 q, exact: (q ^ 0x80) is q + 128 as an unsigned byte,
// fp16(1024 + (q + 128)) - 1152 = q
__device__ __forceinline__ h16_t q8_b(const uint8_t * p, bool ok) {
    const uint16_t * h = (const uint16_t *)p;
    const h2_t o2 = {(_Float16)(-1152.0f), (_Float16)(-1152.0f)};
    uint32_t o[8];
#pragma unroll
    for (int w = 0; w < 4; w++) {
        const uint32_t u = ok ? (((uint32_t)h[2 * w] | ((uint32_t)h[2 * w + 1] << 16)) ^ 0x80808080u) : 0x80808080u;
        const uint32_t p01 = __builtin_amdgcn_perm(0u, u, 0x0c010c00u) | 0x64006400u;
        const uint32_t p23 = __builtin_amdgcn_perm(0u, u, 0x0c030c02u) | 0x64006400u;
        o[2 * w] = as_u(as_h2(p01) + o2); o[2 * w + 1] = as_u(as_h2(p23) + o2);
    }
    return __builtin_bit_cast(h16_t, o);
}

// The weights two blocks at a time: a block pair is 68 bytes at a 4-byte aligned offset (rows are 272-byte multiples:
// K / 32 blocks of 34 B, K a multiple of 256), so it is 17 dword loads (one block alone was 33 two-byte loads), and the
// next pair's loads are issued before this pair's WMMAs (one pair ahead, across the stage barriers too). The B operand
// is q8_b's bytes and arithmetic from registers (q8_wb), so every sum keeps its bits (the per-block order, the WMMA
// chain, the fma with d: unchanged). Before: the loads of each block waited right before its WMMAs (Q8_0 37.5 TF at the
// Swift shapes, 54.8 with the weight path removed: benchmarks/q35_prefill_gemm.py --module q35_prefill_abl1).
struct Q8Pair { uint32_t w[17]; };
__device__ __forceinline__ void q8_pair_load(Q8Pair & P, const uint8_t * wrow, int pair) {
    const uint32_t * src = (const uint32_t *)(wrow + (size_t)pair * 68);
#pragma unroll
    for (int i = 0; i < 17; i++) P.w[i] = src[i];
}
// 4 dwords of the pair's bytes starting at byte o (o % 2 == 0, a compile-time constant)
template <int O>
__device__ __forceinline__ void q8_pair_bytes(const Q8Pair & P, uint32_t (&q)[4]) {
#pragma unroll
    for (int i = 0; i < 4; i++)
        q[i] = (O & 3) ? __builtin_amdgcn_alignbyte(P.w[O / 4 + i + 1], P.w[O / 4 + i], O & 3) : P.w[O / 4 + i];
}
// q8_b on 16 int8 held in 4 dwords (the same per-word statements)
__device__ __forceinline__ h16_t q8_wb(const uint32_t (&q)[4], bool ok) {
    const h2_t o2 = {(_Float16)(-1152.0f), (_Float16)(-1152.0f)};
    uint32_t o[8];
#pragma unroll
    for (int w = 0; w < 4; w++) {
        const uint32_t u = ok ? (q[w] ^ 0x80808080u) : 0x80808080u;
        const uint32_t p01 = __builtin_amdgcn_perm(0u, u, 0x0c010c00u) | 0x64006400u;
        const uint32_t p23 = __builtin_amdgcn_perm(0u, u, 0x0c030c02u) | 0x64006400u;
        o[2 * w] = as_u(as_h2(p01) + o2); o[2 * w + 1] = as_u(as_h2(p23) + o2);
    }
    return __builtin_bit_cast(h16_t, o);
}
// block jb (0: the pair's first, 1: its second) of a pair: P = its two WMMA k-steps, then acc += d P (q35p_mm_q8_0's order)
template <int JB>
__device__ __forceinline__ void q8_pair_block(const Q8Pair & PR, int j, bool rok, int lane, f8_t (&acc)[4]) {
    const float d = rok ? (float)__builtin_bit_cast(_Float16, (uint16_t)(JB ? (PR.w[8] >> 16) : (PR.w[0] & 0xffffu))) : 0.0f;
    f8_t P[4];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) P[mt] = Q3P_ABL == 3 ? acc[mt] : (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int hf = 0; hf < 2; hf++) {
        uint32_t q[4];
        if (JB == 0 && hf == 0) q8_pair_bytes<2>(PR, q);
        else if (JB == 0) q8_pair_bytes<18>(PR, q);
        else if (hf == 0) q8_pair_bytes<36>(PR, q);
        else q8_pair_bytes<52>(PR, q);
#if Q3P_ABL == 1
        const h16_t B = (h16_t)(_Float16)1.0f; (void)q;
#else
        const h16_t B = q8_wb(q, rok);
#endif
#pragma unroll
        for (int mt = 0; mt < 4; mt++) P[mt] = wmma(mm_a(mt, j * 32 + hf * 16, lane), B, P[mt]);
    }
#if Q3P_ABL == 3
#pragma unroll
    for (int mt = 0; mt < 4; mt++) acc[mt] = P[mt];
    (void)d;
#else
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int v = 0; v < 8; v++) acc[mt][v] = __builtin_fmaf(d, P[mt][v], acc[mt][v]);
#endif
}

extern "C" __global__ void __launch_bounds__(256) q35p_mm_q8_0(
        const uint8_t * __restrict__ W, int R, int K, const _Float16 * __restrict__ X, int ldx, const int * __restrict__ xsrc,
        float * __restrict__ Y, int ldy, const int * __restrict__ ydst, const MMItem * __restrict__ items, const int * __restrict__ nitems,
        int ntok) {
    MMItem it;
    if (items) { if ((int)blockIdx.x >= *nitems) return; it = items[blockIdx.x]; }
    else { it.e = 0; it.off = blockIdx.x * MM_M; it.cnt = min(MM_M, ntok - it.off); }
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int r = blockIdx.y * MM_N + wave * 16 + (lane & 15);
    const bool rok = r < R;
    const int NB = K / 256;                       // (staging steps of 256 values: 8 Q8_0 blocks = 4 pairs each)
    const size_t RB = (size_t)(K / 32) * 34;
    const uint8_t * wrow = W + ((size_t)it.e * R + (rok ? r : 0)) * RB;
    f8_t acc[4];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) acc[mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
    const int NP = NB * 4;
    Q8Pair cur, nxt;
    q8_pair_load(cur, wrow, 0);
    for (int b = 0; b < NB; b++) {
        __syncthreads();
#if Q3P_ABL != 2
        mm_stage<false>(X, ldx, xsrc, it.off, it.cnt, b, tid);
#endif
        __syncthreads();
#pragma unroll 1
        for (int jp = 0; jp < 4; jp++) {
            const int pr = b * 4 + jp;
            if (pr + 1 < NP) q8_pair_load(nxt, wrow, pr + 1);     // (the next pair in flight during this pair's WMMAs)
            q8_pair_block<0>(cur, 2 * jp, rok, lane, acc);
            __builtin_amdgcn_sched_barrier(0);
            q8_pair_block<1>(cur, 2 * jp + 1, rok, lane, acc);
            __builtin_amdgcn_sched_barrier(0);
            cur = nxt;
        }
    }
    if (!rok) return;
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int v = 0; v < 8; v++) {
            const int t = mt * 16 + 2 * v + (lane >> 4);
            if (t < it.cnt) Y[(size_t)(ydst ? ydst[it.off + t] : it.off + t) * ldy + r] = acc[mt][v];
        }
}

// q4k_b with fp16's exponent byte (0x64: 1024 + q) taken by v_perm from its first source instead of OR-ed in after it:
// the same bits, two VALU instructions fewer per four values (the wide tile's limit is this unpack: with B = 1 it runs at
// ~104 TF on the XT, q35_prefill_abl1)
__device__ __forceinline__ h16_t q4k_bw(const u32x4_t & qw, int sh, _Float16 sc) {
    const h2_t s2 = {sc, sc}, o2 = {(_Float16)(-1024.0f) * sc, (_Float16)(-1024.0f) * sc};
    const uint32_t w4[4] = {qw.x, qw.y, qw.z, qw.w};
    uint32_t o[8];
#pragma unroll
    for (int w = 0; w < 4; w++) {
        const uint32_t n = (w4[w] >> sh) & 0x0f0f0f0fu;
        const uint32_t p01 = __builtin_amdgcn_perm(0x64646464u, n, 0x04010400u);   // bytes: q0, 0x64, q1, 0x64
        const uint32_t p23 = __builtin_amdgcn_perm(0x64646464u, n, 0x04030402u);
        o[2 * w] = as_u(__builtin_elementwise_fma(as_h2(p01), s2, o2));
        o[2 * w + 1] = as_u(__builtin_elementwise_fma(as_h2(p23), s2, o2));
    }
    return __builtin_bit_cast(h16_t, o);
}

// ---- q35p_mm_q4k_w: Q4_K, the wide tile (dense rows only) ----------------------------------------------------------
// q3p_mm_q4k with two 16-row tiles per wave: a workgroup is 64 tokens x 256 rows (8 waves x 32 rows), and each A operand
// (a token tile's 16 values, from LDS) feeds the WMMAs of both row tiles. q3p_mm_q4k reads A once per WMMA; here once
// per two and stages each token tile's X once per 256 rows instead of per 128 (an experiment: q3p_mm_q4k stops near
// 52-57 TF with its weight path, staging or scale step removed, benchmarks/q35_prefill_gemm.py --module q35_prefill_abl1/2/3;
// halving the A reads is one candidate limit, measured by --arms ab). Every row's statements are
// q3p_mm_q4k's in its order (the B operand from the row's own scale, P per block in the same k-step order, the min term
// against the same sub-block sums, the same overflow branch), so each output keeps its bits: tests/prefill/
// test_q35_prefill_mm.py runs both kernels and compares them bitwise. Rows past R read row 0 and are never written.
// Grid (tokens / 64, rows / 256), block 256; X fp16 [ntok][ldx], Y fp32 [ntok][ldy].
#define MMW_N 256
extern "C" __global__ void __launch_bounds__(256) q35p_mm_q4k_w(const uint8_t * __restrict__ W, int R, int K,
                                                                const _Float16 * __restrict__ X, int ldx,
                                                                float * __restrict__ Y, int ldy, int ntok) {
    MMItem it; it.e = 0; it.off = blockIdx.x * MM_M; it.cnt = min(MM_M, ntok - it.off);
    const int * xsrc = nullptr;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int NB = K / 256;
    const size_t RB = (size_t)NB * 144;
    int r[2]; bool rok[2]; const uint8_t * wrow[2];
#pragma unroll
    for (int rt = 0; rt < 2; rt++) {
        r[rt] = blockIdx.y * MMW_N + wave * 32 + rt * 16 + (lane & 15);
        rok[rt] = r[rt] < R;
        wrow[rt] = W + (size_t)(rok[rt] ? r[rt] : 0) * RB;
    }
    f8_t acc[2][4];
#pragma unroll
    for (int rt = 0; rt < 2; rt++)
#pragma unroll
        for (int mt = 0; mt < 4; mt++) acc[rt][mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
    if (tid < MM_M) MMF[tid] = 0;
    if (tid == 0) MMBIG = 0;
    for (int b = 0; b < NB; b++) {
        u32x4_t hd[2], qs[2][8];
#pragma unroll
        for (int rt = 0; rt < 2; rt++) {
            const u32x4_t * bp = (const u32x4_t *)(wrow[rt] + (size_t)b * 144);
            hd[rt] = bp[0];
#pragma unroll
            for (int i = 0; i < 8; i++) qs[rt][i] = bp[1 + i];
        }
        __syncthreads();
        mm_stage<true>(X, ldx, xsrc, it.off, it.cnt, b, tid);
        __syncthreads();
        int sc[2][8], m[2][8];
#pragma unroll
        for (int rt = 0; rt < 2; rt++) q4k_scales(hd[rt], sc[rt], m[rt]);
        f8_t P[2][4];
#pragma unroll
        for (int rt = 0; rt < 2; rt++)
#pragma unroll
            for (int mt = 0; mt < 4; mt++) P[rt][mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
#pragma unroll
                for (int hi = 0; hi < 2; hi++) {
                    const int s = 2 * j + hi;
#if Q3P_ABL == 1
                    const h16_t B0 = (h16_t)(_Float16)1.0f, B1 = B0; (void)qs; (void)sc;   // (measurement build: no weight unpack)
#else
                    const h16_t B0 = q4k_bw(qs[0][2 * j + hf], 4 * hi, rok[0] ? (_Float16)sc[0][s] : (_Float16)0.0f);
                    const h16_t B1 = q4k_bw(qs[1][2 * j + hf], 4 * hi, rok[1] ? (_Float16)sc[1][s] : (_Float16)0.0f);
#endif
#pragma unroll
                    for (int mt = 0; mt < 4; mt++) {
                        const h16_t A = mm_a(mt, s * 32 + hf * 16, lane);
                        P[0][mt] = wmma(A, B0, P[0][mt]);
                        P[1][mt] = wmma(A, B1, P[1][mt]);
                    }
                }
            }
        }
#pragma unroll
        for (int rt = 0; rt < 2; rt++) {
            uint32_t bm[8];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const h2_t pm = {(_Float16)m[rt][2 * i], (_Float16)m[rt][2 * i + 1]};
                bm[i] = as_u(pm); bm[4 + i] = as_u(pm);
            }
            const h16_t BM = __builtin_bit_cast(h16_t, bm);
            const uint16_t * hp = (const uint16_t *)&hd[rt];
            const float d = rok[rt] ? (float)__builtin_bit_cast(_Float16, hp[0]) : 0.0f;
            const float dmin = rok[rt] ? (float)__builtin_bit_cast(_Float16, hp[1]) : 0.0f;
#pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                const u32x4_t * sp = (const u32x4_t *)(MMS + (mt * 16 + (lane & 15)) * 16);
                const h16_t AS = h16of(sp[0], sp[1]);
                const f8_t MS = wmma(AS, BM, (f8_t){0, 0, 0, 0, 0, 0, 0, 0});
#pragma unroll
                for (int v = 0; v < 8; v++) acc[rt][mt][v] = __builtin_fmaf(d, P[rt][mt][v], __builtin_fmaf(-dmin, MS[v], acc[rt][mt][v]));
            }
        }
        if (__builtin_amdgcn_readfirstlane(MMBIG) == b + 1) {   // rare: q3p_mm_q4k's other 63 / 64 of the scaled tokens' min term
#pragma unroll
            for (int rt = 0; rt < 2; rt++) {
                const u32x4_t hd2 = *(const u32x4_t *)(wrow[rt] + (size_t)b * 144);
                int sc2[8], m2[8];
                q4k_scales(hd2, sc2, m2);
                uint32_t bm2[8];
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const h2_t p2 = {(_Float16)m2[2 * i], (_Float16)m2[2 * i + 1]};
                    bm2[i] = as_u(p2); bm2[4 + i] = as_u(p2);
                }
                const float dm63 = rok[rt] ? -63.0f * (float)__builtin_bit_cast(_Float16, ((const uint16_t *)&hd2)[1]) : 0.0f;
#pragma unroll
                for (int mt = 0; mt < 4; mt++) {
                    const u32x4_t * sp = (const u32x4_t *)(MMS + (mt * 16 + (lane & 15)) * 16);
                    const f8_t MS = wmma(h16of(sp[0], sp[1]), __builtin_bit_cast(h16_t, bm2), (f8_t){0, 0, 0, 0, 0, 0, 0, 0});
#pragma unroll
                    for (int v = 0; v < 8; v++)
                        if (MMF[mt * 16 + 2 * v + (lane >> 4)] == b + 1) acc[rt][mt][v] = __builtin_fmaf(dm63, MS[v], acc[rt][mt][v]);
                }
            }
        }
    }
#pragma unroll
    for (int rt = 0; rt < 2; rt++) {
        if (!rok[rt]) continue;
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int v = 0; v < 8; v++) {
                const int t = mt * 16 + 2 * v + (lane >> 4);
                if (t < it.cnt) Y[(size_t)(it.off + t) * ldy + r[rt]] = acc[rt][mt][v];
            }
    }
}

// ---- Q5_K ------------------------------------------------------------------------------------------------------------
// 16 values: nibbles (4 dwords, shift sh) | bit s of the qh bytes (4 dwords) << 4 -> fp16 sc * q (exact)
__device__ __forceinline__ h16_t q5k_b(const u32x4_t & qw, int sh, const u32x4_t & hw, int s, _Float16 sc) {
    const h2_t s2 = {sc, sc}, o2 = {(_Float16)(-1024.0f) * sc, (_Float16)(-1024.0f) * sc};
    const uint32_t w4[4] = {qw.x, qw.y, qw.z, qw.w}, h4[4] = {hw.x, hw.y, hw.z, hw.w};
    uint32_t o[8];
#pragma unroll
    for (int w = 0; w < 4; w++) {
        const uint32_t n = ((w4[w] >> sh) & 0x0f0f0f0fu) | (((h4[w] >> s) & 0x01010101u) << 4);
        const uint32_t p01 = __builtin_amdgcn_perm(0u, n, 0x0c010c00u) | 0x64006400u;
        const uint32_t p23 = __builtin_amdgcn_perm(0u, n, 0x0c030c02u) | 0x64006400u;
        o[2 * w] = as_u(__builtin_elementwise_fma(as_h2(p01), s2, o2));
        o[2 * w + 1] = as_u(__builtin_elementwise_fma(as_h2(p23), s2, o2));
    }
    return __builtin_bit_cast(h16_t, o);
}

extern "C" __global__ void __launch_bounds__(256) q35p_mm_q5k(
        const uint8_t * __restrict__ W, int R, int K, const _Float16 * __restrict__ X, int ldx, const int * __restrict__ xsrc,
        float * __restrict__ Y, int ldy, const int * __restrict__ ydst, const MMItem * __restrict__ items, const int * __restrict__ nitems,
        int ntok) {
    MMItem it;
    if (items) { if ((int)blockIdx.x >= *nitems) return; it = items[blockIdx.x]; }
    else { it.e = 0; it.off = blockIdx.x * MM_M; it.cnt = min(MM_M, ntok - it.off); }
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int r = blockIdx.y * MM_N + wave * 16 + (lane & 15);
    const bool rok = r < R;
    const int NB = K / 256;
    const size_t RB = (size_t)NB * 176;
    const uint8_t * wrow = W + ((size_t)it.e * R + (rok ? r : 0)) * RB;
    f8_t acc[4];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) acc[mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
    if (tid < MM_M) MMF[tid] = 0;
    if (tid == 0) MMBIG = 0;
    for (int b = 0; b < NB; b++) {
        const u32x4_t * bp = (const u32x4_t *)(wrow + (size_t)b * 176);
        const u32x4_t hd = bp[0];                 // d, dmin, the 12 scale bytes (Q4_K's header)
        const u32x4_t qh0 = bp[1], qh1 = bp[2];   // qh[0 .. 15], qh[16 .. 31]
        u32x4_t qs[8];
#pragma unroll
        for (int i = 0; i < 8; i++) qs[i] = bp[3 + i];
        __syncthreads();
        mm_stage<true>(X, ldx, xsrc, it.off, it.cnt, b, tid);
        __syncthreads();
        int sc[8], m[8];
        q4k_scales(hd, sc, m);
        f8_t P[4];
#pragma unroll
        for (int mt = 0; mt < 4; mt++) P[mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
#pragma unroll
                for (int hi = 0; hi < 2; hi++) {
                    const int s = 2 * j + hi;
                    const h16_t B = q5k_b(qs[2 * j + hf], 4 * hi, hf ? qh1 : qh0, s, rok ? (_Float16)sc[s] : (_Float16)0.0f);
#pragma unroll
                    for (int mt = 0; mt < 4; mt++) P[mt] = wmma(mm_a(mt, s * 32 + hf * 16, lane), B, P[mt]);
                }
            }
        }
        uint32_t bm[8];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const h2_t p = {(_Float16)m[2 * i], (_Float16)m[2 * i + 1]};
            bm[i] = as_u(p); bm[4 + i] = as_u(p);
        }
        const h16_t BM = __builtin_bit_cast(h16_t, bm);
        const uint16_t * hp = (const uint16_t *)&hd;
        const float d = rok ? (float)__builtin_bit_cast(_Float16, hp[0]) : 0.0f;
        const float dmin = rok ? (float)__builtin_bit_cast(_Float16, hp[1]) : 0.0f;
#pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            const u32x4_t * sp = (const u32x4_t *)(MMS + (mt * 16 + (lane & 15)) * 16);
            const f8_t MS = wmma(h16of(sp[0], sp[1]), BM, (f8_t){0, 0, 0, 0, 0, 0, 0, 0});
#pragma unroll
            for (int v = 0; v < 8; v++) acc[mt][v] = __builtin_fmaf(d, P[mt][v], __builtin_fmaf(-dmin, MS[v], acc[mt][v]));
        }
        if (__builtin_amdgcn_readfirstlane(MMBIG) == b + 1) {   // rare: q3p_mm_q4k's overflow branch, unchanged
            const u32x4_t hd2 = *(const u32x4_t *)(wrow + (size_t)b * 176);
            int sc2[8], m2[8];
            q4k_scales(hd2, sc2, m2);
            uint32_t bm2[8];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const h2_t p2 = {(_Float16)m2[2 * i], (_Float16)m2[2 * i + 1]};
                bm2[i] = as_u(p2); bm2[4 + i] = as_u(p2);
            }
            const float dm63 = rok ? -63.0f * (float)__builtin_bit_cast(_Float16, ((const uint16_t *)&hd2)[1]) : 0.0f;
#pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                const u32x4_t * sp = (const u32x4_t *)(MMS + (mt * 16 + (lane & 15)) * 16);
                const f8_t MS = wmma(h16of(sp[0], sp[1]), __builtin_bit_cast(h16_t, bm2), (f8_t){0, 0, 0, 0, 0, 0, 0, 0});
#pragma unroll
                for (int v = 0; v < 8; v++)
                    if (MMF[mt * 16 + 2 * v + (lane >> 4)] == b + 1) acc[mt][v] = __builtin_fmaf(dm63, MS[v], acc[mt][v]);
            }
        }
    }
    if (!rok) return;
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int v = 0; v < 8; v++) {
            const int t = mt * 16 + 2 * v + (lane >> 4);
            if (t < it.cnt) Y[(size_t)(ydst ? ydst[it.off + t] : it.off + t) * ldy + r] = acc[mt][v];
        }
}

// ---- q35p_add_rms16: per token (one workgroup of 256), OP_OWN_ADD_RMS16's statements for any row length N = 256 NI --------
// (kernels/k3_q35_ops.h q35_op_add_rms16; q3p_add_rms16 stops at N = 2048 and adds weighted slots):
//   x[i] += d[i]   (d nullptr: no add);   h[i] = (x[i] * rsqrt(sum x^2 / N + eps)) * nw[i];   h16[i] = fp16(h[i])
// The same order of operations: a thread's fma chain over its elements tid + 256 j in j order, the wave sum (xor 16 .. 1),
// the eight waves in order, no contraction. Grid n. h is always written, as the decode op writes it: the compiler rounds
// the fp16 copy straight from the last product (v_fma_mixlo_f16: one rounding, the decode module's instruction) only
// when that product's fp32 store sits in the same basic block. Behind `if (h)` it took v_cvt_f16_f32 of the rounded fp32
// value instead (two roundings): 11 and 15 of 189,440 fp16 values differed from the decode's on the RX 7900 XT.
extern "C" __global__ void __launch_bounds__(256) q35p_add_rms16(float * __restrict__ x, const float * __restrict__ d,
                                                                 const float * __restrict__ nw, float eps, float * __restrict__ h,
                                                                 _Float16 * __restrict__ h16, int N) {
#pragma clang fp contract(off)
    __shared__ float part[8];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, NI = N >> 8, t = blockIdx.x;
    float * xr = x + (size_t)t * N;
    const float * dr = d ? d + (size_t)t * N : nullptr;
    float ss = 0.0f;
    for (int j = 0; j < NI; j++) {
        const int i = tid + 256 * j;
        float xv = xr[i];
        if (dr) { xv = xv + dr[i]; xr[i] = xv; }
        ss = __builtin_fmaf(xv, xv, ss);
    }
    ss = wsum(ss); if (lane == 0) part[wave] = ss; __syncthreads();
    float tot = 0.0f; for (int q = 0; q < 8; q++) tot += part[q];
    const float scale = rsqrtf(tot / N + eps);
    for (int j = 0; j < NI; j++) {
        const int i = tid + 256 * j;
        const float v = (xr[i] * scale) * nw[i];
        h[(size_t)t * N + i] = v;
        h16[(size_t)t * N + i] = (_Float16)v;
    }
}

// ---- the attention and GDN glue of the dense qwen35 line on own code (Swift's decode module: kernels/k3_q35_swift.hip) ---
// Each kernel below runs its decode op's statements (kernels/k3_q35_ops.h, k3_own_ops.h) over a chunk's rows with one
// workgroup of 256 threads per decode task, so it is bitwise that op on the same inputs (tests/prefill/test_q35_prefill_ops.py
// against the K3 ops on k3_q35_swift, one token per pass). The ops the prefill takes from other objects as they are:
// pf_qk_nr (prefill_attn_q35d: OP_QK_NR's statements), pf_gdn_gates / pf_gdn_l2 / pf_gdn_step (prefill_gdn_q35),
// pf_gdn_conv / pf_conv_state / pf_f32_mv_k5120 (prefill_glue_q35d).

__device__ __forceinline__ float q35p_silu(float x) { return x / (1.0f + expf(-x)); }   // (k3_q35_ops.h q35_silu)

// q35p_gdn_norm16: OP_GDN_NORM16 per (row, value head): y16 = fp16(((o * rsqrt(mean(o^2) + eps)) * w) * silu(z)).
// o / z: [T * H, K] fp32, w [K], y16 [T * H, K]. Grid T * H.
extern "C" __global__ void __launch_bounds__(256) q35p_gdn_norm16(const float * __restrict__ o, const float * __restrict__ w,
                                                                  const float * __restrict__ z, _Float16 * __restrict__ y16,
                                                                  int K, float eps) {
#pragma clang fp contract(off)
    __shared__ float part[8];
    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    const size_t t = blockIdx.x;
    const float * orow = o + t * K, * zr = z + t * K;
    _Float16 * y = y16 + t * K;
    float ss = 0.0f;
    for (int i = tid; i < K; i += blockDim.x) ss += orow[i] * orow[i];
    ss = wsum(ss); if (lane == 0) part[wave] = ss; __syncthreads();
    float tot = 0.0f; for (int q = 0; q < 8; q++) tot += part[q];
    const float scale = rsqrtf(tot / K + eps);
    for (int i = tid; i < K; i += blockDim.x) {
        const float nrm = (scale * orow[i]) * w[i];
        const float v = nrm * q35p_silu(zr[i]);
        y[i] = (_Float16)v;
    }
}

// q35p_kv_append16: OP_KV_APPEND's f16 rows: cache[pos0 + t] = (k[t], v[t]) as fp16 ([K | V] per position, 2 M halves).
// k / v: [T, M] fp32 at row strides ldk / ldv. A row at or past `rows` is not written and sets *err |= 1 (the host refuses
// such chunks first: k3.check_span). Grid T.
extern "C" __global__ void __launch_bounds__(256) q35p_kv_append16(const float * __restrict__ k, int ldk, const float * __restrict__ v,
                                                                   int ldv, _Float16 * __restrict__ cache, int M, int pos0, int rows,
                                                                   int * __restrict__ err) {
    const int tt = blockIdx.x, pos = pos0 + tt;
    if (pos < 0 || pos >= rows) { if (threadIdx.x == 0) atomicOr(err, 1); return; }
    _Float16 * c = cache + (size_t)pos * 2 * M;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        c[i] = (_Float16)k[(size_t)tt * ldk + i];
        c[M + i] = (_Float16)v[(size_t)tt * ldv + i];
    }
}

// q35p_attn16: OP_ATTN_DEC over a chunk's rows (own_op_attn_dec's batched form on the f16 cache, k3_own_ops.h): one
// workgroup per (row, query head), the decode's task index h = row * heads + head; row r sees cache positions 0 .. pos0 + r.
// The same statements in the same order: 512-position chunks, a wave scores positions t and t + 8 (4 K loads of each in
// flight), wave_max / wave_sum, the 8 wave partials in order, online softmax with expf, a thread sums V over 8 positions
// per round for its dims tid and tid + 256. q: [rows of this launch, heads, D] fp32 (the q norm + RoPE output), out the
// same shape. D a multiple of 128 and <= 512, heads a multiple of nkv (host-checked). Rows r0 .. r0 + gridDim.x / heads - 1
// of the chunk: the host splits a chunk into launches by token-positions (every launch well under the 200 ms bound).
#define Q35P_AT_CH 512
extern "C" __global__ void __launch_bounds__(256) q35p_attn16(const float * __restrict__ qin, const _Float16 * __restrict__ cache,
                                                              int pos0, int r0, int heads, int nkv, int D, float scale,
                                                              float * __restrict__ out) {
    constexpr int CH = Q35P_AT_CH;
    __shared__ float sc[CH], red[8];
    const int h = blockIdx.x, tt = h / heads;                      // (task = row * heads + head, rows from r0)
    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32, g = (h - tt * heads) / (heads / nkv);
    const int npos = pos0 + r0 + tt + 1, M = nkv * D;
    const size_t pb = (size_t)2 * M;                               // (halves per position)
    const float * q = qin + (size_t)h * D;
    float run_m = -INFINITY, run_s = 0.0f, acc[2] = {0.0f, 0.0f};
#pragma unroll 1
    for (int c0 = 0; c0 < npos; c0 += CH) {
        const int n = min(CH, npos - c0);
        for (int t = wave; t < n; t += 16) {                       // positions t and t + 8 (the second clamped, unused past n)
            const int t1 = min(t + 8, n - 1);
            float a0 = 0.0f, a1 = 0.0f;
            const _Float16 * kp0 = cache + (size_t)(c0 + t) * pb + g * D;
            const _Float16 * kp1 = cache + (size_t)(c0 + t1) * pb + g * D;
            for (int d = lane; d < D; d += 128) {
                float qv[4]; _Float16 k0[4], k1[4];
#pragma unroll
                for (int j = 0; j < 4; j++) { qv[j] = q[d + 32 * j]; k0[j] = kp0[d + 32 * j]; k1[j] = kp1[d + 32 * j]; }
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int j = 0; j < 4; j++) { a0 += qv[j] * (float)k0[j]; a1 += qv[j] * (float)k1[j]; }
                __builtin_amdgcn_sched_barrier(0);
            }
            a0 = wsum(a0); a1 = wsum(a1);
            if (lane == 0) { sc[t] = a0 * scale; if (t + 8 < n) sc[t + 8] = a1 * scale; }
        }
        __syncthreads();
        float m = -INFINITY;
        for (int t = tid; t < n; t += blockDim.x) m = fmaxf(m, sc[t]);
        m = wmax(m); if (lane == 0) red[wave] = m; __syncthreads();
        m = red[0]; for (int w = 1; w < 8; w++) m = fmaxf(m, red[w]); __syncthreads();
        const float mn = fmaxf(run_m, m), corr = expf(run_m - mn);
        float sum = 0.0f;
        for (int t = tid; t < n; t += blockDim.x) { const float e = expf(sc[t] - mn); sc[t] = e; sum += e; }
        sum = wsum(sum); if (lane == 0) red[wave] = sum; __syncthreads();
        sum = 0.0f; for (int w = 0; w < 8; w++) sum += red[w];
        run_s = c0 == 0 ? sum : run_s * corr + sum;
        for (int j = 0; j < 2; j++) {
            const int d = tid + j * (int)blockDim.x;
            if (d >= D) break;
            float part = 0.0f;
            const _Float16 * vb = cache + (size_t)c0 * pb + M + g * D + d;   // (V of position c0)
            for (int t = 0; t < n; t += 8) {
                _Float16 vv[8];
#pragma unroll
                for (int i = 0; i < 8; i++) vv[i] = vb[(size_t)min(t + i, n - 1) * pb];
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int i = 0; i < 8; i++) { if (t + i < n) part += sc[t + i] * (float)vv[i]; }
                __builtin_amdgcn_sched_barrier(0);
            }
            acc[j] = c0 == 0 ? part : acc[j] * corr + part;
        }
        run_m = mn;
        __syncthreads();
    }
    const float inv = 1.0f / run_s;
    for (int j = 0; j < 2; j++) {
        const int d = tid + j * (int)blockDim.x;
        if (d < D) out[(size_t)h * D + d] = acc[j] * inv;
    }
}

// ---- q35p_attn_wm: the chunk's causal attention as WMMA tiles (fp16 q / k / v / p, fp32 scores, softmax state and
// sums), head dim 256, up to 8 query heads per kv head. q3m_prefill.hip's q3p_attn (head dim 128) widened to 256: its
// absolute-position layout, its transposed products, its zero-product rules, unchanged; what 256 adds is the wave pair
// (two waves per 16 query rows, each holding half of the head's dims: its half of Q in registers, its half of O; the two
// halves' score partials summed through LDS as lo + hi in both waves) - rocKE's split of head widths above 128 across a
// wave pair (ROCm/rocm-libraries, hip-kernel-provider/rocke, rdna_attention.py; MIT License, Copyright (C) Advanced Micro
// Devices, Inc.: THIRD_PARTY.md), written here from that description; no rocKE code. A wave's registers: Q 64 VGPRs, O 64.
// Work by ABSOLUTE position: workgroup i (8 waves) = positions 8 i .. 8 i + 7 x the G query heads of kv head blockIdx.y;
// pair w (waves 2 w, 2 w + 1) = positions q0 = 8 i + 2 w, q1 = q0 + 1; its 16 columns n = lane & 15 -> position
// q0 + (n >> 3), head G g + (n & 7) (columns with n & 7 >= G: head G g + G - 1 again, never written). Positions outside
// the chunk [pos0, pos0 + n) are computed and never written. Chunks of 32 positions from 0: K [32][AW_KP] and V^T
// [256][AW_VP] staged once for the 8 waves, the score partials [wave][tile][v][lane] (53.8 KB of LDS).
// The same bits at every chunk geometry (the prompt cache's stops) and every launch split: a row's sums depend only on
// absolute positions (q3p_attn's argument: rows past q0 in the pair's diagonal tile zeroed in V^T, q1's own row added
// with fp32 fmas, tiles past the pair skipped, K's rows independent).
// q: the chunk's [n][heads][256] fp32 (the q norm + RoPE output; rounded to fp16 here), out the same shape (fp32).
// scale_log2 = scale * log2(e). Grid (8-position blocks blk0 .. blk0 + gridDim.x - 1, nkv), block 256.
#define AW_C 32
#define AW_D 256
#define AW_KP (AW_D + 8)
#define AW_VP (AW_C + 8)
extern "C" __global__ void __launch_bounds__(256) q35p_attn_wm(const float * __restrict__ qin, const _Float16 * __restrict__ cache,
                                                               int pos0, int n, int blk0, int heads, int nkv, float scale_log2,
                                                               float * __restrict__ out) {
    __shared__ _Float16 Ks[AW_C * AW_KP];
    __shared__ _Float16 Vt[AW_D * AW_VP];
    __shared__ float Sx[8][2][8][32];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, hi = lane >> 4, half = wave & 1, dh = half * 128;
    const int G = heads / nkv, g = blockIdx.y, b8 = (blk0 + (int)blockIdx.x) * 8;
    const size_t M = (size_t)nkv * AW_D, RS = 2 * M;   // halves per cache row
    const int nq = lane & 15, sec = nq >> 3, hh = nq & 7;
    const int q0 = b8 + 2 * (wave >> 1), q1 = q0 + 1;
    const int qpos = q0 + sec, tq = qpos - pos0, hq = g * G + min(hh, G - 1);
    const bool qin_ok = tq >= 0 && tq < n, qok = qin_ok && hh < G;
    h16_t Qb[8];                                        // this wave's half of the column's q, 16 dims per k-step
    {
        const float4 * qp = (const float4 *)(qin + ((size_t)(qin_ok ? tq : 0) * heads + hq) * AW_D + dh);
#pragma unroll
        for (int s = 0; s < 8; s++) {
            h16_t v;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float4 f = qp[4 * s + j];
                v[4 * j] = (_Float16)f.x; v[4 * j + 1] = (_Float16)f.y; v[4 * j + 2] = (_Float16)f.z; v[4 * j + 3] = (_Float16)f.w;
            }
            Qb[s] = v;
        }
    }
    f8_t O[8];
#pragma unroll
    for (int d = 0; d < 8; d++) O[d] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.0f;
    const int npos = min(pos0 + n, b8 + 8);             // rows 0 .. npos - 1 exist for this workgroup (its last position's)
    const int sp = tid >> 3, sq = tid & 7;              // staging: position sp of the chunk, dims 32 sq .. 32 sq + 31
    for (int c0 = 0; c0 < npos; c0 += AW_C) {
        const int prow = min(c0 + sp, npos - 1);       // (rows past the end: a valid row; never in a written column's sum)
        const u32x4_t * kp = (const u32x4_t *)(cache + (size_t)prow * RS + (size_t)g * AW_D + sq * 32);
        const u32x4_t * vp = (const u32x4_t *)(cache + (size_t)prow * RS + M + (size_t)g * AW_D + sq * 32);
        u32x4_t kv[4], vv[4];
#pragma unroll
        for (int j = 0; j < 4; j++) { kv[j] = kp[j]; vv[j] = vp[j]; }
        __syncthreads();
        u32x4_t * kd = (u32x4_t *)(Ks + sp * AW_KP + sq * 32);
#pragma unroll
        for (int j = 0; j < 4; j++) kd[j] = kv[j];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const _Float16 * hv = (const _Float16 *)&vv[j];
#pragma unroll
            for (int e = 0; e < 8; e++) Vt[(sq * 32 + j * 8 + e) * AW_VP + sp] = hv[e];
        }
        __syncthreads();
        const bool act = c0 <= q1;                      // (wave-uniform, the same in both waves of a pair; then c0 <= q0)
        // the pair's tile: positions c0 + 16 std .. + 15 hold q0 and q1 (q0 even: never across a tile edge); 1 when the
        // pair lies past this chunk. Tiles past it are all masked: skipped.
        const int std_ = min(1, (q0 - c0) >> 4), kq0 = q0 - c0 - 16 * std_;   // kq0: q0's index in its tile (even; >= 16: none)
        f8_t S[2];
        if (act) {
#pragma unroll
            for (int st = 0; st < 2; st++) {
                S[st] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
                if (st > std_) continue;
                const u32x4_t * ap = (const u32x4_t *)(Ks + (st * 16 + (lane & 15)) * AW_KP + dh);
#pragma unroll
                for (int s = 0; s < 8; s++) S[st] = wmma(h16of(ap[2 * s], ap[2 * s + 1]), Qb[s], S[st]);
#pragma unroll
                for (int v = 0; v < 8; v++) Sx[wave][st][v][lane] = S[st][v];
            }
        }
        __syncthreads();
        if (!act) continue;
        float cm = -INFINITY;
#pragma unroll
        for (int st = 0; st < 2; st++)
#pragma unroll
            for (int v = 0; v < 8; v++) {
                const int pp = c0 + st * 16 + 2 * v + hi;
                const float sv = Sx[wave & ~1][st][v][lane] + Sx[wave | 1][st][v][lane];   // (lo + hi: the same in both waves)
                const float x = (st <= std_ && pp <= qpos) ? sv * scale_log2 : -INFINITY;
                S[st][v] = x; cm = fmaxf(cm, x);
            }
        cm = fmaxf(cm, __shfl_xor(cm, 16, 32));
        const float mn = fmaxf(m, cm);                  // (position 0 is every row's: mn is finite)
        const float alpha = exp2f(m - mn);
        float rsum = 0.0f;
        uint32_t pw[2][4];                              // fp16 p, two positions per dword: [st][v pair]
#pragma unroll
        for (int st = 0; st < 2; st++)
#pragma unroll
            for (int v = 0; v < 8; v += 2) {
                const _Float16 p0 = (_Float16)exp2f(S[st][v] - mn), p1 = (_Float16)exp2f(S[st][v + 1] - mn);
                rsum += (float)p0 + (float)p1;
                pw[st][v >> 1] = as_u((h2_t){p0, p1});
            }
        rsum += __shfl_xor(rsum, 16, 32);
        l = l * alpha + rsum; m = mn;
#pragma unroll
        for (int d = 0; d < 8; d++)
#pragma unroll
            for (int v = 0; v < 8; v++) O[d][v] *= alpha;
#pragma unroll
        for (int st = 0; st < 2; st++) {
            if (st > std_) continue;
            // P^T operand: positions 16 st .. 16 st + 15 of the lane's row: own (2 v + hi) and lane ^ 16's (2 v + 1 - hi)
            uint32_t o[8];
#pragma unroll
            for (int w = 0; w < 4; w++) {
                const uint32_t mine = pw[st][w], other = (uint32_t)__shfl_xor((int)mine, 16, 32);
                const uint32_t ev = hi ? other : mine, od = hi ? mine : other;   // even positions, odd positions
                o[2 * w] = __builtin_amdgcn_perm(od, ev, 0x05040100u);       // positions 4w, 4w + 1
                o[2 * w + 1] = __builtin_amdgcn_perm(od, ev, 0x07060302u);   // positions 4w + 2, 4w + 3
            }
            const bool diag = st == std_ && kq0 < 16;
            float p_q1 = 0.0f;
            if (diag) {                                 // q1's own entry (index kq0 + 1: dword kq0 / 2, high half) out of B
                const int wq = kq0 >> 1;
#pragma unroll
                for (int w = 0; w < 8; w++)
                    if (w == wq) { p_q1 = (float)as_h2(o[w]).y; if (sec) o[w] &= 0x0000ffffu; }
            }
            const h16_t PB = __builtin_bit_cast(h16_t, o);
#pragma unroll
            for (int d = 0; d < 8; d++) {
                const u32x4_t * ap = (const u32x4_t *)(Vt + (dh + d * 16 + (lane & 15)) * AW_VP + st * 16);
                u32x4_t a0 = ap[0], a1 = ap[1];
                if (diag) {                             // V^T past q0 zeroed (kq0 even: dword kq0 / 2 keeps its low half)
                    uint32_t aw[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                    const int wq = kq0 >> 1;
#pragma unroll
                    for (int w = 0; w < 8; w++) aw[w] = w < wq ? aw[w] : w == wq ? (aw[w] & 0x0000ffffu) : 0u;
                    a0 = (u32x4_t){aw[0], aw[1], aw[2], aw[3]}; a1 = (u32x4_t){aw[4], aw[5], aw[6], aw[7]};
                }
                O[d] = wmma(h16of(a0, a1), PB, O[d]);
            }
            if (diag && sec) {                          // q1's rows: + p(q1) v(q1), fp32 fmas over the lane's dims
                const int kc = 16 * st + kq0 + 1;       // q1's row in the chunk
#pragma unroll
                for (int d = 0; d < 8; d++)
#pragma unroll
                    for (int v = 0; v < 8; v++)
                        O[d][v] = __builtin_fmaf(p_q1, (float)Vt[(dh + d * 16 + 2 * v + hi) * AW_VP + kc], O[d][v]);
            }
        }
    }
    if (!qok) return;
    const float il = 1.0f / l;
    float * op = out + ((size_t)tq * heads + hq) * AW_D + dh;
#pragma unroll
    for (int d = 0; d < 8; d++)
#pragma unroll
        for (int v = 0; v < 8; v++) op[d * 16 + 2 * v + hi] = O[d][v] * il;
}

// q35p_gate_mul16: OP_GATE_MUL16 over values i < N: y16[i] = fp16(a[i] * sigmoid(gate[h * stride + off + d])), h = i / D,
// d = i % D (the q GEMM's [q | gate] head pairs: stride 2 D, offset D; h counts heads across the chunk's rows).
extern "C" __global__ void __launch_bounds__(256) q35p_gate_mul16(const float * __restrict__ a, const float * __restrict__ gt,
                                                                  _Float16 * __restrict__ y, long long N, int D, int stride, int off) {
#pragma clang fp contract(off)
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N; i += (long long)gridDim.x * blockDim.x) {
        const long long h = i / D; const int d = (int)(i - h * D);
        const float sgm = 1.0f / (1.0f + expf(-gt[(size_t)h * stride + off + d]));
        const float v = a[i] * sgm;
        y[i] = (_Float16)v;
    }
}
