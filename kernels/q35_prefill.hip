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
    const int NB = K / 256;                       // (staging steps of 256 values: 8 Q8_0 blocks each)
    const size_t RB = (size_t)(K / 32) * 34;
    const uint8_t * wrow = W + ((size_t)it.e * R + (rok ? r : 0)) * RB;
    f8_t acc[4];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) acc[mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
    for (int b = 0; b < NB; b++) {
        __syncthreads();
        mm_stage<false>(X, ldx, xsrc, it.off, it.cnt, b, tid);
        __syncthreads();
#pragma unroll 1
        for (int j = 0; j < 8; j++) {          // (rolled: unrolled, the 8 blocks' loads were hoisted: 256 VGPRs, 171 spilled)
            const uint8_t * bp = wrow + (size_t)(b * 8 + j) * 34;
            const float d = rok ? (float)__builtin_bit_cast(_Float16, *(const uint16_t *)bp) : 0.0f;
            f8_t P[4];
#pragma unroll
            for (int mt = 0; mt < 4; mt++) P[mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                const h16_t B = q8_b(bp + 2 + 16 * hf, rok);
#pragma unroll
                for (int mt = 0; mt < 4; mt++) P[mt] = wmma(mm_a(mt, j * 32 + hf * 16, lane), B, P[mt]);
            }
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int v = 0; v < 8; v++) acc[mt][v] = __builtin_fmaf(d, P[mt][v], acc[mt][v]);
            __builtin_amdgcn_sched_barrier(0);    // (one block at a time: q3p_mm_q6k's register lesson)
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
