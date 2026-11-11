// The dense qwen35 line's batched prefill for ThinkingCap (Qwen3.8-27B RCO 4.2 bpw; python/caminus/prefill/q35own.py):
// kernels/q35_prefill.hip as it is (Swift's object: the glue, Q4_K / Q6_K / Q8_0 / Q5_K GEMMs), the IQ study's
// codebook-format GEMMs (kernels/q3m_prefill_iq.h: IQ4_XS, IQ3_XXS, IQ3_S; the same text as the IQ study's object), and
// this file's IQ2_S GEMM and IQ2_S embedding. No ggml-cuda code; the grids are format constants (kernels/iq_grids.h,
// kernels/iq2s_grid.h: ggml's data, MIT notice there). Swift's prefill object and the IQ study's object don't include
// this file: their code objects are unchanged.
#include "q35_prefill.hip"
#include "iq_grids.h"
#include "q3m_prefill_iq.h"
#include "iq2s_grid.h"

// 4 weights' 2-bit codes (bits 2k .. 2k + 1 of c: weight k) -> their 4 magnitude bytes (code 0: 8, 1: 25, 2: 43)
__device__ __forceinline__ uint32_t q35t_mag4(uint32_t c) {
    const uint32_t t = (c | (c << 6) | (c << 12) | (c << 18)) & 0x03030303u;
    return __builtin_amdgcn_perm(0u, 0x002b1908u, t);
}

// ---- q35t_mm_iq2s: q3p_mm_q4k's contract (Y[ydst[i]][r] = X[xsrc[i]] . W_e[r] in fp32, fp16 activations through WMMA,
// the same work items and tiles; items nullptr: dense rows), the IQ study's GEMM form for IQ2_S. W = the decode engine's
// copy in the record layout (caminus.engine.own_layout.iq_records: a 10-byte record per unit of 32 values, 4 index bytes
// | 4 sign bytes | qh | the scale byte; then d per block; Rtot = the rows of the whole buffer).
// w = d 0.125 (1 + 2 s) (+-grid), s per 16 values: B = (1 + 2 s) (+-grid) is an integer of at most 31 * 43 = 1,333, exact
// in fp16 (iqp_b: exact while |B| < 2048), so one product P over the whole block and acc += (d 0.125) P: a product's
// only rounding is the fp32 accumulation, as in the IQ study's GEMMs. The grid packed to 2 bits a magnitude (512 words:
// the IQ study's LDS array IQP_G), staged by the 256 threads before the first block's barrier.
extern "C" __global__ void __launch_bounds__(256) q35t_mm_iq2s(
        const uint8_t * __restrict__ W, int R, int Rtot, int K, const _Float16 * __restrict__ X, int ldx, const int * __restrict__ xsrc,
        float * __restrict__ Y, int ldy, const int * __restrict__ ydst, const MMItem * __restrict__ items, const int * __restrict__ nitems,
        int ntok) {
    MMItem it;
    if (items) { if ((int)blockIdx.x >= *nitems) return; it = items[blockIdx.x]; }
    else { it.e = 0; it.off = blockIdx.x * MM_M; it.cnt = min(MM_M, ntok - it.off); }
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int r = blockIdx.y * MM_N + wave * 16 + (lane & 15);
    const bool rok = r < R;
    const int NB = K / 256;
    constexpr int RS = 10, HS = 2;
    const size_t g = (size_t)it.e * R + (rok ? r : 0);
    const uint8_t * pa = W + g * NB * 8 * RS;
    const uint8_t * pb = W + (size_t)Rtot * NB * 8 * RS + g * NB * HS;
    IQP_G[tid] = IQ2S_GRID[tid]; IQP_G[256 + tid] = IQ2S_GRID[256 + tid];   // (read after the first block's barriers)
    f8_t acc[4];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) acc[mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
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
            const uint32_t iw = rc[ib].x, sg = rc[ib].y, qh = rc[ib].z & 0xffu, scb = (rc[ib].z >> 8) & 0xffu;
            uint32_t u[8];                                   // the unit's 32 weights + 128, values 4 w .. 4 w + 3 in u[w]
#pragma unroll
            for (int q = 0; q < 4; q++) {                    // index q: 8 weights, two 4-weight groups
                const uint32_t idx = ((iw >> (8 * q)) & 0xffu) | (((qh >> (2 * q)) & 3u) << 8);
                const uint32_t e = IQP_G[idx >> 1] >> (16 * (idx & 1u));
                u[2 * q] = iqp_signed(q35t_mag4(e & 0xffu), sg >> (8 * q));
                u[2 * q + 1] = iqp_signed(q35t_mag4((e >> 8) & 0xffu), sg >> (8 * q + 4));
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
    if (!rok) return;
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int v = 0; v < 8; v++) {
            const int t = mt * 16 + 2 * v + (lane >> 4);
            if (t < it.cnt) Y[(size_t)(ydst ? ydst[it.off + t] : it.off + t) * ldy + r] = acc[mt][v];
        }
}

// q35t_emb_iq2s: x [n][N] fp32 = the IQ2_S rows toks[i] of the embedding (raw GGUF blocks, rb bytes per row), clamped to
// the vocabulary: kernels/k3_q35t_ops.h q35t_op_emb's statements (the host decoder's float32 operations in its order;
// bitwise). Grid n, block 256.
extern "C" __global__ void __launch_bounds__(256) q35t_emb_iq2s(const uint8_t * __restrict__ W, int rb, int vocab,
                                                                const int * __restrict__ toks, float * __restrict__ x, int N) {
    const int tok = min(max(toks[blockIdx.x], 0), vocab - 1);
    const uint8_t * row = W + (size_t)tok * rb;
    float * xr = x + (size_t)blockIdx.x * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const uint8_t * blk = row + (size_t)(i >> 8) * 82;
        const int j = i & 255, ib = j >> 5, q = (j >> 3) & 3, k = j & 7;
        const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(blk[0] | (blk[1] << 8)));
        const uint32_t idx = blk[2 + 4 * ib + q] | (((uint32_t)(blk[66 + ib] >> (2 * q)) & 3u) << 8);
        const uint32_t code = ((IQ2S_GRID[idx >> 1] >> (16 * (idx & 1u))) >> (2 * k)) & 3u;
        const float mag = code == 0 ? 8.0f : code == 1 ? 25.0f : 43.0f;
        const float s = (float)((blk[74 + ib] >> (4 * ((j >> 4) & 1))) & 15);
        const float db = __fmul_rn(__fmul_rn(d, __fadd_rn(0.5f, s)), 0.25f);
        const float v = __fmul_rn(db, mag);
        xr[i] = ((blk[34 + (j >> 3)] >> k) & 1) ? -v : v;
    }
}
