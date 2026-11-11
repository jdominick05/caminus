// The IQ study's batched prefill (docs/dev/iq-quant-study.md): kernels/q3m_prefill.hip's kernels as they are, plus the
// quantized GEMM for GGUF's codebook formats on the experts (IQ4_XS, IQ3_XXS, IQ3_S, IQ2_XXS). A study object: the
// shipped prefill object is built from q3m_prefill.hip alone and does not change. No ggml-cuda code; the grids are format
// constants (kernels/iq_grids.h, ggml's data, MIT notice there).
//
// ---- q3p_mm_iq4xs / q3p_mm_iq3xxs / q3p_mm_iq3s / q3p_mm_iq2xxs: q3p_mm_q4k's contract (Y[ydst[i]][r] = X[xsrc[i]] . W_e[r]
// in fp32, fp16 activations through WMMA, the same work items and tiles) -------------------------------------------------
// The decode engine's own weight copies are read in place: IQ4_XS raw GGUF blocks (136 B per 256 values), the grid
// formats in the record layout (caminus.engine.own_layout.iq_records: plane A a record per unit of 32 values, plane B a
// header per block, 16 bytes of padding; Rtot = the rows of the whole buffer). The weights are EXACT in the B operand:
//   IQ4_XS   w = d (ls - 32) kv: B = kv (|kv| <= 127), one product P per unit of 32 values, acc += d (ls - 32) P
//            (kv (ls - 32) reaches 4,064: not every such integer is an fp16, so the unit's scale stays outside B)
//   IQ3_XXS  w = d 0.25 (1 + 2 s) (+-grid): B = (1 + 2 s) (+-grid), an integer of at most 31 * 62 = 1,922: exact in fp16
//   IQ3_S    w = d (1 + 2 s) (+-grid): at most 31 * 15 = 465
//   IQ2_XXS  w = d 0.125 (1 + 2 s) (+-grid): at most 31 * 43 = 1,333
//            the grid formats: one product P over the whole 256-value block, acc += (d c) P
// so a product's only rounding is the fp32 accumulation, as in q3p_mm_q4k. The grids sit in the workgroup's local memory
// (2 KB), staged by the 256 threads before the first block's barrier: IQ3 one word per entry (IQ3_S 512 entries),
// IQ2_XXS entry e as words e (values 0 .. 3) and 256 + e (4 .. 7), the decode tile's form.
#include "q3m_prefill.hip"
#include "iq_grids.h"

typedef uint32_t u32x2p_t __attribute__((ext_vector_type(2)));
__shared__ uint32_t IQP_G[512];

// kv + 128 as bytes, entries 0 .. 15 in four dwords (kvalues_iq4nl: -127 -104 -83 -65 -49 -35 -22 -10 1 13 25 38 53 69 89 113)
#define IQP_T0 0x3f2d1801u
#define IQP_T1 0x766a5d4fu
#define IQP_T2 0xa6998d81u
#define IQP_T3 0xf1d9c5b5u
__device__ __forceinline__ uint32_t iqp_lut4(uint32_t n) {   // 4 nibbles as bytes -> 4 bytes kv + 128
    const uint32_t sel = n & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(IQP_T1, IQP_T0, sel), hi = __builtin_amdgcn_perm(IQP_T3, IQP_T2, sel);
    const uint32_t m = ((n >> 3) & 0x01010101u) * 0xffu;
    return lo ^ ((lo ^ hi) & m);
}
// 4 magnitude bytes m (1 .. 127) and 4 sign bits sb (bit k set: weight k negative) -> 4 bytes weight + 128
__device__ __forceinline__ uint32_t iqp_signed(uint32_t m, uint32_t sb) {
    const uint32_t b = ((sb & 0xfu) * 0x00204081u) & 0x01010101u;
    return ((m ^ (b * 0xffu)) + b) ^ 0x80808080u;
}
// the 8 sign bits of 8 weights from their 7-bit code: the code and its parity
__device__ __forceinline__ uint32_t iqp_signs8(uint32_t s7) { return s7 | ((__builtin_popcount(s7) & 1u) << 7); }
// 16 weights as bytes weight + 128 (4 dwords) -> the B operand weight * sc: fp16 (1024 + byte) * sc - 1152 * sc, one
// fma each (exact while |weight * sc| < 2048; 1152 * sc = 2^7 * 9 sc is an fp16 for sc <= 31)
__device__ __forceinline__ h16_t iqp_b(const uint32_t * u, _Float16 sc) {
    const h2_t s2 = {sc, sc}, o2 = {(_Float16)(-1152.0f) * sc, (_Float16)(-1152.0f) * sc};
    uint32_t o[8];
#pragma unroll
    for (int w = 0; w < 4; w++) {
        const uint32_t p01 = __builtin_amdgcn_perm(0u, u[w], 0x0c010c00u) | 0x64006400u;
        const uint32_t p23 = __builtin_amdgcn_perm(0u, u[w], 0x0c030c02u) | 0x64006400u;
        o[2 * w] = as_u(__builtin_elementwise_fma(as_h2(p01), s2, o2));
        o[2 * w + 1] = as_u(__builtin_elementwise_fma(as_h2(p23), s2, o2));
    }
    return __builtin_bit_cast(h16_t, o);
}

// FMT: 0 IQ3_XXS, 1 IQ3_S, 2 IQ2_XXS (records: 12 / 13 / 8 bytes per unit, header 2 / 6 / 2 per block), 3 IQ4_XS
template <int FMT>
__device__ __forceinline__ void iqp_mm(
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
    const int RS = FMT == 0 ? 12 : FMT == 1 ? 13 : 8, HS = FMT == 1 ? 6 : 2;
    const size_t g = (size_t)it.e * R + (rok ? r : 0);
    const uint8_t * pa = FMT == 3 ? W + g * NB * 136 : W + g * NB * 8 * RS;
    const uint8_t * pb = FMT == 3 ? pa : W + (size_t)Rtot * NB * 8 * RS + g * NB * HS;
    // the grid, thread tid its entry (read after the first block's barriers)
    if (FMT == 0) IQP_G[tid] = IQ3XXS_GRID[tid];
    if (FMT == 1) { IQP_G[tid] = IQ3S_GRID[tid]; IQP_G[256 + tid] = IQ3S_GRID[256 + tid]; }
    if (FMT == 2) { const uint64_t e = IQ2XXS_GRID[tid]; IQP_G[tid] = (uint32_t)e; IQP_G[256 + tid] = (uint32_t)(e >> 32); }
    f8_t acc[4];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) acc[mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
    for (int b = 0; b < NB; b++) {
        // this lane's block: the header and the 8 units (16-byte loads; a record's load runs past it into the next
        // bytes of the buffer), all in flight before the stage's barrier
        u32x4_t rc[8];
        u32x2p_t hd;
        if (FMT == 3) {
            const uint8_t * blk = pa + (size_t)b * 136;
            hd = *(const u32x2p_t *)blk;
#pragma unroll
            for (int i = 0; i < 8; i++) rc[i] = *(const u32x4_t *)(blk + 8 + 16 * i);
        } else {
            hd = *(const u32x2p_t *)(pb + (size_t)b * HS);
#pragma unroll
            for (int i = 0; i < 8; i++) rc[i] = *(const u32x4_t *)(pa + ((size_t)b * 8 + i) * RS);
        }
        __syncthreads();
        mm_stage<false>(X, ldx, xsrc, it.off, it.cnt, b, tid);
        __syncthreads();
        const float d = rok ? (float)__builtin_bit_cast(_Float16, (uint16_t)(hd.x & 0xffffu)) : 0.0f;
        f8_t P[4];
#pragma unroll
        for (int mt = 0; mt < 4; mt++) P[mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int ib = 0; ib < 8; ib++) {
            uint32_t u[8];                                   // the unit's 32 weights + 128, values 4 w .. 4 w + 3 in u[w]
            if (FMT == 3) {
                const uint32_t qw[4] = {rc[ib].x, rc[ib].y, rc[ib].z, rc[ib].w};   // byte k: value k | value 16 + k
#pragma unroll
                for (int w = 0; w < 4; w++) {
                    u[w] = iqp_lut4(qw[w] & 0x0f0f0f0fu); u[4 + w] = iqp_lut4((qw[w] >> 4) & 0x0f0f0f0fu);
                }
                const h16_t B0 = iqp_b(u, (_Float16)1.0f), B1 = iqp_b(u + 4, (_Float16)1.0f);
                const uint32_t ls = ((hd.y >> (4 * ib)) & 0xfu) | ((((hd.x >> 16) >> (2 * ib)) & 3u) << 4);
                const float ds = d * (float)((int)ls - 32);
#pragma unroll
                for (int mt = 0; mt < 4; mt++) {
                    f8_t Pu = wmma(mm_a(mt, 32 * ib, lane), B0, (f8_t){0, 0, 0, 0, 0, 0, 0, 0});
                    Pu = wmma(mm_a(mt, 32 * ib + 16, lane), B1, Pu);
#pragma unroll
                    for (int v = 0; v < 8; v++) acc[mt][v] = __builtin_fmaf(ds, Pu[v], acc[mt][v]);
                }
            } else {
                // index bytes (IQ2_XXS: its 4, each twice: the entry's two words, bit 8 set for the second), the ninth
                // index bits (IQ3_S), the 32 sign bits (IQ3_S; else four 7-bit codes with their parity), the scale
                const uint32_t sw = FMT == 2 ? rc[ib].y : rc[ib].z;
                const uint32_t iw[2] = {FMT == 2 ? __builtin_amdgcn_perm(0u, rc[ib].x, 0x01010000u) : rc[ib].x,
                                        FMT == 2 ? __builtin_amdgcn_perm(0u, rc[ib].x, 0x03030202u) : rc[ib].y};
                const uint32_t qh = FMT == 1 ? rc[ib].w : FMT == 2 ? 0xaau : 0u;
                const uint32_t sg = FMT == 1 ? sw : iqp_signs8(sw & 127u) | (iqp_signs8((sw >> 7) & 127u) << 8) |
                                                    (iqp_signs8((sw >> 14) & 127u) << 16) | (iqp_signs8((sw >> 21) & 127u) << 24);
                const uint32_t si = FMT == 1 ? (((hd.x >> 16) | (hd.y << 16)) >> (4 * ib)) & 0xfu : sw >> 28;
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    const uint32_t m = IQP_G[((iw[k >> 2] >> (8 * (k & 3))) & 0xffu) | (((qh >> k) & 1u) << 8)];
                    u[k] = iqp_signed(m, sg >> (4 * k));
                }
                const _Float16 sc = rok ? (_Float16)(float)(1u + 2u * si) : (_Float16)0.0f;
                const h16_t B0 = iqp_b(u, sc), B1 = iqp_b(u + 4, sc);
#pragma unroll
                for (int mt = 0; mt < 4; mt++) {
                    P[mt] = wmma(mm_a(mt, 32 * ib, lane), B0, P[mt]);
                    P[mt] = wmma(mm_a(mt, 32 * ib + 16, lane), B1, P[mt]);
                }
            }
            // (one unit at a time, as q3p_mm_q6k's sub-blocks: unfenced, the scheduler hoists every unit's decode)
            __builtin_amdgcn_sched_barrier(0);
        }
        if (FMT != 3) {
            const float dc = d * (FMT == 0 ? 0.25f : FMT == 1 ? 1.0f : 0.125f);
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int v = 0; v < 8; v++) acc[mt][v] = __builtin_fmaf(dc, P[mt][v], acc[mt][v]);
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

#define IQP_KERNEL(name, FMT) \
extern "C" __global__ void __launch_bounds__(256) name( \
        const uint8_t * __restrict__ W, int R, int Rtot, int K, const _Float16 * __restrict__ X, int ldx, const int * __restrict__ xsrc, \
        float * __restrict__ Y, int ldy, const int * __restrict__ ydst, const MMItem * __restrict__ items, \
        const int * __restrict__ nitems, int ntok) { \
    iqp_mm<FMT>(W, R, Rtot, K, X, ldx, xsrc, Y, ldy, ydst, items, nitems, ntok); \
}
IQP_KERNEL(q3p_mm_iq3xxs, 0)
IQP_KERNEL(q3p_mm_iq3s, 1)
IQP_KERNEL(q3p_mm_iq2xxs, 2)
IQP_KERNEL(q3p_mm_iq4xs, 3)
