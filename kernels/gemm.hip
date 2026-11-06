// bf16 GEMM on gfx1100 WMMA (milestone M3, v1): C[M, N] = A[M, K] . B[N, K]^T (both operands K-contiguous, as
// F.linear(x, W) and, with a cached W^T, the dgrad), fp32 accumulation in a fixed K order (no split-K), output bf16 or
// fp32, optional per-row (rs[M]) and per-column (cs[N]) fp32 scales applied to the fp32 accumulator (the integer-exact
// linear's epilogue: y = s_x * s_w * (Q_x . Q_w)).
// Workgroup BM x BN, 4 waves in a 2 x 2 grid, each wave (BM/2) x (BN/2) as 16 x 16 WMMA tiles; K-stage 32; LDS double
// buffered (rows padded to 40 bf16 = 80 B so the 16 lanes of a ds_read_b128 hit distinct banks); global loads for
// stage k+1 are issued before stage k's WMMAs and stored to the other LDS buffer after them (no async copy on gfx11).
// WMMA operand layout (gfx11): lane l holds row (A) or column (B) l % 16 with K 0..15 as its 16 elements; lanes 16-31
// duplicate lanes 0-15. Accumulator VGPR r of lane l holds C[2r + l / 16][l % 16].
// Requires M % BM == 0, N % BN == 0, K % 32 == 0, 16-byte aligned rows (python/caminus/gemm.py checks).
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef short s16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));
typedef uint16_t bf16_t;

constexpr int BK = 32, LDR = BK + 8;   // K-stage and padded LDS row (bf16 elements)

__device__ __forceinline__ s16x16 frag(const bf16_t * p) {   // 16 consecutive bf16 from LDS (two ds_read_b128)
    const u32x4 lo = *(const u32x4 *)p, hi = *(const u32x4 *)(p + 8);
    return __builtin_bit_cast(s16x16, __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
}

__device__ __forceinline__ bf16_t to_bf16(float v) {         // round to nearest even (NaN not expected)
    const uint32_t u = __builtin_bit_cast(uint32_t, v);
    return (bf16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

// A given as up to 8 column segments (each its own base pointer and row stride; K-stages never straddle two, so every
// segment boundary is a multiple of 32): C = sum_s A_s . B[:, seg s]^T, the dgrad of a GEMM whose output was split
// into views, without concatenating their gradients first. Same accumulation order as on the concatenated A.
struct ASeg { const bf16_t * p[8]; int ld[8]; int kend[8]; int n; };

// RBF (the *_rbf entry points, fp32 output only): each fp32 result is rounded to bf16 (round to nearest even, to_bf16) and
// stored as that bf16 value in fp32, which is what llama.cpp's own batched BF16 mul_mat gives on RDNA3 above 8 columns
// (ggml_cuda_mul_mat_cublas: hipBLAS with BF16 compute writes a bf16 C, then converts it to fp32). Measured on the XT
// (fix:prefill-s6, golden rows, T 9 .. 768): every element of llama.cpp's output has its low 16 bits zero, and this RNE of
// our fp32 accumulation equals it on 99.7-100 % of elements (100 % at T 255 / 768 for the router, ple_value, indexer q).
template <int BM, int BN, bool LDS_ONLY, bool SEG = false, bool RBF = false>
__device__ __forceinline__ void gemm_tn(const bf16_t * __restrict__ A, const bf16_t * __restrict__ B, void * __restrict__ C,
                                        const float * __restrict__ rs, const float * __restrict__ cs,
                                        int K, int lda, int ldb, int ldc, int out_f32, const ASeg * seg = nullptr) {
    constexpr int WTM = BM / 2, WTN = BN / 2, TM = WTM / 16, TN = WTN / 16;
    constexpr int CA = BM * BK / 8 / 128, CB = BN * BK / 8 / 128;   // 16-byte chunks per thread per stage
    constexpr int LDC = BN + 8;                                         // epilogue staging row (bf16)
    constexpr int SMEM = 2 * (BM + BN) * LDR > BM * LDC ? 2 * (BM + BN) * LDR : BM * LDC;
    __shared__ __attribute__((aligned(16))) bf16_t smem[SMEM];
    bf16_t (*sA)[BM * LDR] = (bf16_t (*)[BM * LDR])smem, (*sB)[BN * LDR] = (bf16_t (*)[BN * LDR])(smem + 2 * BM * LDR);
    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32, wm = wave / 2, wn = wave % 2;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const bf16_t * Ab = A + (size_t)m0 * lda, * Bb = B + (size_t)n0 * ldb;
    u32x4 ra[CA], rb[CB];
    auto gload = [&](int k0) {
        const bf16_t * Ak = Ab + k0; int la = lda;
        if constexpr (SEG) {   // the segment holding this K-stage (uniform across the workgroup)
            int sgi = 0, kb = 0;
            while (sgi + 1 < seg->n && k0 >= seg->kend[sgi]) { kb = seg->kend[sgi]; sgi++; }
            la = seg->ld[sgi]; Ak = seg->p[sgi] + (size_t)m0 * la + (k0 - kb);
        }
#pragma unroll
        for (int j = 0; j < CA; j++) { const int c = tid + 128 * j; ra[j] = *(const u32x4 *)(Ak + (size_t)(c / 4) * la + (c % 4) * 8); }
#pragma unroll
        for (int j = 0; j < CB; j++) { const int c = tid + 128 * j; rb[j] = *(const u32x4 *)(Bb + (size_t)(c / 4) * ldb + k0 + (c % 4) * 8); }
    };
    auto sstore = [&](int buf) {
#pragma unroll
        for (int j = 0; j < CA; j++) { const int c = tid + 128 * j; *(u32x4 *)&sA[buf][(c / 4) * LDR + (c % 4) * 8] = ra[j]; }
#pragma unroll
        for (int j = 0; j < CB; j++) { const int c = tid + 128 * j; *(u32x4 *)&sB[buf][(c / 4) * LDR + (c % 4) * 8] = rb[j]; }
    };
    f32x8 acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
    const int KT = K / BK;
    gload(0); sstore(0); __syncthreads();
    const bf16_t * pa = &sA[0][(wm * WTM + lane % 16) * LDR], * pb = &sB[0][(wn * WTN + lane % 16) * LDR];
    for (int kt = 0; kt < KT; kt++) {
        const int buf = LDS_ONLY ? 0 : kt & 1;
        if (!LDS_ONLY && kt + 1 < KT) gload((kt + 1) * BK);
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            s16x16 af[TM], bfr[TN];
#pragma unroll
            for (int i = 0; i < TM; i++) af[i] = frag(pa + buf * BM * LDR + i * 16 * LDR + ks);
#pragma unroll
            for (int j = 0; j < TN; j++) bfr[j] = frag(pb + buf * BN * LDR + j * 16 * LDR + ks);
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[i], bfr[j], acc[i][j]);
        }
        if (!LDS_ONLY && kt + 1 < KT) sstore(buf ^ 1);
        __syncthreads();
    }
    if (out_f32) {   // fp32 output (exactness tests, scaled integer-exact epilogue): direct stores
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++) {
                const int n = n0 + wn * WTN + j * 16 + lane % 16;
                const float cn = cs ? cs[n] : 1.0f;
#pragma unroll
                for (int r = 0; r < 8; r++) {
                    const int m = m0 + wm * WTM + i * 16 + 2 * r + lane / 16;
                    float v = acc[i][j][r] * cn;
                    if (rs) v *= rs[m];
                    if constexpr (RBF) v = __builtin_bit_cast(float, (uint32_t)to_bf16(v) << 16);
                    ((float *)C)[(size_t)m * ldc + n] = v;
                }
            }
        return;
    }
    // bf16 output: stage the tile in LDS (the main loop's buffers are free after its last barrier), then 16-byte
    // coalesced stores instead of one 2-byte store per lane per accumulator element
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) {
            const int nl = wn * WTN + j * 16 + lane % 16;
            const float cn = cs ? cs[n0 + nl] : 1.0f;
#pragma unroll
            for (int r = 0; r < 8; r++) {
                const int ml = wm * WTM + i * 16 + 2 * r + lane / 16;
                float v = acc[i][j][r] * cn;
                if (rs) v *= rs[m0 + ml];
                smem[ml * LDC + nl] = to_bf16(v);
            }
        }
    __syncthreads();
    constexpr int CPR = BN / 8, CT = BM * BN / 8 / 128;   // 16-byte chunks per row, per thread
#pragma unroll
    for (int j = 0; j < CT; j++) {
        const int c = tid + 128 * j, row = c / CPR, col = (c % CPR) * 8;
        *(u32x4 *)((bf16_t *)C + (size_t)(m0 + row) * ldc + n0 + col) = *(const u32x4 *)&smem[row * LDC + col];
    }
}

#define GEMM(name, BM, BN, LDS_ONLY) \
extern "C" __global__ void __launch_bounds__(128) name(const bf16_t * A, const bf16_t * B, void * C, const float * rs, \
        const float * cs, int K, int lda, int ldb, int ldc, int out_f32) { \
    gemm_tn<BM, BN, LDS_ONLY>(A, B, C, rs, cs, K, lda, ldb, ldc, out_f32); }

#define GEMM_SEG(name, BM, BN) \
extern "C" __global__ void __launch_bounds__(128) name(ASeg seg, const bf16_t * B, void * C, int K, int ldb, int ldc) { \
    gemm_tn<BM, BN, false, true>(nullptr, B, C, nullptr, nullptr, K, 0, ldb, ldc, 0, &seg); }
GEMM_SEG(gemm_tn_seg_128x128, 128, 128)
GEMM_SEG(gemm_tn_seg_128x64, 128, 64)
GEMM_SEG(gemm_tn_seg_64x128, 64, 128)

GEMM(gemm_tn_128x128, 128, 128, false)
GEMM(gemm_tn_128x64, 128, 64, false)
GEMM(gemm_tn_64x128, 64, 128, false)
GEMM(gemm_tn_128x128_lds_only, 128, 128, true)   // microbench: the main loop with operands already in LDS

// fp32 output holding each result rounded to bf16 (RBF above): the batched prefill's wide BF16 matrices in llama.cpp's
// output precision (caminus.prefill_ops.Glue(wmma_out='bf16'))
#define GEMM_RBF(name, BM, BN) \
extern "C" __global__ void __launch_bounds__(128) name(const bf16_t * A, const bf16_t * B, void * C, int K, int lda, int ldb, int ldc) { \
    gemm_tn<BM, BN, false, false, true>(A, B, C, nullptr, nullptr, K, lda, ldb, ldc, 1); }
GEMM_RBF(gemm_tn_128x128_rbf, 128, 128)
GEMM_RBF(gemm_tn_128x64_rbf, 128, 64)
GEMM_RBF(gemm_tn_64x128_rbf, 64, 128)

// ---- iu4 (milestone M4, the integer-exact W4A4 linear): C[M, N] = rs[m] * cs[n] * sum_k qa[m, k] qb[n, k] -----------
// qa [M, K], qb [N, K] signed int4 codes packed two per byte along K (element 2j in the low nibble of byte j), rows of
// K / 2 bytes. v_wmma_i32_16x16x16_iu4 accumulates in int32, exactly (|acc| <= 64 K); the epilogue converts once and
// applies the per-channel weight scale and per-row activation scale (fp32, that order). Same geometry as the bf16
// kernel: a 128-element K-stage of codes is 64 bytes per row, rows padded to 80 bytes in LDS; per K16 step a lane reads
// 8 bytes (16 nibbles) of its row. Requires M % BM, N % BN, K % 128 == 0.
typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef int i32x8 __attribute__((ext_vector_type(8)));

template <int BM, int BN>
__device__ __forceinline__ void gemm_tn_iu4(const uint8_t * __restrict__ A, const uint8_t * __restrict__ B, void * __restrict__ C,
                                            const float * __restrict__ rs, const float * __restrict__ cs,
                                            int K, int lda, int ldb, int ldc, int out_f32) {
    constexpr int SB = 64, LDB = SB + 16;                            // bytes per row per stage (128 codes), padded row
    constexpr int WTM = BM / 2, WTN = BN / 2, TM = WTM / 16, TN = WTN / 16;
    constexpr int CA = BM * SB / 16 / 128, CB = BN * SB / 16 / 128;   // 16-byte chunks per thread per stage
    constexpr int LDC = BN + 8;
    constexpr int SMEM = 2 * (BM + BN) * LDB > BM * LDC * 2 ? 2 * (BM + BN) * LDB : BM * LDC * 2;
    __shared__ __attribute__((aligned(16))) uint8_t smem[SMEM];
    uint8_t (*sA)[BM * LDB] = (uint8_t (*)[BM * LDB])smem, (*sB)[BN * LDB] = (uint8_t (*)[BN * LDB])(smem + 2 * BM * LDB);
    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32, wm = wave / 2, wn = wave % 2;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const uint8_t * Ab = A + (size_t)m0 * lda, * Bb = B + (size_t)n0 * ldb;
    u32x4 ra[CA], rb[CB];
    auto gload = [&](int b0) {
#pragma unroll
        for (int j = 0; j < CA; j++) { const int c = tid + 128 * j; ra[j] = *(const u32x4 *)(Ab + (size_t)(c / 4) * lda + b0 + (c % 4) * 16); }
#pragma unroll
        for (int j = 0; j < CB; j++) { const int c = tid + 128 * j; rb[j] = *(const u32x4 *)(Bb + (size_t)(c / 4) * ldb + b0 + (c % 4) * 16); }
    };
    auto sstore = [&](int buf) {
#pragma unroll
        for (int j = 0; j < CA; j++) { const int c = tid + 128 * j; *(u32x4 *)&sA[buf][(c / 4) * LDB + (c % 4) * 16] = ra[j]; }
#pragma unroll
        for (int j = 0; j < CB; j++) { const int c = tid + 128 * j; *(u32x4 *)&sB[buf][(c / 4) * LDB + (c % 4) * 16] = rb[j]; }
    };
    i32x8 acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = i32x8{0, 0, 0, 0, 0, 0, 0, 0};
    const int KT = K / 128;
    gload(0); sstore(0); __syncthreads();
    const uint8_t * pa = &sA[0][(wm * WTM + lane % 16) * LDB], * pb = &sB[0][(wn * WTN + lane % 16) * LDB];
    for (int kt = 0; kt < KT; kt++) {
        const int buf = kt & 1;
        if (kt + 1 < KT) gload((kt + 1) * SB);
#pragma unroll
        for (int ks = 0; ks < SB; ks += 8) {
            i32x2 af[TM], bfr[TN];
#pragma unroll
            for (int i = 0; i < TM; i++) af[i] = *(const i32x2 *)(pa + buf * BM * LDB + i * 16 * LDB + ks);
#pragma unroll
            for (int j = 0; j < TN; j++) bfr[j] = *(const i32x2 *)(pb + buf * BN * LDB + j * 16 * LDB + ks);
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, af[i], true, bfr[j], acc[i][j], false);
        }
        if (kt + 1 < KT) sstore(buf ^ 1);
        __syncthreads();
    }
    bf16_t * sC = (bf16_t *)smem;
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) {
            const int nl = wn * WTN + j * 16 + lane % 16;
            const float cn = cs ? cs[n0 + nl] : 1.0f;
#pragma unroll
            for (int r = 0; r < 8; r++) {
                const int ml = wm * WTM + i * 16 + 2 * r + lane / 16;
                float v = (float)acc[i][j][r] * cn;
                if (rs) v *= rs[m0 + ml];
                if (out_f32) ((float *)C)[(size_t)(m0 + ml) * ldc + n0 + nl] = v;
                else sC[ml * LDC + nl] = to_bf16(v);
            }
        }
    if (out_f32) return;
    __syncthreads();
    constexpr int CPR = BN / 8, CT = BM * BN / 8 / 128;
#pragma unroll
    for (int j = 0; j < CT; j++) {
        const int c = tid + 128 * j, row = c / CPR, col = (c % CPR) * 8;
        *(u32x4 *)((bf16_t *)C + (size_t)(m0 + row) * ldc + n0 + col) = *(const u32x4 *)&sC[row * LDC + col];
    }
}

#define GEMM_IU4(name, BM, BN) \
extern "C" __global__ void __launch_bounds__(128) name(const uint8_t * A, const uint8_t * B, void * C, const float * rs, \
        const float * cs, int K, int lda, int ldb, int ldc, int out_f32) { \
    gemm_tn_iu4<BM, BN>(A, B, C, rs, cs, K, lda, ldb, ldc, out_f32); }

GEMM_IU4(gemm_tn_iu4_128x128, 128, 128)
GEMM_IU4(gemm_tn_iu4_128x64, 128, 64)
GEMM_IU4(gemm_tn_iu4_64x128, 64, 128)

// ---- iu4 with per-(column, 128-group) weight scales (the g128 quantizer, kept): -----------------------------------
// C[m, n] = rs[m] * acc,  acc = fma(float(P_g[m, n]), gs[n * G + g], acc) for g = 0 .. G-1 in order, acc0 = 0, where
// P_g is the exact int32 dot of the 128 codes of group g. One K-stage = one group: 8 iu4 WMMAs into an int32 tile, then
// one convert + FMA per accumulator element. Both accumulator sets live in VGPRs, so only 64-wide wave blocks (tiles
// 64x128 / 128x64). Deterministic: every kernel using this order gives the same bits (not order-free like per-channel).
template <int BM, int BN>
__device__ __forceinline__ void gemm_tn_iu4g(const uint8_t * __restrict__ A, const uint8_t * __restrict__ B, void * __restrict__ C,
                                             const float * __restrict__ rs, const float * __restrict__ gs,
                                             int K, int lda, int ldb, int ldc, int out_f32) {
    constexpr int SB = 64, LDB = SB + 16;
    constexpr int WTM = BM / 2, WTN = BN / 2, TM = WTM / 16, TN = WTN / 16;
    constexpr int CA = BM * SB / 16 / 128, CB = BN * SB / 16 / 128;
    constexpr int LDC = BN + 8;
    constexpr int SMEM = 2 * (BM + BN) * LDB > BM * LDC * 2 ? 2 * (BM + BN) * LDB : BM * LDC * 2;
    __shared__ __attribute__((aligned(16))) uint8_t smem[SMEM];
    uint8_t (*sA)[BM * LDB] = (uint8_t (*)[BM * LDB])smem, (*sB)[BN * LDB] = (uint8_t (*)[BN * LDB])(smem + 2 * BM * LDB);
    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32, wm = wave / 2, wn = wave % 2;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, G = K / 128;
    const uint8_t * Ab = A + (size_t)m0 * lda, * Bb = B + (size_t)n0 * ldb;
    u32x4 ra[CA], rb[CB];
    auto gload = [&](int b0) {
#pragma unroll
        for (int j = 0; j < CA; j++) { const int c = tid + 128 * j; ra[j] = *(const u32x4 *)(Ab + (size_t)(c / 4) * lda + b0 + (c % 4) * 16); }
#pragma unroll
        for (int j = 0; j < CB; j++) { const int c = tid + 128 * j; rb[j] = *(const u32x4 *)(Bb + (size_t)(c / 4) * ldb + b0 + (c % 4) * 16); }
    };
    auto sstore = [&](int buf) {
#pragma unroll
        for (int j = 0; j < CA; j++) { const int c = tid + 128 * j; *(u32x4 *)&sA[buf][(c / 4) * LDB + (c % 4) * 16] = ra[j]; }
#pragma unroll
        for (int j = 0; j < CB; j++) { const int c = tid + 128 * j; *(u32x4 *)&sB[buf][(c / 4) * LDB + (c % 4) * 16] = rb[j]; }
    };
    f32x8 acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
    const float * gcol[TN];                                     // this lane's columns' group-scale rows
#pragma unroll
    for (int j = 0; j < TN; j++) gcol[j] = gs + (size_t)(n0 + wn * WTN + j * 16 + lane % 16) * G;
    gload(0); sstore(0); __syncthreads();
    const uint8_t * pa = &sA[0][(wm * WTM + lane % 16) * LDB], * pb = &sB[0][(wn * WTN + lane % 16) * LDB];
    for (int g = 0; g < G; g++) {
        const int buf = g & 1;
        if (g + 1 < G) gload((g + 1) * SB);
        float sg[TN];
#pragma unroll
        for (int j = 0; j < TN; j++) sg[j] = gcol[j][g];
        i32x8 t[TM][TN];
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++) t[i][j] = i32x8{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int ks = 0; ks < SB; ks += 8) {
            i32x2 af[TM], bfr[TN];
#pragma unroll
            for (int i = 0; i < TM; i++) af[i] = *(const i32x2 *)(pa + buf * BM * LDB + i * 16 * LDB + ks);
#pragma unroll
            for (int j = 0; j < TN; j++) bfr[j] = *(const i32x2 *)(pb + buf * BN * LDB + j * 16 * LDB + ks);
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++)
                    t[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, af[i], true, bfr[j], t[i][j], false);
        }
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++)
#pragma unroll
                for (int r = 0; r < 8; r++) acc[i][j][r] = __builtin_fmaf((float)t[i][j][r], sg[j], acc[i][j][r]);
        if (g + 1 < G) sstore(buf ^ 1);
        __syncthreads();
    }
    bf16_t * sC = (bf16_t *)smem;
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) {
            const int nl = wn * WTN + j * 16 + lane % 16;
#pragma unroll
            for (int r = 0; r < 8; r++) {
                const int ml = wm * WTM + i * 16 + 2 * r + lane / 16;
                float v = acc[i][j][r];
                if (rs) v *= rs[m0 + ml];
                if (out_f32) ((float *)C)[(size_t)(m0 + ml) * ldc + n0 + nl] = v;
                else sC[ml * LDC + nl] = to_bf16(v);
            }
        }
    if (out_f32) return;
    __syncthreads();
    constexpr int CPR = BN / 8, CT = BM * BN / 8 / 128;
#pragma unroll
    for (int j = 0; j < CT; j++) {
        const int c = tid + 128 * j, row = c / CPR, col = (c % CPR) * 8;
        *(u32x4 *)((bf16_t *)C + (size_t)(m0 + row) * ldc + n0 + col) = *(const u32x4 *)&sC[row * LDC + col];
    }
}

#define GEMM_IU4G(name, BM, BN) \
extern "C" __global__ void __launch_bounds__(128) name(const uint8_t * A, const uint8_t * B, void * C, const float * rs, \
        const float * gs, int K, int lda, int ldb, int ldc, int out_f32) { \
    gemm_tn_iu4g<BM, BN>(A, B, C, rs, gs, K, lda, ldb, ldc, out_f32); }

GEMM_IU4G(gemm_tn_iu4g_128x64, 128, 64)
GEMM_IU4G(gemm_tn_iu4g_64x128, 64, 128)
