// Batched prefill, the dense quantized GEMMs of our own (impl:q6k-dense; project decision 11-03: prefill GEMM kernels of our
// own): out[c * rows + r] = sum_k W[r, k] x[c, k] for a dense Q6_K weight matrix (the Coder's attn_qkv / attn_gate / ssm_out /
// attn_q / k / v / attn_output / shared-expert gate / up shards: 128 of the 48 layers' dense GEMMs per chunk) on the activation
// MMQ already quantizes (caminus.mmq.MMQ.quantize: block_q8_1_mmq, D4 layout), replacing the vendored MMQ's mul_mat_q for them.
//
//   k3_dense_mm_v<VD>_<J>   one workgroup per (64-row block, J-column tile), MMQ's own Q6_K configs (128 threads, 4 waves of
//                           16 rows, K in 256-value iterations, two 128-value activation halves each); VD picks the tile product:
//     VD 0  the vendored load_tiles + vec_dot (MMQ itself with the dense driver below: the A/B control)
//     VD 1  the vendored load_tiles, our vec_dot: MMQ's per-output expression with the scale product C * sc as __mul24
//     VD 2  our load_tiles (the row's 16 sub-block scales stored as fp32 s = sc * d) and our vec_dot: t = float(C) * s
//     VD 3  as VD 2, the WMMA's accumulator input set to 0x4B400000: t = fma(asfloat(C + 0x4B400000), s, -12582912 * s)
//   k3_dense_deq_bf16       Q6_K -> bf16 rows (candidate b: dequantize + gemm.hip's bf16 WMMA, a measured tolerance, not bitwise)
//     (measured 11-04, the XT, every Coder Q6_K shard shape at n 1024 / 2048, interleaved medians: VD 0 1.00-1.01x MMQ's
//     time, VD 1 0.80, VD 2 0.73, VD 3 0.71 on the large shapes; the 48 layers' dense Q6_K per 2K chunk at card 0's shapes
//     150.8 -> 107.2 ms as one launch per op, 158.9 -> 113.5 as 2 x 1024. A variant that loaded the next phase's weight /
//     activation words into registers before each tile product (gfx11 has no async global -> LDS copy) was no faster:
//     122.6 / 131.4 ms at J 128, 109.3 / 114.2 at its best J, so the loads' latency is not what bounds these kernels)
//
// Numerics: BITWISE ggml's MMQ per output, every VD and J. MMQ (vec_dot_q6_K_q8_1_mma, its instructions:
// v_mul_lo_u32, v_cvt_f32_i32, v_mul_f32, v_fmac_f32) computes per 16-value sub-block, in K order,
// sum = fma(fl(float(C * sc) * d), dB, sum) with C the sub-block's exact int dot (one int8 WMMA), sc its int8 scale, d the
// block's fp16 scale as fp32, dB the activation column's 32-value scale. |C| <= 16 * 32 * 127 = 65,024 and |sc| <= 128, so
// C * sc (< 2^23) is exact as a 24-bit product (VD 1) and as an fp32 value, and fl(float(C * sc) * d) is ONE rounding of the
// exact real C * sc * d. s = sc * d has at most 8 + 11 significant bits, so it is exact in fp32, and fl(float(C) * s) is one
// rounding of the same real (VD 2). In VD 3 asfloat(C + 0x4B400000) = 12,582,912 + C exactly (|C| < 2^22) and
// -12,582,912 * s = -3 * 2^22 * s is exact, so the fma's exact intermediate is C * s again, rounded once. The sum's fma is
// the same instruction on the same operands in the same order, starting from +0.0f, so the outputs are MMQ's bits; an
// output depends only on its weight row and its activation column (not on T, J, the tile, the chunk split: G2b).
// tests/test_prefill_dense.py g1 / g2b check every VD and J against mmq.mm on real Q6_K shards.
//
// What else is ours: the dense driver (grid rows / 64 x ceil(ncols / J), no ids, no stream-k). It reads the weights' rows x K
// and, like MMQ, whole J-column activation tiles (within mmq.y_bytes(K, ncols), its Y_TAIL included: the host check), and
// writes only dst[0 .. ncols * rows). Compiled like kernels/prefill_mmq.hip (ignite's flags, a full HIP compile, the gfx1100 object
// unbundled: build.sh), one object per format: build/prefill_dense_q6_k.hsaco.
//
// The load_tiles and vec_dot of VD 1 .. 3 and the process-tile loop are modified copies of ggml-cuda's
// mmq-load-tiles.cuh ggml_cuda_mmq_load_tiles_q6_K, mmq-vec-dot.cuh ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma and mmq.cuh
// mul_mat_q / mul_mat_q_process_tile (MIT, The ggml authors; notice below and in THIRD_PARTY.md).
//
// ---- ggml-cuda (llama.cpp) notice for the copied code ----------------------------------------------------------------
// MIT License
//
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "ggml_mmq/mmq.cuh"

#if !defined(K3_DENSE_TYPE)
#error "prefill_dense.hip: build with -DK3_DENSE_TYPE=<ggml_type> (build.sh)"
#endif
static_assert(K3_DENSE_TYPE == GGML_TYPE_Q6_K, "prefill_dense.hip: only Q6_K has a tile product here");

#define K3_DENSE_PREC GGML_PREC_Q8
#define K3_DENSE_MAGIC 0x4B400000          // asfloat: 1.5 * 2^23 = 12,582,912.0f
#define K3_DENSE_MAGIC_F 12582912.0f

template <ggml_type type, int J, int VD> struct dense_cfg {
    static constexpr int warp_size = 32;
    static constexpr int nthreads  = ggml_cuda_mmq_get_config_rdna3(type, J, false).nthreads;
    static constexpr int nwarps    = nthreads / warp_size;
    static constexpr int I         = ggml_cuda_mmq_get_config_rdna3(type, J, false).I;
    static constexpr int occupancy = ggml_cuda_mmq_get_config_rdna3(type, J, false).occupancy;
    static constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    static constexpr int ITER_K    = ggml_cuda_mmq_get_config_rdna3(type, J, false).K_vram;
    static constexpr int bpi       = ITER_K / qk;                        // weight blocks per K iteration
    // the weight tile's row: VD 0 / 1 MMQ's Q6_K layout (64 words of int8 values, d, 4 words of int8 scales, padding: 76);
    // VD 2 / 3 the 64 value words and the 16 fp32 sub-block scales s = sc * d (+ 4: the stride stays 4 mod 8 like ggml's)
    static constexpr int stride    = VD <= 1 ? ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K) : 2*MMQ_TILE_NE_K + 16 + 4;
    static constexpr int ytile     = GGML_PAD(J * MMQ_TILE_Y_K, nthreads); // ints: one 128-value activation half-tile
    static constexpr int xtile     = I * stride;
    static constexpr int nbytes    = 4 * (J + ytile + xtile);
    static_assert(I == nwarps * 16, "the WMMA vec_dot covers 16 rows per warp");
    static_assert(ITER_K == 2 * QK8_1_MMQ && bpi == 1, "Q6_K: one 256-value block per iteration, two activation halves");
    static_assert(stride % 8 == 4, "LDS row padding");
    static_assert(VD > 1 || stride == ggml_cuda_mmq_get_sram_stride(type, J, false, K3_DENSE_PREC), "VD 0 / 1: the vendored tile layout");
};

// ---- the weight tile, VD 2 / 3: ggml_cuda_mmq_load_tiles_q6_K's MMA branch, the scales as fp32 s = sc * d ----------------
template <ggml_type type, int J, int VD>
static __device__ __forceinline__ void dense_load_x_q6_K(const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0,
                                                         const int stride_row) {
    using C = dense_cfg<type, J, VD>;
    int   * x_qs = x_tile;
    float * x_s  = (float *) (x_qs + 2*MMQ_TILE_NE_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR6_K);
    constexpr int nrows = C::warp_size / threads_per_row;
    const int txi = C::warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
#pragma unroll
    for (int i0 = 0; i0 < C::I; i0 += nrows*C::nwarps) {
        const int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride_row;

        const int ql = get_int_b2(bxi->ql, txi);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b2(bxi->qh, (QI6_K/4) * (txi / (QI6_K/2)) + txi % (QI6_K/4));
        const int qh0 = ((qh >> ((txi & 0x08) >> 2)) << 4) & 0x30303030;
        const int qh1 =  (qh >> ((txi & 0x08) >> 2))       & 0x30303030;

        const int kq0 = 2*txi - txi % (QI6_K/2) + 0;
        const int kq1 = 2*txi - txi % (QI6_K/2) + QI6_K/2;

        x_qs[i*C::stride + kq0] = __vsub4(ql0 | qh0, 0x20202020);
        x_qs[i*C::stride + kq1] = __vsub4(ql1 | qh1, 0x20202020);
    }

    // the 16 scales of each row: 4 threads per row, 4 scales each; s = (float) sc * d is exact (<= 19 significant bits)
    constexpr int rows_per_warp = C::warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < C::I; i0 += C::nwarps*rows_per_warp) {
        const int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/4) % C::I;
        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride_row;
        const float d = bxi->d;
        const int q = threadIdx.x % 4;
        const int sc4 = get_int_b2(bxi->scales, q);
        const int8_t * sc = (const int8_t *) &sc4;
#pragma unroll
        for (int e = 0; e < 4; ++e) x_s[i*C::stride + 4*q + e] = (float) sc[e] * d;
    }
}

// ---- the tile product, VD 1 .. 3: ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma (AMD WMMA), per output the same operations -----------
template <ggml_type type, int J, int VD>
static __device__ __forceinline__ void dense_vec_dot_q6_K(const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum,
                                                          const int k00) {
#if defined(AMD_WMMA_AVAILABLE)
    using Cf = dense_cfg<type, J, VD>;
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  4, int, input_layout>        tile_A;
    typedef tile<16,  4, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int sram_stride   = Cf::stride;
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, false, K3_DENSE_PREC);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;          // VD 1: d; VD 2 / 3: the 16 s values
    const int   * x_sc = (const int   *) x_df + MMQ_TILE_NE_K/QI6_K;      // VD 1 only
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                if constexpr (VD == 3) {
#pragma unroll
                    for (int l = 0; l < tile_C::ne; ++l) C.x[l] = K3_DENSE_MAGIC;
                }
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                    float & acc = sum[(j0/tile_C::J + n)*tile_C::ne + l];
                    if constexpr (VD == 1) {
                        const int8_t * sc = (const int8_t *) (x_sc + i*sram_stride + k00/16);
                        acc += __mul24(C.x[l], sc[k01/4]) * x_df[i*sram_stride] * dB;
                    } else if constexpr (VD == 2) {
                        const float t = (float) C.x[l] * x_df[i*sram_stride + k0/4];
                        acc = __builtin_fmaf(t, dB, acc);
                    } else {
                        const float s = x_df[i*sram_stride + k0/4];
                        const float t = __builtin_fmaf(__int_as_float(C.x[l]), s, -K3_DENSE_MAGIC_F * s);
                        acc = __builtin_fmaf(t, dB, acc);
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, sum, k00);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE)
}

// one iteration's activation half-tile (128 values of the tile's columns) into LDS: mul_mat_q_process_tile's copy loop as is.
// Like MMQ it reads the whole tile, up to J - 1 columns past ncols in the last one (the next K block's columns, or after the
// last K block mmq.y_bytes' Y_TAIL, which the host check requires within y's view); those columns' sums are never written.
// Measured (XT, 11-04): zero-filling them instead cost 1.7-2x MMQ's time at J 128: a conditional load compiled to a branch per
// word (87 s_cbranch vs 20), and a clamped index to 36 hoisted 64-bit addresses (221 VGPRs, the WMMAs then issued one at a time)
template <ggml_type type, int J, int VD>
static __device__ __forceinline__ void dense_load_y(int * tile_y, const int * __restrict__ by0) {
    using C = dense_cfg<type, J, VD>;
#pragma unroll
    for (int l0 = 0; l0 < C::ytile; l0 += C::nthreads) {
        const int l = l0 + threadIdx.y * C::warp_size + threadIdx.x;
        tile_y[l] = by0[l];
    }
}

// ---- k3_dense_mm: x [rows][nb] Q6_K blocks (K = nb * 256); y block_q8_1_mmq [K_pad / 128][ncols]; dst [ncols][rows] fp32 ------
// Grid (rows / I, ceil(ncols / J)): mul_mat_q's dense (no ids, no stream-k) workgroup, each output tile written once
template <ggml_type type, int J, int VD>
static __device__ __forceinline__ void dense_mm_body(const char * __restrict__ x, const int * __restrict__ y, float * __restrict__ dst,
                                                     const int rows, const int nb, const int ncols) {
    using C = dense_cfg<type, J, VD>;
    constexpr ggml_cuda_mmq_load_tiles_t load_tiles = ggml_cuda_mmq_get_load_tiles<type, J, false, K3_DENSE_PREC>();
    constexpr ggml_cuda_mmq_vec_dot_t    vec_dot    = ggml_cuda_mmq_get_vec_dot<type, J, false, K3_DENSE_PREC>();
    constexpr ggml_cuda_mmq_write_back_t write_back = ggml_cuda_mmq_get_write_back<type, J, false, K3_DENSE_PREC>();
    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);

    const int it = blockIdx.x, jt = blockIdx.y;
    if ((it + 1) * C::I > rows || jt * J >= ncols) return;       // (the host's grid never has one)

    extern __shared__ int data_dense[];
    int * ids_sh = data_dense;
    int * tile_y = data_dense + J;
    int * tile_x = tile_y + C::ytile;
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += C::nthreads) {
        const int j = j0 + threadIdx.y * C::warp_size + threadIdx.x;
        if (j < J) ids_sh[j] = j;
    }
    __syncthreads();

    const int offset_x = it * C::I * nb;                          // in blocks
    const int * yt = y + jt * J * sz;
    const int ncj = min(J, ncols - jt * J);
    float sum[J * C::I / C::nthreads] = {0.0f};

    for (int kb0 = 0; kb0 < nb; kb0 += C::bpi) {
        if constexpr (VD <= 1) load_tiles(x, tile_x, offset_x + kb0, C::I - 1, nb);
        else dense_load_x_q6_K<type, J, VD>(x, tile_x, offset_x + kb0, nb);
        dense_load_y<type, J, VD>(tile_y, yt + ncols * (kb0 * C::qk / QK8_1_MMQ) * sz);
        __syncthreads();
        if constexpr (VD == 0) vec_dot(tile_x, tile_y, sum, 0); else dense_vec_dot_q6_K<type, J, VD>(tile_x, tile_y, sum, 0);
        __syncthreads();
        dense_load_y<type, J, VD>(tile_y, yt + ncols * ((kb0 * C::qk / QK8_1_MMQ) * sz + sz));
        __syncthreads();
        if constexpr (VD == 0) vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K); else dense_vec_dot_q6_K<type, J, VD>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();
    }
    write_back(sum, ids_sh, dst + jt * J * rows + it * C::I, nullptr, rows, C::I - 1, ncj - 1);
}

// ---- the kernels, per VD and J -----------------------------------------------------------------------------------------
#define K3_DENSE_MM(T, VD, J)                                                                                               \
extern "C" __global__ void __launch_bounds__((dense_cfg<T, J, VD>::nthreads), (dense_cfg<T, J, VD>::occupancy))             \
k3_dense_mm_v##VD##_##J(const char * x, const int * y, float * dst, int rows, int nb, int ncols) {                         \
    dense_mm_body<T, J, VD>(x, y, dst, rows, nb, ncols);                                                                    \
}
#define K3_DENSE_JS(X, T, VD) X(T, VD, 16) X(T, VD, 32) X(T, VD, 48) X(T, VD, 64) X(T, VD, 96) X(T, VD, 128)
K3_DENSE_JS(K3_DENSE_MM, K3_DENSE_TYPE, 0)
K3_DENSE_JS(K3_DENSE_MM, K3_DENSE_TYPE, 1)
K3_DENSE_JS(K3_DENSE_MM, K3_DENSE_TYPE, 2)
K3_DENSE_JS(K3_DENSE_MM, K3_DENSE_TYPE, 3)

// ---- k3_dense_deq_bf16: x [rows][nb] Q6_K blocks -> y [rows][nb * 256] bf16 (candidate b of impl:q6k-dense: dequantize, then
// gemm.hip's bf16 WMMA; caminus.dense.Dense.deq_bf16, ChunkStack(q6k='wmma')). ggml's dequantize_row_q6_K expressions in fp32
// (d * sc * q), each value rounded to bf16 (RNE). One 64-thread workgroup per (row, block): thread t holds half t / 32 and
// l = t % 32, the four values l, l + 32, l + 64, l + 96 of that half. Not bitwise anything: a measured tolerance
// (tests/test_prefill_dense.py tol: the GEMM's error vs fp64 <= llama.cpp's own batched op's, its stock hipBLAS path)
static __device__ __forceinline__ uint16_t dense_bf16_rne(const float v) {
    const uint32_t u = __float_as_uint(v);
    return (uint16_t) ((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}
extern "C" __global__ void __launch_bounds__(64) k3_dense_deq_bf16(const char * __restrict__ x, uint16_t * __restrict__ y, const int nblocks) {
    const int b = blockIdx.x;
    if (b >= nblocks) return;
    const block_q6_K * bx = (const block_q6_K *) x + b;
    const int t = threadIdx.x, n = t / 32, l = t % 32, is = l / 16;
    const uint8_t * ql = bx->ql + 64*n;
    const uint8_t * qh = bx->qh + 32*n;
    const int8_t  * sc = bx->scales + 8*n;
    const float d = bx->d;
    const int q1 = (int8_t) ((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
    const int q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
    const int q3 = (int8_t) ((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
    const int q4 = (int8_t) ((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
    uint16_t * o = y + (size_t) b * QK_K + 128*n;
    o[l +  0] = dense_bf16_rne(d * sc[is + 0] * q1);
    o[l + 32] = dense_bf16_rne(d * sc[is + 2] * q2);
    o[l + 64] = dense_bf16_rne(d * sc[is + 4] * q3);
    o[l + 96] = dense_bf16_rne(d * sc[is + 6] * q4);
}

// the configs: VD, J, nthreads, I, occupancy, sram stride, qk, nbytes, K per iteration, per (VD, J)
#define K3_DENSE_CFG_WORDS 9
#define K3_DENSE_CFG(T, VD, J) {                                                                                            \
        const int v[K3_DENSE_CFG_WORDS] = {VD, J, dense_cfg<T, J, VD>::nthreads, dense_cfg<T, J, VD>::I,                    \
            dense_cfg<T, J, VD>::occupancy, dense_cfg<T, J, VD>::stride, dense_cfg<T, J, VD>::qk, dense_cfg<T, J, VD>::nbytes, \
            dense_cfg<T, J, VD>::ITER_K};                                                                                   \
        for (int w = 0; w < K3_DENSE_CFG_WORDS; ++w) out[n * K3_DENSE_CFG_WORDS + w] = v[w];                               \
        ++n; }
extern "C" __global__ void k3_dense_configs(int * out, int * count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int n = 0;
    K3_DENSE_JS(K3_DENSE_CFG, K3_DENSE_TYPE, 0) K3_DENSE_JS(K3_DENSE_CFG, K3_DENSE_TYPE, 1)
    K3_DENSE_JS(K3_DENSE_CFG, K3_DENSE_TYPE, 2) K3_DENSE_JS(K3_DENSE_CFG, K3_DENSE_TYPE, 3)
    *count = n;
}
