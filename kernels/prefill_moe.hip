// Batched prefill, the routed experts' grouped GEMMs of our own (project decision 11-03: prefill GEMM kernels of our own;
// docs/agentic/spec-prefill-cache.md). Replaces the vendored MMQ's mul_mat_id for the MoE gate / up / down projections:
//
//   k3_moe_group   (-DK3_MOE_GLUE) the grouping, ONE launch per layer and micro-chunk: the router's ids (clamped in range and
//                  distinct per token by k3_mmq_ids_clamp first, on the same stream) -> expert_bounds, ids_dst (compact
//                  column -> t * k + slot) and the inverse map (t * k + slot -> compact column): the SAME tables as ggml's
//                  mm_ids_helper (experts ascending, tokens ascending within an expert), plus the flat tile lists the GEMMs
//                  run over (expert, first compact column, columns <= J). One workgroup: per-expert counts (integer LDS
//                  atomics: order-free), an exclusive scan, then the tokens 64 at a time, each (token, slot) placed at
//                  start[e] + popcount(the chunk's earlier tokens of e) from a 64-bit LDS mask per expert (deterministic).
//                  mm_ids_helper runs one wave per expert over every id (0.36 ms per call at 2048 tokens on the XT, measured
//                  11-04; ChunkStack called it twice per layer); this is one call.
//   k3_moe_mm_<J>  (-DK3_MOE_TYPE=<ggml_type>) one quantized GEMM over a tile list (the down projection): every workgroup one
//                  (tile, row block of I rows); no empty workgroups (MMQ's mul_mat_id grid is n_exp x ceil(ntok / J) x
//                  rows / I, and at real routing ~90 % of it exits at once: 225K workgroups for one down projection at
//                  2048 tokens). The K loop stops at K: the card-0 down shards (K = 384) skip the half-iteration MMQ runs
//                  on the zero-padded activations. A tile's 16-column groups past its columns are skipped (moe_vec_dot).
//   k3_moe_gu_<J>  the gate and up projections of the same tile in one workgroup (one activation tile in LDS for both; the
//                  gate's weight tile, then the up's, through the same LDS: K3_MOE_GU_SEQ),
//                  then swiglu (pf_swiglu's gg_silu(g) * u, contract off) and the q8_1 quantization of the down
//                  projection's input (quantize_mmq_q8_1's D4 expressions) straight into its compact column: the gate /
//                  up compact column j is the down's compact column j (the same grouping), so og / ou / the swiglu pass /
//                  the down's gather-quantize and its second grouping are gone.
//
// Numerics: BITWISE ggml's MMQ per output. The per-format dequantization is the vendored load_tiles itself (kernels/ggml_mmq,
// unchanged) at the same (type, J) configs; the tile product (moe_vec_dot) repeats the vendored vec_dot's per-output
// operations (per sub-block in K order one int WMMA, then sum += C * dA * dB with the same expression, so the same fp32
// contraction), so an output's bits depend only on its weight row and its activation column (MMQ's own property, G2 in
// tests/test_prefill_mmq.py: any J gives the same bits). What is ours: the tile list, the K stop (the skipped half adds C = 0
// times dB = 0 to every sum, i.e. +0 to a sum that is never -0: it starts at +0.0f and round-to-nearest never makes -0 from
// +0 by adding), the skipped empty 16-column groups (sums never written), the gate / up fusion, and the epilogue. The
// process-tile loop and moe_vec_dot are modified copies of ggml-cuda mmq.cuh's mul_mat_q_process_tile and
// mmq-vec-dot.cuh's vec_dot_q8_0_q8_1_mma / vec_dot_q8_0_16_q8_1_mma (MIT, The ggml authors; notice below and in
// THIRD_PARTY.md); -DK3_MOE_VD_MMQ=1 builds the kernels on the vendored vec_dot itself, for A/Bs.
// Measured 11-04 (XT, real routing, n 1024, per layer of the MoE path incl. its grouping / quantize / swiglu): the MMQ path
// 2.62-4.26 ms, these kernels 1.55-2.82 ms (1.51-1.69x; 1.57-2.00x at n 512); of that, the tile list + one grouping + the fused
// epilogue gave 1.23-1.50x, the 16-column skip and the shared activation fragment 1.41-1.79x, the one-weight-tile LDS layout
// (K3_MOE_GU_SEQ) with J 64 the rest. Where the fused kernel's time goes (variants with parts skipped, same accesses): the tile
// product and its activation loads ~80 %, the weight loads + dequantization ~35 % alone (partly overlapped); unrolling the
// sub-block loop changed nothing (<= 3 %). Compiled per format by build.sh like kernels/prefill_mmq.hip (ignite's flags, a full HIP
// compile, the gfx1100 object unbundled): build/prefill_moe_<format>.hsaco and build/prefill_moe_glue.hsaco.
//
// Safety: the host (caminus/moe.py) refuses any launch whose buffers do not cover what it touches (caminus.bufcheck views,
// no overlapping written buffers). On the device every table entry is checked before use: a tile whose expert, columns or
// rows fall outside the launch's ranges is skipped and sets the routing flag word (ROUTE_TILES), and a destination row id
// outside [0, ntok * k) is never written through (redirected to row 0 and flagged), so no table state, stale or not, can
// make a GEMM write outside its destination.
//
// ---- ggml-cuda (llama.cpp) notice for the copied process-tile loop -----------------------------------------------------
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

#if defined(K3_MOE_TYPE) == defined(K3_MOE_GLUE)
#error "prefill_moe.hip: build with exactly one of -DK3_MOE_TYPE=<ggml_type> or -DK3_MOE_GLUE (build.sh)"
#endif

#define K3_ROUTE_RANGE 2    // caminus.prefill_ops ROUTE_RANGE: an expert id out of [0, n_exp)
#define K3_ROUTE_DUP   4    // ROUTE_DUP: an expert id repeated within a token
#define K3_ROUTE_TILES 8    // ROUTE_TILES: a grouping / tile table entry out of its range (never expected: the tables are ours)
#define K3_MOE_EMAX 512     // experts per layer the grouping kernel holds in LDS (the Coder: 256)
#define K3_MOE_TCHUNK 64    // tokens per placement round (one 64-bit mask word per expert)

#if defined(K3_MOE_TYPE)

#define K3_MOE_PREC GGML_PREC_Q8

// the fused kernel's LDS: 1 (default): ONE weight tile, gate then up through the same LDS, with both 128-value activation
// halves of the iteration resident; 0: both weight tiles + one activation half. Measured 11-04 on the XT (real routing, n 512 /
// 1024, medians of 9; outputs bitwise equal): 1 is 0-5 % faster on IQ3_XXS and lets IQ2_S use its J 64 config (256 threads,
// 128 rows: 61.7 KB, which two weight tiles do not fit), 1.447 -> 1.365 ms (card 1's shapes) and 2.122 -> 1.959 (card 0's) at
// n 1024
#ifndef K3_MOE_GU_SEQ
#define K3_MOE_GU_SEQ 1
#endif

template <ggml_type type, int J> struct moe_cfg {
    static constexpr int warp_size = 32;
    static constexpr int nthreads  = ggml_cuda_mmq_get_config_rdna3(type, J, false).nthreads;
    static constexpr int nwarps    = nthreads / warp_size;
    static constexpr int I         = ggml_cuda_mmq_get_config_rdna3(type, J, false).I;
    static constexpr int occupancy = ggml_cuda_mmq_get_config_rdna3(type, J, false).occupancy;
    static constexpr int stride    = ggml_cuda_mmq_get_sram_stride(ggml_cuda_mmq_get_config_rdna3(type, J, false).sram_layout);
    static constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    static constexpr int ITER_K    = ggml_cuda_mmq_get_config_rdna3(type, J, false).K_vram;
    static constexpr int bpi       = ITER_K / qk;                       // weight blocks per K iteration
    static constexpr int ytile     = GGML_PAD(J * MMQ_TILE_Y_K, nthreads); // ints: one 128-value activation half-tile
    static constexpr int xtile     = I * stride;                        // ints: one weight tile (the MMA data layout)
    static constexpr int nbytes_mm = 4 * (J + ytile + xtile);
    // the fused kernel: K3_MOE_GU_SEQ 0: both weight tiles + one activation half-tile; 1: ONE weight tile (gate, then up, through
    // the same LDS) + both activation halves (less LDS: more workgroups per WGP)
    static constexpr int nbytes_gu = K3_MOE_GU_SEQ ? 4 * (J + 2 * ytile + xtile) : 4 * (J + ytile + 2 * xtile);
    static_assert(I == nwarps * 16, "vec_dot's AMD WMMA path covers 16 rows per warp");
    // the fused kernel: two weight tiles of 64 rows + the activation tile within 64 KB of LDS
    static constexpr bool gu_ok    = (I == 64 || K3_MOE_GU_SEQ) && nbytes_gu <= 65536 && J <= 64;   // (J 96 / 128 spill: 16 / 191 VGPRs)
    static_assert(ITER_K == 2 * QK8_1_MMQ, "K_vram 256: two 128-value activation halves per iteration");
};

// ---- the tile product: ggml's vec_dot_q8_0_q8_1_mma (D4) / vec_dot_q8_0_16_q8_1_mma for AMD WMMA, per output the same --------
// operations in the same order (per sub-block of SUB 32-bit words in K order: one int WMMA, then sum += C * dA * dB with dA the
// weight row's sub-block scale and dB the activation column's 32-value scale), with two changes that touch no output's
// arithmetic: the 16-column groups at or past jn (the tile's columns rounded up to 16) are skipped (their sums stay 0 and are
// never written), and moe_vec_dot2 runs the gate and the up tile against the same activation fragment (one B load, two
// independent WMMA chains). Which ggml variant a format uses is asserted against ggml's own table (moe_sub_words)
#ifndef K3_MOE_KUNROLL
#define K3_MOE_KUNROLL 1     // the sub-block loop's unroll
#endif

template <ggml_type type, int J> static constexpr __device__ int moe_sub_words() {
    constexpr ggml_cuda_mmq_vec_dot_t vd = ggml_cuda_mmq_get_vec_dot<type, J, false, GGML_PREC_Q8>();
    constexpr bool w16 = vd == ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, J, false, GGML_PREC_Q8>;
    constexpr bool w32 = vd == ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, false, MMQ_Q8_1_DS_LAYOUT_D4>;
    static_assert(w16 != w32, "prefill_moe: a format whose MMQ vec_dot is neither q8_0_16 nor q8_0 (D4)");
    return w16 ? 4 : QI8_0;     // 32-bit words per scale sub-block: 16 or 32 values
}

template <ggml_type type, int J, int NM>
static __device__ __forceinline__ void moe_vec_dot(const int * __restrict__ x0, const int * __restrict__ x1, const int * __restrict__ y,
                                                   float * __restrict__ s0, float * __restrict__ s1, const int k00, const int jn) {
#if defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    constexpr int SUB = moe_sub_words<type, J>();
    typedef tile<16, SUB, int, input_layout>        tile_A;
    typedef tile<16, SUB, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, false, GGML_PREC_Q8);
    static_assert(ggml_cuda_mmq_get_rows_per_warp(type, J, false, GGML_PREC_Q8) == tile_C::I, "one 16-row minitile per warp");

    const int   * x0_qs = x0;
    const float * x0_df = (const float *) x0 + 2*MMQ_TILE_NE_K;
    const int   * x1_qs = x1;
    const float * x1_df = (const float *) x1 + 2*MMQ_TILE_NE_K;
    const int   * y_qs  = y + 4;
    const float * y_df  = (const float *) y;
    const int i0 = threadIdx.y * tile_C::I;

#pragma unroll K3_MOE_KUNROLL
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += SUB) {
        const int k0 = k00 + k01;
        tile_A A0, A1;
        load_ldmatrix(A0, x0_qs + i0*sram_stride + k0, sram_stride);
        if constexpr (NM == 2) load_ldmatrix(A1, x1_qs + i0*sram_stride + k0, sram_stride);
#pragma unroll
        for (int j0 = 0; j0 < J; j0 += tile_C::J) {
            if (j0 >= jn) break;                                    // (uniform over the workgroup)
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
            const int j = j0 + tile_C::get_j(0);
            const float dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            tile_C C0, C1;
            mma(C0, A0, B);
            if constexpr (NM == 2) mma(C1, A1, B);
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + tile_C::get_i(l);
                const float dA0 = x0_df[i*sram_stride + k0/SUB];
                s0[(j0/tile_C::J)*tile_C::ne + l] += C0.x[l]*dA0*dB;
                if constexpr (NM == 2) {
                    const float dA1 = x1_df[i*sram_stride + k0/SUB];
                    s1[(j0/tile_C::J)*tile_C::ne + l] += C1.x[l]*dA1*dB;
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x0, x1, y, s0, s1, k00, jn);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE)
}

#ifndef K3_MOE_VD_MMQ
#define K3_MOE_VD_MMQ 0      // 1: the vendored vec_dot itself (no column skip, no shared fragment), for A/Bs
#endif

// a tile list entry: (expert, first compact column, columns, 0)
static __device__ __forceinline__ bool moe_tile_ok(const int4 tl, const int n_exp, const int ncols, const int J) {
    return tl.x >= 0 && tl.x < n_exp && tl.y >= 0 && tl.z >= 1 && tl.z <= J && tl.y + tl.z <= ncols;
}

// the destination row ids of a tile's columns into LDS (MMQ's ids_dst_shared), each checked: an id outside [0, ndst) is never
// written through (row 0 instead, flagged; the clamped grouping never produces one)
template <int J, int nthreads> static __device__ __forceinline__ void moe_load_ids(int * ids_sh, const int * __restrict__ ids_dst,
                                                                               const int c0, const int nc, const int ndst, int * err) {
    for (int j = threadIdx.y * 32 + threadIdx.x; j < J; j += nthreads) {
        int v = 0;
        if (j < nc) {
            v = ids_dst[c0 + j];
            if (v < 0 || v >= ndst) { atomicOr(err, K3_ROUTE_TILES); v = 0; }
        }
        ids_sh[j] = v;
    }
}

// one K iteration's activation half-tile (128 values of J compact columns) into LDS: mul_mat_q_process_tile's copy loop
template <ggml_type type, int J> static __device__ __forceinline__ void moe_load_y(int * tile_y, const int * __restrict__ by0) {
    using C = moe_cfg<type, J>;
#pragma unroll
    for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += C::nthreads) {
        const int l = l0 + threadIdx.y * C::warp_size + threadIdx.x;
        tile_y[l] = by0[l];
    }
}

// ---- k3_moe_mm: one GEMM over a tile list --------------------------------------------------------------------------------
// x: [n_exp][rows][K] raw blocks (K = nb * qk); y: block_q8_1_mmq [K_pad / 128][ncols_y] (compact columns); tiles / ntiles: the
// list (k3_moe_group); dst[ids_dst[c] * rows + r] for every compact column c of a tile and every row r of the block.
// Grid (rows / I, max tiles); a workgroup past ntiles exits at once
template <ggml_type type, int J>
static __device__ __forceinline__ void moe_mm_body(const char * __restrict__ x, const int * __restrict__ y, const int * __restrict__ ids_dst,
        const int4 * __restrict__ tiles, const int * __restrict__ ntiles, float * __restrict__ dst, int * err,
        const int n_exp, const int rows, const int nb, const int K, const int ncols_y, const int ndst) {
    using C = moe_cfg<type, J>;
    constexpr ggml_cuda_mmq_load_tiles_t load_tiles = ggml_cuda_mmq_get_load_tiles<type, J, false, K3_MOE_PREC>();
    constexpr ggml_cuda_mmq_vec_dot_t    vec_dot    = ggml_cuda_mmq_get_vec_dot<type, J, false, K3_MOE_PREC>();
    constexpr ggml_cuda_mmq_write_back_t write_back = ggml_cuda_mmq_get_write_back<type, J, false, K3_MOE_PREC>();
    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);

    if ((int) blockIdx.y >= *ntiles) return;
    const int4 tl = tiles[blockIdx.y];
    const int it = blockIdx.x;
    if (!moe_tile_ok(tl, n_exp, ncols_y, J) || (it + 1) * C::I > rows) {
        if (threadIdx.x == 0 && threadIdx.y == 0) atomicOr(err, K3_ROUTE_TILES);
        return;
    }
    const int e = tl.x, c0 = tl.y, nc = tl.z;

    extern __shared__ int data_mul_mat_q[];
    int * ids_sh = data_mul_mat_q;
    int * tile_y = data_mul_mat_q + J;
    int * tile_x = tile_y + C::ytile;
    moe_load_ids<J, C::nthreads>(ids_sh, ids_dst, c0, nc, ndst, err);
    __syncthreads();

    const int offset_x = (e * rows + it * C::I) * nb;           // in blocks
    const int * yt = y + c0 * sz;
    float sum[J * C::I / C::nthreads] = {0.0f};

    const int jn = (nc + 15) / 16 * 16;
    for (int kb0 = 0; kb0 < nb; kb0 += C::bpi) {
        load_tiles(x, tile_x, offset_x + kb0, C::I - 1, nb);
        moe_load_y<type, J>(tile_y, yt + ncols_y * (kb0 * C::qk / QK8_1_MMQ) * sz);
        __syncthreads();
        if (K3_MOE_VD_MMQ) vec_dot(tile_x, tile_y, sum, 0); else moe_vec_dot<type, J, 1>(tile_x, tile_x, tile_y, sum, sum, 0, jn);
        __syncthreads();
        if (kb0 * C::qk + QK8_1_MMQ < K) {          // the second 128 values of this iteration lie inside K (MMQ runs it anyway)
            moe_load_y<type, J>(tile_y, yt + ncols_y * ((kb0 * C::qk / QK8_1_MMQ) * sz + sz));
            __syncthreads();
            if (K3_MOE_VD_MMQ) vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K); else moe_vec_dot<type, J, 1>(tile_x, tile_x, tile_y, sum, sum, MMQ_TILE_NE_K, jn);
            __syncthreads();
        }
    }
    write_back(sum, ids_sh, dst + it * C::I, nullptr, rows, C::I - 1, nc - 1);
}

// ---- k3_moe_gu: gate + up of a tile, swiglu, the down input quantized into its compact columns -------------------------------
// xg / xu: [n_exp][rows][K] (rows = the card's ffc slice); y: the gate / up activations [K_pad / 128][ncols_y]; yd: the down
// projection's activations, block_q8_1_mmq [kd_pad / 128][ncols_y] (D4: a float scale per 32 values; kd = rows): for a
// tile's column c and row block it, the values it * I .. it * I + I - 1 of compact column c
template <ggml_type type, int J>
static __device__ __forceinline__ void moe_gu_body(const char * __restrict__ xg, const char * __restrict__ xu, const int * __restrict__ y,
        const int4 * __restrict__ tiles, const int * __restrict__ ntiles, block_q8_1_mmq * __restrict__ yd, int * err,
        const int n_exp, const int rows, const int nb, const int ncols_y) {
    using C = moe_cfg<type, J>;
    constexpr ggml_cuda_mmq_load_tiles_t load_tiles = ggml_cuda_mmq_get_load_tiles<type, J, false, K3_MOE_PREC>();
    constexpr ggml_cuda_mmq_vec_dot_t    vec_dot    = ggml_cuda_mmq_get_vec_dot<type, J, false, K3_MOE_PREC>();
    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);
    static_assert(4 * J * C::I <= C::nbytes_gu - 4 * J, "the epilogue's [J][I] fp32 tile lives in the tiles' LDS");
    static_assert(C::I % 32 == 0, "a row block holds whole 32-value quantization groups");

    if ((int) blockIdx.y >= *ntiles) return;
    const int4 tl = tiles[blockIdx.y];
    const int it = blockIdx.x;
    if (!moe_tile_ok(tl, n_exp, ncols_y, J) || (it + 1) * C::I > rows) {
        if (threadIdx.x == 0 && threadIdx.y == 0) atomicOr(err, K3_ROUTE_TILES);
        return;
    }
    const int e = tl.x, c0 = tl.y, nc = tl.z;

    extern __shared__ int data_mul_mat_q[];
    int * tile_y  = data_mul_mat_q + J;
    int * tile_xg = tile_y + (K3_MOE_GU_SEQ ? 2 : 1) * C::ytile;
    int * tile_xu = K3_MOE_GU_SEQ ? tile_xg : tile_xg + C::xtile;

    const int offset_x = (e * rows + it * C::I) * nb;
    const int * yt = y + c0 * sz;
    float sg[J * C::I / C::nthreads] = {0.0f};
    float su[J * C::I / C::nthreads] = {0.0f};
    const int jn = (nc + 15) / 16 * 16;

    for (int kb0 = 0; kb0 < nb && K3_MOE_GU_SEQ; kb0 += C::bpi) {   // one weight tile at a time, both activation halves
        int * tile_y1 = tile_y + C::ytile;
        load_tiles(xg, tile_xg, offset_x + kb0, C::I - 1, nb);
        moe_load_y<type, J>(tile_y,  yt + ncols_y * (kb0 * C::qk / QK8_1_MMQ) * sz);
        moe_load_y<type, J>(tile_y1, yt + ncols_y * ((kb0 * C::qk / QK8_1_MMQ) * sz + sz));
        __syncthreads();
        moe_vec_dot<type, J, 1>(tile_xg, tile_xg, tile_y,  sg, sg, 0, jn);
        moe_vec_dot<type, J, 1>(tile_xg, tile_xg, tile_y1, sg, sg, MMQ_TILE_NE_K, jn);
        __syncthreads();
        load_tiles(xu, tile_xg, offset_x + kb0, C::I - 1, nb);
        __syncthreads();
        moe_vec_dot<type, J, 1>(tile_xg, tile_xg, tile_y,  su, su, 0, jn);
        moe_vec_dot<type, J, 1>(tile_xg, tile_xg, tile_y1, su, su, MMQ_TILE_NE_K, jn);
        __syncthreads();
    }
    for (int kb0 = 0; kb0 < nb && !K3_MOE_GU_SEQ; kb0 += C::bpi) {
        load_tiles(xg, tile_xg, offset_x + kb0, C::I - 1, nb);
        load_tiles(xu, tile_xu, offset_x + kb0, C::I - 1, nb);
        moe_load_y<type, J>(tile_y, yt + ncols_y * (kb0 * C::qk / QK8_1_MMQ) * sz);
        __syncthreads();
        if (K3_MOE_VD_MMQ) { vec_dot(tile_xg, tile_y, sg, 0); vec_dot(tile_xu, tile_y, su, 0); }
        else moe_vec_dot<type, J, 2>(tile_xg, tile_xu, tile_y, sg, su, 0, jn);
        __syncthreads();
        moe_load_y<type, J>(tile_y, yt + ncols_y * ((kb0 * C::qk / QK8_1_MMQ) * sz + sz));
        __syncthreads();
        if (K3_MOE_VD_MMQ) { vec_dot(tile_xg, tile_y, sg, MMQ_TILE_NE_K); vec_dot(tile_xu, tile_y, su, MMQ_TILE_NE_K); }
        else moe_vec_dot<type, J, 2>(tile_xg, tile_xu, tile_y, sg, su, MMQ_TILE_NE_K, jn);
        __syncthreads();
    }

    // swiglu into LDS, h[j][i] (the C tiles' elements as ggml_cuda_mmq_write_back_mma maps them)
    float * hb = (float *) tile_y;   // (every tile is dead after the last barrier)
    {
#pragma clang fp contract(off)
        typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;
        constexpr int ntx = 16 / tile_C::I;                         // rows_per_warp (16 on AMD) / 16
        const int i0 = (threadIdx.y / ntx) * (ntx * tile_C::I);
#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
#pragma unroll
            for (int n = 0; n < ntx; ++n) {
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                    const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                    const int s = (j0 / tile_C::J + n) * tile_C::ne + l;
                    const float g = sg[s], u = su[s];
                    const float hv = (g / (1.0f + expf(-g))) * u;  // pf_swiglu: gg_silu(g) * u (ggml-cuda op_silu)
                    hb[j * C::I + i] = hv;
                }
            }
        }
    }
    __syncthreads();

    // quantize_mmq_q8_1 (D4) of h along the rows: 4 values per thread, a 32-value group on 8 aligned lanes (the quantizer's
    // shuffle pattern); every column of the tile takes part, only the tile's nc columns are written
    {
        constexpr int per_col = C::I / 4;
#pragma unroll
        for (int q0 = 0; q0 < J * per_col; q0 += C::nthreads) {
            const int q = q0 + threadIdx.y * C::warp_size + threadIdx.x;
            const int j = q / per_col, i4 = (q % per_col) * 4;
            const float4 xi = *(const float4 *) (hb + j * C::I + i4);
            float amax = fabsf(xi.x);
            amax = fmaxf(amax, fabsf(xi.y));
            amax = fmaxf(amax, fabsf(xi.z));
            amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
            for (int offset = 32 / 8; offset > 0; offset >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
            }
            const float d_inv = 127.0f / amax;
            char4 qv;
            qv.x = roundf(xi.x * d_inv);
            qv.y = roundf(xi.y * d_inv);
            qv.z = roundf(xi.z * d_inv);
            qv.w = roundf(xi.w * d_inv);
            const float d = 1.0f / d_inv;
            if (j < nc) {
                const int r = it * C::I + i4;                       // the down input's value index (its K)
                block_q8_1_mmq * b = yd + (int64_t) (r / QK8_1_MMQ) * ncols_y + (c0 + j);
                const int iqs = r % QK8_1_MMQ;
                ((char4 *) b->qs)[iqs / 4] = qv;
                if (iqs % 32 == 0) b->d4[iqs / 32] = d;
            }
        }
    }
}

// ---- the kernels, per J ------------------------------------------------------------------------------------------------
#define K3_MOE_MM(T, J)                                                                                                     \
extern "C" __global__ void __launch_bounds__((moe_cfg<T, J>::nthreads), (moe_cfg<T, J>::occupancy)) k3_moe_mm_##J(               \
        const char * x, const int * y, const int * ids_dst, const int4 * tiles, const int * ntiles, float * dst, int * err,  \
        int n_exp, int rows, int nb, int K, int ncols_y, int ndst) {                                                        \
    moe_mm_body<T, J>(x, y, ids_dst, tiles, ntiles, dst, err, n_exp, rows, nb, K, ncols_y, ndst);                          \
}
#define K3_MOE_GU(T, J)                                                                                                     \
extern "C" __global__ void __launch_bounds__((moe_cfg<T, J>::nthreads), (moe_cfg<T, J>::occupancy)) k3_moe_gu_##J(               \
        const char * xg, const char * xu, const int * y, const int4 * tiles, const int * ntiles, block_q8_1_mmq * yd,       \
        int * err, int n_exp, int rows, int nb, int ncols_y) {                                                              \
    if constexpr (moe_cfg<T, J>::gu_ok) moe_gu_body<T, J>(xg, xu, y, tiles, ntiles, yd, err, n_exp, rows, nb, ncols_y);    \
}
K3_MOE_MM(K3_MOE_TYPE, 16) K3_MOE_MM(K3_MOE_TYPE, 32) K3_MOE_MM(K3_MOE_TYPE, 48) K3_MOE_MM(K3_MOE_TYPE, 64)
K3_MOE_MM(K3_MOE_TYPE, 96) K3_MOE_MM(K3_MOE_TYPE, 128)
// the fused kernel has code only where the config's row block is 64 and its LDS fits (moe_cfg::gu_ok; k3_moe_configs reports it)
K3_MOE_GU(K3_MOE_TYPE, 16) K3_MOE_GU(K3_MOE_TYPE, 32) K3_MOE_GU(K3_MOE_TYPE, 48) K3_MOE_GU(K3_MOE_TYPE, 64)
K3_MOE_GU(K3_MOE_TYPE, 96) K3_MOE_GU(K3_MOE_TYPE, 128)

// the configs: J, nthreads, I, occupancy, sram stride, qk, nbytes (mm), nbytes (gu, 0 = not built), K per iteration, per J
#define K3_MOE_CFG_WORDS 9
#define K3_MOE_CFG(T, J) {                                                                                              \
        const int v[K3_MOE_CFG_WORDS] = {J, moe_cfg<T, J>::nthreads, moe_cfg<T, J>::I, moe_cfg<T, J>::occupancy,             \
            moe_cfg<T, J>::stride, moe_cfg<T, J>::qk, moe_cfg<T, J>::nbytes_mm, moe_cfg<T, J>::gu_ok ? moe_cfg<T, J>::nbytes_gu : 0,           \
            moe_cfg<T, J>::ITER_K};                                                                                         \
        for (int w = 0; w < K3_MOE_CFG_WORDS; ++w) out[n * K3_MOE_CFG_WORDS + w] = v[w];                                   \
        ++n; }
extern "C" __global__ void k3_moe_configs(int * out, int * count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int n = 0;
    K3_MOE_CFG(K3_MOE_TYPE, 16) K3_MOE_CFG(K3_MOE_TYPE, 32) K3_MOE_CFG(K3_MOE_TYPE, 48)
    K3_MOE_CFG(K3_MOE_TYPE, 64) K3_MOE_CFG(K3_MOE_TYPE, 96) K3_MOE_CFG(K3_MOE_TYPE, 128)
    *count = n;
}

#else   // K3_MOE_GLUE

// ---- k3_moe_group ------------------------------------------------------------------------------------------------------
// ids [ntok][k] (clamped: in [0, n_exp), distinct per token; an id that is not is skipped and flagged, never placed) ->
// bounds [n_exp + 1], ids_dst [ntok * k] (compact -> t * k + slot), ids_inv [ntok * k] (t * k + slot -> compact), and two
// tile lists (column tiles of ja / jb columns per expert, in expert order: (e, first compact column, columns, 0)), their
// counts in ntiles[0] / ntiles[1] (a list longer than its buffer's max_a / max_b entries is cut there and flagged; the host
// sizes both to the bound n_exp + ceil(ntok * k / J), so it never is). One workgroup of 1024 threads
extern "C" __global__ void __launch_bounds__(1024) k3_moe_group(const int * ids, int ntok, int k, int n_exp, int * bounds, int * ids_dst,
        int * ids_inv, int4 * tiles_a, int ja, int max_a, int4 * tiles_b, int jb, int max_b, int * ntiles, int * err) {
    __shared__ int cnt[K3_MOE_EMAX];
    __shared__ int start[K3_MOE_EMAX];
    __shared__ unsigned long long mask[K3_MOE_EMAX];
    __shared__ int toff[2][K3_MOE_EMAX];
    const int tid = threadIdx.x;
    if (n_exp < 1 || n_exp > K3_MOE_EMAX || k < 1 || k * K3_MOE_TCHUNK > 1024 || ntok < 1 || ja < 1 || jb < 1) {
        if (tid == 0) atomicOr(err, K3_ROUTE_TILES);   // (the host refuses those shapes before launching)
        return;
    }
    const int N = ntok * k;
    int flag = 0;
    for (int e = tid; e < n_exp; e += 1024) { cnt[e] = 0; mask[e] = 0ull; }
    __syncthreads();
    for (int i = tid; i < N; i += 1024) {
        const int e = ids[i];
        if (e >= 0 && e < n_exp) atomicAdd(&cnt[e], 1); else flag |= K3_ROUTE_RANGE;
    }
    __syncthreads();
    if (tid == 0) {                                     // exclusive scans: the bounds and both lists' offsets
        int s = 0, sa = 0, sb = 0;
        for (int e = 0; e < n_exp; ++e) {
            start[e] = s; bounds[e] = s; toff[0][e] = sa; toff[1][e] = sb;
            s += cnt[e]; sa += (cnt[e] + ja - 1) / ja; sb += (cnt[e] + jb - 1) / jb;
        }
        bounds[n_exp] = s;
        if (sa > max_a || sb > max_b) flag |= K3_ROUTE_TILES;
        ntiles[0] = min(sa, max_a); ntiles[1] = min(sb, max_b);
    }
    __syncthreads();
    // placement, K3_MOE_TCHUNK tokens a round: (token, slot) -> start[e] + the number of this round's earlier tokens of e
    for (int t0 = 0; t0 < ntok; t0 += K3_MOE_TCHUNK) {
        const int t = t0 + tid / k, s = tid % k;
        const bool mine = tid < K3_MOE_TCHUNK * k && t < ntok;
        int e = -1;
        unsigned long long bit = 0ull;
        if (mine) {
            e = ids[t * k + s];
            if (e >= 0 && e < n_exp) {
                bit = 1ull << (t - t0);
                const unsigned long long old = atomicOr(&mask[e], bit);
                if (old & bit) { flag |= K3_ROUTE_DUP; e = -1; }   // a second slot of this token with the same expert
            } else e = -1;
        }
        __syncthreads();
        if (e >= 0) {
            const int pos = start[e] + __popcll(mask[e] & (bit - 1ull));
            if (pos >= 0 && pos < N) { ids_dst[pos] = t * k + s; ids_inv[t * k + s] = pos; }
            else flag |= K3_ROUTE_TILES;
        }
        __syncthreads();
        for (int e2 = tid; e2 < n_exp; e2 += 1024) { start[e2] += __popcll(mask[e2]); mask[e2] = 0ull; }
        __syncthreads();
    }
    // the tile lists
    for (int e = tid; e < n_exp; e += 1024) {
        const int c = cnt[e], b0 = bounds[e];
        for (int j = 0; j * ja < c; ++j) {
            const int o = toff[0][e] + j;
            if (o < max_a) tiles_a[o] = make_int4(e, b0 + j * ja, min(ja, c - j * ja), 0);
        }
        for (int j = 0; j * jb < c; ++j) {
            const int o = toff[1][e] + j;
            if (o < max_b) tiles_b[o] = make_int4(e, b0 + j * jb, min(jb, c - j * jb), 0);
        }
    }
    if (flag) atomicOr(err, flag);
}

#endif
