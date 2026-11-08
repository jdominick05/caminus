// GGUF block formats for the K3 engine (docs/INFERENCE-PLAN.md, I2): llama.cpp's own per-format dot products
// (ggml-cuda vecdotq.cuh, MIT, included from ~/src/ignite-llama, see build.sh) on q8_1-quantized activations, so a GGUF
// cut computes in our engine what it computes in llama.cpp (same int8 activations, same per-block integer dot products;
// only the fp32 summation order over blocks can differ). These are the standalone forms that validate each format; the
// engine's matvec op will call the same device functions.
#include "common.cuh"
#include "vecdotq.cuh"
#include "quantize.cuh"

// the persistent engine with the GGUF ops (kernels/k3_gguf_ops.h) compiled in: build/k3_gguf.hsaco exports k3_run too
#define K3_EXTRA_OPS_HEADER "k3_gguf_ops.h"
#define K3_NO_G128   // no S1 int4 matvec here: its prefetch registers took k3_run from 132 to 223 VGPRs
#ifndef K3_NO_OPAQUE_TID
#define K3_OPAQUE_TID     // workitem id through an opaque asm (kernels/k3.hip): k3_run 144 VGPRs + 23 spills -> 90 + 0
#endif
#ifndef K3_MAX_VGPR
#ifdef K3_OPAQUE_TID
#define K3_MAX_VGPR 96    // 16 waves / SIMD = 8 workgroups / WGP (k3.resident_grid); the code needs 90. TP decode 15.53 ->
                          // 14.29 ms/token (at 5 / WGP the opaque id alone gives 14.98; caps 120 / 144 also land at 90)
#else
#define K3_MAX_VGPR 144   // 10 waves / SIMD (granule 24). With the multi-token ops the code wants ~165: at 120 it spilled 30
                          // (59.4 tok/s), at 144 it spills 6 (67.9), uncapped-ish 168 (65.9). Run it with K3_PER_WGP=5
#endif
#endif
#include "k3.hip"

// x [n] fp32 -> q8_1 blocks, one 32-lane wave per block (llama.cpp's quantize_q8_1_lane); n is padded to 32 by the
// caller, values past n quantize as 0
extern "C" __global__ void gg_quantize_q8_1(const float * x, block_q8_1 * y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float xi = i < n ? x[i] : 0.0f;
    float d, sum;
    const int8_t q = quantize_q8_1_lane(xi, d, sum);
    y[i / QK8_1].qs[i % QK8_1] = q;
    if (i % QK8_1 == 0) y[i / QK8_1].ds = make_half2(d, sum);
}

// one wave per row: lanes walk the row's blocks exactly as mul_mat_vec_q does (kbx = lane / (qi / vdr), stride
// vdr * 32 / qi blocks), then a wave reduction; rows are contiguous (row r starts at block r * ncols / qk)
#define GG_MV(NAME, TYPE, VDR, VECDOT)                                                                                  \
extern "C" __global__ void NAME(const void * vx, const block_q8_1 * y, float * out, int nrows, int ncols) {          \
    constexpr int qk = ggml_cuda_type_traits<TYPE>::qk, qi = ggml_cuda_type_traits<TYPE>::qi, vdr = VDR;               \
    const int row = blockIdx.x * blockDim.y + threadIdx.y;                                                             \
    if (row >= nrows) return;                                                                                          \
    const int blocks_per_row = ncols / qk;                                                                             \
    constexpr int blocks_per_iter = vdr * 32 / qi;                                                                     \
    float tmp = 0.0f;                                                                                                  \
    for (int kbx = threadIdx.x / (qi / vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {                           \
        const int kby = kbx * (qk / QK8_1);                                                                            \
        const int kqs = vdr * (threadIdx.x % (qi / vdr));                                                              \
        tmp += VECDOT(vx, &y[kby], row * blocks_per_row + kbx, kqs);                                                   \
    }                                                                                                                  \
    tmp = warp_reduce_sum<32>(tmp);                                                                                    \
    if (threadIdx.x == 0) out[row] = tmp;                                                                              \
}

GG_MV(gg_mv_iq3_xxs, GGML_TYPE_IQ3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1)
GG_MV(gg_mv_iq4_nl,  GGML_TYPE_IQ4_NL,  VDR_IQ4_NL_Q8_1_MMVQ,  vec_dot_iq4_nl_q8_1)
GG_MV(gg_mv_q6_k,    GGML_TYPE_Q6_K,    VDR_Q6_K_Q8_1_MMVQ,    vec_dot_q6_K_q8_1)
GG_MV(gg_mv_q8_0,    GGML_TYPE_Q8_0,    VDR_Q8_0_Q8_1_MMVQ,    vec_dot_q8_0_q8_1)
#ifdef K3_GSQ   // the GSQ-RCO Coder's formats (kernels/k3_gguf_gsq.hip; the REAP-308 code object stays as it was)
GG_MV(gg_mv_q4_k,    GGML_TYPE_Q4_K,    VDR_Q4_K_Q8_1_MMVQ,    vec_dot_q4_K_q8_1)
GG_MV(gg_mv_q5_k,    GGML_TYPE_Q5_K,    VDR_Q5_K_Q8_1_MMVQ,    vec_dot_q5_K_q8_1)
GG_MV(gg_mv_iq4_xs,  GGML_TYPE_IQ4_XS,  VDR_IQ4_XS_Q8_1_MMVQ,  vec_dot_iq4_xs_q8_1)
GG_MV(gg_mv_iq2_s,   GGML_TYPE_IQ2_S,   VDR_IQ2_S_Q8_1_MMVQ,   vec_dot_iq2_s_q8_1)
GG_MV(gg_mv_iq3_s,   GGML_TYPE_IQ3_S,   VDR_IQ3_S_Q8_1_MMVQ,   vec_dot_iq3_s_q8_1)
GG_MV(gg_mv_q2_0,    GGML_TYPE_Q2_0,    VDR_Q2_0_Q8_1_MMVQ,    vec_dot_q2_0_q8_1)
#endif
#ifdef K3_MV_Q3K   // kernels/k3_gguf_q35h.hip only: Q3_K's standalone matvec (tests/test_q36_formats.py, caminus.gguf.matvec)
GG_MV(gg_mv_q3_k,    GGML_TYPE_Q3_K,    VDR_Q3_K_Q8_1_MMVQ,    vec_dot_q3_K_q8_1)
#endif
#ifdef K3_GSQ
// h[i] = silu(g[i]) * u[i] with the engine's own gg_silu: the standalone composition of OP_GG_MOE_GU (tests)
extern "C" __global__ void gg_silu_mul(const float * g, const float * u, float * h, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = gg_silu(g[i]) * u[i];
}
#endif
