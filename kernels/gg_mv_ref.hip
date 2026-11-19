// llama.cpp's quantized matvec, the yardstick of docs/specs/spec-own-quant.md's G-acc (benchmarks/q35_mtp_gacc.py,
// tests/engine/test_k3_gguf_gsq.py (e)): ggml-cuda's vec_dot on q8_1 activations, one warp of 32 per row as in
// kernels/k3_gguf.hip's gg_mv_* (the same loop), and its q8_1 quantizer. Q4_0 (Swift 1.5's MTP block, which no K3
// module includes) and the GSQ-RCO Coder's ten formats: those are also k3_gguf.hip's gg_mv_* inside the shipped Coder
// modules, and stay here when those modules stop including the ggml headers (test_k3_gguf_gsq.py checks the two give
// the same bits until then). A test object of its own, so no shipped code object changes when a format is added
// here; not shipped (caminus.device.kernels). It compiles in ggml-cuda's
// common.cuh, vecdotq.cuh and quantize.cuh, and GG_REF_MV's loop is ggml-cuda's mul_mat_vec_q one-row loop (mmvq.cu)
// (MIT, The ggml authors; notice below and in THIRD_PARTY.md).
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
#include "common.cuh"
#include "vecdotq.cuh"
#include "quantize.cuh"

extern "C" __global__ void gg_ref_quantize_q8_1(const float * x, block_q8_1 * y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float xi = i < n ? x[i] : 0.0f;
    float d, sum;
    const int8_t q = quantize_q8_1_lane(xi, d, sum);
    y[i / QK8_1].qs[i % QK8_1] = q;
    if (i % QK8_1 == 0) y[i / QK8_1].ds = make_half2(d, sum);
}

#define GG_REF_MV(NAME, TYPE, VDR, VECDOT)                                                                              \
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
GG_REF_MV(gg_ref_mv_q4_0, GGML_TYPE_Q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1)
// the GSQ-RCO Coder's formats (the same instances as kernels/k3_gguf.hip's gg_mv_*)
GG_REF_MV(gg_ref_mv_q4_k,    GGML_TYPE_Q4_K,    VDR_Q4_K_Q8_1_MMVQ,    vec_dot_q4_K_q8_1)
GG_REF_MV(gg_ref_mv_q5_k,    GGML_TYPE_Q5_K,    VDR_Q5_K_Q8_1_MMVQ,    vec_dot_q5_K_q8_1)
GG_REF_MV(gg_ref_mv_q6_k,    GGML_TYPE_Q6_K,    VDR_Q6_K_Q8_1_MMVQ,    vec_dot_q6_K_q8_1)
GG_REF_MV(gg_ref_mv_q8_0,    GGML_TYPE_Q8_0,    VDR_Q8_0_Q8_1_MMVQ,    vec_dot_q8_0_q8_1)
GG_REF_MV(gg_ref_mv_iq4_xs,  GGML_TYPE_IQ4_XS,  VDR_IQ4_XS_Q8_1_MMVQ,  vec_dot_iq4_xs_q8_1)
GG_REF_MV(gg_ref_mv_iq4_nl,  GGML_TYPE_IQ4_NL,  VDR_IQ4_NL_Q8_1_MMVQ,  vec_dot_iq4_nl_q8_1)
GG_REF_MV(gg_ref_mv_iq3_xxs, GGML_TYPE_IQ3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1)
GG_REF_MV(gg_ref_mv_iq3_s,   GGML_TYPE_IQ3_S,   VDR_IQ3_S_Q8_1_MMVQ,   vec_dot_iq3_s_q8_1)
GG_REF_MV(gg_ref_mv_iq2_s,   GGML_TYPE_IQ2_S,   VDR_IQ2_S_Q8_1_MMVQ,   vec_dot_iq2_s_q8_1)
GG_REF_MV(gg_ref_mv_q2_0,    GGML_TYPE_Q2_0,    VDR_Q2_0_Q8_1_MMVQ,    vec_dot_q2_0_q8_1)
