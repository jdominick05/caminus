// llama.cpp's quantized matvec for formats no K3 module includes (the yardstick of docs/specs/spec-own-quant.md's G-acc
// and of benchmarks/q35_mtp_gacc.py): ggml-cuda's vec_dot on q8_1 activations, one warp of 32 per row as in
// kernels/k3_gguf.hip's gg_mv_* (the same loop), and its q8_1 quantizer. A test object of its own, so no shipped code
// object changes when a format is added here; not shipped (caminus.device.kernels).
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
