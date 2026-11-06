// Per-output-channel weight scale with the MSE-optimal clip ratio (tools/s1_patches.py channel_scale), fused: one
// 256-thread workgroup per row reads the row once and scores every candidate ratio in registers:
//   amax = max |w|,  for each ratio r:  s_r = (amax * r) / qmax,  err_r = sum (clamp(rint(w / s_r), -qmax-1, qmax) * s_r - w)^2
//   scale = s_r of the smallest err_r (first one on ties), as the torch reference.
// The reference sums err_r in torch's reduction order; this sums per thread then across the workgroup, so a row whose
// two best ratios tie to within rounding may pick the other one.
// Arithmetic follows torch's: dividing a tensor by the Python scalar qmax is a multiply by its fp32 reciprocal, and
// q * s - w is a multiply then a subtract (no FMA contraction).
#include <hip/hip_runtime.h>
#include <stdint.h>
#pragma clang fp contract(off)

#define NR_MAX 32

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }

template <typename T> __device__ __forceinline__ float ldw(const T * p);
template <> __device__ __forceinline__ float ldw<uint16_t>(const uint16_t * p) { return bf2f(*p); }
template <> __device__ __forceinline__ float ldw<float>(const float * p) { return *p; }

template <typename T>
__device__ void channel_scale(const T * __restrict__ w, const float * __restrict__ ratios, float * __restrict__ scale,
                              int K, int nr, int qmax) {
    __shared__ float red[8][NR_MAX + 1];
    const int row = blockIdx.x, tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
    const T * wr = w + (size_t)row * K;
    float amax = 0.f;
    for (int k = tid; k < K; k += 256) amax = fmaxf(amax, fabsf(ldw<T>(wr + k)));
    for (int m = 16; m > 0; m >>= 1) amax = fmaxf(amax, __shfl_xor(amax, m));
    if (lane == 0) red[wave][0] = amax;
    __syncthreads();
    amax = red[0][0];
    for (int i = 1; i < 8; i++) amax = fmaxf(amax, red[i][0]);
    amax = fmaxf(amax, 1e-8f);
    __syncthreads();
    float sc[NR_MAX], err[NR_MAX];
    const float inv_q = 1.0f / (float)qmax;
#pragma unroll
    for (int i = 0; i < NR_MAX; i++) { sc[i] = i < nr ? (amax * ratios[i]) * inv_q : 1.f; err[i] = 0.f; }
    for (int k = tid; k < K; k += 256) {
        const float x = ldw<T>(wr + k);
#pragma unroll
        for (int i = 0; i < NR_MAX; i++) {
            if (i < nr) {
                const float q = fminf(fmaxf(rintf(x / sc[i]), (float)(-qmax - 1)), (float)qmax);
                const float e = q * sc[i] - x;
                err[i] += e * e;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < NR_MAX; i++) {
        if (i < nr) {
            float v = err[i];
            for (int m = 16; m > 0; m >>= 1) v += __shfl_xor(v, m);
            if (lane == 0) red[wave][i] = v;
        }
    }
    __syncthreads();
    if (tid == 0) {
        int best = 0; float be = 0.f;
        for (int i = 0; i < nr; i++) {
            float e = 0.f;
            for (int wv = 0; wv < 8; wv++) e += red[wv][i];
            if (i == 0 || e < be) { be = e; best = i; }
        }
        scale[row] = (amax * ratios[best]) * (1.0f / (float)qmax);
    }
}

extern "C" {
__global__ void __launch_bounds__(256) channel_scale_bf16(const uint16_t * w, const float * ratios, float * scale, int K, int nr, int qmax) {
    channel_scale<uint16_t>(w, ratios, scale, K, nr, qmax);
}
__global__ void __launch_bounds__(256) channel_scale_f32(const float * w, const float * ratios, float * scale, int K, int nr, int qmax) {
    channel_scale<float>(w, ratios, scale, K, nr, qmax);
}
}
