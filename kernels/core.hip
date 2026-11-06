// Elementwise and row kernels for the S1 core forward in the C++ runtime. Each reproduces the PyTorch op the
// training forward runs under bf16 autocast:
//   embedding_f32   out[t, :] = emb[tok[t], :]                                        (exact: a gather)
//   rmsnorm_f32     out = x * rsqrt(mean(x^2) + eps) * w, fp32 (torch's _fused_rms_norm; its reduction order is
//                   internal to libtorch, so this matches to rounding, not bit for bit)
//   silu_mul_bf16   act = bf16(bf16(silu(g)) * u) over gu = [g | u] (torch: silu then mul, each in fp32 rounded
//                   back to bf16)
//   residual3       x_out = (x + m) + y with x fp32 and m, y bf16 (torch's promotion: fp32 adds, left to right)
//   cast_bf16       out = bf16(x) (autocast's cast of an fp32 activation, round to nearest even)
//   copy_cols_bf16  out[r, c] = in[r, c0 + c] for c < n (torch.cat / .contiguous() of a column block)
#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }

__device__ __forceinline__ uint16_t f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

extern "C" __global__ void embedding_f32(const int * __restrict__ tok, const float * __restrict__ emb,
                                         float * __restrict__ out, int D) {
    const int t = blockIdx.x;
    const float * src = emb + (size_t)tok[t] * D;
    for (int j = threadIdx.x; j < D; j += blockDim.x) out[(size_t)t * D + j] = src[j];
}

extern "C" __global__ void __launch_bounds__(256)
rmsnorm_f32(const float * __restrict__ x, const float * __restrict__ w, float * __restrict__ out, int D, float eps) {
    __shared__ float part[8];
    const int r = blockIdx.x, t = threadIdx.x;
    const float * xr = x + (size_t)r * D;
    float s = 0.f;
    for (int j = t; j < D; j += 256) s += xr[j] * xr[j];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor(s, o);
    if (t % 32 == 0) part[t / 32] = s;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) tot += part[i];
    const float inv = rsqrtf(tot / (float)D + eps);
    for (int j = t; j < D; j += 256) out[(size_t)r * D + j] = xr[j] * inv * w[j];
}

extern "C" __global__ void silu_mul_bf16(const uint16_t * __restrict__ gu, uint16_t * __restrict__ act, int H, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / H, c = i % H;
    const float g = bf2f(gu[r * 2 * H + c]), u = bf2f(gu[r * 2 * H + H + c]);
    const float s = bf2f(f2bf(g / (1.f + expf(-g))));
    act[i] = f2bf(s * u);
}

extern "C" __global__ void residual3(const float * __restrict__ x, const uint16_t * __restrict__ m,
                                     const uint16_t * __restrict__ y, float * __restrict__ out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (x[i] + bf2f(m[i])) + bf2f(y[i]);
}

extern "C" __global__ void cast_bf16(const float * __restrict__ x, uint16_t * __restrict__ out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f2bf(x[i]);
}

// rows x n block copy between row-major matrices with leading dimensions ld_in / ld_out and column offsets
extern "C" __global__ void copy_cols_bf16(const uint16_t * __restrict__ in, int ld_in, int c_in,
                                          uint16_t * __restrict__ out, int ld_out, int c_out, int n, int64_t rows) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n) return;
    const int64_t r = i / n, c = i % n;
    out[r * ld_out + c_out + c] = in[r * ld_in + c_in + c];
}
