// Fused glue for the S1 block (fusion phase): the residual sum with the slices' scatter.
//   forward:  out[r, c] = ((x[r, c] + m[r, c]) + mlp[r, c]) + (pos[r] >= 0 ? ys[pos[r], c] : 0)   (fp32, torch's add order;
//             m, mlp, ys bf16, promoted exactly; ys = the routed rows' slice outputs in section-sorted order)
//   backward: dm = dmlp = bf16(dout), dys[pos[r]] = bf16(dout[r]) (autograd's casts back to the bf16 operands); dx = dout.
// One thread per 4 consecutive columns of a row; d % 4 == 0.
#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef uint16_t u16x4 __attribute__((ext_vector_type(4)));

extern "C" __global__ void __launch_bounds__(256) residual_slices_fwd(const float * x, const uint16_t * m, const uint16_t * mlp,
        const uint16_t * ys, const int * pos, float * out, int d, long n4) {
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n4) return;
    const long e = i * 4; const int r = (int)(e / d), c = (int)(e % d);
    const f32x4 xv = *(const f32x4 *)(x + e);
    const u16x4 mv = *(const u16x4 *)(m + e), lv = *(const u16x4 *)(mlp + e);
    const int p = pos ? pos[r] : -1;
    u16x4 sv = {0, 0, 0, 0};
    if (p >= 0) sv = *(const u16x4 *)(ys + (size_t)p * d + c);
    f32x4 o;
#pragma unroll
    for (int k = 0; k < 4; k++) o[k] = ((xv[k] + bf2f(mv[k])) + bf2f(lv[k])) + (p >= 0 ? bf2f(sv[k]) : 0.0f);
    *(f32x4 *)(out + e) = o;
}

extern "C" __global__ void __launch_bounds__(256) residual_slices_bwd(const float * dout, const int * pos, uint16_t * dm,
        uint16_t * dmlp, uint16_t * dys, int d, long n4) {
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n4) return;
    const long e = i * 4; const int r = (int)(e / d), c = (int)(e % d);
    const f32x4 g = *(const f32x4 *)(dout + e);
    u16x4 b;
#pragma unroll
    for (int k = 0; k < 4; k++) b[k] = f2bf(g[k]);
    *(u16x4 *)(dm + e) = b; *(u16x4 *)(dmlp + e) = b;
    const int p = pos ? pos[r] : -1;
    if (p >= 0) *(u16x4 *)(dys + (size_t)p * d + c) = b;
}
