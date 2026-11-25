// Row-wise kernels for the S1 adaptive head's loss (s1.py AdaptiveHead.logq), over bf16 logits L [rows, V] with
// leading dimension ld (V <= ld; columns >= V are padding and never read).
//
//   lse_gather: lse[r] = logsumexp_j L[r, j] in fp32 (one online pass), and gath[r, i] = L[r, cols[r, i]] for
//               cols >= 0 (else 0). This replaces the reference's fp32 cast + logsumexp + gather passes.
//   softmax_grad (in place): L[r, j] <- bf16(sum_i coef[r, i] [j == cols[r, i]] - total[r] * exp(L[r, j] - lse[r]))
//               = the gradient of sum_i coef[r, i] (L[r, cols[r, i]] - lse[r]) when total[r] = sum_i coef[r, i].
//               Padding columns are left untouched.
// One 256-thread workgroup per row; consecutive threads read consecutive columns (coalesced).
#include <hip/hip_runtime.h>
#include <stdint.h>

#define TPB 256
#define MAXK 16

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }

__device__ __forceinline__ uint16_t f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

// merge two (max, sum of exp(x - max)) pairs
__device__ __forceinline__ void lse_merge(float & m, float & s, float m2, float s2) {
    const float mn = fmaxf(m, m2);
    if (mn == -INFINITY) return;
    s = s * __expf(m - mn) + s2 * __expf(m2 - mn);
    m = mn;
}

extern "C" __global__ void __launch_bounds__(TPB)
head_lse_gather_bf16(const uint16_t * __restrict__ L, int V, int ld, const int * __restrict__ cols, int k,
                     float * __restrict__ lse, float * __restrict__ gath) {
    __shared__ float sm[TPB / 32], ss[TPB / 32];
    const int r = blockIdx.x, t = threadIdx.x, lane = t % 32, wave = t / 32;
    const uint16_t * row = L + (size_t)r * ld;
    float m = -INFINITY, s = 0.f;
    for (int j = t; j < V; j += TPB) {
        const float x = bf2f(row[j]);
        if (x > m) { s = s * expf(m - x) + 1.f; m = x; } else s += expf(x - m);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) lse_merge(m, s, __shfl_xor(m, o), __shfl_xor(s, o));
    if (lane == 0) { sm[wave] = m; ss[wave] = s; }
    __syncthreads();
    if (t == 0) {
        float M = sm[0], S = ss[0];
        for (int w = 1; w < TPB / 32; w++) lse_merge(M, S, sm[w], ss[w]);
        lse[r] = M + logf(S);
    }
    if (t < k) {
        const int c = cols[(size_t)r * k + t];
        gath[(size_t)r * k + t] = c >= 0 ? bf2f(row[c]) : 0.f;
    }
}

extern "C" __global__ void __launch_bounds__(TPB)
head_softmax_grad_bf16(uint16_t * __restrict__ L, int V, int ld, const float * __restrict__ lse,
                       const int * __restrict__ cols, const float * __restrict__ coef, int k,
                       const float * __restrict__ total) {
    __shared__ int sc[MAXK];
    __shared__ float sv[MAXK];
    const int r = blockIdx.x, t = threadIdx.x;
    if (t < k) { sc[t] = cols[(size_t)r * k + t]; sv[t] = coef[(size_t)r * k + t]; }
    __syncthreads();
    uint16_t * row = L + (size_t)r * ld;
    const float l = lse[r], tot = total[r];
    for (int j = t; j < V; j += TPB) {
        float d = -tot * expf(bf2f(row[j]) - l);
        for (int i = 0; i < k; i++) if (sc[i] == j) d += sv[i];
        row[j] = f2bf(d);
    }
}
