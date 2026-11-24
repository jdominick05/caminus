// Attention pieces for the S1 attention blocks (model.py Attention: RoPE, causal GQA SDPA), runtime forward.
//   rope_split_bf16  q [T, H, D] and kv [T, 2, KV, D] (the projection outputs) -> q_rope [H, T, D],
//                    k_rope [KV, T, D], v [KV, T, D]. Reproduces model.rope in bf16 op by op: inv = base^(-2i/D),
//                    ang = t * inv (fp32), cos / sin rounded to bf16, then x1*c - x2*s and x1*s + x2*c with every
//                    product and sum rounded to bf16 (torch's bf16 elementwise ops).
//   attn_causal_bf16 o[t, h*D + d] = softmax_j<=t(q_h,t . k_g,j * scale) v_g,j with g = h / (H / KV) (GQA); scores,
//                    softmax and the weighted sum in fp32, out bf16. A correctness kernel: one workgroup per
//                    (head, query); PyTorch runs AOTriton's flash kernel (P rounded to bf16 before P V), so the
//                    two agree to bf16 rounding, not bit for bit.
#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }

__device__ __forceinline__ uint16_t f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }

// one thread per (row, pair): rotate one (x1, x2) pair of head `hh` at position t
__device__ __forceinline__ void rope_pair(const uint16_t * src, uint16_t * dst, int t, int i, int D, float base) {
    const float inv = powf(base, -(float)(2 * i) / (float)D);
    const float ang = (float)t * inv;
    const float c = rbf(cosf(ang)), s = rbf(sinf(ang));
    const float x1 = bf2f(src[2 * i]), x2 = bf2f(src[2 * i + 1]);
    dst[2 * i] = f2bf(rbf(x1 * c) - rbf(x2 * s));
    dst[2 * i + 1] = f2bf(rbf(x1 * s) + rbf(x2 * c));
}

extern "C" __global__ void rope_split_bf16(const uint16_t * __restrict__ q, const uint16_t * __restrict__ kv,
                                          uint16_t * __restrict__ qr, uint16_t * __restrict__ kr,
                                          uint16_t * __restrict__ vo, int T, int H, int KV, int D, float base) {
    const int t = blockIdx.x, half = D / 2;
    for (int e = threadIdx.x; e < (H + KV) * half; e += blockDim.x) {
        const int hh = e / half, i = e % half;
        if (hh < H) rope_pair(q + ((size_t)t * H + hh) * D, qr + ((size_t)hh * T + t) * D, t, i, D, base);
        else { const int g = hh - H; rope_pair(kv + ((size_t)t * 2 * KV + g) * D, kr + ((size_t)g * T + t) * D, t, i, D, base); }
    }
    for (int e = threadIdx.x; e < KV * D; e += blockDim.x) {
        const int g = e / D, d = e % D;
        vo[((size_t)g * T + t) * D + d] = kv[((size_t)t * 2 * KV + KV + g) * D + d];
    }
}

#define ATH 256
#define MAXT 8192
extern "C" __global__ void __launch_bounds__(ATH)
attn_causal_bf16(const uint16_t * __restrict__ qr, const uint16_t * __restrict__ kr, const uint16_t * __restrict__ v,
                 uint16_t * __restrict__ o, int T, int H, int KV, int D, float scale) {
    __shared__ float sq[128], sp[MAXT], red[ATH / 32];
    const int t = blockIdx.x % T, hh = blockIdx.x / T, g = hh / (H / KV), tid = threadIdx.x;
    for (int d = tid; d < D; d += ATH) sq[d] = bf2f(qr[((size_t)hh * T + t) * D + d]);
    __syncthreads();
    float mx = -INFINITY;
    for (int j = tid; j <= t; j += ATH) {
        const uint16_t * kj = kr + ((size_t)g * T + j) * D;
        float s = 0.f;
        for (int d = 0; d < D; d++) s += sq[d] * bf2f(kj[d]);
        s *= scale; sp[j] = s; mx = fmaxf(mx, s);
    }
    for (int m = 16; m > 0; m >>= 1) mx = fmaxf(mx, __shfl_xor(mx, m));
    if (tid % 32 == 0) red[tid / 32] = mx;
    __syncthreads();
    mx = red[0]; for (int w = 1; w < ATH / 32; w++) mx = fmaxf(mx, red[w]);
    __syncthreads();
    float sum = 0.f;
    for (int j = tid; j <= t; j += ATH) { const float p = expf(sp[j] - mx); sp[j] = p; sum += p; }
    for (int m = 16; m > 0; m >>= 1) sum += __shfl_xor(sum, m);
    if (tid % 32 == 0) red[tid / 32] = sum;
    __syncthreads();
    sum = 0.f; for (int w = 0; w < ATH / 32; w++) sum += red[w];
    for (int d = tid; d < D; d += ATH) {
        float acc = 0.f;
        for (int j = 0; j <= t; j++) acc += sp[j] * bf2f(v[((size_t)g * T + j) * D + d]);
        o[((size_t)t * H + hh) * D + d] = f2bf(acc / sum);
    }
}
