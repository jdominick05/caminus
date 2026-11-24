// Fused activation fake-quant for QLinear (qat.py: fq_act(rotate(x, 128), bits, clip)), forward and backward.
//
// Forward, per row of K (a multiple of 128):
//   u = bf16(x_block @ H128)        H = normalized Sylvester Hadamard; the reference's matmul runs in bf16, so the
//                                   1/sqrt(128) factor is its bf16 value and u is rounded to bf16 before quantizing
//   s = max(amax|u| * clip / qmax, 1e-8)
//   y = bf16(clamp(round_half_even(u / s), -qmax-1, qmax) * s)
// Backward reproduces autograd through that graph. The STE mask is strict (r > lo && r < hi): torch.clamp's backward
// (torch 2.14) gives zero gradient at the bounds, so values that quantize to +qmax (each row's positive max among
// them) pass no direct gradient. The gradient into s through amax is kept (fq_act does not detach s): dL/ds = sum_j g_j (q_j - m_j t_j); the amax gradient goes to the row's max-|u| elements, split
// evenly over ties, times sign(u); then du is rounded to bf16 (u's dtype) and rotated back (H is symmetric).
//
// One 256-thread workgroup (8 wave32s) per row. A wave transforms one 128-chunk at a time: lane l holds elements
// 4l..4l+3, butterfly stages 1-2 in registers, stages 4..64 across lanes with xor shuffles.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define WAVES 8
#define MAXC 8                               // chunks per wave held in registers: K <= WAVES * MAXC * 128 = 8192

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }

__device__ __forceinline__ uint16_t f2bf(float f) {          // round to nearest even, as torch's .to(bfloat16)
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }

template <typename T> __device__ __forceinline__ float ld(const T * p);
template <> __device__ __forceinline__ float ld<uint16_t>(const uint16_t * p) { return bf2f(*p); }
template <> __device__ __forceinline__ float ld<float>(const float * p) { return *p; }
template <typename T> __device__ __forceinline__ void st(T * p, float v);
template <> __device__ __forceinline__ void st<uint16_t>(uint16_t * p, float v) { *p = f2bf(v); }
template <> __device__ __forceinline__ void st<float>(float * p, float v) { *p = v; }

// in-place unnormalized FWHT over a 128-chunk spread as 4 values per lane
__device__ __forceinline__ void fwht128(float v[4], int lane) {
    float a = v[0], b = v[1]; v[0] = a + b; v[1] = a - b;
    a = v[2]; b = v[3]; v[2] = a + b; v[3] = a - b;
    a = v[0]; b = v[2]; v[0] = a + b; v[2] = a - b;
    a = v[1]; b = v[3]; v[1] = a + b; v[3] = a - b;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const float p = __shfl_xor(v[e], m);
            v[e] = (lane & m) ? (p - v[e]) : (v[e] + p);
        }
    }
}

__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor(v, m));
    return v;
}

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor(v, m);
    return v;
}

// block-wide reduce of one value per wave (lane 0 holds it); every thread gets the result
__device__ __forceinline__ float block_reduce(float v, float * lds, int wave, int lane, bool is_max) {
    __syncthreads();
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float r = lds[0];
    for (int w = 1; w < WAVES; w++) r = is_max ? fmaxf(r, lds[w]) : r + lds[w];
    return r;
}

// the bf16 value of 1/sqrt(128), as the reference's bf16 matmul sees it
#define HSCALE 0.08837890625f

// codes (optional): the integer codes packed two per byte along K (element 2j in the low nibble of byte j), the iu4
// GEMM's operand (needs qmax <= 7); y (optional when codes are written): the dequantized bf16 values
// Sources give element k of row `row` as a float (actq rounds it to bf16, as autocast's input); the plain source reads
// a [M, K] tensor, SiluMulSrc computes the MLP activation bf16(bf16(silu(g)) * u) from gu = [g | u] on the fly.
template <typename T> struct PlainSrc {
    const T * x; int K;
    __device__ __forceinline__ float operator()(int row, int k) const { return ld<T>(x + (size_t)row * K + k); }
};
__device__ __forceinline__ float silu_bf(float g) { return rbf(g / (1.0f + expf(-g))); }   // torch: x / (1 + exp(-x)), rounded
struct SiluMulSrc {
    const uint16_t * gu; int ld, F;
    __device__ __forceinline__ float operator()(int row, int k) const {
        const uint16_t * r = gu + (size_t)row * ld;
        return rbf(silu_bf(bf2f(r[k])) * bf2f(r[F + k]));
    }
};
template <typename Src>
__device__ void actq_fwd_src(Src src, uint16_t * __restrict__ y, float * __restrict__ s_out,
                             int K, float clip, int qmax, uint16_t * __restrict__ codes) {
    __shared__ float lds[WAVES];
    const int row = blockIdx.x, wave = threadIdx.x / 32, lane = threadIdx.x % 32, nch = K / 128;
    float u[MAXC][4];
    float amax = 0.f;
#pragma unroll
    for (int c = 0; c < MAXC; c++) {
        const int ch = wave + c * WAVES;
        if (ch < nch) {
#pragma unroll
            for (int e = 0; e < 4; e++) u[c][e] = rbf(src(row, ch * 128 + lane * 4 + e));   // autocast's bf16 input
            fwht128(u[c], lane);
#pragma unroll
            for (int e = 0; e < 4; e++) { u[c][e] = rbf(u[c][e] * HSCALE); amax = fmaxf(amax, fabsf(u[c][e])); }
        }
    }
    amax = block_reduce(wave_max(amax), lds, wave, lane, true);
    const float s = fmaxf(amax * clip / (float)qmax, 1e-8f);
    if (threadIdx.x == 0 && s_out) s_out[row] = s;
    uint16_t * yr = y ? y + (size_t)row * K : nullptr;
#pragma unroll
    for (int c = 0; c < MAXC; c++) {
        const int ch = wave + c * WAVES;
        if (ch < nch) {
            uint32_t pk = 0;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float q = fminf(fmaxf(rintf(u[c][e] / s), (float)(-qmax - 1)), (float)qmax);
                if (y) yr[ch * 128 + lane * 4 + e] = f2bf(q * s);
                pk |= ((uint32_t)(int)q & 0xFu) << (4 * e);
            }
            if (codes) codes[(size_t)row * (K / 4) + ch * 32 + lane] = (uint16_t)pk;   // 4 codes = 2 bytes per lane
        }
    }
}
template <typename T>
__device__ void actq_fwd(const T * __restrict__ x, uint16_t * __restrict__ y, float * __restrict__ s_out,
                         int K, float clip, int qmax, uint16_t * __restrict__ codes = nullptr) {
    actq_fwd_src(PlainSrc<T>{x, K}, y, s_out, K, clip, qmax, codes);
}

// Sinks take the input gradient of element k (fp32, before the store's rounding)
template <typename T> struct PlainSink {
    T * dx; int K;
    __device__ __forceinline__ void operator()(int row, int k, float v) const { st<T>(dx + (size_t)row * K + k, v); }
};
// the MLP: dact (rounded to bf16, the activation's dtype) back through act = bf16(s * u), s = bf16(silu(g)), as torch's
// MulBackward0 (each product rounded) and silu_backward (dy * sig * (1 + x * (1 - sig)), rounded), into dgu = [dg | du]
struct SiluMulSink {
    const uint16_t * gu; int ld, F; uint16_t * dgu; int ldd;
    __device__ __forceinline__ void operator()(int row, int k, float v) const {
        const uint16_t * r = gu + (size_t)row * ld;
        const float dact = rbf(v), g = bf2f(r[k]), u = bf2f(r[F + k]), sl = silu_bf(g);
        const float du = rbf(dact * sl), ds = rbf(dact * u);
        const float sig = 1.0f / (1.0f + expf(-g));
        uint16_t * o = dgu + (size_t)row * ldd;
        o[k] = f2bf(ds * sig * (1.0f + g * (1.0f - sig)));
        o[F + k] = f2bf(du);
    }
};
template <typename Src, typename Sink>
__device__ void actq_bwd_src(Src src, const uint16_t * __restrict__ g, Sink sink, int K, float clip, int qmax) {
    __shared__ float lds[WAVES];
    const int row = blockIdx.x, wave = threadIdx.x / 32, lane = threadIdx.x % 32, nch = K / 128;
    const uint16_t * gr = g + (size_t)row * K;
    float u[MAXC][4], gv[MAXC][4];
    float amax = 0.f;
#pragma unroll
    for (int c = 0; c < MAXC; c++) {
        const int ch = wave + c * WAVES;
        if (ch < nch) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                u[c][e] = rbf(src(row, ch * 128 + lane * 4 + e));
                gv[c][e] = bf2f(gr[ch * 128 + lane * 4 + e]);
            }
            fwht128(u[c], lane);
#pragma unroll
            for (int e = 0; e < 4; e++) { u[c][e] = rbf(u[c][e] * HSCALE); amax = fmaxf(amax, fabsf(u[c][e])); }
        }
    }
    amax = block_reduce(wave_max(amax), lds, wave, lane, true);
    const float s_raw = amax * clip / (float)qmax, s = fmaxf(s_raw, 1e-8f);
    const float lo = (float)(-qmax - 1), hi = (float)qmax;
    // dL/ds and the number of max-|u| elements (ties share the amax gradient)
    float dlds = 0.f, ties = 0.f;
#pragma unroll
    for (int c = 0; c < MAXC; c++) {
        const int ch = wave + c * WAVES;
        if (ch < nch) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float t = u[c][e] / s, r = rintf(t), q = fminf(fmaxf(r, lo), hi);
                const float m = (r > lo && r < hi) ? 1.f : 0.f;   // torch.clamp's backward excludes the bounds
                dlds += gv[c][e] * (q - m * t);
                ties += (fabsf(u[c][e]) == amax) ? 1.f : 0.f;
            }
        }
    }
    dlds = block_reduce(wave_sum(dlds), lds, wave, lane, false);
    ties = block_reduce(wave_sum(ties), lds, wave, lane, false);
    const float coef = (s_raw >= 1e-8f) ? dlds * clip / (float)qmax / ties : 0.f;
#pragma unroll
    for (int c = 0; c < MAXC; c++) {
        const int ch = wave + c * WAVES;
        if (ch < nch) {
            float d[4];
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float t = u[c][e] / s, r = rintf(t);
                const float m = (r > lo && r < hi) ? 1.f : 0.f;   // torch.clamp's backward excludes the bounds
                float v = gv[c][e] * m;
                if (fabsf(u[c][e]) == amax && u[c][e] != 0.f) v += coef * (u[c][e] > 0.f ? 1.f : -1.f);
                d[e] = rbf(v);                                 // the gradient of the bf16 tensor u
            }
            fwht128(d, lane);
#pragma unroll
            for (int e = 0; e < 4; e++) sink(row, ch * 128 + lane * 4 + e, rbf(d[e] * HSCALE));
        }
    }
}
template <typename T>
__device__ void actq_bwd(const T * __restrict__ x, const uint16_t * __restrict__ g, T * __restrict__ dx,
                         int K, float clip, int qmax) {
    actq_bwd_src(PlainSrc<T>{x, K}, g, PlainSink<T>{dx, K}, K, clip, qmax);
}

// ---- block entry (fusion F5): h = RMSNorm(x) never materialized --------------------------------------------------------
// forward per row: rstd = rsqrt(mean(x^2) + eps), h = x * rstd * w; h's activation quant (codes, scales, optionally the
// bf16 values); nab gate projections ab[j] = bf16(sum_k bf16(h_k) * wab[j][k]) (fla's a_proj / b_proj on autocast's bf16 h)
// backward per row: dh = actq_bwd(h, dy) + bf16(sum_j dab_j wab[j]) (each projection's dgrad rounded to bf16, as
// autograd's casts) then the RMSNorm backward dx = rstd * (dh w - xhat * mean(dh w xhat)), xhat = x * rstd.
struct RmsSrc {
    const float * x; const float * w; float rstd; int d;
    __device__ __forceinline__ float operator()(int row, int k) const { return x[(size_t)row * d + k] * rstd * w[k]; }
};
struct LdsSink {
    float * buf;
    __device__ __forceinline__ void operator()(int, int k, float v) const { buf[k] = v; }
};
#define NAB 16
__device__ __forceinline__ float row_rstd(const float * xr, int d, float eps, float * lds, int wave, int lane) {
    float ss = 0.f;
    for (int k = threadIdx.x; k < d; k += 256) ss += xr[k] * xr[k];
    ss = block_reduce(wave_sum(ss), lds, wave, lane, false);
    return rsqrtf(ss / (float)d + eps);
}

extern "C" {
__global__ void __launch_bounds__(256) block_entry_fwd(const float * x, const float * nw, const uint16_t * wab, int nab, float eps,
                                                         uint16_t * y, float * s, uint16_t * codes, uint16_t * ab, float * rstd_out,
                                                         int d, float clip, int qmax) {
    __shared__ float lds[WAVES];
    __shared__ float part[WAVES][NAB];
    const int row = blockIdx.x, wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const float rstd = row_rstd(x + (size_t)row * d, d, eps, lds, wave, lane);
    if (threadIdx.x == 0) rstd_out[row] = rstd;
    actq_fwd_src(RmsSrc{x, nw, rstd, d}, y, s, d, clip, qmax, codes);
    if (nab) {   // gate projections on bf16(h)
        float acc[NAB];
#pragma unroll
        for (int j = 0; j < NAB; j++) acc[j] = 0.f;
        for (int k = threadIdx.x; k < d; k += 256) {
            const float hb = rbf(x[(size_t)row * d + k] * rstd * nw[k]);
#pragma unroll
            for (int j = 0; j < NAB; j++) if (j < nab) acc[j] += hb * bf2f(wab[(size_t)j * d + k]);
        }
#pragma unroll
        for (int j = 0; j < NAB; j++) if (j < nab) { const float v = wave_sum(acc[j]); if (lane == 0) part[wave][j] = v; }
        __syncthreads();
        if (threadIdx.x < nab) {
            float v = 0.f;
            for (int w = 0; w < WAVES; w++) v += part[w][threadIdx.x];
            ab[(size_t)row * nab + threadIdx.x] = f2bf(v);
        }
    }
}
__global__ void __launch_bounds__(256) block_entry_bwd(const float * x, const float * nw, const float * rstd_in, const uint16_t * wab,
                                                         int nab, const uint16_t * dy, const uint16_t * dab, float * dx,
                                                         int d, float clip, int qmax) {
    __shared__ float dh[1024];
    __shared__ float lds[WAVES];
    __shared__ float dabs[NAB];
    const int row = blockIdx.x, wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const float rstd = rstd_in[row];
    actq_bwd_src(RmsSrc{x, nw, rstd, d}, dy, LdsSink{dh}, d, clip, qmax);
    if (threadIdx.x < nab) dabs[threadIdx.x] = bf2f(dab[(size_t)row * nab + threadIdx.x]);
    __syncthreads();
    const int half = nab / 2;   // projections come in two groups (a_proj rows, then b_proj rows), each its own dgrad
    float dot = 0.f;
    for (int k = threadIdx.x; k < d; k += 256) {
        float g = dh[k];
        if (nab) {
            float pa = 0.f, pb = 0.f;
            for (int j = 0; j < half; j++) pa += dabs[j] * bf2f(wab[(size_t)j * d + k]);
            for (int j = half; j < nab; j++) pb += dabs[j] * bf2f(wab[(size_t)j * d + k]);
            g = (g + rbf(pa)) + rbf(pb);
        }
        dh[k] = g;
        dot += g * nw[k] * (x[(size_t)row * d + k] * rstd);
    }
    dot = block_reduce(wave_sum(dot), lds, wave, lane, false) / (float)d;
    for (int k = threadIdx.x; k < d; k += 256) {
        const float xh = x[(size_t)row * d + k] * rstd;
        dx[(size_t)row * d + k] = rstd * (dh[k] * nw[k] - xh * dot);
    }
}
// the MLP's silu * mul fused with the down projection's activation quant; gu rows of stride ld (a view of a wider
// GEMM output is fine), F = the activation width (K of the down projection)
__global__ void __launch_bounds__(256) silu_mul_actq_fwd(const uint16_t * gu, int ld, int F, uint16_t * y, float * s, uint16_t * codes,
                                                          float clip, int qmax) {
    actq_fwd_src(SiluMulSrc{gu, ld, F}, y, s, F, clip, qmax, codes);
}
__global__ void __launch_bounds__(256) silu_mul_actq_bwd(const uint16_t * gu, int ld, int F, const uint16_t * g, uint16_t * dgu,
                                                          float clip, int qmax) {
    actq_bwd_src(SiluMulSrc{gu, ld, F}, g, SiluMulSink{gu, ld, F, dgu, 2 * F}, F, clip, qmax);
}
__global__ void __launch_bounds__(256) actq_fwd_codes_bf16(const uint16_t * x, uint16_t * y, float * s, uint16_t * codes, int K, float clip, int qmax) {
    actq_fwd<uint16_t>(x, y, s, K, clip, qmax, codes);
}
__global__ void __launch_bounds__(256) actq_fwd_codes_f32(const float * x, uint16_t * y, float * s, uint16_t * codes, int K, float clip, int qmax) {
    actq_fwd<float>(x, y, s, K, clip, qmax, codes);
}
__global__ void __launch_bounds__(256) actq_fwd_bf16(const uint16_t * x, uint16_t * y, float * s, int K, float clip, int qmax) {
    actq_fwd<uint16_t>(x, y, s, K, clip, qmax);
}
__global__ void __launch_bounds__(256) actq_fwd_f32(const float * x, uint16_t * y, float * s, int K, float clip, int qmax) {
    actq_fwd<float>(x, y, s, K, clip, qmax);
}
__global__ void __launch_bounds__(256) actq_bwd_bf16(const uint16_t * x, const uint16_t * g, uint16_t * dx, int K, float clip, int qmax) {
    actq_bwd<uint16_t>(x, g, dx, K, clip, qmax);
}
__global__ void __launch_bounds__(256) actq_bwd_f32(const float * x, const uint16_t * g, float * dx, int K, float clip, int qmax) {
    actq_bwd<float>(x, g, dx, K, clip, qmax);
}
}
