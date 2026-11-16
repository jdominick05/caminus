// The two-card batched prefill of Qwen3-30B-A3B (python/caminus/prefill/q3m_tp.py): the one device op the one-card prefill
// object (kernels/q3m_prefill.hip) has no form of. Its own object, so q3m_prefill.hsaco stays byte-identical; no ggml code.
//
// q3p_slot_sum: y[t][i] = sum over slots s < S of w[t][s] d[t][s][i] (fma in slot order from 0: OP_OWN_SLOT_SUM's
// statements, kernels/k3_q3m_ops.h, per token). d is the down GEMM's output, one row per (token, slot) pair at row t S + s
// (q3p_group's yd), w the router's weights [n][S]. Each card weighs its own partial down rows over the top-k slots before
// the exchange, so D floats per token cross the link instead of S x D (the decode's reason: 8 x 2,048 floats per exchange
// measured 0.70 ms per token over 48 exchanges, 2,048 floats 0.19 ms). Grid n (one workgroup per token), block 256.
#include <hip/hip_runtime.h>

extern "C" __global__ void __launch_bounds__(256) q3p_slot_sum(const float * __restrict__ d, const float * __restrict__ w, int S,
                                                               float * __restrict__ y, int N) {
    const int t = blockIdx.x;
    const float * dr = d + (size_t)t * S * N;
    const float * wr = w + (size_t)t * S;
    float * yr = y + (size_t)t * N;
    for (int i = threadIdx.x; i < N; i += 256) {
        float acc = 0.0f;
        for (int s = 0; s < S; s++) acc = __builtin_fmaf(wr[s], dr[(size_t)s * N + i], acc);
        yr[i] = acc;
    }
}
