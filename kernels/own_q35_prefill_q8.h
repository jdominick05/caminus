// Own dense27 Q8 prefill's wide-value fallback. Included only by the Q8 object,
// after q3m_prefill.hip's wave sum helper; no device allocation or fp16 rounding.
#pragma once
#include "own_kv_codec.h"

__device__ __forceinline__ float q35p_q8_value(const OwnKvQ8Block * row, int i) {
    const OwnKvQ8Block & b = row[i / 32];
    return (float)b.qs[i % 32] * (float)b.d;
}

// One wave per absolute query position of this workgroup's eight-position
// block. Every lane owns eight dimensions, and all lanes of an active wave
// take the same head/cell loops. No workgroup barriers: partial first/last
// blocks and rows which keep WMMA may have inactive waves. A row falls back
// exactly when its Q is unsafe or its own causal prefix contains unsafe KV;
// this predicate is independent of the chunk extent and launch splitting.
__device__ __noinline__ void q35p_attn_q8_fp32(const float * qin, const uint8_t * cache,
                                            int pos0, int n, int b8, int heads, int nkv,
                                            float scale_log2, float * out, int first_unsafe, unsigned q_unsafe) {
#pragma clang fp contract(off)
    const int lane = threadIdx.x & 31, pos = b8 + (threadIdx.x >> 5), tt = pos - pos0;
    if (tt < 0 || tt >= n) return;
    if (pos < first_unsafe && !(q_unsafe & (1u << (pos - b8)))) return;
    const int G = heads / nkv, g = blockIdx.y;
    const size_t rb = own_kv_q8_row_bytes(nkv * 256);
    for (int h = 0; h < G; h++) {
        const size_t off = ((size_t)tt * heads + g * G + h) * 256;
        float q[8], acc[8];
#pragma unroll
        for (int j = 0; j < 8; j++) { q[j] = qin[off + lane + 32 * j]; acc[j] = 0.0f; }
        float m = -INFINITY, l = 0.0f;
        for (int c = 0; c <= pos; c++) {
            const auto * row = (const OwnKvQ8Block *)(cache + (size_t)c * rb);
            float dot = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; j++)
                dot += q[j] * q35p_q8_value(row, g * 256 + lane + 32 * j);
            const float score = wsum(dot) * scale_log2;
            const float mn = fmaxf(m, score), a = exp2f(m - mn), p = exp2f(score - mn);
            l = l * a + p;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float v = q35p_q8_value(row, nkv * 256 + g * 256 + lane + 32 * j);
                acc[j] = acc[j] * a + p * v;
            }
            m = mn;
        }
        const float inv = 1.0f / l;
#pragma unroll
        for (int j = 0; j < 8; j++) out[off + lane + 32 * j] = acc[j] * inv;
    }
}
