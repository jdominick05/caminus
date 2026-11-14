// Own Q8-only dense attention helpers. Included only by K3_Q35_KV_Q8 wrappers;
// the F16 objects do not acquire either dispatch or scratch from this header.
#pragma once
#include "own_kv_codec.h"
#define K3_Q35_GDN_LOCAL_BASE   // keep replay addresses branch-local with the additional Q8 operations

// The host supplies the matching Q/K256 and V64 transforms. A refused block
// leaves its bytes alone, but others may have been written: any error poisons
// the whole pass. Seg.gs points to a dedicated sticky word OUTSIDE Sync's timeout
// status, so a timeout cannot overwrite a numerical failure. Host checks this
// word before any retry, logits consumption or parking.
__device__ __forceinline__ void q35_op_kv_append_q8(const Seg & sg, int tt) {
    const int pos = ((const int *)sg.asum)[0] + tt;
    int M = sg.N;
#ifdef K3_XCHG_PUSH_ONLY
    asm volatile("" : "+s"(M));   // keep Q8 row geometry inside the append branch
#endif
    auto * dst = (OwnKvQ8Block *)((uint8_t *)sg.y + (size_t)pos * own_kv_q8_row_bytes(M));
    const int nb = M / 32;
    for (int b = K3TID; b < 2 * nb; b += blockDim.x) {
        const float * x = (const float *)(b < nb ? sg.a : (const void *)sg.W) + (size_t)tt * M + (size_t)(b % nb) * 32;
        if (!own_kv_q8_encode(x, dst + b)) atomicOr((int *)sg.gs, 2);
    }
}

// Same Sylvester butterflies / order / normalization as pf_hadamard and the
// existing Coder op, without a ggml include. Q/K256, V/output64; output rotation
// MUST precede channel-wise gating. Input and output are fp32, in place.
__device__ __forceinline__ void q35_op_hadamard(const Seg & sg, int t) {
#pragma clang fp contract(off)
    __shared__ float hb[256];
    int n = sg.K;
#ifdef K3_XCHG_PUSH_ONLY
    asm volatile("" : "+s"(n));   // keep transform geometry inside its dispatch branch
#endif
    const int tid = K3TID;
    const float scale = 1.0f / sqrtf((float)n);
    const int c0 = t * sg.rows_per_task, c1 = min(sg.N, c0 + sg.rows_per_task);
    for (int c = c0; c < c1; c++) {
        float * v = (float *)sg.y + (size_t)c * n;
        for (int i = tid; i < n; i += blockDim.x) hb[i] = v[i];
        __syncthreads();
        for (int h = 1; h < n; h <<= 1) {
            for (int k = tid; k < n / 2; k += blockDim.x) {
                const int i = (k / h) * 2 * h + k % h;
                const float a = hb[i], b = hb[i + h];
                hb[i] = a + b; hb[i + h] = a - b;
            }
            __syncthreads();
        }
        for (int i = tid; i < n; i += blockDim.x) v[i] = hb[i] * scale;
        __syncthreads();
    }
}
