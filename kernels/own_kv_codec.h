// Own Q8_0 KV wire codec. No weights, head geometry, rotation or ggml dependency.
// The scale / codes follow k3_gguf_ops.h's KV writer: fp32 amax/127, fp32 reciprocal,
// roundf(x * reciprocal), with ONLY the stored scale rounded to half. The finite
// encode domain is specified independently by caminus.ref_kv; compiled GPU parity
// (including subnormals and division) must pass the codec gate before model use.
#ifndef CAMINUS_OWN_KV_CODEC_H
#define CAMINUS_OWN_KV_CODEC_H
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef __HIPCC__
#define OWN_KV_FN __host__ __device__ __forceinline__
#else
#define OWN_KV_FN inline
#endif

struct OwnKvQ8Block { _Float16 d; int8_t qs[32]; };
static_assert(sizeof(OwnKvQ8Block) == 34, "Q8 KV block: fp16 scale + 32 signed codes");
static_assert(offsetof(OwnKvQ8Block, qs) == 2, "Q8 KV code offset");

OWN_KV_FN size_t own_kv_q8_row_bytes(int M) {
    return (size_t)2 * (M / 32) * sizeof(OwnKvQ8Block);
}

// Validate before writing any byte of the block. Nonfinite inputs and nonzero
// blocks with fp32 scale underflow / reciprocal overflow or half scale overflow
// are refused. A normal fp32 scale MAY round to stored half zero: codes survive.
// On failure the caller must invalidate the whole append, even if other blocks
// succeeded; this helper does not promise a transactional multi-block row write.
OWN_KV_FN bool own_kv_q8_encode(const float * x, OwnKvQ8Block * y) {
    float amax = 0.0f;
    bool finite = true;
    for (int j = 0; j < 32; j++) {
        // Scan before branching: per-element early returns enlarge execution-mask
        // live ranges in persistent callers and increase spilled scalar registers.
        finite &= isfinite(x[j]);
        amax = fmaxf(amax, fabsf(x[j]));
    }
    if (!finite) return false;
    const float d = amax / 127.0f;
    const float inv = d ? 1.0f / d : 0.0f;
    const _Float16 stored = (_Float16)d;
    if (amax != 0.0f && (d == 0.0f || !isfinite(d) || !isfinite(inv) || !isfinite((float)stored))) return false;
    // The finite max value normalizes to <=127 plus fp32 rounding error (far
    // below the 127.5 roundf boundary); every signed-byte conversion is in range.
    y->d = stored;
    for (int j = 0; j < 32; j++) y->qs[j] = (int8_t)roundf(x[j] * inv);
    return true;
}
#ifdef __HIPCC__
// The same encode with one block per wave32 (lane l holds value l; all 32 lanes active): own_kv_q8_encode's statements,
// the max as a wave reduction (fmaxf is exact and order-free on the finite blocks that are stored; a non-finite value
// makes the max inf and refuses the block). store: write the block (34 bytes at y) or, refused, OR 2 into *err and leave
// its bytes alone. Returns the lane's decoded value, float code * float stored scale (exact in fp32), on every lane.
// The qwen3moe Q8 KV writers (kernels/k3_own_atts.h, kernels/q3m_prefill.hip) use it.
__device__ __forceinline__ float own_kv_q8_encode_wave(float x, uint8_t * y, int lane, bool store, int * err) {
#pragma clang fp contract(off)
    float amax = isfinite(x) ? fabsf(x) : INFINITY;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
    const float d = amax / 127.0f;
    const float inv = d ? 1.0f / d : 0.0f;
    const _Float16 stored = (_Float16)d;
    const bool refuse = !isfinite(amax) || (amax != 0.0f && (d == 0.0f || !isfinite(d) || !isfinite(inv) || !isfinite((float)stored)));
    const int8_t q = (int8_t)roundf(x * inv);
    if (store) {
        if (refuse) { if (lane == 0) atomicOr(err, 2); }
        else { y[2 + lane] = (uint8_t)q; if (lane == 0) *(_Float16 *)y = stored; }
    }
    return (float)q * (float)stored;
}
#endif
#undef OWN_KV_FN
#endif
