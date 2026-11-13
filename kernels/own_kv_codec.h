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
    for (int j = 0; j < 32; j++) {
        if (!isfinite(x[j])) return false;
        amax = fmaxf(amax, fabsf(x[j]));
    }
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
#undef OWN_KV_FN
#endif
