// The orthonormal Walsh-Hadamard transform of a 128-value head (Sylvester order) for the qwen3moe Q8 KV cache's K and Q:
// caminus.ref_kv.hadamard128 is its CPU reference, bit for bit. In exact arithmetic (H k) . (H q) = k . q (H orthonormal),
// so the scores do not change; the Q8_0 blocks of H k spread each head's outlier dims over all 128 values, which cuts the
// encode's error (a CPU float64 model of Qwen3-30B-A3B on the 128-token reference prompt: mean KLD against the float64
// reference 3.4e-3 .. 5.0e-3 unrotated, 1.65e-3 .. 3.0e-3 with K and Q rotated, over 21 draws).
//
// The butterflies and their order are the dense line's (kernels/ops_q35_kvq8.h q35_op_hadamard): for h = 1, 2, 4 .. 64
// the pair (i, i + h) of i with bit h clear becomes (a + b, a - b), then each value times 1/sqrt(128) rounded to fp32
// (0x1.6a09e6p-4: not a power of two, unlike the dense line's 64 and 256, so the product is rounded once more).
// Layout: a wave's lane l holds the values at dims l, l + 32, l + 64, l + 96 (u0 .. u3) and returns dim l + 32 qw's result;
// the stages h = 1 .. 16 pair lanes (one shuffle per value), h = 32 and 64 pair the lane's own values. Every wave of a head
// computes all four quarters (the same statements, so the same bits), which needs no LDS buffer and no barrier: four times
// the shuffles of one quarter. All 32 lanes must be active.
#pragma once
#define OWN_KV_ROT128_SCALE 0x1.6a09e6p-4f   // (float)(1 / sqrt(128)), caminus.ref_kv.HADAMARD128_SCALE

// x (an fp32 result, e.g. RoPE's fma) rounded to fp16 and back, as the F16 objects store it: the opaque copy keeps the
// backend from fusing the fma into v_fma_mix*_f16, which rounds the exact value to fp16 once instead of fp32 then fp16
__device__ __forceinline__ float own_kv_f16r(float x) { asm volatile("" : "+v"(x)); return (float)(_Float16)x; }

__device__ __forceinline__ float own_kv_rot128(float u0, float u1, float u2, float u3, int qw, int lane) {
#pragma clang fp contract(off)
#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        const bool hi = (lane & h) != 0;   // (this lane is the pair's i + h: a - b = partner - own)
        const float p0 = __shfl_xor(u0, h, 32), p1 = __shfl_xor(u1, h, 32), p2 = __shfl_xor(u2, h, 32), p3 = __shfl_xor(u3, h, 32);
        u0 = hi ? p0 - u0 : u0 + p0; u1 = hi ? p1 - u1 : u1 + p1;
        u2 = hi ? p2 - u2 : u2 + p2; u3 = hi ? p3 - u3 : u3 + p3;
    }
    const float a0 = u0 + u1, a1 = u0 - u1, a2 = u2 + u3, a3 = u2 - u3;   // h = 32: dims (l, l + 32), (l + 64, l + 96)
    const float x = qw == 0 ? a0 + a2 : qw == 1 ? a1 + a3 : qw == 2 ? a0 - a2 : a1 - a3;   // h = 64
    return x * OWN_KV_ROT128_SCALE;
}
