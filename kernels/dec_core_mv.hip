// K3 engine, M1 step 1: the decode matvec on int4 g128 weights, bit-identical to gemm_tn_iu4g.
// y[n] = rs * acc_n,  acc_n = fma(float(P_g), gs[n * G + g], acc_n) for g = 0 .. G-1 in order, P_g the exact int32 dot of
// group g's 128 int4 weight codes with the token's 128 int4 activation codes; then bf16 (RNE) or fp32. That is exactly
// what gemm_tn_iu4g computes for one row, so the decode matvec and the training GEMM give the same bits.
// Weight layout (repacked offline, caminus.k3.repack_mv): row n is K/2 bytes; each 32-bit word w covers 8 consecutive k,
// byte i holding u[k+i] (low nibble) and u[k+4+i] (high nibble) where u = code + 8 (offset binary, 0..15). So
// (w & 0x0F0F0F0F) and ((w >> 4) & 0x0F0F0F0F) are the unsigned bytes of k..k+3 and k+4..k+7, and v_dot4_i32_iu8
// (signed activations x unsigned weights) sums a * u; P_g = sum(a * u) - 8 * sum(a) over the group, exactly.
// Activations: the token's int4 codes as int8, natural order, plus per-group sums S_g = sum(a) over the group.
// One wave32 per row (M1 correctness form; the engine's MV_ROWS / MV_ROWGROUP forms come later): a 16-byte load per
// lane covers 32 k, 4 lanes a group, 8 groups per wave pass; int32 partials reduce across the group's 4 lanes (exact),
// then lane 0 runs the per-group FMA chain in order.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef int32_t i32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ uint16_t to_bf16(float v) {
    const uint32_t u = __builtin_bit_cast(uint32_t, v);
    return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

extern "C" __global__ void __launch_bounds__(256) k3_mv_g128(
        const uint8_t * __restrict__ W,      // [N, K/2] repacked offset-binary nibbles
        const float * __restrict__ gs,       // [N, G] group scales
        const int8_t * __restrict__ a,       // [K] activation codes (int8, -8..7)
        const int32_t * __restrict__ asum,   // [G] per-group sum of a
        float rs,                            // the token's activation row scale
        void * __restrict__ y, int N, int K, int out_f32) {
    const int lane = threadIdx.x % 32, n = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (n >= N) return;
    const int G = K / 128;
    const uint8_t * wr = W + (size_t)n * (K / 2);
    __shared__ int32_t pg_s[8][64];          // P_g per wave (G <= 64: K <= 8192)
    int32_t * pg = pg_s[threadIdx.x / 32];
    for (int k0 = 0; k0 < K; k0 += 1024) {   // a wave pass: 32 lanes x 32 k
        const int k = k0 + lane * 32;
        int32_t part = 0;
        if (k < K) {
            const u32x4 w = *(const u32x4 *)(wr + k / 2);
            const i32x4 a0 = *(const i32x4 *)(a + k), a1 = *(const i32x4 *)(a + k + 16);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t lo = w[j] & 0x0F0F0F0Fu, hi = (w[j] >> 4) & 0x0F0F0F0Fu;
                const int32_t al = j < 2 ? a0[2 * j] : a1[2 * (j - 2)], ah = j < 2 ? a0[2 * j + 1] : a1[2 * (j - 2) + 1];
                part = __builtin_amdgcn_sudot4(true, al, false, (int)lo, part, false);
                part = __builtin_amdgcn_sudot4(true, ah, false, (int)hi, part, false);
            }
        }
        part += __shfl_xor(part, 1);         // the group's 4 lanes (int32: exact in any order)
        part += __shfl_xor(part, 2);
        const int g = k / 128;
        if ((lane & 3) == 0 && k < K) pg[g] = part - 8 * asum[g];
    }
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
    if (lane == 0) {
        const float * sr = gs + (size_t)n * G;
        float acc = 0.f;
        for (int g = 0; g < G; g++) acc = __builtin_fmaf((float)pg[g], sr[g], acc);
        const float v = acc * rs;
        if (out_f32) ((float *)y)[n] = v; else ((uint16_t *)y)[n] = to_bf16(v);
    }
}
