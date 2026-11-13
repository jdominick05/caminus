// fp8_pack_probe.hip - standalone microprobe on gfx1100: can packed low-bit weights beat bf16 WMMA at prefill-GEMM
// shapes? One 64x64-tile WMMA loop (128 threads, 4 waves), K-contiguous operands, C[M,N] = A[M,K] . B[N,K]^T:
//   bf16   f32_16x16x16_bf16_w32 with bf16 A/B - the rate reference
//   iu8    i32_16x16x16_iu8_w32 with int8 A/B - gfx1100's only 8-bit WMMA (same K=16 per instruction as bf16, half the
//          load bytes; the fp8 WMMA builtins in the headers are gfx12/gfx950-only, none for gfx11)
//   pack   A stored as e4m3 bytes (half the bytes of bf16), each fp8 unpacked to bf16 in the fragment (v << 8, exact),
//          bf16 B, then the bf16 WMMA - the software-emulation route with its unpack ALU cost
// Timing only (C is stored, not checked). Compile:
//   ROCM_PATH/bin/hipcc -O3 --offload-arch=gfx1100 -o build/fp8_pack_probe benchmarks/fp8_pack_probe.hip
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

typedef uint16_t bf16_t;
typedef short s16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef int32_t i32x8 __attribute__((ext_vector_type(8)));

constexpr int BM = 64, BN = 64, BK = 32;      // bf16 K-stage (elements)
constexpr int SB = 32;                        // byte K-stage (elements)
constexpr int LDRB = BK * 2 + 16;             // padded bf16 LDS row, bytes (80)
constexpr int LDBB = SB + 16;                 // padded byte LDS row, bytes (48)
constexpr int WTM = BM / 2, WTN = BN / 2, TM = WTM / 16, TN = WTN / 16;

__device__ __forceinline__ s16x16 frag16(const bf16_t *p) {   // 16 consecutive bf16 from LDS
    const u32x4 lo = *(const u32x4 *)p, hi = *(const u32x4 *)(p + 8);
    return __builtin_bit_cast(s16x16, __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
}
__device__ __forceinline__ s16x16 frag_fp8(const uint8_t *p) { // 16 e4m3 bytes -> 16 bf16 (v << 8, exact)
    const u32x4 w = *(const u32x4 *)p;
    s16x16 r;
#pragma unroll
    for (int i = 0; i < 16; i++) r[i] = (short)(((w[i / 4] >> (8 * (i % 4))) & 0xFFu) << 8);
    return r;
}
__host__ __device__ __forceinline__ bf16_t to_bf16(float v) {
    const uint32_t u = __builtin_bit_cast(uint32_t, v);
    return (bf16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

// ARM 0: bf16 A/B. ARM 1: int8 A/B (both byte buffers, byte LDS). ARM 2: packed-e4m3 A (byte buffer) + bf16 B,
// bf16 wmma with frag_fp8 on A. Byte buffers are chunked as 16-byte loads: chunk c -> row c/2, offset (c%2)*16.
template <int ARM>
__global__ void gemm_probe(const void *A, const void *B, float *C, int M, int N, int K) {
    constexpr bool A_BYTES = (ARM != 0), B_BYTES = (ARM == 1);
    constexpr int LA = A_BYTES ? LDBB : LDRB, LB = B_BYTES ? LDBB : LDRB;
    constexpr int CA = A_BYTES ? BM * SB / 16 / 128 : BM * BK / 8 / 128;
    constexpr int CB = B_BYTES ? BN * SB / 16 / 128 : BN * BK / 8 / 128;
    constexpr int KST = A_BYTES || B_BYTES ? SB : BK;   // K-stage in elements
    __shared__ __attribute__((aligned(16))) uint8_t smem[2 * BM * 80 + 2 * BN * 80];
    uint8_t *sA = smem, *sB = smem + 2 * BM * LA;
    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32, wm = wave / 2, wn = wave % 2;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    u32x4 ra[CA], rb[CB];
    auto gload = [&](int k0) {
        if constexpr (A_BYTES) {   // byte buffer: 16B chunks, row = c/2, offset (c%2)*16 (SB = 32 bytes/stage)
            const uint8_t *Ak = (const uint8_t *)A + (size_t)m0 * K + k0;
#pragma unroll
            for (int j = 0; j < CA; j++) { const int c = tid + 128 * j; ra[j] = *(const u32x4 *)(Ak + (size_t)(c / 2) * K + (c % 2) * 16); }
        } else {                    // bf16 buffer: 16B chunks, row = c/4, offset (c%4)*8 elements
            const bf16_t *Ak = (const bf16_t *)A + (size_t)m0 * K + k0;
#pragma unroll
            for (int j = 0; j < CA; j++) { const int c = tid + 128 * j; ra[j] = *(const u32x4 *)(Ak + (size_t)(c / 4) * K + (c % 4) * 8); }
        }
        if constexpr (B_BYTES) {
            const uint8_t *Bk = (const uint8_t *)B + (size_t)n0 * K + k0;
#pragma unroll
            for (int j = 0; j < CB; j++) { const int c = tid + 128 * j; rb[j] = *(const u32x4 *)(Bk + (size_t)(c / 2) * K + (c % 2) * 16); }
        } else {
            const bf16_t *Bk = (const bf16_t *)B + (size_t)n0 * K + k0;
#pragma unroll
            for (int j = 0; j < CB; j++) { const int c = tid + 128 * j; rb[j] = *(const u32x4 *)(Bk + (size_t)(c / 4) * K + (c % 4) * 8); }
        }
    };
    auto sstore = [&](int buf) {
#pragma unroll
        for (int j = 0; j < CA; j++) {
            const int c = tid + 128 * j;
            if constexpr (A_BYTES) *(u32x4 *)&sA[buf * BM * LA + (c / 2) * LA + (c % 2) * 16] = ra[j];
            else *(u32x4 *)&sA[buf * BM * LA + (c / 4) * LA + (c % 4) * 8] = ra[j];
        }
#pragma unroll
        for (int j = 0; j < CB; j++) {
            const int c = tid + 128 * j;
            if constexpr (B_BYTES) *(u32x4 *)&sB[buf * BN * LB + (c / 2) * LB + (c % 2) * 16] = rb[j];
            else *(u32x4 *)&sB[buf * BN * LB + (c / 4) * LB + (c % 4) * 8] = rb[j];
        }
    };
    const int KT = K / KST;
    gload(0); sstore(0); __syncthreads();
    if constexpr (ARM == 1) {
        i32x8 acc[TM][TN];
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++) acc[i][j] = i32x8{0, 0, 0, 0, 0, 0, 0, 0};
        const uint8_t *pa = sA + (wm * WTM + lane % 16) * LA, *pb = sB + (wn * WTN + lane % 16) * LB;
        for (int kt = 0; kt < KT; kt++) {
            const int buf = kt & 1;
            if (kt + 1 < KT) gload((kt + 1) * KST);
#pragma unroll
            for (int ks = 0; ks < KST; ks += 16) {
                u32x4 af[TM], bfr[TN];
#pragma unroll
                for (int i = 0; i < TM; i++) af[i] = *(const u32x4 *)(pa + buf * BM * LA + i * 16 * LA + ks);
#pragma unroll
                for (int j = 0; j < TN; j++) bfr[j] = *(const u32x4 *)(pb + buf * BN * LB + j * 16 * LB + ks);
#pragma unroll
                for (int i = 0; i < TM; i++)
#pragma unroll
                    for (int j = 0; j < TN; j++)
                        acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[i], true, bfr[j], acc[i][j], false);
            }
            if (kt + 1 < KT) sstore(buf ^ 1);
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++)
#pragma unroll
                for (int r = 0; r < 8; r++) {
                    const int row = m0 + wm * WTM + 2 * r + lane / 16, col = n0 + wn * WTN + j * 16 + lane % 16;
                    if (row < M && col < N) C[(size_t)row * N + col] = (float)acc[i][j][r];
                }
    } else {
        f32x8 acc[TM][TN];
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++) acc[i][j] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
        const uint8_t *pa = sA + (wm * WTM + lane % 16) * LA;
        const uint8_t *pb = sB + (wn * WTN + lane % 16) * LB;
        for (int kt = 0; kt < KT; kt++) {
            const int buf = kt & 1;
            if (kt + 1 < KT) gload((kt + 1) * KST);
#pragma unroll
            for (int ks = 0; ks < KST; ks += 16) {
                s16x16 af[TM], bfr[TN];
#pragma unroll
                for (int i = 0; i < TM; i++)
                    af[i] = A_BYTES ? frag_fp8(pa + buf * BM * LA + i * 16 * LA + ks)
                                    : frag16((const bf16_t *)(pa + buf * BM * LA + i * 16 * LA + ks * 2));
#pragma unroll
                for (int j = 0; j < TN; j++) bfr[j] = frag16((const bf16_t *)(pb + buf * BN * LB + j * 16 * LB + ks * 2));
#pragma unroll
                for (int i = 0; i < TM; i++)
#pragma unroll
                    for (int j = 0; j < TN; j++) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[i], bfr[j], acc[i][j]);
            }
            if (kt + 1 < KT) sstore(buf ^ 1);
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++)
#pragma unroll
                for (int r = 0; r < 8; r++) {
                    const int row = m0 + wm * WTM + 2 * r + lane / 16, col = n0 + wn * WTN + j * 16 + lane % 16;
                    if (row < M && col < N) C[(size_t)row * N + col] = acc[i][j][r];
                }
    }
}

static double run_arm(int arm, const void *A, const void *B, float *C, int M, int N, int K, int reps) {
    dim3 grid(N / BN, M / BM);
    auto launch = [&]() {
        if (arm == 0) gemm_probe<0><<<grid, 128>>>(A, B, C, M, N, K);
        else if (arm == 1) gemm_probe<1><<<grid, 128>>>(A, B, C, M, N, K);
        else gemm_probe<2><<<grid, 128>>>(A, B, C, M, N, K);
    };
    for (int i = 0; i < 3; i++) launch();
    hipDeviceSynchronize();
    hipEvent_t t0, t1; hipEventCreate(&t0); hipEventCreate(&t1);
    hipEventRecord(t0);
    for (int i = 0; i < reps; i++) launch();
    hipEventRecord(t1); hipEventSynchronize(t1);
    float ms; hipEventElapsedTime(&ms, t0, t1);
    return ms / reps;
}

int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 20;
    hipDeviceProp_t prop; hipGetDeviceProperties(&prop, 0);
    printf("%s\n", prop.name);
    const int M = 2048, K = 5120;
    const int Ns[3] = {5120, 17408, 20480};
    std::vector<bf16_t> Ab(M * K), Bb(20480 * K);
    for (auto &v : Ab) v = to_bf16((rand() / (float)RAND_MAX - 0.5f) * 0.2f);
    for (auto &v : Bb) v = to_bf16((rand() / (float)RAND_MAX - 0.5f) * 0.2f);
    bf16_t *dA, *dB; uint8_t *dAi, *dAp, *dBi; float *dC;
    hipMalloc(&dA, (size_t)M * K * 2); hipMalloc(&dB, (size_t)20480 * K * 2);
    hipMalloc(&dAi, (size_t)M * K); hipMalloc(&dAp, (size_t)M * K); hipMalloc(&dBi, (size_t)20480 * K);
    hipMalloc(&dC, (size_t)M * 20480 * 4);
    hipMemcpy(dA, Ab.data(), (size_t)M * K * 2, hipMemcpyHostToDevice);
    hipMemcpy(dB, Bb.data(), (size_t)20480 * K * 2, hipMemcpyHostToDevice);
    std::vector<uint8_t> Ai(M * K), Bi(20480 * K);
    for (int i = 0; i < M * K; i++) Ai[i] = (uint8_t)(Ab[i] >> 8);
    for (int i = 0; i < 20480 * K; i++) Bi[i] = (uint8_t)(Bb[i] >> 8);
    hipMemcpy(dAi, Ai.data(), (size_t)M * K, hipMemcpyHostToDevice);
    hipMemcpy(dAp, Ai.data(), (size_t)M * K, hipMemcpyHostToDevice);   // e4m3 bits = bf16 bits >> 8
    hipMemcpy(dBi, Bi.data(), (size_t)20480 * K, hipMemcpyHostToDevice);
    for (int N : Ns) {
        double ms0 = run_arm(0, dA, dB, dC, M, N, K, reps);
        double ms1 = run_arm(1, dAi, dBi, dC, M, N, K, reps);
        double ms2 = run_arm(2, dAp, dB, dC, M, N, K, reps);
        double tf = 2.0 * M * N * K / 1e12;
        printf("PROBE {\"shape\": \"%dx%dx%d\", \"bf16_TF\": %.1f, \"iu8_TF\": %.1f, \"pack_fp8_TF\": %.1f, \"ms\": [%.2f, %.2f, %.2f]}\n",
               M, K, N, tf / (ms0 / 1e3), tf / (ms1 / 1e3), tf / (ms2 / 1e3), ms0, ms1, ms2);
        fflush(stdout);
    }
    printf("RESULT PASS (ran to the end; a measurement, not a gate)\n");
    return 0;
}
