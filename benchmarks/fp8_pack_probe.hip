// fp8_pack_probe.hip - standalone microprobe on gfx1100: can packed low-bit weights beat bf16 WMMA at prefill-GEMM
// shapes? One 64x64-tile WMMA loop (128 threads, 4 waves), K-contiguous operands, C[M,N] = A[M,K] . B[N,K]^T:
//   bf16   f32_16x16x16_bf16_w32 with bf16 A/B - the rate reference
//   iu8    i32_16x16x16_iu8_w32 with int8 A/B - gfx1100's only 8-bit WMMA (same K=16 per instruction as bf16, half the
//          load bytes; the fp8 WMMA builtins in the headers are gfx12/gfx950-only, none for gfx11)
//   pack   A stored as e4m3 bytes (half the bytes of bf16), each fp8 decoded to a real E4M3 value and rounded to bf16
//          in the fragment, bf16 B, then the bf16 WMMA - the software-emulation route with its unpack ALU cost
// --check additionally validates the 64x64 corner of C per arm against a fp64 CPU reference. Compile:
//   ROCM_PATH/bin/hipcc -O3 --offload-arch=gfx1100 -o build/fp8_pack_probe benchmarks/fp8_pack_probe.hip
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <cmath>

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

__host__ __device__ __forceinline__ bf16_t to_bf16(float v) {
    const uint32_t u = __builtin_bit_cast(uint32_t, v);
    return (bf16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

__device__ __forceinline__ s16x16 frag16(const bf16_t *p) {   // 16 consecutive bf16 from LDS
    const u32x4 lo = *(const u32x4 *)p, hi = *(const u32x4 *)(p + 8);
    return __builtin_bit_cast(s16x16, __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
}
// E4M3 (OCP e4m3fn: no inf, 0x7F/0xFF = NaN), bias 7, 3 mantissa bits. Decode is exact; encode rounds to nearest even.
__host__ __device__ __forceinline__ float e4m3_to_f32(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 0xF, m = b & 7;
    float v;
    if (e == 0) v = ldexpf((float)m, -9);                 // subnormal: m * 2^-9
    else if (e == 15 && m == 7) v = __builtin_nanf("");   // NaN
    else v = ldexpf((float)(8 + m), e - 10);              // (1 + m/8) * 2^(e-7)
    return s ? -v : v;
}
__host__ __device__ __forceinline__ uint8_t f32_to_e4m3(float v) {
    const uint8_t sign = (uint8_t)(__builtin_bit_cast(uint32_t, v) >> 31) << 7;
    const float a = fabsf(v);
    if (a != a) return 0x7F;                              // NaN in, NaN out
    if (a == 0.f) return sign;
    int ee; const float frac = frexpf(a, &ee);            // a = frac * 2^ee, frac in [0.5, 1)
    const int E = ee + 6;                                 // e4m3 exponent field of the normal encoding
    if (E >= 1) {                                         // normal: a = (1 + m/8) * 2^(E-7)
        const float mf = (frac * 2.f - 1.f) * 8.f;
        int m = (int)mf; const float r = mf - m;
        if (r > 0.5f || (r == 0.5f && (m & 1))) m++;
        if (m == 8) {                                     // mantissa carry: bump the exponent
            if (E + 1 > 15) return sign | 0x7F;           // past max finite 448 -> NaN (e4m3fn has no inf)
            return sign | (uint8_t)((E + 1) << 3);
        }
        return sign | (uint8_t)((E << 3) | m);
    }
    const float mf = a * 512.f;                           // subnormal: a = m * 2^-9
    int m = (int)mf; const float r = mf - m;
    if (r > 0.5f || (r == 0.5f && (m & 1))) m++;
    if (m >= 8) return sign | 0x08;                       // rounded up to the smallest normal 2^-6
    return sign | (uint8_t)m;
}
__device__ __forceinline__ s16x16 frag_fp8(const uint8_t *p) { // 16 e4m3 bytes -> 16 bf16 (real E4M3 decode + bf16 round)
    const u32x4 w = *(const u32x4 *)p;
    s16x16 r;
#pragma unroll
    for (int i = 0; i < 16; i++) r[i] = (short)to_bf16(e4m3_to_f32((uint8_t)(w[i / 4] >> (8 * (i % 4)))));
    return r;
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
            else *(u32x4 *)&sA[buf * BM * LA + (c / 4) * LA + (c % 4) * 16] = ra[j];   // 8 bf16 = 16 bytes
        }
#pragma unroll
        for (int j = 0; j < CB; j++) {
            const int c = tid + 128 * j;
            if constexpr (B_BYTES) *(u32x4 *)&sB[buf * BN * LB + (c / 2) * LB + (c % 2) * 16] = rb[j];
            else *(u32x4 *)&sB[buf * BN * LB + (c / 4) * LB + (c % 4) * 16] = rb[j];   // 8 bf16 = 16 bytes
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
                    const int row = m0 + wm * WTM + i * 16 + 2 * r + lane / 16, col = n0 + wn * WTN + j * 16 + lane % 16;
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
                    const int row = m0 + wm * WTM + i * 16 + 2 * r + lane / 16, col = n0 + wn * WTN + j * 16 + lane % 16;
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

static float bf16_to_f32(uint16_t b) { uint32_t u = (uint32_t)b << 16; return __builtin_bit_cast(float, u); }

// CPU reference for the 64x64 corner of C, with each arm's operand values, fp64 accumulation; prints max abs
// error and any NaN/inf in the corner - catches OOB/garbage stores, not just wrong math.
static void check_corner(int arm, const float *C, const bf16_t *Ab, const uint8_t *Ai, const uint8_t *Bi,
                         const bf16_t *Bb, int M, int N, int K) {
    double maxerr = 0; int bad = 0;
    for (int m = 0; m < 64; m++)
        for (int n = 0; n < 64; n++) {
            double s = 0;
            for (int k = 0; k < K; k++) {
                double a, b = bf16_to_f32(Bb[(size_t)n * K + k]);
                if (arm == 0) a = bf16_to_f32(Ab[(size_t)m * K + k]);
                else if (arm == 1) { a = (int8_t)Ai[(size_t)m * K + k]; b = (int8_t)Bi[(size_t)n * K + k]; }
                else a = (double)e4m3_to_f32(Ai[(size_t)m * K + k]);
                s += a * b;
            }
            float c = C[(size_t)m * N + n];
            if (!std::isfinite(c)) bad++;
            double e = std::fabs((double)c - s);
            if (e > maxerr) maxerr = e;
        }
    printf("CHECK {\"arm\": %d, \"corner\": \"64x64\", \"max_abs_err_vs_fp64\": %.6g, \"nan_inf\": %d}\n", arm, maxerr, bad);
    fflush(stdout);
}

int main(int argc, char **argv) {
    int reps = 20, check = 0;
    std::vector<int> Ns;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--shapes") && i + 1 < argc) { char *s = strtok(argv[++i], ","); while (s) { Ns.push_back(atoi(s)); s = strtok(nullptr, ","); } }
        else if (!strcmp(argv[i], "--check")) check = 1;
        else reps = atoi(argv[i]);
    }
    if (Ns.empty()) Ns = {5120, 17408, 20480};
    for (int N : Ns)
        if (N <= 0 || N % BN != 0 || N > 20480) {
            fprintf(stderr, "bad --shapes N=%d: the fixed buffers hold 0 < N <= 20480 with N %% %d == 0\n", N, BN);
            return 2;
        }
    hipDeviceProp_t prop; hipGetDeviceProperties(&prop, 0);
    printf("%s\n", prop.name);
    const int M = 2048, K = 5120;
    // project the probe's peak device allocation and check it against free VRAM (keeping 512 MiB free) before allocating
    const size_t projected = (size_t)M * K * 2 + (size_t)20480 * K * 2 + 2 * (size_t)M * K + (size_t)20480 * K
                           + (size_t)M * 20480 * 4;   // dA dB dAi dAp dBi dC
    size_t freeB, totalB; hipMemGetInfo(&freeB, &totalB);
    const size_t margin = (size_t)512 << 20;
    printf("VRAM free %.0f MiB of %.0f; probe needs %.0f MiB (+ %.0f MiB kept free)\n",
           freeB / 1048576.0, totalB / 1048576.0, projected / 1048576.0, margin / 1048576.0);
    if (freeB < projected + margin) { fprintf(stderr, "refusing to allocate: would cross the 512 MiB-free line\n"); return 2; }
    std::vector<float> Af(M * K), Bf(20480 * K);
    for (auto &v : Af) v = (rand() / (float)RAND_MAX - 0.5f) * 0.2f;
    for (auto &v : Bf) v = (rand() / (float)RAND_MAX - 0.5f) * 0.2f;
    std::vector<bf16_t> Ab(M * K), Bb(20480 * K);
    for (int i = 0; i < M * K; i++) Ab[i] = to_bf16(Af[i]);
    for (int i = 0; i < 20480 * K; i++) Bb[i] = to_bf16(Bf[i]);
    bf16_t *dA, *dB; uint8_t *dAi, *dAp, *dBi; float *dC;
    hipMalloc(&dA, (size_t)M * K * 2); hipMalloc(&dB, (size_t)20480 * K * 2);
    hipMalloc(&dAi, (size_t)M * K); hipMalloc(&dAp, (size_t)M * K); hipMalloc(&dBi, (size_t)20480 * K);
    hipMalloc(&dC, (size_t)M * 20480 * 4);
    hipMemcpy(dA, Ab.data(), (size_t)M * K * 2, hipMemcpyHostToDevice);
    hipMemcpy(dB, Bb.data(), (size_t)20480 * K * 2, hipMemcpyHostToDevice);
    std::vector<uint8_t> Ai(M * K), Bi(20480 * K);
    for (int i = 0; i < M * K; i++) Ai[i] = f32_to_e4m3(Af[i]);       // int8 arm: the fp32 truncated to 8 bits
    for (int i = 0; i < 20480 * K; i++) Bi[i] = f32_to_e4m3(Bf[i]);
    hipMemcpy(dAi, Ai.data(), (size_t)M * K, hipMemcpyHostToDevice);
    hipMemcpy(dAp, Ai.data(), (size_t)M * K, hipMemcpyHostToDevice);   // same e4m3 bytes for the pack arm
    hipMemcpy(dBi, Bi.data(), (size_t)20480 * K, hipMemcpyHostToDevice);
    std::vector<float> hc((size_t)M * 20480);
    for (int N : Ns) {
        double ms[3];
        const void *aa[3] = {dA, dAi, dAp}, *bb[3] = {dB, dBi, dB};
        for (int arm = 0; arm < 3; arm++) {
            ms[arm] = run_arm(arm, aa[arm], bb[arm], dC, M, N, K, reps);
            if (check) {
                hipMemcpy(hc.data(), dC, (size_t)M * N * 4, hipMemcpyDeviceToHost);
                check_corner(arm, hc.data(), Ab.data(), Ai.data(), Bi.data(), Bb.data(), M, N, K);
            }
        }
        double tf = 2.0 * M * N * K / 1e12;
        printf("PROBE {\"shape\": \"%dx%dx%d\", \"bf16_TF\": %.1f, \"iu8_TF\": %.1f, \"pack_fp8_TF\": %.1f, \"ms\": [%.2f, %.2f, %.2f]}\n",
               M, K, N, tf / (ms[0] / 1e3), tf / (ms[1] / 1e3), tf / (ms[2] / 1e3), ms[0], ms[1], ms[2]);
        fflush(stdout);
    }
    printf("RESULT PASS (ran to the end; a measurement, not a gate)\n");
    return 0;
}
