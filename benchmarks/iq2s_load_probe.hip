// IQ2_S load-shape probe (gfx1100): what the decode tile's weight loads cost by shape, at ThinkingCap's IQ2_S ffn size
// (17408 x 5120: 2,785,280 units of 32 weights, 10 bytes each = 26.6 MiB, plus 2 bytes of d per 8 units).
// The own tile (kernels/own_quant_iq2s.h) reads a unit's 10-byte record with ONE 16-byte load at a 10-byte stride:
// 2-byte aligned, 6 of its 16 bytes another unit's. The arms read the same bytes with the tile's lane / unit mapping
// (lane = unit within a row, 32 lanes a wave, 8 waves a workgroup, a row per wave pass) and keep every loaded word live:
//   rec16   one 16-byte load at a 10-byte stride (today's layout)
//   planes  an 8-byte load at an 8-byte stride (4 index | 4 sign bytes) + a 2-byte load at a 2-byte stride (qh | scale)
//   planes4 the 8-byte plane, and the 2-byte plane read 4 units' at a time (an 8-byte load per lane group of 4, shuffled)
//   flat16  aligned 16-byte loads over the same byte count (the read ceiling of the shape)
// each arm also loads its block's d (2 bytes per 8 units), as the tile does. The passes rotate over --copies buffer sets
// (default 8: ~220 MiB, past the 80 / 96 MB Infinity Cache), every launch on a different copy, the arms interleaved, so each pass reads from memory as a decode token does. Prints us per pass and GB/s (bytes of the
// matrix / time), median of --reps. Memory: --copies x ~28 MiB (default ~220 MiB, far below either card's free line).
//   build: amdclang++ -x hip --offload-arch=gfx1100 -O3 -std=c++17 benchmarks/iq2s_load_probe.hip -o PROBE
//   run:   tools/guard.sh LOG 180 python -I benchmarks/iq2s_load_probe.py --probe PROBE --card N (checks the VRAM
//          through caminus.device.capacity first), or directly: HIP_VISIBLE_DEVICES=N tools/guard.sh LOG 120 PROBE --card 0 [--reps 50] [--rows 17408] [--k 5120] [--copies 8]
//          (N: the non-display card, `caminus ps`; --card indexes the visible devices)
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));

enum { REC16 = 0, PLANES = 1, PLANES4 = 2, FLAT16 = 3 };

template <int ARM>
__global__ void __launch_bounds__(256) probe(const unsigned char * __restrict__ A, const unsigned char * __restrict__ A2,
                                             const unsigned char * __restrict__ B, int rows, int nunits, unsigned * out) {
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const int NB = nunits / 8;
    unsigned acc = 0;
    for (int row = blockIdx.x * 8 + wave; row < rows; row += gridDim.x * 8) {
        for (int u0 = 0; u0 < nunits; u0 += 64) {          // two units per lane per pass, as the tile's UP = 2
            unsigned w[2][5];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int u = min(u0 + lane + 32 * i, nunits - 1);
                const size_t g = (size_t)row * nunits + u;
                if (ARM == REC16) {
                    const u32x4 v = *(const u32x4 *)(A + g * 10);
                    w[i][0] = v.x; w[i][1] = v.y; w[i][2] = v.z; w[i][3] = v.w;
                } else if (ARM == PLANES) {
                    const u32x2 v = *(const u32x2 *)(A + g * 8);
                    w[i][0] = v.x; w[i][1] = v.y; w[i][2] = *(const unsigned short *)(A2 + g * 2); w[i][3] = 0;
                } else if (ARM == PLANES4) {
                    const u32x2 v = *(const u32x2 *)(A + g * 8);
                    const u32x2 q = *(const u32x2 *)(A2 + (g & ~(size_t)3) * 2);   // 4 units' qh | scale pairs
                    const unsigned sel = (g & 2) ? q.y : q.x;
                    w[i][0] = v.x; w[i][1] = v.y; w[i][2] = (g & 1) ? sel >> 16 : sel & 0xffffu; w[i][3] = 0;
                } else {   // FLAT16: aligned 16-byte loads over exactly a pass's bytes (64 units x 10 B = 40 chunks: lanes
                           // 0 .. 31 then 0 .. 7; no chunk read twice)
                    const int j = lane + 32 * i;
                    if (j < 40) {
                        const size_t span = (size_t)nunits * 10 / 16;
                        const size_t q = (size_t)row * span + min((size_t)(u0 / 64 * 40 + j), span - 1);
                        const u32x4 v = *(const u32x4 *)(A + q * 16);
                        w[i][0] = v.x; w[i][1] = v.y; w[i][2] = v.z; w[i][3] = v.w;
                    } else { w[i][0] = w[i][1] = w[i][2] = w[i][3] = 0; }
                }
                w[i][4] = *(const unsigned short *)(B + ((size_t)row * NB + u / 8) * 2);
            }
#pragma unroll
            for (int i = 0; i < 2; i++) acc ^= w[i][0] + 3u * w[i][1] + 5u * w[i][2] + 7u * w[i][3] + 11u * w[i][4];
        }
    }
    if (acc == 0x9e3779b9u) out[0] = acc;   // (keeps the loads live)
}

int main(int argc, char ** argv) {
    int reps = 50, rows = 17408, K = 5120, copies = 8, card = -1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rows") && i + 1 < argc) rows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i + 1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--copies") && i + 1 < argc) copies = std::max(1, atoi(argv[++i]));
        else if (!strcmp(argv[i], "--card") && i + 1 < argc) card = atoi(argv[++i]);
    }
    if (card < 0) { printf("pass --card N (the HIP device index; run it on the non-display card first)\nRESULT FAIL\n"); return 2; }
    if (K % 256 || rows < 8) { printf("--k must be a multiple of 256 (whole IQ2_S blocks) and --rows >= 8\nRESULT FAIL\n"); return 2; }
    CK(hipSetDevice(card));
    const int nunits = K / 32;
    const size_t n = (size_t)rows * nunits, bytesA = n * 10 + 64, bytesB = n / 8 * 2 + 64;
    hipDeviceProp_t pr; CK(hipGetDeviceProperties(&pr, card));
    printf("%s, IQ2_S %d x %d: %.1f MiB of records + %.2f MiB of d\n", pr.name, rows, K, n * 10 / 1048576.0, n / 4 / 1048576.0);
    {   // (the launcher, benchmarks/iq2s_load_probe.py, checks this through caminus.device.capacity first; here the
        //  card's own free memory too, keeping 1 GiB: a display card's 512 MiB and the guard's margin)
        size_t fr = 0, tot = 0; CK(hipMemGetInfo(&fr, &tot));
        const size_t need = (size_t)copies * (bytesA + bytesB) + 4;
        if (need + ((size_t)1 << 30) > fr) { printf("needs %zu MiB, %zu MiB free (keeping 1 GiB)\nRESULT FAIL\n", need >> 20, fr >> 20); return 2; }
    }
    std::vector<unsigned char *> As(copies), Bs(copies); unsigned * out;
    std::vector<unsigned char> h(bytesA);
    for (size_t i = 0; i < bytesA; i++) h[i] = (unsigned char)(i * 2654435761u >> 13);
    for (int c = 0; c < copies; c++) {
        CK(hipMalloc(&As[c], bytesA)); CK(hipMalloc(&Bs[c], bytesB));
        CK(hipMemcpy(As[c], h.data(), bytesA, hipMemcpyHostToDevice)); CK(hipMemcpy(Bs[c], h.data(), bytesB, hipMemcpyHostToDevice));
    }
    CK(hipMalloc(&out, 4));
    const int grid = std::min(rows / 8, 96 * 2);   // ~2 workgroups per CU on either card, a row per wave pass
    const char * names[4] = {"rec16", "planes", "planes4", "flat16"};
    hipEvent_t t0, t1; CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
    std::vector<float> ms[4];
    for (int r = 0; r < reps + 3; r++) {                 // the arms interleaved: each rep runs all four in turn
        for (int arm = 0; arm < 4; arm++) {
            const int c = (r * 4 + arm) % copies;          // (every launch a different copy: cold memory each pass)
            unsigned char * A = As[c], * B = Bs[c];
            const unsigned char * A2 = A + n * 8;   // (the planes arms: the 8-byte plane, then the 2-byte plane, same buffer)
            CK(hipEventRecord(t0));
            if (arm == REC16) probe<REC16><<<grid, 256>>>(A, A2, B, rows, nunits, out);
            else if (arm == PLANES) probe<PLANES><<<grid, 256>>>(A, A2, B, rows, nunits, out);
            else if (arm == PLANES4) probe<PLANES4><<<grid, 256>>>(A, A2, B, rows, nunits, out);
            else probe<FLAT16><<<grid, 256>>>(A, A2, B, rows, nunits, out);
            CK(hipEventRecord(t1)); CK(hipEventSynchronize(t1));
            float t; CK(hipEventElapsedTime(&t, t0, t1)); if (r >= 3) ms[arm].push_back(t);
        }
    }
    for (int arm = 0; arm < 4; arm++) {
        std::sort(ms[arm].begin(), ms[arm].end());
        const double us = ms[arm][ms[arm].size() / 2] * 1000.0;
        printf("%-8s %8.1f us  %6.0f GB/s\n", names[arm], us, (n * 10.0 + n / 4.0) / (us * 1e3));
    }
    CK(hipGetLastError());
    printf("RESULT PASS\n");
    return 0;
}
