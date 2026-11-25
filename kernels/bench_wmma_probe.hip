// Probe of gfx1100's iu4 WMMA (v_wmma_i32_16x16x16_iu4_w32): D[16x16] = A[16x16] . B[16x16]^T on int4 codes, one wave.
// Inputs as int8 codes in [-8, 7]: a[m][k] (row-major M x K), b[n][k] (N x K). `mode` bit 0 = pack element k of a lane's
// 16 codes into nibble k (low nibble first) else nibble 15 - k; bit 1 = leave lanes 16-31 zero instead of duplicating
// lanes 0-15 (tests whether the hardware reads the upper half-wave's A/B). D is written row-major from the documented
// C layout: VGPR r of lane l holds D[2r + l / 16][l % 16].
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));

__device__ __forceinline__ v2i pack16(const int8_t * c, int mode) {
    uint32_t w[2] = {0, 0};
    for (int k = 0; k < 16; k++) {
        const int slot = (mode & 1) ? k : 15 - k;
        w[slot / 8] |= ((uint32_t)c[k] & 0xF) << (4 * (slot % 8));
    }
    v2i r; r.x = (int)w[0]; r.y = (int)w[1];
    return r;
}

extern "C" __global__ void __launch_bounds__(32) iu4_probe(const int8_t * a, const int8_t * b, int32_t * d, int mode) {
    const int l = threadIdx.x, r16 = l % 16;
    v2i A = pack16(a + r16 * 16, mode), B = pack16(b + r16 * 16, mode);
    if ((mode & 2) && l >= 16) { A.x = A.y = 0; B.x = B.y = 0; }
    v8i C = {0, 0, 0, 0, 0, 0, 0, 0};
    C = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, A, true, B, C, false);
    for (int r = 0; r < 8; r++) d[(2 * r + l / 16) * 16 + r16] = C[r];
}

// bf16 WMMA probe: D = A . B^T + C for one 16 x 16 x 16 tile, a [16][16] and b [16][16] bf16 (raw uint16), c [16][16]
// fp32 in the documented C layout; `iu8` = 1 runs the same codes through v_wmma_i32_16x16x16_iu8 instead (codes in int8)
typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef int v4i __attribute__((ext_vector_type(4)));
extern "C" __global__ void __launch_bounds__(32) bf16_probe(const uint16_t * a, const uint16_t * b, const float * c, float * d) {
    const int l = threadIdx.x, r16 = l % 16;
    v16s A, B; v8f C;
    for (int k = 0; k < 16; k++) { A[k] = (short)a[r16 * 16 + k]; B[k] = (short)b[r16 * 16 + k]; }
    for (int r = 0; r < 8; r++) C[r] = c[(2 * r + l / 16) * 16 + r16];
    C = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(A, B, C);
    for (int r = 0; r < 8; r++) d[(2 * r + l / 16) * 16 + r16] = C[r];
}
extern "C" __global__ void __launch_bounds__(32) iu8_probe(const int8_t * a, const int8_t * b, int32_t * d) {
    const int l = threadIdx.x, r16 = l % 16;
    v4i A, B;
    for (int w = 0; w < 4; w++) {
        uint32_t x = 0, y = 0;
        for (int j = 0; j < 4; j++) { x |= (uint32_t)(uint8_t)a[r16 * 16 + 4 * w + j] << (8 * j); y |= (uint32_t)(uint8_t)b[r16 * 16 + 4 * w + j] << (8 * j); }
        A[w] = (int)x; B[w] = (int)y;
    }
    v8i C = {0, 0, 0, 0, 0, 0, 0, 0};
    C = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, A, true, B, C, false);
    for (int r = 0; r < 8; r++) d[(2 * r + l / 16) * 16 + r16] = C[r];
}
