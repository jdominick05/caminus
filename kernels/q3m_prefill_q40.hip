// The batched prefill for the Q4_0 family (the official GPTQ-Int4 checkpoint converted to Q4_0, tools/eval/gptq_to_gguf.py;
// the decode modules kernels/k3_q3m_q40*.hip): kernels/q3m_prefill.hip's kernels as they are, plus the Q4_0 GEMM. A
// separate object, so the shipped prefill object keeps its code. The Q8 KV cache's and the long-context mode's qk_rope /
// attention stay on their own objects (q3m_prefill_kvq8, q3m_prefill_yarn, ...): they read no weight.
//
// ---- q3p_mm_q40: q3p_mm_q4k's contract (Y[ydst[i]][r] = X[xsrc[i]] . W_e[r] in fp32, fp16 activations through WMMA, the
// same work items and tiles) ---------------------------------------------------------------------------------------------
// The decode engine's own weight copy is read in place: the plane layout (caminus.engine.own_layout.q4_0_planes: plane A
// the 16 code bytes of every 32-value unit, plane B every unit's d (fp16), each over the whole buffer of Rtot rows, row
// major). Q4_0: w = d (q - 8), byte k of a unit holding value k (low nibble) and value 16 + k (high nibble). B = q - 8
// (fp16 (1024 + q) - 1032: exact), the unit's d stays outside B (d (q - 8) is not an fp16 for every fp16 d: 11 bits of
// mantissa times up to 4), so per unit P = the two 16-value products chained in one accumulator (low half, then high)
// and acc += d P. A product's only rounding is the fp32 accumulation, as in q3p_mm_q4k / q3p_mm_q6k (not the decode
// tile's order: G-model gated, as they are).
// iu8 / iu4 WMMA are not used: on gfx1100 iu8 runs at the fp16 rate (and needs int8 activations), and iu4 needs 4-bit
// activations the checkpoint was not calibrated for.
#include "q3m_prefill.hip"

// 16 nibbles (4 dwords, shift sh: 0 the low half of a unit, 4 the high) -> fp16 q - 8 (exact) as one B operand
__device__ __forceinline__ h16_t q40_b(const u32x4_t & qw, int sh) {
    const h2_t o2 = {(_Float16)(-1032.0f), (_Float16)(-1032.0f)};                  // 1024 + 8
    const uint32_t w4[4] = {qw.x, qw.y, qw.z, qw.w};
    uint32_t o[8];
#pragma unroll
    for (int w = 0; w < 4; w++) {
        const uint32_t n = (w4[w] >> sh) & 0x0f0f0f0fu;
        const uint32_t p01 = __builtin_amdgcn_perm(0u, n, 0x0c010c00u) | 0x64006400u;   // fp16 1024 + q (bytes 0, 1)
        const uint32_t p23 = __builtin_amdgcn_perm(0u, n, 0x0c030c02u) | 0x64006400u;
        o[2 * w] = as_u(as_h2(p01) + o2); o[2 * w + 1] = as_u(as_h2(p23) + o2);
    }
    return __builtin_bit_cast(h16_t, o);
}

extern "C" __global__ void __launch_bounds__(256) q3p_mm_q40(
        const uint8_t * __restrict__ W, int R, int Rtot, int K, const _Float16 * __restrict__ X, int ldx, const int * __restrict__ xsrc,
        float * __restrict__ Y, int ldy, const int * __restrict__ ydst, const MMItem * __restrict__ items, const int * __restrict__ nitems,
        int ntok) {
    MMItem it;
    if (items) { if ((int)blockIdx.x >= *nitems) return; it = items[blockIdx.x]; }
    else { it.e = 0; it.off = blockIdx.x * MM_M; it.cnt = min(MM_M, ntok - it.off); }
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int r = blockIdx.y * MM_N + wave * 16 + (lane & 15);
    const bool rok = r < R;
    const int NB = K / 256, NU = K / 32;
    const size_t g = (size_t)it.e * R + (rok ? r : 0);
    const u32x4_t * pa = (const u32x4_t *)(W + g * NU * 16);
    const uint16_t * pd = (const uint16_t *)(W + (size_t)Rtot * NU * 16) + g * NU;
    f8_t acc[4];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) acc[mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
    for (int b = 0; b < NB; b++) {
        __syncthreads();
        mm_stage<false>(X, ldx, xsrc, it.off, it.cnt, b, tid);
        __syncthreads();
#pragma unroll 1
        for (int j = 0; j < 8; j++) {                                 // the block's 8 units of 32 values
            const u32x4_t qw = pa[b * 8 + j];
            const float d = rok ? (float)__builtin_bit_cast(_Float16, pd[b * 8 + j]) : 0.0f;
            f8_t P[4];
            const h16_t B0 = q40_b(qw, 0);                            // values 0 .. 15 of the unit, every token tile
#pragma unroll
            for (int mt = 0; mt < 4; mt++) P[mt] = wmma(mm_a(mt, j * 32, lane), B0, (f8_t){0, 0, 0, 0, 0, 0, 0, 0});
            __builtin_amdgcn_sched_barrier(0);
            const h16_t B1 = q40_b(qw, 4);                            // values 16 .. 31 into the same products
#pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                P[mt] = wmma(mm_a(mt, j * 32 + 16, lane), B1, P[mt]);
#pragma unroll
                for (int v = 0; v < 8; v++) acc[mt][v] = __builtin_fmaf(d, P[mt][v], acc[mt][v]);
            }
            // (one half at a time, as q3p_mm_q6k's sub-blocks: the scheduler otherwise hoists every unit's loads and decodes)
            __builtin_amdgcn_sched_barrier(0);
        }
    }
    if (!rok) return;
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int v = 0; v < 8; v++) {
            const int t = mt * 16 + 2 * v + (lane >> 4);
            if (t < it.cnt) Y[(size_t)(ydst ? ydst[it.off + t] : it.off + t) * ldy + r] = acc[mt][v];
        }
}
