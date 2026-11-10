// The batched prefill of Qwen3-30B-A3B (qwen3moe) on Caminus's own kernels: no ggml-cuda code, written from the GGUF
// block layouts (the file format). Positions [a, b) of a prompt run through every layer at once and write their KV rows;
// the prompt's last token then runs as a K3 decode pass (python/caminus/prefill/q3m.py). This file: the quantized GEMMs.
//
// ---- q3p_mm_q4k / q3p_mm_q6k: Y[ydst[i]][r] = X[xsrc[i]] . W_e[r] for the tokens i of a work item, in fp32 -----------------
// The decode engine's own weight copies are read in place (no second copy on the card): Q4_K raw GGUF blocks (144 B per 256
// values), Q6_K in the decode's plane layout (engine/own_layout.q6k_planes: per row and block ql | qh 192 B, then the
// reordered scales 16 B, then d). Activations fp16 (the decode's precision), products and sums fp32 through RDNA3's WMMA
// (v_wmma_f32_16x16x16_f16), the weights EXACT in the B operand:
//   Q4_K  w = d sc q - dmin m: B = sc * q (an integer <= 945: exact in fp16) over the whole 256-value block, then
//         acc += d * P - dmin * MS, with MS = sum over the 8 sub-blocks of m * (sum of x over the sub-block) as one more
//         WMMA: A = the per-token sub-block sums split hi + lo in fp16 (8 + 8 columns, error ~2^-22 of the sum), B = m twice
//   Q6_K  w = d sc (q - 32): B = q - 32 (exact) per 16-value sub-block, acc += (d * sc) * P
// so a product's only rounding is the fp32 accumulation (the decode's matvec: fp16 x, fp32 sums; llama.cpp's MMQ: int8 x).
// A workgroup (8 waves) computes 64 tokens x 128 rows: wave w rows 16 w .. 16 w + 15 (lane l: row l & 15; lanes 16 .. 31
// hold the same B, as WMMA's wave32 operands require) for all 64 tokens (4 WMMA tiles reuse each decoded B). The tokens'
// x for one 256-value block is staged in LDS (64 x 256 fp16, padded rows) with the Q4_K sub-block sums.
// Work: blockIdx.x = a work item (expert e, offset o, count c <= 64: tokens list[o .. o + c)), blockIdx.y = the 128-row
// tile; item index >= *nitems: the workgroup exits (the host launches the bound, the device knows the count). Dense
// matrices: items = nullptr, ntok tokens, blockIdx.x = tokens 64 x .. 64 x + 63, identity lists (xsrc / ydst unused).
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef _Float16 h2_t __attribute__((ext_vector_type(2)));
typedef _Float16 h16_t __attribute__((ext_vector_type(16)));
typedef float f8_t __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));

#define MM_M 64          // tokens per workgroup
#define MM_N 128         // rows per workgroup (8 waves x 16)
#define MM_XP 264        // fp16 per staged x row (256 + 8: rows start in different banks)

struct MMItem { int e, off, cnt, pad; };

__device__ __forceinline__ h2_t as_h2(uint32_t u) { return __builtin_bit_cast(h2_t, u); }
__device__ __forceinline__ uint32_t as_u(h2_t h) { return __builtin_bit_cast(uint32_t, h); }
struct U2 { u32x4_t a, b; };
__device__ __forceinline__ h16_t h16of(u32x4_t a, u32x4_t b) { return __builtin_bit_cast(h16_t, (U2){a, b}); }
__device__ __forceinline__ f8_t wmma(const h16_t & a, const h16_t & b, const f8_t & c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// LDS: x [64][MM_XP] fp16; Q4_K sub-block sums [64][16] fp16 (hi 0..7, lo 8..15)
__shared__ _Float16 MMX[MM_M * MM_XP];
__shared__ _Float16 MMS[MM_M * 16];

// stage block b of the item's tokens: thread (token t = tid / 4, quarter q = tid % 4: values 64 q .. 64 q + 63); with
// sums: the fp32 sums of sub-blocks 2 q and 2 q + 1 (32 values each), split hi + lo fp16
template <bool SUMS>
__device__ __forceinline__ void mm_stage(const _Float16 * __restrict__ X, int ldx, const int * __restrict__ xsrc, int off, int cnt,
                                         int b, int tid) {
    const int t = tid >> 2, q = tid & 3;
    u32x4_t v[8];
    if (t < cnt) {
        const u32x4_t * src = (const u32x4_t *)(X + (size_t)(xsrc ? xsrc[off + t] : off + t) * ldx + b * 256 + q * 64);
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = src[i];
    } else {
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = (u32x4_t){0u, 0u, 0u, 0u};
    }
    u32x4_t * dst = (u32x4_t *)(MMX + t * MM_XP + q * 64);
#pragma unroll
    for (int i = 0; i < 8; i++) dst[i] = v[i];
    if (SUMS) {
        const h2_t one = {(_Float16)1.0f, (_Float16)1.0f};
#pragma unroll
        for (int hsb = 0; hsb < 2; hsb++) {
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const u32x4_t w = v[hsb * 4 + i];
                s = __builtin_amdgcn_fdot2(as_h2(w.x), one, s, false); s = __builtin_amdgcn_fdot2(as_h2(w.y), one, s, false);
                s = __builtin_amdgcn_fdot2(as_h2(w.z), one, s, false); s = __builtin_amdgcn_fdot2(as_h2(w.w), one, s, false);
            }
            const _Float16 hi = (_Float16)s; const _Float16 lo = (_Float16)(s - (float)hi);
            MMS[t * 16 + 2 * q + hsb] = hi; MMS[t * 16 + 8 + 2 * q + hsb] = lo;
        }
    }
}

// the A operand of tile mt for k values k0 .. k0 + 15: lane l's token row mt * 16 + (l & 15)
__device__ __forceinline__ h16_t mm_a(int mt, int k0, int lane) {
    const u32x4_t * p = (const u32x4_t *)(MMX + (mt * 16 + (lane & 15)) * MM_XP + k0);
    return h16of(p[0], p[1]);
}

// 16 nibbles (4 dwords, shift sh: 0 low, 4 high) -> fp16 values sc * q (exact: q <= 15, sc <= 63) as one B operand
__device__ __forceinline__ h16_t q4k_b(const u32x4_t & qw, int sh, _Float16 sc) {
    const h2_t s2 = {sc, sc}, o2 = {(_Float16)(-1024.0f) * sc, (_Float16)(-1024.0f) * sc};
    const uint32_t w4[4] = {qw.x, qw.y, qw.z, qw.w};
    uint32_t o[8];
#pragma unroll
    for (int w = 0; w < 4; w++) {
        const uint32_t n = (w4[w] >> sh) & 0x0f0f0f0fu;
        const uint32_t p01 = __builtin_amdgcn_perm(0u, n, 0x0c010c00u) | 0x64006400u;   // fp16 1024 + q (bytes 0, 1)
        const uint32_t p23 = __builtin_amdgcn_perm(0u, n, 0x0c030c02u) | 0x64006400u;
        o[2 * w] = as_u(__builtin_elementwise_fma(as_h2(p01), s2, o2));                   // (1024 + q) sc - 1024 sc = q sc
        o[2 * w + 1] = as_u(__builtin_elementwise_fma(as_h2(p23), s2, o2));
    }
    return __builtin_bit_cast(h16_t, o);
}

// ---- Q4_K ------------------------------------------------------------------------------------------------------------
// the 12 scale bytes -> sc[8], m[8] (6 bits each)
__device__ __forceinline__ void q4k_scales(const u32x4_t & hd, int * sc, int * m) {
    const uint8_t * s = (const uint8_t *)&hd + 4;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        if (j < 4) { sc[j] = s[j] & 63; m[j] = s[j + 4] & 63; }
        else { sc[j] = (s[j + 4] & 0xf) | ((s[j - 4] >> 6) << 4); m[j] = (s[j + 4] >> 4) | ((s[j] >> 6) << 4); }
    }
}

extern "C" __global__ void __launch_bounds__(256) q3p_mm_q4k(
        const uint8_t * __restrict__ W, int R, int K, const _Float16 * __restrict__ X, int ldx, const int * __restrict__ xsrc,
        float * __restrict__ Y, int ldy, const int * __restrict__ ydst, const MMItem * __restrict__ items, const int * __restrict__ nitems,
        int ntok) {
    MMItem it;
    if (items) { if ((int)blockIdx.x >= *nitems) return; it = items[blockIdx.x]; }
    else { it.e = 0; it.off = blockIdx.x * MM_M; it.cnt = min(MM_M, ntok - it.off); }
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int r = blockIdx.y * MM_N + wave * 16 + (lane & 15);
    const bool rok = r < R;
    const int NB = K / 256;
    const size_t RB = (size_t)NB * 144;
    const uint8_t * wrow = W + ((size_t)it.e * R + (rok ? r : 0)) * RB;
    f8_t acc[4];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) acc[mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
    for (int b = 0; b < NB; b++) {
        // this lane's block: header (d, dmin, scales) and the 128 quant bytes, all in flight before the stage's barrier
        const u32x4_t * bp = (const u32x4_t *)(wrow + (size_t)b * 144);
        const u32x4_t hd = bp[0];
        u32x4_t qs[8];
#pragma unroll
        for (int i = 0; i < 8; i++) qs[i] = bp[1 + i];
        __syncthreads();
        mm_stage<true>(X, ldx, xsrc, it.off, it.cnt, b, tid);
        __syncthreads();
        int sc[8], m[8];
        q4k_scales(hd, sc, m);
        f8_t P[4];
#pragma unroll
        for (int mt = 0; mt < 4; mt++) P[mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
        // 16 k-steps: chunk j (sub-blocks 2 j low nibbles, 2 j + 1 high), half hf: quant bytes 32 j + 16 hf
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
#pragma unroll
                for (int hi = 0; hi < 2; hi++) {
                    const int s = 2 * j + hi;
                    const h16_t B = q4k_b(qs[2 * j + hf], 4 * hi, rok ? (_Float16)sc[s] : (_Float16)0.0f);
#pragma unroll
                    for (int mt = 0; mt < 4; mt++) P[mt] = wmma(mm_a(mt, s * 32 + hf * 16, lane), B, P[mt]);
                }
            }
        }
        // the min term: MS = sum_s m_s * S_s (S hi + lo against m twice)
        uint32_t bm[8];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const h2_t p = {(_Float16)m[2 * i], (_Float16)m[2 * i + 1]};
            bm[i] = as_u(p); bm[4 + i] = as_u(p);
        }
        const h16_t BM = __builtin_bit_cast(h16_t, bm);
        const uint16_t * hp = (const uint16_t *)&hd;
        const float d = rok ? (float)__builtin_bit_cast(_Float16, hp[0]) : 0.0f;
        const float dmin = rok ? (float)__builtin_bit_cast(_Float16, hp[1]) : 0.0f;
#pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            const u32x4_t * sp = (const u32x4_t *)(MMS + (mt * 16 + (lane & 15)) * 16);
            const h16_t AS = h16of(sp[0], sp[1]);
            const f8_t MS = wmma(AS, BM, (f8_t){0, 0, 0, 0, 0, 0, 0, 0});
#pragma unroll
            for (int v = 0; v < 8; v++) acc[mt][v] = __builtin_fmaf(d, P[mt][v], __builtin_fmaf(-dmin, MS[v], acc[mt][v]));
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

// ---- Q6_K (the decode's plane layout) -------------------------------------------------------------------------------------
// plane A: (row g, block b) at (g NB + b) 192: ql[128] | qh[64]; plane B at Rtot NB 192 + (g NB + b) 16: the 16 int8
// scales, position p = n 8 + 4 h + qd holding scale n 8 + h + 2 qd; plane C at Rtot NB 208 + (g NB + b) 2: d.
// Value v of a block (half n = v / 128, o = v % 128, l = o % 32, group gr = o / 32): ql[64 n + 32 (gr & 1) + l] (low
// nibble for gr < 2, high otherwise) | ((qh[32 n + l] >> 2 gr) & 3) << 4, minus 32; its scale is v / 16.
// 16 values of one sub-block (q - 32 exact) as one B operand: ql dwords (4), qh dwords (4), nibble shift, qh shift
__device__ __forceinline__ h16_t q6k_b(const u32x4_t & ql, const u32x4_t & qh, int shl, int shh) {
    const h2_t o2 = {(_Float16)(-1056.0f), (_Float16)(-1056.0f)};                  // 1024 + 32
    const uint32_t l4[4] = {ql.x, ql.y, ql.z, ql.w}, h4[4] = {qh.x, qh.y, qh.z, qh.w};
    uint32_t o[8];
#pragma unroll
    for (int w = 0; w < 4; w++) {
        const uint32_t q = ((l4[w] >> shl) & 0x0f0f0f0fu) | (((h4[w] >> shh) & 0x03030303u) << 4);
        const uint32_t p01 = __builtin_amdgcn_perm(0u, q, 0x0c010c00u) | 0x64006400u;
        const uint32_t p23 = __builtin_amdgcn_perm(0u, q, 0x0c030c02u) | 0x64006400u;
        o[2 * w] = as_u(as_h2(p01) + o2); o[2 * w + 1] = as_u(as_h2(p23) + o2);
    }
    return __builtin_bit_cast(h16_t, o);
}

extern "C" __global__ void __launch_bounds__(256) q3p_mm_q6k(
        const uint8_t * __restrict__ W, int R, int Rtot, int K, const _Float16 * __restrict__ X, int ldx, const int * __restrict__ xsrc,
        float * __restrict__ Y, int ldy, const int * __restrict__ ydst, const MMItem * __restrict__ items, const int * __restrict__ nitems,
        int ntok) {
    MMItem it;
    if (items) { if ((int)blockIdx.x >= *nitems) return; it = items[blockIdx.x]; }
    else { it.e = 0; it.off = blockIdx.x * MM_M; it.cnt = min(MM_M, ntok - it.off); }
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int r = blockIdx.y * MM_N + wave * 16 + (lane & 15);
    const bool rok = r < R;
    const int NB = K / 256;
    const size_t g = (size_t)it.e * R + (rok ? r : 0);
    const uint8_t * pa = W + g * NB * 192, * pb = W + (size_t)Rtot * NB * 192 + g * NB * 16;
    const uint16_t * pc = (const uint16_t *)(W + (size_t)Rtot * NB * 208) + g * NB;
    f8_t acc[4];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) acc[mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
    for (int b = 0; b < NB; b++) {
        const u32x4_t scw = *(const u32x4_t *)(pb + (size_t)b * 16);
        const float d = rok ? (float)__builtin_bit_cast(_Float16, pc[b]) : 0.0f;
        __syncthreads();
        mm_stage<false>(X, ldx, xsrc, it.off, it.cnt, b, tid);
        __syncthreads();
        const int8_t * scp = (const int8_t *)&scw;
#pragma unroll
        for (int n = 0; n < 2; n++) {
            const u32x4_t * q = (const u32x4_t *)(pa + (size_t)b * 192);
            const u32x4_t ql0 = q[4 * n + 0], ql1 = q[4 * n + 1], ql2 = q[4 * n + 2], ql3 = q[4 * n + 3];   // ql[64 n .. 64 n + 63]
            const u32x4_t qh0 = q[8 + 2 * n], qh1 = q[8 + 2 * n + 1];                                      // qh[32 n .. 32 n + 31]
#pragma unroll
            for (int gr = 0; gr < 4; gr++) {
#pragma unroll
                for (int l0 = 0; l0 < 2; l0++) {                      // l = 16 l0 .. 16 l0 + 15
                    const int sb = n * 8 + gr * 2 + l0;               // the sub-block (v / 16)
                    const u32x4_t & ql = (gr & 1) ? (l0 ? ql3 : ql2) : (l0 ? ql1 : ql0);
                    const u32x4_t & qh = l0 ? qh1 : qh0;
                    const h16_t B = q6k_b(ql, qh, gr >= 2 ? 4 : 0, 2 * gr);
                    const int nn = sb >> 3, rem = sb & 7, p = nn * 8 + 4 * (rem & 1) + (rem >> 1);
                    const float ds = d * (float)scp[p];
#pragma unroll
                    for (int mt = 0; mt < 4; mt++) {
                        const f8_t P = wmma(mm_a(mt, sb * 16, lane), B, (f8_t){0, 0, 0, 0, 0, 0, 0, 0});
#pragma unroll
                        for (int v = 0; v < 8; v++) acc[mt][v] = __builtin_fmaf(ds, P[v], acc[mt][v]);
                    }
                    // (one sub-block at a time: without the fence the scheduler hoisted the 16 sub-blocks' A loads and
                    //  decodes: 256 VGPRs, 25 spilled; with the half loop rolled 82)
                    __builtin_amdgcn_sched_barrier(0);
                }
            }
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

// ---- the glue ------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ float wsum(float v) {   // the decode's wave sums (xor 16, 8, 4, 2, 1)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
    return v;
}

// q3p_emb: x [n][N] fp32 = the Q4_K rows toks[i] of the embedding (raw GGUF blocks, rb bytes per row), clamped to the
// vocabulary; OP_OWN_EMB's arithmetic (no fma: the host decoder's bits). Grid n, block 256.
extern "C" __global__ void __launch_bounds__(256) q3p_emb(const uint8_t * __restrict__ W, int rb, int vocab, const int * __restrict__ toks,
                                                          float * __restrict__ x, int N) {
    const int tok = min(max(toks[blockIdx.x], 0), vocab - 1);
    const uint8_t * row = W + (size_t)tok * rb;
    float * xr = x + (size_t)blockIdx.x * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const uint8_t * blk = row + (size_t)(i >> 8) * 144;
        const int j = i & 255, sb = j >> 5, l = j & 31, is = sb >> 1, hi = sb & 1;
        const uint8_t * s = blk + 4;
        const uint32_t sc = sb < 4 ? (s[sb] & 63u) : ((s[sb + 4] & 0xfu) | ((s[sb - 4] >> 6) << 4));
        const uint32_t m = sb < 4 ? (s[sb + 4] & 63u) : ((s[sb + 4] >> 4) | ((s[sb] >> 6) << 4));
        const float d = (float)__builtin_bit_cast(_Float16, (uint16_t)(blk[0] | (blk[1] << 8)));
        const float dm = (float)__builtin_bit_cast(_Float16, (uint16_t)(blk[2] | (blk[3] << 8)));
        const float q = (float)((blk[16 + is * 32 + l] >> (4 * hi)) & 15);
        xr[i] = __fsub_rn(__fmul_rn(__fmul_rn(d, (float)sc), q), __fmul_rn(dm, (float)m));
    }
}

// q3p_add_rms16: per token (one workgroup of 256): OP_OWN_ADD_RMS16's statements in its order:
//   x[i] += sum over slots s < S of w[s] d[s][i] (fma in slot order from 0; S = 0: no add; w nullptr: + d)
//   h[i] = (x[i] rsqrt(sum x^2 / N + eps)) nw[i]; h16[i] = fp16(h[i]).  N = 256 NI, NI <= 8. Grid n.
extern "C" __global__ void __launch_bounds__(256) q3p_add_rms16(float * __restrict__ x, const float * __restrict__ d, const float * __restrict__ w,
                                                                int S, const float * __restrict__ nw, float eps, float * __restrict__ h,
                                                                _Float16 * __restrict__ h16, int N) {
    __shared__ float part[8];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, NI = N >> 8, t = blockIdx.x;
    float * xr = x + (size_t)t * N;
    const float * dr = d ? d + (size_t)t * S * N : nullptr;
    const float * wr = w ? w + (size_t)t * S : nullptr;
    float xv[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        if (j >= NI) break;
        float acc = 0.0f;
        for (int sl = 0; sl < S; sl++) acc = wr ? __builtin_fmaf(wr[sl], dr[(size_t)sl * N + tid + 256 * j], acc) : acc + dr[(size_t)sl * N + tid + 256 * j];
        xv[j] = xr[tid + 256 * j];
        if (S > 0) { xv[j] += acc; xr[tid + 256 * j] = xv[j]; }
    }
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; j++) if (j < NI) ss = __builtin_fmaf(xv[j], xv[j], ss);
    ss = wsum(ss); if (lane == 0) part[wave] = ss; __syncthreads();
    float tot = 0.0f; for (int q = 0; q < 8; q++) tot += part[q];
    const float scale = rsqrtf(tot / N + eps);
#pragma unroll
    for (int j = 0; j < 8; j++)
        if (j < NI) {
            const float v = (xv[j] * scale) * nw[tid + 256 * j];
            if (h) h[(size_t)t * N + tid + 256 * j] = v;
            h16[(size_t)t * N + tid + 256 * j] = (_Float16)v;
        }
}

// q3p_router: logits [n][E] = h [n][K] . Wr [E][K] (fp32; OP_F32_MV's lane order: lane k0 = 4 lane + 128 i, a float4 dot
// per step, the wave's xor sums). Grid n, block 256: wave w rows w, w + 8, ...
extern "C" __global__ void __launch_bounds__(256) q3p_router(const float * __restrict__ h, const float * __restrict__ Wr, int E, int K,
                                                             float * __restrict__ logits) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, t = blockIdx.x;
    const float * x = h + (size_t)t * K;
    for (int e = wave; e < E; e += 8) {
        const float * wr = Wr + (size_t)e * K;
        float acc = 0.0f;
        for (int k0 = lane * 4; k0 < K; k0 += 128) {
            const float4 wv = *(const float4 *)(wr + k0), xv = *(const float4 *)(x + k0);
            acc += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
        }
        acc = wsum(acc);
        if (lane == 0) logits[(size_t)t * E + e] = acc;
    }
}

// q3p_topk: OP_ROUTER's gating per token (softmax over the E logits, the top k by probability with lower-index-first ties,
// weights p / max(sum of the k, floor)); E <= 128: one wave per token, lane owns logits lane + 32 j. A logit that is not
// finite, or an id out of range, sets err (the chunk is refused). Grid n, block 32.
extern "C" __global__ void __launch_bounds__(32) q3p_topk(const float * __restrict__ logits, int E, int k, float floor_,
                                                          int * __restrict__ ids, float * __restrict__ wts, int * __restrict__ err) {
    const int lane = threadIdx.x, t = blockIdx.x;
    const float * lg = logits + (size_t)t * E;
    float v[4]; bool bad = false;
#pragma unroll
    for (int j = 0; j < 4; j++) { const int e = lane + 32 * j; v[j] = e < E ? lg[e] : -INFINITY; if (e < E && !__builtin_isfinite(v[j])) bad = true; }
    float m = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3])); m = wmax(m);
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; j++) if (lane + 32 * j < E) { v[j] = expf(v[j] - m); sum += v[j]; } else v[j] = 0.0f;
    sum = wsum(sum);
    const float inv = 1.0f / sum;
#pragma unroll
    for (int j = 0; j < 4; j++) v[j] = v[j] * inv;
    // rank of each of this lane's probabilities: #{others with p_j > p_i, or p_j == p_i and j < i}
    int rk[4] = {0, 0, 0, 0};
    for (int src = 0; src < 32; src++) {
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
            const float pj = __shfl(v[jj], src, 32); const int ej = src + 32 * jj;
#pragma unroll
            for (int j = 0; j < 4; j++) { const int ei = lane + 32 * j; if (ej < E && ei < E) rk[j] += (pj > v[j]) || (pj == v[j] && ej < ei); }
        }
    }
    __shared__ int sel[64];
#pragma unroll
    for (int j = 0; j < 4; j++) if (lane + 32 * j < E && rk[j] < k) sel[rk[j]] = lane + 32 * j;
    __syncthreads();
    // the k probabilities summed in slot order (lane j holds slot j)
    const int i = lane < k ? sel[lane] : 0;
    float pv[4];                                             // (lane i & 31's four probabilities; slot i >> 5 is p_i)
#pragma unroll
    for (int j = 0; j < 4; j++) pv[j] = __shfl(v[j], i & 31, 32);
    const float p_i = pv[(i >> 5) & 3];
    float ws = 0.0f;
    for (int j = 0; j < k; j++) ws = ws + __shfl(p_i, j, 32);
    ws = fmaxf(ws, floor_);
    if (lane < k) { ids[(size_t)t * k + lane] = i; wts[(size_t)t * k + lane] = p_i / ws; }
    if (bad) atomicOr(err, 1);   // (whichever lane holds the bad logit)
}

// q3p_group: the (token, slot) pairs of a chunk grouped by expert (one workgroup of 1024; E <= 1024): xs[i] = the token,
// yd[i] = the pair (token k + slot) for the pairs of expert e at [off_e, off_e + cnt_e); work items of <= 64 pairs per
// expert (the GEMMs' blockIdx.x), their count in *nitems. An id outside [0, E) sets err and counts as expert 0 (the chunk
// is refused by the host before its result is used). The order inside an expert's list follows the atomics, which changes
// no bit: each output row of the GEMMs depends on its own input row only.
extern "C" __global__ void __launch_bounds__(1024) q3p_group(const int * __restrict__ ids, int npairs, int k, int E, int * __restrict__ xs,
                                                             int * __restrict__ yd, MMItem * __restrict__ items, int * __restrict__ nitems,
                                                             int * __restrict__ err) {
    __shared__ int cnt[1024], off[1024], cur[1024], ioff[1024];
    const int tid = threadIdx.x;
    if (tid < E) { cnt[tid] = 0; cur[tid] = 0; }
    __syncthreads();
    for (int p = tid; p < npairs; p += 1024) {
        int e = ids[p];
        if (e < 0 || e >= E) { atomicOr(err, 2); e = 0; }
        atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    if (tid == 0) {
        int o = 0, io = 0;
        for (int e = 0; e < E; e++) { off[e] = o; ioff[e] = io; o += cnt[e]; io += (cnt[e] + MM_M - 1) / MM_M; }
        *nitems = io;
    }
    __syncthreads();
    for (int p = tid; p < npairs; p += 1024) {
        int e = ids[p]; if (e < 0 || e >= E) e = 0;
        const int sl = off[e] + atomicAdd(&cur[e], 1);
        xs[sl] = p / k; yd[sl] = p;
    }
    if (tid < E) {
        const int c = cnt[tid];
        for (int j = 0; j * MM_M < c; j++) { MMItem it; it.e = tid; it.off = off[tid] + j * MM_M; it.cnt = min(MM_M, c - j * MM_M); it.pad = 0; items[ioff[tid] + j] = it; }
    }
}

// q3p_swiglu: y16[i] = fp16(silu(g[i]) u[i]) (OP_OWN_SWIGLU16's expression), i < n
extern "C" __global__ void __launch_bounds__(256) q3p_swiglu(const float * __restrict__ g, const float * __restrict__ u, _Float16 * __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) { const float gv = g[i]; y[i] = (_Float16)(gv / (1.0f + expf(-gv)) * u[i]); }
}

// q3p_qk_rope: per token t (position pos0 + t) and head: q heads (blockIdx.y < nh): RMS norm with q_norm, NEOX RoPE over
// all 128 dims -> q16 [n][nh][128]; kv heads (blockIdx.y - nh = g): k the same with k_norm -> the cache row's K part
// (fp16), v -> its V part (fp16). OP_OWN_ATTS's statements (the four wave sums in order, rsqrtf, (scale x) w, theta =
// pos powf(theta_scale, j), fma(x0, c, -(x1 s)) / fma(x1, c, x0 s)). qkv: fp32 [n][ldq] = q | k | v. Grid (n, nh + nkv),
// block 128. par: q_norm [128] | k_norm [128] | theta_scale | eps (the decode's block). The host checks the positions
// against the cache's rows.
extern "C" __global__ void __launch_bounds__(128) q3p_qk_rope(const float * __restrict__ qkv, int ldq, int nh, int nkv, const float * __restrict__ par,
                                                              int pos0, _Float16 * __restrict__ q16, _Float16 * __restrict__ cache) {
    __shared__ float part[4], y1[128], rc[64], rs[64];
    const int i = threadIdx.x, lane = i & 31, wave = i >> 5, t = blockIdx.x, hy = blockIdx.y, pos = pos0 + t;
    const bool isq = hy < nh;
    const int g = hy - nh;
    const float * row = qkv + (size_t)t * ldq;
    const size_t M = (size_t)nkv * 128;
    _Float16 * cw = cache + (size_t)pos * 2 * M;
    if (!isq) {   // v: copied as is
        cw[M + (size_t)g * 128 + i] = (_Float16)row[nh * 128 + M + (size_t)g * 128 + i];
    }
    const float x = isq ? row[(size_t)hy * 128 + i] : row[nh * 128 + (size_t)g * 128 + i];
    const float wn = isq ? par[i] : par[128 + i], eps = par[257];
    if (i < 64) { const float theta = (float)pos * powf(par[256], (float)i); rc[i] = cosf(theta); rs[i] = sinf(theta); }
    float ss = x * x; ss = wsum(ss); if (lane == 0) part[wave] = ss;
    __syncthreads();
    float tot = 0.0f; for (int q = 0; q < 4; q++) tot += part[q];
    const float scale = rsqrtf(tot / 128 + eps);
    y1[i] = (scale * x) * wn;
    __syncthreads();
    const int j = i < 64 ? i : i - 64;
    const float c = rc[j], sn = rs[j], x0 = y1[j], x1 = y1[j + 64];
    const float y = i < 64 ? __builtin_fmaf(x0, c, -(x1 * sn)) : __builtin_fmaf(x1, c, x0 * sn);
    if (isq) q16[((size_t)t * nh + hy) * 128 + i] = (_Float16)y;
    else cw[(size_t)g * 128 + i] = (_Float16)y;
}

// ---- q3p_attn: causal attention of a chunk's queries over the cache (rows 0 .. p for the token at position p), WMMA, fp16
// q / k / v and p, fp32 scores, softmax state and sums.
// Work by ABSOLUTE position: workgroup i (8 waves) = positions 16 i .. 16 i + 15 x the 8 query heads of one kv head; wave
// w = the pair q0 = 16 i + 2 w, q1 = q0 + 1; its 16 query rows n = lane & 15 -> position q0 + (n >> 3), head 8 g + (n & 7).
// Positions outside the chunk [pos0, pos0 + n) are computed and never written.
// Transposed products so each lane holds ONE query row's state (no cross-lane softmax beyond lane ^ 16):
//   S^T (16 positions x 16 rows) = K (A: a position's 16 dims, LDS) x Q^T (B: the lane's row, 16 dims, registers)
//   O^T (16 dims x 16 rows) += V^T (A: a dim's 16 positions, LDS, staged transposed) x P^T (B: the lane's row, 16
//   positions: its own 8 (S^T's layout: positions 2 v + (lane >> 4)) and lane ^ 16's 8)
// Chunks of 64 positions from 0: K [64][AT_KP] and V^T [128][AT_VP] staged once for the 8 waves. Online softmax in base 2.
// Chunk geometry changes no bit (the prompt cache's stops; tests/prefill/test_q3m_prefill.py): RDNA3's WMMA is not exact
// for a zero product: an A value where B is 0 still moves the sum's rounding (measured: kernels' test, 600 of 5,120 D values
// with B ~1e-3 and A ~1e3 where B = 0), so a V row past a query must never sit in its column's A operand with a value
// that depends on how the prompt was chunked. Hence the absolute pairing (a pair is the same in every layout), the
// pair's diagonal tile with V^T zeroed past q0 (and q1's own entry zeroed in B), and q1's own row added with fp32 fmas
// (p(q1) v(q1)); positions past q1 are skipped. K needs none of this: S^T's rows (positions) are WMMA's M dimension,
// each row's result its own. Grid (16-position blocks, nkv), block 256; the host splits a long chunk into launches.
#define AT_C 64
#define AT_KP 136
#define AT_VP 72
extern "C" __global__ void __launch_bounds__(256) q3p_attn(const _Float16 * __restrict__ q16, const _Float16 * __restrict__ cache, int pos0, int n,
                                                           int blk0, int nkv, float scale_log2, _Float16 * __restrict__ out16) {
    __shared__ _Float16 Ks[AT_C * AT_KP];
    __shared__ _Float16 Vt[128 * AT_VP];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, hi = lane >> 4;
    const int g = blockIdx.y, b16 = (blk0 + (int)blockIdx.x) * 16, nh = nkv * 8;
    const size_t M = (size_t)nkv * 128, RS = 2 * M;   // halves per cache row
    const int nq = lane & 15, q0 = b16 + 2 * wave, q1 = q0 + 1, sec = nq >> 3;   // sec: this lane's row is q1's
    const int qpos = q0 + sec, tq = qpos - pos0, hq = g * 8 + (nq & 7);
    const bool qok = tq >= 0 && tq < n;
    h16_t Qb[8];
    {
        const u32x4_t * qp = (const u32x4_t *)(q16 + ((size_t)(qok ? tq : 0) * nh + hq) * 128);
#pragma unroll
        for (int s = 0; s < 8; s++) Qb[s] = h16of(qp[2 * s], qp[2 * s + 1]);
    }
    f8_t O[8];
#pragma unroll
    for (int d = 0; d < 8; d++) O[d] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.0f;
    const int npos = min(pos0 + n, b16 + 16);           // rows 0 .. npos - 1 exist for this workgroup (its last position's)
    const int pr = tid >> 2, qd = tid & 3;              // staging: position pr of the chunk, dims 32 qd .. 32 qd + 31
    for (int c0 = 0; c0 < npos; c0 += AT_C) {
        const int prow = min(c0 + pr, npos - 1);       // (rows past the end: a valid row; never in a written column's sum)
        const u32x4_t * kp = (const u32x4_t *)(cache + (size_t)prow * RS + (size_t)g * 128 + qd * 32);
        const u32x4_t * vp = (const u32x4_t *)(cache + (size_t)prow * RS + M + (size_t)g * 128 + qd * 32);
        u32x4_t kv[4], vv[4];
#pragma unroll
        for (int j = 0; j < 4; j++) { kv[j] = kp[j]; vv[j] = vp[j]; }
        __syncthreads();
        u32x4_t * kd = (u32x4_t *)(Ks + pr * AT_KP + qd * 32);
#pragma unroll
        for (int j = 0; j < 4; j++) kd[j] = kv[j];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const _Float16 * hv = (const _Float16 *)&vv[j];
#pragma unroll
            for (int e = 0; e < 8; e++) Vt[(qd * 32 + j * 8 + e) * AT_VP + pr] = hv[e];
        }
        __syncthreads();
        if (c0 > q1) continue;                          // (wave-uniform: every position past the pair)
        // the pair's tile: positions c0 + 16 std .. + 15 hold q0 and q1 (q0 even: never across a tile edge); 3 when the
        // chunk ends below q0. Tiles past it are all masked: skipped.
        const int std_ = min(3, (q0 - c0) >> 4), kq0 = q0 - c0 - 16 * std_;   // kq0: q0's index in its tile (even; >= 16: none)
        f8_t S[4];
#pragma unroll
        for (int st = 0; st < 4; st++) {
            S[st] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
            if (st > std_) continue;
            const u32x4_t * ap = (const u32x4_t *)(Ks + (st * 16 + (lane & 15)) * AT_KP);
#pragma unroll
            for (int s = 0; s < 8; s++) S[st] = wmma(h16of(ap[2 * s], ap[2 * s + 1]), Qb[s], S[st]);
        }
        float cm = -INFINITY;
#pragma unroll
        for (int st = 0; st < 4; st++)
#pragma unroll
            for (int v = 0; v < 8; v++) {
                const int pp = c0 + st * 16 + 2 * v + hi;
                const float x = (st <= std_ && pp <= qpos) ? S[st][v] * scale_log2 : -INFINITY;
                S[st][v] = x; cm = fmaxf(cm, x);
            }
        cm = fmaxf(cm, __shfl_xor(cm, 16, 32));
        const float mn = fmaxf(m, cm);                  // (position 0 is every row's: mn is finite)
        const float alpha = exp2f(m - mn);
        float rsum = 0.0f;
        uint32_t pw[4][4];                              // fp16 p, two positions per dword: [st][v pair]
#pragma unroll
        for (int st = 0; st < 4; st++)
#pragma unroll
            for (int v = 0; v < 8; v += 2) {
                const _Float16 p0 = (_Float16)exp2f(S[st][v] - mn), p1 = (_Float16)exp2f(S[st][v + 1] - mn);
                rsum += (float)p0 + (float)p1;
                pw[st][v >> 1] = as_u((h2_t){p0, p1});
            }
        rsum += __shfl_xor(rsum, 16, 32);
        l = l * alpha + rsum; m = mn;
#pragma unroll
        for (int d = 0; d < 8; d++)
#pragma unroll
            for (int v = 0; v < 8; v++) O[d][v] *= alpha;
#pragma unroll
        for (int st = 0; st < 4; st++) {
            if (st > std_) continue;
            // P^T operand: positions 16 st .. 16 st + 15 of the lane's row: own (2 v + hi) and lane ^ 16's (2 v + 1 - hi)
            uint32_t o[8];
#pragma unroll
            for (int w = 0; w < 4; w++) {
                const uint32_t mine = pw[st][w], other = (uint32_t)__shfl_xor((int)mine, 16, 32);
                // mine = positions 4w + hi, 4w + 2 + hi; other: 4w + 1 - hi, 4w + 3 - hi
                const uint32_t ev = hi ? other : mine, od = hi ? mine : other;   // even positions, odd positions
                o[2 * w] = __builtin_amdgcn_perm(od, ev, 0x05040100u);       // (ev.lo, od.lo): positions 4w, 4w + 1
                o[2 * w + 1] = __builtin_amdgcn_perm(od, ev, 0x07060302u);   // (ev.hi, od.hi): positions 4w + 2, 4w + 3
            }
            const bool diag = st == std_ && kq0 < 16;
            float p_q1 = 0.0f;
            if (diag) {                                 // q1's own entry (index kq0 + 1: dword kq0 / 2, high half) out of B
                const int wq = kq0 >> 1;
#pragma unroll
                for (int w = 0; w < 8; w++)
                    if (w == wq) { p_q1 = (float)as_h2(o[w]).y; if (sec) o[w] &= 0x0000ffffu; }
            }
            const h16_t PB = __builtin_bit_cast(h16_t, o);
#pragma unroll
            for (int d = 0; d < 8; d++) {
                const u32x4_t * ap = (const u32x4_t *)(Vt + (d * 16 + (lane & 15)) * AT_VP + st * 16);
                u32x4_t a0 = ap[0], a1 = ap[1];
                if (diag) {                             // V^T past q0 zeroed (kq0 even: dword kq0 / 2 keeps its low half)
                    uint32_t aw[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                    const int wq = kq0 >> 1;
#pragma unroll
                    for (int w = 0; w < 8; w++) aw[w] = w < wq ? aw[w] : w == wq ? (aw[w] & 0x0000ffffu) : 0u;
                    a0 = (u32x4_t){aw[0], aw[1], aw[2], aw[3]}; a1 = (u32x4_t){aw[4], aw[5], aw[6], aw[7]};
                }
                O[d] = wmma(h16of(a0, a1), PB, O[d]);
            }
            if (diag && sec) {                          // q1's rows: + p(q1) v(q1), fp32 fmas over the lane's dims
                const int kc = 16 * st + kq0 + 1;       // q1's row in the chunk
#pragma unroll
                for (int d = 0; d < 8; d++)
#pragma unroll
                    for (int v = 0; v < 8; v++) O[d][v] = __builtin_fmaf(p_q1, (float)Vt[(d * 16 + 2 * v + hi) * AT_VP + kc], O[d][v]);
            }
        }
    }
    if (!qok) return;
    const float il = 1.0f / l;
    _Float16 * op = out16 + ((size_t)tq * nh + hq) * 128;
#pragma unroll
    for (int d = 0; d < 8; d++)
#pragma unroll
        for (int v = 0; v < 8; v++) op[d * 16 + 2 * v + hi] = (_Float16)(O[d][v] * il);
}
