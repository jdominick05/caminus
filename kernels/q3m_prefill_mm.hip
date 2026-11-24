// The qwen3moe batched prefill's wide GEMM tiles (q3p_mm_q4k_w, the fused gate | up + swiglu q3p_mm_q4k_gu, q3p_mm_q6k_w;
// each row bitwise q3p_mm_q4k / q3p_mm_q6k's): a separate object, so q3m_prefill keeps its code and these build under
// LLVM's max-ilp scheduler (build.sh: 205 vs 216 ms of Q4_K GEMMs per 2048-token chunk, the same bits; under it
// q3m_prefill's own q3p_mm_q4k and attention spill, so they stay on the default). Q3P_WIDE_OBJECT: q3m_prefill.hip's
// helpers without its kernels. caminus.prefill.q3m_ops launches the three from this object. No ggml-cuda code.
#define Q3P_WIDE_OBJECT
#include "q3m_prefill.hip"

// ---- the wide tiles: two 16-row tiles per wave ---------------------------------------------------------------------------
// q3p_mm_q4k / q3p_mm_q6k with two row tiles per wave: each A operand (a token tile's 16 values, from LDS) feeds the WMMAs
// of both row tiles, so A is read once per two WMMAs and each token tile's X is staged once per 256 rows instead of per
// 128. Every row's statements are the narrow tile's in its order (the B operand from the row's own scale, P per block or
// sub-block in the same k-step order, the min term against the same sub-block sums, the same overflow branch), so every
// output keeps its bits (tests/prefill/test_q3m_ops.py and benchmarks/q35_prefill_gemm.py --engine --ref-module compare
// them bitwise). Written from the dense qwen35 line's q35p_mm_q4k_w / q35p_mm_q6k_w (kernels/q35_prefill.hip), here for
// dense matrices (items null) and the grouped experts (items, xsrc / ydst) as the narrow tiles, with the experts' gate |
// up fused, a partial item's empty token tiles skipped and the Q4_K weights one chunk ahead. The dense line keeps its own
// (this body on its objects measured 0.8 % slower on Swift's GEMMs under their default scheduler). Rows past R read
// row 0 and are never written.
#define MMW_N 256   // (rows per workgroup of the wide tiles: 8 waves x 32 rows)

// q4k_b with fp16's exponent byte (0x64: 1024 + q) taken by v_perm from its first source instead of OR-ed in after it:
// the same bits, two VALU instructions fewer per four values (the wide tile's limit is this unpack: with B = 1 it runs at
// ~104 TF on the XT, q35_prefill_abl1)
__device__ __forceinline__ h16_t q4k_bw(const u32x4_t & qw, int sh, _Float16 sc) {
    const h2_t s2 = {sc, sc}, o2 = {(_Float16)(-1024.0f) * sc, (_Float16)(-1024.0f) * sc};
    const uint32_t w4[4] = {qw.x, qw.y, qw.z, qw.w};
    uint32_t o[8];
#pragma unroll
    for (int w = 0; w < 4; w++) {
        const uint32_t n = (w4[w] >> sh) & 0x0f0f0f0fu;
        const uint32_t p01 = __builtin_amdgcn_perm(0x64646464u, n, 0x04010400u);   // bytes: q0, 0x64, q1, 0x64
        const uint32_t p23 = __builtin_amdgcn_perm(0x64646464u, n, 0x04030402u);
        o[2 * w] = as_u(__builtin_elementwise_fma(as_h2(p01), s2, o2));
        o[2 * w + 1] = as_u(__builtin_elementwise_fma(as_h2(p23), s2, o2));
    }
    return __builtin_bit_cast(h16_t, o);
}

// the wide Q4_K tile. GU false: row tile rt is rows 32 w + 16 rt of W (a workgroup 64 tokens x 256 rows), Y fp32 [.][ldy].
// GU true (the experts' gate and up in one launch): row tile 0 is rows 16 w of W (gate), row tile 1 the same rows of W1
// (up), a workgroup 64 tokens x 128 rows of each, and the output is Y16[dst][r] = fp16(silu(g) u), q3p_swiglu's
// expression on the same fp32 g and u: the bits the gate GEMM, the up GEMM and q3p_swiglu write, without g and u in
// memory (2 x 4 bytes per pair and row written and read back) and without the two launches
template <bool GU>
__device__ __forceinline__ void mm_q4k_w(const uint8_t * __restrict__ W, const uint8_t * __restrict__ W1, int R, int K,
                                         const _Float16 * __restrict__ X, int ldx, const int * __restrict__ xsrc,
                                         float * __restrict__ Y, _Float16 * __restrict__ Y16, int ldy, const int * __restrict__ ydst,
                                         const MMItem * __restrict__ items, const int * __restrict__ nitems, int ntok) {
    MMItem it;
    if (items) { if ((int)blockIdx.x >= *nitems) return; it = items[blockIdx.x]; }
    else { it.e = 0; it.off = blockIdx.x * MM_M; it.cnt = min(MM_M, ntok - it.off); }
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int NB = K / 256;
    const size_t RB = (size_t)NB * 144;
    // the token tiles holding the item's tokens (uniform): a partial item (an expert's last, or the chunk's last 64 tokens)
    // skips the WMMAs of its empty tiles, whose outputs are never written (each tile's sums are its own: the same bits)
    const int nmt = (it.cnt + 15) >> 4;
    int r[2]; bool rok[2]; const uint8_t * wrow[2];
#pragma unroll
    for (int rt = 0; rt < 2; rt++) {
        r[rt] = GU ? blockIdx.y * MM_N + wave * 16 + (lane & 15) : blockIdx.y * MMW_N + wave * 32 + rt * 16 + (lane & 15);
        rok[rt] = r[rt] < R;
        // (dense: the row's offset written as the dense line's tile had it, so its object keeps its code)
        wrow[rt] = (GU && rt ? W1 : W) + (items ? (size_t)it.e * R + (rok[rt] ? r[rt] : 0) : (size_t)(rok[rt] ? r[rt] : 0)) * RB;
    }
    f8_t acc[2][4];
#pragma unroll
    for (int rt = 0; rt < 2; rt++)
#pragma unroll
        for (int mt = 0; mt < 4; mt++) acc[rt][mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
    if (tid < MM_M) MMF[tid] = 0;
    if (tid == 0) MMBIG = 0;
    // the weights one chunk ahead: chunk j of a block (its sub-blocks 2 j, 2 j + 1) reads quant dwords 2 j and 2 j + 1 of
    // each row tile; a two-chunk ring holds the chunk being multiplied and the next one in flight (chunk j + 1, or the next
    // block's header and chunk 0 during chunk 3: across the scale step and the stage). 32 + 16 registers where the whole
    // block's quant bytes and header took 72, loaded right before the stage that hid them
    u32x4_t hd[2], hdn[2], qc[2][2][2];   // (qc [ring][row tile][half])
#pragma unroll
    for (int rt = 0; rt < 2; rt++) {
        const u32x4_t * bp = (const u32x4_t *)wrow[rt];
        hd[rt] = bp[0]; qc[0][rt][0] = bp[1]; qc[0][rt][1] = bp[2];
    }
    for (int b = 0; b < NB; b++) {
        __syncthreads();
        mm_stage<true>(X, ldx, xsrc, it.off, it.cnt, b, tid);
        __syncthreads();
        int sc[2][8], m[2][8];
#pragma unroll
        for (int rt = 0; rt < 2; rt++) q4k_scales(hd[rt], sc[rt], m[rt]);
        f8_t P[2][4];
#pragma unroll
        for (int rt = 0; rt < 2; rt++)
#pragma unroll
            for (int mt = 0; mt < 4; mt++) P[rt][mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            __builtin_amdgcn_sched_barrier(0);   // (the chunk's loads issued here, not hoisted: the ring's registers are the point)
            if (j < 3) {
#pragma unroll
                for (int rt = 0; rt < 2; rt++)
#pragma unroll
                    for (int hf = 0; hf < 2; hf++)
                        qc[(j + 1) & 1][rt][hf] = ((const u32x4_t *)(wrow[rt] + (size_t)b * 144))[1 + 2 * (j + 1) + hf];
            } else {   // (the last block reloads its own: no branch)
                const size_t bn = (size_t)min(b + 1, NB - 1) * 144;
#pragma unroll
                for (int rt = 0; rt < 2; rt++) {
                    const u32x4_t * bp = (const u32x4_t *)(wrow[rt] + bn);
                    hdn[rt] = bp[0]; qc[0][rt][0] = bp[1]; qc[0][rt][1] = bp[2];
                }
            }
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
#pragma unroll
                for (int hi = 0; hi < 2; hi++) {
                    const int s = 2 * j + hi;
#if Q3P_ABL == 1
                    const h16_t B0 = (h16_t)(_Float16)1.0f, B1 = B0; (void)qc; (void)sc;   // (measurement build: no weight unpack)
#else
                    const h16_t B0 = q4k_bw(qc[j & 1][0][hf], 4 * hi, rok[0] ? (_Float16)sc[0][s] : (_Float16)0.0f);
                    const h16_t B1 = q4k_bw(qc[j & 1][1][hf], 4 * hi, rok[1] ? (_Float16)sc[1][s] : (_Float16)0.0f);
#endif
#pragma unroll
                    for (int mt = 0; mt < 4; mt++) {
                        if (mt >= nmt) break;
                        const h16_t A = mm_a(mt, s * 32 + hf * 16, lane);
                        P[0][mt] = wmma(A, B0, P[0][mt]);
                        P[1][mt] = wmma(A, B1, P[1][mt]);
                    }
                }
            }
        }
#pragma unroll
        for (int rt = 0; rt < 2; rt++) {
            uint32_t bm[8];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const h2_t pm = {(_Float16)m[rt][2 * i], (_Float16)m[rt][2 * i + 1]};
                bm[i] = as_u(pm); bm[4 + i] = as_u(pm);
            }
            const h16_t BM = __builtin_bit_cast(h16_t, bm);
            const uint16_t * hp = (const uint16_t *)&hd[rt];
            const float d = rok[rt] ? (float)__builtin_bit_cast(_Float16, hp[0]) : 0.0f;
            const float dmin = rok[rt] ? (float)__builtin_bit_cast(_Float16, hp[1]) : 0.0f;
#pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                if (mt >= nmt) break;
                const u32x4_t * sp = (const u32x4_t *)(MMS + (mt * 16 + (lane & 15)) * 16);
                const h16_t AS = h16of(sp[0], sp[1]);
                const f8_t MS = wmma(AS, BM, (f8_t){0, 0, 0, 0, 0, 0, 0, 0});
#pragma unroll
                for (int v = 0; v < 8; v++) acc[rt][mt][v] = __builtin_fmaf(d, P[rt][mt][v], __builtin_fmaf(-dmin, MS[v], acc[rt][mt][v]));
            }
        }
        if (__builtin_amdgcn_readfirstlane(MMBIG) == b + 1) {   // rare: q3p_mm_q4k's other 63 / 64 of the scaled tokens' min term
#pragma unroll
            for (int rt = 0; rt < 2; rt++) {
                const u32x4_t hd2 = *(const u32x4_t *)(wrow[rt] + (size_t)b * 144);
                int sc2[8], m2[8];
                q4k_scales(hd2, sc2, m2);
                uint32_t bm2[8];
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const h2_t p2 = {(_Float16)m2[2 * i], (_Float16)m2[2 * i + 1]};
                    bm2[i] = as_u(p2); bm2[4 + i] = as_u(p2);
                }
                const float dm63 = rok[rt] ? -63.0f * (float)__builtin_bit_cast(_Float16, ((const uint16_t *)&hd2)[1]) : 0.0f;
#pragma unroll
                for (int mt = 0; mt < 4; mt++) {
                    const u32x4_t * sp = (const u32x4_t *)(MMS + (mt * 16 + (lane & 15)) * 16);
                    const f8_t MS = wmma(h16of(sp[0], sp[1]), __builtin_bit_cast(h16_t, bm2), (f8_t){0, 0, 0, 0, 0, 0, 0, 0});
#pragma unroll
                    for (int v = 0; v < 8; v++)
                        if (MMF[mt * 16 + 2 * v + (lane >> 4)] == b + 1) acc[rt][mt][v] = __builtin_fmaf(dm63, MS[v], acc[rt][mt][v]);
                }
            }
        }
#pragma unroll
        for (int rt = 0; rt < 2; rt++) hd[rt] = hdn[rt];
    }
    if (GU) {
        if (!rok[0]) return;
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int v = 0; v < 8; v++) {
                const int t = mt * 16 + 2 * v + (lane >> 4);
                if (t < it.cnt) {
                    const float gv = acc[0][mt][v];
                    Y16[(size_t)(ydst ? ydst[it.off + t] : it.off + t) * ldy + r[0]] = (_Float16)(gv / (1.0f + expf(-gv)) * acc[1][mt][v]);
                }
            }
        return;
    }
#pragma unroll
    for (int rt = 0; rt < 2; rt++) {
        if (!rok[rt]) continue;
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int v = 0; v < 8; v++) {
                const int t = mt * 16 + 2 * v + (lane >> 4);
                if (t < it.cnt) Y[(size_t)(ydst ? ydst[it.off + t] : it.off + t) * ldy + r[rt]] = acc[rt][mt][v];
            }
    }
}

// q4k_b's Q6_K form (q6k_b) with the exponent byte from v_perm (the same bits)
__device__ __forceinline__ h16_t q6k_bw(const u32x4_t & ql, const u32x4_t & qh, int shl, int shh) {
    const h2_t o2 = {(_Float16)(-1056.0f), (_Float16)(-1056.0f)};                  // 1024 + 32
    const uint32_t l4[4] = {ql.x, ql.y, ql.z, ql.w}, h4[4] = {qh.x, qh.y, qh.z, qh.w};
    uint32_t o[8];
#pragma unroll
    for (int w = 0; w < 4; w++) {
        const uint32_t q = ((l4[w] >> shl) & 0x0f0f0f0fu) | (((h4[w] >> shh) & 0x03030303u) << 4);
        const uint32_t p01 = __builtin_amdgcn_perm(0x64646464u, q, 0x04010400u);   // bytes: q0, 0x64, q1, 0x64
        const uint32_t p23 = __builtin_amdgcn_perm(0x64646464u, q, 0x04030402u);
        o[2 * w] = as_u(as_h2(p01) + o2); o[2 * w + 1] = as_u(as_h2(p23) + o2);
    }
    return __builtin_bit_cast(h16_t, o);
}

// the wide Q6_K tile (the decode's plane layout): row tile rt is rows 32 w + 16 rt; W a plane-layout buffer of Rtot rows
// (expert e's rows e R .. e R + R - 1)
__device__ __forceinline__ void mm_q6k_w(const uint8_t * __restrict__ W, int R, int Rtot, int K, const _Float16 * __restrict__ X,
                                         int ldx, const int * __restrict__ xsrc, float * __restrict__ Y, int ldy,
                                         const int * __restrict__ ydst, const MMItem * __restrict__ items,
                                         const int * __restrict__ nitems, int ntok) {
    MMItem it;
    if (items) { if ((int)blockIdx.x >= *nitems) return; it = items[blockIdx.x]; }
    else { it.e = 0; it.off = blockIdx.x * MM_M; it.cnt = min(MM_M, ntok - it.off); }
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int NB = K / 256;
    const int nmt = (it.cnt + 15) >> 4;   // (mm_q4k_w's skip of a partial item's empty token tiles)
    int r[2]; bool rok[2]; const uint8_t * pa[2], * pb[2]; const uint16_t * pc[2];
#pragma unroll
    for (int rt = 0; rt < 2; rt++) {
        r[rt] = blockIdx.y * MMW_N + wave * 32 + rt * 16 + (lane & 15);
        rok[rt] = r[rt] < R;
        const size_t g = items ? (size_t)it.e * R + (rok[rt] ? r[rt] : 0) : (size_t)(rok[rt] ? r[rt] : 0);
        pa[rt] = W + g * NB * 192; pb[rt] = W + (size_t)Rtot * NB * 192 + g * NB * 16;
        pc[rt] = (const uint16_t *)(W + (size_t)Rtot * NB * 208) + g * NB;
    }
    f8_t acc[2][4];
#pragma unroll
    for (int rt = 0; rt < 2; rt++)
#pragma unroll
        for (int mt = 0; mt < 4; mt++) acc[rt][mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
    for (int b = 0; b < NB; b++) {
        u32x4_t scw[2]; float d[2];
#pragma unroll
        for (int rt = 0; rt < 2; rt++) {
            scw[rt] = *(const u32x4_t *)(pb[rt] + (size_t)b * 16);
            d[rt] = rok[rt] ? (float)__builtin_bit_cast(_Float16, pc[rt][b]) : 0.0f;
        }
        __syncthreads();
        mm_stage<false>(X, ldx, xsrc, it.off, it.cnt, b, tid);
        __syncthreads();
#pragma unroll
        for (int n = 0; n < 2; n++) {
            u32x4_t ql[2][4], qh[2][2];
#pragma unroll
            for (int rt = 0; rt < 2; rt++) {
                const u32x4_t * q = (const u32x4_t *)(pa[rt] + (size_t)b * 192);
#pragma unroll
                for (int i = 0; i < 4; i++) ql[rt][i] = q[4 * n + i];
                qh[rt][0] = q[8 + 2 * n]; qh[rt][1] = q[8 + 2 * n + 1];
            }
#pragma unroll
            for (int gr = 0; gr < 4; gr++) {
#pragma unroll
                for (int l0 = 0; l0 < 2; l0++) {
                    const int sb = n * 8 + gr * 2 + l0;
                    const int qi = (gr & 1) ? (l0 ? 3 : 2) : (l0 ? 1 : 0);
                    const int nn = sb >> 3, rem = sb & 7, p = nn * 8 + 4 * (rem & 1) + (rem >> 1);
#if Q3P_ABL == 1
                    const h16_t B0 = (h16_t)(_Float16)1.0f, B1 = B0; (void)ql; (void)qh;
#else
                    const h16_t B0 = q6k_bw(ql[0][qi], qh[0][l0], gr >= 2 ? 4 : 0, 2 * gr);
                    const h16_t B1 = q6k_bw(ql[1][qi], qh[1][l0], gr >= 2 ? 4 : 0, 2 * gr);
#endif
                    const float ds0 = d[0] * (float)((const int8_t *)&scw[0])[p];
                    const float ds1 = d[1] * (float)((const int8_t *)&scw[1])[p];
#pragma unroll
                    for (int mt = 0; mt < 4; mt++) {
                        if (mt >= nmt) break;
                        const h16_t A = mm_a(mt, sb * 16, lane);
                        const f8_t P0 = wmma(A, B0, (f8_t){0, 0, 0, 0, 0, 0, 0, 0});
                        const f8_t P1 = wmma(A, B1, (f8_t){0, 0, 0, 0, 0, 0, 0, 0});
#pragma unroll
                        for (int v = 0; v < 8; v++) {
                            acc[0][mt][v] = __builtin_fmaf(ds0, P0[v], acc[0][mt][v]);
                            acc[1][mt][v] = __builtin_fmaf(ds1, P1[v], acc[1][mt][v]);
                        }
                    }
                    __builtin_amdgcn_sched_barrier(0);   // (one sub-block at a time, as q3p_mm_q6k)
                }
            }
        }
    }
#pragma unroll
    for (int rt = 0; rt < 2; rt++) {
        if (!rok[rt]) continue;
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int v = 0; v < 8; v++) {
                const int t = mt * 16 + 2 * v + (lane >> 4);
                if (t < it.cnt) Y[(size_t)(ydst ? ydst[it.off + t] : it.off + t) * ldy + r[rt]] = acc[rt][mt][v];
            }
    }
}

// ---- the kernels: q3p_mm_q4k / q3p_mm_q6k's interface; grid (work items or tokens / 64, rows / 256). q3p_mm_q4k_gu: W the
// gate tensor, W1 the up tensor (the same format, rows and experts), Y16 fp16 [.][ldy], grid (work items, rows / 128)
extern "C" __global__ void __launch_bounds__(256) q3p_mm_q4k_w(
        const uint8_t * __restrict__ W, int R, int K, const _Float16 * __restrict__ X, int ldx, const int * __restrict__ xsrc,
        float * __restrict__ Y, int ldy, const int * __restrict__ ydst, const MMItem * __restrict__ items, const int * __restrict__ nitems,
        int ntok) {
    mm_q4k_w<false>(W, W, R, K, X, ldx, xsrc, Y, nullptr, ldy, ydst, items, nitems, ntok);
}
extern "C" __global__ void __launch_bounds__(256) q3p_mm_q4k_gu(
        const uint8_t * __restrict__ W, const uint8_t * __restrict__ W1, int R, int K, const _Float16 * __restrict__ X, int ldx,
        const int * __restrict__ xsrc, _Float16 * __restrict__ Y16, int ldy, const int * __restrict__ ydst,
        const MMItem * __restrict__ items, const int * __restrict__ nitems) {
    mm_q4k_w<true>(W, W1, R, K, X, ldx, xsrc, nullptr, Y16, ldy, ydst, items, nitems, 0);
}
extern "C" __global__ void __launch_bounds__(256) q3p_mm_q6k_w(
        const uint8_t * __restrict__ W, int R, int Rtot, int K, const _Float16 * __restrict__ X, int ldx, const int * __restrict__ xsrc,
        float * __restrict__ Y, int ldy, const int * __restrict__ ydst, const MMItem * __restrict__ items, const int * __restrict__ nitems,
        int ntok) {
    mm_q6k_w(W, R, Rtot, K, X, ldx, xsrc, Y, ldy, ydst, items, nitems, ntok);
}
