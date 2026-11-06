// Batched prefill glue (docs/agentic/spec-prefill-cache.md M5 S3; host side python/caminus/prefill_ops.py): the per-token
// K3 decode ops of a layer, run over a whole chunk of T tokens with ordinary launches (not K3: no tape, no persistent
// kernel). Every kernel here computes each output element with the expressions of its K3 decode op
// (kernels/k3_gguf_ops.h) in the same order, so it is BITWISE that op on the same inputs (gate G1, tests/test_prefill_glue.py,
// against the K3 op itself run per token through caminus.k3.Engine):
//   pf_rms_mul      OP_RMS_MUL's n_embd fast path (rows of 2560: GLUE_NI = 10 values per thread, fma chain of the squares
//                   from a zero the compiler cannot fold, the DPP wave sum, the 8 wave partials summed in order, rsqrtf);
//                   outputs, each optional: the fp32 rows, their bf16 copy (round to nearest even, for a WMMA GEMM), the
//                   row scale (below)
//   pf_hc_post_rms  OP_HC_POST_RMS's fast path (residual update + the next mix's RMS norm of one (token, stream) row), the
//                   same outputs
//   pf_hc_post      OP_HC_POST (residual update alone: the last layer and the layer before the PLE layer)
//   pf_hc_mean      OP_HC_MEAN (the stream mix: fma chain over the 4 streams, then the 1 / hc scale)
//   pf_router       OP_ROUTER verbatim (softmax, top-k by rank on the probabilities' bit patterns: on equal probability
//                   the lower expert index first, weights summed in slot order, clamped, divided), plus the routing flag
//                   (a NaN / inf logit; ids out of range or repeated, replaced: mm_ids_helper needs them distinct, in range)
//   pf_gdn_conv     OP_GDN_CONV's per-token arithmetic (fma chain over [3 state / previous inputs, x], then silu); a
//                   chunk is parallel over (token, channel): token t reads inputs t - 3 .. t (or the committed state)
//   pf_conv_state   the conv state after the chunk: the last 3 inputs per channel (a second launch: pf_gdn_conv's
//                   first 3 tokens read the old state)
//   pf_ple_gate     OP_PLE_GATE (per token: the 4 stream dot products, contract off, then the gated value)
//   pf_ple_conv     OP_PLE_CONV's per-token arithmetic (taps at t - 9, t - 6, t - 3, t; contract off), optionally fused
//                   with OP_ADD_SUM's residual add (y = y + (a + conv), the decode's next op, same expressions)
//   pf_ple_hist     the PLE conv history after the chunk (the last 3 * dil inputs per channel)
//   pf_add_sum      OP_ADD_SUM
//   pf_bf16_mv_n*   OP_BF16_MV's split-K order (bf16_sk: virtual warp v = wave v, lane l takes the bf16 pairs
//                   c = 32 v + l + 256 i, i ascending, two fmas per pair from a zero; the DPP wave sum per virtual warp;
//                   the final wave sum over lanes v < 8, +0.0f elsewhere) for many tokens at once: each lane keeps its
//                   pairs of x for TT tokens in registers and walks the weight rows. K = 512 NI (2560: NI 5, 10240:
//                   NI 20). For the thin BF16 matrices (router logits, shared-expert gate, ssm alpha / beta, hc inject)
//   pf_silu_bf16    bf16(silu(fma(rs, x, 0))) (the hc up's activation, ignite's fused scale + silu, OP_BF16_MV work 1),
//                   rows padded with zeros: the A operand of the hc up WMMA GEMM
//   pf_to_bf16      bf16(x) with zero padding rows (the PLE key / value GEMMs' A operand)
// The normed rows are not stored in fp32 on the prefill's path: the norms write a row scale, and pf_hc_mean, pf_bf16_mv
// (the hc inject), pf_ple_gate (the query norm) and pf_ple_conv / pf_ple_hist (its input norm) recompute the normed value
// as (x * scale) * w, the norm's own expression on the same stored bits (xn_of below).
// The wide BF16 GEMMs (hc down / up, output_hc, ple key / value, the indexer projections) run on kernels/gemm.hip's bf16
// WMMA (build/gemm.hsaco, caminus.gemm.gemm_tn, fp32 output) on the bf16 activations these kernels write.
// Compiled by build.sh's generic loop like the other non-K3 objects; the decode code objects are untouched.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <limits.h>

typedef __attribute__((address_space(1))) const float GF;   // global (not flat) loads, as k3.hip
typedef __attribute__((address_space(1))) float GW;         // global stores, as k3_gguf_ops.h
typedef __attribute__((address_space(1))) const uint32_t GU32;
#define GLUE_NI 10                                           // n_embd 2560 = GLUE_NI x 256 threads

// ---- helpers, copied from kernels/k3_gguf_ops.h (same code, so the same instructions and bits) ----------------------
template <bool MAX>
__device__ __forceinline__ float wred(float x) {
    const unsigned u = __builtin_bit_cast(unsigned, x);
    float y = __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(u, u, 0x76543210u, 0xfedcba98u, false, false));
    x = MAX ? fmaxf(x, y) : x + y;
#pragma unroll
    for (int m = 8; m >= 1; m >>= 1) {   // row_xmask:m (DPP16 0x160 | m): lane ^ m within each row of 16
        const int xi = __builtin_bit_cast(int, x);
        int yi;
        switch (m) {
            case 8: yi = __builtin_amdgcn_update_dpp(xi, xi, 0x168, 0xf, 0xf, false); break;
            case 4: yi = __builtin_amdgcn_update_dpp(xi, xi, 0x164, 0xf, 0xf, false); break;
            case 2: yi = __builtin_amdgcn_update_dpp(xi, xi, 0x162, 0xf, 0xf, false); break;
            default: yi = __builtin_amdgcn_update_dpp(xi, xi, 0x161, 0xf, 0xf, false); break;
        }
        y = __builtin_bit_cast(float, yi);
        x = MAX ? fmaxf(x, y) : x + y;
    }
    return x;
}
__device__ __forceinline__ float wsum(float x) { return wred<false>(x); }
// an accumulator's 0.0f the compiler cannot see (k3_gguf_ops.h acc0: keeps the fma chain's first fma(a, b, 0))
__device__ __forceinline__ float acc0() { float z = 0.0f; asm volatile("" : "+v"(z)); return z; }
__device__ __forceinline__ float gg_silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ uint16_t to_bf16(float v) {     // round to nearest even (k3.hip / gemm.hip to_bf16)
    const uint32_t u = __builtin_bit_cast(uint32_t, v);
    return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}
__device__ __forceinline__ float bf16_lo(uint32_t w) { return __builtin_bit_cast(float, w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __builtin_bit_cast(float, w & 0xffff0000u); }

// The normed rows (xn) are not stored in fp32: a norm writes its per-row scale rsqrt(mean(x^2) + eps) (and, for the WMMA
// GEMMs, the bf16 copy of xn), and each fp32 consumer recomputes xn = (x * scale) * w with the norm's own expression on
// the same stored bits, so it gets the decode op's xn bits (measured on the XT: storing xn as well cost 0.39 ms of the
// 0.71 ms hc_post_rms, one more 84 MB write stream per hc mix at T = 2048; and 80 MiB of buffer per card)
__device__ __forceinline__ float xn_of(float x, float scale, float w) { return (x * scale) * w; }

// ---- pf_rms_mul: OP_RMS_MUL, rows of K = 2560 ---------------------------------------------------------------------
// y[r] = (x[r] * rsqrt(sum(x[r]^2) / K + eps)) * w[r % wr]; R rows, one workgroup per row; y (optional; in place, y == x,
// is safe: a thread loads its 10 values before any store, and only it touches them), ybf (optional): its bf16 copy, scl
// (optional): the row's scale [R]
extern "C" __global__ void __launch_bounds__(256) pf_rms_mul(const float * x, const float * w, float * y, uint16_t * ybf,
                                                             float * scl, int R, int wr, float eps) {
    __shared__ float part[8];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid % 32, wave = tid / 32, K = GLUE_NI * 256;
    if (t >= R) return;
    GF * xg = (GF *)x + (size_t)t * K + tid, * wg = (GF *)w + (size_t)(t % wr) * K + tid;
    float xv[GLUE_NI], wv[GLUE_NI];
#pragma unroll
    for (int j = 0; j < GLUE_NI; j++) xv[j] = xg[j * 256];
#pragma unroll
    for (int j = 0; j < GLUE_NI; j++) wv[j] = wg[j * 256];
    __builtin_amdgcn_sched_barrier(0);
    float ss = acc0();
#pragma unroll
    for (int j = 0; j < GLUE_NI; j++) ss = __builtin_fmaf(xv[j], xv[j], ss);
    ss = wsum(ss); if (lane == 0) part[wave] = ss; __syncthreads();
    float tot = 0.0f; for (int q = 0; q < 8; q++) tot += part[q];
    const float scale = rsqrtf(tot / K + eps);
    if (scl && tid == 0) scl[t] = scale;
#pragma unroll
    for (int j = 0; j < GLUE_NI; j++) xv[j] = xn_of(xv[j], scale, wv[j]);
    if (y) {
        GW * yw = (GW *)y + (size_t)t * K + tid;
#pragma unroll
        for (int j = 0; j < GLUE_NI; j++) yw[j * 256] = xv[j];
    }
    if (ybf) {
#pragma unroll
        for (int j = 0; j < GLUE_NI; j++) ybf[(size_t)t * K + tid + j * 256] = to_bf16(xv[j]);
    }
}

// ---- pf_hc_post_rms: OP_HC_POST_RMS, one workgroup per (token, stream) ---------------------------------------------
// res[t, h] = b[t] * 2 sigmoid(inj[t, h] * rs) + res[t, h] (in place), then xn[t, h] = (res * rsqrt(mean(res^2) + eps)) *
// w[h]: xn (optional) fp32, xnbf (optional) bf16, scl (optional) the scale [T * HC]. b [T, 2560], inj [T, HC],
// res / xn [T, HC, 2560], w [HC, 2560].
extern "C" __global__ void __launch_bounds__(256) pf_hc_post_rms(const float * b, const float * inj, float * res, const float * w,
                                                                 float * xn, uint16_t * xnbf, float * scl, int T, int HC, float rs, float eps) {
    __shared__ float part[8];
    const int q = blockIdx.x, tid = threadIdx.x, lane = tid % 32, wave = tid / 32, N = GLUE_NI * 256;
    if (q >= T * HC) return;
    const int h = q % HC, tt = q / HC;
    GF * bg = (GF *)b + (size_t)tt * N + tid, * rg = (GF *)res + (size_t)q * N + tid, * wg = (GF *)w + (size_t)h * N + tid;
    GW * rw = (GW *)res + (size_t)q * N + tid;
    float bv[GLUE_NI], nr[GLUE_NI], wv[GLUE_NI];
#pragma unroll
    for (int j = 0; j < GLUE_NI; j++) { bv[j] = bg[j * 256]; nr[j] = rg[j * 256]; }
#pragma unroll
    for (int j = 0; j < GLUE_NI; j++) wv[j] = wg[j * 256];
    float p;
    {
#pragma clang fp contract(off)
        const float v = inj[q] * rs; const float sgm = 1.0f / (1.0f + expf(-v)); p = sgm * 2.0f;
    }
    float ss = acc0();
#pragma unroll
    for (int j = 0; j < GLUE_NI; j++) {
        nr[j] = __builtin_fmaf(bv[j], p, nr[j]);   // the dsv4_hc_post kernel's contracted x * p + residual
        ss = __builtin_fmaf(nr[j], nr[j], ss);     // as op_rms_mul
    }
    ss = wsum(ss); if (lane == 0) part[wave] = ss; __syncthreads();
#pragma unroll
    for (int j = 0; j < GLUE_NI; j++) rw[j * 256] = nr[j];
    float tot = 0.0f; for (int k = 0; k < 8; k++) tot += part[k];
    const float scale = rsqrtf(tot / N + eps);
    if (scl && tid == 0) scl[q] = scale;
#pragma unroll
    for (int j = 0; j < GLUE_NI; j++) nr[j] = xn_of(nr[j], scale, wv[j]);
    if (xn) {
        GW * xw = (GW *)xn + (size_t)q * N + tid;
#pragma unroll
        for (int j = 0; j < GLUE_NI; j++) xw[j * 256] = nr[j];
    }
    if (xnbf) {
#pragma unroll
        for (int j = 0; j < GLUE_NI; j++) xnbf[(size_t)q * N + tid + j * 256] = to_bf16(nr[j]);
    }
}

// ---- pf_hc_post: OP_HC_POST, one thread per (token, element) ------------------------------------------------------
// res[t, h, i] = b[t, i] * 2 sigmoid(inj[t, h] * rs) + res[t, h, i] for every stream h; n = T * D elements
extern "C" __global__ void __launch_bounds__(256) pf_hc_post(const float * b, const float * inj, float * res, int T, int HC, int D,
                                                             float rs) {
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long)T * D) return;
    const int tt = (int)(i / D), ii = (int)(i - (long)tt * D);
    for (int h = 0; h < HC; h++) {
        float ph;
        {
#pragma clang fp contract(off)
            const float v = inj[tt * HC + h] * rs; const float sgm = 1.0f / (1.0f + expf(-v)); ph = sgm * 2.0f;
        }
        const size_t o = (size_t)(tt * HC + h) * D + ii;
        res[o] = __builtin_fmaf(b[i], ph, res[o]);
    }
}

// ---- pf_hc_mean: OP_HC_MEAN, one thread per (token, element) ------------------------------------------------------
// out[t, i] = rs * sum_h xn[t, h, i] * sigmoid(g[t, h, i]) (fma chain over h from 0); g [T, HC, D]. xn: given [T, HC, D]
// (scl null), or recomputed from the residual (x = res [T, HC, D], scl [T * HC], w [HC, D]: the norm's own expression)
extern "C" __global__ void __launch_bounds__(256) pf_hc_mean(const float * x, const float * scl, const float * w, const float * g,
                                                             float * out, int T, int HC, int D, float rs) {
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long)T * D) return;
    const int tt = (int)(i / D), ii = (int)(i - (long)tt * D);
    float sum = 0.0f;
    for (int h = 0; h < HC; h++) {
        const size_t o = (size_t)(tt * HC + h) * D + ii;
        const float xv = scl ? xn_of(x[o], scl[tt * HC + h], w[(size_t)h * D + ii]) : x[o];
        sum = __builtin_fmaf(xv, 1.0f / (1.0f + expf(-g[o])), sum);
    }
    out[i] = rs * sum;
}

// ---- pf_router: OP_ROUTER (kernels/k3_gguf_ops.h op_router, verbatim), one workgroup per token ----------------------
#define RK_C 16
__device__ __forceinline__ void router_rank_chunk(const int (&q)[RK_C], int c0, int b, int lane, int pi, int (&c)[2]) {
    if (c0 >= b && c0 < b + 32) {
        int ln = lane - (c0 - b); asm volatile("" : "+v"(ln));
#pragma unroll
        for (int u = 0; u < RK_C; u++) c[u % 2] += q[u] > (u < ln ? pi - 1 : pi);
    } else {
        const int t = c0 < b ? pi - 1 : pi;
#pragma unroll
        for (int u = 0; u < RK_C; u++) c[u % 2] += q[u] > t;
    }
}
template <int NE>
__device__ __forceinline__ void router_rank(const int * pb, int E32, int b0, int lane, int p0, int p1, int & r0, int & r1) {
    int c0s[2] = {0, 0}, c1s[2] = {0, 0};
#pragma unroll 1
    for (int c0 = 0; c0 < E32; c0 += RK_C) {
        int q[RK_C];
#pragma unroll
        for (int u = 0; u < RK_C; u += 4) { const int4 t = *(const int4 *)(pb + c0 + u); q[u] = t.x; q[u + 1] = t.y; q[u + 2] = t.z; q[u + 3] = t.w; }
        router_rank_chunk(q, c0, b0, lane, p0, c0s);
        if (NE == 2) router_rank_chunk(q, c0, b0 + 256, lane, p1, c1s);
    }
    r0 = c0s[0] + c0s[1]; r1 = c1s[0] + c1s[1];
}
// logits [T, E] (E <= 1024) -> ids [T, k] int32, wts [T, k] (k <= 32 on the fast path, <= 64), err: the sticky routing flag
// word (caminus.prefill_ops.check_routing; the host refuses the chunk when it is not 0):
//   1  a logit of the token is NaN or +-inf (the softmax is then NaN: the ids are still k distinct experts in range, see
//      below, but meaningless)
//   2  / 4  a selected id out of [0, E) / two slots holding the same id. Never seen: the rank is a strict total order on
//      (probability bits, index), so sel holds k distinct indices < E; checked anyway, because mm_ids_helper after this
//      corrupts its tables (and MMQ then writes out of bounds) on such ids: the token's ids become 0 .. k - 1 (k <= E)
// The checks touch no value the outputs are computed from, so valid tokens' ids / wts are the decode op's bits (G1)
extern "C" __global__ void __launch_bounds__(256) pf_router(const float * logits, int E, int k, float clamp, int * ids, float * wts,
                                                            int * err) {
#pragma clang fp contract(off)
    __shared__ float rv[8]; __shared__ int probs[1024]; __shared__ int sel[64];
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32, tt = blockIdx.x;
    const float * lg = logits + (size_t)tt * E;
    int * ids_out = ids + tt * k; float * wts_out = wts + tt * k;
    float v[4]; int idx[4];
#pragma unroll
    for (int jj = 0; jj < 4; jj++) { idx[jj] = tid + jj * 256; v[jj] = idx[jj] < E ? ((GF *)lg)[idx[jj]] : -INFINITY; }
    {
        bool nf = false;
#pragma unroll
        for (int jj = 0; jj < 4; jj++) nf |= idx[jj] < E && !__builtin_isfinite(v[jj]);
        if (nf) atomicOr(err, 1);
    }
    float m = -INFINITY;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) if (idx[jj] < E) m = fmaxf(m, v[jj]);
    m = wred<true>(m); if (lane == 0) rv[wave] = m; __syncthreads();
    m = rv[0]; for (int w = 1; w < 8; w++) m = fmaxf(m, rv[w]);
    float sum = 0.0f;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) if (idx[jj] < E) { v[jj] = expf(v[jj] - m); sum += v[jj]; }
    sum = wred<false>(sum); if (lane == 0) sel[wave] = __builtin_bit_cast(int, sum); __syncthreads();
    sum = __builtin_bit_cast(float, sel[0]); for (int w = 1; w < 8; w++) sum += __builtin_bit_cast(float, sel[w]);
    const float inv = 1.0f / sum;
    const int E32 = (E + 31) & ~31;
    int pi[4];
#pragma unroll
    for (int jj = 0; jj < 4; jj++) {
        v[jj] = v[jj] * inv; pi[jj] = __builtin_bit_cast(int, v[jj]);
        if (idx[jj] < E) probs[idx[jj]] = pi[jj]; else if (idx[jj] < E32) probs[idx[jj]] = INT_MIN;
    }
    __syncthreads();
    int cnt[4];
    const int nj = (E - wave * 32 + 255) / 256;
    if (nj >= 2) router_rank<2>(probs, E32, wave * 32, lane, pi[0], pi[1], cnt[0], cnt[1]);
    else if (nj == 1) router_rank<1>(probs, E32, wave * 32, lane, pi[0], pi[1], cnt[0], cnt[1]);
    if (nj >= 4) router_rank<2>(probs, E32, wave * 32 + 512, lane, pi[2], pi[3], cnt[2], cnt[3]);
    else if (nj == 3) router_rank<1>(probs, E32, wave * 32 + 512, lane, pi[2], pi[3], cnt[2], cnt[3]);
#pragma unroll
    for (int jj = 0; jj < 4; jj++) if (jj < nj && idx[jj] < E && cnt[jj] < k) sel[cnt[jj]] = idx[jj];
    __syncthreads();
    if (wave == 0) {
        float ws = 0.0f;
        if (k <= 32) {
            int i = lane < k ? sel[lane] : 0;
            bool br = lane < k && (i < 0 || i >= E), bd = false;
            for (int j = 0; j < k; j++) bd |= j < lane && lane < k && __builtin_amdgcn_readlane(i, j) == i;
            const int bad = (__builtin_amdgcn_ballot_w32(br) ? 2 : 0) | (__builtin_amdgcn_ballot_w32(bd) ? 4 : 0);
            if (bad) { i = lane < k ? lane : 0; if (lane == 0) atomicOr(err, bad); }
            const int pj = probs[i];
            for (int j = 0; j < k; j++) ws = ws + __builtin_bit_cast(float, __builtin_amdgcn_readlane(pj, j));
            ws = fmaxf(ws, clamp);
            if (lane < k) { ids_out[lane] = i; wts_out[lane] = __builtin_bit_cast(float, pj) / ws; }
        } else {
            bool br = false, bd = false;
            for (int j = lane; j < k; j += 32) { const int i = sel[j]; br |= i < 0 || i >= E; for (int q = 0; q < j; q++) bd |= sel[q] == i; }
            const int bad = (__builtin_amdgcn_ballot_w32(br) ? 2 : 0) | (__builtin_amdgcn_ballot_w32(bd) ? 4 : 0);
            if (bad && lane == 0) atomicOr(err, bad);
            for (int j = 0; j < k; j++) ws = ws + __builtin_bit_cast(float, probs[bad ? j : sel[j]]);
            ws = fmaxf(ws, clamp);
            for (int j = lane; j < k; j += 32) { const int i = bad ? j : sel[j]; ids_out[j] = i; wts_out[j] = __builtin_bit_cast(float, probs[i]) / ws; }
        }
    }
}

// ---- pf_gdn_conv / pf_conv_state: OP_GDN_CONV over a chunk --------------------------------------------------------
// y[t, c] = silu(fma chain over in = [x(t - 3), x(t - 2), x(t - 1), x(t)] . w[c]) where x(-1 .. -3) are the committed state
// st[c] = [x(-3), x(-2), x(-1)]; x / y [T, C] (row stride ldx / ldy floats), w [C, 4]. y must not alias x (another token
// reads x[t] after this one). The state is rewritten by pf_conv_state afterwards.
extern "C" __global__ void __launch_bounds__(256) pf_gdn_conv(const float * x, const float * w, const float * st, float * y,
                                                              int C, int T, int ldx, int ldy) {
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long)T * C) return;
    const int t = (int)(i / C), c = (int)(i - (long)t * C);
    const float4 w4 = *(const __attribute__((address_space(1))) float4 *)(w + (size_t)c * 4);
    float in[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int tj = t - 3 + j;                                      // the input feeding tap j
        in[j] = tj >= 0 ? ((GF *)x)[(size_t)tj * ldx + c] : ((GF *)st)[(size_t)c * 3 + 3 + tj];
    }
    const float wv[4] = {w4.x, w4.y, w4.z, w4.w};
    float sumf = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; j++) sumf = __builtin_fmaf(in[j], wv[j], sumf);
    ((GW *)y)[(size_t)t * ldy + c] = gg_silu(sumf);
}
// st[c] = the last 3 inputs of [st[c], x[0..T-1, c]] (one thread per channel)
extern "C" __global__ void __launch_bounds__(256) pf_conv_state(const float * x, float * st, int C, int T, int ldx) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    float s[3];
#pragma unroll
    for (int j = 0; j < 3; j++) { const int tj = T - 3 + j; s[j] = tj >= 0 ? x[(size_t)tj * ldx + c] : st[(size_t)c * 3 + 3 + tj]; }
#pragma unroll
    for (int j = 0; j < 3; j++) st[(size_t)c * 3 + j] = s[j];
}

// ---- pf_ple_gate: OP_PLE_GATE, one workgroup per token --------------------------------------------------------------
// s[st] = (sum_i kn[t, st, i] qn[t, st, i]) / sqrt(D); gate[t, st] = sigmoid(sgn(s) sqrt(max(|s|, 1e-6)));
// gated[t, st, i] = val[t, i] * gate[t, st]. kn / gated [T, HC, D], val [T, D] (row stride ldv), gate [T, HC]. qn: given
// [T, HC, D] (qscl null) or recomputed from the residual (q = res, qscl [T * HC], qw [HC, D]). gated may alias kn (this
// workgroup reads its token's rows before it writes them)
extern "C" __global__ void __launch_bounds__(256) pf_ple_gate(const float * kn, const float * q, const float * qscl, const float * qw,
                                                              const float * val, int ldv, float * gated, float * gate_out, int HC, int D) {
#pragma clang fp contract(off)
    __shared__ float part[8]; __shared__ float gate[8];
    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32, tt = blockIdx.x;
    const float * k_ = kn + (size_t)tt * HC * D, * q_ = q + (size_t)tt * HC * D, * v_ = val + (size_t)tt * ldv;
    for (int st = 0; st < HC; st++) {
        float acc = 0.0f;
        for (int i = tid; i < D; i += 256) {
            const float qv = qscl ? xn_of(q_[(size_t)st * D + i], qscl[tt * HC + st], qw[(size_t)st * D + i]) : q_[(size_t)st * D + i];
            acc += k_[(size_t)st * D + i] * qv;
        }
        acc = wsum(acc); if (lane == 0) part[wave] = acc; __syncthreads();
        if (tid == 0) {
            float sum = 0.0f; for (int w = 0; w < 8; w++) sum += part[w];
            const float s = sum * (1.0f / sqrtf((float)D));
            const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
            const float sgn = s > 0.0f ? 1.0f : (s < 0.0f ? -1.0f : 0.0f);
            const float v = sgn * mag;
            gate[st] = 1.0f / (1.0f + expf(-v));
            gate_out[tt * HC + st] = gate[st];
        }
        __syncthreads();
    }
    for (int c = tid; c < HC * D; c += 256) gated[(size_t)tt * HC * D + c] = v_[c % D] * gate[c / D];
}

// ---- pf_ple_conv / pf_ple_hist: OP_PLE_CONV over a chunk (+ OP_ADD_SUM fused) ---------------------------------------
// conv[t, c] = silu(((x(t - 3 dil) w0 + x(t - 2 dil) w1) + x(t - dil) w2) + x(t) w3) (separate roundings), x(< 0) from the
// history hist[c] = [x(-3 dil) .. x(-1)] (oldest first); w f16 [C, 4]. x: given [T, C] (xscl null) or the normed rows
// recomputed (x = the pre-norm rows [T, C = HC * D], xscl [T * HC], xw [HC * D]). y (optional): conv out [T, C]; res
// (optional): res[t, c] = res[t, c] + (a[t, c] + conv) (OP_ADD_SUM: the decode's next op, add_sum(gated, conv, res)).
// Neither y nor res may alias x. The history (of the normed values) is rewritten by pf_ple_hist afterwards.
extern "C" __global__ void __launch_bounds__(256) pf_ple_conv(const float * x, const float * xscl, const float * xw, const __half * w,
                                                              const float * hist, float * y, const float * a, float * res, int C, int T,
                                                              int D, int dil) {
#pragma clang fp contract(off)
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long)T * C) return;
    const int t = (int)(i / C), c = (int)(i - (long)t * C), hl = 3 * dil;
    float conv = 0.0f;
    for (int k = 0; k < 4; k++) {
        const int back = (3 - k) * dil;
        float xv;
        if (back <= t) {
            const int tb = t - back;
            xv = xscl ? xn_of(x[(size_t)tb * C + c], xscl[tb * (C / D) + c / D], xw[c]) : x[(size_t)tb * C + c];
        } else xv = hist[(size_t)c * hl + hl - (back - t)];
        const float term = xv * __half2float(w[(size_t)c * 4 + k]);
        conv = k == 0 ? term : conv + term;
    }
    const float o = gg_silu(conv);
    if (y) y[i] = o;
    if (res) { const float ab = a[i] + o; res[i] = res[i] + ab; }
}
// hist[c] = the last 3 dil (normed) inputs of [hist[c], x[0..T-1, c]] (one thread per channel; x as pf_ple_conv's)
extern "C" __global__ void __launch_bounds__(256) pf_ple_hist(const float * x, const float * xscl, const float * xw, float * hist, int C,
                                                              int T, int D, int dil) {
    const int c = blockIdx.x * 256 + threadIdx.x, hl = 3 * dil;
    if (c >= C || hl > 32) return;
    float h[32];
    for (int j = 0; j < hl; j++) {
        const int tj = T - hl + j;
        h[j] = tj < 0 ? hist[(size_t)c * hl + hl + tj] : xscl ? xn_of(x[(size_t)tj * C + c], xscl[tj * (C / D) + c / D], xw[c]) : x[(size_t)tj * C + c];
    }
    for (int j = 0; j < hl; j++) hist[(size_t)c * hl + j] = h[j];
}

// ---- pf_add_sum: OP_ADD_SUM -------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) pf_add_sum(const float * a, const float * b, float * y, long n) {
#pragma clang fp contract(off)
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const float ab = a[i] + b[i];
    y[i] = y[i] + ab;
}

// ---- pf_bf16_mv: OP_BF16_MV's order (bf16_sk), many tokens per launch -------------------------------------------------
// out[t, r] = W[r] . x[t]: W bf16 [N, K] (raw pairs), x fp32 [T, K] (row stride K), out fp32 [T, N] (row stride ldo).
// Grid (ceil(T / TT), ceil(N / rpb)); a workgroup = the 8 virtual warps (wave v) x TT tokens, walking its rpb rows R at a
// time. Each (row, token) is computed exactly as bf16_sk computes it (any R, TT, rpb or grid: the order is per element).
template <int NI, int TT, int R>
__device__ __forceinline__ void bf16_mv_body(const uint32_t * W2, const float * X, const float * xscl, const float * xw, int xd,
                                             float * out, int N, int T, int ldo, int rpb) {
    __shared__ float BP[R * TT * 8];
    const int v = threadIdx.x / 32, lane = threadIdx.x % 32;
    constexpr int K = 512 * NI, n2 = K / 2;
    const int t0 = blockIdx.x * TT, nt = min(TT, T - t0);
    const int r0 = blockIdx.y * rpb, r1 = min(N, r0 + rpb);
    if (nt <= 0 || r0 >= r1) return;                                   // (workgroup-uniform)
    const int c0 = 32 * v + lane;
    float2 xv[TT][NI];
#pragma unroll
    for (int j = 0; j < TT; j++)
#pragma unroll
        for (int i = 0; i < NI; i++) {
            const int tj = t0 + min(j, nt - 1), k = 2 * (c0 + 256 * i);
            xv[j][i] = *(const __attribute__((address_space(1))) float2 *)(X + (size_t)tj * K + k);
            if (xscl) {   // x = the normed row recomputed (X = the residual, rows of xd within the K; a pair never straddles two)
                const float s = xscl[tj * (K / xd) + k / xd];
                xv[j][i] = make_float2(xn_of(xv[j][i].x, s, xw[k]), xn_of(xv[j][i].y, s, xw[k + 1]));
            }
        }
    for (int rb = r0; rb < r1; rb += R) {
        uint32_t wv[R][NI];
#pragma unroll
        for (int r = 0; r < R; r++) {
            GU32 * wr = (GU32 *)W2 + (size_t)min(rb + r, r1 - 1) * n2 + c0;   // past r1: re-read, discarded
#pragma unroll
            for (int i = 0; i < NI; i++) wv[r][i] = wr[256 * i];
        }
        float acc[R][TT];
#pragma unroll
        for (int r = 0; r < R; r++)
#pragma unroll
            for (int j = 0; j < TT; j++) acc[r][j] = acc0();
#pragma unroll
        for (int i = 0; i < NI; i++)
#pragma unroll
            for (int r = 0; r < R; r++)
#pragma unroll
                for (int j = 0; j < TT; j++) {
                    acc[r][j] = __builtin_fmaf(bf16_lo(wv[r][i]), xv[j][i].x, acc[r][j]);
                    acc[r][j] = __builtin_fmaf(bf16_hi(wv[r][i]), xv[j][i].y, acc[r][j]);
                }
#pragma unroll
        for (int r = 0; r < R; r++)
#pragma unroll
            for (int j = 0; j < TT; j++) { const float s = wsum(acc[r][j]); if (lane == 0) BP[(r * TT + j) * 8 + v] = s; }
        __syncthreads();
        for (int q = v; q < R * TT; q += 8) {
            const int r = q / TT, j = q % TT;
            const float f = wsum(lane < 8 ? BP[q * 8 + lane] : 0.0f);
            if (lane == 0 && rb + r < r1 && j < nt) out[(size_t)(t0 + j) * ldo + rb + r] = f;
        }
        __syncthreads();
    }
}
#define PF_BF16_MV(NAME, NI, TT, R)                                                                                    \
extern "C" __global__ void __launch_bounds__(256) NAME(const uint32_t * W2, const float * X, const float * xscl, const float * xw, \
        int xd, float * out, int N, int T, int ldo, int rpb) { bf16_mv_body<NI, TT, R>(W2, X, xscl, xw, xd, out, N, T, ldo, rpb); }
PF_BF16_MV(pf_bf16_mv_n5, 5, 8, 4)      // K 2560: x for 8 tokens in 80 VGPRs, 4 rows per round
PF_BF16_MV(pf_bf16_mv_n20, 20, 2, 2)    // K 10240: x for 2 tokens in 80 VGPRs, 2 rows per round

// ---- pf_silu_bf16 / pf_to_bf16: the WMMA GEMMs' bf16 A operands ----------------------------------------------------
// y[t, j] = bf16(silu(fma(rs, x[t, j], 0))) for t < T (x row stride ldx), 0 for T <= t < Tpad; y [Tpad, K]
extern "C" __global__ void __launch_bounds__(256) pf_silu_bf16(const float * x, int ldx, uint16_t * y, int T, int Tpad, int K, float rs) {
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long)Tpad * K) return;
    const int t = (int)(i / K), j = (int)(i - (long)t * K);
    y[i] = t < T ? to_bf16(gg_silu(__builtin_fmaf(rs, x[(size_t)t * ldx + j], acc0()))) : (uint16_t)0;
}
// y[t, j] = bf16(x[t, j]) for t < T, 0 for T <= t < Tpad
extern "C" __global__ void __launch_bounds__(256) pf_to_bf16(const float * x, int ldx, uint16_t * y, int T, int Tpad, int K) {
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long)Tpad * K) return;
    const int t = (int)(i / K), j = (int)(i - (long)t * K);
    y[i] = t < T ? to_bf16(x[(size_t)t * ldx + j]) : (uint16_t)0;
}
