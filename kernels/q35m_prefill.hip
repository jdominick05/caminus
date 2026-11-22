// The batched prefill of the qwen35moe line (Qwen3.6-35B-A3B) on Caminus's own kernels (python/caminus/prefill/q35moe.py):
// no ggml-cuda code. This object holds q35_prefill.hip's kernels (the dense qwen35 line's GEMMs, GDN glue and attention
// tiles, and q3m_prefill.hip's under them: included as they are, so neither file's own object changes) and the ops the MoE
// line adds:
//
//   q35mp_mm_q2_0    the routed experts' Q2_0 GEMM (grouped by expert, q3p_mm_q4k's interface and tiling)
//   q35mp_router_sh  the router's softmax top-k plus the gated shared expert's weight (the decode's OP_ROUTER on the
//                    qwen35moe module, kernels/k3_q35_ops.h op_router_shexp), one token per workgroup
//   q3p_group        built here with Q3P_GROUP_Y9: a third index, each (token, slot) pair's row in the 9-slot down buffer
//   q35mp_emb_bf16   the BF16 embedding rows (kernels/k3_q35_ops.h own_op_emb_bf16's statements)
//
// The feed-forward's combine is q3p_add_rms16 over 9 slots: the 8 routed experts' down rows, then the shared expert's
// in slot 8, weighted by q35mp_router_sh's [n][9] weights (the decode's own_op_add_rms16 9-slot form: fma in slot order).
// q3p_add_rms16 writes h16 as the qwen35moe decode module's mixed fma does (one rounding of the exact product): only in this
// object, so q3m_prefill's and q35_prefill's code stays unchanged; likewise q3p_group's third index (Q3P_GROUP_Y9)
#define Q3P_ADD_RMS16_MIX
#define Q3P_GROUP_Y9
#include "q35_prefill.hip"

// ---- q35mp_mm_q2_0: Y[ydst[i]][r] = X[xsrc[i]] . W[e][r] in fp32 ------------------------------------------------------
// Q2_0 (kernels/own_quant_q2_0.h): 64 values per 18-byte block, d (fp16) then 16 code bytes; value j is bits 2 (j % 4) of
// byte j / 4, w = d (q - 1). As little-endian dwords, code dword c of a block holds values 16 c .. 16 c + 15, value
// 16 c + j at bits 2 j. A weight is d times an integer in {-1, 0, 1, 2}: the B operand is that integer as fp16 (exact),
// each block's 4 WMMA k-steps start from a zero accumulator and the block ends with acc = fma(d, P, acc) (q35p_mm_q8_0's
// per-block form), so the weights are exact and every product exact; the sums are fp32 in WMMA order.
// A staging step is 256 values = 4 blocks = 72 bytes of a row. Rows are K / 64 * 18 bytes (K % 256 == 0: a multiple of
// 72 * 4 / 4), so every row and every step starts 4-byte aligned: 18 dword loads, the next step's issued before this
// step's WMMAs. In place: the GGUF's [E][R][K] blocks as they are (expert e's row r at (e R + r) * row bytes).
// Grid (work items or tokens / 64, rows / 128), block 256; X fp16 [ntok][ldx], Y fp32 [.][ldy]; items: q3p_group's /
// q3p_group's (null: dense rows of expert 0).
__device__ __forceinline__ h16_t q2_0_b(uint32_t v) {
    const h2_t o2 = {(_Float16)(-1025.0f), (_Float16)(-1025.0f)};   // fp16 (1024 + q) - 1025 = q - 1, exact
    uint32_t o[8];
#pragma unroll
    for (int m = 0; m < 8; m++) {   // values 2 m, 2 m + 1: bits 4 m and 4 m + 2 of v
        const uint32_t t = v >> (4 * m);
        o[m] = as_u(as_h2((t & 3u) | ((t & 0xcu) << 14) | 0x64006400u) + o2);
    }
    return __builtin_bit_cast(h16_t, o);
}
struct Q2Step { uint32_t w[18]; };
__device__ __forceinline__ void q2_step_load(Q2Step & S, const uint8_t * wrow, int b) {
    const uint32_t * src = (const uint32_t *)(wrow + (size_t)b * 72);
#pragma unroll
    for (int i = 0; i < 18; i++) S.w[i] = src[i];
}
// block I (0 .. 3) of the step: bytes 18 I .. 18 I + 17. Even I: d in the low half of dword 9 I / 2, the codes two bytes
// into it (each code dword straddles two loaded dwords); odd I: d in the high half of dword (9 I - 1) / 2, the codes in the
// four dwords after it
template <int I>
__device__ __forceinline__ void q2_block(const Q2Step & S, bool rok, int lane, f8_t (&acc)[4]) {
    constexpr int DI = (I & 1) ? (9 * I - 1) / 2 : 9 * I / 2;
    const uint16_t dh = (I & 1) ? (uint16_t)(S.w[DI] >> 16) : (uint16_t)(S.w[DI] & 0xffffu);
    const float d = rok ? (float)__builtin_bit_cast(_Float16, dh) : 0.0f;
    h16_t B[4];
#pragma unroll
    for (int c = 0; c < 4; c++)
        B[c] = q2_0_b((I & 1) ? S.w[DI + 1 + c] : ((S.w[DI + c] >> 16) | (S.w[DI + c + 1] << 16)));
#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
        f8_t P = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int c = 0; c < 4; c++) P = wmma(mm_a(mt, 64 * I + 16 * c, lane), B[c], P);
#pragma unroll
        for (int v = 0; v < 8; v++) acc[mt][v] = __builtin_fmaf(d, P[v], acc[mt][v]);
    }
}
extern "C" __global__ void __launch_bounds__(256) q35mp_mm_q2_0(
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
    const size_t RB = (size_t)(K / 64) * 18;
    const uint8_t * wrow = W + ((size_t)it.e * R + (rok ? r : 0)) * RB;
    f8_t acc[4];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) acc[mt] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
    Q2Step cur, nxt;
    q2_step_load(cur, wrow, 0);
    for (int b = 0; b < NB; b++) {
        __syncthreads();
        mm_stage<false>(X, ldx, xsrc, it.off, it.cnt, b, tid);
        __syncthreads();
        if (b + 1 < NB) q2_step_load(nxt, wrow, b + 1);     // (the next step in flight during this step's WMMAs)
        q2_block<0>(cur, rok, lane, acc);
        __builtin_amdgcn_sched_barrier(0);
        q2_block<1>(cur, rok, lane, acc);
        __builtin_amdgcn_sched_barrier(0);
        q2_block<2>(cur, rok, lane, acc);
        __builtin_amdgcn_sched_barrier(0);
        q2_block<3>(cur, rok, lane, acc);
        __builtin_amdgcn_sched_barrier(0);
        cur = nxt;
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

// ---- q35mp_router_sh: the decode's OP_ROUTER on the qwen35moe module, per token ------------------------------------------
// op_router's statements (kernels/k3_q3m_ops.h: softmax, top k by rank on the probabilities' bit patterns, the k weights
// over their sum clamped at floor_) with its helpers below copied verbatim, then op_router_shexp's gated shared weight
// 1 / (1 + exp(-sgl)) as slot k. The wave sums are q3m_prefill.hip's wsum / wmax: the decode's partners (xor 16 .. 1) in
// its order, so the same bits. logits [n][E] (E <= 1024), sgl [n], ids [n][k] (k <= 32), wts [n][k + 1]; a logit that is
// not finite sets err bit 1 (the chunk is refused), as q3p_topk does. Grid n, block 256.
#define Q35MP_RK_C 16
__device__ __forceinline__ void q35mp_rank_chunk(const int (&q)[Q35MP_RK_C], int c0, int b, int lane, int pi, int (&c)[2]) {
    if (c0 >= b && c0 < b + 32) {   // (opaque lane: the per-u thresholds must not be hoisted out of the chunk loop)
        int ln = lane - (c0 - b); asm volatile("" : "+v"(ln));
#pragma unroll
        for (int u = 0; u < Q35MP_RK_C; u++) c[u % 2] += q[u] > (u < ln ? pi - 1 : pi);
    } else {
        const int t = c0 < b ? pi - 1 : pi;
#pragma unroll
        for (int u = 0; u < Q35MP_RK_C; u++) c[u % 2] += q[u] > t;
    }
}
template <int NE>
__device__ __forceinline__ void q35mp_rank(const int * pb, int E32, int b0, int lane, int p0, int p1, int & r0, int & r1) {
    int c0s[2] = {0, 0}, c1s[2] = {0, 0};
#pragma unroll 1
    for (int c0 = 0; c0 < E32; c0 += Q35MP_RK_C) {
        int q[Q35MP_RK_C];
#pragma unroll
        for (int u = 0; u < Q35MP_RK_C; u += 4) { const int4 t = *(const int4 *)(pb + c0 + u); q[u] = t.x; q[u + 1] = t.y; q[u + 2] = t.z; q[u + 3] = t.w; }
        q35mp_rank_chunk(q, c0, b0, lane, p0, c0s);
        if (NE == 2) q35mp_rank_chunk(q, c0, b0 + 256, lane, p1, c1s);
    }
    r0 = c0s[0] + c0s[1]; r1 = c1s[0] + c1s[1];
}
extern "C" __global__ void __launch_bounds__(256) q35mp_router_sh(const float * __restrict__ logits, const float * __restrict__ sgl,
                                                                  int E, int k, float floor_, int * __restrict__ ids,
                                                                  float * __restrict__ wts, int * __restrict__ err) {
#pragma clang fp contract(off)
    __shared__ float rv[8]; __shared__ int probs[1024]; __shared__ int sel[64];   // sel[0..7]: the sum partials first
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid % 32, t = blockIdx.x;
    const float * lg = logits + (size_t)t * E;
    int * ids_out = ids + (size_t)t * k; float * wts_out = wts + (size_t)t * (k + 1);
    float v[4]; int idx[4]; bool bad = false;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) {
        idx[jj] = tid + jj * 256; v[jj] = idx[jj] < E ? lg[idx[jj]] : -INFINITY;
        if (idx[jj] < E && !__builtin_isfinite(v[jj])) bad = true;
    }
    float m = -INFINITY;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) if (idx[jj] < E) m = fmaxf(m, v[jj]);
    m = wmax(m); if (lane == 0) rv[wave] = m; __syncthreads();
    m = rv[0]; for (int w = 1; w < 8; w++) m = fmaxf(m, rv[w]);
    float sum = 0.0f;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) if (idx[jj] < E) { v[jj] = expf(v[jj] - m); sum += v[jj]; }
    sum = wsum(sum); if (lane == 0) sel[wave] = __builtin_bit_cast(int, sum); __syncthreads();
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
    const int nj = (E - wave * 32 + 255) / 256;   // wave-uniform: how many of this wave's jj slots hold a logit
    if (nj >= 2) q35mp_rank<2>(probs, E32, wave * 32, lane, pi[0], pi[1], cnt[0], cnt[1]);
    else if (nj == 1) q35mp_rank<1>(probs, E32, wave * 32, lane, pi[0], pi[1], cnt[0], cnt[1]);
    if (nj >= 4) q35mp_rank<2>(probs, E32, wave * 32 + 512, lane, pi[2], pi[3], cnt[2], cnt[3]);
    else if (nj == 3) q35mp_rank<1>(probs, E32, wave * 32 + 512, lane, pi[2], pi[3], cnt[2], cnt[3]);
#pragma unroll
    for (int jj = 0; jj < 4; jj++) if (jj < nj && idx[jj] < E && cnt[jj] < k) sel[cnt[jj]] = idx[jj];   // (sum partials all read: 2 barriers ago)
    __syncthreads();
    if (wave == 0) {   // (k <= 32: lane j holds slot j's p, the sum walks the lanes in order)
        float ws = 0.0f;
        const int i = lane < k ? sel[lane] : 0, pj = probs[i];
        for (int j = 0; j < k; j++) ws = ws + __builtin_bit_cast(float, __builtin_amdgcn_readlane(pj, j));
        ws = fmaxf(ws, floor_);
        if (lane < k) { ids_out[lane] = i; wts_out[lane] = __builtin_bit_cast(float, pj) / ws; }
    }
    if (tid == 0) wts_out[k] = 1.0f / (1.0f + expf(-sgl[t]));
    if (bad) atomicOr(err, 1);
}

// ---- q35mp_emb_bf16: x [n][N] fp32 = the BF16 rows toks[i] (own_op_emb_bf16's statements: bf16 -> f32 is a shift) ------
// toks clamped to [0, vocab). Grid n, block 256.
extern "C" __global__ void __launch_bounds__(256) q35mp_emb_bf16(const uint16_t * __restrict__ W, int vocab, const int * __restrict__ toks,
                                                                 float * __restrict__ x, int N) {
    const int t = blockIdx.x;
    const int tok = min(max(toks[t], 0), vocab - 1);
    const uint16_t * row = W + (size_t)tok * N;
    float * xr = x + (size_t)t * N;
    for (int i = threadIdx.x; i < N; i += 256) xr[i] = __builtin_bit_cast(float, (uint32_t)row[i] << 16);
}
