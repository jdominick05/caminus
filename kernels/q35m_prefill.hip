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
//   q35mp_attn       the chunk's attention tiles (q35_prefill.hip's q35p_attn_wm, which this object leaves out:
//                    Q35P_NO_ATTN_WM, reworked after rocKE's prefill pipeline; q35mp_attn_q8 in the Q8 object)
//
// The feed-forward's combine is q3p_add_rms16 over 9 slots: the 8 routed experts' down rows, then the shared expert's
// in slot 8, weighted by q35mp_router_sh's [n][9] weights (the decode's own_op_add_rms16 9-slot form: fma in slot order).
// q3p_add_rms16 writes h16 as the qwen35moe decode module's mixed fma does (one rounding of the exact product): only in this
// object, so q3m_prefill's and q35_prefill's code stays unchanged; likewise q3p_group's third index (Q3P_GROUP_Y9)
#define Q3P_ADD_RMS16_MIX
#define Q3P_GROUP_Y9
#define Q35P_NO_ATTN_WM
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

// ---- q35mp_attn: the chunk's causal attention as WMMA tiles, head dim 256, up to 8 query heads per kv head (the 35B: 8,
// every one of WMMA's 16 columns used), fp16 q / k / v / p, fp32 scores, softmax state and sums. q35_prefill.hip's
// q35p_attn_wm (the dense line's: kept there) reworked after rocKE's gfx11 prefill pipeline (ROCm/rocm-libraries,
// dnn-providers/hip-kernel-provider/rocke, library/kernels/gfx1151/wmma_fmha_fwd_v2.py and
// platform/python/rocke/helpers/rdna_attention.py at 9e91949; MIT License, Copyright (c) Advanced Micro Devices, Inc.:
// THIRD_PARTY.md), written here from that design; no rocKE code. A 2048-row chunk at the 35B's geometry (XT): 1.49 -> 0.80 ms
// at P0 0, 41.3 -> 20.95 ms at P0 30K (25.8 -> 50.8 TF; rocKE 19.6 ms), Q8 43.0 -> 22.1 ms. What follows rocKE:
//   - the next 32-position tile's K / V are loaded into registers at the top of a tile and stored to LDS behind it (K after
//     the scores' barrier, V after the sums'): two barriers per tile, the global latency behind the tile's 32 WMMAs.
//     q35p_attn_wm loaded each tile, then waited, behind three barriers: 2.85 ms -> rocKE 0.94 ms at P0 0 (dense)
//   - V^T staged by 4-position units: a thread loads 4 rows x 8 dims and writes 8 runs of 4 positions (ds_write_b64, two
//     v_perm each) instead of 32 scalar half writes
//   - the pair's score partials: each wave publishes its own and reads its partner's (own + partner: the same sum in both
//     waves, fp addition is commutative), as float4s, instead of both waves reading both
//   - lazy rescale: a column keeps its running max while a tile's max exceeds it by at most 8 (log2 units: p <= 256,
//     exact in fp16's range); the wave skips O's rescale when every alpha is exactly 1 (a skipped multiply by 1 changes
//     no bit). Per column, unlike rocKE's wave vote, so a column's bits never depend on the other columns of its wave
//   - the interior tiles (every position before the workgroup's first) run without masks; heaviest blocks first
// What stays Caminus's (q35p_attn_wm's argument, unchanged): work by ABSOLUTE position, so chunk geometry changes no bit.
// Workgroup i = positions 8 i .. 8 i + 7 x the G query heads of kv head blockIdx.y; pair w (waves 2 w, 2 w + 1, each half
// of the head's dims: Q 64 VGPRs, O 64) = positions q0 = 8 i + 2 w, q1 = q0 + 1; column n = lane & 15 -> position
// q0 + (n >> 3), head G g + (n & 7) (n & 7 >= G: head G g + G - 1 again, never written). Tiles of 32 positions from 0.
// RDNA3's WMMA is not exact for a zero product (kernels/q3m_prefill.hip q3p_attn), so in the pair's diagonal tile V^T is
// zeroed past q0 and q1's own row added with fp32 fmas; tiles past the pair are skipped. Only the last tile holds the
// workgroup's own positions; rows past the chunk's end are clamped to its last row (never in a written column's sum).
// Q8 rows (q35mp_attn_q8, the Q8 KV object; K and q rotated by Hadamard256 on the host, V and the output by Hadamard64:
// the kernel is rotation-blind): each staged value fp16(float code * float scale), q3p_q8_stage's arithmetic. A staging
// chunk (8 dims of a position) loads the dword holding its block's scale and the 3 dwords holding its 8 codes; every byte
// read lies inside its position's row (an even block is never a row's last, an odd one never its first).
// q: the chunk's [n][heads][256] fp32 (rounded to fp16 here), out the same shape (fp32). scale_log2 = scale * log2(e).
// Grid (8-position blocks blk0 .. blk0 + gridDim.x - 1, nkv), block 256; LDS K [32][MA_KP], V^T [256][MA_VP], the
// partials [8][4][32] float4 (53,760 B: two workgroups per WGP).
#define MA_C 32
#define MA_D 256
#define MA_KP (MA_D + 8)
#define MA_VP (MA_C + 8)
#ifdef Q35P_KV_Q8
#define Q35MP_ATTN_NAME q35mp_attn_q8
#define Q35MP_CACHE uint8_t
#else
#define Q35MP_ATTN_NAME q35mp_attn
#define Q35MP_CACHE _Float16
#endif
template <bool B> struct mp_bool { static constexpr bool v = B; };
// a barrier that orders LDS only: __syncthreads' workgroup fence also waits for every global load in flight (vmcnt(0) in
// WGP mode), which would drain the next tile's prefetch at each barrier; the tiles' global data is read-only here
__device__ __forceinline__ void mp_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
// staging chunk c (0 .. 31: dims 8 c .. 8 c + 7) of kv head g's K (part 0) or V (part 1) in position row prow (>= 0), raw
__device__ __forceinline__ u32x4_t mp_load(const Q35MP_CACHE * __restrict__ cache, int prow, int M, int part, int g, int c) {
#ifdef Q35P_KV_Q8
    // block (part M / 32 + g 8 + c / 4) of the row; od = 2 for an odd block (rows and kv heads are whole dwords)
    const uint8_t * blk = cache + (uint64_t)(uint32_t)prow * (uint32_t)own_kv_q8_row_bytes(M) + (uint32_t)(part * (M / 32) + g * (MA_D / 32) + (c >> 2)) * 34u;
    const int od = (c & 4) >> 1;
    const uint32_t * s = (const uint32_t *)(blk - od);                 // the scale: the low (od 0) or high half
    const uint32_t * w = (const uint32_t *)(blk + 8 * (c & 3) - od);   // codes from byte 2 (od 0) or 4 (od 2) of w[0]
    return (u32x4_t){s[0], w[0], w[1], w[2]};
#else
    return *(const u32x4_t *)((const uint8_t *)cache + (uint64_t)(uint32_t)prow * (uint32_t)(4 * M) + (uint32_t)(2 * (part * M + g * MA_D + c * 8)));
#endif
}
// the chunk's 8 values as fp16 pairs (dims 2 j, 2 j + 1 in dword j). Q8: fp16(code * d), q3p_q8_stage's value (code and d
// exact in fp16, their product exact in fp32, one rounding to fp16) by packed fp16 math: two codes' bytes into halves 0 and 1
// biased to fp16 1024 + (code ^ 0x80) = 1152 + code (exact), less 1152 (exact), times (d, d) (v_pk_mul: the exact product
// rounded once, the same bits): 44.8 -> 47.6 TF at P0 30K (XT) against the fp32 conversions
__device__ __forceinline__ u32x4_t mp_h8(u32x4_t r, int c) {
#ifdef Q35P_KV_Q8
    const bool od = (c & 4) != 0;
    const uint32_t d2 = __builtin_amdgcn_perm(r.x, r.x, od ? 0x03020302u : 0x01000100u);   // (d, d)
    const uint32_t w0 = od ? r.z : __builtin_amdgcn_alignbyte(r.z, r.y, 2), w1 = od ? r.w : __builtin_amdgcn_alignbyte(r.w, r.z, 2);
    const h2_t b2 = {(_Float16)1152.0f, (_Float16)1152.0f};
    uint32_t o[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const uint32_t sel = (j & 1) ? 0x0c030c02u : 0x0c010c00u;   // (bytes 0, 1 or 2, 3 of w into halves 0, 1)
        o[j] = as_u((as_h2(__builtin_amdgcn_perm(0u, j < 2 ? w0 : w1, sel) ^ 0x64806480u) - b2) * as_h2(d2));
    }
    return (u32x4_t){o[0], o[1], o[2], o[3]};
#else
    (void)c;
    return r;
#endif
}
extern "C" __global__ void __launch_bounds__(256) Q35MP_ATTN_NAME(const float * __restrict__ qin, const Q35MP_CACHE * __restrict__ cache,
                                                                 int pos0, int n, int blk0, int heads, int nkv, float scale_log2,
                                                                 float * __restrict__ out) {
    __shared__ __attribute__((aligned(16))) _Float16 Ks[MA_C * MA_KP];
    __shared__ __attribute__((aligned(16))) _Float16 Vt[MA_D * MA_VP];
    __shared__ __attribute__((aligned(16))) float Sx[8][4][32][4];   // [wave][2 st + h][lane]: its S[st] slots 4 h .. 4 h + 3
    // (wave through readfirstlane: q0, the pair's subtile and every branch on them are scalar, not exec-masked)
    const int tid = threadIdx.x, lane = tid & 31, wave = __builtin_amdgcn_readfirstlane(tid >> 5), hi = lane >> 4, dh = (wave & 1) * 128;
    const int G = heads / nkv, g = blockIdx.y, M = nkv * MA_D;
    const int b8 = (blk0 + (int)(gridDim.x - 1 - blockIdx.x)) * 8;   // (the deepest blocks first)
    const int nq = lane & 15, sec = nq >> 3, hh = nq & 7;
    const int q0 = b8 + 2 * (wave >> 1), q1 = q0 + 1;
    const int qpos = q0 + sec, tq = qpos - pos0, hq = g * G + min(hh, G - 1);
    const bool qin_ok = tq >= 0 && tq < n, qok = qin_ok && hh < G;
    h16_t Qb[8];                                        // this wave's half of the column's q, 16 dims per k-step
    {
        const float4 * qp = (const float4 *)(qin + ((size_t)(qin_ok ? tq : 0) * heads + hq) * MA_D + dh);
#pragma unroll
        for (int s = 0; s < 8; s++) {
            h16_t v;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float4 f = qp[4 * s + j];
                v[4 * j] = (_Float16)f.x; v[4 * j + 1] = (_Float16)f.y; v[4 * j + 2] = (_Float16)f.z; v[4 * j + 3] = (_Float16)f.w;
            }
            Qb[s] = v;
            if (s == 3) __builtin_amdgcn_sched_barrier(0);   // (half of q's loads in flight at a time: all 32 spilled)
        }
    }
    f8_t O[8];
#pragma unroll
    for (int d = 0; d < 8; d++) O[d] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.0f;
    // rows 0 .. npos - 1 exist for this workgroup (its last position's); tiles below clast are interior (every position
    // before b8 <= q0 of every pair), clast's tile holds the workgroup's own positions (npos > b8 >= clast)
    const int npos = min(pos0 + n, b8 + 8), clast = b8 & ~(MA_C - 1);
    // staging: K chunk kc of rows kr + 8 i (a wave = one row's 512 bytes); V chunk vc of rows 4 vg .. 4 vg + 3
    const int kr = tid >> 5, kc = tid & 31, vg = tid & 7, vc = tid >> 3;
    u32x4_t rk[4], rv[4];
    auto load_k = [&](int c0) {
#pragma unroll
        for (int i = 0; i < 4; i++) rk[i] = mp_load(cache, min(c0 + kr + 8 * i, npos - 1), M, 0, g, kc);
    };
    auto load_v = [&](int c0) {
#pragma unroll
        for (int r = 0; r < 4; r++) rv[r] = mp_load(cache, min(c0 + 4 * vg + r, npos - 1), M, 1, g, vc);
    };
    auto store_k = [&]() {
#pragma unroll
        for (int i = 0; i < 4; i++) *(u32x4_t *)(Ks + (kr + 8 * i) * MA_KP + kc * 8) = mp_h8(rk[i], kc);
    };
    auto store_v = [&]() {
        u32x4_t h[4];
#pragma unroll
        for (int r = 0; r < 4; r++) h[r] = mp_h8(rv[r], vc);
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int od = 0; od < 2; od++) {             // dim 8 vc + 2 j + od: its half of dword j of the 4 rows
                const uint32_t sel = od ? 0x07060302u : 0x05040100u;
                *(uint2 *)(Vt + (vc * 8 + 2 * j + od) * MA_VP + 4 * vg) =
                    (uint2){__builtin_amdgcn_perm(h[1][j], h[0][j], sel), __builtin_amdgcn_perm(h[3][j], h[2][j], sel)};
            }
    };
    auto tile = [&](int c0, auto lastc) {
        constexpr bool LAST = decltype(lastc)::v;
        // the pair's subtile: positions c0 + 16 std .. + 15 hold q0 and q1 (q0 even: never across a subtile edge); later
        // subtiles are all masked: skipped (interior tiles: both, no masks)
        const int std_ = LAST ? min(1, (q0 - c0) >> 4) : 1, kq0 = q0 - c0 - 16 * std_;
        f8_t S[2] = {(f8_t){0, 0, 0, 0, 0, 0, 0, 0}, (f8_t){0, 0, 0, 0, 0, 0, 0, 0}};
#pragma unroll
        for (int s = 0; s < 8; s++)                     // (k-step outer: the two subtiles' chains interleave)
#pragma unroll
            for (int st = 0; st < 2; st++) {
                if (LAST && st > std_) continue;
                const u32x4_t * ap = (const u32x4_t *)(Ks + (st * 16 + (lane & 15)) * MA_KP + dh + 16 * s);
                S[st] = wmma(h16of(ap[0], ap[1]), Qb[s], S[st]);
                if (st == 1 && (s & 1)) __builtin_amdgcn_sched_barrier(0);   // (K fragments read 4 ahead, not all 16)
            }
#pragma unroll
        for (int st = 0; st < 2; st++)
#pragma unroll
            for (int h = 0; h < 2; h++)
                *(float4 *)Sx[wave][2 * st + h][lane] = (float4){S[st][4 * h], S[st][4 * h + 1], S[st][4 * h + 2], S[st][4 * h + 3]};
        mp_lds_barrier();                                // (the partials visible; every wave done reading K)
        if (!LAST) {                                    // the next tile's K to LDS, its V loads issued behind the sums
            store_k();
            load_v(c0 + MA_C);
            __builtin_amdgcn_sched_barrier(0);
        }
        float cm = -INFINITY;
#pragma unroll
        for (int st = 0; st < 2; st++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const float4 t = *(const float4 *)Sx[wave ^ 1][2 * st + h][lane];
                const float pv[4] = {t.x, t.y, t.z, t.w};
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int v = 4 * h + e;
                    float x = S[st][v] + pv[e];
                    if (LAST && !(st <= std_ && c0 + st * 16 + 2 * v + hi <= qpos)) x = -INFINITY;
                    S[st][v] = x; cm = fmaxf(cm, x);
                }
            }
        // (finite: every tile holds a position the column sees; scaled after the max: the same value, scale_log2 > 0)
        cm = fmaxf(cm, __shfl_xor(cm, 16, 32)) * scale_log2;
        const float mn = cm - m <= 8.0f ? m : fmaxf(m, cm);
        const float alpha = __builtin_amdgcn_exp2f(m - mn);   // (0 on the first tile, exactly 1 when the max is kept)
        float rsum = 0.0f;
        uint32_t pw[2][4];                              // fp16 p, two positions per dword: [st][v pair]
#pragma unroll
        for (int st = 0; st < 2; st++)
#pragma unroll
            for (int v = 0; v < 8; v += 2) {
                const _Float16 p0 = (_Float16)__builtin_amdgcn_exp2f(__builtin_fmaf(S[st][v], scale_log2, -mn)),
                               p1 = (_Float16)__builtin_amdgcn_exp2f(__builtin_fmaf(S[st][v + 1], scale_log2, -mn));
                rsum += (float)p0 + (float)p1;
                pw[st][v >> 1] = as_u((h2_t){p0, p1});
            }
        rsum += __shfl_xor(rsum, 16, 32);
        l = l * alpha + rsum; m = mn;
        if (__builtin_amdgcn_ballot_w32(alpha != 1.0f)) {
#pragma unroll
            for (int d = 0; d < 8; d++)
#pragma unroll
                for (int v = 0; v < 8; v++) O[d][v] *= alpha;
        }
#pragma unroll
        for (int st = 0; st < 2; st++) {
            if (LAST && st > std_) continue;
            // P^T operand: positions 16 st .. 16 st + 15 of the lane's column: own (2 v + hi) and lane ^ 16's (2 v + 1 - hi)
            uint32_t o[8];
#pragma unroll
            for (int w = 0; w < 4; w++) {
                const uint32_t mine = pw[st][w], other = (uint32_t)__shfl_xor((int)mine, 16, 32);
                const uint32_t ev = hi ? other : mine, odd = hi ? mine : other;   // even positions, odd positions
                o[2 * w] = __builtin_amdgcn_perm(odd, ev, 0x05040100u);       // positions 4w, 4w + 1
                o[2 * w + 1] = __builtin_amdgcn_perm(odd, ev, 0x07060302u);   // positions 4w + 2, 4w + 3
            }
            const bool diag = LAST && st == std_ && kq0 < 16;
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
                const u32x4_t * ap = (const u32x4_t *)(Vt + (dh + d * 16 + (lane & 15)) * MA_VP + st * 16);
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
                const int kc1 = 16 * st + kq0 + 1;      // q1's row in the tile
#pragma unroll
                for (int d = 0; d < 8; d++)
#pragma unroll
                    for (int v = 0; v < 8; v++)
                        O[d][v] = __builtin_fmaf(p_q1, (float)Vt[(dh + d * 16 + 2 * v + hi) * MA_VP + kc1], O[d][v]);
            }
        }
        if (!LAST) {
            mp_lds_barrier();                            // (every wave done reading V^T and the partials)
            store_v();
        }
    };
    load_k(0); load_v(0); store_k(); store_v();
    mp_lds_barrier();
    for (int c0 = 0; c0 < clast; c0 += MA_C) {
        load_k(c0 + MA_C);                              // (the next tile's K behind this one's score WMMAs; at most clast's)
        __builtin_amdgcn_sched_barrier(0);              // (issued here: the scheduler sank them past the WMMAs)
        tile(c0, mp_bool<false>{});
    }
    tile(clast, mp_bool<true>{});
    if (!qok) return;
    const float il = 1.0f / l;
    float * op = out + ((size_t)tq * heads + hq) * MA_D + dh;
#pragma unroll
    for (int d = 0; d < 8; d++)
#pragma unroll
        for (int v = 0; v < 8; v++) op[d * 16 + 2 * v + hi] = O[d][v] * il;
}
