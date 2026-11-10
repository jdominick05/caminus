// The qwen3moe decode module's ops without llama.cpp's headers (kernels/k3_q3m_lean.hip): the K3 core (kernels/k3.hip)
// plus exactly the ops of the decode tape (k3.Q3M_LEAN_OPS): Caminus's own OP_OWN_EMB / ADD_RMS16 / MV / ATTS
// (kernels/k3_own_ops.h) and the engine's OP_F32_MV / OP_ROUTER. The last two and their helpers are copied verbatim from
// kernels/k3_gguf_ops.h (the same statements: the module's k3_run is instruction for instruction the one the ggml-include
// build produced, checked by disassembly when this file changed); the wave reductions the own ops call are written here
// with ggml's statement order (xor partners width / 2 .. 1, __shfl_xor), so their bits are the same too.
#pragma once
#include <stdint.h>

template <int width = 32>
__device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) x += __shfl_xor(x, offset, width);
    return x;
}
template <int width = 32>
__device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) x = fmaxf(x, __shfl_xor(x, offset, width));
    return x;
}

// tokens per pass (speculative verification): Seg.team (k3_run ignores teams; <= 1 means one token)
__device__ __forceinline__ int seg_T(const Seg & sg) { return sg.team > 1 ? sg.team : 1; }

// wave-wide xor-butterfly reductions on DPP / permlanex16 (VALU) instead of ds_bpermute (an LDS round trip per step):
// the same partners in the same order (16, 8, 4, 2, 1), and each step's op (add, fmaxf) is commutative, so every lane
// gets warp_reduce_sum<32> / warp_reduce_max<32>'s bits. All 32 lanes must be active (as for the shuffles).
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

// an accumulator's 0.0f the compiler cannot see: unrolled, `acc = 0; acc += a0 * b0; acc += a1 * b1` would fold the
// first add and contract fadd(a0 * b0, a1 * b1) into an fma on EITHER product (it picked fma(a0, b0, a1 * b1)); the
// general loops it must match keep the zero and run fma(a1, b1, fma(a0, b0, 0)), so the fast paths start from this
__device__ __forceinline__ float acc0() { float z = 0.0f; asm volatile("" : "+v"(z)); return z; }
typedef __attribute__((address_space(1))) float GW;   // global (not flat) stores

#ifndef K3_F32_U
#define K3_F32_U 5
#endif
typedef __attribute__((address_space(1))) const float4 GF4;
// OP_F32_MV: out[r] = W[r] . x for F32 weights (the router ffn_gate_inp and the shared-expert gate). Seg.W = fp32 [N, K]
//   row-major, Seg.a = fp32 x [K], Seg.y = fp32 out [N]; one wave per row, lanes stride K by 4, then a wave reduction.
__device__ __forceinline__ void op_f32_mv(const Seg & sg, int t) {
    const int wave = K3WAVE, lane = K3TID % 32;
    const int r0 = t * sg.rows_per_task, r1 = min(sg.N, r0 + sg.rows_per_task);
    const float * W = (const float *)sg.W;
    const int nr = sg.N / seg_T(sg);           // weight rows; rows [T, nr], token tt reads x + tt * K
    for (int row = r0 + wave; row < r1; row += 8) {
        const int tt = row / nr;
        const float * w = W + (size_t)(row - tt * nr) * sg.K, * x = (const float *)sg.a + (size_t)tt * sg.K;
        // K3_F32_U lane iterations per batch: all their loads first, then the adds in the same per-lane order
        float acc = acc0();
        for (int k0 = lane * 4; k0 < sg.K; k0 += 128 * K3_F32_U) {
            float4 wv[K3_F32_U], xv[K3_F32_U];
#pragma unroll
            for (int u = 0; u < K3_F32_U; u++) if (k0 + u * 128 < sg.K) { wv[u] = *(const GF4 *)(w + k0 + u * 128); xv[u] = *(const GF4 *)(x + k0 + u * 128); }
#pragma unroll
            for (int u = 0; u < K3_F32_U; u++) if (k0 + u * 128 < sg.K)
                acc += wv[u].x * xv[u].x + wv[u].y * xv[u].y + wv[u].z * xv[u].z + wv[u].w * xv[u].w;
        }
        acc = warp_reduce_sum<32>(acc);
        if (lane == 0) ((float *)sg.y)[row] = acc;
    }
}

// OP_ROUTER (one task, one workgroup): llama.cpp's softmax gating with normalised weights (build_moe_ffn: soft_max,
//   argsort_top_k, get_rows, sum_rows, clamp, div). Seg.a = fp32 logits [E = N], Seg.y = int32 ids [k] out,
//   Seg.gs = fp32 weights [k] out, Seg.pad1 = k, Seg.rs = the clamp floor (6.103515625e-5).
// Thread tid owns logits i = tid + jj * 256 (jj < 4, E <= 1024), loaded once and kept in registers through the softmax
// (the same max / sum order as the LDS loops this replaced: per thread i ascending from 0.0f, the wave butterfly, then
// waves 0..7 in order). Top-k by RANK, no serial rounds: rank_i = #{j : p_j > p_i, or p_j == p_i and j < i} (the order
// repeated argmax with lower-index-first ties picks them in); the k ranked lanes park their index in LDS slot `rank`,
// and wave 0 sums p over slots 0..k-1 in order (bitwise the old per-round ws = ws + bv; the old rounds' wavefront
// release fences each waited for a global store: ~1.3 us per selected expert, 15 us per router at k = 10).
// Ranks count on the probabilities' bit patterns (p >= +0, so signed int order == float order, denormals included; the
// padding to a multiple of 32 is INT_MIN): j < i counts p_j >= p_i (q > pi - 1), j > i counts p_j > p_i. Each pass reads
// RK_C candidates per LDS round (16: 32 spills at the 96-VGPR cap) and counts them for both of a lane's logits (i and
// i + 256); the 32-lane groups are 32-aligned, so only the 2 chunks covering the lane's own group need the per-lane
// j < i test, the rest compare against one threshold.
#ifndef RK_C
#define RK_C 16
#endif
// (integer counts: any summation order is exact, so independent partial counts, not one dependent add chain)
__device__ __forceinline__ void router_rank_chunk(const int (&q)[RK_C], int c0, int b, int lane, int pi, int (&c)[2]) {
    if (c0 >= b && c0 < b + 32) {   // (opaque lane: the per-u thresholds must not be hoisted out of the chunk loop)
        int ln = lane - (c0 - b); asm volatile("" : "+v"(ln));
#pragma unroll
        for (int u = 0; u < RK_C; u++) c[u % 2] += q[u] > (u < ln ? pi - 1 : pi);
    } else {
        const int t = c0 < b ? pi - 1 : pi;
#pragma unroll
        for (int u = 0; u < RK_C; u++) c[u % 2] += q[u] > t;
    }
}
// ranks of NE elements (NE = 1 or 2: this lane's logits b0 + lane and b0 + 256 + lane) in one pass over the candidates
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
__device__ __forceinline__ void op_router(const Seg & sg, int tt) {
#pragma clang fp contract(off)
    __shared__ float rv[8]; __shared__ int probs[1024]; __shared__ int sel[64];   // sel[0..7]: the sum partials first
    const int tid = K3TID, wave = K3WAVE, lane = tid % 32, E = sg.N, k = sg.pad1;
    const float * lg = (const float *)sg.a + (size_t)tt * E;        // task tt = token tt: logits [T, E] -> ids / wts [T, k]
    int * ids_out = (int *)sg.y + tt * k; float * wts_out = (float *)sg.gs + tt * k;
    float v[4]; int idx[4];
#pragma unroll
    for (int jj = 0; jj < 4; jj++) { idx[jj] = tid + jj * 256; v[jj] = idx[jj] < E ? ((const GW *)lg)[idx[jj]] : -INFINITY; }
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
    const int nj = (E - wave * 32 + 255) / 256;   // wave-uniform: how many of this wave's jj slots hold a logit
    if (nj >= 2) router_rank<2>(probs, E32, wave * 32, lane, pi[0], pi[1], cnt[0], cnt[1]);
    else if (nj == 1) router_rank<1>(probs, E32, wave * 32, lane, pi[0], pi[1], cnt[0], cnt[1]);
    if (nj >= 4) router_rank<2>(probs, E32, wave * 32 + 512, lane, pi[2], pi[3], cnt[2], cnt[3]);
    else if (nj == 3) router_rank<1>(probs, E32, wave * 32 + 512, lane, pi[2], pi[3], cnt[2], cnt[3]);
#pragma unroll
    for (int jj = 0; jj < 4; jj++) if (jj < nj && idx[jj] < E && cnt[jj] < k) sel[cnt[jj]] = idx[jj];   // (sum partials all read: 2 barriers ago)
    __syncthreads();
    if (wave == 0) {
        float ws = 0.0f;
        if (k <= 32) {   // lane j holds slot j's p (one LDS round), the sum walks the lanes in order
            const int i = lane < k ? sel[lane] : 0, pj = probs[i];
            for (int j = 0; j < k; j++) ws = ws + __builtin_bit_cast(float, __builtin_amdgcn_readlane(pj, j));
            ws = fmaxf(ws, sg.rs);
            if (lane < k) { ids_out[lane] = i; wts_out[lane] = __builtin_bit_cast(float, pj) / ws; }
        } else {
            for (int j = 0; j < k; j++) ws = ws + __builtin_bit_cast(float, probs[sel[j]]);
            ws = fmaxf(ws, sg.rs);
            for (int j = lane; j < k; j += 32) { const int i = sel[j]; ids_out[j] = i; wts_out[j] = __builtin_bit_cast(float, probs[i]) / ws; }
        }
    }
    __syncthreads();
}

// touch(): pull [p, p + n) toward L2: one dword per 128-byte line, one line per thread per round, the value sunk into an
// empty asm (no arithmetic, no store). Global loads (not flat), no scratch; a <= 32 KB range is one round, one wait
__device__ __forceinline__ void touch(const uint8_t * p, uint32_t n) {   // p wave-uniform, n < 4 GB
    for (uint32_t o = K3TID * 128u; o < n; o += blockDim.x * 128u) {
        const uint32_t v = *(const __attribute__((address_space(1))) uint32_t *)(p + o);
        asm volatile("" :: "v"(v));
    }
}

// KV cache layouts per position (k3_gguf_ops.h's, for k3_own_ops.h's other ops: the lean tape runs none of them): [K | V],
// M values each; type 0 = f16, type 1 = q8_0 blocks {fp16 d; int8 qs[32]} (34 bytes)
struct KvQ8Block { _Float16 d; int8_t qs[32]; };
static_assert(sizeof(KvQ8Block) == 34, "q8_0 block");
__device__ __forceinline__ size_t kv_pos_bytes(int M, int type) { return type ? (size_t)2 * (M / 32) * sizeof(KvQ8Block) : (size_t)4 * M; }
__device__ __forceinline__ float kv_get(const uint8_t * pos_base, int M, int type, int which, int i) {   // which 0 = K, 1 = V
    if (type == 0) return (float)((const _Float16 *)pos_base)[(size_t)which * M + i];
    const KvQ8Block * b = (const KvQ8Block *)pos_base + (size_t)which * (M / 32) + i / 32;
    return (float)b->qs[i % 32] * (float)b->d;
}

#include "k3_own_ops.h"

__device__ __forceinline__ bool k3_extra_op(const Seg & sg, int t, bool fresh, Sync * sy, int limit_us, int seg,
                                            uint64_t * tr = nullptr) {
    if (sg.op == OP_OWN_MV) { own_op_mv_1rt(sg, t, (int)__builtin_amdgcn_readfirstlane(K3TID / 32), K3TID % 32); return true; }
    if (sg.op == OP_OWN_ATTS) { own_op_atts(sg, t); return true; }
#ifdef K3_Q3M_LEAN_ATTN   // (measurement builds only: the per-head attention too, for A/Bs inside one code object)
    if (sg.op == OP_OWN_ATTN) { own_op_attn(sg, t); return true; }
#endif
    if (sg.op == OP_OWN_ADD_RMS16) { own_op_add_rms16(sg); return true; }
    if (sg.op == OP_F32_MV) { op_f32_mv(sg, t); return true; }
    if (sg.op == OP_ROUTER) { op_router(sg, t); return true; }
    if (sg.op == OP_OWN_EMB) { own_op_emb(sg); return true; }
    return false;
}
__device__ __forceinline__ uint32_t * k3_extra_warm(const Seg &) { return nullptr; }
// before the dependency wait: the router matvec's (OP_F32_MV, one token) first weight bytes toward L2 (k3_gguf_ops.h's
// k3_extra_prefetch with K3_XWARM 0, its test kept whole: OP_GG_MV never runs here, but dropping it changed k3_run's
// branch layout)
__device__ __forceinline__ void k3_extra_prefetch(const Seg & sg, int t) {
    if (sg.op != OP_GG_MV && !(sg.op == OP_F32_MV && seg_T(sg) == 1 && sg.pad2 > 0)) return;
    const size_t row_bytes = (size_t)sg.pad2;
    const int r0 = t * sg.rows_per_task, r1 = min(sg.N, r0 + sg.rows_per_task);
    size_t n = (size_t)(r1 - r0) * row_bytes;
    const size_t cap = (size_t)sg.aux;
    if (cap) n = min(n, cap - 1);
    touch(sg.W + (size_t)r0 * row_bytes, (uint32_t)n);
}
