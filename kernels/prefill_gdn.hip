// Batched prefill: the GDN (gated delta net) half of a layer over a chunk of T tokens (docs/agentic/spec-prefill-cache.md
// M5 S4; host side python/caminus/prefill_gdn.py). Ordinary launches (not K3: no tape, no persistent kernel). Every kernel
// computes each output element with the expressions of its K3 decode op (kernels/k3_gguf_ops.h) in the same order, so it
// is BITWISE that op on the same inputs (gate G1, tests/test_prefill_gdn.py, against the K3 op itself run per token
// through caminus.k3.Engine):
//   pf_gdn_l2        OP_GDN_L2: per (token, head) of 128 values: y = (x * rsqrt(sum(x^2) / K + eps / K)) * (1 / sqrt(K)),
//                    256 threads (thread i < 128 holds x[i]), the wave sums, the 8 wave partials summed in order
//   pf_gdn_gates     OP_GDN_GATES: gate = softplus(alpha + dt) * a, beta_sig = sigmoid(beta) (contract off)
//   pf_gdn_step_*    OP_GDN_STEP: the gated delta rule, a PER-TOKEN fp32 RECURRENCE (not a chunked scan): each
//                    workgroup keeps its columns of one value head's state in registers and walks the chunk's tokens in
//                    order with OP_GDN_STEP's update code (gdn_dot / gdn_upd, copied verbatim: the same explicit fmaf, the
//                    same contraction of v - g * kv under the same default fp-contract) and its partial sums and
//                    butterfly adds. Optionally the decode's lazy commit first (mprev saved-input records replayed, as
//                    OP_GDN_STEP does), so a prefill can follow K3 decode passes whose accepted tokens are not committed
//                    yet. State snapshots at listed positions inside the chunk are written on the fly (the prompt cache's
//                    checkpoints: no chunk has to be split at a checkpoint). Variants (the host picks one: every one gives
//                    the same bits, G2a, since a column's arithmetic does not depend on where it runs):
//                      _g8 (default), _g8b8, _g8w4: 8 lanes per state column (below), inputs staged through LDS in blocks
//                        of 16 / 8 / 16 tokens, workgroups of 8 / 8 / 4 waves
//                      _w8b16: OP_GDN_STEP's own lane map (32 lanes per column: lane l holds values 4l .. 4l + 3), LDS
//                        blocks of 16 tokens
//                      _direct: the first version, 32 lanes per column, no LDS, one token of register prefetch
//   pf_gdn_norm      OP_GDN_NORM: per (token, value head): y = ((o * rsqrt(mean(o^2) + eps)) * w) * sigmoid(z)
//                    (contract off; no q8 copy: the prefill's ssm_out MMQ quantizes its own fp32 input)
// The wave sums use the DPP / permlanex16 butterfly (wsum, k3_gguf_ops.h wred's partners): the partners and order of
// ggml's warp_reduce_sum<32> (xor 16, 8, 4, 2, 1), and each step's add is commutative, so every lane gets that sum's bits.
// Compiled by build.sh's generic loop like the other non-K3 objects; the decode code objects are untouched.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef __attribute__((address_space(1))) const float GF;   // global (not flat) loads, as k3.hip
typedef __attribute__((address_space(1))) float GW;         // global stores, as k3_gguf_ops.h

// ---- helpers ------------------------------------------------------------------------------------------------------------
// the wave sum: k3_gguf_ops.h wred<false>'s butterfly (permlanex16 for lane ^ 16, then DPP row_xmask 8, 4, 2, 1: the
// partners and order of warp_reduce_sum<32>), written with mov_dpp and bound_ctrl set (row_xmask with full row / bank
// masks gives every lane a source inside its row, so bound_ctrl never applies) so LLVM's DPP combiner folds each step
// into its add (v_add_f32_dpp; it needs bound_ctrl or an immediate-zero old). k3_gguf_ops.h's update_dpp(x, x)
// ties the destination to the live x and costs a copy + a DPP move + an add per step: 8 reductions x 5 steps per token
// and wave in the recurrence, ~1/3 of its VALU instructions. The same adds of the same values: the same bits (G1)
__device__ __forceinline__ float wsum(float x) {
    const unsigned u = __builtin_bit_cast(unsigned, x);
    x = x + __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(u, u, 0x76543210u, 0xfedcba98u, false, false));
    x = x + __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x168, 0xf, 0xf, true));   // row_xmask:8
    x = x + __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x164, 0xf, 0xf, true));   // row_xmask:4
    x = x + __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x162, 0xf, 0xf, true));   // row_xmask:2
    x = x + __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x161, 0xf, 0xf, true));   // row_xmask:1
    return x;
}
// OP_GDN_STEP's state column update and dot product (k3_gguf_ops.h gdn_upd / gdn_dot, verbatim)
__device__ __forceinline__ float4 gdn_upd(const float4 m, const float4 k4, float kv, float vcol, float g, float beta) {
    // the state column update; explicit operations so every inlined copy rounds alike
    const float t1 = g * kv, t2 = vcol - t1, delta = t2 * beta;
    float4 sv;
    sv.x = fmaf(k4.x, delta, g * m.x); sv.y = fmaf(k4.y, delta, g * m.y);
    sv.z = fmaf(k4.z, delta, g * m.z); sv.w = fmaf(k4.w, delta, g * m.w);
    return sv;
}
__device__ __forceinline__ float gdn_dot(const float4 m, const float4 k4) {
    return fmaf(m.w, k4.w, fmaf(m.z, k4.z, fmaf(m.y, k4.y, m.x * k4.x)));
}

// ---- pf_gdn_l2: OP_GDN_L2 over T tokens, one workgroup of 256 per (token, head) -------------------------------------
// x: head hh of token tt at x + tt * ldx + hh * K (ldx = the conv output's row: q and k heads first); y [T * heads, K];
// rs = eps / K (the decode's Seg.rs, rounded to fp32 on the host from the same double)
extern "C" __global__ void __launch_bounds__(256) pf_gdn_l2(const float * x, float * y, int heads, int K, int ldx, float rs) {
    __shared__ float part[8];
    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32, t = blockIdx.x;
    const int tt = t / heads, hh = t - tt * heads;
    const float * xr = x + (size_t)tt * ldx + (size_t)hh * K;
    float * yr = y + (size_t)t * K;
    float ss = 0.0f;
    for (int i = tid; i < K; i += 256) ss += xr[i] * xr[i];
    ss = wsum(ss); if (lane == 0) part[wave] = ss; __syncthreads();
    float tot = 0.0f; for (int q = 0; q < 8; q++) tot += part[q];
    const float scale = rsqrtf(tot / K + rs), s2 = 1.0f / sqrtf((float)K);
    for (int i = tid; i < K; i += 256) yr[i] = (scale * xr[i]) * s2;
}

// ---- pf_gdn_gates: OP_GDN_GATES over T tokens, one thread per (token, value head) -----------------------------------
// alpha / beta [T, H] (row strides lda / ldb), dt / a [H]; out [T, 2H] (gate, then beta_sig)
extern "C" __global__ void __launch_bounds__(256) pf_gdn_gates(const float * al, int lda, const float * be, int ldb, const float * dt,
                                                               const float * A, float * y, int H, int T) {
#pragma clang fp contract(off)
    const long q = (long)blockIdx.x * 256 + threadIdx.x;
    if (q >= (long)T * H) return;
    const int tt = (int)(q / H), h = (int)(q - (long)tt * H);
    const float ab = al[(size_t)tt * lda + h] + dt[h];
    const float sp = (ab > 20.0f) ? ab : logf(1.0f + expf(ab));
    y[(size_t)tt * 2 * H + h] = sp * A[h];
    y[(size_t)tt * 2 * H + H + h] = 1.0f / (1.0f + expf(-be[(size_t)tt * ldb + h]));
}

// ---- pf_gdn_step: OP_GDN_STEP over a chunk ---------------------------------------------------------------------------
// qk [T, 2, nk, S] (q heads, then k heads; l2-normed), v: value head h of token tt at v + tt * ldv + h * S, gates [T, 2H]
// (gate, beta_sig), state [H, S(col), S(i)] (llama.cpp's transposed layout, K3's M[h][col][i]: read once, written once
// after the chunk), out [T, H, S]. Value head h reads key head h % nk (ggml_repeat tiling). rec / mprev: the decode's
// lazy commit (OP_GDN_SAVE's records [mprev][nk * S + H * S + 2H]: k, v, gates), replayed before the chunk's first token
// exactly as OP_GDN_STEP replays them (0 = none). Snapshots: after the token at chunk position pos[i] - 1 (that is, the
// state after pos[i] tokens, 1 <= pos[i] <= T, increasing), the state goes to snap + i * snap_stride (same layout).
// A workgroup = WAVES waves, CPW = 4 state columns per wave (K3's lane map), all of one value head; grid H * S / CPT.
#define GDN_S 128
#define GDN_CPW 4                 // state columns per wave (K3's 4: one float4 of each column per lane)
#define GDN_MAX_SNAP 8
struct GdnSnaps { int n; int pos[GDN_MAX_SNAP]; };

// one token of OP_GDN_STEP for a wave's CPW columns (the op's code: kv, then update + attention per column)
__device__ __forceinline__ void gdn_token(float4 (&m)[GDN_CPW], const float4 k4, const float4 q4, const float4 vv, float ga,
                                          float beta, float (&at)[GDN_CPW]) {
    const float g = expf(ga);
    const float vc[GDN_CPW] = {vv.x, vv.y, vv.z, vv.w};
    float kv[GDN_CPW];
#pragma unroll
    for (int j = 0; j < GDN_CPW; j++) kv[j] = wsum(gdn_dot(m[j], k4));
#pragma unroll
    for (int j = 0; j < GDN_CPW; j++) {
        m[j] = gdn_upd(m[j], k4, kv[j], vc[j], g, beta);
        at[j] = wsum(gdn_dot(m[j], q4));
    }
}
// the state load + the lazy-commit replay (OP_GDN_STEP's, before its pass's tokens)
__device__ __forceinline__ void gdn_begin(float4 (&m)[GDN_CPW], const float4 * M, const float * rec, int mprev, int c0, int lane,
                                          int h, int kh, int H, int nk) {
    constexpr int S = GDN_S, CPW = GDN_CPW;
    const int RS = nk * S + H * S + 2 * H;                         // a saved input record per token (OP_GDN_SAVE)
#pragma unroll
    for (int j = 0; j < CPW; j++) m[j] = M[(size_t)(c0 + j) * (S / 4) + lane];
#pragma unroll 1
    for (int r = 0; r < mprev; r++) {
        const float * rr = rec + (size_t)r * RS;
        const float4 k4 = ((const float4 *)(rr + (size_t)kh * S))[lane];
        const float * vr = rr + nk * S + (size_t)h * S;
        const float g = expf(rr[nk * S + H * S + h]), beta = rr[nk * S + H * S + H + h];
        float kv[CPW];
#pragma unroll
        for (int j = 0; j < CPW; j++) kv[j] = wsum(gdn_dot(m[j], k4));
#pragma unroll
        for (int j = 0; j < CPW; j++) m[j] = gdn_upd(m[j], k4, kv[j], vr[c0 + j], g, beta);
    }
}
// after token tt: its attention row (lane 0: the wave's 4 columns as one float4) and a snapshot if tt + 1 is listed
__device__ __forceinline__ void gdn_emit(const float4 (&m)[GDN_CPW], const float (&at)[GDN_CPW], float * out, float * snap,
                                         long snap_stride, const GdnSnaps & sp, int ns, int & si, int & snext, int tt, int c0,
                                         int lane, int h, int H) {
    constexpr int S = GDN_S;
    const float scale = 1.0f / sqrtf((float)S);
    if (lane == 0) *(float4 *)(out + ((size_t)tt * H + h) * S + c0) = make_float4(at[0] * scale, at[1] * scale, at[2] * scale, at[3] * scale);
    if (tt + 1 == snext) {                                         // a checkpoint: the state after tt + 1 tokens
        float4 * Sn = (float4 *)(snap + (size_t)si * snap_stride + (size_t)h * S * S);
#pragma unroll
        for (int j = 0; j < GDN_CPW; j++) Sn[(size_t)(c0 + j) * (S / 4) + lane] = m[j];
        si++; snext = si < ns ? sp.pos[si] : -1;
    }
}

// pf_gdn_step_w8b16: the chunk's inputs staged through LDS in blocks of B tokens, double-buffered:
// the whole workgroup loads block b + 1 (k and q rows of its key head, its columns' v, the head's gate / beta) with
// vector loads while it computes block b from LDS, then one barrier. Measured on the XT (card 0's shapes, T = 2048):
// the first version (pf_gdn_step_direct below: each wave prefetched one token ahead in registers) ran 1.87 ms per layer
// at every workgroup size: v and the gates of a token are uniform per wave, the compiler loaded them with scalar loads,
// and scalar loads return out of order, so every use waits for all of them (lgkmcnt(0)) including the prefetch issued
// in the same iteration: one memory latency (~0.9 us) per token. A block of B tokens in flight hides it.
template <int WAVES, int B>
__device__ __forceinline__ void gdn_step_lds(const float * qk, const float * v, int ldv, const float * gates, float * state,
                                             const float * rec, int mprev, float * out, float * snap, long snap_stride,
                                             const GdnSnaps & sp, int H, int nk, int T) {
    constexpr int S = GDN_S, CPW = GDN_CPW, CPT = WAVES * CPW, NT = 32 * WAVES;
    constexpr int NKQ = 2 * B * 32, PT = (NKQ + NT - 1) / NT;      // float4s of k + q per block; per thread
    static_assert(B * WAVES <= NT && B <= NT, "one v / gate load per thread");
    __shared__ float4 Lkq[2][2 * B][32];                           // [buffer][token * 2 + (0: k, 1: q)][lane]
    __shared__ float4 Lv[2][B][WAVES];                             // [buffer][token][wave]: the wave's 4 columns of v
    __shared__ float2 Lg[2][B];                                    // [buffer][token]: gate, beta_sig of the head
    const int tid = threadIdx.x, lane = tid % 32, wave = __builtin_amdgcn_readfirstlane(tid / 32);
    const int h = blockIdx.x / (S / CPT), cw = (blockIdx.x % (S / CPT)) * CPT, c0 = cw + wave * CPW, kh = h % nk;
    const size_t qks = (size_t)2 * nk * S;
    float4 m[CPW];
    gdn_begin(m, (const float4 *)(state + (size_t)h * S * S), rec, mprev, c0, lane, h, kh, H, nk);
    const int ns = min(max(sp.n, 0), GDN_MAX_SNAP);
    int si = 0, snext = ns > 0 ? sp.pos[0] : -1;
    float4 rkq[PT], rv; float2 rg;
    auto load = [&](int b) {                                       // block b's inputs into registers (zeros past T)
        const int t0 = b * B;
#pragma unroll
        for (int p = 0; p < PT; p++) {
            const int e = tid + p * NT, tok = e / 64, kind = (e / 32) % 2, l = e % 32;
            rkq[p] = (e < NKQ && t0 + tok < T) ? ((const float4 *)(qk + (size_t)(t0 + tok) * qks + (size_t)(kind ? kh : nk + kh) * S))[l]
                                               : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        {
            const int tok = tid / WAVES, w = tid % WAVES;
            rv = (tid < B * WAVES && t0 + tok < T) ? *(const float4 *)(v + (size_t)(t0 + tok) * ldv + (size_t)h * S + cw + w * CPW)
                                                   : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            rg = (tid < B && t0 + tid < T) ? make_float2(gates[(size_t)(t0 + tid) * 2 * H + h], gates[(size_t)(t0 + tid) * 2 * H + H + h])
                                           : make_float2(0.0f, 0.0f);
        }
    };
    auto stash = [&](int buf) {
#pragma unroll
        for (int p = 0; p < PT; p++) {
            const int e = tid + p * NT;
            if (e < NKQ) Lkq[buf][2 * (e / 64) + (e / 32) % 2][e % 32] = rkq[p];   // (kind 0 = the k row: Lkq[.][2 tok])
        }
        if (tid < B * WAVES) Lv[buf][tid / WAVES][tid % WAVES] = rv;
        if (tid < B) Lg[buf][tid] = rg;
    };
    const int nb = (T + B - 1) / B;
    load(0); stash(0); __syncthreads();
#pragma unroll 1
    for (int b = 0; b < nb; b++) {
        if (b + 1 < nb) load(b + 1);                               // in flight while block b computes
        const int buf = b & 1, n = min(B, T - b * B);
#pragma unroll 1
        for (int i = 0; i < n; i++) {
            const int tt = b * B + i;
            const float4 k4 = Lkq[buf][2 * i][lane], q4 = Lkq[buf][2 * i + 1][lane], vv = Lv[buf][i][wave];
            const float2 g2 = Lg[buf][i];
            float at[CPW];
            gdn_token(m, k4, q4, vv, g2.x, g2.y, at);
            gdn_emit(m, at, out, snap, snap_stride, sp, ns, si, snext, tt, c0, lane, h, H);
        }
        if (b + 1 < nb) stash(buf ^ 1);                            // (the other buffer: every wave left it at the last barrier)
        __syncthreads();
    }
    float4 * M = (float4 *)(state + (size_t)h * S * S);
#pragma unroll
    for (int j = 0; j < CPW; j++) M[(size_t)(c0 + j) * (S / 4) + lane] = m[j];
}

// pf_gdn_step_g8 (the default): the same arithmetic with 8 lanes per state column instead of 32. Lane j of a column's
// group of 8 holds that column's values of K3's lanes j, j + 8, j + 16, j + 24 (four float4: values 4 (j + 8 b) .. + 3),
// computes those four lanes' partial dot products exactly as they do (gdn_dot), and performs the first two levels of
// K3's butterfly in registers: (p0 + p2) is K3's lane-j step lane ^ 16, (p1 + p3) its lane-(j + 8) step, their sum its
// lane-j step lane ^ 8; then lane ^ 4, ^ 2, ^ 1 by DPP inside the group (row_xmask stays inside it). Every add has K3's
// operands, so every value is K3's (G1); the update runs on the same values with the same delta. One wave instruction
// now serves 4 columns: per token and wave 8 reductions of 6 VALU instructions become 2 groups of 3 + 3 DPP, and the
// per-column delta is computed once per group. Measured on the XT (card 0's shapes, T = 2048): see tests/test_prefill_gdn.py
// speed (the 32-lane layout, w8b16, 1.48 ms; it was VALU-issue bound: one head alone 0.24 us per token, 30 heads 0.73,
// 60 heads 1.35)
__device__ __forceinline__ float g8_sum(const float4 (&m)[4], const float4 (&k)[4]) {
    const float p0 = gdn_dot(m[0], k[0]), p1 = gdn_dot(m[1], k[1]), p2 = gdn_dot(m[2], k[2]), p3 = gdn_dot(m[3], k[3]);
    float x = (p0 + p2) + (p1 + p3);
    x = x + __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x164, 0xf, 0xf, true));   // row_xmask:4
    x = x + __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x162, 0xf, 0xf, true));   // row_xmask:2
    x = x + __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x161, 0xf, 0xf, true));   // row_xmask:1
    return x;
}
template <int WAVES, int B>
__device__ __forceinline__ void gdn_step_g8(const float * qk, const float * v, int ldv, const float * gates, float * state,
                                            const float * rec, int mprev, float * out, float * snap, long snap_stride,
                                            const GdnSnaps & sp, int H, int nk, int T) {
    constexpr int S = GDN_S, CPW = GDN_CPW, CPT = WAVES * CPW, NT = 32 * WAVES;
    constexpr int NKQ = 2 * B * 32, PT = (NKQ + NT - 1) / NT, NV = B * CPT / 4;   // float4s of k + q / of v per block
    static_assert(NV <= NT && B <= NT, "one v / gate load per thread");
    __shared__ float4 Lkq[2][2 * B][32];                           // [buffer][token * 2 + (0: k, 1: q)][float4 of the row]
    __shared__ float Lv[2][B][CPT];                                // [buffer][token][column of the workgroup]
    __shared__ float2 Lg[2][B];                                    // [buffer][token]: gate, beta_sig of the head
    const int tid = threadIdx.x, lane = tid % 32, wave = __builtin_amdgcn_readfirstlane(tid / 32), grp = lane / 8, j = lane % 8;
    const int h = blockIdx.x / (S / CPT), cw = (blockIdx.x % (S / CPT)) * CPT, cl = wave * CPW + grp, col = cw + cl, kh = h % nk;
    const int RS = nk * S + H * S + 2 * H;
    const size_t qks = (size_t)2 * nk * S;
    float4 * M = (float4 *)(state + (size_t)h * S * S);
    float4 m[4];
#pragma unroll
    for (int b = 0; b < 4; b++) m[b] = M[(size_t)col * (S / 4) + j + 8 * b];
#pragma unroll 1
    for (int r = 0; r < mprev; r++) {                              // the decode's lazy commit (OP_GDN_STEP's replay)
        const float * rr = rec + (size_t)r * RS;
        float4 k4[4];
#pragma unroll
        for (int b = 0; b < 4; b++) k4[b] = ((const float4 *)(rr + (size_t)kh * S))[j + 8 * b];
        const float g = expf(rr[nk * S + H * S + h]), beta = rr[nk * S + H * S + H + h];
        const float kv = g8_sum(m, k4), vc = rr[nk * S + (size_t)h * S + col];
#pragma unroll
        for (int b = 0; b < 4; b++) m[b] = gdn_upd(m[b], k4[b], kv, vc, g, beta);
    }
    const int ns = min(max(sp.n, 0), GDN_MAX_SNAP);
    int si = 0, snext = ns > 0 ? sp.pos[0] : -1;
    const float scale = 1.0f / sqrtf((float)S);
    float4 rkq[PT], rv; float2 rg;
    auto load = [&](int b) {                                       // block b's inputs into registers (zeros past T)
        const int t0 = b * B;
#pragma unroll
        for (int p = 0; p < PT; p++) {
            const int e = tid + p * NT, tok = e / 64, kind = (e / 32) % 2, l = e % 32;
            rkq[p] = (e < NKQ && t0 + tok < T) ? ((const float4 *)(qk + (size_t)(t0 + tok) * qks + (size_t)(kind ? kh : nk + kh) * S))[l]
                                               : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        const int tok = tid / (CPT / 4), c4 = tid % (CPT / 4);
        rv = (tid < NV && t0 + tok < T) ? *(const float4 *)(v + (size_t)(t0 + tok) * ldv + (size_t)h * S + cw + 4 * c4)
                                        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        rg = (tid < B && t0 + tid < T) ? make_float2(gates[(size_t)(t0 + tid) * 2 * H + h], gates[(size_t)(t0 + tid) * 2 * H + H + h])
                                       : make_float2(0.0f, 0.0f);
    };
    auto stash = [&](int buf) {
#pragma unroll
        for (int p = 0; p < PT; p++) {
            const int e = tid + p * NT;
            if (e < NKQ) Lkq[buf][2 * (e / 64) + (e / 32) % 2][e % 32] = rkq[p];
        }
        if (tid < NV) *(float4 *)&Lv[buf][tid / (CPT / 4)][4 * (tid % (CPT / 4))] = rv;
        if (tid < B) Lg[buf][tid] = rg;
    };
    const int nb = (T + B - 1) / B;
    load(0); stash(0); __syncthreads();
#pragma unroll 1
    for (int b = 0; b < nb; b++) {
        if (b + 1 < nb) load(b + 1);                               // in flight while block b computes
        const int buf = b & 1, n = min(B, T - b * B);
#pragma unroll 1
        for (int i = 0; i < n; i++) {
            const int tt = b * B + i;
            float4 k4[4], q4[4];
#pragma unroll
            for (int c = 0; c < 4; c++) { k4[c] = Lkq[buf][2 * i][j + 8 * c]; q4[c] = Lkq[buf][2 * i + 1][j + 8 * c]; }
            const float vc = Lv[buf][i][cl];
            const float2 g2 = Lg[buf][i];
            const float g = expf(g2.x);
            const float kv = g8_sum(m, k4);
#pragma unroll
            for (int c = 0; c < 4; c++) m[c] = gdn_upd(m[c], k4[c], kv, vc, g, g2.y);
            const float at = g8_sum(m, q4);
            if (j == 0) out[((size_t)tt * H + h) * S + col] = at * scale;
            if (tt + 1 == snext) {                                 // a checkpoint: the state after tt + 1 tokens
                float4 * Sn = (float4 *)(snap + (size_t)si * snap_stride + (size_t)h * S * S);
#pragma unroll
                for (int c = 0; c < 4; c++) Sn[(size_t)col * (S / 4) + j + 8 * c] = m[c];
                si++; snext = si < ns ? sp.pos[si] : -1;
            }
        }
        if (b + 1 < nb) stash(buf ^ 1);
        __syncthreads();
    }
#pragma unroll
    for (int b = 0; b < 4; b++) M[(size_t)col * (S / 4) + j + 8 * b] = m[b];
}

// pf_gdn_step_direct: the first version (kept for the A/B and as one more geometry of G2a): no LDS, one token of
// register prefetch per wave (see above for why it is latency-bound)
template <int WAVES>
__device__ __forceinline__ void gdn_step_direct(const float * __restrict__ qk, const float * __restrict__ v, int ldv,
                                                const float * __restrict__ gates, float * state, const float * __restrict__ rec,
                                                int mprev, float * out, float * snap, long snap_stride, const GdnSnaps & sp,
                                                int H, int nk, int T) {
    constexpr int S = GDN_S, CPW = GDN_CPW, CPT = WAVES * CPW;
    const int lane = threadIdx.x % 32, wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 32);
    const int h = blockIdx.x / (S / CPT), c0 = (blockIdx.x % (S / CPT)) * CPT + wave * CPW, kh = h % nk;
    float4 m[CPW];
    gdn_begin(m, (const float4 *)(state + (size_t)h * S * S), rec, mprev, c0, lane, h, kh, H, nk);
    const int ns = min(max(sp.n, 0), GDN_MAX_SNAP);
    int si = 0, snext = ns > 0 ? sp.pos[0] : -1;
    const size_t qks = (size_t)2 * nk * S;
    float4 k4n = ((const float4 *)(qk + (size_t)(nk + kh) * S))[lane];
    float4 q4n = ((const float4 *)(qk + (size_t)kh * S))[lane];
    float4 vn = *(const float4 *)(v + (size_t)h * S + c0);
    float gn = gates[h], bn = gates[H + h];
#pragma unroll 1
    for (int tt = 0; tt < T; tt++) {
        const float4 k4 = k4n, q4 = q4n, vv = vn;
        const float ga = gn, beta = bn;
        if (tt + 1 < T) {
            const float * qn = qk + (size_t)(tt + 1) * qks;
            k4n = ((const float4 *)(qn + (size_t)(nk + kh) * S))[lane];
            q4n = ((const float4 *)(qn + (size_t)kh * S))[lane];
            vn = *(const float4 *)(v + (size_t)(tt + 1) * ldv + (size_t)h * S + c0);
            gn = gates[(size_t)(tt + 1) * 2 * H + h]; bn = gates[(size_t)(tt + 1) * 2 * H + H + h];
        }
        float at[CPW];
        gdn_token(m, k4, q4, vv, ga, beta, at);
        gdn_emit(m, at, out, snap, snap_stride, sp, ns, si, snext, tt, c0, lane, h, H);
    }
    float4 * M = (float4 *)(state + (size_t)h * S * S);
#pragma unroll
    for (int j = 0; j < CPW; j++) M[(size_t)(c0 + j) * (S / 4) + lane] = m[j];
}

#define GDN_ARGS const float * qk, const float * v, int ldv, const float * gates, float * state, const float * rec, int mprev, \
                 float * out, float * snap, long snap_stride, GdnSnaps sp, int H, int nk, int T
#define GDN_PASS qk, v, ldv, gates, state, rec, mprev, out, snap, snap_stride, sp, H, nk, T
extern "C" __global__ void __launch_bounds__(256) pf_gdn_step_w8b16(GDN_ARGS) { gdn_step_lds<8, 16>(GDN_PASS); }
extern "C" __global__ void __launch_bounds__(64) pf_gdn_step_direct(GDN_ARGS) { gdn_step_direct<2>(GDN_PASS); }
extern "C" __global__ void __launch_bounds__(256) pf_gdn_step_g8(GDN_ARGS) { gdn_step_g8<8, 16>(GDN_PASS); }
extern "C" __global__ void __launch_bounds__(128) pf_gdn_step_g8w4(GDN_ARGS) { gdn_step_g8<4, 16>(GDN_PASS); }
extern "C" __global__ void __launch_bounds__(256) pf_gdn_step_g8b8(GDN_ARGS) { gdn_step_g8<8, 8>(GDN_PASS); }

// ---- pf_gdn_norm: OP_GDN_NORM over T tokens, one workgroup of 256 per (token, value head) ----------------------------
// o / z / y [T * H, K] (z: the attn_gate projection's output rows), w [K]; eps = the decode's Seg.rs
extern "C" __global__ void __launch_bounds__(256) pf_gdn_norm(const float * o, const float * w, const float * z, float * y, int K, float eps) {
#pragma clang fp contract(off)
    __shared__ float part[8];
    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32, t = blockIdx.x;
    const float * orow = o + (size_t)t * K, * zr = z + (size_t)t * K;
    float * yr = y + (size_t)t * K;
    float ss = 0.0f;
    for (int i = tid; i < K; i += 256) ss += orow[i] * orow[i];
    ss = wsum(ss); if (lane == 0) part[wave] = ss; __syncthreads();
    float tot = 0.0f; for (int q = 0; q < 8; q++) tot += part[q];
    const float scale = rsqrtf(tot / K + eps);
    for (int i = tid; i < K; i += 256) {
        const float nrm = (scale * orow[i]) * w[i];
        const float sgm = 1.0f / (1.0f + expf(-zr[i]));
        yr[i] = nrm * sgm;
    }
}
