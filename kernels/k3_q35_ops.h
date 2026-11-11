// The dense qwen35 line's decode ops without llama.cpp's headers (kernels/k3_q35_swift.hip: Swift 1.5 Q4_K_M, a
// Qwen3.8-27B finetune; python/caminus/engine/q35_dense.py). The K3 core (kernels/k3.hip) plus exactly the ops of that
// engine's tape:
//   the matvecs      OP_OWN_MV's one-round-trip tiles (Q4_K, Q6_K planes; the feed-forward's gate | up fused when the two
//                    share a format) and OP_OWN_MV16 below (Q8_0, Q5_K: own_quant.h's general rows tiles), all on fp16
//                    activations; OP_F32_MV (the Gated DeltaNet's alpha / beta, F32 [48, 5120])
//   the glue         OP_OWN_EMB; OP_OWN_ADD_RMS16 at any row length (below: this model's is 5120, the qwen3moe form
//                    stops at 2048); OP_OWN_SWIGLU16 (a block whose gate and up differ in format)
//   Gated DeltaNet   OP_GDN_CONV / L2 / GATES / STEP / SAVE and OP_GDN_NORM16: the engine's own ops, lifted from
//                    kernels/k3_gguf_ops.h (which includes ggml-cuda's headers) statement for statement, as
//                    kernels/k3_q3m_ops.h lifted OP_F32_MV; value head h reads key head h % nk (tiled), the SiLU output gate
//   attention        OP_QK_NR, OP_KV_APPEND (f16 cache) and OP_ATTN_DEC (k3_own_ops.h's batched form), then
//                    OP_GATE_MUL16; head dim 256, 6 query heads per kv head, RoPE on the first 64 dims
// The three ops whose output is fp16 here where the GGUF engine's writes fp32 (+ q8_1 codes) carry numbers of their own:
// a tape built with the other engine's methods cannot run them by mistake. The wave reductions are own_quant.h's
// wave_sum (the xor butterfly, partners 16 .. 1), as everywhere in the own modules.
#pragma once
#include <stdint.h>

#define K3_OPS_NO_DISPATCH   // k3_q3m_ops.h's helpers and ops (seg_T, OP_F32_MV, the KV helpers, k3_own_ops.h) without its dispatch
#include "k3_q3m_ops.h"
#ifdef K3_TP   // (the two-card modules k3_q35_*_tp: OP_XCHG, its warm word and claim-time store)
#include "k3_xchg.h"
#endif

enum { OP_OWN_MV16 = 52, OP_GDN_NORM16 = 53, OP_GATE_MUL16 = 54 };
enum { OWN_Q5_K = 5 };   // (the GG_* number of the format, as OWN_Q6_K / OWN_Q4_K / OWN_Q8_0)

__device__ __forceinline__ float q35_silu(float x) { return x / (1.0f + expf(-x)); }

// OP_OWN_MV16: out[r] = W[r] . x for raw GGUF blocks W of N contiguous rows (Seg.work = OWN_Q8_0 / OWN_Q5_K) and an fp16
// activation buffer x [K] (Seg.a); Seg.y = fp32 out [N]. own_quant.h's general rows tiles with Act16: a wave takes
// OWN_R = 4 rows, a task = rows_per_task rows (a multiple of 4), wave w the row groups r0 + 4 w, r0 + 4 (w + 8), ...
// A row's sum does not depend on the grid or the task size. T = 1.
__device__ __forceinline__ void q35_op_mv16(const Seg & sg, int t) {
    const int wave = K3WAVE, lane = K3TID % 32;
    const int r0 = t * sg.rows_per_task, r1 = min(sg.N, r0 + sg.rows_per_task);
    const uint8_t * W = sg.W; const void * x = sg.a; float * y = (float *)sg.y;
    for (int row = r0 + wave * OWN_R; row < r1; row += 8 * OWN_R) {
        if (sg.work == OWN_Q5_K) own_rows_q5_k<Act16>(W, row, r1, sg.K, x, nullptr, nullptr, y, lane);
        else own_rows_q8_0<Act16>(W, row, r1, sg.K, x, nullptr, nullptr, y, lane);
    }
}

// OP_OWN_ADD_RMS16 for any row length N = 256 NI (one task, one workgroup; a thread owns elements tid + 256 j):
//   x[i] += d[i]   (Seg.pad1 = 1; 0: no add)
//   h[i] = (x[i] * rsqrt(sum x^2 / N + eps)) * nw[i];   h16[i] = fp16(h[i])
// Seg.a = d, Seg.y = x (in / out), Seg.W = nw, Seg.asum = h (fp32), Seg.aux = h16, Seg.rs = eps. Two walks of the
// row, four elements per round trip: the add and the squares (a thread's fma chain in element order, the wave sum, the
// eight waves in order), then the norm from x as stored. k3_own_ops.h's form holds a thread's whole row in registers
// (8 elements); at N = 5120 that is 20.
__device__ __forceinline__ void q35_op_add_rms16(const Seg & sg) {
#pragma clang fp contract(off)
    __shared__ float part[8];
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, N = sg.N, NI = N >> 8;
    const float * d = (const float *)sg.a; const float * nw = (const float *)sg.W;
    float * x = (float *)sg.y; float * h = (float *)sg.asum; _Float16 * h16 = (_Float16 *)sg.aux;
    const bool add = sg.pad1 > 0;
    float ss = 0.0f;
#pragma unroll 1
    for (int j0 = 0; j0 < NI; j0 += 4) {
        float xv[4], dv[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int i = tid + 256 * min(j0 + j, NI - 1);
            xv[j] = x[i]; dv[j] = add ? d[i] : 0.0f;
        }
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j0 + j >= NI) continue;
            if (add) { xv[j] = xv[j] + dv[j]; x[tid + 256 * (j0 + j)] = xv[j]; }
            ss = __builtin_fmaf(xv[j], xv[j], ss);
        }
        __builtin_amdgcn_sched_barrier(0);
    }
    ss = wave_sum(ss); if (lane == 0) part[wave] = ss; __syncthreads();
    float tot = 0.0f; for (int q = 0; q < 8; q++) tot += part[q];
    const float scale = rsqrtf(tot / N + sg.rs);
#pragma unroll 1
    for (int j0 = 0; j0 < NI; j0 += 4) {
        float xv[4], wn[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int i = tid + 256 * min(j0 + j, NI - 1);
            xv[j] = x[i]; wn[j] = nw[i];
        }
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j0 + j >= NI) continue;
            const float v = (xv[j] * scale) * wn[j];
            h[tid + 256 * (j0 + j)] = v; h16[tid + 256 * (j0 + j)] = (_Float16)v;
        }
        __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();
}

// ---- Gated DeltaNet decode step (kernels/k3_gguf_ops.h's ops: the same statements) ----
// OP_GDN_CONV: per channel c: in = [st[c][0..2], x[c]], out[c] = silu(sum_j in[j] * w[c][j]), and the state shifts to
//   [in[1..3]]. Seg.a = x [C] (the qkv matvec's output), Seg.W = fp32 w [C, 4], Seg.gs = fp32 state [C, 3] (updated in
//   place), Seg.aux = the previous pass's inputs [T, C], Seg.asum = int32 pass params ([1] = mprev: the previous pass's
//   accepted tokens, committed first), Seg.y = out [C]; N = C; a task = rows_per_task channels.
__device__ __forceinline__ void q35_op_gdn_conv(const Seg & sg, int t) {
    const int c0 = t * sg.rows_per_task, c1 = min(sg.N, c0 + sg.rows_per_task);
    const float * x = (const float *)sg.a, * w = (const float *)sg.W;
    float * st = (float *)sg.gs, * xp = (float *)sg.aux;
    const int C = sg.N, T = seg_T(sg), mprev = ((const int *)sg.asum)[1];
    for (int c = c0 + (int)K3TID; c < c1; c += blockDim.x) {
        GF * stg = (GF *)st + (size_t)c * 3;
        float s0 = stg[0], s1 = stg[1], s2 = stg[2];
        const float4 w4 = *(const __attribute__((address_space(1))) float4 *)(w + (size_t)c * 4);
        const float x0 = ((GF *)x)[c], xp0 = ((GF *)xp)[c];
        __builtin_amdgcn_sched_barrier(0);
        for (int j = 0; j < mprev; j++) { s0 = s1; s1 = s2; s2 = j == 0 ? xp0 : xp[(size_t)j * C + c]; }   // commit the accepted tokens
        GW * stw = (GW *)st + (size_t)c * 3;
        stw[0] = s0; stw[1] = s1; stw[2] = s2;
        const float wv[4] = {w4.x, w4.y, w4.z, w4.w};
        for (int tt = 0; tt < T; tt++) {
            const float xv = tt == 0 ? x0 : x[(size_t)tt * C + c];
            const float in[4] = {s0, s1, s2, xv};
            float sumf = 0.0f;
            for (int j = 0; j < 4; j++) sumf += in[j] * wv[j];
            ((GW *)sg.y)[(size_t)tt * C + c] = q35_silu(sumf);
            s0 = s1; s1 = s2; s2 = xv;
            ((GW *)xp)[(size_t)tt * C + c] = xv;                   // this pass's input, for the next pass's commit
        }
    }
}

// OP_GDN_L2: per head of K values: y = (x * rsqrt(sum(x^2) / K + rs)) * (1 / sqrt(K)), rs = eps / K (so y = x /
//   sqrt(sum(x^2) + eps)). Seg.a = in (token stride pad1, 0: heads * K), Seg.y = out, N = heads; one task per head.
__device__ __forceinline__ void q35_op_gdn_l2(const Seg & sg, int t) {
    __shared__ float part[8];
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, K = sg.K;
    const int heads = sg.N / seg_T(sg), tt = t / heads, hh = t - tt * heads;
    const float * x = (const float *)sg.a + (size_t)tt * (sg.pad1 ? sg.pad1 : heads * K) + (size_t)hh * K;
    float * y = (float *)sg.y + (size_t)t * K;
    float ss = 0.0f;
    for (int i = tid; i < K; i += blockDim.x) ss += x[i] * x[i];
    ss = wave_sum(ss); if (lane == 0) part[wave] = ss; __syncthreads();
    float tot = 0.0f; for (int q = 0; q < 8; q++) tot += part[q];
    const float scale = rsqrtf(tot / K + sg.rs), s2 = 1.0f / sqrtf((float)K);
    for (int i = tid; i < K; i += blockDim.x) y[i] = (scale * x[i]) * s2;
    __syncthreads();
}

// OP_GDN_GATES: gate[h] = softplus(alpha[h] + dt[h]) * a[h]; beta_sig[h] = sigmoid(beta[h]) (separate roundings).
//   Seg.a = alpha [H], Seg.W = fp32 beta [H], Seg.gs = fp32 dt [H], Seg.asum = fp32 ssm_a [H] (cast), Seg.y = out [2H]
//   (gate, then beta_sig); N = H; one task.
__device__ __forceinline__ void q35_op_gdn_gates(const Seg & sg) {
#pragma clang fp contract(off)
    const float * al = (const float *)sg.a, * be = (const float *)sg.W, * dt = sg.gs, * A = (const float *)sg.asum;
    float * y = (float *)sg.y;
    const int H = sg.N;
    for (int q = K3TID; q < seg_T(sg) * H; q += blockDim.x) {   // q = token * H + head; out [T, 2H]
        const int tt = q / H, h = q - tt * H;
        const float ab = al[q] + dt[h];
        const float sp = (ab > 20.0f) ? ab : logf(1.0f + expf(ab));
        y[tt * 2 * H + h] = sp * A[h];
        y[tt * 2 * H + H + h] = 1.0f / (1.0f + expf(-be[q]));
    }
}

// OP_GDN_STEP: the gated delta rule per value head h, key / query head h % nk; state M[h][col][i] (S[i][col] at col * S
//   + i). Per token: kv = sum_i M[col][i] k[i]; delta = (v[col] - g kv) beta; M = g M + k[i] delta; attn[col] = scale
//   sum_i M[col][i] q[i], g = exp(gate[h]), scale = 1 / sqrt(S). Lazy commit: the stored state is the one before the
//   previous pass's accepted tokens; each pass first replays those mprev tokens from their saved inputs (OP_GDN_SAVE),
//   stores the result, then runs its T tokens without storing. The same update code (explicit fmaf) runs for the replay
//   and the new tokens.
//   Seg.a = q|k [T, 2, nk, S] (l2-normed), Seg.W = v [T, ...] (token stride pad2), Seg.gs = gates [T, 2H] (gate,
//   beta_sig), Seg.aux = the layer's block [state H*S*S | saved inputs T x (nk*S + H*S + 2H)], Seg.asum = int32 pass
//   params ([1] = mprev), Seg.y = attn [T, H, S]; N = H, K = S = 128 (host-checked), pad1 = nk. A task = 32 columns of
//   one head (4 per wave); a state column is one float4 per lane.
__device__ __forceinline__ float4 q35_gdn_upd(const float4 m, const float4 k4, float kv, float vcol, float g, float beta) {
    const float t1 = g * kv, t2 = vcol - t1, delta = t2 * beta;
    float4 sv;
    sv.x = fmaf(k4.x, delta, g * m.x); sv.y = fmaf(k4.y, delta, g * m.y);
    sv.z = fmaf(k4.z, delta, g * m.z); sv.w = fmaf(k4.w, delta, g * m.w);
    return sv;
}
__device__ __forceinline__ float q35_gdn_dot(const float4 m, const float4 k4) {
    return fmaf(m.w, k4.w, fmaf(m.z, k4.z, fmaf(m.y, k4.y, m.x * k4.x)));
}
__device__ __forceinline__ void q35_op_gdn_step(const Seg & sg, int t) {
    constexpr int S = 128, CPT = 32, CPW = CPT / 8;
    const int h = t / (S / CPT), c0 = (t % (S / CPT)) * CPT, nk = sg.pad1, kh = h % nk, H = sg.N, T = seg_T(sg);
    const int lane = K3TID % 32, wave = K3WAVE, mprev = ((const int *)sg.asum)[1];
    const int RS = nk * S + H * S + 2 * H;                         // a saved input record per token
    float * base = (float *)sg.aux; const float * rec = base + (size_t)H * S * S;
    float4 * M = (float4 *)(base + (size_t)h * S * S);
    const float scale = 1.0f / sqrtf((float)S);
    float4 m[CPW];
#pragma unroll
    for (int j = 0; j < CPW; j++) m[j] = M[(size_t)(c0 + wave * CPW + j) * (S / 4) + lane];
#pragma unroll 1
    for (int r = 0; r < mprev; r++) {                              // replay the previous pass's accepted tokens
        const float * rr = rec + (size_t)r * RS;
        const float4 k4 = ((const float4 *)(rr + (size_t)kh * S))[lane];
        const float * v = rr + nk * S + (size_t)h * S;
        const float g = expf(rr[nk * S + H * S + h]), beta = rr[nk * S + H * S + H + h];
        float kv[CPW];
#pragma unroll
        for (int j = 0; j < CPW; j++) kv[j] = wave_sum(q35_gdn_dot(m[j], k4));
#pragma unroll
        for (int j = 0; j < CPW; j++) m[j] = q35_gdn_upd(m[j], k4, kv[j], v[c0 + wave * CPW + j], g, beta);
    }
#pragma unroll
    for (int j = 0; j < CPW; j++) M[(size_t)(c0 + wave * CPW + j) * (S / 4) + lane] = m[j];   // the committed state
#pragma unroll 1
    for (int tt = 0; tt < T; tt++) {
        const float * qk = (const float *)sg.a + (size_t)tt * 2 * nk * S;
        const float4 k4 = ((const float4 *)(qk + (size_t)(nk + kh) * S))[lane];
        const float4 q4 = ((const float4 *)(qk + (size_t)kh * S))[lane];
        const float * v = (const float *)sg.W + (size_t)tt * sg.pad2 + (size_t)h * S;
        const float * gt = sg.gs + (size_t)tt * 2 * H;
        const float g = expf(gt[h]), beta = gt[H + h];
        float kv[CPW], at[CPW];
#pragma unroll
        for (int j = 0; j < CPW; j++) kv[j] = wave_sum(q35_gdn_dot(m[j], k4));
#pragma unroll
        for (int j = 0; j < CPW; j++) {
            m[j] = q35_gdn_upd(m[j], k4, kv[j], v[c0 + wave * CPW + j], g, beta);
            at[j] = wave_sum(q35_gdn_dot(m[j], q4));
        }
        if (lane == 0) {
#pragma unroll
            for (int j = 0; j < CPW; j++) ((float *)sg.y)[((size_t)tt * H + h) * S + c0 + wave * CPW + j] = at[j] * scale;
        }
    }
}

// OP_GDN_SAVE: after the step, this pass's step inputs per token (k [nk, S], v [H, S], gates [2H]) into the layer's
//   saved-input records (read by the next pass's commit replay). Same Seg fields as OP_GDN_STEP; tasks = tokens.
__device__ __forceinline__ void q35_op_gdn_save(const Seg & sg, int tt) {
    constexpr int S = 128;
    const int nk = sg.pad1, H = sg.N, RS = nk * S + H * S + 2 * H;
    float * rr = (float *)sg.aux + (size_t)H * S * S + (size_t)tt * RS;
    const float * qk = (const float *)sg.a + (size_t)tt * 2 * nk * S, * v = (const float *)sg.W + (size_t)tt * sg.pad2;
    const float * gt = sg.gs + (size_t)tt * 2 * H;
    for (int i = K3TID; i < nk * S; i += blockDim.x) rr[i] = qk[nk * S + i];
    for (int i = K3TID; i < H * S; i += blockDim.x) rr[nk * S + i] = v[i];
    for (int i = K3TID; i < 2 * H; i += blockDim.x) rr[nk * S + H * S + i] = gt[i];
}

// OP_GDN_NORM16: per value head: y16 = fp16(((o * rsqrt(mean(o^2) + eps)) * w) * silu(z)): OP_GDN_NORM's arithmetic
//   (the SiLU output gate), its result stored as the output matvec's fp16 input. Seg.a = o [H, S], Seg.W = fp32 w [S],
//   Seg.gs = fp32 z [H, S], Seg.y = fp16 out [H, S], N = H, K = S, rs = eps; one task per head.
__device__ __forceinline__ void q35_op_gdn_norm16(const Seg & sg, int t) {
#pragma clang fp contract(off)
    __shared__ float part[8];
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, K = sg.K;
    const float * o = (const float *)sg.a + (size_t)t * K, * w = (const float *)sg.W, * z = sg.gs + (size_t)t * K;
    _Float16 * y = (_Float16 *)sg.y + (size_t)t * K;
    float ss = 0.0f;
    for (int i = tid; i < K; i += blockDim.x) ss += o[i] * o[i];
    ss = wave_sum(ss); if (lane == 0) part[wave] = ss; __syncthreads();
    float tot = 0.0f; for (int q = 0; q < 8; q++) tot += part[q];
    const float scale = rsqrtf(tot / K + sg.rs);
    for (int i = tid; i < K; i += blockDim.x) {
        const float nrm = (scale * o[i]) * w[i];
        const float v = nrm * q35_silu(z[i]);
        y[i] = (_Float16)v;
    }
    __syncthreads();
}

// ---- gated attention ----
// OP_QK_NR: per head: y = rms_norm(x) * w, then RoPE on the first n_rot dims, pairs (j, j + n_rot / 2), theta = pos *
//   theta_scale^j. Seg.a = in base, Seg.W = fp32 norm w [D], Seg.asum = int32 pos [1], Seg.y = out [heads, D]; N =
//   heads, K = D (<= 512), pad1 = input head stride (floats), out_f32 = n_rot, rs = theta_scale, pad2 = eps bits; one
//   task per head.
__device__ __forceinline__ void q35_op_qk_nr(const Seg & sg, int t) {
    __shared__ float part[8]; __shared__ float nrm[512];
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, D = sg.K, nrot = sg.out_f32;
    const float * x = (const float *)sg.a + (size_t)t * sg.pad1, * w = (const float *)sg.W;
    float * y = (float *)sg.y + (size_t)t * D;
    const float eps = __builtin_bit_cast(float, sg.pad2);
    float ss = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) ss += x[i] * x[i];
    ss = wave_sum(ss); if (lane == 0) part[wave] = ss; __syncthreads();
    float tot = 0.0f; for (int q = 0; q < 8; q++) tot += part[q];
    const float scale = rsqrtf(tot / D + eps);
    for (int i = tid; i < D; i += blockDim.x) nrm[i] = (scale * x[i]) * w[i];
    __syncthreads();
    const float pos = (float)(((const int *)sg.asum)[0] + t / (sg.N / seg_T(sg)));   // task t = token * heads + head
    for (int i = tid; i < D; i += blockDim.x) {
        if (i >= nrot) { y[i] = nrm[i]; continue; }
        const int j = i < nrot / 2 ? i : i - nrot / 2;
        const float theta = pos * powf(sg.rs, (float)j);
        const float c = cosf(theta), sn = sinf(theta);
        const float x0 = nrm[j], x1 = nrm[j + nrot / 2];
        y[i] = i < nrot / 2 ? x0 * c - x1 * sn : x0 * sn + x1 * c;
    }
    __syncthreads();
}

// OP_KV_APPEND: cache[pos] = (k, v) as fp16 (round to nearest even). Seg.a = k fp32 [M], Seg.W = fp32 v [M], Seg.y =
//   the cache ([K | V] per position, 2 M halves), Seg.asum = int32 pos [1]; N = M; one task. The f16 cache only
//   (Seg.work = 0, host-checked).
__device__ __forceinline__ void q35_op_kv_append(const Seg & sg, int tt) {
    const int pos = ((const int *)sg.asum)[0] + tt, M = sg.N;   // task tt = token tt: k / v [T, M]
    _Float16 * c = (_Float16 *)sg.y + (size_t)pos * 2 * M;
    for (int i = K3TID; i < M; i += blockDim.x) {
        c[i] = (_Float16)((const float *)sg.a)[(size_t)tt * M + i];
        c[M + i] = (_Float16)((const float *)sg.W)[(size_t)tt * M + i];
    }
}

// OP_GATE_MUL16: y16[h * D + d] = fp16(a[h * D + d] * sigmoid(gate[h * stride + off + d])): OP_GATE_MUL's arithmetic,
//   stored as the output matvec's fp16 input. Seg.a = a fp32, Seg.W = gate fp32, Seg.y = fp16 out; N = values, K = D,
//   pad1 = the gate's head stride, pad2 = its offset in a head; a task = rows_per_task values.
__device__ __forceinline__ void q35_op_gate_mul16(const Seg & sg, int t) {
#pragma clang fp contract(off)
    const int i0 = t * sg.rows_per_task, i1 = min(sg.N, i0 + sg.rows_per_task);
    const float * a = (const float *)sg.a, * gt = (const float *)sg.W;
    for (int i = i0 + (int)K3TID; i < i1; i += blockDim.x) {
        const int h = i / sg.K, d = i - h * sg.K;
        const float sgm = 1.0f / (1.0f + expf(-gt[(size_t)h * sg.pad1 + sg.pad2 + d]));
        const float v = a[i] * sgm;
        ((_Float16 *)sg.y)[i] = (_Float16)v;
    }
}

// OP_GDN_SAVE as the dispatch runs it: a bailed launch (sy->err set) leaves the saved-input records alone, as
// kernels/k3_gguf_ops.h's OP_GDN_SAVE. A bailed launch still runs every task once, its gdn_step tasks replaying the previous
// pass's records into the state, and this task's dependency wait may have given up before they read them: a save then
// overwrites records a replay has not read yet, and the retry (commit 0) cannot repair the state. err == 0 here means
// every wait so far succeeded (a wait sets err before it gives up). Thread 0 decides for the workgroup, so a record is
// written whole or not at all. (Seen with a launch whose bound expired early in the pass, so that every later task ran at
// once: without this its retry differed from the uninterrupted pass, and every later pass with it; tools/gates/forcebail.py.)
__device__ __forceinline__ void q35_gdn_save_guarded(const Seg & sg, int t, Sync * sy) {
    __shared__ int save_ok;
    if (thread0()) save_ok = __hip_atomic_load(&sy->err, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == 0;
    __syncthreads();
    const bool ok = save_ok;
    __syncthreads();
    if (ok) q35_op_gdn_save(sg, t);
}

#ifndef K3_Q35_NO_DISPATCH   // (kernels/k3_q35t_ops.h: these ops under its own dispatch)
__device__ __forceinline__ bool k3_extra_op(const Seg & sg, int t, bool fresh, Sync * sy, int limit_us, int seg,
                                            uint64_t * tr = nullptr) {
    if (sg.op == OP_OWN_MV) { own_op_mv_1rt(sg, t, (int)__builtin_amdgcn_readfirstlane(K3TID / 32), K3TID % 32); return true; }
    if (sg.op == OP_OWN_MV16) { q35_op_mv16(sg, t); return true; }
    if (sg.op == OP_OWN_ADD_RMS16) { q35_op_add_rms16(sg); return true; }
    if (sg.op == OP_F32_MV) { op_f32_mv(sg, t); return true; }
    if (sg.op == OP_GDN_CONV) { q35_op_gdn_conv(sg, t); return true; }
    if (sg.op == OP_GDN_L2) { q35_op_gdn_l2(sg, t); return true; }
    if (sg.op == OP_GDN_GATES) { q35_op_gdn_gates(sg); return true; }
    if (sg.op == OP_GDN_STEP) { q35_op_gdn_step(sg, t); return true; }
    if (sg.op == OP_GDN_SAVE) { q35_gdn_save_guarded(sg, t, sy); return true; }
    if (sg.op == OP_GDN_NORM16) { q35_op_gdn_norm16(sg, t); return true; }
    if (sg.op == OP_QK_NR) { q35_op_qk_nr(sg, t); return true; }
    if (sg.op == OP_KV_APPEND) { q35_op_kv_append(sg, t); return true; }
    if (sg.op == OP_ATTN_DEC) { own_op_attn_dec(sg, t); return true; }
    if (sg.op == OP_GATE_MUL16) { q35_op_gate_mul16(sg, t); return true; }
    if (sg.op == OP_OWN_SWIGLU16) { own_op_swiglu16(sg, t); return true; }
    if (sg.op == OP_OWN_EMB) { own_op_emb(sg); return true; }
#ifdef K3_TP
    if (sg.op == OP_XCHG) { op_xchg(sg, sy, limit_us, seg, tr); return true; }
#endif
    return false;
}
#ifdef K3_TP
__device__ __forceinline__ uint32_t * k3_extra_warm(const Seg & sg) { return xchg_warm(sg); }
__device__ __forceinline__ void k3_extra_prefetch(const Seg & sg, int) { xchg_prefetch(sg); }
#else
__device__ __forceinline__ uint32_t * k3_extra_warm(const Seg &) { return nullptr; }
__device__ __forceinline__ void k3_extra_prefetch(const Seg &, int) {}
#endif
#endif
