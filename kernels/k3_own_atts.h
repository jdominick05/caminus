// OP_OWN_ATTS: qwen3moe decode attention split over positions (flash decoding) and query heads, for speed at depth.
// OP_OWN_ATTN runs one workgroup per query head over every position: 32 of the ~288 workgroups work, each re-reads its kv
// head's K / V rows (8 query heads read the same bytes), and each position costs a serial chain of LDS-path and load
// latencies: 2.75 us per position per token (XTX), 66 tok/s at 4K. Here a task is (split s, kv head g, head group hs):
// one read of the split's K / V rows serves the GH query heads of the group, the splits run on as many workgroups as the
// depth needs, and the last task of (g, hs) to finish merges the splits' partial softmax states in split order (no
// second segment, no extra hop). Shallow positions use few splits and small head groups (8 x 4 tasks of one head, as
// OP_OWN_ATTN: in the model one task of 8 heads took 13.8 us against 6.7 for one head, XT trace at position ~34); deep
// ones many splits of all 8 heads (one read of the rows).
//
// Not bitwise OP_OWN_ATTN (the splits change the summation order); q's norm + RoPE and the cache row ARE its bits (the
// same statements, lanes and wave sums), so the attention's sums are the only difference. A head's arithmetic does not
// depend on GH, and the split layout depends only on the position, S and Lmin (never on the grid or which workgroup runs
// what): the same bits run to run, at any K3_PER_WGP and at any GH; two split counts S1 < S2 give the same bits while
// npos <= S1 * Lmin (both layouts are then L = Lmin). Head dim 128, 8 query heads per kv head, n_rot = 128, f16 cache,
// T = 1 (host-checked).
//
//   Seg.a = q | k | v fp32 (the three matvecs' outputs, one buffer: [heads][128], then [2][kv heads][128])
//   Seg.gs = scratch: uint32 counters [64] (one per (kv head, head group); zero between launches: the merging task resets
//            its own), then records [kv heads * 8 / GH][splits][ATS_RS] (m [8], l [8], acc [8][128]; the first GH used)
//   Seg.aux = fp32 [2 D + 2]: q_norm, k_norm, theta_scale, eps;  Seg.W = the cache;  Seg.asum = int32 pos [1]
//   Seg.y = fp16 out [heads][128];  N = heads, K = 128, out_f32 = n_rot, work = splits S, pad1 = kv heads,
//   pad2 = Lmin (the fewest positions per split), rows_per_task = GH (1, 2, 4, 8), rs = the score scale.
//   Tasks: t = (s * kv_heads + g) * (8 / GH) + hs; heads g * 8 + hs * GH .. + GH - 1.
// The split layout of npos = pos + 1 positions: L = max(Lmin, roundup32(ceil(npos / S))) positions per split, nact =
// ceil(npos / L) active splits; split s holds [s L, min(npos, (s + 1) L)); the last active split holds pos itself, computes
// the position's k / v (norm + RoPE, f16 rounding; its hs == 0 task writes the cache row). Inactive splits do nothing.
// nact == 1: the task writes the output itself. Otherwise each active task writes its record, then counts itself on its
// (g, hs) counter (release); the one that counts nact merges the nact records (acquire) and resets the counter.
//
// Per task, chunks of ATS_CH = 64 positions, each chunk's loads in one round trip:
//   scores  a quad of lanes per position (8 waves x 8 quads = the chunk), lane quarter qq dims 32 qq .. 32 qq + 31 (4 x
//           16-byte loads), q from LDS; per head a fma chain in dim order, then the quad's sum (DPP lane ^ 1, ^ 2) * rs
//   softmax wave h owns head h: the chunk max, p = expf(s - m) (0 past the chunk), the running max / sum (online softmax)
//   V       thread (dims 2 dp, 2 dp + 1; positions pq + 4 e): acc[h][2] = acc * corr, then fma(p, v, acc) in position order
//   merge   the four position quarters ((0 + 2) + (1 + 3)), then the record (or the output)
#pragma once
enum { OP_OWN_ATTS = 51 };
#define ATS_D 128                   // head dim
#define ATS_G 8                     // query heads per kv head
#define ATS_CH 64                   // positions per chunk (one round trip of K and V loads)
#define ATS_QP 36                   // floats per quarter row of q in LDS (32 + 4: the four quarters in distinct banks)
#define ATS_RS (16 + ATS_G * ATS_D) // floats per record: m [8], l [8], acc [8][128]
#define ATS_CNT 64                  // counter words before the records
#define ATS_PAD 64                  // cache rows past the last usable position (= ATS_CH: a chunk's loads run to row c0 + 63)
// LDS (floats): q [8][4][ATS_QP], scores / p [ATS_CH][8] (1024 reserved: q's norms use 4 x 256 of it first), cos / sin
// [64] each, own k / v / unrotated k [128] each, misc [80]. The end-of-task merges reuse q + scores (2176 floats).
enum { ATS_QS = 0, ATS_SC = ATS_G * 4 * ATS_QP, ATS_RC = ATS_SC + 1024, ATS_RSN = ATS_RC + 64, ATS_KK = ATS_RSN + 64,
       ATS_VV = ATS_KK + ATS_D, ATS_TK = ATS_VV + ATS_D, ATS_MISC = ATS_TK + ATS_D, ATS_NLDS = ATS_MISC + 80 };
__shared__ float ATS[ATS_NLDS];
typedef __attribute__((address_space(3))) float AtsL;
typedef __attribute__((address_space(1))) const uint8_t AtsG;

template <int CTRL> __device__ __forceinline__ float ats_quad(float x) {    // a quad_perm DPP move of x
    return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), CTRL, 0xf, 0xf, false));
}
__device__ __forceinline__ float ats_wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}
__device__ __forceinline__ float ats_wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
    return v;
}
// a workgroup barrier for LDS only: waits for this wave's LDS operations (lgkmcnt), not its global loads, and does not
// invalidate L0. __syncthreads() drains every outstanding load first (vmcnt(0)) and invalidates L0: each chunk's V loads
// then completed at the barrier before the softmax instead of during it, and the prologue's barriers serialized too
__device__ __forceinline__ void ats_bar() {
#ifdef ATS_LDS_BAR
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
    __syncthreads();
#endif
}
// fp32 -> fp16 of x without the backend fusing a preceding multiply into v_fma_mix (one rounding instead of two)
__device__ __forceinline__ _Float16 ats_h(float x) { __asm__ volatile("" : "+v"(x)); return (_Float16)x; }

// the split layout of npos positions: L positions per split, nact active splits
__device__ __forceinline__ void ats_layout(int npos, int S, int Lmin, int & L, int & nact) {
    L = (npos + S - 1) / S; L = (L + 31) & ~31; L = max(L, Lmin);
    nact = (npos + L - 1) / L;
}

__device__ __forceinline__ void own_op_atts(const Seg & sg, int t) {
#pragma clang fp contract(off)
    AtsL * L_ = (AtsL *)ATS;
    AtsL * qs = L_ + ATS_QS, * sc = L_ + ATS_SC, * rc = L_ + ATS_RC, * rsn = L_ + ATS_RSN, * kk = L_ + ATS_KK, * vv = L_ + ATS_VV;
    AtsL * tk = L_ + ATS_TK;       // [128] the position's k before its rotation
    AtsL * misc = L_ + ATS_MISC;   // [0, 40) norm partials (5 rounds x 8 waves), [40, 48) chunk corrections,
                                   // [48, 56) run_s, [56, 64) run_m; then the merge's [64, 72) sums, [72] the counter
    // (the thread id through an opaque copy: every lane-derived address of this op (q slices, scores, rows) is then
    //  computed inside the op; left visible, LLVM hoisted them out of k3_run's segment loop)
    int tid = K3TID; asm volatile("" : "+v"(tid));
    const int lane = tid & 31, wave = (int)__builtin_amdgcn_readfirstlane(tid >> 5);
#ifdef ATS_GH_RUNTIME   // (heads per task from Seg.rows_per_task: measured slower, the runtime GH raised k3_run's spilled scalars 53 -> 76)
    const int GH = sg.rows_per_task;
#else
    constexpr int GH = ATS_G;
#endif
    const int nkv = sg.pad1, S = sg.work, heads = sg.N, M = nkv * ATS_D, HS = ATS_G / GH;
    asm volatile("" : "+s"(t));
    const int hs = t % HS, sgi = t / HS, s = sgi / nkv, g = sgi - s * nkv, h0 = g * ATS_G + hs * GH;   // h0: the first head
    const int pos = (int)__builtin_amdgcn_readfirstlane(((const int *)sg.asum)[0]), npos = pos + 1;
    int Ls, nact; ats_layout(npos, S, sg.pad2, Ls, nact);
    if (s >= nact) return;                       // (uniform: an inactive split)
    const int a0 = s * Ls, b0 = min(npos, a0 + Ls);
    const bool last = s == nact - 1;
    const float * par = (const float *)sg.aux;
    const float * qin = (const float *)sg.a, * kin = qin + (size_t)heads * ATS_D, * vin = kin + M;
    const size_t pb = (size_t)4 * M;             // bytes per position: [K | V] f16
    // -- q's norm + RoPE for the GH heads (thread i of each 128-thread half: local heads 2 r + half) and, in the last
    //    split, the position's k (threads 0 .. 127) and v (128 .. 255): every input in one round trip, the rounds' sums
    //    with one barrier. op_qk_nr's statements: each head's sum is its four waves' sums in order, the same bits as
    //    OP_OWN_ATTN's q and cache row.
    const int i = tid & (ATS_D - 1), hw = tid >> 7, NR = (GH + 1) / 2;
    float xq[ATS_G / 2];
#pragma unroll
    for (int r = 0; r < ATS_G / 2; r++) xq[r] = (r < NR && 2 * r + hw < GH) ? qin[(size_t)(h0 + 2 * r + hw) * ATS_D + i] : 0.0f;
    const float wq = par[i], eps = par[2 * ATS_D + 1];
    const float xk = last ? (hw ? vin[(size_t)g * ATS_D + i] : kin[(size_t)g * ATS_D + i]) : 0.0f;
    const float wk = par[ATS_D + i];
    if (tid < ATS_D / 2) {
        const float theta = (float)pos * powf(par[2 * ATS_D], (float)tid);
        rc[tid] = cosf(theta); rsn[tid] = sinf(theta);
    }
#pragma unroll
    for (int r = 0; r < ATS_G / 2; r++) {
        if (r >= NR) break;
        float ss = xq[r] * xq[r]; ss = ats_wsum(ss); if (lane == 0) misc[8 * r + wave] = ss;
    }
    if (last) { float ss = xk * xk; ss = ats_wsum(ss); if (lane == 0) misc[32 + wave] = ss; }
    ats_bar();
#pragma unroll
    for (int r = 0; r < ATS_G / 2; r++) {
        if (r >= NR) break;
        float tot = 0.0f; for (int q = 0; q < 4; q++) tot += misc[8 * r + hw * 4 + q];
        const float scale = rsqrtf(tot / ATS_D + eps);
        sc[r * 2 * ATS_D + hw * ATS_D + i] = (scale * xq[r]) * wq;
    }
    if (last && hw == 0) {
        float tot = 0.0f; for (int q = 0; q < 4; q++) tot += misc[32 + q];
        const float scale = rsqrtf(tot / ATS_D + eps);
        tk[i] = (scale * xk) * wk;
    }
    ats_bar();
    {
        const int j = i < ATS_D / 2 ? i : i - ATS_D / 2;
        const float c = rc[j], sn = rsn[j];
#pragma unroll
        for (int r = 0; r < ATS_G / 2; r++) {
            if (r >= NR) break;
            if (2 * r + hw >= GH) continue;
            const int nb = r * 2 * ATS_D + hw * ATS_D;
            const float x0 = sc[nb + j], x1 = sc[nb + j + ATS_D / 2];
            const float y = i < ATS_D / 2 ? __builtin_fmaf(x0, c, -(x1 * sn)) : __builtin_fmaf(x1, c, x0 * sn);
            qs[(2 * r + hw) * 4 * ATS_QP + (i >> 5) * ATS_QP + (i & 31)] = y;
        }
        if (last) {
            _Float16 * cw = (_Float16 *)((uint8_t *)sg.W + (size_t)pos * pb);
            if (hw == 0) {
                const float x0 = tk[j], x1 = tk[j + ATS_D / 2];
                const float y = i < ATS_D / 2 ? __builtin_fmaf(x0, c, -(x1 * sn)) : __builtin_fmaf(x1, c, x0 * sn);
                const _Float16 kh = (_Float16)y; kk[i] = (float)kh; if (hs == 0) cw[g * ATS_D + i] = kh;
            } else { const _Float16 vh = (_Float16)xk; vv[i] = (float)vh; if (hs == 0) cw[M + g * ATS_D + i] = vh; }
        }
    }
    ats_bar();
    // -- the chunks of ATS_CH = 64 positions
    AtsG * cache = (AtsG *)sg.W;
    const int qd = lane >> 2, qq = lane & 3, tp = wave * 8 + qd, dp = tid & 63, pq = tid >> 6;   // (tp: this quad's position)
    const uint32_t koff = (uint32_t)((g * ATS_D + qq * 32) * 2), voff = (uint32_t)((M + g * ATS_D + 2 * dp) * 2);
    float run_m = -INFINITY, run_s = 0.0f;       // (wave h: head h's, the same in every lane)
    float acc[ATS_G][2];
#pragma unroll
    for (int h = 0; h < ATS_G; h++) { acc[h][0] = 0.0f; acc[h][1] = 0.0f; }
#pragma unroll 1
    for (int c0 = a0; c0 < b0; c0 += ATS_CH) {
        const int n = min(ATS_CH, b0 - c0), nc = (last && c0 + n == npos) ? n - 1 : n;   // nc: positions from the cache
        // loads past nc (up to row c0 + 63 <= pos + 63) read the cache's ATS_PAD padding rows or stale rows; their scores
        // are never stored and their V values are selected away (the host bounds pos to the cache's rows - ATS_PAD)
        AtsG * cb = cache + (size_t)c0 * pb;     // (scalar: the chunk's first row)
        uint32_t kl = (uint32_t)(tp * (int)pb) + koff, vl = (uint32_t)(pq * (int)pb) + voff;
        asm volatile("" : "+v"(kl), "+v"(vl));   // (per chunk: the 16 V offsets below hoisted out of the loop were 16 live VGPRs)
        uint4 kr[4];
#pragma unroll
        for (int j = 0; j < 4; j++) kr[j] = *(const __attribute__((address_space(1))) uint4 *)(cb + (kl + 16u * j));
        // (loading the first chunk's K before the prologue, or the next chunk's at the end of this one, spilled k3_run:
        //  13 / 395 VGPRs: the loads stay at the chunk's start)
        __builtin_amdgcn_sched_barrier(0);
        {
            float a[ATS_G];
#pragma unroll
            for (int h = 0; h < ATS_G; h++) a[h] = 0.0f;
            // dims in groups of 8 (one 16-byte load): the group's K converted once, then each head's q slice from LDS
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const h2_t k0 = as_h2(kr[j].x), k1 = as_h2(kr[j].y), k2 = as_h2(kr[j].z), k3 = as_h2(kr[j].w);
                const float kf[8] = {(float)k0[0], (float)k0[1], (float)k1[0], (float)k1[1], (float)k2[0], (float)k2[1], (float)k3[0], (float)k3[1]};
#pragma unroll
                for (int h = 0; h < ATS_G; h++) {
                    if (h >= GH) break;
                    const AtsL * qh = qs + h * 4 * ATS_QP + qq * ATS_QP + 8 * j;
                    const float4 q0 = *(const __attribute__((address_space(3))) float4 *)(qh);
                    const float4 q1 = *(const __attribute__((address_space(3))) float4 *)(qh + 4);
                    const float qv[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
                    float v = a[h];
#pragma unroll
                    for (int e = 0; e < 8; e++) v = __builtin_fmaf(qv[e], kf[e], v);
                    a[h] = v;
                    __builtin_amdgcn_sched_barrier(0);   // (one head's q slice live at a time: max-ilp hoisted all 8)
                }
            }
#pragma unroll
            for (int h = 0; h < ATS_G; h++) { if (h >= GH) break; a[h] += ats_quad<0xB1>(a[h]); a[h] += ats_quad<0x4E>(a[h]); }
            if (qq == 0 && tp < nc) {
#pragma unroll
                for (int h = 0; h < ATS_G; h++) { if (h >= GH) break; sc[tp * ATS_G + h] = a[h] * sg.rs; }
            }
            if (nc < n && wave == 0) {                   // the position's own k, from LDS (quad 0 of wave 0)
#pragma unroll
                for (int h = 0; h < ATS_G; h++) {
                    if (h >= GH) break;
                    float v = 0.0f;
                    if (qd == 0) { const AtsL * qh = qs + h * 4 * ATS_QP + qq * ATS_QP; for (int d = 0; d < 32; d++) v = __builtin_fmaf(qh[d], kk[qq * 32 + d], v); }
                    v += ats_quad<0xB1>(v); v += ats_quad<0x4E>(v);
                    if (lane == 0) sc[nc * ATS_G + h] = v * sg.rs;
                }
            }
        }
        // V loads now, in flight across the barrier and the softmax (issued with K's they spilled k3_run: 16 more VGPRs
        // live through the scores)
        uint32_t vr[ATS_CH / 4];                 // this thread's positions pq + 4 e (V)
#pragma unroll
        for (int e = 0; e < ATS_CH / 4; e++)
            vr[e] = *(const __attribute__((address_space(1))) uint32_t *)(cb + (vl + (uint32_t)(4 * e) * (uint32_t)pb));
        ats_bar();
        if (wave < GH) {   // softmax: wave h = head h, two positions per lane
            const int h = wave;
            float sv[ATS_CH / 32], m = -INFINITY;
#pragma unroll
            for (int e = 0; e < ATS_CH / 32; e++) { const int tq = lane + 32 * e; sv[e] = tq < n ? sc[tq * ATS_G + h] : -INFINITY; m = fmaxf(m, sv[e]); }
            m = ats_wmax(m);
            const float mn = fmaxf(run_m, m), corr = expf(run_m - mn);
            float sum = 0.0f;
#pragma unroll
            for (int e = 0; e < ATS_CH / 32; e++) {
                const int tq = lane + 32 * e;
                const float ex = tq < n ? expf(sv[e] - mn) : 0.0f;   // (positions past the chunk: p = 0, so the V sums run
                sc[tq * ATS_G + h] = ex; sum += ex;                     //  over all ATS_CH without a branch)
            }
            sum = ats_wsum(sum);
            run_s = __builtin_fmaf(run_s, corr, sum);
            run_m = mn;
            if (lane == 0) misc[40 + h] = corr;
        }
        ats_bar();
        {   // V: dims 2 dp, 2 dp + 1; positions pq + 4 e of the chunk, in order
            float cr[ATS_G];
#pragma unroll
            for (int h = 0; h < ATS_G; h++) { if (h >= GH) break; cr[h] = misc[40 + h]; acc[h][0] *= cr[h]; acc[h][1] *= cr[h]; }
            // (branch-free: p is 0 past the chunk; the position's own v from LDS by a select. A per-lane branch per
            //  position spilled ~300 VGPRs in k3_run)
            const float vo0 = vv[2 * dp], vo1 = vv[2 * dp + 1];
#pragma unroll
            for (int e = 0; e < ATS_CH / 4; e++) {
                const int tq = pq + 4 * e;
                const h2_t v2 = as_h2(vr[e]);
                const int lim = nc - 4 * e;      // (scalar: tq < nc <=> pq < lim)
                const bool own = pq == lim && nc < n, in = pq < lim;   // (vv is unset outside the last split)
                const float v0 = own ? vo0 : in ? (float)v2[0] : 0.0f, v1 = own ? vo1 : in ? (float)v2[1] : 0.0f;
                if (GH > 4) {
                    const float4 p0 = *(const __attribute__((address_space(3))) float4 *)(sc + tq * ATS_G);
                    const float4 p1 = *(const __attribute__((address_space(3))) float4 *)(sc + tq * ATS_G + 4);
                    const float p[ATS_G] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
                    for (int h = 0; h < ATS_G; h++) { acc[h][0] = __builtin_fmaf(p[h], v0, acc[h][0]); acc[h][1] = __builtin_fmaf(p[h], v1, acc[h][1]); }
                } else {
#pragma unroll
                    for (int h = 0; h < 4; h++) {
                        if (h >= GH) break;
                        const float p = sc[tq * ATS_G + h];
                        acc[h][0] = __builtin_fmaf(p, v0, acc[h][0]); acc[h][1] = __builtin_fmaf(p, v1, acc[h][1]);
                    }
                }
                if (e & 1) __builtin_amdgcn_sched_barrier(0);   // (two positions' p at a time)
            }
        }
        ats_bar();
    }
    // -- the four position quarters: (0 + 2) + (1 + 3), in LDS over q + scores
    AtsL * red = qs;
    if (pq >= 2)
#pragma unroll
        for (int h = 0; h < ATS_G; h++) { if (h >= GH) break; red[(pq - 2) * 1024 + h * ATS_D + 2 * dp] = acc[h][0]; red[(pq - 2) * 1024 + h * ATS_D + 2 * dp + 1] = acc[h][1]; }
    ats_bar();
    if (pq < 2)
#pragma unroll
        for (int h = 0; h < ATS_G; h++) { if (h >= GH) break; acc[h][0] += red[pq * 1024 + h * ATS_D + 2 * dp]; acc[h][1] += red[pq * 1024 + h * ATS_D + 2 * dp + 1]; }
    ats_bar();
    if (pq == 1)
#pragma unroll
        for (int h = 0; h < ATS_G; h++) { if (h >= GH) break; red[h * ATS_D + 2 * dp] = acc[h][0]; red[h * ATS_D + 2 * dp + 1] = acc[h][1]; }
    if (lane == 0) { misc[48 + wave] = run_s; misc[56 + wave] = run_m; }
    ats_bar();
    if (nact == 1) {                             // the only split: the output itself
        if (pq == 0)
#pragma unroll
            for (int h = 0; h < ATS_G; h++) {
                if (h >= GH) break;
                const float inv = 1.0f / misc[48 + h];
                const float o0 = (acc[h][0] + red[h * ATS_D + 2 * dp]) * inv, o1 = (acc[h][1] + red[h * ATS_D + 2 * dp + 1]) * inv;
                _Float16 * out = (_Float16 *)sg.y + (size_t)(h0 + h) * ATS_D + 2 * dp;
                out[0] = ats_h(o0); out[1] = ats_h(o1);
            }
        return;
    }
    const int ci = g * HS + hs;                  // the (kv head, head group)'s counter and record row
    float * rec = (float *)sg.gs + ATS_CNT + ((size_t)ci * S + s) * ATS_RS;
    if (pq == 0)
#pragma unroll
        for (int h = 0; h < ATS_G; h++) {
            if (h >= GH) break;
            *(float2 *)(rec + 16 + h * ATS_D + 2 * dp) = make_float2(acc[h][0] + red[h * ATS_D + 2 * dp], acc[h][1] + red[h * ATS_D + 2 * dp + 1]);
        }
    if (tid < GH) { rec[tid] = misc[56 + tid]; rec[8 + tid] = misc[48 + tid]; }
    // -- count; the task that counts nact merges
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    uint32_t * cnt = (uint32_t *)sg.gs + ci;
    if (tid == 0) misc[72] = __builtin_bit_cast(float, __hip_atomic_fetch_add(cnt, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
    ats_bar();
    const uint32_t old = __builtin_amdgcn_readfirstlane(__builtin_bit_cast(uint32_t, misc[72]));
    if (old != (uint32_t)nact - 1) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    // the merge, in split order: every split's m, l into LDS (one round trip), wave h its weights w_s = exp(m_s - M) and
    // the sum l = sum w_s l_s, then thread (h, 4 dims) the weighted sum of the splits' acc, 8 splits per round trip
    const float * r0 = (const float *)sg.gs + ATS_CNT + (size_t)ci * S * ATS_RS;
    AtsL * ml = qs, * wv = qs + 16 * 64;         // [nact][16] m | l, then [nact][8] weights (nact <= 64: host-checked)
    if (tid < 4 * nact) {
        const float4 v = *(const float4 *)(r0 + (size_t)(tid >> 2) * ATS_RS + (tid & 3) * 4);
        *(__attribute__((address_space(3))) float4 *)(ml + (tid >> 2) * 16 + (tid & 3) * 4) = v;
    }
    ats_bar();
    if (wave < GH) {
        const int h = wave;
        float mx = -INFINITY;
        for (int s2 = lane; s2 < nact; s2 += 32) mx = fmaxf(mx, ml[s2 * 16 + h]);
        mx = ats_wmax(mx);
        float l = 0.0f;
        for (int s2 = lane; s2 < nact; s2 += 32) { const float w = expf(ml[s2 * 16 + h] - mx); wv[s2 * 8 + h] = w; l = __builtin_fmaf(ml[s2 * 16 + 8 + h], w, l); }
        l = ats_wsum(l);
        if (lane == 0) misc[64 + h] = l;
    }
    ats_bar();
    {
        const int h = tid >> 5, d4 = (tid & 31) * 4;
        if (h < GH) {
            float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int s0 = 0; s0 < nact; s0 += 8) {   // (8 splits per round trip: 16 spilled k3_run)
                float4 v[8];
#pragma unroll
                for (int e = 0; e < 8; e++) v[e] = *(const float4 *)(r0 + (size_t)min(s0 + e, nact - 1) * ATS_RS + 16 + h * ATS_D + d4);
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    if (s0 + e < nact) {
                        const float w = wv[(s0 + e) * 8 + h];
                        o[0] = __builtin_fmaf(v[e].x, w, o[0]); o[1] = __builtin_fmaf(v[e].y, w, o[1]);
                        o[2] = __builtin_fmaf(v[e].z, w, o[2]); o[3] = __builtin_fmaf(v[e].w, w, o[3]);
                    }
                }
            }
            const float inv = 1.0f / misc[64 + h];
            _Float16 * out = (_Float16 *)sg.y + (size_t)(h0 + h) * ATS_D + d4;
#pragma unroll
            for (int j = 0; j < 4; j++) out[j] = ats_h(o[j] * inv);
        }
    }
    if (tid == 0) __hip_atomic_store(cnt, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
