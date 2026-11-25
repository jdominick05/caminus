// OP_OWN_ATTS: qwen3moe decode attention split over positions (flash decoding) and query heads, for speed at depth.
// OP_OWN_ATTN runs one workgroup per query head over every position: 32 of the ~288 workgroups work, each re-reads its kv
// head's K / V rows (8 query heads read the same bytes), and each position costs a serial chain of LDS-path and load
// latencies: 2.75 us per position per token (XTX), 66 tok/s at 4K. Here a task is (split s, kv head g, head group hs):
// one read of the split's K / V rows serves the GH query heads of the group, the splits run on as many workgroups as the
// depth needs, and the last task of (g, hs) to finish merges the splits' partial softmax states in split order (no
// second segment, no extra hop). Shallow positions use few splits and small head groups (8 x 4 tasks of one head, as
// OP_OWN_ATTN: in the model one task of 8 heads took 13.8 us against 6.7 for one head, XT trace at position ~34; 10.3 us
// since its wave sums moved to DPP and its own-position scores run side by side, F16 cache, positions 9 .. 18); deep
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
//   Seg.y = fp16 out [heads][128];  N = heads, K = 128, out_f32 = n_rot, work = splits S (1 .. ATS_MAXS), pad1 = kv heads,
//   pad2 = Lmin (the fewest positions per split), rows_per_task = GH (1, 2, 4, 8), rs = the score scale.
//   Tasks: t = (s * kv_heads + g) * (8 / GH) + hs; heads g * 8 + hs * GH .. + GH - 1.
// The split layout of npos = pos + 1 positions: L = max(Lmin, ceil(npos / S) rounded up to whole chunks of ATS_CH = 64)
// positions per split (a split's last, partial chunk costs a round trip like a whole one: at 16,384 positions on the XTX
// 6.432 ms per token with splits of 288 positions, 6.245 with 320), nact = ceil(npos / L) active splits; split s holds
// [s L, min(npos, (s + 1) L)); the last active split holds pos itself, computes
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
// ATS_MAXS: the most splits per kv head the end-of-task merge takes (it holds every split's m, l in LDS). The two-card
// modules (K3_Q3M_TP: half the kv heads per card) take 128, so each card's grid runs one attention task per workgroup
// (XTX 192 / XT 210 workgroups over 2 kv heads); their merge writes each split's weight over its m ([128][16] floats =
// 8 KB, inside the 8.5 KB of q + scores: the workgroup's LDS is unchanged). Every other module: 64, its merge text as before.
#ifndef ATS_MAXS
#ifdef K3_Q3M_TP
#define ATS_MAXS 128
#else
#define ATS_MAXS 64
#endif
#endif
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
// the wave sums / maxima: wred (kernels/ops_q3m.h, kernels/ops_gg.h: DPP and v_permlanex16, the shuffles' partners in
// their order, the same bits). __shfl_xor compiled to ds_bpermute, an LDS round trip per step: the prologue's five wave
// sums were 25 of them back to back, the softmax's max and sum 10 more per chunk.
// The Q8 KV build keeps the shuffles and __syncthreads (ats_bar below): in dec_q3m_kvq8 the DPP sums measured slower
// (XT decode, ms per token at position 8: 4.972 against 4.928) and the LDS-only barrier alone no different (4.933)
#ifdef K3_Q3M_KV_Q8
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
#else
__device__ __forceinline__ float ats_wsum(float v) { return wred<false, true>(v); }
__device__ __forceinline__ float ats_wmax(float v) { return wred<true, true>(v); }
#endif
// a workgroup barrier for LDS only: waits for this wave's LDS operations (lgkmcnt), not its global loads, and does not
// invalidate L0. __syncthreads() drains every outstanding load first (vmcnt(0)) and invalidates L0: each chunk's V loads
// then completed at the barrier before the softmax instead of during it, and the prologue's barriers serialized too. No
// barrier here orders global memory: the task reads no global data another wave of it wrote (the cache row it stores is
// read back from LDS), and the merge's counter keeps its own __syncthreads and agent fences
__device__ __forceinline__ void ats_bar() {
#ifdef K3_Q3M_KV_Q8
    __syncthreads();
#else
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#endif
}
// fp32 -> fp16 of x without the backend fusing a preceding multiply into v_fma_mix (one rounding instead of two)
__device__ __forceinline__ _Float16 ats_h(float x) { __asm__ volatile("" : "+v"(x)); return (_Float16)x; }

// the split layout of npos positions: L positions per split, nact active splits
__device__ __forceinline__ void ats_layout(int npos, int S, int Lmin, int & L, int & nact) {
    L = (npos + S - 1) / S; L = (L + ATS_CH - 1) / ATS_CH * ATS_CH; L = max(L, Lmin);
    nact = (npos + L - 1) / L;
}

#if defined(K3_Q3M_YARN) && defined(K3_Q3M_VER)
#error "YaRN is built for the decode modules only (one card, two cards, F16 and Q8), not the verify module"
#endif
#ifdef K3_Q3M_KV_Q8
// The Q8_0 KV build (kernels/objects.tsv dec_q3m_kvq8): the cache rows are kernels/kv_codec_ops.h's wire format ([K blocks |
// V blocks], 32 values per block: a half scale and 32 signed codes, 34 bytes). K is rotated: a row is
// caminus.ref_kv.encode_q8_kv(hadamard128(K), V) of the fp16-rounded K / V the F16 object stores (kernels/kv_rot128.h),
// and q gets the same transform after its RoPE, so the scores are q . k in exact arithmetic. V is not rotated (llama.cpp
// also rotates V in 64s, which then needs the output rotated back; a CPU float64 model of this model put that at mean KLD
// 1.87e-3 against 2.09e-3 for K and Q alone, inside the draws' spread, and K's encode was most of the error: 4.49e-3 for K
// alone, 2.15e-3 for V alone). Every other statement of the op is the F16 one; the position's own k / v in LDS are the
// decoded codes (what every later pass reads).
// Decode only: the verify build keeps the F16 cache. The two-card module's Q8 build (kernels/objects.tsv dec_q3m_tp_kvq8) runs
// this text at ATS_MAXS 128 (the V scales below sit in sc[512 .. 768) while the chunks run; the 128 merge reuses q + scores
// only after the last chunk), 33 spilled scalars against dec_q3m_tp's 66. k3_run beside dec_q3m's (96 VGPRs, 0 spilled, 53
// spilled scalars): before the rotation 96, 0, 54 with this op alone, 31 with the module's K3_Q3M_ADD_RMS_PIN
// (kernels/objects.tsv dec_q3m_kvq8; the F16 object with the same pin: 33); with the rotation 31, and 19 with this op's own
// pin of its counts (below). Measured on the way: the V offsets left unpinned per chunk spilled 15 VGPRs; the
// K / V half selects as 64-bit shifts spilled 1; the error word's address kept live across k3_run, 2 more spilled scalars.
#ifdef K3_Q3M_VER
#error "the Q8 KV cache is built for the decode modules only (one card, two cards), not the verify module"
#endif
#include "kv_codec_ops.h"
#include "kv_rot128.h"
__device__ __forceinline__ float ats_h2f(uint32_t u16) { return (float)__builtin_bit_cast(_Float16, (uint16_t)u16); }
__device__ __forceinline__ void own_op_atts(const Seg & sg, int t, Sync * sy) {
#else
__device__ __forceinline__ void own_op_atts(const Seg & sg, int t) {
#endif
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
#ifdef K3_Q3M_VER
    // the verify module (T tokens per pass, kernels/dec_q3m_ver.hip): Seg.pad1 = kv heads | flags << 8 | token << 16.
    //   token: this segment's position is Seg.asum[0] + token (one position word for the pass's block)
    //   flag 1: the cache row only (tasks = kv heads: task g is kv head g of the split that holds the position)
    //   flag 2: no row store (a flag-1 segment stored it: the tokens' attentions then wait for the rows, not each other)
    const int nkv = sg.pad1 & 255, vf = (int)__builtin_amdgcn_readfirstlane((sg.pad1 >> 8) & 255);
    const int S = sg.work, heads = sg.N, M = nkv * ATS_D, HS = ATS_G / GH;
    asm volatile("" : "+s"(t));
    const int pos = (int)__builtin_amdgcn_readfirstlane(((const int *)sg.asum)[0]) + ((sg.pad1 >> 16) & 255), npos = pos + 1;
    int Ls, nact; ats_layout(npos, S, sg.pad2, Ls, nact);
    const int hs = t % HS, sgi = t / HS, s = (vf & 1) ? nact - 1 : sgi / nkv, g = (vf & 1) ? sgi : sgi - s * nkv, h0 = g * ATS_G + hs * GH;
    if (s >= nact || g >= nkv) return;
#else
#ifdef K3_Q3M_KV_Q8
    // (the kv heads, splits and heads through an opaque scalar copy, as OP_OWN_ADD_RMS16's pin: the op derives its values
    //  itself instead of k3_run keeping them live across the segment loop; same statements, same bits. k3_run's spilled
    //  scalars with the rotation below: 31 without this pin (as before the rotation), 19 with it; 96 VGPRs, 0 spilled)
    int nkv = sg.pad1, S = sg.work, heads = sg.N; asm volatile("" : "+s"(nkv), "+s"(S), "+s"(heads));
    const int M = nkv * ATS_D, HS = ATS_G / GH;
#else
    const int nkv = sg.pad1, S = sg.work, heads = sg.N, M = nkv * ATS_D, HS = ATS_G / GH;
#endif
    asm volatile("" : "+s"(t));
    const int hs = t % HS, sgi = t / HS, s = sgi / nkv, g = sgi - s * nkv, h0 = g * ATS_G + hs * GH;   // h0: the first head
    const int pos = (int)__builtin_amdgcn_readfirstlane(((const int *)sg.asum)[0]), npos = pos + 1;
    int Ls, nact; ats_layout(npos, S, sg.pad2, Ls, nact);
    if (s >= nact) return;                       // (uniform: an inactive split)
#endif
    const int a0 = s * Ls, b0 = min(npos, a0 + Ls);
    const bool last = s == nact - 1;
    const float * par = (const float *)sg.aux;
    const float * qin = (const float *)sg.a, * kin = qin + (size_t)heads * ATS_D, * vin = kin + M;
#ifdef K3_Q3M_KV_Q8
    const size_t pb = (size_t)(uint32_t)(M / 32 * 68);    // bytes per position: [K | V] Q8_0 blocks (2 M / 32 x 34, a multiple
                                                          // of 4; a 32-bit product: as size_t, 55 spilled scalars, 54 so)
#else
    const size_t pb = (size_t)4 * M;             // bytes per position: [K | V] f16
#endif
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
#ifdef K3_Q3M_YARN
    // YaRN (the long-context mode's modules, kernels/dec_q3m*_yarn.hip): par[2 D + 2 + j] = pair j's frequency, par[2 D
    // + 2 + D / 2] = the attention factor llama.cpp's rope_yarn multiplies cos and sin by (so the stored K rows carry it,
    // as llama.cpp's do). The host fills both (caminus.rope.yarn_table), the prefill's q3p_qk_rope reads the same block:
    // decode and prefill take the same angle and factor with the same statements
    if (tid < ATS_D / 2) {
        const float theta = (float)pos * par[2 * ATS_D + 2 + tid], ms = par[2 * ATS_D + 2 + ATS_D / 2];
        rc[tid] = cosf(theta) * ms; rsn[tid] = sinf(theta) * ms;
    }
#else
    if (tid < ATS_D / 2) {
        const float theta = (float)pos * powf(par[2 * ATS_D], (float)tid);
        rc[tid] = cosf(theta); rsn[tid] = sinf(theta);
    }
#endif
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
#ifdef K3_Q3M_KV_Q8
        // q and k rotated (kernels/kv_rot128.h): lane l of wave w computes RoPE's statements for its dims l, l + 32,
        // l + 64, l + 96 (pairs (l, l + 64) and (l + 32, l + 96): the same expressions as one dim per thread, so the same
        // bits), then the head's Walsh-Hadamard transform, and keeps dim i = 32 (w & 3) + l of the result. No new LDS
        // and no barrier (the op's 15,748 bytes are what 8 workgroups per WGP leave room for)
        const int qw = wave & 3;   // (= i >> 5, as the wave's scalar; written i >> 5, k3_run spilled one scalar more: 32)
        const float c0 = rc[lane], s0 = rsn[lane], c1 = rc[lane + 32], s1 = rsn[lane + 32];
#pragma unroll
        for (int r = 0; r < ATS_G / 2; r++) {
            if (r >= NR) break;
            if (2 * r + hw >= GH) continue;
            const AtsL * xs = sc + r * 2 * ATS_D + hw * ATS_D;
            const float a0 = xs[lane], a1 = xs[lane + 32], a2 = xs[lane + 64], a3 = xs[lane + 96];
            const float y = own_kv_rot128(__builtin_fmaf(a0, c0, -(a2 * s0)), __builtin_fmaf(a1, c1, -(a3 * s1)),
                                          __builtin_fmaf(a2, c0, a0 * s0), __builtin_fmaf(a3, c1, a1 * s1), qw, lane);
            qs[(2 * r + hw) * 4 * ATS_QP + qw * ATS_QP + lane] = y;
        }
#else
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
#endif
        if (last) {
#ifdef K3_Q3M_KV_Q8
            // wave w: block w & 3 of kv head g, K (waves 0 .. 3) or V (4 .. 7); lane l = value 32 (w & 3) + l. K: the F16
            // object's stored halves (fp16 of RoPE's values), rotated as q is; V: its stored half, unrotated. The hs == 0
            // task stores the row, every task of the split keeps the decoded values for its own position
            float xr;
            if (hw == 0) {
                const float a0 = tk[lane], a1 = tk[lane + 32], a2 = tk[lane + 64], a3 = tk[lane + 96];
                xr = own_kv_rot128(own_kv_f16r(__builtin_fmaf(a0, c0, -(a2 * s0))), own_kv_f16r(__builtin_fmaf(a1, c1, -(a3 * s1))),
                                   own_kv_f16r(__builtin_fmaf(a2, c0, a0 * s0)), own_kv_f16r(__builtin_fmaf(a3, c1, a1 * s1)), qw, lane);
            } else xr = (float)(_Float16)xk;
            uint8_t * blk = (uint8_t *)sg.W + (size_t)pos * pb + (size_t)(hw * (M / 32) + g * (ATS_D / 32) + (i >> 5)) * 34;
            Sync * se = sy; asm volatile("" : "+s"(se));   // (the error word's address made here, not kept live across k3_run)
            const float dq = own_kv_q8_encode_wave(xr, blk, lane, hs == 0, (int *)(se + 1));
            if (hw == 0) kk[i] = dq; else vv[i] = dq;
#else
            _Float16 * cw = (_Float16 *)((uint8_t *)sg.W + (size_t)pos * pb);
            if (hw == 0) {
                const float x0 = tk[j], x1 = tk[j + ATS_D / 2];
                const float y = i < ATS_D / 2 ? __builtin_fmaf(x0, c, -(x1 * sn)) : __builtin_fmaf(x1, c, x0 * sn);
#ifdef K3_Q3M_VER
                const _Float16 kh = (_Float16)y; kk[i] = (float)kh; if (hs == 0 && !(vf & 2)) cw[g * ATS_D + i] = kh;
            } else { const _Float16 vh = (_Float16)xk; vv[i] = (float)vh; if (hs == 0 && !(vf & 2)) cw[M + g * ATS_D + i] = vh; }
#else
                const _Float16 kh = (_Float16)y; kk[i] = (float)kh; if (hs == 0) cw[g * ATS_D + i] = kh;
            } else { const _Float16 vh = (_Float16)xk; vv[i] = (float)vh; if (hs == 0) cw[M + g * ATS_D + i] = vh; }
#endif
#endif
        }
    }
    ats_bar();
#ifdef K3_Q3M_VER
    if (vf & 1) return;                          // (the row only)
#endif
    // -- the chunks of ATS_CH = 64 positions
    AtsG * cache = (AtsG *)sg.W;
    const int qd = lane >> 2, qq = lane & 3, tp = wave * 8 + qd, dp = tid & 63, pq = tid >> 6;   // (tp: this quad's position)
#ifdef K3_Q3M_KV_Q8
    // Q8 rows: a row (pb) and a kv head's 4 blocks (g x 136) are whole dwords, so block b starts 2 (b & 1) bytes past a
    // dword. Quad lane qq reads its K block (dims 32 qq .. 32 qq + 31) as the 9 dwords from the one at or before it; thread
    // (dp, pq) reads the dword holding its V code pair (dims 2 dp, 2 dp + 1 of block dp / 16) per position, the half it sits
    // in fixed per thread; the chunk's V scales (64 positions x 4 blocks) go to LDS once, thread s: position s / 4, block s % 4
    const uint32_t KB = (uint32_t)(M / 32 * 34);  // bytes of a row's K blocks (a multiple of 4)
#else
    const uint32_t koff = (uint32_t)((g * ATS_D + qq * 32) * 2), voff = (uint32_t)((M + g * ATS_D + 2 * dp) * 2);
#endif
    float run_m = -INFINITY, run_s = 0.0f;       // (wave h: head h's, the same in every lane)
    float acc[ATS_G][2];
#pragma unroll
    for (int h = 0; h < ATS_G; h++) { acc[h][0] = 0.0f; acc[h][1] = 0.0f; }
#pragma unroll 1
    for (int c0 = a0; c0 < b0; c0 += ATS_CH) {
        const int n = min(ATS_CH, b0 - c0), nc = (last && c0 + n == npos) ? n - 1 : n;   // nc: positions from the cache
        // loads past nc (up to row c0 + 63 <= pos + 63) read the cache's ATS_PAD padding rows or stale rows; their scores
        // are never stored and their V values are selected away (the host bounds pos to the cache's rows - ATS_PAD)
#ifdef K3_Q3M_KV_Q8
        AtsG * cb = cache + (uint32_t)(c0 * (int)pb);   // (a row offset: < 4 GB at any max_ctx a card holds)
#else
        AtsG * cb = cache + (size_t)c0 * pb;     // (scalar: the chunk's first row)
#endif
#ifdef K3_Q3M_KV_Q8
        int t2 = tid; asm volatile("" : "+v"(t2));   // (the per-thread offsets derived in the chunk, as kl / vl below)
        const uint32_t kl = (uint32_t)(tp * (int)pb) + ((uint32_t)((g * 4 + (t2 & 3)) * 34) & ~3u);
        const uint32_t sl = (uint32_t)((t2 >> 2) * (int)pb) + ((KB + (uint32_t)((g * 4 + (t2 & 3)) * 34)) & ~3u);
        uint32_t kw[9];
#pragma unroll
        for (int j = 0; j < 9; j++) kw[j] = *(const __attribute__((address_space(1))) uint32_t *)(cb + (kl + 4u * j));
        const uint32_t sw = *(const __attribute__((address_space(1))) uint32_t *)(cb + sl);
#else
        uint32_t kl = (uint32_t)(tp * (int)pb) + koff, vl = (uint32_t)(pq * (int)pb) + voff;
        asm volatile("" : "+v"(kl), "+v"(vl));   // (per chunk: the 16 V offsets below hoisted out of the loop were 16 live VGPRs)
        uint4 kr[4];
#pragma unroll
        for (int j = 0; j < 4; j++) kr[j] = *(const __attribute__((address_space(1))) uint4 *)(cb + (kl + 16u * j));
#endif
        // (the loads stay at the chunk's start. Under the old 96-VGPR budget, the first chunk's K loaded before the
        //  prologue, or the next chunk's at the end of this one, spilled k3_run 13 / 395 VGPRs. Under the 144 cap of 5
        //  workgroups per WGP: the first chunk's K before the prologue with the next chunk's after the scores took 109
        //  VGPRs, 0 spilled, but 4 more spilled scalars, and was no faster (XT decode, ms per token: -0.005 at position
        //  8, +0.016 at 4,096, +0.10 at 16,384); the next chunk's at the end of this one still spilled 112 VGPRs)
        __builtin_amdgcn_sched_barrier(0);
        {
            float a[ATS_G];
#pragma unroll
            for (int h = 0; h < ATS_G; h++) a[h] = 0.0f;
            // dims in groups of 8 (one 16-byte load): the group's K converted once, then each head's q slice from LDS
#ifdef K3_Q3M_KV_Q8
            const uint32_t ko = (uint32_t)(qq & 1) * 16u;   // (block qq starts in kw[0]'s high half when qq is odd)
            const float kd = ats_h2f((kw[0] >> ko) & 0xffffu);
#endif
#pragma unroll
            for (int j = 0; j < 4; j++) {
#ifdef K3_Q3M_KV_Q8
                // codes 8 j .. 8 j + 7: kw[2 j + 1 ..] (odd block) or two bytes later; k = float code x float scale (kv_get)
                const uint32_t c0 = ko ? kw[2 * j + 1] : __builtin_amdgcn_alignbyte(kw[2 * j + 1], kw[2 * j], 2);
                const uint32_t c1 = ko ? kw[2 * j + 2] : __builtin_amdgcn_alignbyte(kw[2 * j + 2], kw[2 * j + 1], 2);
                float kf[8];
#pragma unroll
                for (int e = 0; e < 4; e++) { kf[e] = (float)(int8_t)(c0 >> (8 * e)) * kd; kf[4 + e] = (float)(int8_t)(c1 >> (8 * e)) * kd; }
#else
                const h2_t k0 = as_h2(kr[j].x), k1 = as_h2(kr[j].y), k2 = as_h2(kr[j].z), k3 = as_h2(kr[j].w);
                const float kf[8] = {(float)k0[0], (float)k0[1], (float)k1[0], (float)k1[1], (float)k2[0], (float)k2[1], (float)k3[0], (float)k3[1]};
#endif
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
#ifdef K3_Q3M_KV_Q8
            // (the Q8 module keeps the heads one after another in quad 0: the F16 form below spilled one more scalar in
            //  k3_run there, 19 -> 20, in every form tried, with or without the DPP sums: a head per wave, a pinned head
            //  index, a head per quad)
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
#else
            // the position's own k, from LDS: quad h of wave 0 takes head h (lane qq dims 32 qq .. 32 qq + 31 in order, then
            // the quad's sum: each head's statements as when quad 0 took the 8 heads one after another). That serial loop
            // held wave 0, and the barrier every other wave waits at, for 8 chains of 32 LDS loads and fmas: the task took
            // 12.4 us with it, 10.3 with the heads side by side (XT, positions 9 .. 18, with the DPP wave sums above; a
            // head per wave instead: 10.4)
            if (nc < n && wave == 0 && qd < GH) {
                const int h = qd;
                float v = 0.0f;
                { const AtsL * qh = qs + h * 4 * ATS_QP + qq * ATS_QP; for (int d = 0; d < 32; d++) v = __builtin_fmaf(qh[d], kk[qq * 32 + d], v); }
                v += ats_quad<0xB1>(v); v += ats_quad<0x4E>(v);
                if (qq == 0) sc[nc * ATS_G + h] = v * sg.rs;
            }
#endif
        }
#ifdef K3_Q3M_KV_Q8
        {   // the chunk's V scales into LDS past the scores (sc holds [ATS_CH][8] scores of its 1024 floats)
            int t3 = tid; asm volatile("" : "+v"(t3));
            sc[ATS_CH * ATS_G + t3] = ats_h2f((sw >> ((t3 & 1) * 16)) & 0xffffu);
        }
        uint32_t vl = (uint32_t)(pq * (int)pb) + ((KB + (uint32_t)((g * 4 + (dp >> 4)) * 34 + 2 + 2 * (dp & 15))) & ~3u);
        asm volatile("" : "+v"(vl));             // (as the F16 build's kl / vl: its 16 offsets must not leave the loop)
#endif
        // V loads now, in flight across the barrier and the softmax (issued with K's: under the 96 budget they spilled
        // k3_run, 16 more VGPRs live through the scores; under the 144 cap 97 VGPRs, 0 spilled, the same bits, and slower:
        // XT decode +0.02 ms per token at position 8, +0.11 at 4,096, +0.19 at 16,384)
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
#ifdef K3_Q3M_KV_Q8
                // the code pair sits in the dword's high half when block (dp / 16) + 1 + dp is odd (2 + 2 (dp % 16) bytes
                // into a block that starts 2 (dp / 16 & 1) past a dword); its scale from LDS
                const uint32_t pr = ((((dp >> 4) + 1 + dp) & 1) ? vr[e] >> 16 : vr[e]);
                const float vs = sc[ATS_CH * ATS_G + tq * 4 + (dp >> 4)];
                const float w0 = (float)(int8_t)pr * vs, w1 = (float)(int8_t)(pr >> 8) * vs;
                const int lim = nc - 4 * e;      // (scalar: tq < nc <=> pq < lim)
                const bool own = pq == lim && nc < n, in = pq < lim;   // (vv is unset outside the last split)
                const float v0 = own ? vo0 : in ? w0 : 0.0f, v1 = own ? vo1 : in ? w1 : 0.0f;
#else
                const h2_t v2 = as_h2(vr[e]);
                const int lim = nc - 4 * e;      // (scalar: tq < nc <=> pq < lim)
                const bool own = pq == lim && nc < n, in = pq < lim;   // (vv is unset outside the last split)
                const float v0 = own ? vo0 : in ? (float)v2[0] : 0.0f, v1 = own ? vo1 : in ? (float)v2[1] : 0.0f;
#endif
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
#if ATS_MAXS > 64
    static_assert(16 * ATS_MAXS <= ATS_RC, "the merge's m | l rows fit q + scores");
    AtsL * ml = qs;                              // [nact][16] m | l; split s's weight for head h over its m, ml[s * 16 + h]
#define ATS_WV(s2, h) ml[(s2) * 16 + (h)]       //   (nact <= ATS_MAXS: host-checked; only wave h, lane s2 % 32 reads and writes it)
    for (int j = tid; j < 4 * nact; j += 256) {  // (256 threads: one float4 each per round, 64 splits a round)
        const float4 v = *(const float4 *)(r0 + (size_t)(j >> 2) * ATS_RS + (j & 3) * 4);
        *(__attribute__((address_space(3))) float4 *)(ml + (j >> 2) * 16 + (j & 3) * 4) = v;
    }
#else
    AtsL * ml = qs, * wv = qs + 16 * 64;         // [nact][16] m | l, then [nact][8] weights (nact <= 64: host-checked)
#define ATS_WV(s2, h) wv[(s2) * 8 + (h)]
    if (tid < 4 * nact) {
        const float4 v = *(const float4 *)(r0 + (size_t)(tid >> 2) * ATS_RS + (tid & 3) * 4);
        *(__attribute__((address_space(3))) float4 *)(ml + (tid >> 2) * 16 + (tid & 3) * 4) = v;
    }
#endif
    ats_bar();
    if (wave < GH) {
        const int h = wave;
        float mx = -INFINITY;
        for (int s2 = lane; s2 < nact; s2 += 32) mx = fmaxf(mx, ml[s2 * 16 + h]);
        mx = ats_wmax(mx);
        float l = 0.0f;
        for (int s2 = lane; s2 < nact; s2 += 32) { const float w = expf(ml[s2 * 16 + h] - mx); ATS_WV(s2, h) = w; l = __builtin_fmaf(ml[s2 * 16 + 8 + h], w, l); }
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
                        const float w = ATS_WV(s0 + e, h);
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
#undef ATS_WV
}
