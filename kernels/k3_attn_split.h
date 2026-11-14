// Split attention (flash decoding) for the GSQ-RCO Coder's trunk and QSA modules: OP_ATTN_PART (partial softmax states of
// fixed slices of a row's cell list, many workgroups per row) and OP_ATTN_COMB (their merge per query head). Replaces
// op_attn_dec / op_attn_sel on the Coder's tapes (caminus.k3.Engine._attn_split; python/caminus/flashnext_tp.py): those
// ran one workgroup per query head with a serial walk over every cell (2051 cells x 12 layers: ~15 ms per token at
// depth); here a row's list is cut into splits of ATTN_SS list positions, each split one task for all ATTN_G query heads
// of its kv head (one read of the split's K / V rows), and a second segment merges the splits in ascending order.
// Included by kernels/k3_gguf_ops.h when K3_GSQ (not the MTP block's modules, not the drafter's), and by kernels/k3_q35_ops.h
// for the own dense qwen35 modules (K3_Q35_ATTN_SPLIT: Swift / ThinkingCap, decode, verify, two cards; f16 cache): every code object with
// these ops runs the same source, so the dense tape (trunk module, identity list) and the QSA tape (QSA module, SEL list or
// identity) give the same bits for the same list. Contraction is off and every fused multiply-add is written out, so the
// T = 1 and multi-token modules do the same arithmetic; a row's slices depend only on its own list length (never on the
// grid or T): bitwise the same at T = 1 and T = 4, and run to run. Not bitwise op_attn_dec (another summation order).
//
// The cell list of pass row tt (position pos = Seg.asum[0] + tt), as op_attn_sel's: Seg.gs = nullptr or a row whose count
// covers every visible pool: the identity 0 .. pos; otherwise SEL[tt] (OP_QSA_SELECT's row: [0] count, ids from [4]): the
// cells 4 s .. 4 s + 3 of the selected pools s (ascending), then the tail 4 nv .. pos (nv = (pos + 1) / 4). Device clamps
// (a stale or bailed SEL row must never become a read out of bounds): the count to [0, min(nv, 512)], every pool id to
// [0, nv - 1] (so every cell <= pos, which k3_run's prologue guard keeps inside the cache), the list length to the
// records' capacity (Seg.pad2 splits; the host sizes it for the longest list); an empty list writes zeros (no 0 / 0).
// KV: the cache layouts of kv_get, per position [K | V], M = kv_heads * 256 values each: Seg.work 1 = llama.cpp's q8_0
// (the daily server's; Hadamard-rotated by the tape: q / k over 256, v in blocks of 64, the output rotated back after the
// combine), 0 = f16.
#ifndef QSA_TOPK
#define QSA_TOPK 512
#define QSA_SELW 516
#endif
#ifndef ATTN_SS
#define ATTN_SS 32                    // list positions per split (one LDS stage; = caminus.k3.ATTN_SS_OF[module])
#endif
#ifndef ATTN_CPR
#define ATTN_CPR 2                    // cells per wave per scoring round (1 or 2)
#endif
#ifndef ATTN_VH
#define ATTN_VH 12                    // query heads per V pass (12 or 6: accumulators per thread)
#endif
#ifndef ATTN_G
#define ATTN_G 12                     // query heads per kv head (the Coder: 24 / 2)
#endif
#define ATTN_D 256                    // head dim
#define ATTN_RS 260                   // floats per partial record: m, l, 0, 0, acc[256]
#ifdef K3_Q35_ATTN_SPLIT   // the own dense qwen35 modules (kernels/k3_q35_ops.h): no ggml header, an f16 cache only
// (the host refuses q8_0 there), the scratch in its own LDS; every expression below is the GGUF modules' text, so the
// same cells give the same bits as k3_gguf_q35d's pair (the GGUF modules' code is unchanged by this branch)
#ifdef K3_Q35_ATTN_PACK   // (the packed PART below: every pass row's scores of a split, and each row's cell count)
#define ATTN_PT 4                     // pass rows per packed task (the verify modules' T <= OWN_MT = 4; host-checked)
__shared__ float ATTN_OWN_LDS[ATTN_SS * ATTN_G * ATTN_PT + ATTN_SS + 1 + ATTN_PT];
#else
__shared__ float ATTN_OWN_LDS[ATTN_SS * ATTN_G + ATTN_SS + 1];
#endif
#define ATTN_LDS ((__attribute__((address_space(3))) float *)ATTN_OWN_LDS)   // [ATTN_SS][ATTN_G] scores / p, then cells [ATTN_SS]
#ifdef K3_Q35_KV_Q8
#ifdef K3_Q35_ATTN_PACK
#error "Q8 verify uses the unpacked PART until its packed loads are implemented and gated"
#endif
#define ATTN_KV_TYPE(sg) 1
#else
#define ATTN_KV_TYPE(sg) 0
#endif
#define ATTN_Q8_BYTES 34   // (the q8_0 block: half d + 32 int8; unreachable with ATTN_KV_TYPE 0)
#define ATTN_H2F(p) ((float)*(const _Float16 *)(p))
#define ATTN_H22F(u) make_float2((float)__builtin_bit_cast(_Float16, (uint16_t)((u) & 0xffffu)), (float)__builtin_bit_cast(_Float16, (uint16_t)((u) >> 16)))
#else
#define ATTN_LDS ((__attribute__((address_space(3))) float *)AQ)   // [ATTN_SS][ATTN_G] scores / p, then cells [ATTN_SS]
static_assert(AQ_ALLOC * (int)sizeof(block_q8_1) >= (ATTN_SS * ATTN_G + ATTN_SS + 1) * 4, "OP_ATTN_PART's scratch lives in AQ's storage");
#define ATTN_KV_TYPE(sg) (sg).work
#define ATTN_Q8_BYTES sizeof(block_q8_0)
#define ATTN_H2F(p) __half2float(*(const half *)(p))
#define ATTN_H22F(u) __half22float2(__builtin_bit_cast(__half2, (u)))
#endif
static_assert(ATTN_G % ATTN_VH == 0 && (ATTN_CPR == 1 || ATTN_CPR == 2) && (ATTN_SS == 32 || ATTN_SS == 64 || ATTN_SS == 128), "OP_ATTN_PART's layout");
static_assert(ATTN_D == 256, "one thread per dim of V (256 threads)");

struct AttnList { int n, np4, tail0, nvc; const int * pl; };
// row tt's list: n cells; list position k is cell k < np4 ? 4 clamp(pl[k / 4], 0, nvc) + k % 4 : k - np4 + tail0
__device__ __forceinline__ AttnList attn_list(const Seg & sg, int tt) {
    const int pos = ((const int *)sg.asum)[0] + tt;
    AttnList L; L.n = pos + 1; L.np4 = 0; L.tail0 = 0; L.nvc = 0; L.pl = nullptr;
    const int * sel = (const int *)sg.gs;
    if (sel) {
        const int * row = sel + (size_t)tt * QSA_SELW;
        const int nv = (pos + 1) >> 2;
        const int np = min(max(row[0], 0), min(nv, (int)QSA_TOPK));
        if (np != nv) { L.np4 = 4 * np; L.tail0 = 4 * nv; L.nvc = nv - 1; L.pl = row + 4; L.n = L.np4 + (pos + 1 - L.tail0); }
    }
    L.n = min(L.n, sg.pad2 * ATTN_SS);
    return L;
}
__device__ __forceinline__ int attn_cell(const AttnList & L, int k) {
    return k < L.np4 ? 4 * min(max(L.pl[k >> 2], 0), L.nvc) + (k & 3) : k - L.np4 + L.tail0;
}
// lane ^ 16 (permlanex16) and lane ^ m within rows of 16 (DPP row_xmask), as wred's steps
__device__ __forceinline__ float attn_x16(float x) {
    const unsigned u = __builtin_bit_cast(unsigned, x);
    return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(u, u, 0x76543210u, 0xfedcba98u, false, false));
}
template <int M> __device__ __forceinline__ float attn_xm(float x) {
    const int xi = __builtin_bit_cast(int, x);
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(xi, xi, 0x160 | M, 0xf, 0xf, false));
}

// OP_ATTN_PART: task t = (split s, row tt, kv head g), s major (t = s * T * nkv + tt * nkv + g: a row's first splits are
//   claimed first): the split's list positions [s * ATTN_SS, + ATTN_SS) (a task past the row's list does nothing).
//   The split's cells into LDS; scores: wave w takes cells 4 w .. 4 w + 3 in rounds of ATTN_CPR, lane l dims 8 l .. 8 l + 7
//   (q8_0: block l / 4; k = (float) qs * d as kv_get), per query head h the 8-term dot q . k (a product, then fmaf in dim
//   order), summed over the wave by a butterfly (lane ^ 16 (with 2 cells: transposing, each half keeps one), then ^ 8, 4,
//   2, 1), score = dot * rs; softmax: wave w owns heads w and w + 8 (lanes = cells): m = the max, p = expf(s - m), l = the
//   sum (wred); V: thread d, for the heads in groups of ATTN_VH: acc[h] = fmaf(p[h][c], v[c][d], acc[h]) over the cells in
//   order. Record (Seg.y + ((tt * heads + h) * NS + s) * ATTN_RS): m, l, 0, 0, acc[0 .. 255]. Seg.a = q fp32
//   [T][heads][256], Seg.W = the cache (Seg.work: 1 q8_0, 0 f16), Seg.gs = SEL or nullptr, Seg.asum = int32 pos [1],
//   N = T * heads, K = 256, pad1 = kv heads, pad2 = NS (splits per row and head in the records), rs = scale.
// the task's indices (split s, row tt, kv head g) and its record base (head h: + h * NS * ATTN_RS floats). Each phase
// below recomputes them from an opaque copy of t (and reads the cell count from LDS) instead of keeping them live across
// the phases: k3_run sits at its 96-VGPR cap, and in the multi-token trunk module values live from one phase into the next
// turned its SGPR spills from VGPR lanes into scratch memory (~450 reloads; T = 4 pass +5 ms)
struct AttnTask { int T, nkv, heads, NS, s, tt, g; };
__device__ __forceinline__ AttnTask attn_task(const Seg & sg, int t) {
    asm volatile("" : "+s"(t));
    AttnTask a; a.T = seg_T(sg); a.nkv = sg.pad1; a.heads = sg.N / a.T; a.NS = sg.pad2;
    a.s = t / (a.T * a.nkv); const int r = t - a.s * (a.T * a.nkv); a.tt = r / a.nkv; a.g = r - a.tt * a.nkv;
    return a;
}
__device__ __forceinline__ float * attn_rec(const Seg & sg, const AttnTask & a) {
    return (float *)sg.y + ((size_t)(a.tt * a.heads + a.g * ATTN_G) * a.NS + a.s) * ATTN_RS;
}
__device__ __forceinline__ void op_attn_part(const Seg & sg, int t) {
#pragma clang fp contract(off)
    __attribute__((address_space(3))) float * sp = ATTN_LDS;                            // [ATTN_SS][ATTN_G]
    __attribute__((address_space(3))) int * cl = (__attribute__((address_space(3))) int *)(ATTN_LDS + ATTN_SS * ATTN_G);   // [ATTN_SS] + nb
    {
        const AttnTask a = attn_task(sg, t);
        const AttnList L = attn_list(sg, a.tt);
        const int k0 = a.s * ATTN_SS;
        if (k0 >= L.n || a.heads != a.nkv * ATTN_G || sg.K != ATTN_D) return;   // (uniform; the claim's barrier protects the LDS)
        const int nb = min(ATTN_SS, L.n - k0), tid = K3TID;
        __syncthreads();
        if (tid < nb) cl[tid] = attn_cell(L, k0 + tid);
        if (tid == 0) cl[ATTN_SS] = nb;
        __syncthreads();
    }
    {   // scores
        const AttnTask a = attn_task(sg, t);
        const int tt = a.tt, g = a.g, heads = a.heads, type = ATTN_KV_TYPE(sg), tid = K3TID, lane = tid % 32, wave = K3WAVE;   // type 1: q8_0, 0: f16
        const int nb = __builtin_amdgcn_readfirstlane(cl[ATTN_SS]);
        const size_t pb = kv_pos_bytes(a.nkv * ATTN_D, type);
        const float * qb = (const float *)sg.a + ((size_t)tt * heads + (size_t)g * ATTN_G) * ATTN_D + 8 * lane;
        // this lane's K bytes in a position's row (+ cell * pb): q8_0 its block and qs offset, f16 its 8 halves
        const uint8_t * kb = sg.W + (type ? (size_t)(g * 8 + (lane >> 2)) * ATTN_Q8_BYTES : (size_t)(g * ATTN_D + 8 * lane) * 2);
        const int ko = 2 + (lane & 3) * 8, ci = ATTN_CPR == 2 ? (lane >> 4) & 1 : 0;
#if ATTN_SS == 32
#pragma unroll 1
        for (int cb = 4 * wave; cb < min(nb, 4 * wave + 4); cb += ATTN_CPR) {
#else   // (the qwen3next / qwen35moe modules' longer splits) wave w scores cells (SS / 8) w .. (SS / 8) (w + 1) - 1
        constexpr int ATTN_CW = ATTN_SS / 8;
#pragma unroll 1
        for (int cb = ATTN_CW * wave; cb < min(nb, ATTN_CW * wave + ATTN_CW); cb += ATTN_CPR) {
#endif
            float kf[ATTN_CPR][8];
#pragma unroll
            for (int i = 0; i < ATTN_CPR; i++) {
                if (cb + i < nb && type) {
                    const uint8_t * kp = kb + (size_t)cl[cb + i] * pb;
                    const float d = ATTN_H2F(kp);
                    const uint16_t * qs = (const uint16_t *)(kp + ko);
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const uint16_t w = qs[j];
                        kf[i][2 * j] = (float)(int8_t)(w & 0xff) * d; kf[i][2 * j + 1] = (float)(int8_t)(w >> 8) * d;
                    }
                } else if (cb + i < nb) {
                    const uint4 w = *(const uint4 *)(kb + (size_t)cl[cb + i] * pb);
                    const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const float2 kk = ATTN_H22F(wv[j]);
                        kf[i][2 * j] = kk.x; kf[i][2 * j + 1] = kk.y;
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 8; j++) kf[i][j] = 0.0f;
                }
            }
#pragma unroll 1
            for (int h = 0; h < ATTN_G; h++) {
                const float4 qa = *(const float4 *)(qb + h * ATTN_D), qc = *(const float4 *)(qb + h * ATTN_D + 4);
                const float qv[8] = {qa.x, qa.y, qa.z, qa.w, qc.x, qc.y, qc.z, qc.w};
                float a[ATTN_CPR];
#pragma unroll
                for (int i = 0; i < ATTN_CPR; i++) {
                    float x = qv[0] * kf[i][0];
#pragma unroll
                    for (int j = 1; j < 8; j++) x = __builtin_fmaf(qv[j], kf[i][j], x);
                    a[i] = x;
                }
#if ATTN_CPR == 2
                const bool h16 = lane & 16;
                float u = (h16 ? a[1] : a[0]) + attn_x16(h16 ? a[0] : a[1]);
#else
                float u = a[0] + attn_x16(a[0]);
#endif
                u = u + attn_xm<8>(u); u = u + attn_xm<4>(u); u = u + attn_xm<2>(u); u = u + attn_xm<1>(u);
                if ((lane & 15) == 0 && cb + ci < nb) sp[(cb + ci) * ATTN_G + h] = u * sg.rs;
            }
        }
    }
    __syncthreads();
    {   // softmax (wave w: heads w, w + 8): p into LDS, m and l into the records
    const AttnTask a = attn_task(sg, t);
    float * rec = attn_rec(sg, a);
    const int nb = __builtin_amdgcn_readfirstlane(cl[ATTN_SS]), lane = K3TID % 32, wave = K3WAVE, NS = a.NS;
#if ATTN_SS == 32
#pragma unroll 1
    for (int h = wave; h < ATTN_G; h += 8) {
        const float v = lane < nb ? sp[lane * ATTN_G + h] : -INFINITY;
        const float m = wred<true>(v);
        const float p = lane < nb ? expf(v - m) : 0.0f;
        if (lane < nb) sp[lane * ATTN_G + h] = p;
        const float l = wred<false>(p);
        if (lane == 0) { float * rh = rec + (size_t)h * NS * ATTN_RS; rh[0] = m; rh[1] = l; }
    }
#else   // SS / 32 cells per lane (lane + 32 i): the lane's max / sum, then the wave's (fixed order: deterministic)
    constexpr int ATTN_R = ATTN_SS / 32;
#pragma unroll 1
    for (int h = wave; h < ATTN_G; h += 8) {
        float v[ATTN_R], mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < ATTN_R; i++) { const int c = lane + 32 * i; v[i] = c < nb ? sp[c * ATTN_G + h] : -INFINITY; mx = fmaxf(mx, v[i]); }
        const float m = wred<true>(mx);
        float ps = 0.0f;
#pragma unroll
        for (int i = 0; i < ATTN_R; i++) {
            const int c = lane + 32 * i;
            const float p = c < nb ? expf(v[i] - m) : 0.0f;
            if (c < nb) sp[c * ATTN_G + h] = p;
            ps = ps + p;
        }
        const float l = wred<false>(ps);
        if (lane == 0) { float * rh = rec + (size_t)h * NS * ATTN_RS; rh[0] = m; rh[1] = l; }
    }
#endif
    }
    __syncthreads();
    {   // V: this thread's value of each cell (q8_0: its block and qs byte; f16: its half)
        const AttnTask a = attn_task(sg, t);
        float * rec = attn_rec(sg, a);
        const int nb = __builtin_amdgcn_readfirstlane(cl[ATTN_SS]), tid = K3TID, g = a.g, nkv = a.nkv, NS = a.NS, type = ATTN_KV_TYPE(sg);
        const size_t pb = kv_pos_bytes(nkv * ATTN_D, type);
        const uint8_t * vb = sg.W + (type ? (size_t)(nkv * 8 + ((g * ATTN_D + tid) >> 5)) * ATTN_Q8_BYTES
                                          : (size_t)(nkv * ATTN_D + g * ATTN_D + tid) * 2);
        const int vo = 2 + (tid & 31);
#pragma unroll 1
        for (int h0 = 0; h0 < ATTN_G; h0 += ATTN_VH) {
            float acc[ATTN_VH];
#pragma unroll
            for (int h = 0; h < ATTN_VH; h++) acc[h] = 0.0f;
#pragma unroll 1
            for (int c = 0; c < nb; c++) {
                const uint8_t * vp = vb + (size_t)cl[c] * pb;
                const float v = type ? (float)(int8_t)vp[vo] * ATTN_H2F(vp) : ATTN_H2F(vp);
#pragma unroll
                for (int h = 0; h < ATTN_VH; h++) acc[h] = __builtin_fmaf(sp[c * ATTN_G + h0 + h], v, acc[h]);
            }
#pragma unroll
            for (int h = 0; h < ATTN_VH; h++) rec[(size_t)(h0 + h) * NS * ATTN_RS + 4 + tid] = acc[h];
        }
    }
}

#ifdef K3_Q35_ATTN_PACK
// OP_ATTN_PART, packed (the dense qwen35 verify modules: K3_Q35_ATTN_PACK): task t = (split s, kv head g) = s * nkv + g
//   for ALL the pass's T rows: the split's cells staged once, each cell's K loaded once for every row's scores, each
//   cell's V value loaded once for every row's sums. Row tt's split s holds its list positions [s * ATTN_SS, + ATTN_SS)
//   below its own list length (pos + tt + 1: the identity list only, host-checked: no SEL rows); a row whose list ends
//   before the split takes no part in it and writes no record, as op_attn_part's task for it returns. Every row's score,
//   softmax and V sum are op_attn_part's statements in its order on the same cells (scores: the same 8-term dot and
//   butterfly per (cell, head); softmax: lanes = its own nb cells; V: its fmas over its own cells in ascending order), so
//   each record holds op_attn_part's bits: T = 4 rows bitwise one-row passes, bitwise the unpacked pair. What it saves:
//   T - 1 of every split's staging, barriers, claims and K / V loads. LDS: scores [ATTN_SS][T][ATTN_G], then the cells,
//   the longest row's count, each row's count.
__device__ __forceinline__ void op_attn_part_packed(const Seg & sg, int t) {
#pragma clang fp contract(off)
    static_assert(ATTN_SS == 32 && ATTN_VH == ATTN_G, "the packed PART: one stage of 32 cells, every head's V in one pass");
    __attribute__((address_space(3))) float * sp = ATTN_LDS;                                       // [ATTN_SS][T][ATTN_G]
    __attribute__((address_space(3))) int * cl = (__attribute__((address_space(3))) int *)(ATTN_LDS + ATTN_SS * ATTN_G * ATTN_PT);   // [ATTN_SS], nb, nbr[ATTN_PT]
    {
        asm volatile("" : "+s"(t));
        const int T = seg_T(sg), nkv = sg.pad1, s = t / nkv, pos = ((const int *)sg.asum)[0];
        const int k0 = s * ATTN_SS, nmax = min(pos + T, sg.pad2 * ATTN_SS);   // (the longest row's list: row T - 1's)
        if (k0 >= nmax || sg.N != T * nkv * ATTN_G || sg.K != ATTN_D || T > ATTN_PT || sg.gs) return;   // (uniform)
        const int nb = min(ATTN_SS, nmax - k0), tid = K3TID;
        __syncthreads();
        if (tid < nb) cl[tid] = k0 + tid;                                 // (the identity list: list position = cell)
        if (tid == 0) cl[ATTN_SS] = nb;
        if (tid < T) cl[ATTN_SS + 1 + tid] = min(ATTN_SS, max(0, min(pos + tid + 1, sg.pad2 * ATTN_SS) - k0));
        __syncthreads();
    }
    {   // scores: op_attn_part's, the cell's K loaded once for every row
        asm volatile("" : "+s"(t));
        const int T = seg_T(sg), nkv = sg.pad1, g = t - (t / nkv) * nkv, heads = nkv * ATTN_G, TG = T * ATTN_G;
        const int tid = K3TID, lane = tid % 32, wave = K3WAVE;
        const int nb = __builtin_amdgcn_readfirstlane(cl[ATTN_SS]);
        const size_t pb = kv_pos_bytes(nkv * ATTN_D, 0);
        const uint8_t * kb = sg.W + (size_t)(g * ATTN_D + 8 * lane) * 2;
        const int ci = ATTN_CPR == 2 ? (lane >> 4) & 1 : 0;
#pragma unroll 1
        for (int cb = 4 * wave; cb < min(nb, 4 * wave + 4); cb += ATTN_CPR) {
            float kf[ATTN_CPR][8];
#pragma unroll
            for (int i = 0; i < ATTN_CPR; i++) {
                if (cb + i < nb) {
                    const uint4 w = *(const uint4 *)(kb + (size_t)cl[cb + i] * pb);
                    const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const float2 kk = ATTN_H22F(wv[j]);
                        kf[i][2 * j] = kk.x; kf[i][2 * j + 1] = kk.y;
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 8; j++) kf[i][j] = 0.0f;
                }
            }
#pragma unroll 1
            for (int tt = 0; tt < T; tt++) {
                const int nbr = __builtin_amdgcn_readfirstlane(cl[ATTN_SS + 1 + tt]);
                if (cb >= nbr) continue;                                     // (wave-uniform: no cell of this round is row tt's)
                const float * qb = (const float *)sg.a + ((size_t)tt * heads + (size_t)g * ATTN_G) * ATTN_D + 8 * lane;
#pragma unroll 1
                for (int h = 0; h < ATTN_G; h++) {
                    const float4 qa = *(const float4 *)(qb + h * ATTN_D), qc = *(const float4 *)(qb + h * ATTN_D + 4);
                    const float qv[8] = {qa.x, qa.y, qa.z, qa.w, qc.x, qc.y, qc.z, qc.w};
                    float a[ATTN_CPR];
#pragma unroll
                    for (int i = 0; i < ATTN_CPR; i++) {
                        float x = qv[0] * kf[i][0];
#pragma unroll
                        for (int j = 1; j < 8; j++) x = __builtin_fmaf(qv[j], kf[i][j], x);
                        a[i] = x;
                    }
#if ATTN_CPR == 2
                    const bool h16 = lane & 16;
                    float u = (h16 ? a[1] : a[0]) + attn_x16(h16 ? a[0] : a[1]);
#else
                    float u = a[0] + attn_x16(a[0]);
#endif
                    u = u + attn_xm<8>(u); u = u + attn_xm<4>(u); u = u + attn_xm<2>(u); u = u + attn_xm<1>(u);
                    if ((lane & 15) == 0 && cb + ci < nbr) sp[(cb + ci) * TG + tt * ATTN_G + h] = u * sg.rs;
                }
            }
        }
    }
    __syncthreads();
    {   // softmax: (row, head) pairs w, w + 8, ...: lanes = the row's own cells; m and l into the row's records
        asm volatile("" : "+s"(t));
        const int T = seg_T(sg), nkv = sg.pad1, s = t / nkv, g = t - s * nkv, heads = nkv * ATTN_G, TG = T * ATTN_G, NS = sg.pad2;
        const int lane = K3TID % 32, wave = K3WAVE;
#pragma unroll 1
        for (int ph = wave; ph < TG; ph += 8) {
            const int tt = ph / ATTN_G, h = ph - tt * ATTN_G;
            const int nb = __builtin_amdgcn_readfirstlane(cl[ATTN_SS + 1 + tt]);
            if (nb <= 0) continue;
            const float v = lane < nb ? sp[lane * TG + ph] : -INFINITY;
            const float m = wred<true>(v);
            const float p = lane < nb ? expf(v - m) : 0.0f;
            if (lane < nb) sp[lane * TG + ph] = p;
            const float l = wred<false>(p);
            if (lane == 0) {
                float * rh = (float *)sg.y + ((size_t)(tt * heads + g * ATTN_G + h) * NS + s) * ATTN_RS;
                rh[0] = m; rh[1] = l;
            }
        }
    }
    __syncthreads();
    {   // V: thread d, the cell's value loaded once; each row's heads summed over its own cells in ascending order
        asm volatile("" : "+s"(t));
        const int T = seg_T(sg), nkv = sg.pad1, s = t / nkv, g = t - s * nkv, heads = nkv * ATTN_G, TG = T * ATTN_G, NS = sg.pad2;
        const int tid = K3TID;
        const size_t pb = kv_pos_bytes(nkv * ATTN_D, 0);
        const uint8_t * vb = sg.W + (size_t)(nkv * ATTN_D + g * ATTN_D + tid) * 2;
#pragma unroll 1
        for (int tt = 0; tt < T; tt++) {
            const int nbr = __builtin_amdgcn_readfirstlane(cl[ATTN_SS + 1 + tt]);
            if (nbr <= 0) continue;
            float acc[ATTN_G];
#pragma unroll
            for (int h = 0; h < ATTN_G; h++) acc[h] = 0.0f;
#pragma unroll 1
            for (int c = 0; c < nbr; c++) {
                const float v = ATTN_H2F(vb + (size_t)cl[c] * pb);
#pragma unroll
                for (int h = 0; h < ATTN_G; h++) acc[h] = __builtin_fmaf(sp[c * TG + tt * ATTN_G + h], v, acc[h]);
            }
            float * rec = (float *)sg.y + ((size_t)(tt * heads + g * ATTN_G) * NS + s) * ATTN_RS;
#pragma unroll
            for (int h = 0; h < ATTN_G; h++) rec[(size_t)h * NS * ATTN_RS + 4 + tid] = acc[h];
        }
    }
}
#endif

#ifdef K3_Q35_ATTN_PACK
#define ATTN_PART_FN op_attn_part_packed
#else
#define ATTN_PART_FN op_attn_part
#endif

// OP_ATTN_COMB: task q = (row tt, query head h) = tt * heads + h; thread d: over the row's splits u in ascending order
//   (ns = ceil(n / ATTN_SS) of its list, as OP_ATTN_PART cut it): m = max m_u, f_u = expf(m_u - m), l = sum l_u f_u,
//   acc = sum acc_u[d] f_u (in u order); out[q][d] = acc * (1 / l). An empty list writes zeros. Seg.a = the records,
//   Seg.y = out fp32 [T][heads][256], Seg.gs / asum / N / K / pad1 / pad2 as OP_ATTN_PART's.
__device__ __forceinline__ void op_attn_comb(const Seg & sg, int q) {
#pragma clang fp contract(off)
    const int T = seg_T(sg), heads = sg.N / T, tt = q / heads, NS = sg.pad2, tid = K3TID;
    const AttnList L = attn_list(sg, tt);
    float * out = (float *)sg.y + (size_t)q * ATTN_D;
    if (L.n <= 0 || sg.K != ATTN_D) { out[tid] = 0.0f; return; }
    const int ns = (L.n + ATTN_SS - 1) / ATTN_SS;
    const float * rec = (const float *)sg.a + (size_t)q * NS * ATTN_RS;
#ifdef K3_ATTN_COMB_FAST
    // (the qwen3next / qwen35moe modules) the same m, l and acc as the serial walk below, with more loads in flight: at
    // 32K cells the walk was 76% of the pair (1,024 splits, 16 heads = 16 workgroups, one dependent record read at a time;
    // tools/attn_bench.py --trace, XT: COMB 463 of 613 us). m: the whole workgroup's max over the splits (fmaxf over
    // finite values is exact and order-free: the serial max's value); then the sums in u order, the next COMB_U splits'
    // record words loaded before any of them is used (the same additions in the same order: the same bits)
    __shared__ float comb_m[8];
    {
        const int lane = tid % 32, wave = tid / 32;
        float mm = -INFINITY;
        for (int u = tid; u < ns; u += 256) mm = fmaxf(mm, rec[(size_t)u * ATTN_RS]);
        mm = wred<true>(mm);
        if (lane == 0) comb_m[wave] = mm;
        __syncthreads();
    }
    float m = comb_m[0];
#pragma unroll
    for (int w = 1; w < 8; w++) m = fmaxf(m, comb_m[w]);
    __syncthreads();
    constexpr int COMB_U = 8;
    float l = 0.0f, acc = 0.0f;
    int u = 0;
#pragma unroll 1
    for (; u + COMB_U <= ns; u += COMB_U) {
        float r0[COMB_U], r1[COMB_U], ra[COMB_U];
#pragma unroll
        for (int j = 0; j < COMB_U; j++) {
            const float * ru = rec + (size_t)(u + j) * ATTN_RS;
            r0[j] = ru[0]; r1[j] = ru[1]; ra[j] = ru[4 + tid];
        }
#pragma unroll
        for (int j = 0; j < COMB_U; j++) {
            const float f = expf(r0[j] - m);
            l = l + r1[j] * f;
            acc = acc + ra[j] * f;
        }
    }
#pragma unroll 1
    for (; u < ns; u++) {
        const float * ru = rec + (size_t)u * ATTN_RS;
        const float f = expf(ru[0] - m);
        l = l + ru[1] * f;
        acc = acc + ru[4 + tid] * f;
    }
#else
    float m = rec[0];
    for (int u = 1; u < ns; u++) m = fmaxf(m, rec[(size_t)u * ATTN_RS]);
    float l = 0.0f, acc = 0.0f;
#pragma unroll 1
    for (int u = 0; u < ns; u++) {
        const float * ru = rec + (size_t)u * ATTN_RS;
        const float f = expf(ru[0] - m);
        l = l + ru[1] * f;
        acc = acc + ru[4 + tid] * f;
    }
#endif
    out[tid] = acc * (1.0f / l);
}
