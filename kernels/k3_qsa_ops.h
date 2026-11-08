// QSA indexer ops for the K3 engine (docs/specs/spec-qsa.md sections 1 and 3, steps S2-S4): the indexer cache append with
// pooling, the pool scores and the top-512 pool selection of qwen4exp's QSA layers. Included by kernels/k3_gguf_ops.h only
// when K3_QSA is defined: kernels/k3_gguf_gsq_qsa.hip / _qsa_mt.hip, the GSQ-RCO Coder's one- / multi-token trunk modules
// plus these ops (separate code objects: inlined into k3_gguf_gsq* they cost the trunk's tapes ~0.7 ms per token). Graph reference: ignite-llama src/models/qwen4exp.cpp build_qsa_sel and
// src/llama-memory-hybrid-idx.cpp (pool layout, visibility, tail); kernels: ggml-cuda lightning-indexer.cu (vec kernel,
// F32 keys), norm.cu (rms_norm_f32<32, 1, vec>), rope.cu (rope_multi), scale.cu, getrows.cu / cpy-utils.cuh (q8_0).
// Every fused multiply-add below is written out (__builtin_fmaf) where llama.cpp's code object has one (its ISA: v_fma /
// v_fmac), and contraction is off everywhere else, so the arithmetic does not depend on what the compiler fuses here.
// Layouts (per QSA layer, per card; q8_0 as the daily server's -ctk q8_0, which the indexer cache follows):
//   IDXRAW  [rows][136 B]: position p's raw indexer key (k_proj output, unnormed, unrotated) as 4 block_q8_0
//   IDXPOOL [rows / 4][136 B]: pool j = positions 4j .. 4j + 3 (pools start at token 0): the mean of the members'
//           dequantized raw keys, RMS-normed (k_norm), RoPE'd (NEOX, n_rot 64) at position 4j, as 4 block_q8_0
//   SCORE   [T][K] fp32: row t, pool j < nv_t: sum_h relu(q_t,h . pooled_j) * w, heads 0..3 in order (w = 1 / sqrt(128))
//   SEL     [T][QSA_SELW] int32: [0] pools selected, [1] tail cells (pos + 1) % 4, [2] nv, [3] the threshold key, then
//           the selected pool ids ascending. A row at position pos sees nv = (pos + 1) / 4 pools (those ending at or before
//           it) and attends to the cells of its selected pools plus the tail pos, pos - 1, .. (n_tail cells).
// Positions come from the tape's position word (Seg.asum, the same int32 the KV ops read and k3_run's prologue guards);
// every position-indexed access is also clamped against the buffer's rows carried in the segment.
enum { OP_IDX_APPEND = 37, OP_QSA_SCORE = 38, OP_QSA_SELECT = 39, OP_ATTN_SEL = 40 };
#define QSA_ROWB 136   // one 128-value row: 4 x block_q8_0 (34 B)
#define QSA_TOPK 512   // pools selected (the indexer's top_k 2048 tokens / kpool 4)
#define QSA_SELW 516   // ints per SEL row: 4 header words + QSA_TOPK pool ids
#define QSA_LDS ((__attribute__((address_space(3))) int *)AQ)   // OP_QSA_SELECT's scratch: 276 ints in AQ's storage
static_assert(AQ_ALLOC * (int)sizeof(block_q8_1) >= 276 * 4, "OP_QSA_SELECT's scratch lives in AQ's storage");

// lane l of a wave holds values 4l .. 4l + 3 of a 128-value row (the float4 layout of llama.cpp's vec indexer and of its
// rms_norm at ncols 128): block l / 8 of the row, bytes (l % 8) * 4 .. + 3 of its qs
__device__ __forceinline__ float4 qsa_ld_row(const uint8_t * row, int lane) {
    const block_q8_0 * b = (const block_q8_0 *)row + lane / 8;
    const int o = (lane % 8) * 4;
    const float d = __half2float(b->d);   // getrows.cu dequantize_q8_0: (float)q * d
    return make_float4((float)b->qs[o] * d, (float)b->qs[o + 1] * d, (float)b->qs[o + 2] * d, (float)b->qs[o + 3] * d);
}
// a lane-held row -> q8_0 as quantize_f32_q8_0_block (cpy-utils.cuh: amax / 127, id = 1 / d, roundf(x * id), d as half),
// the block's amax over its 8 lanes (fmaxf: any order). Writes the row when row != nullptr; returns its dequantized values
__device__ __forceinline__ float4 qsa_q8_row(float4 v, uint8_t * row, int lane) {
#pragma clang fp contract(off)
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    amax = fmaxf(amax, __shfl_xor(amax, 1, 32));
    amax = fmaxf(amax, __shfl_xor(amax, 2, 32));
    amax = fmaxf(amax, __shfl_xor(amax, 4, 32));
    const float d = amax / ((1 << 7) - 1), id = d ? 1.0f / d : 0.0f;
    const half dh = __float2half(d);
    const int8_t q0 = roundf(v.x * id), q1 = roundf(v.y * id), q2 = roundf(v.z * id), q3 = roundf(v.w * id);
    if (row) {
        block_q8_0 * b = (block_q8_0 *)row + lane / 8;
        const int o = (lane % 8) * 4;
        b->qs[o] = q0; b->qs[o + 1] = q1; b->qs[o + 2] = q2; b->qs[o + 3] = q3;
        if (lane % 8 == 0) b->d = dh;
    }
    const float df = __half2float(dh);
    return make_float4((float)q0 * df, (float)q1 * df, (float)q2 * df, (float)q3 * df);
}

// OP_IDX_APPEND (one task per pass token, wave 0): IDXRAW[pos + tt] = q8_0(k_raw[tt]); when position pos + tt ends a pool
//   (its % 4 == 3), IDXPOOL[(pos + tt) / 4] = the pooled key of that pool (llama.cpp's build_qsa_sel order):
//   members m = 4j .. 4j + 3 dequantized from their q8_0 raw rows (getrows: llama.cpp gathers them from its cache), summed
//   ((k0 + k1) + k2) + k3 (ggml_cont + ggml_add), times 0.25 (ggml_scale: fma(0.25, x, 0)), RMSNorm with eps then the
//   k_norm weight (rms_norm_f32<32, 1, vec, mul>: per lane y*y then fma x, z, w, the xor butterfly, / 128, + eps, rsqrt,
//   (scale * x) * w), NEOX RoPE of dims 0..63 at position 4j (rope_multi with equal text positions: theta = 4j *
//   powf(theta_scale, i), x0' = fma(x0, cos, -(x1 sin)), x1' = fma(x1, cos, x0 sin)), then q8_0. A member written by this
//   same pass is re-quantized from its k_raw row here (the bytes its own task writes) instead of read back, so the tasks
//   are independent; members before the pass come from IDXRAW. A pool completed by a rejected draft row is pooled again
//   when its last position is written again, and only rows at or past that position ever see it.
//   Seg.a = k_raw fp32 [T][128], Seg.y = IDXRAW, Seg.aux = IDXPOOL, Seg.W = k_norm fp32 [128], Seg.asum = int32 pos [1],
//   Seg.gs = the pooled keys before quantization fp32 [T][128] (rows that end a pool; tests) or nullptr,
//   N = IDXRAW rows, K = IDXPOOL rows, rs = theta_scale, pad2 = eps bits, out_f32 = n_rot (64).
__device__ __forceinline__ void op_idx_append(const Seg & sg, int tt) {
#pragma clang fp contract(off)
    if (K3WAVE != 0) return;
    const int lane = K3TID % 32, p0 = ((const int *)sg.asum)[0], pos = p0 + tt;
    if (pos < 0 || pos >= sg.N) return;                                   // device clamp (k3_run's prologue guards too)
    const float * kr = (const float *)sg.a;
    uint8_t * raw = (uint8_t *)sg.y;
    const float4 own = qsa_q8_row(((const float4 *)(kr + (size_t)tt * 128))[lane], raw + (size_t)pos * QSA_ROWB, lane);
    if ((pos & 3) != 3 || (pos >> 2) >= sg.K) return;
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 1
    for (int j = 0; j < 3; j++) {                                         // ((m0 + m1) + m2) + own (a rolled loop: code size)
        const int mp = pos - 3 + j, ti = mp - p0;                         // mp >= 0: pos >= 3 here; ti: wave-uniform
        const float4 v = ti >= 0 ? qsa_q8_row(((const float4 *)(kr + (size_t)ti * 128))[lane], nullptr, lane)
                                 : qsa_ld_row(raw + (size_t)mp * QSA_ROWB, lane);
        s = j == 0 ? v : make_float4(s.x + v.x, s.y + v.y, s.z + v.z, s.w + v.w);
    }
    s = make_float4(s.x + own.x, s.y + own.y, s.z + own.z, s.w + own.w);
    s.x = __builtin_fmaf(0.25f, s.x, 0.0f); s.y = __builtin_fmaf(0.25f, s.y, 0.0f);
    s.z = __builtin_fmaf(0.25f, s.z, 0.0f); s.w = __builtin_fmaf(0.25f, s.w, 0.0f);
    float t = s.y * s.y;
    t = __builtin_fmaf(s.x, s.x, t); t = __builtin_fmaf(s.z, s.z, t); t = __builtin_fmaf(s.w, s.w, t);
    t = wsum(t);
    const float eps = __builtin_bit_cast(float, sg.pad2);
    const float mean = t / 128.0f;                                        // exact (power of 2), as / ncols
    const float sc = rsqrtf(mean + eps);
    const float4 nw = ((const float4 *)sg.W)[lane];
    float4 r = make_float4(sc * s.x, sc * s.y, sc * s.z, sc * s.w);
    r.x = r.x * nw.x; r.y = r.y * nw.y; r.z = r.z * nw.z; r.w = r.w * nw.w;
    // RoPE: lanes 0..7 hold dims 0..31 (x0), lanes 8..15 dims 32..63 (x1) of the same pairs; lanes 16..31 pass through
    float4 pr;
    pr.x = __shfl_xor(r.x, 8, 32); pr.y = __shfl_xor(r.y, 8, 32); pr.z = __shfl_xor(r.z, 8, 32); pr.w = __shfl_xor(r.w, 8, 32);
    if (lane < 16) {
        const bool lo = lane < 8;
        const float fp = (float)(pos - 3);
        const float4 x0 = lo ? r : pr, x1 = lo ? pr : r;
        float a0[4] = {x0.x, x0.y, x0.z, x0.w}, a1[4] = {x1.x, x1.y, x1.z, x1.w}, o[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
        for (int c = 0; c < 4; c++) {                                     // rolled (one copy of the trig code); values rotate
            const float theta = fp * powf(sg.rs, (float)(4 * (lane & 7) + c));
            const float ct = cosf(theta), st = sinf(theta);
            const float v = lo ? __builtin_fmaf(a0[0], ct, -(a1[0] * st)) : __builtin_fmaf(a1[0], ct, a0[0] * st);
#pragma unroll
            for (int k = 0; k < 3; k++) { a0[k] = a0[k + 1]; a1[k] = a1[k + 1]; o[k] = o[k + 1]; }
            o[3] = v;
        }
        r = make_float4(o[0], o[1], o[2], o[3]);
    }
    if (sg.gs) ((float4 *)((float *)sg.gs + (size_t)tt * 128))[lane] = r;
    qsa_q8_row(r, (uint8_t *)sg.aux + (size_t)(pos >> 2) * QSA_ROWB, lane);
}

// OP_ATTN_SEL: op_attn_dec over the cells a QSA layer's selection keeps (spec-qsa.md S5): pass row tt (position pos) walks
//   its cell list: the cells 4 s .. 4 s + 3 of its selected pools s = SEL[tt][4 ..] (ascending ids, so ascending cells),
//   then its tail cells 4 nv .. pos (nv = (pos + 1) / 4; empty when (pos + 1) % 4 == 0: llama.cpp's quirk, the own block
//   must win the top-k). List position k plays the role of cell k in op_attn_dec: the same 512-wide chunks in LDS, the
//   same online softmax and the same arithmetic in the same order, only the KV row read changes. With the identity
//   selection (nv <= 512: every pool) the list is 0 .. pos, so the result is op_attn_dec's bit for bit; Seg.gs = nullptr
//   is that identity too (OP_ATTN_DEC in the QSA modules runs this function: one attention body in their k3_run).
//   Seg.gs = SEL int32 [T][QSA_SELW] (OP_QSA_SELECT's rows), the rest as op_attn_dec. A row selecting all nv pools is the
//   identity (cell = list position, no list reads: the cost of op_attn_dec); otherwise each chunk's 512 cells go into LDS
//   first (AQ's storage, as OP_QSA_SELECT's scratch) and both loops read them there. Device clamps (the SEL rows are the
//   device's own, but a bailed or stale row must never become a KV read out of bounds): the pool count to [0, min(nv,
//   512)], every pool id to [0, nv - 1] (so a cell <= pos, which k3_run's prologue guard keeps inside the cache), and an
//   empty list writes zeros (no 0 / 0).
__device__ __forceinline__ void op_attn_sel(const Seg & sg, int h) {
    constexpr int CH = 512;
    __shared__ float sc[CH]; __shared__ float red[8];
    const int heads = sg.N / seg_T(sg), tt = h / heads;          // task = token * heads + head
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, D = sg.K, nkv = sg.pad1, g = (h - tt * heads) / (heads / nkv);
    const int pos = ((const int *)sg.asum)[0] + tt, M = nkv * D, type = sg.work;
    const size_t pb = kv_pos_bytes(M, type);
    const int * sel = (const int *)sg.gs;
    int npos = pos + 1, np4 = 0, tail0 = 0, nvc = 0;              // list: [0, np4) pool cells, then tail0 .. pos
    const int * pl = nullptr;
    bool ident = true;
    if (sel) {
        const int * row = sel + (size_t)tt * QSA_SELW;
        const int nv = (pos + 1) >> 2;
        const int np = min(max(row[0], 0), min(nv, (int)QSA_TOPK));
        ident = np == nv;                                         // every pool: the list is 0 .. pos
        np4 = 4 * np; tail0 = 4 * nv; nvc = nv - 1; pl = row + 4;
        npos = np4 + (pos + 1 - tail0);
        if (npos <= 0) {   // (a garbage count with an empty tail: no cell to attend to)
            for (int j = 0; j < 2; j++) { const int d = tid + j * (int)blockDim.x; if (d < D) ((float *)sg.y)[(size_t)h * D + d] = 0.0f; }
            return;
        }
    }
    __attribute__((address_space(3))) int * cl = QSA_LDS;           // the chunk's cells (non-identity lists)
    static_assert(AQ_ALLOC * (int)sizeof(block_q8_1) >= 512 * 4, "OP_ATTN_SEL's cell list lives in AQ's storage");
    const float * q = (const float *)sg.a + (size_t)h * D;
    float run_m = -INFINITY, run_s = 0.0f, acc[2] = {0.0f, 0.0f};
#pragma unroll 1
    for (int c0 = 0; c0 < npos; c0 += CH) {
        const int n = min(CH, npos - c0);
        if (!ident) {
            for (int k = tid; k < n; k += blockDim.x) {
                const int kk = c0 + k;
                cl[k] = kk < np4 ? 4 * min(max(pl[kk >> 2], 0), nvc) + (kk & 3) : kk - np4 + tail0;
            }
            __syncthreads();
        }
        for (int t = wave; t < n; t += 8) {
            const uint8_t * kp = sg.W + (size_t)(ident ? c0 + t : cl[t]) * pb;   // (one row per wave: cheap either way)
            float a = 0.0f;
            for (int d = lane; d < D; d += 32) a += q[d] * kv_get(kp, M, type, 0, g * D + d);
            a = warp_reduce_sum<32>(a);
            if (lane == 0) sc[t] = a * sg.rs;
        }
        __syncthreads();
        float m = -INFINITY;
        for (int t = tid; t < n; t += blockDim.x) m = fmaxf(m, sc[t]);
        m = warp_reduce_max<32>(m); if (lane == 0) red[wave] = m; __syncthreads();
        m = red[0]; for (int w = 1; w < 8; w++) m = fmaxf(m, red[w]); __syncthreads();
        const float mn = fmaxf(run_m, m), corr = expf(run_m - mn);
        float sum = 0.0f;
        for (int t = tid; t < n; t += blockDim.x) { const float e = expf(sc[t] - mn); sc[t] = e; sum += e; }
        sum = warp_reduce_sum<32>(sum); if (lane == 0) red[wave] = sum; __syncthreads();
        sum = 0.0f; for (int w = 0; w < 8; w++) sum += red[w];
        run_s = c0 == 0 ? sum : run_s * corr + sum;
        for (int j = 0; j < 2; j++) {
            const int d = tid + j * (int)blockDim.x;
            if (d >= D) break;
            float part = 0.0f;
            // (the serial walk over the chunk: two copies, so the identity walk has no list read in its dependent chain)
            if (ident) { const uint8_t * cb = sg.W + (size_t)c0 * pb; for (int t = 0; t < n; t++) part += sc[t] * kv_get(cb + (size_t)t * pb, M, type, 1, g * D + d); }
            else for (int t = 0; t < n; t++) part += sc[t] * kv_get(sg.W + (size_t)cl[t] * pb, M, type, 1, g * D + d);
            acc[j] = c0 == 0 ? part : acc[j] * corr + part;
        }
        run_m = mn;
        __syncthreads();
    }
    const float inv = 1.0f / run_s;
    for (int j = 0; j < 2; j++) {
        const int d = tid + j * (int)blockDim.x;
        if (d < D) ((float *)sg.y)[(size_t)h * D + d] = acc[j] * inv;
    }
}

// OP_QSA_SCORE: SCORE[t][j] for the pools j < nv_t = (pos + t + 1) / 4 of each pass row t, llama.cpp's vec lightning indexer
//   with F32 keys (the dequantized IDXPOOL rows, as its get_rows): per head h, lane l: fma(q.x, k.x, 0), then fma .y, .z, .w
//   (dims 4l .. 4l + 3), the xor butterfly (wsum = warp_reduce_sum's partners and order), relu, then s = fma(sum, w, s)
//   from 0 over heads 0..3; + the visible mask (0) is exact. The same per-row order whatever T. Pools past nv_t are not
//   written (the selection never reads them). Skipped (no task does anything) while the last row's nv <= pad1 (512: the
//   selection is the identity there).
//   A task = chunks of 32 pools (8 waves x 4) t, t + ntask, ...; the 4 pools' keys are loaded once per chunk for all rows.
//   Seg.a = q fp32 [T][4][128] (normed, RoPE'd), Seg.W = IDXPOOL, Seg.y = SCORE fp32 [T][K], Seg.asum = int32 pos [1],
//   N = IDXPOOL rows, K = SCORE row stride (>= N), rs = w, pad1 = the skip threshold on nv.
__device__ __forceinline__ void op_qsa_score(const Seg & sg, int t) {
#pragma clang fp contract(off)
    const int T = seg_T(sg), p0 = ((const int *)sg.asum)[0], lane = K3TID % 32, wave = K3WAVE;
    const int nvmax = min((p0 + T) >> 2, sg.N);
    if (p0 < 0 || nvmax <= sg.pad1) return;
    const float w = sg.rs;
    const uint8_t * pool = (const uint8_t *)sg.W;
    float * out = (float *)sg.y;
    for (int c0 = t * 32; c0 < nvmax; c0 += sg.ntask * 32) {
        const int pb = c0 + wave * 4;
        if (pb >= nvmax) continue;
        float4 k[4];
#pragma unroll
        for (int i = 0; i < 4; i++) k[i] = pb + i < nvmax ? qsa_ld_row(pool + (size_t)(pb + i) * QSA_ROWB, lane) : make_float4(0.f, 0.f, 0.f, 0.f);
        for (int r = 0; r < T; r++) {
            const int nv = min((p0 + r + 1) >> 2, sg.N);
            if (pb >= nv) continue;
            const float4 * q = (const float4 *)((const float *)sg.a + (size_t)r * 512);
            float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 1
            for (int h = 0; h < 4; h++) {                                 // heads in order (rolled: code size)
                const float4 qh = q[h * 32 + lane];
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    float a = __builtin_fmaf(qh.x, k[i].x, 0.0f);
                    a = __builtin_fmaf(qh.y, k[i].y, a); a = __builtin_fmaf(qh.z, k[i].z, a); a = __builtin_fmaf(qh.w, k[i].w, a);
                    a = wsum(a);
                    a = a > 0.0f ? a : 0.0f;
                    s[i] = __builtin_fmaf(a, w, s[i]);
                }
            }
            const float v = lane == 0 ? s[0] : lane == 1 ? s[1] : lane == 2 ? s[2] : s[3];
            if (lane < 4 && pb + lane < nv) out[(size_t)r * sg.K + pb + lane] = v;
        }
    }
}

// the scores' total order as unsigned keys (larger key = larger float; NaN above +inf, -0 below +0)
__device__ __forceinline__ unsigned qsa_key(float v) {
    const unsigned b = __builtin_bit_cast(unsigned, v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
// OP_QSA_SELECT (one task per pass row): the pools row t attends to. nv <= 512: all of them (0 .. nv - 1). Otherwise the 512
//   with the largest scores, ties broken by the lowest pool index (deterministic, independent of the grid), emitted in
//   ascending index order: a 4-pass 8-bit radix select of the threshold key (LDS histograms with integer atomics: exact
//   counts), then one ascending pass taking every key above it and the lowest-indexed keys equal to it (wave ballots and
//   per-wave prefix counts). llama.cpp's own top-k keeps a set; its tie order at the threshold differs (bitonic network /
//   atomics), so only ties straddling rank 512 can differ from it.
//   Seg.a = SCORE fp32 [T][K], Seg.y = SEL int32 [T][QSA_SELW], Seg.asum = int32 pos [1], N = pool rows (nv clamp), K = SCORE
//   row stride.
__device__ __forceinline__ void op_qsa_select(const Seg & sg, int r) {
    __attribute__((address_space(3))) int * hist = QSA_LDS, * misc = QSA_LDS + 256, * wt = QSA_LDS + 260, * wt2 = QSA_LDS + 268;
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, pos = ((const int *)sg.asum)[0] + r;
    if (pos < 0) return;
    const int nv = min((pos + 1) >> 2, sg.N), ntail = (pos + 1) & 3;
    int * out = (int *)sg.y + (size_t)r * QSA_SELW;
    const float * sc = (const float *)sg.a + (size_t)r * sg.K;
    if (nv <= QSA_TOPK) {
        for (int i = tid; i < nv; i += blockDim.x) out[4 + i] = i;
        if (tid == 0) { out[0] = nv; out[1] = ntail; out[2] = nv; out[3] = 0; }
        return;
    }
    unsigned prefix = 0; int need = QSA_TOPK;
    for (int shift = 24; shift >= 0; shift -= 8) {
        const unsigned hm = shift == 24 ? 0u : 0xFFFFFFFFu << (shift + 8);
        hist[tid] = 0;   // (256 threads, 256 bins)
        __syncthreads();
        for (int i = tid; i < nv; i += blockDim.x) {
            const unsigned k = qsa_key(sc[i]);
            if ((k & hm) == prefix) __hip_atomic_fetch_add(hist + ((k >> shift) & 255), 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        }
        __syncthreads();
        if (tid == 0) {
            int above = 0, b = 255;
            for (; b > 0; b--) { if (above + hist[b] >= need) break; above += hist[b]; }
            misc[0] = b; misc[1] = need - above;
        }
        __syncthreads();
        prefix |= (unsigned)misc[0] << shift; need = misc[1];
        __syncthreads();
    }
    const unsigned lt = (1u << lane) - 1u;
    int base = 0, eqbase = 0;
    for (int c = 0; c < nv && base < QSA_TOPK; c += blockDim.x) {
        const int i = c + tid;
        const unsigned k = i < nv ? qsa_key(sc[i]) : 0u;
        const bool gt = i < nv && k > prefix, eq = i < nv && k == prefix;
        const unsigned em = __builtin_amdgcn_ballot_w32(eq);
        if (lane == 0) wt[wave] = __builtin_popcount(em);
        __syncthreads();
        int eo = eqbase, et = 0;
        for (int v = 0; v < 8; v++) { const int n = wt[v]; et += n; if (v < wave) eo += n; }
        const bool take = gt || (eq && eo + __builtin_popcount(em & lt) < need);
        const unsigned tm = __builtin_amdgcn_ballot_w32(take);
        if (lane == 0) wt2[wave] = __builtin_popcount(tm);
        __syncthreads();
        int to = base, tt = 0;
        for (int v = 0; v < 8; v++) { const int n = wt2[v]; tt += n; if (v < wave) to += n; }
        const int o = to + __builtin_popcount(tm & lt);
        if (take && o < QSA_TOPK) out[4 + o] = i;
        base += tt; eqbase += et;
        __syncthreads();
    }
    if (tid == 0) { out[0] = min(base, QSA_TOPK); out[1] = ntail; out[2] = nv; out[3] = (int)prefix; }
}
