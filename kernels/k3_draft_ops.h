// DFlash 2 drafter ops for the K3 engine (docs/specs/spec-drafters.md 1, S2-S3; python/caminus/dflash_k3.py). Included
// by kernels/k3_gguf_ops.h only when K3_DRAFT is defined (kernels/k3_gguf_draft.hip): the trunk's code objects do not
// contain any of this. Scratch LDS reuses AQ's storage (5148 B in the multi-token GSQ module): a segment's tasks never hold
// AQ data across these ops (they are not fp32-activation GG ops), so the module keeps the trunk's LDS size and occupancy.
// Graph reference: ignite-llama src/models/dflash.cpp (build_dflash2_conv, build_dflash2_selector, graph<false>) and
// common/speculative.cpp (draft-dflash: the greedy lattice walk).
#ifndef DF_NPT
#define DF_NPT 16   // logits per lane in OP_DF_TOPK's scan (a chunk = 256 * DF_NPT ids: 4096, so a 124160-row shard merges 31 x 16 <= 640 pairs)
#endif
#define DF_CHUNK (256 * DF_NPT)
enum { OP_DF_CONV = 31, OP_DF_TOPK = 32, OP_DF_SELECT = 33, OP_DF_EMBED = 34, OP_DF_MVCONV = 35, OP_DF_ACCEPT = 36 };

#define DF_LDS ((__attribute__((address_space(3))) int *)AQ)   // 1287 ints
static_assert(AQ_ALLOC * (int)sizeof(block_q8_1) >= 1280 * 4, "the drafter ops' scratch lives in AQ's storage");

// OP_DF_CONV: build_dflash2_conv for one block of T tokens (the draft block: n_blocks = 1, block_size = T): per channel c of
//   group g = c / gs, out[t, c] = sum_{tap < min(ks, T)} (dyn[t, g + ng * (tap + ks * side)] + base[c + C * (tap + ks * side)])
//   * h[t - tap, c] (0 before the block), terms added in tap order (ggml_mul, then a chain of ggml_add; the coefficient sum
//   is ggml_add(coeff, base)); optionally + res[t, c] (the residual ggml_add that follows) and a q8_1 copy.
//   Seg.a = h fp32 [T, C], Seg.gs = dyn fp32 [T, 2 * ks * ng], Seg.W = base fp32 [2][ks][C], Seg.y = out [T, C],
//   Seg.aux = res [T, C] or 0, Seg.asum = q8_1 copy or 0; N = T * C, K = C, pad1 = side, pad2 = gs, work = ks;
//   a task = rows_per_task elements (a multiple of 256: whole waves of 32 consecutive elements for the q8_1 blocks).
__device__ __forceinline__ void op_df_conv(const Seg & sg, int t) {
#pragma clang fp contract(off)
    const int C = sg.K, gs = sg.pad2, ng = C / gs, ks = sg.work, side = sg.pad1, T = seg_T(sg);
    const int ntap = min(ks, T), ndyn = 2 * ks * ng;
    const float * h = (const float *)sg.a, * dyn = (const float *)sg.gs, * base = (const float *)sg.W, * res = (const float *)sg.aux;
    const int i0 = t * sg.rows_per_task, i1 = min(sg.N, i0 + sg.rows_per_task);
    for (int i = i0 + (int)K3TID; i < i1; i += blockDim.x) {
        const int tt = i / C, c = i - tt * C, g = c / gs;
        float acc = 0.0f;
        for (int tap = 0; tap < ntap; tap++) {
            const int k = tap + ks * side;
            const float w = dyn[(size_t)tt * ndyn + g + ng * k] + base[c + C * k];
            const float v = tt - tap >= 0 ? h[(size_t)(tt - tap) * C + c] : 0.0f;
            const float term = w * v;
            acc = tap == 0 ? term : acc + term;
        }
        if (res) acc = acc + res[i];
        ((float *)sg.y)[i] = acc;
        if (sg.asum) q8_emit((void *)sg.asum, (size_t)i, acc);
    }
}

// ---- top-k with a total order: (value desc, id asc). Keys: the float's order-preserving int (NaN above +inf, -0 below +0;
// the same order as caminus.dflash._topk). Padding: key INT_MIN, a unique large id.
__device__ __forceinline__ int df_key(float v) { const int b = __builtin_bit_cast(int, v); return b >= 0 ? b : b ^ 0x7FFFFFFF; }
__device__ __forceinline__ float df_val(int k) { return __builtin_bit_cast(float, k >= 0 ? k : k ^ 0x7FFFFFFF); }
__device__ __forceinline__ bool df_better(int ka, int ia, int kb, int ib) { return ka > kb || (ka == kb && ia < ib); }

// rank selection over n (key, id) pairs in LDS (l[2 j], l[2 j + 1]), all threads: pair j's rank = #pairs better than it;
// ranks < K are written to out (as kv: int pairs, or fl: (value, float id) pairs). Ids must be unique.
__device__ __forceinline__ void df_rank_out(const __attribute__((address_space(3))) int * l, int n, int K, int * kv, float * fl) {
    for (int j = K3TID; j < n; j += blockDim.x) {
        const int kj = l[2 * j], ij = l[2 * j + 1];
        int r = 0;
        for (int q = 0; q < n; q++) r += df_better(l[2 * q], l[2 * q + 1], kj, ij);
        if (r < K) {
            if (kv) { kv[2 * r] = kj; kv[2 * r + 1] = ij; }
            else { fl[2 * r] = df_val(kj); fl[2 * r + 1] = (float)ij; }
        }
    }
}

// OP_DF_TOPK: the top K (<= 16) of logit rows, by (value desc, id asc).
//   work 0 (scan): task = (row, chunk of DF_CHUNK ids); Seg.a = logits fp32 [rows, N] (row r at r * N), Seg.y = int pairs
//     (key, id) [rows - pad2][nchunk][K], N = row length, K, pad1 = nchunk, pad2 = first row read (rows below are skipped),
//     Seg.rs unused, Seg.aux = the id offset (this card's first vocabulary row). Each wave takes 32 * DF_NPT ids (DF_NPT per lane) and
//     finds its top K in K rounds of a wave argmax (no barriers); the 8 waves' 8 K are then ranked in LDS.
//   work 1 (merge): task = (row, part), pad1 = parts per row; Seg.a = int pairs [rows][N] (N = nchunk * K <= 640), Seg.y = float pairs (value,
//     float(id)) [rows][K] (ids < 2^24 are exact: the exchange adds them as floats).
__device__ __forceinline__ void df_argmax_scan(const Seg & sg, int t);   // (work 2: below, with the device accept)
__device__ __forceinline__ void op_df_topk(const Seg & sg, int t) {
    if (sg.work == 2) { df_argmax_scan(sg, t); return; }
    const int K = sg.K, tid = K3TID, lane = tid % 32, wave = tid / 32;
    __attribute__((address_space(3))) int * l = DF_LDS;
    if (sg.work == 0) {
        const int nch = sg.pad1, row = sg.pad2 + t / nch, ch = t % nch, N = sg.N, off = (int)(size_t)sg.aux;
        const float * x = (const float *)sg.a + (size_t)row * N;
        const int b0 = ch * DF_CHUNK + wave * (32 * DF_NPT);
        int key[DF_NPT]; unsigned taken = 0;
#pragma unroll
        for (int j = 0; j < DF_NPT; j++) { const int i = b0 + j * 32 + lane; key[j] = i < N ? df_key(x[i]) : INT_MIN; }
        for (int r = 0; r < K; r++) {
            int bk = INT_MIN, bi = INT_MAX;
#pragma unroll
            for (int j = 0; j < DF_NPT; j++) {
                const int i = b0 + j * 32 + lane, id = i < N ? off + i : INT_MAX - (b0 + j * 32 + lane);
                if (!((taken >> j) & 1u) && df_better(key[j], id, bk, bi)) { bk = key[j]; bi = id; }
            }
#pragma unroll
            for (int m = 16; m >= 1; m >>= 1) {
                const int ok = __shfl_xor(bk, m, 32), oi = __shfl_xor(bi, m, 32);
                if (df_better(ok, oi, bk, bi)) { bk = ok; bi = oi; }
            }
#pragma unroll
            for (int j = 0; j < DF_NPT; j++) {
                const int i = b0 + j * 32 + lane, id = i < N ? off + i : INT_MAX - (b0 + j * 32 + lane);
                if (id == bi) taken |= 1u << j;
            }
            if (lane == 0) { l[2 * (wave * K + r)] = bk; l[2 * (wave * K + r) + 1] = bi; }
        }
        __syncthreads();
        df_rank_out(l, 8 * K, K, (int *)sg.y + (size_t)((row - sg.pad2) * nch + ch) * K * 2, nullptr);
        __syncthreads();
        return;
    }
    // work 1: task = (row, part of pad1 parts): every part loads the row's N pairs, ranks its own slice against all of them
    const int N = sg.N, P = max(1, sg.pad1), row = t / P, part = t % P;
    const int * in = (const int *)sg.a + (size_t)row * N * 2;
    for (int j = tid; j < 2 * N; j += blockDim.x) l[j] = in[j];
    __syncthreads();
    float * fl = (float *)sg.y + (size_t)row * K * 2;
    const int j0 = part * N / P, j1 = (part + 1) * N / P;
    for (int j = j0 + tid; j < j1; j += blockDim.x) {
        const int kj = l[2 * j], ij = l[2 * j + 1];
        int r = 0;
        for (int q2 = 0; q2 < N; q2++) r += df_better(l[2 * q2], l[2 * q2 + 1], kj, ij);
        if (r < K) { fl[2 * r] = df_val(kj); fl[2 * r + 1] = (float)ij; }
    }
    __syncthreads();
}

// OP_DF_SELECT (one task): the DFlash 2 selector and the greedy lattice walk for the k = N draft positions 1 .. N.
//   Candidates: Seg.a = float pairs [halves = pad1][N][KC] (each card's top KC of its vocabulary rows, after the exchange),
//   merged per position into the top KC by (value desc, id asc) (the candidate order only matters for ties).
//   Position i's predecessor rows: the anchor (i = 1; Seg.asum = int32 token id) or position i - 1's candidates. With
//   cond = prev_row * gate[i] (ggml_mul), score[p][s] = succ_row(cand_i[s]) . cond_p + unary_i[s] (ggml_mul_mat + ggml_add);
//   the walk takes the first maximum of score[pred] (std::max_element), from pred 0 at position 1.
//   Seg.W = selector_predecessor Q8_0 [vocab][rank], Seg.gs = selector_successor Q8_0, Seg.aux = gate fp32 [T][rank]
//   (position i at row i), Seg.y = int32 out: drafts [N], then candidates [N][KC] (int), unary [N][KC] and the walked score
//   rows [N][KC] (float); K = rank (<= 768; lane l takes element l of each Q8_0 block, blocks in order, then a wave sum),
//   pad2 = KC (16), work = row bytes, out_f32 = vocabulary rows (every row id is clamped into [0, vocab)).
__device__ __forceinline__ void op_df_select(const Seg & sg, int t) {
#pragma clang fp contract(off)
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, N = sg.N, KC = sg.pad2, H = sg.pad1, rank = sg.K, rb = sg.work;
    // LDS: [N][KC] cand, [N][KC] unary (bits), [KC + 1] scores + pick, then from 512: cond [rank] (<= 768), and during
    // the merge the H * N * KC input pairs (key, id) from 512
    __attribute__((address_space(3))) int * l = DF_LDS;
    int * out = (int *)sg.y;
    const float * cp = (const float *)sg.a;
    // 1. merge the halves per position (rank selection over the position's H * KC pairs, as df_rank_out)
    __attribute__((address_space(3))) int * in = l + 512;
    const int n = H * KC, tot = N * n;
    for (int j = tid; j < tot; j += blockDim.x) {        // pair j = (position w, half hh, s): from [hh][w][s]
        const int w = j / n, r = j % n, hh = r / KC, s2 = r % KC;
        const float * p = cp + ((size_t)(hh * N + w) * KC + s2) * 2;
        const float fi = p[1];   // (an id outside [0, vocab), NaN included, never cast: -1, clamped where it is used)
        in[2 * j] = df_key(p[0]); in[2 * j + 1] = (fi >= 0.0f && fi < (float)sg.out_f32) ? (int)fi : -1;
    }
    __syncthreads();
    for (int j = tid; j < tot; j += blockDim.x) {
        const int w = j / n, kj = in[2 * j], ij = in[2 * j + 1];
        int r = 0;
        for (int q = w * n; q < w * n + n; q++) r += df_better(in[2 * q], in[2 * q + 1], kj, ij);
        if (r < KC) { l[w * KC + r] = ij; l[N * KC + w * KC + r] = __builtin_bit_cast(int, df_val(kj)); }
    }
    __syncthreads();
    const float * gate = (const float *)sg.aux;
    const uint8_t * Wp = (const uint8_t *)sg.W, * Ws = (const uint8_t *)sg.gs;
    __attribute__((address_space(3))) float * sc = (__attribute__((address_space(3))) float *)(l + 2 * N * KC);
    int pr = 0;
    for (int i = 0; i < N; i++) {
        // ids clamped into [0, vocab) (Seg.out_f32 = vocab rows): they crossed the exchange as floats, and a task after a
        // bailed wait reads whatever its slot holds (memory safety only: every writer emits valid ids)
        const int pid = min(max(i == 0 ? ((const int *)sg.asum)[0] : l[(i - 1) * KC + pr], 0), sg.out_f32 - 1);
        // cond = prev_row * gate[i + 1] into LDS (element r = 32 m + lane of block m): registers spilled (cond[8] in each lane)
        __attribute__((address_space(3))) float * cond = (__attribute__((address_space(3))) float *)(l + 512);
        const float * g = gate + (size_t)(i + 1) * rank;
        for (int r = tid; r < rank; r += blockDim.x) {
            const block_q8_0 * pb = (const block_q8_0 *)(Wp + (size_t)pid * rb) + r / 32;
            const float pv = (float)pb->qs[r % 32] * __half2float(pb->d);
            cond[r] = pv * g[r];
        }
        __syncthreads();
        for (int s = wave; s < KC; s += 8) {
            const int sid = min(max(l[i * KC + s], 0), sg.out_f32 - 1);
            const block_q8_0 * sb = (const block_q8_0 *)(Ws + (size_t)sid * rb);
            float a = 0.0f;
#pragma unroll 1
            for (int m = 0; m < rank / 32; m++) { const float sv = (float)sb[m].qs[lane] * __half2float(sb[m].d); a = a + sv * cond[m * 32 + lane]; }
            a = wsum(a);
            if (lane == 0) sc[s] = a + __builtin_bit_cast(float, l[N * KC + i * KC + s]);
        }
        __syncthreads();
        if (tid == 0) {
            int b = 0;
            for (int s = 1; s < KC; s++) if (sc[b] < sc[s]) b = s;   // std::max_element: the first maximum
            sc[KC] = __builtin_bit_cast(float, b);
            out[i] = min(max(l[i * KC + b], 0), sg.out_f32 - 1);   // the draft id clamped into [0, vocab)
        }
        if (tid < KC) {
            out[N + i * KC + tid] = l[i * KC + tid];
            out[N + N * KC + i * KC + tid] = l[N * KC + i * KC + tid];
            out[N + 2 * N * KC + i * KC + tid] = __builtin_bit_cast(int, sc[tid]);
        }
        __syncthreads();
        pr = __builtin_bit_cast(int, sc[KC]);
        __syncthreads();
    }
}

// OP_DF_EMBED: token_embd rows (IQ4_XS) dequantized to fp32 as ggml's dequantize_row_iq4_xs: dl = d * (ls - 32), then
//   dl * kvalues_iq4nl[q]. Task tt = row tt: Seg.asum = int32 token ids [ntask], Seg.W = token_embd base, Seg.y = out
//   [ntask, N]; N = row length (a multiple of 256), pad2 = row bytes, pad1 = vocabulary rows (ids outside: a zero row).
__device__ __forceinline__ void op_df_embed(const Seg & sg, int tt) {
    const int id = ((const int *)sg.asum)[tt], N = sg.N;
    float * y = (float *)sg.y + (size_t)tt * N;
    const bool ok = id >= 0 && id < sg.pad1;
    const block_iq4_xs * x = (const block_iq4_xs *)((const uint8_t *)sg.W + (size_t)(ok ? id : 0) * sg.pad2);
    for (int e = K3TID; e < N; e += blockDim.x) {
        const int b = e / 256, w = e % 256, ib = w / 32, j = w % 32;
        const float d = __half2float(x[b].d);
        const int ls = ((x[b].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x[b].scales_h >> 2 * ib) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        const uint8_t q = x[b].qs[16 * ib + (j & 15)];
        y[e] = ok ? dl * (float)kvalues_iq4nl[j < 16 ? (q & 0xf) : (q >> 4)] : 0.0f;
    }
}

// OP_KV_APPEND with a ring (Seg.aux = ring mask, rows - 1, rows a power of two): f16 only; logical position p lives in row
// p & mask, so any position addresses inside the allocation.
__device__ __forceinline__ void op_kv_append_ring(const Seg & sg, int tt) {
    const int pos = ((const int *)sg.asum)[0] + tt, M = sg.N;
    const uint32_t mask = (uint32_t)(size_t)sg.aux;
    half * c = (half *)((uint8_t *)sg.y + (size_t)((uint32_t)pos & mask) * kv_pos_bytes(M, 0));
    for (int i = K3TID; i < M; i += blockDim.x) {
        c[i] = __float2half(((const float *)sg.a)[(size_t)tt * M + i]);
        c[M + i] = __float2half(((const float *)sg.W)[(size_t)tt * M + i]);
    }
}

// OP_ATTN_DEC with the drafter's flags (Seg.pad2 != 0): pad2 bit 0 = non-causal (every row of the pass sees positions up to
// pos + T - 1, the whole block), bit 1 = a sliding window, Seg.aux = window W (low 32 bits) | ring mask << 32 (0 = linear).
// Row tt (position p = pos + tt) attends over [lo, hi]: lo = max(0, p - (W - 1)) with the window (llama.cpp
// is_masked_swa STANDARD per query: p - p0 >= W is masked, also for non-causal: LLAMA_NON_CAUSAL_TYPE_ALL), else 0; hi = pos
// + T - 1 non-causal, else p. f16 cache only. The same chunked online softmax and per-position order as op_attn_dec.
__device__ __forceinline__ void op_attn_dec_df(const Seg & sg, int h) {
    constexpr int CH = 512;
    __attribute__((address_space(3))) float * s_ = (__attribute__((address_space(3))) float *)DF_LDS;   // [CH] + [8]
    __attribute__((address_space(3))) float * red = s_ + CH;
    const int T = seg_T(sg), heads = sg.N / T, tt = h / heads;
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, D = sg.K, nkv = sg.pad1, g = (h - tt * heads) / (heads / nkv);
    const int pos0 = ((const int *)sg.asum)[0], p = pos0 + tt;
    const uint64_t ax = (uint64_t)(size_t)sg.aux;
    const int W = (int)(uint32_t)ax; const uint32_t mask = (uint32_t)(ax >> 32);
    const int hi = (sg.pad2 & 1) ? pos0 + T - 1 : p, lo = ((sg.pad2 & 2) && W > 0) ? max(0, p - (W - 1)) : 0;
    const int npos = hi - lo + 1, M = nkv * D;
    const size_t pb = kv_pos_bytes(M, 0);
    const float * q = (const float *)sg.a + (size_t)h * D;
    const uint8_t * cache = (const uint8_t *)sg.W;
    float run_m = -INFINITY, run_s = 0.0f, acc[2] = {0.0f, 0.0f};
#pragma unroll 1
    for (int c0 = 0; c0 < npos; c0 += CH) {
        const int n = min(CH, npos - c0);
        for (int t = wave; t < n; t += 8) {
            const uint32_t row = mask ? ((uint32_t)(lo + c0 + t) & mask) : (uint32_t)(lo + c0 + t);
            float a = 0.0f;
            for (int d = lane; d < D; d += 32) a += q[d] * kv_get(cache + (size_t)row * pb, M, 0, 0, g * D + d);
            a = warp_reduce_sum<32>(a);
            if (lane == 0) s_[t] = a * sg.rs;
        }
        __syncthreads();
        float m = -INFINITY;
        for (int t = tid; t < n; t += blockDim.x) m = fmaxf(m, s_[t]);
        m = warp_reduce_max<32>(m); if (lane == 0) red[wave] = m; __syncthreads();
        m = red[0]; for (int w = 1; w < 8; w++) m = fmaxf(m, red[w]); __syncthreads();
        const float mn = fmaxf(run_m, m), corr = expf(run_m - mn);
        float sum = 0.0f;
        for (int t = tid; t < n; t += blockDim.x) { const float e = expf(s_[t] - mn); s_[t] = e; sum += e; }
        sum = warp_reduce_sum<32>(sum); if (lane == 0) red[wave] = sum; __syncthreads();
        sum = 0.0f; for (int w = 0; w < 8; w++) sum += red[w];
        run_s = c0 == 0 ? sum : run_s * corr + sum;
        for (int j = 0; j < 2; j++) {
            const int d = tid + j * (int)blockDim.x;
            if (d >= D) break;
            float part = 0.0f;
            for (int t = 0; t < n; t++) {
                const uint32_t row = mask ? ((uint32_t)(lo + c0 + t) & mask) : (uint32_t)(lo + c0 + t);
                part += s_[t] * kv_get(cache + (size_t)row * pb, M, 0, 1, g * D + d);
            }
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

// ---- llama.cpp's own norm layouts (bitwise ggml-cuda norm.cu / rope.cu as built for gfx1100), selected by Seg.aux on
// OP_RMS_MUL / OP_QK_NR in this module only (the trunk's ops keep their own order). Variant bits in aux (>= 1):
//   OP_RMS_MUL (rows of K = 4 * n4, n4 <= 1024: rms_norm_f32<1024, 1, vec> with the fused mul): thread v < n4 of 1024 adds
//     its float4's squares (aux bits 1-2: 1 = ((x x + y y) + z z) + w w, 2 = fma chain from x x, 3 = fma chain from y y),
//     tmp = 0 + that; warp_reduce_sum per 32 threads (xor 16 .. 1), the 32 warp sums through LDS, warp_reduce_sum again;
//     scale = rsqrtf(tmp / K + eps); out = (scale * x) * w. 256 threads walk the 1024 virtual threads 4 at a time.
//   OP_QK_NR (per head, K < 1024: rms_norm_mul_rope_f32<256>): one element per thread, warp sums, the 8 warp sums padded
//     to 32 lanes and warp-reduced, (scale * x) * w, then NeoX RoPE: theta = pos * powf(theta_scale, j), cosf / sinf,
//     (aux bits 1-2 for the rotation's contraction: 1 = plain products, 2 = fma(x0, c, -(x1 s)) / fma(x1, c, x0 s) (the
//     llama.cpp build's: measured element by element against its dump),
//     3 = fma(-x1, s, x0 c) / fma(x1, c, x0 s)).
__device__ __forceinline__ float df_sq4(float x, float y, float z, float w, int v) {
#pragma clang fp contract(off)
    if (v == 2) return __builtin_fmaf(w, w, __builtin_fmaf(z, z, __builtin_fmaf(y, y, x * x)));
    if (v == 3) return __builtin_fmaf(w, w, __builtin_fmaf(z, z, __builtin_fmaf(x, x, y * y)));
    return ((x * x + y * y) + z * z) + w * w;
}
__device__ __forceinline__ void op_rms_mul_ll(const Seg & sg, int t) {
#pragma clang fp contract(off)
    __attribute__((address_space(3))) float * part = (__attribute__((address_space(3))) float *)DF_LDS;   // [32]
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, K = sg.K, n4 = K / 4, var = (int)((size_t)sg.aux & 3);
    const float * x = (const float *)sg.a + (size_t)t * K, * w = (const float *)sg.W + (size_t)(sg.pad1 ? t % sg.pad1 : t) * K;
    float * y = (float *)sg.y + (size_t)t * K;
    for (int j = 0; j < 4; j++) {                 // virtual thread v = tid + 256 j: virtual warp wave + 8 j
        const int v = tid + 256 * j;
        float tmp = 0.0f;
        if (v < n4) { const float4 q = ((const float4 *)x)[v]; tmp = 0.0f + df_sq4(q.x, q.y, q.z, q.w, var); }
        tmp = wsum(tmp);
        if (lane == 0) part[wave + 8 * j] = tmp;
    }
    __syncthreads();
    float tot = wsum(part[lane]);                 // (every wave the same)
    const float scale = rsqrtf(tot / K + sg.rs);
    for (int i = tid; i < K; i += blockDim.x) {
        const float v = (scale * x[i]) * w[i];
        y[i] = v;
        if (sg.asum) q8_emit((void *)sg.asum, (size_t)t * K + i, v);
    }
    __syncthreads();
}
__device__ __forceinline__ void op_qk_nr_ll(const Seg & sg, int t) {
#pragma clang fp contract(off)
    __attribute__((address_space(3))) float * part = (__attribute__((address_space(3))) float *)DF_LDS;   // [8]
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, D = sg.K, nrot = sg.out_f32, var = (int)((size_t)sg.aux & 3);
    // addressing: task t = token tt * heads + head hh; x at tt * pad1 (the token stride: q / k / v may sit in one row) + hh * D
    const int heads = sg.N / seg_T(sg), tt = t / heads, hh = t - tt * heads;
    const float * x = (const float *)sg.a + (size_t)tt * sg.pad1 + (size_t)hh * D, * w = (const float *)sg.W;
    // aux bit 3: write K (f16) and the head's V (Seg.y = the V rows, same token stride) into the ring (Seg.gs = cache, mask
    // in aux >> 32) instead of fp32 out: the kv append fused (op_kv_append_ring's conversion: __float2half)
    const bool to_ring = ((size_t)sg.aux >> 3) & 1;
    const uint32_t rmask = (uint32_t)((uint64_t)(size_t)sg.aux >> 32);
    half * kc = to_ring ? (half *)((uint8_t *)sg.gs + (size_t)((uint32_t)(((const int *)sg.asum)[0] + tt) & rmask) * kv_pos_bytes(heads * D, 0)) + hh * D : nullptr;
    float * y = to_ring ? nullptr : (float *)sg.y + (size_t)t * D;
    const float eps = __builtin_bit_cast(float, sg.pad2);
    float tot;
    if ((size_t)sg.aux & 4) {   // unfused (rms_norm_f32 vec, D / 4 <= 32: one warp of float4 partials, df_sq4 variant 2)
        float tmp = 0.0f;
        if (lane < D / 4) { const float4 q = ((const float4 *)x)[lane]; tmp = 0.0f + df_sq4(q.x, q.y, q.z, q.w, 2); }
        tot = wsum(tmp);
    } else {
        float tmp = 0.0f;
        for (int i = tid; i < D; i += blockDim.x) { const float xi = x[i]; tmp = __builtin_fmaf(xi, xi, tmp); }
        tmp = wsum(tmp);
        if (lane == 0) part[wave] = tmp;
        __syncthreads();
        tot = wsum(lane < 8 ? part[lane] : 0.0f);
    }
    const float scale = rsqrtf(tot / D + eps);
    const float pos = (float)(((const int *)sg.asum)[0] + tt);
    for (int i0 = 2 * tid; i0 < D; i0 += 2 * blockDim.x) {
        const bool rot = i0 < nrot;
        const int ix0 = rot ? i0 / 2 : i0, ix1 = rot ? i0 / 2 + nrot / 2 : i0 + 1;
        const float x0 = (scale * x[ix0]) * w[ix0], x1 = (scale * x[ix1]) * w[ix1];
        if (!rot) { if (to_ring) { kc[ix0] = __float2half(x0); kc[ix1] = __float2half(x1); } else { y[ix0] = x0; y[ix1] = x1; } continue; }
        const float theta = pos * powf(sg.rs, (float)i0 / 2.0f);
        const float c = cosf(theta), sn = sinf(theta);
        float r0, r1;
        if (var == 2) { r0 = __builtin_fmaf(x0, c, -(x1 * sn)); r1 = __builtin_fmaf(x1, c, x0 * sn); }
        else if (var == 3) { r0 = __builtin_fmaf(-x1, sn, x0 * c); r1 = __builtin_fmaf(x1, c, x0 * sn); }
        else { r0 = x0 * c - x1 * sn; r1 = x0 * sn + x1 * c; }
        if (to_ring) { kc[ix0] = __float2half(r0); kc[ix1] = __float2half(r1); } else { y[ix0] = r0; y[ix1] = r1; }
    }
    if (to_ring) {   // the head's V row: [K | V] per position, V at heads * D
        const float * v = (const float *)sg.y + (size_t)tt * sg.pad1 + (size_t)hh * D;
        for (int i = tid; i < D; i += blockDim.x) kc[heads * D + i] = __float2half(v[i]);
    }
    __syncthreads();
}

// op_attn_dec_df's D = 128 path: the same flags and window, but each lane holds 4 dims (one 8-byte f16 load per K / V row),
// waves split the positions for the scores AND the weighted V sum (the generic path's V pass was one sequential loop over
// every position per thread: 143 us per layer at 334 positions), the 8 waves' V partials summed through LDS in wave order.
// Chunks of 128 positions with the online softmax. Its own summation order (fixed: deterministic); not bitwise the generic
// path's, nor llama.cpp's flash-attention kernel (which computes KQ in half precision).
__device__ __forceinline__ void op_attn_dec_df128(const Seg & sg, int h) {
#pragma clang fp contract(off)
    constexpr int CH = 128, D = 128;
    __attribute__((address_space(3))) float * s_ = (__attribute__((address_space(3))) float *)DF_LDS;   // [CH]
    __attribute__((address_space(3))) float * red = s_ + CH;                                             // [8]
    __attribute__((address_space(3))) float * va = red + 8;                                              // [8][D]
    const int T = seg_T(sg), heads = sg.N / T, tt = h / heads;
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, nkv = sg.pad1, g = (h - tt * heads) / (heads / nkv);
    const int pos0 = ((const int *)sg.asum)[0], p = pos0 + tt;
    const uint64_t ax = (uint64_t)(size_t)sg.aux;
    const int W = (int)(uint32_t)ax; const uint32_t mask = (uint32_t)(ax >> 32);
    const int hi = (sg.pad2 & 1) ? pos0 + T - 1 : p, lo = ((sg.pad2 & 2) && W > 0) ? max(0, p - (W - 1)) : 0;
    const int npos = hi - lo + 1, M = nkv * D;
    const size_t pb = kv_pos_bytes(M, 0);
    const float4 q = ((const float4 *)((const float *)sg.a + (size_t)h * D))[lane];
    const uint8_t * kbase = (const uint8_t *)sg.W + (size_t)(g * D + 4 * lane) * 2, * vbase = kbase + (size_t)M * 2;
    float run_m = -INFINITY, run_s = 0.0f, acc = 0.0f;
#pragma unroll 1
    for (int c0 = 0; c0 < npos; c0 += CH) {
        const int n = min(CH, npos - c0);
#pragma unroll 4
        for (int t = wave; t < n; t += 8) {
            const uint32_t row = mask ? ((uint32_t)(lo + c0 + t) & mask) : (uint32_t)(lo + c0 + t);
            const uint2 kw = *(const uint2 *)(kbase + (size_t)row * pb);
            const float2 k01 = __half22float2(__builtin_bit_cast(__half2, kw.x)), k23 = __half22float2(__builtin_bit_cast(__half2, kw.y));
            float a = ((q.x * k01.x + q.y * k01.y) + q.z * k23.x) + q.w * k23.y;
            a = wsum(a);
            if (lane == 0) s_[t] = a * sg.rs;
        }
        __syncthreads();
        float m = -INFINITY;
        for (int t = tid; t < n; t += blockDim.x) m = fmaxf(m, s_[t]);
        m = wred<true>(m); if (lane == 0) red[wave] = m; __syncthreads();
        m = red[0]; for (int w = 1; w < 8; w++) m = fmaxf(m, red[w]); __syncthreads();
        const float mn = fmaxf(run_m, m), corr = expf(run_m - mn);
        float sum = 0.0f;
        for (int t = tid; t < n; t += blockDim.x) { const float e = expf(s_[t] - mn); s_[t] = e; sum += e; }
        sum = wsum(sum); if (lane == 0) red[wave] = sum; __syncthreads();
        sum = 0.0f; for (int w = 0; w < 8; w++) sum += red[w];
        run_s = c0 == 0 ? sum : run_s * corr + sum;
        float4 pa = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll 4
        for (int t = wave; t < n; t += 8) {
            const uint32_t row = mask ? ((uint32_t)(lo + c0 + t) & mask) : (uint32_t)(lo + c0 + t);
            const uint2 vw = *(const uint2 *)(vbase + (size_t)row * pb);
            const float2 v01 = __half22float2(__builtin_bit_cast(__half2, vw.x)), v23 = __half22float2(__builtin_bit_cast(__half2, vw.y));
            const float e = s_[t];
            pa.x = pa.x + e * v01.x; pa.y = pa.y + e * v01.y; pa.z = pa.z + e * v23.x; pa.w = pa.w + e * v23.y;
        }
        va[wave * D + 4 * lane + 0] = pa.x; va[wave * D + 4 * lane + 1] = pa.y; va[wave * D + 4 * lane + 2] = pa.z; va[wave * D + 4 * lane + 3] = pa.w;
        __syncthreads();
        if (tid < D) {
            float part = 0.0f;
            for (int w = 0; w < 8; w++) part += va[w * D + tid];
            acc = c0 == 0 ? part : acc * corr + part;
        }
        run_m = mn;
        __syncthreads();
    }
    if (tid < D) {
        const float o = acc * (1.0f / run_s);
        ((float *)sg.y)[(size_t)h * D + tid] = o;
        if (sg.gs) q8_emit((void *)sg.gs, (size_t)h * D + tid, o);   // (waves 0 .. 3: one q8_1 block each; Seg.gs = q8_1 copy)
    }
}

// OP_DF_MVCONV: a Q8_0 matvec over T <= TG tokens (the multi-token rows of gg_op_mv_t, gg_row_ut) whose rows are channels,
// with OP_DF_CONV's side-1 conv + residual as its epilogue (lane 0 holds the row's T values): out[t, c] = conv(v)[t, c] +
// res[t, c], the same expressions in the same order as the separate segments (bitwise). Seg.W = Q8_0 rows, Seg.a = q8_1
// activations [T, K], Seg.y = out [T, N], Seg.gs = dyn [T, 2 ks ng], Seg.asum = base [2][ks][N], Seg.aux = res [T, N];
// N = channels, K, pad1 = side, pad2 = gs, work = ks (2 only); a task = rows_per_task rows (8 waves, a row each).
__device__ __forceinline__ void op_df_mvconv(const Seg & sg, int t) {
#pragma clang fp contract(off)
    const int wave = K3WAVE, lane = K3TID % 32, T = seg_T(sg), C = sg.N, kb = sg.K / QK8_1, nb = sg.K / QK8_0;
    const int r0 = t * sg.rows_per_task, r1 = min(C, r0 + sg.rows_per_task);
    const int gs = sg.pad2, ng = C / gs, side = sg.pad1, ndyn = 4 * ng;   // ks = 2
    const float * dyn = (const float *)sg.gs, * base = (const float *)sg.asum, * res = (const float *)sg.aux;
    for (int row = r0 + wave; row < r1; row += 8) {
        float v[TG];
        gg_row_ut<K3_MVT_U>(sg.W, (const block_q8_1 *)sg.a, kb, row, nb, lane, T, v);
        // ks = 2 (asserted on the host): lane tt < T computes token tt (every lane holds the T sums after wsum): (w0 v[tt]) +
        // (w1 v[tt - 1]) with v[-1] = 0 (the left padding's term w1 * 0 is still added, as ggml's chain does), then + res
        const float vt = lane == 0 ? v[0] : lane == 1 ? v[1] : lane == 2 ? v[2] : v[3];
        const float vp = lane == 0 ? 0.0f : lane == 1 ? v[0] : lane == 2 ? v[1] : v[2];
        if (lane < T) {
            const int g = row / gs, k0 = 2 * side, k1 = 2 * side + 1;
            const float w0 = dyn[(size_t)lane * ndyn + g + ng * k0] + base[row + C * k0];
            const float w1 = dyn[(size_t)lane * ndyn + g + ng * k1] + base[row + C * k1];
            const float t0 = w0 * vt, t1 = w1 * vp;
            ((float *)sg.y)[(size_t)lane * C + row] = (t0 + t1) + res[(size_t)lane * C + row];
        }
    }
}

// ---- split attention (pad2 bit 2: the partial pass; bit 3: the combine), D = 128, 4 query heads per kv head. The window
// [lo, hi] of row tt (as op_attn_dec_df) is cut into sub-splits of 64 positions from lo; each WAVE takes one (query head,
// sub-split) and keeps its own softmax state, so no barrier and no LDS: task t = (split s, row tt, kv head g) with s major
// (t = s * T * nkv + tt * nkv + g), 8 waves = 4 heads x 2 sub-splits (sub = 2 s + wave / 4, head = g * 4 + wave % 4).
//   scores: lane l = (key slot l / 8, 16 dims (l % 8) * 16 ..): 4 keys per step, 16 steps, the 8 lanes' sum by xor 1, 2, 4
//   (an invalid key, past hi: -inf); m = the wave max, p = expf(s - m), l = the sum of p (lane order, then xor 8, 16);
//   V: the same lane layout, acc[16] += p * v over the 16 steps, then the 4 key slots summed by xor 8, 16.
//   Partial record per (row, head, sub): 132 floats (m, l, 0, 0, acc[128]) at Seg.y + ((tt * heads + h) * NSUB + sub) * 132,
//   NSUB = 2 * ceil((W + T - 1) / 128) (W = window: every row's window fits). A sub-split past hi writes nothing.
// The combine (task = (row tt, head h), threads d < 128): over the row's sub-splits in order, m = max m_s, l = sum l_s
//   expf(m_s - m), acc = sum acc_s[d] expf(m_s - m); out = acc * (1 / l) (+ the q8_1 copy at Seg.gs). Seg.a = the partials.
// Deterministic (a fixed order), not bitwise op_attn_dec_df128 (another summation order) nor llama.cpp's flash attention.
__device__ __forceinline__ void df_attn_window(const Seg & sg, int T, int tt, int & lo, int & hi) {
    const int pos0 = ((const int *)sg.asum)[0], p = pos0 + tt, W = (int)(uint32_t)(uint64_t)(size_t)sg.aux;
    hi = (sg.pad2 & 1) ? pos0 + T - 1 : p; lo = ((sg.pad2 & 2) && W > 0) ? max(0, p - (W - 1)) : 0;
}
// sub-split size of a row: 64 positions (a fixed rule of the row's span, never of the grid, so the bits do not depend on
// it). 128 for long windows (two online-merged blocks per wave: one round of the grid at 2051) cost 23 VGPR spills in this
// module (96-VGPR cap): not used
__device__ __forceinline__ int df_attn_subsz(int lo, int hi) { return 64; }
__device__ __forceinline__ int df_attn_nsub(const Seg & sg, int T) {
    const int W = (int)(uint32_t)(uint64_t)(size_t)sg.aux;
    return 2 * ((W + T - 1 + 127) / 128);
}
// one 64-position block [k0, k1] of op_attn_split's wave, merged online into (m, ls, acc): m' = max(m, the block's max), the
// old sums scaled by expf(m - m') (0 for the first block: m = -inf), the block's p = expf(s - m'); acc holds each lane's
// key-slot partials (summed across the slots by the caller after the last block)
template <bool FIRST> __device__ __forceinline__ void df_attn_block(const float * q, const uint8_t * kb, const uint8_t * vb, int k0, int k1, uint32_t mask,
                                              size_t pb, float rs, int ks, float & m, float & ls, float * acc) {
#pragma clang fp contract(off)
    float sc[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        const int key = k0 + 4 * i + ks;
        const uint32_t row = mask ? ((uint32_t)key & mask) : (uint32_t)key;
        float a = 0.0f;
        if (key <= k1) {
            const uint4 w0 = *(const uint4 *)(kb + (size_t)row * pb), w1 = *(const uint4 *)(kb + (size_t)row * pb + 16);
            const uint32_t wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float2 kk = __half22float2(__builtin_bit_cast(__half2, wv[j]));
                a = a + q[2 * j] * kk.x; a = a + q[2 * j + 1] * kk.y;
            }
        }
        a = a + __shfl_xor(a, 1, 32); a = a + __shfl_xor(a, 2, 32); a = a + __shfl_xor(a, 4, 32);
        sc[i] = key <= k1 ? a * rs : -INFINITY;
    }
    float mb = sc[0];
#pragma unroll
    for (int i = 1; i < 16; i++) mb = fmaxf(mb, sc[i]);
    mb = fmaxf(mb, __shfl_xor(mb, 8, 32)); mb = fmaxf(mb, __shfl_xor(mb, 16, 32));
    const float mn = FIRST ? mb : fmaxf(m, mb), co = FIRST ? 0.0f : expf(m - mn);
    float lb = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; i++) { sc[i] = expf(sc[i] - mn); lb = lb + sc[i]; }
    lb = lb + __shfl_xor(lb, 8, 32); lb = lb + __shfl_xor(lb, 16, 32);
    ls = FIRST ? lb : ls * co + lb; m = mn;
    if (!FIRST) {
#pragma unroll
        for (int j = 0; j < 16; j++) acc[j] = acc[j] * co;
    }
#pragma unroll 4
    for (int i = 0; i < 16; i++) {
        const int key = k0 + 4 * i + ks;
        if (key <= k1) {
            const uint32_t row = mask ? ((uint32_t)key & mask) : (uint32_t)key;
            const uint4 w0 = *(const uint4 *)(vb + (size_t)row * pb), w1 = *(const uint4 *)(vb + (size_t)row * pb + 16);
            const uint32_t wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
            const float e = sc[i];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float2 vv = __half22float2(__builtin_bit_cast(__half2, wv[j]));
                acc[2 * j] = acc[2 * j] + e * vv.x; acc[2 * j + 1] = acc[2 * j + 1] + e * vv.y;
            }
        }
    }
}
__device__ __forceinline__ void op_attn_split(const Seg & sg, int t) {
#pragma clang fp contract(off)
    constexpr int D = 128, G = 4;
    const int T = seg_T(sg), heads = sg.N / T, nkv = sg.pad1, lane = K3TID % 32, wave = K3TID / 32;
    const int s = t / (T * nkv), r = t - s * (T * nkv), tt = r / nkv, g = r - tt * nkv;
    const int h = g * G + (wave & 3), sub = 2 * s + (wave >> 2), NSUB = df_attn_nsub(sg, T);
    int lo, hi; df_attn_window(sg, T, tt, lo, hi);
    const int sz = df_attn_subsz(lo, hi), ks = lane >> 3, db = lane & 7;
    if (lo + sz * sub > hi) return;                    // (the claim's barrier protects nothing here: no LDS used)
    const uint32_t mask = (uint32_t)((uint64_t)(size_t)sg.aux >> 32);
    const int M = nkv * D; const size_t pb = kv_pos_bytes(M, 0);
    const float * qp = (const float *)sg.a + ((size_t)(tt * heads + h)) * D + db * 16;
    float q[16];
#pragma unroll
    for (int j = 0; j < 16; j += 4) { const float4 v = *(const float4 *)(qp + j); q[j] = v.x; q[j + 1] = v.y; q[j + 2] = v.z; q[j + 3] = v.w; }
    const uint8_t * kb = (const uint8_t *)sg.W + (size_t)(g * D + db * 16) * 2, * vb = kb + (size_t)M * 2;
    float m = -INFINITY, ls = 0.0f, acc[16];
#pragma unroll
    for (int j = 0; j < 16; j++) acc[j] = 0.0f;
    // the sub-split's 64-position blocks in order (one, or two when sz = 128), merged online (df_attn_block)
    const int kA = lo + sz * sub;
    df_attn_block<true>(q, kb, vb, kA, min(hi, kA + 63), mask, pb, sg.rs, ks, m, ls, acc);
#pragma unroll
    for (int j = 0; j < 16; j++) { acc[j] = acc[j] + __shfl_xor(acc[j], 8, 32); acc[j] = acc[j] + __shfl_xor(acc[j], 16, 32); }
    float * rec = (float *)sg.y + ((size_t)(tt * heads + h) * NSUB + sub) * 132;
    if (lane < 8) {
#pragma unroll
        for (int j = 0; j < 16; j += 4) *(float4 *)(rec + 4 + db * 16 + j) = make_float4(acc[j], acc[j + 1], acc[j + 2], acc[j + 3]);
    }
    if (lane == 0) { rec[0] = m; rec[1] = ls; }   // (m, ls, acc: the online merge's state after the last block)
}
__device__ __forceinline__ void op_attn_combine(const Seg & sg, int t) {
#pragma clang fp contract(off)
    constexpr int D = 128;
    __attribute__((address_space(3))) float * cs = (__attribute__((address_space(3))) float *)DF_LDS;   // [NSUB] factors, [NSUB] l, m
    const int T = seg_T(sg), heads = sg.N / T, tt = t / heads, tid = K3TID, NSUB = df_attn_nsub(sg, T);
    int lo, hi; df_attn_window(sg, T, tt, lo, hi);
    const int sz = df_attn_subsz(lo, hi), nsub = (hi - lo + sz) / sz;   // ceil((hi - lo + 1) / sz)
    const float * rec = (const float *)sg.a + (size_t)t * NSUB * 132;
    // the records' (m, l) once into LDS (one thread each), then the max and the factors expf(m_u - m) in sub order
    if (tid < nsub) { cs[tid] = rec[tid * 132]; cs[NSUB + tid] = rec[tid * 132 + 1]; }
    __syncthreads();
    float m = cs[0];
    for (int u = 1; u < nsub; u++) m = fmaxf(m, cs[u]);
    __syncthreads();
    if (tid < nsub) cs[tid] = expf(cs[tid] - m);
    __syncthreads();
    if (tid < D) {
        float l = 0.0f, acc = 0.0f;
#pragma unroll 8
        for (int u = 0; u < nsub; u++) { const float c = cs[u]; l = l + cs[NSUB + u] * c; acc = acc + rec[u * 132 + 4 + tid] * c; }
        const float o = acc * (1.0f / l);
        ((float *)sg.y)[(size_t)t * D + tid] = o;
        if (sg.gs) q8_emit((void *)sg.gs, (size_t)t * D + tid, o);
    }
    __syncthreads();
}

// ---- the device accept (spec-drafters S3). The argmax key of the host rule (caminus.flashnext_tp._run: torch.max per card,
// the first index on ties, a NaN counts as the maximum; card 0 on a cross-card tie): any NaN above everything, +0 == -0.
__device__ __forceinline__ int df_hkey(float v) { return v != v ? INT_MAX : (v == 0.0f ? 0 : df_key(v)); }
// OP_DF_TOPK work 2 (argmax scan): task = (row r < T, chunk ch of DF_CHUNK ids) of logits [T, N] (Seg.a): the row's first
//   maximum in the chunk by df_hkey, written as (value, float(id + Seg.aux)) at Seg.y + (r * nch + ch) * 2 (nch = pad1).
__device__ __forceinline__ void df_argmax_scan(const Seg & sg, int t) {
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, nch = sg.pad1, row = t / nch, ch = t % nch, N = sg.N;
    __attribute__((address_space(3))) int * l = DF_LDS;
    const float * x = (const float *)sg.a + (size_t)row * N;
    const int b0 = ch * DF_CHUNK + wave * (32 * DF_NPT);
    int bk = INT_MIN, bi = INT_MAX; float bv = 0.0f;
#pragma unroll
    for (int j = 0; j < DF_NPT; j++) {
        const int i = b0 + j * 32 + lane;
        if (i < N) { const float v = x[i]; const int k = df_hkey(v); if (k > bk) { bk = k; bi = i; bv = v; } }   // ascending i per lane
    }
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) {
        const int ok = __shfl_xor(bk, m, 32), oi = __shfl_xor(bi, m, 32); const float ov = __shfl_xor(bv, m, 32);
        if (df_better(ok, oi, bk, bi)) { bk = ok; bi = oi; bv = ov; }
    }
    if (lane == 0) { l[3 * wave] = bk; l[3 * wave + 1] = bi; l[3 * wave + 2] = __builtin_bit_cast(int, bv); }
    __syncthreads();
    if (tid == 0) {
        int k = l[0], i = l[1], v = l[2];
        for (int w = 1; w < 8; w++) if (df_better(l[3 * w], l[3 * w + 1], k, i)) { k = l[3 * w]; i = l[3 * w + 1]; v = l[3 * w + 2]; }
        float * y = (float *)sg.y + (size_t)(row * nch + ch) * 2;
        y[0] = __builtin_bit_cast(float, v); y[1] = (float)(i + (int)(size_t)sg.aux);   // (i == INT_MAX: an empty chunk, never: nch * CHUNK covers N exactly)
    }
    __syncthreads();
}
// OP_DF_ACCEPT (one task, thread 0): sets the draft's position and anchor in the launch's parameter words (Seg.asum = int32
//   blk: [16] pos_i (the verify / inject's first position), [18] pos_d (written), [19] anchor (written in mode 1), [20] F,
//   [21 ..] the T tokens the verify ran at pos_i ..), from
//   work 0 (set): pos_d = pos_i + F (F = the host's draft offset in [1, T]; blk[19] = the host's anchor, checked);
//   work 1 (accept): the target's greedy token per verify row from both cards' argmax scans (Seg.a = (value, float id) pairs
//     [2 cards][T][nch], nch = K: card 0's chunks first, so the first maximum is the lowest id), then F forced tokens
//     (blk[20] in [1, T]) and j = the drafts accepted (tokens[F + j] == a[F - 1 + j]); the bonus a[F - 1 + j] is the anchor
//     at pos_d = pos_i + F + j.
//   Every value is clamped (F into [1, T], the anchor into [0, vocab = pad1), pos_d into [0, cap - T] with cap = pad2) and a
//   clamp sets the record's flag (2 also for any row's id a[r] outside [0, vocab): an exchanged id NaN / inf / out of range
//   reads as -1): the host refuses the launch's result (k3.ContextBoundsError). Record (Seg.y, int32):
//   [a[0 .. T - 1], j, bonus, pos_d, flag].
__device__ __forceinline__ void op_df_accept(const Seg & sg, int t) {
    __attribute__((address_space(3))) int * l = DF_LDS;   // mode 1: the pairs as (key, id) [2 T nch], then a [T]
    const int T = seg_T(sg), tid = K3TID;
    if (sg.work == 1) {   // the pairs into LDS (all threads), one thread per row ranks them in [card][chunk] order
        const int nch = sg.K, n = 2 * T * nch; const float * pr = (const float *)sg.a;
        // ids outside [0, vocab) (NaN, +-inf, negatives, huge: never cast) become -1: the ranking compares keys only, and a
        // winning -1 is flagged below (thread 0's range check of every a[r])
        for (int j = tid; j < n; j += blockDim.x) {
            const float fi = pr[2 * j + 1];
            l[2 * j] = df_hkey(pr[2 * j]); l[2 * j + 1] = (fi >= 0.0f && fi < (float)sg.pad1) ? (int)fi : -1;
        }
        __syncthreads();
        if (tid < T) {
            int bk = INT_MIN, bi = 0;
            for (int c = 0; c < 2; c++)
                for (int ch = 0; ch < nch; ch++) {
                    const int q = (c * T + tid) * nch + ch;
                    if (l[2 * q] > bk) { bk = l[2 * q]; bi = l[2 * q + 1]; }   // strictly greater: the first (lowest id) maximum
                }
            l[2 * n + tid] = bi;
        }
        __syncthreads();
    }
    if (tid == 0) {
        int * blk = (int *)sg.asum; int * rec = (int *)sg.y;
        const int vocab = sg.pad1, cap = sg.pad2, pi = blk[16];
        int flag = 0, F = blk[20], j = 0, anchor = blk[19];
        if (F < 1 || F > T) { flag |= 1; F = min(max(F, 1), T); }
        if (sg.work == 1) {
            const __attribute__((address_space(3))) int * a = l + 4 * T * sg.K;
            for (int r = 0; r < T; r++) { rec[r] = a[r]; if (a[r] < 0 || a[r] >= vocab) flag |= 2; }   // every row's id
            while (j < T - F && blk[21 + F + j] == a[F - 1 + j]) j++;
            anchor = a[F - 1 + j];
        } else {
            for (int r = 0; r < T; r++) rec[r] = -1;
        }
        if (anchor < 0 || anchor >= vocab) { flag |= 2; anchor = min(max(anchor, 0), vocab - 1); }
        int pd = pi + F + j;
        if (pd < 0 || pd > cap - T) { flag |= 4; pd = min(max(pd, 0), cap - T); }
        blk[18] = pd; blk[19] = anchor;
        rec[T] = j; rec[T + 1] = anchor; rec[T + 2] = pd; rec[T + 3] = flag;
        __threadfence();
    }
    __syncthreads();
}

// k3_extra_op's drafter branches: the flagged variants of trunk ops first in its chain (before their unflagged handlers),
// the drafter-only ops after the GG matvec dispatch (DF_DISPATCH_LATE; the placement moves the inlined code: the Q8_0
// multi-token matvec measured 28.8 - 37.3 us per 7680 x 2560 segment depending on what sits where in k3_run)
__device__ __forceinline__ bool k3_draft_flagged(const Seg & sg, int t) {
    if (sg.op == OP_ATTN_DEC && sg.pad2) {
        if (sg.pad2 & 4) op_attn_split(sg, t); else if (sg.pad2 & 8) op_attn_combine(sg, t);
        else if (sg.K == 128) op_attn_dec_df128(sg, t); else op_attn_dec_df(sg, t);
        return true;
    }
    if (sg.op == OP_KV_APPEND && sg.aux) { op_kv_append_ring(sg, t); return true; }
    if (sg.op == OP_RMS_MUL && sg.aux) { op_rms_mul_ll(sg, t); return true; }
    if (sg.op == OP_QK_NR && sg.aux) { op_qk_nr_ll(sg, t); return true; }
    return false;
}
__device__ __forceinline__ bool k3_draft_only(const Seg & sg, int t) {
    if (sg.op == OP_DF_MVCONV) { op_df_mvconv(sg, t); return true; }
    if (sg.op == OP_DF_CONV) { op_df_conv(sg, t); return true; }
    if (sg.op == OP_DF_TOPK) { op_df_topk(sg, t); return true; }
    if (sg.op == OP_DF_SELECT) { op_df_select(sg, t); return true; }
    if (sg.op == OP_DF_EMBED) { op_df_embed(sg, t); return true; }
    if (sg.op == OP_DF_ACCEPT) { op_df_accept(sg, t); return true; }
    return false;
}
