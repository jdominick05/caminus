// DFlash 2 drafter ops on Caminus's own code, for the dense qwen35 line's drafter module (kernels/k3_q35_dflash.hip;
// docs/dev/qwen38-27b.md). Included by kernels/k3_q35_ops.h only under K3_Q35_DF: no other code object contains any of
// this. The fp32 ops are kernels/k3_draft_ops.h's (the Coder's drafter, ggml-backed) with the ggml dependencies removed:
// the scratch in LDS of their own (not AQ's), the selector's Q8_0 rows decoded inline (half scale + 32 int8 values), no
// q8_1 copies (the matvecs here take fp16 activations: OP_OWN_MV16), the cache in the own modules' f16 layout. The op
// numbers and segment layouts are the Coder's (OP_DF_CONV / TOPK / SELECT), so the host's wrappers drive both.
// Graph reference: ignite-llama src/models/dflash.cpp (build_dflash2_conv, build_dflash2_selector) and
// common/speculative.cpp (draft-dflash: the greedy lattice walk); the float64 oracle: python/caminus/ref_dflash.py.
#ifndef DF_NPT
#define DF_NPT 16   // logits per lane in OP_DF_TOPK's scan (a chunk = 256 * DF_NPT = 4096 ids)
#endif
#define DF_CHUNK (256 * DF_NPT)
enum { OP_DF_CONV = 31, OP_DF_TOPK = 32, OP_DF_SELECT = 33 };

// the drafter ops' LDS (a workgroup runs one op at a time): 1287 ints, as the Coder's ops use (cand / unary / scores /
// cond, the top-k pairs, the attention chunk's scores + reductions)
__shared__ int DFO_LDS[1288];
#define DF_LDS ((__attribute__((address_space(3))) int *)DFO_LDS)

// the tape's token count: k3.Engine.build stamps T into every segment's team field on a T > 1 tape (k3_draft_ops.h's seg_T)
__device__ __forceinline__ int df_T(const Seg & sg) { return sg.team > 1 ? sg.team : 1; }
__device__ __forceinline__ float df_wsum(float v) {
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor(v, m, 32);
    return v;
}
__device__ __forceinline__ float df_wmax(float v) {
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) v = fmaxf(v, __shfl_xor(v, m, 32));
    return v;
}
// a Q8_0 block: an fp16 scale, then 32 int8 values (34 bytes)
__device__ __forceinline__ float df_q8_0(const uint8_t * row, int r) {
    const uint8_t * b = row + (size_t)(r / 32) * 34;
    return (float)(int8_t)b[2 + r % 32] * (float)__builtin_bit_cast(_Float16, (uint16_t)(b[0] | (b[1] << 8)));
}

// OP_DF_CONV: build_dflash2_conv for one block of T tokens: per channel c of group g = c / gs,
//   out[t, c] = sum_{tap < min(ks, T)} (dyn[t, g + ng * (tap + ks * side)] + base[c + C * (tap + ks * side)]) * h[t - tap, c]
//   (0 before the block), terms added in tap order; optionally + res[t, c]. Seg.a = h fp32 [T, C], Seg.gs = dyn fp32
//   [T, 2 * ks * ng], Seg.W = base fp32 [2][ks][C], Seg.y = out [T, C], Seg.aux = res or 0, Seg.asum = an fp16 copy of out
//   [T, C] or 0 (the next matvec's activations: OP_OWN_MV16 reads fp16); N = T * C, K = C, pad1 = side,
//   pad2 = gs, work = ks; a task = rows_per_task elements.
__device__ __forceinline__ void op_df_conv(const Seg & sg, int t) {
#pragma clang fp contract(off)
    const int C = sg.K, gs = sg.pad2, ng = C / gs, ks = sg.work, side = sg.pad1, T = df_T(sg);
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
        if (sg.asum) ((_Float16 *)sg.asum)[i] = (_Float16)acc;
    }
}

// ---- top-k with a total order: (value desc, id asc) (caminus.ref_dflash RefDFlash._topk's order)
__device__ __forceinline__ int df_key(float v) { const int b = __builtin_bit_cast(int, v); return b >= 0 ? b : b ^ 0x7FFFFFFF; }
__device__ __forceinline__ float df_val(int k) { return __builtin_bit_cast(float, k >= 0 ? k : k ^ 0x7FFFFFFF); }
__device__ __forceinline__ bool df_better(int ka, int ia, int kb, int ib) { return ka > kb || (ka == kb && ia < ib); }

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

// OP_DF_TOPK: the top K (<= 16) of logit rows. work 0 (scan): task = (row, chunk of DF_CHUNK ids); Seg.a = logits fp32
//   [rows, N] (row r at r * out_f32 when out_f32 > 1: a column slice of wider rows), Seg.y = int pairs (key, id) [rows - pad2]
//   [nchunk][K]; N = the ids scanned per row, K, pad1 = nchunk, pad2 = first row,
//   Seg.aux = the id offset. work 1 (merge): task = (row, part), pad1 = parts; Seg.a = int pairs [rows][N], Seg.y = float
//   pairs (value, float(id)) [rows][K]. kernels/k3_draft_ops.h's algorithm, unchanged.
__device__ __forceinline__ void op_df_topk(const Seg & sg, int t) {
    const int K = sg.K, tid = K3TID, lane = tid % 32, wave = tid / 32;
    __attribute__((address_space(3))) int * l = DF_LDS;
    if (sg.work == 0) {
        const int nch = sg.pad1, row = sg.pad2 + t / nch, ch = t % nch, N = sg.N, off = (int)(size_t)sg.aux;
        const float * x = (const float *)sg.a + (size_t)row * (sg.out_f32 > 1 ? sg.out_f32 : N);
        const int b0 = ch * DF_CHUNK + wave * (32 * DF_NPT);
        int key[DF_NPT]; unsigned taken = 0;   // (INT_MIN / INT_MAX: the padding key and ids, as the Coder's op)
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

// OP_DF_SELECT (one task): the selector and the greedy lattice walk for the k = N draft positions 1 .. N, as
//   kernels/k3_draft_ops.h (Seg layout unchanged): Seg.a = candidate float pairs [halves = pad1][N][KC], Seg.asum = int32
//   anchor id, Seg.W / Seg.gs = selector_predecessor / successor Q8_0 [vocab][rank], Seg.aux = gate fp32 [T][rank], Seg.y =
//   int32 out (drafts [N], candidates [N][KC], unary [N][KC], walked scores [N][KC]); K = rank (<= 768), pad2 = KC, work =
//   row bytes, out_f32 = vocabulary rows (ids clamped into [0, vocab)).
__device__ __forceinline__ void op_df_select(const Seg & sg, int t) {
#pragma clang fp contract(off)
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, N = sg.N, KC = sg.pad2, H = sg.pad1, rank = sg.K, rb = sg.work;
    __attribute__((address_space(3))) int * l = DF_LDS;
    int * out = (int *)sg.y;
    const float * cp = (const float *)sg.a;
    __attribute__((address_space(3))) int * in = l + 512;
    const int n = H * KC, tot = N * n;
    for (int j = tid; j < tot; j += blockDim.x) {
        const int w = j / n, r = j % n, hh = r / KC, s2 = r % KC;
        const float * p = cp + ((size_t)(hh * N + w) * KC + s2) * 2;
        const float fi = p[1];
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
        const int pid = min(max(i == 0 ? ((const int *)sg.asum)[0] : l[(i - 1) * KC + pr], 0), sg.out_f32 - 1);
        __attribute__((address_space(3))) float * cond = (__attribute__((address_space(3))) float *)(l + 512);
        const float * g = gate + (size_t)(i + 1) * rank;
        for (int r = tid; r < rank; r += blockDim.x) cond[r] = df_q8_0(Wp + (size_t)pid * rb, r) * g[r];
        __syncthreads();
        for (int s = wave; s < KC; s += 8) {
            const int sid = min(max(l[i * KC + s], 0), sg.out_f32 - 1);
            const uint8_t * srow = Ws + (size_t)sid * rb;
            float a = 0.0f;
#pragma unroll 1
            for (int m = 0; m < rank / 32; m++) a = a + df_q8_0(srow, m * 32 + lane) * cond[m * 32 + lane];
            a = df_wsum(a);
            if (lane == 0) sc[s] = a + __builtin_bit_cast(float, l[N * KC + i * KC + s]);
        }
        __syncthreads();
        if (tid == 0) {
            int b = 0;
            for (int s = 1; s < KC; s++) if (sc[b] < sc[s]) b = s;   // the first maximum
            sc[KC] = __builtin_bit_cast(float, b);
            out[i] = min(max(l[i * KC + b], 0), sg.out_f32 - 1);
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

// OP_KV_APPEND into the drafter's ring: the own f16 layout (per row: K [M] then V [M]), logical position p in row p & mask
//   (Seg.aux = mask, rows - 1, rows a power of two). Seg.asum = int32 first position, Seg.a / Seg.W = k / v fp32 [T, M],
//   Seg.y = the ring; N = M; task tt = token tt.
__device__ __forceinline__ void df_op_kv_append_ring(const Seg & sg, int tt) {
    const int pos = ((const int *)sg.asum)[0] + tt, M = sg.N;
    const uint32_t mask = (uint32_t)(size_t)sg.aux;
    _Float16 * c = (_Float16 *)sg.y + (size_t)((uint32_t)pos & mask) * 2 * M;
    for (int i = K3TID; i < M; i += blockDim.x) {
        c[i] = (_Float16)((const float *)sg.a)[(size_t)tt * M + i];
        c[M + i] = (_Float16)((const float *)sg.W)[(size_t)tt * M + i];
    }
}

// OP_ATTN_DEC with the drafter's flags (Seg.pad2 != 0): bit 0 = non-causal (every row of the pass sees positions up to
//   pos + T - 1, the whole block), bit 1 = the sliding window, Seg.aux = W (low 32 bits) | ring mask << 32. Row tt (position
//   p = pos + tt) attends over [lo, hi]: lo = max(0, p - (W - 1)) with the window (llama.cpp is_masked_swa STANDARD per
//   query, also non-causal), hi = pos + T - 1 non-causal. The f16 ring in the own layout. task h = (row tt, query head);
//   Seg.a = q fp32 [T, heads, D], Seg.W = the ring, Seg.asum = int32 first position, Seg.y = out fp32 [T, heads, D], Seg.gs =
//   its fp16 copy or 0 (the output matvec's activations);
//   N = T * heads, K = D (<= 512), pad1 = kv heads, rs = the scale. kernels/k3_draft_ops.h's chunked online
//   softmax and per-position order.
__device__ __forceinline__ void df_op_attn(const Seg & sg, int h) {
    constexpr int CH = 512;
    __attribute__((address_space(3))) float * s_ = (__attribute__((address_space(3))) float *)DF_LDS;   // [CH] + [8]
    __attribute__((address_space(3))) float * red = s_ + CH;
    const int T = df_T(sg), heads = sg.N / T, tt = h / heads;
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, D = sg.K, nkv = sg.pad1, g = (h - tt * heads) / (heads / nkv);
    const int pos0 = ((const int *)sg.asum)[0], p = pos0 + tt;
    const uint64_t ax = (uint64_t)(size_t)sg.aux;
    const int W = (int)(uint32_t)ax; const uint32_t mask = (uint32_t)(ax >> 32);
    const int hi = (sg.pad2 & 1) ? pos0 + T - 1 : p, lo = ((sg.pad2 & 2) && W > 0) ? max(0, p - (W - 1)) : 0;
    const int npos = hi - lo + 1, M = nkv * D;
    const float * q = (const float *)sg.a + (size_t)h * D;
    const _Float16 * cache = (const _Float16 *)sg.W;
    float run_m = -INFINITY, run_s = 0.0f, acc[2] = {0.0f, 0.0f};
#pragma unroll 1
    for (int c0 = 0; c0 < npos; c0 += CH) {
        const int n = min(CH, npos - c0);
        for (int t = wave; t < n; t += 8) {
            const uint32_t row = mask ? ((uint32_t)(lo + c0 + t) & mask) : (uint32_t)(lo + c0 + t);
            const _Float16 * kr = cache + (size_t)row * 2 * M + g * D;
            float a = 0.0f;
            for (int d = lane; d < D; d += 32) a += q[d] * (float)kr[d];
            a = df_wsum(a);
            if (lane == 0) s_[t] = a * sg.rs;
        }
        __syncthreads();
        float m = -INFINITY;
        for (int t = tid; t < n; t += blockDim.x) m = fmaxf(m, s_[t]);
        m = df_wmax(m); if (lane == 0) red[wave] = m; __syncthreads();
        m = red[0]; for (int w = 1; w < 8; w++) m = fmaxf(m, red[w]); __syncthreads();
        const float mn = fmaxf(run_m, m), corr = expf(run_m - mn);
        float sum = 0.0f;
        for (int t = tid; t < n; t += blockDim.x) { const float e = expf(s_[t] - mn); s_[t] = e; sum += e; }
        sum = df_wsum(sum); if (lane == 0) red[wave] = sum; __syncthreads();
        sum = 0.0f; for (int w = 0; w < 8; w++) sum += red[w];
        run_s = c0 == 0 ? sum : run_s * corr + sum;
        for (int j = 0; j < 2; j++) {
            const int d = tid + j * (int)blockDim.x;
            if (d >= D) break;
            float part = 0.0f;
            for (int t = 0; t < n; t++) {
                const uint32_t row = mask ? ((uint32_t)(lo + c0 + t) & mask) : (uint32_t)(lo + c0 + t);
                part += s_[t] * (float)cache[(size_t)row * 2 * M + M + g * D + d];
            }
            acc[j] = c0 == 0 ? part : acc[j] * corr + part;
        }
        run_m = mn;
        __syncthreads();
    }
    const float inv = 1.0f / run_s;
    for (int j = 0; j < 2; j++) {
        const int d = tid + j * (int)blockDim.x;
        if (d < D) {
            const float o = acc[j] * inv;
            ((float *)sg.y)[(size_t)h * D + d] = o;
            if (sg.gs) ((_Float16 *)sg.gs)[(size_t)h * D + d] = (_Float16)o;
        }
    }
}
