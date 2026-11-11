// The device sampler's candidates (python/caminus/device/topk.py): per row of a logits shard, the k largest values
// with their column ids, sorted by value descending and id ascending, and the raw logits of a list of penalty ids.
// Written for Caminus (no PyTorch, no llama.cpp): it replaces torch.topk / index_select in K3Session.queue_rows.
//
// A float maps to an unsigned key that orders like the float (NaN above +inf, as torch.topk ranks it; -0.0 just below
// +0.0). One launch: grid (chunks, rows), each workgroup one 2048-column chunk of one row.
//   chunk step: the chunk's keys go to LDS; a radix select (8 bits at a time from the top; one histogram per 32 threads,
//     then a parallel suffix sum finds the digit) gives its k-th key T; the keys above T and the lowest-id keys equal to
//     T are sorted (bitonic, key descending, id ascending) and written to the scratch lists.
//   merge step: the last workgroup of the row to finish (an atomic count per row; nothing waits) selects the row's k
//     from the chunks' lists the same way. A chunk's list is sorted, so the keys equal to the cut form one run in id
//     order in each list, and chunks are in id order: the lowest ids equal to T are taken chunk by chunk.
// The row's k largest are always inside the union of the chunks' k largest (with the lowest-id rule for ties), so the
// result equals a select over the whole row. Values are read back from the row (NaN payloads included). Every output is
// the same on every run. A first version (one workgroup per row, one shared histogram) took 289 us per call against
// torch.topk's 87 us at 1 row of 75,968 columns on the RX 7900 XT: one CU busy and every thread on the same few bins.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define TK_THREADS 256
#define TK_GROUPS (TK_THREADS / 32)   // histograms per workgroup (one per 32 threads: fewer threads on one bin)
#define TK_CHUNK 2048                 // columns per workgroup (the keys fit in LDS)
#define TK_KCAP 512                   // the largest k (the sort buffer); K3Session's candidate buffers are far narrower
#define TK_MAXCHUNKS 256              // chunks per row (n <= 524,288; the merge's per-chunk scan has one thread per chunk)

__device__ __forceinline__ uint32_t tk_key(float x) {
    const uint32_t u = __float_as_uint(x);
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu)) return 0xffffffffu;   // NaN: first
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

struct TkShared {
    uint32_t key[TK_CHUNK];          // the chunk's keys
    uint32_t ck[TK_KCAP];            // the candidates being sorted
    int ci[TK_KCAP];
    int hist[TK_GROUPS][256];
    int suf[256];
    int cnt[TK_MAXCHUNKS];
    int sel, need, heq, ngt, last;
};

// the k-th largest key among M keys (key_at(i), i < M): T, how many keys equal to T to take (need), how many equal T (neq)
template <class F>
__device__ void tk_radix(TkShared &s, F key_at, int M, int k, uint32_t &T, int &need, int &neq) {
    const int t = threadIdx.x;
    uint32_t prefix = 0u, mask = 0u;
    int nd = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int g = 0; g < TK_GROUPS; ++g) s.hist[g][t] = 0;
        __syncthreads();
        for (int i = t; i < M; i += TK_THREADS) {
            const uint32_t key = key_at(i);
            if ((key & mask) == prefix) atomicAdd(&s.hist[t >> 5][(key >> shift) & 255u], 1);
        }
        __syncthreads();
        int h = 0;
        for (int g = 0; g < TK_GROUPS; ++g) h += s.hist[g][t];
        s.suf[t] = h;
        __syncthreads();
        for (int off = 1; off < 256; off <<= 1) {        // suffix sums: suf[d] = keys with this digit d or above
            const int v = t + off < 256 ? s.suf[t + off] : 0;
            __syncthreads();
            s.suf[t] += v;
            __syncthreads();
        }
        const int above = t < 255 ? s.suf[t + 1] : 0;
        if (s.suf[t] >= nd && above < nd) { s.sel = t; s.need = nd - above; s.heq = h; }   // (exactly one digit)
        __syncthreads();
        const int d = s.sel;
        nd = s.need;
        prefix |= (uint32_t)d << shift;
        mask |= 255u << shift;
        if (shift == 0) neq = s.heq;                       // (all 32 bits fixed: the keys equal to T)
        __syncthreads();
    }
    T = prefix; need = nd;
}

// sort s.ck / s.ci [0, kk): key descending, id ascending (bitonic over the next power of two; padding sorts last)
__device__ void tk_sort(TkShared &s, int kk) {
    const int t = threadIdx.x;
    int N = 1;
    while (N < kk) N <<= 1;
    for (int i = kk + t; i < N; i += TK_THREADS) { s.ck[i] = 0u; s.ci[i] = 0x7fffffff; }
    __syncthreads();
    for (int size = 2; size <= N; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = t; i < N; i += TK_THREADS) {
                const int j = i ^ stride;
                if (j > i) {
                    const uint32_t ki = s.ck[i], kj = s.ck[j];
                    const int ii = s.ci[i], ij = s.ci[j];
                    const bool i_first = ki > kj || (ki == kj && ii < ij);
                    if (((i & size) == 0) != i_first) { s.ck[i] = kj; s.ck[j] = ki; s.ci[i] = ij; s.ci[j] = ii; }
                }
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ uint32_t tk_ld(const uint32_t *p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
__device__ __forceinline__ int tk_ld(const int *p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }

// L: the shard's first row; row r starts at L + r * ld. n: columns per row; k: candidates per row (k > n takes all n).
// val / idx: [rows, ldo] (float32 / int64). pid: npid column ids inside this shard, pen: [rows, ldp] their raw values.
// Scratch (the caller's, per Candidates): lk / li [rows, chunks, k] the chunks' lists, ln [rows, chunks] their lengths,
// done [rows] the finished-chunk counts (zero between launches: the merging workgroup resets its row's).
extern "C" __global__ void __launch_bounds__(TK_THREADS)
own_topk_rows(const float * __restrict__ L, long long ld, int n, int k,
              float * __restrict__ val, long long * __restrict__ idx, int ldo,
              const int * __restrict__ pid, int npid, float * __restrict__ pen, int ldp,
              uint32_t *lk, int *li, int *ln, unsigned int *done) {
    __shared__ TkShared s;
    const int t = threadIdx.x, c = blockIdx.x, r = blockIdx.y, C = gridDim.x;
    const float * __restrict__ row = L + (long long)r * ld;
    const int kk = k < n ? k : n;
    if (kk > TK_KCAP || kk <= 0 || C > TK_MAXCHUNKS) return;   // (the host wrapper refuses these before launching)

    if (c == 0)       // the penalty ids' raw logits (independent of the selection)
        for (int j = t; j < npid; j += TK_THREADS) pen[(long long)r * ldp + j] = row[pid[j]];

    // ---- chunk step
    const int a = c * TK_CHUNK, len = min(TK_CHUNK, n - a), kc = min(k, len);
    for (int i = t; i < len; i += TK_THREADS) s.key[i] = tk_key(row[a + i]);
    if (t == 0) s.ngt = 0;
    __syncthreads();
    if (kc < len) {
        uint32_t T; int need, neq = 0;
        tk_radix(s, [&](int i) { return s.key[i]; }, len, kc, T, need, neq);
        for (int i = t; i < len; i += TK_THREADS) {
            const uint32_t key = s.key[i];
            if (key > T || (key == T && neq == need)) { const int q = atomicAdd(&s.ngt, 1); s.ck[q] = key; s.ci[q] = a + i; }
        }
        __syncthreads();
        if (neq > need) {      // ties at the chunk's cut: its `need` lowest ids equal to T (8 keys per thread, in order)
            const int per = (len + TK_THREADS - 1) / TK_THREADS, b0 = t * per, b1 = min(b0 + per, len);
            int m = 0;
            for (int i = b0; i < b1; ++i) m += s.key[i] == T;
            s.suf[t] = m;
            __syncthreads();
            if (t == 0) { int acc = 0; for (int j = 0; j < TK_THREADS; ++j) { const int x = s.suf[j]; s.suf[j] = acc; acc += x; } }
            __syncthreads();
            int rank = s.suf[t];
            const int base = s.ngt;
            for (int i = b0; i < b1 && rank < need; ++i)
                if (s.key[i] == T) { s.ck[base + rank] = T; s.ci[base + rank] = a + i; ++rank; }
            __syncthreads();
        }
    } else {
        for (int i = t; i < len; i += TK_THREADS) { s.ck[i] = s.key[i]; s.ci[i] = a + i; }
        __syncthreads();
    }
    tk_sort(s, kc);
    const long long lb = ((long long)r * C + c) * k;
    for (int j = t; j < kc; j += TK_THREADS) { lk[lb + j] = s.ck[j]; li[lb + j] = s.ci[j]; }
    if (t == 0) ln[r * C + c] = kc;

    // ---- the last workgroup of the row merges
    __threadfence();
    __syncthreads();
    if (t == 0) s.last = atomicAdd(&done[r], 1u) == (unsigned int)(C - 1);
    __syncthreads();
    if (!s.last) return;
    __threadfence();
    const uint32_t *rk = lk + (long long)r * C * k;
    const int *ri = li + (long long)r * C * k, *rn = ln + r * C;
    if (t < C) s.cnt[t] = tk_ld(rn + t);
    if (t == 0) s.ngt = 0;
    __syncthreads();
    // the lists flattened: entry i = (chunk i / k, position i % k); positions past a list's length read as key 0 (below
    // every real key: the smallest, -inf's, is 0x007fffff), never selected because the lists hold >= kk real keys
    uint32_t T; int need, neq = 0;
    tk_radix(s, [&](int i) { const int q = i / k, j = i - q * k; return j < s.cnt[q] ? tk_ld(rk + i) : 0u; }, C * k, kk, T, need, neq);
    for (int i = t; i < C * k; i += TK_THREADS) {
        const int q = i / k, j = i - q * k;
        if (j >= s.cnt[q]) continue;
        const uint32_t key = tk_ld(rk + i);
        if (key > T || (key == T && neq == need)) { const int p = atomicAdd(&s.ngt, 1); s.ck[p] = key; s.ci[p] = tk_ld(ri + i); }
    }
    __syncthreads();
    if (neq > need) {          // ties at the row's cut: chunk by chunk, each list's run of T in id order
        int m = 0, first = 0;
        if (t < C) {
            for (int j = 0; j < s.cnt[t]; ++j) { const uint32_t key = tk_ld(rk + t * k + j); if (key == T) { if (!m) first = j; ++m; } }
        }
        __syncthreads();
        if (t < C) s.suf[t] = m;                            // (C <= 256: suf is free after tk_radix)
        __syncthreads();
        if (t == 0) { int acc = 0; for (int j = 0; j < C; ++j) { const int x = s.suf[j]; s.suf[j] = acc; acc += x; } }
        __syncthreads();
        if (t < C) {
            const int base = s.ngt;
            for (int j = 0, rank = s.suf[t]; j < m && rank < need; ++j, ++rank) {
                s.ck[base + rank] = T; s.ci[base + rank] = tk_ld(ri + t * k + first + j);
            }
        }
        __syncthreads();
    }
    tk_sort(s, kk);
    for (int j = t; j < kk; j += TK_THREADS) {
        const int col = s.ci[j];
        val[(long long)r * ldo + j] = row[col];
        idx[(long long)r * ldo + j] = col;
    }
    if (t == 0) __hip_atomic_store(&done[r], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
