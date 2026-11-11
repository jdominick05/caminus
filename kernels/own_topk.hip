// The device sampler's candidates (python/caminus/device/topk.py): per row of a logits shard, the k largest values
// with their column ids, sorted by value descending and id ascending, and the raw logits of a list of penalty ids.
// Written for Caminus (no PyTorch, no llama.cpp): it replaces torch.topk / index_select in K3Session.queue_rows.
//
// One workgroup per row. A float maps to an unsigned key that orders like the float (NaN above +inf, as torch.topk ranks
// it; -0.0 just below +0.0). A radix select over the keys, 8 bits at a time from the top, finds the k-th largest key T in
// four histogram passes; the elements above T are collected, then as many elements equal to T as are still needed, the
// lowest ids first (an ordered pass that runs only when more elements equal T than are needed). The k pairs are sorted in
// LDS (bitonic) and written with the values read back from the row, so a value is the row's own bits, NaN payload included.
// Every output is the same on every run: the sort fixes the order and the tie rule fixes the set.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define TK_THREADS 256
#define TK_KCAP 1024              // the largest k (the session's candidate buffers are far narrower: K3Session.kmax)

__device__ __forceinline__ uint32_t tk_key(float x) {
    const uint32_t u = __float_as_uint(x);
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu)) return 0xffffffffu;   // NaN: first
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// L: the shard's first row; row r starts at L + r * ld. n: columns per row; k: candidates per row (k > n takes all n).
// val / idx: [rows, ldo] (float32 / int64). pid: npid column ids inside this shard, pen: [rows, ldp] their raw values.
extern "C" __global__ void __launch_bounds__(TK_THREADS)
own_topk_rows(const float * __restrict__ L, long long ld, int n, int k,
              float * __restrict__ val, long long * __restrict__ idx, int ldo,
              const int * __restrict__ pid, int npid, float * __restrict__ pen, int ldp) {
    __shared__ uint32_t s_key[TK_KCAP];
    __shared__ int s_id[TK_KCAP];
    __shared__ int s_hist[256];
    __shared__ int s_cnt[TK_THREADS];
    __shared__ uint32_t s_prefix, s_mask;
    __shared__ int s_need, s_ngt, s_neq_all;

    const int t = threadIdx.x, r = blockIdx.x;
    const float * __restrict__ row = L + (long long)r * ld;
    const int kk = k < n ? k : n;
    if (kk > TK_KCAP || kk <= 0) return;          // (the host wrapper refuses these before launching)

    // the penalty ids' raw logits (independent of the selection)
    for (int j = t; j < npid; j += TK_THREADS) pen[(long long)r * ldp + j] = row[pid[j]];

    if (t == 0) { s_prefix = 0u; s_mask = 0u; s_need = kk; s_ngt = 0; s_neq_all = 0; }
    __syncthreads();

    if (kk < n) {
        // ---- the k-th largest key: four 8-bit digits from the top
        for (int shift = 24; shift >= 0; shift -= 8) {
            s_hist[t] = 0;                          // (TK_THREADS == 256 bins)
            __syncthreads();
            const uint32_t prefix = s_prefix, mask = s_mask;
            for (int i = t; i < n; i += TK_THREADS) {
                const uint32_t key = tk_key(row[i]);
                if ((key & mask) == prefix) atomicAdd(&s_hist[(key >> shift) & 255u], 1);
            }
            __syncthreads();
            if (t == 0) {
                int need = s_need, d = 255;
                for (; d > 0; --d) {
                    if (s_hist[d] >= need) break;
                    need -= s_hist[d];
                }
                s_need = need;                       // (of the elements whose key has this digit: how many to take)
                s_prefix = prefix | ((uint32_t)d << shift);
                s_mask = mask | (255u << shift);
                if (shift == 0) s_neq_all = s_hist[d];   // (elements whose key equals T exactly)
            }
            __syncthreads();
        }
        const uint32_t T = s_prefix;
        const int need = s_need, neq = s_neq_all;   // take `need` of the `neq` elements equal to T

        // ---- the elements above T (exactly kk - need of them), and all those equal to T when none is left out
        for (int i = t; i < n; i += TK_THREADS) {
            const uint32_t key = tk_key(row[i]);
            if (key > T || (key == T && neq == need)) {
                const int s = atomicAdd(&s_ngt, 1);
                s_key[s] = key; s_id[s] = i;
            }
        }
        __syncthreads();
        if (neq > need) {
            // ---- ties at the cut: the `need` lowest ids equal to T (contiguous chunks per thread, then a scan)
            const int c = (n + TK_THREADS - 1) / TK_THREADS, a = t * c, b = min(a + c, n);
            int m = 0;
            for (int i = a; i < b; ++i) m += tk_key(row[i]) == T;
            s_cnt[t] = m;
            __syncthreads();
            if (t == 0) { int acc = 0; for (int j = 0; j < TK_THREADS; ++j) { const int x = s_cnt[j]; s_cnt[j] = acc; acc += x; } }
            __syncthreads();
            int rank = s_cnt[t];
            const int base = s_ngt;                  // (= kk - need: the elements above T)
            for (int i = a; i < b && rank < need; ++i)
                if (tk_key(row[i]) == T) { s_key[base + rank] = T; s_id[base + rank] = i; ++rank; }
            __syncthreads();
        }
    } else {
        // ---- k >= n: every column is a candidate
        for (int i = t; i < n; i += TK_THREADS) { s_key[i] = tk_key(row[i]); s_id[i] = i; }
        __syncthreads();
    }

    // ---- sort the kk pairs: key descending, id ascending (bitonic over the next power of two; padding sorts last)
    int N = 1;
    while (N < kk) N <<= 1;
    for (int i = kk + t; i < N; i += TK_THREADS) { s_key[i] = 0u; s_id[i] = 0x7fffffff; }
    __syncthreads();
    for (int size = 2; size <= N; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = t; i < N; i += TK_THREADS) {
                const int j = i ^ stride;
                if (j > i) {
                    const uint32_t ki = s_key[i], kj = s_key[j];
                    const int ii = s_id[i], ij = s_id[j];
                    const bool i_first = ki > kj || (ki == kj && ii < ij);   // the output order
                    const bool up = (i & size) == 0;                           // this pair's half sorts toward i_first
                    if (up != i_first) { s_key[i] = kj; s_key[j] = ki; s_id[i] = ij; s_id[j] = ii; }
                }
            }
            __syncthreads();
        }
    }

    // ---- out: the row's own values (NaN payloads included) and their ids
    for (int j = t; j < kk; j += TK_THREADS) {
        const int c = s_id[j];
        val[(long long)r * ldo + j] = row[c];
        idx[(long long)r * ldo + j] = c;
    }
}
