// The batched prefill's attention core (docs/agentic/spec-prefill-cache.md M5 S6, milestone A: dense attention up to
// capacity.QSA_EXACT = 2051 positions; host side python/caminus/prefill_attn.py): the per-token K3 decode ops of an
// attention layer's core run over a chunk of rows with ordinary launches (not K3). Each kernel computes every element
// with the expressions of its K3 decode op (kernels/k3_gguf_ops.h, kernels/k3_attn_split.h) in the same order, so it is
// BITWISE that op on the same inputs (gate G1, tests/test_prefill_attn.py, against the K3 ops through caminus.k3.Engine on
// the Coder's trunk module, one token per pass):
//   pf_qk_nr        OP_QK_NR: per (row, head) the RMS norm of the head (one value per thread, the warp butterfly, the 8
//                   wave partials in order, rsqrtf) times the norm weight, then NEOX RoPE on the first n_rot dims at the
//                   row's position (powf / cosf / sinf as the decode op; the same default contraction: the op has no
//                   contract pragma, so its source lines are copied as they are)
//   pf_hadamard     OP_HADAMARD: in place, the orthonormal Walsh-Hadamard transform of chunks of n values (q / k over the
//                   head dim, v and the attention output in blocks of 64: llama.cpp's attn_rot before a q8_0 KV cache)
//   pf_kv_append    OP_KV_APPEND's q8_0 rows (ggml's quantize_f32_q8_0_block, one thread per block of 32): K3's KV cache
//                   rows [K | V] for every row of the chunk; also the QSA indexer's raw key rows (IDXRAW: M = 64, the K and
//                   V halves of one 128-wide key, what the dense tape writes). Rows outside the cache are refused on the
//                   host; the device clamps nothing silently: a row past `rows` sets the error word and is not written
//   pf_attn_part    OP_ATTN_PART over the identity list 0 .. pos (the dense regime: every position below 2051 sees all its
//                   cells, as the decode's dense tape and the QSA tape's identity selection): one workgroup per (split s,
//                   row, kv head), the split's 32 cells, the same lane / wave map, butterflies, softmax and V order
//   pf_attn_comb    OP_ATTN_COMB: per (row, query head) the splits merged in ascending order
//                   The records [rows][heads][NS][260] are bounded per launch pair (the host runs a chunk's rows in blocks:
//                   K3's record layout for 2048 rows would be ~1.7 GB per card), and a row's splits depend only on its own
//                   list length, so the blocks change no bit (G2b)
//   pf_gate_mul     OP_GATE_MUL: out = a * sigmoid(the head's gate half) (the attention output into attn_output's MMQ)
//   pf_hc_avg       OP_HC_AVG: the mean over the hc streams of the residual (the DFlash drafter's target features)
// The QSA prompt path (M5 S7: rows at positions >= capacity.QSA_EXACT = 2051, where the model's indexer selects; the decode's
// ops are kernels/k3_qsa_ops.h and k3_attn_split.h on the QSA modules):
//   pf_idx_pool     OP_IDX_APPEND's pooling: IDXPOOL[j] from the four IDXRAW rows 4j .. 4j + 3 (dequantized, summed in
//                   order, x 0.25, RMS norm * k_norm, NEOX RoPE of 64 dims at position 4j, q8_0), one wave per pool, the
//                   decode op's expressions (its members from this pass are re-quantized from their k_raw rows there: the
//                   same bytes the IDXRAW row holds, so reading the row back gives the same values)
//   pf_qsa_score    OP_QSA_SCORE: SCORE[r][j] for the pools j < nv of row r of a query sub-block: per head in order the
//                   lane's 4-dim fma chain, the xor butterfly (wred), relu, s = fma(sum, w, s); the 4 pools of a wave
//                   loaded once for a group of rows (block = (32-pool chunk, row group))
//   pf_qsa_select   OP_QSA_SELECT: per row the 512 pools with the largest scores, the lowest index first among ties, ascending
//                   (a 4-pass 8-bit radix select of the threshold key, then one ascending pass), the same header words
//   pf_attn_part_sel / pf_attn_comb_sel  the split pair over each row's SEL list (its selected pools' cells ascending, then the
//                   tail 4 nv .. pos): attn_list / attn_cell of k3_attn_split.h; the arithmetic is pf_attn_part / comb's (one
//                   template), only the cell of a list position changes, as OP_ATTN_PART's with Seg.gs = SEL
// Compiled by build.sh's generic loop like the other non-K3 objects; the decode code objects are untouched.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define ATTN_SS 32                    // list positions per split (= kernels/k3_attn_split.h, caminus.k3.ATTN_SS)
#define ATTN_CPR 2                    // cells per wave per scoring round (k3_attn_split.h's default; the Coder's modules)
#define ATTN_VH 12                    // query heads per V pass
#define ATTN_G 12                     // query heads per kv head
#define ATTN_D 256                    // head dim
#define ATTN_RS 260                   // floats per partial record: m, l, 0, 0, acc[256]
#define QSA_ROWB 136                  // one 128-value indexer row: 4 x block_q8_0 (= kernels/k3_qsa_ops.h)
#define QSA_TOPK 512                  // pools selected
#define QSA_SELW 516                  // ints per SEL row: 4 header words + QSA_TOPK pool ids

struct block_q8_0 { half d; int8_t qs[32]; };                 // ggml's block_q8_0 (34 bytes)
static_assert(sizeof(block_q8_0) == 34, "block_q8_0");

// ---- helpers, copied from kernels/k3_gguf_ops.h / k3_attn_split.h (same code, so the same instructions and bits) -----------
template <bool MAX>
__device__ __forceinline__ float wred(float x) {
    const unsigned u = __builtin_bit_cast(unsigned, x);
    float y = __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(u, u, 0x76543210u, 0xfedcba98u, false, false));
    x = MAX ? fmaxf(x, y) : x + y;
#pragma unroll
    for (int m = 8; m >= 1; m >>= 1) {   // row_xmask:m (DPP16 0x160 | m): lane ^ m within each row of 16
        const int xi = __builtin_bit_cast(int, x);
        int yi;
        switch (m) {
            case 8: yi = __builtin_amdgcn_update_dpp(xi, xi, 0x168, 0xf, 0xf, false); break;
            case 4: yi = __builtin_amdgcn_update_dpp(xi, xi, 0x164, 0xf, 0xf, false); break;
            case 2: yi = __builtin_amdgcn_update_dpp(xi, xi, 0x162, 0xf, 0xf, false); break;
            default: yi = __builtin_amdgcn_update_dpp(xi, xi, 0x161, 0xf, 0xf, false); break;
        }
        y = __builtin_bit_cast(float, yi);
        x = MAX ? fmaxf(x, y) : x + y;
    }
    return x;
}
// ggml-cuda's warp_reduce_sum<32> (common.cuh: x += shfl_xor(x, 16, 8, 4, 2, 1)): OP_QK_NR's reduction. fp32 addition is
// commutative, so any lane pairing in this order gives its bits
__device__ __forceinline__ float warp_sum32(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) x += __shfl_xor(x, offset, 32);
    return x;
}
__device__ __forceinline__ float attn_x16(float x) {
    const unsigned u = __builtin_bit_cast(unsigned, x);
    return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(u, u, 0x76543210u, 0xfedcba98u, false, false));
}
template <int M> __device__ __forceinline__ float attn_xm(float x) {
    const int xi = __builtin_bit_cast(int, x);
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(xi, xi, 0x160 | M, 0xf, 0xf, false));
}
__device__ __forceinline__ size_t kv_pos_bytes(int M) { return (size_t)2 * (M / 32) * sizeof(block_q8_0); }

// ---- pf_qk_nr: OP_QK_NR over rows: block t = row * heads + head (the decode's task index) ---------------------------------
//   x: [rows, heads] heads at head_stride floats (the q GEMM's [q | gate] head pairs: stride 2 D), w: the norm [D], out
//   [rows, heads, D]; position of row r: pos0 + r. D <= 512, 256 threads
extern "C" __global__ void __launch_bounds__(256) pf_qk_nr(const float * xin, int head_stride, int heads, const float * w, int D,
                                                           int nrot, float theta_scale, float eps, int pos0, float * out) {
    __shared__ float part[8]; __shared__ float nrm[512];
    const int t = blockIdx.x;
    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    const float * x = xin + (size_t)t * head_stride;
    float * y = out + (size_t)t * D;
    float ss = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) ss += x[i] * x[i];
    ss = warp_sum32(ss); if (lane == 0) part[wave] = ss; __syncthreads();
    float tot = 0.0f; for (int q = 0; q < 8; q++) tot += part[q];
    const float scale = rsqrtf(tot / D + eps);
    for (int i = tid; i < D; i += blockDim.x) nrm[i] = (scale * x[i]) * w[i];
    __syncthreads();
    const float pos = (float)(pos0 + t / heads);
    for (int i = tid; i < D; i += blockDim.x) {
        if (i >= nrot) { y[i] = nrm[i]; continue; }
        const int j = i < nrot / 2 ? i : i - nrot / 2;
        const float theta = pos * powf(theta_scale, (float)j);
        const float c = cosf(theta), sn = sinf(theta);
        const float x0 = nrm[j], x1 = nrm[j + nrot / 2];
        y[i] = i < nrot / 2 ? x0 * c - x1 * sn : x0 * sn + x1 * c;
    }
}

// ---- pf_hadamard: OP_HADAMARD in place over C chunks of n values (64 <= n <= 256), rpb chunks per block -------------------
extern "C" __global__ void __launch_bounds__(256) pf_hadamard(float * x, int C, int n, int rpb) {
#pragma clang fp contract(off)
    __shared__ float hb[256];
    const int tid = threadIdx.x;
    const float scale = 1.0f / sqrtf((float)n);
    const int c0 = blockIdx.x * rpb, c1 = min(C, c0 + rpb);
    for (int c = c0; c < c1; c++) {
        float * v = x + (size_t)c * n;
        for (int i = tid; i < n; i += blockDim.x) hb[i] = v[i];
        __syncthreads();
        for (int h = 1; h < n; h <<= 1) {
            for (int k = tid; k < n / 2; k += blockDim.x) {
                const int i = (k / h) * 2 * h + k % h;
                const float a = hb[i], b = hb[i + h];
                hb[i] = a + b; hb[i + h] = a - b;
            }
            __syncthreads();
        }
        for (int i = tid; i < n; i += blockDim.x) v[i] = hb[i] * scale;
        __syncthreads();
    }
}

// ---- pf_kv_append: q8_0 rows [K | V] (M values each) at cache rows pos0 + r for the rows' k / v (row strides ldk / ldv) ----
//   one block per row; a row at or past `rows` (the cache's positions) is not written and sets err bit 0
extern "C" __global__ void __launch_bounds__(256) pf_kv_append(const float * k, int ldk, const float * v, int ldv, int M,
                                                               uint8_t * cache, int pos0, int rows, int * err) {
    const int tt = blockIdx.x, pos = pos0 + tt;
    if (pos < 0 || pos >= rows) { if (threadIdx.x == 0) atomicOr(err, 1); return; }
    uint8_t * base = cache + (size_t)pos * kv_pos_bytes(M);
    const int nb = M / 32;
    for (int b = threadIdx.x; b < 2 * nb; b += blockDim.x) {     // one thread per block, as the reference loop
        const float * x = (b < nb ? k + (size_t)tt * ldk : v + (size_t)tt * ldv) + (size_t)(b % nb) * 32;
        block_q8_0 * y = (block_q8_0 *)base + b;
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[j]));
        const float d = amax / ((1 << 7) - 1), id = d ? 1.0f / d : 0.0f;
        y->d = d;
        for (int j = 0; j < 32; j++) y->qs[j] = roundf(x[j] * id);
    }
}

// ---- pf_attn_part / pf_attn_comb: the split attention pair over the identity list, rows [0, R) of a block -----------------
//   Row r of the block is at position pos0 + r and attends to cells 0 .. pos0 + r (n = pos + 1, capped at NS * ATTN_SS and
//   at the cache's rows: a row past them sets err bit 1 and reads only cells < rows). q fp32 [R][heads][256] (heads =
//   nkv * 12), the cache's q8_0 rows (kv_pos_bytes(nkv * 256)), rec fp32 [R][heads][NS][260]; out fp32 [R][heads][256].
//   pf_attn_part: block t = (split s, row r, kv head g), s major (t = s * R * nkv + r * nkv + g), as OP_ATTN_PART's tasks
__device__ __forceinline__ int list_n(int pos, int NS, int rows) {
    return min(min(pos + 1, NS * ATTN_SS), rows);
}
// a row's cell list (k3_attn_split.h attn_list / attn_cell): the identity 0 .. pos (SEL false: list_n's length), or the row's
// SEL row (OP_QSA_SELECT's: [0] count, ids from [4]): the cells 4 s .. 4 s + 3 of its selected pools s ascending, then the tail
// 4 nv .. pos (nv = (pos + 1) / 4). The clamps are the decode's (the count to [0, min(nv, 512)], every pool id to [0, nv - 1]:
// every cell <= pos); a row at or past the cache's rows has an empty list (err bit 1 is set by the caller)
struct PfList { int n, np4, tail0, nvc; const int * pl; };
template <bool SEL>
__device__ __forceinline__ PfList pf_list(int pos, int NS, int rows, const int * row) {
    PfList L; L.np4 = 0; L.tail0 = 0; L.nvc = 0; L.pl = nullptr;
    if (!SEL) { L.n = list_n(pos, NS, rows); return L; }
    if (pos < 0 || pos >= rows) { L.n = 0; return L; }
    const int nv = (pos + 1) >> 2;
    const int np = min(max(row[0], 0), min(nv, (int)QSA_TOPK));
    L.n = pos + 1;
    if (np != nv) { L.np4 = 4 * np; L.tail0 = 4 * nv; L.nvc = nv - 1; L.pl = row + 4; L.n = L.np4 + (pos + 1 - L.tail0); }
    L.n = min(L.n, NS * ATTN_SS);
    return L;
}
__device__ __forceinline__ int pf_cell(const PfList & L, int k) {
    return k < L.np4 ? 4 * min(max(L.pl[k >> 2], 0), L.nvc) + (k & 3) : k - L.np4 + L.tail0;
}
template <bool SEL>
__device__ __forceinline__ void attn_part_body(const float * q, const uint8_t * cache, int pos0, int R, int nkv, int NS, float rs,
                                               int rows, float * recs, int * err, const int * sel) {
#pragma clang fp contract(off)
    __shared__ float sp[ATTN_SS * ATTN_G];                       // scores, then p
    __shared__ int cl[ATTN_SS + 1];                              // the split's cells, then their count
    const int t = blockIdx.x, heads = nkv * ATTN_G;
    const int s = t / (R * nkv), rr = t - s * (R * nkv), tt = rr / nkv, g = rr - tt * nkv;
    const int pos = pos0 + tt;
    if (pos >= rows && threadIdx.x == 0 && s == 0 && g == 0) atomicOr(err, 2);
    const PfList L = pf_list<SEL>(pos, NS, rows, SEL ? sel + (size_t)tt * QSA_SELW : nullptr);
    const int n = L.n;
    const int k0 = s * ATTN_SS;
    if (k0 >= n) return;                                         // (uniform: a split past the row's list does nothing)
    {
        const int nb = min(ATTN_SS, n - k0), tid = threadIdx.x;
        __syncthreads();
        if (tid < nb) cl[tid] = SEL ? pf_cell(L, k0 + tid) : k0 + tid;   // the identity list: list position k is cell k
        if (tid == 0) cl[ATTN_SS] = nb;
        __syncthreads();
    }
    {   // scores
        const int tid = threadIdx.x, lane = tid % 32, wave = (int)__builtin_amdgcn_readfirstlane(tid / 32);
        const int nb = __builtin_amdgcn_readfirstlane(cl[ATTN_SS]);
        const size_t pb = kv_pos_bytes(nkv * ATTN_D);
        const float * qb = q + ((size_t)tt * heads + (size_t)g * ATTN_G) * ATTN_D + 8 * lane;
        const uint8_t * kb = cache + (size_t)(g * 8 + (lane >> 2)) * sizeof(block_q8_0);
        const int ko = 2 + (lane & 3) * 8, ci = ATTN_CPR == 2 ? (lane >> 4) & 1 : 0;
#pragma unroll 1
        for (int cb = 4 * wave; cb < min(nb, 4 * wave + 4); cb += ATTN_CPR) {
            float kf[ATTN_CPR][8];
#pragma unroll
            for (int i = 0; i < ATTN_CPR; i++) {
                if (cb + i < nb) {
                    const uint8_t * kp = kb + (size_t)cl[cb + i] * pb;
                    const float d = __half2float(*(const half *)kp);
                    const uint16_t * qs = (const uint16_t *)(kp + ko);
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const uint16_t w = qs[j];
                        kf[i][2 * j] = (float)(int8_t)(w & 0xff) * d; kf[i][2 * j + 1] = (float)(int8_t)(w >> 8) * d;
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
                const bool h16 = lane & 16;
                float u = (h16 ? a[1] : a[0]) + attn_x16(h16 ? a[0] : a[1]);
                u = u + attn_xm<8>(u); u = u + attn_xm<4>(u); u = u + attn_xm<2>(u); u = u + attn_xm<1>(u);
                if ((lane & 15) == 0 && cb + ci < nb) sp[(cb + ci) * ATTN_G + h] = u * rs;
            }
        }
    }
    __syncthreads();
    float * rec = recs + ((size_t)(tt * heads + g * ATTN_G) * NS + s) * ATTN_RS;
    {   // softmax (wave w: heads w, w + 8): p into LDS, m and l into the records
        const int nb = __builtin_amdgcn_readfirstlane(cl[ATTN_SS]), lane = threadIdx.x % 32, wave = (int)__builtin_amdgcn_readfirstlane(threadIdx.x / 32);
#pragma unroll 1
        for (int h = wave; h < ATTN_G; h += 8) {
            const float v = lane < nb ? sp[lane * ATTN_G + h] : -INFINITY;
            const float m = wred<true>(v);
            const float p = lane < nb ? expf(v - m) : 0.0f;
            if (lane < nb) sp[lane * ATTN_G + h] = p;
            const float l = wred<false>(p);
            if (lane == 0) { float * rh = rec + (size_t)h * NS * ATTN_RS; rh[0] = m; rh[1] = l; }
        }
    }
    __syncthreads();
    {   // V: this thread's value of each cell (its q8_0 block and qs byte)
        const int nb = __builtin_amdgcn_readfirstlane(cl[ATTN_SS]), tid = threadIdx.x;
        const size_t pb = kv_pos_bytes(nkv * ATTN_D);
        const uint8_t * vb = cache + (size_t)(nkv * 8 + ((g * ATTN_D + tid) >> 5)) * sizeof(block_q8_0);
        const int vo = 2 + (tid & 31);
#pragma unroll 1
        for (int h0 = 0; h0 < ATTN_G; h0 += ATTN_VH) {
            float acc[ATTN_VH];
#pragma unroll
            for (int h = 0; h < ATTN_VH; h++) acc[h] = 0.0f;
#pragma unroll 1
            for (int c = 0; c < nb; c++) {
                const uint8_t * vp = vb + (size_t)cl[c] * pb;
                const float v = (float)(int8_t)vp[vo] * __half2float(*(const half *)vp);
#pragma unroll
                for (int h = 0; h < ATTN_VH; h++) acc[h] = __builtin_fmaf(sp[c * ATTN_G + h0 + h], v, acc[h]);
            }
#pragma unroll
            for (int h = 0; h < ATTN_VH; h++) rec[(size_t)(h0 + h) * NS * ATTN_RS + 4 + tid] = acc[h];
        }
    }
}
extern "C" __global__ void __launch_bounds__(256) pf_attn_part(const float * q, const uint8_t * cache, int pos0, int R, int nkv,
                                                               int NS, float rs, int rows, float * recs, int * err) {
    attn_part_body<false>(q, cache, pos0, R, nkv, NS, rs, rows, recs, err, nullptr);
}
//   pf_attn_part_sel: the same over the SEL rows sel [R][QSA_SELW] of the block's rows (QSA rows: every list <= 2051 cells)
extern "C" __global__ void __launch_bounds__(256) pf_attn_part_sel(const float * q, const uint8_t * cache, int pos0, int R, int nkv,
                                                                   int NS, float rs, int rows, float * recs, int * err, const int * sel) {
    attn_part_body<true>(q, cache, pos0, R, nkv, NS, rs, rows, recs, err, sel);
}

//   pf_attn_comb: block q = (row r, query head h) = r * heads + h; thread d
template <bool SEL>
__device__ __forceinline__ void attn_comb_body(const float * recs, int pos0, int heads, int NS, int rows, float * out, const int * sel) {
#pragma clang fp contract(off)
    const int q = blockIdx.x, tt = q / heads, tid = threadIdx.x;
    const int n = pf_list<SEL>(pos0 + tt, NS, rows, SEL ? sel + (size_t)tt * QSA_SELW : nullptr).n;
    float * o = out + (size_t)q * ATTN_D;
    if (n <= 0) { o[tid] = 0.0f; return; }
    const int ns = (n + ATTN_SS - 1) / ATTN_SS;
    const float * rec = recs + (size_t)q * NS * ATTN_RS;
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
    o[tid] = acc * (1.0f / l);
}
extern "C" __global__ void __launch_bounds__(256) pf_attn_comb(const float * recs, int pos0, int heads, int NS, int rows, float * out) {
    attn_comb_body<false>(recs, pos0, heads, NS, rows, out, nullptr);
}
extern "C" __global__ void __launch_bounds__(256) pf_attn_comb_sel(const float * recs, int pos0, int heads, int NS, int rows, float * out,
                                                                   const int * sel) {
    attn_comb_body<true>(recs, pos0, heads, NS, rows, out, sel);
}

// ---- pf_gate_mul: OP_GATE_MUL: out[i] = a[i] * sigmoid(gate[h * stride + off + d]), h = i / D, over N elements -----------
extern "C" __global__ void __launch_bounds__(256) pf_gate_mul(const float * a, const float * gt, float * y, long N, int D, int stride, int off) {
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i >= N) return;
    const long h = i / D; const int d = (int)(i - h * D);
    const float sgm = 1.0f / (1.0f + expf(-gt[(size_t)h * stride + off + d]));
    const float v = a[i] * sgm;
    y[i] = v;
}

// ---- pf_hc_avg: OP_HC_AVG: out[t * ldo + i] = (((r0 + r1) + r2) + r3) * (1 / hc) of res [T, hc, D] --------------------------
extern "C" __global__ void __launch_bounds__(256) pf_hc_avg(const float * r, float * y, int T, int hc, int D, int ldo) {
#pragma clang fp contract(off)
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long)T * D) return;
    const int tt = (int)(i / D), ii = (int)(i - (long)tt * D);
    float acc = r[(size_t)(tt * hc) * D + ii];
    for (int h = 1; h < hc; h++) acc = acc + r[(size_t)(tt * hc + h) * D + ii];
    y[(size_t)tt * ldo + ii] = acc * (1.0f / (float)hc);
}

// ==== the QSA prompt path (M5 S7): the indexer's pools, scores and selection over many rows ================================
// lane l of a wave holds values 4l .. 4l + 3 of a 128-value row (k3_qsa_ops.h's qsa_ld_row / qsa_q8_row, copied)
__device__ __forceinline__ float4 qsa_ld_row(const uint8_t * row, int lane) {
    const block_q8_0 * b = (const block_q8_0 *)row + lane / 8;
    const int o = (lane % 8) * 4;
    const float d = __half2float(b->d);
    return make_float4((float)b->qs[o] * d, (float)b->qs[o + 1] * d, (float)b->qs[o + 2] * d, (float)b->qs[o + 3] * d);
}
__device__ __forceinline__ void qsa_q8_row(float4 v, uint8_t * row, int lane) {
#pragma clang fp contract(off)
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    amax = fmaxf(amax, __shfl_xor(amax, 1, 32));
    amax = fmaxf(amax, __shfl_xor(amax, 2, 32));
    amax = fmaxf(amax, __shfl_xor(amax, 4, 32));
    const float d = amax / ((1 << 7) - 1), id = d ? 1.0f / d : 0.0f;
    const half dh = __float2half(d);
    const int8_t q0 = roundf(v.x * id), q1 = roundf(v.y * id), q2 = roundf(v.z * id), q3 = roundf(v.w * id);
    block_q8_0 * b = (block_q8_0 *)row + lane / 8;
    const int o = (lane % 8) * 4;
    b->qs[o] = q0; b->qs[o + 1] = q1; b->qs[o + 2] = q2; b->qs[o + 3] = q3;
    if (lane % 8 == 0) b->d = dh;
}

// ---- pf_idx_pool: IDXPOOL[j] for j in [j0, j0 + nj), one wave per pool (8 per block), from the IDXRAW rows 4j .. 4j + 3 ----
//   op_idx_append's pooling, the same expressions in the same order: members dequantized ((float) q * d), summed
//   ((m0 + m1) + m2) + m3, fma(0.25, x, 0), y*y then fma x, z, w per lane, wsum (the xor butterfly), / 128, + eps, rsqrtf,
//   (scale * x) * k_norm, NEOX RoPE of dims 0 .. 63 at position 4j (theta = 4j * powf(theta_scale, i), x0' = fma(x0, cos,
//   -(x1 sin)), x1' = fma(x1, cos, x0 sin)), q8_0. A pool whose rows reach past IDXRAW's rows or past IDXPOOL's sets err bit
//   2 and is not written (the host refuses it first)
extern "C" __global__ void __launch_bounds__(256) pf_idx_pool(const uint8_t * raw, int raw_rows, uint8_t * pool, int pool_rows, int j0,
                                                              int nj, const float * kn, float theta_scale, float eps, int * err) {
#pragma clang fp contract(off)
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32, jj = blockIdx.x * 8 + wave;
    if (jj >= nj) return;
    const int j = j0 + jj, pos = 4 * j + 3;
    if (j < 0 || pos >= raw_rows || j >= pool_rows) { if (lane == 0) atomicOr(err, 4); return; }
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 1
    for (int m = 0; m < 3; m++) {
        const float4 v = qsa_ld_row(raw + (size_t)(pos - 3 + m) * QSA_ROWB, lane);
        s = m == 0 ? v : make_float4(s.x + v.x, s.y + v.y, s.z + v.z, s.w + v.w);
    }
    const float4 own = qsa_ld_row(raw + (size_t)pos * QSA_ROWB, lane);
    s = make_float4(s.x + own.x, s.y + own.y, s.z + own.z, s.w + own.w);
    s.x = __builtin_fmaf(0.25f, s.x, 0.0f); s.y = __builtin_fmaf(0.25f, s.y, 0.0f);
    s.z = __builtin_fmaf(0.25f, s.z, 0.0f); s.w = __builtin_fmaf(0.25f, s.w, 0.0f);
    float t = s.y * s.y;
    t = __builtin_fmaf(s.x, s.x, t); t = __builtin_fmaf(s.z, s.z, t); t = __builtin_fmaf(s.w, s.w, t);
    t = wred<false>(t);
    const float mean = t / 128.0f;
    const float sc = rsqrtf(mean + eps);
    const float4 nw = ((const float4 *)kn)[lane];
    float4 r = make_float4(sc * s.x, sc * s.y, sc * s.z, sc * s.w);
    r.x = r.x * nw.x; r.y = r.y * nw.y; r.z = r.z * nw.z; r.w = r.w * nw.w;
    float4 pr;
    pr.x = __shfl_xor(r.x, 8, 32); pr.y = __shfl_xor(r.y, 8, 32); pr.z = __shfl_xor(r.z, 8, 32); pr.w = __shfl_xor(r.w, 8, 32);
    if (lane < 16) {
        const bool lo = lane < 8;
        const float fp = (float)(pos - 3);
        const float4 x0 = lo ? r : pr, x1 = lo ? pr : r;
        float a0[4] = {x0.x, x0.y, x0.z, x0.w}, a1[4] = {x1.x, x1.y, x1.z, x1.w}, o[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
        for (int c = 0; c < 4; c++) {
            const float theta = fp * powf(theta_scale, (float)(4 * (lane & 7) + c));
            const float ct = cosf(theta), st = sinf(theta);
            const float v = lo ? __builtin_fmaf(a0[0], ct, -(a1[0] * st)) : __builtin_fmaf(a1[0], ct, a0[0] * st);
#pragma unroll
            for (int k = 0; k < 3; k++) { a0[k] = a0[k + 1]; a1[k] = a1[k + 1]; o[k] = o[k + 1]; }
            o[3] = v;
        }
        r = make_float4(o[0], o[1], o[2], o[3]);
    }
    qsa_q8_row(r, pool + (size_t)j * QSA_ROWB, lane);
}

// ---- pf_qsa_score_v0 / pf_qsa_score: SCORE[r * ld + j] for rows r of a sub-block (position pos0 + r) and pools j < nv_r = min((pos + 1) / 4,
//   pool_rows) --------------------------------------------------------------------------------------------------------------
//   op_qsa_score's per-element arithmetic: per head h = 0 .. 3 in order, lane l: fma(q.x, k.x, 0), then fma .y, .z, .w (dims
//   4l .. 4l + 3), wred<false> (the xor butterfly), relu, s = fma(sum, w, s) from 0. Block (blockIdx.x = 32-pool chunk,
//   blockIdx.y = group of rpg rows): wave w's 4 pools c0 + 4w .. + 3 loaded once for the group's rows. q fp32 [R][4][128]
//   (16-byte aligned), pool the dequantized IDXPOOL rows' q8_0 bytes; pools at or past nv_r are not written
extern "C" __global__ void __launch_bounds__(256) pf_qsa_score_v0(const float * q, const uint8_t * pool, int pool_rows, int pos0, int R,
                                                                  int rpg, float w, float * out, long ld) {
#pragma clang fp contract(off)
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const int r0 = blockIdx.y * rpg, r1 = min(R, r0 + rpg);
    if (r0 >= r1 || pos0 < 0) return;
    const int nvmax = min((pos0 + r1) >> 2, pool_rows);
    const int pb = blockIdx.x * 32 + wave * 4;
    if (pb >= nvmax) return;
    float4 k[4];
#pragma unroll
    for (int i = 0; i < 4; i++) k[i] = pb + i < nvmax ? qsa_ld_row(pool + (size_t)(pb + i) * QSA_ROWB, lane) : make_float4(0.f, 0.f, 0.f, 0.f);
    for (int r = r0; r < r1; r++) {
        const int nv = min((pos0 + r + 1) >> 2, pool_rows);
        if (pb >= nv) continue;
        const float4 * qr = (const float4 *)(q + (size_t)r * 512);
        float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 1
        for (int h = 0; h < 4; h++) {
            const float4 qh = qr[h * 32 + lane];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float a = __builtin_fmaf(qh.x, k[i].x, 0.0f);
                a = __builtin_fmaf(qh.y, k[i].y, a); a = __builtin_fmaf(qh.z, k[i].z, a); a = __builtin_fmaf(qh.w, k[i].w, a);
                a = wred<false>(a);
                a = a > 0.0f ? a : 0.0f;
                s[i] = __builtin_fmaf(a, w, s[i]);
            }
        }
        const float v = lane == 0 ? s[0] : lane == 1 ? s[1] : lane == 2 ? s[2] : s[3];
        if (lane < 4 && pb + lane < nv) out[(size_t)r * ld + pb + lane] = v;
    }
}

//   pf_qsa_score (the default): the same sums with a reduce-scatter instead of 16 full butterflies per row and wave. Each lane
//   holds the 16 partial dots v[i * 4 + h] (pool i, head h: the same 4-term fma chains); level 1 pairs lane l with l ^ 16 and
//   keeps half the values (own + the partner's: the butterfly's level-1 operands of those values), levels 2 .. 4 (l ^ 8, 4, 2)
//   halve again, level 5 (l ^ 1) adds the last one, so lanes 2k, 2k + 1 end with value k's sum over the butterfly's own tree
//   (each level's add is the butterfly's, commuted: the same bits as wred on every lane). Then relu, and lane 8i gathers its
//   pool's 4 heads (lanes 8i + 2h) for s = fma(a_h, w, s), h = 0 .. 3 from 0: op_qsa_score's chain. 16 cross-lane moves and
//   adds per row and wave instead of 80 + 80 (tests/test_prefill_qsa.py: == pf_qsa_score_v0 and K3's QSA_SCORE, bitwise)
template <int M> __device__ __forceinline__ float qsa_xm(float x) {
    const int xi = __builtin_bit_cast(int, x);
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(xi, xi, 0x160 | M, 0xf, 0xf, false));
}
extern "C" __global__ void __launch_bounds__(256) pf_qsa_score(const float * q, const uint8_t * pool, int pool_rows, int pos0, int R,
                                                               int rpg, float w, float * out, long ld) {
#pragma clang fp contract(off)
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const int r0 = blockIdx.y * rpg, r1 = min(R, r0 + rpg);
    if (r0 >= r1 || pos0 < 0) return;
    const int nvmax = min((pos0 + r1) >> 2, pool_rows);
    const int pb = blockIdx.x * 32 + wave * 4;
    if (pb >= nvmax) return;
    float4 k[4];
#pragma unroll
    for (int i = 0; i < 4; i++) k[i] = pb + i < nvmax ? qsa_ld_row(pool + (size_t)(pb + i) * QSA_ROWB, lane) : make_float4(0.f, 0.f, 0.f, 0.f);
    const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4, b1 = lane & 2;
    const int io = lane >> 3;                                    // lane 8 io: pool io's score
    for (int r = r0; r < r1; r++) {
        const int nv = min((pos0 + r + 1) >> 2, pool_rows);
        if (pb >= nv) continue;
        const float4 * qr = (const float4 *)(q + (size_t)r * 512);
        float v[16];
#pragma unroll
        for (int h = 0; h < 4; h++) {
            const float4 qh = qr[h * 32 + lane];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float a = __builtin_fmaf(qh.x, k[i].x, 0.0f);
                a = __builtin_fmaf(qh.y, k[i].y, a); a = __builtin_fmaf(qh.z, k[i].z, a); a = __builtin_fmaf(qh.w, k[i].w, a);
                v[i * 4 + h] = a;
            }
        }
        float u8[8], u4[4], u2[2];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float snd = b4 ? v[j] : v[8 + j], kp = b4 ? v[8 + j] : v[j];
            const unsigned su = __builtin_bit_cast(unsigned, snd);
            u8[j] = kp + __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(su, su, 0x76543210u, 0xfedcba98u, false, false));
        }
#pragma unroll
        for (int j = 0; j < 4; j++) { const float snd = b3 ? u8[j] : u8[4 + j], kp = b3 ? u8[4 + j] : u8[j]; u4[j] = kp + qsa_xm<8>(snd); }
#pragma unroll
        for (int j = 0; j < 2; j++) { const float snd = b2 ? u4[j] : u4[2 + j], kp = b2 ? u4[2 + j] : u4[j]; u2[j] = kp + qsa_xm<4>(snd); }
        const float snd = b1 ? u2[0] : u2[1], kp = b1 ? u2[1] : u2[0];
        float a = kp + qsa_xm<2>(snd);
        a = a + qsa_xm<1>(a);                                    // value (lane >> 1) & 15 = pool io, head (lane >> 1) & 3
        a = a > 0.0f ? a : 0.0f;
        const float a1 = __shfl_down(a, 2, 8), a2 = __shfl_down(a, 4, 8), a3 = __shfl_down(a, 6, 8);
        float sc = __builtin_fmaf(a, w, 0.0f);
        sc = __builtin_fmaf(a1, w, sc); sc = __builtin_fmaf(a2, w, sc); sc = __builtin_fmaf(a3, w, sc);
        if ((lane & 7) == 0 && pb + io < nv) out[(size_t)r * ld + pb + io] = sc;
    }
}

// ---- pf_qsa_select: SEL[r] for rows r of a sub-block, one block (256 threads) per row: op_qsa_select's rule and outputs --------
//   nv <= 512: all pools 0 .. nv - 1; otherwise the 512 largest scores, the lowest pool index first among equal keys, emitted
//   ascending; [0] count, [1] tail cells (pos + 1) % 4, [2] nv, [3] the threshold key (0 for the identity). The selected set
//   and every header word are fixed by the rule, so any grid gives the same row
__device__ __forceinline__ unsigned qsa_key(float v) {
    const unsigned b = __builtin_bit_cast(unsigned, v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
extern "C" __global__ void __launch_bounds__(256) pf_qsa_select(const float * score, long ld, int pool_rows, int pos0, int * sel) {
    __shared__ int hist[256]; __shared__ int misc[4]; __shared__ int wt[8]; __shared__ int wt2[8];
    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32, r = blockIdx.x, pos = pos0 + r;
    if (pos < 0) return;
    const int nv = min((pos + 1) >> 2, pool_rows), ntail = (pos + 1) & 3;
    int * out = sel + (size_t)r * QSA_SELW;
    const float * sc = score + (size_t)r * ld;
    if (nv <= QSA_TOPK) {
        for (int i = tid; i < nv; i += blockDim.x) out[4 + i] = i;
        if (tid == 0) { out[0] = nv; out[1] = ntail; out[2] = nv; out[3] = 0; }
        return;
    }
    unsigned prefix = 0; int need = QSA_TOPK;
    for (int shift = 24; shift >= 0; shift -= 8) {
        const unsigned hm = shift == 24 ? 0u : 0xFFFFFFFFu << (shift + 8);
        hist[tid] = 0;
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
