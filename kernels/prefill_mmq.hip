// Batched prefill, quantized GEMMs (docs/agentic/spec-prefill-cache.md S2): llama.cpp's ggml-cuda MMQ, vendored in
// kernels/ggml_mmq/ (MIT, tools/vendor_mmq.py; THIRD_PARTY.md), instantiated for the GSQ-RCO Coder's 10 GEMM formats on
// gfx1100. Modules of their own: the K3 decode code objects include none of it (they stay byte-identical).
// python/caminus/mmq.py launches these kernels (it replaces ggml's host launch code: J selection, grids, the q8_1
// buffers, mul_mat_id's grouping) and reads their configs from k3_mmq_configs below, so host and device agree on the
// tile sizes by construction.
//
// One source, compiled by build.sh into 11 code objects, each the way llama.cpp compiles the same kernels (ignite's flags,
// a full HIP compile through the offload driver's device LTO, the gfx1100 object unbundled):
//   build/prefill_mmq_<format>.hsaco   -DK3_MMQ_TYPE=<ggml_type>: one format per object, as ggml's
//                                      template-instances/mmq-instance-<format>.cu
//   build/prefill_mmq_glue.hsaco       -DK3_MMQ_GLUE: the activation quantizer and mul_mat_id's grouping
// Measured (tests/test_prefill_mmq.py codegen): with one format per object every kernel's instructions equal those of
// ignite's shipped objects (build-hip-therock, the llama.cpp the gates compare against; PC-relative literals masked).
// All 10 formats in one object did not: IQ2_S and IQ3_S came out with differently scheduled sign unpacking (the
// formats share helpers, and the inliner's choices follow the whole object); a device-only compile (build.sh's usual
// --offload-device-only, no LTO) differed in 27 of 68 kernels.
//
// What is compiled, and why:
// - mul_mat_q<type, J, fallback=false, GGML_PREC_Q8> for every J the RDNA3 config defines for that type (fallback=false
//   rows), so llama.cpp's own J choice (the fewest column tiles) can be reproduced at any T. fallback=true (row bound
//   checks) is not built: every per-card row count of the Coder's TP shards is a multiple of 128 (6400 / 3840, 3840 /
//   2304, 6144, 256, 384 / 256, 2560), and the host refuses any other (mmq.py, as ggml_cuda_mmq_needs_fallback).
// - quantize_mmq_q8_1 in the D4 (fp32 scale per 32: the Q8_0-layout formats) and DS4 (f16 scale + f16 partial sum:
//   Q4_K / Q5_K) layouts, plain (dense, and the per-slot rows of a MoE down projection, gathered by ids) and scatter (a
//   MoE gate / up projection: each token quantized once and written to its k compact rows).
// - mm_ids_helper<10> (the Coder's 10 experts per token): mul_mat_id's grouping, one wave per expert, a scan in token
//   order (no atomics), so the groups and their order are deterministic. It assumes valid ids, so every launch of it is
//   preceded by k3_mmq_ids_clamp (ours, below: ids in range and distinct per token by construction, a sticky flag word
//   for the host); extern "C", so the codegen comparison with ignite's objects (the _ZL kernels) does not cover it.
// The kernels keep their (internal-linkage) mangled names; mmq.py finds them in the code objects' symbol tables.
#include "ggml_mmq/mmq.cuh"

#if defined(K3_MMQ_TYPE) == defined(K3_MMQ_GLUE)
#error "prefill_mmq.hip: build with exactly one of -DK3_MMQ_TYPE=<ggml_type> or -DK3_MMQ_GLUE (build.sh)"
#endif

#if defined(K3_MMQ_TYPE)

#define K3_MMQ_J6(X, T) X(T, 16) X(T, 32) X(T, 48) X(T, 64) X(T, 96) X(T, 128)
#define K3_MMQ_J8(X, T) X(T, 16) X(T, 32) X(T, 48) X(T, 64) X(T, 80) X(T, 96) X(T, 112) X(T, 128)
// the J values of the RDNA3 config's fallback=false rows (mmq-config-rdna3.cuh): Q2_0 has 8, the other 9 formats 6
#define K3_MMQ_JS(X, T) K3_MMQ_J_##T(X, T)
#define K3_MMQ_J_GGML_TYPE_Q2_0(X, T) K3_MMQ_J8(X, T)
#define K3_MMQ_J_GGML_TYPE_IQ2_S(X, T) K3_MMQ_J6(X, T)
#define K3_MMQ_J_GGML_TYPE_IQ3_S(X, T) K3_MMQ_J6(X, T)
#define K3_MMQ_J_GGML_TYPE_IQ3_XXS(X, T) K3_MMQ_J6(X, T)
#define K3_MMQ_J_GGML_TYPE_IQ4_NL(X, T) K3_MMQ_J6(X, T)
#define K3_MMQ_J_GGML_TYPE_IQ4_XS(X, T) K3_MMQ_J6(X, T)
#define K3_MMQ_J_GGML_TYPE_Q4_K(X, T) K3_MMQ_J6(X, T)
#define K3_MMQ_J_GGML_TYPE_Q5_K(X, T) K3_MMQ_J6(X, T)
#define K3_MMQ_J_GGML_TYPE_Q6_K(X, T) K3_MMQ_J6(X, T)
#define K3_MMQ_J_GGML_TYPE_Q8_0(X, T) K3_MMQ_J6(X, T)
#define K3_MMQ_EXPAND(X, T) K3_MMQ_JS(X, T)

#define K3_MMQ_INST(T, J)                                                                                                   \
    static_assert(ggml_cuda_mmq_get_config_rdna3(T, J, false).type == T, "no RDNA3 MMQ config for this (type, J)");        \
    template __global__ void mul_mat_q<T, J, false, GGML_PREC_Q8>(                                                         \
        const char * __restrict__ x, const int * __restrict__ y, const int32_t * __restrict__ ids_dst,                     \
        const int32_t * __restrict__ expert_bounds, float * __restrict__ dst, float * __restrict__ tmp_fixup,              \
        const float * __restrict__ y_scale,                                                                               \
        const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y,    \
        const int stride_col_dst,                                                                                         \
        const uint3 channel_ratio, const uint3 nchannels_y, const int stride_channel_x, const int stride_channel_y,        \
        const int stride_channel_dst,                                                                                     \
        const uint3 sample_ratio, const uint3 nsamples_y, const int stride_sample_x, const int stride_sample_y,            \
        const int stride_sample_dst, const uint3 ntx);
K3_MMQ_EXPAND(K3_MMQ_INST, K3_MMQ_TYPE)

// the configs of this object's mul_mat_q kernels, K3_MMQ_CFG_WORDS ints each: type, J, fallback, nthreads, occupancy, I,
// sram_stride (32-bit words per tile row), K_vram (K per tile load), stream_k, the dynamic LDS bytes (ggml's
// mmq_get_nbytes_shared with the MMA data layout: J ids + I * sram_stride words + J q8_1 blocks padded to nthreads words),
// qk (the format's block), the activation layout (0 = D4, 1 = DS4: ggml's mmq_get_q8_1_ds_layout for these formats).
// One thread writes them; mmq.py reads them once per code object.
#define K3_MMQ_CFG_WORDS 12
static constexpr __device__ int k3_ds_layout(ggml_type t) {
    return t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ? 1 : 0;
}
#define K3_MMQ_CFG(T, J) {                                                                                                  \
        constexpr ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config_rdna3(T, J, false);                                     \
        constexpr int stride = ggml_cuda_mmq_get_sram_stride(c.sram_layout);                                                 \
        constexpr int nbs_y = J * (int) sizeof(block_q8_1_mmq), pad = c.nthreads * (int) sizeof(int);                     \
        const int v[K3_MMQ_CFG_WORDS] = {(int) T, J, 0, c.nthreads, c.occupancy, c.I, stride, c.K_vram, (int) c.stream_k,  \
            J * (int) sizeof(int) + c.I * stride * 4 + (nbs_y + pad - 1) / pad * pad, ggml_cuda_type_traits<T>::qk,         \
            k3_ds_layout(T)};                                                                                               \
        for (int w = 0; w < K3_MMQ_CFG_WORDS; ++w) out[n * K3_MMQ_CFG_WORDS + w] = v[w];                                   \
        ++n; }
extern "C" __global__ void k3_mmq_configs(int * out, int * count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int n = 0;
    K3_MMQ_EXPAND(K3_MMQ_CFG, K3_MMQ_TYPE)
    *count = n;
}

#else   // K3_MMQ_GLUE

#include "ggml_mmq/quantize_mmq.cuh"
#include "ggml_mmq/mmid.cuh"

#define K3_QUANT_INST(L, S)                                                                                                 \
    template __global__ void quantize_mmq_q8_1<L, S>(const float * __restrict__ x, const int32_t * __restrict__ ids,       \
        void * __restrict__ vy, const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,                \
        const int64_t ne0, const int ne1, const int ne2, const int n_expert_used);
K3_QUANT_INST(MMQ_Q8_1_DS_LAYOUT_D4, false)
K3_QUANT_INST(MMQ_Q8_1_DS_LAYOUT_D4, true)
K3_QUANT_INST(MMQ_Q8_1_DS_LAYOUT_DS4, false)
K3_QUANT_INST(MMQ_Q8_1_DS_LAYOUT_DS4, true)

template __global__ void mm_ids_helper<10>(const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1,
    int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds, const int n_tokens, const int n_expert_used_var,
    const int nchannels_y, const int si1, const int sis1, const bool write_inverse);

// fill n int32 words with v (the padding of mul_mat_id's ids_dst: MMQ reads up to J - 1 entries past the last group)
extern "C" __global__ void k3_mmq_fill_i32(int * p, int v, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

// The router's choices made safe for mm_ids_helper, which (verbatim ggml) assumes every id in [0, n_exp) and the k ids of a
// token distinct: an id out of range is counted in the expert bounds (if < 0) but matched by no expert's wave, and a
// duplicate id is matched once for two slots, so entries of ids_src1 / ids_dst are never written and a group's bounds
// cover columns no token fills; the quantizer and MMQ after it then index with stale words (out-of-bounds device writes
// on stale or uninitialised tables: CPU emulation in tests/test_prefill_safety.py, never launched). mmq.py's group()
// runs this before every mm_ids_helper launch, on the same stream. One thread per token, its k ids slot by slot: an id is
// kept if it is in [0, n_exp) and no earlier slot of the token holds it; otherwise it becomes the lowest expert the token
// does not hold yet (k <= n_exp, checked on the host), so the result is always k distinct ids in range, and the sticky
// flag word gets err |= 2 (an id out of range: NaN-born garbage, negatives, >= n_exp) / 4 (a duplicate). The host refuses
// the chunk on a non-zero flag (caminus.prefill_ops.check_routing). In place: a token's row is stored only when one of its
// slots changed, so valid ids are never written (the launch is a pure check). K3_IDS_KMAX bounds k (unrolled: registers,
// no scratch)
#define K3_IDS_KMAX 16
extern "C" __global__ void __launch_bounds__(256) k3_mmq_ids_clamp(int * ids, int ntok, int k, int n_exp, int * err) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ntok || k < 1 || k > K3_IDS_KMAX || k > n_exp) return;   // (the host refuses those shapes before launching)
    int * row = ids + (size_t)t * k;
    int v[K3_IDS_KMAX];
    int flag = 0;
#pragma unroll
    for (int j = 0; j < K3_IDS_KMAX; j++) {
        if (j >= k) break;
        int x = row[j];
        bool ok = x >= 0 && x < n_exp;
        if (!ok) flag |= 2;
#pragma unroll
        for (int i = 0; i < j; i++) if (ok && v[i] == x) { ok = false; flag |= 4; }
        if (!ok) {   // the lowest expert not held by slots 0 .. j - 1 (one of 0 .. j is free: j ids block at most j of them)
            x = j;
#pragma unroll
            for (int e = K3_IDS_KMAX - 1; e >= 0; e--) {
                if (e > j) continue;
                bool used = false;
#pragma unroll
                for (int i = 0; i < j; i++) used |= v[i] == e;
                if (!used) x = e;
            }
        }
        v[j] = x;
    }
    if (flag) {
#pragma unroll
        for (int j = 0; j < K3_IDS_KMAX; j++) if (j < k) row[j] = v[j];
        atomicOr(err, flag);
    }
}

#endif
