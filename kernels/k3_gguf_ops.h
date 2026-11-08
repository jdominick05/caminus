// K3 engine ops for GGUF block formats (included by kernels/k3.hip through K3_EXTRA_OPS_HEADER in the k3_gguf build).
// OP_GG_Q8_1: Seg.a = fp32 input [N], Seg.y = q8_1 blocks out; a task = 8 blocks, one wave per block (llama.cpp's
//             quantize_q8_1_lane, so the bytes equal gg_quantize_q8_1's). Seg.work: 0 x, 1 silu(x * Seg.rs), 2 silu(x) *
//             Seg.gs[i] (fp32 [N]: llama.cpp's unfused swiglu, the shared expert when gate and up differ in format; only
//             with K3_Q8_SWIGLU: the k3_gguf_gsq modules and the REAP-308 test build k3_gguf_swiglu*).
// OP_GG_MV:   Seg.W = raw GGUF blocks of N contiguous rows, Seg.a = q8_1 activations, Seg.y = fp32 out [N], K = ncols,
//             Seg.work = format (GG_IQ3_XXS 0, GG_IQ4_NL 1, GG_Q6_K 2, GG_Q8_0 3); a task = rows_per_task rows, wave w
//             takes rows w, w + 8, ...; each row is computed exactly as gg_mv_* does (same lane mapping, same vec_dot,
//             same wave reduction), so the results are bitwise equal to the standalone kernels.
// 4-9: the GSQ-RCO Coder's formats (python/caminus/k3.py GG_FMT). Each op instantiates only the formats of its whitelist
// (K3_FMTS_<op>, below: the REAP-308 four by default, kernels/k3_gguf_gsq.hip sets the Coder's from tools/gg_inventory.py);
// k3.Engine refuses tapes whose (op, format) pairs a module does not implement
enum { GG_IQ3_XXS = 0, GG_IQ4_NL = 1, GG_Q6_K = 2, GG_Q8_0 = 3, GG_Q4_K = 4, GG_Q5_K = 5, GG_IQ4_XS = 6, GG_IQ2_S = 7,
       GG_IQ3_S = 8, GG_Q2_0 = 9, GG_Q3_K = 10 };   // Q3_K: the qwen35moe modules (HauhauCS Q3_K_P)
// per-op format whitelists (comma lists of the enum above; empty = the op is compiled out). Code size and registers: every
// listed format is one more instance of the op in k3_run (one register-heavy instance sets the occupancy of the kernel)
#ifndef K3_FMTS_MV
#define K3_FMTS_MV GG_IQ3_XXS, GG_IQ4_NL, GG_Q6_K, GG_Q8_0
#endif
#ifndef K3_FMTS_MOE_GU
#define K3_FMTS_MOE_GU GG_IQ3_XXS, GG_IQ4_NL, GG_Q6_K, GG_Q8_0
#endif
#ifndef K3_FMTS_MOE_DOWN
#define K3_FMTS_MOE_DOWN GG_IQ3_XXS, GG_IQ4_NL, GG_Q6_K, GG_Q8_0
#endif
#ifndef K3_FMTS_GU_DENSE
#define K3_FMTS_GU_DENSE GG_IQ3_XXS, GG_IQ4_NL, GG_Q6_K, GG_Q8_0
#endif
constexpr unsigned gg_bits() { return 0u; }
template <typename... R> constexpr unsigned gg_bits(int f, R... r) { return (1u << f) | gg_bits(r...); }
constexpr unsigned K3_MASK_MV = gg_bits(K3_FMTS_MV), K3_MASK_MOE_GU = gg_bits(K3_FMTS_MOE_GU),
                   K3_MASK_MOE_DOWN = gg_bits(K3_FMTS_MOE_DOWN), K3_MASK_GU_DENSE = gg_bits(K3_FMTS_GU_DENSE);
// Activation source of the GG matvec / MoE ops, in Seg.out_f32: GG_ACT_Q8 = q8_1 blocks in global memory (from an
// OP_GG_Q8_1 segment); GG_ACT_F32 = fp32, which each workgroup quantizes into LDS (AQ) with quantize_q8_1_lane when it
// starts its first task of the segment (the same bytes OP_GG_Q8_1 would write, without its segment and hand-off);
// GG_ACT_F32_SILU = the same with silu(x * Seg.rs) first (OP_GG_Q8_1 work 1). At most AQ_BLOCKS blocks.
enum { GG_ACT_Q8 = 1, GG_ACT_F32 = 2, GG_ACT_F32_SILU = 3 };
// tokens per pass (speculative verification): Seg.team (k3_run ignores teams; <= 1 means one token)
__device__ __forceinline__ int seg_T(const Seg & sg) { return sg.team > 1 ? sg.team : 1; }
#define AQ_BLOCKS 80
#if defined(K3_GSQ) && defined(K3_MT)
// the multi-token GSQ module: AQ's storage also holds OP_BF16_MV's staged activation (BX, below: 4 tokens x 320 floats,
// 5120 B). Only fp32-activation GG ops at T = 1 fill AQ, and a workgroup never returns to a segment, so the two never hold
// live data at once; LDS 13164 -> 15432 B (8 workgroups per WGP: 123.5 KB of the 128). The one-token module's BX (1280 B)
// fits in AQ's 2880 B as it is
#define AQ_ALLOC 143
#else
#define AQ_ALLOC AQ_BLOCKS
#endif
__shared__ block_q8_1 AQ[AQ_ALLOC];

// ggml-cuda's silu (unary.cu op_silu): x / (1 + exp(-x))
__device__ __forceinline__ float gg_silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ void gg_op_q8_1(const Seg & sg, int t) {
    const int wave = K3WAVE, lane = K3TID % 32, b = t * 8 + wave, i = b * 32 + lane;
    const float * x = (const float *)sg.a;
    block_q8_1 * y = (block_q8_1 *)sg.y;
    if (b * 32 >= sg.N + 31) return;                 // past the last block (whole wave: uniform)
#ifdef K3_DRAFT   // work 3 (the DFlash inject's fc input): x [L = pad1][T][K] read as [T][L * K] (target-layer order per token)
    float xi;
    if (sg.work == 3) {
        const int LK = sg.pad1 * sg.K, tt = i / LK, r = i - tt * LK, ll = r / sg.K;
        xi = i < sg.N ? x[((size_t)ll * seg_T(sg) + tt) * sg.K + (r - ll * sg.K)] : 0.0f;
    } else xi = i < sg.N ? x[i] : 0.0f;
#else
    float xi = i < sg.N ? x[i] : 0.0f;
#endif
    if (sg.work == 1 && i < sg.N) { const float v = xi * sg.rs; xi = gg_silu(v); }   // ggml_scale, then ggml_silu
#ifdef K3_Q8_SWIGLU   // not in the REAP-308 modules: the never-taken branch alone cost their k3_run ~0.04-0.07 ms per token
    if (sg.work == 2 && i < sg.N) xi = gg_silu(xi) * ((const float *)sg.gs)[i];     // swiglu: op_silu(x) * g (unary.cu)
#endif
    float d, sum;
    const int8_t q = quantize_q8_1_lane(xi, d, sum);
    y[b].qs[lane] = q;
    if (lane == 0) y[b].ds = make_half2(d, sum);
}

// quantize n fp32 values (n % 32 == 0, n <= 32 * AQ_BLOCKS) into AQ, all 256 threads: 8 lanes per block, a float4
// each, 4 passes of loads issued before any is used. d = amax / 127 and q = roundf(x / d) per element exactly as
// quantize_q8_1_lane; the block sum (ds.y) is summed in another order, and none of the formats here reads it (their
// vec_dots use d only), so the matvec results are bitwise those of an OP_GG_Q8_1 segment.
__device__ __forceinline__ void aq_fill(const float * x, int n, int mode, float rs) {
    constexpr int P = 4;
    const int tid = K3TID, l8 = tid % 8, nb = n / 32;
    __syncthreads();
    for (int base = 0; base < nb; base += 32 * P) {
        float4 v[P];
#pragma unroll
        for (int p = 0; p < P; p++) {
            const int b = base + p * 32 + tid / 8;
            v[p] = b < nb ? *(const float4 *)(x + (size_t)b * 32 + l8 * 4) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
#pragma unroll
        for (int p = 0; p < P; p++) {
            const int b = base + p * 32 + tid / 8;
            float e[4] = {v[p].x, v[p].y, v[p].z, v[p].w};
            if (mode == GG_ACT_F32_SILU) {
#pragma unroll
                for (int i = 0; i < 4; i++) { const float t = e[i] * rs; e[i] = gg_silu(t); }
            }
            float amax = fmaxf(fmaxf(fabsf(e[0]), fabsf(e[1])), fmaxf(fabsf(e[2]), fabsf(e[3])));
            float sum = (e[0] + e[1]) + (e[2] + e[3]);
#pragma unroll
            for (int m = 4; m >= 1; m >>= 1) {
                amax = fmaxf(amax, __shfl_xor(amax, m, 32));
                sum += __shfl_xor(sum, m, 32);
            }
            const float d = amax / 127.0f;
            if (b < nb) {
                char4 q;
                q.x = amax == 0.0f ? 0 : (int8_t)roundf(e[0] / d); q.y = amax == 0.0f ? 0 : (int8_t)roundf(e[1] / d);
                q.z = amax == 0.0f ? 0 : (int8_t)roundf(e[2] / d); q.w = amax == 0.0f ? 0 : (int8_t)roundf(e[3] / d);
                *(char4 *)&AQ[b].qs[l8 * 4] = q;
                if (l8 == 0) AQ[b].ds = make_half2(d, sum);
            }
        }
    }
    __syncthreads();
}
// wave-wide xor-butterfly reductions on DPP / permlanex16 (VALU) instead of ds_bpermute (an LDS round trip per step):
// the same partners in the same order (16, 8, 4, 2, 1), and each step's op (add, fmaxf) is commutative, so every lane
// gets warp_reduce_sum<32> / warp_reduce_max<32>'s bits. All 32 lanes must be active (as for the shuffles).
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
__device__ __forceinline__ float wsum(float x) { return wred<false>(x); }
// quantize_q8_1_lane with the DPP reductions (bitwise the same d, sum and code)
__device__ __forceinline__ int8_t q8_lane(const float xi, float & d, float & sum) {
    float amax = fabsf(xi);
    sum = xi;
    amax = wred<true>(amax);
    sum = wred<false>(sum);
    d = amax / 127.0f;
    return amax == 0.0f ? 0 : roundf(xi / d);
}
// producer-side q8_1: a whole wave holding 32 consecutive outputs (element i in this lane, i % 32 == lane) writes their
// q8_1 block, the same bytes an OP_GG_Q8_1 segment would (quantize_q8_1_lane)
__device__ __forceinline__ void q8_emit(void * q8, size_t i, float v) {
    float d, sum;
    const int8_t q = q8_lane(v, d, sum);
    block_q8_1 * b = (block_q8_1 *)q8 + i / 32;
    b->qs[i % 32] = q;
    if (i % 32 == 0) b->ds = make_half2(d, sum);
}

// glue ops' exact-size fast paths (n_embd 2560 = GLUE_NI x 256 threads): each thread's elements tid + j * 256 are
// loaded together through global (not flat) pointers before any is used, then computed with the same expressions in
// the same order as the general loops (bitwise). The general loops stay for other sizes.
#define GLUE_NI 10
// an accumulator's 0.0f the compiler cannot see: unrolled, `acc = 0; acc += a0 * b0; acc += a1 * b1` would fold the
// first add and contract fadd(a0 * b0, a1 * b1) into an fma on EITHER product (it picked fma(a0, b0, a1 * b1)); the
// general loops it must match keep the zero and run fma(a1, b1, fma(a0, b0, 0)), so the fast paths start from this
__device__ __forceinline__ float acc0() { float z = 0.0f; asm volatile("" : "+v"(z)); return z; }
typedef __attribute__((address_space(1))) float GW;   // global (not flat) stores
typedef __attribute__((address_space(1))) block_q8_1 GQ8;
// q8_emit for NI values per lane (element i0 + j * stride, stride % 32 == 0). NI > 2 runs as a rolled loop: these ops
// run with a cold instruction cache (k3_run's code is ~3x the I-cache), and fetching NI unrolled copies of the
// quantization cost more than its VALU latency (hc_post_rms, 10 values: 3.6 -> 2.3 us); the values rotate through the
// registers so every index stays static
template <int NI>
__device__ __forceinline__ void q8_emit_n(const void * q8, size_t i0, int stride, const float * v) {
    float r[NI];
#pragma unroll
    for (int j = 0; j < NI; j++) r[j] = v[j];
    GQ8 * bq = (GQ8 *)q8 + i0 / 32;
#pragma unroll(NI <= 2 ? NI : 1)
    for (int j = 0; j < NI; j++) {
        float d, sum;
        const int8_t q = q8_lane(r[0], d, sum);
        bq->qs[i0 % 32] = q;
        if (i0 % 32 == 0) bq->ds = make_half2(d, sum);
        bq += stride / 32;
        const float r0 = r[0];
#pragma unroll
        for (int k = 0; k + 1 < NI; k++) r[k] = r[k + 1];
        r[NI - 1] = r0;
    }
}

// the q8_1 activation an op reads: AQ, or the global blocks at Seg.a
__device__ __forceinline__ const block_q8_1 * gg_act(const Seg & sg) {
    return sg.out_f32 == GG_ACT_Q8 ? (const block_q8_1 *)sg.a : (const block_q8_1 *)AQ;
}

// the row helpers below (gg_row, gg_row_tg, gg_row_n) each name the format's vec_dot in their own if-constexpr chain: a
// shared gg_vdot<TYPE> wrapper (spec-gsq-port.md 4a) changed the REAP-308 module's k3_run code, so the chains stay
template <ggml_type TYPE, int VDR>
__device__ __forceinline__ float gg_row(const void * vx, const block_q8_1 * y, int row, int ncols, int lane) {
    constexpr int qk = ggml_cuda_type_traits<TYPE>::qk, qi = ggml_cuda_type_traits<TYPE>::qi;
    constexpr int blocks_per_iter = VDR * 32 / qi;
    const int blocks_per_row = ncols / qk;
    float tmp = 0.0f;
    for (int kbx = lane / (qi / VDR); kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1), kqs = VDR * (lane % (qi / VDR)), ib = row * blocks_per_row + kbx;
        if constexpr (TYPE == GGML_TYPE_IQ3_XXS) tmp += vec_dot_iq3_xxs_q8_1(vx, &y[kby], ib, kqs);
        else if constexpr (TYPE == GGML_TYPE_IQ4_NL) tmp += vec_dot_iq4_nl_q8_1(vx, &y[kby], ib, kqs);
        else if constexpr (TYPE == GGML_TYPE_Q6_K) tmp += vec_dot_q6_K_q8_1(vx, &y[kby], ib, kqs);
        else if constexpr (TYPE == GGML_TYPE_Q4_K) tmp += vec_dot_q4_K_q8_1(vx, &y[kby], ib, kqs);
        else if constexpr (TYPE == GGML_TYPE_Q5_K) tmp += vec_dot_q5_K_q8_1(vx, &y[kby], ib, kqs);
        else if constexpr (TYPE == GGML_TYPE_IQ4_XS) tmp += vec_dot_iq4_xs_q8_1(vx, &y[kby], ib, kqs);
        else if constexpr (TYPE == GGML_TYPE_IQ2_S) tmp += vec_dot_iq2_s_q8_1(vx, &y[kby], ib, kqs);
        else if constexpr (TYPE == GGML_TYPE_IQ3_S) tmp += vec_dot_iq3_s_q8_1(vx, &y[kby], ib, kqs);
        else if constexpr (TYPE == GGML_TYPE_Q2_0) tmp += vec_dot_q2_0_q8_1(vx, &y[kby], ib, kqs);
        else if constexpr (TYPE == GGML_TYPE_Q3_K) tmp += vec_dot_q3_K_q8_1(vx, &y[kby], ib, kqs);
        else tmp += vec_dot_q8_0_q8_1(vx, &y[kby], ib, kqs);
    }
    return warp_reduce_sum<32>(tmp);
}

// one row against up to TG tokens' activations (token tt's blocks at tt * kb): each weight block is loaded once for all
// of them (the vec_dot loads are the same addresses, so the compiler shares them); per token the same blocks are added
// in the same order as gg_row, so each result is bitwise gg_row's
#define TG 4
// tokens per group of a format's multi-token rows (gg_op_mv_t): fewer for the register-heavy vec_dots, whose TG = 4 group
// spilled at the 96-VGPR cap (IQ4_XS alone: 36 VGPRs, Q5_K 4). Each token's sum is the same whatever the group (bitwise)
#ifndef K3_TG_IQ4_XS
#define K3_TG_IQ4_XS 2
#endif
#ifndef K3_TG_Q5_K
#define K3_TG_Q5_K 2
#endif
#ifndef K3_TG_IQ3_S
#define K3_TG_IQ3_S TG   // IQ3_S as a dense format (k3_gguf_q3n's ssm_ba) sets its own: TG 4 spilled there
#endif
#ifndef K3_TG_Q3_K
#define K3_TG_Q3_K TG    // Q3_K (the qwen35moe HauhauCS modules) may set its own
#endif
template <ggml_type TYPE>
constexpr int gg_tg() { return TYPE == GGML_TYPE_IQ4_XS ? K3_TG_IQ4_XS : TYPE == GGML_TYPE_Q5_K ? K3_TG_Q5_K :
                               TYPE == GGML_TYPE_IQ3_S ? K3_TG_IQ3_S : TYPE == GGML_TYPE_Q3_K ? K3_TG_Q3_K : TG; }
template <ggml_type TYPE, int VDR, int G = TG>
__device__ __forceinline__ void gg_row_tg(const void * vx, const block_q8_1 * y, int kb, int row, int ncols, int lane, int nt, float * out) {
    constexpr int qk = ggml_cuda_type_traits<TYPE>::qk, qi = ggml_cuda_type_traits<TYPE>::qi;
    constexpr int blocks_per_iter = VDR * 32 / qi;
    const int blocks_per_row = ncols / qk;
    float tmp[G];
#pragma unroll
    for (int j = 0; j < G; j++) tmp[j] = 0.0f;
    for (int kbx = lane / (qi / VDR); kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1), kqs = VDR * (lane % (qi / VDR)), ib = row * blocks_per_row + kbx;
#pragma unroll
        for (int j = 0; j < G; j++) {
            if (j >= nt) break;
            const block_q8_1 * yj = y + j * kb + kby;
            if constexpr (TYPE == GGML_TYPE_IQ3_XXS) tmp[j] += vec_dot_iq3_xxs_q8_1(vx, yj, ib, kqs);
            else if constexpr (TYPE == GGML_TYPE_IQ4_NL) tmp[j] += vec_dot_iq4_nl_q8_1(vx, yj, ib, kqs);
            else if constexpr (TYPE == GGML_TYPE_Q6_K) tmp[j] += vec_dot_q6_K_q8_1(vx, yj, ib, kqs);
            else if constexpr (TYPE == GGML_TYPE_Q4_K) tmp[j] += vec_dot_q4_K_q8_1(vx, yj, ib, kqs);
            else if constexpr (TYPE == GGML_TYPE_Q5_K) tmp[j] += vec_dot_q5_K_q8_1(vx, yj, ib, kqs);
            else if constexpr (TYPE == GGML_TYPE_IQ4_XS) tmp[j] += vec_dot_iq4_xs_q8_1(vx, yj, ib, kqs);
            else if constexpr (TYPE == GGML_TYPE_IQ2_S) tmp[j] += vec_dot_iq2_s_q8_1(vx, yj, ib, kqs);
            else if constexpr (TYPE == GGML_TYPE_IQ3_S) tmp[j] += vec_dot_iq3_s_q8_1(vx, yj, ib, kqs);
            else if constexpr (TYPE == GGML_TYPE_Q2_0) tmp[j] += vec_dot_q2_0_q8_1(vx, yj, ib, kqs);
            else if constexpr (TYPE == GGML_TYPE_Q3_K) tmp[j] += vec_dot_q3_K_q8_1(vx, yj, ib, kqs);
            else tmp[j] += vec_dot_q8_0_q8_1(vx, yj, ib, kqs);
        }
    }
#pragma unroll
    for (int j = 0; j < G; j++) if (j < nt) out[j] = warp_reduce_sum<32>(tmp[j]);
}

// ---- one-token Q8_0 matvec rows with their loads first (K3_MV1) ----
// gg_row's lane iterations walk a row's blocks with one dependent load round trip each (the loop's trip count is only
// known at run time). For one token and Q8_0 weights (the hc down / up matrices, attn k / v, ssm alpha / beta, the
// shared expert's down) the paths below issue U lane iterations' loads, or R short rows' loads, before any is used, with
// the block index clamped to the row (an iteration past the end re-reads the last block and adds nothing), then add in
// gg_row's order starting from an opaque zero (the loop's fma chain: acc0) and reduce with wsum (warp_reduce_sum<32>'s
// partners and order): bitwise gg_row. TP decode 11.50 -> 10.80 ms/token. Q6_K rows (attn_qkv, attn_q, ssm_out,
// attn_output, the head) the same way were no faster on the wall (U 3; U 5 spills 26 VGPRs at the 96 cap): those
// segments run near bandwidth beside their off-path neighbours (attn_gate beside attn_qkv). Loads through
// address_space(1) pointers (global_load instead of flat_load) were no faster either (10.87 vs 10.87).
#ifndef K3_MV1
#define K3_MV1 1
#endif
#ifndef K3_MV_U
#define K3_MV_U 8     // lane iterations per batch of loads (hc down: 40 iterations; K 2560: 10); 5 is no faster, 10 spills
#endif
#ifndef K3_MV_R
#define K3_MV_R 2     // short rows (at most 2 lane iterations: hc up, K 320) per batch (1 / 2 / 3 / 4: 10.82 / 10.81 / 10.87 / 10.88)
#endif
template <int U>
__device__ __forceinline__ float gg_row_u(const void * vx, const block_q8_1 * y, int row, int nb, int lane) {
    constexpr int VDR = VDR_Q8_0_Q8_1_MMVQ, qi = ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qi, bpi = VDR * 32 / qi;
    const int k0 = lane / (qi / VDR), kqs = VDR * (lane % (qi / VDR));
    const int nit = (nb + bpi - 1) / bpi;          // lane 0's iteration count (the most): the same for the whole wave
    float tmp = acc0();
    for (int i0 = 0; i0 < nit; i0 += U) {
        float v[U];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int kc = min(k0 + (i0 + u) * bpi, nb - 1);
            v[u] = vec_dot_q8_0_q8_1(vx, y + kc, row * nb + kc, kqs);
        }
#pragma unroll
        for (int u = 0; u < U; u++) if (k0 + (i0 + u) * bpi < nb) tmp += v[u];
    }
    return wsum(tmp);
}
// rows of at most 2 lane iterations (nb <= 16 blocks): R rows' loads first (rows row + 8 j, clamped to the task)
template <int R>
__device__ __forceinline__ void gg_mv1_short(const void * vx, const block_q8_1 * y, float * out, int r0, int r1, int nb, int wave, int lane) {
    constexpr int VDR = VDR_Q8_0_Q8_1_MMVQ, qi = ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qi, bpi = VDR * 32 / qi;
    const int k0 = lane / (qi / VDR), kqs = VDR * (lane % (qi / VDR));
    for (int row = r0 + wave; row < r1; row += 8 * R) {
        float v[R][2];
#pragma unroll
        for (int j = 0; j < R; j++) {
            const int rj = min(row + 8 * j, r1 - 1);
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int kc = min(k0 + i * bpi, nb - 1);
                v[j][i] = vec_dot_q8_0_q8_1(vx, y + kc, rj * nb + kc, kqs);
            }
        }
#pragma unroll
        for (int j = 0; j < R; j++) {
            float tmp = acc0();
#pragma unroll
            for (int i = 0; i < 2; i++) if (k0 + i * bpi < nb) tmp += v[j][i];
            tmp = wsum(tmp);
            if (lane == 0 && row + 8 * j < r1) out[row + 8 * j] = tmp;
        }
    }
}
// true when it ran the task (one token, Q8_0)
template <ggml_type TYPE>
__device__ __forceinline__ bool gg_mv1(const Seg & sg, int r0, int r1, int wave, int lane) {
#if K3_MV1
    if constexpr (TYPE == GGML_TYPE_Q8_0) {
        const int nb = sg.K / QK8_0;
        float * out = (float *)sg.y;
        if (nb <= 16) {   // 2 lane iterations (8 blocks each)
            if (sg.out_f32 == GG_ACT_Q8) gg_mv1_short<K3_MV_R>(sg.W, (const block_q8_1 *)sg.a, out, r0, r1, nb, wave, lane);
            else gg_mv1_short<K3_MV_R>(sg.W, AQ, out, r0, r1, nb, wave, lane);
            return true;
        }
        if (sg.out_f32 == GG_ACT_Q8) {
            for (int row = r0 + wave; row < r1; row += 8) {
                const float v = gg_row_u<K3_MV_U>(sg.W, (const block_q8_1 *)sg.a, row, nb, lane);
                if (lane == 0) out[row] = v;
            }
            return true;
        }
    }
#endif
    return false;
}

// ---- multi-token Q8_0 rows with their loads first (T > 1, K3_MVT_U) ----
// gg_row_u for TG tokens (token j's blocks at j * kb; tokens past nt re-read token nt - 1 and are discarded): U lane
// iterations' weight words are loaded once for all tokens, every token's activation words of those iterations too,
// before any is used; then each token adds in gg_row's order from its own opaque zero and reduces with wsum: per token
// bitwise gg_row (and so gg_row_tg).
#ifndef K3_MVT_U
#ifdef K3_MT
#define K3_MVT_U 2
#else
#define K3_MVT_U 0
#endif
#endif
template <int U>
__device__ __forceinline__ void gg_row_ut(const void * vx, const block_q8_1 * y, int kb, int row, int nb, int lane, int nt, float * out) {
    constexpr int VDR = VDR_Q8_0_Q8_1_MMVQ, qi = ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qi, bpi = VDR * 32 / qi;
    const int k0 = lane / (qi / VDR), kqs = VDR * (lane % (qi / VDR));
    const int nit = (nb + bpi - 1) / bpi;
    float tmp[TG];
#pragma unroll
    for (int j = 0; j < TG; j++) tmp[j] = acc0();
    for (int i0 = 0; i0 < nit; i0 += U) {
        float v[U][TG];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int kc = min(k0 + (i0 + u) * bpi, nb - 1);
#pragma unroll
            for (int j = 0; j < TG; j++) v[u][j] = vec_dot_q8_0_q8_1(vx, y + min(j, nt - 1) * kb + kc, row * nb + kc, kqs);
        }
#pragma unroll
        for (int u = 0; u < U; u++)
            if (k0 + (i0 + u) * bpi < nb) {
#pragma unroll
                for (int j = 0; j < TG; j++) tmp[j] += v[u][j];
            }
    }
#pragma unroll
    for (int j = 0; j < TG; j++) out[j] = wsum(tmp[j]);
}
// gg_mv1_short for T <= TG tokens (rows of at most 2 lane iterations): R rows' weight words and every token's
// activation words first; per token gg_row's order from an opaque zero, wsum: bitwise gg_row_tg. Out [T, N]
#ifndef K3_MVT_R
#ifdef K3_MT
#define K3_MVT_R 1
#else
#define K3_MVT_R 0
#endif
#endif
template <int R>
__device__ __forceinline__ void gg_mvt_short(const void * vx, const block_q8_1 * y, int kb, float * out, int N, int r0, int r1, int nb, int wave, int lane, int T) {
    constexpr int VDR = VDR_Q8_0_Q8_1_MMVQ, qi = ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qi, bpi = VDR * 32 / qi;
    const int k0 = lane / (qi / VDR), kqs = VDR * (lane % (qi / VDR));
    for (int row = r0 + wave; row < r1; row += 8 * R) {
        float v[R][2][TG];
#pragma unroll
        for (int j = 0; j < R; j++) {
            const int rj = min(row + 8 * j, r1 - 1);
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int kc = min(k0 + i * bpi, nb - 1);
#pragma unroll
                for (int tt = 0; tt < TG; tt++) v[j][i][tt] = vec_dot_q8_0_q8_1(vx, y + min(tt, T - 1) * kb + kc, rj * nb + kc, kqs);
            }
        }
#pragma unroll
        for (int j = 0; j < R; j++)
#pragma unroll
            for (int tt = 0; tt < TG; tt++) {
                float tmp = acc0();
#pragma unroll
                for (int i = 0; i < 2; i++) if (k0 + i * bpi < nb) tmp += v[j][i][tt];
                tmp = wsum(tmp);
                if (lane == 0 && row + 8 * j < r1 && tt < T) out[(size_t)tt * N + row + 8 * j] = tmp;
            }
    }
}

template <ggml_type TYPE, int VDR>
__device__ __forceinline__ void gg_op_mv_t(const Seg & sg, int t) {
    const int wave = K3WAVE, lane = K3TID % 32;
    const int r0 = t * sg.rows_per_task, r1 = min(sg.N, r0 + sg.rows_per_task);
    const int T = seg_T(sg), kb = sg.K / QK8_1;
    if (T == 1) {
        if (gg_mv1<TYPE>(sg, r0, r1, wave, lane)) return;
        for (int row = r0 + wave; row < r1; row += 8) {
            const float v = gg_row<TYPE, VDR>(sg.W, gg_act(sg), row, sg.K, lane);
            if (lane == 0) ((float *)sg.y)[row] = v;
        }
        return;
    }
#if K3_MVT_U
    if constexpr (TYPE == GGML_TYPE_Q8_0) {
#if K3_MVT_R
        if (sg.K / QK8_0 <= 16 && T <= TG) {   // short rows (hc up): R rows x 2 lane iterations x T tokens' loads first
            gg_mvt_short<K3_MVT_R>(sg.W, gg_act(sg), kb, (float *)sg.y, sg.N, r0, r1, sg.K / QK8_0, wave, lane, T);
            return;
        }
#endif
        if (sg.out_f32 == GG_ACT_Q8 && sg.K / QK8_0 > 16) {
            const int nb = sg.K / QK8_0;
            for (int row = r0 + wave; row < r1; row += 8)
                for (int t0 = 0; t0 < T; t0 += TG) {
                    float v[TG];
                    gg_row_ut<K3_MVT_U>(sg.W, (const block_q8_1 *)sg.a + t0 * kb, kb, row, nb, lane, min(TG, T - t0), v);
                    if (lane == 0) {
#pragma unroll
                        for (int j = 0; j < TG; j++) if (t0 + j < T) ((float *)sg.y)[(size_t)(t0 + j) * sg.N + row] = v[j];
                    }
                }
            return;
        }
    }
#endif
    constexpr int G = gg_tg<TYPE>();
    for (int row = r0 + wave; row < r1; row += 8)
        for (int t0 = 0; t0 < T; t0 += G) {       // T tokens in groups of G: activation tt at block tt * K / 32, out [T, N]
            float v[G];
            gg_row_tg<TYPE, VDR, G>(sg.W, gg_act(sg) + t0 * kb, kb, row, sg.K, lane, min(G, T - t0), v);
            if (lane == 0) {
#pragma unroll
                for (int j = 0; j < G; j++) if (t0 + j < T) ((float *)sg.y)[(size_t)(t0 + j) * sg.N + row] = v[j];
            }
        }
}

// gg_row with the row's block count NB known at compile time: the lane's iterations are unrolled, each block index
// clamped to the row (a lane past the end re-reads the last block and adds 0), so every load is unconditional and the
// compiler can issue them all before the first is used. Same blocks, same vec_dot, same order of additions as gg_row:
// bitwise the same result.
template <ggml_type TYPE, int VDR, int NB>
__device__ __forceinline__ float gg_row_n(const void * vx, const block_q8_1 * y, int row, int lane) {
    constexpr int qk = ggml_cuda_type_traits<TYPE>::qk, qi = ggml_cuda_type_traits<TYPE>::qi;
    constexpr int bpi = VDR * 32 / qi, IT = (NB + bpi - 1) / bpi;
    const int k0 = lane / (qi / VDR), kqs = VDR * (lane % (qi / VDR));
    float v[IT];
#pragma unroll
    for (int i = 0; i < IT; i++) {
        const int kc = min(k0 + i * bpi, NB - 1), ib = row * NB + kc;
        if constexpr (TYPE == GGML_TYPE_IQ3_XXS) v[i] = vec_dot_iq3_xxs_q8_1(vx, &y[kc * (qk / QK8_1)], ib, kqs);
        else if constexpr (TYPE == GGML_TYPE_IQ4_NL) v[i] = vec_dot_iq4_nl_q8_1(vx, &y[kc * (qk / QK8_1)], ib, kqs);
        else if constexpr (TYPE == GGML_TYPE_Q6_K) v[i] = vec_dot_q6_K_q8_1(vx, &y[kc * (qk / QK8_1)], ib, kqs);
        else if constexpr (TYPE == GGML_TYPE_Q4_K) v[i] = vec_dot_q4_K_q8_1(vx, &y[kc * (qk / QK8_1)], ib, kqs);
        else if constexpr (TYPE == GGML_TYPE_Q5_K) v[i] = vec_dot_q5_K_q8_1(vx, &y[kc * (qk / QK8_1)], ib, kqs);
        else if constexpr (TYPE == GGML_TYPE_IQ4_XS) v[i] = vec_dot_iq4_xs_q8_1(vx, &y[kc * (qk / QK8_1)], ib, kqs);
        else if constexpr (TYPE == GGML_TYPE_IQ2_S) v[i] = vec_dot_iq2_s_q8_1(vx, &y[kc * (qk / QK8_1)], ib, kqs);
        else if constexpr (TYPE == GGML_TYPE_IQ3_S) v[i] = vec_dot_iq3_s_q8_1(vx, &y[kc * (qk / QK8_1)], ib, kqs);
        else if constexpr (TYPE == GGML_TYPE_Q2_0) v[i] = vec_dot_q2_0_q8_1(vx, &y[kc * (qk / QK8_1)], ib, kqs);
        else if constexpr (TYPE == GGML_TYPE_Q3_K) v[i] = vec_dot_q3_K_q8_1(vx, &y[kc * (qk / QK8_1)], ib, kqs);
        else v[i] = vec_dot_q8_0_q8_1(vx, &y[kc * (qk / QK8_1)], ib, kqs);
    }
    float tmp = 0.0f;
#pragma unroll
    for (int i = 0; i < IT; i++) if (k0 + i * bpi < NB) tmp += v[i];
    return warp_reduce_sum<32>(tmp);
}

// OP_GG_MOE_GU: h[j * ffn + r] = silu(gate_e . x) * (up_e . x) for the k routed experts e = ids[j] (on the device).
//   Seg.W = gate_exps base, Seg.gs = up_exps base (both raw blocks, one slab of ffn rows per expert, Seg.pad2 bytes
//   apart), Seg.a = x (gg_act), Seg.asum = int32 ids [k], Seg.y = fp32 h [k * ffn], or with Seg.rs != 0 its q8_1
//   blocks (rows_per_task 32); N = k * ffn, K = d, pad1 = ffn.
template <ggml_type TYPE, int VDR>
__device__ __forceinline__ void gg_moe_gu_t(const Seg & sg, int t) {
    const int wave = K3WAVE, lane = K3TID % 32, ffn = sg.pad1;
    const int r0 = t * sg.rows_per_task, r1 = min(sg.N, r0 + sg.rows_per_task);
    const int * ids = (const int *)sg.asum;
    __shared__ float hb[32];
    const bool q8 = sg.rs != 0.0f;     // y = q8_1 blocks of h (tasks of 32 rows: one block each), else fp32 h
    const int kf = sg.N / seg_T(sg);           // rows per token (k * ffn)
    for (int row = r0 + wave; row < r1; row += 8) {
        const int tt = row / kf, j = (row - tt * kf) / ffn, lr = row - tt * kf - j * ffn, k = kf / ffn;
        const size_t off = (size_t)ids[tt * k + j] * (size_t)sg.pad2;
        const block_q8_1 * act = gg_act(sg) + tt * (sg.K / QK8_1);
        const float g = gg_row<TYPE, VDR>(sg.W + off, act, lr, sg.K, lane);
        const float u = gg_row<TYPE, VDR>((const uint8_t *)sg.gs + off, act, lr, sg.K, lane);
        if (lane == 0) { const float hv = gg_silu(g) * u; if (q8) hb[row - r0] = hv; else ((float *)sg.y)[row] = hv; }
    }
    if (q8) {
        __syncthreads();
        if (wave == 0) q8_emit(sg.y, (size_t)r0 + lane, hb[lane]);
        __syncthreads();
    }
}

// OP_GG_MOE_DOWN: out[r] = sum_j (down_e . h_j)[r] * w_j for e = ids[j], j = 0..k-1, summed in order as separate
//   multiplies and adds (ggml_mul then a chain of ggml_add, as llama.cpp's build_moe_ffn).
//   Seg.W = down_exps base (slabs Seg.pad2 bytes apart), Seg.a = q8_1 h [k * ffn], Seg.asum = int32 ids [k],
//   Seg.gs = fp32 router weights [k], Seg.y = fp32 out [d]; N = d, K = ffn, pad1 = k.
template <ggml_type TYPE, int VDR>
__device__ __forceinline__ void gg_moe_down_t(const Seg & sg, int t) {
#pragma clang fp contract(off)   // llama.cpp multiplies and adds in separate kernels: no FMA here (HIP contracts by default)
    const int wave = K3WAVE, lane = K3TID % 32, k = sg.pad1, fb = sg.K / QK8_1;
    const int r0 = t * sg.rows_per_task, r1 = min(sg.N, r0 + sg.rows_per_task);
    const int * ids = (const int *)sg.asum;
    const float * wts = (const float *)sg.gs;
    const int dd = sg.N / seg_T(sg);           // output rows per token
    for (int row_ = r0 + wave; row_ < r1; row_ += 8) {
        const int tt = row_ / dd, row = row_ - tt * dd;
        float acc = 0.0f;
        for (int j = 0; j < k; j++) {
            const float v = gg_row<TYPE, VDR>(sg.W + (size_t)ids[tt * k + j] * (size_t)sg.pad2, gg_act(sg) + (tt * k + j) * fb, row, sg.K, lane);
            const float vw = v * wts[tt * k + j];  // written out here, not via __fmul_rn / __fadd_rn: the pragma only
            acc = j == 0 ? vw : acc + vw;         // covers expressions in this scope (the intrinsics' bodies contract)
        }
        if (lane == 0) ((float *)sg.y)[row_] = acc;
    }
}

// fixed-shape MoE paths (rows of NB blocks, KU experts): straight-line code, so the loads of all experts' row pieces
// can be in flight together; additions in llama.cpp's order as gg_moe_down_t / gg_moe_gu_t (bitwise the same).
// K3_MOE_N (bit 1: gu, bit 2: down) picks which run for one token; default down only. With the max-ilp scheduler
// (build.sh) at the 96-VGPR cap: down JU 2 fits (92 VGPRs, 0 spills; MOE_DOWN 24.1 / 20.0 -> 18.1 / 15.4 us XTX / XT,
// 12.30 -> 12.04 ms/token vs the same code with the path never taken); down JU 5 spills 138, JU 10 184, gu_n 76
// (14.6 / 13.6 ms/token). Under the default scheduler JU 2 / 5 / 10 compile to 0 / 0 / 166 spills.
#ifndef K3_MOE_N
#define K3_MOE_N 2
#endif
#ifndef K3_MOE_JU
#define K3_MOE_JU 2   // experts per batch of loads; must divide KU
#endif
#ifndef K3_MOE_NT
#ifdef K3_MT
#define K3_MOE_NT 1   // the unrolled down for T > 1 tokens too (a task = one token's rows)
#else
#define K3_MOE_NT 0
#endif
#endif
template <ggml_type TYPE, int VDR, int NB, int KU, bool MT = false>
__device__ __forceinline__ void gg_moe_down_n(const Seg & sg, int t) {
#pragma clang fp contract(off)
    const int wave = K3WAVE, lane = K3TID % 32, fb = sg.K / QK8_1;
    const int r0 = t * sg.rows_per_task, r1 = min(sg.N, r0 + sg.rows_per_task);
    // T > 1: the task's rows are one token's (the host keeps d % rows_per_task == 0), token tt = r0 / d
    const int dd = MT ? sg.N / seg_T(sg) : sg.N, tt = MT ? r0 / dd : 0;   // (the one-token instance keeps its own code)
    const int * ids = (const int *)sg.asum + tt * KU;
    const float * wts = (const float *)sg.gs + tt * KU;
    const block_q8_1 * act = gg_act(sg) + tt * KU * fb;
    int e[KU];
#pragma unroll
    for (int j = 0; j < KU; j++) e[j] = ids[j];
    for (int row = r0 + wave; row < r1; row += 8) {
        float acc = 0.0f;
#pragma unroll 1
        for (int j0 = 0; j0 < KU; j0 += K3_MOE_JU) {    // K3_MOE_JU experts' loads in flight at a time
            float v[K3_MOE_JU];
#pragma unroll
            for (int u = 0; u < K3_MOE_JU; u++) v[u] = gg_row_n<TYPE, VDR, NB>(sg.W + (size_t)e[j0 + u] * (size_t)sg.pad2, act + (j0 + u) * fb, row - tt * dd, lane);
#pragma unroll
            for (int u = 0; u < K3_MOE_JU; u++) { const float vw = v[u] * wts[j0 + u]; acc = j0 + u == 0 ? vw : acc + vw; }
        }
        if (lane == 0) ((float *)sg.y)[row] = acc;
    }
}
template <ggml_type TYPE, int VDR, int NB>
__device__ __forceinline__ void gg_moe_gu_n(const Seg & sg, int t) {
    const int wave = K3WAVE, lane = K3TID % 32, ffn = sg.pad1;
    const int r0 = t * sg.rows_per_task, r1 = min(sg.N, r0 + sg.rows_per_task);
    const int * ids = (const int *)sg.asum;
    __shared__ float hb2[32];
    const bool q8 = sg.rs != 0.0f;
    for (int row = r0 + wave; row < r1; row += 8) {
        const int j = row / ffn, lr = row - j * ffn;
        const size_t off = (size_t)ids[j] * (size_t)sg.pad2;
        const float g = gg_row_n<TYPE, VDR, NB>(sg.W + off, gg_act(sg), lr, lane);
        const float u = gg_row_n<TYPE, VDR, NB>((const uint8_t *)sg.gs + off, gg_act(sg), lr, lane);
        if (lane == 0) { const float hv = gg_silu(g) * u; if (q8) hb2[row - r0] = hv; else ((float *)sg.y)[row] = hv; }
    }
    if (q8) {
        __syncthreads();
        if (wave == 0) q8_emit(sg.y, (size_t)r0 + lane, hb2[lane]);
        __syncthreads();
    }
}

#ifdef K3_GSQ
// ---- lane-packed MoE down (the GSQ-RCO Coder's short TP down rows, K3_MOE_P) ----
// A down row of NB blocks takes LPR = NB * qi / VDR lanes in gg_row's one lane iteration (Q2_0 K 256 / 384: 8 / 12 lanes,
// IQ4_NL K 256: 16); the other lanes hold 0.0f. Here each group of GS (8 or 16) lanes computes one expert's row, so a
// wave runs G = 32 / GS experts per round of loads, and RB rounds' loads are issued before any is used. Bitwise
// gg_moe_down_t: lane s of a group does what lane s of gg_row does (kbx = s / (qi / VDR), kqs = VDR * (s % (qi / VDR)),
// tmp = 0.0f, += the dot when s < LPR), and warp_reduce_sum<32>'s steps xor 16 (and 8 when GS = 8) only add the +0.0f of
// lanes past the row; they are kept (an opaque +0.0f add per step), the steps below GS run inside the group (DPP
// row_xmask, wred's partners and order), so every lane of the group holds gg_row's sum. The k-chain then takes expert j
// (round j / G, group j % G) by readlane, in j order: vw = v * w, acc = vw / acc + vw (contract off, as gg_moe_down_t).
#ifndef K3_MOE_P
#define K3_MOE_P 3    // bit 1: Q2_0 (K 384 / 256), bit 2: IQ4_NL K 256; 0 = the gg_moe_down_n / _t paths
#endif
#ifndef K3_MOE_P_RB6
#define K3_MOE_P_RB6 1   // rounds whose loads are in flight together (Q2_0 K 384: 5 rounds of 2 experts; 2 spills 2-5 VGPRs)
#endif
#ifndef K3_MOE_P_RB4
#define K3_MOE_P_RB4 1   // Q2_0 K 256: 3 rounds of 4 experts (2: the same speed, but the trace module spills its SGPRs to
                         // scratch: k3_gguf_gsq_mt_trace 4 ms slower untraced; 3 spills 17-20 VGPRs)
#endif
#ifndef K3_MOE_P_RBN
#define K3_MOE_P_RBN 3   // IQ4_NL K 256: 5 rounds of 2 experts (4-5 spill)
#endif
template <int GS>
__device__ __forceinline__ float gsum(float x) {
    // the 32-lane butterfly's steps at and above GS only add +0.0f here (lanes past the row): kept as written
    x = x + acc0();
    if constexpr (GS == 8) x = x + acc0();
#pragma unroll
    for (int m = GS / 2; m >= 1; m >>= 1) {   // row_xmask:m within each row of 16 lanes (wred)
        const int xi = __builtin_bit_cast(int, x);
        int yi;
        switch (m) {
            case 8: yi = __builtin_amdgcn_update_dpp(xi, xi, 0x168, 0xf, 0xf, false); break;
            case 4: yi = __builtin_amdgcn_update_dpp(xi, xi, 0x164, 0xf, 0xf, false); break;
            case 2: yi = __builtin_amdgcn_update_dpp(xi, xi, 0x162, 0xf, 0xf, false); break;
            default: yi = __builtin_amdgcn_update_dpp(xi, xi, 0x161, 0xf, 0xf, false); break;
        }
        x = x + __builtin_bit_cast(float, yi);
    }
    return x;
}
template <ggml_type TYPE, int VDR, int NB, int KU, bool MT, int RB>
__device__ __forceinline__ void gg_moe_down_p(const Seg & sg, int t) {
#pragma clang fp contract(off)
    constexpr int qk = ggml_cuda_type_traits<TYPE>::qk, qi = ggml_cuda_type_traits<TYPE>::qi, LPB = qi / VDR, LPR = NB * LPB;
    static_assert(VDR * 32 / qi >= NB, "one lane iteration per row");
    static_assert(LPR <= 16, "a row fits a group of 16 lanes");
    constexpr int GS = LPR <= 8 ? 8 : 16, G = 32 / GS, NR = (KU + G - 1) / G;
    const int wave = K3WAVE, lane = K3TID % 32, fb = sg.K / QK8_1;
    const int g = lane / GS, s = lane % GS;
    const bool in = s < LPR;
    const int kbx = min(s / LPB, NB - 1), kqs = VDR * (s % LPB);   // lanes past the row re-read the last block, add nothing
    const int r0 = t * sg.rows_per_task, r1 = min(sg.N, r0 + sg.rows_per_task);
    const int dd = MT ? sg.N / seg_T(sg) : sg.N, tt = MT ? r0 / dd : 0;   // T > 1: a task is one token's rows
    const int * ids = (const int *)sg.asum + tt * KU;
    const float * wts = (const float *)sg.gs + tt * KU;
    const block_q8_1 * act = gg_act(sg) + tt * KU * fb + kbx * (qk / QK8_1);
    for (int row = r0 + wave; row < r1; row += 8) {
        const int ib = (row - tt * dd) * NB + kbx;
        float acc = 0.0f;
#pragma unroll 1
        for (int rb = 0; rb < NR; rb += RB) {
            float v[RB];
#pragma unroll
            for (int u = 0; u < RB; u++) {
                const int j = min((min(rb + u, NR - 1)) * G + g, KU - 1);   // slots past KU / NR re-read the last, unused
                if constexpr (TYPE == GGML_TYPE_Q2_0) v[u] = vec_dot_q2_0_q8_1(sg.W + (size_t)ids[j] * (size_t)sg.pad2, act + j * fb, ib, kqs);
                else v[u] = vec_dot_iq4_nl_q8_1(sg.W + (size_t)ids[j] * (size_t)sg.pad2, act + j * fb, ib, kqs);
            }
#pragma unroll
            for (int u = 0; u < RB; u++) {
                float tmp = 0.0f;
                if (in) tmp += v[u];
                const float sm = gsum<GS>(tmp);
#pragma unroll
                for (int gi = 0; gi < G; gi++) {
                    const int j = (rb + u) * G + gi;
                    if (rb + u < NR && j < KU) {
                        const float vj = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, sm), gi * GS));
                        const float vw = vj * wts[j];
                        acc = j == 0 ? vw : acc + vw;
                    }
                }
            }
        }
        if (lane == 0) ((float *)sg.y)[row] = acc;
    }
}
#endif

// OP_GG_GU_DENSE (the shared expert, the same weights for every token): h[tt, r] = silu(gate_r . x_tt) * (up_r . x_tt),
//   the multi-token row dot for both matrices. Seg.W = gate, Seg.gs = up (raw blocks, rows contiguous), Seg.a = q8_1 x
//   [T, d], Seg.y = h [T, ffn]; N = ffn, K = d, work = format. Bitwise gg_moe_gu_t with one expert per token.
template <ggml_type TYPE, int VDR>
__device__ __forceinline__ void gg_gu_dense_t(const Seg & sg, int t) {
    const int wave = K3WAVE, lane = K3TID % 32, T = seg_T(sg), kb = sg.K / QK8_1;
    const int r0 = t * sg.rows_per_task, r1 = min(sg.N, r0 + sg.rows_per_task);
    for (int row = r0 + wave; row < r1; row += 8)
        for (int t0 = 0; t0 < T; t0 += TG) {
            float g[TG], u[TG]; const int nt = min(TG, T - t0);
            gg_row_tg<TYPE, VDR>(sg.W, gg_act(sg) + t0 * kb, kb, row, sg.K, lane, nt, g);
            gg_row_tg<TYPE, VDR>(sg.gs, gg_act(sg) + t0 * kb, kb, row, sg.K, lane, nt, u);
            if (lane == 0) {
#pragma unroll
                for (int j = 0; j < TG; j++) if (j < nt) ((float *)sg.y)[(size_t)(t0 + j) * sg.N + row] = gg_silu(g[j]) * u[j];
            }
        }
}

#ifndef K3_F32_U
#define K3_F32_U 5
#endif
typedef __attribute__((address_space(1))) const float4 GF4;
// OP_F32_MV: out[r] = W[r] . x for F32 weights (the router ffn_gate_inp and the shared-expert gate). Seg.W = fp32 [N, K]
//   row-major, Seg.a = fp32 x [K], Seg.y = fp32 out [N]; one wave per row, lanes stride K by 4, then a wave reduction.
__device__ __forceinline__ void op_f32_mv(const Seg & sg, int t) {
    const int wave = K3WAVE, lane = K3TID % 32;
    const int r0 = t * sg.rows_per_task, r1 = min(sg.N, r0 + sg.rows_per_task);
    const float * W = (const float *)sg.W;
    const int nr = sg.N / seg_T(sg);           // weight rows; rows [T, nr], token tt reads x + tt * K
    for (int row = r0 + wave; row < r1; row += 8) {
        const int tt = row / nr;
        const float * w = W + (size_t)(row - tt * nr) * sg.K, * x = (const float *)sg.a + (size_t)tt * sg.K;
        // K3_F32_U lane iterations per batch: all their loads first, then the adds in the same per-lane order
        float acc = acc0();
        for (int k0 = lane * 4; k0 < sg.K; k0 += 128 * K3_F32_U) {
            float4 wv[K3_F32_U], xv[K3_F32_U];
#pragma unroll
            for (int u = 0; u < K3_F32_U; u++) if (k0 + u * 128 < sg.K) { wv[u] = *(const GF4 *)(w + k0 + u * 128); xv[u] = *(const GF4 *)(x + k0 + u * 128); }
#pragma unroll
            for (int u = 0; u < K3_F32_U; u++) if (k0 + u * 128 < sg.K)
                acc += wv[u].x * xv[u].x + wv[u].y * xv[u].y + wv[u].z * xv[u].z + wv[u].w * xv[u].w;
        }
        acc = warp_reduce_sum<32>(acc);
        if (lane == 0) ((float *)sg.y)[row] = acc;
    }
}

#ifdef K3_GSQ
// OP_BF16_MV (the GSQ-RCO Coder's BF16 matrices: hc down / up / inject, output_hc, router, shared-expert gate, ssm alpha /
//   beta, ple key / value): out[tt, r] = W[r] . x[tt] with W bf16 [N, K] row-major (Seg.W), x fp32 [T, K] (Seg.a),
//   out fp32 [T, N] (Seg.y); Seg.N = weight rows (a task = rows_per_task rows, all T tokens), Seg.pad1 = W, the
//   virtual warps, Seg.work = 1: the activation is silu(rs * x + 0) (Seg.rs; the hc up after the hc down: ggml_scale +
//   ggml_silu, which ignite runs fused as scale_unary_kernel<op_silu>: one fma, then x / (1 + expf(-x))), staged in LDS
//   once per workgroup and segment. Seg.pad2 = row bytes (the pre-wait prefetch).
// Bitwise llama.cpp's mul_mat_vec_f<nv_bfloat16, float, ncols_dst, block_size> (ignite-llama mmvf.cu, the HIP int path;
// tests/test_k3_bf16_mv.py checks against the shipped kernel itself). There block_size = 32 W threads (W = 8 for K 2560 /
// 10240, 5 for K 320: the host picks it with launch_mul_mat_vec_f_cuda's rule) take a row; thread (v, lane) adds the
// bf16 pairs c = 32 v + lane, + 32 W, ... as fma(x0, y.x, acc) then fma(x1, y.y, acc) from 0.0f (two v_fmac_f32 per
// pair in libggml-hip's code object: G2), warp_reduce_sum<32> per warp (xor 16, 8, 4, 2, 1), the W sums through LDS
// slots zeroed to 32, and one more warp_reduce_sum<32> over lanes 0..31 (lanes >= W add +0.0f). Here one wave walks the W
// virtual warps one after another (one accumulator per token: the same per-thread order), reduces each with wsum (the
// same partners and order on DPP), keeps virtual warp v's sum on lane v (+0.0f elsewhere), and ends with the same final
// wsum. Each token's result is the same whatever T (the per-token order does not depend on the other tokens): T = 4 rows
// are bitwise T = 1 rows, and bitwise mul_mat_vec_f<..., ncols_dst = 4, ...> per token (the test checks both).
#ifndef K3_BF16_U
#ifdef K3_MT
#define K3_BF16_U 1   // lane iterations whose loads are issued before any is used (T = 4: 9 words per iteration; 2 spilled)
#else
#define K3_BF16_U 4
#endif
#endif
#ifdef K3_MT
#define BF16_TT TG    // tokens per row pass (tokens past T re-read token T - 1 and are discarded)
#else
#define BF16_TT 1
#endif
#define BF16_XS (320 * BF16_TT)   // LDS floats for the staged silu activation: T * K <= this (the hc up: K 320)
static_assert(BF16_XS * 4 <= AQ_ALLOC * (int)sizeof(block_q8_1), "BX lives in AQ's storage");
#define BX ((__attribute__((address_space(3))) float *)AQ)
__device__ __forceinline__ float bf16_lo(uint32_t w) { return __builtin_bit_cast(float, w << 16); }           // exact
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __builtin_bit_cast(float, w & 0xffff0000u); }
typedef __attribute__((address_space(1))) const uint32_t GU32;   // global (not flat) loads of the weights / activations
typedef __attribute__((address_space(3))) const float LF;        // LDS (ds_load) for the staged activation
__device__ __forceinline__ float2 ld2(const GW * p) { return *(__attribute__((address_space(1))) const float2 *)p; }
__device__ __forceinline__ float2 ld2(LF * p) { return *(__attribute__((address_space(3))) const float2 *)p; }
// x [n] (n = T * K) -> BX: silu(fma(rs, x, 0)), the expressions of ignite's scale_unary_kernel<op_silu> (unary.cu)
__device__ __forceinline__ void bx_fill(const float * x, int n, float rs) {
    __syncthreads();
    for (int i = K3TID; i < n; i += 256) { const float t = __builtin_fmaf(rs, x[i], acc0()); BX[i] = gg_silu(t); }
    __syncthreads();
}
#ifdef K3_MTP_SWIGLU
// the MTP module pair only (kernels/k3_gguf_gsq_mtp*.hip; the trunk's code objects do not contain it): Seg.work = 2 stages
// BX = silu(x) * m (Seg.gs = m), ggml's swiglu (unary.cuh ggml_cuda_op_swiglu_single: x / (1 + expf(-x)) * g) in fp32, for
// the MTP block's BF16 shared-expert down (its gate / up are BF16 too: the activation stays fp32, as llama.cpp's)
__device__ __forceinline__ void bx_fill_swiglu(const float * x, const float * m, int n) {
    __syncthreads();
    for (int i = K3TID; i < n; i += 256) BX[i] = gg_silu(x[i]) * m[i];
    __syncthreads();
}
#endif
// one weight row (bf16 pairs w2[0 .. K/2)) against TT tokens' activations (token j at x + min(j, T - 1) * K) -> res[j]
// on every lane
template <int TT, typename XP>
__device__ __forceinline__ void bf16_row(GU32 * w2, XP x, int K, int W, int lane, int T, float (&res)[TT]) {
    constexpr int U = K3_BF16_U;
    const int n2 = K / 2, bs = 32 * W;
    float fin[TT];
#pragma unroll
    for (int j = 0; j < TT; j++) fin[j] = 0.0f;
    for (int v = 0; v < W; v++) {
        float acc[TT];
#pragma unroll
        for (int j = 0; j < TT; j++) acc[j] = acc0();
        const int c0 = 32 * v + lane;
        const int nit = (n2 - 32 * v + bs - 1) / bs;   // lane 0's iteration count (the most): wave-uniform
        for (int i0 = 0; i0 < nit; i0 += U) {
            uint32_t wv[U]; float2 xv[U][TT];
#pragma unroll
            for (int u = 0; u < U; u++) {
                const int c = min(c0 + (i0 + u) * bs, n2 - 1);   // past the row's end: re-read the last pair, add nothing
                wv[u] = w2[c];
#pragma unroll
                for (int j = 0; j < TT; j++) xv[u][j] = ld2(x + (size_t)min(j, T - 1) * K + 2 * c);
            }
#pragma unroll
            for (int u = 0; u < U; u++)
                if (c0 + (i0 + u) * bs < n2) {
                    const float a0 = bf16_lo(wv[u]), a1 = bf16_hi(wv[u]);
#pragma unroll
                    for (int j = 0; j < TT; j++) { acc[j] = __builtin_fmaf(a0, xv[u][j].x, acc[j]); acc[j] = __builtin_fmaf(a1, xv[u][j].y, acc[j]); }
                }
        }
#pragma unroll
        for (int j = 0; j < TT; j++) { const float s = wsum(acc[j]); if (lane == v) fin[j] = s; }
    }
#pragma unroll
    for (int j = 0; j < TT; j++) res[j] = wsum(fin[j]);
}
// The fast paths below change only which wave / register / load batch does the work: every (virtual warp v, lane) keeps
// its pairs c = 32 v + lane + i * 32 W, i ascending, the same two fmas per pair from acc0(), one wsum per virtual warp,
// and the final wsum over lanes with lane v = virtual warp v's sum and +0.0f on lanes >= W. So each row (and each token)
// is bitwise bf16_row's, whatever the path, rows per task or T (tests/test_k3_bf16_mv.py G2b checks all of them).
//
// bf16_sk (split K, W = 8, K / 2 = 256 NI: K 10240 / 2560): llama.cpp's own layout, wave v = virtual warp v. Each wave
// walks R rows at once over its slice (NI pairs per lane per row), loads first: U iterations' weight words for the R
// rows and the x pairs for the TT tokens (shared by the R rows) are issued before any is used. The per-virtual-warp sums
// go to LDS (AQ's storage: a work = 0 segment, so BX is not live there, and a workgroup never returns to a segment),
// [R][TT][8]; after a barrier the waves reduce them (row, token) by (row, token) with the final wsum. The old walk had
// one wave per row going through the 8 virtual warps one after another (U = 4 / T = 4: 1 load round in flight): 40
// (K 10240) / 16 (K 2560) dependent load rounds per row at T = 1, 160 at T = 4.
#ifndef K3_BF16_SK
#define K3_BF16_SK 1
#endif
// loads per round (registers: 96-VGPR cap, 0 spills, checked per module): T = 1: K 10240 one row x 10 iterations (20 /
// 5 iterations spill: each load's 64-bit address), K 2560 two rows x 5; T > 1: x token by token, K 10240 two rows x 5,
// K 2560 two rows x 1
#ifdef K3_MT
#ifndef SK20_R
#define SK20_R 2
#endif
#ifndef SK20_U
#define SK20_U 5
#endif
#ifndef SK5_R
#define SK5_R 2
#endif
#ifndef SK5_U
#define SK5_U 1
#endif
#else
#ifndef SK20_R
#define SK20_R 1
#endif
#ifndef SK20_U
#define SK20_U 10
#endif
#ifndef SK5_R
#define SK5_R 2
#endif
#ifndef SK5_U
#define SK5_U 5
#endif
#endif
#ifndef K3_SKB
#define K3_SKB 1
#endif
#define BP ((__attribute__((address_space(3))) float *)AQ)
template <int NI, int R, int U, int TT>
__device__ __forceinline__ void bf16_sk(const Seg & sg, int t) {
    static_assert(NI % U == 0, "U divides NI");
    static_assert(R * TT * 8 * 4 <= AQ_ALLOC * (int)sizeof(block_q8_1), "the partials live in AQ's storage");
    const int v = K3WAVE, lane = K3TID % 32;
    const int r0 = t * sg.rows_per_task, r1 = min(sg.N, r0 + sg.rows_per_task);   // workgroup-uniform (the barriers)
    const int T = seg_T(sg), K = sg.K, N = sg.N, n2 = K / 2;
    GU32 * W2 = (GU32 *)sg.W;
    const GW * X = (const GW *)sg.a;
    float * out = (float *)sg.y;
    const int c0 = 32 * v + lane;
    for (int tg = 0; tg < T; tg += TT) {
        const int nt = min(TT, T - tg);
        for (int rb = r0; rb < r1; rb += R) {
            float acc[R][TT];
#pragma unroll
            for (int r = 0; r < R; r++)
#pragma unroll
                for (int j = 0; j < TT; j++) acc[r][j] = acc0();
            GU32 * wr[R];
#pragma unroll
            for (int r = 0; r < R; r++) wr[r] = W2 + (size_t)min(rb + r, r1 - 1) * n2 + c0;   // past r1: re-read, discarded
#pragma unroll 1
            for (int i0 = 0; i0 < NI; i0 += U) {
                uint32_t wv[R][U];
#pragma unroll
                for (int r = 0; r < R; r++)
#pragma unroll
                    for (int u = 0; u < U; u++) wv[r][u] = wr[r][(i0 + u) * 256];
#pragma unroll
                for (int j = 0; j < TT; j++) {   // T > 1: token by token (x[j] for the U iterations; the weights stay)
                    if (TT > 1 && K3_SKB) __builtin_amdgcn_sched_barrier(0);   // (max-ilp would hoist every token's x loads)
                    float2 xv[U];
#pragma unroll
                    for (int u = 0; u < U; u++) xv[u] = ld2(X + (size_t)(tg + min(j, nt - 1)) * K + 2 * (c0 + (i0 + u) * 256));
#pragma unroll
                    for (int r = 0; r < R; r++)
#pragma unroll
                        for (int u = 0; u < U; u++) {
                            acc[r][j] = __builtin_fmaf(bf16_lo(wv[r][u]), xv[u].x, acc[r][j]);
                            acc[r][j] = __builtin_fmaf(bf16_hi(wv[r][u]), xv[u].y, acc[r][j]);
                        }
                }
            }
#pragma unroll
            for (int r = 0; r < R; r++)
#pragma unroll
                for (int j = 0; j < TT; j++) { const float s = wsum(acc[r][j]); if (lane == 0) BP[(r * TT + j) * 8 + v] = s; }
            __syncthreads();
            for (int q = v; q < R * TT; q += 8) {   // (row rb + q / TT, token tg + q % TT): the final wsum
                const int r = q / TT, j = q % TT;
                const float f = wsum(lane < 8 ? BP[q * 8 + lane] : 0.0f);
                if (lane == 0 && rb + r < r1 && j < nt) out[(size_t)(tg + j) * N + rb + r] = f;
            }
            __syncthreads();
        }
    }
}
// bf16_short (K = 64 W: each (virtual warp, lane) has exactly one pair: the hc up, K 320, W 5): one wave per row as
// before, but the W pairs of x per token are loaded once per task and held (they do not depend on the row), and the
// weight words of R rows (the wave's rows rw, rw + 8, ...) are all issued before any is used. The old walk had W
// dependent load rounds per row (each also issuing U - 1 = 3 discarded re-reads).
// T > 1 (the multi-token module): the token loop unrolled in groups of K3_SH_TB tokens (a sched_barrier between groups: the
// max-ilp scheduler otherwise hoists every token's x loads and spills 19-48 VGPRs), one row per wave per pass. hc up
// (K 320) on the T=4 pass, interleaved x 3 vs the token loop (R 2): 20.48 -> 20.36 ms (R 2 grouped: 20.42; all 4 tokens
// unrolled: spills). T = 1 keeps R 2 (x held per task)
#ifdef K3_MT
#ifndef SH_R
#define SH_R 1
#endif
#ifndef K3_SH_TU
#define K3_SH_TU 1
#endif
#endif
#ifndef SH_R
#define SH_R 2
#endif
#ifndef K3_SH_TB
#define K3_SH_TB 2
#endif
#ifndef K3_SH_TU
#define K3_SH_TU 0
#endif
template <int WV, int R, int TT, typename XP>
__device__ __forceinline__ void bf16_short(const Seg & sg, int t, XP X) {
    const int wave = K3WAVE, lane = K3TID % 32;
    const int r0 = t * sg.rows_per_task, r1 = min(sg.N, r0 + sg.rows_per_task);
    const int T = seg_T(sg), K = sg.K, N = sg.N;
    GU32 * W2 = (GU32 *)sg.W;
    float * out = (float *)sg.y;
    for (int tg = 0; tg < T; tg += TT) {
        const int nt = min(TT, T - tg);
        float2 xv[TT == 1 ? WV : 1];   // T = 1: x held across the task's rows; T > 1: per token from X (LDS: BX)
        if (TT == 1) {
#pragma unroll
            for (int v = 0; v < WV; v++) xv[v] = ld2(X + (size_t)tg * K + 2 * (32 * v + lane));
        }
        for (int rw = r0 + wave; rw < r1; rw += 8 * R) {
            uint32_t wv[R][WV];
#pragma unroll
            for (int r = 0; r < R; r++) {
                GU32 * w2 = W2 + (size_t)min(rw + 8 * r, r1 - 1) * (K / 2) + lane;   // past r1: re-read, discarded
#pragma unroll
                for (int v = 0; v < WV; v++) wv[r][v] = w2[32 * v];
            }
#pragma unroll
            for (int r = 0; r < R; r++) {
#if K3_SH_TU
                if constexpr (TT > 1) {   // tokens unrolled: TT x WV independent reduction chains in flight (tokens past nt
                    float res[TT];        // re-read token nt - 1 and are discarded); each token's order as below
#pragma unroll
                    for (int j = 0; j < TT; j++) {
                        if (j && j % K3_SH_TB == 0) __builtin_amdgcn_sched_barrier(0);   // K3_SH_TB tokens' chains per group
                        const int jj = min(j, nt - 1);
                        float fin = 0.0f;
#pragma unroll
                        for (int v = 0; v < WV; v++) {
                            const float2 x = ld2(X + (size_t)(tg + jj) * K + 2 * (32 * v + lane));
                            float acc = acc0();
                            acc = __builtin_fmaf(bf16_lo(wv[r][v]), x.x, acc);
                            acc = __builtin_fmaf(bf16_hi(wv[r][v]), x.y, acc);
                            const float s = wsum(acc);
                            if (lane == v) fin = s;
                        }
                        res[j] = wsum(fin);
                    }
#pragma unroll
                    for (int j = 0; j < TT; j++)
                        if (j < nt && lane == 0 && rw + 8 * r < r1) out[(size_t)(tg + j) * N + rw + 8 * r] = res[j];
                    continue;
                }
#endif
#pragma unroll 1
                for (int j = 0; j < nt; j++) {   // T > 1: a loop over the tokens (x from X per token; code size)
                    float fin = 0.0f;
#pragma unroll
                    for (int v = 0; v < WV; v++) {
                        const float2 x = TT == 1 ? xv[TT == 1 ? v : 0] : ld2(X + (size_t)(tg + j) * K + 2 * (32 * v + lane));
                        float acc = acc0();
                        acc = __builtin_fmaf(bf16_lo(wv[r][v]), x.x, acc);
                        acc = __builtin_fmaf(bf16_hi(wv[r][v]), x.y, acc);
                        const float s = wsum(acc);
                        if (lane == v) fin = s;
                    }
                    const float res = wsum(fin);
                    if (lane == 0 && rw + 8 * r < r1) out[(size_t)(tg + j) * N + rw + 8 * r] = res;
                }
            }
        }
    }
}
__device__ __forceinline__ void op_bf16_mv(const Seg & sg, int t) {
    const int wave = K3WAVE, lane = K3TID % 32;
    const int r0 = t * sg.rows_per_task, r1 = min(sg.N, r0 + sg.rows_per_task);
    const int T = seg_T(sg), K = sg.K, W = sg.pad1, N = sg.N;
#if K3_BF16_SK
#ifndef K3_SK20
#define K3_SK20 1
#endif
#ifndef K3_SK5
#define K3_SK5 1
#endif
#ifndef K3_SH
#define K3_SH 1
#endif
    if (K3_SK20 && W == 8 && !sg.work && K == 512 * 20) {   // K = 512 NI: every lane's NI pairs in the row
        bf16_sk<20, SK20_R, SK20_U, BF16_TT>(sg, t); return; }
    if (K3_SK5 && W == 8 && !sg.work && K == 512 * 5) {
        bf16_sk<5, SK5_R, SK5_U, BF16_TT>(sg, t); return; }
#ifdef K3_MTP_SWIGLU   // the MTP modules only: eh_proj (K 5120 = 512 NI 10) on the split-K path (bitwise bf16_row, like the others)
#ifdef K3_MT   // (R 2 spilled 3 VGPRs here; R 1 = the module's 1 spill before)
#define SK10_R 1
#define SK10_U 5
#else          // (R 1 U 10: 11 VGPRs spilled)
#define SK10_R 2
#define SK10_U 5
#endif
    if (W == 8 && !sg.work && K == 512 * 10) {
        bf16_sk<10, SK10_R, SK10_U, BF16_TT>(sg, t); return; }
#endif
    if (K3_SH && W == 5 && K == 64 * 5) {   // one pair per (virtual warp, lane)
        if (sg.work) bf16_short<5, SH_R, BF16_TT>(sg, t, (LF *)BX); else bf16_short<5, SH_R, BF16_TT>(sg, t, (const GW *)sg.a);
        return;
    }
#endif
    GU32 * W2 = (GU32 *)sg.W;
    float * out = (float *)sg.y;
    for (int tg = 0; tg < T; tg += BF16_TT) {
        const int nt = min(BF16_TT, T - tg);
        for (int row = r0 + wave; row < r1; row += 8) {
            float res[BF16_TT];
            GU32 * w2 = W2 + (size_t)row * (K / 2);
            if (sg.work) bf16_row<BF16_TT>(w2, (LF *)BX + (size_t)tg * K, K, W, lane, nt, res);
            else bf16_row<BF16_TT>(w2, (const GW *)sg.a + (size_t)tg * K, K, W, lane, nt, res);
            if (lane == 0) {
#pragma unroll
                for (int j = 0; j < BF16_TT; j++) if (j < nt) out[(size_t)(tg + j) * N + row] = res[j];
            }
        }
    }
}
#endif

// OP_ROUTER (one task, one workgroup): llama.cpp's softmax gating with normalised weights (build_moe_ffn: soft_max,
//   argsort_top_k, get_rows, sum_rows, clamp, div). Seg.a = fp32 logits [E = N], Seg.y = int32 ids [k] out,
//   Seg.gs = fp32 weights [k] out, Seg.pad1 = k, Seg.rs = the clamp floor (6.103515625e-5).
// Thread tid owns logits i = tid + jj * 256 (jj < 4, E <= 1024), loaded once and kept in registers through the softmax
// (the same max / sum order as the LDS loops this replaced: per thread i ascending from 0.0f, the wave butterfly, then
// waves 0..7 in order). Top-k by RANK, no serial rounds: rank_i = #{j : p_j > p_i, or p_j == p_i and j < i} (the order
// repeated argmax with lower-index-first ties picks them in); the k ranked lanes park their index in LDS slot `rank`,
// and wave 0 sums p over slots 0..k-1 in order (bitwise the old per-round ws = ws + bv; the old rounds' wavefront
// release fences each waited for a global store: ~1.3 us per selected expert, 15 us per router at k = 10).
// Ranks count on the probabilities' bit patterns (p >= +0, so signed int order == float order, denormals included; the
// padding to a multiple of 32 is INT_MIN): j < i counts p_j >= p_i (q > pi - 1), j > i counts p_j > p_i. Each pass reads
// RK_C candidates per LDS round (16: 32 spills at the 96-VGPR cap) and counts them for both of a lane's logits (i and
// i + 256); the 32-lane groups are 32-aligned, so only the 2 chunks covering the lane's own group need the per-lane
// j < i test, the rest compare against one threshold.
#ifndef RK_C
#define RK_C 16
#endif
// (integer counts: any summation order is exact, so independent partial counts, not one dependent add chain)
__device__ __forceinline__ void router_rank_chunk(const int (&q)[RK_C], int c0, int b, int lane, int pi, int (&c)[2]) {
    if (c0 >= b && c0 < b + 32) {   // (opaque lane: the per-u thresholds must not be hoisted out of the chunk loop)
        int ln = lane - (c0 - b); asm volatile("" : "+v"(ln));
#pragma unroll
        for (int u = 0; u < RK_C; u++) c[u % 2] += q[u] > (u < ln ? pi - 1 : pi);
    } else {
        const int t = c0 < b ? pi - 1 : pi;
#pragma unroll
        for (int u = 0; u < RK_C; u++) c[u % 2] += q[u] > t;
    }
}
// ranks of NE elements (NE = 1 or 2: this lane's logits b0 + lane and b0 + 256 + lane) in one pass over the candidates
template <int NE>
__device__ __forceinline__ void router_rank(const int * pb, int E32, int b0, int lane, int p0, int p1, int & r0, int & r1) {
    int c0s[2] = {0, 0}, c1s[2] = {0, 0};
#pragma unroll 1
    for (int c0 = 0; c0 < E32; c0 += RK_C) {
        int q[RK_C];
#pragma unroll
        for (int u = 0; u < RK_C; u += 4) { const int4 t = *(const int4 *)(pb + c0 + u); q[u] = t.x; q[u + 1] = t.y; q[u + 2] = t.z; q[u + 3] = t.w; }
        router_rank_chunk(q, c0, b0, lane, p0, c0s);
        if (NE == 2) router_rank_chunk(q, c0, b0 + 256, lane, p1, c1s);
    }
    r0 = c0s[0] + c0s[1]; r1 = c1s[0] + c1s[1];
}
__device__ __forceinline__ void op_router(const Seg & sg, int tt) {
#pragma clang fp contract(off)
    __shared__ float rv[8]; __shared__ int probs[1024]; __shared__ int sel[64];   // sel[0..7]: the sum partials first
    const int tid = K3TID, wave = K3WAVE, lane = tid % 32, E = sg.N, k = sg.pad1;
    const float * lg = (const float *)sg.a + (size_t)tt * E;        // task tt = token tt: logits [T, E] -> ids / wts [T, k]
    int * ids_out = (int *)sg.y + tt * k; float * wts_out = (float *)sg.gs + tt * k;
    float v[4]; int idx[4];
#pragma unroll
    for (int jj = 0; jj < 4; jj++) { idx[jj] = tid + jj * 256; v[jj] = idx[jj] < E ? ((const GW *)lg)[idx[jj]] : -INFINITY; }
    float m = -INFINITY;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) if (idx[jj] < E) m = fmaxf(m, v[jj]);
    m = wred<true>(m); if (lane == 0) rv[wave] = m; __syncthreads();
    m = rv[0]; for (int w = 1; w < 8; w++) m = fmaxf(m, rv[w]);
    float sum = 0.0f;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) if (idx[jj] < E) { v[jj] = expf(v[jj] - m); sum += v[jj]; }
    sum = wred<false>(sum); if (lane == 0) sel[wave] = __builtin_bit_cast(int, sum); __syncthreads();
    sum = __builtin_bit_cast(float, sel[0]); for (int w = 1; w < 8; w++) sum += __builtin_bit_cast(float, sel[w]);
    const float inv = 1.0f / sum;
    const int E32 = (E + 31) & ~31;
    int pi[4];
#pragma unroll
    for (int jj = 0; jj < 4; jj++) {
        v[jj] = v[jj] * inv; pi[jj] = __builtin_bit_cast(int, v[jj]);
        if (idx[jj] < E) probs[idx[jj]] = pi[jj]; else if (idx[jj] < E32) probs[idx[jj]] = INT_MIN;
    }
    __syncthreads();
    int cnt[4];
    const int nj = (E - wave * 32 + 255) / 256;   // wave-uniform: how many of this wave's jj slots hold a logit
    if (nj >= 2) router_rank<2>(probs, E32, wave * 32, lane, pi[0], pi[1], cnt[0], cnt[1]);
    else if (nj == 1) router_rank<1>(probs, E32, wave * 32, lane, pi[0], pi[1], cnt[0], cnt[1]);
    if (nj >= 4) router_rank<2>(probs, E32, wave * 32 + 512, lane, pi[2], pi[3], cnt[2], cnt[3]);
    else if (nj == 3) router_rank<1>(probs, E32, wave * 32 + 512, lane, pi[2], pi[3], cnt[2], cnt[3]);
#pragma unroll
    for (int jj = 0; jj < 4; jj++) if (jj < nj && idx[jj] < E && cnt[jj] < k) sel[cnt[jj]] = idx[jj];   // (sum partials all read: 2 barriers ago)
    __syncthreads();
    if (wave == 0) {
        float ws = 0.0f;
        if (k <= 32) {   // lane j holds slot j's p (one LDS round), the sum walks the lanes in order
            const int i = lane < k ? sel[lane] : 0, pj = probs[i];
            for (int j = 0; j < k; j++) ws = ws + __builtin_bit_cast(float, __builtin_amdgcn_readlane(pj, j));
            ws = fmaxf(ws, sg.rs);
            if (lane < k) { ids_out[lane] = i; wts_out[lane] = __builtin_bit_cast(float, pj) / ws; }
        } else {
            for (int j = 0; j < k; j++) ws = ws + __builtin_bit_cast(float, probs[sel[j]]);
            ws = fmaxf(ws, sg.rs);
            for (int j = lane; j < k; j += 32) { const int i = sel[j]; ids_out[j] = i; wts_out[j] = __builtin_bit_cast(float, probs[i]) / ws; }
        }
    }
    __syncthreads();
}

// OP_SHEXP_ADD: out = moe + shexp * sigmoid(g) (ggml_sigmoid, ggml_mul, ggml_add: separate roundings). Seg.a = fp32 moe
//   [N], Seg.W = fp32 shexp [N], Seg.gs = fp32 gate logit [1], Seg.y = fp32 out [N]; a task = rows_per_task elements.
__device__ __forceinline__ void op_shexp_add(const Seg & sg, int t) {
#pragma clang fp contract(off)
    const int i0 = t * sg.rows_per_task, i1 = min(sg.N, i0 + sg.rows_per_task), dd = sg.N / seg_T(sg);
    for (int i = i0 + (int)K3TID; i < i1; i += blockDim.x) {
        const float s = 1.0f / (1.0f + expf(-((const float *)sg.gs)[i / dd]));   // token i / dd's gate logit
        const float g = ((const float *)sg.W)[i] * s;
        ((float *)sg.y)[i] = ((const float *)sg.a)[i] + g;
    }
}

// OP_RMS_MUL: y[r] = (x[r] * rsqrt(mean(x[r]^2) + eps)) * w[r] for rows of length K (ggml_rms_norm then ggml_mul; for
//   the hyper-connection streams, rows = the hc streams and w = the [n_embd, hc] gamma). Seg.a = x [N, K], Seg.W = fp32
//   w [N, K], Seg.y = out [N, K], Seg.rs = eps; one task per row (256 threads, block reduction of the squares).
__device__ __forceinline__ void op_rms_mul(const Seg & sg, int t) {
    __shared__ float part[8];
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, K = sg.K;
    const float * x = (const float *)sg.a + (size_t)t * K, * w = (const float *)sg.W + (size_t)(sg.pad1 ? t % sg.pad1 : t) * K;
    float * y = (float *)sg.y + (size_t)t * K;
    if (K == GLUE_NI * 256) {   // n_embd rows: x and w loaded once, together; x stays in registers
        float xv[GLUE_NI], wv[GLUE_NI];
#pragma unroll
        for (int j = 0; j < GLUE_NI; j++) xv[j] = ((GF *)x)[tid + j * 256];
#pragma unroll
        for (int j = 0; j < GLUE_NI; j++) wv[j] = ((GF *)w)[tid + j * 256];
        __builtin_amdgcn_sched_barrier(0);
        float ss = acc0();
#pragma unroll
        for (int j = 0; j < GLUE_NI; j++) ss += xv[j] * xv[j];
        ss = wsum(ss); if (lane == 0) part[wave] = ss; __syncthreads();
        float tot = 0.0f; for (int q = 0; q < 8; q++) tot += part[q];
        const float scale = rsqrtf(tot / K + sg.rs);
#pragma unroll
        for (int j = 0; j < GLUE_NI; j++) { xv[j] = (xv[j] * scale) * wv[j]; ((GW *)y)[tid + j * 256] = xv[j]; }
        if (sg.asum) q8_emit_n<GLUE_NI>(sg.asum, (size_t)t * K + tid, 256, xv);
        __syncthreads();
        return;
    }
    float ss = 0.0f;
    for (int i = tid; i < K; i += blockDim.x) ss += x[i] * x[i];
    ss = warp_reduce_sum<32>(ss); if (lane == 0) part[wave] = ss; __syncthreads();
    float tot = 0.0f; for (int q = 0; q < 8; q++) tot += part[q];
    const float scale = rsqrtf(tot / K + sg.rs);
    for (int i = tid; i < K; i += blockDim.x) {
        const float v = (x[i] * scale) * w[i];
        y[i] = v;
        if (sg.asum) q8_emit((void *)sg.asum, (size_t)t * K + i, v);   // optional q8_1 copy (K % 32 == 0)
    }
    __syncthreads();
}

// OP_HC_MEAN: out[i] = rs * sum_s x[s * n + i] * sigmoid(g[s * n + i]) (ggml_dsv4_hc_pre_gated, scale = 1 / hc).
//   Seg.a = xn [hc, n], Seg.W = fp32 gate [hc, n], Seg.y = out [n], N = n, K = hc; a task = rows_per_task elements.
__device__ __forceinline__ void op_hc_mean(const Seg & sg, int t) {
    const int i0 = t * sg.rows_per_task, i1 = min(sg.N, i0 + sg.rows_per_task);
    const float * x = (const float *)sg.a, * g = (const float *)sg.W;
    for (int i = i0 + (int)K3TID; i < i1; i += blockDim.x) {
        const int Dt = sg.pad1 ? sg.pad1 : sg.N, tt = i / Dt, ii = i - tt * Dt;   // [T, hc, D] inputs, [T, D] out
        float sum = 0.0f;
        for (int h = 0; h < sg.K; h++) sum += x[(size_t)(tt * sg.K + h) * Dt + ii] * (1.0f / (1.0f + expf(-g[(size_t)(tt * sg.K + h) * Dt + ii])));
        const float v = sg.rs * sum;
        ((float *)sg.y)[i] = v;
        if (sg.asum) q8_emit((void *)sg.asum, i, v);   // optional q8_1 copy (task bounds multiples of 32)
    }
}

// OP_HC_POST: res[s * n + i] = block[i] * p[s] + res[s * n + i], p[s] = 2 * sigmoid(inject[s] / hc) (ggml_scale,
//   ggml_sigmoid, ggml_scale, then ggml_dsv4_hc_post). Seg.a = block [n], Seg.gs = inject [hc], Seg.W = fp32 residual in
//   [hc, n], Seg.y = residual out [hc, n] (may equal the input), N = n, K = hc, rs = 1 / hc.
__device__ __forceinline__ void op_hc_post(const Seg & sg, int t) {
#pragma clang fp contract(off)
    const int i0 = t * sg.rows_per_task, i1 = min(sg.N, i0 + sg.rows_per_task);
    const float * b = (const float *)sg.a, * r = (const float *)sg.W, * inj = (const float *)sg.gs;
    const int Dt = sg.pad1 ? sg.pad1 : sg.N;   // elements [T, D]: token i / D, residual [T, hc, D]
    for (int i = i0 + (int)K3TID; i < i1; i += blockDim.x) {
        const int tt = i / Dt, ii = i - tt * Dt;
        for (int h = 0; h < sg.K; h++) {
            const float v = inj[tt * sg.K + h] * sg.rs; const float sgm = 1.0f / (1.0f + expf(-v)); const float ph = sgm * 2.0f;
            const size_t o = (size_t)(tt * sg.K + h) * Dt + ii;
            {
#pragma clang fp contract(fast)   // the dsv4_hc_post kernel itself contracts x * p + residual
                ((float *)sg.y)[o] = b[i] * ph + r[o];
            }
        }
    }
}

// ---- GDN (gated delta net) decode step, as llama.cpp's qwen4exp build_layer_attn_linear + gated_delta_net_cuda ----
// OP_GDN_CONV: per channel c: in = [st[c][0..2], x[c]], out[c] = silu(sum_j in[j] * w[c][j]) (ssm_conv, then silu),
//   and the state shifts to [in[1..3]]. Seg.a = x [C] (qkv_mixed), Seg.W = fp32 w [C, 4], Seg.gs = fp32 state [C, 3]
//   (updated in place), Seg.y = out [C]; N = C; a task = rows_per_task channels.
__device__ __forceinline__ void op_gdn_conv(const Seg & sg, int t) {
    const int c0 = t * sg.rows_per_task, c1 = min(sg.N, c0 + sg.rows_per_task);
    const float * x = (const float *)sg.a, * w = (const float *)sg.W;
    float * st = (float *)sg.gs, * xp = (float *)sg.aux;        // committed state; the previous pass's inputs [T, C]
    const int C = sg.N, T = seg_T(sg), mprev = ((const int *)sg.asum)[1];
    // every load of the channel first (state, the 4 taps as one float4, this pass's first input, the previous pass's
    // first input); then the same arithmetic in the same order
    for (int c = c0 + (int)K3TID; c < c1; c += blockDim.x) {
        GF * stg = (GF *)st + (size_t)c * 3;
        float s0 = stg[0], s1 = stg[1], s2 = stg[2];
        const float4 w4 = *(const __attribute__((address_space(1))) float4 *)(w + (size_t)c * 4);
        const float x0 = ((GF *)x)[c], xp0 = ((GF *)xp)[c];
        __builtin_amdgcn_sched_barrier(0);
        for (int j = 0; j < mprev; j++) { s0 = s1; s1 = s2; s2 = j == 0 ? xp0 : xp[(size_t)j * C + c]; }   // commit the accepted tokens
        GW * stw = (GW *)st + (size_t)c * 3;
        stw[0] = s0; stw[1] = s1; stw[2] = s2;
        const float wv[4] = {w4.x, w4.y, w4.z, w4.w};
        for (int tt = 0; tt < T; tt++) {
            const float xv = tt == 0 ? x0 : x[(size_t)tt * C + c];
            const float in[4] = {s0, s1, s2, xv};
            float sumf = 0.0f;
            for (int j = 0; j < 4; j++) sumf += in[j] * wv[j];
            ((GW *)sg.y)[(size_t)tt * C + c] = gg_silu(sumf);
            s0 = s1; s1 = s2; s2 = xv;
            ((GW *)xp)[(size_t)tt * C + c] = xv;                   // this pass's input, for the next pass's commit
        }
    }
}

// OP_GDN_L2: per head of K values: y = (x * rsqrt(sum(x^2) / K + rs)) * (1 / sqrt(K)) (build_gdn_l2_norm: rms_norm with
//   eps / K, then ggml_scale). Seg.a = in, Seg.y = out, N = heads, K = head dim, rs = eps / K; one task per head.
__device__ __forceinline__ void op_gdn_l2(const Seg & sg, int t) {
    __shared__ float part[8];
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, K = sg.K;
    const int heads = sg.N / seg_T(sg), tt = t / heads, hh = t - tt * heads;
    const float * x = (const float *)sg.a + (size_t)tt * (sg.pad1 ? sg.pad1 : heads * K) + (size_t)hh * K;
    float * y = (float *)sg.y + (size_t)t * K;
    float ss = 0.0f;
    for (int i = tid; i < K; i += blockDim.x) ss += x[i] * x[i];
    ss = warp_reduce_sum<32>(ss); if (lane == 0) part[wave] = ss; __syncthreads();
    float tot = 0.0f; for (int q = 0; q < 8; q++) tot += part[q];
    const float scale = rsqrtf(tot / K + sg.rs), s2 = 1.0f / sqrtf((float)K);
    for (int i = tid; i < K; i += blockDim.x) y[i] = (scale * x[i]) * s2;
    __syncthreads();
}

// OP_GDN_GATES: gate[h] = softplus(alpha[h] + dt[h]) * a[h]; beta_sig[h] = sigmoid(beta[h]) (separate roundings, as
//   ggml_add, ggml_softplus, ggml_mul, ggml_sigmoid). Seg.a = alpha [H], Seg.W = fp32 beta [H], Seg.gs = fp32 dt [H],
//   Seg.asum = fp32 ssm_a [H] (cast), Seg.y = out [2H] (gate, then beta_sig); N = H; one task.
__device__ __forceinline__ void op_gdn_gates(const Seg & sg) {
#pragma clang fp contract(off)
    const float * al = (const float *)sg.a, * be = (const float *)sg.W, * dt = sg.gs, * A = (const float *)sg.asum;
    float * y = (float *)sg.y;
    const int H = sg.N;
    for (int q = K3TID; q < seg_T(sg) * H; q += blockDim.x) {   // q = token * H + head; out [T, 2H]
        const int tt = q / H, h = q - tt * H;
        const float ab = al[q] + dt[h];
        const float sp = (ab > 20.0f) ? ab : logf(1.0f + expf(ab));
        y[tt * 2 * H + h] = sp * A[h];
        y[tt * 2 * H + H + h] = 1.0f / (1.0f + expf(-be[q]));
    }
}

// OP_GDN_STEP: the gated delta rule per value head h, k / q head = h % nk (ggml_repeat tiling; K3_GDN_GROUPED: h / (H / nk), qwen3next); state M[h][col][i]
//   (llama.cpp's transposed layout: S[i][col] at col * S + i). Per token: kv = sum_i M[col][i] k[i];
//   delta = (v[col] - g kv) beta; M = g M + k[i] delta; attn[col] = scale sum_i M[col][i] q[i], g = exp(gate[h]),
//   scale = 1 / sqrt(S). Lazy commit (speculative verification): the stored state is the one before the previous pass's
//   accepted tokens; each pass first replays those mprev tokens from their saved inputs (OP_GDN_SAVE), stores the
//   result, then runs its T tokens without storing (a rejected draft never touches the stored state). The same
//   update code (explicit fmaf) runs for the replay and the new tokens, so a T-token pass equals T one-token passes.
//   Seg.a = q|k [T, 2, nk, S] (l2-normed), Seg.W = v [T, ...] (token stride pad2), Seg.gs = gates [T, 2H] (gate,
//   beta_sig), Seg.aux = the layer's block [state H*S*S | saved inputs T x (nk*S + H*S + 2H)], Seg.asum = int32 pass
//   params ([1] = mprev), Seg.y = attn [T, H, S]; N = H, K = S, pad1 = nk, team = T.
__device__ __forceinline__ float4 gdn_upd(const float4 m, const float4 k4, float kv, float vcol, float g, float beta) {
    // the state column update; explicit operations so every inlined copy rounds alike
    const float t1 = g * kv, t2 = vcol - t1, delta = t2 * beta;
    float4 sv;
    sv.x = fmaf(k4.x, delta, g * m.x); sv.y = fmaf(k4.y, delta, g * m.y);
    sv.z = fmaf(k4.z, delta, g * m.z); sv.w = fmaf(k4.w, delta, g * m.w);
    return sv;
}
__device__ __forceinline__ float gdn_dot(const float4 m, const float4 k4) {
    return fmaf(m.w, k4.w, fmaf(m.z, k4.z, fmaf(m.y, k4.y, m.x * k4.x)));
}
__device__ __forceinline__ void op_gdn_step(const Seg & sg, int t) {
    // S == 128 (host-checked): a state column is 128 contiguous floats, one float4 per lane of a wave, so every load is
    // coalesced; a task is 32 columns of one head (4 per wave), the two dot products are wave reductions
    constexpr int S = 128, CPT = 32, CPW = CPT / 8;
#ifdef K3_GDN_GROUPED   // qwen3next: q / k repeat-interleaved (ggml_repeat_4d over a unit axis): value head h reads key head h / (H / nk)
    const int h = t / (S / CPT), c0 = (t % (S / CPT)) * CPT, nk = sg.pad1, H = sg.N, kh = h / (H / nk), T = seg_T(sg);
#else
    const int h = t / (S / CPT), c0 = (t % (S / CPT)) * CPT, nk = sg.pad1, kh = h % nk, H = sg.N, T = seg_T(sg);
#endif
    const int lane = K3TID % 32, wave = K3WAVE, mprev = ((const int *)sg.asum)[1];
    const int RS = nk * S + H * S + 2 * H;                         // a saved input record per token
    float * base = (float *)sg.aux; const float * rec = base + (size_t)H * S * S;
    float4 * M = (float4 *)(base + (size_t)h * S * S);
    const float scale = 1.0f / sqrtf((float)S);
    float4 m[CPW];
#pragma unroll
    for (int j = 0; j < CPW; j++) m[j] = M[(size_t)(c0 + wave * CPW + j) * (S / 4) + lane];
#pragma unroll 1
    for (int r = 0; r < mprev; r++) {                              // replay the previous pass's accepted tokens
        const float * rr = rec + (size_t)r * RS;
        const float4 k4 = ((const float4 *)(rr + (size_t)kh * S))[lane];
        const float * v = rr + nk * S + (size_t)h * S;
        const float g = expf(rr[nk * S + H * S + h]), beta = rr[nk * S + H * S + H + h];
        float kv[CPW];
#pragma unroll
        for (int j = 0; j < CPW; j++) kv[j] = warp_reduce_sum<32>(gdn_dot(m[j], k4));
#pragma unroll
        for (int j = 0; j < CPW; j++) m[j] = gdn_upd(m[j], k4, kv[j], v[c0 + wave * CPW + j], g, beta);
    }
#pragma unroll
    for (int j = 0; j < CPW; j++) M[(size_t)(c0 + wave * CPW + j) * (S / 4) + lane] = m[j];   // the committed state
#pragma unroll 1
    for (int tt = 0; tt < T; tt++) {
        const float * qk = (const float *)sg.a + (size_t)tt * 2 * nk * S;
        const float4 k4 = ((const float4 *)(qk + (size_t)(nk + kh) * S))[lane];
        const float4 q4 = ((const float4 *)(qk + (size_t)kh * S))[lane];
        const float * v = (const float *)sg.W + (size_t)tt * sg.pad2 + (size_t)h * S;
        const float * gt = sg.gs + (size_t)tt * 2 * H;
        const float g = expf(gt[h]), beta = gt[H + h];
        float kv[CPW], at[CPW];
#pragma unroll
        for (int j = 0; j < CPW; j++) kv[j] = warp_reduce_sum<32>(gdn_dot(m[j], k4));
#pragma unroll
        for (int j = 0; j < CPW; j++) {
            m[j] = gdn_upd(m[j], k4, kv[j], v[c0 + wave * CPW + j], g, beta);
            at[j] = warp_reduce_sum<32>(gdn_dot(m[j], q4));
        }
        if (lane == 0) {
#pragma unroll
            for (int j = 0; j < CPW; j++) ((float *)sg.y)[((size_t)tt * H + h) * S + c0 + wave * CPW + j] = at[j] * scale;
        }
    }
}

// OP_GDN_SAVE: after the step, this pass's step inputs per token (k [nk, S], v [H, S], gates [2H]) into the layer's
//   saved-input records (read by the next pass's commit replay). Same Seg fields as OP_GDN_STEP; tasks = tokens.
__device__ __forceinline__ void op_gdn_save(const Seg & sg, int tt) {
    constexpr int S = 128;
    const int nk = sg.pad1, H = sg.N, RS = nk * S + H * S + 2 * H;
    float * rr = (float *)sg.aux + (size_t)H * S * S + (size_t)tt * RS;
    const float * qk = (const float *)sg.a + (size_t)tt * 2 * nk * S, * v = (const float *)sg.W + (size_t)tt * sg.pad2;
    const float * gt = sg.gs + (size_t)tt * 2 * H;
    for (int i = K3TID; i < nk * S; i += blockDim.x) rr[i] = qk[nk * S + i];
    for (int i = K3TID; i < H * S; i += blockDim.x) rr[nk * S + i] = v[i];
    for (int i = K3TID; i < 2 * H; i += blockDim.x) rr[nk * S + H * S + i] = gt[i];
}

// OP_GDN_NORM: per value head: out = ((o * rsqrt(mean(o^2) + eps)) * w) * sigmoid(z) (build_norm RMS with the ssm_norm
//   weight, then ggml_mul by ggml_sigmoid(z): qwen4exp's sigmoid output gate; K3_GDN_SILU: gg_silu(z), qwen3next's / qwen35moe's). Seg.a = o [H, S], Seg.W = fp32 w [S],
//   Seg.gs = fp32 z [H, S], Seg.y = out [H, S], N = H, K = S, rs = eps; one task per head.
__device__ __forceinline__ void op_gdn_norm(const Seg & sg, int t) {
#pragma clang fp contract(off)
    __shared__ float part[8];
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, K = sg.K;
    const float * o = (const float *)sg.a + (size_t)t * K, * w = (const float *)sg.W, * z = sg.gs + (size_t)t * K;
    float * y = (float *)sg.y + (size_t)t * K;
    float ss = 0.0f;
    for (int i = tid; i < K; i += blockDim.x) ss += o[i] * o[i];
    ss = warp_reduce_sum<32>(ss); if (lane == 0) part[wave] = ss; __syncthreads();
    float tot = 0.0f; for (int q = 0; q < 8; q++) tot += part[q];
    const float scale = rsqrtf(tot / K + sg.rs);
    for (int i = tid; i < K; i += blockDim.x) {
        const float nrm = (scale * o[i]) * w[i];
#ifdef K3_GDN_SILU   // qwen3next / qwen35moe build_norm_gated: ggml_mul(normalized, ggml_silu(gate))
        const float v = nrm * gg_silu(z[i]);
#else
        const float sgm = 1.0f / (1.0f + expf(-z[i]));
        const float v = nrm * sgm;
#endif
        y[i] = v;
        if (sg.asum) q8_emit((void *)sg.asum, (size_t)t * K + i, v);
    }
    __syncthreads();
}

// ---- attention (qwen4exp build_layer_attn, full attention: equal to QSA while its block selection covers the context) ----
// OP_QK_NR: per head: y = rms_norm(x) * w (build_norm RMS), then RoPE on the first n_rot dims, NEOX pairs (j, j + n_rot/2),
//   theta = pos * theta_scale^j (rope_multi with equal text positions in every section, no YaRN: freq_scale 1,
//   ext_factor 0, mscale 1). Seg.a = in base, Seg.W = fp32 norm w [D], Seg.asum = int32 pos [1] (cast), Seg.y = out
//   [heads, D]; N = heads, K = D, pad1 = input head stride (floats), out_f32 = n_rot, rs = theta_scale,
//   pad2 = eps bits; one task per head.
__device__ __forceinline__ void op_qk_nr(const Seg & sg, int t) {
    __shared__ float part[8]; __shared__ float nrm[512];
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, D = sg.K, nrot = sg.out_f32;
    const float * x = (const float *)sg.a + (size_t)t * sg.pad1, * w = (const float *)sg.W;
    float * y = (float *)sg.y + (size_t)t * D;
    const float eps = __builtin_bit_cast(float, sg.pad2);
    float ss = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) ss += x[i] * x[i];
    ss = warp_reduce_sum<32>(ss); if (lane == 0) part[wave] = ss; __syncthreads();
    float tot = 0.0f; for (int q = 0; q < 8; q++) tot += part[q];
    const float scale = rsqrtf(tot / D + eps);
    for (int i = tid; i < D; i += blockDim.x) nrm[i] = (scale * x[i]) * w[i];
    __syncthreads();
    const float pos = (float)(((const int *)sg.asum)[0] + t / (sg.N / seg_T(sg)));   // task t = token * heads + head
    for (int i = tid; i < D; i += blockDim.x) {
        if (i >= nrot) { y[i] = nrm[i]; continue; }
        const int j = i < nrot / 2 ? i : i - nrot / 2;
        const float theta = pos * powf(sg.rs, (float)j);
        const float c = cosf(theta), sn = sinf(theta);
        const float x0 = nrm[j], x1 = nrm[j + nrot / 2];
        y[i] = i < nrot / 2 ? x0 * c - x1 * sn : x0 * sn + x1 * c;
    }
    __syncthreads();
}

// KV cache layouts per position: [K | V], M values each (M = kv_heads * D). Seg.work = 0: f16 (2 M halves);
// Seg.work = 1: q8_0 (llama.cpp's block_q8_0 {half d; int8 qs[32]}, 2 * M / 32 blocks), as the daily server runs (-ctk / -ctv q8_0).
__device__ __forceinline__ size_t kv_pos_bytes(int M, int type) { return type ? (size_t)2 * (M / 32) * sizeof(block_q8_0) : (size_t)4 * M; }
__device__ __forceinline__ float kv_get(const uint8_t * pos_base, int M, int type, int which, int i) {   // which 0 = K, 1 = V
    if (type == 0) return __half2float(((const half *)pos_base)[(size_t)which * M + i]);
    const block_q8_0 * b = (const block_q8_0 *)pos_base + (size_t)which * (M / 32) + i / 32;
    return (float)b->qs[i % 32] * __half2float(b->d);
}

// OP_KV_APPEND: cache[pos] = (k, v): f16, or q8_0 quantized as ggml's quantize_f32_q8_0_block (d = amax / 127,
//   q = roundf(x * (1 / d))). Seg.a = k fp32 [M], Seg.W = fp32 v [M], Seg.y = cache base, Seg.asum = int32 pos [1];
//   N = M, work = cache type; one task.
__device__ __forceinline__ void op_kv_append(const Seg & sg, int tt) {
    const int pos = ((const int *)sg.asum)[0] + tt, M = sg.N;   // task tt = token tt: k / v [T, M]
    uint8_t * base = (uint8_t *)sg.y + (size_t)pos * kv_pos_bytes(M, sg.work);
    if (sg.work == 0) {
        half * c = (half *)base;
        for (int i = K3TID; i < M; i += blockDim.x) {
            c[i] = __float2half(((const float *)sg.a)[(size_t)tt * M + i]);
            c[M + i] = __float2half(((const float *)sg.W)[(size_t)tt * M + i]);
        }
        return;
    }
    const int nb = M / 32;
    for (int b = K3TID; b < 2 * nb; b += blockDim.x) {     // one thread per block, as the reference loop
        const float * x = (b < nb ? (const float *)sg.a : (const float *)sg.W) + (size_t)tt * M + (size_t)(b % nb) * 32;
        block_q8_0 * y = (block_q8_0 *)base + b;
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[j]));
        const float d = amax / ((1 << 7) - 1), id = d ? 1.0f / d : 0.0f;
        y->d = d;
        for (int j = 0; j < 32; j++) y->qs[j] = roundf(x[j] * id);
    }
}

// OP_ATTN_DEC: out[h] = softmax_t(scale q_h . K[t][g]) . V[t][g] over t = 0..pos, g = h / (heads / kv_heads), K and V
//   read from the f16 or q8_0 cache in fp32 (Q stays fp32). Seg.a = q fp32 [heads, D], Seg.W = cache base,
//   Seg.asum = int32 pos [1], Seg.y = out fp32 [heads, D]; N = heads, K = D (<= 512), pad1 = kv_heads, rs = scale,
//   work = cache type; one task per head. Positions go in chunks of 512 scores in LDS with an online softmax (running
//   max and sum, the accumulators rescaled per chunk); up to 512 positions this is exactly the two-pass softmax.
__device__ __forceinline__ void op_attn_dec(const Seg & sg, int h) {
    constexpr int CH = 512;
    __shared__ float sc[CH]; __shared__ float red[8];
    const int heads = sg.N / seg_T(sg), tt = h / heads;          // task = token * heads + head
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, D = sg.K, nkv = sg.pad1, g = (h - tt * heads) / (heads / nkv);
    const int npos = ((const int *)sg.asum)[0] + tt + 1, M = nkv * D, type = sg.work;
    const size_t pb = kv_pos_bytes(M, type);
    const float * q = (const float *)sg.a + (size_t)h * D;
    const uint8_t * cache = sg.W;
    float run_m = -INFINITY, run_s = 0.0f, acc[2] = {0.0f, 0.0f};
#pragma unroll 1
    for (int c0 = 0; c0 < npos; c0 += CH) {
        const int n = min(CH, npos - c0);
        for (int t = wave; t < n; t += 8) {
            float a = 0.0f;
            for (int d = lane; d < D; d += 32) a += q[d] * kv_get(cache + (size_t)(c0 + t) * pb, M, type, 0, g * D + d);
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
            for (int t = 0; t < n; t++) part += sc[t] * kv_get(cache + (size_t)(c0 + t) * pb, M, type, 1, g * D + d);
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

// OP_GATE_MUL: out[h * D + d] = a[h * D + d] * sigmoid(gate[h * stride + off + d]) (ggml_sigmoid then ggml_mul).
//   Seg.a = a fp32, Seg.W = fp32 gate base, Seg.y = out; N = heads * D, K = D, pad1 = gate head stride, pad2 = offset.
__device__ __forceinline__ void op_gate_mul(const Seg & sg, int t) {
    const int i0 = t * sg.rows_per_task, i1 = min(sg.N, i0 + sg.rows_per_task);
    const float * a = (const float *)sg.a, * gt = (const float *)sg.W;
    for (int i = i0 + (int)K3TID; i < i1; i += blockDim.x) {
        const int h = i / sg.K, d = i - h * sg.K;
        const float sgm = 1.0f / (1.0f + expf(-gt[(size_t)h * sg.pad1 + sg.pad2 + d]));
        const float v = a[i] * sgm;
        ((float *)sg.y)[i] = v;
        if (sg.asum) q8_emit((void *)sg.asum, i, v);
    }
}

// ---- PLE n-gram memory injection (qwen4exp build_ple; the hashed table rows arrive as the embedding) ----
// OP_PLE_GATE (one task): s[st] = (sum_i keyn[st, i] * queryn[st, i]) * (1 / sqrt(D)); gate[st] =
//   sigmoid(sgn(s) * sqrt(clamp(|s|, 1e-6, inf))); gated[st, i] = value[i] * gate[st] (ggml_sum_rows, ggml_scale,
//   ggml_abs / clamp / sqrt / sgn / mul / sigmoid, ggml_mul; separate roundings). Seg.a = keyn [hc, D], Seg.W = fp32
//   queryn [hc, D], Seg.gs = fp32 value [D], Seg.y = gated out [hc, D], Seg.asum = fp32 gate out [hc] (cast);
//   N = D, K = hc.
__device__ __forceinline__ void op_ple_gate(const Seg & sg, int tt) {
#pragma clang fp contract(off)
    __shared__ float part[8]; __shared__ float gate[8];
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, D = sg.N;
    const float * kn = (const float *)sg.a + (size_t)tt * sg.K * D, * qn = (const float *)sg.W + (size_t)tt * sg.K * D, * val = sg.gs + (size_t)tt * D;
    for (int st = 0; st < sg.K; st++) {
        float acc = 0.0f;
        for (int i = tid; i < D; i += blockDim.x) acc += kn[(size_t)st * D + i] * qn[(size_t)st * D + i];
        acc = warp_reduce_sum<32>(acc); if (lane == 0) part[wave] = acc; __syncthreads();
        if (tid == 0) {
            float sum = 0.0f; for (int w = 0; w < 8; w++) sum += part[w];
            const float s = sum * (1.0f / sqrtf((float)D));
            const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
            const float sgn = s > 0.0f ? 1.0f : (s < 0.0f ? -1.0f : 0.0f);
            const float v = sgn * mag;
            gate[st] = 1.0f / (1.0f + expf(-v));
            ((float *)sg.asum)[tt * sg.K + st] = gate[st];
        }
        __syncthreads();
    }
    for (int c = tid; c < sg.K * D; c += blockDim.x) ((float *)sg.y)[(size_t)tt * sg.K * D + c] = val[c % D] * gate[c / D];
    __syncthreads();
}

// OP_PLE_CONV: per channel c: the dilated causal conv over the normalised history, taps k = 0..3 at t - (3 - k) * dil
//   (out = ((x0 w0 + x1 w1) + x2 w2) + x3 w3 with separate roundings: ggml_mul / ggml_add per tap), then silu; the history
//   [hist = 3 * dil per channel, oldest first] shifts by one. Seg.a = x (normalised) [C], Seg.W = f16 w [C, 4] (as
//   bytes), Seg.gs = fp32 history [C, hist] (in place), Seg.y = out [C]; N = C, pad1 = dil.
__device__ __forceinline__ void op_ple_conv(const Seg & sg, int t) {
#pragma clang fp contract(off)
    const int dil = sg.pad1, hist = 3 * dil;
    const int c0 = t * sg.rows_per_task, c1 = min(sg.N, c0 + sg.rows_per_task);
    const float * x = (const float *)sg.a; const half * w = (const half *)sg.W; float * st = (float *)sg.gs;
    float * xp = (float *)sg.aux;                                 // the previous pass's inputs [T, C] (lazy commit)
    const int C = sg.N, T = seg_T(sg), mprev = ((const int *)sg.asum)[1];
    for (int c = c0 + (int)K3TID; c < c1; c += blockDim.x) {
        float * h = st + (size_t)c * hist;
        for (int r = 0; r < mprev; r++) {                          // commit the accepted tokens into the history
            for (int j = 0; j < hist - 1; j++) h[j] = h[j + 1];
            h[hist - 1] = xp[(size_t)r * C + c];
        }
        for (int tt = 0; tt < T; tt++) {
            const float xc = x[(size_t)tt * C + c];
            float conv = 0.0f;
            for (int k = 0; k < 4; k++) {
                const int back = (3 - k) * dil;                          // positions back from this token
                const float xv = back == 0 ? xc : (back <= tt ? x[(size_t)(tt - back) * C + c] : h[hist - (back - tt)]);
                const float term = xv * __half2float(w[(size_t)c * 4 + k]);
                conv = k == 0 ? term : conv + term;
            }
            ((float *)sg.y)[(size_t)tt * C + c] = gg_silu(conv);
        }
        for (int tt = 0; tt < T; tt++) xp[(size_t)tt * C + c] = x[(size_t)tt * C + c];
    }
}

// OP_ADD_SUM: y = y + (a + b) (ggml_add(hidden, ggml_add(gated, conv_out))). Seg.a = a, Seg.W = fp32 b, Seg.y = y in
//   place; N = n; a task = rows_per_task elements.
__device__ __forceinline__ void op_add_sum(const Seg & sg, int t) {
#pragma clang fp contract(off)
    const int i0 = t * sg.rows_per_task, i1 = min(sg.N, i0 + sg.rows_per_task);
    for (int i = i0 + (int)K3TID; i < i1; i += blockDim.x) {
        const float ab = ((const float *)sg.a)[i] + ((const float *)sg.W)[i];
        ((float *)sg.y)[i] = ((float *)sg.y)[i] + ab;
    }
}

// OP_HADAMARD: in place, the orthonormal Walsh-Hadamard transform (Sylvester order: H[i][j] = (-1)^popcount(i & j) /
//   sqrt(n), its own inverse) of each contiguous chunk of n values (power of 2, 64 <= n <= 256). llama.cpp rotates q / k
//   (n = head dim) and v (n = 64) before a quantized KV cache and rotates the attention output back
//   (llama-kv-cache.cpp attn_rot, llama_mul_mat_hadamard). Butterflies in LDS, then the exact power-of-2 scale.
//   Seg.y = data (fp32); N = chunks, K = n; a task = rows_per_task chunks.
__device__ __forceinline__ void op_hadamard(const Seg & sg, int t) {
#pragma clang fp contract(off)
    __shared__ float hb[256];
    const int n = sg.K, tid = K3TID;
    const float scale = 1.0f / sqrtf((float)n);
    float * x = (float *)sg.y;
    const int c0 = t * sg.rows_per_task, c1 = min(sg.N, c0 + sg.rows_per_task);
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

// OP_HC_POST_RMS (one task per stream h): the hyper-connection residual update (op_hc_post) and the next mix's RMS norm
//   (op_rms_mul with a q8_1 copy) of that stream in one pass: res[h] = block * 2 sigmoid(inj[h] * rs) + res[h], then
//   xn[h] = (res[h] * rsqrt(mean(res[h]^2) + eps)) * w[h]. The same expressions in the same order as the two ops (bitwise).
//   Seg.a = block [N], Seg.W = res [hc, N] (in place), Seg.gs = inj [hc], Seg.aux = norm w [hc, N], Seg.y = xn,
//   Seg.asum = q8_1 of xn; N = D, K = hc, rs = inject scale, pad2 = eps bits.
__device__ __forceinline__ void op_hc_post_rms(const Seg & sg, int q) {
    __shared__ float part[8];
    const int tid = K3TID, lane = tid % 32, wave = tid / 32, N = sg.N, h = q % sg.K, tt = q / sg.K;   // task = (token, stream)
    const float * __restrict__ b = (const float *)sg.a + (size_t)tt * N, * __restrict__ inj = (const float *)sg.gs;
    const float * __restrict__ w = (const float *)sg.aux + (size_t)h * N;
    float * __restrict__ r = (float *)sg.W + (size_t)q * N, * __restrict__ xn = (float *)sg.y + (size_t)q * N;
    if (N == GLUE_NI * 256) {   // n_embd 2560: one exact-size pass, every load issued first (no per-element branch)
        GF * bg = (GF *)b + tid, * rg = (GF *)r + tid, * wg = (GF *)w + tid;
        GW * rw = (GW *)r + tid, * xw = (GW *)xn + tid;
        float bv[GLUE_NI], nr[GLUE_NI], wv[GLUE_NI];
#pragma unroll
        for (int j = 0; j < GLUE_NI; j++) { bv[j] = bg[j * 256]; nr[j] = rg[j * 256]; }
#pragma unroll
        for (int j = 0; j < GLUE_NI; j++) wv[j] = wg[j * 256];
        float p;
        {
#pragma clang fp contract(off)
            const float v = inj[q] * sg.rs; const float sgm = 1.0f / (1.0f + expf(-v)); p = sgm * 2.0f;
        }
        float ss = acc0();
#pragma unroll
        for (int j = 0; j < GLUE_NI; j++) {
            {
#pragma clang fp contract(fast)   // as op_hc_post (the dsv4_hc_post kernel contracts x * p + residual)
                nr[j] = bv[j] * p + nr[j];
            }
            ss += nr[j] * nr[j];          // as op_rms_mul
        }
        ss = wsum(ss); if (lane == 0) part[wave] = ss; __syncthreads();
#pragma unroll
        for (int j = 0; j < GLUE_NI; j++) rw[j * 256] = nr[j];   // after the barrier: it does not wait for them
        float tot = 0.0f; for (int q = 0; q < 8; q++) tot += part[q];
        const float scale = rsqrtf(tot / N + __int_as_float(sg.pad2));
#pragma unroll
        for (int j = 0; j < GLUE_NI; j++) { nr[j] = (nr[j] * scale) * wv[j]; xw[j * 256] = nr[j]; }
        if (sg.asum) q8_emit_n<GLUE_NI>(sg.asum, (size_t)q * N + tid, 256, nr);
        __syncthreads();
        return;
    }
    float p;
    {
#pragma clang fp contract(off)
        const float v = inj[q] * sg.rs; const float sgm = 1.0f / (1.0f + expf(-v)); p = sgm * 2.0f;
    }
    float ss = 0.0f;
#pragma unroll 10
    for (int i = tid; i < N; i += blockDim.x) {
        float nr;
        {
#pragma clang fp contract(fast)   // as op_hc_post (the dsv4_hc_post kernel contracts x * p + residual)
            nr = b[i] * p + r[i];
        }
        r[i] = nr;
        ss += nr * nr;                // as op_rms_mul
    }
    ss = warp_reduce_sum<32>(ss); if (lane == 0) part[wave] = ss; __syncthreads();
    float tot = 0.0f; for (int q = 0; q < 8; q++) tot += part[q];
    const float scale = rsqrtf(tot / N + __int_as_float(sg.pad2));
    for (int i = tid; i < N; i += blockDim.x) {
        const float v = (r[i] * scale) * w[i];
        xn[i] = v;
        if (sg.asum) q8_emit((void *)sg.asum, (size_t)q * N + i, v);
    }
    __syncthreads();
}

// OP_XCHG (one task): the two-card all-reduce of a partial vector (tensor parallelism). This card's partial (Seg.a, fp32
//   [N]) goes into the peer's receive slot (Seg.W: 16 header words, word 0 the flag, then N floats; uncached VRAM on
//   the peer, written over PCIe), then that slot's flag gets this exchange's epoch (system-scope release). The peer's
//   partial arrives in this card's slot (Seg.gs, same layout, uncached); after the flag reaches the epoch,
//   out = p0 + p1 with the XTX's (card 0's) partial first on both cards, so both get the same bits. Seg.asum = int32
//   epoch base on this card (written by the host per token), K = exchange index (epoch = base + K + 1, monotonic, never
//   reset), work = this card (0 / 1), pad1 = 1: local only (out = own partial; timing without the peer).
typedef __attribute__((address_space(1))) float4 XG4;   // global (not flat) float4: lets loads pass earlier stores
#define XCHG_R 3   // float4 per thread per round: T = 1 (N = 2560, 640 float4) is one round of <= 3 per thread
// every round issues all of its loads before any store; with flat pointers that may alias, each store waited for its
// load and the next load was issued only after it. Indices are clamped to the last float4, loads and stores alike (no
// branches, which made the compiler keep the loaded float4 in scratch): a lane past the end rewrites the last float4
// with the same bits as the lane that owns it
__device__ __forceinline__ void op_xchg(const Seg & sg, Sync * sy, int limit_us, int seg, uint64_t * tr) {
    __shared__ int ok;
    const int tid = K3TID, B = blockDim.x, N4 = sg.N / 4;
    const XG4 * __restrict__ a = (const XG4 *)sg.a;
    XG4 * __restrict__ out = (XG4 *)sg.y;
    if (sg.pad1) {
        for (int i0 = tid; i0 < N4; i0 += XCHG_R * B) {
            float4 v[XCHG_R];
#pragma unroll
            for (int j = 0; j < XCHG_R; j++) v[j] = a[min(i0 + j * B, N4 - 1)];
#pragma unroll
            for (int j = 0; j < XCHG_R; j++) out[min(i0 + j * B, N4 - 1)] = v[j];
        }
        __syncthreads(); return;
    }
    const uint32_t want = (uint32_t)((const int *)sg.asum)[0] + (uint32_t)sg.K + 1u;
    float * peer = (float *)sg.W; const float * mine = (const float *)sg.gs;
    // pad2 = 1 (pull, the producing card): the payload goes to aux (this card's own uncached slot), only the flag to the
    // peer; pad2 = 2 (pull, the reading card): the peer's partial is read from aux (the peer's uncached slot)
#ifdef K3_MT   // (the multi-token module has no pull: its payload is T x larger and the selects cost it 4 VGPR spills)
    XG4 * __restrict__ px = (XG4 *)(peer + 16);
#else
    XG4 * __restrict__ px = (XG4 *)(sg.pad2 == 1 ? (float *)sg.aux : peer + 16);
#endif
    for (int i0 = tid; i0 < N4; i0 += XCHG_R * B) {
        float4 v[XCHG_R];
#pragma unroll
        for (int j = 0; j < XCHG_R; j++) v[j] = a[min(i0 + j * B, N4 - 1)];
#pragma unroll
        for (int j = 0; j < XCHG_R; j++) px[min(i0 + j * B, N4 - 1)] = v[j];
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");              // system scope: the payload before the flag
    __syncthreads();
    if (tid == 0) {
#ifdef K3_TRACE
        if (tr) tr[5] = __builtin_readsteadycounter();          // push + fence done (every thread, after the barrier)
#endif
        __hip_atomic_store((uint32_t *)peer, want, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        const uint64_t t0 = __builtin_readsteadycounter();
        uint32_t v; int n = 0, good = 1;
        while ((v = __hip_atomic_load((const uint32_t *)mine, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM)) < want) {
            __builtin_amdgcn_s_sleep(1);
            if ((++n & 63) == 0 && wait_expired(sy, seg, v, t0, limit_us)) { good = 0; break; }
        }
        ok = good;
#ifdef K3_TRACE
        if (tr) tr[6] = __builtin_readsteadycounter();          // the peer's flag seen (or the wait gave up)
#endif
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
#ifdef K3_MT
    const XG4 * __restrict__ rx = (const XG4 *)(mine + 16);
#else
    const XG4 * __restrict__ rx = (const XG4 *)(sg.pad2 == 2 ? (const float *)sg.aux : mine + 16);
#endif
    const bool first = sg.work == 0;
    for (int i0 = tid; i0 < N4; i0 += XCHG_R * B) {
        float4 va[XCHG_R], vr[XCHG_R];
#pragma unroll
        for (int j = 0; j < XCHG_R; j++) { const int i = min(i0 + j * B, N4 - 1); vr[j] = rx[i]; va[j] = a[i]; }
        __builtin_amdgcn_sched_barrier(0);   // keep all six loads ahead of the adds (the scheduler interleaved them)
#pragma unroll
        for (int j = 0; j < XCHG_R; j++) {
            const float4 x = va[j], r = vr[j];   // card 0's partial first on both cards (an fp32 add, no contraction)
            out[min(i0 + j * B, N4 - 1)] = first ? make_float4(x.x + r.x, x.y + r.y, x.z + r.z, x.w + r.w)
                                    : make_float4(r.x + x.x, r.y + x.y, r.z + x.z, r.w + x.w);
        }
    }
    __syncthreads();
}

// OP_HC_AVG: the mean over the hc streams of the residual, as llama.cpp's build_hc_mean (the features a DFlash drafter
//   reads at its target layers): out[t, i] = (((r0 + r1) + r2) + r3) * (1 / hc). Seg.a = residual [T, hc, D],
//   Seg.y = out [T, D]; N = T * D, K = hc, pad1 = D; a task = rows_per_task elements.
__device__ __forceinline__ void op_hc_avg(const Seg & sg, int t) {
#pragma clang fp contract(off)
    const int i0 = t * sg.rows_per_task, i1 = min(sg.N, i0 + sg.rows_per_task), D = sg.pad1;
    const float * r = (const float *)sg.a;
    for (int i = i0 + (int)K3TID; i < i1; i += blockDim.x) {
        const int tt = i / D, ii = i - tt * D;
        float acc = r[(size_t)(tt * sg.K) * D + ii];
        for (int h = 1; h < sg.K; h++) acc = acc + r[(size_t)(tt * sg.K + h) * D + ii];
        ((float *)sg.y)[i] = acc * (1.0f / (float)sg.K);
    }
}

#ifdef K3_DRAFT
#include "k3_draft_ops.h"
#endif
#ifdef K3_QSA   // the GSQ-RCO Coder's trunk modules with the QSA ops only (kernels/k3_gguf_gsq_qsa*.hip)
#include "k3_qsa_ops.h"
#endif
#if defined(K3_GSQ) && !defined(K3_MTP_SWIGLU) && !defined(K3_DRAFT)   // the Coder's trunk / QSA modules: split attention
#define K3_ATTN_SPLIT
enum { OP_ATTN_PART = 41, OP_ATTN_COMB = 42 };
#include "k3_attn_split.h"
#endif
// a GG op's format dispatch over its whitelist MASK: FN<type, vdr>(sg, t) for Seg.work. The REAP-308 four keep the switch
// they always had, written out in k3_extra_op (unlisted formats decode as Q8_0): its code object is unchanged (the same
// switch inside a forceinline function, or a generic lambda, changed k3_run's code). Any other mask goes through
// gg_dispatch_<FN><MASK>: formats outside it run nothing (k3.Engine refuses such tapes before any launch), Q8_0 only if listed
constexpr unsigned GG_MASK_REAP = gg_bits(GG_IQ3_XXS, GG_IQ4_NL, GG_Q6_K, GG_Q8_0);
#define GG_IF(FN, E, TY, VD) if constexpr ((MASK >> (E)) & 1u) { if (sg.work == (E)) { FN<TY, VD>(sg, t); return; } }
#define GG_DEF_DISPATCH(FN)                                                                                             \
template <unsigned MASK>                                                                                                \
__device__ __forceinline__ void gg_dispatch_##FN(const Seg & sg, int t) {                                               \
    GG_IF(FN, GG_IQ3_XXS, GGML_TYPE_IQ3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ)                                                         \
    GG_IF(FN, GG_IQ4_NL,  GGML_TYPE_IQ4_NL,  VDR_IQ4_NL_Q8_1_MMVQ)                                                          \
    GG_IF(FN, GG_Q6_K,    GGML_TYPE_Q6_K,    VDR_Q6_K_Q8_1_MMVQ)                                                            \
    GG_IF(FN, GG_Q4_K,    GGML_TYPE_Q4_K,    VDR_Q4_K_Q8_1_MMVQ)                                                            \
    GG_IF(FN, GG_Q5_K,    GGML_TYPE_Q5_K,    VDR_Q5_K_Q8_1_MMVQ)                                                            \
    GG_IF(FN, GG_IQ4_XS,  GGML_TYPE_IQ4_XS,  VDR_IQ4_XS_Q8_1_MMVQ)                                                          \
    GG_IF(FN, GG_IQ2_S,   GGML_TYPE_IQ2_S,   VDR_IQ2_S_Q8_1_MMVQ)                                                           \
    GG_IF(FN, GG_IQ3_S,   GGML_TYPE_IQ3_S,   VDR_IQ3_S_Q8_1_MMVQ)                                                           \
    GG_IF(FN, GG_Q2_0,    GGML_TYPE_Q2_0,    VDR_Q2_0_Q8_1_MMVQ)                                                            \
    GG_IF(FN, GG_Q3_K,    GGML_TYPE_Q3_K,    VDR_Q3_K_Q8_1_MMVQ)                                                            \
    GG_IF(FN, GG_Q8_0,    GGML_TYPE_Q8_0,    VDR_Q8_0_Q8_1_MMVQ)                                                            \
}
GG_DEF_DISPATCH(gg_op_mv_t)
GG_DEF_DISPATCH(gg_moe_gu_t)
GG_DEF_DISPATCH(gg_moe_down_t)
GG_DEF_DISPATCH(gg_gu_dense_t)
#undef GG_DEF_DISPATCH
#undef GG_IF
#define GG_DISPATCH(MASK, FN, sg, t)                                                                                    \
    if constexpr ((MASK) == GG_MASK_REAP) {                                                                             \
        switch ((sg).work) {                                                                                            \
            case GG_IQ3_XXS: FN<GGML_TYPE_IQ3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ>(sg, t); break;                                \
            case GG_IQ4_NL:  FN<GGML_TYPE_IQ4_NL,  VDR_IQ4_NL_Q8_1_MMVQ>(sg, t); break;                                 \
            case GG_Q6_K:    FN<GGML_TYPE_Q6_K,    VDR_Q6_K_Q8_1_MMVQ>(sg, t); break;                                   \
            default:         FN<GGML_TYPE_Q8_0,    VDR_Q8_0_Q8_1_MMVQ>(sg, t); break;                                   \
        }                                                                                                               \
    } else gg_dispatch_##FN<MASK>(sg, t)

__device__ __forceinline__ bool k3_extra_op(const Seg & sg, int t, bool fresh, Sync * sy, int limit_us, int seg,
                                            uint64_t * tr = nullptr) {
    if (sg.op == OP_XCHG) { op_xchg(sg, sy, limit_us, seg, tr); return true; }
#ifdef K3_DRAFT
    if (k3_draft_flagged(sg, t)) return true;
#endif
#ifndef K3_DRAFT   // (the drafter's module: none of the trunk-only ops below are compiled in; python/caminus/k3.py DRAFT_OPS
                   // refuses tapes with them. A smaller k3_run: its matvecs ran up to 20 % slower with them inlined)
    if (fresh && sg.out_f32 >= GG_ACT_F32 && (sg.op == OP_GG_MV || sg.op == OP_GG_MOE_GU || sg.op == OP_GG_MOE_DOWN))
        aq_fill((const float *)sg.a, sg.op == OP_GG_MOE_DOWN ? sg.pad1 * sg.K : sg.K, sg.out_f32, sg.rs);
    if (sg.op == OP_HC_POST_RMS) { op_hc_post_rms(sg, t); return true; }
    if (sg.op == OP_GDN_SAVE) {
        // a bailed launch (sy->err) must leave the saved-input records alone: its gdn_step tasks still run (every task runs
        // exactly once, waits or not) and replay the previous pass's records into the state, and this task's dependency
        // wait may have given up before they read them. err == 0 here means every wait so far succeeded (a wait sets err
        // before it gives up), so the records are complete; the host then retries the pass with commit 0
        // (FlashNextTP.verify). Thread 0 decides for the workgroup, so a record is written whole or not at all
        __shared__ int save_ok;
        if (thread0()) save_ok = __hip_atomic_load(&sy->err, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == 0;
        __syncthreads();
        const bool ok = save_ok;
        __syncthreads();
        if (ok) op_gdn_save(sg, t);
        return true;
    }
    if (sg.op == OP_HC_AVG) { op_hc_avg(sg, t); return true; }
    if (sg.op == OP_HADAMARD) { op_hadamard(sg, t); return true; }
    if (sg.op == OP_PLE_GATE) { op_ple_gate(sg, t); return true; }
    if (sg.op == OP_PLE_CONV) { op_ple_conv(sg, t); return true; }
    if (sg.op == OP_ADD_SUM) { op_add_sum(sg, t); return true; }
    if (sg.op == OP_QK_NR) { op_qk_nr(sg, t); return true; }
    if (sg.op == OP_KV_APPEND) { op_kv_append(sg, t); return true; }
#if defined(K3_ATTN_SPLIT)   // (kernels/k3_attn_split.h: the Coder's trunk / QSA modules run their attention only as the
                            // flash-decoding pair, one body for the dense and QSA tapes; no OP_ATTN_DEC / OP_ATTN_SEL here:
                            // caminus.k3.Engine.attn_dec / attn_sel emit the pair on these modules, build() refuses the others)
    if (sg.op == OP_ATTN_PART) { op_attn_part(sg, t); return true; }
    if (sg.op == OP_ATTN_COMB) { op_attn_comb(sg, t); return true; }
#elif defined(K3_QSA)   // (the QSA modules: one attention body, op_attn_sel; Seg.gs = nullptr for OP_ATTN_DEC: the identity list)
    if (sg.op == OP_ATTN_DEC || sg.op == OP_ATTN_SEL) { op_attn_sel(sg, t); return true; }
#else
    if (sg.op == OP_ATTN_DEC) { op_attn_dec(sg, t); return true; }
#endif
    if (sg.op == OP_GATE_MUL) { op_gate_mul(sg, t); return true; }
    if (sg.op == OP_GDN_CONV) { op_gdn_conv(sg, t); return true; }
    if (sg.op == OP_GDN_L2) { op_gdn_l2(sg, t); return true; }
    if (sg.op == OP_GDN_GATES) { op_gdn_gates(sg); return true; }
    if (sg.op == OP_GDN_STEP) { op_gdn_step(sg, t); return true; }
    if (sg.op == OP_GDN_NORM) { op_gdn_norm(sg, t); return true; }
    if (sg.op == OP_RMS_MUL) { op_rms_mul(sg, t); return true; }
    if (sg.op == OP_HC_MEAN) { op_hc_mean(sg, t); return true; }
    if (sg.op == OP_HC_POST) { op_hc_post(sg, t); return true; }
    if (sg.op == OP_F32_MV) { op_f32_mv(sg, t); return true; }
#ifdef K3_GSQ
    if (sg.op == OP_BF16_MV) {
#ifdef K3_MTP_SWIGLU
        if (fresh && sg.work == 2) bx_fill_swiglu((const float *)sg.a, (const float *)sg.gs, seg_T(sg) * sg.K);
        else
#endif
        if (fresh && sg.work) bx_fill((const float *)sg.a, seg_T(sg) * sg.K, sg.rs);
        op_bf16_mv(sg, t); return true;
    }
#endif
    if (sg.op == OP_ROUTER) { op_router(sg, t); return true; }
    if (sg.op == OP_SHEXP_ADD) { op_shexp_add(sg, t); return true; }
    if (sg.op == OP_GG_MOE_GU) {
#if K3_MOE_N & 1
        static_assert((K3_MASK_MOE_GU >> GG_IQ3_XXS) & 1u, "K3_MOE_N & 1: the unrolled IQ3_XXS gate/up needs IQ3_XXS in K3_FMTS_MOE_GU");
        if (sg.work == GG_IQ3_XXS && sg.K == 2560 && seg_T(sg) == 1) { gg_moe_gu_n<GGML_TYPE_IQ3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ, 10>(sg, t); return true; }
#endif
        GG_DISPATCH(K3_MASK_MOE_GU, gg_moe_gu_t, sg, t);
        return true;
    }
    if (sg.op == OP_GG_MOE_DOWN) {   // the TP split's down shapes: k = 10 experts, K = 384 (card 0) / 256 (card 1) IQ4_NL
#if defined(K3_GSQ) && K3_MOE_P
        {   // the Coder's TP down shapes, lane-packed (T = 1, and T > 1 with tasks of one token's rows)
            const bool tok = seg_T(sg) == 1 || (K3_MOE_NT && (sg.N / seg_T(sg)) % sg.rows_per_task == 0);
#ifdef K3_MT
#define K3_MOE_P_MT true
#else
#define K3_MOE_P_MT false
#endif
            if constexpr (((K3_MOE_P & 1) != 0) && ((K3_MASK_MOE_DOWN >> GG_Q2_0) & 1u)) {
                if (sg.work == GG_Q2_0 && sg.pad1 == 10 && tok) {
                    if (sg.K == 384) { gg_moe_down_p<GGML_TYPE_Q2_0, VDR_Q2_0_Q8_1_MMVQ, 6, 10, K3_MOE_P_MT, K3_MOE_P_RB6>(sg, t); return true; }
                    if (sg.K == 256) { gg_moe_down_p<GGML_TYPE_Q2_0, VDR_Q2_0_Q8_1_MMVQ, 4, 10, K3_MOE_P_MT, K3_MOE_P_RB4>(sg, t); return true; }
                }
            }
            if constexpr (((K3_MOE_P & 2) != 0) && ((K3_MASK_MOE_DOWN >> GG_IQ4_NL) & 1u)) {
                if (sg.work == GG_IQ4_NL && sg.pad1 == 10 && sg.K == 256 && tok) {
                    gg_moe_down_p<GGML_TYPE_IQ4_NL, VDR_IQ4_NL_Q8_1_MMVQ, 8, 10, K3_MOE_P_MT, K3_MOE_P_RBN>(sg, t); return true;
                }
            }
#undef K3_MOE_P_MT
        }
#endif
#if K3_MOE_N & 2
        static_assert((K3_MASK_MOE_DOWN >> GG_IQ4_NL) & 1u, "K3_MOE_N & 2: the unrolled IQ4_NL down needs IQ4_NL in K3_FMTS_MOE_DOWN");
        if (sg.work == GG_IQ4_NL && sg.K == 384 && sg.pad1 == 10 && seg_T(sg) == 1) { gg_moe_down_n<GGML_TYPE_IQ4_NL, VDR_IQ4_NL_Q8_1_MMVQ, 12, 10>(sg, t); return true; }
        if (sg.work == GG_IQ4_NL && sg.K == 256 && sg.pad1 == 10 && seg_T(sg) == 1) { gg_moe_down_n<GGML_TYPE_IQ4_NL, VDR_IQ4_NL_Q8_1_MMVQ, 8, 10>(sg, t); return true; }
#if K3_MOE_NT
        if (sg.work == GG_IQ4_NL && sg.pad1 == 10 && (sg.N / seg_T(sg)) % sg.rows_per_task == 0) {
            if (sg.K == 384) { gg_moe_down_n<GGML_TYPE_IQ4_NL, VDR_IQ4_NL_Q8_1_MMVQ, 12, 10, true>(sg, t); return true; }
            if (sg.K == 256) { gg_moe_down_n<GGML_TYPE_IQ4_NL, VDR_IQ4_NL_Q8_1_MMVQ, 8, 10, true>(sg, t); return true; }
        }
#endif
#endif
        GG_DISPATCH(K3_MASK_MOE_DOWN, gg_moe_down_t, sg, t);
        return true;
    }
    if (sg.op == OP_GG_Q8_1) { gg_op_q8_1(sg, t); return true; }
    if (sg.op == OP_GG_GU_DENSE) { GG_DISPATCH(K3_MASK_GU_DENSE, gg_gu_dense_t, sg, t); return true; }
#else
    if (sg.op == OP_GG_Q8_1) { gg_op_q8_1(sg, t); return true; }
#endif
#ifdef K3_DRAFT
    if (sg.op != OP_GG_MV) return k3_draft_only(sg, t);
#elif defined(K3_QSA)
    if (sg.op != OP_GG_MV) {   // (kernels/k3_gguf_gsq_qsa*.hip only)
        if (sg.op == OP_IDX_APPEND) { op_idx_append(sg, t); return true; }
        if (sg.op == OP_QSA_SCORE) { op_qsa_score(sg, t); return true; }
        if (sg.op == OP_QSA_SELECT) { op_qsa_select(sg, t); return true; }
        return false;
    }
#else
    if (sg.op != OP_GG_MV) return false;
#endif
    GG_DISPATCH(K3_MASK_MV, gg_op_mv_t, sg, t);
    return true;
}

// touch(): pull [p, p + n) toward L2: one dword per 128-byte line, one line per thread per round, the value sunk into an
// empty asm (no arithmetic, no store). Global loads (not flat), no scratch; a <= 32 KB range is one round, one wait
__device__ __forceinline__ void touch(const uint8_t * p, uint32_t n) {   // p wave-uniform, n < 4 GB
    for (uint32_t o = K3TID * 128u; o < n; o += blockDim.x * 128u) {
        const uint32_t v = *(const __attribute__((address_space(1))) uint32_t *)(p + o);
        asm volatile("" :: "v"(v));
    }
}

// before the dependency wait: touch the start of the task's weight bytes, so the vec_dot loads after the wait partly hit
// L2. Seg.aux = a per-segment byte cap + 1, set by the host (0: the whole task, 1: off). flashnext_tp caps every matvec
// at 16 KB (K3_PF_CAP): fn_bench 96 --tp 10.74 -> 10.40 ms/token. The whole task (the old serial loop: one load + wait
// per 32 KB round, so 3 rounds for an hc down task) is 0.3 ms slower; 8 loads in flight per thread (no cap) slower still
// (10.90); no prefetch 10.84; a cap of 8 / 4 / 1 KB 10.59 / 10.73 / 10.86, 12-24 KB flat; one round spread over the rows'
// starts 10.70. Per segment: hc down or attn_qkv uncapped +0.17 / +0.07 ms; attn_gate / ple off +0.08; the head flat
#ifndef K3_XWARM
#define K3_XWARM 1
#endif
// the dependency wait's warm word (k3.hip wait_for): OP_XCHG's peer slot header word 1. The exchange is claimed ~20-50 us
// before its producer finishes (past the ~15 us window), so the wait stores every ~8 us until the push. XT push 6.8 ->
// 2.5 us, exchange 8.9 -> 4.5 us; fn_bench 96 --tp 10.45 -> 9.92 ms/token (a 2 / 4 us period: the same)
__device__ __forceinline__ uint32_t * k3_extra_warm(const Seg & sg) {
#if K3_XWARM
    if (sg.op == OP_XCHG && !sg.pad1) return (uint32_t *)sg.W + 1;
#endif
    return nullptr;
}
__device__ __forceinline__ void k3_extra_prefetch(const Seg & sg, int t) {
#if K3_XWARM
    // OP_XCHG, claimed before the producer finishes: one 4-byte store into the peer's receive slot (header word 1,
    // read by nobody) warms the path to the peer. The first remote store after ~15-20 us of no traffic to the peer
    // takes ~4.4 us longer to complete, so the push's release fence waited for it (XT push 6.6 us); a store within the
    // last ~15 us makes the push ~1.2 us (s14/xb.py)
    if (sg.op == OP_XCHG) {
        if (!sg.pad1 && thread0()) __hip_atomic_store((uint32_t *)sg.W + 1, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        return;
    }
#endif
#ifdef K3_GSQ
    if (sg.op != OP_GG_MV && sg.op != OP_BF16_MV && !(sg.op == OP_F32_MV && seg_T(sg) == 1 && sg.pad2 > 0)) return;   // BF16_MV: N = weight rows
#else
    if (sg.op != OP_GG_MV && !(sg.op == OP_F32_MV && seg_T(sg) == 1 && sg.pad2 > 0)) return;   // F32_MV: one token's rows = weight rows
#endif
    const size_t row_bytes = (size_t)sg.pad2;                    // bytes per row, set by the host
    const int r0 = t * sg.rows_per_task, r1 = min(sg.N, r0 + sg.rows_per_task);
    size_t n = (size_t)(r1 - r0) * row_bytes;
    const size_t cap = (size_t)sg.aux;
    if (cap) n = min(n, cap - 1);
    touch(sg.W + (size_t)r0 * row_bytes, (uint32_t)n);
}
