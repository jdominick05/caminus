// ---- PA_NAME: a prefill chunk's causal attention as WMMA tiles, up to 8 query heads per kv head (every one of WMMA's 16
// columns used at 8), head dim PA_D 256 (Qwen3.6-35B-A3B: kernels/q35m_prefill.hip q35mp_attn) or 128 (Qwen3-30B-A3B:
// kernels/q3m_prefill.hip q3mp_attn), fp16 q / k / v / p, fp32 scores, softmax state and sums. One source, one kernel per
// object: the includer defines PA_NAME, PA_D, PA_QT (q's type: float, rounded to fp16 here, or _Float16) and PA_OT (out's
// type: float or _Float16, rounded once from the fp32 sums); Q3P_KV_Q8 (the Q8 KV objects) reads Q8_0 rows. Needs
// q3m_prefill.hip's types and wmma / h16of / as_h2 / as_u, and own_kv_codec.h under Q3P_KV_Q8.
// q35_prefill.hip's q35p_attn_wm (the dense line's: kept there) reworked after rocKE's gfx11 prefill pipeline
// (ROCm/rocm-libraries, dnn-providers/hip-kernel-provider/rocke, library/kernels/gfx1151/wmma_fmha_fwd_v2.py and
// platform/python/rocke/helpers/rdna_attention.py at 9e91949; MIT License, Copyright (c) Advanced Micro Devices, Inc.:
// THIRD_PARTY.md), written here from that design; no rocKE code. A 2048-row chunk at the 35B's geometry (XT): 1.49 -> 0.80 ms
// at P0 0, 41.3 -> 20.95 ms at P0 30K (25.8 -> 50.8 TF; rocKE 19.6 ms), Q8 43.0 -> 22.1 ms. What follows rocKE:
//   - the next tile's K / V are loaded into registers at the top of a tile and stored to LDS behind it (K after the
//     scores' barrier, V after the sums'): two barriers per tile, the global latency behind the tile's WMMAs.
//     q35p_attn_wm loaded each tile, then waited, behind three barriers: 2.85 ms -> rocKE 0.94 ms at P0 0 (dense)
//   - V^T staged by 4-position units: a thread loads 4 rows x 8 dims and writes 8 runs of 4 positions (ds_write_b64, two
//     v_perm each) instead of 32 scalar half writes
//   - head dim 256: the pair's score partials: each wave publishes its own and reads its partner's (own + partner: the same
//     sum in both waves, fp addition is commutative), as float4s, instead of both waves reading both
//   - lazy rescale: a column keeps its running max while a tile's max exceeds it by at most 8 (log2 units: p <= 256,
//     exact in fp16's range); the wave skips O's rescale when every alpha is exactly 1 (a skipped multiply by 1 changes
//     no bit). Per column, unlike rocKE's wave vote, so a column's bits never depend on the other columns of its wave
//   - the interior tiles (every position before the workgroup's first) run without masks; heaviest blocks first
// What stays Caminus's (q35p_attn_wm's argument, unchanged): work by ABSOLUTE position, so chunk geometry changes no bit.
// A query pair (positions q0 even, q1 = q0 + 1) x the G query heads of kv head blockIdx.y is 16 columns, column n = lane & 15
// -> position q0 + (n >> 3), head G g + (n & 7) (n & 7 >= G: head G g + G - 1 again, never written). Its Q and O take 64
// VGPRs each at 128 dims: head dim 256 splits them over a wave pair (waves 2 w, 2 w + 1, each half of the dims; workgroup
// i = positions 8 i .. 8 i + 7, pair w = q0 8 i + 2 w), head dim 128 gives each wave its own pair (workgroup i = positions
// 16 i .. 16 i + 15, wave w = q0 16 i + 2 w). Tiles of MA_C positions from 0 (MA_C x MA_D = 8,192 values: every thread
// stages 4 K and 4 V chunks of 8 values at either head dim).
// RDNA3's WMMA is not exact for a zero product (kernels/q3m_prefill.hip q3p_attn), so in the pair's diagonal tile V^T is
// zeroed past q0 and q1's own row added with fp32 fmas; tiles past the pair are skipped. Only the last tile holds the
// workgroup's own positions; rows past the chunk's end are clamped to its last row (never in a written column's sum).
// Q8 rows (Q3P_KV_Q8; the kernel is rotation-blind: the host rotates q and the rows alike): each staged value
// fp16(float code * float scale), q3p_q8_stage's arithmetic. A staging chunk (8 dims of a position) loads the dword holding
// its block's scale and the 3 dwords holding its 8 codes; every byte read lies inside its position's row (an even block is
// never a row's last, an odd one never its first: a row is 2 nkv MA_D / 32 blocks, a kv head's part MA_D / 32, both even).
// q: the chunk's [n][heads][MA_D] (PA_QT), out the same shape (PA_OT). scale_log2 = scale * log2(e).
// Grid (MA_B-position blocks blk0 .. blk0 + gridDim.x - 1, nkv), block 256. LDS K [MA_C][MA_KP], V^T [MA_D][MA_VP], at head
// dim 256 the partials [8][4][32] float4 (53,760 B: two workgroups per WGP).
#ifndef PREFILL_ATTN_G8_H
#define PREFILL_ATTN_G8_H
#define MA_D PA_D
#define MA_WP (MA_D / 128)                  // waves per query pair
#define MA_B (16 / MA_WP)                   // positions per workgroup
#define MA_C (8192 / MA_D)                  // positions per tile
#define MA_KP (MA_D + 8)
#define MA_VP (MA_C + 8)
#define MA_NS (MA_C / 16)                   // 16-position subtiles per tile
#define MA_KL (MA_D == 256 ? 5 : 4)         // log2 of the 8-dim chunks per row
#define MA_VL (MA_C == 32 ? 3 : 4)          // log2 of the 4-position units per tile
#ifdef Q3P_KV_Q8
#define PA_CACHE uint8_t
#else
#define PA_CACHE _Float16
#endif
template <bool B> struct mp_bool { static constexpr bool v = B; };
// a barrier that orders LDS only: __syncthreads' workgroup fence also waits for every global load in flight (vmcnt(0) in
// WGP mode), which would drain the next tile's prefetch at each barrier; the tiles' global data is read-only here
__device__ __forceinline__ void mp_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
// staging chunk c (0 .. MA_D / 8 - 1: dims 8 c .. 8 c + 7) of kv head g's K (part 0) or V (part 1) in position row prow (>= 0), raw
__device__ __forceinline__ u32x4_t mp_load(const PA_CACHE * __restrict__ cache, int prow, int M, int part, int g, int c) {
#ifdef Q3P_KV_Q8
    // block (part M / 32 + g MA_D / 32 + c / 4) of the row; od = 2 for an odd block (rows and kv heads are whole dwords)
    const uint8_t * blk = cache + (uint64_t)(uint32_t)prow * (uint32_t)own_kv_q8_row_bytes(M) + (uint32_t)(part * (M / 32) + g * (MA_D / 32) + (c >> 2)) * 34u;
    const int od = (c & 4) >> 1;
    const uint32_t * s = (const uint32_t *)(blk - od);                 // the scale: the low (od 0) or high half
    const uint32_t * w = (const uint32_t *)(blk + 8 * (c & 3) - od);   // codes from byte 2 (od 0) or 4 (od 2) of w[0]
    return (u32x4_t){s[0], w[0], w[1], w[2]};
#else
    return *(const u32x4_t *)((const uint8_t *)cache + (uint64_t)(uint32_t)prow * (uint32_t)(4 * M) + (uint32_t)(2 * (part * M + g * MA_D + c * 8)));
#endif
}
// the chunk's 8 values as fp16 pairs (dims 2 j, 2 j + 1 in dword j). Q8: fp16(code * d), q3p_q8_stage's value (code and d
// exact in fp16, their product exact in fp32, one rounding to fp16) by packed fp16 math: two codes' bytes into halves 0 and 1
// biased to fp16 1024 + (code ^ 0x80) = 1152 + code (exact), less 1152 (exact), times (d, d) (v_pk_mul: the exact product
// rounded once, the same bits): 44.8 -> 47.6 TF at P0 30K (XT, head dim 256) against the fp32 conversions
__device__ __forceinline__ u32x4_t mp_h8(u32x4_t r, int c) {
#ifdef Q3P_KV_Q8
    const bool od = (c & 4) != 0;
    const uint32_t d2 = __builtin_amdgcn_perm(r.x, r.x, od ? 0x03020302u : 0x01000100u);   // (d, d)
    const uint32_t w0 = od ? r.z : __builtin_amdgcn_alignbyte(r.z, r.y, 2), w1 = od ? r.w : __builtin_amdgcn_alignbyte(r.w, r.z, 2);
    const h2_t b2 = {(_Float16)1152.0f, (_Float16)1152.0f};
    uint32_t o[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const uint32_t sel = (j & 1) ? 0x0c030c02u : 0x0c010c00u;   // (bytes 0, 1 or 2, 3 of w into halves 0, 1)
        o[j] = as_u((as_h2(__builtin_amdgcn_perm(0u, j < 2 ? w0 : w1, sel) ^ 0x64806480u) - b2) * as_h2(d2));
    }
    return (u32x4_t){o[0], o[1], o[2], o[3]};
#else
    (void)c;
    return r;
#endif
}
extern "C" __global__ void __launch_bounds__(256) PA_NAME(const PA_QT * __restrict__ qin, const PA_CACHE * __restrict__ cache,
                                                         int pos0, int n, int blk0, int heads, int nkv, float scale_log2,
                                                         PA_OT * __restrict__ out) {
    __shared__ __attribute__((aligned(16))) _Float16 Ks[MA_C * MA_KP];
    __shared__ __attribute__((aligned(16))) _Float16 Vt[MA_D * MA_VP];
    __shared__ __attribute__((aligned(16))) float Sx[MA_WP == 2 ? 8 : 1][4][32][4];   // [wave][2 st + h][lane]: its S[st] slots 4 h .. 4 h + 3
    // (wave through readfirstlane: q0, the pair's subtile and every branch on them are scalar, not exec-masked)
    const int tid = threadIdx.x, lane = tid & 31, wave = __builtin_amdgcn_readfirstlane(tid >> 5), hi = lane >> 4, dh = MA_WP == 2 ? (wave & 1) * 128 : 0;
    const int G = heads / nkv, g = blockIdx.y, M = nkv * MA_D;
    const int b8 = (blk0 + (int)(gridDim.x - 1 - blockIdx.x)) * MA_B;   // (the deepest blocks first)
    const int nq = lane & 15, sec = nq >> 3, hh = nq & 7;
    const int q0 = b8 + 2 * (wave >> (MA_WP - 1)), q1 = q0 + 1;
    const int qpos = q0 + sec, tq = qpos - pos0, hq = g * G + min(hh, G - 1);
    const bool qin_ok = tq >= 0 && tq < n, qok = qin_ok && hh < G;
    h16_t Qb[8];                                        // this wave's 128 dims of the column's q, 16 dims per k-step
    if constexpr (sizeof(PA_QT) == 4) {
        const float4 * qp = (const float4 *)(qin + ((size_t)(qin_ok ? tq : 0) * heads + hq) * MA_D + dh);
#pragma unroll
        for (int s = 0; s < 8; s++) {
            h16_t v;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float4 f = qp[4 * s + j];
                v[4 * j] = (_Float16)f.x; v[4 * j + 1] = (_Float16)f.y; v[4 * j + 2] = (_Float16)f.z; v[4 * j + 3] = (_Float16)f.w;
            }
            Qb[s] = v;
            if (s == 3) __builtin_amdgcn_sched_barrier(0);   // (half of q's loads in flight at a time: all 32 spilled)
        }
    } else {
        const u32x4_t * qp = (const u32x4_t *)(qin + ((size_t)(qin_ok ? tq : 0) * heads + hq) * MA_D + dh);
#pragma unroll
        for (int s = 0; s < 8; s++) Qb[s] = h16of(qp[2 * s], qp[2 * s + 1]);
    }
    f8_t O[8];
#pragma unroll
    for (int d = 0; d < 8; d++) O[d] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.0f;
    // rows 0 .. npos - 1 exist for this workgroup (its last position's); tiles below clast are interior (every position
    // before b8 <= q0 of every pair), clast's tile holds the workgroup's own positions (npos > b8 >= clast)
    const int npos = min(pos0 + n, b8 + MA_B), clast = b8 & ~(MA_C - 1);
    // staging: K chunk kc of rows kr + (256 / chunks per row) i (head dim 256: a wave = one row's 512 bytes); V chunk vc of
    // rows 4 vg .. 4 vg + 3
    const int kr = tid >> MA_KL, kc = tid & ((1 << MA_KL) - 1), vg = tid & ((1 << MA_VL) - 1), vc = tid >> MA_VL;
    u32x4_t rk[4], rv[4];
    auto load_k = [&](int c0) {
#pragma unroll
        for (int i = 0; i < 4; i++) rk[i] = mp_load(cache, min(c0 + kr + (256 >> MA_KL) * i, npos - 1), M, 0, g, kc);
    };
    auto load_v = [&](int c0) {
#pragma unroll
        for (int r = 0; r < 4; r++) rv[r] = mp_load(cache, min(c0 + 4 * vg + r, npos - 1), M, 1, g, vc);
    };
    auto store_k = [&]() {
#pragma unroll
        for (int i = 0; i < 4; i++) *(u32x4_t *)(Ks + (kr + (256 >> MA_KL) * i) * MA_KP + kc * 8) = mp_h8(rk[i], kc);
    };
    auto store_v = [&]() {
        u32x4_t h[4];
#pragma unroll
        for (int r = 0; r < 4; r++) h[r] = mp_h8(rv[r], vc);
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int od = 0; od < 2; od++) {             // dim 8 vc + 2 j + od: its half of dword j of the 4 rows
                const uint32_t sel = od ? 0x07060302u : 0x05040100u;
                *(uint2 *)(Vt + (vc * 8 + 2 * j + od) * MA_VP + 4 * vg) =
                    (uint2){__builtin_amdgcn_perm(h[1][j], h[0][j], sel), __builtin_amdgcn_perm(h[3][j], h[2][j], sel)};
            }
    };
    auto tile = [&](int c0, auto lastc) {
        constexpr bool LAST = decltype(lastc)::v;
        // the pair's subtile: positions c0 + 16 std .. + 15 hold q0 and q1 (q0 even: never across a subtile edge); later
        // subtiles are all masked: skipped (interior tiles: all, no masks)
        const int std_ = LAST ? min(MA_NS - 1, (q0 - c0) >> 4) : MA_NS - 1, kq0 = q0 - c0 - 16 * std_;
        f8_t S[MA_NS];
#pragma unroll
        for (int st = 0; st < MA_NS; st++) S[st] = (f8_t){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int s = 0; s < 8; s++)                     // (k-step outer: the subtiles' chains interleave)
#pragma unroll
            for (int st = 0; st < MA_NS; st++) {
                if (LAST && st > std_) continue;
                const u32x4_t * ap = (const u32x4_t *)(Ks + (st * 16 + (lane & 15)) * MA_KP + dh + 16 * s);
                S[st] = wmma(h16of(ap[0], ap[1]), Qb[s], S[st]);
                if (st == MA_NS - 1 && (s & 1)) __builtin_amdgcn_sched_barrier(0);   // (K fragments read 4 ahead, not all 16)
            }
        if constexpr (MA_WP == 2) {
#pragma unroll
            for (int st = 0; st < MA_NS; st++)
#pragma unroll
                for (int h = 0; h < 2; h++)
                    *(float4 *)Sx[wave][2 * st + h][lane] = (float4){S[st][4 * h], S[st][4 * h + 1], S[st][4 * h + 2], S[st][4 * h + 3]};
        }
        // (the partials visible; every wave done reading K; and the previous tile's V^T stores, made after its last
        // barrier, visible before this tile's V^T reads: in the last tile too, at either head dim)
        mp_lds_barrier();
        if (!LAST) {                                    // the next tile's K to LDS, its V loads issued behind the sums
            store_k();
            load_v(c0 + MA_C);
            __builtin_amdgcn_sched_barrier(0);
        }
        float cm = -INFINITY;
#pragma unroll
        for (int st = 0; st < MA_NS; st++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                float pv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                if constexpr (MA_WP == 2) {
                    const float4 t = *(const float4 *)Sx[wave ^ 1][2 * st + h][lane];
                    pv[0] = t.x; pv[1] = t.y; pv[2] = t.z; pv[3] = t.w;
                }
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int v = 4 * h + e;
                    float x = MA_WP == 2 ? S[st][v] + pv[e] : S[st][v];
                    if (LAST && !(st <= std_ && c0 + st * 16 + 2 * v + hi <= qpos)) x = -INFINITY;
                    S[st][v] = x; cm = fmaxf(cm, x);
                }
            }
        // (finite: every tile holds a position the column sees; scaled after the max: the same value, scale_log2 > 0)
        cm = fmaxf(cm, __shfl_xor(cm, 16, 32)) * scale_log2;
        const float mn = cm - m <= 8.0f ? m : fmaxf(m, cm);
        const float alpha = __builtin_amdgcn_exp2f(m - mn);   // (0 on the first tile, exactly 1 when the max is kept)
        float rsum = 0.0f;
        uint32_t pw[MA_NS][4];                          // fp16 p, two positions per dword: [st][v pair]
#pragma unroll
        for (int st = 0; st < MA_NS; st++)
#pragma unroll
            for (int v = 0; v < 8; v += 2) {
                const _Float16 p0 = (_Float16)__builtin_amdgcn_exp2f(__builtin_fmaf(S[st][v], scale_log2, -mn)),
                               p1 = (_Float16)__builtin_amdgcn_exp2f(__builtin_fmaf(S[st][v + 1], scale_log2, -mn));
                rsum += (float)p0 + (float)p1;
                pw[st][v >> 1] = as_u((h2_t){p0, p1});
            }
        rsum += __shfl_xor(rsum, 16, 32);
        l = l * alpha + rsum; m = mn;
        if (__builtin_amdgcn_ballot_w32(alpha != 1.0f)) {
#pragma unroll
            for (int d = 0; d < 8; d++)
#pragma unroll
                for (int v = 0; v < 8; v++) O[d][v] *= alpha;
        }
#pragma unroll
        for (int st = 0; st < MA_NS; st++) {
            if (LAST && st > std_) continue;
            // P^T operand: positions 16 st .. 16 st + 15 of the lane's column: own (2 v + hi) and lane ^ 16's (2 v + 1 - hi)
            uint32_t o[8];
#pragma unroll
            for (int w = 0; w < 4; w++) {
                const uint32_t mine = pw[st][w], other = (uint32_t)__shfl_xor((int)mine, 16, 32);
                const uint32_t ev = hi ? other : mine, odd = hi ? mine : other;   // even positions, odd positions
                o[2 * w] = __builtin_amdgcn_perm(odd, ev, 0x05040100u);       // positions 4w, 4w + 1
                o[2 * w + 1] = __builtin_amdgcn_perm(odd, ev, 0x07060302u);   // positions 4w + 2, 4w + 3
            }
            const bool diag = LAST && st == std_ && kq0 < 16;
            float p_q1 = 0.0f;
            if (diag) {                                 // q1's own entry (index kq0 + 1: dword kq0 / 2, high half) out of B
                const int wq = kq0 >> 1;
#pragma unroll
                for (int w = 0; w < 8; w++)
                    if (w == wq) { p_q1 = (float)as_h2(o[w]).y; if (sec) o[w] &= 0x0000ffffu; }
            }
            const h16_t PB = __builtin_bit_cast(h16_t, o);
#pragma unroll
            for (int d = 0; d < 8; d++) {
                const u32x4_t * ap = (const u32x4_t *)(Vt + (dh + d * 16 + (lane & 15)) * MA_VP + st * 16);
                u32x4_t a0 = ap[0], a1 = ap[1];
                if (diag) {                             // V^T past q0 zeroed (kq0 even: dword kq0 / 2 keeps its low half)
                    uint32_t aw[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                    const int wq = kq0 >> 1;
#pragma unroll
                    for (int w = 0; w < 8; w++) aw[w] = w < wq ? aw[w] : w == wq ? (aw[w] & 0x0000ffffu) : 0u;
                    a0 = (u32x4_t){aw[0], aw[1], aw[2], aw[3]}; a1 = (u32x4_t){aw[4], aw[5], aw[6], aw[7]};
                }
                O[d] = wmma(h16of(a0, a1), PB, O[d]);
            }
            if (diag && sec) {                          // q1's rows: + p(q1) v(q1), fp32 fmas over the lane's dims
                const int kc1 = 16 * st + kq0 + 1;      // q1's row in the tile
#pragma unroll
                for (int d = 0; d < 8; d++)
#pragma unroll
                    for (int v = 0; v < 8; v++)
                        O[d][v] = __builtin_fmaf(p_q1, (float)Vt[(dh + d * 16 + 2 * v + hi) * MA_VP + kc1], O[d][v]);
            }
        }
        if (!LAST) {
            mp_lds_barrier();                            // (every wave done reading V^T and the partials)
            store_v();
        }
    };
    load_k(0); load_v(0); store_k(); store_v();
    mp_lds_barrier();
    for (int c0 = 0; c0 < clast; c0 += MA_C) {
        load_k(c0 + MA_C);                              // (the next tile's K behind this one's score WMMAs; at most clast's)
        __builtin_amdgcn_sched_barrier(0);              // (issued here: the scheduler sank them past the WMMAs)
        tile(c0, mp_bool<false>{});
    }
    tile(clast, mp_bool<true>{});
    if (!qok) return;
    const float il = 1.0f / l;
    PA_OT * op = out + ((size_t)tq * heads + hq) * MA_D + dh;
#pragma unroll
    for (int d = 0; d < 8; d++)
#pragma unroll
        for (int v = 0; v < 8; v++) op[d * 16 + 2 * v + hi] = (PA_OT)(O[d][v] * il);
}
#endif
