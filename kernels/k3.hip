// K3 decode engine, M1 skeleton (docs/K3-megakernel-design.md): one persistent kernel runs a TAPE of segments built on
// the host from a model description. Each segment = an op, a number of tasks and (optionally) the segment it depends on.
// Every workgroup walks the segments in order: it waits until the dependency's tasks are all done, then claims tasks of
// the segment until none are left, publishing each finished task. A workgroup only ever waits on segments whose tasks
// were claimed by workgroups that are running (it drained its own claims first), so ONE resident workgroup suffices for
// progress: the grid need not be co-resident (the XT drives a display; the RDP compositor shares the XTX).
// Sync (ignite-llama megakernel.cu:149-211, measured in tests/bench_sync*.py): per-segment claim / done counters, every
// wave releases at agent scope before one release add, waits poll with acquire; the last workgroup out resets the
// counters for the next launch. A wait that exceeds `limit_us` sets the status word (sticky) with the segment id and
// the count it saw, and gives up, and so does every later wait: the host checks status after every launch, so a hang
// becomes a loud error within ~limit_us.
// Ops (M1): MATVEC_G128 (the decode matvec, bitwise equal to gemm_tn_iu4g, kernels/k3_mv.hip's arithmetic) and NOP
// (busy work, for sync tests).
#include <hip/hip_runtime.h>
#include <stdint.h>

#define K3_MAXSEG 4096   // segments per tape (python/caminus/k3.py MAXSEG must match; the host checks nseg <= it)
enum { OP_NOP = 0, OP_MATVEC_G128 = 1, OP_GG_Q8_1 = 2, OP_GG_MV = 3, OP_GG_MOE_GU = 4, OP_GG_MOE_DOWN = 5,
       OP_F32_MV = 6, OP_ROUTER = 7, OP_SHEXP_ADD = 8, OP_RMS_MUL = 9, OP_HC_MEAN = 10, OP_HC_POST = 11,
       OP_GDN_CONV = 12, OP_GDN_L2 = 13, OP_GDN_GATES = 14, OP_GDN_STEP = 15, OP_GDN_NORM = 16,
       OP_QK_NR = 17, OP_KV_APPEND = 18, OP_ATTN_DEC = 19, OP_GATE_MUL = 20,
       OP_PLE_GATE = 21, OP_PLE_CONV = 22, OP_ADD_SUM = 23, OP_HADAMARD = 24,
       OP_HC_POST_RMS = 25, OP_XCHG = 26, OP_GDN_SAVE = 27, OP_HC_AVG = 28, OP_GG_GU_DENSE = 29,
       OP_BF16_MV = 30 };   // OP_GG_*: kernels/k3_gguf_ops.h (k3_gguf build only); OP_BF16_MV: the k3_gguf_gsq builds only

struct Seg {
    int op, ntask, dep, rows_per_task;
    const uint8_t * W;        // MATVEC: [N, K/2] repacked (caminus.k3.repack_mv)
    const float * gs;         // [N, K/128] group scales
    const int8_t * a;         // [K] activation codes
    const int32_t * asum;     // unused (the matvec sums each lane's activation codes itself)
    void * y;                 // [N] bf16 or fp32
    float rs; int N, K, out_f32;
    int work, team, pad1, pad2;   // team: -1 = any workgroup (k3_run2 teams; k3_run ignores it)
    int dep2p1;                   // a second dependency + 1 (0: none); k3_run waits for both
    const void * aux;             // one more pointer for fused ops
};
// bnd .. bnd_cap: the bound guard's report (bnd 2: k3_run's prologue below skipped the tape; 1: kernels/k3_guard.hip
// clamped the values): bnd_what -1 = the position
// (bnd_val = pos, bnd_cap = the KV cache's rows), -2 = the lazy commit (bnd_cap = T); python/caminus/k3.py bounds_status
struct Sync { uint32_t claim[K3_MAXSEG]; uint32_t done[K3_MAXSEG]; uint32_t exit; int err; int err_seg; uint32_t err_seen;
              int bnd, bnd_what, bnd_val, bnd_cap; };

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef int32_t i32x4 __attribute__((ext_vector_type(4)));

// K3_OPAQUE_TID: the workitem id through an empty asm, so the compiler cannot rematerialize / re-derive values from it
// (it otherwise keeps per-thread address arithmetic live across the whole persistent loop); K3WAVE is wave-uniform (SGPR)
#ifdef K3_OPAQUE_TID
__device__ __forceinline__ unsigned k3_tid() { unsigned x = __builtin_amdgcn_workitem_id_x(); asm volatile("" : "+v"(x)); return x; }
#define K3TID k3_tid()
#define K3WAVE ((int)__builtin_amdgcn_readfirstlane(K3TID / 32))
#else
#define K3TID threadIdx.x
#define K3WAVE ((int)(threadIdx.x / 32))
#endif
__device__ __forceinline__ bool thread0() { return K3TID == 0; }
__device__ __forceinline__ uint16_t to_bf16(float v) {
    const uint32_t u = __builtin_bit_cast(uint32_t, v);
    return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

__device__ __forceinline__ uint32_t claim(uint32_t * c) {
    __shared__ uint32_t s;
    __syncthreads();
    if (thread0()) s = atomicAdd(c, 1u);
    __syncthreads();
    return s;
}
__device__ __forceinline__ void publish(uint32_t * d) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (thread0()) __hip_atomic_fetch_add(d, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
// Waits are bounded in TIME (the steady counter, 100 MHz), and every wait bails as soon as any wait has given up (err
// set): a stuck tape then ends within ~limit_us instead of holding the GPU for seconds. That matters on the XTX: the
// desktop compositor shares it, and a launch that holds every CU for ~10 s makes its gfx ring time out (2026-10-31: a
// test build that waited ~30 s reset the ring and crashed gnome-shell).
__device__ __forceinline__ bool wait_expired(Sync * sy, int seg, uint32_t seen, uint64_t t0, int limit_us) {
    if (__hip_atomic_load(&sy->err, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) ||
        __builtin_readsteadycounter() - t0 > (uint64_t)limit_us * 100) {
        if (atomicExch(&sy->err, 1) == 0) { sy->err_seg = seg; sy->err_seen = seen; }
        return true;
    }
    return false;
}
// warm (nullptr = none): a word in the peer card's memory that gets a 4-byte store every ~8 us of the wait (OP_XCHG:
// keeps the path to the peer from going cold before the push; see k3_gguf_ops.h k3_extra_warm)
#ifndef K3_XWARM_TICKS
#define K3_XWARM_TICKS 800
#endif
#ifndef K3_XWARM_MASK
#define K3_XWARM_MASK 7
#endif
__device__ __forceinline__ void wait_for(Sync * sy, int seg, uint32_t target, int limit_us, uint32_t * warm = nullptr) {
    if (thread0()) {
        const uint64_t t0 = __builtin_readsteadycounter();
        uint64_t tw = t0;
        uint32_t v; int n = 0;
        while ((v = __hip_atomic_load(&sy->done[seg], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) < target) {
            __builtin_amdgcn_s_sleep(1);
            if (warm && (n & K3_XWARM_MASK) == K3_XWARM_MASK) {
                const uint64_t tn = __builtin_readsteadycounter();
                if (tn - tw >= K3_XWARM_TICKS) { tw = tn; __hip_atomic_store(warm, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); }
            }
            if ((++n & 63) == 0 && wait_expired(sy, seg, v, t0, limit_us)) break;
        }
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");   // one acquire after the wait, for every wave
}

// ---- MATVEC_G128. A task = rows [r0, r0 + rows_per_task). K is cut into passes of 1024 codes (a wave's 32 lanes x 32);
// with P passes the 8 waves split into P passes x floor(8 / P) row lanes (any waves left over idle): wave w takes pass
// w % P and rows w / P, w / P + 8 / P, ... (up to MV_U rows). All of a wave's weight loads (16 bytes per lane per row)
// and the task's group scales are issued together - for the first task of a segment BEFORE the dependency wait, since
// weights never depend on it; only the activation codes are loaded after. Each lane's 32-code dot is exact int32
// (sudot4 on offset-binary weights, minus 8 x the lane's activation sum); the 4 lanes of a 128-group sum to the group's
// P_g, written to LDS; then thread i runs row i's FMA chain over the groups in order (gemm_tn_iu4g's arithmetic, so the
// result is bitwise equal). Limits (checked on the host): K <= 8192, rows_per_task <= MV_U x (8 / P),
// rows_per_task x K / 128 <= MV_LDS.
#ifndef MV_U
#define MV_U 8       // 16 measured slower (173 vs 109 VGPRs): 81 vs 78 us on the token tape
#endif
#define MV_LDS 1024
struct MVPre { u32x4 w[MV_U]; };
typedef __attribute__((address_space(1))) const float GF;   // global (not flat) loads
__shared__ int32_t MVP[MV_LDS];
__shared__ float MVS[MV_LDS];

struct MVMap { int p, row0, rstep, nrow, k; bool on, kon; };
__device__ __forceinline__ MVMap mv_map(const Seg & sg, int t) {
    const int wave = K3WAVE, P = (sg.K + 1023) / 1024, nw = 8 / P;
    MVMap m;
    m.p = wave % P; m.row0 = wave / P; m.rstep = nw; m.on = wave < P * nw;
    m.nrow = min(sg.rows_per_task, sg.N - t * sg.rows_per_task);
    m.k = m.p * 1024 + (K3TID % 32) * 32; m.kon = m.on && m.k < sg.K;
    return m;
}

__device__ __forceinline__ void mv_load_w(const Seg & sg, int t, MVPre & pf) {
    const MVMap m = mv_map(sg, t);
    const uint8_t * base = sg.W + (size_t)(t * sg.rows_per_task) * (sg.K / 2) + m.k / 2;
#pragma unroll
    for (int i = 0; i < MV_U; i++) {
        const int r = m.row0 + i * m.rstep;
        if (!m.on || r >= m.nrow) break;
        if (m.kon) pf.w[i] = *(const u32x4 *)(base + (size_t)r * (sg.K / 2));
    }
}
__device__ __forceinline__ void mv_load_s(const Seg & sg, int t) {
    const MVMap m = mv_map(sg, t);
    // the task's group scales -> LDS, group-major (MVS[g * nrow + r]: the chain's reads are then conflict-free);
    // n <= MV_LDS = 4 x 256, so each thread issues its <= 4 loads before any write
    const int G = sg.K / 128, n = m.nrow * G;
    const GF * src = (const GF *)(sg.gs + (size_t)(t * sg.rows_per_task) * G);
    float v[MV_LDS / 256];
#pragma unroll
    for (int j = 0; j < MV_LDS / 256; j++) { const int i = K3TID + 256 * j; if (i < n) v[j] = src[i]; }
#pragma unroll
    for (int j = 0; j < MV_LDS / 256; j++) {
        const int i = K3TID + 256 * j;
        if (i < n) MVS[(i % G) * m.nrow + i / G] = v[j];
    }
}
__device__ __forceinline__ void mv_load(const Seg & sg, int t, MVPre & pf) { mv_load_w(sg, t, pf); mv_load_s(sg, t); }

__device__ void op_matvec(const Seg & sg, int t, const MVPre & pf, uint64_t * tr = nullptr) {
    const MVMap m = mv_map(sg, t);
    const int lane = K3TID % 32, G = sg.K / 128, g = m.k / 128;
    i32x4 a0 = {0, 0, 0, 0}, a1 = {0, 0, 0, 0};
    if (m.kon) { a0 = *(const i32x4 *)(sg.a + m.k); a1 = *(const i32x4 *)(sg.a + m.k + 16); }
    int32_t sa = 0;   // this lane's 32 activation codes summed (exact; dot4 against ones)
#pragma unroll
    for (int j = 0; j < 4; j++) sa = __builtin_amdgcn_sudot4(true, a0[j], true, 0x01010101, sa, false);
#pragma unroll
    for (int j = 0; j < 4; j++) sa = __builtin_amdgcn_sudot4(true, a1[j], true, 0x01010101, sa, false);
#ifdef K3_TRACE
    if (tr) { if (lane == 0) MVP[MV_LDS - 1 - K3WAVE] = sa; __syncthreads(); if (thread0()) tr[5] = __builtin_readsteadycounter(); }
#endif
#pragma unroll
    for (int i = 0; i < MV_U; i++) {
        const int r = m.row0 + i * m.rstep;
        if (!m.on || r >= m.nrow) break;
        int32_t part = 0;
        if (m.kon) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t lo = pf.w[i][j] & 0x0F0F0F0Fu, hi = (pf.w[i][j] >> 4) & 0x0F0F0F0Fu;
                const int32_t al = j < 2 ? a0[2 * j] : a1[2 * (j - 2)], ah = j < 2 ? a0[2 * j + 1] : a1[2 * (j - 2) + 1];
                part = __builtin_amdgcn_sudot4(true, al, false, (int)lo, part, false);
                part = __builtin_amdgcn_sudot4(true, ah, false, (int)hi, part, false);
            }
            part -= 8 * sa;   // offset-binary correction
        }
        // the group's 4 lanes: DPP quad permutes xor 1 (0xB1) and xor 2 (0x4E); int32, so exact in any order
        part += __builtin_amdgcn_update_dpp(0, part, 0xB1, 0xF, 0xF, false);
        part += __builtin_amdgcn_update_dpp(0, part, 0x4E, 0xF, 0xF, false);
        if ((lane & 3) == 0 && m.kon) MVP[g * m.nrow + r] = part;
    }
    __syncthreads();
#ifdef K3_TRACE
    if (tr && thread0()) tr[6] = __builtin_readsteadycounter();
#endif
    if ((int)K3TID < m.nrow) {
        const int r = K3TID, n = t * sg.rows_per_task + r;
        float acc = 0.f; int gg = 0;
        for (; gg + 8 <= G; gg += 8) {
            int32_t pv[8]; float sv[8];
#pragma unroll
            for (int j = 0; j < 8; j++) { pv[j] = MVP[(gg + j) * m.nrow + r]; sv[j] = MVS[(gg + j) * m.nrow + r]; }
#pragma unroll
            for (int j = 0; j < 8; j++) acc = __builtin_fmaf((float)pv[j], sv[j], acc);
        }
        for (; gg < G; gg++) acc = __builtin_fmaf((float)MVP[gg * m.nrow + r], MVS[gg * m.nrow + r], acc);
        const float v = acc * sg.rs;
        if (sg.out_f32) ((float *)sg.y)[n] = v; else ((uint16_t *)sg.y)[n] = to_bf16(v);
    }
}

// K3_QSA (kernels/k3_gguf_gsq_qsa*.hip: the trunk module with the QSA indexer ops): op_nop forced inline. There the inliner
// left it out of line, and a call inside k3_run's segment loop moved its SGPR spills from VGPR lanes to scratch memory
// (436 scratch loads; fn_bench 9.9 -> 18.0 ms per token). Other modules keep the inliner's choice (byte for byte the same)
#ifdef K3_QSA
#define K3_NOP_INLINE __forceinline__
#else
#define K3_NOP_INLINE
#endif
__device__ K3_NOP_INLINE void op_nop(const Seg & sg) {
    float v = K3TID;
    for (int i = 0; i < sg.work; i++) v = v * 1.0001f + 0.5f;
    if (v == -1.0f) ((float *)sg.y)[0] = v;
}

// K3_TRACE (measurement builds): thread 0 of each workgroup records the steady counter (100 MHz) per segment at its first
// claim's return, the end of the dependency wait, the end of its first task's compute and the end of its publish, plus
// the task id, and inside the matvec the activation codes' arrival and the P_g barrier (OP_XCHG: push + fence done and
// the peer's flag seen), into trace[(wg * nseg + s) * 8 + 0..6], and the end of its last publish into [7]; nullptr = off
// Extra ops (the GGUF formats, kernels/k3_gguf_ops.h) are compiled in only by builds that define K3_EXTRA_OPS_HEADER
// (kernels/k3_gguf.hip); they provide k3_extra_prefetch (before the dependency wait) and k3_extra_op (returns true if
// it ran the task; `fresh` = the workgroup's first task in this segment; sy / limit_us / s for ops with waits of their
// own, such as the cross-card exchange).
#ifdef K3_EXTRA_OPS_HEADER
#include K3_EXTRA_OPS_HEADER
#else
__device__ __forceinline__ void k3_extra_prefetch(const Seg &, int) {}
__device__ __forceinline__ bool k3_extra_op(const Seg &, int, bool, Sync *, int, int, uint64_t * = nullptr) { return false; }
__device__ __forceinline__ uint32_t * k3_extra_warm(const Seg &) { return nullptr; }
#endif

#ifndef K3_WAVES_PER_EU
#define K3_WAVES_PER_EU 1
#endif
#if defined(K3_MAX_VGPR)   // a VGPR cap: gfx11 allocates VGPRs in granules of 24 (wave32), so 120 = 12 waves / SIMD, 144 = 10
#define K3_RUN_ATTR __attribute__((amdgpu_num_vgpr(K3_MAX_VGPR)))
#elif defined(K3_MIN_WAVES)   // waves per SIMD the register allocator must allow (caps VGPRs at 1536 / K3_MIN_WAVES)
#define K3_RUN_ATTR __attribute__((amdgpu_waves_per_eu(K3_MIN_WAVES, 16), amdgpu_num_vgpr(1536 / K3_MIN_WAVES)))
#else
#define K3_RUN_ATTR
#endif
// the bound guard's check (k3_run's prologue): what = -1 (the position), -2 (the commit) or 0 (in range); val = the value
__device__ __forceinline__ int k3_guard_check(const int * gpos, const int * gparams, int gcap, int gT, int & val) {
    int what = 0; val = 0;
    if (gpos) { const int p = *gpos; if (p < 0 || (long long)p + gT > (long long)gcap) { what = -1; val = p; } }
    if (gparams) { const int m = gparams[1]; if (!what && (m < 0 || m > gT)) { what = -2; val = m; } }
    return what;
}
// its report (one thread): not inlined on purpose, inlined it shifted k3_run's register allocation (a VGPR spill in the
// k3_gguf_gsq_mt module, more SGPR spills elsewhere)
__device__ __noinline__ void k3_guard_report(Sync * sy, int what, int val, int bound) {
    sy->bnd_what = what; sy->bnd_val = val; sy->bnd_cap = bound; sy->bnd = 2;   // 2: the tape was skipped
    if (atomicCAS(&sy->err, 0, 1) == 0) { sy->err_seg = 0; sy->err_seen = 0xFFFFFFFFu; }
    __threadfence();
}
// gpos / gparams / gcap / gT: the bound guard (defense in depth behind the host checks, k3.check_span; k3.Engine.run passes
// the tape's position word with its smallest KV cache's rows, and params when an op replays the lazy commit params[1]).
// Every workgroup reads the two words first (uniform loads; no op of a tape writes them), and when the T positions
// pos .. pos + gT - 1 leave [0, gcap) or the commit leaves [0, gT], it returns before claiming any task: no task of the
// launch runs, so nothing is written (the KV cache, the states and every output keep their bytes) and the launch ends at
// once. Workgroup 0 reports it in the sync block (bnd words + err; the claim / done counters stay 0) and the host raises
// k3.ContextBoundsError. gpos = gparams = nullptr: no check. Not inside the ops on purpose: checks in op_kv_append /
// op_attn_dec / the replays reshuffled k3_run's registers at its 96-VGPR cap (+0.64 ms per token, Coder T = 1), and a
// separate guard dispatch per launch (kernels/k3_guard.hip) cost ~0.05 - 0.10 ms per token.
extern "C" __global__ void __launch_bounds__(256, K3_WAVES_PER_EU) K3_RUN_ATTR k3_run(const Seg * __restrict__ tape, int nseg, Sync * sy, int limit_us,
                                                          uint64_t * trace, const int * gpos, const int * gparams, int gcap, int gT) {
    {
        int val; const int what = k3_guard_check(gpos, gparams, gcap, gT, val);
        if (what && blockIdx.x == 0 && threadIdx.x == 0) k3_guard_report(sy, what, val, what == -1 ? gcap : gT);
        if (what) return;
    }
    for (int s = 0; s < nseg; s++) {
        const Seg sg = tape[s];
#ifdef K3_NO_G128
        const bool mv = false;   // builds without the S1 int4 matvec (the GGUF engine): its prefetch registers cost occupancy
#else
        const bool mv = sg.op == OP_MATVEC_G128;
#endif
        // claim the first task BEFORE waiting on the dependency and issue its weight and scale loads
        // (prefetch-before-wait): safe, a claimed task only runs after the wait, and every task of the dependency was
        // claimed (by a running workgroup) before this one got here
        uint32_t t = claim(&sy->claim[s]);
#ifdef K3_TRACE
        uint64_t * tr = trace ? trace + ((size_t)blockIdx.x * nseg + s) * 8 : nullptr;
        if (thread0() && tr) { tr[0] = __builtin_readsteadycounter(); tr[4] = t; tr[2] = 0; tr[3] = 0; }
        bool first = true;
#endif
        MVPre pf;
        if (mv && t < (uint32_t)sg.ntask) mv_load(sg, (int)t, pf);
        else if (t < (uint32_t)sg.ntask) k3_extra_prefetch(sg, (int)t);
        // only a workgroup that claimed a task waits; one that got none moves on, so independent segments later in the
        // tape (a parallel branch) start while earlier ones still wait. Safe: claims stay in tape order, so the lowest
        // unfinished segment's tasks are held by running workgroups whose dependencies are all done
        if (t < (uint32_t)sg.ntask) {
            if (sg.dep >= 0) wait_for(sy, sg.dep, (uint32_t)tape[sg.dep].ntask, limit_us, k3_extra_warm(sg));
            if (sg.dep2p1 > 0) wait_for(sy, sg.dep2p1 - 1, (uint32_t)tape[sg.dep2p1 - 1].ntask, limit_us);
        }
#ifdef K3_TRACE
        if (thread0() && tr) tr[1] = __builtin_readsteadycounter();
#endif
        bool fresh = true;
        while (t < (uint32_t)sg.ntask) {
#ifdef K3_TRACE
            if (mv) op_matvec(sg, (int)t, pf, first ? tr : nullptr); else if (!k3_extra_op(sg, (int)t, fresh, sy, limit_us, s, first ? tr : nullptr)) op_nop(sg);
#else
            if (mv) op_matvec(sg, (int)t, pf); else if (!k3_extra_op(sg, (int)t, fresh, sy, limit_us, s)) op_nop(sg);
#endif
#ifdef K3_TRACE
            __syncthreads();
            if (thread0() && first && tr) tr[2] = __builtin_readsteadycounter();
#endif
            fresh = false;
            publish(&sy->done[s]);
#ifdef K3_TRACE
            if (thread0() && tr) { tr[7] = __builtin_readsteadycounter(); if (first) tr[3] = tr[7]; }
            first = false;
#endif
            t = claim(&sy->claim[s]);
            if (mv && t < (uint32_t)sg.ntask) mv_load(sg, (int)t, pf);
        }
    }
    // the last workgroup out resets the counters for the next launch; after a bail (err) it keeps them for the host's
    // stall report (Engine.stall_report: where the chain stopped), and the host clears the sync block before relaunching
    __syncthreads();
    if (thread0()) {
        const uint32_t n = __hip_atomic_fetch_add(&sy->exit, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (n + 1 == gridDim.x) {
            if (!__hip_atomic_load(&sy->err, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
                for (int s = 0; s < nseg; s++) { sy->claim[s] = 0; sy->done[s] = 0; }
            sy->exit = 0;
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        }
    }
}

// ---- k3_run2: teams (measurement of lever 1, docs/K3-megakernel-design.md "M1 engine skeleton, measured"). The grid
// is cut into up to 4 teams by `tb` (team j = workgroups [tb[j], tb[j + 1]), tb[nteam] = grid); a segment with team >= 0
// is run only by that team (team -1: everyone). With stage-1 and stage-2 segments alternating over two teams each
// (even / odd blocks), a workgroup claims its next segment and issues that task's weight loads two segments before the
// one it waits on, so the weights have about a block to arrive. (Claiming further ahead into a second register buffer
// does not work: vmcnt is in order per wave, so waiting on the current task's activation codes also waits on every
// load issued before them.) Deadlock freedom as in k3_run, plus: a team may be absent (partial residency), so a wait
// that stalls K3_HELP_US while the awaited segment still has unclaimed tasks whose own dependency is done runs
// them itself.
#define K3_HELP_US 200   // a wait stalled this long, with unclaimed runnable tasks in the awaited segment, helps it
__device__ __forceinline__ bool mine(int tm, int team) { return tm < 0 || tm == team; }

__device__ void run_task(const Seg & sg, int t, MVPre & buf) {
    if (sg.op == OP_MATVEC_G128) op_matvec(sg, t, buf); else op_nop(sg);
}
// the help path's task runner, kept out of line: inlined, its second copy of the matvec costs ~100 VGPRs (occupancy)
__device__ __attribute__((noinline)) void help_task(const Seg & sg, int t) {
    MVPre buf;
    if (sg.op == OP_MATVEC_G128) { mv_load(sg, t, buf); op_matvec(sg, t, buf); } else op_nop(sg);
}

// wait until segment d is done, helping it if it stalls with unclaimed runnable tasks; true if it helped (the LDS
// scales were overwritten)
__device__ bool wait_help(const Seg * tape, Sync * sy, int d, int limit_us) {
    __shared__ int s_state;   // 0 done, 1 help, 2 keep waiting, 3 gave up
    const Seg sd = tape[d];
    bool helped = false;
    const uint64_t t0 = __builtin_readsteadycounter();
    for (;;) {
        if (thread0()) {
            int st = 2; uint32_t v = 0;
            const uint64_t r0 = __builtin_readsteadycounter();
            for (int n = 1;; n++) {
                v = __hip_atomic_load(&sy->done[d], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                if (v >= (uint32_t)sd.ntask) { st = 0; break; }
                __builtin_amdgcn_s_sleep(1);
                if ((n & 63) == 0) {
                    if (wait_expired(sy, d, v, t0, limit_us)) { st = 3; break; }
                    if (__builtin_readsteadycounter() - r0 > (uint64_t)K3_HELP_US * 100) break;
                }
            }
            if (st == 2 && __hip_atomic_load(&sy->claim[d], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < (uint32_t)sd.ntask &&
                (sd.dep < 0 || __hip_atomic_load(&sy->done[sd.dep], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) >=
                                   (uint32_t)tape[sd.dep].ntask)) st = 1;
            s_state = st;
        }
        __syncthreads();
        const int st = s_state;
        __syncthreads();
        if (st == 0 || st == 3) break;
        if (st == 1) {
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");   // d's dependency is done: see its outputs
            for (;;) {
                const uint32_t t = claim(&sy->claim[d]);
                if (t >= (uint32_t)sd.ntask) break;
                help_task(sd, (int)t);
                publish(&sy->done[d]);
            }
            helped = true;
        }
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    return helped;
}

extern "C" __global__ void __launch_bounds__(256) k3_run2(const Seg * __restrict__ tape, int nseg, Sync * sy, int limit_us,
                                                           int tb1, int tb2, int tb3) {
    const int b = blockIdx.x;
    const int team = b >= tb3 ? 3 : b >= tb2 ? 2 : b >= tb1 ? 1 : 0;
    MVPre pf;
    for (int s = 0; s < nseg; s++) {
        const Seg sg = tape[s];
        if (!mine(sg.team, team)) continue;
        const bool mv = sg.op == OP_MATVEC_G128;
        uint32_t t = claim(&sy->claim[s]);
        if (mv && t < (uint32_t)sg.ntask) mv_load(sg, (int)t, pf);
        if (sg.dep >= 0 && wait_help(tape, sy, sg.dep, limit_us))   // helped: restage the scales it overwrote
            if (mv && t < (uint32_t)sg.ntask) mv_load_s(sg, (int)t);
        while (t < (uint32_t)sg.ntask) {
            run_task(sg, (int)t, pf);
            publish(&sy->done[s]);
            t = claim(&sy->claim[s]);
            if (mv && t < (uint32_t)sg.ntask) mv_load(sg, (int)t, pf);
        }
    }
    __syncthreads();
    if (thread0()) {
        const uint32_t n = __hip_atomic_fetch_add(&sy->exit, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (n + 1 == gridDim.x) {
            for (int i = 0; i < nseg; i++) { sy->claim[i] = 0; sy->done[i] = 0; }
            sy->exit = 0;
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        }
    }
}
