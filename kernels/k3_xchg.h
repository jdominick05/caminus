// The two-card exchange op (OP_XCHG): the all-reduce of a partial vector between two cards, written for Caminus (no
// ggml code). Included by kernels/k3_gguf_ops.h (the GGUF modules, as it always was there) and, under K3_TP, by the own
// dense qwen35 modules' ops (kernels/k3_q35_ops.h, kernels/k3_q35t_ops.h: the two-card modules k3_q35_*_tp). Needs the
// K3 core (kernels/k3.hip: Seg, Sync, wait_expired, K3TID) before it.
#pragma once

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
#ifdef K3_XCHG_PUSH_ONLY
    int N = sg.N;
    asm volatile("" : "+s"(N));   // keep exchange geometry inside its dispatch branch
    const int tid = K3TID, B = blockDim.x, N4 = N / 4;
#else
    const int tid = K3TID, B = blockDim.x, N4 = sg.N / 4;
#endif
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
#ifdef K3_XCHG_PUSH_ONLY
    int index = sg.K;
    asm volatile("" : "+s"(index));   // keep the exchange epoch offset branch-local
    const uint32_t want = (uint32_t)((const int *)sg.asum)[0] + (uint32_t)index + 1u;
    uintptr_t peer_addr = (uintptr_t)sg.W;
    asm volatile("" : "+v"(peer_addr));   // keep the remote payload address inside the exchange
    float * peer = (float *)peer_addr;
#else
    const uint32_t want = (uint32_t)((const int *)sg.asum)[0] + (uint32_t)sg.K + 1u;
    float * peer = (float *)sg.W;
#endif
    const float * mine = (const float *)sg.gs;
    // pad2 = 1 (pull, the producing card): the payload goes to aux (this card's own uncached slot), only the flag to the
    // peer; pad2 = 2 (pull, the reading card): the peer's partial is read from aux (the peer's uncached slot)
#if defined(K3_MT) || defined(K3_XCHG_PUSH_ONLY)
    // Multi-token and Q8 TP verify modules only push; unused pull selects spill vectors.
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
#if defined(K3_MT) || defined(K3_XCHG_PUSH_ONLY)
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

// the dependency wait's warm word and the claim-time store for OP_XCHG (as kernels/k3_gguf_ops.h k3_extra_warm /
// k3_extra_prefetch with K3_XWARM: the first remote store after ~15-20 us of no traffic to the peer takes ~4.4 us
// longer; a store into the peer slot's header word 1, read by nobody, keeps the path warm until the push)
__device__ __forceinline__ uint32_t * xchg_warm(const Seg & sg) {
    return (sg.op == OP_XCHG && !sg.pad1) ? (uint32_t *)sg.W + 1 : nullptr;
}
__device__ __forceinline__ void xchg_prefetch(const Seg & sg) {
    if (sg.op == OP_XCHG && !sg.pad1 && thread0()) __hip_atomic_store((uint32_t *)sg.W + 1, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
