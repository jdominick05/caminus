// K3 engine cost model, part 1: what one stage handoff costs inside a persistent kernel on gfx1100.
// claim_done: blocks claim tasks from a per-stage counter, publish finished ones (agent-scope release by every wave, one
// release add), and wait only for the previous stage's done count (acquire); the last block out resets the counters
// (ignite-llama megakernel.cu:35-47, 149-211). Nothing waits on a block that is not running, so the grid need not be
// co-resident. arrival: a generation barrier (every block arrives, the last flips the generation): only safe when the
// whole grid is resident; for comparison. empty: one empty kernel per stage (the per-node baseline).
// `work` busy-loop iterations per task stand in for a stage's compute.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define MAXS 512
struct Sync { uint32_t claim[MAXS]; uint32_t done[MAXS]; uint32_t arrive; uint32_t gen; uint32_t exit; int err; };

__device__ __forceinline__ bool t0() { return threadIdx.x == 0; }

__device__ __forceinline__ uint32_t claim(uint32_t * c, uint32_t n) {
    __shared__ uint32_t s;
    __syncthreads();
    if (t0()) s = atomicAdd(c, n);
    __syncthreads();
    return s;
}
__device__ __forceinline__ void publish(uint32_t * d, uint32_t n) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (t0()) __hip_atomic_fetch_add(d, n, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void wait_done(Sync * sy, const uint32_t * d, uint32_t target, int spin_limit) {
    if (t0()) {
        int spins = 0;
        while (__hip_atomic_load(d, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < target) {
            __builtin_amdgcn_s_sleep(1);
            if (++spins > spin_limit) { atomicExch(&sy->err, 1); break; }
        }
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}
__device__ __forceinline__ void busy(int work, float * sink) {
    float v = threadIdx.x;
    for (int i = 0; i < work; i++) v = v * 1.0001f + 0.5f;
    if (v == -1.0f) *sink = v;   // keeps the loop
}

extern "C" __global__ void __launch_bounds__(256) claim_done(Sync * sy, int stages, uint32_t tasks, int work, int spin_limit, float * sink) {
    for (int s = 0; s < stages; s++) {
        for (;;) {
            const uint32_t t = claim(&sy->claim[s], 1);
            if (t >= tasks) break;
            busy(work, sink);
            publish(&sy->done[s], 1);
        }
        if (s + 1 < stages) wait_done(sy, &sy->done[s], tasks, spin_limit);
    }
    // the last block out resets the counters for the next launch
    __syncthreads();
    if (t0()) {
        const uint32_t n = __hip_atomic_fetch_add(&sy->exit, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (n + 1 == gridDim.x) {
            for (int s = 0; s < stages; s++) { sy->claim[s] = 0; sy->done[s] = 0; }
            sy->exit = 0;
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) arrival(Sync * sy, int stages, uint32_t tasks, int work, int spin_limit, float * sink) {
    for (int s = 0; s < stages; s++) {
        busy(work, sink);
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __syncthreads();
        if (t0()) {
            const uint32_t g = __hip_atomic_load(&sy->gen, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
            const uint32_t n = __hip_atomic_fetch_add(&sy->arrive, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
            if (n + 1 == gridDim.x) {
                sy->arrive = 0;
                __hip_atomic_fetch_add(&sy->gen, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            } else {
                int spins = 0;
                while (__hip_atomic_load(&sy->gen, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == g) {
                    __builtin_amdgcn_s_sleep(1);
                    if (++spins > spin_limit) { atomicExch(&sy->err, 1); break; }
                }
            }
        }
        __syncthreads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
}

extern "C" __global__ void __launch_bounds__(256) empty_stage(float * sink, int work) { busy(work, sink); }

// Under load: every stage is `tasks` tasks (fixed, independent of the grid), each streaming `chunk16` 16-byte words of a
// cache-resident buffer (stage s reads region s mod nreg), like one stage of block matvecs. `wait` = 1 chains the stages
// with the claim/done hand-off; 0 lets blocks run on into the next stage (no dependency: the bandwidth-only time).
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
extern "C" __global__ void __launch_bounds__(256) stream_stages(Sync * sy, const u32x4 * __restrict__ buf, int stages, uint32_t tasks,
                                                                 uint32_t chunk16, int nreg, int wait, int spin_limit, uint32_t * out) {
    u32x4 acc = {0, 0, 0, 0};
    for (int s = 0; s < stages; s++) {
        const u32x4 * reg = buf + (size_t)(s % nreg) * tasks * chunk16;
        for (;;) {
            const uint32_t t = claim(&sy->claim[s], 1);
            if (t >= tasks) break;
            const u32x4 * p = reg + (size_t)t * chunk16;
            for (uint32_t i = threadIdx.x; i < chunk16; i += blockDim.x) acc ^= p[i];
            publish(&sy->done[s], 1);
        }
        if (wait && s + 1 < stages) wait_done(sy, &sy->done[s], tasks, spin_limit);
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9E3779B9u) *out = acc.x;
    __syncthreads();
    if (t0()) {
        const uint32_t n = __hip_atomic_fetch_add(&sy->exit, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (n + 1 == gridDim.x) {
            for (int s = 0; s < stages; s++) { sy->claim[s] = 0; sy->done[s] = 0; }
            sy->exit = 0;
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        }
    }
}
