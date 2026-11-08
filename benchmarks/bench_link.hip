// I0 link bench (docs/INFERENCE-PLAN.md): how fast the XTX (HIP device 0) and the XT (device 1, OCuLink PCIe 4.0 x4)
// can exchange data and signals. (1) peer access available both ways; (2) copy latency / bandwidth 64 B - 64 MB, peer
// and host-staged, both directions; (3) device-to-device signal round trip: one single-thread kernel per card ping-pongs
// a counter, flags either in the receiving card's VRAM (written by the peer over P2P) or in fine-grained pinned host
// memory. Kernels are single-thread and give up after LIMIT_MS (steady counter), so the desktop compositor is safe.
//   build: see the command in docs/INFERENCE-PLAN.md I0 or tools; run: build/bench_link
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)
static const int LIMIT_MS = 200;

// role 0 starts: writes i to the peer's flag, waits for its own flag == i; role 1 waits, then answers
extern "C" __global__ void pingpong(uint32_t * mine, uint32_t * peer, int iters, int role, uint64_t * out) {
    const uint64_t t0 = __builtin_readsteadycounter(), limit = (uint64_t)LIMIT_MS * 100000;   // 100 MHz
    int done = 0;
    for (int i = 1; i <= iters; i++) {
        if (role == 0) __hip_atomic_store(peer, (uint32_t)i, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        bool ok = true;
        while (__hip_atomic_load(mine, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < (uint32_t)i) {
            if (__builtin_readsteadycounter() - t0 > limit) { ok = false; break; }
        }
        if (!ok) break;
        if (role == 1) __hip_atomic_store(peer, (uint32_t)i, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        done = i;
    }
    out[0] = __builtin_readsteadycounter() - t0; out[1] = done;
}

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
extern "C" __global__ void pingpong_payload(uint32_t * mine, uint32_t * peer, u32x4 * my_buf, u32x4 * peer_buf, int words,
                                             int iters, int role, uint64_t * out) {
    __shared__ int s_ok;
    const uint64_t t0 = __builtin_readsteadycounter(), limit = (uint64_t)LIMIT_MS * 100000;
    const int lane = threadIdx.x;
    int done = 0; uint32_t bad = 0;
    for (int i = 1; i <= iters; i++) {
        if (role == 0) {
            for (int w = lane; w < words; w += blockDim.x) peer_buf[w] = u32x4{(uint32_t)i, (uint32_t)w, 7u, 9u};
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");          // system scope
            __syncthreads();
            if (lane == 0) __hip_atomic_store(peer, (uint32_t)i, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
        if (lane == 0) {
            s_ok = 1;
            while (__hip_atomic_load(mine, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < (uint32_t)i)
                if (__builtin_readsteadycounter() - t0 > limit) { s_ok = 0; break; }
        }
        __syncthreads();
        if (!s_ok) break;
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
        for (int w = lane; w < words; w += blockDim.x) { const u32x4 v = my_buf[w]; bad += (v[0] != (uint32_t)i) | (v[1] != (uint32_t)w); }
        if (role == 1) {
            for (int w = lane; w < words; w += blockDim.x) peer_buf[w] = u32x4{(uint32_t)i, (uint32_t)w, 7u, 9u};
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
            __syncthreads();
            if (lane == 0) __hip_atomic_store(peer, (uint32_t)i, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
        done = i;
    }
    if (lane == 0) { out[0] = __builtin_readsteadycounter() - t0; out[1] = done; }
    if (bad) atomicAdd((unsigned long long *)&out[2], (unsigned long long)bad);
}

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
    int n = 0; CK(hipGetDeviceCount(&n));
    if (n < 2) { printf("need 2 devices, have %d\n", n); return 1; }
    for (int d = 0; d < 2; d++) { hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, d)); printf("device %d: %s, %.1f GB, pci %02x:%02x\n", d, p.name, p.totalGlobalMem / 1e9, p.pciBusID, p.pciDeviceID); }
    int c01 = 0, c10 = 0; CK(hipDeviceCanAccessPeer(&c01, 0, 1)); CK(hipDeviceCanAccessPeer(&c10, 1, 0));
    printf("peer access: 0->1 %d, 1->0 %d\n", c01, c10);
    bool p2p = c01 && c10;
    if (p2p) { CK(hipSetDevice(0)); CK(hipDeviceEnablePeerAccess(1, 0)); CK(hipSetDevice(1)); CK(hipDeviceEnablePeerAccess(0, 0)); }

    const size_t MAXB = 64 << 20;
    void * buf[2]; hipStream_t st[2];
    for (int d = 0; d < 2; d++) { CK(hipSetDevice(d)); CK(hipMalloc(&buf[d], MAXB)); CK(hipMemset(buf[d], d, MAXB)); CK(hipStreamCreate(&st[d])); }
    void * host; CK(hipHostMalloc(&host, MAXB, hipHostMallocDefault));
    const size_t sizes[] = {64, 4 << 10, 20 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20};

    printf("\ncopy (us per copy, GB/s)          peer 0->1        peer 1->0        staged 0->h->1   staged 1->h->0\n");
    for (size_t b : sizes) {
        const int it = b <= (64 << 10) ? 400 : b <= (4 << 20) ? 50 : 8;
        double r[4] = {0, 0, 0, 0};
        for (int dir = 0; dir < 2; dir++) {
            const int s = dir, t = 1 - dir;
            if (p2p) {
                CK(hipSetDevice(s)); CK(hipMemcpyPeerAsync(buf[t], t, buf[s], s, b, st[s])); CK(hipStreamSynchronize(st[s]));
                const double t0 = now_us();
                for (int i = 0; i < it; i++) CK(hipMemcpyPeerAsync(buf[t], t, buf[s], s, b, st[s]));
                CK(hipStreamSynchronize(st[s])); r[dir] = (now_us() - t0) / it;
            }
            const double t1 = now_us();
            for (int i = 0; i < it; i++) {
                CK(hipSetDevice(s)); CK(hipMemcpyAsync(host, buf[s], b, hipMemcpyDeviceToHost, st[s])); CK(hipStreamSynchronize(st[s]));
                CK(hipSetDevice(t)); CK(hipMemcpyAsync(buf[t], host, b, hipMemcpyHostToDevice, st[t])); CK(hipStreamSynchronize(st[t]));
            }
            r[2 + dir] = (now_us() - t1) / it;
        }
        printf("%9zu B  ", b);
        for (int k = 0; k < 4; k++) {
            if (k < 2 && !p2p) { printf("      n/a        "); continue; }
            printf("%8.1f %6.2f  ", r[k], b / r[k] / 1e3);
        }
        printf("\n");
    }

    // signal and payload round trips; memory kinds for the flags and the payload buffers (each lives on the receiver)
    const int iters = 1000;
    struct Kind { const char * name; int mode; };   // 0 cached VRAM, 1 uncached VRAM, 2 fine-grained VRAM, 3 pinned host
    const Kind kinds[] = {{"VRAM, default (cached)", 0}, {"VRAM, uncached", 1}, {"VRAM, fine-grained", 2}, {"pinned host, coherent", 3}};
    uint64_t * out[2];
    for (int d = 0; d < 2; d++) { CK(hipSetDevice(d)); CK(hipMalloc(&out[d], 32)); }
    const size_t PB = 128 << 10;
    printf("\nround trips (us), %d iterations, one 32-lane wave per card, payload pushed into the receiver's buffer, limit %d ms\n", iters, LIMIT_MS);
    printf("%-26s %10s %10s %10s %10s %10s\n", "flag + payload memory", "0 B", "5 KB", "20 KB", "80 KB", "check");
    for (const Kind & k : kinds) {
        if (k.mode < 3 && !p2p) continue;
        uint32_t * f[2]; void * pb[2];
        for (int d = 0; d < 2; d++) {
            CK(hipSetDevice(d));
            if (k.mode == 0) { CK(hipMalloc(&f[d], 256)); CK(hipMalloc(&pb[d], PB)); }
            else if (k.mode == 1) { CK(hipExtMallocWithFlags((void **)&f[d], 256, hipDeviceMallocUncached)); CK(hipExtMallocWithFlags(&pb[d], PB, hipDeviceMallocUncached)); }
            else if (k.mode == 2) { CK(hipExtMallocWithFlags((void **)&f[d], 256, hipDeviceMallocFinegrained)); CK(hipExtMallocWithFlags(&pb[d], PB, hipDeviceMallocFinegrained)); }
            else { CK(hipHostMalloc((void **)&f[d], 256, hipHostMallocCoherent | hipHostMallocMapped)); CK(hipHostMalloc(&pb[d], PB, hipHostMallocCoherent | hipHostMallocMapped)); }
        }
        printf("%-26s", k.name);
        unsigned long long badsum = 0;
        for (int bytes : {0, 5 << 10, 20 << 10, 80 << 10}) {
            double best = 1e30; bool ok = true;
            for (int rep = 0; rep < 3; rep++) {
                for (int d = 0; d < 2; d++) { CK(hipSetDevice(d)); CK(hipMemset(f[d], 0, 4)); CK(hipMemset(out[d], 0, 32)); CK(hipDeviceSynchronize()); }
                const int words = bytes / 16;
                CK(hipSetDevice(1)); hipLaunchKernelGGL(pingpong_payload, dim3(1), dim3(32), 0, st[1], f[1], f[0], (u32x4 *)pb[1], (u32x4 *)pb[0], words, iters, 1, out[1]);
                CK(hipSetDevice(0)); hipLaunchKernelGGL(pingpong_payload, dim3(1), dim3(32), 0, st[0], f[0], f[1], (u32x4 *)pb[0], (u32x4 *)pb[1], words, iters, 0, out[0]);
                CK(hipStreamSynchronize(st[0])); CK(hipSetDevice(1)); CK(hipStreamSynchronize(st[1]));
                uint64_t o0[3], o1[3]; CK(hipSetDevice(0)); CK(hipMemcpy(o0, out[0], 24, hipMemcpyDeviceToHost)); CK(hipSetDevice(1)); CK(hipMemcpy(o1, out[1], 24, hipMemcpyDeviceToHost));
                if (o0[1] != (uint64_t)iters) { ok = false; break; }
                badsum += o0[2] + o1[2];
                best = std::min(best, o0[0] / 100.0 / iters);
            }
            if (ok) printf(" %10.2f", best); else printf(" %10s", "TIMEOUT");
        }
        printf(" %10s\n", badsum ? "CORRUPT" : "ok");
    }
    printf("done\n");
    return 0;
}
