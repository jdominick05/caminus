// rdna3-stack C++ runtime: HIP helpers, device arena, code-object kernels, tensor files, rocBLAS linear.
// The process links torch's bundled libamdhip64 / librocblas (and their Tensile files) so GEMMs are the exact
// binaries PyTorch training ran; kernels are the gfx1100 code objects in build/ (kernels/*.hip).
#pragma once
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d: %s\n", hipGetErrorString(e_), __FILE__, __LINE__, #x); exit(1); } } while (0)
#define BLAS_CHECK(x) do { rocblas_status s_ = (x); if (s_ != rocblas_status_success) { \
    fprintf(stderr, "rocBLAS status %d at %s:%d: %s\n", (int)s_, __FILE__, __LINE__, #x); exit(1); } } while (0)

namespace rt {

enum class DType { f32, bf16, i32, i64 };
size_t dsize(DType t);
DType dtype_of(const std::string & s);
const char * dname(DType t);

struct Tensor {
    void * ptr = nullptr;               // device memory
    DType dt = DType::f32;
    std::vector<int64_t> shape;
    int64_t numel() const { int64_t n = 1; for (auto d : shape) n *= d; return n; }
    size_t bytes() const { return numel() * dsize(dt); }
    int64_t dim(int i) const { return shape[i < 0 ? shape.size() + i : i]; }
    template <typename T> T * as() const { return (T *)ptr; }
};

// one device allocation, handed out by offset; reset() reuses it (no allocation inside a step)
struct Arena {
    char * base = nullptr; size_t cap = 0, used = 0;
    void init(size_t bytes);
    Tensor alloc(DType dt, std::vector<int64_t> shape);
    void reset(size_t to = 0) { used = to; }
};

// tensors from an .idx/.bin pair written by training/tools/dump_golden.py (`name dtype d0,d1,.. offset nbytes`)
struct TensorFile {
    std::map<std::string, Tensor> t;
    void load(const std::string & stem, Arena & arena);       // uploads every tensor
    const Tensor & operator[](const std::string & name) const;
    bool has(const std::string & name) const { return t.count(name) > 0; }
};

// a code object from build/<name>.hsaco and its kernels; launch() packs arguments the way hipModuleLaunchKernel wants
struct Module {
    hipModule_t mod = nullptr;
    std::map<std::string, hipFunction_t> fns;
    void load(const std::string & name);
    hipFunction_t fn(const std::string & name);
};

struct Args {
    std::vector<std::vector<char>> store; std::vector<void *> ptrs;
    template <typename T> Args & add(T v) {
        store.emplace_back(sizeof(T)); memcpy(store.back().data(), &v, sizeof(T)); return *this;
    }
    void ** pack() { ptrs.clear(); for (auto & s : store) ptrs.push_back(s.data()); return ptrs.data(); }
};

void launch(hipFunction_t f, dim3 grid, dim3 block, Args & a, hipStream_t s, unsigned shmem = 0);

// Triton kernels exported by training/tools/capture_triton.py (build/triton/manifest.txt): the exact code objects and
// parameter lists a PyTorch forward launched. call() passes named pointers from `ptrs`, the recorded scalars, then
// Triton's two trailing scratch pointers (null: no captured kernel uses scratch).
struct TritonLaunch {
    std::string kernel, file; unsigned grid[3] = {1, 1, 1}, block = 0, shared = 0;
    struct P { std::string name, kind; double value = 0; };   // kind: ptr | null | i32 | i64 | fp32
    std::vector<P> params; hipFunction_t fn = nullptr;
};
struct TritonManifest {
    std::vector<TritonLaunch> launches; std::map<std::string, Module> mods;
    void load(const std::string & dir);
    void call(const TritonLaunch & L, const std::map<std::string, void *> & ptrs, hipStream_t s);
};

// y[M, N] = x[M, K] @ w[N, K]^T, bf16 in / bf16 out / fp32 compute: the exact rocblas_gemm_ex call that reproduces
// torch's F.linear bit for bit (verified against torch 2.14's bundled rocBLAS 5.2)
void linear(rocblas_handle h, const Tensor & x, const Tensor & w, Tensor & y);

// host copies and comparisons for tests
std::vector<float> to_host_f32(const Tensor & t);
struct Diff { int64_t n = 0, bad = 0; double max_abs = 0, rel = 0; bool exact() const { return bad == 0; } };
Diff compare(const Tensor & got, const Tensor & want);

std::string root_dir();   // the repository root (for build/ and golden paths)

}  // namespace rt
