#include "rt.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unistd.h>

namespace rt {

size_t dsize(DType t) { return t == DType::f32 || t == DType::i32 ? 4 : t == DType::bf16 ? 2 : 8; }

DType dtype_of(const std::string & s) {
    if (s == "f32") return DType::f32;
    if (s == "bf16") return DType::bf16;
    if (s == "i32") return DType::i32;
    if (s == "i64") return DType::i64;
    fprintf(stderr, "unknown dtype %s\n", s.c_str()); exit(1);
}

const char * dname(DType t) { return t == DType::f32 ? "f32" : t == DType::bf16 ? "bf16" : t == DType::i32 ? "i32" : "i64"; }

void Arena::init(size_t bytes) { HIP_CHECK(hipMalloc(&base, bytes)); cap = bytes; used = 0; }

Tensor Arena::alloc(DType dt, std::vector<int64_t> shape) {
    Tensor t; t.dt = dt; t.shape = std::move(shape);
    const size_t n = (t.bytes() + 255) & ~size_t(255);                     // 256-byte aligned
    if (used + n > cap) { fprintf(stderr, "arena full: need %zu, have %zu of %zu\n", n, cap - used, cap); exit(1); }
    t.ptr = base + used; used += n;
    return t;
}

void TensorFile::load(const std::string & stem, Arena & arena) {
    std::ifstream idx(stem + ".idx"); std::ifstream bin(stem + ".bin", std::ios::binary);
    if (!idx || !bin) { fprintf(stderr, "cannot open %s.{idx,bin}\n", stem.c_str()); exit(1); }
    std::string line; std::vector<char> buf;
    while (std::getline(idx, line)) {
        std::istringstream ls(line); std::string name, dt, dims; size_t off, nb;
        ls >> name >> dt >> dims >> off >> nb;
        Tensor x; x.dt = dtype_of(dt);
        std::stringstream ds(dims); std::string d;
        while (std::getline(ds, d, ',')) if (!d.empty()) x.shape.push_back(std::stoll(d));
        Tensor dev = arena.alloc(x.dt, x.shape);
        buf.resize(nb); bin.seekg(off); bin.read(buf.data(), nb);
        HIP_CHECK(hipMemcpy(dev.ptr, buf.data(), nb, hipMemcpyHostToDevice));
        t[name] = dev;
    }
}

const Tensor & TensorFile::operator[](const std::string & name) const {
    auto it = t.find(name);
    if (it == t.end()) { fprintf(stderr, "missing tensor %s\n", name.c_str()); exit(1); }
    return it->second;
}

std::string root_dir() {
    char p[4096]; ssize_t n = readlink("/proc/self/exe", p, sizeof(p) - 1); p[n > 0 ? n : 0] = 0;
    std::string s(p); for (int i = 0; i < 2; i++) s = s.substr(0, s.rfind('/'));   // <root>/build/<exe>
    return s;
}

void Module::load(const std::string & name) {
    std::string path = root_dir() + "/build/" + name + ".hsaco";
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    std::vector<char> img((std::istreambuf_iterator<char>(f)), {});
    HIP_CHECK(hipModuleLoadData(&mod, img.data()));
}

hipFunction_t Module::fn(const std::string & name) {
    auto it = fns.find(name);
    if (it != fns.end()) return it->second;
    hipFunction_t f; HIP_CHECK(hipModuleGetFunction(&f, mod, name.c_str()));
    return fns[name] = f;
}

void launch(hipFunction_t f, dim3 grid, dim3 block, Args & a, hipStream_t s, unsigned shmem) {
    HIP_CHECK(hipModuleLaunchKernel(f, grid.x, grid.y, grid.z, block.x, block.y, block.z, shmem, s, a.pack(), nullptr));
}

void TritonManifest::load(const std::string & dir) {
    std::ifstream f(dir + "/manifest.txt");
    if (!f) { fprintf(stderr, "cannot open %s/manifest.txt\n", dir.c_str()); exit(1); }
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ls(line); TritonLaunch L; int np;
        ls >> L.kernel >> L.file >> L.grid[0] >> L.grid[1] >> L.grid[2] >> L.block >> L.shared >> np;
        for (int i = 0; i < np; i++) {
            std::string tok; ls >> tok;
            const size_t c = tok.find(':'), e = tok.find('=');
            TritonLaunch::P p; p.name = tok.substr(0, c);
            p.kind = tok.substr(c + 1, e == std::string::npos ? std::string::npos : e - c - 1);
            if (e != std::string::npos) p.value = strtod(tok.c_str() + e + 1, nullptr);   // hex floats parse exactly
            L.params.push_back(p);
        }
        auto it = mods.find(L.file);
        if (it == mods.end()) {
            Module m; std::string path = dir + "/" + L.file;
            std::ifstream b(path, std::ios::binary);
            if (!b) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
            std::vector<char> img((std::istreambuf_iterator<char>(b)), {});
            HIP_CHECK(hipModuleLoadData(&m.mod, img.data()));
            it = mods.emplace(L.file, m).first;
        }
        L.fn = it->second.fn(L.kernel);
        launches.push_back(L);
    }
}

void TritonManifest::call(const TritonLaunch & L, const std::map<std::string, void *> & ptrs, hipStream_t s) {
    Args a;
    for (auto & p : L.params) {
        if (p.kind == "ptr") {
            auto it = ptrs.find(p.name);
            if (it == ptrs.end()) { fprintf(stderr, "%s: no pointer for %s\n", L.kernel.c_str(), p.name.c_str()); exit(1); }
            a.add(it->second);
        } else if (p.kind == "null") a.add((void *)nullptr);
        else if (p.kind == "i32") a.add((int32_t)p.value);
        else if (p.kind == "i64") a.add((int64_t)p.value);
        else if (p.kind == "fp32") a.add((float)p.value);
        else { fprintf(stderr, "%s: unsupported parameter kind %s\n", L.kernel.c_str(), p.kind.c_str()); exit(1); }
    }
    a.add((void *)nullptr).add((void *)nullptr);   // global and profile scratch
    launch(L.fn, dim3(L.grid[0], L.grid[1], L.grid[2]), dim3(L.block), a, s, L.shared);
}

void linear(rocblas_handle h, const Tensor & x, const Tensor & w, Tensor & y) {
    const int K = (int)x.dim(-1), M = (int)(x.numel() / K), N = (int)w.dim(0);
    if (w.dim(1) != K || y.numel() != (int64_t)M * N || x.dt != DType::bf16 || w.dt != DType::bf16 || y.dt != DType::bf16) {
        fprintf(stderr, "linear: bad shapes/dtypes x %s [..,%d] w [%lld,%lld] y %lld\n", dname(x.dt), K,
                (long long)w.dim(0), (long long)w.dim(1), (long long)y.numel()); exit(1);
    }
    const float one = 1.f, zero = 0.f;
    BLAS_CHECK(rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none, N, M, K, &one,
                               w.ptr, rocblas_datatype_bf16_r, K, x.ptr, rocblas_datatype_bf16_r, K, &zero,
                               y.ptr, rocblas_datatype_bf16_r, N, y.ptr, rocblas_datatype_bf16_r, N,
                               rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
}

static float bf2f(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }

std::vector<float> to_host_f32(const Tensor & t) {
    std::vector<float> out(t.numel());
    if (t.dt == DType::f32) {
        HIP_CHECK(hipMemcpy(out.data(), t.ptr, t.bytes(), hipMemcpyDeviceToHost));
    } else if (t.dt == DType::bf16) {
        std::vector<uint16_t> h(t.numel()); HIP_CHECK(hipMemcpy(h.data(), t.ptr, t.bytes(), hipMemcpyDeviceToHost));
        for (size_t i = 0; i < h.size(); i++) out[i] = bf2f(h[i]);
    } else { fprintf(stderr, "to_host_f32: unsupported dtype\n"); exit(1); }
    return out;
}

Diff compare(const Tensor & got, const Tensor & want) {
    Diff d;
    if (got.numel() != want.numel()) { fprintf(stderr, "compare: size %lld vs %lld\n", (long long)got.numel(), (long long)want.numel()); exit(1); }
    auto a = to_host_f32(got), b = to_host_f32(want);
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); i++) {
        const double e = (double)a[i] - b[i];
        if (memcmp(&a[i], &b[i], 4) != 0) d.bad++;
        d.max_abs = std::max(d.max_abs, std::fabs(e)); num += e * e; den += (double)b[i] * b[i];
    }
    d.n = a.size(); d.rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    return d;
}

}  // namespace rt
