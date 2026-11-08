// llama.cpp's quantized matrix products as references for the batched prefill's vendored MMQ (caminus.prefill.mmq,
// docs/specs/spec-prefill-cache.md S2 gate G1): runs ggml_mul_mat / ggml_mul_mat_id on the first ROCm device with the
// ignite-llama fork's libggml, on raw GGUF rows and fp32 activations from files, and writes the fp32 result.
//   build: tools/reference/build_gg_mmq_ref.sh   run: gg_mmq_ref JOBS
// JOBS: one job per line (whitespace separated; '#' lines ignored):
//   mm   FMT ROWS K NCOLS W X OUT [REPS]                     W: [ROWS][K] of FMT (raw rows), X: f32 [NCOLS][K],
//                                                            OUT: f32 [NCOLS][ROWS] = ggml_mul_mat(W, X)
//   mmid FMT NEXP ROWS K NTOK KUSED PERSLOT W X IDS OUT [REPS]
//                                                            W: [NEXP][ROWS][K], IDS: i32 [NTOK][KUSED], X: f32 [NTOK][K]
//                                                            (PERSLOT 0: one row per token, ggml's gate / up shape
//                                                            [K, 1, NTOK]) or [NTOK][KUSED][K] (PERSLOT 1: down),
//                                                            OUT: f32 [NTOK][KUSED][ROWS] = ggml_mul_mat_id(W, X, IDS)
// REPS > 0: the op is also run REPS more times and the median / min wall time of one graph compute (+ synchronize)
// printed: llama.cpp's own time for that op (its quantization of X and, for mmid, its expert grouping included).
// Which kernel ggml runs is ggml's choice: link against a fork build with GGML_CUDA_FORCE_MMQ=ON for MMQ at every T (the
// stock build-hip-therock runs dense Q6_K / IQ2_S / Q2_K through hipBLAS above 128 columns on RDNA3,
// mmq.cu ggml_cuda_should_use_mmq).
// Each job prints "JOB <n> ok <bytes> [ms median <m> min <m>]". The weights are ggml-allocated (padded and zeroed the
// way ggml pads every quantized tensor), so K need not be a multiple of 256.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<char> slurp(const std::string & p, size_t want) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(2); }
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (b.size() != want) { fprintf(stderr, "%s: %zu bytes, want %zu\n", p.c_str(), b.size(), want); exit(2); }
    return b;
}

static ggml_type type_of(std::string s) {
    for (auto & c : s) c = (char) tolower(c);
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * n = ggml_type_name((ggml_type) t);
        if (!n) continue;
        std::string l(n);
        for (auto & c : l) c = (char) tolower(c);
        if (s == l) return (ggml_type) t;
    }
    fprintf(stderr, "unknown type %s\n", s.c_str()); exit(2);
}

int main(int argc, char ** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s JOBS\n", argv[0]); return 2; }
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("ROCm0");
    if (!dev) { fprintf(stderr, "no ROCm0 device\n"); return 2; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    printf("device %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
    std::ifstream jf(argv[1]); std::string line; int n = 0;
    while (std::getline(jf, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line); std::string kind, fmt; ss >> kind >> fmt;
        const ggml_type type = type_of(fmt);
        long rows = 0, K = 0, ncols = 0, nexp = 1, ntok = 0, kused = 0, perslot = 0, reps = 0;
        std::string wf, xf, idf, of;
        if (kind == "mm") { ss >> rows >> K >> ncols >> wf >> xf >> of; }
        else if (kind == "mmid") { ss >> nexp >> rows >> K >> ntok >> kused >> perslot >> wf >> xf >> idf >> of; }
        else { fprintf(stderr, "bad job: %s\n", line.c_str()); return 2; }
        if (!(ss >> reps)) reps = 0;
        ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ci = ggml_init(ip);
        ggml_tensor * W = kind == "mm" ? ggml_new_tensor_2d(ci, type, K, rows) : ggml_new_tensor_3d(ci, type, K, rows, nexp);
        ggml_tensor * X = kind == "mm" ? ggml_new_tensor_2d(ci, GGML_TYPE_F32, K, ncols)
                                       : ggml_new_tensor_3d(ci, GGML_TYPE_F32, K, perslot ? kused : 1, ntok);
        ggml_tensor * ids = kind == "mm" ? nullptr : ggml_new_tensor_2d(ci, GGML_TYPE_I32, kused, ntok);
        ggml_backend_buffer_t bi = ggml_backend_alloc_ctx_tensors(ci, be);
        { auto b = slurp(wf, ggml_nbytes(W)); ggml_backend_tensor_set(W, b.data(), 0, b.size()); }
        { auto b = slurp(xf, ggml_nbytes(X)); ggml_backend_tensor_set(X, b.data(), 0, b.size()); }
        if (ids) { auto b = slurp(idf, ggml_nbytes(ids)); ggml_backend_tensor_set(ids, b.data(), 0, b.size()); }
        ggml_context * cg = ggml_init(ip);
        ggml_tensor * out = kind == "mm" ? ggml_mul_mat(cg, W, X) : ggml_mul_mat_id(cg, W, X, ids);
        ggml_cgraph * gf = ggml_new_graph(cg);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(ga, gf)) { fprintf(stderr, "graph allocation failed\n"); return 2; }
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 2; }
        ggml_backend_synchronize(be);
        std::vector<char> ob(ggml_nbytes(out));
        ggml_backend_tensor_get(out, ob.data(), 0, ob.size());
        { std::ofstream f(of, std::ios::binary); f.write(ob.data(), (std::streamsize) ob.size()); }
        printf("JOB %d ok %zu", n, ob.size());
        if (reps > 0) {
            std::vector<double> t;
            for (long r = 0; r < reps; r++) {
                auto t0 = std::chrono::steady_clock::now();
                ggml_backend_graph_compute(be, gf); ggml_backend_synchronize(be);
                t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            std::sort(t.begin(), t.end());
            printf(" ms median %.4f min %.4f", t[t.size() / 2], t[0]);
        }
        printf("\n"); fflush(stdout);
        ggml_gallocr_free(ga); ggml_free(cg); ggml_backend_buffer_free(bi); ggml_free(ci);
        n++;
    }
    ggml_backend_free(be);
    return 0;
}
