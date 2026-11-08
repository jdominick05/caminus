// Milestone 4a, part 4: the whole S1 core forward chained in the C++ runtime (embedding -> 12 blocks -> final norm)
// against the PyTorch golden hidden states. Not bit-exact end to end: RMSNorm (1 fp32 ulp) and the attention core
// (bf16 rounding vs AOTriton) differ and feed forward; this reports how that grows per block, the final hidden-state
// error, and the forward's time.
//   build/test_forward GOLDEN_DIR [HIDDEN_OUT.bin]
#include "s1.h"
#include <chrono>

using namespace rt;

static const TensorFile * Gp;
static void trace(int i, const Tensor & x) {
    const Diff d = compare(x, (*Gp)["b" + std::to_string(i) + ".x_out"]);
    printf("  block %2d x_out: rel err %.2e, max abs %.3g, bitwise mismatches %.1f%%\n", i, d.rel, d.max_abs, 100.0 * d.bad / d.n);
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s GOLDEN_DIR\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    Arena wa; wa.init(size_t(5) << 30);
    TensorFile W, G; W.load(dir + "/weights", wa); G.load(dir + "/golden", wa); Gp = &G;
    hipStream_t s; HIP_CHECK(hipStreamCreate(&s));
    rocblas_handle blas; BLAS_CHECK(rocblas_create_handle(&blas)); BLAS_CHECK(rocblas_set_stream(blas, s));
    S1Core m; m.init(W, blas, s);
    Arena act; act.init(size_t(2) << 30);
    printf("chained forward vs PyTorch, per block:\n");
    Tensor hid = m.hidden(G["tokens"], act, trace);
    HIP_CHECK(hipStreamSynchronize(s));
    const Diff d = compare(hid, G["hidden"]);
    printf("final hidden: rel err %.2e, max abs %.3g\n", d.rel, d.max_abs);
    if (argc > 2) {   // the hidden states for training/tools/score_hidden.py
        auto hh = to_host_f32(hid); FILE * f = fopen(argv[2], "wb"); fwrite(hh.data(), 4, hh.size(), f); fclose(f);
        printf("wrote %s\n", argv[2]);
    }
    // timing: the same forward again, no tracing
    for (int r = 0; r < 2; r++) { act.reset(); m.hidden(G["tokens"], act); }
    HIP_CHECK(hipStreamSynchronize(s));
    const int n = 10; auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < n; r++) { act.reset(); m.hidden(G["tokens"], act); }
    HIP_CHECK(hipStreamSynchronize(s));
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / n;
    printf("forward (1 x %lld tokens): %.2f ms = %.0f tok/s\n", (long long)G["tokens"].numel(), ms, G["tokens"].numel() / ms * 1e3);
    // The W4A4 core is chaotic: +-1 ulp noise on PyTorch's own RMSNorm outputs moves its final hidden states by
    // 0.20-0.21 relative (training/tools/score_hidden.py). So this checks the chain stays inside that envelope; whether the
    // forward predicts as well is scored by the loss (training/tools/score_hidden.py), the real acceptance test.
    const bool ok = d.rel < 0.25;
    printf(ok ? "PASS (inside the measured chaos envelope; score the loss with training/tools/score_hidden.py)\n" : "FAIL\n");
    return ok ? 0 : 1;
}
