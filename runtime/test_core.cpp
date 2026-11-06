// Milestone 4a, part 1: embedding, RMSNorm, the MLP path and the residual add against the PyTorch golden dump,
// each block from its own golden inputs (a mismatch cannot spread into later blocks).
//   build/test_core GOLDEN_DIR      (GOLDEN_DIR from tools/dump_golden.py)
// Expected: embedding, MLP (actq -> rocBLAS -> silu*u -> actq -> rocBLAS) and residual bit-exact; RMSNorm within
// fp32 rounding (torch's fused kernel sums in its own order).
#include "rt.h"

using namespace rt;

static int fails = 0;
static void report(const char * what, const Diff & d, bool must_be_exact, double tol = 0) {
    const bool ok = must_be_exact ? d.exact() : d.rel <= tol;
    if (!ok) fails++;
    printf("  %-28s %s  bitwise mismatches %lld / %lld, rel err %.2e, max abs %.3g\n", what, ok ? "ok  " : "FAIL",
           (long long)d.bad, (long long)d.n, d.rel, d.max_abs);
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s GOLDEN_DIR\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    Arena arena; arena.init(size_t(6) << 30);
    TensorFile W, G; W.load(dir + "/weights", arena); G.load(dir + "/golden", arena);
    const size_t base = arena.used;
    hipStream_t s; HIP_CHECK(hipStreamCreate(&s));
    rocblas_handle blas; BLAS_CHECK(rocblas_create_handle(&blas)); BLAS_CHECK(rocblas_set_stream(blas, s));
    char ver[64]; rocblas_get_version_string(ver, sizeof(ver)); printf("rocBLAS %s\n", ver);
    Module core, actq; core.load("core"); actq.load("actq");

    const Tensor & tok = G["tokens"]; const int T = (int)tok.numel(), D = 896;
    {   // embedding
        Tensor e = arena.alloc(DType::f32, {T, D});
        Args a; a.add(tok.ptr).add(W["core.embed.weight"].ptr).add(e.ptr).add(D);
        launch(core.fn("embedding_f32"), T, 256, a, s);
        HIP_CHECK(hipStreamSynchronize(s));
        printf("embedding\n"); report("b0.x_in", compare(e, G["b0.x_in"]), true);
    }
    for (int i = 0; i < 12; i++) {
        arena.reset(base);
        const std::string p = "core.blocks." + std::to_string(i) + ".", b = "b" + std::to_string(i) + ".";
        printf("block %d\n", i);
        Tensor h = arena.alloc(DType::f32, {T, D});
        { Args a; a.add(G[b + "x_in"].ptr).add(W[p + "norm.w"].ptr).add(h.ptr).add(D).add(1e-6f);
          launch(core.fn("rmsnorm_f32"), T, 256, a, s); }
        HIP_CHECK(hipStreamSynchronize(s));
        report("rmsnorm -> h", compare(h, G[b + "h"]), false, 1e-6);
        // MLP from the golden h: gu = linear(actq(h)), act = silu(g) * u, mlp = linear(actq(act))
        const Tensor & hg = G[b + "h"];
        Tensor hq = arena.alloc(DType::bf16, {T, D}), gu = arena.alloc(DType::bf16, {T, 4096});
        Tensor act = arena.alloc(DType::bf16, {T, 2048}), aq = arena.alloc(DType::bf16, {T, 2048});
        Tensor y = arena.alloc(DType::bf16, {T, D}), xo = arena.alloc(DType::f32, {T, D});
        { Args a; a.add(hg.ptr).add(hq.ptr).add((void *)nullptr).add(D).add(1.0f).add(7);
          launch(actq.fn("actq_fwd_f32"), T, 256, a, s); }
        linear(blas, hq, W[p + "mlp.gu.weight"], gu);
        HIP_CHECK(hipStreamSynchronize(s));
        report("mlp gu", compare(gu, G[b + "mlp_gu"]), true);
        { const int64_t n = (int64_t)T * 2048; Args a; a.add(G[b + "mlp_gu"].ptr).add(act.ptr).add(2048).add(n);
          launch(core.fn("silu_mul_bf16"), (unsigned)((n + 255) / 256), 256, a, s); }
        HIP_CHECK(hipStreamSynchronize(s));
        report("silu(g) * u", compare(act, G[b + "mlp_act"]), true);
        { Args a; a.add(G[b + "mlp_act"].ptr).add(aq.ptr).add((void *)nullptr).add(2048).add(1.0f).add(7);
          launch(actq.fn("actq_fwd_bf16"), T, 256, a, s); }
        linear(blas, aq, W[p + "mlp.down.weight"], y);
        HIP_CHECK(hipStreamSynchronize(s));
        report("mlp down", compare(y, G[b + "mlp"]), true);
        { const int64_t n = (int64_t)T * D; Args a; a.add(G[b + "x_in"].ptr).add(G[b + "mixer"].ptr).add(G[b + "mlp"].ptr).add(xo.ptr).add(n);
          launch(core.fn("residual3"), (unsigned)((n + 255) / 256), 256, a, s); }
        HIP_CHECK(hipStreamSynchronize(s));
        report("residual -> x_out", compare(xo, G[b + "x_out"]), true);
    }
    {   // final norm
        arena.reset(base);
        Tensor hid = arena.alloc(DType::f32, {T, D});
        Args a; a.add(G["b11.x_out"].ptr).add(W["core.norm.w"].ptr).add(hid.ptr).add(D).add(1e-6f);
        launch(core.fn("rmsnorm_f32"), T, 256, a, s);
        HIP_CHECK(hipStreamSynchronize(s));
        printf("final\n"); report("rmsnorm -> hidden", compare(hid, G["hidden"]), false, 1e-6);
    }
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
