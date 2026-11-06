// Milestone 4a, part 3: the S1 GatedDeltaNet blocks against the golden dump, from each block's golden input h,
// with fla's own Triton kernels replayed from build/triton (tools/capture_triton.py) between our ops:
//   q/k/v/g projections (actq + rocBLAS), a/b projections (bf16 cast + rocBLAS), cat -> causal conv (Triton) ->
//   contiguous q/k/v -> l2norm q, k -> beta sigmoid -> gate cumsum -> kkt solve -> recompute w/u -> state
//   recurrence -> chunk output (Triton) -> gated RMSNorm (Triton) -> o projection (actq + rocBLAS).
// Same binaries and inputs as PyTorch, so every checkpoint is expected bit-exact.
//   build/test_gdn GOLDEN_DIR
#include "rt.h"

using namespace rt;

static int fails = 0;
static void report(const char * what, const Diff & d, bool must_be_exact, double tol = 0) {
    const bool ok = must_be_exact ? d.exact() : d.rel <= tol;
    if (!ok) fails++;
    printf("  %-22s %s  bitwise mismatches %lld / %lld, rel err %.2e, max abs %.3g\n", what, ok ? "ok  " : "FAIL",
           (long long)d.bad, (long long)d.n, d.rel, d.max_abs);
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s GOLDEN_DIR\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    Arena arena; arena.init(size_t(7) << 30);
    TensorFile W, G; W.load(dir + "/weights", arena); G.load(dir + "/golden", arena);
    hipStream_t s; HIP_CHECK(hipStreamCreate(&s));
    rocblas_handle blas; BLAS_CHECK(rocblas_create_handle(&blas)); BLAS_CHECK(rocblas_set_stream(blas, s));
    Module actq, core; actq.load("actq"); core.load("core");
    TritonManifest tri; tri.load(root_dir() + "/build/triton");
    if (tri.launches.size() != 10) { fprintf(stderr, "expected 10 Triton launches, got %zu\n", tri.launches.size()); return 1; }
    const auto & L = tri.launches;
    const int T = (int)G["tokens"].numel(), D = 896, H = 7, HD = 128, QKV = 3 * D, NT = (T + 63) / 64;
    auto grid1 = [](int64_t n) { return (unsigned)((n + 255) / 256); };
    const size_t base = arena.used;

    for (int i : {0, 1, 2, 4, 5, 6, 8, 9, 10}) {
        arena.reset(base);
        const std::string p = "core.blocks." + std::to_string(i) + ".mixer.", b = "b" + std::to_string(i) + ".";
        printf("block %d (GatedDeltaNet)\n", i);
        const Tensor & h = G[b + "h"];
        // projections: the q / k / v / g QLinears quantize the same h (each call gives identical bits: once here)
        Tensor hq = arena.alloc(DType::bf16, {T, D});
        { Args a; a.add(h.ptr).add(hq.ptr).add((void *)nullptr).add(D).add(1.0f).add(7);
          launch(actq.fn("actq_fwd_f32"), T, 256, a, s); }
        Tensor q = arena.alloc(DType::bf16, {T, D}), k = arena.alloc(DType::bf16, {T, D}), v = arena.alloc(DType::bf16, {T, D}),
               gg = arena.alloc(DType::bf16, {T, D});
        linear(blas, hq, W[p + "q_proj.weight"], q); linear(blas, hq, W[p + "k_proj.weight"], k);
        linear(blas, hq, W[p + "v_proj.weight"], v); linear(blas, hq, W[p + "g_proj.weight"], gg);
        // a / b: plain Linear under autocast (h cast to bf16, bf16 weight)
        Tensor hb = arena.alloc(DType::bf16, {T, D}), av = arena.alloc(DType::bf16, {T, H}), bv = arena.alloc(DType::bf16, {T, H});
        { Args a; a.add(h.ptr).add(hb.ptr).add((int64_t)T * D); launch(core.fn("cast_bf16"), grid1((int64_t)T * D), 256, a, s); }
        linear(blas, hb, W[p + "a_proj.weight"], av); linear(blas, hb, W[p + "b_proj.weight"], bv);
        HIP_CHECK(hipStreamSynchronize(s));
        report("q_proj", compare(q, G[b + "gdn_q_proj_out"]), true);
        report("k_proj", compare(k, G[b + "gdn_k_proj_out"]), true);
        report("v_proj", compare(v, G[b + "gdn_v_proj_out"]), true);
        report("g_proj", compare(gg, G[b + "gdn_g_proj_out"]), true);
        report("a_proj", compare(av, G[b + "gdn_a_proj_out"]), true);
        report("b_proj", compare(bv, G[b + "gdn_b_proj_out"]), true);
        // cat(q, k, v) -> causal conv (SiLU) with the three conv kernels concatenated
        Tensor qkv = arena.alloc(DType::bf16, {T, QKV}), cy = arena.alloc(DType::bf16, {T, QKV}), cw = arena.alloc(DType::f32, {QKV, 4});
        const Tensor * parts[3] = {&q, &k, &v};
        const char * convs[3] = {"q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"};
        for (int j = 0; j < 3; j++) {
            Args a; a.add(parts[j]->ptr).add(D).add(0).add(qkv.ptr).add(QKV).add(j * D).add(D).add((int64_t)T);
            launch(core.fn("copy_cols_bf16"), grid1((int64_t)T * D), 256, a, s);
            HIP_CHECK(hipMemcpyAsync((char *)cw.ptr + (size_t)j * D * 4 * 4, W[p + convs[j]].ptr, (size_t)D * 4 * 4, hipMemcpyDeviceToDevice, s));
        }
        tri.call(L[0], {{"x", qkv.ptr}, {"y", cy.ptr}, {"weight", cw.ptr}}, s);
        // contiguous q / k / v out of the conv output, then the gated delta rule's own kernels
        Tensor qc = arena.alloc(DType::bf16, {T, D}), kc = arena.alloc(DType::bf16, {T, D}), vc = arena.alloc(DType::bf16, {T, D});
        Tensor * outs[3] = {&qc, &kc, &vc};
        for (int j = 0; j < 3; j++) {
            Args a; a.add(cy.ptr).add(QKV).add(j * D).add(outs[j]->ptr).add(D).add(0).add(D).add((int64_t)T);
            launch(core.fn("copy_cols_bf16"), grid1((int64_t)T * D), 256, a, s);
        }
        Tensor qn = arena.alloc(DType::bf16, {T, D}), kn = arena.alloc(DType::bf16, {T, D});
        Tensor rq = arena.alloc(DType::f32, {T * H}), rk = arena.alloc(DType::f32, {T * H});
        tri.call(L[1], {{"x", qc.ptr}, {"y", qn.ptr}, {"rstd", rq.ptr}}, s);
        tri.call(L[2], {{"x", kc.ptr}, {"y", kn.ptr}, {"rstd", rk.ptr}}, s);
        Tensor beta = arena.alloc(DType::f32, {T, H}), gcs = arena.alloc(DType::f32, {T, H});
        tri.call(L[3], {{"x", bv.ptr}, {"y", beta.ptr}}, s);
        tri.call(L[4], {{"g", av.ptr}, {"A_log", W[p + "A_log"].ptr}, {"dt_bias", W[p + "dt_bias"].ptr}, {"o", gcs.ptr}}, s);
        Tensor A = arena.alloc(DType::bf16, {T, H, 64});
        HIP_CHECK(hipMemsetAsync(A.ptr, 0, A.bytes(), s));
        tri.call(L[5], {{"k", kn.ptr}, {"g", gcs.ptr}, {"beta", beta.ptr}, {"A", A.ptr}}, s);
        Tensor w = arena.alloc(DType::bf16, {T, D}), u = arena.alloc(DType::bf16, {T, D});
        tri.call(L[6], {{"k", kn.ptr}, {"v", vc.ptr}, {"beta", beta.ptr}, {"w", w.ptr}, {"u", u.ptr}, {"A", A.ptr}, {"g", gcs.ptr}}, s);
        Tensor vnew = arena.alloc(DType::bf16, {T, D}), hs = arena.alloc(DType::bf16, {NT, H, HD, HD});
        tri.call(L[7], {{"k", kn.ptr}, {"v", u.ptr}, {"w", w.ptr}, {"v_new", vnew.ptr}, {"g", gcs.ptr}, {"h", hs.ptr}}, s);
        Tensor o = arena.alloc(DType::bf16, {T, D});
        tri.call(L[8], {{"q", qn.ptr}, {"k", kn.ptr}, {"v", vnew.ptr}, {"h", hs.ptr}, {"g", gcs.ptr}, {"o", o.ptr}}, s);
        HIP_CHECK(hipStreamSynchronize(s));
        report("delta rule output o", compare(o, G[b + "gdn_o_norm_in0"]), true);
        // gated RMSNorm on the golden o (isolates it), then the o projection on the golden norm output
        Tensor y = arena.alloc(DType::bf16, {T, D}), rs = arena.alloc(DType::f32, {T * H});
        tri.call(L[9], {{"x", G[b + "gdn_o_norm_in0"].ptr}, {"g", G[b + "gdn_o_norm_in1"].ptr}, {"y", y.ptr},
                        {"w", W[p + "o_norm.weight"].ptr}, {"rstd", rs.ptr}}, s);
        Tensor yq = arena.alloc(DType::bf16, {T, D}), m = arena.alloc(DType::bf16, {T, D});
        { Args a; a.add(G[b + "gdn_o_norm_out"].ptr).add(yq.ptr).add((void *)nullptr).add(D).add(1.0f).add(7);
          launch(actq.fn("actq_fwd_bf16"), T, 256, a, s); }
        linear(blas, yq, W[p + "o_proj.weight"], m);
        HIP_CHECK(hipStreamSynchronize(s));
        report("gated RMSNorm", compare(y, G[b + "gdn_o_norm_out"]), true);
        report("o_proj -> mixer", compare(m, G[b + "mixer"]), true);
    }
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
