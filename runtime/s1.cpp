#include "s1.h"

namespace rt {

static unsigned g256(int64_t n) { return (unsigned)((n + 255) / 256); }

void S1Core::init(const TensorFile & weights, rocblas_handle b, hipStream_t st) {
    W = &weights; blas = b; s = st;
    core.load("core"); actq.load("actq"); attn.load("attn");
    tri.load(root_dir() + "/build/triton");
    if (tri.launches.size() != 10) { fprintf(stderr, "S1Core: expected 10 Triton launches, got %zu\n", tri.launches.size()); exit(1); }
}

void S1Core::rmsnorm(const Tensor & x, const Tensor & w, Tensor & out) {
    const int T = (int)(x.numel() / D);
    Args a; a.add(x.ptr).add(w.ptr).add(out.ptr).add(D).add(1e-6f);
    launch(core.fn("rmsnorm_f32"), T, 256, a, s);
}

void S1Core::quant(const Tensor & x, Tensor & xq) {
    const int K = (int)x.dim(-1), M = (int)(x.numel() / K);
    Args a; a.add(x.ptr).add(xq.ptr).add((void *)nullptr).add(K).add(1.0f).add(7);
    launch(actq.fn(x.dt == DType::f32 ? "actq_fwd_f32" : "actq_fwd_bf16"), M, 256, a, s);
}

void S1Core::mlp(const std::string & p, const Tensor & h, Tensor & out, Arena & ar) {
    const int T = (int)(h.numel() / D);
    Tensor hq = ar.alloc(DType::bf16, {T, D}), gu = ar.alloc(DType::bf16, {T, 2 * MLP});
    Tensor act = ar.alloc(DType::bf16, {T, MLP}), aq = ar.alloc(DType::bf16, {T, MLP});
    quant(h, hq); linear(blas, hq, (*W)[p + "mlp.gu.weight"], gu);
    { const int64_t n = (int64_t)T * MLP; Args a; a.add(gu.ptr).add(act.ptr).add(MLP).add(n); launch(core.fn("silu_mul_bf16"), g256(n), 256, a, s); }
    quant(act, aq); linear(blas, aq, (*W)[p + "mlp.down.weight"], out);
}

void S1Core::attention(const std::string & p, const Tensor & h, Tensor & out, Arena & ar) {
    const int T = (int)(h.numel() / D);
    Tensor hq = ar.alloc(DType::bf16, {T, D}), q = ar.alloc(DType::bf16, {T, D}), kv = ar.alloc(DType::bf16, {T, 2 * AKV * AHD});
    quant(h, hq); linear(blas, hq, (*W)[p + "q.weight"], q); linear(blas, hq, (*W)[p + "kv.weight"], kv);
    Tensor qr = ar.alloc(DType::bf16, {AH, T, AHD}), kr = ar.alloc(DType::bf16, {AKV, T, AHD}), v = ar.alloc(DType::bf16, {AKV, T, AHD});
    { Args a; a.add(q.ptr).add(kv.ptr).add(qr.ptr).add(kr.ptr).add(v.ptr).add(T).add(AH).add(AKV).add(AHD).add(10000.0f);
      launch(attn.fn("rope_split_bf16"), T, 256, a, s); }
    Tensor o = ar.alloc(DType::bf16, {T, D}), oq = ar.alloc(DType::bf16, {T, D});
    { Args a; a.add(qr.ptr).add(kr.ptr).add(v.ptr).add(o.ptr).add(T).add(AH).add(AKV).add(AHD).add(0.125f);
      launch(attn.fn("attn_causal_bf16"), AH * T, 256, a, s); }
    quant(o, oq); linear(blas, oq, (*W)[p + "o.weight"], out);
}

void S1Core::gdn(const std::string & p, const Tensor & h, Tensor & out, Arena & ar) {
    const int T = (int)(h.numel() / D), QKV = 3 * D, NT = (T + 63) / 64;
    const auto & Lc = tri.launches;
    Tensor hq = ar.alloc(DType::bf16, {T, D});
    quant(h, hq);
    Tensor qkv = ar.alloc(DType::bf16, {T, QKV}), gg = ar.alloc(DType::bf16, {T, D}), tmp = ar.alloc(DType::bf16, {T, D});
    const char * projs[3] = {"q_proj.weight", "k_proj.weight", "v_proj.weight"};
    const char * convs[3] = {"q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"};
    Tensor cw = ar.alloc(DType::f32, {QKV, 4});
    for (int j = 0; j < 3; j++) {   // torch.cat of the three projections, and of the three conv kernels
        linear(blas, hq, (*W)[p + projs[j]], tmp);
        Args a; a.add(tmp.ptr).add(D).add(0).add(qkv.ptr).add(QKV).add(j * D).add(D).add((int64_t)T);
        launch(core.fn("copy_cols_bf16"), g256((int64_t)T * D), 256, a, s);
        HIP_CHECK(hipMemcpyAsync((char *)cw.ptr + (size_t)j * D * 16, (*W)[p + convs[j]].ptr, (size_t)D * 16, hipMemcpyDeviceToDevice, s));
    }
    linear(blas, hq, (*W)[p + "g_proj.weight"], gg);
    Tensor hb = ar.alloc(DType::bf16, {T, D}), av = ar.alloc(DType::bf16, {T, GH}), bv = ar.alloc(DType::bf16, {T, GH});
    { Args a; a.add(h.ptr).add(hb.ptr).add((int64_t)T * D); launch(core.fn("cast_bf16"), g256((int64_t)T * D), 256, a, s); }
    linear(blas, hb, (*W)[p + "a_proj.weight"], av); linear(blas, hb, (*W)[p + "b_proj.weight"], bv);
    Tensor cy = ar.alloc(DType::bf16, {T, QKV});
    tri.call(Lc[0], {{"x", qkv.ptr}, {"y", cy.ptr}, {"weight", cw.ptr}}, s);
    Tensor qc = ar.alloc(DType::bf16, {T, D}), kc = ar.alloc(DType::bf16, {T, D}), vc = ar.alloc(DType::bf16, {T, D});
    Tensor * cs[3] = {&qc, &kc, &vc};
    for (int j = 0; j < 3; j++) {
        Args a; a.add(cy.ptr).add(QKV).add(j * D).add(cs[j]->ptr).add(D).add(0).add(D).add((int64_t)T);
        launch(core.fn("copy_cols_bf16"), g256((int64_t)T * D), 256, a, s);
    }
    Tensor qn = ar.alloc(DType::bf16, {T, D}), kn = ar.alloc(DType::bf16, {T, D});
    Tensor rq = ar.alloc(DType::f32, {T * GH}), rk = ar.alloc(DType::f32, {T * GH});
    tri.call(Lc[1], {{"x", qc.ptr}, {"y", qn.ptr}, {"rstd", rq.ptr}}, s);
    tri.call(Lc[2], {{"x", kc.ptr}, {"y", kn.ptr}, {"rstd", rk.ptr}}, s);
    Tensor beta = ar.alloc(DType::f32, {T, GH}), gcs = ar.alloc(DType::f32, {T, GH}), A = ar.alloc(DType::bf16, {T, GH, 64});
    tri.call(Lc[3], {{"x", bv.ptr}, {"y", beta.ptr}}, s);
    tri.call(Lc[4], {{"g", av.ptr}, {"A_log", (*W)[p + "A_log"].ptr}, {"dt_bias", (*W)[p + "dt_bias"].ptr}, {"o", gcs.ptr}}, s);
    HIP_CHECK(hipMemsetAsync(A.ptr, 0, A.bytes(), s));
    tri.call(Lc[5], {{"k", kn.ptr}, {"g", gcs.ptr}, {"beta", beta.ptr}, {"A", A.ptr}}, s);
    Tensor w = ar.alloc(DType::bf16, {T, D}), u = ar.alloc(DType::bf16, {T, D});
    tri.call(Lc[6], {{"k", kn.ptr}, {"v", vc.ptr}, {"beta", beta.ptr}, {"w", w.ptr}, {"u", u.ptr}, {"A", A.ptr}, {"g", gcs.ptr}}, s);
    Tensor vnew = ar.alloc(DType::bf16, {T, D}), hs = ar.alloc(DType::bf16, {NT, GH, GHD, GHD}), o = ar.alloc(DType::bf16, {T, D});
    tri.call(Lc[7], {{"k", kn.ptr}, {"v", u.ptr}, {"w", w.ptr}, {"v_new", vnew.ptr}, {"g", gcs.ptr}, {"h", hs.ptr}}, s);
    tri.call(Lc[8], {{"q", qn.ptr}, {"k", kn.ptr}, {"v", vnew.ptr}, {"h", hs.ptr}, {"g", gcs.ptr}, {"o", o.ptr}}, s);
    Tensor y = ar.alloc(DType::bf16, {T, D}), rs = ar.alloc(DType::f32, {T * GH}), yq = ar.alloc(DType::bf16, {T, D});
    tri.call(Lc[9], {{"x", o.ptr}, {"g", gg.ptr}, {"y", y.ptr}, {"w", (*W)[p + "o_norm.weight"].ptr}, {"rstd", rs.ptr}}, s);
    quant(y, yq); linear(blas, yq, (*W)[p + "o_proj.weight"], out);
}

Tensor S1Core::hidden(const Tensor & tokens, Arena & ar, void (*trace)(int, const Tensor &)) {
    const int T = (int)tokens.numel();
    Tensor x = ar.alloc(DType::f32, {T, D}), xn = ar.alloc(DType::f32, {T, D}), hid = ar.alloc(DType::f32, {T, D});
    { Args a; a.add(tokens.ptr).add((*W)["core.embed.weight"].ptr).add(x.ptr).add(D); launch(core.fn("embedding_f32"), T, 256, a, s); }
    const size_t mark = ar.used;
    for (int i = 0; i < L; i++) {
        ar.reset(mark);
        const std::string p = "core.blocks." + std::to_string(i) + ".";
        Tensor h = ar.alloc(DType::f32, {T, D}), m = ar.alloc(DType::bf16, {T, D}), y = ar.alloc(DType::bf16, {T, D});
        rmsnorm(x, (*W)[p + "norm.w"], h);
        if ((i + 1) % 4 == 0) attention(p + "mixer.", h, m, ar); else gdn(p + "mixer.", h, m, ar);
        mlp(p, h, y, ar);
        { const int64_t n = (int64_t)T * D; Args a; a.add(x.ptr).add(m.ptr).add(y.ptr).add(xn.ptr).add(n); launch(core.fn("residual3"), g256(n), 256, a, s); }
        std::swap(x.ptr, xn.ptr);
        if (trace) { HIP_CHECK(hipStreamSynchronize(s)); trace(i, x); }
    }
    ar.reset(mark);
    rmsnorm(x, (*W)["core.norm.w"], hid);
    return hid;
}

}  // namespace rt
