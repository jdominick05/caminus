// Milestone 4a, part 2: the S1 attention blocks (3, 7, 11) against the golden dump, each piece from its golden input:
// q / kv projections (actq + rocBLAS, bit-exact), RoPE + head split (bit-exact), causal GQA attention (our fp32
// kernel vs AOTriton's flash kernel: bf16 rounding), output projection (bit-exact).
//   build/test_attn GOLDEN_DIR
#include "rt.h"

using namespace rt;

static int fails = 0;
static void report(const char * what, const Diff & d, bool must_be_exact, double tol = 0) {
    const bool ok = must_be_exact ? d.exact() : d.rel <= tol;
    if (!ok) fails++;
    printf("  %-24s %s  bitwise mismatches %lld / %lld, rel err %.2e, max abs %.3g\n", what, ok ? "ok  " : "FAIL",
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
    Module actq, attn; actq.load("actq"); attn.load("attn");
    const int T = (int)G["tokens"].numel(), D = 896, H = 14, KV = 2, HD = 64;
    for (int i : {3, 7, 11}) {
        arena.reset(base);
        const std::string p = "core.blocks." + std::to_string(i) + ".mixer.", b = "b" + std::to_string(i) + ".";
        printf("block %d (attention)\n", i);
        Tensor hq = arena.alloc(DType::bf16, {T, D}), q = arena.alloc(DType::bf16, {T, D}), kv = arena.alloc(DType::bf16, {T, 2 * KV * HD});
        { Args a; a.add(G[b + "h"].ptr).add(hq.ptr).add((void *)nullptr).add(D).add(1.0f).add(7);
          launch(actq.fn("actq_fwd_f32"), T, 256, a, s); }
        linear(blas, hq, W[p + "q.weight"], q); linear(blas, hq, W[p + "kv.weight"], kv);
        HIP_CHECK(hipStreamSynchronize(s));
        report("q projection", compare(q, G[b + "attn_q"]), true);
        report("kv projection", compare(kv, G[b + "attn_kv"]), true);
        Tensor qr = arena.alloc(DType::bf16, {H, T, HD}), kr = arena.alloc(DType::bf16, {KV, T, HD}), v = arena.alloc(DType::bf16, {KV, T, HD});
        { Args a; a.add(G[b + "attn_q"].ptr).add(G[b + "attn_kv"].ptr).add(qr.ptr).add(kr.ptr).add(v.ptr)
                    .add(T).add(H).add(KV).add(HD).add(10000.0f);
          launch(attn.fn("rope_split_bf16"), T, 256, a, s); }
        HIP_CHECK(hipStreamSynchronize(s));
        report("rope q", compare(qr, G[b + "attn_q_rope"]), true);
        report("rope k", compare(kr, G[b + "attn_k_rope"]), true);
        report("v split", compare(v, G[b + "attn_v"]), true);
        Tensor o = arena.alloc(DType::bf16, {T, D});
        { Args a; a.add(G[b + "attn_q_rope"].ptr).add(G[b + "attn_k_rope"].ptr).add(G[b + "attn_v"].ptr).add(o.ptr)
                    .add(T).add(H).add(KV).add(HD).add(0.125f);
          launch(attn.fn("attn_causal_bf16"), H * T, 256, a, s); }
        HIP_CHECK(hipStreamSynchronize(s));
        report("attention (vs AOTriton)", compare(o, G[b + "attn_o"]), false, 1e-2);
        Tensor oq = arena.alloc(DType::bf16, {T, D}), m = arena.alloc(DType::bf16, {T, D});
        { Args a; a.add(G[b + "attn_o"].ptr).add(oq.ptr).add((void *)nullptr).add(D).add(1.0f).add(7);
          launch(actq.fn("actq_fwd_bf16"), T, 256, a, s); }
        linear(blas, oq, W[p + "o.weight"], m);
        HIP_CHECK(hipStreamSynchronize(s));
        report("o projection", compare(m, G[b + "mixer"]), true);
    }
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
