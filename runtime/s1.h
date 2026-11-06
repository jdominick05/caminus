// The S1 core forward (frozen, null route) in the C++ runtime: embedding -> 12 parallel blocks
// (x + mixer(norm x) + mlp(norm x)) -> final norm. Attention blocks 3 / 7 / 11, GatedDeltaNet blocks elsewhere.
#pragma once
#include "rt.h"

namespace rt {

struct S1Core {
    static constexpr int D = 896, L = 12, GH = 7, GHD = 128, AH = 14, AKV = 2, AHD = 64, MLP = 2048;
    const TensorFile * W = nullptr;
    rocblas_handle blas = nullptr; hipStream_t s = nullptr;
    Module core, actq, attn; TritonManifest tri;
    void init(const TensorFile & weights, rocblas_handle b, hipStream_t st);
    // hidden states [T, D] fp32 for tokens [T] (int32); `trace(i, x_out)` is called after each block if set
    Tensor hidden(const Tensor & tokens, Arena & arena, void (*trace)(int, const Tensor &) = nullptr);

    // pieces (each from its inputs; scratch from `arena`)
    void rmsnorm(const Tensor & x, const Tensor & w, Tensor & out);
    void quant(const Tensor & x, Tensor & xq);                          // actq: fp32 or bf16 in, bf16 out
    void mlp(const std::string & p, const Tensor & h, Tensor & out, Arena & arena);
    void attention(const std::string & p, const Tensor & h, Tensor & out, Arena & arena);
    void gdn(const std::string & p, const Tensor & h, Tensor & out, Arena & arena);
};

}  // namespace rt
