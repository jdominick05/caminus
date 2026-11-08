// llama.cpp's mul_mat_vec_f (ignite-llama ggml/src/ggml-cuda/mmvf.cu, MIT), compiled from the ignite tree as it is, with
// ignite's own HIP flags (build.sh: -O3 -std=gnu++17, no max-ilp scheduler): the reference that OP_BF16_MV
// (kernels/k3_gguf_ops.h) must match bitwise (docs/specs/spec-gsq-port.md S2, gate G2b). The kernels keep their
// mangled names (_ZL13mul_mat_vec_f..., instantiated by mmvf.cu's own launch code); tests/test_k3_bf16_mv.py launches
// them directly and also runs the kernels of ignite's shipped object (build-hip-therock mmvf.cu.o).
#include "mmvf.cu"
