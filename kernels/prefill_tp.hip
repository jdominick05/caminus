// The batched prefill's tensor-parallel glue (docs/agentic/spec-prefill-cache.md M5 S5; host side python/caminus/prefill_tp.py):
// the exchange's sum and the FFN glue between S2's quantized GEMMs, over a chunk of tokens with ordinary launches (not K3).
// Each kernel computes an element with the expressions of the K3 decode op it stands for (kernels/k3_gguf_ops.h), in its
// order, contraction off:
//   pf_xchg_add     OP_XCHG's sum: y = p0 + p1 with card 0's partial first on BOTH cards (a = card 0's, b = card 1's), so
//                   both replicas get the same bits. Exact in place (y = a or y = b): a thread reads its float4 before it
//                   writes it. The copies between the cards are the host's (hipMemcpyPeerAsync on the copy streams,
//                   caminus.prefill_tp.Exchange): no kernel here waits on the peer or spins
//   pf_swiglu       y = silu(g) * u with ggml's silu, x / (1 + expf(-x)) (OP_GG_MOE_GU / OP_GG_GU_DENSE's hv = gg_silu(g) * u;
//                   llama.cpp's swiglu): the routed experts' [T, k, ffc] and the shared expert's [T, ffc] down inputs
//   pf_moe_combine  the routed experts' weighted sum in slot order (OP_GG_MOE_DOWN: vw = v * w, acc = j == 0 ? vw : acc + vw,
//                   ggml_mul then a chain of ggml_add) plus the gated shared expert (OP_SHEXP_ADD: s = 1 / (1 + expf(-g)),
//                   out = moe + shexp * s), one card's FFN partial; y may be the shared expert's buffer (exact in place)
//   pf_attn_standin NOT attention (the attention core is S6 / S7): each token attends to itself only, so the output is its
//                   own v row (what attention gives at position 0), for every query head of the card's kv head, times
//                   sigmoid(the head's gate) as OP_GATE_MUL; it keeps the attention layers' data flow (q / k / v GEMMs ->
//                   attn_output GEMM -> exchange) so a measured chunk runs every GEMM on real inputs
// Compiled by build.sh's generic loop; the decode code objects are untouched.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef __attribute__((address_space(1))) const float4 GF4;
typedef __attribute__((address_space(1))) float4 GW4;

__device__ __forceinline__ float gg_silu(float x) { return x / (1.0f + expf(-x)); }   // ggml-cuda op_silu

// ---- pf_xchg_add: y = a + b (a = card 0's partial), n4 float4s -------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) pf_xchg_add(const float * a, const float * b, float * y, long n4) {
#pragma clang fp contract(off)
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n4) return;
    const float4 x = ((GF4 *)a)[i], r = ((GF4 *)b)[i];
    ((GW4 *)y)[i] = make_float4(x.x + r.x, x.y + r.y, x.z + r.z, x.w + r.w);
}

// ---- pf_swiglu: y = silu(g) * u, n elements --------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) pf_swiglu(const float * g, const float * u, float * y, long n) {
#pragma clang fp contract(off)
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const float hv = gg_silu(g[i]) * u[i];
    y[i] = hv;
}

// ---- pf_moe_combine: y[t, i] = (sum_j down[t, j, i] * w[t, j]) + shx[t, i] * sigmoid(sgl[t]) ---------------------------
// down [T, k, D] (mul_mat_id's per-slot outputs), w [T, k] (the router's weights), shx [T, D] (the shared expert's down
// output), sgl [T] (its gate logit), y [T, D] (may be shx: each thread reads its element of shx before it writes y there)
extern "C" __global__ void __launch_bounds__(256) pf_moe_combine(const float * down, const float * w, const float * shx,
                                                                 const float * sgl, float * y, int T, int k, int D) {
#pragma clang fp contract(off)
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long)T * D) return;
    const int tt = (int)(i / D), ii = (int)(i - (long)tt * D);
    float acc = 0.0f;
    for (int j = 0; j < k; j++) {
        const float vw = down[((size_t)tt * k + j) * D + ii] * w[tt * k + j];
        acc = j == 0 ? vw : acc + vw;
    }
    const float s = 1.0f / (1.0f + expf(-sgl[tt]));
    const float g = shx[i] * s;
    y[i] = acc + g;
}

// ---- pf_attn_standin: y[t, h, d] = v[t, d] * sigmoid(qf[t, h, hd + d]) (NOT attention: see the header) -----------------
// v [T, hd] (the card's kv head), qf [T, nh, 2 hd] (attn_q's rows per head: q, then the gate), y [T, nh, hd]
extern "C" __global__ void __launch_bounds__(256) pf_attn_standin(const float * v, const float * qf, float * y, int T, int nh, int hd) {
#pragma clang fp contract(off)
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long)T * nh * hd) return;
    const int tt = (int)(i / ((long)nh * hd)), r = (int)(i - (long)tt * nh * hd), h = r / hd, d = r - h * hd;
    const float sgm = 1.0f / (1.0f + expf(-qf[((size_t)tt * nh + h) * 2 * hd + hd + d]));
    const float o = v[(size_t)tt * hd + d] * sgm;
    y[i] = o;
}
