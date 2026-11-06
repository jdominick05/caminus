// This file and python/caminus/gdn.py implement flash-linear-attention's (fla) chunk_gated_delta_rule_fwd_h and _bwd_dhu
// (fla/ops/common/chunk_delta_h.py, https://github.com/fla-org/flash-linear-attention) for HIP, following its algorithm and
// operand roundings; fla is distributed under the following license (THIRD_PARTY.md):
//
// MIT License
//
// Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Gated DeltaNet chunk state recurrences, fla's chunk_gated_delta_rule_fwd_h and _bwd_dhu (ops/common/chunk_delta_h.py),
// for scalar gates, K = 128, chunk 64, equal-length sequences, H == HV; state layout [K, V] or, with vfirst, [V, K].
//
// gdn_fwd_h, per (batch b, head h, value slice), chunks first to last, state S [K, BV] fp32 (from h0, or 0):
//   store bf16(S) as h[b, i_t, h]
//   vn_t = u_t - w_t @ bf16(S)                                  stored as bf16 (v_new), before gating
//   S    = S * 2^g_last + k_t^T @ bf16(vn_t * 2^(g_last - g_t))  (rows past T are 0)
// and finally ht = S (fp32) when requested.
//
// gdn_bwd_dhu:
//
// Per (batch b, head h, value slice of BV columns), walking chunks from last to first, with the state gradient
// dH [K, BV] in fp32 (initialised from dht, or 0):
//   store bf16(dH) as dh[b, i_t, h]
//   dv2_t = (k_t @ bf16(dH)) * 2^(g_last - g_t) + dv_t           (rows past T are 0); stored as bf16
//   dH   = dH * 2^g_last + scale * (q_t * 2^g_t)^T @ do_t - w_t^T @ bf16(dv2_t)
// and finally dh0 = dH (fp32) when requested. g is the chunk-local cumulative log2 gate (fp32), as fla passes it.
// Operand roundings follow fla's Triton kernel: dH and dv2 enter their products as bf16; the q*2^g product with do
// runs in fp32 (that kernel multiplies q by an fp32 gate before its dot). Products here are fp32 FMAs over bf16
// operands, so only the summation order differs.
//
// One 128-thread workgroup per (b, h, slice): thread k owns dH[k, 0..BV). Each chunk's k, q, w rows (64 x 128 bf16)
// and do, dv slices are staged in LDS. fla launches (V / 32) x B x H tiny programs that are latency-bound on gfx1100;
// this kernel launches (V / 16) x B x H workgroups of four waves.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define K 128
#define BT 64
#define BV 16
#define NTH 128

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }

__device__ __forceinline__ uint16_t f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }

// index of state element (k, v) of head h in batch b: [b, h, K, V], or [b, h, V, K] when vfirst (fla's
// state_v_first, which its GatedDeltaNet layer uses)
__device__ __forceinline__ size_t sidx(int b, int h, int k, int v, int H, int V, int vfirst) {
    const size_t base = ((size_t)b * H + h) * (size_t)K * V;
    return base + (vfirst ? (size_t)v * K + k : (size_t)k * V + v);
}

extern "C" __global__ void __launch_bounds__(NTH)
gdn_bwd_dhu(const uint16_t * __restrict__ q, const uint16_t * __restrict__ k, const uint16_t * __restrict__ w,
            const float * __restrict__ g, const uint16_t * __restrict__ dout, const uint16_t * __restrict__ dv,
            const float * __restrict__ dht, uint16_t * __restrict__ dh, uint16_t * __restrict__ dv2,
            float * __restrict__ dh0, int T, int H, int V, float scale, int vfirst) {
    __shared__ uint16_t sk[BT][K + 2], sq[BT][K + 2], sw[BT][K + 2];   // bf16 as loaded; +2 avoids bank conflicts
    __shared__ __attribute__((aligned(16))) float sdo[BT][BV], sdv[BT][BV];
    __shared__ float sg[BT], sge[BT];                                  // sge = 2^g_t
    __shared__ __attribute__((aligned(16))) uint16_t sdh[K][BV];       // bf16(dH), for k_t @ dH
    const int nvs = V / BV;
    const int vs = blockIdx.x % nvs, h = (blockIdx.x / nvs) % H, b = blockIdx.x / (nvs * H);
    const int tid = threadIdx.x, v0 = vs * BV, NT = (T + BT - 1) / BT;

    float st[BV];
#pragma unroll
    for (int j = 0; j < BV; j++) st[j] = dht ? dht[sidx(b, h, tid, v0 + j, H, V, vfirst)] : 0.f;

    for (int it = NT - 1; it >= 0; it--) {
        const int t0 = it * BT, tl = min(BT, T - t0);
        // stage the chunk: rows past T are zero
        for (int e = tid; e < BT * K / 8; e += NTH) {          // 16-byte loads: 8 bf16 per thread per tensor
            const int r = e / (K / 8), c = (e % (K / 8)) * 8;
            const size_t qi = (((size_t)b * T + t0 + r) * H + h) * K + c;
            const bool ok = r < tl;
            const uint4 z = make_uint4(0, 0, 0, 0);
            const uint4 kv = ok ? *(const uint4 *)(k + qi) : z, qv = ok ? *(const uint4 *)(q + qi) : z,
                        wv = ok ? *(const uint4 *)(w + qi) : z;
            // LDS rows are K + 2 long (4-byte aligned only): store as 32-bit words
            uint32_t * dk = (uint32_t *)&sk[r][c], * dq = (uint32_t *)&sq[r][c], * dw = (uint32_t *)&sw[r][c];
            dk[0] = kv.x; dk[1] = kv.y; dk[2] = kv.z; dk[3] = kv.w;
            dq[0] = qv.x; dq[1] = qv.y; dq[2] = qv.z; dq[3] = qv.w;
            dw[0] = wv.x; dw[1] = wv.y; dw[2] = wv.z; dw[3] = wv.w;
        }
        for (int e = tid; e < BT * BV; e += NTH) {
            const int r = e / BV, c = e % BV;
            const size_t vi = (((size_t)b * T + t0 + r) * H + h) * V + v0 + c;
            const bool ok = r < tl;
            sdo[r][c] = ok ? bf2f(dout[vi]) : 0.f;
            sdv[r][c] = ok ? bf2f(dv[vi]) : 0.f;
        }
        if (tid < BT) { const float gv = tid < tl ? g[((size_t)b * T + t0 + tid) * H + h] : 0.f; sg[tid] = gv; sge[tid] = exp2f(gv); }
        // store dH for this chunk and keep its bf16 copy for k_t @ dH
        uint16_t * dho = dh + ((size_t)b * NT + it) * H * K * V;
#pragma unroll
        for (int j = 0; j < BV; j++) { const uint16_t hb = f2bf(st[j]); dho[sidx(0, h, tid, v0 + j, H, V, vfirst)] = hb; sdh[tid][j] = hb; }
        __syncthreads();

        const float glast = sg[tl - 1];
        // dv2 = (k_t @ dH) * 2^(glast - g_t) + dv_t: thread -> (row r = tid / 2, 8 columns)
        {
            const int r = tid / 2, c0 = (tid % 2) * (BV / 2);
            float acc[BV / 2] = {};
            for (int c = 0; c < K; c++) {
                const float kv = bf2f(sk[r][c]);
                const uint4 hv = *(const uint4 *)&sdh[c][c0];
                const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
#pragma unroll
                for (int j = 0; j < BV / 2; j++) acc[j] += kv * bf2f((uint16_t)(hw[j / 2] >> (16 * (j % 2))));
            }
            const float f = r < tl ? exp2f(glast - sg[r]) : 0.f;
            uint16_t * o = dv2 + (((size_t)b * T + t0 + r) * H + h) * V + v0 + c0;
#pragma unroll
            for (int j = 0; j < BV / 2; j++) {
                const uint16_t vb = f2bf(acc[j] * f + sdv[r][c0 + j]);
                if (r < tl) o[j] = vb;
                sdv[r][c0 + j] = bf2f(vb);           // dv2 in bf16, the operand of w^T @ dv2
            }
        }
        __syncthreads();
        // dH = dH * 2^glast + scale * (q_t * 2^g_t)^T @ do_t - w_t^T @ dv2_t: thread k owns row k
        {
            float aq[BV] = {}, aw[BV] = {};
            for (int r = 0; r < BT; r++) {
                const float qg = bf2f(sq[r][tid]) * sge[r], wv = bf2f(sw[r][tid]);
#pragma unroll
                for (int j = 0; j < BV; j += 4) {
                    const float4 d = *(const float4 *)&sdo[r][j], v = *(const float4 *)&sdv[r][j];
                    aq[j] += qg * d.x; aq[j + 1] += qg * d.y; aq[j + 2] += qg * d.z; aq[j + 3] += qg * d.w;
                    aw[j] += wv * v.x; aw[j + 1] += wv * v.y; aw[j + 2] += wv * v.z; aw[j + 3] += wv * v.w;
                }
            }
            const float decay = exp2f(glast);
#pragma unroll
            for (int j = 0; j < BV; j++) st[j] = st[j] * decay + (aq[j] * scale - aw[j]);
        }
        __syncthreads();
    }
    if (dh0) {
#pragma unroll
        for (int j = 0; j < BV; j++) dh0[sidx(b, h, tid, v0 + j, H, V, vfirst)] = st[j];
    }
}

extern "C" __global__ void __launch_bounds__(NTH)
gdn_fwd_h(const uint16_t * __restrict__ k, const uint16_t * __restrict__ w, const uint16_t * __restrict__ u,
          const float * __restrict__ g, const float * __restrict__ h0, uint16_t * __restrict__ hout,
          uint16_t * __restrict__ vnew, float * __restrict__ ht, int T, int H, int V, int vfirst) {
    __shared__ uint16_t sk[BT][K + 2], sw[BT][K + 2];
    __shared__ __attribute__((aligned(16))) uint16_t sh[K][BV];        // bf16(S), for w_t @ S
    __shared__ __attribute__((aligned(16))) float su[BT][BV];          // u, then the gated bf16(vn) for k^T @ vn
    __shared__ float sg[BT];
    const int nvs = V / BV;
    const int vs = blockIdx.x % nvs, h = (blockIdx.x / nvs) % H, b = blockIdx.x / (nvs * H);
    const int tid = threadIdx.x, v0 = vs * BV, NT = (T + BT - 1) / BT;

    float st[BV];
#pragma unroll
    for (int j = 0; j < BV; j++) st[j] = h0 ? h0[sidx(b, h, tid, v0 + j, H, V, vfirst)] : 0.f;

    for (int it = 0; it < NT; it++) {
        const int t0 = it * BT, tl = min(BT, T - t0);
        for (int e = tid; e < BT * K / 8; e += NTH) {
            const int r = e / (K / 8), c = (e % (K / 8)) * 8;
            const size_t qi = (((size_t)b * T + t0 + r) * H + h) * K + c;
            const bool ok = r < tl;
            const uint4 z = make_uint4(0, 0, 0, 0);
            const uint4 kv = ok ? *(const uint4 *)(k + qi) : z, wv = ok ? *(const uint4 *)(w + qi) : z;
            uint32_t * dk = (uint32_t *)&sk[r][c], * dw = (uint32_t *)&sw[r][c];
            dk[0] = kv.x; dk[1] = kv.y; dk[2] = kv.z; dk[3] = kv.w;
            dw[0] = wv.x; dw[1] = wv.y; dw[2] = wv.z; dw[3] = wv.w;
        }
        for (int e = tid; e < BT * BV; e += NTH) {
            const int r = e / BV, c = e % BV;
            su[r][c] = r < tl ? bf2f(u[(((size_t)b * T + t0 + r) * H + h) * V + v0 + c]) : 0.f;
        }
        if (tid < BT) sg[tid] = tid < tl ? g[((size_t)b * T + t0 + tid) * H + h] : 0.f;
        uint16_t * ho = hout + ((size_t)b * NT + it) * H * K * V;
#pragma unroll
        for (int j = 0; j < BV; j++) { const uint16_t hb = f2bf(st[j]); ho[sidx(0, h, tid, v0 + j, H, V, vfirst)] = hb; sh[tid][j] = hb; }
        __syncthreads();

        const float glast = sg[tl - 1];
        // vn = u - w_t @ bf16(S): thread -> (row r = tid / 2, 8 columns); then gate and round for the state update
        {
            const int r = tid / 2, c0 = (tid % 2) * (BV / 2);
            float acc[BV / 2] = {};
            for (int c = 0; c < K; c++) {
                const float wv = bf2f(sw[r][c]);
                const uint4 hv = *(const uint4 *)&sh[c][c0];
                const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
#pragma unroll
                for (int j = 0; j < BV / 2; j++) acc[j] += wv * bf2f((uint16_t)(hw[j / 2] >> (16 * (j % 2))));
            }
            const float f = r < tl ? exp2f(glast - sg[r]) : 0.f;
            uint16_t * o = vnew ? vnew + (((size_t)b * T + t0 + r) * H + h) * V + v0 + c0 : nullptr;
            float vg[BV / 2];
#pragma unroll
            for (int j = 0; j < BV / 2; j++) {
                const float vn = su[r][c0 + j] - acc[j];
                if (o && r < tl) o[j] = f2bf(vn);
                vg[j] = rbf(vn * f);
            }
            __syncthreads();                       // every thread has read su before it is overwritten
#pragma unroll
            for (int j = 0; j < BV / 2; j++) su[r][c0 + j] = vg[j];
        }
        __syncthreads();
        // S = S * 2^glast + k_t^T @ vg: thread k owns row k
        {
            float a[BV] = {};
            for (int r = 0; r < BT; r++) {
                const float kv = bf2f(sk[r][tid]);
#pragma unroll
                for (int j = 0; j < BV; j += 4) {
                    const float4 v = *(const float4 *)&su[r][j];
                    a[j] += kv * v.x; a[j + 1] += kv * v.y; a[j + 2] += kv * v.z; a[j + 3] += kv * v.w;
                }
            }
            const float decay = exp2f(glast);
#pragma unroll
            for (int j = 0; j < BV; j++) st[j] = st[j] * decay + a[j];
        }
        __syncthreads();
    }
    if (ht) {
#pragma unroll
        for (int j = 0; j < BV; j++) ht[sidx(b, h, tid, v0 + j, H, V, vfirst)] = st[j];
    }
}
