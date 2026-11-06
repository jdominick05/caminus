// Prefill host feed, device half (python/caminus/prefill_feed.py; docs/agentic/spec-prefill-cache.md S1 "Host feed"):
// the raw GGUF rows of a whole chunk are uploaded (PLE: the chunk's distinct rows, 90 B of IQ4_NL each; the token embedding:
// only the token ids, the IQ4_XS table lives on the card) and dequantized here, instead of per position on the host.
// Bitwise the host's dequantization (gguf-py quants.py / caminus.flashnext.dequantize_rows): every value is the same
// float32 products in the same order, IQ4_NL d * kvalue and IQ4_XS (d * (scale - 32)) * kvalue, d converted exactly from
// f16; a lone multiply has nothing to contract or reassociate (__fmul_rn makes the rounding explicit). The products never
// reach the f32 denormal range (|d| >= 2^-24, |kvalue| >= 1), so the denormal mode cannot matter either.
// One thread per output float (coalesced stores; the inputs are tiny: <= 3 MB of PLE rows per 2048-position chunk).
// Bounds: an index outside its table is clamped before the read and reported in err (bit 0: a PLE inverse index outside
// [0, nuniq); bit 1: a token outside [0, nvocab)); the host checks both before the upload, so err stays 0.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

__constant__ float kIQ4NL[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f};

__device__ __forceinline__ float f16_at(const uint8_t * p) {
    return __half2float(__ushort_as_half((unsigned short)(p[0] | (p[1] << 8))));
}

// PLE rows: out[r * HD + k] = the dequantized value k of table row uniq[inv[r]] (HD = bpr * 32; r = position * n_heads + head:
// heads slowest within a position, PLE.embeddings' order). raw: [nuniq][bpr * 18] (block_iq4_nl: f16 d, 16 bytes of nibbles,
// low nibble = value j, high = j + 16).
extern "C" __global__ void __launch_bounds__(256) ple_rows_iq4nl(const uint8_t * raw, const int * inv, int nuniq, int nrows,
        int bpr, float * out, int * err) {
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    const int hd = bpr * 32;
    if (i >= (long)nrows * hd) return;
    const int r = (int)(i / hd), k = (int)(i % hd), b = k >> 5, j = k & 31;
    int u = inv[r];
    if (u < 0 || u >= nuniq) { atomicOr(err, 1); u = u < 0 ? 0 : nuniq - 1; }
    const uint8_t * blk = raw + (size_t)u * (size_t)(bpr * 18) + (size_t)b * 18;
    const uint8_t q = blk[2 + (j & 15)];
    out[i] = __fmul_rn(f16_at(blk), kIQ4NL[j < 16 ? (q & 15) : (q >> 4)]);
}

// token embedding rows: out[p * D + k] = the dequantized value k of token_embd row tok[p] (D = nblk * 256). table:
// [nvocab][nblk * 136] (block_iq4_xs: f16 d, u16 scales_h, u8 scales_l[4], u8 qs[128]; sub-block ib's 6-bit scale is
// scales_l[ib / 2] >> 4 (ib & 1) & 15 | (scales_h >> 2 ib & 3) << 4, its 32 values qs[16 ib ..] low nibbles then high)
extern "C" __global__ void __launch_bounds__(256) embd_rows_iq4xs(const uint8_t * table, const int * tok, int nvocab, int n,
        int nblk, float * out, int * err) {
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    const int D = nblk * 256;
    if (i >= (long)n * D) return;
    const int p = (int)(i / D), k = (int)(i % D), bl = k >> 8, w = k & 255, ib = w >> 5, j = w & 31;
    int t = tok[p];
    if (t < 0 || t >= nvocab) { atomicOr(err, 2); t = t < 0 ? 0 : nvocab - 1; }
    const uint8_t * b = table + (size_t)t * (size_t)(nblk * 136) + (size_t)bl * 136;
    const int sh = b[2] | (b[3] << 8), sl = b[4 + (ib >> 1)];
    const int ls = ((sl >> ((ib & 1) * 4)) & 15) | (((sh >> (2 * ib)) & 3) << 4);
    const float dl = __fmul_rn(f16_at(b), (float)(ls - 32));
    const uint8_t q = b[8 + 16 * ib + (j & 15)];
    out[i] = __fmul_rn(dl, kIQ4NL[j < 16 ? (q & 15) : (q >> 4)]);
}
