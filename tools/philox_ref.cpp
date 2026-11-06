// philox_ref: Philox4x32-10 blocks from torch's header-only at::philox_engine (ATen/core/PhiloxRNGEngine.h), the
// known-answer reference for caminus.serve.sampler.philox4x32. CPU only.
// stdin: lines "seed subsequence offset" (decimal uint64); stdout: the 4 uint32 words of that block, hex.
// philox_engine(seed, subsequence, offset): key = seed (lo, hi), counter = (offset lo, offset hi, subseq lo, subseq hi).
#include <ATen/core/PhiloxRNGEngine.h>

#include <cinttypes>
#include <cstdio>

int main() {
    unsigned long long seed, sub, off;
    while (scanf("%llu %llu %llu", &seed, &sub, &off) == 3) {
        at::philox_engine e(seed, sub, off);
        uint32_t w[4];
        for (auto & x : w) x = e();
        printf("%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n", w[0], w[1], w[2], w[3]);
    }
    return 0;
}
