// sampler_ref: llama.cpp's default sampler chain (penalties -> top_k -> top_p -> min_p -> temp -> dist) on fixed
// logits, for the S3 parity gate (tests/serve/test_sampler.py). CPU only, no model: the samplers take n_vocab.
// Protocol, little-endian binary on stdin/stdout, one row per request:
//   in:  int32 n_vocab (0 ends), int32 n_hist, float temp, int32 top_k, float top_p, float min_p,
//        float repeat, float freq, float present, int32 last_n, int32 min_keep, int32 hist[n_hist], float logits[n_vocab]
//   out: int32 n, int32 ids[n], float p[n]   (the candidates dist draws from, with its normalized p)
// The history is fed through llama_sampler_accept, as llama-server's init_sampler feeds the prompt.
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <vector>

template <typename T> static bool rd(T * p, size_t n) { return fread(p, sizeof(T), n, stdin) == n; }

int main() {
    for (;;) {
        int32_t nv = 0, nh = 0, top_k = 0, last_n = 0, min_keep = 0;
        float temp, top_p, min_p, rep, freq, pres;
        if (!rd(&nv, 1) || nv == 0) break;
        if (!(rd(&nh, 1) && rd(&temp, 1) && rd(&top_k, 1) && rd(&top_p, 1) && rd(&min_p, 1) && rd(&rep, 1) &&
              rd(&freq, 1) && rd(&pres, 1) && rd(&last_n, 1) && rd(&min_keep, 1))) return 1;
        std::vector<int32_t> hist(nh);
        std::vector<float> logits(nv);
        if (nh && !rd(hist.data(), nh)) return 1;
        if (!rd(logits.data(), nv)) return 1;

        llama_sampler * ch = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(ch, llama_sampler_init_penalties(nv, last_n, rep, freq, pres));
        llama_sampler_chain_add(ch, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(ch, llama_sampler_init_top_p(top_p, min_keep));
        llama_sampler_chain_add(ch, llama_sampler_init_min_p(min_p, min_keep));
        llama_sampler_chain_add(ch, llama_sampler_init_temp_ext(temp, 0.0f, 1.0f));
        llama_sampler_chain_add(ch, llama_sampler_init_dist(1234));
        for (int32_t t : hist) llama_sampler_accept(ch, t);

        std::vector<llama_token_data> cur(nv);
        for (int32_t i = 0; i < nv; ++i) cur[i] = llama_token_data{ i, logits[i], 0.0f };
        llama_token_data_array arr{ cur.data(), cur.size(), -1, false };
        llama_sampler_apply(ch, &arr);

        int32_t n = (int32_t) arr.size;
        std::vector<int32_t> ids(n);
        std::vector<float> p(n);
        for (int32_t i = 0; i < n; ++i) { ids[i] = arr.data[i].id; p[i] = arr.data[i].p; }
        fwrite(&n, 4, 1, stdout);
        fwrite(ids.data(), 4, n, stdout);
        fwrite(p.data(), 4, n, stdout);
        fflush(stdout);
        llama_sampler_free(ch);
    }
    return 0;
}
