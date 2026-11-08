// This file is a modified copy of llama.cpp's examples/speculative-simple/speculative-simple.cpp (https://github.com/ggml-org/
// llama.cpp, through the fork at ~/src/ignite-llama): the speculative loop is kept, with a per-pass JSON log and a cb_eval tensor
// dumper added (below). llama.cpp is distributed under the following license (THIRD_PARTY.md):
//
// MIT License
//
// Copyright (c) 2023-2026 The ggml authors
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
// dflash_dump: llama.cpp's speculative-simple loop (as ignite-llama's examples, with a per-pass JSON log: SPEC_TOKFILE /
// SPEC_NTOK / SPEC_LOG) plus a cb_eval tensor dumper for the DFlash drafter (DF_DUMP_DIR, DF_DUMP_DFT): the reference of
// tests/decode/test_dflash_k3_ops.py (python/caminus/decode/dflash.py). Built outside the ignite-llama tree against its HIP build:
//   L=~/src/ignite-llama; g++ -O2 -std=c++17 -I$L/common -I$L/include -I$L/ggml/include -I$L/vendor -o dflash_dump \
//     tools/reference/dflash_dump.cpp -L$L/build-hip-therock/bin -lllama-common -lllama -lggml-base -lggml -Wl,-rpath,$L/build-hip-therock/bin
// Run with tools/reference/dflash_refdump.sh under guard.sh.
#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <utility>


// ---- DFlash reference dumper (cb_eval on both contexts). Graphs are told apart by their first named node: the target's
// "inp_embd", the drafter's inject "inp_target_features", the drafter's draft "inp_noise_embd". Every inject call is dumped
// (the K3 replay rebuilds the drafter's whole KV context from them); the draft graphs' per-layer tensors for the first
// DF_DUMP_DFT calls, their lattice / noise rows for every call.
#include "ggml.h"
#include "ggml-backend.h"
static FILE * g_idx = nullptr; static std::string g_dir; static int g_inj = 0, g_dft = 0, g_lim_dft = 3;
static int g_mode = 0;   // 0 target, 1 inject, 2 draft
static long g_file_no = 0;
static bool want_name(const char * n, int mode, int ev) {
    if (mode == 0) return !strncmp(n, "layer_inp-", 10);   // the target's exported features (the inject's input rows)
    if (mode == 1) {
        return !strcmp(n, "inp_target_features") || !strcmp(n, "inp_g_embeddings") || !strncmp(n, "Kcur_injected-", 14) || !strncmp(n, "Vcur_injected-", 14);
    }
    if (mode == 2) {
        if (!strcmp(n, "inp_noise_embd") || !strcmp(n, "dflash2_lattice")) return true;
        if (ev > g_lim_dft) return false;
        static const char * P[] = { "noise_norm-", "attn_conv_in-", "Qcur-", "Kcur-", "Vcur-", "kqv_out-", "attn_conv_out-", "ffn_inp-",
                                    "ffn_norm-", "ffn_conv_in-", "ffn_out-", "ffn_conv_out-", "l_out-", nullptr };
        for (const char ** q = P; *q; ++q) if (!strncmp(n, *q, strlen(*q))) return true;
        return !strcmp(n, "result_norm") || !strcmp(n, "result_output");
    }
    return false;
}
static bool df_cb(struct ggml_tensor * t, bool ask, void *) {
    const char * n = ggml_get_name(t);
    if (ask) {
        if (!strncmp(n, "hc_", 3) || !strcmp(n, "model.input_embed")) g_mode = 0;   // target-only names (the drafter has no hc)
        else if (!strcmp(n, "inp_g_embeddings")) { g_mode = 1; g_inj++; }   // (inp_target_features is a leaf: never asked)
        else if (!strcmp(n, "inp_noise_embd")) { g_mode = 2; g_dft++; }
    }
    static int g_tgt = 0;
    if (ask && !strcmp(n, "model.input_embed")) g_tgt++;
    const int ev = g_mode == 1 ? g_inj : g_mode == 2 ? g_dft : g_tgt;
    if (!want_name(n, g_mode, ev)) return ask ? false : true;
    if (ask) return true;
    const size_t nb = ggml_nbytes(t);
    std::vector<uint8_t> buf(nb);
    ggml_backend_tensor_get(t, buf.data(), 0, nb);
    char fn[512]; snprintf(fn, sizeof fn, "%s/%06ld.bin", g_dir.c_str(), g_file_no++);
    FILE * f = fopen(fn, "wb"); fwrite(buf.data(), 1, nb, f); fclose(f);
    fprintf(g_idx, "{\"file\":\"%s\",\"ctx\":\"%s\",\"eval\":%d,\"name\":\"%s\",\"type\":\"%s\",\"ne\":[%lld,%lld,%lld,%lld],\"nb\":[%zu,%zu,%zu,%zu]}\n",
            strrchr(fn, '/') + 1, g_mode == 1 ? "inj" : g_mode == 2 ? "dft" : "tgt", ev, n, ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
    fflush(g_idx);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    const auto output_limits = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // load the target model

    if (const char * dd = getenv("DF_DUMP_DIR")) {
        g_dir = dd; g_idx = fopen((g_dir + "/index.jsonl").c_str(), "w");
        if (getenv("DF_DUMP_DFT")) g_lim_dft = atoi(getenv("DF_DUMP_DFT"));
        params.cb_eval = df_cb; params.cb_eval_user_data = nullptr;
        params.warmup = false;
    }
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model (if any) - this also creates the MTP draft context when MTP speculation is enabled
    common_speculative_init_result_ptr spec_init;

    {
        common_params params_dft = common_base_params_to_speculative(params);

        spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);

        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = spec_init->context();
    }

    llama_context * ctx_dft = params.speculative.draft.ctx_dft;

    // check if the context supports partial sequence removal
    const bool use_ckpt_tgt = common_context_can_seq_rm(ctx_tgt) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const bool use_ckpt_dft = common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

    if (use_ckpt_tgt) {
        LOG_INF("speculative decoding will use checkpoints (context does not support partial sequence removal)\n");
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);
    // M0b: prompt token ids from SPEC_TOKFILE (first SPEC_NTOK, default 200), per-pass log to SPEC_LOG
    if (const char * tf = getenv("SPEC_TOKFILE")) {
        std::ifstream is(tf); inp.clear(); int v; const int nt = getenv("SPEC_NTOK") ? atoi(getenv("SPEC_NTOK")) : 200;
        while ((int) inp.size() < nt && (is >> v)) inp.push_back(v);
    }
    FILE * flog = getenv("SPEC_LOG") ? fopen(getenv("SPEC_LOG"), "w") : nullptr;
    const int n_vocab_tgt = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
    int n_pass_log = 0;

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    llama_seq_id seq_id = 0;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

    // init the speculator
    const auto & params_spec = params.speculative;

    struct common_speculative * spec = common_speculative_init(params.speculative, 1);

    if (spec == nullptr) {
        LOG_ERR("%s", "failed to initialize speculative decoding\n");
        return 1;
    }

    // eval the prompt on the target and feed it to the speculative implementation(s)
    {
        common_batch batch_prompt(ctx_tgt);
        for (size_t i = 0; i < inp.size() - 1; ++i) {
            batch_prompt.add(inp[i], i, seq_id, false);
        }

        llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_prompt.get());

        if (!common_speculative_process(spec, batch_prompt)) {
            LOG_ERR("%s", "failed to process speculative prompt\n");
            return 1;
        }
    }

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    common_speculative_begin(spec, seq_id, prompt_tgt);

    common_batch batch_tgt(ctx_tgt);

    llama_tokens draft;

    common_prompt_checkpoint ckpt;

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    while (true) {
        bool fresh = false;
        // generate or reuse draft tokens
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        if (draft.empty()) {
            ckpt.update_pos(
                    prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));

            if (use_ckpt_dft) {
                ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            // determine the max draft that fits the remaining context and generation budget
            int n_draft_max = (int) llama_n_ctx(ctx_tgt) - n_past - 2;
            if (params.n_predict >= 0) {
                n_draft_max = std::min(n_draft_max, params.n_predict - n_predict - 1);
            }
            n_draft_max = std::max(n_draft_max, 0);

            // generate a new draft
            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting   = */ true,
                /* .n_max      = */ n_draft_max,
                /* .pos0       = */ n_past,
                /* .id_last    = */ id_last,
                /* .prompt     = */ &prompt_tgt,
                /* .result     = */ &draft, // output
            };
            common_speculative_draft(spec);
            fresh = !draft.empty();

            // save a checkpoint of the target context before evaluating the draft
            // this allows us to restore the state if partial draft acceptance occurs
            if (!draft.empty()) {
                if (use_ckpt_tgt) {
                    ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }

            // reset the draft context to the checkpoint before verification
            if (ctx_dft) {
                if (use_ckpt_dft) {
                    GGML_ASSERT(ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                }

                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }
        } else {
            // we have a previous (partial) draft to reuse from checkpoint restoration
            if (use_ckpt_tgt) {
                GGML_ASSERT(!ckpt.empty());
            }
        }

        // always have a token to evaluate from before - id_last
        batch_tgt.clear();
        batch_tgt.add(id_last, n_past++, seq_id, true);

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            for (size_t i = 0; i < draft.size(); ++i) {
                batch_tgt.add(draft[i], n_past + i, seq_id, true);
            }


            llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get());
        }
        // M0b log: for each verify position i (0..n_draft) the target's top-20 raw logits and the full logsumexp
        if (flog) {
            fprintf(flog, "{\"fresh\":%d,\"pass\":%d,\"pos\":%d,\"last\":%d,\"draft\":[", (int) fresh, n_pass_log++, n_past - 1, id_last);
            for (size_t i = 0; i < draft.size(); ++i) fprintf(flog, "%s%d", i ? "," : "", draft[i]);
            fprintf(flog, "],\"pos_logits\":[");
            std::vector<int> ix(n_vocab_tgt);
            for (size_t i = 0; i <= draft.size(); ++i) {
                const float * lg = llama_get_logits_ith(ctx_tgt, (int) i);
                for (int v = 0; v < n_vocab_tgt; ++v) ix[v] = v;
                std::partial_sort(ix.begin(), ix.begin() + 20, ix.end(), [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
                double mx = lg[ix[0]], se = 0; for (int v = 0; v < n_vocab_tgt; ++v) se += std::exp((double) lg[v] - mx);
                fprintf(flog, "%s{\"lse\":%.9g,\"ids\":[", i ? "," : "", mx + std::log(se));
                for (int k = 0; k < 20; ++k) fprintf(flog, "%s%d", k ? "," : "", ix[k]);
                fprintf(flog, "],\"lg\":[");
                for (int k = 0; k < 20; ++k) fprintf(flog, "%s%.9g", k ? "," : "", lg[ix[k]]);
                fprintf(flog, "]}");
            }
            fprintf(flog, "]");
        }

        // feed the batch to the speculative implementation(s) - this drives the draft model, MTP, Eagle3, etc.
        if (!common_speculative_process(spec, batch_tgt)) {
            LOG_ERR("%s", "failed to process speculative batch\n");
            break;
        }

        // only save the sampler sampler state if we use checkpoints
        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }

        // save the size of the draft being verified
        const size_t n_draft = draft.size();

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
        if (flog) {
            fprintf(flog, ",\"ids\":[");
            for (size_t i = 0; i < ids.size(); ++i) fprintf(flog, "%s%d", i ? "," : "", ids[i]);
            fprintf(flog, "]}\n"); fflush(flog);
        }

        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        // check for partial draft acceptance:
        // if the context doesn't support partial sequence removal, restore the checkpoint
        // and make the accepted tokens the new partial draft for the next iteration
        if (use_ckpt_tgt && ids.size() - 1 < n_draft) {
            LOG_DBG("partial acceptance: %zu < %zu, restoring checkpoint\n", ids.size() - 1, n_draft);

            draft = std::move(ids);

            {
                GGML_ASSERT(ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));

                llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
            }

            if (ctx_dft) {
                GGML_ASSERT(ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));

                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }

            prompt_tgt.resize(ckpt.n_tokens);
            smpl = std::move(smpl_save);

            n_past = (int) prompt_tgt.size();

            continue;
        }

        common_speculative_accept(spec, seq_id, ids.size() - 1);

        // full acceptance: consume the draft and commit accepted tokens
        n_past    += ids.size() - 1;
        n_drafted += n_draft; // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;
        n_predict += ids.size();

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        // clear the draft since it has been consumed
        draft.clear();

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);

            if (ctx_dft) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
            }
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", params_spec.draft.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    common_speculative_print_stats(spec);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl.get());


    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
