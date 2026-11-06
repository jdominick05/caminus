// Golden tensors from llama.cpp for the K3 GGUF engine (docs/INFERENCE-PLAN.md I1): runs a GGUF model with the
// ignite-llama fork's libllama, ONE token per llama_decode (the decode graph our engine replaces), and saves every F32 /
// I32 graph tensor of the chosen layers (plus the embedding, result_norm and result_output) to
// <out>/step<i>/<seq>-<name>.bin, in graph order (names repeat: e.g. the attention and ffn hc_norm of a layer share a
// name). <out>/index.tsv lists step, seq, name, type, ne0..ne3, file. Contiguous tensors only; others are listed as skipped.
//   build: tools/build_gg_golden.sh   run: build/gg_golden MODEL OUTDIR "prompt" [steps=4] [layers=0,1,3,47] [kv=q8_0|f16]
// Greedy mode (no tensor dumps): build/gg_golden MODEL OUTDIR TOKENS_FILE P - [kv] greedy G -> the first P token ids of
// TOKENS_FILE (one per line), then G greedy tokens, ONE token per llama_decode; OUTDIR/greedy.tsv per position: pos,
// token in, argmax, FNV-1a hash of the full logits row, then the top-16 "id:logit" (%.9g); OUTDIR/tokens.txt = P + G ids.
// The KV cache defaults to q8_0 with flash attention on, as the llama-server reference runs did (-ctk/-ctv q8_0 -fa on).
// The prompt must be public text (the dumps hold activations, not weights).
// Long-context modes (QSA references; the modes above are unchanged):
//   build/gg_golden MODEL --tokenize IN OUT  vocab only: IN = UTF-8 pieces separated by 0x1E, each tokenized alone (no
//                                           special tokens added, none parsed) -> OUT: one line of ids per piece
//   build/gg_golden MODEL --lc JOBS          one model load, the M0 settings (layer split 0.55 / 0.45, -fa on, q8_0 K / V,
//                                           n_batch 2048, n_ubatch 512, direct IO); JOBS lines (token files: ids, whitespace):
//     nctx N                                the context size (first line; >= every job's last position + 1)
//     tf NAME TOKFILE D NPOS OUT            memory cleared, tokens [0, D) as batched prompt passes (no logits), then tokens
//                                           D .. D + NPOS - 1 one per llama_decode: the logits rows -> OUT.bin (f32
//                                           [NPOS][n_vocab]), OUT.tsv per row: pos, token in, argmax, FNV-1a of the row
//     greedy NAME TOKFILE D P G OUT         memory cleared, [0, D) batched, tokens D .. P - 1 one per llama_decode, then G
//                                           greedy tokens one per llama_decode -> OUT.tsv (pos, token in, argmax, top-8
//                                           id:logit) and OUT.tokens (the G generated ids)
//     qsa NAME TOKFILE D NSTEP LAYERS OUT   memory cleared, [0, D) batched (no dumps), then tokens D .. D + NSTEP - 1 one
//                                           per llama_decode (the vec lightning-indexer path, T < 8), each step's
//                                           indexer_* tensors of LAYERS (comma list) -> OUT/step<pos>/<seq>-<name>.bin and
//                                           OUT/index.tsv (step = the position), as the dump mode above
//   Each job prints a JOB line: fill seconds and tok/s, decode ms per token.
// G4 floor jobs (docs/agentic/spec-prefill-cache.md G4: llama.cpp's own prefill-then-decode against its one-token decode;
// tools/gates/g4_floor.py drives them; the jobs above are unchanged):
//     ubatch N                              (optional, before the jobs) n_ubatch N instead of 512 (n_batch stays 2048 and
//                                           must be >= N); every job of the run uses it
//     tfx NAME TOKFILE D NPOS B SAVE OUT    memory cleared, tokens [0, D) as prompt passes of B tokens each (no logits;
//                                           B = 1: one token per llama_decode, the decode path; D = 0: none), then tokens
//                                           D .. D + NPOS - 1 one per llama_decode: OUT.tsv as tf's, and OUT.bin (f32
//                                           [NPOS][n_vocab]) only if SAVE is 1
//     slide NAME TOKFILE P0 P1 SAVE OUT     for every p in [P0, P1): memory cleared, tokens [0, p) as ONE prompt pass
//                                           (p <= n_batch), then token p alone: its logits row (p - P0) -> OUT.tsv (pos,
//                                           token in, argmax, FNV-1a), OUT.bin only if SAVE is 1
// QSA depth jobs (impl:qsa-depth, tools/qsa_depth.py: the indexer's selection at the G4 window positions; the jobs above
// are unchanged). NAMES: a comma list of tensor base names (e.g. indexer_q,indexer_score,indexer_top_k); a name ending in
// "@last" is dumped at the job's last step only (indexer_k@last: the pooled keys once, ~4 MB per layer at 32K). Dumps go
// into ONE packed file OUT/dump.bin; OUT/index.tsv lists step, seq, name, type, ne0..ne3, byte offset, bytes:
//     qsax NAME TOKFILE D NSTEP B LAYERS NAMES SAVE OUT
//                                           memory cleared, tokens [0, D) as prompt passes of B tokens (B = 1: one token
//                                           per llama_decode, the decode path), then D .. D + NSTEP - 1 one per
//                                           llama_decode with the dumps (step = the position); OUT/steps.tsv (pos, token
//                                           in, argmax, FNV-1a of the logits row), OUT/logits.bin (f32 [NSTEP][n_vocab])
//                                           only if SAVE is 1
//     qsab NAME TOKFILE D NROW LAYERS NAMES OUT
//                                           memory cleared, tokens [0, D) as prompt passes of n_batch tokens, then D .. D +
//                                           NROW - 1 as ONE llama_decode (NROW <= n_ubatch: one ubatch, the batched
//                                           indexer path) with the dumps (step = D) and every row's logits; OUT/steps.tsv
//                                           per row (pos, token in, argmax, FNV-1a)
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <set>
#include <string>
#include <vector>
#include <sys/stat.h>

struct Golden {
    std::string out;
    int step = 0, seq = 0;
    std::set<int> layers;
    bool active = true;        // false: no tensor is asked for (the lc jobs' batched fills)
    std::string prefix;        // non-empty: only names starting with it (and of the chosen layers)
    std::set<std::string> names, last_names;   // non-empty: only these base names (last_names: only while last is set)
    bool last = false;
    FILE * pack = nullptr;     // non-null: every dump appended here (the qsax / qsab jobs), index lines carry offsets
    uint64_t off = 0;
    FILE * index = nullptr;
    std::vector<uint8_t> buf;
};

static bool wanted(const char * name, const Golden & g) {
    const std::string n(name);
    if (!g.active) return false;
    if (!g.prefix.empty() && n.rfind(g.prefix, 0) != 0) return false;
    if (!g.names.empty() || !g.last_names.empty()) {
        const size_t q = n.rfind('-');
        const std::string base = q == std::string::npos ? n : n.substr(0, q);
        if (!g.names.count(base) && !(g.last && g.last_names.count(base))) return false;
    }
    if (n.rfind("result", 0) == 0 || n.rfind("model.input_embed", 0) == 0 || n.rfind("hc_init", 0) == 0) return true;
    const size_t p = n.rfind('-');
    if (p == std::string::npos || p + 1 >= n.size()) return false;
    for (size_t i = p + 1; i < n.size(); i++) if (n[i] < '0' || n[i] > '9') return false;
    return g.layers.count(std::stoi(n.substr(p + 1))) > 0;
}

static bool eval_cb(ggml_tensor * t, bool ask, void * ud) {
    Golden & g = *(Golden *)ud;
    const bool typed = t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_I32;
    if (ask) return typed && wanted(t->name, g);
    if (!typed || !wanted(t->name, g)) return true;
    const int seq = g.seq++;
    char file[512];
    snprintf(file, sizeof file, "step%d/%04d-%s.bin", g.step, seq, t->name);
    if (!ggml_is_contiguous(t)) {
        fprintf(g.index, "%d\t%d\t%s\t%s\t%lld\t%lld\t%lld\t%lld\tskipped-noncontiguous\n", g.step, seq, t->name, ggml_type_name(t->type),
                (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
        return true;
    }
    const size_t nb = ggml_nbytes(t);
    g.buf.resize(nb);
    ggml_backend_tensor_get(t, g.buf.data(), 0, nb);
    if (g.pack) {
        const size_t w = fwrite(g.buf.data(), 1, nb, g.pack);
        fprintf(g.index, "%d\t%d\t%s\t%s\t%lld\t%lld\t%lld\t%lld\t%llu\t%zu%s\n", g.step, seq, t->name, ggml_type_name(t->type),
                (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3], (unsigned long long)g.off, nb,
                w == nb ? "" : "\tshort-write");
        g.off += nb;
        return true;
    }
    const std::string path = g.out + "/" + file;
    FILE * f = fopen(path.c_str(), "wb");
    if (f) { fwrite(g.buf.data(), 1, nb, f); fclose(f); }
    fprintf(g.index, "%d\t%d\t%s\t%s\t%lld\t%lld\t%lld\t%lld\t%s\n", g.step, seq, t->name, ggml_type_name(t->type),
            (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3], file);
    return true;
}

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static std::vector<llama_token> read_ids(const std::string & path) {
    std::vector<llama_token> v; std::ifstream f(path); long long x;
    while (f >> x) v.push_back((llama_token)x);
    return v;
}

static int tokenize_mode(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s MODEL --tokenize IN OUT\n", argv[0]); return 1; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::ifstream in(argv[3], std::ios::binary); std::stringstream ss; ss << in.rdbuf(); const std::string all = ss.str();
    FILE * out = fopen(argv[4], "w"); if (!out) { fprintf(stderr, "cannot write %s\n", argv[4]); return 1; }
    size_t a = 0; int n_pieces = 0;
    while (a <= all.size()) {
        size_t b = all.find('\x1e', a); if (b == std::string::npos) b = all.size();
        const std::string piece = all.substr(a, b - a);
        std::vector<llama_token> tok(piece.size() + 16);
        int n = llama_tokenize(vocab, piece.data(), (int)piece.size(), tok.data(), (int)tok.size(), false, false);
        if (n < 0) { tok.resize(-n); n = llama_tokenize(vocab, piece.data(), (int)piece.size(), tok.data(), (int)tok.size(), false, false); }
        if (n < 0) { fprintf(stderr, "tokenize failed on piece %d\n", n_pieces); return 1; }
        for (int i = 0; i < n; i++) fprintf(out, i ? " %d" : "%d", tok[i]);
        fprintf(out, "\n"); n_pieces++;
        a = b + 1;
    }
    fclose(out);
    fprintf(stderr, "tokenized %d pieces (vocab add_bos %d)\n", n_pieces, (int)llama_vocab_get_add_bos(vocab));
    llama_model_free(model); llama_backend_free();
    return 0;
}

// tokens [0, n) as prompt passes of n_batch tokens (positions explicit, sequence 0), no logits
static int lc_prefill(llama_context * ctx, const std::vector<llama_token> & t, int n, int nb) {
    for (int i = 0; i < n; i += nb) {
        const int m = std::min(nb, n - i);
        llama_batch b = llama_batch_init(m, 0, 1);
        for (int j = 0; j < m; j++) { b.token[j] = t[i + j]; b.pos[j] = i + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = 0; }
        b.n_tokens = m;
        const int rc = llama_decode(ctx, b); llama_batch_free(b);
        if (rc != 0) return rc;
    }
    llama_synchronize(ctx);
    return 0;
}

// one token at position pos; returns its logits row (nullptr on failure)
static const float * lc_step(llama_context * ctx, llama_token tok, int pos) {
    llama_batch b = llama_batch_init(1, 0, 1);
    b.token[0] = tok; b.pos[0] = pos; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1; b.n_tokens = 1;
    const int rc = llama_decode(ctx, b); llama_batch_free(b);
    return rc == 0 ? llama_get_logits_ith(ctx, -1) : nullptr;
}

static int lc_argmax(const float * lg, int nv) { int best = 0; for (int v = 1; v < nv; v++) if (lg[v] > lg[best]) best = v; return best; }

static uint64_t lc_fnv(const float * lg, int nv) {
    uint64_t h = 1469598103934665603ull; const uint8_t * by = (const uint8_t *)lg;
    for (size_t k = 0; k < (size_t)nv * sizeof(float); k++) { h ^= by[k]; h *= 1099511628211ull; }
    return h;
}

static int lc_mode(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s MODEL --lc JOBS\n", argv[0]); return 1; }
    std::ifstream jf(argv[3]); std::string line; std::vector<std::vector<std::string>> jobs; int nctx = 0, nub = 512;
    while (std::getline(jf, line)) {
        std::istringstream ls(line); std::vector<std::string> w; std::string x;
        while (ls >> x) w.push_back(x);
        if (w.empty() || w[0][0] == '#') continue;
        if (w[0] == "nctx") { if (w.size() != 2) { fprintf(stderr, "JOBS: bad nctx line (nctx N)\n"); return 1; } nctx = atoi(w[1].c_str()); }
        else if (w[0] == "ubatch") { if (w.size() != 2 || atoi(w[1].c_str()) < 1 || atoi(w[1].c_str()) > 2048) { fprintf(stderr, "JOBS: bad ubatch line (ubatch N, 1..2048)\n"); return 1; } nub = atoi(w[1].c_str()); }
        else jobs.push_back(w);
    }
    if (nctx <= 0) { fprintf(stderr, "JOBS: nctx N first\n"); return 1; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    static float split[16] = {0.55f, 0.45f};
    mp.n_gpu_layers = 999; mp.split_mode = LLAMA_SPLIT_MODE_LAYER; mp.load_mode = LLAMA_LOAD_MODE_DIRECT_IO; mp.tensor_split = split;
    double t0 = now_s();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = nctx; cp.n_batch = 2048; cp.n_ubatch = nub; cp.n_seq_max = 1;
    cp.type_k = cp.type_v = GGML_TYPE_Q8_0; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    // the qsa jobs' dumps (inactive outside their one-token steps: nothing is asked for, the graph is not split)
    Golden g; g.active = false; g.prefix = "indexer_";
    bool any_qsa = false;
    for (const auto & w : jobs) any_qsa |= w[0] == "qsa" || w[0] == "qsax" || w[0] == "qsab";
    if (any_qsa) { cp.cb_eval = eval_cb; cp.cb_eval_user_data = &g; }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    printf("LOAD %.1f s; n_ctx %u n_batch %u n_ubatch %u, KV q8_0, flash attention on, split layer 0.55/0.45, n_vocab %d\n",
           now_s() - t0, llama_n_ctx(ctx), llama_n_batch(ctx), llama_n_ubatch(ctx), nv);
    fflush(stdout);
    auto ids_ok = [nv](const std::vector<llama_token> & t, const std::string & name) {
        for (size_t i = 0; i < t.size(); i++)
            if (t[i] < 0 || t[i] >= nv) { fprintf(stderr, "job %s: token %d at %zu outside [0, %d)\n", name.c_str(), (int)t[i], i, nv); return false; }
        return true;
    };
    for (const auto & w : jobs) {
        if (w[0] == "qsa") {
            if (w.size() != 7) { fprintf(stderr, "bad job line (qsa NAME TOKFILE D NSTEP LAYERS OUT)\n"); return 1; }
            const std::vector<llama_token> t = read_ids(w[2]);
            if (!ids_ok(t, w[1])) return 1;
            const int D = atoi(w[3].c_str()), ns = atoi(w[4].c_str());
            if (D < 1 || ns < 1 || D + ns > (int)t.size() || D + ns > nctx) { fprintf(stderr, "job %s: positions out of range\n", w[1].c_str()); return 1; }
            g.out = w[6]; g.layers.clear();
            for (size_t a = 0; a < w[5].size();) { size_t b = w[5].find(',', a); if (b == std::string::npos) b = w[5].size(); g.layers.insert(atoi(w[5].substr(a, b - a).c_str())); a = b + 1; }
            mkdir(g.out.c_str(), 0700);
            g.index = fopen((g.out + "/index.tsv").c_str(), "w");
            if (!g.index) { fprintf(stderr, "cannot write %s/index.tsv\n", g.out.c_str()); return 1; }
            llama_memory_clear(llama_get_memory(ctx), true);
            double a = now_s();
            g.active = false;
            if (lc_prefill(ctx, t, D, (int)llama_n_batch(ctx)) != 0) { fprintf(stderr, "job %s: prefill failed\n", w[1].c_str()); return 1; }
            const double t_fill = now_s() - a;
            FILE * tsv = fopen((g.out + "/steps.tsv").c_str(), "w");
            if (!tsv) { fprintf(stderr, "cannot write %s/steps.tsv\n", g.out.c_str()); return 1; }
            a = now_s();
            for (int j = 0; j < ns; j++) {
                g.step = D + j; g.seq = 0; g.active = true;
                mkdir((g.out + "/step" + std::to_string(D + j)).c_str(), 0700);
                const float * lg = lc_step(ctx, t[D + j], D + j);
                g.active = false;
                if (!lg) { fprintf(stderr, "job %s: decode failed at %d\n", w[1].c_str(), D + j); return 1; }
                fprintf(tsv, "%d\t%d\t%d\t%016llx\n", D + j, t[D + j], lc_argmax(lg, nv), (unsigned long long)lc_fnv(lg, nv));
            }
            fclose(tsv); fclose(g.index); g.index = nullptr;
            printf("JOB qsa %s: fill %d tokens %.2f s (%.0f tok/s); %d one-token decodes with dumps %.2f ms each\n", w[1].c_str(),
                   D, t_fill, D / t_fill, ns, 1e3 * (now_s() - a) / ns);
            fflush(stdout);
            continue;
        }
        if (w[0] == "qsax" || w[0] == "qsab") {
            const bool bx = w[0] == "qsab";
            if (w.size() != (bx ? 8u : 10u)) {
                fprintf(stderr, "bad job line (%s)\n", bx ? "qsab NAME TOKFILE D NROW LAYERS NAMES OUT" : "qsax NAME TOKFILE D NSTEP B LAYERS NAMES SAVE OUT");
                return 1;
            }
            const std::string name = w[1]; const std::vector<llama_token> t = read_ids(w[2]);
            if (!ids_ok(t, name)) return 1;
            const int D = atoi(w[3].c_str()), ns = atoi(w[4].c_str());
            const int B = bx ? (int)llama_n_batch(ctx) : atoi(w[5].c_str());
            const std::string &lay = w[bx ? 5 : 6], &nam = w[bx ? 6 : 7];
            const bool save = !bx && atoi(w[8].c_str()) == 1;
            if (D < 1 || ns < 1 || B < 1 || B > (int)llama_n_batch(ctx) || D + ns > (int)t.size() || D + ns > nctx ||
                (bx && ns > (int)llama_n_ubatch(ctx))) {
                fprintf(stderr, "job %s: positions out of range (D %d, rows %d, B %d, tokens %zu, nctx %d, n_ubatch %u)\n", name.c_str(), D, ns, B,
                        t.size(), nctx, llama_n_ubatch(ctx));
                return 1;
            }
            g.out = w.back(); g.layers.clear(); g.names.clear(); g.last_names.clear(); g.prefix.clear();
            for (size_t a = 0; a < lay.size();) { size_t b = lay.find(',', a); if (b == std::string::npos) b = lay.size(); g.layers.insert(atoi(lay.substr(a, b - a).c_str())); a = b + 1; }
            for (size_t a = 0; a < nam.size();) {
                size_t b = nam.find(',', a); if (b == std::string::npos) b = nam.size();
                std::string s = nam.substr(a, b - a);
                if (s.size() > 5 && s.compare(s.size() - 5, 5, "@last") == 0) g.last_names.insert(s.substr(0, s.size() - 5)); else if (!s.empty()) g.names.insert(s);
                a = b + 1;
            }
            mkdir(g.out.c_str(), 0700);
            g.index = fopen((g.out + "/index.tsv").c_str(), "w"); g.pack = fopen((g.out + "/dump.bin").c_str(), "wb"); g.off = 0;
            FILE * tsv = fopen((g.out + "/steps.tsv").c_str(), "w");
            FILE * bin = save ? fopen((g.out + "/logits.bin").c_str(), "wb") : nullptr;
            if (!g.index || !g.pack || !tsv || (save && !bin)) { fprintf(stderr, "cannot write into %s\n", g.out.c_str()); return 1; }
            llama_memory_clear(llama_get_memory(ctx), true);
            double a = now_s();
            g.active = false; g.last = false;
            if (lc_prefill(ctx, t, D, B) != 0) { fprintf(stderr, "job %s: prefill failed\n", name.c_str()); return 1; }
            const double t_fill = now_s() - a;
            a = now_s();
            if (bx) {
                llama_batch b = llama_batch_init(ns, 0, 1);
                for (int j = 0; j < ns; j++) { b.token[j] = t[D + j]; b.pos[j] = D + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = 1; }
                b.n_tokens = ns; g.step = D; g.seq = 0; g.active = true; g.last = true;
                const int rc = llama_decode(ctx, b); llama_batch_free(b);
                g.active = false;
                if (rc != 0) { fprintf(stderr, "job %s: batched decode failed (%d)\n", name.c_str(), rc); return 1; }
                for (int j = 0; j < ns; j++) {
                    const float * lg = llama_get_logits_ith(ctx, j);
                    if (!lg) { fprintf(stderr, "job %s: no logits for row %d\n", name.c_str(), j); return 1; }
                    fprintf(tsv, "%d\t%d\t%d\t%016llx\n", D + j, t[D + j], lc_argmax(lg, nv), (unsigned long long)lc_fnv(lg, nv));
                }
            } else {
                for (int j = 0; j < ns; j++) {
                    g.step = D + j; g.seq = 0; g.active = true; g.last = j == ns - 1;
                    const float * lg = lc_step(ctx, t[D + j], D + j);
                    g.active = false;
                    if (!lg) { fprintf(stderr, "job %s: decode failed at %d\n", name.c_str(), D + j); return 1; }
                    if (bin) fwrite(lg, sizeof(float), nv, bin);
                    fprintf(tsv, "%d\t%d\t%d\t%016llx\n", D + j, t[D + j], lc_argmax(lg, nv), (unsigned long long)lc_fnv(lg, nv));
                }
            }
            fclose(tsv); if (bin) fclose(bin);
            fclose(g.pack); g.pack = nullptr; fclose(g.index); g.index = nullptr;
            g.names.clear(); g.last_names.clear(); g.prefix = "indexer_"; g.last = false;
            printf("JOB %s %s: fill %d tokens in passes of %d %.2f s; %d %s %.2f ms, ubatch %u, dumps %.1f MB%s\n", w[0].c_str(), name.c_str(), D, B,
                   t_fill, ns, bx ? "rows in one batched decode" : "one-token decodes, each", 1e3 * (now_s() - a) / (bx ? 1 : ns), llama_n_ubatch(ctx),
                   g.off / 1e6, save ? ", logits saved" : "");
            fflush(stdout);
            continue;
        }
        if (w[0] == "tfx" || w[0] == "slide") {
            const bool sl = w[0] == "slide";
            if (w.size() != (sl ? 7u : 8u)) { fprintf(stderr, "bad job line (%s)\n", sl ? "slide NAME TOKFILE P0 P1 SAVE OUT" : "tfx NAME TOKFILE D NPOS B SAVE OUT"); return 1; }
            const std::string name = w[1]; const std::vector<llama_token> t = read_ids(w[2]);
            if (!ids_ok(t, name)) return 1;
            const int a0 = atoi(w[3].c_str()), a1 = atoi(w[4].c_str()), B = sl ? (int)llama_n_batch(ctx) : atoi(w[5].c_str());
            const bool save = atoi(w[sl ? 5 : 6].c_str()) == 1; const std::string out = w.back();
            // tfx: D = a0, NPOS = a1; slide: P0 = a0, P1 = a1 (the rows are positions P0 .. P1 - 1)
            const int first = a0, last = sl ? a1 - 1 : a0 + a1 - 1;
            if (a0 < 0 || (sl ? (a0 < 1 || a1 <= a0 || a1 - 1 > (int)llama_n_batch(ctx)) : (a1 < 1 || B < 1 || B > (int)llama_n_batch(ctx))) ||
                last >= (int)t.size() || last >= nctx) {
                fprintf(stderr, "job %s: positions out of range (%d .. %d, tokens %zu, nctx %d, B %d)\n", name.c_str(), first, last, t.size(), nctx, B); return 1;
            }
            FILE * tsv = fopen((out + ".tsv").c_str(), "w");
            FILE * bin = save ? fopen((out + ".bin").c_str(), "wb") : nullptr;
            if (!tsv || (save && !bin)) { fprintf(stderr, "cannot write %s.tsv / .bin\n", out.c_str()); return 1; }
            double t_fill = 0, t_dec = 0; int steps = 0;
            const int nrow = sl ? a1 - a0 : 1;
            for (int r = 0; r < nrow; r++) {
                const int D = sl ? a0 + r : a0, np = sl ? 1 : a1;
                llama_memory_clear(llama_get_memory(ctx), true);
                double a = now_s();
                if (D > 0 && lc_prefill(ctx, t, D, B) != 0) { fprintf(stderr, "job %s: prefill failed (D %d)\n", name.c_str(), D); return 1; }
                t_fill += now_s() - a; a = now_s();
                for (int j = 0; j < np; j++) {
                    const float * lg = lc_step(ctx, t[D + j], D + j); steps++;
                    if (!lg) { fprintf(stderr, "job %s: decode failed at %d\n", name.c_str(), D + j); return 1; }
                    if (bin) fwrite(lg, sizeof(float), nv, bin);
                    fprintf(tsv, "%d\t%d\t%d\t%016llx\n", D + j, t[D + j], lc_argmax(lg, nv), (unsigned long long)lc_fnv(lg, nv));
                }
                t_dec += now_s() - a;
            }
            fclose(tsv); if (bin) fclose(bin);
            printf("JOB %s %s: %d row(s), fills %.2f s, %d one-token decodes %.2f ms each, ubatch %u%s\n", w[0].c_str(), name.c_str(), steps,
                   t_fill, steps, 1e3 * t_dec / steps, llama_n_ubatch(ctx), save ? ", logits saved" : "");
            fflush(stdout);
            continue;
        }
        const bool tf = w[0] == "tf", gr = w[0] == "greedy";
        if (!(tf && w.size() == 6) && !(gr && w.size() == 7)) { fprintf(stderr, "bad job line (%s ...)\n", w[0].c_str()); return 1; }
        const std::string name = w[1]; const std::vector<llama_token> t = read_ids(w[2]);
        if (!ids_ok(t, name)) return 1;
        const int D = atoi(w[3].c_str());
        const int npos = tf ? atoi(w[4].c_str()) : 0, P = gr ? atoi(w[4].c_str()) : 0, G = gr ? atoi(w[5].c_str()) : 0;
        const std::string out = w.back();
        const int last = tf ? D + npos - 1 : P + G - 2;
        if (D < 1 || (tf && (npos < 1 || D + npos > (int)t.size())) || (gr && (P <= D || P > (int)t.size() || G < 1)) || last >= nctx) {
            fprintf(stderr, "job %s: positions out of range (D %d, tokens %zu, nctx %d)\n", name.c_str(), D, t.size(), nctx); return 1;
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        double a = now_s();
        if (lc_prefill(ctx, t, D, (int)llama_n_batch(ctx)) != 0) { fprintf(stderr, "job %s: prefill failed\n", name.c_str()); return 1; }
        const double t_fill = now_s() - a;
        FILE * tsv = fopen((out + ".tsv").c_str(), "w");
        if (!tsv) { fprintf(stderr, "cannot write %s.tsv\n", out.c_str()); return 1; }
        a = now_s(); int steps = 0;
        if (tf) {
            FILE * bin = fopen((out + ".bin").c_str(), "wb");
            if (!bin) { fprintf(stderr, "cannot write %s.bin\n", out.c_str()); return 1; }
            for (int j = 0; j < npos; j++) {
                const float * lg = lc_step(ctx, t[D + j], D + j); steps++;
                if (!lg) { fprintf(stderr, "job %s: decode failed at %d\n", name.c_str(), D + j); return 1; }
                fwrite(lg, sizeof(float), nv, bin);
                fprintf(tsv, "%d\t%d\t%d\t%016llx\n", D + j, t[D + j], lc_argmax(lg, nv), (unsigned long long)lc_fnv(lg, nv));
            }
            fclose(bin);
        } else {
            std::vector<llama_token> gen; llama_token cur = 0;
            for (int p = D; p < P + G - 1; p++) {
                const llama_token in = p < P ? t[p] : cur;
                const float * lg = lc_step(ctx, in, p); steps++;
                if (!lg) { fprintf(stderr, "job %s: decode failed at %d\n", name.c_str(), p); return 1; }
                std::vector<int> top;   // top-8 by repeated scan (ties: lowest id first, as argmax)
                for (int k = 0; k < 8; k++) {
                    int best = -1;
                    for (int v = 0; v < nv; v++) {
                        if (std::find(top.begin(), top.end(), v) != top.end()) continue;
                        if (best < 0 || lg[v] > lg[best]) best = v;
                    }
                    top.push_back(best);
                }
                fprintf(tsv, "%d\t%d\t%d", p, in, top[0]);
                for (int u : top) fprintf(tsv, "\t%d:%.9g", u, lg[u]);
                fprintf(tsv, "\n");
                if (p >= P - 1) { cur = top[0]; gen.push_back(cur); }
            }
            FILE * tk = fopen((out + ".tokens").c_str(), "w");
            if (!tk) { fprintf(stderr, "cannot write %s.tokens\n", out.c_str()); return 1; }
            for (llama_token v : gen) fprintf(tk, "%d\n", v);
            fclose(tk);
        }
        fclose(tsv);
        const double t_dec = now_s() - a;
        printf("JOB %s %s: fill %d tokens %.2f s (%.0f tok/s); %d one-token decodes %.2f ms each\n", w[0].c_str(), name.c_str(),
               D, t_fill, D / t_fill, steps, 1e3 * t_dec / steps);
        fflush(stdout);
    }
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}

int main(int argc, char ** argv) {
    if (argc >= 3 && std::string(argv[2]) == "--tokenize") return tokenize_mode(argc, argv);
    if (argc >= 3 && std::string(argv[2]) == "--lc") return lc_mode(argc, argv);
    if (argc < 4) { fprintf(stderr, "usage: %s MODEL OUTDIR \"prompt\" [steps=4] [layers=0,1,3,47] [kv=q8_0|f16] [mode=decode|batch]\n"
                                    "       %s MODEL OUTDIR TOKENS_FILE P - [kv=q8_0|f16] greedy G\n", argv[0], argv[0]); return 1; }
    const bool greedy = argc > 8 && std::string(argv[7]) == "greedy";
    Golden g; g.out = argv[2];
    const int steps = argc > 4 ? atoi(argv[4]) : 4;
    std::string ls = argc > 5 ? argv[5] : "0,1,3,47";
    for (size_t a = 0; a < ls.size();) { size_t b = ls.find(',', a); if (b == std::string::npos) b = ls.size(); g.layers.insert(atoi(ls.substr(a, b - a).c_str())); a = b + 1; }
    mkdir(g.out.c_str(), 0700);
    g.index = fopen((g.out + "/index.tsv").c_str(), "w");

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    static float split[16] = {0.55f, 0.45f};
    mp.n_gpu_layers = 999; mp.split_mode = LLAMA_SPLIT_MODE_LAYER; mp.load_mode = LLAMA_LOAD_MODE_DIRECT_IO; mp.tensor_split = split;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512;
    if (!greedy) { cp.cb_eval = eval_cb; cp.cb_eval_user_data = &g; }
    const std::string kv = argc > 6 ? argv[6] : "q8_0";
    cp.type_k = cp.type_v = kv == "f16" ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    fprintf(stderr, "KV cache %s, flash attention on\n", kv.c_str());
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (greedy) {
        const int P = steps, G = atoi(argv[8]), nv = llama_vocab_n_tokens(vocab);
        std::vector<llama_token> seq;
        FILE * in = fopen(argv[3], "r"); if (!in) { fprintf(stderr, "cannot read %s\n", argv[3]); return 1; }
        for (int v; (int)seq.size() < P && fscanf(in, "%d", &v) == 1;) seq.push_back(v);
        fclose(in);
        if ((int)seq.size() < P || P < 1 || P + G > (int)cp.n_ctx) { fprintf(stderr, "need P <= tokens in file, P + G <= %u\n", cp.n_ctx); return 1; }
        FILE * gt = fopen((g.out + "/greedy.tsv").c_str(), "w");
        for (int i = 0; i < P + G - 1; i++) {
            llama_batch b = llama_batch_get_one(&seq[i], 1);
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at step %d\n", i); return 1; }
            const float * lg = llama_get_logits_ith(ctx, -1);
            uint64_t h = 1469598103934665603ull;
            const uint8_t * by = (const uint8_t *)lg;
            for (size_t k = 0; k < (size_t)nv * sizeof(float); k++) { h ^= by[k]; h *= 1099511628211ull; }
            std::vector<int> top;   // top-16 by repeated scan (ties: lowest id first, as argmax)
            for (int t = 0; t < 16; t++) {
                int best = -1;
                for (int v = 0; v < nv; v++) {
                    bool used = false; for (int u : top) if (u == v) { used = true; break; }
                    if (!used && (best < 0 || lg[v] > lg[best])) best = v;
                }
                top.push_back(best);
            }
            fprintf(gt, "%d\t%d\t%d\t%016llx", i, seq[i], top[0], (unsigned long long)h);
            for (int u : top) fprintf(gt, "\t%d:%.9g", u, lg[u]);
            fprintf(gt, "\n");
            if (i >= P - 1) seq.push_back(top[0]);
        }
        fclose(gt);
        FILE * tf = fopen((g.out + "/tokens.txt").c_str(), "w");
        for (int v : seq) fprintf(tf, "%d\n", v);
        fclose(tf); fclose(g.index);
        fprintf(stderr, "greedy: %d prompt + %d generated tokens\n", P, (int)seq.size() - P);
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 0;
    }
    std::vector<llama_token> tok(8192);
    const int n = llama_tokenize(vocab, argv[3], (int)strlen(argv[3]), tok.data(), (int)tok.size(), true, true);
    if (n <= 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    FILE * tf = fopen((g.out + "/tokens.txt").c_str(), "w");
    if (argc > 7 && std::string(argv[7]) == "batch") {
        // the whole prompt in ONE llama_decode with logits for every position (llama.cpp's normal prompt processing);
        // saved as result_output of step 0 with ne1 = positions, for measuring llama.cpp's batch-vs-decode noise
        const int m = std::min(n, steps);
        llama_batch b = llama_batch_init(m, 0, 1);
        for (int i = 0; i < m; i++) {
            b.token[i] = tok[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
            fprintf(tf, "%d\n", tok[i]);
        }
        b.n_tokens = m; g.step = 0; g.seq = 0;
        mkdir((g.out + "/step0").c_str(), 0700);
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "batch decode failed\n"); return 1; }
        llama_batch_free(b);
        fclose(tf); fclose(g.index);
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 0;
    }
    for (int i = 0; i < n && i < steps; i++) {
        g.step = i; g.seq = 0;
        mkdir((g.out + "/step" + std::to_string(i)).c_str(), 0700);
        fprintf(tf, "%d\n", tok[i]);
        llama_batch b = llama_batch_get_one(&tok[i], 1);
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at step %d\n", i); return 1; }
        // the logits of this token, as llama.cpp returns them (also saved as result_output by the callback)
        const float * lg = llama_get_logits_ith(ctx, -1);
        int best = 0; for (int v = 1; v < llama_vocab_n_tokens(vocab); v++) if (lg[v] > lg[best]) best = v;
        fprintf(stderr, "step %d token %d -> argmax %d\n", i, tok[i], best);
    }
    fclose(tf); fclose(g.index);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
