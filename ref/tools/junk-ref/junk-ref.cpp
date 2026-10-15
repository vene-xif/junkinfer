// junk-ref: reference log-prob dumper / comparator for JunkInfer quality gates.
//
// Modes
//   dump    : run the model over rows of a JTOK file and write, for every scored position,
//             the top-K log-probs, the exact tail mass and the target token's log-prob (JREF).
//             Optionally write a per-token MoE routing trace (JRTR).
//   compare : run the model over the same rows and compare against a JREF file:
//             KL(ref || model) over the ref's top-K plus one lumped tail bucket, PPL of both,
//             top-1 agreement. Prints a summary and optionally writes it as JSON.
//
// All llama.cpp CLI flags (-m, -ngl, -fa, -ctk, --cpu-moe, -ot, -c, -b, -ub, -t ...) are accepted.
// Own flags start with --jr- and are stripped before llama.cpp's parser sees argv.
//
// File formats (little endian)
//   JTOK v1: "JTOK" u32 ver=1, u32 n_rows; per row: u32 len, u32 a, u32 b, i32 tok[len]
//            logits at positions p in [a, b) are scored against tok[p + 1]; requires a < b <= len - 1
//   JREF v1: "JREF" u32 ver=1, u32 n_rows, u32 K, u32 n_vocab; per row: u32 n_pos (= b - a);
//            per position: f32 target_lp, f32 tail_p, i32 id[K], f32 lp[K] (lp sorted descending)
//   JRTR v1: "JRTR" u32 ver=1, u32 n_layer, u32 k; per row: u32 row, u32 len,
//            u8 ids[len][n_layer][k] (255 = layer not seen), u16 fp16 w[len][n_layer][k]

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

struct jtok_row {
    uint32_t a = 0, b = 0;
    std::vector<int32_t> tok;
};

static bool read_exact(FILE * f, void * dst, size_t n) { return fread(dst, 1, n, f) == n; }

static bool read_jtok(const std::string & path, std::vector<jtok_row> & rows) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { LOG_ERR("cannot open %s\n", path.c_str()); return false; }
    char magic[4]; uint32_t ver = 0, n = 0;
    if (!read_exact(f, magic, 4) || memcmp(magic, "JTOK", 4) != 0 || !read_exact(f, &ver, 4) || ver != 1 || !read_exact(f, &n, 4)) {
        LOG_ERR("%s: bad JTOK header\n", path.c_str()); fclose(f); return false;
    }
    rows.resize(n);
    for (uint32_t r = 0; r < n; ++r) {
        uint32_t len = 0;
        if (!read_exact(f, &len, 4) || !read_exact(f, &rows[r].a, 4) || !read_exact(f, &rows[r].b, 4)) { fclose(f); return false; }
        rows[r].tok.resize(len);
        if (!read_exact(f, rows[r].tok.data(), size_t(len) * 4)) { LOG_ERR("%s: truncated row %u\n", path.c_str(), r); fclose(f); return false; }
        if (!(rows[r].a < rows[r].b && rows[r].b <= len - 1)) {
            LOG_ERR("%s: row %u bad range [%u,%u) len %u\n", path.c_str(), r, rows[r].a, rows[r].b, len); fclose(f); return false;
        }
    }
    fclose(f);
    return true;
}

// ---------------------------------------------------------------- routing trace

struct route_state {
    bool     on      = false;
    int      n_layer = 0;
    int      k       = 0;
    int64_t  ub_T    = -1;          // tokens in the current ubatch
    std::vector<uint8_t>  ub_ids;   // [T][n_layer][k]
    std::vector<uint16_t> ub_w;     // [T][n_layer][k]
    std::vector<uint8_t>  seen;     // [n_layer] topk seen in current ubatch
    std::vector<uint8_t>  seen_w;   // [n_layer] weights seen in current ubatch
    std::vector<char>     tmp;
    int last_topk_il = -1;          // MoE layers arrive in order; a wrap-around starts a new ubatch
    // ubatches of the current decode call, concatenated in token order (single sequence)
    std::vector<uint8_t>  acc_ids;
    std::vector<uint16_t> acc_w;
    int64_t acc_T = 0;
    bool error = false;
};

static int layer_of(const char * name, const char * prefix) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0) return -1;
    char * end = nullptr;
    long v = strtol(name + n, &end, 10);
    if (end == name + n || *end != '\0') return -1;
    return (int) v;
}

static void route_begin_ubatch(route_state & rs, int64_t T) {
    rs.ub_T = T;
    rs.ub_ids.assign(size_t(T) * rs.n_layer * rs.k, 255);
    rs.ub_w.assign(size_t(T) * rs.n_layer * rs.k, 0);
    rs.seen.assign(rs.n_layer, 0);
    rs.seen_w.assign(rs.n_layer, 0);
}

static void route_flush(route_state & rs) {
    if (rs.ub_T <= 0) return;
    for (int il = 0; il < rs.n_layer; ++il) {
        if (rs.seen[il] != rs.seen_w[il]) rs.error = true;
    }
    rs.acc_ids.insert(rs.acc_ids.end(), rs.ub_ids.begin(), rs.ub_ids.end());
    rs.acc_w.insert(rs.acc_w.end(), rs.ub_w.begin(), rs.ub_w.end());
    rs.acc_T += rs.ub_T;
    rs.ub_T = -1;
}

static void route_begin_call(route_state & rs) {
    rs.acc_ids.clear(); rs.acc_w.clear(); rs.acc_T = 0;
    rs.ub_T = -1; rs.last_topk_il = -1;
}

static bool route_cb(struct ggml_tensor * t, bool ask, void * ud) {
    route_state & rs = *(route_state *) ud;
    const int il_topk = layer_of(t->name, "ffn_moe_topk-");
    const int il_w    = layer_of(t->name, "ffn_moe_weights_norm-");
    const bool want = il_topk >= 0 || il_w >= 0;
    if (ask) return want;
    if (!want) return true;

    const int il = il_topk >= 0 ? il_topk : il_w;
    if (il >= rs.n_layer) { rs.error = true; return true; }

    // both tensors are [k, T] (topk: i32 view of the argsort, weights: f32 after the norm div)
    const int64_t k = t->ne[0];
    const int64_t T = t->ne[1];
    if (k != rs.k || t->ne[2] != 1 || t->ne[3] != 1) { rs.error = true; return true; }
    if (il_topk >= 0 && (rs.ub_T < 0 || il_topk <= rs.last_topk_il)) {
        // first MoE layer of a new ubatch: keep the previous one, start a fresh buffer
        route_flush(rs);
        route_begin_ubatch(rs, T);
    }
    if (rs.ub_T != T) { rs.error = true; return true; }   // weights without their topk, or a size mismatch
    const size_t nb = ggml_nbytes(t);
    rs.tmp.resize(nb);
    ggml_backend_tensor_get(t, rs.tmp.data(), 0, nb);
    for (int64_t j = 0; j < T; ++j) {
        for (int64_t e = 0; e < k; ++e) {
            const char * p = rs.tmp.data() + e * t->nb[0] + j * t->nb[1];
            const size_t o = (size_t(j) * rs.n_layer + il) * rs.k + e;
            if (il_topk >= 0) {
                if (t->type != GGML_TYPE_I32) { rs.error = true; return true; }
                const int32_t id = *(const int32_t *) p;
                rs.ub_ids[o] = (id >= 0 && id < 255) ? (uint8_t) id : 254;
            } else {
                if (t->type != GGML_TYPE_F32) { rs.error = true; return true; }
                rs.ub_w[o] = ggml_fp32_to_fp16(*(const float *) p);
            }
        }
    }
    if (il_topk >= 0) { rs.seen[il] = 1; rs.last_topk_il = il; } else rs.seen_w[il] = 1;
    return true;
}

// ---------------------------------------------------------------- math helpers

static double log_sum_exp(const float * x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, x[i]);
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::exp(double(x[i]) - mx);
    return mx + std::log(s);
}

struct kl_stats {
    std::vector<float> kl;
    double nll_model = 0.0, nll_ref = 0.0;
    uint64_t n = 0, top1_agree = 0;
};

static double pct(std::vector<float> & v, double q) {
    if (v.empty()) return 0.0;
    const size_t i = std::min(v.size() - 1, size_t(q * (v.size() - 1) + 0.5));
    return v[i];
}

// ---------------------------------------------------------------- main

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string mode, tok_path, ref_path, out_path, route_path, summary_path, label;
    int topk = 512;
    int row_first = 0, row_last = -1;

    // strip --jr-* flags
    std::vector<char *> rest;
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char * what) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", what); exit(2); }
            return argv[++i];
        };
        if      (a == "--jr-mode")    mode = need("--jr-mode");
        else if (a == "--jr-tokens")  tok_path = need("--jr-tokens");
        else if (a == "--jr-ref")     ref_path = need("--jr-ref");
        else if (a == "--jr-out")     out_path = need("--jr-out");
        else if (a == "--jr-route")   route_path = need("--jr-route");
        else if (a == "--jr-summary") summary_path = need("--jr-summary");
        else if (a == "--jr-label")   label = need("--jr-label");
        else if (a == "--jr-topk")    topk = std::atoi(need("--jr-topk").c_str());
        else if (a == "--jr-rows") {
            std::string v = need("--jr-rows");
            if (sscanf(v.c_str(), "%d:%d", &row_first, &row_last) != 2) { fprintf(stderr, "--jr-rows a:b\n"); return 2; }
        }
        else rest.push_back(argv[i]);
    }
    if (mode != "dump" && mode != "compare") { fprintf(stderr, "--jr-mode dump|compare required\n"); return 2; }
    if (tok_path.empty()) { fprintf(stderr, "--jr-tokens required\n"); return 2; }
    if (mode == "dump" && out_path.empty()) { fprintf(stderr, "dump needs --jr-out\n"); return 2; }
    if (mode == "compare" && ref_path.empty()) { fprintf(stderr, "compare needs --jr-ref\n"); return 2; }
    if (topk < 1 || topk > 65536) { fprintf(stderr, "bad --jr-topk\n"); return 2; }

    common_params params;
    common_init();
    int rargc = (int) rest.size();
    if (!common_params_parse(rargc, rest.data(), params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }
    params.warmup = false;

    std::vector<jtok_row> rows;
    if (!read_jtok(tok_path, rows)) return 1;
    if (row_last < 0 || row_last > (int) rows.size()) row_last = (int) rows.size();
    if (row_first < 0 || row_first > row_last) { fprintf(stderr, "bad --jr-rows\n"); return 2; }

    route_state rs;
    if (!route_path.empty()) {
        rs.on = true;
        params.cb_eval = route_cb;
        params.cb_eval_user_data = &rs;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();
    if (!model || !ctx) { LOG_ERR("failed to init\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_ctx   = (int) llama_n_ctx(ctx);
    const int n_batch = (int) llama_n_batch(ctx);

    if (rs.on) {
        rs.n_layer = llama_model_n_layer(model);
        rs.k       = 0;
        // n_expert_used from metadata
        char buf[64];
        char key[128];
        snprintf(key, sizeof(key), "%s.expert_used_count", "qwen35moe");
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) rs.k = std::atoi(buf);
        if (rs.k <= 0 || rs.k > 64) { LOG_ERR("could not read expert_used_count (%d)\n", rs.k); return 1; }
    }

    for (int r = row_first; r < row_last; ++r) {
        if ((int) rows[r].tok.size() > n_ctx) {
            LOG_ERR("row %d has %zu tokens > n_ctx %d\n", r, rows[r].tok.size(), n_ctx);
            return 1;
        }
        for (int32_t t : rows[r].tok) {
            if (t < 0 || t >= n_vocab) { LOG_ERR("row %d: token %d out of vocab %d\n", r, t, n_vocab); return 1; }
        }
    }

    // ---- open outputs / inputs
    FILE * fout = nullptr, * fref = nullptr, * froute = nullptr;
    uint32_t ref_K = 0;
    bool ref_skip_tail = false;
    if (mode == "dump") {
        fout = fopen(out_path.c_str(), "wb");
        if (!fout) { LOG_ERR("cannot write %s\n", out_path.c_str()); return 1; }
        const uint32_t ver = 1, nr = uint32_t(row_last - row_first), K = uint32_t(topk), nv = uint32_t(n_vocab);
        fwrite("JREF", 1, 4, fout); fwrite(&ver, 4, 1, fout); fwrite(&nr, 4, 1, fout); fwrite(&K, 4, 1, fout); fwrite(&nv, 4, 1, fout);
    } else {
        fref = fopen(ref_path.c_str(), "rb");
        if (!fref) { LOG_ERR("cannot open %s\n", ref_path.c_str()); return 1; }
        char magic[4]; uint32_t ver, nr, K, nv;
        if (!read_exact(fref, magic, 4) || memcmp(magic, "JREF", 4) || !read_exact(fref, &ver, 4) || ver != 1 ||
            !read_exact(fref, &nr, 4) || !read_exact(fref, &K, 4) || !read_exact(fref, &nv, 4)) {
            LOG_ERR("bad JREF header\n"); return 1;
        }
        if (nv != uint32_t(n_vocab)) { LOG_ERR("JREF vocab %u != model vocab %d\n", nv, n_vocab); return 1; }
        ref_K = K;
        if (nr == uint32_t(rows.size()) && (row_first > 0 || row_last < (int) rows.size())) {
            // full reference, row subset selected: skip the reference records of the rows before row_first
            ref_skip_tail = row_last < (int) rows.size();
            const size_t rec = 8 + size_t(K) * 8;
            for (int r = 0; r < row_first; ++r) {
                uint32_t np = 0;
                if (!read_exact(fref, &np, 4) || np != rows[r].b - rows[r].a || fseek(fref, long(np * rec), SEEK_CUR) != 0) {
                    LOG_ERR("JREF skip failed at row %d\n", r); return 1;
                }
            }
        } else if (nr != uint32_t(row_last - row_first)) {
            LOG_ERR("JREF has %u rows, selection has %d (tokens file %zu)\n", nr, row_last - row_first, rows.size()); return 1;
        }
    }
    if (rs.on) {
        froute = fopen(route_path.c_str(), "wb");
        if (!froute) { LOG_ERR("cannot write %s\n", route_path.c_str()); return 1; }
        const uint32_t ver = 1, nl = uint32_t(rs.n_layer), k = uint32_t(rs.k);
        fwrite("JRTR", 1, 4, froute); fwrite(&ver, 4, 1, froute); fwrite(&nl, 4, 1, froute); fwrite(&k, 4, 1, froute);
    }

    kl_stats st;
    std::vector<int32_t> idx(n_vocab);
    std::vector<float>   lp(n_vocab);
    std::vector<int32_t> rid; std::vector<float> rlp, topv;
    common_batch batch(ctx);

    const auto t_all = std::chrono::steady_clock::now();
    uint64_t tokens_done = 0;

    for (int r = row_first; r < row_last; ++r) {
        const jtok_row & row = rows[r];
        const int len = (int) row.tok.size();
        const uint32_t n_pos = row.b - row.a;
        const auto t0 = std::chrono::steady_clock::now();

        if (fout) fwrite(&n_pos, 4, 1, fout);
        if (fref) {
            uint32_t np = 0;
            if (!read_exact(fref, &np, 4) || np != n_pos) { LOG_ERR("JREF row %d n_pos mismatch (%u vs %u)\n", r, np, n_pos); return 1; }
        }

        std::vector<uint8_t>  row_ids;
        std::vector<uint16_t> row_w;
        if (rs.on) {
            row_ids.assign(size_t(len) * rs.n_layer * rs.k, 255);
            row_w.assign(size_t(len) * rs.n_layer * rs.k, 0);
        }

        llama_memory_clear(llama_get_memory(ctx), true);
        double row_kl = 0.0; uint64_t row_n = 0;

        for (int start = 0; start < len; start += n_batch) {
            const int n = std::min(n_batch, len - start);
            batch.clear();
            std::vector<int> out_pos;
            for (int i = 0; i < n; ++i) {
                const int pos = start + i;
                const bool out = pos >= (int) row.a && pos < (int) row.b;
                batch.add(row.tok[pos], pos, 0, out);
                if (out) out_pos.push_back(pos);
            }
            if (rs.on) route_begin_call(rs);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                LOG_ERR("decode failed at row %d pos %d\n", r, start); return 1;
            }
            if (rs.on) {
                route_flush(rs);
                if (rs.error) { LOG_ERR("routing callback saw an unexpected tensor shape/type/order\n"); return 1; }
                if (rs.acc_T != n) { LOG_ERR("routing: ubatches covered %lld tokens, batch %d\n", (long long) rs.acc_T, n); return 1; }
                std::copy(rs.acc_ids.begin(), rs.acc_ids.end(), row_ids.begin() + size_t(start) * rs.n_layer * rs.k);
                std::copy(rs.acc_w.begin(),   rs.acc_w.end(),   row_w.begin()   + size_t(start) * rs.n_layer * rs.k);
            }
            if (out_pos.empty()) continue;
            const float * logits = llama_get_logits(ctx);
            for (size_t o = 0; o < out_pos.size(); ++o) {
                const float * lg = logits + o * size_t(n_vocab);
                const int pos = out_pos[o];
                const int32_t target = row.tok[pos + 1];
                const double lse = log_sum_exp(lg, n_vocab);
                if (fout) {
                    for (int i = 0; i < n_vocab; ++i) lp[i] = float(double(lg[i]) - lse);
                    std::iota(idx.begin(), idx.end(), 0);
                    const int K = std::min(topk, n_vocab);
                    std::nth_element(idx.begin(), idx.begin() + K, idx.end(), [&](int x, int y) { return lp[x] > lp[y]; });
                    std::sort(idx.begin(), idx.begin() + K, [&](int x, int y) { return lp[x] > lp[y]; });
                    double top = 0.0;
                    for (int i = 0; i < K; ++i) top += std::exp((double) lp[idx[i]]);
                    const float target_lp = lp[target];
                    const float tail = float(std::max(0.0, 1.0 - top));
                    fwrite(&target_lp, 4, 1, fout);
                    fwrite(&tail, 4, 1, fout);
                    fwrite(idx.data(), 4, K, fout);
                    topv.resize(K);
                    for (int i = 0; i < K; ++i) topv[i] = lp[idx[i]];
                    fwrite(topv.data(), 4, K, fout);
                    st.nll_ref -= target_lp; st.n++;
                } else {
                    float tlp, tail;
                    rid.resize(ref_K); rlp.resize(ref_K);
                    if (!read_exact(fref, &tlp, 4) || !read_exact(fref, &tail, 4) ||
                        !read_exact(fref, rid.data(), size_t(ref_K) * 4) || !read_exact(fref, rlp.data(), size_t(ref_K) * 4)) {
                        LOG_ERR("JREF truncated at row %d pos %d\n", r, pos); return 1;
                    }
                    double kl = 0.0, q_top = 0.0;
                    for (uint32_t i = 0; i < ref_K; ++i) {
                        const double lq = double(lg[rid[i]]) - lse;
                        const double lpr = rlp[i];
                        kl += std::exp(lpr) * (lpr - lq);
                        q_top += std::exp(lq);
                    }
                    if (tail > 0.0f) {
                        const double q_tail = std::max(1e-30, 1.0 - q_top);
                        kl += double(tail) * (std::log(double(tail)) - std::log(q_tail));
                    }
                    int am = 0;
                    for (int i = 1; i < n_vocab; ++i) if (lg[i] > lg[am]) am = i;
                    st.kl.push_back(float(kl));
                    st.top1_agree += (am == rid[0]);
                    st.nll_model -= double(lg[target]) - lse;
                    st.nll_ref   -= tlp;
                    st.n++;
                    row_kl += kl; row_n++;
                }
            }
        }

        if (froute) {
            const uint32_t ur = uint32_t(r), ul = uint32_t(len);
            fwrite(&ur, 4, 1, froute); fwrite(&ul, 4, 1, froute);
            fwrite(row_ids.data(), 1, row_ids.size(), froute);
            fwrite(row_w.data(), 2, row_w.size(), froute);
        }
        tokens_done += len;
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (mode == "compare") {
            LOG_INF("row %d/%d len %d: %.1f s (%.0f tok/s), row KL %.6f\n", r + 1, row_last, len, dt, len / dt, row_n ? row_kl / row_n : 0.0);
        } else {
            LOG_INF("row %d/%d len %d: %.1f s (%.0f tok/s)\n", r + 1, row_last, len, dt, len / dt);
        }
    }

    const double dt_all = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all).count();
    if (fout) { fclose(fout); }
    if (froute) { fclose(froute); }
    if (fref) {
        char extra;
        if (!ref_skip_tail && fread(&extra, 1, 1, fref) == 1) { LOG_ERR("JREF has trailing data - row selection mismatch?\n"); return 1; }
        fclose(fref);
    }

    char summary[2048];
    if (mode == "compare") {
        std::vector<float> s = st.kl;
        std::sort(s.begin(), s.end());
        const double mean = s.empty() ? 0.0 : std::accumulate(s.begin(), s.end(), 0.0) / s.size();
        snprintf(summary, sizeof(summary),
            "{\"label\": \"%s\", \"tokens\": \"%s\", \"n\": %llu, \"kl_mean\": %.8f, \"kl_p50\": %.8f, \"kl_p90\": %.8f, "
            "\"kl_p99\": %.8f, \"kl_p999\": %.8f, \"kl_max\": %.8f, \"top1_agree\": %.6f, \"ppl_model\": %.6f, \"ppl_ref\": %.6f, "
            "\"seconds\": %.1f, \"tok_per_s\": %.1f}",
            label.c_str(), tok_path.c_str(), (unsigned long long) st.n, mean, pct(s, 0.5), pct(s, 0.9), pct(s, 0.99), pct(s, 0.999),
            s.empty() ? 0.0 : s.back(), st.n ? double(st.top1_agree) / st.n : 0.0,
            st.n ? std::exp(st.nll_model / st.n) : 0.0, st.n ? std::exp(st.nll_ref / st.n) : 0.0, dt_all, tokens_done / dt_all);
    } else {
        snprintf(summary, sizeof(summary),
            "{\"label\": \"%s\", \"tokens\": \"%s\", \"n\": %llu, \"ppl_ref\": %.6f, \"seconds\": %.1f, \"tok_per_s\": %.1f}",
            label.c_str(), tok_path.c_str(), (unsigned long long) st.n, st.n ? std::exp(st.nll_ref / st.n) : 0.0, dt_all, tokens_done / dt_all);
    }
    LOG_INF("SUMMARY %s\n", summary);
    if (!summary_path.empty()) {
        FILE * fs = fopen(summary_path.c_str(), "w");
        if (fs) { fprintf(fs, "%s\n", summary); fclose(fs); }
    }

    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
