// oc_activations.h -- stream per-token activations out of llama-server.
//
// Self-contained on purpose. The server sources are large and change often, so
// a patch against their internals would rot within weeks. Everything here
// lives in one header; apply_patch.py adds the few insertion points that wire
// it in, and PATCH.md explains each.
//
// WHAT IT EMITS
//
// During generation llama_decode is called once per token, so the ffn_down
// node fires once per token with ne[1] == 1. We capture the same quantity
// cett-dump does -- the input to down_proj, normalised by the layer's output
// norm -- reduce it in-process, and hand it to whoever is subscribed.
//
//     |a| / ||layer output||
//
// Note this is CETT *without* the weight column norm. cett-dump multiplies by
// ||W[:, j]|| afterwards, read from the GGUF. Doing that here would mean
// dequantising every down_proj at startup and holding it in RAM, which is a
// large cost for a live view whose job is showing relative movement. The
// consequence is that live frames are not numerically comparable to replay
// frames; they are the same shape and the same story, on a different scale.
// If you need them comparable, supply the norms to the client and multiply
// there.
//
// CONCURRENCY
//
// cb_eval runs on the inference thread. Subscribers are served from HTTP
// threads. A frame is built under no lock, then published under a short one
// into per-subscriber bounded queues that drop rather than block: a slow
// client must never stall the inference thread, because that would slow
// generation for everyone.
//
// LIMITATION: single slot. With --parallel > 1 a batch interleaves tokens from
// several sequences, and the node carries no sequence id we can recover here.
// oc_activations refuses to enable itself when n_parallel > 1 rather than
// emitting frames that silently mix two conversations.

#pragma once

#include "llama.h"
#include "ggml.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ns {

enum class tier { sparse, binned, raw };

struct config {
    bool    enabled   = false;
    // Optional classifier from scripts/classifier.py, exported flat. Without
    // it the server can report activity but cannot flag anything: "this
    // neuron is busy" and "this token is likely fabricated" are different
    // claims, and only the second needs trained weights.
    std::string classifier_path;
    tier    reduction = tier::sparse;
    int     top_k     = 48;     // sparse: cells kept per token
    int     bins      = 512;    // binned: columns per layer
    int     max_queue = 8;      // frames held per subscriber before dropping
};

// One token's worth, already reduced. Serialised to JSON by the route.
struct frame {
    int64_t              index = 0;
    tier                 kind  = tier::sparse;
    int                  n_layers = 0;
    int                  n_cols   = 0;   // binned: columns per layer
    std::vector<float>   values;         // binned/raw: n_layers * n_cols
    std::vector<int32_t> cells;          // sparse: (layer, neuron) pairs
    std::vector<float>   cell_values;    // sparse: parallel to cells
    float                peak  = 0.0f;
    float                score = 0.0f;    // classifier output, 0 if none
    bool                 scored = false;
};

// Flat float32: [n_layers * n_ff] coefficients, then one float intercept.
// Written by scripts/export_classifier_bin.py.
struct classifier {
    std::vector<float> coef;
    float              intercept = 0.0f;
    bool               ok = false;

    bool load(const std::string & path, int expect) {
        ok = false;
        FILE * f = fopen(path.c_str(), "rb");
        if (!f) return false;
        fseek(f, 0, SEEK_END);
        const long bytes = ftell(f);
        fseek(f, 0, SEEK_SET);
        const long n = bytes / 4 - 1;
        if (n != expect) {
            fprintf(stderr, "[ns] classifier has %ld weights, model needs %d; "
                            "wrong model for this classifier\n", n, expect);
            fclose(f);
            return false;
        }
        coef.resize(n);
        if (fread(coef.data(), 4, n, f) != (size_t) n ||
            fread(&intercept, 4, 1, f) != 1) {
            fclose(f);
            return false;
        }
        fclose(f);
        ok = true;
        return true;
    }

    // Dotted against the *unreduced* rows: reducing first would discard the
    // coordinates the classifier was trained on.
    float score(const std::vector<float> & rows) const {
        if (!ok || rows.size() != coef.size()) return 0.0f;
        double acc = intercept;
        for (size_t i = 0; i < rows.size(); ++i) acc += (double) rows[i] * coef[i];
        return (float) acc;
    }
};

class sink {
public:
    explicit sink(size_t cap) : cap_(cap) {}

    // Called from the inference thread. Never blocks.
    void push(const std::shared_ptr<frame> & f) {
        std::lock_guard<std::mutex> lk(m_);
        if (q_.size() >= cap_) {
            q_.pop_front();
            dropped_++;
        }
        q_.push_back(f);
    }

    // Called from an HTTP thread.
    std::shared_ptr<frame> pop() {
        std::lock_guard<std::mutex> lk(m_);
        if (q_.empty()) return nullptr;
        auto f = q_.front();
        q_.pop_front();
        return f;
    }

    uint64_t dropped() const { return dropped_.load(); }

private:
    mutable std::mutex m_;
    std::deque<std::shared_ptr<frame>> q_;
    size_t cap_;
    std::atomic<uint64_t> dropped_{0};
};

class streamer {
public:
    config     cfg;
    classifier clf;

    void subscribe(const std::shared_ptr<sink> & s) {
        std::lock_guard<std::mutex> lk(subs_m_);
        subs_.push_back(s);
    }

    void unsubscribe(const std::shared_ptr<sink> & s) {
        std::lock_guard<std::mutex> lk(subs_m_);
        for (size_t i = 0; i < subs_.size(); ++i) {
            if (subs_[i] == s) { subs_.erase(subs_.begin() + i); return; }
        }
    }

    bool has_subscribers() {
        std::lock_guard<std::mutex> lk(subs_m_);
        return !subs_.empty();
    }

    void publish(const std::shared_ptr<frame> & f) {
        std::lock_guard<std::mutex> lk(subs_m_);
        for (auto & s : subs_) s->push(f);
    }

    int64_t next_index() { return index_++; }

private:
    std::mutex subs_m_;
    std::vector<std::shared_ptr<sink>> subs_;
    std::atomic<int64_t> index_{0};
};

// ---------------------------------------------------------------- internals

inline int parse_layer(const char * name, const char * prefix) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0) return -1;
    const char * p = name + n;
    if (*p == '\0') return -1;
    int v = 0;
    for (; *p; ++p) {
        if (*p < '0' || *p > '9') return -1;
        v = v * 10 + (*p - '0');
    }
    return v;
}

// Per-token accumulation across the layers of one decode. Reset when layer 0
// is seen again, which is how we detect a new token without threading extra
// state through the callback.
struct accum {
    int                n_layers = 0;
    int                n_ff     = 0;
    std::vector<float> rows;                 // n_layers * n_ff
    std::vector<uint8_t> filled;
    std::vector<uint8_t> scratch;
    std::vector<float>   f32;

    void reset(int layers, int ff) {
        if (n_layers != layers || n_ff != ff) {
            n_layers = layers; n_ff = ff;
            rows.assign((size_t) layers * ff, 0.0f);
            filled.assign(layers, 0);
        } else {
            std::fill(rows.begin(), rows.end(), 0.0f);
            std::fill(filled.begin(), filled.end(), 0);
        }
    }

    bool complete() const {
        for (auto v : filled) if (!v) return false;
        return n_layers > 0;
    }
};

inline bool fetch_f32(const ggml_tensor * t, std::vector<uint8_t> & raw,
                      std::vector<float> & out) {
    const size_t nb = ggml_nbytes(t);
    raw.resize(nb);
    if (ggml_backend_buffer_is_host(t->buffer)) {
        memcpy(raw.data(), t->data, nb);
    } else {
        // Required with a GPU backend: the data is in device memory.
        ggml_backend_tensor_get(t, raw.data(), 0, nb);
    }
    const int64_t n = ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), raw.data(), n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *) raw.data();
        for (int64_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        return false;
    }
    return true;
}

inline std::shared_ptr<frame> reduce(const accum & a, const config & cfg,
                                     int64_t index,
                                     const classifier * clf = nullptr) {
    auto f = std::make_shared<frame>();
    f->index    = index;
    f->kind     = cfg.reduction;
    f->n_layers = a.n_layers;

    float peak = 0.0f;
    for (float v : a.rows) if (v > peak) peak = v;
    f->peak = peak;

    if (clf && clf->ok) {
        f->score  = clf->score(a.rows);
        f->scored = true;
    }

    if (cfg.reduction == tier::raw) {
        f->n_cols = a.n_ff;
        f->values = a.rows;
        return f;
    }

    if (cfg.reduction == tier::binned) {
        const int bins = cfg.bins < a.n_ff ? cfg.bins : a.n_ff;
        const int step = (a.n_ff + bins - 1) / bins;
        f->n_cols = bins;
        f->values.assign((size_t) a.n_layers * bins, 0.0f);
        for (int l = 0; l < a.n_layers; ++l) {
            const float * row = a.rows.data() + (size_t) l * a.n_ff;
            for (int b = 0; b < bins; ++b) {
                // Max, not mean: the cells worth watching are a tiny fraction
                // of the row and averaging erases them.
                float m = 0.0f;
                for (int i = b * step; i < (b + 1) * step && i < a.n_ff; ++i) {
                    if (row[i] > m) m = row[i];
                }
                f->values[(size_t) l * bins + b] = m;
            }
        }
        return f;
    }

    // sparse: a partial selection of the top_k strongest cells.
    struct hit { float v; int32_t l, n; };
    std::vector<hit> hits;
    hits.reserve((size_t) a.n_layers * 8);
    for (int l = 0; l < a.n_layers; ++l) {
        const float * row = a.rows.data() + (size_t) l * a.n_ff;
        for (int i = 0; i < a.n_ff; ++i) {
            if (row[i] > 0.0f) hits.push_back({row[i], l, i});
        }
    }
    const size_t k = (size_t) cfg.top_k < hits.size() ? (size_t) cfg.top_k
                                                      : hits.size();
    std::partial_sort(hits.begin(), hits.begin() + k, hits.end(),
                      [](const hit & x, const hit & y) { return x.v > y.v; });
    f->cells.reserve(k * 2);
    f->cell_values.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        f->cells.push_back(hits[i].l);
        f->cells.push_back(hits[i].n);
        f->cell_values.push_back(hits[i].v);
    }
    return f;
}

// ------------------------------------------------------------- the callback

struct ctx {
    streamer * str = nullptr;
    accum      acc;
    int        n_layers = 0;      // filled from the model at startup
};

inline bool eval_callback(ggml_tensor * t, bool ask, void * user_data) {
    auto * c = (ctx *) user_data;
    if (!c || !c->str || !c->str->cfg.enabled) return false;

    // The dense down projection is "ffn_down-N" in older llama.cpp and
    // "ffn_out-N" in current builds (the same names cett-dump accepts). Only
    // the matmul itself is wanted: with a bias, the named node can be the add.
    const char * name = ggml_get_name(t);
    int layer = -1;
    for (const char * pre : {"ffn_down-", "ffn_out-", "ffn_down_out-"}) {
        layer = parse_layer(name, pre);
        if (layer >= 0) break;
    }
    if (layer >= 0 && t->op != GGML_OP_MUL_MAT) layer = -1;
    const bool moe = layer < 0;
    if (moe) layer = parse_layer(name, "ffn_moe_down-");

    if (ask) return layer >= 0;
    if (layer < 0) return true;

    // Nothing is watching: skip the device copy entirely. This is what keeps
    // the flag free to leave on.
    if (!c->str->has_subscribers()) return true;

    const ggml_tensor * act = moe ? t->src[1] : t->src[1];
    if (!act) return true;

    // Generation decodes one token at a time. A multi-token batch is prefill
    // or a parallel slot; either way it is not a frame.
    const int64_t n_tok = moe ? act->ne[2] : act->ne[1];
    if (n_tok != 1) return true;

    const int n_ff  = (int) act->ne[0];
    const int n_embd = (int) t->ne[0];

    if (c->acc.n_layers != c->n_layers || c->acc.n_ff != n_ff || layer == 0) {
        c->acc.reset(c->n_layers, n_ff);
    }
    if (layer >= c->acc.n_layers) return true;

    std::vector<float> out, in;
    if (!fetch_f32(t, c->acc.scratch, out)) return true;
    if (!fetch_f32(act, c->acc.scratch, in)) return true;

    double acc2 = 0.0;
    for (int i = 0; i < n_embd; ++i) acc2 += (double) out[i] * out[i];
    const float inv = 1.0f / ((float) sqrt(acc2) + 1e-8f);

    float * dst = c->acc.rows.data() + (size_t) layer * n_ff;
    for (int i = 0; i < n_ff; ++i) dst[i] = fabsf(in[i]) * inv;
    c->acc.filled[layer] = 1;

    if (c->acc.complete()) {
        c->str->publish(reduce(c->acc, c->str->cfg, c->str->next_index(),
                               &c->str->clf));
        std::fill(c->acc.filled.begin(), c->acc.filled.end(), 0);
    }
    return true;
}

// ---------------------------------------------------------------- JSON out

inline std::string to_json(const frame & f, const std::string & token) {
    // Keys match viz/stream.py: i index, s score, t tier, l layers, v/c data.
    // `s` is the classifier output when one is loaded and the peak activation
    // otherwise; `scored` says which, so the client never mistakes activity
    // for a flag.
    std::string s = "{\"i\":" + std::to_string(f.index)
                  + ",\"s\":" + std::to_string(f.scored ? f.score : f.peak)
                  + ",\"scored\":" + (f.scored ? "true" : "false")
                  + ",\"p\":" + std::to_string(f.peak)
                  + ",\"l\":" + std::to_string(f.n_layers);
    if (!token.empty()) {
        std::string esc;
        for (char ch : token) {
            if (ch == '"' || ch == '\\') { esc += '\\'; esc += ch; }
            else if ((unsigned char) ch < 0x20) { esc += ' '; }
            else esc += ch;
        }
        s += ",\"tok\":\"" + esc + "\"";
    }
    if (f.kind == tier::sparse) {
        s += ",\"t\":\"sparse\",\"c\":[";
        for (size_t i = 0; i < f.cell_values.size(); ++i) {
            if (i) s += ",";
            s += "[" + std::to_string(f.cells[i * 2]) + ","
               + std::to_string(f.cells[i * 2 + 1]) + ","
               + std::to_string(f.cell_values[i]) + "]";
        }
        s += "]";
    } else {
        s += f.kind == tier::binned ? ",\"t\":\"binned\",\"v\":["
                                    : ",\"t\":\"raw\",\"v\":[";
        const int cols = f.n_cols;
        for (int l = 0; l < f.n_layers; ++l) {
            if (l) s += ",";
            s += "[";
            for (int i = 0; i < cols; ++i) {
                if (i) s += ",";
                s += std::to_string(f.values[(size_t) l * cols + i]);
            }
            s += "]";
        }
        s += "]";
    }
    return s + "}";
}

} // namespace ns
