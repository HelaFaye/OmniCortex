// cett-dump: capture ffn_down inputs and outputs from a llama.cpp forward pass.
//
// This is the piece that lets OmniCortex do activation extraction without PyTorch,
// without ROCm, and from a quantized GGUF. On a 16GB card a Q6_K 9B model plus
// this callback fits comfortably, where bf16 in PyTorch does not.
//
// How it works. llama.cpp exposes ggml_backend_sched_set_eval_callback via
// llama_context_params::cb_eval -- the same mechanism llama-cvector-generator
// uses. Every graph node is offered to the callback; we accept the ones named
// "ffn_down-<il>". For that node:
//
// File layout:
//     "CETT" | u32 version | u32 n_tokens | i32 token_ids[n_tokens]
//     then, unless --tokenize-only, one aggregate block (see write_aggregate):
//     per span and layer, the mean (or --max) of the ffn_down input over the
//     span's tokens. Without spans the whole sequence is one span. Read it with
//     scripts/extract_activations_gguf.py read_aggregate().
//
//     t          = output of down_proj for layer il   [n_embd, n_tokens]
//     t->src[0]  = the down_proj weight (quantized)
//     t->src[1]  = input to down_proj                 [n_ff,   n_tokens]
//
// t->src[1] is exactly what the PyTorch forward hook captures, so the two
// extraction paths produce the same quantity. We dump the raw activations and
// the output norms; the weight column norms come from the GGUF on the Python
// side (extract_activations_gguf.py already reads and dequantizes ffn_down),
// which keeps this file small and keeps all the CETT arithmetic in one place.
//
// Deliberately prompt-only: we evaluate the full question+response sequence in
// a single decode and never generate. That is all the pipeline needs, and it
// means one forward pass per sample.
//
// Build: see CMakeLists.txt in this directory.
//
//   # phase 1: token ids only, no forward passes
//   ./llama-cett-dump -m model-Q6_K.gguf --tokenize-only
//       --manifest seqs.jsonl --outdir toks/
//
//   # phase 2: driver rewrites the manifest with token spans, then
//   ./llama-cett-dump -m model-Q6_K.gguf -ngl 99 --n-layers 32
//       --manifest seqs_spans.jsonl --outdir dumps/
//
// Two phases because the spans must be expressed in the tokenization that
// produces the activations. Reduction happens in-process: writing raw
// [n_tok, n_ff] per layer would be ~918MB per sample at n_ff=14336 x 32 layers,
// against ~1MB for the reduced result.
//
// The manifest is JSONL, one {"id": "...", "text": "..."} per line. The model
// is loaded once and every sequence is evaluated in that one process. This
// matters enormously on an iGPU: reloading an 8GB GGUF per sample would cost
// far more than the forward passes themselves.
//
// Untested against real weights -- llama.cpp's API churns, so expect to adjust
// names against the version you build. It is modelled closely on
// examples/eval-callback/eval-callback.cpp to minimise that risk.

#include "arg.h"      // common_params_parse moved here
#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <algorithm> // std::fill
#include <cmath>     // sqrt, fabsf
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <climits>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static const char * MAGIC = "CETT";
static const uint32_t FORMAT_VERSION = 1;

// cb_eval is fixed at context creation (llama_context_params::cb_eval); there
// is no setter afterwards. So the callback writes through a stable pointer and
// run_one swaps what it points at between sequences.
struct dump_ctx;
static dump_ctx * g_active = nullptr;

struct span_t { int32_t start; int32_t end; };

struct dump_ctx {
    std::ofstream out;
    int  n_records = 0;
    bool failed    = false;
    std::vector<uint8_t> scratch;

    // Aggregation happens here rather than on disk. Writing raw [n_tok, n_ff]
    // per layer is ~918MB per sample on a 32-layer model with n_ff=14336;
    // reducing over the token spans first brings that to ~1MB.
    std::vector<span_t> spans;
    int32_t n_layers  = 0;
    int32_t n_ff      = 0;
    int32_t n_experts = 1;   // 1 = dense; >1 routes on the expert axis
    bool    use_max   = false;
    // [span][layer][neuron], flattened
    std::vector<float> agg;
    std::vector<char>  seen;   // [span][layer], did any token land here

    // Counts are tracked per (span, layer, expert) because a token only reaches
    // the experts it was routed to: dividing by the span length would be wrong.
    std::vector<int32_t> counts;

    // llama_decode splits a batch into ubatches (-ub, default 512) and the
    // callback fires once per layer PER UBATCH, with token indices local to
    // that ubatch. Spans are in sequence positions, so the ubatch offset has
    // to be tracked here. Without it, every token past the first ubatch was
    // dropped and early spans were overwritten with later tokens.
    int32_t n_total    = 0;          // tokens in the sequence
    int32_t next_off   = 0;          // start of the next ubatch
    int32_t ub_off     = 0;          // start of the current ubatch
    int32_t ub_size    = 0;          // tokens in the current ubatch
    int32_t last_layer = INT32_MAX;  // sentinel: first callback opens ubatch 0
    int32_t pruned     = -1;         // layer that only computed output rows

    // --by-weight: blocks are found by their ffn_down weight, so a block that runs several times in one forward
    // pass (a looped model) is seen once per pass. occ[block] counts this ubatch's passes so far.
    std::vector<int32_t> occ;
    int32_t first_block = -1;        // the first block seen; it coming round again opens a new ubatch

    // Called before a layer is accumulated. Returns false if the layer's rows
    // are not one-per-token (llama.cpp computes the final layer's FFN only for
    // tokens that produce logits, usually just the last one) and must be
    // skipped rather than misattributed to the start of the sequence.
    // new_ub: -1 detect a new ubatch from the layer index wrapping (named nodes), 0/1 told by the caller
    // (--by-weight, where a looped model's index also falls between loops within one ubatch).
    bool begin_layer(int32_t layer, int32_t n_tok, int new_ub = -1) {
        if (new_ub < 0 ? layer < last_layer : new_ub == 1) {   // new ubatch
            ub_off   = next_off;
            ub_size  = n_tok;
            next_off += n_tok;
        }
        last_layer = layer;
        if (n_tok != ub_size) {
            if (pruned != layer) {
                fprintf(stderr, "cett-dump: layer %d has %d rows for a "
                        "%d-token ubatch (output-only); left unseen\n",
                        layer, n_tok, ub_size);
                pruned = layer;
            }
            return false;
        }
        return true;
    }

    // Intersection of span si with the current ubatch, in ubatch-local rows.
    bool local_range(size_t si, int32_t & a, int32_t & b) const {
        const int32_t s0 = spans[si].start;
        const int32_t s1 = spans[si].end < 0 ? n_total : spans[si].end;
        if (s0 < 0 || s1 > n_total || s1 <= s0) return false;
        const int32_t lo = s0 > ub_off ? s0 : ub_off;
        const int32_t hi = s1 < ub_off + ub_size ? s1 : ub_off + ub_size;
        if (hi <= lo) return false;
        a = lo - ub_off;
        b = hi - ub_off;
        return true;
    }

    void init(int32_t nl, int32_t nff, int32_t ne) {
        n_layers = nl; n_ff = nff; n_experts = ne;
        const size_t cells = (size_t) spans.size() * nl * ne;
        agg.assign(cells * nff, 0.0f);
        seen.assign(cells, 0);
        counts.assign(cells, 0);
    }

    // flat = ((span * n_layers + layer) * n_experts + expert) * n_ff + neuron
    inline size_t cell(size_t si, int32_t layer, int32_t expert) const {
        return ((si * (size_t) n_layers + layer) * (size_t) n_experts + expert);
    }
};

// "ffn_down-13" -> 13; anything else -> -1.
// Note ffn_down_exps-<il> (MoE) is deliberately NOT matched: a mixture model
// has per-expert neurons and the flat (layer, neuron) index the rest of the
// pipeline assumes does not describe it.
// MoE: llama.cpp's build_moe_ffn emits ffn_moe_down-<il>, computed with
// ggml_mul_mat_id. Its sources carry everything we need:
//   src[0]  expert weights   [n_embd, n_ff_exp, n_expert]
//   src[1]  activations      [n_ff_exp, n_expert_used, n_tokens]
//   src[2]  selected experts [n_expert_used, n_tokens]   (I32)
//   t       output           [n_embd, n_expert_used, n_tokens]
// So each (token, slot) pair already knows which expert produced it, and the
// three-axis (layer, expert, neuron) index falls out directly.
static int parse_moe_down_layer(const char * name) {
    static const char * prefix = "ffn_moe_down-";
    const size_t plen = strlen(prefix);
    if (strncmp(name, prefix, plen) != 0) return -1;
    const char * p = name + plen;
    if (*p == '\0') return -1;
    int layer = 0;
    for (; *p; ++p) {
        if (*p < '0' || *p > '9') return -1;
        layer = layer * 10 + (*p - '0');
    }
    return layer;
}

// Tensor naming varies by architecture. PR #20785 documents l_out-{il} on
// llama/qwen2/qwen3 but final_output-{il} on qwen3next and qwen3.5 -- and
// Ornith is Qwen3.5-based. The ffn_down node may be named differently there
// too, so try the variants rather than finding zero records and blaming the
// build.
static const char * FFN_DOWN_PREFIXES[] = {
    "ffn_down-", "ffn_out-", "ffn_down_out-", nullptr
};

static int parse_ffn_down_layer_any(const char * name) {
    for (int i = 0; FFN_DOWN_PREFIXES[i]; ++i) {
        const size_t n = strlen(FFN_DOWN_PREFIXES[i]);
        if (strncmp(name, FFN_DOWN_PREFIXES[i], n) != 0) continue;
        const char * p = name + n;
        if (*p == '\0') continue;
        int v = 0;
        bool ok = true;
        for (; *p; ++p) {
            if (*p < '0' || *p > '9') { ok = false; break; }
            v = v * 10 + (*p - '0');
        }
        if (ok) return v;
    }
    return -1;
}

// --by-weight L: match the down projection by its weight, "blk.<N>.ffn_down.weight", instead of by the node's
// name. Some architectures (NVIDIA's Dory fork) never name that node. Pass k of block N is recorded as layer
// k*L + N, so a looped model's repeated passes stay apart (the reader maps v back with v % L and v / L).
static int32_t g_by_weight = 0;

static int parse_blk_ffn_down(const char * name) {
    if (strncmp(name, "blk.", 4) != 0) return -1;
    const char * p = name + 4;
    int v = 0, digits = 0;
    for (; *p >= '0' && *p <= '9'; ++p, ++digits) v = v * 10 + (*p - '0');
    if (digits == 0 || strcmp(p, ".ffn_down.weight") != 0) return -1;
    return v;
}

static int by_weight_block(const ggml_tensor * t) {
    if (t->op != GGML_OP_MUL_MAT || t->src[0] == nullptr) return -1;
    return parse_blk_ffn_down(ggml_get_name(t->src[0]));
}

static int parse_ffn_down_layer(const char * name) {
    static const char * prefix = "ffn_down-";
    const size_t plen = strlen(prefix);
    if (strncmp(name, prefix, plen) != 0) {
        return -1;
    }
    const char * p = name + plen;
    if (*p == '\0') {
        return -1;
    }
    int layer = 0;
    for (; *p; ++p) {
        if (*p < '0' || *p > '9') {
            return -1;
        }
        layer = layer * 10 + (*p - '0');
    }
    return layer;
}

// Copy a tensor back to host. With a GPU backend the data lives in device
// memory, so this is required rather than optional.
static bool fetch(const ggml_tensor * t, std::vector<uint8_t> & buf) {
    const size_t nbytes = ggml_nbytes(t);
    buf.resize(nbytes);
    if (ggml_backend_buffer_is_host(t->buffer)) {
        memcpy(buf.data(), t->data, nbytes);
    } else {
        ggml_backend_tensor_get(t, buf.data(), 0, nbytes);
    }
    return true;
}

static bool to_f32(const ggml_tensor * t, std::vector<uint8_t> & raw,
                   std::vector<float> & out) {
    if (!fetch(t, raw)) {
        return false;
    }
    const int64_t n = ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), raw.data(), n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *) raw.data();
        for (int64_t i = 0; i < n; ++i) {
            out[i] = ggml_fp16_to_fp32(src[i]);
        }
    } else {
        // Activations are f32 or f16 in practice; a quantized activation would
        // mean the graph changed shape under us, so fail loudly.
        fprintf(stderr, "cett-dump: unexpected activation type %s for %s\n",
                ggml_type_name(t->type), ggml_get_name(t));
        return false;
    }
    return true;
}

// Reads one string field out of a flat JSON object. The manifest is written by
// our own driver, so a full parser would be overkill; this handles the escapes
// that driver can emit and rejects anything else loudly.
static bool json_field(const std::string & line, const char * key,
                       std::string & out) {
    const std::string needle = std::string("\"") + key + "\"";
    size_t p = line.find(needle);
    if (p == std::string::npos) return false;
    p = line.find(':', p + needle.size());
    if (p == std::string::npos) return false;
    p = line.find('"', p);
    if (p == std::string::npos) return false;
    ++p;
    out.clear();
    for (; p < line.size(); ++p) {
        char c = line[p];
        if (c == '\\' && p + 1 < line.size()) {
            char n = line[++p];
            switch (n) {
                case 'n':  out += '\n'; break;
                case 't':  out += '\t'; break;
                case 'r':  out += '\r'; break;
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case 'u': {
                    if (p + 4 >= line.size()) return false;
                    int cp = (int) strtol(line.substr(p + 1, 4).c_str(), nullptr, 16);
                    p += 4;
                    // UTF-8 encode; surrogate pairs are not emitted by the driver
                    if (cp < 0x80) out += (char) cp;
                    else if (cp < 0x800) {
                        out += (char) (0xC0 | (cp >> 6));
                        out += (char) (0x80 | (cp & 0x3F));
                    } else {
                        out += (char) (0xE0 | (cp >> 12));
                        out += (char) (0x80 | ((cp >> 6) & 0x3F));
                        out += (char) (0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default: out += n;
            }
        } else if (c == '"') {
            return true;
        } else {
            out += c;
        }
    }
    return false;
}

// Reads "spans": [[a,b],[c,d]] out of a manifest line. -1 as an end means
// "to the last token", which is how the driver expresses open-ended regions.
static bool json_spans(const std::string & line, std::vector<span_t> & out) {
    size_t p = line.find("\"spans\"");
    if (p == std::string::npos) return false;
    p = line.find('[', p);
    if (p == std::string::npos) return false;
    const size_t close = line.find(']', line.rfind("]", line.size()) );
    out.clear();
    size_t q = p + 1;
    while (true) {
        size_t a = line.find('[', q);
        if (a == std::string::npos) break;
        size_t b = line.find(']', a);
        if (b == std::string::npos) break;
        const std::string body = line.substr(a + 1, b - a - 1);
        size_t comma = body.find(',');
        if (comma == std::string::npos) return false;
        span_t sp;
        sp.start = (int32_t) strtol(body.substr(0, comma).c_str(), nullptr, 10);
        sp.end   = (int32_t) strtol(body.substr(comma + 1).c_str(), nullptr, 10);
        out.push_back(sp);
        q = b + 1;
        size_t nxt = line.find_first_not_of(" \t", q);
        if (nxt == std::string::npos || line[nxt] != ',') break;
    }
    (void) close;
    return !out.empty();
}

// Accumulate one MoE layer. Unlike the dense path, a token contributes only to
// the experts it was routed to, so counts are tracked per (span, layer, expert)
// and the mean is taken over that expert's own observations.
static void moe_accumulate(dump_ctx * d, ggml_tensor * t, int layer) {
    const ggml_tensor * act = t->src[1];   // [n_ff_exp, n_used, n_tokens]
    const ggml_tensor * ids = t->src[2];   // [n_used, n_tokens] I32
    const ggml_tensor * w   = t->src[0];   // [n_embd, n_ff_exp, n_expert]
    if (!act || !ids || !w) {
        fprintf(stderr, "cett-dump: %s is missing MoE sources\n", ggml_get_name(t));
        d->failed = true;
        return;
    }
    if (ids->type != GGML_TYPE_I32) {
        fprintf(stderr, "cett-dump: expert ids are %s, expected I32\n",
                ggml_type_name(ids->type));
        d->failed = true;
        return;
    }

    const int32_t n_ff_exp = (int32_t) act->ne[0];
    const int32_t n_used   = (int32_t) act->ne[1];
    const int32_t n_tok    = (int32_t) act->ne[2];
    const int32_t n_embd   = (int32_t) t->ne[0];
    const int32_t n_expert = (int32_t) w->ne[2];

    if (!d->begin_layer(layer, n_tok)) return;
    if (d->agg.empty()) {
        d->init(d->n_layers, n_ff_exp, n_expert);
    }
    if (n_ff_exp != d->n_ff || n_expert != d->n_experts || layer >= d->n_layers) {
        fprintf(stderr, "cett-dump: MoE shape changed mid-run at layer %d\n", layer);
        d->failed = true;
        return;
    }

    std::vector<float> out, in;
    std::vector<uint8_t> raw;
    if (!to_f32(t, d->scratch, out) || !to_f32(act, raw, in)) {
        d->failed = true;
        return;
    }
    std::vector<uint8_t> idbuf;
    if (!fetch(ids, idbuf)) { d->failed = true; return; }
    const int32_t * eid = (const int32_t *) idbuf.data();

    for (size_t si = 0; si < d->spans.size(); ++si) {
        int32_t s0, s1;
        if (!d->local_range(si, s0, s1)) continue;

        for (int32_t j = s0; j < s1; ++j) {
            for (int32_t u = 0; u < n_used; ++u) {
                const int32_t e = eid[(size_t) j * n_used + u];
                if (e < 0 || e >= n_expert) continue;

                // ||output|| for this (token, slot), over the embedding axis
                const float * ocol = out.data()
                    + (((size_t) j * n_used) + u) * n_embd;
                double acc = 0.0;
                for (int32_t i = 0; i < n_embd; ++i) acc += (double) ocol[i] * ocol[i];
                const float inv = 1.0f / ((float) sqrt(acc) + 1e-8f);

                const size_t c = d->cell(si, layer, e);
                float * dst = d->agg.data() + c * (size_t) n_ff_exp;
                const float * icol = in.data()
                    + (((size_t) j * n_used) + u) * n_ff_exp;
                if (d->use_max) {
                    for (int32_t i = 0; i < n_ff_exp; ++i) {
                        const float v = fabsf(icol[i]) * inv;
                        if (v > dst[i]) dst[i] = v;
                    }
                } else {
                    for (int32_t i = 0; i < n_ff_exp; ++i) {
                        dst[i] += fabsf(icol[i]) * inv;
                    }
                }
                d->counts[c]++;
                d->seen[c] = 1;
            }
        }
    }
    d->n_records++;
}

static bool eval_callback(ggml_tensor * t, bool ask, void * user_data) {
    (void) user_data;
    dump_ctx * d = g_active;
    if (d == nullptr) {
        return false;   // between sequences: decline everything
    }

    const int block = g_by_weight ? by_weight_block(t) : -1;
    const int moe_layer = g_by_weight ? -1 : parse_moe_down_layer(ggml_get_name(t));
    int layer = g_by_weight ? block
                            : moe_layer >= 0 ? moe_layer : parse_ffn_down_layer_any(ggml_get_name(t));
    if (ask) {
        // Phase 1: llama.cpp asks whether we want this node's data.
        return layer >= 0;
    }
    int new_ub = -1;
    if (g_by_weight && block >= 0) {
        if (block >= g_by_weight) {
            fprintf(stderr, "cett-dump: blk.%d is beyond --by-weight %d\n", block, g_by_weight);
            d->failed = true;
            return true;
        }
        if ((int32_t) d->occ.size() != g_by_weight) d->occ.assign(g_by_weight, 0);
        new_ub = 0;
        if (d->first_block < 0) {
            d->first_block = block;
            new_ub = 1;
        } else if (block == d->first_block && d->occ[block] > 0) {
            std::fill(d->occ.begin(), d->occ.end(), 0);     // the first block again: the next ubatch
            new_ub = 1;
        }
        layer = d->occ[block]++ * g_by_weight + block;
    }
    if (moe_layer >= 0) {
        moe_accumulate(d, t, moe_layer);
        return true;
    }
    if (layer < 0 || d->failed) {
        return true;
    }
    if (t->ne[2] != 1 || t->ne[3] != 1) {
        fprintf(stderr, "cett-dump: unexpected rank for %s\n", ggml_get_name(t));
        d->failed = true;
        return true;
    }

    const ggml_tensor * act = t->src[1];   // input to down_proj: [n_ff, n_tok]
    if (act == nullptr) {
        fprintf(stderr, "cett-dump: %s has no src[1]\n", ggml_get_name(t));
        d->failed = true;
        return true;
    }

    std::vector<float> down_out, down_in;
    if (!to_f32(t, d->scratch, down_out) || !to_f32(act, d->scratch, down_in)) {
        d->failed = true;
        return true;
    }

    const int32_t n_embd = (int32_t) t->ne[0];
    const int32_t n_tok  = (int32_t) t->ne[1];
    const int32_t n_ff   = (int32_t) act->ne[0];

    if (act->ne[1] != t->ne[1]) {
        fprintf(stderr, "cett-dump: token count mismatch at layer %d\n", layer);
        d->failed = true;
        return true;
    }

    // Per-token L2 norm of the layer's down_proj output. Reducing here rather
    // than dumping [n_embd, n_tok] keeps the file to the activations alone.
    std::vector<float> norms(n_tok, 0.0f);
    for (int32_t j = 0; j < n_tok; ++j) {
        double acc = 0.0;
        const float * col = down_out.data() + (size_t) j * n_embd;
        for (int32_t i = 0; i < n_embd; ++i) {
            acc += (double) col[i] * (double) col[i];
        }
        norms[j] = (float) sqrt(acc);
    }

    if (!d->begin_layer(layer, n_tok, new_ub)) return true;
    if (d->agg.empty()) {
        d->init(d->n_layers, n_ff, 1);
    }
    if (n_ff != d->n_ff || layer >= d->n_layers) {
        fprintf(stderr, "cett-dump: layer %d / n_ff %d outside declared "
                        "%d / %d\n", layer, n_ff, d->n_layers, d->n_ff);
        d->failed = true;
        return true;
    }

    // We accumulate |a| / ||layer output||, NOT the full CETT. The weight
    // column norm w_j is a per-neuron constant, and both mean and max commute
    // with multiplication by a non-negative constant:
    //     mean_t(|a|*w_j/||o||) = w_j * mean_t(|a|/||o||)
    // so the Python side applies w_j once at the end, from the GGUF. This keeps
    // the weights out of this process entirely.
    for (size_t si = 0; si < d->spans.size(); ++si) {
        int32_t s0, s1;
        if (!d->local_range(si, s0, s1)) {
            continue;   // span not in this ubatch
        }
        float * dst = d->agg.data() + d->cell(si, layer, 0) * (size_t) n_ff;
        int32_t count = 0;
        for (int32_t j = s0; j < s1; ++j) {
            const float * col = down_in.data() + (size_t) j * n_ff;
            const float inv = 1.0f / (norms[j] + 1e-8f);
            if (d->use_max) {
                for (int32_t i = 0; i < n_ff; ++i) {
                    const float v = fabsf(col[i]) * inv;
                    if (v > dst[i]) dst[i] = v;
                }
            } else {
                for (int32_t i = 0; i < n_ff; ++i) {
                    dst[i] += fabsf(col[i]) * inv;
                }
            }
            count++;
        }
        // Division happens in write_aggregate, uniformly with the MoE path.
        d->counts[d->cell(si, layer, 0)] += count;
        if (count > 0) d->seen[d->cell(si, layer, 0)] = 1;
    }
    d->n_records++;

    return true;
}

// Evaluate one sequence and write its dump. Returns false on failure.
static bool write_aggregate(dump_ctx & dump, const std::string & out_path);

// llama.cpp computes the final layer's FFN only for tokens flagged as
// outputs. Flagging every token makes that layer complete, at the cost of an
// n_vocab x n_tokens logits buffer (~1 GB for 2k tokens at a 150k vocab).
// --last-layer-outputs-only restores the cheap behaviour (last layer unseen).
static bool g_all_outputs = true;

static bool run_one(llama_context * ctx, const common_params & params,
                    const std::string & text, const std::string & out_path,
                    const std::vector<span_t> & spans, int32_t n_layers,
                    bool use_max, bool tokenize_only) {
    dump_ctx dump;
    dump.spans    = spans;
    dump.n_layers = n_layers;
    dump.use_max  = use_max;
    dump.out.open(out_path, std::ios::binary);
    if (!dump.out) {
        fprintf(stderr, "cett-dump: cannot write %s\n", out_path.c_str());
        return false;
    }

    std::vector<llama_token> tokens = common_tokenize(ctx, text, true, true);
    if (tokens.empty()) {
        fprintf(stderr, "cett-dump: sequence tokenized to nothing\n");
        return false;
    }
    if ((int) tokens.size() > params.n_batch) {
        fprintf(stderr, "cett-dump: %zu tokens exceeds -b %d; raise the batch "
                        "size so the sequence is evaluated in one pass\n",
                tokens.size(), params.n_batch);
        return false;
    }

    // Header. Token ids are emitted so the Python side locates answer spans
    // against the tokenization that actually produced these activations.
    const uint32_t n_tokens = (uint32_t) tokens.size();
    dump.n_total = (int32_t) n_tokens;
    dump.out.write(MAGIC, 4);
    dump.out.write((const char *) &FORMAT_VERSION, sizeof(uint32_t));
    dump.out.write((const char *) &n_tokens, sizeof(uint32_t));
    {
        std::vector<int32_t> ids(tokens.begin(), tokens.end());
        dump.out.write((const char *) ids.data(), ids.size() * sizeof(int32_t));
    }

    if (tokenize_only) {
        // Phase 1: ids only, so the driver can compute token spans against the
        // tokenization that will produce the activations. No forward pass.
        dump.out.close();
        return true;
    }

    // Fresh KV state per sequence: these are independent samples, not a
    // conversation. Without this, position offsets accumulate across the run.
    llama_memory_clear(llama_get_memory(ctx), true);

    g_active = &dump;
    bool ok;
    if (g_all_outputs) {
        const int32_t n = (int32_t) tokens.size();
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int32_t i = 0; i < n; ++i) {
            batch.token[i]     = tokens[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;
        }
        batch.n_tokens = n;
        ok = llama_decode(ctx, batch) == 0;
        llama_batch_free(batch);
    } else {
        ok = llama_decode(
            ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size())) == 0;
    }
    g_active = nullptr;

    if (ok && !dump.failed && dump.next_off != dump.n_total) {
        fprintf(stderr, "cett-dump: saw %d of %d tokens across ubatches; "
                        "refusing to write a partial dump\n",
                dump.next_off, dump.n_total);
        dump.failed = true;
    }

    dump.out.close();
    if (ok && !dump.failed && !dump.agg.empty()) {
        if (!write_aggregate(dump, out_path)) {
            fprintf(stderr, "cett-dump: failed writing aggregate\n");
            return false;
        }
    }

    if (!ok) {
        fprintf(stderr, "cett-dump: llama_decode failed\n");
        return false;
    }
    if (dump.failed || dump.n_records == 0) {
        fprintf(stderr, "cett-dump: capture failed (%d records)\n", dump.n_records);
        return false;
    }
    return true;
}

// Aggregate record, appended after the token ids:
//   i32 n_spans | i32 n_layers | i32 n_ff | u8 seen[n_spans*n_layers]
//   f16 agg[n_spans][n_layers][n_ff]
static bool write_aggregate(dump_ctx & dump, const std::string & out_path) {
    std::ofstream f(out_path, std::ios::binary | std::ios::app);
    if (!f) return false;
    // Means are finished here, not in the callback: an expert's divisor is the
    // number of tokens routed to it, which is only known once the span is done.
    if (!dump.use_max) {
        for (size_t c = 0; c < dump.counts.size(); ++c) {
            if (dump.counts[c] <= 1) continue;
            float * p = dump.agg.data() + c * (size_t) dump.n_ff;
            const float inv = 1.0f / (float) dump.counts[c];
            for (int32_t i = 0; i < dump.n_ff; ++i) p[i] *= inv;
        }
    }
    const int32_t ns = (int32_t) dump.spans.size();
    f.write((const char *) &ns, sizeof(int32_t));
    f.write((const char *) &dump.n_layers, sizeof(int32_t));
    f.write((const char *) &dump.n_experts, sizeof(int32_t));
    f.write((const char *) &dump.n_ff, sizeof(int32_t));
    f.write((const char *) dump.counts.data(),
            dump.counts.size() * sizeof(int32_t));
    f.write((const char *) dump.seen.data(), dump.seen.size());
    std::vector<ggml_fp16_t> half(dump.agg.size());
    for (size_t i = 0; i < dump.agg.size(); ++i) {
        half[i] = ggml_fp32_to_fp16(dump.agg[i]);
    }
    f.write((const char *) half.data(), half.size() * sizeof(ggml_fp16_t));
    return f.good();
}

int main(int argc, char ** argv) {
    // Our flags are stripped before common_params_parse so we inherit
    // -m, -ngl, --device, -b and friends unchanged.
    std::string out_path = "dump.bin";
    std::string prompt_file, manifest_file, outdir;
    bool tokenize_only = false, use_max = false;
    int32_t n_layers_arg = 0;
    std::vector<char *> passthrough;
    passthrough.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            // Our flags first: llama.cpp's own --help lists hundreds of options and none of these.
            printf("cett-dump: capture ffn_down activations (CETT inputs) from a llama.cpp forward pass\n\n"
                   "  --prompt-file F      one sequence from a text file (a quick test)\n"
                   "  --out F              output for --prompt-file (default dump.bin)\n"
                   "  --manifest F         JSONL, one {\"id\",\"text\"[,\"spans\"]} per line; model loads once\n"
                   "  --outdir D           output directory for --manifest (one <id>.bin per line)\n"
                   "  --n-layers N         decoder layers in the model (required unless --tokenize-only)\n"
                   "  --tokenize-only      write token ids only, no forward pass\n"
                   "  --max                max over a span instead of the mean\n"
                   "  --by-weight L        find each block's down projection by its weight (blk.N.ffn_down.weight)\n"
                   "                       for models that do not name it; L = the GGUF's block count. Pass k of\n"
                   "                       block N is layer k*L+N (looped models), so --n-layers is L x passes\n"
                   "  --last-layer-outputs-only\n\n"
                   "Every other flag goes to llama.cpp (-m, -ngl, -b, --device, ...); its options follow.\n\n");
            passthrough.push_back(argv[i]);
            continue;
        }
        if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt-file") == 0 && i + 1 < argc) {
            prompt_file = argv[++i];
        } else if (strcmp(argv[i], "--manifest") == 0 && i + 1 < argc) {
            manifest_file = argv[++i];
        } else if (strcmp(argv[i], "--outdir") == 0 && i + 1 < argc) {
            outdir = argv[++i];
        } else if (strcmp(argv[i], "--tokenize-only") == 0) {
            tokenize_only = true;
        } else if (strcmp(argv[i], "--max") == 0) {
            use_max = true;
        } else if (strcmp(argv[i], "--last-layer-outputs-only") == 0) {
            g_all_outputs = false;
        } else if (strcmp(argv[i], "--by-weight") == 0 && i + 1 < argc) {
            g_by_weight = (int32_t) atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-layers") == 0 && i + 1 < argc) {
            n_layers_arg = (int32_t) atoi(argv[++i]);
        } else {
            passthrough.push_back(argv[i]);
        }
    }

    common_params params;
    if (!common_params_parse((int) passthrough.size(), passthrough.data(),
                             params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (prompt_file.empty() && manifest_file.empty()) {
        fprintf(stderr, "cett-dump: need --prompt-file or --manifest\n");
        return 1;
    }
    if (!manifest_file.empty() && outdir.empty()) {
        fprintf(stderr, "cett-dump: --manifest requires --outdir\n");
        return 1;
    }
    if (!tokenize_only && n_layers_arg <= 0) {
        fprintf(stderr, "cett-dump: --n-layers is required (run preflight.py)\n");
        return 1;
    }

    params.warmup            = false;   // a warmup pass would emit junk records
    params.cb_eval           = eval_callback;
    params.cb_eval_user_data = nullptr;  // routed via g_active, see dump_ctx

    llama_backend_init();
    llama_numa_init(params.numa);

    // Loaded exactly once, however many sequences follow.
    // common_init_from_params returns common_init_result_ptr (a unique_ptr)
    // and model/context are accessors, not fields, in current llama.cpp.
    auto init = common_init_from_params(params);
    if (!init || init->model() == nullptr || init->context() == nullptr) {
        fprintf(stderr, "cett-dump: failed to load model\n");
        return 1;
    }
    llama_context * ctx = init->context();

    int done = 0, failed = 0;

    if (!prompt_file.empty()) {
        std::ifstream f(prompt_file, std::ios::binary);
        if (!f) {
            fprintf(stderr, "cett-dump: cannot read %s\n", prompt_file.c_str());
            return 1;
        }
        std::string text((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
        std::vector<span_t> spans{{0, -1}};
        if (!run_one(ctx, params, text, out_path, spans, n_layers_arg,
                     use_max, tokenize_only)) return 1;
        done = 1;
    } else {
        std::ifstream f(manifest_file);
        if (!f) {
            fprintf(stderr, "cett-dump: cannot read %s\n", manifest_file.c_str());
            return 1;
        }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::string id, text;
            if (!json_field(line, "id", id) || !json_field(line, "text", text)) {
                fprintf(stderr, "cett-dump: bad manifest line, skipping\n");
                failed++;
                continue;
            }
            std::vector<span_t> spans;
            if (!tokenize_only && !json_spans(line, spans)) {
                spans.push_back({0, -1});   // whole sequence
            }
            const std::string path = outdir + "/" + id +
                                     (tokenize_only ? ".toks" : ".bin");
            // Resume: an existing dump is left alone so an interrupted run can
            // be restarted without redoing work.
            std::ifstream probe(path, std::ios::binary);
            if (probe.good()) { probe.close(); done++; continue; }

            if (run_one(ctx, params, text, path, spans, n_layers_arg,
                        use_max, tokenize_only)) {
                done++;
            } else {
                failed++;
                fprintf(stderr, "cett-dump: failed on %s\n", id.c_str());
                std::remove(path.c_str());
            }
            if (done % 25 == 0) {
                fprintf(stderr, "cett-dump: %d done, %d failed\n", done, failed);
            }
        }
    }

    fprintf(stderr, "cett-dump: finished, %d ok, %d failed\n", done, failed);
    llama_backend_free();
    return failed > 0 && done == 0 ? 1 : 0;
}
