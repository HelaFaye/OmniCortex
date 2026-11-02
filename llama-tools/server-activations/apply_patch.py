#!/usr/bin/env python3
"""
Apply the OmniCortex activation-streaming patch to a llama.cpp checkout.

    python llama-tools/server-activations/apply_patch.py ~/llama.cpp
    cmake --build ~/llama.cpp/build --target llama-server

Idempotent: every insertion is marked `ns-activations` and skipped if already
present. Targets the current server layout (tools/server/server-context.cpp,
handler-based routes in server.cpp); each anchor is checked and the script
stops with a clear message if llama.cpp has moved on, rather than patching
the wrong place. See PATCH.md for what each change does and why.

Then run:

    NS_ACTIVATIONS=sparse NS_CLASSIFIER=models/classifier.bin \\
        llama-server -m model.gguf --parallel 1
    curl -N http://127.0.0.1:8080/activations     # one SSE frame per generated token
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARK = "ns-activations"

GLUE = r'''#pragma once
// ns-activations: globals and the /activations route, shared by
// server-context.cpp (callback setup) and server.cpp (route).
#include "ns_activations.h"
#include "server-http.h"

#include <chrono>
#include <memory>
#include <string>
#include <thread>

inline ns::streamer g_ns_stream;
inline ns::ctx      g_ns_ctx;

inline server_http_res_ptr ns_activations_handler(const server_http_req & req) {
    auto res = std::make_unique<server_http_res>();
    if (!g_ns_stream.cfg.enabled) {
        res->status = 503;
        res->data = "{\"error\":\"start llama-server with NS_ACTIVATIONS=sparse|binned|raw and --parallel 1\"}";
        return res;
    }
    res->content_type = "text/event-stream";
    auto s = std::make_shared<ns::sink>(g_ns_stream.cfg.max_queue);
    g_ns_stream.subscribe(s);
    // Unsubscribes when the response (and with it this lambda) is destroyed.
    // Built in place: a temporary guard would unsubscribe as it is destroyed.
    struct guard {
        std::shared_ptr<ns::sink> s;
        explicit guard(std::shared_ptr<ns::sink> s_) : s(std::move(s_)) {}
        guard(const guard &) = delete;
        ~guard() { g_ns_stream.unsubscribe(s); }
    };
    auto g = std::make_shared<guard>(s);
    res->next = [s, g, &req](std::string & output) -> bool {
        output.clear();
        for (int i = 0; i < 500; ++i) {           // ~10 s, then a keepalive comment
            if (req.should_stop()) return false;
            for (int k = 0; k < 64; ++k) {
                auto f = s->pop();
                if (!f) break;
                output += "data: " + ns::to_json(*f, "") + "\n\n";
            }
            if (!output.empty()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        output = ": keepalive\n\n";
        return true;
    };
    return res;
}
'''

CONFIG_BLOCK = r'''        // --- ns-activations: register the eval callback before the context exists
        if (const char * ns_mode = getenv("NS_ACTIVATIONS")) {
            if (params_base.n_parallel > 1) {
                // Several slots interleave sequences in one batch, and graph nodes
                // carry no sequence id, so frames would mix conversations.
                SRV_WRN("%s", "NS_ACTIVATIONS ignored: needs --parallel 1\n");
            } else {
                g_ns_stream.cfg.enabled = true;
                g_ns_stream.cfg.reduction =
                    strcmp(ns_mode, "binned") == 0 ? ns::tier::binned :
                    strcmp(ns_mode, "raw")    == 0 ? ns::tier::raw    : ns::tier::sparse;
                if (const char * v = getenv("NS_TOPK")) g_ns_stream.cfg.top_k = atoi(v);
                if (const char * v = getenv("NS_BINS")) g_ns_stream.cfg.bins  = atoi(v);
                g_ns_ctx.str = &g_ns_stream;
                params_base.cb_eval           = ns::eval_callback;
                params_base.cb_eval_user_data = &g_ns_ctx;
            }
        }
'''

SETUP_BLOCK = r'''
        // --- ns-activations: layer count and optional classifier, now the model is known
        if (g_ns_stream.cfg.enabled) {
            g_ns_ctx.n_layers = llama_model_n_layer(model_tgt);
            if (const char * p = getenv("NS_CLASSIFIER")) {
                int n_ff = 0;
                if (const char * v = getenv("NS_NFF")) {
                    n_ff = atoi(v);
                } else {
                    char arch[128] = {0}, val[64] = {0};
                    if (llama_model_meta_val_str(model_tgt, "general.architecture", arch, sizeof(arch)) > 0) {
                        const std::string key = std::string(arch) + ".feed_forward_length";
                        if (llama_model_meta_val_str(model_tgt, key.c_str(), val, sizeof(val)) > 0) {
                            n_ff = atoi(val);
                        }
                    }
                }
                if (n_ff > 0 && g_ns_stream.clf.load(p, g_ns_ctx.n_layers * n_ff)) {
                    SRV_INF("ns activations scored by %s\n", p);
                } else {
                    SRV_WRN("ns activations: classifier %s not loaded (n_ff=%d); frames will be unscored\n", p, n_ff);
                }
            }
            SRV_INF("ns activations on: %d layers, GET /activations\n", g_ns_ctx.n_layers);
        }
'''

REUSE_FIX = ("    // ns-activations: set the eval callback on every ubatch, including when the\n"
             "    // previous graph is reused; otherwise it fires once and silently stops.\n"
             "    ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);\n\n")


def patch_file(path: Path, edits: list[tuple[str, str, str]]) -> None:
    """edits: (anchor, text, where) with where in {'before', 'after'}."""
    s = path.read_text()
    changed = False
    for anchor, text, where in edits:
        if text.strip() and text.strip().splitlines()[0] in s:
            continue                              # already applied
        if anchor not in s:
            raise SystemExit(f"{path}: anchor not found:\n    {anchor.strip()}\n"
                             "llama.cpp has changed here; apply PATCH.md by hand.")
        s = s.replace(anchor, (text + anchor) if where == "before" else (anchor + text), 1)
        changed = True
    if changed:
        path.write_text(s)
        print(f"patched {path}")
    else:
        print(f"already patched {path}")


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        raise SystemExit(__doc__)
    root = Path(argv[0]).expanduser().resolve()
    server = root / "tools" / "server"
    if not (server / "server-context.cpp").exists():
        raise SystemExit(f"{server}/server-context.cpp not found: older llama.cpp layout; apply PATCH.md by hand")
    shutil.copy(HERE / "ns_activations.h", server / "ns_activations.h")
    (server / "ns_server_glue.h").write_text(GLUE)
    print(f"wrote {server / 'ns_activations.h'} and ns_server_glue.h")

    patch_file(server / "server-context.cpp", [
        ('#include "server-context.h"\n', '#include "ns_server_glue.h" // ns-activations\n', "after"),
        ("        llama_init = common_init_from_params(params_base);\n", CONFIG_BLOCK, "before"),
        ("        vocab = llama_model_get_vocab(model_tgt);\n", SETUP_BLOCK, "after"),
    ])
    patch_file(server / "server.cpp", [
        ('#include "server-http.h"\n', '#include "ns_server_glue.h" // ns-activations\n', "after"),
        ('    ctx_http.get ("/metrics",', '    ctx_http.get ("/activations",              ex_wrapper(ns_activations_handler)); // ns-activations\n', "before"),
    ])
    patch_file(root / "src" / "llama-context.cpp", [
        ("    if (!graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams)) {\n", REUSE_FIX, "before"),
    ])
    print("done. Rebuild: cmake --build build --target llama-server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
