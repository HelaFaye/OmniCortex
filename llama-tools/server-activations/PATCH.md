# Live activations from llama-server

`apply_patch.py` adds `GET /activations` to llama-server: one server-sent event
per generated token, carrying that token's down_proj activations (reduced
in-process) and, optionally, the H-Neuron classifier score.

```bash
python llama-tools/server-activations/apply_patch.py ~/llama.cpp
cmake --build ~/llama.cpp/build --target llama-server
# or both at once: scripts/build_llama_tools.sh --server-activations
```

The script is idempotent (each insertion is marked `ns-activations`) and checks
every anchor first. If llama.cpp has moved one, it stops and names it rather
than patching the wrong place; the sections below are then the manual recipe.
Tested against llama.cpp with the `tools/server/server-context.cpp` layout
(2026). Configuration is read from the environment, so `common/arg.cpp` is
left alone.

## What it changes

**Files added to `tools/server/`:** `oc_activations.h` (callback, reduction,
classifier, per-subscriber queues; no llama.cpp internals) and
`oc_server_glue.h` (the two globals and the SSE handler).

**1. `server-context.cpp`, before `common_init_from_params`:** with
`OC_ACTIVATIONS=sparse|binned|raw` set, put `ns::eval_callback` into
`params_base.cb_eval`. Refused with a warning when `--parallel > 1`: a batch
then interleaves several sequences and graph nodes carry no sequence id, so
frames would mix conversations. `OC_TOPK` and `OC_BINS` tune the reductions.

**2. `server-context.cpp`, after the model loads:** read the layer count, and
load `OC_CLASSIFIER` if set. The expected size is layers × `feed_forward_length`
from the GGUF metadata (`OC_NFF` overrides it); a classifier for another model
is refused.

**3. `src/llama-context.cpp`, before the graph-reuse check in
`process_ubatch`:** set the eval callback on every ubatch. **Without this the
callback fires once and then silently stops**, because the callback is only
installed when the graph is rebuilt and a server reuses it between decodes.
Found by hrhdegenetrix in llama.cpp PR #20785. cett-dump never hits this: it
evaluates one sequence per process.

**4. `server.cpp`:** register `GET /activations` beside `/metrics`. Each client
gets a bounded queue that drops frames rather than block the inference thread;
the subscription ends when the connection closes. A keepalive comment goes out
every ~10 s of silence. Without `OC_ACTIVATIONS` the route answers 503.
It is not a public endpoint, so llama-server's `--api-key` applies to it as to
`/completion`; frames reveal what is being generated, so expose it no more
widely than the completion API.

## Node names

The dense down projection is the `GGML_OP_MUL_MAT` node named `ffn_down-N` in
older llama.cpp and `ffn_out-N` in current builds, the same names cett-dump
accepts; `src[1]` is its input, the activation vector. MoE models use
`ffn_moe_down-N`.

## The classifier blob

```bash
python llama-tools/server-activations/export_classifier_bin.py \
    models/classifier.npz models/classifier.bin --gguf model.gguf
```

Flat little-endian float32 coefficients, then the intercept. The server streams
`|a| / ||layer output||`, which is CETT without the `||W[:, j]||` factor;
`--gguf` folds those column norms into the coefficients so the streamed score
equals the classifier applied to full CETT (the integration test checks this
against PyTorch). It is a logit: sigmoid it for a probability.

## Token text

The callback sees graph nodes, not tokens, so frames carry an index `i`, not
the token text. Correlate by order with the completion stream, or set a global
where the server detokenises the sampled token and pass it to `ns::to_json`.

## Prior art

llama.cpp PR #20785 (hrhdegenetrix, closed as out of scope for the server) does
the same thing at a different granularity: it captures `l_out`, the residual
stream after each layer, which is what sparse autoencoders want. This captures
`ffn_down` inputs, which is per-neuron and what H-Neurons needs. They are
complementary, not competing, and the branch is worth reading --
`hrhdegenetrix:activation-capture`, with a companion repo at
`llama-sae-feature-interpretability`.

Their measured overhead on an RTX 3090 with Qwen3-8B: ~140.7 tok/s disabled,
~131.8 enabled, about 6%. That matches the design assumption here that an
unsubscribed callback costs nothing.

## Running it

```bash
OC_ACTIVATIONS=sparse OC_TOPK=48 \
OC_CLASSIFIER=models/classifier.bin \
  ./build/bin/llama-server -m model.gguf --parallel 1 --port 8080

python viz/stream.py --source http://127.0.0.1:8080 --token-file viewer.token --host 0.0.0.0 --allow-plaintext
```

## What to expect

**Cost when idle is nothing.** The callback returns early if no one is
subscribed, before any device copy. Leaving the flag on is free.

**Cost when watching** is one `ggml_backend_tensor_get` per layer per token
plus a pass over `n_layers * n_ff` floats. On a 36x14336 model that is ~2 MB of
device-to-host copy per token. At 4 tok/s it is not the bottleneck; at 60 tok/s
on a fast card it might be, in which case use `OC_ACTIVATIONS=sparse` and
accept the top-k selection cost instead.

**Prefill is skipped.** Frames are only emitted when the batch is a single
token, which is generation. The first generated token comes out of prompt
processing, so N generated tokens give N - 1 frames.

**Raw values are not CETT; scores are.** Values lack the `||W[:, j]||`
factor (holding dequantised down_proj weights in RAM is a large cost for a live
view); the classifier blob carries it instead. `scored` in each frame says
whether a classifier was loaded; when it was not, `s` is peak activation, which
shows activity but flags nothing.
