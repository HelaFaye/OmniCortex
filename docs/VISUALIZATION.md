# Visualization and live tracing

## Dashboards

Two frontends over one record format, because no single toolkit does both jobs.

**Web dashboard** (`viz/server.py`) -- stdlib only, no install. It has no
authentication, so it binds to loopback; reach it remotely through an SSH
tunnel or the hub. Pipeline state read from the filesystem, layer map, and a live
alpha sweep driver. This is the remote half.

```bash
python viz/server.py --root . --port 7860
ssh -L 7860:127.0.0.1:7860 gpu-host          # from another machine
```

**Fastplotlib explorer** -- for the 2D/3D activation map. Fastplotlib sits on
Pygfx/WGPU and is built for exactly this shape of data; a 32x14336 map is 458k
points, comfortably inside its envelope, and even integrated GPUs' Vulkan
support is sufficient. Of the four candidates: Open3D is for point clouds and meshes, not
array-shaped scientific data; Piviz-3d I could not verify exists. Pygfx is the
right rendering layer and Fastplotlib is the right API on top of it.

```bash
python viz/explore.py runs/model-q6 --h-neurons models/h_neurons.json
python viz/explore.py runs/model-q6 --dump prepared.npz   # headless
```

Four panels: mean map, contrast (mean of hallucinated minus mean of correct),
a MIP volume over samples x layers x neurons, and per-layer profiles with
H-Neuron counts overlaid. Click a 2D panel to print the layer, neuron and value.

The contrast panel is the one worth looking at. Your H-Neurons should appear
there as bright columns; if the classifier found neurons the contrast map does
not corroborate, that is a warning about the neuron set, not about the plot.

All array preparation is in module-level functions that never import
fastplotlib, so `--dump` works headless with no GPU and any other frontend can
reuse them. The neuron axis is **max**-pooled for the volume, not mean-pooled:
H-Neurons are under 0.1% of neurons and averaging 14336 columns erases them.

The catch is that Fastplotlib renders locally and has no native remote mode --
its Jupyter backend is the only remote path and it is clunky. Hence the split:
dashboard for remote monitoring, Fastplotlib for local exploration, both
reading the same records.

**Record format** (`viz/records.py`) -- a directory, not a single file,
so it is append-only, resumable and survives a crash mid-run. Deliberately not
HDF5 or a database: a directory of `.npz` opens in numpy on any platform with no
driver and no version pinning, and you can rsync half of one off a remote box
while it is still being written.

```python
from viz.records import Recorder, Session
with Recorder("runs/model-q6", meta) as r:
    r.add(qid, agg=cett_aggregate, tokens=ids, scores=per_token, verdict="wrong")
    r.event("sweep", alpha=0.5, correct=182, wrong=11)
```

Sessions are keyed by model fingerprint **and quantization**. Two sessions from
the same model at different quants have identical shapes and different meanings,
so a shape check alone would not catch the mistake; `merge_check` compares both
and refuses.

---


## Which view when

Start from the question you have, not the tool:

| question | view | how to get there |
|---|---|---|
| Did this reply hallucinate, and where? | **Studio check** | in chat, *check* on a reply (or turn on *check replies*); tokens are shaded by risk and the flagged ones underlined |
| What did the network do on those tokens? | **3D view** (`viz/bloom.py`, Godot) | *open 3D view* on a checked reply, or Replay in the hub; step token by token, `N` jumps to the next flagged token in Godot |
| Is this exact cell / layer really different? | **timeline** (`viz/timeline.py`) | the same trace directory; exact sizes, no glow, for deciding |
| Which neurons matter for this model at all? | explorer, weights | `viz/explore.py`, `viz/weights.py` over a profiling run |
| Is model A worse than model B? | comparison, graded stats | `viz/compare.py`; `model_stats.py rank` |

The usual path is the first three in order: **check** a reply in Studio, look
at the shaded text for the flagged span, **open the 3D view** on it to see
which layers lit up, and go to **timeline** only if you need to measure. Every
check is saved as a trace session under `~/.omnicortex/traces/<id>`, which
the hub's Replay picker, `bloom.py` and `timeline.py` all open.

The 3D views read the same way everywhere:

- **dim dots**: neurons that fire at least once in the reply; **bright**: firing on this token;
- **rings** (red-orange in the default `dark` theme): H-neurons firing on a flagged token;
- **risk strip** along the bottom: risk per token, the dashed line is the flag threshold, shaded columns are flagged tokens; click to jump;
- **reply text** under it: each token shaded by its risk, flagged ones underlined, the current one outlined; click a token to jump.

A checked reply's risk is a signal, not a verdict: the classifier is around
0.7 AUROC on whole replies. Treat a flagged span as "verify this", and an
unflagged reply as unflagged, not as correct.

## Visualization modes

| mode | shows | where |
|---|---|---|
| token trace | per-token classifier score, reasoning vs answer | `score_tokens.py` -> HTML |
| dashboard | pipeline state, per-layer H-Neuron bars, live alpha sweep | `viz/server.py`, remote |
| explorer | mean map, contrast map, 3D MIP volume, click-inspect | `viz/explore.py`, local GPU |
| comparison | depth profiles, concentration, failure overlap, tiered index diff | `viz/compare.py` |
| weights | magnitude, quantization error, H-Neuron enrichment | `viz/weights.py` |

### Time-resolved 3D

`scripts/trace_sample.py` captures CETT **per token** rather than aggregated
over a region. No change to cett-dump was needed: spans are arbitrary
`[start, end)` ranges, so one span per token gives `[tokens, layers, neurons]`.

```bash
python scripts/trace_sample.py --binary ... --gguf ... --tokenizer ... \
    --input_path data/consistency_samples.jsonl --qid <id> \
    --out runs/trace-<id> --bin-neurons 512 --classifier models/classifier.npz
python viz/timeline.py runs/trace-<id> --play
```

Size is why this is per-sample: one 500-token sequence on a 32x14336 model is
459 MB unbinned, about 16 MB at 512 bins. Binning max-pools, never means --
H-Neurons are under 0.1% of neurons and averaging erases them.

**What counts as hallucinating.** Each token gets a risk: the classifier's
probability that its activity looks like a hallucination (sigmoid of the
per-token logit). A token is **flagged when its risk reaches 0.5**, the
classifier's own decision boundary, so a clean reply flags nothing and a bad
one flags only the tokens that crossed. `--flag-prob` moves the threshold and
`--smooth N` averages risk over N tokens (the classifier was trained on span
means). `--relative-z Z` is the old behaviour, flagging tokens Z standard
deviations above the trace's own mean; it always flags something, even in a
clean reply, so it is a way of looking at shape, not a verdict, and the HUD
says "relative" when it is on.

On a flagged token, the **H-neuron cells** that are firing are ringed. The
cells come from an H-neuron profile: `--h-neurons models/h_neurons.json`, or,
when the trace was recorded with `--classifier`, the classifier's positive
weights stored in the trace. Without either, the token is still flagged in the
risk strip and text, but no cell is ringed, and the HUD says why. Columns are
ordered by classifier weight by default (`--order weight`), so the H-neurons
form a band at the left of every layer instead of scattering by neuron index;
`--order index` restores the raw layout.

The active threshold is global across the trace, not per frame. A per-frame
percentile would mark the same fraction active at every token and erase the
variation the animation exists to show.

### Three frontends, one backend

| frontend | renderer | for |
|---|---|---|
| `viz/timeline.py` | pygfx / WGPU | analysis. Exact sizes, no effects. |
| `plugins/godot-client/client/` | Godot 4.7 Forward+ | standalone desktop, real glow |
| `viz/bloom.py` | three.js in a browser | remote, zero install |

Parity survives because the clients are thin. `viz/bloom.py` serves the API all
three consume:

    GET /api/meta     frames, cells, layers, labels, per-token risk, flagged tokens, mode
    GET /api/theme    the selected theme, from themes.json
    GET /api/trace    i32 T,N,L | i32 layer[N] | i32 x[N] (column position)
                      | f32 intensity[T][N] | u8 state[T][N]

The Godot client is verified against the Godot 4.7 source, not from memory --
which caught three bugs: `background_mode = 3` is `BG_CANVAS` (wanted
`BG_COLOR = 1`), `billboard_mode = 3` is `BILLBOARD_PARTICLES` (wanted
`BILLBOARD_ENABLED = 1`), and the HUD `Label` was parented to a `Node3D`
instead of a `CanvasLayer`, so it inherited no canvas transform.

State (0 idle, 1 active, 2 flagged) is decided **once, in Python**. The clients
only draw. One implementation of the thresholding, the H-Neuron mapping and the
scoring; three renderers. `tests/` asserts the byte offsets, state semantics,
point sizes and frame rate match across the Godot and three.js clients, so a
change to one that is not mirrored in the other fails.

It also keeps the GPU free: classification is CPU work done once at load, not
per frame on the device that is also running the model. Godot uses a MultiMesh
so the whole field is one draw call with half-resolution glow; three.js sends
positions once and streams only intensities.

```bash
python viz/bloom.py runs/trace-abc --host 0.0.0.0   # backend + web
OC_API=http://127.0.0.1:7880 godot --path plugins/godot-client/client                # desktop
python viz/timeline.py runs/trace-abc                # analysis
```

No trace yet? `python viz/bloom.py --demo` serves a synthetic one (a quiet
field whose designated neurons burst on an invented claim, "moved to Lyon").
Nothing in it was measured, and the HUD says so; it is for trying the clients.
A real trace comes from `scripts/trace_sample.py`.

**Running the Godot client** (any Godot 4.7 or newer, including a
`godot-git` build; the binary may be called `godot` or `godot4`):

```bash
python viz/bloom.py --demo                                  # terminal 1
OC_API=http://127.0.0.1:7880 godot --path plugins/godot-client/client         # terminal 2
OC_FRAME=67 OC_PAUSED=1 OC_API=... godot --path plugins/godot-client/client   # open on one token, paused
```

Space pauses, Left/Right step a token, N jumps to the next flagged token,
clicking the risk strip seeks, Esc quits. `OC_API` may include a path (Studio
serves each checked reply at `.../viz/<id>/`) and `OC_TOKEN` sends a bearer
token for a Studio that needs one. It runs the project
directly, no editor needed; opening `plugins/godot-client/client/project.godot` in the editor
works too. Forward+ needs Vulkan; on a GPU or driver without it, add
`--rendering-method gl_compatibility` (glow still works, a little softer).
Verified with the official 4.7.2 build on a software Vulkan driver.

Bloom is deliberately absent from `timeline.py`. It blurs neighbours together,
so a bright cell reads as a smear and you lose the spatial precision the view
exists for. Use the pretty ones to show people; use pygfx to decide anything.

**Themes** (`--list-themes`): `dark` (default: blue activations, red-orange
H-neurons on a dark field), `clinical`, `ember`, `cool`, `mono`. Each sets
colormap, background, the two accent colours, and how state maps to point size
and opacity. Pygfx has no bloom or glow post-processing, so there are no shader
effects -- at a few hundred thousand points, size and opacity read better than
a glow would anyway.

### Weight views

`viz/weights.py` reads the `ffn_down` tensors directly -- a fraction of a model
load -- and answers what activations cannot.

```bash
python viz/weights.py --gguf model-Q6_K.gguf --reference model-F16.gguf \
    --h-neurons models/h_neurons.json
python viz/weights.py --gguf ... --reference ... --dump w.npz   # headless
```

**Magnitude** is `||W[:, j]||` per neuron. A neuron that fires often but writes
weakly is not the same as one that fires rarely and writes hard; CETT folds the
two together and this separates them.

**Quantization error** is the relative per-neuron difference between two quants
of one model. Normalising by the reference norm is load-bearing: absolute error
correlates with magnitude at r=0.97 and would just redraw the magnitude map,
while the relative version sits at r=0.06.

**Enrichment** is the one worth running. It tests whether the H-Neurons fall in
the high-error tail more often than chance, against a permutation null rather
than an assumed distribution -- the values are neither independent nor normal,
so a parametric test would be quietly wrong. A positive result is a mechanistic
link between quantization level and hallucination, not a correlation between two
summary numbers. It is the version of "low quants hallucinate more" that can be
measured instead of repeated.

---


## Realtime activations

Four ways to get activations while you work:

| | how | available | latency |
|---|---|---|---|
| **A** | trace the finished response (`scripts/autotrace.py`) | **now** | one prefill behind |
| **B** | PyTorch hooks while the decoding lab generates (`scripts/decode_lab.py --watch-*`) | **now**, Hugging Face weights | per token, with emotions and features |
| **C** | `llama-cpp-python` + `cb_eval` via ctypes | no compile, fiddly | per token |
| **D** | patched llama-server (`llama-tools/server-activations/`) | **now**, one rebuild | per token |

The assumption worth dropping is that activations must arrive *with* the
tokens. They do not.

### A. The tracing proxy

`scripts/autotrace.py` sits in front of llama-server, forwards requests
untouched, streams the response back with no added latency, then traces the
completed text in the background and pushes frames to the viewer.

```bash
llama-server -m model.gguf --port 8080 &
python viz/stream.py --token-file viewer.token --host 0.0.0.0 --allow-plaintext &
python scripts/autotrace.py --upstream http://127.0.0.1:8080 \
    --binary ~/llama.cpp/build/bin/llama-cett-dump --gguf model.gguf \
    --tokenizer Qwen/Qwen3-8B --n-layers 36 \
    --publish http://127.0.0.1:7890 --port 8088
```

Point Cline or Studio at `:8088` instead of `:8080`; nothing downstream knows.
The request is forwarded unaltered, so **the traced run is the served run** --
a proxy that changed sampling would be visualising a different generation than
the one you read.

Cost is one extra prefill per traced response on whichever machine holds the
GGUF, which competes with the next generation for the GPU. So tracing is
skipped while a trace is already running, and `--trace-every N` samples instead
of tracing everything.

`viz/stream.py` gained `/api/push`, so a trace produced anywhere on the network
can drive the viewer.

### B. The decoding lab

`scripts/decode_lab.py` runs a Hugging Face model behind an OpenAI-compatible
API. With `--watch-*` it reads the model as each token is chosen and sends
the frame to the viewer before the token reaches the client:

```bash
python viz/stream.py --token-file viewer.token &
python scripts/decode_lab.py --model Qwen/Qwen2.5-0.5B-Instruct --port 8091 \
    --watch-cett --watch-emotions runs/emotions/m --watch-sae runs/features/m-r12 \
    --publish http://127.0.0.1:7890 --publish-token-file viewer.token \
    --record runs/traces
```

Each frame carries:

- **CETT** of every MLP neuron, max-pooled to `--watch-bins` per layer (256 by default);
- **emotions**: the top three, plus every emotion at least 0.5 standard deviations from its story mean, so the viewer's lanes stay continuous;
- **features**: the top five of each dictionary, relative to each feature's maximum, with your labels.

`--record` saves each reply as a trace session. The 3D replay, the timeline
and the Godot client open it like any other, and also show the emotion lanes
and the features at the current token. A request can skip watching with
`"omnicortex": {"watch": false}` or skip recording with `{"record": false}`.
It cannot choose the folder.

Cost on a CPU: 40 tokens of a 0.5B model with all three watched took 6 to 10
seconds, against about 4 unwatched. Without a classifier nothing is flagged;
the views show activity, emotions and features, not risk.

[FEATURES.md](FEATURES.md#watch-while-it-generates) covers training the
dictionaries and emotion vectors these read.

## D. The patched llama-server

`llama-tools/server-activations/` makes llama-server stream per-token CETT at
`GET /activations` (server-sent events). One self-contained header, a small
glue header, and four insertions, applied by a script; `PATCH.md` explains
each. Configuration comes from the environment, so `common/arg.cpp` is
untouched.

```bash
scripts/build_llama_tools.sh --server-activations     # or: apply_patch.py ~/llama.cpp, then rebuild
python llama-tools/server-activations/export_classifier_bin.py \
    models/classifier.npz models/classifier.bin --gguf model.gguf
OC_ACTIVATIONS=sparse OC_CLASSIFIER=models/classifier.bin \
  ~/llama.cpp/build/bin/llama-server -m model.gguf --parallel 1 --port 8080
python viz/stream.py --source http://127.0.0.1:8080 --token-file viewer.token
```

A test (`tests/test_llamacpp_integration.py`, needs `OC_LLAMA` pointing at a
patched build) checks every frame against PyTorch CETT on all layers, and the
streamed score against the classifier applied in PyTorch.

**It carries the classifier.** Without one the server can report activity but
cannot flag anything: "this neuron is busy" and "this token is likely
fabricated" are different claims, and only the second needs trained weights.
`export_classifier_bin.py` writes a flat float32 blob the server reads with no
parser, and the server refuses it if the length does not match the model.
Every frame carries `scored`; the client shows "peak (no classifier)" rather
than a flag when it is false.

**Scores are on the classifier's scale; raw values are not CETT.** The server
streams `|a| / ||layer output||` without the `||W[:, j]||` factor, since holding
dequantised down_proj weights in RAM is a large cost for a live view.
`--gguf` folds each column norm into its classifier coefficient instead, so
the score equals the classifier on full CETT. Multiply the streamed values by
the column norms if you need CETT itself.

**Idle costs nothing.** The callback returns before any device copy when no one
is subscribed, so the flag can stay on.

**Single slot only.** With `--parallel > 1` a batch interleaves tokens from
several sequences and the node carries no recoverable sequence id, so it
refuses to enable rather than emitting frames that mix conversations.

**Generation only.** Frames are emitted for single-token decodes. The first
generated token comes out of prompt processing, so N generated tokens give
N - 1 frames.

## Live streaming viewer

`viz/stream.py` relays frames from the patched server (or `--simulate`) to a
phone or SBC, adding auth, TLS and per-viewer backpressure.

```bash
python viz/stream.py --simulate --token-file viewer.token --host 0.0.0.0 --allow-plaintext
```

Bandwidth drives the design. Measured on a 36x14336 field:

| tier | per token | at 4 tok/s | for |
|---|---|---|---|
| raw | 2.5 MB | 10 MB/s | nothing; it is here as the baseline |
| binned | 92 KB | 360 KB/s | a desktop on the same switch |
| sparse | 0.7 KB | 2.7 KB/s | a phone, over LTE |

Reduction happens server-side and the client picks a tier, so a phone asking
for `sparse` and a desktop asking for `binned` share one generation. Binning
max-pools: mean pooling over 28 neurons would turn a 1.8 peak into 0.08.

Subscribers are independently backpressured with a bounded queue and drop
frames rather than blocking. A stalled phone must not stall the desktop, and
more importantly must not stall generation behind it.

Auth is the same bearer token or cookie as Studio, with a query-string fallback
because `EventSource` cannot set headers. TLS via `--tls-cert`, though on an
untrusted network a WireGuard or Tailscale tunnel beats a self-signed cert that
users learn to click through.

## Camera (browser 3D view)

| control | what it does | key |
|---|---|---|
| drag / right-drag / wheel | orbit, pan, zoom (manual) | |
| **Front**, **Side**, **Top** | glide to a fixed view | 1, 2, 3 |
| **Reset** | back to the starting view | R |
| **Orbit** | a slow automatic turn around the model | O |
| **Follow flags** | on every token with flagged hallucination neurons, glide to them; back to the overview after | F |

Any drag stops a glide in progress, so the camera never fights the hand.
