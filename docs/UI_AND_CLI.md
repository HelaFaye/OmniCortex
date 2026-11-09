# Every feature: in the UI, on the command line, and for agents

After `./install.sh`, nothing needs a terminal: start **OmniCortex** from the
app menu (or `./omnicortex`, or `OmniCortex.bat` on Windows) and Studio opens
in your browser. Everything below is also scriptable, for people who prefer a
shell and for automation; and the third column is what an AI agent (Cline,
Claude Desktop) can do through OmniCortex's MCP server ([CONNECT.md](CONNECT.md)).

Every page has the same sidebar on the left (a bar along the bottom on a
phone, with the rest under **More**). Related pages share a group: hover a
group to see its pages, and inside one, tabs along the top switch between
them. A group's icon opens the page you last used in it.

| Sidebar | What it is |
|---|---|
| **Chat** | chats with your models |
| **Models**: My Models, Discover | the models in your folders (load, settings, measured accuracy); finding and downloading models from Hugging Face |
| **Docs**, **Tools** | document collections replies can cite; MCP tool servers |
| **Tasks**: Jobs, Projects, Workers | longer runs; projects split into tasks for worker models; where those tasks run |
| **Studio**: Features, Review, Lab | the analysis tools: feature dictionaries, circuits, emotion vectors and steering; neuron behaviour over time; labs and 3D views |
| **Connect**: Apps, Remote hosts, Devices | using Studio from other apps; models on other computers; pairing devices and linking Studios |
| **System**, **Setup**, **Settings** | what runs and restarting it, monitoring, automation; paths, llama.cpp, hardware and health check; themes, workspaces, models per subject, tool approval, plugins |

Wherever a file or folder is asked for, **Browse…** opens a picker on the
computer Studio runs on, and lists of folders are edited one row at a time.

Jobs on the Jobs page are the same scripts as the command-line column, built
from typed forms: no free-form command line, values cannot smuggle in flags,
each run keeps its log, and the form's fields are tested against the script's
own options.

The terminal client (`scripts/oc.py`, `./omnicortex cli`) and the Godot
client (`./omnicortex godot`) reach a running Studio, here or on another
machine; [CLIENTS.md](CLIENTS.md) compares the three.

## Setting up

| feature | in the UI | command line | MCP tool |
|---|---|---|---|
| Start Studio and open it | app menu → OmniCortex | `./omnicortex`, `python scripts/launch.py` | — |
| Models folder, llama-server, vLLM, cett-dump, unloading, routing thresholds, workers and jobs, document search, MCP config | **Setup → Settings** | `~/.omnicortex/config.json`, or Studio's flags | — |
| Listen address, port, token, TLS, pairing limits | **Setup → Settings → Advanced** (refuses an address Studio could not start on) | `--host`, `--port`, `--token-file`, `--tls-cert/--tls-key`, `--pair-*` | — |
| Build llama.cpp with cett-dump | **Setup → Build**, or Jobs → Setup | `scripts/build_llama_tools.sh --backend vulkan` | `start_job build_llama` |
| Health check (packages, GPUs, build, model, pipeline) | **Setup → Health check** | `python scripts/doctor.py` | `doctor` |
| Hardware for worker models (ROCm, Vulkan, CUDA, Metal, CPU) | **Setup → Hardware** | `~/.omnicortex/hardware.json`, `python scripts/accelerators.py` | `devices` |
| Restart Studio | **Setup → Restart Studio** | stop and start it | — |
| Where models, llama.cpp, pipeline data and Studio's state live; switch layouts by name | **Settings → Workspaces** | `python scripts/oc_workspace.py`, `--workspace` on Studio, the launcher and doctor.py | — |
| The model `auto` uses for a subject | **Settings → Models per subject** | `subject_models` in `config.json` | — |
| When chat tool calls are allowed, asked or denied (rules, kinds, argument regexes, limits) | **Settings → Tool approval**; *always…* on an approval card | `python scripts/tool_rules.py` | — |
| What each feature needs (packages, programs, llama.cpp backends, drivers, devices, memory), with fixes for this OS | **Setup → Requirements** | `python scripts/oc_requirements.py` | `requirements` |
| Install a missing Python package | **Setup → Requirements → Install**, or Jobs → Setup | `python scripts/oc_requirements.py install <pkg>` | `start_job install_package` |
| Environment snapshots and what changed | **Setup → Requirements → snapshots** | `oc_requirements.py snapshot`, `diff`, `freeze` | `start_job env_snapshot` |

Network exposure (binding beyond this computer, tokens, TLS, remote jobs) is
deliberately not on the Setup page: set it in `config.json` or with flags
([SECURITY.md](SECURITY.md)).

## Using models

| feature | in the UI | command line | MCP tool |
|---|---|---|---|
| Find and download models (Hugging Face) | Studio → **Models** | `hf download …` | — |
| Load, unload, per-model settings | Studio → **Models** | `POST /api/load`, `/api/unload` | `load_model`, `unload_model` |
| Engine (llama.cpp or vLLM) and its options: flash attention, KV cache, speculative decoding, context shift, devices, CPU placement, environment ([ENGINE.md](ENGINE.md)) | Studio → **Models** → a model → **Engine options** | `python scripts/engine_options.py`; `GET /api/engine/options` | — |
| Chat (images, saved chats, presets) | Studio → **Chats** | `/v1/chat/completions` (any OpenAI client) | `chat` |
| `auto` routing by measured skill | model picker → auto | `"model": "auto"`; `python scripts/model_stats.py rank "…"` | `route_prompt` |
| Check a reply for hallucination risk, 3D view of it | chat → **check** | `POST /api/trace` | `check_reply` |
| Chat with documents (RAG) | Studio → **Docs** | `python scripts/rag.py` | — |
| MCP tools inside chat | Studio → **Tools** | `~/.omnicortex/mcp.json`, `python scripts/mcp_client.py` | — |
| Other devices and linked machines | **Connect → Devices** | `/api/pair/*`, `/api/links/*` | — |
| Computers running LM Studio, Ollama or llama.cpp without OmniCortex (remote hosts): pair, models, load/unload, send and install models | **Remote** | the agent: `omnicortex-remote` ([REMOTE.md](REMOTE.md)); `/api/remote/*` | `remote_hosts`, `remote_call` |
| Cline, Claude Desktop, OpenAI apps | **Connect** | `python scripts/oc_connect.py` | — |

## Measuring and improving models

| feature | in the UI | command line | MCP tool |
|---|---|---|---|
| TestQA (graded, per subject) | **Jobs → Evaluate** | `python scripts/testqa.py` | `start_job testqa` |
| Per-model, per-subject results | Studio → Models (stats) | `python scripts/model_stats.py show` | `model_stats` |
| SWE-bench, LiveBench, CLIP_benchmark | **Jobs → Benchmarks** | `python scripts/benchmarks.py`, `clip_bench.py` | `start_job …` |
| Retraining on failures (data, fine-tune, merge) | **Jobs → Retrain** | `deficits.py`, `finetune.py`, `merge_export.py` | `start_job …` |
| A whole retraining cycle: baseline, data, one or more layer arms, holdout, verdict | **Jobs → Retrain → Retraining cycle** | `python scripts/retrain_cycle.py` | `start_job retrain_cycle` |
| Find and remove a concept's direction from the weights (norm-preserving, KL damage meter, optimized) | **Jobs → Retrain → Concept edit** | `python scripts/concept_edit.py` ([CONCEPT_EDITING.md](CONCEPT_EDITING.md)) | `start_job concept_edit` |
| Adapt to a task while protecting a subspace (SAGE) | **Jobs → Retrain → SAGE** | `python scripts/sage_edit.py train / chat / export` | `start_job sage_train` |
| Share results with a scoreboard (opt-in; what is sent is listed and previewed) | **Settings → Sharing results** | `python scripts/score_cards.py`, `scripts/scoreboard_server.py` ([SHARING.md](SHARING.md)) | — |
| Plugins: turn on, open their pages; their jobs and reply checks | **Settings → Plugins**, Jobs → *Plugins*, *plugins* under a reply | a plugin's own scripts ([PLUGINS.md](PLUGINS.md)) | — |
| Where project tasks run: workers, endpoints, routing by subject, pause/drain, GPU temperatures | **Workers** (`/workers`) | `workers.json`; the API in [WORKERS.md](WORKERS.md) | — |
| How sure the model was, per token | chat: tick **confidence** | — | — |
| Logit lens over a reply (Hugging Face models) | Jobs → **Logit lens over a reply** | `python scripts/logit_lens.py` ([CONFIDENCE.md](CONFIDENCE.md)) | `start_job logit_lens` |
| Compare decoding variants (SLED, adaptive steering) | Jobs → **Decoding lab**, then TestQA and the gate | `python scripts/decode_lab.py` ([DECODING.md](DECODING.md)) | `start_job decode_lab` |
| Keep or reject a retrained model (holdout, per subject, hallucination) | **Jobs → Retrain → Keep or reject a retrain** | `python scripts/retrain_gate.py` | `start_job retrain_gate` |
| Compare models, item by item; does suppression help | **Jobs → Analysis** | `merge_eval.py`, `compare_models.py` | `start_job …` |
| Neuron review: which neurons go with wrong answers, by source, subject and time; 3D over time | **Review** | `python scripts/neuron_review.py` | `review_sources`, `review_summary`, `review_label` |
| Add TestQA runs and other benchmarks to the review | **Jobs → Review** | `neuron_review.py ingest-testqa`, `ingest-items` | `start_job review_testqa`, `review_items` |
| Mark a checked chat reply right or wrong | chat → check → *right / wrong* | `POST /api/review/label` | `review_label` |
| Feature dictionaries: record activations, train, describe, map onto neurons | **Features → Start here** (each step opens its job) | `python scripts/features.py collect / train / import / interpret / neurons` | `start_job feat_*` |
| What a reply was made of: features, emotions and logit lens per token | **Features → Read a reply** | decoding lab `POST /ns/inspect` | |
| Attribution graph, interventions, what drives a completion | **Features → Circuits** | `python scripts/feature_circuits.py graph / intervene / attribute` | `start_job feat_graph` |
| Emotion vectors: stories, vectors, trace, scan, sweep | **Features → Emotions** | `python scripts/emotions.py stories / build / trace / scan / sweep` | `start_job emo_*` |
| Steer with features and emotions; as a decoding-lab variant; as a llama.cpp control vector | **Features → Steer** | `decode_lab.py --variant`, `emotions.py steer / cvec` | `start_job emo_steer`, `emo_cvec` |
| A project's own requirements (toolchain, libraries, packages, hardware), inferred from its build files, checked before Start | **Projects → Requirements** | `python scripts/oc_requirements.py project <dir>` | `project_requirements`, `set_project_requirements`, `scan_project_requirements` |
| Projects split by skill, worker models, review | **Projects** | `python scripts/director.py analyze`, `/api/projects` | `create_project`, `get_project`, `review_task`, … |

## Finding and editing H-Neurons

| feature | in the UI | command line | MCP tool |
|---|---|---|---|
| Pipeline stages 0–9 | **Jobs → H-Neurons** (in stage order) | the scripts in [PIPELINE.md](PIPELINE.md) | `start_job preflight`, `collect`, … |
| What each stage has produced | **Lab → Pipeline dashboard**; Setup → Health check | `python scripts/doctor.py` | `doctor` |
| Edit a GGUF, export a LoRA | **Jobs → H-Neurons** (stage 9) | `suppress_gguf.py`, `export_lora.py` | `start_job suppress_gguf` |
| Vision (CLIP/SigLIP), mmproj edits | **Jobs → Vision** | `clip_neurons.py`, `suppress_mmproj.py` | `start_job clip_*` |
| Detector checks (cross-validation, per task, where it spikes) | **Jobs → Analysis** | `cv.py`, `task_neurons.py`, `when.py`, `score_tokens.py` | `start_job …` |
| Selective merge, MoE expert sweep | **Jobs → Merge** | `merge_selective.py`, `sweep_experts.py` | `start_job …` |
| Quantization lab, scale sweep, adaptive tuning | **Lab** | `viz/quant_lab.py`, `viz/scale_sweep_lab.py`, `viz/adaptive_tuning_lab.py` | — |

## Looking inside

| feature | in the UI | command line | MCP tool |
|---|---|---|---|
| Live activations while generating | **Lab → Live** | `python viz/stream.py` | — |
| 3D replay of a trace or checked reply | **Lab → Replay**; chat → check → open 3D view | `python viz/bloom.py <trace>`; Godot client | `list_traces` |
| Timeline, explorer, weights (desktop windows) | **Jobs → Views** | `viz/timeline.py`, `viz/explore.py`, `viz/weights.py` | `start_job view_*` |
| Compare activation sessions | **Jobs → Views → Compare** | `python viz/compare.py` | `start_job compare` |
| Trace one sample for the 3D views | **Jobs → H-Neurons → Trace one sample** | `python scripts/trace_sample.py` | `start_job trace` |

## Moving models

| feature | in the UI | command line | MCP tool |
|---|---|---|---|
| Browser-to-browser encrypted transfer | **Lab → Model transfer** | `python viz/transfer_lab.py` | — |
| Resumable HTTPS transfer, watch mode | — | `python scripts/oc_transfer.py` | (`oc_transfer_mcp.py`) |
| Remote tuning worker | Lab → Adaptive tuning | `python scripts/tuning_worker.py` on the worker | — |

The worker side of a remote transfer or tuning run is started on the other
machine, by design: it opens a port there.

## Links between sections

When something on one page needs a step on another, the page links to that
section and option rather than naming it. Error messages and fixes that say
"Setup → Build llama.cpp", "Settings → Workspaces" or "Jobs → Evaluate"
turn into links wherever they are shown, and a link can open a job already
filled in. Addresses you can use yourself:

| address | opens |
|---|---|
| `/setup#build`, `#hardware`, `#health`, `#requirements`, `#snapshots`, `#paths` | that part of Setup |
| `/settings#workspaces`, `#subjects`, `#tools` | that part of Settings |
| `/jobs#kind=testqa&endpoint=m=http://127.0.0.1:7870/v1@MODEL&out=runs/t.json` | a job form, filled in (any field by name) |
| `/#models`, `/#tools`, `/#docs`, `/#hub`, `/#chats` | a Studio tab |
| `/#model=MODEL&field=classifier` | a model's settings, at that field |

`doctor.py` prints the matching Studio address after the first problem it
finds.

## Long text and known values

Fields and tables never spill text past their edges:

- **Cut text has a tooltip.** A table cell, link, button, note or label too
  narrow for its text shows all of it on hover. A long unbroken value in a
  cell, such as a path, id or hash, is cut with … instead of widening the
  table. A text field whose value is longer than the box adds the whole value
  to its tooltip, after the field's help. This works on every page, including
  for content a page adds later (`nsFit` in `viz/ui_links.py`).
- **Help sits in tooltips.** A job field's help is in its ⓘ tooltip and shows
  under the field while it has focus, so it never crowds the form.
- **Known values come as dropdowns.** You can still type anything:
  - **Job forms** list your models, the Studio endpoint for each model
    (`label=URL@model`, for TestQA), finished TestQA results, H-Neuron sets
    and the workspace's data files.
  - **Engine device fields** (devices, draft device) list what
    `llama-server --list-devices` reports, with free memory.
  - **Project tasks** pick skills and prerequisite tasks as toggle chips, not
    typed lists.
