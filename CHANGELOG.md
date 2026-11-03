# Changelog

## Unreleased

### Added (integrated options, second batch)

- `quant_mask.py`: more precision for the layers with the most H-neurons, less for the safest, plus an importance matrix from mistake contexts.
- `model_lab.py jitter` (brittle layers under noise, DPO pairs from flipped items) and `model_lab.py hijack` (cross-layer back-patching, optionally gated by a feature direction).
- Neuron aging and decay: a served LoRA correction eases off when unused and returns at full strength when needed (off by default).
- `neuro_fuzz.py`: new, verified, deduplicated TestQA items aimed at the model's failures and risky replies.

### Added (integrated options, first batch)

- Fix This Mistake: a **fix** button under each reply keeps your correction as a training pair (and the reply's activations as a known mistake); an optional idle-time DPO micro-tune.
- Neural Reset: a reply caught repeating itself is cut at the loop and resumed with the loop pushed down.
- Prompt sanitizer for chats sent to linked hosts: secrets and listed names become placeholders and are restored in the reply.
- Core shunt: jobs run at low priority, on efficiency cores where the CPU has them.
- Dependency GC and dependency snapshots (diff, pip plan, sandbox venv; `/api/env/snapshot`, `/api/env/compare`).
- Settings → Integrated options; docs/INTEGRATED_OPTIONS.md says what each does and does not do.

### Added (unified review and browser extension)

- `scripts/unified_review.py` and Studio's `/api/lookup`: everything known about a model in one record, from review boards (verified accuracy, speed per hardware), hardware-profile reports, SAELens's catalogue and this Studio's observations. Re-packagings (GGUF, AWQ, …) are marked related.
- Cards may carry self-reported emotion and feature summaries (`score_cards.py --extras`); boards average them and every viewer marks them as not re-checked.
- `extension/`: a browser extension for Chromium browsers (Chrome, Edge, Opera, Brave, Vivaldi) and Firefox (desktop and Android). Its icon glows blue on model pages with data, and it has a filterable popup. CI packages it. Studio's `/#hub=org/name` opens a repo's files.

### Added (hardware support profiles)

- Profiles per hardware family, legacy and modern alike, with setups stored per purpose (inference, retraining, or general): `scripts/hw_profiles.py`, `profiles/hardware/`, Setup → Hardware support profiles, the `hw_apply` job.
- Share them through the review board: submit a profile for community review, report whether one worked; reports are aggregated in `board.json`. See docs/HARDWARE_PROFILES.md.

### Changed (the name)

- NeuronScope is now **OmniCortex**, because [RudrenduPaul/NeuronScope](https://github.com/RudrenduPaul/NeuronScope)
  already uses the name. The installers are `Install OmniCortex.bat` / `.command`, the Windows launcher is
  `OmniCortex.bat`, and `./omnicortex` joins `./neuronscope`. Data folders (`~/.neuronscope`), `NS_*` variables,
  HTTP headers and ComfyUI node ids are unchanged, so existing installs and saved workflows keep working.

### Added (binaries)

- `.github/workflows/binaries.yml`: llama.cpp tools and the remote agent as one executable for Windows (x64,
  x86, ARM64), macOS (Apple silicon, Intel) and Linux (x64, ARM64, x86, ARMv7), attached to releases
  ([docs/BINARIES.md](docs/BINARIES.md)).

### Added (MCP over the network)
- `ns_transfer_mcp.py --http`: the transfer MCP server over streamable HTTP
  for controllers on other computers.
  - A bearer token is always required, compared in constant time, and
    wrong guesses are throttled. Requests from other sites' web pages are
    refused.
  - A network address needs TLS or `--allow-plaintext`; behind `tailscale
    serve` it stays on 127.0.0.1.
  - The MCP session manager starts and stops with the server, and the
    hostname check is replaced by the token check, so a proxy's own
    hostname works. `GET /health` answers without the token.
  - Its tools carry read-only and destructive annotations, and its
    instructions say the tools act on the computer running it.
- `mcp.json`: an HTTP server's `token_file` or `token_env`, and
  `${file:…}` / `${env:…}` in header and env values, so tokens need not be
  copied into the file. They are read at each connect.
- Studio's Tools tab: **Test** per server. It lists the tools again and
  calls one read-only tool that needs no arguments.

### Fixed
- With the MCP library 2.x, Studio read no tool annotations (2.x names
  them `read_only_hint`…), so tool approval treated every tool as
  unannotated. Both spellings are read now.

### Added (newcomers)
- An optional tour of Studio, offered once on the first visit (*Show me
  around*, *Not now*, *Don't ask again*). It has eleven stops across
  Studio, Setup, Jobs, Review, Projects, Workers, Connect and Settings,
  each pointing at the real control.
  - Back and Next (or the arrow keys), and Esc to end it.
  - A ? button on every page restarts it, jumps to the current page's stop,
    or opens the help pages.

### Added (installing)
- A cross-platform installer wizard (`scripts/installer.py`, standard
  library only), opened by double-clicking `Install OmniCortex.bat`
  (Windows), `Install OmniCortex.command` (macOS) or `install-gui.sh`
  (Linux).
  - It finds a suitable Python first, and offers or explains how to get
    one.
  - It shows what it found and each platform's peculiarities.
  - It picks the PyTorch build per GPU (CUDA per generation, ROCm per AMD
    target, Metal, CPU, or none), installs with live output and advice on
    failure, smoke-tests the GPU, and adds a Start menu, Applications or
    app-menu shortcut.
  - `--yes` and `--check` run it in a terminal. `scripts/torch_smoke.py` is
    the shared GPU check.
- Batch files are stored with Windows line endings (`.gitattributes`).

### Added (roles and help)
- docs/ROLES.md: every connection with its controller and the side that
  does the work, every default port and whether it speaks MCP, and who may
  do what.
- The Connect page shows who controls whom.
- Studio's MCP instructions and `studio_status` name the computer the tools
  act on.
- `/help/<page>` renders OmniCortex's docs inside Studio.

### Added (work distribution)
- Workers page (`/workers`): monitor and control where project tasks run.
  - Workers are this computer's models, endpoints you run yourself (any
    OpenAI-compatible server, here or elsewhere), and remote hosts and
    linked Studios.
  - Pause, drain, per-worker parallel limits, the subjects each takes, a
    timed test, and stopping a local model.
  - Routing by subject: up to three workers in order.
  - The queue shows why each task waits; a task can be moved to another
    worker.
  - GPU temperatures, VRAM, load and power for this computer and each
    remote host, plus an activity log.
  - The director dispatches to all three kinds of worker. An endpoint that
    stops answering is marked down at once, so its tasks wait instead of
    failing. docs/WORKERS.md.
- OmniCortex Remote: GPU telemetry (nvidia-smi, rocm-smi, or Linux amdgpu
  sensors, without guessing) as the `telemetry` tool, and a thermal guard.
  Off until the host's owner turns it on; above a chosen limit the host
  takes no new work and unloads what it can, and it resumes once cooler.

### Added (confidence signals)
- Confidence strip in the chat (*confidence*, on by default). Studio asks
  llama-server or vLLM for each token's top-5 probabilities and shows where
  the model was unsure: spread (entropy of the top 5 plus the rest, over
  log 6) and the chosen token's probability. It is saved with the chat.
- `scripts/logit_lens.py` (Jobs → *Logit lens over a reply*). For Hugging
  Face models: each layer's guess for every token of a reply, its spread,
  the settle layer and the top-k decay layer. Tokens that settle late with
  a spread-out final layer are marked. Writes JSON and a heatmap page.
- `scripts/decode_lab.py` (Jobs → *Decoding lab*): one Hugging Face model
  served as several decoding variants on an OpenAI-compatible port, for
  TestQA and the gate to compare.
  - SLED (self logits evolution decoding).
  - Steering with a concept_edit direction, at a fixed strength or one that
    follows the model's uncertainty (output spread, or attention entropy).
  - docs/DECODING.md.
- docs/CONFIDENCE.md: the three signals side by side (confidence strip,
  logit lens, H-Neuron check), and what it would take to tell dishonesty
  from a confident mistake.

### Changed (Setup)
- Setup → Settings now holds every Studio flag worth a setting, in groups:
  models and engines (vLLM command, backend port), routing (stats
  window), activation checks, workers and jobs (hardware file, turning
  jobs off, jobs on the network), documents and tools (embedding
  endpoint and GPU layers, mcp.json). Network, security and pairing are
  under *Advanced*.
  - A listen address Studio would refuse at startup (no token, or HTTP
    without TLS and without plaintext allowed) is refused when saving, so
    the setting cannot lock you out.
  - Clearing a field returns it to the flag's default.
  - Folders stay under Settings → Workspaces.

### Added (plugins)
- A plugin framework (`viz/plugins.py`, docs/PLUGINS.md). A plugin is a
  folder with a `plugin.json` that adds jobs, checks under chat replies, a
  page and MCP tools. None runs until the owner turns it on in Settings →
  Plugins; a SHA-256 of its files is recorded then, and any change turns it
  off again. Scripts run as separate processes.
- Example plugin `plugins/claim-spotter`: marks the numbers, dates and
  quotations in a reply as claims to verify, and collects them from saved
  chats into a CSV.
- docs/CONNECT.md: *MCP, step by step*, for the three MCP pieces (Studio as
  tools for an agent app, another computer's LLMs through the Remote
  bridge, tools in Studio's chat).

### Added (remote hosts)
- Streaming chat with hosts that call in to Studio. The agent reads its
  backend's reply as it is written and posts the new text a few times a
  second (MCP progress token, `POST /api/remote/progress`); Studio passes
  it straight to the chat. Closing the chat stops generation on the host.
  Older agents still answer in one piece.
- 3D checks of a host's models. Studio traces the reply itself when it has
  the identical GGUF (matched by SHA-256), else the host runs OmniCortex's
  tracer with the classifier Studio sends and returns the trace. New agent
  tools `model_file` and `trace_reply` (the host can switch the latter
  off); `hscore.py --trace-out` saves a trace as .npz.
- A host may only answer or stream the messages queued for it.
- The Remote page shows when a calling-in host was last heard from and its
  agent version.

### Added (visualizer)
- Subject colours in the 3D views. From the neuron review,
  `neuron_review.subject_map()` finds which neurons go with which subject
  (Cohen's d against the other subjects, above the review's significance
  floor). A checked reply's 3D view colours active neurons by subject, in
  the theme's subject palette, with a legend toggle. Flagged hallucination
  neurons keep their colour.
- Camera controls in the browser 3D view.
  - Manual: Front, Side and Top presets and Reset (keys 1, 2, 3, R).
  - Automatic: Orbit, and Follow flags, which glides to the flagged neurons
    of each flagged token (keys O, F).
  - Moves are smooth and give way to a drag.

### Added (sharing results, opt-in)
- Score cards (`scripts/score_cards.py`, Settings → **Sharing results**),
  off unless turned on and given a scoreboard address.
  - A card holds: the model's SHA-256, name, quant and verified Hugging
    Face repo; the setup variant (H-Neuron set and scale, LoRA, lineage,
    engine and its answer-changing options, sampling); coarse hardware;
    the per-subject counts.
  - A private verification log holds the replies, used only to re-grade.
  - Public only for models found on Hugging Face (file SHA-256 verified)
    or listed by hand. Unlisted models are submitted automatically only
    with the Advanced switch, and then privately.
  - Preview of exactly what is sent, and a local copy of every submission.
- `scripts/scoreboard_server.py`: a reference scoreboard.
  - It re-grades every reply, rejects cards that do not match their
    replies, and keeps logs private (admin token only).
  - It shows public verified results grouped by model file and setup.
- Jobs: an on_finish hook, used for automatic submission after TestQA.
- docs/SHARING.md.

### Added (appearance)
- Named themes (Settings → **Appearance**; the header ◐ button steps
  through them).
  - The themes: Midnight, Daylight, Arcade, Ember, Forest, Solar, High
    contrast and Clinical.
  - Each colours the pages, the charts and the 3D activation and
    hallucination views. The 3D views follow the viewer's choice; **Make
    default** sets it for others and for the Godot 3D view.
  - All in viz/themes.json, with three new viewer themes (arcade, forest,
    solar).
- Model radar (Models → a model): per-subject accuracy and honesty
  (1 − hallucination rate), DDR-style, with a second model to compare (2P),
  in the theme's colours.

### Added (concept editing)
- `scripts/concept_edit.py` (Jobs → Retrain → *Concept edit*): finds the
  direction or PCA cone that separates two prompt sets in the residual
  stream, with biprojection against the baseline mean.
  - It projects that subspace out of W_O and W_out with norm-preserving
    column rescaling, at per-layer strengths from a peak/floor/distance
    kernel.
  - It measures damage as KL against the original on a neutral set, plus
    remaining separation and an optional behaviour regex.
  - `--search` optimizes kernel and direction layer, Heretic-style (Optuna
    TPE when installed) under a KL limit; `--analyze` only locates the
    concept.
- `scripts/sage_edit.py` (SAGE, Nonta & Rambhatla 2026): task adaptation
  through gated per-head additive and multiplicative activation edits
  (Hard-Concrete L0 gates), constrained to the orthogonal complement of a
  protected subspace.
  - The protected subspace can be the decline direction, so retraining
    does not erode "I don't know".
  - The invariant is checked in the running model after training.
  - `chat` applies the edits; `export` writes the additive part as a
    llama.cpp control vector, tested to load in llama-server.
- Fixed: merge_export.py did not create its output folder for full models
  (`finetune.py --method full`, edited models), so their GGUF conversion
  failed.
- docs/CONCEPT_EDITING.md.

### Added (engines)
- Engine options (`scripts/engine_options.py`, model panel → **Engine
  options**). One table drives the form, the command line and the
  optimizer.
  - Covers flash attention, KV cache types, loading mode, context shift
    with tokens to keep, prompt cache reuse, and speculative decoding
    (draft model, its device, n-gram draft-free methods, EAGLE-3/MTP).
  - Also covers devices and RPC servers, CPU core range/mask and strict
    placement, NUMA, a closed list of documented environment variables
    (ROCm, Vulkan, CUDA, vLLM), and `numactl`/`taskset` wrappers.
  - Each option is checked against the installed `llama-server --help`;
    those a build lacks are folded away and reported after loading.
- Fixed: llama.cpp renamed `--draft-max`/`--draft-min`, so loading with a
  draft model failed on current builds. The spelling now follows the build.
- vLLM as a second engine, per model. Hugging Face checkpoint folders appear
  in the model list. Options include prefix caching, chunked prefill, fp8
  KV cache, tensor/pipeline parallel, a speculative config and
  quantization; `--vllm`/`$NS_VLLM` names the command. The remote agent
  detects vLLM on port 8000.
- `scripts/engine_optimizer.py` (Jobs → Setup → *Find the fastest engine
  settings*): it measures tokens/s per combination of engine options with
  the real server.
  - Bayesian search with Optuna TPE when installed, else random search plus
    refinement.
  - Baseline first, cached trials, `--apply` to Studio's settings.
- docs/ENGINE.md: the options, recipes (small GPU and long chats, draft-free
  speedups, NVIDIA + AMD over Vulkan, ROCm + vLLM, Ryzen core pinning,
  NUMA), and what is outside an application's reach.

### Improved (retraining on deficits)
- `scripts/retrain_cycle.py` (Jobs → Retrain → *Retraining cycle*): one
  command from baseline to verdict. It serves the base GGUF with
  llama-server, measures, builds the dataset with the base model probed as
  the student, then trains, converts and serves each arm, and asks each
  exactly the baseline's holdout items. It gates each arm and recommends
  one or none.
  - Arms are layer choices (`all`, `plan`, `10-17`), so `--arms all,plan`
    tests focus layers against all layers on the same data.
  - It resumes after interruption (`--redo STEP`).
  - It stops with a reason when the baseline is mostly errors or the
    dataset is too small.
- finetune.py says when `--max-length` cut every reply off, instead of
  failing inside the trainer.
- Knowledge probing (`deficits.py --student`). Each failed fact is
  sampled several times:
  - weakly known facts (some sample right) are trained with the gold
    answer;
  - unknown facts are trained to decline (`--unknown abstain`, the
    default; or `skip` / `teach`);
  - unknown facts the model already declines are left alone.

  Teaching a model facts it does not know raises its hallucination rate
  (Gekhman et al., 2024).
- Fixed: synthetic variations' DPO pairs used the original item's failing
  reply as the rejected answer to a different question. The rejected side
  is now the student's own failing reply to the variation; without a
  student, variations get SFT only.
- Holdout guard: synthetic variations that restate a holdout question
  (a shared 8-word run, or 60% trigram overlap) are dropped.
- Focus layers (`deficits.py --review-model`, `finetune.py --layers plan`
  or `--layers 10-17`): train only the layers where the neuron review found
  wrong answers differ most from right ones on the failing subjects.
  - LoRA: adapters on those layers only.
  - Full: only those layers' weights.
- `scripts/retrain_gate.py` (Jobs → Retrain → *Keep or reject a retrain*):
  before/after on the holdout and the trained split.
  - Per subject: fixed vs broken items with an exact McNemar test, and
    hallucination and abstention rates.
  - A keep / reject / inconclusive verdict, the holdout size needed to
    tell, and a memorisation warning.

### Added (remote hosts)
- OmniCortex Remote (`remote/`): an agent for computers that run LLMs but
  not OmniCortex. It is distributed on its own: a pip package, a zip from
  Studio, and launchers for Windows and macOS/Linux. It uses only the
  standard library.
  - Its MCP server is the host's one control surface. Tools: host info,
    LLM servers, models, chat, load/unload (LM Studio's `lms`, Ollama),
    receive and install model files into LM Studio or Ollama, and delete
    (off by default).
  - It also offers an OpenAI-compatible pass-through for chat.
  - A browser console, on the host only, shows who is the host and who are
    the controllers, the pairing code, permissions and activity, and
    revokes controllers.
- Link methods:
  - same network, with a pinned self-signed certificate and LAN discovery;
  - any VPN address;
  - SSH tunnels, which Studio runs itself;
  - public tunnels;
  - call-home for hosts behind NAT: the host long-polls Studio, needs no
    open port, and has its own token.
- Pairing: a 10-minute, single-use code. The code never travels; HMAC
  proofs are bound to the TLS certificate the controller sees, and the
  host proves itself back.
- Studio → **Remote** (`/remote`):
  - a wizard (install, link method, pair) with *This computer:
    CONTROLLER*;
  - per-host info, models, load/unload, a test chat, and *Send model*
    (resumable, checksummed, optional install).
  - Hosts' models join Studio's model list as `<host>:<model>`.
  - MCP tools `remote_hosts` and `remote_call` (also terminal commands).
- Other MCP clients as controllers: `neuronscope-remote add-host` pairs
  from the command line, and `bridge` carries MCP over stdio to the host
  with the certificate pinned (Claude Desktop, Cline, Cursor).

### Added (settings)
- Workspaces (`scripts/ns_workspace.py`, Settings → Workspaces): named sets
  of folders and binaries (pipeline data root, models folders, downloads,
  llama.cpp checkout or explicit binaries, the pipeline's model, Studio's
  state folder) and environment for jobs and model servers. Studio, its jobs
  (which now run in the workspace root), doctor.py and the launcher read the
  active one; `--workspace` picks another. Setup's folder fields write to
  the running workspace.
- Models per subject (Settings): the measured best model per subject, and a
  pick that overrides it for `auto` whenever the subject is at least 30% of
  the prompt, before any model has enough graded results.
- Tool approval rules (`scripts/tool_rules.py`, Settings → Tool approval):
  ordered allow/ask/deny rules on server and tool globs, kind of operation
  (from MCP annotations) and argument regexes with paths and negation, with
  hourly limits, expiry and hit counts; then mcp.json `autoApprove`, quick
  settings per kind, and a default. Approval cards in chat offer
  *always allow this tool*, *these exact arguments* and *always deny*
  (owner only). A tester shows what a call would get and why.
- Transfer receiver: IPv6 binds (`--family`, bracketed hosts,
  `IPV6_V6ONLY`), from the local working copy.

### Added (tuning in LM Studio)
- `scripts/eval_lmstudio.py`: the adaptive tuner's evaluator for a model
  that will run in LM Studio, here or on another machine. It waits for the
  transferred candidate to appear, asks it a fixed seeded TestQA sample (or
  your own tasks) at temperature 0, and prints the score: accuracy,
  not-hallucinated, or a cost-weighted utility. `adaptive_tuner.py tune
  --worker local` writes candidates locally for the transfer watcher and
  scores them with any evaluator. Job kinds *Adaptive scale tuning* and
  *Score a candidate in LM Studio* (Jobs → Tuning). Nothing is deleted
  unless `--auto-delete`.

### Added (clients)
- Godot client as a full UI (`viz/godot`, `./neuronscope godot`): Connect
  (address, token, a trusted certificate, or pairing; connections
  remembered), Chat with reply checks and right/wrong labels, Models, Jobs
  (forms from Studio's job specs, filled from the workspace), Projects
  (approve, start, review results, answer questions, requirements), Review,
  native 3D (traces and review periods, layer filter, screenshots),
  Settings, Setup and Lab. It runs on another computer by default, so
  drawing never takes the model's GPU. A paired device gets the tabs it may
  use. `NS_SHOTS` screenshots every tab.
- Terminal client (`scripts/ns.py`, `./neuronscope cli`): every MCP tool as
  a command, interactive chat with `/check`, `/right`, `/wrong`, `follow`
  for job logs, and workspaces, models per subject, tool approval and
  restart for a remote Studio; `--cafile` for a self-signed Studio.
- `/api/status` says the caller's role (owner or device) and the workspace.

### Added (links between sections)
- Messages and fixes that name another section ("Setup → Build llama.cpp",
  "Settings → Workspaces", "Jobs → Evaluate"…) are links on every page; the
  reply check, routing and load errors carry links to where they are fixed.
  Jobs open filled in from the address (`/jobs#kind=…&field=…`) and fill
  llama.cpp binaries and the pipeline's GGUF from the workspace; Studio's
  tabs and a model's settings field open from `/#…`. Models without stats
  link to a TestQA job for them; the Lab's desktop viewers and the Review
  page link to their jobs; project errors show on the page with links.

### Fixed
- Saving a load preset (`/api/config`) wrote into `config.json`'s Studio
  section instead of the model settings: two functions shared a name.
- README and Getting Started command blocks are safe to paste whole (no
  `<placeholders>`, Studio in its own block). doctor.py names the Vulkan ICD
  that fails to load and its package, and which one to pin. install.sh
  repairs a reused venv before installing torch, so pip reports no conflicts.

### Added (per-project requirements)
- Each project has its own requirement list (tools, libraries, Python
  packages, hardware, OmniCortex features, notes), inferred from its
  checkout's build files when created or re-scanned (CMake, Meson, xmake,
  premake, autotools, Cargo, Go, Node, Python, Docker, Gradle, Maven, Zig,
  Godot, Nix, vcpkg, Conan, just, .NET), editable on the project page, and
  checked on this machine with the fix for its OS (pkg-config for libraries,
  the project's virtualenv for packages). Start answers 409 with what is
  missing unless forced; workers are told what the machine has and lacks.
  `ns_requirements.py project <dir>`; MCP `project_requirements`,
  `set_project_requirements`, `scan_project_requirements`, and `force` on
  `start_project`.

### Added (neuron review)
- Review page (`/review`) and `scripts/neuron_review.py`: every reply
  OmniCortex looks at is stored as an activation profile with its source
  (TestQA, other benchmarks, chat checks, project work, live scoring), kind,
  subjects, verdict, risk and time. Per neuron: Cohen's d between wrong and
  right answers, correlation with the classifier's risk, or firing rate,
  coloured only above a noise floor corrected for the number of neurons and
  replies. Filters by source, subject, verdict and dates; error rate per
  day, week or month; heatmap; strongest neurons in each direction; and the
  periods played as frames in the 3D view.
- Chat checks and background scoring record observations automatically;
  *right / wrong* under a check labels the reply. Jobs → Review ingests
  TestQA results and any graded JSONL (re-running each reply through the
  model for its activations). `HScorer.profile()` (no classifier needed).
- MCP tools `review_sources`, `review_summary`, `review_label`; job kinds
  `review_testqa`, `review_items`.

### Added (requirements)
- `scripts/ns_requirements.py` and Setup → Requirements: what each feature
  needs (Python packages at the versions `requirements*.txt` set, programs,
  llama.cpp builds and their GPU backends, ROCm/Vulkan/CUDA device access and
  groups, memory), what this machine has, and the fix for its package
  manager; vendors the machine lacks are n/a. Install buttons for missing
  Python packages (catalog only, a job); environment snapshots, diffs and a
  frozen requirements file. MCP tool `requirements`; job kinds
  `install_package`, `env_snapshot`.

### Added (everything from the browser)
- After `./install.sh`, no terminal is needed: an app-menu entry (Linux),
  `./neuronscope` and `OmniCortex.bat` start Studio (`scripts/launch.py`)
  and open it, on Setup until llama-server and models are found. Studio's
  main page points at Setup while they are missing.
- Setup page (`/setup`): models folders, llama-server, cett-dump, unloading,
  routing thresholds and worker limits, saved to `~/.neuronscope/config.json`
  (read at startup, flags still win) and applied at once where possible;
  build llama.cpp with a live log; hardware (devices on/off, a llama-server
  per backend or device, pinned ROCm, environment); the health check;
  restart Studio. Network exposure stays in the file and flags only.
- Lab page (`/lab`): start, open and stop the live view, 3D replay (with a
  trace picker), the pipeline dashboard, the quantization, scale-sweep and
  adaptive-tuning labs and the transfer page, from Studio.
- Jobs: 46 kinds (was 10): the H-Neuron pipeline stage by stage, building
  llama.cpp, vision, analysis (cross-validation, per-task neurons, score
  spikes, code hallucination, memory budget), selective merge, MoE expert
  sweep, and the desktop viewers. Jobs support positional arguments and a
  non-Python interpreter; a test checks every field against its script.
- `docs/UI_AND_CLI.md`: every feature's place in the UI, its command and its
  MCP tool. One navigation bar across Studio's pages.
- Fixed: the timeline's theme list offered `_ui`, which is not a theme.

### Added (connect)
- OmniCortex as an MCP server (`scripts/ns_mcp.py`, dependency-free): 26
  tools over Studio's API (models, chat, reply checks, routing, stats,
  devices, doctor, jobs, projects, traces), with read-only and destructive
  hints. Served by Studio at `/mcp` (streamable HTTP; refuses other web
  origins and non-JSON) and as a stdio process. Tested with the official
  `mcp` client over both transports. Every call carries the caller's token.
- Studio's Connect page (`/connect`): settings for Cline (OpenAI-compatible
  provider with a picker of this Studio's models, and the MCP entry), Claude
  Desktop and other OpenAI clients, with copy buttons, and a button that
  creates a revocable app token (a paired device). `scripts/ns_connect.py`
  prints the same and writes Cline's or Claude Desktop's config, keeping a
  backup and other servers. `POST /api/route` and `GET /api/doctor`.
- Fixed: a job log request for an unknown job id returned an empty log
  instead of 404.

### Fixed (from a newcomer dry run of the docs)
- TestQA crashed while printing per-subject results when a subject had no
  right answers.
- `install.sh --no-torch` (new) skips PyTorch for Studio, TestQA, Projects,
  transfer and the llama.cpp path; `requirements-core.txt` holds everything
  that does not need it (`accelerate` used to pull a multi-GB PyTorch in
  silently). A failed PyTorch install now gives advice for this machine
  (CPU, CUDA or ROCm) instead of always pointing at ROCm, and the installer
  ends by pointing at `doctor.py`.
- `doctor.py`: hardware advice from the detected devices (it assumed a Vega
  laptop), a devices section, build hints that use `build_llama_tools.sh` and
  `NS_LLAMA`, stage names instead of bare numbers, and a next step for Studio
  as well as for the pipeline.
- Studio prints the model ids it found at startup (what `"model"` and
  TestQA's `<model-id>` need).
- `cett-dump --help` lists its own flags; `ns_transfer_mcp.py --help` prints
  help instead of waiting on stdin; `--sandbox` without Docker says what to do.
- Docs: a glossary for enthusiasts (`docs/GLOSSARY.md`), how to get a first
  model, a way to verify the first extraction, a plain-language overview of
  the pipeline, the dataset link, the `--allow-exec` / `--sandbox` trade-off in
  the quick starts, and a broken README link.

### Security
- Studio refuses to start a backend on a port something else already holds
  (an orphaned server would otherwise answer the health check).
- Tuning worker results are signed (HMAC-SHA256, key derived from the worker
  token, per-job nonce); the controller refuses unsigned, altered or replayed
  results.
- WebRTC TURN credentials go only to peers inside a room, and with
  `--turn-secret-file` are minted per peer in coturn's REST format and expire.
- TestQA and deficits.py `--sandbox docker|podman`: model-written code runs in
  a container with no network, read-only filesystem, no capabilities, an
  unprivileged user and resource limits.
- One bind policy for every network service (`scripts/ns_security.py`). A
  non-loopback bind requires a token of at least 32 characters, and TLS unless
  `--allow-plaintext` is passed. Tokens are compared in constant time, can come
  from `--token-file` or the environment, and argv use prints a warning.
  `ns_security.py token` and `selfsigned` generate a 256-bit token and a LAN cert.
- Transfer receiver and tuning worker: built-in TLS, bounded request bodies
  (JSON 64 KiB, chunks 8 MiB), socket timeouts, failed-auth throttling, 128-bit
  transfer ids, a cap on active transfers, a disk-space check, verified `pull`,
  and source/profile/candidate SHA-256 per tuning job.
- WebRTC signaling: a 128-bit invitation key per room, carried in the link
  fragment; knock/admit for code-only joins; join throttling; Origin
  allowlist; TLS/WSS; relay-only and LAN ICE modes; security headers.
- WebRTC page: a verification code over both DTLS fingerprints, streaming
  per-file SHA-256 verified by the receiver, and a `randomUUID` fallback.
- MCP: `mcp>=2.3,<3`, checked at startup. Tokens live in the OS keyring or
  0600 files, are passed via the environment instead of argv, and are
  redacted from output. Legacy inline tokens are migrated.
- Studio: same bind policy, login throttling, Secure cookie under TLS, bounded bodies.

### Changed (visualization)
- Flagging is absolute: a token is flagged when the classifier's probability
  reaches `--flag-prob` (0.5, its own decision boundary), optionally smoothed
  (`--smooth`). Before, a token was flagged when its score was one SD above
  the reply's own mean, so every reply had flags, even clean ones; that is now
  only `--relative-z`, labelled as such. Cells are flagged only for H-neurons
  (an h_neurons.json, or the classifier's positive weights that
  `trace_sample.py --classifier` now stores in the trace); with neither, risky
  tokens are shown and no cell is.
- three.js and Godot show the reply text shaded by risk and a risk-over-time
  strip with the threshold (click either to jump), draw flagged H-neurons as
  rings on top of the field in a colour that keeps its hue under glow, order
  columns by classifier weight so H-neurons form a band, and state the
  flagging mode in the HUD. Godot: `N` jumps to the next flagged token.
- Themes: flagged colours chosen to stay distinct from active colours under
  common colour-vision deficiencies (ember: cyan on amber; cool: orange on teal).
- Fixed: Godot billboards ignored instance scale, so every cell drew at one size.
- Studio: *check* on a reply (or the *check replies* toggle) scores every token
  of it with the model's classifier, shades the reply text by risk, underlines
  flagged tokens, and links to a 3D view of that reply served by Studio at
  `/viz/<id>/`. Checks are saved with the chat and as trace sessions in
  `--traces-dir`. `POST /api/trace`, `GET /api/traces`; `HScorer.trace`.
- Fixed: the GGUF tokenizer decoded SentencePiece byte-fallback tokens
  (`<0xNN>`) to nothing, so characters outside the vocabulary vanished from
  decoded text and trace labels; multi-byte characters now land on the last
  of their byte tokens.
- Default theme is `dark`: a dark field with blue activations and red-orange
  flagged H-neurons (bloom.py, timeline.py, Godot via the API, Studio's 3D
  view, and the Live page). Studio, Jobs and the pairing pages open in dark
  mode unless light was chosen with the toggle.
- Subject classifier: returns every subject above a 30% share instead of one,
  and "unknown" when evidence is thin, which routing (`auto`, `route`,
  `model_stats.py rank`) treats as "use the best model overall". New task
  subjects graphics, systems (build) and reverse-engineering; 158 seed lines
  phrased as tasks rather than quiz questions; keywords match at word starts
  and count once per subject. `evaluate --tasks FILE` scores multi-label task
  files kept outside the repo.
- Director (`scripts/director.py`, Studio `/projects`): a description (and
  optionally a checkout) becomes skill-labelled tasks; a versioned plan you
  approve; after approval the director only proposes changes (follow-ups,
  a different model after repeated rejections) and you accept or reject them;
  each task gets the model whose graded results fit its skills (else the
  default, else the largest that fits, skipping models whose context is too
  short); workers end with a JSON report; results wait for review by policy
  (all, flagged, none); rejected work goes back with your feedback; blocked
  tasks ask questions; errors back off. Owner only.
- Worker models run as separate llama-servers placed by
  `scripts/accelerators.py` on ROCm, Vulkan, CUDA, Metal or the CPU, AMD
  first: per-device llama-server builds, settings, environment and pinned ROCm
  releases (`~/.neuronscope/hardware.json`), per-project device limits and
  settings, APU memory counted as VRAM plus GTT and shared with the CPU, Vega
  APUs (gfx90c) on Vulkan unless ROCm is opted into, automatic
  `HSA_OVERRIDE_GFX_VERSION` for RDNA2/3 small dies, multi-GPU layer splits
  before estimated-memory devices. `--max-workers`, `--worker-idle`, `--hardware`.
- Studio's llama-server command building is shared (`server_cmd`).
- Hub: Replay lists saved traces (Studio checks and `runs/`) to pick from.
- Docs: VISUALIZATION.md "Which view when" and how to read the 3D views.
- Removed tabs with nothing behind them: the planned Recipes tab, and the
  example Eval package is no longer enabled by default.

### Added
- Studio pairing: persistent (until revoked) or temporary access per pairing
  link, with host-configured expirations (`--pair-code-ttl`,
  `--pair-durations`, `--pair-max`, `--pair-default`,
  `--no-persistent-pairing`). Expiry is enforced on every request, browser
  cookies end with it, linked hosts report it, and the owner can extend,
  shorten, renew or make persistent from the Link page.
- `viz/bloom.py --demo`: a synthetic, clearly labelled trace for trying the
  three.js and Godot clients without a model. Godot usage (keys, `NS_FRAME`,
  `NS_PAUSED`, renderer fallback) in docs/VISUALIZATION.md.

### Added (CUDA)
- `scripts/cuda_info.py`: every NVIDIA GPU via nvidia-smi, and the decisions
  that follow: PyTorch wheel (CUDA 12.6 + `torch<2.15` for Maxwell/Pascal/Volta,
  e.g. Tesla M10), llama.cpp CUDA architectures and toolkit limit (CUDA 13
  cannot target sm<75), per-GPU training precision, QLoRA availability,
  multi-GPU split.
- `install.sh` installs the wheel that has kernels for every GPU, and its smoke
  test checks each GPU's kernels and compares an fp32 matmul against the CPU.
- `build_llama_tools.sh --backend cuda` builds for the detected GPUs
  (`--cuda-arch` to override) and refuses a CUDA 13 build for pre-Turing cards;
  `docker/llama-cuda.Dockerfile` builds with CUDA 12.9 (sm_50 by default),
  pinned to the llama.cpp commit the activation patch was tested on, with an
  optional CA secret for TLS-intercepting proxies. Built and tested here: the
  image's llama-server and cett-dump match PyTorch and the host build (CPU
  fallback; no GPU in this environment). The build script now clones shallowly.
- Studio: Visible GPUs, Split, Tensor split and Main GPU per model; the fit
  estimate uses free VRAM across all GPUs; `/api/gpus`.
- `finetune.py` picks precision from compute capability (fp32 on Maxwell and
  consumer Pascal; `is_bf16_supported()` also reports emulated bf16) and
  refuses QLoRA on Maxwell. `doctor.py`, `hostcheck.py`, `vram_budget.py` and
  host profiles see every GPU.

### Fixed
- Godot and three.js clients drew a narrow strip for most traces: neuron columns
  were spaced 0.06 units apart and the camera was fixed. Both now fit the field
  (640 units wide whatever the trace width) and frame it. three.js keeps the
  current token at a fixed depth, so long traces no longer drift past the
  camera.
- Godot HUD never showed FLAGGED (`Array.has` does not equate 146 with the
  JSON float 146.0). `NS_FRAME` / `NS_PAUSED` open the client on one token.
- The three.js viewer says when three.js cannot be loaded instead of showing a
  black page.
- Hub: the Live service can use a patched llama-server (`live source`) instead
  of always simulating; config values that would be read as flags are refused.
- Studio on phones: the chat log is no longer squeezed out by the sidebar.

### Added
- Studio Link: one-time pairing links (certificate fingerprint pinned) give
  browsers and API clients their own revocable device tokens; linked hosts
  serve another machine's models from this Studio as `name:model` in the
  picker and `/v1` (`scripts/ns_pairing.py`).
- Studio MCP client: tools from MCP servers (stdio or streamable HTTP,
  `mcp.json` in LM Studio / Claude Desktop format) offered to the chat model;
  every call is shown with Allow/Deny unless auto-approved; Tools tab.
- Studio RAG: document collections (Docs tab, `scripts/rag.py`), BM25 or hybrid
  retrieval with an embedding GGUF sidecar or any embeddings endpoint, cited
  passages under each reply.
- Studio `/jobs`: evaluation, retraining and benchmark runs from the browser
  (TestQA, deficits, SWE-bench training data, fine-tune, merge/GGUF, SWE-bench,
  LiveBench, CLIP_benchmark). Typed per-job forms with no free-form command
  line, persistent logs, cancel, a concurrency limit; loopback-only unless
  `--allow-remote-jobs`.
- `benchmarks.py swebench train-data`: retraining data from SWE-bench's train
  split (gold patches, DPO against the model's own patches, failure modes
  prioritised by a test-run report, test-set exclusion, replay anchors).
- Vision retraining: deficits.py now writes `sft_vision.jsonl` /
  `dpo_vision.jsonl` with rendered images for failed vision items, and
  `vision_synth.py` adds variations whose answers are known by construction
  (no teacher). `finetune.py --vision [--freeze-projector]` trains VLMs with
  their processor (vision encoder frozen); `merge_export.py --vision` writes the
  language GGUF and the mmproj where llama.cpp can convert it.
- Live activations from llama-server, compiled and tested:
  `llama-tools/server-activations/apply_patch.py` (or
  `build_llama_tools.sh --server-activations`) patches current llama.cpp to
  serve `GET /activations` over SSE, one frame per generated token. Frames
  match PyTorch CETT on every layer in an integration test.
  `export_classifier_bin.py --gguf` folds the down_proj column norms into the
  classifier, so streamed scores are on the classifier's own scale. Fixes found
  by compiling: current llama.cpp names the down projection `ffn_out-N`, and the
  SSE subscription guard was destroyed as soon as it was built.
- External benchmarks (docs/BENCHMARKS.md): `clip_bench.py` runs CLIP/SigLIP
  and H-Neuron-edited variants through CLIP_benchmark (ImageNetV2,
  ImageNet-Sketch, VTAB via task_adaptation, ...) with selective metrics per
  scale, and exports ImageFolders for `clip_neurons.py`; `benchmarks.py`
  drives LiveBench and SWE-bench (prediction, Docker harness, import) into
  rolling stats, and imports published leaderboard scores (BenchLM) as
  reference-only stats shown in Studio.
- Retraining on deficits: `deficits.py` (categorised, grader-verified SFT and
  DPO data, verified synthetic expansion, replay buffer, holdout),
  `finetune.py` (QLoRA / LoRA / full, SFT then DPO, DDP or FSDP via
  `--launch`), `merge_export.py` (merge + GGUF + quantize, or GGUF LoRA);
  `testqa.py --only-ids`. See docs/RETRAINING.md.
- `scripts/build_llama_tools.sh` builds llama.cpp with cett-dump for any
  backend; `tests/test_llamacpp_integration.py` checks the compiled tool
  against PyTorch on a tiny converted model.
- CLIP/SigLIP H-Neurons (`clip_neurons.py`: collect, extract, evaluate,
  export) and `suppress_mmproj.py` for llama.cpp vision projectors.
- TestQA (`testqa.py`) with a 77-task bank: reasoning, executable coding through
  an opt-in resource-limited interpreter, factual QA, and canaries. Scores per
  kind and per subject, with paired comparison and a reply cache.
- Subject classifier and router (`subject_classifier.py`).
- Studio: OpenAI-compatible `/v1` API with JIT loading, `--idle-ttl`, and
  `model: "auto"` routing. Vision models are paired with their mmproj
  automatically and accept image attachments. Also adds saved chats with
  Markdown export, tokens/s and time to first token, stop/regenerate, and
  dark mode.
- Adaptive Tuning lab rebuilt: labelled form, resumable state path, CA file,
  score-by-scale chart with confidence intervals, baseline and next interval,
  results table, light/dark.
- `tests/test_ui_pages.py` syntax-checks every embedded page's JavaScript and
  the pages as actually served (needs `node`).
- `install.sh` detects NVIDIA (CUDA) and macOS (MPS) as well as AMD (ROCm) and CPU.

- Rolling per-model stats (`model_stats.py`): graded right/hallucinated/
  abstained per subject from TestQA, H-Neuron activation scores
  (`hscore.py`) and live abstentions; tied to the model file.
- Studio's `model: "auto"` ranks only models with enough graded results, on
  pessimistic per-subject utility; manual choices are always honoured. UI
  shows an orange ⚠ with an explanation for models without stats, stats lines
  and tables, and the reason behind each auto pick.
- TestQA bank reorganised by subject (210 graded items, 30 per subject):
  new logic, science, factual (including false-premise) and code
  (API-existence) items; writing graded by mechanical `constraints`
  checks; vision items with deterministically generated images
  (`qa_images.py`). `--per-subject`, `--list`, a per-subject
  right/hallucinated/abstained report with CIs, and `--record-stats` /
  `--publish-stats`.

### Fixed
- `extract_activations_gguf.py` measured the prompt with the transformers
  tokenizer on one path, and fell back to counting decoded characters when
  the prompt/response boundary token merged. It now always tokenizes the
  prompt with the same binary and uses the common token-id prefix; verified
  end to end against PyTorch on a converted model.
- `cett-dump` silently dropped the last decoder layer: llama.cpp computes the
  final layer's FFN only for tokens that produce logits, so that layer was
  never seen. Every token is now an output (`--last-layer-outputs-only`
  restores the old behaviour). Found by compiling it for the first time;
  it now matches the PyTorch reference to f16 precision on every layer.
- `hscore.py` located the response by decoded character length, which broke
  on byte-fallback tokens; it now compares prompt and full token ids.
- `delegate.py`'s H-Neuron gate called `/api/score` on the tracing proxy, which
  never implemented it, so every gate check failed as "unreachable". The proxy
  now implements it (`autotrace.py --classifier`).
- The WebRTC page sent 1 MiB DataChannel frames, above Chromium's 256 KiB
  SCTP limit, so the channel closed on the first chunk. Frames are now sized
  from `pc.sctp.maxMessageSize`.
- `adaptive_tuner.py tune` crashed because `--token` and `--auto-delete` were
  never defined. The tuning lab GUI dropped the token it collected.
- Studio listed `mmproj-*.gguf` projector files as loadable models.
- Quantization Lab page never initialised: the server injected the models
  directory into a `<script>` unquoted, and never filled the profiles
  placeholder (`__PROFILES__ is not defined`). Values are now JSON-encoded.
- Scale Sweep Lab page had a JavaScript syntax error (a stray `${''}`), so
  none of its controls worked.
- `mcp` 2.x renamed `FastMCP` to `MCPServer`; the MCP server imports either.

### Changed
- Documentation reorganised into `docs/`; machine-specific paths, IPs and
  model names replaced with generic examples. `env.sh` no longer embeds a
  volume UUID or model path (use `env.local.sh`).
- Removed stray empty files and an editor config with a personal path, and
  stopped tracking `delegation.json` (use `scripts/delegation.example.json`).
  `autotune.sh` became the parameterised `scripts/lmstudio_scale_compare.sh`.

## 2026-10-02
- Split torch-free helpers into `ns_constants.py`; single GGUF metadata path in
  `gguf_utils.py`; validated profile selections and geometry; classifier
  train/test overlap guard; fixed the GGUF extractor's per-sample prompt fallback;
  made tokenizer/extractor subprocess failures fatal.
