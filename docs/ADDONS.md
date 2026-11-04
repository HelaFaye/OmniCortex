# Add-ons

**Studio → Add-ons** (`/addons`) manages OmniCortex's optional parts:

- **Built into Studio.** Parts not everyone needs. **Remove** one and its pages, API routes, jobs and sidebar entries go; **Reinstall** brings them back. The code stays in the checkout, so nothing is downloaded, and your data (dictionaries, graphs, projects…) is kept either way.
- **Packages.** Plugins ([PLUGINS.md](PLUGINS.md)) installed from a `.zip` file or a repository, updated, turned on or off, and uninstalled. A package runs code on this computer, so it stays off until you turn it on, and goes off again whenever its files change (a SHA-256 over every file is checked).

## Built-in add-ons

| add-on | what removing it takes away | needs |
|---|---|---|
| Decoding lab | the decoding lab job (SLED, steering, random projections, live watching), random-projection fitting, the logit lens | torch, transformers |
| Features and emotions | `/features`, the Interpret jobs, the emotion and feature screen nodes | the decoding lab |
| Graphs | `/graphs` and its API | |
| Neuron review | `/review`, the Review jobs | the H-Neuron pipeline |
| Projects and workers | `/projects`, `/workers`, the director | |
| Labs and 3D views | the live viewer, 3D replay, pipeline dashboard and the quantization and sweep labs; the view jobs | |
| Documents (RAG) | document collections, the document search and grounding nodes | |
| System monitor | `/system` | |
| Automation and Home Assistant | power and activity rules, the Home Assistant connection | |
| Retraining | the Retrain and Merge jobs | torch, transformers, peft, trl, bitsandbytes |
| SWE-bench and LiveBench | SWE-bench prediction, evaluation, import and training data; LiveBench | datasets, swebench, Docker |
| H-Neuron pipeline | the H-Neurons, Analysis and Tuning jobs, the hallucination check node | |
| Sharing results | publishing to the scoreboard | |

An add-on another one needs cannot be removed alone: Studio offers to remove
both (removing the decoding lab removes Features too). Reinstalling an add-on
brings back what it needs.

**Always part of Studio:** chat, models, jobs, settings, setup, permissions and
help, and also vision (CLIP neurons, mmproj, vision benchmarks), remote hosts and
model transfer (the transfer lab stays on the Lab page), and device pairing.

## Default packages

These ship in the checkout's `plugins/` folder, marked `"default": true`. They are
on unless you turn them off, and they are trusted like the rest of the checkout, so
updating OmniCortex does not turn them off. Like any package, they can be packed
and installed elsewhere.

| package | what it is |
|---|---|
| `godot-client` | The Godot client: a sidebar tab, a job that downloads Godot 4 and checks it against the release's SHA-512 list, a job that opens the client here, and the client as a `.zip` for another computer (`/plugins/godot-client/download/neuronscope-godot.zip`). `./neuronscope godot` uses it. |
| `jewel-styles` | The icosphere and tesseract jewel styles, for the browser views (`styles.js`) and the Godot client (`styles.gd`, compiled by the client when it connects). Without it the jewel has the reactor core and off. |
| `theme-pack` | The Arcade, Ember, Forest and Solar themes and the Ember, Cool, Arcade, Forest and Solar 3D palettes. The core keeps Midnight, Daylight, Contrast and Clinical. |
| `graph-screens` | Graph nodes: a code security screen (shell injection, eval, disabled certificate checks, hard-coded secrets, destructive commands) and a prompt-injection screen for text a model is about to read. |
| `semantic-friction` | Less repetition at the sampler, with no extra model pass. Inside each reply it turns on llama.cpp's DRY sampler. Across replies it applies a small negative logit bias to the distinctive words this chat's earlier replies keep reusing (the same opener, the same sign-off), never to words in your latest message. Its **Repetition** check shows what a reply repeats. |
| `adversarial-canary` | A shadow test behind the chat. The **Canary** check asks the model again with small perturbations; an unstable answer is a hallucination risk. When your message carries pasted material, it plants an instruction there and flags the model if it obeys text it should only read. **Injection signs** (automatic, no model call) screens your message. The **Canary** graph node runs the same probes before the main model answers. The probes cost a few short extra model calls, so the Canary check runs when you click it. |
| `concept-translation` | Maps one model's representations onto another's: a ridge map from model A's layer to model B's, fitted on the same texts and scored with held-out R². It then carries a concept direction (an emotion vector, an SAE decoder row, a steering vector) from A into B. Hugging Face weights. Below an R² of about 0.3 the map is mostly noise, and the job says so. |
| `audio-stem-map` | A frequency map of WAV stems. Each band is judged against the spectrum's own tilt, so the check finds harshness (2–5 kHz), sibilance, boom, mud and hiss that stand out, as well as clipping, DC offset and out-of-phase channels. A repair pass turns flagged bands down by the amount they stick out, or by gains you give. MIDI files get velocity-spike and stuck-note checks. This is signal analysis on the audio itself; OmniCortex has no generative-audio model whose layers it could scale. |
| `cluster-offload` | Where something should run. A live page of free GPU memory on this computer and every connected host. A plan job places a model of a given size: one device when it fits, otherwise the fewest devices that together do, as RPC servers and a tensor split in llama-server's device order (layer streaming, [INTEGRATED_OPTIONS.md](INTEGRATED_OPTIONS.md)). A Proxmox job reads a cluster's nodes, guests (VMs and LXC containers) and GPUs with a read-only API token. |

`comfyui`, `claim-spotter` and `pose` ([POSE.md](POSE.md)) also ship here, but stay off until you turn them on.

A package can also set sampler options for Studio's chat while it is on (`"sampling"`: DRY, repeat, presence
and frequency penalties, XTC, within fixed limits; [PLUGINS.md](PLUGINS.md)). Its checks, nodes and jobs get
`OMNICORTEX_URL` and `OMNICORTEX_TOKEN` to call Studio, through `scripts/plugin_api.py`.

A package can add jewel styles (`"widgets": {"jewel_styles": [...], "js": "...", "gd": "..."}`
inside its page folder) and themes (`"themes": "themes.json"`, the same shape as
`viz/themes.json`), as well as jobs, checks, pages, tabs, MCP tools, graph nodes and
downloads ([PLUGINS.md](PLUGINS.md)).

## Packages

- **From a file.** Drop a `.zip` on the Add-ons page. `plugin.json` sits at the top of the zip or in one top folder. Absolute paths, `..`, links and device files are refused; at most 200 MB and 5000 files.
- **From a repository.** Add the address of an `index.json` (https, or a path on this computer). Every package it lists can be installed or updated, and each download must match the `sha256` the index gives.
- **Uninstall** deletes an installed package's folder. A package that came with OmniCortex (in the checkout's `plugins/`) can only be turned off.

Make a package and a repository:

```bash
python viz/addons.py pack plugins/comfyui -o dist/       # dist/comfyui-1.0.zip
python viz/addons.py index dist/                         # dist/index.json, with each package's sha256
```

Serve `dist/` over https (for example a GitHub release or Pages site) and add
`https://…/index.json` as a repository.

A package can add jobs, reply checks, a page, MCP tools and graph nodes
([PLUGINS.md](PLUGINS.md)). With `"tab": {"label": "…"}` beside its `page`, its
page also gets a sidebar tab (under *Add-ons*) while it is on.

## ComfyUI

`plugins/comfyui` comes with OmniCortex. Turn it on, and a **ComfyUI** tab shows ComfyUI inside Studio. Its jobs (Jobs → Plugins: comfyui):

- **install** clones [ComfyUI](https://github.com/comfy-org/ComfyUI) into its own folder (default `~/.neuronscope/comfyui`) and gives it its own Python environment (`.venv`). It installs the PyTorch build for this machine: CUDA, ROCm, XPU, Apple (MPS), DirectML or CPU, picked from the GPUs found. Optionally it also adds:
  - [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager);
  - the LLM node packs [comfyui_LLM_party](https://github.com/heshengtao/comfyui_LLM_party), [ComfyUI-LLMs-Toolkit](https://github.com/ComfyUI-Kelin/ComfyUI-LLMs-Toolkit) and [ComfyUI-LLMs](https://github.com/leoleelxh/ComfyUI-LLMs);
  - OmniCortex's own nodes (below).
  
  Run it again to repair an install.
- **run** starts ComfyUI on a port (8188) and keeps it running until you cancel the job. It listens on this computer only unless you change that; ComfyUI has no login of its own.
- **update** runs `git pull` for ComfyUI and its custom nodes, then updates their requirements.
- **add a custom node** clones any GitHub, GitLab or Codeberg repository into `custom_nodes`.

**OmniCortex's nodes in ComfyUI** (category *OmniCortex*):

- **chat** asks your models through Studio. `auto` picks a model by subject. Reasoning comes out separately.
- **hallucination check** gives each token's risk from the model's neurons, the flagged text, and a link to the 3D view.
- **run a graph** runs a Studio graph ([GRAPHS.md](GRAPHS.md)), such as `template:screen-and-verify`, and returns its output.

They reach Studio at `http://127.0.0.1:7870`, set per node, using `NS_STUDIO_TOKEN` or `~/.neuronscope/studio.token` if Studio has a token. ComfyUI is GPL-3.0 and is downloaded from its own repository; none of it is shipped here.

## Language packs

Not yet. Studio's text is written into its pages, so translating it needs a
string catalogue first; a language pack can then supply one (and translated
docs for `/help`). Until then this is the one item of the list that stays undone.
