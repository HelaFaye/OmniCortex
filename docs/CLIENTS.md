# Three ways to use Studio: web, terminal, Godot

Studio is the server: it runs on the computer with the GPU and the models,
and keeps the models, jobs, projects, review data and settings. Three clients
use the same HTTP API, so what one can do the others can, wherever it
affects your work:

| | web pages | terminal (`scripts/oc.py`) | Godot (`plugins/godot-client/client`) |
|---|---|---|---|
| start | `./omnicortex` (opens the browser) | `./omnicortex cli …` | `./omnicortex godot`, or the Godot client tab (the `godot-client` add-on: gets Godot, opens the client, or downloads it for another computer) |
| chat, `auto` routing, check a reply, mark it right/wrong | ✓ (plus tools and documents) | ✓ `chat` (`/check`, `/right`, `/wrong`) | ✓ |
| models: list, stats, load | ✓ (plus downloads, per-model settings) | ✓ `list-models`, `load-model` | ✓ |
| jobs: start any kind, follow the log | ✓ | ✓ `start-job`, `follow` | ✓ (forms filled from the workspace) |
| projects: plan, approve, start, review results, answer questions | ✓ | ✓ `create-project`, `get-project`, `review-task`… | ✓ |
| project requirements | ✓ | ✓ `project-requirements` | status and check |
| neuron review | ✓ | ✓ `review-summary` | ✓ summary, and the 3D view natively |
| features and emotions: train, read a reply, circuits, steer | ✓ | ✓ `list-features`; train with `start-job --kind feat_*` / `emo_*` | the list and the jobs; reading and steering open in the browser |
| 3D views of replies and review periods | three.js in the browser | the link to the browser view | **native**: orbit, layer filter, F12 screenshots |
| workspaces, models per subject, tool approval | ✓ (edit everything) | ✓ switch, pick, quick settings | ✓ switch, pick, quick settings |
| setup: health check, requirements, build llama.cpp, restart | ✓ | ✓ `doctor`, `requirements`, `start-job --kind build_llama`, `restart` | ✓ |
| labs | ✓ | `python viz/hub.py` | start, stop, open |
| remote hosts (LM Studio, Ollama… without OmniCortex) | ✓ add, pair, send models | ✓ `remote-hosts`, `remote-call` | their models in the model list |

Edited only on the web pages (or their own command-line tools): workspace
folders, ordered tool rules with argument regexes (`scripts/tool_rules.py`),
per-model llama-server settings, MCP servers, documents for chat, pairing,
linked machines and adding remote hosts ([REMOTE.md](REMOTE.md)). These are
set up once, not used from minute to minute.

## Client and server on different machines

The Godot client and the terminal client are built to run on another
computer. Drawing the 3D views takes GPU time and memory that the model
would otherwise use, so on a laptop with an integrated GPU, or a box that
runs a big model, keep Studio there and look at it from elsewhere.

1. On the GPU machine, start Studio for the network with a token and TLS
   ([SECURITY.md](SECURITY.md)). Over a VPN such as Tailscale, plain HTTP is
   possible with `--allow-plaintext`.
2. On the other computer:
   - **Godot**: **Connect** → address, token, and the certificate file for
     a self-signed Studio; or **pair** with a link from Studio's **Link**
     panel. Connections are remembered.
   - **Terminal**:
     `python scripts/oc.py --studio https://gpu-box:7870 --token-file studio.token --cafile studio.pem list-models`,
     or set `OC_STUDIO_URL` and `OC_STUDIO_TOKEN`.

The owner token can do everything. A paired computer may use models, chat
and checks; jobs, projects, review and settings need the owner. The Godot
client greys out the tabs a pairing cannot use.

## The Godot client

```bash
./omnicortex godot                                     # or: godot --path plugins/godot-client/client
OC_STUDIO=https://gpu-box:7870 OC_TOKEN=… godot --path plugins/godot-client/client
OC_API=http://127.0.0.1:7880 godot --path plugins/godot-client/client res://main.tscn   # the 3D viewer alone (viz/bloom.py)
```

Tabs, grouped as in the web sidebar: Connect, Chat, Models, **Tasks** (Jobs,
Projects), **Studio** (Features, Review, 3D, Lab), Settings, Setup.
`OC_TAB` opens any of them, including a page inside a group. Links in
messages work as on the web pages: "Setup → Build llama.cpp" opens the build
job, "Jobs → Evaluate" opens TestQA filled in for a model, and a checked
reply's *open in 3D* switches to the 3D tab. The 3D tab:
- plays a checked reply token by token, or a review's periods frame by frame;
- plays a decoding-lab recording with its emotion lanes, the emotion jewel and the features at each token;
- keeps its controls on screen however you move the camera: orbit, pan, zoom, preset views and the sway toggle ([WIDGETS.md](WIDGETS.md));
- **Layers** limits the view to a range of layers;
- **Screenshot** or F12 saves a PNG under the app's user folder;
- keys: Space play/pause, ←/→ step, N next flagged token.

`OC_SHOTS=<folder>` makes the client screenshot every tab and quit, which
is how the docs' images are made.

Needs Godot 4.7 or newer (the standard build; .NET is not needed).

## The terminal client

```bash
python scripts/oc.py                                  # the commands
python scripts/oc.py chat --model auto                # interactive
python scripts/oc.py start-job --kind testqa --values '{"endpoint": ["m=http://127.0.0.1:7870/v1@MODEL"], "out": "runs/t.json"}'
python scripts/oc.py follow JOB_ID
python scripts/oc.py subject-defaults --set code=acme/coder-q4_k_m
python scripts/oc.py tool-approval --kind destructive=deny
```

Every MCP tool ([CONNECT.md](CONNECT.md)) is a command, with the same
arguments; `--json` prints the raw answer.
