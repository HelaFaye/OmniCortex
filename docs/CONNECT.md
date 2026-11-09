# Connecting Cline, Claude Desktop and other apps

Studio offers other apps two separate things. Set up either or both:

1. **Your local models as the app's AI.** Studio's OpenAI-compatible API at
   `http://127.0.0.1:7870/v1`. Any app with an "OpenAI compatible" provider
   can chat with the models in your Studio, including `auto`, which picks the
   model with the best measured record for each prompt.
2. **OmniCortex as tools (MCP).** An agent can list and route models, check
   a reply for hallucination risk, run TestQA, retraining and benchmark jobs,
   and plan and review projects. These are the same operations as the UI, under
   the same rules.

The quickest way in is Studio's **Connect** page (`/connect`, link in the
header). It shows every setting below for the address you opened Studio
with, with copy buttons.

Which computer controls and which does the work, and which port is which: [ROLES.md](ROLES.md).

## MCP, step by step

OmniCortex has three MCP pieces. Pick by what should control what:

| you want | the MCP piece | set up |
|---|---|---|
| an agent app (Cline, Claude Desktop, Cursor…) to use OmniCortex: route models, check replies, run jobs | Studio's MCP **server** | A below |
| an agent app to use the LLMs on another computer (LM Studio, Ollama, llama.cpp there) | the Remote agent's MCP **bridge** | B below |
| Studio's chat to call tools (files, search, your plugins…) | Studio's MCP **client** | C below |

**A. Studio as tools for an agent app**
1. Start Studio and open **Connect** (header). Copy the entry for your app.
2. Paste it into the app's MCP settings (Cline: *MCP Servers → Configure*;
   Claude Desktop: *Settings → Developer → Edit Config*). For Cline,
   `python scripts/oc_connect.py cline --write` does this for you.
3. Restart the app's MCP servers. `studio_status` should answer.
   - Nothing to install: the HTTP entry needs only Studio running; the stdio
     entry runs `scripts/oc_mcp.py`, which uses the standard library only.
   - With a Studio token, the entry carries it; keep that file private.

**B. Another computer's LLMs for an agent app**
1. On the computer with the models (the *host*), run OmniCortex Remote
   (Studio → Connect → Remote hosts → *Download OmniCortex Remote*, or copy `remote/`).
   Python 3.9+, nothing else.
2. On the computer with the agent app (the *controller*), pair it once:
   `python -m omnicortex_remote add-host nsr://…` with the pairing link the
   host's console shows. Run it in the folder that holds
   `omnicortex_remote` (OmniCortex's `remote/`, or the unpacked Remote zip).
3. Add to the app's MCP settings:
   ```json
   {"mcpServers": {"gpu-box": {"command": "python",
                               "args": ["-m", "omnicortex_remote", "bridge", "gpu-box"],
                               "env": {"PYTHONPATH": "/folder/that/holds/omnicortex_remote"}}}}
   ```
   The app then has the host's tools: `list_models`, `chat`, `load_model`…
   ([REMOTE.md](REMOTE.md) lists them). The host's owner sees each call and
   can revoke the controller.
   - For an app on the host itself: `"args": ["-m", "omnicortex_remote", "stdio"]`.

**C. Tools in Studio's chat**
1. Install the MCP client library once: Setup → Requirements (`mcp`), or
   `pip install mcp`.
2. Write `~/.omnicortex/mcp.json` in the format LM Studio and Claude
   Desktop use ([STUDIO.md](STUDIO.md#tools-mcp) has an example), or turn on
   a plugin that brings an MCP server (Settings → Plugins).
3. In the chat, tick **tools**. Each call asks first unless its tool is
   auto-approved (Settings → Tool approval).

## Cline

**Models.** In Cline's settings choose *API Provider → OpenAI Compatible*:

| field | value |
|---|---|
| Base URL | `http://127.0.0.1:7870/v1` |
| API Key | Studio's token if it has one; otherwise any text (Cline wants something) |
| Model ID | a model id from Studio's list (printed at startup, shown under Models), or `auto` |

Coding agents work best with models that follow long instructions: a 14B+
coder model if your hardware allows. TestQA's code subject is a quick way to
compare yours (`model_stats` shows the results to an agent).

**Tools.** In Cline open *MCP Servers → Configure* (it opens
`cline_mcp_settings.json`) and add the entry from the Connect page:

```json
{
  "mcpServers": {
    "omnicortex": {
      "type": "streamableHttp",
      "url": "http://127.0.0.1:7870/mcp",
      "disabled": false,
      "autoApprove": ["studio_status", "list_models", "route_prompt", "model_stats", "devices", "doctor",
                      "job_kinds", "job_status", "job_log", "list_projects", "get_project", "list_traces",
                      "list_features", "list_graphs", "requirements", "project_requirements", "review_sources",
                      "review_summary"]
    }
  }
}
```

Or let OmniCortex write it: `python scripts/oc_connect.py cline --write`
(`--editor cursor`, `codium`, `code-insiders` or `windsurf` for other
editors). It adds only the `omnicortex` entry and keeps a backup of the file.

`autoApprove` lists the tools that only read; Cline will ask before
anything that chats, starts a job or changes a project. Studio must be
running for this entry. If your Cline cannot reach Studio over HTTP, use the
stdio entry from the Connect page (`--transport stdio` for the CLI). Cline
then starts `scripts/oc_mcp.py` itself, which talks to Studio for it.

## Claude Desktop

*Settings → Developer → Edit Config*, add the stdio entry from the Connect
page (or `python scripts/oc_connect.py claude-desktop --write`), and restart
Claude Desktop:

```json
{"mcpServers": {"omnicortex": {"command": "/path/to/omnicortex/venv/bin/python",
                                "args": ["/path/to/omnicortex/scripts/oc_mcp.py", "--studio", "http://127.0.0.1:7870"]}}}
```

If Studio needs a token, add `"--token-file", "/path/to/token"` to `args`.

## Claude Code

The Connect page shows both commands with this Studio's address filled in:

```bash
# over HTTP, while Studio runs (add --header "Authorization: Bearer <token>" if Studio has a token)
claude mcp add --transport http --scope user omnicortex http://127.0.0.1:7870/mcp
# or let Claude Code start the bridge itself
claude mcp add --scope user omnicortex -- /path/to/omnicortex/venv/bin/python /path/to/omnicortex/scripts/oc_mcp.py --studio http://127.0.0.1:7870
```

`claude mcp list` (or `/mcp` inside Claude Code) shows whether it connected.

## claude.ai (web and mobile)

Not directly. Custom connectors on claude.ai (*Settings → Connectors → Add
custom connector*) are reached from Anthropic's servers, not from your
computer. The URL has to be public HTTPS that anyone on the internet can
reach, and sign-in goes through OAuth. Studio signs apps in with a bearer
token, and that dialog has no field for one. Opening Studio to the
internet without sign-in would give your models and tools to anyone. Use
Claude Desktop or Claude Code on a computer that can reach Studio instead.

## Other apps

Open WebUI, Continue, LibreChat or your own scripts: base URL
`http://127.0.0.1:7870/v1`, any API key unless Studio has a token, and a
model id. Any MCP client can use `http://127.0.0.1:7870/mcp` (streamable
HTTP) or start `scripts/oc_mcp.py` (stdio).

## The tools

| tool | kind | what it does |
|---|---|---|
| `studio_status` | reads | What Studio is serving right now: the loaded model, its uptime, whether auto routing has stats to work with. |
| `list_models` | reads | Local GGUF models Studio can serve, with their measured accuracy and hallucination rate when TestQA has graded them. The 'id' is what every other tool's 'model' takes. |
| `load_model` | changes state | Load a model into Studio's chat server (unloads the current one). Chat requests also load models on demand, so this is only needed to preload. |
| `unload_model` | changes state | Unload Studio's chat model to free memory. |
| `chat` | changes state | Ask one of the user's local models. 'model' may be 'auto' to let Studio pick the model with the best measured record for the prompt's subject. |
| `check_reply` | changes state | Score each token of a reply for hallucination risk from the model's own neuron activations (needs a classifier for that model). Returns the flagged text and a link to a 3D view. |
| `route_prompt` | reads | Which subjects a prompt or task involves, and which model 'auto' would pick for it and why. |
| `model_stats` | reads | Per-model, per-subject results from graded runs: accuracy, hallucination and abstention rates with confidence intervals. |
| `devices` | reads | Accelerators on this machine (ROCm, Vulkan, CUDA, Metal, CPU) with free memory, and the worker models running on them. |
| `doctor` | reads | Check the installation: packages, GPUs, llama.cpp build, model, pipeline progress, with a fix for each problem. |
| `job_kinds` | reads | Jobs Studio can run (TestQA, retraining, benchmarks, pipeline stages, builds) and the typed fields each takes. |
| `start_job` | **starts or stops work** | Start a job of one of the kinds from job_kinds, with its fields as 'values'. Runs on the user's machine; long jobs can take hours. |
| `job_status` | reads | One job's state, or the most recent jobs. |
| `job_log` | reads | The end of a job's output. |
| `cancel_job` | **starts or stops work** | Stop a running job. |
| `list_projects` | reads | Projects the director is planning or running. |
| `create_project` | changes state | Turn a description (notes, a task list, a README) into a draft plan of skill-labelled tasks. Nothing runs until a person approves it. |
| `get_project` | reads | A project's plan: tasks, their status, assigned models, results awaiting review, pending proposals. |
| `edit_project` | changes state | Change a plan: changes is a list of {op: add|update|drop|reopen|assign, ...} (see docs/DIRECTOR.md). Logged as the person's edit. |
| `approve_project` | changes state | Approve a draft plan: assigns a model to every task and freezes it. |
| `start_project` | **starts or stops work** | Start (or resume) an approved project: worker models begin on ready tasks. Refused while the machine lacks a required item, unless `force`. |
| `pause_project` | changes state | Pause a running project. |
| `review_task` | changes state | Accept a task's result, or send it back with feedback for the next attempt. |
| `decide_proposal` | changes state | Accept or reject a change the director proposed. |
| `answer_task` | changes state | Answer a blocked task's questions; it goes back in the queue. |
| `list_traces` | reads | Saved per-reply checks and traces, viewable in 3D at <studio>/viz/<id>/. |
| `check_permission` | reads | Whether the permission policy lets a model read, write, run, start, reach or call something, and which rule decided. |
| `list_graphs` | reads | Saved node graphs, the templates, and recent runs. |
| `run_graph` | changes state | Run a graph with inputs and wait for its outputs; a run paused for approval returns its id. |
| `decide_graph` | changes state | Approve or reject a graph run that waits for a person. |
| `list_features` | reads | Feature dictionaries, emotion vectors, recorded activations and attribution graphs in the workspace, with their checks. |
| `requirements` | reads | What each feature needs (packages, programs, llama.cpp backends, drivers, devices, memory), what is missing and the fix for this OS. |
| `project_requirements` | reads | Check a project's own requirements (toolchain, libraries, packages, hardware, features, notes) on this machine. |
| `set_project_requirements` | changes state | Replace a project's requirement list. |
| `scan_project_requirements` | changes state | Infer a project's requirements again from its checkout's build files; a person's items are kept. |
| `review_sources` | reads | Models with recorded replies, and the sources, subjects and verdicts to filter the neuron review by. |
| `review_summary` | reads | Neurons that fire more on wrong answers (or with risk, or most often) for a filter, and the error rate per day, week or month. |
| `review_label` | changes state | Record whether a checked reply was right or wrong, so it counts in the review. |

`python scripts/oc_mcp.py --list` prints the same list.

## Tokens and safety

- **On this computer** (Studio on `127.0.0.1`, the default) no token is
  needed.
- **Studio with a token** (any network bind): the owner token gives an agent
  everything the owner can do, including jobs and projects. For an app,
  prefer the **Create a token for an app** button on the Connect page. That
  token is a paired device: it can chat, use `/v1`, check replies and read
  status, but cannot start jobs or change projects, and you can revoke it under
  Link. Cline sends it as an `Authorization: Bearer` header; `oc_connect.py
  --token-file` writes it into the settings file and makes the file
  private (0600).
- Every tool call goes back through Studio's API with the caller's own token,
  so an agent can never do more than the person or device it acts for.
- `/mcp` refuses requests from other web origins and anything that is not
  JSON, so a web page you visit cannot drive it through your browser.
- Jobs run on your machine and can take hours, and TestQA with `--allow-exec`
  runs model-written code. Keep `start_job` and `start_project` out of
  `autoApprove`.

## Without the UI

```bash
python scripts/oc_connect.py show                     # every snippet for this Studio
python scripts/oc_connect.py cline --write             # Cline's MCP settings
python scripts/oc_mcp.py --list                        # the tools
python scripts/oc_mcp.py --studio http://HOST:7870 --token-file studio.token   # stdio server, for any client
```

## The other direction: controlling LLMs on other computers

Everything above makes OmniCortex a tool for other apps. To have
OmniCortex, or any MCP client, use the models on a computer that runs LM
Studio, Ollama or llama.cpp but not OmniCortex, put the OmniCortex Remote
agent there. That computer is then a *host*, and Studio or the MCP client is
its *controller*. See [REMOTE.md](REMOTE.md).
