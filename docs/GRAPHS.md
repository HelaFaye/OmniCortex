# Graphs

Node graphs that control what goes into and comes out of models: screen it,
verify it, route it, and ask a person when it matters. Build them in
**Studio → Graphs** (`/graphs`), run them there, from a project, over MCP, or
from the command line (`scripts/graph_engine.py`).

## How a graph runs

- **Values flow along wires.** A node runs once each of its required inputs that has a wire holds a value. It runs again when a new value arrives, which is how loops work.
- **Pass and fail ports route.** A screening or verification node puts the value on `pass` or on `fail` (with the reason), never both. A node whose required input was skipped is skipped too. So everything after a failed check is skipped, and a fail branch runs only on failure.
- **Errors do not have to stop the run.** Every node has an `error` output. If it is wired, a failure sends its message there and the run continues; if not, the run stops and names the node.
- **Loops are bounded.** A cycle must pass through a Loop node. The Loop sends the value round again up to `max` times, then out of `done`. A graph with an unbounded cycle is refused when saved. A run also stops after 500 node executions.
- **People can be asked.** An *Ask a person* node pauses the run. The **Approvals** tab shows the value; you can edit it and approve, or reject it with a note. A paused run lives in memory, so it does not survive a Studio restart.
- **Everything is traced.** Each node execution records its inputs, outputs, time and error. The canvas colours nodes and wires by what happened (green: produced a value; dashed: skipped). **History** keeps the last 200 runs.

## Nodes

| group | nodes |
|---|---|
| Input and output | Input (a run input by name), Output (a result by name), Template (`{a}` `{b}` … and the run's inputs) |
| Models | **Model** (any of your models, `auto`, or another OpenAI-compatible endpoint; reasoning split from the answer; JSON-schema answers checked and asked once more when wrong; optional memory across loop rounds), Pick a model (Studio's routing) |
| Screening | Regex (block or require), **Secrets and personal data** (keys, passwords, emails, card numbers; also gives a redacted copy), Length, JSON schema check, **Permission gate** (allow / ask / deny by the permission policy), **Emotion screen** (fails when an emotion vector peaks past a threshold), **Feature screen** (fails when chosen dictionary features fire) |
| Verification | **Judge** (rubric and reference, 0–10), **Hallucination check** (each token's risk from the model's own neurons; links the 3D view), **Self-consistency** (majority answer of N samples), **Cross-check** by another model, **Grounded in sources** (your document collections), **Run code against tests** (TestQA's sandbox) |
| Routing | Gate, Switch (by regex), First available, Loop, Ask a person, Fallback |
| Data | JSON extract, build and combine; Join text; Translate |
| Tools | MCP tool, Fetch a URL, Read a file, Write a file, Search documents |
| Delegation | **Worker** (a task done the way project workers do it, told its permissions), Graph (another saved graph as one step) |
| Plugins | node types from plugins you turned on |

Emotion and feature screens read text through the decoding lab ([FEATURES.md](FEATURES.md)).
The hallucination check needs an H-Neuron classifier for the model ([PIPELINE.md](PIPELINE.md)).

## Permissions

Nodes that act on the machine check the permission policy
([PERMISSIONS.md](PERMISSIONS.md)): reading and writing files, fetching URLs,
starting the code sandbox, and MCP tools (which also go through tool
approval). A check uses job `graph`, the run's profile (chosen beside **Run**),
and the node's own `profile` if set; the strictest answer wins. Inside a graph
there is no one to ask mid-call, so *ask* counts as no. To let a person decide,
put a Permission gate and an *Ask a person* node in front.

## Templates

| template | what it does |
|---|---|
| `screen-and-verify` | answer → secrets → length → hallucination check → judge; any failure goes to `blocked` with the reason |
| `retry-until-judged` | answer, judge, and on a low score try again with the judge's feedback, up to three more times |
| `cross-model-verify` | a second model checks the first; a person decides when they disagree |
| `self-consistent-answer` | five samples; the majority answer when 60% agree, else a person |
| `code-with-tests` | a worker writes code, it runs against tests in a sandbox, and failures go back to the worker |
| `verify-delegated-result` | for projects: screens a worker's result before review |
| `emotion-guard` | a reply whose emotion vectors peak as desperate or hostile goes to a person before it is sent |
| `grounded-answer` | search your documents, answer from them, and pass only an answer they support |

Pick one in the graph menu, change it, and save it under your own name.

## Projects (delegation)

A project's policy can name a **verification graph** (Projects → policy). It
runs after each worker's result, with the inputs `task` and `result`:

- an `accepted` output passes the result;
- a `feedback` output fails it, and the reviewer sees why;
- with **send failures back automatically**, a failed result goes straight back to the worker with the feedback, counting as an attempt;
- a graph that asks a person, or breaks, leaves the result to the reviewer.

`verify-delegated-result` is the starting point. Workers are also told their
task's permissions ([PERMISSIONS.md](PERMISSIONS.md)), and the **Worker** node
brings the same worker prompt into any graph.

## Debugging and evals

- **Run one node.** Select a node, open *run this node alone*, give its inputs and see its outputs. This is PySpur's node-level debugging.
- **Evals.** On the Evals tab, paste test cases as one JSON object per line, holding the run's inputs plus the expected value. Choose which output to compare and how (contains, exact, regex), and get the pass rate with every case's result.

## Writing node types

A plugin ([PLUGINS.md](PLUGINS.md)) adds node types with a `nodes` list in
`plugin.json`. Each node is one script, which reads `{"params", "inputs"}` as
JSON on stdin and prints its outputs as a JSON object. An output it leaves out
is skipped; printing `{"error": "..."}` fails the node. Built-in node types
are registered with `@node` in `scripts/graph_engine.py`.

## Other ways to run

```bash
python scripts/graph_engine.py types
python scripts/graph_engine.py run my-graph.json --input question="Is water wet?" --studio http://127.0.0.1:7870
python scripts/graph_engine.py eval my-graph.json --cases cases.jsonl --expect answer
```

MCP (and so the terminal client): `list_graphs`, `run_graph`, `decide_graph`
([CONNECT.md](CONNECT.md)). Studio's API: `GET /api/graphs`, `POST /api/graphs/run`,
`GET /api/graphs/runs/<id>`, `POST /api/graphs/runs/<id>/decide`,
`POST /api/graphs/eval`, `POST /api/graphs/node`, `POST /api/graphs/save`.

## Where the ideas come from

The pause-for-approval breakpoints, node-level debugging, test cases, traces
and one-file nodes follow [PySpur](https://github.com/PySpur-Dev/PySpur).
Some ideas come from ComfyUI LLM node packs:

- MCP tool nodes and reasoning split from the answer: [comfyui_LLM_party](https://github.com/heshengtao/comfyui_LLM_party);
- nodes that report an error rather than break the workflow, and the JSON build, extract and combine nodes: [ComfyUI-LLMs-Toolkit](https://github.com/ComfyUI-Kelin/ComfyUI-LLMs-Toolkit);
- one place to pick any model or endpoint: [ComfyUI-LLMs](https://github.com/leoleelxh/ComfyUI-LLMs).

[CREDITS.md](CREDITS.md) lists them all.
