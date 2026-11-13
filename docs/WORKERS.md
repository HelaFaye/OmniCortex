# Workers: where project tasks run, and how work is spread

Studio's director splits a project into tasks (Projects page). The **Workers**
page (`/workers`, *Workers* in the sidebar) shows where those tasks run and
lets you steer them.

## Who does what

- **This Studio is the controller.** It plans, assigns, and sends each task
  to a worker, and it keeps the results.
- **A worker is where a model runs and the work happens:**

| worker | what it is | how it starts |
|---|---|---|
| **this computer** | a model in Studio's model folders | Studio starts it on a free device when a task needs it, and stops it when idle |
| **endpoint** | an OpenAI-compatible server you run yourself (llama-server, vLLM, LM Studio, Ollama), here or on another computer | you start it; Studio only sends it tasks |
| **remote host** | a model on a computer running OmniCortex Remote, or on a linked Studio | through that host's agent ([REMOTE.md](REMOTE.md)) |

An endpoint on another computer is reached directly over the network, so
put it on a VPN or a LAN you trust. llama-server can require a key
(`--api-key`), which you give Studio when you add the endpoint. A remote host
through OmniCortex Remote adds pairing, TLS, the host owner's permissions
and its thermal guard.

All the roles and ports in one place: [ROLES.md](ROLES.md).

## What the page shows

- **Tiles**: tasks running and waiting, done and failed since Studio
  started, workers ready, and workers paused or draining.
- **Workers**: state, where each runs, tasks running against its limit,
  results (done, failed, time per task, tokens per second, last error).
  This computer's models appear once they run, have results or settings;
  *show every local model* lists all of them.
- **GPUs**: temperatures (edge, junction, memory), VRAM, load and power, for
  this computer and for each remote host. The sources are nvidia-smi,
  rocm-smi, or Linux's amdgpu sensors. A value that cannot be read shows as
  ?; nothing is estimated.
- **Queue**: every unfinished task, its worker, why it was chosen, and why it
  waits: dependencies, project paused, worker paused, draining, busy, down
  or too hot, or a retry after a failed attempt.
- **Activity**: assignments, starts, finishes, failures, tests and changes.

## Controls

- **Pause**: the worker takes no new tasks, and tasks assigned to it wait.
- **Drain**: it finishes what it has, then pauses.
- **Tasks at once**: a limit per worker. An endpoint defaults to the value
  you gave it; this computer's models follow the project's parallel limit
  unless you set one.
- **Subjects**: the subjects a worker takes. None ticked means any.
- **Test** (endpoints, remote hosts): a one-line chat, timed.
- **Stop** (this computer): stops the model now. A task running on it fails
  and is retried.
- **Move a task**: pick another worker in the queue. It takes effect the
  next time the task starts.

## Routing by subject

For each subject (code, math, writing…), choose up to three workers in
order. A task goes to the first one that is not paused, draining or too hot.
Subjects without a route use the director's choice: measured TestQA
results, then the project's default model, then the largest model that
fits. Routes apply when a plan is approved; approve again to re-route a
plan.

For example, with a heavy GPU elsewhere and a small one here:
- route code, math and logic to the heavy endpoint;
- route writing to the small local model;
- leave the rest to measured results.

To know which worker is better at a subject rather than guessing, measure
both with TestQA (Jobs → *TestQA run*) and let the measured route decide.

## Overheating

OmniCortex Remote can pause a host before its GPUs get too hot (host
console → *GPUs and thermal guard*):
- above the limit you set, the host refuses new chats and traces, and
  unloads what LM Studio and Ollama can unload;
- it resumes once it is a few degrees cooler;
- Studio sees a paused host as *too hot*, and its tasks wait.

GPUs also throttle and protect themselves near their own maximum. The guard
is an earlier, gentler stop that you choose.

## Work distribution

**Tasks → Distribution** (`/work`) shows where the work went over a window: the
last hour, 24 hours, 7 days, 30 days or everything kept. The Workers page counts
only since Studio started; this page reads a log that lasts across restarts.

Studio logs every unit of work:

- each chat reply through its API, including linked hosts;
- each project task attempt;
- each job.

For each it records the worker, model, device and computer, the prompt and
generated tokens, and the time taken. Studio also samples every GPU's load,
VRAM, temperature and power every 30 seconds, for this computer and for the
remote hosts the Workers page is watching.

| part | what it shows |
|---|---|
| tiles | units of work, failures, prompt and generated tokens, summed busy time, the fastest device |
| who did the work | one row per device, worker, model, kind of work (chat, task, job) or source. Each row has its share of the work and of the tokens, tokens in and out, generation and prompt speed, busy time, utilisation, average time and failures |
| tokens over time | stacked bars of tokens per device |
| hardware | per GPU: load, VRAM, temperature and power curves, with average load, peak temperature and VRAM, average power and the energy used |
| predicted speed | each model on each device: predicted tokens per second with an 80% range, beside the measured speed where there is one ([HARDWARE.md](HARDWARE.md#speed-predictions)) |
| how measurements are weighted | the weight each measurement system gets when models are ranked, its agreement, and its history ([REVIEW.md](REVIEW.md#weighting-measurements)) |
| neuron activations | for a model with recorded activation profiles ([REVIEW.md](REVIEW.md)), filtered by verdict if you like: a box plot per layer (5th, 25th, 50th, 75th and 95th percentiles and the mean), a histogram of every value, a layer × neuron-position heatmap of the mean, and wrong-answer mean minus correct-answer mean |

Speeds come from the server's own timings where it reports them
(llama-server does). Otherwise they are tokens over wall time, which also
counts queueing and the network.

The log is JSON Lines, one file per day: `worklog/work-*.jsonl` and
`worklog/hw-*.jsonl` next to Studio's `config.json` (or in `OC_WORK_LOG`).
It is kept for 90 days and never leaves this computer. `--no-work-log` (or
`OC_WORK_LOG=off`) turns it off. `python scripts/work_log.py --since 7d`
prints the same breakdown as JSON.

## API

| route | what |
|---|---|
| `GET /api/workers/overview` | workers, routes, subjects, queue, telemetry, activity |
| `POST /api/workers/endpoint` | add or update `{name, url, model, token, max_parallel, note}`, or `{remove: name}` |
| `POST /api/workers/control` | `{id, paused, draining, max_parallel, takes, stop}` |
| `POST /api/workers/routes` | `{routes: {subject: [worker ids]}}` |
| `POST /api/workers/test` | `{id}`: a timed one-line chat (endpoints and remote hosts) |
| `GET /api/work?since=24h` | the work distribution: totals, `by_device`, `by_worker`, `by_model`, `by_kind`, `by_source`, `timeline`, `hardware` |
| `GET /api/work/predict` | speed predictions for every local model and device, and the fitted model |
| `GET /api/work/measures` | measurement weights, details, combined ranking and history |
| `GET /api/work/neurons[?model=&verdict=]` | the models with activation profiles, or one model's activation stats |

Worker ids: a local model's id; `ep:<name>` for an endpoint;
`<host>:<model>` for a remote one. Endpoints, controls and routes are kept
in `workers.json` next to Studio's `config.json` (owner-readable only, since
it holds endpoint keys).
