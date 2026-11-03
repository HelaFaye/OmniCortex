# Integrated options

These features are built into OmniCortex rather than shipped as add-ons. Each one started as a brainstormed
idea. This page says what each one actually does and where it stops short of the idea, so you know what
you are turning on.

Studio's settings for them are under **Settings → Integrated options**, served at `/api/settings/options`.
One registry, `scripts/integrated.py`, sets their names, defaults and limits. Settings that can break things
are locked; unlock one to edit it, after reading its warning.

| Option | Status | Where |
|---|---|---|
| Fix This Mistake (deficit harvesting) | in Studio | `scripts/harvest.py`, the **fix** button under each reply |
| Neural Reset (loop breaker) | in Studio, on by default | `scripts/loopguard.py` |
| Prompt sanitizer | in Studio, off by default | `scripts/sanitizer.py` |
| Core shunt | in Studio's jobs, on by default | `scripts/core_shunt.py` |
| Dependency garbage collector | job + command | `scripts/dep_gc.py` |
| Dependency snapshots and sandboxes | API + job + command | `scripts/dep_snapshot.py` |
| Speculative quantization mask rescaling, layer-wise token jitter, attention hijacking, neuron aging and decay, neuro-fuzzing | next | |
| Layer streaming, expert sharding, backend hot-swap, morphing, KV compaction, feature exchange | after that | |

## Fix This Mistake

The **fix** button sits under every reply in the chat. Write the reply the model should have given, and
OmniCortex keeps:
- the conversation up to that point;
- the reply, as the rejected answer;
- your version, as the chosen answer.

When the model has an activation method (`llama-cett-dump`), the reply's activation profile also goes into
the neuron review with verdict *wrong*. Neurons that fire on confident mistakes are then counted there.

The pairs collect in `~/.neuronscope/harvest/`.

The **Fix This Mistake: micro-tune** job (`harvest.py train`) runs a short SFT + DPO pass (`finetune.py`) on
the pairs no earlier run used. Turn on **auto train** to run it by itself when two things hold:
- at least *min pairs* new pairs are waiting;
- the keyboard and mouse have been idle for *idle minutes*, and no chat is in progress.

It needs the chat model's base weights in Hugging Face format, not the GGUF. The result is an adapter in
`runs/harvest/`, and nothing is swapped in for you. Use **Retrain** to check it against a holdout and merge
it.

**Not done:** isolating "the H-neurons responsible for that specific error" per reply. A per-reply profile
only shows how strongly each layer fired. Which neurons drove the error comes from the neuron review over
many such replies, or from a classifier trained on them (`docs/REVIEW.md`).

## Neural Reset

Neural Reset is a loop breaker for Studio's chat. While a reply streams, it watches for two patterns:
- the reply's end is one passage repeated (3 or more times, 60 or more characters);
- the same line keeps coming back.

When it sees a loop, it:
1. stops generation;
2. keeps the reply up to the end of the passage's first copy;
3. tells the page to trim what it showed;
4. sends the request again, with the kept text as the start of the assistant's reply.

The resumed request pushes the loop's opening words down with `logit_bias` and turns on the DRY sampler and
a repeat penalty for the rest of that reply. The reply's stats show "↺ loop cut". It resets at most twice per
reply by default.

**Not done:** "erasing the activations" of the stuck neurons mid-sentence. llama-server has no API for that,
so the reset works at the sampling level. Continuing from a trailing assistant message needs a llama-server
recent enough to support it; older ones restart the reply instead. OpenAI-API clients of Studio never get a
reset, because text a client has already received cannot be taken back.

## Prompt sanitizer

This applies to chats sent to a **linked host**, a model on another computer. Before the request leaves,
these become placeholders like `[SECRET_1]`, `[EMAIL_1]` and `[NAME_1]`:
- API keys and tokens: OpenAI, Anthropic, GitHub, Hugging Face, AWS, Slack, Google, JWTs, bearer tokens,
  `api_key = …` assignments and private key blocks;
- e-mail addresses;
- optionally, IPv4 addresses;
- every term you list: company names, code names, hostnames.

The reply's placeholders are filled back in on this computer as it streams. The same value always gets the
same placeholder, so the remote model can still tell that two mentions are the same key.

**Limits:** it matches patterns. A secret in a form it has no pattern for, or a name you did not list, goes
through unchanged. It is off by default; turning it off is a locked setting.

## Core shunt

Every job Studio starts runs at low priority:
- **Linux:** nice 10 and idle-class disk I/O, pinned to the efficiency cores on Intel hybrid or ARM
  big.LITTLE CPUs;
- **macOS:** background QoS, which also moves the job onto the E-cores;
- **Windows:** below-normal priority class.

Each job records what was applied (the `shunt` field of `job.json`).

**Not done:** GPU priority. No vendor offers a portable per-process GPU priority, so a GPU-bound job still
shares the GPU with your desktop.

## Dependency garbage collector

`python scripts/dep_gc.py`, or the **Clean up build and download leftovers** job, lists what it finds. It
deletes only with `--apply`. It looks for:
- interrupted downloads;
- Hugging Face cache blobs that no snapshot points to;
- llama.cpp build folders with no `llama-server` in them;
- `__pycache__` folders;
- pip's cache;
- downloaded add-on packages that are no longer installed.

It also reports, and never deletes:
- `.gguf` files without the GGUF signature;
- duplicate model files, found by SHA-256.

It never touches models, workspaces, runs or settings.

## Dependency snapshots and sandboxes

`dep_snapshot.py snapshot` records:
- every installed package and its version;
- the PyTorch build, read from package metadata without importing it;
- the llama.cpp tools found, with their versions;
- `NS_*` settings (never ones containing tokens or keys);
- the requirements files' hashes.

`diff A B` lists:
- packages missing, or at another version;
- tools missing, or at another version;
- a different PyTorch build or Python;
- the pip pins that would close the gap.

`sandbox` makes a venv over the main environment (`--system-site-packages`) that adds only those pins. A job
can then run in it without changing the main environment.

Studio serves its snapshot at `/api/env/snapshot`. `POST /api/env/compare {"link": name}` compares this
computer with a linked Studio before you send it a model or a graph.

**Not done:** installing anything on the other computer by itself. The diff and plan tell you what is
needed. Running it there is a job you start on that host.
