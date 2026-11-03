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
| Speculative quantization mask rescaling | job + command | `scripts/quant_mask.py` |
| Layer-wise token jitter | job + command (Hugging Face weights) | `scripts/model_lab.py jitter` |
| Cross-layer attention hijacking | job + command (Hugging Face weights) | `scripts/model_lab.py hijack` |
| Neuron aging and decay | in Studio, off by default | `scripts/neuron_decay.py` |
| Neuro-fuzzing | job + command | `scripts/neuro_fuzz.py` |
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

## Speculative quantization mask rescaling

`quant_mask.py plan` quantizes with more precision where the model makes confident mistakes, and less where
it is safe. It ranks layers by risk, taken from one of:
- the share of `classifier.py`'s H-neurons in each layer;
- a neuron review summary;
- a list you give.

The riskiest layers' MLP and attention weights get `--high` (q8_0), the safest get `--low` (q3_k), and the
rest get your base quant. It writes a `llama-quantize --tensor-type-file` and runs it with `--run`.

Quant types are per tensor, so precision cannot be set per neuron. The per-neuron part is the importance
matrix. `quant_mask.py contexts` writes the mistake contexts (TestQA failures, Fix This Mistake pairs).
`llama-imatrix -f` on that file weighs exactly the neurons those contexts use, and `--imatrix` spends
precision on them inside every tensor.

Compare the result with a uniform quant of the same size on TestQA before using it. A mixed quant is not
better by construction.

## Layer-wise token jitter

`model_lab.py jitter` finds the layers that break under small noise. For each item (a prompt and its answer),
it adds Gaussian noise (`--sigma` × the residual stream's RMS) at one layer at a time and measures:
- how far the answer's log-probability falls;
- how often the first answer token flips.

Items come from a JSONL file, TestQA results or Fix This Mistake pairs. Layers where small noise flips answers
are listed as *brittle*: a focus list for `finetune.py --layers` or the H-neuron pipeline. The flipped items
become DPO pairs (`--dpo-out`), with the answer chosen and what the jittered model said rejected.

It runs on Hugging Face weights through PyTorch hooks. A GGUF served by llama.cpp cannot take injected noise.

## Cross-layer attention hijacking

`model_lab.py hijack` is the version of this idea that can be built: back-patching. A later layer's state
cannot feed an earlier layer within one forward pass, because it has not been computed yet. So it takes two
passes:
1. Pass 1 records the residual stream at `--source` (say layer 28).
2. Pass 2 mixes it into `--at` (say layer 12) with weight `--mix`.

The mix applies at the positions you choose. It can be limited to where a trigger direction (an SAE feature
or emotion vector, `.npy`) is active above `--threshold`. The source state is rescaled to the target layer's
norm. The output shows the next-token prediction before and after, and the KL divergence between them.

**Not done:** rerouting attention heads during live serving. This is a lab tool for finding whether a layer's
breakdown can be bypassed, and it runs on Hugging Face weights only.

## Neuron aging and decay

Studio's live correction strength is the scale of the LoRA adapter a model is served with (the slider on the
model, `/lora-adapters`). With decay on, two things happen:
- After `grace days` with no trigger, the live scale moves toward `floor` × the configured scale, halving the
  remaining distance every `half life days`.
- A trigger sets it straight back to full strength.

A trigger is either:
- a reply whose H-neuron risk is at least `risk`, from the model's classifier;
- a Fix This Mistake correction for the model.

Every change is logged in `decay.json` beside Studio's settings. The latest events appear in
`/api/settings/options`. Decay is off by default, and turning it on is a locked setting: it changes how a model
answers without asking.

**Not done:** ageing inside the weights. A suppressed or concept-edited GGUF has its scales baked in, and only
an adapter can be eased live. For a baked edit, the log shows when the correction stopped being needed, so you
can rebuild it at a smaller scale.

## Neuro-fuzzing

`neuro_fuzz.py` writes new test items aimed at the model's known weak spots. The seeds are:
- the TestQA items it got wrong, or got right with a high activation risk;
- Fix This Mistake corrections;
- risky replies in the neuron review.

A fuzzer model writes new items for each seed, in TestQA's own format: `qa` with aliases, or `reasoning` with
an answer. They test the same skill with new names, numbers, false premises and edge cases. Each item is then:
- deduplicated against the bank and against the others (word overlap of 0.9 or more);
- with `--verify`, answered by a second model and graded by TestQA's grader.

Verified items go to `fuzz.jsonl`, a bank folder `testqa.py --tasks DIR` reads. The rest go to
`unverified.jsonl.txt` for you to look at. Item ids start with `fz-` and record their seed, so a result can be
traced back to the weakness that produced it.

**Not done:** reading "neural coordinates" straight off the model to aim the fuzzer. Seeds are behaviours
(wrong or risky answers). Which neurons they involve is what the neuron review and the classifier work out.
