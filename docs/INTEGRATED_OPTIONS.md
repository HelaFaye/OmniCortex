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
| Backend hot-swap | in Studio, on by default | `scripts/backend_swap.py` |
| Layer streaming across computers | per-model setting + job | `scripts/layer_stream.py` (llama.cpp RPC) |
| Feature-interlocked KV compression | chat option "summarize by topic" | `scripts/semantic_compact.py` |
| Morphing architecture | in Studio, off by default | `scripts/morph_router.py` |
| Neuron-granular expert sharding | job + command | `scripts/expert_shards.py` |
| P2P feature exchange | jobs + board submissions | `scripts/feature_pack.py` |

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

## Backend hot-swap

Sometimes llama-server cannot load a model because it runs out of memory, an allocation fails, the device is
lost or the server crashes. Studio then retries down a ladder:
1. the same backend with half, then a quarter, of the GPU layers;
2. each alternate llama-server you list, for example a Vulkan build beside the CUDA one;
3. CPU only.

The first setup that starts is used, and what failed before it is recorded (`PROC.fallback`, the model's
status). A failure caused by a bad setting, such as an unknown flag or a missing file, fails as before, since
no fallback would fix it.

If the server dies mid-chat, Studio reloads it the same way and sends the request once more.

**Not done:** moving a generation between backends mid-token. A reply in flight when the server dies is sent
again from the start.

## Layer streaming across computers

This uses llama.cpp's RPC backend:
1. Each helper computer runs `rpc-server`. Use the **Layer streaming: lend this computer's memory** job
   there, or `layer_stream.py serve`; build it with `build_llama_tools.sh --rpc`.
2. On the computer serving the model, set the model's **RPC servers** to `helper:50052,helper2:50052`.

llama-server then places a share of the layers on each helper and passes activations between them as it
computes. A model too large for any one computer can run across several. `layer_stream.py check` tests that
each helper is reachable. `split` turns each device's free memory into a tensor split.

**Security:** rpc-server has no authentication and no encryption. Anyone who reaches its port can allocate
memory and run computations there. Bind it to a trusted network address, and across networks put it behind
WireGuard, Tailscale or an SSH tunnel. It does not use OmniCortex's WebRTC channel, because this traffic is
large and latency-bound and needs a direct path.

**Speed:** every token crosses the network once per helper. Use it for models that cannot fit otherwise,
not to speed up one that fits.

## Feature-interlocked KV compression

In the chat's compression settings, choose **When the chat no longer fits → summarize them by topic
(anchors)**.

The oldest turns are grouped by topic: TF-IDF vectors merged by cosine similarity, so a plot point or a
module discussed in turns 3, 9 and 14 becomes one group. The model writes one short anchor note per topic,
keeping every name, number, path and decision. The notes stand in for all the turns they cover, so a detail
mentioned many times costs context once.

**Not done:** compressing the KV cache itself. llama-server has no way to merge cache entries, so this works
on the text that goes back into the context.

## Morphing architecture

Set a **full model** and a **small model**. Requests for the full model that score as simple go to the small
one. The score, 0 to 1, comes from:
- length;
- code;
- maths;
- several questions or constraints at once;
- reasoning words;
- a long conversation;
- images.

Each part is reported in the request's route (`X-NeuronScope-Route`, the reply's route note).

**Not done:** skipping layers inside one model. llama.cpp runs every layer it loads, and early exit only works
for models trained for it, so two real models are used instead. Keep the small one where it does not evict the
large one: a worker, a linked host, or both in memory. A local model swap costs a reload each time.

## Neuron-granular expert sharding

`expert_shards.py` builds one suppressed variant of a model per subject. Each variant has only that
subject's H-neurons scaled down, made with `suppress_gguf.py`. `--apply` then sets them as Studio's
**Models per subject**, so the subject classifier sends each prompt to its subject's variant. One subject's
correction then never bleeds into another's answers.

To get each subject's H-neurons, train a classifier on that subject's TestQA items.

**Not done:** one model with a router inside. Each shard is a whole model of the original's size, so this
is not a speed-up, and switching subjects on one GPU reloads the model.

## P2P feature exchange

A feature pack shares what you found inside a model without sharing the model: a small zip of
- H-neuron sets;
- emotion vectors and steering directions (`.npy`);
- SAE feature picks;
- small control vectors;
- steering and graph presets.

Its manifest names the base model it was made on (repo, architecture, layer count, width) and the SHA-256 of
every file. Build one with the **Feature pack** job or `feature_pack.py pack`. Share it either:
- over Connect's encrypted transfer, like any file;
- through a review board: an issue marked `omnicortex-feature-pack`, with the pack inline up to about 45 KB.
  The board files it under `features/pending/` and lists it in `board.json`.

Installing checks every hash. A pack made on a model of another shape is refused, because neuron indices and
directions mean nothing in a different network. A pack made on another repo of the same shape installs with a
warning that it is probably a fine-tune of the same base, and to check it.
