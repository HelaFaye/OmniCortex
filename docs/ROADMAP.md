# Roadmap

The plan for the next stages, and how far each item has got. [STATUS.md](STATUS.md) says where things stand
today; this file holds the longer plan, from the engineering spec of 2026-11-23 and the audit that came with it.

**How to read it:**
- Each phase builds on the ones before it, and phases run in this order unless you reorder them.
- Every item has a status: **done**, **partial** (part of it exists; what is missing is said), or **planned**.
- **Check on hardware** marks a claim that has to be measured on the real card before it is relied on.

**Rules that hold throughout:**
- **API keys** (a teacher's, such as DeepSeek's) come only from environment variables or a protected config file.
  They never go into datasets, logs, manifests, job records or reports.
- **Nothing is promoted without a gate.** A trained adapter, a new quant or an automatic change is kept only if
  it passes the acceptance gate (phase 2), and the gate's report is kept with it.
- **Credits:** every model, dataset, paper and tool that a phase uses goes into [CREDITS.md](CREDITS.md) in the
  same pull request.

## Target setup

The setup the spec is written for:

| Part | Choice | Notes |
|---|---|---|
| Teacher | DeepSeek V4.1 Flash, through an OpenAI-compatible API | the audit gives the endpoint as `https://api.deepseek.com/v1@deepseek-flash`; confirm the model id against DeepSeek's model list before use |
| Student | Qwen3.8-27B (dense) or Qwen3.6-35B-A3B (mixture of experts) | |
| Card | 32 GB (MI50 32 GB) | the 27B at 4 to 5 bits fits for inference; see the training notes |
| Context | 200K to 262K tokens, Q8_0 KV cache | the KV size depends on each model's attention layout: measure with **Memory budget** (`vram_budget`), don't assume |

**Training on one 32 GB card (check on hardware):**
- A 27B dense model in FP16 is about 54 GB of weights, and the 35B-A3B about 70 GB, so neither trains in FP16 on
  one 32 GB card, not even with LoRA. The options:
  - shard the FP16 base over several cards (DeepSpeed ZeRO-3 or FSDP);
  - train LoRA on a GPTQ or AWQ base (ZeRO-2 is enough);
  - distil into a smaller student.
- bitsandbytes doesn't run on gfx906, so QLoRA isn't available there. `finetune.py` already refuses QLoRA on ROCm
  and says so.

## Phase 0: interface review and quick starts (2026-11-23)

**Status: done in this pull request.**

**What was reviewed:**
- the sidebar (Chat; Models; Docs; Tools; Tasks; Studio; Connect; Add-ons; System; Setup; Settings);
- the Jobs list: about 110 jobs in 13 groups;
- the help menu;
- the standard use cases.

**Findings:**
- **The sidebar is clear;** its labels match what people come to do.
- **Jobs is the hard part.** It holds about 110 jobs, named by method ("Deficit dataset", "SAGE", "CETT"), so
  someone with a goal ("make this model better at my code") has to know the method first.
  - **Fix:** task-first [QUICKSTART.md](QUICKSTART.md) guides, linked from the top of Jobs and from the help
    menu. Each step opens its job with the form already filled in.
  - **Decision:** the Jobs labels themselves stay as they are for now, as the audit advised. Renaming them would
    break people's habits and the docs.
- **Asking about a repository took one upload per file.** The Docs tab now adds a whole folder, keeping each
  file's path. On the command line: `rag.py add --folder`.
- **Lab's "Open in Jobs" buttons opened the job list, not the job.** They now open the job.
- **Left for later** (cosmetic, recorded so they aren't lost):
  - two job labels say nearly the same thing: *Snapshot the environment* (`env_snapshot`) and *Environment
    snapshot* (`dep_snapshot`);
  - the older pages still draw their own row of links under the title, besides the sidebar, and the Remote page's
    row lists different pages from the others;
  - the Retrain group mixes data, training, merging and quantizing jobs.

**The six quick-start paths:**
1. ask about a repository;
2. fix or extend one;
3. measure coding ability;
4. improve weaknesses;
5. adapt a model to a repository (waits on phase 1);
6. prepare for deployment.

## Phase 1: adapt a model to a repository ("parametric context distillation")

**Status: partial.** `scripts/repo_adapt.py` ([REPO_ADAPT.md](REPO_ADAPT.md)) does the scan, the questions, the
judge, the mix, the manifest, staleness and the size check, as two jobs. It reuses what exists:
- `deficits.py`: teacher, verification, replay, holdout;
- `finetune.py`: LoRA;
- `merge_export.py --lora-gguf`;
- Studio's LoRA hot-swap and scale;
- `truth_probe.py`.

| Item | Status |
|---|---|
| **Repository scanner:** files, symbols, public interfaces, build files, tests; the same skip rules as the Docs folder import | done (`repo_adapt.py`: Python by its parser; JavaScript and TypeScript, Rust, Go, C and C++ headers, Java, Kotlin, C# by pattern) |
| **Questions grounded in the source:** the teacher writes question-and-answer pairs, each citing the file and lines it rests on | done |
| **Judge:** a pair is kept only if its answer is supported by the cited source (a second teacher pass, or a check that runs the code) | done |
| **Provenance manifest:** for each pair, the repository, commit, files and lines, teacher, judge verdict and time; no keys | done |
| **Expiry:** the adapter records a fingerprint of the repository's core interfaces and the git commit. When the interfaces change, Studio says the adapter is stale. Commits that don't touch them don't expire it | done |
| **Training mix:** 85% repository pairs and 15% general replay, so the model doesn't forget what it knew (`deficits.py`'s own default is a 25% deficit share, for failures) | done (too little replay is a warning) |
| **LoRA:** rank 16 or 32; exported as a GGUF LoRA so Studio can switch it on per chat; a size check refuses an adapter larger than its budget | partial (the LoRA, the export, the hot-swap and the size check exist; a Studio indicator for a stale adapter is planned) |
| **Truth-probe calibration check:** the base model's probe still reads the adapted model (AUROC drop within the gate's limit) | partial (`quant_fidelity.py`'s truth transfer does this for quants; it will reuse it) |
| **Tests:** the scanner, the manifest, expiry against real git histories, and the mix ratios | done (`tests/test_repo_adapt.py`, with a fake teacher; a trial with a real small teacher is in REPO_ADAPT.md) |
| **API keys from the environment only** | partial: done in `repo_adapt.py` (`--key-env`; the tests check that no output holds the key). `deficits.py` still takes `--api-key` on the command line, where job records and the process list can see it; move it to the same scheme |

## Phase 2: one acceptance gate

**Status: partial.** The gate itself is done (`accept_gate.py`). Training-side guards, the CI mocks and the
diff view are left. Before it, two gates existed:
- `retrain_gate.py`: fixed against broken items on the holdout;
- `quant_fidelity.py`: what a quant kept.

The plan is one gate that any change (a retrain, an adapter, a quant, an automatic change) must pass, with one
report.

| Item | Status |
|---|---|
| **Holdout fixed against broken items,** per subject | done (`retrain_gate.py`) |
| **Next-token KL and same top token** against the reference | done (`quant_fidelity.py`) |
| **TestQA answers lost,** judged by their 95% interval | done |
| **H-neuron drift and truth-probe AUROC transfer** | done |
| **One gate over every report:** retrain, quant fidelity and repository adapter; accept, review or reject | done (`accept_gate.py`, [ACCEPT_GATE.md](ACCEPT_GATE.md)) |
| **A strict preset:** truth-probe AUROC drop at most 0.02 (the default is 0.05) | done (`--preset strict` in `accept_gate.py` and `quant_fidelity.py`) |
| **KL anchor during training:** a KL penalty against the base model's outputs on replay data | planned |
| **Collapse guardrails:** stop training when output diversity, abstention rate or KL to the base moves past a limit | planned |
| **Mock `llama-cett-dump` and judge fixtures,** so CI runs the whole gate without a model | partial (the integration test builds a tiny model; the mocks are planned) |
| **Cross-quant regression diff view:** item by item, which answers each quant gained and lost, side by side in Studio | planned |

## Phase 3: training on MI50 (gfx906)

**Status: planned.** It needs the hardware to verify.

| Item | Status |
|---|---|
| **FP16 LoRA** on gfx906, which has no bf16, with [STATUS.md](STATUS.md)'s fp16 range check (item 4) before training | planned |
| **SimPO and ORPO** preference training: no reference model, so it fits more easily | planned |
| **DeepSpeed:** ZeRO-2 for a GPTQ or AWQ base, ZeRO-3 for an FP16 base over several cards | planned |
| **Capability flags** in `profiles/hardware`: bf16, fp16, int8 dot products, flash attention, xGMI, bitsandbytes. Jobs read them to refuse what a card can't do | partial (the profiles exist; the flags are planned) |
| **Hybrid MI50 + M10 cluster:** a scheduler that gives each card what it does well. The producer and consumer SimPO pipeline: the M10s generate candidate replies (inference only), and the MI50s train | planned |
| **xGMI notes:** what multi-MI50 links do and don't speed up. llama.cpp's tensor split doesn't use xGMI collectives, and AllReduce over xGMI is unverified | planned (check on hardware) |
| **`GGML_CUDA_DISABLE_GRAPHS`:** confirm whether it applies to HIP builds before any profile sets it | planned (check on hardware) |

## Phase 4: long context and multi-turn

**Status: planned.**

| Item | Status |
|---|---|
| **TestQA long-context items:** facts placed at depths through 200K to 262K-token contexts (needle and multi-needle), with Q8_0 KV | planned |
| **Multi-turn items:** a fact given early, asked about later; a correction mid-conversation that must stick | planned |
| **Memory check before running:** `vram_budget` for the model, the context and the KV type; refuse with the numbers rather than running out of memory | partial (the job exists; the link to TestQA is planned) |
| **Telemetry overhead:** tokens per second with and without each probe, at long context | planned |

## Phase 5: better teaching data

**Status: partial.** Teacher-guided deficit distillation exists in `deficits.py`; these extend it.

| Item | Status |
|---|---|
| **Teacher-guided deficit distillation** with verification, replay and a holdout | done |
| **Multi-teacher:** two or more teachers; a pair is kept when they agree, or when the judge picks one | planned |
| **Counterfactual activation patching:** which layers carry a failure, to focus training there | partial (activation patching exists in Interventions; using it to pick training layers is planned) |
| **Teacher-labelled SAE features:** the teacher names the features of a trained dictionary | partial (Features 3 describes features; teacher labelling is planned) |
| **Sycophancy traps:** items where the user asserts a wrong answer; the model should hold the right one | planned |
| **Contrastive truth probes:** probes trained on paired true and false statements that differ in one fact | planned |
| **Fidelity-verified requantization:** after retraining, requantize and pass the gate before the new GGUF replaces the old | partial (each step exists; chaining them is planned) |
| **Training-data source lists:** curated and licensed sources per field (software engineering, reverse engineering, chemistry, physics), with licences recorded | planned |
| **Provenance and credits extraction:** each training row keeps its source and licence; credits are generated from them | planned |

## Phase 6: serving

**Status: planned.**

| Item | Status |
|---|---|
| **Cost- and confidence-aware routing:** route to the cheapest model whose calibrated confidence clears the bar | partial (routing by subject and measured accuracy exists) |
| **Post-quant Platt scaling:** recalibrate confidence after quantizing, on the holdout | planned |
| **Prompt-lookup speculative decoding:** drafts copied from the prompt, for edits and code | partial (`engine_optimize --spec` tries draft-free speculative decoding) |
| **Prefix-KV pinning:** keep the system prompt and tool definitions cached across requests | planned |
| **Pareto per-layer suppression solver:** the per-layer H-neuron suppression scale that trades hallucination against accuracy best, jointly with [STATUS.md](STATUS.md)'s per-layer size budget | planned |
| **MoE router-gate probing:** which experts the router picks for each subject, and whether the router's confidence predicts errors | partial (the MoE expert sweep exists) |
| **MCP loop and thrash detector:** spot an agent calling the same tool with the same arguments, or undoing its own edits; warn, then stop | planned |
| **Truth-weighted context compaction:** when the context fills, drop what the truth probe rates least reliable first | planned |

## Phase 7: telemetry on MI50

**Status: planned.**

| Item | Status |
|---|---|
| **Decode-only projected probes:** read a few directions per layer during generation, not whole activations | planned |
| **Sparse H-neuron subset:** read only the classifier's neurons | planned |
| **Checkpoint logit lens:** the lens at a few layers, not all | planned |
| **Forensic replay:** record the inputs and seeds so that a flagged reply can be recomputed in full later | planned |
| **Overhead budget:** each probe's cost is measured (phase 4), and a probe that costs more than its budget is off by default | planned |

## Phase 8: image-model attribution

**Status: planned.**

| Item | Status |
|---|---|
| **Models:** SDXL; CommonCanvas, which is trained on Creative Commons images, so its sources can be named | planned |
| **Nearest training images:** a FAISS index over LAION-Aesthetics embeddings; check the dataset's current availability and terms (LAION re-released its sets as Re-LAION) | planned |
| **Gray-box membership check:** a denoising-loss Z-score of an image against reference images | planned |
| **Interfaces:** an MCP tool, a Studio panel, and a watched folder that reports new images over server-sent events | planned |
| **Workers:** an opt-in split between M10 and MI50 cards | planned |
| **Wording:** a membership score is evidence, not proof, and the panel says so | planned |

## Phase 9: unattended improvement

**Status: planned.** Every change still passes the gate (phase 2), and nothing is promoted without it.

| Item | Status |
|---|---|
| **Git commit shadowing:** a shadow branch records each automatic change, so it can be reviewed and reverted | planned |
| **Curiosity scout:** finds candidate improvements on GitHub, arXiv and SearXNG, through a safe browser that only reads | planned |
| **Curiosity reviewer:** ranks what the scout found. With `manual_lock` on, a person approves each change | planned |
| **Sandbox:** anything the scout brings in runs in Podman with `--network=none` | planned |
| **Overnight orchestrator, promotion gate, systemd timer** | planned |
| **Installer scripts** for all of the above | planned |

## Order, as the audit recommends

1. Phase 1's scanner, question generator, judge, manifest and expiry tests. *Done.*
2. Wire them into `finetune.py` and `merge_export.py`, with the 15% replay and the size check. *Done as job
   steps (quick-start path 5); a Studio stale-adapter indicator is left.*
3. Phase 2's strict preset (AUROC drop at most 0.02). *Done, with one gate over every report.*
4. A Studio job for phase 1, and quick-start path 5 completed. *Done.*
5. Phase 3: the MI50 training path and the capability flags, checked on hardware.
6. Phase 4: long context, xGMI and telemetry benchmarks.

Phases 5 to 9 follow. The open items in [STATUS.md](STATUS.md)'s *Next stage* still stand alongside these:
- the per-layer size budget;
- the fp16 range check;
- A/B tests of settings.
