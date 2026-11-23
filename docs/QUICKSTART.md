# Quick start: what do you want to do?

Six common kinds of work, each as a short path through Studio. A link that opens a job fills in the form with
sensible values; check them, add your model or files, then press **Run**. Nothing runs until you do.

New to the words? [GLOSSARY.md](GLOSSARY.md). Installing? [GETTING_STARTED.md](GETTING_STARTED.md). Every feature
in one table: [UI_AND_CLI.md](UI_AND_CLI.md).

Most paths need a model that Studio can serve. Load one under **Models → My Models**, or find one under
**Models → Discover**. Jobs that ask for an *endpoint* take `label=URL@model`; Studio's own is
`http://127.0.0.1:7870/v1` with the model's id.

## 1. Ask questions about a repository

Replies that cite the repository's own files, without training anything.

1. **Docs**: type a name (such as the repository's) and press **Create**.
2. **+ Add a folder (a repository)** and choose the repository's folder. Every text and code file is indexed;
   `.git`, dependencies (`node_modules`, virtual environments) and build output are skipped. Adding the folder
   again later updates the files that changed.
3. **Chat**: pick the collection in the **docs** selector under the message box, and ask. Each reply lists the
   passages it used, with their paths in the repository.

For better matches on meaning rather than names, give Studio an embedding model ([STUDIO.md](STUDIO.md),
*Documents*). Command line: `python scripts/rag.py add --collection myrepo --folder ~/src/myrepo`.

## 2. Fix or extend a repository

Two ways, by how much you want to hand over.

- **One change at a time, in your editor:** connect Cline, Claude Desktop or another agent to Studio's models and
  tools under **Connect → Apps** ([CONNECT.md](CONNECT.md)). The agent edits; Studio serves the model, checks
  replies and keeps the record.
- **A larger piece of work:** **Tasks → Projects** splits it into tasks by skill, runs them on worker models
  (**Tasks → Workers**), and asks you to review each one ([DIRECTOR.md](DIRECTOR.md)). **Requirements** checks
  the project's toolchain and packages before it starts.

Pair with path 1, so that the model can look things up in the repository while it works.

## 3. Measure how well a model codes

1. **TestQA on code**: [open the job](/jobs#kind=testqa&subject=code&per_subject=50&sandbox=docker). Give the
   endpoint. Tick *allow exec* only with a sandbox: the model's code runs to grade it.
2. **SWE-bench** (real GitHub issues): [predict](/jobs#kind=swe_predict&limit=50), then
   [evaluate](/jobs#kind=swe_evaluate) the predictions (needs Docker).
3. The results show under **Models → My Models** (measured accuracy per subject), and **Studio → Review** shows
   which neurons go with the wrong answers.

To compare two models item by item: [Compare models item by item](/jobs#kind=merge_eval).
[BENCHMARKS.md](BENCHMARKS.md) has the details.

## 4. Improve what a model gets wrong

The retraining cycle, in one job or step by step ([RETRAINING.md](RETRAINING.md)):

- **In one job:** [Retraining cycle](/jobs#kind=retrain_cycle&method=lora&quant=Q4_K_M). It measures, curates the
  failures into training data, trains, converts, measures again on held-out items, and keeps or rejects the
  result. It resumes where it stopped.
- **Step by step:**
  1. [TestQA](/jobs#kind=testqa) with a *cache* folder;
  2. [Deficit dataset](/jobs#kind=deficits&unknown=abstain): the failures become verified training pairs, with
     replay and a holdout. A *teacher* (a stronger model) can write and check the fixes;
  3. [Fine-tune](/jobs#kind=finetune&method=lora): LoRA on Hugging Face weights. On an AMD card use `lora`:
     QLoRA's bitsandbytes doesn't run on ROCm;
  4. [Merge and convert](/jobs#kind=merge&gguf=Q4_K_M) to a GGUF, or to a GGUF LoRA that Studio can switch on
     and off;
  5. [Keep or reject the retrain](/jobs#kind=retrain_gate): fixed against broken items on the holdout.

For a single wrong reply in chat, **Fix This Mistake** under the reply keeps the corrected pair, and
[Fix This Mistake: micro-tune](/jobs#kind=harvest_train) trains on the kept pairs.

## 5. Adapt a model to one repository

*Partly available.* A dedicated workflow is planned ([ROADMAP.md](ROADMAP.md), phase 1). It will scan the
repository, have a teacher write questions and answers grounded in its source, train a small LoRA mixed with
general replay data, record where every pair came from, and expire the adapter when the repository's core
interfaces change.

Today:
- **Without training:** path 1. Retrieval over the repository is often enough, and it never goes stale.
- **With training:** write question-and-answer pairs about the repository in the deficit dataset's format, then
  use steps 3 to 5 of path 4. Keep general replay data in the mix (the deficit dataset adds it), and check the
  result with the gate.

## 6. Prepare a model for deployment

1. **Pick the quantization:** [Choose a quantization](/jobs#kind=quant_select&objective=gen) measures fit, speed
   and fidelity on this computer and keeps the fastest faithful one.
2. **Check what the quant lost:** [Did a quantization keep what the model knows?](/jobs#kind=quant_fidelity&testqa=qa/bank)
   compares it with the original: next-token agreement, TestQA answers lost, H-neuron drift, and whether the truth
   probe still works ([QUANT_FIDELITY.md](QUANT_FIDELITY.md)).
3. **Tune the engine:** [Find the fastest engine settings](/jobs#kind=engine_optimize&kv=1) for this card, and
   apply the winner to Studio's settings.
4. **Match the hardware:** [Hardware: apply a support profile](/jobs#kind=hw_apply) ([HARDWARE_PROFILES.md](HARDWARE_PROFILES.md)).
5. **Serve it:** **Connect → Apps** for other programs, **Connect → Devices** for phones and other computers
   ([REMOTE.md](REMOTE.md)).
