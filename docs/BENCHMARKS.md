# External benchmarks

TestQA is the fast, local, per-subject check. These are the standard external
benchmarks, wired to the same models and, where it makes sense, to the same
rolling stats that Studio's `auto` routing and the retraining pipeline use.

| benchmark | for | runs where | feeds stats as |
|---|---|---|---|
| [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark) incl. ImageNetV2, ImageNet-Sketch, VTAB | CLIP / SigLIP, and their H-Neuron edits | local, `scripts/clip_bench.py` | (report only) |
| [LiveBench](https://github.com/LiveBench/LiveBench) | chat models | LiveBench's own runner against any OpenAI endpoint | graded, per category |
| [SWE-bench](https://github.com/SWE-bench/SWE-bench) | coding models | predictions here, scoring in SWE-bench's Docker harness | graded, subject `code` |
| [BenchLM.ai](https://benchlm.ai/benchmarks) and other leaderboards | published vendor scores | import of a CSV/JSON export | **reference only** |

## CLIP_benchmark (ImageNetV2, ImageNet-Sketch, VTAB, ...)

```bash
pip install clip_benchmark
python scripts/clip_bench.py eval --model openai/clip-vit-base-patch32 \
    --dataset imagenet1k imagenetv2 imagenet_sketch imagenet-r vtab/cifar100 vtab/dtd \
    --dataset-root ~/datasets/clip_benchmark \
    --h_neurons models/clip/h_neurons.json --scales 1 0.75 0.5 0 --out runs/clip-bench.json
```

- Any Hugging Face CLIP/SigLIP is wrapped to the open_clip interface, and
  classification runs through CLIP_benchmark's own `zero_shot_classifier` and
  `run_classification` with its class names and templates. Top-1 and top-5 are
  therefore comparable with its published numbers for the same checkpoint.
- With `--h_neurons`, every dataset is evaluated at each `--scale`, and the
  report adds OmniCortex's selective metrics from the same logits: coverage,
  confident error rate, selective accuracy. That shows whether suppression
  helps *out of distribution* (Sketch, ImageNet-R/A, VTAB), which is the real
  test of a profile found on one image set.
- **ImageNetV2** ([modestyachts/ImageNetV2](https://github.com/modestyachts/ImageNetV2))
  downloads automatically. **ImageNet-Sketch**
  ([HaohanWang/ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch))
  and ImageNet-1k need a manual download into `--dataset-root`; see
  CLIP_benchmark's dataset notes.
- **VTAB** ([google-research/task_adaptation](https://github.com/google-research/task_adaptation)):
  `vtab/<task>` names (cifar100, dtd, flowers, pets, svhn, sun397, caltech101,
  eurosat, resisc45, pcam, clevr_count_all, dsprites_*, smallnorb_*, kitti, ...)
  are built through task_adaptation and TensorFlow Datasets:
  `pip install task_adaptation tensorflow tensorflow-datasets`.
- `--dataset imagefolder:/path` evaluates any local ImageFolder; `--limit N`
  takes a random subset for quick runs.

To find H-Neurons on the distribution you benchmark on, export images
(from a *different split* than you evaluate on) and run the vision pipeline:

```bash
python scripts/clip_bench.py export --dataset imagenetv2 --split train --per-class 5 --out data/inv2-folder
python scripts/clip_neurons.py collect --model openai/clip-vit-base-patch32 --images data/inv2-folder --out runs/clip
```

## LiveBench

```bash
git clone https://github.com/LiveBench/LiveBench && pip install -e LiveBench
python scripts/benchmarks.py livebench run --livebench LiveBench \
    --endpoint http://127.0.0.1:7870/v1@my-model --bench live_bench/math live_bench/coding \
    --publish-stats http://127.0.0.1:7870
```

This drives LiveBench's own `run_livebench.py` against the endpoint (Studio
loads the model on demand), then imports `ground_truth_judgment.jsonl`.
Categories map to subjects: coding → code, math and data_analysis → math,
reasoning → logic, language and instruction_following → writing. LiveBench
scores some tasks with partial credit and has no notion of abstaining; full
marks count as correct (`--full-marks` to change the threshold), anything
else as wrong. `livebench import` re-imports results you already have.
Agentic-coding categories need Docker, as in LiveBench itself.

## SWE-bench

```bash
pip install swebench
python scripts/benchmarks.py swebench predict --endpoint http://127.0.0.1:7870/v1@my-model \
    --dataset princeton-nlp/SWE-bench_Lite_bm25_13K --limit 50 --out runs/swe/preds.jsonl
python scripts/benchmarks.py swebench evaluate --predictions runs/swe/preds.jsonl \
    --dataset princeton-nlp/SWE-bench_Lite --run-id my-model-1 --max-workers 2
python scripts/benchmarks.py swebench import --report my-model.my-model-1.json --model my-model \
    --publish-stats http://127.0.0.1:7870
```

- `predict` is the single-shot retrieval baseline from the SWE-bench paper:
  the `*_bm25_*` (or `*_oracle`) datasets carry the issue plus retrieved source
  files in a ready prompt; the model answers with a unified diff in
  `<patch>` tags. It resumes where it stopped. Agent scaffolds such as
  SWE-agent score much higher; their predictions files import the same way.
- `evaluate` calls `swebench.harness.run_evaluation`, which applies each patch
  in a Docker container and runs the repository's tests. It needs Docker and
  plenty of disk (see the SWE-bench README).
- `import` records resolved as correct, empty patches as abstained, and the
  rest as wrong, all under subject `code`.

### Retraining on SWE-bench deficits

Never retrain on the instances you evaluate on. SWE-bench's train split comes
from different repositories than its test sets and carries gold patches, so
`train-data` builds the retraining set from it:

```bash
# the model's own attempts on the train split (for DPO pairs and failure modes)
python scripts/benchmarks.py swebench predict --endpoint ...@my-model \
    --dataset princeton-nlp/SWE-bench_bm25_13K --split train --limit 2000 --out runs/swe/train-preds.jsonl
python scripts/benchmarks.py swebench train-data --dataset princeton-nlp/SWE-bench_bm25_13K \
    --predictions runs/swe/train-preds.jsonl --report my-model.my-model-1.json \
    --exclude princeton-nlp/SWE-bench_Lite princeton-nlp/SWE-bench_Verified \
    --anchors runs/retrain/sft.jsonl --limit 1500 --out runs/swe-retrain
python scripts/finetune.py --model org/base-model --data runs/swe-retrain --method qlora --dpo \
    --max-length 16384 --gradient-checkpointing --out runs/swe-adapter
```

| piece | detail |
|---|---|
| targets | the gold patch in `<patch>` tags, the format `predict` asks for; kept only if it is a well-formed diff |
| failure modes | each train instance the model got wrong is labelled `no_patch` (no diff at all), `malformed_patch`, `wrong_files` (edited none of the files the fix touches) or `wrong_fix` |
| priority | `--report` from a test run says which failure dominates (e.g. mostly empty patches); training instances with that failure come first under `--limit` |
| DPO | gold patch chosen, the model's own patch rejected |
| replay | instances the model already solved, plus `--anchors`, make up the rest at `--deficit-fraction` |
| leakage | `--exclude` drops any instance id that appears in the named test sets |

Retrieval prompts are long (the 13K variants are about 13K tokens), so set
`--max-length` to fit and expect memory to scale with it; `--max-chars` drops
outliers. Measure afterwards on the test set you evaluated before.

## BenchLM.ai and other leaderboards

BenchLM aggregates *published* results across many benchmarks. It is a
leaderboard, not a harness: it says how a vendor's model scored, not how your
local quant or edited copy does. So those numbers are imported as
**reference** stats, shown in Studio's tooltip, and never used by `auto`:

```bash
python scripts/benchmarks.py leaderboard import --file benchlm-export.csv \
    --map qwen3-8b-gguf/qwen3-8b-q6_k=Qwen3-8B --source benchlm
```

The file needs `model`, `benchmark` and `score` columns (`--model-col` etc.
to rename; percentages or 0-1 both work). Benchmark names are mapped to
subjects by keyword (GPQA → science, SWE-bench/HumanEval/LiveCodeBench → code,
AIME/MATH → math, MMLU/SimpleQA → factual, IFEval → writing, MMMU → vision,
ARC/BBH → logic). To measure a model yourself on those suites, use the
harnesses above or TestQA.

## Tested here

The adapters are tested offline: CLIP_benchmark's zero-shot code on a tiny
CLIP (with and without scaling), LiveBench judgment import, SWE-bench
prediction (against a fake endpoint, including resume), report import,
train-data construction (failure modes, priority, exclusion, replay), and
leaderboard import. Real datasets, the SWE-bench Docker harness and LiveBench
runs need network access to Hugging Face and the dataset hosts, which this
environment did not have.
