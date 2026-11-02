# Merging and MoE sweeps

## Selective weight merging

`scripts/merge_selective.py` transplants or blends weights at neuron
granularity, guided by the H-Neuron map or a layer range.

```bash
python scripts/merge_selective.py --base A --donor B \
    --h-neurons models/h_neurons.json --dry-run
python scripts/merge_selective.py --base A --donor B --layers 8:24 \
    --alpha 0.5 --assert-lineage --output_path models/merged
```

**A neuron is three weight vectors, not one.** In a gated MLP, neuron *j* is
defined jointly by `gate_proj[j,:]`, `up_proj[j,:]` and `down_proj[:,j]`.
Splicing only the `down_proj` column pairs the base's activation pattern with
the donor's output projection, producing a neuron present in *neither* model.
Verified: a full triple transplant reproduces the donor neuron exactly, while a
`down_proj`-only splice matches neither source. The output stays fluent either
way, so nothing but an evaluation would catch it. This tool always moves all
three.

**Merging needs lineage, not just matching shapes.** Same tiering as
`compare.py`: identical fingerprints merge freely, different fingerprints with
matching geometry require `--assert-lineage`, and differing geometry is refused.
Two independently trained models have no neuron correspondence, so a merge
between them is noise -- fluent noise, which is worse.

`--dry-run` reports per-layer divergence on the selected neurons before writing
anything. Near-zero divergence means the merge is a no-op and not worth an
evaluation run.


## Merge methods, ranked

`scripts/merge_methods.py` lists all 18 methods in the installed mergekit,
ordered by how likely each is to work on a first attempt, and generates a valid
config for any of them.

```bash
python scripts/merge_methods.py                    # ranked list with rationale
python scripts/merge_methods.py --short            # names and summaries only
python scripts/merge_methods.py --verify           # check against your mergekit
python scripts/merge_methods.py --method slerp --models A B
python scripts/merge_methods.py --method ties --base BASE --models A B C
```

**Reach for first:** `slerp` (exactly two models -- averaging flattens weight
magnitudes, arc interpolation preserves them), `ties` (three or more, with a
base -- handles the sign conflicts and magnitude swamping that break naive
averaging), `dare_ties` (usually a small win over TIES for the same cost).

**Narrower use:** `task_arithmetic` -- still the right tool for *removing* a
behaviour, since a negative weight subtracts one. `linear` when the models are
already very close. `model_stock` when you genuinely have three or more
fine-tunes of one base.

**With a reason:** `dare_linear`, `della`, `della_linear`, `breadcrumbs`,
`breadcrumbs_ties`, `nuslerp`, `multislerp`, `karcher`, `sce`, `nearswap`,
`arcee_fusion`.

**Special cases:** `passthrough` is not a merge -- it stacks layer ranges and
changes the parameter count, so nothing else in this toolkit applies to the
result. `neuronscope_select` is ours and needs a trained classifier first.

The generator enforces each method's arity and base requirement, so
`slerp` with three models or `ties` without a base fails immediately rather
than at merge time.

Whatever you pick, run the result through `merge_eval.py`. A merge that
retained nothing still produces fluent text; a paired comparison against the
sources on untuned tasks is the only evidence you get.

## mergekit integration

mergekit already does general merging well -- TIES, DARE, SLERP, task
arithmetic, model stock, SCE -- and reimplementing any of it would be waste.
What OmniCortex adds is a different **selection criterion**.

TIES and DARE decide what to keep by *magnitude*: large deltas survive, small
ones are pruned or randomly dropped. That is a statistical heuristic with no
reference to what the parameters do. OmniCortex selects by a classifier trained on
measured behaviour. Semantic selection instead of magnitude selection, and to
my knowledge nobody has tried it as a merge criterion.

`scripts/mergekit_neuronscope.py` registers `neuronscope_select` via mergekit's
`@merge_method` decorator:

```bash
pip install -e /path/to/mergekit
export NEURONSCOPE_NEURONS=models/h_neurons.json
python scripts/mergekit_neuronscope.py            # prints a config to start from
mergekit-yaml merge.yaml ./out
```

```yaml
merge_method: neuronscope_select
base_model: org/base-model
models:
  - model: org/finetuned-model
    parameters: {weight: 1.0}
parameters:
  invert: false      # true merges everything EXCEPT the H-Neurons
```

`invert: true` is the more interesting run: keep the base's H-Neurons and take
the donor everywhere else. If the H-Neurons matter, the two directions should
behave differently, and that difference is the experiment.

Attention, norm and embedding tensors pass through as the base unchanged -- the
neuron index describes the MLP intermediate dimension and means nothing
elsewhere. gate/up are indexed by row, down_proj by column, so the complete
triple moves together (see `merge_selective.py` for why a partial one is
silently wrong).

**On mergekit-moe.** frankenMoE routers are randomly initialised or derived from
prompt hidden states, not trained. If a genuinely trained MoE of similar active
size exists for your model family, assembling one from dense copies is very
likely worse than just using it.

**Lineage is unchecked.** mergekit will merge any two models with matching
shapes. Neuron indices correspond only if both were post-trained from the same
base -- e.g. two fine-tunes of the same base checkpoint -- and false in general.
Neither mergekit nor this plugin can verify it.

## Merging and measuring the result

`scripts/merge_eval.py` compares any set of served endpoints on the same tasks
and reports **what improved and what regressed, per item**. It does not care how
the variants were produced -- mergekit, `merge_selective.py`, a fine-tune, or
just two checkpoints. Serve each and point at it.

```bash
python scripts/merge_eval.py \
    --endpoint base=http://127.0.0.1:8080 \
    --endpoint merged=http://127.0.0.1:8081 \
    --tasks data/eval_tasks.jsonl --reference base --out cmp.json
```

Aggregate accuracy hides the answer. Going from 60% to 64% might be 12 fixed
and 8 broken, or 4 fixed and 0 broken -- identical headline, completely
different models. Verified: those two cases give chi2 0.80 and 4.00
respectively, one not significant and one significant. The tool prints the
decomposition and names the regressions.

Three task kinds, mixable in one file:

```
{"id": "qa-1",  "prompt": "...", "aliases": ["Paris"]}      graded on aliases
{"id": "cod-1", "prompt": "...", "modules": ["requests"]}   graded by symbol resolution
{"id": "can-1", "prompt": "..."}                            canary: answered or refused
```

**Include canaries.** They are ungraded general-capability prompts, and they
catch the failure mode that a target metric cannot: a merge whose hallucination
rate improves because the model now refuses more. In the test above a variant
scored +1 net while newly refusing 3 of 4 canaries -- narrower, not better, and
invisible without them. `data/eval_tasks.example.jsonl` shows the format.

## MoE active-parameter sweeps

The top-k expert count is baked into the GGUF as `<arch>.expert_used_count`,
and llama.cpp lets you override it at load time -- so you can trade quality
against speed with no re-quantization:

```bash
python scripts/sweep_experts.py \
    --gguf ~/models/Qwen3-30B-A3B-Q4_K_M.gguf \
    --server ~/llama.cpp/build/bin/llama-server \
    --tasks data/eval_tasks.jsonl --experts 2 4 6 8 --ngl 99 \
    --override-tensor '\.ffn_.*_exps\.=CPU'
```

It reads the architecture and expert count from the GGUF itself (the override
key is `qwen3moe.` for one model and `glm4moe.` for another, so hardcoding it
breaks), launches a server per value, measures tokens/sec, and runs the same
paired comparison as `merge_eval.py`.

Read it as a curve, not a winner: if k=6 matches k=8 with no significant
difference and runs faster, the extra experts were buying nothing.

**Dense models have no such knob** and the script says so rather than failing
obscurely; it needs an MoE checkpoint (e.g. a 30B-A3B).

The `--override-tensor` regex above is the standard consumer-hardware MoE
recipe: routed experts stay in RAM, always-active tensors (attention, shared
experts, embeddings) go to the GPU. Generation is bandwidth-bound on *active*
parameters, so a Q4 30B-A3B reads roughly 2-3 GB per token against ~8 GB for a
dense Q6 9B -- potentially faster despite being three times the size.
