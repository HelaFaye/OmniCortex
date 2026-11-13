# Interventions: which components drove an output

`scripts/interventions.py` answers "why did the model say that?" at the level of attention heads, MLP
neurons, layer outputs and SAE features. It works on the Hugging Face models OmniCortex already loads
(Llama, Qwen, Gemma, Mistral, GPT-2, GPT-NeoX…). You can run it from:
- the command line;
- **Jobs → Interpret → Interventions**;
- agents, through the MCP tool `intervene`.

Every result is one JSON document with `schema_version` and `operation`, so a job, a script and an agent
all get the same thing.

| operation | what it does |
|---|---|
| `trace` | The predicted token (or `--target`), with attention heads and MLP neurons ranked by **direct logit attribution**: each one's contribution to that token's logit, read through the final norm as it was on this prompt. Neurons are also listed by raw activation. For models without biases (Llama, Qwen…), the heads, neurons and token embedding add up exactly to the logit; the tests check this. |
| `activations` | Per layer: shape, mean, std, min, max and the position of the largest norm, for the residual stream, attention output, MLP output and neurons. |
| `patch` | Ablate one thing and report the logit change, the probability change, whether the prediction flipped, and the KL divergence. The thing can be a layer's input or output residual, attention output, MLP output or neurons, one head, one neuron, or one SAE feature. **Zero** ablation sets it to zero. **Mean** ablation sets it to its average over the prompt, or over `--mean-prompt` texts, which keeps it in its usual range. `--positions last`, `all`, `3`, `2,5,-1` or `0:4` picks the tokens. |
| `act-patch` | **Activation patching** with a clean prompt and a corrupted one, for example "…John gave a drink to" and "…Mary gave a drink to". Each clean activation goes into the corrupted run, one at a time: by layer and position (`--by resid|attn|mlp`) or by head (`--by head`). The score is the share of the clean-vs-corrupted logit difference (`--target` minus `--baseline`) it restores: 1 restores the clean answer completely, 0 changes nothing. |
| `path-patch` | **Path patching**: the corrupted output of one head (`--sender 9.6`) reaches only the receivers you name (`--receiver 10:attn`, `11:mlp`); every other component keeps its clean value. This measures that head's effect through those receivers alone, which single-component ablation cannot separate. |
| `circuit` | A sketch. Candidates are ranked by attribution and each one is ablated on its own. With `--corrupt`, each candidate head is also scored by activation patching. |
| `neuronpedia` | A feature's descriptions from [Neuronpedia](https://www.neuronpedia.org), e.g. `--np-model gemma-2-2b --np-layer 12-gemmascope-res-16k --index 1234`. |

```bash
python scripts/interventions.py trace --model Qwen/Qwen3-0.6B --prompt "The capital of Japan is" --top-k 5
python scripts/interventions.py patch --model gpt2 --prompt "The Eiffel Tower is in" --layer 9 --component head --index 8
python scripts/interventions.py act-patch --model gpt2 \
  --clean "When Mary and John went to the store, John gave a drink to" \
  --corrupt "When Mary and John went to the store, Mary gave a drink to" --target " Mary" --baseline " John" --by head
python scripts/interventions.py path-patch --model gpt2 --clean "…" --corrupt "…" --target " Mary" --baseline " John" \
  --sender 9.6 --receiver 10:attn --receiver 11:attn
python scripts/interventions.py patch --model org/m --prompt "…" --layer 12 --sae runs/features/m-resid-12 --feature 3120 --value 0
```

A token argument can be text (its first token is used) or `#ID` for a token id.

## What each measure can and cannot tell you

- **Attribution is linear.** It says how much a component wrote toward a token through the direct path.
  A component that matters by changing what *later* layers do shows up in ablation and patching, not in
  attribution.
- **Ablation and patching here are one component at a time.** They do not measure interactions: two heads
  that back each other up can each look unimportant when removed alone.
  - `path-patch` measures one path.
  - For full circuits, use attribution graphs over transcoders: `feature_circuits.py graph`
    ([FEATURES.md](FEATURES.md#attribution-graphs-circuits)). They follow Anthropic's circuit-tracer method.
- **Zero ablation can push a component far outside its normal range.** Mean ablation is usually the
  fairer test.
- **Activation patching needs prompts of the same token length.** Positions are compared one by one,
  and the tool says so when the lengths differ.
- **SAE features.** `patch --sae` sets one feature and keeps the SAE's reconstruction error, so only that
  feature's decoder direction changes the stream. To steer with features and emotion vectors while
  generating, use the Features page ([FEATURES.md](FEATURES.md)).

## Dictionaries from SAELens

`features.py import --release <release> --sae-id <id>` (Jobs → Features: import a dictionary) downloads any
dictionary in [SAELens](https://github.com/decoderesearch/SAELens)'s catalogue of pretrained releases from
Hugging Face. It covers Gemma Scope, Llama Scope and the rest. The catalogue is read from SAELens's
repository and cached for a week. `--path` still takes a folder you already have.

## Where these come from

`trace`, `activations`, `patch` and `circuit` follow
[RudrenduPaul/NeuronScope](https://github.com/RudrenduPaul/NeuronScope) (MIT). It wraps TransformerLens to do
the same single-prompt operations from a CLI and an MCP server, and its README is clear about what
single-component ablation does not capture. This implementation:
- needs no TransformerLens;
- works on the models OmniCortex serves and edits;
- adds the deeper tools that README points to: mean ablation, position-resolved activation patching with
  clean and corrupted prompts, path patching, SAE-feature ablation and Neuronpedia lookups.

It is also the reason this project is no longer called OmniCortex.
