# Explaining a reply

Every reply in Studio has an **explain** button. It lists every way OmniCortex can look inside that reply,
in the order to try them:
- what each one answers;
- whether it can run for this model on this computer, and if not, exactly what is missing;
- what it costs.

One click starts it: the check on the reply itself, the Features page, or a job filled in with the
model and the conversation. **start here** marks the first one that can run. The same list is available
at `GET /api/explain/plan?model=…` (`scripts/explain.py`).

| question | method | needs | cost |
|---|---|---|---|
| Is this a confident guess? | H-Neuron risk per token ([PIPELINE.md](PIPELINE.md)) | `llama-cett-dump` and a classifier for the model | one pass |
| Which concepts and emotions run through it? | feature dictionaries and emotion vectors ([FEATURES.md](FEATURES.md)) | PyTorch, and a dictionary or emotion vectors for the model | one pass with hooks |
| Which heads and neurons chose its next word? | attribution, then ablation ([INTERVENTIONS.md](INTERVENTIONS.md)) | PyTorch and the model's Hugging Face weights | a few passes |
| What makes it answer this instead of that? | activation patching between two prompts | as above | one pass per layer and position |
| What chain of features led there? | an attribution graph over transcoders | transcoders for the model | heavy |
| Which neurons carried its answer, on a CPU or an older GPU? | gradient × activation in float32 ([PIPELINE.md](PIPELINE.md#without-cett-older-gpus-and-cpus)) | PyTorch and the HF weights | one forward and backward pass |
| Which neurons can write these words at all? | weights-only attribution from the GGUF | only the GGUF | seconds per layer |

Methods that need the Hugging Face weights use the model's **HF model** setting (its settings → Tokenizer /
HF model).
