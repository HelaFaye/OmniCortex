# Confidence signals: how sure the model was, token by token

OmniCortex has three ways to see where a reply may be wrong. They measure
different things and cost different amounts:

| signal | what it reads | needs | cost | where |
|---|---|---|---|---|
| **Confidence strip** | the final layer's top-5 next-token probabilities, as the engine produced them | llama-server or vLLM (any model) | nearly none | chat: tick *confidence* |
| **Logit lens** | every layer's guess for each token: how spread out, and where the answer settled | a Hugging Face copy of the model | one extra pass, on transformers | Jobs → *Logit lens over a reply*, `scripts/logit_lens.py` |
| **H-Neuron check** | the MLP neurons a classifier associated with confident wrong answers | llama-cett-dump and a classifier for the model | one extra pass | *check* under a reply (3D view) |

## Confidence strip

With *confidence* ticked (on by default), Studio asks the engine for each
token's five most likely candidates. This doesn't change what is sampled.
For each token it shows:
- **spread**: the entropy of those five plus the rest, divided by log 6. 0 is
  certain; 1 is evenly spread.
- **p**: the probability of the token the model actually wrote.

A token is marked when its spread is 55% or more, or its p is under 35%.
*show tokens* shades every token by its spread; hover a token for its
numbers.

What it catches: the model hesitating between candidates (names, numbers,
dates), and sampling picking an unlikely token. What it misses: a model
that is confidently wrong. Confident hallucinations often have low spread,
which is what the next two signals are for.

Hosts that call in (Remote) don't pass the probabilities on yet. Linked
Studios and other OpenAI-compatible servers do when their engine supports
`logprobs`.

## Logit lens

```bash
python scripts/logit_lens.py --model org/m --prompt "Who wrote Hamlet?" --reply "Christopher Marlowe." \
    --out runs/lens.json --html runs/lens.html
python scripts/logit_lens.py --model org/m --prompt "Who wrote Hamlet?" --generate 40 --out runs/lens.json
```

For each token of the reply and each layer, the residual stream goes
through the model's final norm and output head (the logit lens). The result
gives, per token:
- the **spread** at every layer, the same measure as the strip;
- the **settle layer**: the first layer from which every later layer agrees
  with the final answer;
- the **decay layer**: where the spread first falls below half its peak.

A token is **marked** when it settles late (after `--late`, 75% of the depth
by default) and its final spread is still at least `--spread` (0.55).

The heatmap page shows tokens across and layers up, green for sure and red
for spread. A line marks where each token settled. Hover a cell for that
layer's top guess.

Why settling matters: facts the model holds tend to appear in middle layers
and stay. Answers assembled at the very end, with the final layers still
undecided, are where confident-sounding errors cluster. This is the
intuition behind the "associated hallucination" work and the
early-exit/contrastive decoders (DoLa, SLED).

Treat the marks as places to look, not verdicts:
- the plain lens is crude on early layers;
- tuned lenses, or Studio's per-model H-Neuron classifier, are better
  calibrated;
- check any threshold you rely on against TestQA results for your model.

## Can these tell dishonesty from a confident mistake?

Not on their own:
- All three measure **uncertainty or risk** in what the model produced.
- A model that "knows" the right answer and writes another, and a model that
  believes a wrong answer, can look the same on the final layer.

What can separate them is a probe for the model's own belief:
1. Build matched statement sets the model knows to be true and false (TestQA
   items it answers right, rephrased as statements).
2. Find the direction that separates them:
   `concept_edit.py --analyze --concept true.txt --baseline false.txt`.
3. Read that direction on the claims in a reply.

A claim the probe reads as *false* while the model asserts it looks like
dishonesty. A claim the probe can't place, with a late-settling lens, looks
like a confident mistake.

The logit lens adds one hint: an answer whose middle layers held a
different top-1 than the final layer is worth a look for this.

OmniCortex doesn't label replies "dishonest". The probe has to be
validated per model before such a label would mean anything.
