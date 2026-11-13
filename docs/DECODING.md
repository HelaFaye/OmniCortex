# Decoding lab: SLED and adaptive steering, measured

`scripts/decode_lab.py` loads one Hugging Face model and serves it as
several **variants**, each with its own decoding options, on an
OpenAI-compatible port. The request's `model` names the variant. TestQA
can then grade the variants side by side, and the retrain gate says whether
a variant is better than plain decoding.

Nothing here is on by default. These are research methods with mixed
published results, and they are worth keeping only where your measurements
say so.

```bash
python scripts/decode_lab.py --model org/m --port 8090 \
    --variant plain='{}' \
    --variant sled='{"sled": {"alpha": 2, "k": 10}}' \
    --variant calm='{"steer": {"vectors": "runs/dk/directions.pt", "layer": 14, "alpha": 4, "adapt": "spread"}}'
python scripts/testqa.py --endpoint plain=http://127.0.0.1:8090/v1@plain \
    --endpoint sled=http://127.0.0.1:8090/v1@sled --endpoint calm=http://127.0.0.1:8090/v1@calm \
    --per-subject 30 --out runs/decoding.json
python scripts/retrain_gate.py --results runs/decoding.json --before plain --after sled
```

In Studio: Jobs → *Decoding lab* starts the server, and it runs until you
cancel it. Then run Jobs → *TestQA* against its variants, and Jobs → *Gate*.
To chat with a variant, link it as an OpenAI-compatible endpoint, or try
all variants once from the command line:
`python scripts/decode_lab.py --model org/m --prompt "…" --variant …`.

The model stays in memory once, and requests are answered one at a time.
Each answer also returns `decode_lab.spread` (per-token uncertainty, as in
[CONFIDENCE.md](CONFIDENCE.md)) and the steering strength used per token.

## SLED: self logits evolution decoding

Zhang et al., NeurIPS 2024. Earlier layers often "know" the right token
before the last layers settle. SLED compares the logit-lens logits of
earlier layers with the final ones, over the final top-k tokens:

- for each layer n and candidate i: `m_i(n) = max(0, cos(logits_n − logits_N, p_N − e_i))²`.
  This is high when moving from layer n to the final layer went toward
  token i;
- the layers' normalised `m` are averaged, weighted by each layer's total,
  into a latent distribution;
- the final logits take one step toward it,
  `logits_N[i] −= alpha·(p_N[i] − P_latent[i])`. Tokens outside the top k
  are dropped.

Options:

| option | meaning | default |
|---|---|---|
| `alpha` | evolution rate: how far the step goes | 2 |
| `k` | evolution scale: candidates considered | 10 |
| `layers` | `"even"` (every second layer below the last), `"all"`, or a list | `"even"` |
| `temperature` | for p_N | 1 |

This follows the paper's description. The authors' code differs in
details, so compare it with `plain` on your model before trusting it.

## Steering with adaptive strength

`steer` adds `alpha·v` to the residual stream after one layer at every step:
- `v` is a direction from `concept_edit.py` (`directions.pt`; the position
  after that layer is used), or any `.pt` / `.npy` vector;
- a positive `alpha` adds the concept, a negative one pushes away from it.

| option | meaning | default |
|---|---|---|
| `vectors`, `layer` | the direction and where it is added | required |
| `alpha` | full strength | 1 |
| `direction` | which column of a multi-direction cone | 0 |
| `adapt` | `fixed`, `spread` or `attention` | `fixed` |
| `target` | the uncertainty at which the full alpha applies | 0.5 |
| `min`, `max` | bounds on the scale | 0, 2 |

With `adapt`, the strength follows the model's uncertainty on the previous
token: `alpha_t = alpha · clip(u / target, min, max)`. Here `u` is either
the output spread, or the mean entropy of the last layer's attention at the
current position over log(context length). The second needs eager
attention, which the lab turns on by itself and which is slower.

This is the "context-aware α" idea: steer hard where the model hesitates,
and leave it alone where it is sure. It suits a direction like *declining
when unsure* from CONCEPT_EDITING.md: the model leans toward "I don't
know" only when it has started to guess.

## Steering in a random low-dimensional projection

Suppressing single H-Neurons can disturb what else those neurons do.
`scripts/rp_steer.py` works on the whole MLP output of a layer instead
(Jobs → *Random-projection steering: fit*):

1. **Fit.** From answers labelled right and wrong (a JSONL of prompt,
   reply and label, or TestQA results with `--testqa`), it averages each
   chosen layer's MLP output over the reply's tokens and projects it onto
   `--dim` random orthonormal directions. In that space it finds the
   direction separating wrong from right (LDA with shrinkage, which a small
   space makes reliable from a few hundred examples).
2. **Check.** For each layer it prints the held-out AUC in the
   projection, the same classifier's AUC on the full space, and the spread
   over `--projections` random projections. A projection keeps only part of
   a single direction's signal, about √(k/d) against even noise, so a large
   gap to the full space means raise `--dim`, or that the signal is not
   spread widely in this model. A small spread across projections is the
   redundancy the method relies on.
3. **Steer.** Add an `rp` variant to the decoding lab:

```bash
python scripts/rp_steer.py fit --model org/m --testqa runs/testqa.json --endpoint base --layers 12 16 20 --out runs/rp.pt
python scripts/decode_lab.py --model org/m --variant plain='{}' \
    --variant rp='{"rp": {"file": "runs/rp.pt", "alpha": 1.0, "mode": "direction"}}'
```

At each step the layer's MLP output is scored along the fitted direction.
Only when it drifts past the right answers' typical level is it pulled back:
along the direction (`direction`), or across the whole projected subspace
toward the right answers' mean, gated by how far it drifted (`subspace`).
Each reply reports `rp_drift` per token, which shows where the model leaned
toward its wrong-answer pattern. Compare `plain` and `rp` with TestQA and
the retrain gate before trusting it.

`rp_steer.py export` writes the directions as a llama.cpp control vector
(`llama-server --control-vector-scaled rp-cvec.gguf 0.5`) for GGUF models.
That one is fixed: llama.cpp adds it at every step, without the gate.

### Measured once: no gain on a small model

One run, on Qwen2.5-0.5B-Instruct with Natural Questions (NQ-open) short
answers:

- **Fit**: 599 labelled answers, `--dim 48`, 8 projections, layers 6, 12
  and 18.
  - Held-out AUC of the chosen projection was 0.55 to 0.67 by layer, and
    the mean over projections 0.55 to 0.62. The same classifier on the full
    space scored 0.64 to 0.69.
  - Across projections the AUC spread by 0.18 to 0.27.
- **Steer**: layers 12 and 18 at `alpha` 1 and 2, on 300 other questions.
  - Right answers went from 12% (plain) to 11% and 11%.
  - At alpha 1, 0 answers became right and 2 became wrong; at alpha 2, 1
    became right and 4 became wrong. Both are too few changes to call.

The method needs a wrong-answer signal that is strong and spread over many
directions, so that any random projection keeps it. Here it was neither:
AUC stayed well under 0.7, and the result depended on which projection was
drawn. A model that gets 88% of these questions wrong also has little
right-answer behaviour to steer toward.

Before steering, read the fit's numbers. A full-space AUC near 0.5, or a
wide spread across projections, means steering has little to work with on
that model and layer. Try a larger model, or other layers.

## Reading the result

The gate compares the variants item by item on the same TestQA bank, and
reports right, wrong (hallucinated) and declined rates per subject with
significance tests. A useful variant:
- raises right answers or declines without raising wrong ones;
- breaks no subject.

Decoding changes cost speed too: SLED computes the output head for each
layer it reads, so expect it to be noticeably slower. Each answer reports
tokens per second.

## Not here (yet)

- **SLED or adaptive steering inside llama.cpp:** the engines Studio runs
  don't expose per-layer logits or per-token hooks to callers. GGUF models
  get the measurement half (the confidence strip) but not these decoders.
  Control vectors (fixed strength) work in llama.cpp; see sage_edit.py
  export.
