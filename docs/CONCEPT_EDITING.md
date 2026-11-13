# Concept editing: finding, removing and protecting directions

OmniCortex's H-Neuron suppression scales a few dozen individual neurons.
Two tools here work on *directions*, the combinations of many neurons that
models use for a concept:

| tool | changes | result | serve with |
|---|---|---|---|
| `concept_edit.py` | the weights that write into the residual stream (W_O, W_out) | a new model without the concept's direction | anything, after GGUF conversion |
| `sage_edit.py` | nothing in the weights: small gated edits to attention heads, learned for a task | the task learned, with a chosen subspace provably untouched | transformers exactly; llama.cpp approximately (control vector) |

They are also in Studio: Jobs → Retrain → *Concept edit*, *SAGE: adapt with
a protected subspace*, *SAGE: export a control vector*.

## Prompt sets decide everything

Both tools start from two prompt sets that differ in one concept (A) and
not otherwise (B). The direction found is *whatever separates your two
sets*. If A is all questions about history and B all about cooking, you
will find "history vs cooking". Make them matched:
- the same topics and lengths;
- only the concept differs;
- a few hundred of each is better than a few dozen.

Useful pairs for OmniCortex's goals:

| concept | A | B |
|---|---|---|
| confident guessing | questions the model answered wrong (TestQA `wrong`) | questions it declined (`abstained`) |
| knowing | questions answered right | questions answered wrong |
| sycophancy | prompts asserting a false premise as the user's belief | the same questions asked neutrally |
| verbosity | prompts asking for detail | the same asking for brevity |
| refusal | prompts the model refuses | matched prompts it answers |

## concept_edit.py

```bash
# where does it live? (nothing edited)
python scripts/concept_edit.py --model org/m --concept a.txt --baseline b.txt --out runs/ce --analyze
# one edit, by hand
python scripts/concept_edit.py --model org/m --concept a.txt --baseline b.txt --neutral n.txt \
    --kernel 1.0,18,0.4,8 --direction-layer 16 --out runs/ce
# optimized: the Heretic approach
python scripts/concept_edit.py --model org/m --concept a.txt --baseline b.txt --neutral n.txt \
    --search 40 --max-kl 0.3 --marker "I (cannot|can't)" --out runs/ce
python scripts/merge_export.py --base org/m --adapter runs/ce/model --out runs/ce/gguf --gguf Q4_K_M
```

1. **Activations.** The residual stream at the last prompt token, after
   every layer, for both sets.
2. **Subspace.**
   - `--method mean`: the difference of means, one direction per layer.
   - `--method pca --k 3`: a three-dimensional cone around the baseline
     mean, whose first axis is close to the mean difference. Concepts such
     as refusal span several such dimensions (Wollschläger et al., 2025).
   - **Biprojection** (on by default) removes from each direction its
     component along the baseline's own mean, so the edit does not also
     take away what B has.
3. **Edit.** W' = W − w·B·Bᵀ·W on attention output and MLP output: nothing
   the layer writes points along the subspace any more.
   - **Norm preservation** (on by default; `--plain` turns it off): each
     column is rescaled to its original length, so later layers see inputs
     of the usual size. This is what keeps maths and knowledge benchmarks
     close to the original in published comparisons (Young, 2026,
     arXiv:2512.13655).
   - The strength per layer follows a kernel: `max_w` at layer `pos`,
     falling to `min_w` at `dist` layers away. `--direction-layer` uses one
     direction everywhere, taken after that (fractional) layer; without it,
     each layer uses its own.
4. **Damage meter.**
   - **KL divergence** between the original and edited next-token
     distributions on the neutral set. Below 0.1 is close to untouched;
     above 1 is a different model.
   - **Remaining separation**: how much A still differs from B along the
     subspace, 1.0 = unchanged.
   - With `--marker`: how often greedy replies to A match a regex.

`--search N` optimizes the kernel and the direction layer (Optuna TPE when
installed). It minimizes remaining separation (plus the marker rate) plus
`--kl-weight` × KL, and rejects anything over `--max-kl`. `edit.json` keeps
every trial; `directions.pt` keeps the subspace for SAGE or another run.

Check the result the way any changed model is checked: TestQA on the
holdout, then `retrain_gate.py` against the original. KL measures how much
changed, not whether what changed was good.

Removing a model's safety behaviour, such as refusal, is possible with this
tool and is your responsibility. It is done under the model's licence and
the law where you use it.

## sage_edit.py (SAGE)

Nonta & Rambhatla (ICLR 2026 TTU workshop), *Refusal-Orthogonal Gated
Editing for Safer Localized LLM Adaptation*. The paper protects refusal
while adapting a model to a task. Here any subspace can be protected. The
one most useful to OmniCortex is the decline direction, so that retraining
on deficits does not erode the model's willingness to say it does not know.

```bash
python scripts/concept_edit.py --model org/m --concept declined.txt --baseline guessed.txt --out runs/dk --analyze
python scripts/sage_edit.py train --model org/m --protect runs/dk/directions.pt --data runs/retrain --out runs/sage
python scripts/sage_edit.py chat --edits runs/sage/edits.pt "Who won the 1987 Tour de France?"
python scripts/sage_edit.py export --edits runs/sage/edits.pt --cvector runs/sage/sage.gguf
```

- **Stage 1**: the protected subspace R, from directions.pt or from
  `--protect-concept` / `--protect-baseline`.
- **Stage 2**: every attention head gets:
  - an additive edit and a multiplicative edit;
  - two Hard-Concrete gates with an L0 penalty, so most heads stay
    unedited.

  Edits act only on the part of each head's output outside R:

      z' = P_R z + P_R⊥ ((1 + g_m·m) ⊙ P_R⊥ z + g_a·a)

  The parameters are projected onto R⊥ after every step. After training,
  the script runs the model and reports the largest change it saw along R
  (`max_change_along_protected`, which should be about 0), and the
  validation loss before and after.
- **Serving.**
  - `chat` (transformers) applies the edits exactly.
  - `export` writes the gated additive edits, summed per layer, as a
    llama.cpp control vector. Load it with
    `--control-vector sage.gguf` in a model's Extra flags. That is an
    approximation, for three reasons:
    - llama.cpp adds the vector at the end of the layer rather than after
      attention;
    - it has no vector for layer 0;
    - multiplicative edits are dropped.

  Measure the exported version separately.

## Heretic

Heretic is a published tool that automates abliteration: it searches
ablation strength per component and the direction layer, scoring KL against
refusals. `concept_edit.py --search` follows the same parameterization
(peak weight, its position, floor weight, distance, fractional direction
index; see a Heretic run's saved `parameters`). The differences:
- it adds norm preservation and biprojection;
- it adds PCA cones;
- it works for any concept pair, not refusal only;
- it reports remaining separation when there is no behaviour regex.

Heretic itself is a separate project. Models it produces load in OmniCortex
like any other, and can be checked the same way.
