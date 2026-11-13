# Features: what a model represents and how it decides

H-Neurons and CETT look at single neurons. A neuron, though, takes part in
many unrelated things, so reading neurons one by one rarely explains a
model. The tools here follow Anthropic's interpretability research and work
with **features** instead. A feature is a direction in a layer's
activations that stands for one thing.

| Tool | What you get | Where |
|---|---|---|
| **Feature dictionaries** (sparse autoencoders) | The features of one layer. For each one: its strongest examples, examples across activation levels, the tokens it raises and lowers, its nearest features and a map of all of them | Studio → **Features → Dictionaries** |
| **Feature → neuron map** | The neurons that write, read and fire with each feature, and how much of it sits on H-Neurons | the same feature card |
| **Read a reply** | For each token of any text or reply: the active features, the emotion activations, and what every layer would have said next (logit lens) | **Features → Read a reply** |
| **Attribution graphs** (features as computational intermediates) | The features the model used to produce an output and how they feed each other; turn any of them off or up to test it | **Features → Circuits** |
| **Emotion vectors** | Directions for 171 emotion concepts (or your own), where they are active, how they relate, and how they react to a prompt | **Features → Emotions** |
| **Steering** | Replies with features clamped or emotions added or removed, next to the reply without | **Steer** tab, every feature card, the decoding lab |

Everything is optional and runs only when you start it. Studio's page,
the jobs and the scripts do the same things. The methods and their sources
are listed in [CREDITS.md](CREDITS.md).

## What it needs

- **A Hugging Face copy of the model** (a folder, or an id). Activations
  and gradients come from PyTorch, as for the rest of the H-Neuron
  pipeline; GGUF files cannot provide them. The results apply to the GGUF
  made from the same weights.
- **A GPU helps but is not required.** Every number on this page comes
  from a 4-core CPU with Qwen2.5-0.5B-Instruct (a 0.5B model). Bigger
  models need proportionally more memory and time.
- **The decoding lab** for the interactive parts: reading replies, graphs,
  steering. It is one job (Jobs → *Decoding lab*), or the **Start lab…**
  button on the Features page. It keeps the model loaded and answers only
  on this computer.

## Feature dictionaries

### 1. Record activations

`features.py collect` runs the model over text and stores what chosen
layers produce. You pick the layers and these **sites** within each layer:

| Site | What it is | Use it for |
|---|---|---|
| `resid` | the residual stream after the layer | general-purpose dictionaries (what Scaling Monosemanticity used, at the middle layer) |
| `mlp_in`, `mlp_out` | an MLP's input and output | transcoders, for attribution graphs (record both) |
| `neurons` | the MLP's hidden neurons, the ones H-Neurons and CETT measure | a dictionary in neuron space; correlating features with neurons |

The corpus can be any of these:

- a folder of `.txt` or `.md` files;
- a `.txt` file, split into paragraphs at blank lines;
- a `.jsonl` file (text, messages, or prompt plus reply), which can be your own chats or TestQA cache;
- a Hugging Face dataset id, such as `NeelNanda/pile-10k`, a 10k-document sample of The Pile.

How much text to record:

- **A first look:** a few hundred thousand tokens.
- **A dictionary worth trusting:** millions.

With 60,000 tokens, a 3,584-feature dictionary on Qwen2.5-0.5B scored as
follows:

- **Fraction of variance unexplained (FVU):** 0.08 on the training tokens, but 0.24 on held-out text. It had memorised its training text.
- **Next-token loss:** rose from 2.78 to 3.46 with the dictionary spliced in.

Even so, most of its features were readable. Examples include "at least",
"recording", the possessive "'s", hotel and evening stays, "feed conversion
ratio" and the drug name "entecavir".

The first position of each window is left out, as its activations are
outliers (the attention sink).

### 2. Train a dictionary

`features.py train` learns a sparse autoencoder:

- the features are `f = act(W_enc x + b_enc)`;
- the reconstruction is `x̂ = W_dec f + b_dec`.

There are three activation functions:

| `--arch` | How sparsity is enforced | Source |
|---|---|---|
| `topk` (default) | exactly *k* features per token; an auxiliary loss revives dead features | Gao et al. 2024 |
| `relu` | an L1 penalty weighted by each decoder column's norm; the decoder is unconstrained and inputs are scaled to mean norm √d | Anthropic's April 2024 update, used in Scaling Monosemanticity |
| `jumprelu` | a learned threshold per feature, trained with an L0 penalty through straight-through estimates | Rajamanoharan et al. 2024, the form Gemma Scope uses |

Training follows the April 2024 update:

- Adam;
- the sparsity coefficient ramps up over the first 5% of steps;
- the learning rate decays linearly to zero over the last 20%;
- gradients are clipped at 1;
- decoder rows are initialised at norm 0.1, with the encoder as their transpose;
- `relu` dictionaries finally fold the decoder norms into the encoder.

For `relu` and `jumprelu`, give `--target-l0`, the number of features
active per token you want. Training then adjusts the coefficient toward
it instead of you tuning it.

`jumprelu` thresholds start from the data:

- each starts where it leaves about twice the wanted features active;
- the kernel bandwidth starts at a twentieth of a typical threshold.

The JumpReLU defaults assume inputs normalised to unit norm, and these
starting points adapt them to our √d normalisation. Without them, short
runs never moved their thresholds.

On synthetic data built from 256 known directions, all three
architectures recovered every direction (`tests/test_features.py`).

After training, two checks run on held-out text:

- **Fraction of variance unexplained (FVU)** and **L0**, the number of active features per token.
- **Next-token loss with the dictionary spliced in** (`--check-model`). For MLP sites it also reports the **share of loss recovered**, against zeroing that MLP's output.

A **transcoder** reads one site and reconstructs another
(`--site mlp_in --target-site mlp_out`). Its features stand in for the
MLP's computation, which is what attribution graphs need.

| Run | Held-out FVU | Next-token loss | Recovered |
|---|---|---|---|
| resid, layer 12, topk k=32, 3,584 features, 60k tokens | 0.24 | 2.78 → 3.46 | |
| transcoder, layer 6 | | 2.74 → 2.77 | 84% |

**Bringing a dictionary trained elsewhere.** `features.py import` reads
these formats, and the dictionary is then used like your own:

- SAELens folders (`cfg.json` and `sae_weights.safetensors`; the layer and site come from the hook name);
- Gemma Scope `params.npz`;
- any `.safetensors`, `.npz` or `.pt` file with `W_enc`, `W_dec`, `b_enc`, `b_dec` (and `threshold`).

### 3. Describe the features

`features.py interpret` goes over the recorded tokens once and writes, for
every feature:

- **how often it fires**, its strongest activation, and a histogram of its activations;
- **examples**: the strongest ones, plus examples sampled from activation buckets spaced evenly between zero and the maximum, as Scaling Monosemanticity shows them. Weak activations often mean something looser than strong ones;
- **output effects**: the tokens its direction raises and lowers through the final norm and the output layer (the logit lens). This ignores the layers in between, so read it as a hint;
- **nearest features** by the cosine similarity of their directions, and a **map** of all features (two principal components; Scaling Monosemanticity used UMAP);
- the active features per token (`--keep-codes`), so other views can show them.

**Labels** are optional. Give `--labeler URL@model`, any OpenAI-compatible
endpoint (LM Studio, llama-server or the decoding lab), and it labels the
`--label` most informative features from their top examples. Each label is
then scored, with `--score` choosing how:

- **specificity** (default): Scaling Monosemanticity's 0–3 rubric. The labeller rates how well the label fits excerpts across activation levels, plus a few excerpts where the feature is off as a control, and the score is reported on a 0–1 scale.
- **detection**: can the labeller pick the active excerpts out of a mix by the label alone? This is a check from Paulo et al. 2024 and Bills et al. 2023.

A label you type on a feature card replaces the model's (`labels.json`).
Rerunning `interpret` keeps it.

### 4. Map features onto neurons

`features.py neurons` maps each feature onto neurons in up to three ways:

- **writes**: MLP neurons in this layer and earlier ones whose output weights point along the feature (`W_down` columns · direction);
- **reads**: neurons in later layers whose input weights read it;
- **fires with**: Pearson correlation between the feature and each neuron over the recorded tokens, when `neurons` was recorded at the same layer.

For a `neurons`-site dictionary, the decoder and encoder rows already are
the map. With `--h-neurons runs/.../h_neurons.json`, each feature also
shows how much of its top neurons' weight is on H-Neurons. That shows
which concepts the hallucination neurons take part in.

Scaling Monosemanticity found that for 82% of features, no neuron
correlated above 0.3. On Qwen2.5-0.5B, layer 12, with the dictionary above,
86% fell at or below 0.3. Features really are spread across neurons, which
is why the dictionary is needed.

## Read a reply

**Features → Read a reply** takes any text, or a prompt plus the model's
reply. For each token it shows:

- the top features of each chosen dictionary, relative to each feature's maximum;
- emotion activations, as z-scores against the stories;
- with *logit lens* on, what every layer would have said next, how spread out the last layer is, and the layer where the answer settled (as in [CONFIDENCE.md](CONFIDENCE.md)).

Colour the tokens by any feature or emotion to see where it rises and
falls. Click a feature to open its card.

Two things are left out of this view:

- **The first position**, whose activations are outliers.
- **Dense features**, active on over a fifth of all tokens. They rarely mean one thing. The request field `dense` brings them back.

## Watch while it generates

The decoding lab can watch its own replies. Each generated token produces
a frame with its top features and emotion activations, and optionally each
MLP neuron's CETT:

```bash
python scripts/decode_lab.py --model M --port 8091 \
    --watch-sae runs/features/m-r12 --watch-emotions runs/emotions/m --watch-cett \
    --publish http://127.0.0.1:7890 --record runs/traces
```

- **Live viewer** (`viz/stream.py`, opened from Studio's Live page). Shows emotion lanes under the token strip, the current token's features as chips, and a *colour by* menu to tint tokens by any emotion or the neuron peak.
- **3D replay** (`viz/bloom.py`), **timeline** and **Godot client**. Open the recorded session; emotion lanes replace the risk strip, and the info line lists the emotions and features at the current token.
- **The reply itself.** It carries `decode_lab.watch`: the tokens, their top emotions and their features, for scripts that want the numbers.

Steering combines with watching, so you can clamp a feature and watch the
emotions move as the reply is written. Watching reads the residual stream
after any steering, so the frames show the steered model.
[VISUALIZATION.md](VISUALIZATION.md#b-the-decoding-lab) has the frame format
and cost.

## Attribution graphs (Circuits)

An attribution graph answers "which features produced this output, and
how did they feed each other". `feature_circuits.py graph` (or **Build in
the lab**) follows the circuit tracing method of Ameisen et al. 2025 and
Lindsey et al. 2025:

1. **A local replacement model.** Each MLP with a transcoder is replaced by
   it, plus an error term: the part of the MLP's output it misses. An MLP
   without a transcoder becomes an error term as a whole. The model's
   outputs are unchanged.
2. **Frozen nonlinearities.** Attention patterns and norm denominators are
   frozen at their values on the prompt. A feature's pre-activation is then
   a linear function of earlier features, errors and token embeddings.
3. **Edges.** The edge from *s* to *t* is `a_s × ∂pre_t/∂a_s`, the source's
   activation times the gradient through the frozen model.
4. **Output nodes.** The output tokens making up 95% of the probability (up
   to 10). Each one is its logit minus the mean logit.
5. **Order.** Features are attributed most-influential first, up to
   `--max-features`.
6. **Pruning.** Influence on the output is `B = A + A² + …` over the
   row-normalised |edges|. Nodes making up 80% of it and edges making up
   98% are kept (both adjustable).
7. **Scores.** The *replacement score* is the share of the path from prompt
   to output that runs through features rather than errors. The
   *completeness score* is how much of each node's input features explain.

The paper's method differs in one way: it uses **cross-layer** transcoders,
whose features write to every later MLP. Here they are per-layer.
Per-layer graphs have bigger error nodes; on the paper's own comparison,
the replacement score fell from 0.61 to 0.37. Train transcoders for every
layer you want in the graph. Layers without one simply count as error.

Click a feature node to see its strongest edges, open its card, or
**intervene**. Turning it off, doubling it or flipping its sign reruns the
prompt with the change, and shows:

- the next-token predictions before and after;
- how the features it feeds changed.

Interventions run the real model downstream: no attention is frozen. They
test whether the graph's story holds up, as the paper does with its
perturbation experiments.

**What drives a completion** is the lighter measure from Scaling
Monosemanticity, for any residual-stream dictionary. A feature's
attribution is its activation times the gradient of (token logit minus a
baseline logit) along its direction. The top features are then clamped to
zero one at a time to measure the real effect, and the page reports how
well the two agree. That paper found a correlation of about 0.8.

## Emotion vectors

`emotions.py` follows Sofroniew et al. 2026, *Emotion Concepts and their
Function in a Large Language Model*:

1. **Stories.** A model you choose writes short stories, one paragraph
   each. In each, a character feels the emotion, shown only through actions,
   body, speech, thoughts and situation, never named. The writer also
   produces emotionally neutral dialogues. The default list is the paper's
   171 emotions (`--preset core` is a 24-emotion subset). Stories that name
   their emotion anyway are dropped. The paper used 100 premises with 12
   stories each; 30 premises are built in, or use your own with
   `--topics-file`.
2. **Vectors.** The model under study reads the stories. For each layer:
   - average the residual stream over each story's tokens from the 50th on (`--start`);
   - an emotion's vector is the mean over its stories minus the mean over all emotions;
   - then project out the neutral dialogues' top principal components, enough to explain 50% of their variance (`--denoise`).

   Held-out stories check each layer: does the right emotion's vector
   score highest? With a small neutral set, the components are noisy and
   can remove emotion signal too, so build reports accuracy both with and
   without denoising. The default layer is two thirds of the way through
   the model, as in the paper.
3. **Look.**
   - **Trace** any text token by token, as z-scores against the stories.
   - **Scan** a corpus for the passages each vector fires on most.
   - **Sweep** one quantity in a prompt (the paper's example: a Tylenol dose rising to dangerous levels) and watch the emotions just before the reply.
   - **The map** shows the vectors' two main directions. The paper finds valence and arousal there; give `--judge URL@model` to rate each emotion and test that.
   - **The similarity matrix** and **output effects** round out the view.
4. **Steer.** Add `s × |h| × v` after the default layer, where *s* is a
   fraction of the residual stream's norm at each token. The paper's
   behavioural sweeps used −0.1 to +0.1.

**Quality depends on the writer.** In our test, Qwen2.5-0.5B wrote its own
stories: 41 stories for 6 emotions, several of them summaries. Held-out
accuracy reached 43% at layers 12–18 against 17% chance without
denoising, and 29% with it. Use a capable writer model and more stories
per emotion.

## Steering

| What | How | Units |
|---|---|---|
| Clamp a feature | the activation is split into reconstruction and error; the reconstruction is rebuilt with the feature set to the value, at every token; the error is kept (Scaling Monosemanticity) | multiples of the feature's largest observed activation |
| Add a feature | add value × max × its direction | the same |
| Emotion mix | add Σ s·\|h\|·v after the chosen layers | fractions of the residual norm |

**Start small and raise.** The paper clamped Claude 3 Sonnet's features at
up to 10× their maximum; ±100× produced gibberish. On Qwen2.5-0.5B with
the dictionary above, 8× already produced gibberish. The size that works
depends on the model and the dictionary.

Where steering can run:

- the **Steer** tab, which combines features and emotions, compares with the plain reply, and saves setting profiles;
- each feature card, and each emotion's detail;
- the **decoding lab**, as variants (`--variant 'calm={"emotion": {"file": "runs/emotions/m", "mix": {"calm": 0.05}}}'`), so TestQA and the retrain gate can measure a steered model against plain. The Steer tab writes this line for you;
- per request, under `"omnicortex": {...}` in a chat completion to the lab;
- **llama.cpp**: `emotions.py cvec` writes an emotion mix as a control vector for `llama-server --control-vector`. It is fixed, scaled by the layer's typical norm, since llama.cpp cannot scale by each token.

## Files

```
runs/features/acts-NAME/   meta.json, <site>.<layer>/shard_*.npy, tokens.npy, doc.npy, pos.npy, win_tokens.npy, win_ptr.npy
runs/features/NAME/        sae.pt, sae.json (config, training curve, held-out checks)
                           features.json, examples/, vocab.json, codes_*.npy   (interpret)
                           neurons.json (neurons), labels.json (your labels)
runs/graphs/NAME.json      attribution graphs
runs/emotions/stories/     stories.jsonl, neutral.jsonl
runs/emotions/NAME/        emotions.pt (vectors per layer), emotions.json (accuracy, map, similarity, logit effects)
```

Studio's Features page lists everything under the workspace's `runs/`.
Settings can add other folders (`_features_dirs`).

## Limits

- **Dictionaries describe the data you record.** Scaling Monosemanticity
  notes this for its own work: features absent from the corpus will not
  appear. Record text like the conversations you care about.
- **Attribution graphs** explain the replacement model with frozen
  attention. They do not explain why attention moved information, only
  what moved. Interventions on the real model are the test.
- **Emotion vectors** are linear directions for emotion *concepts*. The
  paper finds they are "local", tracking the emotion relevant to the
  current text rather than a lasting state. It does not claim the model
  feels anything, and neither does this tool.

## Command line

```bash
python scripts/features.py collect --model M --corpus NeelNanda/pile-10k --site resid mlp_in mlp_out --layers 6 12 --out runs/features/acts-m
python scripts/features.py train --acts runs/features/acts-m --site resid --layer 12 --out runs/features/m-r12
python scripts/features.py train --acts runs/features/acts-m --site mlp_in --target-site mlp_out --layer 12 --out runs/features/m-tc12
python scripts/features.py interpret --sae runs/features/m-r12 --labeler http://127.0.0.1:1234/v1@labeller --label 200
python scripts/features.py neurons --sae runs/features/m-r12 --h-neurons runs/classifier/h_neurons.json
python scripts/feature_circuits.py graph --transcoders 'runs/features/m-tc*' --prompt "..." --out runs/graphs/x.json
python scripts/feature_circuits.py attribute --sae runs/features/m-r12 --prompt "..." --token " Austin" --baseline " Dallas"
python scripts/feature_circuits.py intervene --transcoders 'runs/features/m-tc*' --prompt "..." --set 12:877=0
python scripts/emotions.py stories --writer http://127.0.0.1:1234/v1@writer --out runs/emotions/stories
python scripts/emotions.py build --model M --stories runs/emotions/stories --out runs/emotions/m
python scripts/emotions.py trace --emotions runs/emotions/m --text "..."
python scripts/emotions.py sweep --emotions runs/emotions/m --template "I took {x} mg of Tylenol. OK?" --values 500 2000 8000
python scripts/emotions.py steer --emotions runs/emotions/m --prompt "Hi." --mix calm=0.05
```
