# Neuron review: how each neuron behaves across everything a model does

The 3D trace view shows one reply, token by token. The **Review** page
(`/review` in Studio) shows the opposite slice: every reply a model has
written that OmniCortex has looked at (tests, benchmarks, chat), and for
each MLP neuron, how its activity relates to whether the model was right.
Filter by where the data came from and what it was about, and play it over
time.

## Where the data comes from

Each **observation** is one reply's activation profile: CETT for every
neuron, averaged over the reply's tokens and binned to 512 columns per layer
(the same binning as the 3D traces). It is stored with:

| field | meaning |
|---|---|
| source | the run it came from: `testqa`, `livebench`, `swe-bench`…, `chat-check`, `project`, `live` |
| kind | **test** (TestQA), **benchmark** (anything else graded), **observed** (replies from use) |
| subjects | the subject classifier's labels for the prompt (code, math, science, …) or the benchmark's own |
| verdict | right / wrong / abstained when graded or when you mark a chat reply; otherwise not graded |
| risk | the hallucination classifier's probability, when the model has one |
| time | when the reply was written |

Three ways observations arrive:

- **Chat checks.** Every **check** under a chat reply records one. Under the
  check result, *was it right? right / wrong* records your verdict, so chat
  replies count in the right-vs-wrong statistics too. Checked project work
  is recorded as `project`.
- **Live scoring.** If a model has a classifier and Studio scores replies in
  the background (`score_every`), each scored reply is recorded as `live`.
- **Tests and benchmarks.** Jobs → **Review**: *ingest a TestQA run* takes a
  `testqa.py --out` file; *ingest graded replies* takes JSONL with `prompt`,
  `response`, `verdict` (right/wrong/abstained, or pass/fail) and optionally
  `subject`, from any other benchmark. Each reply is run through the model
  once more (a prefill, no generation) to record its activations, so pass the
  same GGUF that wrote the replies. Vision items are skipped (their image is
  part of the prompt).

Observations live in `~/.omnicortex/review/<model>/` (`index.jsonl` plus one
small `.npy` per reply, about 30 KB for a 28-layer model). Nothing leaves the
machine.

## The statistics

Pick one; the heatmap shows it for every layer (rows) and neuron bin
(columns), and the table lists the strongest in each direction.

- **Hallucination association.** Cohen's *d* between wrong and right
  answers: how many standard deviations more the neuron fires when the model
  is wrong. Red-orange: fires more on hallucinations. Blue: fires more on
  right answers. Needs graded replies (at least two of each).
- **Risk correlation.** For replies nobody graded: the correlation between
  the neuron and the classifier's risk. Useful for chat-heavy models, but it
  can only find what the classifier already looks at.
- **Firing rate.** How often the neuron is among the top 3% of activity in
  the selected replies. No verdicts needed: which neurons a subject or a
  benchmark recruits.

**The noise floor.** With thousands of neurons, some differ between right and
wrong answers by chance alone. A cell is coloured only when its effect clears
a floor set by the number of replies and the number of neurons (a
Bonferroni-corrected critical value, widened for small samples), so that a
review with no real effect lights up nothing in about 19 of 20 cases. The
legend shows the floor. Small selections have high floors: with 10 right and 10
wrong answers a neuron needs |d| above about 2.7; with 100 of each, about 0.7
(for a 28-layer model).
Grey means "can't tell from this much data", not "no effect".

## Filters and time

- **Sources**, grouped as tests, benchmarks and observed: compare what a
  benchmark shows with what happens in use.
- **Skills and subjects**: e.g. only math, to see whether math errors have
  their own neurons.
- **Verdicts**: e.g. only ungraded replies, with the risk statistic.
- **From / to** dates, and **over time by** day, week, month or every N replies.

The error-rate chart shows each period's share of wrong answers among graded
replies (bars: how many replies). **Open 3D over time** plays the periods as
frames in the 3D view: each frame colours the neurons by that period's
statistic, and the strip at the bottom is the error rate. Use it after a
retrain, an edit or a quantization change to see whether the same neurons are
still the ones that go with wrong answers.

## Without the UI

    python scripts/neuron_review.py ingest-testqa --results runs/testqa.json \
        --gguf ~/models/acme/Model-GGUF/model-Q4_K_M.gguf --binary ~/llama.cpp/build/bin/llama-cett-dump
    python scripts/neuron_review.py ingest-items --items livebench-graded.jsonl --source livebench --gguf … --binary …
    python scripts/neuron_review.py models
    python scripts/neuron_review.py summary --model acme/model-q4_k_m --subject math --by month

The model id is Studio's (`<folder>/<file stem>`, lower case), so ingested
runs and chat checks land together. API: `GET /api/review`,
`POST /api/review/summary`, `/api/review/view`, `/api/review/label`. MCP:
`review_sources`, `review_summary`, `review_label`.

## Reading it carefully

- An association is not a cause. A neuron that fires on wrong answers may
  track a topic that is simply harder. Filter to one subject before drawing
  conclusions, and confirm with the suppression tools ([PIPELINE.md](PIPELINE.md)).
- Columns are bins of neurons (max-pooled), as in the traces: a bright bin
  says one of its neurons stands out. The H-Neuron classifier works on single
  neurons.
- Observations from different model files are never mixed: a quantization
  or an edit is a different model id.

## Weighting measurements

A model can be measured several ways:
- TestQA, LiveBench and SWE-bench (graded answers, by source);
- published reference scores;
- how often H-Neuron activation flags its replies;
- the replies you marked right or wrong here.

They do not always agree. **Tasks → Distribution → How measurements are
weighted** shows how much each one counts when models are ranked together
(`scripts/measure_weights.py`):

- **Against your verdicts.** Once at least five models have replies you
  labelled, each measurement is weighted by its rank correlation with those
  labels. It counts for as much as it agrees with what you found in use.
- **By consensus.** Until then, each measurement is weighted by its
  correlation with the mean of all the others. A measurement that disagrees
  with everything else counts less.

The rules for turning correlations into weights:
- A correlation from few models is shrunk toward an equal share
  (n ÷ (n + 5)).
- A negative correlation counts as zero.
- The combined score is the weighted mean of z-scores, over the
  measurements each model has.

Every fit is added to a history, shown beside each weight, so you can watch
the weights settle as results come in. A scoreboard computes the same
weights per subject, and they show which subjects separate models.

## Subject colours in the 3D views

Once the review has observations across subjects (TestQA runs and checked
chat replies are labelled by subject), it can say which neurons go with
which subject. A neuron goes with a subject when its activation on that
subject's observations is higher than on the others':
- by more than the same Bonferroni-corrected floor the association uses;
- for the subject it separates most.

A 3D view of a checked reply then colours each active neuron by its
subject, in the theme's subject palette. Neurons flagged for hallucination
keep their ring and colour on top. **Colour by subject** in the legend
turns it off.

Two conditions apply:
- The map needs at least six observations of the model, and at least three
  per subject.
- It only applies when the trace has the review's layer count.

`neuron_review.subject_map()` returns it for other uses.
