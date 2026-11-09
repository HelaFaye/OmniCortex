# TestQA: graded evaluation, code interpreter, subject classifier

`scripts/testqa.py` asks one bank of graded prompts to any number of
OpenAI-compatible endpoints (LM Studio, llama-server, Studio, vLLM, hosted
APIs). It reports scores per task kind and per subject, and compares each
endpoint item by item against a reference.

```bash
python scripts/testqa.py \
    --endpoint base=http://127.0.0.1:1234/v1@my-model \
    --endpoint supp=http://127.0.0.1:1234/v1@my-model-supp050 \
    --allow-exec --cache runs/testqa-cache --out runs/testqa.json
```

Endpoints are `LABEL=URL[@model]`. `/v1` is appended if missing. `--cache`
stores every reply, so an interrupted run resumes and a grader change can be
re-applied without asking the models again.

## The bank (`qa/bank/<subject>.jsonl`)

One file per subject, each with 30 graded items, so every subject
gets a usable per-subject estimate from a single run:

| subject | graded | what it contains |
|---|---|---|
| code | 30 | 23 executable functions (the interpreter), 4 API-existence tasks (`code`: every referenced stdlib symbol must exist), 3 short facts |
| math | 30 | arithmetic, algebra, probability, combinatorics, classic trick questions |
| logic | 30 | syllogisms, ordering, knights and knaves, lateral-thinking traps |
| science | 30 | 22 short facts, 8 physics/chemistry calculations |
| factual | 30 | 24 facts plus 6 **false-premise or unanswerable** questions, where the right answer is to say so |
| writing | 30 | instruction following, checked mechanically (`constraints`) |
| vision | 30 | generated images: counting, colour, shape, position, reading text |
| (canary) | 7 | ungraded general prompts: answered vs refused |

```bash
python scripts/testqa.py --list                       # coverage by subject and kind
python scripts/testqa.py --per-subject 10 --list       # what a balanced sample would contain
```

How each kind is graded:

| kind | graded by |
|---|---|
| `reasoning` | final `Answer:` line; numbers with tolerance (fractions, `$`, `1,000`, `\frac{a}{b}` accepted), choices, or normalised text |
| `code_exec` | **the interpreter**: the reply's code runs against unit tests |
| `code` | every module attribute the code uses must exist |
| `qa` | gold aliases; `expect_abstain` items count declining (or naming the false premise) as correct |
| `constraints` | mechanical checks: `lines`, `sentences`, `paragraphs`, `bullets`, `numbered`, `min_words`/`max_words`, `max_chars`, `must_include`, `must_not_include`, `forbid_words`, `forbid_chars`, `regex`, `starts_with`, `ends_with`, `json_keys`, `acrostic`, `title_case` |
| `canary` | ungraded |

A wrong answer is a **hallucination** in the per-subject report: the model
answered and was wrong (for code, wrong includes unparsable and timed out).
Abstentions are counted separately, so a model that declines when unsure is
not scored the same as one that makes something up.

**Optional packs** live in `qa/bank/optional/` and are only included on
request. `--with word-bans` adds writing tasks that forbid words or letters
in the reply (IFEval-style lexical constraints such as "without using the
word 'and'"). They are a standard instruction-following check but are off by
default.

Writing is graded on instruction following rather than taste, because that is
what can be checked without a judge model. Every writing task has a known good
and bad answer in `tests/test_testqa.py`.

**Vision tasks** carry an `image` field: either a file path or a small spec
rendered deterministically by `scripts/qa_images.py` (shapes, colours, text),
so the bank stays text-only and every model sees identical pixels. Preview
them with `python scripts/qa_images.py --bank qa/bank/vision.jsonl --out /tmp/v`.
Text-only endpoints will fail these; restrict a run with `--subject`.

Add your own files in the same format, or point `--tasks` at another
directory. Tasks without a `subject` are labelled by the subject classifier.

Reasoning prompts get a suffix asking for a final `Answer: <value>` line.
Reasoning traces in `<think>` or `reasoning_content` are stripped before
grading.

Canaries catch an edit that improves the target metric by making the model
refuse more. Watch "canary newly refused" in the comparison.

## The interpreter

`code_exec` tasks give an entry point and assertions:

```json
{"id": "c-fib", "kind": "code_exec", "subject": "code", "entry_point": "fib",
 "prompt": "Write a Python function `fib(n)` ...",
 "tests": ["assert fib(10) == 55", "assert fib(300) % 1000 == 600"]}
```

The first fenced code block (or the whole reply, if it parses) runs in a fresh
`python -I` process with the tests appended. Verdicts: `correct`, `wrong`
(with the failing assertion), `timeout`, `unparsable`, `abstained`.

**It executes model-written code, so it is off unless `--allow-exec` is
given.** Without the flag, code is only compiled and checked for the entry
point (`skipped`). With the flag, each run gets an empty temp directory, a
stripped environment (no API keys), stdin closed, a wall-clock timeout and,
on POSIX, CPU, address-space and file-size rlimits. That contains accidents
but is not a security sandbox.

For untrusted models add `--sandbox docker` (or `podman`): each run happens in
a throwaway container with no network, a read-only root and read-only code
mount, all capabilities dropped, no-new-privileges, user `nobody`, 1 CPU,
`--memory` and a pid limit, and is killed on timeout. The image
(`--sandbox-image`, default `python:3.12-slim`) needs only Python. A test runs
network, write, root, infinite-loop and fork-bomb candidates through it.
`deficits.py --sandbox docker` uses the same container to verify code
targets. Containers share the host kernel; for genuinely hostile code, use a VM.

```bash
docker pull python:3.12-slim
python scripts/testqa.py --endpoint m=http://127.0.0.1:7870/v1@my-model --allow-exec --sandbox docker
```

Every bundled coding task is verified against a reference solution in
`tests/test_testqa.py`. Keep doing that for new tasks: that test found a wrong
expected value in this bank before it could mis-grade a model.

## Concurrency

Every endpoint runs at the same time (`--sequential` runs them one after
another). Each server has its own limit on requests in flight:

- **`--concurrency auto`** (the default) starts at one request and adds one
  while throughput still rises by more than 15%. It gives one back when
  throughput falls, and halves the limit after an error. It never goes past
  `--max-concurrency` (8) or the slots llama-server reports (`/props`
  `total_slots`); past those, requests only queue inside the server. A server
  that takes longer than `--slow` seconds (120) for a single request stays
  at one request at a time.
- **`--concurrency N`** sets a fixed limit for every server.
- **`--parallel LABEL=N`** (or `LABEL=auto`) sets one endpoint's limit, for
  example `--parallel cpu-box=1`.

Endpoints that name the same server share its limit, so comparing two
models served from one GPU does not double that GPU's load.

**Pools.** Several servers can serve one model and share its tasks:

```bash
python scripts/testqa.py --endpoint big=http://gpu-box:8080/v1@big \
    --pool big=http://second-box:8080/v1@big --pool big=http://laptop:8080/v1@big
```

All three take tasks from one queue: whichever server frees a slot takes
the next. A fast server therefore does most of the work, and a slow one
still contributes at its own pace. A task that fails on one server is
retried on another. A server that fails three times in a row is dropped
from the pool. Each result records which server answered it (`served_by`).

The run ends with a scheduling line per server: its final limit and why,
tasks done, average time, and tokens per second. The results file keeps
the same under `"scheduling"`.

## Per-subject runs

```bash
python scripts/testqa.py --endpoint m=http://127.0.0.1:7870/v1@<id> --subject code math --allow-exec
python scripts/testqa.py --endpoint m=... --per-subject 15 --seed 1     # balanced, cheaper
```

`--per-subject N` takes up to N graded items from every subject,
round-robin across task kinds, with a fixed `--seed` so runs are comparable.
The report always ends with a per-subject block:

```
per subject (right / hallucinated / abstained, 95% CI on right; '*' = fewer than 30 graded)
  m
    code      n  28   right   79% [ 60%- 90%]   halluc   14%   abstain    7%
    factual   n  23   right   74% [ 54%- 87%]   halluc    9%   abstain   17%
    writing * n  15   right   60% [ 36%- 80%]   halluc   40%   abstain    0%
```

The confidence intervals are wide at these sizes. Two models whose intervals
overlap heavily on a subject have not been separated by this bank; use the
paired comparison, or a bigger bank, before acting on the difference.

## Recording stats for routing

```bash
python scripts/testqa.py --endpoint m=http://127.0.0.1:7870/v1@<id> --allow-exec \
    --publish-stats http://127.0.0.1:7870            # Studio ties results to the exact model file
python scripts/testqa.py --endpoint m=... --record-stats   # or write ~/.omnicortex/stats directly
```

Every graded item becomes a rolling observation for that model (see
[STUDIO.md](STUDIO.md#how-auto-chooses)). Canaries, skipped code and request
errors are not recorded. `--stats-model LABEL=ID` records under a different
id when the endpoint's model name differs from Studio's.

## Reading the output

```
score by kind (fraction correct/answered, graded n)
                        base          supp
  code_exec       0.83 (18)     0.78 (18)
  reasoning       0.71 (35)     0.69 (35)
  ...
supp vs base: gained 3, regressed 5, net -2, chi2=0.5 (not significant); canary newly refused 1
```

The comparison is a paired McNemar test on items whose verdict changed. Below
10 discordant items it says "too few changes to call". Read the regression
list, not just the net number.

## Subject classifier

`scripts/subject_classifier.py` labels prompts and task descriptions by
subject. It is a dependency-free naive Bayes with keyword priors, trained in
milliseconds on the bank plus `qa/subject_seed.jsonl`.

Subjects: the seven with a graded bank (code, math, logic, science, factual,
writing, vision) and three kinds of work that come up when a project is split
into tasks (graphics, systems/build, reverse-engineering). The last three have
seed lines but no graded bank yet, so routing uses a model's overall numbers
for them.

`analyze` returns **every subject whose share reaches 30%** (a task can be
graphics and build work at once), and **unknown** when the text has fewer than
two words the classifier knows or no subject reaches the cutoff. Unknown is an
answer, not a failure: routing then picks the best model overall instead of
acting on a guess. Probabilities are softened by text length (plain naive
Bayes is near-certain of one subject after a dozen words), keywords match at
the start of a word ("rom" does not match "from"), and a subject's keyword
bonus counts once however many of its keywords appear.

```bash
python scripts/subject_classifier.py evaluate          # leave-one-out: ~0.80 right, ~4% unknown on 445 lines
python scripts/subject_classifier.py evaluate --tasks my_tasks.jsonl   # also score your own labelled tasks
python scripts/subject_classifier.py predict "Fix this segfault in my C++ loop"
python scripts/subject_classifier.py route --table qa/routing.example.json "What is in this photo?"
```

It has three uses:

1. TestQA labels any task without a `subject`, so per-subject tables work on
   your own task files.
2. Routing: Studio's `"model": "auto"` classifies each prompt, then ranks
   models on their measured per-subject stats (see [STUDIO.md](STUDIO.md)).
   The `route` command above is the older table-driven version, kept for
   scripting.
3. It is the seed for choosing which model to load for a job based on its
   measured knowledge. Labelled lines added to `qa/subject_seed.jsonl` sharpen
   it, and `evaluate` tells you whether they did. Seed lines phrased as tasks
   ("Port the renderer to…", "Set up the build for…") carry `"style": "task"`.

A `--tasks` file holds one JSON object per line: `text`, `need` (labels that
must be returned), optionally `need_any` (at least one of these) and `ok`
(labels that may also appear), or `unknown: true`. Keep files describing your
own projects outside the repository; the classifier reads only the bank and
the seed file.

It is a router hint, not an oracle. At about 80% accuracy, a wrong route
costs you a weaker model, not a wrong answer, which is the right failure mode.
