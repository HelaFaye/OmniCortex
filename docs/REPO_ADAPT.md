# Adapting a model to one repository

`scripts/repo_adapt.py` (Jobs → Retrain → **Adapt to a repository**) makes training data that teaches a model
one repository: questions and answers written by a teacher from the repository's own source, kept only when the
source supports them. It also keeps a record of where every pair came from, and later says when the repository
has moved on far enough that the adapter is out of date.

Try retrieval first: Docs → **+ Add a folder** ([QUICKSTART.md](QUICKSTART.md), path 1). It needs no training and
never goes stale. An adapter helps when the model must use the repository's names and conventions without
looking them up: completions, short answers, and agents with little context to spare.

## The steps

```bash
export OC_TEACHER_KEY=...        # the teacher's key: from the environment only
python scripts/repo_adapt.py run --repo ~/src/app --teacher https://api.example.com/v1@model \
    --replay general.jsonl --out runs/adapt-app --limit 20       # a trial; drop --limit for the whole repository
python scripts/finetune.py --model <base> --data runs/adapt-app --method lora --lora-r 16
python scripts/merge_export.py --base <base> --adapter <adapter> --out <out> --lora-gguf
python scripts/repo_adapt.py check --manifest runs/adapt-app/manifest.json --repo ~/src/app \
    --adapter <out>/<adapter>-lora.gguf --max-adapter-mb 200
```

1. **Scan.** It reads:
   - **the files:** the same as the Docs folder import, so version control, dependencies, build output and files
     over 1 MB are skipped;
   - **the public symbols with their signatures:**
     - Python: functions, classes and methods, read with Python's own parser;
     - JavaScript and TypeScript: exports;
     - Rust: `pub` items;
     - Go: capitalised names;
     - C and C++ headers: prototypes;
     - Java, Kotlin and C#: `public` members;
   - **which files are tests, build files and documents**;
   - **the git commit**.
2. **Questions.** For each public symbol, and each piece of the documents, the teacher sees the source with line
   numbers. It writes `--per-unit` questions (default 3), each answer citing the lines it rests on.
3. **Judge.** A pair is kept only if all three checks pass:
   - its cited lines are inside what the teacher was shown;
   - every name it puts in backticks exists somewhere in the repository (made-up functions are the commonest
     teacher error, and this catches them without a model);
   - the judge model answers SUPPORTED when shown the source, the question and the answer.

   The judge is the teacher unless `--judge` names another, or `--no-judge-model` leaves only the cheap checks.
4. **Build.** It writes three files:
   - **`sft.jsonl`** for `finetune.py`: 85% repository pairs and 15% general replay (`--replay`, chat-format
     JSONL), so the model keeps what it knew. Too little replay is reported as a warning.
   - **`holdout.jsonl`**: about 10% of the symbols, every question about each one, never trained on. Measure the
     adapted model on these.
   - **`manifest.json`**: the provenance record, described below.

Every step resumes: an interrupted run asks nothing twice. If the repository's interfaces changed since the last
run, the questions start again.

## The manifest

`manifest.json` holds:
- **the repository:** its name (not its path), commit, and whether the working tree had uncommitted changes;
- **the core interfaces and their fingerprint;**
- **the teacher and judge:** model ids and hosts only, never keys or URLs with queries;
- **counts and the training mix;**
- **for every pair:** the file, cited lines, the file's hash, the verdict and why, and whether it went to
  training or the holdout.

The API key is read from `--key-env` (default `OC_TEACHER_KEY`) and goes only into the request to the teacher.
The tests check that it appears in no output file.

## When an adapter goes out of date

`check` scans the repository again and compares its interfaces with the manifest's:
- **Fresh:** a commit that only changes function bodies, tests or documents leaves the interfaces alone, so the
  adapter stays current.
- **Stale** (exit status 1): an interface was changed, added or removed. It lists each one, such as
  `api.py::Studio.chat` gaining a parameter.

`--core` (repeatable globs, such as `src/api/**`) limits which paths count, so internal helpers can change freely.
Pass the same `--core` to `run`; `check` reads it from the manifest.

`--adapter` and `--max-adapter-mb` also refuse an adapter larger than its budget. A rank-16 LoRA on a 27B model is
a few hundred MB; one much larger than expected usually means the wrong layers or rank.

## Choosing the teacher and judge

Use a strong teacher and judge. The judge's checks are only as good as its reading of code.

In a trial on OmniCortex's own Kivy client (`clients/kivy`), Qwen2.5-1.5B served as both teacher and judge, over 8
symbols with 3 questions each:
- it kept 20 pairs and rejected 4;
- **the rejected four:** vague claims the source doesn't show ("how is the `Pairing` class typically used?");
- **most kept answers were right:** parameters, return values, what `parse_pair_link` returns;
- **at least one wasn't:** "returns a list of model IDs *from the repository*". The models come from Studio, so a
  stronger judge should have rejected it.

So:
- use a small model only for a trial of the pipeline;
- for training data, use a teacher the size of the target setup's (DeepSeek V4.1 Flash in
  [ROADMAP.md](ROADMAP.md));
- if the teacher is weak, use a separate judge (`--judge`).

## What's next

[ROADMAP.md](ROADMAP.md), phase 1:
- a Studio indicator that marks a loaded adapter as stale;
- checking the base model's truth probe on the adapted model;
- running the holdout through TestQA's grader.
