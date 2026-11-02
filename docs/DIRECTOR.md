# Director: projects, plans and worker models

Studio's **Projects** page (`/projects`) takes a project description, splits it
into tasks by skill, keeps a plan that a person approves, hands each task to a
worker model it starts for the purpose, and routes every result back to a
person. `scripts/director.py` is the logic; Studio runs it and starts the
models.

```
describe ──► draft plan ──► (re-plan with a model) ──► you approve ──► start
                                                                       │
        ┌──────────────── review: accept / send back with feedback ◄──┤ worker
        │                 blocked: answer its questions          ◄────┤ models
        ▼                 proposals: accept / reject             ◄────┘
      done
```

## 1. Describe the project

Paste notes, a skill list, an issue or a README. List items, table rows and
`Name: description` lines become candidate tasks; when a document has lists or
tables, its prose paragraphs are treated as context, not tasks. Code blocks
are skipped.

Each task is labelled by the subject classifier (see
[TESTQA.md](TESTQA.md#subject-classifier)): code, math, logic, science,
factual, writing, vision, graphics, systems, reverse-engineering. A task can
carry several labels, and one the classifier cannot place says **unknown**
rather than guessing. The chips at the top of a project are the split by skill.

Optionally give a local checkout to survey. File types, build files and
imports say which skills the code needs (shaders mean graphics, `CMakeLists.txt`
means build work, `import PIL` means image analysis, files under `asm/` mean
reverse engineering), and the page points out skills the code needs that no
task covers. Only file names and the first 4 KB of source files are read.

## 2. Draft, then approve

The draft is yours to edit: change a task's detail, acceptance criteria
("done means"), skills, which tasks it waits for, or pin a model. **Re-plan
with model** asks a model to merge duplicates, split mixed tasks, drop lines
that only describe history, and add dependencies and acceptance criteria; its
answer is validated (no unknown dependencies, no cycles, labels re-checked), and
a malformed answer leaves the draft as it was.

**Approve plan** assigns a model to every task and freezes the plan:

- With graded results (`testqa.py --publish-stats`), the model whose measured
  per-subject numbers best fit the task's labels wins, the same ranking as
  Studio's `auto` routing.
- Without them: the project's default model, else the largest model that fits
  the machine. The plan says which rule picked each model.
- Models whose trained context is shorter than the task needs are skipped, and
  approval says so rather than failing later.

## 3. The plan, and sticking to it

Once approved, the director runs only the approved tasks, in dependency order,
up to the project's parallel limit. It does not change the plan by itself.
When it wants a change, it **proposes** one, and the change waits for you:

- a worker reports follow-up work: "add these tasks after T4";
- a task is sent back `attempts` times: "try T2 with the next-best model".

Your own edits apply at once at any stage. Every change, whoever made it,
bumps the version and is in the history with who and why.

## 4. Workers

A worker gets the project goal, the plan's task list, its own task and
acceptance criteria, the **accepted** output of the tasks it depends on, and
any feedback from earlier attempts or answers to its questions. It ends its
reply with a small JSON report:

```json
{"status": "done" | "blocked", "summary": "…", "followups": ["…"], "questions": ["…"]}
```

A reply without the report is marked "unreported" and goes to review, never
straight to done. Workers produce text (code, patches, analyses, drafts).
Nothing here runs commands or writes to your repository; applying a result is
your step.

When the model has a hallucination classifier and Studio has `--cett`, each
result gets the activation check from chat (flagged tokens and a link to the
3D view of that reply).

## 5. Review

The policy decides what waits for you:

| review | what waits |
|---|---|
| `all` (default) | every result |
| `flagged` | results the activation check flagged, results without a report, and blocked tasks |
| `none` | only blocked tasks |

**Send back** with feedback: the next attempt sees it. **Blocked** tasks show
the worker's questions; answering them requeues the task. Errors retry with
back-off (15 s, 30 s, 60 s); errors that retrying cannot fix (the prompt does
not fit the model's context) block at once with the reason.

## Worker models and hardware

Workers are separate llama-server processes next to the chat model Studio
serves, at most `--max-workers` at a time (default 2), stopped after
`--worker-idle` seconds (default 300). Each is placed on the hardware
`scripts/accelerators.py` finds, **AMD first**:

- **ROCm** and **Vulkan** for AMD GPUs and APUs, **CUDA** for NVIDIA, **Metal**
  on Apple silicon, and the **CPU**;
- one GPU with room (AMD before others, discrete before shared memory, then
  the tightest fit); else a layer split over several GPUs of one backend; else
  a GPU whose memory is only estimated; else the CPU;
- on an APU, usable memory is the VRAM carve-out plus GTT, and it is the same
  RAM the CPU uses, so a model on one leaves less for the other.

Each device can have its own llama-server build, settings and environment,
and each project can limit itself to some devices and override settings per
device (the project's policy). Details, including AMD APUs that need a
particular ROCm release, are in [HARDWARE.md](HARDWARE.md#worker-devices).

```bash
python scripts/accelerators.py          # what Studio would use, and how
python scripts/director.py analyze notes.md --repo ~/src/project   # the split, offline
```

## Requirements of a project

A project has its own needs, separate from OmniCortex's: the toolchain,
libraries and Python packages its build files name, the hardware it wants,
which OmniCortex features it relies on, and anything to check by hand. The
project page's **Requirements** card keeps that list:

- **Inferred** when the project is created from a checkout (and on **Scan
  build files**): CMake (`cmake_minimum_required`, `find_package`,
  `pkg_check_modules`), Meson `dependency()`, Makefiles, xmake (its own
  packages listed as a note), premake, autotools, Cargo and
  `rust-toolchain`, `go.mod`, `package.json` (`engines.node`, the package
  manager from its lock file), `.nvmrc`, `pyproject.toml`, top-level
  `requirements*.txt` (`dev`/`test` files optional), `.python-version`,
  Dockerfiles, Gradle, Maven, Zig, Godot, Nix, vcpkg, Conan, `just` and .NET
  projects. Only build and manifest files in the top two folder levels are
  read (not `node_modules`, `build`, vendored or third-party folders), never
  sources.
- **Edited** by you: add a tool, library (pkg-config name), Python package,
  hardware (`ram_gib`, `vram_gib`, `disk_gib`, `gpu` vendor), a OmniCortex
  feature, or a note; remove what doesn't apply. Your items survive
  re-scans. Every change is in the plan's history.
- **Checked** on this machine (**Check now**): versions against the
  needed ones, libraries through pkg-config, Python packages in the
  project's `.venv`/`venv` when it has one (else Studio's Python, and the
  page says so), with the install command for this OS for each gap.

**Start** checks first and refuses while a required item is missing,
listing what; you can start anyway. Workers are told what the machine has
and what it lacks, so their build steps and commands fit it.

```bash
python scripts/ns_requirements.py project ~/src/game     # the same, for any checkout
```

## API

Owner only (a paired device gets 403), like Jobs.

| | |
|---|---|
| `GET /api/projects` | projects, models, subjects |
| `POST /api/projects` | `{title, goal, text, repo?}` → draft |
| `GET /api/projects/<id>` | the plan, skill split, coverage gaps, ready tasks |
| `POST /api/projects/<id>/edit` | `{changes: [{op: add\|update\|drop\|reopen\|assign, …}], reason}` |
| `POST /api/projects/<id>/approve`, `/start`, `/pause`, `/delete` | lifecycle; `/start` answers 409 with `missing` unless `{force: true}` |
| `GET /api/projects/<id>/requirements` | check the project's requirements on this machine |
| `POST /api/projects/<id>/requirements` | `{items: [{kind, name, need, why, optional}], reason}`: replace the list |
| `POST /api/projects/<id>/requirements/scan` | `{root?}`: infer again from the checkout's build files |
| `POST /api/projects/<id>/refine` | `{model}`: re-plan a draft with a model |
| `POST /api/projects/<id>/decide` | `{proposal, accept, note?}` |
| `POST /api/projects/<id>/review` | `{task, accept, feedback}` |
| `POST /api/projects/<id>/answer` | `{task, answers: [...]}` |
| `POST /api/projects/<id>/policy` | `{policy: {review, max_parallel, max_attempts, default_model, check, devices, device_settings}}` |
| `GET /api/workers`, `POST /api/workers/stop` | running workers and detected devices |

Plans live in `--projects-dir` (default `~/.neuronscope/projects/<id>/plan.json`).

## Verification graphs

A project's policy can name a node graph that screens every result before
review: secrets, length, a judge, a hallucination check, anything a graph can
do. The graph sees `task` and `result`; an `accepted` output passes the result
and a `feedback` output fails it. With *send failures back automatically*, a
failed result returns to the worker with the feedback. The verdict and a link
to the graph show on each attempt. See [GRAPHS.md](GRAPHS.md#projects-delegation).
Workers are also told what their task may do ([PERMISSIONS.md](PERMISSIONS.md)).

