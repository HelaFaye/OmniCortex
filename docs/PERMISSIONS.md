# Permissions

What a model may do on this machine, per model and per config
(`scripts/permissions.py`). Edit it in **Settings → Permissions**; the same
policy is used everywhere a model acts:

| where | what is checked |
|---|---|
| Studio chat, when the model calls a tool | the tool, plus every path, command line, program and URL in its arguments. Tool approval ([CONNECT.md](CONNECT.md)) answers first; the stricter of the two wins. |
| TestQA with `--allow-exec` | starting the program that runs the model's code (the sandbox engine, or Python), for job `testqa` |
| Project tasks (the director) | each worker is told what its task may do, from job `project_task` and the task's own `permissions` field |
| Graphs | every node that reads or writes files, fetches URLs, starts the code sandbox or calls MCP tools, for job `graph`, the run's profile and the node's own ([GRAPHS.md](GRAPHS.md#permissions)) |
| Anything else | `check_permission` over MCP, `POST /api/settings/permissions/check`, or `python scripts/permissions.py check` |

## Actions

| action | target |
|---|---|
| `read` | a path |
| `write` | a path (also changes and deletes) |
| `run` | a command line. It is checked whole and split at `;` `&&` `\|\|` `\|`, newlines and command substitutions, so an allowed command cannot carry another. |
| `exec` | a program, by name or path |
| `net` | a host or URL |
| `system` | a component: `gpu`, `docker`, `packages`, `services`, `power`, `drivers`, `users`, `registry`, `firewall`, `clipboard`, `screen`, `camera`, `microphone` |
| `tool` | an MCP tool, as `server/tool` |

Paths are made absolute, `..` is normalized, and existing symlinks (including
symlinked parent directories of a not-yet-created output file) are resolved
before matching. This prevents a path inside an allowed checkout from
silently resolving outside it. The policy is still an authorization check, not
a filesystem sandbox: callers must use the same validated destination for the
actual operation and guard against symlink replacement between checking and
opening a file.

## Rules, profiles and layers

A rule allows, asks or denies one action (or `*`) on targets matching a
pattern: `exact`, `prefix`, `glob` (`*` within a folder, `**` across folders)
or `regex`. Allow rules are the whitelist and deny rules the blacklist.
Patterns may use `{workdir}`, `{repo}`, `{home}`, `{tmp}` and
`{omnicortex}`.

A profile is a set of rules plus a default per action. A check stacks:

1. the **base** profile (`standard` unless you choose another);
2. the profiles **bound to the model** (a glob on the model id);
3. the **job's** profile (below);
4. a profile **asked for** by a project task or a graph node.

Every matching rule in every layer counts, and the most restrictive wins:
deny over ask over allow. A job can open up what it needs, but it cannot
lift a deny from the base or the model's profile. If no rule matches, the
default comes from the most specific layer that sets one.

### Built-in profiles

| profile | for |
|---|---|
| `standard` | The base. Denies the dangerous commands (removing home or root, formatting, raw device writes, fork bombs, power, `curl … \| sh`, `sudo`) and your secrets (`~/.ssh`, `~/.aws`, `.env`, keys…). Reading is fine; everything else asks. |
| `read_only` | Reading only. |
| `no_actions` | Nothing at all: question answering (TriviaQA, LiveBench, TestQA without code). |
| `code_sandboxed` | Python, Docker or Podman; writes only in the job's folder; no network. |
| `swe_bench` | git, Python, pytest, pip and Docker inside the checkout, packages from PyPI, never `git push` or git hooks. |
| `developer` | A coding agent: writes in the project, its build tools, asks for the rest. |
| `full` | Everything, except what the base denies. |

Built-in profiles cannot be edited; **Copy as…** makes an editable one.

### Jobs

Each job kind starts with the preset that fits it (`JOB_PRESETS`):

| job | profile |
|---|---|
| TestQA (`testqa`) | `code_sandboxed` |
| SWE-bench evaluation | `swe_bench` |
| SWE-bench prediction, LiveBench, LM Studio evaluation, judging and the other question-answering runs | `no_actions` |
| project tasks | `developer` |

Settings → Permissions → Jobs lets you pick another profile for any of them.

## Prompts

When the answer is **ask**, the chat shows what the call would do: each
command line, path, program and host, with its own answer. It then offers:

- **Allow once**: this call only.
- **Always allow**: this exact call stops asking for this model. Its targets go into the model's own profile (`remembered-<model>`, exact matches), and its exact arguments into tool approval. Both can be edited or removed in Settings.
- **Deny**.
- **more…**: broader standing rules, such as any arguments to this tool, or always deny it.

Batch jobs have no one to ask, so they treat ask as no.

## Command line

```bash
python scripts/permissions.py                                   # the policy (~/.omnicortex/permissions.json, or $OC_PERMISSIONS)
python scripts/permissions.py presets
python scripts/permissions.py check run "git apply fix.patch" --job swe_evaluate --repo /work/repo
python scripts/permissions.py bind '*coder*' developer
python scripts/permissions.py job testqa read_only
```

## Limits

A policy decides what may be *started*. It cannot see inside a program it
allows: allowing `python` allows whatever that Python script does. For code
you do not trust, combine a narrow profile with a sandbox (TestQA's
`--sandbox docker`, SWE-bench's containers).
