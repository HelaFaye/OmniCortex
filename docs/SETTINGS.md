# Settings: workspaces, the model per subject, tool approval

Studio's **Settings** page (`/settings`) has three parts. Each also works
from the command line. Behaviour settings (unloading, routing thresholds,
worker limits) and building llama.cpp stay on **Setup**.

## Workspaces

A workspace says where things are, so nothing assumes a layout:

| field | what | default |
|---|---|---|
| Pipeline data root | the folder holding the pipeline's `data/`, `models/`, `runs/`, `profiles/`, `adapters/`; jobs run here | the repository |
| Models folders | where Studio looks for `.gguf` files | `<root>/models/gguf`, `~/models`, LM Studio's folders |
| Downloads folder | where model downloads go | the first models folder |
| llama.cpp checkout | binaries are looked for in its `build/bin` | `$OC_LLAMA` or `~/llama.cpp` |
| llama-server, llama-cett-dump | explicit binaries, if they live elsewhere | from the checkout |
| Model for the pipeline | the GGUF the H-Neuron pipeline and `doctor.py` work on | `$OC_GGUF` |
| Studio state folder | chats, traces, review, projects, jobs, stats, documents | `~/.omnicortex` |
| Environment | variables for jobs and model servers, e.g. `VK_DRIVER_FILES`, `HSA_OVERRIDE_GFX_VERSION` | none |

Keep one per disk layout or machine: say the repository on an external drive
with its data, models on the internal disk, and a Vulkan driver pinned
because a second ICD fails to load. **Use this workspace** makes it the
active one; Studio switches after a restart (the button is right there).
`PATH`, `PYTHONPATH`, `LD_PRELOAD` and similar variables cannot be set from a
workspace.

Precedence, highest first: Studio's command-line flags, the `studio`
section of `~/.omnicortex/config.json` (behaviour settings), then the
workspace. Setup's folder fields write to the running workspace.

```bash
python scripts/oc_workspace.py list
python scripts/oc_workspace.py add ext --root /run/media/me/disk/omnicortex --models-dir ~/models --use
python scripts/oc_workspace.py set ext env.VK_DRIVER_FILES /usr/share/vulkan/icd.d/radeon_icd.x86_64.json
eval "$(python scripts/oc_workspace.py env)"      # NS_* for scripts you run by hand
./omnicortex --workspace ext                       # or start Studio in one
python scripts/doctor.py --workspace ext
```

## Models per subject

With model `auto`, Studio sends each request to the model with the best
measured record for the prompt's subject (from TestQA and benchmark results;
see [TESTQA.md](TESTQA.md)). The table shows that measured best per subject.
Pick a model for a subject to make it the default for that subject. Your
pick wins whenever the subject is at least 30% of the prompt, even before
any model has enough graded results. Leave it on *measured best* to follow
the numbers. The routing reason in the chat says which happened.

Stored as `subject_models` in the `studio` section of `config.json`.

## Tool approval

When a model calls an MCP tool in chat (Studio → Tools), each call is
decided in this order:

1. **Rules**, top to bottom; the first match decides *allow*, *ask* or
   *deny*. A rule matches on:
   - the **server** and **tool** (globs: `filesystem`, `read_*`, `*`);
   - a **kind of operation** (below), or any;
   - **argument conditions**: a dotted path into the arguments and a
     regular expression, optionally negated. `files.*.path` reaches into
     lists and objects; `*` alone means any text anywhere in the arguments.
     All conditions of a rule must hold.

   It can also have a **limit** (allow at most N calls an hour, then ask),
   an **expiry** and a note. Each rule counts its hits.
2. The server's **`autoApprove`** list in `mcp.json`, as in Cline and Claude
   Desktop.
3. **Quick settings** per kind of operation, from the tool's own MCP
   annotations:
   - read-only;
   - writes (not destructive);
   - destructive (also any tool not marked read-only, which is the MCP
     default);
   - reaches the outside world;
   - no annotations.

   A tool of several kinds gets the strictest of their settings.
4. The **default** (ask).

On an approval card in chat, besides *Allow once* and *Deny*, **always…**
adds a rule:
- always allow this tool;
- always allow these exact arguments (each argument matched literally);
- always deny this tool.

Paired devices can answer an approval but not create rules.

**Add common rules…** inserts starting points to review:
- deny destructive shell commands anywhere in the arguments;
- deny paths to keys and secrets;
- allow read-only tools up to 200 an hour;
- an example for reading a notes folder.

**Try a call** shows what a call would get, and why. Annotations are the
tool author's claim; a rule naming the tool is the stronger statement.

```bash
python scripts/tool_rules.py                                   # rules, quick settings, default
python scripts/tool_rules.py add allow --server filesystem --tool 'read_*' --arg 'path=^/home/me/notes/'
python scripts/tool_rules.py add deny --arg '*=(?i)rm -rf'
python scripts/tool_rules.py add allow --kind read_only --max-per-hour 200
python scripts/tool_rules.py category destructive deny
python scripts/tool_rules.py test filesystem read_file '{"path": "/home/me/notes/a.md"}' --annotations '{"readOnlyHint": true}'
```

Stored in `~/.omnicortex/tool-rules.json` (or `$OC_TOOL_RULES`).

## Appearance

Settings → **Appearance** lists the themes. Each one sets:
- Studio's page colours;
- the chart colours (the model radar: 1P, 2P…);
- the 3D viewer's colours: background, active neurons, flagged
  hallucination neurons.

Clicking a theme applies it in this browser and remembers it there. The 3D
views opened from this browser follow it. **Make default** saves it in
`config.json` (`studio.theme`). Browsers that have not chosen a theme use it,
and so does the Godot client's 3D view. The header's ◐ button steps through
the themes.

Themes live in `viz/themes.json`:
- under `_studio`: a label, a `base` (dark or light), `ui` colours,
  `chart` colours, and the `viz` theme for the 3D views;
- at the top level: the viewer themes themselves (colour map, active and
  flagged colours, point sizes).

A new theme is an entry there. The flagged-neuron colour of every viewer
theme is chosen to stay distinct from the active colour for common kinds
of colour blindness; keep that when adding one.

## The model radar

Studio → Models → a model shows a radar, in the theme's chart colours, of:
- its accuracy per subject, for subjects with at least 3 graded items;
- *honesty*, which is 1 − its hallucination rate.

**Compare with (2P)** overlays another model. The bars under it are the
average over the axes, in tenths.

## Plugins

Lists the plugins found in `~/.omnicortex/plugins/` and the repository's
`plugins/`, with what each adds (jobs, reply checks, a page, MCP tools).
Plugins run code, so each stays off until you turn it on. If its files
change later, it switches itself off until you turn it on again.
[PLUGINS.md](PLUGINS.md) explains how to write one.
