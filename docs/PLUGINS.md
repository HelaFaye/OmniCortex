# Plugins

A plugin adds to Studio without changing OmniCortex's code. It is a folder
with a `plugin.json`, and it can add any of:

| part | what it is | where it shows |
|---|---|---|
| `jobs` | a script with a form | Jobs → *Plugins: name* |
| `checks` | a script that looks at a reply | *plugins* under each reply in the chat |
| `page` | a folder of HTML, CSS and JavaScript | Settings → Plugins → *open page* (`/plugins/<name>/`) |
| `mcp` | an MCP server | the chat's **tools**, like servers in `mcp.json` |

`plugins/claim-spotter/` in the repository has one of each but MCP. It marks
the checkable claims in a reply (numbers, dates, quotations) and collects
them from saved chats into a CSV.

## Installing and turning on

1. Put the plugin's folder in `~/.neuronscope/plugins/` (Studio's home; the
   folder sits next to `config.json` for other workspaces). The examples in
   the repository's `plugins/` are found too.
2. Settings → **Plugins** lists every plugin found, with what it adds.
3. **Turn on** the ones you trust.

**Plugins run code on your computer with your permissions.** So:
- none runs until you turn it on;
- turning it on records a SHA-256 of all its files. If any file changes
  afterwards (an update, or anything else), the plugin is off again and
  shows *changed* until you turn it on anew;
- scripts run as separate processes in the plugin's folder: a failing
  plugin cannot take Studio down;
- only the owner can turn plugins on, run their checks or start their
  jobs; paired devices cannot;
- a plugin page is served with a policy that lets it call Studio's API but
  load nothing from other sites.

## plugin.json

```json
{
  "name": "claim-spotter",
  "version": "1.0",
  "author": "you",
  "description": "One or two sentences shown in Settings.",
  "checks": [
    {"name": "claims", "title": "Claims to verify", "script": "check.py", "auto": false,
     "help": "shown on hover"}
  ],
  "jobs": [
    {"kind": "collect", "title": "Collect claims from saved chats", "script": "collect.py",
     "help": "shown above the form",
     "fields": [
       {"name": "chats", "required": true, "help": "…"},
       {"name": "limit", "kind": "int", "default": 100},
       {"name": "verbose", "kind": "bool"},
       {"name": "mode", "choices": ["fast", "thorough"], "default": "fast"}
     ]}
  ],
  "page": "page",
  "mcp": {"command": "python", "args": ["server.py"]}
}
```

- `name`: lower-case letters, digits, `-` and `_`.
- Paths are relative to the plugin's folder and must stay inside it.
- Scripts ending in `.py` run with Studio's Python. For anything else, give
  `"interp": ["node"]` (or another program) on the job or check.

### Jobs

Each field becomes a flag: `limit` → `--limit 100`. A field may set its
own `"flag"`, or `"flag": ""` for a positional argument.
- Field kinds: `str` (default), `int`, `float`, `bool` (the flag alone,
  when ticked).
- `multi: true` takes several space-separated values after one flag.

Values are checked the way built-in jobs' are: no value may start with
`-` or contain control characters.

The job runs in the current workspace's folder. Its output is the job's
log, and it can be cancelled from the Jobs page.

### Checks

The script reads one JSON object on stdin:

```json
{"model": "the model id", "messages": [{"role": "user", "content": "…"}], "text": "the reply"}
```

It prints one JSON object as its last line of output:

```json
{"label": "2 claims to verify", "score": 2, "flag": false, "note": "1 year, 1 number",
 "spans": [[18, 22, "year"], [30, 35, "number"]]}
```

- `label` is the headline.
- `flag: true` shows it as a warning.
- `spans` are character offsets into `text`; *show* highlights them.
- Every field is optional.

Checks marked `"auto": true` run after every reply. The rest run when you
click *plugins* under a reply. A check has 60 seconds, and its results are
saved with the chat.

### Pages

The `page` folder is served at `/plugins/<name>/`, with `index.html` as
its start. It is on Studio's own address, so it can use Studio's API with
`fetch` (the session cookie applies): for example `/api/plugins/check`,
`/v1/chat/completions` or `/api/traces`.

### MCP

`mcp` takes what an `mcp.json` entry takes (`command` + `args` + `env`, or
`url` + `headers`). The server starts in the plugin's folder when the plugin
is turned on, and its tools join the chat's **tools** as `plugin-<name>`.
They follow the tool-approval rules like any other tool.

## Studio API

| route | who | what |
|---|---|---|
| `GET /api/plugins` | signed in | plugins found, their status and what they add; the checks of plugins that are on |
| `POST /api/plugins/enable` | owner | `{"name", "on"}` |
| `POST /api/plugins/check` | owner | `{"plugin", "check", "model", "messages", "text"}` → the check's result |
| `GET /plugins/<name>/<path>` | signed in | the plugin's page (plugins that are on only) |

Plugin jobs go through `/api/jobs` as `plugin:<name>:<kind>`.

## Graph nodes

`"nodes"` adds node types to Studio's Graphs ([GRAPHS.md](GRAPHS.md)):

```json
"nodes": [{"type": "word_count", "title": "Count words", "script": "node.py",
           "inputs": [["text", "text", true]], "outputs": [["count", "number"], ["long", "text"]],
           "params": [{"name": "over", "kind": "int", "default": 100}], "help": "..."}]
```

The script reads `{"params": {...}, "inputs": {...}}` as JSON on stdin and
prints one JSON object of outputs. An output it leaves out is skipped, so a
node can route; printing `{"error": "..."}` fails the node. The node type
appears as `plugin:<plugin>:<type>` while the plugin is on.

## Packaging and tabs

`python viz/addons.py pack <folder>` makes an installable `.zip`, and
`python viz/addons.py index <dist>` makes a repository index
([ADDONS.md](ADDONS.md)). A plugin with a `page` can also declare
`"tab": {"label": "ComfyUI", "tip": "..."}` to get a sidebar tab while it is on.

## Defaults, widgets, themes and downloads

- `"default": true` marks a plugin that ships in the checkout's `plugins/` folder as on unless the owner turns it off ([ADDONS.md](ADDONS.md#default-packages)).
- `"widgets": {"jewel_styles": ["name"], "js": "styles.js", "gd": "styles.gd"}` adds jewel styles. The files sit in the page folder. The script calls `NSWidgets.jewelStyle(name, (ctx, J) => …)`; the GDScript has `STYLES` and `draw(style, jewel, centre, radius)`.
- `"themes": "themes.json"` adds Studio themes (under `"_studio"`) and 3D palettes.
- `"downloads": {"name.zip": "folder"}` serves a folder of the plugin as a zip at `/plugins/<name>/download/name.zip`.

