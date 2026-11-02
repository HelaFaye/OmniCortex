# Mode packages

A mode package adds a tab to OmniCortex's interfaces. Drop a directory here
(`ns-eval/` is an example of the layout; it ships without `ENABLED` because it
has no implementation behind its route):

```
viz/modes.d/ns-eval/
  mode.json     the registry entry
  ENABLED       present = active; absent = listed but off
  ...           the implementation
```

`mode.json` needs `id`, `label`, `summary`, `surfaces`, and usually `glyph` and
`route`. `id` may not shadow a built-in mode. Optional `grants` declares what
the package wants to reach, which is shown before you enable it.

```json
{
  "id": "eval",
  "label": "Eval",
  "glyph": "✓",
  "summary": "Paired evaluation of two endpoints on held-out tasks.",
  "surfaces": ["control"],
  "route": "/#eval",
  "author": "example",
  "grants": ["read:profiles", "run:merge_eval"]
}
```

## Why enabling is a separate step

Getting a package can use the same interface as downloading a model. Enabling
one should not. A model is inert data: it cannot run until you load it, and
even then it only emits tokens. A mode package is code that runs in your
browser against an API that starts processes on this machine.

So a package is discovered, listed and described automatically, and does
nothing until an `ENABLED` file exists. `shell.validate()` produces the summary
to show first — which tabs it adds, who wrote it, what it asks for, and
explicitly whether it can start processes.

A malformed package is reported and skipped; it cannot take the registry down.

```bash
python viz/shell.py --packages          # what is installed, and its state
```
