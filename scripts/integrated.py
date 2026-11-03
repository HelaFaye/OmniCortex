#!/usr/bin/env python3
"""
The integrated options' settings (docs/INTEGRATED_OPTIONS.md): one registry, so Studio's Settings page, its
API (/api/settings/options) and the scripts agree on names, defaults and limits.

Each option: a title, what it does, its fields (kind, default, limits), and for the sensitive ones a warning
shown beside a lock (Settings → Integrated options; the same unlock-to-edit pattern as other risky settings).
"""
from __future__ import annotations

import copy

# kind: bool | int | float | str | list (of strings) | choice
OPTIONS: dict[str, dict] = {
    "neural_reset": {
        "title": "Neural Reset (loop breaker)",
        "help": "When a reply in Studio's chat starts repeating itself, stop it, keep the text up to the loop, and "
                "resume with the loop's opening pushed down and a repeat penalty for the rest of that reply.",
        "fields": {"enabled": ("bool", True), "max_resets": ("int", 2, 0, 10), "min_repeats": ("int", 3, 2, 20),
                   "min_span": ("int", 60, 10, 5000)}},
    "sanitizer": {
        "title": "Prompt sanitizer (for linked hosts)",
        "help": "Before a chat goes to a model on another computer, replace keys, tokens, e-mail addresses and the "
                "names you list with placeholders; restore them in the reply here.",
        "fields": {"enabled": ("bool", False), "terms": ("list", []), "patterns": ("bool", True),
                   "emails": ("bool", True), "ips": ("bool", False)},
        "sensitive": {"enabled": "Off means chats to linked hosts carry your text exactly as typed, secrets included."}},
    "harvest": {
        "title": "Fix This Mistake (deficit harvesting)",
        "help": "A button under each reply takes your correction and keeps the pair for training; with auto-train, "
                "a short DPO micro-tune runs on them when the computer has been idle a while.",
        "fields": {"enabled": ("bool", True), "auto_train": ("bool", False), "train_model": ("str", ""),
                   "min_pairs": ("int", 20, 1, 100000), "idle_minutes": ("int", 20, 1, 1440), "steps": ("int", 40, 1, 5000)},
        "sensitive": {"auto_train": "Starts GPU-heavy training on its own when the computer is idle. It never "
                                    "replaces a model, but it can keep a GPU busy for a long time."}},
    "core_shunt": {
        "title": "Core shunt (background job priority)",
        "help": "Run jobs at low priority, on the CPU's efficiency cores when it has them, so the computer stays "
                "responsive. GPU priority cannot be set portably; a GPU-bound job still shares the GPU.",
        "fields": {"priority": ("choice", "low", ["low", "normal"]), "cores": ("str", "auto")},
        "sensitive": {"cores": "A wrong core list can make every job crawl on one slow core. auto is safe."}},
}


def defaults() -> dict:
    return {k: {f: spec[1] for f, spec in o["fields"].items()} for k, o in OPTIONS.items()}


def merged(saved: dict | None) -> dict:
    out = defaults()
    for k, vals in (saved or {}).items():
        if k in out and isinstance(vals, dict):
            out[k].update({f: v for f, v in vals.items() if f in out[k]})
    return out


def validate(name: str, values: dict, current: dict | None = None) -> dict:
    """values -> the option's full, checked settings (current fills what values leaves out)."""
    if name not in OPTIONS:
        raise ValueError(f"no option {name!r}")
    cur = copy.deepcopy((current or {}).get(name) or defaults()[name])
    for f, v in values.items():
        spec = OPTIONS[name]["fields"].get(f)
        if spec is None:
            raise ValueError(f"{name}: no field {f!r}")
        kind = spec[0]
        if kind == "bool":
            v = bool(v)
        elif kind in ("int", "float"):
            try:
                v = int(v) if kind == "int" else float(v)
            except (TypeError, ValueError):
                raise ValueError(f"{name}.{f}: a number")
            lo, hi = spec[2], spec[3]
            if not lo <= v <= hi:
                raise ValueError(f"{name}.{f}: between {lo} and {hi}")
        elif kind == "choice":
            if v not in spec[2]:
                raise ValueError(f"{name}.{f}: one of {', '.join(spec[2])}")
        elif kind == "list":
            if isinstance(v, str):
                v = [x.strip() for x in v.replace("\n", ",").split(",")]
            v = [str(x).strip()[:200] for x in (v or []) if str(x).strip()][:500]
        else:
            v = str(v or "").strip()[:500]
        cur[f] = v
    if name == "core_shunt" and cur["cores"] not in ("auto", "all"):
        import core_shunt
        try:
            if not core_shunt.parse_cpus(cur["cores"]):
                raise ValueError
        except ValueError:
            raise ValueError("core_shunt.cores: auto, all, or a list like 0-3,8")
    return cur


def describe(saved: dict | None) -> dict:
    vals = merged(saved)
    return {k: {"title": o["title"], "help": o["help"], "values": vals[k],
                "fields": {f: {"kind": s[0], "default": s[1], **({"min": s[2], "max": s[3]} if s[0] in ("int", "float") else {}),
                               **({"choices": s[2]} if s[0] == "choice" else {})} for f, s in o["fields"].items()},
                "sensitive": o.get("sensitive") or {}} for k, o in OPTIONS.items()}
