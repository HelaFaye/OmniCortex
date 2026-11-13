#!/usr/bin/env python3
"""
The shared tab shell. Every OmniCortex surface renders its navigation here.

Four frontends with four hardcoded tab bars would drift within a week, so the
modes live in viz/modes.json and this turns them into markup. A surface asks
for the shell, says which mode it is showing, and gets the same navigation as
every other surface -- including tabs it does not implement, shown disabled.

Showing the unimplemented ones matters. Hiding them makes each surface look
complete while the set of features silently differs between them; disabling
them makes the gap visible and tells you where to go instead.

    from shell import tabs_html, shell_css, modes
    page = f"<style>{shell_css()}</style>{tabs_html('control', 'run')}..."

Adding a mode is an entry in modes.json. Adding it to a surface is appending
that surface's name to the mode's `surfaces` list, and implementing it.
"""

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(_HERE, "modes.json")
_PKGS = os.path.join(_HERE, "modes.d")
_CACHE = {"stamp": None, "data": None}

# A mode package is a directory under viz/modes.d/ holding mode.json plus its
# implementation. Built-in modes and packaged ones render identically; the
# difference is that a package is code someone else wrote, so it carries
# `packaged` and stays disabled until explicitly enabled.
REQUIRED = ("id", "label", "summary", "surfaces")


def _stamp():
    parts = []
    for p in (_PATH, _PKGS):
        try:
            parts.append(os.path.getmtime(p))
        except OSError:
            parts.append(0)
    try:
        for d in sorted(os.listdir(_PKGS)):
            f = os.path.join(_PKGS, d, "mode.json")
            parts.append(os.path.getmtime(f) if os.path.exists(f) else 0)
    except OSError:
        pass
    return tuple(parts)


def packages():
    """-> [{dir, mode, enabled, error}] for everything in viz/modes.d/."""
    out = []
    try:
        names = sorted(os.listdir(_PKGS))
    except OSError:
        return out
    for name in names:
        d = os.path.join(_PKGS, name)
        f = os.path.join(d, "mode.json")
        if not os.path.isfile(f):
            continue
        rec = {"dir": name, "mode": None, "enabled": False, "error": None}
        try:
            with open(f) as fh:
                m = json.load(fh)
            missing = [k for k in REQUIRED if k not in m]
            if missing:
                rec["error"] = f"mode.json missing {', '.join(missing)}"
            elif not isinstance(m.get("surfaces"), list):
                rec["error"] = "surfaces must be a list"
            else:
                m["packaged"] = name
                rec["mode"] = m
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["enabled"] = os.path.exists(os.path.join(d, "ENABLED"))
        out.append(rec)
    return out


def modes(include_disabled=False):
    """Built-in modes plus enabled packages, re-read when anything changes."""
    st = _stamp()
    if _CACHE["stamp"] != st:
        try:
            with open(_PATH) as f:
                base = json.load(f).get("modes", [])
        except OSError:
            base = []
        seen = {m["id"] for m in base}
        for rec in packages():
            m = rec["mode"]
            if not m or m["id"] in seen:
                continue          # a package may not shadow a built-in id
            m = dict(m, enabled=rec["enabled"])
            base.append(m)
            seen.add(m["id"])
        _CACHE["data"] = base
        _CACHE["stamp"] = st
    if include_disabled:
        return _CACHE["data"]
    return [m for m in _CACHE["data"]
            if not m.get("packaged") or m.get("enabled")]


def modes_for(surface):
    return [m for m in modes() if surface in m.get("surfaces", [])]


def shell_css():
    return """
:root{--ns-bg:#0b0b10;--ns-panel:#14141c;--ns-line:#23232e;--ns-fg:#c9c9d2;
--ns-dim:#8a8a92;--ns-accent:#5DCAA5;--ns-warn:#EF9F27}
.ns-tabs{display:flex;gap:2px;overflow-x:auto;border-bottom:1px solid var(--ns-line);
margin:-14px -14px 14px;padding:0 8px;background:var(--ns-panel);
padding-top:env(safe-area-inset-top,0px);-webkit-overflow-scrolling:touch}
.ns-tab{flex:none;display:flex;align-items:center;gap:6px;padding:11px 14px;
color:var(--ns-dim);text-decoration:none;font:500 13px system-ui;
border-bottom:2px solid transparent;white-space:nowrap;min-height:44px}
.ns-tab:hover{color:var(--ns-fg)}
.ns-tab.on{color:#fff;border-bottom-color:var(--ns-accent)}
.ns-tab.off{opacity:.38;cursor:not-allowed}
.ns-tab .g{font-size:12px;opacity:.85}
.ns-tab .badge{font-size:10px;padding:1px 5px;border-radius:6px;
background:var(--ns-line);color:var(--ns-dim)}
.ns-hint{color:var(--ns-dim);font-size:12px;margin:-6px 0 12px}
"""


def tabs_html(surface, active, links=None):
    """Tab bar for one surface.

    `links` maps a mode id to a URL on another surface, so a tab this surface
    cannot show can still point at one that can, instead of being a dead end.
    """
    links = links or {}
    out = ['<nav class="ns-tabs">']
    for m in modes():
        mid = m["id"]
        mine = surface in m.get("surfaces", [])
        planned = m.get("planned")
        href = m.get("route", "#") if mine else links.get(mid)
        classes = "ns-tab" + (" on" if mid == active else "")
        if not mine and not href:
            classes += " off"
        title = m.get("summary", "")
        if not mine:
            where = ", ".join(m.get("surfaces", [])) or "nothing yet"
            title += f"  (lives in: {where})"
        badge = '<span class="badge">soon</span>' if planned else ""
        if m.get("packaged"):
            badge += '<span class="badge">pkg</span>' 
        tag = "a" if href else "span"
        attr = f' href="{href}"' if href else ""
        out.append(
            f'<{tag} class="{classes}"{attr} title="{_esc(title)}">'
            f'<span class="g">{m.get("glyph", "")}</span>'
            f'{_esc(m["label"])}{badge}</{tag}>')
    out.append("</nav>")
    cur = next((m for m in modes() if m["id"] == active), None)
    if cur:
        out.append(f'<div class="ns-hint">{_esc(cur.get("summary", ""))}</div>')
    return "".join(out)


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def validate(rec):
    """What a package would add, for the install prompt.

    Deliberately not automatic. A model you download is inert data; a mode
    package is code that runs in the browser against an API which can start
    processes. The interface for getting one can be the same as for models;
    the moment of enabling it should not be.
    """
    if rec.get("error"):
        return {"ok": False, "reasons": [rec["error"]]}
    m = rec["mode"]
    notes, warn = [], []
    notes.append(f"adds the tab \"{m['label']}\" to: "
                 f"{', '.join(m.get('surfaces', []))}")
    if m.get("author"):
        notes.append(f"author: {m['author']}")
    for g in m.get("grants", []):
        warn.append(f"requests {g}")
    if any(g.startswith("run:") for g in m.get("grants", [])):
        warn.append("can start processes on this machine")
    if m.get("route", "").startswith(("http://", "https://")):
        warn.append(f"loads from an external origin: {m['route']}")
    return {"ok": True, "id": m["id"], "label": m["label"],
            "notes": notes, "warnings": warn}


def set_enabled(dirname, on):
    d = os.path.join(_PKGS, dirname)
    if not os.path.isdir(d):
        return False
    flag = os.path.join(d, "ENABLED")
    if on:
        open(flag, "w").close()
    elif os.path.exists(flag):
        os.remove(flag)
    _CACHE["stamp"] = None
    return True


def as_json(surface=None):
    """The registry for a non-HTML client -- Godot, or a native frontend."""
    ms = modes_for(surface) if surface else modes()
    return json.dumps({"surface": surface, "modes": ms})


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="inspect the mode registry")
    p.add_argument("--surface")
    p.add_argument("--packages", action="store_true")
    p.add_argument("--enable", metavar="DIR")
    p.add_argument("--disable", metavar="DIR")
    a = p.parse_args()
    if a.enable or a.disable:
        d = a.enable or a.disable
        ok = set_enabled(d, bool(a.enable))
        print(f"{d}: {'enabled' if a.enable else 'disabled'}" if ok
              else f"no such package: {d}")
        raise SystemExit(0 if ok else 1)
    if a.packages:
        pk = packages()
        if not pk:
            print("no packages in viz/modes.d/")
        for rec in pk:
            v = validate(rec)
            mark = "on " if rec["enabled"] else "off"
            if not v["ok"]:
                print(f"  [err] {rec['dir']}: {v['reasons'][0]}")
                continue
            print(f"  [{mark}] {rec['dir']}  ->  {v['label']}")
            for n in v["notes"]:
                print(f"         {n}")
            for w in v["warnings"]:
                print(f"         WARN {w}")
        raise SystemExit(0)
    if a.surface:
        ms = modes_for(a.surface)
        print(f"{a.surface}: {len(ms)} of {len(modes())} modes")
        for m in ms:
            print(f"  {m['glyph']} {m['label']:<9} {m['summary']}")
        missing = [m for m in modes() if m not in ms]
        if missing:
            print("\n  shown disabled here:")
            for m in missing:
                print(f"  {m['glyph']} {m['label']:<9} "
                      f"-> {', '.join(m['surfaces']) or 'nowhere yet'}")
    else:
        for m in modes():
            print(f"{m['glyph']} {m['label']:<9} {','.join(m['surfaces'])}")
