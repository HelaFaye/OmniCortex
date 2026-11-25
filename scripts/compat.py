#!/usr/bin/env python3
"""
Hardware compatibility: what each part of OmniCortex does on each family of hardware, said plainly.

The matrix lives in profiles/compatibility.json: one row per hardware profile (profiles/hardware/), one column per
engine or feature. Each cell has a status:

- **verified**: run on this hardware by OmniCortex's tests or a recorded trial (`evidence` says which);
- **expected**: should work from the engine's or library's own support, not run here yet;
- **partial**: works with the limit its note states;
- **unsupported**: does not work (its note says why);
- **unknown**, or **n/a**.

    python scripts/compat.py show                 # this computer's matching profiles
    python scripts/compat.py show amd-gfx906-mx   # one profile
    python scripts/compat.py doc > docs/COMPATIBILITY.md
    python scripts/compat.py check                # the matrix is complete, and the doc matches it
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "profiles" / "compatibility.json"
DOC = ROOT / "docs" / "COMPATIBILITY.md"
STATUSES = ("verified", "expected", "partial", "unsupported", "unknown", "n/a")
MARK = {"verified": "✅ verified", "expected": "expected", "partial": "◐ partial", "unsupported": "✗ no",
        "unknown": "?", "n/a": "–"}


def load(path: Path = MATRIX) -> dict:
    return json.loads(path.read_text())


def problems(m: dict, profile_ids: list[str]) -> list[str]:
    """Everything wrong with the matrix: missing rows or cells, bad statuses, unexplained claims."""
    out = []
    feats = list(m["features"])
    for pid in profile_ids:
        if pid not in m["profiles"]:
            out.append(f"{pid}: no row (every hardware profile needs one)")
    for pid, row in m["profiles"].items():
        if pid not in profile_ids:
            out.append(f"{pid}: a row for a profile that doesn't exist")
        for f in feats:
            c = row.get(f)
            if c is None:
                out.append(f"{pid}.{f}: missing")
                continue
            if c.get("status") not in STATUSES:
                out.append(f"{pid}.{f}: status {c.get('status')!r} is not one of {', '.join(STATUSES)}")
            if c.get("status") == "verified" and not c.get("evidence"):
                out.append(f"{pid}.{f}: verified needs evidence")
            if c.get("status") in ("partial", "unsupported") and not c.get("note"):
                out.append(f"{pid}.{f}: {c['status']} needs a note saying why")
        for f in row:
            if f not in feats:
                out.append(f"{pid}.{f}: not a feature")
    return out


def row_text(m: dict, pid: str, title: str = "") -> str:
    L = [f"{pid}" + (f": {title}" if title else "")]
    for f, desc in m["features"].items():
        c = m["profiles"][pid][f]
        why = c.get("evidence") or c.get("note") or ""
        L.append(f"  {c['status']:<11} {desc}" + (f"\n              {why}" if why else ""))
    return "\n".join(L)


def doc(m: dict, titles: dict) -> str:
    feats = list(m["features"])
    short = {"llama_cpp": "llama.cpp", "mx_llama": "mx-llama.cpp", "radiance": "Radiance", "pytorch": "PyTorch",
             "cett_dump": "H-neurons (GGUF)", "truth_probe": "Truth probe (GGUF)", "lora": "LoRA",
             "qlora": "QLoRA", "bf16": "bf16", "quant_fidelity": "Quant fidelity"}
    L = ["# Hardware compatibility", "",
         "What each part of OmniCortex does on each family of hardware. This page is generated from",
         "`profiles/compatibility.json` by `python scripts/compat.py doc`, and a test fails if the two disagree.",
         f"Reviewed {m['reviewed']}.", "",
         "**Statuses:**", ""]
    L += [f"- **{s}**: {d}" for s, d in m["statuses"].items()]
    L += ["", "**Only the CPU rows have been run on real hardware so far**: everything OmniCortex tests runs in CI on an",
          "x86 CPU. Every GPU cell is *expected*, *partial* or *unsupported* until it has been run on that card and its",
          "evidence recorded here.", "",
          "| Hardware | " + " | ".join(short.get(f, f) for f in feats) + " |",
          "|---|" + "---|" * len(feats)]
    for pid, row in m["profiles"].items():
        L.append(f"| `{pid}` | " + " | ".join(MARK[row[f]["status"]] for f in feats) + " |")
    L += ["", "## The reasons, by hardware", ""]
    for pid, row in m["profiles"].items():
        L += [f"### `{pid}`" + (f": {titles[pid]}" if titles.get(pid) else ""), ""]
        for f in feats:
            c = row[f]
            why = c.get("evidence") or c.get("note")
            if c["status"] in ("verified", "partial", "unsupported", "unknown") or why:
                L.append(f"- **{short.get(f, f)}**: {c['status']}" + (f". {why[0].upper() + why[1:]}." if why else "."))
        L.append("")
    L += ["## Changing it", "",
          "- **A cell moves to verified** only with evidence: a CI test that runs on that hardware, or a trial",
          "  recorded with its date and results (Setup → Hardware support profiles → *It worked here* records one).",
          "- **A new hardware profile** needs a row; `compat.py check` fails without one.",
          "- **Every partial or unsupported cell** says why.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import hw_profiles as HP
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("profile", nargs="?", help="a profile id (default: those that match this computer)")
    sub.add_parser("doc")
    sub.add_parser("check")
    a = p.parse_args(argv)
    m = load()
    profiles = {x["id"]: x for x in HP.catalogue() if x.get("status") == "built-in"}
    titles = {k: v.get("title", "") for k, v in profiles.items()}
    if a.cmd == "check":
        errs = problems(m, sorted(profiles))
        if DOC.read_text() != doc(m, titles):
            errs.append("docs/COMPATIBILITY.md is out of date: python scripts/compat.py doc > docs/COMPATIBILITY.md")
        print("\n".join(errs) or "ok")
        return 1 if errs else 0
    if a.cmd == "doc":
        sys.stdout.write(doc(m, titles))
        return 0
    ids = [a.profile] if a.profile else [x["id"] for x in HP.match(list(profiles.values()), HP.facts())]
    if not ids:
        print("no profile matches this computer; name one: " + ", ".join(sorted(profiles)))
        return 1
    for pid in ids:
        if pid not in m["profiles"]:
            print(f"no compatibility row for {pid}")
            return 1
        print(row_text(m, pid, titles.get(pid, "")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
