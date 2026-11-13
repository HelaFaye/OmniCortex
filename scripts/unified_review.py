#!/usr/bin/env python3
"""
One review record per model, from every source OmniCortex knows (docs/UNIFIED_REVIEW.md).

The browser extension (extension/), Studio's /api/lookup and this command all answer the same question:
"what do we know about org/name?" Each source adds statistics in one shape, so a viewer can show every
statistic with a dropdown of where each value came from, and filter by source or group.

    board      a review board's board.json (GitHub board or scoreboard server): verified TestQA accuracy,
               hallucination and decline rates, per subject; measured speed per hardware; and optional,
               self-reported extras (emotion-vector and feature summaries), marked as not re-checked
    hw         the board's hardware-profile reports, for the hardware the model was tested on
    saelens    which SAELens dictionaries exist for this model (features you can open in Studio)
    local      this Studio's own observations of the model (activation profiles collected by default)

    python scripts/unified_review.py lookup Qwen/Qwen2.5-7B-Instruct --board someone/omnicortex-board
    python scripts/unified_review.py lookup org/name --board-file board.json --json

The record (schema omnicortex-lookup/1):
    {"model", "aliases", "sources": [{id, kind, name, ok, error, url}],
     "stats": [{key, group, label, value, unit, n, source, variant, related, detail}]}
groups: quality, performance, features, emotions, activity. `related` marks a value measured on a
re-packaging of the model (a -GGUF, -AWQ, … repo of the same name) rather than the repo itself.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "omnicortex-lookup/1"
GROUPS = ("quality", "performance", "features", "emotions", "activity")
_PACKAGING = re.compile(r"[-_.](gguf|ggml|awq|gptq|exl2|mlx|onnx|int4|int8|fp8|bnb[-_]?4bit|\d+bit)$", re.I)
REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]{0,95}/[A-Za-z0-9][\w.-]{0,95}$")


def base_name(repo: str) -> str:
    """The model's own name with packaging suffixes removed: TheBloke/Foo-7B-GGUF -> foo-7b."""
    name = str(repo or "").rsplit("/", 1)[-1].lower()
    while True:
        new = _PACKAGING.sub("", name)
        if new == name:
            return name
        name = new


def relation(key: str, repo: str | None) -> str | None:
    """'same', 'related' (same model name, another packaging or uploader) or None."""
    if not repo:
        return None
    if repo.lower() == key.lower():
        return "same"
    return "related" if base_name(repo) == base_name(key) else None


# -- self-reported extras on a score card ---------------------------------------------------------

def sanitize_extras(x) -> dict:
    """The optional `extras` block a card may carry: small, numeric, no free text beyond short labels.
    {"emotions": {name: mean activation}, "features": [{"sae", "id", "label", "freq"}]}."""
    out = {}
    if not isinstance(x, dict):
        return out
    emo = x.get("emotions")
    if isinstance(emo, dict):
        out["emotions"] = {re.sub(r"[^a-z_-]", "", str(k).lower())[:32]: round(float(v), 4)
                           for k, v in list(emo.items())[:64] if isinstance(v, (int, float)) and str(k).strip()}
        out["emotions"].pop("", None)
    feats = x.get("features")
    if isinstance(feats, list):
        keep = []
        for f in feats[:32]:
            if isinstance(f, dict) and isinstance(f.get("freq"), (int, float)) and f.get("id") is not None:
                keep.append({"sae": str(f.get("sae") or "")[:80], "id": str(f["id"])[:16],
                             "label": re.sub(r"[\x00-\x1f<>]", "", str(f.get("label") or ""))[:80],
                             "freq": round(float(f["freq"]), 4)})
        out["features"] = keep
    return {k: v for k, v in out.items() if v}


def aggregate_extras(cards: list[dict]) -> dict:
    """Board side: mean emotion activations and merged feature frequencies over a row's cards."""
    emo, feats = {}, {}
    for c in cards:
        ex = sanitize_extras(c.get("extras"))
        for k, v in (ex.get("emotions") or {}).items():
            emo.setdefault(k, []).append(v)
        for f in ex.get("features") or []:
            feats.setdefault((f["sae"], f["id"]), {**f, "freqs": []})["freqs"].append(f["freq"])
    out = {}
    if emo:
        out["emotions"] = {k: {"mean": round(statistics.fmean(v), 4), "n": len(v)} for k, v in emo.items()}
    if feats:
        out["features"] = sorted(({"sae": f["sae"], "id": f["id"], "label": f["label"],
                                   "freq": round(statistics.fmean(f["freqs"]), 4), "n": len(f["freqs"])}
                                  for f in feats.values()), key=lambda f: -f["freq"])[:32]
    return out


# -- sources -> stats -----------------------------------------------------------------------------

def _stat(key, group, label, value, source, unit="", n=None, variant="", related=False, detail=None):
    return {"key": key, "group": group, "label": label, "value": value, "unit": unit, "n": n, "source": source,
            "variant": variant, "related": related, "detail": detail}


def from_board(board: dict, key: str, source: str) -> list[dict]:
    """board.json (GitHub board) or a scoreboard server's /api/board rows."""
    rows = board.get("rows") if isinstance(board.get("rows"), list) else board.get("board") or []
    out = []
    for r in rows:
        rel = relation(key, r.get("hf_repo"))
        if not rel:
            continue
        v = r.get("variant") or {}
        var = " · ".join(str(x) for x in (r.get("hf_repo") if rel == "related" else None, r.get("quant"),
                                          v.get("engine"), "LoRA" if v.get("lora") else None,
                                          f"suppression {v['suppression']:g}" if isinstance(v.get("suppression"), (int, float)) and v["suppression"] else None) if x)
        common = dict(source=source, variant=var, related=rel == "related")
        n = r.get("items")
        if r.get("accuracy") is not None:
            out.append(_stat("accuracy", "quality", "Accuracy (TestQA, verified)", r["accuracy"], unit="ratio", n=n,
                             detail={"submissions": r.get("submissions"), "per_subject": r.get("per_subject")}, **common))
        if r.get("hallucination_rate") is not None:
            out.append(_stat("hallucination_rate", "quality", "Confident mistakes", r["hallucination_rate"], unit="ratio",
                             n=n, **common))
        for s, ps in sorted((r.get("per_subject") or {}).items()):
            if ps.get("accuracy") is not None:
                out.append(_stat(f"accuracy:{s}", "quality", f"Accuracy: {s}", ps["accuracy"], unit="ratio",
                                 n=ps.get("n"), **common))
        for p in r.get("performance") or []:
            if not p.get("gen_tok_s"):
                continue
            hw = p.get("hardware") or {}
            gpus = ", ".join(g.get("name") or "?" for g in hw.get("gpus") or []) or hw.get("cpu") or "CPU"
            out.append(_stat(f"gen_tok_s:{gpus.lower()}", "performance", f"Generation speed: {gpus}", p["gen_tok_s"],
                             unit="tok/s",
                             n=p.get("requests"), detail={"hardware": gpus, "parallel": p.get("parallel"),
                                                          "ngl": p.get("ngl"), "ctx": p.get("ctx"),
                                                          "flash_attn": p.get("flash_attn")},
                             **{**common, "variant": " · ".join(x for x in (var, gpus) if x)}))
        ex = r.get("extras") or {}
        for name, e in sorted((ex.get("emotions") or {}).items(), key=lambda kv: -abs(kv[1]["mean"])):
            out.append(_stat(f"emotion:{name}", "emotions", name, e["mean"], unit="activation", n=e.get("n"),
                             detail={"self_reported": True}, **common))
        for f in ex.get("features") or []:
            out.append(_stat(f"feature:{f['sae']}:{f['id']}", "features", f["label"] or f"#{f['id']}", f["freq"],
                             unit="freq", n=f.get("n"), detail={"sae": f["sae"], "id": f["id"], "self_reported": True},
                             **common))
    return out


def from_hw_reports(board: dict, hardware_names: set[str], source: str) -> list[dict]:
    """Hardware-profile reports whose devices were seen running this model (by GPU name)."""
    out = []
    reps = ((board.get("hw_profiles") or {}).get("reports")) or {}
    for pid, by in sorted(reps.items()):
        for purpose, a in sorted(by.items()):
            names = {str(h).lower() for h in a.get("hardware") or []}
            if not names & hardware_names:
                continue
            out.append(_stat(f"hw:{pid}:{purpose}", "performance", f"Profile {pid} ({purpose}) worked",
                             round(a["worked"] / a["reports"], 3) if a.get("reports") else None, unit="ratio",
                             n=a.get("reports"), source=source,
                             detail={"worked": a.get("worked"), "failed": a.get("failed"),
                                     "median_tok_s": a.get("median_tok_s")}))
    return out


def from_saelens(registry: dict, key: str) -> list[dict]:
    want = {base_name(key), key.rsplit("/", 1)[-1].lower()}
    out = []
    for rel, info in sorted(registry.items()):
        m = str((info or {}).get("model") or "").lower()
        if not m or not (m in want or any(w.endswith(m) or m.endswith(w) for w in want if len(m) > 3)):
            continue
        saes = info.get("saes") or []
        out.append(_stat(f"saelens:{rel}", "features", f"SAELens: {rel}", len(saes), unit="dictionaries",
                         source="saelens", detail={"repo": info.get("repo_id"), "model": info.get("model"),
                                                   "first": [s.get("id") for s in saes[:3]]}))
    return out


def from_local(store, key: str) -> list[dict]:
    """This Studio's review store: models whose id names the same model."""
    want = base_name(key)
    out = []
    for m in store.models():
        mid = str(m["model"]).lower()
        if want not in re.sub(r"[-_.]q\d.*$", "", mid.rsplit("/", 1)[-1]) and want not in mid:
            continue
        f = store.facets(m["model"])
        out.append(_stat("observations", "activity", "Observed replies (activation profiles)", m["observations"],
                         source="local", variant=m["model"],
                         detail={"sources": {k: v["n"] for k, v in f["sources"].items()},
                                 "verdicts": f["verdicts"], "subjects": dict(list(f["subjects"].items())[:8]),
                                 "first": f["first"], "last": f["last"]}))
        bad = sum(v for k, v in f["verdicts"].items() if k in ("wrong", "hallucination", "bad"))
        if f["n"]:
            out.append(_stat("local_flagged", "quality", "Locally marked wrong", round(bad / f["n"], 4), unit="ratio",
                             n=f["n"], source="local", variant=m["model"]))
    return out


def lookup(model: str, boards: list[tuple[str, dict | Exception]] = (), saelens: dict | Exception | None = None,
           store=None) -> dict:
    """boards: [(source id, board.json or the error fetching it)]. saelens: the registry, an error, or None."""
    model = str(model).strip().strip("/")
    if not REPO_RE.match(model):
        raise ValueError("model: an org/name repository id")
    sources, stats = [], []
    for sid, b in boards:
        ok = not isinstance(b, Exception)
        sources.append({"id": sid, "kind": "board", "name": sid.split(":", 1)[-1], "ok": ok,
                        "error": None if ok else f"{type(b).__name__}: {b}"[:200]})
        if ok:
            got = from_board(b, model, sid)
            stats += got
            hwn = {str(s["detail"]["hardware"]).lower() for s in got if s["key"].startswith("gen_tok_s:")}
            stats += from_hw_reports(b, {n for h in hwn for n in h.split(", ")}, sid)
    if saelens is not None:
        ok = not isinstance(saelens, Exception)
        sources.append({"id": "saelens", "kind": "saelens", "name": "SAELens pretrained dictionaries", "ok": ok,
                        "error": None if ok else str(saelens)[:200]})
        if ok:
            stats += from_saelens(saelens, model)
    if store is not None:
        sources.append({"id": "local", "kind": "local", "name": "this Studio", "ok": True, "error": None})
        stats += from_local(store, model)
    for s in sources:
        s["stats"] = sum(1 for x in stats if x["source"] == s["id"])
    return {"schema": SCHEMA, "model": model, "aliases": sorted({model, base_name(model)}), "sources": sources,
            "stats": stats, "available": bool(stats)}


def board_url(spec: str) -> str:
    """github:owner/repo, owner/repo, or a URL (a scoreboard server, or a board.json) -> the JSON's URL."""
    spec = spec.strip()
    if spec.startswith(("http://", "https://")):
        return spec if spec.endswith(".json") else spec.rstrip("/") + "/api/board"
    spec = spec[7:] if spec.startswith("github:") else spec
    return f"https://raw.githubusercontent.com/{spec}/main/board.json"


def fetch_json(url: str, timeout: float = 20) -> dict:
    import urllib.request
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "OmniCortex"}), timeout=timeout) as r:
        return json.loads(r.read())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    lk = sub.add_parser("lookup", help="everything known about one model")
    lk.add_argument("model", help="org/name")
    lk.add_argument("--board", action="append", default=[], help="github:owner/repo, owner/repo or a URL (repeatable)")
    lk.add_argument("--board-file", action="append", default=[], help="a local board.json")
    lk.add_argument("--saelens", action="store_true", help="also check SAELens's catalogue (network)")
    lk.add_argument("--review-dir", help="a local review store")
    lk.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    sys.path.insert(0, str(HERE))
    boards = []
    for b in a.board:
        try:
            boards.append((f"board:{b}", fetch_json(board_url(b))))
        except Exception as e:
            boards.append((f"board:{b}", e))
    for f in a.board_file:
        boards.append((f"board:{Path(f).name}", json.loads(Path(f).read_text())))
    reg = None
    if a.saelens:
        import sae
        try:
            reg = sae.saelens_registry()
        except Exception as e:
            reg = e
    store = None
    if a.review_dir:
        import neuron_review
        store = neuron_review.ReviewStore(a.review_dir)
    rec = lookup(a.model, boards, reg, store)
    if a.json:
        print(json.dumps(rec, indent=1))
        return 0
    for s in rec["sources"]:
        print(f"{s['id']:30} {'ok' if s['ok'] else 'error: ' + str(s['error'])}  ({s['stats']} statistics)")
    for g in GROUPS:
        rows = [s for s in rec["stats"] if s["group"] == g]
        if rows:
            print(f"\n{g}")
            for s in rows:
                print(f"  {s['label'][:44]:44} {s['value']!s:>10} {s['unit']:12} {s['source']}"
                      + (f"  [{s['variant']}]" if s["variant"] else "") + ("  (related)" if s["related"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
