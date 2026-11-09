"""
Studio's side of the Features page: what dictionaries, emotion vectors and attribution graphs exist in the
workspace, their contents, and feature labels people edit. Pure file reading (no torch); the model work
happens in jobs and in the decoding lab (scripts/feature_lab.py), which Studio proxies to.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

SUBDIRS = ("runs/features", "runs/emotions", "runs/graphs")
_CACHE: dict = {}


def roots(ws_root: str, extra: list[str] | None = None) -> list[Path]:
    out = [Path(ws_root) / s for s in SUBDIRS] + [Path(os.path.expanduser(x)) for x in (extra or []) if x]
    return [p for p in out if p.is_dir()]


def allowed(path: str, ws_root: str, extra: list[str] | None = None) -> Path:
    """The path, if it lies inside the workspace's runs/ (or an added folder); else ValueError."""
    # lexical (abspath folds ".."), so folders linked into runs/ from another disk still count
    p = Path(os.path.abspath(os.path.expanduser(str(path or ""))))
    for r in [Path(ws_root) / "runs"] + [Path(os.path.expanduser(x)) for x in (extra or []) if x]:
        try:
            p.relative_to(os.path.abspath(r))
            return p
        except ValueError:
            continue
    raise ValueError(f"{path}: not inside the workspace's runs/ folder")


def _json(p: Path):
    key = (str(p), p.stat().st_mtime)
    if key not in _CACHE:
        if len(_CACHE) > 64:
            _CACHE.clear()
        _CACHE[key] = json.loads(p.read_text())
    return _CACHE[key]


def _walk(base: Path, name: str, depth: int = 3):
    if not base.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(base, followlinks=True):
        rel = Path(dirpath).relative_to(base)
        if len(rel.parts) >= depth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if d not in ("examples", "__pycache__") and not d.startswith(".")]
        if name in filenames:
            yield Path(dirpath)


def list_all(ws_root: str, extra: list[str] | None = None) -> dict:
    out = {"dicts": [], "acts": [], "emotions": [], "stories": [], "graphs": [],
           "root": str(Path(ws_root) / "runs")}
    for base in [Path(ws_root) / "runs"] + [Path(os.path.expanduser(x)) for x in (extra or []) if x]:
        for d in _walk(base, "sae.json"):
            try:
                m = _json(d / "sae.json")
            except (ValueError, OSError):
                continue
            ev, sp = m.get("eval") or {}, m.get("splice") or {}
            out["dicts"].append({
                "path": str(d), "name": d.name, "model": m.get("model"), "site": m.get("site"), "layer": m.get("layer"),
                "target_site": m.get("target_site"), "arch": m.get("arch"), "n_features": m.get("n_features"),
                "fvu": ev.get("fvu"), "l0": ev.get("l0"), "delta_loss": sp.get("delta"),
                "recovered": sp.get("recovered"), "interpreted": (d / "features.json").exists(),
                "neurons": (d / "neurons.json").exists(), "imported": bool(m.get("imported_from")),
                "created": m.get("created")})
        for d in _walk(base, "meta.json"):
            try:
                m = _json(d / "meta.json")
            except (ValueError, OSError):
                continue
            if m.get("format") == "omnicortex-acts/1":
                out["acts"].append({"path": str(d), "name": d.name, "model": m.get("model"), "points": m.get("points"),
                                    "tokens": m.get("tokens"), "created": m.get("created")})
        for d in _walk(base, "emotions.json"):
            try:
                m = _json(d / "emotions.json")
            except (ValueError, OSError):
                continue
            acc = m.get("accuracy") or {}
            out["emotions"].append({"path": str(d), "name": d.name, "model": m.get("model"),
                                    "n": len(m.get("names") or []), "default_layer": m.get("default_layer"),
                                    "best_layer": m.get("best_layer"),
                                    "best_accuracy": max(acc.values()) if acc else None, "created": m.get("created")})
        for d in _walk(base, "stories.jsonl"):
            try:
                n = sum(1 for _ in (d / "stories.jsonl").open())
            except OSError:
                n = 0
            out["stories"].append({"path": str(d), "name": d.name, "stories": n,
                                   "neutral": (d / "neutral.jsonl").exists()})
        g = base / "graphs"
        for p in sorted(g.glob("*.json")) if g.is_dir() else []:
            try:
                st = p.stat()
                if st.st_size > 50_000_000:
                    continue
                g_ = _json(p)
            except (ValueError, OSError):
                continue
            if not isinstance(g_, dict) or "nodes" not in g_:
                continue
            out["graphs"].append({"path": str(p), "name": p.stem, "prompt": g_.get("prompt"),
                                  "kind": "attribution" if "edges" in g_ else "features",
                                  "created": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))})
    for k in ("dicts", "acts", "emotions", "stories", "graphs"):
        out[k].sort(key=lambda r: r.get("created") or "", reverse=True)
    return out


def labels_of(d: Path) -> dict[int, str]:
    lf = d / "labels.json"
    if lf.exists():
        try:
            return {int(k): v for k, v in json.loads(lf.read_text()).items()}
        except (ValueError, OSError):
            pass
    return {}


def dictionary(d: Path) -> dict:
    """Columns for the browser: everything needed to list, sort, search and map the features."""
    info = _json(d / "sae.json")
    out = {"path": str(d), "name": d.name, "info": {k: v for k, v in info.items() if k not in ("train", "source_cfg")},
           "history": (info.get("train") or {}).get("history", [])[-200:], "interpreted": (d / "features.json").exists()}
    if not out["interpreted"]:
        return out
    fj = _json(d / "features.json")
    user = labels_of(d)
    feats = fj.get("features", [])
    out.update({"n": len(feats), "tokens": fj.get("tokens"), "alive": fj.get("alive"),
                "density_hist": fj.get("density_hist"), "density_edges": fj.get("density_edges"),
                "id": [f["id"] for f in feats], "freq": [round(f["freq"], 7) for f in feats],
                "max": [round(f["max"], 3) for f in feats], "xy": [f.get("xy") for f in feats],
                "label": [user.get(f["id"], f.get("label", "")) for f in feats],
                "score": [f.get("label_score") for f in feats]})
    return out


def _decode(vocab: dict, ids: list[int]) -> list[str]:
    return [vocab.get(str(t), f"<{t}>") for t in ids]


def feature(d: Path, fid: int) -> dict:
    fj = _json(d / "features.json")
    feats = fj.get("features", [])
    if not 0 <= fid < len(feats):
        raise ValueError(f"no feature {fid}")
    vocab = _json(d / "vocab.json") if (d / "vocab.json").exists() else {}
    f = dict(feats[fid])
    user = labels_of(d)
    if fid in user:
        f["label"], f["label_by"] = user[fid], "you"
    per = int(fj.get("per_file") or 256)
    ex = {}
    ep = d / "examples" / f"{fid // per:05d}.json"
    if ep.exists():
        ex = _json(ep).get(str(fid)) or {}

    def dec(x):
        return {"tok": _decode(vocab, x["tok"]), "act": x["act"], "at": x["at"]}
    examples = {"top": [dec(x) for x in ex.get("top", [])]}
    if ex.get("buckets"):
        examples["buckets"] = [{"lo": b["lo"], "hi": b["hi"], "ex": [dec(x) for x in b["ex"]]} for b in ex["buckets"]]
    if ex.get("random"):
        examples["random"] = [dec(x) for x in ex["random"]]
    f["examples"] = examples
    for k in ("up", "down"):
        if f.get(k):
            f[k] = [[vocab.get(str(t), f"<{t}>"), v] for t, v in f[k]]
    labels = {e["id"]: user.get(e["id"], e.get("label", "")) for e in feats}
    f["nb"] = [[i, c, labels.get(i, "")] for i, c in f.get("nb", [])]
    f["act_bins"] = fj.get("act_bins")
    nj = d / "neurons.json"
    if nj.exists():
        n = _json(nj)
        f["neurons"] = (n.get("features") or {}).get(str(fid))
        f["neuron_method"] = n.get("method")
    return f


LABEL_RX = re.compile(r"^[^\x00-\x1f]{0,120}$")


def set_label(d: Path, fid: int, label: str) -> dict:
    label = str(label or "").strip()
    if not LABEL_RX.match(label):
        raise ValueError("a label is one line of at most 120 characters")
    lab = labels_of(d)
    if label:
        lab[int(fid)] = label
    else:
        lab.pop(int(fid), None)
    tmp = d / "labels.json.tmp"
    tmp.write_text(json.dumps({str(k): v for k, v in sorted(lab.items())}, indent=1, ensure_ascii=False))
    tmp.replace(d / "labels.json")
    return {"ok": True, "id": int(fid), "label": label}


def emotions(d: Path) -> dict:
    return {"path": str(d), "name": d.name, **_json(d / "emotions.json")}


def graph(p: Path) -> dict:
    return {"path": str(p), **_json(p)}


def search(d: Path, q: str, limit: int = 50) -> list[dict]:
    """Features whose label, top tokens or logit tokens contain q (or the id itself)."""
    q = (q or "").strip().lower()
    fj = _json(d / "features.json")
    vocab = _json(d / "vocab.json") if (d / "vocab.json").exists() else {}
    user = labels_of(d)
    hits = []
    for f in fj.get("features", []):
        lab = user.get(f["id"], f.get("label", ""))
        toks = " ".join(vocab.get(str(t), "") for t, _ in (f.get("up") or [])[:10]).lower()
        if q == str(f["id"]) or (q and (q in lab.lower() or q in toks)):
            hits.append({"id": f["id"], "label": lab, "freq": f["freq"], "max": f["max"]})
            if len(hits) >= limit:
                break
    return hits
