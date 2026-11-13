#!/usr/bin/env python3
"""
Score cards: share TestQA results with a scoreboard, opt-in, and only what is
listed below.

Nothing is sent unless you turn sharing on and give a scoreboard address
(Studio → Settings → Sharing results, or --endpoint here). OmniCortex
does not run a public scoreboard itself. scripts/scoreboard_server.py is
a reference server that anyone can host.

## What a submission contains

**The card**, which may be shown publicly:
- the model's identity:
  - the SHA-256 and size of the GGUF file, its file name (not its path),
    the quantization;
  - its Hugging Face repo when the file is verified to come from one;
- what makes this run different from another run of the same file, under
  `variant`:
  - the hash of the H-Neuron set and the suppression scale;
  - the LoRA file's hash and scale;
  - fine-tune, merge or concept-edit lineage recorded next to the model;
  - the engine (llama.cpp / vLLM / remote) and its version;
  - the engine options that can change answers (context length, KV cache
    types, speculative decoding, flash attention);
  - the sampling settings;
- coarse hardware: OS family, CPU model, core count, RAM and GPU memory
  rounded to whole GB, GPU names;
- the test: the TestQA bank's fingerprint, and the counts per subject
  (right, wrong, declined).

**The verification log**, private: for each test item, its id, the verdict,
and the model's reply. A scoreboard uses it only to check that the card is
true, by grading the replies again and comparing. It must never be
published (the reference server keeps it in a separate table that no public
endpoint reads).

**Not sent**: your chats, prompts you typed, file paths, user or host names,
tokens, IP addresses (beyond what any web request reveals to the server),
anything about other models.

## Public or private

A card is public only for a model that can be found by model search:
- its file's SHA-256 matches a file in the Hugging Face repo it was
  downloaded from;
- or you list an unlisted model by hand (`--public`, or the per-model switch
  in Studio).

Everything else is submitted privately, or not at all:
- Studio submits automatically for unlisted models only when *Also submit
  results for unlisted models* is on (Advanced);
- even then, those cards stay private unless listed by hand.

Every submission is saved in ~/.omnicortex/submissions/ exactly as sent.

    python scripts/score_cards.py preview --results runs/base.json --label m --model models/x/y.gguf
    python scripts/score_cards.py check --model models/org/repo/file.gguf
    python scripts/score_cards.py submit --endpoint https://board.example --results ... --model ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CARD_VERSION = 1
HOME = Path(os.environ.get("OC_HOME") or "~/.omnicortex").expanduser()
# Engine options that can change what a model answers (speed-only ones are left out).
OUTPUT_OPTIONS = ("ctx", "cache_type_k", "cache_type_v", "cache_type", "flash_attn", "spec_type", "draft_max",
                  "draft_p_min", "experts", "vision", "kv_cache_dtype", "quantization", "dtype", "max_model_len")


def sha256_file(path: str | Path, cache: bool = True) -> str:
    """SHA-256 of a (large) file, cached by path, size and mtime."""
    p = Path(path)
    st = p.stat()
    key = f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    cpath = HOME / "hash-cache.json"
    store = {}
    if cache:
        try:
            store = json.loads(cpath.read_text())
        except (OSError, ValueError):
            store = {}
        if key in store:
            return store[key]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    d = h.hexdigest()
    if cache:
        store[key] = d
        try:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            cpath.write_text(json.dumps(dict(list(store.items())[-5000:])))
        except OSError:
            pass
    return d


def bank_fingerprint(tasks: list[dict]) -> str:
    h = hashlib.sha256()
    for t in sorted(tasks, key=lambda t: t["id"]):
        h.update(json.dumps([t["id"], t.get("prompt"), t.get("answer"), t.get("aliases"), t.get("tests"),
                             t.get("checks")], sort_keys=True).encode())
    return h.hexdigest()[:16]


def _short_hash(path) -> str | None:
    if not path:
        return None
    p = Path(str(path)).expanduser()
    return sha256_file(p)[:16] if p.is_file() else None


def lineage(model_path: str) -> list[dict]:
    """Records OmniCortex writes next to models it produced: fine-tunes, exports, concept edits."""
    out = []
    d = Path(model_path).parent
    for name in ("omnicortex-export.json", "omnicortex-finetune.json", "edit.json", "sage.json"):
        for p in (d / name, d.parent / name):
            if p.is_file():
                try:
                    j = json.loads(p.read_text())
                except ValueError:
                    continue
                keep = {k: j.get(k) for k in ("method", "lora", "layers", "dpo", "vision", "norm_preserving",
                                              "components", "k", "protected_dims") if k in j}
                if "chosen" in j:
                    keep["kl"] = j["chosen"].get("kl")
                out.append({"kind": name.split(".")[0], "sha": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
                            **keep})
                break
    return out


def engine_version(binary: str | None) -> str | None:
    if not binary or not Path(binary).exists():
        return None
    try:
        r = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=20)
        m = re.search(r"version:\s*(\S+)(?:\s*\((\w+)\))?", r.stdout + r.stderr)
        return f"{m.group(1)}{'+' + m.group(2) if m.group(2) else ''}" if m else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def variant(settings: dict, engine_bin: str | None = None, sampling: dict | None = None,
            model_path: str | None = None) -> dict:
    s = settings or {}
    v = {"engine": {"name": s.get("engine") or "llama", "version": engine_version(engine_bin),
                    "options": {k: s[k] for k in OUTPUT_OPTIONS if s.get(k) not in (None, "", False)}},
         "sampling": sampling or {"temperature": 0.0},
         "suppression": None, "lora": None, "lineage": lineage(model_path) if model_path else []}
    if s.get("classifier"):
        v["suppression"] = {"h_neurons": _short_hash(s["classifier"])}
    if s.get("lora"):
        v["lora"] = {"sha": _short_hash(s["lora"]), "scale": s.get("lora_scale", 1.0)}
    if s.get("draft_model"):
        v["engine"]["options"]["draft_model"] = _short_hash(s["draft_model"])
    return v


def variant_id(v: dict) -> str:
    keep = {k: v[k] for k in ("engine", "sampling", "suppression", "lora", "lineage")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]


def hardware() -> dict:
    hw = {"os": platform.system(), "machine": platform.machine(), "cpus": os.cpu_count()}
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                hw["cpu"] = line.split(":", 1)[1].strip()
                break
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                hw["ram_gb"] = round(int(line.split()[1]) / 2**20)
                break
    except OSError:
        pass
    try:
        sys.path.insert(0, str(HERE))
        import accelerators
        hw["gpus"] = [{"name": d.get("name"), "backend": d.get("backend"),
                       "vram_gb": round((d.get("memory_total") or 0) / 2**30), "unified": bool(d.get("unified"))}
                      for d in accelerators.detect() if d.get("backend") != "cpu"][:8]
    except Exception:
        hw["gpus"] = []
    return hw


def hf_check(model_path: str, repo: str | None = None, timeout: float = 15) -> dict:
    """Is this exact file in a Hugging Face repo? The repo is given, or read from the
    download path (<models>/<org>/<repo>/<file>). -> {listed, repo, file, reason}."""
    p = Path(model_path)
    repo = repo or f"{p.parent.parent.name}/{p.parent.name}"
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        return {"listed": False, "repo": None, "reason": "no Hugging Face repo known for this file"}
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=timeout) as r:
            info = json.loads(r.read())
    except Exception as e:
        return {"listed": False, "repo": repo, "reason": f"could not check {repo}: {type(e).__name__}"}
    sha = sha256_file(p)
    for s in info.get("siblings") or []:
        lfs = s.get("lfs") or {}
        if lfs.get("sha256") == sha:
            return {"listed": True, "repo": repo, "file": s.get("rfilename"), "reason": "file found in the repo"}
    return {"listed": False, "repo": repo, "reason": "the repo has no file with this SHA-256 (changed or renamed?)"}


def build(results_path: str, label: str | None, model_path: str, settings: dict | None = None,
          engine_bin: str | None = None, cache_dir: str | None = None, tasks_paths=None,
          listed: dict | None = None, public_by_hand: bool = False, extras: dict | None = None) -> tuple[dict, dict]:
    """-> (card, verification log)."""
    sys.path.insert(0, str(HERE))
    import testqa as tq
    d = json.loads(Path(results_path).read_text())
    raw = d["raw"]
    label = label or d.get("reference") or next(iter(raw))
    rows = raw[label]
    tasks = tq.load_tasks(tasks_paths or [str(ROOT / "qa" / "bank")], None, None, 0)
    ids = {r["id"] for r in rows}
    asked = [t for t in tasks if t["id"] in ids]
    per = {}
    for r in rows:
        if r.get("kind") == "canary" or r["verdict"] in ("error", "skipped"):
            continue
        c = per.setdefault(r.get("subject") or "?", {"n": 0, "correct": 0, "wrong": 0, "abstained": 0, "other": 0})
        c["n"] += 1
        key = "correct" if r["verdict"] in ("correct", "answered") else r["verdict"] \
            if r["verdict"] in ("wrong", "abstained") else "other"
        c[key] += 1
    n = sum(c["n"] for c in per.values())
    tot = {k: sum(c[k] for c in per.values()) for k in ("correct", "wrong", "abstained")}
    mp = Path(model_path)
    ep = (d.get("endpoints") or {}).get(label) or {}
    v = variant(settings or {}, engine_bin, {"temperature": ep.get("temperature", 0.0)}, model_path)
    lst = listed or {"listed": False}
    card = {"card_version": CARD_VERSION, "created": int(time.time()),
            "model": {"sha256": sha256_file(mp), "size": mp.stat().st_size, "file": mp.name,
                      "quant": _quant(mp.name), "hf_repo": lst.get("repo") if lst.get("listed") else None,
                      "listed": bool(lst.get("listed"))},
            "variant": v, "variant_id": variant_id(v), "hardware": hardware(),
            "test": {"bank": bank_fingerprint(asked), "items": n, "per_subject": per,
                     "accuracy": round(tot["correct"] / n, 4) if n else None,
                     "hallucination_rate": round(tot["wrong"] / n, 4) if n else None,
                     "abstention_rate": round(tot["abstained"] / n, 4) if n else None},
            "visibility": "public" if (lst.get("listed") or public_by_hand) else "private",
            "omnicortex": _version()}
    perf = performance(d, label, settings or {})
    if perf:
        card["performance"] = perf
    if extras:
        import unified_review
        ex = unified_review.sanitize_extras(extras)
        if ex:
            card["extras"] = ex                 # self-reported: the board cannot re-check these
    cache = {}
    if cache_dir:
        cp = Path(cache_dir) / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', label)}.jsonl"
        if cp.exists():
            for line in cp.read_text(encoding="utf-8").splitlines():
                j = json.loads(line)
                cache[j["id"]] = j["text"]
    log = {"items": [{"id": r["id"], "verdict": r["verdict"], "reply": cache.get(r["id"], r.get("text", ""))}
                     for r in rows if r.get("kind") != "canary"]}
    return card, log


def performance(results: dict, label: str, settings: dict) -> list[dict]:
    """Speed measured during the run (testqa's scheduling summary), with the settings that decide it, for
    hardware performance prediction (perf_predict.py). Only what the server measured; nothing estimated."""
    ep = (results.get("endpoints") or {}).get(label) or {}
    url = ep.get("url")
    out = []
    for s in results.get("scheduling") or []:
        if s.get("server") != url or not (s.get("gen_tok_s") or s.get("avg_s")):
            continue
        out.append({"source": "testqa", "gen_tok_s": s.get("gen_tok_s"), "avg_s": s.get("avg_s"),
                    "parallel": s.get("limit"), "requests": s.get("done"),
                    **{k: settings[k] for k in ("ngl", "ctx", "device", "split_mode", "tensor_split", "cache_type_k",
                                                "flash_attn", "spec_type", "draft_model") if settings.get(k) not in (None, "")}})
        if out[-1].get("draft_model"):
            out[-1]["draft_model"] = Path(str(out[-1]["draft_model"])).name
    return out


def _quant(name: str) -> str | None:
    m = re.search(r"(I?Q\d_[A-Z0-9_]+|Q\d_\d|F16|BF16|F32|TQ\d_\d)", name.upper())
    return m.group(1) if m else None


def _version() -> str:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"], capture_output=True,
                           text=True, timeout=5)
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def github_token() -> str | None:
    """For a scoreboard on GitHub: OC_GITHUB_TOKEN, or the token saved in Settings → Sharing results."""
    t = os.environ.get("OC_GITHUB_TOKEN")
    if t:
        return t.strip()
    p = HOME / "github.token"
    return p.read_text().strip() if p.exists() else None


def submit(endpoint: str, card: dict, log: dict, timeout: float = 60) -> dict:
    """To a scoreboard server (https://…) or a scoreboard on GitHub (github:owner/repo)."""
    if endpoint.startswith("github:"):
        import github_board
        out = github_board.submit(endpoint[7:], card, log, github_token())
    else:
        body = json.dumps({"card": card, "log": log}).encode()
        req = urllib.request.Request(endpoint.rstrip("/") + "/api/submit", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
    d = HOME / "submissions"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{time.strftime('%Y%m%d-%H%M%S')}-{card['variant_id']}.json").write_text(
        json.dumps({"endpoint": endpoint, "sent": {"card": card, "log": log}, "answer": out}, indent=1))
    return out


def submissions(limit: int = 50) -> list[dict]:
    d = HOME / "submissions"
    out = []
    for p in sorted(d.glob("*.json"), reverse=True)[:limit] if d.is_dir() else []:
        try:
            j = json.loads(p.read_text())
            out.append({"file": p.name, "endpoint": j["endpoint"], "model": j["sent"]["card"]["model"]["file"],
                        "visibility": j["sent"]["card"]["visibility"], "answer": j.get("answer")})
        except (OSError, ValueError, KeyError):
            continue
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("preview", "submit"):
        s = sub.add_parser(name)
        s.add_argument("--results", required=True)
        s.add_argument("--label")
        s.add_argument("--model", required=True, help="the GGUF the results came from")
        s.add_argument("--settings", help="JSON of the settings it was served with (Studio's per-model settings)")
        s.add_argument("--server", help="llama-server, for its version")
        s.add_argument("--cache", help="testqa.py --cache folder, for the full replies")
        s.add_argument("--repo", help="Hugging Face repo to verify the file against")
        s.add_argument("--public", action="store_true", help="list this model publicly although it is unlisted")
        s.add_argument("--offline", action="store_true", help="do not check Hugging Face")
        s.add_argument("--extras", help="JSON of self-reported summaries to add (emotions, features; "
                                        "docs/UNIFIED_REVIEW.md); the board shows them as not re-checked")
        if name == "submit":
            s.add_argument("--endpoint", required=True)
    c = sub.add_parser("check", help="is this file the one in its Hugging Face repo?")
    c.add_argument("--model", required=True)
    c.add_argument("--repo")
    sub.add_parser("history", help="what was sent so far")
    a = p.parse_args(argv)
    if a.cmd == "check":
        print(json.dumps(hf_check(a.model, a.repo), indent=1))
        return 0
    if a.cmd == "history":
        print(json.dumps(submissions(), indent=1))
        return 0
    settings = json.loads(Path(a.settings).read_text()) if a.settings else {}
    listed = {"listed": False} if a.offline else hf_check(a.model, a.repo)
    card, log = build(a.results, a.label, a.model, settings, a.server, a.cache, listed=listed,
                      public_by_hand=a.public,
                      extras=json.loads(Path(a.extras).read_text()) if a.extras else None)
    if a.cmd == "preview":
        print(json.dumps({"card": card, "log": {"items": log["items"][:3], "total_items": len(log["items"]),
                                                "note": "private: used only to verify the card"}}, indent=1))
        return 0
    print(json.dumps(submit(a.endpoint, card, log), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
