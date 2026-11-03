#!/usr/bin/env python3
"""
Where recorded activations go (docs/ACTIVATIONS.md). Every reply's activation profile (float16 [layers, bins])
is collected by default and written through a Sink:

    local    the review store on this computer (~/.neuronscope/review, or --review-dir)
    folder   a review store somewhere else: another disk, a network share, a mounted bucket
    studio   another OmniCortex Studio, over HTTPS with its token (POST /api/review/ingest)
    memory   kept in memory only, for a host with little or no persistent storage; shown, exported or
             forwarded, and lost on restart

A folder or Studio that cannot be reached does not lose anything at once: observations wait in a bounded
in-memory spool (and, unless the host has no persistent storage, a spool file) and are delivered when it
answers again. max_gb caps the local and folder stores: the oldest observations are removed first.
"""
from __future__ import annotations

import base64
import collections
import io
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

MODES = ("local", "folder", "studio", "memory")
DEFAULTS = {"collect": True, "every": 1, "store": "local", "folder": "", "studio_url": "", "studio_token": "",
            "max_gb": 20.0, "spool": 2000, "spool_file": True}


def encode_vec(vec) -> str:
    buf = io.BytesIO()
    np.save(buf, np.asarray(vec, dtype=np.float16), allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode()


def decode_vec(s: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(s)), allow_pickle=False)


class Sink:
    def __init__(self, cfg: dict | None = None, local_root=None, home=None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        if self.cfg["store"] not in MODES:
            raise ValueError(f"store: one of {', '.join(MODES)}")
        self.local_root = local_root
        self.home = Path(home or Path.home() / ".neuronscope").expanduser()
        self.spool: collections.deque = collections.deque(maxlen=int(self.cfg["spool"]))
        self.memory: collections.deque = collections.deque(maxlen=int(self.cfg["spool"]))
        self.lock = threading.Lock()
        self.stats = {"recorded": 0, "delivered": 0, "spooled": 0, "dropped": 0, "pruned": 0, "last_error": None}
        self._store = None
        self._load_spool()

    # -- the stores
    def store(self):
        import neuron_review
        if self._store is None:
            root = self.cfg["folder"] if self.cfg["store"] == "folder" else self.local_root
            if self.cfg["store"] == "folder" and not root:
                raise ValueError("store 'folder' needs a folder")
            self._store = neuron_review.ReviewStore(root)
        return self._store

    def _spool_path(self) -> Path:
        return self.home / "activation-spool.jsonl"

    def _load_spool(self):
        p = self._spool_path()
        if self.cfg["spool_file"] and p.exists():
            for line in p.read_text().splitlines():
                try:
                    self.spool.append(json.loads(line))
                except ValueError:
                    continue

    def _save_spool(self):
        if not self.cfg["spool_file"] or self.cfg["store"] == "memory":
            return
        p = self._spool_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("".join(json.dumps(x) + "\n" for x in self.spool))
        except OSError:
            pass                             # no persistent storage: the in-memory spool still holds them

    # -- recording
    def record(self, model: str, vec, **meta) -> str:
        """Deliver one observation (or keep it for later). -> where it went."""
        item = {"model": model, "vec": encode_vec(vec), "meta": {**meta, "t": meta.get("t") or time.time()}}
        with self.lock:
            self.stats["recorded"] += 1
            if self.cfg["store"] == "memory":
                if len(self.memory) == self.memory.maxlen:
                    self.stats["dropped"] += 1
                self.memory.append(item)
                return "memory"
            try:
                self._deliver(item)
                self.stats["delivered"] += 1
                self._flush()
                return self.cfg["store"]
            except Exception as e:
                self.stats["last_error"] = f"{type(e).__name__}: {e}"[:300]
                if len(self.spool) == self.spool.maxlen:
                    self.stats["dropped"] += 1
                self.spool.append(item)
                self.stats["spooled"] += 1
                self._save_spool()
                return "spool"

    def _deliver(self, item):
        if self.cfg["store"] == "studio":
            url = self.cfg["studio_url"].rstrip("/") + "/api/review/ingest"
            req = urllib.request.Request(url, data=json.dumps({"items": [item]}).encode(), method="POST",
                                         headers={"Content-Type": "application/json",
                                                  **({"Authorization": f"Bearer {self.cfg['studio_token']}"}
                                                     if self.cfg["studio_token"] else {})})
            with urllib.request.urlopen(req, timeout=20) as r:
                r.read()
            return
        st = self.store()
        m = dict(item["meta"])
        st.record(item["model"], decode_vec(item["vec"]), source=m.pop("source", "live"), kind=m.pop("kind", "observed"),
                  subjects=m.pop("subjects", None), verdict=m.pop("verdict", "unknown"), risk=m.pop("risk", None),
                  t=m.pop("t", None), item=m.pop("item", None), extra=m or None)
        self.prune()

    def _flush(self):
        """Deliver what waited, oldest first, until one fails."""
        n = 0
        while self.spool:
            try:
                self._deliver(self.spool[0])
            except Exception:
                break
            self.spool.popleft()
            self.stats["delivered"] += 1
            n += 1
        if n:
            self._save_spool()

    def retry(self) -> int:
        with self.lock:
            before = len(self.spool)
            self._flush()
            return before - len(self.spool)

    # -- size cap
    def prune(self) -> int:
        """Keep the store under max_gb by removing the oldest observations' vectors and index lines."""
        cap = float(self.cfg.get("max_gb") or 0) * 2**30
        if not cap or self.cfg["store"] not in ("local", "folder"):
            return 0
        root = Path(self.store().root)
        files = sorted(root.glob("*/vecs/*.npy"), key=lambda p: p.stat().st_mtime) if root.is_dir() else []
        total = sum(p.stat().st_size for p in files)
        removed = 0
        while files and total > cap:
            p = files.pop(0)
            total -= p.stat().st_size
            p.unlink(missing_ok=True)
            removed += 1
        if removed:
            for idx in root.glob("*/index.jsonl"):
                vec_dir = idx.parent / "vecs"
                keep = [line for line in idx.read_text().splitlines()
                        if line.strip() and ('"op": "label"' in line or (vec_dir / f"{json.loads(line)['id']}.npy").exists())]
                idx.write_text("".join(x + "\n" for x in keep))
            self.stats["pruned"] += removed
        return removed

    def status(self) -> dict:
        out = {"store": self.cfg["store"], "collect": self.cfg["collect"], "spooled_now": len(self.spool),
               "in_memory": len(self.memory), **self.stats}
        if self.cfg["store"] in ("local", "folder"):
            try:
                root = Path(self.store().root)
                out["root"] = str(root)
                out["bytes"] = sum(p.stat().st_size for p in root.glob("*/vecs/*.npy")) if root.is_dir() else 0
            except (OSError, ValueError) as e:
                out["root_error"] = str(e)
        return out

    def export_memory(self) -> list[dict]:
        with self.lock:
            return list(self.memory)


def ingest(store, items: list[dict], limit: int = 500) -> int:
    """Receiving end of store 'studio': items from another Studio's Sink into this review store."""
    n = 0
    for it in items[:limit]:
        m = dict(it.get("meta") or {})
        store.record(str(it["model"]), decode_vec(it["vec"]), source=str(m.pop("source", "remote")),
                     kind=m.pop("kind", "observed"), subjects=m.pop("subjects", None), verdict=m.pop("verdict", "unknown"),
                     risk=m.pop("risk", None), t=m.pop("t", None), item=m.pop("item", None),
                     extra={**{k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))}, "forwarded": True})
        n += 1
    return n


def disk_free(path) -> float | None:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize / 2**30
    except (OSError, AttributeError):
        return None
