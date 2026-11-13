#!/usr/bin/env python3
"""
OmniCortex recording format.

The interface that makes this a toolkit rather than two disconnected apps. The
web dashboard and the Fastplotlib explorer both read these; so can anything you
write later. Nothing here imports torch, llama.cpp or a GUI.

Layout -- a directory, not a single file, so it is append-only, resumable, and
survives a crash mid-run:

    session/
      manifest.json          model, quant, dims, profile, alpha, timestamps
      samples/<qid>.npz      per-sample arrays
      events.jsonl           append-only log: sweeps, verdicts, notes

Deliberately not HDF5 or a database. A directory of .npz opens in numpy on any
platform with no driver, no server and no version pinning, and you can rsync
half of one off a remote box while it is still being written.

Per-sample arrays:
    agg      [n_layers, n_neurons] float16  CETT aggregated over a token region
    tokens   [n_tokens] int32               ids, optional
    scores   [n_tokens] float32             per-token classifier score, optional

Sessions are keyed by model fingerprint. Records from different models, or the
same model at different quantizations, are not comparable: neuron indices are a
property of specific weights. Mixing them silently is the single easiest way to
produce a convincing wrong answer, so `merge_check` refuses.
"""

import json
import os
import time

import numpy as np

FORMAT_VERSION = 1


def _sanitize(qid):
    """Filesystem-safe id. Windows also forbids <>:"/\\|?* and trailing dots."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(qid))
    return (safe.rstrip(". ") or "unnamed")[:120]


class Recorder:
    """Writes a session. Safe to reopen and append to an existing one."""

    def __init__(self, path, meta=None, resume=True):
        self.path = os.path.abspath(path)
        self.samples_dir = os.path.join(self.path, "samples")
        os.makedirs(self.samples_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.path, "manifest.json")
        self.events_path = os.path.join(self.path, "events.jsonl")

        if resume and os.path.exists(self.manifest_path):
            with open(self.manifest_path) as f:
                self.meta = json.load(f)
            if meta:
                incoming = dict(meta)
                for key in ("fingerprint", "n_layers", "n_neurons"):
                    old, new = self.meta.get(key), incoming.get(key)
                    if old is not None and new is not None and old != new:
                        raise ValueError(
                            f"session {self.path} was recorded with {key}={old}, "
                            f"but this run has {key}={new}. Records from "
                            "different models or quantizations are not "
                            "comparable; start a new session.")
                self.meta.update(incoming)
        else:
            self.meta = dict(meta or {})
            self.meta.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.meta["format_version"] = FORMAT_VERSION
        self.meta["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._flush_manifest()

    def _flush_manifest(self):
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.meta, f, indent=2)
        os.replace(tmp, self.manifest_path)   # atomic: no half-written manifest

    def add(self, qid, agg=None, tokens=None, scores=None, **fields):
        """Record one sample. `agg` is [n_layers, n_neurons]."""
        arrays = {}
        if agg is not None:
            agg = np.asarray(agg)
            if agg.ndim != 2:
                raise ValueError(f"agg must be 2D [layers, neurons], got {agg.shape}")
            nl, nn = self.meta.get("n_layers"), self.meta.get("n_neurons")
            if nl and nn:
                rows, cols = agg.shape
                # [layers, neurons] normally; a trace stacks frames, so
                # [frames * layers, neurons] is also valid. Requiring a whole
                # multiple still catches a genuinely wrong shape.
                if cols != nn or rows % nl:
                    raise ValueError(
                        f"agg is {agg.shape}; the session declares "
                        f"{nn} neurons and {nl} layers, so rows must be a "
                        f"multiple of {nl}")
            arrays["agg"] = agg.astype(np.float16)
        if tokens is not None:
            arrays["tokens"] = np.asarray(tokens, dtype=np.int32)
        if scores is not None:
            arrays["scores"] = np.asarray(scores, dtype=np.float32)
        if fields:
            arrays["fields"] = np.frombuffer(
                json.dumps(fields).encode(), dtype=np.uint8)

        path = os.path.join(self.samples_dir, _sanitize(qid) + ".npz")
        tmp = path + ".tmp"
        # Write through a handle: savez_compressed appends ".npz" to a path
        # that lacks it, which would defeat the atomic rename.
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)
        return path

    def event(self, kind, **payload):
        """Append to the log. Sweep points, verdicts, notes -- anything you
        want to be able to reconstruct afterwards."""
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind}
        rec.update(payload)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def close(self):
        self._flush_manifest()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class Session:
    """Reads a session. Lazy: arrays load per sample, not all at once."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        mp = os.path.join(self.path, "manifest.json")
        if not os.path.exists(mp):
            raise FileNotFoundError(f"no manifest.json in {self.path}")
        with open(mp) as f:
            self.meta = json.load(f)
        v = self.meta.get("format_version")
        if v != FORMAT_VERSION:
            raise ValueError(f"format v{v}, this reader expects v{FORMAT_VERSION}")
        self.samples_dir = os.path.join(self.path, "samples")

    @property
    def ids(self):
        if not os.path.isdir(self.samples_dir):
            return []
        return sorted(f[:-4] for f in os.listdir(self.samples_dir)
                      if f.endswith(".npz"))

    def __len__(self):
        return len(self.ids)

    def get(self, qid):
        path = os.path.join(self.samples_dir, _sanitize(qid) + ".npz")
        with np.load(path, allow_pickle=False) as z:
            out = {k: z[k] for k in z.files if k != "fields"}
            if "fields" in z.files:
                out["fields"] = json.loads(bytes(z["fields"]).decode())
        return out

    def events(self, kind=None):
        p = os.path.join(self.path, "events.jsonl")
        if not os.path.exists(p):
            return []
        out = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue      # tolerate a torn final line from a crash
                if kind is None or e.get("kind") == kind:
                    out.append(e)
        return out

    def stack(self, ids=None, key="agg", max_samples=None):
        """[n_samples, n_layers, n_neurons] for the explorer.

        A 9B at 32x14336 is ~1.8MB per sample in float32, so 500 samples is
        ~1.5GB. Pass max_samples on a laptop.
        """
        ids = list(ids if ids is not None else self.ids)
        if max_samples:
            ids = ids[:max_samples]
        out, kept = [], []
        for qid in ids:
            d = self.get(qid)
            if key in d:
                out.append(d[key].astype(np.float32))
                kept.append(qid)
        if not out:
            return np.zeros((0, 0, 0), dtype=np.float32), []
        return np.stack(out), kept


def merge_check(sessions):
    """Refuse to combine sessions that are not comparable.

    Neuron indices belong to specific weights. Two sessions from the same model
    at different quantizations have the same shape and different meanings, so a
    shape check alone would not catch it -- the fingerprint and quant must match
    too.
    """
    if not sessions:
        return
    ref = sessions[0].meta
    for s in sessions[1:]:
        for key in ("fingerprint", "n_layers", "n_neurons", "quant"):
            a, b = ref.get(key), s.meta.get(key)
            if a != b:
                raise ValueError(
                    f"cannot combine sessions: {key} differs ({a!r} vs {b!r}).\n"
                    f"  {ref.get('model', '?')} vs {s.meta.get('model', '?')}\n"
                    "Neuron indices are a property of specific weights.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Inspect a OmniCortex session")
    p.add_argument("path")
    a = p.parse_args()
    s = Session(a.path)
    print(json.dumps(s.meta, indent=2))
    print(f"\n{len(s)} samples, {len(s.events())} events")
    for e in s.events()[-5:]:
        print("  ", json.dumps(e)[:120])
