#!/usr/bin/env python3
"""OmniCortex Scale Sweep Lab core.

Builds static GGUF scale variants from an existing OmniCortex H-Neuron profile.
It delegates actual GGUF editing to scripts/suppress_gguf.py and adds:

- deterministic scale-grid generation (suppression and amplification)
- configurable batch concurrency
- RAM/tmpfs/disk scratch staging
- resumable manifests
- optional external evaluation command
- confidence-aware candidate pruning
- activation/profile visualization data
- no shell exit() calls; errors are returned/raised cleanly
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUPPRESS = ROOT / "scripts" / "suppress_gguf.py"


class SweepError(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    scale: float
    name: str
    output: str
    scratch: str = ""
    status: str = "pending"
    score: float | None = None
    lower_ci: float | None = None
    upper_ci: float | None = None
    correct: int | None = None
    total: int | None = None
    wrong: int | None = None
    abstained: int | None = None
    error: str | None = None


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_scales(min_scale: float, max_scale: float, step: float, include_zero: bool = False) -> list[float]:
    if not all(math.isfinite(x) for x in (min_scale, max_scale, step)):
        raise SweepError("scale range values must be finite")
    if step <= 0:
        raise SweepError("scale step/resolution must be > 0")
    if min_scale > max_scale:
        raise SweepError("min_scale must be <= max_scale")
    values: list[float] = []
    x = min_scale
    # Decimal-ish rounding prevents 0.30000000000000004 filenames.
    digits = max(0, min(8, len(str(step).split(".")[-1].rstrip("0")) if "." in str(step) else 0))
    while x <= max_scale + step * 1e-9:
        xr = round(x, digits + 2)
        if include_zero or xr != 0:
            values.append(xr)
        x += step
    # Always include exact endpoints.
    for endpoint in (min_scale, max_scale):
        ep = round(endpoint, digits + 2)
        if ep not in values and (include_zero or ep != 0):
            values.append(ep)
    return sorted(set(values))


def scale_tag(scale: float) -> str:
    """Stable human-friendly tag: 0.20 -> supp020, 1.20 -> amp120."""
    if abs(scale - 1.0) < 1e-9:
        return "base100"
    sign = "supp" if scale < 1.0 else "amp"
    milli = int(round(abs(scale) * 1000))
    return f"{sign}{milli:03d}"


def candidate_name(stem: str, scale: float, extension: str = ".gguf") -> str:
    return f"{stem}-supp{scale_tag(scale)}{extension}"


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        raise SweepError("total must be > 0 for confidence interval")
    p = correct / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / den
    return max(0.0, center - half), min(1.0, center + half)


def parse_eval_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise SweepError("evaluation command returned no JSON")
    # Permit logs around a final JSON object.
    starts = [i for i, c in enumerate(text) if c == "{"]
    for i in reversed(starts):
        try:
            obj = json.loads(text[i:])
            break
        except Exception:
            continue
    else:
        raise SweepError("evaluation command did not return a JSON object")
    if "score" not in obj:
        if "correct" in obj and "total" in obj:
            obj["score"] = float(obj["correct"]) / max(1, int(obj["total"]))
        else:
            raise SweepError("evaluation JSON requires score or correct+total")
    obj["score"] = float(obj["score"])
    if not 0 <= obj["score"] <= 1:
        raise SweepError("evaluation score must be in [0,1]")
    if "correct" in obj and "total" in obj:
        obj["correct"] = int(obj["correct"])
        obj["total"] = int(obj["total"])
        obj["lower_ci"], obj["upper_ci"] = wilson_interval(obj["correct"], obj["total"])
    return obj


def scratch_status(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    if path:
        p = Path(os.path.expanduser(str(path)))
        p.mkdir(parents=True, exist_ok=True)
    else:
        p = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())
    usage = shutil.disk_usage(p)
    return {
        "path": str(p),
        "free_bytes": usage.free,
        "free_gib": usage.free / 2**30,
        "total_gib": usage.total / 2**30,
        "is_tmpfs_hint": str(p).startswith("/dev/shm"),
    }


def choose_scratch(policy: str, model_size: int, candidates: int, requested: str = "") -> Path | None:
    if policy == "none":
        return None
    if policy == "disk":
        root = Path(os.path.expanduser(requested or tempfile.gettempdir()))
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="neuronscope-scale-", dir=root))
    # auto/ram need capacity for the current batch, not the entire sweep.
    if policy == "ram":
        roots = [Path("/dev/shm")] if Path("/dev/shm").is_dir() else []
    else:
        roots = [Path("/dev/shm")] if Path("/dev/shm").is_dir() else []
        roots.append(Path(os.path.expanduser(requested or tempfile.gettempdir())))
    need = int(model_size * max(1, candidates) * 1.03)
    for root in roots:
        try:
            root.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(root).free >= need:
                return Path(tempfile.mkdtemp(prefix="neuronscope-scale-", dir=root))
        except OSError:
            continue
    if policy == "ram":
        raise SweepError(f"insufficient RAM/tmpfs scratch for {candidates} model(s); need about {need/2**30:.2f} GiB")
    return Path(tempfile.mkdtemp(prefix="neuronscope-scale-", dir=Path(tempfile.gettempdir())))


def _run_suppressor(
    suppressor: Path,
    model: str,
    profile: str,
    scale: float,
    output: Path,
    timeout: int,
) -> None:
    if not suppressor.is_file():
        raise SweepError(f"suppression backend not found: {suppressor}")
    cmd = [
        os.environ.get("PYTHON", "python3"), str(suppressor),
        "--gguf", model,
        "--h_neurons", profile,
        "--scale", str(scale),
        "--out", str(output),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "backend failed").strip()
        raise SweepError(f"suppression backend failed ({proc.returncode}): {detail[-2000:]}")
    if not output.is_file() or output.stat().st_size == 0:
        raise SweepError("suppression backend returned success but produced no GGUF")


def _publish(src: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + f".tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        shutil.copy2(src, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, destination)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def activation_profile_data(profile_path: str) -> dict[str, Any]:
    with open(profile_path, encoding="utf-8") as f:
        d = json.load(f)
    if "by_layer" not in d:
        raise SweepError(f"not a OmniCortex H-Neuron profile: {profile_path}")
    n = int(d.get("n_neurons", 0))
    layers = int(d.get("n_layers", 0))
    by = {str(k): sorted(int(x) for x in v) for k, v in (d.get("by_layer") or {}).items()}
    points = []
    for layer in range(layers):
        neurons = by.get(str(layer), [])
        points.append({
            "layer": layer,
            "selected": len(neurons),
            "density": (len(neurons) / n) if n else 0.0,
            "neurons": neurons,
        })
    return {"n_layers": layers, "n_neurons": n, "layers": points}


class SweepEngine:
    def __init__(self, state_path: str | os.PathLike[str] | None = None):
        self.state_path = Path(state_path) if state_path else None
        self.lock = threading.Lock()
        self.stop_requested = False
        self.history: list[dict[str, Any]] = []

    def stop(self) -> None:
        self.stop_requested = True

    def _write_state(self, state: dict[str, Any]) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def run(
        self,
        *,
        model: str,
        profile: str,
        output_dir: str,
        scales: Sequence[float],
        batch_size: int = 1,
        scratch_policy: str = "auto",
        scratch_root: str = "",
        suppressor: str = str(DEFAULT_SUPPRESS),
        timeout: int = 3600,
        evaluator_cmd: str = "",
        baseline_score: float | None = None,
        margin_of_error: float = 0.05,
        auto_delete: bool = False,
        extension: str = ".gguf",
    ) -> dict[str, Any]:
        model = os.path.abspath(os.path.expanduser(model))
        profile = os.path.abspath(os.path.expanduser(profile))
        out_root = Path(os.path.abspath(os.path.expanduser(output_dir)))
        out_root.mkdir(parents=True, exist_ok=True)
        if not os.path.isfile(model):
            raise SweepError(f"model not found: {model}")
        if not os.path.isfile(profile):
            raise SweepError(f"profile not found: {profile}")
        if batch_size < 1:
            raise SweepError("batch_size must be >= 1")
        if margin_of_error < 0:
            raise SweepError("margin_of_error must be >= 0")
        if not scales:
            raise SweepError("no scales supplied")
        if auto_delete and not evaluator_cmd:
            raise SweepError("auto_delete requires evaluator_cmd")

        stem = Path(model).stem
        model_size = os.path.getsize(model)
        candidates = [
            Candidate(scale=float(s), name=f"{stem}-supp{scale_tag(float(s))}", output=str(out_root / candidate_name(stem, float(s), extension)))
            for s in scales
        ]

        state = {
            "version": 1,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": model,
            "model_sha256": sha256_file(model),
            "profile": profile,
            "profile_sha256": sha256_file(profile),
            "scales": [c.scale for c in candidates],
            "batch_size": batch_size,
            "scratch_policy": scratch_policy,
            "margin_of_error": margin_of_error,
            "auto_delete": auto_delete,
            "candidates": [asdict(c) for c in candidates],
            "history": [],
        }
        self._write_state(state)

        for start in range(0, len(candidates), batch_size):
            if self.stop_requested:
                break
            batch = candidates[start:start + batch_size]
            scratch = choose_scratch(scratch_policy, model_size, len(batch), scratch_root)
            try:
                def build_one(c: Candidate) -> dict[str, Any]:
                    dest = Path(c.output)
                    if dest.is_file() and dest.stat().st_size == model_size:
                        return {**asdict(c), "status": "exists", "sha256": sha256_file(dest)}
                    scratch_file = None
                    if scratch:
                        scratch_file = scratch / dest.name
                        work = scratch_file
                    else:
                        work = dest.with_name(dest.name + f".work-{os.getpid()}-{threading.get_ident()}")
                    try:
                        _run_suppressor(Path(suppressor), model, profile, c.scale, work, timeout)
                        if scratch_file:
                            _publish(work, dest)
                        elif work != dest:
                            _publish(work, dest)
                        result = {**asdict(c), "status": "built", "sha256": sha256_file(dest), "size_bytes": dest.stat().st_size}
                        if evaluator_cmd:
                            env = os.environ.copy()
                            env.update({"NS_MODEL": str(dest), "NS_SCALE": str(c.scale), "NS_BASELINE": str(baseline_score if baseline_score is not None else ""), "NS_PROFILE": profile})
                            cmd = evaluator_cmd.format(model=str(dest), scale=str(c.scale), baseline=str(baseline_score if baseline_score is not None else ""), profile=profile)
                            ep = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
                            if ep.returncode != 0:
                                raise SweepError(f"evaluator failed ({ep.returncode}): {(ep.stderr or ep.stdout)[-2000:]}")
                            ev = parse_eval_json(ep.stdout)
                            result.update({"score": ev["score"], "correct": ev.get("correct"), "total": ev.get("total"), "lower_ci": ev.get("lower_ci"), "upper_ci": ev.get("upper_ci"), "wrong": ev.get("wrong"), "abstained": ev.get("abstained")})
                            if auto_delete and baseline_score is not None:
                                upper = result.get("upper_ci")
                                if upper is not None:
                                    under = upper < float(baseline_score) - margin_of_error
                                else:
                                    under = result["score"] < float(baseline_score) - margin_of_error
                                if under:
                                    dest.unlink(missing_ok=True)
                                    result["status"] = "deleted_underperformer"
                        return result
                    finally:
                        if work.exists() and work != dest:
                            try:
                                work.unlink()
                            except OSError:
                                pass

                with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as pool:
                    results = list(pool.map(build_one, batch))
                state["history"].extend(results)
                for c, r in zip(batch, results):
                    cidx = state["candidates"].index(asdict(c))
                    state["candidates"][cidx].update(r)
                self._write_state(state)
            finally:
                if scratch:
                    shutil.rmtree(scratch, ignore_errors=True)

        state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._write_state(state)
        self.history.extend(state["history"])
        return state


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build and evaluate OmniCortex GGUF scale sweeps")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--model", required=True)
    s.add_argument("--profile", required=True)
    s.add_argument("--output-dir", required=True)
    s.add_argument("--min-scale", type=float, default=0.20)
    s.add_argument("--max-scale", type=float, default=0.40)
    s.add_argument("--step", type=float, default=0.05)
    s.add_argument("--batch-size", type=int, default=1)
    s.add_argument("--scratch", choices=("auto", "ram", "disk", "none"), default="auto")
    s.add_argument("--scratch-root", default="")
    s.add_argument("--suppressor", default=str(DEFAULT_SUPPRESS))
    s.add_argument("--timeout", type=int, default=3600)
    s.add_argument("--state", default="")
    s.add_argument("--evaluator-cmd", default="")
    s.add_argument("--baseline-score", type=float)
    s.add_argument("--margin-of-error", type=float, default=0.05)
    s.add_argument("--auto-delete", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scales = parse_scales(args.min_scale, args.max_scale, args.step)
        engine = SweepEngine(args.state or str(Path(args.output_dir) / "sweep_state.json"))
        result = engine.run(
            model=args.model,
            profile=args.profile,
            output_dir=args.output_dir,
            scales=scales,
            batch_size=args.batch_size,
            scratch_policy=args.scratch,
            scratch_root=args.scratch_root,
            suppressor=args.suppressor,
            timeout=args.timeout,
            evaluator_cmd=args.evaluator_cmd,
            baseline_score=args.baseline_score,
            margin_of_error=args.margin_of_error,
            auto_delete=args.auto_delete,
        )
        print(json.dumps(result, indent=2))
        return 0
    except (SweepError, OSError, ValueError) as e:
        print(f"error: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
