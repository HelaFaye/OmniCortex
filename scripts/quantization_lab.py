#!/usr/bin/env python3
"""OmniCortex capability-preserving quantization planner and runner.

This module deliberately does not invent a new GGUF quantizer. It builds on
llama.cpp's supported importance-matrix and per-tensor-type quantization paths:

  1. Weighted calibration corpora -> llama-imatrix
  2. H-Neuron profiles -> layer/tensor protection scores
  3. llama-quantize --imatrix + --tensor-type -> mixed-precision GGUF
  4. Optional tmpfs scratch -> validate -> atomic copy to destination

The first implementation protects whole FFN down-projection tensors for the
highest-scoring layers. That is an intentional safety/compatibility boundary:
llama-quantize can select tensor types, but it does not expose arbitrary
per-column bit allocation through its normal CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

try:
    import gguf  # optional for exact model tensor inspection
except Exception:  # pragma: no cover - availability depends on environment
    gguf = None


QUANT_BPW = {
    "Q2_K": 2.5625,
    "Q3_K_S": 3.4375,
    "Q3_K_M": 3.999,
    "Q3_K_L": 4.4375,
    "Q4_0": 4.5,
    "Q4_K_S": 4.5,
    "Q4_K_M": 4.75,
    "Q5_0": 5.5,
    "Q5_K_S": 5.5,
    "Q5_K_M": 5.75,
    "Q6_K": 6.5625,
    "Q8_0": 8.5,
    "F16": 16.0,
    "BF16": 16.0,
    "F32": 32.0,
}

QUANT_RE = re.compile(r"(?:Q\d+(?:_[A-Z0-9]+)?|IQ\d+(?:_[A-Z0-9]+)?|F16|BF16|F32)", re.I)
LAYER_RE = re.compile(r"(?:^|\.)blk\.(\d+)\.")


@dataclass(frozen=True)
class CapabilityInput:
    path: str
    weight: float = 1.0
    label: str = ""


@dataclass(frozen=True)
class LayerScore:
    layer: int
    score: float
    selected_neurons: int
    density: float
    contributions: dict[str, float]


class QuantizationLabError(RuntimeError):
    pass


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _read_text_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n\r") for line in f if line.strip()]


def _even_sample(items: Sequence[str], n: int) -> list[str]:
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


def parse_weighted_specs(specs: Sequence[str]) -> list[CapabilityInput]:
    out: list[CapabilityInput] = []
    for raw in specs:
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.rsplit("::", 1)
        if len(parts) == 2:
            path, weight_s = parts
            try:
                weight = float(weight_s)
            except ValueError as e:
                raise QuantizationLabError(f"invalid weight in {raw!r}") from e
        else:
            path, weight = raw, 1.0
        path = os.path.expanduser(path.strip())
        if not os.path.isfile(path):
            raise QuantizationLabError(f"file not found: {path}")
        if weight <= 0:
            raise QuantizationLabError(f"weight must be > 0: {raw!r}")
        out.append(CapabilityInput(path=path, weight=weight, label=Path(path).stem))
    if not out:
        raise QuantizationLabError("no weighted inputs were supplied")
    return out


def build_weighted_calibration(
    inputs: Sequence[CapabilityInput],
    output: str,
    max_lines: int = 20000,
) -> dict:
    """Create a deterministic, weighted calibration text corpus.

    We sample evenly within each source file, then interleave according to
    weights. This avoids simply duplicating corpora and keeps the combined file
    bounded and reproducible.
    """
    if max_lines <= 0:
        raise QuantizationLabError("max_lines must be > 0")
    total_weight = sum(x.weight for x in inputs)
    data = []
    counts = {}
    for x in inputs:
        lines = _read_text_lines(x.path)
        if not lines:
            continue
        n = max(1, int(round(max_lines * x.weight / total_weight)))
        n = min(n, len(lines))
        data.append((x, _even_sample(lines, n)))
        counts[x.path] = n

    if not data:
        raise QuantizationLabError("all calibration files were empty")

    # Deterministic weighted round-robin based on the normalized line quotas.
    remaining = [list(lines) for _, lines in data]
    labels = [x.label or x.path for x, _ in data]
    weights = [x.weight for x, _ in data]
    emitted: list[tuple[int, str]] = []
    cursors = [0] * len(remaining)
    while len(emitted) < max_lines and any(cursors[i] < len(remaining[i]) for i in range(len(remaining))):
        best = None
        best_ratio = -1.0
        for i in range(len(remaining)):
            if cursors[i] >= len(remaining[i]):
                continue
            # Weighted fair scheduling: desired share minus current share.
            desired = weights[i] / sum(weights)
            current = sum(1 for j, _ in emitted if j == i) / max(1, len(emitted))
            ratio = desired - current
            if ratio > best_ratio:
                best_ratio = ratio
                best = i
        assert best is not None
        emitted.append((best, remaining[best][cursors[best]]))
        cursors[best] += 1

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for _, line in emitted:
            f.write(line)
            f.write("\n")

    return {
        "output": str(out_path),
        "sha256": sha256_file(out_path),
        "line_count": len(emitted),
        "sources": [
            {"path": x.path, "weight": x.weight, "label": labels[i], "sampled_lines": counts.get(x.path, 0)}
            for i, (x, _) in enumerate(data)
        ],
    }


def load_profiles(inputs: Sequence[CapabilityInput]) -> list[tuple[CapabilityInput, dict]]:
    loaded = []
    for spec in inputs:
        try:
            with open(spec.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise QuantizationLabError(f"cannot read profile {spec.path}: {e}") from e
        if not isinstance(data, dict) or "by_layer" not in data:
            raise QuantizationLabError(f"not a OmniCortex profile: {spec.path}")
        raw = data.get("by_layer") or {}
        by_layer: dict[int, list[int]] = {}
        for k, values in raw.items():
            layer = int(k)
            vals = [int(v) for v in values]
            if len(vals) != len(set(vals)):
                raise QuantizationLabError(f"profile {spec.path}: duplicate neurons in layer {layer}")
            by_layer[layer] = sorted(vals)
        n_layers = int(data.get("n_layers", max(by_layer, default=-1) + 1))
        n_neurons = int(data.get("n_neurons", max((max(v) for v in by_layer.values() if v), default=-1) + 1))
        if n_layers <= 0 or n_neurons <= 0:
            raise QuantizationLabError(f"profile {spec.path}: invalid dimensions")
        for layer, neurons in by_layer.items():
            if layer < 0 or layer >= n_layers:
                raise QuantizationLabError(f"profile {spec.path}: layer {layer} outside 0..{n_layers - 1}")
            bad = [n for n in neurons if n < 0 or n >= n_neurons]
            if bad:
                raise QuantizationLabError(f"profile {spec.path}: invalid neuron indices {bad[:5]}")
        loaded.append((spec, {"data": data, "by_layer": by_layer, "n_layers": n_layers, "n_neurons": n_neurons}))
    return loaded


def score_layers(profile_inputs: Sequence[CapabilityInput]) -> tuple[list[LayerScore], int]:
    loaded = load_profiles(profile_inputs)
    if not loaded:
        return [], 0
    total_weight = sum(x.weight for x, _ in loaded)
    n_layers = min(v["n_layers"] for _, v in loaded)
    scores: dict[int, dict] = {}
    for spec, p in loaded:
        for layer in range(n_layers):
            neurons = p["by_layer"].get(layer, [])
            density = len(neurons) / p["n_neurons"]
            contribution = spec.weight / total_weight * density
            rec = scores.setdefault(layer, {"score": 0.0, "selected": 0, "contrib": {}})
            rec["score"] += contribution
            rec["selected"] += len(neurons)
            rec["contrib"][spec.label or spec.path] = contribution

    ranked = []
    for layer, rec in scores.items():
        density = rec["selected"] / max(1, sum(p["n_neurons"] for _, p in loaded) / len(loaded))
        ranked.append(LayerScore(layer, rec["score"], rec["selected"], density, rec["contrib"]))
    ranked.sort(key=lambda x: (-x.score, x.layer))
    return ranked, n_layers


def _quant_guess(path: str) -> str | None:
    m = QUANT_RE.search(Path(path).name)
    return m.group(0).upper() if m else None


def _is_high_precision(path: str) -> bool:
    q = _quant_guess(path)
    return q in ("F16", "BF16", "F32")


def read_gguf_info(path: str) -> dict:
    out = {
        "path": str(path),
        "size_bytes": os.path.getsize(path),
        "size_gib": os.path.getsize(path) / 2**30,
        "quant": _quant_guess(path),
        "arch": None,
        "n_layers": None,
        "n_neurons": None,
        "tensor_names": [],
        "tensor_elements": {},
        "warning": None,
    }
    if gguf is None:
        out["warning"] = "python package 'gguf' is not installed; using filename/profile metadata only"
        return out
    try:
        r = gguf.GGUFReader(path)
        fields = getattr(r, "fields", {})
        def kv(key):
            f = fields.get(key)
            if f is None:
                return None
            try:
                v = f.contents()
                return v.decode("utf-8") if isinstance(v, bytes) else v
            except Exception:
                try:
                    value = f.parts[f.data[0]][0]
                    return value.item() if hasattr(value, "item") else value
                except Exception:
                    return None
        out["arch"] = kv("general.architecture")
        if out["arch"]:
            out["n_layers"] = kv(f"{out['arch']}.block_count")
            out["n_neurons"] = kv(f"{out['arch']}.intermediate_length") or kv(f"{out['arch']}.feed_forward_length")
        names = []
        elements = {}
        for t in getattr(r, "tensors", []):
            name = str(getattr(t, "name", ""))
            if not name:
                continue
            names.append(name)
            shape = getattr(t, "shape", None)
            if shape is not None:
                try:
                    n = 1
                    for x in shape:
                        n *= int(x)
                    elements[name] = int(n)
                except Exception:
                    pass
        out["tensor_names"] = names
        out["tensor_elements"] = elements
    except Exception as e:
        out["warning"] = f"GGUF inspection failed: {e}"
    return out


def _eligible_down_tensors(info: dict, layers: Iterable[int], include_shared_expert: bool = False) -> dict[int, list[str]]:
    names = info.get("tensor_names") or []
    wanted = set(int(x) for x in layers)
    mapping = {layer: [] for layer in wanted}
    for name in names:
        m = re.search(r"(?:^|\.)blk\.(\d+)\.", name)
        if not m:
            continue
        layer = int(m.group(1))
        if layer not in wanted:
            continue
        if re.search(r"ffn_down(?:_exps)?\.weight$", name):
            mapping.setdefault(layer, []).append(name)
        elif include_shared_expert and re.search(r"ffn_down_shexp\.weight$", name):
            mapping.setdefault(layer, []).append(name)
    return mapping


def tensor_regex_for_layers(layers: Sequence[int], include_shared_expert: bool = False) -> str:
    if not layers:
        raise QuantizationLabError("cannot build tensor regex with no layers")
    alt = "|".join(str(x) for x in sorted(set(layers)))
    if include_shared_expert:
        return rf"blk\.({alt})\.ffn_down(_exps|_shexp)?\.weight"
    return rf"blk\.({alt})\.ffn_down(_exps)?\.weight"


def estimate_precision_delta(
    info: dict,
    tensor_names: Sequence[str],
    base_quant: str,
    protected_quant: str,
) -> dict:
    base_bpw = QUANT_BPW.get(base_quant.upper())
    prot_bpw = QUANT_BPW.get(protected_quant.upper())
    elems = sum(int(info.get("tensor_elements", {}).get(n, 0)) for n in tensor_names)
    if not elems or not base_bpw or not prot_bpw:
        return {"elements": elems, "estimated_delta_bytes": None, "estimated_delta_gib": None}
    delta = elems * max(0.0, prot_bpw - base_bpw) / 8.0
    return {"elements": elems, "estimated_delta_bytes": int(delta), "estimated_delta_gib": delta / 2**30}


def build_plan(
    model_path: str,
    base_quant: str,
    protected_quant: str,
    budget_percent: float,
    profile_inputs: Sequence[CapabilityInput] = (),
    include_shared_expert: bool = False,
) -> dict:
    if not (0 <= budget_percent <= 100):
        raise QuantizationLabError("budget_percent must be between 0 and 100")
    model_path = os.path.expanduser(model_path)
    if not os.path.isfile(model_path):
        raise QuantizationLabError(f"model not found: {model_path}")
    base_quant = base_quant.upper()
    protected_quant = protected_quant.upper()
    if base_quant not in QUANT_BPW:
        raise QuantizationLabError(f"unknown/unsupported planning quant type: {base_quant}")
    if protected_quant not in QUANT_BPW:
        raise QuantizationLabError(f"unknown/unsupported protected quant type: {protected_quant}")

    info = read_gguf_info(model_path)
    ranked, profile_layers = score_layers(profile_inputs) if profile_inputs else ([], 0)
    n_layers = int(info.get("n_layers") or profile_layers or (max((r.layer for r in ranked), default=-1) + 1))
    if n_layers <= 0:
        raise QuantizationLabError("could not determine model layer count; supply a valid OmniCortex profile")

    if info.get("n_layers") and profile_layers and int(info["n_layers"]) != profile_layers:
        raise QuantizationLabError(
            f"profile/model layer mismatch: profile {profile_layers}, model {info['n_layers']}"
        )

    eligible_layers = [r.layer for r in ranked if r.score > 0]
    if not profile_inputs:
        eligible_layers = []
    max_protected = int(math.floor(n_layers * budget_percent / 100.0 + 1e-9))
    if budget_percent > 0 and eligible_layers and max_protected == 0:
        max_protected = 1
    selected = [r for r in ranked if r.layer in eligible_layers][:max_protected]
    selected_layers = [r.layer for r in selected]
    tensor_map = _eligible_down_tensors(info, selected_layers, include_shared_expert)
    matched_names = [n for names in tensor_map.values() for n in names]
    unmatched_layers = [layer for layer in selected_layers if not tensor_map.get(layer)]
    regex = tensor_regex_for_layers(selected_layers, include_shared_expert) if selected_layers else None
    delta = estimate_precision_delta(info, matched_names, base_quant, protected_quant)

    return {
        "version": 1,
        "model": info,
        "base_quant": base_quant,
        "protected_quant": protected_quant,
        "budget_percent": budget_percent,
        "n_layers": n_layers,
        "selected_layers": selected_layers,
        "selected_layer_count": len(selected_layers),
        "layer_scores": [asdict(r) for r in ranked],
        "tensor_map": tensor_map,
        "matched_tensor_names": matched_names,
        "unmatched_layers": unmatched_layers,
        "tensor_regex": regex,
        "estimate": delta,
        "warnings": [
            *(["GGUF tensor names were not available; the quantizer will use the generated regex directly."] if not info.get("tensor_names") else []),
            *([f"{len(unmatched_layers)} selected layers had no matching ffn_down tensor in the inspected GGUF."] if unmatched_layers else []),
            *(["Source filename looks quantized; canonical builds should start from F16/BF16 to avoid requantization loss."] if not _is_high_precision(model_path) else []),
        ],
    }


def plan_command(plan: dict) -> list[str]:
    cmd = []
    if plan.get("tensor_regex"):
        cmd += ["--tensor-type", f"{plan['tensor_regex']}={plan['protected_quant'].lower()}"]
    return cmd


def scratch_status(path: str, required_bytes: int = 0) -> dict:
    root = os.path.expanduser(path)
    os.makedirs(root, exist_ok=True)
    u = shutil.disk_usage(root)
    is_tmpfs = False
    try:
        mounts = Path("/proc/mounts").read_text(errors="replace").splitlines()
        real = os.path.realpath(root)
        for line in mounts:
            fields = line.split()
            if len(fields) >= 3 and fields[1] == real:
                is_tmpfs = fields[2] == "tmpfs"
                break
        if not is_tmpfs:
            is_tmpfs = "/dev/shm" in real
    except Exception:
        pass
    return {
        "path": root,
        "total_bytes": u.total,
        "free_bytes": u.free,
        "free_gib": u.free / 2**30,
        "is_tmpfs": is_tmpfs,
        "required_bytes": required_bytes,
        "fits": u.free >= required_bytes if required_bytes else None,
    }


def choose_scratch(requested: str, required_bytes: int) -> tuple[str, dict]:
    requested = (requested or "auto").strip()
    if requested not in ("auto", "ram", "disk", "none") and os.path.isabs(requested):
        st = scratch_status(requested, required_bytes)
        if not st["fits"]:
            raise QuantizationLabError(f"scratch path lacks required free space: {st}")
        return requested, st

    if requested in ("auto", "ram"):
        ram_root = "/dev/shm" if os.path.isdir("/dev/shm") else None
        if ram_root:
            st = scratch_status(ram_root, required_bytes)
            if st["fits"]:
                return ram_root, st
            if requested == "ram":
                raise QuantizationLabError(
                    f"RAM scratch requested but insufficient free tmpfs space: {st['free_gib']:.2f} GiB"
                )

    if requested == "none":
        return "", {"path": "", "is_tmpfs": False, "free_bytes": None, "required_bytes": 0, "fits": True}

    root = tempfile.gettempdir()
    st = scratch_status(root, required_bytes)
    if not st["fits"]:
        raise QuantizationLabError(f"system scratch lacks required free space: {st}")
    return root, st


def _run(cmd: Sequence[str], log: list[str], env: dict | None = None) -> None:
    proc = subprocess.Popen(
        list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        log.append(line)
        if len(log) > 500:
            del log[:-500]
    rc = proc.wait()
    if rc != 0:
        raise QuantizationLabError(f"command failed with exit code {rc}: {' '.join(cmd)}")


def _estimate_output_bytes(source_size: int, target_quant: str) -> int:
    bpw = QUANT_BPW.get(target_quant.upper(), 5.0)
    # F16 source weights are close to 16 bits/weight; GGUF metadata and block
    # overhead make this an estimate, not a promise.
    return int(source_size * (bpw / 16.0) * 1.15)


def atomic_copy(src: str, dst: str, force: bool = False) -> None:
    src = os.path.abspath(src)
    dst = os.path.abspath(os.path.expanduser(dst))
    if src == dst:
        raise QuantizationLabError("temporary output and destination must differ")
    if os.path.exists(dst) and not force:
        raise QuantizationLabError(f"destination exists; use --force to replace: {dst}")
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = dst + f".partial.{os.getpid()}"
    try:
        with open(src, "rb") as rf, open(tmp, "wb") as wf:
            while True:
                chunk = rf.read(16 * 1024 * 1024)
                if not chunk:
                    break
                wf.write(chunk)
            wf.flush()
            os.fsync(wf.fileno())
        os.replace(tmp, dst)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def build_model(
    *,
    model: str,
    output: str,
    calibration: Sequence[CapabilityInput],
    profiles: Sequence[CapabilityInput] = (),
    base_quant: str = "Q4_K_M",
    protected_quant: str = "Q6_K",
    budget_percent: float = 10.0,
    scratch: str = "auto",
    imatrix_bin: str = "llama-imatrix",
    quantize_bin: str = "llama-quantize",
    threads: int = 0,
    max_calibration_lines: int = 20000,
    include_shared_expert: bool = False,
    allow_requantize_source: bool = False,
    force: bool = False,
    log: list[str] | None = None,
) -> dict:
    log = log if log is not None else []
    model = os.path.abspath(os.path.expanduser(model))
    output = os.path.abspath(os.path.expanduser(output))
    imatrix_bin = os.path.expanduser(imatrix_bin)
    quantize_bin = os.path.expanduser(quantize_bin)
    if not os.path.isfile(model):
        raise QuantizationLabError(f"model not found: {model}")
    if model == output:
        raise QuantizationLabError("output cannot be the source model")
    if not calibration:
        raise QuantizationLabError("at least one calibration file is required")
    if not _is_high_precision(model) and not allow_requantize_source:
        raise QuantizationLabError(
            "source model does not look like F16/BF16/F32; canonical OmniCortex builds should start from a high-precision source. "
            "Use --allow-requantize-source only when you explicitly accept source requantization."
        )

    plan = build_plan(
        model_path=model,
        base_quant=base_quant,
        protected_quant=protected_quant,
        budget_percent=budget_percent,
        profile_inputs=profiles,
        include_shared_expert=include_shared_expert,
    )

    required = _estimate_output_bytes(os.path.getsize(model), base_quant)
    required += max(256 * 1024 * 1024, required // 10)  # imatrix/headroom
    scratch_root, scratch_info = choose_scratch(scratch, required)
    owned_tmp = None
    if scratch_root:
        owned_tmp = tempfile.mkdtemp(prefix="neuronscope-quant-", dir=scratch_root)
    else:
        owned_tmp = tempfile.mkdtemp(prefix="neuronscope-quant-")

    try:
        calib_path = os.path.join(owned_tmp, "calibration.txt")
        calib_meta = build_weighted_calibration(calibration, calib_path, max_calibration_lines)
        imatrix_path = os.path.join(owned_tmp, "imatrix.gguf")
        output_tmp = os.path.join(owned_tmp, Path(output).name)

        imatrix_cmd = [imatrix_bin, "-m", model, "-f", calib_path, "-o", imatrix_path]
        quant_cmd = [quantize_bin, "--imatrix", imatrix_path]
        if allow_requantize_source:
            quant_cmd.append("--allow-requantize")
        quant_cmd += plan_command(plan)
        quant_cmd += [model, output_tmp, base_quant.lower()]
        if threads > 0:
            quant_cmd.append(str(threads))

        log.append(f"scratch: {owned_tmp} ({'tmpfs' if scratch_info.get('is_tmpfs') else 'disk'})")
        log.append(f"calibration lines: {calib_meta['line_count']}")
        log.append("$ " + " ".join(imatrix_cmd))
        _run(imatrix_cmd, log)
        if not os.path.isfile(imatrix_path):
            raise QuantizationLabError("llama-imatrix completed without producing the expected imatrix file")

        log.append("$ " + " ".join(quant_cmd))
        _run(quant_cmd, log)
        if not os.path.isfile(output_tmp) or os.path.getsize(output_tmp) == 0:
            raise QuantizationLabError("llama-quantize completed without a non-empty output file")

        digest = sha256_file(output_tmp)
        atomic_copy(output_tmp, output, force=force)
        manifest = {
            "version": 1,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": {"path": model, "sha256": sha256_file(model), "size_bytes": os.path.getsize(model)},
            "output": {"path": output, "sha256": digest, "size_bytes": os.path.getsize(output)},
            "base_quant": base_quant.upper(),
            "protected_quant": protected_quant.upper(),
            "budget_percent": budget_percent,
            "plan": plan,
            "calibration": calib_meta,
            "commands": {"imatrix": imatrix_cmd, "quantize": quant_cmd},
            "scratch": scratch_info,
            "allow_requantize_source": allow_requantize_source,
        }
        manifest_path = output + ".neuronscope.json"
        tmp_manifest = manifest_path + f".partial.{os.getpid()}"
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_manifest, manifest_path)
        log.append(f"wrote: {output}")
        log.append(f"sha256: {digest}")
        return manifest
    finally:
        shutil.rmtree(owned_tmp, ignore_errors=True)


def _cli() -> int:
    p = argparse.ArgumentParser(description="OmniCortex capability-preserving quantization lab")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("inspect", help="inspect GGUF metadata")
    pi.add_argument("model")

    pp = sub.add_parser("plan", help="score H-Neuron profiles and print a protection plan")
    pp.add_argument("--model", required=True)
    pp.add_argument("--profile", action="append", default=[], help="PROFILE.json[::weight]")
    pp.add_argument("--base-quant", default="Q4_K_M")
    pp.add_argument("--protected-quant", default="Q6_K")
    pp.add_argument("--budget-percent", type=float, default=10.0)
    pp.add_argument("--include-shared-expert", action="store_true")
    pp.add_argument("--json", action="store_true")

    pb = sub.add_parser("build", help="run weighted imatrix + capability-aware quantization")
    pb.add_argument("--model", required=True)
    pb.add_argument("--output", required=True)
    pb.add_argument("--calibration", action="append", required=True, help="TEXT[::weight]")
    pb.add_argument("--profile", action="append", default=[], help="PROFILE.json[::weight]")
    pb.add_argument("--base-quant", default="Q4_K_M")
    pb.add_argument("--protected-quant", default="Q6_K")
    pb.add_argument("--budget-percent", type=float, default=10.0)
    pb.add_argument("--scratch", default="auto", help="auto|ram|disk|none|/path")
    pb.add_argument("--imatrix-bin", default="llama-imatrix")
    pb.add_argument("--quantize-bin", default="llama-quantize")
    pb.add_argument("--threads", type=int, default=0)
    pb.add_argument("--max-calibration-lines", type=int, default=20000)
    pb.add_argument("--include-shared-expert", action="store_true")
    pb.add_argument("--allow-requantize-source", action="store_true")
    pb.add_argument("--force", action="store_true")

    pc = sub.add_parser("calibration", help="create a deterministic weighted calibration corpus")
    pc.add_argument("--input", action="append", required=True, help="TEXT[::weight]")
    pc.add_argument("--output", required=True)
    pc.add_argument("--max-lines", type=int, default=20000)

    ps = sub.add_parser("scratch", help="report scratch space")
    ps.add_argument("path")
    ps.add_argument("--required-gib", type=float, default=0)

    a = p.parse_args()
    try:
        if a.command == "inspect":
            print(json.dumps(read_gguf_info(a.model), indent=2))
            return 0
        if a.command == "plan":
            profiles = parse_weighted_specs(a.profile) if a.profile else []
            plan = build_plan(a.model, a.base_quant, a.protected_quant, a.budget_percent, profiles, a.include_shared_expert)
            if a.json:
                print(json.dumps(plan, indent=2))
            else:
                print(f"Model: {plan['model']['path']}")
                print(f"Layers: {plan['n_layers']}")
                print(f"Protected layers: {plan['selected_layers']}")
                print(f"Protected tensor rule: {plan['tensor_regex'] or 'none'}")
                for w in plan["warnings"]:
                    print(f"WARNING: {w}")
            return 0
        if a.command == "calibration":
            meta = build_weighted_calibration(parse_weighted_specs(a.input), a.output, a.max_lines)
            print(json.dumps(meta, indent=2))
            return 0
        if a.command == "scratch":
            print(json.dumps(scratch_status(a.path, int(a.required_gib * 2**30)), indent=2))
            return 0
        if a.command == "build":
            logs: list[str] = []
            manifest = build_model(
                model=a.model,
                output=a.output,
                calibration=parse_weighted_specs(a.calibration),
                profiles=parse_weighted_specs(a.profile) if a.profile else [],
                base_quant=a.base_quant,
                protected_quant=a.protected_quant,
                budget_percent=a.budget_percent,
                scratch=a.scratch,
                imatrix_bin=a.imatrix_bin,
                quantize_bin=a.quantize_bin,
                threads=a.threads,
                max_calibration_lines=a.max_calibration_lines,
                include_shared_expert=a.include_shared_expert,
                allow_requantize_source=a.allow_requantize_source,
                force=a.force,
                log=logs,
            )
            print(json.dumps({"manifest": manifest, "log": logs}, indent=2))
            return 0
    except QuantizationLabError as e:
        print(f"error: {e}", file=os.sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
