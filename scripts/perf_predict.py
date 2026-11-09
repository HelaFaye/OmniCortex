#!/usr/bin/env python3
"""
Predict generation speed (tokens per second) for a model on a device, and learn from every measurement
(docs/HARDWARE.md#speed-predictions).

The starting point is physics. Generating one token reads every active weight once, so
speed ≈ efficiency × memory bandwidth / model bytes. Prompt processing is bound by compute, which the
measurements here rarely isolate, so only generation is predicted.

On top of that, a ridge regression learns a correction in log space from the measurements. Its features:
- the backend: CUDA, ROCm, Vulkan, Metal, SYCL or CPU;
- bits per weight;
- model size;
- the share of layers left on the CPU;
- flash attention;
- speculative decoding;
- parallel requests;
- unified memory.

Each device also gets its own offset, shrunk toward zero until it has several measurements (empirical
Bayes: n / (n + 3)). An unmeasured device is predicted from its specs and the shared weights, with a
wider range. Leave-one-device-out error says how well that works for hardware it has never seen.

Measurements come from Studio's work log (this computer and its hosts) and from a scoreboard's
board.json (cards carry the speeds their runs measured), so the weights improve as results arrive from
more and more different hardware.

    python scripts/perf_predict.py --board board.json --worklog ~/.omnicortex/worklog --show
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

# name pattern -> (memory bandwidth GB/s, dense fp16 TFLOPS). Published specs, rounded; a prior only, the
# fitted per-device offset corrects it. Unknown devices fall back by kind.
SPECS = [
    (r"H100", 3350, 990), (r"A100", 2039, 312), (r"L40S?", 864, 181), (r"A6000", 768, 155), (r"\bT4\b", 320, 65),
    (r"P100", 732, 19), (r"P40", 346, 12),
    (r"5090", 1792, 210), (r"5080", 960, 113), (r"5070 Ti", 896, 88), (r"5070", 672, 62), (r"5060 Ti", 448, 47),
    (r"4090", 1008, 165), (r"4080", 717, 97), (r"4070 Ti", 504, 80), (r"4070", 504, 58), (r"4060 Ti", 288, 44),
    (r"4060", 272, 30), (r"3090", 936, 71), (r"3080", 760, 59), (r"3070", 448, 40), (r"3060", 360, 25),
    (r"2080 Ti", 616, 27), (r"2080", 448, 20), (r"2070", 448, 15), (r"2060", 336, 13),
    (r"1080 Ti", 484, 11), (r"1080", 320, 9), (r"1070", 256, 6.5), (r"1060", 192, 4.4),
    (r"MI300", 5300, 1300), (r"MI250", 3277, 362), (r"MI210", 1638, 181), (r"MI100", 1229, 184), (r"MI50|Radeon VII", 1024, 27),
    (r"7900 XTX", 960, 123), (r"7900 XT", 800, 103), (r"7900 GRE", 576, 92), (r"7800 XT", 624, 75), (r"7700 XT", 432, 70),
    (r"7600", 288, 43), (r"9070 XT", 640, 97), (r"9070", 640, 72), (r"6950 XT", 576, 47), (r"6900 XT", 512, 46),
    (r"6800 XT", 512, 41), (r"6800", 512, 32), (r"6700 XT", 384, 26), (r"6600 XT", 256, 21), (r"6600", 224, 18),
    (r"A770", 560, 39), (r"A750", 512, 35), (r"A380", 186, 8), (r"B580", 456, 47),
    (r"890M", 120, 12), (r"880M", 120, 10), (r"780M", 90, 9), (r"760M", 90, 7), (r"680M", 80, 7), (r"8060S", 256, 30),
    (r"M4 Max", 546, 34), (r"M4 Pro", 273, 17), (r"\bM4\b", 120, 9), (r"M3 Max", 400, 28), (r"M3 Pro", 150, 14),
    (r"\bM3\b", 100, 9), (r"M2 Ultra", 800, 54), (r"M2 Max", 400, 27), (r"M2 Pro", 200, 14), (r"\bM2\b", 100, 7),
    (r"M1 Ultra", 800, 42), (r"M1 Max", 400, 21), (r"M1 Pro", 200, 10), (r"\bM1\b", 68, 5),
]
FALLBACK = {"cpu": (60, 1.0), "igpu": (80, 6.0), "gpu": (300, 20.0)}
BACKENDS = ("cuda", "rocm", "vulkan", "metal", "sycl", "cpu")
QUANT_BITS = {"F32": 32, "F16": 16, "BF16": 16, "Q8_0": 8.5, "Q6_K": 6.56, "Q5_K": 5.5, "Q5_0": 5.5, "Q5_1": 6,
              "Q4_K": 4.5, "Q4_0": 4.5, "Q4_1": 5, "IQ4": 4.3, "Q3_K": 3.4, "IQ3": 3.1, "Q2_K": 2.6, "IQ2": 2.3,
              "IQ1": 1.6, "TQ": 2}
FEATURES = ["backend_" + b for b in BACKENDS] + ["bits", "log_gb", "cpu_share", "flash_attn", "spec", "log_parallel",
                                                  "unified"]
SHRINK = 3.0
RIDGE = 1.0
EFFICIENCY = 0.6                   # typical share of peak bandwidth reached while generating


def spec(device: str, backend: str = "", unified: bool | None = None) -> tuple[float, float, str]:
    """-> (bandwidth GB/s, TFLOPS, where the numbers came from)."""
    name = device or ""
    for pat, bw, tf in SPECS:
        if re.search(pat, name, re.I):
            return float(bw), float(tf), "spec table"
    is_cpu = backend == "cpu" or (re.search(r"\bcpu\b|ryzen|\bcore\b|xeon|epyc|threadripper", name, re.I)
                                  and not re.search(r"radeon|graphics|geforce|\barc\b", name, re.I))
    if is_cpu:
        kind = "cpu"
    elif unified or re.search(r"graphics|igpu|integrated|\bUHD\b|Iris", name, re.I):
        kind = "igpu"
    else:
        kind = "gpu"
    bw, tf = FALLBACK[kind]
    return float(bw), float(tf), f"typical {kind}"


def quant_bits(quant: str | None, model_bytes: int | None = None, params: float | None = None) -> float:
    q = (quant or "").upper()
    for k, v in QUANT_BITS.items():
        if q.startswith(k):
            return float(v)
    if model_bytes and params:
        return 8.0 * model_bytes / params
    return 4.5


@dataclass
class Sample:
    device: str
    backend: str
    model_bytes: int
    tok_s: float
    quant: str | None = None
    cpu_share: float = 0.0
    flash_attn: bool = False
    spec: bool = False
    parallel: int = 1
    unified: bool = False
    source: str = ""
    weight: float = 1.0                      # how many requests the speed was averaged over (capped)


def features(s: Sample) -> list[float]:
    b = (s.backend or "").lower()
    b = "rocm" if b == "hip" else b
    return [1.0 if b == x else 0.0 for x in BACKENDS] + [
        quant_bits(s.quant) / 8.0, math.log(max(s.model_bytes, 1) / 2**30), float(s.cpu_share),
        1.0 if s.flash_attn else 0.0, 1.0 if s.spec else 0.0, math.log(max(1, s.parallel)), 1.0 if s.unified else 0.0]


def prior(s: Sample) -> float:
    """log tokens/s from bandwidth alone (layers on the CPU read at its bandwidth)."""
    bw, _, _ = spec(s.device, s.backend, s.unified)
    cbw = FALLBACK["cpu"][0]
    t = s.model_bytes * ((1 - s.cpu_share) / (bw * 1e9) + s.cpu_share / (cbw * 1e9))
    return math.log(EFFICIENCY / max(t, 1e-9))


@dataclass
class Model:
    beta: list[float] = field(default_factory=lambda: [0.0] * len(FEATURES))
    device: dict = field(default_factory=dict)          # device -> {"offset", "n"}
    sigma: float = 0.5
    sigma_device: float = 0.4
    n: int = 0
    cv_error: float | None = None                       # leave-one-device-out median |error|, as a fraction
    fit_error: float | None = None
    updated: float = 0.0

    def predict(self, s: Sample) -> dict:
        f = features(s)
        base = prior(s) + sum(b * x for b, x in zip(self.beta, f))
        d = self.device.get(s.device)
        off = d["offset"] if d else 0.0
        sd = math.sqrt(self.sigma ** 2 + (0 if d and d["n"] >= SHRINK else self.sigma_device ** 2))
        mid = math.exp(base + off)
        bw, tf, src = spec(s.device, s.backend, s.unified)
        return {"tok_s": round(mid, 1), "low": round(mid * math.exp(-1.28 * sd), 1),
                "high": round(mid * math.exp(1.28 * sd), 1), "measured_on_device": d["n"] if d else 0,
                "bandwidth_gbs": bw, "specs_from": src}

    def to_json(self) -> dict:
        return {"features": FEATURES, "beta": [round(b, 4) for b in self.beta], "device": self.device,
                "sigma": round(self.sigma, 4), "sigma_device": round(self.sigma_device, 4), "n": self.n,
                "cv_error": self.cv_error, "fit_error": self.fit_error, "updated": self.updated,
                "explain": explain(self)}

    @classmethod
    def from_json(cls, d: dict) -> "Model":
        m = cls()
        if d.get("features") == FEATURES:
            m.beta = list(d["beta"])
        m.device = d.get("device") or {}
        for k in ("sigma", "sigma_device", "n", "cv_error", "fit_error", "updated"):
            if d.get(k) is not None:
                setattr(m, k, d[k])
        return m


def _solve(rows: list[list[float]], ys: list[float], ws: list[float], lam: float = RIDGE) -> list[float]:
    import numpy as np
    X, y, w = np.array(rows), np.array(ys), np.sqrt(np.array(ws))
    Xw, yw = X * w[:, None], y * w
    return np.linalg.solve(Xw.T @ Xw + lam * np.eye(X.shape[1]), Xw.T @ yw).tolist()


def fit(samples: list[Sample]) -> Model:
    samples = [s for s in samples if s.tok_s and s.tok_s > 0 and s.model_bytes]
    m = Model(n=len(samples), updated=time.time())
    if not samples:
        return m
    res = [math.log(s.tok_s) - prior(s) for s in samples]
    rows, ws = [features(s) for s in samples], [min(s.weight, 20.0) for s in samples]
    m.beta = _solve(rows, res, ws)
    left = [r - sum(b * x for b, x in zip(m.beta, f)) for r, f in zip(res, rows)]
    by = {}
    for s, r in zip(samples, left):
        by.setdefault(s.device, []).append(r)
    m.device = {d: {"offset": round(sum(v) / len(v) * len(v) / (len(v) + SHRINK), 4), "n": len(v)} for d, v in by.items()}
    final = [r - m.device[s.device]["offset"] for s, r in zip(samples, left)]
    m.sigma = max(0.08, math.sqrt(sum(x * x for x in final) / max(1, len(final) - len(FEATURES) // 2)))
    offs = [v["offset"] / (v["n"] / (v["n"] + SHRINK)) for v in m.device.values() if v["n"] >= 2]
    m.sigma_device = max(0.15, math.sqrt(sum(o * o for o in offs) / len(offs))) if offs else 0.4
    m.fit_error = round(sorted(abs(math.expm1(x)) for x in final)[len(final) // 2], 3)
    devices = sorted(by)
    if len(devices) >= 3:                      # leave one device out: how well unseen hardware is predicted
        errs = []
        for d in devices:
            tr = [s for s in samples if s.device != d]
            te = [s for s in samples if s.device == d]
            sub = fit_no_cv(tr)
            errs += [abs(sub.predict(s)["tok_s"] / s.tok_s - 1) for s in te]
        m.cv_error = round(sorted(errs)[len(errs) // 2], 3)
    return m


def fit_no_cv(samples: list[Sample]) -> Model:
    m = Model(n=len(samples))
    res = [math.log(s.tok_s) - prior(s) for s in samples]
    m.beta = _solve([features(s) for s in samples], res, [min(s.weight, 20.0) for s in samples])
    return m


def explain(m: Model) -> list[dict]:
    """The learned corrections in words: × factor on tokens/s per unit of each feature."""
    out = []
    for name, b in zip(FEATURES, m.beta):
        if abs(b) < 0.02:
            continue
        unit = {"bits": "per 8 bits per weight", "log_gb": "per ×e in model size", "cpu_share": "all layers on the CPU",
                "log_parallel": "per ×e parallel requests"}.get(name, "when on")
        out.append({"feature": name, "factor": round(math.exp(b), 3), "unit": unit})
    return sorted(out, key=lambda x: -abs(math.log(x["factor"])))


# ---------------------------------------------------------------- sources

def samples_from_board(rows: list[dict]) -> list[Sample]:
    out = []
    for r in rows:
        for p in r.get("performance") or []:
            hw = p.get("hardware") or {}
            gpus = hw.get("gpus") or []
            ngl = p.get("ngl")
            on_cpu = not gpus or ngl == 0
            g = gpus[0] if gpus else {}
            if not p.get("gen_tok_s") or not p.get("model_bytes"):
                continue
            out.append(Sample(device=(hw.get("cpu") or "CPU") if on_cpu else g.get("name") or "GPU",
                              backend="cpu" if on_cpu else (g.get("backend") or ""), model_bytes=int(p["model_bytes"]),
                              tok_s=float(p["gen_tok_s"]), quant=p.get("quant"), flash_attn=p.get("flash_attn") in (True, "on"),
                              spec=bool(p.get("spec_type") not in (None, "", "none") or p.get("draft_model")),
                              parallel=int(p.get("parallel") or 1), unified=bool(g.get("unified")), source="board",
                              weight=float(p.get("requests") or 1)))
    return out


def samples_from_worklog(rows: list[dict], model_info, device_info) -> list[Sample]:
    """Work log rows (work_log.py) with server timings. model_info(model) -> {"bytes", "quant", "settings"} or None;
    device_info(host, device) -> {"name", "backend", "unified"} or None."""
    agg = {}
    for r in rows:
        if not r.get("gen_ms") or not r.get("completion_tokens") or not r.get("ok", True):
            continue
        k = (r.get("host") or "this computer", r.get("device") or "", r.get("model") or r.get("worker"))
        a = agg.setdefault(k, [0, 0.0, 0])
        a[0] += r["completion_tokens"]
        a[1] += r["gen_ms"] / 1000
        a[2] += 1
    out = []
    for (host, dev, model), (tok, sec, n) in agg.items():
        mi, di = model_info(model), device_info(host, dev)
        if not mi or not mi.get("bytes") or not di or sec <= 0:
            continue
        st = mi.get("settings") or {}
        out.append(Sample(device=di["name"], backend=di.get("backend") or "", model_bytes=int(mi["bytes"]),
                          tok_s=tok / sec, quant=mi.get("quant"), flash_attn=st.get("flash_attn") in (True, "on"),
                          spec=bool(st.get("draft_model") or st.get("spec_type") not in (None, "", "none")),
                          cpu_share=0.0 if di.get("backend") != "cpu" else 1.0, unified=bool(di.get("unified")),
                          source="this computer" if host == "this computer" else host, weight=float(n)))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--board", help="a board.json (file or https URL)")
    p.add_argument("--predict", nargs=3, metavar=("DEVICE", "BACKEND", "MODEL_GB"),
                   help="e.g. 'RX 7900 XTX' rocm 4.9")
    p.add_argument("--quant", default="Q4_K_M")
    p.add_argument("--show", action="store_true", help="print the fitted model")
    a = p.parse_args(argv)
    rows = []
    if a.board:
        if a.board.startswith("https://"):
            import urllib.request
            with urllib.request.urlopen(a.board, timeout=30) as r:
                rows = json.loads(r.read()).get("rows") or []
        else:
            rows = json.loads(Path(a.board).read_text()).get("rows") or []
    m = fit(samples_from_board(rows))
    if a.show or not a.predict:
        print(json.dumps(m.to_json(), indent=1))
    if a.predict:
        dev, be, gb = a.predict
        print(json.dumps(m.predict(Sample(device=dev, backend=be, model_bytes=int(float(gb) * 2**30), tok_s=0,
                                          quant=a.quant)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
