#!/usr/bin/env python3
"""
OmniCortex Studio: model manager, server supervisor and chat, over llama-server.

Scope, deliberately narrow. This does not implement inference, tokenization,
sampling or GGUF loading -- llama.cpp does all of that, better than a
reimplementation would. What this adds is the layer around it: finding models,
remembering per-model settings, starting and stopping the server, a chat
window, and the runtime controls LM Studio does not expose.

That last part is the reason to build it at all:

    LoRA scale slider     live suppression strength via /lora-adapters
    expert count          --override-kv <arch>.expert_used_count for MoE
    host fit estimate     before loading, not after it fails

Everything else here is table stakes that any front end needs.

    python viz/studio.py --models-dir ~/.models --server ~/llama.cpp/build/bin/llama-server
    python viz/studio.py --models-dir A --models-dir B --host 0.0.0.0

Stdlib only, except the optional `gguf` package for reading model metadata.
No build step, no bundled browser, and it works over the LAN because the GPU is
often on another machine.

It also exposes an OpenAI-compatible API at /v1 (models, chat, completions,
embeddings) with just-in-time loading: a request naming another model loads
it, `"model": "auto"` picks one from the subject classifier and each model's
rolling performance stats (models without stats are never auto-picked), and --idle-ttl unloads after inactivity. Vision models are paired with
their mmproj file automatically and accept image attachments in chat.

Binding beyond loopback requires --token (or NS_STUDIO_TOKEN), and TLS unless
--allow-plaintext is passed for a VPN / reverse-proxy deployment.
"""

import argparse
import glob
import hashlib
import hmac
import http.cookies
import json
import mimetypes
import os
import queue
import re
import secrets
import shutil
import socket
import subprocess
import sys
import collections
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))
import gguf_utils
import ns_security as sec
import model_stats
import jobs as ns_jobs
import rag as ns_rag
import mcp_client as ns_mcp
import context_compress as ns_compress
import quant_select as ns_qsel
import ns_pairing
import work_log as ns_work
try:
    from hostcheck import Host, check as host_check
except ImportError:
    Host, host_check = None, None

STATE = {"models_dirs": [], "server_bin": None, "settings_path": None,
         "download_dir": None, "offline": False, "token": None,
         "chats_dir": None, "stats": None, "min_graded": 20, "min_subject": 5,
         "halluc_cost": 1.0, "traces_dir": None, "cett": None, "score_ngl": 0, "score_every": 1,
         "idle_ttl": 0, "jit": True,
         "backend_port": 8080, "tls": False, "max_workers": 2, "worker_idle": 300}
ACTIVITY = {"last": time.time(), "active": 0}
ACTIVITY_LOCK = threading.Lock()
MAX_BODY = 64 * 1024 * 1024        # chat bodies may carry base64 images
MAX_SMALL_BODY = 256 * 1024
THROTTLE = sec.FailureThrottle()
JOBS = {}          # download id -> progress record
JOBS_LOCK = threading.Lock()

HF_API = "https://huggingface.co/api"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{path}"
LOCK = threading.Lock()
PROC = {"proc": None, "model": None, "port": None, "args": None,
        "started": None, "lora": None, "lora_scale": 1.0}


# ------------------------------------------------------------------ metadata

def _gguf_meta(path):
    """Architecture, parameter and quant info. Falls back to filename cues."""
    out = {"arch": None, "n_layers": None, "n_experts": None,
           "context": None, "quant": None, "params": None}
    m = re.search(r"(Q\d+_[A-Z0-9_]+|IQ\d+_[A-Z]+|F16|BF16|F32)",
                  os.path.basename(path), re.I)
    if m:
        out["quant"] = m.group(1).upper()
    try:
        import gguf
    except ImportError:
        return out
    try:
        r = gguf.GGUFReader(path)

        def kv(key):
            return gguf_utils.read_kv(r, key)

        arch = kv("general.architecture")
        out["arch"] = arch
        if arch:
            out["n_layers"] = kv(f"{arch}.block_count")
            out["n_experts"] = kv(f"{arch}.expert_count")
            out["n_experts_used"] = kv(f"{arch}.expert_used_count")
            out["context"] = kv(f"{arch}.context_length")
        out["params"] = kv("general.parameter_count") or kv("general.size_label")
    except Exception as e:
        out["error"] = str(e)[:120]
    return out


_META_CACHE = {}


MMPROJ_RE = re.compile(r"(^|[-_.])mmproj([-_.]|$)", re.I)


def is_mmproj(path):
    return bool(MMPROJ_RE.search(os.path.basename(path)))


def model_id(path):
    """Stable OpenAI-style id: <repo dir>/<file stem>, multi-part suffix removed."""
    stem = re.sub(r"-\d{5}-of-\d{5}$", "", os.path.basename(path)[:-5])
    return f"{os.path.basename(os.path.dirname(path))}/{stem}".lower()


def scan_models():
    """Every .gguf under the configured directories, with cached metadata.

    Multi-part files (`-00001-of-0000N`) are listed once, by their first part,
    which is what llama.cpp wants passed to -m. Vision projector files
    (mmproj-*.gguf) are not models; they are attached to the models in the
    same directory so vision just works when loaded.
    """
    found, seen = [], set()
    for root in STATE["models_dirs"]:
        for path in sorted(glob.glob(os.path.join(root, "**", "*.gguf"),
                                     recursive=True)):
            if is_mmproj(path):
                continue
            base = os.path.basename(path)
            part = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", base)
            if part:
                if part.group(1) != "00001":
                    continue
                key = base[:part.start()]
                if key in seen:
                    continue
                seen.add(key)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            st = os.stat(path)
            ck = (path, st.st_mtime, size)
            if ck not in _META_CACHE:
                _META_CACHE[ck] = _gguf_meta(path)
            meta = _META_CACHE[ck]
            projectors = sorted(p for p in glob.glob(os.path.join(os.path.dirname(path), "*.gguf"))
                                if is_mmproj(p))
            found.append({
                "path": path, "name": base, "size": size, "id": model_id(path),
                "publisher": os.path.basename(os.path.dirname(
                    os.path.dirname(path))),
                "mmproj": projectors[0] if projectors else None,
                **meta,
            })
    # Hugging Face checkpoint folders (config.json + safetensors): served by vLLM.
    for root in STATE["models_dirs"]:
        for cfg in sorted(glob.glob(os.path.join(root, "**", "config.json"), recursive=True)):
            d = os.path.dirname(cfg)
            weights = glob.glob(os.path.join(d, "*.safetensors"))
            if not weights or d in seen:
                continue
            seen.add(d)
            try:
                arch = json.loads(open(cfg).read()).get("model_type")
            except (OSError, ValueError):
                arch = None
            found.append({"path": d, "name": os.path.basename(d), "format": "hf", "arch": arch,
                          "size": sum(os.path.getsize(w) for w in weights),
                          "id": f"{os.path.basename(os.path.dirname(d))}/{os.path.basename(d)}".lower(),
                          "publisher": os.path.basename(os.path.dirname(d)), "mmproj": None})
    return found


def find_model(ref, models=None):
    """Resolve an id, file name, path or served alias to a scanned model."""
    if not ref:
        return None
    ref_l = ref.lower()
    st = load_settings()
    for m in models or scan_models():
        alias = (st.get(_key(m["path"]), {}).get("served_name") or "").lower()
        if ref_l in (m["id"], m["name"].lower(), m["path"].lower(), alias) or \
                ref_l == m["name"].lower()[:-5]:
            return m
    return None


_GPU_CACHE = {"t": 0.0, "gpus": []}


def nvidia_gpus():
    """NVIDIA GPUs via nvidia-smi (cached 10 s); [] elsewhere."""
    if time.time() - _GPU_CACHE["t"] > 10:
        try:
            import cuda_info
            _GPU_CACHE["gpus"] = cuda_info.query_gpus()
        except Exception:
            _GPU_CACHE["gpus"] = []
        _GPU_CACHE["t"] = time.time()
    return _GPU_CACHE["gpus"]


_ACCEL = {"t": 0.0, "devs": []}


def accel_devices() -> list:
    """GPUs and other accelerators with memory, any vendor (scripts/accelerators.py; cached 30 s)."""
    if time.time() - _ACCEL["t"] > 30:
        try:
            import accelerators
            devs = accelerators.detect(accelerators.load_config(STATE.get("hardware_path")))
            _ACCEL["devs"] = [d for d in devs if d.get("enabled", True) and d.get("backend") != "cpu"
                              and d.get("memory_total") and not d.get("twin_of")]
        except Exception:
            _ACCEL["devs"] = []
        _ACCEL["t"] = time.time()
    return _ACCEL["devs"]


def fit_estimate(size_bytes):
    """Will this load, on this machine, and where? -> {ok, level, note}, level one of
    gpu (all layers on accelerators), partial (some layers stay on the CPU), cpu (no accelerator; fits in RAM),
    no (too large). Rough by design: weights plus a modest allowance for the KV cache."""
    if not size_bytes:
        return {"ok": None, "level": None, "note": "size unknown"}
    need = int(size_bytes * 1.15)
    gib = lambda b: f"{b / 2**30:.1f}"            # noqa: E731
    ram = None
    if Host is not None:
        try:
            h = Host()
            ram = h.ram_available or h.ram
        except Exception:
            ram = None
    gpus = nvidia_gpus()
    if gpus:
        vram = sum(g["memory_free"] or g["memory_total"] for g in gpus)
        where = f"VRAM on {len(gpus)} GPUs" if len(gpus) > 1 else "VRAM"
        unified = False
    else:
        devs = accel_devices()
        dedicated = [d for d in devs if not d.get("unified")]
        shared = [d for d in devs if d.get("unified")]
        pick = dedicated or shared
        vram = sum(max(0, (d.get("memory_free") or d["memory_total"]) - (d.get("reserve") or 0)) for d in pick)
        unified = not dedicated and bool(shared)
        where = (f"memory on {len(pick)} GPUs" if len(pick) > 1 else
                 "memory shared by the GPU and the CPU" if unified else "VRAM") if pick else ""
    if vram:
        if need <= vram:
            tight = need > 0.85 * vram
            return {"ok": True, "level": "gpu", "note": f"{'tight: ' if tight else ''}full GPU offload: ~{gib(need)} "
                                                       f"of {gib(vram)} GiB {where}"}
        if not unified and ram and need <= vram + 0.9 * ram:
            return {"ok": True, "level": "partial",
                    "note": f"partial GPU offload: ~{gib(need)} GiB, {gib(vram)} GiB {where}; the rest runs on the CPU "
                            "(slower)"}
        return {"ok": False, "level": "no", "note": f"needs ~{gib(need)} GiB, {gib(vram)} GiB free {where}"
                                                    + (f" and {gib(ram)} GiB RAM" if ram and not unified else "")
                                                    + " (lower GPU layers to keep more on the CPU)"}
    if not ram:
        return {"ok": None, "level": None, "note": "cannot read memory"}
    if need > ram:
        return {"ok": False, "level": "no", "note": f"needs ~{gib(need)} GiB, {gib(ram)} GiB available"}
    return {"ok": True, "level": "cpu", "note": f"{'tight: ' if need > 0.85 * ram else ''}on the CPU: ~{gib(need)} "
                                                f"of {gib(ram)} GiB RAM"}


# ------------------------------------------------------------- hub search

def _hf_get(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": "neuronscope"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def hub_search(query, limit=20):
    """GGUF repos matching a query. Read-only, no dependency on huggingface_hub."""
    q = urllib.parse.urlencode({"search": query, "filter": "gguf",
                                "limit": limit, "sort": "downloads",
                                "direction": -1})
    out = []
    for m in _hf_get(f"{HF_API}/models?{q}"):
        out.append({"repo": m.get("modelId") or m.get("id"),
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                    "gated": bool(m.get("gated"))})
    return out


def hub_files(repo):
    """The .gguf files in a repo, with sizes where the API reports them.

    Multi-part files are grouped: downloading one part of a split model is
    useless, so the UI offers the set rather than the pieces.
    """
    info = _hf_get(f"{HF_API}/models/{urllib.parse.quote(repo)}?blobs=true")
    files = []
    for sib in info.get("siblings", []):
        name = sib.get("rfilename", "")
        if not name.endswith(".gguf"):
            continue
        files.append({"path": name, "size": sib.get("size"),
                      "quant": (re.search(r"(Q\d+_[A-Z0-9_]+|IQ\d+_[A-Z]+|"
                                          r"BF16|F16|F32)", name, re.I).group(1).upper()
                                if re.search(r"(Q\d+_[A-Z0-9_]+|IQ\d+_[A-Z]+|"
                                             r"BF16|F16|F32)", name, re.I) else None)})
    groups = {}
    for f in files:
        m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", f["path"])
        key = f["path"][:m.start()] if m else f["path"]
        g = groups.setdefault(key, {"name": key, "parts": [], "size": 0,
                                    "quant": f["quant"]})
        g["parts"].append(f["path"])
        if f["size"]:
            g["size"] += f["size"]
    for g in groups.values():
        g["parts"].sort()
        g["fit"] = fit_estimate(g["size"])
    out = sorted(groups.values(), key=lambda g: (g["size"] or 0, g["name"]))   # smallest first, as in LM Studio
    # The download to suggest: the largest file that runs fully on the GPU (or in RAM without one), skipping
    # unquantized copies, which are rarely worth their size for chatting.
    ok = [g for g in out if g["fit"].get("level") in ("gpu", "cpu") and g["quant"] not in ("F32", "F16", "BF16")
          and not str(g["name"]).lower().startswith("mmproj")]
    if ok:
        max(ok, key=lambda g: g["size"])["recommended"] = True
    return out


def _download_one(repo, path, dest, job):
    url = HF_RESOLVE.format(repo=urllib.parse.quote(repo),
                            path=urllib.parse.quote(path))
    # The directory has to exist before the .part file is opened, not just
    # before the final rename.
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    # Resume: the Hub honours Range, and these files are large enough that
    # restarting from zero after a dropped connection is not acceptable.
    start = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "neuronscope"})
    if start:
        req.add_header("Range", f"bytes={start}-")
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0) + start
        job["total"] = max(job.get("total", 0), total)
        mode = "ab" if start and r.status == 206 else "wb"
        if mode == "wb":
            start = 0
        with open(tmp, mode) as f:
            done = start
            while not job.get("cancel"):
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                job["done"] = job.get("base", 0) + done
    if job.get("cancel"):
        return False
    os.replace(tmp, dest)
    return True


def start_download(repo, group, root):
    job_id = hashlib.sha256(f"{repo}/{group['name']}".encode()).hexdigest()[:12]
    with JOBS_LOCK:
        if job_id in JOBS and JOBS[job_id]["state"] == "running":
            return job_id
        JOBS[job_id] = {"id": job_id, "repo": repo, "name": group["name"],
                        "state": "running", "done": 0,
                        "total": group.get("size") or 0, "error": None,
                        "base": 0, "cancel": False}
    job = JOBS[job_id]
    target_dir = os.path.join(root, *repo.split("/"))

    def run():
        try:
            for part in group["parts"]:
                dest = os.path.join(target_dir, os.path.basename(part))
                if os.path.exists(dest):
                    job["base"] = job.get("base", 0) + os.path.getsize(dest)
                    job["done"] = job["base"]
                    continue
                if not _download_one(repo, part, dest, job):
                    job["state"] = "cancelled"
                    return
                job["base"] = job["done"]
            job["state"] = "done"
            _META_CACHE.clear()
        except Exception as e:
            job["state"] = "error"
            job["error"] = str(e)[:200]

    threading.Thread(target=run, daemon=True).start()
    return job_id


# ------------------------------------------------------------------ settings

def _key(path):
    return hashlib.sha256(path.encode()).hexdigest()[:16]


def features_lab_call(lab: str | None, op: str, body: dict | None, timeout: float = 1800):
    """One call to the decoding lab's /ns/* endpoints (scripts/feature_lab.py). Loopback only, as the lab is."""
    url = (lab or "http://127.0.0.1:8090").rstrip("/")
    u = urllib.parse.urlsplit(url)
    if u.scheme != "http" or u.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("the decoding lab listens on this computer only: http://127.0.0.1:PORT")
    if not re.fullmatch(r"[a-z_]{2,20}", op or ""):
        raise ValueError("bad lab endpoint")
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"{url}/ns/{op}", data=data, headers={"Content-Type": "application/json"},
                                 method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def load_settings():
    p = STATE["settings_path"]
    if p and os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_settings(all_settings):
    p = STATE["settings_path"]
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(all_settings, f, indent=2)
    os.replace(tmp, p)


DEFAULTS = {"ngl": 99, "ctx": 8192, "batch": 2048, "threads": 0,
            "parallel": 1, "flash_attn": "auto", "lora": "", "lora_scale": 1.0,
            # llama (llama-server) or vllm (`vllm serve`); engine options and their
            # spellings live in scripts/engine_options.py
            "engine": "llama", "env": {}, "wrap": "",
            "experts": 0, "extra": "",
            # Speculative decoding. A small draft model proposes tokens that the
            # large one verifies in a batch, so on a bandwidth-bound host the
            # win can be large -- this is the biggest speed lever available on
            # an iGPU, where generation is limited by weight reads per token.
            "draft_model": "",
            # Load the model's mmproj projector when one sits next to it.
            "vision": True,
            # classifier.npz for this model: enables per-reply activation stats
            "classifier": "",
            # Several GPUs (a Tesla M10 is four): which ones this model may use
            # (CUDA_VISIBLE_DEVICES), how llama-server splits it (layer | row |
            # none), the per-GPU proportions ("1,1,1,1") and the main GPU.
            "gpus": "", "split_mode": "", "tensor_split": "", "main_gpu": -1}

# A named config is a complete, reusable setup: model, load settings, preset,
# visualizer and hardware limits, under a name. The name is also what the model
# is served as, so Cline sees "model-suppressed" rather than a filename.
CONFIG_DEFAULTS = {
    "name": "", "path": "", "settings": {}, "preset": "default",
    "viz": "pygfx", "viz_device": "auto",
    "cache_type": "f16", "cache_type_v": "",
    "turboquant": False,          # no implementation exists; see warnings
    "cpu_cores": 0, "gpu_reserve_gib": 0.75, "vram_gib": 0.0,
    # Which machine this config is meant for. Empty means "wherever it runs",
    # audited against the current host.
    "host": "",
}


def load_configs():
    return load_settings().get("__configs__", {})


def save_config(cfg):
    st = load_settings()
    st.setdefault("__configs__", {})[cfg["name"]] = {
        **CONFIG_DEFAULTS, **{k: v for k, v in cfg.items()
                              if k in CONFIG_DEFAULTS}}
    save_settings(st)


def delete_config(name):
    st = load_settings()
    st.get("__configs__", {}).pop(name, None)
    save_settings(st)


def host_profiles():
    try:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
        import hostprofiles as HP
        cur, _ = HP.current_profile()
        return HP.load_all(), cur
    except Exception:
        return {}, None


def audit_config(cfg, model):
    """-> [{level, field, message}]. Everything the user should see in orange.

    Warnings are attached to the field that caused them, so the UI can put the
    marker where the setting is rather than in a list at the bottom.
    """
    out = []
    s = {**DEFAULTS, **(cfg.get("settings") or {})}

    def warn(field, msg, level="warn"):
        out.append({"level": level, "field": field, "message": msg})

    if cfg.get("turboquant"):
        warn("turboquant",
             "TurboQuant has no implementation in llama.cpp. It is a research "
             "method for KV-cache compression, not a switch. Use cache_type "
             "q8_0 for the same lever at lower sophistication.", "error")

    if cfg.get("cache_type") in ("q4_0", "q4_1") or \
            cfg.get("cache_type_v") in ("q4_0", "q4_1"):
        warn("cache_type", "4-bit KV cache measurably degrades long-context "
                           "recall. q8_0 is the usual safe choice.")

    if s.get("ctx", 0) > 32768:
        warn("ctx", f"context {s['ctx']} makes the KV cache the dominant "
                    "memory cost. Check the budget below before loading.")

    if cfg.get("cpu_cores") and cfg["cpu_cores"] > (os.cpu_count() or 1):
        warn("cpu_cores", f"{cfg['cpu_cores']} threads on "
                          f"{os.cpu_count()} cores will thrash.")

    if s.get("parallel", 1) > 1 and s.get("ctx", 0) > 16384:
        warn("parallel", "concurrent slots each get ctx/parallel tokens; at "
                         "this context that may truncate requests.")

    if cfg.get("viz_device") == "gpu" and cfg.get("viz") in ("godot", "threejs"):
        warn("viz_device", "forcing the visualizer onto the GPU competes with "
                           "the model for VRAM. 'auto' evicts it to CPU when "
                           "memory is short.")

    if not model:
        return out

    # The real numbers, from the model's own architecture.
    try:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
        from vram_budget import GIB, detect_vram, plan, read_gguf_shape
        shape = read_gguf_shape(model["path"])
        # A config can name the machine it is for, so you can plan for another
        # host from this one. Falls back to whatever this machine has.
        reserve_mult = 1.0
        vram = int((cfg.get("vram_gib") or 0) * GIB)
        hosts, current = host_profiles()
        target = cfg.get("host") or current
        if not vram and target and target in hosts:
            h = hosts[target]
            vram = int((h.get("vram_gib") or 0) * GIB)
            if not h.get("dedicated"):
                reserve_mult = 1.6   # the desktop draws from the same pool
            if target != current:
                warn("host", f"planned for host '{target}', not the machine "
                             "you are on.", "info")
        vram = vram or detect_vram()
        if vram:
            r = plan(shape, s.get("ctx", 8192), vram,
                     cfg.get("viz", "pygfx"),
                     cfg.get("cache_type", "f16"),
                     cfg.get("cache_type_v") or None,
                     None,
                     (cfg.get("gpu_reserve_gib") or 0.75) * GIB * reserve_mult)
            for w in r["warnings"]:
                warn("ctx", w)
            if r["viz_device"] == "cpu" and cfg.get("viz_device") != "cpu":
                warn("viz", "not enough VRAM for model, cache and visualizer; "
                            "the visualizer will run on CPU.")
            if r["layers_on_gpu"] < r["n_layers"]:
                warn("ngl", f"only {r['layers_on_gpu']} of {r['n_layers']} "
                            f"layers fit; the rest run on CPU and generation "
                            f"will be much slower.")
            out.append({"level": "info", "field": "budget", "message":
                        f"weights {r['model_bytes'] / GIB:.1f} + KV "
                        f"{r['kv_bytes'] / GIB:.1f} + viz "
                        f"{r['viz_bytes'] / GIB:.1f} GiB, "
                        f"{r['gpu_used'] / GIB:.1f} of "
                        f"{r['budget_bytes'] / GIB:.1f} GiB usable",
                        "plan": {k: r[k] for k in
                                 ("layers_on_gpu", "n_layers", "viz_device",
                                  "fits")}})
    except Exception as e:
        warn("budget", f"could not compute the budget: {e}", "info")
    return out


# Presets are chat-time, load settings are load-time. Keeping them apart means
# switching a system prompt does not restart the server.
PRESET_DEFAULTS = {"system": "", "temperature": 0.7, "top_p": 0.95,
                   "max_tokens": 2048,
                   # Structured output: either a JSON schema or a GBNF grammar.
                   # Both are constrained decoding in llama.cpp, so they are
                   # guarantees about the output shape, not requests.
                   "json_schema": "", "grammar": ""}
BUILTIN_PRESETS = {
    "default": {},
    "deterministic": {"temperature": 0.0, "top_p": 1.0},
    "reasoning": {"temperature": 0.6, "top_p": 0.95},
    "terse": {"system": "Answer concisely. No preamble.", "temperature": 0.3},
    "json object": {"temperature": 0.2,
                    "json_schema": '{"type":"object"}'},
}


def load_presets():
    st = load_settings()
    user = st.get("__presets__", {})
    out = {k: {**PRESET_DEFAULTS, **v} for k, v in BUILTIN_PRESETS.items()}
    out.update({k: {**PRESET_DEFAULTS, **v} for k, v in user.items()})
    return out


def save_preset(name, cfg):
    st = load_settings()
    st.setdefault("__presets__", {})[name] = {
        k: v for k, v in cfg.items() if k in PRESET_DEFAULTS}
    save_settings(st)


# -------------------------------------------------------------------- server

def server_running():
    p = PROC["proc"]
    return p is not None and p.poll() is None


def port_busy(port) -> bool:
    import socket
    with socket.socket() as sk:
        sk.settimeout(0.5)
        return sk.connect_ex(("127.0.0.1", port)) == 0


def wait_healthy(port, proc, timeout=600):
    url = f"http://127.0.0.1:{port}/health"
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            return False, f"exited with code {proc.returncode}"
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True, "ready"
        except Exception:
            pass
        time.sleep(1)
    return False, f"not healthy after {timeout}s"


def start_server(model, settings, port):
    if (settings.get("engine") or "llama") == "llama" and not STATE["server_bin"]:
        return False, "no llama-server set: Setup -> Build llama.cpp, or point Settings -> Workspaces at one"
    stop_server()
    if port_busy(port):
        # Something else (often an orphaned llama-server) holds the port and would
        # answer our health check while our own server fails to bind.
        return False, f"port {port} is already in use; stop whatever holds it or pass --backend-port"
    try:
        cmd, env = server_cmd(model, settings, port)
    except ValueError as e:
        return False, str(e)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, env=env)
    ok, why = wait_healthy(port, proc)
    if not ok:
        err = ""
        try:
            proc.terminate()
            err = (proc.stderr.read() or b"").decode(errors="replace")[-800:]
        except Exception:
            pass
        return False, f"{why}\n{err}"
    PROC.update({"proc": proc, "model": model, "port": port,
                 "args": cmd, "started": time.time(),
                 "lora": settings.get("lora") or None,
                 "lora_scale": settings.get("lora_scale", 1.0)})
    return True, "ready"


def engine_settings(settings: dict) -> dict:
    """Studio's settings as engine_options keys, mapping the older ones: flash_attn
    True/False, a config's cache_type, draft limits only when drafting."""
    s = dict(settings)
    fa = s.get("flash_attn")
    if fa is True:
        s["flash_attn"] = "on"
    elif fa is False:
        s["flash_attn"] = "off"
    if s.get("cache_type") and not s.get("cache_type_k"):
        s["cache_type_k"] = s["cache_type"]
        s.setdefault("cache_type_v", s.get("cache_type_v") or s["cache_type"])
    if not s.get("cache_type_v"):
        s.pop("cache_type_v", None)
    drafting = s.get("draft_model") or (s.get("spec_type") not in (None, "", "none"))
    if not drafting:
        for k in ("draft_max", "draft_min", "draft_p_min", "draft_ngl", "draft_device"):
            s.pop(k, None)
    return s


def server_cmd(model, settings, port):
    """The engine's argv and environment for `model` with `settings`.
    Raises ValueError on a setting that is not safe to pass through."""
    import engine_options as eo
    es = engine_settings(settings)
    engine = settings.get("engine") or "llama"
    alias = settings.get("served_name") or model.get("id") or model["name"]
    if engine == "vllm":
        cmd = vllm_cmd(model, es, port, alias)
    elif engine != "llama":
        raise ValueError(f"unknown engine {engine}")
    else:
        if model.get("format") == "hf":
            raise ValueError("llama-server needs a GGUF. Serve this checkpoint with the vLLM engine, or convert it "
                             "(Jobs → Merge and convert)")
        cmd = [STATE["server_bin"], "-m", model["path"],
               "-ngl", str(settings["ngl"]), "-c", str(settings["ctx"]),
               "-b", str(settings["batch"]),
               "--port", str(port), "--host", "127.0.0.1"]
        if settings.get("threads"):
            cmd += ["-t", str(settings["threads"])]
        elif settings.get("cpu_cores"):
            cmd += ["-t", str(settings["cpu_cores"])]
        if settings.get("parallel", 1) > 1:
            es["parallel"] = settings["parallel"]
        else:
            es.pop("parallel", None)
        args, notes = eo.build_args(es, "llama", eo.probe(STATE["server_bin"], "llama"))
        cmd += args
        STATE["engine_notes"] = notes
        # What Cline and every other client sees in the model list.
        cmd += ["--alias", alias]
        if model.get("mmproj") and settings.get("vision", True):
            cmd += ["--mmproj", model["mmproj"]]
        if settings.get("lora"):
            cmd += ["--lora-scaled", settings["lora"],
                    str(settings.get("lora_scale", 1.0))]
        # MoE only, and only when the model declares experts. The override key is
        # architecture-prefixed, so it is read from the file rather than assumed.
        if settings.get("experts") and model.get("n_experts"):
            arch = model.get("arch")
            if arch:
                cmd += ["--override-kv",
                        f"{arch}.expert_used_count=int:{settings['experts']}"]
        if settings.get("split_mode"):
            if settings["split_mode"] not in ("layer", "row", "none", "tensor"):
                raise ValueError("split mode must be layer, row, tensor or none")
            cmd += ["-sm", settings["split_mode"]]
        ts = str(settings.get("tensor_split") or "").replace(" ", "")
        if ts:
            if not re.fullmatch(r"\d+(\.\d+)?(,\d+(\.\d+)?)*", ts):
                raise ValueError("tensor split must look like 1,1,1,1 or 3,1")
            cmd += ["-ts", ts]
        mg = settings.get("main_gpu")
        if mg not in (None, "") and int(mg) >= 0:
            cmd += ["-mg", str(int(mg))]
    env = None
    extra_env = eo.build_env(settings)
    gpus = str(settings.get("gpus") or "").replace(" ", "")
    if gpus:
        if not re.fullmatch(r"\d+(,\d+)*", gpus):
            raise ValueError("visible GPUs must look like 0,1,2")
        extra_env["CUDA_VISIBLE_DEVICES"] = gpus
    if extra_env:
        env = {**os.environ, **extra_env}
    if settings.get("extra"):
        cmd += settings["extra"].split()
    return eo.wrapper(settings) + cmd, env


def vllm_cmd(model, es, port, alias):
    """`vllm serve` for a Hugging Face checkpoint folder (or a GGUF, which vLLM
    reads experimentally and needs the original tokenizer for)."""
    import engine_options as eo
    import shlex
    exe = STATE.get("vllm_bin") or "vllm"
    if not shutil.which(shlex.split(exe)[0]):
        raise ValueError("vLLM is not installed (pip install vllm, or set its command under Setup)")
    cmd = [*shlex.split(exe), "serve", model["path"], "--host", "127.0.0.1", "--port", str(port),
           "--served-model-name", alias]
    if model["path"].endswith(".gguf"):
        tok = es.get("tokenizer") or model.get("tokenizer")
        if not tok:
            raise ValueError("vLLM needs the original model's tokenizer for a GGUF (set Tokenizer)")
        cmd += ["--tokenizer", str(tok)]
    if not es.get("max_model_len") and es.get("ctx"):
        es["max_model_len"] = es["ctx"]
    args, notes = eo.build_args(es, "vllm", None)
    STATE["engine_notes"] = notes
    return cmd + args


def stop_server():
    p = PROC["proc"]
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
        time.sleep(1)
    PROC.update({"proc": None, "model": None, "port": None, "args": None,
                 "started": None, "lora": None})


# ---------------------------------------------------------------- linked hosts

# What a paired device may POST. Everything else needs the owner's token.
DEVICE_POSTS = {"/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/api/chat", "/api/chats",
                "/api/chats/delete", "/api/rag/search", "/api/tools/approve", "/api/trace",
                "/mcp", "/api/route", "/api/context"}

LINKS = {"cache": {}, "lock": threading.Lock()}
LINK_SEP = ":"


def load_links() -> list[dict]:
    try:
        return json.loads(open(STATE["links_path"]).read()).get("links", [])
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return []


def save_links(links: list[dict]) -> None:
    os.makedirs(os.path.dirname(STATE["links_path"]), exist_ok=True)
    sec.write_secret_file(Path(STATE["links_path"]), json.dumps({"links": links}, indent=1))


REMOTE: dict = {"obj": None}


def remote():
    """Remote hosts: computers that run models and the OmniCortex Remote agent, not Studio."""
    if REMOTE["obj"] is None:
        import remote_hosts
        REMOTE["obj"] = remote_hosts.Remote(load_links, save_links)
    return REMOTE["obj"]


# ---- system monitor, busy computers and automation rules (System page; scripts/automation.py)

import automation as ns_auto  # noqa: E402

SYSMON = {"lock": threading.Lock(), "want": 0.0, "thread": None, "local": collections.deque(maxlen=1800),
          "hosts": {}, "presence": {}, "cpu_prev": None, "t_presence": 0.0, "t_hosts": 0.0}
BUSY: dict = {}            # target ("local" or "host:<name>") -> {why, since, until, source, actions}
AUTO = {"lock": threading.Lock(), "results": {}, "applied": {}, "log": collections.deque(maxlen=300),
        "samples": {}, "history_at": {}, "throttled": {}, "thread": None}
BUSY_PREFIX = "HOST BUSY:"


def worker_target(wid: str | None) -> str | None:
    """Which computer a worker runs on: "local", "host:<name>", or None (an endpoint)."""
    if not wid or wid.startswith("ep:"):
        return None
    if worker_kind(wid) == "remote":
        return "host:" + wid.partition(LINK_SEP)[0]
    return "local"


def target_busy(target: str | None) -> dict | None:
    b = BUSY.get(target) if target else None
    if b and b.get("until") and time.time() >= b["until"] and b.get("source") == "api":
        BUSY.pop(target, None)
        return None
    return b


def _cpu_percent():
    try:
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        idle, total = v[3] + (v[4] if len(v) > 4 else 0), sum(v)
        prev, SYSMON["cpu_prev"] = SYSMON["cpu_prev"], (idle, total)
        if prev is None or total == prev[1]:
            return None
        return round(100 * (1 - (idle - prev[0]) / (total - prev[1])), 1)
    except (OSError, ValueError, IndexError):
        try:
            import psutil
            return psutil.cpu_percent(None)
        except Exception:
            return None


def _ram():
    if Host is None:
        return None, None
    try:
        h = Host()
        if h.ram:
            return round((h.ram - (h.ram_available or 0)) / 2**30, 2), round(h.ram / 2**30, 2)
    except Exception:
        pass
    return None, None


def sample_local() -> dict:
    import remote_hosts
    used, total = _ram()
    gpus = [{k: g.get(k) for k in ("id", "name", "vendor", "util", "temp_c", "junction_c", "vram_used_mb", "vram_total_mb",
                                   "power_w")} for g in remote_hosts.nsr.gpu_telemetry()]
    return {"t": time.time(), "cpu": _cpu_percent(), "ram_used_gb": used, "ram_total_gb": total, "gpus": gpus}


def _sysmon_loop():
    """While the System page is open or automation rules are on: this computer every 2 s, its presence every
    5 s, every host every 10 s."""
    import remote_hosts  # noqa: F401  (puts the agent package on the path)
    from neuronscope_remote import presence as ns_presence
    while True:
        wanted = time.time() - SYSMON["want"] < 60 or any(r.get("enabled") for r in automation_config()["rules"])
        if not wanted:
            time.sleep(2)
            continue
        try:
            row = sample_local()
            with SYSMON["lock"]:
                SYSMON["local"].append(row)
            now = time.time()
            if now - SYSMON["t_presence"] >= 5:
                SYSMON["t_presence"] = now
                SYSMON["presence"]["local"] = ns_presence.snapshot()
            if now - SYSMON["t_hosts"] >= 10:
                SYSMON["t_hosts"] = now
                try:
                    hosts = remote().hosts()
                except Exception:
                    hosts = []
                for h in hosts:
                    try:
                        tel = remote().call(h["name"], "telemetry", timeout=8)
                        rowh = {"t": now, "gpus": tel.get("gpus") or [], "busy": tel.get("busy")}
                        SYSMON["presence"]["host:" + h["name"]] = tel.get("presence") or {}
                        b = tel.get("busy")
                        if b and b.get("source") == "owner":
                            BUSY["host:" + h["name"]] = {"why": b.get("why"), "since": b.get("since"), "source": "host",
                                                         "until": None, "actions": ["busy"]}
                        elif (BUSY.get("host:" + h["name"]) or {}).get("source") == "host":
                            BUSY.pop("host:" + h["name"], None)
                    except Exception as e:
                        rowh = {"t": now, "error": str(e)[:200]}
                    with SYSMON["lock"]:
                        SYSMON["hosts"].setdefault(h["name"], collections.deque(maxlen=360)).append(rowh)
        except Exception as e:
            print(f"[studio] monitor: {e}", file=sys.stderr)
        time.sleep(2)


def sysmon_start():
    SYSMON["want"] = time.time()
    if SYSMON["thread"] is None:
        SYSMON["thread"] = threading.Thread(target=_sysmon_loop, daemon=True)
        SYSMON["thread"].start()
    if AUTO["thread"] is None:
        AUTO["thread"] = threading.Thread(target=automation_loop, daemon=True)
        AUTO["thread"].start()


def sysmon_view(points: int = 300) -> dict:
    with SYSMON["lock"]:
        loc = list(SYSMON["local"])
        hosts = {k: list(v) for k, v in SYSMON["hosts"].items()}
    step = max(1, len(loc) // points)
    return {"local": loc[::step][-points:], "hosts": {k: v[-points:] for k, v in hosts.items()},
            "presence": SYSMON["presence"], "busy": BUSY, "computer": socket.gethostname()}


def _auto_path() -> str:
    return os.path.join(os.path.dirname(os.path.expanduser(STATE.get("settings_path") or "~/.neuronscope/studio.json")),
                        "automation.json")


def automation_config() -> dict:
    try:
        with open(_auto_path()) as f:
            d = json.load(f)
    except (OSError, ValueError):
        d = {}
    return {"rules": d.get("rules") or [], "ha": d.get("ha") or {}, "manual": d.get("manual") or {}}


def automation_save(cfg: dict) -> None:
    p = _auto_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=1)
    os.chmod(tmp, 0o600)                     # the Home Assistant token lives here
    os.replace(tmp, p)


def _ha():
    ha = automation_config()["ha"]
    tok = ha.get("token") or ""
    if ha.get("token_file"):
        try:
            tok = open(os.path.expanduser(ha["token_file"])).read().strip()
        except OSError:
            tok = ""
    return ns_auto.HomeAssistant(ha["url"], tok) if ha.get("url") and tok else None


def _power_inputs(rules: list, now: float) -> dict:
    """Readings for each power rule's sensor: Home Assistant's history since the period began (refreshed
    every minute), plus its current state every pass."""
    ha = _ha()
    out = {}
    if ha is None:
        return out
    for r in rules:
        w = r["when"]
        if w["kind"] != "power" or not r.get("enabled"):
            continue
        ent = w["entity"]
        start, _ = ns_auto.period_bounds(w["period"], now, w.get("reset_day", 1), w.get("week_start", 0))
        buf = AUTO["samples"].setdefault(ent, [])
        try:
            if w["period"] not in ("second", "minute") and now - AUTO["history_at"].get(ent, 0) > 60:
                AUTO["history_at"][ent] = now
                hist = ha.history(ent, start - 3600)
                buf[:] = sorted(set(buf) | set(hist))
            st = ha.state(ent)
            if st["value"] is not None:
                buf.append((now, st["value"]))
            out[ent] = {"value": st["value"], "unit": st["unit"], "samples": buf}
        except Exception as e:
            out[ent] = {"error": str(e)[:200], "samples": buf}
        del buf[:-5000]
    return out


def _throttle(target: str, on: bool) -> None:
    reg = workers_registry()
    wids = [w["id"] for w in all_workers() if worker_target(w["id"]) == target]
    for wid in wids:
        if on:
            AUTO["throttled"].setdefault(wid, reg.control(wid).get("max_parallel"))
            reg.set_control(wid, max_parallel=1)
        elif wid in AUTO["throttled"]:
            prev = AUTO["throttled"].pop(wid)
            reg.set_control(wid, max_parallel=prev or 99)


def apply_target(target: str, want: dict | None, source: str = "rule") -> None:
    """Make `target` match what the rules (or the API) want; only changes are acted on."""
    have = AUTO["applied"].get(target)
    acts = set((want or {}).get("actions") or [])
    why = "; ".join((want or {}).get("reasons") or []) or (want or {}).get("why") or ""
    if want and not have:
        if acts & {"busy", "requeue", "unload"}:
            BUSY[target] = {"why": why, "since": time.time(), "until": want.get("until"), "source": source,
                            "actions": sorted(acts)}
        if "throttle" in acts:
            _throttle(target, True)
        if target == "local" and "unload" in acts:
            stop_server()
            POOL.stop()
        if target.startswith("host:") and acts & {"busy", "unload", "requeue"}:
            try:
                remote().call(target[5:], "set_busy", {"busy": True, "reason": why[:300], "until": want.get("until"),
                                                       "unload": "unload" in acts}, timeout=15)
            except Exception as e:
                AUTO["log"].appendleft({"t": time.time(), "target": target, "what": "could not tell the host", "detail": str(e)[:200]})
        AUTO["applied"][target] = {"actions": sorted(acts), "why": why, "since": time.time(), "until": want.get("until")}
        AUTO["log"].appendleft({"t": time.time(), "target": target, "what": "applied " + ", ".join(sorted(acts)), "detail": why})
    elif have and not want:
        if (BUSY.get(target) or {}).get("source") in (source, "rule", "api"):
            BUSY.pop(target, None)
        if "throttle" in have["actions"]:
            _throttle(target, False)
        if target.startswith("host:"):
            try:
                remote().call(target[5:], "set_busy", {"busy": False}, timeout=15)
            except Exception:
                pass
        AUTO["applied"].pop(target, None)
        AUTO["log"].appendleft({"t": time.time(), "target": target, "what": "released", "detail": have.get("why", "")})


def automation_tick(now: float | None = None) -> dict:
    now = now or time.time()
    cfg = automation_config()
    rules = [r for r in cfg["rules"] if r.get("enabled")]
    inputs = {"presence": dict(SYSMON["presence"]), "power": _power_inputs(rules, now), "manual": cfg["manual"]}
    with AUTO["lock"]:
        prev = AUTO["results"]
        res = {r["id"]: ns_auto.evaluate(r, inputs, now, prev.get(r["id"])) for r in rules}
        AUTO["results"] = res
        combined = ns_auto.combine(rules, res)
        targets = set(combined) | {t for t, a in AUTO["applied"].items() if a.get("source", "rule") == "rule"}
        for t in targets:
            if (AUTO["applied"].get(t) or {}).get("source") == "api":
                continue                                  # set by hand through the API: released by hand
            apply_target(t, combined.get(t))
    return res


def automation_loop():
    while True:
        try:
            if any(r.get("enabled") for r in automation_config()["rules"]):
                automation_tick()
        except Exception as e:
            print(f"[studio] automation: {e}", file=sys.stderr)
        time.sleep(10)


def automation_action(target: str, action: str, minutes: float = 0, reason: str = "") -> dict:
    """The automation API: what Home Assistant automations, scripts or the System page ask for directly."""
    if target != "local" and not target.startswith("host:"):
        raise ValueError("target: local or host:<name>")
    reason = reason or f"set through the automation API ({action})"
    until = time.time() + minutes * 60 if minutes and minutes > 0 else None
    with AUTO["lock"]:
        if action in ("busy", "unload", "throttle"):
            cur = AUTO["applied"].get(target)
            if cur:
                apply_target(target, None)
            apply_target(target, {"actions": [action] if action != "unload" else ["unload", "busy"], "why": reason,
                                  "until": until}, source="api")
            AUTO["applied"][target]["source"] = "api"
        elif action in ("free", "unthrottle"):
            apply_target(target, None, source="api")
            BUSY.pop(target, None)
        else:
            raise ValueError("action: busy, unload, throttle, free or unthrottle")
    return {"busy": BUSY.get(target), "applied": AUTO["applied"].get(target)}


# ---- setting profiles: any number of named settings per tool, and one saved default

PROFILE_SCOPE = re.compile(r"^[A-Za-z0-9_.:@/-]{1,120}$")
PROFILE_NAME = re.compile(r"^[^\x00-\x1f/\\]{1,80}$")
_PLOCK = threading.Lock()


def _profiles_path() -> str:
    return os.path.join(os.path.dirname(os.path.expanduser(STATE.get("settings_path") or "~/.neuronscope/studio.json")),
                        "profiles.json")


def profiles_load() -> dict:
    try:
        with open(_profiles_path()) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def profiles_get(scope: str) -> dict:
    sc = profiles_load().get(scope) or {}
    return {"scope": scope, "profiles": sc.get("profiles") or {}, "default": sc.get("default")}


def profiles_change(scope: str, action: str, name: str = "", values=None, new_name: str = "") -> dict:
    if not PROFILE_SCOPE.match(scope):
        raise ValueError("bad scope")
    if action != "clear_default" and not PROFILE_NAME.match(name or ""):
        raise ValueError("a profile needs a name (up to 80 characters, no slashes)")
    with _PLOCK:
        allp = profiles_load()
        sc = allp.setdefault(scope, {"profiles": {}, "default": None})
        prof = sc.setdefault("profiles", {})
        if action == "save":
            if not isinstance(values, dict):
                raise ValueError("values must be an object")
            if len(json.dumps(values)) > 200_000:
                raise ValueError("profile too large")
            prof[name] = values
        elif action == "delete":
            prof.pop(name, None)
            if sc.get("default") == name:
                sc["default"] = None
        elif action == "rename":
            if name not in prof or not PROFILE_NAME.match(new_name or ""):
                raise ValueError("rename needs an existing profile and a new name")
            prof[new_name] = prof.pop(name)
            if sc.get("default") == name:
                sc["default"] = new_name
        elif action == "default":
            if name not in prof:
                raise ValueError(f"no profile {name!r}")
            sc["default"] = name
        elif action == "clear_default":
            sc["default"] = None
        else:
            raise ValueError("action: save, delete, rename, default or clear_default")
        p = _profiles_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(allp, f, indent=1)
        os.replace(tmp, p)
    return profiles_get(scope)


# ---- jobs on remote hosts (the agent's run_job / jobs / job_log / cancel_job tools)

_RCACHE: dict = {}
_RLOCK = threading.Lock()


def _cached(key: str, ttl: float, fn, fail_ttl: float = 30):
    """fn() at most every ttl seconds; a failure is remembered for fail_ttl, so an offline host does not stall
    every page refresh. -> (value, error)"""
    now = time.time()
    with _RLOCK:
        hit = _RCACHE.get(key)
    if hit and now - hit[0] < (ttl if hit[2] is None else fail_ttl):
        return hit[1], hit[2]
    try:
        val, err = fn(), None
    except Exception as e:
        val, err = None, str(e)[:300]
    with _RLOCK:
        _RCACHE[key] = (now, val, err)
    return val, err


def job_targets() -> list[dict]:
    """Where a job can run: this computer, and every linked host whose owner allows jobs."""
    out = [{"id": "local", "label": f"This computer ({socket.gethostname()})", "ok": STATE.get("jobs") is not None}]
    try:
        hosts = remote().hosts()
    except Exception:
        hosts = []
    for rec in hosts:
        name = rec["name"]
        info, err = _cached("jobs_info:" + name, 30, lambda: remote().call(name, "jobs_info", {}, timeout=8))
        ok = bool(info and info.get("allowed") and info.get("neuronscope_dir"))
        why = err and ("unreachable: " + err) or (None if ok else "its owner has not allowed jobs (agent console → "
                                                  "What controllers may do)" if info and not info.get("allowed")
                                                  else "no OmniCortex folder on that host")
        out.append({"id": "host:" + name, "label": f"{name} (remote host)", "ok": ok, "why": why,
                    "kinds": (info or {}).get("kinds")})
    return out


def remote_jobs() -> list[dict]:
    out = []
    for t in job_targets()[1:]:
        if not t["ok"]:
            continue
        name = t["id"][5:]
        res, err = _cached("jobs:" + name, 3, lambda: remote().call(name, "jobs", {}, timeout=6), fail_ttl=15)
        for j in (res or {}).get("jobs", []):
            out.append({**j, "host": name})
    return out


def link_models(refresh: bool = False) -> list[dict]:
    """Models of every linked Studio and remote host, as '<link>:<remote id>' (cached 30 s)."""
    out = []
    for ln in load_links():
        with LINKS["lock"]:
            hit = LINKS["cache"].get(ln["name"])
        if hit and not refresh and time.time() - hit[0] < 30:
            out += hit[1]
            continue
        if ln.get("expires") and ln["expires"] <= time.time():
            with LINKS["lock"]:
                LINKS["cache"][ln["name"]] = (time.time(), [], "access expired " + time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(ln["expires"])) + "; ask that host for a new pairing link")
            continue
        try:
            if ln.get("kind") == "agent":
                ms = [{"id": f"{ln['name']}{LINK_SEP}{m['id']}", "object": "model", "owned_by": f"remote:{ln['name']}",
                       "remote_id": m["id"], "link": ln["name"], "vision": False, "loaded": m.get("loaded"),
                       "backend": m.get("backend")}
                      for m in remote().models(ln) if m.get("type") not in ("embeddings", "embedding")]
            else:
                r = ns_pairing.request(ln["url"], "GET", "/v1/models", token=ln["token"],
                                       fingerprint=ln.get("fingerprint", ""), timeout=5)
                ms = [{"id": f"{ln['name']}{LINK_SEP}{m['id']}", "object": "model", "owned_by": f"link:{ln['name']}",
                       "remote_id": m["id"], "link": ln["name"], "vision": m.get("vision", False)}
                      for m in r.get("data", []) if m.get("id") != "auto"]
            err = None
        except Exception as e:
            ms, err = [], f"{type(e).__name__}: {e}"[:200]
        with LINKS["lock"]:
            LINKS["cache"][ln["name"]] = (time.time(), ms, err)
        out += ms
    return out


def link_target(model_id: str | None):
    """-> (upstream dict, remote model id) when model_id names a linked host's model."""
    if not model_id or LINK_SEP not in model_id:
        return None
    name, _, rid = model_id.partition(LINK_SEP)
    ln = next((x for x in load_links() if x["name"] == name), None)
    if ln is None:
        return None
    if ln.get("kind") == "agent":
        return {**remote().upstream(ln), "name": name}, rid
    return {"base": ln["url"], "token": ln["token"], "fingerprint": ln.get("fingerprint", ""), "name": name}, rid


def _upstream_stream(upstream: dict, path: str, payload: dict):
    r = ns_pairing.request(upstream["base"], "POST", path, payload, token=upstream["token"],
                           fingerprint=upstream["fingerprint"], timeout=900, stream=True)
    if r.status >= 400:
        raise RuntimeError(f"linked host: HTTP {r.status}: {r.read()[:300].decode(errors='replace')}")
    return r


# ---------------------------------------------------------------- MCP tools

APPROVALS: dict = {}            # call key -> {"event": Event, "allow": bool}
PENDING_CALLS: dict = {}        # call key -> {server, tool, args}: what an "always" button turns into a rule
APPROVALS_LOCK = threading.Lock()
MAX_TOOL_ROUNDS = 8


def tool_rules():
    import tool_rules as tr
    return tr.Rules(STATE.get("tool_rules_path"))


def fs_places():
    """Shortcuts for the folder browser: home and its usual folders, Studio's model and download
    folders, LM Studio's and Hugging Face's model folders, and drives or mounted disks."""
    import platform
    home = os.path.expanduser("~")
    out = [{"label": "Home", "path": home}]
    for n in ("Desktop", "Documents", "Downloads"):
        p = os.path.join(home, n)
        if os.path.isdir(p):
            out.append({"label": n, "path": p})
    for d in STATE.get("models_dirs") or []:
        out.append({"label": "Models: " + (os.path.basename(d.rstrip("/\\")) or d), "path": d, "kind": "models"})
    if STATE.get("download_dir"):
        out.append({"label": "Downloads (models)", "path": STATE["download_dir"], "kind": "models"})
    for label, p in (("LM Studio models", os.path.join(home, ".lmstudio", "models")),
                     ("LM Studio models (old)", os.path.join(home, ".cache", "lm-studio", "models")),
                     ("Hugging Face cache", os.path.join(home, ".cache", "huggingface", "hub")),
                     ("OmniCortex", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ("Studio's settings", os.path.dirname(
                         os.path.expanduser(STATE.get("config_path") or "~/.neuronscope/config.json")))):
        if os.path.isdir(p):
            out.append({"label": label, "path": p})
    if platform.system() == "Windows":
        import string
        for c in string.ascii_uppercase:
            if os.path.exists(f"{c}:\\"):
                out.append({"label": f"Drive {c}:", "path": f"{c}:\\", "kind": "drive"})
    else:
        out.append({"label": "Computer (/)", "path": "/", "kind": "drive"})
        for base in ("/run/media", "/media", "/mnt", "/Volumes"):
            try:
                tops = sorted((e for e in os.scandir(base) if e.is_dir()), key=lambda e: e.name)
            except OSError:
                continue
            for e in tops:
                # /run/media/<user>/<disk> and /media/<user>/<disk>: the disks are one level down
                disks = [e]
                if base in ("/run/media", "/media") and not os.path.ismount(e.path):
                    try:
                        disks = sorted((x for x in os.scandir(e.path) if x.is_dir()), key=lambda x: x.name) or [e]
                    except OSError:
                        disks = [e]
                for x in disks:
                    out.append({"label": f"{x.name} ({base})", "path": x.path, "kind": "drive"})
    seen, uniq = set(), []
    for x in out:
        k = os.path.normcase(os.path.abspath(x["path"]))
        if k not in seen and os.path.isdir(x["path"]):
            seen.add(k)
            uniq.append(x)
    return uniq


def fs_list(path, want="any", exts=(), hidden=False, limit=3000):
    """One folder for the browser: its subfolders, and files when want is file or any (filtered by
    extension). -> {path, parent, sep, entries: [{name, path, dir, size, mtime}], truncated}."""
    p = os.path.abspath(os.path.expanduser(path or "~"))
    while not os.path.isdir(p):
        up = os.path.dirname(p)
        if up == p:
            break
        p = up
    exts = tuple(e.lower() if e.startswith(".") else "." + e.lower() for e in exts if e)
    rows, truncated = [], False
    try:
        it = list(os.scandir(p))
    except OSError as e:
        return {"path": p, "parent": os.path.dirname(p) if os.path.dirname(p) != p else None, "sep": os.sep,
                "entries": [], "error": f"cannot open this folder: {e.strerror or e}"}
    for e in it:
        if not hidden and (e.name.startswith(".") or e.name in ("$RECYCLE.BIN", "System Volume Information")):
            continue
        try:
            is_dir = e.is_dir()
        except OSError:
            continue
        if not is_dir and (want == "dir" or (exts and not e.name.lower().endswith(exts))):
            continue
        try:
            st = e.stat()
            size, mt = (None if is_dir else st.st_size), st.st_mtime
        except OSError:
            size = mt = None
        rows.append({"name": e.name, "path": e.path, "dir": is_dir, "size": size, "mtime": mt})
        if len(rows) >= limit:
            truncated = True
            break
    rows.sort(key=lambda r: (not r["dir"], r["name"].lower()))
    parent = os.path.dirname(p)
    return {"path": p, "parent": parent if parent != p else None, "sep": os.sep, "entries": rows,
            "truncated": truncated, "writable": os.access(p, os.W_OK)}


def plugins():
    """Studio's plugin manager (viz/plugins.py): the plugins folder in Studio's home, and the
    repository's plugins/ examples. None of them runs until the owner turns it on."""
    pm = STATE.get("plugins")
    if pm is None:
        import plugins as ns_plugins
        home = os.path.dirname(os.path.expanduser(STATE.get("config_path") or "~/.neuronscope/config.json"))
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dirs = STATE.get("plugin_dirs") or [os.path.join(home, "plugins"), os.path.join(repo, "plugins")]
        pm = STATE["plugins"] = ns_plugins.Plugins(dirs, os.path.join(home, "plugins.json"),
                                                   bundled=[os.path.join(repo, "plugins")])
    return pm


ns_jobs.EXTRA = lambda: plugins().job_specs()


def addon_builtins():
    """Built-in add-ons (viz/addons.py): which parts of Studio the owner removed."""
    import addons

    def save(v):
        STATE["addons_off"] = list(v)
        save_studio_config({"addons_off": list(v)})
    return addons.Builtins(lambda: STATE.get("addons_off") or [], save)


def addon_packages():
    import addons
    home = os.path.dirname(os.path.expanduser(STATE.get("config_path") or "~/.neuronscope/config.json"))
    pm = plugins()
    install_dir = pm.dirs[0] if pm.dirs else os.path.join(home, "plugins")

    def save(v):
        STATE["addon_repos"] = list(v)
        save_studio_config({"addon_repos": list(v)})
    return addons.Packages(pm, install_dir, lambda: STATE.get("addon_repos") or [], save)


ns_jobs.HIDDEN = lambda kind, spec: addon_builtins().job_hidden(kind, spec)


def mcp_hub():
    hub = STATE.get("mcp")
    if hub is None and (STATE.get("mcp_config") or plugins().mcp_servers()):
        hub = STATE["mcp"] = ns_mcp.MCPHub(STATE.get("mcp_config"))
        hub.extra = lambda: plugins().mcp_servers()
        hub.connect()
    return hub


def connectors_path() -> str:
    """The mcp.json the Connectors page edits: the one in use, else where Studio looks for one."""
    return (STATE.get("mcp_config") or STATE.get("mcp_config_path")
            or os.path.join(os.path.dirname(STATE["settings_path"]), "mcp.json"))


def connectors() -> dict:
    path = connectors_path()
    try:
        cfg, err = ns_mcp.read_config(path), None
    except (OSError, ValueError) as e:
        cfg, err = {"mcpServers": {}}, f"{path}: {e}"
    hub = STATE.get("mcp")
    live = {x["name"]: x for x in (hub.status() if hub else [])}
    out = []
    for name, spec in cfg.get("mcpServers", {}).items():
        if not isinstance(spec, dict):
            continue
        st = live.get(ns_mcp._safe(name)) or {}
        out.append({"name": name, "spec": ns_mcp.public_spec(spec), "transport": "http" if spec.get("url") else "stdio",
                    "disabled": bool(spec.get("disabled")), "connected": bool(st.get("connected")),
                    "error": st.get("error"), "tools": [t["name"] for t in st.get("tools") or []]})
    plug = []
    try:
        for name, spec in (plugins().mcp_servers() or {}).items():
            st = live.get(ns_mcp._safe(name)) or {}
            plug.append({"name": name, "plugin": True, "transport": "http" if spec.get("url") else "stdio",
                         "connected": bool(st.get("connected")), "error": st.get("error"),
                         "tools": [t["name"] for t in st.get("tools") or []]})
    except Exception:
        pass
    return {"config": path, "exists": os.path.exists(path), "error": err, "servers": out, "plugins": plug}


def connectors_changed() -> None:
    """After mcp.json changed: use it, and reconnect every server."""
    path = connectors_path()
    STATE["mcp_config"] = path
    hub = STATE.get("mcp")
    if hub is not None and str(hub.config_path) != os.path.expanduser(path):
        hub = STATE["mcp"] = None
    hub = mcp_hub()
    if hub is not None:
        hub.connect(timeout=30)


def wait_approval(key: str, timeout: float = 300) -> bool:
    ev = threading.Event()
    with APPROVALS_LOCK:
        APPROVALS[key] = {"event": ev, "allow": False}
    try:
        ev.wait(timeout)
        with APPROVALS_LOCK:
            return APPROVALS[key]["allow"]
    finally:
        with APPROVALS_LOCK:
            APPROVALS.pop(key, None)


def sse(wfile, obj) -> None:
    wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
    wfile.flush()


def permissions():
    import permissions as P
    return P.Policy(STATE.get("permissions_path"))


def permissions_state() -> dict:
    import permissions as P
    pol = permissions()
    return {"policy": pol.load(), "path": str(pol.path), "presets": P.presets(), "actions": P.ACTIONS,
            "effects": list(P.EFFECTS), "matches": list(P.MATCHES), "job_presets": P.JOB_PRESETS,
            "job_kinds": sorted(ns_jobs.all_specs()), "models": [m["id"] for m in scan_models()][:300]}


def with_permissions(decision: dict, model: str, info: dict, args: dict) -> dict:
    """Tool approval's answer, made stricter by the permission policy for this model
    (scripts/permissions.py): what the call would read, write, run or reach."""
    import permissions as P
    model = model if model and model != "auto" else str((PROC.get("model") or {}).get("name") or "")
    p = P.check_call(permissions(), model, info["server"], info["tool"], args,
                     read_only=(info.get("annotations") or {}).get("readOnlyHint"))
    # what the call would do, shown in the prompt; rules_action is tool approval's own answer
    out = {**decision, "rules_action": decision["action"], "model": model, "checks": p["checks"]}
    if P.RANK[p["effect"]] > P.RANK.get(decision["action"], 1):
        out.update({"action": p["effect"], "by": "permissions", "reason": p["why"]})
    return out


def chat_with_tools(wfile, payload: dict, msgs: list, hub, upstream=None) -> dict:
    """Stream a chat in which the model may call MCP tools. Each call is shown
    to the user and, unless auto-approved, waits for Allow/Deny."""
    payload = {**payload, "tools": hub.openai_tools()}
    out = {"text": ""}
    for _ in range(MAX_TOOL_ROUNDS):
        payload["messages"] = msgs
        out = proxy("/v1/chat/completions", payload, stream_to=wfile, hold_done=True, upstream=upstream)
        calls = out.get("tool_calls") or []
        if not calls:
            break
        msgs = msgs + [{"role": "assistant", "content": out["text"] or None, "tool_calls": [
            {"id": c["id"] or f"call_{i}", "type": "function",
             "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}} for i, c in enumerate(calls)]}]
        for i, c in enumerate(calls):
            cid = c["id"] or f"call_{i}"
            key = os.urandom(8).hex()
            try:
                args = json.loads(c["arguments"] or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
            except ValueError as e:
                args, err = None, f"invalid arguments: {e}"
            else:
                err = None
            decision = None
            if err is None:
                try:
                    info = hub.tool_info(c["name"])
                    decision = tool_rules().decide(info["server"], info["tool"], args, info["annotations"],
                                                   auto_approve=info["auto_approve"])
                    decision = with_permissions(decision, payload.get("model") or "", info, args)
                except KeyError as e:
                    err = str(e).strip("'\"")
            need = err is None and decision["action"] == "ask"
            sse(wfile, {"tool_call": {"key": key, "name": c["name"], "arguments": args if args is not None
                                      else c["arguments"], "needs_approval": need,
                                      "decision": decision and {k: decision[k] for k in ("action", "by", "reason")},
                                      "checks": (decision or {}).get("checks") or []}})
            if err is None and decision["action"] == "deny":
                err = f"not allowed: {decision['reason']} (Settings -> Tool approval)"
            elif err is None and need:
                with APPROVALS_LOCK:
                    PENDING_CALLS[key] = {"server": info["server"], "tool": info["tool"], "args": args,
                                          "model": decision.get("model", ""), "checks": decision.get("checks") or [],
                                          "rules_action": decision.get("rules_action", "ask")}
                try:
                    if not wait_approval(key):
                        err = "the user declined this tool call"
                finally:
                    with APPROVALS_LOCK:
                        PENDING_CALLS.pop(key, None)
            if err is None:
                try:
                    ok, text = hub.call(c["name"], args)
                except Exception as e:
                    ok, text = False, f"{type(e).__name__}: {e}"
            else:
                ok, text = False, err
            sse(wfile, {"tool_result": {"key": key, "ok": ok, "text": text[:4000]}})
            msgs = msgs + [{"role": "tool", "tool_call_id": cid, "content": text if ok else f"Error: {text}"}]
    wfile.write(b"data: [DONE]\n\n")
    wfile.flush()
    return out


# ---------------------------------------------------------------- RAG

EMBED = {"proc": None, "lock": threading.Lock()}


def rag_embedder():
    """The embedder for dense retrieval: --rag-embed URL@model, or a llama-server
    --embedding sidecar for --rag-embed-gguf, started on first use."""
    if STATE.get("rag_embed"):
        return ns_rag.Embedder(STATE["rag_embed"])
    gguf = STATE.get("rag_embed_gguf")
    if not gguf:
        return None
    port = STATE["backend_port"] + 1
    with EMBED["lock"]:
        p = EMBED["proc"]
        if p is None or p.poll() is not None:
            if not STATE["server_bin"]:
                raise RuntimeError("document embeddings need llama-server: Setup -> Build llama.cpp")
            if port_busy(port):
                raise RuntimeError(f"port {port} (embedding sidecar) is already in use")
            cmd = [STATE["server_bin"], "-m", gguf, "--embedding", "--pooling", "mean", "-ngl",
                   str(STATE.get("rag_embed_ngl", 0)), "-c", "2048", "-b", "2048", "-ub", "2048",
                   "--port", str(port), "--host", "127.0.0.1", "--alias", "rag-embed"]
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            ok, why = wait_healthy(port, proc)
            if not ok:
                proc.terminate()
                raise RuntimeError(f"embedding server did not start: {why}")
            EMBED["proc"] = proc
    return ns_rag.Embedder(f"http://127.0.0.1:{port}/v1@rag-embed")


def rag_store():
    st = STATE.get("rag")
    if st is None:
        st = STATE["rag"] = ns_rag.RagStore(STATE.get("rag_dir"))
    return st


def rag_context(req: dict, msgs: list) -> tuple[list, list]:
    """Retrieve for the last user message; returns (messages with context, sources)."""
    spec = req.get("rag") or {}
    name = spec.get("collection")
    if not name:
        return msgs, []
    last = next((m for m in reversed(msgs) if m.get("role") == "user"), None)
    query = last["content"] if last and isinstance(last.get("content"), str) else \
        " ".join(c.get("text", "") for c in (last or {}).get("content") or [] if isinstance(c, dict))
    store = rag_store()
    coll = store.get(name)
    emb = rag_embedder() if coll.dense is not None else None
    hits = coll.search(query, max(1, min(int(spec.get("k", 4)), 12)), emb)
    if not hits:
        return msgs, []
    if (req.get("compress") or {}).get("passages"):
        # Keep the sentences that bear on the question, within about a fifth of the context window.
        hits, info = ns_compress.compress_passages(query, hits, budget=max(300, (_ctx_size() or 8192) // 5))
        req["_compressed_passages"] = info
    ctx = {"role": "system", "content": ns_rag.context_message(hits)}
    i = 1 if msgs and msgs[0].get("role") == "system" else 0
    return msgs[:i] + [ctx] + msgs[i:], hits


def _ctx_size():
    """The running engine's context window, from its command line."""
    args = PROC["args"] or []
    for flag in ("-c", "--ctx-size", "--max-model-len"):
        if flag in args[:-1]:
            try:
                return int(args[args.index(flag) + 1])
            except ValueError:
                pass
    return None


def _msg_text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    return " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict) and p.get("type") == "text")


def count_tokens(texts: list) -> tuple[list, str]:
    """Tokens in each text: the running engine's tokenizer when it answers, else about 4 characters a token."""
    if server_running() and PROC["port"]:
        try:
            out = []
            for t in texts:
                if not t:
                    out.append(0)
                    continue
                req = urllib.request.Request(f"http://127.0.0.1:{PROC['port']}/tokenize",
                                             data=json.dumps({"content": t, "prompt": t}).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=3) as r:
                    d = json.loads(r.read())
                n = d.get("count") if isinstance(d.get("count"), int) else len(d.get("tokens") or [])
                if not n and t.strip():
                    raise ValueError("no tokens")
                out.append(n)
            return out, "tokenizer"
        except (OSError, ValueError, KeyError):
            pass
    return [round(len(t) / 4) if t else 0 for t in texts], "estimate"


def compress_history(msgs: list, req: dict, reserve: int, upstream=None) -> tuple[list, dict | None]:
    """Fit a long chat into the context window: drop the oldest turns ("cut"), or stand a model-written
    summary in for them ("summarize"). The client keeps the summary and sends it back as
    req["summary"] = {upto, text}, so earlier turns are summarized once. -> (messages, what was done)"""
    mode = (req.get("compress") or {}).get("history") or "off"
    n_ctx = _ctx_size()
    if mode not in ("cut", "summarize") or not n_ctx:
        return msgs, None
    head = 0
    while head < len(msgs) and msgs[head].get("role") == "system":
        head += 1
    fixed, conv = msgs[:head], msgs[head:]
    upto, text = 0, None
    sm = req.get("summary") or {}
    if mode == "summarize" and isinstance(sm, dict) and sm.get("text") and 0 < int(sm.get("upto") or 0) <= len(conv):
        upto, text = int(sm["upto"]), str(sm["text"])[:8000]
    count = lambda t: int(ns_compress.est_tokens(t) * 1.15) + 1      # noqa: E731  # leave a margin: code tokenizes densely
    budget = max(256, n_ctx - reserve - 64)
    cur = fixed + ([ns_compress.summary_message(text, upto)] if text else []) + conv[upto:]
    cut = ns_compress.fit_history(cur, budget, count)
    if cut == 0:
        return cur, ({"history": "summary", "upto": upto} if text else None)
    lead = len(fixed) + (1 if text else 0)              # the summary counts as a leading system message
    if mode == "cut":
        return cur[:lead] + cur[lead + cut:], {"history": "cut", "dropped": upto + cut, "upto": upto + cut}
    folded = conv[upto:upto + cut]
    prompt = ns_compress.summary_prompt(folded, text)
    r = proxy("/v1/chat/completions", {"messages": prompt, "max_tokens": 500, "temperature": 0.2, "stream": False},
              upstream=upstream)
    new = ((r.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    new = re.sub(r"(?s)<think>.*?</think>", "", new).strip()
    if not new:
        return cur[:lead] + cur[lead + cut:], {"history": "cut", "dropped": upto + cut, "upto": upto + cut,
                                               "note": "the model gave no summary; the oldest turns were dropped"}
    upto += cut
    return fixed + [ns_compress.summary_message(new, upto)] + conv[upto:], \
        {"history": "summarized", "upto": upto, "summary": new}


def context_usage(req: dict, role: str) -> dict:
    """What the next chat request would put in the model's context, by part (the chat side panel)."""
    msgs = [m for m in req.get("messages") or [] if isinstance(m, dict)]
    preset = load_presets().get(req.get("preset", "default"), PRESET_DEFAULTS)
    system = ""
    if msgs and msgs[0].get("role") == "system":
        system, msgs = _msg_text(msgs[0]), msgs[1:]
    elif preset.get("system"):
        system = preset["system"]
    tools = ""
    if req.get("tools") and role == "owner":
        hub = mcp_hub()
        if hub is not None:
            try:
                tools = json.dumps(hub.openai_tools())
            except Exception:
                tools = ""
    summarized = 0
    sm = req.get("summary") or {}
    if ((req.get("compress") or {}).get("history") == "summarize" and isinstance(sm, dict) and sm.get("text")
            and 0 < int(sm.get("upto") or 0) <= len(msgs)):
        summarized = int(sm["upto"])
        msgs = [ns_compress.summary_message(str(sm["text"]), summarized)] + msgs[summarized:]
    docs = str(req.get("documents") or "")[:200_000]
    convo = "\n".join(f"{m.get('role')}: {_msg_text(m)}" for m in msgs)
    (n_sys, n_docs, n_tools, n_msgs), method = count_tokens([system, docs, tools, convo])
    images = sum(1 for m in msgs if isinstance(m.get("content"), list)
                 for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url")
    parts = [{"key": "system", "label": "System prompt", "tokens": n_sys},
             {"key": "documents", "label": "Documents", "tokens": n_docs},
             {"key": "tools", "label": "Tool definitions", "tokens": n_tools},
             {"key": "messages", "label": "Messages", "tokens": n_msgs}]
    m = PROC["model"] or {}
    return {"n_ctx": _ctx_size(), "model": m.get("id"), "trained_ctx": m.get("context"),
            "reply_reserve": int(preset.get("max_tokens", 2048) or 0), "parts": parts,
            "total": sum(p["tokens"] for p in parts), "images": images, "method": method,
            "turns": len(msgs), "summarized": summarized}


CHAT_PATHS = ("/v1/chat/completions", "/v1/completions")


def proxy(path, payload, stream_to=None, hold_done=False, upstream=None, record=True, source="api"):
    """_proxy, with each chat reply written to the work log (who answered, on what, its tokens and time).
    record=False for a call its caller logs itself (a project task)."""
    if not record or path not in CHAT_PATHS:
        return _proxy(path, payload, stream_to, hold_done, upstream)
    t0, meas, err = time.time(), {}, None
    worker = model = host = device = None
    if upstream is not None:
        host = upstream.get("name") or "linked host"
        model = payload.get("model")
        worker = f"{host}{LINK_SEP}{model}" if model else host
    elif PROC["model"]:
        worker = model = PROC["model"]["id"]
        device = chat_device()
    try:
        out = _proxy(path, payload, stream_to, hold_done, upstream, meas)
        if isinstance(out, dict) and ("usage" in out or "timings" in out):
            meas.update(ns_work.llama_timings(out))
        return out
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        log_work("chat", worker or "chat model", model=model, host=host, device=device, error=err,
                 seconds=time.time() - t0, source=source, **meas)


def _proxy(path, payload, stream_to=None, hold_done=False, upstream=None, meas=None):
    """Forward to the running llama-server. Streams SSE when asked.

    Streaming matters here: on an iGPU at a few tokens per second, a
    non-streaming chat window looks indistinguishable from a hang.
    With hold_done the final [DONE] is not forwarded (a tool round follows),
    and streamed tool calls are assembled and returned.
    """
    if upstream is not None and upstream.get("callhome"):
        return _callhome_chat(upstream["callhome"], path, payload, stream_to, hold_done)
    if upstream is not None:
        # a linked host's model: same OpenAI API, its device token, its pinned certificate
        if stream_to is None:
            return ns_pairing.request(upstream["base"], "POST", path, payload, token=upstream["token"],
                                      fingerprint=upstream["fingerprint"], timeout=900)
        opener = lambda: _upstream_stream(upstream, path, payload)  # noqa: E731
    else:
        if not server_running():
            raise RuntimeError("no model loaded")
        url = f"http://127.0.0.1:{PROC['port']}{path}"
        req = urllib.request.Request(
            url, method="POST", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        if stream_to is None:
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.loads(r.read())
        opener = lambda: urllib.request.urlopen(req, timeout=900)  # noqa: E731
    # Stream through untouched, keeping a copy of the text for stats.
    text, calls, finish = [], {}, None
    r = opener()
    with r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if hold_done and line.startswith("data:") and line[5:].strip() == "[DONE]":
                continue
            stream_to.write(raw)
            stream_to.flush()
            if line.startswith("data:") and "[DONE]" not in line:
                try:
                    d = json.loads(line[5:])
                    if meas is not None and ("usage" in d or "timings" in d):
                        meas.update(ns_work.llama_timings(d))
                    ch = d["choices"][0]
                    delta = ch.get("delta") or {}
                    text.append(delta.get("content") or "")
                    finish = ch.get("finish_reason") or finish
                    for tc in delta.get("tool_calls") or []:
                        c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                        c["id"] = tc.get("id") or c["id"]
                        fn = tc.get("function") or {}
                        c["name"] += fn.get("name") or ""
                        c["arguments"] += fn.get("arguments") or ""
                except (ValueError, KeyError, IndexError, TypeError):
                    pass
    return {"text": "".join(text), "tool_calls": [calls[k] for k in sorted(calls)], "finish_reason": finish}


def _callhome_chat(host, path, payload, stream_to, hold_done):
    """A host that calls in has no address to proxy to: chat goes as its MCP chat tool.
    Agents that stream post the reply in pieces while it is written (MCP progress), and
    those pieces go straight on to the client; older agents answer whole, and the
    reply is sent as one piece."""
    if path != "/v1/chat/completions":
        raise RuntimeError(f"{host} calls in to this Studio; only chat is available through it")
    args = {k: payload[k] for k in ("model", "messages", "temperature", "max_tokens", "top_p", "seed", "stop")
            if payload.get(k) is not None}
    base = {"id": "callhome-" + secrets.token_hex(4), "object": "chat.completion.chunk", "model": args.get("model")}
    sent = []

    def emit(delta, finish=None):
        stream_to.write(f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]})}\n\n".encode())
        stream_to.flush()

    def piece(p):
        text = p if isinstance(p, str) else str(p.get("content") or "")
        if text:
            emit({"content": text} if sent else {"role": "assistant", "content": text})
            sent.append(text)

    if stream_to is not None:
        args["stream"] = True
    r = remote().call(host, "chat", args, on_progress=piece if stream_to is not None else None)
    resp = r.get("response") or {"choices": [{"index": 0, "message": {"role": "assistant", "content": r.get("reply", "")},
                                              "finish_reason": "stop"}], "model": args.get("model")}
    if stream_to is None:
        return resp
    text, so_far = r.get("reply", ""), "".join(sent)
    # What streamed is normally the whole reply; send whatever it lacks (all of it, from an older agent).
    rest = text[len(so_far):] if text.startswith(so_far) else ("" if so_far else text)
    if rest:
        piece(rest)
    finish = ((resp.get("choices") or [{}])[0].get("finish_reason")) or "stop"
    emit({}, finish)
    if resp.get("usage"):
        stream_to.write(f"data: {json.dumps({**base, 'choices': [], 'usage': resp['usage']})}\n\n".encode())
    if not hold_done:
        stream_to.write(b"data: [DONE]\n\n")
    stream_to.flush()
    return {"text": text or so_far, "tool_calls": [], "finish_reason": finish}


# ------------------------------------------------------- OpenAI-compatible API

def touch(delta=0):
    with ACTIVITY_LOCK:
        ACTIVITY["last"] = time.time()
        ACTIVITY["active"] += delta


def idle_reaper():
    """Unload the model after --idle-ttl seconds without requests (LM Studio's TTL)."""
    while True:
        time.sleep(5)
        ttl = STATE["idle_ttl"]
        if not ttl or not server_running():
            continue
        with ACTIVITY_LOCK:
            idle = ACTIVITY["active"] == 0 and time.time() - ACTIVITY["last"] > ttl
        if idle:
            with LOCK:
                if server_running():
                    print(f"idle for {ttl}s: unloading {PROC['model']['name']}")
                    stop_server()


def _last_user_text(body):
    for m in reversed(body.get("messages") or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                return " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            return str(c or "")
    return str(body.get("prompt") or "")


def _has_image(body):
    for m in body.get("messages") or []:
        c = m.get("content")
        if isinstance(c, list) and any(isinstance(x, dict) and x.get("type") == "image_url" for x in c):
            return True
    return False


def stats_ids(m, st=None):
    """Every name a model's stats may have been recorded under."""
    st = st if st is not None else load_settings()
    alias = st.get(_key(m["path"]), {}).get("served_name")
    return [x for x in (m["id"], alias, m["name"], m["name"][:-5]) if x]


def model_summary(m, st=None):
    s = STATE["stats"].summary(stats_ids(m, st), size=m["size"])
    n = s["graded"].get("n", 0)
    s["eligible"] = n >= STATE["min_graded"]
    s["why_not"] = None if s["eligible"] else (
        "No performance stats for this model file yet." if n == 0 else
        f"Only {n} graded results; auto routing needs {STATE['min_graded']}.")
    return s


def auto_pick(body, models):
    """Stats-based routing. Only models with enough graded results compete."""
    from subject_classifier import default_classifier
    text = _last_user_text(body)
    pool = models
    if _has_image(body):
        text += " image photo picture"
        pool = [m for m in models if m.get("mmproj")]
    clf = default_classifier()
    # Every subject the request involves, weighted; {} when the subject is
    # unknown, which ranks models on their overall numbers instead of a guess.
    proba = clf.route_weights(text)
    if _has_image(body) and not proba:
        proba = {"vision": 1.0}
    st = load_settings()
    by_id = {m["id"]: m for m in pool}
    # Your pick for a subject (Settings -> Models per subject) wins over the
    # measured best, for the strongest subject in the prompt that has one.
    chosen = STATE.get("subject_models") or {}
    for subj, w in sorted(proba.items(), key=lambda kv: -kv[1]):
        mid = chosen.get(subj)
        if mid and w >= 0.3 and mid in by_id:
            return by_id[mid], {"model": mid, "subject": subj, "candidates": [], "by": "setting",
                                "reason": f"your default for {subj} (Settings), {w:.0%} {subj}",
                                "proba": {k: round(v, 3) for k, v in list(proba.items())[:3]}}
    sums = {m["id"]: model_summary(m, st) for m in pool}
    pick = model_stats.rank(proba, sums, STATE["min_graded"], STATE["min_subject"], STATE["halluc_cost"])
    pick["proba"] = {k: round(v, 3) for k, v in list(proba.items())[:3]}
    pick["by"] = "measured"
    return by_id.get(pick["model"]), pick


def resolve_request_model(body):
    """Which scanned model should serve this request? -> (model or None, route info).

    "auto" ranks only models with performance stats; any other model name is a
    manual choice and is honoured whether or not stats exist."""
    ref = (body.get("model") or "").strip()
    current = PROC["model"]
    models = scan_models()
    if ref.lower() == "auto":
        target, pick = auto_pick(body, models)
        if target is None:
            excluded = "; ".join(f"{k}: {v}" for k, v in list(pick.get("excluded", {}).items())[:6])
            return None, {"mode": "auto", "error": (
                "auto: no model has enough performance stats to choose from "
                f"(need {STATE['min_graded']} graded results). Grade your models with Jobs -> Evaluate "
                "(TestQA), pick a default per subject under Settings -> Models per subject, or name a model "
                f"explicitly. {excluded}")}
        return target, {"mode": "auto", "model": target["id"], "reason": pick["reason"],
                        "candidates": pick["candidates"][:5]}
    if not ref or (current and find_model(ref, [current])):
        return current, {"mode": "loaded", "model": current["id"] if current else None}
    target = find_model(ref, models)
    return target, {"mode": "manual", "model": target["id"] if target else ref}


# ---------------------------------------------------------- reply statistics

SCORE_JOBS = queue.Queue(maxsize=4)
_SCORERS = {}
_REPLIES = {"n": 0}


def note_reply(model, messages, text):
    """Record an ungraded live reply, and queue an activation score if this
    model has a classifier configured. Never blocks the response."""
    if not model or STATE["stats"] is None:
        return
    try:
        from subject_classifier import default_classifier
        subject = default_classifier().predict(_last_user_text({"messages": messages}))
    except Exception:
        subject = None
    STATE["stats"].record(model["id"], "live", size=model["size"], subject=subject,
                          abstained=model_stats.looks_abstained(text), chars=len(text or ""))
    clf = load_settings().get(_key(model["path"]), {}).get("classifier")
    _REPLIES["n"] += 1
    if clf and STATE["cett"] and text and _REPLIES["n"] % max(1, STATE["score_every"]) == 0:
        try:
            SCORE_JOBS.put_nowait((model, clf, messages, text, subject))
        except queue.Full:
            pass        # never queue behind scoring; the next reply matters more


def score_worker():
    while True:
        model, clf, messages, text, subject = SCORE_JOBS.get()
        # Activation scoring is an extra prefill; wait for a quiet moment.
        while ACTIVITY["active"] > 0:
            time.sleep(1)
        try:
            sc = _scorer(model, clf)
            msgs = [m for m in messages if m.get("role") != "system"] or messages
            prof = getattr(sc, "profile", None)
            res = prof(msgs, text) if prof else sc.score(msgs, text)
            if "cett" in res:
                try:
                    import neuron_review
                    review_store().record(model["id"], neuron_review.bin_profile(res["cett"]), source="live",
                                          kind="observed", subjects=[subject] if subject else None,
                                          risk=res.get("prob"))
                except Exception as e:
                    print(f"[studio] review record failed: {e}", file=sys.stderr)
            STATE["stats"].record(model["id"], "activation", size=model["size"], subject=subject,
                                  h_score=res["score"], prob=res["prob"], n_tokens=res["n_tokens"],
                                  threshold=0.0)
        except Exception as e:
            print(f"[studio] activation scoring failed for {model['name']}: {e}", file=sys.stderr)


# ------------------------------------------------------- per-reply checks

TRACES = {"lock": threading.Lock(), "payloads": {}}
REVIEW_VIEWS = {"lock": threading.Lock(), "views": {}}
TRACE_ID = re.compile(r"[A-Za-z0-9_-]{6,64}")


def _scorer(model, clf):
    key = (model["path"], clf)
    if key not in _SCORERS:
        from hscore import HScorer
        _SCORERS[key] = HScorer(STATE["cett"], model["path"], clf, ngl=STATE["score_ngl"])
    return _SCORERS[key]


def _configured_bins():
    return {"llama-server": STATE.get("server_bin"), "llama-cett-dump": STATE.get("cett")}


def review_store():
    if "review" not in STATE or STATE["review"] is None:
        import neuron_review
        STATE["review"] = neuron_review.ReviewStore(STATE.get("review_dir"))
    return STATE["review"]


def _review_subjects(text):
    try:
        from subject_classifier import default_classifier
        return default_classifier().analyze(text)["labels"] or ["unknown"]
    except Exception:
        return ["unknown"]


def trace_reply(model, messages, text, threshold=0.5, source="chat-check"):
    """Score one reply token by token and save it as a trace session that the
    3D view (viz/bloom.py) and timeline.py can open. -> summary for the chat.
    The reply's mean activation also goes into the neuron review store."""
    clf = load_settings().get(_key(model["path"]), {}).get("classifier")
    if not clf:
        raise ValueError(f"{model['id']} has no classifier set (Studio -> Models, the model's settings: classifier)")
    if not STATE["cett"]:
        raise ValueError("activation checks need llama-cett-dump: Setup -> Build llama.cpp, "
                         "or point Settings -> Workspaces at an existing build")
    msgs = [m for m in messages if m.get("role") != "system"] or messages
    r = _scorer(model, clf).trace(msgs, text)
    return save_trace(model["name"], model["id"], msgs, text, r, threshold, source)


def save_trace(model_name, model_id, msgs, text, r, threshold=0.5, source="chat-check", extra=None):
    """Save a tracer result (HScorer.trace or one sent by a host) as a trace session
    for the 3D view, and record its mean activation in the neuron review."""
    frames = r["frames"]
    T, L, N = frames.shape
    tid = time.strftime("%Y%m%d-%H%M%S-") + hashlib.sha1(text.encode()).hexdigest()[:8]
    import numpy as np
    from records import Recorder
    with Recorder(os.path.join(STATE["traces_dir"], tid),
                  {"model": model_name, "n_layers": L, "n_neurons": N, "kind": "trace",
                   "stride": r["stride"], "source": "studio", **(extra or {})}, resume=False) as rec:
        rec.add("reply", agg=frames.reshape(-1, N), tokens=r["tokens"], scores=r["scores"],
                kind="trace", n_frames=T, n_layers=L, pieces=r["pieces"], stride=1,
                question=(_last_user_text({"messages": msgs}) or "")[:500], verdict=None,
                h_cells=r["h_cells"], col_weight=np.asarray(r["col_weight"]).tolist())
    prob = [round(float(p), 4) for p in r["prob"]]
    review = None
    try:
        q = _last_user_text({"messages": msgs}) or ""
        oid = review_store().record(model_id, frames.mean(axis=0), source=source, kind="observed",
                                    subjects=_review_subjects(q), risk=float(sum(prob) / len(prob)),
                                    item=tid)
        review = {"model": model_id, "id": oid}
    except Exception as e:
        print(f"[studio] review record failed: {e}", file=sys.stderr)
    return {"id": tid, "pieces": r["pieces"], "prob": prob,
            "flagged": [i for i, p in enumerate(prob) if p >= threshold], "threshold": threshold,
            "max": max(prob), "mean": round(sum(prob) / len(prob), 4),
            "url": f"viz/{tid}/", "review": review, **(extra or {})}


def _local_twin(size, sha):
    """A local model whose file is byte-for-byte the given one (same size, then same SHA-256)."""
    import score_cards
    for m in scan_models():
        try:
            if m.get("format", "gguf") == "gguf" and os.path.getsize(m["path"]) == size \
                    and score_cards.sha256_file(m["path"]) == sha:
                return m
        except OSError:
            continue
    return None


def _remote_classifier(host_model_id, twin, file_name):
    """The classifier for a host's model: set for that id, else the twin's, else a local model
    with the same file name's."""
    st = load_settings()
    for key in (_key("remote:" + host_model_id), _key(twin["path"]) if twin else None):
        if key and st.get(key, {}).get("classifier"):
            return st[key]["classifier"]
    for m in scan_models():
        if os.path.basename(m["path"]) == file_name and st.get(_key(m["path"]), {}).get("classifier"):
            return st[_key(m["path"])]["classifier"]
    return None


def trace_remote(model_id, messages, text, threshold=0.5):
    """Check a reply from a remote host's model. With the same file here, Studio traces it
    itself; otherwise the host runs OmniCortex's tracer and sends the trace back."""
    host, _, rid = model_id.partition(LINK_SEP)
    rec = remote().get(host)
    info = remote().call(host, "model_file", {"model": rid}, timeout=1800)
    twin = _local_twin(info["size"], info["sha256"])
    extra = {"host": host, "host_model": rid}
    if twin is not None and STATE["cett"] and load_settings().get(_key(twin["path"]), {}).get("classifier"):
        out = trace_reply(twin, messages, text, threshold, source="host-check")
        return {**out, **extra, "traced_on": "this computer (same file)"}
    clf = _remote_classifier(model_id, twin, info["file"])
    if not clf:
        raise ValueError(f"{model_id} has no classifier: set one for this host model (Models, its settings), or for "
                         f"a local copy of {info['file']}")
    import base64
    import hscore
    msgs = [m for m in messages if m.get("role") != "system"] or messages
    with open(clf, "rb") as f:
        args = {"model": rid, "messages": msgs, "text": text, "classifier": base64.b64encode(f.read()).decode()}
    r = remote().call(rec["name"], "trace_reply", args, timeout=1800)
    tr = hscore.load_trace(base64.b64decode(r["trace"]))
    return save_trace(model_id, model_id, msgs, text, tr, threshold, "host-check",
                      {**extra, "traced_on": host})


def trace_subjects(path):
    """The neuron review's subject map for the model a trace belongs to, if the review has enough
    observations across subjects (cached per model and observation count)."""
    try:
        model = json.load(open(os.path.join(path, "manifest.json"))).get("model")
        if not model:
            return None
        import neuron_review as nrv
        st = review_store()
        n = len(st.query(model))
        cache = STATE.setdefault("subject_maps", {})
        if cache.get(model, (None,))[0] != n:
            cache[model] = (n, nrv.subject_map(st, model) if n >= 6 else None)
        return cache[model][1]
    except Exception:
        return None


def trace_payload(tid):
    """bloom payload for a saved trace, built once and cached (a few per process)."""
    with TRACES["lock"]:
        hit = TRACES["payloads"].get(tid)
    if hit:
        return hit
    path = os.path.join(STATE["traces_dir"], tid)
    if not TRACE_ID.fullmatch(tid) or not os.path.isdir(path):
        return None
    import bloom
    blob, meta = bloom.build_payload(path, None, 97.0, 40000, subjects=trace_subjects(path))
    out = {"blob": blob, "meta": meta, "theme": bloom.THEMES.get(STATE.get("viz_theme") or "dark")
           or next(iter(bloom.THEMES.values()))}
    with TRACES["lock"]:
        if len(TRACES["payloads"]) >= 8:
            TRACES["payloads"].pop(next(iter(TRACES["payloads"])))
        TRACES["payloads"][tid] = out
    return out


def list_traces(limit=200):
    d = STATE.get("traces_dir")
    out = []
    if d and os.path.isdir(d):
        for name in sorted(os.listdir(d), reverse=True)[:limit]:
            try:
                meta = json.load(open(os.path.join(d, name, "manifest.json")))
            except (OSError, ValueError):
                continue
            out.append({"id": name, "model": meta.get("model"), "created": meta.get("created")})
    return out


# ------------------------------------------------------------ worker models

class WorkerPool:
    """llama-server processes for the director's tasks, next to the chat model
    Studio serves. Placement uses every accelerator scripts/accelerators.py
    finds (ROCm, Vulkan, CUDA, Metal, the CPU), AMD first by default, each
    device with its own llama-server build, base settings and environment from
    the hardware config, and a project's own per-device settings on top. Idle
    workers stop after --worker-idle seconds, sooner when a new one needs room."""

    def __init__(self):
        self.lock = threading.RLock()
        self.workers = {}              # key -> worker
        self._devs = {"t": 0.0, "list": []}

    def devices(self, refresh=False):
        import accelerators
        if refresh or time.time() - self._devs["t"] > 10:
            cfg = accelerators.load_config(STATE.get("hardware_path"))
            self._devs = {"t": time.time(), "list": accelerators.detect(cfg),
                          "prefer": tuple(cfg.get("prefer") or accelerators.PREFER)}
        return self._devs["list"]

    def _reserved(self):
        r = {}
        for w in self.workers.values():
            for dev, b in w["booked"].items():
                r[dev] = r.get(dev, 0) + b
        return r

    @staticmethod
    def need_bytes(model, ctx):
        # Weights, plus KV cache and scratch: ~0.5 GiB, more for long contexts.
        return int(model["size"] * 1.1) + int(512 * 2**20 * max(1, (ctx or 8192) / 8192))

    def _find(self, model, allowed):
        for w in self.workers.values():
            if w["model"]["path"] == model["path"] and (not allowed or set(w["device_ids"]) <= set(allowed)):
                return w
        return None

    def acquire(self, model, allowed=None, project_settings=None, timeout=900):
        """-> worker dict (busy += 1), starting the model if needed. Other
        callers of a worker that is still loading wait until it is healthy."""
        import accelerators
        allowed = list(allowed or [])
        if not allowed and server_running() and PROC["model"] and PROC["model"]["path"] == model["path"]:
            return {"url": f"http://127.0.0.1:{PROC['port']}/v1", "shared": True, "model": model}
        with self.lock:
            w = self._find(model, allowed)
            if w and w["proc"].poll() is None:
                w["busy"] += 1
                w["last"] = time.time()
                starting = False
            else:
                if w:
                    self._stop(w)
                base = {**DEFAULTS, **load_settings().get(_key(model["path"]), {})}
                while True:
                    placed = None
                    if len(self.workers) < STATE["max_workers"]:
                        devs = self.devices()
                        placed = accelerators.place(devs, self.need_bytes(model, base.get("ctx")),
                                                    self._reserved(), allowed, self._devs.get("prefer"))
                    if placed:
                        break
                    idle = sorted((x for x in self.workers.values() if x["busy"] == 0), key=lambda x: x["last"])
                    if not idle:
                        where = f" on {', '.join(allowed)}" if allowed else ""
                        raise RuntimeError(f"no room for {model['id']}{where}: "
                                           f"{len(self.workers)} worker(s) busy (max {STATE['max_workers']})")
                    self._stop(idle[0])
                devs = placed["devices"]
                settings = dict(base)
                for d in devs:
                    settings.update(d.get("settings") or {})
                    settings.update((project_settings or {}).get(d["id"]) or {})
                settings.update({"gpus": "", "ngl": placed["ngl"] if placed["ngl"] == 0 else settings.get("ngl", 99),
                                 "split_mode": "layer" if placed["split"] else "",
                                 "tensor_split": ",".join(map(str, placed["split"])) if placed["split"] else ""})
                port = STATE["backend_port"] + 10
                used = {x["port"] for x in self.workers.values()}
                while port in used or port_busy(port):
                    port += 1
                binary = devs[0].get("server") or STATE["server_bin"]
                if not binary:
                    raise RuntimeError(f"no llama-server for {devs[0]['backend']}: set one under Setup -> Hardware "
                                       "(servers per backend), or Setup -> Build llama.cpp for it")
                cmd, env = server_cmd(model, settings, port)
                cmd[0] = binary
                env = {**(env or os.environ), **accelerators.launch_env(devs)}
                need = self.need_bytes(model, settings.get("ctx"))
                tot = sum(max(1, x) for x in (placed["split"] or [1]))
                booked = {d["id"]: int(need * (placed["split"][i] if placed["split"] else 1) / tot)
                          for i, d in enumerate(devs)}
                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
                w = {"model": model, "proc": proc, "port": port, "url": f"http://127.0.0.1:{port}/v1",
                     "device_ids": [d["id"] for d in devs], "backend": devs[0]["backend"],
                     "where": " + ".join(f"{d['id']} ({d['name']})" for d in devs), "booked": booked,
                     "busy": 1, "last": time.time(), "started": time.time(), "shared": False,
                     "ready": threading.Event(), "error": None, "unmeasured": bool(placed.get("unmeasured"))}
                self.workers[f"{model['path']}|{','.join(w['device_ids'])}"] = w
                starting = True
        if starting:
            ok, why = wait_healthy(w["port"], w["proc"])
            if not ok:
                err = ""
                try:
                    w["proc"].terminate()
                    err = (w["proc"].stderr.read() or b"").decode(errors="replace")[-500:]
                except Exception:
                    pass
                w["error"] = f"worker {model['id']} on {w['where']} failed to start: {why}\n{err}"
                with self.lock:
                    self._stop(w)
            w["ready"].set()
        elif not w["ready"].wait(timeout):
            self.release(w)
            raise RuntimeError(f"worker {model['id']} still loading after {timeout}s")
        if w["error"]:
            raise RuntimeError(w["error"])
        return w

    def release(self, w):
        if w.get("shared"):
            return
        with self.lock:
            w["busy"] = max(0, w["busy"] - 1)
            w["last"] = time.time()

    def _stop(self, w):
        for k, v in list(self.workers.items()):
            if v is w:
                self.workers.pop(k)
        p = w["proc"]
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()

    def stop(self, path=None):
        with self.lock:
            for w in list(self.workers.values()):
                if path is None or w["model"]["path"] == path:
                    self._stop(w)

    def reap(self, idle):
        with self.lock:
            for w in list(self.workers.values()):
                if not w["ready"].is_set():
                    continue
                if w["proc"].poll() is not None or (w["busy"] == 0 and time.time() - w["last"] > idle):
                    self._stop(w)

    def list(self):
        with self.lock:
            return [{"model": w["model"]["id"], "where": w["where"], "backend": w["backend"],
                     "devices": w["device_ids"], "port": w["port"], "busy": w["busy"],
                     "loading": not w["ready"].is_set(),
                     "idle_s": round(time.time() - w["last"]), "up_s": round(time.time() - w["started"]),
                     "alive": w["proc"].poll() is None} for w in self.workers.values()]


class ContextTooLong(RuntimeError):
    pass


POOL = WorkerPool()


def worker_chat(url, alias, messages, max_tokens=4096, timeout=1800, token=None, usage=None):
    """One chat with a worker's OpenAI-compatible server -> the reply text. With usage (a dict),
    the server's token counts are copied into it."""
    body = json.dumps({"model": alias, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0.3, "stream": False}).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", {})
        except Exception:
            msg = {}
        text = msg.get("message") if isinstance(msg, dict) else str(msg)
        if isinstance(msg, dict) and msg.get("type") == "exceed_context_size_error":
            raise ContextTooLong(f"{text} (pick a model with a longer context, or shorten the task's inputs)")
        raise RuntimeError(f"HTTP {e.code}: {text or e.reason}")
    if usage is not None:
        usage.update(d.get("usage") or {})
    return d["choices"][0]["message"].get("content") or ""


# ------------------------------------------------------------ the director

DIRECTOR = {"store": None, "running": set(), "lock": threading.Lock(), "notes": {}}


def _director():
    import director
    return director


# ------------------------------------------------------------ workers: where tasks run

def workers_registry():
    reg = STATE.get("workers_reg")
    if reg is None:
        import worker_registry
        home = os.path.dirname(os.path.expanduser(STATE.get("config_path") or "~/.neuronscope/config.json"))
        reg = STATE["workers_reg"] = worker_registry.Registry(os.path.join(home, "workers.json"))
    return reg


def work_log():
    """The work log (scripts/work_log.py), beside Studio's config unless NS_WORK_LOG names a folder."""
    wl = STATE.get("work_log")
    if wl is None:
        home = os.path.dirname(os.path.expanduser(STATE.get("config_path") or "~/.neuronscope/config.json"))
        wl = STATE["work_log"] = ns_work.WorkLog(os.environ.get("NS_WORK_LOG") or os.path.join(home, "worklog"))
    return wl


def log_work(kind, worker, **kw):
    """Never lets bookkeeping fail the work it describes."""
    if STATE.get("work_log_off"):
        return
    try:
        work_log().record(kind, worker, **kw)
    except Exception as e:
        print(f"work log: {e}", file=sys.stderr)


def chat_device():
    """Where the chat model runs, from its llama-server arguments."""
    args = [str(a) for a in PROC.get("args") or []]
    for flag in ("--device", "-dev"):
        if flag in args[:-1]:
            return args[args.index(flag) + 1]
    for flag in ("-ngl", "--n-gpu-layers", "--gpu-layers"):
        if flag in args[:-1] and args[args.index(flag) + 1] == "0":
            return "CPU"
    gpus = TELEMETRY.get("local") or []
    return gpus[0].get("name") or "GPU" if len(gpus) == 1 else "GPU"


def _hw_sampler():
    """This computer's GPUs every 30 s, and each linked host's while the Workers page keeps them fresh."""
    import remote_hosts
    while True:
        if not STATE.get("work_log_off"):
            try:
                work_log().sample(remote_hosts.nsr.gpu_telemetry())
                with TELEMETRY["lock"]:
                    hosts = dict(TELEMETRY["hosts"]) if time.time() - TELEMETRY["t"] < 60 else {}
                for name, h in hosts.items():
                    if isinstance(h, dict) and h.get("gpus"):
                        work_log().sample(h["gpus"], host=name)
            except Exception as e:
                print(f"work log: hardware sample: {e}", file=sys.stderr)
        time.sleep(30)


def job_finished(meta: dict) -> None:
    started, ended = meta.get("started"), meta.get("ended")
    rc = meta.get("returncode")
    log_work("job", meta.get("kind") or "job", seconds=(ended - started) if started and ended else None,
             ok=rc == 0, error=None if rc == 0 else f"exit {rc}" if rc is not None else meta.get("status"),
             host=meta.get("host"), device=(meta.get("values") or {}).get("device"), source=meta.get("id"))
    threading.Thread(target=sharing_after_job, args=(meta,), daemon=True).start()


TELEMETRY = {"lock": threading.Lock(), "want": 0.0, "local": [], "hosts": {}, "t": 0.0, "thread": None,
             "health": {}}


def _endpoint_health(ep):
    try:
        req = urllib.request.Request(ep["url"].rstrip("/") + "/models",
                                     headers={"Authorization": f"Bearer {ep['token']}"} if ep.get("token") else {})
        with urllib.request.urlopen(req, timeout=3) as r:
            ids = [m.get("id") for m in json.loads(r.read()).get("data", [])]
        return {"ok": True, "models": ids[:20]}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}


def _telemetry_loop():
    """While someone watches the Workers page: this computer's GPUs, each remote host's,
    and whether each endpoint answers, every few seconds."""
    import remote_hosts
    while True:
        if time.time() - TELEMETRY["want"] > 30:
            time.sleep(1)
            continue
        local = remote_hosts.nsr.gpu_telemetry()
        hosts = {}
        for h in remote().hosts():
            try:
                hosts[h["name"]] = remote().call(h["name"], "telemetry", timeout=8)
            except Exception as e:
                hosts[h["name"]] = {"error": str(e)[:200]}
        health = {"ep:" + ep["name"]: _endpoint_health(ep) for ep in workers_registry().endpoints()}
        with TELEMETRY["lock"]:
            TELEMETRY.update(local=local, hosts=hosts, health=health, t=time.time())
        time.sleep(4)


def telemetry_snapshot():
    TELEMETRY["want"] = time.time()
    if TELEMETRY["thread"] is None:
        TELEMETRY["thread"] = threading.Thread(target=_telemetry_loop, daemon=True)
        TELEMETRY["thread"].start()
    with TELEMETRY["lock"]:
        return {k: TELEMETRY[k] for k in ("local", "hosts", "health", "t")}


def worker_kind(wid):
    if wid.startswith("ep:"):
        return "endpoint"
    if LINK_SEP in wid and wid.partition(LINK_SEP)[0] in {x["name"] for x in load_links()}:
        return "remote"
    return "local"


def all_workers():
    """Every place a task can run, with its controls, live counters and status."""
    reg = workers_registry()
    tel = telemetry_snapshot()
    pool = {w["model"]: w for w in POOL.list()}
    rows = []
    for m in scan_models():
        w = pool.get(m["id"])
        rows.append({"id": m["id"], "kind": "local", "name": m["id"], "size": m["size"],
                     "where": w["where"] if w else "this computer (starts when a task needs it)",
                     "status": ("loading" if w["loading"] else "running") if w else "stopped",
                     "port": w["port"] if w else None})
    for ep in reg.endpoints():
        h = tel["health"].get("ep:" + ep["name"])
        rows.append({"id": "ep:" + ep["name"], "kind": "endpoint", "name": ep["name"], "where": ep["url"],
                     "model": ep.get("model") or "", "note": ep.get("note", ""), "has_token": bool(ep.get("token")),
                     "status": "checking" if h is None else ("up" if h["ok"] else "down"),
                     "error": None if h is None or h["ok"] else h["error"],
                     "served": (h or {}).get("models")})
    for m in link_models():
        host = m.get("link")
        ht = tel["hosts"].get(host) or {}
        th = (ht.get("thermal") or {}) if isinstance(ht, dict) else {}
        rows.append({"id": m["id"], "kind": "remote", "name": m.get("remote_id"), "host": host,
                     "where": f"{host} ({'OmniCortex Remote' if m.get('owned_by', '').startswith('remote:') else 'linked Studio'})",
                     "status": "too hot" if th.get("tripped") else ("loaded" if m.get("loaded") else "available"),
                     "backend": m.get("backend")})
    for r in rows:
        r.update(control=reg.control(r["id"]), stats=reg.summary(r["id"]), limit=reg.limit(r["id"]))
    return rows


def work_queue():
    """Tasks of approved, running and paused projects that are not finished: who has them and
    why they wait."""
    store = DIRECTOR["store"]
    if store is None:
        return []
    dr, reg, out = _director(), workers_registry(), []
    for row in store.list():
        if row["status"] not in ("running", "paused", "approved"):
            continue
        try:
            plan = store.load(row["id"])
        except Exception:
            continue
        ready = {t["id"] for t in dr.ready_tasks(plan)}
        for t in plan["tasks"]:
            if t["status"] not in ("todo", "rework", "running"):
                continue
            wid = (t.get("assignee") or {}).get("model") or ""
            c = reg.control(wid) if wid else {}
            if t["status"] == "running":
                why = "running"
            elif plan["status"] != "running":
                why = f"project {plan['status']}"
            elif (t.get("not_before") or 0) > time.time():
                why = f"retry in {int(t['not_before'] - time.time()) + 1}s (the last attempt failed)"
            elif t["id"] not in ready:
                why = "waiting for other tasks"
            elif c.get("paused"):
                why = "worker paused"
            elif c.get("draining"):
                why = "worker draining"
            elif not worker_alive(wid):
                why = "worker down or too hot"
            elif not reg.has_slot(wid):
                why = "worker busy"
            else:
                why = "next"
            out.append({"project": plan["id"], "project_title": plan["title"], "task": t["id"], "title": t["title"],
                        "labels": t.get("labels") or [], "status": t["status"], "worker": wid,
                        "reason": (t.get("assignee") or {}).get("reason"), "why": why})
    return out


def worker_alive(wid):
    if target_busy(worker_target(wid)):
        return False
    if wid.startswith("ep:"):
        h = TELEMETRY["health"].get(wid)
        return h is None or h["ok"]
    if worker_kind(wid) == "remote":
        th = ((TELEMETRY["hosts"].get(wid.partition(LINK_SEP)[0]) or {}).get("thermal") or {})
        return not th.get("tripped")
    return True


def chat_off_machine(wid, msgs, usage):
    """A task's chat on an endpoint or a remote model -> reply text."""
    if wid.startswith("ep:"):
        ep = workers_registry().endpoint(wid)
        if ep is None:
            raise RuntimeError(f"no endpoint {wid[3:]!r} (Workers page)")
        return worker_chat(ep["url"], ep.get("model") or "default", msgs, token=ep.get("token"), usage=usage)
    up, rid = link_target(wid)
    r = proxy("/v1/chat/completions", {"model": rid, "messages": msgs, "max_tokens": 4096,
                                       "temperature": 0.3, "stream": False}, upstream=up, record=False)
    usage.update(r.get("usage") or {})
    return ((r.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


# ------------------------------------------------------------ studio settings

# Settings the Setup page may change, with their type and whether they apply
# without a restart. Network exposure (host, token, TLS, remote jobs) is never
# editable from the web: those stay in the config file and on the command line.
# key: (kind, applies without a restart, help, group). Kinds: dirs, file (an executable), readfile (a file
# that must exist), path (may not exist yet), dir, int, float, bool, str, duration, durations, url.
SETUP_FIELDS = {
    "models_dir": ("dirs", True, "Folders searched for .gguf models", "Models and engines"),
    "server": ("file", True, "llama-server binary", "Models and engines"),
    "vllm": ("str", True, "The vLLM command, for models served with the vLLM engine", "Models and engines"),
    "download_dir": ("dir", True, "Where model downloads go (default: the first models folder)", "Models and engines"),
    "backend_port": ("int", False, "Local port for llama-server or vLLM", "Models and engines"),
    "idle_ttl": ("int", True, "Unload the chat model after this many idle seconds (0 = never)", "Models and engines"),
    "no_jit": ("bool", True, "Never load or swap models for API requests", "Models and engines"),
    "min_graded": ("int", True, "Graded results a model needs before auto routing considers it", "Routing"),
    "hallucination_cost": ("float", True, "How much a wrong answer costs against a right one, for routing", "Routing"),
    "stats_window": ("int", False, "Most recent observations per kind that count in a model's record", "Routing"),
    "cett": ("file", True, "llama-cett-dump binary (activation checks, the GGUF pipeline)", "Activation checks"),
    "score_every": ("int", True, "Score one live reply in N with the activation classifier", "Activation checks"),
    "score_ngl": ("int", True, "GPU layers for activation scoring (0 keeps it off the GPU)", "Activation checks"),
    "max_workers": ("int", True, "Worker models the director may run at once", "Workers and jobs"),
    "worker_idle": ("int", True, "Stop an idle worker model after this many seconds", "Workers and jobs"),
    "hardware": ("path", False, "Accelerator settings for worker models (Setup → Hardware writes it)", "Workers and jobs"),
    "max_jobs": ("int", False, "Jobs that may run at once", "Workers and jobs"),
    "no_jobs": ("bool", False, "Turn the Jobs page off entirely", "Workers and jobs"),
    "allow_remote_jobs": ("bool", False, "Allow jobs when Studio listens on the network (jobs train models and can "
                          "run model-written code)", "Workers and jobs"),
    "rag_embed_gguf": ("readfile", False, "Embedding GGUF for document search", "Documents and tools"),
    "rag_embed": ("url", False, "Or an OpenAI-compatible embeddings endpoint: URL, optionally @model",
                  "Documents and tools"),
    "rag_embed_ngl": ("int", False, "GPU layers for the embedding model", "Documents and tools"),
    "mcp_config": ("path", False, "MCP servers for chat tools (mcp.json)", "Documents and tools"),
    "host": ("str", False, "Address Studio listens on: 127.0.0.1 is this computer only; 0.0.0.0 or a LAN/VPN address "
             "needs a token, and TLS or plaintext allowed", "Network and security"),
    "port": ("int", False, "Studio's port", "Network and security"),
    "token_file": ("readfile", False, "File with Studio's access token (ns_security.py token makes one)",
                   "Network and security"),
    "tls_cert": ("readfile", False, "TLS certificate (PEM); with the key, Studio serves HTTPS", "Network and security"),
    "tls_key": ("readfile", False, "TLS private key (PEM)", "Network and security"),
    "allow_plaintext": ("bool", False, "Allow HTTP without TLS on the network (only behind a VPN or a TLS proxy)",
                        "Network and security"),
    "pair_code_ttl": ("duration", False, "How long a pairing link can be claimed (e.g. 90s, 5m, 1h)", "Pairing"),
    "pair_durations": ("durations", False, "Temporary-access durations offered when pairing", "Pairing"),
    "pair_max": ("duration", False, "Longest temporary access a pairing may grant", "Pairing"),
    "pair_default": ("str", False, "What a pairing grants unless chosen otherwise: persistent, or a duration",
                     "Pairing"),
    "no_persistent_pairing": ("bool", False, "Only grant temporary access: every paired device expires", "Pairing"),
}
ADVANCED_GROUPS = ("Network and security", "Pairing")


def _ws_path():
    from pathlib import Path as _P
    return _P(os.path.expanduser(STATE.get("config_path") or "~/.neuronscope/config.json"))


def subject_defaults():
    """Per subject: the model measured best for it, and the one you chose (which wins)."""
    from subject_classifier import SUBJECTS
    models = scan_models()
    st = load_settings()
    sums = {m["id"]: model_summary(m, st) for m in models}
    chosen = STATE.get("subject_models") or {}
    out = {}
    for subj in SUBJECTS:
        if subj == "unknown":
            continue
        pick = model_stats.rank({subj: 1.0}, sums, STATE["min_graded"], STATE["min_subject"], STATE["halluc_cost"])
        best = pick.get("model")
        row = next((r for r in pick.get("candidates") or [] if r["model"] == best), None)
        own = (sums.get(best) or {}).get("subjects", {}).get(subj, {}).get("n", 0) if best else 0
        out[subj] = {"best": best, "reason": pick.get("reason"), "expected": row and row["expected"],
                     "subject_n": own, "chosen": chosen.get(subj),
                     "chosen_missing": bool(chosen.get(subj)) and not any(m["id"] == chosen[subj] for m in models),
                     "uses": chosen.get(subj) or best}
    return out


def settings_state():
    rules = tool_rules()
    hub = STATE.get("mcp")
    tools = []
    if hub is not None:
        try:
            for t in hub.tool_list():
                d = rules.decide(t["server"], t["tool"], {}, t["annotations"], auto_approve=t["auto_approve"],
                                 record=False)
                tools.append({**t, "decision": d})
        except Exception:
            tools = []
    import tool_rules as tr
    return {"subjects": subject_defaults(), "models": [m["id"] for m in scan_models()],
            "min_graded": STATE["min_graded"], "tool_rules": rules.load(), "tool_rules_path": str(rules.path),
            "tool_kinds": tr.KINDS, "tools": tools, "mcp_configured": bool(STATE.get("mcp_config"))}


def workspaces_state():
    import ns_workspace
    d = ns_workspace.describe(_ws_path(), STATE.get("workspace_forced"))
    d["running"] = STATE.get("workspace_name") or d["active"]
    return d


SHARING_DEFAULTS = {"enabled": False, "endpoint": "", "auto_listed": False, "auto_unlisted": False,
                    "public_models": []}


def sharing() -> dict:
    """Score sharing settings (Settings → Sharing results). Off until the owner turns it on."""
    return {**SHARING_DEFAULTS, **(load_studio_config().get("sharing") or {})}


def sharing_listed(path: str, refresh: bool = False) -> dict:
    """Is this model file the one in its Hugging Face repo (cached; checking hashes the file)."""
    cache = STATE.setdefault("hf_checks", {})
    if refresh or path not in cache:
        import score_cards
        cache[path] = score_cards.hf_check(path)
    return cache[path]


def sharing_results() -> list[dict]:
    """TestQA results files from Studio's jobs, newest first."""
    out, seen = [], set()
    jobs = STATE.get("jobs")
    for j in (jobs.list(200) if jobs else []):
        if j.get("kind") != "testqa" or j.get("status") != "done":
            continue
        f = (j.get("values") or {}).get("out")
        if not f:
            continue
        p = Path(f) if os.path.isabs(f) else Path(jobs.cwd or ".") / f
        if str(p) in seen or not p.exists():
            continue
        seen.add(str(p))
        try:
            d = json.loads(p.read_text())
            eps = d.get("endpoints") or {}
        except (OSError, ValueError):
            continue
        out.append({"file": str(p), "ended": j.get("ended"), "cache": (j.get("values") or {}).get("cache"),
                    "labels": [{"label": k, "model": v.get("model")} for k, v in eps.items()]})
    return out


def sharing_card(results: str, label: str, model: dict, cache: str | None = None):
    import score_cards
    st = load_settings().get(_key(model["path"]), {})
    sh = sharing()
    listed = sharing_listed(model["path"])
    return score_cards.build(results, label, model["path"], {**DEFAULTS, **st}, STATE.get("server_bin"), cache,
                             listed=listed, public_by_hand=model["id"] in sh["public_models"])


def sharing_after_job(meta: dict) -> None:
    """Automatic submission when a TestQA job ends, within the owner's settings."""
    sh = sharing()
    if meta.get("kind") != "testqa" or meta.get("status") != "done" or not (sh["enabled"] and sh["endpoint"]):
        return
    if not (sh["auto_listed"] or sh["auto_unlisted"]):
        return
    import score_cards
    for r in sharing_results():
        if r["ended"] != meta.get("ended"):
            continue
        for lb in r["labels"]:
            m = find_model(lb.get("model") or lb["label"])
            if m is None:
                continue
            listed = sharing_listed(m["path"])["listed"]
            if not ((listed and sh["auto_listed"]) or (not listed and sh["auto_unlisted"])):
                continue
            try:
                card, log = sharing_card(r["file"], lb["label"], m, r.get("cache"))
                score_cards.submit(sh["endpoint"], card, log)
            except Exception as e:
                STATE.setdefault("sharing_errors", []).append(f"{m['id']}: {type(e).__name__}: {e}"[:300])


def job_suggestions(ws: dict) -> dict:
    """Values the job forms offer in their dropdowns: models, finished TestQA results, H-Neuron sets and the
    workspace's data files, so a field is picked rather than typed."""
    root = Path(ws.get("root") or ".")
    models = scan_models()

    def files(patterns, limit=100):
        out = []
        for pat in patterns:
            out += [str(p) for p in sorted(root.glob(pat))[:limit]]
        return sorted(dict.fromkeys(out))[:limit]
    try:
        results = [r["file"] for r in sharing_results()][:50]
    except Exception:
        results = []
    return {"models": [m["id"] for m in models][:200],
            "hf_models": [m["path"] for m in models if m.get("format") == "hf"][:50],
            "results": results,
            "neuron_sets": files(["models/h_neurons*.json", "runs/**/h_neurons*.json"], 50),
            "data_files": files(["data/*.jsonl", "data/*.json"], 100),
            "subjects": sorted(__import__("subject_classifier").SUBJECTS)}


def shared_board_rows() -> list[dict]:
    """The scoreboard's public rows (cached for an hour), whose speed measurements train the predictions.
    Empty unless sharing is on."""
    sh = sharing()
    ep = sh.get("endpoint")
    if not (sh.get("enabled") and ep):
        return []
    c = STATE.setdefault("board_cache", {})
    if c.get("ep") == ep and time.time() - c.get("t", 0) < 3600:
        return c["rows"]
    try:
        if ep.startswith("github:"):
            import github_board
            rows = github_board.fetch(ep[7:]).get("rows") or []
        else:
            with urllib.request.urlopen(ep.rstrip("/") + "/api/board", timeout=15) as r:
                rows = json.loads(r.read()).get("board") or []
        c.pop("error", None)
    except Exception as e:
        rows, c["error"] = [], f"{type(e).__name__}: {e}"[:200]
    c.update(ep=ep, t=time.time(), rows=rows)
    return rows


def _device_info(host, dev):
    """A work-log device string -> {name, backend, unified} for speed predictions."""
    dev = str(dev or "")
    if host not in (None, "this computer"):
        return {"name": dev or host, "backend": "", "unified": False}
    for d in POOL.devices():
        if d["id"] in dev or (d.get("name") and d["name"] in dev):
            return {"name": d.get("name") or d["id"], "backend": d.get("backend") or "", "unified": bool(d.get("unified"))}
    m = re.match(r"(Vulkan|CUDA|ROCm|Metal|SYCL)(\d+)$", dev, re.I)
    if m:
        be = m.group(1).lower()
        same = [d for d in POOL.devices() if d.get("backend") == be or (be == "vulkan" and d.get("backend") != "cpu")]
        d = same[int(m.group(2))] if int(m.group(2)) < len(same) else None
        if d:
            return {"name": d.get("name") or d["id"], "backend": be, "unified": bool(d.get("unified"))}
    if dev.upper() == "CPU":
        cpu = next((d for d in POOL.devices() if d.get("backend") == "cpu"), {})
        return {"name": cpu.get("name") or "CPU", "backend": "cpu", "unified": True}
    return {"name": dev, "backend": "", "unified": False} if dev and dev != "GPU" else None


def speed_predictions(since="30d") -> dict:
    """Fit the speed model on this computer's work log and the scoreboard's measurements, then predict every
    local model on every local device."""
    import perf_predict as pp
    models = {m["id"]: m for m in scan_models() if m.get("format") != "hf"}
    settings = load_settings()

    def model_info(mid):
        m = models.get(mid) or find_model(mid, list(models.values()))
        if not m:
            return None
        return {"bytes": m.get("size"), "quant": (m.get("meta") or {}).get("quant") or score_quant(m["path"]),
                "settings": settings.get(_key(m["path"]), {})}
    local = pp.samples_from_worklog(work_log().read("work", ns_work.parse_since(since)), model_info, _device_info)
    board = pp.samples_from_board(shared_board_rows())
    model = pp.fit(local + board)
    measured = {}
    for s in local:
        measured.setdefault(s.device, []).append(s)
    devs = [d for d in POOL.devices() if d.get("enabled", True)]
    rows = []
    for mid, m in sorted(models.items()):
        if not m.get("size"):
            continue
        q = model_info(mid)["quant"]
        cells = {}
        for d in devs:
            fits = not d.get("memory_total") or d["memory_total"] >= m["size"] * 1.1
            smp = pp.Sample(device=d.get("name") or d["id"], backend=d.get("backend") or "", model_bytes=m["size"],
                            tok_s=0, quant=q, unified=bool(d.get("unified")))
            pr = model.predict(smp)
            got = [x.tok_s for x in local if x.device == smp.device and x.model_bytes == m["size"]]
            cells[d["id"]] = {**pr, "fits": fits, "measured": round(sum(got) / len(got), 1) if got else None}
        rows.append({"model": mid, "gb": round(m["size"] / 2**30, 2), "quant": q, "devices": cells})
    return {"devices": [{"id": d["id"], "name": d.get("name"), "backend": d.get("backend"),
                         "memory_gb": round((d.get("memory_total") or 0) / 2**30, 1)} for d in devs],
            "rows": rows, "model": model.to_json(), "samples": {"this computer": len(local), "scoreboard": len(board)},
            "board_error": STATE.get("board_cache", {}).get("error")}


def score_quant(path):
    import score_cards
    return score_cards._quant(os.path.basename(path))


def measurement_weights() -> dict:
    import measure_weights as mw
    ids = [m["id"] for m in scan_models()]
    table = mw.table_from_stats(STATE["stats"], ids, review=review_store()) if STATE.get("stats") else {}
    res = mw.fit(table)
    home = os.path.dirname(os.path.expanduser(STATE.get("config_path") or "~/.neuronscope/config.json"))
    hist = mw.remember(res, os.path.join(home, "measure-weights.json")) if table else []
    return {**res, "table": table, "history": hist[-30:]}


def studio_themes() -> dict:
    """Named Studio themes (viz/themes.json, "_studio", plus those of theme add-ons that are on):
    UI colours, chart colours, and the 3D viewer theme each one uses."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "themes.json")) as f:
            core = json.load(f).get("_studio") or {}
    except (OSError, ValueError):
        core = {}
    try:
        extra = plugins().themes().get("_studio") or {}
    except Exception:
        extra = {}
    return {**core, **{k: v for k, v in extra.items() if k not in core}}


def load_studio_config(path=None):
    p = os.path.expanduser(path or STATE.get("config_path") or "~/.neuronscope/config.json")
    try:
        with open(p) as f:
            d = json.load(f)
        return d.get("studio", {}) if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_studio_config(values):
    p = os.path.expanduser(STATE.get("config_path") or "~/.neuronscope/config.json")
    try:
        with open(p) as f:
            d = json.load(f)
    except (OSError, ValueError):
        d = {}
    st = d.setdefault("studio", {})
    st.update(values)
    for k in [k for k, v in values.items() if v is None and k in SETUP_FIELDS]:
        st.pop(k)          # cleared: back to the flag's default
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, p)


def clean_setup(values):
    """Validate Setup page values -> (clean, errors)."""
    out, errs = {}, {}
    for k, v in values.items():
        if k not in SETUP_FIELDS:
            errs[k] = "not editable here"
            continue
        kind = SETUP_FIELDS[k][0]
        try:
            if kind == "dirs":
                v = [os.path.expanduser(str(x).strip()) for x in (v if isinstance(v, list) else [v]) if str(x).strip()]
                bad = [x for x in v if not os.path.isdir(x)]
                if bad:
                    raise ValueError(f"not a folder: {', '.join(bad)}")
            elif kind in ("file", "dir", "readfile", "path"):
                v = os.path.expanduser(str(v or "").strip()) or None
                if v and kind == "file" and not (os.path.isfile(v) and os.access(v, os.X_OK)):
                    raise ValueError("not an executable file")
                if v and kind == "readfile" and not os.path.isfile(v):
                    raise ValueError("no such file")
                if v and kind == "dir" and not os.path.isdir(v):
                    raise ValueError("not a folder")
            elif kind in ("str", "url"):
                v = str(v or "").strip() or None
                if v and (len(v) > 500 or not re.fullmatch(r"[^\x00-\x1f]*", v)):
                    raise ValueError("too long, or has control characters")
                if v and kind == "url" and not re.match(r"https?://", v):
                    raise ValueError("must start with http:// or https://")
                if v and k == "pair_default" and v != "persistent":
                    ns_pairing.parse_duration(v)
            elif kind == "duration":
                v = str(v or "").strip() or None
                if v:
                    ns_pairing.parse_duration(v)
            elif kind == "durations":
                v = ",".join(x.strip() for x in str(v or "").split(",") if x.strip()) or None
                for x in (v or "").split(",") if v else []:
                    ns_pairing.parse_duration(x)
            elif kind in ("int", "float") and v in (None, ""):
                v = None
            elif kind == "int":
                v = int(v)
                if v < 0:
                    raise ValueError("must be 0 or more")
            elif kind == "float":
                v = float(v)
            elif kind == "bool":
                v = bool(v)
            out[k] = v
        except (TypeError, ValueError) as e:
            errs[k] = str(e) or "invalid"
    net = {"host", "token_file", "tls_cert", "tls_key", "allow_plaintext"}
    if not (net & set(errs)) and net & set(out):
        # A listen address Studio would refuse at startup is refused now, before it can lock you out.
        eff = {**setup_effective(), **out}
        tok = STATE.get("token") or ""
        if eff.get("token_file"):
            try:
                tok = open(eff["token_file"]).read().strip()
            except OSError:
                tok = ""
        try:
            sec.check_bind(eff.get("host") or "127.0.0.1", tok, tls=bool(eff.get("tls_cert") and eff.get("tls_key")),
                           allow_plaintext=bool(eff.get("allow_plaintext")))
        except sec.SecurityConfigError as e:
            errs["host"] = " ".join(str(e).split())
    if not errs and out.get("port") is not None and not 1 <= out["port"] <= 65535:
        errs["port"] = "must be 1-65535"
    return out, errs


def apply_setup(values):
    """Apply what can change without a restart; -> keys that need one."""
    later = []
    setup_effective()          # loads the flags' defaults
    for k, v in values.items():
        if v is None and SETUP_FIELDS[k][0] in ("int", "float", "bool"):
            v = SETUP_DEFAULTS.get(k)       # cleared: back to the default
        if k == "models_dir":
            STATE["models_dirs"] = v or STATE["models_dirs"]
        elif k == "server":
            STATE["server_bin"] = v
        elif k == "cett":
            STATE["cett"] = v
            _SCORERS.clear()
        elif k == "download_dir":
            STATE["download_dir"] = v
        elif k == "idle_ttl":
            STATE["idle_ttl"] = v
        elif k == "no_jit":
            STATE["jit"] = not v
        elif k == "min_graded":
            STATE["min_graded"] = max(1, v)
        elif k == "hallucination_cost":
            STATE["halluc_cost"] = v
        elif k == "max_workers":
            STATE["max_workers"] = max(1, v)
        elif k == "worker_idle":
            STATE["worker_idle"] = max(30, v)
        elif k in ("score_every", "score_ngl"):
            STATE[k] = v
        elif k == "vllm":
            STATE["vllm_bin"] = v or "vllm"
        else:
            later.append(k)
    return later


def setup_effective():
    """What each Setup field is now: running values where Studio keeps them, else the saved
    setting, else the flag's default."""
    cfg = load_studio_config()
    if not SETUP_DEFAULTS:
        p = build_parser()
        SETUP_DEFAULTS.update({k: p.get_default(k) for k in SETUP_FIELDS})
    eff = {**SETUP_DEFAULTS, **{k: v for k, v in cfg.items() if k in SETUP_FIELDS}}
    eff.update({"models_dir": STATE["models_dirs"], "server": STATE["server_bin"], "cett": STATE["cett"],
                "download_dir": STATE["download_dir"], "idle_ttl": STATE["idle_ttl"], "no_jit": not STATE["jit"],
                "min_graded": STATE["min_graded"], "hallucination_cost": STATE["halluc_cost"],
                "max_workers": STATE["max_workers"], "worker_idle": STATE["worker_idle"],
                "score_every": STATE["score_every"], "score_ngl": STATE["score_ngl"],
                "rag_embed_gguf": STATE.get("rag_embed_gguf"), "vllm": STATE.get("vllm_bin")})
    return eff


SETUP_DEFAULTS = {}     # the flags' defaults, read from build_parser() once


def setup_state():
    eff = setup_effective()
    # Binaries a llama.cpp build would have produced, to offer when nothing is set.
    found = {}
    for root in [os.path.expanduser("~/llama.cpp"), os.environ.get("NS_LLAMA") or ""]:
        for name, key in (("llama-server", "server"), ("llama-cett-dump", "cett")):
            pth = os.path.join(root, "build", "bin", name)
            if root and os.path.isfile(pth):
                found.setdefault(key, pth)
    return {"fields": {k: {"kind": v[0], "live": v[1], "help": v[2], "group": v[3], "value": eff.get(k),
                           "advanced": v[3] in ADVANCED_GROUPS}
                       for k, v in SETUP_FIELDS.items()},
            "config_path": os.path.expanduser(STATE.get("config_path") or "~/.neuronscope/config.json"),
            "found": found, "models": len(scan_models()),
            "ready": {"server": bool(STATE["server_bin"] and os.path.exists(STATE["server_bin"])),
                      "models": bool(scan_models()), "cett": bool(STATE["cett"])}}


_HUB = {}


def lab_hub():
    """viz/hub.py's service manager, run inside Studio: the labs, live and replay
    views and the pipeline dashboard. Studio holds what it starts until stopped."""
    if "m" not in _HUB:
        import hub
        root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        hub.STATE["root"] = root
        hub.SERVICES.update({k: v for k, v in hub.declare(root, sys.executable).items() if k != "chat"})
        _HUB["m"] = hub
    return _HUB["m"]


def stop_labs():
    if "m" in _HUB:
        for name in list(_HUB["m"].RUNNING):
            _HUB["m"].stop(name)


# ---------------------------------------------------------------- services (System page)

_CODE_DIRS = [os.path.dirname(os.path.abspath(__file__)),
              os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")]


def code_stamp() -> float:
    """Newest modification time of Studio's own code: if it moved since Studio started, the running
    process is out of date (pages or features added since then answer "not found")."""
    newest = 0.0
    for d in _CODE_DIRS:
        try:
            for e in os.scandir(d):
                if e.name.endswith(".py"):
                    newest = max(newest, e.stat().st_mtime)
        except OSError:
            pass
    return newest


STARTED = {"t": time.time(), "code": code_stamp(), "checked": 0.0, "stale": False}


def code_stale() -> bool:
    if time.time() - STARTED["checked"] > 10:
        STARTED["checked"] = time.time()
        STARTED["stale"] = code_stamp() > STARTED["code"] + 1
    return STARTED["stale"]


def services() -> list[dict]:
    """Each part of the harness this Studio runs, with what can be done to it."""
    out = [{"id": "studio", "name": "Studio", "state": "running", "detail":
            f"up {int(time.time() - STARTED['t'])} s, pid {os.getpid()}"
            + (" · its code changed on disk since it started" if code_stale() else ""),
            "warn": code_stale(), "actions": ["restart"]}]
    m = PROC.get("model")
    out.append({"id": "chat", "name": "Chat model server", "state": "running" if server_running() else "stopped",
                "detail": (m or {}).get("id") or "no model loaded", "actions": ["reload", "stop"] if m else []})
    ws = POOL.list()
    out.append({"id": "workers", "name": "Worker models", "state": f"{len(ws)} running" if ws else "none",
                "detail": ", ".join(w["model"] for w in ws)[:200], "actions": ["stop"] if ws else []})
    ep = EMBED.get("proc")
    out.append({"id": "embed", "name": "Embedding server (documents)",
                "state": "running" if ep is not None and ep.poll() is None else "stopped",
                "detail": STATE.get("rag_embed") or STATE.get("rag_embed_gguf") or "not configured",
                "actions": ["restart"] if STATE.get("rag_embed_gguf") else []})
    hub = STATE.get("mcp")
    st = hub.status() if hub else []
    out.append({"id": "mcp", "name": "MCP connectors", "state": f"{sum(x['connected'] for x in st)} of {len(st)} connected"
                if st else "none", "detail": ", ".join(x["name"] + ("" if x["connected"] else " (down)") for x in st)[:200],
                "warn": any(not x["connected"] and not x.get("disabled") for x in st), "actions": ["reconnect"] if st else []})
    labs = list(_HUB["m"].RUNNING) if "m" in _HUB else []
    out.append({"id": "labs", "name": "Labs", "state": f"{len(labs)} running" if labs else "none",
                "detail": ", ".join(labs), "actions": ["stop"] if labs else []})
    try:
        hosts = remote().hosts()
    except Exception:
        hosts = []
    for rec in hosts:
        info, err = _cached("jobs_info:" + rec["name"], 30, lambda: remote().call(rec["name"], "jobs_info", {}, timeout=8))
        out.append({"id": "host:" + rec["name"], "name": f"Remote agent: {rec['name']}",
                    "state": "unreachable" if err else "connected", "warn": bool(err),
                    "detail": err or ("runs jobs from " + info["neuronscope_dir"] if info and info.get("allowed")
                                      and info.get("neuronscope_dir") else "jobs not allowed by its owner"),
                    "actions": ["restart"]})
    runner = STATE.get("jobs")
    running = [j for j in runner.list(20) if j.get("status") == "running"] if runner else []
    out.append({"id": "jobs", "name": "Jobs", "state": f"{len(running)} running" if running else "idle",
                "detail": ", ".join(j["title"] for j in running)[:200], "actions": []})
    return out


def service_action(sid: str, action: str) -> str:
    if sid == "studio" and action == "restart":
        threading.Thread(target=restart_studio, daemon=True).start()
        return "Studio restarts in a moment"
    if sid == "chat" and action == "stop":
        stop_server()
        return "chat model stopped"
    if sid == "chat" and action == "reload":
        m = PROC.get("model")
        if not m:
            raise ValueError("no model loaded")
        stop_server()
        with LOCK:
            ok, why = ensure_loaded(m, own=0)
        if not ok:
            raise RuntimeError(f"reload failed: {why}")
        return f"{m['id']} reloaded"
    if sid == "workers" and action == "stop":
        POOL.stop()
        return "worker models stopped"
    if sid == "embed" and action == "restart":
        with EMBED["lock"]:
            p = EMBED.get("proc")
            if p is not None and p.poll() is None:
                p.terminate()
            EMBED["proc"] = None
        return "embedding server stopped; it starts again on the next document search"
    if sid == "mcp" and action == "reconnect":
        hub = mcp_hub()
        if hub is None:
            raise ValueError("no MCP connectors configured")
        hub.connect(timeout=30)
        return "connectors reconnected"
    if sid.startswith("host:") and action == "restart":
        res = remote().call(sid[5:], "restart_agent", {}, timeout=15)
        with _RLOCK:
            _RCACHE.pop("jobs_info:" + sid[5:], None)
        return res.get("note") or "the agent restarts"
    if sid == "labs" and action == "stop":
        stop_labs()
        return "labs stopped"
    raise ValueError(f"cannot {action} {sid}")


def restart_studio():
    """Re-exec Studio with the same arguments, after stopping the model servers it
    started (an orphaned llama-server would hold their ports)."""
    time.sleep(0.5)                       # let the response reach the browser
    stop_server()
    POOL.stop()
    stop_labs()
    os.execv(sys.executable, [sys.executable] + sys.argv)


def run_doctor():
    """scripts/doctor.py --json, with this Studio's llama.cpp and model."""
    env = dict(os.environ)
    if STATE.get("server_bin"):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(STATE["server_bin"]))))
        env.setdefault("NS_LLAMA", root)
    try:
        r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                                         "scripts", "doctor.py"), "--json"],
                           capture_output=True, text=True, timeout=150, env=env)
        checks = json.loads(r.stdout or "[]")
    except Exception as e:
        return {"ok": False, "error": f"doctor failed: {e}", "checks": []}
    return {"ok": not any(c["status"] == "fail" for c in checks), "checks": checks}


def assign_candidates():
    st = load_settings()
    out, sums = [], {}
    for m in scan_models():
        fit = fit_estimate(m["size"])
        out.append({"id": m["id"], "size": m["size"], "fits": fit.get("ok"), "context": m.get("context")})
        try:
            s = model_summary(m, st)
            if s.get("eligible"):
                sums[m["id"]] = s
        except Exception:
            pass
    return out, sums


def assign_fn_for(plan):
    """Routes first (Workers page: per subject, the workers to try), then the director's own
    choice among the workers that take the task's subjects and are not paused."""
    dr = _director()
    reg = workers_registry()
    cands, sums = assign_candidates()
    cands = cands + [{"id": "ep:" + ep["name"], "size": 0, "fits": True, "context": None}
                     for ep in reg.endpoints()]

    def fn(t):
        labels = t.get("labels") or []
        r = reg.route(labels, worker_alive)
        if r:
            reg.event(r["model"], "assigned", f"{t.get('id')}: {r['reason']}")
            return r
        ok = [c for c in cands if reg.takes(c["id"], labels) and not reg.control(c["id"])["paused"]
              and not target_busy(worker_target(c["id"]))]
        out = dr.assign(t, ok, sums, plan["policy"], STATE["min_graded"], STATE["min_subject"], STATE["halluc_cost"],
                        exclude=tuple(t.get("avoid") or ()))
        if out.get("model"):
            reg.event(out["model"], "assigned", f"{t.get('id')}: {out['reason']}")
        return out
    return fn


def next_model_fn(t):
    """After repeated rejections: the best model other than the one that failed."""
    dr = _director()
    cands, sums = assign_candidates()
    cur = (t.get("assignee") or {}).get("model")
    tried = {a.get("model") for a in t["attempts"]} | {cur}
    return dr.assign(t, cands, sums, {}, STATE["min_graded"], STATE["min_subject"], STATE["halluc_cost"],
                     exclude=tuple(x for x in tried if x))


def screen_result(graph_name: str, t: dict, text: str, model: str | None) -> dict:
    """A project's verification graph over one result (scripts/graph_engine.py): "accepted" passes it,
    "feedback" fails it; a graph that stops to ask a person, or breaks, leaves it to the reviewer."""
    import graph_engine as G
    import graphs_api
    import ns_mcp as studio_client
    import permissions as P
    try:
        store = graphs_api.Store(STATE.get("graphs_dir") or os.path.expanduser("~/.neuronscope/graphs"))
        G.sync_plugins(plugins())
        G.HIDDEN = addon_builtins().node_hidden
        g = store.get(graph_name)
        base = f"{'https' if STATE['tls'] else 'http'}://127.0.0.1:{STATE.get('port') or 7870}"
        env = G.Env(studio_client.Studio(base, STATE.get("token"), insecure_loopback=True), P.Policy(STATE.get("permissions_path")),
                    model=model or "", graph_loader=store.get)
        run = G.Runner(g, env).start({"task": f"{t['title']}\n\n{t.get('detail') or ''}".strip(), "result": text})
        store.save_run(G.public_run(run) | {"kind": "screen"})
        out = run.get("outputs") or {}
        if run["status"] == "done" and "feedback" in out:
            return {"status": "failed", "feedback": G._text(out["feedback"])[:3000], "run": run["id"], "graph": graph_name}
        if run["status"] == "done" and "accepted" in out:
            return {"status": "passed", "run": run["id"], "graph": graph_name}
        return {"status": "person" if run["status"] == "waiting" else "error", "run": run["id"], "graph": graph_name,
                "feedback": run.get("error") or "the graph gave neither accepted nor feedback"}
    except Exception as e:                                                     # noqa: BLE001
        return {"status": "error", "graph": graph_name, "feedback": f"{type(e).__name__}: {e}"[:500]}


def run_task(pid, tid, wid=None, t0=None):
    """One attempt at one task, on its assigned worker. Runs in a thread. The director
    counted the task against worker wid at t0 (Workers page); every way out uncounts it."""
    dr = _director()
    store = DIRECTOR["store"]
    reg, usage, label = workers_registry(), {}, f"{pid}/{tid}"
    if t0 is None and wid:
        t0 = reg.begin(wid, label)
    text = err = check = screen = mid = where = None
    fatal = False
    try:
        with store.lock(pid):
            plan = store.load(pid)
            t = dr.task(plan, tid)
            msgs = dr.worker_messages(plan, t)
            mid = (t.get("assignee") or {}).get("model")
            policy = dict(plan["policy"])
        model = None
        if worker_kind(mid) == "local":
            model = find_model(mid)
            if model is None:
                raise RuntimeError(f"assigned model {mid!r} is not in the models directories")
            w = POOL.acquire(model, policy.get("devices"), policy.get("device_settings"))
            where = w.get("where") or (chat_device() if w.get("shared") else None)
            try:
                alias = load_settings().get(_key(model["path"]), {}).get("served_name") or model["id"]
                text = worker_chat(w["url"], alias, msgs, usage=usage)
            finally:
                POOL.release(w)
        else:
            text = chat_off_machine(mid, msgs, usage)
        if policy.get("check") and STATE["cett"] and text and model is not None:
            try:
                r = trace_reply(model, msgs, text, source="project")
                check = {"id": r["id"], "url": r["url"], "max": r["max"], "mean": r["mean"],
                         "n_flagged": len(r["flagged"]), "flagged": bool(r["flagged"])}
            except ValueError:
                check = None            # no classifier for this model: unchecked, and the page says so
            except Exception as e:
                check = {"error": str(e)[:300]}
        if policy.get("verify_graph") and text:
            screen = screen_result(policy["verify_graph"], t, text, mid)
    except ContextTooLong as e:
        err, fatal = str(e), True
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    finally:
        if wid:
            reg.end(wid, t0, label, err, usage.get("completion_tokens") or 0)
            kind = worker_kind(wid)
            host = (urllib.parse.urlsplit((reg.endpoint(wid) or {}).get("url", "")).hostname if kind == "endpoint"
                    else wid.partition(LINK_SEP)[0] if kind == "remote" else None)
            log_work("task", wid, model=mid, device=where, host=host, error=err, source=f"project {pid}",
                     seconds=time.time() - t0 if t0 else None, prompt_tokens=usage.get("prompt_tokens"),
                     completion_tokens=usage.get("completion_tokens"))
            if err and wid.startswith("ep:") and re.search(r"URLError|ConnectionRefused|timed out|Errno", err):
                # Down now, not at the next health check: its other tasks wait instead of failing too.
                with TELEMETRY["lock"]:
                    TELEMETRY["health"][wid] = {"ok": False, "error": err[:200]}
        taken = err and (BUSY_PREFIX in err or target_busy(worker_target(wid)))
        if taken and BUSY_PREFIX in (err or "") and worker_target(wid) and not target_busy(worker_target(wid)):
            BUSY[worker_target(wid)] = {"why": err.split(BUSY_PREFIX, 1)[1].strip()[:300], "since": time.time(),
                                        "until": None, "source": "host", "actions": ["busy"]}
        with store.lock(pid):
            plan = store.load(pid)
            t = dr.task(plan, tid)
            if t["status"] == "running":
                if taken:
                    why = (target_busy(worker_target(wid)) or {}).get("why") or "the computer was taken back"
                    dr.yield_attempt(plan, t, why)
                    reg.event(wid, "requeued", f"{label}: {why}"[:300])
                else:
                    dr.finish_attempt(plan, t, text, check=check, error=err, fatal=fatal, screen=screen)
                store.save(plan)
        with DIRECTOR["lock"]:
            DIRECTOR["running"].discard((pid, tid))


def director_loop():
    """Start ready tasks of running projects, within each project's parallel
    limit. Tasks only ever start from an approved plan that a person started."""
    dr = _director()
    while True:
        time.sleep(STATE.get("director_tick", 2))
        store = DIRECTOR["store"]
        if store is None:
            continue
        try:
            POOL.reap(STATE["worker_idle"])
            for row in store.list():
                if row["status"] != "running":
                    continue
                pid = row["id"]
                with store.lock(pid):
                    plan = store.load(pid)
                    with DIRECTOR["lock"]:
                        mine = sum(1 for p, _ in DIRECTOR["running"] if p == pid)
                    slots = plan["policy"]["max_parallel"] - mine
                    started = []
                    reg = workers_registry()
                    for t in dr.ready_tasks(plan):
                        if len(started) >= max(0, slots):
                            break
                        wid = (t.get("assignee") or {}).get("model") or ""
                        if wid and target_busy(worker_target(wid)):
                            # its computer is taken: hand the task to another worker, if one can take it
                            new = assign_fn_for(plan)(t)
                            if new.get("model") and not target_busy(worker_target(new["model"])):
                                reg.event(new["model"], "reassigned", f"{t['id']}: {worker_target(wid)} is busy")
                                t["assignee"] = new
                                wid = new["model"]
                            else:
                                continue
                        # A paused, draining, full or overheated worker holds its tasks back.
                        if wid and not (reg.has_slot(wid) and worker_alive(wid)):
                            continue
                        dr.start_attempt(plan, t)
                        started.append((t["id"], wid, reg.begin(wid, f"{pid}/{t['id']}") if wid else None))
                    if started:
                        store.save(plan)
                for tid, wid, t0 in started:
                    with DIRECTOR["lock"]:
                        DIRECTOR["running"].add((pid, tid))
                    threading.Thread(target=run_task, args=(pid, tid, wid, t0), daemon=True).start()
        except Exception as e:
            print(f"[studio] director: {e}", file=sys.stderr)


def recover_projects():
    """After a restart, attempts that were running have no thread behind them."""
    dr = _director()
    store = DIRECTOR["store"]
    for row in store.list():
        with store.lock(row["id"]):
            plan = store.load(row["id"])
            hit = False
            for t in plan["tasks"]:
                if t["status"] == "running":
                    dr.finish_attempt(plan, t, None, error="Studio restarted during this attempt")
                    hit = True
            if hit:
                store.save(plan)


def refine_project(pid, model_id):
    dr = _director()
    store = DIRECTOR["store"]
    note = None
    try:
        with store.lock(pid):
            plan = store.load(pid)
            msgs = dr.refine_messages(plan)
        model = find_model(model_id) if model_id else None
        if model is None:
            raise RuntimeError("choose a model to plan with")
        w = POOL.acquire(model, plan["policy"].get("devices"), plan["policy"].get("device_settings"))
        try:
            alias = load_settings().get(_key(model["path"]), {}).get("served_name") or model["id"]
            text = worker_chat(w["url"], alias, msgs, max_tokens=4096)
        finally:
            POOL.release(w)
        refined = dr.parse_refined(text)
        with store.lock(pid):
            plan = store.load(pid)
            dr.replace_tasks(plan, refined, "director", f"re-planned by {model['id']}")
            store.save(plan)
        note = f"re-planned by {model['id']}: {len(refined)} task(s)"
    except Exception as e:
        note = f"re-planning failed, draft kept: {e}"
    DIRECTOR["notes"][pid] = {"at": time.time(), "text": note, "busy": False}


def ensure_loaded(model, own=1):
    """JIT: load `model` with its saved settings unless it is already serving.
    `own` is how many in-flight requests belong to the caller."""
    if server_running() and PROC["model"] and PROC["model"]["path"] == model["path"]:
        return True, "loaded"
    if not STATE["jit"]:
        return False, f"{model['id']} is not loaded and JIT loading is disabled"
    # One llama-server at a time: let in-flight requests (other than this
    # one) finish before swapping the model out from under them.
    deadline = time.time() + 600
    while ACTIVITY["active"] > own and time.time() < deadline:
        time.sleep(0.5)
    settings = {**DEFAULTS, **load_settings().get(_key(model["path"]), {})}
    return start_server(model, settings, STATE["backend_port"])


def openai_models():
    st = load_settings()
    data = []
    any_eligible = False
    for m in scan_models():
        alias = st.get(_key(m["path"]), {}).get("served_name")
        sm = model_summary(m, st)
        g = sm["graded"]
        any_eligible |= sm["eligible"]
        data.append({"id": alias or m["id"], "object": "model", "owned_by": "local",
                     "stats": {"graded": g.get("n", 0), "accuracy": g.get("accuracy"),
                               "hallucination_rate": g.get("hallucination_rate"),
                               "abstention_rate": g.get("abstention_rate"),
                               "auto_eligible": sm["eligible"]},
                     "created": int(os.path.getmtime(m["path"])),
                     "loaded": bool(PROC["model"] and PROC["model"]["path"] == m["path"]),
                     "vision": bool(m.get("mmproj")), "arch": m.get("arch"),
                     "quant": m.get("quant"), "size": m["size"]})
    if any_eligible:
        data.insert(0, {"id": "auto", "object": "model", "owned_by": "neuronscope",
                        "description": "per prompt: subject classifier + rolling performance stats"})
    data += [{k: v for k, v in m.items() if k != "remote_id"} for m in link_models()]
    return {"object": "list", "data": data}


# ---------------------------------------------------------------- chat history

CHAT_ID_RE = re.compile(r"^[a-f0-9]{8,32}$")


def _chat_path(cid):
    if not CHAT_ID_RE.match(cid or ""):
        raise ValueError("bad chat id")
    return os.path.join(STATE["chats_dir"], f"{cid}.json")


def list_chats():
    d = STATE["chats_dir"]
    out = []
    if d and os.path.isdir(d):
        for f in glob.glob(os.path.join(d, "*.json")):
            try:
                with open(f) as fh:
                    c = json.load(fh)
                out.append({k: c.get(k) for k in ("id", "title", "updated", "model")} |
                           {"n": len(c.get("messages", []))})
            except Exception:
                continue
    return sorted(out, key=lambda c: -(c.get("updated") or 0))


def save_chat(chat):
    cid = chat.get("id") or hashlib.sha256(os.urandom(16)).hexdigest()[:16]
    path = _chat_path(cid)
    os.makedirs(STATE["chats_dir"], exist_ok=True)
    rec = {"id": cid, "title": str(chat.get("title") or "New chat")[:120],
           "model": chat.get("model"), "preset": chat.get("preset"),
           "messages": chat.get("messages", []), "updated": time.time()}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, path)
    return rec


# ---------------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    # -- auth ---------------------------------------------------------------
    # This endpoint loads models, downloads files and runs inference. On a LAN
    # that is not something to leave open, so --token gates everything except
    # the login page itself.
    _role = None      # "owner" (master token, or no token configured) or "device" (a paired device)

    def _presented(self) -> list[str]:
        out = []
        hdr = self.headers.get("Authorization", "")
        if hdr.startswith("Bearer "):
            out.append(hdr[7:])
        raw = self.headers.get("Cookie", "")
        if raw:
            try:
                c = http.cookies.SimpleCookie(raw)
                if "ns_token" in c:
                    out.append(c["ns_token"].value)
            except Exception:
                pass
        return out

    def _authed(self):
        self._role = None
        tok = STATE["token"]
        if not tok:
            self._role = "owner"
            return True
        presented = self._presented()
        if any(hmac.compare_digest(p, tok) for p in presented):
            self._role = "owner"
            return True
        reg = STATE.get("devices")
        if reg is not None and any(reg.check(p) for p in presented):
            self._role = "device"
            return True
        return False

    def _plugins_get(self):
        import plugins as ns_plugins
        pm = plugins()
        if self.path == "/api/plugins":
            return self._json(200, {"plugins": pm.summary(), "checks": pm.checks(),
                                    "dirs": [str(d) for d in pm.dirs]})
        m = re.fullmatch(r"/plugins/([a-z0-9][a-z0-9_-]{0,39})/(.*)", self.path.split("?")[0])
        if not m:
            return self._json(404, {"error": "not found"})
        d = re.fullmatch(r"download/([A-Za-z0-9_.-]{1,60}\.zip)", m.group(2))
        if d:
            try:
                body = pm.download(m.group(1), d.group(1))
            except ns_plugins.PluginError as e:
                return self._json(404, {"error": str(e)})
            if body is None:
                return self._json(404, {"error": "not found"})
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{d.group(1)}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        try:
            f = pm.page_file(m.group(1), urllib.parse.unquote(m.group(2)))
        except ns_plugins.PluginError as e:
            return self._json(404, {"error": str(e)})
        if f is None:
            return self._json(404, {"error": "not found"})
        body = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        # A plugin page may talk to Studio's API (same origin) but loads nothing from elsewhere.
        self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline' data: blob:")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _plugins_post(self):
        import plugins as ns_plugins
        pm = plugins()
        req = self._read(MAX_BODY)
        try:
            if self.path == "/api/plugins/enable":
                if not self._owner_only():
                    return
                p = pm.set_enabled(str(req.get("name", "")), bool(req.get("on")))
                if STATE.get("mcp") is not None:
                    threading.Thread(target=STATE["mcp"].connect, daemon=True).start()
                elif p["manifest"].get("mcp"):
                    threading.Thread(target=mcp_hub, daemon=True).start()
                return self._json(200, {"name": p["name"], "status": p["status"]})
            if self.path == "/api/plugins/check":
                payload = {"model": req.get("model"), "messages": req.get("messages") or [],
                           "text": str(req.get("text") or "")}
                return self._json(200, pm.run_check(str(req.get("plugin", "")), str(req.get("check", "")), payload))
        except ns_plugins.PluginError as e:
            return self._json(400, {"error": str(e)})
        return self._json(404, {"error": "not found"})

    def _owner_only(self):
        """Pairing, device management and links are for the owner, not for paired devices."""
        if self._role != "owner":
            self._json(403, {"error": "only the owner (master token) can do this"})
            return False
        return True

    def _deny(self):
        if self.path == "/" or self.path.startswith("/login"):
            body = LOGIN.encode()
            self.send_response(401)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json(401, {"error": "unauthorized"})

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        if self._route:
            self.send_header("X-NeuronScope-Route", json.dumps(self._route)[:4000])
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self, limit=MAX_SMALL_BODY):
        n = int(self.headers.get("Content-Length", 0))
        if n < 0 or n > limit:
            raise ValueError(f"request body too large (limit {limit} bytes)")
        return json.loads(self.rfile.read(n) or b"{}")

    _route = None

    def _send_headers_sse(self):
        self.send_response(200)
        if self._route:
            self.send_header("X-NeuronScope-Route", json.dumps(self._route)[:4000])
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _v1(self, path):
        """OpenAI-compatible passthrough with JIT model loading."""
        body = self._read(MAX_BODY)
        touch(+1)
        try:
            lt = link_target(body.get("model"))
            if lt is not None:
                upstream, rid = lt
                body = {**body, "model": rid}
                if body.get("stream"):
                    self._send_headers_sse()
                    proxy(path, body, stream_to=self.wfile, upstream=upstream)
                else:
                    self._json(200, proxy(path, body, upstream=upstream))
                return
            with LOCK:
                target, route = resolve_request_model(body)
                if target is None:
                    if route.get("error"):
                        return self._json(409, {"error": {"message": route["error"], "type": "invalid_request_error"}})
                    return self._json(404, {"error": {"message": f"model not found: {body.get('model')}" if body.get("model")
                                                      else "no model is loaded: name one (or auto), or load one under "
                                                           "Studio -> Models",
                                                      "type": "invalid_request_error"}})
                ok, msg = ensure_loaded(target)
            if not ok:
                return self._json(503, {"error": {"message": f"could not load {target['id']}: {msg}",
                                                  "type": "server_error"}})
            if _has_image(body) and not target.get("mmproj"):
                return self._json(400, {"error": {"message": f"{target['id']} has no mmproj vision projector",
                                                  "type": "invalid_request_error"}})
            self._route = route
            if body.get("stream"):
                self._send_headers_sse()
                out = proxy(path, body, stream_to=self.wfile)
                text = out["text"]
            else:
                out = proxy(path, body)
                text = ((out.get("choices") or [{}])[0].get("message") or {}).get("content") \
                    if isinstance(out, dict) else None
                self._json(200, out)
            if path == "/v1/chat/completions":
                note_reply(target, body.get("messages") or [], text or "")
        finally:
            touch(-1)

    def _pairing_post(self):
        if not self._owner_only():
            return
        req = self._read()
        reg = STATE["devices"]
        if self.path == "/api/pair/start":
            if not STATE["token"]:
                return self._json(400, {"error": "pairing needs Studio to run with a token (and TLS off loopback); "
                                                 "without one, nothing is protected to pair into"})
            try:
                code, exp, access = reg.new_code(req.get("persistent"), req.get("ttl"))
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            host = self.headers.get("Host") or "127.0.0.1"
            scheme = "https" if STATE["tls"] else "http"
            # `a` only tells the claiming page what it will get; the host enforces it.
            frag = (f"c={code[:4]}-{code[4:8]}-{code[8:]}"
                    + (f"&fp={STATE['fingerprint']}" if STATE.get("fingerprint") else "")
                    + f"&a={'p' if access is None else ns_pairing.fmt_duration(access)}")
            return self._json(200, {"code": code, "expires": exp, "link": f"{scheme}://{host}/pair#{frag}",
                                    "fingerprint": STATE.get("fingerprint"), "persistent": access is None,
                                    "access_seconds": access})
        if self.path == "/api/devices/revoke":
            return self._json(200, {"ok": reg.revoke(str(req.get("id", "")))})
        if self.path == "/api/devices/update":
            try:
                return self._json(200, reg.update(str(req.get("id", "")), req.get("persistent"), req.get("ttl")))
            except KeyError:
                return self._json(404, {"error": "no such device"})
            except ValueError as e:
                return self._json(400, {"error": str(e)})
        if self.path == "/api/links/add":
            name = re.sub(r"[^A-Za-z0-9_.-]", "-", str(req.get("name") or "remote"))[:32].strip("-") or "remote"
            links = [x for x in load_links() if x["name"] != name]
            try:
                r = ns_pairing.claim(str(req.get("link", "")), str(req.get("device_name") or "studio"))
            except (ValueError, RuntimeError, OSError) as e:
                return self._json(400, {"error": f"could not pair: {e}"})
            links.append({"name": name, "url": r["url"], "token": r["token"], "fingerprint": r["fingerprint"],
                          "device_id": r["device_id"], "added": time.time(), "expires": r.get("expires")})
            save_links(links)
            with LINKS["lock"]:
                LINKS["cache"].pop(name, None)
            return self._json(200, {"name": name, "models": [m["id"] for m in link_models(refresh=True)
                                                             if m["link"] == name]})
        if self.path == "/api/links/remove":
            name = str(req.get("name", ""))
            save_links([x for x in load_links() if x["name"] != name])
            with LINKS["lock"]:
                LINKS["cache"].pop(name, None)
            return self._json(200, {"ok": True})

    def _self_url(self, public=False):
        """This Studio's URL: the loopback address for calling itself, or the address
        the client used (Host header) for settings shown to it."""
        scheme = "https" if STATE["tls"] else "http"
        if public and self.headers.get("Host"):
            return f"{scheme}://{self.headers['Host']}"
        return f"{scheme}://127.0.0.1:{self.server.server_address[1]}"

    # ---------------------------------------------------------- MCP
    def _mcp_post(self):
        """MCP over streamable HTTP: JSON-RPC in, JSON-RPC out (scripts/ns_mcp.py).
        Every tool call goes back through this API with the caller's own token,
        so an agent can do exactly what the person or device it acts for can."""
        import ns_mcp
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if origin and urllib.parse.urlparse(origin).netloc != host:
            return self._json(403, {"error": "cross-origin MCP requests are refused"})
        if "application/json" not in self.headers.get("Content-Type", ""):
            return self._json(415, {"error": "MCP requests must be application/json"})
        try:
            payload = self._read(4 * 1024 * 1024)
        except ValueError as e:
            body = json.dumps(ns_mcp._err(None, -32700, f"parse error: {e}")).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        token = (self._presented() or [None])[0] if STATE["token"] else None
        studio = ns_mcp.Studio(self._self_url(), token, insecure_loopback=True)
        resp = ns_mcp.handle_payload(payload, studio)
        if resp is None:
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        return self._json(200, resp)

    # ---------------------------------------------------------- projects
    def _projects_get(self):
        if not self._owner_only():
            return
        dr = _director()
        store = DIRECTOR["store"]
        if self.path == "/projects":
            import projects_page
            body = projects_page.PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/workers":
            import workers_page
            body = workers_page.PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/workers/overview":
            reg = workers_registry()
            return self._json(200, {"workers": all_workers(), "routes": reg.load()["routes"],
                                    "subjects": dr.sc.SUBJECTS, "events": list(reg.events)[-80:][::-1],
                                    "telemetry": telemetry_snapshot(), "queue": work_queue(),
                                    "max_workers": STATE["max_workers"]})
        if self.path == "/api/workers":
            return self._json(200, {"workers": POOL.list(), "max_workers": STATE["max_workers"],
                                    "devices": [{k: d.get(k) for k in ("id", "backend", "name", "vendor", "memory_total",
                                                                       "memory_free", "unified", "estimated", "enabled",
                                                                       "twin_of")} for d in POOL.devices()],
                                    "hardware_config": STATE.get("hardware_path")})
        if self.path == "/api/projects":
            return self._json(200, {"projects": store.list(),
                                    "models": [{"id": m["id"], "size": m["size"]} for m in scan_models()],
                                    "subjects": dr.sc.SUBJECTS, "checks": bool(STATE["cett"]),
                                    "req_kinds": dr.nrq.PROJECT_KINDS,
                                    "req_features": {k: v["title"] for k, v in dr.nrq.FEATURES.items()},
                                    "req_hardware": dr.nrq.HARDWARE_ITEMS})
        m = re.fullmatch(r"/api/projects/([A-Za-z0-9_-]+)/requirements", self.path)
        if m:
            try:
                with store.lock(m.group(1)):
                    plan = store.load(m.group(1))
                    dr.check_requirements(plan, configured=_configured_bins())
                    store.save(plan)
            except dr.PlanError as e:
                return self._json(404, {"error": str(e)})
            return self._json(200, plan["requirements"])
        m = re.fullmatch(r"/api/projects/([A-Za-z0-9_-]+)", self.path)
        if m:
            try:
                plan = store.load(m.group(1))
            except dr.PlanError as e:
                return self._json(404, {"error": str(e)})
            plan.pop("source", None)
            dr._reqs(plan)                    # plans from before per-project requirements
            plan["skills"] = dr.skill_breakdown(plan["tasks"])
            plan["gaps"] = dr.coverage_gaps(plan["tasks"], plan.get("census"))
            plan["ready"] = [t["id"] for t in dr.ready_tasks(plan)] if plan["status"] == "running" else []
            plan["note"] = DIRECTOR["notes"].get(plan["id"])
            return self._json(200, plan)
        return self._json(404, {"error": "not found"})

    def _projects_post(self):
        if not self._owner_only():
            return
        dr = _director()
        store = DIRECTOR["store"]
        if self.path == "/api/workers/stop":
            req = self._read()
            m = find_model(req.get("model")) if req.get("model") else None
            POOL.stop(m["path"] if m else None)
            return self._json(200, {"workers": POOL.list()})
        if self.path.startswith("/api/workers/"):
            import worker_registry
            reg, req = workers_registry(), self._read()
            try:
                if self.path == "/api/workers/endpoint":
                    if req.get("remove"):
                        reg.remove_endpoint(str(req["remove"]))
                        return self._json(200, {"ok": True})
                    return self._json(200, reg.put_endpoint(req))
                if self.path == "/api/workers/control":
                    wid = str(req.get("id", ""))
                    if not wid:
                        raise worker_registry.RegistryError("which worker?")
                    out = reg.set_control(wid, paused=req.get("paused"), draining=req.get("draining"),
                                          max_parallel=req.get("max_parallel"), takes=req.get("takes"))
                    if req.get("stop") and worker_kind(wid) == "local":
                        m = find_model(wid)
                        if m:
                            POOL.stop(m["path"])
                            reg.event(wid, "stopped", "by the owner")
                    return self._json(200, out)
                if self.path == "/api/workers/routes":
                    return self._json(200, {"routes": reg.set_routes(req.get("routes") or {}, dr.sc.SUBJECTS)})
                if self.path == "/api/workers/test":
                    wid = str(req.get("id", ""))
                    if worker_kind(wid) == "local":
                        return self._json(400, {"error": "a local model starts when a task needs it; test it in the "
                                                         "chat instead"})
                    t0, usage = time.time(), {}
                    text = chat_off_machine(wid, [{"role": "user", "content": "Reply with the single word: ready"}],
                                            usage)
                    reg.event(wid, "tested", f"{time.time() - t0:.1f}s: {text[:60]!r}")
                    return self._json(200, {"reply": text[:500], "seconds": round(time.time() - t0, 2),
                                            "usage": usage})
            except (worker_registry.RegistryError, ValueError) as e:
                return self._json(400, {"error": str(e)})
            except Exception as e:
                return self._json(502, {"error": f"{type(e).__name__}: {e}"})
            return self._json(404, {"error": "not found"})
        try:
            if self.path == "/api/projects":
                req = self._read(4 * 1024 * 1024)
                text = str(req.get("text") or "")
                if not text.strip():
                    return self._json(400, {"error": "describe the project: paste its notes, task list or README"})
                census = dr.analyze_repo(req["repo"]) if req.get("repo") else None
                plan = dr.new_plan(str(req.get("title") or ""), str(req.get("goal") or ""),
                                   dr.analyze_text(text), source=text, census=census)
                store.save(plan)
                return self._json(200, {"id": plan["id"]})
            m = re.fullmatch(r"/api/projects/([A-Za-z0-9_-]+)/(edit|approve|start|pause|decide|review|answer|"
                             r"policy|refine|delete|requirements|requirements/scan)", self.path)
            if not m:
                return self._json(404, {"error": "not found"})
            pid, action = m.groups()
            req = self._read(1024 * 1024)
            if action == "refine":
                note = DIRECTOR["notes"].get(pid)
                if note and note.get("busy"):
                    return self._json(409, {"error": "already re-planning"})
                DIRECTOR["notes"][pid] = {"at": time.time(), "text": "re-planning…", "busy": True}
                threading.Thread(target=refine_project, args=(pid, req.get("model")), daemon=True).start()
                return self._json(200, {"ok": True})
            with store.lock(pid):
                plan = store.load(pid)
                out = {}
                if action == "delete":
                    if plan["status"] == "running":
                        return self._json(409, {"error": "pause the project first"})
                    import shutil
                    shutil.rmtree(store.root / pid)
                    return self._json(200, {"ok": True})
                if action == "edit":
                    out = dr.edit(plan, req.get("changes") or [], by="human", reason=str(req.get("reason") or ""))
                elif action == "approve":
                    dr.approve(plan, assign_fn=assign_fn_for(plan))
                elif action == "start":
                    # Nothing starts on a machine that lacks what the project needs,
                    # unless the person says so.
                    if (plan.get("requirements") or {}).get("items") and not req.get("force"):
                        last = dr.check_requirements(plan, configured=_configured_bins())
                        if last["missing"]:
                            store.save(plan)
                            return self._json(409, {"error": "this machine is missing what the project needs: "
                                                             + ", ".join(last["missing"])
                                                             + " (fix it, or start anyway)",
                                                    "missing": last["missing"]})
                    dr.set_running(plan, True)
                elif action == "pause":
                    dr.set_running(plan, False)
                elif action == "requirements":
                    out = dr.set_requirements(plan, req.get("items"), reason=str(req.get("reason") or ""))
                elif action == "requirements/scan":
                    root = str(req.get("root") or (plan.get("requirements") or {}).get("root")
                               or (plan.get("census") or {}).get("path") or "")
                    if not root:
                        return self._json(400, {"error": "which folder? give the project's checkout"})
                    try:
                        out = dr.scan_requirements(plan, root, by="human")
                    except ValueError as e:
                        return self._json(400, {"error": str(e)})
                elif action == "decide":
                    out = dr.decide(plan, str(req.get("proposal")), bool(req.get("accept")),
                                    note=str(req.get("note") or ""))
                elif action == "review":
                    out = dr.review(plan, str(req.get("task")), bool(req.get("accept")),
                                    str(req.get("feedback") or ""), next_model_fn=next_model_fn)
                elif action == "answer":
                    out = dr.answer(plan, str(req.get("task")), [str(x) for x in req.get("answers") or []])
                elif action == "policy":
                    out = dr.set_policy(plan, req.get("policy") or {})
                store.save(plan)
            return self._json(200, {"ok": True, "result": out, "version": plan["version"], "status": plan["status"]})
        except dr.PlanError as e:
            return self._json(400, {"error": str(e)})

    def _jobs_off(self):
        return self._json(403, {"error": STATE.get("jobs_off") or "jobs are disabled"})

    def _jobs_get(self):
        runner = STATE.get("jobs")
        if self.path == "/api/jobs/specs":
            return self._json(200, ns_jobs.public_specs())
        if self.path == "/api/jobs/defaults":
            # What the forms can fill in from the running workspace.
            ws = STATE.get("workspace") or {}
            ex = lambda p: p if p and os.path.exists(p) else None   # noqa: E731
            return self._json(200, {"gguf": ex(ws.get("gguf")), "cett": ex(STATE.get("cett") or ws.get("cett")),
                                    "server": ex(STATE.get("server_bin") or ws.get("server")),
                                    "llama": ex(ws.get("llama_dir")), "studio": self._self_url(public=False) + "/v1",
                                    "workspace": STATE.get("workspace_name"), "root": ws.get("root"),
                                    "traces": STATE.get("traces_dir"),       # where the 3D views look
                                    "ggufs": [m["path"] for m in scan_models()][:200], **job_suggestions(ws)})
        if runner is None and not self.path.startswith(("/api/jobs/targets", "/api/jobs/options")):
            return self._jobs_off()
        if self.path == "/api/jobs":
            mine = runner.list() if self._role == "owner" else []
            return self._json(200, sorted(mine + remote_jobs(), key=lambda m: -m.get("started", 0)))
        if self.path == "/api/jobs/targets":
            return self._json(200, job_targets())
        if self.path.startswith("/api/jobs/options"):
            kind = urllib.parse.parse_qs(self.path.split("?", 1)[-1]).get("kind", [""])[0]
            if kind not in ns_jobs.all_specs():
                return self._json(404, {"error": "no such job kind"})
            return self._json(200, ns_jobs.script_options(kind))
        m = re.fullmatch(r"/api/jobs/host/([^/]+)/([0-9a-f]{12})/log", self.path)
        if m:
            try:
                return self._json(200, remote().call(urllib.parse.unquote(m.group(1)), "job_log",
                                                     {"id": m.group(2)}, timeout=15))
            except Exception as e:
                return self._json(502, {"error": f"host: {e}"})
        m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/log", self.path)
        if m:
            try:
                return self._json(200, {"log": runner.log(m.group(1))})
            except (ValueError, FileNotFoundError):
                return self._json(404, {"error": "no such job"})
        return self._json(404, {"error": "not found"})

    # ---------------------------------------------------------------- features (docs/FEATURES.md)
    def _features_ws(self):
        ws = STATE.get("workspace") or {}
        return (ws.get("root") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                list(load_settings().get("_features_dirs") or []))

    def _features_get(self):
        import features_api as FA
        if self.path == "/features":
            import features_page
            return self._html(features_page.PAGE)
        root, extra = self._features_ws()
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        arg = lambda k, d="": (q.get(k) or [d])[0]                              # noqa: E731
        route = urllib.parse.urlsplit(self.path).path
        try:
            if route == "/api/features/list":
                return self._json(200, FA.list_all(root, extra))
            if route == "/api/features/lab/info":
                # a status probe: "not running" is an answer, not an error the browser logs
                try:
                    return self._json(200, {"running": True, **features_lab_call(arg("lab"), "info", None, timeout=5)})
                except OSError as e:
                    return self._json(200, {"running": False, "error": f"{type(e).__name__}: {e}"})
            p = FA.allowed(arg("path"), root, extra)
            if route == "/api/features/dict":
                return self._json(200, FA.dictionary(p))
            if route == "/api/features/feature":
                return self._json(200, FA.feature(p, int(arg("id", "0"))))
            if route == "/api/features/search":
                return self._json(200, {"hits": FA.search(p, arg("q"))})
            if route == "/api/features/emotions":
                return self._json(200, FA.emotions(p))
            if route == "/api/features/graph":
                return self._json(200, FA.graph(p))
        except (ValueError, KeyError) as e:
            return self._json(400, {"error": str(e)})
        except FileNotFoundError as e:
            return self._json(404, {"error": f"not found: {e.filename or e}"})
        except OSError as e:
            return self._json(502, {"error": str(e)})
        return self._json(404, {"error": "not found"})

    # ---------------------------------------------------------- add-ons (addons.py)

    def _addon_removed(self) -> bool:
        """A request to a part of Studio whose add-on was removed: say so, and how to get it back."""
        name = addon_builtins().blocks(self.path)
        if not name:
            return False
        import addons
        msg = f"The {addons.BUILTIN[name]['title']} add-on is removed. Reinstall it under Add-ons (/addons)."
        if self.path.startswith("/api/"):
            self._json(404, {"error": msg, "addon": name})
        else:
            self._html(f"<!doctype html><meta charset=utf-8><title>Removed</title><body style='font:15px system-ui;"
                       f"padding:2rem'><p>{msg}</p><p><a href='/addons'>Add-ons</a></p>")
        return True

    def _addons(self, req):
        import addons
        b, pk = addon_builtins(), addon_packages()
        path = urllib.parse.urlsplit(self.path).path
        try:
            if req is None:
                if path == "/addons":
                    import addons_page
                    return self._html(addons_page.PAGE)
                if path == "/api/addons":
                    pk_rows = plugins().summary()
                    inst = str(pk.dir.resolve())
                    for r in pk_rows:
                        r["removable"] = str(Path(r["folder"]).resolve().parent) == inst
                        r["tab"] = ((next((x for x in plugins().scan() if x["name"] == r["name"]), {}).get("manifest") or {})
                                    .get("tab"))
                    return self._json(200, {"builtin": b.list(), "packages": pk_rows, "repos": pk.repos(),
                                            "install_dir": inst})
                if path == "/api/addons/catalog":
                    return self._json(200, {"addons": pk.catalog()})
                if path == "/api/addons/nav":
                    tabs = []
                    for p in plugins().enabled():
                        t = p["manifest"].get("tab")
                        if isinstance(t, dict) and p["manifest"].get("page"):
                            tabs.append({"href": f"/plugins/{p['name']}/", "label": str(t.get("label") or p["name"])[:24],
                                         "tip": str(t.get("tip") or p["manifest"].get("description") or "")[:160]})
                    return self._json(200, {"off": b.nav_off(), "tabs": tabs})
                return self._json(404, {"error": "not found"})
            if path == "/api/addons/builtin":
                changed = b.set(str(req.get("name")), bool(req.get("installed")), bool(req.get("with_dependents")))
                return self._json(200, {"builtin": b.list(), "changed": changed})
            if path == "/api/addons/install":
                out = pk.install_from_repo(str(req.get("name") or ""), req.get("repo"))
                STATE.pop("mcp", None)
                return self._json(200, out)
            if path == "/api/addons/uninstall":
                pk.uninstall(str(req.get("name") or ""))
                return self._json(200, {"ok": True})
            if path == "/api/addons/repos":
                return self._json(200, {"repos": pk.set_repos(req.get("repos") or [])})
        except (addons.AddonError, ValueError) as e:
            return self._json(400, {"error": str(e)[:500]})
        except OSError as e:
            return self._json(502, {"error": f"{type(e).__name__}: {e}"[:500]})
        return self._json(404, {"error": "not found"})

    def _addon_upload(self):
        """A package file (.zip) as the request body."""
        import addons
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > addons.MAX_PACKAGE:
            return self._json(413, {"error": "send a .zip of at most 200 MB"})
        data = self.rfile.read(n)
        try:
            return self._json(200, addon_packages().install_bytes(data))
        except (addons.AddonError, ValueError) as e:
            return self._json(400, {"error": str(e)[:500]})

    # ---------------------------------------------------------- node graphs (graph_engine.py)

    def _graph_env(self, req):
        import graph_engine as G
        import ns_mcp as studio_client
        token = (self._presented() or [None])[0] if STATE["token"] else None
        st = studio_client.Studio(self._self_url(), token, insecure_loopback=True)

        def mcp_call(server, tool, args):
            import mcp_client
            hub = mcp_hub()
            if hub is None:
                raise G.GraphError("no MCP servers are configured (Settings -> Connectors)")
            fn = server + mcp_client.SEP + mcp_client._safe(tool)
            info = hub.tool_info(fn)
            d = tool_rules().decide(info["server"], info["tool"], args, info["annotations"], auto_approve=info["auto_approve"])
            if d["action"] != "allow":
                raise G.GraphError(f"tool approval says {d['action']}: {d['reason']} (Settings -> Tool approval)")
            return hub.call(fn, args)
        store = self._graph_store()
        return G.Env(st, permissions(), model=str(req.get("model") or ""), profile=str(req.get("profile") or ""),
                     workdir=req.get("workdir") or None, mcp_call=mcp_call, graph_loader=store.get)

    def _work(self):
        """/work and its API: the work log's breakdown, and activation stats from the review store."""
        u = urllib.parse.urlsplit(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        try:
            if u.path == "/work":
                import work_page
                return self._html(work_page.PAGE)
            if u.path == "/api/work":
                return self._json(200, ns_work.breakdown(work_log(), q.get("since", "24h")))
            if u.path == "/api/work/predict":
                return self._json(200, speed_predictions(q.get("since", "30d")))
            if u.path == "/api/work/measures":
                return self._json(200, measurement_weights())
            if u.path == "/api/work/neurons":
                store = review_store()
                if not q.get("model"):
                    return self._json(200, {"models": store.models()})
                import neuron_review
                verdicts = [q["verdict"]] if q.get("verdict") else None
                return self._json(200, neuron_review.activation_stats(store, q["model"], verdicts=verdicts))
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        return self._json(404, {"error": "not found"})

    def _graph_store(self):
        import graphs_api
        return graphs_api.Store(STATE.get("graphs_dir") or os.path.expanduser("~/.neuronscope/graphs"))

    def _graphs(self, req):
        import graph_engine as G
        import graphs_api as GA
        store = self._graph_store()
        G.sync_plugins(plugins())                  # node types from enabled plugins
        G.HIDDEN = addon_builtins().node_hidden    # and none from removed add-ons
        path = urllib.parse.urlsplit(self.path).path
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        try:
            if req is None:
                if path == "/graphs":
                    import graphs_page
                    return self._html(graphs_page.PAGE)
                if path == "/api/graphs":
                    return self._json(200, GA.catalog(store))
                if path == "/api/graphs/get":
                    return self._json(200, store.get((q.get("name") or [""])[0]))
                if path == "/api/graphs/runs":
                    return self._json(200, {"runs": store.runs(50)})
                m = re.fullmatch(r"/api/graphs/runs/([0-9a-f]{12})", path)
                if m:
                    return self._json(200, store.run(m.group(1)))
                return self._json(404, {"error": "not found"})
            graph = req.get("graph") or (store.get(str(req["name"])) if req.get("name") else None)
            if path == "/api/graphs/save":
                return self._json(200, store.save(req.get("graph") or {}))
            if path == "/api/graphs/delete":
                store.delete(str(req.get("name") or ""))
                return self._json(200, {"ok": True})
            if path == "/api/graphs/run":
                return self._json(200, {"id": GA.start(store, graph, req.get("inputs") or {}, self._graph_env(req))})
            if path == "/api/graphs/eval":
                cases = req.get("cases")
                if isinstance(cases, str):
                    cases = [json.loads(x) for x in cases.splitlines() if x.strip()]
                return self._json(200, {"id": GA.start_eval(store, graph, cases or [], self._graph_env(req),
                                                            str(req.get("expect") or ""), str(req.get("output") or "result"),
                                                            str(req.get("grade") or "contains"))})
            if path == "/api/graphs/node":
                return self._json(200, GA.run_node(graph, str(req.get("node") or ""), req.get("inputs") or {},
                                                   self._graph_env(req)))
            m = re.fullmatch(r"/api/graphs/runs/([0-9a-f]{12})/decide", path)
            if m:
                return self._json(200, GA.decide(store, m.group(1), str(req.get("node") or ""), bool(req.get("approve")),
                                                 req.get("value"), str(req.get("note") or "")))
        except FileNotFoundError as e:
            return self._json(404, {"error": f"not found: {e}"})
        except (ValueError, KeyError, TypeError) as e:
            return self._json(400, {"error": str(e)[:500]})
        return self._json(404, {"error": "not found"})

    def _features_post(self, req):
        import features_api as FA
        root, extra = self._features_ws()
        try:
            if self.path == "/api/features/label":
                return self._json(200, FA.set_label(FA.allowed(req.get("path"), root, extra), int(req.get("id")),
                                                    str(req.get("label") or "")))
            if self.path.startswith("/api/features/lab/"):
                op = self.path.rsplit("/", 1)[-1]
                body = {k: v for k, v in req.items() if k != "lab"}
                for k in ("sae", "transcoders"):            # the lab reads these files: workspace runs/ only
                    if isinstance(body.get(k), list):
                        body[k] = [str(FA.allowed(x, root, extra)) for x in body[k]]
                for k in ("emotions", "save"):
                    if body.get(k):
                        body[k] = str(FA.allowed(body[k], root, extra))
                st = body.get("steer") or {}
                if (st.get("feature") or {}).get("sae"):
                    st["feature"]["sae"] = str(FA.allowed(st["feature"]["sae"], root, extra))
                if (st.get("emotion") or {}).get("file"):
                    st["emotion"]["file"] = str(FA.allowed(st["emotion"]["file"], root, extra))
                if isinstance(body.get("sae"), str):
                    body["sae"] = str(FA.allowed(body["sae"], root, extra))
                return self._json(200, features_lab_call(req.get("lab"), op, body))
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error")
            except Exception:
                msg = None
            return self._json(e.code, {"error": msg or f"lab: HTTP {e.code}"})
        except OSError as e:
            return self._json(502, {"error": f"the decoding lab is not answering ({e}); start it from this page"})
        return self._json(404, {"error": "not found"})

    def _jobs_post(self):
        runner = STATE.get("jobs")
        if runner is None:
            return self._jobs_off()
        if "application/json" not in self.headers.get("Content-Type", ""):
            return self._json(415, {"error": "JSON only"})     # no cross-site form posts
        req = self._read()
        host = str(req.get("host") or "") or (str(req.get("target") or "")[5:] if str(req.get("target") or "").startswith("host:") else "")
        if host:
            try:
                if self.path == "/api/jobs/cancel":
                    out = remote().call(host, "cancel_job", {"id": str(req.get("id", ""))}, timeout=15)
                else:
                    out = remote().call(host, "run_job", {"kind": str(req.get("kind", "")),
                                                         "values": dict(req.get("values") or {})}, timeout=30)
                with _RLOCK:
                    _RCACHE.pop("jobs:" + host, None)
                return self._json(200, {**out, "host": host})
            except Exception as e:
                return self._json(502, {"error": f"{host}: {e}"})
        try:
            if self.path == "/api/jobs/cancel":
                return self._json(200, runner.cancel(str(req.get("id", ""))))
            return self._json(200, runner.start(str(req.get("kind", "")), dict(req.get("values") or {})))
        except (ValueError, FileNotFoundError) as e:
            return self._json(400, {"error": str(e)})
        except RuntimeError as e:
            return self._json(429, {"error": str(e)})

    def _rag_post(self):
        import base64
        store = rag_store()
        try:
            if self.path == "/api/rag/upload":
                req = self._read(48 * 1024 * 1024)
                data = base64.b64decode(str(req.get("data", "")).split(",", 1)[-1], validate=False)
                if len(data) > 32 * 1024 * 1024:
                    return self._json(413, {"error": "file larger than 32 MB"})
                coll = store.get(str(req.get("collection", "")), create=True)
                emb = rag_embedder() if (coll.dense is not None or (not coll.chunks and
                                                                    req.get("dense", True))) else None
                return self._json(200, coll.add(str(req.get("filename", "upload.txt")), data, emb))
            req = self._read()
            name = str(req.get("collection", ""))
            if self.path == "/api/rag/create":
                return self._json(200, store.get(name, create=True).info())
            if self.path == "/api/rag/delete":
                if req.get("doc"):
                    return self._json(200, {"removed": store.get(name).remove(str(req["doc"]))})
                store.drop(name)
                return self._json(200, {"ok": True})
            if self.path == "/api/rag/search":
                coll = store.get(name)
                emb = rag_embedder() if coll.dense is not None else None
                return self._json(200, coll.search(str(req.get("query", "")), int(req.get("k", 4)), emb))
        except FileNotFoundError as e:
            return self._json(404, {"error": str(e)})
        except (ValueError, RuntimeError, OSError) as e:
            return self._json(400, {"error": str(e)[:300]})
        return self._json(404, {"error": "not found"})

    # ------------------------------------------------ remote hosts (this Studio is the controller)

    def _viz_theme(self, default):
        """The 3D views follow the viewer's own theme (cookie ns_viz, set by the theme picker)."""
        import bloom
        try:
            c = http.cookies.SimpleCookie(self.headers.get("Cookie") or "")
            name = c["ns_viz"].value if "ns_viz" in c else ""
        except Exception:
            name = ""
        import widgets
        return widgets.with_widgets(bloom.THEMES.get(name) or default, STATE.get("widgets"))

    def _host_token(self):
        auth = self.headers.get("Authorization", "")
        tok = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        rec = remote().host_for_token(tok) if tok else None
        if rec is None:
            THROTTLE.fail(self.client_address[0])
            self._json(401, {"error": "this host is not joined to this Studio (removed?): join again with a new invite"})
        return rec

    def _remote_public(self, method):
        """The routes a host that calls in uses. They carry the host's own token, never the owner's."""
        ip = self.client_address[0]
        if THROTTLE.blocked(ip):
            return self._json(429, {"error": "too many attempts; wait a few minutes"})
        path = self.path.split("?")[0]
        hub = remote().hub
        if method == "POST" and path == "/api/remote/claim":
            req = self._read()
            got = hub.claim(str(req.get("proof", "")))
            if got is None:
                THROTTLE.fail(ip)
                return self._json(403, {"error": "the invite is wrong, used or expired: make a new one in Studio "
                                                 "(Remote → Add a host)"})
            inv, code = got
            name = re.sub(r"[^A-Za-z0-9_.-]", "-", inv["name"] or str(req.get("name") or "host"))[:32].strip("-") or "host"
            tok = secrets.token_urlsafe(32)
            import remote_hosts
            remote().add({"name": name, "method": "callhome", "url": f"callhome:{name}", "token": "",
                          "host_token_sha256": remote_hosts.token_hash(tok), "host": str(req.get("name", ""))[:40],
                          "studio_fingerprint": inv["fingerprint"], "agent_version": str(req.get("version", ""))[:20]})
            with LINKS["lock"]:
                LINKS["cache"].pop(name, None)
            return self._json(200, {"token": tok, "name": name, "controller": inv["controller"],
                                    "confirm": remote_hosts.nsr.pair_proof(code, inv["fingerprint"], "accept")})
        rec = None
        if path in ("/api/remote/next", "/api/remote/reply", "/api/remote/progress"):
            rec = self._host_token()
            if rec is None:
                return
        if method == "GET" and path == "/api/remote/next":
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            return self._json(200, {"messages": hub.next(rec["name"], float((q.get("wait") or ["25"])[0]))})
        if method == "POST" and path == "/api/remote/reply":
            req = self._read(MAX_BODY)
            return self._json(200, {"ok": hub.reply(str(req.get("id", "")), req.get("response"), rec["name"])})
        if method == "POST" and path == "/api/remote/progress":
            req = self._read(MAX_BODY)
            pieces = req.get("pieces") if isinstance(req.get("pieces"), list) else []
            return self._json(200, hub.progress(str(req.get("id", "")), pieces[:4096], rec["name"]))
        m = re.fullmatch(r"/api/remote/file/([A-Za-z0-9_-]+)", path)
        if method == "GET" and m:
            fp = hub.ticket_path(m.group(1))
            if not fp or not os.path.isfile(fp):
                THROTTLE.fail(ip)
                return self._json(404, {"error": "no such file, or the link expired"})
            return self._send_file(fp)
        return self._json(404, {"error": "not found"})

    def _send_file(self, fp):
        size = os.path.getsize(fp)
        start = 0
        rng = re.fullmatch(r"bytes=(\d+)-", self.headers.get("Range", "").strip())
        if rng:
            start = min(int(rng.group(1)), size)
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size - start))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{size - 1}/{size}")
        self.end_headers()
        with open(fp, "rb") as f:
            f.seek(start)
            shutil.copyfileobj(f, self.wfile, 1 << 20)

    def _remote_state(self):
        import remote_hosts
        r = remote()
        hosts = []
        for h in r.hosts():
            st = r.status(h)
            with LINKS["lock"]:
                hit = LINKS["cache"].get(h["name"])
            st["models"] = [m["remote_id"] for m in (hit[1] if hit else [])]
            st["error"] = hit[2] if hit else None
            hosts.append(st)
        bind = STATE.get("bind_host") or "127.0.0.1"
        return {"role": "controller", "hosts": hosts, "sends": list(r.sends.values())[-20:],
                "studio": {"bind": bind, "network": not sec.is_loopback(bind), "tls": bool(STATE.get("tls")),
                           "token": bool(STATE.get("token")),
                           "short_fingerprint": remote_hosts.nsr.short_fp(STATE.get("fingerprint") or ""),
                           "url": self._self_url(public=True)},
                "local_models": [{"id": m["id"], "path": m["path"], "size": m["size"]} for m in scan_models()]}

    def _remote_get(self):
        if not self._owner_only():
            return
        if self.path == "/remote":
            import remote_page
            return self._html(remote_page.PAGE)
        if self.path == "/remote/download":
            import remote_hosts
            body = remote_hosts.package_zip()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", 'attachment; filename="neuronscope-remote.zip"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/api/remote", "/api/remote?refresh=1"):
            if self.path.endswith("refresh=1"):
                link_models(refresh=True)
            return self._json(200, self._remote_state())
        return self._json(404, {"error": "not found"})

    def _remote_post(self):
        if not self._owner_only():
            return
        import remote_hosts
        r = remote()
        req = self._read()
        controller = f"OmniCortex Studio ({socket.gethostname()})"
        try:
            if self.path == "/api/remote/discover":
                return self._json(200, {"hosts": remote_hosts.discover(float(req.get("seconds") or 3))})
            if self.path == "/api/remote/classifier":
                host, model, clf = str(req.get("host", "")), str(req.get("model", "")), str(req.get("classifier") or "")
                r.get(host)
                if clf and not os.path.isfile(os.path.expanduser(clf)):
                    return self._json(400, {"error": f"no such file: {clf}"})
                allset = load_settings()
                key = _key("remote:" + host + LINK_SEP + model)
                allset.setdefault(key, {})["classifier"] = os.path.expanduser(clf) if clf else None
                save_settings(allset)
                return self._json(200, {"ok": True})
            if self.path == "/api/remote/pair":
                method = req.get("method") or "lan"
                address, code, fp = str(req.get("address", "")).strip(), str(req.get("code", "")), ""
                if req.get("link"):
                    inv = remote_hosts.nsr.parse_invite(str(req["link"]))
                    address, code, fp = inv["url"], inv["code"], inv["fingerprint"]
                    req.setdefault("name", inv["name"])
                ssh = None
                if method == "ssh":
                    ssh = {"target": str(req.get("ssh_target", "")).strip(), "port": req.get("ssh_port") or None,
                           "remote_port": int(req.get("remote_port") or remote_hosts.nsr.DEFAULT_PORT),
                           "local_port": remote_hosts.Tunnels.free_port()}
                    if not ssh["target"]:
                        raise remote_hosts.RemoteError("the SSH login is needed, like you@host")
                    tmp = {"name": "__pairing__", "ssh": ssh}
                    r.tunnels.ensure(tmp)
                    address = f"http://127.0.0.1:{ssh['local_port']}"
                if not address or not remote_hosts.nsr.norm_code(code):
                    raise remote_hosts.RemoteError("the host's address and its pairing code are needed")
                try:
                    out = remote_hosts.pair_direct(address, code, controller, fp)
                finally:
                    if ssh:
                        r.tunnels.stop("__pairing__")
                name = re.sub(r"[^A-Za-z0-9_.-]", "-", str(req.get("name") or out["host"] or "host"))[:32].strip("-") \
                    or "host"
                rec = {"name": name, "method": method, "url": out["url"], "token": out["token"],
                       "fingerprint": out["fingerprint"], "host": out["host"]}
                if ssh:
                    rec["ssh"] = ssh
                r.add(rec)
                with LINKS["lock"]:
                    LINKS["cache"].pop(name, None)
                models = [m["remote_id"] for m in link_models(refresh=True) if m["link"] == name]
                return self._json(200, {"name": name, "host": out["host"], "models": models,
                                        "short_fingerprint": remote_hosts.nsr.short_fp(out["fingerprint"]),
                                        "error": (LINKS["cache"].get(name) or (0, [], None))[2]})
            if self.path == "/api/remote/invite":
                url = str(req.get("url") or self._self_url(public=True)).rstrip("/")
                same = urllib.parse.urlsplit(url).netloc == (self.headers.get("Host") or "")
                fp = (STATE.get("fingerprint") or "") if (STATE.get("tls") and same) else ""
                name = re.sub(r"[^A-Za-z0-9_.-]", "-", str(req.get("name") or ""))[:32].strip("-")
                out = r.hub.invite(url, name, fp, controller)
                warn = None
                host = urllib.parse.urlsplit(url).hostname or ""
                if sec.is_loopback(host) or sec.is_loopback(STATE.get("bind_host") or "127.0.0.1"):
                    warn = ("This Studio listens on this computer only, so another computer cannot call it. Start "
                            "Studio for the network (with a token and TLS, see Security), or put a tunnel in front "
                            "of it and give its address here.")
                out["warning"] = warn
                out["command"] = f"neuronscope-remote connect {out['invite']}"
                return self._json(200, out)
            if self.path == "/api/remote/call":
                return self._json(200, r.call(str(req["host"]), str(req["tool"]), req.get("args") or {},
                                              timeout=float(req.get("timeout") or 900)))
            if self.path == "/api/remote/tools":
                return self._json(200, {"tools": r.tools(str(req["host"]))})
            if self.path == "/api/remote/send":
                path = str(req.get("path", ""))
                if path not in {m["path"] for m in scan_models()} and not req.get("any_file"):
                    raise remote_hosts.RemoteError("pick one of this Studio's models to send")
                return self._json(200, {"id": r.send(str(req["host"]), path, self._self_url(public=True),
                                                       bool(req.get("install")))})
            if self.path == "/api/remote/remove":
                r.remove(str(req.get("host", "")))
                with LINKS["lock"]:
                    LINKS["cache"].pop(str(req.get("host", "")), None)
                return self._json(200, {"ok": True})
        except remote_hosts.RemoteError as e:
            return self._json(400, {"error": str(e)})
        except (ValueError, KeyError, OSError) as e:
            return self._json(400, {"error": f"{type(e).__name__}: {e}"})
        return self._json(404, {"error": "not found"})

    # ------------------------------------------------ neuron review

    def _html(self, page, code=200):
        body = page.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _review_get(self):
        if not self._owner_only():
            return
        if self.path == "/review":
            import review_page
            return self._html(review_page.PAGE)
        if self.path == "/api/review":
            st = review_store()
            ms = st.models()
            return self._json(200, {"models": ms, "facets": {m["model"]: st.facets(m["model"]) for m in ms},
                                    "store": str(st.root)})
        m = re.fullmatch(r"/review/view/([0-9a-f]{12})/(|api/meta|api/theme|api/trace)", self.path)
        with REVIEW_VIEWS["lock"]:
            pl = REVIEW_VIEWS["views"].get(m.group(1)) if m else None
        if pl is None:
            return self._json(404, {"error": "no such view (views last until Studio restarts)"})
        import bloom
        kind = m.group(2)
        if kind == "":
            return self._html(bloom.PAGE)
        if kind == "api/trace":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(pl["blob"])))
            self.end_headers()
            self.wfile.write(pl["blob"])
            return
        return self._json(200, pl["meta"] if kind == "api/meta" else self._viz_theme(pl["theme"]))

    def _review_post(self):
        if not self._owner_only():
            return
        import neuron_review as nrv
        req = self._read()
        st = review_store()
        model = str(req.get("model") or "")
        if self.path == "/api/review/label":
            try:
                st.label(model, str(req.get("id") or ""), str(req.get("verdict") or ""))
            except KeyError as e:
                return self._json(404, {"error": str(e).strip("'\"")})
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            return self._json(200, {"ok": True})
        if self.path not in ("/api/review/summary", "/api/review/view"):
            return self._json(404, {"error": "not found"})
        if not any(m["model"] == model for m in st.models()):
            return self._json(404, {"error": f"no observations for {model!r}"})
        stat = req.get("stat") or "association"
        how = req.get("by") or "week"
        if stat not in nrv.STATS or how not in nrv.BUCKETS:
            return self._json(400, {"error": f"stat is one of {nrv.STATS}, by one of {nrv.BUCKETS}"})

        def lst(k):
            v = req.get(k)
            return [str(x) for x in v] if isinstance(v, list) and v else None
        filters = {"sources": lst("sources"), "kinds": lst("kinds"), "subjects": lst("subjects"),
                   "verdicts": lst("verdicts"),
                   "since": float(req["since"]) if req.get("since") else None,
                   "until": float(req["until"]) if req.get("until") else None}
        every = max(0, int(req.get("every") or 0))
        try:
            if self.path == "/api/review/summary":
                return self._json(200, nrv.summary(st, model, stat, how, every, top=int(req.get("top") or 25),
                                                   **filters))
            blob, meta = nrv.view_payload(st, model, stat, how, every, **filters)
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        import bloom
        vid = uuid.uuid4().hex[:12]
        theme = dict(bloom.THEMES.get(STATE.get("viz_theme") or "dark") or next(iter(bloom.THEMES.values())))
        with REVIEW_VIEWS["lock"]:
            if len(REVIEW_VIEWS["views"]) >= 8:
                REVIEW_VIEWS["views"].pop(next(iter(REVIEW_VIEWS["views"])))
            REVIEW_VIEWS["views"][vid] = {"blob": blob, "meta": meta, "theme": theme}
        return self._json(200, {"id": vid, "url": f"/review/view/{vid}/", "frames": meta["frames"],
                                "cells": meta["cells"]})

    # ------------------------------------------------ settings: workspaces, subject defaults, tool rules

    def _settings(self):
        if not self._owner_only():
            return
        import ns_workspace
        if self.command == "GET":
            if self.path == "/settings":
                import settings_page
                return self._html(settings_page.PAGE)
            if self.path == "/api/workspaces":
                return self._json(200, workspaces_state())
            if self.path == "/api/settings/sharing":
                import score_cards
                cache = STATE.get("hf_checks", {})
                return self._json(200, {"settings": sharing(), "github_token_set": bool(score_cards.github_token()),
                                        "models": [{"id": m["id"], "path": m["path"], "check": cache.get(m["path"])}
                                                   for m in scan_models() if m.get("format") != "hf"],
                                        "results": sharing_results(), "history": score_cards.submissions(),
                                        "errors": STATE.get("sharing_errors", [])[-10:]})
            if self.path == "/api/settings/permissions":
                return self._json(200, permissions_state())
            return self._json(200, settings_state())
        req = self._read()
        path = _ws_path()
        try:
            if self.path == "/api/workspaces/save":
                name = str(req.get("name") or "")
                ns_workspace.save(name, req.get("values") or {}, path=path, replace=bool(req.get("replace")))
                out = workspaces_state()
                out["restart_needed"] = name == STATE.get("workspace_name")
                return self._json(200, out)
            if self.path == "/api/workspaces/use":
                ns_workspace.use(str(req.get("name") or ""), path=path)
                out = workspaces_state()
                out["restart_needed"] = out["active"] != STATE.get("workspace_name")
                return self._json(200, out)
            if self.path == "/api/workspaces/delete":
                name = str(req.get("name") or "")
                if name == STATE.get("workspace_name"):
                    return self._json(409, {"error": "Studio is running in this workspace; switch first"})
                ns_workspace.delete(name, path=path)
                return self._json(200, workspaces_state())
            if self.path.startswith("/api/settings/"):
                return self._settings_post(req)
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        return self._json(404, {"error": "not found"})

    def _settings_post(self, req):
        import tool_rules as tr
        if self.path == "/api/settings/subjects":
            vals = req.get("subject_models") or {}
            from subject_classifier import SUBJECTS
            ids = {m["id"] for m in scan_models()}
            bad = [f"{k}: {v}" for k, v in vals.items() if k not in SUBJECTS or (v and v not in ids)]
            if bad:
                return self._json(400, {"error": "unknown subject or model: " + ", ".join(bad)})
            clean = {k: v for k, v in vals.items() if v}
            save_studio_config({"subject_models": clean})
            STATE["subject_models"] = clean
            return self._json(200, {"subjects": subject_defaults()})
        if self.path == "/api/settings/sharing":
            cur = sharing()
            for k in ("enabled", "auto_listed", "auto_unlisted"):
                if k in req:
                    cur[k] = bool(req[k])
            if "endpoint" in req:
                ep = str(req["endpoint"]).strip().rstrip("/")
                if ep and not (re.match(r"https?://[\w.\-\[\]:]+(/[\w./-]*)?$", ep)
                               or re.fullmatch(r"github:[\w.-]+/[\w.-]+", ep)):
                    return self._json(400, {"error": "the scoreboard is a URL like https://board.example, "
                                                     "or a GitHub repository as github:owner/repo"})
                cur["endpoint"] = ep
            if req.get("github_token") is not None:
                import score_cards
                tok = str(req["github_token"]).strip()
                tp = score_cards.HOME / "github.token"
                if tok:
                    if not re.fullmatch(r"[\w-]{20,255}", tok):
                        return self._json(400, {"error": "that does not look like a GitHub token"})
                    sec.write_secret_file(tp, tok)
                elif tp.exists():
                    tp.unlink()
            if "public_models" in req:
                ids = {m["id"] for m in scan_models()}
                cur["public_models"] = sorted(str(x) for x in req["public_models"] if x in ids)
            save_studio_config({"sharing": cur})
            return self._json(200, {"settings": cur})
        if self.path in ("/api/settings/sharing/check", "/api/settings/sharing/preview",
                         "/api/settings/sharing/submit"):
            m = next((x for x in scan_models() if x["path"] == req.get("path")), None)
            if m is None:
                return self._json(404, {"error": "no such model"})
            if self.path.endswith("/check"):
                return self._json(200, sharing_listed(m["path"], refresh=True))
            known = {r["file"]: r for r in sharing_results()}
            res = known.get(str(req.get("results", "")))
            if res is None:
                return self._json(400, {"error": "pick a TestQA results file from the list"})
            card, log = sharing_card(res["file"], str(req.get("label") or ""), m, res.get("cache"))
            if self.path.endswith("/preview"):
                return self._json(200, {"card": card, "log": {"items": log["items"][:5],
                                                              "total_items": len(log["items"])}})
            sh = sharing()
            if not (sh["enabled"] and sh["endpoint"]):
                return self._json(400, {"error": "turn sharing on and give a scoreboard address first"})
            import score_cards
            try:
                return self._json(200, score_cards.submit(sh["endpoint"], card, log))
            except Exception as e:
                return self._json(502, {"error": f"the scoreboard did not accept it: {type(e).__name__}: {e}"})
        if self.path == "/api/settings/theme":
            name = str(req.get("theme", ""))
            th = studio_themes()
            if name not in th:
                return self._json(400, {"error": f"no theme {name!r}"})
            save_studio_config({"theme": name})
            STATE["studio_theme"], STATE["viz_theme"] = name, th[name].get("viz") or "dark"
            return self._json(200, {"default": name})
        if self.path == "/api/settings/permissions":
            # replaced whole: profiles, model bindings, job profiles, base (scripts/permissions.py)
            permissions().save(req.get("policy") or {})
            return self._json(200, permissions_state())
        if self.path == "/api/settings/permissions/check":
            import permissions as P
            ctx = P.context(workdir=req.get("workdir"), repo=req.get("repo"))
            return self._json(200, permissions().check(str(req.get("action") or ""), str(req.get("target") or ""),
                                                      str(req.get("model") or ""), str(req.get("job") or ""),
                                                      str(req.get("profile") or ""), ctx))
        if self.path == "/api/settings/widgets":
            # the owner's widget settings, served with every 3D theme (viz/widgets.py)
            import widgets
            try:
                clean = widgets.validate(req.get("widgets") or {}, plugins().widget_assets()["styles"])
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            save_studio_config({"widgets": clean})
            STATE["widgets"] = clean
            return self._json(200, {"values": widgets.merged(None, clean)})
        if self.path == "/api/settings/tool-rules":
            rules = tool_rules().replace(req.get("rules") or {})
            return self._json(200, {"tool_rules": rules})
        if self.path == "/api/settings/tool-rules/test":
            try:
                args = req.get("args") or {}
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                ann = req.get("annotations") or {}
            except ValueError as e:
                return self._json(400, {"error": f"arguments: {e}"})
            d = tool_rules().decide(str(req.get("server") or ""), str(req.get("tool") or ""), args, ann,
                                    record=False)
            return self._json(200, d)
        return self._json(404, {"error": "not found"})

    # ------------------------------------------------ requirements (scripts/ns_requirements.py)

    def _requirements(self):
        if not self._owner_only():
            return
        import ns_requirements as nrq
        post = self.command == "POST"
        req = self._read() if post else {}
        configured = _configured_bins()
        try:
            if not post and self.path == "/api/requirements":
                return self._json(200, nrq.check(configured=configured))
            if not post and self.path == "/api/requirements/snapshots":
                return self._json(200, {"snapshots": nrq.list_snapshots(), "dir": str(nrq.SNAP_DIR)})
            if not post and self.path == "/api/requirements/freeze":
                body = nrq.freeze().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="neuronscope-env.txt"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if post and self.path == "/api/requirements/snapshot":
                path = nrq.save_snapshot(nrq.snapshot(configured))
                return self._json(200, {"id": path.stem})
            if post and self.path == "/api/requirements/diff":
                snaps = [x["id"] for x in nrq.list_snapshots()]
                a = req.get("a") or (snaps[1] if len(snaps) > 1 else None)
                b = req.get("b") or (snaps[0] if snaps else None)
                if not a or not b:
                    return self._json(400, {"error": "take two snapshots first (one now, one after a change)"})
                return self._json(200, nrq.diff(nrq.load_snapshot(str(a)), nrq.load_snapshot(str(b))))
        except FileNotFoundError:
            return self._json(404, {"error": "no such snapshot"})
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        return self._json(404, {"error": "not found"})

    PAGES = {"/", "/setup", "/settings", "/lab", "/review", "/projects", "/workers", "/jobs", "/connect", "/remote",
             "/link", "/system", "/features", "/graphs", "/addons", "/work"}

    def do_GET(self):
        self._route = None
        base = self.path.split("?", 1)[0].split("#", 1)[0]
        if base != "/" and base.rstrip("/") in self.PAGES:       # /workers/ and /workers?x are /workers
            self.path = base.rstrip("/")
        elif base == "/" and self.path != "/":
            self.path = "/"
        if self._addon_removed():
            return
        if self.path.startswith(("/api/remote/next", "/api/remote/file/")):
            return self._remote_public("GET")
        if self.path.split("?")[0] == "/pair":
            body = PAIR_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if not self._authed():
            return self._deny()
        if self.path == "/link":
            body = LINK_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/devices":
            if not self._owner_only():
                return
            return self._json(200, {"devices": STATE["devices"].list(),
                                    "policy": STATE["devices"].policy.describe(),
                                    "links": [{"name": x["name"], "url": x["url"],
                                               "pinned": bool(x.get("fingerprint")),
                                               "expires": x.get("expires"),
                                               "error": (LINKS["cache"].get(x["name"]) or (0, [], None))[2],
                                               "models": len((LINKS["cache"].get(x["name"]) or (0, []))[1])}
                                              for x in load_links() if x.get("kind") != "agent"]})
        if self.path == "/api/links/models":
            return self._json(200, link_models())
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        m = re.fullmatch(r"/viz/([A-Za-z0-9_-]+)/(|api/meta|api/theme|api/trace)", self.path)
        if m:
            pl = trace_payload(m.group(1))
            if pl is None:
                return self._json(404, {"error": "no such trace"})
            import bloom
            kind = m.group(2)
            body, ctype = ((bloom.PAGE.encode(), "text/html; charset=utf-8") if kind == "" else
                           (pl["blob"], "application/octet-stream") if kind == "api/trace" else
                           (json.dumps(pl["meta"] if kind == "api/meta" else self._viz_theme(pl["theme"])).encode(),
                            "application/json"))
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/traces":
            return self._json(200, {"traces": list_traces()})
        if self.path in ("/lab", "/api/lab"):
            if not self._owner_only():
                return
            if self.path == "/lab":
                import lab_page
                body = lab_page.PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            hub = lab_hub()
            b = addon_builtins()
            return self._json(200, {"services": {k: v for k, v in hub.service_state().items() if not b.service_hidden(k)},
                                    "traces": hub.find_traces(extra=[STATE.get("traces_dir")])})
        if self.path in ("/setup", "/api/setup", "/api/hardware"):
            if not self._owner_only():
                return
            if self.path == "/setup":
                import setup_page
                body = setup_page.PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/setup":
                return self._json(200, setup_state())
            import accelerators
            path = STATE.get("hardware_path") or str(accelerators.DEFAULT_CONFIG)
            return self._json(200, {"path": path, "config": accelerators.load_config(path),
                                    "devices": POOL.devices(refresh=True)})
        if self.path == "/api/themes":
            import widgets
            return self._json(200, {"themes": studio_themes(), "default": STATE.get("studio_theme") or "midnight",
                                    "viz": sorted(__import__("bloom").THEMES),
                                    "widgets": {"spec": widgets.spec(plugins().widget_assets()["styles"]),
                                                "values": widgets.merged(None, STATE.get("widgets"))}})
        if self.path.startswith("/api/engine/options"):
            import engine_options as eo
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            engine = (q.get("engine") or ["llama"])[0]
            if engine not in ("llama", "vllm"):
                return self._json(400, {"error": "engine is llama or vllm"})
            sup = eo.probe(STATE["server_bin"], "llama") if engine == "llama" and STATE.get("server_bin") else None
            sc = eo.schema(engine, sup)
            sc["binary"] = STATE.get("server_bin") if engine == "llama" else STATE.get("vllm_bin")
            sc["vllm_installed"] = bool(shutil.which((STATE.get("vllm_bin") or "vllm").split()[0]))
            return self._json(200, sc)
        if self.path == "/api/engine/devices":
            # what llama-server --list-devices names, for the device fields' dropdowns (cached a minute)
            c = STATE.setdefault("engine_devices", {"t": 0, "bin": None, "list": []})
            if STATE.get("server_bin") and (c["bin"] != STATE["server_bin"] or time.time() - c["t"] > 60):
                import draft_pick
                c.update(t=time.time(), bin=STATE["server_bin"], list=draft_pick.list_devices(STATE["server_bin"]))
            return self._json(200, {"devices": c["list"]})
        if self.path.startswith("/api/models/draft"):
            # compatible draft models for speculative decoding, and a device for the draft
            import draft_pick
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            ms = scan_models()
            m = find_model((q.get("model") or [""])[0], ms)
            if m is None:
                return self._json(404, {"error": "no such model"})
            try:
                cands = draft_pick.candidates(m["path"], [x["path"] for x in ms])
            except (ValueError, ImportError) as e:
                return self._json(400, {"error": str(e)})
            out = {"model": m["id"], "candidates": [{**c, "id": next((x["id"] for x in ms if x["path"] == c["path"]),
                                                                     c["name"])} for c in cands[:6]]}
            if cands and STATE.get("server_bin"):
                dev = str(load_settings().get(_key(m["path"]), {}).get("device") or "")
                out["placement"] = draft_pick.place(draft_pick.list_devices(STATE["server_bin"]),
                                                    [x for x in dev.split(",") if x], cands[0]["bytes"],
                                                    os.path.getsize(m["path"]))
            return self._json(200, out)
        if self.path in ("/remote", "/remote/download") or self.path.startswith("/api/remote"):
            return self._remote_get()
        if self.path in ("/connect", "/api/connect"):
            if not self._owner_only():
                return
            if self.path == "/connect":
                import connect_page
                body = connect_page.PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            import ns_connect
            base = self._self_url(public=True)
            out = ns_connect.snippets(base, "token" if STATE["token"] else None)
            out["token_required"] = bool(STATE["token"])
            out["models"] = ["auto"] + [m["id"] for m in scan_models()]
            return self._json(200, out)
        if self.path == "/mcp":
            # Streamable HTTP lets a server offer an SSE stream here; this one answers POSTs only.
            self.send_response(405)
            self.send_header("Allow", "POST")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/api/doctor":
            if not self._owner_only():
                return
            return self._json(200, run_doctor())
        if self.path in ("/review", "/api/review") or self.path.startswith("/review/view/"):
            return self._review_get()
        if self.path.startswith("/api/requirements"):
            return self._requirements()
        if self.path in ("/settings", "/api/workspaces", "/api/settings", "/api/settings/sharing", "/api/settings/permissions"):
            return self._settings()
        if self.path.startswith("/api/fs/"):
            # The folder browser: the owner's own files on Studio's computer.
            if not self._owner_only():
                return
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            if self.path.startswith("/api/fs/places"):
                return self._json(200, {"places": fs_places(), "sep": os.sep, "home": os.path.expanduser("~")})
            if self.path.startswith("/api/fs/list"):
                return self._json(200, fs_list((q.get("path") or [""])[0], (q.get("want") or ["any"])[0],
                                               [x for x in (q.get("ext") or [""])[0].split(",") if x],
                                               (q.get("hidden") or ["0"])[0] == "1"))
        if self.path == "/api/plugins" or self.path.startswith("/plugins/"):
            return self._plugins_get()
        if self.path == "/api/help/index":
            import help_page
            docs = sorted(p for p in help_page.DOCS.glob("*.md") if help_page.NAME.match(p.stem))
            return self._json(200, {"pages": [{"name": p.stem, "title": p.stem.replace("_", " ").title()} for p in docs]})
        if self.path.startswith("/help"):
            import help_page
            name = self.path.split("?")[0].split("#")[0][len("/help"):].strip("/") or "GETTING_STARTED"
            body = help_page.page(name)
            if body is None:
                return self._json(404, {"error": f"no help page {name!r}"})
            body = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/projects", "/workers", "/api/workers", "/api/workers/overview") \
                or self.path.startswith("/api/projects"):
            return self._projects_get()
        if self.path == "/features" or self.path.startswith("/api/features/"):
            if not self._owner_only():
                return
            return self._features_get()
        if self.path == "/graphs" or self.path.startswith("/api/graphs"):
            if not self._owner_only():
                return
            return self._graphs(None)
        if self.path == "/work" or self.path.startswith("/api/work"):
            if not self._owner_only():
                return
            return self._work()
        if self.path == "/api/addons/widgets":
            # any signed-in viewer: which widget scripts the pages and the Godot client should load
            return self._json(200, plugins().widget_assets())
        if self.path == "/addons" or self.path.startswith("/api/addons"):
            if not self._owner_only():
                return
            return self._addons(None)
        if self.path.startswith("/api/jobs") or self.path == "/jobs":
            if self._role != "owner":
                return self._json(403, {"error": "jobs need the owner"})
            if self.path == "/jobs":
                body = ns_jobs.PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            return self._jobs_get()
        if self.path == "/api/gpus":
            import cuda_info
            return self._json(200, cuda_info.advise(nvidia_gpus()))
        if self.path == "/api/connectors":
            if not self._owner_only():
                return
            mcp_hub()
            return self._json(200, connectors())
        if self.path == "/api/mcp":
            hub = mcp_hub()
            return self._json(200, {"servers": hub.status() if hub else [], "config": STATE.get("mcp_config"),
                                    "tools": len(hub.openai_tools()) if hub else 0})
        if self.path == "/api/rag":
            return self._json(200, {"collections": rag_store().list(),
                                    "embedder": bool(STATE.get("rag_embed") or STATE.get("rag_embed_gguf"))})
        m = re.fullmatch(r"/api/rag/docs\?c=(.+)", self.path)
        if m:
            try:
                return self._json(200, rag_store().get(urllib.parse.unquote(m.group(1))).docs())
            except (ValueError, FileNotFoundError) as e:
                return self._json(404, {"error": str(e)})
        if self.path == "/v1/models":
            return self._json(200, openai_models())
        if self.path == "/api/chats":
            return self._json(200, list_chats())
        if self.path.startswith("/api/chats/"):
            try:
                with open(_chat_path(self.path.rsplit("/", 1)[-1])) as f:
                    return self._json(200, json.load(f))
            except (ValueError, FileNotFoundError):
                return self._json(404, {"error": "no such chat"})
        m = re.fullmatch(r"/api/quant/choice\?family=(.+)", self.path)
        if m:
            # What quant_select.py measured for this model family on this hardware (Jobs -> Choose a quantization).
            fam = urllib.parse.unquote(m.group(1))
            if "hw" not in STATE:
                STATE["hw"] = ns_qsel.hardware()
            try:
                allc = json.loads(Path(STATE.get("quant_choices") or ns_qsel.CHOICES).read_text())
            except (OSError, ValueError):
                allc = {}
            here = (allc.get(fam) or {}).get(STATE["hw"]["id"])
            return self._json(200, {"family": fam, "here": here, "elsewhere": len(allc.get(fam) or {}) - bool(here)})
        if self.path == "/api/models":
            ms = scan_models()
            st = load_settings()
            for m in ms:
                m["settings"] = {**DEFAULTS, **st.get(_key(m["path"]), {})}
                m["fit"] = fit_estimate(m["size"])
                m["stats"] = model_summary(m, st)
                m["family"], m["quant"] = ns_qsel.family(m["path"]), ns_qsel.quant_of(m["path"])
            return self._json(200, ms)
        if self.path == "/api/stats":
            return self._json(200, {"min_graded": STATE["min_graded"], "min_subject": STATE["min_subject"],
                                    "hallucination_cost": STATE["halluc_cost"],
                                    "window": STATE["stats"].window, "scoring": bool(STATE["cett"]),
                                    "models": {m["id"]: model_summary(m) for m in scan_models()}})
        if self.path == "/api/hosts":
            hosts, current = host_profiles()
            return self._json(200, {"hosts": hosts, "current": current})
        if self.path == "/api/configs":
            models = {m["path"]: m for m in scan_models()}
            out = []
            for name, cfg in load_configs().items():
                c = {**CONFIG_DEFAULTS, **cfg, "name": name}
                out.append({**c, "audit": audit_config(c, models.get(c["path"])),
                            "missing": c["path"] not in models})
            return self._json(200, out)
        if self.path == "/api/presets":
            return self._json(200, load_presets())
        if self.path == "/api/downloads":
            with JOBS_LOCK:
                return self._json(200, [
                    {k: v for k, v in j.items() if k not in ("cancel", "base")}
                    for j in JOBS.values()])
        if self.path.startswith("/api/search"):
            q = urllib.parse.parse_qs(self.path.split("?", 1)[-1]).get("q", [""])[0]
            if not q:
                return self._json(400, {"error": "q required"})
            try:
                return self._json(200, hub_search(q))
            except Exception as e:
                return self._json(502, {"error": f"hub unreachable: {e}"})
        if self.path.startswith("/api/files"):
            repo = urllib.parse.parse_qs(
                self.path.split("?", 1)[-1]).get("repo", [""])[0]
            if not repo:
                return self._json(400, {"error": "repo required"})
            try:
                return self._json(200, hub_files(repo))
            except Exception as e:
                return self._json(502, {"error": f"hub unreachable: {e}"})
        if self.path == "/api/status":
            return self._json(200, {
                "started": STARTED["t"], "stale": code_stale(),
                "computer": socket.gethostname(),
                "roles": "this Studio runs the tools and models; whoever calls its API or MCP is the controller",
                "running": server_running(),
                "model": PROC["model"]["name"] if PROC["model"] else None,
                "path": PROC["model"]["path"] if PROC["model"] else None,
                "port": PROC["port"],
                "uptime": int(time.time() - PROC["started"]) if PROC["started"] else 0,
                "lora": PROC["lora"], "lora_scale": PROC["lora_scale"],
                "arch": PROC["model"].get("arch") if PROC["model"] else None,
                "n_experts": PROC["model"].get("n_experts") if PROC["model"] else None,
                "id": PROC["model"].get("id") if PROC["model"] else None,
                "vision": bool(PROC["model"] and PROC["model"].get("mmproj")
                               and "--mmproj" in (PROC["args"] or [])),
                "idle_ttl": STATE["idle_ttl"], "jit": STATE["jit"],
                "auto_ready": any(model_summary(m)["eligible"] for m in scan_models()),
                "idle_for": int(time.time() - ACTIVITY["last"]),
                # What this caller may do: "owner" (everything) or "device" (models, chat, checks).
                "role": self._role, "workspace": STATE.get("workspace_name"),
            })
        if self.path == "/system":
            import system_page
            return self._html(system_page.PAGE)
        if self.path == "/api/services":
            if not self._owner_only():
                return
            return self._json(200, {"services": services(), "stale": code_stale()})
        if self.path.startswith("/api/system/monitor"):
            if not self._owner_only():
                return
            sysmon_start()
            return self._json(200, sysmon_view())
        if self.path == "/api/automation":
            if not self._owner_only():
                return
            sysmon_start()
            cfg = automation_config()
            ha = cfg["ha"]
            try:
                hosts = [h["name"] for h in remote().hosts()]
            except Exception:
                hosts = []
            return self._json(200, {"rules": cfg["rules"], "manual": cfg["manual"],
                                    "ha": {"url": ha.get("url", ""), "token_file": ha.get("token_file", ""),
                                           "has_token": bool(ha.get("token") or ha.get("token_file"))},
                                    "results": AUTO["results"], "applied": AUTO["applied"], "busy": BUSY,
                                    "log": list(AUTO["log"])[:60], "targets": ["local"] + ["host:" + h for h in hosts],
                                    "computer": socket.gethostname(), "periods": ns_auto.PERIODS,
                                    "actions": ns_auto.ACTIONS})
        if self.path.startswith("/api/profiles"):
            if not self._owner_only():
                return
            scope = urllib.parse.parse_qs(self.path.split("?", 1)[-1]).get("scope", [""])[0]
            if not PROFILE_SCOPE.match(scope):
                return self._json(400, {"error": "scope required"})
            return self._json(200, profiles_get(scope))
        if not self.path.startswith(("/api/", "/v1/")) and "text/html" in (self.headers.get("Accept") or ""):
            return self._not_found_page()
        self._json(404, {"error": "not found"})

    def _not_found_page(self):
        """A page this Studio does not have: say why, most often an update that needs a restart."""
        import html as _h
        from ui_links import js as _ui_js
        stale = code_stale()
        page = f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Not in this Studio</title><style>body{{margin:0;font:15px/1.6 ui-sans-serif,system-ui;background:#151514;color:#ececea}}
main{{max-width:620px;margin:12vh auto;padding:0 16px}}a{{color:#6f93ff}}button{{font:600 14px ui-sans-serif,system-ui;padding:.45rem 1rem;
border-radius:6px;border:1px solid #6f93ff;background:#6f93ff;color:#0b0b0a;cursor:pointer}}code{{background:#262624;padding:0 .3rem;border-radius:4px}}</style>
<main><h1>This page is not in the running Studio</h1>
<p><code>{_h.escape(self.path[:200])}</code> {"exists in OmniCortex's files, which changed after this Studio started. Restart it to load the new version." if stale else "is not a Studio page. If you updated OmniCortex recently, restarting Studio loads the new pages."}</p>
<p><button onclick="nsRestart()">Restart Studio</button> &nbsp; <a href="/">Back to Studio</a> · <a href="/system">System</a></p></main>
<script>{_ui_js()}</script>"""
        return self._html(page, 404)

    def do_POST(self):
        self._route = None
        if self._addon_removed():
            return
        if self.path == "/api/login":
            req = self._read()
            if STATE["token"] and not THROTTLE.blocked(self.client_address[0]) and hmac.compare_digest(
                    str(req.get("token", "")), STATE["token"]):
                body = json.dumps({"ok": True}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Set-Cookie",
                                 f"ns_token={STATE['token']}; Path=/; "
                                 "HttpOnly; SameSite=Strict; Max-Age=604800"
                                 + ("; Secure" if STATE["tls"] else ""))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                THROTTLE.fail(self.client_address[0])
                self._json(401, {"error": "bad token"})
            return
        if self.path in ("/api/remote/claim", "/api/remote/reply", "/api/remote/progress"):
            return self._remote_public("POST")
        if self.path == "/api/pair/claim":
            ip = self.client_address[0]
            if THROTTLE.blocked(ip):
                return self._json(429, {"error": "too many attempts; wait a few minutes"})
            req = self._read()
            r = STATE["devices"].claim(str(req.get("code", "")), str(req.get("name", "device")))
            if r is None:
                THROTTLE.fail(ip)
                return self._json(403, {"error": "pairing code is wrong, used or expired"})
            r["fingerprint"] = STATE.get("fingerprint") or ""
            if req.get("cookie"):
                body = json.dumps(r).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                # The cookie lives as long as the access: a year for persistent pairing (the
                # host can still revoke it), exactly the granted time for temporary access.
                age = 31536000 if r["expires"] is None else max(1, int(r["expires"] - time.time()))
                self.send_header("Set-Cookie", f"ns_token={r['token']}; Path=/; HttpOnly; SameSite=Strict; "
                                 f"Max-Age={age}" + ("; Secure" if STATE["tls"] else ""))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            return self._json(200, r)
        if not self._authed():
            return self._deny()
        if self._role == "device" and self.path not in DEVICE_POSTS:
            # A paired device may use models, not administer this machine: no jobs (they
            # run code), no load flags or downloads, no MCP reloads, no pairing.
            return self._json(403, {"error": "paired devices can chat and use /v1; this needs the owner"})
        try:
            if self.path.startswith("/api/remote/"):
                return self._remote_post()
            if self.path.startswith("/api/plugins/"):
                return self._plugins_post()
            if self.path in ("/api/pair/start", "/api/devices/revoke", "/api/devices/update", "/api/links/add",
                             "/api/links/remove"):
                return self._pairing_post()
            if self.path in ("/v1/chat/completions", "/v1/completions", "/v1/embeddings"):
                return self._v1(self.path)

            if self.path == "/api/chats":
                return self._json(200, save_chat(self._read(MAX_BODY)))

            if self.path in ("/api/jobs", "/api/jobs/cancel"):
                return self._jobs_post()

            if self.path.startswith(("/api/projects", "/api/workers/")):
                return self._projects_post()

            if self.path == "/mcp":
                return self._mcp_post()

            if self.path.startswith("/api/review/"):
                return self._review_post()

            if self.path.startswith("/api/requirements/"):
                return self._requirements()

            if self.path.startswith("/api/workspaces/") or self.path.startswith("/api/settings/"):
                return self._settings()

            if self.path in ("/api/lab/start", "/api/lab/stop"):
                if not self._owner_only():
                    return
                req = self._read()
                hub = lab_hub()
                name = str(req.get("name") or "")
                if name not in hub.SERVICES or (self.path == "/api/lab/start" and addon_builtins().service_hidden(name)):
                    return self._json(404, {"error": f"no lab service {name!r} (Add-ons shows what is removed)"})
                if self.path == "/api/lab/stop":
                    hub.stop(name)
                    return self._json(200, {"ok": True})
                ok, why, port = hub.acquire(name, "studio", req.get("cfg") or {})
                return self._json(200 if ok else 400, {"ok": ok, "port": port, "message": why} if ok
                                  else {"error": why})

            if self.path.startswith("/api/features/"):
                if not self._owner_only():
                    return
                return self._features_post(self._read())

            if self.path.startswith("/api/graphs"):
                if not self._owner_only():
                    return
                return self._graphs(self._read())

            if self.path.startswith("/api/addons"):
                if not self._owner_only():
                    return
                if self.path == "/api/addons/upload":
                    return self._addon_upload()
                return self._addons(self._read())

            if self.path.startswith("/api/automation/"):
                if not self._owner_only():
                    return
                req = self._read()
                try:
                    cfg = automation_config()
                    if self.path == "/api/automation/rules":
                        cfg["rules"] = [ns_auto.clean_rule(r) for r in req.get("rules") or []]
                        seen = set()
                        for r in cfg["rules"]:                  # every rule its own id, even if a client repeated one
                            if r["id"] in seen:
                                r["id"] = ns_auto.clean_rule({**r, "id": None})["id"]
                            seen.add(r["id"])
                        automation_save(cfg)
                        sysmon_start()
                        return self._json(200, {"rules": cfg["rules"], "results": automation_tick()})
                    if self.path == "/api/automation/ha":
                        url = str(req.get("url", "")).strip().rstrip("/")
                        if url and not re.match(r"^https?://[^\s/]+", url):
                            raise ValueError("Home Assistant address: http(s)://host:8123")
                        ha = dict(cfg["ha"])
                        ha["url"] = url
                        if req.get("token"):
                            ha["token"], ha["token_file"] = str(req["token"]).strip(), ""
                        if req.get("token_file") is not None and str(req.get("token_file")).strip():
                            ha["token_file"], ha["token"] = str(req["token_file"]).strip(), ""
                        cfg["ha"] = ha
                        automation_save(cfg)
                        out = {"ok": True}
                        if req.get("test_entity"):
                            h = _ha()
                            if h is None:
                                raise ValueError("set the address and a token first")
                            out["reading"] = h.state(str(req["test_entity"]))
                        return self._json(200, out)
                    if self.path == "/api/automation/action":
                        return self._json(200, automation_action(str(req.get("target", "local")),
                                                                 str(req.get("action", "")),
                                                                 float(req.get("minutes") or 0),
                                                                 str(req.get("reason") or "")))
                    if self.path == "/api/automation/manual":
                        cfg["manual"][str(req.get("rule", ""))] = bool(req.get("on"))
                        automation_save(cfg)
                        return self._json(200, {"manual": cfg["manual"], "results": automation_tick()})
                except (ValueError, KeyError, OSError) as e:
                    return self._json(400, {"error": str(e)[:300]})
                except Exception as e:
                    return self._json(502, {"error": f"{type(e).__name__}: {e}"[:300]})
                return self._json(404, {"error": "not found"})
            if self.path == "/api/profiles":
                if not self._owner_only():
                    return
                req = self._read(400_000)
                try:
                    return self._json(200, profiles_change(str(req.get("scope", "")), str(req.get("action", "")),
                                                           str(req.get("name", "")), req.get("values"),
                                                           str(req.get("new_name", ""))))
                except ValueError as e:
                    return self._json(400, {"error": str(e)})
            if self.path == "/api/services":
                if not self._owner_only():
                    return
                req = self._read()
                try:
                    note = service_action(str(req.get("id", "")), str(req.get("action", "")))
                except Exception as e:          # includes a host's refusal (remote agents)
                    return self._json(400, {"error": str(e)[:400]})
                return self._json(200, {"ok": True, "note": note, "services": services()})
            if self.path in ("/api/setup", "/api/hardware", "/api/restart"):
                if not self._owner_only():
                    return
                if self.path == "/api/restart":
                    threading.Thread(target=restart_studio, daemon=True).start()
                    return self._json(200, {"ok": True, "note": "Studio restarts in a moment"})
                req = self._read()
                if self.path == "/api/setup":
                    clean, errs = clean_setup(req.get("values") or {})
                    if errs:
                        return self._json(400, {"error": "; ".join(f"{k}: {v}" for k, v in errs.items()), "fields": errs})
                    import ns_workspace
                    ws_keys = {"models_dir": "models_dirs", "server": "server", "cett": "cett",
                               "download_dir": "download_dir"}
                    to_ws = {ws_keys[k]: v for k, v in clean.items() if k in ws_keys}
                    if to_ws:
                        ns_workspace.save(STATE.get("workspace_name") or ns_workspace.active_name(_ws_path()),
                                          to_ws, path=_ws_path(), check_exists=False)
                    rest = {k: v for k, v in clean.items() if k not in ws_keys}
                    if rest:
                        save_studio_config(rest)
                    later = apply_setup(clean)
                    return self._json(200, {"saved": sorted(clean), "restart_needed": later, **setup_state()})
                cfg = req.get("config")
                if not isinstance(cfg, dict) or set(cfg) - {"prefer", "servers", "devices", "manual"}:
                    return self._json(400, {"error": "hardware config must be an object with prefer, servers, "
                                                     "devices and manual"})
                for b, pth in (cfg.get("servers") or {}).items():
                    if pth and not os.path.isfile(os.path.expanduser(pth)):
                        return self._json(400, {"error": f"servers.{b}: {pth} does not exist"})
                for dev, c in (cfg.get("devices") or {}).items():
                    if not isinstance(c, dict):
                        return self._json(400, {"error": f"devices.{dev} must be an object"})
                    if c.get("server") and not os.path.isfile(os.path.expanduser(c["server"])):
                        return self._json(400, {"error": f"{dev}: llama-server {c['server']} does not exist"})
                import accelerators
                path = os.path.expanduser(STATE.get("hardware_path") or str(accelerators.DEFAULT_CONFIG))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(cfg, f, indent=2)
                os.replace(tmp, path)
                return self._json(200, {"ok": True, "devices": POOL.devices(refresh=True)})

            if self.path == "/api/connect/token":
                # A separate, revocable token for an app (Cline, Claude Desktop): a paired
                # device with persistent access, minted and claimed in one step.
                if not self._owner_only():
                    return
                reg = STATE.get("devices")
                if reg is None:
                    return self._json(400, {"error": "device pairing is not available"})
                req = self._read()
                try:
                    code, _, _ = reg.new_code(True, None)
                except ValueError:          # this host only grants temporary access
                    code, _, _ = reg.new_code(False, None)
                r = reg.claim(code, str(req.get("name") or "app")[:60])
                return self._json(200, {"token": r["token"], "name": r.get("name"),
                                        "note": "a paired device: chat, /v1, MCP read tools and reply checks; "
                                                "revoke it under Link"})

            if self.path == "/api/route":
                req = self._read()
                text = str(req.get("text") or "")
                if not text.strip():
                    return self._json(400, {"error": "text required"})
                from subject_classifier import default_classifier
                a = default_classifier().analyze(text)
                models = scan_models()
                _, pick = auto_pick({"messages": [{"role": "user", "content": text}]}, models)
                return self._json(200, {"subjects": a["labels"] or ["unknown"], "why": a["why"],
                                        "proba": dict(list(a["proba"].items())[:5]),
                                        "auto_would_pick": pick.get("model"), "reason": pick.get("reason"),
                                        "candidates": pick.get("candidates", [])[:5]})

            if self.path == "/api/trace":
                req = self._read(MAX_BODY)
                ref = req.get("model")
                text = str(req.get("text") or "")
                if not text.strip():
                    return self._json(400, {"error": "nothing to check"})
                if ref and LINK_SEP in ref and ref.partition(LINK_SEP)[0] not in [h["name"] for h in remote().hosts()] \
                        and link_target(ref) is not None:
                    return self._json(400, {"error": "checks of a linked Studio's models run in that Studio"})
                if ref and LINK_SEP in ref and link_target(ref) is not None:
                    try:
                        return self._json(200, trace_remote(ref, req.get("messages") or [], text,
                                                            float(req.get("threshold", 0.5))))
                    except ValueError as e:
                        return self._json(400, {"error": str(e)})
                    except Exception as e:
                        return self._json(502, {"error": f"{type(e).__name__}: {e}"})
                model = find_model(ref) if ref and ref != "auto" else PROC["model"]
                if model is None:
                    return self._json(404, {"error": "no such model"})
                try:
                    return self._json(200, trace_reply(model, req.get("messages") or [], text,
                                                       float(req.get("threshold", 0.5))))
                except ValueError as e:
                    links = []
                    if "classifier" in str(e):
                        links = [{"href": f"/#model={urllib.parse.quote(model['id'])}&field=classifier",
                                  "label": "set this model's classifier"},
                                 {"href": "/jobs#kind=classify", "label": "make one (H-Neuron pipeline, stage 5)"}]
                    elif "cett" in str(e):
                        links = [{"href": "/setup#build", "label": "build llama.cpp with cett-dump"},
                                 {"href": "/settings#workspaces", "label": "use an existing build"}]
                    return self._json(400, {"error": str(e), "links": links})

            if self.path.startswith("/api/rag/"):
                return self._rag_post()

            if self.path == "/api/tools/approve":
                req = self._read()
                key, remember = str(req.get("key", "")), req.get("remember")
                rule = None
                if remember:
                    # A standing rule changes what every later chat may do: the owner's call.
                    if self._role != "owner":
                        return self._json(403, {"error": "only the owner can make a standing rule"})
                    import tool_rules as tr
                    with APPROVALS_LOCK:
                        call = PENDING_CALLS.get(key)
                    if call is None:
                        return self._json(404, {"error": "that call is no longer waiting"})
                    try:
                        if remember == "always":
                            # the quick "Always allow": this exact call stops asking, in both places
                            import permissions as P
                            asks = [(c["action"], c["target"]) for c in call.get("checks") or [] if c["effect"] == "ask"]
                            if asks:
                                rule = {"permissions": P.remember(permissions(), call.get("model", ""), asks)}
                            if call.get("rules_action") == "ask":
                                rule = {**(rule or {}), "tool_rule": tool_rules().add(
                                    tr.rule_from_call("args", call["server"], call["tool"], call["args"]))}
                        else:
                            rule = tool_rules().add(tr.rule_from_call(str(remember), call["server"], call["tool"],
                                                                      call["args"]))
                    except ValueError as e:
                        return self._json(400, {"error": str(e)})
                allow = bool(req.get("allow")) and remember != "deny"
                with APPROVALS_LOCK:
                    a = APPROVALS.get(key)
                    if a:
                        a["allow"] = allow
                        a["event"].set()
                return self._json(200 if a else 404, {"ok": bool(a), "rule": rule})

            if self.path == "/api/mcp/reload":
                hub = mcp_hub()
                if hub is None:
                    return self._json(400, {"error": "no MCP config (--mcp-config)"})
                hub.connect()
                return self._json(200, {"servers": hub.status(), "config": str(hub.config_path)})

            if self.path in ("/api/connectors/save", "/api/connectors/remove", "/api/connectors/toggle"):
                if not self._owner_only():
                    return
                req, path = self._read(), connectors_path()
                try:
                    if self.path.endswith("/save"):
                        ns_mcp.save_server(path, req.get("name", ""), req.get("spec") or {}, req.get("old_name"))
                    elif self.path.endswith("/remove"):
                        ns_mcp.remove_server(path, str(req.get("name", "")))
                    else:
                        ns_mcp.set_disabled(path, str(req.get("name", "")), bool(req.get("disabled")))
                except (OSError, ValueError) as e:
                    return self._json(400, {"error": str(e)})
                connectors_changed()
                return self._json(200, connectors())

            if self.path == "/api/mcp/test":
                # The owner's own check: lists the tools again and calls one read-only tool that
                # needs no arguments, if the server marks one so.
                hub = mcp_hub()
                if hub is None:
                    return self._json(400, {"error": "no MCP servers configured"})
                return self._json(200, hub.test(str(self._read().get("server", ""))))

            if self.path == "/api/stats/ingest":
                req = self._read(8 * 1024 * 1024)
                target = find_model(str(req.get("model", "")))
                if target is None:
                    return self._json(404, {"error": f"no local model matches {req.get('model')!r}"})
                n = 0
                for r in req.get("records", [])[:20000]:
                    if not isinstance(r, dict):
                        continue
                    rec = {k: r[k] for k in ("subject", "task_kind", "verdict", "task", "source") if k in r}
                    if STATE["stats"].record(target["id"], "graded", size=target["size"], **rec):
                        n += 1
                return self._json(200, {"model": target["id"], "recorded": n})

            if self.path == "/api/chats/delete":
                try:
                    os.remove(_chat_path(self._read().get("id")))
                except FileNotFoundError:
                    pass
                return self._json(200, {"ok": True})

            if self.path == "/api/load":
                req = self._read()
                target = next((m for m in scan_models()
                               if m["path"] == req["path"]), None)
                if not target:
                    return self._json(404, {"error": "model not found"})
                settings = {**DEFAULTS, **req.get("settings", {})}
                allset = load_settings()
                allset[_key(target["path"])] = settings
                save_settings(allset)
                with LOCK:
                    ok, why = start_server(target, settings, req.get("port", 8080))
                return self._json(200 if ok else 500,
                                  {"ok": ok, "message": why, "notes": STATE.get("engine_notes") or [],
                                   "args": PROC.get("args") if ok else None})

            if self.path == "/api/unload":
                with LOCK:
                    stop_server()
                return self._json(200, {"ok": True})

            if self.path == "/api/lora":
                req = self._read()
                scale = float(req["scale"])
                proxy("/lora-adapters", [{"id": 0, "scale": scale}])
                PROC["lora_scale"] = scale
                return self._json(200, {"ok": True, "scale": scale})

            if self.path == "/api/config":
                req = self._read()
                if not req.get("name"):
                    return self._json(400, {"error": "name required"})
                save_config(req)
                models = {m["path"]: m for m in scan_models()}
                return self._json(200, {"ok": True,
                                        "audit": audit_config(
                                            {**CONFIG_DEFAULTS, **req},
                                            models.get(req.get("path")))})

            if self.path == "/api/config/delete":
                delete_config(self._read()["name"])
                return self._json(200, {"ok": True})

            if self.path == "/api/preset":
                req = self._read()
                save_preset(req["name"], req["config"])
                return self._json(200, {"ok": True})

            if self.path == "/api/download":
                req = self._read()
                root = STATE["download_dir"] or STATE["models_dirs"][0]
                if not os.path.isdir(root):
                    return self._json(400,
                                      {"error": f"{root} does not exist"})
                jid = start_download(req["repo"], req["group"], root)
                return self._json(200, {"ok": True, "id": jid})

            if self.path == "/api/download/cancel":
                req = self._read()
                with JOBS_LOCK:
                    j = JOBS.get(req["id"])
                if j:
                    j["cancel"] = True
                return self._json(200, {"ok": bool(j)})

            if self.path == "/api/context":
                return self._json(200, context_usage(self._read(MAX_BODY), self._role))
            if self.path == "/api/chat":
                req = self._read(MAX_BODY)
                lt = link_target(req.get("model"))
                upstream = None
                if lt is not None:
                    upstream, rid = lt
                    target, route = None, {"mode": "link", "model": req["model"]}
                else:
                    with LOCK:
                        target, route = resolve_request_model({"model": req.get("model", ""),
                                                               "messages": req.get("messages", [])})
                        if target is None:
                            msg = route.get("error") or (f"model not found: {req.get('model')}" if req.get("model")
                                                         else "no model loaded")
                            return self._json(409, {"error": msg})
                        ok, why = ensure_loaded(target, own=0)
                    if not ok:
                        return self._json(503, {"error": f"could not load {target['id']}: {why}"})
                if target is not None and _has_image(req) and not target.get("mmproj"):
                    return self._json(400, {"error": f"{target['id']} has no vision projector (mmproj)"})
                preset = load_presets().get(req.get("preset", "default"),
                                            PRESET_DEFAULTS)
                msgs = list(req["messages"])
                if preset.get("system") and not (
                        msgs and msgs[0].get("role") == "system"):
                    msgs.insert(0, {"role": "system",
                                    "content": preset["system"]})
                try:
                    msgs, sources = rag_context(req, msgs)
                except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
                    return self._json(400, {"error": f"retrieval failed: {e}"})
                if req.get("compress"):
                    try:
                        msgs, hist = compress_history(msgs, req, int(preset.get("max_tokens", 2048) or 0), upstream)
                    except Exception as e:      # compression is a help, never the reason a chat fails
                        hist = {"history": "failed", "error": str(e)[:200]}
                    req["_compressed_history"] = hist
                payload = {"messages": msgs,
                           "temperature": preset.get("temperature", 0.7),
                           "top_p": preset.get("top_p", 0.95),
                           "max_tokens": preset.get("max_tokens", 2048),
                           "stream": True,
                           "stream_options": {"include_usage": True}}
                if req.get("confidence"):
                    # Per-token top-5 probabilities from the engine (llama-server, vLLM), for the
                    # confidence strip under the reply. They cost little and change nothing sampled.
                    payload.update(logprobs=True, top_logprobs=5)
                if preset.get("json_schema"):
                    try:
                        schema = json.loads(preset["json_schema"])
                    except json.JSONDecodeError as e:
                        return self._json(400, {"error": f"bad json_schema: {e}"})
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "response", "schema": schema},
                    }
                elif preset.get("grammar"):
                    payload["grammar"] = preset["grammar"]
                if upstream is not None:
                    payload["model"] = rid
                self._route = route
                self._send_headers_sse()
                if sources:
                    self.wfile.write(f"data: {json.dumps({'sources': sources})}\n\n".encode())
                comp = {k: v for k, v in (("passages", req.get("_compressed_passages")),
                                          ("history", req.get("_compressed_history"))) if v}
                if comp:
                    self.wfile.write(f"data: {json.dumps({'compressed': comp})}\n\n".encode())
                touch(+1)
                try:
                    # MCP tools run with the host owner's permissions: owner sessions only
                    hub = mcp_hub() if req.get("tools") and self._role == "owner" else None
                    if hub is not None and hub.openai_tools():
                        out = chat_with_tools(self.wfile, payload, msgs, hub, upstream=upstream)
                    else:
                        out = proxy("/v1/chat/completions", payload, stream_to=self.wfile, upstream=upstream)
                    if target is not None:
                        note_reply(target, msgs, out["text"])
                except Exception as e:
                    self.wfile.write(
                        f"data: {json.dumps({'error': str(e)})}\n\n".encode())
                finally:
                    touch(-1)
                return

            self._json(404, {"error": "not found"})
        except BrokenPipeError:
            pass
        except ValueError as e:
            self._json(400, {"error": str(e)[:300]})
        except Exception as e:
            try:
                self._json(500, {"error": str(e)[:300]})
            except Exception:
                pass


_PAIR_STYLE = """<style>
:root{--bg:#f7f7f5;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b64;--line:#e3e2dd;--acc:#2c5f8a;--accfg:#fff;--ok:#2f7d4f;--no:#b23c2e;--code:#f3f2ee;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161615;--panel:#1e1e1c;--fg:#ecebe6;--mut:#9b9a93;--line:#34332f;--acc:#7aa7d6;--accfg:#0f0f0e;--ok:#5fb27f;--no:#e0705f;--code:#262522;color-scheme:dark}}
:root[data-theme=dark]{--bg:#161615;--panel:#1e1e1c;--fg:#ecebe6;--mut:#9b9a93;--line:#34332f;--acc:#7aa7d6;--accfg:#0f0f0e;--ok:#5fb27f;--no:#e0705f;--code:#262522;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--fg)}
header{display:flex;gap:1rem;align-items:center;padding:.6rem 1rem;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:15px;margin:0}a{color:var(--acc)}
main{max-width:900px;margin:0 auto;padding:1rem;display:grid;gap:1rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.9rem}
h2{font-size:13px;margin:0 0 .5rem;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
input{font:13px ui-monospace,monospace;padding:.35rem .45rem;border:1px solid var(--line);border-radius:5px;width:100%;background:var(--bg);color:var(--fg)}
select{font:13px ui-sans-serif,system-ui;padding:.3rem .4rem;border:1px solid var(--line);border-radius:5px;background:var(--bg);color:var(--fg)}
#access{width:100%}
button{font:500 13px ui-sans-serif,system-ui;padding:.4rem .8rem;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer}
button.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}
.note{font-size:12.5px;color:var(--mut)}.err{color:var(--no)}.ok{color:var(--ok)}
code,.code{font:12.5px ui-monospace,monospace;background:var(--code);padding:.15rem .35rem;border-radius:4px;word-break:break-all}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:.3rem .25rem;border-bottom:1px solid var(--line)}
.row{display:flex;gap:.5rem;align-items:flex-end;flex-wrap:wrap}.row>div{flex:1;min-width:180px}
label{display:block;font-size:12px;color:var(--mut);margin:.3rem 0 .1rem}
</style><script>try{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{}');for(const k in v)d.style.setProperty('--'+k,v[k])}catch{document.documentElement.dataset.theme='dark'}</script>"""

PAIR_PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pair device</title>""" + _PAIR_STYLE + r"""
<header><h1>OmniCortex Studio · Pair this device</h1></header>
<main><div class="card"><h2>Pair</h2><div id="msg" class="note">Reading the pairing code…</div>
<div id="form" style="display:none"><label>Name for this device</label><input id="name">
<button class="pri" id="go" style="margin-top:.6rem">Pair and open Studio</button></div></div></main>
<script>
const p=new URLSearchParams(location.hash.slice(1)), code=p.get('c'), acc=p.get('a'), $=s=>document.querySelector(s);
const accText=acc==='p'?'Access lasts until the owner revokes it.':acc?`Access is temporary: it ends ${acc} after pairing.`:'';
history.replaceState(null,'',location.pathname);         // keep the code out of history
if(!code){ $('#msg').innerHTML='<span class="err">This link has no pairing code. Ask the owner for a new one.</span>'; }
else { $('#msg').textContent='Pairing code '+code+'. This browser will get its own access, which the owner can revoke. '+accText;
  $('#name').value=(navigator.userAgentData?.platform||navigator.platform||'browser')+' browser'; $('#form').style.display=''; }
$('#go').onclick=async()=>{ const r=await fetch('/api/pair/claim',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({code,name:$('#name').value,cookie:true})}); const j=await r.json().catch(()=>({}));
  if(r.ok){ if(j.expires) alert('Paired. Access ends '+new Date(j.expires*1000).toLocaleString()+'.'); location.href='/'; } else { $('#msg').innerHTML='<span class="err">'+(j.error||r.status)+'</span>'; } };
</script>"""

LINK_PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio Devices</title>""" + _PAIR_STYLE + r"""
<header><h1>Studio · Devices</h1><a href="/">← Studio</a></header>
<main>
<div class="card"><h2>Pair a device with this Studio</h2>
 <div class="note">Creates a one-time link (<span id="codettl">5 minutes</span>). Open it on a phone or laptop, or paste it into another Studio below. Each device gets its own token, revocable here; the master token is never shared. The link carries this server's certificate fingerprint, so the other side pins it instead of trusting any certificate.</div>
 <div class="row" style="margin-top:.5rem"><div style="max-width:260px"><label>Access</label><select id="access"></select></div>
 <div id="customBox" style="max-width:160px;display:none"><label>Duration (e.g. 12h, 3d)</label><input id="custom"></div>
 <button class="pri" id="start">Create pairing link</button></div>
 <div id="policy" class="note" style="margin-top:.3rem"></div>
 <div id="pairout" style="margin-top:.6rem"></div></div>
<div class="card"><h2>Paired devices</h2><div id="devs" class="note">loading…</div></div>
<div class="card"><h2>Use another machine's models here</h2>
 <div class="note">On the other machine's Studio, open Connect → Devices → Create pairing link, and paste it here. Its models then appear in this Studio's model picker and its <code>/v1</code> API as <code>name:model</code>; requests are served by that machine.</div>
 <div class="row"><div><label>Pairing link from the other Studio</label><input id="lnk" placeholder="https://host:7870/pair#c=…&fp=…"></div>
 <div style="max-width:200px"><label>Name here</label><input id="lname" placeholder="gpu-box"></div><button class="pri" id="add">Link</button></div>
 <div id="addmsg" class="note" style="margin-top:.4rem"></div>
 <div id="links" style="margin-top:.6rem"></div></div>
</main>
<script>
const $=s=>document.querySelector(s), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const post=(u,b)=>fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});
const when=t=>t?new Date(t*1000).toLocaleString():'never';
let policy=null;
const left=t=>{ const s=t-Date.now()/1000; if(s<=0) return 'expired';
  return s>86400?`${Math.floor(s/86400)}d ${Math.floor(s%86400/3600)}h left`:s>3600?`${Math.floor(s/3600)}h ${Math.floor(s%3600/60)}m left`:`${Math.ceil(s/60)}m left`; };
const accessCell=d=>d.persistent?'persistent':d.expired?`<span class="err">expired ${when(d.expires)}</span>`:`until ${when(d.expires)} <span class="note">(${left(d.expires)})</span>`;
function accessChoice(){ const v=$('#access').value; if(v==='persistent') return {persistent:true};
  if(v==='custom') return {persistent:false, ttl:$('#custom').value.trim()}; return {persistent:false, ttl:v}; }
function renderPolicy(pol){ policy=pol; const keep=$('#access').value;
  $('#access').innerHTML=(pol.allow_persistent?'<option value="persistent">Persistent (until revoked)</option>':'')+
    pol.presets.map(p=>`<option value="${p}">Temporary: ${p}</option>`).join('')+`<option value="custom">Temporary: custom…</option>`;
  $('#access').value=keep||(pol.default==='persistent'?'persistent':pol.default);
  if(!$('#access').value) $('#access').selectedIndex=0;
  $('#codettl').textContent=pol.code_ttl>=120?`${Math.round(pol.code_ttl/60)} minutes`:`${pol.code_ttl} seconds`;
  $('#policy').textContent=`Host policy: temporary access up to ${pol.max_ttl}`+(pol.allow_persistent?'; persistent allowed.':'; persistent pairing is disabled on this host.'); }
$('#access').onchange=()=>{ $('#customBox').style.display=$('#access').value==='custom'?'':'none'; };
async function load(){ const r=await fetch('/api/devices'); const j=await r.json();
  if(!r.ok){ $('#devs').innerHTML='<span class="err">'+esc(j.error)+'</span>'; $('#start').disabled=true; $('#add').disabled=true; return; }
  renderPolicy(j.policy);
  $('#devs').innerHTML=j.devices.length?'<table><tr><th>device</th><th>paired</th><th>last seen</th><th>access</th><th></th></tr>'+j.devices.map(d=>`<tr><td>${esc(d.name)}</td><td>${when(d.created)}</td><td>${when(d.last_seen)}</td><td>${accessCell(d)}</td><td style="white-space:nowrap"><select data-u="${esc(d.id)}"><option value="">change…</option>${policy.allow_persistent&&!d.persistent?'<option value="persistent">make persistent</option>':''}${policy.presets.map(p=>`<option value="${p}">${d.persistent?'expire in':d.expired?'renew for':'reset to'} ${p}</option>`).join('')}</select> <button data-r="${esc(d.id)}">Revoke</button></td></tr>`).join('')+'</table>':'No paired devices.';
  document.querySelectorAll('[data-r]').forEach(b=>b.onclick=async()=>{ if(confirm('Revoke this device? It loses access immediately.')){ await post('/api/devices/revoke',{id:b.dataset.r}); load(); } });
  document.querySelectorAll('[data-u]').forEach(sel=>sel.onchange=async()=>{ const v=sel.value; if(!v) return;
    const r=await post('/api/devices/update', v==='persistent'?{id:sel.dataset.u,persistent:true}:{id:sel.dataset.u,persistent:false,ttl:v});
    if(!r.ok) alert((await r.json()).error); load(); });
  $('#links').innerHTML=j.links.length?'<table><tr><th>name</th><th>host</th><th>access</th><th>models</th><th></th></tr>'+j.links.map(l=>`<tr><td>${esc(l.name)}</td><td><code>${esc(l.url)}</code> ${l.pinned?'🔒 pinned':''}</td><td>${l.expires?accessCell({expires:l.expires,expired:l.expires<Date.now()/1000}):'persistent'}</td><td>${l.error?'<span class="err">'+esc(l.error)+'</span>':l.models}</td><td><button data-l="${esc(l.name)}">Unlink</button></td></tr>`).join('')+'</table>':'';
  document.querySelectorAll('[data-l]').forEach(b=>b.onclick=async()=>{ await post('/api/links/remove',{name:b.dataset.l}); load(); }); }
$('#start').onclick=async()=>{ const r=await post('/api/pair/start',accessChoice()); const j=await r.json();
  const grants=j.persistent?'persistent access (until revoked)':`temporary access for ${Math.round(j.access_seconds/3600*10)/10} h after pairing`;
  $('#pairout').innerHTML=r.ok?`<div><span class="code" id="plink">${esc(j.link)}</span> <button id="cp">Copy</button></div><div class="note">Grants ${grants}. Code <b>${esc(j.code)}</b>, claimable until ${new Date(j.expires*1000).toLocaleTimeString()}, single use.${j.fingerprint?'':' <span class="err">No TLS on this Studio: use the link only inside a VPN.</span>'}</div>`:'<span class="err">'+esc(j.error)+'</span>';
  if(r.ok) $('#cp').onclick=()=>navigator.clipboard.writeText(j.link); };
$('#add').onclick=async()=>{ $('#addmsg').textContent='pairing…'; const r=await post('/api/links/add',{link:$('#lnk').value.trim(),name:$('#lname').value.trim()||'remote',device_name:location.host});
  const j=await r.json(); $('#addmsg').innerHTML=r.ok?`<span class="ok">Linked ${esc(j.name)}: ${j.models.length} models.</span>`:'<span class="err">'+esc(j.error)+'</span>'; if(r.ok){ $('#lnk').value=''; } load(); };
load();
</script>"""

PAGE = r"""<!DOCTYPE html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniCortex Studio</title><link rel="icon" href="data:,"><script>try{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{}');for(const k in v)d.style.setProperty('--'+k,v[k])}catch{document.documentElement.dataset.theme='dark'}</script>
<style>
:root{--bg:#f7f7f5;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b64;--line:#e3e2dd;--ok:#2f7d4f;--no:#b23c2e;--warn:#a8701c;
  --acc:#2c5f8a;--accfg:#fff;--soft:#f0f5fa;--code:#f3f2ee;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#111316;--panel:#181b20;--fg:#e6e8eb;--mut:#9aa1ab;
  --line:#2a2f37;--ok:#5fcf8f;--no:#ff8b7e;--warn:#e7b65c;--acc:#6ea8e0;--accfg:#0d1117;--soft:#1d2733;--code:#20242b;color-scheme:dark}}
:root[data-theme=dark]{--bg:#111316;--panel:#181b20;--fg:#e6e8eb;--mut:#9aa1ab;--line:#2a2f37;--ok:#5fcf8f;--no:#ff8b7e;
  --warn:#e7b65c;--acc:#6ea8e0;--accfg:#0d1117;--soft:#1d2733;--code:#20242b;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--fg);height:100vh;display:flex;flex-direction:column}
header{padding:.55rem 1rem;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:.5rem;background:var(--panel);flex-wrap:wrap}
h1{margin:0;font-size:15px;font-weight:600}
.status{font-size:13px;color:var(--mut);min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--line);margin-right:.35rem}.dot.on{background:var(--ok)}
.chip{font:12px ui-monospace,monospace;border:1px solid var(--line);border-radius:999px;padding:.1rem .5rem;color:var(--mut);cursor:pointer;background:var(--bg)}
main{flex:1;display:grid;grid-template-columns:330px 1fr;min-height:0}
@media(max-width:800px){main{grid-template-columns:1fr;grid-template-rows:auto 1fr}aside{max-height:30vh;border-right:none;border-bottom:1px solid var(--line)}
  .dials{gap:.45rem .7rem;padding:.4rem .8rem}.dials input[type=range]{width:90px}#export{display:none}
  form{padding:.5rem .8rem}#log{padding:.7rem .8rem}header{gap:.5rem}#api{display:none}}
aside{border-right:1px solid var(--line);overflow-y:auto;padding:.7rem;background:var(--panel)}
.tabs{display:flex;gap:.25rem;margin-bottom:.6rem;flex-wrap:wrap}.tabs button{flex:1 1 auto;min-width:0;padding:.35rem .45rem;font-size:12.5px}
section.chat{display:flex;flex-direction:column;min-height:0;min-width:0}
.m{border:1px solid var(--line);border-radius:8px;padding:.5rem .65rem;margin-bottom:.4rem;cursor:pointer;background:var(--panel);position:relative}
.m:hover{background:var(--soft)}.m.sel{border-color:var(--acc);background:var(--soft)}
.mn{font-weight:500;font-size:13px;word-break:break-all}.mm{font-size:12px;color:var(--mut);margin-top:.1rem}
.x{position:absolute;right:.4rem;top:.3rem;border:none;background:none;color:var(--mut);cursor:pointer;padding:0 .3rem}
.tag{display:inline-block;font-size:11px;padding:0 .35rem;border:1px solid var(--line);border-radius:4px;margin-right:.25rem}
.bad{color:var(--no)}.warn{color:var(--warn)}.okc{color:var(--ok)}
details.eo{margin-top:.5rem;border:1px solid var(--line);border-radius:6px;padding:.3rem .5rem}details.eo>summary{cursor:pointer;font-size:12.5px}
.eoG{margin:.3rem 0}.eoG>summary{cursor:pointer;font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.03em}
.eoRow{margin:.25rem 0}.eoRow label{margin:.2rem 0 .1rem}.eoRow code{font-size:10.5px;color:var(--mut)}.eoRow .note{font-size:11px}
label{display:block;font-size:12px;color:var(--mut);margin:.4rem 0 .1rem}
input,select,textarea{font:12px ui-monospace,monospace;padding:.3rem .4rem;border:1px solid var(--line);border-radius:5px;width:100%;background:var(--bg);color:var(--fg)}
input[type=checkbox]{width:auto}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:.5rem}
button{font:500 13px ui-sans-serif,system-ui;padding:.4rem .8rem;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer}
button:hover{background:var(--soft)}button:disabled{opacity:.45;cursor:default}
button.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}
#log{flex:1;overflow-y:auto;padding:1rem 1.2rem}
.msg{max-width:800px;margin:0 auto 1.1rem}
.who{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;margin-bottom:.2rem;display:flex;gap:.6rem;align-items:center}
.who button{font-size:11px;padding:0 .4rem;border:none;background:none;color:var(--mut)}
.body{white-space:pre-wrap;word-wrap:break-word}
.body pre{background:var(--code);padding:.6rem .7rem;border-radius:6px;overflow-x:auto;white-space:pre;font:12.5px ui-monospace,monospace}
.body code{background:var(--code);padding:0 .25rem;border-radius:3px;font:12.5px ui-monospace,monospace}
details.think{color:var(--mut);font-size:13px;border-left:2px solid var(--line);padding-left:.7rem;margin-bottom:.5rem}
details.think summary{cursor:pointer}
.stats{font-size:11px;color:var(--mut);margin-top:.3rem}
.risk{font-size:12px;margin-top:.35rem}.risk:empty{display:none}
.risk .sum{color:var(--mut);margin-bottom:.25rem}.risk .sum b{color:var(--fg)}
.risk .rt{white-space:pre-wrap;word-wrap:break-word;line-height:1.7;font-size:13px}
.pchk{font-size:12px;margin-top:.35rem}.pchk:empty{display:none}.pchk .sum{color:var(--mut);margin-bottom:.25rem}.pchk .sum b{color:var(--fg)}.pchk .sum b.bad{color:var(--no)}
.pchk .rt{white-space:pre-wrap;word-wrap:break-word;line-height:1.7;font-size:13px;color:var(--fg)}.pchk mark{background:color-mix(in srgb,var(--acc) 30%,transparent);color:inherit;border-radius:3px;padding:0 1px}
.risk .rt span{border-radius:2px}.risk .rt .fl{text-decoration:underline 2px var(--no);text-underline-offset:3px}
.imgs{display:flex;gap:.4rem;flex-wrap:wrap;margin:.3rem 0}.imgs img{max-height:120px;max-width:200px;border-radius:6px;border:1px solid var(--line)}
form{border-top:1px solid var(--line);padding:.6rem 1.2rem;display:flex;gap:.5rem;background:var(--panel);align-items:flex-end;flex-wrap:wrap}
#in{flex:1;min-width:200px;font:14px ui-sans-serif,system-ui;padding:.5rem .6rem;resize:none}
#pending{width:100%}
.dials{border-top:1px solid var(--line);padding:.5rem 1.2rem;font-size:12px;color:var(--mut);display:flex;gap:1rem;align-items:center;background:var(--panel);flex-wrap:wrap}
.dials select{width:auto}.dials input[type=range]{width:140px}
.note{font-size:12px;color:var(--mut);margin:.4rem 0}
.empty{max-width:560px;margin:15vh auto;text-align:center;color:var(--mut)}
ol.gs{list-style:none;padding:0;margin:.8rem 0 0;display:grid;gap:.55rem}ol.gs li{display:flex;gap:.6rem;align-items:flex-start;color:var(--fg)}
ol.gs .ck{flex:none;width:20px;height:20px;border-radius:50%;border:1.5px solid var(--line);display:flex;align-items:center;justify-content:center;font-size:12px;margin-top:1px}
ol.gs li.done .ck{background:var(--ok);border-color:var(--ok);color:var(--accfg)}ol.gs li.done b{color:var(--mut)}
.starters{display:grid;grid-template-columns:1fr 1fr;gap:.4rem;margin-top:.35rem}.starters button{text-align:left;font-weight:400;font-size:12.5px;line-height:1.35}
.tasks{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.4rem;margin-top:.35rem}
.task{display:flex;flex-direction:column;gap:.15rem;padding:.5rem .6rem;border:1px solid var(--line);border-radius:8px;text-decoration:none;color:inherit;font-size:12.5px;line-height:1.35}
.task:hover{border-color:var(--acc)}.task b{color:var(--fg)}.task span{color:var(--mut);font-size:12px}
@media(max-width:760px){.tasks{grid-template-columns:1fr 1fr}}
@media(max-width:600px){.starters{grid-template-columns:1fr}.tasks{grid-template-columns:1fr}.empty{margin:4vh auto}}
.warnico{color:#e8890c;cursor:help;font-weight:700;margin-left:.3rem}
.stat{font-size:11px;color:var(--mut);cursor:help}
.route{font-size:11px;color:var(--mut);margin-top:.2rem}
table.st{width:100%;border-collapse:collapse;font-size:11.5px;margin-top:.4rem}
table.st th,table.st td{text-align:left;padding:.15rem .3rem;border-bottom:1px solid var(--line)}
#modelPick{width:auto;max-width:260px}
.tool{font-size:12px;border:1px solid var(--line);border-radius:6px;padding:.35rem .5rem;margin:.3rem 0;background:var(--code)}
.tool .will{margin:.3rem 0}.tool .wl{display:flex;gap:.4rem;align-items:baseline;margin:.15rem 0}.tool .wl code{flex:1;white-space:pre-wrap;word-break:break-all;background:var(--bg);padding:.15rem .35rem;border-radius:4px}
.tool .wl.deny code{outline:1px solid var(--no)}.tool .wl.ask code{outline:1px solid var(--warn,#e7b65c)}
.tool .qp{display:flex;gap:.35rem;flex-wrap:wrap;align-items:center;margin-top:.3rem}
.tool .tn{font:600 12px ui-monospace,monospace}.tool pre{white-space:pre-wrap;margin:.2rem 0;font:12px ui-monospace,monospace;max-height:220px;overflow:auto}
.tool button{font-size:11px;padding:.1rem .5rem;margin-right:.3rem}
.src{font-size:12px;color:var(--mut);margin-top:.3rem}.src details{margin:.1rem 0}.src summary{cursor:pointer}
.src pre{white-space:pre-wrap;font:12px ui-monospace,monospace;background:var(--code);padding:.4rem;border-radius:5px;margin:.2rem 0}
/* the context strip over the chat, and what it opens: environment, context usage, outputs */
section.chat{position:relative}
#ctxStrip{display:flex;align-items:center;gap:.6rem;padding:.3rem 1.2rem;border-bottom:1px solid var(--line);background:var(--panel);font-size:12.5px;color:var(--mut);cursor:pointer;user-select:none;min-width:0}
#ctxStrip:hover{background:var(--soft)}#ctxStrip:focus-visible{outline:2px solid var(--acc);outline-offset:-2px}
#ctxStrip .sm{color:var(--fg);font-weight:500;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:32%}
#ctxStrip .cbar{flex:1;max-width:340px;height:7px;margin:0}#ctxStrip .chev{margin-left:auto;transition:transform .15s}
#ctxStrip[aria-expanded=true] .chev{transform:rotate(180deg)}
#side{position:absolute;left:0;right:0;top:var(--strip-h,31px);z-index:50;background:var(--panel);border-bottom:1px solid var(--line);box-shadow:0 10px 24px rgba(0,0,0,.22);
  padding:.4rem 1.2rem .7rem;font-size:13px;display:grid;grid-template-columns:1fr 1.2fr 1fr;gap:1.2rem;max-height:60vh;overflow-y:auto}
#side[hidden]{display:none}
#side h3{margin:.3rem 0 .45rem;font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);display:flex;gap:.4rem}#side h3 .r{margin-left:auto;text-transform:none;letter-spacing:0}
@media(max-width:900px){#side{grid-template-columns:1fr;gap:.3rem}}
.kv{display:grid;grid-template-columns:auto 1fr;gap:.15rem .7rem;font-size:12.5px;margin:0}.kv dt{color:var(--mut)}.kv dd{margin:0;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cbar{display:flex;height:10px;border-radius:5px;overflow:hidden;background:var(--code);margin:.2rem 0 .5rem;border:1px solid var(--line)}
.cbar span{display:block;height:100%;min-width:0}.cbar .rsv{background:repeating-linear-gradient(45deg,var(--line) 0 3px,transparent 3px 6px)}
.cleg{display:grid;grid-template-columns:10px 1fr auto;gap:.2rem .45rem;align-items:center;font-size:12.5px}.cleg i{width:10px;height:10px;border-radius:3px;display:block}
.cleg .n{color:var(--mut);font-variant-numeric:tabular-nums}
.outs a,.outs div.o{display:flex;gap:.45rem;align-items:baseline;padding:.25rem .3rem;border-radius:5px;color:var(--fg);text-decoration:none;font-size:12.5px}
.outs a:hover{background:var(--soft)}.outs .k{color:var(--mut);font-size:11px;margin-left:auto;white-space:nowrap}
</style>
<header>
  <h1>OmniCortex Studio</h1>
  <div class="status"><span id="dot" class="dot"></span><span id="st">no model loaded</span></div>
  <span class="chip" id="api" title="OpenAI-compatible endpoint; click to copy"></span>
  <span style="margin-left:auto"></span>
  <button id="theme" title="toggle theme">◐</button>
  <button id="unload" disabled>Unload</button>
</header>
<main>
<aside>
  <div class="tabs">
    <button data-tab="chats" class="pri">Chats</button>
    <button data-tab="local">Models</button>
    <button data-tab="hub">Discover</button>
    <button data-tab="docs">Docs</button>
    <button data-tab="tools">Tools</button>
  </div>
  <div id="tab-tools" style="display:none">
    <div class="note">MCP servers whose tools the model may call when <b>tools</b> is on under the chat. Every call asks first unless the server's <code>autoApprove</code> allows it.</div>
    <div id="mcpcfg" class="note"></div>
    <button id="mcpreload" style="width:100%;margin:.4rem 0">Reconnect / reload config</button>
    <div id="mcplist"></div>
  </div>
  <div id="tab-docs" style="display:none">
    <div class="note">Document collections for chat. Pick one under the chat ("docs") and replies cite the passages they used.</div>
    <div style="display:flex;gap:.4rem;margin:.4rem 0"><input id="newcoll" placeholder="new collection name"><button id="mkcoll">Create</button></div>
    <div id="colls"></div>
    <div id="collpanel" style="display:none">
      <hr style="border:none;border-top:1px solid var(--line);margin:.6rem 0">
      <b id="collname"></b> <span id="colldense" class="note"></span>
      <input type="file" id="docfile" multiple hidden>
      <button id="adddocs" style="width:100%;margin:.4rem 0">+ Add files (txt, md, code, html, pdf, docx)</button>
      <div id="docmsg" class="note"></div>
      <div id="docs"></div>
      <label>Try a search</label><input id="ragq" placeholder="query, Enter to search"><div id="raghits" class="note"></div>
      <button id="dropcoll" style="margin-top:.6rem;width:100%;color:var(--no)">Delete collection</button>
    </div>
  </div>
  <div id="tab-chats">
    <button id="newchat" style="width:100%;margin-bottom:.5rem">+ New chat</button>
    <div id="chats"></div>
  </div>
  <div id="tab-hub" style="display:none">
    <input id="q" placeholder="search GGUF models on Hugging Face">
    <button id="go" style="width:100%;margin-top:.4rem">Search</button>
    <div id="hits" class="note">Search by model name, e.g. <a href="#" data-qs="qwen2.5 instruct">qwen2.5 instruct</a>, <a href="#" data-qs="llama 3.1 instruct">llama 3.1 instruct</a>, <a href="#" data-qs="gemma">gemma</a>, <a href="#" data-qs="phi">phi</a>, <a href="#" data-qs="mistral instruct">mistral instruct</a>. Each file then shows whether it fits this computer, and ★ marks the one to start with.</div>
    <div id="jobs"></div>
  </div>
  <div id="tab-local" style="display:none">
  <input id="filter" placeholder="filter models" style="margin-bottom:.5rem">
  <div id="list">scanning…</div>
  <div id="panel" style="display:none">
    <hr style="border:none;border-top:1px solid var(--line);margin:.7rem 0">
    <div id="mprof" style="margin-bottom:.4rem"></div>
    <label>Engine</label><select id="engine"><option value="llama">llama.cpp (llama-server)</option><option value="vllm">vLLM</option></select>
    <div class="row2">
      <div><label>GPU layers</label><input id="ngl" value="99"></div>
      <div><label>Context</label><input id="ctx" value="8192"></div>
    </div>
    <div class="row2">
      <div><label>Batch</label><input id="batch" value="2048"></div>
      <div><label>Threads (0=auto)</label><input id="threads" value="0"></div>
    </div>
    <div id="gpubox" style="display:none">
      <label>GPUs <span id="gpuhint" class="note"></span></label>
      <div class="row2">
        <div><label>Visible GPUs</label><input id="gpus" placeholder="all, or 0,1"></div>
        <div><label>Split</label><select id="split_mode"><option value="">default (layer)</option><option>layer</option><option>row</option><option>none</option></select></div>
      </div>
      <div class="row2">
        <div><label>Tensor split</label><input id="tensor_split" placeholder="1,1,1,1"></div>
        <div><label>Main GPU (-1 = auto)</label><input id="main_gpu" value="-1"></div>
      </div>
    </div>
    <div id="moebox" style="display:none">
      <label>Active experts <span id="moehint" class="note"></span></label>
      <input id="experts" value="0">
    </div>
    <label>Served name (API model id)</label><input id="served_name" placeholder="defaults to the model id">
    <label>LoRA adapter (optional)</label><input id="lora" placeholder="a LoRA .gguf file" data-pick="file" data-ext="gguf">
    <details id="eoBox" class="eo"><summary>Engine options <span id="eoCount" class="note"></span></summary>
      <div id="eoForm" class="note">loading…</div></details>
    <label>Extra flags</label><input id="extra" placeholder="--override-tensor '\.ffn_.*_exps\.=CPU'">
    <label id="visionrow" style="display:none"><input type="checkbox" id="vision" checked> load vision projector <span id="mmname"></span></label>
    <label>H-Neuron classifier (optional, enables activation stats)</label><input id="classifier" placeholder="models/classifier.npz for this model">
    <button id="load" class="pri" style="margin-top:.7rem;width:100%">Load</button>
    <div id="radarBox" style="display:none"><div class="row2" style="align-items:end"><div><label>Compare with (2P)</label><select id="vsPick"></select></div><div class="note" id="radarNote"></div></div><div id="radar"></div></div>
    <div id="statsbox"></div>
    <div id="quantbox" class="note"></div>
    <div id="loadmsg" class="note"></div>
  </div>
  </div>
</aside>
<section class="chat">
  <div id="ctxStrip" role="button" tabindex="0" aria-controls="side" aria-expanded="false" title="Environment, context usage and outputs of this chat">
    <span class="sm" id="stripModel">no model loaded</span><div class="cbar" id="stripBar" aria-hidden="true"></div><span id="stripPct"></span><span id="stripOut"></span><span class="chev" aria-hidden="true">▾</span></div>
  <div id="side" hidden aria-label="Environment, context and outputs">
    <section><h3>Environment <span class="r" id="envState"></span></h3><dl class="kv" id="envKv"></dl></section>
    <section><h3>Context usage <span class="r" id="ctxPct"></span></h3>
      <div class="cbar" id="ctxBar" role="img" aria-label="context usage"></div><div class="cleg" id="ctxLeg"></div><div class="note" id="ctxNote"></div>
      <label style="margin-top:.5rem"><input type="checkbox" id="cmpPass"> Trim document passages to the sentences that match the question</label>
      <label>When the chat no longer fits <select id="cmpHist" style="width:auto;margin-left:.3rem"><option value="off">do nothing</option><option value="summarize">summarize the oldest messages</option><option value="cut">drop the oldest messages</option></select></label></section>
    <section><h3>Outputs <span class="r" id="outN"></span></h3><div class="outs" id="outList"></div>
      <button id="outExport" style="margin-top:.4rem;font-size:12px;padding:.25rem .6rem">Export chat as .md</button></section>
  </div>
  <div id="log"></div>
  <div class="dials">
    <span>preset</span><select id="preset"></select>
    <span title="retrieve passages from a document collection for each message">docs</span><select id="ragPick"><option value="">none</option></select>
    <label id="toolsBox" style="display:none;margin:0" title="let the model call MCP tools (each call asks first)"><input type="checkbox" id="toolsOn"> tools</label>
    <label style="margin:0" title="after each reply, score every token with the model's hallucination classifier (needs studio --cett and a classifier in the model's settings)"><input type="checkbox" id="autoCheck"> check replies</label>
    <label style="margin:0" title="ask the engine for each token's top-5 probabilities and mark where the model was unsure (llama-server and vLLM; costs almost nothing)"><input type="checkbox" id="confOn"> confidence</label>
    <span>suppression α</span>
    <input type="range" id="alpha" min="0" max="1" step="0.05" value="1" disabled>
    <span id="av">1.00</span>
    <span id="adesc">no adapter loaded</span>
    <button id="export" style="margin-left:auto">Export .md</button>
  </div>
  <form id="f">
    <div id="pending" class="imgs"></div>
    <input type="file" id="file" accept="image/*" multiple hidden>
    <select id="modelPick" title="which model answers"></select><span id="pickWarn" class="warnico" style="display:none">⚠</span>
    <button type="button" id="attach" title="attach image (vision models)" disabled>📎</button>
    <textarea id="in" rows="2" placeholder="Message… (Enter to send, Shift+Enter for a new line)"></textarea>
    <button class="pri" id="send">Send</button>
  </form>
</section>
</main>
<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let models=[], sel=null, busy=false, ctrl=null, status_={}, pendingImgs=[];
let chat={id:null,title:'New chat',messages:[]};
const F=["ngl","ctx","batch","threads","experts","served_name","lora","extra","classifier","gpus","split_mode","tensor_split","main_gpu"];
const TEXT_FIELDS=["served_name","lora","extra","classifier","gpus","split_mode","tensor_split"];
// ---------- engine options (scripts/engine_options.py: one table for the form, the flags and the optimizer)
let EO=null;
async function loadEO(engine, values){
  const r=await fetch('/api/engine/options?engine='+engine); EO=await r.json(); renderEO(values||{});
}
let EO_DEVS=[];
fetch('/api/engine/devices').then(r=>r.json()).then(j=>{EO_DEVS=j.devices||[]; if(EO_DEVS.length&&EO&&$('#eoForm').innerHTML) renderEO({...(sel&&sel.settings||{}),...readEO()})}).catch(()=>{});
const EO_DEV_KEYS=['device','draft_device'];
function eoInput(o,v){ const id='eo_'+o.key, dis=o.available?'':' disabled';
  if(EO_DEV_KEYS.includes(o.key)&&EO_DEVS.length) return `<div class="eoRow" title="${esc(o.help)}"><label>${esc(o.label)} <code>${esc(o.flag)}</code></label><input id="${id}" value="${esc(v??'')}" list="eoDevs_${o.key}" placeholder="engine default"${dis}><datalist id="eoDevs_${o.key}">${EO_DEVS.map(d=>`<option value="${esc(d.name)}">${esc(d.desc)} · ${d.free_mb} of ${d.total_mb} MiB free</option>`).join('')}</datalist>${o.help?`<div class="note">${esc(o.help)}</div>`:''}</div>`;
  if(o.kind==='bool') return `<label class="eoRow" title="${esc(o.help)}"><input type="checkbox" id="${id}"${v?' checked':''}${dis}> ${esc(o.label)} <code>${esc(o.flag)}</code></label>`;
  const ctl = o.kind==='choice' ? `<select id="${id}"${dis}><option value="">${o.default?esc(o.default)+' (default)':'engine default'}</option>${o.choices.map(c=>`<option${String(v)===c?' selected':''}>${esc(c)}</option>`).join('')}</select>`
    : `<input id="${id}" value="${esc(v??'')}" placeholder="${o.kind==='text'?'':'engine default'}"${dis}>`;
  return `<div class="eoRow" title="${esc(o.help)}"><label>${esc(o.label)} <code>${esc(o.flag)}</code></label>${ctl}${o.help?`<div class="note">${esc(o.help)}</div>`:''}</div>`; }
function renderEO(s){
  if(!EO) return; let n=0, html='';
  for(const [g,opts] of Object.entries(EO.groups)){
    const have=opts.filter(o=>o.available), lack=opts.filter(o=>!o.available);
    n+=opts.filter(o=>s[o.key]!==undefined&&s[o.key]!==''&&s[o.key]!==null&&s[o.key]!==false&&!(o.key==='flash_attn'&&s[o.key]==='auto')&&!(o.key==='parallel'&&+s[o.key]===1)).length;
    html+=`<details class="eoG"${opts.some(o=>s[o.key])?' open':''}><summary>${esc(g)}</summary>${have.map(o=>eoInput(o,s[o.key])).join('')}`+
      (g==='Speculative decoding'&&EO.engine==='llama'?`<div class="eoRow"><button type="button" onclick="suggestDraft()" title="Find models in your folders that share this model's vocabulary, and a device the big model does not use">Suggest a draft model and device</button><div id="draftOut" class="note"></div></div>`:'')+
      (lack.length?`<details><summary class="note">${lack.length} not in this ${EO.engine==='llama'?'llama-server':'vLLM'} build</summary>${lack.map(o=>eoInput(o,s[o.key])).join('')}<div class="note">Update the engine (Setup → Build llama.cpp) to use these.</div></details>`:'')+`</details>`;
  }
  const env=s.env||{};
  html+=`<details class="eoG"${Object.keys(env).length||s.wrap?' open':''}><summary>Environment and launch</summary>
    <div class="note">Only these documented variables can be set. ${Object.entries(EO.env).map(([k,h])=>`<div class="eoRow"><label>${esc(k)}</label><input data-env="${esc(k)}" value="${esc(env[k]??'')}" placeholder="unset"><div class="note">${esc(h)}</div></div>`).join('')}</div>
    <div class="eoRow"><label>Launch wrapper</label><select id="eo_wrap"><option value="">none</option>${Object.entries(EO.wrappers).map(([k,h])=>`<option value="${k}"${s.wrap===k?' selected':''}>${esc(h)}</option>`).join('')}</select></div>
    <div class="row2"><div><label>NUMA node</label><input id="eo_numa_node" value="${esc(s.numa_node??0)}"></div><div><label>taskset cores</label><input id="eo_taskset_cores" value="${esc(s.taskset_cores??'')}" placeholder="0,2,4,6"></div></div></details>`;
  if(EO.engine==='vllm') html=(EO.vllm_installed?'':'<div class="bad">vLLM is not installed here (pip install vllm), or set its command with studio --vllm.</div>')+
    `<div class="eoRow"><label>Tokenizer (GGUF only)</label><input id="eo_tokenizer" value="${esc(s.tokenizer??'')}" placeholder="the original HF model id"></div>`+html;
  $('#eoForm').innerHTML=html; $('#eoCount').textContent=n?`(${n} set)`:'';
}
async function suggestDraft(){
  const out=$('#draftOut'); if(!sel){out.textContent='Pick a model first.';return}
  out.textContent='Reading vocabularies…';
  const r=await (await fetch('/api/models/draft?model='+encodeURIComponent(sel.id))).json();
  if(r.error){out.innerHTML=`<span class="bad">${esc(r.error)}</span>`;return}
  if(!r.candidates.length){out.textContent='No model in your folders shares this one\'s vocabulary and is small enough (under 40% of its size). Download a smaller model of the same family, or try an n-gram method, which needs no draft.';return}
  const pl=r.placement||{};
  out.innerHTML=(pl.why?`<div>${esc(pl.why)}.</div>`:'')+r.candidates.map((c,i)=>`<div class="eoRow" style="display:flex;gap:.4rem;align-items:center"><button type="button" data-i="${i}">Use</button><span title="${esc(c.path)}" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(c.id)}</span><span class="note">${(c.ratio*100).toFixed(0)}% of the size${c.same_arch?', same family':''}</span></div>`).join('');
  out.querySelectorAll('button[data-i]').forEach(b=>b.onclick=()=>{const c=r.candidates[+b.dataset.i],set=(k,v)=>{const el=$('#eo_'+k);if(el&&v!=null&&v!=='')el.value=v};
    set('draft_model',c.path);set('spec_type',$('#eo_spec_type')&&[...$('#eo_spec_type').options].some(o=>o.value==='draft-simple')?'draft-simple':'');
    set('draft_device',pl.draft_device);set('draft_ngl',pl.draft_ngl);if(pl.device)set('device',pl.device);
    b.textContent='✓ set'; out.insertAdjacentHTML('beforeend','<div>Filled in: save the settings and load the model to use it.</div>')});
}
function readEO(){ const s={};
  for(const opts of Object.values(EO?EO.groups:{})) for(const o of opts){ const el=$('#eo_'+o.key); if(!el) continue;
    if(o.kind==='bool'){ if(el.checked) s[o.key]=true; continue; }
    const v=el.value.trim(); if(v==='') continue; s[o.key]= o.kind==='int'?parseInt(v): o.kind==='float'?parseFloat(v): v; }
  s.env={}; $$('[data-env]').forEach(el=>{ if(el.value.trim()) s.env[el.dataset.env]=el.value.trim(); });
  s.wrap=$('#eo_wrap')?$('#eo_wrap').value:''; if(s.wrap){ s.numa_node=$('#eo_numa_node').value.trim(); s.taskset_cores=$('#eo_taskset_cores').value.trim(); }
  if($('#eo_tokenizer')&&$('#eo_tokenizer').value.trim()) s.tokenizer=$('#eo_tokenizer').value.trim();
  return s; }
$('#engine').onchange=()=>loadEO($('#engine').value, {...(sel&&sel.settings||{}), ...readEO()});
const pct=x=>x==null?'–':(100*x).toFixed(0)+'%';
const NOSTATS_HELP=' Auto routing skips it: it only picks models whose answers have been graded. You can still load it or pick it by name. To add stats, grade it with TestQA (Jobs → Evaluate, or the link under its settings).';
function statBadge(m){ const s=m.stats; if(!s) return '';
  if(!s.eligible) return `<span class="warnico" title="${esc(s.why_not+NOSTATS_HELP)}">⚠</span>`;
  return ''; }
function statTitle(s){ const g=s.graded; let t=`Rolling stats over the last ${g.n} graded answers:\n`+
  `correct ${pct(g.accuracy)}, hallucinated (answered wrong) ${pct(g.hallucination_rate)}, abstained ${pct(g.abstention_rate)}\n`;
  for(const [k,v] of Object.entries(s.subjects||{})) t+=`  ${k}: ${pct(v.accuracy)} right, ${pct(v.hallucination_rate)} wrong, ${pct(v.abstention_rate)} abstained (n ${v.n})\n`;
  if(s.activation&&s.activation.n) t+=`H-Neuron activation over ${s.activation.n} replies: mean ${s.activation.mean.toFixed(2)}, flagged ${pct(s.activation.flagged_rate)}\n`;
  if(s.live&&s.live.n) t+=`Live traffic: ${s.live.n} replies, ${pct(s.live.abstention_rate)} declined\n`;
  if(g.sources) t+=`Graded by: ${Object.entries(g.sources).map(([k,v])=>k+' '+v).join(', ')}\n`;
  const refs=Object.entries(s.reference||{}); if(refs.length) t+=`Published scores (reference only, not used by auto): `+refs.map(([k,v])=>`${k} ${pct(v.score)}`).join(', ');
  return t; }
function statLine(m){ const s=m.stats; if(!s||!s.graded.n) return '';
  const g=s.graded; return `<div class="stat" title="${esc(statTitle(s))}">✓ ${pct(g.accuracy)} · halluc ${pct(g.hallucination_rate)} · abst ${pct(g.abstention_rate)} · n ${g.n}</div>`; }
// A TestQA job for this model, filled in (Jobs page).
function gradeLink(m){ if(!m) return ''; const ep=`m=${location.origin}/v1@${m.id}`;
  const out='runs/testqa-'+m.id.replace(/[^A-Za-z0-9_.-]+/g,'_')+'.json';
  return `<a href="/jobs#kind=testqa&endpoint=${encodeURIComponent(ep)}&publish_stats=${encodeURIComponent(location.origin)}&out=${encodeURIComponent(out)}">Grade it with TestQA →</a>`; }
function statsTable(s){ if(!s) return '';
  if(!s.graded.n && !(s.activation&&s.activation.n) && !(s.live&&s.live.n)) return `<div class="note"><span class="warnico">⚠</span> ${esc(s.why_not+NOSTATS_HELP)} ${gradeLink(sel)}</div>`;
  let h='<table class="st"><tr><th>subject</th><th>n</th><th>right</th><th>halluc</th><th>abstain</th></tr>';
  const rows=[['all',s.graded],...Object.entries(s.subjects||{})];
  for(const [k,v] of rows) if(v.n) h+=`<tr><td>${esc(k)}</td><td>${v.n}</td><td>${pct(v.accuracy)}</td><td>${pct(v.hallucination_rate)}</td><td>${pct(v.abstention_rate)}</td></tr>`;
  h+='</table>';
  if(s.activation&&s.activation.n) h+=`<div class="note">H-Neuron activation: ${s.activation.n} replies, mean score ${s.activation.mean.toFixed(2)}, p95 ${s.activation.p95.toFixed(2)}, flagged ${pct(s.activation.flagged_rate)}</div>`;
  if(s.live&&s.live.n) h+=`<div class="note">Live: ${s.live.n} replies, ${pct(s.live.abstention_rate)} declined</div>`;
  if(!s.eligible) h+=`<div class="note"><span class="warnico">⚠</span> ${esc(s.why_not)}</div>`;
  return h; }
// ---------- radar (DDR-style): per-subject accuracy and honesty, 1P = this model, 2P = a comparison
function radarAxes(stats){ const out={}; if(!stats) return out;
  for(const [k,v] of Object.entries(stats.subjects||{})) if(v.n>=3) out[k]=v.accuracy;
  if(stats.graded&&stats.graded.n>=3) out['honesty']=1-(stats.graded.hallucination_rate||0);
  return out; }
function renderRadar(){ if(!sel) return; const a=radarAxes(sel.stats);
  const others=models.filter(m=>m!==sel&&Object.keys(radarAxes(m.stats)).length>=3);
  const keep=$('#vsPick').value;
  $('#vsPick').innerHTML='<option value="">nobody</option>'+others.map(m=>`<option value="${esc(m.path)}">${esc(m.id)}</option>`).join('');
  if(keep&&others.some(m=>m.path===keep)) $('#vsPick').value=keep;
  const vs=others.find(m=>m.path===$('#vsPick').value);
  const b=vs?radarAxes(vs.stats):{};
  const axes=[...new Set([...Object.keys(a),...Object.keys(b)])];
  $('#radarBox').style.display=Object.keys(a).length?'':'none';
  if(axes.length<3){ $('#radar').innerHTML=''; $('#radarNote').textContent='The radar needs results in at least 3 subjects (3+ graded items each).'; return; }
  $('#radarNote').textContent='accuracy per subject; honesty = 1 − hallucination rate';
  $('#radar').innerHTML=radarSVG(axes,[{name:sel.id,v:a},...(vs?[{name:vs.id,v:b}]:[])]); }
function radarSVG(axes,series){ const C=nsChart(), W=380, H=280, cx=190, cy=142, R=92, n=axes.length;
  const pt=(i,r)=>{ const t=-Math.PI/2+2*Math.PI*i/n; return [cx+r*Math.cos(t), cy+r*Math.sin(t)]; };
  const poly=r=>axes.map((_,i)=>pt(i,r).map(x=>x.toFixed(1)).join(',')).join(' ');
  let g=''; for(const f of [1,.75,.5,.25]) g+=`<polygon points="${poly(R*f)}" fill="${f===1?'var(--soft,rgba(127,127,127,.08))':'none'}" stroke="var(--line)" stroke-width="${f===1?2:1}"/>`;
  axes.forEach((_,i)=>{ const [x,y]=pt(i,R); g+=`<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="var(--line)"/>`; });
  series.forEach((s,k)=>{ const c=C[k%C.length]; const pts=axes.map((ax,i)=>pt(i,R*Math.max(0,Math.min(1,s.v[ax]??0))).map(x=>x.toFixed(1)).join(',')).join(' ');
    g+=`<polygon points="${pts}" fill="${c}" fill-opacity="${k?0.22:0.32}" stroke="${c}" stroke-width="2.5" stroke-linejoin="round"/>`; });
  axes.forEach((ax,i)=>{ const [x,y]=pt(i,R+22); const anchor=Math.abs(x-cx)<8?'middle':(x>cx?'start':'end');
    const vals=series.map((s,k)=>s.v[ax]==null?'–':Math.round(s.v[ax]*100)).join(' / ');
    g+=`<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="${anchor}" font-size="11" font-weight="700" fill="var(--fg)" style="text-transform:uppercase;letter-spacing:.06em">${esc(ax)}</text>`+
       `<text x="${x.toFixed(1)}" y="${(y+12).toFixed(1)}" text-anchor="${anchor}" font-size="10" fill="var(--mut)">${vals}</text>`; });
  const legend=series.map((s,k)=>{ const c=C[k%C.length]; const vals=Object.values(s.v); const avg=vals.length?vals.reduce((x,y)=>x+y,0)/vals.length:0;
    const seg=Array.from({length:10},(_,i)=>`<span style="display:inline-block;width:9px;height:12px;margin-right:2px;border-radius:2px;background:${i<Math.round(avg*10)?c:'var(--line)'}"></span>`).join('');
    return `<div style="display:flex;align-items:center;gap:.4rem;margin-top:.25rem"><span style="font:700 11px ui-monospace,monospace;color:${c};border:2px solid ${c};border-radius:5px;padding:0 .3rem">${k+1}P</span><span class="note" style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(s.name)}</span>${seg}</div>`; }).join('');
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:380px;display:block;margin:.3rem auto 0">${g}</svg>${legend}`; }
$('#vsPick').onchange=renderRadar;
window.addEventListener('ns-theme',renderRadar);
function human(b){const u=["B","KB","MB","GB","TB"];let i=0;while(b>=1024&&i<4){b/=1024;i++}return b.toFixed(1)+' '+u[i]}
const post=(u,b)=>fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b??{})});

// theme
// Dark by default (set in <head>); the toggle remembers the choice.
$('#theme').onclick=async()=>{ const nx=await nsCycleTheme(); $('#theme').title='theme: '+nx+' (click for the next; all of them under Settings → Appearance)'; };
$('#theme').title='theme: '+nsThemeName()+' (click for the next; all of them under Settings → Appearance)';
$('#api').textContent=location.origin+'/v1';
$('#api').onclick=async()=>{try{await navigator.clipboard.writeText(location.origin+'/v1'); $('#api').textContent='copied';
  setTimeout(()=>$('#api').textContent=location.origin+'/v1',1200)}catch{}};

// tabs
function showTab(name){ $$('.tabs button').forEach(x=>x.classList.toggle('pri',x.dataset.tab===name));
  // The sidebar shows which tab is open.
  window.NS_TAB=name; document.dispatchEvent(new CustomEvent('ns-tab',{detail:name}));
  ['chats','local','hub','docs','tools'].forEach(t=>$('#tab-'+t).style.display=t===name?'block':'none'); }
$$('.tabs button').forEach(b=>b.onclick=()=>{ showTab(b.dataset.tab); history.replaceState(null,'','#'+(b.dataset.tab==='local'?'models':b.dataset.tab)); });
// Links from other pages: /#tools, /#models, /#hub, /#docs, /#chats, or
// /#model=<id>&field=classifier to open a model's settings at that field.
const TAB_ALIAS={models:'local',local:'local',chats:'chats',hub:'hub',docs:'docs',tools:'tools'};
function applyHash(){ const h=location.hash.slice(1); if(!h) return;
  if(TAB_ALIAS[h]) return showTab(TAB_ALIAS[h]);
  const q=new URLSearchParams(h), id=q.get('model');
  if(id){ showTab('local'); const i=models.findIndex(m=>m.id===id); if(i<0) return;
    pick(i); const f=q.get('field'); const el=f&&document.getElementById(f);
    if(el){ el.scrollIntoView({block:'center'}); el.focus(); el.style.outline='2px solid var(--acc)'; setTimeout(()=>el.style.outline='',2500); } } }
addEventListener('hashchange',applyHash);

// ---------- models
let linked=[];
async function refresh(){
  models=await (await fetch('/api/models')).json(); modelsLoaded=true;
  try{ linked=await (await fetch('/api/links/models')).json(); }catch{ linked=[]; }
  renderModels(); renderPicker();
  if(chat&&!chat.messages.length) emptyState();
}
function renderPicker(){
  const cur=$('#modelPick').value; const anyOk=models.some(m=>m.stats&&m.stats.eligible);
  let o=`<option value="">Loaded model</option><option value="auto"${anyOk?'':' disabled'}>Auto (by performance stats)${anyOk?'':' - no models have stats'}</option>`;
  o+=models.map(m=>`<option value="${esc(m.id)}">${m.stats&&!m.stats.eligible?'⚠ ':''}${esc(m.id)}${m.stats&&!m.stats.eligible?' (no stats)':''}</option>`).join('');
  if(linked.length) o+=`<optgroup label="linked hosts">`+linked.map(m=>`<option value="${esc(m.id)}">⇢ ${esc(m.id)}</option>`).join('')+`</optgroup>`;
  $('#modelPick').innerHTML=o; if([...$('#modelPick').options].some(x=>x.value===cur&&!x.disabled)) $('#modelPick').value=cur;
  pickChanged();
}
function pickChanged(){
  const m=models.find(x=>x.id===$('#modelPick').value); const w=$('#pickWarn');
  if(m&&m.stats&&!m.stats.eligible){ w.style.display='inline'; w.title=m.stats.why_not+' You picked it manually, so it will be used.'+NOSTATS_HELP; }
  else w.style.display='none';
  $('#attach').disabled=!( status_.vision || (m&&m.mmproj) || $('#modelPick').value==='auto');
}
$('#modelPick').onchange=pickChanged;
function renderModels(){
  const q=$('#filter').value.toLowerCase();
  const rows=models.map((m,i)=>[m,i]).filter(([m])=>!q||m.name.toLowerCase().includes(q)||(m.arch||'').includes(q));
  $('#list').innerHTML = rows.length ? rows.map(([m,i])=>{
    const f=m.fit||{}; const cls=f.ok===false?'bad':(/tight/.test(f.note||'')||f.level==='partial'?'warn':'');
    return `<div class="m${sel&&sel.path===m.path?' sel':''}" data-i="${i}">
      <div class="mn">${esc(m.name)}${statBadge(m)}</div>
      <div class="mm">${m.quant?`<span class="tag">${esc(m.quant)}</span>`:''}${m.arch?`<span class="tag">${esc(m.arch)}</span>`:''}
        ${m.n_experts?`<span class="tag">MoE ${m.n_experts}</span>`:''}${m.mmproj?'<span class="tag">vision</span>':''}
        <span class="tag">${human(m.size)}</span>${fitBadge(f)}</div>
      <div class="mm ${cls}">${esc(f.note||'')}</div>${statLine(m)}</div>`;
  }).join('') : '<div class="note">No .gguf files found under the configured directories.</div>';
  $$('#list .m').forEach(el=>el.onclick=()=>pick(+el.dataset.i));
}
$('#filter').oninput=renderModels;
// LM Studio-style: will it run here, and how.
function fitBadge(f){ const t={gpu:['okc','Full GPU offload'],partial:['warn','Partial GPU offload'],cpu:['','Runs on the CPU'],no:['bad','Likely too large']}[f&&f.level];
  return t?`<span class="tag ${t[0]}" title="${esc(f.note||'')}">${t[1]}</span>`:''; }
function pick(i){
  sel=models[i]; renderModels(); $('#panel').style.display='block'; modelProfiles();
  const s=sel.settings||{};
  F.forEach(k=>{ if($('#'+k)) $('#'+k).value = s[k] ?? ''; });
  $('#moebox').style.display = sel.n_experts?'block':'none';
  if(sel.n_experts) $('#moehint').textContent = `0 = model default (of ${sel.n_experts})`;
  $('#visionrow').style.display = sel.mmproj?'block':'none';
  $('#statsbox').innerHTML = statsTable(sel.stats); renderRadar(); quantBox();
  $('#vision').checked = s.vision!==false; $('#mmname').textContent = sel.mmproj?'('+sel.mmproj.split('/').pop()+')':'';
  const eng = s.engine || (sel.format==='hf'?'vllm':'llama'); $('#engine').value=eng;
  const fa = s.flash_attn===true?'on':s.flash_attn===false?'off':s.flash_attn;
  loadEO(eng, {...s, flash_attn: fa});
}
// Other quantizations of the same model, and which one was measured best on this computer.
async function quantBox(){ const m=sel, box=$('#quantbox'); box.innerHTML=''; if(!m||!m.family) return;
  const sib=models.filter(x=>x.family===m.family).sort((a,b)=>b.size-a.size);
  let ch=null; try{ ch=await (await fetch('/api/quant/choice?family='+encodeURIComponent(m.family))).json(); }catch{}
  if(sel!==m) return;
  const best=ch&&ch.here;
  const chips=sib.map(x=>`<button type="button" data-q="${esc(x.id)}" style="font-size:11.5px;padding:.1rem .45rem;margin:.1rem .15rem 0 0;${x.id===m.id?'border-color:var(--acc)':''}" title="${esc(x.path)}">${esc(x.quant||'?')} · ${human(x.size)}${best&&best.model===x.path?' ★':''}</button>`).join('');
  const jobs='/jobs#kind=quant_select&model='+encodeURIComponent(sib.map(x=>x.path).join(' '))+'&out='+encodeURIComponent('runs/qsel-'+m.family);
  box.innerHTML=`<b>Quantizations</b> ${sib.length>1?chips:''}<div>${best?`★ <b>${esc(best.quant)}</b> was measured the best fit for this computer (${best.speed&&best.speed.gen_tps?best.speed.gen_tps.toFixed(1)+' tok/s':''}${best.fidelity&&best.fidelity.kld!=null?', KLD '+best.fidelity.kld.toFixed(3):best.fidelity&&best.fidelity.agreement!=null?', '+Math.round(best.fidelity.agreement*100)+'% agreement':''}).`:''}
    ${sib.length>1?`<a href="${jobs}">Measure which suits this computer</a>`:`Only ${esc(m.quant||'this file')} is here. <a href="/jobs#kind=quant_select&from=${encodeURIComponent(m.path)}&make=Q8_0,Q6_K,Q4_K_M&out=${encodeURIComponent('runs/qsel-'+m.family)}">Make others and measure</a> (from a high-precision file works best).`}</div>`;
  box.querySelectorAll('[data-q]').forEach(b=>b.onclick=()=>{ const i=models.findIndex(x=>x.id===b.dataset.q); if(i>=0) pick(i); }); }
// The model settings form as values, and back: what Load sends, and what setting profiles keep.
function readModelForm(){ const s={}; F.forEach(k=>{ const el=$('#'+k); if(!el) return; s[k] = TEXT_FIELDS.includes(k) ? el.value : (parseFloat(el.value)||0); });
  if($('#main_gpu').value.trim()==='' || isNaN(parseFloat($('#main_gpu').value))) s.main_gpu=-1; else s.main_gpu=parseInt($('#main_gpu').value);
  s.lora_scale=1.0; s.vision=$('#vision').checked; s.engine=$('#engine').value; Object.assign(s, readEO()); return s; }
function fillModelForm(v){ F.forEach(k=>{ if($('#'+k)&&k in v) $('#'+k).value=v[k]??''; }); if('vision' in v) $('#vision').checked=v.vision!==false;
  const eng=v.engine||$('#engine').value; $('#engine').value=eng; const fa=v.flash_attn===true?'on':v.flash_attn===false?'off':v.flash_attn; loadEO(eng,{...v,flash_attn:fa}); }
const MODEL_DEFAULTS=__MODEL_DEFAULTS__;   // DEFAULTS in studio.py: what a model gets before anyone changes it
let MPROF=null;
function modelProfiles(){ if(MPROF) return; MPROF=nsProfiles($('#mprof'),{scope:'model-settings',get:readModelForm,set:fillModelForm,
  system:()=>fillModelForm({...MODEL_DEFAULTS})}); }
$('#load').onclick=async()=>{
  if(!sel) return;
  const s=readModelForm();
  $('#load').disabled=true; $('#loadmsg').textContent=s.engine==='vllm'?'starting vLLM…':'starting llama-server…';
  const r=await (await post('/api/load',{path:sel.path,settings:s})).json();
  $('#load').disabled=false; sel.settings=s;
  $('#loadmsg').innerHTML = (r.ok?'<span class="okc">loaded</span>':'<span class="bad">'+linkify(esc(r.message||r.error||'failed'))+'</span>')+
    (r.notes&&r.notes.length?'<div class="warn">'+r.notes.map(esc).join('<br>')+'</div>':'')+
    (r.args?`<details><summary class="note">command</summary><code style="word-break:break-all">${esc(r.args.join(' '))}</code></details>`:'');
  status();
};
$('#unload').onclick=async()=>{ await post('/api/unload'); status(); };
async function status(){
  const s=status_=await (await fetch('/api/status')).json();
  $('#dot').className='dot'+(s.running?' on':'');
  $('#st').textContent = s.running ? `${s.id||s.model}${s.vision?' · vision':''} · up ${s.uptime}s${s.idle_ttl?` · TTL ${s.idle_ttl}s`:''}` : 'no model loaded';
  $('#unload').disabled=!s.running; pickChanged(); sideEnv();
  if(s.running!==status.was){ status.was=s.running; if(chat&&!chat.messages.length) emptyState(); }
  const hasLora=!!s.lora; $('#alpha').disabled=!hasLora;
  $('#adesc').textContent = hasLora ? s.lora.split('/').pop() : 'no adapter loaded';
  if(hasLora && document.activeElement!==$('#alpha')){ $('#alpha').value=s.lora_scale; $('#av').textContent=(+s.lora_scale).toFixed(2); }
}
$('#alpha').oninput=async e=>{ const v=parseFloat(e.target.value); $('#av').textContent=v.toFixed(2); await post('/api/lora',{scale:v}); };

// ---------- chats
async function loadChats(){
  const cs=await (await fetch('/api/chats')).json();
  $('#chats').innerHTML = cs.length ? cs.map(c=>`<div class="m${c.id===chat.id?' sel':''}" data-id="${esc(c.id)}">
    <button class="x" data-del="${esc(c.id)}" title="delete">×</button>
    <div class="mn">${esc(c.title||'Untitled')}</div>
    <div class="mm">${c.n} messages · ${new Date(c.updated*1000).toLocaleString()}</div></div>`).join('')
    : '<div class="note">Conversations are saved here automatically.</div>';
  $$('#chats .m').forEach(el=>el.onclick=async e=>{
    if(e.target.dataset.del){ e.stopPropagation(); if(confirm('Delete this chat?')){ await post('/api/chats/delete',{id:e.target.dataset.del});
      if(chat.id===e.target.dataset.del) newChat(); loadChats(); } return; }
    chat=await (await fetch('/api/chats/'+el.dataset.id)).json(); renderChat(); loadChats(); });
}
function newChat(){ chat={id:null,title:'New chat',messages:[]}; renderChat(); loadChats(); }
$('#newchat').onclick=newChat;
async function persist(){
  if(!chat.messages.length) return;
  if(chat.title==='New chat'){ const first=chat.messages.find(m=>m.role==='user'); chat.title=textOf(first?.content).slice(0,60)||'New chat'; }
  chat.model=status_.id||null; chat.preset=$('#preset').value;
  const r=await (await post('/api/chats',chat)).json(); chat.id=r.id; loadChats();
}
const textOf=c=>Array.isArray(c)?c.filter(x=>x.type==='text').map(x=>x.text).join(' '):String(c||'');
const imagesOf=c=>Array.isArray(c)?c.filter(x=>x.type==='image_url').map(x=>x.image_url.url):[];

// minimal, safe markdown: fenced code blocks and inline code only
function render(text){
  const parts=String(text).split(/```/); let out='';
  parts.forEach((p,i)=>{ if(i%2){ const nl=p.indexOf('\n'); out+='<pre>'+esc(nl>=0?p.slice(nl+1):p)+'</pre>'; }
    else out+=esc(p).replace(/`([^`\n]+)`/g,'<code>$1</code>'); });
  return out;
}
function bubble(role,content,extra={}){
  if(!$('#log .msg')) $('#log').innerHTML='';
  const d=document.createElement('div'); d.className='msg';
  d.innerHTML=`<div class="who">${role==='user'?'you':'assistant'}<span class="acts"></span></div>`+
    (extra.think!==undefined?`<details class="think"><summary>reasoning</summary><div></div></details>`:'')+
    `<div class="imgs"></div><div class="tools"></div><div class="body"></div><div class="src"></div><div class="stats"></div><div class="risk"></div><div class="pchk"></div>`;
  d.querySelector('.imgs').innerHTML=imagesOf(content).map(u=>`<img src="${esc(u)}" alt="attached image">`).join('');
  d.querySelector('.body').innerHTML=render(textOf(content));
  if(extra.think) d.querySelector('.think div').textContent=extra.think;
  if(extra.stats) d.querySelector('.stats').textContent=extra.stats;
  if(extra.sources) showSources(d.querySelector('.src'),extra.sources);
  for(const t of extra.tools||[]) toolCard(d.querySelector('.tools'),t);
  if(role==='assistant'){ const a=d.querySelector('.acts');
    a.innerHTML='<button data-a="copy">copy</button><button data-a="regen">regenerate</button><button data-a="check" title="score each token of this reply for hallucination risk">check</button>';
    a.querySelector('[data-a=copy]').onclick=()=>navigator.clipboard.writeText(d.querySelector('.body').textContent);
    a.querySelector('[data-a=regen]').onclick=regenerate;
    a.querySelector('[data-a=check]').onclick=()=>checkReply(d);
    if(PLUGIN_CHECKS.length){ const b=document.createElement('button'); b.textContent='plugins'; b.title='run the checks from your plugins: '+PLUGIN_CHECKS.map(c=>c.title).join(', '); b.onclick=()=>pluginChecks(d,false); a.appendChild(b); }
    if(extra.check) showRisk(d.querySelector('.risk'),extra.check);
    if(extra.plugin_checks) showPluginChecks(d,extra.plugin_checks);
    if(extra.confidence) showConfidence(d,extra.confidence); }
  $('#log').appendChild(d); $('#log').scrollTop=1e9;
  return {root:d, think:d.querySelector('.think div'), thinkBox:d.querySelector('.think'), body:d.querySelector('.body'), stats:d.querySelector('.stats'), src:d.querySelector('.src'), tools:d.querySelector('.tools')};
}
function toolCard(box,t){
  let el=box.querySelector(`[data-k="${t.key}"]`);
  if(!el){ el=document.createElement('div'); el.className='tool'; el.dataset.k=t.key; box.appendChild(el); }
  const args=typeof t.arguments==='string'?t.arguments:JSON.stringify(t.arguments,null,1);
  const state=t.result?(t.result.ok?'<span class="okc">done</span>':'<span class="bad">failed</span>'):(t.needs_approval&&!t.decided?'<span class="warn">waiting for you</span>':'<span>running…</span>');
  // what the call would do (Settings → Permissions): the command line, paths, programs, hosts, each with its answer
  const acts=(t.checks||[]).filter(c=>c.action!=='tool');
  const will=acts.length?`<div class="will">${acts.map(c=>`<div class="wl ${c.effect}"><span class="tag">${esc(c.action)}</span><code>${esc(c.target)}</code><span class="note" title="${esc(c.why)}">${esc(c.effect)}</span></div>`).join('')}</div>`:'';
  const waiting=t.needs_approval&&!t.decided&&!t.result;
  el.innerHTML=`<span class="tn">🔧 ${esc(t.name)}</span> ${state}${waiting?will:''}<details ${waiting&&!acts.length?'open':''}><summary>${!waiting&&acts.length?'what it does, ':''}arguments</summary>${waiting?'':will}<pre>${esc(args)}</pre></details>`+
    (waiting?`<div class="qp"><button class="pri" data-a="1" title="just this time">Allow once</button><button data-q="always" title="this exact call stops asking for this model (Settings → Permissions, Tool approval)">Always allow</button><button class="bad" data-a="0">Deny</button>
      <select data-r title="a broader standing rule (Settings → Tool approval)"><option value="">more…</option>
      <option value="tool">always allow this tool, any arguments</option><option value="args">always allow these exact arguments</option>
      <option value="deny">always deny this tool</option></select></div>`:'')+
    (!t.needs_approval&&t.decision&&t.decision.by!=='default'?`<div class="note">${esc(t.decision.action==='deny'?'blocked':'allowed')} by ${esc(t.decision.reason)} · <a href="/settings#tools">rules</a></div>`:'')+
    (t.result?`<details><summary>result</summary><pre>${esc(t.result.text)}</pre></details>`:'');
  el.querySelectorAll('button[data-a]').forEach(b=>b.onclick=async()=>{ t.decided=true; toolCard(box,t); await post('/api/tools/approve',{key:t.key,allow:b.dataset.a==='1'}); });
  el.querySelectorAll('button[data-q]').forEach(b=>b.onclick=async()=>{ t.decided=true; toolCard(box,t);
    const r=await post('/api/tools/approve',{key:t.key,allow:true,remember:'always'}); if(!r.ok){ try{ alert((await r.json()).error) }catch{} } });
  const rs=el.querySelector('select[data-r]'); if(rs) rs.onchange=async()=>{ const v=rs.value; if(!v) return; t.decided=true; toolCard(box,t);
    const r=await post('/api/tools/approve',{key:t.key,allow:v!=='deny',remember:v}); if(!r.ok){ try{ alert((await r.json()).error) }catch{} } };
}
function showSources(el,src){ el.innerHTML='sources: '+src.map((h,i)=>`<details><summary>[${i+1}] ${esc(h.source)}${h.page?' p.'+h.page:''}</summary><pre>${esc(h.text)}</pre></details>`).join(''); }
// What an empty chat shows: where you are in getting started, and something to ask.
let modelsLoaded=false;
const STARTERS=['Explain how a transformer decides the next word, in plain language.','Write a Python function that checks whether a string is a palindrome.',
  'What were the main causes of the French Revolution?','Plan a three-day trip to Lisbon on a small budget.'];
async function emptyState(){ const el=$('#log');
  el.innerHTML='<div class="empty"><h2>Start a conversation</h2></div>';
  let ready=null; try{ const r=await fetch('api/setup'); if(r.ok) ready=(await r.json()).ready; }catch{}
  if(chat.messages.length) return;
  const nModels=models.length, running=!!(status_&&status_.running);
  const step=(done,title,body)=>`<li class="${done?'done':''}"><span class="ck" aria-hidden="true">${done?'✓':''}</span><div><b>${title}</b><div class="note" style="margin:.1rem 0 0">${body}</div></div></li>`;
  const steps=[
    ready?step(ready.server,'The engine (llama.cpp)',ready.server?'Ready.':'Not found yet. <a href="/setup#build">Setup → Build llama.cpp</a> makes it with one button, or point to an existing llama-server.'):'',
    !modelsLoaded&&!nModels?step(false,'A model','Looking in your model folders…'):step(nModels>0,'A model',nModels?`${nModels} model${nModels>1?'s':''} in <a href="#models">My Models</a>.`:'<a href="#hub">Discover</a> finds and downloads one (each file says whether it fits this computer), or <a href="/setup#paths">add a folder</a> that already has .gguf files.'),
    step(running,'Load it',running?`${esc(status_.id||status_.model)} is loaded.`:'Pick one in <a href="#models">My Models</a>, or just send a message: with the model picker on <i>auto</i>, Studio loads the best one for the question.'),
  ].join('');
  const allSet=ready&&ready.server&&nModels>0&&running;
  const pair=models.slice(0,2).map(m=>`${m.id.split('/').pop().replace(/[^\w.-]/g,'-').slice(0,24)}=${location.origin}/v1@${m.id}`);
  // the most common things people come to do, the ones this person uses most first
  const TASKS=[
    ['Find a model that fits this computer','Each file says whether it fits in memory, and how fast it should run','#hub'],
    ['Compare models on your own questions','A graded TestQA run: accuracy, confident mistakes and declines per subject','/jobs#'+new URLSearchParams({kind:'testqa',endpoint:pair.join(' '),out:'runs/compare.json'}).toString()],
    ['Use these models in another app','Claude Desktop, Cline, or any OpenAI client, pointed at this Studio','/connect'],
    ['See what a reply is made of','Features, emotions and the logit lens over any reply','/features#read'],
    ['Split a project across models','Tasks planned, assigned to the right models, checked and reviewed','/projects'],
    ['Where the work went','Tokens, speed and hardware per device, and predicted speed for every model','/work']];
  const use=typeof nsUsage==='function'?nsUsage():{};
  const used=h=>use[h.split('?')[0].split('#')[0]+(h.startsWith('#')?h:'')]||use[h.split('#')[0]]||0;
  const tasks=TASKS.map((t,i)=>[t,i]).sort((a,b)=>used(b[0][2])-used(a[0][2])||a[1]-b[1]).map(x=>x[0]);
  const tasksHtml=`<div class="note" style="margin-top:1rem">Or:</div><div class="tasks">${tasks.map(([t,d,h])=>`<a class="task" href="${esc(h)}" title="${esc(d)}"><b>${esc(t)}</b><span>${esc(d)}</span></a>`).join('')}</div>`;
  el.innerHTML=`<div class="empty" style="text-align:left;max-width:${allSet?640:560}px"><h2 style="text-align:center">Start a conversation</h2>
    ${allSet?`<div class="note" style="text-align:center">✓ engine · ✓ ${nModels} model${nModels>1?'s':''} · ✓ ${esc(status_.id||status_.model)} loaded</div>`:`<ol class="gs">${steps}</ol>`}
    <div class="note" style="margin-top:1rem">Try one:</div><div class="starters">${STARTERS.map((q,i)=>`<button type="button" data-st="${i}">${esc(q)}</button>`).join('')}</div>
    ${allSet?tasksHtml:`<div class="note" style="margin-top:1rem">Other apps can use these models too: point any OpenAI client at <code>${esc(location.origin)}/v1</code>, or see <a href="/connect">Connect</a>. New here? The <b>?</b> button has a two-minute tour.</div>`}</div>`;
  el.querySelectorAll('[data-st]').forEach(b=>b.onclick=()=>{ $('#in').value=STARTERS[+b.dataset.st]; $('#in').focus(); sideSoon(); }); }
function renderChat(){
  sideSoon(); $('#log').innerHTML='';
  if(!chat.messages.length){ emptyState(); return; }
  for(const m of chat.messages){ const b=bubble(m.role,m.content,{think:m.think,stats:m.stats,sources:m.sources,tools:m.tools,check:m.check,plugin_checks:m.plugin_checks,confidence:m.confidence}); b.root._msg=m; }
}
// ---------- per-reply check: token risk from the activation classifier
function showRisk(el,c){
  if(c.error){ el.innerHTML=`<div class="sum bad">check failed: ${linkify(esc(c.error))}${c.links&&c.links.length?' · '+linkList(c.links):''}</div>`; return; }
  const n=c.flagged.length, T=c.prob.length;
  const verdict=n?`<b class="bad">${n} of ${T} tokens flagged</b>`:`<b class="okc">no tokens flagged</b>`;
  el.innerHTML=`<div class="sum">risk: ${verdict} · peak ${(c.max*100).toFixed(0)}% · mean ${(c.mean*100).toFixed(0)}% `+
    `(flag at ${(c.threshold*100).toFixed(0)}%) · <a href="${esc(c.url)}" target="_blank" rel="noopener">open 3D view ↗</a>`+
    (c.traced_on?` · <span title="where the activations were computed">traced on ${esc(c.traced_on)}</span>`:'')+
    ` · <a href="#" data-a="tog">${n?'hide':'show'} tokens</a>`+
    (c.review?` · <span class="lbl">was it right? <a href="#" data-v="correct">right</a> / <a href="#" data-v="wrong">wrong</a>${c.label?` <b>(${esc(c.label==='correct'?'right':c.label)})</b>`:''}</span>`:'')+
    `</div><div class="rt"${n?'':' hidden'}></div>`;
  // A person's verdict makes the reply count in the Review page's right-vs-wrong statistics.
  el.querySelectorAll('[data-v]').forEach(a=>a.onclick=async e=>{ e.preventDefault();
    const r=await post('/api/review/label',{model:c.review.model,id:c.review.id,verdict:a.dataset.v});
    if(r.ok){ c.label=a.dataset.v; const m=el.closest('.msg')?._msg; if(m&&m.check){ m.check.label=c.label; await persist(); } showRisk(el,c); }
    else alert((await r.json()).error||r.statusText); });
  // Shade each token by its risk; underline the ones over the threshold.
  // Shading starts at half the threshold, so tokens the classifier calls clean stay unshaded.
  el.querySelector('.rt').innerHTML=c.pieces.map((p,i)=>{ const r=c.prob[i]||0, a=Math.max(0,Math.min(1,(r-c.threshold/2)/(c.threshold/2)));
    return `<span class="${r>=c.threshold?'fl':''}" title="risk ${(r*100).toFixed(0)}%" style="background:color-mix(in srgb,var(--no) ${Math.round(a*(r>=c.threshold?50:30))}%,transparent)">${esc(p)}</span>`; }).join('');
  el.querySelector('[data-a=tog]').onclick=e=>{ e.preventDefault(); const rt=el.querySelector('.rt'); rt.hidden=!rt.hidden; e.target.textContent=(rt.hidden?'show':'hide')+' tokens'; };
}
// ---------- plugin checks (Settings -> Plugins)
let PLUGIN_CHECKS=[];
fetch('/api/plugins').then(r=>r.ok?r.json():{checks:[]}).then(j=>{ PLUGIN_CHECKS=j.checks||[]; }).catch(()=>{});
function showPluginChecks(d,res){ const el=d.querySelector('.pchk');
  el.innerHTML=res.map((c,i)=>`<div class="sum">${esc(c.title)}: ${c.error?'<span class="bad">'+esc(c.error)+'</span>':`<b class="${c.flag?'bad':''}">${esc(c.label)}</b>${c.note?' · '+esc(c.note):''}${c.spans&&c.spans.length?` · <a href="#" data-pc="${i}">show</a><div class="rt" hidden></div>`:''}`}</div>`).join('');
  el.querySelectorAll('[data-pc]').forEach(a=>a.onclick=e=>{ e.preventDefault(); const c=res[+a.dataset.pc], box=a.nextElementSibling, text=d._msg?textOf(d._msg.content):d.querySelector('.body').textContent;
    if(box.hidden&&!box.innerHTML){ let h='',i=0; for(const [x,y,n] of c.spans){ if(x<i) continue; h+=esc(text.slice(i,x))+`<mark title="${esc(n)}">${esc(text.slice(x,y))}</mark>`; i=y; } box.innerHTML=h+esc(text.slice(i)); }
    box.hidden=!box.hidden; a.textContent=box.hidden?'show':'hide'; }); }
async function pluginChecks(d,autoOnly){
  const m=d._msg; if(!m) return; const list=PLUGIN_CHECKS.filter(c=>!autoOnly||c.auto); if(!list.length) return;
  const idx=chat.messages.indexOf(m), el=d.querySelector('.pchk'); el.innerHTML='<div class="sum">running plugin checks…</div>';
  const body={model:m.model,messages:chat.messages.slice(0,idx).map(x=>({role:x.role,content:textOf(x.content)})),text:textOf(m.content)};
  const res=await Promise.all(list.map(async c=>{ try{ const r=await post('/api/plugins/check',{...body,plugin:c.plugin,check:c.name}); const j=await r.json(); return r.ok?j:{title:c.title,error:j.error}; }catch(e){ return {title:c.title,error:String(e)}; } }));
  showPluginChecks(d,res); m.plugin_checks=res.filter(x=>!x.error); await persist(); }
// ---------- confidence from the engine's top-5 probabilities per token
// u = entropy of the top 5 plus the rest, over log(6): 0 = certain, 1 = spread evenly.
// A token is uncertain when u >= CONF_U or the chosen token had p < CONF_P.
const CONF_U=0.55, CONF_P=0.35;
function confidenceOf(lp){
  const pieces=[], u=[], p=[];
  for(const t of lp){ const tops=(t.top_logprobs||[]).map(x=>Math.exp(x.logprob)).filter(x=>isFinite(x));
    let sum=0,h=0; for(const q of tops){ sum+=q; if(q>0) h-=q*Math.log(q); }
    const rest=Math.max(0,1-sum); if(rest>1e-9) h-=rest*Math.log(rest);
    pieces.push(t.token??''); u.push(+(h/Math.log(6)).toFixed(3)); p.push(+Math.exp(t.logprob??0).toFixed(3)); }
  const flagged=u.map((x,i)=>x>=CONF_U||p[i]<CONF_P?i:-1).filter(i=>i>=0);
  return {pieces,u,p,flagged,mean:+(u.reduce((a,b)=>a+b,0)/Math.max(1,u.length)).toFixed(3)}; }
function showConfidence(d,c){ let el=d.querySelector('.conf'); if(!el){ el=document.createElement('div'); el.className='conf risk'; d.querySelector('.risk').before(el); }
  const n=c.flagged.length, T=c.u.length, low=Math.min(...c.p);
  el.innerHTML=`<div class="sum">confidence: ${n?`<b class="bad">${n} of ${T} tokens uncertain</b>`:'<b class="okc">no uncertain tokens</b>'} · mean spread ${(c.mean*100).toFixed(0)}% · lowest chosen-token p ${(low*100).toFixed(0)}% · <a href="#" data-a="ctog">${n?'show':'show'} tokens</a></div><div class="rt" hidden></div>`;
  const set=new Set(c.flagged);
  el.querySelector('[data-a=ctog]').onclick=e=>{ e.preventDefault(); const rt=el.querySelector('.rt');
    if(!rt.innerHTML) rt.innerHTML=c.pieces.map((t,i)=>`<span class="${set.has(i)?'fl':''}" title="spread ${(c.u[i]*100).toFixed(0)}% · p ${(c.p[i]*100).toFixed(0)}%" style="background:color-mix(in srgb,var(--warn,#a8701c) ${Math.round(Math.min(1,c.u[i]/CONF_U)*(set.has(i)?45:20))}%,transparent)">${esc(t)}</span>`).join('');
    rt.hidden=!rt.hidden; e.target.textContent=(rt.hidden?'show':'hide')+' tokens'; }; }
async function checkReply(d){
  const m=d._msg; if(!m) return;
  const idx=chat.messages.indexOf(m), el=d.querySelector('.risk');
  el.innerHTML='<div class="sum">checking… (one extra pass over the reply)</div>';
  let c;
  try{ const r=await post('/api/trace',{model:m.model||$('#modelPick').value||undefined,messages:chat.messages.slice(0,idx).map(x=>({role:x.role,content:textOf(x.content)})),text:textOf(m.content)});
    c=await r.json(); if(!r.ok) c={error:c.error||r.statusText,links:c.links}; }catch(e){ c={error:String(e)}; }
  showRisk(el,c);
  if(!c.error){ m.check={id:c.id,pieces:c.pieces,prob:c.prob,flagged:c.flagged,threshold:c.threshold,max:c.max,mean:c.mean,url:c.url,review:c.review,traced_on:c.traced_on}; await persist(); }
}
try{ $('#autoCheck').checked=localStorage.getItem('ns_autocheck')==='1'; }catch{}
$('#autoCheck').onchange=()=>{ try{ localStorage.setItem('ns_autocheck',$('#autoCheck').checked?'1':'0'); }catch{} };
try{ $('#confOn').checked=localStorage.getItem('ns_conf')!=='0'; }catch{ $('#confOn').checked=true; }
$('#confOn').onchange=()=>{ try{ localStorage.setItem('ns_conf',$('#confOn').checked?'1':'0'); }catch{} };

// ---------- images
$('#attach').onclick=()=>$('#file').click();
$('#file').onchange=async()=>{ for(const f of $('#file').files){ if(f.size>20e6){ alert(f.name+' is larger than 20 MB'); continue; }
    pendingImgs.push(await new Promise(r=>{const fr=new FileReader(); fr.onload=()=>r(fr.result); fr.readAsDataURL(f);})); }
  $('#file').value=''; drawPending(); };
function drawPending(){ $('#pending').innerHTML=pendingImgs.map((u,i)=>`<img src="${u}" title="click to remove" data-i="${i}">`).join('');
  $$('#pending img').forEach(im=>im.onclick=()=>{pendingImgs.splice(+im.dataset.i,1); drawPending();}); }

// ---------- send / stream
$('#in').onkeydown=e=>{ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); $('#f').requestSubmit(); } };
$('#f').onsubmit=async e=>{
  e.preventDefault();
  if(busy){ ctrl?.abort(); return; }
  const text=$('#in').value.trim(); if(!text&&!pendingImgs.length) return;
  const content = pendingImgs.length ? [{type:'text',text}, ...pendingImgs.map(u=>({type:'image_url',image_url:{url:u}}))] : text;
  $('#in').value=''; pendingImgs=[]; drawPending();
  chat.messages.push({role:'user',content}); bubble('user',content);
  await stream();
};
async function regenerate(){ if(busy) return; while(chat.messages.length && chat.messages.at(-1).role==='assistant') chat.messages.pop(); renderChat(); await stream(); }
async function stream(){
  busy=true; $('#send').textContent='Stop'; ctrl=new AbortController();
  const out=bubble('assistant','',{think:''}); const lp=[]; let acc='', think='', t0=performance.now(), tFirst=0, usage=null, timings=null, routeNote='', sources=null, tools=[], servedBy=null, compressed=null;
  try{
    const r=await fetch('/api/chat',{method:'POST',signal:ctrl.signal,headers:{'Content-Type':'application/json'},
      body:JSON.stringify({messages:chat.messages.map(m=>({role:m.role,content:m.content})),preset:$('#preset').value,model:$('#modelPick').value,
        rag:$('#ragPick').value?{collection:$('#ragPick').value,k:4}:undefined, tools:$('#toolsOn').checked||undefined,
        confidence:$('#confOn').checked||undefined, ...cmpReq()})});
    if(!r.ok){ let e=await r.text(); try{e=JSON.parse(e).error||e}catch{} e=typeof e==='string'?e:(e.message||JSON.stringify(e)); out.body.innerHTML='error: '+linkify(esc(e)); throw 0; }
    try{ const rt=JSON.parse(r.headers.get('X-NeuronScope-Route')||'null');
      if(rt) servedBy=rt.model;
      if(rt){ routeNote = rt.mode==='auto' ? `auto → ${rt.model}: ${rt.reason}` : `model: ${rt.model}${rt.mode==='manual'?' (chosen manually)':''}`;
        const m=models.find(x=>x.id===rt.model); if(m&&m.stats&&!m.stats.eligible) routeNote+=' ⚠ no performance stats'; } }catch{}
    const rd=r.body.getReader(), dec=new TextDecoder(); let buf='';
    while(true){
      const {done,value}=await rd.read(); if(done) break;
      buf+=dec.decode(value,{stream:true}); let idx;
      while((idx=buf.indexOf('\n'))>=0){
        const line=buf.slice(0,idx).trim(); buf=buf.slice(idx+1);
        if(!line.startsWith('data:')) continue;
        const d=line.slice(5).trim(); if(d==='[DONE]') continue;
        let j; try{ j=JSON.parse(d) }catch{ continue }
        if(j.error){ out.body.innerHTML='error: '+linkify(esc(j.error.message||j.error)); continue; }
        if(j.sources){ sources=j.sources; showSources(out.src,sources); continue; }
        if(j.compressed){ compressed=j.compressed; const h=j.compressed.history;
          if(h&&h.history==='summarized') chat.summary={upto:h.upto,text:h.summary};
          out.stats.textContent=cmpNote(j.compressed); continue; }
        if(j.tool_call){ tools.push(j.tool_call); toolCard(out.tools,j.tool_call); $('#log').scrollTop=1e9; continue; }
        if(j.tool_result){ const t=tools.find(x=>x.key===j.tool_result.key); if(t){ t.result=j.tool_result; t.decided=true; toolCard(out.tools,t); } continue; }
        if(j.usage) usage=j.usage; if(j.timings) timings=j.timings;
        const delta=j.choices?.[0]?.delta||{};
        if((delta.reasoning_content||delta.content) && !tFirst) tFirst=performance.now();
        if(delta.reasoning_content){ think+=delta.reasoning_content; out.think.textContent=think; }
        if(delta.content){ acc+=delta.content; out.body.innerHTML=render(acc); }
        for(const t of j.choices?.[0]?.logprobs?.content||[]) lp.push(t);
        $('#log').scrollTop=1e9;
      }
    }
  }catch(err){ if(err && err.name==='AbortError') out.body.innerHTML=render(acc+'\n[stopped]'); else if(err) out.body.textContent='error: '+err; }
  if(!think) out.thinkBox.remove();
  const secs=(performance.now()-(tFirst||t0))/1000;
  const ntok=usage?.completion_tokens ?? timings?.predicted_n;
  const tps=timings?.predicted_per_second ?? (ntok?ntok/secs:null);
  const stats=[ntok?`${ntok} tokens`:null, tps?`${tps.toFixed(1)} tok/s`:null, tFirst?`first token ${((tFirst-t0)/1000).toFixed(2)}s`:null].filter(Boolean).join(' · ');
  const fullStats=[routeNote,stats,compressed?cmpNote(compressed):''].filter(Boolean).join(' · ');
  out.stats.textContent=fullStats;
  let msg=null;
  const conf=lp.length?confidenceOf(lp):null; if(conf) showConfidence(out.root,conf);
  if(acc||think){ msg={role:'assistant',content:acc,model:servedBy||undefined,think:think||undefined,stats:fullStats,sources:sources||undefined,tools:tools.length?tools:undefined,confidence:conf||undefined,usage:usage||undefined,compressed:compressed||undefined};
    chat.messages.push(msg); out.root._msg=msg; await persist(); }
  status(); refresh();
  busy=false; $('#send').textContent='Send';
  if(msg&&acc&&$('#autoCheck').checked) checkReply(out.root);
  if(msg&&acc) pluginChecks(out.root,true);
  sideSoon();
}
// ---------- side panel: environment, context usage, outputs
const CTX_COL={system:'var(--acc)',documents:'var(--ok)',tools:'var(--warn)',messages:'color-mix(in srgb,var(--fg) 55%,transparent)'};
let sideT=null, ctxLast=null;
function sideShown(){ return !$('#side').hidden; }
function sideSet(open){ $('#side').hidden=!open; $('#ctxStrip').setAttribute('aria-expanded',open);
  $('section.chat').style.setProperty('--strip-h',$('#ctxStrip').offsetHeight+'px'); try{ localStorage.setItem('ns-side',open?'1':'0'); }catch{} }
$('#ctxStrip').onclick=()=>sideSet(!sideShown());
$('#ctxStrip').onkeydown=e=>{ if(e.key==='Enter'||e.key===' '){ e.preventDefault(); sideSet(!sideShown()); } };
document.addEventListener('keydown',e=>{ if(e.key==='Escape'&&sideShown()) sideSet(false); });
function sideSoon(){ clearTimeout(sideT); sideT=setTimeout(()=>{ ctxRefresh(); sideOutputs(); },400); }
function fmtN(n){ return n==null?'?':n>=10000?(n/1000).toFixed(1)+'k':String(n); }
function sideEnv(){ const s=status_||{};
  $('#stripModel').textContent=s.running?(s.id||s.model):'no model loaded'; const m=models.find(x=>x.id===s.id); const st=m?.settings||{};
  $('#envState').innerHTML=s.running?'<span class="okc">● running</span>':'not loaded';
  const row=(k,v)=>v==null||v===''?'':`<dt>${esc(k)}</dt><dd title="${esc(v)}">${esc(v)}</dd>`;
  $('#envKv').innerHTML=row('Model',s.id||s.model||'none')+row('Context',ctxLast?.n_ctx?ctxLast.n_ctx.toLocaleString()+' tokens':(st.ctx?st.ctx.toLocaleString()+' (setting)':null))
    +row('Trained for',ctxLast?.trained_ctx?ctxLast.trained_ctx.toLocaleString():null)+row('GPU layers',s.running?st.ngl:null)+row('Vision',s.running?(s.vision?'yes':'no'):null)
    +row('LoRA',s.lora?s.lora.split(/[\\/]/).pop():null)+row('Workspace',s.workspace)+row('Computer',s.computer)
    +(s.id?`<dt></dt><dd><a href="#model=${encodeURIComponent(s.id)}&field=ctx">Model settings</a></dd>`:''); }
async function ctxRefresh(){
  const draft=$('#in').value.trim(); const msgs=chat.messages.map(m=>({role:m.role,content:m.content})); if(draft) msgs.push({role:'user',content:draft});
  const last=[...chat.messages].reverse().find(m=>m.role==='assistant'&&m.sources);
  const documents=$('#ragPick').value&&last?last.sources.map(h=>h.text).join('\n'):'';
  let c; try{ c=await post('/api/context',{messages:msgs,preset:$('#preset').value,tools:$('#toolsOn').checked||undefined,documents,...cmpReq()}).then(r=>r.json()); }catch{ return; }
  ctxLast=c; const cap=c.n_ctx||(models.find(x=>x.id===(status_||{}).id)?.settings?.ctx)||null;
  const used=c.total, rsv=c.reply_reserve||0, denom=Math.max(cap||0,used+rsv,1);
  $('#ctxBar').innerHTML=c.parts.map(p=>p.tokens?`<span style="width:${100*p.tokens/denom}%;background:${CTX_COL[p.key]}" title="${esc(p.label)}: ${p.tokens}"></span>`:'').join('')
    +(rsv?`<span class="rsv" style="width:${100*Math.min(rsv,Math.max(0,denom-used))/denom}%" title="room kept for the reply: ${rsv}"></span>`:'');
  const pct=cap?Math.round(100*used/cap):null;
  $('#stripBar').innerHTML=$('#ctxBar').innerHTML;
  $('#stripPct').innerHTML=cap?`<span class="${pct>=90?'bad':pct>=75?'warn':''}">${fmtN(used)} / ${fmtN(cap)} tokens · ${pct}%</span>`:`${fmtN(used)} tokens`;
  $('#ctxPct').innerHTML=cap?`<span class="${pct>=90?'bad':pct>=75?'warn':''}">${fmtN(used)} / ${fmtN(cap)} · ${pct}%</span>`:fmtN(used)+' tokens';
  $('#ctxLeg').innerHTML=c.parts.map(p=>`<i style="background:${CTX_COL[p.key]}"></i><span>${esc(p.label)}</span><span class="n">${p.tokens.toLocaleString()}</span>`).join('')
    +(rsv?`<i class="rsv" style="background:repeating-linear-gradient(45deg,var(--line) 0 3px,transparent 3px 6px)"></i><span>Room for the reply</span><span class="n">${rsv.toLocaleString()}</span>`:'');
  const lastU=[...chat.messages].reverse().find(m=>m.usage)?.usage;
  const notes=[c.method==='estimate'?'Estimated at about 4 characters a token; load the model for exact counts.':'Counted with the model\'s tokenizer; the chat template adds a little.',
    lastU?`The last reply used ${lastU.prompt_tokens?.toLocaleString?.()??'?'} prompt + ${lastU.completion_tokens?.toLocaleString?.()??'?'} reply tokens.`:'',
    c.images?`${c.images} image(s) also take context (how much depends on the vision projector).`:'',
    c.summarized?`The first ${c.summarized} messages count as their summary.`:'',
    pct!=null&&used+rsv>cap?(cmpGet().history&&cmpGet().history!=='off'?`<span class="warn">Over the window: the oldest messages will be ${cmpGet().history==='cut'?'left out':'summarized'} on the next send.</span>`:'<b class="bad">The next request will not fit:</b> start a new chat, raise the context in Model settings, or choose what happens below.'):pct>=75?'<span class="warn">Getting full.</span>':''];
  $('#ctxNote').innerHTML=notes.filter(Boolean).join(' '); sideEnv(); }
function sideOutputs(){ const out=[]; let n=0;
  chat.messages.forEach((m,i)=>{ if(m.role!=='assistant') return; n++;
    if(m.check&&m.check.url) out.push(`<a href="${esc(m.check.url)}" target="_blank" rel="noopener">🧠 3D view of reply ${n}<span class="k">${m.check.flagged?m.check.flagged+' flagged':'checked'}</span></a>`);
    if(m.check&&m.check.review) out.push(`<a href="${esc(m.check.review)}">▦ Neuron review, reply ${n}</a>`);
    for(const t of m.tools||[]) out.push(`<div class="o">🔧 ${esc(t.name||t.server+'.'+t.tool||'tool call')}<span class="k">reply ${n}${t.result?.error?' · failed':''}</span></div>`);
    if(m.sources&&m.sources.length) out.push(`<div class="o">📄 ${m.sources.length} source(s): ${esc([...new Set(m.sources.map(h=>h.source))].slice(0,3).join(', '))}<span class="k">reply ${n}</span></div>`);
    if(m.compressed&&m.compressed.history&&m.compressed.history.summary) out.push(`<details class="o" style="display:block"><summary>🗜 Summary of messages 1–${m.compressed.history.upto}<span class="k">reply ${n}</span></summary><div class="note" style="white-space:pre-wrap">${esc(m.compressed.history.summary)}</div></details>`);
    for(const pc of m.plugin_checks||[]) out.push(`<div class="o">🧩 ${esc(pc.title||pc.plugin||'plugin check')}<span class="k">reply ${n}</span></div>`); });
  $('#outN').textContent=out.length||''; $('#stripOut').textContent=out.length?`· ${out.length} output${out.length>1?'s':''}`:'';
  $('#outList').innerHTML=out.length?out.join(''):'<div class="note">Nothing yet. 3D views of checked replies, tool calls, cited sources and plugin checks from this chat collect here.</div>'; }
$('#outExport').onclick=()=>$('#export').click();
// contextual compression (scripts/context_compress.py): passages trimmed to matching sentences; old turns cut or summarized
function cmpGet(){ try{ return JSON.parse(localStorage.getItem('ns-compress')||'{}'); }catch{ return {}; } }
function cmpReq(){ const c=cmpGet(), o={};
  if(c.passages||(c.history&&c.history!=='off')) o.compress={passages:!!c.passages,history:c.history||'off'};
  if(c.history==='summarize'&&chat.summary) o.summary=chat.summary; return o; }
function cmpNote(c){ const p=c.passages, h=c.history, out=[];
  if(p&&p.before) out.push(`passages ${p.before}→${p.after} tokens`);
  if(h&&h.history==='summarized') out.push(`summarized the first ${h.upto} messages`);
  if(h&&h.history==='summary') out.push(`first ${h.upto} messages as a summary`);
  if(h&&h.history==='cut') out.push(`left out the first ${h.dropped} messages`);
  if(h&&h.history==='failed') out.push('compression failed: '+h.error);
  return out.join(' · '); }
{ const c=cmpGet(); $('#cmpPass').checked=!!c.passages; $('#cmpHist').value=c.history||'off'; }
['#cmpPass','#cmpHist'].forEach(s=>$(s).addEventListener('change',()=>{ try{ localStorage.setItem('ns-compress',JSON.stringify({passages:$('#cmpPass').checked,history:$('#cmpHist').value})); }catch{} sideSoon(); }));
$('#in').addEventListener('input',sideSoon); ['#preset','#ragPick','#toolsOn'].forEach(s=>$(s).addEventListener('change',sideSoon));
try{ if(localStorage.getItem('ns-side')==='1') setTimeout(()=>sideSet(true),0); }catch{}
sideSoon();
$('#export').onclick=()=>{
  const md=`# ${chat.title}\n\n`+chat.messages.map(m=>`**${m.role}**\n\n${textOf(m.content)}${imagesOf(m.content).length?`\n\n_[${imagesOf(m.content).length} image(s)]_`:''}\n`).join('\n');
  const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([md],{type:'text/markdown'}));
  a.download=(chat.title||'chat').replace(/[^\w -]+/g,'_').slice(0,50)+'.md'; a.click(); URL.revokeObjectURL(a.href);
};

// ---------- hub
$('#q').onkeydown=e=>{ if(e.key==='Enter'){ e.preventDefault(); $('#go').click(); } };
$$('[data-qs]').forEach(a=>a.onclick=e=>{ e.preventDefault(); $('#q').value=a.dataset.qs; $('#go').click(); });
$('#go').onclick=async()=>{
  const q=$('#q').value.trim(); if(!q) return;
  $('#hits').textContent='searching…';
  const r=await (await fetch('/api/search?q='+encodeURIComponent(q))).json();
  if(r.error){ $('#hits').innerHTML='<span class="bad">'+esc(r.error)+'</span>'; return; }
  $('#hits').innerHTML=r.map(m=>`<div class="m" data-repo="${esc(m.repo)}"><div class="mn">${esc(m.repo)}</div>
    <div class="mm">${m.downloads.toLocaleString()} downloads ${m.gated?'<span class="tag warn">gated</span>':''}</div></div>`).join('');
  $$('#hits .m').forEach(el=>el.onclick=()=>files(el.dataset.repo));
};
async function files(repo){
  $('#hits').innerHTML='<div class="note">loading files…</div>';
  const r=await (await fetch('/api/files?repo='+encodeURIComponent(repo))).json();
  if(r.error){ $('#hits').innerHTML='<span class="bad">'+esc(r.error)+'</span>'; return; }
  $('#hits').innerHTML=`<div class="note"><b>${esc(repo)}</b> · click a file to download</div>`+r.map((g,i)=>
    `<div class="m" data-i="${i}"><div class="mn">${esc(g.name)}</div>
     <div class="mm">${g.recommended?'<span class="tag okc" title="the largest file that runs fully on this computer\'s GPU (or in RAM)">★ Recommended</span>':''}${g.quant?`<span class="tag">${esc(g.quant)}</span>`:''}${g.size?human(g.size):'size unknown'}
     ${g.parts.length>1?`<span class="tag">${g.parts.length} parts</span>`:''}${fitBadge(g.fit)}</div>${g.fit&&g.fit.note?`<div class="mm ${g.fit.ok===false?'bad':g.fit.level==='partial'?'warn':''}">${esc(g.fit.note)}</div>`:''}</div>`).join('');
  $$('#hits .m[data-i]').forEach(el=>el.onclick=async()=>{ await post('/api/download',{repo,group:r[+el.dataset.i]}); jobs(); });
}
async function jobs(){
  const js=await (await fetch('/api/downloads')).json();
  $('#jobs').innerHTML = js.map(j=>{ const pct=j.total?Math.min(100,100*j.done/j.total):0;
    return `<div class="m"><div class="mn">${esc(j.name)}</div>
      <div class="mm">${esc(j.state)}${j.error?': <span class="bad">'+esc(j.error)+'</span>':''}${j.total?` · ${pct.toFixed(1)}% of ${human(j.total)}`:''}</div>
      <div style="height:4px;background:var(--line);border-radius:2px;margin-top:.3rem"><div style="height:4px;width:${pct}%;background:var(--acc);border-radius:2px"></div></div>
      ${j.state==='running'?`<button data-c="${esc(j.id)}" style="margin-top:.35rem;font-size:11px;padding:.15rem .5rem">Cancel</button>`:''}</div>`;
  }).join('');
  $$('#jobs button[data-c]').forEach(b=>b.onclick=async()=>{ await post('/api/download/cancel',{id:b.dataset.c}); jobs(); });
  if(js.some(j=>j.state==='running')) setTimeout(jobs,1200); else if(js.some(j=>j.state==='done')) refresh();
}
async function presets(){
  const p=await (await fetch('/api/presets')).json();
  $('#preset').innerHTML=Object.keys(p).map(k=>`<option>${esc(k)}</option>`).join('');
}
// ---------- docs (RAG)
let curColl=null;
async function colls(){
  const r=await (await fetch('/api/rag')).json(), cs=r.collections||[];
  const keep=$('#ragPick').value;
  $('#ragPick').innerHTML='<option value="">none</option>'+cs.map(c=>`<option ${c.name===keep?'selected':''}>${esc(c.name)}</option>`).join('');
  $('#colls').innerHTML=cs.length?cs.map(c=>`<div class="m" data-c="${esc(c.name)}"><div class="mn">${esc(c.name)}</div><div class="mm">${c.docs} files · ${c.chunks} passages${c.dense?' · <span class="tag">dense</span>':''}</div></div>`).join(''):'<div class="note">No collections yet.</div>';
  $$('#colls .m').forEach(el=>el.onclick=()=>openColl(el.dataset.c));
  return r;
}
async function openColl(name){ curColl=name; $('#collpanel').style.display=''; $('#collname').textContent=name; $('#raghits').textContent='';
  const ds=await (await fetch('/api/rag/docs?c='+encodeURIComponent(name))).json();
  $('#docs').innerHTML=(ds.length?ds:[]).map(d=>`<div class="m"><div class="mn">${esc(d.source)}</div><div class="mm">${d.chunks} passages</div><button class="x" data-d="${esc(d.doc)}" title="remove">×</button></div>`).join('')||'<div class="note">empty</div>';
  $$('#docs button[data-d]').forEach(b=>b.onclick=async()=>{ await post('/api/rag/delete',{collection:name,doc:b.dataset.d}); openColl(name); colls(); }); }
$('#mkcoll').onclick=async()=>{ const n=$('#newcoll').value.trim(); if(!n) return;
  const r=await post('/api/rag/create',{collection:n}); if(!r.ok){ alert((await r.json()).error); return; } $('#newcoll').value=''; await colls(); openColl(n); };
$('#adddocs').onclick=()=>$('#docfile').click();
$('#docfile').onchange=async()=>{ const fs=[...$('#docfile').files]; $('#docfile').value='';
  for(const f of fs){ if(f.size>32e6){ $('#docmsg').textContent=f.name+' is larger than 32 MB'; continue; }
    $('#docmsg').textContent='indexing '+f.name+'…';
    const data=await new Promise(r=>{const fr=new FileReader(); fr.onload=()=>r(fr.result); fr.readAsDataURL(f);});
    const r=await post('/api/rag/upload',{collection:curColl,filename:f.name,data}); const j=await r.json();
    $('#docmsg').textContent=r.ok?`${f.name}: ${j.chunks} passages`:`${f.name}: ${j.error}`; }
  openColl(curColl); colls(); };
$('#ragq').onkeydown=async e=>{ if(e.key!=='Enter') return; const r=await post('/api/rag/search',{collection:curColl,query:$('#ragq').value,k:4}); const j=await r.json();
  $('#raghits').innerHTML=r.ok?(j.map((h,i)=>`<details><summary>[${i+1}] ${esc(h.source)}${h.page?' p.'+h.page:''}</summary><pre style="white-space:pre-wrap">${esc(h.text)}</pre></details>`).join('')||'no match'):esc(j.error); };
$('#dropcoll').onclick=async()=>{ if(!confirm('Delete collection '+curColl+'?')) return; await post('/api/rag/delete',{collection:curColl}); $('#collpanel').style.display='none'; colls(); };

// ---------- GPUs
async function gpus(){ try{ const r=await (await fetch('/api/gpus')).json(); if(!r.nvidia||!r.count) return;
  $('#gpubox').style.display='';
  $('#gpuhint').textContent=`${r.count} × ${r.gpus[0].name}, ${(r.total_vram/2**30).toFixed(0)} GiB`+(r.multi_gpu?` · suggested: ${r.multi_gpu.llama_server}`:'');
  $('#gpuhint').title=(r.multi_gpu?r.multi_gpu.why+'\n\n':'')+(r.notes||[]).join('\n');
}catch{} }

// ---------- MCP tools
async function mcp(reload){
  const r=reload?await (await post('/api/mcp/reload',{})).json():await (await fetch('/api/mcp')).json();
  if(r.error){ $('#mcplist').innerHTML='<span class="bad">'+esc(r.error)+'</span>'; return; }
  $('#mcpcfg').innerHTML=(r.config?'config: <code>'+esc(r.config)+'</code> · ':'No connectors yet. ')+'<a href="/settings#connectors">Manage connectors</a>';
  const sv=r.servers||[]; const n=sv.reduce((a,x)=>a+(x.connected?x.tools.length:0),0);
  $('#toolsBox').style.display=n?'':'none';
  $('#mcplist').innerHTML=sv.map(x=>`<div class="m"><div class="mn">${esc(x.name)} <span class="tag">${esc(x.transport)}</span>${x.disabled?'<span class="tag">disabled</span>':x.connected?'<span class="tag okc">connected</span>':'<span class="tag bad">down</span>'}${x.auto_approve===true?'<span class="tag warn">auto-approve all</span>':''}${x.disabled?'':` <button class="x" style="width:auto;padding:0 .4rem" data-mtest="${esc(x.name)}" title="list its tools again, and call one read-only tool that needs no arguments">Test</button>`}</div>
    <div class="mm" data-mres="${esc(x.name)}"></div><div class="mm">${x.error?'<span class="bad">'+esc(x.error)+'</span>':x.tools.map(t=>`<div title="${esc(t.description)}">${esc(t.name)}${Array.isArray(x.auto_approve)&&x.auto_approve.includes(t.name)?' <span class="tag">auto</span>':''}</div>`).join('')}</div></div>`).join('');
  document.querySelectorAll('[data-mtest]').forEach(b=>b.onclick=async()=>{ const out=document.querySelector(`[data-mres="${b.dataset.mtest}"]`); out.textContent='testing…';
    try{ const j=await (await post('/api/mcp/test',{server:b.dataset.mtest})).json();
      out.innerHTML=j.ok?`<span class="okc">answered</span> in ${j.seconds}s · ${j.tools} tools`+(j.called?` · ${esc(j.called)}: ${j.call_ok?'<span class="okc">ok</span>':'<span class="bad">failed</span>'} <code>${esc((j.result||'').slice(0,160))}</code>`:' · no read-only tool without arguments to call')
        :`<span class="bad">${esc(j.error||'failed')}</span>`; }catch(e){ out.innerHTML=`<span class="bad">${esc(String(e))}</span>`; } });
}
$('#mcpreload').onclick=()=>mcp(true);

refresh(); if(TAB_ALIAS[location.hash.slice(1)]) applyHash(); (function hashWhenReady(n){ if(models.length||n>40) applyHash(); else setTimeout(()=>hashWhenReady(n+1),150); })(0); status(); presets(); jobs(); loadChats(); renderChat(); colls(); mcp(); gpus(); setInterval(status,4000);
// First run: point at Setup while llama-server or models are missing (owner only; devices get 403).
fetch('/api/setup').then(r=>r.ok?r.json():null).then(s=>{ if(!s||(s.ready.server&&s.ready.models)) return;
  const miss=[!s.ready.server&&'llama-server',!s.ready.models&&'models'].filter(Boolean).join(' and ');
  const d=document.createElement('div'); d.id='setupBanner';
  d.style.cssText='padding:.55rem 1rem;background:color-mix(in srgb,var(--acc) 18%,var(--panel));border-bottom:1px solid var(--line);font-size:13px';
  d.innerHTML=`Almost there: Studio needs ${miss}. <a href="/setup" style="color:var(--acc);font-weight:600">Open Setup →</a>`;
  document.querySelector('header').after(d); }).catch(()=>{});
</script>"""
from ui_links import inject as _inject_links  # noqa: E402
PAGE = _inject_links(PAGE).replace("__MODEL_DEFAULTS__", json.dumps(DEFAULTS))
LINK_PAGE = _inject_links(LINK_PAGE)



LOGIN = """<!DOCTYPE html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniCortex Studio</title>
<style>:root{color-scheme:dark}body{font:14px ui-sans-serif,system-ui,sans-serif;background:#111316;color:#e6e8eb;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
form{background:#181b20;border:1px solid #2a2f37;border-radius:10px;padding:1.6rem;width:320px}
h1{font-size:15px;margin:0 0 1rem}input{width:100%;padding:.5rem;border:1px solid #2a2f37;background:#20242b;color:#e6e8eb;
border-radius:6px;font:13px ui-monospace,monospace}
button{width:100%;margin-top:.7rem;padding:.5rem;border:1px solid #6ea8e0;background:#6ea8e0;
color:#0d1117;border-radius:6px;cursor:pointer;font:500 13px ui-sans-serif,system-ui}
.e{color:#ff8b7e;font-size:13px;margin-top:.5rem;min-height:1em}</style>
<form id="f"><h1>OmniCortex Studio</h1>
<input id="t" type="password" placeholder="access token" autofocus>
<button>Unlock</button><div class="e" id="e"></div></form>
<script>document.getElementById('f').onsubmit=async e=>{e.preventDefault();
const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({token:document.getElementById('t').value})});
if(r.ok) location.reload(); else document.getElementById('e').textContent='Incorrect token.';};
</script>"""

def build_parser():
    p = argparse.ArgumentParser(description="OmniCortex Studio: model manager, chat and OpenAI-compatible server")
    p.add_argument("--models-dir", action="append", default=[],
                   help="repeatable; searched recursively for .gguf")
    p.add_argument("--server", help="path to llama-server (or $NS_LLAMA_SERVER)")
    p.add_argument("--vllm", default=os.environ.get("NS_VLLM", "vllm"),
                   help="the vLLM command, for models served with the vLLM engine (default $NS_VLLM or vllm)")
    p.add_argument("--port", type=int, default=7870)
    p.add_argument("--host", default="127.0.0.1",
                   help="non-loopback binds require a token, and TLS unless --allow-plaintext")
    p.add_argument("--backend-port", type=int, default=8080, help="local port for llama-server")
    p.add_argument("--download-dir",
                   help="where hub downloads land (default: first --models-dir)")
    p.add_argument("--settings",
                   default=os.path.expanduser("~/.neuronscope/studio.json"))
    p.add_argument("--chats-dir", default=os.path.expanduser("~/.neuronscope/chats"))
    p.add_argument("--stats-dir", default=str(model_stats.DEFAULT_DIR),
                   help="rolling per-model stats (shared with testqa.py --record-stats)")
    p.add_argument("--stats-window", type=int, default=500, help="most recent observations per kind that count")
    p.add_argument("--min-graded", type=int, default=20,
                   help="graded results a model needs before model \"auto\" will consider it")
    p.add_argument("--hallucination-cost", type=float, default=1.0,
                   help="how much a wrong answer costs relative to a right one when auto ranks models")
    p.add_argument("--cett", default=os.environ.get("NS_CETT"),
                   help="llama-cett-dump binary; enables activation scoring for models with a classifier set")
    p.add_argument("--score-ngl", type=int, default=0,
                   help="GPU layers for activation scoring (0 keeps it off the GPU llama-server is using)")
    p.add_argument("--score-every", type=int, default=1, help="score one reply in N")
    p.add_argument("--projects-dir", default=os.path.expanduser("~/.neuronscope/projects"),
                   help="director projects: plans, results and history (/projects)")
    p.add_argument("--max-workers", type=int, default=2,
                   help="worker models the director may run at once, besides the chat model")
    p.add_argument("--worker-idle", type=int, default=300, help="stop a worker model after N idle seconds")
    p.add_argument("--hardware", default=os.path.expanduser("~/.neuronscope/hardware.json"),
                   help="accelerators for worker models: llama-server per backend (rocm, vulkan, cuda, metal, "
                        "cpu), per-device settings, devices to skip (see scripts/accelerators.py)")
    p.add_argument("--no-work-log", action="store_true",
                   help="do not record who did what work and hardware samples (/work)")
    p.add_argument("--graphs-dir", default=os.path.expanduser("~/.neuronscope/graphs"),
                   help="node graphs and their runs (/graphs)")
    p.add_argument("--traces-dir", default=os.path.expanduser("~/.neuronscope/traces"),
                   help="where per-reply checks are saved (open them in the 3D view or timeline.py)")
    p.add_argument("--review-dir", default=os.path.expanduser("~/.neuronscope/review"),
                   help="neuron review store: every checked reply's activations, for the /review page")
    p.add_argument("--idle-ttl", type=int, default=0, help="unload the model after N idle seconds (0 = never)")
    p.add_argument("--no-jit", action="store_true", help="/v1 requests never load or swap models")
    p.add_argument("--jobs-dir", default=os.path.expanduser("~/.neuronscope/jobs"),
                   help="logs and state of evaluation/retraining jobs started from /jobs")
    p.add_argument("--max-jobs", type=int, default=2, help="jobs that may run at once")
    p.add_argument("--allow-remote-jobs", action="store_true",
                   help="allow /jobs on a non-loopback bind (jobs can train models and run model-written code)")
    p.add_argument("--no-jobs", action="store_true", help="disable /jobs entirely")
    p.add_argument("--devices", default=os.path.expanduser("~/.neuronscope/devices.json"),
                   help="paired devices (token hashes only)")
    p.add_argument("--pair-code-ttl", default="5m",
                   help="how long a pairing link can be claimed (e.g. 90s, 5m, 1h; max 1d)")
    p.add_argument("--pair-durations", default="1h,8h,1d,7d,30d",
                   help="temporary-access durations offered when pairing")
    p.add_argument("--pair-max", default="90d", help="longest temporary access a pairing may grant")
    p.add_argument("--pair-default", default="persistent",
                   help="what a pairing grants unless chosen otherwise: persistent, or a duration like 8h")
    p.add_argument("--no-persistent-pairing", action="store_true",
                   help="only grant temporary access; every paired device expires")
    p.add_argument("--links", default=os.path.expanduser("~/.neuronscope/links.json"),
                   help="linked hosts whose models this Studio serves (holds their device tokens; 0600)")
    p.add_argument("--mcp-config", default=str(ns_mcp.DEFAULT_CONFIG),
                   help="MCP servers for chat tools (mcpServers JSON, as in LM Studio / Claude Desktop)")
    p.add_argument("--rag-dir", default=os.path.expanduser("~/.neuronscope/rag"), help="document collections")
    p.add_argument("--rag-embed", help="URL[@model] of an OpenAI-compatible embeddings endpoint for dense retrieval")
    p.add_argument("--rag-embed-gguf", help="embedding GGUF; Studio runs it with llama-server --embedding on demand")
    p.add_argument("--rag-embed-ngl", type=int, default=0, help="GPU layers for the embedding model")
    sec.add_server_security_args(p)
    p.add_argument("--config", default=os.environ.get("NS_CONFIG", "~/.neuronscope/config.json"),
                   help="settings file (\"studio\" section); the Setup page writes it, and flags override it")
    p.add_argument("--workspace", help="which workspace's folders to use (Settings -> Workspaces; "
                                       "default: the active one)")
    return p


def main(argv=None):
    p = build_parser()
    pre, _ = p.parse_known_args(argv)
    STATE["config_path"] = pre.config
    os.environ["NS_CONFIG"] = os.path.expanduser(pre.config)
    import ns_workspace
    # A workspace chosen for this run (flag or variable) wins over the file's choice.
    STATE["workspace_forced"] = pre.workspace or os.environ.get("NS_WORKSPACE")
    if pre.workspace:
        if pre.workspace not in ns_workspace.names():
            print(f"error: no workspace {pre.workspace!r}; have {', '.join(ns_workspace.names())}", file=sys.stderr)
            return 2
        os.environ["NS_WORKSPACE"] = pre.workspace
    ws = ns_workspace.active()
    # Where things are comes from the workspace; the "studio" section holds behaviour.
    # Precedence: flags, then the studio section, then the workspace.
    WS_FLAGS = {"models_dir", "server", "cett", "download_dir", "chats_dir", "traces_dir", "review_dir",
                "projects_dir", "jobs_dir", "stats_dir", "rag_dir", "graphs_dir"}
    builtin = {d: p.get_default(d) for d in WS_FLAGS}
    cfg = load_studio_config(pre.config)
    dests = {x.dest for x in p._actions}
    p.set_defaults(**{k: (list(v) if isinstance(v, list) else v) for k, v in cfg.items()
                      if k in dests and k not in WS_FLAGS})
    a = p.parse_args(argv)
    for d in WS_FLAGS - {"models_dir", "server", "cett", "download_dir"}:
        if getattr(a, d) == builtin[d]:
            setattr(a, d, ws[d])
    STATE["workspace_name"], STATE["workspace"] = ws["name"], ws
    STATE["subject_models"] = dict(cfg.get("subject_models") or {})
    STATE["addons_off"] = list(cfg.get("addons_off") or [])
    STATE["addon_repos"] = list(cfg.get("addon_repos") or [])
    STATE["studio_theme"] = cfg.get("theme") if cfg.get("theme") in studio_themes() else "midnight"
    STATE["viz_theme"] = (studio_themes().get(STATE["studio_theme"]) or {}).get("viz") or "dark"
    try:
        import widgets as _w
        STATE["widgets"] = _w.validate(cfg.get("widgets") or {})
    except ValueError:
        STATE["widgets"] = {}
    STATE["tool_rules_path"] = cfg.get("tool_rules") or None
    STATE["permissions_path"] = cfg.get("permissions") or None
    os.environ.update(ns_workspace.env_for(ws))

    STATE["models_dirs"] = [os.path.expanduser(d) for d in a.models_dir] or ws["models_dirs"]
    a.server = a.server or (ws["server"] if os.path.isfile(ws["server"]) else None)
    a.cett = a.cett or (ws["cett"] if os.path.isfile(ws["cett"]) else None)
    a.download_dir = a.download_dir or ws["download_dir"]
    STATE["server_bin"] = a.server or os.environ.get("NS_LLAMA_SERVER")
    STATE["vllm_bin"] = a.vllm
    STATE["settings_path"] = os.path.expanduser(a.settings)
    STATE["chats_dir"] = os.path.expanduser(a.chats_dir)
    STATE["traces_dir"] = os.path.expanduser(a.traces_dir)
    STATE["review_dir"], STATE["review"] = os.path.expanduser(a.review_dir), None
    STATE.update(max_workers=max(1, a.max_workers), worker_idle=max(30, a.worker_idle),
                 hardware_path=os.path.expanduser(a.hardware))
    import director
    DIRECTOR["store"] = director.ProjectStore(os.path.expanduser(a.projects_dir))
    STATE["graphs_dir"] = os.path.expanduser(a.graphs_dir)
    os.environ.setdefault(sec.TOKEN_ENV, os.environ.get("NS_STUDIO_TOKEN", ""))
    STATE["token"] = sec.resolve_token(a.token, a.token_file) or None
    STATE["download_dir"] = (os.path.expanduser(a.download_dir)
                             if a.download_dir else None)
    STATE["idle_ttl"] = max(0, a.idle_ttl)
    STATE["jit"] = not a.no_jit
    STATE["backend_port"] = a.backend_port
    STATE["stats"] = model_stats.StatsStore(a.stats_dir, a.stats_window)
    STATE.update(min_graded=a.min_graded, halluc_cost=a.hallucination_cost, score_ngl=a.score_ngl,
                 score_every=a.score_every,
                 cett=a.cett if a.cett and os.path.exists(a.cett) else None)
    tls = bool(a.tls_cert and a.tls_key)
    STATE["tls"] = tls
    STATE["bind_host"] = a.host
    try:
        for w in sec.check_bind(a.host, STATE["token"] or "", tls=tls,
                                allow_plaintext=a.allow_plaintext):
            print("warning:", w)
    except sec.SecurityConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        policy = ns_pairing.PairingPolicy(a.pair_code_ttl, a.pair_durations.split(","), a.pair_max,
                                          not a.no_persistent_pairing, a.pair_default)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    STATE.update(devices=ns_pairing.DeviceRegistry(a.devices, policy), links_path=os.path.expanduser(a.links),
                 fingerprint=ns_pairing.cert_fingerprint(a.tls_cert) if a.tls_cert and a.tls_key else None)
    STATE.update(mcp_config=a.mcp_config if os.path.exists(os.path.expanduser(a.mcp_config)) else None, mcp=None,
                 mcp_config_path=os.path.expanduser(a.mcp_config))
    STATE.update(rag_dir=a.rag_dir, rag_embed=a.rag_embed, rag_embed_gguf=a.rag_embed_gguf,
                 rag_embed_ngl=a.rag_embed_ngl, rag=None)
    if a.no_jobs:
        STATE["jobs"], STATE["jobs_off"] = None, "jobs are disabled (--no-jobs)"
    elif not sec.is_loopback(a.host) and not a.allow_remote_jobs:
        STATE["jobs"] = None
        STATE["jobs_off"] = ("jobs are off on a network bind: they can train models and run model-written code. "
                             "Restart Studio with --allow-remote-jobs to enable them.")
    else:
        STATE["jobs"] = ns_jobs.JobRunner(a.jobs_dir, a.max_jobs, cwd=ws["root"])
        STATE["jobs"].on_finish = job_finished

    scheme = "https" if tls else "http"
    print(f"OmniCortex Studio on {scheme}://{a.host}:{a.port}")
    notes = ["JIT " + ("on" if STATE["jit"] else "off")]
    if STATE["idle_ttl"]:
        notes.append(f"idle TTL {STATE['idle_ttl']}s")
    notes.append(f'model "auto" over models with >= {STATE["min_graded"]} graded results')
    if STATE["cett"]:
        notes.append("activation scoring available")
    print(f"OpenAI-compatible API: {scheme}://{a.host}:{a.port}/v1  ({', '.join(notes)})")
    print(f"workspace: {ws['name']}  (root {ws['root']})")
    print("model dirs:")
    for d in STATE["models_dirs"]:
        print(f"  {d}{'' if os.path.isdir(d) else '   (missing)'}")
    if not any(os.path.isdir(d) for d in STATE["models_dirs"]):
        print(f"  set your models folder: {scheme}://{a.host}:{a.port}/settings#workspaces, or\n"
              f"  python scripts/ns_workspace.py set {ws['name']} models_dirs /path/to/models")
    try:
        found = [m["id"] for m in scan_models()]
    except Exception:
        found = []
    if found:
        # These ids are what API clients, TestQA (--endpoint x=URL/v1@<id>) and Cline put in "model".
        print(f"models ({len(found)}): " + ", ".join(found[:8]) + (" …" if len(found) > 8 else ""))
    else:
        print("no .gguf models found yet: download some from the Models tab, or set the folder under Settings")
    if not STATE["server_bin"]:
        print("\nno --server given: models can be listed but not loaded")
    elif not os.path.exists(STATE["server_bin"]):
        print(f"\n!! {STATE['server_bin']} does not exist")
    if STATE["token"]:
        print("authentication enabled")

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    STATE["port"] = srv.server_address[1]
    if tls:
        srv.socket = sec.server_ssl_context(a.tls_cert, a.tls_key).wrap_socket(
            srv.socket, server_side=True, do_handshake_on_connect=False)
    threading.Thread(target=idle_reaper, daemon=True).start()
    threading.Thread(target=score_worker, daemon=True).start()
    STATE["work_log_off"] = a.no_work_log or os.environ.get("NS_WORK_LOG") == "off"
    threading.Thread(target=_hw_sampler, daemon=True).start()
    recover_projects()
    threading.Thread(target=director_loop, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_server()
        if REMOTE["obj"] is not None:
            REMOTE["obj"].tunnels.stop_all()
        POOL.stop()
        stop_labs()
        if EMBED["proc"] is not None and EMBED["proc"].poll() is None:
            EMBED["proc"].terminate()
        if STATE.get("mcp") is not None:
            STATE["mcp"].close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
