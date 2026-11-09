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

Binding beyond loopback requires --token (or OC_STUDIO_TOKEN), and TLS unless
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
import oc_security as sec
import model_stats
import jobs as oc_jobs
import rag as oc_rag
import mcp_client as oc_mcp
import context_compress as oc_compress
import quant_select as oc_qsel
import oc_pairing
import work_log as oc_work
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
    req = urllib.request.Request(url, headers={"User-Agent": "omnicortex"})
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
    req = urllib.request.Request(url, headers={"User-Agent": "omnicortex"})
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
            "gpus": "", "split_mode": "", "tensor_split": "", "main_gpu": -1,
            # Layer streaming (docs/INTEGRATED_OPTIONS.md): llama.cpp rpc-servers on other computers that hold
            # and run a share of the layers ("host:50052,host2:50052"); -ts then splits across them too.
            "rpc": ""}

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
    """Start llama-server; when it cannot load for lack of memory or a driver fault, walk the backend hot-swap
    ladder (scripts/backend_swap.py): fewer GPU layers, the alternate servers you listed, then CPU only."""
    ok, why = _start_server_once(model, settings, port)
    hs = options()["hot_swap"]
    if ok or not hs["enabled"] or (settings.get("engine") or "llama") != "llama":
        PROC["fallback"] = None
        return ok, why
    import backend_swap
    if not backend_swap.is_resource_failure(why):
        return ok, why
    def brief(w):
        lines = [x.strip() for x in str(w).splitlines() if x.strip()]
        return (lines[0] + (f": {lines[-1]}" if len(lines) > 1 else ""))[:240] if lines else ""
    tried = [brief(why)]
    for alt, srv, label in backend_swap.ladder(settings, hs["servers"], STATE.get("server_bin")):
        if srv and not os.path.exists(srv):
            continue
        ok2, why2 = _start_server_once(model, {**alt, **({"_server_bin": srv} if srv else {})}, port)
        if ok2:
            PROC["fallback"] = {"setup": label, "after": tried, "settings": {k: alt.get(k) for k in ("ngl", "split_mode")}, "server": srv}
            print(f"[studio] backend hot-swap: {model['id']} loaded on {label} after: {tried[0]}", file=sys.stderr)
            return True, f"ready (fallback: {label})"
        tried.append(f"{label}: {brief(why2)}")
    return False, why + "\nbackend hot-swap tried: " + "; ".join(tried[1:])


def _start_server_once(model, settings, port):
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
        cmd = [settings.get("_server_bin") or STATE["server_bin"], "-m", model["path"],
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
        rpc = str(settings.get("rpc") or "").replace(" ", "")
        if rpc:
            if not re.fullmatch(r"[\w.\-\[\]:]+:\d{2,5}(,[\w.\-\[\]:]+:\d{2,5})*", rpc):
                raise ValueError("rpc servers look like host:50052,other-host:50052")
            cmd += ["--rpc", rpc]
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

import automation as oc_auto  # noqa: E402

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
    from omnicortex_remote import presence as oc_presence
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
                SYSMON["presence"]["local"] = oc_presence.snapshot()
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
    return os.path.join(os.path.dirname(os.path.expanduser(STATE.get("settings_path") or "~/.omnicortex/studio.json")),
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
    return oc_auto.HomeAssistant(ha["url"], tok) if ha.get("url") and tok else None


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
        start, _ = oc_auto.period_bounds(w["period"], now, w.get("reset_day", 1), w.get("week_start", 0))
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
        res = {r["id"]: oc_auto.evaluate(r, inputs, now, prev.get(r["id"])) for r in rules}
        AUTO["results"] = res
        combined = oc_auto.combine(rules, res)
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
    return os.path.join(os.path.dirname(os.path.expanduser(STATE.get("settings_path") or "~/.omnicortex/studio.json")),
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
        ok = bool(info and info.get("allowed") and info.get("omnicortex_dir"))
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
                r = oc_pairing.request(ln["url"], "GET", "/v1/models", token=ln["token"],
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
    r = oc_pairing.request(upstream["base"], "POST", path, payload, token=upstream["token"],
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
                         os.path.expanduser(STATE.get("config_path") or "~/.omnicortex/config.json")))):
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
        import plugins as oc_plugins
        home = os.path.dirname(os.path.expanduser(STATE.get("config_path") or "~/.omnicortex/config.json"))
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dirs = STATE.get("plugin_dirs") or [os.path.join(home, "plugins"), os.path.join(repo, "plugins")]
        pm = STATE["plugins"] = oc_plugins.Plugins(dirs, os.path.join(home, "plugins.json"),
                                                   bundled=[os.path.join(repo, "plugins")])
    return pm


oc_jobs.EXTRA = lambda: plugins().job_specs()


def addon_builtins():
    """Built-in add-ons (viz/addons.py): which parts of Studio the owner removed."""
    import addons

    def save(v):
        STATE["addons_off"] = list(v)
        save_studio_config({"addons_off": list(v)})
    return addons.Builtins(lambda: STATE.get("addons_off") or [], save)


def addon_packages():
    import addons
    home = os.path.dirname(os.path.expanduser(STATE.get("config_path") or "~/.omnicortex/config.json"))
    pm = plugins()
    install_dir = pm.dirs[0] if pm.dirs else os.path.join(home, "plugins")

    def save(v):
        STATE["addon_repos"] = list(v)
        save_studio_config({"addon_repos": list(v)})
    return addons.Packages(pm, install_dir, lambda: STATE.get("addon_repos") or [], save)


oc_jobs.HIDDEN = lambda kind, spec: addon_builtins().job_hidden(kind, spec)


def mcp_hub():
    hub = STATE.get("mcp")
    if hub is None and (STATE.get("mcp_config") or plugins().mcp_servers()):
        hub = STATE["mcp"] = oc_mcp.MCPHub(STATE.get("mcp_config"))
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
        cfg, err = oc_mcp.read_config(path), None
    except (OSError, ValueError) as e:
        cfg, err = {"mcpServers": {}}, f"{path}: {e}"
    hub = STATE.get("mcp")
    live = {x["name"]: x for x in (hub.status() if hub else [])}
    out = []
    for name, spec in cfg.get("mcpServers", {}).items():
        if not isinstance(spec, dict):
            continue
        st = live.get(oc_mcp._safe(name)) or {}
        out.append({"name": name, "spec": oc_mcp.public_spec(spec), "transport": "http" if spec.get("url") else "stdio",
                    "disabled": bool(spec.get("disabled")), "connected": bool(st.get("connected")),
                    "error": st.get("error"), "tools": [t["name"] for t in st.get("tools") or []]})
    plug = []
    try:
        for name, spec in (plugins().mcp_servers() or {}).items():
            st = live.get(oc_mcp._safe(name)) or {}
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
            "job_kinds": sorted(oc_jobs.all_specs()), "models": [m["id"] for m in scan_models()][:300]}


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
        out = proxy("/v1/chat/completions", payload, stream_to=wfile, hold_done=True, upstream=upstream, guard=True)
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
        return oc_rag.Embedder(STATE["rag_embed"])
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
    return oc_rag.Embedder(f"http://127.0.0.1:{port}/v1@rag-embed")


def rag_store():
    st = STATE.get("rag")
    if st is None:
        st = STATE["rag"] = oc_rag.RagStore(STATE.get("rag_dir"))
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
        hits, info = oc_compress.compress_passages(query, hits, budget=max(300, (_ctx_size() or 8192) // 5))
        req["_compressed_passages"] = info
    ctx = {"role": "system", "content": oc_rag.context_message(hits)}
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
    if mode not in ("cut", "summarize", "anchor") or not n_ctx:
        return msgs, None
    anchor = mode == "anchor"                 # summarize, grouped by topic (scripts/semantic_compact.py)
    if anchor:
        mode = "summarize"
    head = 0
    while head < len(msgs) and msgs[head].get("role") == "system":
        head += 1
    fixed, conv = msgs[:head], msgs[head:]
    upto, text = 0, None
    sm = req.get("summary") or {}
    if mode == "summarize" and isinstance(sm, dict) and sm.get("text") and 0 < int(sm.get("upto") or 0) <= len(conv):
        upto, text = int(sm["upto"]), str(sm["text"])[:8000]
    count = lambda t: int(oc_compress.est_tokens(t) * 1.15) + 1      # noqa: E731  # leave a margin: code tokenizes densely
    budget = max(256, n_ctx - reserve - 64)
    cur = fixed + ([oc_compress.summary_message(text, upto)] if text else []) + conv[upto:]
    cut = oc_compress.fit_history(cur, budget, count)
    if cut == 0:
        return cur, ({"history": "summary", "upto": upto} if text else None)
    lead = len(fixed) + (1 if text else 0)              # the summary counts as a leading system message
    if mode == "cut":
        return cur[:lead] + cur[lead + cut:], {"history": "cut", "dropped": upto + cut, "upto": upto + cut}
    folded = conv[upto:upto + cut]
    if anchor:
        import semantic_compact
        prompt = semantic_compact.anchor_prompt(folded, text)
    else:
        prompt = oc_compress.summary_prompt(folded, text)
    r = proxy("/v1/chat/completions", {"messages": prompt, "max_tokens": 900 if anchor else 500, "temperature": 0.2,
                                       "stream": False}, upstream=upstream)
    new = ((r.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    new = re.sub(r"(?s)<think>.*?</think>", "", new).strip()
    if not new:
        return cur[:lead] + cur[lead + cut:], {"history": "cut", "dropped": upto + cut, "upto": upto + cut,
                                               "note": "the model gave no summary; the oldest turns were dropped"}
    upto += cut
    return fixed + [oc_compress.summary_message(new, upto)] + conv[upto:], \
        {"history": "summarized", "upto": upto, "summary": new, **({"anchored": True} if anchor else {})}


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
    if ((req.get("compress") or {}).get("history") in ("summarize", "anchor") and isinstance(sm, dict) and sm.get("text")
            and 0 < int(sm.get("upto") or 0) <= len(msgs)):
        summarized = int(sm["upto"])
        msgs = [oc_compress.summary_message(str(sm["text"]), summarized)] + msgs[summarized:]
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


def options() -> dict:
    """Settings → Integrated options (scripts/integrated.py)."""
    import integrated
    return integrated.merged(load_studio_config().get("options"))


def apply_options() -> None:
    """Push settings that live outside a request: the job runner's priority and cores (core shunt)."""
    if STATE.get("jobs") is not None:
        STATE["jobs"].shunt = options()["core_shunt"]


def harvest_add(req: dict) -> dict:
    import harvest
    opt = options()["harvest"]
    if not opt["enabled"]:
        raise ValueError("Fix This Mistake is turned off (Settings → Integrated options)")
    model_id = str(req.get("model") or "")
    msgs = req.get("messages") if isinstance(req.get("messages"), list) else []
    reply = str(req.get("reply") or "")
    item = harvest.add(model_id, msgs, reply, str(req.get("correction") or ""), str(req.get("note") or ""),
                       home=os.path.dirname(os.path.expanduser(STATE.get("config_path") or "~/.omnicortex/config.json")))
    m = find_model(model_id) if model_id else None
    if m is not None and STATE.get("cett") and activation_settings()["collect"]:
        clf = load_settings().get(_key(m["path"]), {}).get("classifier")
        try:                         # the reply's activations, as a known mistake, for the neuron review
            SCORE_JOBS.put_nowait((m, clf, item["context"], reply, None, "wrong"))
        except queue.Full:
            pass
    if options()["decay"]["enabled"] and model_id:
        decay_log().trigger(model_id, "Fix This Mistake correction")
    pend = len(harvest.unused(home=os.path.dirname(os.path.expanduser(STATE.get("config_path") or "~/.omnicortex/config.json"))))
    return {"id": item["id"], "pending": pend, "auto_train": opt["auto_train"] and bool(opt["train_model"]),
            "min_pairs": opt["min_pairs"]}


def _harvest_home():
    return os.path.dirname(os.path.expanduser(STATE.get("config_path") or "~/.omnicortex/config.json"))


def decay_log():
    if STATE.get("decay_log") is None:
        import neuron_decay
        STATE["decay_log"] = neuron_decay.Log(os.path.join(_harvest_home(), "decay.json"))
    return STATE["decay_log"]


def decay_step(now=None):
    """Neuron aging: ease the loaded model's live LoRA scale toward its floor when its corrections go unused."""
    dc = options()["decay"]
    m = PROC.get("model")
    if not (dc["enabled"] and m and PROC.get("lora") and server_running()):
        return None
    full = float(load_settings().get(_key(m["path"]), {}).get("lora_scale", 1.0) or 0)
    s = decay_log().decide(m["id"], full, dc, now)
    if s is not None and abs(s - float(PROC.get("lora_scale") or 0)) >= 0.01:
        proxy("/lora-adapters", [{"id": 0, "scale": s}], record=False)
        PROC["lora_scale"] = s
    return s


def decay_loop():
    while True:
        time.sleep(3600)
        try:
            decay_step()
        except Exception as e:
            print(f"[studio] decay: {e}", file=sys.stderr)


def harvest_autotrain_loop():
    """Fix This Mistake's micro-tune: when enough new pairs wait and the computer has been idle long enough."""
    from omnicortex_remote import presence as oc_presence
    while True:
        time.sleep(300)
        try:
            opt = options()["harvest"]
            if not (opt["enabled"] and opt["auto_train"] and opt["train_model"]) or STATE.get("jobs") is None:
                continue
            import harvest
            if len(harvest.unused(home=_harvest_home())) < opt["min_pairs"]:
                continue
            idle = oc_presence.snapshot().get("idle_s")
            if idle is None or idle < opt["idle_minutes"] * 60 or ACTIVITY["active"] > 0:
                continue
            if any(j.get("kind") == "harvest_train" and j.get("status") == "running" for j in STATE["jobs"].list()):
                continue
            STATE["jobs"].start("harvest_train", {"model": opt["train_model"], "steps": opt["steps"],
                                                  "min": opt["min_pairs"],
                                                  "out": f"runs/harvest/{time.strftime('%Y%m%d-%H%M%S')}"})
        except Exception as e:
            print(f"[studio] harvest auto-train: {e}", file=sys.stderr)


def proxy(path, payload, stream_to=None, hold_done=False, upstream=None, record=True, source="api", guard=False):
    """_proxy, with each chat reply written to the work log (who answered, on what, its tokens and time).
    record=False for a call its caller logs itself (a project task). guard: Studio's own chat, which can be
    told to trim text it already shows, so Neural Reset may cut a looping reply and resume it."""
    if upstream is not None and path in CHAT_PATHS and options()["sanitizer"]["enabled"]:
        return _sanitized(path, payload, stream_to, hold_done, upstream, record, source, guard)
    if not record or path not in CHAT_PATHS:
        return _proxy(path, payload, stream_to, hold_done, upstream, guard=guard)
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
        out = _proxy(path, payload, stream_to, hold_done, upstream, meas, guard=guard)
        if isinstance(out, dict) and ("usage" in out or "timings" in out):
            meas.update(oc_work.llama_timings(out))
        return out
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        log_work("chat", worker or "chat model", model=model, host=host, device=device, error=err,
                 seconds=time.time() - t0, source=source, **meas)


class _RestoringWriter:
    """Wraps the client's stream: puts the sanitizer's placeholders back into each SSE delta."""

    def __init__(self, w, sess):
        import sanitizer
        self.w, self.r = w, sanitizer.StreamRestorer(sess)

    def write(self, raw):
        line = raw.decode(errors="replace")
        if line.startswith("data:") and "[DONE]" not in line:
            try:
                d = json.loads(line[5:])
                for ch in d.get("choices") or []:
                    dl = ch.get("delta") or {}
                    if isinstance(dl.get("content"), str):
                        dl["content"] = self.r.feed(dl["content"])
                        if ch.get("finish_reason"):
                            dl["content"] += self.r.flush()
                raw = f"data: {json.dumps(d)}\n".encode()
            except (ValueError, TypeError):
                pass
        elif "[DONE]" in line:
            rest = self.r.flush()
            if rest:
                self.w.write(f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': rest}}]})}\n\n".encode())
        self.w.write(raw)

    def flush(self):
        self.w.flush()


def _sanitized(path, payload, stream_to, hold_done, upstream, record, source, guard):
    """A chat for a linked host: placeholders out, real values back in (scripts/sanitizer.py)."""
    import sanitizer
    sess = sanitizer.Session(options()["sanitizer"])
    clean = {**payload, "messages": sess.clean_messages(payload.get("messages") or [])}
    w = _RestoringWriter(stream_to, sess) if stream_to is not None else None
    out = _proxy_logged(path, clean, w, hold_done, upstream, record, source, guard)
    if isinstance(out, dict):
        if isinstance(out.get("text"), str):
            out["text"] = sess.restore(out["text"])
        for ch in out.get("choices") or []:
            msg = ch.get("message") or {}
            if isinstance(msg.get("content"), str):
                msg["content"] = sess.restore(msg["content"])
        out["sanitized"] = {k: n for k, n in sess.counts.items()}
    return out


def _proxy_logged(path, payload, stream_to, hold_done, upstream, record, source, guard):
    if not record:
        return _proxy(path, payload, stream_to, hold_done, upstream, guard=guard)
    t0, meas, err = time.time(), {}, None
    host = upstream.get("name") or "linked host"
    model = payload.get("model")
    try:
        out = _proxy(path, payload, stream_to, hold_done, upstream, meas, guard=guard)
        if isinstance(out, dict) and ("usage" in out or "timings" in out):
            meas.update(oc_work.llama_timings(out))
        return out
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        log_work("chat", f"{host}{LINK_SEP}{model}" if model else host, model=model, host=host, error=err,
                 seconds=time.time() - t0, source=source, **meas)


def _proxy(path, payload, stream_to=None, hold_done=False, upstream=None, meas=None, guard=False):
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
            return oc_pairing.request(upstream["base"], "POST", path, payload, token=upstream["token"],
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
    lg = None
    if guard and path == "/v1/chat/completions":
        nr = options()["neural_reset"]
        if nr["enabled"] and nr["max_resets"] > 0:
            import loopguard
            lg = loopguard.Guard(max_resets=nr["max_resets"], min_repeats=nr["min_repeats"], min_span=nr["min_span"])
    while True:
        hit = None
        try:
            r = opener()
        except (urllib.error.URLError, ConnectionError) as e:
            # backend hot-swap: the local server died; reload it on the fallback ladder and send this once more
            m = PROC.get("model")
            if upstream is not None or m is None or server_running() or not options()["hot_swap"]["retry_request"] \
                    or (meas or {}).get("hot_swapped"):
                raise
            settings = {**DEFAULTS, **load_settings().get(_key(m["path"]), {})}
            ok, why = start_server(m, settings, STATE["backend_port"])
            if not ok:
                raise RuntimeError(f"the model server stopped ({e}) and could not be restarted: {why[:300]}")
            if meas is not None:
                meas["hot_swapped"] = True
            req3 = urllib.request.Request(f"http://127.0.0.1:{PROC['port']}{path}", method="POST",
                                          data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            r = urllib.request.urlopen(req3, timeout=900)
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
                            meas.update(oc_work.llama_timings(d))
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
                    if lg is not None and not calls and delta.get("content"):
                        hit = lg.check("".join(text))
                        if hit:
                            break
        if not hit:
            break
        # Neural Reset: keep the reply up to the loop, tell the page to trim, resume past it
        import loopguard
        full = "".join(text)
        kept = loopguard.keep_upto(full, hit)
        text, finish = [kept], None
        stream_to.write(f"data: {json.dumps({'neural_reset': {'drop': len(full) - len(kept), 'unit': hit['unit'][:120], 'repeats': hit['repeats']}})}\n\n".encode())
        stream_to.flush()
        payload = loopguard.resume_payload(payload, kept, hit)
        if upstream is not None:
            opener = lambda p=payload: _upstream_stream(upstream, path, p)  # noqa: E731
        else:
            req2 = urllib.request.Request(f"http://127.0.0.1:{PROC['port']}{path}", method="POST",
                                          data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            opener = lambda q=req2: urllib.request.urlopen(q, timeout=900)  # noqa: E731
        if meas is not None:
            meas["neural_resets"] = meas.get("neural_resets", 0) + 1
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
    ac = activation_settings()
    # Every reply's activations are collected by default (Settings -> Activations turns it off); with a
    # classifier the same pass also scores the reply's risk.
    want = (clf and _REPLIES["n"] % max(1, STATE["score_every"]) == 0) or \
        (ac["collect"] and _REPLIES["n"] % max(1, int(ac["every"] or 1)) == 0)
    if want and STATE["cett"] and text:
        try:
            SCORE_JOBS.put_nowait((model, clf, messages, text, subject))
        except queue.Full:
            ACT_STATE["skipped_busy"] += 1   # never queue behind scoring; the next reply matters more
    elif want and text:
        ACT_STATE["no_method"] += 1


def score_worker():
    while True:
        model, clf, messages, text, subject, *extra = SCORE_JOBS.get()
        verdict = extra[0] if extra else "unknown"          # "wrong" for a reply fixed with Fix This Mistake
        # Activation scoring is an extra prefill; wait for a quiet moment.
        while ACTIVITY["active"] > 0:
            time.sleep(1)
        try:
            sc = _scorer(model, clf)
            msgs = [m for m in messages if m.get("role") != "system"] or messages
            prof = getattr(sc, "profile", None)   # with a classifier, profile() also scores the reply
            res = prof(msgs, text) if prof else sc.score(msgs, text)
            if "cett" in res and activation_settings()["collect"]:
                try:
                    import neuron_review
                    activation_sink().record(model["id"], neuron_review.bin_profile(res["cett"]),
                                             source="harvest" if verdict != "unknown" else "live",
                                             kind="labelled" if verdict != "unknown" else "observed",
                                             subjects=[subject] if subject else None, verdict=verdict,
                                             risk=res.get("prob"))
                except Exception as e:
                    print(f"[studio] activation record failed: {e}", file=sys.stderr)
            dc = options()["decay"]
            if dc["enabled"] and isinstance(res.get("prob"), (int, float)) and res["prob"] >= dc["risk"]:
                decay_log().trigger(model["id"], f"risky reply ({res['prob']:.2f})")
            if clf and res.get("score") is not None:
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


ACT_STATE = {"skipped_busy": 0, "no_method": 0}


def activation_settings() -> dict:
    """Settings -> Activations (scripts/activation_store.py): collected by default, stored where the owner says."""
    import activation_store
    return {**activation_store.DEFAULTS, **(load_studio_config().get("activations") or {})}


def activation_sink():
    sink = STATE.get("act_sink")
    if sink is None:
        import activation_store
        home = os.path.dirname(os.path.expanduser(STATE.get("config_path") or "~/.omnicortex/config.json"))
        sink = STATE["act_sink"] = activation_store.Sink(activation_settings(), STATE.get("review_dir"), home)
        if sink.cfg["store"] in ("local",):
            sink._store = review_store()          # the same store the Review page reads
    return sink


def activation_status() -> dict:
    import activation_store
    ac = activation_settings()
    sink = activation_sink()
    st = sink.status()
    st["method"] = ("llama-cett-dump (llama.cpp)" if STATE.get("cett") else None)
    st["why_not"] = None if STATE.get("cett") else (
        "no compatible method on this computer yet: activations of GGUF models are read with llama-cett-dump "
        "(Setup -> Build llama.cpp). Until then replies are not profiled.")
    st.update(ACT_STATE)
    if st.get("root"):
        st["free_gb"] = activation_store.disk_free(st["root"] if os.path.isdir(st["root"]) else os.path.dirname(st["root"]))
    return {"settings": {**ac, "studio_token": "set" if ac.get("studio_token") else ""}, "status": st}


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
        home = os.path.dirname(os.path.expanduser(STATE.get("config_path") or "~/.omnicortex/config.json"))
        reg = STATE["workers_reg"] = worker_registry.Registry(os.path.join(home, "workers.json"))
    return reg


def work_log():
    """The work log (scripts/work_log.py), beside Studio's config unless OC_WORK_LOG names a folder."""
    wl = STATE.get("work_log")
    if wl is None:
        home = os.path.dirname(os.path.expanduser(STATE.get("config_path") or "~/.omnicortex/config.json"))
        wl = STATE["work_log"] = oc_work.WorkLog(os.environ.get("OC_WORK_LOG") or os.path.join(home, "worklog"))
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
# Settings that can lock the owner out, expose Studio, or lose data: the Setup page shows them locked, with this
# warning, behind an Unlock button.
SENSITIVE_FIELDS = {
    "host": "Which network address Studio listens on. Anything other than 127.0.0.1 makes it reachable from other "
            "computers: it then needs a token and TLS, and a wrong address can make Studio unreachable after a restart.",
    "port": "Studio's own port. Bookmarks, the Godot client, paired devices, Cline/Claude settings and other "
            "Studios linked to this one all use it; they stop connecting until they are told the new port.",
    "backend_port": "The port of the llama-server Studio starts. If another program already uses it, models fail to load.",
    "token_file": "The file holding Studio's access token. Pointing it elsewhere replaces the token: every paired "
                  "device, linked Studio and app using the old one is locked out.",
    "tls_cert": "The HTTPS certificate. Paired devices and linked Studios pin its fingerprint; a new one makes them "
                "refuse to connect until they are paired again.",
    "tls_key": "The private key for the HTTPS certificate; a mismatch with the certificate stops HTTPS from starting.",
    "allow_plaintext": "Serves Studio over plain HTTP on a network address: the token and every chat travel unencrypted.",
    "allow_remote_jobs": "Lets jobs (which train models and run model-written code) start while Studio is reachable "
                         "from the network. Anyone with the token could then run code on this computer.",
    "no_jobs": "Turns the jobs runner off: evaluation, retraining and every pipeline stage stop being available.",
    "no_persistent_pairing": "Paired devices are then forgotten on every restart and must pair again.",
    "mcp_config": "The MCP servers Studio connects to and lets models call. A different file changes which tools "
                  "models can use, and what they can reach.",
}

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
    "token_file": ("readfile", False, "File with Studio's access token (oc_security.py token makes one)",
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
    return _P(os.path.expanduser(STATE.get("config_path") or "~/.omnicortex/config.json"))


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
    import oc_workspace
    d = oc_workspace.describe(_ws_path(), STATE.get("workspace_forced"))
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


def hw_board() -> dict:
    """The board's hardware profile section: accepted profiles, pending ones and reports (cached an hour)."""
    sh = sharing()
    ep = str(sh.get("endpoint") or "")
    if not (sh.get("enabled") and ep.startswith("github:")):
        return {}
    c = STATE.setdefault("hw_board_cache", {})
    if c.get("ep") != ep or time.time() - c.get("t", 0) > 3600:
        try:
            import github_board
            c.update(ep=ep, t=time.time(), data=github_board.fetch(ep[7:]).get("hw_profiles") or {})
        except Exception as e:
            c.update(ep=ep, t=time.time(), data={}, error=str(e)[:200])
    return c.get("data") or {}


def lookup_board():
    """(source id, the whole board.json or the error) for the sharing board, cached an hour; None if off."""
    sh = sharing()
    ep = str(sh.get("endpoint") or "")
    if not (sh.get("enabled") and ep):
        return None
    c = STATE.setdefault("lookup_board_cache", {})
    if c.get("ep") != ep or time.time() - c.get("t", 0) > 3600:
        import unified_review
        try:
            data = unified_review.fetch_json(unified_review.board_url(ep), timeout=15)
        except Exception as e:
            data = e
        c.update(ep=ep, t=time.time(), data=data)
    return (f"board:{ep[7:] if ep.startswith('github:') else ep}", c["data"])


def unified_lookup(model: str) -> dict:
    """Everything this Studio knows about a Hugging Face model (docs/UNIFIED_REVIEW.md): the sharing board,
    SAELens's catalogue (cached a week) and this Studio's own observations."""
    import unified_review
    b = lookup_board()
    cached = Path(os.environ.get("OC_HOME") or Path.home() / ".omnicortex") / "saelens" / "pretrained_saes.yaml"
    if not cached.exists() or time.time() - cached.stat().st_mtime > 7 * 86400:
        if not STATE.get("saelens_fetching"):      # never make a lookup wait for a download (or for PyTorch)
            STATE["saelens_fetching"] = True

            def fetch():
                try:
                    import sae
                    sae.saelens_registry()
                except Exception:
                    pass
                STATE["saelens_fetching"] = False
            threading.Thread(target=fetch, daemon=True).start()
    if cached.exists():
        c = STATE.setdefault("saelens_cache", {})
        try:
            if c.get("mtime") != cached.stat().st_mtime:
                import yaml
                c.update(mtime=cached.stat().st_mtime, reg=yaml.safe_load(cached.read_text()) or {})
            reg = c["reg"]
        except Exception as e:
            reg = e
    else:
        reg = RuntimeError("fetching SAELens's catalogue; ask again in a minute")
    rec = unified_review.lookup(model, [b] if b else [], reg, review_store())
    rec["open"] = {"hub": "/#hub=" + urllib.parse.quote(rec["model"]), "review": "/review"}
    return rec


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
    local = pp.samples_from_worklog(work_log().read("work", oc_work.parse_since(since)), model_info, _device_info)
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
    home = os.path.dirname(os.path.expanduser(STATE.get("config_path") or "~/.omnicortex/config.json"))
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
    p = os.path.expanduser(path or STATE.get("config_path") or "~/.omnicortex/config.json")
    try:
        with open(p) as f:
            d = json.load(f)
        return d.get("studio", {}) if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_studio_config(values):
    p = os.path.expanduser(STATE.get("config_path") or "~/.omnicortex/config.json")
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
                    oc_pairing.parse_duration(v)
            elif kind == "duration":
                v = str(v or "").strip() or None
                if v:
                    oc_pairing.parse_duration(v)
            elif kind == "durations":
                v = ",".join(x.strip() for x in str(v or "").split(",") if x.strip()) or None
                for x in (v or "").split(",") if v else []:
                    oc_pairing.parse_duration(x)
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
    for root in [os.path.expanduser("~/llama.cpp"), os.environ.get("OC_LLAMA") or ""]:
        for name, key in (("llama-server", "server"), ("llama-cett-dump", "cett")):
            pth = os.path.join(root, "build", "bin", name)
            if root and os.path.isfile(pth):
                found.setdefault(key, pth)
    return {"fields": {k: {"kind": v[0], "live": v[1], "help": v[2], "group": v[3], "value": eff.get(k),
                           "advanced": v[3] in ADVANCED_GROUPS, "sensitive": SENSITIVE_FIELDS.get(k)}
                       for k, v in SETUP_FIELDS.items()},
            "config_path": os.path.expanduser(STATE.get("config_path") or "~/.omnicortex/config.json"),
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
                    "detail": err or ("runs jobs from " + info["omnicortex_dir"] if info and info.get("allowed")
                                      and info.get("omnicortex_dir") else "jobs not allowed by its owner"),
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
        env.setdefault("OC_LLAMA", root)
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
    import oc_mcp as studio_client
    import permissions as P
    try:
        store = graphs_api.Store(STATE.get("graphs_dir") or os.path.expanduser("~/.omnicortex/graphs"))
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
        data.insert(0, {"id": "auto", "object": "model", "owned_by": "omnicortex",
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
                if "oc_token" in c:
                    out.append(c["oc_token"].value)
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
        import plugins as oc_plugins
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
            except oc_plugins.PluginError as e:
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
        except oc_plugins.PluginError as e:
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
        import plugins as oc_plugins
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
        except oc_plugins.PluginError as e:
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
            self.send_header("X-OmniCortex-Route", json.dumps(self._route)[:4000])
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
            self.send_header("X-OmniCortex-Route", json.dumps(self._route)[:4000])
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _v1(self, path):
        """OpenAI-compatible passthrough with JIT model loading."""
        body = self._read(MAX_BODY)
        import morph_router
        mm, morph = morph_router.route(body.get("model") or "", body.get("messages") or [], options()["morph"])
        if morph:
            body = {**body, "model": mm}
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
                    + f"&a={'p' if access is None else oc_pairing.fmt_duration(access)}")
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
                r = oc_pairing.claim(str(req.get("link", "")), str(req.get("device_name") or "studio"))
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
        """MCP over streamable HTTP: JSON-RPC in, JSON-RPC out (scripts/oc_mcp.py).
        Every tool call goes back through this API with the caller's own token,
        so an agent can do exactly what the person or device it acts for can."""
        import oc_mcp
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if origin and urllib.parse.urlparse(origin).netloc != host:
            return self._json(403, {"error": "cross-origin MCP requests are refused"})
        if "application/json" not in self.headers.get("Content-Type", ""):
            return self._json(415, {"error": "MCP requests must be application/json"})
        try:
            payload = self._read(4 * 1024 * 1024)
        except ValueError as e:
            body = json.dumps(oc_mcp._err(None, -32700, f"parse error: {e}")).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        token = (self._presented() or [None])[0] if STATE["token"] else None
        studio = oc_mcp.Studio(self._self_url(), token, insecure_loopback=True)
        resp = oc_mcp.handle_payload(payload, studio)
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
            return self._json(200, oc_jobs.public_specs())
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
            if kind not in oc_jobs.all_specs():
                return self._json(404, {"error": "no such job kind"})
            return self._json(200, oc_jobs.script_options(kind))
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
        import oc_mcp as studio_client
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
                return self._json(200, oc_work.breakdown(work_log(), q.get("since", "24h")))
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
        return graphs_api.Store(STATE.get("graphs_dir") or os.path.expanduser("~/.omnicortex/graphs"))

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
        """The 3D views follow the viewer's own theme (cookie oc_viz, set by the theme picker)."""
        import bloom
        try:
            c = http.cookies.SimpleCookie(self.headers.get("Cookie") or "")
            name = c["oc_viz"].value if "oc_viz" in c else ""
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
            self.send_header("Content-Disposition", 'attachment; filename="omnicortex-remote.zip"')
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
                out["command"] = f"omnicortex-remote connect {out['invite']}"
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
        if self.path == "/api/review/ingest":
            # observations forwarded by another Studio whose activations are stored here
            import activation_store
            items = req.get("items") if isinstance(req.get("items"), list) else []
            try:
                return self._json(200, {"stored": activation_store.ingest(st, items)})
            except (KeyError, ValueError, TypeError) as e:
                return self._json(400, {"error": f"bad item: {e}"})
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

    def _hw_profiles(self, req):
        """Hardware support profiles (scripts/hw_profiles.py): what fits here, plans, your own, sharing, reports."""
        if not self._owner_only():
            return
        import hw_profiles as HP
        u = urllib.parse.urlsplit(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        board = hw_board()
        community = [{**p, "status": "community"} for p in board.get("accepted", [])]
        try:
            if req is None:
                cat = HP.catalogue() + [p for p in community if p.get("id") not in {x["id"] for x in HP.catalogue()}]
                if u.path == "/api/hw-profiles":
                    devs = HP.facts()
                    return self._json(200, {"devices": devs, "profiles": cat,
                                            "matches": [{"id": m["profile"]["id"], "devices": m["devices"]}
                                                        for m in HP.match(cat, devs)],
                                            "reports": board.get("reports", {}), "pending": board.get("pending", []),
                                            "board": sharing().get("endpoint") or None})
                if u.path == "/api/hw-profiles/plan":
                    p = next((x for x in cat if x.get("id") == q.get("id")), None)
                    if p is None:
                        return self._json(404, {"error": "no such profile"})
                    venv = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                    return self._json(200, HP.plan(p, q.get("purpose"), str(venv) if venv.exists() else None,
                                                   (STATE.get("workspace") or {}).get("llama_dir")))
                return self._json(404, {"error": "not found"})
            if u.path == "/api/hw-profiles/save":
                f = HP.save_user(req.get("profile") or {})
                return self._json(200, {"saved": str(f)})
            sh = sharing()
            if u.path == "/api/hw-profiles/export":
                p = next((x for x in HP.catalogue() if x["id"] == req.get("id")), None)
                if p is None:
                    return self._json(404, {"error": "no such profile"})
                return self._json(200, HP.submission(p))
            if u.path in ("/api/hw-profiles/submit", "/api/hw-profiles/report"):
                if not str(sh.get("endpoint") or "").startswith("github:"):
                    return self._json(400, {"error": "sharing profiles needs a GitHub board: Settings → Sharing results, "
                                                     "scoreboard github:owner/repo (or use Export and share the file)"})
                import github_board
                import score_cards
                repo = sh["endpoint"][7:]
                if u.path.endswith("/submit"):
                    p = next((x for x in HP.catalogue() if x["id"] == req.get("id")), None)
                    if p is None:
                        return self._json(404, {"error": "no such profile"})
                    return self._json(200, github_board.submit_profile(repo, p, score_cards.github_token()))
                rep = HP.report(str(req.get("id") or ""), str(req.get("purpose") or "general"), bool(req.get("worked")),
                                HP.facts(), str(req.get("note") or ""), req.get("gen_tok_s"))
                return self._json(200, github_board.submit_report(repo, rep, score_cards.github_token()))
        except (HP.ProfileError, ValueError) as e:
            return self._json(400, {"error": str(e)})
        except OSError as e:
            return self._json(502, {"error": f"{type(e).__name__}: {e}"})
        return self._json(404, {"error": "not found"})

    def _settings(self):
        if not self._owner_only():
            return
        import oc_workspace
        if self.command == "GET":
            if self.path == "/settings":
                import settings_page
                return self._html(settings_page.PAGE)
            if self.path == "/api/workspaces":
                return self._json(200, workspaces_state())
            if self.path == "/api/settings/activations":
                return self._json(200, activation_status())
            if self.path == "/api/settings/options":
                import integrated
                import core_shunt
                return self._json(200, {"options": integrated.describe(load_studio_config().get("options")),
                                        "core_shunt": core_shunt.describe(options()["core_shunt"]),
                                        "decay": {k: {"scale": v.get("scale"), "last_trigger": v.get("last_trigger"),
                                                      "events": v.get("events", [])[-5:]}
                                                  for k, v in decay_log().d["models"].items()}})
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
                oc_workspace.save(name, req.get("values") or {}, path=path, replace=bool(req.get("replace")))
                out = workspaces_state()
                out["restart_needed"] = name == STATE.get("workspace_name")
                return self._json(200, out)
            if self.path == "/api/workspaces/use":
                oc_workspace.use(str(req.get("name") or ""), path=path)
                out = workspaces_state()
                out["restart_needed"] = out["active"] != STATE.get("workspace_name")
                return self._json(200, out)
            if self.path == "/api/workspaces/delete":
                name = str(req.get("name") or "")
                if name == STATE.get("workspace_name"):
                    return self._json(409, {"error": "Studio is running in this workspace; switch first"})
                oc_workspace.delete(name, path=path)
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
        if self.path == "/api/settings/options":
            import integrated
            saved = load_studio_config().get("options") or {}
            try:
                vals = integrated.validate(str(req.get("name") or ""), req.get("values") or {}, integrated.merged(saved))
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            saved[req["name"]] = vals
            save_studio_config({"options": saved})
            apply_options()
            return self._json(200, integrated.describe(saved))
        if self.path == "/api/settings/activations":
            import activation_store
            cur = activation_settings()
            for k in ("collect", "spool_file"):
                if k in req:
                    cur[k] = bool(req[k])
            if "every" in req:
                cur["every"] = max(1, min(1000, int(req["every"] or 1)))
            if "max_gb" in req:
                cur["max_gb"] = max(0.0, float(req["max_gb"] or 0))
            if "store" in req:
                if req["store"] not in activation_store.MODES:
                    return self._json(400, {"error": f"store: one of {', '.join(activation_store.MODES)}"})
                cur["store"] = req["store"]
            if "folder" in req:
                cur["folder"] = os.path.expanduser(str(req["folder"]).strip())
            if "studio_url" in req:
                u = str(req["studio_url"]).strip().rstrip("/")
                if u and not re.match(r"https?://[\w.\-\[\]:]+(/[\w./-]*)?$", u):
                    return self._json(400, {"error": "the Studio address is a URL like https://server:7870"})
                cur["studio_url"] = u
            if req.get("studio_token") not in (None, "set"):
                cur["studio_token"] = str(req["studio_token"]).strip()
            if cur["store"] == "folder" and not cur["folder"]:
                return self._json(400, {"error": "store 'folder' needs a folder"})
            if cur["store"] == "studio" and not cur["studio_url"]:
                return self._json(400, {"error": "store 'studio' needs the other Studio's address"})
            save_studio_config({"activations": cur})
            STATE["act_sink"] = None                     # rebuilt with the new settings on the next reply
            return self._json(200, activation_status())
        if self.path == "/api/settings/activations/retry":
            return self._json(200, {"delivered": activation_sink().retry(), **activation_status()})
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

    # ------------------------------------------------ requirements (scripts/oc_requirements.py)

    def _requirements(self):
        if not self._owner_only():
            return
        import oc_requirements as nrq
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
                self.send_header("Content-Disposition", 'attachment; filename="omnicortex-env.txt"')
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
        if self.path.startswith("/api/explain/plan"):
            # every way to explain a reply of this model, which can run here, and how to start each (scripts/explain.py)
            import explain
            import features_api as FA
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            m = find_model((q.get("model") or [""])[0]) or (PROC["model"] if PROC.get("model") else None)
            if m is None:
                return self._json(404, {"error": "no such model"})
            root, extra = self._features_ws()
            try:
                inv = FA.list_all(root, extra)
            except OSError:
                inv = {}
            caps = explain.capabilities(m, load_settings().get(_key(m["path"]), {}), STATE.get("cett"), inv)
            return self._json(200, {"model": m["id"], "capabilities": caps,
                                    "plan": explain.plan(caps, (q.get("prompt") or [""])[0], (q.get("reply") or [""])[0])})
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
            import oc_connect
            base = self._self_url(public=True)
            out = oc_connect.snippets(base, "token" if STATE["token"] else None)
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
        if self.path.startswith("/api/hw-profiles"):
            return self._hw_profiles(None)
        if self.path == "/api/harvest":
            import harvest
            its = harvest.items(home=_harvest_home())
            return self._json(200, {"settings": options()["harvest"], "pairs": [
                {k: it.get(k) for k in ("id", "t", "model", "note", "used_in")} | {"question": it["context"][-1]["content"][:200]}
                for it in its[-200:]], "unused": sum(1 for i in its if not i.get("used_in"))})
        if self.path == "/api/env/snapshot":
            import dep_snapshot
            return self._json(200, dep_snapshot.snapshot([os.path.dirname(STATE["cett"])] if STATE.get("cett") else None))
        if self.path.split("?")[0] == "/api/lookup":
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            try:
                return self._json(200, unified_lookup(q.get("model", [""])[0]))
            except ValueError as e:
                return self._json(400, {"error": str(e)})
        if self.path in ("/settings", "/api/workspaces", "/api/settings", "/api/settings/sharing", "/api/settings/permissions",
                         "/api/settings/activations", "/api/settings/options"):
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
                body = oc_jobs.PAGE.encode()
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
                STATE["hw"] = oc_qsel.hardware()
            try:
                allc = json.loads(Path(STATE.get("quant_choices") or oc_qsel.CHOICES).read_text())
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
                m["family"], m["quant"] = oc_qsel.family(m["path"]), oc_qsel.quant_of(m["path"])
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
                                    "computer": socket.gethostname(), "periods": oc_auto.PERIODS,
                                    "actions": oc_auto.ACTIONS})
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
                                 f"oc_token={STATE['token']}; Path=/; "
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
                self.send_header("Set-Cookie", f"oc_token={r['token']}; Path=/; HttpOnly; SameSite=Strict; "
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

            if self.path.startswith("/api/hw-profiles/"):
                return self._hw_profiles(self._read())

            if self.path in ("/api/harvest", "/api/harvest/delete"):
                if not self._owner_only():
                    return
                req = self._read(MAX_BODY)
                if self.path == "/api/harvest/delete":
                    import harvest
                    return self._json(200, {"removed": harvest.remove(str(req.get("id") or ""), home=_harvest_home())})
                try:
                    return self._json(200, harvest_add(req))
                except ValueError as e:
                    return self._json(400, {"error": str(e)})

            if self.path == "/api/env/compare":
                # this computer's dependencies against a linked host's (scripts/dep_snapshot.py)
                import dep_snapshot
                req = self._read()
                ln = next((x for x in load_links() if x["name"] == req.get("link") and x.get("kind") != "agent"), None)
                if ln is None:
                    return self._json(404, {"error": "no such linked Studio"})
                try:
                    there = oc_pairing.request(ln["url"], "GET", "/api/env/snapshot", None, token=ln["token"],
                                               fingerprint=ln.get("fingerprint", ""), timeout=30)
                except Exception as e:
                    return self._json(502, {"error": f"{ln['name']}: {e}"})
                here = dep_snapshot.snapshot([os.path.dirname(STATE["cett"])] if STATE.get("cett") else None)
                d = dep_snapshot.diff(here, there)
                return self._json(200, {**d, "plan": dep_snapshot.plan(d), "there": ln["name"]})

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
                        cfg["rules"] = [oc_auto.clean_rule(r) for r in req.get("rules") or []]
                        seen = set()
                        for r in cfg["rules"]:                  # every rule its own id, even if a client repeated one
                            if r["id"] in seen:
                                r["id"] = oc_auto.clean_rule({**r, "id": None})["id"]
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
                    import oc_workspace
                    ws_keys = {"models_dir": "models_dirs", "server": "server", "cett": "cett",
                               "download_dir": "download_dir"}
                    to_ws = {ws_keys[k]: v for k, v in clean.items() if k in ws_keys}
                    if to_ws:
                        oc_workspace.save(STATE.get("workspace_name") or oc_workspace.active_name(_ws_path()),
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
                        oc_mcp.save_server(path, req.get("name", ""), req.get("spec") or {}, req.get("old_name"))
                    elif self.path.endswith("/remove"):
                        oc_mcp.remove_server(path, str(req.get("name", "")))
                    else:
                        oc_mcp.set_disabled(path, str(req.get("name", "")), bool(req.get("disabled")))
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
                import morph_router
                req["model"], morph = morph_router.route(req.get("model") or "", req.get("messages") or [], options()["morph"])
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
                try:                                  # sampling from add-ons that are on (semantic friction)
                    smp = plugins().sampling()
                    if smp:
                        import friction
                        payload = friction.apply(payload, smp)
                except Exception as e:
                    print(f"[studio] plugin sampling: {e}", file=sys.stderr)
                if upstream is not None:
                    payload["model"] = rid
                if morph:
                    route = {**route, "morph": morph}
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
                        out = proxy("/v1/chat/completions", payload, stream_to=self.wfile, upstream=upstream,
                                    source="chat", guard=True)
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


from studio_page import LINK_PAGE as _LINK_PAGE, LOGIN, PAGE as _PAGE, PAIR_PAGE  # noqa: E402,F401
from ui_links import inject as _inject_links  # noqa: E402
PAGE = _inject_links(_PAGE).replace("__MODEL_DEFAULTS__", json.dumps(DEFAULTS))
LINK_PAGE = _inject_links(_LINK_PAGE)




def build_parser():
    p = argparse.ArgumentParser(description="OmniCortex Studio: model manager, chat and OpenAI-compatible server")
    p.add_argument("--models-dir", action="append", default=[],
                   help="repeatable; searched recursively for .gguf")
    p.add_argument("--server", help="path to llama-server (or $OC_LLAMA_SERVER)")
    p.add_argument("--vllm", default=os.environ.get("OC_VLLM", "vllm"),
                   help="the vLLM command, for models served with the vLLM engine (default $OC_VLLM or vllm)")
    p.add_argument("--port", type=int, default=7870)
    p.add_argument("--host", default="127.0.0.1",
                   help="non-loopback binds require a token, and TLS unless --allow-plaintext")
    p.add_argument("--backend-port", type=int, default=8080, help="local port for llama-server")
    p.add_argument("--download-dir",
                   help="where hub downloads land (default: first --models-dir)")
    p.add_argument("--settings",
                   default=os.path.expanduser("~/.omnicortex/studio.json"))
    p.add_argument("--chats-dir", default=os.path.expanduser("~/.omnicortex/chats"))
    p.add_argument("--stats-dir", default=str(model_stats.DEFAULT_DIR),
                   help="rolling per-model stats (shared with testqa.py --record-stats)")
    p.add_argument("--stats-window", type=int, default=500, help="most recent observations per kind that count")
    p.add_argument("--min-graded", type=int, default=20,
                   help="graded results a model needs before model \"auto\" will consider it")
    p.add_argument("--hallucination-cost", type=float, default=1.0,
                   help="how much a wrong answer costs relative to a right one when auto ranks models")
    p.add_argument("--cett", default=os.environ.get("OC_CETT"),
                   help="llama-cett-dump binary; enables activation scoring for models with a classifier set")
    p.add_argument("--score-ngl", type=int, default=0,
                   help="GPU layers for activation scoring (0 keeps it off the GPU llama-server is using)")
    p.add_argument("--score-every", type=int, default=1, help="score one reply in N")
    p.add_argument("--projects-dir", default=os.path.expanduser("~/.omnicortex/projects"),
                   help="director projects: plans, results and history (/projects)")
    p.add_argument("--max-workers", type=int, default=2,
                   help="worker models the director may run at once, besides the chat model")
    p.add_argument("--worker-idle", type=int, default=300, help="stop a worker model after N idle seconds")
    p.add_argument("--hardware", default=os.path.expanduser("~/.omnicortex/hardware.json"),
                   help="accelerators for worker models: llama-server per backend (rocm, vulkan, cuda, metal, "
                        "cpu), per-device settings, devices to skip (see scripts/accelerators.py)")
    p.add_argument("--no-work-log", action="store_true",
                   help="do not record who did what work and hardware samples (/work)")
    p.add_argument("--graphs-dir", default=os.path.expanduser("~/.omnicortex/graphs"),
                   help="node graphs and their runs (/graphs)")
    p.add_argument("--traces-dir", default=os.path.expanduser("~/.omnicortex/traces"),
                   help="where per-reply checks are saved (open them in the 3D view or timeline.py)")
    p.add_argument("--review-dir", default=os.path.expanduser("~/.omnicortex/review"),
                   help="neuron review store: every checked reply's activations, for the /review page")
    p.add_argument("--idle-ttl", type=int, default=0, help="unload the model after N idle seconds (0 = never)")
    p.add_argument("--no-jit", action="store_true", help="/v1 requests never load or swap models")
    p.add_argument("--jobs-dir", default=os.path.expanduser("~/.omnicortex/jobs"),
                   help="logs and state of evaluation/retraining jobs started from /jobs")
    p.add_argument("--max-jobs", type=int, default=2, help="jobs that may run at once")
    p.add_argument("--allow-remote-jobs", action="store_true",
                   help="allow /jobs on a non-loopback bind (jobs can train models and run model-written code)")
    p.add_argument("--no-jobs", action="store_true", help="disable /jobs entirely")
    p.add_argument("--devices", default=os.path.expanduser("~/.omnicortex/devices.json"),
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
    p.add_argument("--links", default=os.path.expanduser("~/.omnicortex/links.json"),
                   help="linked hosts whose models this Studio serves (holds their device tokens; 0600)")
    p.add_argument("--mcp-config", default=str(oc_mcp.DEFAULT_CONFIG),
                   help="MCP servers for chat tools (mcpServers JSON, as in LM Studio / Claude Desktop)")
    p.add_argument("--rag-dir", default=os.path.expanduser("~/.omnicortex/rag"), help="document collections")
    p.add_argument("--rag-embed", help="URL[@model] of an OpenAI-compatible embeddings endpoint for dense retrieval")
    p.add_argument("--rag-embed-gguf", help="embedding GGUF; Studio runs it with llama-server --embedding on demand")
    p.add_argument("--rag-embed-ngl", type=int, default=0, help="GPU layers for the embedding model")
    sec.add_server_security_args(p)
    p.add_argument("--config", default=os.environ.get("OC_CONFIG", "~/.omnicortex/config.json"),
                   help="settings file (\"studio\" section); the Setup page writes it, and flags override it")
    p.add_argument("--workspace", help="which workspace's folders to use (Settings -> Workspaces; "
                                       "default: the active one)")
    return p


def main(argv=None):
    try:
        import oc_migrate
        oc_migrate.migrate()
    except ImportError:
        pass
    p = build_parser()
    pre, _ = p.parse_known_args(argv)
    STATE["config_path"] = pre.config
    os.environ["OC_CONFIG"] = os.path.expanduser(pre.config)
    import oc_workspace
    # A workspace chosen for this run (flag or variable) wins over the file's choice.
    STATE["workspace_forced"] = pre.workspace or os.environ.get("OC_WORKSPACE")
    if pre.workspace:
        if pre.workspace not in oc_workspace.names():
            print(f"error: no workspace {pre.workspace!r}; have {', '.join(oc_workspace.names())}", file=sys.stderr)
            return 2
        os.environ["OC_WORKSPACE"] = pre.workspace
    ws = oc_workspace.active()
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
    os.environ.update(oc_workspace.env_for(ws))

    STATE["models_dirs"] = [os.path.expanduser(d) for d in a.models_dir] or ws["models_dirs"]
    a.server = a.server or (ws["server"] if os.path.isfile(ws["server"]) else None)
    a.cett = a.cett or (ws["cett"] if os.path.isfile(ws["cett"]) else None)
    a.download_dir = a.download_dir or ws["download_dir"]
    STATE["server_bin"] = a.server or os.environ.get("OC_LLAMA_SERVER")
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
    os.environ.setdefault(sec.TOKEN_ENV, os.environ.get("OC_STUDIO_TOKEN", ""))
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
        policy = oc_pairing.PairingPolicy(a.pair_code_ttl, a.pair_durations.split(","), a.pair_max,
                                          not a.no_persistent_pairing, a.pair_default)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    STATE.update(devices=oc_pairing.DeviceRegistry(a.devices, policy), links_path=os.path.expanduser(a.links),
                 fingerprint=oc_pairing.cert_fingerprint(a.tls_cert) if a.tls_cert and a.tls_key else None)
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
        STATE["jobs"] = oc_jobs.JobRunner(a.jobs_dir, a.max_jobs, cwd=ws["root"])
        STATE["jobs"].on_finish = job_finished
        apply_options()
        threading.Thread(target=harvest_autotrain_loop, daemon=True).start()
    threading.Thread(target=decay_loop, daemon=True).start()

    scheme = "https" if tls else "http"
    penv = {"OMNICORTEX_URL": f"{scheme}://127.0.0.1:{a.port}", **({"OMNICORTEX_TOKEN": STATE["token"]} if STATE.get("token") else {})}
    plugins().env = penv                      # plugin checks, nodes and jobs can call Studio's API
    if STATE.get("jobs") is not None:
        STATE["jobs"].plugin_env = penv
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
              f"  python scripts/oc_workspace.py set {ws['name']} models_dirs /path/to/models")
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
    STATE["work_log_off"] = a.no_work_log or os.environ.get("OC_WORK_LOG") == "off"
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
