#!/usr/bin/env python3
"""
Inference engine options, described once: what each one does, how it is
spelled on each engine's command line, and which build supports it.

Studio's model panel (Engine options), the optimizer (engine_optimizer.py)
and the command line all use this table, so a flag added here appears in the
UI with its help text, is checked against the installed binary, and is
passed in the spelling that binary understands (llama.cpp renames flags:
--draft-max became --spec-draft-n-max).

    python scripts/engine_options.py                       # the options, grouped
    python scripts/engine_options.py --server ~/llama.cpp/build/bin/llama-server   # which this build has
    python scripts/engine_options.py --server ... --args '{"flash_attn": "on", "cache_type_k": "q8_0"}'

Engines: llama.cpp's llama-server, and vLLM (`vllm serve`). An option without a
spelling for an engine is not offered for it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys

KV_TYPES = ["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "f32"]
SPEC_TYPES = ["none", "draft-simple", "ngram-simple", "ngram-map-k", "ngram-mod", "ngram-cache", "draft-eagle3",
              "draft-mtp"]

# key, group, label, kind, default, choices, help, {engine: [flag spellings, preferred first]}
# kind: bool (flag present when true), onoff (flag value on/off), int, float, choice, text.
# A default of None means "not passed": the engine's own default applies.
OPTIONS = [
    # ---------------------------------------------------------------- speed
    ("flash_attn", "Speed", "Flash attention", "choice", "auto", ["auto", "on", "off"],
     "Attention computed in tiles without the full score matrix: less memory traffic, faster prefill, needed for a "
     "quantized V cache. auto lets the engine decide per device (Vulkan, CUDA, ROCm, Metal and CPU have kernels).",
     {"llama": ["-fa", "--flash-attn"]}),
    ("threads_batch", "Speed", "Threads for prompt processing", "int", None, None,
     "Threads while reading the prompt (compute-bound; more helps). Generation threads are set above.",
     {"llama": ["-tb", "--threads-batch"]}),
    ("ubatch", "Speed", "Physical batch (ubatch)", "int", None, None,
     "Tokens computed at once inside a batch. Larger is faster prefill and more memory; 512 is the usual default.",
     {"llama": ["-ub", "--ubatch-size"]}),
    ("parallel", "Speed", "Parallel requests (slots)", "int", None, None,
     "Requests served at the same time; the context is shared between them.",
     {"llama": ["-np", "--parallel"], "vllm": ["--max-num-seqs"]}),
    ("poll", "Speed", "Busy-wait level (0-100)", "int", None, None,
     "How hard idle threads spin waiting for work. Lower saves power and heat on laptops; higher cuts latency.",
     {"llama": ["--poll"]}),
    ("prio", "Speed", "Process priority", "choice", None, ["-1", "0", "1", "2", "3"],
     "-1 low, 0 normal, 1 medium, 2 high, 3 realtime.", {"llama": ["--prio"]}),
    # ---------------------------------------------------------------- memory / KV cache
    ("cache_type_k", "Memory and KV cache", "K cache type", "choice", None, KV_TYPES,
     "Precision of the attention keys kept per token. q8_0 halves the cache with little loss; q4_0 quarters it "
     "with more. Long contexts on small GPUs are where this matters.",
     {"llama": ["-ctk", "--cache-type-k"]}),
    ("kv_cache_dtype", "Memory and KV cache", "KV cache type", "choice", None, ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
     "vLLM: fp8 halves the KV cache.", {"vllm": ["--kv-cache-dtype"]}),
    ("cache_type_v", "Memory and KV cache", "V cache type", "choice", None, KV_TYPES,
     "Precision of the attention values. A quantized V cache needs flash attention.",
     {"llama": ["-ctv", "--cache-type-v"]}),
    ("no_kv_offload", "Memory and KV cache", "Keep the KV cache in system RAM", "bool", False, None,
     "Frees GPU memory for layers at the cost of speed.", {"llama": ["-nkvo", "--no-kv-offload"]}),
    ("load_mode", "Memory and KV cache", "Loading mode", "choice", None,
     ["auto", "mmap", "mlock", "mmap+mlock", "none"],
     "mmap maps the file (fast start, pages can be evicted); mlock keeps the model resident in RAM; none reads it "
     "in. Newer llama.cpp; older builds have the two options below instead.", {"llama": ["-lm", "--load-mode"]}),
    ("mlock", "Memory and KV cache", "Lock the model in RAM", "bool", False, None,
     "Prevents the OS from paging weights out (needs enough RAM and permission).", {"llama": ["--mlock"]}),
    ("no_mmap", "Memory and KV cache", "Load into memory instead of mapping the file", "bool", False, None,
     "Slower to start; can help on network or slow disks and with some drivers.", {"llama": ["--no-mmap"]}),
    ("gpu_memory_utilization", "Memory and KV cache", "GPU memory share", "float", None, None,
     "vLLM: fraction of each GPU's memory it may use (0.9 by default).", {"vllm": ["--gpu-memory-utilization"]}),
    # ---------------------------------------------------------------- long context
    ("keep", "Long context", "Tokens to keep when the context fills", "int", None, None,
     "With context shift: the first N tokens (the system prompt, the attention sinks) always stay; the oldest "
     "middle tokens are dropped. -1 keeps the whole initial prompt. Together this is StreamingLLM-style "
     "streaming: memory and speed stay flat in long chats.", {"llama": ["--keep"]}),
    ("context_shift", "Long context", "Context shift", "bool", False, None,
     "When the context is full, drop old tokens and continue instead of stopping.",
     {"llama": ["--context-shift"]}),
    ("cache_reuse", "Long context", "Prompt cache reuse (tokens)", "int", None, None,
     "Reuse cached prompt chunks of at least this many tokens across requests (shifting the KV cache), so a "
     "repeated system prompt or document is not processed again. 256 is a common value.",
     {"llama": ["--cache-reuse"]}),
    ("cache_ram", "Long context", "Prompt cache in RAM (MiB)", "int", None, None,
     "Room in system RAM for cached prompts across slots; -1 no limit, 0 off.", {"llama": ["-cram", "--cache-ram"]}),
    ("prefix_caching", "Long context", "Prefix caching", "bool", False, None,
     "vLLM: reuse the KV cache of identical prompt prefixes.", {"vllm": ["--enable-prefix-caching"]}),
    ("chunked_prefill", "Long context", "Chunked prefill", "bool", False, None,
     "vLLM: split long prompts into chunks scheduled alongside generation, avoiding memory spikes and long "
     "stalls. (llama.cpp always processes prompts in batches of the batch size.)",
     {"vllm": ["--enable-chunked-prefill"]}),
    # ---------------------------------------------------------------- speculative decoding
    ("spec_type", "Speculative decoding", "Method", "choice", None, SPEC_TYPES,
     "draft-simple: a small draft model proposes tokens the big one verifies in one pass. ngram-*: no draft "
     "model; proposals come from n-grams already in the context (good for code and edits, zero extra memory). "
     "eagle3 / mtp: draft heads trained for the model, where the model ships them.",
     {"llama": ["--spec-type"]}),
    ("draft_model", "Speculative decoding", "Draft model (GGUF)", "text", None, None,
     "A small model with the same tokenizer (e.g. the 0.5B of the same family).",
     {"llama": ["-md", "--spec-draft-model", "--model-draft"]}),
    ("draft_max", "Speculative decoding", "Draft tokens, at most", "int", None, None,
     "Tokens proposed per step. More pays off when most are accepted.",
     {"llama": ["--spec-draft-n-max", "--draft-max"]}),
    ("draft_min", "Speculative decoding", "Draft tokens, at least", "int", None, None, "",
     {"llama": ["--spec-draft-n-min", "--draft-min"]}),
    ("draft_p_min", "Speculative decoding", "Minimum draft confidence", "float", None, None,
     "Stop drafting when the draft model is less sure than this (0-1).",
     {"llama": ["--spec-draft-p-min", "--draft-p-min"]}),
    ("draft_ngl", "Speculative decoding", "Draft model GPU layers", "int", None, None, "",
     {"llama": ["-ngld", "--spec-draft-ngl", "--gpu-layers-draft"]}),
    ("draft_device", "Speculative decoding", "Draft model device", "text", None, None,
     "Put the draft model on another device, e.g. the iGPU (Vulkan1) while the big model runs on the dGPU.",
     {"llama": ["-devd", "--spec-draft-device", "--device-draft"]}),
    ("speculative_config", "Speculative decoding", "Speculative config (JSON)", "text", None, None,
     'vLLM: e.g. {"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4} or '
     '{"model": "org/draft", "num_speculative_tokens": 5}.', {"vllm": ["--speculative-config"]}),
    # ---------------------------------------------------------------- devices
    ("device", "Devices", "Devices", "text", None, None,
     "Which devices hold layers, e.g. Vulkan0,Vulkan1. With the Vulkan build, an NVIDIA dGPU and an AMD iGPU are "
     "both Vulkan devices and can share one model. `llama-server --list-devices` shows the names.",
     {"llama": ["-dev", "--device"]}),
    ("rpc", "Devices", "RPC servers (other machines)", "text", None, None,
     "Split one model across machines: host:port of llama.cpp rpc-server instances (comma-separated), whose "
     "devices then hold layers like local ones. Use only on a trusted network or a VPN; rpc-server has no "
     "authentication.", {"llama": ["--rpc"]}),
    ("tensor_parallel", "Devices", "Tensor parallel GPUs", "int", None, None,
     "vLLM: shard every layer across N GPUs (needs fast links; do not use across vendors).",
     {"vllm": ["--tensor-parallel-size", "-tp"]}),
    ("pipeline_parallel", "Devices", "Pipeline parallel stages", "int", None, None,
     "vLLM: split the layers into N consecutive stages, one handoff per token. The better choice when devices "
     "differ in speed or are linked by a slow bus.", {"vllm": ["--pipeline-parallel-size", "-pp"]}),
    # ---------------------------------------------------------------- CPU placement
    ("cpu_range", "CPU placement", "CPU cores (range)", "text", None, None,
     "Pin threads to these cores, e.g. 0-7. On SMT CPUs the physical cores are often the even numbers or the "
     "first half; pinning keeps threads on one CCX and its L3 cache.", {"llama": ["-Cr", "--cpu-range"]}),
    ("cpu_mask", "CPU placement", "CPU mask (hex)", "text", None, None,
     "Exact cores as a hex mask, e.g. 0x5555 for every other core (physical cores, no SMT siblings).",
     {"llama": ["-C", "--cpu-mask"]}),
    ("cpu_strict", "CPU placement", "Strict placement", "bool", False, None,
     "One thread per listed core, no migration.", {"llama": ["--cpu-strict"]}),
    ("numa", "CPU placement", "NUMA", "choice", None, ["distribute", "isolate", "numactl"],
     "Multi-socket and some Threadripper/EPYC systems: spread over nodes, stay on the starting node, or follow "
     "the numactl wrapper below.", {"llama": ["--numa"]}),
    # ---------------------------------------------------------------- vLLM-specific
    ("dtype", "vLLM", "Weights dtype", "choice", None, ["auto", "half", "bfloat16", "float16", "float32"], "",
     {"vllm": ["--dtype"]}),
    ("quantization", "vLLM", "Quantization", "choice", None, ["awq", "gptq", "fp8", "bitsandbytes", "gguf"],
     "Pre-quantized checkpoints (AWQ, GPTQ...) or on-the-fly fp8.", {"vllm": ["--quantization", "-q"]}),
    ("max_model_len", "vLLM", "Max context", "int", None, None, "", {"vllm": ["--max-model-len"]}),
    ("enforce_eager", "vLLM", "No CUDA/HIP graphs", "bool", False, None,
     "vLLM captures graphs by default (less CPU launch overhead per token); turn off to save memory or debug.",
     {"vllm": ["--enforce-eager"]}),
]
OPTION = {o[0]: o for o in OPTIONS}

# Environment for the engine process, by name. Only these names are accepted
# from the UI; each is documented. (The process inherits the rest unchanged.)
ENV_OPTIONS = {
    "VLLM_USE_TRITON_FLASH_ATTN": "ROCm + vLLM: 0 uses the Composable Kernel flash attention instead of Triton "
                                  "(often faster prefill on Instinct and RDNA3).",
    "VLLM_ROCM_USE_AITER": "ROCm + vLLM: 1 enables AMD's AITER kernels where the build has them.",
    "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "ROCm + vLLM multi-GPU: FP/INT8/INT6/INT4 quick all-reduce.",
    "HSA_ENABLE_SDMA": "ROCm: 0 or 1, whether copies use the SDMA engines (0 works around transfer hangs on "
                       "some consumer cards).",
    "HSA_OVERRIDE_GFX_VERSION": "ROCm: run on a GPU the runtime does not list, as a close one (e.g. 10.3.0, 11.0.0).",
    "HIP_VISIBLE_DEVICES": "ROCm: which GPUs this process sees.",
    "CUDA_VISIBLE_DEVICES": "CUDA: which GPUs this process sees.",
    "GGML_VK_VISIBLE_DEVICES": "Vulkan (llama.cpp): which Vulkan devices this process sees, e.g. 0,1.",
    "GGML_VK_PREFER_HOST_MEMORY": "Vulkan: 1 places buffers in host memory (iGPUs).",
    "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "CUDA: 1 lets the model spill into system RAM instead of failing.",
    "OMP_NUM_THREADS": "CPU threads for OpenMP builds.",
}
ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]{1,63}")

# Launch wrappers: a fixed program and arguments built from validated fields,
# never a free-form shell string.
WRAPPERS = {
    "numactl": "numactl --cpunodebind=N --membind=N: keep threads and memory on one NUMA node",
    "taskset": "taskset -c CORES: pin the process to cores (Linux)",
}


def probe(binary: str, engine: str = "llama", timeout: float = 20) -> set[str]:
    """Flags the installed engine accepts, from its --help (cached per file and mtime)."""
    if not binary:
        return set()
    key = (binary, engine, _mtime(binary))
    if key in _CACHE:
        return _CACHE[key]
    cmd = [binary, "--help"] if engine == "llama" else [*shlex.split(binary), "serve", "--help=all"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        text = r.stdout + r.stderr
    except (OSError, subprocess.TimeoutExpired):
        text = ""
    flags = set(re.findall(r"(?<![\w-])(--?[A-Za-z][\w-]*)", text))
    _CACHE[key] = flags
    return flags


_CACHE: dict = {}


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(shlex.split(path)[0] if " " in path else path)
    except (OSError, IndexError):
        return 0.0


def spelling(key: str, engine: str, supported: set[str] | None) -> str | None:
    """The flag for `key` this engine build understands; with no probe, the preferred one."""
    flags = OPTION[key][7].get(engine) or []
    if not flags:
        return None
    if not supported:
        return flags[0]
    return next((f for f in flags if f in supported), None)


def _value(key: str, v):
    kind = OPTION[key][3]
    if kind in ("bool",):
        return bool(v) and str(v).lower() not in ("0", "false", "off", "no")
    if v is None or v == "":
        return None
    if kind == "int":
        return int(float(v))
    if kind == "float":
        return float(v)
    if kind == "choice":
        v = str(v)
        if v not in OPTION[key][5]:
            raise ValueError(f"{OPTION[key][2]}: {v!r} is not one of {', '.join(OPTION[key][5])}")
        return v
    v = str(v).strip()
    if key in ("cpu_range",) and v and not re.fullmatch(r"\d+(-\d+)?(,\d+(-\d+)?)*", v):
        raise ValueError("CPU cores look like 0-7 or 0,2,4,6")
    if key in ("cpu_mask",) and v and not re.fullmatch(r"(0x)?[0-9a-fA-F]+", v):
        raise ValueError("CPU mask is hexadecimal, like 0xff")
    if key == "rpc" and v and not re.fullmatch(r"[\w.\[\]:-]+:\d+(,[\w.\[\]:-]+:\d+)*", v):
        raise ValueError("RPC servers look like 192.168.1.20:50052,host2:50052")
    if key in ("device", "draft_device") and v and not re.fullmatch(r"[\w.:-]+(,[\w.:-]+)*", v):
        raise ValueError("devices look like Vulkan0,Vulkan1 or CUDA0")
    if key == "speculative_config" and v:
        json.loads(v)
    return v or None


def build_args(settings: dict, engine: str = "llama", supported: set[str] | None = None) -> tuple[list, list]:
    """-> (argv fragment, notes). Only options that are set are passed; an option
    this engine build lacks is left out and named in the notes."""
    args, notes = [], []
    for key, _g, label, kind, default, _c, _h, engines in OPTIONS:
        if key not in settings or engine not in engines:
            continue
        v = _value(key, settings[key])
        if v is None or v is False or (default is not None and v == default and kind != "choice"):
            continue
        if key == "flash_attn" and v == "auto":
            continue                           # the engine's own default
        flag = spelling(key, engine, supported)
        if flag is None:
            notes.append(f"{label}: this {engine} build has no such option; update it, or leave the option unset")
            continue
        if key == "draft_model" and not os.path.exists(v):
            raise ValueError(f"draft model not found: {v}")
        args += [flag] if kind == "bool" else [flag, str(v)]
    return args, notes


def build_env(settings: dict) -> dict:
    env = {}
    for k, v in (settings.get("env") or {}).items():
        if not ENV_NAME.fullmatch(str(k)) or k not in ENV_OPTIONS:
            raise ValueError(f"environment variable {k} is not one of the documented engine settings")
        if not re.fullmatch(r"[\w.,:-]{0,64}", str(v)):
            raise ValueError(f"{k}: unexpected characters in the value")
        env[k] = str(v)
    return env


def wrapper(settings: dict) -> list[str]:
    """The launch wrapper's argv prefix (numactl / taskset), from validated fields."""
    w = settings.get("wrap") or ""
    if not w:
        return []
    if w == "numactl":
        node = str(settings.get("numa_node", "0"))
        if not node.isdigit():
            raise ValueError("NUMA node is a number")
        return ["numactl", f"--cpunodebind={node}", f"--membind={node}"]
    if w == "taskset":
        cores = str(settings.get("taskset_cores") or "")
        if not re.fullmatch(r"\d+(-\d+)?(,\d+(-\d+)?)*", cores):
            raise ValueError("taskset cores look like 0,2,4,6 or 0-7")
        return ["taskset", "-c", cores]
    raise ValueError(f"unknown launch wrapper {w}")


def schema(engine: str = "llama", supported: set[str] | None = None) -> dict:
    """What a UI needs: options grouped, each with its help and whether this build has it."""
    groups: dict = {}
    for key, group, label, kind, default, choices, help_, engines in OPTIONS:
        if engine not in engines:
            continue
        groups.setdefault(group, []).append({
            "key": key, "label": label, "kind": kind, "default": default, "choices": choices, "help": help_,
            "flag": spelling(key, engine, supported) or engines[engine][0],
            "available": spelling(key, engine, supported) is not None})
    return {"engine": engine, "probed": bool(supported), "groups": groups,
            "env": ENV_OPTIONS, "wrappers": WRAPPERS}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--engine", choices=["llama", "vllm"], default="llama")
    p.add_argument("--server", help="the engine binary to check (llama-server path, or the vllm command)")
    p.add_argument("--args", help="settings as JSON: print the argv fragment they produce")
    a = p.parse_args(argv)
    sup = probe(a.server, a.engine) if a.server else None
    if a.args:
        args, notes = build_args(json.loads(a.args), a.engine, sup)
        print(shlex.join(args))
        for n in notes:
            print("note:", n, file=sys.stderr)
        return 0
    sc = schema(a.engine, sup)
    for g, opts in sc["groups"].items():
        print(f"\n{g}")
        for o in opts:
            mark = "" if o["available"] else "   (not in this build)"
            print(f"  {o['key']:<22} {o['flag']:<22} {o['label']}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
