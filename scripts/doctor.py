#!/usr/bin/env python3
"""
Check what on this machine is ready, and say what to do next.

Twenty scripts with different prerequisites is hard to start. This walks them in
dependency order and reports what works, what is missing, and what that blocks,
so the first thing you learn is not a traceback four stages in.

    python scripts/doctor.py
    python scripts/doctor.py --gguf ~/models/model-Q6_K.gguf
    python scripts/doctor.py --json           # machine-readable

Exit status is 0 if the core path is usable, 1 if something required is missing.
"""

import argparse
import gguf_utils
import importlib
import json
import os
import shutil
import socket
import subprocess
import sys

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"
MARK = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL ", SKIP: " --   "}

results = []


def add(stage, name, status, detail="", fix="", blocks=""):
    results.append(dict(stage=stage, name=name, status=status, detail=detail,
                        fix=fix, blocks=blocks))
    return status


def have(mod, min_version=None):
    try:
        m = importlib.import_module(mod)
    except Exception:
        return None, None
    v = getattr(m, "__version__", None)
    return m, v


def which(*names):
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def port_free(port):
    with socket.socket() as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) != 0


# --------------------------------------------------------------------- checks

def check_python():
    v = sys.version_info
    s = OK if v >= (3, 10) else FAIL
    add("core", "python >= 3.10", s, f"{v.major}.{v.minor}.{v.micro}",
        "install a newer python", "everything")


def check_packages():
    required = {
        "numpy": "arrays, everywhere",
        "requests": "all HTTP-based evaluation",
        "tqdm": "progress bars",
    }
    optional = {
        "sklearn": ("classifier.py", "pip install scikit-learn"),
        "transformers": ("tokenizer and chat templates", "pip install transformers"),
        "torch": ("the PyTorch extraction path and merging", "see install.sh"),
        "gguf": ("GGUF metadata, expert sweeps, exact LoRA export",
                 "pip install gguf"),
        "pandas": ("reading the TriviaQA parquet", "pip install pandas pyarrow"),
        "openai": ("stage 1 collection", "pip install openai"),
        "fastplotlib": ("the 2D/3D explorer", "pip install fastplotlib"),
        "mergekit": ("merging", "pip install -e /path/to/mergekit"),
    }
    for mod, why in required.items():
        m, v = have(mod)
        add("core", mod, OK if m else FAIL, v or "", f"pip install {mod}", why)
    for mod, (why, fix) in optional.items():
        m, v = have(mod)
        add("packages", mod, OK if m else WARN, v or "", fix, why)


def check_cuda(gpus=None, torch=None):
    """NVIDIA: every GPU, and whether torch, the toolkit and the driver fit them."""
    import cuda_info
    gpus = cuda_info.query_gpus() if gpus is None else gpus
    if not gpus:
        return
    a = cuda_info.advise(gpus, cuda_info.nvcc_version())
    for g in gpus:
        add("cuda", f"GPU {g['index']}", OK, f"{g['name']} sm_{g['cc']} {g['memory_total'] / 2**30:.0f} GiB, "
            f"train in {a['precision'][g['index']]['dtype']}")
    if torch is not None and torch.cuda.is_available() and not getattr(torch.version, "hip", None):
        archs = torch.cuda.get_arch_list()
        missing = sorted({f"sm_{g['cc']}" for g in gpus if f"sm_{g['cc']}" not in archs})
        t = a["torch"]
        fix = f"pip install '{t['spec']}'" + (f" --index-url {t['index']}" if t["index"] else "")
        add("cuda", "torch kernels", FAIL if missing else OK,
            f"no kernels for {', '.join(missing)} in torch {torch.__version__}" if missing
            else f"torch {torch.__version__} covers every GPU", fix, "PyTorch on these GPUs")
    lc = a["llama_cpp"]
    if lc.get("error"):
        add("cuda", "CUDA toolkit", FAIL, f"nvcc {a.get('nvcc')}", lc["error"], "building llama.cpp for these GPUs")
    elif a.get("nvcc"):
        add("cuda", "CUDA toolkit", OK, f"nvcc {a['nvcc']}; build with --cuda-arch '{lc['cmake_archs']}'")
    else:
        add("cuda", "CUDA toolkit", WARN, "nvcc not found", "install CUDA 12.9 (pre-Turing) or newer, or use "
            "docker/llama-cuda.Dockerfile", "building llama.cpp with CUDA")
    for n in a.get("notes", []):
        add("cuda", "note", WARN if "newer than" in n else OK, n)


def check_vulkan_icds(icds=None):
    """A driver failed to load: try each installed ICD alone and say which."""
    import accelerators
    icds = accelerators.vulkan_icds() if icds is None else icds
    bad = [i for i in icds if not i["ok"]]
    good = [i for i in icds if i["gpu"]] or [i for i in icds if i["ok"]]
    if not bad:
        add("compute", "vulkan ICD conflict", WARN, "a driver failed to initialise and was skipped",
            "run VK_LOADER_DEBUG=driver vulkaninfo --summary to see which", "llama.cpp may pick the wrong driver")
        return
    names = ", ".join(os.path.basename(i["path"]) + (f" (package {i['owner']})" if i.get("owner") else "")
                      for i in bad)
    pin = good[0]["path"] if good else None
    fix = []
    if pin:
        fix.append(f"use the working one: export VK_DRIVER_FILES={pin}  (or set it as the environment of the "
                   f"Vulkan devices in Studio: Setup -> Hardware)")
    owners = sorted({i["owner"] for i in bad if i.get("owner")})
    if owners:
        fix.append("or remove the broken driver's package: " + ", ".join(owners))
    add("compute", "vulkan ICD conflict", WARN, f"fails to load: {names}",
        "; ".join(fix) or "remove the failing manifest(s) from /usr/share/vulkan/icd.d",
        "llama.cpp and wgpu may pick the broken driver")


_DEVS = {}


def _devices():
    """Accelerators as Studio's worker pool sees them (scripts/accelerators.py)."""
    if "list" not in _DEVS:
        try:
            import accelerators
            _DEVS["list"] = accelerators.detect(accelerators.load_config())
        except Exception:
            _DEVS["list"] = []
    return _DEVS["list"]


def check_devices():
    for d in _devices():
        mem = "memory unknown" if d["memory_total"] is None else \
            f"{(d['memory_free'] or 0) / 2**30:.1f} of {d['memory_total'] / 2**30:.1f} GiB free"
        if not d["enabled"]:
            add("devices", d["id"], SKIP, f"{d['name']}, off" + (f" ({d['note']})" if d.get("note") else ""))
        else:
            add("devices", d["id"], OK, f"{d['name']}, {mem}" + (f"; {d['note']}" if d.get("note") else ""))


def check_compute():
    torch, _ = have("torch")
    try:
        check_cuda(torch=torch)
    except Exception as e:
        add("cuda", "nvidia-smi", WARN, str(e)[:80])
    if torch is None:
        add("compute", "torch device", SKIP, "", "", "PyTorch extraction")
    else:
        try:
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                hip = getattr(torch.version, "hip", None)
                add("compute", "torch GPU", OK, f"{name}"
                    + (f" (ROCm {hip})" if hip else ""))
            else:
                gpus = [d for d in _devices() if d["backend"] != "cpu" and d["enabled"]]
                amd = [d for d in gpus if d.get("vendor") == "amd"]
                if amd:
                    fix = (f"{amd[0]['name']} found: use the llama.cpp path (Vulkan or ROCm), "
                           "which needs no PyTorch GPU support; see docs/HARDWARE.md")
                elif gpus:
                    fix = (f"{gpus[0]['name']} found but torch cannot use it: re-run ./install.sh "
                           "(it picks the wheel for this GPU), or use the llama.cpp path")
                else:
                    fix = "no GPU found; CPU works, and the llama.cpp path is the faster way to extract"
                add("compute", "torch GPU", WARN, "CPU only", fix, "PyTorch extraction will be slow")
        except Exception as e:
            add("compute", "torch GPU", WARN, str(e)[:60])

    vk = which("vulkaninfo")
    if not vk:
        add("compute", "vulkan", WARN, "vulkaninfo not found",
            "install vulkan-tools", "llama.cpp Vulkan backend")
    else:
        try:
            r = subprocess.run([vk, "--summary"], capture_output=True,
                               text=True, timeout=25)
            out = r.stdout + r.stderr
            dev = next((l.split("=", 1)[-1].strip() for l in out.splitlines()
                        if "deviceName" in l), None)
            if dev:
                add("compute", "vulkan device", OK, dev)
            else:
                add("compute", "vulkan device", FAIL, "no device enumerated",
                    "check your ICD and driver", "llama.cpp Vulkan")
            if "Skipping this driver" in out or "-3 from call" in out or "Failed to CreateInstance" in out:
                check_vulkan_icds()
        except Exception as e:
            add("compute", "vulkan", WARN, str(e)[:60])

    fpl, _ = have("fastplotlib")
    if fpl is None:
        add("compute", "wgpu adapters", SKIP, "", "", "the explorer")
    else:
        try:
            n = len(fpl.enumerate_adapters())
            add("compute", "wgpu adapters", OK if n else FAIL, f"{n} found",
                "install a Vulkan driver, or lavapipe for software rendering",
                "viz/explore.py")
        except Exception as e:
            add("compute", "wgpu adapters", WARN, str(e)[:60])


def check_llama(root):
    root = os.path.expanduser(root)
    if not os.path.isdir(root):
        add("llama.cpp", "source tree", WARN, f"{root} not found",
            "scripts/build_llama_tools.sh --backend vulkan  (or cpu, cuda, hip, metal), or point "
            "OC_LLAMA at an existing checkout",
            "Studio, extraction, serving, merging to GGUF")
        return
    add("llama.cpp", "source tree", OK, root)
    bins = {
        "llama-server": "serving, alpha sweeps, Studio",
        "llama-cett-dump": "activation extraction (this repo's tool)",
        "llama-export-lora": "merging an adapter for LM Studio",
        "llama-quantize": "quantizing a merged model",
    }
    for b, why in bins.items():
        p = os.path.join(root, "build", "bin", b)
        found = os.path.exists(p) or which(b)
        status = OK if found else WARN
        fix = (f"cmake --build {root}/build --target {b} -j"
               if b != "llama-cett-dump" else
               f"scripts/build_llama_tools.sh --dir {root}  (adds cett-dump and rebuilds)")
        add("llama.cpp", b, status, p if found else "", fix, why)
    for s in ("convert_hf_to_gguf.py", "convert_lora_to_gguf.py"):
        add("llama.cpp", s, OK if os.path.exists(os.path.join(root, s)) else WARN,
            "", "update your llama.cpp checkout", "GGUF conversion")


def check_model(gguf):
    if not gguf:
        add("model", "OC_GGUF", WARN, "not set",
            "source env.sh, or pass --gguf", "everything model-specific")
        return
    if not os.path.exists(gguf):
        add("model", "file", FAIL, gguf,
            "check the volume is mounted (env.sh prints the udisksctl command)",
            "everything model-specific")
        return
    size = os.path.getsize(gguf)
    add("model", "file", OK, f"{os.path.basename(gguf)} "
        f"({size / 2**30:.1f} GiB)")
    g, _ = have("gguf")
    if g is None:
        add("model", "metadata", SKIP, "", "pip install gguf", "expert sweeps")
        return
    try:
        r = g.GGUFReader(gguf)

        def kv(k):
            return gguf_utils.read_kv(r, k)

        arch = kv("general.architecture")
        n_layers = kv(f"{arch}.block_count") if arch else None
        n_exp = kv(f"{arch}.expert_count") if arch else None
        add("model", "architecture", OK if arch else WARN,
            f"{arch}, {n_layers} layers"
            + (f", MoE {kv(f'{arch}.expert_used_count')}/{n_exp}" if n_exp
               else ", dense"))
        if n_layers:
            add("model", "OC_LAYERS", OK, f"pass --n-layers {n_layers} to "
                                          "llama-cett-dump")
        names = {t.name for t in r.tensors}
        dense = any(n.startswith("blk.0.ffn_down.") for n in names)
        moe = any(n.startswith("blk.0.ffn_down_exps.") for n in names)
        add("model", "ffn_down tensors", OK if (dense or moe) else FAIL,
            "dense" if dense else ("MoE (expert tensors)" if moe else "none"),
            "", "extraction and exact LoRA export")
    except Exception as e:
        add("model", "metadata", WARN, str(e)[:70])


def check_data(root):
    # Stage numbers follow docs/PIPELINE.md. Stages 6 (visualise) and 8 (apply
    # at runtime) write no files, so there is nothing to check for them.
    stages = [
        ("data/consistency_samples.jsonl", "1 collect answers",
         "scripts/collect_responses_lmstudio.py  (needs data/TriviaQA, see data/README.md)"),
        ("data/answer_tokens.jsonl", "2 tag answer tokens", "scripts/make_answer_tokens.py"),
        ("data/train_qids.json", "3 balanced split", "scripts/sample_balanced_ids.py"),
        ("data/activations/neuron_index.json", "4 extract CETT",
         "scripts/extract_activations_gguf.py"),
        ("models/h_neurons.json", "5 find H-Neurons", "scripts/classifier.py"),
        ("profiles", "7 tune the scale", "scripts/tune_scale_server.py"),
        ("adapters", "9 export", "scripts/export_lora.py"),
    ]
    first_missing = None
    for path, label, how in stages:
        p = os.path.join(root, path)
        done = os.path.exists(p) and (not os.path.isdir(p) or os.listdir(p))
        add("pipeline", label, OK if done else SKIP,
            path if done else "", how if not done else "", "")
        if not done and first_missing is None:
            first_missing = (label, how)
    return first_missing


def check_ports():
    for port, what in ((8080, "llama-server"), (7860, "dashboard"),
                       (7870, "Studio")):
        free = port_free(port)
        add("ports", str(port), OK if free else WARN,
            "free" if free else "in use", "", what)


# ---------------------------------------------------------------------- main

def studio_section(r):
    """Where in Studio a problem is fixed (the same places the web pages link to)."""
    text = f"{r['stage']} {r['name']} {r['fix']}".lower()
    for keys, where in ((("llama-server", "cett", "llama.cpp", "build_llama"), "/setup#build"),
                        (("vulkan", "icd", "rocm", "cuda", "gpu"), "/setup#hardware"),
                        (("pip install", "package", "python"), "/setup#requirements"),
                        (("workspace", "models folder"), "/settings#workspaces")):
        if any(k in text for k in keys):
            return where
    return None


def main():
    import oc_workspace
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", help="whose folders to check (default: the active workspace, Settings -> Workspaces)")
    p.add_argument("--gguf")
    p.add_argument("--llama")
    p.add_argument("--root")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    # Flags, then NS_* variables, then the workspace: no layout is assumed.
    if a.workspace:
        if a.workspace not in oc_workspace.names():
            print(f"no workspace {a.workspace!r}; have {', '.join(oc_workspace.names())}", file=sys.stderr)
            return 2
        os.environ["OC_WORKSPACE"] = a.workspace
    ws = oc_workspace.active()
    a.gguf = a.gguf or os.environ.get("OC_GGUF") or ws["gguf"]
    a.llama = a.llama or os.environ.get("OC_LLAMA") or ws["llama_dir"]
    a.root = a.root or os.environ.get("OC_ROOT") or ws["root"]
    add("core", "workspace", OK, f"{ws['name']} (root {ws['root']})")

    check_python()
    check_packages()
    check_compute()
    check_devices()
    check_llama(a.llama)
    check_model(a.gguf)
    first_missing = check_data(os.path.abspath(a.root))
    check_ports()

    if a.json:
        print(json.dumps(results, indent=2))
        return 0 if not any(r["status"] == FAIL for r in results) else 1

    stage = None
    for r in results:
        if r["stage"] != stage:
            stage = r["stage"]
            print(f"\n{stage.upper()}")
        line = f"  [{MARK[r['status']]}] {r['name']:<24}"
        if r["detail"]:
            line += f" {r['detail']}"
        print(line)
        if r["status"] in (WARN, FAIL):
            if r["blocks"]:
                print(f"           blocks: {r['blocks']}")
            if r["fix"]:
                print(f"           fix:    {r['fix']}")

    fails = [r for r in results if r["status"] == FAIL]
    warns = [r for r in results if r["status"] == WARN]
    print(f"\n{len(results)} checks: "
          f"{sum(1 for r in results if r['status'] == OK)} ok, "
          f"{len(warns)} warnings, {len(fails)} failures")

    print("\nNEXT")
    if fails:
        print(f"  Fix first: {fails[0]['name']} -- {fails[0]['fix']}")
        where = studio_section(fails[0])
        if where:
            print(f"  In Studio: http://127.0.0.1:7870{where}")
        return 1
    server = next((r["detail"] for r in results if r["name"] == "llama-server" and r["status"] == OK), None)
    print("  Use and compare models (Studio: chat, /v1 API, TestQA, Projects):")
    if server:
        print("    ./omnicortex            (Studio in your browser; set your models folder under Settings)")
        print(f"    or: python viz/studio.py --models-dir ~/models --server {server}")
    else:
        print("    first build llama-server:  scripts/build_llama_tools.sh --backend vulkan  (or cpu, cuda, hip, metal)")
    print("\n  Find H-Neurons in a model (docs/PIPELINE.md):")
    if first_missing:
        label, how = first_missing
        print(f"    stage {label} has no output yet:  python {how}")
        if label.startswith("4"):
            print("    This is the gate: if llama-cett-dump works on a toy prompt (docs/GETTING_STARTED.md),\n"
                  "    the whole local half works.")
    else:
        print("    every stage has output. Compare a suppressed model against the base:\n"
              "    python scripts/merge_eval.py --endpoint base=... --endpoint tuned=...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
