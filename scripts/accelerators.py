#!/usr/bin/env python3
"""
Every accelerator this machine has, whichever API reaches it: CUDA, ROCm,
Vulkan, Metal, and the CPU.

The director's worker pool places models on these. Each device has an id like
"rocm:0" or "vulkan:1", a backend, the memory a model can use, and which
llama-server build drives it. Laptops with AMD APUs are a first-class case:
an APU's dedicated VRAM is a small carve-out (often 512 MiB to 2 GiB), but
llama.cpp can also use GTT, the system memory the GPU maps, so usable memory is
VRAM plus GTT, shared with everything else the machine is doing.

Sources, each optional (a missing tool means that backend is not listed):
  nvidia-smi                      NVIDIA GPUs (cuda)
  /sys/class/drm/card*/device     AMD (and Intel) GPUs with VRAM and GTT sizes;
                                  works without ROCm installed
  rocminfo                        whether ROCm can drive the AMD GPUs (rocm),
                                  and their gfx targets
  vulkaninfo --summary            every Vulkan device (vulkan), including APUs
                                  that ROCm does not support
  macOS on Apple silicon          one Metal device with unified memory (metal)
  always                          the CPU (cpu)

The same GPU often shows up twice, natively and through Vulkan. Both entries
are listed; the Vulkan twin of a GPU that a native backend reaches is disabled
by default, so one GPU is not booked twice. Turn either on or off in the
hardware config:

    ~/.omnicortex/hardware.json
    {"servers": {"rocm": "~/llama.cpp/build-rocm/bin/llama-server",
                 "vulkan": "~/llama.cpp/build-vulkan/bin/llama-server"},
     "devices": {"rocm:0": {"enabled": true, "reserve_gib": 2,
                            "rocm_path": "/opt/rocm-6.2.4",
                            "server": "~/llama.cpp/build-rocm62/bin/llama-server",
                            "settings": {"ctx": 8192, "threads": 8},
                            "env": {"HSA_OVERRIDE_GFX_VERSION": "9.0.0"}},
                 "vulkan:0": {"enabled": false}},
     "manual": [{"id": "vulkan:2", "backend": "vulkan", "index": 2,
                 "name": "eGPU", "memory_total_gib": 8}]}

A backend with no server listed uses Studio's --server binary. A device can
name its own server and its own ROCm install ("rocm_path"): GPUs that need a
particular ROCm/HIP release get it, through ROCM_PATH, HIP_PATH and
LD_LIBRARY_PATH set for that worker only, while the rest of the machine keeps
whatever ROCm it has.

GPUs ROCm does not support are handled the way install.sh handles them. The
smaller RDNA2/RDNA3 dies run with the usual HSA_OVERRIDE_GFX_VERSION
(10.3.0 / 11.0.0), applied automatically. Vega-based APUs (gfx90c: Renoir,
Cezanne, Barcelo, e.g. Ryzen 5000U/7030U) can only masquerade as gfx900, which
can hang the GPU or compute wrong results, so their ROCm entry is off by default
and the Vulkan entry for the same GPU is used instead. Enable the ROCm entry in
the config to opt in (the 9.0.0 override is then applied); check it first with
scripts/try_rocm.sh.

On an APU the GPU and the CPU share system memory, so a model booked on one
reduces what the other can take.

AMD comes first: when several devices have room for a model, AMD GPUs are
used before others (then smaller before larger, discrete before shared
memory). "prefer": ["amd", "nvidia", "intel", "apple"] in the config changes the
order; vendors left out come after the listed ones.

    python scripts/accelerators.py            # what is here, and how it would be used
    python scripts/accelerators.py --json
"""
from __future__ import annotations

import argparse
import json
import glob
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

GIB = 1024 ** 3
BACKENDS = ("cuda", "rocm", "vulkan", "metal", "cpu")
DEFAULT_CONFIG = Path.home() / ".omnicortex" / "hardware.json"
VENDORS = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel", "0x106b": "apple", "0x13b5": "arm",
           "0x5143": "qualcomm"}
# Environment a device entry may set for its llama-server. Nothing else passes.
ENV_OK = re.compile(r"^(HSA_|HIP_|ROCR_|ROCM_|GGML_|CUDA_|MTL_|VK_|AMD_|RADV_)[A-Z0-9_]+$")
# Which variable makes a llama-server see only the chosen devices.
VISIBLE = {"cuda": "CUDA_VISIBLE_DEVICES", "rocm": "HIP_VISIBLE_DEVICES", "vulkan": "GGML_VK_VISIBLE_DEVICES"}
APU_VRAM_MAX = 4 * GIB
# Real gfx target -> (HSA_OVERRIDE_GFX_VERSION, safe enough to use by default)
GFX_OVERRIDE = {
    **{g: ("9.0.0", False) for g in ("gfx90c", "gfx902", "gfx909", "gfx90b")},     # Vega APUs: masquerade only
    **{g: ("10.3.0", True) for g in ("gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036")},
    **{g: ("11.0.0", True) for g in ("gfx1103",)},
}


def _run(cmd, timeout=15, env=None):
    if not shutil.which(cmd[0]):
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


# ------------------------------------------------------------------ parsers

def drm_cards(root: str = "/sys/class/drm") -> list[dict]:
    """GPUs from sysfs: vendor, PCI slot, VRAM and GTT totals/used (AMD
    exposes these on dGPUs and APUs alike), sorted by card number."""
    out = []
    base = Path(root)
    if not base.is_dir():
        return out
    for card in sorted(base.glob("card[0-9]*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
        if "-" in card.name:                      # card0-HDMI-A-1 and friends are connectors
            continue
        dev = card / "device"

        def rd(name):
            try:
                return (dev / name).read_text().strip()
            except OSError:
                return None

        vendor = (rd("vendor") or "").lower()
        if not vendor:
            continue

        def num(name):
            v = rd(name)
            return int(v) if v and v.isdigit() else None

        slot = None
        for line in (rd("uevent") or "").splitlines():
            if line.startswith("PCI_SLOT_NAME="):
                slot = line.split("=", 1)[1]
        out.append({"card": card.name, "vendor": VENDORS.get(vendor, vendor), "pci": slot,
                    "name": rd("product_name") or None,
                    "vram_total": num("mem_info_vram_total"), "vram_used": num("mem_info_vram_used"),
                    "gtt_total": num("mem_info_gtt_total"), "gtt_used": num("mem_info_gtt_used")})
    return out


def parse_vulkan_summary(text: str) -> list[dict]:
    """`vulkaninfo --summary` -> [{index, name, type, vendor, driver}]."""
    devs, cur = [], None
    for line in text.splitlines():
        m = re.match(r"^GPU(\d+):\s*$", line.strip())
        if m:
            cur = {"index": int(m.group(1))}
            devs.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"^\s*(\w+)\s*=\s*(.*)$", line)
        if m:
            k, v = m.group(1), m.group(2).strip()
            if k == "deviceName":
                cur["name"] = v
            elif k == "deviceType":
                cur["type"] = v.replace("PHYSICAL_DEVICE_TYPE_", "").lower()
            elif k == "vendorID":
                cur["vendor"] = VENDORS.get(v.lower(), v.lower())
            elif k == "driverName":
                cur["driver"] = v
    return devs


ICD_DIRS = ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d", "/usr/local/share/vulkan/icd.d")


def icd_owner(path: str) -> str | None:
    """The distro package that installed an ICD manifest, if the package manager knows."""
    for cmd, pick in ((["pacman", "-Qoq", path], lambda o: o.split()[0]),
                      (["dpkg", "-S", path], lambda o: o.split(":")[0]),
                      (["rpm", "-qf", "--qf", "%{NAME}", path], lambda o: o.split()[0])):
        out = _run(cmd, timeout=10)
        if out and out.strip():
            try:
                return pick(out.strip())
            except IndexError:
                continue
    return None


def vulkan_icds(dirs=ICD_DIRS, probe=None) -> list[dict]:
    """Every installed Vulkan driver manifest, tried on its own:
    [{path, ok, devices, owner}]. A machine with a broken ICD next to a working
    one (common after installing AMDVLK, a vendor driver or a Mesa variant)
    makes the loader skip drivers or pick the wrong one; trying each manifest
    alone says which. probe(path) -> vulkaninfo text, for tests."""
    out = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            if probe is not None:
                text = probe(path)
            else:
                env = {**os.environ, "VK_DRIVER_FILES": path, "VK_ICD_FILENAMES": path}
                text = _run(["vulkaninfo", "--summary"], timeout=20, env=env) or ""
            devs = [x.get("name") for x in parse_vulkan_summary(text) if x.get("name")]
            # llvmpipe is a CPU rasterizer: present, but no use for compute here.
            real = [x for x in devs if "llvmpipe" not in x.lower()]
            out.append({"path": path, "ok": bool(devs), "devices": devs, "gpu": bool(real),
                        "owner": None if devs or probe is not None else icd_owner(path)})
    return out


def parse_rocminfo(text: str) -> list[str]:
    """gfx targets of the GPU agents rocminfo lists, in HIP device order."""
    out = []
    for block in re.split(r"\n\*{5,}\s*\n", text):
        if re.search(r"Device Type:\s+GPU", block):
            m = re.search(r"\bName:\s+(gfx[0-9a-f]+)", block)
            if m:
                out.append(m.group(1))
    return out


# ------------------------------------------------------------------ detect

def _amd_memory(c: dict) -> tuple[int | None, int | None, bool]:
    """-> (total, free, is_apu) for an AMD card from sysfs. An APU (small VRAM
    carve-out, large GTT) can use both; a discrete card counts VRAM only."""
    vt, vu, gt, gu = c.get("vram_total"), c.get("vram_used") or 0, c.get("gtt_total"), c.get("gtt_used") or 0
    if vt is None:
        return None, None, False
    apu = vt <= APU_VRAM_MAX and (gt or 0) > 2 * vt
    if apu:
        return vt + gt, (vt - vu) + (gt - gu), True
    return vt, vt - vu, False


def _nvidia_noncoherent_patch(sysfs: str = "/sys/module/nvidia/parameters") -> str | None:
    """The m10-arm patch set (NVIDIA proprietary driver on non-cache-coherent
    arm64 PCIe, e.g. RK3588) adds module parameters; when it maps system
    memory uncached, CPU access to pinned host buffers is slow."""
    p = Path(sysfs) / "arm_force_uncached"
    try:
        v = p.read_text().strip()
    except OSError:
        return None
    if v in ("1", "Y"):
        return ("non-coherent PCIe driver patch active (system memory uncached). If prompt processing is slow, "
                "try env GGML_CUDA_NO_PINNED=1 for these devices; unverified on hardware")
    return "non-coherent PCIe driver patch present, uncached mode off"


def detect(config: dict | None = None, *, smi=None, drm=None, vulkan=None, rocm=None, system=None,
           ram=None, sysfs: str = "/sys/module/nvidia/parameters") -> list[dict]:
    """-> devices [{id, backend, index, name, vendor, memory_total, memory_free,
    unified, estimated, enabled, twin_of, server, settings, env, reserve}].
    The keyword arguments replace each probe's output (for tests, or another
    machine's readings); None means "probe this machine"."""
    cfg = config or {}
    devcfg = cfg.get("devices") or {}
    servers = {k: os.path.expanduser(v) for k, v in (cfg.get("servers") or {}).items() if k in BACKENDS}
    out = []

    # NVIDIA
    if smi is None:
        try:
            import cuda_info
            smi = cuda_info.query_gpus()
        except Exception:
            smi = []
    noncoherent = _nvidia_noncoherent_patch(sysfs)
    for g in smi:
        out.append({"id": f"cuda:{g['index']}", "backend": "cuda", "index": g["index"], "name": g["name"],
                    "note": noncoherent,
                    "vendor": "nvidia", "memory_total": g["memory_total"],
                    "memory_free": g.get("memory_free") or g["memory_total"], "unified": False,
                    "estimated": False, "pci": (g.get("bus_id") or "").lower()[-12:] or None})

    # AMD (and Intel) from sysfs; ROCm if rocminfo can see the AMD GPUs
    cards = drm_cards() if drm is None else drm
    amd = [c for c in cards if c["vendor"] == "amd"]
    if rocm is None:
        # With HSA_OVERRIDE_GFX_VERSION set, rocminfo reports the masquerade, not the chip.
        env = {k: v for k, v in os.environ.items() if k != "HSA_OVERRIDE_GFX_VERSION"}
        rocm = parse_rocminfo(_run(["rocminfo"], env=env) or "")
    rocm_targets = rocm
    if rocm_targets:
        for i, c in enumerate(amd[:len(rocm_targets)]):
            total, free, apu = _amd_memory(c)
            gfx = rocm_targets[i]
            override, safe = GFX_OVERRIDE.get(gfx, (None, True))
            note = None
            if override and not safe:
                note = (f"{gfx} is not supported by ROCm; it can only pose as gfx{override.replace('.', '')[:4]}, "
                        "which can hang or compute wrong results. Off by default; the Vulkan entry is used. "
                        "Enable it in the hardware config to try (check with scripts/try_rocm.sh).")
            elif override:
                note = f"{gfx} runs with HSA_OVERRIDE_GFX_VERSION={override}"
            out.append({"id": f"rocm:{i}", "backend": "rocm", "index": i,
                        "name": c.get("name") or f"AMD {gfx}" + (" APU" if apu else ""),
                        "vendor": "amd", "memory_total": total, "memory_free": free, "unified": apu,
                        "estimated": False, "pci": c.get("pci"), "gfx": gfx, "override": override,
                        "default_on": safe, "note": note})

    # Vulkan: everything the loader sees, memory borrowed from sysfs or nvidia-smi
    vk = parse_vulkan_summary(_run(["vulkaninfo", "--summary"]) or "") if vulkan is None else vulkan
    by_vendor = {"amd": list(amd), "intel": [c for c in cards if c["vendor"] == "intel"]}
    nv = list(smi)
    sysram = ram if ram is not None else _ram()
    for v in vk:
        if v.get("type") == "cpu":                # llvmpipe/lavapipe: the CPU entry covers it
            continue
        vend = v.get("vendor")
        total = free = None
        unified, estimated, pci = v.get("type") == "integrated_gpu", False, None
        if vend in by_vendor and by_vendor[vend]:
            c = by_vendor[vend].pop(0)
            pci = c.get("pci")
            if vend == "amd":
                total, free, unified = _amd_memory(c)
        elif vend == "nvidia" and nv:
            g = nv.pop(0)
            total, free = g["memory_total"], g.get("memory_free") or g["memory_total"]
            pci = (g.get("bus_id") or "").lower()[-12:] or None
        if total is None and unified and sysram:
            total, free, estimated = sysram[0] // 2, sysram[1] // 2, True     # shared memory: assume half
        twin = next((d["id"] for d in out if d["backend"] in ("cuda", "rocm") and
                     ((pci and d.get("pci") == pci) or (not pci and d["vendor"] == vend))), None)
        out.append({"id": f"vulkan:{v['index']}", "backend": "vulkan", "index": v["index"],
                    "name": v.get("name") or "Vulkan device", "vendor": vend, "memory_total": total,
                    "memory_free": free, "unified": unified, "estimated": estimated or total is None,
                    "pci": pci, "twin_of": twin})

    # Apple silicon: one Metal device sharing system memory
    sysname = system if system is not None else (platform.system(), platform.machine())
    if sysname[0] == "Darwin" and sysname[1] == "arm64" and sysram:
        # macOS lets the GPU wire roughly three quarters of memory by default.
        out.append({"id": "metal:0", "backend": "metal", "index": 0, "name": "Apple GPU", "vendor": "apple",
                    "memory_total": int(sysram[0] * 0.75), "memory_free": int(sysram[1] * 0.75),
                    "unified": True, "estimated": True, "pci": None})

    out.append({"id": "cpu:0", "backend": "cpu", "index": 0, "name": platform.processor() or "CPU",
                "vendor": "cpu", "memory_total": sysram[0] if sysram else None,
                "memory_free": sysram[1] if sysram else None, "unified": True, "estimated": False, "pci": None})

    for m in cfg.get("manual") or []:
        if m.get("backend") in BACKENDS and m.get("id") and not any(d["id"] == m["id"] for d in out):
            gib = m.get("memory_total_gib")
            out.append({"id": m["id"], "backend": m["backend"], "index": int(m.get("index", 0)),
                        "name": m.get("name") or m["id"], "vendor": m.get("vendor"),
                        "memory_total": int(gib * GIB) if gib else None,
                        "memory_free": int(gib * GIB) if gib else None, "unified": bool(m.get("unified")),
                        "estimated": True, "pci": None, "manual": True})

    for d in out:                                  # native entries first: twins depend on them
        c = devcfg.get(d["id"]) or {}
        d.setdefault("twin_of", None)
        d.setdefault("note", None)
        if d["twin_of"]:
            continue
        d["enabled"] = bool(c["enabled"]) if "enabled" in c else d.get("default_on", True)
    by_id = {d["id"]: d for d in out}
    for d in out:
        c = devcfg.get(d["id"]) or {}
        if d["twin_of"]:
            native = by_id[d["twin_of"]]
            # One GPU, two ways to reach it: use Vulkan only when the native entry is off.
            d["enabled"] = bool(c["enabled"]) if "enabled" in c else not native["enabled"]
            if d["enabled"] and native["enabled"]:
                d["note"] = f"same GPU as {native['id']}; both are on, so it can be booked twice"
        d["server"] = os.path.expanduser(c["server"]) if c.get("server") else servers.get(d["backend"])
        d["settings"] = {k: v for k, v in (c.get("settings") or {}).items()
                         if k in ("ngl", "ctx", "batch", "threads", "flash_attn", "cache_type", "parallel", "extra")}
        d["env"] = {k: str(v) for k, v in (c.get("env") or {}).items()
                    if ENV_OK.match(k) and "\n" not in str(v)}
        if d.get("override") and "HSA_OVERRIDE_GFX_VERSION" not in d["env"]:
            d["env"]["HSA_OVERRIDE_GFX_VERSION"] = d["override"]
        if c.get("rocm_path") and d["backend"] == "rocm":
            # A pinned ROCm/HIP release for this device's llama-server only.
            rp = os.path.expanduser(str(c["rocm_path"]))
            if os.path.isdir(rp):
                libs = [x for x in (os.path.join(rp, "lib"), os.path.join(rp, "lib64")) if os.path.isdir(x)]
                d["env"].update({"ROCM_PATH": rp, "HIP_PATH": rp})
                d["lib_path"] = libs
            else:
                d["enabled"] = False
                d["note"] = f"rocm_path {rp} does not exist; device off until it does"
        d["reserve"] = int(float(c.get("reserve_gib", 1 if d["unified"] else 0.5)) * GIB)
        if c.get("memory_total_gib"):
            d["memory_total"] = d["memory_free"] = int(float(c["memory_total_gib"]) * GIB)
            d["estimated"] = False
    return out


def _ram() -> tuple[int, int] | None:
    try:
        from hostcheck import Host
        h = Host()
        if h.ram:
            return int(h.ram), int(h.ram_available or h.ram)
    except Exception:
        pass
    return None


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path or DEFAULT_CONFIG).expanduser()
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def launch_env(devs: list[dict]) -> dict:
    """Environment for one llama-server driving these devices (one backend)."""
    backend = devs[0]["backend"]
    env = {}
    for d in devs:
        env.update(d.get("env") or {})
    libs = [x for d in devs for x in d.get("lib_path") or []]
    if libs:
        env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(libs + [x for x in os.environ.get(
            "LD_LIBRARY_PATH", "").split(":") if x]))
    var = VISIBLE.get(backend)
    if var:
        env[var] = ",".join(str(d["index"]) for d in devs)
    if backend == "rocm" and any(d.get("unified") for d in devs):
        # Lets HIP allocate from GTT on an APU instead of only the VRAM carve-out.
        env.setdefault("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "1")
    return env


PREFER = ("amd",)


def place(devices: list[dict], need: int, reserved: dict | None = None, allowed: list | None = None,
          prefer: tuple | list = PREFER) -> dict | None:
    """Where a model needing `need` bytes goes. -> {devices: [...], split:
    [weights] or None, ngl} or None.

    Preference: one GPU with room, by vendor preference (AMD first by
    default), then discrete before shared memory, then smallest that fits;
    then several GPUs of one backend (layer split); then the CPU. Devices
    with unknown memory are tried after the measured ones."""
    rank = {v: i for i, v in enumerate(prefer or ())}
    vrank = lambda d: rank.get(d.get("vendor"), len(rank))  # noqa: E731
    reserved = reserved or {}
    pool = [d for d in devices if d["enabled"] and (not allowed or d["id"] in allowed)]

    # APUs, Metal and the CPU draw on the same system memory: whatever is booked
    # on any of them is gone for all of them, and none can exceed what is free.
    shared = sum(b for i, b in reserved.items() if any(x["id"] == i and x["unified"] for x in devices))
    cpu = next((x for x in devices if x["backend"] == "cpu"), None)
    sys_free = cpu["memory_free"] if cpu else None

    def free(d):
        if d["memory_free"] is None:
            return None
        if d["unified"]:
            cap = d["memory_free"] if sys_free is None or d["backend"] == "cpu" else \
                min(d["memory_free"], sys_free + (d["memory_free"] if d["backend"] == "cpu" else 0))
            return cap - d["reserve"] - shared
        return d["memory_free"] - d["reserve"] - reserved.get(d["id"], 0)

    gpus = [d for d in pool if d["backend"] != "cpu"]
    fits = sorted((d for d in gpus if free(d) is not None and free(d) >= need),
                  key=lambda d: (vrank(d), d["unified"], free(d)))
    measured = [d for d in fits if not d.get("estimated")]
    if measured:
        return {"devices": [measured[0]], "split": None, "ngl": 99}
    order = sorted(("rocm", "vulkan", "cuda"), key=lambda b: min(
        [vrank(d) for d in gpus if d["backend"] == b] or [99]))
    for backend in order:
        group = sorted((d for d in gpus if d["backend"] == backend and (free(d) or 0) > 0),
                       key=lambda d: -free(d))
        chosen, tot = [], 0
        for d in group:
            chosen.append(d)
            tot += free(d)
            if tot >= need and len(chosen) > 1:
                chosen.sort(key=lambda d: d["index"])
                return {"devices": chosen, "split": [max(1, round(free(d) / GIB)) for d in chosen], "ngl": 99}
    if fits:          # only devices whose memory is a guess: after a split over measured ones
        return {"devices": [fits[0]], "split": None, "ngl": 99, "unmeasured": True}
    unknown = sorted((d for d in gpus if free(d) is None), key=vrank)
    if unknown:
        return {"devices": [unknown[0]], "split": None, "ngl": 99, "unmeasured": True}
    cpu = next((d for d in pool if d["backend"] == "cpu"), None)
    if cpu and (free(cpu) is None or free(cpu) >= need):
        return {"devices": [cpu], "split": None, "ngl": 0}
    return None


def describe(d: dict) -> str:
    mem = "memory unknown" if d["memory_total"] is None else \
        f"{(d['memory_free'] or 0) / GIB:.1f} of {d['memory_total'] / GIB:.1f} GiB free" + \
        (" (estimated)" if d["estimated"] else "")
    tags = [t for t, on in (("shared memory", d["unified"] and d["backend"] != "cpu"),
                            (d.get("gfx") or "", bool(d.get("gfx"))),
                            (f"same GPU as {d['twin_of']}", d.get("twin_of")), ("off", not d["enabled"])) if on]
    return f"{d['id']:<10} {d['name']}  {mem}" + (f"  [{', '.join(tags)}]" if tags else "") + \
        (f"\n           {d['note']}" if d.get("note") else "")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    devs = detect(load_config(a.config))
    if a.json:
        print(json.dumps(devs, indent=1))
    else:
        for d in devs:
            print(describe(d) + (f"\n           server: {d['server']}" if d.get("server") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
