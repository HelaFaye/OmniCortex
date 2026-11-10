#!/usr/bin/env python3
"""
Proxmox backend (docs/PROXMOX.md): run OmniCortex, either Studio or the remote agent, in an LXC container on a
Proxmox VE host, with the host's GPUs shared into it.

Why a container and not a VM: passing a GPU to a VM (VFIO) needs an IOMMU that reports cache coherency, and the
kernel refuses it otherwise. Boards whose PCIe is not cache-coherent (RK3588, Raspberry Pi 5) can never do it,
whatever the Proxmox build. A container shares the host's kernel and its GPU driver, needs no VFIO, and has no
virtualisation overhead. On x86 servers a VM with VFIO works too; this backend does the container either way.

    host     what this host has: CPU, board, coherency, GPUs and their device nodes, fast cores, PCIe window
    plan     the pct commands and container settings for a container (prints; changes nothing)
    apply    run that plan's pct commands (on the Proxmox host, as root, with --yes)
    check    inside the container: can OmniCortex open the GPU, and is the environment right

    python scripts/proxmox_lxc.py host
    python scripts/proxmox_lxc.py plan --ctid 101 --role agent --models /srv/models
    python scripts/proxmox_lxc.py apply --ctid 101 --role agent --models /srv/models --yes
    python scripts/proxmox_lxc.py check          # inside the container

The GPU driver stays on the host: amdgpu (with KFD for ROCm) or NVIDIA's kernel module, patched there if the board
needs it (the mi50-arm and m10-arm guides). The container holds only user space: Mesa or ROCm for AMD, and for NVIDIA
the same driver version's user-space libraries, installed with --no-kernel-module.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# device-tree compatibles of boards whose PCIe is not cache-coherent with the CPU, and their small 64-bit windows
NONCOHERENT = {"rockchip,rk3588": 1 << 30, "rockchip,rk3588s": 1 << 30, "brcm,bcm2712": None}
NVIDIA_NAMES = {"0x13bd": "Tesla M10", "0x17fd": "Tesla M40", "0x1b38": "Tesla P40", "0x15f8": "Tesla P100",
                "0x1db4": "Tesla V100"}
ROLES = {"studio": 7870, "agent": 47871}


def _read(p: Path) -> str:
    try:
        return p.read_text(errors="replace").strip()
    except OSError:
        return ""


# ================================================================ host facts

def big_cores(root: str = "/") -> str | None:
    """The fastest cores as a cpuset ("4-7"), when cores differ in capacity (big.LITTLE); else None."""
    caps = {}
    for d in (Path(root) / "sys/devices/system/cpu").glob("cpu[0-9]*"):
        c = _read(d / "cpu_capacity")
        if c.isdigit():
            caps[int(d.name[3:])] = int(c)
    if not caps or len(set(caps.values())) < 2:
        return None
    top = max(caps.values())
    ids = sorted(i for i, c in caps.items() if c == top)
    runs, start = [], ids[0]
    for a, b in zip(ids, ids[1:] + [None]):
        if b != a + 1:
            runs.append(f"{start}-{a}" if a != start else str(a))
            start = b
    return ",".join(runs)


def gpus(root: str = "/") -> list[dict]:
    """AMD and NVIDIA GPUs on the PCI bus, with their device nodes."""
    out = []
    base = Path(root) / "sys/bus/pci/devices"
    for dev in sorted(base.glob("*")) if base.is_dir() else []:
        vendor, cls = _read(dev / "vendor"), _read(dev / "class")
        if vendor not in ("0x1002", "0x10de") or not cls.startswith(("0x03", "0x12")):
            continue
        devid = _read(dev / "device")
        g = {"pci": dev.name, "vendor": "amd" if vendor == "0x1002" else "nvidia", "device": devid,
             "driver": Path(os.readlink(dev / "driver")).name if (dev / "driver").is_symlink() else None,
             "render": [], "card": []}
        for n in sorted((dev / "drm").glob("*")) if (dev / "drm").is_dir() else []:
            if n.name.startswith("renderD"):
                g["render"].append(f"/dev/dri/{n.name}")
            elif n.name.startswith("card"):
                g["card"].append(f"/dev/dri/{n.name}")
        if g["vendor"] == "amd":
            import amd_vbios
            g["model"] = amd_vbios.VEGA20.get(devid)
        else:
            g["model"] = NVIDIA_NAMES.get(devid)
        out.append(g)
    return out


def nvidia_nodes(root: str = "/") -> list[str]:
    d = Path(root) / "dev"
    names = sorted(p.name for p in d.glob("nvidia*") if p.name != "nvidia-caps")
    return [f"/dev/{n}" for n in names if re.fullmatch(r"nvidia(\d+|ctl|-uvm|-uvm-tools)", n)]


def host_facts(root: str = "/", machine: str | None = None) -> dict:
    compat = _read(Path(root) / "proc/device-tree/compatible").replace("\x00", " ").split()
    board = next((c for c in compat if c in NONCOHERENT), None)
    gs = gpus(root)
    nv_ver = _read(Path(root) / "proc/driver/nvidia/version")
    m = re.search(r"(\d{3}\.\d+(\.\d+)?)", nv_ver)
    return {
        "machine": machine or platform.machine(),
        "board": board or (compat[0] if compat else None),
        "coherent_pcie": board is None,
        "pcie_window": NONCOHERENT.get(board) if board else None,
        "vfio_possible": board is None and any((Path(root) / "sys/kernel/iommu_groups").glob("*")),
        "proxmox": (Path(root) / "usr/sbin/pct").exists() or (Path(root) / "etc/pve").is_dir(),
        "big_cores": big_cores(root),
        "gpus": gs,
        "kfd": (Path(root) / "dev/kfd").exists(),
        "nvidia_nodes": nvidia_nodes(root),
        "nvidia_driver": m.group(1) if m else None,
    }


# ================================================================ the plan

def plan(facts: dict, ctid: int, role: str = "agent", render_gid: int | None = None, video_gid: int | None = None,
         rocm: str | None = None, models: str | None = None, cpus: str | None = "auto", card_nodes: bool = False,
         port: int | None = None) -> dict:
    """{pct: [[args…]], conf: [raw lines], container: [shell lines to run inside], env: {…}, warnings: [...]}"""
    if role not in ROLES:
        raise ValueError(f"role: {' or '.join(ROLES)}")
    if not 100 <= int(ctid) <= 999999999:
        raise ValueError("ctid: a Proxmox container id (100 or more)")
    warn, pct, conf, inside, env = [], [], [], [], {}
    amd = [g for g in facts["gpus"] if g["vendor"] == "amd"]
    nv = [g for g in facts["gpus"] if g["vendor"] == "nvidia"]
    n = 0

    def dev(path, gid=None):
        nonlocal n
        spec = path + (f",gid={gid}" if gid is not None else "")
        pct.append(["pct", "set", str(ctid), f"-dev{n}", spec])
        n += 1

    if not facts["gpus"]:
        warn.append("no AMD or NVIDIA GPU found on this host: the container will run on the CPU")
    if not facts["coherent_pcie"]:
        warn.append(f"{facts['board']}: PCIe is not cache-coherent, so a VM cannot be given the GPU (VFIO refuses); "
                    "this container route is the only one. The host kernel's GPU driver must carry the "
                    "non-coherent fixes (mi50-arm for AMD, m10-arm for NVIDIA Maxwell).")
    if amd:
        if render_gid is None:
            warn.append("render group id inside the container unknown: run `pct exec "
                        f"{ctid} -- getent group render` and pass --render-gid (apply looks it up itself)")
        if any(g["driver"] not in (None, "amdgpu") for g in amd):
            warn.append("an AMD GPU is bound to " + ", ".join(sorted({g['driver'] for g in amd if g['driver']}))
                        + ": amdgpu must bind on the host (no vfio-pci.ids=, no amdgpu blacklist)")
        if facts["kfd"] or rocm:
            dev("/dev/kfd", render_gid)
        else:
            warn.append("/dev/kfd missing: ROCm (HIP) will not work; Vulkan still can. Build the host kernel with "
                        "HSA_AMD=y")
        for g in amd:
            for r in g["render"]:
                dev(r, render_gid)
            if card_nodes:
                for c in g["card"]:
                    dev(c, video_gid)
        if rocm:
            pct.append(["pct", "set", str(ctid), "-mp0", f"{rocm},mp=/opt/rocm,ro=1"])
            env.update({"ROCM_PATH": "/opt/rocm", "HIP_PLATFORM": "amd"})
            if not facts["coherent_pcie"]:
                env["HSA_USERPTR_FOR_PAGED_MEM"] = "0"
        inside.append("apt-get install -y mesa-vulkan-drivers vulkan-tools   # Vulkan: the first backend to try")
    if nv:
        nodes = facts["nvidia_nodes"]
        if not any(x.startswith("/dev/nvidia-uvm") for x in nodes):
            warn.append("/dev/nvidia-uvm is missing until something on the host uses CUDA: run "
                        "`nvidia-modprobe -u -c=0` on the host at boot (a systemd unit) before the container starts")
            nodes = nodes + ["/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"]
        for x in nodes:
            dev(x)
        ver = facts.get("nvidia_driver")
        if ver:
            inside.append(f"sh NVIDIA-Linux-$(uname -m)-{ver}.run --no-kernel-module --silent   "
                          "# user space only, the host's exact version")
        else:
            warn.append("NVIDIA driver version unknown (no /proc/driver/nvidia/version): install the container's "
                        "user-space libraries at exactly the host driver's version, with --no-kernel-module")
        if any(g.get("model") == "Tesla M10" for g in nv):
            warn.append("Tesla M10: four GPUs on one card; llama.cpp splits a model across them with "
                        "--split-mode layer. Maxwell needs CUDA 12.9 at most (profile nvidia-legacy-sm50-70).")
    if models:
        pct.append(["pct", "set", str(ctid), "-mp1", f"{models},mp=/models"])
    cpus = facts.get("big_cores") if cpus == "auto" else cpus
    if cpus:
        conf.append(f"lxc.cgroup2.cpuset.cpus: {cpus}")
    port = port or ROLES[role]
    inside += ["git clone <your OmniCortex repository> ~/omnicortex && cd ~/omnicortex && ./install.sh",
               "python scripts/hw_profiles.py match   # which hardware profile fits this GPU"]
    if role == "agent":
        inside.append(f"remote/omnicortex-remote.sh serve --mode lan --port {port}   # then pair it from "
                      "Studio (Remote); it finds llama-server on its usual port")
    else:
        inside.append(f"venv/bin/python viz/studio.py --host 0.0.0.0 --port {port}"
                      + (" --models-dir /models" if models else "")
                      + "   # a non-loopback bind needs Studio's token and TLS")
    if facts["vfio_possible"]:
        warn.append("this host could also pass the GPU to a VM (VFIO); the container shares it with the host "
                    "instead, which is simpler and has no overhead")
    return {"ctid": int(ctid), "role": role, "port": port, "pct": pct, "conf": conf, "container": inside,
            "env": env, "warnings": warn}


def render(pl: dict) -> str:
    lines = [f"# OmniCortex {pl['role']} in container {pl['ctid']} (port {pl['port']})", "", "## on the Proxmox host (root@pam)"]
    lines += [" ".join(shlex.quote(x) for x in c) for c in pl["pct"]] or ["# no devices to pass"]
    if pl["conf"]:
        lines += ["", f"## add to /etc/pve/lxc/{pl['ctid']}.conf"] + pl["conf"]
    if pl["env"]:
        lines += ["", "## environment inside the container (e.g. /etc/environment)"]
        lines += [f"{k}={v}" for k, v in pl["env"].items()]
    lines += ["", "## inside the container"] + pl["container"]
    lines += ["", "## then, inside the container", "python scripts/proxmox_lxc.py check"]
    if pl["warnings"]:
        lines += ["", "## notes"] + [f"- {w}" for w in pl["warnings"]]
    return "\n".join(lines)


def _container_gid(ctid: int, group: str) -> int | None:
    r = subprocess.run(["pct", "exec", str(ctid), "--", "getent", "group", group], capture_output=True, text=True)
    parts = r.stdout.strip().split(":")
    return int(parts[2]) if r.returncode == 0 and len(parts) > 2 and parts[2].isdigit() else None


def apply(pl: dict, log=print) -> int:
    if not shutil.which("pct"):
        raise SystemExit("pct not found: run apply on the Proxmox host")
    for c in pl["pct"]:
        log("$ " + " ".join(shlex.quote(x) for x in c))
        r = subprocess.run(c)
        if r.returncode:
            return r.returncode
    if pl["conf"]:
        p = Path(f"/etc/pve/lxc/{pl['ctid']}.conf")
        text = p.read_text()
        add = [ln for ln in pl["conf"] if ln not in text]
        if add:
            p.write_text(text.rstrip("\n") + "\n" + "\n".join(add) + "\n")
            log(f"added to {p}: " + "; ".join(add))
    return 0


# ================================================================ inside the container

def check(root: str = "/", env: dict | None = None) -> list[dict]:
    """PASS / WARN / FAIL lines for OmniCortex's GPU access inside a container."""
    env = os.environ if env is None else env
    out = []

    def r(level, text):
        out.append({"level": level, "text": text})
    d = Path(root) / "dev"
    kfd, render = d / "kfd", sorted((d / "dri").glob("renderD*")) if (d / "dri").is_dir() else []
    nv = [d / x.split("/")[-1] for x in nvidia_nodes(root)]
    if not (kfd.exists() or render or nv):
        r("fail", "no GPU device nodes (/dev/kfd, /dev/dri/renderD*, /dev/nvidia*): pass them with pct set -devN")
    for p in [kfd, *render, *nv]:
        if not p.exists():
            continue
        if os.access(p, os.R_OK | os.W_OK):
            r("pass", f"{p} opens for reading and writing")
        else:
            r("fail", f"{p} exists but this user cannot open it: check the gid= of its devN entry against "
                      "`getent group render` here (mode=0666 is the blunt fallback)")
    if kfd.exists() and render:
        if env.get("HSA_USERPTR_FOR_PAGED_MEM") != "0" and not host_coherent_hint(root):
            r("warn", "HSA_USERPTR_FOR_PAGED_MEM is not 0: on a host with non-coherent PCIe, ROCm needs it "
                      "(set it in /etc/environment)")
    if nv and not any(p.name == "nvidia-uvm" for p in nv):
        r("fail", "/dev/nvidia-uvm missing: CUDA cannot start. On the host: nvidia-modprobe -u -c=0, then pass it")
    if nv and not shutil.which("nvidia-smi"):
        r("warn", "nvidia-smi not found: install the host driver's user-space libraries (--no-kernel-module)")
    return out


def host_coherent_hint(root: str = "/") -> bool:
    """A container sees the host's device tree; a non-coherent board's compatible string is there too."""
    compat = _read(Path(root) / "proc/device-tree/compatible").replace("\x00", " ").split()
    return not any(c in NONCOHERENT for c in compat)


# ================================================================ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("host")
    h.add_argument("--json", action="store_true")
    for name in ("plan", "apply"):
        p = sub.add_parser(name)
        p.add_argument("--ctid", type=int, required=True)
        p.add_argument("--role", choices=list(ROLES), default="agent", help="remote agent (default) or Studio")
        p.add_argument("--render-gid", type=int, help="the render group's id inside the container")
        p.add_argument("--video-gid", type=int)
        p.add_argument("--card-nodes", action="store_true", help="also pass /dev/dri/card* (not needed for compute)")
        p.add_argument("--rocm", help="the host's ROCm install to mount read-only (e.g. /opt/rocm)")
        p.add_argument("--models", help="a host folder of models to mount at /models")
        p.add_argument("--cpus", default="auto", help="cpuset for the container (auto: the fastest cores)")
        p.add_argument("--port", type=int)
        p.add_argument("--json", action="store_true")
        if name == "apply":
            p.add_argument("--yes", action="store_true", help="run the pct commands (otherwise only prints them)")
    c = sub.add_parser("check")
    c.add_argument("--json", action="store_true")
    for p in sub.choices.values():
        p.add_argument("--root", default="/", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.cmd == "check":
        res = check(a.root)
        if a.json:
            print(json.dumps(res, indent=1))
        for x in res if not a.json else []:
            print(f"[{x['level'].upper()}] {x['text']}")
        return 1 if any(x["level"] == "fail" for x in res) else 0
    facts = host_facts(a.root)
    if a.cmd == "host":
        if a.json:
            print(json.dumps(facts, indent=1))
            return 0
        print(f"{facts['machine']}  board {facts['board'] or '?'}  PCIe {'coherent' if facts['coherent_pcie'] else 'NOT coherent'}"
              f"  Proxmox {'yes' if facts['proxmox'] else 'no'}  fast cores {facts['big_cores'] or '-'}")
        for g in facts["gpus"]:
            print(f"  {g['pci']} {g['vendor']} {g.get('model') or g['device']} driver={g['driver']} "
                  f"{' '.join(g['render'])}")
        if facts["nvidia_driver"]:
            print(f"  NVIDIA driver {facts['nvidia_driver']}; nodes {' '.join(facts['nvidia_nodes'])}")
        if facts["vfio_possible"]:
            print("  GPU sharing: an LXC container (this backend), or a VM with VFIO")
        elif not facts["coherent_pcie"]:
            print("  GPU sharing: LXC container only (non-coherent PCIe: VFIO is refused)")
        else:
            print("  GPU sharing: LXC container (no IOMMU groups: turn on the IOMMU for VFIO)")
        return 0
    rg = a.render_gid
    if a.cmd == "apply" and a.yes and rg is None and any(g["vendor"] == "amd" for g in facts["gpus"]):
        rg = _container_gid(a.ctid, "render")
    pl = plan(facts, a.ctid, a.role, rg, a.video_gid, a.rocm, a.models, a.cpus, a.card_nodes, a.port)
    if a.json:
        print(json.dumps(pl, indent=1))
    else:
        print(render(pl))
    if a.cmd == "apply":
        if not a.yes:
            print("\n(dry run: add --yes to run the pct commands)")
            return 0
        if any(g["vendor"] == "amd" for g in facts["gpus"]) and rg is None:
            raise SystemExit("could not read the container's render group id; pass --render-gid")
        return apply(pl)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    sys.exit(main())
