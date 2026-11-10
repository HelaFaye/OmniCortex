#!/usr/bin/env python3
"""
AMD Instinct MI50/MI60 (Vega 20) VBIOS check (docs/PROXMOX.md): which VBIOS each card runs, what that means for
OmniCortex, and whether the host can give the card the BAR it asks for.

The VBIOS table comes from a community-maintained comparison of MI50 32 GB VBIOS ROMs (tested on x86 with AMD's
original PCB; see docs/CREDITS.md). Two of its findings matter here:

    113-D1631711-100   no resizable BAR and a 16 GB BAR: llama.cpp's Vulkan backend sees only 16 GB of the 32, and
                       on a host with a small PCIe window (RK3588: 1 GiB) the BAR cannot be placed or shrunk at all
    -111, V420         resizable BAR: on a small-window host, shrink BAR0 to 256 MiB before amdgpu binds

Cards with different VBIOS versions in one machine have caused ROCm failures in that testing, so a mix is flagged.

    python scripts/amd_vbios.py            # every AMD card here
    python scripts/amd_vbios.py --json

Read-only: it reads sysfs and never flashes anything. Reflashing a VBIOS can brick a card; that is a decision for
you, with a backup of the original ROM, on an x86 machine.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GIB = 1 << 30

# VBIOS part number (rocm-smi --showvbios, sysfs vbios_version) -> what is known about it
KNOWN = {
    "113-D163A1XT-045": {"name": "Apple Radeon Pro VII 32 GB", "rebar": True, "bar": "32 GB", "pcie_gen": 3,
                         "power_w": 300, "display": True, "uefi": True,
                         "notes": "PCIe Gen 3 only, and no link power saving (idle 27-29 W)."},
    "113-D1631711-100": {"name": "AMD MI50 32 GB (2021)", "rebar": False, "bar": "16 GB", "pcie_gen": 4,
                         "power_w": 225, "display": False, "uefi": False,
                         "notes": "No resizable BAR and a 16 GB BAR: Vulkan (llama.cpp's Vulkan backend) sees only "
                                  "16 GB of VRAM. Use the HIP backend, or a VBIOS with resizable BAR."},
    "113-D1631700-111": {"name": "AMD MI50 32 GB (2020)", "rebar": True, "bar": "32 GB", "pcie_gen": 4,
                         "power_w": 225, "display": False, "uefi": False,
                         "notes": "Resizable BAR, full 32 GB BAR. Recommended when no display output is needed."},
    "113-D1640200-043": {"name": "Radeon Pro V420 32 GB", "rebar": True, "bar": "32 GB", "pcie_gen": 4,
                         "power_w": 178, "display": True, "uefi": True,
                         "notes": "Resizable BAR, display output, 178 W cap (about 1% slower), lowest idle power."},
}
VEGA20 = {"0x66a0": "Instinct MI60 / Radeon Pro VII", "0x66a1": "Instinct MI50", "0x66a2": "Vega 20",
          "0x66a3": "Radeon Pro Vega II", "0x66a7": "Vega 20", "0x66af": "Radeon VII"}


def _read(p: Path) -> str:
    try:
        return p.read_text().strip()
    except OSError:
        return ""


def bar0(dev: Path) -> dict:
    """BAR0's assigned size from sysfs `resource` (0 when unassigned) and whether it can be resized."""
    lines = _read(dev / "resource").splitlines()
    size = 0
    if lines:
        try:
            start, end, _flags = (int(x, 16) for x in lines[0].split()[:3])
            size = end - start + 1 if end > start else 0
        except ValueError:
            size = 0
    resize = _read(dev / "resource0_resize")
    return {"size": size, "resizable": bool(resize), "resize_mask": resize or None}


def cards(root: str = "/") -> list[dict]:
    """Every AMD display/compute device on the PCI bus, with its VBIOS and BAR0."""
    out = []
    base = Path(root) / "sys/bus/pci/devices"
    for dev in sorted(base.glob("*")) if base.is_dir() else []:
        if _read(dev / "vendor") != "0x1002" or not _read(dev / "class").startswith(("0x03", "0x12")):
            continue
        devid = _read(dev / "device")
        vb = ""
        for drm in sorted((dev / "drm").glob("card*")) if (dev / "drm").is_dir() else []:
            vb = _read(drm / "device" / "vbios_version") or vb
        vb = vb or _read(dev / "vbios_version")
        out.append({"pci": dev.name, "device": devid, "model": VEGA20.get(devid), "vbios": vb or None,
                    "known": KNOWN.get(vb), **{"bar0_" + k: v for k, v in bar0(dev).items()}})
    return out


def verdicts(cs: list[dict], window: int | None = None) -> list[dict]:
    """[{level: ok|warn|fail, pci, text}] for these cards. `window`: the host's largest 64-bit PCIe window in
    bytes when it is small (RK3588's pcie3x4: 1 GiB), else None."""
    v = []
    vega = [c for c in cs if c["model"]]
    for c in vega:
        k, pci = c["known"], c["pci"]
        if not c["vbios"]:
            v.append({"level": "warn", "pci": pci, "text": "VBIOS version unreadable (amdgpu not bound?)"})
        elif k is None:
            v.append({"level": "warn", "pci": pci, "text": f"VBIOS {c['vbios']} is not in the table; check its BAR "
                                                         "and resizable-BAR support with lspci -vv"})
        else:
            v.append({"level": "warn" if not k["rebar"] else "ok", "pci": pci,
                      "text": f"{c['vbios']} ({k['name']}): {k['notes']}"})
        if window:
            assigned = c["bar0_size"]
            if assigned and assigned <= window:
                v.append({"level": "ok", "pci": pci, "text": f"BAR0 is {assigned >> 20} MiB and fits the "
                                                             f"{window >> 20} MiB PCIe window"})
            elif c["bar0_resizable"] or (k and k["rebar"]):
                v.append({"level": "warn", "pci": pci, "text": (
                    f"BAR0 must be shrunk to fit the {window >> 20} MiB PCIe window: before amdgpu binds, "
                    f"echo 8 > /sys/bus/pci/devices/{pci}/resource0_resize (256 MiB). The GPU still uses all "
                    "its VRAM.")})
            else:
                v.append({"level": "fail", "pci": pci, "text": (
                    f"BAR0 cannot fit the {window >> 20} MiB PCIe window and this VBIOS cannot resize it: amdgpu "
                    "will not load on this host. A VBIOS with resizable BAR (113-D1631700-111 or 113-D1640200-043) "
                    "is the only route; reflash only on an x86 machine, with a backup.")})
    versions = {c["vbios"] for c in vega if c["vbios"]}
    if len(versions) > 1:
        v.append({"level": "warn", "pci": "*", "text": f"cards run different VBIOS versions ({', '.join(sorted(versions))}); "
                                                      "mixing them has caused ROCm failures, prefer one version"})
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--window-mib", type=int, help="the host's PCIe 64-bit window, if small (RK3588: 1024)")
    ap.add_argument("--root", default="/", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    cs = cards(a.root)
    if a.window_mib is None:
        import proxmox_lxc
        a.window_mib = (proxmox_lxc.host_facts(a.root).get("pcie_window") or 0) >> 20 or None
    vs = verdicts(cs, (a.window_mib << 20) if a.window_mib else None)
    if a.json:
        print(json.dumps({"cards": cs, "verdicts": vs}, indent=1))
    else:
        if not cs:
            print("no AMD GPUs found")
        for c in cs:
            print(f"{c['pci']}  {c['model'] or c['device']}  VBIOS {c['vbios'] or '?'}  BAR0 "
                  f"{(c['bar0_size'] >> 20) if c['bar0_size'] else 'unassigned'}"
                  f"{' MiB' if c['bar0_size'] else ''}{' (resizable)' if c['bar0_resizable'] else ''}")
        for x in vs:
            print(f"  [{x['level'].upper()}] {x['pci']}: {x['text']}")
    return 1 if any(x["level"] == "fail" for x in vs) else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
