#!/usr/bin/env python3
"""
Named hardware profiles, so a config can be planned for a machine you are not
sitting at.

A config says what you want: model, context, cache type, visualizer. A host
says what a machine can do: VRAM, RAM, cores, backend. Keeping them apart is
what lets one config be checked against several machines -- 47104 context is
comfortable on a 16 GiB discrete card and marginal on a 12 GiB shared carve-out,
and the config did not change, the host did.

    python scripts/hostprofiles.py detect --save laptop
    python scripts/hostprofiles.py add rig --vram 16 --ram 32 --cores 16 \\
        --gpu "RX 9060 XT" --backend vulkan --dedicated
    python scripts/hostprofiles.py list
    python scripts/hostprofiles.py plan --gguf model.gguf --ctx 47104 \\
        --host rig --host laptop

Profiles live in ~/.omnicortex/hosts.json. `detect` fingerprints the current
machine so the right profile is selected automatically when one matches.

The dedicated/shared distinction matters more than the number. A discrete card's
VRAM is exclusively yours; an iGPU's carve-out comes out of system RAM, so a
large allocation there is also a large allocation away from everything else.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys

GIB = 1024 ** 3
STORE = os.path.expanduser("~/.omnicortex/hosts.json")

FIELDS = {"vram_gib": 0.0, "ram_gib": 0.0, "cores": 0, "gpu": "",
          "backend": "cpu", "dedicated": False, "reserve_gib": 0.75,
          "fingerprint": "", "note": ""}


def _meminfo(key):
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(key):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def detect():
    """Best-effort profile for the machine this is running on."""
    ram = _meminfo("MemTotal:") or 0
    gpu, backend, vram, dedicated = "", "cpu", 0.0, False

    try:
        from cuda_info import query_gpus
        nv = query_gpus()
        if nv:
            gpu = nv[0]["name"] if len(nv) == 1 else f"{len(nv)} x {nv[0]['name']}"
            vram = sum(g["memory_total"] for g in nv) / GIB
            backend, dedicated = "cuda", True
    except Exception:
        pass
    try:
        import torch
        if not gpu and torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.mem_get_info()[1] / GIB
            backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
            dedicated = True
    except Exception:
        pass

    if not gpu and shutil.which("vulkaninfo"):
        try:
            out = subprocess.run(["vulkaninfo", "--summary"],
                                 capture_output=True, text=True,
                                 timeout=25).stdout
            name = dtype = None
            for line in out.splitlines():
                if "deviceName" in line and name is None:
                    name = line.split("=", 1)[-1].strip()
                if "deviceType" in line and dtype is None:
                    dtype = line.split("=", 1)[-1].strip()
            if name:
                gpu, backend = name, "vulkan"
                dedicated = "INTEGRATED" not in (dtype or "")
                # An integrated GPU's usable share is a BIOS carve-out that is
                # not reported here, so leave it for the user rather than
                # guessing a number that will silently be wrong.
                if dedicated:
                    vram = 0.0
        except (OSError, subprocess.SubprocessError):
            pass

    fp = hashlib.sha256(
        f"{platform.node()}|{platform.machine()}|{gpu}|{ram}".encode()
    ).hexdigest()[:16]

    return {**FIELDS, "ram_gib": round(ram / GIB, 1),
            "cores": os.cpu_count() or 0, "gpu": gpu, "backend": backend,
            "vram_gib": round(vram, 1), "dedicated": dedicated,
            "fingerprint": fp,
            "note": "" if vram else
                    "VRAM not detected; set --vram (an iGPU carve-out is set "
                    "in the BIOS and not reported)"}


def load_all():
    if os.path.exists(STORE):
        try:
            with open(STORE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_all(d):
    os.makedirs(os.path.dirname(STORE), exist_ok=True)
    tmp = STORE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, STORE)


def current_profile():
    """The saved profile matching this machine, or a detected one."""
    d = detect()
    for name, p in load_all().items():
        if p.get("fingerprint") and p["fingerprint"] == d["fingerprint"]:
            return name, {**FIELDS, **p}
    return None, d


def get(name):
    all_ = load_all()
    if name not in all_:
        raise SystemExit(f"no host profile {name!r}; "
                         f"have {sorted(all_) or 'none'}")
    return {**FIELDS, **all_[name]}


def plan_for(host, gguf, ctx, viz="pygfx", cache_type="f16", cache_type_v=None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vram_budget import plan, read_gguf_shape
    shape = read_gguf_shape(gguf)
    vram = int(host["vram_gib"] * GIB)
    if not vram:
        return shape, None
    # A shared carve-out needs more headroom: the desktop compositor is drawing
    # from the same pool.
    reserve = host["reserve_gib"] * GIB * (1.0 if host["dedicated"] else 1.6)
    return shape, plan(shape, ctx, vram, viz, cache_type, cache_type_v,
                       None, reserve)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="inspect this machine")
    d.add_argument("--save", metavar="NAME")
    d.add_argument("--vram", type=float, help="GiB; required for an iGPU")

    a = sub.add_parser("add", help="define a machine by hand")
    a.add_argument("name")
    a.add_argument("--vram", type=float, required=True)
    a.add_argument("--ram", type=float, default=0.0)
    a.add_argument("--cores", type=int, default=0)
    a.add_argument("--gpu", default="")
    a.add_argument("--backend", default="vulkan",
                   choices=["cpu", "vulkan", "rocm", "cuda", "metal"])
    a.add_argument("--dedicated", action="store_true",
                   help="discrete card; omit for an iGPU carve-out")
    a.add_argument("--reserve", type=float, default=0.75)
    a.add_argument("--note", default="")

    sub.add_parser("list")
    rm = sub.add_parser("remove")
    rm.add_argument("name")

    pl = sub.add_parser("plan", help="plan one config across machines")
    pl.add_argument("--gguf", required=True)
    pl.add_argument("--ctx", type=int, default=8192)
    pl.add_argument("--viz", default="pygfx")
    pl.add_argument("--cache-type", default="f16")
    pl.add_argument("--cache-type-v")
    pl.add_argument("--host", action="append", default=[],
                    help="repeatable; omit for every saved profile")
    pl.add_argument("--json", action="store_true")

    args = p.parse_args()

    if args.cmd == "detect":
        prof = detect()
        if args.vram:
            prof["vram_gib"], prof["note"] = args.vram, ""
        for k, v in prof.items():
            if v not in ("", 0, 0.0, False):
                print(f"  {k:<12} {v}")
        if prof["note"]:
            print(f"  {'!':<12} {prof['note']}")
        if args.save:
            all_ = load_all()
            all_[args.save] = prof
            save_all(all_)
            print(f"\nsaved as {args.save!r} in {STORE}")
        return

    if args.cmd == "add":
        all_ = load_all()
        all_[args.name] = {**FIELDS, "vram_gib": args.vram,
                           "ram_gib": args.ram, "cores": args.cores,
                           "gpu": args.gpu, "backend": args.backend,
                           "dedicated": args.dedicated,
                           "reserve_gib": args.reserve, "note": args.note}
        save_all(all_)
        print(f"saved {args.name!r}")
        return

    if args.cmd == "list":
        cur, _ = current_profile()
        all_ = load_all()
        if not all_:
            print("no profiles. Try: hostprofiles.py detect --save laptop")
            return
        for name, h in all_.items():
            h = {**FIELDS, **h}
            mark = " <- this machine" if name == cur else ""
            kind = "dedicated" if h["dedicated"] else "shared carve-out"
            print(f"  {name:<12} {h['vram_gib']:>5.1f} GiB VRAM ({kind})  "
                  f"{h['ram_gib']:>5.1f} GiB RAM  {h['cores']:>3} cores  "
                  f"{h['backend']:<7} {h['gpu'][:28]}{mark}")
        return

    if args.cmd == "remove":
        all_ = load_all()
        all_.pop(args.name, None)
        save_all(all_)
        print(f"removed {args.name!r}")
        return

    names = args.host or list(load_all())
    if not names:
        raise SystemExit("no host profiles; run detect --save first")
    rows = []
    for name in names:
        h = get(name)
        shape, r = plan_for(h, args.gguf, args.ctx, args.viz,
                            args.cache_type, args.cache_type_v)
        rows.append((name, h, r))

    if args.json:
        print(json.dumps([{"host": n, "plan": r} for n, _, r in rows], indent=2))
        return

    print(f"{os.path.basename(args.gguf)} @ ctx {args.ctx}, "
          f"cache {args.cache_type}, viz {args.viz}\n")
    print(f"  {'host':<12} {'VRAM':>8} {'KV':>8} {'layers':>9} "
          f"{'viz':>5}  verdict")
    for name, h, r in rows:
        if r is None:
            print(f"  {name:<12} {'unknown':>8}   set --vram for this profile")
            continue
        verdict = "fits" if r["fits"] else "does not fit"
        print(f"  {name:<12} {h['vram_gib']:>6.1f}G {r['kv_bytes'] / GIB:>7.2f}G "
              f"{r['layers_on_gpu']:>4}/{r['n_layers']:<4} "
              f"{r['viz_device']:>5}  {verdict}")
    print()
    for name, h, r in rows:
        if r is None:
            continue
        for n in r["notes"]:
            print(f"  {name}: {n}")
        for w in r["warnings"]:
            print(f"  {name}: WARNING {w}")


if __name__ == "__main__":
    main()
