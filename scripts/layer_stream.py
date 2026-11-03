#!/usr/bin/env python3
"""
Layer streaming across computers (docs/INTEGRATED_OPTIONS.md) with llama.cpp's RPC backend: each helper computer
runs rpc-server; the computer serving the model starts llama-server with --rpc host:port,... and llama.cpp
places a share of the layers on each helper, passing activations between them as it computes. A model too large
for any one computer can run across several.

    serve   run rpc-server on this computer (build it with scripts/build_llama_tools.sh --rpc); --cache keeps
            received weights on disk, so the next load does not send them again
    check   can this computer reach each helper, and how quickly (TCP connect time)
    split   a -ts (tensor split) from each device's free memory, e.g. "24,24,11" for two 24 GB helpers and the
            local 11 GB card, in the order llama-server lists devices (RPC devices first)

Then set the model's RPC servers (Studio → the model → RPC servers) to "helper1:50052,helper2:50052" and, if the
split should not follow llama.cpp's own guess, its tensor split.

Security: rpc-server has no authentication and no encryption, and it lets whoever connects allocate memory and run
computations on that machine. Bind it to a trusted network interface (--host), never the internet; across
networks put it behind WireGuard, Tailscale or an SSH tunnel (ssh -L 50052:localhost:50052 helper). This does not
use OmniCortex's WebRTC channel: RPC's traffic is large and latency-bound and needs a direct path.

Speed: every token crosses the network once per helper. Expect a fraction of a single machine's speed on
gigabit Ethernet; it is for models that do not fit at all otherwise, not to make a fitting model faster.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse(spec: str) -> list[tuple[str, int]]:
    out = []
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        host, _, port = part.rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(f"{part!r}: host:port")
        out.append((host.strip("[]"), int(port)))
    return out


def check(spec: str, timeout: float = 3.0) -> list[dict]:
    res = []
    for host, port in parse(spec):
        t0 = time.monotonic()
        try:
            with socket.create_connection((host, port), timeout=timeout):
                res.append({"server": f"{host}:{port}", "ok": True, "connect_ms": round((time.monotonic() - t0) * 1000, 1)})
        except OSError as e:
            res.append({"server": f"{host}:{port}", "ok": False, "error": str(e)})
    return res


def split(free_gb: list[float]) -> str:
    """Proportions for -ts from each device's free memory (rounded, nonzero)."""
    vals = [max(0.0, float(x)) for x in free_gb]
    if not vals or not any(vals):
        raise ValueError("give each device's free memory in GB")
    return ",".join(str(max(1, round(v))) if v > 0 else "0" for v in vals)


def find_rpc_server() -> str | None:
    for c in (shutil.which("rpc-server"), os.environ.get("NS_LLAMA") and str(Path(os.environ["NS_LLAMA"]) / "build" / "bin" / "rpc-server"),
              str(Path.home() / "llama.cpp" / "build" / "bin" / "rpc-server"),
              str(ROOT / "build" / "llama.cpp" / "build" / "bin" / "rpc-server")):
        if c and Path(c).exists():
            return c
    return None


def serve_cmd(binary: str, host: str, port: int, cache: bool, threads: int | None, device: str | None) -> list[str]:
    cmd = [binary, "-H", host, "-p", str(port)]
    if cache:
        cmd.append("-c")
    if threads:
        cmd += ["-t", str(threads)]
    if device:
        cmd += ["-d", device]
    return cmd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1", help="the interface to listen on (a trusted LAN or tunnel address)")
    s.add_argument("--port", type=int, default=50052)
    s.add_argument("--cache", action="store_true", help="cache received weights on disk")
    s.add_argument("--threads", type=int)
    s.add_argument("--device", help="which local device(s) to offer, e.g. CUDA0 or Vulkan0,Vulkan1")
    s.add_argument("--binary", help="rpc-server (default: found in the llama.cpp build)")
    c = sub.add_parser("check")
    c.add_argument("servers", help="host:port,host:port")
    sp = sub.add_parser("split")
    sp.add_argument("free_gb", nargs="+", type=float)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        ok = True
        for r in check(a.servers):
            ok &= r["ok"]
            print(f"{r['server']:28} " + (f"reachable, {r['connect_ms']} ms" if r["ok"] else f"unreachable: {r['error']}"))
        return 0 if ok else 1
    if a.cmd == "split":
        print(split(a.free_gb))
        return 0
    binary = a.binary or find_rpc_server()
    if not binary:
        raise SystemExit("no rpc-server: build it with scripts/build_llama_tools.sh --rpc")
    if a.host in ("0.0.0.0", "::"):
        print("warning: listening on every interface; anyone who can reach this port can use this machine's "
              "memory and compute. Prefer the trusted network's own address.", file=sys.stderr)
    cmd = serve_cmd(binary, a.host, a.port, a.cache, a.threads, a.device)
    print("$ " + " ".join(cmd), flush=True)
    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())
