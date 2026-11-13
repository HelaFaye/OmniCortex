"""Godot client: download Godot 4 for this computer and check it against the release's SHA-512 list."""
import argparse
import hashlib
import io
import os
import platform
import stat
import sys
import urllib.request
import zipfile
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--version", default="4.7.2-stable")
p.add_argument("--folder", default="~/.omnicortex/godot")
a = p.parse_args()
if not a.version.replace(".", "").replace("-", "").isalnum():
    raise SystemExit("version: a Godot release tag such as 4.7.2-stable")
system, machine = platform.system(), platform.machine().lower()
v = a.version
asset = {("Linux", "x86_64"): f"Godot_v{v}_linux.x86_64.zip", ("Linux", "aarch64"): f"Godot_v{v}_linux.arm64.zip",
         ("Linux", "arm64"): f"Godot_v{v}_linux.arm64.zip", ("Windows", "amd64"): f"Godot_v{v}_win64.exe.zip",
         ("Windows", "x86_64"): f"Godot_v{v}_win64.exe.zip", ("Windows", "arm64"): f"Godot_v{v}_windows_arm64.exe.zip",
         }.get((system, machine)) or (f"Godot_v{v}_macos.universal.zip" if system == "Darwin" else None)
if not asset:
    raise SystemExit(f"no Godot build known for {system} {machine}: get one from https://godotengine.org/download")
base = f"https://github.com/godotengine/godot/releases/download/{v}/"


def get(url, limit=400 * 1024 * 1024):
    print(f"downloading {url}", flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "OmniCortex"}), timeout=120) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise SystemExit("download too large")
    return data


sums = get(base + "SHA512-SUMS.txt", 1 << 20).decode()
want = next((line.split()[0] for line in sums.splitlines() if line.strip().endswith(asset)), None)
if not want:
    raise SystemExit(f"{asset} is not in the release's SHA512-SUMS.txt")
data = get(base + asset)
if hashlib.sha512(data).hexdigest() != want:
    raise SystemExit("the download does not match the release's SHA-512: not installed")
print("SHA-512 matches", flush=True)
dest = Path(os.path.expanduser(a.folder))
dest.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(data)) as z:
    for n in z.namelist():
        if n.startswith("/") or ".." in Path(n).parts:
            raise SystemExit(f"unsafe path in the download: {n}")
    z.extractall(dest)
if system == "Darwin":
    exe = dest / "Godot.app" / "Contents" / "MacOS" / "Godot"
else:
    exe = next(p for p in dest.iterdir() if p.name.startswith(f"Godot_v{v}") and not p.name.endswith(".zip")
               and "console" not in p.name)
exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
link = dest / ("godot.exe" if system == "Windows" else "godot")
if link.exists() or link.is_symlink():
    link.unlink()
try:
    link.symlink_to(exe)
except OSError:
    import shutil
    shutil.copy2(exe, link)
print(f"Godot {v}: {link}", flush=True)
