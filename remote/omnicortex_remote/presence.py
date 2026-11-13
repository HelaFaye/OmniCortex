"""
Is someone using this computer right now? Seconds since the last keyboard or mouse input,
whether a game controller is being used, and whether a VR session is running.

Used by the remote agent ("give the computer back when its owner uses it") and by Studio's
automation rules. Standard library only, best effort per platform:

  Windows  GetLastInputInfo (keyboard, mouse, touch); XInput for game controllers (any button
           or stick moved since the last look); SteamVR / Oculus / WMR processes for VR
  Linux    X11: XScreenSaver extension (libXss); else systemd-logind's IdleSinceHint; game
           controllers: /dev/input/js* activity; VR: SteamVR / Monado processes
  macOS    IOHIDSystem's HIDIdleTime (ioreg)

What cannot be measured is None, never a guess; a rule then only uses what is known.
"""
from __future__ import annotations

import ctypes
import glob
import os
import platform
import re
import subprocess
import time

VR_PROCESSES = ("vrserver", "vrcompositor", "vrmonitor", "OVRServer_x64", "OculusClient", "MixedRealityPortal",
                "monado-service", "wivrn-server", "ALVR Dashboard", "alvr_dashboard")
_PAD = {"state": None, "t": 0.0}


def _windows_idle() -> float | None:
    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
    try:
        li = LASTINPUTINFO()
        li.cbSize = ctypes.sizeof(li)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(li)):
            return None
        return max(0.0, (ctypes.windll.kernel32.GetTickCount() - li.dwTime) / 1000.0)
    except Exception:
        return None


def _windows_gamepad() -> bool | None:
    """True when any XInput controller's state changed since the last call (a button or a stick moved)."""
    class GAMEPAD(ctypes.Structure):
        _fields_ = [("wButtons", ctypes.c_ushort), ("bLeftTrigger", ctypes.c_ubyte), ("bRightTrigger", ctypes.c_ubyte),
                    ("sThumbLX", ctypes.c_short), ("sThumbLY", ctypes.c_short), ("sThumbRX", ctypes.c_short),
                    ("sThumbRY", ctypes.c_short)]

    class STATE(ctypes.Structure):
        _fields_ = [("dwPacketNumber", ctypes.c_uint), ("Gamepad", GAMEPAD)]
    lib = None
    for name in ("xinput1_4", "xinput1_3", "xinput9_1_0"):
        try:
            lib = getattr(ctypes.windll, name)
            break
        except OSError:
            continue
    if lib is None:
        return None
    packets = []
    for i in range(4):
        st = STATE()
        if lib.XInputGetState(i, ctypes.byref(st)) == 0:
            packets.append((i, st.dwPacketNumber))
    prev, _PAD["state"] = _PAD["state"], packets
    return bool(prev is not None and packets != prev and packets)


def _x11_idle() -> float | None:
    if not os.environ.get("DISPLAY"):
        return None
    try:
        x11 = ctypes.cdll.LoadLibrary("libX11.so.6")
        xss = ctypes.cdll.LoadLibrary("libXss.so.1")
    except OSError:
        return None

    class XScreenSaverInfo(ctypes.Structure):
        _fields_ = [("window", ctypes.c_ulong), ("state", ctypes.c_int), ("kind", ctypes.c_int),
                    ("til_or_since", ctypes.c_ulong), ("idle", ctypes.c_ulong), ("eventMask", ctypes.c_ulong)]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    xss.XScreenSaverAllocInfo.restype = ctypes.POINTER(XScreenSaverInfo)
    dpy = x11.XOpenDisplay(None)
    if not dpy:
        return None
    try:
        info = xss.XScreenSaverAllocInfo()
        root = x11.XDefaultRootWindow(ctypes.c_void_p(dpy))
        xss.XScreenSaverQueryInfo(ctypes.c_void_p(dpy), root, info)
        return info.contents.idle / 1000.0
    except Exception:
        return None
    finally:
        x11.XCloseDisplay(ctypes.c_void_p(dpy))


def _logind_idle() -> float | None:
    """systemd-logind's idle hint for the active graphical session (works on Wayland desktops that set it)."""
    try:
        sess = subprocess.run(["loginctl", "list-sessions", "--no-legend"], capture_output=True, text=True,
                              timeout=3).stdout.split()
        best = None
        for sid in sess[::1]:
            if not re.fullmatch(r"\w+", sid):
                continue
            out = subprocess.run(["loginctl", "show-session", sid, "-p", "Type", "-p", "Active", "-p", "IdleHint",
                                  "-p", "IdleSinceHintMonotonic"], capture_output=True, text=True, timeout=3).stdout
            d = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
            if d.get("Active") != "yes" or d.get("Type") not in ("x11", "wayland", "mir"):
                continue
            if d.get("IdleHint") == "no":
                return 0.0
            since = int(d.get("IdleSinceHintMonotonic") or 0)
            if since:
                idle = time.monotonic() - since / 1e6
                best = idle if best is None else min(best, idle)
        return best
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _linux_gamepad(window: float = 10) -> bool | None:
    """A joystick device that produced input in the last `window` seconds (its access time moves)."""
    devs = glob.glob("/dev/input/js*")
    if not devs:
        return None
    now = time.time()
    try:
        return any(now - os.stat(d).st_atime < window for d in devs)
    except OSError:
        return None


def _mac_idle() -> float | None:
    try:
        out = subprocess.run(["ioreg", "-c", "IOHIDSystem"], capture_output=True, text=True, timeout=3).stdout
        m = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', out)
        return int(m.group(1)) / 1e9 if m else None
    except (OSError, subprocess.SubprocessError):
        return None


def _processes() -> list[str]:
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True, timeout=5).stdout
            return [line.split('","')[0].strip('"') for line in out.splitlines() if line]
        out = subprocess.run(["ps", "-eo", "comm="], capture_output=True, text=True, timeout=5).stdout
        return [x.strip() for x in out.splitlines()]
    except (OSError, subprocess.SubprocessError):
        return []


def vr_active() -> bool | None:
    procs = _processes()
    if not procs:
        return None
    names = {p.lower().removesuffix(".exe") for p in procs}
    return any(v.lower() in names for v in VR_PROCESSES)


def idle_seconds() -> float | None:
    sysname = platform.system()
    if sysname == "Windows":
        return _windows_idle()
    if sysname == "Darwin":
        return _mac_idle()
    v = _x11_idle()
    return v if v is not None else _logind_idle()


def gamepad_active() -> bool | None:
    return _windows_gamepad() if platform.system() == "Windows" else _linux_gamepad() \
        if platform.system() == "Linux" else None


def snapshot() -> dict:
    """{idle_s, gamepad, vr, checked}: None where this computer cannot tell."""
    return {"idle_s": idle_seconds(), "gamepad": gamepad_active(), "vr": vr_active(), "checked": time.time()}


def in_use(snap: dict, idle_under: float = 60) -> tuple[bool, str]:
    """Is the owner using the computer? -> (yes, why)."""
    if snap.get("vr"):
        return True, "a VR session is running"
    if snap.get("gamepad"):
        return True, "a game controller is in use"
    if snap.get("idle_s") is not None and snap["idle_s"] < idle_under:
        return True, f"keyboard or mouse used {int(snap['idle_s'])} s ago"
    return False, ""


if __name__ == "__main__":
    import json
    s = snapshot()
    print(json.dumps(s, indent=1))
    print("in use:", in_use(s))
