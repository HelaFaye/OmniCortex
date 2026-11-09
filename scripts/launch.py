#!/usr/bin/env python3
"""
Start OmniCortex Studio and open it in the browser: the one thing to run
after install (the app-menu entry, `./omnicortex` and OmniCortex.bat all
call this).

If Studio is already running it just opens it. Otherwise it starts Studio in
the background with the settings from ~/.omnicortex/config.json (edited on
the Setup page), waits until it answers, and opens Setup if anything is
missing, Studio otherwise. Studio keeps running after the browser closes;
stop it from the terminal it printed, or with --stop.

    python scripts/launch.py            # start if needed, open the browser
    python scripts/launch.py --no-browser
    python scripts/launch.py --stop
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = Path.home() / ".omnicortex"
PIDFILE = STATE_DIR / "studio.pid"
LOG = STATE_DIR / "studio.log"


def _get(url: str, timeout: float = 2):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:
        return None, None


def running(base: str) -> bool:
    code, _ = _get(base + "/api/status")
    return code in (200, 401)


def start(port: int, workspace: str | None = None) -> subprocess.Popen:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG, "ab")
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0)
    else:
        kw["start_new_session"] = True             # keeps running when the launcher's terminal closes
    proc = subprocess.Popen([sys.executable, "-u", str(ROOT / "viz" / "studio.py"), "--port", str(port),
                             *(["--workspace", workspace] if workspace else [])],
                            stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), **kw)
    PIDFILE.write_text(str(proc.pid))
    return proc


def main(argv=None) -> int:
    try:
        import oc_migrate
        oc_migrate.migrate()
    except ImportError:
        pass
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", type=int, default=7870)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--stop", action="store_true", help="stop the Studio this launcher started")
    p.add_argument("--workspace", help="start Studio in this workspace (Settings -> Workspaces)")
    a = p.parse_args(argv)
    base = f"http://127.0.0.1:{a.port}"

    if a.stop:
        try:
            pid = int(PIDFILE.read_text())
            os.kill(pid, signal.SIGTERM)
            PIDFILE.unlink()
            print(f"stopped Studio (pid {pid})")
        except (OSError, ValueError):
            print("no Studio started by the launcher is running")
        return 0

    if not running(base):
        proc = start(a.port, a.workspace)
        print(f"starting OmniCortex Studio on {base} (log: {LOG})")
        end = time.time() + 60
        while time.time() < end and not running(base):
            if proc.poll() is not None:
                print(f"Studio exited (code {proc.returncode}); the end of {LOG}:", file=sys.stderr)
                print("".join(LOG.read_text(errors="replace").splitlines(True)[-15:]), file=sys.stderr)
                return 1
            time.sleep(0.5)
        if not running(base):
            print(f"Studio did not answer within a minute; see {LOG}", file=sys.stderr)
            return 1
    else:
        print(f"Studio is already running on {base}")

    code, setup = _get(base + "/api/setup")
    ready = bool(setup and setup["ready"]["server"] and setup["ready"]["models"])
    url = base + ("/" if ready or code != 200 else "/setup")
    print(f"open {url}")
    if not a.no_browser:
        webbrowser.open(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
