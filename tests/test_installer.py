"""The cross-platform installer: what it decides for each kind of computer, its plan,
and its wizard's access rules (token in the path, same-origin posts)."""
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import installer as ins  # noqa: E402


def fake(system="Linux", machine="x86_64", nvidia=(), amd=(), gfx=None, kfd=False, apple=False, py_ok=True,
         venv_ok=True, free=200, like="ubuntu debian", root="/home/me/neuronscope"):
    d = {"system": system, "release": "x", "machine": machine, "python": "/usr/bin/python3",
         "python_version": "3.12.0" if py_ok else "3.9.6", "python_ok": py_ok, "store_python": False,
         "venv": "/tmp/v", "venv_exists": False, "free_gb": free, "root": root, "venv_ok": venv_ok,
         "distro_like": like, "gpus": {"nvidia": list(nvidia), "amd": list(amd), "apple": apple, "other": [],
                                       "gfx": gfx, "kfd": kfd}}
    return d


def notes(monkeypatch, **kw):
    d = fake(**kw)
    monkeypatch.setattr(ins, "SYSTEM", d["system"])
    monkeypatch.setattr(ins, "_run", lambda *a, **k: "video render")
    return ins.platform_notes(d), d


def test_torch_choices_per_platform(monkeypatch):
    monkeypatch.setattr(ins, "SYSTEM", "Darwin")
    assert ins.torch_choice(fake("Darwin", "arm64", apple=True))["spec"] == "torch"
    assert ins.torch_choice(fake("Darwin", "x86_64"))["spec"] == "torch<2.3"          # no Intel builds after 2.2
    monkeypatch.setattr(ins, "SYSTEM", "Linux")
    t = ins.torch_choice(fake(amd=["Radeon"], gfx="gfx1200", kfd=True))
    assert t["mode"] == "rocm" and t["index"].endswith("rocm7.0")                       # RDNA4 needs ROCm 7
    assert ins.torch_choice(fake(amd=["Radeon"], gfx="gfx1100", kfd=True))["index"].endswith("rocm6.4")
    assert ins.torch_choice(fake(amd=["Radeon"]))["mode"] == "cpu"                       # no /dev/kfd
    monkeypatch.setattr(ins, "_run", lambda *a, **k: "torch --index-url https://download.pytorch.org/whl/cu126\n")
    t = ins.torch_choice(fake(nvidia=["Tesla V100"]))
    assert t["mode"] == "cuda" and t["index"].endswith("cu126")
    monkeypatch.setattr(ins, "SYSTEM", "Windows")
    assert ins.torch_choice(fake("Windows", amd=["AMD Radeon RX"]))["mode"] == "cpu"     # no ROCm on Windows
    assert ins.torch_choice(fake(), "none")["spec"] is None
    with pytest.raises(ValueError):
        ins.torch_choice(fake(), "tpu")


def test_platform_notes(monkeypatch):
    n, _ = notes(monkeypatch, py_ok=False)
    assert any(x["level"] == "stop" and "3.10" in x["text"] for x in n)
    n, _ = notes(monkeypatch, venv_ok=False)
    assert any(x["level"] == "stop" and "apt install python3-venv" in x["text"] for x in n)
    n, _ = notes(monkeypatch, venv_ok=False, like="arch")
    assert any("pacman -S python" in x["text"] for x in n)
    n, _ = notes(monkeypatch, gfx="gfx1201", kfd=True, amd=["Radeon"])
    assert any("do not set HSA_OVERRIDE_GFX_VERSION" in x["text"] for x in n)
    n, _ = notes(monkeypatch, system="Windows", amd=["Radeon"], root="C:\\" + "x" * 130)
    texts = " ".join(x["text"] for x in n)
    assert "no ROCm build for Windows" in texts and "260 characters" in texts and "Build Tools" in texts
    n, _ = notes(monkeypatch, system="Darwin", machine="x86_64")
    assert any("Intel Mac" in x["text"] for x in n)
    n, _ = notes(monkeypatch, free=3)
    assert any(x["level"] == "warn" and "GB free" in x["text"] for x in n)


def test_plan(monkeypatch):
    monkeypatch.setattr(ins, "SYSTEM", "Linux")
    d = fake()
    ids = [s["id"] for s in ins.plan(d, ins.torch_choice(d, "cpu"))]
    assert ids == ["venv", "pip", "core", "torch", "rest", "extras", "smoke", "shortcut"]
    ids = [s["id"] for s in ins.plan(d, ins.torch_choice(d, "none"), extras=False, shortcut=False)]
    assert ids == ["venv", "pip", "core"]
    st = ins.plan(d, ins.torch_choice(d, "cpu"))
    torch_cmd = next(s for s in st if s["id"] == "torch")["cmd"]
    assert torch_cmd[-2:] == ["--index-url", "https://download.pytorch.org/whl/cpu"]
    assert next(s for s in st if s["id"] == "smoke").get("optional")


def test_a_failing_step_stops_the_run_with_advice(tmp_path):
    d = fake()
    steps = [{"id": "core", "title": "ok", "cmd": [sys.executable, "-c", "print('fine')"]},
             {"id": "extras", "title": "optional", "cmd": [sys.executable, "-c", "raise SystemExit(3)"], "optional": True},
             {"id": "torch", "title": "boom", "cmd": [sys.executable, "-c", "raise SystemExit(2)"]},
             {"id": "rest", "title": "never", "cmd": [sys.executable, "-c", "print('no')"]}]
    r = ins.Run(d, steps)
    assert r.run() is False
    assert [s["status"] for s in steps] == ["done", "skipped", "failed", "waiting"]
    assert any("PyTorch download failed" in x for x in r.lines) and "fine" in r.lines


def test_wizard_access():
    p = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "installer.py"), "--no-browser", "--port", "0"],
                         stdout=subprocess.PIPE, text=True)
    try:
        url = p.stdout.readline().split()[-1]
        base = url.rsplit("/", 2)[0]
        page = urllib.request.urlopen(url).read().decode()
        assert "Install OmniCortex" in page
        det = json.loads(urllib.request.urlopen(url + "api/detect").read())
        assert det["python_ok"] and "auto" in det["torch"]
        with pytest.raises(urllib.error.HTTPError) as e:                 # no token, no page
            urllib.request.urlopen(base + "/")
        assert e.value.code == 404
        req = urllib.request.Request(url + "api/install", data=b"{}", headers={"Origin": "http://evil.example",
                                                                              "Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:                 # another site cannot start an install
            urllib.request.urlopen(req)
        assert e.value.code == 403
        urllib.request.urlopen(urllib.request.Request(url + "api/quit", data=b"{}"))
        p.wait(timeout=10)
    finally:
        if p.poll() is None:
            p.terminate()


def test_launchers_are_well_formed():
    bat = (ROOT / "Install OmniCortex.bat").read_bytes()
    assert b"\r\n" in bat and b"\n" not in bat.replace(b"\r\n", b"")     # CRLF throughout: cmd's labels need it
    assert b"WindowsApps" in bat and b"winget install" in bat
    for f in ("Install OmniCortex.command", "install-gui.sh"):
        assert subprocess.run(["bash", "-n", str(ROOT / f)]).returncode == 0
        assert (ROOT / f).stat().st_mode & 0o111
