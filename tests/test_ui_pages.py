"""Every embedded web page's inline JavaScript must at least parse.

Two lab pages shipped with syntax/reference errors that only showed up in a
browser; this catches that class of bug without one. Needs `node` (skipped
otherwise)."""
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "viz"))

PAGES = [
    ("viz/studio.py", "PAGE"), ("viz/studio.py", "LOGIN"),
    ("viz/hub.py", "PAGE"), ("viz/hub.py", "LOGIN"),
    ("viz/control.py", "PAGE"), ("viz/control.py", "LOGIN"),
    ("viz/stream.py", "PAGE"), ("viz/stream.py", "LOGIN"),
    ("viz/server.py", "PAGE"), ("viz/bloom.py", "PAGE"),
    ("viz/quant_lab.py", "HTML"), ("viz/scale_sweep_lab.py", "HTML"),
    ("viz/adaptive_tuning_lab.py", "HTML"), ("viz/jobs.py", "PAGE"), ("viz/studio.py", "PAIR_PAGE"), ("viz/studio.py", "LINK_PAGE"),
    ("viz/projects_page.py", "PAGE"), ("viz/connect_page.py", "PAGE"), ("viz/setup_page.py", "PAGE"),
    ("viz/lab_page.py", "PAGE"), ("viz/review_page.py", "PAGE"), ("viz/settings_page.py", "PAGE"),
    ("viz/system_page.py", "PAGE"),
    ("viz/features_page.py", "PAGE"), ("viz/graphs_page.py", "PAGE"), ("viz/work_page.py", "PAGE"),
]
PLACEHOLDERS = {"__MODELS__": '""', "__PROFILES__": '""'}


def scripts_of(html: str) -> list[tuple[str, str]]:
    """(type attribute, body) for every inline <script>."""
    out = []
    for attrs, body in re.findall(r"<script((?![^>]*\bsrc=)[^>]*)>(.*?)</script>", html, re.S):
        m = re.search(r"type=[\"']?([\w/-]+)", attrs)
        out.append((m.group(1) if m else "", body))
    return out


def load_const(path, name):
    # A module that is already imported is used as it is: a second copy of studio.py would point
    # shared hooks (the jobs list's plugin jobs) at its own state, and later tests would see that.
    loaded = sys.modules.get(Path(path).stem)
    if loaded is not None and Path(getattr(loaded, "__file__", "") or "").resolve() == (ROOT / path).resolve():
        return getattr(loaded, name)
    spec = importlib.util.spec_from_file_location(f"page_{Path(path).stem}", ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("path,name", PAGES + [("web/model_transfer.html", None)])
def test_inline_js_parses(path, name, tmp_path):
    html = (ROOT / path).read_text() if name is None else load_const(path, name)
    for k, v in PLACEHOLDERS.items():
        html = html.replace(k, v)
    for i, (kind, js) in enumerate(scripts_of(html)):
        if kind == "importmap":
            json.loads(js)
            continue
        if kind == "module":
            f = tmp_path / f"b{i}.mjs"
            f.write_text(js)
        else:
            f = tmp_path / f"b{i}.js"
            # Some pages use top-level await; wrap classic scripts in an async body.
            f.write_text("(async () => {\n" + js + "\n})();")
        r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, f"{path}:{name} block {i}: {r.stderr[:600]}"


SERVED = [("viz/quant_lab.py", []), ("viz/scale_sweep_lab.py", []), ("viz/adaptive_tuning_lab.py", []),
          ("viz/server.py", []), ("viz/control.py", []), ("viz/hub.py", []),
          ("viz/stream.py", ["--simulate"]), ("viz/transfer_lab.py", [])]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("path,extra", SERVED)
def test_served_page_has_no_template_leftovers(path, extra, tmp_path):
    """Render the page the way the server does (placeholders filled at request
    time) and check it parses; this is where the Quantization Lab broke."""
    import socket
    import time
    import urllib.request
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    proc = subprocess.Popen([sys.executable, str(ROOT / path), "--port", str(port), *extra],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        html = None
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as r:
                    html = r.read().decode()
                break
            except Exception:
                if proc.poll() is not None:
                    pytest.fail(proc.stderr.read().decode()[-800:])
                time.sleep(0.2)
        assert html, "server did not answer"
        assert not re.search(r"__[A-Z]{3,}__", html), "unfilled template placeholder"
        for i, (kind, js) in enumerate(scripts_of(html)):
            if kind == "importmap":
                continue
            f = tmp_path / (f"b{i}.mjs" if kind == "module" else f"b{i}.js")
            f.write_text(js if kind == "module" else "(async () => {\n" + js + "\n})();")
            r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
            assert r.returncode == 0, f"{path} block {i}: {r.stderr[:600]}"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_hub_lists_trace_sessions_for_replay(tmp_path, monkeypatch):
    import hub
    for name, kind, created in [("a", "trace", "2026-01-01T00:00:00"), ("b", "trace", "2026-02-01T00:00:00"),
                                ("c", "profile", "2026-03-01T00:00:00")]:
        (tmp_path / "runs" / name).mkdir(parents=True)
        (tmp_path / "runs" / name / "manifest.json").write_text(json.dumps({"kind": kind, "created": created,
                                                                            "model": "m.gguf"}))
    (tmp_path / "runs" / "broken").mkdir()
    monkeypatch.setitem(hub.STATE, "root", str(tmp_path))
    monkeypatch.setattr(hub, "TRACE_DIRS", ["runs", str(tmp_path / "missing")])
    assert [t["path"] for t in hub.find_traces()] == [str(Path("runs") / "b"), str(Path("runs") / "a")]


def test_no_dead_tabs():
    """Every tab in the registry is implemented by at least one surface; no
    'planned' placeholders, and no enabled package without an implementation."""
    import shell
    for m in shell.modes():
        assert not m.get("planned"), m["id"]
        assert m["id"] != "eval", "the example Eval package has nothing behind its route"
