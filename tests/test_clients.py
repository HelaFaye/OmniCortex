"""The terminal client (scripts/oc.py) against a running Studio, and the Godot
client's scripts (parsed when a Godot 4 binary is available: $OC_GODOT or
godot4/godot on PATH)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_studio import CODER, studio_srv  # noqa: E402,F401
import oc  # noqa: E402


def run(base, *args, capsys):
    rc = oc.main(["--studio", base, *args])
    return rc, capsys.readouterr().out


def test_terminal_client(studio_srv, capsys, tmp_path):  # noqa: F811
    import director
    import studio
    studio.DIRECTOR["store"] = director.ProjectStore(tmp_path / "projects")
    rc, out = run(studio_srv, "list-models", capsys=capsys)
    assert rc == 0 and CODER in out and "no graded results" in out
    rc, out = run(studio_srv, "chat", "--model", CODER, "hello", capsys=capsys)
    assert rc == 0 and f"model={CODER}" in out
    rc, out = run(studio_srv, "subject-defaults", "--set", f"code={CODER}", capsys=capsys)
    assert rc == 0 and "(your pick)" in out
    rc, out = run(studio_srv, "tool-approval", "--kind", "read_only=allow", capsys=capsys)
    assert rc == 0 and "read_only allow" in out
    rc, out = run(studio_srv, "workspaces", capsys=capsys)
    assert rc == 0 and "default" in out
    rc, out = run(studio_srv, "start-job", "--kind", "env_snapshot", "--values", "{}", capsys=capsys)
    assert rc == 0
    jid = __import__("json").loads(out)["id"]
    rc, out = run(studio_srv, "follow", jid, capsys=capsys)
    assert rc == 0 and "[done]" in out
    rc, out = run(studio_srv, "get-project", "--id", "nope", capsys=capsys)
    assert rc == 1


def _godot():
    g = os.environ.get("OC_GODOT") or shutil.which("godot4") or shutil.which("godot")
    return g if g and os.path.exists(g) else None


@pytest.mark.skipif(_godot() is None, reason="no Godot 4 binary ($OC_GODOT, godot4 or godot)")
@pytest.mark.parametrize("script", ["studio.gd", "main.gd", "app.gd", "widgets/widget.gd", "widgets/jewel.gd",
                                    "widgets/emotion_graph.gd", "widgets/nav_gizmo.gd", "widgets/chat_box.gd"])
def test_godot_scripts_parse(script):
    r = subprocess.run([_godot(), "--headless", "--path", str(ROOT / "plugins" / "godot-client" / "client"), "--check-only",
                        "--script", f"res://{script}"], capture_output=True, text=True, timeout=120)
    assert "SCRIPT ERROR" not in r.stdout + r.stderr and "Parse Error" not in r.stdout + r.stderr, r.stderr[-800:]
