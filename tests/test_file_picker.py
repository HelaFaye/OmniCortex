"""Files and folders are chosen with Browse… and edited one entry at a time."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "scripts"))
import studio  # noqa: E402
import ui_links  # noqa: E402
from test_studio import call, studio_srv  # noqa: E402,F401


def test_fs_list_filters_and_sorts(tmp_path):
    (tmp_path / "b-dir").mkdir()
    (tmp_path / "A-dir").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "m.gguf").write_bytes(b"x" * 10)
    (tmp_path / "notes.txt").write_text("x")
    r = studio.fs_list(str(tmp_path), want="file", exts=("gguf",))
    names = [e["name"] for e in r["entries"]]
    assert names == ["A-dir", "b-dir", "m.gguf"]                  # folders first, then matching files
    assert r["entries"][2]["size"] == 10 and r["parent"] == str(tmp_path.parent)
    assert [e["name"] for e in studio.fs_list(str(tmp_path), want="dir")["entries"]] == ["A-dir", "b-dir"]
    assert ".hidden" in [e["name"] for e in studio.fs_list(str(tmp_path), want="dir", hidden=True)["entries"]]
    # A path that does not exist yet opens its nearest existing folder.
    assert studio.fs_list(str(tmp_path / "nope" / "deeper"))["path"] == str(tmp_path)
    assert studio.fs_list(str(tmp_path), limit=1)["truncated"]


def test_places_exist():
    ps = studio.fs_places()
    assert ps and all(Path(p["path"]).is_dir() for p in ps)
    assert len({p["path"] for p in ps}) == len(ps)


def test_fs_api_and_pages(studio_srv, tmp_path):
    code, j = call(studio_srv, "/api/fs/places")
    assert code == 200 and j["places"]
    code, j = call(studio_srv, "/api/fs/list?want=file&ext=gguf&path=" + str(tmp_path / "models" / "acme" / "Coder-GGUF"))
    assert code == 200 and [e["name"] for e in j["entries"]] == ["coder-7b-Q4_K_M.gguf"]
    _, setup = call(studio_srv, "/setup")
    assert "nsListEditor" in setup and "nsPickPath" in setup and 'data-pick="${pick}"' in setup
    _, settings = call(studio_srv, "/settings")
    assert "data-wl=" in settings
    _, jobs = call(studio_srv, "/jobs")
    assert "dl-ggufs" in jobs
    _, defaults = call(studio_srv, "/api/jobs/defaults")
    assert any(p.endswith("coder-7b-Q4_K_M.gguf") for p in defaults["ggufs"])
    _, home = call(studio_srv, "/")
    assert 'id="lora"' in home and re.search(r'id="lora"[^>]*data-pick="file"', home)


def test_picker_js_parses():
    import shutil
    import subprocess
    if not shutil.which("node"):
        return
    out = subprocess.run(["node", "--check", "-"], input=ui_links.js(), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
