"""oc_requirements: the feature catalog, version checks, per-OS fixes and
environment snapshots."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import oc_requirements as nr  # noqa: E402


def test_specs_come_from_the_requirements_files():
    specs = nr.requirement_specs()
    assert specs["scikit-learn"]["spec"] == ">=1.4" and specs["scikit-learn"]["file"] == "requirements-core.txt"
    assert specs["accelerate"]["file"] == "requirements.txt"
    assert "torch" not in specs                                   # install.sh decides torch
    # Every catalog feature names packages and tools the catalog knows.
    for f in nr.FEATURES.values():
        for p in f.get("packages", []) + f.get("optional_packages", []):
            assert p in nr.PACKAGES
        for t in f.get("tools", []) + f.get("optional_tools", []):
            assert t in nr.TOOLS


@pytest.mark.parametrize("v,spec,ok", [("1.5.2", ">=1.4", True), ("1.3", ">=1.4", False), ("3.11.0", ">=3.10,<4", True),
                                       ("4.0", ">=3.10,<4", False), ("2.3.1", "", True), ("24.1", ">=24", True)])
def test_satisfies(v, spec, ok):
    assert nr.satisfies(v, spec) is ok


def test_install_commands_per_os():
    assert nr.install_cmd("apt", "cmake") == "sudo apt install cmake"
    assert nr.install_cmd("pacman", "shaderc vulkan-headers") == "sudo pacman -S shaderc vulkan-headers"
    assert nr.install_cmd("dnf", "cuda (NVIDIA repo)") == "cuda (NVIDIA repo)"
    assert nr.install_cmd("apt", None) == ""


def test_check_reports_features_and_skips_absent_vendors(monkeypatch):
    rep = nr.check(hw={"ram_gib": 32.0, "devices": []}, vendors={"amd"})
    f = rep["features"]
    assert f["core"]["status"] in ("ok", "warn")                  # this test suite runs on the core install
    assert f["nvidia_cuda"]["status"] == "n/a" and "NVIDIA" in f["nvidia_cuda"]["note"]
    rocm = {i["name"]: i for i in f["amd_rocm"]["items"]}
    assert "/dev/kfd readable and writable (ROCm compute)" in rocm and "rocminfo" in rocm
    assert rocm["llama-server with hip"]["optional"]
    # A missing package names the pip command with the files' version floor.
    monkeypatch.setattr(nr, "dist_version", lambda name: None)
    rep = nr.check(["core"], hw={"ram_gib": 32.0}, vendors=set())
    sk = next(i for i in rep["features"]["core"]["items"] if i["name"] == "scikit-learn")
    assert sk["status"] == "missing" and sk["fix"] == "pip install 'scikit-learn>=1.4'" and sk["installable"]
    rep = nr.check(["torch_pipeline"], hw={"ram_gib": 8.0}, vendors=set())
    items = {i["name"]: i for i in rep["features"]["torch_pipeline"]["items"]}
    assert not items["torch"]["installable"] and "install.sh" in items["torch"]["fix"]
    assert items["memory"]["status"] == "warn"
    with pytest.raises(ValueError):
        nr.check(["nonsense"])


def test_snapshot_diff_and_freeze(tmp_path):
    a = {"when": "1", "os": "Linux 6.1", "python": "3.11", "packages": {"numpy": "1.26", "old": "1"},
         "tools": {"cmake": "3.28"}, "llama": {"llama-server": {"built": "2026-01-01", "backends": []}},
         "hardware": {"devices": [{"id": "cpu:0", "name": "x", "backend": "cpu", "memory_gib": 16}]}}
    b = json.loads(json.dumps(a))
    b.update(when="2", os="Linux 6.8")
    b["packages"] = {"numpy": "2.0", "new": "3"}
    b["llama"]["llama-server"] = {"built": "2026-02-01", "backends": ["vulkan"]}
    b["hardware"]["devices"].append({"id": "vulkan:0", "name": "gfx90c", "backend": "vulkan", "memory_gib": 16})
    d = nr.diff(a, b)
    assert d["packages"]["changed"] == {"numpy": ["1.26", "2.0"]} and d["packages"]["added"] == {"new": "3"}
    assert d["packages"]["removed"] == {"old": "1"} and d["system"]["os"] == ["Linux 6.1", "Linux 6.8"]
    assert "vulkan" in d["llama"]["changed"]["llama-server"][1] and "vulkan:0" in d["devices"]["added"]
    assert not d["unchanged"] and nr.diff(a, a)["unchanged"]
    p = nr.save_snapshot({**a, "time": 1767225600.0}, tmp_path)
    assert [s["id"] for s in nr.list_snapshots(tmp_path)] == [p.stem]
    assert nr.load_snapshot(p.stem, tmp_path)["os"] == "Linux 6.1"
    with pytest.raises(ValueError):
        nr.load_snapshot("../../etc/passwd", tmp_path)
    assert "numpy==" in nr.freeze()


def test_install_refuses_outside_the_catalog():
    with pytest.raises(SystemExit, match="not in the catalog"):
        nr.pip_install("evil-package")
    with pytest.raises(SystemExit, match="install.sh"):
        nr.pip_install("torch")


def test_cli_json():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "oc_requirements.py"), "check", "--feature", "core",
                        "--json"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["features"]["core"]["title"].startswith("Core")


# ---------------------------------------------------------------- per project

def _checkout(root):
    (root / "src").mkdir(parents=True)
    (root / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.22)\nproject(demo)\nfind_package(SDL3 REQUIRED)\nfind_package(Threads)\n"
        "find_package(MyWeirdLib)\nfind_package(PkgConfig)\npkg_check_modules(GL REQUIRED IMPORTED_TARGET gl glfw3>=3.3)\n")
    (root / "xmake.lua").write_text('add_requires("zlib", "imgui 1.90")\ntarget("demo")\n')
    (root / "package.json").write_text(json.dumps({"engines": {"node": ">=18"}}))
    (root / "pnpm-lock.yaml").write_text("")
    (root / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.10"\ndependencies = ["numpy>=1.20", "rich"]\n')
    (root / "requirements-dev.txt").write_text("pytest>=7\n")
    (root / "go.mod").write_text("module x\n\ngo 1.22\n")
    (root / "src" / "main.c").write_text("int main(){}")
    (root / "node_modules" / "deep").mkdir(parents=True)
    (root / "node_modules" / "deep" / "CMakeLists.txt").write_text("find_package(Nope)")
    return root


def test_infer_project_reads_build_files(tmp_path):
    r = nr.infer_project(str(_checkout(tmp_path / "p")))
    got = {(i["kind"], i["name"]): i for i in r["items"]}
    assert got[("tool", "cmake")]["need"] == ">=3.22" and ("tool", "c++ compiler") in got
    assert ("library", "sdl3") in got and not got[("library", "sdl3")]["optional"]
    assert got[("library", "myweirdlib")]["optional"]              # unknown CMake package: maybe not pkg-config
    assert got[("library", "glfw3")]["need"] == ">=3.3" and ("library", "gl") in got
    assert ("tool", "pkg-config") in got and ("tool", "xmake") in got
    assert any(i["kind"] == "note" and "zlib" in i["name"] for i in r["items"])
    assert got[("tool", "node")]["need"] == ">=18" and ("tool", "pnpm") in got
    assert got[("tool", "python3")]["need"] == ">=3.10" and got[("package", "numpy")]["need"] == ">=1.20"
    assert got[("package", "pytest")]["optional"]                  # a dev requirements file
    assert got[("tool", "go")]["need"] == ">=1.22"
    assert ("library", "nope") not in got                           # node_modules is not the project
    assert ("library", "threads") not in got and r["venv"] is None
    with pytest.raises(ValueError):
        nr.infer_project(str(tmp_path / "missing"))


def test_normalize_items_validates():
    ok = nr.normalize_items([{"kind": "tool", "name": "cmake", "need": ">= 3.20"},
                             {"kind": "hardware", "name": "ram_gib", "need": "32"},
                             {"kind": "hardware", "name": "gpu", "need": "amd"},
                             {"kind": "feature", "name": "amd_vulkan"}, {"kind": "note", "name": "a dev kit"},
                             {"kind": "tool", "name": "CMake"}])
    assert len(ok) == 5 and ok[0]["need"] == ">=3.20" and ok[0]["source"] == "human"
    for bad in ([{"kind": "shell", "name": "x"}], [{"kind": "tool", "name": "rm -rf /;"}],
                [{"kind": "tool", "name": "x", "need": "$(evil)"}], [{"kind": "hardware", "name": "ram_gib", "need": "lots"}],
                [{"kind": "hardware", "name": "cpu", "need": "1"}], [{"kind": "feature", "name": "nope"}],
                [{"kind": "note", "name": "two\nlines"}], "notalist"):
        with pytest.raises(ValueError):
            nr.normalize_items(bad)


def test_check_project(monkeypatch):
    monkeypatch.setattr(nr, "library_version", lambda n: {"sdl3": "3.2.0"}.get(n))
    spec = {"items": nr.normalize_items([
        {"kind": "tool", "name": "git"}, {"kind": "tool", "name": "definitely-not-a-tool-xyz"},
        {"kind": "tool", "name": "git", "need": ">=999"},
        {"kind": "library", "name": "sdl3", "need": ">=3.1"}, {"kind": "library", "name": "glfw3", "optional": True},
        {"kind": "package", "name": "numpy"}, {"kind": "package", "name": "not-a-package-xyz", "need": ">=1"},
        {"kind": "hardware", "name": "ram_gib", "need": "1"}, {"kind": "hardware", "name": "vram_gib", "need": "8"},
        {"kind": "hardware", "name": "gpu", "need": "amd"}, {"kind": "feature", "name": "core"},
        {"kind": "note", "name": "check the dev kit by hand"}])}
    r = nr.check_project(spec, hw={"ram_gib": 16.0, "devices": [], "disk_free_gib": 50}, vendors=set())
    st = {(i["kind"], i["name"]): i for i in r["items"]}
    if shutil_which("git"):
        assert st[("tool", "git")]["status"] == "ok"
    t = st[("tool", "definitely-not-a-tool-xyz")]
    assert t["status"] == "missing" and t["fix"]
    assert st[("library", "sdl3")]["status"] == "ok" and st[("library", "glfw3")]["status"] == "warn"
    assert st[("package", "numpy")]["status"] == "ok"
    assert st[("package", "not-a-package-xyz")]["fix"] == "pip install 'not-a-package-xyz>=1'"
    assert st[("hardware", "ram_gib")]["status"] == "ok" and st[("hardware", "vram_gib")]["status"] == "warn"
    assert st[("hardware", "gpu")]["status"] == "warn" and st[("feature", "core")]["status"] == "ok"
    assert st[("note", "check the dev kit by hand")]["status"] == "note"
    assert r["status"] == "missing" and set(r["missing"]) == {"definitely-not-a-tool-xyz", "not-a-package-xyz"}
    brief = nr.environment_brief(r)
    assert "numpy" in brief and "Not available" in brief and "definitely-not-a-tool-xyz" in brief


def test_project_venv_is_used(tmp_path):
    venv = tmp_path / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(sys.executable)
    r = nr.infer_project(str(tmp_path))
    assert r["venv"] == str(tmp_path / ".venv")
    out = nr.check_project({"venv": r["venv"], "items": nr.normalize_items([{"kind": "package", "name": "numpy"}])},
                           hw={"devices": []}, vendors=set())
    assert out["items"][0]["status"] == "ok" and "virtualenv" in out["python_from"]


def shutil_which(n):
    import shutil
    return shutil.which(n)


def test_director_keeps_requirements(tmp_path):
    import director as dr
    root = _checkout(tmp_path / "p")
    census = dr.analyze_repo(str(root))
    plan = dr.new_plan("Demo", "", dr.analyze_text("- Build: set up CMake."), census=census)
    items = plan["requirements"]["items"]
    assert any(i["name"] == "sdl3" for i in items) and plan["requirements"]["root"] == str(root.resolve())
    # A person edits: drops pnpm, adds 32 GiB RAM; inferred items left alone stay inferred.
    keep = [i for i in items if i["name"] != "pnpm"] + [{"kind": "hardware", "name": "ram_gib", "need": "32"}]
    dr.set_requirements(plan, keep, reason="trim")
    srcs = {i["name"]: i["source"] for i in plan["requirements"]["items"]}
    assert "pnpm" not in srcs and srcs["ram_gib"] == "human" and srcs["sdl3"] == "inferred"
    assert plan["history"][-1]["changes"][0]["removed"] == ["tool:pnpm"]
    # Re-scanning keeps the person's items and brings inferred ones back.
    dr.scan_requirements(plan, str(root), by="human")
    names = [i["name"] for i in plan["requirements"]["items"]]
    assert "ram_gib" in names and "pnpm" in names
    with pytest.raises(dr.PlanError):
        dr.set_requirements(plan, [{"kind": "tool", "name": "a;b"}])
    last = dr.check_requirements(plan, hw={"ram_gib": 8.0, "devices": []}, vendors=set())
    assert plan["requirements"]["last"] is last
    t = plan["tasks"][0]
    msg = dr.worker_messages(plan, t)[1]["content"]
    assert "Environment:" in msg
