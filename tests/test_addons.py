"""Add-ons (viz/addons.py): built-ins that can be removed, packages from files and repositories, the ComfyUI add-on."""
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
import addons as A  # noqa: E402
import plugins as PL  # noqa: E402
from test_studio import studio_srv  # noqa: E402,F401


def _zip(files: dict, links=()) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for n, t in files.items():
            z.writestr(n, t)
        for n in links:
            zi = zipfile.ZipInfo(n)
            zi.external_attr = (0o120777 << 16)
            z.writestr(zi, "/etc/passwd")
    return b.getvalue()


MANIFEST = json.dumps({"name": "hello", "version": "1.0", "description": "says hello",
                       "jobs": [{"kind": "hi", "script": "hi.py"}]})


def test_unpack_checks(tmp_path):
    name, folder = A.unpack(_zip({"hello/plugin.json": MANIFEST, "hello/hi.py": "print('hi')"}), tmp_path)
    assert name == "hello" and (folder / "hi.py").exists()
    name, folder = A.unpack(_zip({"plugin.json": MANIFEST, "hi.py": "print('v2')"}), tmp_path)   # update in place
    assert (folder / "hi.py").read_text() == "print('v2')"
    for bad in (_zip({"../evil/plugin.json": MANIFEST}), _zip({"x/plugin.json": MANIFEST, "/etc/x": "1"}),
                _zip({"plugin.json": MANIFEST}, links=["link"]), _zip({"readme.txt": "no manifest"}), b"not a zip"):
        with pytest.raises((A.AddonError, PL.PluginError)):
            A.unpack(bad, tmp_path / "bad")
    assert not (tmp_path.parent / "evil").exists()


def test_packages_install_update_uninstall_and_repo(tmp_path):
    inst = tmp_path / "plugins"
    bundled = tmp_path / "bundled"
    (bundled / "ships").mkdir(parents=True)
    (bundled / "ships" / "plugin.json").write_text(json.dumps({"name": "ships"}))
    pm = PL.Plugins([inst, bundled], tmp_path / "state.json")
    repos = []
    pk = A.Packages(pm, inst, lambda: repos, lambda v: repos.__setitem__(slice(None), v))
    data = _zip({"plugin.json": MANIFEST, "hi.py": "print('hi')"})
    with pytest.raises(A.AddonError):
        pk.install_bytes(data, sha256="0" * 64)
    r = pk.install_bytes(data)
    assert r["name"] == "hello" and r["status"] == "off"
    pm.set_enabled("hello", True)
    pk.install_bytes(_zip({"plugin.json": MANIFEST, "hi.py": "print('changed')"}))
    assert pm.get("hello", need_on=False)["status"] == "off"            # an update waits for review again
    with pytest.raises(A.AddonError):
        pk.uninstall("ships")                                          # came with OmniCortex: turn it off instead
    # a repository: pack, index, install by name, sha256 checked
    src = tmp_path / "src" / "greeter"
    src.mkdir(parents=True)
    (src / "plugin.json").write_text(json.dumps({"name": "greeter", "version": "2.0", "description": "greets"}))
    zp = A.pack(src, tmp_path / "dist")
    idx = A.index(tmp_path / "dist")
    pk.set_repos([str(idx)])
    cat = pk.catalog()
    assert cat[0]["name"] == "greeter" and cat[0]["installed"] is None
    pk.install_from_repo("greeter")
    assert pk.catalog()[0]["installed"] == "2.0"
    zp.write_bytes(zp.read_bytes() + b"tampered")
    pk.uninstall("greeter")
    with pytest.raises(A.AddonError):
        pk.install_from_repo("greeter")                                 # no longer matches its sha256
    with pytest.raises(A.AddonError):
        pk.set_repos(["http://example.com/index.json"])


def test_builtins_block_routes_and_jobs():
    off = []
    b = A.Builtins(lambda: off, lambda v: off.__setitem__(slice(None), v))
    assert b.blocks("/features") is None
    b.set("features", False)
    b.set("retraining", False)
    assert b.blocks("/features") == "features" and b.blocks("/api/features/list?x=1") == "features"
    assert b.blocks("/featuresX") is None and b.blocks("/graphs") is None
    assert b.job_hidden("feat_train", {"group": "Interpret"}) and not b.job_hidden("decode_lab", {"group": "Evaluate"})
    assert b.job_hidden("finetune", {"group": "Retrain"}) and not b.job_hidden("testqa", {"group": "Evaluate"})
    assert "/features" in b.nav_off()
    b.set("features", True)
    assert b.blocks("/features") is None
    with pytest.raises(A.AddonError):
        b.set("chat", False)                                            # not removable
    for core in ("remote", "devices", "vision", "transfer"):            # part of the core: always there
        assert core not in A.BUILTIN
    assert not b.job_hidden("clip_collect", {"group": "Vision"})
    # features needs the decoding lab: removing the lab alone is refused, with_dependents takes both
    b.set("features", True)
    with pytest.raises(A.AddonError):
        b.set("decoding_lab", False)
    assert set(b.set("decoding_lab", False, with_dependents=True)) == {"decoding_lab", "features"}
    assert b.node_hidden("emotion_screen") and b.job_hidden("decode_lab", {"group": "Evaluate"})
    assert set(b.set("features", True)) == {"features", "decoding_lab"}  # reinstalling brings back what it needs
    b.set("labs", False)
    assert b.service_hidden("live") and not b.service_hidden("transfer")


def _post(base, path, body, ctype="application/json"):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=20) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_studio_addons(studio_srv, tmp_path):  # noqa: F811
    import studio
    import jobs
    studio.STATE["plugin_dirs"] = [str(tmp_path / "plugins"), str(ROOT / "plugins")]
    studio.STATE.pop("plugins", None)
    studio.STATE["config_path"] = str(tmp_path / "config.json")
    try:
        code, page = _get(studio_srv, "/addons")
        assert code == 200 and "Add-ons" in page
        j = json.loads(_get(studio_srv, "/api/addons")[1])
        assert any(b["name"] == "graphs" for b in j["builtin"]) and any(p["name"] == "comfyui" for p in j["packages"])
        assert _post(studio_srv, "/api/addons/builtin", {"name": "graphs", "installed": False})[0] == 200
        assert _get(studio_srv, "/graphs")[0] == 200 and "removed" in _get(studio_srv, "/graphs")[1]
        assert _get(studio_srv, "/api/graphs")[0] == 404
        assert "/graphs" in json.loads(_get(studio_srv, "/api/addons/nav")[1])["off"]
        _post(studio_srv, "/api/addons/builtin", {"name": "features", "installed": False})
        assert "feat_train" not in jobs.all_specs()
        assert _post(studio_srv, "/api/addons/builtin", {"name": "decoding_lab", "installed": False})[0] == 200
        assert "decode_lab" not in jobs.all_specs()
        _post(studio_srv, "/api/addons/builtin", {"name": "graphs", "installed": True})
        assert not any(t["name"] == "emotion_screen" for t in json.loads(_get(studio_srv, "/api/graphs")[1])["types"])
        _post(studio_srv, "/api/addons/builtin", {"name": "labs", "installed": False})
        lab = json.loads(_get(studio_srv, "/api/lab")[1])["services"]
        assert "transfer" in lab and "live" not in lab
        _post(studio_srv, "/api/addons/builtin", {"name": "labs", "installed": True})
        _post(studio_srv, "/api/addons/builtin", {"name": "features", "installed": True})
        _post(studio_srv, "/api/addons/builtin", {"name": "graphs", "installed": True})
        assert _get(studio_srv, "/api/graphs")[0] == 200 and "feat_train" in jobs.all_specs()
        # a package file
        code, r = _post(studio_srv, "/api/addons/upload", _zip({"plugin.json": MANIFEST, "hi.py": "print(1)"}), "application/zip")
        assert code == 200 and r["name"] == "hello" and r["status"] == "off"
        assert _post(studio_srv, "/api/addons/upload", b"junk", "application/zip")[0] == 400
        # the ComfyUI add-on: its tab appears in the sidebar once it is on
        assert _post(studio_srv, "/api/plugins/enable", {"name": "comfyui", "on": True})[0] == 200
        nav = json.loads(_get(studio_srv, "/api/addons/nav")[1])
        assert {"href": "/plugins/comfyui/", "label": "ComfyUI"}.items() <= nav["tabs"][0].items()
        assert "plugin:comfyui:install" in jobs.all_specs()
        assert "not running" in _get(studio_srv, "/plugins/comfyui/")[1]
        assert _post(studio_srv, "/api/addons/uninstall", {"name": "hello"})[0] == 200
        assert _post(studio_srv, "/api/addons/uninstall", {"name": "comfyui"})[0] == 400        # ships with OmniCortex
    finally:
        studio.STATE["addons_off"] = []
        studio.STATE.pop("plugin_dirs", None)
        studio.STATE.pop("plugins", None)


def test_comfyui_addon_files():
    m = PL.read_manifest(ROOT / "plugins" / "comfyui")
    assert m["tab"]["label"] == "ComfyUI" and {j["kind"] for j in m["jobs"]} == {"install", "run", "update", "node"}
    import subprocess
    for s in ("install.py", "run.py", "update.py", "node.py"):
        out = subprocess.run([sys.executable, str(ROOT / "plugins" / "comfyui" / s), "--help"], capture_output=True,
                             text=True, cwd=ROOT / "plugins" / "comfyui")
        assert out.returncode == 0, out.stderr
        job = next(j for j in m["jobs"] if j["script"] == s)
        for f in job["fields"]:
            assert "--" + f["name"].replace("_", "-") in out.stdout
    r = subprocess.run([sys.executable, "run.py", "--folder", "/nonexistent/comfy"], capture_output=True, text=True,
                       cwd=ROOT / "plugins" / "comfyui")
    assert r.returncode != 0 and "install" in r.stderr
    sys.path.insert(0, str(ROOT / "plugins" / "comfyui" / "comfy_nodes"))
    import neuronscope_nodes as NN
    assert set(NN.NODE_CLASS_MAPPINGS) == {"NeuronScopeChat", "NeuronScopeCheck", "NeuronScopeGraph"}
    for c in NN.NODE_CLASS_MAPPINGS.values():
        assert c.INPUT_TYPES()["required"] and c.FUNCTION == "run"


def test_default_addons(studio_srv, tmp_path):  # noqa: F811
    """Add-ons that ship with OmniCortex and are on unless turned off: the Godot client, jewel styles,
    the theme pack and the graph screens. ComfyUI ships too but waits to be turned on."""
    import studio
    import graph_engine as G
    studio.STATE["plugin_dirs"] = [str(tmp_path / "plugins"), str(ROOT / "plugins")]
    studio.STATE.pop("plugins", None)
    studio.STATE["config_path"] = str(tmp_path / "config.json")
    try:
        pk = {p["name"]: p for p in json.loads(_get(studio_srv, "/api/addons")[1])["packages"]}
        for name in ("godot-client", "jewel-styles", "theme-pack", "graph-screens"):
            assert pk[name]["status"] == "on" and pk[name]["default"], name
        assert pk["comfyui"]["status"] == "off" and pk["claim-spotter"]["status"] == "off"
        # the Godot client: a tab, its jobs, and a download for another computer
        nav = json.loads(_get(studio_srv, "/api/addons/nav")[1])
        assert any(t["href"] == "/plugins/godot-client/" for t in nav["tabs"])
        with urllib.request.urlopen(studio_srv + "/plugins/godot-client/download/neuronscope-godot.zip", timeout=20) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        names = z.namelist()
        assert "neuronscope-godot/project.godot" in names and "neuronscope-godot/widgets/jewel.gd" in names
        assert not any("/.godot/" in n for n in names)
        # the theme pack's themes are listed while it is on
        assert "arcade" in json.loads(_get(studio_srv, "/api/themes")[1])["themes"]
        # graph screens from the node pack
        cat = json.loads(_get(studio_srv, "/api/graphs")[1])
        assert {"plugin:graph-screens:code_security", "plugin:graph-screens:injection_screen"} <= {t["name"] for t in cat["types"]}
        g = {"nodes": [{"id": "in", "type": "input", "params": {}}, {"id": "s", "type": "plugin:graph-screens:code_security",
                                                                     "params": {"fail_on": "medium"}},
                       {"id": "o", "type": "output", "params": {"name": "bad"}}],
             "edges": [{"from": ["in", "value"], "to": ["s", "code"]}, {"from": ["s", "fail"], "to": ["o", "value"]}]}
        r = G.Runner(g, G.Env(policy=None)).start({"input": "requests.get(u, verify=False)"})
        assert "certificate checks" in r["outputs"]["bad"]
        # turning a default add-on off sticks, and its parts go
        assert _post(studio_srv, "/api/plugins/enable", {"name": "theme-pack", "on": False})[0] == 200
        assert "arcade" not in json.loads(_get(studio_srv, "/api/themes")[1])["themes"]
        assert _post(studio_srv, "/api/plugins/enable", {"name": "theme-pack", "on": True})[0] == 200
        assert "arcade" in json.loads(_get(studio_srv, "/api/themes")[1])["themes"]
    finally:
        studio.STATE.pop("plugin_dirs", None)
        studio.STATE.pop("plugins", None)


def test_godot_client_jobs():
    import subprocess
    m = PL.read_manifest(ROOT / "plugins" / "godot-client")
    for j in m["jobs"]:
        out = subprocess.run([sys.executable, str(ROOT / "plugins" / "godot-client" / j["script"]), "--help"],
                             capture_output=True, text=True)
        assert out.returncode == 0 and all("--" + f["name"] in out.stdout for f in j["fields"])
