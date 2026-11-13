"""The unified review lookup (scripts/unified_review.py), the browser extension's JS twin, and Studio's /api/lookup."""
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import unified_review as U  # noqa: E402

BOARD = json.loads((ROOT / "tests" / "fixtures" / "lookup_board.json").read_text())
EXT = ROOT / "extension"
NODE = shutil.which("node")


def test_names():
    assert U.base_name("bartowski/Qwen2.5-7B-Instruct-GGUF") == "qwen2.5-7b-instruct"
    assert U.base_name("x/Llama-3-8B-AWQ-4bit") == "llama-3-8b"
    assert U.relation("Qwen/Qwen2.5-7B-Instruct", "qwen/qwen2.5-7b-instruct") == "same"
    assert U.relation("Qwen/Qwen2.5-7B-Instruct", "bartowski/Qwen2.5-7B-Instruct-GGUF") == "related"
    assert U.relation("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct") is None
    with pytest.raises(ValueError):
        U.lookup("not a repo")


def test_lookup_board_and_hw_reports():
    rec = U.lookup("Qwen/Qwen2.5-7B-Instruct", [("board:o/b", BOARD), ("board:down", OSError("offline"))])
    assert rec["available"] and rec["schema"] == U.SCHEMA
    assert [s["ok"] for s in rec["sources"]] == [True, False]
    keys = [(s["key"], s["related"]) for s in rec["stats"]]
    assert ("accuracy", False) in keys and ("accuracy", True) in keys and ("accuracy:math", False) in keys
    acc = next(s for s in rec["stats"] if s["key"] == "accuracy" and not s["related"])
    assert acc["variant"] == "Q4_K_M · llama.cpp · suppression 0.5" and acc["n"] == 400
    emo = [s["label"] for s in rec["stats"] if s["group"] == "emotions"]
    assert emo == ["desperate", "calm"]                                     # strongest first
    assert all(s["detail"]["self_reported"] for s in rec["stats"] if s["group"] in ("emotions", "features"))
    hw = [s for s in rec["stats"] if s["key"].startswith("hw:")]
    assert [s["key"] for s in hw] == ["hw:nvidia-legacy-sm50-70:inference"] and hw[0]["value"] == 0.667
    assert not any("Other" in s["variant"] for s in rec["stats"])
    assert not U.lookup("nobody/Unknown-1B", [("board:o/b", BOARD)])["available"]


def test_extras_are_sanitized_and_aggregated():
    ex = U.sanitize_extras({"emotions": {"Calm<script>": 0.2, "x": "high", "": 1},
                            "features": [{"id": 3, "freq": 0.1, "label": "<b>a</b>" * 30}, {"id": 4}], "other": 1})
    assert ex["emotions"] == {"calmscript": 0.2} and len(ex["features"]) == 1
    assert "<" not in ex["features"][0]["label"] and len(ex["features"][0]["label"]) <= 80
    agg = U.aggregate_extras([{"extras": {"emotions": {"calm": 0.2}}}, {"extras": {"emotions": {"calm": 0.4}}}, {}])
    assert agg["emotions"]["calm"] == {"mean": 0.3, "n": 2}
    import scoreboard_server as sb
    card = {"model": {"sha256": "s", "file": "f"}, "variant": {"engine": "llama.cpp"}, "hardware": {},
            "test": {"per_subject": {"m": {"n": 1, "correct": 1}}}, "extras": {"emotions": {"calm": 0.2}}}
    assert sb.aggregate([card])[0]["extras"]["emotions"]["calm"]["n"] == 1


def test_saelens_and_local(tmp_path):
    reg = {"gemma-scope-2b-pt-res": {"model": "gemma-2-2b", "repo_id": "google/gemma-scope-2b-pt-res",
                                     "saes": [{"id": "layer_0/width_16k"}]},
           "gpt2-small-res-jb": {"model": "gpt2-small", "saes": []}}
    import neuron_review as nr
    st = nr.ReviewStore(tmp_path / "rev")
    st.record("gemma-gguf/gemma-2-2b-q4_k_m", np.ones((2, 8)), source="live", kind="observed", verdict="wrong")
    st.record("gemma-gguf/gemma-2-2b-q4_k_m", np.ones((2, 8)), source="live", kind="observed")
    rec = U.lookup("google/gemma-2-2b", [], reg, st)
    by = {s["key"]: s for s in rec["stats"]}
    assert by["saelens:gemma-scope-2b-pt-res"]["value"] == 1 and "saelens:gpt2-small-res-jb" not in by
    assert by["observations"]["value"] == 2 and by["local_flagged"]["value"] == 0.5
    assert {s["id"] for s in rec["sources"]} == {"saelens", "local"}


def node(script: str) -> str:
    return subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True, timeout=60).stdout


@pytest.mark.skipif(not NODE, reason="node is not installed")
def test_js_matches_python():
    src = EXT / "src"
    out = node(f"""require({json.dumps(str(src / 'normalize.js'))});
const b = {json.dumps(BOARD)};
console.log(JSON.stringify(OC_normalize.boardStats(b, "Qwen/Qwen2.5-7B-Instruct", "board:o/b")));""")
    js = json.loads(out)
    py = U.lookup("Qwen/Qwen2.5-7B-Instruct", [("board:o/b", BOARD)])["stats"]
    assert js == py


@pytest.mark.skipif(not NODE, reason="node is not installed")
def test_site_detection():
    urls = ["https://huggingface.co/Qwen/Qwen2.5-7B-Instruct", "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/main",
            "https://huggingface.co/datasets/x/y", "https://huggingface.co/models?search=qwen", "https://huggingface.co/Qwen",
            "https://hf.co/a/b", "https://www.modelscope.cn/models/qwen/Qwen2-7B", "https://lmstudio.ai/models/qwen/qwen3-8b",
            "http://huggingface.co/a/b", "https://evil.example/huggingface.co/a/b"]
    out = node(f"""require({json.dumps(str(EXT / 'src' / 'sites.js'))});
console.log(JSON.stringify({json.dumps(urls)}.map(u => (OC_sites.modelFromUrl(u) || {{}}).model || null)));""")
    assert json.loads(out) == ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", None, None, None, "a/b",
                               "qwen/Qwen2-7B", "qwen/qwen3-8b", None, None]


def test_build(tmp_path):
    sys.path.insert(0, str(EXT))
    import build
    zips = build.build(tmp_path, "1.2.3")
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            names = set(zf.namelist())
            m = json.loads(zf.read("manifest.json"))
        assert {"popup.html", "background.js", "sites.js", "normalize.js", "icons/lit-32.png", "options.html"} <= names
        assert m["manifest_version"] == 3 and m["version"] == "1.2.3"
        for f in m["content_scripts"][0]["js"] + [m["action"]["default_popup"], m["options_ui"]["page"]]:
            assert f in names
    chrome, ff = (json.loads(zipfile.ZipFile(z).read("manifest.json")) for z in zips)
    assert chrome["background"] == {"service_worker": "background.js"}
    assert ff["background"]["scripts"][-1] == "background.js" and "gecko_android" in ff["browser_specific_settings"]
    if NODE:
        for f in (EXT / "src").glob("*.js"):
            subprocess.run([NODE, "--check", str(f)], check=True)


def test_studio_lookup(tmp_path, monkeypatch):
    import test_studio
    studio = test_studio.studio
    monkeypatch.setenv("OC_HOME", str(tmp_path / "home"))
    (tmp_path / "home" / "saelens").mkdir(parents=True)
    (tmp_path / "home" / "saelens" / "pretrained_saes.yaml").write_text(
        "gemma-scope-2b-pt-res:\n  model: gemma-2-2b\n  repo_id: google/gemma-scope-2b-pt-res\n  saes:\n  - id: a\n")
    gen = test_studio.studio_srv.__wrapped__(tmp_path)
    base = next(gen)
    try:
        monkeypatch.setattr(studio, "lookup_board", lambda: ("board:o/b", BOARD))
        code, r = test_studio.call(base, "/api/lookup?model=Qwen/Qwen2.5-7B-Instruct")
        assert code == 200 and r["available"] and r["open"]["hub"] == "/#hub=Qwen/Qwen2.5-7B-Instruct"
        assert {s["id"] for s in r["sources"]} == {"board:o/b", "saelens", "local"}
        code, r = test_studio.call(base, "/api/lookup?model=google/gemma-2-2b")
        assert any(s["key"] == "saelens:gemma-scope-2b-pt-res" for s in r["stats"])
        assert test_studio.call(base, "/api/lookup?model=bad")[0] == 400
        assert "q.get('hub')" in test_studio.call(base, "/")[1]
    finally:
        try:
            next(gen)
        except StopIteration:
            pass
