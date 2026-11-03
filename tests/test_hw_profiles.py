"""Hardware support profiles (scripts/hw_profiles.py), their board submissions, and Studio's API."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import hw_profiles as HP  # noqa: E402


def test_catalogue_is_valid_and_purposes_are_labelled():
    cat = HP.catalogue()
    assert len(cat) >= 9 and not [p for p in cat if p.get("invalid")]
    for p in cat:
        assert p["purposes"] and (len(p["purposes"]) > 1 or "general" in p["purposes"]), p["id"]
        for spec in p["purposes"].values():
            assert spec["summary"]
    eras = {p["era"] for p in cat}
    assert {"legacy", "modern"} <= eras                         # legacy hardware is first-class


def test_validation():
    good = {"schema": HP.SCHEMA, "id": "x-y", "title": "X", "purposes": {"general": {"summary": "s"}}}
    assert HP.validate(good)
    bad = [({**good, "id": "Bad Id"}, "id"), ({**good, "purposes": {"inference": {"summary": "s"}}}, "single purpose"),
           ({**good, "purposes": {"general": {"summary": ""}}}, "summary"),
           ({**good, "purposes": {"general": {"summary": "s", "llama": {"backend": "cuda", "cmake": "-DX=1; rm -rf /"}}}}, "cmake"),
           ({**good, "purposes": {"general": {"summary": "s", "torch": {"index": "https://evil.example/whl"}}}}, "index"),
           ({**good, "purposes": {"fast": {"summary": "s"}}}, "unknown")]
    for p, why in bad:
        with pytest.raises(HP.ProfileError, match=why):
            HP.validate(p)


def test_matching_legacy_and_modern():
    cat = HP.catalogue()
    devs = HP.facts(devices=[{"id": "cuda:0", "index": 0, "backend": "cuda", "vendor": "nvidia", "name": "Tesla M10"},
                             {"id": "cuda:1", "index": 1, "backend": "cuda", "vendor": "nvidia", "name": "RTX 4090"},
                             {"id": "rocm:0", "backend": "rocm", "vendor": "amd", "name": "MI50", "gfx": "gfx906"},
                             {"id": "rocm:1", "backend": "rocm", "vendor": "amd", "name": "RX 7900 XTX", "gfx": "gfx1100"},
                             {"id": "cpu:0", "backend": "cpu", "vendor": "cpu", "name": "CPU"}],
                    smi=[{"index": 0, "cc": 50}, {"index": 1, "cc": 89}])
    got = {m["profile"]["id"]: m["devices"] for m in HP.match(cat, devs, "x86_64")}
    assert "Tesla M10, compute 5.0" in got["nvidia-legacy-sm50-70"][0]
    assert "RTX 4090" in got["nvidia-modern-sm75"][0] and "gfx906" in got["amd-legacy-gcn"][0]
    assert "gfx1100" in got["amd-rocm"][0] and "cpu-x86" in got and "arm-board" not in got


def test_plans_and_reports(tmp_path, monkeypatch):
    p = HP.get("nvidia-legacy-sm50-70")
    inf = HP.plan(p, "inference", "/py", "/l")
    assert inf["steps"][0]["cmd"][-3:] == ["--cuda-arch", "50-real;52-real;60-real;61-real;70-real", "--portable"]
    tr = HP.plan(p, "retraining", "/py")
    assert tr["steps"][0]["cmd"][-2:] == ["--index-url", "https://download.pytorch.org/whl/cu126"]
    assert tr["training"]["dtype"] == "float32"
    g = HP.plan(HP.get("apple-silicon"), "inference")                   # one purpose: general covers it
    assert g["purpose"] == "general"
    with pytest.raises(HP.ProfileError, match="no 'general'"):
        HP.purpose_of(HP.get("amd-rocm"), "general")
    lines = []
    assert HP.apply(inf, dry_run=True, log=lines.append) == 0 and any(x.startswith("$ bash") for x in lines)
    monkeypatch.setattr(HP, "USER_DIR", tmp_path / "mine")
    mine = {**{k: v for k, v in p.items() if k not in ("file", "status")}, "id": "my-m10", "title": "Mine"}
    HP.save_user(mine)
    assert HP.get("my-m10")["status"] == "yours"
    r = HP.report("my-m10", "inference", True, [{"name": "Tesla M10", "backend": "cuda", "compute": 5.0}], tok_s=12)
    agg = HP.aggregate_reports([r, {**r, "worked": False, "gen_tok_s": 8}, r])
    a = agg["my-m10"]["inference"]
    assert (a["reports"], a["worked"], a["failed"], a["median_tok_s"]) == (3, 2, 1, 12)


def test_board_accepts_profiles_and_reports(tmp_path):
    pytest.importorskip("cryptography")
    import github_board as gb
    board = Path(gb.init(tmp_path / "board", "someone/omnicortex")["board"])
    p = HP.submission(HP.get("amd-legacy-gcn"))
    _, body = gb.plain_issue(gb.MARK_PROFILE, "t", {**p, "id": "vega-tuned"}, "x")
    out = gb.accept(board, 3, body, [], "")
    assert out["status"] == "verified" and (board / "profiles" / "pending" / "vega-tuned.json").exists()
    _, bad = gb.plain_issue(gb.MARK_PROFILE, "t", {**p, "purposes": {}}, "x")
    assert gb.accept(board, 4, bad, [], "")["status"] == "rejected"
    rep = HP.report("amd-legacy-gcn", "general", True, [{"name": "Radeon VII", "backend": "vulkan"}])
    _, rb = gb.plain_issue(gb.MARK_REPORT, "t", rep, "x")
    assert gb.accept(board, 5, rb, [], "")["status"] == "verified"
    bj = json.loads((board / "board.json").read_text())["hw_profiles"]
    assert bj["pending"][0]["id"] == "vega-tuned" and bj["reports"]["amd-legacy-gcn"]["general"]["worked"] == 1
    wf = (board / ".github" / "workflows" / "accept.yml").read_text()
    assert gb.MARK_PROFILE in wf and "profiles reports" in wf


def test_studio_api(tmp_path):
    import test_studio
    gen = test_studio.studio_srv.__wrapped__(tmp_path)
    base = next(gen)
    try:
        code, r = test_studio.call(base, "/api/hw-profiles")
        assert code == 200 and len(r["profiles"]) >= 9 and isinstance(r["matches"], list)
        code, pl = test_studio.call(base, "/api/hw-profiles/plan?id=amd-rocm&purpose=retraining")
        assert code == 200 and "rocm" in pl["steps"][-1]["cmd"][-1]
        assert test_studio.call(base, "/api/hw-profiles/plan?id=nope")[0] == 404
        code, e = test_studio.call(base, "/api/hw-profiles/export", {"id": "cpu-x86"})
        assert code == 200 and e["id"] == "cpu-x86" and "file" not in e
        code, s = test_studio.call(base, "/api/hw-profiles/submit", {"id": "cpu-x86"})
        assert code == 400 and "GitHub board" in s["error"]
        assert "hwprofiles" in test_studio.call(base, "/setup")[1]
    finally:
        try:
            next(gen)
        except StopIteration:
            pass
