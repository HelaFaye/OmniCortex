"""repo_adapt.py: scanning, grounded pairs, the judge, the 85/15 mix, provenance, staleness and size."""
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import repo_adapt as RA  # noqa: E402

PY = '''"""Rocket control."""


def launch(pad: str, countdown: int = 10) -> bool:
    """Start the countdown on a pad."""
    if countdown < 0:
        raise ValueError("countdown must be positive")
    return pad.startswith("LC")


def _private():
    return 1


class Booster:
    def __init__(self, stages):
        self.stages = stages

    def land(self, ship: str) -> str:
        return f"landed on {ship}"
'''
JS = "export function orbit(alt, inc) {\n  return alt * inc;\n}\nfunction hidden() {}\n"
GO = "package fly\n\nfunc Burn(sec int) int {\n\treturn sec\n}\n\nfunc quiet() {}\n\ntype Engine struct {\n\tThrust int\n}\n"
H = "#pragma once\nint ignite(int stage, float fuel);\n// not(this);\n"


def git(repo, *a):
    subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin"})


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "rocket"
    (r / "rocket").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / "rocket" / "control.py").write_text(PY)
    (r / "rocket" / "orbit.js").write_text(JS)
    (r / "rocket" / "fly.go").write_text(GO)
    (r / "rocket" / "ignite.h").write_text(H)
    (r / "tests" / "test_control.py").write_text("def test_launch():\n    pass\n")
    (r / "README.md").write_text("# Rocket\n\nCall `launch` with a pad name such as LC39A.\n")
    (r / "pyproject.toml").write_text("[project]\nname = 'rocket'\n")
    git(r, "init", "-q")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "first")
    return r


def test_scan_finds_public_interfaces(repo):
    sc = RA.scan(repo)
    names = {s["name"] for s in sc["symbols"]}
    assert {"launch", "Booster", "Booster.land", "Booster.__init__", "orbit", "Burn", "Engine", "ignite"} <= names
    assert not names & {"_private", "hidden", "quiet", "test_launch", "not"}
    kinds = {f["path"]: f["kind"] for f in sc["files"]}
    assert kinds["tests/test_control.py"] == "test" and kinds["pyproject.toml"] == "build"
    assert kinds["README.md"] == "doc" and kinds["rocket/control.py"] == "source"
    assert "rocket/control.py::launch(pad: str, countdown: int=10) -> bool" in sc["interfaces"]
    assert len(sc["commit"]) == 40 and sc["dirty"] is False
    us = RA.units(sc, repo)
    launch = next(u for u in us if u["id"] == "rocket/control.py::launch")
    assert launch["text"].startswith("def launch") and "raise ValueError" in launch["text"]
    assert any(u["kind"] == "doc" for u in us) and not any(u["id"].endswith("__init__") for u in us)


def test_fingerprint_follows_interfaces_not_bodies(repo, tmp_path):
    m = {"core": [], "interfaces": RA.scan(repo)["interfaces"], "fingerprint": RA.scan(repo)["fingerprint"],
         "commit": "x"}
    body = (repo / "rocket" / "control.py").read_text().replace('return pad.startswith("LC")',
                                                                 'return pad.upper().startswith("LC")')
    (repo / "rocket" / "control.py").write_text(body)
    git(repo, "commit", "-qam", "body only")
    assert RA.check(m, repo)["fresh"]                         # a new commit, the same interfaces: still fresh
    (repo / "rocket" / "control.py").write_text(body.replace("countdown: int = 10", "countdown: int = 10, "
                                                                                     "abort: bool = False"))
    (repo / "rocket" / "orbit.js").write_text(JS + "export const apogee = 1;\n")
    rep = RA.check(m, repo)
    assert not rep["fresh"] and rep["changed"] == ["rocket/control.py::launch"]
    assert rep["added"] == ["rocket/orbit.js::apogee"] and rep["removed"] == []
    # core paths: a change outside them doesn't count
    core = RA.scan(repo, ["rocket/*.go"])
    assert all(k.startswith("rocket/fly.go") for k in core["interfaces"])
    adapter = tmp_path / "a.gguf"
    adapter.write_bytes(b"\0" * 3 * 2 ** 20)
    assert RA.check(m, repo, str(adapter), 2)["adapter_ok"] is False
    assert RA.check(m, repo, str(adapter), 4)["adapter_ok"] is True


def test_judge_keeps_only_grounded_pairs(repo):
    sc = RA.scan(repo)
    us = RA.units(sc, repo)
    u = next(x for x in us if x["id"] == "rocket/control.py::launch")
    base = {"unit": u["id"], "path": u["path"], "unit_lines": [u["start"], u["end"]]}
    pairs = [
        {**base, "id": "good", "question": "What does `launch` raise?", "answer": "`ValueError` if countdown < 0.",
         "lines": [u["start"] + 2, u["start"] + 3]},
        {**base, "id": "madeup", "question": "How do I abort?", "answer": "Call `launch.abort_sequence()`.",
         "lines": [u["start"], u["start"]]},
        {**base, "id": "badlines", "question": "Default countdown?", "answer": "10.", "lines": [1, 400]},
        {**base, "id": "judged", "question": "What does `launch` return?", "answer": "Always true.",
         "lines": [u["start"], u["end"]]},
    ]
    calls = []

    def model(prompt):
        calls.append(prompt)
        return "UNSUPPORTED: it returns whether the pad starts with LC" if "Always true" in prompt else "SUPPORTED"
    out = {p["id"]: p for p in RA.judge(pairs, us, sc, repo, model)}
    assert out["good"]["verdict"] == "kept"
    assert "abort_sequence" in out["madeup"]["why"] and "not inside" in out["badlines"]["why"]
    assert out["judged"]["why"].startswith("judge: UNSUPPORTED")
    assert len(calls) == 2                                     # the cheap checks run first, without the model
    again = RA.judge(pairs, us, sc, repo, lambda p: 1 / 0, done={k: {"verdict": v["verdict"], "why": v["why"]}
                                                                 for k, v in out.items()})
    assert [p["verdict"] for p in again] == [out[p["id"]]["verdict"] for p in pairs]   # resumes, asks nothing


def test_parse_pairs():
    r = 'Sure!\n```json\n[{"question": "q1", "answer": "a1", "lines": [3, 5]}, {"question": "", "answer": "x"},' \
        ' {"question": "q2", "answer": "a2", "lines": "bad"}, 7]\n```'
    assert RA.parse_pairs(r) == [{"question": "q1", "answer": "a1", "lines": [3, 5]},
                                 {"question": "q2", "answer": "a2", "lines": None}]
    assert RA.parse_pairs("no json here") == [] and RA.parse_pairs("[not json") == []


class FakeTeacher(BaseHTTPRequestHandler):
    """An OpenAI-compatible teacher: grounded pairs from whatever lines it is shown; records the auth header."""
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeTeacher.seen.append(self.headers.get("Authorization"))
        prompt = body["messages"][0]["content"]
        if prompt.startswith("Source:"):
            text = "SUPPORTED"
        else:
            nums = [int(x) for x in __import__("re").findall(r"^\s*(\d+)  ", prompt, __import__("re").M)]
            name = __import__("re").search(r"`([\w.]+)`", prompt.split("lines")[0])
            ident = name.group(1) if name else "Rocket"
            text = json.dumps([{"question": f"What is `{ident}` for? ({i})", "answer": f"See `{ident}`.",
                                "lines": [nums[0], nums[-1]]} for i in range(2)])
        out = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_run_end_to_end_mix_holdout_manifest_and_no_key(repo, tmp_path, monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), FakeTeacher)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    secret = "sk-TESTSECRET-0123456789"
    monkeypatch.setenv("OC_TEACHER_KEY", secret)
    replay = tmp_path / "general.jsonl"
    replay.write_text("".join(json.dumps({"messages": [{"role": "user", "content": f"q{i}"},
                                                       {"role": "assistant", "content": f"a{i}"}]}) + "\n"
                              for i in range(50)))
    out = tmp_path / "adapt"
    argv = ["run", "--repo", str(repo), "--out", str(out), "--teacher",
            f"http://127.0.0.1:{srv.server_port}/v1@teach-1", "--replay", str(replay), "--per-unit", "2"]
    try:
        assert RA.main(argv) == 0
        n_calls = len(FakeTeacher.seen)
        assert RA.main(argv) == 0                             # resumes: nothing is asked again
        assert len(FakeTeacher.seen) == n_calls
    finally:
        srv.shutdown()
    assert set(FakeTeacher.seen) == {f"Bearer {secret}"}      # the key went to the teacher ...
    for f in out.iterdir():
        assert secret not in f.read_text(), f.name            # ... and nowhere else
    m = json.loads((out / "manifest.json").read_text())
    c = m["counts"]
    assert c["kept"] == c["pairs"] > 0 and c["train"] + c["holdout"] == c["kept"] and c["holdout"] > 0
    assert c["replay"] == c["replay_wanted"] == round(c["train"] * 0.15 / 0.85) and not m["warnings"]
    assert m["mix"]["replay_share"] == pytest.approx(0.15, abs=0.04)          # 2 of 16 rows here
    assert m["teacher"] == "teach-1 at 127.0.0.1" and len(m["commit"]) == 40
    sft = [json.loads(x) for x in (out / "sft.jsonl").read_text().splitlines()]
    assert sum(r["source"] == "repo" for r in sft) == c["train"] and sum(r["source"] == "replay" for r in sft) == c["replay"]
    assert sft[0]["messages"][0]["role"] == "user"
    held = {p["unit"] for p in m["pairs"] if p["split"] == "holdout"}
    trained = {p["unit"] for p in m["pairs"] if p["split"] == "train"}
    assert held and not held & trained                         # whole symbols are held out
    assert all(p["file_sha"] and p["lines"] for p in m["pairs"])
    assert RA.main(["check", "--manifest", str(out / "manifest.json"), "--repo", str(repo)]) == 0
    (repo / "rocket" / "fly.go").write_text(GO.replace("func Burn(sec int)", "func Burn(sec, n int)"))
    assert RA.main(["check", "--manifest", str(out / "manifest.json"), "--repo", str(repo)]) == 1


def test_short_replay_is_reported(repo, tmp_path):
    sc = RA.scan(repo)
    judged = [{"id": f"p{i}", "unit": f"u{i}", "path": "rocket/control.py", "lines": [1, 2], "question": "q",
               "answer": "a", "verdict": "kept", "why": ""} for i in range(20)]
    m = RA.build(sc, judged, tmp_path / "o", replay=[], holdout=0)
    assert m["counts"]["replay"] == 0 and "replay rows" in m["warnings"][0]


def test_cli_needs_a_teacher(repo, tmp_path):
    with pytest.raises(SystemExit):
        RA.main(["run", "--repo", str(repo), "--out", str(tmp_path / "o")])
    assert RA.main(["run", "--repo", str(repo), "--out", str(tmp_path / "o"), "--scan-only"]) == 0
    assert (tmp_path / "o" / "scan.json").exists()
