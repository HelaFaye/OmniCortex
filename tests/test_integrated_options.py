"""Integrated options (docs/INTEGRATED_OPTIONS.md), batch 1: Neural Reset, the prompt sanitizer, Fix This Mistake,
the core shunt, dependency GC and snapshots, and their settings."""
import json
import os
import sys
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import core_shunt  # noqa: E402
import dep_gc  # noqa: E402
import dep_snapshot  # noqa: E402
import harvest  # noqa: E402
import integrated  # noqa: E402
import loopguard  # noqa: E402
import sanitizer  # noqa: E402


def test_loop_detection_and_resume():
    t = "Intro. " + "I will try again to explain it. " * 4
    hit = loopguard.detect(t)
    assert hit and hit["repeats"] >= 3
    kept = loopguard.keep_upto(t, hit)
    assert kept.count("explain it.") == 1 and kept.startswith("Intro.")
    assert loopguard.detect("Ordinary prose that moves forward, sentence after sentence, without looping.") is None
    assert loopguard.detect("a b " * 3) is None                       # too short to count as a loop
    lines = "intro\n" + "for i in range(3):\n" * 6 + "x"
    assert loopguard.detect(lines)["lines"]
    p = loopguard.resume_payload({"messages": [{"role": "user", "content": "q"}], "temperature": 0.7}, kept, hit)
    assert p["messages"][-1] == {"role": "assistant", "content": kept} and p["dry_multiplier"] == 0.8
    assert p["logit_bias"][0][1] < 0 and p["repeat_penalty"] >= 1.15
    g = loopguard.Guard(max_resets=1, every=1)
    assert g.check(t) and g.check(t) is None                          # at most max_resets


def test_sanitizer_round_trip():
    s = sanitizer.Session({"terms": ["Acme Corp", "skunkworks"], "ips": True})
    raw = ("Use sk-proj-abcdefghijklmnopqrstuvwx for Acme Corp (acme corp again) at 10.0.0.5; "
           "mail ops@acme.example; api_key=\"hunter2hunter2\"; Bearer abcdefghijklmnopqrstuvwxyz0123; "
           "token ghp_" + "a" * 36 + " in skunkworks.")
    clean = s.clean(raw)
    for secret in ("sk-proj-", "Acme", "10.0.0.5", "ops@", "hunter2", "abcdefghijklmnopqrstuvwxyz0123", "ghp_", "skunkworks"):
        assert secret.lower() not in clean.lower(), secret
    acme = s.rev["acme corp"]
    assert clean.count(acme) == 2 and "api_key=\"[SECRET_" in clean and "Bearer [SECRET_" in clean
    assert s.restore(clean) .replace("acme corp", "Acme Corp") == raw.replace("acme corp", "Acme Corp")
    r = sanitizer.StreamRestorer(s)
    n = acme[1:-1]
    out = "".join(r.feed(x) for x in ["See [" + n[:3], n[3:] + "] and [SEC", "RET_1]."]) + r.flush()
    assert out == "See Acme Corp and sk-proj-abcdefghijklmnopqrstuvwx."
    msgs = s.clean_messages([{"role": "user", "content": [{"type": "text", "text": "Acme Corp"}, {"type": "image_url"}]}])
    assert msgs[0]["content"][0]["text"] == acme


def test_harvest_pairs_and_export(tmp_path):
    ctx = [{"role": "user", "content": "What is 2+2?"}]
    with pytest.raises(ValueError):
        harvest.add("m", ctx, "5", "", home=tmp_path)
    with pytest.raises(ValueError):
        harvest.add("m", ctx + [{"role": "assistant", "content": "5"}], "5", "4", home=tmp_path)
    it = harvest.add("m", ctx, "5", "4", note="arithmetic", home=tmp_path)
    assert harvest.add("m", ctx, "5", "Four.", home=tmp_path)["id"] == it["id"]      # same reply: one pair, updated
    ex = harvest.export(tmp_path / "data", home=tmp_path)
    dpo = [json.loads(l) for l in (tmp_path / "data" / "dpo.jsonl").read_text().splitlines()]
    assert ex["pairs"] == 1 and dpo[0]["chosen"][0]["content"] == "Four." and dpo[0]["rejected"][0]["content"] == "5"
    harvest.mark_used(ex["ids"], "run1", home=tmp_path)
    assert harvest.unused(home=tmp_path) == [] and harvest.train("x", tmp_path / "r", min_pairs=1, home=tmp_path) == 0


def test_core_shunt(tmp_path):
    assert core_shunt.parse_cpus("0-2,5") == [0, 1, 2, 5]
    import subprocess
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        done = core_shunt.apply(p.pid, {"priority": "low", "cores": "0"})
        if sys.platform.startswith("linux"):
            assert os.getpriority(os.PRIO_PROCESS, p.pid) >= 10 and os.sched_getaffinity(p.pid) == {0}
            assert "nice 10" in done
    finally:
        p.kill()
    assert core_shunt.describe({"cores": "all"})["will_pin"] == []


def test_dep_gc_and_snapshot(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / "half.gguf.part").write_bytes(b"x" * 10)
    (models / "bad.gguf").write_bytes(b"NOPE" + b"0" * 10)
    (models / "ok.gguf").write_bytes(b"GGUF" + b"0" * 10)
    hub = tmp_path / "hub" / "models--org--m"
    (hub / "blobs").mkdir(parents=True)
    (hub / "snapshots" / "abc").mkdir(parents=True)
    (hub / "blobs" / "used").write_bytes(b"1")
    (hub / "blobs" / "orphan").write_bytes(b"22")
    os.symlink(hub / "blobs" / "used", hub / "snapshots" / "abc" / "model.safetensors")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    found = dep_gc.find([models], ("partial", "hf-orphans"))
    kinds = {(x["kind"], Path(x["path"]).name) for x in found}
    assert {("partial", "half.gguf.part"), ("hf-orphans", "orphan"), ("bad-gguf", "bad.gguf")} <= kinds
    assert not any(n in ("ok.gguf", "used") for _, n in kinds)
    dep_gc.remove(found)
    assert not (models / "half.gguf.part").exists() and not (hub / "blobs" / "orphan").exists()
    assert (models / "bad.gguf").exists() and (hub / "blobs" / "used").exists()        # report-only kept
    a = dep_snapshot.snapshot()
    assert a["packages"] and a["python"]
    b = json.loads(json.dumps(a))
    b["packages"].pop("numpy", None)
    b["packages"]["pytest"] = "0.0.1"
    d = dep_snapshot.diff(a, b)
    assert "numpy" in d["missing"] and d["different"]["pytest"]["there"] == "0.0.1" and not d["same"]
    plan = dep_snapshot.plan(d)
    assert any(x.startswith("numpy==") for x in plan) and any(x.startswith("pytest==") for x in plan)
    assert dep_snapshot.diff(a, a)["same"]


def test_option_validation():
    cur = integrated.merged({})
    assert cur["neural_reset"]["enabled"] and not cur["sanitizer"]["enabled"]
    v = integrated.validate("sanitizer", {"terms": "Acme, Beta\nGamma", "enabled": 1}, cur)
    assert v["terms"] == ["Acme", "Beta", "Gamma"] and v["enabled"] is True
    for name, vals in (("neural_reset", {"max_resets": 99}), ("core_shunt", {"priority": "max"}),
                       ("core_shunt", {"cores": "lots"}), ("nope", {}), ("harvest", {"bogus": 1})):
        with pytest.raises(ValueError):
            integrated.validate(name, vals, cur)
    assert integrated.describe({})["sanitizer"]["sensitive"]["enabled"]


def test_studio_neural_reset_harvest_and_settings(tmp_path):
    import test_studio
    studio = test_studio.studio
    gen = test_studio.studio_srv.__wrapped__(tmp_path)
    base = next(gen)
    try:
        req = urllib.request.Request(base + "/api/chat", method="POST", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"model": "coder-gguf/coder-7b-q4_k_m",
                                                      "messages": [{"role": "user", "content": "loop please"}]}).encode())
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode()
        ev = [json.loads(l[5:]) for l in body.splitlines() if l.startswith("data:") and "[DONE]" not in l]
        reset = next(e["neural_reset"] for e in ev if "neural_reset" in e)
        assert reset["repeats"] >= 3 and reset["drop"] > 0
        assert "Done. bias=" in body and "dry=0.8" in body                        # resumed with the penalties
        chat_text = ""
        for e in ev:
            if "neural_reset" in e:
                chat_text = chat_text[:len(chat_text) - e["neural_reset"]["drop"]]
            for ch in e.get("choices") or []:
                chat_text += (ch.get("delta") or {}).get("content") or ""
        assert chat_text.split(" Done.")[0].count("again and again.") == 1 and chat_text.startswith("Start.")
        code, o = test_studio.call(base, "/api/settings/options", {"name": "neural_reset", "values": {"enabled": False}})
        assert code == 200 and o["neural_reset"]["values"]["enabled"] is False
        with urllib.request.urlopen(urllib.request.Request(req.full_url, data=req.data, method="POST",
                                                           headers={"Content-Type": "application/json"}), timeout=60) as r:
            assert "neural_reset" not in r.read().decode()
        code, h = test_studio.call(base, "/api/harvest", {"model": "coder-gguf/coder-7b-q4_k_m",
                                                         "messages": [{"role": "user", "content": "2+2?"}],
                                                         "reply": "5", "correction": "4"})
        assert code == 200 and h["pending"] == 1
        code, lst = test_studio.call(base, "/api/harvest")
        assert lst["unused"] == 1 and lst["pairs"][0]["question"] == "2+2?"
        assert test_studio.call(base, "/api/harvest", {"messages": [], "reply": "a", "correction": "b"})[0] == 400
        code, g = test_studio.call(base, "/api/settings/options")
        assert code == 200 and "core_shunt" in g and "decay" in g and g["options"]["harvest"]["values"]["enabled"]
        assert g["options"]["decay"]["values"]["enabled"] is False and studio.decay_step() is None
        assert test_studio.call(base, "/api/settings/options", {"name": "core_shunt", "values": {"cores": "x"}})[0] == 400
        code, snap = test_studio.call(base, "/api/env/snapshot")
        assert code == 200 and snap["schema"] == "omnicortex-env/1"
        page = test_studio.call(base, "/settings")[1]
        assert 'id="options"' in page and "opLoad" in page
        assert "fixMistake" in test_studio.call(base, "/")[1]
    finally:
        try:
            next(gen)
        except StopIteration:
            pass
