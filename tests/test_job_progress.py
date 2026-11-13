"""Jobs say how far they are and, when they fail, why and what fixes it."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "scripts"))
import jobs  # noqa: E402


def test_parse_progress_formats():
    tq = "Map:  45%|████▌     | 135/300 [00:02<00:03, 47.20 ex/s]\rMap:  60%|██████    | 180/300 [00:03<00:02, 47.6 ex/s]"
    p = jobs.parse_progress(tq)
    assert (p["done"], p["total"], p["pct"], p["label"], p["eta_s"]) == (180, 300, 60.0, "Map", 2.0)
    assert jobs.parse_progress("PROGRESS 12/50 predicting x\nsome line\n")["done"] == 12      # newest wins
    assert jobs.parse_progress("[3/24] gen 51.0 t/s")["total"] == 24
    assert jobs.parse_progress("  25/600\n")["done"] == 25
    assert jobs.parse_progress("ratio 3/2 is not progress") is None
    assert jobs.parse_progress("loading model\n") is None
    assert jobs.last_line("a\nPROGRESS 1/2 x\nthe step\nPROGRESS 2/2 y\n") == "the step"


def test_explain_failure():
    h = jobs.explain_failure("Traceback\nModuleNotFoundError: No module named 'sklearn.linear_model'\n")
    assert h["install"] == "scikit-learn"
    assert "memory" in jobs.explain_failure("torch.OutOfMemoryError: CUDA out of memory")["reason"]
    assert "port" in jobs.explain_failure("OSError: [Errno 98] Address already in use")["reason"]
    assert jobs.explain_failure("all good") is None


def test_runner_annotates(tmp_path):
    r = jobs.JobRunner(tmp_path / "jobs", 2)
    now = time.time()
    for jid, status, log in [("a" * 12, "running", "PROGRESS 40/100 item-40\nworking on item-40\n"),
                             ("b" * 12, "failed", "ModuleNotFoundError: No module named 'datasets'\n")]:
        d = tmp_path / "jobs" / jid
        d.mkdir(parents=True)
        (d / "log.txt").write_text(log)
        (d / "job.json").write_text(json.dumps({"id": jid, "kind": "x", "title": "x", "values": {}, "argv": [],
                                                "status": status, "started": now - 20, "ended": None}))
    r.procs["a" * 12] = type("P", (), {"poll": lambda self: None})()       # still running
    by = {m["id"]: m for m in r.list()}
    run, bad = by["a" * 12], by["b" * 12]
    assert run["progress"]["pct"] == 40.0 and run["step"] == "working on item-40"
    assert 25 <= run["progress"]["eta_s"] <= 35                            # 20 s for 40, so about 30 s for 60 more
    assert bad["hint"]["install"] == "datasets"
