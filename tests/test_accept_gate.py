"""accept_gate.py: one verdict from the retrain gate, quantization fidelity and the repository adapter check."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import accept_gate as AG  # noqa: E402
import quant_fidelity as QF  # noqa: E402


def fid(tmp_path, auc_drop=0.03, kld=0.02, **extra):
    rep = {"quants": {"m-Q4_K_M": {"verdict": "keeps", "reasons": [],
                                   "next_token": {"kld": kld, "same_top": 93.0, **extra},
                                   "truth": {"auroc_ref": 0.93, "auroc_quant": 0.93 - auc_drop,
                                             "auroc_drop": auc_drop}},
                      "m-Q2_K": {"verdict": "loses", "reasons": [], "next_token": {"kld": 1.2, "same_top": 47.0}}}}
    p = tmp_path / f"fidelity-{auc_drop}-{kld}-{len(extra)}.json"
    p.write_text(json.dumps(rep))
    return str(p)


def run(argv, tmp_path):
    rc = AG.main(argv + ["--out", str(tmp_path / "acc")])
    return rc, json.loads((tmp_path / "acc" / "accept.json").read_text())


def test_presets():
    assert QF.limits("strict")["max_auc_drop"] == 0.02 and QF.limits()["max_auc_drop"] == 0.05
    assert QF.limits("strict", max_auc_drop=0.03, max_kld=None)["max_auc_drop"] == 0.03
    with pytest.raises(ValueError):
        QF.limits("lenient")


def test_strict_preset_rejudges_a_quant(tmp_path):
    f = fid(tmp_path, auc_drop=0.03)
    rc, g = run(["--fidelity", f, "--quant", "m-Q4_K_M"], tmp_path)
    assert rc == 0 and g["verdict"] == "accept"                       # 0.03 is within the default 0.05
    rc, g = run(["--fidelity", f, "--quant", "m-Q4_K_M", "--preset", "strict"], tmp_path)
    assert rc == 2 and g["verdict"] == "review" and "AUROC" in g["reasons"][0]   # past 0.02, not past 0.04
    rc, g = run(["--fidelity", fid(tmp_path, auc_drop=0.05), "--quant", "m-Q4_K_M", "--preset", "strict"], tmp_path)
    assert rc == 1 and g["verdict"] == "reject"
    rc, g = run(["--fidelity", f, "--quant", "m-Q4_K_M", "--preset", "strict", "--max-auc-drop", "0.04"], tmp_path)
    assert g["verdict"] == "accept" and g["limits"]["max_auc_drop"] == 0.04
    assert run(["--fidelity", f], tmp_path)[1]["verdict"] == "reject"     # two quants: say which
    assert run(["--fidelity", f, "--quant", "m-Q2_K"], tmp_path)[1]["verdict"] == "reject"


def test_a_retried_kl_run_needs_review(tmp_path):
    f = fid(tmp_path, auc_drop=0.0, retried_after=["killed by SIGABRT: kl_divergence ..."])
    rc, g = run(["--fidelity", f, "--quant", "m-Q4_K_M"], tmp_path)
    assert g["verdict"] == "review" and "crashed once" in g["reasons"][0]


def test_retrain_and_required_checks(tmp_path):
    for v, want in (("keep", "accept"), ("inconclusive", "review"), ("reject", "reject")):
        (tmp_path / "gate.json").write_text(json.dumps({"verdict": v, "reasons": [f"why {v}"]}))
        assert run(["--retrain", str(tmp_path / "gate.json")], tmp_path)[1]["verdict"] == want
    (tmp_path / "gate.json").write_text(json.dumps({"verdict": "keep", "reasons": []}))
    rc, g = run(["--retrain", str(tmp_path / "gate.json"), "--require", "fidelity"], tmp_path)
    assert rc == 1 and "required but not given: fidelity" in g["reasons"][0]
    rc, g = run([], tmp_path)
    assert rc == 1 and "nothing to judge" in g["reasons"][0]
    md = (tmp_path / "acc" / "accept.md").read_text()
    assert "| fidelity | not given |" in md


def test_adapter_check(tmp_path):
    import repo_adapt as RA
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "lib.py").write_text("def area(w, h):\n    return w * h\n")
    sc = RA.scan(repo)
    m = {"core": [], "interfaces": sc["interfaces"], "fingerprint": sc["fingerprint"], "commit": None,
         "dirty": None, "warnings": []}
    (tmp_path / "m.json").write_text(json.dumps(m))
    adapter = tmp_path / "a.gguf"
    adapter.write_bytes(b"\0" * 2 ** 20)
    base = ["--adapt-manifest", str(tmp_path / "m.json"), "--repo", str(repo), "--adapter", str(adapter)]
    assert run(base + ["--max-adapter-mb", "2"], tmp_path)[1]["verdict"] == "accept"
    rc, g = run(base + ["--max-adapter-mb", "0.5"], tmp_path)
    assert rc == 1 and "over its 0.5 MB budget" in g["reasons"][0]
    (repo / "lib.py").write_text("def area(w, h, d=1):\n    return w * h * d\n")
    rc, g = run(base, tmp_path)
    assert rc == 1 and "changed: lib.py::area" in g["reasons"][0]
    (repo / "lib.py").write_text("def area(w, h):\n    return w * h\n")
    m["warnings"] = ["only 0 replay rows for the 3 that a 15% share needs"]
    (tmp_path / "m.json").write_text(json.dumps(m))
    assert run(base, tmp_path)[1]["verdict"] == "review"
