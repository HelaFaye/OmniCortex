"""quant_fidelity.py's comparisons and verdicts, without llama.cpp (tests/test_llamacpp_integration.py runs the
whole thing on real quants of a tiny model)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import quant_fidelity as QF  # noqa: E402


def row(i, verdict, text="x", kind="qa"):
    return {"id": f"t{i}", "kind": kind, "subject": "geo", "verdict": verdict, "text": text}


def test_knowledge_splits_mistakes_from_abstentions():
    ref = [row(0, "correct"), row(1, "correct"), row(2, "correct"), row(3, "wrong"), row(4, "correct"),
           row(5, "correct", kind="canary"), row(6, "error")]
    quant = [row(0, "correct"), row(1, "wrong", "Sydney"), row(2, "abstained"), row(3, "correct"),
             row(4, "unparsable"), row(5, "refused", kind="canary"), row(6, "correct")]
    k = QF.knowledge(ref, quant)
    assert k["n"] == 5                                    # the canary and the error are left out
    assert (k["confident_mistakes"], k["now_abstains"], k["gained"]) == (2, 1, 1)
    assert k["lost_rate"] == 0.6 and k["mistake_examples"][0] == {"id": "t1", "subject": "geo", "now": "wrong",
                                                                    "answer": "Sydney"}


def test_hneuron_drift():
    rng = np.random.default_rng(0)
    ref = {"cett": [rng.normal(size=(3, 8)) for _ in range(5)], "prob": [0.9, 0.1, 0.6, 0.2, 0.4]}
    same = QF.hneurons(ref, {"cett": [c.copy() for c in ref["cett"]], "prob": list(ref["prob"])})
    assert same["cett_cos_mean"] == 1.0 and same["flag_flips"] == 0 and same["risk_delta_mean"] == 0
    noisy = [c.copy() for c in ref["cett"]]
    for c in noisy:
        c[2] += rng.normal(scale=3.0, size=8)              # the last layer drifts most
    d = QF.hneurons(ref, {"cett": noisy, "prob": [0.9, 0.7, 0.4, 0.2, 0.4]})
    assert d["cett_cos_mean"] < 1 and d["weakest_layer"] == 2 and d["cett_cos_per_layer"][0] == 1.0
    assert d["flag_flips"] == 2 and d["flagged_ref"] == 2 and d["flagged_quant"] == 2
    assert d["risk_delta_max"] == pytest.approx(0.6)
    assert "risk_delta_mean" not in QF.hneurons({"cett": ref["cett"], "prob": None}, {"cett": noisy, "prob": None})


def test_truth_transfer():
    class P:
        stages, best, auc = ["layer.0", "layer.1"], 1, [0.5, 0.9]
    y = np.array([1, 1, 0, 0])
    ref = {"p": np.array([[0.5, 0.9], [0.5, 0.8], [0.5, 0.2], [0.5, 0.1]]), "y": y}
    quant = {"p": np.array([[0.5, 0.6], [0.5, 0.3], [0.5, 0.4], [0.5, 0.1]]), "y": y}
    t = QF.truth_transfer(P, ref, quant)
    assert (t["auroc_ref"], t["auroc_quant"], t["auroc_drop"]) == (1.0, 0.75, 0.25) and t["stage"] == "layer.1"
    assert QF.auroc([0.1, 0.2], [1, 1]) is None           # one class only: no AUROC


def test_verdicts():
    t = dict(QF.DEFAULTS)
    assert QF.verdict({"next_token": {"kld": 0.02, "same_top": 95.0}}, t) == ("keeps", [])
    v, why = QF.verdict({"next_token": {"kld": 0.15, "same_top": 95.0}}, t)
    assert v == "check" and "KL divergence 0.1500" in why[0]
    v, why = QF.verdict({"next_token": {"kld": 0.01, "same_top": 80.0}}, t)
    assert v == "check" and "same top token 80.0%" in why[0]
    assert QF.verdict({"next_token": {"kld": 0.5}}, t)[0] == "loses"
    # lost answers are judged by the 95% lower bound: one flip in 30 is noise, 5 in 30 is not
    one = QF.knowledge([row(i, "correct") for i in range(30)], [row(0, "wrong")] + [row(i, "correct") for i in range(1, 30)])
    assert one["lost_rate"] == pytest.approx(1 / 30, abs=1e-4) and one["lost_ci"][0] < 0.03
    assert QF.verdict({"knowledge": one}, t) == ("keeps", [])
    five = QF.knowledge([row(i, "correct") for i in range(30)],
                        [row(i, "wrong") for i in range(5)] + [row(i, "correct") for i in range(5, 30)])
    v, why = QF.verdict({"knowledge": five}, t)
    assert v == "loses" and "5 answers to confident mistakes and 0 to abstentions of 30" in why[0]
    k = {"n": 1000, "confident_mistakes": 40, "now_abstains": 5, "lost_rate": 0.045, "lost_ci": [0.034, 0.06]}
    assert QF.verdict({"knowledge": k}, t)[0] == "check"
    assert QF.verdict({"knowledge": {"n": 9, "confident_mistakes": 1, "now_abstains": 0, "lost_rate": 0.1}},
                      t)[0] == "loses"                                          # no interval: the rate itself
    assert QF.verdict({"hneurons": {"cett_cos_mean": 0.9, "weakest_layer": 7, "texts": 10, "flag_flips": 0,
                                    "flag_flip_rate": 0.0}}, t)[0] == "check"
    assert QF.verdict({"truth": {"auroc_ref": 0.92, "auroc_quant": 0.78, "auroc_drop": 0.14}}, t)[0] == "loses"
    assert QF.verdict({"knowledge": {"error": "TestQA failed"}}, t) == ("check", ["knowledge: TestQA failed"])
    assert QF.verdict({"next_token": {"agreement": 0.5}}, t)[0] == "check"


def test_markdown_report():
    rep = {"reference": "/m/model-F16.gguf", "quants": {
        "model-Q4_K_M": {"verdict": "check", "reasons": ["lost 3 answers"],
                         "next_token": {"kld": 0.031, "same_top": 93.2},
                         "knowledge": {"n": 50, "confident_mistakes": 3, "now_abstains": 0,
                                       "mistake_examples": [{"id": "q7", "subject": "geo", "answer": "Sydney"}]},
                         "hneurons": {"cett_cos_mean": 0.987, "flag_flips": 1, "texts": 50},
                         "truth": {"auroc_ref": 0.93, "auroc_quant": 0.91}},
        "model-IQ2_XS": {"verdict": "loses", "reasons": ["x"], "next_token": {"agreement": 0.4}}}}
    md = QF.markdown(rep)
    assert "| model-Q4_K_M | **check** | 0.0310 / 93.2% | 3 + 0 of 50 | 0.987 | 1/50 | 0.930 → 0.910 |" in md
    assert "| model-IQ2_XS | **loses** | agreement 0.40 | - | - | - | - |" in md
    assert "`q7` (geo): 'Sydney'" in md


def test_cli_refuses_missing_files(tmp_path):
    with pytest.raises(SystemExit):
        QF.main(["--ref", str(tmp_path / "nope.gguf"), "--quant", str(tmp_path / "nope2.gguf")])
