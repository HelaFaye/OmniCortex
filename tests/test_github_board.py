"""A scoreboard in a GitHub repository (github_board.py), speed predictions (perf_predict.py) and measurement
weights (measure_weights.py)."""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
pytest.importorskip("cryptography")
import github_board as gb  # noqa: E402
import measure_weights as mw  # noqa: E402
import perf_predict as pp  # noqa: E402
import score_cards as sc  # noqa: E402
from test_score_cards import honest_run  # noqa: E402


def test_seal_roundtrip_and_wrong_key():
    priv, pub = gb.keygen()
    blob = gb.seal(b"replies" * 1000, pub)
    assert gb.unseal(blob, priv) == b"replies" * 1000 and b"replies" not in blob.encode()
    other, _ = gb.keygen()
    with pytest.raises(Exception):
        gb.unseal(blob, other)


def test_issue_parts_split_and_parse(monkeypatch):
    priv, pub = gb.keygen()
    monkeypatch.setattr(gb, "PART", 500)
    card = {"card_version": 1, "model": {"file": "m.gguf", "sha256": "a" * 64}, "variant_id": "v", "visibility": "public"}
    log = {"items": [{"id": f"i{k}", "verdict": "correct", "reply": f"reply number {k} " * 20} for k in range(40)]}
    title, body, comments = gb.issue_parts(card, log, pub)
    assert "m.gguf" in title and comments and gb.MARK in body
    c2, sealed = gb.parse_issue(body, comments[::-1])                  # comments in any order
    assert c2 == card and json.loads(gb.unseal(sealed, priv)) == log
    with pytest.raises(ValueError, match="parts"):
        gb.parse_issue(body, comments[:-1])
    with pytest.raises(ValueError):
        gb.parse_issue("an ordinary issue")


def test_accept_verifies_commits_and_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "HOME", tmp_path / "home")
    info = gb.init(tmp_path / "board", "someone/omnicortex")
    board = Path(info["board"])
    assert (board / ".github" / "workflows" / "accept.yml").read_text().count("someone/omnicortex") == 1
    assert (board / ".gitignore").read_text().strip() == "board.key"
    priv = (board / "board.key").read_text().strip()
    pub = (board / "board.pub").read_text().strip()
    res, model = honest_run(tmp_path)
    card, log = sc.build(str(res), "m", str(model), {}, cache_dir=str(tmp_path / "cache"), public_by_hand=True)
    _, body, comments = gb.issue_parts(card, log, pub)
    out = gb.accept(board, 7, body, comments, priv)
    assert out["status"] == "verified", out
    saved = json.loads((board / out["path"]).read_text())
    assert "items" not in saved and saved["_issue"] == 7                  # the card, never the log
    bj = json.loads((board / "board.json").read_text())
    assert bj["cards"] == 1 and bj["rows"][0]["items"] == 12 and "reply" not in json.dumps(bj)
    assert gb.accept(board, 8, body, comments, priv)["note"] == "this exact result was already accepted"
    forged = {"items": [dict(i, verdict="correct") for i in log["items"]]}
    _, fb, fc = gb.issue_parts(card, forged, pub)
    assert gb.accept(board, 9, fb, fc, priv)["status"] == "rejected"
    pcard, plog = sc.build(str(res), "m", str(model), {"ctx": 2048}, cache_dir=str(tmp_path / "cache"))
    _, pbody, pc = gb.issue_parts(pcard, plog, pub)
    assert "public board" in gb.accept(board, 10, pbody, pc, priv)["note"]
    assert gb.accept(board, 11, body, comments, gb.keygen()[0])["note"].startswith("unreadable")
    with pytest.raises(ValueError, match="public"):
        gb.submit("o/r", pcard, plog, "t" * 30, pub)
    with pytest.raises(ValueError, match="token"):
        gb.submit("o/r", card, log, None, pub)


def test_card_carries_measured_speed(tmp_path):
    res, model = honest_run(tmp_path)
    d = json.loads(res.read_text())
    d["scheduling"] = [{"server": "http://x/v1", "limit": 2, "done": 12, "avg_s": 1.5, "gen_tok_s": 41.2},
                       {"server": "http://other/v1", "gen_tok_s": 5}]
    res.write_text(json.dumps(d))
    card, _ = sc.build(str(res), "m", str(model), {"ngl": 99, "ctx": 4096, "threads": 8})
    assert card["performance"] == [{"source": "testqa", "gen_tok_s": 41.2, "avg_s": 1.5, "parallel": 2, "requests": 12,
                                    "ngl": 99, "ctx": 4096}]


def synth(device, backend, bw_eff, gb_, quant="Q4_K_M", n=4, **kw):
    """Measurements of a device whose real efficiency differs from the default."""
    bw, _, _ = pp.spec(device, backend)
    tok = bw_eff * bw * 1e9 / (gb_ * 2**30)
    return [pp.Sample(device=device, backend=backend, model_bytes=int(gb_ * 2**30), tok_s=tok * (1 + 0.02 * (i - 1)),
                      quant=quant, **kw) for i in range(n)]


def test_prior_follows_bandwidth():
    big = pp.Model().predict(pp.Sample("RTX 4090", "cuda", 5 * 2**30, 0))
    small = pp.Model().predict(pp.Sample("RTX 4060", "cuda", 5 * 2**30, 0))
    assert big["tok_s"] / small["tok_s"] == pytest.approx(1008 / 272, rel=0.01)
    assert pp.spec("Some Future Card", "vulkan")[2] == "typical gpu"
    assert pp.spec("AMD Ryzen 9 7950X", "")[2] == "typical cpu"
    assert pp.spec("AMD Radeon Graphics", "vulkan")[2] == "typical igpu"


def test_fit_learns_backend_and_device_offsets():
    data = []
    for dev, be, eff in [("RTX 4090", "cuda", 0.75), ("RTX 3090", "cuda", 0.75), ("RTX 3060", "cuda", 0.72),
                         ("RX 7900 XTX", "vulkan", 0.45), ("RX 6800 XT", "vulkan", 0.45), ("RX 7600", "vulkan", 0.43)]:
        for gb_ in (2, 5, 9):
            data += synth(dev, be, eff, gb_)
    m = pp.fit(data)
    names = {e["feature"]: e["factor"] for e in pp.explain(m)}
    assert names["backend_cuda"] > names.get("backend_vulkan", 1.0)        # CUDA reaches more of its bandwidth here
    assert m.cv_error is not None and m.cv_error < 0.15                     # unseen cards predicted within ~15%
    p = m.predict(pp.Sample("RX 7800 XT", "vulkan", 5 * 2**30, 0))          # never measured
    want = 0.45 * 624e9 / (5 * 2**30)
    assert abs(p["tok_s"] / want - 1) < 0.2 and p["measured_on_device"] == 0
    seen = m.predict(pp.Sample("RTX 4090", "cuda", 5 * 2**30, 0))
    assert seen["high"] / seen["low"] < p["high"] / p["low"]                # measured devices get narrower ranges
    again = pp.Model.from_json(json.loads(json.dumps(m.to_json())))
    assert again.predict(pp.Sample("RX 7800 XT", "vulkan", 5 * 2**30, 0))["tok_s"] == p["tok_s"]


def test_samples_from_board_and_worklog():
    rows = [{"performance": [{"gen_tok_s": 30.0, "model_bytes": 4 * 2**30, "quant": "Q4_K_M", "ngl": 99, "requests": 10,
                              "hardware": {"cpu": "x", "gpus": [{"name": "RTX 3090", "backend": "cuda"}]}},
                             {"gen_tok_s": 3.0, "model_bytes": 4 * 2**30, "ngl": 0,
                              "hardware": {"cpu": "AMD Ryzen 7", "gpus": [{"name": "RTX 3090", "backend": "cuda"}]}}]}]
    s = pp.samples_from_board(rows)
    assert [(x.device, x.backend) for x in s] == [("RTX 3090", "cuda"), ("AMD Ryzen 7", "cpu")] and s[0].weight == 10
    log = [{"model": "m", "device": "Vulkan0", "completion_tokens": 100, "gen_ms": 2000, "ok": True},
           {"model": "m", "device": "Vulkan0", "completion_tokens": 100, "gen_ms": 3000, "ok": True},
           {"model": "m", "device": "Vulkan0", "completion_tokens": 100, "ok": True},           # no timings: left out
           {"model": "gone", "device": "Vulkan0", "completion_tokens": 5, "gen_ms": 100}]
    w = pp.samples_from_worklog(log, lambda m: {"bytes": 2**30, "quant": "Q8_0"} if m == "m" else None,
                                lambda h, d: {"name": "RX 7600", "backend": "vulkan"})
    assert len(w) == 1 and w[0].tok_s == 40.0 and w[0].weight == 2


def test_measure_weights_consensus_and_target():
    models = [f"m{i}" for i in range(8)]
    truth = {m: i / 7 for i, m in enumerate(models)}
    table = {m: {"testqa": truth[m] + 0.01, "livebench": truth[m] * 0.8, "noise": [0.3, 0.9, 0.1, 0.7, 0.5, 0.2, 0.8, 0.4][i]}
             for i, m in enumerate(models)}
    r = mw.fit(table, target=None)
    assert r["mode"] == "consensus" and r["weights"]["noise"] < r["weights"]["testqa"]
    assert math.isclose(sum(r["weights"].values()), 1, abs_tol=1e-3)
    assert r["ranking"][0]["model"] == "m7" and r["ranking"][-1]["model"] == "m0"
    for i, m in enumerate(models):
        table[m]["user_verdicts"] = [0.3, 0.9, 0.1, 0.7, 0.5, 0.2, 0.8, 0.4][i]       # people agree with "noise"
    r2 = mw.fit(table)
    assert r2["mode"] == "against user_verdicts" and r2["weights"]["noise"] > r2["weights"]["testqa"]
    few = mw.fit({"a": {"x": 1, "y": 2}, "b": {"x": 0, "y": 3}}, target=None)
    assert few["weights"] == {"x": 0.5, "y": 0.5}                                    # too few models: equal shares


def test_measure_weights_history(tmp_path):
    r = mw.fit({f"m{i}": {"a": i, "b": i * 2} for i in range(4)}, target=None)
    h = mw.remember(r, tmp_path / "h.json")
    h = mw.remember(r, tmp_path / "h.json")
    assert len(h) == 2 and h[-1]["weights"] == r["weights"]
