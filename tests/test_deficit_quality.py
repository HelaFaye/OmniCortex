"""Retraining quality: knowledge probing, honest synthetic DPO pairs, the holdout
guard, focus layers from the neuron review, layer-limited training, and the
keep/reject gate."""
import argparse
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import deficits  # noqa: E402
import finetune  # noqa: E402
import neuron_review as nr  # noqa: E402
import retrain_gate as rg  # noqa: E402
import testqa as tq  # noqa: E402

GRADE = argparse.Namespace(allow_exec=False, exec_timeout=5)


class Fake(BaseHTTPRequestHandler):
    """Teacher and student in one server, told apart by the model name."""
    weak: set = set()          # student: right only when sampling (temperature > 0)
    gold: dict = {}
    leak_prompt = ""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        content = body["messages"][0]["content"]
        prompt = content if isinstance(content, str) else content[0]["text"]
        if body.get("model") == "teacher":
            if "Write 3 NEW items" in prompt:
                reply = json.dumps([{"prompt": "What is 6 times 7?", "answer": "42", "answer_type": "number"},
                                    {"prompt": "What is 9 plus 10?", "answer": "19", "answer_type": "number"},
                                    {"prompt": self.leak_prompt, "answer": "1", "answer_type": "number"}])
            elif prompt.startswith("What is 6 times 7?"):
                reply = "Answer: 42"
            elif prompt.startswith("What is 9 plus 10?"):
                reply = "Answer: 19"
            else:
                reply = self.gold.get(prompt, "Answer: 1")
        else:   # the student
            if prompt.startswith("What is 6 times 7?"):
                reply = "Answer: 41"
            elif prompt.startswith("What is 9 plus 10?"):
                reply = "Answer: 19"
            elif prompt in self.weak and body.get("temperature", 0) > 0:
                reply = self.gold[prompt]
            else:
                reply = "Zanzibar Quuxley, in 1742."
        out = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("dq")
    tasks = [t for t in tq.load_tasks([str(ROOT / "qa" / "bank")], None, None, 0) if not t.get("image")]
    train = [t for t in tasks if not deficits.in_holdout(t["id"], 0.5)]
    hold = [t for t in tasks if deficits.in_holdout(t["id"], 0.5)]
    facts = [t for t in train if t["kind"] == "qa" and not t.get("expect_abstain")][:4]
    reason = next(t for t in train if t["kind"] == "reasoning")
    Fake.gold = {tq.prompt_for(t): f"It is {t['aliases'][0]}." for t in facts}
    Fake.gold[tq.prompt_for(reason)] = f"Answer: {reason['answer']}"
    Fake.weak = {tq.prompt_for(t) for t in facts[:2]}
    # a variation that restates a holdout question word for word
    Fake.leak_prompt = next(t for t in hold if t["kind"] == "reasoning" and len(t["prompt"].split()) > 10)["prompt"]
    rows, cache = [], []
    verdict_of = {facts[0]["id"]: "wrong", facts[1]["id"]: "wrong", facts[2]["id"]: "wrong",
                  facts[3]["id"]: "abstained", reason["id"]: "wrong"}
    for t in train + hold:
        v = verdict_of.get(t["id"])
        if v is None:
            continue
        text = "I don't know." if v == "abstained" else ("Answer: 999" if t is reason else "Zanzibar Quuxley.")
        rows.append({"id": t["id"], "kind": t["kind"], "subject": t["subject"], "verdict": v, "text": text})
        cache.append({"id": t["id"], "text": text})
    (d / "res.json").write_text(json.dumps({"reference": "m", "raw": {"m": rows}}))
    (d / "cache").mkdir()
    (d / "cache" / "m.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cache))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}/v1"
    yield {"dir": d, "teacher": base + "@teacher", "student": base + "@student", "facts": facts, "reason": reason}
    srv.shutdown()


def build(w, out, **kw):
    ns = argparse.Namespace(
        results=str(w["dir"] / "res.json"), label="m", cache=str(w["dir"] / "cache"),
        tasks=[str(ROOT / "qa" / "bank")], teacher=w["teacher"], teacher_tries=1, teacher_temperature=0.0,
        teacher_max_tokens=256, api_key="", expand=3, allow_exec=False, deficit_fraction=0.25, general=None,
        holdout_frac=0.5, seed=0, out=str(out), student=None, probe_samples=3, probe_temperature=0.8,
        unknown="abstain", review_model=None, review_dir=None, focus_top=8)
    for k, v in kw.items():
        setattr(ns, k, v)
    plan = deficits.build(ns)
    read = lambda f: [json.loads(x) for x in (out / f).read_text().splitlines()]  # noqa: E731
    return plan, read("sft.jsonl"), read("dpo.jsonl"), (out / "report.txt").read_text()


def test_unknown_facts_become_declines_and_known_ones_are_taught(world, tmp_path):
    plan, sft, dpo, report = build(world, tmp_path / "o", student=world["student"])
    f = world["facts"]
    target = {r["task"]: r for r in sft if r["source"] == "correction"}
    for t in f[:2]:                                       # weakly known: the gold answer
        assert target[t["id"]]["knowledge"] == "weakly_known"
        assert tq.grade(t, target[t["id"]]["messages"][-1]["content"], GRADE)[0] == "correct"
    assert target[f[2]["id"]]["messages"][-1]["content"] == deficits.ABSTAIN_TARGET   # unknown, answered wrong
    assert target[f[2]["id"]]["knowledge"] == "unknown"
    assert f[3]["id"] not in target                       # unknown and already declined: nothing to fix
    assert plan["knowledge"] == {"weakly_known": 2, "unknown": 2}
    pair = next(p for p in dpo if p["task"] == f[2]["id"])
    assert pair["chosen"][0]["content"] == deficits.ABSTAIN_TARGET and "Zanzibar" in pair["rejected"][0]["content"]
    assert "2 weakly known" in report and "trained to decline" in report


def test_synthetic_pairs_use_the_students_own_reply(world, tmp_path):
    plan, sft, dpo, _ = build(world, tmp_path / "o", student=world["student"])
    syn = [p for p in dpo if p["source"] == "synthetic"]
    # 6x7: the student fails it, so its own reply is the rejected side
    assert [p["rejected"][0]["content"] for p in syn] == ["Answer: 41"]
    # 9+10: the student already answers it, so SFT only
    assert any(r["messages"][0]["content"].startswith("What is 9 plus 10?") for r in sft)
    assert plan["skipped"]["variation the model already answers (SFT only)"] == 1
    # the variation that restates a holdout question never reaches training
    assert not any(r["messages"][0]["content"].startswith(Fake.leak_prompt) for r in sft)
    assert plan["skipped"]["synthetic item too close to a holdout question"] == 1


def test_without_a_student_no_mismatched_pairs(world, tmp_path):
    plan, sft, dpo, report = build(world, tmp_path / "o")
    assert not [p for p in dpo if p["source"] == "synthetic"]
    assert any(r["source"] == "synthetic" for r in sft)
    assert plan["knowledge"] == {} and "Pass --student" in report


def test_holdout_guard():
    g = deficits.HoldoutGuard(["What is the capital city of the country that borders Spain to the north?"])
    assert g.leaks("what is the capital city of the country that borders spain to the north")
    assert g.leaks("Name the capital city of the country that borders Spain to the north.")
    assert not g.leaks("What is the capital of Peru?")


def test_focus_layers_from_the_review(tmp_path):
    s = nr.ReviewStore(tmp_path)
    rng = np.random.default_rng(0)
    for k in range(30):
        wrong = k % 2 == 0
        v = rng.normal(0, 0.05, (6, 32)).astype(np.float32)
        if wrong:
            v[4, :8] += 1.0                               # wrong answers light up layer 4
        s.record("m", v, source="testqa", kind="test", subjects=["math"],
                 verdict="wrong" if wrong else "correct")
    fl = nr.deficit_layers(s, "m", ["math"], top=2)
    assert fl["layers"][0] == 4 or 4 in fl["layers"]
    assert fl["evidence"][0]["layer"] == 4 and fl["n_layers"] == 6
    assert nr.deficit_layers(s, "m", ["history"])["layers"] == []      # no observations for that subject


def test_parse_layers(tmp_path):
    assert finetune.parse_layers("all") is None
    assert finetune.parse_layers("3-5,9") == [3, 4, 5, 9]
    (tmp_path / "plan.json").write_text(json.dumps({"focus_layers": {"layers": [7, 2]}}))
    assert finetune.parse_layers("plan", tmp_path) == [2, 7]
    (tmp_path / "plan.json").write_text(json.dumps({"focus_layers": {"layers": [], "note": "too few"}}))
    with pytest.raises(SystemExit, match="too few"):
        finetune.parse_layers("plan", tmp_path)


def test_lora_only_on_focus_layers(tmp_path):
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    pytest.importorskip("sentencepiece")
    import tiny_models
    hf = tiny_models.make_tiny_llama(tmp_path / "tiny", layers=4)
    data = tmp_path / "data"
    data.mkdir()
    (data / "sft.jsonl").write_text("".join(json.dumps({"messages": [{"role": "user", "content": f"q{i}"},
                                                                     {"role": "assistant", "content": f"a{i}"}]})
                                            + "\n" for i in range(8)))
    (data / "plan.json").write_text(json.dumps({"focus_layers": {"layers": [1, 2]}}))
    out = tmp_path / "adapter"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "finetune.py"), "--model", str(hf), "--data",
                        str(data), "--out", str(out), "--method", "lora", "--lora-r", "4", "--max-steps", "1",
                        "--batch-size", "2", "--grad-accum", "1", "--max-length", "64", "--layers", "plan"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    cfg = json.loads((out / "adapter_config.json").read_text())
    assert cfg["layers_to_transform"] == [1, 2]
    from safetensors import safe_open
    with safe_open(str(out / "adapter_model.safetensors"), "pt") as f:
        layers = {int(k.split(".layers.")[1].split(".")[0]) for k in f.keys() if ".layers." in k}
    assert layers == {1, 2}
    assert json.loads((out / "omnicortex-finetune.json").read_text())["layers"] == [1, 2]
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "finetune.py"), "--model", str(hf), "--data",
                        str(data), "--out", str(out), "--method", "lora", "--layers", "9"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "layers 0-3" in (r.stderr + r.stdout)


# ------------------------------------------------------------------ the gate

def results(path, before, after, kinds=None):
    raw = {}
    for label, verdicts in (("before", before), ("after", after)):
        raw[label] = [{"id": i, "kind": (kinds or {}).get(i, "qa"), "subject": "math" if i.startswith("m") else "code",
                       "verdict": v} for i, v in verdicts.items()]
    path.write_text(json.dumps({"reference": "before", "raw": raw}))
    return path


def test_gate_keeps_a_real_improvement(tmp_path):
    ids = [f"m{i}" for i in range(40)] + [f"c{i}" for i in range(40)]
    before = {i: ("correct" if k % 2 else "wrong") for k, i in enumerate(ids)}
    after = dict(before)
    for i in ids[0:30:2]:                                  # 15 fixed
        after[i] = "correct"
    after[ids[1]] = "wrong"                                # 1 broken
    p = results(tmp_path / "r.json", before, after)
    (tmp_path / "h.json").write_text(json.dumps({"ids": ids}))
    g = rg.gate(rg.load(str(p), "before")[1], rg.load(str(p), "after")[1], set(ids))
    assert g["verdict"] == "keep" and g["holdout"]["fixed"] == 15 and g["holdout"]["broke"] == 1
    assert rg.main(["--results", str(p), "--before", "before", "--after", "after", "--holdout",
                    str(tmp_path / "h.json"), "--strict", "--out", str(tmp_path / "g.json")]) == 0


def test_gate_rejects_more_hallucination(tmp_path):
    ids = [f"m{i}" for i in range(60)]
    before = {i: ("correct" if k < 20 else "abstained") for k, i in enumerate(ids)}
    after = dict(before)
    for i in ids[20:40]:
        after[i] = "wrong"                               # declines turned into confident wrong answers
    g = rg.gate(before and {i: {"verdict": v, "kind": "qa", "subject": "math"} for i, v in before.items()},
                {i: {"verdict": v, "kind": "qa", "subject": "math"} for i, v in after.items()}, set(ids))
    assert g["verdict"] == "reject" and "hallucination rose" in g["reasons"][0]


def test_gate_inconclusive_and_memorisation(tmp_path):
    hold = [f"m{i}" for i in range(20)]
    trained = [f"t{i}" for i in range(20)]
    before = {i: "wrong" for i in hold + trained}
    after = dict(before)
    after.update({hold[0]: "correct", hold[1]: "correct", hold[2]: "correct", hold[3]: "abstained"})
    after.update({i: "correct" for i in trained[:15]})   # the trained items, learned by heart
    mk = lambda d: {i: {"verdict": v, "kind": "qa", "subject": "math"} for i, v in d.items()}  # noqa: E731
    g = rg.gate(mk(before), mk(after), set(hold))
    assert g["verdict"] == "inconclusive" and g.get("memorisation")
    assert "needs about" in g["reasons"][0]
    assert rg.binom_two_sided(0, 6) == pytest.approx(0.03125)
