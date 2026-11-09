"""The whole retraining cycle on a tiny Llama: baseline served by llama-server,
deficit data probed against it, two arms trained and converted, each measured
on the holdout and gated, resumable."""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import testqa as tq  # noqa: E402

LLAMA = Path(os.environ.get("OC_LLAMA") or "~/llama.cpp").expanduser()


class Teacher(BaseHTTPRequestHandler):
    gold: dict = {}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        reply = self.gold.get(body["messages"][0]["content"], "no idea")
        out = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_cycle_end_to_end(tmp_path):
    for m in ("peft", "trl", "sentencepiece"):
        pytest.importorskip(m)
    if not (LLAMA / "convert_hf_to_gguf.py").exists() or not (LLAMA / "build" / "bin" / "llama-server").exists():
        pytest.skip("needs a llama.cpp checkout with llama-server (OC_LLAMA)")
    import tiny_models
    hf = tiny_models.make_tiny_llama(tmp_path / "tiny", layers=4)
    base_gguf = tmp_path / "base.gguf"
    r = subprocess.run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(hf), "--outfile", str(base_gguf),
                        "--outtype", "f16"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    tasks = tq.load_tasks([str(ROOT / "qa" / "bank")], None, {"logic"}, 0)
    Teacher.gold = {tq.prompt_for(t): f"Answer: {t['answer']}" for t in tasks if t["kind"] == "reasoning"}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Teacher)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    out = tmp_path / "cycle"
    cmd = [sys.executable, str(ROOT / "scripts" / "retrain_cycle.py"), "--out", str(out), "--base-model", str(hf),
           "--base-gguf", str(base_gguf), "--teacher", f"http://127.0.0.1:{srv.server_port}/v1@teacher",
           "--arms", "all,1-2,plan", "--method", "lora", "--subject", "logic", "--per-subject", "30",
           "--quant", "F16", "--merge-dtype", "float32", "--ctx", "4096", "--max-tokens", "32", "--llama", str(LLAMA),
           "--server-arg=-n", "--server-arg=16", "--server-arg=--chat-template", "--server-arg=chatml",
           "--finetune-arg=--max-steps=2", "--finetune-arg=--lora-r=4", "--finetune-arg=--batch-size=2",
           "--finetune-arg=--grad-accum=1", "--finetune-arg=--max-length=512"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
        st = json.loads((out / "cycle.json").read_text())
        assert st["base_how"] == "llama-server"
        assert {k: v["status"] for k, v in st["steps"].items()} == {
            "baseline": "done", "data": "done", "train/all": "done", "convert/all": "done", "eval/all": "done",
            "train/1-2": "done", "convert/1-2": "done", "eval/1-2": "done"}
        arms = st["arms"]
        assert arms["plan"]["status"] == "skipped" and "focus layers" in arms["plan"]["why"]
        for name in ("all", "1-2"):
            assert arms[name]["verdict"] in ("keep", "reject", "inconclusive")
            assert Path(arms[name]["gguf"]).exists() and arms[name]["holdout"]["n"] > 0
        lay = json.loads((out / "arms" / "1-2" / "adapter" / "omnicortex-finetune.json").read_text())["layers"]
        assert lay == [1, 2]
        # every arm was asked exactly the baseline's holdout items
        ids = set(json.loads((out / "data" / "eval_ids.json").read_text())["ids"])
        for name in ("all", "1-2"):
            got = {x["id"] for x in json.loads(Path(arms[name]["eval"]).read_text())["raw"][name]
                   if x["kind"] != "canary"}
            assert got == ids
        assert len(st["head_to_head"]) == 1 and st["head_to_head"][0]["after"] == "1-2"
        report = (out / "report.txt").read_text()
        assert "arm all" in report and "RECOMMENDED" in report
        # resume: nothing runs again
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0 and "train/" not in r.stdout.replace("[cycle] train", "")
        assert r.stdout.count("serving:") == 0
    finally:
        srv.shutdown()
