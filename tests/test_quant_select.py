"""Choosing a quantization per computer: fit, speed and fidelity measured, the fastest faithful one kept."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import quant_select as qs  # noqa: E402

# llama-server stand-in: faster with fewer bits; answers drift from the reference as bits drop.
SERVER = r'''#!{py}
import argparse, json, re, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
p = argparse.ArgumentParser(); p.add_argument("-m"); p.add_argument("--port", type=int); p.add_argument("--host")
a, _ = p.parse_known_args()
BITS = {{"Q8_0": 8.5, "Q4_K_M": 4.75, "Q2_K": 2.6, "TQ1_0": 1.7}}
bits = next(v for k, v in BITS.items() if k in a.m)
WORDS = "one two three four five six seven eight nine ten".split()
class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def _send(self, o):
        b = json.dumps(o).encode(); self.send_response(200); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._send({{"status": "ok"}})
    def do_POST(self):
        json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        keep = 10 if bits >= 4 else 6 if bits >= 2.5 else 1
        text = " ".join(WORDS[:keep] + ["x"] * (10 - keep))
        self._send({{"content": text, "timings": {{"prompt_n": 512, "predicted_n": 128,
                    "prompt_per_second": 400 / bits, "predicted_per_second": 80 / bits}}}})
ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
'''
PERPLEXITY = r'''#!{py}
import sys
args = sys.argv[1:]
m = args[args.index("-m") + 1]
base = args[args.index("--kl-divergence-base") + 1]
if "--kl-divergence" not in args:
    open(base, "w").write("logits"); sys.exit(0)
k = {{"Q8_0": 0.002, "Q4_K_M": 0.03, "Q2_K": 0.25, "TQ1_0": 2.9}}
v = next(x for n, x in k.items() if n in m)
print(f"Mean    KLD:   {{v:.6f}} ±   0.0001")
print(f"Same top p: {{100 - v * 30:.3f}} ± 0.1 %")
'''


def tools(tmp_path, perplexity=False):
    d = tmp_path / "bin"
    d.mkdir()
    for name, body in [("llama-server", SERVER)] + ([("llama-perplexity", PERPLEXITY)] if perplexity else []):
        f = d / name
        f.write_text(body.format(py=sys.executable))
        os.chmod(f, 0o755)
    files = []
    for q, mb in [("Q8_0", 8), ("Q4_K_M", 4), ("Q2_K", 3), ("TQ1_0", 2)]:
        f = tmp_path / f"mymodel-{q}.gguf"
        f.write_bytes(b"\0" * (mb * 1024))
        files.append(str(f))
    return str(d / "llama-server"), files


def test_names():
    assert qs.quant_of("Llama-3-8B-Instruct-Q4_K_M.gguf") == "Q4_K_M" and qs.quant_of("x-TQ2_0.gguf") == "TQ2_0"
    assert qs.quant_of("m-IQ3_XS.gguf") == "IQ3_XS" and qs.quant_of("model.gguf") is None
    assert qs.quant_of("qwen2.5-0.5b-instruct-fp16.gguf") == "F16" and qs.family("qwen2.5-0.5b-instruct-fp16.gguf") == \
        qs.family("qwen2.5-0.5b-instruct-q4_k_m.gguf") == "qwen2.5-0.5b-instruct"
    assert qs.family("Llama-3-8B-Instruct-Q4_K_M.gguf") == "Llama-3-8B-Instruct" == qs.family("Llama-3-8B-Instruct.Q8_0.gguf")


def test_picks_fastest_faithful_by_agreement(tmp_path):
    server, files = tools(tmp_path)
    choices = tmp_path / "choices.json"
    rc = qs.main(["--model", *files, "--server", server, "--out", str(tmp_path / "run"), "--fidelity", "agreement",
                  "--memory-gb", "1", "--repeats", "1", "--choices", str(choices), "--min-agreement", "0.9"])
    assert rc == 0
    rep = json.loads((tmp_path / "run" / "quant_select.json").read_text())
    rows = {r["quant"]: r for r in rep["rows"]}
    assert rows["Q8_0"]["reference"] and rows["Q8_0"]["fidelity"]["agreement"] == 1.0
    assert rows["TQ1_0"]["fidelity"]["agreement"] < 0.2 and rows["TQ1_0"]["ternary"]
    assert rep["chosen"]["quant"] == "Q4_K_M"                       # faster than Q8_0, still faithful
    saved = json.loads(choices.read_text())["mymodel"]
    assert list(saved.values())[0]["quant"] == "Q4_K_M"


def test_kld_and_memory(tmp_path):
    server, files = tools(tmp_path, perplexity=True)
    rc = qs.main(["--model", *files, "--server", server, "--out", str(tmp_path / "run"), "--repeats", "1",
                  "--memory-gb", str(3.5 / 1024 / 1024), "--choices", str(tmp_path / "c.json")])
    rep = json.loads((tmp_path / "run" / "quant_select.json").read_text())
    rows = {r["quant"]: r for r in rep["rows"]}
    assert rows["Q4_K_M"]["fidelity"]["method"] == "kld" and rows["Q4_K_M"]["fidelity"]["kld"] == 0.03
    assert not rows["Q8_0"]["fits"] and not rows["Q4_K_M"]["fits"] and rows["Q2_K"]["fits"]
    # nothing that fits is faithful enough: the most faithful that fits, and it says so
    assert rc == 0 and rep["chosen"]["quant"] == "Q2_K" and "none clears" in rep["chosen"]["why"]


def test_studio_shows_the_choice(tmp_path):
    sys.path.insert(0, str(ROOT / "tests"))
    sys.path.insert(0, str(ROOT / "viz"))
    import test_studio
    import studio
    gen = test_studio.studio_srv.__wrapped__(tmp_path)
    base = next(gen)
    try:
        _, ms = test_studio.call(base, "/api/models")
        coder = next(m for m in ms if "coder" in m["id"])
        assert coder["family"] == "coder-7b" and coder["quant"] == "Q4_K_M"
        studio.STATE["hw"] = {"id": "hw1"}
        studio.STATE["quant_choices"] = str(tmp_path / "qc.json")
        (tmp_path / "qc.json").write_text(json.dumps({"coder-7b": {"hw1": {"quant": "Q4_K_M", "model": coder["path"]},
                                                                   "hw2": {"quant": "Q8_0"}}}))
        _, c = test_studio.call(base, "/api/quant/choice?family=coder-7b")
        assert c["here"]["quant"] == "Q4_K_M" and c["elsewhere"] == 1
        _, c = test_studio.call(base, "/api/quant/choice?family=nope")
        assert c["here"] is None
    finally:
        studio.STATE.pop("quant_choices", None)
        studio.STATE.pop("hw", None)
        try:
            next(gen)
        except StopIteration:
            pass
