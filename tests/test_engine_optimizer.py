"""The engine optimizer on a tiny GGUF with the real llama-server."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import engine_optimizer as opt  # noqa: E402

LLAMA = Path(os.environ.get("OC_LLAMA") or "~/llama.cpp").expanduser()


def test_space_rules():
    assert not opt.valid({"batch": 256, "ubatch": 512})
    assert not opt.valid({"flash_attn": "off", "cache_type_v": "q8_0"})
    assert opt.valid({"flash_attn": "on", "cache_type_v": "q8_0", "batch": 512, "ubatch": 256})
    a = type("A", (), {"kv": True, "spec": False, "placement": False, "fix": ["device=Vulkan0", "ubatch=256"]})()
    sp = opt.space(a, {"-fa", "-ub", "-ctk", "-ctv", "-dev"})
    assert sp["device"] == ["Vulkan0"] and sp["ubatch"] == [256] and "cache_type_k" in sp
    assert "spec_type" not in opt.space(type("A", (), {"kv": False, "spec": True, "placement": False, "fix": None})(),
                                        {"-fa"})              # not in that build: not searched


def test_optimizer_end_to_end(tmp_path):
    pytest.importorskip("sentencepiece")
    if not (LLAMA / "build" / "bin" / "llama-server").exists():
        pytest.skip("needs llama-server")
    import tiny_models
    hf = tiny_models.make_tiny_llama(tmp_path / "tiny", layers=2)
    gguf = tmp_path / "tiny.gguf"
    r = subprocess.run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(hf), "--outfile", str(gguf),
                        "--outtype", "f16"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    out = tmp_path / "opt"
    studio_settings = tmp_path / "studio.json"
    args = ["--model", str(gguf), "--server", str(LLAMA / "build" / "bin" / "llama-server"), "--out", str(out),
            "--trials", "4", "--prompt-tokens", "64", "--gen-tokens", "16", "--repeats", "1", "--ctx", "512",
            "--ngl", "0", "--apply", str(studio_settings)]
    assert opt.main(args) == 0
    best = json.loads((out / "best.json").read_text())
    assert best["best"]["result"]["gen_tps"] > 0 and best["baseline"] is not None and best["trials"] == 4
    assert len((out / "trials.jsonl").read_text().splitlines()) == 4
    saved = json.loads(studio_settings.read_text())
    import hashlib
    k = hashlib.sha256(os.path.abspath(str(gguf)).encode()).hexdigest()[:16]
    assert saved[k] == {key: v for key, v in best["best"]["settings"].items() if not (key == "spec_type" and v == "none")}
    # resume: cached trials are free, two more are new
    assert opt.main(args[:-2] + ["--trials", "2"]) == 0
    assert len((out / "trials.jsonl").read_text().splitlines()) == 6
