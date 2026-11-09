"""Engine options: one table for the form, the flags and the optimizer;
spellings checked against the installed engine."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "viz"))
import engine_options as eo  # noqa: E402

LLAMA = Path(os.environ.get("OC_LLAMA") or "~/llama.cpp").expanduser() / "build" / "bin" / "llama-server"
NEW = {"-fa", "-ctk", "-ctv", "--spec-type", "--spec-draft-n-max", "--spec-draft-n-min", "-md", "--keep",
       "--context-shift", "--cache-reuse", "-Cr", "-lm", "--rpc", "-dev"}
OLD = {"-fa", "-ctk", "-ctv", "--draft-max", "--draft-min", "-md", "--mlock", "--no-mmap"}


def test_spellings_follow_the_build():
    s = {"draft_max": 8, "draft_min": 2}
    assert eo.build_args(s, "llama", NEW)[0] == ["--spec-draft-n-max", "8", "--spec-draft-n-min", "2"]
    assert eo.build_args(s, "llama", OLD)[0] == ["--draft-max", "8", "--draft-min", "2"]
    args, notes = eo.build_args({"cache_reuse": 256, "mlock": True}, "llama", OLD)
    assert args == ["--mlock"] and "Prompt cache reuse" in notes[0]


def test_values_are_validated_and_defaults_not_passed():
    assert eo.build_args({"flash_attn": "auto", "context_shift": False}, "llama", NEW)[0] == []
    assert eo.build_args({"flash_attn": "on", "cache_type_k": "q8_0", "cache_type_v": "q4_0"}, "llama", NEW)[0] == \
        ["-fa", "on", "-ctk", "q8_0", "-ctv", "q4_0"]
    for bad in ({"cache_type_k": "q3"}, {"cpu_range": "0-7; rm -rf /"}, {"device": "a b"},
                {"rpc": "host"}, {"speculative_config": "{nope"}):
        with pytest.raises(ValueError):
            eo.build_args(bad, "llama" if "speculative_config" not in bad else "vllm", NEW)
    assert eo.build_args({"rpc": "10.0.0.2:50052,box:50052"}, "llama", NEW)[0] == ["--rpc", "10.0.0.2:50052,box:50052"]


def test_env_and_wrappers_are_closed_lists():
    assert eo.build_env({"env": {"HSA_ENABLE_SDMA": "0"}}) == {"HSA_ENABLE_SDMA": "0"}
    for bad in ({"env": {"LD_PRELOAD": "/x.so"}}, {"env": {"HSA_ENABLE_SDMA": "1; rm"}}):
        with pytest.raises(ValueError):
            eo.build_env(bad)
    assert eo.wrapper({"wrap": "taskset", "taskset_cores": "0,2,4,6"}) == ["taskset", "-c", "0,2,4,6"]
    assert eo.wrapper({"wrap": "numactl", "numa_node": "1"}) == ["numactl", "--cpunodebind=1", "--membind=1"]
    with pytest.raises(ValueError):
        eo.wrapper({"wrap": "taskset", "taskset_cores": "$(id)"})
    with pytest.raises(ValueError):
        eo.wrapper({"wrap": "bash"})


def test_vllm_options():
    args, _ = eo.build_args({"prefix_caching": True, "chunked_prefill": True, "kv_cache_dtype": "fp8",
                             "pipeline_parallel": 2, "speculative_config": '{"method": "ngram"}',
                             "cache_type_k": "q8_0"}, "vllm", None)
    assert args == ["--kv-cache-dtype", "fp8", "--enable-prefix-caching", "--enable-chunked-prefill",
                    "--speculative-config", '{"method": "ngram"}', "--pipeline-parallel-size", "2"]
    sc = eo.schema("vllm")
    assert "cache_type_k" not in {o["key"] for g in sc["groups"].values() for o in g}


@pytest.mark.skipif(not LLAMA.exists(), reason="needs llama-server")
def test_probe_real_llama_server():
    sup = eo.probe(str(LLAMA))
    assert "-fa" in sup and "--port" in sup
    sc = eo.schema("llama", sup)
    fa = next(o for o in sc["groups"]["Speed"] if o["key"] == "flash_attn")
    assert fa["available"] and sc["probed"]


def test_studio_server_cmd(tmp_path, monkeypatch):
    import studio
    monkeypatch.setitem(studio.STATE, "server_bin", str(LLAMA) if LLAMA.exists() else "/nonexistent/llama-server")
    model = {"path": str(tmp_path / "m.gguf"), "name": "m.gguf", "id": "x/m"}
    s = {**studio.DEFAULTS, "flash_attn": True, "cache_type": "q8_0", "spec_type": "ngram-mod", "draft_max": 6,
         "env": {"GGML_VK_VISIBLE_DEVICES": "0,1"}, "wrap": "taskset", "taskset_cores": "0-7"}
    cmd, env = studio.server_cmd(model, s, 8099)
    assert cmd[:3] == ["taskset", "-c", "0-7"]
    j = " ".join(cmd)
    assert "-fa on" in j and "-ctk q8_0 -ctv q8_0" in j and "--spec-type ngram-mod" in j
    assert ("--spec-draft-n-max 6" in j) or ("--draft-max 6" in j)
    assert env["GGML_VK_VISIBLE_DEVICES"] == "0,1"
    # no drafting: the draft limits from DEFAULTS are not passed
    cmd, _ = studio.server_cmd(model, {**studio.DEFAULTS}, 8099)
    assert not any("draft" in c for c in cmd) and "-fa" not in cmd
    hf = {"path": str(tmp_path), "name": "ck", "id": "x/ck", "format": "hf"}
    with pytest.raises(ValueError, match="vLLM"):
        studio.server_cmd(hf, {**studio.DEFAULTS}, 8099)
    monkeypatch.setitem(studio.STATE, "vllm_bin", "definitely-not-vllm")
    with pytest.raises(ValueError, match="not installed"):
        studio.server_cmd(hf, {**studio.DEFAULTS, "engine": "vllm"}, 8099)
    monkeypatch.setitem(studio.STATE, "vllm_bin", sys.executable + " -m fake_vllm")
    cmd, _ = studio.server_cmd(hf, {**studio.DEFAULTS, "engine": "vllm", "prefix_caching": True}, 8099)
    assert cmd[2:5] == ["fake_vllm", "serve", str(tmp_path)] and "--enable-prefix-caching" in cmd
    assert cmd[cmd.index("--max-model-len") + 1] == "8192"


def test_studio_api_and_load(tmp_path):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_studio import call, studio_srv  # noqa: F401
    import studio
    gen = studio_srv.__wrapped__(tmp_path) if hasattr(studio_srv, "__wrapped__") else None
    if gen is None:
        pytest.skip("fixture not callable directly")
    base = next(gen)
    try:
        ck = Path(studio.STATE["models_dirs"][0]) / "acme" / "Tiny-HF"
        ck.mkdir(parents=True)
        (ck / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (ck / "model.safetensors").write_bytes(b"\0" * 16)
        st, sc = call(base, "/api/engine/options?engine=llama")
        assert st == 200 and "Speculative decoding" in sc["groups"] and "HSA_ENABLE_SDMA" in sc["env"]
        st, sc = call(base, "/api/engine/options?engine=vllm")
        assert st == 200 and "vLLM" in sc["groups"]
        _, ms = call(base, "/api/models")
        hf = next(m for m in ms if m.get("format") == "hf")
        assert hf["id"] == "acme/tiny-hf" and hf["arch"] == "llama"
        gguf = next(m for m in ms if m.get("format") != "hf")
        st, r = call(base, "/api/load", {"path": gguf["path"], "settings": {"flash_attn": "on", "cache_reuse": 256,
                                                                           "keep": -1, "context_shift": True}})
        assert st == 200 and r["ok"], r
        # the test server's --help lists none of these: left out, and said so
        assert "-fa" not in r["args"] and any("Flash attention" in n for n in r["notes"])
        assert any("Prompt cache reuse" in n for n in r["notes"])
        st, r = call(base, "/api/load", {"path": hf["path"], "settings": {}})
        assert st == 500 and "vLLM" in r["message"]
    finally:
        next(gen, None)
