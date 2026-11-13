"""Integrated options, batch 3: backend hot-swap, layer streaming, semantic anchors, morphing, expert shards,
feature packs."""
import io
import json
import os
import socket
import stat
import sys
import urllib.request
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import backend_swap  # noqa: E402
import expert_shards  # noqa: E402
import feature_pack  # noqa: E402
import layer_stream  # noqa: E402
import morph_router  # noqa: E402
import semantic_compact  # noqa: E402


def test_hot_swap_ladder_and_classification():
    assert backend_swap.is_resource_failure("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9000 MiB: cudaMalloc failed: out of memory")
    assert backend_swap.is_resource_failure("vk::Device::allocateMemory: ErrorOutOfDeviceMemory")
    assert not backend_swap.is_resource_failure("error: unknown argument: --bogus")
    assert not backend_swap.is_resource_failure("failed to open model.gguf: no such file")
    lad = backend_swap.ladder({"ngl": 99, "split_mode": "layer"}, ["/opt/llama-vulkan/llama-server"], "/opt/cuda/llama-server")
    whys = [w for _, _, w in lad]
    assert whys[0] == "same backend, 49 layers on the GPU" and "vulkan" in whys[2] and whys[-1] == "CPU only"
    assert lad[-1][0]["ngl"] == 0 and lad[-1][0]["split_mode"] == ""
    assert [s.get("ngl") for s, _, _ in backend_swap.ladder({"ngl": 0})] == [0]


def test_studio_falls_back_when_the_gpu_runs_out(tmp_path):
    import test_studio
    studio = test_studio.studio
    wrapper = tmp_path / "oom-server.py"
    wrapper.write_text(f"""#!{sys.executable}
import os, sys
ngl = int(sys.argv[sys.argv.index("-ngl") + 1]) if "-ngl" in sys.argv else 0
if ngl > 10:
    sys.stderr.write("ggml_cuda: cudaMalloc failed: out of memory\\n"); sys.exit(1)
os.execv(sys.executable, [sys.executable, {str(ROOT / 'tests' / 'fake_llama_server.py')!r}] + sys.argv[1:])
""")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    gen = test_studio.studio_srv.__wrapped__(tmp_path)
    base = next(gen)
    try:
        studio.STATE["server_bin"] = str(wrapper)
        studio.save_settings({studio._key(studio.find_model("coder-gguf/coder-7b-q4_k_m")["path"]): {"ngl": 40}}) \
            if hasattr(studio, "save_settings") else None
        code, r = test_studio.call(base, "/v1/chat/completions", {"model": "coder-gguf/coder-7b-q4_k_m",
                                                                   "messages": [{"role": "user", "content": "x"}]})
        assert code == 200 and "model=" in r["choices"][0]["message"]["content"]
        fb = studio.PROC["fallback"]
        assert fb and ("layers on the GPU" in fb["setup"] or fb["setup"] == "CPU only") and "out of memory" in fb["after"][0]
        assert int(studio.PROC["args"][studio.PROC["args"].index("-ngl") + 1]) <= 10
        studio.stop_server()
        code, o = test_studio.call(base, "/api/settings/options", {"name": "hot_swap", "values": {"enabled": False}})
        assert code == 200
        code, r = test_studio.call(base, "/v1/chat/completions", {"model": "coder-gguf/coder-7b-q4_k_m",
                                                                   "messages": [{"role": "user", "content": "x"}]})
        assert code != 200                                                          # off: the failure stands
    finally:
        try:
            next(gen)
        except StopIteration:
            pass


def test_rpc_setting_and_layer_stream(tmp_path):
    import test_studio
    studio = test_studio.studio
    m = {"id": "x", "name": "x", "path": "/m.gguf"}
    cmd, _ = studio.server_cmd(m, {**studio.DEFAULTS, "rpc": "box2:50052,10.0.0.7:50052"}, 9999)
    assert cmd[cmd.index("--rpc") + 1] == "box2:50052,10.0.0.7:50052"
    with pytest.raises(ValueError):
        studio.server_cmd(m, {**studio.DEFAULTS, "rpc": "box2;rm -rf /"}, 9999)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    port = srv.getsockname()[1]
    try:
        res = layer_stream.check(f"127.0.0.1:{port},127.0.0.1:1")
    finally:
        srv.close()
    assert res[0]["ok"] and not res[1]["ok"]
    assert layer_stream.split([24, 24, 11.4]) == "24,24,11"
    assert layer_stream.serve_cmd("rpc-server", "10.0.0.2", 50052, True, None, "CUDA0") == \
        ["rpc-server", "-H", "10.0.0.2", "-p", "50052", "-c", "-d", "CUDA0"]
    with pytest.raises(ValueError):
        layer_stream.parse("nohost")


def test_semantic_anchors_group_by_topic():
    msgs = [{"role": "user", "content": "The dragon Vex guards the northern pass of the kingdom."},
            {"role": "user", "content": "Write the parser in Rust: tokenize, then build the AST with a Pratt parser."},
            {"role": "assistant", "content": "Vex the dragon sleeps at the northern pass; travelers fear the dragon."},
            {"role": "user", "content": "The Rust parser should report AST errors with spans from the tokenizer."},
            {"role": "user", "content": "Remember the dragon Vex hates the kingdom's bells."}]
    groups = semantic_compact.cluster(msgs)
    dragon = next(g for g in groups if 0 in g)
    rust = next(g for g in groups if 1 in g)
    assert set(dragon) == {0, 2, 4} and set(rust) == {1, 3}
    p = semantic_compact.anchor_prompt(msgs, previous="earlier note")
    assert "TOPIC 1 (" in p[1]["content"] and "EARLIER NOTES" in p[1]["content"] and "message 5" in p[1]["content"]


def test_morph_routing():
    cfg = {"enabled": True, "full_model": "big", "small_model": "small", "threshold": 0.35}
    easy = [{"role": "user", "content": "Hi! What's a good name for a cat?"}]
    hard = [{"role": "user", "content": "Debug this and explain why it fails:\n```python\ndef f(x):\n  return x[0]\n```\n"
                                        "Traceback: IndexError. Then derive the complexity and compare two fixes."}]
    assert morph_router.route("big", easy, cfg)[0] == "small"
    m, why = morph_router.route("big", hard, cfg)
    assert m == "big" and why["morph"] == "full" and why["parts"]["code"] > 0
    assert morph_router.route("other", easy, cfg) == ("other", None)
    assert morph_router.route("big", easy, {**cfg, "enabled": False}) == ("big", None)


def test_expert_shards(tmp_path):
    calls = []
    mp = expert_shards.build(Path("/m/model-Q4.gguf"), {"code": "h-code.json", "math": "h-math.json"}, 0.3,
                             tmp_path, run=lambda c: calls.append(c) or 0)
    assert mp == {"code": str(tmp_path / "model-Q4.code.gguf"), "math": str(tmp_path / "model-Q4.math.gguf")}
    assert calls[0][calls[0].index("--h_neurons") + 1] == "h-code.json" and "0.3" in calls[0]
    assert json.loads((tmp_path / "shards.json").read_text())["subjects"] == mp
    with pytest.raises(RuntimeError):
        expert_shards.build(Path("/m/x.gguf"), {"code": "h.json"}, 0.3, tmp_path, run=lambda c: 1)


def test_feature_packs(tmp_path):
    h = tmp_path / "h_neurons.json"
    h.write_text(json.dumps({"n_layers": 4, "by_layer": {"1": [3, 7]}}))
    import numpy as np
    np.save(tmp_path / "calm.npy", np.ones(8, dtype=np.float32))
    data = feature_pack.pack([str(h), str(tmp_path / "calm.npy")],
                             {"id": "tiny-pack", "kind": "mixed", "base": {"hf_repo": "org/tiny", "n_layers": 4, "d_model": 8}})
    man, files = feature_pack.read(data)
    assert man["base"]["hf_repo"] == "org/tiny" and set(files) == {"h_neurons.json", "calm.npy"}
    with pytest.raises(feature_pack.PackError, match="layers"):
        feature_pack.install(data, 32, 4096, root=tmp_path / "inst")
    ok = feature_pack.install(data, 4, 8, "org/other", root=tmp_path / "inst")
    assert "fine-tune" in ok["note"] and (tmp_path / "inst" / "tiny-pack" / "calm.npy").exists()
    assert feature_pack.installed(tmp_path / "inst")[0]["id"] == "tiny-pack"
    bad = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as zi, zipfile.ZipFile(bad, "w") as zo:
        for n in zi.namelist():
            zo.writestr(n, b"tampered" if n == "calm.npy" else zi.read(n))
    with pytest.raises(feature_pack.PackError, match="hash"):
        feature_pack.read(bad.getvalue())
    with pytest.raises(feature_pack.PackError):
        feature_pack.pack([str(h)], {"id": "Bad Id", "kind": "mixed", "base": {"n_layers": 4, "d_model": 8}})
    pytest.importorskip("cryptography")
    import github_board as gb
    board = Path(gb.init(tmp_path / "board", "someone/omnicortex")["board"])
    _, body = gb.plain_issue(gb.MARK_FEATURE, "t", feature_pack.board_doc(data), "x")
    assert gb.accept(board, 9, body, [], "")["status"] == "verified"
    bj = json.loads((board / "board.json").read_text())
    assert bj["feature_packs"]["pending"][0]["id"] == "tiny-pack"
    assert gb.MARK_FEATURE in (board / ".github" / "workflows" / "accept.yml").read_text()
