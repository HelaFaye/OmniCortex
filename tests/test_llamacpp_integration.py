"""Real llama.cpp integration: compiled cett-dump against PyTorch hooks.

Skipped unless OC_LLAMA points at a llama.cpp checkout built with
scripts/build_llama_tools.sh (needs sentencepiece). Builds a tiny random Llama,
converts it to GGUF with llama.cpp's own converter, and checks that
cett-dump's CETT matches oc_common.CETTManager on every layer, and that
hscore scores a reply end to end."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
LLAMA = Path(os.environ.get("OC_LLAMA", "")).expanduser()
CETT = LLAMA / "build" / "bin" / "llama-cett-dump"
pytestmark = pytest.mark.skipif(not CETT.exists(), reason="set OC_LLAMA to a llama.cpp build with cett-dump")


@pytest.fixture(scope="module")
def tiny_gguf(tmp_path_factory):
    spm = pytest.importorskip("sentencepiece")
    torch = pytest.importorskip("torch")
    from transformers import LlamaConfig, LlamaForCausalLM
    d = tmp_path_factory.mktemp("tiny")
    words = "the cat dog sat on mat who wrote hamlet shakespeare paris capital france yes no know".split()
    rng = np.random.default_rng(0)
    (d / "corpus.txt").write_text("\n".join(" ".join(rng.choice(words, 10)) for _ in range(2000)))
    spm.SentencePieceTrainer.train(input=str(d / "corpus.txt"), model_prefix=str(d / "tok"), vocab_size=300,
                                   model_type="bpe", bos_id=1, eos_id=2, unk_id=0, pad_id=-1, byte_fallback=True)
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      bos_token_id=1, eos_token_id=2, tie_word_embeddings=False)
    hf = d / "hf"
    LlamaForCausalLM(cfg).save_pretrained(hf, safe_serialization=True)
    (hf / "tokenizer.model").write_bytes((d / "tok.model").read_bytes())
    (hf / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "LlamaTokenizer", "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>",
        "chat_template": "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}"
                         "{% if add_generation_prompt %}<|assistant|>{% endif %}"}))
    out = d / "tiny-f32.gguf"
    subprocess.run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(hf), "--outfile", str(out),
                    "--outtype", "f32"], check=True, capture_output=True)
    return hf, out


def test_cett_dump_matches_pytorch(tiny_gguf, tmp_path):
    import torch
    from transformers import LlamaForCausalLM
    import oc_common
    from extract_activations_gguf import read_aggregate, read_tokens, weight_col_norms
    hf, gguf = tiny_gguf
    text = "the cat sat on the mat and the dog wrote hamlet"
    (tmp_path / "m.jsonl").write_text(json.dumps({"id": "x", "text": text}) + "\n")
    subprocess.run([str(CETT), "-m", str(gguf), "--tokenize-only", "-ngl", "0", "-c", "512",
                    "--manifest", str(tmp_path / "m.jsonl"), "--outdir", str(tmp_path)], check=True, capture_output=True)
    ids = read_tokens(tmp_path / "x.toks")
    (tmp_path / "s.jsonl").write_text(json.dumps({"id": "x", "text": text, "spans": [[0, -1], [2, 6]]}) + "\n")
    r = subprocess.run([str(CETT), "-m", str(gguf), "-ngl", "0", "-b", "512", "-c", "512", "--n-layers", "4",
                        "--manifest", str(tmp_path / "s.jsonl"), "--outdir", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0 and "left unseen" not in r.stderr, r.stderr[-800:]
    _, agg, seen, _, _ = read_aggregate(tmp_path / "x.bin")
    assert seen.all(), "every layer, including the last, must be captured"
    wn = np.stack([weight_col_norms(str(gguf), 4)[l] for l in range(4)])
    m = LlamaForCausalLM.from_pretrained(hf, dtype=torch.float32).eval()
    mgr = oc_common.CETTManager(m)
    with torch.no_grad():
        m(torch.tensor([ids.tolist()]))
    ref = mgr.cett().numpy()
    for si, (a, b) in enumerate([(0, len(ids)), (2, 6)]):
        want = ref[:, a:b, :].mean(1)
        got = agg[si] * wn
        assert np.abs(got - want).max() / np.abs(want).max() < 5e-3



def test_by_weight_matches_named_nodes(tiny_gguf, tmp_path):
    """--by-weight (finding the down projection by its blk.N.ffn_down.weight, for looped models whose graph does
    not name it) records exactly what name matching does, across several ubatches; room for a second pass stays
    unseen on a model that runs each block once."""
    from extract_activations_gguf import read_aggregate
    _, gguf = tiny_gguf
    text = " ".join(["the cat sat on the mat and the dog wrote hamlet"] * 6)
    (tmp_path / "m.jsonl").write_text(json.dumps({"id": "x", "text": text, "spans": [[0, 10], [5, 50], [30, -1]]}) + "\n")

    def run(name, *extra):
        (tmp_path / name).mkdir()
        r = subprocess.run([str(CETT), "-m", str(gguf), "-ngl", "0", "-c", "512", "-b", "512", "-ub", "16",
                            "--manifest", str(tmp_path / "m.jsonl"), "--outdir", str(tmp_path / name), *extra],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]
        return read_aggregate(tmp_path / name / "x.bin")
    ids, a, sa, _, _ = run("named", "--n-layers", "4")
    assert len(ids) > 32                                     # several 16-token ubatches
    _, b, sb, _, _ = run("weight", "--n-layers", "4", "--by-weight", "4")
    assert np.array_equal(sa, sb) and np.array_equal(a, b)
    _, c, sc, _, _ = run("weight2", "--n-layers", "8", "--by-weight", "4")
    assert np.array_equal(c[:, :4], a) and not sc[:, 4:].any()

def test_hscore_end_to_end(tiny_gguf, tmp_path):
    import hscore
    _, gguf = tiny_gguf
    coef = np.random.default_rng(1).normal(size=4 * 128).astype(np.float32)
    np.savez(tmp_path / "clf.npz", coef=coef, intercept=0.0, n_layers=4, n_neurons=128)
    sc = hscore.HScorer(str(CETT), str(gguf), str(tmp_path / "clf.npz"), ngl=0, batch=512)
    res = sc.score([{"role": "user", "content": "who wrote hamlet"}], "shakespeare wrote hamlet")
    assert res["n_tokens"] > 0 and np.isfinite(res["score"]) and 0 < res["prob"] < 1


def test_extraction_pipeline_matches_pytorch(tiny_gguf, tmp_path):
    """extract_activations_gguf.py end to end (GGUF tokenizer, real cett-dump):
    the response-region features equal a PyTorch CETT mean over the same tokens."""
    import torch
    from transformers import LlamaForCausalLM
    import gguf_tokenizer
    import oc_common
    from extract_activations_gguf import read_tokens
    hf, gguf = tiny_gguf
    samples = {"q1": ("who wrote hamlet", "shakespeare wrote hamlet"),
               "q2": ("what is the capital of france", "paris is the capital"),
               "q3": ("the cat sat on", "the mat")}
    with open(tmp_path / "ans.jsonl", "w") as f:
        for q, (question, response) in samples.items():
            f.write(json.dumps({q: {"question": question, "response": response, "answer_tokens": []}}) + "\n")
    (tmp_path / "ids.json").write_text(json.dumps({"t": ["q1", "q3"], "f": ["q2"]}))
    out = tmp_path / "acts"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "extract_activations_gguf.py"), "--binary", str(CETT),
                        "--gguf", str(gguf), "--input_path", str(tmp_path / "ans.jsonl"), "--ids_path",
                        str(tmp_path / "ids.json"), "--output_root", str(out), "--locations", "output",
                        "--ngl", "0", "--batch", "512"], capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    m = LlamaForCausalLM.from_pretrained(hf, dtype=torch.float32).eval()
    mgr = oc_common.CETTManager(m)
    gt = gguf_tokenizer.load(str(gguf))
    for q, (question, _) in samples.items():
        got = np.load(out / "output" / f"act_{q}.npy").astype(np.float32)
        ids = read_tokens(next((out).rglob(f"{q}.toks")))
        pids = read_tokens(next((out).rglob(f"{q}__prompt.toks")))
        start = 0
        while start < len(pids) and pids[start] == ids[start]:
            start += 1
        with torch.no_grad():
            m(torch.tensor([ids.tolist()]))
        want = mgr.cett().numpy()[:, start:, :].mean(1)
        assert got.shape == want.shape
        assert np.abs(got - want).max() / np.abs(want).max() < 5e-3, q
        mgr.clear()


SERVER = LLAMA / "build" / "bin" / "llama-server"


DOCKER_IMAGE = os.environ.get("OC_DOCKER_IMAGE", "")
# For a CUDA image on a machine without an NVIDIA driver: a libcuda.so.1 (the
# toolkit's stub) to mount, so the binaries load and llama.cpp falls back to CPU.
DOCKER_LIBCUDA = os.environ.get("OC_DOCKER_LIBCUDA", "")


def _launch(where, gguf, clf, port, env_extra):
    """Start a patched llama-server on the host build, or inside OC_DOCKER_IMAGE."""
    args = ["-m", "MODEL", "--port", str(port), "--parallel", "1", "-c", "256", "-ngl", "0"]
    if where == "host":
        args[1] = str(gguf)
        return subprocess.Popen([str(SERVER), *args], env=dict(os.environ, **env_extra, OC_CLASSIFIER=str(clf)),
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    args[1] = "/m/" + gguf.name
    cmd = ["docker", "run", "--rm", "--network", "host", "-v", f"{gguf.parent}:/m:ro", "-v", f"{clf.parent}:/c:ro",
           *[x for k, v in env_extra.items() for x in ("-e", f"{k}={v}")], "-e", f"OC_CLASSIFIER=/c/{clf.name}"]
    if DOCKER_LIBCUDA:
        cmd += ["-v", f"{DOCKER_LIBCUDA}:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro"]
    return subprocess.Popen([*cmd, DOCKER_IMAGE, "llama-server", *args],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


@pytest.mark.parametrize("where", ["host", "docker"])
def test_server_activations_match_pytorch(tiny_gguf, tmp_path, where):
    """The patched llama-server streams one frame per decoded token whose raw
    values are |a|/||out|| for every layer, and whose score, with the column
    norms folded in by export_classifier_bin.py --gguf, equals the classifier
    applied to full CETT in PyTorch."""
    import socket
    import threading
    import time
    import urllib.request
    import torch
    from transformers import LlamaForCausalLM
    import oc_common
    from extract_activations_gguf import weight_col_norms
    if where == "host" and not (LLAMA / "tools" / "server" / "oc_server_glue.h").exists():
        pytest.skip("llama-server not patched with llama-tools/server-activations/apply_patch.py")
    if where == "docker" and not DOCKER_IMAGE:
        pytest.skip("set OC_DOCKER_IMAGE (e.g. omnicortex-llama:cuda) to test the container's server")
    hf, gguf = tiny_gguf
    coef = np.random.default_rng(2).normal(size=4 * 128).astype(np.float32)
    np.savez(tmp_path / "clf.npz", coef=coef, intercept=0.25, n_layers=4, n_neurons=128)
    subprocess.run([sys.executable, str(ROOT / "llama-tools" / "server-activations" / "export_classifier_bin.py"),
                    str(tmp_path / "clf.npz"), str(tmp_path / "clf.bin"), "--gguf", str(gguf)], check=True)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = _launch(where, gguf, tmp_path / "clf.bin", port, {"OC_ACTIVATIONS": "raw"})
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(base + "/health", timeout=1).read()
                break
            except Exception:
                time.sleep(0.1)
        frames = []

        def listen():
            with urllib.request.urlopen(base + "/activations", timeout=30) as r:
                for line in r:
                    if line.startswith(b"data: "):
                        frames.append(json.loads(line[6:]))
                        if len(frames) == 5:
                            return
        t = threading.Thread(target=listen, daemon=True)
        t.start()
        time.sleep(0.5)
        prompt = [1, 20, 31, 42, 53, 64]
        body = json.dumps({"prompt": prompt, "n_predict": 6, "temperature": 0, "ignore_eos": True,
                           "return_tokens": True}).encode()
        out = json.loads(urllib.request.urlopen(urllib.request.Request(
            base + "/completion", body, {"Content-Type": "application/json"}), timeout=30).read())
        t.join(10)
    finally:
        proc.terminate()
        err = proc.communicate(timeout=10)[1]
    gen = out["tokens"]
    assert len(gen) == 6 and len(frames) == 5, err[-1500:]
    assert all(f["scored"] and f["t"] == "raw" and f["l"] == 4 for f in frames)
    # frame k is the decode of gen[k], at position len(prompt) + k
    m = LlamaForCausalLM.from_pretrained(hf, dtype=torch.float32).eval()
    mgr = oc_common.CETTManager(m)
    with torch.no_grad():
        m(torch.tensor([prompt + gen]))
    cett = mgr.cett().numpy()                                   # [layers, tokens, n_ff]
    wn = np.stack([weight_col_norms(str(gguf), 4)[l] for l in range(4)])
    for k, f in enumerate(frames):
        want = cett[:, len(prompt) + k, :]
        got = np.asarray(f["v"], dtype=np.float32)
        assert np.abs(got * wn - want).max() / np.abs(want).max() < 5e-3, k
        assert f["s"] == pytest.approx(float(want.ravel() @ coef + 0.25), rel=5e-3, abs=5e-3)


def test_docker_cett_dump_matches_host(tiny_gguf, tmp_path):
    """The container's cett-dump writes the same CETT as the host build."""
    if not DOCKER_IMAGE:
        pytest.skip("set OC_DOCKER_IMAGE to test the container's cett-dump")
    from extract_activations_gguf import read_aggregate
    _, gguf = tiny_gguf
    text = "the cat sat on the mat and the dog wrote hamlet"
    (tmp_path / "s.jsonl").write_text(json.dumps({"id": "x", "text": text, "spans": [[0, -1]]}) + "\n")
    common = ["-ngl", "0", "-b", "512", "-c", "512", "--n-layers", "4"]
    (tmp_path / "host").mkdir()
    (tmp_path / "box").mkdir()
    subprocess.run([str(CETT), "-m", str(gguf), *common, "--manifest", str(tmp_path / "s.jsonl"),
                    "--outdir", str(tmp_path / "host")], check=True, capture_output=True)
    cmd = ["docker", "run", "--rm", "-v", f"{gguf.parent}:/m:ro", "-v", f"{tmp_path}:/w"]
    if DOCKER_LIBCUDA:
        cmd += ["-v", f"{DOCKER_LIBCUDA}:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro"]
    r = subprocess.run([*cmd, DOCKER_IMAGE, "llama-cett-dump", "-m", "/m/" + gguf.name, *common,
                        "--manifest", "/w/s.jsonl", "--outdir", "/w/box"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    _, a, seen_a, _, _ = read_aggregate(tmp_path / "host" / "x.bin")
    _, b, seen_b, _, _ = read_aggregate(tmp_path / "box" / "x.bin")
    assert seen_b.all() and np.abs(a - b).max() / np.abs(a).max() < 1e-4
